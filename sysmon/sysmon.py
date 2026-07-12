"""Host system monitor: CPU, RAM, all GPUs (rocm-smi), all disks,
and this app's own data usage. Serves /api/system for the web UI.

/proc/stat and /proc/meminfo are not namespaced, so psutil inside the
container reports HOST cpu/ram. Disks come from the host root mounted
read-only at /host; GPUs via the GPU runtime (all devices, read-only
queries only).
"""

import collections
import json
import logging
import os
import subprocess
import threading
import time

import psutil
import uvicorn
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("sysmon")

HOST = "/host"
APP_DIR = os.environ.get("APP_DIR", "")          # app path, /host-prefixed
SAMPLE_S = 5
HIST_N = 180                                     # 15 min of history

app = FastAPI(title="ccvt sysmon")

history = {
    "t": collections.deque(maxlen=HIST_N),
    "cpu": collections.deque(maxlen=HIST_N),
    "ram": collections.deque(maxlen=HIST_N),
}
gpu_hist: dict[str, dict] = {}                   # index -> {"util": dq, "vram": dq}
gpu_now: list[dict] = []
app_usage = {"dirs": {}, "total": 0}


def _read(path: str) -> str:
    try:
        return open(path).read()
    except OSError:
        return ""


def collect_specs() -> dict:
    """Static machine identity — gathered once at startup."""
    os_name = ""
    for line in _read(f"{HOST}/etc/os-release").splitlines():
        if line.startswith("PRETTY_NAME="):
            os_name = line.split("=", 1)[1].strip('"')
            break

    cpu_model = ""
    for line in _read("/proc/cpuinfo").splitlines():   # cpuinfo is host-wide
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break

    uname = os.uname()
    driver = ""
    try:
        # amdgpu kernel driver version, from the host sysfs the container mounts
        driver = _read(f"{HOST}/sys/module/amdgpu/version").strip()
    except OSError:
        pass

    return {
        "hostname": _read(f"{HOST}/etc/hostname").strip() or uname.nodename,
        "os": os_name,
        "kernel": uname.release,
        "arch": uname.machine,
        "cpu_model": cpu_model,
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_total": psutil.virtual_memory().total,
        "gpu_driver": driver,
        "gpus": [{"index": g["index"], "name": g["name"], "vram_total": g["vram_total"]}
                 for g in query_gpus()],
    }


def host_uptime() -> float:
    try:
        return float(_read("/proc/uptime").split()[0])   # host-wide
    except (ValueError, IndexError):
        return 0.0


def query_gpus() -> list[dict]:
    """AMD MI300X via rocm-smi JSON. The web UI
    expects MB for VRAM, so the byte counts rocm-smi reports are scaled here."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname", "--showuse",
             "--showmeminfo", "vram", "--showtemp", "--showpower", "--json"],
            capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    try:
        data = json.loads(out)
    except (ValueError, TypeError):
        return []

    def pick(d, *keys, default="0"):
        for k in keys:
            for full, v in d.items():
                if full.startswith(k):
                    return v
        return default

    gpus = []
    for card, d in sorted(data.items()):
        if not card.startswith("card"):
            continue
        try:
            idx = int(card[4:])
            used_b = float(pick(d, "VRAM Total Used Memory"))
            total_b = float(pick(d, "VRAM Total Memory"))
            gpus.append({
                "index": idx,
                "name": pick(d, "Card Series", "Device Name", default="AMD GPU"),
                "util": float(pick(d, "GPU use (%)")),
                "vram_used": used_b / 1024**2,      # bytes -> MB (UI expects MB)
                "vram_total": total_b / 1024**2,
                "temp": float(pick(d, "Temperature (Sensor junction)",
                                   "Temperature (Sensor edge)")),
                "power": float(pick(d, "Current Socket Graphics Package Power",
                                   "Average Graphics Package Power")),
            })
        except (ValueError, TypeError):
            continue
    return gpus


HWMON = "/sys/class/hwmon"


def host_temps() -> list[dict]:
    """Per-device temperatures from /sys/class/hwmon (host-shared sysfs):
    CPU package, each NVMe SSD, and the NIC. GPUs report their own temp
    via rocm-smi (see query_gpus), so they are added by /api/system."""
    cpu = None
    disks: list[dict] = []
    others: list[dict] = []
    nvme_i = 0
    try:
        chips = sorted(os.listdir(HWMON))
    except OSError:
        return []
    for h in chips:
        d = os.path.join(HWMON, h)
        name = _read(os.path.join(d, "name")).strip()
        try:
            files = os.listdir(d)
        except OSError:
            continue
        temps = []                          # (label, current, crit)
        for f in files:
            if f.startswith("temp") and f.endswith("_input"):
                raw = _read(os.path.join(d, f)).strip()
                try:
                    cur = int(raw) / 1000.0
                except ValueError:
                    continue
                idx = f[4:-6]
                label = _read(os.path.join(d, f"temp{idx}_label")).strip()
                crit = (_read(os.path.join(d, f"temp{idx}_crit")).strip()
                        or _read(os.path.join(d, f"temp{idx}_max")).strip())
                crit = int(crit) / 1000.0 if crit.isdigit() else None
                temps.append((label, cur, crit))
        if not temps:
            continue
        if name in ("coretemp", "k10temp", "zenpower"):
            # CPU: prefer the package / Tctl reading, else the hottest core
            pkg = next((t for t in temps
                        if "package" in (t[0] or "").lower()
                        or (t[0] or "").lower() in ("tctl", "tdie")), None)
            head = pkg or max(temps, key=lambda t: t[1])
            cpu = {"label": "CPU", "sub": name, "kind": "cpu",
                   "current": round(head[1], 1),
                   "critical": head[2] or 100.0}
        elif name == "nvme":
            nvme_i += 1
            comp = next((t for t in temps
                         if (t[0] or "").lower() == "composite"), temps[0])
            disks.append({"label": f"NVMe {nvme_i}", "sub": "SSD",
                          "kind": "disk", "current": round(comp[1], 1),
                          "critical": comp[2]})
        elif name.startswith("r816") or "eth" in name:
            others.append({"label": "Ethernet", "sub": name, "kind": "net",
                           "current": round(temps[0][1], 1),
                           "critical": temps[0][2]})
    out = []
    if cpu:
        out.append(cpu)
    out += disks
    out += others
    return out


def host_disks() -> list[dict]:
    """All real block-device filesystems of the host."""
    seen = {}
    try:
        # /proc/mounts is namespace-relative even via the bind mount —
        # host PID 1's table is the real host view.
        mounts = open(f"{HOST}/proc/1/mounts").read().splitlines()
    except OSError:
        mounts = []
    for line in mounts:
        parts = line.split()
        if len(parts) < 3:
            continue
        dev, mnt, fstype = parts[0], parts[1], parts[2]
        if not dev.startswith("/dev/") or fstype in ("squashfs", "overlay", "tmpfs"):
            continue
        # a device appears once per bind mount — keep its real (shortest) mountpoint
        if dev in seen and len(seen[dev]["mount"]) <= len(mnt):
            continue
        try:
            st = os.statvfs(HOST + mnt)
        except OSError:
            continue
        total = st.f_blocks * st.f_frsize
        free = st.f_bavail * st.f_frsize
        if total < 2 * 1024**3:             # ignore tiny boot/efi partitions
            continue
        seen[dev] = {"device": dev, "mount": mnt, "total": total,
                     "free": free, "used": total - free}
    return list(seen.values())


def dir_size(path: str) -> int:
    total = 0
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.is_file(follow_symlinks=False):
                        total += e.stat(follow_symlinks=False).st_size
                    elif e.is_dir(follow_symlinks=False):
                        total += dir_size(e.path)
                except OSError:
                    pass
    except OSError:
        pass
    return total


def usage_loop() -> None:
    """App data usage — heavier scan, refreshed every 60s."""
    while True:
        if APP_DIR:
            dirs = {}
            for name in ("recordings", "output", "ai/models"):
                dirs[name] = dir_size(os.path.join(APP_DIR, name))
            app_usage["dirs"] = dirs
            app_usage["total"] = sum(dirs.values())
        time.sleep(60)


def sample_loop() -> None:
    psutil.cpu_percent(None)          # prime
    while True:
        time.sleep(SAMPLE_S)
        now = int(time.time())
        history["t"].append(now)
        history["cpu"].append(round(psutil.cpu_percent(None), 1))
        history["ram"].append(round(psutil.virtual_memory().percent, 1))
        global gpu_now
        gpu_now = query_gpus()
        for g in gpu_now:
            h = gpu_hist.setdefault(str(g["index"]), {
                "util": collections.deque(maxlen=HIST_N),
                "vram": collections.deque(maxlen=HIST_N),
            })
            h["util"].append(g["util"])
            h["vram"].append(round(100 * g["vram_used"] / max(1, g["vram_total"]), 1))


@app.get("/api/system")
def system():
    vm = psutil.virtual_memory()
    disks = host_disks()
    # attribute app bytes to the disk holding the app dir
    app_mount = ""
    if APP_DIR:
        rel = APP_DIR[len(HOST):] if APP_DIR.startswith(HOST) else APP_DIR
        for d in disks:
            if rel.startswith(d["mount"].rstrip("/") + "/") or d["mount"] == "/":
                if len(d["mount"]) > len(app_mount):
                    app_mount = d["mount"]
    for d in disks:
        d["app_bytes"] = app_usage["total"] if d["mount"] == app_mount else 0

    return {
        "time": int(time.time()),
        "specs": SPECS,
        "uptime": host_uptime(),
        "cpu": {
            "percent": history["cpu"][-1] if history["cpu"] else 0,
            "cores": psutil.cpu_count(),
            "load": list(os.getloadavg()),
        },
        "ram": {"percent": vm.percent, "used": vm.used, "total": vm.total},
        "gpus": gpu_now,
        # per-device temperatures: GPUs (own sensor) + CPU/NVMe/NIC (hwmon)
        "temps": ([{"label": f"GPU {g['index']}", "sub": g["name"],
                    "kind": "gpu", "current": round(g["temp"], 1),
                    "critical": 90.0} for g in gpu_now]
                  + host_temps()),
        "history": {
            "t": list(history["t"]),
            "cpu": list(history["cpu"]),
            "ram": list(history["ram"]),
            "gpus": {k: {"util": list(v["util"]), "vram": list(v["vram"])}
                     for k, v in gpu_hist.items()},
        },
        "disks": sorted(disks, key=lambda d: -d["total"]),
        "app": app_usage,
    }


SPECS = collect_specs()

if __name__ == "__main__":
    threading.Thread(target=sample_loop, daemon=True).start()
    threading.Thread(target=usage_loop, daemon=True).start()
    log.info("sysmon on :8083 (app dir: %s)", APP_DIR or "-")
    uvicorn.run(app, host="0.0.0.0", port=8083, log_level="warning")
