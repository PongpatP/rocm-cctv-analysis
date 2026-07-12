"""Continuous CCTV recorder: one ffmpeg per camera, H.264 stream copy
(no transcode) into 60s MP4 segments, with time-based retention.

Layout: /recordings/<camera>/<YYYYMMDD>_<HHMMSS>.mp4  (local time)
Cameras come from the webapp config (single source of truth); streams
are pulled from the gateway's internal RTSP re-stream, so recording
adds no extra NVR/forwarder connections.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("recorder")

CONFIG = Path(os.environ.get("RECORDER_CONFIG", "/config/config.json"))
OUT = Path(os.environ.get("RECORDER_DIR", "/recordings"))
GATEWAY = os.environ.get("RECORDER_GATEWAY", "rtsp://gateway:8554")
# "sub", "main", or "sub,main"/"both" to record every camera twice.
# Layout stays compatible with playback: sub segments keep living in
# /recordings/<cam>/; main segments go to /recordings/<cam>_main/.
_p = os.environ.get("RECORDER_PROFILE", "sub").strip().lower()
PROFILES = ["sub", "main"] if _p == "both" else \
    [x.strip() for x in _p.split(",") if x.strip()]
RETENTION_H = float(os.environ.get("RETENTION_HOURS", "24"))
# main is ~6x the bitrate of sub — allow a shorter keep for it when disk
# demands (unset = same retention as everything else)
RETENTION_H_MAIN = float(os.environ.get("RETENTION_HOURS_MAIN", "0")) or RETENTION_H
SEGMENT_S = int(os.environ.get("SEGMENT_SECONDS", "60"))
RECONNECT_S = 5


def profile_dir(cam: str, profile: str) -> Path:
    return OUT / (cam if profile == "sub" or len(PROFILES) == 1
                  else f"{cam}_{profile}")

procs: dict[str, subprocess.Popen] = {}
stopping = False


def cameras() -> list[str]:
    cfg = json.loads(CONFIG.read_text())
    cams = []
    for nvr in cfg.get("nvrs", []):
        skip = set(nvr.get("skip") or [])
        for ch in range(1, int(nvr.get("channels", 0)) + 1):
            if ch not in skip:
                cams.append(f"{nvr['id']}_ch{ch:02d}")
    for cam in cfg.get("extra_cameras", []):     # single custom-URL cameras
        if cam.get("id"):
            cams.append(cam["id"])
    return cams


MAX_SEGMENT_BYTES = int(os.environ.get("MAX_SEGMENT_MB", "40")) * 1024 * 1024


def record_loop(cam: str, profile: str) -> None:
    d = profile_dir(cam, profile)
    d.mkdir(parents=True, exist_ok=True)
    key = f"{cam}:{profile}"
    while not stopping:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            # The segment muxer decides when a minute has passed by reading the
            # timestamps the camera sends. On 2026-07-10 at 04:10 two cameras
            # restarted their pts every few seconds, so "60 seconds" never
            # arrived and one file grew to 4.7 hours and 664 MiB. Wall-clock
            # timestamps come from this machine, cannot reset, and cost nothing.
            "-use_wallclock_as_timestamps", "1",
            "-fflags", "+genpts",
            "-i", f"{GATEWAY}/{cam}_{profile}",
            "-c", "copy", "-an",                      # no audio: G.711 won't fit MP4
            "-f", "segment",
            "-segment_time", str(SEGMENT_S),
            "-segment_format", "mp4",
            "-segment_format_options", "movflags=+faststart",
            "-reset_timestamps", "1",
            "-strftime", "1",
            str(d / "%Y%m%d_%H%M%S.mp4"),
        ]
        proc = subprocess.Popen(cmd)
        procs[key] = proc
        threading.Thread(target=watchdog, args=(cam, proc, d), daemon=True).start()
        rc = proc.wait()
        if stopping:
            return
        log.warning("%s: ffmpeg exited rc=%s — reconnecting in %ds", key, rc, RECONNECT_S)
        time.sleep(RECONNECT_S)


def watchdog(cam: str, proc: subprocess.Popen, d: Path) -> None:
    """Second line of defence: a segment that keeps growing is a segment that is
    not rotating. Kill ffmpeg; `record_loop` reconnects and starts a fresh file."""
    while proc.poll() is None and not stopping:
        time.sleep(20)
        try:
            newest = max(d.glob("*.mp4"), key=lambda f: f.stat().st_mtime, default=None)
            if newest and newest.stat().st_size > MAX_SEGMENT_BYTES:
                log.error("%s: %s reached %.0f MiB without rotating — restarting ffmpeg",
                          cam, newest.name, newest.stat().st_size / 1048576)
                proc.terminate()
                return
        except (OSError, ValueError):
            pass


def retention_loop() -> None:
    while not stopping:
        now = time.time()
        removed = 0
        for f in OUT.glob("*/*.mp4"):
            keep_h = RETENTION_H_MAIN if f.parent.name.endswith("_main") else RETENTION_H
            try:
                if f.stat().st_mtime < now - keep_h * 3600:
                    f.unlink()
                    removed += 1
            except OSError:
                pass
        if removed:
            log.info("retention: removed %d segments (keep %.0fh, main %.0fh)",
                     removed, RETENTION_H, RETENTION_H_MAIN)
        time.sleep(600)


def shutdown(*_):
    global stopping
    stopping = True
    for cam, proc in procs.items():
        if proc.poll() is None:
            proc.terminate()
    sys.exit(0)


def main() -> None:
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    cams = cameras()
    log.info("recording %d cameras x %s (%ds segments, keep %.0fh / main %.0fh) -> %s",
             len(cams), PROFILES, SEGMENT_S, RETENTION_H, RETENTION_H_MAIN, OUT)
    for cam in cams:
        for profile in PROFILES:
            threading.Thread(target=record_loop, args=(cam, profile),
                             daemon=True).start()
            time.sleep(0.2)  # stagger startups
    retention_loop()


if __name__ == "__main__":
    main()
