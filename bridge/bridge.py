#!/usr/bin/env python3
"""Detection bridge: detector -> browsers + ClickHouse.

    POST /ingest      JSON array of detection records (from the detector)
    WS   /ws          live broadcast of those records to web UI clients
    GET  /api/health  {"clients": N, "received": M}

Runs on the host (not in the AI container). Start: scripts/start_bridge.sh
"""

import collections
import json
import subprocess
import sys
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import ch
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("bridge")

# Configured via environment (set in docker-compose.yml); the defaults
# also work when run directly on the host for development.
PORT = int(os.environ.get("BRIDGE_PORT", "8081"))
# Everything the bridge reads from disk hangs off this one directory. It used to
# be derived from the JSONL archive's parent, which tied unrelated paths to a
# file that no longer exists: detections live in ClickHouse now.
OUTPUT = Path(os.environ.get(
    "BRIDGE_OUTPUT", Path(__file__).resolve().parent.parent / "output"))

app = FastAPI(title="ccvt detection bridge")

clients: set[WebSocket] = set()
received = 0

OUTPUT.mkdir(parents=True, exist_ok=True)

recent: collections.deque = collections.deque(maxlen=300)


# ---- runtime AI settings (adjustable from the web UI, no restarts) ----

SETTINGS_PATH = OUTPUT / "ai_settings.json"
DEFAULT_SETTINGS = {"min_confidence": 0.30, "disabled_classes": [],
                    # IANA zone the web UI formats times in
                    "timezone": "Asia/Bangkok"}


def load_settings() -> dict:
    try:
        data = json.loads(SETTINGS_PATH.read_text())
        return {**DEFAULT_SETTINGS, **data}
    except (OSError, ValueError):
        return dict(DEFAULT_SETTINGS)


def save_settings(s: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(s, indent=2))


settings = load_settings()


def record_passes(r: dict) -> bool:
    if (r.get("confidence") or 0) < settings["min_confidence"]:
        return False
    if r.get("class_name") in settings["disabled_classes"]:
        return False
    return True


ch.wait_ready()


@app.post("/ingest")
async def ingest(request: Request):
    global received
    records = await request.json()
    if not isinstance(records, list):
        records = [records]
    # the detector runs at a low floor; the user threshold applies here
    records = [r for r in records if record_passes(r)]
    received += len(records)
    if not records:
        return {"ok": True, "count": 0}

    for r in records:
        recent.appendleft(r)
    # boxes AND the skeletons the old sqlite writer discarded
    ch.write_records(records)

    if clients:
        payload = json.dumps({"type": "detections", "records": records})
        dead = []
        for ws in clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)
    return {"ok": True, "count": len(records)}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    log.info("web client connected (%d total)", len(clients))
    try:
        while True:
            # Browsers never send; this blocks until the client disconnects.
            await ws.receive_text()
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        clients.discard(ws)
        log.info("web client disconnected (%d total)", len(clients))


@app.get("/api/health")
def health():
    return {"clients": len(clients), "received": received,
            "clickhouse": ch.stats()}


def _masked(s: dict) -> dict:
    """Never return raw secrets (Anthropic key, HF token) through the API."""
    out = {k: v for k, v in s.items()
           if k not in ("anthropic_api_key", "hf_token")}
    key = s.get("anthropic_api_key") or ""
    out["anthropic_key_set"] = bool(key)
    out["anthropic_key_hint"] = f"…{key[-4:]}" if key else ""
    hf = s.get("hf_token") or ""
    out["hf_token_set"] = bool(hf)
    out["hf_token_hint"] = f"…{hf[-4:]}" if hf else ""
    return out


@app.get("/api/settings")
def get_settings():
    return _masked(settings)


@app.post("/api/settings")
async def set_settings(request: Request):
    body = await request.json()
    mc = body.get("min_confidence")
    if isinstance(mc, (int, float)) and 0.05 <= mc <= 0.95:
        settings["min_confidence"] = round(float(mc), 2)
    dc = body.get("disabled_classes")
    if isinstance(dc, list):
        settings["disabled_classes"] = [str(c) for c in dc][:50]
    tz = body.get("timezone")
    if isinstance(tz, str) and tz.strip():
        try:
            from zoneinfo import ZoneInfo
            ZoneInfo(tz.strip())              # must be a real IANA zone
            settings["timezone"] = tz.strip()
        except Exception:
            pass                              # keep the old zone on bad input
    ak = body.get("anthropic_api_key")
    if isinstance(ak, str) and ak.strip().startswith("sk-ant-"):
        settings["anthropic_api_key"] = ak.strip()   # set/replace only —
    # an empty field means "keep the stored key"
    save_settings(settings)
    log.info("AI settings updated: %s", _masked(settings))
    return _masked(settings)


@app.get("/api/detections/live")
def detections_live(window_ms: int = 3000, cls: str = "person"):
    """Newest in-memory detections per camera — the GATE for the
    2D->3D tracking worker: it runs depth only where the detector (already
    computing in the detection pipeline) currently sees a person. One call, no DB."""
    cutoff = time.time() * 1000 - max(200, min(window_ms, 30000))
    out: dict[str, list] = {}
    for r in recent:                       # newest first
        try:
            ts = datetime.fromisoformat(r["timestamp"]).timestamp() * 1000
        except (KeyError, ValueError):
            continue
        if ts < cutoff:
            break
        if cls and r.get("class_name") != cls:
            continue
        cam = r.get("camera_id")
        if not cam:
            continue
        rec = {
            "box": r.get("bounding_box"), "conf": r.get("confidence"),
            "track": r.get("track_id"), "frame": r.get("frame"),
            "class": r.get("class_name")}
        if r.get("keypoints"):
            rec["keypoints"] = r["keypoints"]
        out.setdefault(cam, []).append(rec)
    return {"cams": out, "cls": cls}


@app.get("/api/detections")
def detections_window(camera: str, start_ms: int, end_ms: int):
    """Raw boxes for a time window — used by the playback overlay."""
    if not re.fullmatch(r"[a-z0-9_]{1,40}", camera or ""):
        return JSONResponse({"error": "bad camera"}, status_code=400)
    end_ms = min(end_ms, start_ms + 180_000)   # cap: one segment + margin
    try:
        # NOTE: alias to ts_ms, never ts — an alias named after the column wins
        # in WHERE, so `ts BETWEEN <DateTime64>` would compare against the Int64
        # milliseconds and silently match nothing.
        rows = ch.query(
            "SELECT toUnixTimestamp64Milli(ts) AS ts_ms, class, conf, track, "
            "x, y, w, h, fw, fh FROM detections "
            "WHERE camera = {cam:String} AND ts BETWEEN fromUnixTimestamp64Milli({a:Int64}) "
            "AND fromUnixTimestamp64Milli({b:Int64}) ORDER BY ts LIMIT 20000",
            {"param_cam": camera, "param_a": str(start_ms), "param_b": str(end_ms)})
    except Exception as e:
        return JSONResponse({"error": f"clickhouse: {e}"}, status_code=503)
    return [
        {"ts": int(r["ts_ms"]), "class_name": r["class"], "confidence": r["conf"],
         "track_id": None if r["track"] == -1 else r["track"],
         "bounding_box": {"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"]},
         "frame": {"width": r["fw"], "height": r["fh"]}}
        for r in rows
    ]


@app.get("/api/detections/summary")
def detections_summary(camera: str, start_ms: int, end_ms: int):
    """Per-minute class summary for a camera over a day, from the minute_stats
    rollup. Powers the little corner markers on the playback timeline (which
    minutes had a person, a dog, a vehicle...). -> {minute_epoch_s: {class: n}}"""
    if not re.fullmatch(r"[a-z0-9_]{1,40}", camera or ""):
        return JSONResponse({"error": "bad camera"}, status_code=400)
    try:
        rows = ch.query(
            "SELECT toUnixTimestamp(minute) AS m, class, uniqMerge(objects) AS n "
            "FROM ccvt.minute_stats "
            "WHERE camera = {cam:String} "
            "AND minute >= fromUnixTimestamp({a:Int64}) "
            "AND minute <  fromUnixTimestamp({b:Int64}) "
            "GROUP BY m, class HAVING n > 0 ORDER BY m",
            {"param_cam": camera, "param_a": str(int(start_ms) // 1000),
             "param_b": str(int(end_ms) // 1000)})
    except Exception as e:
        return JSONResponse({"error": f"clickhouse: {e}"}, status_code=503)
    out: dict[int, dict[str, int]] = {}
    for r in rows:
        out.setdefault(int(r["m"]), {})[r["class"]] = int(r["n"])
    return out


@app.get("/api/poses")
def poses_window(camera: str, start_ms: int, end_ms: int):
    """The skeletons for the same window. Nothing consumed these before, because
    nothing stored them."""
    if not re.fullmatch(r"[a-z0-9_]{1,40}", camera or ""):
        return JSONResponse({"error": "bad camera"}, status_code=400)
    end_ms = min(end_ms, start_ms + 180_000)
    try:
        rows = ch.query(
            "SELECT toUnixTimestamp64Milli(ts) AS ts_ms, track, kp_x, kp_y, kp_conf, "
            "pose_conf, fw, fh FROM poses "
            "WHERE camera = {cam:String} AND ts BETWEEN fromUnixTimestamp64Milli({a:Int64}) "
            "AND fromUnixTimestamp64Milli({b:Int64}) ORDER BY ts LIMIT 20000",
            {"param_cam": camera, "param_a": str(start_ms), "param_b": str(end_ms)})
    except Exception as e:
        return JSONResponse({"error": f"clickhouse: {e}"}, status_code=503)
    return [
        {"ts": int(r["ts_ms"]), "track_id": None if r["track"] == -1 else r["track"],
         "keypoints": list(zip(r["kp_x"], r["kp_y"], r["kp_conf"])),
         "confidence": r["pose_conf"],
         "frame": {"width": r["fw"], "height": r["fh"]}}
        for r in rows
    ]


# ---- recordings playback (segments written by the recorder service) ---

REC_DIR = Path(os.environ.get("BRIDGE_RECORDINGS", "/recordings"))
CAM_RE = re.compile(r"^[a-z0-9_]{1,40}$")
SEG_RE = re.compile(r"^(\d{8})_(\d{6})\.mp4$")


def _rec_variants(cam):
    """Folders to serve a camera from, MAIN first: playback shows the highest
    quality that's still kept (main = last 3 days), falling back to sub (7
    days) for older footage. Recorder writes main to <cam>_main, sub to <cam>."""
    return [REC_DIR / f"{cam}_main", REC_DIR / cam]


def _rec_dir_for_date(cam, date):
    """The folder that actually holds this camera+date, main preferred."""
    for d in _rec_variants(cam):
        if d.is_dir() and any(d.glob(f"{date}_*.mp4")):
            return d
    return REC_DIR / cam


@app.get("/api/recordings/{cam}/dates")
def recordings_dates(cam: str):
    if not CAM_RE.fullmatch(cam):
        return JSONResponse({"error": "bad camera"}, status_code=400)
    dates = set()
    for d in _rec_variants(cam):
        if d.is_dir():
            for f in d.iterdir():
                m = SEG_RE.fullmatch(f.name)
                if m:
                    dates.add(m.group(1))
    return sorted(dates)


def _seg_start_epoch(date: str, tm: str) -> float:
    # recorder names segments with LOCAL wall-clock (strftime); the container's
    # TZ makes datetime.timestamp() interpret them in that same local zone.
    from datetime import datetime
    return datetime.strptime(date + tm, "%Y%m%d%H%M%S").timestamp()


# NOTE: this MUST be registered before "/api/recordings/{cam}" below, or that
# path parameter would capture "frame" and 400.
@app.get("/api/recordings/frame")
def recordings_frame(cam: str, ts_ms: int):
    """One JPEG frame from `cam`'s recording at wall-clock `ts_ms`. Lets the
    playback review pull the SAME MOMENT from a surrounding camera. Prefers the
    lighter sub-stream."""
    import subprocess
    from fastapi.responses import Response
    if not CAM_RE.fullmatch(cam or ""):
        return JSONResponse({"error": "bad camera"}, status_code=400)
    ts = ts_ms / 1000.0
    date = time.strftime("%Y%m%d", time.localtime(ts))
    hit = None
    for d in (REC_DIR / cam, REC_DIR / f"{cam}_main"):   # sub first
        if not d.is_dir():
            continue
        for f in sorted(d.glob(f"{date}_*.mp4")):
            m = SEG_RE.fullmatch(f.name)
            if not m:
                continue
            start = _seg_start_epoch(m.group(1), m.group(2))
            if start <= ts < start + 70:
                hit = (f, max(0.0, ts - start))
                break
        if hit:
            break
    if not hit:
        return JSONResponse({"error": "no footage at that time"}, status_code=404)
    path, off = hit
    try:
        out = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{off:.2f}",
             "-i", str(path), "-frames:v", "1", "-q:v", "4", "-f", "image2", "-"],
            capture_output=True, timeout=15).stdout
    except Exception as e:
        return JSONResponse({"error": f"ffmpeg: {e}"}, status_code=503)
    if not out:
        return JSONResponse({"error": "could not extract frame"}, status_code=503)
    return Response(content=out, media_type="image/jpeg")


@app.get("/api/recordings/{cam}")
def recordings_list(cam: str, date: str = ""):
    if not CAM_RE.fullmatch(cam) or not re.fullmatch(r"\d{8}", date or ""):
        return JSONResponse({"error": "bad camera/date"}, status_code=400)
    d = _rec_dir_for_date(cam, date)
    out = []
    if d.is_dir():
        now = time.time()
        quality = "main" if d.name.endswith("_main") else "sub"
        for f in sorted(d.glob(f"{date}_*.mp4")):
            if not SEG_RE.fullmatch(f.name):
                continue
            st = f.stat()
            # skip the segment still being written (fresh mtime)
            if now - st.st_mtime < 5:
                continue
            out.append({"file": f.name, "time": f.name[9:15],
                        "size": st.st_size, "quality": quality})
    return out


@app.get("/api/recordings/file/{cam}/{fname}")
def recordings_file(cam: str, fname: str):
    if not CAM_RE.fullmatch(cam) or not SEG_RE.fullmatch(fname):
        return JSONResponse({"error": "bad path"}, status_code=400)
    for d in _rec_variants(cam):
        path = d / fname
        if path.is_file():
            return FileResponse(path, media_type="video/mp4")
    return JSONResponse({"error": "not found"}, status_code=404)


# ---- per-camera floor assignments ----
# Ground truth for which floor each camera is on (the camera graph groups by
# it, and the ReID gate treats a match across DIFFERENT floors as a false
# positive — look-alike corridors). This used to live in the calib container's
# floors.json; calib was removed on the AMD server, so the bridge now owns it.

FLOORS_PATH = OUTPUT / "floors.json"
_floors_lock = threading.Lock()


def _load_floors() -> dict:
    try:
        return json.loads(FLOORS_PATH.read_text())
    except (OSError, ValueError):
        return {"floors": {}, "n_floors": 8}


@app.get("/api/calib/floors")
def get_floors():
    return _load_floors()


@app.post("/api/calib/floors")
async def set_floors(request: Request):
    body = await request.json()
    with _floors_lock:
        data = _load_floors()
        data.setdefault("floors", {})
        patch = body.get("floors_patch")
        if isinstance(patch, dict):
            data["floors"].update({str(k): str(v) for k, v in patch.items()})
        if isinstance(body.get("n_floors"), int):
            data["n_floors"] = body["n_floors"]
        FLOORS_PATH.write_text(json.dumps(data, indent=2))
    return data


# ---- detector classes (Settings -> Detector classes) -----------------------
# RT-DETR always sees all 80 COCO classes; airocm re-reads this file every 2 s
# and filters. Saving here applies live — no restart, no recompile.
CLASSES_PATH = OUTPUT / "ai_classes.json"
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
DEFAULT_CLASSES = ["person", "bicycle", "car", "motorcycle", "truck"]


@app.get("/api/config/classes")
def get_classes():
    try:
        data = json.loads(CLASSES_PATH.read_text())
        # both shapes exist in the wild: {"selected": [...]} and a bare list
        sel = (data.get("selected") if isinstance(data, dict) else data) \
            or DEFAULT_CLASSES
    except (OSError, ValueError):
        sel = DEFAULT_CLASSES
    interval = 0
    try:
        interval = max(0, int((data if isinstance(data, dict) else {}).get("interval", 0)))
    except (TypeError, ValueError, NameError):
        interval = 0
    return {"selected": [c for c in sel if c in COCO80], "available": COCO80,
            "interval": interval}


@app.post("/api/config/classes")
async def set_classes(request: Request):
    body = await request.json()
    sel = [c for c in (body.get("selected") or []) if c in COCO80]
    if not sel:
        return {"ok": False, "error": "select at least one class"}
    try:
        interval = max(0, min(30, int(body.get("interval", 0))))
    except (TypeError, ValueError):
        interval = 0
    CLASSES_PATH.write_text(json.dumps({"selected": sel, "interval": interval}, indent=2))
    # keep the web UI's own class list (legend colours, display toggles) in step
    try:
        cfg_path = REPO / "webapp" / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["classes"] = sel
        cfg_path.write_text(json.dumps(cfg, indent=2))
    except (OSError, ValueError):
        pass
    log.info("detector classes -> %s (interval=%d)", sel, interval)
    return {"ok": True, "selected": sel, "interval": interval}


# ---- AI pipeline status (dashboard) ---------------------------------------
def _svc_up(url, timeout=2.0):
    import requests as rq
    try:
        return rq.get(url, timeout=timeout).status_code < 500
    except Exception:
        return False


@app.get("/api/ai/status")
def ai_status():
    """Live up/down for every AI model in the pipeline — the dashboard's
    AI panel. Cheap: HTTP pings where a service exposes health, ClickHouse
    freshness where it doesn't (detection/captioning have no HTTP)."""
    # detection + tracking share the airocm service: fresh detections = up
    det_up = False
    try:
        r = ch.query("SELECT count() c FROM detections WHERE ts > now() - INTERVAL 2 MINUTE")
        det_up = bool(r and int(r[0]["c"]) > 0)
    except Exception:
        pass
    cap_up = False
    try:
        r = ch.query("SELECT count() c FROM clip_captions WHERE ts > now() - INTERVAL 10 MINUTE")
        cap_up = bool(r and int(r[0]["c"]) > 0)
    except Exception:
        pass
    reid_up = _svc_up("http://siglip:8085/reid/healthz")
    ground_up = _svc_up("http://ground:8090/healthz")
    vllm_up = _svc_up("http://vllm:8000/health")

    def _count(sql):
        try:
            r = ch.query(sql)
            return int(r[0]["c"]) if r else 0
        except Exception:
            return None
    H = "ts > now() - INTERVAL 24 HOUR"
    out = {
        "det": _count(f"SELECT count() c FROM detections WHERE {H}"),
        "trk": _count(f"SELECT uniqExact(camera, track) c FROM detections WHERE {H} AND track >= 0"),
        "reid": _count(f"SELECT count() c FROM sightings WHERE {H}"),
        "bhv": _count(f"SELECT count() c FROM behaviors WHERE {H}"),
        "cap": _count("SELECT count() c FROM clip_captions WHERE caption != ''"),
    }
    def _n(v):
        return "—" if v is None else (f"{v/1e6:.1f}M" if v >= 1e6
                                      else f"{v/1e3:.1f}k" if v >= 1e3 else str(v))
    models = [
        {"role": "Object detection", "name": "RT-DETR R50", "up": det_up,
         "meta": "MI300X · MIGraphX", "out": f"{_n(out['det'])} boxes / 24h"},
        {"role": "In-camera tracking", "name": "BoT-SORT + OSNet", "up": det_up,
         "meta": "appearance ReID", "out": f"{_n(out['trk'])} tracks / 24h"},
        {"role": "Cross-camera ReID", "name": "YoutuReID", "up": reid_up,
         "meta": "Global IDs", "out": f"{_n(out['reid'])} sightings / 24h"},
        {"role": "Visual grounding", "name": "Grounding DINO", "up": ground_up,
         "meta": "point-at-anything", "out": "on demand"},
        {"role": "VLM · vision", "name": "Gemma 4 31B", "up": vllm_up,
         "meta": "vLLM · this machine", "out": f"{_n(out['bhv'])} behaviours / 24h"},
        {"role": "LLM · text", "name": "Gemma 4 31B", "up": vllm_up,
         "meta": "vLLM · this machine", "out": "investigator + stories"},
        {"role": "Video captioning", "name": "Gemma (vlmscan)", "up": cap_up,
         "meta": "narrates recordings", "out": f"{_n(out['cap'])} captions"},
    ]
    return {"models": models, "up": sum(m["up"] for m in models),
            "total": len(models)}


# ---- camera / NVR management (Settings -> Cameras & NVRs) ------------------
# The UI edits NVRs freely; changes land in output/nvrs_override.json (so
# config.local.yaml keeps its comments + credentials), configure.py regenerates
# every derived config, and new streams are pushed into go2rtc LIVE. Detection
# (airocm) and the recorder read their camera list at start — they pick up
# added/removed cameras on their next restart.
REPO = Path(os.environ.get("REPO_DIR", "/repo"))
NVRS_OVERRIDE = OUTPUT / "nvrs_override.json"
_nvr_lock = threading.Lock()

_NVR_ID_RE = re.compile(r"^[a-z][a-z0-9]{0,15}$")


def _effective_nvrs() -> list:
    try:
        nvrs = json.loads(NVRS_OVERRIDE.read_text()).get("nvrs")
        if isinstance(nvrs, list) and nvrs:
            return nvrs
    except (OSError, ValueError):
        pass
    try:
        import yaml
        return yaml.safe_load((REPO / "config.local.yaml").read_text())["nvrs"]
    except Exception:
        return []


_CAM_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,23}$")
_FLOOR_RE = re.compile(r"^\d{1,2}F$", re.I)


def _parse_sources(text, current):
    """One line = one source. Two shapes, documented in the UI:

        <id> <host[:port]> <user> <password> <channels> [skip=14,16] [floor=1F]
        cam <id> <rtsp-url> [sub=<rtsp-url>] [floor=3F]

    `#` starts a comment; password `•••` or `-` keeps the stored one. Floors
    collect into a floors_patch. Errors carry their line number — a form that
    can't say WHAT is wrong is how weird port-forward configs go unnoticed."""
    nvrs, cams, floors, errors = [], [], {}, []
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tok = line.split()
        opts = {k.lower(): v for k, v in
                (t.split("=", 1) for t in tok if "=" in t)}
        pos = [t for t in tok if "=" not in t]
        floor = (opts.get("floor") or "").upper()
        if floor and not _FLOOR_RE.fullmatch(floor):
            errors.append(f"line {ln}: floor must look like 1F..99F")
            continue
        if pos and pos[0].lower() == "cam":
            if len(pos) < 3:
                errors.append(f"line {ln}: cam needs `cam <id> <rtsp-url>`")
                continue
            cid, url = pos[1].lower(), pos[2]
            if not _CAM_ID_RE.fullmatch(cid):
                errors.append(f"line {ln}: bad camera id {cid!r} (a-z, 0-9, _)")
                continue
            if not re.match(r"^rtsps?://\S+$", url):
                errors.append(f"line {ln}: {cid}: url must start rtsp://")
                continue
            sub = opts.get("sub", "")
            if sub and not re.match(r"^rtsps?://\S+$", sub):
                errors.append(f"line {ln}: {cid}: sub= must be an rtsp url")
                continue
            cams.append({"id": cid, "url": url, "url_sub": sub,
                         "name": opts.get("name") or cid})
            if floor:
                floors[cid] = floor
            continue
        if len(pos) < 5:
            errors.append(f"line {ln}: need `<id> <host[:port]> <user> "
                          f"<password> <channels>` (got {len(pos)} fields)")
            continue
        nid, hostport, user, pw, chans = pos[:5]
        nid = nid.lower()
        if not _NVR_ID_RE.fullmatch(nid):
            errors.append(f"line {ln}: bad NVR id {nid!r}")
            continue
        host, _, port = hostport.partition(":")
        if not re.fullmatch(r"[a-zA-Z0-9._-]{1,80}", host):
            errors.append(f"line {ln}: {nid}: bad host {host!r}")
            continue
        try:
            port = int(port) if port else 554
            assert 1 <= port <= 65535
        except (ValueError, AssertionError):
            errors.append(f"line {ln}: {nid}: bad port")
            continue
        try:
            channels = int(chans)
            assert 1 <= channels <= 64
        except (ValueError, AssertionError):
            errors.append(f"line {ln}: {nid}: channels must be 1-64")
            continue
        skip = []
        if opts.get("skip"):
            try:
                skip = [int(x) for x in opts["skip"].split(",") if x.strip()]
            except ValueError:
                errors.append(f"line {ln}: {nid}: skip= must be numbers")
                continue
        if pw in ("•••", "-", "***"):
            pw = str((current.get(nid) or {}).get("password") or "")
        nvrs.append({"id": nid, "name": opts.get("name") or nid.upper(),
                     "host": host, "port": port, "user": user,
                     "password": pw, "channels": channels, "skip": skip})
        if floor:
            for ch in range(1, channels + 1):
                if ch not in skip:
                    floors[f"{nid}_ch{ch:02d}"] = floor
    return nvrs, cams, floors, errors


def _render_sources(nvrs, cams):
    """The stored config, back as editable lines (passwords masked)."""
    lines = []
    for n in nvrs:
        parts = [n["id"], f"{n['host']}:{n.get('port', 554)}",
                 n.get("user", "admin"), "•••" if n.get("password") else "-",
                 str(n.get("channels", 8))]
        if n.get("skip"):
            parts.append("skip=" + ",".join(str(x) for x in n["skip"]))
        lines.append(" ".join(parts))
    for c in cams:
        parts = ["cam", c["id"], c["url"]]
        if c.get("url_sub"):
            parts.append(f"sub={c['url_sub']}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _stored_cams() -> list:
    try:
        cams = json.loads(NVRS_OVERRIDE.read_text()).get("cameras")
        return cams if isinstance(cams, list) else []
    except (OSError, ValueError):
        return []


# ---- camera BOXES: the owner's final design -------------------------------
# One square box per camera. A box holds ONE rtsp url + a floor. Nothing to
# learn. Internally, boxes whose url is a standard NVR channel
# (rtsp://user:pw@host:port/cam/realmonitor?channel=N&subtype=0) are regrouped
# into nvr entries so the live-wall's per-NVR tabs keep working; every other
# url is a first-class custom camera. The user never sees this grouping.
_DAHUA_RE = re.compile(
    r"^rtsp://([^:@/]+):([^@/]*)@([a-zA-Z0-9._-]+):(\d+)"
    r"/cam/realmonitor\?channel=(\d+)&subtype=0$")


def _boxes_from_config():
    """Expand the stored config into per-camera boxes (passwords masked)."""
    boxes = []
    for n in _effective_nvrs():
        pw = "•••" if n.get("password") else ""
        skip = set(n.get("skip") or [])
        for ch in range(1, int(n.get("channels", 0)) + 1):
            if ch in skip:
                continue
            boxes.append({
                "id": f"{n['id']}_ch{ch:02d}",
                "url": (f"rtsp://{n.get('user','admin')}:{pw}@{n['host']}"
                        f":{n.get('port',554)}/cam/realmonitor?channel={ch}"
                        f"&subtype=0")})
    for c in _stored_cams():
        url = c["url"]
        m = re.match(r"^(rtsps?://[^:@/]+:)([^@/]+)(@.*)$", url)
        boxes.append({"id": c["id"],
                      "url": f"{m.group(1)}•••{m.group(3)}" if m else url})
    floors = _load_floors().get("floors", {})
    for b in boxes:
        b["floor"] = floors.get(b["id"], "")
    return boxes


def _real_password(box_id, masked_url):
    """A ••• password in a box url means: keep whatever is stored."""
    for n in _effective_nvrs():
        if box_id.startswith(n["id"] + "_ch"):
            return str(n.get("password") or "")
    for c in _stored_cams():
        if c["id"] == box_id:
            m = re.match(r"^rtsps?://[^:@/]+:([^@/]+)@", c.get("url") or "")
            return m.group(1) if m else ""
    return ""


@app.get("/api/config/cameras")
def get_camera_boxes():
    return {"boxes": _boxes_from_config(),
            "n_floors": _load_floors().get("n_floors", 8)}


@app.post("/api/config/cameras")
async def set_camera_boxes(request: Request):
    body = await request.json()
    boxes = body.get("boxes")
    if not isinstance(boxes, list):
        return {"ok": False, "error": "boxes must be a list"}
    groups, customs, floors_patch, errors = {}, [], {}, []
    used_ids = set()
    auto_n = 0
    for i, b in enumerate(boxes, 1):
        url = str(b.get("url") or "").strip()
        if not url:
            continue                        # an empty box is just ignored
        if not re.match(r"^rtsps?://\S+$", url):
            errors.append(f"box {i}: url must start rtsp://")
            continue
        bid = str(b.get("id") or "").strip().lower()
        if "•••" in url:
            url = url.replace("•••", _real_password(bid, url) or "•••", 1)
        floor = str(b.get("floor") or "").strip().upper()
        if floor and not _FLOOR_RE.fullmatch(floor):
            errors.append(f"box {i}: floor must look like 1F..99F")
            continue
        m = _DAHUA_RE.match(url)
        if m and bid and not re.fullmatch(r"[a-z0-9]+_ch\d{2}", bid):
            m = None                         # user renamed it: keep as custom
        if m:                                # standard NVR channel -> regroup
            user, pw, host, port, ch = (m.group(1), m.group(2), m.group(3),
                                        int(m.group(4)), int(m.group(5)))
            key = (host, port, user, pw)
            g = groups.setdefault(key, {"channels": set()})
            if ch in g["channels"]:
                errors.append(f"box {i}: duplicate channel {ch} on {host}")
                continue
            g["channels"].add(ch)
            gid = bid.split("_ch")[0] if "_ch" in bid else None
            if gid and _NVR_ID_RE.fullmatch(gid):
                g.setdefault("id", gid)
            cam_id = bid or None            # resolved after grouping
            g.setdefault("boxes", []).append((ch, cam_id, floor))
        else:                                # anything else: custom camera
            if not bid:
                auto_n += 1
                while f"cam{auto_n:02d}" in used_ids:
                    auto_n += 1
                bid = f"cam{auto_n:02d}"
            if not _CAM_ID_RE.fullmatch(bid):
                errors.append(f"box {i}: bad camera id {bid!r}")
                continue
            if bid in used_ids:
                errors.append(f"box {i}: duplicate camera id {bid!r}")
                continue
            used_ids.add(bid)
            customs.append({"id": bid, "url": url, "url_sub": "", "name": bid})
            if floor:
                floors_patch[bid] = floor
    if errors:
        return {"ok": False, "error": "; ".join(errors[:4])}
    nvrs, n_auto = [], 0
    for (host, port, user, pw), g in groups.items():
        gid = g.get("id")
        if not gid:
            n_auto += 1
            existing = {n["id"] for n in nvrs}
            gid = next(f"nvr{k}" for k in range(1, 99)
                       if f"nvr{k}" not in existing)
        channels = max(g["channels"])
        skip = [ch for ch in range(1, channels + 1) if ch not in g["channels"]]
        nvrs.append({"id": gid, "name": gid.upper(), "host": host,
                     "port": port, "user": user, "password": pw,
                     "channels": channels, "skip": skip})
        for ch, cam_id, floor in g["boxes"]:
            if floor:
                floors_patch[cam_id or f"{gid}_ch{ch:02d}"] = floor
    if not nvrs and not customs:
        return {"ok": False, "error": "no cameras — add at least one box"}
    with _nvr_lock:
        NVRS_OVERRIDE.write_text(json.dumps(
            {"nvrs": nvrs, "cameras": customs}, indent=2))
        r = subprocess.run([sys.executable, str(REPO / "configure.py")],
                           cwd=str(REPO), capture_output=True, text=True,
                           timeout=30)
        if r.returncode != 0:
            return {"ok": False,
                    "error": f"configure.py: {(r.stderr or r.stdout)[-300:]}"}
    if floors_patch:
        with _floors_lock:
            data = _load_floors()
            data.setdefault("floors", {}).update(floors_patch)
            FLOORS_PATH.write_text(json.dumps(data, indent=2))
    pushed = _push_streams_live()
    total = sum(n["channels"] - len(n["skip"]) for n in nvrs) + len(customs)
    log.info("camera boxes saved: %d cameras (%d NVR-grouped + %d custom), "
             "%d streams live", total, total - len(customs), len(customs), pushed)
    return {"ok": True, "cameras": total, "streams_live": pushed,
            "floors_set": len(floors_patch),
            "restart_needed": ["detection (~1 min)", "recorder", "siglip"]}


@app.get("/api/config/nvrs")
def get_nvrs():
    out = []
    for n in _effective_nvrs():
        n = dict(n)
        n["password"] = "•••" if n.get("password") else ""
        out.append(n)
    return {"nvrs": out,
            "sources_text": _render_sources(_effective_nvrs(), _stored_cams()),
            "cameras": _stored_cams()}


@app.post("/api/config/nvrs")
async def set_nvrs(request: Request):
    body = await request.json()
    current = {n.get("id"): n for n in _effective_nvrs()}
    floors_patch, cams = {}, _stored_cams()
    if isinstance(body.get("sources_text"), str):
        nvrs, cams, floors_patch, errors = _parse_sources(
            body["sources_text"], current)
        if errors:
            return {"ok": False, "error": "; ".join(errors[:4])}
        if not nvrs and not cams:
            return {"ok": False, "error": "no sources — at least one line"}
        # skip re-validation below for parsed nvrs: the parser already did it
        clean = nvrs
        with _nvr_lock:
            NVRS_OVERRIDE.write_text(json.dumps(
                {"nvrs": clean, "cameras": cams}, indent=2))
            r = subprocess.run([sys.executable, str(REPO / "configure.py")],
                               cwd=str(REPO), capture_output=True, text=True,
                               timeout=30)
            if r.returncode != 0:
                return {"ok": False,
                        "error": f"configure.py: {(r.stderr or r.stdout)[-300:]}"}
        if floors_patch:
            with _floors_lock:
                data = _load_floors()
                data.setdefault("floors", {}).update(floors_patch)
                FLOORS_PATH.write_text(json.dumps(data, indent=2))
        pushed = _push_streams_live()
        n_cams = sum(c["channels"] - len(c["skip"]) for c in clean) + len(cams)
        log.info("sources updated: %d NVRs + %d custom cams = %d cameras, "
                 "%d streams live", len(clean), len(cams), n_cams, pushed)
        return {"ok": True, "cameras": n_cams, "streams_live": pushed,
                "floors_set": len(floors_patch),
                "restart_needed": ["airocm (detection — ~1 min, cached compile)",
                                   "recorder", "siglip"]}
    nvrs = body.get("nvrs")
    if not isinstance(nvrs, list) or not nvrs:
        return {"ok": False, "error": "nvrs must be a non-empty list"}
    clean, seen = [], set()
    for n in nvrs:
        nid = str(n.get("id") or "").strip().lower()
        if not _NVR_ID_RE.fullmatch(nid):
            return {"ok": False, "error": f"bad NVR id {nid!r} (a-z, digits)"}
        if nid in seen:
            return {"ok": False, "error": f"duplicate NVR id {nid!r}"}
        seen.add(nid)
        try:
            channels = int(n.get("channels"))
            assert 1 <= channels <= 64
        except (TypeError, ValueError, AssertionError):
            return {"ok": False, "error": f"{nid}: channels must be 1-64"}
        host = str(n.get("host") or "").strip()
        if not re.fullmatch(r"[a-zA-Z0-9._-]{1,80}", host):
            return {"ok": False, "error": f"{nid}: bad host"}
        skip = [int(x) for x in (n.get("skip") or []) if str(x).strip().isdigit()]
        pw = str(n.get("password") or "")
        if pw in ("", "•••"):                 # blank/masked = keep the old one
            pw = str((current.get(nid) or {}).get("password") or "")
        clean.append({"id": nid, "name": str(n.get("name") or nid)[:40],
                      "host": host, "port": int(n.get("port") or 554),
                      "user": str(n.get("user") or "admin")[:40],
                      "password": pw, "channels": channels, "skip": skip})
    with _nvr_lock:
        NVRS_OVERRIDE.write_text(json.dumps(
            {"nvrs": clean, "cameras": _stored_cams()}, indent=2))
        r = subprocess.run([sys.executable, str(REPO / "configure.py")],
                           cwd=str(REPO), capture_output=True, text=True,
                           timeout=30)
        if r.returncode != 0:
            return {"ok": False, "error": f"configure.py: {(r.stderr or r.stdout)[-300:]}"}
    pushed = _push_streams_live()
    n_cams = sum(c["channels"] - len(c["skip"]) for c in clean)
    log.info("NVR config updated from the UI: %d NVRs, %d cameras, "
             "%d streams pushed live", len(clean), n_cams, pushed)
    return {"ok": True, "cameras": n_cams, "streams_live": pushed,
            "restart_needed": ["airocm (AI detection — recompiles, ~10 min)",
                               "recorder", "siglip"]}


def _push_streams_live() -> int:
    """New/changed streams work in the live view immediately: go2rtc accepts
    runtime stream definitions over its API, no restart needed."""
    import requests as rq
    n = 0
    try:
        txt = (REPO / "gateway" / "go2rtc.yaml").read_text()
        in_streams = False
        for line in txt.splitlines():
            if line.startswith("streams:"):
                in_streams = True
                continue
            if in_streams and line.startswith("  ") and ": " in line:
                name, src = line.strip().split(": ", 1)
                try:
                    rq.put("http://gateway:1984/api/streams",
                           params={"name": name, "src": src}, timeout=4)
                    n += 1
                except Exception:
                    pass
    except OSError:
        pass
    return n


# ---- calibration results (MapAnything scan, see calib/) ----

CALIB_DIR = OUTPUT / "calib" / "result"


@app.get("/api/calib/map")
def calib_map():
    path = CALIB_DIR / "map.json"
    if not path.is_file():
        return JSONResponse(
            {"error": "not built yet", "hint": "run a scan on /map3d.html"},
            status_code=404)
    return FileResponse(path, media_type="application/json",
                        headers={"Cache-Control": "no-cache"})


@app.get("/api/calib/points.bin")
def calib_points():
    path = CALIB_DIR / "points.bin"
    if not path.is_file():
        return JSONResponse({"error": "not built yet"}, status_code=404)
    return FileResponse(path, media_type="application/octet-stream",
                        headers={"Cache-Control": "no-cache"})


@app.get("/api/calib/scene.glb")
def calib_scene():
    path = CALIB_DIR / "scene.glb"
    if not path.is_file():
        return JSONResponse({"error": "not built yet"}, status_code=404)
    return FileResponse(path, media_type="model/gltf-binary",
                        headers={"Cache-Control": "no-cache"})


@app.get("/api/calib/thumb/{cam}")
def calib_thumb(cam: str):
    if not CAM_RE.fullmatch(cam):
        return JSONResponse({"error": "bad camera"}, status_code=400)
    path = CALIB_DIR / "thumbs" / f"{cam}.jpg"
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    # results change on every scan — force revalidation (a heuristically
    # cached copy from a previous scan mixes old/new data in the viewer)
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "no-cache"})


# ---- face registration (consumed later by the recognition service) ----

FACES_DIR = OUTPUT / "faces"
FACES_REGISTRY = FACES_DIR / "registry.json"
IMG_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
MAX_IMG_BYTES = 10 * 1024 * 1024


def load_faces() -> list[dict]:
    try:
        data = json.loads(FACES_REGISTRY.read_text())
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def save_faces(items: list[dict]) -> None:
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    tmp = FACES_REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, indent=2))
    tmp.replace(FACES_REGISTRY)


def _clean_id(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{12}", value or ""):
        raise ValueError("bad id")
    return value


async def _store_images(person_id: str, images: list[UploadFile]) -> list[str]:
    dest = FACES_DIR / "images" / person_id
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for up in images[:10]:
        ext = IMG_TYPES.get(up.content_type)
        if not ext:
            continue
        data = await up.read()
        if not data or len(data) > MAX_IMG_BYTES:
            continue
        fname = uuid.uuid4().hex[:12] + ext
        (dest / fname).write_bytes(data)
        saved.append(fname)
    return saved


@app.get("/api/faces")
def faces_list():
    return load_faces()


@app.post("/api/faces")
async def faces_create(
    name: str = Form(...),
    role: str = Form(""),
    images: list[UploadFile] = File([]),
):
    name = name.strip()[:60]
    role = role.strip()[:40]
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)
    person_id = uuid.uuid4().hex[:12]
    files = await _store_images(person_id, images)
    if not files:
        return JSONResponse({"error": "at least one valid image (jpg/png/webp, <10MB)"},
                            status_code=400)
    items = load_faces()
    person = {"id": person_id, "name": name, "role": role,
              "images": files, "created_at": time.time()}
    items.append(person)
    save_faces(items)
    log.info("face registered: %s (%s), %d images", name, role, len(files))
    return person


@app.post("/api/faces/{person_id}/images")
async def faces_add_images(person_id: str, images: list[UploadFile] = File(...)):
    try:
        person_id = _clean_id(person_id)
    except ValueError:
        return JSONResponse({"error": "bad id"}, status_code=400)
    items = load_faces()
    person = next((p for p in items if p["id"] == person_id), None)
    if not person:
        return JSONResponse({"error": "not found"}, status_code=404)
    files = await _store_images(person_id, images)
    if not files:
        return JSONResponse({"error": "no valid images"}, status_code=400)
    person["images"] = (person.get("images") or []) + files
    save_faces(items)
    return person


@app.delete("/api/faces/{person_id}")
def faces_delete(person_id: str):
    try:
        person_id = _clean_id(person_id)
    except ValueError:
        return JSONResponse({"error": "bad id"}, status_code=400)
    items = load_faces()
    keep = [p for p in items if p["id"] != person_id]
    if len(keep) == len(items):
        return JSONResponse({"error": "not found"}, status_code=404)
    save_faces(keep)
    imgdir = FACES_DIR / "images" / person_id
    if imgdir.is_dir():
        for f in imgdir.iterdir():
            f.unlink(missing_ok=True)
        imgdir.rmdir()
    log.info("face deleted: %s", person_id)
    return {"ok": True}


@app.get("/api/faces/image/{person_id}/{fname}")
def faces_image(person_id: str, fname: str):
    try:
        person_id = _clean_id(person_id)
    except ValueError:
        return JSONResponse({"error": "bad id"}, status_code=400)
    if not re.fullmatch(r"[0-9a-f]{12}\.(jpg|png|webp)", fname):
        return JSONResponse({"error": "bad filename"}, status_code=400)
    path = FACES_DIR / "images" / person_id / fname
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.get("/api/history")
def history(limit: int = 100, camera: str = ""):
    items = list(recent)
    if camera:
        items = [r for r in items if r.get("camera_id") == camera]
    return items[: max(1, min(limit, 300))]


@app.get("/api/stats")
def stats(hours: int = 24):
    """Distinct objects per bucket / camera / class.

    `objects` counts distinct track ids, so a person walking through is 1, not
    200 frames. In ClickHouse that is `uniqMerge` over the AggregatingMergeTree
    state the materialised view keeps — summing per-minute uniques would double
    count anyone who crosses a minute boundary, which the sqlite version did.
    """
    hours = max(1, min(hours, 24 * 30))
    now_min = int(time.time() // 60)
    since = now_min - hours * 60
    bucket = max(1, (hours * 60) // 96)  # ~96 points regardless of range
    since_s = since * 60

    q = ("SELECT intDiv(toUnixTimestamp(minute) - {since:Int64}, {bucket:Int64}) AS idx, "
         "class, uniqMerge(objects) AS objs FROM minute_stats "
         "WHERE minute >= fromUnixTimestamp({since:Int64}) GROUP BY idx, class")
    args = {"param_since": str(since_s), "param_bucket": str(bucket * 60)}
    try:
        rows = ch.query(q, args)
        n_buckets = (hours * 60 + bucket - 1) // bucket
        buckets = [{"t": (since + i * bucket) * 60, "classes": {}}
                   for i in range(n_buckets)]
        for r in rows:
            i = int(r["idx"])
            if 0 <= i < n_buckets:
                buckets[i]["classes"][r["class"]] = int(r["objs"])

        cameras = [(r["camera"], int(r["objs"])) for r in ch.query(
            "SELECT camera, uniqMerge(objects) AS objs FROM minute_stats "
            "WHERE minute >= fromUnixTimestamp({since:Int64}) "
            "GROUP BY camera ORDER BY objs DESC", {"param_since": str(since_s)})]
        totals = {r["class"]: int(r["objs"]) for r in ch.query(
            "SELECT class, uniqMerge(objects) AS objs FROM minute_stats "
            "WHERE minute >= fromUnixTimestamp({since:Int64}) GROUP BY class",
            {"param_since": str(since_s)})}
        lh = ch.query("SELECT uniqMerge(objects) AS objs FROM minute_stats "
                      "WHERE minute >= fromUnixTimestamp({since:Int64})",
                      {"param_since": str((now_min - 60) * 60)})
        last_hour = int(lh[0]["objs"]) if lh else 0
    except Exception as e:
        return JSONResponse({"error": f"clickhouse: {e}"}, status_code=503)

    return {
        "hours": hours,
        "bucket_minutes": bucket,
        "buckets": buckets,
        "cameras": [{"camera": c, "objects": o} for c, o in cameras],
        "totals": totals,
        "total": sum(totals.values()),
        "last_hour": last_hour,
    }


if __name__ == "__main__":
    log.info("detection bridge on :%d, results -> clickhouse", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
