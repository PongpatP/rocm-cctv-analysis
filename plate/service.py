"""Thai licence-plate OCR for the car-park entry camera.

Scope, as the owner set it: **nvr1_ch03 only**. That camera is labelled "Covered
Car Park Entry" in the Camera graph; running plate OCR on twenty-seven corridors
would burn GPU 1 to read nothing.

It adds no vehicle detector. The detection pipeline already detects and tracks cars,
motorcycles, trucks and buses on every camera and broadcasts them on the bridge's
WebSocket, with a stable `track_id`. This service listens for that camera's
vehicles, fetches the MAIN frame once per interval, and looks for a plate inside
each vehicle box.

Reading a plate off a moving car in one frame is noise. `PlateTracker` — the
module as written, unmodified — buffers plate crops per vehicle, OCRs the
sharpest of them, and takes a confidence-weighted majority vote. The answer is
emitted when the vehicle leaves the frame, once.

Models: YOLO plate detector (1 class) on GPU 1; the Thai recognition network runs
on CPU through `paddle.inference`, as its authors intended — a plate arrives every
few seconds, not every frame.
"""
import base64
import collections
import io
import json
import os
import threading
import time

import cv2
import numpy as np
import requests
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

import ch
from plate_preprocess import preprocess_crop
from plate_tracker import PlateTracker
from upstream_pipeline import PlateOCR

GATEWAY = os.environ.get("GATEWAY_URL", "http://gateway:1984")
BRIDGE_WS = os.environ.get("BRIDGE_WS", "ws://bridge:8081/ws")
VLM_URL = os.environ.get("VLM_URL", "http://vllm:8000/v1")
VLM_MODEL = os.environ.get("VLM_MODEL", "gemma-4-31b-it")
SNAP_DIR = "/output/vehicles"
MODEL_DIR = "/app/models"

# The VLM is shown the vehicle crop and NOTHING else — no plate, no camera name.
# It answers in fixed fields so the page can filter on colour and body type.
VEHICLE_PROMPT = (
    "This is a CCTV crop of one vehicle. Answer with EXACTLY these four lines, "
    "nothing else:\n"
    "colour: <the body colour, one word: white|black|grey|silver|red|blue|green|"
    "yellow|orange|brown|unknown>\n"
    "body: <sedan|hatchback|suv|pickup|van|truck|bus|scooter|motorcycle|unknown>\n"
    "markings: <writing, a logo, a roof rack, a delivery box, damage — or none>\n"
    "summary: <one short clause describing the vehicle>\n\n"
    "Do not read or guess a licence plate. Do not invent a brand you cannot see. "
    "Say unknown rather than guess."
)

CFG = {
    "enabled": True,
    "cameras": [c for c in os.environ.get("PLATE_CAMERAS", "nvr1_ch03").split(",") if c],
    "vehicles": ["car", "motorcycle", "truck", "bus"],
    "frame_fps": 2.0,          # gateway grabs per camera per second
    "det_conf": 0.30,          # YOLO plate confidence
    "gone_s": 3.0,             # vehicle unseen this long -> read the plate and emit
    "min_vehicle_px": 80,      # a vehicle smaller than this carries no readable plate
    # A plate narrower than this is roughly seven pixels per character. At
    # nvr1_ch03 every plate measures ~34 px, so OCR was called 28,330 times and
    # never once produced a confident read. Skip it: the CPU is better spent, and
    # `plate_px` still records how wide the plate really was.
    # A vehicle worth describing: seen this many frames and this big at its
    # closest. Measured at ch03: 23 such vehicles an hour, out of 92 tracks.
    "describe": True,
    "buffer_size": 8,
    "top_k": 3,
    "ocr_confidence": 0.50,
    "min_plate_length": 4,
}

app = FastAPI()
_state = {"ready": False, "device": "?"}
_stats = {"frames": 0, "vehicles": 0, "plate_crops": 0, "emitted": 0,
          "no_plate": 0, "described": 0, "errors": 0,
          "last_error": ""}
_lock = threading.Lock()

_vehicles = {}      # track -> {"cls":str, "last":ts, "first_ms":int, "best":(conf,jpeg)}
_last_grab = {}
# YOLO on a 1920x1080 frame takes tens of milliseconds. Doing it inside the
# WebSocket receive loop made this a slow consumer and the bridge dropped us
# ("feed lost: Connection reset by peer") every few seconds. The socket thread
# now only parses; one worker does the vision.
_work = collections.deque(maxlen=4)
_work_evt = threading.Event()


def _load():
    ocr = PlateOCR(os.path.join(MODEL_DIR, "ocr"),
                   os.path.join(MODEL_DIR, "ocr", "thai_plate_dict.txt"))

    # Apache-2.0 RF-DETR plate detector on MIGraphX GPU (~5ms/frame).
    det = None
    dev = "cpu"
    onnx_path = os.path.join(MODEL_DIR, "rfdetr-large-plate.onnx")
    if os.path.exists(onnx_path):
        from plate_detector import MigraphxYOLO
        det = MigraphxYOLO(onnx_path, conf=CFG["det_conf"])
        dev = "migraphx" if det.on_gpu else "cpu (onnx)"
    if det is None:
        raise RuntimeError(f"Plate detector not found at {onnx_path}")

    def recognizer_fn(crops):
        out = []
        for c in crops:
            try:
                text, conf = ocr.predict(c)
            except Exception:
                text, conf = "", 0.0
            out.append((text, float(conf)))
        return out

    _state["device"] = dev
    _state["ready"] = True
    print(f"[plate] RF-DETR (Apache-2.0) on {dev}, Thai OCR on CPU, cameras={CFG['cameras']}",
          flush=True)
    return det, recognizer_fn


det_model, recognizer = _load()
tracker = PlateTracker(recognizer, buffer_size=CFG["buffer_size"],
                       top_k=CFG["top_k"], ocr_confidence=CFG["ocr_confidence"],
                       min_plate_length=CFG["min_plate_length"])


def _describe(crop_bgr):
    """Ask the local Gemma what the vehicle looks like. -> (colour, body, markings,
    summary) or None. It never sees the plate string, so it cannot echo one."""
    ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode()
    body = {"model": VLM_MODEL, "max_tokens": 120, "temperature": 0.0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": VEHICLE_PROMPT}]}]}
    r = requests.post(f"{VLM_URL.rstrip('/')}/chat/completions", json=body, timeout=60)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip(" -*").lower()
        v = v.strip(" .'\"").lower()
        if k in ("colour", "color", "body", "markings", "summary") and v:
            out["colour" if k == "color" else k] = v
    return out or None


def _grab(camera):
    r = requests.get(f"{GATEWAY}/api/frame.jpeg", params={"src": f"{camera}_main"},
                     timeout=8)
    r.raise_for_status()
    arr = np.frombuffer(r.content, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _plates_in(vehicle_crop):
    """(x1,y1,x2,y2,conf) of every plate the detector sees inside one vehicle."""
    return det_model.predict(vehicle_crop)


def _emit(track):
    """A vehicle left the frame.

    A visit is recorded only if the plate was actually READ. `PlateTracker` votes
    over every frame of the visit and keeps nothing below the module's own
    `ocr_confidence`, so this asks it for a verdict and believes it. Cars parked
    far down the drive are never read, are never described, and never reach the
    database — they are not what this camera watches.
    """
    v = _vehicles.pop(track, None)
    if not v:
        return
    plate = tracker.get_normalized(track)
    raw, raw_conf = tracker.get_raw(track)
    tracker.cleanup(track)

    if not plate:
        # The owner chose this camera because cars entering the building gate
        # give a plate he can read with his own eyes. A car parked far down the
        # drive does not, and is not what the camera is for. No plate, no row,
        # no VLM call.
        with _lock:
            _stats["no_plate"] += 1
        return

    snap = ""
    if v.get("veh") is not None:
        try:
            os.makedirs(SNAP_DIR, exist_ok=True)
            snap = f"{v['cam']}_{track}_{int(time.time()*1000)}.jpg"
            cv2.imwrite(os.path.join(SNAP_DIR, snap), v["veh"])
        except Exception:
            snap = ""

    tags = None
    if CFG["describe"] and v.get("veh") is not None:
        try:
            tags = _describe(v["veh"])
        except Exception as e:
            with _lock:
                _stats["errors"] += 1
                _stats["last_error"] = f"vlm: {str(e)[:140]}"
    tags = tags or {}
    summary = tags.get("summary") or ""
    if not summary:
        bits = [tags.get("colour"), tags.get("body")]
        summary = " ".join(b for b in bits if b and b != "unknown") or v["cls"]

    ch.vehicle(camera=v["cam"], track=track, cls=v["cls"], frames=v["frames"],
               colour=tags.get("colour", ""), body=tags.get("body", ""),
               markings=tags.get("markings", ""), description=summary,
               plate=plate, plate_conf=float(raw_conf or 0.0),
               plate_px=v["plate_px"], first_ms=v["first_ms"], snapshot=snap)
    ch.plate(camera=v["cam"], track=track, vehicle=v["cls"], plate=plate,
             confidence=float(raw_conf or 0.0), votes=1, reads=v["reads"],
             first_ms=v["first_ms"], snapshot=snap)
    with _lock:
        _stats["emitted"] += 1
        if tags:
            _stats["described"] += 1
    print(f"[plate] {v['cam']} {v['cls']} trk={track} frames={v['frames']} "
          f"plate={plate} (conf {raw_conf or 0:.2f}, {v['plate_px']}px) · {summary}",
          flush=True)


def _on_frame(camera, vehicles, now):
    if camera not in CFG["cameras"] or not vehicles:
        return
    period = 1.0 / max(CFG["frame_fps"], 0.01)
    if now - _last_grab.get(camera, 0.0) < period:
        return
    _last_grab[camera] = now
    try:
        frame = _grab(camera)
    except Exception as e:
        with _lock:
            _stats["errors"] += 1
            _stats["last_error"] = str(e)[:160]
        return
    if frame is None:
        return
    with _lock:
        _stats["frames"] += 1
    H, W = frame.shape[:2]

    for v in vehicles:
        x, y, w, h = v["box"]
        if max(w, h) < CFG["min_vehicle_px"]:
            continue
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(W, int(x + w)), min(H, int(y + h))
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        veh = frame[y1:y2, x1:x2]
        trk = v["track"]
        e = _vehicles.setdefault(trk, {"cam": camera, "cls": v["cls"],
                                       "first_ms": int(now * 1000), "reads": 0,
                                       "best": None, "frames": 0,
                                       "veh": None, "veh_px": 0, "plate_px": 0})
        e["last"] = now
        e["frames"] += 1
        # keep the biggest view of the vehicle itself — that is what a human, and
        # the VLM, needs to say what colour and shape it is
        if (x2 - x1) > e["veh_px"]:
            e["veh_px"] = x2 - x1
            e["veh"] = veh.copy()
        try:
            found = _plates_in(veh)
        except Exception as ex:
            with _lock:
                _stats["errors"] += 1
                _stats["last_error"] = str(ex)[:160]
            continue
        for (px1, py1, px2, py2, conf) in found:
            e["plate_px"] = max(e["plate_px"], px2 - px1)
            crop = preprocess_crop(veh, px1, py1, px2, py2)
            if crop is None or crop.size == 0:
                continue
            tracker.add_crop(trk, crop, conf)
            e["reads"] += 1
            if not e["best"] or conf > e["best"][0]:
                e["best"] = (conf, crop)
            with _lock:
                _stats["plate_crops"] += 1


def _vision_worker():
    while True:
        _work_evt.wait(1.0)
        _work_evt.clear()
        while _work:
            camera, vehicles, now = _work.popleft()
            try:
                _on_frame(camera, vehicles, now)
            except Exception as e:
                with _lock:
                    _stats["errors"] += 1
                    _stats["last_error"] = str(e)[:160]


def _sweeper():
    while True:
        time.sleep(1.0)
        now = time.time()
        for trk in [t for t, v in list(_vehicles.items())
                    if now - v.get("last", 0) > CFG["gone_s"]]:
            try:
                _emit(trk)
            except Exception as e:
                with _lock:
                    _stats["errors"] += 1
                    _stats["last_error"] = str(e)[:160]
                _vehicles.pop(trk, None)


def _feed():
    import websocket
    while True:
        try:
            ws = websocket.create_connection(BRIDGE_WS, timeout=30)
            print("[plate] connected to the bridge", flush=True)
            while True:
                msg = json.loads(ws.recv())
                if msg.get("type") != "detections" or not CFG["enabled"]:
                    continue
                by_cam = collections.defaultdict(list)
                for r in msg["records"]:
                    cam = r.get("camera_id")
                    if cam not in CFG["cameras"]:
                        continue
                    if r.get("class_name") not in CFG["vehicles"]:
                        continue
                    if r.get("track_id") is None:
                        continue
                    b = r.get("bounding_box") or {}
                    by_cam[cam].append({
                        "track": int(r["track_id"]), "cls": r["class_name"],
                        "box": (b.get("x", 0), b.get("y", 0),
                                b.get("w", 0), b.get("h", 0))})
                now = time.time()
                for cam, vs in by_cam.items():
                    with _lock:
                        _stats["vehicles"] = len(_vehicles)
                    _work.append((cam, vs, now))   # bounded: old frames are dropped
                    _work_evt.set()
        except Exception as e:
            print(f"[plate] feed lost: {str(e)[:120]}", flush=True)
            time.sleep(5)


@app.get("/healthz")
def healthz():
    return {"ready": _state["ready"], "device": _state["device"],
            "cfg": CFG, "stats": _stats, "tracking": len(_vehicles)}


class CfgReq(BaseModel):
    enabled: bool | None = None
    cameras: list[str] | None = None
    frame_fps: float | None = None
    det_conf: float | None = None
    gone_s: float | None = None
    min_vehicle_px: int | None = None


@app.post("/config")
def config(r: CfgReq):
    CFG.update({k: v for k, v in r.dict().items() if v is not None})
    return {"ok": True, "cfg": CFG}


def _camera_list():
    """The cameras this service is allowed to speak about, as a SQL literal."""
    return ", ".join("'%s'" % c.replace("'", "") for c in CFG["cameras"]) or "''"


@app.get("/vehicles")
def vehicles(hours: float = 24.0, limit: int = 100):
    where = ["ts > now() - INTERVAL {h:Float64} HOUR", "plate != ''",
             f"camera IN ({_camera_list()})"]
    try:
        rows = ch.query(
            "SELECT toUnixTimestamp64Milli(ts) AS ts_ms, "
            "toUnixTimestamp64Milli(first_ts) AS first_ms, camera, track, class, "
            "frames, colour, body, markings, description, plate, plate_conf, "
            "plate_px, snapshot FROM vehicles FINAL "
            f"WHERE {' AND '.join(where)} ORDER BY ts DESC LIMIT {int(limit)}",
            {"param_h": str(hours)})
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "vehicles": []}
    return {"ok": True, "vehicles": rows}


@app.get("/snapshot")
def snapshot(p: str):
    from fastapi.responses import FileResponse, JSONResponse
    full = os.path.normpath(os.path.join(SNAP_DIR, p))
    if not full.startswith(SNAP_DIR + "/") or not os.path.exists(full):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(full, media_type="image/jpeg")


@app.get("/live")
def live():
    """Plates of vehicles that are STILL IN FRAME.

    `_emit()` only fires when a vehicle leaves, three seconds after its last box.
    By then the live overlay has no box left to label, so a plate read from
    ClickHouse can never appear on the moving car that produced it. `PlateTracker`
    holds a running majority vote from the first confident frame onward, and that
    is what the overlay needs. History still comes from `plates`; this is the
    present tense of the same fact."""
    out = []
    with _lock:
        tracks = list(_vehicles.items())
    for trk, v in tracks:
        plate = tracker.get_normalized(trk)
        if not plate:
            continue
        _, conf = tracker.get_raw(trk)
        out.append({"camera": v["cam"], "track": int(trk), "vehicle": v["cls"],
                    "plate": plate, "confidence": float(conf or 0.0),
                    "reads": v["reads"]})
    return {"ok": True, "plates": out}


@app.get("/recent")
def recent(hours: float = 24.0, limit: int = 50):
    try:
        rows = ch.query(
            "SELECT toUnixTimestamp64Milli(ts) AS ts_ms, camera, track, vehicle, "
            "plate, confidence, reads, snapshot FROM plates "
            "WHERE ts > now() - INTERVAL {h:Float64} HOUR "
            # Only a configured camera is evidence. Rows a self-test wrote carry a
            # made-up camera name; they stay in the table but never reach a screen.
            f"AND camera IN ({_camera_list()}) "
            f"ORDER BY ts DESC LIMIT {int(limit)}", {"param_h": str(hours)})
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "plates": []}
    return {"ok": True, "plates": rows}


threading.Thread(target=_feed, daemon=True).start()
threading.Thread(target=_vision_worker, daemon=True).start()
threading.Thread(target=_sweeper, daemon=True).start()
