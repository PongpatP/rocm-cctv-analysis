"""Detection pipeline for AMD GPUs.

The service runs entirely on AMD parts:

    RTSP (gateway re-stream) -> ffmpeg decode -> RT-DETR on MIGraphX
                             -> IoU tracker -> POST /ingest on the bridge

Model choice is forced, not preferred. The obvious alternative is an AGPL-3.0
model this project bans (cannot ship in a commercial product), and MIGraphX 2.15
miscompiles it anyway (silently returns an all-zero tensor; `migraphx-driver
verify` reports FAILED). RT-DETR R50 is Apache-2.0 and verified bit-exact against
the CPU provider on this GPU.

The exported ONNX carries its own decode head (the DETR decode), so there is
one output:

    [batch, 300, 6]   x1, y1, x2, y2, score, class_id   in 640x640 pixels

No NMS: DETR's one-query-one-object training makes it redundant.

Structure mirrors a stream-batching muxer, and for the same reason. A first version read a
frame, inferred, and POSTed inline per camera. Each POST is a small ClickHouse
insert, and while it was in flight nobody drained ffmpeg's pipe — the RTSP
buffer overflowed, h264 threw `error while decoding MB`, and go2rtc dropped the
consumer after ~10 seconds. So: reader threads keep the sockets drained into a
one-slot-per-camera buffer (a late frame is worthless, drop it), and one worker
batches every camera into a single Run() and a single POST.

Decode is ffmpeg/CPU, not rocDecode. The VCN engines do 2133 fps aggregate over
29 streams (measured) against the ~348 fps this needs, so the GPU path is the
eventual answer — but ffmpeg keeps the first version debuggable, and 29 1080p
streams at 4 fps cost ~42% of a single core each (~1.5% × 28 ≈ 42% total),
well within 20 cores. Swap in pyRocVideoDecode once this is trusted.

Streams: main (1080p) downscaled to 640×640 by ffmpeg, not sub (352×288)
upscaled. The downscale gives far sharper input to the detector — small people
and distant vehicles that were invisible in the sub-stream mush now resolve.
"""
import argparse
import base64
import collections
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone, timedelta

import io

import numpy as np
import onnxruntime as ort
from PIL import Image

TZ = timezone(timedelta(hours=7))
SIZE = 640                       # RT-DETR is a fixed 640x640 graph

# RT-DETR is trained on COCO — it ALWAYS sees all 80 classes; this pipeline
# only filters. Which classes get through is the user's choice, edited live
# from the AI-settings page (output/ai_classes.json, re-read every 2 s like
# the ReID gate config). The old build hardcoded 5.
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
CLASSES_CFG = os.environ.get("CLASSES_CFG", "/output/ai_classes.json")


class ClassFilter:
    """Live class selection: id -> name for the enabled classes only."""

    def __init__(self):
        self.keep = {}
        self.interval = 0          # 0 = every frame; N = one infer per N+1
        self._read = 0.0
        self._mtime = None
        self._load(DEFAULT_CLASSES)

    def _load(self, names):
        wanted = {n for n in names if n in COCO80}
        self.keep = {i: n for i, n in enumerate(COCO80) if n in wanted}

    def refresh(self):
        now = time.monotonic()
        if now - self._read < 2.0:
            return
        self._read = now
        try:
            mt = os.path.getmtime(CLASSES_CFG)
            if mt == self._mtime:
                return
            self._mtime = mt
            with open(CLASSES_CFG) as f:
                data = json.load(f)
            sel = data.get("selected") if isinstance(data, dict) else data
            if isinstance(data, dict):
                try:
                    self.interval = max(0, int(data.get("interval", 0)))
                except (TypeError, ValueError):
                    self.interval = 0
            if isinstance(sel, list) and sel:
                self._load(sel)
                print(f"[classes] detecting {sorted(self.keep.values())} "
                      f"(interval={self.interval})", flush=True)
        except (OSError, ValueError):
            pass

# A 0.10 floor with the bridge applying the user's threshold cannot work here:
# DETR emits 300 queries with no NMS, so a 0.10 floor
# feeds the tracker a cloud of low-score ghosts that steal IoU matches from the
# real objects — 99 person detections produced 91 distinct track ids. Track on
# what the model is actually sure about.
CONF_FLOOR = float(os.environ.get("CONF_FLOOR", "0.30"))

MODEL = os.environ.get("RTDETR_ONNX", "/models/rtdetr_r50_uint8.onnx")
BRIDGE = os.environ.get("BRIDGE_URL", "http://bridge:8081/ingest")
GATEWAY_RTSP = os.environ.get("GATEWAY_RTSP", "rtsp://gateway:8554")

# The behaviour VLM reads the detection-synced frame from here instead of an
# out-of-sync gateway grab (empty "no person" vanish snapshots fell
# 47% -> 9% with this fix). Mounted from
# output/siglip/frames — the same dir siglip reads. Written atomically, at
# most FRAME_FPS per camera; a missing mount just disables the feature.
FRAMES_DIR = os.environ.get("FRAMES_DIR", "/frames")
FRAME_FPS = float(os.environ.get("FRAME_FPS", "2"))


class FrameSaver:
    """Best-effort JPEG writer for the behaviour watcher. A failed write must
    never touch the detection loop."""

    def __init__(self):
        self.period = 1.0 / max(FRAME_FPS, 0.1)
        self.last = {}                       # camera -> monotonic ts
        self.enabled = os.path.isdir(FRAMES_DIR)
        if not self.enabled:
            print(f"[frames] {FRAMES_DIR} not mounted — "
                  "behaviour falls back to gateway grabs", flush=True)

    def save(self, cam, frame):
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self.last.get(cam, 0.0) < self.period:
            return
        self.last[cam] = now
        try:
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, "JPEG", quality=85)
            tmp = os.path.join(FRAMES_DIR, f".{cam}.tmp.jpg")
            with open(tmp, "wb") as f:
                f.write(buf.getvalue())
            os.replace(tmp, os.path.join(FRAMES_DIR, f"{cam}.jpg"))
        except Exception:
            pass


# Cross-camera ReID: siglip's matcher is PUSH-fed — the detection pipeline
# must POST person crops to /embed (the detector does this).
# Without this feed there are no embeddings, no Global IDs, no sightings, and
# the Tracking page stays empty.
SIGLIP_URL = os.environ.get("SIGLIP_URL", "http://siglip:8085")


class SiglipSender:
    """Best-effort async sender of gated person crops to siglip /embed.
    A bounded queue drops on overflow,
    a background thread POSTs batches, all failures are swallowed — the
    service being down means no embeddings, never a stalled loop."""

    def __init__(self, maxq=256, batch=16):
        self.batch = batch
        self.q = queue.Queue(maxsize=maxq)
        self.dropped = 0
        self.sent = 0
        # "<camera>:<track>" -> Global ID, learned from the service's reply
        self.gids = {}
        threading.Thread(target=self._run, daemon=True).start()

    def gid(self, camera, track):
        return self.gids.get(f"{camera}:{track}")

    def submit(self, ev):
        try:
            self.q.put_nowait(ev)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while True:
            try:
                evs = [self.q.get(timeout=1.0)]
            except queue.Empty:
                continue
            while len(evs) < self.batch:
                try:
                    evs.append(self.q.get_nowait())
                except queue.Empty:
                    break
            try:
                req = urllib.request.Request(
                    SIGLIP_URL.rstrip("/") + "/embed",
                    data=json.dumps({"events": evs}).encode(),
                    headers={"Content-Type": "application/json"})
                raw = urllib.request.urlopen(req, timeout=5).read()
                self.sent += len(evs)
                assigned = (json.loads(raw) or {}).get("assigned") or {}
                if assigned:
                    self.gids.update(assigned)
                    if len(self.gids) > 4000:      # bound: drop the oldest half
                        for k in list(self.gids)[:2000]:
                            self.gids.pop(k, None)
            except Exception:
                pass


class ReidGate:
    """ReID gate, in 640x640 frame space: emit a crop for a NEW
    track in a fast burst (a 1-crop tracklet is what creates duplicate IDs),
    then on period + movement. Overlapping person boxes are dropped whole —
    a mixed crop must never enter an identity bank.

    Tunables live in output/ai_siglip.json, the file the AI-settings page
    edits through siglip /config — re-read every 2 s,
    so gate changes from the web UI apply live with no restart.
    CAUTION: min_box_w/h are in bounding-box pixels; ours are 640x640-stretched
    (a 1080p person is ~3x narrower, ~1.7x shorter here)."""

    CFG_PATH = os.environ.get("GATE_CFG", "/output/ai_siglip.json")
    DEFAULTS = {"enabled": True, "min_conf": 0.4, "move_thresh": 0.04,
                "period_s": 3.0, "iou_thresh": 0.2, "min_shots": 4,
                "burst_period_s": 0.12, "min_box_w": 13, "min_box_h": 53}

    def __init__(self, sender):
        self.sender = sender
        self.seen = {}            # (cam, track) -> (cx, cy, mono, shots)
        self.cfg = dict(self.DEFAULTS)
        self._cfg_read = 0.0
        self._cfg_mtime = None

    def _refresh_cfg(self):
        now = time.monotonic()
        if now - self._cfg_read < 2.0:
            return
        self._cfg_read = now
        try:
            mt = os.path.getmtime(self.CFG_PATH)
            if mt == self._cfg_mtime:
                return
            self._cfg_mtime = mt
            with open(self.CFG_PATH) as f:
                data = json.load(f)
            cfg = dict(self.DEFAULTS)
            if isinstance(data, dict):
                cfg.update({k: data[k] for k in self.DEFAULTS if k in data})
            if cfg != self.cfg:
                self.cfg = cfg
                print(f"[reid] gate cfg: {cfg} (sent={self.sender.sent} "
                      f"dropped={self.sender.dropped})", flush=True)
        except (OSError, ValueError):
            pass                  # missing/broken file -> keep current cfg

    @staticmethod
    def _crop_jpeg(frame, box, margin=0.12):
        x1, y1, x2, y2 = box
        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
        L = max(0, int(x1 - mx)); T = max(0, int(y1 - my))
        R = min(SIZE, int(x2 + mx)); B = min(SIZE, int(y2 + my))
        if R - L < 8 or B - T < 16:
            return None
        buf = io.BytesIO()
        Image.fromarray(frame[T:B, L:R]).save(buf, "JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()

    def feed(self, cam, frame, persons, ts_ms):
        """persons: [(box(x1,y1,x2,y2), track_id, conf)] for ONE camera."""
        self._refresh_cfg()
        cfg = self.cfg
        if not cfg["enabled"]:
            return
        # occlusion reject (§2.3): boxes that overlap another person yield
        # mixed crops — skip them this frame, wait for a separated view
        occ = set()
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                if IoUTracker._iou(persons[i][0],
                                   persons[j][0]) > float(cfg["iou_thresh"]):
                    occ.add(i); occ.add(j)
        now = time.monotonic()
        for i, (box, track, conf) in enumerate(persons):
            if i in occ or track < 0 or conf < float(cfg["min_conf"]):
                continue
            x1, y1, x2, y2 = box
            if (x2 - x1 < float(cfg["min_box_w"])
                    or y2 - y1 < float(cfg["min_box_h"])):
                continue                  # too small to carry an identity
            cx, cy = (x1 + x2) / 2 / SIZE, (y1 + y2) / 2 / SIZE
            key = (cam, track)
            prev = self.seen.get(key)
            if prev is None:
                fire, shots = True, 0
            else:
                shots = prev[3]
                if shots < int(cfg["min_shots"]):     # burst: period only
                    fire = now - prev[2] >= float(cfg["burst_period_s"])
                else:                                  # steady: period + move
                    fire = (now - prev[2] >= float(cfg["period_s"])
                            and math.hypot(cx - prev[0],
                                           cy - prev[1]) >= float(cfg["move_thresh"]))
            if not fire:
                continue
            self.seen[key] = (cx, cy, now, shots + 1)
            img = self._crop_jpeg(frame, box)
            if img is None:
                continue
            self.sender.submit({
                "camera": cam, "track": int(track), "ts": ts_ms,
                "conf": float(conf),
                "nx": round(x1 / SIZE, 5), "ny": round(y1 / SIZE, 5),
                "nw": round((x2 - x1) / SIZE, 5), "nh": round((y2 - y1) / SIZE, 5),
                "img": img,
            })
        if len(self.seen) > 4000:         # bound memory
            cut = now - 120
            self.seen = {k: v for k, v in self.seen.items() if v[2] > cut}


def build_session():
    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = [
        ("MIGraphXExecutionProvider", {
            "migraphx_fp16_enable": True,
        }),
        "CPUExecutionProvider",
    ]
    s = ort.InferenceSession(MODEL, so, providers=providers)
    if "MIGraphXExecutionProvider" not in s.get_providers():
        sys.exit("MIGraphX provider unavailable — refusing to run on CPU")
    return s


class Track:
    __slots__ = ("id", "box", "cls", "miss")

    def __init__(self, tid, box, cls):
        self.id, self.box, self.cls, self.miss = tid, box, cls, 0


class _KalmanBox:
    """Constant-velocity Kalman filter on [cx, cy, w, h] — the BoT-SORT state
    (xywh, not ByteTrack's aspect-ratio state). Numpy only.

    Clean-room implementation of the published algorithm (the original
    BoT-SORT repo is MIT); written to REPLACE the ultralytics import, whose
    AGPL-3.0 license this project bans (same ruling as the other AGPL-3.0
    models this project excludes)."""

    SP, SV = 1 / 20, 1 / 160          # std weights: position, velocity

    def __init__(self, cx, cy, w, h):
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], np.float64)
        sp, sv = self.SP, self.SV
        std = [2 * sp * w, 2 * sp * h, 2 * sp * w, 2 * sp * h,
               10 * sv * w, 10 * sv * h, 10 * sv * w, 10 * sv * h]
        self.P = np.diag(np.square(std))
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def predict(self, damp=False):
        if damp:                       # lost track: velocity decays toward 0 so
            self.x[4:] *= 0.8          # the prediction stays near the last seen
        w, h = max(self.x[2], 1.0), max(self.x[3], 1.0)
        sp, sv = self.SP, self.SV
        q = [sp * w, sp * h, sp * w, sp * h, sv * w, sv * h, sv * w, sv * h]
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + np.diag(np.square(q))

    def correct(self, cx, cy, w, h):
        sp = self.SP
        mw, mh = max(self.x[2], 1.0), max(self.x[3], 1.0)
        R = np.diag(np.square([sp * mw, sp * mh, sp * mw, sp * mh]))
        z = np.array([cx, cy, w, h], np.float64)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def tlbr(self):
        cx, cy, w, h = self.x[:4]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def tlbr_forward(self, steps):
        """Predict box position `steps` frames into the future using velocity.
        Used for latency compensation: the UI receives the bbox ~2 frames after
        the detection was computed, so we extrapolate where the object WILL be."""
        cx, cy, w, h = self.x[:4]
        vcx, vcy, vw, vh = self.x[4:]
        cx += vcx * steps
        cy += vcy * steps
        w = max(w + vw * steps, 1.0)
        h = max(h + vh * steps, 1.0)
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class _Track:
    __slots__ = ("id", "kf", "cls", "score", "lost", "hits", "feat", "feat_age")

    def __init__(self, tid, box, cls, score):
        x1, y1, x2, y2 = box
        self.kf = _KalmanBox((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        self.id, self.cls, self.score = tid, cls, score
        self.lost, self.hits = 0, 1
        self.feat, self.feat_age = None, 0     # EMA appearance (osnet, L2)


class _OsNet:
    """OSNet-x0.25 appearance embedder for WITHIN-camera identity. Fixed
    batch-16 ONNX — pad and slice; the single concrete shape is exactly what
    MIGraphX wants (one compile, then a few ms on the idle GPU headroom).
    Fails soft: no model / no provider -> tracker runs IoU-only as before."""

    PATH = os.environ.get("REID_MODEL", "/reid_models/osnet_x0_25_msmt17.onnx")
    _MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
    _STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)

    def __init__(self):
        self.sess = None
        if not os.path.exists(self.PATH):
            print(f"[reid-app] {self.PATH} missing — appearance OFF", flush=True)
            return
        try:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            self.sess = ort.InferenceSession(self.PATH, so, providers=[
                "MIGraphXExecutionProvider", "CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
            self.sess.run(None, {self.inp: np.zeros((16, 3, 256, 128),
                                                    np.float32)})  # compile now
            print(f"[reid-app] OSNet ready on "
                  f"{self.sess.get_providers()[0]}", flush=True)
        except Exception as e:
            self.sess = None
            print(f"[reid-app] load failed — appearance OFF: {str(e)[:120]}",
                  flush=True)

    def embed(self, frame, boxes):
        """boxes: [(x1,y1,x2,y2)] on the 640x640 frame -> L2-normed (N,512),
        or None when the embedder is unavailable."""
        if self.sess is None or not boxes:
            return None
        crops = []
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(SIZE, int(x2)), min(SIZE, int(y2))
            if x2 - x1 < 8 or y2 - y1 < 16:
                crops.append(np.zeros((3, 256, 128), np.float32))
                continue
            im = Image.fromarray(frame[y1:y2, x1:x2]).resize(
                (128, 256), Image.BILINEAR)
            a = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0
            crops.append((a - self._MEAN) / self._STD)
        feats = []
        for i in range(0, len(crops), 16):
            chunk = crops[i:i + 16]
            batch = np.stack(chunk + [np.zeros((3, 256, 128), np.float32)]
                             * (16 - len(chunk)))
            y = self.sess.run(None, {self.inp: batch})[0][:len(chunk)]
            feats.append(y)
        f = np.concatenate(feats).astype(np.float32)
        n = np.linalg.norm(f, axis=1, keepdims=True)
        return f / np.maximum(n, 1e-6)


class BotSortTracker:
    """BoT-SORT-style tracker, dependency-free: Kalman prediction (xywh) +
    ByteTrack two-stage association + a lost-track buffer that re-finds people
    after occlusion. Exactly the subset the previous ultralytics import ran
    (its ReID term and camera-motion compensation were both OFF), with the
    same tuning — but no AGPL package in the product.

    TODO (unchanged from before): a dynamic-batch osnet export would allow an
    appearance term in stage 1; cross-camera identity stays in siglip."""

    TRACK_HIGH = 0.2       # dets >= this join the first association stage
    TRACK_LOW = 0.05       # dets in [low, high) may rescue an active track
    NEW_TRACK = 0.65       # unmatched dets >= this start a new track
    MATCH_FUSED = 0.5      # stage 1: accept iou * det_score >= this
    MATCH_IOU2 = 0.5       # stage 2: accept plain iou >= this
    BUFFER = 100           # frames a lost track is kept (~12 s at 8 fps)

    OSNET = None                       # class-level: one embedder, all cameras
    AMBIG_MARGIN = 0.25    # top-2 IoU candidates closer than this = ambiguous
    APPEAR_W = 0.5         # fused cost: iou*(1-w) + cosine*w on ambiguous pairs
    REFRESH_FRAMES = 8     # EMA feature refresh cadence (~1/s at 8 fps)

    def __init__(self):
        self.active, self.lost_tracks, self.next_id = [], [], 1
        self.tracks = self.active          # report-line compatibility
        if BotSortTracker.OSNET is None:
            BotSortTracker.OSNET = _OsNet()
        appear = "ON (osnet, ambiguity-gated)" if self.OSNET.sess else \
            "OFF (Kalman+IoU)"
        print(f"[tracker] BoT-SORT (clean-room MIT-math, no ultralytics), "
              f"reid={appear}", flush=True)

    @staticmethod
    def _iou(a, b):
        """Pairwise IoU on (x1,y1,x2,y2) — kept because ReidGate's occlusion
        check calls IoUTracker._iou."""
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0:
            return 0.0
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    @staticmethod
    def _iou_matrix(tracks, boxes):
        if not tracks or not boxes:
            return np.zeros((len(tracks), len(boxes)))
        t = np.array([tr.kf.tlbr() for tr in tracks])
        d = np.array(boxes, np.float64)
        x1 = np.maximum(t[:, None, 0], d[None, :, 0])
        y1 = np.maximum(t[:, None, 1], d[None, :, 1])
        x2 = np.minimum(t[:, None, 2], d[None, :, 2])
        y2 = np.minimum(t[:, None, 3], d[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        at = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
        ad = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
        union = at[:, None] + ad[None, :] - inter
        return np.where(union > 0, inter / union, 0.0)

    @staticmethod
    def _greedy(cost, thresh):
        """Highest-value pairs first, no conflicts, floor at `thresh`."""
        pairs = []
        if cost.size:
            order = np.dstack(np.unravel_index(
                np.argsort(cost, axis=None)[::-1], cost.shape))[0]
            used_t, used_d = set(), set()
            for ti, di in order:
                if cost[ti, di] < thresh:
                    break
                if ti in used_t or di in used_d:
                    continue
                used_t.add(ti); used_d.add(di)
                pairs.append((int(ti), int(di)))
        return pairs

    def _assign(self, tracks, det_idx, dets, fused, out, img=None):
        """Match `tracks` against dets[det_idx]; returns unmatched of each."""
        boxes = [dets[i][0] for i in det_idx]
        iou = self._iou_matrix(tracks, boxes)
        # never merge different classes into one identity
        for r, tr in enumerate(tracks):
            for c, i in enumerate(det_idx):
                if dets[i][1] != tr.cls:
                    iou[r, c] = 0.0
        if fused:
            score = np.array([dets[i][2] for i in det_idx])
            cost, floor = iou * score[None, :], self.MATCH_FUSED
        else:
            cost, floor = iou, self.MATCH_IOU2
        # ---- appearance, only where IoU alone could swap identities ----
        # (two people crossing = a det whose best and 2nd-best tracks are
        # nearly tied, or a lost track being recovered). This is the fix for
        # "IoU can swap IDs when two persons pass by".
        if (fused and img is not None and self.OSNET.sess is not None
                and cost.shape[0] > 0 and cost.shape[1] > 0):
            iou_any = iou.max(axis=0)
            top2 = np.sort(cost, axis=0)[::-1] if cost.shape[0] > 1 else None
            need = []
            for c in range(cost.shape[1]):
                tied = (top2 is not None and top2[0, c] >= floor
                        and top2[0, c] - top2[1, c] < self.AMBIG_MARGIN)
                weak = floor * 0.1 <= iou_any[c] < floor   # crossing/recovery
                if tied or weak:
                    need.append(c)
            cand_t = [r for r, tr in enumerate(tracks)
                      if tr.feat is not None
                      and any(iou[r, c] >= floor * 0.1 for c in need)]
            if need and cand_t:
                feats = self.OSNET.embed(img, [boxes[c] for c in need])
                if feats is not None:
                    tf = np.stack([tracks[r].feat for r in cand_t])
                    cos = tf @ feats.T                    # (T, N) in [-1, 1]
                    for ai, c in enumerate(need):
                        for ti, r in enumerate(cand_t):
                            if iou[r, c] < floor * 0.1:
                                continue                  # too far: no rescue
                            blend = ((1 - self.APPEAR_W) * cost[r, c]
                                     + self.APPEAR_W * max(cos[ti, ai], 0.0))
                            # a confident appearance match may also LIFT a
                            # weak-IoU pair over the floor (the swap fix)
                            if cos[ti, ai] >= 0.45:
                                blend = max(blend, floor + 0.05 * cos[ti, ai])
                            cost[r, c] = blend
        pairs = self._greedy(cost, floor)
        mt, md = set(), set()
        for ti, ci in pairs:
            tr, i = tracks[ti], det_idx[ci]
            box = dets[i][0]
            tr.kf.correct((box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
                          box[2] - box[0], box[3] - box[1])
            tr.score, tr.lost, tr.hits = dets[i][2], 0, tr.hits + 1
            out[i] = tr.id
            mt.add(ti); md.add(ci)
        return ([t for k, t in enumerate(tracks) if k not in mt],
                [i for k, i in enumerate(det_idx) if k not in md])

    def update(self, dets, img=None):
        """dets: [(box(x1,y1,x2,y2), cls, score)] -> {det_index: track_id}"""
        for tr in self.active + self.lost_tracks:
            tr.kf.predict(damp=tr.lost > 0)
        out = {}
        high = [i for i, d in enumerate(dets) if d[2] >= self.TRACK_HIGH]
        low = [i for i, d in enumerate(dets)
               if self.TRACK_LOW <= d[2] < self.TRACK_HIGH]

        # stage 1: high dets vs active + lost (lost re-found here)
        pool = self.active + self.lost_tracks
        un_tracks, un_high = self._assign(pool, high, dets, True, out, img)
        # stage 2: low dets rescue still-unmatched ACTIVE tracks
        un_active = [t for t in un_tracks if t in self.active]
        un_active, _ = self._assign(un_active, low, dets, False, out)

        # ---- EMA feature refresh: clean, separated crops only (a crop taken
        # mid-crossing would poison BOTH identities' features) ----
        if img is not None and self.OSNET.sess is not None:
            by_id = {t.id: t for t in pool}
            fresh = []
            det_boxes = [d[0] for d in dets]
            for i, tid in out.items():
                tr = by_id.get(tid)
                if tr is None:
                    continue
                tr.feat_age += 1
                if tr.feat is not None and tr.feat_age < self.REFRESH_FRAMES:
                    continue
                b = dets[i][0]
                if (b[3] - b[1]) < 50:                  # too small to describe
                    continue
                if any(self._iou(b, ob) > 0.25
                       for j, ob in enumerate(det_boxes) if j != i):
                    continue                             # overlapping: skip
                fresh.append((tr, b))
            if fresh:
                feats = self.OSNET.embed(img, [b for _, b in fresh])
                if feats is not None:
                    for (tr, _), f in zip(fresh, feats):
                        tr.feat = f if tr.feat is None else \
                            0.9 * tr.feat + 0.1 * f
                        tr.feat = tr.feat / max(np.linalg.norm(tr.feat), 1e-6)
                        tr.feat_age = 0

        matched_ids = set(out.values())
        survivors, still_lost = [], []
        for tr in pool:
            if tr.id in matched_ids:
                survivors.append(tr)
            else:
                tr.lost += 1
                if tr.lost <= self.BUFFER:
                    still_lost.append(tr)
        for i in un_high:                       # births
            if dets[i][2] >= self.NEW_TRACK:
                tr = _Track(self.next_id, dets[i][0], dets[i][1], dets[i][2])
                self.next_id += 1
                survivors.append(tr)
                out[i] = tr.id
        self.active = survivors
        self.lost_tracks = still_lost
        self.tracks = self.active
        return out

    def predicted_box(self, track_id, steps=2):
        """Return the latency-compensated (x1,y1,x2,y2) for a track, predicted
        `steps` frames into the future. Returns None if track not found."""
        for tr in self.active:
            if tr.id == track_id:
                return tr.kf.tlbr_forward(steps)
        return None


# same algorithm, zero deps — the alias keeps older call sites working
IoUTracker = BotSortTracker


class CameraReader(threading.Thread):
    """Keeps one RTSP socket drained. Holds only the newest frame: if the
    worker is busy, an older frame has no value and dropping it is what stops
    the backpressure that kills the stream."""

    def __init__(self, cam, fps):
        super().__init__(daemon=True)
        self.cam, self.fps = cam, fps
        self.url = f"{GATEWAY_RTSP}/{cam}_main"
        self.lock = threading.Lock()
        self.frame = None       # unconsumed frame, or None
        self.last = None        # most recent frame ever seen, for batch padding
        self.frames = 0
        self.drops = 0

    def _spawn(self):
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "500000",
            "-probesize", "32768",
            "-rtsp_transport", "tcp", "-i", self.url,
            "-vf", f"fps={self.fps},scale={SIZE}:{SIZE}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        err = collections.deque(maxlen=4)
        threading.Thread(
            target=lambda: [err.append(l.decode(errors="replace").strip())
                            for l in iter(p.stderr.readline, b"")],
            daemon=True).start()
        return p, err

    def run(self):
        n = SIZE * SIZE * 3
        while True:
            p, err = self._spawn()
            try:
                while True:
                    buf = p.stdout.read(n)
                    if len(buf) < n:
                        break
                    with self.lock:
                        if self.frame is not None:
                            self.drops += 1
                        self.frame = self.last = np.frombuffer(
                            buf, np.uint8).reshape(SIZE, SIZE, 3)
                    self.frames += 1
            except Exception as e:
                print(f"[{self.cam}] reader {type(e).__name__}: {e}", flush=True)
            finally:
                p.kill()
            print(f"[{self.cam}] ffmpeg ended: {' | '.join(err) or 'clean eof'}"
                  f" — reconnecting", flush=True)
            time.sleep(3)

    def take(self):
        """(frame, is_fresh). Never None once the stream has produced one
        frame: the batch shape must stay constant (see main)."""
        with self.lock:
            if self.frame is not None:
                f, self.frame = self.frame, None
                return f, True
            return self.last, False


class AsyncPoster:
    """Fire-and-forget HTTP POST on a background thread. The inference loop
    never blocks on the bridge. One thread is enough: if the bridge is slower
    than inference, the queue backs up and we drop the oldest batch (the UI
    only cares about the freshest detections anyway)."""

    def __init__(self, url, maxq=4):
        self.url = url
        self._q = collections.deque(maxlen=maxq)
        self._lock = threading.Lock()
        self._ev = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def send(self, records):
        if not records:
            return
        with self._lock:
            self._q.append(records)
        self._ev.set()

    def _run(self):
        while True:
            self._ev.wait()
            self._ev.clear()
            while True:
                with self._lock:
                    if not self._q:
                        break
                    records = self._q.popleft()
                req = urllib.request.Request(
                    self.url, data=json.dumps(records).encode(),
                    headers={"Content-Type": "application/json"},
                )
                try:
                    urllib.request.urlopen(req, timeout=5).read()
                except Exception as e:
                    print(f"[ingest] {type(e).__name__}: {e}", flush=True)


def load_cameras(spec):
    """`all` reads the same generated camera list, so one
    config.local.yaml still drives every camera in the stack."""
    if spec != ["all"]:
        return spec
    import yaml
    cfg = yaml.safe_load(open(os.environ.get(
        "CAMERAS_YAML", "/configs/cameras.yaml")))
    return [c["id"] for c in cfg["cameras"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", nargs="+", required=True)
    ap.add_argument("--fps", type=float, default=4.0)
    args = ap.parse_args()
    args.cameras = load_cameras(args.cameras)

    sess = build_session()
    inp = sess.get_inputs()[0].name
    print(f"[airocm] {sess.get_providers()[0]}, {len(args.cameras)} cameras, "
          f"conf>={CONF_FLOOR}", flush=True)

    readers = {c: CameraReader(c, args.fps) for c in args.cameras}
    for r in readers.values():
        r.start()
    trackers = {c: IoUTracker() for c in args.cameras}
    saver = FrameSaver()
    poster = AsyncPoster(BRIDGE)
    reid_gate = ReidGate(SiglipSender())
    classes = ClassFilter()

    # The ONNX has a dynamic batch axis and MIGraphX compiles per concrete
    # shape — a batch that grew and shrank with camera availability recompiled
    # every iteration and cost 65-95 SECONDS per batch. The batch is therefore
    # always len(cameras): a camera with no new frame repeats its last one and
    # its result is discarded. One compile, then ~30 ms.
    # Preprocessing (/255 + NHWC->NCHW) lives INSIDE the ONNX graph
    # (rtdetr_r50_uint8.onnx) — it cost 149 ms/batch in numpy, more than the
    # inference itself, and pinned the loop to 3.9 fps/cam at 39% GPU use.
    # The graph input is the raw uint8 NHWC frame exactly as ffmpeg pipes it.
    order = list(readers)
    warm = np.zeros((len(order), SIZE, SIZE, 3), np.uint8)
    t0 = time.time()
    sess.run(None, {inp: warm})
    print(f"[airocm] compiled batch={len(order)} in {time.time()-t0:.0f}s",
          flush=True)

    period = 1.0 / args.fps
    infer_ms, batches, last_report = 0.0, 0, time.time()
    frame_i = 0
    frame_i = 0

    # Double-buffer: we pre-allocate two batch arrays and alternate between
    # them. While the GPU infers on buf_a, we fill buf_b from the readers.
    buf_a = np.zeros((len(order), SIZE, SIZE, 3), np.uint8)
    buf_b = np.zeros((len(order), SIZE, SIZE, 3), np.uint8)

    while True:
        t0 = time.time()

        # --- Collect frames into buf_a ---
        taken = {c: readers[c].take() for c in order}
        if not any(fresh for _, fresh in taken.values()):
            time.sleep(period / 4)
            continue
        if any(f is None for f, _ in taken.values()):
            time.sleep(period / 4)      # a camera has not produced anything yet
            continue

        classes.refresh()
        # frame interval: 0 = infer every frame; N = infer 1 of every (N+1).
        # skipped frames are still read (readers drain in their own threads)
        # so the RTSP sockets never back up — we just don't run the GPU.
        frame_i += 1
        if classes.interval and (frame_i % (classes.interval + 1)):
            slack = period - (time.time() - t0)
            if slack > 0:
                time.sleep(slack)
            continue

        for i, c in enumerate(order):
            buf_a[i] = taken[c][0]

        # --- Inference (GPU) ---
        t1 = time.time()
        outs = sess.run(None, {inp: buf_a})[0]       # [N, 300, 6]
        infer_ms += (time.time() - t1) * 1000
        batches += 1

        # --- Post-process + async POST (overlaps with next collection) ---
        ts = datetime.now(TZ).isoformat(timespec="milliseconds")
        records = []
        for cam, out in zip(order, outs):
            if not taken[cam][1]:        # stale padding frame — do not re-report
                continue
            saver.save(cam, taken[cam][0])
            dets = [((float(a), float(b), float(c), float(d)), int(k), float(s))
                    for a, b, c, d, s, k in out
                    if s >= CONF_FLOOR and int(k) in classes.keep]
            ids = trackers[cam].update(dets, taken[cam][0])
            persons = []
            for i, (box, cid, score) in enumerate(dets):
                x1, y1, x2, y2 = box
                tid = ids.get(i, -1)
                # Latency compensation: the UI displays this bbox ~200ms after
                # it was computed. Use the Kalman velocity to predict where the
                # object will be by the time the user sees it (2 frames at 8fps).
                pbox = trackers[cam].predicted_box(tid, steps=2) if tid > 0 else None
                if pbox is not None:
                    px1, py1, px2, py2 = pbox
                    # Clip to frame bounds
                    px1 = max(0, min(SIZE, px1))
                    py1 = max(0, min(SIZE, py1))
                    px2 = max(0, min(SIZE, px2))
                    py2 = max(0, min(SIZE, py2))
                else:
                    px1, py1, px2, py2 = x1, y1, x2, y2
                rec = {
                    "camera_id": cam, "timestamp": ts,
                    "class_name": classes.keep[cid], "confidence": round(score, 4),
                    "bounding_box": {"x": round(px1, 1), "y": round(py1, 1),
                                     "w": round(px2 - px1, 1), "h": round(py2 - py1, 1)},
                    "frame": {"width": SIZE, "height": SIZE,
                              "number": readers[cam].frames},
                    "track_id": tid,
                }
                if cid == 0:                            # person
                    persons.append((box, rec["track_id"], score))
                    # stamp the identity learned from earlier /embed replies so
                    # the UI + behaviour see a Global ID, not a tracker id
                    g = reid_gate.sender.gid(cam, rec["track_id"])
                    if g is not None:
                        rec["global_id"] = g
                records.append(rec)
            reid_gate.feed(cam, taken[cam][0], persons,
                           int(time.time() * 1000))
        poster.send(records)                    # non-blocking

        # Swap buffers so next iteration fills the other one
        buf_a, buf_b = buf_b, buf_a

        if time.time() - last_report >= 15:
            fr = "  ".join(f"{c}: {r.frames}f/{r.drops}drop/"
                           f"{len(trackers[c].tracks)}trk"
                           for c, r in sorted(readers.items()))
            print(f"[airocm] {fr}  | {infer_ms / max(batches,1):.1f} ms/batch, "
                  f"{len(records)} recs last", flush=True)
            infer_ms, batches, last_report = 0.0, 0, time.time()

        slack = period - (time.time() - t0)
        if slack > 0:
            time.sleep(slack)


if __name__ == "__main__":
    main()
