"""Ask the VLM what a person is DOING — but only when something is worth asking.

The detector already answers "is there a person". This module answers "what are
they up to", and it is deliberately stingy: a VLM call costs ~1.1 s of GPU 0, so
a rule must fire before a frame is ever sent.

The rules, all owner-defined:

  crowd        three or more people in one frame at once
  dwell        a tracked person whose box has not moved for `dwell_s`
  vanish       a track that existed for `vanish_min_s` and then disappeared —
               the frame examined is the LAST ONE THAT STILL CONTAINED THE BOX,
               because by the time we know they are gone they are not in the
               picture any more. Hence the frame ring buffer below.
  parcel_sweep every `parcel_period_s`, look at each parcel-room camera
  parcel_new   a track id never seen before appears on a parcel-room camera

Everything is read from the bridge's WebSocket, which already carries
bounding_box, track_id, global_id and keypoints for every detection. Nothing is
added to the detection pipeline.

The verdict is stored in ClickHouse against the LOCAL track id, so it joins back
to the cross-camera identity through `sightings` even when the matcher had not
yet assigned a Global ID at that instant.
"""
import collections
import io
import json
import os
import threading
import time

import requests
from PIL import Image

import ch
import providers

GATEWAY = "http://gateway:1984"
BRIDGE_WS = "ws://bridge:8081/ws"
CFG_PATH = "/output/ai_behavior.json"
SNAP_DIR = "/output/siglip/behavior"     # the exact frame the VLM was shown
# the detector writes the DETECTION-SYNCED full frame it already cut the person crops from
# here, as <cam>.jpg. Reading it (instead of grabbing the gateway, which lags the
# detection) is what stops the VLM from seeing an empty frame after the person left.
FRAMES_DIR = "/output/siglip/frames"
FRAME_MAX_AGE_S = 3.0                     # older than this -> fall back to the gateway

DEFAULTS = {
    "enabled": True,
    "provider": "local-gemma",       # the fast one; 1.1 s per sub frame
    "stream": "sub",                 # 352x288 — the owner's choice, whole frame
    "max_calls_per_min": 180,        # measured ceiling is 212/min at conc 4
    "concurrency": 4,                # matches vLLM's --max-num-seqs
    "cooldown_s": 30,                # per (camera, track, rule)

    "crowd_enabled": True,
    "crowd_min_persons": 3,          # "more than two"

    "dwell_enabled": True,
    "dwell_s": 8.0,                  # standing still this long -> what are they doing
    "dwell_move_frac": 0.05,         # movement < this fraction of box height = still

    "vanish_enabled": True,
    "vanish_min_s": 3.0,             # must have existed this long to count
    "vanish_gap_s": 2.0,             # unseen this long -> gone
    # A person whose last box touched the frame border simply WALKED OUT of the
    # camera's view — that is not a disappearance. The interesting case is a
    # track that ends in the middle of the picture: a door, a lift, a room.
    "vanish_edge_frac": 0.06,

    "parcel_cameras": [],            # e.g. ["nvr1_ch10"]
    "parcel_period_s": 60.0,
    "parcel_new_id": True,

    "frame_buffer_s": 6.0,           # ring buffer depth, for the vanish rule
    "frame_fps": 1.0,                # gateway grabs per camera per second
    "keep_snapshots": True,          # evidence: a verdict without its frame is unauditable
}

# No verbatim example answers — the model copied them ("nobody is doing anything
# unusual" came back on nearly every frame). Describe the task and let it look.
# The description must be USEFUL, not a single verb: "sitting" tells an operator
# nothing; "a person sitting with an empty plate beside them" is worth keeping.
PROMPT = (
    "This is a frame from a fixed indoor security camera.\n\n"
    "Line 1: describe the scene in ONE informative sentence. Include: how many "
    "people, what each is doing, and any object they are clearly holding or "
    "interacting with (a bag, a parcel, a plate, a phone, a door, a chair). You "
    "may add ONE short, obvious inference that is grounded in a visible object "
    "(an empty plate implies a finished meal; a held parcel implies a delivery). "
    "Do NOT invent objects you cannot see, and do NOT guess the room's purpose or "
    "anyone's job or name. If nobody is visible, write exactly: no person.\n"
    "Line 2: exactly `suspicious: yes` or `suspicious: no`.\n\n"
    "Mark suspicious ONLY for: handling parcels or bags that are not theirs, "
    "loitering beside parcels, forcing a door, or concealing an object. Standing, "
    "waiting, walking, sitting and talking are normal."
)


def _parse(text):
    """-> (activity, suspicious). The second line is a fixed contract."""
    activity, susp = "", 0
    for line in (text or "").strip().splitlines():
        low = line.strip().lower()
        if low.startswith("suspicious:"):
            susp = 1 if "yes" in low else 0
        elif line.strip() and not activity:
            activity = line.strip()[:300]
    return activity, susp


class Behavior:
    def __init__(self):
        self.cfg = dict(DEFAULTS)
        self._load()
        self.lock = threading.Lock()
        # camera -> deque[(ts, jpeg_bytes)]  — only cameras that have people
        self.frames = collections.defaultdict(collections.deque)
        # (camera, track) -> {"first": ts, "last": ts, "box": (x,y,w,h),
        #                     "still_since": ts, "gid": int}
        self.tracks = {}
        self.seen_tracks = set()          # for parcel_new
        self.cooldown = {}                # (camera, track, rule) -> ts
        self.calls = collections.deque()  # timestamps of VLM calls, for the budget
        self.last_grab = {}               # camera -> ts of the last gateway fetch
        self.last_saved = {}              # camera -> mtime of the last saved frame read
        self.stats = {"fired": 0, "skipped_budget": 0, "skipped_cooldown": 0,
                      "errors": 0, "last_error": ""}
        self.sema = threading.Semaphore(int(self.cfg["concurrency"]))
        self.last_parcel = 0.0

    # ---- config -----------------------------------------------------------
    def _load(self):
        try:
            with open(CFG_PATH) as f:
                self.cfg.update(json.load(f))
        except (OSError, ValueError):
            pass

    def set_cfg(self, patch):
        with self.lock:
            self.cfg.update({k: v for k, v in patch.items() if v is not None})
            with open(CFG_PATH, "w") as f:
                json.dump(self.cfg, f, indent=1)
            self.sema = threading.Semaphore(int(self.cfg["concurrency"]))
        return dict(self.cfg)

    # ---- budget -----------------------------------------------------------
    def _budget_ok(self):
        now = time.time()
        with self.lock:
            while self.calls and now - self.calls[0] > 60:
                self.calls.popleft()
            if len(self.calls) >= int(self.cfg["max_calls_per_min"]):
                self.stats["skipped_budget"] += 1
                return False
            self.calls.append(now)
            return True

    def _cool(self, key):
        now = time.time()
        with self.lock:
            if now - self.cooldown.get(key, 0) < float(self.cfg["cooldown_s"]):
                self.stats["skipped_cooldown"] += 1
                return False
            self.cooldown[key] = now
            return True

    # ---- frames -----------------------------------------------------------
    def _grab(self, camera):
        r = requests.get(f"{GATEWAY}/api/frame.jpeg",
                         params={"src": f"{camera}_{self.cfg['stream']}"}, timeout=8)
        r.raise_for_status()
        return r.content

    def _ensure_frame(self, camera, ts):
        """Feed the ring buffer with the frame the VLM will look at.

        First choice is the DETECTION-SYNCED full frame the detection pipeline
        already cut the person crops from (`/output/siglip/frames/<cam>.jpg`). That
        frame is guaranteed to contain the person, because the detection that fired
        the rule came from it — which is the whole point: no gateway lag, no empty
        frames. Its capture time is the file mtime.

        Only when that frame is missing or stale (cold start, or DS_PIPELINE_CROP=0)
        do we fall back to a throttled gateway grab, the old behaviour.
        """
        dq = self.frames[camera]
        used_saved = False
        try:
            mtime = os.path.getmtime(os.path.join(FRAMES_DIR, f"{camera}.jpg"))
            if (time.time() - mtime <= FRAME_MAX_AGE_S
                    and mtime > self.last_saved.get(camera, 0.0)):
                with open(os.path.join(FRAMES_DIR, f"{camera}.jpg"), "rb") as f:
                    jpeg = f.read()
                if jpeg:
                    self.last_saved[camera] = mtime
                    dq.append((mtime, jpeg))
                    used_saved = True
        except OSError:
            pass

        if not used_saved:
            period = 1.0 / max(float(self.cfg["frame_fps"]), 0.01)
            if ts - self.last_grab.get(camera, 0.0) >= period:
                self.last_grab[camera] = ts
                try:
                    dq.append((ts, self._grab(camera)))
                except Exception:
                    pass

        keep = float(self.cfg["frame_buffer_s"])
        now = time.time()
        while dq and now - dq[0][0] > keep:
            dq.popleft()

    @staticmethod
    def _at_edge(box, frame, frac):
        """True when the box touches the border: the person left the picture."""
        fw, fh = frame
        if not fw or not fh:
            return False
        x, y, w, h = box
        m = float(frac)
        return (x <= m * fw or y <= m * fh
                or x + w >= (1 - m) * fw or y + h >= (1 - m) * fh)

    def _frame_at(self, camera, ts):
        """The buffered frame closest to `ts`, as (capture_ts, jpeg). Used by every
        rule; the vanish rule asks for the moment the person was still there."""
        dq = self.frames.get(camera)
        if not dq:
            return None, None
        return min(dq, key=lambda x: abs(x[0] - ts))

    def _save_snapshot(self, camera, track, jpeg, ts):
        """Write the frame next to its verdict. RAG has to be able to cite it,
        and a human has to be able to see what the VLM was actually looking at."""
        if not self.cfg.get("keep_snapshots"):
            return ""
        d = os.path.join(SNAP_DIR, camera)
        os.makedirs(d, exist_ok=True)
        rel = f"{camera}/{int(ts*1000)}_{track}.jpg"
        with open(os.path.join(SNAP_DIR, rel), "wb") as f:
            f.write(jpeg)
        return rel

    # ---- the ask ----------------------------------------------------------
    def _ask(self, camera, track, gid, trigger, n_persons, jpeg, frame_lag_ms=0):
        if not self._budget_ok():
            return
        with self.sema:
            t0 = time.time()
            try:
                im = Image.open(io.BytesIO(jpeg)).convert("RGB")
                text, err = providers.caption_image(
                    im, provider=self.cfg["provider"], prompt=PROMPT)
                if err:
                    raise RuntimeError(err)
            except Exception as e:
                with self.lock:
                    self.stats["errors"] += 1
                    self.stats["last_error"] = str(e)[:200]
                return
            dt = int((time.time() - t0) * 1000)
        activity, susp = _parse(text)
        snap = ""
        try:
            snap = self._save_snapshot(camera, track, jpeg, time.time())
        except Exception as e:
            with self.lock:
                self.stats["last_error"] = f"snapshot: {str(e)[:120]}"
        ch.behavior(camera=camera, track=track, gid=gid, trigger=trigger,
                    n_persons=n_persons, activity=activity, suspicious=susp,
                    snapshot=snap, model=self.cfg["provider"], latency_ms=dt,
                    frame_lag_ms=frame_lag_ms)
        with self.lock:
            self.stats["fired"] += 1
        print(f"[behavior] {camera} trk={track} gid={gid} {trigger} "
              f"({dt} ms, frame_lag {frame_lag_ms} ms) susp={susp}: {activity}",
              flush=True)

    def _fire(self, camera, track, gid, trigger, n_persons, jpeg, frame_ts=None):
        # how stale the frame the VLM sees is vs the moment the rule fired
        lag = int(max(0.0, time.time() - frame_ts) * 1000) if frame_ts else 0
        threading.Thread(target=self._ask, daemon=True,
                         args=(camera, track, gid, trigger, n_persons, jpeg, lag)
                         ).start()

    # ---- the rules --------------------------------------------------------
    def on_frame(self, camera, persons, ts):
        """persons: [{track, gid, box(x,y,w,h)}] for one camera at one instant."""
        cfg = self.cfg
        if not cfg["enabled"]:
            return
        if not persons:
            return
        self._ensure_frame(camera, ts)
        frame_ts, jpeg = self._frame_at(camera, ts)

        # crowd -------------------------------------------------------------
        if (jpeg and cfg["crowd_enabled"]
                and len(persons) >= int(cfg["crowd_min_persons"])
                and self._cool((camera, -1, "crowd"))):
            self._fire(camera, -1, 0, "crowd", len(persons), jpeg, frame_ts)

        for p in persons:
            key = (camera, p["track"])
            st = self.tracks.get(key)
            if st is None:
                st = {"first": ts, "last": ts, "box": p["box"], "still_since": ts,
                      "gid": p["gid"], "dwelt": False, "frame": p["frame"]}
                self.tracks[key] = st
                # parcel_new ------------------------------------------------
                if (jpeg and cfg["parcel_new_id"] and camera in cfg["parcel_cameras"]
                        and key not in self.seen_tracks
                        and self._cool((camera, p["track"], "parcel_new"))):
                    self._fire(camera, p["track"], p["gid"], "parcel_new",
                               len(persons), jpeg, frame_ts)
                self.seen_tracks.add(key)
                continue

            st["last"] = ts
            st["gid"] = p["gid"] or st["gid"]
            st["frame"] = p["frame"]
            # dwell ---------------------------------------------------------
            ox, oy, _, oh = st["box"]
            nx, ny, _, nh = p["box"]
            moved = ((nx - ox) ** 2 + (ny - oy) ** 2) ** 0.5
            if moved > float(cfg["dwell_move_frac"]) * max(nh, 1):
                st["still_since"] = ts
                st["dwelt"] = False
            st["box"] = p["box"]
            if (jpeg and cfg["dwell_enabled"] and not st["dwelt"]
                    and ts - st["still_since"] >= float(cfg["dwell_s"])
                    and self._cool((camera, p["track"], "dwell"))):
                st["dwelt"] = True
                self._fire(camera, p["track"], st["gid"], "dwell", len(persons),
                           jpeg, frame_ts)

    def sweep(self):
        """Vanished tracks, and the parcel-room clock. Called on a timer."""
        cfg = self.cfg
        if not cfg["enabled"]:
            return
        now = time.time()

        if cfg["vanish_enabled"]:
            for key, st in list(self.tracks.items()):
                if now - st["last"] < float(cfg["vanish_gap_s"]):
                    continue
                camera, track = key
                lived = st["last"] - st["first"]
                del self.tracks[key]
                if lived < float(cfg["vanish_min_s"]):
                    continue
                if self._at_edge(st["box"], st["frame"], cfg["vanish_edge_frac"]):
                    continue          # walked out of view, not into a room
                # the frame from the second they were STILL THERE — now the
                # detection-synced frame the pipeline cut, so it actually has them
                frame_ts, jpeg = self._frame_at(camera, st["last"])
                if jpeg and self._cool((camera, track, "vanish")):
                    self._fire(camera, track, st["gid"], "vanish", 1, jpeg, frame_ts)

        if cfg["parcel_cameras"] and now - self.last_parcel >= float(cfg["parcel_period_s"]):
            self.last_parcel = now
            for cam in cfg["parcel_cameras"]:
                try:
                    jpeg = self._grab(cam)
                except Exception:
                    continue
                self._fire(cam, -1, 0, "parcel_sweep", 0, jpeg)

    # ---- the feed ---------------------------------------------------------
    def run(self):
        """Consume the bridge's detection WebSocket forever."""
        import websocket                    # from websocket-client
        while True:
            try:
                ws = websocket.create_connection(BRIDGE_WS, timeout=30)
                print("[behavior] connected to the bridge", flush=True)
                while True:
                    msg = json.loads(ws.recv())
                    if msg.get("type") != "detections":
                        continue
                    by_cam = collections.defaultdict(list)
                    for r in msg["records"]:
                        if r.get("class_name") != "person":
                            continue
                        b = r.get("bounding_box") or {}
                        f = r.get("frame") or {}
                        by_cam[r["camera_id"]].append({
                            "track": int(r.get("track_id") or -1),
                            "gid": int(r.get("global_id") or 0),
                            "box": (b.get("x", 0), b.get("y", 0),
                                    b.get("w", 0), b.get("h", 0)),
                            "frame": (f.get("width", 0), f.get("height", 0)),
                        })
                    now = time.time()
                    for cam, persons in by_cam.items():
                        self.on_frame(cam, persons, now)
            except Exception as e:
                print(f"[behavior] feed lost: {str(e)[:120]}", flush=True)
                time.sleep(5)


behavior = Behavior()


def _ticker():
    while True:
        time.sleep(1.0)
        try:
            behavior.sweep()
        except Exception as e:
            print(f"[behavior] sweep error: {str(e)[:120]}", flush=True)


def start():
    threading.Thread(target=behavior.run, daemon=True).start()
    threading.Thread(target=_ticker, daemon=True).start()
