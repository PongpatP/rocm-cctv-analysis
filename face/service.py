"""Face recognition service (GPU 1).

It adds no detector and fetches no frames. The detection pipeline already cuts a
person crop from the MAIN stream for every gated detection, siglip already asks
the VLM whether that track is a person, and only then are the crops forwarded
here. So this service sees exactly the crops the rest of the system trusts.

Recognition is multi-shot, as `facerec.FaceRecognizer` was written to be: crops of
ONE local track are buffered, and when there are enough the module aligns each
with MediaPipe, embeds it with `recognition.onnx`, and votes. A single blurred
frame therefore cannot put a name on anybody.

The gallery is the one the operator already curated on the Face registration page
(`output/faces/registry.json` + `output/faces/images/<id>/*`). It is reloaded when
that file changes, so enrolling somebody does not need a restart.

Results go to ClickHouse `ccvt.faces`, keyed by the LOCAL track and the Global ID,
which is how they join the movement record and the behaviour log.
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
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

import ch
from facerec import FaceRecognizer

FACES_DIR = os.environ.get("FACES_DIR", "/output/faces")
REGISTRY = os.path.join(FACES_DIR, "registry.json")
IMAGES = os.path.join(FACES_DIR, "images")
MODEL_DIR = os.environ.get("FACE_MODELS", "/app/models")

CFG = {
    "enabled": True,
    "min_crops": int(os.environ.get("FACE_MIN_CROPS", "3")),   # votes needed
    "max_crops": int(os.environ.get("FACE_MAX_CROPS", "30")),
    "settle_s": 4.0,          # a track quiet this long is recognised with what we have
    "cooldown_s": 60.0,       # do not re-recognise the same track for this long
    "match_threshold": float(os.environ.get("FACE_MATCH_THRESH", "0.45")),
    # A RAW cosine floor, on top of the module's blended confidence.
    #
    # `FaceRecognizer._blend()` is 0.4*ratio + 0.6*score, and with a single face
    # crop the vote ratio is always 1.0 — so confidence is 0.4 + 0.6*score and
    # clears the 0.40 threshold no matter how bad the match is. That named seven
    # strangers at cosine 0.14-0.18. Enrolled faces score 0.83-0.997 on their own
    # photographs; the best CCTV read across 260 faces was 0.322. Naming the
    # wrong person is worse than naming nobody, so a read must clear this on the
    # raw cosine, and be agreed by at least `min_votes` crops.
    # 0.45 was one hundredth above the best DIFFERENT-person pair measured on the
    # enrolled photographs (0.445). Same-person pairs there start at 0.526.
    "min_score": float(os.environ.get("FACE_MIN_SCORE", "0.55")),
    "min_votes": int(os.environ.get("FACE_MIN_VOTES", "2")),
}

app = FastAPI()
_state = {"ready": False, "gallery": 0, "registry_mtime": 0.0}
_stats = {"received": 0, "recognised": 0, "unknown": 0, "no_face": 0,
          "errors": 0, "last_error": ""}
_lock = threading.Lock()

# (camera, track) -> {"crops": [ndarray], "gid": int, "last": ts, "done": bool}
_buf = collections.defaultdict(lambda: {"crops": [], "gid": 0, "last": 0.0,
                                        "done": False})
_names = {}          # person_id -> display name

rec = FaceRecognizer(model_dir=MODEL_DIR,
                     match_threshold=CFG["match_threshold"],
                     min_votes=CFG["min_crops"])


# ---- gallery ---------------------------------------------------------------
def load_gallery():
    """Enrol everyone the operator registered. Cheap: a handful of people."""
    try:
        mtime = os.path.getmtime(REGISTRY)
    except OSError:
        return
    if mtime == _state["registry_mtime"]:
        return
    try:
        people = json.load(open(REGISTRY))
    except (OSError, ValueError) as e:
        _stats["errors"] += 1
        _stats["last_error"] = f"registry: {str(e)[:120]}"
        return

    n = 0
    for p in people:
        pid = p.get("id")
        if not pid:
            continue
        imgs = []
        for fn in p.get("images", []):
            path = os.path.join(IMAGES, pid, fn)
            im = cv2.imread(path)
            if im is not None:
                imgs.append(im)
        if imgs and rec.enroll(pid, imgs):
            _names[pid] = p.get("name") or pid
            n += 1
        else:
            print(f"[face] {p.get('name', pid)}: no usable face in "
                  f"{len(imgs)} image(s)", flush=True)
    _state["registry_mtime"] = mtime
    _state["gallery"] = n
    _state["ready"] = True
    print(f"[face] gallery: {n} people ({', '.join(_names.values())})", flush=True)


def _gallery_watcher():
    while True:
        try:
            load_gallery()
        except Exception as e:
            _stats["errors"] += 1
            _stats["last_error"] = str(e)[:160]
        time.sleep(10)


# ---- the work --------------------------------------------------------------
def _decode(b64):
    raw = base64.b64decode(b64)
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def _recognise(key, entry):
    camera, track = key
    crops = entry["crops"]
    t0 = time.time()
    try:
        # `recognize_many()` aligns, embeds and votes, but throws the embeddings
        # away. Doing the alignment ourselves lets us KEEP them without paying for
        # MediaPipe twice — and the verdict still comes from the module's own vote
        # rather than a reimplementation of it.
        aligned = [a for a in (rec.align(im) for im in crops)
                   if a is not None and rec._passes_gates(a)]
        if not aligned:
            with _lock:
                _stats["no_face"] += 1
            return
        aligned.sort(key=lambda a: a.sharpness, reverse=True)
        aligned = aligned[:rec.max_face_crops]
        vecs = rec._get_embedder().embed_many([a.crop for a in aligned])
        match = rec._vote(vecs)
    except Exception as e:
        with _lock:
            _stats["errors"] += 1
            _stats["last_error"] = str(e)[:160]
        return
    if match is None:
        with _lock:
            _stats["no_face"] += 1
        return

    votes, of_votes = match.votes or (0, len(crops))
    known = bool(match.is_known and match.person_id
                 and match.score >= CFG["min_score"]
                 and votes >= CFG["min_votes"])
    ch.face(camera=camera, track=track, gid=entry["gid"],
            person_id=match.person_id if known else "",
            person_name=_names.get(match.person_id, "") if known else "",
            score=float(match.score), confidence=float(match.confidence),
            votes=int(votes), of_votes=int(of_votes),
            runner_up=(match.runner_up[0] if match.runner_up else ""))
    # Keep the embedding. A face is the only evidence about identity that does not
    # depend on clothes, light or angle, so it is what can merge two Global IDs
    # the body vector never linked. Thrown away, it is unrecoverable.
    mean = np.asarray(vecs, dtype=np.float32).mean(0)
    n = float(np.linalg.norm(mean))
    if n > 1e-9:
        ch.face_vector(camera, track, entry["gid"], len(vecs),
                       float(match.score), mean / n)
    with _lock:
        _stats["recognised" if known else "unknown"] += 1
    who = _names.get(match.person_id, match.person_id) if known else "unknown"
    print(f"[face] {camera} trk={track} gid={entry['gid']} -> {who} "
          f"(score {match.score:.3f}, conf {match.confidence:.3f}, "
          f"{votes}/{of_votes} votes, {len(crops)} crops, "
          f"{(time.time()-t0)*1000:.0f} ms)", flush=True)


def _sweeper():
    """Recognise tracks that have gone quiet, then forget them."""
    while True:
        time.sleep(1.0)
        now = time.time()
        due = []
        with _lock:
            for key, e in list(_buf.items()):
                if e["done"] or not e["crops"]:
                    if now - e["last"] > CFG["cooldown_s"]:
                        _buf.pop(key, None)
                    continue
                if (len(e["crops"]) >= CFG["max_crops"]
                        or now - e["last"] > CFG["settle_s"]):
                    e["done"] = True
                    due.append((key, {"crops": list(e["crops"]),
                                      "gid": e["gid"]}))
                    e["crops"] = []
        for key, entry in due:
            if len(entry["crops"]) >= 1:
                threading.Thread(target=_recognise, args=(key, entry),
                                 daemon=True).start()


class Crop(BaseModel):
    camera: str
    track: int
    gid: int = 0
    img: str            # base64 jpeg, the same crop siglip embedded


class Batch(BaseModel):
    crops: list[Crop]


@app.post("/crops")
def crops(batch: Batch):
    """siglip forwards the crops of tracks its gate accepted as people."""
    if not CFG["enabled"] or not _state["ready"]:
        return {"taken": 0}
    taken = 0
    now = time.time()
    with _lock:
        for c in batch.crops:
            key = (c.camera, c.track)
            e = _buf[key]
            if e["done"]:
                continue
            if len(e["crops"]) >= CFG["max_crops"]:
                continue
            try:
                e["crops"].append(_decode(c.img))
            except Exception:
                continue
            e["gid"] = c.gid or e["gid"]
            e["last"] = now
            taken += 1
        _stats["received"] += taken
    return {"taken": taken}


@app.get("/healthz")
def healthz():
    return {"ready": _state["ready"], "gallery": _state["gallery"],
            "people": list(_names.values()), "stats": _stats,
            "pending_tracks": len(_buf), "cfg": CFG}


class CfgReq(BaseModel):
    enabled: bool | None = None
    min_crops: int | None = None
    max_crops: int | None = None
    settle_s: float | None = None
    cooldown_s: float | None = None
    min_score: float | None = None
    min_votes: int | None = None


@app.post("/config")
def config(r: CfgReq):
    CFG.update({k: v for k, v in r.dict().items() if v is not None})
    return {"ok": True, "cfg": CFG}


@app.get("/recent")
def recent(hours: float = 6.0, limit: int = 50, known_only: bool = False):
    """Rows written before `min_score` existed carry names decided on a cosine of
    0.11 — the blended confidence alone was enough to clear the module's
    threshold. Those names are wrong, so the floor is applied on read as well as
    on write: a stale row comes back as `unknown` rather than pointing the live
    overlay at the wrong human."""
    where = ["ts > now() - INTERVAL {h:Float64} HOUR"]
    if known_only:
        where.append(f"person_id != '' AND score >= {CFG['min_score']}")
    try:
        rows = ch.query(
            "SELECT toUnixTimestamp64Milli(ts) AS ts_ms, camera, track, global_id, "
            "person_id, person_name, score, confidence, votes, of_votes "
            f"FROM faces WHERE {' AND '.join(where)} ORDER BY ts DESC LIMIT {int(limit)}",
            {"param_h": str(hours)})
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "faces": []}
    floor = CFG["min_score"]
    for r in rows:
        if float(r["score"]) < floor:
            r["person_id"] = ""
            r["person_name"] = ""
    return {"ok": True, "faces": rows}


threading.Thread(target=_gallery_watcher, daemon=True).start()
threading.Thread(target=_sweeper, daemon=True).start()
