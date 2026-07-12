"""Forward the person crops the gate accepted to the face service.

Fire and forget. The face service runs on GPU 1 and may be slow, restarting, or
absent; none of that is allowed to stall the ReID hot path, so crops go onto a
bounded queue and a daemon thread posts them. When the queue is full the oldest
are dropped — a face missed is a face missed, a stalled pipeline is an outage.

Only crops of tracks the VLM already confirmed as people are sent, which is why
this lives downstream of `verify.TrackVerifier` and not inside it.
"""
import base64
import io
import os
import queue
import threading

import requests

URL = os.environ.get("FACE_URL", "")           # empty disables the sink
BATCH = 8
_q = queue.Queue(maxsize=2000)
_stats = {"sent": 0, "dropped": 0, "errors": 0, "last_error": ""}
_lock = threading.Lock()


def _worker():
    while True:
        item = _q.get()
        batch = [item]
        while len(batch) < BATCH:
            try:
                batch.append(_q.get_nowait())
            except queue.Empty:
                break
        try:
            r = requests.post(f"{URL.rstrip('/')}/crops",
                              json={"crops": batch}, timeout=5)
            r.raise_for_status()
            with _lock:
                _stats["sent"] += len(batch)
        except Exception as e:
            with _lock:
                _stats["errors"] += 1
                _stats["last_error"] = str(e)[:160]


if URL:
    threading.Thread(target=_worker, daemon=True).start()


def send(camera, track, gid, crop):
    """crop: a PIL image already cut from the MAIN stream."""
    if not URL:
        return
    try:
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=88)
        item = {"camera": camera, "track": int(track), "gid": int(gid or 0),
                "img": base64.b64encode(buf.getvalue()).decode()}
        _q.put_nowait(item)
    except queue.Full:
        with _lock:
            _stats["dropped"] += 1
    except Exception as e:
        with _lock:
            _stats["errors"] += 1
            _stats["last_error"] = str(e)[:160]


def stats():
    with _lock:
        return dict(_stats, queued=_q.qsize(), enabled=bool(URL))
