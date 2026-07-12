"""Mirror person-tracking results into ClickHouse.

sqlite (reid.db) stays the matcher's working memory: the vector bank and the
same-camera exclusivity claims are upserted on every frame, and a columnar store
rewrites parts on every mutation. What ClickHouse gets is the immutable record —
every sighting, and every VLM tag set — which is what an operator queries months
later ("who took it, where did they go").

A failed insert is dropped, not retried: this runs on the /embed hot path.
"""
import json
import logging
import os
import queue
import threading
import time

import requests

log = logging.getLogger("ch")

URL = os.environ.get("CLICKHOUSE_URL", "http://clickhouse:8123")
DB = os.environ.get("CLICKHOUSE_DB", "ccvt")
USER = os.environ.get("CLICKHOUSE_USER", "ccvt")
PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD", "ccvt")
ENABLED = os.environ.get("CLICKHOUSE_URL", "") != ""

_stats = {"faces": 0, "plates": 0, "dropped": 0, "errors": 0, "last_error": ""}
_lock = threading.Lock()


def _insert_now(table, rows):
    if not ENABLED or not rows:
        return
    body = "\n".join(json.dumps(r, separators=(",", ":")) for r in rows)
    try:
        r = requests.post(URL, params={
            "database": DB,
            "query": f"INSERT INTO {table} FORMAT JSONEachRow",
            "async_insert": "1", "wait_for_async_insert": "0",
        }, data=body.encode(), auth=(USER, PASSWORD), timeout=5)
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
        with _lock:
            _stats[table] = _stats.get(table, 0) + len(rows)
    except Exception as e:
        with _lock:
            _stats["errors"] += 1
            _stats["last_error"] = str(e)[:200]
        log.warning("clickhouse insert into %s failed: %s", table, e)


# ---- non-blocking sink ------------------------------------------------------
# These calls sit on hot paths (the /ingest handler's event loop; the matcher's
# per-tracklet assignment). A stalled ClickHouse must never stall the pipeline,
# so rows go onto a bounded queue and a daemon thread does the HTTP. When the
# queue is full the OLDEST rows are dropped and counted: losing a second of
# boxes is survivable, blocking the CCTV pipeline is not.
_q = queue.Queue(maxsize=20000)


def _worker():
    while True:
        table, rows = _q.get()
        _insert_now(table, rows)


threading.Thread(target=_worker, daemon=True).start()


def _enqueue(table, rows):
    if not rows:
        return
    try:
        _q.put_nowait((table, rows))
    except queue.Full:
        try:
            _q.get_nowait()                  # drop the oldest, keep the newest
            _q.put_nowait((table, rows))
        except queue.Empty:
            pass
        with _lock:
            _stats["dropped"] = _stats.get("dropped", 0) + len(rows)


def _insert(table, rows):
    _enqueue(table, rows)


def query(sql, params=None):
    """Run a SELECT, return parsed JSON rows. Synchronous — for offline scripts,
    never for the hot path."""
    p = {"database": DB, "default_format": "JSON"}
    p.update(params or {})
    r = requests.post(URL, params=p, data=sql.encode(),
                      auth=(USER, PASSWORD), timeout=120)
    r.raise_for_status()
    return r.json().get("data", [])


def face(camera, track, gid, person_id, person_name, score, confidence,
         votes, of_votes, runner_up):
    _insert("faces", [{
        "ts": int(time.time() * 1000), "camera": camera, "track": int(track),
        "global_id": int(gid or 0), "person_id": person_id or "",
        "person_name": person_name or "", "score": float(score),
        "confidence": float(confidence), "votes": int(votes),
        "of_votes": int(of_votes), "runner_up": runner_up or "",
    }])


def face_vector(camera, track, gid, n_faces, best_score, vec):
    _insert("face_vectors", [{
        "ts": int(time.time() * 1000), "camera": camera, "track": int(track),
        "global_id": int(gid or 0), "n_faces": int(n_faces),
        "best_score": float(best_score), "vec": [float(x) for x in vec],
    }])


def plate(camera, track, vehicle, plate, confidence, votes, reads,
          first_ms, snapshot):
    _insert("plates", [{
        "ts": int(time.time() * 1000), "first_ts": int(first_ms),
        "camera": camera, "track": int(track), "vehicle": vehicle,
        "plate": plate, "confidence": float(confidence), "votes": int(votes),
        "reads": int(reads), "snapshot": snapshot or "",
    }])


def stats():
    with _lock:
        return dict(_stats, queued=_q.qsize())
