"""Mirror person-tracking results into ClickHouse.

sqlite (reid.db) stays the matcher's working memory: the vector bank and the
same-camera exclusivity claims are upserted on every frame, and a columnar store
rewrites parts on every mutation. What ClickHouse gets is the immutable record —
every sighting, and every VLM tag set — which is what an operator queries months
later ("who took it, where did they go").

Two sinks live here. Detection-shaped rows go on a bounded queue and are dropped
when it overflows — losing a second of boxes is survivable, stalling the pipeline
is not. Plates and vehicles go through `_insert_durable()`: confirmed by the
server, retried, and spooled to disk if ClickHouse is down. There are tens of them
a day and each is the only record that a vehicle passed the gate.
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

_stats = {"faces": 0, "plates": 0, "vehicles": 0, "dropped": 0, "spooled": 0,
          "errors": 0, "last_error": ""}
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


# ---- durable sink -----------------------------------------------------------
# A plate is not a detection box. There are tens of them a day, each one the only
# record that a particular vehicle passed the gate, and the owner asked for them
# to be kept properly. So these rows do NOT go on the drop-oldest queue: the
# insert is confirmed by the server, retried, and if ClickHouse is unreachable
# the row is written to a spool file on disk and replayed later. `_emit()` already
# blocks for seconds waiting on the VLM, so blocking a little longer costs nothing.
SPOOL = os.environ.get("PLATE_SPOOL", "/output/plate/spool.jsonl")
_spool_lock = threading.Lock()


def _insert_confirmed(table, rows, tries=3):
    body = "\n".join(json.dumps(r, separators=(",", ":")) for r in rows)
    last = None
    for attempt in range(tries):
        try:
            r = requests.post(URL, params={
                "database": DB,
                "query": f"INSERT INTO {table} FORMAT JSONEachRow",
                "async_insert": "1", "wait_for_async_insert": "1",
            }, data=body.encode(), auth=(USER, PASSWORD), timeout=15)
            if r.status_code != 200:
                raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
            with _lock:
                _stats[table] = _stats.get(table, 0) + len(rows)
            return True
        except Exception as e:
            last = e
            time.sleep(0.5 * (attempt + 1))
    with _lock:
        _stats["errors"] += 1
        _stats["last_error"] = str(last)[:200]
    return False


def _spool(table, rows):
    try:
        os.makedirs(os.path.dirname(SPOOL), exist_ok=True)
        with _spool_lock, open(SPOOL, "a") as f:
            for row in rows:
                f.write(json.dumps({"table": table, "row": row}) + "\n")
        with _lock:
            _stats["spooled"] += len(rows)
        log.error("clickhouse unreachable — %d row(s) spooled to %s", len(rows), SPOOL)
    except OSError as e:
        log.error("could not even spool the row: %s", e)


def _insert_durable(table, rows):
    if not ENABLED or not rows:
        return
    if not _insert_confirmed(table, rows):
        _spool(table, rows)


def replay_spool():
    """Push anything a previous outage left on disk. Rewrites the file with the
    rows that still would not go in, so nothing is lost and nothing doubles up."""
    if not os.path.exists(SPOOL):
        return
    with _spool_lock:
        try:
            lines = [l for l in open(SPOOL).read().splitlines() if l.strip()]
        except OSError:
            return
        if not lines:
            return
        left = []
        for line in lines:
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if not _insert_confirmed(item["table"], [item["row"]], tries=1):
                left.append(line)
        if left:
            open(SPOOL, "w").write("\n".join(left) + "\n")
        else:
            os.remove(SPOOL)
        log.info("spool replay: %d recovered, %d still pending",
                 len(lines) - len(left), len(left))


def _spool_watcher():
    while True:
        try:
            replay_spool()
        except Exception as e:
            log.warning("spool replay failed: %s", e)
        time.sleep(60)


threading.Thread(target=_spool_watcher, daemon=True).start()


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


def plate(camera, track, vehicle, plate, confidence, votes, reads,
          first_ms, snapshot):
    _insert_durable("plates", [{
        "ts": int(time.time() * 1000), "first_ts": int(first_ms),
        "camera": camera, "track": int(track), "vehicle": vehicle,
        "plate": plate, "confidence": float(confidence), "votes": int(votes),
        "reads": int(reads), "snapshot": snapshot or "",
    }])


def vehicle(camera, track, cls, frames, colour, body, markings, description,
            plate, plate_conf, plate_px, first_ms, snapshot):
    _insert_durable("vehicles", [{
        "ts": int(time.time() * 1000), "first_ts": int(first_ms),
        "camera": camera, "track": int(track), "class": cls,
        "frames": int(frames), "colour": colour, "body": body,
        "markings": markings, "description": description,
        "plate": plate or "", "plate_conf": float(plate_conf or 0.0),
        "plate_px": int(plate_px), "snapshot": snapshot or "",
    }])


def stats():
    with _lock:
        return dict(_stats, queued=_q.qsize())
