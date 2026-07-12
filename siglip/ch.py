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

_stats = {"sightings": 0, "persons": 0, "behaviors": 0, "dropped": 0,
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


def query(sql, params=None):
    """Run a SELECT, return parsed JSON rows. Synchronous — for offline scripts,
    never for the hot path."""
    p = {"database": DB, "default_format": "JSON"}
    p.update(params or {})
    r = requests.post(URL, params=p, data=sql.encode(),
                      auth=(USER, PASSWORD), timeout=120)
    r.raise_for_status()
    return r.json().get("data", [])


def sighting(gid, camera, ts, matched, score, n_obs, prev_cam, track):
    _insert("sightings", [{
        "ts": int(ts), "gid": int(gid), "camera": camera,
        "track": -1 if track is None else int(track),
        "matched": 1 if matched else 0,
        "score": float(score) if score is not None else 0.0,
        "n_obs": int(n_obs or 0), "prev_cam": prev_cam or "",
    }])


def person(gid, described_ts, tags, description):
    _insert("persons", [{
        "gid": int(gid), "described_ts": int(described_ts),
        "mode": tags.get("mode", ""), "visibility": tags.get("visibility", ""),
        "sex": tags.get("sex", ""), "upper": tags.get("upper", ""),
        "lower": tags.get("lower", ""), "carry": tags.get("carry", ""),
        "head": tags.get("head", ""), "description": description or "",
    }])


def behavior(camera, track, gid, trigger, n_persons, activity, suspicious,
             snapshot, model, latency_ms, frame_lag_ms=0):
    _insert("behaviors", [{
        "ts": int(time.time() * 1000), "camera": camera, "track": int(track),
        "global_id": int(gid or 0), "trigger": trigger,
        "n_persons": int(n_persons), "activity": activity,
        "snapshot": snapshot or "", "suspicious": int(suspicious),
        "model": model, "latency_ms": int(latency_ms),
        "frame_lag_ms": int(frame_lag_ms),
    }])


def episode(gid, camera, track, first_ms, last_ms, n_events, place, summary, model):
    _insert("episodes", [{
        "gid": int(gid), "camera": camera, "track": int(track),
        "first_ts": int(first_ms), "last_ts": int(last_ms),
        "n_events": int(n_events), "place": place or "", "summary": summary,
        "model": model, "made_ts": int(time.time() * 1000),
    }])


def article(gid, n_episodes, n_events, places, text, model):
    _insert("articles", [{
        "gid": int(gid), "made_ts": int(time.time() * 1000),
        "n_episodes": int(n_episodes), "n_events": int(n_events),
        # place labels contain commas ("Covered Car Park Inner Corner Right,
        # Closed Gate"), so they cannot be comma-separated
        "places": " | ".join(places or []), "article": text, "model": model,
    }])


def stats():
    with _lock:
        return dict(_stats, queued=_q.qsize())
