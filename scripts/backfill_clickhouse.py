"""One-time copy of the history already in sqlite into ClickHouse.

  * reid.db:sighting  -> ccvt.sightings   (every cross-camera recognition)
  * reid.db:person    -> ccvt.persons     (the VLM tag sets)
  * stats.db:det_raw  -> ccvt.detections  (boxes still inside the 7-day TTL)

Idempotent for `persons` (ReplacingMergeTree on gid). NOT idempotent for
`sightings`/`detections` — run once, or TRUNCATE first. Run inside the siglip
container, which can reach both the volumes and the clickhouse service.

    docker compose exec siglip python3 /app/backfill.py --what all
"""
import argparse
import json
import sqlite3
import sys
import time

sys.path.insert(0, "/app")
import ch                                    # noqa: E402

REID_DB = "/output/siglip/reid.db"
STATS_DB = "/output/stats.db"
CHUNK = 20000


def _flush(table, rows, n):
    ch._insert_now(table, rows)
    n[0] += len(rows)
    print(f"  {table}: {n[0]:,}", flush=True)


def sightings():
    c = sqlite3.connect(f"file:{REID_DB}?mode=ro", uri=True)
    rows, n = [], [0]
    for gid, cam, ts, matched, score, n_obs, prev, trk in c.execute(
            "SELECT gid,camera,ts,matched,score,n_obs,prev_cam,track FROM sighting "
            "ORDER BY ts"):
        rows.append({"ts": int(ts), "gid": int(gid), "camera": cam,
                     "track": -1 if trk is None else int(trk),
                     "matched": int(bool(matched)),
                     "score": float(score) if score is not None else 0.0,
                     "n_obs": int(n_obs or 0), "prev_cam": prev or ""})
        if len(rows) >= CHUNK:
            _flush("sightings", rows, n); rows = []
    _flush("sightings", rows, n)
    c.close()


def persons():
    c = sqlite3.connect(f"file:{REID_DB}?mode=ro", uri=True)
    rows, n = [], [0]
    for gid, desc, tg, dts in c.execute(
            "SELECT gid, description, tags, described_ts FROM person "
            "WHERE tags IS NOT NULL"):
        t = json.loads(tg)
        rows.append({"gid": int(gid), "described_ts": int(dts or 0),
                     "mode": t.get("mode", ""), "visibility": t.get("visibility", ""),
                     "sex": t.get("sex", ""), "upper": t.get("upper", ""),
                     "lower": t.get("lower", ""), "carry": t.get("carry", ""),
                     "head": t.get("head", ""), "description": desc or ""})
        if len(rows) >= CHUNK:
            _flush("persons", rows, n); rows = []
    _flush("persons", rows, n)
    c.close()


def detections(days):
    """Only what still fits the 7-day TTL — older rows would be deleted anyway."""
    cut = int((time.time() - days * 86400) * 1000)
    c = sqlite3.connect(f"file:{STATS_DB}?mode=ro", uri=True)
    c.execute("PRAGMA busy_timeout=10000")
    rows, n = [], [0]
    for ts, cam, cls, conf, trk, x, y, w, h, fw, fh in c.execute(
            "SELECT ts,camera,class,conf,track,x,y,w,h,fw,fh FROM det_raw "
            "WHERE ts>? ORDER BY ts", (cut,)):
        rows.append({"ts": int(ts), "camera": cam or "", "class": cls or "",
                     "conf": conf or 0.0, "track": -1 if trk is None else int(trk),
                     "global_id": 0,
                     "x": x or 0.0, "y": y or 0.0, "w": w or 0.0, "h": h or 0.0,
                     "fw": int(fw or 0), "fh": int(fh or 0)})
        if len(rows) >= CHUNK:
            _flush("detections", rows, n); rows = []
    _flush("detections", rows, n)
    c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all",
                    choices=["all", "sightings", "persons", "detections"])
    ap.add_argument("--days", type=float, default=7.0)
    a = ap.parse_args()
    t0 = time.time()
    if a.what in ("all", "persons"):
        persons()
    if a.what in ("all", "sightings"):
        sightings()
    if a.what in ("all", "detections"):
        detections(a.days)
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
