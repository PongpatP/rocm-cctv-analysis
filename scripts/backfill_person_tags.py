"""Give every rebuilt Global ID its description — with no VLM call at all.

The person gate already asked the VLM about every local track and stored the six
description fields alongside its verdict (`track_check.tags`). An identity is a
set of accepted tracks, so its description is simply the description of its best
track. That is why the identity-level tagging pass was deleted: it was asking a
question that had already been answered.

"Best" = the most complete view, then the most observations. A full body beats an
upper body; among equals, the track the matcher pooled most crops from wins.

    docker compose run --rm --no-deps -v $PWD/scripts:/scripts:ro \
        --entrypoint python3 siglip /scripts/backfill_person_tags.py
"""
import json
import sqlite3
import sys
import time

sys.path.insert(0, "/app")
import providers                                       # noqa: E402

REID_DB = "/output/siglip/reid.db"
RANK = {"full body": 3, "upper body": 2, "head facing camera": 1}


def main():
    c = sqlite3.connect(REID_DB, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")

    checks = {}
    for cam, trk, verdict, tags in c.execute(
            "SELECT camera, track, verdict, tags FROM track_check"):
        if verdict != "person" or not tags:
            continue
        try:
            checks[(cam, trk)] = json.loads(tags)
        except ValueError:
            continue
    print(f"accepted tracks with tags: {len(checks):,}", flush=True)

    rows = c.execute(
        "SELECT gid, camera, track, MAX(n_obs) FROM sighting "
        "WHERE track IS NOT NULL GROUP BY gid, camera, track").fetchall()
    by_gid = {}
    for gid, cam, trk, n_obs in rows:
        t = checks.get((cam, trk))
        if not t:
            continue
        key = (RANK.get(t.get("visibility"), 0), n_obs or 0)
        if gid not in by_gid or key > by_gid[gid][0]:
            by_gid[gid] = (key, t)

    now = int(time.time() * 1000)
    n = 0
    for gid, (_, tags) in by_gid.items():
        line = providers.tags_line(tags)[:140]
        c.execute("INSERT INTO person(gid,description,visibility,tags,described_ts) "
                  "VALUES(?,?,?,?,?) ON CONFLICT(gid) DO UPDATE SET "
                  "description=excluded.description, visibility=excluded.visibility, "
                  "tags=excluded.tags, described_ts=excluded.described_ts",
                  (gid, line, tags.get("visibility"), json.dumps(tags), now))
        n += 1
    c.commit()

    total = c.execute("SELECT COUNT(*) FROM global_id").fetchone()[0]
    print(f"tagged {n:,} of {total:,} identities — 0 VLM calls", flush=True)
    print("\nvisibility of the identities:")
    for vis, k in c.execute("SELECT visibility, COUNT(*) FROM person "
                            "GROUP BY visibility ORDER BY 2 DESC"):
        print(f"  {str(vis):20s} {k:,}")
    c.close()


if __name__ == "__main__":
    main()
