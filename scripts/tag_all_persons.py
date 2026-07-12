"""Give every existing Global ID its VLM tags, once.

Runs against the live siglip service, so it uses the same describe path as the
web page: a person that already has tags costs nothing (cached), and a person
whose crops are all gone is skipped. Only identities that actually have a stored
crop can be tagged.

    docker compose exec -T siglip python3 /app/tag_all.py --workers 3
"""
import argparse
import json
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://localhost:8085"
REID_DB = "/output/siglip/reid.db"


def todo():
    c = sqlite3.connect(REID_DB)
    rows = c.execute(
        "SELECT DISTINCT s.gid FROM sighting s "
        "LEFT JOIN person p ON p.gid = s.gid "
        "WHERE s.track IS NOT NULL AND p.tags IS NULL "
        "ORDER BY s.gid").fetchall()
    c.close()
    return [r[0] for r in rows]


def describe(gid, provider):
    body = json.dumps({"provider": provider} if provider else {}).encode()
    req = urllib.request.Request(f"{BASE}/reid/person/{gid}/describe", body,
                                 {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=240))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"ok": False, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--provider", default=None, help="default: whatever AI Settings says")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    gids = todo()
    if a.limit:
        gids = gids[:a.limit]
    print(f"{len(gids)} identities need tags", flush=True)

    lock = threading.Lock()
    done = {"n": 0, "ok": 0, "fail": 0}
    t0 = time.time()

    def one(gid):
        r = describe(gid, a.provider)
        with lock:
            done["n"] += 1
            if r.get("ok"):
                done["ok"] += 1
            else:
                done["fail"] += 1
            if done["n"] % 20 == 0 or done["n"] == len(gids):
                el = time.time() - t0
                rate = done["n"] / el
                left = (len(gids) - done["n"]) / rate if rate else 0
                print(f"  {done['n']}/{len(gids)}  ok={done['ok']} fail={done['fail']}  "
                      f"{rate*60:.0f}/min  ~{left/60:.0f} min left", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, gids))

    print(f"finished in {(time.time()-t0)/60:.1f} min: "
          f"{done['ok']} tagged, {done['fail']} failed", flush=True)
    return 0 if done["fail"] < len(gids) else 1


if __name__ == "__main__":
    sys.exit(main())
