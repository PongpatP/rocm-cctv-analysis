"""Run the person gate over the historical crop archive.

The live gate (`siglip/verify.py`) only judges tracks that arrive after it was
deployed. Everything already on disk — the tracklets a rebuild would replay —
has never been asked. Rebuilding without this would let the same rolled mat and
the same bollard back into fresh Global IDs.

Two machines answer, weighted by measured throughput, not by guesswork:

    local Gemma 4 (GPU 0)  236 crops/min at concurrency 4
    secondary VLM (LAN)     48 crops/min at concurrency 4      -> 4.9 : 1

so the worker pools are sized in that ratio and each pulls from one queue. A slow
machine can never become the bottleneck: it simply takes fewer items.

Verdicts land in `track_check` — the same table the live gate writes — so the
rebuild, the person page and the tag pass all read one source of truth.

    docker compose exec siglip python3 /app/verify_tracks.py --min-crops 2
"""
import argparse
import collections
import glob
import json
import os
import queue
import sqlite3
import sys
import threading
import time

sys.path.insert(0, "/app")
import providers                                        # noqa: E402
from PIL import Image                                   # noqa: E402

REID_DB = "/output/siglip/reid.db"
CROPS = "/output/siglip/crops"

# (provider, workers) — proportional to the measured crops/min of each machine
POOLS = [("local-gemma", 10), ("spark", 2)]


def tracklets(min_crops):
    """(camera, track) -> [crop paths], for tracks with enough evidence."""
    by = collections.defaultdict(list)
    for p in glob.glob(f"{CROPS}/*/*.jpg"):
        cam = p.split("/")[-2]
        base = os.path.basename(p)[:-4]
        try:
            trk = int(base.split("_")[0])
        except ValueError:
            continue
        by[(cam, trk)].append(p)
    return {k: sorted(v) for k, v in by.items() if len(v) >= min_crops}


def already_done():
    c = sqlite3.connect(REID_DB)
    c.execute("""CREATE TABLE IF NOT EXISTS track_check(
        camera TEXT NOT NULL, track INTEGER NOT NULL, verdict TEXT NOT NULL,
        visibility TEXT, tags TEXT, ts INTEGER NOT NULL,
        PRIMARY KEY(camera, track))""")
    done = {(a, b) for a, b in c.execute("SELECT camera, track FROM track_check")}
    c.close()
    return done


class Writer(threading.Thread):
    """One thread owns the sqlite handle: reid.db is also serving the live gate."""

    def __init__(self, q):
        super().__init__(daemon=True)
        self.q = q
        self.n = 0

    def run(self):
        c = sqlite3.connect(REID_DB, timeout=30)
        c.execute("PRAGMA journal_mode=WAL")
        while True:
            item = self.q.get()
            if item is None:
                c.commit(); c.close(); return
            cam, trk, verdict, tags = item
            c.execute(
                "INSERT INTO track_check(camera,track,verdict,visibility,tags,ts) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(camera,track) DO UPDATE SET "
                "verdict=excluded.verdict, visibility=excluded.visibility, "
                "tags=excluded.tags, ts=excluded.ts",
                (cam, trk, verdict, (tags or {}).get("visibility"),
                 json.dumps(tags) if tags else None, int(time.time() * 1000)))
            self.n += 1
            if self.n % 50 == 0:
                c.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-crops", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gemma-workers", type=int, default=10)
    ap.add_argument("--spark-workers", type=int, default=2)
    a = ap.parse_args()

    tl = tracklets(a.min_crops)
    done = already_done()
    todo = [k for k in tl if k not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"tracklets with >= {a.min_crops} crops: {len(tl):,}   "
          f"already judged: {len(done):,}   to do: {len(todo):,}", flush=True)
    if not todo:
        return

    work = queue.Queue()
    for k in todo:
        work.put(k)
    out = queue.Queue()
    writer = Writer(out); writer.start()

    stats = collections.Counter()
    lock = threading.Lock()
    t0 = time.time()

    def worker(provider):
        while True:
            try:
                cam, trk = work.get_nowait()
            except queue.Empty:
                return
            paths = tl[(cam, trk)]
            pick = [paths[len(paths) // 2]]
            if len(paths) > 3:
                pick.append(paths[-1])                  # a second viewpoint
            try:
                imgs = [Image.open(p).convert("RGB") for p in pick]
                grey = all(providers.is_greyscale(im) for im in imgs)
                text, err = providers.caption_image(
                    imgs, provider=provider, prompt=providers.person_prompt(grey))
                if err:
                    raise RuntimeError(err)
            except Exception as e:
                with lock:
                    stats["error"] += 1
                    stats[f"err_{provider}"] += 1
                    if stats["error"] < 4:
                        print(f"  {provider} failed: {str(e)[:90]}", flush=True)
                continue
            tags = providers.parse_tags(text, grey)
            vis = tags.get("visibility")
            verdict = "person" if vis in providers.VISIBILITY_OK else "reject"
            out.put((cam, trk, verdict, tags))
            with lock:
                stats[verdict] += 1
                stats[provider] += 1
                n = stats["person"] + stats["reject"]
                if n % 100 == 0:
                    el = time.time() - t0
                    print(f"  {n}/{len(todo)}  {n/el*60:.0f} tracks/min  "
                          f"person={stats['person']} reject={stats['reject']} "
                          f"err={stats['error']}  "
                          f"[gemma {stats['local-gemma']} / spark {stats['spark']}]"
                          f"  ~{(len(todo)-n)/max(n/el,1e-6)/60:.1f} min left",
                          flush=True)

    threads = []
    for prov, n in (("local-gemma", a.gemma_workers), ("spark", a.spark_workers)):
        for _ in range(n):
            t = threading.Thread(target=worker, args=(prov,), daemon=True)
            t.start(); threads.append(t)
    for t in threads:
        t.join()
    out.put(None); writer.join()

    el = time.time() - t0
    print(f"\ndone in {el/60:.1f} min · person {stats['person']:,} · "
          f"reject {stats['reject']:,} · errors {stats['error']}", flush=True)
    print(f"  work split: gemma {stats['local-gemma']:,} · spark {stats['spark']:,}",
          flush=True)

    c = sqlite3.connect(REID_DB)
    print("\n  verdict by visibility:")
    for vis, verdict, n in c.execute(
            "SELECT visibility, verdict, COUNT(*) FROM track_check "
            "GROUP BY visibility, verdict ORDER BY 3 DESC"):
        print(f"    {str(vis):20s} {verdict:8s} {n:,}")
    c.close()


if __name__ == "__main__":
    main()
