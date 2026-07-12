"""The gate: no local track reaches the ReID matcher until the VLM says it is a
person.

Why here and not later. The old design asked the VLM about a GLOBAL ID, after the
matcher had already merged tracks together — two crops chosen out of a bag that
could hold 54 tracks. A rolled mat on a sofa and a real human ended up in one
identity (G4061), the VLM was handed the human's crop, and it answered honestly:
"woman, white top". Nothing was broken except the place the question was asked.

A local track is the right unit: it is one continuous appearance of one thing on
one camera. Ask once, at the start, and a Global ID can never contain a
non-person.

Accepted, per the owner:      full body · upper body · head facing camera
Rejected:                     lower body · legs only · arms only ·
                              head facing away · not a person

The same call returns the six description fields, so the tag pass that used to
run per-identity is gone: an identity's tags are simply the tags of its best
accepted track. One VLM call per track, none per identity.

Cost, measured: ~480 tracks/hour reach the matcher (peak ~1000), 0.9 s per call
on the local Gemma at concurrency 4 — about 2 minutes of GPU 0 per hour.

While a track is pending, its vectors are HELD, not dropped: the owner chose
correctness of the retrospective record over a second of live latency. When the
verdict lands, the whole buffer is replayed into the matcher in order.
"""
import collections
import json
import sqlite3
import threading
import time
from contextlib import closing

import providers

PENDING, PERSON, REJECT = "pending", "person", "reject"

DEFAULTS = {
    "enabled": True,
    "provider": "local-gemma",
    "min_crops": 2,          # look at this many crops of the track before judging
    "max_wait_s": 6.0,       # judge with whatever we have after this long
    "concurrency": 4,        # matches vLLM --max-num-seqs
    "max_buffer": 60,        # vectors held per pending track
}


class TrackVerifier:
    def __init__(self, db_path, on_accept):
        """`on_accept(camera, track, buffered)` replays a track into the matcher."""
        self.db_path = db_path
        self.on_accept = on_accept
        self.cfg = dict(DEFAULTS)
        self.lock = threading.Lock()
        self.state = {}                       # (cam, trk) -> PENDING/PERSON/REJECT
        self.buffer = collections.defaultdict(list)   # (cam, trk) -> [(v,q,ts)]
        self.crops = collections.defaultdict(list)    # (cam, trk) -> [PIL]
        self.first_seen = {}
        self.inflight = set()
        self.tags = {}                        # (cam, trk) -> tag dict
        self.sema = threading.Semaphore(self.cfg["concurrency"])
        self.stats = {"asked": 0, "person": 0, "reject": 0, "errors": 0,
                      "held": 0, "last_error": ""}
        self._init_db()
        self._restore()

    # ---- persistence -------------------------------------------------------
    def _db(self):
        c = sqlite3.connect(self.db_path, timeout=10)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _init_db(self):
        with closing(self._db()) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS track_check(
                camera TEXT NOT NULL, track INTEGER NOT NULL,
                verdict TEXT NOT NULL, visibility TEXT, tags TEXT,
                ts INTEGER NOT NULL, PRIMARY KEY(camera, track))""")
            c.commit()

    def _restore(self):
        """A restart must not re-ask about tracks already judged."""
        with closing(self._db()) as c:
            for cam, trk, verdict, tags in c.execute(
                    "SELECT camera, track, verdict, tags FROM track_check"):
                self.state[(cam, trk)] = verdict
                if tags:
                    try:
                        self.tags[(cam, trk)] = json.loads(tags)
                    except ValueError:
                        pass
        print(f"[verify] restored {len(self.state)} track verdicts", flush=True)

    def _record(self, key, verdict, tags):
        with closing(self._db()) as c:
            c.execute("INSERT INTO track_check(camera,track,verdict,visibility,tags,ts)"
                      " VALUES(?,?,?,?,?,?) ON CONFLICT(camera,track) DO UPDATE SET "
                      "verdict=excluded.verdict, visibility=excluded.visibility, "
                      "tags=excluded.tags, ts=excluded.ts",
                      (key[0], key[1], verdict, (tags or {}).get("visibility"),
                       json.dumps(tags) if tags else None, int(time.time() * 1000)))
            c.commit()

    # ---- config ------------------------------------------------------------
    def set_cfg(self, patch):
        with self.lock:
            self.cfg.update({k: v for k, v in patch.items() if v is not None})
            self.sema = threading.Semaphore(int(self.cfg["concurrency"]))
        return dict(self.cfg)

    # ---- the gate ----------------------------------------------------------
    def submit(self, camera, track, crop, vec, quality, ts):
        """Returns True when the caller may hand this vector to the matcher."""
        if not self.cfg["enabled"]:
            return True
        key = (camera, track)
        st = self.state.get(key)
        if st == PERSON:
            return True
        if st == REJECT:
            return False

        with self.lock:
            if key not in self.first_seen:
                self.first_seen[key] = ts
                self.state[key] = PENDING
            buf = self.buffer[key]
            if len(buf) < int(self.cfg["max_buffer"]):
                buf.append((vec, quality, ts))
                self.stats["held"] = sum(len(b) for b in self.buffer.values())
            crops = self.crops[key]
            if len(crops) < int(self.cfg["min_crops"]):
                crops.append(crop)
            ready = (len(crops) >= int(self.cfg["min_crops"])
                     or (time.time() - self.first_seen[key] / 1000.0
                         > float(self.cfg["max_wait_s"]) and crops))
            if not ready or key in self.inflight:
                return False
            self.inflight.add(key)
            imgs = list(crops)
        threading.Thread(target=self._judge, args=(key, imgs), daemon=True).start()
        return False

    def _judge(self, key, imgs):
        with self.sema:
            t0 = time.time()
            try:
                grey = all(providers.is_greyscale(im) for im in imgs)
                text, err = providers.caption_image(
                    imgs, provider=self.cfg["provider"],
                    prompt=providers.person_prompt(grey))
                if err:
                    raise RuntimeError(err)
            except Exception as e:
                with self.lock:
                    self.stats["errors"] += 1
                    self.stats["last_error"] = str(e)[:200]
                    self.inflight.discard(key)
                    # a failed call must not strand the track forever: let the
                    # next crop try again
                    self.crops[key] = []
                return
            tags = providers.parse_tags(text, grey)
            vis = tags.get("visibility")
            person = vis in providers.VISIBILITY_OK
            dt = int((time.time() - t0) * 1000)

        with self.lock:
            self.stats["asked"] += 1
            self.inflight.discard(key)
            self.state[key] = PERSON if person else REJECT
            self.tags[key] = tags
            buf = self.buffer.pop(key, [])
            self.crops.pop(key, None)
            self.first_seen.pop(key, None)
            self.stats["person" if person else "reject"] += 1
            self.stats["held"] = sum(len(b) for b in self.buffer.values())
        self._record(key, PERSON if person else REJECT, tags)
        print(f"[verify] {key[0]} trk={key[1]} {vis!r} -> "
              f"{'PERSON' if person else 'REJECT'} ({dt} ms, {len(buf)} held)",
              flush=True)
        if person and buf:
            try:
                self.on_accept(key[0], key[1], buf)
            except Exception as e:
                print(f"[verify] replay failed: {str(e)[:120]}", flush=True)

    def tags_for(self, camera, track):
        return self.tags.get((camera, track))

    def sweep(self, older_than_s=120.0):
        """Drop buffers of tracks that never gathered enough crops."""
        now = time.time() * 1000
        with self.lock:
            for key in [k for k, t in self.first_seen.items()
                        if now - t > older_than_s * 1000 and k not in self.inflight]:
                self.buffer.pop(key, None)
                self.crops.pop(key, None)
                self.first_seen.pop(key, None)
                self.state.pop(key, None)
