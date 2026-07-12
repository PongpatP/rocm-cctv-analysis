"""Cross-camera ReID brain — lives inside the siglip service.

Design (locked by the brief):
  * ONE embedding space for identity. It is now **YoutuReID** (OpenCV Zoo,
    Apache-2.0, 768-d), chosen by measurement, NOT SigLIP2: on 9,941 crops /
    1,689 identities cut from our own recordings, tracklet-level (pooled) AUC
    was youtureid .971 > osnet-x0.25 .966 > siglip2 .948. Within-camera
    tracking stays in the BoT-SORT tracker; this vector is only the identity /
    cross-camera one. Set REID_EMBEDDER=siglip to switch back.
  * Per (camera, track_id) we pool several quality-gated crop embeddings into
    one tracklet vector, then match it against the active gallery — but ONLY
    against Global IDs whose last-seen camera is REACHABLE (has a manual edge)
    from this tracklet's camera. No transition-time / time-window gating.
  * Appearance bank per Global ID: a primary EMA + a small set of diverse
    exemplars. Every bank update is quality-gated first (blurry / low-conf /
    occluded crops never enter the bank — they may still be matched).
  * Same-day / same-outfit by nature: a Global ID absent from every camera is
    evicted (its vectors dropped). The scene graph (nodes/edges) is durable.

Not over-engineered on purpose (single building, ~30 cams). Online greedy.
"""
import json
import math
import sqlite3
import threading
import time
from contextlib import closing

import numpy as np

import providers

try:
    import ch                       # ClickHouse mirror; absent in offline scripts
except ImportError:
    ch = None

DEFAULTS = {
    # measured on 9,941 crops / 1,689 identities from our recordings, POOLED
    # (tracklet-level): YoutuReID thr .67 -> recall 93.4%, precision 90.9%
    "match_thresh": 0.67,     # cosine to accept same Global ID
    "min_obs": 4,             # crop embeddings pooled before a tracklet is matched
    # An identity is represented by a BANK of vectors and matched by MAJORITY
    # VOTE (the technique the owner uses in face recognition at 10k identities).
    # 30 x 768 float32 = 92 KB/person -> 10,000 people = ~0.9 GB, trivial on the
    # 96 GB GPU. Scoring is one matmul per candidate.
    # A person is a SET OF TRACKLET CENTROIDS — one vector per tracklet, i.e.
    # one per (camera, appearance episode). Matching takes the MAXIMUM cosine
    # over that set: seeing the person from one matching angle is enough.
    # (A majority vote over a multi-modal set punishes diversity: the more views
    # an identity collects, the harder it becomes to match. That was the engine
    # of the ID fragmentation.)
    "bank_size": 30,           # MAX tracklet centroids per Global ID
    # MAJORITY VOTE (the owner's method from face recognition). A candidate is the
    # same person when at least this FRACTION of the identity's bank agrees.
    #   0.0 -> one member is enough (= plain max, merges strangers easily)
    #   0.5 -> more than half must agree
    # Vote only means something when the bank holds genuinely DIFFERENT views,
    # which is what div_thresh enforces below.
    "vote_ratio": 0.5,
    "margin": 0.0,             # Lowe ratio test; 0 = off. best must beat 2nd by this
    # A new view enters the bank only when it is DIFFERENT from every view already
    # there (max cosine < div_thresh). The tracker guarantees the crops are the
    # same person; diversity guarantees they are not the same picture. At 0.97 the
    # bank filled with 30 near-identical frames of one 3-second episode.
    "div_thresh": 0.85,
    # CLIP-family features share a large common direction: two crops of DIFFERENT
    # strangers score cosine ~0.71 raw (measured on 256 of our crops), which is
    # above any usable threshold. Subtracting the mean feature restores a mean of
    # ~0.00. Required for clipreid; harmless for youtureid (0.218 -> -0.004).
    "global_center": True,     # subtract the running mean over ALL features
    "camera_center": False,    # additionally subtract the per-camera mean
    "ema_alpha": 0.9,          # retired: ema is written to gid_vec for compat only
    "bank_quality_min": 0.45,  # min quality to ENTER the bank (not to match)
    "idle_finalize_s": 5.0,    # tracklet idle -> finalize
    # A person who leaves and comes back hours later (morning -> evening) must
    # get the SAME Global ID. So an absent identity is never deleted: after
    # evict_s it goes DORMANT (kept, persisted, still matchable) and is only
    # forgotten after forget_h. A dormant identity left the scene entirely, so
    # it may re-enter at ANY camera — the reachability gate only constrains
    # continuous movement.
    "evict_s": 900.0,          # absent this long -> dormant (not deleted)
    "forget_h": 24.0,          # dormant this long -> finally dropped
    # EXCLUSIVITY: one identity is one body. A Global ID that is currently
    # bound to a live tracklet cannot simultaneously be another tracklet, so it
    # is withheld from the candidate set for this long after its last update.
    # Without this the gallery collapses: one gid becomes an attractor that
    # absorbs everybody (observed: 2 tracklets on the same camera in the same
    # millisecond sharing one gid).
    "busy_s": 2.0,
    # A person can slip past a camera without being detected, so allow a
    # candidate whose last camera is up to reach_hops edges away, not just a
    # direct neighbour.
    "reach_hops": 2,
    # A tracklet seen in ONE crop is not evidence of a new human being — 58% of
    # the Global IDs were born that way. It may JOIN an existing identity, but
    # it may not create one.
    "new_id_min_obs": 2,
    # Same camera = same scene, lighting and viewpoint, so a tracker that just
    # dropped and re-acquired a track needs a much lower bar to be reconnected
    # than a genuine cross-camera match.
    "same_cam_thresh": 0.55,
    # LANE 2 — human-legible corroboration. The vector is kept strict (a black
    # box the owner cannot audit); but a link the vector is only MODERATELY sure
    # of (>= tag_vector_floor) is still made when the VLM's clothing tags agree
    # on >= tag_min_agree non-lighting attributes. Such a link is derived from
    # garments that really exist in the frame, so a human accepts it even when
    # imperfect — the failure mode ("both in black") is one people understand.
    "tag_vector_floor": 0.70,
    "tag_min_agree": 2,
    # A Global ID belongs to ONE calendar day (local time). A tracklet on a new
    # day never joins yesterday's identity — it starts a fresh Global ID — so
    # journeys and their summaries split cleanly by day and stay searchable.
    "day_scoped": True,
}


def _cos(a, b):
    return float(np.dot(a, b))   # inputs are L2-normalised


class ReID:
    def __init__(self, db_path, cfg=None):
        self.db_path = db_path
        self.cfg = dict(DEFAULTS)
        if cfg:
            self.cfg.update(cfg)
        self.lock = threading.RLock()
        # active gallery (in memory; same-day, rebuilt on restart)
        #   gid -> {ema, bank:[vec], last_cam, last_ts, attrs, n_obs}
        self.gallery = {}
        # tracklet buffers: (cam,track) -> {vecs, quals, first_ts, last_ts,
        #                                   gid, cam}
        self.tracklets = {}
        self.cam_mean = {}          # camera -> running mean feature (optional)
        self._next_gid = 1
        self._init_db()

    # ---- storage (graph durable; ids/journey for audit) --------------------
    def _db(self):
        c = sqlite3.connect(self.db_path, timeout=10)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _init_db(self):
        c = self._db()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS node(
            camera TEXT PRIMARY KEY,
            vlm_caption TEXT, confirmed_label TEXT, human_comment TEXT,
            connectivity_summary TEXT, updated_ts INTEGER,
            x REAL, y REAL);
        CREATE TABLE IF NOT EXISTS edge(
            node_a TEXT NOT NULL, node_b TEXT NOT NULL,
            PRIMARY KEY(node_a, node_b));
        CREATE TABLE IF NOT EXISTS global_id(
            gid INTEGER PRIMARY KEY, created_ts INTEGER,
            last_seen_ts INTEGER, last_cam TEXT, attrs TEXT);
        CREATE TABLE IF NOT EXISTS journey(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gid INTEGER NOT NULL, ts INTEGER, camera TEXT,
            exit_snapshot TEXT, caption TEXT);
        -- one row EVERY time a tracklet is bound to a Global ID. This is the
        -- record the cross-camera evaluation reads: matched=1 means the vector
        -- was recognised as an existing identity (a real re-identification).
        CREATE TABLE IF NOT EXISTS sighting(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gid INTEGER NOT NULL, camera TEXT NOT NULL, ts INTEGER NOT NULL,
            matched INTEGER NOT NULL, score REAL, n_obs INTEGER,
            prev_cam TEXT, track INTEGER);
        CREATE INDEX IF NOT EXISTS ix_sight_gid ON sighting(gid, ts);
        CREATE INDEX IF NOT EXISTS ix_sight_ts ON sighting(ts);
        -- identity vectors survive a restart, so the morning person is still
        -- recognised in the evening
        CREATE TABLE IF NOT EXISTS gid_vec(
            gid INTEGER PRIMARY KEY, dim INTEGER, ema BLOB, bank BLOB,
            last_ts INTEGER, last_cam TEXT);
        CREATE TABLE IF NOT EXISTS cam_mean(
            camera TEXT PRIMARY KEY, dim INTEGER, mean BLOB);
        """)
        for col in ("x REAL", "y REAL", "floor INTEGER",
                    "slot_col INTEGER", "slot_row INTEGER"):   # migrate older reid.db
            try:
                c.execute("ALTER TABLE node ADD COLUMN %s" % col)
            except sqlite3.OperationalError:
                pass
        try:
            c.execute("ALTER TABLE sighting ADD COLUMN track INTEGER")
        except sqlite3.OperationalError:
            pass
        try:                       # which tracklet each bank vector came from
            c.execute("ALTER TABLE gid_vec ADD COLUMN bank_keys TEXT")
        except sqlite3.OperationalError:
            pass
        c.execute("""CREATE TABLE IF NOT EXISTS person(
            gid INTEGER PRIMARY KEY, description TEXT, described_ts INTEGER)""")
        # person_id / person_name are written by the FACE, never by this matcher.
        # A named identity is one a human enrolled and the face module recognised.
        for col in ("visibility TEXT", "tags TEXT", "person_id TEXT",
                    "person_name TEXT", "face_score REAL", "face_ts INTEGER"):
            try:
                c.execute("ALTER TABLE person ADD COLUMN %s" % col)
            except sqlite3.OperationalError:
                pass
        # continue Global-ID numbering past any rows persisted by a previous
        # run (the in-memory gallery is same-day and starts empty, but the
        # global_id table survives -> seed _next_gid past MAX to avoid a
        # UNIQUE collision on the next INSERT).
        row = c.execute("SELECT MAX(gid) FROM global_id").fetchone()
        self._next_gid = (row[0] or 0) + 1
        # restore the gallery so restarts do not create duplicate identities
        cut = int((time.time() - self.cfg["forget_h"] * 3600) * 1000)
        n = 0
        for gid, dim, ema, bank, ts, cam, bkeys in c.execute(
                "SELECT gid,dim,ema,bank,last_ts,last_cam,bank_keys FROM gid_vec "
                "WHERE last_ts>?", (cut,)):
            try:
                e = np.frombuffer(ema, dtype=np.float32)
                b = np.frombuffer(bank, dtype=np.float32).reshape(-1, dim)
                try:
                    ks = [tuple(k) if k else None for k in json.loads(bkeys or "[]")]
                except Exception:
                    ks = []
                ks += [None] * (len(b) - len(ks))
                self.gallery[gid] = {"ema": e.copy(), "bank": [x.copy() for x in b],
                                     "bank_keys": ks[:len(b)],
                                     "last_cam": cam, "last_ts": ts, "attrs": {},
                                     "n_obs": 1, "busy_key": None, "busy_ts": 0}
                n += 1
            except Exception:
                pass
        # restore each identity's clothing tags so Lane 2 (tag corroboration)
        # keeps working across a restart, not just for gids born after it.
        try:
            for gid, tg in c.execute(
                    "SELECT gid, tags FROM person WHERE tags IS NOT NULL"):
                g = self.gallery.get(gid)
                if g is not None and tg:
                    try:
                        g["attrs"] = json.loads(tg) or {}
                    except ValueError:
                        pass
        except sqlite3.OperationalError:
            pass
        for cam, dim, mean in c.execute("SELECT camera,dim,mean FROM cam_mean"):
            try:
                self.cam_mean[cam] = np.frombuffer(mean, dtype=np.float32).copy()
            except Exception:
                pass
        c.commit()
        c.close()
        if n:
            print(f"[reid] restored {n} identities from disk", flush=True)

    def _persist(self, gid):
        g = self.gallery.get(gid)
        if g is None:
            return
        bank = np.stack(g["bank"]).astype(np.float32)
        keys = json.dumps([list(k) if k else None
                           for k in g.get("bank_keys", [])])
        with closing(self._db()) as c:
            c.execute("INSERT INTO gid_vec(gid,dim,ema,bank,bank_keys,last_ts,"
                      "last_cam) VALUES(?,?,?,?,?,?,?) "
                      "ON CONFLICT(gid) DO UPDATE SET ema=excluded.ema,"
                      "bank=excluded.bank,bank_keys=excluded.bank_keys,"
                      "last_ts=excluded.last_ts,last_cam=excluded.last_cam",
                      (gid, int(bank.shape[1]), g["ema"].astype(np.float32).tobytes(),
                       bank.tobytes(), keys, g["last_ts"], g["last_cam"]))
            if self.cfg.get("camera_center"):
                for cam, m in self.cam_mean.items():
                    c.execute("INSERT INTO cam_mean(camera,dim,mean) VALUES(?,?,?)"
                              " ON CONFLICT(camera) DO UPDATE SET mean=excluded.mean",
                              (cam, int(m.shape[0]), m.astype(np.float32).tobytes()))
            c.commit()

    # ---- scene graph -------------------------------------------------------
    def neighbors(self, camera):
        with self.lock, closing(self._db()) as c:
            rows = c.execute(
                "SELECT node_b FROM edge WHERE node_a=? "
                "UNION SELECT node_a FROM edge WHERE node_b=?",
                (camera, camera)).fetchall()
        return {r[0] for r in rows}

    def reachable(self, camera):
        """Cameras within reach_hops edges (BFS). 1 hop = direct neighbour."""
        seen = {camera}
        frontier = {camera}
        for _ in range(max(1, int(self.cfg["reach_hops"]))):
            nxt = set()
            for c in frontier:
                nxt |= self.neighbors(c)
            frontier = nxt - seen
            seen |= nxt
            if not frontier:
                break
        return seen

    def get_graph(self):
        with self.lock, closing(self._db()) as c:
            nodes = [dict(camera=r[0], vlm_caption=r[1], confirmed_label=r[2],
                          human_comment=r[3], connectivity_summary=r[4],
                          updated_ts=r[5], x=r[6], y=r[7], floor=r[8],
                          slot_col=r[9], slot_row=r[10])
                     for r in c.execute("SELECT camera,vlm_caption,"
                     "confirmed_label,human_comment,connectivity_summary,"
                     "updated_ts,x,y,floor,slot_col,slot_row FROM node")]
            edges = [[r[0], r[1]] for r in c.execute(
                "SELECT node_a,node_b FROM edge")]
        return {"nodes": nodes, "edges": edges}

    def set_pos(self, camera, x, y):
        x = None if x is None else float(x)
        y = None if y is None else float(y)
        with self.lock, closing(self._db()) as c:
            c.execute("INSERT INTO node(camera,x,y,updated_ts) VALUES(?,?,?,?) "
                      "ON CONFLICT(camera) DO UPDATE SET x=excluded.x,y=excluded.y",
                      (camera, x, y, int(time.time() * 1000)))
            c.commit()

    def set_slot(self, camera, col, row):
        """Grid slot inside the camera's floor lane. Slots (not pixels) are the
        stored truth: a floor above growing a row must not move everyone else."""
        with self.lock, closing(self._db()) as c:
            c.execute("INSERT INTO node(camera,slot_col,slot_row,updated_ts) "
                      "VALUES(?,?,?,?) ON CONFLICT(camera) DO UPDATE SET "
                      "slot_col=excluded.slot_col,slot_row=excluded.slot_row",
                      (camera, int(col), int(row), int(time.time() * 1000)))
            c.commit()

    def set_floor(self, camera, floor):
        # floor is a label string ("1F") or None to clear the override; the
        # authoritative default comes from calib's floors.json (seeded client-side)
        floor = None if floor in (None, "") else str(floor)
        with self.lock, closing(self._db()) as c:
            c.execute("INSERT INTO node(camera,floor,updated_ts) VALUES(?,?,?) "
                      "ON CONFLICT(camera) DO UPDATE SET floor=excluded.floor",
                      (camera, floor, int(time.time() * 1000)))
            c.commit()

    def get_node(self, camera):
        with self.lock, closing(self._db()) as c:
            r = c.execute("SELECT camera,vlm_caption,confirmed_label,"
                          "human_comment,connectivity_summary FROM node "
                          "WHERE camera=?", (camera,)).fetchone()
        if not r:
            return {"camera": camera, "vlm_caption": None,
                    "confirmed_label": None, "human_comment": None,
                    "connectivity_summary": None}
        return {"camera": r[0], "vlm_caption": r[1], "confirmed_label": r[2],
                "human_comment": r[3], "connectivity_summary": r[4]}

    def place_text(self, camera):
        """Best human-or-machine description of a node's place."""
        n = self.get_node(camera)
        return n["confirmed_label"] or n["vlm_caption"]

    def set_cfg(self, patch):
        if patch:
            with self.lock:
                self.cfg.update({k: patch[k] for k in patch if k in DEFAULTS})

    def add_edge(self, a, b):
        if a == b:
            return
        a, b = sorted((a, b))
        with self.lock, closing(self._db()) as c:
            c.execute("INSERT OR IGNORE INTO edge(node_a,node_b) VALUES(?,?)",
                      (a, b))
            c.commit()

    def del_edge(self, a, b):
        a, b = sorted((a, b))
        with self.lock, closing(self._db()) as c:
            c.execute("DELETE FROM edge WHERE node_a=? AND node_b=?", (a, b))
            c.commit()

    def set_node(self, camera, field, value):
        """Update ONE field. The three human/machine layers stay separate;
        callers must never route human_comment into a machine field or
        overwrite confirmed_label from a machine path (enforced by endpoint)."""
        if field not in ("vlm_caption", "confirmed_label", "human_comment",
                         "connectivity_summary"):
            raise ValueError("bad field")
        with self.lock, closing(self._db()) as c:
            c.execute("INSERT INTO node(camera,%s,updated_ts) VALUES(?,?,?) "
                      "ON CONFLICT(camera) DO UPDATE SET %s=excluded.%s,"
                      "updated_ts=excluded.updated_ts" % (field, field, field),
                      (camera, value, int(time.time() * 1000)))
            c.commit()

    # ---- matcher core ------------------------------------------------------
    def observe(self, camera, track, vec, quality, attrs=None, ts=None):
        """One accepted crop embedding for (camera, track). Buffers it; when the
        tracklet matures it is pooled and matched. Returns the assigned Global
        ID once known, else None."""
        ts = ts or int(time.time() * 1000)
        vec = np.asarray(vec, dtype=np.float32)
        key = (camera, int(track))
        with self.lock:
            tl = self.tracklets.get(key)
            if tl is None:
                tl = {"vecs": [], "quals": [], "first_ts": ts, "last_ts": ts,
                      "gid": None, "cam": camera, "attrs": attrs or {}}
                self.tracklets[key] = tl
            tl["vecs"].append(vec)
            tl["quals"].append(float(quality))
            self._update_cam_mean(camera, vec)
            tl["last_ts"] = ts
            if attrs:
                tl["attrs"].update(attrs)
            if tl["gid"] is None and len(tl["vecs"]) >= self.cfg["min_obs"]:
                tl["gid"] = self._match_and_assign(tl, camera, ts, key)
            elif tl["gid"] is not None:
                # keep the assigned identity fresh AND keep holding the claim.
                # The bank entry for THIS tracklet is refined in place with the
                # updated centroid — one tracklet, one vector.
                self._claim(tl["gid"], key, ts)
                self._touch(tl["gid"], camera, ts)
                self._bank_update(tl["gid"],
                                  self._center(self._pool(tl), camera), key)
            return tl["gid"]

    def _pool(self, tl):
        """The tracklet centroid: a PLAIN mean of its accepted crop vectors.

        It used to be weighted by an invented quality score
        (0.5*conf + 0.3*blur/120 + 0.2*area/20000) with no provenance. Quality is
        now only an admission filter upstream (bank_quality_min) — a bad crop is
        simply never handed to observe().
        """
        pooled = np.stack(tl["vecs"]).mean(0)
        n = np.linalg.norm(pooled)
        return pooled / n if n > 0 else pooled

    GLOBAL = "__global__"      # cam_mean key for the mean over every camera

    def _center(self, vec, camera):
        """Remove the component every feature shares, then renormalise.

        `global_center` removes the embedding's common direction — mandatory for
        CLIP-family vectors, whose raw cosine between two strangers is ~0.71.
        `camera_center` additionally removes that camera's lighting/viewpoint bias.
        """
        v = vec
        for key, on in ((self.GLOBAL, self.cfg.get("global_center")),
                        (camera, self.cfg.get("camera_center"))):
            if not on:
                continue
            m = self.cam_mean.get(key)
            if m is None:
                continue
            v = v - m
        n = np.linalg.norm(v)
        return v / n if n > 0 else vec

    def _update_cam_mean(self, camera, vec):
        for key, on in ((self.GLOBAL, self.cfg.get("global_center")),
                        (camera, self.cfg.get("camera_center"))):
            if not on:
                continue
            m = self.cam_mean.get(key)
            self.cam_mean[key] = vec.copy() if m is None else 0.99 * m + 0.01 * vec

    @staticmethod
    def _day(ts_ms):
        """Local calendar day key for a millisecond timestamp. Uses the
        container timezone (TZ=Asia/Bangkok), so 'day' is the operator's day."""
        lt = time.localtime((ts_ms or 0) / 1000.0)
        return lt.tm_year * 1000 + lt.tm_yday

    def _match_and_assign(self, tl, camera, ts, key=None):
        # The appearance vector is still pooled and banked FOR THE RECORD (crops,
        # possible future use) but it NO LONGER decides identity. The owner
        # withdrew trust from the black-box cosine (it silently merged a woman
        # into a men's ID at 0.82); Global IDs are now decided PURELY by the
        # VLM's clothing tags — a rule a human can read and vouch for.
        vec = self._center(self._pool(tl), camera)
        tl_tags = tl["attrs"]
        dkey = self._day(ts)
        # IDENTITY = VLM CLOTHING TAGS, within one calendar day. Owner's rule: the
        # VLM tags decide who is who — the same-camera iron rule, the camera-graph
        # reachability and the min-crop floor are all gone. The ONE non-tag gate kept
        # is DAY-SCOPE: a Global ID belongs to a single day, so the same uniform seen
        # tomorrow becomes a fresh ID (the owner wants per-day identities).
        best_gid, best_agree, best_ts = None, -1, -1
        for gid, g in self.gallery.items():
            if self._day(g["last_ts"]) != dkey:
                continue                       # a Global ID is one calendar day
            if providers.tags_verdict(tl_tags, g["attrs"]) == "DIFFERENT":
                continue                       # VETO: sex / trousers / hair / silhouette
            if not providers.upper_agree(tl_tags, g["attrs"]):
                continue                       # the SHIRT must match (primary rule)
            # among shirt-matching, no-contradiction candidates, the strongest
            # overall clothing agreement wins; tie -> most recently seen.
            agree = providers.tags_agreement(tl_tags, g["attrs"])
            if agree > best_agree or (agree == best_agree and g["last_ts"] > best_ts):
                best_gid, best_agree, best_ts = gid, agree, g["last_ts"]
        n_obs = len(tl["vecs"])
        if best_gid is not None:
            prev_cam = self.gallery[best_gid]["last_cam"]
            self._claim(best_gid, key, ts)
            self._touch(best_gid, camera, ts)
            self._bank_update(best_gid, vec, key)
            # score is now NULL: identity was decided by tags, not a cosine
            self._log_sighting(best_gid, camera, ts, 1, None, n_obs, prev_cam,
                               key[1] if key else None)
            self._persist(best_gid)
            return best_gid
        gid = self._new_gid(vec, camera, ts, tl["attrs"], key)
        self._claim(gid, key, ts)
        self._log_sighting(gid, camera, ts, 0, None, n_obs, None,
                           key[1] if key else None)
        return gid

    def _claim(self, gid, key, ts):
        g = self.gallery.get(gid)
        if g is not None:
            g["busy_key"] = key
            g["busy_ts"] = ts

    def merge_gid(self, keep, drop):
        """Fold `drop` into `keep`: its sightings, journey and vectors move over."""
        with self.lock, closing(self._db()) as c:
            c.execute("UPDATE sighting SET gid=? WHERE gid=?", (keep, drop))
            c.execute("UPDATE journey  SET gid=? WHERE gid=?", (keep, drop))
            c.execute("DELETE FROM gid_vec   WHERE gid=?", (drop,))
            c.execute("DELETE FROM global_id WHERE gid=?", (drop,))
            c.execute("DELETE FROM person    WHERE gid=?", (drop,))
            c.commit()
        gk, gd = self.gallery.get(keep), self.gallery.pop(drop, None)
        if gk is not None and gd is not None:
            gkeys = gk.setdefault("bank_keys", [None] * len(gk["bank"]))
            dkeys = gd.get("bank_keys", [None] * len(gd["bank"]))
            for v, k in zip(gd["bank"], dkeys):   # union the centroid sets
                if all(_cos(v, ex) < self.cfg["div_thresh"] for ex in gk["bank"]):
                    gk["bank"].append(v); gkeys.append(k)
            gk["bank"] = gk["bank"][: int(self.cfg["bank_size"])]
            gk["bank_keys"] = gkeys[: int(self.cfg["bank_size"])]
            if gd["last_ts"] > gk["last_ts"]:
                gk["last_ts"], gk["last_cam"] = gd["last_ts"], gd["last_cam"]
            self._persist(keep)

    def split_gid(self, gid, move_keys):
        """Move the given (camera, track) sightings OUT of `gid` into a brand
        new Global ID and return it. Used to break a wrongly-merged identity
        apart so each person shows as its own card. Reassigns the durable
        record (sighting, journey) and rebuilds both galleries' banks."""
        move = {(str(k[0]), int(k[1])) for k in move_keys}
        if not move:
            return None
        with self.lock, closing(self._db()) as c:
            new = self._next_gid
            self._next_gid += 1
            for cam, trk in move:
                c.execute("UPDATE sighting SET gid=? WHERE gid=? AND camera=? "
                          "AND track=?", (new, gid, cam, trk))
            # journey has no track column (it is a per-camera-visit summary);
            # it regenerates from sightings, so we leave it and let it refresh.
            row = c.execute("SELECT MAX(ts), last_cam FROM sighting WHERE gid=? "
                            "ORDER BY ts DESC LIMIT 1", (new,)).fetchone() \
                if False else c.execute(
                    "SELECT MAX(ts) FROM sighting WHERE gid=?", (new,)).fetchone()
            last_ts = (row[0] if row else 0) or int(time.time() * 1000)
            lc = c.execute("SELECT camera FROM sighting WHERE gid=? ORDER BY ts "
                           "DESC LIMIT 1", (new,)).fetchone()
            last_cam = lc[0] if lc else ""
            c.execute("INSERT INTO global_id(gid,created_ts,last_seen_ts,"
                      "last_cam,attrs) VALUES(?,?,?,?,?)",
                      (new, last_ts, last_ts, last_cam, "{}"))
            c.commit()
        # rebuild in-memory banks: move the matching centroids to the new gid
        src = self.gallery.get(gid)
        if src and src.get("bank"):
            keys = src.get("bank_keys", [None] * len(src["bank"]))
            keepB, keepK, mvB, mvK = [], [], [], []
            for v, k in zip(src["bank"], keys):
                if k and (str(k[0]), int(k[1])) in move:
                    mvB.append(v); mvK.append(k)
                else:
                    keepB.append(v); keepK.append(k)
            if keepB:
                src["bank"], src["bank_keys"] = keepB, keepK
                self._persist(gid)
            else:
                self.gallery.pop(gid, None)
            if mvB:
                self.gallery[new] = {"ema": mvB[0].copy(), "bank": mvB,
                                     "bank_keys": mvK, "last_cam": last_cam,
                                     "last_ts": last_ts, "attrs": {},
                                     "n_obs": len(mvB), "busy_key": None,
                                     "busy_ts": last_ts}
                self._persist(new)
        return new

    def enforce_exclusivity(self):
        """Iron rule, applied to existing data: no Global ID may hold two
        tracklets that were alive on the SAME camera in OVERLAPPING time — they
        are two physical bodies. For each offending identity, greedily colour its
        tracklets by start time into the fewest 'slots' such that no slot has a
        same-camera time overlap (optimal for interval graphs), keep slot 0 in
        place and move every other slot into its own new Global ID. Returns a
        summary. Safe to re-run: a clean gallery yields zero splits."""
        from collections import defaultdict
        with closing(self._db()) as c:
            rows = c.execute("SELECT gid,camera,track,MIN(ts),MAX(ts) FROM "
                             "sighting WHERE track IS NOT NULL "
                             "GROUP BY gid,camera,track").fetchall()
        by_gid = defaultdict(list)
        for gid, cam, trk, mn, mx in rows:
            by_gid[gid].append((str(cam), int(trk), mn, mx))
        plans = []                       # (gid, [ [keys slot1], [keys slot2] ])
        for gid, tracks in by_gid.items():
            tracks.sort(key=lambda t: t[2])
            slots = []
            for cam, trk, mn, mx in tracks:
                for slot in slots:
                    if all(not (s[0] == cam and s[2] <= mx and mn <= s[3])
                           for s in slot):
                        slot.append((cam, trk, mn, mx))
                        break
                else:
                    slots.append([(cam, trk, mn, mx)])
            if len(slots) > 1:
                extra = [[(s[0], s[1]) for s in slot] for slot in slots[1:]]
                plans.append((gid, extra))
        made = 0
        for gid, extra in plans:
            for move in extra:
                if self.split_gid(gid, move):
                    made += 1
        return {"violating_gids": len(plans), "new_gids": made}

    def split_below_threshold(self, thr=None):
        """Retroactively apply the match threshold: any tracklet that joined an
        identity with a best score BELOW `thr` was admitted under a looser bar
        (or a mis-loaded config) and is treated as a wrong link. Move each such
        tracklet to its OWN new Global ID — never lump them together, since two
        weak links are two different strangers, not one group. The seed sighting
        (matched=0, no score) always stays, so a gid is never emptied. Returns a
        summary. Safe to re-run."""
        from collections import defaultdict
        thr = float(thr if thr is not None else self.cfg["match_thresh"])
        with closing(self._db()) as c:
            rows = c.execute("SELECT gid,camera,track,MAX(score) FROM sighting "
                             "WHERE track IS NOT NULL AND matched=1 "
                             "GROUP BY gid,camera,track").fetchall()
        weak = defaultdict(list)
        for gid, cam, trk, sc in rows:
            if (sc or 0.0) < thr:
                weak[gid].append((str(cam), int(trk)))
        made = 0
        for gid, keys in weak.items():
            for key in keys:                 # one new gid per weak tracklet
                if self.split_gid(gid, [key]):
                    made += 1
        return {"gids_touched": len(weak), "new_gids": made, "thr": thr}

    def split_by_day(self):
        """Backfill the day-scoping rule onto existing identities: any Global ID
        whose tracklets span more than one local calendar day is split so each
        day becomes its own Global ID (the earliest day keeps the original id).
        Deterministic; idempotent."""
        from collections import defaultdict
        with closing(self._db()) as c:
            rows = c.execute("SELECT gid,camera,track,MIN(ts) FROM sighting "
                             "WHERE track IS NOT NULL "
                             "GROUP BY gid,camera,track").fetchall()
        gid_days = defaultdict(lambda: defaultdict(list))
        for gid, cam, trk, mn in rows:
            gid_days[gid][self._day(mn)].append((str(cam), int(trk)))
        split_gids = 0
        made = 0
        for gid, dmap in gid_days.items():
            if len(dmap) <= 1:
                continue
            split_gids += 1
            for d in sorted(dmap)[1:]:          # keep earliest day, move the rest
                if self.split_gid(gid, dmap[d]):
                    made += 1
        return {"gids_split": split_gids, "new_gids": made}

    def split_lane_inconsistent(self):
        """Backfill the two-lane rule: a tracklet that joined at a vector score
        BELOW match_thresh survives only as a valid Lane-2 link — its vector
        cleared `tag_vector_floor` AND its VLM tags agree with the identity on
        >= `tag_min_agree` non-lighting attributes. What satisfies NEITHER lane
        is peeled to its own Global ID (two weak, unexplained links are two
        strangers). The identity's tags are taken from its ANCHOR track — the
        highest-scoring member's per-track tags in `track_check`, which is dense
        — NOT the sparse per-gid `person.tags`, so a track is never peeled merely
        because its gid lacked a stored summary. Uses recorded score + stored
        tags, mirroring how the live matcher admits a track."""
        from collections import defaultdict
        thr = float(self.cfg["match_thresh"])
        floor = float(self.cfg.get("tag_vector_floor", 0.70))
        need = int(self.cfg.get("tag_min_agree", 2))
        with closing(self._db()) as c:
            ttags = {}
            for cam, trk, t in c.execute(
                    "SELECT camera,track,tags FROM track_check WHERE tags IS NOT NULL"):
                try:
                    ttags[(str(cam), int(trk))] = json.loads(t)
                except (ValueError, TypeError):
                    pass
            rows = c.execute("SELECT gid,camera,track,MAX(score) FROM sighting "
                             "WHERE track IS NOT NULL AND matched=1 "
                             "GROUP BY gid,camera,track").fetchall()
        g_tracks = defaultdict(list)
        for gid, cam, trk, sc in rows:
            g_tracks[gid].append((str(cam), int(trk), sc or 0.0))
        made = 0
        for gid, tl in g_tracks.items():
            anchor_key = max(tl, key=lambda x: x[2])[:2]   # most confident member
            anchor = ttags.get(anchor_key)
            if not anchor:
                continue                        # no tags to judge against -> leave gid
            for cam, trk, sc in tl:
                if (cam, trk) == anchor_key:
                    continue
                a = ttags.get((cam, trk))
                # TAG VETO first: a positive contradiction (sex, or garments with
                # nothing in common) peels the track no matter how high the vector
                # scored — this is what a strong-but-wrong vector merge looks like.
                if providers.tags_verdict(a, anchor) == "DIFFERENT":
                    if self.split_gid(gid, [(cam, trk)]):
                        made += 1
                    continue
                if sc >= thr:
                    continue                    # Lane 1: strong vector, tags don't object
                if sc >= floor and providers.tags_agreement(a, anchor) >= need:
                    continue                    # Lane 2: corroborated by clothing
                if self.split_gid(gid, [(cam, trk)]):
                    made += 1
        return {"peeled": made, "new_gids": made, "thr": thr, "floor": floor}

    def recluster_by_tags(self):
        """Rebuild EVERY Global ID from scratch using ONLY the VLM tags. Replays
        all historical tracklets in time order through the exact rule the live
        matcher now uses — day scope, reachability, same-camera iron rule, and
        tag agreement (>= tag_min_agree, no contradiction) — with NO vector at
        all. Renumbers identities and rewrites sighting/global_id/person/gid_vec.
        The appearance vectors are carried into the new banks for the record but
        never gate a merge. Restart (or gallery reload) after calling."""
        from collections import defaultdict
        need = int(self.cfg.get("tag_min_agree", 2))
        dormant_ms = self.cfg["evict_s"] * 1000
        with self.lock, closing(self._db()) as c:
            ttags = {}
            for cam, trk, t in c.execute(
                    "SELECT camera,track,tags FROM track_check WHERE tags IS NOT NULL"):
                try:
                    ttags[(str(cam), int(trk))] = json.loads(t)
                except (ValueError, TypeError):
                    pass
            # per-tracklet appearance vector, salvaged from the old banks
            tvec = {}
            for dim, bank, bkeys in c.execute("SELECT dim,bank,bank_keys FROM gid_vec"):
                try:
                    B = np.frombuffer(bank, dtype=np.float32).reshape(-1, dim)
                    ks = json.loads(bkeys or "[]")
                    for v, k in zip(B, ks):
                        if k:
                            tvec[(str(k[0]), int(k[1]))] = v.copy()
                except Exception:
                    pass
            rows = c.execute("SELECT camera,track,MIN(ts),MAX(ts) FROM sighting "
                             "WHERE track IS NOT NULL GROUP BY camera,track").fetchall()
            tracklets = sorted(((str(cam), int(trk), mn, mx)
                                for cam, trk, mn, mx in rows), key=lambda x: x[2])
            groups = []           # each: last_cam,last_ts,tags,day,spans,members
            for cam, trk, mn, mx in tracklets:
                tags = ttags.get((cam, trk))
                d = self._day(mn)
                ck = providers._upper_kind((tags or {}).get("upper"))
                best, best_agree, best_ts = None, -1, -1
                if tags:
                    for gi, g in enumerate(groups):
                        if g["day"] != d:
                            continue           # a Global ID is one calendar day
                        # IDENTITY = VLM TAGS (within a day). No reachability, no
                        # iron rule — a day's clothing bucket spans every camera.
                        # a group LOCKS its garment silhouette once any member
                        # has one: a gown group never takes a t-shirt tracklet
                        # (and vice versa), even if the anchor tags were an
                        # ambiguous cardigan/jacket that would not veto on its own.
                        if ck and g.get("ukind") and ck != g["ukind"]:
                            continue
                        if providers.tags_verdict(tags, g["tags"]) == "DIFFERENT":
                            continue
                        if not providers.upper_agree(tags, g["tags"]):
                            continue
                        agree = providers.tags_agreement(tags, g["tags"])
                        if agree > best_agree or (agree == best_agree
                                                  and g["last_ts"] > best_ts):
                            best, best_agree, best_ts = gi, agree, g["last_ts"]
                if best is None:
                    g = {"last_cam": cam, "last_ts": mx, "tags": tags or {},
                         "day": d, "spans": defaultdict(list), "members": [],
                         "ukind": ck}
                    groups.append(g)
                    best = len(groups) - 1
                else:
                    g = groups[best]
                    g["last_cam"] = cam
                    g["last_ts"] = max(g["last_ts"], mx)
                    if not g["tags"] and tags:
                        g["tags"] = tags
                    if ck and not g.get("ukind"):
                        g["ukind"] = ck                    # lock on first silhouette
                groups[best]["spans"][cam].append((mn, mx))
                groups[best]["members"].append((cam, trk))
            # rewrite the durable record with fresh, contiguous Global IDs
            c.execute("DELETE FROM global_id")
            c.execute("DELETE FROM person")
            c.execute("DELETE FROM gid_vec")
            for gi, g in enumerate(groups, start=1):
                for cam, trk in g["members"]:
                    c.execute("UPDATE sighting SET gid=? WHERE camera=? AND track=?",
                              (gi, cam, trk))
                tags = g["tags"] or {}
                line = providers.tags_line(tags)[:140] if tags else ""
                c.execute("INSERT INTO global_id(gid,created_ts,last_seen_ts,"
                          "last_cam,attrs) VALUES(?,?,?,?,?)",
                          (gi, g["last_ts"], g["last_ts"], g["last_cam"],
                           json.dumps(tags)))
                c.execute("INSERT INTO person(gid,description,visibility,tags,"
                          "described_ts) VALUES(?,?,?,?,?)",
                          (gi, line, tags.get("visibility"),
                           json.dumps(tags) if tags else None, g["last_ts"]))
                # rebuild the bank (one centroid per member tracklet) for the
                # record. Historical banks can carry different dims (embedder
                # changes); keep only vectors that match the first one's shape.
                vecs, keys, dim0 = [], [], None
                for cam, trk in g["members"]:
                    v = tvec.get((cam, trk))
                    if v is None:
                        continue
                    if dim0 is None:
                        dim0 = v.shape[0]
                    if v.shape[0] != dim0:
                        continue
                    vecs.append(v)
                    keys.append([cam, trk])
                if vecs:
                    B = np.stack(vecs[: int(self.cfg["bank_size"])]).astype(np.float32)
                    ema = B.mean(0)
                    c.execute("INSERT INTO gid_vec(gid,dim,ema,bank,bank_keys,"
                              "last_ts,last_cam) VALUES(?,?,?,?,?,?,?)",
                              (gi, B.shape[1], ema.tobytes(), B.tobytes(),
                               json.dumps(keys[: int(self.cfg["bank_size"])]),
                               g["last_ts"], g["last_cam"]))
            self._next_gid = len(groups) + 1
            c.commit()
        self.gallery.clear()          # will be reloaded fresh from the new gid_vec
        sizes = sorted((len(g["members"]) for g in groups), reverse=True)
        return {"tracklets": len(tracklets), "new_gids": len(groups),
                "singletons": sum(1 for s in sizes if s == 1),
                "largest_group": sizes[0] if sizes else 0}

    def banks(self):
        """gid -> (N,dim) matrix, for offline duplicate hunting."""
        with self.lock:
            return {g: np.stack(v["bank"]) for g, v in self.gallery.items()
                    if v["bank"]}

    def delete_gid(self, gid):
        """Remove an identity entirely (used when the VLM says it is not a person)."""
        with self.lock, closing(self._db()) as c:
            for t in ("sighting", "gid_vec", "global_id", "person", "journey"):
                try:
                    c.execute(f"DELETE FROM {t} WHERE gid=?", (gid,))
                except sqlite3.OperationalError:
                    pass
            c.commit()
            self.gallery.pop(gid, None)

    def _log_sighting(self, gid, camera, ts, matched, score, n_obs, prev_cam,
                      track=None):
        """Durable record of every tracklet->Global-ID binding + keep the
        global_id row's last_seen fresh (the in-memory gallery is volatile)."""
        with self.lock, closing(self._db()) as c:
            c.execute("INSERT INTO sighting(gid,camera,ts,matched,score,n_obs,"
                      "prev_cam,track) VALUES(?,?,?,?,?,?,?,?)",
                      (gid, camera, ts, int(matched),
                       None if score is None else float(score), n_obs, prev_cam,
                       track))
            c.execute("UPDATE global_id SET last_seen_ts=?,last_cam=? WHERE gid=?",
                      (ts, camera, gid))
            c.commit()
        if ch is not None:
            ch.sighting(gid, camera, ts, matched, score, n_obs, prev_cam, track)

    def _attr_ok(self, a, b):
        """Tag VETO: never merge two tracklets whose VLM tags POSITIVELY
        contradict — a clear man/woman conflict, or garments with nothing in
        common in the same lighting — no matter how close the vector says they
        are. The vector is a black box the owner cannot audit; the legible tags
        overrule it on a contradiction. Missing/uncertain tags never veto (then
        the vector decides). `tags_verdict` already encodes the day/night rule."""
        return providers.tags_verdict(a, b) != "DIFFERENT"

    def _new_gid(self, vec, camera, ts, attrs, key=None):
        gid = self._next_gid
        self._next_gid += 1
        self.gallery[gid] = {"ema": vec.copy(), "bank": [vec.copy()],
                             "bank_keys": [key],
                             "last_cam": camera, "last_ts": ts,
                             "attrs": dict(attrs or {}), "n_obs": 1,
                             "busy_key": None, "busy_ts": ts}
        c = self._db()
        c.execute("INSERT INTO global_id(gid,created_ts,last_seen_ts,last_cam,"
                  "attrs) VALUES(?,?,?,?,?)",
                  (gid, ts, ts, camera, json.dumps(attrs or {})))
        c.commit()
        c.close()
        self._persist(gid)
        return gid

    def _touch(self, gid, camera, ts):
        g = self.gallery.get(gid)
        if not g:
            return
        g["last_cam"] = camera
        g["last_ts"] = ts
        g["n_obs"] += 1

    def _bank_update(self, gid, vec, key=None):
        """UPSERT this tracklet's centroid into the identity's set.

        One tracklet contributes exactly ONE vector, refined as the tracklet
        grows — not one vector per crop. The burst gate fires a crop every
        0.12 s, so per-crop entries filled the bank with 30 near-identical
        frames of a single 3-second episode instead of 30 real viewpoints.
        """
        g = self.gallery.get(gid)
        if not g:
            return
        keys = g.setdefault("bank_keys", [None] * len(g["bank"]))
        if key is not None and key in keys:
            g["bank"][keys.index(key)] = vec.copy()      # refine in place
        else:
            # ADMISSION BY DIVERSITY: a view that already exists in the bank adds
            # no evidence, it only dilutes the vote and gives the max one more
            # lottery ticket. The tracker says this is the same person; the bank
            # only wants the pictures that LOOK different.
            if g["bank"] and g["bank"][0].shape[0] == vec.shape[0] and \
                    float((np.stack(g["bank"]) @ vec).max()) >= \
                    float(self.cfg["div_thresh"]):
                return
            g["bank"].append(vec.copy())
            keys.append(key)
            if len(g["bank"]) > int(self.cfg["bank_size"]):
                B = np.stack(g["bank"])
                S = B @ B.T
                np.fill_diagonal(S, -1.0)
                drop = int(np.argmax(S.max(axis=1)))     # most redundant view
                if drop == len(g["bank"]) - 1:           # never the newest
                    drop = int(np.argmax(S[:-1].max(axis=1)))
                g["bank"].pop(drop)
                keys.pop(drop)
        m = np.stack(g["bank"]).mean(0)                  # gid_vec compat only
        n = np.linalg.norm(m)
        g["ema"] = m / n if n > 0 else m

    # ---- lifecycle ---------------------------------------------------------
    def sweep(self, now_ms=None):
        """Finalise idle tracklets and evict absent Global IDs. Call periodically."""
        now = now_ms or int(time.time() * 1000)
        with self.lock:
            fin = self.cfg["idle_finalize_s"] * 1000
            for key, tl in list(self.tracklets.items()):
                if now - tl["last_ts"] < fin:
                    continue
                if tl["gid"] is None and tl["vecs"]:
                    tl["gid"] = self._match_and_assign(tl, tl["cam"],
                                                       tl["last_ts"], key)
                g = self.gallery.get(tl["gid"]) if tl["gid"] is not None else None
                if g is not None and g.get("busy_key") == key:
                    g["busy_key"] = None          # tracklet ended -> release
                del self.tracklets[key]
            forget = self.cfg["forget_h"] * 3600 * 1000
            for gid, g in list(self.gallery.items()):
                if now - g["last_ts"] > forget:
                    del self.gallery[gid]      # finally forgotten

    def journey_append(self, gid, camera, ts, snapshot=None, caption=None):
        with self.lock, closing(self._db()) as c:
            c.execute("INSERT INTO journey(gid,ts,camera,exit_snapshot,caption) "
                      "VALUES(?,?,?,?,?)", (gid, ts, camera, snapshot, caption))
            c.execute("UPDATE global_id SET last_seen_ts=?,last_cam=? WHERE gid=?",
                      (ts, camera, gid))
            c.commit()

    def stats(self):
        with self.lock:
            return {"active_ids": len(self.gallery),
                    "active_tracklets": len(self.tracklets),
                    "next_gid": self._next_gid}
