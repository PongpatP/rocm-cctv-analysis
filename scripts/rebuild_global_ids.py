"""Rebuild Global IDs from scratch: local ReID -> global vectors. No LLM, no VLM.

Pipeline
  0. DROP every tracklet the person gate refused. `track_check` holds one VLM
     verdict per (camera, track): a rolled mat, a bollard, a pair of legs and the
     back of a head are not people and must never enter a Global ID. This is the
     same table the live gate writes, so offline and online agree by construction.
  1. Every stored crop is a (camera, track_id, ts) sample. The tracker id IS the
     local ReID, so a tracklet = one person within one camera.
  2. DROP any crop that contains more than one person. This is done WITHOUT a
     VLM: the detector already stored every person box in stats.db:det_raw, so
     a crop is rejected when another person's box overlaps its own box.
  3. Embed the surviving crops (GPU 0) with the selected model.
  4. Subtract the GLOBAL MEAN feature. CLIP-family embeddings share a large
     common direction — two strangers score cosine ~0.71 raw — so without this
     every threshold is meaningless. Measured, not assumed.
  5. REPLAY the crops through the live matcher (`siglip.reid.ReID`) in
     chronological order, into a throw-away database. Not a second clustering
     algorithm: the same code that runs live, so the page cannot disagree with
     the pipeline. Majority vote, diversity-gated bank, reachability gate,
     same-camera exclusivity — all of it, for free.
  6. Print the resulting people-count at several thresholds. A human picks by
     looking at the crops, not at a score: there is no cross-camera ground truth.

    docker compose exec siglip python3 /app/rebuild.py --model clipreid
    docker compose exec siglip python3 /app/rebuild.py --model clipreid --thr .35 --write
"""
import argparse
import collections
import datetime as dt
import glob
import os
import shutil
import sqlite3
import sys
import tempfile

import numpy as np
import torch          # noqa: F401  — kept for its bundled runtime libraries
from PIL import Image

sys.path.insert(0, "/app")
from reid import ReID           # the live matcher, verbatim

CROPS = "/output/siglip/crops"
REID_DB = "/output/siglip/reid.db"
EMB_DB = "/output/siglip/embeddings.db"
STATS_DB = "/output/stats.db"
MODELS = {"youtureid": "/cache/reid/youtureid.onnx",
          "clipreid": "/cache/reid/clipreid.onnx"}
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)

idx_ts = {}


def iou(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def load_crops(cut_ms):
    """(path, camera, track, ts) for every crop newer than the cut."""
    out = []
    for p in glob.glob(f"{CROPS}/*/*.jpg"):
        cam = p.split("/")[-2]
        base = os.path.basename(p)[:-4]
        try:
            trk, ts = base.split("_")
            ts = int(ts)
        except ValueError:
            continue
        if ts >= cut_ms:
            out.append((p, cam, int(trk), ts))
    return out


def gate_verdicts():
    """(camera, track) -> 'person' | 'reject', as judged once by the VLM."""
    c = sqlite3.connect(f"file:{REID_DB}?mode=ro", uri=True)
    try:
        d = {(cam, trk): v for cam, trk, v in
             c.execute("SELECT camera, track, verdict FROM track_check")}
    except sqlite3.OperationalError:
        d = {}
    c.close()
    return d


def own_boxes():
    """(camera, track, ts) -> the box this crop was cut from, normalised."""
    c = sqlite3.connect(EMB_DB)
    d = {}
    for cam, trk, ts, x, y, w, h in c.execute(
            "SELECT camera,track,ts,x,y,w,h FROM embeddings WHERE track IS NOT NULL"):
        d[(cam, trk, ts)] = (x, y, w, h)
    c.close()
    return d


def crowd_index(cut_ms):
    """camera -> sorted [(ts, x,y,w,h)] of every person box, normalised.

    Boxes live in ClickHouse now (`ccvt.detections`); the old sqlite `det_raw`
    stopped being written when the bridge was ported. Fall back to it only for
    data older than the switch.
    """
    idx = collections.defaultdict(list)
    try:
        sys.path.insert(0, "/app")
        import ch
        # ts_ms, not ts: an alias named after the column shadows it in WHERE.
        rows = ch.query(
            "SELECT toUnixTimestamp64Milli(ts) AS ts_ms, camera, x, y, w, h, fw, fh "
            "FROM detections WHERE class = 'person' "
            "AND ts > fromUnixTimestamp64Milli({cut:Int64})",
            {"param_cut": str(cut_ms - 5000)})
        for r in rows:
            if not r["fw"] or not r["fh"]:
                continue
            idx[r["camera"]].append((int(r["ts_ms"]), r["x"] / r["fw"], r["y"] / r["fh"],
                                     r["w"] / r["fw"], r["h"] / r["fh"]))
        print(f"  crowd index: {sum(len(v) for v in idx.values()):,} person boxes "
              f"from ClickHouse", flush=True)
    except Exception as e:
        print(f"  clickhouse unavailable ({e}); falling back to sqlite det_raw",
              flush=True)
        c = sqlite3.connect(STATS_DB)
        for ts, cam, x, y, w, h, fw, fh in c.execute(
                "SELECT ts,camera,x,y,w,h,fw,fh FROM det_raw "
                "WHERE class='person' AND ts>?", (cut_ms - 5000,)):
            if not fw or not fh:
                continue
            idx[cam].append((ts, x / fw, y / fh, w / fw, h / fh))
        c.close()
    for cam in idx:
        idx[cam].sort()
    return idx


def is_single_person(cam, ts, own, idx, iou_thr, window=250):
    """True when no OTHER person box overlaps the crop's own box at that moment.

    The crop's own box is also in det_raw, and rounding means it never compares
    equal — so identify it as the single highest-IoU box and judge the SECOND
    highest. That box is a different person standing inside this crop.
    """
    if own is None:
        return True
    rows = idx.get(cam)
    if not rows:
        return True
    ts_list = idx_ts.setdefault(cam, [r[0] for r in rows])
    lo = np.searchsorted(ts_list, ts - window)
    hi = np.searchsorted(ts_list, ts + window)
    ious = sorted((iou(own, r[1:]) for r in rows[lo:hi]), reverse=True)
    if len(ious) < 2:
        return True
    return ious[1] <= iou_thr


def _load(p):
    a = np.asarray(Image.open(p).convert("RGB").resize((128, 256)), np.float32) / 255.0
    return ((a - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1)


def embed(paths, model, batch=64):
    """JPEG decode, not the GPU, is the bottleneck here (0.44 ms/crop on GPU vs
    ~3 ms/crop to decode), so decode on a thread pool — PIL drops the GIL."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    import onnxruntime as ort
    prov = [("MIGraphXExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    sess = ort.InferenceSession(MODELS[model], providers=prov)
    name = sess.get_inputs()[0].name
    print(f"  {model} on {sess.get_providers()[0]}", flush=True)
    out, t_io, t_gpu, t0 = [], 0.0, 0.0, time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i in range(0, len(paths), batch):
            t = time.time()
            X = np.stack(list(ex.map(_load, paths[i:i + batch])))
            t_io += time.time() - t
            t = time.time()
            y = sess.run(None, {name: X})[0]
            t_gpu += time.time() - t
            out.append(y.reshape(y.shape[0], -1).astype(np.float32))
            if (i // batch) % 40 == 0:
                done = i + len(X)
                el = time.time() - t0
                print(f"    {done}/{len(paths)}  {done/max(el,1e-6):.0f} crops/s  "
                      f"(decode {t_io:.0f}s, gpu {t_gpu:.0f}s)", flush=True)
    E = np.concatenate(out)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def replay(samples, E, thr, cfg_over):
    """Feed every crop, in time order, to a fresh copy of the live matcher."""
    tmp = tempfile.mktemp(suffix=".db")
    shutil.copy2(REID_DB, tmp)            # keep the scene graph (reachability!)
    c = sqlite3.connect(tmp)
    for t in ("sighting", "global_id", "gid_vec", "person", "journey"):
        try:
            c.execute(f"DELETE FROM {t}")
        except sqlite3.OperationalError:
            pass
    c.commit(); c.close()

    r = ReID(tmp)
    r.cfg.update(cfg_over)
    r.cfg["match_thresh"] = thr
    r.cfg["same_cam_thresh"] = thr - 0.12     # same gap as the live config (.67/.55)
    # the exact global mean, instead of the running estimate the live path builds
    r.cam_mean[r.GLOBAL] = E.mean(0)
    # The live service sweeps on a timer. Replaying without a clock would leave
    # every tracklet open, so each one would hold its same-camera exclusivity
    # claim forever and nothing after it could match. Advance the clock with the
    # data: sweep whenever a second of recorded time has passed.
    last_sweep = samples[0][3]
    for (p, cam, trk, ts), v in zip(samples, E):
        if ts - last_sweep >= 1000:
            r.sweep(now_ms=ts)
            last_sweep = ts
        r.observe(cam, trk, v, 1.0, ts=ts)
    r.sweep(now_ms=samples[-1][3] + int(r.cfg["idle_finalize_s"] * 1000) + 1)
    return r, tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="clipreid", choices=list(MODELS))
    ap.add_argument("--since", default="00:00")
    ap.add_argument("--iou", type=float, default=0.10)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--min-crops", type=int, default=2)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    hh, mm = map(int, a.since.split(":"))
    today = dt.datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    cut = int(today.timestamp() * 1000)

    crops = load_crops(cut)
    print(f"crops after {a.since}: {len(crops):,}", flush=True)

    gate = gate_verdicts()
    if gate:
        keep = [c for c in crops if gate.get((c[1], c[2]), "unchecked") == "person"]
        judged = sum(1 for c in crops if (c[1], c[2]) in gate)
        print(f"  person gate: {len(gate):,} tracks judged · kept {len(keep):,} crops "
              f"of {judged:,} judged ({len(crops)-judged:,} crops from unchecked tracks "
              f"dropped)", flush=True)
        crops = keep
    else:
        print("  WARNING: track_check is empty — run verify_tracks.py first, or "
              "non-persons will be rebuilt back into identities", flush=True)

    own, idx = own_boxes(), crowd_index(cut)
    clean = [c for c in crops
             if is_single_person(c[1], c[3], own.get((c[1], c[2], c[3])), idx, a.iou)]
    print(f"  single-person crops: {len(clean):,}  "
          f"(dropped {len(crops)-len(clean):,} containing another person)", flush=True)

    by = collections.defaultdict(list)
    for p, cam, trk, ts in clean:
        by[(cam, trk)].append((ts, p))
    keep = {k for k, v in by.items() if len(v) >= a.min_crops}
    samples = sorted((c for c in clean if (c[1], c[2]) in keep), key=lambda c: c[3])
    print(f"  tracklets (local ReID): {len(keep):,} · crops replayed: {len(samples):,}",
          flush=True)
    del own, idx, crops, clean, by      # det_raw for a whole day: gigabytes
    idx_ts.clear()

    print(f"  embedding {len(samples):,} crops…", flush=True)
    E = embed([s[0] for s in samples], a.model)
    cfg = {"global_center": True}

    thrs = [a.thr] if a.thr is not None else [0.25, 0.30, 0.35, 0.40, 0.45]
    for t in thrs:
        r, tmp = replay(samples, E, t, cfg)
        gids = r.gallery
        cams = collections.defaultdict(set)
        c = sqlite3.connect(tmp)
        for g, cam in c.execute("SELECT gid, camera FROM sighting"):
            cams[g].add(cam)
        c.close()
        multi = sum(1 for g in cams.values() if len(g) > 1)
        banks = [len(g["bank"]) for g in gids.values()]
        print(f"  thr={t:.2f} -> {len(gids):5d} people  ({multi} on >1 camera)  "
              f"median bank={int(np.median(banks)) if banks else 0}", flush=True)
        if a.thr is None or not a.write:
            os.unlink(tmp)
        else:
            keep_tmp = tmp

    if not a.write:
        print("\n(dry run — pick a --thr, then re-run with --write)")
        return
    if a.thr is None:
        print("\n--write needs an explicit --thr")
        return
    bak = f"{REID_DB}.bak-{int(dt.datetime.now().timestamp())}"
    shutil.copy2(REID_DB, bak)
    shutil.move(keep_tmp, REID_DB)
    print(f"\nreid.db rebuilt with {a.model} (old copy: {bak})", flush=True)


if __name__ == "__main__":
    main()
