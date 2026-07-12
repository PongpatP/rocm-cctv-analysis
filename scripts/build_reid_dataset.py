"""Build a labelled person-ReID dataset from the recordings + stored detector boxes.

The recorder keeps 60 s sub-stream segments per camera (352x288 @15 fps, named
<YYYYmmdd_HHMMSS>.mp4) and the bridge stores every detection in
output/stats.db:det_raw (ts, camera, class, conf, track, x,y,w,h, fw,fh).
A tracker id IS an identity label, so the two together give a free, large,
in-domain ReID dataset — no manual annotation.

Crops are written to output/siglip/reid_ds/<camera>/<track>_<ts>.jpg.
CPU only. Boxes that overlap another person (IoU > IOU_THR) are skipped, as are
boxes too small to carry appearance.
"""
import argparse
import datetime as dt
import glob
import os
import sqlite3
from collections import defaultdict
from multiprocessing import Pool

import cv2

ROOT = "/workspace/ccvt_stream_ai"
DB = f"{ROOT}/output/stats.db"
REC = f"{ROOT}/recordings"
OUT = f"{ROOT}/output/siglip/reid_ds"
FPS = 15.0
MIN_W, MIN_H = 16, 36          # source pixels — below this there is no appearance
IOU_THR = 0.2                  # same rule as the live gate: overlapping = mixed crop
MARGIN = 0.08


def iou(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def seg_start(path):
    return dt.datetime.strptime(os.path.basename(path)[:-4], "%Y%m%d_%H%M%S").timestamp()


def do_segment(job):
    """Decode one segment once, crop every wanted (frame, box)."""
    cam, path, rows = job                       # rows: (ts, track, x,y,w,h, fw,fh)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    t0 = seg_start(path)
    want = defaultdict(list)                    # frame_idx -> [(track, box_px)]
    for ts, trk, x, y, w, h, fw, fh in rows:
        fi = int(round((ts / 1000.0 - t0) * FPS))
        if not 0 <= fi < 900:
            continue
        bx = (x / fw * W, y / fh * H, w / fw * W, h / fh * H)
        want[fi].append((ts, trk, bx))
    if not want:
        cap.release()
        return 0
    last = max(want)
    outdir = os.path.join(OUT, cam)
    os.makedirs(outdir, exist_ok=True)
    n = 0
    for fi in range(last + 1):
        ok = cap.grab()
        if not ok:
            break
        if fi not in want:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        boxes = want[fi]
        for k, (ts, trk, b) in enumerate(boxes):
            if b[2] < MIN_W or b[3] < MIN_H:
                continue
            if any(iou(b, o[2]) > IOU_THR for j, o in enumerate(boxes) if j != k):
                continue                        # mixed crop: two people overlap
            mx, my = b[2] * MARGIN, b[3] * MARGIN
            L = max(0, int(b[0] - mx)); T = max(0, int(b[1] - my))
            R = min(W, int(b[0] + b[2] + mx)); B = min(H, int(b[1] + b[3] + my))
            if R - L < 8 or B - T < 16:
                continue
            cv2.imwrite(os.path.join(outdir, f"{trk}_{ts}.jpg"),
                        frame[T:B, L:R], [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", default="")        # blank = busiest N
    ap.add_argument("--top-cameras", type=int, default=8)
    ap.add_argument("--segments-per-camera", type=int, default=60)
    ap.add_argument("--per-track", type=int, default=8)
    ap.add_argument("--min-track-dets", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    c = sqlite3.connect(DB)
    if a.cameras:
        cams = a.cameras.split(",")
    else:
        cams = [r[0] for r in c.execute(
            "SELECT camera FROM det_raw WHERE class='person' AND track IS NOT NULL "
            "GROUP BY camera ORDER BY COUNT(*) DESC LIMIT ?", (a.top_cameras,))]
    print("cameras:", cams, flush=True)

    jobs = []
    for cam in cams:
        segs = sorted(glob.glob(f"{REC}/{cam}/*.mp4"))
        if not segs:
            continue
        starts = {p: seg_start(p) for p in segs}
        rows = c.execute(
            "SELECT ts,track,x,y,w,h,fw,fh FROM det_raw WHERE class='person' "
            "AND camera=? AND track IS NOT NULL ORDER BY ts", (cam,)).fetchall()
        # keep only tracks long enough to give several distinct views
        per_track = defaultdict(list)
        for r in rows:
            per_track[r[1]].append(r)
        keep = []
        for trk, rs in per_track.items():
            if len(rs) < a.min_track_dets:
                continue
            step = max(1, len(rs) // a.per_track)
            keep.extend(rs[::step][:a.per_track])
        # bucket by segment
        by_seg = defaultdict(list)
        for r in keep:
            t = r[0] / 1000.0
            p = None
            for path, st in starts.items():
                if st <= t < st + 60:
                    p = path
                    break
            if p:
                by_seg[p].append(r)
        best = sorted(by_seg.items(), key=lambda kv: -len(kv[1]))[:a.segments_per_camera]
        for path, rs in best:
            jobs.append((cam, path, rs))
        print(f"  {cam}: {len(per_track):,} tracks -> {len(keep):,} wanted crops in "
              f"{len(by_seg):,} segments, taking {len(best)}", flush=True)
    c.close()

    print(f"\ndecoding {len(jobs)} segments with {a.workers} workers…", flush=True)
    with Pool(a.workers) as pool:
        total = sum(pool.imap_unordered(do_segment, jobs))
    print(f"\nDONE: {total:,} crops -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
