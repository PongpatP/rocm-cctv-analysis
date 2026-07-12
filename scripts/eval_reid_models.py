"""Measure how well each candidate embedding separates PEOPLE, on our cameras.

Protocol (no manual labels needed):
  positive pair = two crops of the SAME (camera, track)          -> same person
  negative pair = two crops of DIFFERENT tracks on the SAME camera
                  within 2 s of each other                       -> two people
                  co-present in the frame, so they cannot be the same person
Score = cosine similarity of L2-normalised embeddings. Reported as ROC-AUC plus
the accept rates at candidate thresholds. Higher AUC = a usable ReID embedding.
"""
import glob
import itertools
import os
import random

import numpy as np
from PIL import Image

random.seed(0)
np.random.seed(0)
DS = "/output/siglip/reid_ds"
MODELS = "/output/siglip/reid_models"
MAX_PAIRS = 6000


def load_index():
    items = []                       # (path, camera, track, ts)
    for p in glob.glob(f"{DS}/*/*.jpg"):
        cam = p.split("/")[-2]
        base = os.path.basename(p)[:-4]
        trk, ts = base.split("_")
        items.append((p, cam, int(trk), int(ts)))
    return items


def build_pairs(items):
    by_id, by_cam = {}, {}
    for i, (p, cam, trk, ts) in enumerate(items):
        by_id.setdefault((cam, trk), []).append(i)
        by_cam.setdefault(cam, []).append(i)
    pos = []
    for idx in by_id.values():
        if len(idx) < 2:
            continue
        for a, b in itertools.combinations(idx, 2):
            pos.append((a, b))
    random.shuffle(pos); pos = pos[:MAX_PAIRS]
    neg = []
    for cam, idx in by_cam.items():
        idx.sort(key=lambda i: items[i][3])
        for a in range(len(idx)):
            for b in range(a + 1, min(a + 40, len(idx))):
                i, j = idx[a], idx[b]
                if items[j][3] - items[i][3] > 2000:
                    break
                if items[i][2] != items[j][2]:
                    neg.append((i, j))
    random.shuffle(neg); neg = neg[:MAX_PAIRS]
    return pos, neg


def auc_of(sim_pos, sim_neg):
    allv = np.concatenate([sim_pos, sim_neg])
    lab = np.concatenate([np.ones(len(sim_pos)), np.zeros(len(sim_neg))])
    r = np.argsort(np.argsort(allv)) + 1
    return (r[lab == 1].sum() - len(sim_pos) * (len(sim_pos) + 1) / 2) \
        / (len(sim_pos) * len(sim_neg))


def report(name, V, pos, neg):
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    sp = np.array([float(V[a] @ V[b]) for a, b in pos])
    sn = np.array([float(V[a] @ V[b]) for a, b in neg])
    a = auc_of(sp, sn)
    print(f"\n### {name}   dim={V.shape[1]}   AUC = {a:.3f}")
    print(f"    same-person  mean={sp.mean():.3f}  p05={np.percentile(sp,5):.3f}")
    print(f"    diff-person  mean={sn.mean():.3f}  p95={np.percentile(sn,95):.3f}")
    best = None
    for t in np.arange(0.05, 1.0, 0.025):
        tp = (sp >= t).mean(); fp = (sn >= t).mean()
        if tp + fp == 0:
            continue
        prec = tp / (tp + fp)
        f1 = 2 * prec * tp / (prec + tp) if prec + tp else 0
        if best is None or f1 > best[0]:
            best = (f1, t, tp, fp, prec)
    if best:
        f1, t, tp, fp, prec = best
        print(f"    best operating point: thr={t:.3f}  recall={tp*100:.1f}%  "
              f"false-accept={fp*100:.1f}%  precision={prec*100:.1f}%")
    return a


def embed_onnx(path, items, rgb=True, batch=16, fixed=False):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    iname = s.get_inputs()[0].name
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    out = []
    buf = []
    def flush():
        if not buf:
            return
        x = np.stack(buf)
        n = len(buf)
        if fixed and n < batch:                     # osnet has a fixed batch dim
            x = np.concatenate([x, np.zeros((batch - n, *x.shape[1:]), np.float32)])
        y = s.run(None, {iname: x})[0]
        y = y.reshape(y.shape[0], -1)[:n]
        out.append(y)
        buf.clear()
    for p, *_ in items:
        im = Image.open(p).convert("RGB").resize((128, 256))
        a = np.asarray(im, np.float32) / 255.0
        if not rgb:
            a = a[:, :, ::-1].copy()
        a = (a - mean) / std
        buf.append(a.transpose(2, 0, 1))
        if len(buf) == batch:
            flush()
    flush()
    return np.concatenate(out).astype(np.float32)


def embed_siglip(items):
    import torch
    from transformers import AutoModel, AutoProcessor
    m = "google/siglip2-so400m-patch16-384"
    proc = AutoProcessor.from_pretrained(m)
    model = AutoModel.from_pretrained(m, torch_dtype=torch.float16).to("cpu").eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(items), 32):
            ims = [Image.open(p).convert("RGB") for p, *_ in items[i:i + 32]]
            x = proc(images=ims, return_tensors="pt").to("cpu")
            f = model.get_image_features(**x)
            if not torch.is_tensor(f):
                f = getattr(f, "image_embeds", None) or f.pooler_output
            out.append(f.float().cpu().numpy())
    del model
    return np.concatenate(out).astype(np.float32)


def main():
    items = load_index()
    print(f"crops={len(items):,}  identities={len({(c,t) for _,c,t,_ in items}):,}")
    pos, neg = build_pairs(items)
    print(f"positive pairs={len(pos):,}  negative pairs={len(neg):,}")

    res = {}
    res["SigLIP2-so400m (current)"] = report(
        "SigLIP2-so400m (current)", embed_siglip(items), pos, neg)
    res["OSNet-x0.25 MSMT17 (MIT)"] = report(
        "OSNet-x0.25 MSMT17 (MIT)",
        embed_onnx(f"{MODELS}/osnet_x0_25_msmt17.onnx", items, rgb=True, fixed=True),
        pos, neg)
    for rgb in (True, False):
        tag = "RGB" if rgb else "BGR"
        res[f"YoutuReID (Apache-2.0) {tag}"] = report(
            f"YoutuReID (Apache-2.0) {tag}",
            embed_onnx(f"{MODELS}/youtureid.onnx", items, rgb=rgb, batch=32),
            pos, neg)

    print("\n" + "=" * 60)
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  AUC {v:.3f}   {k}")


if __name__ == "__main__":
    main()
