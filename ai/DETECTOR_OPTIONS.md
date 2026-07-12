# Detector options — why RT-DETR R50, and the alternatives

**Why this doc:** the deployed detector is **RT-DETR R50** (Baidu,
`lyuwenyu/RT-DETR`), **Apache-2.0**, run as ONNX via
**onnxruntime-migraphx** on the AMD MI300X. This records why it was chosen,
the **commercially-free (Apache-2.0 / MIT)** alternatives that were on the
table, what they cost in **speed** and **accuracy** (especially for our 5
classes), and how to swap the detector if we ever want to.

> Our reality: the detector runs on the MI300X at 640×640, 28 cameras, and
> we keep only **5 COCO classes** — `person, car, truck, motorcycle,
> bicycle`. All models below are **COCO-pretrained**, so all already know
> these 5 classes; no retraining needed to switch.

---

## TL;DR — the two questions

**Is a DETR slower than a CNN detector?** — *Slightly.* Transformer
detectors do more compute per frame than a CNN detector of the same
accuracy, so on the same GPU you should expect **somewhat lower FPS** (often
0.6–0.9× of a CNN detector at matched accuracy). It is still comfortably
real-time on the MI300X. **The only honest number comes from benchmarking on
our own box** — plan for a modest speed cost, not a cliff.

**Does accuracy drop — for our 5 classes?** — *No meaningful drop; likely a
wash or slightly better.* All candidates score **~50–54 mAP on COCO**. Our 5
classes are **large, common objects** that every COCO model detects well.
DETR-family models are **NMS-free** and tend to be *better in
crowded/overlapping scenes* — a plus for CCTV corridors. Where a CNN
detector can still edge ahead is **very small / far-away** objects.

---

## The candidates — benchmark table

Machine-readable copy: **`ai/models_registry.json`**. Benchmark = **COCO
val2017 mAP@[.5:.95]** + **indicative half-precision** latency/FPS from each
project's paper/repo (mixed sources → *measure on our own GPU for the real
numbers*).

| Model | License | mAP ↑ | FPS ↑ | lat. (ms) | NMS-free | Family | Tag |
|---|---|:--:|:--:|:--:|:--:|---|---|
| **RT-DETR-R50** *(deployed)* | Apache ✅ | 53.1 | 108 | 9.3 | ✅ | DETR | 🎯 in use |
| **RT-DETR-R101** | Apache ✅ | 54.3 | 74 | 13.5 | ✅ | DETR | higher acc. |
| **D-FINE-L** | Apache ✅ | 54.0 | 116 | 8.6 | ✅ | DETR | 🏆 best acc @ speed (free) |
| **D-FINE-X** | Apache ✅ | **55.8** | 78 | 12.9 | ✅ | DETR | 🏆 top mAP |
| **RF-DETR-base** | Apache ✅ | 53.3 | 111 | 9.0 | ✅ | DETR | 2025 SOTA-ish |
| **CNN detector (Apache)** | Apache ✅ | 49.7–54.0 | 69–100 | 10–14.5 | ❌ | CNN | needs NMS |

> ⚠️ Latency figures come from **different papers/GPUs** — treat the mAP as
> solid and the FPS as *ballpark*. The only fair speed comparison is a
> side-by-side run on our own GPU.

### Verdict
- 🏆 **Most accurate (free):** D-FINE-X (55.8) → D-FINE-L (54.0)
- 🎯 **Deployed (NMS-free, strong on crowded scenes, Apache-2.0):**
  RT-DETR-R50 — chosen for its closeness to how we want the pipeline to
  behave and its clean commercial licence.
- 💰 **Licence:** every row above is **Apache-2.0**, free to ship in a
  closed-source commercial product. We deliberately avoided **AGPL** and
  other restrictive detector licences (see note below).

### Notes per family
- **RT-DETR** (Baidu) — real-time DETR, **Apache-2.0**, NMS-free, strong on
  cluttered scenes. This is what we run. Slightly heavier per frame than a
  CNN detector, but the MI300X has ample headroom.
- **D-FINE** and **RF-DETR** — newer (2024–2025) real-time DETRs that
  **beat RT-DETR** on the accuracy/speed curve and are **Apache-2.0**. Same
  export shape as RT-DETR (ONNX, NMS-free), so swapping is cheap if we want
  more accuracy.
- **CNN detectors** — mature, Apache-2.0 options exist and integrate
  easily, but they **need NMS** and score a touch lower mAP. A safe fallback
  if we ever want minimal engineering; some Apache-2.0 CNN detectors also
  need an extra ONNX-export step depending on origin.
- **Avoid for commercial-free:** any detector under **AGPL** or otherwise
  restrictive licences — they require a paid licence to ship closed-source.
  Everything in the table above is clean.

---

## Recommendation (already applied)

We ship **RT-DETR-R50** — Apache-2.0, NMS-free, ~same accuracy as any CNN
detector in its class, and better in crowded corridors. If we later want
best free accuracy/speed, **D-FINE-L** or **RF-DETR** are the drop-in
upgrades (newer, Apache-2.0, ahead of RT-DETR on the curve).

For our 5 big/common classes, **any of these keeps accuracy essentially the
same**; the real trade is *engineering effort* + *a bit of FPS*.

---

## Classes — what these models can detect (COCO-80)

**All candidates are COCO-pretrained, so they all detect the SAME 80
classes.** Switching the model does **not** change the class list — it only
changes speed/accuracy. We currently keep **5** via `class_filter`; the
other 75 are already detected and just filtered out. To use more, add them
to `class_filter` in `config.local.yaml` and re-run `configure.py`.

`✅` = active now · `⭐` = CCTV-useful, easy to add.

| # | class | | # | class | | # | class | | # | class |
|--|--|--|--|--|--|--|--|--|--|--|
| 0 | ✅ person | | 20 | elephant | | 40 | wine glass | | 60 | dining table |
| 1 | ✅ bicycle | | 21 | bear | | 41 | cup | | 61 | toilet |
| 2 | ✅ car | | 22 | zebra | | 42 | fork | | 62 | tv |
| 3 | ✅ motorcycle | | 23 | giraffe | | 43 | ⭐ knife | | 63 | laptop |
| 4 | airplane | | 24 | ⭐ backpack | | 44 | spoon | | 64 | mouse |
| 5 | ⭐ bus | | 25 | ⭐ umbrella | | 45 | bowl | | 65 | remote |
| 6 | train | | 26 | ⭐ handbag | | 46 | banana | | 66 | keyboard |
| 7 | ✅ truck | | 27 | tie | | 47 | apple | | 67 | ⭐ cell phone |
| 8 | boat | | 28 | ⭐ suitcase | | 48 | sandwich | | 68 | microwave |
| 9 | traffic light | | 29 | frisbee | | 49 | orange | | 69 | oven |
| 10 | fire hydrant | | 30 | skis | | 50 | broccoli | | 70 | toaster |
| 11 | stop sign | | 31 | snowboard | | 51 | carrot | | 71 | sink |
| 12 | parking meter | | 32 | sports ball | | 52 | hot dog | | 72 | refrigerator |
| 13 | bench | | 33 | kite | | 53 | pizza | | 73 | book |
| 14 | bird | | 34 | baseball bat | | 54 | donut | | 74 | clock |
| 15 | ⭐ cat | | 35 | baseball glove | | 55 | cake | | 75 | vase |
| 16 | ⭐ dog | | 36 | skateboard | | 56 | ⭐ chair | | 76 | ⭐ scissors |
| 17 | horse | | 37 | surfboard | | 57 | couch | | 77 | teddy bear |
| 18 | sheep | | 38 | tennis racket | | 58 | potted plant | | 78 | hair drier |
| 19 | cow | | 39 | ⭐ bottle | | 59 | bed | | 79 | toothbrush |

**Want more than 80 / custom classes** (e.g. *weapon, parcel, helmet, fire,
fallen-person*)? COCO models can't — that needs **fine-tuning** on a custom
dataset, or an **open-vocabulary** model. The local **VLM** (Google Gemma 4
31B, served by vLLM and driven by the `vlmscan` service) can also describe
anything in a scene. So: RT-DETR/CNN detectors = fast on the 80 fixed
classes; the VLM = slower but *any* concept.

---

## Swapping the detector (how the pipeline is built)

The detector lives in **`ai_rocm/airocm.py`** and loads a single ONNX model
through **onnxruntime-migraphx** on the MI300X. The model path is the
`RTDETR_ONNX` environment variable (see `docker-compose.override.yml`).
There is no separate compiled engine to build by hand — onnxruntime-migraphx
compiles a MIGraphX cache (`.mxr`) on first boot and reuses it after.

To try a different detector:

1. **Weights → ONNX:** export the checkpoint (Apache-2.0) to ONNX at
   640×640. For DETR-family models (RT-DETR, D-FINE, RF-DETR) the output is
   `[num_queries, 4 + num_classes]` (boxes + class logits, **no NMS**),
   which is what the current post-processing expects. A CNN detector's
   output shape differs and needs NMS in the post-processing step.
2. **Point the detector at it:** set `RTDETR_ONNX` to the new file under
   `ai/models/` and restart `airocm` — it recompiles the MIGraphX cache on
   the first boot (a few minutes) and loads from cache after.
3. **Keep the same `class_filter`** — the 5 classes map to the same COCO
   indices, so the rest of the stack (bridge, overlays, ReID, twin) is
   unchanged.
4. **Benchmark both** side-by-side on the MI300X (FPS + a few clips) →
   decide.

Flipping between DETR-family models is essentially a one-file swap; a CNN
detector additionally needs an NMS step in the post-processing.
