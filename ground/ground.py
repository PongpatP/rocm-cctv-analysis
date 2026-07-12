"""Open-vocabulary visual grounding: text -> bounding boxes on a CCTV frame.

The rule this serves (owner's): whenever a user asks WHERE something is, an AI
must draw the box. The RT-DETR detector boxes its 5 COCO classes; everything
else lands here. Grounding DINO (IDEA-Research, Apache-2.0) takes any English
phrase ("fire extinguisher", "blue bag on the floor") and returns boxes.

SAM 3 itself was considered and rejected: it only ships through ultralytics,
which is AGPL-3.0 — the same license this project already banned for YOLO26.
Grounding DINO gives the same ask-in-text/point-at-pixels capability under
Apache-2.0, via plain HF transformers.

CPU on purpose: one frame per chat question, ~2-4 s on this 20-vCPU box, and
it never competes with detection or the VLM for the GPU.
"""
import base64
import io
import logging
import os
import threading
import time

from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("ground")

MODEL_ID = os.environ.get("GROUND_MODEL", "IDEA-Research/grounding-dino-tiny")

app = FastAPI()
_state = {"ready": False, "error": ""}
_lock = threading.Lock()          # one inference at a time — small CPU model


def _load():
    try:
        import torch  # noqa: F401
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        t0 = time.time()
        _state["processor"] = AutoProcessor.from_pretrained(MODEL_ID)
        _state["model"] = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
        _state["model"].eval()
        _state["ready"] = True
        log.info("loaded %s in %.0fs", MODEL_ID, time.time() - t0)
    except Exception as e:
        _state["error"] = str(e)[:300]
        log.error("model load failed: %s", e)


threading.Thread(target=_load, daemon=True).start()


class GroundReq(BaseModel):
    img: str                      # base64 JPEG
    query: str                    # English phrase(s); "a dog. a red bag."
    threshold: float = 0.25
    limit: int = 8


@app.get("/healthz")
def healthz():
    return {"ready": _state["ready"], "model": MODEL_ID, "error": _state["error"]}


@app.post("/ground")
def ground(r: GroundReq):
    if not _state["ready"]:
        return {"ok": False, "error": _state["error"] or "model still loading"}
    import torch
    try:
        img = Image.open(io.BytesIO(base64.b64decode(r.img))).convert("RGB")
    except Exception:
        return {"ok": False, "error": "bad image"}
    # DINO wants lowercase phrases, each terminated with a period
    text = ". ".join(p.strip().lower().rstrip(".")
                     for p in r.query.split(",") if p.strip()) + "."
    t0 = time.time()
    with _lock, torch.no_grad():
        proc, model = _state["processor"], _state["model"]
        inputs = proc(images=img, text=text, return_tensors="pt")
        out = model(**inputs)
        res = proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=r.threshold,
            text_threshold=r.threshold,
            target_sizes=[img.size[::-1]])[0]
    boxes = []
    for score, label, box in sorted(
            zip(res["scores"].tolist(),
                res.get("text_labels", res.get("labels")),
                res["boxes"].tolist()),
            key=lambda x: -x[0])[: r.limit]:
        x1, y1, x2, y2 = [round(v, 1) for v in box]
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "score": round(float(score), 3), "label": str(label)})
    return {"ok": True, "boxes": boxes, "width": img.size[0],
            "height": img.size[1], "ms": int((time.time() - t0) * 1000)}
