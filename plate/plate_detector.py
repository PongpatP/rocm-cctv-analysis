"""MIGraphX plate detector — RF-DETR-Large (Apache-2.0).

Uses the Rickkosse/rfdetr_licences_plate_detector ONNX model, finetuned from
roboflow/rf-detr-large (Apache-2.0) for license plate detection.

Output format (DETR-style, NMS-free):
  dets:   [1, 300, 4]  — cx, cy, w, h (normalized 0-1)
  labels: [1, 300, 2]  — class logits (background, plate) → sigmoid for prob

Runs on MIGraphX GPU (~5.4ms/frame on MI300X), falls back to CPU.
Input: 768×768 RGB, ImageNet-normalized.
"""
import os
import numpy as np

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class MigraphxYOLO:
    """Drop-in replacement interface: .predict(crop) -> [(x1,y1,x2,y2,conf)]"""

    SIZE = 768

    def __init__(self, onnx_path, conf=0.30):
        import onnxruntime as ort
        self.conf = conf
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(onnx_path, so, providers=[
            "MIGraphXExecutionProvider", "CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.on_gpu = self.sess.get_providers()[0] == "MIGraphXExecutionProvider"
        if self.on_gpu:
            # Compile the fixed shape now (cached via ORT_MIGRAPHX_MODEL_CACHE_PATH)
            self.sess.run(None, {self.inp: np.zeros((1, 3, self.SIZE, self.SIZE),
                                                    np.float32)})

    def predict(self, crop):
        """crop: HxWx3 BGR uint8 -> [(x1,y1,x2,y2,conf), ...] in crop pixels."""
        if crop is None or crop.size == 0:
            return []
        h0, w0 = crop.shape[:2]
        blob = self._preprocess(crop)
        outs = self.sess.run(None, {self.inp: blob})
        dets = outs[0][0]      # [300, 4] — cx, cy, w, h (normalized)
        labels = outs[1][0]    # [300, 2] — logits (background, plate)

        # Sigmoid on the plate class (index 1)
        scores = 1.0 / (1.0 + np.exp(-labels[:, 1]))

        keep = scores >= self.conf
        dets, scores = dets[keep], scores[keep]
        if len(dets) == 0:
            return []

        results = []
        for (cx, cy, bw, bh), s in zip(dets, scores):
            x1 = max(0, (cx - bw / 2) * w0)
            y1 = max(0, (cy - bh / 2) * h0)
            x2 = min(w0, (cx + bw / 2) * w0)
            y2 = min(h0, (cy + bh / 2) * h0)
            results.append((int(x1), int(y1), int(x2), int(y2), float(s)))
        return results

    def _preprocess(self, crop):
        """BGR crop -> [1, 3, 768, 768] ImageNet-normalized float32."""
        import cv2
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.SIZE, self.SIZE), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]  # [1, 3, H, W]
        blob = (blob - _MEAN) / _STD
        return blob
