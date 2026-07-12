"""MIGraphX plate detector — MIT-licensed YOLOv9-t end2end model.

Uses the open-image-models YOLOv9-t-640-license-plate-end2end ONNX (MIT license).
The model has NMS baked in, so the output is already filtered:
    [N, 7]: batch_idx, x1, y1, x2, y2, class_id, confidence

Runs on MIGraphX (AMD GPU) when available (~2ms/frame), falls back to CPU (~100ms).
"""
import os
import numpy as np


class MigraphxYOLO:
    SIZE = 640

    def __init__(self, onnx_path, conf=0.30):
        import onnxruntime as ort
        self.conf = conf
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(onnx_path, so, providers=[
            "MIGraphXExecutionProvider", "CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.on_gpu = self.sess.get_providers()[0] == "MIGraphXExecutionProvider"
        if self.on_gpu:                      # compile the fixed shape now
            self.sess.run(None, {self.inp: np.zeros((1, 3, self.SIZE, self.SIZE),
                                                    np.float32)})

    def _letterbox(self, img):
        """Resize keeping aspect ratio, pad to SIZE×SIZE.
        Returns (blob, ratio_w, ratio_h, pad_w, pad_h)."""
        import cv2
        h, w = img.shape[:2]
        r = min(self.SIZE / h, self.SIZE / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.SIZE, self.SIZE, 3), 114, np.uint8)
        pad_w, pad_h = (self.SIZE - nw) // 2, (self.SIZE - nh) // 2
        canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
        # BGR->RGB, HWC->CHW, /255
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[None], r, pad_w, pad_h

    def predict(self, crop):
        """crop: HxWx3 BGR uint8 -> [(x1,y1,x2,y2,conf), ...] in crop pixels."""
        if crop is None or crop.size == 0:
            return []
        blob, r, pad_w, pad_h = self._letterbox(crop)
        out = self.sess.run(None, {self.inp: blob})[0]   # [N, 7]

        if len(out) == 0:
            return []

        # End2end output: [batch_idx, x1, y1, x2, y2, class_id, confidence]
        scores = out[:, 6]
        keep = scores >= self.conf
        out = out[keep]
        if len(out) == 0:
            return []

        h, w = crop.shape[:2]
        res = []
        for row in out:
            # Unletterbox: coords are in the 640×640 padded space
            x1 = (row[1] - pad_w) / r
            y1 = (row[2] - pad_h) / r
            x2 = (row[3] - pad_w) / r
            y2 = (row[4] - pad_h) / r
            # Clip to image bounds
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            res.append((int(x1), int(y1), int(x2), int(y2), float(row[6])))
        return res
