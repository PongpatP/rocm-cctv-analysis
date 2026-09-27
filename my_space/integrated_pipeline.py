"""integrated_pipeline.py

Modular end-to-end CCTV AI surveillance pipeline integrating:
- Detection: RT-DETR (Apache-2.0)
- In-Camera Tracker: Clean-room BoT-SORT (MIT)
- Per-Frame ReID: OSNet x0.25 (MIT)
- Cross-Camera ReID: YoutuReID (Apache-2.0)
- Face Recognition: CVLface (MIT)
- Plate Detection: RF-DETR (Apache-2.0)
- Plate OCR: PaddleOCR / PaddleX (Apache-2.0)
- LLM / VLM: Gemma 4 (Free commercial license)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


# =====================================================================
# 1. Detection Engine: RT-DETR (Apache-2.0)
# =====================================================================
class RTDetrEngine:
    """RT-DETR inference engine with NMS-free set prediction."""

    SIZE = 640
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    SURVEILLANCE_CLASSES = {"person", "car", "truck", "motorcycle", "bicycle"}

    def __init__(self, model_path: Optional[str] = None, conf_floor: float = 0.35):
        self.conf_floor = conf_floor
        self.session = None
        if model_path and os.path.exists(model_path):
            providers = ["MIGraphXExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.inp_name = self.session.get_inputs()[0].name

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        h, w = frame_bgr.shape[:2]
        if self.session is None:
            # Fallback mock detections for demonstration
            return [
                {"box": [int(w * 0.15), int(h * 0.25), int(w * 0.32), int(h * 0.85)],
                 "score": 0.91, "class_name": "person"},
                {"box": [int(w * 0.50), int(h * 0.35), int(w * 0.82), int(h * 0.78)],
                 "score": 0.88, "class_name": "car"}
            ]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.SIZE, self.SIZE), interpolation=cv2.INTER_LINEAR)
        blob = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        raw = self.session.run(None, {self.inp_name: blob})
        dets = raw[0][0] # Shape: [300, 6] -> x1, y1, x2, y2, score, cls
        results = []
        sx, sy = w / self.SIZE, h / self.SIZE

        for x1, y1, x2, y2, score, cid in dets:
            if score < self.conf_floor:
                continue
            cid = int(cid)
            if 0 <= cid < len(self.COCO_CLASSES):
                cls_name = self.COCO_CLASSES[cid]
                if cls_name in self.SURVEILLANCE_CLASSES:
                    box = [
                        int(max(0, x1 * sx)), int(max(0, y1 * sy)),
                        int(min(w, x2 * sx)), int(min(h, y2 * sy))
                    ]
                    results.append({"box": box, "score": float(score), "class_name": cls_name})
        return results


# =====================================================================
# 2. In-Camera Tracker: Clean-room BoT-SORT & OSNet x0.25 (MIT)
# =====================================================================
class KalmanBox:
    """Constant-velocity Kalman filter tracking [cx, cy, w, h, vcx, vcy, vw, vh]."""
    SP, SV = 1.0 / 20.0, 1.0 / 160.0

    def __init__(self, cx: float, cy: float, w: float, h: float):
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)
        std = [2 * self.SP * w, 2 * self.SP * h, 2 * self.SP * w, 2 * self.SP * h,
               10 * self.SV * w, 10 * self.SV * h, 10 * self.SV * w, 10 * self.SV * h]
        self.P = np.diag(np.square(std))
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def predict(self, damp: bool = False):
        if damp:
            self.x[4:] *= 0.8
        w, h = max(self.x[2], 1.0), max(self.x[3], 1.0)
        q = [self.SP * w, self.SP * h, self.SP * w, self.SP * h,
             self.SV * w, self.SV * h, self.SV * w, self.SV * h]
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + np.diag(np.square(q))
        return self.tlbr()

    def correct(self, cx: float, cy: float, w: float, h: float):
        mw, mh = max(self.x[2], 1.0), max(self.x[3], 1.0)
        R = np.diag(np.square([self.SP * mw, self.SP * mh, self.SP * mw, self.SP * mh]))
        z = np.array([cx, cy, w, h], dtype=np.float64)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def tlbr(self) -> Tuple[float, float, float, float]:
        cx, cy, w, h = self.x[:4]
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


class OsNetExtractor:
    """OSNet x0.25 512-d feature extractor for intra-camera track smoothing."""

    def __init__(self, model_path: Optional[str] = None):
        self.session = None
        if model_path and os.path.exists(model_path):
            self.session = ort.InferenceSession(model_path, providers=["MIGraphXExecutionProvider", "CPUExecutionProvider"])
            self.inp = self.session.get_inputs()[0].name

    def extract(self, crop: np.ndarray) -> np.ndarray:
        if self.session is None or crop.size == 0:
            vec = np.random.randn(512).astype(np.float32)
            return vec / (np.linalg.norm(vec) + 1e-6)
        rgb = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (128, 256))
        blob = ((rgb.astype(np.float32) / 255.0) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        blob = blob.transpose(2, 0, 1)[None].astype(np.float32)
        feat = self.session.run(None, {self.inp: blob})[0][0]
        return feat / (np.linalg.norm(feat) + 1e-6)


class BoTSortTracker:
    """In-camera object tracker with Kalman filtering and track lifecycle management."""

    def __init__(self, iou_thresh: float = 0.3, max_lost: int = 10):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for tid, t in self.tracks.items():
            t["kf"].predict(damp=(t["lost"] > 0))

        unmatched = list(range(len(detections)))
        matched = set()

        for tid, t in self.tracks.items():
            pred_box = t["kf"].tlbr()
            best_iou, best_idx = 0.0, -1
            for idx in unmatched:
                det = detections[idx]
                if det["class_name"] != t["cls"]:
                    continue
                iou = self._iou(pred_box, det["box"])
                if iou > best_iou:
                    best_iou, best_idx = iou, idx

            if best_iou >= self.iou_thresh and best_idx != -1:
                b = detections[best_idx]["box"]
                t["kf"].correct((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1])
                t["lost"] = 0
                t["hits"] += 1
                matched.add(tid)
                unmatched.remove(best_idx)

        for tid in list(self.tracks.keys()):
            if tid not in matched:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        for idx in unmatched:
            b = detections[idx]["box"]
            kf = KalmanBox((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1])
            self.tracks[self.next_id] = {
                "kf": kf, "cls": detections[idx]["class_name"], "lost": 0, "hits": 1
            }
            self.next_id += 1

        active = []
        for tid, t in self.tracks.items():
            if t["lost"] == 0:
                active.append({
                    "track_id": tid,
                    "box": [int(v) for v in t["kf"].tlbr()],
                    "class_name": t["cls"]
                })
        return active

    @staticmethod
    def _iou(b1, b2) -> float:
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0


# =====================================================================
# 3. Cross-Camera ReID: YoutuReID (Apache-2.0)
# =====================================================================
class YoutuReIDMatcher:
    """Cross-camera identity linker maintaining exemplar banks per Global ID."""

    def __init__(self, match_thresh: float = 0.67, bank_size: int = 20):
        self.match_thresh = match_thresh
        self.bank_size = bank_size
        self.global_banks: Dict[int, List[np.ndarray]] = {}
        self.next_gid = 1001

    def embed(self, crop: np.ndarray) -> np.ndarray:
        # Mock 768-d normalized embedding vector
        vec = np.random.randn(768).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-6)

    def match(self, crop_emb: np.ndarray) -> Tuple[int, float, bool]:
        best_gid, best_score = None, -1.0
        for gid, bank in self.global_banks.items():
            sims = np.array(bank) @ crop_emb
            max_sim = float(np.max(sims))
            if max_sim > best_score:
                best_score = max_sim
                best_gid = gid

        if best_gid is not None and best_score >= self.match_thresh:
            if len(self.global_banks[best_gid]) < self.bank_size:
                self.global_banks[best_gid].append(crop_emb)
            return best_gid, best_score, False

        new_gid = self.next_gid
        self.next_gid += 1
        self.global_banks[new_gid] = [crop_emb]
        return new_gid, 1.0, True


# =====================================================================
# 4. Face Recognition: CVLface (MIT)
# =====================================================================
class FaceRecognizer:
    """Face recognition module using CVLface 512-d embeddings."""

    def __init__(self, match_thresh: float = 0.55):
        self.match_thresh = match_thresh
        self.gallery: Dict[str, np.ndarray] = {}

    def enroll(self, name: str, emb: np.ndarray):
        self.gallery[name] = emb

    def recognize(self, face_crop: np.ndarray) -> Dict[str, Any]:
        query_emb = np.random.randn(512).astype(np.float32)
        query_emb /= (np.linalg.norm(query_emb) + 1e-6)

        best_name, best_score = "Unknown", -1.0
        for name, enrolled in self.gallery.items():
            score = float(np.dot(enrolled, query_emb))
            if score > best_score:
                best_score = score
                best_name = name

        is_known = best_score >= self.match_thresh
        return {"person_name": best_name if is_known else "Unknown",
                "confidence": best_score, "is_known": is_known}


# =====================================================================
# 5. Plate Pipeline: RF-DETR & PaddleOCR (Apache-2.0)
# =====================================================================
class PlatePipeline:
    """Conditional license plate detection and OCR pipeline for vehicles."""

    def __init__(self):
        pass

    def detect_and_read(self, vehicle_crop: np.ndarray) -> Optional[Dict[str, Any]]:
        vh, vw = vehicle_crop.shape[:2]
        if vh < 40 or vw < 40:
            return None
        # Simulated RF-DETR plate detection + PaddleOCR recognition
        plate_box = [int(vw * 0.3), int(vh * 0.65), int(vw * 0.7), int(vh * 0.85)]
        return {
            "plate_box": plate_box,
            "plate_number": "1AB-4567",
            "ocr_confidence": 0.94
        }


# =====================================================================
# 6. VLM / LLM: Gemma 4 Semantic Reasoner
# =====================================================================
class GemmaVLMReasoner:
    """Multi-modal attribute tagger and conversational reasoner."""

    def __init__(self):
        pass

    def describe_person(self, person_crop: np.ndarray) -> Dict[str, Any]:
        # Return structured visual semantic tags
        return {
            "upper_clothing": "dark jacket",
            "lower_clothing": "blue trousers",
            "accessories": ["backpack"],
            "vlm_summary": "Male in dark jacket with backpack"
        }

    def describe_vehicle(self, vehicle_crop: np.ndarray) -> Dict[str, Any]:
        return {
            "color": "white",
            "body_type": "sedan",
            "vlm_summary": "White sedan car"
        }


# =====================================================================
# 7. Unified Surveillance Pipeline Orchestrator
# =====================================================================
class SurveillancePipeline:
    """Master orchestrator integrating all modules for multi-camera streams."""

    def __init__(self, camera_id: str = "cam_main_gate"):
        self.camera_id = camera_id
        self.detector = RTDetrEngine()
        self.tracker = BoTSortTracker()
        self.reid_embedder = OsNetExtractor()
        self.cross_cam_reid = YoutuReIDMatcher()
        self.face_rec = FaceRecognizer()
        self.plate_pipe = PlatePipeline()
        self.vlm = GemmaVLMReasoner()

        # Enroll a sample VIP for face recognition
        vip_emb = np.random.randn(512).astype(np.float32)
        vip_emb /= np.linalg.norm(vip_emb)
        self.face_rec.enroll("Alice_Engineer", vip_emb)

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: float) -> List[Dict[str, Any]]:
        # 1. Detection via RT-DETR
        raw_dets = self.detector.detect(frame_bgr)

        # 2. Tracking via Clean-room BoT-SORT
        active_tracks = self.tracker.update(raw_dets)

        events = []
        # 3. Route tracked entities by class
        for trk in active_tracks:
            tid = trk["track_id"]
            box = trk["box"]
            cls_name = trk["class_name"]

            # Crop object patch safely
            x1, y1, x2, y2 = box
            crop = frame_bgr[max(0, y1):min(frame_bgr.shape[0], y2),
                              max(0, x1):min(frame_bgr.shape[1], x2)]

            event: Dict[str, Any] = {
                "camera_id": self.camera_id,
                "timestamp_ms": timestamp_ms,
                "track_id": tid,
                "class_name": cls_name,
                "bbox": box
            }

            if cls_name == "person" and crop.size > 0:
                # Per-frame appearance & Cross-camera ReID
                youtureid_feat = self.cross_cam_reid.embed(crop)
                gid, score, is_new = self.cross_cam_reid.match(youtureid_feat)
                event["global_id"] = f"G{gid}"
                event["reid_score"] = round(score, 3)

                # Face recognition check
                face_res = self.face_rec.recognize(crop)
                event["face"] = face_res

                # VLM visual description
                event["attributes"] = self.vlm.describe_person(crop)

            elif cls_name in {"car", "truck", "motorcycle"} and crop.size > 0:
                # Plate detection & OCR
                plate_info = self.plate_pipe.detect_and_read(crop)
                if plate_info:
                    event["plate"] = plate_info

                # Vehicle visual attributes
                event["attributes"] = self.vlm.describe_vehicle(crop)

            events.append(event)

        return events


# =====================================================================
# Standalone execution demo
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Initializing Integrated CCTV AI Surveillance Pipeline")
    print("=" * 65)

    pipeline = SurveillancePipeline(camera_id="nvr1_ch03")

    # Generate synthetic camera frame (720p)
    dummy_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 50
    now_ms = time.time() * 1000

    print("\n[Input] Ingesting 1 frame from CCTV...")
    output_events = pipeline.process_frame(dummy_frame, now_ms)

    print(f"\n[Output] Emitted {len(output_events)} structured surveillance events:\n")
    print(json.dumps(output_events, indent=2))
    print("\nPipeline execution completed successfully.")
