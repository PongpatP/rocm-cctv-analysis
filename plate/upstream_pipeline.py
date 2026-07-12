"""
Dragon-OCR Inference Pipeline
==============================
Thai License Plate Recognition: YOLO vehicle → YOLO plate → PaddleOCR

Requirements:
    pip install paddlepaddle>=3.0 ultralytics opencv-python numpy

Models needed (place in models/):
    - models/license-plate-finetune-v1s.pt   (YOLO plate detector from HuggingFace)
    - models/ocr/inference.json              (PaddleOCR recognition model)
    - models/ocr/inference.pdiparams
    - models/ocr/thai_plate_dict.txt

Usage:
    python inference/run_pipeline.py --video path/to/video.mp4
    python inference/run_pipeline.py --video path/to/video.mp4 --output results/
    python inference/run_pipeline.py --image path/to/frame.jpg
"""

import argparse
import csv
import os
import sys
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher

import cv2
import numpy as np


# ─── OCR Engine ──────────────────────────────────────────────────────────────

class PlateOCR:
    """PaddleOCR-based Thai plate text recognizer."""

    def __init__(self, model_dir, dict_path):
        from paddle import inference as paddle_infer

        config = paddle_infer.Config(
            os.path.join(model_dir, "inference.json"),
            os.path.join(model_dir, "inference.pdiparams"),
        )
        config.disable_gpu()  # CPU inference (fast enough for plates)
        config.set_cpu_math_library_num_threads(4)
        config.disable_glog_info()
        self.predictor = paddle_infer.create_predictor(config)

        # Load character dictionary
        with open(dict_path, "r", encoding="utf-8") as f:
            self.chars = ["blank"] + [line.strip() for line in f if line.strip()]

    def predict(self, img):
        """Run OCR on a plate crop image. Returns (text, confidence)."""
        # Preprocess: resize to 48x320 maintaining aspect ratio
        h, w = img.shape[:2]
        ratio = 48 / h
        new_w = min(int(w * ratio), 320)
        resized = cv2.resize(img, (new_w, 48))
        if new_w < 320:
            pad = np.zeros((48, 320 - new_w, 3), dtype=np.uint8)
            resized = np.concatenate([resized, pad], axis=1)

        # Normalize to 0-1
        blob = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, :]

        # Run inference
        input_handle = self.predictor.get_input_handle("x")
        input_handle.reshape(list(blob.shape))
        input_handle.copy_from_cpu(blob)
        self.predictor.run()
        output = self.predictor.get_output_handle(
            self.predictor.get_output_names()[0]
        ).copy_to_cpu()

        # CTC greedy decode
        logits = output[0]
        # Softmax for confidence
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        indices = probs.argmax(axis=1)
        max_probs = probs.max(axis=1)

        text = []
        conf_sum = 0.0
        prev = -1
        for idx, p in zip(indices, max_probs):
            if idx != 0 and idx != prev and idx < len(self.chars):
                text.append(self.chars[idx])
                conf_sum += p
            prev = idx

        plate_text = "".join(text)
        avg_conf = conf_sum / len(text) if text else 0.0
        return plate_text, avg_conf


# ─── Main Pipeline ───────────────────────────────────────────────────────────

class ALPRPipeline:
    """Automatic License Plate Recognition pipeline."""

    def __init__(self, plate_model_path, ocr_model_dir, ocr_dict_path,
                 vehicle_model_path=None, conf_threshold=0.3):
        from ultralytics import YOLO

        self.plate_model = YOLO(plate_model_path)
        self.vehicle_model = YOLO(vehicle_model_path) if vehicle_model_path else None
        self.ocr = PlateOCR(ocr_model_dir, ocr_dict_path)
        self.conf_threshold = conf_threshold

        # COCO vehicle classes
        self.VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

    def detect_plates_in_frame(self, frame):
        """Detect plates using 2-stage (vehicle→plate) or 1-stage (plate only)."""
        plates = []

        if self.vehicle_model:
            # 2-stage: vehicle first, then plate within vehicle
            vehicle_dets = self.vehicle_model(frame, verbose=False, conf=0.3)[0]
            for vbox, vcls in zip(
                vehicle_dets.boxes.xyxy.cpu().numpy().astype(int),
                vehicle_dets.boxes.cls.cpu().numpy().astype(int),
            ):
                if int(vcls) not in self.VEHICLE_CLASSES:
                    continue
                vx1, vy1, vx2, vy2 = vbox
                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                if vehicle_crop.size == 0:
                    continue

                plate_dets = self.plate_model(vehicle_crop, verbose=False,
                                              conf=self.conf_threshold, imgsz=640)[0]
                if plate_dets.boxes is None:
                    continue
                for pbox, pconf in zip(
                    plate_dets.boxes.xyxy.cpu().numpy().astype(int),
                    plate_dets.boxes.conf.cpu().numpy(),
                ):
                    px1, py1, px2, py2 = pbox
                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                    if plate_crop.size > 0:
                        plates.append((plate_crop, float(pconf)))
        else:
            # 1-stage: detect plates directly
            plate_dets = self.plate_model(frame, verbose=False,
                                          conf=self.conf_threshold, imgsz=1280)[0]
            if plate_dets.boxes is not None:
                for pbox, pconf in zip(
                    plate_dets.boxes.xyxy.cpu().numpy().astype(int),
                    plate_dets.boxes.conf.cpu().numpy(),
                ):
                    x1, y1, x2, y2 = pbox
                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size > 0:
                        plates.append((plate_crop, float(pconf)))

        return plates

    def read_plate(self, plate_crop):
        """OCR a single plate crop. Returns (text, confidence) or None."""
        text, conf = self.ocr.predict(plate_crop)
        if text and len(text) >= 3 and conf > 0.3:
            return text, conf
        return None

    def process_image(self, image_path):
        """Process a single image. Returns list of (text, confidence, crop)."""
        frame = cv2.imread(image_path)
        if frame is None:
            return []
        results = []
        plates = self.detect_plates_in_frame(frame)
        for crop, det_conf in plates:
            reading = self.read_plate(crop)
            if reading:
                text, ocr_conf = reading
                results.append({"text": text, "ocr_conf": ocr_conf,
                                "det_conf": det_conf, "crop": crop})
        return results

    def process_video(self, video_path, output_dir, frame_skip=3):
        """Process a video with multi-frame voting."""
        os.makedirs(output_dir, exist_ok=True)
        crops_dir = os.path.join(output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Video: {total_frames} frames, {fps:.0f} FPS")
        print(f"Processing every {frame_skip} frames...")
        print("=" * 60)

        all_readings = []  # (timestamp, text, conf, crop)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            plates = self.detect_plates_in_frame(frame)
            for crop, det_conf in plates:
                reading = self.read_plate(crop)
                if reading:
                    text, ocr_conf = reading
                    timestamp = frame_idx / fps
                    all_readings.append({
                        "frame": frame_idx,
                        "timestamp": timestamp,
                        "text": text,
                        "ocr_conf": ocr_conf,
                        "det_conf": det_conf,
                    })
                    # Save crop
                    crop_path = os.path.join(crops_dir,
                                            f"frame{frame_idx:04d}_{text}.jpg")
                    cv2.imwrite(crop_path, crop)
                    print(f"  [{timestamp:6.1f}s] {text:12s} conf={ocr_conf:.2f}")

            frame_idx += 1

        cap.release()

        # ─── Voting: group similar readings and pick consensus ────────────
        print("\n" + "=" * 60)
        print("Voting results:")
        print("-" * 60)

        # Simple grouping by text similarity
        voted = self._vote(all_readings)

        # Write CSV
        csv_path = os.path.join(output_dir, "results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["plate", "count", "avg_conf", "first_seen", "last_seen"])
            writer.writeheader()
            for v in voted:
                writer.writerow(v)
                print(f"  {v['plate']:12s} | seen {v['count']}x | conf={v['avg_conf']:.2f} "
                      f"| {v['first_seen']:.1f}s - {v['last_seen']:.1f}s")

        print(f"\nResults saved to: {csv_path}")
        print(f"Crops saved to: {crops_dir}/")
        return voted

    def _vote(self, readings):
        """Group readings by similarity and vote for consensus text."""
        if not readings:
            return []

        # Group readings with >=70% similarity
        groups = []
        used = set()

        for i, r in enumerate(readings):
            if i in used:
                continue
            group = [r]
            used.add(i)
            for j, other in enumerate(readings):
                if j in used:
                    continue
                sim = SequenceMatcher(None, r["text"], other["text"]).ratio()
                if sim >= 0.7:
                    group.append(other)
                    used.add(j)
            groups.append(group)

        # For each group, vote for most common text
        results = []
        for group in groups:
            texts = [r["text"] for r in group]
            counter = Counter(texts)
            best_text = counter.most_common(1)[0][0]
            confs = [r["ocr_conf"] for r in group if r["text"] == best_text]
            results.append({
                "plate": best_text,
                "count": len(group),
                "avg_conf": sum(confs) / len(confs),
                "first_seen": min(r["timestamp"] for r in group),
                "last_seen": max(r["timestamp"] for r in group),
            })

        # Sort by count (most seen first)
        results.sort(key=lambda x: -x["count"])
        return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dragon-OCR: Thai License Plate Recognition")
    parser.add_argument("--video", help="Path to input video")
    parser.add_argument("--image", help="Path to single image")
    parser.add_argument("--output", default="output/results", help="Output directory")
    parser.add_argument("--plate-model", default="models/license-plate-finetune-v1s.pt")
    parser.add_argument("--vehicle-model", default="yolo11n.pt",
                        help="Vehicle detection model (set 'none' to disable)")
    parser.add_argument("--ocr-model", default="models/ocr", help="OCR model directory")
    parser.add_argument("--ocr-dict", default="models/ocr/thai_plate_dict.txt")
    parser.add_argument("--frame-skip", type=int, default=3, help="Process every Nth frame")
    parser.add_argument("--conf", type=float, default=0.3, help="Plate detection confidence")
    args = parser.parse_args()

    vehicle_model = args.vehicle_model if args.vehicle_model != "none" else None

    pipeline = ALPRPipeline(
        plate_model_path=args.plate_model,
        ocr_model_dir=args.ocr_model,
        ocr_dict_path=args.ocr_dict,
        vehicle_model_path=vehicle_model,
        conf_threshold=args.conf,
    )

    if args.video:
        pipeline.process_video(args.video, args.output, args.frame_skip)
    elif args.image:
        results = pipeline.process_image(args.image)
        for r in results:
            print(f"  {r['text']:12s} | conf={r['ocr_conf']:.2f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
