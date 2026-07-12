"""facerec.py — self-contained face recognition module.

Pipeline: align (MediaPipe face mesh) → embed (ONNX) → match (cosine gallery)
with multi-shot consensus voting. No external proprietary dependencies.

Requires: numpy, opencv-python, onnxruntime, mediapipe
Models:   models/face_landmarker.task + models/recognition.onnx
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


# ========================== Types ==========================

@dataclass
class AlignedFace:
    crop: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0
    sharpness: float = 0.0
    bbox: tuple[int, int, int, int] | None = None


@dataclass
class FaceMatch:
    person_id: str | None
    score: float                          # raw best cosine
    confidence: float = 0.0              # blended: ratio*w + score*w
    is_known: bool = False
    votes: tuple[int, int] | None = None  # (winner_votes, total_embedded)
    runner_up: tuple[str, float] | None = None
    bbox: tuple[int, int, int, int] | None = None
    embedding: np.ndarray | None = field(default=None, repr=False)


# ========================== Embedder ==========================

class OnnxEmbedder:
    """Face embedder: BGR crop → L2-normalized 512-d vector via ONNX.

    Runs on MIGraphX (AMD GPU) when available, else CPU. MIGraphX compiles per
    concrete shape, so GPU inference uses a FIXED batch (pad + slice) — the same
    approach the airocm osnet embedder uses. A varying batch would recompile
    (~47s) every time the crop count changed, so we pin it to FIXED_BATCH."""

    FIXED_BATCH = 8        # matches FACE_MAX_CROPS; pad/slice to this on GPU

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        import os
        so = ort.SessionOptions()
        so.log_severity_level = 3
        want_gpu = os.environ.get("FACE_DEVICE", "gpu").lower() != "cpu"
        providers = (["MIGraphXExecutionProvider", "CPUExecutionProvider"]
                     if want_gpu else ["CPUExecutionProvider"])
        self._session = ort.InferenceSession(onnx_path, so, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._on_gpu = self._session.get_providers()[0] == "MIGraphXExecutionProvider"
        if self._on_gpu:
            # Compile the fixed shape now (one-time; cached via
            # ORT_MIGRAPHX_MODEL_CACHE_PATH so restarts are instant).
            self._session.run(
                [self._output_name],
                {self._input_name: np.zeros((self.FIXED_BATCH, 3, 112, 112), np.float32)})
        print(f"[facerec] embedder on {self._session.get_providers()[0]}"
              f"{' (fixed batch %d)' % self.FIXED_BATCH if self._on_gpu else ''}",
              flush=True)

    def embed_many(self, crops: list[np.ndarray]) -> np.ndarray:
        """crops: list of HxWx3 BGR uint8 → (N, 512) float32, L2-normalized."""
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        blob = self._preprocess(crops)
        if self._on_gpu:
            out = self._run_fixed(blob)      # pad/slice to FIXED_BATCH chunks
        else:
            out = self._session.run(
                [self._output_name], {self._input_name: blob})[0]
        out = out.astype(np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return out / norms

    def _run_fixed(self, blob: np.ndarray) -> np.ndarray:
        """MIGraphX path: run in FIXED_BATCH-sized chunks, padding the last."""
        n = len(blob)
        outs = []
        for i in range(0, n, self.FIXED_BATCH):
            chunk = blob[i:i + self.FIXED_BATCH]
            m = len(chunk)
            if m < self.FIXED_BATCH:
                pad = np.zeros((self.FIXED_BATCH - m, 3, 112, 112), np.float32)
                chunk = np.concatenate([chunk, pad])
            y = self._session.run([self._output_name], {self._input_name: chunk})[0]
            outs.append(y[:m])
        return np.concatenate(outs)

    @staticmethod
    def _preprocess(crops: list[np.ndarray]) -> np.ndarray:
        batch = np.empty((len(crops), 3, 112, 112), dtype=np.float32)
        for i, crop in enumerate(crops):
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (112, 112):
                rgb = cv2.resize(rgb, (112, 112))
            x = (rgb.astype(np.float32) - 127.5) / 127.5
            batch[i] = x.transpose(2, 0, 1)
        return batch


# ========================== Aligner ==========================

# MediaPipe face mesh landmark indices for pose estimation
_NOSE_TIP = 1
_LEFT_CHEEK = 234
_RIGHT_CHEEK = 454
_FOREHEAD = 10
_CHIN = 152
_PAD_FRAC = 0.05
_MIN_CROP_PX = 30


class Aligner:
    """Face alignment via MediaPipe FaceLandmarker."""

    def __init__(self, landmarker_path: str, mesh_confidence: float = 0.3):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker, FaceLandmarkerOptions, RunningMode,
        )
        self._landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=landmarker_path),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=mesh_confidence,
                min_face_presence_confidence=mesh_confidence,
            )
        )

    def align(self, image: np.ndarray) -> AlignedFace | None:
        """BGR image → AlignedFace or None if no face found."""
        import mediapipe as mp

        if image is None or image.size == 0:
            return None
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_image)
        if not results.face_landmarks:
            return None

        landmarks = results.face_landmarks[0]
        h, w = image.shape[:2]
        pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)

        # Pose
        nose = pts[_NOSE_TIP]
        lc, rc = pts[_LEFT_CHEEK], pts[_RIGHT_CHEEK]
        face_w = float(np.linalg.norm(rc - lc))
        yaw = float((nose[0] - (lc[0] + rc[0]) / 2) / max(face_w, 1.0))

        fh, ch = pts[_FOREHEAD], pts[_CHIN]
        face_h = float(np.linalg.norm(ch - fh))
        pitch = float((nose[1] - (fh[1] + ch[1]) / 2) / max(face_h, 1.0))

        # Crop
        x_min, y_min = pts.min(axis=0).astype(int)
        x_max, y_max = pts.max(axis=0).astype(int)
        pad_x = int((x_max - x_min) * _PAD_FRAC)
        pad_y = int((y_max - y_min) * _PAD_FRAC)
        x1 = max(0, int(x_min) - pad_x)
        y1 = max(0, int(y_min) - pad_y)
        x2 = min(w, int(x_max) + pad_x)
        y2 = min(h, int(y_max) + pad_y)
        if (x2 - x1) < _MIN_CROP_PX or (y2 - y1) < _MIN_CROP_PX:
            return None

        crop = image[y1:y2, x1:x2]
        sharpness = float(cv2.Laplacian(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_32F
        ).var())

        return AlignedFace(crop=crop, yaw=yaw, pitch=pitch,
                           sharpness=sharpness, bbox=(x1, y1, x2, y2))


# ========================== Gallery ==========================

class Gallery:
    """Cosine-similarity gallery over L2-normalized 512-d vectors."""

    def __init__(self):
        self._ids: list[str] = []
        self._vecs: np.ndarray | None = None  # (N, 512)

    def __len__(self):
        return len(self._ids)

    def ids(self) -> list[str]:
        return list(self._ids)

    def add(self, person_id: str, vec: np.ndarray):
        v = vec.astype(np.float32).ravel()
        n = float(np.linalg.norm(v))
        if n > 1e-12:
            v /= n
        if person_id in self._ids:
            self._vecs[self._ids.index(person_id)] = v
            return
        self._ids.append(person_id)
        self._vecs = v[None, :] if self._vecs is None else np.vstack([self._vecs, v])

    def remove(self, person_id: str):
        if person_id not in self._ids:
            return
        i = self._ids.index(person_id)
        del self._ids[i]
        self._vecs = np.delete(self._vecs, i, axis=0) if self._vecs is not None else None
        if self._vecs is not None and self._vecs.shape[0] == 0:
            self._vecs = None

    def search(self, vec: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        if self._vecs is None:
            return []
        q = vec.astype(np.float32).ravel()
        n = float(np.linalg.norm(q))
        if n > 1e-12:
            q /= n
        scores = self._vecs @ q
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self._ids[i], float(scores[i])) for i in idx]

    def search_many(self, vecs: np.ndarray) -> list[tuple[str, float]]:
        """Best match per query in one batched matmul."""
        if self._vecs is None:
            return [(None, 0.0)] * len(vecs)
        q = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        q = q / norms
        scores = q @ self._vecs.T
        best_idx = np.argmax(scores, axis=1)
        return [(self._ids[i], float(scores[row, i]))
                for row, i in enumerate(best_idx)]

    def save(self, path: str):
        vecs = self._vecs if self._vecs is not None else np.zeros((0, 512), np.float32)
        np.savez(path, ids=np.array(self._ids, dtype=object), vecs=vecs)

    def load(self, path: str):
        with np.load(path, allow_pickle=True) as data:
            self._ids = [str(x) for x in data["ids"].tolist()]
            vecs = data["vecs"].astype(np.float32)
        self._vecs = vecs if vecs.shape[0] > 0 else None


# ========================== Gate Logic ==========================

def passes_gates(face: AlignedFace, *, max_yaw: float, max_pitch: float,
                 min_sharpness: float, min_face_size: int) -> bool:
    if abs(face.yaw) > max_yaw:
        return False
    if abs(face.pitch) > max_pitch:
        return False
    if face.sharpness < min_sharpness:
        return False
    if face.bbox is not None:
        x1, y1, x2, y2 = face.bbox
        if (x2 - x1) < min_face_size or (y2 - y1) < min_face_size:
            return False
    return True


# ========================== FaceRecognizer ==========================

class FaceRecognizer:
    def __init__(
        self,
        model_dir: str = "models",
        *,
        device: str = "cpu",
        embedder=None,
        aligner=None,
        max_yaw: float = 0.60,
        max_pitch: float = 0.60,
        min_sharpness: float = 10.0,
        min_face_size: int = 20,
        mesh_confidence: float = 0.15,
        match_threshold: float = 0.40,
        ratio_weight: float = 0.40,
        score_weight: float = 0.60,
        min_votes: int = 3,
        max_face_crops: int = 10,
        max_workers: int = 4,
        warmup: bool = False,
    ):
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.min_sharpness = min_sharpness
        self.min_face_size = min_face_size
        self.match_threshold = match_threshold
        self.ratio_weight = ratio_weight
        self.score_weight = score_weight
        self.min_votes = min_votes
        self.max_face_crops = max_face_crops
        self.max_workers = max_workers

        model_path = Path(model_dir)
        self._aligner = aligner or Aligner(
            str(model_path / "face_landmarker.task"), mesh_confidence
        )
        self._embedder = embedder or OnnxEmbedder(
            str(model_path / "recognition.onnx")
        )
        self._gallery = Gallery()

    # --- Public API ---

    def recognize(self, image: np.ndarray) -> FaceMatch | None:
        aligned = self.align(image)
        if aligned is None or not self._passes_gates(aligned):
            return None
        vec = self.embed(aligned)
        result = self.match(vec)
        result.bbox = aligned.bbox
        result.embedding = vec
        return result

    def recognize_many(self, images: list[np.ndarray]) -> FaceMatch | None:
        if not images:
            return None
        # Parallel align
        if len(images) == 1:
            aligned_raw = [self.align(images[0])]
        else:
            workers = min(self.max_workers, len(images))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                aligned_raw = list(pool.map(self.align, images))

        aligned_list = [a for a in aligned_raw
                        if a is not None and self._passes_gates(a)]
        if not aligned_list:
            return None

        aligned_list.sort(key=lambda a: a.sharpness, reverse=True)
        best = aligned_list[:self.max_face_crops]

        crops = [a.crop for a in best]
        vecs = self._embedder.embed_many(crops)

        return self._vote(vecs)

    def recognize_many_raw(self, crops: list[np.ndarray]) -> FaceMatch | None:
        """Like recognize_many but skips alignment — embeds raw crops directly.
        For use when MediaPipe can't detect (overhead cameras, etc.)."""
        if not crops:
            return None
        vecs = self._embedder.embed_many(crops)
        return self._vote(vecs)

    def align(self, image: np.ndarray) -> AlignedFace | None:
        return self._aligner.align(image)

    def embed(self, aligned: AlignedFace | np.ndarray) -> np.ndarray:
        crop = aligned.crop if isinstance(aligned, AlignedFace) else aligned
        return self._embedder.embed_many([crop])[0]

    def match(self, vec: np.ndarray, top_k: int = 3) -> FaceMatch:
        hits = self._gallery.search(vec, top_k=top_k)
        if not hits:
            return FaceMatch(person_id=None, score=0.0, confidence=0.0,
                             is_known=False, votes=(1, 1))
        best_id, best_score = hits[0]
        runner = (hits[1][0], hits[1][1]) if len(hits) > 1 else None
        conf = self._blend(best_score, 1, 1)
        known = conf >= self.match_threshold
        return FaceMatch(
            person_id=best_id if known else None,
            score=best_score, confidence=conf, is_known=known,
            votes=(1, 1), runner_up=runner,
        )

    def enroll(self, person_id: str, images) -> bool:
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
        vecs = []
        for img in images:
            aligned = self.align(img)
            if aligned is None or not self._passes_gates(aligned):
                continue
            vecs.append(self.embed(aligned))
        if not vecs:
            return False
        mean_vec = np.mean(vecs, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 1e-12:
            mean_vec /= norm
        self._gallery.add(person_id, mean_vec)
        return True

    def remove(self, person_id: str):
        self._gallery.remove(person_id)

    def gallery_ids(self) -> list[str]:
        return self._gallery.ids()

    def save_gallery(self, path: str):
        self._gallery.save(path)

    def load_gallery(self, path: str):
        self._gallery.load(path)

    # --- Internal ---

    def _get_embedder(self):
        return self._embedder

    def _passes_gates(self, face: AlignedFace) -> bool:
        return passes_gates(face, max_yaw=self.max_yaw, max_pitch=self.max_pitch,
                            min_sharpness=self.min_sharpness,
                            min_face_size=self.min_face_size)

    def _blend(self, best_score: float, count: int, total: int) -> float:
        ratio = (count / total) if total else 0.0
        return ratio * self.ratio_weight + best_score * self.score_weight

    def _vote(self, vecs: np.ndarray) -> FaceMatch | None:
        """Vote across embeddings vs gallery, return consensus FaceMatch."""
        hits = self._gallery.search_many(vecs)
        votes: dict[str, dict] = {}
        for pid, score in hits:
            if pid is None:
                continue
            if pid not in votes:
                votes[pid] = {"count": 0, "best_score": -1.0, "person_id": pid}
            votes[pid]["count"] += 1
            if score > votes[pid]["best_score"]:
                votes[pid]["best_score"] = score

        n_embedded = len(vecs)
        if not votes:
            return FaceMatch(person_id=None, score=0.0, confidence=0.0,
                             is_known=False, votes=(0, n_embedded))

        sorted_votes = sorted(votes.values(),
                              key=lambda v: (v["count"], v["best_score"]),
                              reverse=True)
        winner = sorted_votes[0]
        runner_up = (sorted_votes[1]["person_id"], sorted_votes[1]["best_score"]) \
            if len(sorted_votes) > 1 else None

        k = winner["count"]
        score = winner["best_score"]
        confidence = self._blend(score, k, n_embedded)
        known = k >= self.min_votes and confidence >= self.match_threshold

        return FaceMatch(
            person_id=winner["person_id"] if known else None,
            score=score, confidence=confidence, is_known=known,
            votes=(k, n_embedded), runner_up=runner_up,
        )
