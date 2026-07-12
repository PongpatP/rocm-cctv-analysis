"""
plate_tracker.py — Per-vehicle plate OCR aggregation.

Self-contained distillation of the traffic-pipeline OCR stage.
No dependency on the company pipeline or the fine-tuned recognition model.

The core idea: a single OCR read of a moving vehicle's plate is noisy
(motion blur, angle, partial occlusion). Instead of trusting one frame,
we collect many crops of the SAME vehicle across frames, OCR them, and
take a CONFIDENCE-WEIGHTED MAJORITY VOTE over the normalized strings.

You plug in any `recognizer_fn(list[crop]) -> list[(text, confidence)]`.
See ocr_pipeline.py for a vanilla-PaddleOCR implementation.
"""
import re
from collections import Counter


def normalize_plate(text: str) -> str:
    """
    Canonicalize a raw OCR string so votes for the same plate collapse together.
    Strips spaces, dashes and punctuation; keeps digits, latin letters and Thai
    consonants (ก-ฮ). Adjust the character class for other locales.
    """
    return re.sub(r"[^0-9a-zA-Zก-ฮ]", "", text)


class PlateTracker:
    """
    Buffers plate crops per vehicle (track_id) and produces a majority-voted
    plate string once enough evidence is gathered.

    Lifecycle per vehicle:
        add_crop(id, crop, conf)  # called every frame a plate is detected
        ...                       # internally flushes when buffer is full
        get_normalized(id)        # final voted plate (call when vehicle leaves)
        cleanup(id)               # free memory for that vehicle

    Voting:
        Each flush OCRs the top-K highest-detection-confidence crops.
        Every accepted read contributes `round(ocr_conf * 100)` votes for its
        normalized string. High-confidence reads therefore outweigh low ones.
        Final answer = most-voted string (ties broken by most recent).
    """

    def __init__(
        self,
        recognizer_fn,
        buffer_size: int = 8,
        top_k: int = 3,
        ocr_confidence: float = 0.5,
        min_plate_length: int = 4,
    ):
        """
        Args:
            recognizer_fn: fn(list[crop]) -> list[(text, confidence)].
            buffer_size:   flush OCR once this many crops are buffered for a vehicle.
            top_k:         per flush, only OCR the K crops with highest DETECTION conf.
            ocr_confidence: reject OCR reads below this recognition confidence.
            min_plate_length: reject normalized strings shorter than this.
        """
        self.recognizer_fn = recognizer_fn
        self.buffer_size = buffer_size
        self.top_k = top_k
        self.ocr_confidence = ocr_confidence
        self.min_plate_length = min_plate_length

        self.buffer = {}       # track_id -> [(crop, detection_conf), ...]
        self.all_texts = {}    # track_id -> [normalized_text, ...]  (vote pool)
        self.raw_results = {}  # track_id -> (latest_raw_text, ocr_conf)

    def add_crop(self, track_id, crop, detection_conf):
        """Buffer one plate crop for a vehicle; auto-flush when buffer is full."""
        self.buffer.setdefault(track_id, []).append((crop, detection_conf))
        if len(self.buffer[track_id]) >= self.buffer_size:
            self._flush(track_id)

    def _flush(self, track_id):
        """OCR the best crops in the buffer and fold results into the vote pool."""
        # Prefer crops the plate DETECTOR was most confident about — they tend
        # to be the clearest, most front-facing views.
        entries = sorted(self.buffer[track_id], key=lambda e: e[1], reverse=True)
        crops = [c for c, _ in entries[: self.top_k]]

        rec_res = self.recognizer_fn(crops)  # [(text, conf), ...]
        accepted = [(t, s) for t, s in rec_res if len(t) >= 2 and s >= self.ocr_confidence]

        if accepted:
            # Keep the single best raw read (useful for debugging / display).
            best = max(accepted, key=lambda e: e[1])
            self.raw_results[track_id] = (best[0], round(best[1], 3))

        for text, score in accepted:
            normalized = normalize_plate(text)
            if len(normalized) >= self.min_plate_length:
                votes = max(1, round(score * 100))  # confidence-weighted
                self.all_texts.setdefault(track_id, []).extend([normalized] * votes)

        self.buffer[track_id] = []

    def get_normalized(self, track_id):
        """
        Final majority-voted plate for a vehicle. Call when the vehicle exits.
        Flushes any residual buffered crops first. Ties -> most recent string.
        """
        if self.buffer.get(track_id):
            self._flush(track_id)

        texts = self.all_texts.get(track_id, [])
        if not texts:
            return None

        counts = Counter(texts)
        max_count = counts.most_common(1)[0][1]
        for text in reversed(texts):  # most-recent-wins tie-break
            if counts[text] == max_count:
                return text

    def get_raw(self, track_id):
        """Latest single best raw OCR read (text, conf) — for live display/debug."""
        return self.raw_results.get(track_id, (None, None))

    def cleanup(self, track_id):
        """Release all buffered state for a vehicle."""
        self.buffer.pop(track_id, None)
        self.all_texts.pop(track_id, None)
        self.raw_results.pop(track_id, None)
