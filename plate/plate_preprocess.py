"""
plate_preprocess.py — Crop + clean a detected plate region before OCR.

Two helpers mirror the production pipeline:
  - crop_plate:  pad around the plate box and up-scale tiny crops so the
                 recognizer sees enough pixels.
  - deskew_plate: rotate the crop so text is horizontal (OCR is sensitive
                 to tilt from camera angle).
"""
import cv2
import numpy as np


def crop_plate(frame, x1, y1, x2, y2, pad=0.15, min_h=48):
    """Crop the plate box with padding; up-scale if shorter than min_h px."""
    fh, fw = frame.shape[:2]
    pw, ph = x2 - x1, y2 - y1
    pad_x, pad_y = int(pw * pad), int(ph * pad)
    crop = frame[max(0, y1 - pad_y):min(fh, y2 + pad_y),
                 max(0, x1 - pad_x):min(fw, x2 + pad_x)]
    if crop.size == 0:
        return None
    if crop.shape[0] < min_h:
        scale = min_h / crop.shape[0]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def deskew_plate(crop, min_angle=2.0):
    """Estimate text tilt via Otsu + minAreaRect and rotate to level it."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(bw)
    if coords is None:
        return crop
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < min_angle:  # skip trivial rotations
        return crop
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def preprocess_crop(frame, x1, y1, x2, y2, do_deskew=True):
    """Crop the plate region, optionally deskew it. Returns None if empty."""
    crop = crop_plate(frame, x1, y1, x2, y2)
    if crop is None:
        return None
    return deskew_plate(crop) if do_deskew else crop
