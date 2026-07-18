import cv2
import numpy as np


def apply_field_mask(img, fill=255):
    """Circular field-of-view mask; corners filled with `fill`."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, 255, -1)
    result = np.full_like(img, fill)
    result[mask == 255] = img[mask == 255]
    return result


def apply_vignette(img, strength=0.4):
    """Gaussian vignette: darkening toward edges."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    Y, X = np.ogrid[:h, :w]
    dist = np.hypot(X - cx, Y - cy)
    sigma = min(cx, cy) * 0.8
    gain = (1 - strength) + strength * np.exp(-dist**2 / (2 * sigma**2))
    return np.clip(img * gain[:, :, np.newaxis], 0, 255).astype(np.uint8)


def apply_stage_shift(img, max_shift=3):
    """Random sub-pixel stage mechanical jitter."""
    dx = np.random.randint(-max_shift, max_shift + 1)
    dy = np.random.randint(-max_shift, max_shift + 1)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
