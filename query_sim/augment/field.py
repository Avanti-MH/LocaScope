import cv2
import numpy as np

from functools import lru_cache

#: How many distinct (h, w) the vignette falloff cache holds. Each entry is
#: h*w float64 -- 5.9 MB at the 1440x1024 sensor. A run touches one size per
#: pyramid level and works through levels in order, so 4 is generous.
_CACHE_SIZES = 4


@lru_cache(maxsize=_CACHE_SIZES)
def _vignette_falloff(h: int, w: int) -> np.ndarray:
    """exp(-d^2 / 2*sigma^2) over the frame, float64, read-only.

    This is the expensive half of the vignette: np.hypot over h*w, then a
    square, a divide and an exp over the same. None of it depends on
    `strength`, the only thing that changes between shots, so it is computed
    once per frame size instead of once per exposure.
    """
    cx, cy = w / 2.0, h / 2.0
    Y, X = np.ogrid[:h, :w]
    dist = np.hypot(X - cx, Y - cy)
    sigma = min(cx, cy) * 0.8
    falloff = np.exp(-dist**2 / (2 * sigma**2))
    falloff.flags.writeable = False
    return falloff


def apply_vignette(img, strength=0.4):
    """Gaussian vignette: darkening toward edges.

    The falloff is cached per frame size, and the clip is dropped where it is
    provably a no-op: gain = (1 - s) + s * exp(...), and exp(...) lies in
    (0, 1], so for s in [0, 1] the gain lies in [1 - s, 1] and the product
    with a uint8 image can never leave [0, 255]. Outside that range the clip
    is real, so it stays."""
    if strength == 0.0:
        return img
    h, w = img.shape[:2]
    gain = (1 - strength) + strength * _vignette_falloff(h, w)
    scaled = img * gain[:, :, np.newaxis]
    if 0.0 <= strength <= 1.0:
        return scaled.astype(np.uint8)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def apply_stage_shift(img, dx: int = 0, dy: int = 0):
    """Translate by (dx, dy) whole pixels: stage mechanical jitter.

    THE OFFSETS ARE THE CALLER'S, NOT DRAWN HERE: `_apply_params` hands it
    `p['stage_shift_dx'/'dy']`, the same values `params` reports, so the
    record cannot diverge from the pixels.

    Whole pixels, not sub-pixel: `_sample_params` draws integers, so this is
    a pure re-indexing with no resampling.
    """
    if dx == 0 and dy == 0:
        return img
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
