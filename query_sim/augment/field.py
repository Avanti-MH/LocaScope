import cv2
import numpy as np

from functools import lru_cache

#: Route the public functions through the `_fast` bodies. The `_legacy` bodies
#: stay next to them so test_augment_equivalence.py can measure the two against
#: real Camera output rather than against a claim. Flip to False to fall back.
USE_FAST = True

#: Compute the vignette gain in float32 instead of float64. This is the ONE
#: change in this file that is not exactly equal: `.astype(np.uint8)` truncates
#: rather than rounds, so a product float64 puts at 200.0000001 and float32
#: puts at 199.9999999 becomes 200 against 199. float32's relative error is
#: ~1e-7, which at 255 is ~2.5e-5, so a pixel flips when the exact product
#: lands that close to an integer -- of order 1e-4 of them.
#: Off by default; test_augment_equivalence.py measures the real rate.
VIGNETTE_FLOAT32 = False

#: How many distinct (h, w) the vignette falloff cache holds. Each entry is
#: h*w float64 -- 5.9 MB at the 1440x1024 sensor. A run touches one size per
#: pyramid level and works through levels in order, so 4 is generous.
_CACHE_SIZES = 4


# ══════════════════════════════════════════════════════════════════════════════
#  Geometry that depends only on the frame size
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  Legacy
# ══════════════════════════════════════════════════════════════════════════════

def _apply_vignette_legacy(img, strength=0.4):
    if strength == 0.0:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    Y, X = np.ogrid[:h, :w]
    dist = np.hypot(X - cx, Y - cy)
    sigma = min(cx, cy) * 0.8
    gain = (1 - strength) + strength * np.exp(-dist**2 / (2 * sigma**2))
    return np.clip(img * gain[:, :, np.newaxis], 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  Fast
# ══════════════════════════════════════════════════════════════════════════════

def _apply_vignette_fast(img, strength=0.4):
    """Cached falloff, and the clip dropped where it is provably a no-op.

    gain = (1 - s) + s * exp(...), and exp(...) lies in (0, 1], so for s in
    [0, 1] the gain lies in [1 - s, 1] and the product with a uint8 image can
    never leave [0, 255]. `np.clip` there is a full extra float64 pass over
    h*w*3 -- 74.9 MB on the bounding square -- that cannot change a value.
    Outside that range of s the clip is real, so it stays.

    With VIGNETTE_FLOAT32 the gain is cast to float32 first, which halves the
    two largest allocations and is the only inexact step in this file.
    """
    if strength == 0.0:
        return img
    h, w = img.shape[:2]
    gain = (1 - strength) + strength * _vignette_falloff(h, w)
    if VIGNETTE_FLOAT32:
        gain = gain.astype(np.float32)
    scaled = img * gain[:, :, np.newaxis]
    if 0.0 <= strength <= 1.0:
        return scaled.astype(np.uint8)
    return np.clip(scaled, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  Public
# ══════════════════════════════════════════════════════════════════════════════

def apply_vignette(img, strength=0.4):
    """Gaussian vignette: darkening toward edges."""
    if USE_FAST:
        return _apply_vignette_fast(img, strength)
    return _apply_vignette_legacy(img, strength)


def apply_stage_shift(img, max_shift=3):
    """Random sub-pixel stage mechanical jitter."""
    if max_shift <= 0:
        return img
    dx = np.random.randint(-max_shift, max_shift + 1)
    dy = np.random.randint(-max_shift, max_shift + 1)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
