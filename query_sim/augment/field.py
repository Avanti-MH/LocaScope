import cv2
import numpy as np

from functools import lru_cache

#: Route the public functions through the `_fast` bodies. The `_legacy` bodies
#: stay next to them so test_camera.py (augment section) can measure the two against
#: real camera (Render) output rather than against a claim. Flip to False to fall back.
USE_FAST = True

#: Compute the vignette gain in float32 instead of float64. This is the ONE
#: change in this file that is not exactly equal: `.astype(np.uint8)` truncates
#: rather than rounds, so a product float64 puts at 200.0000001 and float32
#: puts at 199.9999999 becomes 200 against 199. float32's relative error is
#: ~1e-7, which at 255 is ~2.5e-5, so a pixel flips when the exact product
#: lands that close to an integer -- of order 1e-4 of them.
#: Off by default; test_camera.py (augment section) measures the real rate.
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


def apply_stage_shift(img, dx: int = 0, dy: int = 0):
    """Translate by (dx, dy) whole pixels: stage mechanical jitter.

    THE OFFSETS ARE THE CALLER'S, NOT DRAWN HERE. Until 2026-09-16 this
    function drew its own pair from the global `np.random` while
    `pipeline._sample_params` drew ANOTHER pair from the caller's rng and
    recorded it in `params` -- so every shot's recorded `stage_shift_dx/dy`
    named a displacement the image had never been given, and the real one
    obeyed no `seed` any caller could pass. Both halves are fixed by this
    function no longer having randomness of its own: `_apply_params` hands
    it `p['stage_shift_dx'/'dy']`, the same values `params` reports, so the
    record cannot diverge from the pixels because there is only one pair.

    Whole pixels, not sub-pixel: `_sample_params` draws integers, so this is
    a pure re-indexing with no resampling. The old docstring said "sub-pixel"
    and the old code drew `np.random.randint`, which is integer -- the name
    was wrong about its own implementation, not just about this one.
    """
    if dx == 0 and dy == 0:
        return img
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
