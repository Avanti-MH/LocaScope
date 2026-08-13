import cv2
import numpy as np

from functools import lru_cache

#: Route the public functions through the `_fast` bodies. The `_legacy` bodies
#: stay next to them so test_augment_equivalence.py can measure the two against
#: real Camera output rather than against a claim. Flip to False to fall back.
USE_FAST = True

#: Distinct (h, w) held by the distortion grid cache. Each entry is three
#: float32 [h, w] arrays -- 37 MB on the 1767x1767 bounding square. Matches
#: field._CACHE_SIZES; a run touches one frame size per pyramid level.
_CACHE_SIZES = 4


@lru_cache(maxsize=_CACHE_SIZES)
def _distortion_grid(h: int, w: int):
    """(xn, yn, r2) normalised coordinates, float32, read-only.

    `np.mgrid` over h*w plus the two normalisations and the radius square are
    the bulk of the legacy body's allocation -- about 100 MB of float32
    temporaries on the bounding square -- and every one of them depends only on
    the frame size. k1 and k2, the parts that change per shot, enter afterwards.

    The arithmetic is written exactly as the legacy body wrote it, in the same
    order, so the cached values are the same bits it would have computed.
    """
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (X - cx) / (cx + 1e-6)
    yn = (Y - cy) / (cy + 1e-6)
    r2 = xn**2 + yn**2
    for array in (xn, yn, r2):
        array.flags.writeable = False
    return xn, yn, r2


# ══════════════════════════════════════════════════════════════════════════════
#  Legacy
# ══════════════════════════════════════════════════════════════════════════════

def _apply_distortion_legacy(img, k1=0.2, k2=0.0):
    if k1 == 0.0 and k2 == 0.0:
        return img
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (X - cx) / (cx + 1e-6)
    yn = (Y - cy) / (cy + 1e-6)
    r2 = xn**2 + yn**2
    factor = 1.0 + k1 * r2 + k2 * r2**2
    factor = np.where(np.abs(factor) < 1e-6, 1e-6, factor)
    src_x = np.clip(xn / factor * cx + cx, 0, w - 1)
    src_y = np.clip(yn / factor * cy + cy, 0, h - 1)
    return cv2.remap(img, src_x, src_y, cv2.INTER_LINEAR)


# ══════════════════════════════════════════════════════════════════════════════
#  Fast
# ══════════════════════════════════════════════════════════════════════════════

def _apply_distortion_fast(img, k1=0.2, k2=0.0):
    """Same expression, with the frame-size half of it cached.

    Everything after `_distortion_grid` depends on k1/k2 and so is recomputed;
    what disappears is the mgrid, the two normalisations and the radius square.
    """
    if k1 == 0.0 and k2 == 0.0:
        return img
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    xn, yn, r2 = _distortion_grid(h, w)
    factor = 1.0 + k1 * r2 + k2 * r2**2
    factor = np.where(np.abs(factor) < 1e-6, 1e-6, factor)
    src_x = np.clip(xn / factor * cx + cx, 0, w - 1)
    src_y = np.clip(yn / factor * cy + cy, 0, h - 1)
    return cv2.remap(img, src_x, src_y, cv2.INTER_LINEAR)


# ══════════════════════════════════════════════════════════════════════════════
#  Public
# ══════════════════════════════════════════════════════════════════════════════

def apply_distortion(img, k1=0.2, k2=0.0):
    """Barrel (k1>0) or pincushion (k1<0) lens distortion (cv2.remap, sub-pixel)."""
    if USE_FAST:
        return _apply_distortion_fast(img, k1, k2)
    return _apply_distortion_legacy(img, k1, k2)


def apply_defocus(img, radius=2):
    """Disk-kernel blur simulating out-of-focus optics."""
    if radius <= 0:
        return img
    size = 2 * radius + 1
    kernel = np.zeros((size, size), np.uint8)
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    kernel = kernel.astype(np.float32) / kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def apply_chromatic(img, shift=2):
    """Lateral chromatic aberration: shift R and B channels in opposite directions."""
    if shift == 0:
        return img
    h, w = img.shape[:2]
    result = img.copy()
    M_r = np.float32([[1, 0,  shift], [0, 1, 0]])
    M_b = np.float32([[1, 0, -shift], [0, 1, 0]])
    result[:, :, 0] = cv2.warpAffine(img[:, :, 0], M_r, (w, h), borderMode=cv2.BORDER_REFLECT)
    result[:, :, 2] = cv2.warpAffine(img[:, :, 2], M_b, (w, h), borderMode=cv2.BORDER_REFLECT)
    return result
