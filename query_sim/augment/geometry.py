import cv2
import numpy as np

#: Route the public functions through the `_fast` bodies. The `_legacy` bodies
#: stay next to them so test_augment_equivalence.py can measure the two against
#: real Camera output rather than against a claim. Flip to False to fall back.
USE_FAST = True


# ══════════════════════════════════════════════════════════════════════════════
#  Legacy
# ══════════════════════════════════════════════════════════════════════════════

def _apply_rotation_legacy(img, angle=0.0):
    angle_mod = float(angle) % 360.0
    if angle_mod == 0.0:
        return img.copy()
    if angle_mod == 90.0:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle_mod == 180.0:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle_mod == 270.0:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )


def _apply_scale_legacy(img, scale=1.0):
    if scale == 1.0:
        return img.copy()
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if scale > 1.0:
        y0 = (new_h - h) // 2
        x0 = (new_w - w) // 2
        return resized[y0:y0 + h, x0:x0 + w]

    pad_y = (h - new_h) // 2
    pad_x = (w - new_w) // 2
    return cv2.copyMakeBorder(
        resized, pad_y, h - new_h - pad_y, pad_x, w - new_w - pad_x,
        borderType=cv2.BORDER_REFLECT,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Fast
# ══════════════════════════════════════════════════════════════════════════════
#
# The only difference is that a no-op does not copy. On the 1767x1767 bounding
# square each copy is 9.4 MB, and the retrieval benches hold rotation at 0 and
# scale at 1 on purpose, so BOTH fire on every single shot.
#
# The values are identical; what changes is ownership. `_as_rgb_uint8` does
# `np.asarray(img)` for an ndarray caller, which does not copy, so with the
# copies gone `simulate_with_gt` could hand back the caller's own array when
# `cfg.photometric` is off. `pipeline._apply_params` closes that at the one
# point where it can happen -- see the comment there.

def _apply_rotation_fast(img, angle=0.0):
    angle_mod = float(angle) % 360.0
    if angle_mod == 0.0:
        return img
    return _apply_rotation_legacy(img, angle)


def _apply_scale_fast(img, scale=1.0):
    if scale == 1.0:
        return img
    return _apply_scale_legacy(img, scale)


# ══════════════════════════════════════════════════════════════════════════════
#  Public
# ══════════════════════════════════════════════════════════════════════════════

def apply_rotation(img, angle=0.0):
    """Rotate about center, keep same (H, W). 90/180/270 走 lossless cv2.rotate。

    angle: degrees, positive = counter-clockwise (per cv2 convention).
    """
    if USE_FAST:
        return _apply_rotation_fast(img, angle)
    return _apply_rotation_legacy(img, angle)


def apply_scale(img, scale=1.0):
    """Resize by `scale` factor, then centre-crop / centre-pad back to input size.

    scale > 1 → zoom in (crop center); scale < 1 → zoom out (pad borders reflected).
    """
    if USE_FAST:
        return _apply_scale_fast(img, scale)
    return _apply_scale_legacy(img, scale)
