import cv2
import numpy as np


def apply_rotation(img, angle=0.0):
    """Rotate about center, keep same (H, W). 90/180/270 走 lossless cv2.rotate。

    angle: degrees, positive = counter-clockwise (per cv2 convention).
    """
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


def apply_scale(img, scale=1.0):
    """Resize by `scale` factor, then centre-crop / centre-pad back to input size.

    scale > 1 → zoom in (crop center); scale < 1 → zoom out (pad borders reflected).
    """
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
