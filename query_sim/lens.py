import cv2
import numpy as np


def apply_distortion(img, k1=0.2, k2=0.0):
    """Barrel (k1>0) or pincushion (k1<0) lens distortion."""
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
