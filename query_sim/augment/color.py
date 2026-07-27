import cv2
import numpy as np


def apply_color(img, brightness=0, contrast=1.0, saturation=1.0):
    """Adjust brightness, contrast, and saturation in HSV space (RGB input)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * contrast + brightness, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def apply_color_temp(img, temp=0.0):
    """Warm/cool colour-temperature shift. temp>0 warmer (R up, B down)."""
    if temp == 0.0:
        return img
    out = img.astype(np.float32)
    out[..., 0] = np.clip(out[..., 0] * (1 + temp), 0, 255)   # R
    out[..., 2] = np.clip(out[..., 2] * (1 - temp), 0, 255)   # B
    return out.astype(np.uint8)


def apply_brightness_contrast(img, brightness=0.0, contrast=0.0):
    """Linear brightness (additive as *(1+b)) + contrast about image mean.

    Both args are fractions: 0 means no change, +0.1 = +10% brighter / +10% more contrast.
    """
    if brightness == 0.0 and contrast == 0.0:
        return img
    out = img.astype(np.float32)
    out = out * (1 + brightness)
    mean = out.mean()
    out = (out - mean) * (1 + contrast) + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_jpeg(img, quality=85):
    """Simulate JPEG compression artifacts (RGB input)."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
