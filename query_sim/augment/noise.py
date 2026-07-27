import numpy as np


def apply_noise(img, sigma=4.0):
    """Add Gaussian sensor noise (sigma in 0-255 space)."""
    if sigma <= 0:
        return img
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
