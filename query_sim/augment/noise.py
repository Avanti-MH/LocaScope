import numpy as np


def apply_noise(img, sigma=4.0, seed=None):
    """Add Gaussian sensor noise (sigma in 0-255 space).

    `seed` decides WHICH noise, and exists because sigma alone cannot be a
    record of what happened: a shot's noise is `img.shape` worth of values,
    so the "draw it in `_sample_params`, record it in `params`, pass it here"
    pattern every other augment parameter follows cannot carry the noise
    itself -- it carries the seed that generates it instead.

    None means "draw from the global numpy state", which is this function's
    behaviour before 2026-09-16 and is kept for callers that only want SOME
    noise and do not care which (`cli/demo.py`'s effect panel). Every caller
    that needs a reproducible shot passes a seed: `pipeline._apply_params`
    passes `p['noise_seed']`, which `_sample_params` drew from the caller's
    own rng -- so `Render(seed=...)` and `Render.capture(rng=...)` both reach
    the noise, which they could not while this function read `np.random`
    directly.
    """
    if sigma <= 0:
        return img
    gen = np.random.default_rng(seed) if seed is not None else np.random
    noise = gen.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
