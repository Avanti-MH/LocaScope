"""Layer 1: img + cfg -> augmented img (+ params dict).

`simulate_microscope_photo(img)` — old signature kept (no cfg == defaults)
so existing callers (test_gigapath_knn_esti_mpp, notebooks) don't break.

`simulate_with_gt(img, cfg)` — returns (img, dict) with every sampled value,
which `generator.py` folds into a FOVRecord row.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from config import DomainGapConfig
from augment.color    import (
    apply_color, apply_color_temp, apply_brightness_contrast, apply_jpeg,
)
from augment.field    import apply_field_mask, apply_vignette, apply_stage_shift
from augment.lens     import apply_distortion, apply_defocus, apply_chromatic
from augment.geometry import apply_rotation, apply_scale
from augment.noise    import apply_noise


def _as_rgb_uint8(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.array(img.convert('RGB'))
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _uniform(rng: random.Random, lo_hi: Tuple[float, float]) -> float:
    lo, hi = lo_hi
    return lo if lo == hi else rng.uniform(lo, hi)


def _sample_params(cfg: DomainGapConfig, rng: random.Random) -> dict:
    """Sample one concrete set of augment values from a cfg's ranges."""
    dx = rng.randint(-cfg.stage_shift_max, cfg.stage_shift_max) if cfg.stage_shift_max > 0 else 0
    dy = rng.randint(-cfg.stage_shift_max, cfg.stage_shift_max) if cfg.stage_shift_max > 0 else 0

    # mpp jitter overrides scale_range: scale is drawn from (1-j, 1+j) so that
    # effective_mpp = query_mpp / scale falls within +/- jitter of nominal.
    if cfg.query_mpp_jitter > 0:
        j = cfg.query_mpp_jitter
        scale = _uniform(rng, (1.0 - j, 1.0 + j))
    else:
        scale = _uniform(rng, cfg.scale_range)

    return {
        'rot_deg':           rng.choice(cfg.rotation_choices),
        'angle_jitter':      _uniform(rng, (-cfg.angle_jitter_deg, cfg.angle_jitter_deg)),
        'scale':             scale,
        'effective_mpp':     cfg.query_mpp / scale,
        'brightness':        _uniform(rng, cfg.brightness_range),
        'contrast':          _uniform(rng, cfg.contrast_range),
        'color_temp':        _uniform(rng, cfg.color_temp_range),
        'vignette_strength': _uniform(rng, cfg.vignette_range),
        'distortion_k1':     _uniform(rng, cfg.distortion_k1_range),
        'defocus_radius':    cfg.defocus_radius,
        'chromatic_shift':   cfg.chromatic_shift,
        'stage_shift_dx':    dx,
        'stage_shift_dy':    dy,
        'noise_sigma':       cfg.noise_sigma,
        'jpeg_quality':      cfg.jpeg_quality,
        'saturation':        cfg.saturation,
        'distortion_k2':     cfg.distortion_k2,
        'field_mask':        cfg.field_mask,
    }


def _apply_params(img: np.ndarray, cfg: DomainGapConfig, p: dict) -> np.ndarray:
    """Apply one sampled parameter set to img. Order matters — see comments."""
    # Geometry first (rotation + scale). The result may hit borders which the
    # photometric stage will paint over.
    if cfg.geometric:
        angle = p['rot_deg'] + p['angle_jitter']
        img = apply_rotation(img, angle)
        img = apply_scale(img, p['scale'])

    if not cfg.photometric:
        return img

    # Photometric chain. Order mirrors the physical pipeline:
    #   scene colour -> optics (field/lens) -> sensor (noise) -> codec (jpeg)
    img = apply_color(img, brightness=0, contrast=1.0, saturation=p['saturation'])
    img = apply_brightness_contrast(img, brightness=p['brightness'], contrast=p['contrast'])
    img = apply_color_temp(img, temp=p['color_temp'])

    if p['field_mask']:
        img = apply_field_mask(img)
    img = apply_vignette(img, strength=p['vignette_strength'])
    if cfg.stage_shift_max > 0:
        img = apply_stage_shift(img, max_shift=cfg.stage_shift_max)

    img = apply_distortion(img, k1=p['distortion_k1'], k2=p['distortion_k2'])
    img = apply_defocus(img, radius=p['defocus_radius'])
    img = apply_chromatic(img, shift=p['chromatic_shift'])

    img = apply_noise(img, sigma=p['noise_sigma'])
    img = apply_jpeg(img, quality=p['jpeg_quality'])
    return img


def simulate_with_gt(
    img,
    cfg:      Optional[DomainGapConfig] = None,
    rng:      Optional[random.Random]   = None,
    rotation: Optional[float]           = None,
) -> Tuple[np.ndarray, dict]:
    """Return (augmented_img, params_dict). `params_dict` is what got sampled.

    `rotation` overrides the cfg-sampled rotation:
      - None  -> rot_deg + angle_jitter drawn from cfg.rotation_choices + cfg.angle_jitter_deg
      - float -> rot_deg = int(round(rotation)), angle_jitter = rotation - rot_deg
                (so FOVRecord's rot_deg + angle_jitter still sum to the exact angle)
    """
    cfg = cfg or DomainGapConfig()
    rng = rng or random
    arr = _as_rgb_uint8(img)
    params = _sample_params(cfg, rng)
    if rotation is not None:
        rot_int = int(round(float(rotation)))
        params['rot_deg']      = rot_int
        params['angle_jitter'] = float(rotation) - rot_int
    out = _apply_params(arr, cfg, params)
    return out, params


def simulate_microscope_photo(
    img,
    cfg:      Optional[DomainGapConfig] = None,
    rng:      Optional[random.Random]   = None,
    rotation: Optional[float]           = None,
) -> np.ndarray:
    """Return augmented img only (backward-compat entry point for existing callers)."""
    out, _ = simulate_with_gt(img, cfg=cfg, rng=rng, rotation=rotation)
    return out
