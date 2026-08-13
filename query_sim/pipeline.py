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
from augment.field    import apply_vignette, apply_stage_shift
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
    }


#: Pixels kept outside the sensor frame while the ops that read a neighbourhood
#: run, then dropped. `apply_defocus` reaches `radius` px and `apply_chromatic`
#: reaches `shift` -- both 2 by default, both trivial. `apply_distortion` is
#: what sets this number, and it is derived rather than guessed.
#:
#: Under pincushion (k1 < 0) `remap` samples OUTWARD, and `src_x` is clipped to
#: the frame, so a frame that is too tight smears its own corner. The output
#: corner sits (719.5, 511.5) from the centre whatever the margin is -- the
#: margin moves the centre, not the corner -- while the frame's half-width is
#: 719.5 + M. Nothing is clipped when
#:
#:     719.5 / factor + cx <= 2*cx      i.e.   factor >= 719.5 / (719.5 + M)
#:     factor = 1 + k1*r2,  worst case k1 = -0.04 (distortion_k1_range's floor)
#:
#: at 1440x1024:   M=32 gives factor 0.9279 against 0.9574 needed -- CLIPPED.
#:                 M=56 is the crossing.  M=64 gives 0.9347 against 0.9183,
#:                 13.7 px of headroom, and is what is used.
#:
#: The old order never hit this: distortion ran on the 1767^2 bounding square,
#: where the output corner sampled at 1632 against a 1766 edge. Cropping first
#: is what makes the margin load-bearing, so it is sized to the same worst case
#: the square used to absorb for free.
SENSOR_MARGIN = 64


def _centre_crop(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Centre-crop to (width, height); a no-op when the frame is already that
    size or smaller. Every crop in this file is centred, which is what keeps
    `Camera.output_to_level0` a pure rotate-and-scale about the FoV centre."""
    h, w = img.shape[:2]
    if w <= width and h <= height:
        return img
    x0 = max(0, (w - width) // 2)
    y0 = max(0, (h - height) // 2)
    return img[y0:y0 + min(height, h), x0:x0 + min(width, w)]


def _apply_params(img: np.ndarray, cfg: DomainGapConfig, p: dict,
                  output_wh: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Apply one sampled parameter set to img.

    Two stages, and which stage an op belongs to is decided by which frame its
    geometry is measured against.

    SCENE STAGE -- what lands on the sensor. Rotation, scale and stage shift
    move the slide under the objective, so they run on the oversized read while
    there are still pixels outside the frame to rotate in. The crop to the
    sensor happens at the END of this stage.

    SENSOR STAGE -- what the optics and the sensor do to that image. Vignetting
    falls off from the optical axis, distortion is normalised to the sensor
    half-width, and JPEG's 8x8 blocks tile the delivered photograph. All three
    read a frame size, so all three must see the sensor's, not the read's.

    Running the sensor stage on the oversized read is what the earlier order
    did, and it silently weakened every frame-referenced effect: on a
    1767x1767 read cropped to 1440x1024, the vignette's sigma came from the
    read's half-width, so the photograph only ever saw the central 53% of the
    falloff curve. Cropping first is therefore not an optimisation that happens
    to be faster -- it is what makes these effects mean what they are named.
    It IS also 2.12x less work for eleven of the twelve ops.

    `output_wh` is the sensor size. None means the input already is the sensor,
    which leaves both crops as no-ops -- that is the path
    `simulate_microscope_photo` takes when it augments a whole image.
    """
    source = img
    height, width = img.shape[:2]
    out_w, out_h = output_wh if output_wh else (width, height)

    # ── scene stage: choose what lands on the sensor ──────────────────────────
    if cfg.geometric:
        angle = p['rot_deg'] + p['angle_jitter']
        img = apply_rotation(img, angle)
        img = apply_scale(img, p['scale'])
    if cfg.stage_shift_max > 0:
        # Mechanical stage jitter re-aims the field of view, so it belongs with
        # the other framing decisions. It used to sit between the vignette and
        # the lens, which meant the frame was lit and THEN moved -- shifting the
        # vignette's centre off the optical axis, where it is physically fixed.
        img = apply_stage_shift(img, max_shift=cfg.stage_shift_max)

    img = _centre_crop(img, out_w + 2 * SENSOR_MARGIN, out_h + 2 * SENSOR_MARGIN)

    if not cfg.photometric:
        img = _centre_crop(img, out_w, out_h)
        # `apply_rotation(.., 0)` and `apply_scale(.., 1)` no longer copy, and
        # `_as_rgb_uint8` does not copy an ndarray caller's array either, so
        # without this the no-augmentation path could hand back the caller's
        # own buffer and any later write to the "output" would edit their
        # input. Every op below allocates, so this is the only route that can
        # alias.
        return img.copy() if img is source else img

    # ── sensor stage: scene colour -> optics -> sensor -> codec ───────────────
    img = apply_color(img, brightness=0, contrast=1.0, saturation=p['saturation'])
    img = apply_brightness_contrast(img, brightness=p['brightness'], contrast=p['contrast'])
    img = apply_color_temp(img, temp=p['color_temp'])

    # These three read a neighbourhood, so they run while the margin is still
    # there and the sensor's own edge pixels have real neighbours.
    img = apply_distortion(img, k1=p['distortion_k1'], k2=p['distortion_k2'])
    img = apply_defocus(img, radius=p['defocus_radius'])
    img = apply_chromatic(img, shift=p['chromatic_shift'])

    img = _centre_crop(img, out_w, out_h)

    # Per-pixel from here down, so they run on the exact sensor frame: the
    # vignette's falloff is then measured against the real half-width, and
    # JPEG's blocks tile the delivered image rather than a padded one.
    img = apply_vignette(img, strength=p['vignette_strength'])
    img = apply_noise(img, sigma=p['noise_sigma'])
    img = apply_jpeg(img, quality=p['jpeg_quality'])
    return img


def simulate_with_gt(
    img,
    cfg:      Optional[DomainGapConfig] = None,
    rng:      Optional[random.Random]   = None,
    rotation:  Optional[float]           = None,
    output_wh: Optional[Tuple[int, int]]  = None,
) -> Tuple[np.ndarray, dict]:
    """Return (augmented_img, params_dict). `params_dict` is what got sampled.

    `output_wh` is the sensor size. Give it and the augmented image comes back
    already cropped to it, with every frame-referenced effect measured against
    that frame; leave it None and the input is treated as the sensor.

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
    out = _apply_params(arr, cfg, params, output_wh=output_wh)
    return out, params


def simulate_microscope_photo(
    img,
    cfg:      Optional[DomainGapConfig] = None,
    rng:      Optional[random.Random]   = None,
    rotation:  Optional[float]           = None,
    output_wh: Optional[Tuple[int, int]]  = None,
) -> np.ndarray:
    """Return augmented img only (backward-compat entry point for existing callers)."""
    out, _ = simulate_with_gt(img, cfg=cfg, rng=rng, rotation=rotation,
                              output_wh=output_wh)
    return out
