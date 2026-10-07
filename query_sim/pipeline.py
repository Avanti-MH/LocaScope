"""Layer 1: img + cfg -> augmented img + params dict.

`simulate_with_gt(img, cfg)` returns (img, dict) with every sampled value,
which `FOVRecord.from_capture` folds into a row.
"""

from __future__ import annotations

import math
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


def _coin(rng: random.Random, p: float) -> bool:
    """Is this op present on this shot?

    NO DRAW AT ALL at p >= 1.0 (nor at p <= 0.0). That is what keeps the rng
    sequence byte-identical for every config that leaves these probabilities
    at their 1.0 default -- i.e. every corpus generated before they existed
    still reproduces exactly.
    """
    if p >= 1.0:
        return True
    if p <= 0.0:
        return False
    return rng.random() < p


def _maybe(rng: random.Random, p: float, lo_hi: Tuple[float, float]) -> float:
    """`_uniform` with probability `p`, else 0.0 -- the neutral value for both
    ops that use this (`apply_vignette(strength=0)` and
    `apply_distortion(k1=0, k2=0)` are identities).

    The strength is NOT drawn when the coin says absent, so the recorded
    parameter is the value that was ACTUALLY applied rather than a value drawn
    beside the one applied. That second shape is the `stage_shift_dx/dy` bug
    (`augment/field.py`), and it is worth not repeating.
    """
    return _uniform(rng, lo_hi) if _coin(rng, p) else 0.0


def _sample_params(cfg: DomainGapConfig, rng: random.Random,
                   mpp: Optional[float] = None) -> dict:
    """Sample one concrete set of augment values from a cfg's ranges. `mpp` is
    the camera's nominal mpp; `effective_mpp` is None without one."""
    dx = rng.randint(-cfg.stage_shift_max, cfg.stage_shift_max) if cfg.stage_shift_max > 0 else 0
    dy = rng.randint(-cfg.stage_shift_max, cfg.stage_shift_max) if cfg.stage_shift_max > 0 else 0

    # mpp jitter overrides scale_range: scale is drawn from (1-j, 1+j) so that
    # effective_mpp = mpp / scale falls within +/- jitter of nominal.
    if cfg.query_mpp_jitter > 0:
        j = cfg.query_mpp_jitter
        scale = _uniform(rng, (1.0 - j, 1.0 + j))
    else:
        scale = _uniform(rng, cfg.scale_range)

    # The whole lens distortion is one coin, not two: k2 is a cfg constant
    # rather than a drawn value, so deciding it from `k1 == 0` would read the
    # coin off a value that can legitimately be zero on its own. Drawn once
    # here and both terms take it.
    lens_on = _coin(rng, cfg.distortion_p)

    return {
        'rot_deg':           rng.choice(cfg.rotation_choices),
        'angle_jitter':      _uniform(rng, (-cfg.angle_jitter_deg, cfg.angle_jitter_deg)),
        'scale':             scale,
        'effective_mpp':     None if mpp is None else float(mpp) / scale,
        'brightness':        _uniform(rng, cfg.brightness_range),
        'contrast':          _uniform(rng, cfg.contrast_range),
        'color_temp':        _uniform(rng, cfg.color_temp_range),
        'vignette_strength': _maybe(rng, cfg.vignette_p, cfg.vignette_range),
        'distortion_k1':     _uniform(rng, cfg.distortion_k1_range) if lens_on else 0.0,
        'defocus_radius':    cfg.defocus_radius,
        'chromatic_shift':   cfg.chromatic_shift,
        'stage_shift_dx':    dx,
        'stage_shift_dy':    dy,
        'noise_sigma':       cfg.noise_sigma,
        # Which noise, not how much: a shot's noise is img.shape worth of
        # values, so it cannot be recorded the way every other parameter here
        # is. The seed can, and it comes off the same `rng` as everything else
        # -- which is what puts the noise under `Render(seed=)`/`capture(rng=)`.
        # Nothing is expected to READ this back (see
        # training/MppRoutingHead/spec.md on recording for reproducibility
        # rather than for correction); it exists so the same seed reproduces
        # the same shot.
        'noise_seed':        rng.getrandbits(32),
        'jpeg_quality':      cfg.jpeg_quality,
        'saturation':        cfg.saturation,
        # k2 rides on the SAME coin: `distortion_p` turns the lens distortion
        # off, and leaving a non-zero k2 behind would leave it half on.
        'distortion_k2':     cfg.distortion_k2 if lens_on else 0.0,
    }


# ── how far one exposure samples, before anything is read ────────────────────
#
# Walked backwards from the sensor's corner through `_apply_params`, every op at
# the most outward value the config can draw, one px for every resample. The
# read is sized from this (`camera.read_margin`), so no pixel a photo shows
# comes from outside what was read.

def lens_margin(cfg: DomainGapConfig, sensor: Tuple[int, int]) -> Tuple[int, int]:
    """`(mx, my)` output px the sensor stage samples beyond the sensor, per
    side: chromatic shift and defocus, then the distortion at its most
    outward k1. (0, 0) when the sensor stage is off."""
    if not cfg.photometric:
        return 0, 0
    w, h = (int(v) for v in sensor)
    nx, ny = (w - 1) / 2.0, (h - 1) / 2.0
    bx = nx + cfg.chromatic_shift + cfg.defocus_radius
    by = ny + cfg.defocus_radius
    r2 = (bx / nx) ** 2 + (by / ny) ** 2
    lens = [1.0] + [1.0 + k1 * r2 + cfg.distortion_k2 * r2 ** 2
                    for k1 in cfg.distortion_k1_range]
    factor = min(1.0, *lens)
    if factor <= 0.0:
        raise ValueError(f'distortion folds the frame at the sensor corner '
                         f'(factor {factor:.3f}): k1 {cfg.distortion_k1_range}, '
                         f'k2 {cfg.distortion_k2}')
    sx, sy = bx / factor + 1.0, by / factor + 1.0
    return max(0, math.ceil(sx - nx)), max(0, math.ceil(sy - ny))


def _angles(cfg: DomainGapConfig, rotation: Optional[float]):
    if rotation is not None:
        return [float(rotation)]
    j = float(cfg.angle_jitter_deg)
    steps = max(1, int(math.ceil(2 * j / 0.05)))
    return [float(c) - j + 2 * j * i / steps
            for c in cfg.rotation_choices for i in range(steps + 1)]


def read_reach(cfg: DomainGapConfig, sensor: Tuple[int, int],
               rotation: Optional[float] = None) -> Tuple[float, float]:
    """`(ex, ey)`: the half-extents, in read px about the FoV centre (pixel
    centres), that one exposure can sample -- the lens frame, grown by the
    stage shift, divided by the smallest scale, turned through every angle the
    config can draw (or `rotation`)."""
    w, h = (int(v) for v in sensor)
    mx, my = lens_margin(cfg, sensor)
    fx, fy = (w - 1) / 2.0 + mx, (h - 1) / 2.0 + my
    if cfg.stage_shift_max > 0:
        fx, fy = fx + cfg.stage_shift_max, fy + cfg.stage_shift_max
    if not cfg.geometric:
        return fx, fy
    j = cfg.query_mpp_jitter
    low = (1.0 - j) if j > 0 else float(cfg.scale_range[0])
    if low < 1.0:
        fx, fy = fx / low + 1.0, fy / low + 1.0
    ex = ey = 0.0
    turned = False
    for a in _angles(cfg, rotation):
        c, s = abs(math.cos(math.radians(a))), abs(math.sin(math.radians(a)))
        ex, ey = max(ex, fx * c + fy * s), max(ey, fx * s + fy * c)
        turned = turned or a % 360.0 != 0.0
    return (ex + 1.0, ey + 1.0) if turned else (ex, ey)


def _centre_crop(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Centre-crop to (width, height); a no-op when the frame is already that
    size or smaller. Every crop in this file is centred, which is what keeps
    `Render.output_to_level0` a pure rotate-and-scale about the FoV centre."""
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

    The scene stage ends by cropping to the sensor plus `lens_margin` -- what the
    sensor stage reaches -- so the optics run on the sensor and that margin
    only, and each is measured against the sensor: the vignette after the
    crop to it, the distortion normalised to it.

    `output_wh` is the sensor size. None means the input already is the sensor,
    which leaves both crops as no-ops -- that is the path
    a caller takes when it augments a whole image.
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
        # the other framing decisions, before the vignette: moving a frame
        # that is already lit would shift the vignette's centre off the
        # optical axis, where it is physically fixed.
        img = apply_stage_shift(img, dx=p['stage_shift_dx'],
                                dy=p['stage_shift_dy'])

    mx, my = lens_margin(cfg, (out_w, out_h))
    img = _centre_crop(img, out_w + 2 * mx, out_h + 2 * my)

    if not cfg.photometric:
        img = _centre_crop(img, out_w, out_h)
        # `apply_rotation(.., 0)` and `apply_scale(.., 1)` do not copy, and
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
    img = apply_distortion(img, k1=p['distortion_k1'], k2=p['distortion_k2'],
                           sensor=(out_w, out_h))
    img = apply_defocus(img, radius=p['defocus_radius'])
    img = apply_chromatic(img, shift=p['chromatic_shift'])

    img = _centre_crop(img, out_w, out_h)

    # Per-pixel from here down, so they run on the exact sensor frame: the
    # vignette's falloff is then measured against the real half-width, and
    # JPEG's blocks tile the delivered image rather than a padded one.
    img = apply_vignette(img, strength=p['vignette_strength'])
    img = apply_noise(img, sigma=p['noise_sigma'], seed=p['noise_seed'])
    img = apply_jpeg(img, quality=p['jpeg_quality'])
    return img


def simulate_with_gt(
    img,
    cfg:      Optional[DomainGapConfig] = None,
    rng:      Optional[random.Random]   = None,
    rotation:  Optional[float]           = None,
    output_wh: Optional[Tuple[int, int]]  = None,
    mpp:       Optional[float]           = None,
) -> Tuple[np.ndarray, dict]:
    """Return (augmented_img, params_dict). `params_dict` is what got sampled;
    its `effective_mpp` is `mpp / scale`, None when no `mpp` is given.

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
    params = _sample_params(cfg, rng, mpp)
    if rotation is not None:
        rot_int = int(round(float(rotation)))
        params['rot_deg']      = rot_int
        params['angle_jitter'] = float(rotation) - rot_int
    out = _apply_params(arr, cfg, params, output_wh=output_wh)
    return out, params
