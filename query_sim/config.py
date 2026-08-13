"""Unified microscope-FOV simulation config.

Every knob is expressed as a `(lo, hi)` range so `generator.generate()` can
sample per FOV. For a fixed / deterministic run, set `lo == hi`.

`pipeline.simulate_microscope_photo` treats a config as either:
  - ranges (samples uniformly per-call), or
  - fixed values when both ends of a range are equal.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DomainGapConfig:
    # ── Source (query crop shape, consumed by QueryFromWSI) ───────────────────
    wh_ratio: str = '4:3'
    MPixels: float = 12.0
    query_mpp: float = 0.25

    # ── Geometry ──────────────────────────────────────────────────────────────
    rotation_choices: Tuple[int, ...] = (0, 90, 180, 270)
    angle_jitter_deg: float = 3.0
    scale_range:      Tuple[float, float] = (0.90, 1.15)

    # ── mpp calibration jitter ────────────────────────────────────────────────
    # >0 turns on mpp-jitter mode: per-shot scale is drawn from
    # (1-jitter, 1+jitter) INSTEAD OF scale_range, simulating a mis-calibrated
    # microscope. effective_mpp = query_mpp / scale is recorded per shot.
    # 0 = off, use scale_range normally.
    query_mpp_jitter: float = 0.0

    # ── Colour ────────────────────────────────────────────────────────────────
    brightness_range: Tuple[float, float] = (-0.08, 0.08)
    contrast_range:   Tuple[float, float] = (-0.08, 0.08)
    saturation:       float               = 1.0
    color_temp_range: Tuple[float, float] = (-0.12, 0.12)

    # ── Field ─────────────────────────────────────────────────────────────────
    vignette_range:  Tuple[float, float] = (0.15, 0.45)
    stage_shift_max: int                 = 3

    # ── Lens ──────────────────────────────────────────────────────────────────
    distortion_k1_range: Tuple[float, float] = (-0.04, 0.04)
    distortion_k2:       float               = 0.0
    defocus_radius:      int                 = 2
    chromatic_shift:     int                 = 2

    # ── Noise + JPEG ──────────────────────────────────────────────────────────
    noise_sigma:  float = 3.0
    jpeg_quality: int   = 85

    # ── Mode toggles ──────────────────────────────────────────────────────────
    photometric: bool = True   # if False, skip color/vignette/lens/noise/jpeg
    geometric:   bool = True   # if False, skip rotation + scale

    def __post_init__(self):
        for name in (
            'scale_range', 'brightness_range', 'contrast_range',
            'color_temp_range', 'vignette_range', 'distortion_k1_range',
        ):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f'{name}={lo, hi}: lo must be <= hi')
