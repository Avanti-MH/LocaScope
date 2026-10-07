"""The domain gap a microscope photo carries: what is done to the pixels.

Every knob is expressed as a `(lo, hi)` range so each photo (`Render`) can
sample per FOV. For a fixed / deterministic run, set `lo == hi`.

NOT the sensor and NOT the magnification. Both are the camera's
(`Render(reader, sensor, cfg, ds=...)`), given in whole pixels and as a
downsample, so the same gap can be put on any sensor at any objective.
"""

from dataclasses import dataclass
from typing import Tuple

from ConfigIdentity import IdentifiedConfig, register


@register('domain-gap')
@dataclass(frozen=True)
class DomainGapConfig(IdentifiedConfig):
    # ── Geometry ──────────────────────────────────────────────────────────────
    rotation_choices: Tuple[int, ...] = (0, 90, 180, 270)
    angle_jitter_deg: float = 3.0
    scale_range:      Tuple[float, float] = (0.90, 1.15)

    # ── mpp calibration jitter ────────────────────────────────────────────────
    # >0 turns on mpp-jitter mode: per-shot scale is drawn from
    # (1-jitter, 1+jitter) INSTEAD OF scale_range, simulating a mis-calibrated
    # microscope. effective_mpp = the camera's mpp / scale is recorded per shot.
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

    # ── Per-shot presence of the two FRAME-REFERENCED optics ──────────────────
    # Probability that the vignette / the distortion is applied AT ALL. 1.0 is
    # "always"; `_sample_params` does not draw when the probability is 1.0, so
    # the default leaves the rng sequence of the other draws untouched.
    #
    # A probability rather than a wider range, because these two are not
    # continuous in the way brightness is. `apply_vignette`'s falloff and
    # `apply_distortion`'s k1 are normalised to the SENSOR's half-width
    # (`pipeline._apply_params` says so), so their meaning depends on what the
    # frame is. A caller rendering one tile rather than a whole field of view
    # is rendering something that may be a tile from the CENTRE of a frame --
    # where there is no vignette at all -- or from its edge. Drawing strength
    # from (0, 0.45) models only the second kind, ever more weakly; a
    # probability models both kinds.
    vignette_p:   float = 1.0
    distortion_p: float = 1.0
    defocus_radius:      int                 = 2
    chromatic_shift:     int                 = 2

    # ── Noise + JPEG ──────────────────────────────────────────────────────────
    noise_sigma:  float = 3.0
    jpeg_quality: int   = 85

    # ── Mode toggles ──────────────────────────────────────────────────────────
    photometric: bool = True   # if False, skip color/vignette/lens/noise/jpeg
    geometric:   bool = True   # if False, skip rotation + scale

    BASELINE = {
        'rotation_choices': (0, 90, 180, 270), 'angle_jitter_deg': 3.0,
        'scale_range': (0.90, 1.15), 'query_mpp_jitter': 0.0,
        'brightness_range': (-0.08, 0.08), 'contrast_range': (-0.08, 0.08),
        'saturation': 1.0, 'color_temp_range': (-0.12, 0.12),
        'vignette_range': (0.15, 0.45), 'stage_shift_max': 3,
        'distortion_k1_range': (-0.04, 0.04), 'distortion_k2': 0.0,
        'vignette_p': 1.0, 'distortion_p': 1.0, 'defocus_radius': 2,
        'chromatic_shift': 2, 'noise_sigma': 3.0, 'jpeg_quality': 85,
        'photometric': True, 'geometric': True}

    def __post_init__(self):
        for name in (
            'scale_range', 'brightness_range', 'contrast_range',
            'color_temp_range', 'vignette_range', 'distortion_k1_range',
        ):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f'{name}={lo, hi}: lo must be <= hi')
        for name in ('vignette_p', 'distortion_p'):
            p = getattr(self, name)
            if not 0.0 <= p <= 1.0:
                raise ValueError(f'{name}={p}: must be a probability in [0, 1]')
