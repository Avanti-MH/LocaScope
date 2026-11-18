'''The views of a place (spec.md §3): the reference tile (`ref`) and its simulated photos. The geometry is the view's own `Geometry`
(a perturbation level's draw, or none: the scan's own pixels), applied first; the simulation is photometric only (colour, blur,
noise on the photo's pixels; no codec) and never moves a pixel. The pixels are made on the GPU (GpuSims.frames, GpuSims.simulate)
from a view's reference read, its geometry and its drawn parameters (`Shot`, `draw_shot`); this module draws, frames (the CPU
definition) and holds the labs.

    labs = sim_labs()                                          # {'bench', 'bench-real', 'ad-chroma4'}: LabGap, photometric only
    geo  = draw_geometry(rng, LEVELS['large'])                 # a Geometry (Geometry() at level '0')
    base = frame(ref_read, geo, photo_px)                      # uint8 [photo_px, photo_px, 3]: the CPU's framing (the tests' reference)
    shot = draw_shot(lab, rng)                                 # the lab's drawn parameters, no pixel made

THE SIMULATORS are scratch/diag/diag_domain_gap.py's `LabGap` / `lab_simulate` (ExpPhaseSeparation's `sim-*` queries), copied here so
that training does not import scratch: `bench` and `bench-real` are the FoV recipes' gaps without their codec, photometric only; `ad-chroma4` is bench-real
with defocus 2, noise 3, chromatic shift 4, colour temperature +-0.12 (scratch/diag/pca_groups_raw.py `AD`). `probe_labs` adds
the two sources the probe holds out (no arm trains on them).

THE GEOMETRY maps a photo pixel to the reference (continuous coordinates, a pixel k covering [k, k + 1)):
    r = c_ref + shift + R(turn) (p - c_photo) / scale
a scale above 1 zooms in (the photo shows less of the reference, as the real photos do: ~1.12). R is the rotation in image
coordinates (x right, y down); a direction in the photo is R(turn) of itself in the reference.
'''
from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from locascope.core.query_sim.FovSupply import FOV_RECIPES
from locascope.core.query_sim.augment.lens import apply_distortion
from locascope.core.query_sim.pipeline import simulate_with_gt
from locascope.training.CoAE.Config import DataConfig, Level

#: ad-chroma4 on top of bench-real (scratch/diag/pca_groups_raw.py `AD`)
AD_CHROMA4: str = 'defocus_radius=2|noise_sigma=3.0|chromatic_shift=4|color_temp_range=-0.12..0.12'

#: RGB -> grey (ITU-R 601), the one definition: the blank rule here and the Co-figure targets (train.targets_of) both use it
GREY_WEIGHTS: Tuple[float, float, float] = (0.299, 0.587, 0.114)


def centre(img: np.ndarray, size: int) -> np.ndarray:
    """The centred `size` x `size` square of an image ([H, W, ...]) or of a stack of them ([N, H, W, ...]), contiguous."""
    h: int = img.shape[-3] // 2                                       # H of [H, W, C] and of [N, H, W, C]
    t: int = size // 2
    return np.ascontiguousarray(img[..., h - t:h + t, h - t:h + t, :])


def grey_sd(img: np.ndarray) -> float:
    """The standard deviation of an RGB tile's grey: what `DataConfig.blank_sd` thresholds."""
    return float((img.astype(np.float32) @ np.array(GREY_WEIGHTS, dtype=np.float32)).std())


@dataclass(frozen=True)
class LabGap:
    """A `DomainGapConfig` (`base`) and the lab's additions (scratch diag_domain_gap.LabGap without the kernel bank): `blur_factor`, a
    range of resolution losses (1 = none); `noise_corr`, the sigma in px of a blur put on the noise (0 = independent)."""
    base: Any
    blur_factor: Tuple[float, float] = (1.0, 1.0)
    noise_corr: float = 0.0

    @property
    def extra(self) -> bool:
        return tuple(self.blur_factor) != (1.0, 1.0) or self.noise_corr > 0


def parse_variant(text: str, base: Any) -> Tuple[str, LabGap]:
    """`name|field=value|...` -> the name and `base` with those fields changed (an int, a float or a range `a..b`); `blur_factor` and
    `noise_corr` are the lab's own."""
    name, *parts = text.split('|')
    kw: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for part in parts:
        k, v = part.split('=', 1)
        if k in ('jpeg', 'jpeg_quality'):
            raise ValueError(f'{name}: CoAE\'s simulators have no codec (the photos are bmp / png): {k} cannot be set')
        val: Any
        if '..' in v:
            val = tuple(float(x) for x in v.split('..'))
        elif v.lstrip('-').isdigit():
            val = int(v)
        else:
            val = float(v)
        (extra if k in ('blur_factor', 'noise_corr') else kw)[k] = val
    return name, LabGap(replace(base, **kw), **extra)


def _base_real() -> Any:
    """The bench-real gap, photometric only."""
    return replace(FOV_RECIPES['bench-real'].gap, geometric=False, photometric=True)


def without_lens(lab: LabGap) -> LabGap:
    """The simulator with its geometric parts off: no lens distortion, no stage shift (the photometric alone)."""
    return replace(lab, base=replace(lab.base, distortion_k1_range=(0.0, 0.0), distortion_k2=0.0, stage_shift_max=0))


def without_codec(lab: LabGap) -> LabGap:
    """The simulator without its JPEG codec: the real photos are stored as bmp / png (the user, 2026-11-15)."""
    return replace(lab, base=replace(lab.base, jpeg=False))


def sim_labs() -> Dict[str, LabGap]:
    """The simulators by name, photometric only: no lens, no stage shift (the geometry is the view's `Geometry` alone), no codec."""
    base_real: Any = _base_real()
    labs: Dict[str, LabGap] = {'bench': LabGap(replace(FOV_RECIPES['bench'].gap, geometric=False, photometric=True)),
                               'bench-real': LabGap(base_real),
                               'ad-chroma4': parse_variant(f'ad-chroma4|{AD_CHROMA4}', base_real)[1]}
    return {name: without_codec(without_lens(lab)) for name, lab in labs.items()}


def probe_labs(ood: Tuple[str, ...]) -> Dict[str, LabGap]:
    """The probe's sources: `sim_labs()` and each `ood` text (`name|field=value|...` on bench-real's base, `parse_variant`), all
    photometric only, no codec; a name twice is an error."""
    labs: Dict[str, LabGap] = sim_labs()
    for text in ood:
        name, lab = parse_variant(text, _base_real())
        if name in labs:
            raise ValueError(f'probe source {name!r} is already a source: {tuple(labs)}')
        labs[name] = without_codec(without_lens(lab))
    return labs


@dataclass(frozen=True)
class Shot:
    """A simulated view's drawn photometric parameters: what `GpuSims.simulate` needs of `simulate_with_gt`'s params, and the lab's
    additions. Applied in this order: saturation, brightness / contrast, colour temperature, defocus, chromatic shift, vignette,
    resolution loss, noise."""
    saturation: float
    brightness: float
    contrast: float
    color_temp: float
    defocus: int          # disk radius, px (0 = none)
    chromatic: int        # R / B shift, px (0 = none)
    vignette: float       # strength (0 = none)
    res_loss: float       # the canvas taken to 1 / this of its size and back (<= 1 = none)
    noise_sigma: float    # grey levels (0 = none)
    noise_corr: float     # sigma in px of the blur put on the noise (0 = independent)
    noise_seed: int       # which noise


_NO_IMAGE: np.ndarray = np.zeros((1, 1, 3), dtype=np.uint8)


def draw_shot(lab: LabGap, rng: random.Random) -> Shot:
    """The photometric simulation's drawn parameters of a view, off `rng` and without making a pixel (`GpuSims.simulate` makes it):
    `simulate_with_gt`'s draws (run with the sensor stage off on a 1 px image: what is drawn does not depend on the image, on
    `output_wh` or on `photometric`: pipeline._sample_params), then the lab's own (the resolution loss, the noise's seed)."""
    p: Dict[str, Any]
    _, p = simulate_with_gt(_NO_IMAGE, cfg=replace(lab.base, photometric=False), rng=rng, output_wh=(1, 1))
    if p['distortion_k1'] != 0.0 or p['distortion_k2'] != 0.0 or p['stage_shift_dx'] != 0 or p['stage_shift_dy'] != 0:
        raise ValueError(f'a CoAE simulator moves no pixel (sim_labs, probe_labs take the lens and the stage shift off): {p}')
    res: float = 1.0
    sigma: float = float(p['noise_sigma'])
    seed: int = int(p['noise_seed'])
    if lab.extra:                                              # the lab's own draws, in this order: the resolution loss, then the noise's seed
        res = rng.uniform(*lab.blur_factor)
        sigma = float(lab.base.noise_sigma)
        seed = rng.getrandbits(32) if sigma > 0 else 0
    return Shot(float(p['saturation']), float(p['brightness']), float(p['contrast']), float(p['color_temp']), int(p['defocus_radius']),
                int(p['chromatic_shift']), float(p['vignette_strength']), res, sigma, float(lab.noise_corr) if lab.extra else 0.0, seed)


@dataclass(frozen=True)
class Geometry:
    """A version's known geometry, in the order the pixels go through it:
    scale, turn, shift   the framing (`frame`): `scale` photo px a reference px, `turn` degrees, `shift` (dy, dx) reference px of
                         the photo's centre from the place's (any real value)
    stage                (dx, dy) whole photo px of stage jitter: now always (0, 0) (the simulators are photometric only); the aligned
                         path (CoFigure) still reads it
    k1, k2               the lens distortion applied after the framing, about the canvas centre, normalised by its half side
                         (`apply_distortion`); part of a perturbation level"""
    scale: float = 1.0
    turn: float = 0.0
    shift: Tuple[float, float] = (0.0, 0.0)
    stage: Tuple[float, float] = (0.0, 0.0)
    k1: float = 0.0
    k2: float = 0.0

    def rotation(self) -> np.ndarray:
        a: float = math.radians(self.turn)
        return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def lens_forward(p: np.ndarray, geo: Geometry, photo_px: int) -> np.ndarray:
    """Where a point p [..., 2] (x, y) of the framed canvas ends up in the final photo: + the stage shift, then the lens distortion
    (apply_distortion samples src = c + (q - c) / f(q), f = 1 + k1 r^2 + k2 r^4, r normalised by (side - 1) / 2; so the point's q
    solves q = c + (p1 - c) f(q): a few fixed-point steps, f being within a few % of 1)."""
    p1: np.ndarray = p + np.array([geo.stage[0], geo.stage[1]], dtype=np.float64)
    if geo.k1 == 0.0 and geo.k2 == 0.0:
        return p1
    c: float = (photo_px - 1) / 2.0
    n: float = (photo_px - 1) / 2.0
    q: np.ndarray = p1.copy()
    for _ in range(6):
        r2: np.ndarray = (((q - c) / n) ** 2).sum(-1, keepdims=True)
        q = c + (p1 - c) * (1.0 + geo.k1 * r2 + geo.k2 * r2 ** 2)
    return q


def draw_geometry(rng: random.Random, level: Level) -> Geometry:
    """A perturbation level's draw: scale 1 +- level.scale, turn +- level.turn_deg, shift +- level.shift_px on each axis, k1 +-
    level.k1, each uniform and any real value (whole cells only would teach a grid). All four 0: `Geometry()`, nothing drawn.
    (CoFigure.place_moments's box average over the shift is this shift distribution's expectation: change both together.)"""
    if level.shift_px == 0.0 and level.turn_deg == 0.0 and level.scale == 0.0 and level.k1 == 0.0:
        return Geometry()
    scale: float = 1.0 + rng.uniform(-level.scale, level.scale)
    turn: float = rng.uniform(-level.turn_deg, level.turn_deg)
    shift: Tuple[float, float] = (rng.uniform(-level.shift_px, level.shift_px), rng.uniform(-level.shift_px, level.shift_px))
    k1: float = rng.uniform(-level.k1, level.k1)
    return Geometry(scale=scale, turn=turn, shift=shift, k1=k1)


def read_px(cfg: DataConfig, photo_px: int) -> int:
    """The reference square a place is read as, round its centre: every photo canvas pixel of any version maps inside it."""
    half: float = photo_px / 2 / (1.0 - cfg.scale)
    a: float = math.radians(cfg.turn_deg)
    reach: float = half * (abs(math.cos(a)) + abs(math.sin(a))) + cfg.shift_px
    return int(math.ceil(2 * reach / 32.0) * 32 + 32)


def frame(ref_read: np.ndarray, geo: Geometry, photo_px: int) -> np.ndarray:
    """The photo canvas of a geometry: the identity is the scan's own pixels (`centre`, no resampling); else the reference read
    (uint8 [L, L, 3], centred on the place) framed by `geo` (cubic), then its lens distortion (when k1 or k2 is not 0). The CPU
    definition of the framing (cv2): the tests' reference and their way to make a view of a known geometry; the training and the
    probe frame on the GPU (GpuSims.frames), held against this one by tests/TestCoAE/test_sim_gpu.py."""
    if geo == Geometry():
        return centre(ref_read, photo_px)
    img: np.ndarray = _frame(ref_read, geo, photo_px)
    if geo.k1 != 0.0 or geo.k2 != 0.0:
        img = apply_distortion(img, k1=geo.k1, k2=geo.k2)
    return img


def frame_matrix(geo: Geometry, big: int, photo_px: int) -> np.ndarray:
    """The framing as a 2 x 3 matrix, float64: photo pixel INDEX (x, y, 1) -> index into the reference read of side `big`
    (dst index p -> c_ref + rot (p + 0.5 - c_ph) - 0.5: pixel k covers [k, k + 1))."""
    rot: np.ndarray = geo.rotation() / geo.scale
    c_ref: np.ndarray = np.array([big / 2 + geo.shift[1], big / 2 + geo.shift[0]])          # (x, y)
    c_ph: float = photo_px / 2
    m: np.ndarray = np.zeros((2, 3))
    m[:, :2] = rot
    m[:, 2] = c_ref - rot @ np.array([c_ph - 0.5, c_ph - 0.5]) - 0.5
    return m


def _frame(ref_read: np.ndarray, geo: Geometry, photo_px: int) -> np.ndarray:
    """The framing alone (scale, turn, shift)."""
    m: np.ndarray = frame_matrix(geo, ref_read.shape[0], photo_px)
    return cv2.warpAffine(ref_read, m, (photo_px, photo_px), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
