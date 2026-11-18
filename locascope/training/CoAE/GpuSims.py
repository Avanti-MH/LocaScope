'''The simulated views' pixels, made on a torch device: the sensor stage of `core.query_sim.pipeline._apply_params` (no lens, no stage
shift, no codec) plus the lab's resolution loss and correlated noise (`LabGap`'s own: `Sims.draw_shot`), a batch at a time; and the framing of
the views (`Sims.frame`, the CPU's cv2 definition) on the device.

    shots = [draw_shot(lab, rng) ...]                          # Sims: the drawn parameters of each view (a worker's job)
    sim   = simulate(base, shots)                              # uint8 [n, H, W, 3] on any device -> uint8 [n, H, W, 3] there
    bases = frames(refs, rows, geos, photo_px)                 # uint8 [m, photo_px, photo_px, 3]: the framed canvases

What `simulate` does, in `Shot`'s order: saturation, brightness / contrast, colour temperature, defocus, chromatic shift, vignette,
resolution loss, noise. Every step is made on whole numbers (float32 holding 0..255) and ends as the CPU's step ends: clipped and cut
down to a whole number (`np.clip(...).astype(np.uint8)`), or rounded where cv2 rounds a uint8 result. One view's output is decided
by its own base and its own `Shot` alone: no step reads another view of the batch (a mean or an sd is each canvas's own), and the
noise of a view is a `torch.Generator` seeded with its `Shot.noise_seed` (the global torch random state is not touched).
`simulate` and `frames` run in float32 and not under autocast.

TWO DELIBERATE DIFFERENCES from the CPU: the noise is `torch.randn` (the same distribution, other values than numpy's stream), and
the framing's resampling is torch's (`frames`: two resamplings in the CPU's order, each rounded; cv2 rounds the source
coordinates to 1 / 32 px and works in fixed point, torch does not).

SATURATION follows cv2's 8-bit HSV round trip, which `apply_color` runs (`core/query_sim/augment/color.py`: cv2.cvtColor RGB2HSV and
HSV2RGB, not the _FULL pair, so H is 0..179; the middle: S x saturation clipped to 255, H and V as they are, cut to a whole number as
`.astype(np.uint8)`; installed OpenCV 4.10.0.84). `rgb2hsv` and `hsv2rgb` are the arithmetic of OpenCV 4.x's
`modules/imgproc/src/color_hsv.simd.hpp` as the main conversation read it from the 4.x branch (not necessarily the 4.10.0 release):
the integer division tables, the integer RGB -> HSV, and a float32 HSV -> RGB, every step one float32 operation in cv2's order. Not
confirmed by reading: `saturate_cast<int>(double)`'s rounding, the max / min definition, which code is FULL, whether cv2's SIMD
path uses FMA. `tests/TestCoAE/test_sim_gpu.py` t_d_saturation compares them against `cv2.cvtColor` on every colour.
'''
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from locascope.training.CoAE.Sims import Geometry, Shot, frame_matrix

#: the reference reads framed at once by `frames` (32 reads as float: 32 x 3 x 1056^2 x 4 B = 0.43 GB)
FRAME_CHUNK: int = 32


def _u8(x: torch.Tensor) -> torch.Tensor:
    """What numpy's `np.clip(x, 0, 255).astype(np.uint8)` leaves: clipped, cut down to a whole number (still float32)."""
    return x.clamp_(0.0, 255.0).floor_()


def _col(shots: Sequence[Shot], name: str, like: torch.Tensor) -> torch.Tensor:
    """A parameter of every shot, float64 [n, 1, 1, 1] on the device (float64: the gain is formed as numpy forms it, then cast)."""
    return torch.tensor([getattr(s, name) for s in shots], dtype=torch.float64, device=like.device).view(-1, 1, 1, 1)


# ── saturation: cv2's 8-bit HSV ──────────────────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def hsv_tables() -> Tuple[np.ndarray, np.ndarray]:
    """cv2's 8-bit RGB -> HSV division tables (color_hsv.simd.hpp, hsv_shift 12), int32 [256] each, entry 0 = 0:
    sdiv[i] = round((255 << 12) / i), hdiv[i] = round((180 << 12) / (6 i)). No quotient is a half (255 x 2^13 / i and
    15 x 2^14 / i are odd integers for no i <= 255), so any round-to-nearest gives these tables."""
    i: np.ndarray = np.arange(1, 256, dtype=np.float64)
    sdiv: np.ndarray = np.zeros(256, np.int32)
    hdiv: np.ndarray = np.zeros(256, np.int32)
    sdiv[1:] = np.rint((255 << 12) / i)
    hdiv[1:] = np.rint((180 << 12) / (6.0 * i))
    return sdiv, hdiv


def rgb2hsv(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """cv2.cvtColor(uint8 RGB, COLOR_RGB2HSV): (h 0..179, s 0..255, v 0..255), int [n, H, W], in whole-number arithmetic.
    `x`: float32 [n, 3, H, W] of whole numbers 0..255."""
    sdiv_np, hdiv_np = hsv_tables()
    sdiv: torch.Tensor = torch.as_tensor(sdiv_np, device=x.device)
    hdiv: torch.Tensor = torch.as_tensor(hdiv_np, device=x.device)
    r: torch.Tensor = x[:, 0].to(torch.int32)
    g: torch.Tensor = x[:, 1].to(torch.int32)
    b: torch.Tensor = x[:, 2].to(torch.int32)
    v: torch.Tensor = torch.maximum(torch.maximum(r, g), b)
    diff: torch.Tensor = v - torch.minimum(torch.minimum(r, g), b)
    s: torch.Tensor = (diff * sdiv[v.long()] + (1 << 11)) >> 12
    h: torch.Tensor = torch.where(v == r, g - b, torch.where(v == g, b - r + 2 * diff, r - g + 4 * diff))
    h = (h * hdiv[diff.long()] + (1 << 11)) >> 12                       # >> of a signed int: arithmetic, as C's
    h = h + torch.where(h < 0, 180, 0)
    return h, s, v


#: HSV2RGB_native's sector table: (b, g, r) as indices into tab = (v, v (1 - s), v (1 - s h), v (1 - s (1 - h)))
_SECTOR: Tuple[Tuple[int, int, int], ...] = ((1, 3, 0), (1, 0, 2), (3, 0, 1), (0, 2, 1), (0, 1, 3), (2, 1, 0))


def hsv2rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """cv2.cvtColor(uint8 HSV, COLOR_HSV2RGB) of whole-number h (0..179), s, v (0..255) [n, H, W] -> float32 [n, 3, H, W] of
    whole numbers: every step one float32 operation, in cv2's order; the result x 255 cut toward zero (cv2's SIMD path; the
    few pixels of a row's scalar tail are rounded by cv2 instead: none at a width that is a multiple of 64)."""
    f32: torch.dtype = torch.float32
    one: torch.Tensor = torch.tensor(1.0, dtype=f32, device=h.device)
    k: torch.Tensor = one / 255.0                                       # the float32 constant 1.f / 255: s, v are MULTIPLIED by it
    hscale: torch.Tensor = torch.tensor(6.0, dtype=f32, device=h.device) / torch.tensor(180.0, dtype=f32, device=h.device)   # cv2's hscale = 6.f / hrange, a float32 division
    hf: torch.Tensor = h.to(f32) * hscale
    sf: torch.Tensor = s.to(f32) * k
    vf: torch.Tensor = v.to(f32) * k
    sector: torch.Tensor = torch.floor(hf)
    fr: torch.Tensor = hf - sector
    sec: torch.Tensor = sector.long() % 6                               # python's %: 0..5 for a negative too
    tab: torch.Tensor = torch.stack([vf, vf * (1 - sf), vf * (1 - sf * fr), vf * (1 - sf * (1 - fr))], dim=1)     # [n, 4, H, W]
    lut: torch.Tensor = torch.tensor(_SECTOR, device=h.device)          # [6, 3]: (b, g, r)
    idx: torch.Tensor = lut[sec]                                        # [n, H, W, 3]
    bgr: torch.Tensor = torch.gather(tab, 1, idx.permute(0, 3, 1, 2))   # [n, 3, H, W] = (b, g, r)
    bgr = torch.where((s == 0)[:, None], vf[:, None], bgr)              # s == 0: b = g = r = v
    return torch.trunc(bgr.flip(1) * 255.0).clamp_(0.0, 255.0)          # (r, g, b)


def saturate(x: torch.Tensor, sats: Sequence[float]) -> torch.Tensor:
    """`apply_color(img, brightness=0, contrast=1.0, saturation=)`: cv2's 8-bit RGB -> HSV, S x saturation in float32 (clipped
    to 255, cut down to a whole number as `.astype(np.uint8)` does), H and V as they are, cv2's 8-bit HSV -> RGB. The round
    trip is made at saturation 1 too, as the CPU makes it."""
    h, s, v = rgb2hsv(x)
    sat: torch.Tensor = torch.tensor(list(sats), dtype=torch.float32, device=x.device).view(-1, 1, 1)
    s2: torch.Tensor = torch.floor((s.to(torch.float32) * sat).clamp(0.0, 255.0)).to(torch.int32)
    return hsv2rgb(h, s2, v)


# ── the other sensor steps ───────────────────────────────────────────────────────────────────────────────────────────────

def brightness_contrast(x: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """`apply_brightness_contrast`: x (1 + b), then (x - mean) (1 + c) + mean about the mean of the whole canvas, all channels;
    b == c == 0 leaves the image (as the CPU returns it untouched)."""
    y: torch.Tensor = x * (1.0 + b).float()
    m: torch.Tensor = (y.sum(dim=(1, 2, 3), keepdim=True, dtype=torch.float64) / (y.shape[1] * y.shape[2] * y.shape[3])).float()
    y = _u8((y - m) * (1.0 + c).float() + m)
    return torch.where((b == 0.0) & (c == 0.0), x, y)


def colour_temp(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """`apply_color_temp`: R (1 + t), B (1 - t), G as it is."""
    g: torch.Tensor = torch.cat([1.0 + t, torch.ones_like(t), 1.0 - t], dim=1).float()      # [n, 3, 1, 1]
    return _u8(x * g)


@lru_cache(maxsize=8)
def disk(radius: int) -> np.ndarray:
    """`apply_defocus`'s kernel before its normalisation, by the same cv2 call (cv2's own rasterisation of the disk): uint8 0 / 1."""
    k: np.ndarray = np.zeros((2 * radius + 1, 2 * radius + 1), np.uint8)
    cv2.circle(k, (radius, radius), radius, 1, -1)
    return k


def defocus(x: torch.Tensor, radius: int) -> torch.Tensor:
    """`apply_defocus` (cv2.filter2D, its default border BORDER_REFLECT_101 = torch's 'reflect'): the disk's SUM first (whole numbers:
    exact in float32 whatever the convolution algorithm), then / its count, rounded as filter2D rounds a uint8 result."""
    k: np.ndarray = disk(radius)
    w: torch.Tensor = torch.as_tensor(k, dtype=torch.float32, device=x.device)[None, None].expand(3, 1, -1, -1).contiguous()
    s: torch.Tensor = F.conv2d(F.pad(x, (radius,) * 4, mode='reflect'), w, groups=3)
    return torch.round(s / float(k.sum())).clamp_(0.0, 255.0)


def chromatic(x: torch.Tensor, shift: int) -> torch.Tensor:
    """`apply_chromatic`: R moved `shift` px right, B left (whole px: no resampling), the edge mirrored WITH its border pixel
    (cv2.BORDER_REFLECT: index -k -> k - 1, W - 1 + k -> W - k)."""
    wd: int = x.shape[-1]
    out: torch.Tensor = x.clone()
    out[:, 0] = torch.cat([x[:, 0, :, :shift].flip(-1), x[:, 0, :, :wd - shift]], dim=-1)
    out[:, 2] = torch.cat([x[:, 2, :, shift:], x[:, 2, :, wd - shift:].flip(-1)], dim=-1)
    return out


_FALL: Dict[Tuple[int, int, str], torch.Tensor] = {}


def falloff(h: int, w: int, device: torch.device) -> torch.Tensor:
    """`augment.field._vignette_falloff` (its formula: the centre is (w / 2, h / 2), sigma 0.8 of the smaller half side), float64
    [h, w] on the device, kept a size and device."""
    key: Tuple[int, int, str] = (h, w, str(device))
    if key not in _FALL:
        xs: torch.Tensor = torch.arange(w, dtype=torch.float64, device=device) - w / 2.0
        ys: torch.Tensor = torch.arange(h, dtype=torch.float64, device=device) - h / 2.0
        sigma: float = min(w / 2.0, h / 2.0) * 0.8
        _FALL[key] = torch.exp(-(xs[None, :] ** 2 + ys[:, None] ** 2) / (2 * sigma ** 2))
    return _FALL[key]


def vignette(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """`apply_vignette`: x ((1 - v) + v falloff); v == 0 is the gain 1."""
    gain: torch.Tensor = ((1.0 - v) + v * falloff(x.shape[-2], x.shape[-1], x.device)).float()   # [n, 1, H, W]
    return _u8(x * gain)


@lru_cache(maxsize=64)
def area_matrix(n_src: int, n_dst: int) -> np.ndarray:
    """cv2.INTER_AREA along one axis when shrinking: float32 [n_dst, n_src], row d the share of each source pixel in
    [d r, (d + 1) r), r = n_src / n_dst (rows sum to 1)."""
    r: float = n_src / n_dst
    lo: np.ndarray = np.arange(n_dst, dtype=np.float64)[:, None] * r
    j: np.ndarray = np.arange(n_src, dtype=np.float64)[None, :]
    return (np.clip(np.minimum(lo + r, j + 1) - np.maximum(lo, j), 0.0, None) / r).astype(np.float32)


def resolution_loss(x: torch.Tensor, s: float) -> torch.Tensor:
    """The lab's resolution loss of one canvas [1, 3, H, W] (the CPU's: cv2.resize INTER_AREA down, INTER_CUBIC back; tests/TestCoAE/test_sim_gpu.py holds it): to (round(H / s), round(W / s)) by area average, back by bicubic
    (a = -0.75, half-pixel centres, the edge repeated: cv2.INTER_CUBIC = F.interpolate 'bicubic', align_corners False): the
    area average rounded half up (cv2's whole-number path at a ratio of exactly 2 is (sum + 2) >> 2; at any other ratio a half
    does not occur), the bicubic result rounded. s <= 1: as it is."""
    if s <= 1.0:
        return x
    h: int = x.shape[-2]
    w: int = x.shape[-1]
    a_h: torch.Tensor = torch.as_tensor(area_matrix(h, max(1, round(h / s))), device=x.device)
    a_w: torch.Tensor = torch.as_tensor(area_matrix(w, max(1, round(w / s))), device=x.device)
    small: torch.Tensor = torch.floor(torch.einsum('ih,nchw,jw->ncij', a_h, x, a_w) + 0.5).clamp_(0.0, 255.0)
    return torch.round(F.interpolate(small, size=(h, w), mode='bicubic', align_corners=False)).clamp_(0.0, 255.0)


@lru_cache(maxsize=8)
def gauss(sigma: float) -> np.ndarray:
    """cv2.GaussianBlur(float image, (0, 0), sigma)'s 1-D kernel: the size cv2 takes for a float image, round(8 sigma + 1) | 1,
    and cv2's own coefficients (cv2.getGaussianKernel)."""
    size: int = int(round(sigma * 8 + 1)) | 1
    return cv2.getGaussianKernel(size, sigma, cv2.CV_32F).ravel()


def blur(z: torch.Tensor, sigma: float) -> torch.Tensor:
    """cv2.GaussianBlur of a float [3, H, W] (separable, BORDER_REFLECT_101), tap by tap in a fixed order (no cuDNN: the same
    bits every run)."""
    k: np.ndarray = gauss(sigma)
    half: int = len(k) // 2
    h: int = z.shape[-2]
    w: int = z.shape[-1]
    p: torch.Tensor = F.pad(z[None], (half,) * 4, mode='reflect')[0]
    rows: torch.Tensor = sum(float(k[t]) * p[:, :, t:t + w] for t in range(len(k)))          # [3, H + 2 half, W]
    return sum(float(k[t]) * rows[:, t:t + h, :] for t in range(len(k)))


def add_noise(x: torch.Tensor, shots: Sequence[Shot], gen: torch.Generator) -> torch.Tensor:
    """`apply_noise` / the lab's correlated noise: each canvas its own stream (`gen` seeded with the shot's `noise_seed`),
    unit normal [3, H, W]; with `noise_corr` blurred and brought back to unit sd (the canvas's own, population sd); times
    `noise_sigma`, added, `_u8`. Another stream than numpy's: the same distribution, other values."""
    for i, s in enumerate(shots):
        if s.noise_sigma <= 0:
            continue
        gen.manual_seed(s.noise_seed)
        z: torch.Tensor = torch.randn(x.shape[1:], generator=gen, device=x.device, dtype=torch.float32)
        if s.noise_corr > 0:
            z = blur(z, s.noise_corr)
            z = z / (z.std(unbiased=False) + 1e-9)
        x[i] = _u8(x[i] + s.noise_sigma * z)
    return x


def _each(x: torch.Tensor, values: Sequence[int], fn: Callable[[torch.Tensor, int], torch.Tensor]) -> torch.Tensor:
    """`fn(rows, v)` on the rows of each distinct value v > 0 (a few a batch: a lab's constant)."""
    for v in sorted({v for v in values if v > 0}):
        idx: torch.Tensor = torch.tensor([i for i, u in enumerate(values) if u == v], device=x.device)
        x[idx] = fn(x[idx], v)
    return x


_GEN: Dict[str, torch.Generator] = {}


def simulate(base: torch.Tensor, shots: Sequence[Shot], noise: bool = True) -> torch.Tensor:
    """The simulated canvases: `base` uint8 [n, H, W, 3] on any device, a shot each -> uint8 [n, H, W, 3] there. `noise` False
    leaves the noise out (the tests' comparison against the CPU). No gradient; not under autocast."""
    if len(shots) != base.shape[0]:
        raise ValueError(f'{len(shots)} shots for {base.shape[0]} canvases')
    with torch.no_grad():
        x: torch.Tensor = base.permute(0, 3, 1, 2).float()                 # [n, 3, H, W], whole numbers
        x = saturate(x, [s.saturation for s in shots])
        x = brightness_contrast(x, _col(shots, 'brightness', x), _col(shots, 'contrast', x))
        x = colour_temp(x, _col(shots, 'color_temp', x))
        x = _each(x, [s.defocus for s in shots], defocus)
        x = _each(x, [s.chromatic for s in shots], chromatic)
        x = vignette(x, _col(shots, 'vignette', x))
        for i, s in enumerate(shots):
            if s.res_loss > 1.0:
                x[i:i + 1] = resolution_loss(x[i:i + 1], s.res_loss)
        if noise:
            key: str = str(base.device)
            if key not in _GEN:
                _GEN[key] = torch.Generator(device=base.device)
            x = add_noise(x, shots, _GEN[key])
        return x.permute(0, 2, 3, 1).to(torch.uint8).contiguous()


# ── the framing ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def frames(refs: torch.Tensor, rows: Sequence[int], geos: Sequence[Geometry], photo_px: int) -> torch.Tensor:
    """The photo canvases of geometries: uint8 [m, photo_px, photo_px, 3], canvas i = `Sims.frame(refs[rows[i]], geos[i],
    photo_px)` made on the device. `refs`: uint8 [n, L, L, 3], the reference reads. The identity is the centre crop (the
    scan's own pixels, no resampling: bit for bit the CPU's). Any other: the framing by bicubic resampling (a = -0.75, as
    cv2.INTER_CUBIC; the edge mirrored with its border pixel, as cv2.BORDER_REFLECT), rounded; then, when k1 or k2 is not 0, the
    lens distortion by bilinear resampling at `apply_distortion`'s coordinates (clipped to the canvas), rounded. Two resamplings
    in the CPU's order. NOT bit for bit cv2's (cv2 rounds the source coordinates to 1 / 32 px and interpolates in fixed point)."""
    big: int = refs.shape[1]
    dev: torch.device = refs.device
    out: torch.Tensor = torch.empty((len(rows), photo_px, photo_px, 3), dtype=torch.uint8, device=dev)
    a: int = big // 2 - photo_px // 2
    ident: List[int] = [i for i, g in enumerate(geos) if g == Geometry()]
    moved: List[int] = [i for i, g in enumerate(geos) if g != Geometry()]
    if ident:
        out[torch.tensor(ident, device=dev)] = refs[torch.tensor([rows[i] for i in ident], device=dev), a:a + photo_px, a:a + photo_px]
    for c in range(0, len(moved), FRAME_CHUNK):
        part: List[int] = moved[c:c + FRAME_CHUNK]
        src: torch.Tensor = refs[torch.tensor([rows[i] for i in part], device=dev)].permute(0, 3, 1, 2).float()      # [m, 3, L, L]
        mats: torch.Tensor = torch.tensor(np.stack([frame_matrix(geos[i], big, photo_px) for i in part]), dtype=torch.float64, device=dev)   # [m, 2, 3]
        p: torch.Tensor = torch.arange(photo_px, dtype=torch.float64, device=dev)
        ones: torch.Tensor = torch.ones((photo_px, photo_px), dtype=torch.float64, device=dev)
        dst: torch.Tensor = torch.stack([p[None, :] * ones, p[:, None] * ones, ones], dim=-1)                          # [H, W, 3]: (x, y, 1)
        sxy: torch.Tensor = torch.einsum('mij,hwj->mhwi', mats, dst)                                                   # source index (x, y)
        grid: torch.Tensor = ((2.0 * sxy + 1.0) / big - 1.0).float()                                                   # align_corners False
        img: torch.Tensor = torch.round(F.grid_sample(src, grid, mode='bicubic', padding_mode='reflection', align_corners=False)).clamp_(0.0, 255.0)
        k: torch.Tensor = torch.tensor([[geos[i].k1, geos[i].k2] for i in part], dtype=torch.float32, device=dev)      # [m, 2]
        lens: torch.Tensor = (k != 0).any(dim=1)
        if bool(lens.any()):
            cx: float = (photo_px - 1) / 2.0                                                                           # = nx: the canvas is square, sensor None
            q: torch.Tensor = torch.arange(photo_px, dtype=torch.float32, device=dev)
            xn: torch.Tensor = ((q - cx) / (cx + 1e-6))[None, None, :]
            yn: torch.Tensor = ((q - cx) / (cx + 1e-6))[None, :, None]
            r2: torch.Tensor = xn ** 2 + yn ** 2                                                                       # [1, H, W]
            fac: torch.Tensor = 1.0 + k[lens, 0, None, None] * r2 + k[lens, 1, None, None] * r2 ** 2                   # [l, H, W]
            fac = torch.where(fac.abs() < 1e-6, torch.full_like(fac, 1e-6), fac)
            lx: torch.Tensor = (xn / fac * cx + cx).clamp(0.0, photo_px - 1.0)
            ly: torch.Tensor = (yn / fac * cx + cx).clamp(0.0, photo_px - 1.0)
            g2: torch.Tensor = torch.stack([(2.0 * lx + 1.0) / photo_px - 1.0, (2.0 * ly + 1.0) / photo_px - 1.0], dim=-1)
            img[lens] = torch.round(F.grid_sample(img[lens], g2, mode='bilinear', padding_mode='border', align_corners=False)).clamp_(0.0, 255.0)
        out[torch.tensor(part, device=dev)] = img.permute(0, 2, 3, 1).to(torch.uint8)
    return out
