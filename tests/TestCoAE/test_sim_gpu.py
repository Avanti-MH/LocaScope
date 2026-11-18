#!/usr/bin/env python3
"""The GPU simulation of CoAE's views (`locascope/training/CoAE/GpuSims.py`) against the CPU's (`core.query_sim`), step by step and as a
whole: the draws (`Sims.draw_shot` consumes `random.Random` as `Sims.photometric` does), every sensor step, cv2's 8-bit HSV round trip
(every colour), the resolution loss, the end to end result, the noise's statistics, determinism, the Co-figure's targets (printed), the
framing (`GpuSims.frames` against `Sims.frame`) and the timings (printed).

    python tests/TestCoAE/test_sim_gpu.py

    COAE_SIM_OUT=<dir>   also write every printed number line to <dir>/sim_gpu.txt
    COAE_SIM_REAL=0      no real scan canvases: the synthetic ones alone (the default reads the training slides' scans)

A GPU is used when there is one; the real canvases are read only then. Only training slides' scans are read: no photo. Every threshold
is a limit set before anything was measured (the printed numbers are what to read): a failed one goes back, it is not loosened here.
The CPU's reference is `core.query_sim.pipeline.simulate_with_gt` (`jpeg` False in the labs: no codec), the `augment` functions
and cv2 itself.
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
import traceback
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2                                                                           # noqa: E402
import numpy as np                                                                   # noqa: E402
import torch                                                                         # noqa: E402

import locascope.training.CoAE.GpuSims as G                                          # noqa: E402
from locascope.core.query_sim.augment.color import apply_brightness_contrast, apply_color, apply_color_temp, apply_jpeg   # noqa: E402
from locascope.core.query_sim.augment.field import apply_vignette                    # noqa: E402
from locascope.core.query_sim.augment.lens import apply_chromatic, apply_defocus     # noqa: E402
from locascope.core.query_sim.augment.noise import apply_noise                       # noqa: E402
from locascope.core.query_sim.pipeline import simulate_with_gt                       # noqa: E402
from locascope.training.CoAE.CoFigure import field_maps, layout, part_names          # noqa: E402
from locascope.training.CoAE.Config import LEVELS, CoaeConfig, ProbeConfig           # noqa: E402
from locascope.training.CoAE.train import SIM_CHUNK                                  # noqa: E402  (the chunk of canvases a step simulates at once: t_j_timing)
from locascope.training.CoAE.Sims import (GREY_WEIGHTS, Geometry, LabGap, Shot, centre, draw_geometry, draw_shot, frame, probe_labs,   # noqa: E402
                                          read_px, sim_labs)

DEV: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CFG: CoaeConfig = CoaeConfig()
PHOTO: int = CFG.data.photo_px                              # 640
FIELD_PX: int = 224                                         # the FM's field of a 256 px tile (gigapath): a multiple of out_cells
_RESULTS: List[Tuple[str, Optional[BaseException]]] = []
_LINES: List[str] = []


def say(line: str) -> None:
    """Print a line and keep it for `sim_gpu.txt`."""
    print(line, flush=True)
    _LINES.append(line)


def check(name: str, fn: Callable[[], str]) -> None:
    try:
        msg: str = fn()
        _RESULTS.append((name, None))
        print(f'PASS  {name}: {msg}', flush=True)
    except BaseException as exc:                                                     # noqa: BLE001
        _RESULTS.append((name, exc))
        print(f'FAIL  {name}: {type(exc).__name__}: {exc}', flush=True)
        traceback.print_exc()


# ── inputs ───────────────────────────────────────────────────────────────────────────────────────────────────────────────

def texture(side: int, seed: int = 0) -> np.ndarray:
    """A smooth random grey texture float64 [side, side] in 25..225 with structure at many scales (as test_coae.texture)."""
    rng: np.random.Generator = np.random.default_rng(seed)
    f: np.ndarray = np.fft.fftfreq(side)
    r: np.ndarray = np.hypot(f[:, None], f[None, :]) + 1.0 / side
    spec: np.ndarray = (rng.normal(size=(side, side)) + 1j * rng.normal(size=(side, side))) / r ** 1.2
    g: np.ndarray = np.real(np.fft.ifft2(spec))
    return (g - g.min()) / (g.max() - g.min()) * 200 + 25


def colour_canvas(side: int, seed: int) -> np.ndarray:
    """A colour canvas uint8 [side, side, 3]: a texture a channel (its own seed), 40 % of a common brightness mixed in (S in the middle)."""
    common: np.ndarray = texture(side, 1000 + seed)
    chans: List[np.ndarray] = [0.6 * texture(side, 10 * seed + k) + 0.4 * common for k in range(3)]
    return np.clip(np.stack(chans, -1), 0, 255).astype(np.uint8)


def synthetic_canvases() -> np.ndarray:
    """8 colour canvases uint8 [8, 640, 640, 3]: two scaled by 1.3 (part of their pixels saturate at 255), one dark (has 0)."""
    out: List[np.ndarray] = []
    for i in range(8):
        img: np.ndarray = colour_canvas(PHOTO, i).astype(np.float64)
        if i in (2, 3):
            img = img * 1.3
        if i == 4:
            img = img - 90.0
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return np.stack(out)


def stripes_and_checker(side: int) -> List[np.ndarray]:
    """The framing's worst cases: 1 px wide black and white stripes, and a 1 px checkerboard (uint8 [side, side, 3])."""
    col: np.ndarray = (np.arange(side)[None, :] % 2) * 255 + np.zeros((side, 1), dtype=np.int64)
    chk: np.ndarray = ((np.arange(side)[:, None] + np.arange(side)[None, :]) % 2) * 255
    return [np.repeat(m[..., None], 3, -1).astype(np.uint8) for m in (col, chk)]


class Inputs:
    """What the tests run on: `canvases` uint8 [n, 640, 640, 3] (the synthetic ones, then the real scans' when there are any), `reads`
    the reference squares the real canvases were framed from (uint8 [read, read, 3]; the synthetic ones when there are none), `worst`
    the reads that are worst cases (stripes, checkerboard: their rows are only printed), `real` whether the real scans are there."""

    def __init__(self, canvases: np.ndarray, reads: List[np.ndarray], worst: List[bool], real: bool) -> None:
        self.canvases: np.ndarray = canvases
        self.reads: List[np.ndarray] = reads
        self.worst: List[bool] = worst
        self.real: bool = real


_INPUTS: Optional[Inputs] = None


def real_reads() -> List[np.ndarray]:
    """The first 8 readable, non-blank places of each rung of 4 training slides (`Supply._read`): reference squares of the scans."""
    from locascope.training.CoAE.Supply import Supply
    cfg: CoaeConfig = replace(CFG, data=replace(CFG.data, places=8, mask='uni2_pca', draw_cache='TestCoae', slides=4))
    sup: Supply = Supply(cfg, 0, 1, DEV, pool=False)
    try:
        reads: List[np.ndarray] = []
        for ds in sorted({p.ds for p in sup.places}):
            got: int = 0
            for p in sup.places:
                if p.ds != ds or got == 8:
                    continue
                img: Optional[np.ndarray] = sup._read(p)
                if img is not None and not sup._blank(centre(img, sup.tile)):
                    reads.append(img)
                    got += 1
        return reads
    finally:
        sup.close()


def inputs() -> Inputs:
    global _INPUTS
    if _INPUTS is not None:
        return _INPUTS
    syn: np.ndarray = synthetic_canvases()
    big: int = read_px(CFG.data, PHOTO)
    reads: List[np.ndarray] = []
    note: str = 'REAL: none (no GPU or COAE_SIM_REAL=0): the synthetic canvases alone'
    if torch.cuda.is_available() and os.environ.get('COAE_SIM_REAL', '1') != '0':
        reads = real_reads()
    if reads:
        reals: List[np.ndarray] = []
        for i, ref in enumerate(reads):
            reals.append(frame(ref, Geometry(), PHOTO))
            reals.append(frame(ref, draw_geometry(random.Random(i), LEVELS['large']), PHOTO))
        canv: np.ndarray = np.concatenate([syn, np.stack(reals)[:96]])
        _INPUTS = Inputs(canv, reads[:48] + stripes_and_checker(big), [False] * len(reads[:48]) + [True] * 2, True)
        say(f'REAL: {len(reads)} reference reads of {reads[0].shape[0]} px -> {len(reals[:96])} real canvases (scans of training slides), 8 synthetic')
    else:
        say(note)
        worst: List[np.ndarray] = stripes_and_checker(big)
        rd: List[np.ndarray] = [colour_canvas(big, 20 + i) for i in range(4)] + worst
        _INPUTS = Inputs(syn, rd, [False] * 4 + [True] * len(worst), False)
    return _INPUTS


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

def to_dev(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEV)


def gpu_step(imgs: np.ndarray, fn: Callable[[torch.Tensor], torch.Tensor], chunk: int = 16) -> np.ndarray:
    """`fn` on float32 [n, 3, H, W] of whole numbers, `chunk` canvases at a time: uint8 [N, H, W, 3] of what it returns."""
    out: List[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, len(imgs), chunk):
            x: torch.Tensor = to_dev(imgs[s:s + chunk]).permute(0, 3, 1, 2).float()
            out.append(fn(x).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy())
    return np.concatenate(out)


def cpu_step(imgs: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    return np.stack([fn(img) for img in imgs])


def col(values: List[float], n: int) -> torch.Tensor:
    """A parameter of `n` canvases (the same value) as GpuSims' columns: float64 [n, 1, 1, 1]."""
    return torch.tensor(values * n if len(values) == 1 else values, dtype=torch.float64, device=DEV).view(-1, 1, 1, 1)


def diff_stats(g: np.ndarray, c: np.ndarray) -> Dict[str, float]:
    """d = GPU - CPU in grey levels over two uint8 stacks: the share of identical values, mean |d|, bias (mean d), p99.9 |d|, max |d|."""
    hist: np.ndarray = np.zeros(256, dtype=np.int64)
    total_d: int = 0
    n: int = 0
    for s in range(0, len(g), 8):
        d: np.ndarray = g[s:s + 8].astype(np.int16) - c[s:s + 8].astype(np.int16)
        hist += np.bincount(np.abs(d).ravel().astype(np.intp), minlength=256)
        total_d += int(d.sum(dtype=np.int64))
        n += d.size
    cum: np.ndarray = np.cumsum(hist)
    return {'identical': float(hist[0]) / n, 'mean': float((hist * np.arange(256)).sum()) / n, 'bias': total_d / n,
            'p999': float(np.searchsorted(cum, 0.999 * n)), 'max': float(np.nonzero(hist)[0].max())}


def report(name: str, params: str, st: Dict[str, float]) -> None:
    say(f'{name} {params}: identical {st["identical"]:.5f} mean|d| {st["mean"]:.4f} bias {st["bias"]:+.4f} p99.9|d| {st["p999"]:.0f} max|d| {st["max"]:.0f}')


def expect(bad: List[str], ok: bool, what: str) -> None:
    if not ok:
        bad.append(what)


def cv_res_loss(img: np.ndarray, s: float) -> np.ndarray:
    """The CPU's resolution loss (Sims.resolution_loss as it was written): INTER_AREA down to 1 / s, INTER_CUBIC back."""
    if s <= 1.0:
        return img
    h, w = img.shape[:2]
    small: np.ndarray = cv2.resize(img, (max(1, round(w / s)), max(1, round(h / s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def grey_of(canv: np.ndarray) -> torch.Tensor:
    """uint8 [n, S, S, 3] -> float32 grey [n, S, S] on DEV (the targets' definition: `float() @ GREY_WEIGHTS`)."""
    w: torch.Tensor = torch.tensor(GREY_WEIGHTS, dtype=torch.float32, device=DEV)
    return to_dev(canv).float() @ w


def part_sd_rms(a: torch.Tensor, b: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    """RMS of a - b per Co-figure part (`part_names`), divided by that part's sd over `ref` (field maps [n, N, t, t])."""
    out: Dict[str, float] = {}
    for name, part in zip(part_names(CFG.cofig), layout(CFG.cofig)):
        sl: slice = slice(part.start, part.start + part.size)
        out[name] = float((a[:, sl] - b[:, sl]).pow(2).mean().sqrt() / ref[:, sl].std().clamp(min=1e-12))
    return out


def neutral_shot(**kw: Any) -> Shot:
    """A shot that changes nothing (saturation 1, the rest 0, no resolution loss), with `kw` set."""
    base: Dict[str, Any] = dict(saturation=1.0, brightness=0.0, contrast=0.0, color_temp=0.0, defocus=0, chromatic=0, vignette=0.0,
                                res_loss=1.0, noise_sigma=0.0, noise_corr=0.0, noise_seed=0)
    base.update(kw)
    return Shot(**base)


# ── draws, identity, steps ───────────────────────────────────────────────────────────────────────────────────────────────

def t_a_draws() -> str:
    labs: Dict[str, LabGap] = probe_labs(ProbeConfig().ood)
    img: np.ndarray = synthetic_canvases()[0]
    bad: List[str] = []
    for name, lab in labs.items():
        for seed in range(3):
            r1: random.Random = random.Random(seed)
            shot: Shot = draw_shot(lab, r1)
            r2: random.Random = random.Random(seed)
            _, p = simulate_with_gt(img, cfg=lab.base, rng=r2, output_wh=(PHOTO, PHOTO))
            res: float = 1.0
            sigma: float = float(p['noise_sigma'])
            nseed: int = int(p['noise_seed'])
            if lab.extra:
                res = r2.uniform(*lab.blur_factor)
                sigma = float(lab.base.noise_sigma)
                nseed = r2.getrandbits(32) if sigma > 0 else 0
            want: Shot = Shot(float(p['saturation']), float(p['brightness']), float(p['contrast']), float(p['color_temp']), int(p['defocus_radius']),
                              int(p['chromatic_shift']), float(p['vignette_strength']), res, sigma, float(lab.noise_corr) if lab.extra else 0.0, nseed)
            expect(bad, shot == want, f'{name} seed {seed}: {shot} != {want}')
            expect(bad, r1.getstate() == r2.getstate(), f'{name} seed {seed}: the rng states differ')
    assert not bad, '; '.join(bad)
    return f'{len(labs)} labs x 3 seeds: every field is the CPU draw, the rng ends in the same state'


def t_b_identity() -> str:
    """A neutral shot leaves every step but the saturation's HSV round trip as it is (the CPU makes that round trip at saturation 1
    too: cv2's 8-bit HSV is not the identity), so the output is the saturation step alone, bit for bit."""
    canv: np.ndarray = inputs().canvases[:16]
    x: torch.Tensor = to_dev(canv)
    with torch.no_grad():
        out: torch.Tensor = G.simulate(x, [neutral_shot()] * len(canv))
        rt: torch.Tensor = G.saturate(x.permute(0, 3, 1, 2).float(), [1.0] * len(canv)).permute(0, 2, 3, 1).to(torch.uint8)
    assert torch.equal(out, rt), 'a neutral shot changed more than the HSV round trip'
    same: float = float((out == x).float().mean())
    say(f'b_identity: a neutral shot is the saturation 1 round trip; it leaves {same:.5f} of the values as the input has them')
    return f'the neutral shot is the saturation 1 round trip alone (bit for bit); {same:.5f} of the values equal the input'


def t_c_steps() -> str:
    canv: np.ndarray = inputs().canvases
    n: int = len(canv)
    bad: List[str] = []

    def run(name: str, params: str, gpu: np.ndarray, cpu: np.ndarray) -> None:
        st: Dict[str, float] = diff_stats(gpu, cpu)
        report(name, params, st)
        expect(bad, st['max'] <= 1 and st['identical'] >= 0.999, f'{name} {params}')

    for t in (0.03, -0.03, 0.12, -0.12, 0.25, -0.25):
        run('colour_temp', f't={t}', gpu_step(canv, lambda x, t=t: G.colour_temp(x, col([t], x.shape[0]))), cpu_step(canv, lambda i, t=t: apply_color_temp(i, t)))
    for s in (1, 2, 4, 5):
        run('chromatic', f'shift={s}', gpu_step(canv, lambda x, s=s: G.chromatic(x, s)), cpu_step(canv, lambda i, s=s: apply_chromatic(i, s)))
    for r in (1, 2, 3):
        run('defocus', f'radius={r}', gpu_step(canv, lambda x, r=r: G.defocus(x, r)), cpu_step(canv, lambda i, r=r: apply_defocus(i, r)))
    for b, c in ((0.08, 0.08), (-0.08, -0.08), (0.08, -0.08), (-0.08, 0.08), (-0.38, 0.55), (-0.22, 0.25)):
        run('brightness_contrast', f'b={b} c={c}', gpu_step(canv, lambda x, b=b, c=c: G.brightness_contrast(x, col([b], x.shape[0]), col([c], x.shape[0]))),
            cpu_step(canv, lambda i, b=b, c=c: apply_brightness_contrast(i, b, c)))
    for v in (0.05, 0.15, 0.45):
        run('vignette', f'v={v}', gpu_step(canv, lambda x, v=v: G.vignette(x, col([v], x.shape[0]))), cpu_step(canv, lambda i, v=v: apply_vignette(i, v)))
    assert not bad, '; '.join(bad)
    return f'{n} canvases: colour temperature, chromatic shift, defocus, brightness / contrast, vignette each within 1 grey level, >= 0.999 identical'


# ── saturation: cv2's 8-bit HSV ──────────────────────────────────────────────────────────────────────────────────────────

def all_rgb() -> np.ndarray:
    """Every colour: uint8 [4096, 4096, 3] holding the 256^3 (r, g, b)."""
    idx: np.ndarray = np.arange(1 << 24, dtype=np.int64)
    return np.stack([idx >> 16, (idx >> 8) & 255, idx & 255], -1).astype(np.uint8).reshape(4096, 4096, 3)


def all_hsv(width: int) -> np.ndarray:
    """Every (h 0..179, s, v) in a row-major image of `width` columns (the first rows x width colours): uint8 [rows, width, 3]."""
    idx: np.ndarray = np.arange(180 * 65536, dtype=np.int64)
    flat: np.ndarray = np.stack([idx // 65536, (idx >> 8) & 255, idx & 255], -1).astype(np.uint8)
    rows: int = len(flat) // width
    return flat[:rows * width].reshape(rows, width, 3)


def hsv2rgb_gpu(hsv: np.ndarray, rows: int = 720) -> np.ndarray:
    """`G.hsv2rgb` of a uint8 HSV image [H, W, 3]: uint8 [H, W, 3]."""
    out: List[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, hsv.shape[0], rows):
            t: torch.Tensor = to_dev(hsv[s:s + rows]).to(torch.int32)
            out.append(G.hsv2rgb(t[None, ..., 0], t[None, ..., 1], t[None, ..., 2])[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy())
    return np.concatenate(out)


def hsv2rgb_fused(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor, frac: bool, tab: bool) -> torch.Tensor:
    """`G.hsv2rgb` with the two multiply-subtracts that a compiler may fuse into an FMA done fused (float64, where the product of two
    float32 is exact, then to float32): `frac` the fractional part h x hscale - sector, `tab` the table's 1 - s x h and 1 - s x (1 - h).
    A diagnostic only: which one reproduces cv2's wheel."""
    f32: torch.dtype = torch.float32
    k: torch.Tensor = torch.tensor(1.0, dtype=f32, device=h.device) / 255.0
    hscale: torch.Tensor = torch.tensor(6.0, dtype=f32, device=h.device) / torch.tensor(180.0, dtype=f32, device=h.device)
    hf64: torch.Tensor = h.to(f32).double() * hscale.double()
    hf: torch.Tensor = hf64.float()
    sf: torch.Tensor = s.to(f32) * k
    vf: torch.Tensor = v.to(f32) * k
    sector: torch.Tensor = torch.floor(hf)
    fr: torch.Tensor = (hf64 - sector.double()).float() if frac else hf - sector
    t2: torch.Tensor = (1.0 - sf.double() * fr.double()).float() if tab else 1 - sf * fr
    t3: torch.Tensor = (1.0 - sf.double() * (1 - fr).double()).float() if tab else 1 - sf * (1 - fr)
    table: torch.Tensor = torch.stack([vf, vf * (1 - sf), vf * t2, vf * t3], dim=1)
    lut: torch.Tensor = torch.tensor(G._SECTOR, device=h.device)
    idx: torch.Tensor = lut[sector.long() % 6]
    bgr: torch.Tensor = torch.gather(table, 1, idx.permute(0, 3, 1, 2))
    bgr = torch.where((s == 0)[:, None], vf[:, None], bgr)
    return torch.trunc(bgr.flip(1) * 255.0).clamp_(0.0, 255.0)


def hsv2rgb_fused_np(hsv: np.ndarray, frac: bool, tab: bool, rows: int = 720) -> np.ndarray:
    """`hsv2rgb_fused` of a uint8 HSV image [H, W, 3], as `hsv2rgb_gpu` runs `G.hsv2rgb`: uint8 [H, W, 3]."""
    out: List[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, hsv.shape[0], rows):
            t: torch.Tensor = to_dev(hsv[s:s + rows]).to(torch.int32)
            out.append(hsv2rgb_fused(t[None, ..., 0], t[None, ..., 1], t[None, ..., 2], frac, tab)[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy())
    return np.concatenate(out)


def t_d_saturation() -> str:
    bad: List[str] = []
    # (1) RGB -> HSV, every colour: bit for bit
    img: np.ndarray = all_rgb()
    want: np.ndarray = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    n_bad: int = 0
    shown: int = 0
    with torch.no_grad():
        for s in range(0, 4096, 512):
            x: torch.Tensor = to_dev(img[s:s + 512][None]).permute(0, 3, 1, 2).float()
            h, sat, v = G.rgb2hsv(x)
            got: np.ndarray = torch.stack([h.to(torch.int32), sat, v], -1)[0].to(torch.uint8).cpu().numpy()
            diff: np.ndarray = (got != want[s:s + 512]).any(-1)
            n_bad += int(diff.sum())
            for y, xx in zip(*np.nonzero(diff)):
                if shown < 10:
                    say(f'd_saturation (1) differs at rgb {tuple(int(c) for c in img[s + y, xx])}: gpu hsv {tuple(int(c) for c in got[y, xx])}, cv2 {tuple(int(c) for c in want[s + y, xx])}')
                    shown += 1
    say(f'd_saturation (1) rgb2hsv, all 2^24 colours: {n_bad} differ')
    expect(bad, n_bad == 0, f'(1) rgb2hsv differs from cv2 on {n_bad} colours')
    del img, want
    # (2) HSV -> RGB, every (h, s, v), width 4096 (a multiple of 64: no scalar tail)
    hsv: np.ndarray = all_hsv(4096)
    cv: np.ndarray = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    gp: np.ndarray = hsv2rgb_gpu(hsv)
    st: Dict[str, float] = diff_stats(gp[None], cv[None])
    report('d_saturation (2) hsv2rgb', 'all 180 x 256 x 256, width 4096', st)
    expect(bad, st['max'] <= 1 and st['identical'] >= 0.999, '(2) hsv2rgb')
    if st['identical'] < 1.0:
        wrong: np.ndarray = (gp != cv).any(-1)
        sector: np.ndarray = hsv[..., 0] // 30
        say('d_saturation (2) differing pixels by sector: ' + ', '.join(f'{k}: {int(wrong[sector == k].sum())}' for k in range(6))
            + f'; with s == 0: {int(wrong[hsv[..., 1] == 0].sum())}; of {int(wrong.sum())}')
        fused: List[float] = []
        for frac, tab in ((True, False), (False, True), (True, True)):
            fp: np.ndarray = hsv2rgb_fused_np(hsv, frac, tab)
            fused.append(diff_stats(fp[None], cv[None])['identical'])
        say(f'd_saturation (2) fused frac / tab / both: identical {fused[0]:.5f} / {fused[1]:.5f} / {fused[2]:.5f}')
    # (3) the scalar tail (print only)
    for width in (4097, 4127):
        hs: np.ndarray = all_hsv(width)
        cw: np.ndarray = cv2.cvtColor(hs, cv2.COLOR_HSV2RGB)
        gw: np.ndarray = hsv2rgb_gpu(hs)
        same: np.ndarray = (gw == cw).all(-1)
        say(f'd_saturation (3) width {width}: identical {same.mean():.6f}; the last 8 columns {same[:, -8:].mean():.6f}, the last 1 {same[:, -1:].mean():.6f}, the rest {same[:, :-8].mean():.6f}')
    # (4) the whole step against apply_color
    canv: np.ndarray = inputs().canvases
    for s in (1.0, 0.78, 1.2):
        g4: np.ndarray = gpu_step(canv, lambda x, s=s: G.saturate(x, [s] * x.shape[0]))
        c4: np.ndarray = cpu_step(canv, lambda i, s=s: apply_color(i, brightness=0, contrast=1.0, saturation=s))
        st4: Dict[str, float] = diff_stats(g4, c4)
        report('d_saturation (4) saturate', f's={s}', st4)
        expect(bad, st4['max'] <= 1 and st4['identical'] >= 0.999, f'(4) saturate s={s}')
    assert not bad, '; '.join(bad)
    return 'rgb2hsv is cv2 on every colour; hsv2rgb within 1 level on every (h, s, v); saturate within 1 level of apply_color'


def t_e_res_loss() -> str:
    canv: np.ndarray = inputs().canvases[:16]
    bad: List[str] = []
    for s in (1.5, 2.0, 2.5, 1.93):
        g: np.ndarray = gpu_step(canv, lambda x, s=s: torch.cat([G.resolution_loss(x[i:i + 1], s) for i in range(x.shape[0])]), chunk=4)
        c: np.ndarray = cpu_step(canv, lambda i, s=s: cv_res_loss(i, s))
        st: Dict[str, float] = diff_stats(g, c)
        report('res_loss', f's={s}', st)
        expect(bad, st['mean'] <= 0.5 and st['max'] <= 3, f's={s}')
    assert not bad, '; '.join(bad)
    return 'area down / bicubic up: mean |d| <= 0.5, max |d| <= 3 against two cv2.resize'


def t_f_end_to_end() -> str:
    canv: np.ndarray = inputs().canvases
    labs: Dict[str, LabGap] = probe_labs(ProbeConfig().ood)
    bad: List[str] = []
    for name, lab in labs.items():
        shots: List[Shot] = [draw_shot(lab, random.Random(i)) for i in range(len(canv))]
        g: np.ndarray = np.concatenate([G.simulate(to_dev(canv[s:s + 16]), shots[s:s + 16], noise=False).cpu().numpy() for s in range(0, len(canv), 16)])
        cpu: List[np.ndarray] = []
        for i in range(len(canv)):
            out, _ = simulate_with_gt(canv[i], cfg=replace(lab.base, noise_sigma=0.0), rng=random.Random(i), output_wh=(PHOTO, PHOTO))
            cpu.append(cv_res_loss(out, shots[i].res_loss) if lab.extra else out)
        st: Dict[str, float] = diff_stats(g, np.stack(cpu))
        report('end_to_end', f'lab {name} (noise off)', st)
        if name == 'ood-soft':
            expect(bad, st['mean'] <= 0.5 and st['max'] <= 3, f'{name}: mean {st["mean"]:.3f}, max {st["max"]:.0f}')
        else:
            gain: float = ((1 + max(lab.base.brightness_range)) * (1 + max(lab.base.contrast_range))
                           * (1 + max(lab.base.color_temp_range)))        # the largest gain a value can meet: bench-real 0.78 x 1.55 x 1.03 = 1.25, bench 1.08 x 1.08 x 1.12 = 1.31
            # a value the saturation step leaves 1 level off cv2 (d_saturation (4): about 5e-5 of them) goes through the gains: 1 x gain,
            # cut down to a whole number, is at most ceil(gain) levels; a defocus after it averages that one pixel over its disk (>= 5 px)
            limit: int = 1 if lab.base.defocus_radius > 0 else math.ceil(gain)
            say(f'end_to_end lab {name}: limit {limit}')
            assert limit in (1, 2), f'{name}: limit {limit} (gain {gain:.3f}): a lab range is wider than assumed'
            expect(bad, st['max'] <= limit and st['identical'] >= 0.9995, f'{name}: max {st["max"]:.0f} (limit {limit}), identical {st["identical"]:.5f}')
    assert not bad, '; '.join(bad)
    return f'{len(canv)} canvases x {len(labs)} labs, noise off: the whole chain against simulate_with_gt (ood-soft: + the two resizes)'


# ── noise ────────────────────────────────────────────────────────────────────────────────────────────────────────────────

SHIFTS: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (0, 4), (4, 0))


def noise_stats(out: np.ndarray) -> Tuple[float, float, List[float], List[float]]:
    """`out` uint8 [n, H, W, 3] of noise on 128: the mean and sd of out - 128, the autocorrelation at SHIFTS (same channel), the correlation
    between channels (R-G, R-B, G-B)."""
    a: np.ndarray = out.astype(np.float64) - 128.0
    mean: float = float(a.mean())
    a -= mean
    var: float = float((a * a).mean())
    acf: List[float] = []
    for dy, dx in SHIFTS:
        acf.append(float((a[:, :a.shape[1] - dy, :a.shape[2] - dx] * a[:, dy:, dx:]).mean()) / var)
    ch: List[float] = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        ch.append(float((a[..., i] * a[..., j]).mean()) / var)
    return mean, var ** 0.5, acf, ch


def cpu_noise(sigma: float, corr: float, seed: int) -> np.ndarray:
    """`apply_noise` on grey 128 (independent), or `Sims.photometric`'s correlated noise as it was written."""
    img: np.ndarray = np.full((PHOTO, PHOTO, 3), 128, dtype=np.uint8)
    if corr <= 0:
        return apply_noise(img, sigma, seed)
    noise: np.ndarray = np.random.default_rng(seed).normal(0.0, 1.0, img.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), corr)
    noise /= float(noise.std()) + 1e-9
    return np.clip(img.astype(np.float32) + sigma * noise, 0, 255).astype(np.uint8)


def t_g_noise() -> str:
    bad: List[str] = []
    gen: torch.Generator = torch.Generator(device=DEV)
    for sigma, corr, label in ((0.8, 0.0, 'sigma 0.8'), (3.0, 0.0, 'sigma 3.0'), (6.0, 0.0, 'sigma 6.0'), (2.0, 1.5, 'ood-soft (sigma 2.0, corr 1.5)')):
        n: int = 64 if corr > 0 else 16
        shots: List[Shot] = [neutral_shot(noise_sigma=sigma, noise_corr=corr, noise_seed=i) for i in range(n)]
        with torch.no_grad():
            flat: torch.Tensor = torch.full((n, 3, PHOTO, PHOTO), 128.0, device=DEV)
            g_out: np.ndarray = G.add_noise(flat, shots, gen).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        c_out: np.ndarray = np.stack([cpu_noise(sigma, corr, i) for i in range(n)])
        se: float = (max(1.0, 2 * math.pi * corr ** 2) / (n * PHOTO ** 2)) ** 0.5
        say(f'g_noise {label}: standard error of a channel correlation {se:.5f} (n {n})')
        gm, gs, ga, gc = noise_stats(g_out)
        cm, cs, ca, cc = noise_stats(c_out)
        say(f'g_noise {label}: mean gpu {gm:+.4f} cpu {cm:+.4f}; sd gpu {gs:.4f} cpu {cs:.4f} (sqrt(s^2 + 1/12) {(sigma ** 2 + 1 / 12) ** 0.5:.4f})')
        say('g_noise ' + label + ': autocorrelation at ' + ' '.join(str(s) for s in SHIFTS) + ': gpu ' + ' '.join(f'{r:+.4f}' for r in ga)
            + ' | cpu ' + ' '.join(f'{r:+.4f}' for r in ca))
        say(f'g_noise {label}: channels R-G R-B G-B: gpu ' + ' '.join(f'{r:+.4f}' for r in gc) + ' | cpu ' + ' '.join(f'{r:+.4f}' for r in cc))
        expect(bad, abs(gs / cs - 1) <= 0.005, f'{label}: sd gpu/cpu = {gs / cs:.5f}')
        if corr <= 0:
            want: float = (sigma ** 2 + 1 / 12) ** 0.5
            expect(bad, abs(gs / want - 1) <= 0.01 and abs(cs / want - 1) <= 0.01, f'{label}: sd {gs:.4f} / {cs:.4f} not within 1 % of {want:.4f}')
            expect(bad, max(abs(r) for r in ga + ca) <= 0.003, f'{label}: independent noise has an autocorrelation')
        expect(bad, abs(gm - cm) <= 0.01, f'{label}: mean gpu {gm:+.4f} cpu {cm:+.4f}')
        expect(bad, max(abs(x - y) for x, y in zip(ga, ca)) <= 0.003, f'{label}: autocorrelation gpu vs cpu')
        # correlated noise: sum of rho^2 = 2 pi corr^2 (14.1 at 1.5), so a channel correlation's standard error is sqrt(14.1 / (n x 640^2)): 1.5e-3 at 16 canvases, 7e-4 at 64 -- 0.003 is 4 of them only at 64
        expect(bad, max(abs(r) for r in gc + cc) <= 0.003, f'{label}: channels correlated')
    assert not bad, '; '.join(bad)
    return 'sd, mean, autocorrelation and channel correlation of the GPU noise equal the CPU numpy noise (independent and ood-soft)'


# ── determinism ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def t_h_determinism() -> str:
    canv: np.ndarray = inputs().canvases[:30]
    labs: List[LabGap] = list(probe_labs(ProbeConfig().ood).values())
    shots: List[Shot] = [draw_shot(labs[i % len(labs)], random.Random(100 + i)) for i in range(len(canv))]
    base: torch.Tensor = to_dev(canv)
    whole: torch.Tensor = G.simulate(base, shots)
    assert torch.equal(whole, G.simulate(base, shots)), 'the same batch twice'
    parts: torch.Tensor = torch.cat([G.simulate(base[s:s + 7], shots[s:s + 7]) for s in range(0, len(canv), 7)])
    assert torch.equal(whole, parts), 'a view depends on the chunk it is in'
    perm: List[int] = list(range(len(canv)))
    random.Random(0).shuffle(perm)
    shuffled: torch.Tensor = G.simulate(base[perm], [shots[i] for i in perm])
    back: torch.Tensor = torch.empty_like(whole)
    back[torch.tensor(perm, device=DEV)] = shuffled
    assert torch.equal(whole, back), 'a view depends on the order of the batch'
    k: int = next(i for i, s in enumerate(shots) if s.noise_sigma > 0)
    other: List[Shot] = list(shots)
    other[k] = replace(shots[k], noise_seed=shots[k].noise_seed + 1)
    assert not torch.equal(G.simulate(base[k:k + 1], [other[k]]), whole[k:k + 1]), 'another noise_seed gave the same canvas'
    return f'{len(canv)} views: the same batch twice, chunks of 7 vs one, a shuffled order: torch.equal; another noise_seed: another canvas'


# ── targets, timing (printed) ────────────────────────────────────────────────────────────────────────────────────────────

def t_i_targets() -> str:
    canv: np.ndarray = inputs().canvases[:32]
    labs: Dict[str, LabGap] = sim_labs()
    for name in ('bench-real', 'bench'):
        lab: LabGap = labs[name]
        shots: List[Shot] = [draw_shot(lab, random.Random(i)) for i in range(len(canv))]
        g: np.ndarray = np.concatenate([G.simulate(to_dev(canv[s:s + 16]), shots[s:s + 16], noise=False).cpu().numpy() for s in range(0, len(canv), 16)])
        clean: List[np.ndarray] = [simulate_with_gt(canv[i], cfg=replace(lab.base, noise_sigma=0.0), rng=random.Random(i), output_wh=(PHOTO, PHOTO))[0] for i in range(len(canv))]
        sig: float = float(lab.base.noise_sigma)
        noisy_a: np.ndarray = np.stack([apply_noise(c, sig, 1) for c in clean])
        noisy_b: np.ndarray = np.stack([apply_noise(c, sig, 2) for c in clean])
        with torch.no_grad():
            m_gpu: torch.Tensor = field_maps(grey_of(g), CFG.cofig, FIELD_PX)
            m_cpu: torch.Tensor = field_maps(grey_of(np.stack(clean)), CFG.cofig, FIELD_PX)
            m_a: torch.Tensor = field_maps(grey_of(noisy_a), CFG.cofig, FIELD_PX)
            m_b: torch.Tensor = field_maps(grey_of(noisy_b), CFG.cofig, FIELD_PX)
        one: Dict[str, float] = part_sd_rms(m_gpu, m_cpu, m_cpu)
        two: Dict[str, float] = part_sd_rms(m_a, m_b, m_cpu)
        say(f'i_targets {name}: RMS / sd of the part; (i) GPU vs CPU (noise off), (ii) CPU noise seed A vs B, (i)/(ii)')
        for part in one:
            say(f'i_targets {name} {part}: (i) {one[part]:.5f} (ii) {two[part]:.5f} (i)/(ii) {one[part] / max(two[part], 1e-12):.4f}')
    return 'printed: the difference of the targets between GPU and CPU against the difference a change of noise makes'


def t_j_timing() -> str:
    canv: np.ndarray = inputs().canvases
    img: np.ndarray = canv[0]

    def ms(fn: Callable[[], Any], reps: int = 20) -> float:
        fn()
        t0: float = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1e3

    for name, fn in (('apply_color', lambda: apply_color(img, 0, 1.0, 0.9)), ('apply_brightness_contrast', lambda: apply_brightness_contrast(img, 0.05, 0.05)),
                     ('apply_color_temp', lambda: apply_color_temp(img, 0.05)), ('apply_defocus', lambda: apply_defocus(img, 2)),
                     ('apply_chromatic', lambda: apply_chromatic(img, 2)), ('apply_vignette', lambda: apply_vignette(img, 0.3)),
                     ('apply_noise', lambda: apply_noise(img, 3.0, 1)), ('apply_jpeg q85', lambda: apply_jpeg(img, 85))):
        say(f'j_timing cpu {name}: {ms(fn):.2f} ms a canvas (one thread)')
    ref: np.ndarray = inputs().reads[0]
    geo: Geometry = draw_geometry(random.Random(0), LEVELS['large'])
    say(f'j_timing cpu Sims.frame ({ref.shape[0]} px -> {PHOTO}, large): {ms(lambda: frame(ref, geo, PHOTO), 5):.2f} ms a canvas')
    if DEV.type != 'cuda':
        return 'cpu numbers printed; no GPU: the GPU timings skipped'
    labs: List[LabGap] = list(sim_labs().values())
    n: int = 768
    base: torch.Tensor = to_dev(np.concatenate([canv] * (n // len(canv) + 1))[:n])
    shots: List[Shot] = [draw_shot(labs[i % len(labs)], random.Random(i)) for i in range(n)]

    def run(noise: bool) -> float:
        torch.cuda.synchronize()
        t0: float = time.perf_counter()
        for s in range(0, n, SIM_CHUNK):
            G.simulate(base[s:s + SIM_CHUNK], shots[s:s + SIM_CHUNK], noise=noise)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    run(False)
    torch.cuda.reset_peak_memory_stats()
    t_plain: float = run(False)
    t_all: float = run(True)
    say(f'j_timing gpu simulate: {n} canvases in chunks of {SIM_CHUNK}: {t_all:.3f} s with noise, {t_plain:.3f} s without (the noise {t_all - t_plain:.3f} s); '
        f'peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')
    big: int = read_px(CFG.data, PHOTO)
    refs: torch.Tensor = torch.randint(0, 256, (32, big, big, 3), dtype=torch.uint8, device=DEV)
    for count in (128, 384):
        geos: List[Geometry] = [draw_geometry(random.Random(i), LEVELS['large']) for i in range(count)]
        rows: List[int] = [i % 32 for i in range(count)]
        G.frames(refs, rows[:32], geos[:32], PHOTO)
        torch.cuda.synchronize()
        t0: float = time.perf_counter()
        G.frames(refs, rows, geos, PHOTO)
        torch.cuda.synchronize()
        say(f'j_timing gpu frames: {count} large canvases ({big} px -> {PHOTO}): {time.perf_counter() - t0:.3f} s')
    return 'timings printed'


# ── the framing ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def frame_pairs(geos: List[Geometry], rows: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """GPU and CPU canvases of the same geometries over `inputs().reads`: uint8 [m, 640, 640, 3] each."""
    inp: Inputs = inputs()
    refs: torch.Tensor = to_dev(np.stack(inp.reads))
    with torch.no_grad():
        gpu: np.ndarray = G.frames(refs, rows, geos, PHOTO).cpu().numpy()
    cpu: np.ndarray = np.stack([frame(inp.reads[r], g, PHOTO) for r, g in zip(rows, geos)])
    return gpu, cpu


def frame_rows(rows: List[int]) -> bool:
    """Whether every read of `rows` is a worst case (stripes, checkerboard): its numbers are only printed."""
    return all(inputs().worst[r] for r in rows)


def t_k_frame() -> str:
    inp: Inputs = inputs()
    n_ref: int = len(inp.reads)
    bad: List[str] = []
    real_rows: List[int] = [r for r in range(n_ref) if not inp.worst[r]]
    worst_rows: List[int] = [r for r in range(n_ref) if inp.worst[r]]

    def judge(name: str, params: str, geos: List[Geometry], rows: List[int], lim: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
        gpu, cpu = frame_pairs(geos, rows)
        st: Dict[str, float] = diff_stats(gpu, cpu)
        report(name, params + (' [worst case: printed only]' if frame_rows(rows) else ''), st)
        if not frame_rows(rows):
            expect(bad, st['mean'] <= lim[0] and abs(st['bias']) <= lim[1] and st['p999'] <= lim[2] and st['max'] <= lim[3], f'{name} {params}')
        return gpu, cpu

    # (a) identity: the crop, bit for bit
    rows_a: List[int] = list(range(n_ref))
    ga, ca = frame_pairs([Geometry()] * n_ref, rows_a)
    expect(bad, bool(np.array_equal(ga, ca)), '(a) the identity is not the crop')
    say(f'k_frame (a) identity: equal to Sims.frame on {n_ref} reads: {bool(np.array_equal(ga, ca))}')
    framing: Tuple[float, float, float, float] = (0.75, 0.1, 4.0, 12.0)
    whole: Tuple[float, float, float, float] = (1.0, 0.1, 5.0, 16.0)
    cyc: List[int] = (real_rows or worst_rows)
    draws: Dict[str, List[Geometry]] = {lv: [draw_geometry(random.Random(50 + i), LEVELS[lv]) for i in range(8)] for lv in ('light', 'large')}
    for scope, rows_pool in (('real', real_rows), ('worst', worst_rows)):
        if not rows_pool:
            continue
        # (b) the framing alone
        for lv, gs in draws.items():
            geos_b: List[Geometry] = [replace(g, k1=0.0) for g in gs] + ([Geometry(shift=(0.5, 0.5))] if lv == 'light' else [])
            judge('k_frame (b) framing only', f'{lv} {scope}', geos_b, [rows_pool[i % len(rows_pool)] for i in range(len(geos_b))], framing)
        # (c) the lens alone
        geos_c: List[Geometry] = [Geometry(k1=k) for k in (-0.10, -0.05, 0.05, 0.10)]
        judge('k_frame (c) lens only', scope, geos_c, [rows_pool[i % len(rows_pool)] for i in range(len(geos_c))], framing)
        # (d) everything
        for lv, gs in draws.items():
            judge('k_frame (d) all', f'{lv} {scope}', gs, [rows_pool[i % len(rows_pool)] for i in range(len(gs))], whole)
    # (e) no half-pixel error, (f) the signals
    if real_rows:
        all_g: List[Geometry] = draws['light'] + draws['large']
        rows_e: List[int] = [real_rows[i % len(real_rows)] for i in range(len(all_g))]
        gpu, cpu = frame_pairs(all_g, rows_e)
        _, shifted = frame_pairs([replace(g, shift=(g.shift[0], g.shift[1] + 0.25)) for g in all_g], rows_e)
        for i, g in enumerate(all_g):
            a: float = float(np.sqrt(np.mean((gpu[i].astype(np.float64) - cpu[i]) ** 2)))
            b: float = float(np.sqrt(np.mean((shifted[i].astype(np.float64) - cpu[i]) ** 2)))
            say(f'k_frame (e) draw {i}: rms(gpu - cpu) {a:.4f}, rms(cpu(shift + 0.25 px) - cpu) {b:.4f}, ratio {a / max(b, 1e-12):.4f}')
            expect(bad, a <= 0.2 * b, f'(e) draw {i}: A {a:.4f} > 0.2 x B {b:.4f}')
        with torch.no_grad():
            m_g: torch.Tensor = field_maps(grey_of(gpu), CFG.cofig, FIELD_PX)
            m_c: torch.Tensor = field_maps(grey_of(cpu), CFG.cofig, FIELD_PX)
            m_s: torch.Tensor = field_maps(grey_of(shifted), CFG.cofig, FIELD_PX)
        say('k_frame (f) RMS / sd of the part: gpu vs cpu ' + ' '.join(f'{k} {v:.4f}' for k, v in part_sd_rms(m_g, m_c, m_c).items())
            + ' | cpu shifted 0.25 px vs cpu ' + ' '.join(f'{k} {v:.4f}' for k, v in part_sd_rms(m_s, m_c, m_c).items()))
    # (g) determinism
    geos_g: List[Geometry] = draws['light'] + draws['large']
    rows_g: List[int] = [cyc[i % len(cyc)] for i in range(len(geos_g))]
    one, _ = frame_pairs(geos_g, rows_g)
    two, _ = frame_pairs(geos_g, rows_g)
    kept: int = G.FRAME_CHUNK
    G.FRAME_CHUNK = 5
    try:
        five, _ = frame_pairs(geos_g, rows_g)
    finally:
        G.FRAME_CHUNK = kept
    expect(bad, bool(np.array_equal(one, two)), '(g) the same batch twice')
    expect(bad, bool(np.array_equal(one, five)), '(g) FRAME_CHUNK 5 vs the default')
    assert not bad, '; '.join(bad)
    return 'frames: identity = crop; framing, lens, all within the limits; no half-pixel error; deterministic'


_TESTS: List[Callable[[], str]] = [t for n, t in sorted(globals().items()) if n.startswith('t_')]


def main() -> int:
    print(f'device {DEV}', flush=True)
    torch.set_grad_enabled(False)
    try:
        for fn in _TESTS:
            check(fn.__name__[4:].replace('_', ' '), fn)
    finally:
        out: str = os.environ.get('COAE_SIM_OUT', '')
        if out:
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, 'sim_gpu.txt'), 'w') as f:
                f.write('\n'.join(_LINES) + '\n')
    failed: List[str] = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
