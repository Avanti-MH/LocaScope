'''The Co-figure: the signals the experiments found consistent and discriminative, and the targets of a view made from them (spec.md §4).

THE TRAINING PATH (no warp, no average over geometries): every view's Co-figure is made on its own canvas, over the FM's field of its
centre tile, the signals at every pixel and averaged afterwards in `cfg.out_cells` x `cfg.out_cells` cells of field_px / out_cells px.
The views that share a target are decided by the mechanism (which views of a place are twins), what the target is by the variant.

    own   = field_maps(grey, cfg, field_px)                # canvases [B, H, W] (0-255) -> [B, N, t, t], channel order `layout(cfg)`
    tg    = view_targets(own, slots, sources, twins, variant)               # each view's target (and the observations of the NLL)
    desc  = shared_moments(grey, slots, sources, shared, cfg, field_px, pool_cells, dilate_px)   # cls''s target: the place descriptor
    # `place_moments` is the descriptor of a canvas: the raw first and second moments of its Co-figure over pool_cells x pool_cells cells

THE ALIGNED PATH below (`signal_maps`, `observations`, `place_targets` ... ) is not taken by training now; it is kept as it was:

    maps  = signal_maps(grey, cfg)                         # photo canvases [B, H, W] (0-255) -> {signal: [B, C, H // cell, W // cell]}
    obs   = observations(maps, cfg)                        # [B, N, h, w], channel order `layout(cfg)`, T z-scored a version
    tg    = place_targets(obs, geos, cfg, versions, photo_px)   # the targets of each version, in the reference's frame

THE SIGNALS (phasetiles' settings, `RetrieverPhaseTiles.phase_map`, batched; frequencies in cycles / 256 px):
    localphase   a monogenic vector [even, odd x, odd y] a frequency (`freqs`), unit at every pixel, averaged over the cell
    congruency   the phase congruency over `pc_scales` and its orientation: [pc, pc cos 2t, pc sin 2t]
    t_weighted   T weighted on the rings `t_band`: the band of the image itself, averaged over the cell; z-scored a version over the
                 canvas's interior (the taper left out)

THE ALIGNMENT (the aligned path; training does not take it now) is of the Co-figure only (the network's input keeps its geometry). Each version's maps are made on its own photo canvas
(larger than the tile, so that the warp back has data), then sampled at the cells of one reference grid round the place (bilinear,
`Sims.Geometry`'s known map: framing, then the simulator's stage shift and lens distortion) with their directed parts turned into the reference's frame (the odd pair of local phase by the turn,
the doubled-angle pair of congruency by twice it). The versions of a place are averaged on that grid; each version's target is read
off it (`target_points`, bilinear) in one of two frames, `cfg.frame`:
    'scan'   the tile at the version's own centre (the place + its shift, any real value), the scan's axes and scale: the target
             follows the input's shift, NOT its scale, turn, stage shift or lens -- at the tile's edge a 20 % scale moves the input's
             content 25 px from the cell it is asked for, a 5 degree turn 11 px (a local-phase wave is 32 px and 10.7 px long)
    'input'  the cells of the version's own input tile mapped back by its whole known geometry, the directed pairs turned into its
             frame: the target lies where the input shows it. What stays the scan's: the frequency band (cycles / 256 scan px).
`align=False` (the blurred variant): the photo maps are taken as if they were in the
reference's frame (no warp, no turn).

A CELL IS VALID for a version's loss when (1) the version's input sees it (its centre maps inside the photo's tile; in the 'input'
frame every cell is the input's own), (2) every version
of the place has data there (inside its canvas, off the taper), (3) [the taper: part of (2)]. Blank glass is kept (spec.md).
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as TF

from locascope.core.stage2_retrieval.RetrieverPhaseTiles import _local_mean, _taper
from locascope.training.CoAE.Config import CoFigureConfig
from locascope.training.CoAE.Sims import Geometry


# ── the channel layout ─────────────────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Part:
    """One block of channels: a signal (and its frequency for local phase), where it starts, how many, and how its directed pair turns
    (`turn`: 1 = the odd pair of local phase, at channels start+1, start+2; 2 = congruency's doubled-angle pair at start+1, start+2;
    0 = a scalar)."""
    signal: str
    start: int
    size: int
    turn: int
    unit: bool                    # a direction (local phase: vMF in the dist arm), else Gaussian channels


def layout(cfg: CoFigureConfig) -> List[Part]:
    out: List[Part] = []
    at: int = 0
    for s in cfg.signals:
        if s == 'localphase':
            for _ in cfg.freqs:
                out.append(Part(s, at, 3, 1, True))
                at += 3
        elif s == 'congruency':
            out.append(Part(s, at, 3, 2, False))
            at += 3
        else:
            out.append(Part(s, at, 1, 0, False))
            at += 1
    return out


def part_names(cfg: CoFigureConfig) -> List[str]:
    """A short name a part of `layout`, for the logs: lp<frequency> (local phase), pc (congruency), tw (T weighted)."""
    out: List[str] = []
    for s in cfg.signals:
        out += [f'lp{f:g}' for f in cfg.freqs] if s == 'localphase' else ['pc' if s == 'congruency' else 'tw']
    return out


def n_channels(cfg: CoFigureConfig) -> int:
    """Channels of the point output and of `observations`."""
    return sum(p.size for p in layout(cfg))


def n_dist_channels(cfg: CoFigureConfig) -> int:
    """Channels of the dist output: a direction block 3 (mean direction) + 1 (log concentration); a Gaussian channel 2 (mean, log sd)."""
    return sum(4 if p.unit else 2 * p.size for p in layout(cfg))


# ── the signals ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def _cells(x: torch.Tensor, cell: int) -> torch.Tensor:
    h, w = x.shape[-2] // cell * cell, x.shape[-1] // cell * cell
    return TF.avg_pool2d(x[..., :h, :w], cell)


def _pixel_signals(grey: torch.Tensor, cfg: CoFigureConfig) -> Dict[str, torch.Tensor]:
    """The signals of grey canvases [B, H, W] (float 0-255) at every pixel: {signal: [B, C, H, W]}."""
    b, h, w = grey.shape
    dev: torch.device = grey.device
    g: torch.Tensor = grey.float()
    d: torch.Tensor = (g - g.mean(dim=(1, 2), keepdim=True)) * _taper(h, cfg.taper_px, dev)[None, :, None] * _taper(w, cfg.taper_px, dev)[None, None, :]
    f: torch.Tensor = torch.fft.fft2(d)
    fy: torch.Tensor = (torch.fft.fftfreq(h, device=dev) * 256.0)[:, None].expand(h, w)
    fx: torch.Tensor = (torch.fft.fftfreq(w, device=dev) * 256.0)[None, :].expand(h, w)
    r: torch.Tensor = torch.hypot(fy, fx).clamp(min=1e-6)

    def mono(f0: float) -> torch.Tensor:                                     # [B, 3, H, W]: even, odd x, odd y
        lg: torch.Tensor = torch.exp(-torch.log(r / f0) ** 2 / (2 * np.log(cfg.lg_sigma) ** 2))
        lg[0, 0] = 0.0
        fg: torch.Tensor = f * lg
        return torch.stack([torch.fft.ifft2(fg).real, torch.fft.ifft2(fg * (1j * fx / r)).real, torch.fft.ifft2(fg * (1j * fy / r)).real], 1)

    out: Dict[str, torch.Tensor] = {}
    if 'localphase' in cfg.signals:
        parts: List[torch.Tensor] = []
        for f0 in cfg.freqs:
            m: torch.Tensor = mono(float(f0))
            parts.append(m / m.norm(dim=1, keepdim=True).clamp(min=1e-6))
        out['localphase'] = torch.cat(parts, 1)
    if 'congruency' in cfg.signals:
        acc: torch.Tensor = torch.zeros(b, 3, h, w, device=dev)
        amp_sum: torch.Tensor = torch.zeros(b, h, w, device=dev)
        for f0 in cfg.pc_scales:
            m = mono(float(f0))
            acc += m
            amp_sum += m.norm(dim=1)
        energy: torch.Tensor = acc.norm(dim=1)
        pc: torch.Tensor = (energy - 0.05 * _local_mean(amp_sum, 256)).clamp(min=0) / (amp_sum + 1e-6)
        theta: torch.Tensor = torch.atan2(acc[:, 2], acc[:, 1])
        out['congruency'] = torch.stack([pc, pc * torch.cos(2 * theta), pc * torch.sin(2 * theta)], 1)
    if 't_weighted' in cfg.signals:
        lo, hi = cfg.t_band
        ring: torch.Tensor = ((r >= lo - 0.5) & (r < hi + 0.5)).float()
        out['t_weighted'] = torch.fft.ifft2(f * ring).real[:, None]
    return out


def signal_maps(grey: torch.Tensor, cfg: CoFigureConfig) -> Dict[str, torch.Tensor]:
    """The signals of grey canvases [B, H, W] (float 0-255): {signal: [B, C, H // cell, W // cell]}."""
    return {s: _cells(v, cfg.cell) for s, v in _pixel_signals(grey, cfg).items()}


def signal_pixels(grey: torch.Tensor, cfg: CoFigureConfig) -> torch.Tensor:
    """The signals of grey canvases [B, H, W] at every pixel, in `layout` order: [B, N, H, W]; t_weighted z-scored a canvas over its
    interior (the taper left out)."""
    h, w = grey.shape[-2:]
    px: Dict[str, torch.Tensor] = _pixel_signals(grey, cfg)
    p: int = cfg.taper_px
    parts: List[torch.Tensor] = []
    for s in cfg.signals:
        m: torch.Tensor = px[s]
        if s == 't_weighted':
            inner: torch.Tensor = m[..., p:h - p, p:w - p]
            m = (m - inner.mean(dim=(2, 3), keepdim=True)) / inner.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        parts.append(m)
    return torch.cat(parts, 1)


def field_start(canvas_px: int, field_px: int, pad_px: int, taper_px: int) -> int:
    """The first pixel of the centred field, `pad_px` wider on each side; raises when it would reach the taper."""
    y0: int = (canvas_px - field_px) // 2 - pad_px
    if y0 < taper_px:
        raise ValueError(f'a {field_px} px field with {pad_px} px of padding on a {canvas_px} px canvas starts at {y0} px, inside the '
                         f'{taper_px} px taper')
    return y0


def field_maps(grey: torch.Tensor, cfg: CoFigureConfig, field_px: int, chunk: int = 128) -> torch.Tensor:
    """Each canvas's OWN Co-figure on the FM's field of its centre tile: [B, N, t, t], t = `cfg.out_cells`. The signals at every
    pixel, the centred `field_px` square cut out, then averaged a cell (field_px / t px). No warp, no average over versions. `grey`
    [B, S, S] is square; the canvases are taken `chunk` at a time."""
    t: int = cfg.out_cells
    if field_px % t != 0:
        raise ValueError(f'the field {field_px} px is not a multiple of out_cells {t}')
    if grey.shape[-1] != grey.shape[-2]:
        raise ValueError(f'the canvases must be square, got {tuple(grey.shape[-2:])}')
    cell: int = field_px // t
    y0: int = field_start(grey.shape[-1], field_px, 0, cfg.taper_px)
    out: List[torch.Tensor] = []
    for s in range(0, grey.shape[0], chunk):
        px: torch.Tensor = signal_pixels(grey[s:s + chunk], cfg)[..., y0:y0 + field_px, y0:y0 + field_px]
        out.append(TF.avg_pool2d(px, cell))
    return torch.cat(out, 0)


def n_moments(cfg: CoFigureConfig) -> int:
    """Channels of the place descriptor: the first moments (one a Co-figure channel) and the second moments (the products within each
    part: 6 for a part of 3, 1 for a scalar): 10 + (6 + 6 + 6 + 1) = 29."""
    return n_channels(cfg) + sum(p.size * (p.size + 1) // 2 for p in layout(cfg))


def second_moments(x: torch.Tensor, cfg: CoFigureConfig) -> torch.Tensor:
    """The products within each part of x [B, N, H, W] (`layout` order): a part a, b, c -> a2, b2, c2, ab, ac, bc; a part a -> a2.
    [B, n_moments - n_channels, H, W]."""
    out: List[torch.Tensor] = []
    for p in layout(cfg):
        v: List[torch.Tensor] = [x[:, p.start + i] for i in range(p.size)]
        out += [v[i] * v[i] for i in range(p.size)]
        out += [v[i] * v[j] for i in range(p.size) for j in range(i + 1, p.size)]
    return torch.stack(out, 1)


def place_moments(grey: torch.Tensor, cfg: CoFigureConfig, field_px: int, pool_cells: int, dilate_px: float,
                  chunk: int = 128) -> torch.Tensor:
    """The place descriptor of each canvas [n, M, c, c] (M = `n_moments`, c = `pool_cells`): the RAW first and second moments of its
    Co-figure (the products taken at every pixel, averaged afterwards: no centring, no normalisation) over pool_cells x pool_cells
    cells of the field, each cell's average taken over the cell moved by up to `dilate_px` on both axes -- the expectation over the
    shifts of a perturbation level (a separable pixel box of 2 pad + 1 px, pad = round(dilate_px), then the cells). That box is the
    distribution of `Sims.draw_geometry`'s shift (uniform on each axis independently): change one, change the other. The field is
    cut `pad` px wider on each side; it raises when that reaches the taper."""
    t: int = cfg.out_cells
    if field_px % t != 0 or t % pool_cells != 0:
        raise ValueError(f'the field {field_px} px must be a multiple of out_cells {t}, which one of pool_cells {pool_cells}')
    pad: int = int(round(dilate_px))
    k: int = 2 * pad + 1
    y0: int = field_start(grey.shape[-1], field_px, pad, cfg.taper_px)
    side: int = field_px + 2 * pad
    out: List[torch.Tensor] = []
    for s in range(0, grey.shape[0], chunk):
        x: torch.Tensor = signal_pixels(grey[s:s + chunk], cfg)[..., y0:y0 + side, y0:y0 + side]
        f: torch.Tensor = torch.cat([x, second_moments(x, cfg)], 1)                                # [b, M, side, side]
        if pad > 0:
            f = TF.avg_pool2d(TF.avg_pool2d(f, (k, 1), stride=1), (1, k), stride=1)               # [b, M, field, field]
        out.append(TF.avg_pool2d(f, field_px // pool_cells))                                       # [b, M, c, c]
    return torch.cat(out, 0)


def observations(maps: Dict[str, torch.Tensor], cfg: CoFigureConfig) -> torch.Tensor:
    """The per-version maps in `layout` order: [B, N, h, w]; t_weighted z-scored a version over the canvas's interior."""
    m_: int = int(np.ceil(cfg.taper_px / cfg.cell))
    parts: List[torch.Tensor] = []
    for s in cfg.signals:
        m: torch.Tensor = maps[s]
        if s == 't_weighted':
            inner: torch.Tensor = m[..., m_:m.shape[-2] - m_, m_:m.shape[-1] - m_]
            m = (m - inner.mean(dim=(2, 3), keepdim=True)) / inner.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        parts.append(m)
    return torch.cat(parts, 1)


# ── the alignment ───────────────────────────────────────────────────────────────────────────────────────────────────────

def _turn_pairs(x: torch.Tensor, turns_deg: torch.Tensor, cfg: CoFigureConfig) -> torch.Tensor:
    """The directed pairs of x [B, N, h, w] turned from each version's photo frame into the reference's: a direction in the photo is
    R(turn) of itself in the reference (Sims.Geometry)."""
    out: torch.Tensor = x.clone()
    a: torch.Tensor = torch.deg2rad(turns_deg.to(x))[:, None, None]
    for p in layout(cfg):
        if p.turn == 0:
            continue
        c, s = torch.cos(p.turn * a), torch.sin(p.turn * a)
        u, v = x[:, p.start + 1], x[:, p.start + 2]
        out[:, p.start + 1] = c * u - s * v
        out[:, p.start + 2] = s * u + c * v
    return out


def grid_cells(cfg: CoFigureConfig, shift_px: float) -> int:
    """The side in cells of the reference grid round a place: the tile, the largest shift on both sides, and a cell more each side
    (a shift of any real value is cropped bilinearly)."""
    return cfg.tile // cfg.cell + 2 * int(np.ceil(shift_px / cfg.cell)) + 2


def _lens(p: torch.Tensor, g: Geometry, photo_px: int) -> torch.Tensor:
    """Sims.lens_forward in torch: a framed-canvas point p [n, 2] (x, y) to the final photo (+ the stage shift, then the lens)."""
    p1: torch.Tensor = p + torch.tensor([g.stage[0], g.stage[1]], dtype=p.dtype, device=p.device)
    if g.k1 == 0.0 and g.k2 == 0.0:
        return p1
    c: float = (photo_px - 1) / 2.0
    q: torch.Tensor = p1.clone()
    for _ in range(6):
        r2: torch.Tensor = (((q - c) / c) ** 2).sum(-1, keepdim=True)
        q = c + (p1 - c) * (1.0 + g.k1 * r2 + g.k2 * r2 ** 2)
    return q


def _photo_points(geos: Sequence[Geometry], centres: torch.Tensor, photo_px: int, device: torch.device, shifted: bool = True) -> torch.Tensor:
    """Final photo coordinates [B, n, 2] (x, y; continuous) of reference points `centres` [n, 2] (x, y, relative to the place's
    centre; to the version's own centre when `shifted` is False) for each version: the framing p = c_photo + scale R(turn)^T (r -
    shift), then the simulator's stage shift and lens distortion (`_lens`)."""
    out: List[torch.Tensor] = []
    for g in geos:
        rot_t: torch.Tensor = torch.as_tensor(g.rotation().T * g.scale, dtype=torch.float32, device=device)
        rel: torch.Tensor = centres - (torch.tensor([g.shift[1], g.shift[0]], dtype=torch.float32, device=device) if shifted else 0.0)
        out.append(_lens(photo_px / 2 + rel @ rot_t.T, g, photo_px))
    return torch.stack(out)


def warp_to_grid(obs: torch.Tensor, geos: Sequence[Geometry], cfg: CoFigureConfig, photo_px: int, g_cells: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Each version's photo maps obs [B, N, hp, wp] sampled at the reference grid's cells round its place: [B, N, G, G], and where it has
    data [B, G, G] (inside its canvas, off the taper). `cfg.align=False`: the photo maps' centre G x G cells as they are."""
    dev: torch.device = obs.device
    b, n, hp, wp = obs.shape
    if not cfg.align:
        y0, x0 = (hp - g_cells) // 2, (wp - g_cells) // 2
        return obs[..., y0:y0 + g_cells, x0:x0 + g_cells], torch.ones(b, g_cells, g_cells, dtype=torch.bool, device=dev)
    k: torch.Tensor = (torch.arange(g_cells, device=dev, dtype=torch.float32) + 0.5) * cfg.cell - g_cells * cfg.cell / 2
    yy, xx = torch.meshgrid(k, k, indexing='ij')
    pts: torch.Tensor = _photo_points(geos, torch.stack([xx.flatten(), yy.flatten()], 1), photo_px, dev)     # [B, G*G, 2]
    grid: torch.Tensor = (2 * pts / photo_px - 1).view(b, g_cells, g_cells, 2)
    sampled: torch.Tensor = TF.grid_sample(obs, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    lo, hi = float(cfg.taper_px), float(photo_px - cfg.taper_px)
    has: torch.Tensor = ((pts >= lo) & (pts <= hi)).all(-1).view(b, g_cells, g_cells)
    return _turn_pairs(sampled, torch.tensor([g.turn for g in geos]), cfg), has


def visible(geos: Sequence[Geometry], cfg: CoFigureConfig, photo_px: int, device: torch.device) -> torch.Tensor:
    """Condition (1): the cells of each version's target tile (at its own centre) whose centre its input tile sees: [B, t, t]."""
    t: int = cfg.tile // cfg.cell
    k: torch.Tensor = (torch.arange(t, device=device, dtype=torch.float32) + 0.5) * cfg.cell - cfg.tile / 2
    yy, xx = torch.meshgrid(k, k, indexing='ij')
    rel: torch.Tensor = torch.stack([xx.flatten(), yy.flatten()], 1)
    q: torch.Tensor = _photo_points(geos, rel, photo_px, device, shifted=False)          # relative to the version's own centre
    return ((q - photo_px / 2).abs() <= cfg.tile / 2).all(-1).view(len(geos), t, t)


def _lens_back(q: torch.Tensor, g: Geometry, photo_px: int) -> torch.Tensor:
    """`_lens` undone: a final photo point q [n, 2] (x, y) back to the framed canvas (the lens, then the stage shift). Exact: `_lens`
    solves q = c + (p1 - c)(1 + k1 r2(q) + k2 r2(q)^2) for q, which gives p1 from q directly."""
    p1: torch.Tensor = q
    if g.k1 != 0.0 or g.k2 != 0.0:
        c: float = (photo_px - 1) / 2.0
        r2: torch.Tensor = (((q - c) / c) ** 2).sum(-1, keepdim=True)
        p1 = c + (q - c) / (1.0 + g.k1 * r2 + g.k2 * r2 ** 2)
    return p1 - torch.tensor([g.stage[0], g.stage[1]], dtype=q.dtype, device=q.device)


def target_points(geo: Geometry, cfg: CoFigureConfig, photo_px: int, device: torch.device) -> torch.Tensor:
    """Where the cells of a version's target tile lie on the reference grid: [t * t, 2] (x, y; px relative to the place's centre).
    `cfg.frame` 'scan': the tile at the place's centre + the version's shift, the scan's axes and scale. 'input': the cells of the
    version's own input tile (the photo canvas's centre tile), mapped back by its known geometry -- `_photo_points` undone: the lens
    and the stage shift, then the framing r = shift + R(turn) (p - c_photo) / scale."""
    t: int = cfg.tile // cfg.cell
    k: torch.Tensor = (torch.arange(t, device=device, dtype=torch.float32) + 0.5) * cfg.cell - cfg.tile / 2
    yy, xx = torch.meshgrid(k, k, indexing='ij')
    rel: torch.Tensor = torch.stack([xx.flatten(), yy.flatten()], 1)
    shift: torch.Tensor = torch.tensor([geo.shift[1], geo.shift[0]], dtype=torch.float32, device=device)
    if cfg.frame == 'scan':
        return rel + shift
    p: torch.Tensor = _lens_back(photo_px / 2 + rel, geo, photo_px) - photo_px / 2
    rot: torch.Tensor = torch.as_tensor(geo.rotation() / geo.scale, dtype=torch.float32, device=device)
    return p @ rot.T + shift


def _read(x: torch.Tensor, pts: torch.Tensor, cfg: CoFigureConfig, g_cells: int) -> torch.Tensor:
    """A grid x [..., G, G] (float) read at the points pts [t * t, 2] (`target_points`; any real value: bilinear between the grid's
    cells): [..., t, t]."""
    t: int = cfg.tile // cfg.cell
    lead = x.shape[:-2]
    side: float = g_cells * cfg.cell
    grid: torch.Tensor = (2 * (pts + side / 2) / side - 1).view(1, t, t, 2)
    flat: torch.Tensor = x.reshape(1, -1, g_cells, g_cells).float()
    return TF.grid_sample(flat, grid, mode='bilinear', padding_mode='zeros', align_corners=False).reshape(*lead, t, t)


@dataclass
class Targets:
    """Each view's targets (B = places x views, a place's views contiguous), made by `view_targets` (the training path: valid is all
    True) or by `place_targets` (the aligned path: conditions (1)-(3)):
    mean   [B, N, t, t]      the point target (`view_targets`: the shared target of the view's group; `place_targets`: the place's
                             versions averaged on the reference grid, read in the version's `cfg.frame`)
    obs    [B, V, N, t, t]   the maps the view's target is made of: the observations of the dist arm's NLL (V the sharers; 1 under
                             vref, the target's own map)
    valid  [B, t, t]         the cells the losses count
    weight [B] or None       each view's loss weight (its place's rung, Supply.weights_of); None = 1"""
    mean: torch.Tensor
    obs: torch.Tensor
    valid: torch.Tensor
    weight: Optional[torch.Tensor] = None

    @property
    def cells(self) -> torch.Tensor:
        """The losses' cell weights [B, t, t]: `valid`, times the version's `weight` when there is one."""
        m: torch.Tensor = self.valid.float()
        return m if self.weight is None else m * self.weight.to(m.device, m.dtype).view(-1, 1, 1)


def place_targets(obs: torch.Tensor, geos: Sequence[Geometry], cfg: CoFigureConfig, versions: int, photo_px: int,
                  shift_px: float) -> Targets:
    """The targets from each version's photo maps obs [B, N, hp, wp] (B = places x `versions`, contiguous) and its geometry."""
    g_cells: int = grid_cells(cfg, shift_px)
    on_grid, has = warp_to_grid(obs, geos, cfg, photo_px, g_cells)                       # [B, N, G, G], [B, G, G]
    b, n = on_grid.shape[:2]
    p_: int = b // versions
    grid_v: torch.Tensor = on_grid.view(p_, versions, n, g_cells, g_cells)
    has_v: torch.Tensor = has.view(p_, versions, g_cells, g_cells)
    all_have: torch.Tensor = has_v.all(1)                                                 # (2): every version has data
    mean_g: torch.Tensor = grid_v.mean(1)                                                 # [P, N, G, G]
    own: bool = cfg.frame == 'input'                                                      # the target in the version's own frame
    see: Optional[torch.Tensor] = None if own else visible(geos, cfg, photo_px, obs.device)   # (1); its own cells: all seen
    means: List[torch.Tensor] = []
    obs_l: List[torch.Tensor] = []
    valid: List[torch.Tensor] = []
    for i, g in enumerate(geos):
        pl: int = i // versions
        pts: torch.Tensor = target_points(g, cfg, photo_px, obs.device)
        mean_i: torch.Tensor = _read(mean_g[pl], pts, cfg, g_cells)                       # [N, t, t]
        obs_i: torch.Tensor = _read(grid_v[pl], pts, cfg, g_cells)                        # [V, N, t, t]
        if own:                                              # the directed pairs from the reference's frame into the version's
            back: torch.Tensor = torch.tensor([-g.turn])
            mean_i = _turn_pairs(mean_i[None], back, cfg)[0]
            obs_i = _turn_pairs(obs_i, back.expand(versions), cfg)
        has_i: torch.Tensor = _read(all_have[pl].float(), pts, cfg, g_cells) > 0.999      # every grid cell it leans on has data
        means.append(mean_i)
        obs_l.append(obs_i)
        valid.append(has_i if see is None else has_i & see[i])
    return Targets(torch.stack(means), torch.stack(obs_l), torch.stack(valid))


def view_targets(own: torch.Tensor, slots: int, sources: int, twins: bool, variant: str) -> Targets:
    """The targets of each view from the views' OWN maps own [B, N, t, t] (`field_maps`; B = places x slots x sources, a place's views
    contiguous, slot outside, source inside), no warp:
    twins (dep, mix): a view shares with its slot's views. vref: the slot's ref map; vavg, vdist: the slot's mean. obs [B, sources, N, t, t]
    else (ind):       a place's views share one figure, unaligned. vref: the canonical ref's map (slot 0, source 0); vavg, vdist: the
                      mean of all the place's maps. obs [B, slots x sources, N, t, t]
    vref: obs [B, 1, N, t, t], the target's one map itself (the single observation of a point target; `Losses.dist_terms` reads it
    only under vdist).
    valid: every cell."""
    b, n, t, _ = own.shape
    v: int = slots * sources
    p: int = b // v
    x: torch.Tensor = own.view(p, slots, sources, n, t, t)
    valid: torch.Tensor = torch.ones(b, t, t, dtype=torch.bool, device=own.device)
    if variant == 'vref':
        ref: torch.Tensor = x[:, :, :1] if twins else x[:, :1, :1]
        mean_ref: torch.Tensor = ref.expand_as(x).reshape(b, n, t, t)
        return Targets(mean_ref, mean_ref[:, None], valid)
    if twins:
        obs: torch.Tensor = x[:, :, None].expand(p, slots, sources, sources, n, t, t)
        mean: torch.Tensor = x.mean(2, keepdim=True)
    else:
        obs = x.reshape(p, 1, 1, v, n, t, t).expand(p, slots, sources, v, n, t, t)
        mean = x.mean((1, 2), keepdim=True)
    return Targets(mean.expand_as(x).reshape(b, n, t, t), obs.reshape(b, -1, n, t, t), valid)


def targets_of_views(grey: torch.Tensor, cfg: CoFigureConfig, field_px: int, slots: int, sources: int, twins: bool,
                     variant: str, kept: int) -> Targets:
    """`view_targets(field_maps(grey, cfg, field_px), slots, sources, twins, variant)`, the maps made only of the canvases a target
    reads: vref reads the ref canvases alone (twins: each slot's; else the canonical one). `grey` is [p x slots x kept, H, W]: the
    canvases of each slot's first `kept` sources (`CoaeConfig.canvases`; made by `train.views_of`); the other variants read every source's, so `kept` must be `sources`."""
    if variant != 'vref':
        if kept != sources:
            raise ValueError(f'{variant} reads every source\'s canvas: kept {kept} of {sources}')
        return view_targets(field_maps(grey, cfg, field_px), slots, sources, twins, variant)
    _, h, w = grey.shape
    p: int = grey.shape[0] // (slots * kept)
    b: int = p * slots * sources
    g: torch.Tensor = grey.view(p, slots, kept, h, w)
    refs: torch.Tensor = g[:, :, 0] if twins else g[:, :1, 0]                 # [p, r, h, w], r = slots or 1
    r: int = refs.shape[1]
    m: torch.Tensor = field_maps(refs.reshape(p * r, h, w), cfg, field_px)    # [p r, N, t, t]
    n, t = m.shape[1], m.shape[-1]
    mean: torch.Tensor = m.view(p, r, 1, n, t, t).expand(p, slots, sources, n, t, t).reshape(b, n, t, t)
    return Targets(mean, mean[:, None], torch.ones(b, t, t, dtype=torch.bool, device=grey.device))


def shared_moments(grey: torch.Tensor, slots: int, sources: int, shared: bool, cfg: CoFigureConfig, field_px: int,
                   pool_cells: int, dilate_px: float, kept: int) -> torch.Tensor:
    """The place target of every view [B, M, c, c], from the REF canvases only (grey [p x slots x kept, H, W]: each slot's first `kept`
    sources' canvases, `CoaeConfig.canvases`; B = p x slots x sources):
    shared (mix, ind): the canonical ref's descriptor (slot 0, source 0; widened by `dilate_px`), the same for all of the place's views.
    else (dep):        each slot's ref's descriptor (`dilate_px` must be 0), the same for that slot's views."""
    v: int = slots * sources
    p: int = grey.shape[0] // (slots * kept)
    h, w = grey.shape[-2:]
    g: torch.Tensor = grey.view(p, slots, kept, h, w)
    if shared:
        m: torch.Tensor = place_moments(g[:, 0, 0], cfg, field_px, pool_cells, dilate_px)
        m = m[:, None].expand(p, v, *m.shape[1:])
    else:
        if dilate_px != 0.0:
            raise ValueError(f'the descriptor of a slot is not widened: dilate_px must be 0, got {dilate_px}')
        m = place_moments(g[:, :, 0].reshape(p * slots, h, w), cfg, field_px, pool_cells, 0.0)
        m = m.view(p, slots, 1, *m.shape[1:]).expand(p, slots, sources, *m.shape[1:])
    return m.reshape(p * v, *m.shape[-3:])
