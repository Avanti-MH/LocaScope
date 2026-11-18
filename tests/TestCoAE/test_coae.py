#!/usr/bin/env python3
"""Tests of CoAE's parts that need no slide and no FM (`locascope/training/CoAE`): the channel layout, the perturbation levels and
the views of a place (twins, siblings), the Co-figure's field maps, the view targets and the place descriptor (moments), the alignment
of the Co-figure (the aligned path, kept: a version made by a known geometry, its maps warped back, against the reference's), the
visible-cell mask, the place targets, the layer mix's caps, the Swin encoder / ConvNeXt decoder shapes, that cls' carries nothing of q
but its choice, the bottleneck noise, the gradient reversal, the probe's scores and the config, and the losses (values where they are
known, and that each backpropagates).

    python tests/TestCoAE/test_coae.py

CPU is enough (a GPU is used when there is one). The slide and FM parts: tests/TestCoAE/test_coae_slide.py.
"""
from __future__ import annotations

import math
import os
import random
import sys
import tempfile
import traceback
from dataclasses import asdict, replace
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np                                                                   # noqa: E402
import torch                                                                         # noqa: E402
import torch.nn as nn                                                                # noqa: E402
import torch.nn.functional as TF                                                     # noqa: E402

from locascope.training.CoAE.CoFigure import (Targets, _photo_points, _turn_pairs, field_maps, layout, n_channels, n_dist_channels,   # noqa: E402
                                              n_moments, observations, part_names, place_moments, place_targets, second_moments,
                                              shared_moments, signal_maps, signal_pixels, target_points, targets_of_views, view_targets, visible,
                                              warp_to_grid, grid_cells)
from locascope.training.CoAE.Config import LEVELS, CoaeConfig, ProbeConfig, config_of_arm, config_of_dir   # noqa: E402
from locascope.training.CoAE.Losses import (ChannelScale, _vmf3_nll, agreement, cofig_point, dist_terms, koleo, koleo_by_source,   # noqa: E402
                                            point_terms, recon_dist, recon_pooled, recon_point, target_nce, view_nce,
                                            twin_nce, decoder_blocks, LOSSES, StepContext, StepLosses, TargetBank, adversary, koleo_global,
                                            step_losses)
from locascope.training.CoAE.Model import (CoaeOut, ConvNeXtDecoder, CrossToCls, Discriminator, LayerMix, SwinBlock, SwinEncoder,   # noqa: E402
                                           noisy)
from locascope.core.query_sim.pipeline import simulate_with_gt                      # noqa: E402
from locascope.training.CoAE.Sims import (AD_CHROMA4, Geometry, Shot, centre, draw_geometry, draw_shot, frame, parse_variant, probe_labs,   # noqa: E402
                                          read_px, sim_labs)
from locascope.training.CoAE.Supply import (BLANK, PROBE_ANCHOR, USABLE, Place, _morton, groups_of, make_board, plans_of, pool_runs,   # noqa: E402
                                            probe_plans, release_board, screened_pool)
from locascope.training.CoAE.Validate import (CoaeTileEncoder, anchor_spread, part_widths, pool_latent, probe_cell_scores, probe_scores,   # noqa: E402
                                              tile_parts, unit_feature)
from locascope.training.CoAE.train import (epoch_seed, keep_rows, load_checkpoint, log_row_kept, make_cfg, parse, places_read,   # noqa: E402
                                           val_row_kept)
from locascope.core.networks.Heads import resolve_encoder_layers                   # noqa: E402

DEV: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CFG: CoaeConfig = CoaeConfig()
_RESULTS: List[Tuple[str, Optional[BaseException]]] = []


def check(name: str, fn: Callable[[], str]) -> None:
    try:
        msg: str = fn()
        _RESULTS.append((name, None))
        print(f'PASS  {name}: {msg}', flush=True)
    except BaseException as exc:                                                     # noqa: BLE001
        _RESULTS.append((name, exc))
        print(f'FAIL  {name}: {type(exc).__name__}: {exc}', flush=True)
        traceback.print_exc()


def texture(side: int, seed: int = 0) -> np.ndarray:
    """A smooth random RGB texture uint8 [side, side, 3] with structure at many scales (no symmetry)."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(side)
    r = np.hypot(f[:, None], f[None, :]) + 1.0 / side
    spec = (rng.normal(size=(side, side)) + 1j * rng.normal(size=(side, side))) / r ** 1.2
    g = np.real(np.fft.ifft2(spec))
    g = (g - g.min()) / (g.max() - g.min()) * 200 + 25
    return np.repeat(g[..., None], 3, -1).astype(np.uint8)


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-12))


# ── layout ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

def t_a_layout() -> str:
    c = CFG.cofig
    assert n_channels(c) == 10, n_channels(c)                     # local phase 2 x 3, congruency 3, T 1
    assert n_dist_channels(c) == 16, n_dist_channels(c)           # local phase 2 x 4, congruency 6, T 2
    turns = [p.turn for p in layout(c)]
    assert turns == [1, 1, 2, 0], turns
    assert n_moments(c) == 29, n_moments(c)                       # 10 first moments + (6 + 6 + 6 + 1) products
    return '10 point channels, 16 dist channels, 29 moments; local phase turns its odd pair, congruency its doubled-angle pair'


# ── the alignment ───────────────────────────────────────────────────────────────────────────────────────────────────────

def _aligned(geo: Geometry, turn_sign: float = 1.0, distortion: Tuple[float, float] = (0.0, 0.0),
             ignore_lens: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The reference version and `geo`'s version of one texture (with the lens `distortion` when given), their maps warped to the
    reference grid (the lens left out of the warp when `ignore_lens`); and where both have data and `geo`'s input sees."""
    photo: int = CFG.data.photo_px
    big: int = read_px(CFG.data, photo)
    ref = texture(big)
    rng = random.Random(0)
    g0 = Geometry(1.0, 0.0, (0, 0))
    geo = replace(geo, k1=float(distortion[0]), k2=float(distortion[1]))
    im = frame(ref, geo, photo)
    canv = np.stack([frame(ref, g0, photo), im])
    grey = torch.as_tensor(canv.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32), device=DEV)
    obs = observations(signal_maps(grey, CFG.cofig), CFG.cofig)
    gc_ = grid_cells(CFG.cofig, CFG.data.shift_px)
    on, has = warp_to_grid(obs, [g0, replace(geo, k1=0.0, k2=0.0) if ignore_lens else geo], CFG.cofig, photo, gc_)
    if turn_sign != 1.0:                                           # the sampling right, the pairs turned by -turn instead of +turn
        on = _turn_pairs(on, torch.tensor([0.0, (turn_sign - 1.0) * geo.turn]), CFG.cofig)
    t: int = CFG.cofig.tile // CFG.cofig.cell
    y0 = (gc_ - t) // 2
    see = visible([geo], CFG.cofig, photo, DEV)[0]
    crop = lambda x: x[..., y0 + geo.shift[0] // 8:y0 + geo.shift[0] // 8 + t, y0 + geo.shift[1] // 8:y0 + geo.shift[1] // 8 + t]   # noqa: E731
    both = crop(has[0] & has[1]) & see
    return crop(on[0]), crop(on[1]), both


def t_b_alignment_warps_back() -> str:
    geo = Geometry(1.1, 30.0, (16, -24))
    a, b, m = _aligned(geo)
    out: List[str] = []
    for p in layout(CFG.cofig):
        c = corr(a[p.start:p.start + p.size][:, m], b[p.start:p.start + p.size][:, m])
        assert c > 0.6, f'{p.signal} warped back correlates {c:.2f} with the reference'
        out.append(f'{p.signal} {c:.2f}')
    return f'a version scaled 1.1, turned 30 deg, shifted (16, -24), warped back: ' + ', '.join(out)


def t_c_turn_direction() -> str:
    geo = Geometry(1.0, 30.0, (0, 0))
    a, b, m = _aligned(geo)
    a2, b2, _ = _aligned(geo, turn_sign=-1.0)
    odd = [p for p in layout(CFG.cofig) if p.turn]
    right = np.mean([corr(a[p.start + 1:p.start + 3][:, m], b[p.start + 1:p.start + 3][:, m]) for p in odd])
    wrong = np.mean([corr(a2[p.start + 1:p.start + 3][:, m], b2[p.start + 1:p.start + 3][:, m]) for p in odd])
    assert right > wrong + 0.1, f'the directed pairs: right turn {right:.2f}, wrong turn {wrong:.2f}'
    return f'the directed pairs turned the right way correlate {right:.2f}, the wrong way {wrong:.2f}'


def t_c2_lens_is_undone() -> str:
    """A version through a strong lens distortion (and a turn, a scale, a shift): warped back WITH the recorded lens it agrees with the
    reference better than warped back without it."""
    geo = Geometry(1.05, 3.0, (8, -8))
    a, b, m = _aligned(geo, distortion=(0.04, 0.0))
    a2, b2, m2 = _aligned(geo, distortion=(0.04, 0.0), ignore_lens=True)
    with_lens = corr(a[:, m], b[:, m])
    without = corr(a2[:, m2], b2[:, m2])
    assert with_lens > without, (with_lens, without)
    return f'the version with lens k1 0.04, warped back with the lens {with_lens:.3f}, without it {without:.3f}'


def t_d_visible() -> str:
    c = CFG.cofig
    zoom = float(visible([Geometry(1.2, 0.0, (0, 0))], c, CFG.data.photo_px, DEV).float().mean())
    wide = float(visible([Geometry(0.8, 0.0, (0, 0))], c, CFG.data.photo_px, DEV).float().mean())
    turn = float(visible([Geometry(1.0, 5.0, (0, 0))], c, CFG.data.photo_px, DEV).float().mean())
    assert 0.6 < zoom < 0.75, zoom
    assert wide == 1.0, wide
    assert 0.85 < turn < 1.0, turn
    return f'visible share: zoom 1.2 {zoom:.2f}, zoom 0.8 {wide:.2f}, turn 5 deg {turn:.2f}'


def t_e_place_targets() -> str:
    photo: int = CFG.data.photo_px
    big: int = read_px(CFG.data, photo)
    rng = random.Random(1)
    geos: List[Geometry] = []
    canv: List[np.ndarray] = []
    for pl in range(2):
        ref = texture(big, seed=pl + 3)
        for k in range(4):
            g = Geometry(1.0, 0.0, (2.5, -1.7)) if k == 0 else Geometry(1 + 0.15 * (k - 2), 3.0 * (k - 2), (7.3 * k, -5.6 * k))   # shifts of any real value
            geos.append(g)
            canv.append(frame(ref, g, photo))
    grey = torch.as_tensor(np.stack(canv).astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32), device=DEV)
    obs = observations(signal_maps(grey, CFG.cofig), CFG.cofig)
    tg: Targets = place_targets(obs, geos, CFG.cofig, 4, photo, CFG.data.shift_px)
    t = CFG.cofig.tile // CFG.cofig.cell
    assert tg.mean.shape == (8, 10, t, t) and tg.obs.shape == (8, 4, 10, t, t) and tg.valid.shape == (8, t, t)
    assert torch.allclose(tg.mean[0], tg.obs[0].mean(0), atol=1e-5), 'the mean is the versions averaged'
    share = float(tg.valid.float().mean())
    assert 0.5 < share <= 1.0, share
    # the 'input' frame: a version's target read at its own input's cells. The reference version (a shift only) is the same in both
    # frames; a scaled, turned version's target is nearer what its own input shows (its own maps' centre tile) than the scan-frame one
    c_in = replace(CFG.cofig, frame='input')
    tg_in: Targets = place_targets(obs, geos, c_in, 4, photo, CFG.data.shift_px)
    assert tg_in.mean.shape == tg.mean.shape and tg_in.obs.shape == tg.obs.shape
    both = tg.valid[0] & tg_in.valid[0]
    assert float((tg.mean[0] - tg_in.mean[0])[:, both].abs().max()) < 1e-4, 'the reference version: the same target in both frames'
    c0 = (obs.shape[-1] - t) // 2
    own = obs[:, :, c0:c0 + t, c0:c0 + t]                                   # each version's own maps at its input tile
    lp = slice(0, 3)                                                       # local phase, 8 cycles

    def cos(a: torch.Tensor, b_: torch.Tensor, m: torch.Tensor) -> float:
        return float(TF.cosine_similarity(a[:, m].flatten(), b_[:, m].flatten(), dim=0))
    far = [i for i, g in enumerate(geos) if abs(g.scale - 1) > 0.1]
    cs_scan = float(np.mean([cos(tg.mean[i, lp], own[i, lp], tg.valid[i] & tg_in.valid[i]) for i in far]))
    cs_in = float(np.mean([cos(tg_in.mean[i, lp], own[i, lp], tg.valid[i] & tg_in.valid[i]) for i in far]))
    assert cs_in > cs_scan, (cs_in, cs_scan)
    share_in = float(tg_in.valid.float().mean())
    # the input frame's points are `_photo_points` undone, the stage shift and the lens too: sent forward again they are the
    # input tile's own cell centres
    g_all = replace(geos[3], stage=(3.0, -2.0), k1=0.03, k2=-0.01)
    back = _photo_points([g_all], target_points(g_all, c_in, photo, DEV), photo, DEV)[0]
    k_ = (torch.arange(t, device=DEV, dtype=torch.float32) + 0.5) * CFG.cofig.cell - CFG.cofig.tile / 2 + photo / 2
    yy, xx = torch.meshgrid(k_, k_, indexing='ij')
    err = float((back - torch.stack([xx.flatten(), yy.flatten()], 1)).abs().max())
    assert err < 0.05, f'the input frame\'s points do not come back to the input tile: {err:.3f} px'
    return (f'shapes right, the mean is the versions\' average, {share:.2f} of the cells valid (input frame {share_in:.2f}); local phase 8 of '
            f'the 15 % scaled versions against their own input\'s map: cos {cs_scan:.2f} in the scan frame, {cs_in:.2f} in the input frame')


# ── the network's parts ─────────────────────────────────────────────────────────────────────────────────────────────────

def t_f_layer_mix_caps() -> str:
    mix = LayerMix(4, 8, (0.1, 0.2, 0.3, 0.4), (0.05, 1.0, 1.0, 1.0))
    with torch.no_grad():
        mix.logits.copy_(torch.tensor([5.0, 0.0, 0.0, 0.0]))       # the first layer would take ~0.95
    w = mix.weights()
    assert abs(float(w.sum()) - 1) < 1e-5 and float(w[0]) <= 0.05 + 1e-6, w
    out = mix([torch.randn(2, 4, 4, 8, requires_grad=True) for _ in range(4)])
    out.sum().backward()
    assert mix.logits.grad is not None
    return f'weights {np.round(w.detach().numpy(), 3).tolist()} (the first capped), gradients reach the logits'


def t_g_shapes() -> str:
    m = replace(CFG.model, latent_dim=64, swin_dim=32, swin_heads=4, decoder_dim=(32, 16, 8), cls_dim=32, cross_heads=4)
    dec = ConvNeXtDecoder(m, 10)
    lat14 = SwinEncoder(48, m, 14)(torch.randn(2, 14, 14, 48))
    y28 = dec(lat14)
    assert lat14.shape == (2, 7, 7, 64) and y28.shape == (2, 10, 28, 28), (lat14.shape, y28.shape)
    lat16 = SwinEncoder(48, m, 16)(torch.randn(2, 16, 16, 48))
    y32 = dec(lat16)
    assert lat16.shape == (2, 8, 8, 64) and y32.shape == (2, 10, 32, 32), (lat16.shape, y32.shape)
    assert _raises(lambda: SwinEncoder(48, m, 14)(torch.randn(2, 16, 16, 48))), 'tokens of another grid than the encoder was built for'
    assert _raises(lambda: SwinEncoder(48, m, 15)), 'an odd token grid has no 2 x 2 merging'
    return 'Swin 14x14x48 -> 7x7x64, ConvNeXt -> 10x28x28; 16x16x48 -> 8x8x64 -> 10x32x32; a wrong or odd grid raises'


class _Identity(nn.Module):
    """A window attention that returns its input: what is left of a SwinBlock is the roll out, the roll back and the residuals."""

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        return x


def t_ga_shift_odd_window() -> str:
    torch.manual_seed(0)
    for window, grid in ((7, 14), (8, 16)):
        blk = SwinBlock(8, window, 2, shift=True)
        blk.attn = _Identity()
        with torch.no_grad():
            blk.mlp[-1].weight.zero_()
            blk.mlp[-1].bias.zero_()
        x = torch.randn(2, grid, grid, 8)
        err = float((blk(x) - (x + blk.n1(x))).abs().max())
        assert err < 1e-6, (window, grid, err)
    return 'a shifted SwinBlock with the identity attention returns x + n1(x): the roll out and back agree (window 7 on 14 and 8 on 16)'


def t_gb_windows_exchange() -> str:
    torch.manual_seed(0)
    out: List[str] = []
    for grid in (14, 16):
        s = grid // 2
        for depth in (1, 2):
            m = replace(CFG.model, latent_dim=16, swin_dim=32, swin_heads=4, swin_depth=depth)
            enc = SwinEncoder(48, m, grid).eval()
            x = torch.randn(1, grid, grid, 48)
            x2 = x.clone()
            x2[:, 0, 0] += 1.0
            with torch.no_grad():
                d = (enc(x2) - enc(x)).abs().amax(-1)[0]                             # [s, s]: how far each latent cell moved
            assert float(d[s - 1, s - 1]) < 1e-6, (grid, depth, float(d[s - 1, s - 1]))
            assert float(d[0, 0]) > 1e-4, (grid, depth, float(d[0, 0]))
            if depth == 2:
                assert float(d[s // 2 + 1, s // 2 + 1]) > 1e-6, (grid, float(d[s // 2 + 1, s // 2 + 1]))
            out.append(f'g {grid} depth {depth}: cell (0,0) {float(d[0, 0]):.1e}, ({s // 2 + 1},{s // 2 + 1}) {float(d[s // 2 + 1, s // 2 + 1]):.1e}, ({s - 1},{s - 1}) {float(d[s - 1, s - 1]):.1e}')
    return ('a token at (0, 0) reaches the neighbouring window\'s near half only through the shifted block (depth 2); the far half of the '
            'opposite window is out of reach of one pair of blocks (expected): ' + '; '.join(out))


def t_h_cls_carries_no_q() -> str:
    cross = CrossToCls(16, 24, 32, 4)
    lat = torch.randn(1, 1, 1, 24).expand(2, 4, 4, 24).contiguous()   # every cell the same: whatever q chooses, the same values
    q = torch.randn(2, 16)
    out = cross(q, lat)
    assert torch.allclose(out[0], out[1], atol=1e-5), 'cls\' depends on q beyond its choice of cells'
    return 'with every latent cell alike, cls\' is the same for any q'


def t_i_adversary_confusion() -> str:
    from locascope.training.CoAE.Losses import _confusion
    d = Discriminator(8, 16, 4)
    x = torch.randn(6, 8, requires_grad=True)
    TF.cross_entropy(d.fit(x), torch.tensor([0, 1, 2, 3, 0, 1])).backward()
    assert x.grad is None and all(p.grad is not None for p in d.parameters()), 'fit: only the classifier learns'
    d.zero_grad()
    _confusion(d.judge(x)).backward()
    assert x.grad is not None and all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in d.parameters()), 'judge: only the network'
    uniform = float(_confusion(torch.zeros(4, 4)))
    swapped = float(_confusion(torch.tensor([[9.0, 0, 0, 0], [0, 9.0, 0, 0], [0, 0, 9.0, 0], [0, 0, 0, 9.0]]).roll(1, 0)))
    assert abs(uniform) < 1e-6 and swapped > 2.0, (uniform, swapped)
    return (f'the classifier learns on detached features, the network through frozen weights; confusion 0 when the classifier '
            f'cannot tell, {swapped:.1f} when it is sure of a swapped source (swapping is not rewarded)')


# ── the losses ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def _targets(b: int = 4, v: int = 4) -> Targets:
    t = 32
    obs = torch.randn(b, v, 10, t, t)
    return Targets(obs.mean(1), obs, torch.ones(b, t, t, dtype=torch.bool))


def t_j_losses() -> str:
    tg = _targets()
    sc = ChannelScale(10)
    sc.update(tg.mean, tg.valid)
    assert float(recon_point(tg.mean.clone(), tg, sc)) < 1e-10
    off = tg.mean + torch.randn_like(tg.mean)
    terms = point_terms(off, tg, sc, CFG.cofig)
    sizes = [p.size for p in layout(CFG.cofig)]
    whole = sum(float(t_) * s_ for t_, s_ in zip(terms, sizes)) / sum(sizes)
    assert len(terms) == 4 and abs(whole - float(recon_point(off, tg, sc))) < 1e-4, (whole, float(recon_point(off, tg, sc)))
    ag = agreement(tg.mean, tg.mean, tg.valid, sc, CFG.cofig)
    assert list(ag) == part_names(CFG.cofig) == ['lp8', 'lp24', 'pc', 'tw'] and all(v['cos'] > 0.999 and v['mse'] < 1e-10 for v in ag.values()), ag
    assert agreement(off, tg.mean, tg.valid, sc, CFG.cofig)['pc']['cos'] < 0.9
    pm_ = torch.randn(4, 29, 4, 4)                                  # a place descriptor [B, M, c, c] (CoFigure.shared_moments)
    pw = torch.ones(4, 4, 4)
    psc: torch.Tensor = pm_.permute(1, 0, 2, 3).flatten(1).std(dim=1)                  # the sd a channel [M]
    assert float(recon_pooled(pm_.clone(), pm_, pw, psc)) < 1e-10 and float(recon_pooled(pm_ + 1, pm_, pw, psc)) > 0
    d_pred = torch.randn(4, 16, 32, 32)
    assert cofig_point(d_pred, CFG.cofig, sc, 'dist').shape == (4, 10, 32, 32) and cofig_point(off, CFG.cofig, sc, 'point') is off
    x = TF.normalize(torch.randn(2, 3, 4, 4), dim=1)
    good = float(_vmf3_nll(x, torch.full((2, 4, 4), 3.0), x).mean())
    bad = float(_vmf3_nll(-x, torch.full((2, 4, 4), 3.0), x).mean())
    assert good < bad, (good, bad)
    pred = torch.randn(4, 16, 32, 32, requires_grad=True)
    l_d = recon_dist(pred, tg, sc, CFG.cofig, 0.5)
    l_d.backward()
    assert torch.isfinite(l_d) and pred.grad is not None and torch.isfinite(pred.grad).all()
    d_t = dist_terms(pred, tg, sc, CFG.cofig, 0.5)
    assert len(d_t) == 4 and abs(float(torch.stack(d_t).mean()) - float(l_d)) < 1e-5, 'the dist loss is its signals\' mean'
    cls = torch.randn(8, 16, requires_grad=True)
    l_k = koleo_by_source(cls, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
    l_k.backward()
    assert cls.grad is not None and torch.isfinite(cls.grad).all()
    spread = float(koleo(TF.normalize(torch.eye(8), dim=1)))
    close = float(koleo(TF.normalize(torch.ones(8, 8) + 0.01 * torch.randn(8, 8), dim=1)))
    assert spread < close, (spread, close)
    return (f'point MSE 0 at the target; vMF NLL {good:.2f} at the mean < {bad:.2f} opposite; dist NLL and KoLeo backpropagate; '
            f'KoLeo {spread:.2f} spread < {close:.2f} bunched')


def t_j2_target_nce() -> str:
    torch.manual_seed(0)
    p, k, d, tau = 32, 32, 512, 0.1
    keys: torch.Tensor = torch.randn(p, d)
    bank: torch.Tensor = torch.randn(k, d)
    place: torch.Tensor = torch.arange(p)
    w: torch.Tensor = torch.ones(p)
    allow: torch.Tensor = torch.ones(p, k, dtype=torch.bool)
    near: float = float(target_nce(keys.clone(), place, w, keys, bank, allow, tau))
    assert near < 0.05, near                                                          # (a) the head is its own key: the loss is near 0
    rand: float = float(target_nce(torch.randn(p, d), place, w, keys, bank, allow, tau))
    assert abs(rand - math.log(p + k)) < 0.3, (rand, math.log(p + k))                 # (b) a random head: near log(keys + allowed bank rows)
    head0: torch.Tensor = keys.clone() + torch.randn(p, d)
    assert torch.allclose(target_nce(head0, place, w, keys, bank, torch.zeros(p, k, dtype=torch.bool), tau),
                          target_nce(head0, place, w, keys, bank[:0], torch.zeros(p, 0, dtype=torch.bool), tau), atol=1e-6)   # (c) nothing allowed = no bank
    e0: torch.Tensor = torch.zeros(p)
    e0[0] = 1.0                                                                       # only view 0 counts
    twin: torch.Tensor = torch.cat([bank[:-1], keys[:1]])                             # the bank holds view 0's own key (as a negative)
    ok: torch.Tensor = torch.ones(p, k, dtype=torch.bool)
    off: torch.Tensor = ok.clone()
    off[0, -1] = False
    base: float = float(target_nce(keys.clone(), place, e0, keys, twin, off, tau))
    hit: float = float(target_nce(keys.clone(), place, e0, keys, twin, ok, tau))
    assert hit > base + 0.5, (base, hit)                                              # (d) the mask works: an allowed copy of the positive costs ~ log 2
    h: torch.Tensor = torch.randn(p, d, requires_grad=True)
    loss: torch.Tensor = target_nce(h, place, w, keys, bank, allow, tau)
    loss.backward()
    assert keys.grad is None and bank.grad is None and h.grad is not None and bool(h.grad.abs().sum() > 0) and torch.isfinite(h.grad).all()   # (e)
    return (f'own key {near:.3f} ~ 0; random head {rand:.2f} ~ log({p + k}) = {math.log(p + k):.2f}; nothing allowed = no bank; an allowed copy of the '
            f'positive {base:.2f} -> {hit:.2f}; the gradient reaches the head only')


def t_j3_view_nce() -> str:
    torch.manual_seed(0)
    tau: float = 0.1
    place: torch.Tensor = torch.arange(4).repeat_interleave(3)                        # 4 places x 3 views
    cls: torch.Tensor = torch.eye(16)[place]                                          # a place's views alike, the places orthogonal
    w: torch.Tensor = torch.ones(len(place))
    floor: float = float(view_nce(cls, place, w, tau))
    assert abs(floor - math.log(2)) < 0.01, floor        # (a) the least: every positive shares the mass, log(positives a view = 2)
    b: int = 64
    pl: torch.Tensor = torch.arange(b // 2).repeat_interleave(2)
    rand: float = float(view_nce(torch.randn(b, 512), pl, torch.ones(b), tau))
    assert abs(rand - math.log(b - 1)) < 0.3, (rand, math.log(b - 1))                 # (b) random views: near log(B - 1)
    assert _raises(lambda: view_nce(torch.randn(3, 8), torch.tensor([0, 0, 1]), torch.ones(3), tau)), 'a place of one view'   # (c)
    x: torch.Tensor = torch.randn(b, 32)
    assert abs(float(view_nce(x, pl, torch.ones(b), tau)) - float(view_nce(7.0 * x, pl, torch.ones(b), tau))) < 1e-5   # (d) the scale of cls does not matter
    return f'alike views, orthogonal places {floor:.3f} = log 2 (the floor of 2 positives); random {rand:.2f} ~ log({b - 1}) = {math.log(b - 1):.2f}; a lone view raises; scale-free'


def t_j4_twin_nce() -> str:
    torch.manual_seed(0)
    tau: float = 0.1
    groups: int = 8                                                                   # 4 places x 2 slots
    s: int = 3                                                                        # sources a group
    b: int = groups * s
    c: int = 49
    w: torch.Tensor = torch.ones(b)
    same: torch.Tensor = torch.randn(groups, c, 16).repeat_interleave(s, dim=0)       # a group's sources alike, the groups apart
    a: float = float(twin_nce(same, s, w, 8, tau))
    assert abs(a - math.log(s - 1)) < 0.05, (a, math.log(s - 1))                       # (a) the least: log(sources - 1)
    code: torch.Tensor = torch.randn(1, c, 16).expand(b, c, 16)                       # a code of the cell position alone
    sc: float = float(twin_nce(code, s, w, 8, tau))
    assert abs(sc - math.log(b - 1)) < 0.05, (sc, math.log(b - 1))                     # (b) the position shortcut cannot tell the views apart
    rand: float = float(twin_nce(torch.randn(b, c, 512), s, w, 8, tau))
    assert abs(rand - math.log(b - 1)) < 0.3, (rand, math.log(b - 1))                 # (c) independent random vectors
    assert _raises(lambda: twin_nce(same, 1, w, 8, tau)), 'sources 1'                  # (d)
    assert _raises(lambda: twin_nce(same[:23], s, w[:23], 8, tau)), 'a batch that is not a multiple of sources'
    assert _raises(lambda: twin_nce(same, s, w, 50, tau)), 'cells 50 of 49'
    v: torch.Tensor = torch.randn(b, c, 16, requires_grad=True)
    twin_nce(v, s, w, 8, tau).backward()
    assert v.grad is not None and bool(v.grad.abs().sum() > 0) and torch.isfinite(v.grad).all()   # (e)
    return (f'sources alike {a:.3f} ~ log({s - 1}) = {math.log(s - 1):.3f}; a code of the position alone {sc:.2f} = log({b - 1}) = {math.log(b - 1):.2f}; '
            f'random {rand:.2f}; bad sources / batch / cells raise; the gradient reaches the vectors')


def t_j5_decoder_blocks() -> str:
    cofig: torch.Tensor = torch.randn(2, 10, 28, 28)
    blocks: torch.Tensor = decoder_blocks(cofig, 7)
    assert blocks.shape == (2, 49, 160), blocks.shape
    for i in range(7):
        for j in range(7):
            assert torch.equal(blocks[:, 7 * i + j], cofig[:, :, 4 * i:4 * i + 4, 4 * j:4 * j + 4].flatten(1)), (i, j)
    assert _raises(lambda: decoder_blocks(cofig, 5)), 'a side that does not divide'
    return 'the decoder\'s [2, 10, 28, 28] -> [2, 49, 160]; block 7i + j = the cells 4i.. x 4j.. of the output, value by value'


def _fake_step(cfg: CoaeConfig, shares_equal: bool) -> StepContext:
    """A seeded fake training step (4 places x 6 views, 3 sources, latent 7 x 7, Co-figure 28 x 28): the same call gives the same step,
    the bank (only when cfg turns the target contrast on) filled in the same way."""
    torch.manual_seed(1234)
    p: int = 4
    s: int = 3
    v: int = 6
    b: int = p * v
    t: int = 28
    c: int = cfg.head.pool_cells
    m: int = n_moments(cfg.cofig)
    n_out: int = 16 if cfg.head.output == 'dist' else 10
    w_v: torch.Tensor = torch.rand(b) + 0.5
    obs: torch.Tensor = torch.randn(b, 1, 10, t, t)
    tg: Targets = Targets(obs[:, 0], obs, torch.ones(b, t, t, dtype=torch.bool), w_v)
    scale: ChannelScale = ChannelScale(10)
    scale.update(tg.mean, tg.valid)
    net: SimpleNamespace = SimpleNamespace(pool_mu=torch.randn(m), pool_sd=torch.rand(m) + 0.5, latent_side=7, adv_lambda=0.4)
    out: CoaeOut = CoaeOut(latent=torch.randn(b, 7, 7, 8), cls=torch.randn(b, 16), cofig=torch.randn(b, n_out, t, t),
                           pooled=torch.randn(b, m, c, c), cls_fm=torch.randn(b, 16), avg_fm=torch.randn(b, 16),
                           src_cells=torch.randn(b, 49, s), src_cls=torch.randn(b, s), conf_cells=torch.randn(b, 49, s),
                           conf_cls=torch.randn(b, s))
    p_mean: torch.Tensor = torch.randn(p, 1, m, c, c).expand(p, v, m, c, c).reshape(b, m, c, c)
    batch: SimpleNamespace = SimpleNamespace(place=torch.arange(p).repeat_interleave(v), slides=['a'] * p, sources=('ref', 'x', 'y'),
                                             index=[3, 0, 5, 2])
    bank: Optional[TargetBank] = None
    if cfg.head.w_nce > 0:
        bank = TargetBank([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1], (m, c), torch.device('cpu'))
        bank.desc = torch.randn_like(bank.desc)
        bank.ok = torch.arange(16) % 3 != 0
    return StepContext(cfg, net, out, tg, p_mean, torch.ones(b, c, c) * w_v.view(-1, 1, 1), scale, batch, batch.place, w_v,
                       torch.arange(v).remainder(s).repeat(p), shares_equal, bank)


def t_j6_loss_registry() -> str:
    from locascope.training.CoAE.Config import LOSS_TERMS                                        # noqa: PLC0415
    order: List[str] = ['recon', 'pool', 'koleo', 'adv', 'nce', 'vnce', 'hnce', 'dnce', 'lnce']
    assert list(LOSSES) == order, list(LOSSES)                                                     # the order the trainer sums them
    assert all(t.key in LOSSES for t in LOSS_TERMS), 'every named loss is registered'
    assert all(hasattr(CFG.head, f'w_{k}') and term.key == k for k, term in LOSSES.items()), 'every loss has its weight field'
    assert all(term.output in ('cofig', 'head', 'cls', 'latent', 'source-classifier') for term in LOSSES.values())
    assert [k for k, term in LOSSES.items() if term.always] == ['recon', 'pool']
    base: CoaeConfig = CoaeConfig()
    cfgs: List[Tuple[str, CoaeConfig]] = [('default', base)] + [(a, config_of_arm(a)) for a in
        ('RP3', 'RN3', 'RPV2', 'RH2', 'DP2', 'DH2', 'LH2', 'Rx3', 'xV2.weightV1', 'xP2', 'RP3.vdist')]
    cfgs += [('default+koleo+adv', replace(base, head=replace(base.head, w_koleo=0.3, w_adv=0.7))),
             ('RN3+koleo+adv', replace(config_of_arm('RN3'), head=replace(config_of_arm('RN3').head, w_koleo=0.3, w_adv=0.7)))]
    done: List[str] = []
    for name, cfg in cfgs:
        for shares_equal in (True, False):
            torch.manual_seed(77)                                                                  # twin_nce draws its cell positions
            got: StepLosses = step_losses(_fake_step(cfg, shares_equal))
            on: List[str] = [k for k, term in LOSSES.items() if term.always or float(getattr(cfg.head, f'w_{k}')) > 0]
            assert [n for n, _ in got.weighted] == [LOSSES[k].grad_name for k in on], (name, [n for n, _ in got.weighted])
            if cfg.head.w_adv == 0:                                                                # the adversary's gradient-line term is its confusion alone
                total: torch.Tensor = got.weighted[0][1]
                for _, part in got.weighted[1:]:
                    total = total + part                                                           # the order the trainer sums them
                assert torch.equal(got.loss, total), (name, float(got.loss), float(total))         # the total is the sum of the weighted terms, nothing else
            assert all(LOSSES[k].column in got.logged for k in on if LOSSES[k].column is not None), (name, sorted(got.logged))   # every computed loss logs its value
        done.append(name)
    ctx: StepContext = _fake_step(base, True)
    assert abs(LOSSES['vnce'].floor(ctx) - math.log(5)) < 1e-12 and abs(LOSSES['hnce'].floor(ctx) - math.log(5)) < 1e-12
    assert abs(LOSSES['dnce'].floor(ctx) - math.log(2)) < 1e-12 and abs(LOSSES['lnce'].floor(ctx) - math.log(2)) < 1e-12
    assert LOSSES['recon'].floor(ctx) == 0.0 and LOSSES['pool'].floor(ctx) == 0.0 and LOSSES['nce'].floor(ctx) == 0.0
    assert LOSSES['koleo'].floor(ctx) is None and LOSSES['adv'].floor(ctx) is None
    names: List[str] = [n for n, _ in step_losses(_fake_step(config_of_arm('RN3'), True)).weighted]
    assert names == ['rec', 'pool', 'nce'], names
    names = [n for n, _ in step_losses(_fake_step(config_of_arm('xV2.weightV1'), True)).weighted]
    assert names == ['rec', 'pool', 'vnce'], names                                                  # an always-loss stays at weight 0
    names = [n for n, _ in step_losses(_fake_step(config_of_arm('xP2'), True)).weighted]
    assert names == ['rec', 'pool'], names                                                          # xP2: the decoder stays in the graph at weight 0
    return (f'{len(done)} weight sets x shares equal / not: the weighted terms are the losses on (names, order), their sum is the total loss '
            f'(torch.equal; not with the adversary on), each loss logs its column; order {order}; floors log(V - 1), log(S - 1), 0')


def t_k_koleo_global_alone() -> str:
    from locascope.training.CoAE.Losses import koleo_global
    cls = torch.randn(12, 8)
    src = torch.tensor([0, 1, 2] * 4)
    assert torch.allclose(koleo_global(cls, src), koleo_by_source(cls, src)), 'without a process group it is koleo_by_source'
    return 'without a process group the gathered KoLeo is the per-source KoLeo (the 2-process run is step 3 of TestCoae)'


# ── the perturbation levels, the views, the Co-figure's fields and moments ──────────────────────────────────────────────────

def _grey(n: int, seed: int = 0) -> torch.Tensor:
    """n distinct 640 px textures as grey canvases [n, 640, 640] (float 0-255) on DEV."""
    return torch.as_tensor(np.stack([texture(640, seed + i)[..., 0] for i in range(n)]).astype(np.float32), device=DEV)


def _raises(fn: Callable[[], object]) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def t_l_levels() -> str:
    rng: random.Random = random.Random(0)
    assert draw_geometry(rng, LEVELS['0']) == Geometry()
    out: List[str] = []
    for name in ('light', 'large'):
        lv = LEVELS[name]
        gs: List[Geometry] = [draw_geometry(rng, lv) for _ in range(500)]
        for g in gs:
            assert abs(g.scale - 1) <= lv.scale and abs(g.turn) <= lv.turn_deg and abs(g.k1) <= lv.k1, (name, g)
            assert abs(g.shift[0]) <= lv.shift_px and abs(g.shift[1]) <= lv.shift_px, (name, g)
        assert len(set(gs)) > 1 and any(g.k1 != 0.0 for g in gs), name
        out.append(f'{name}: |shift| max {max(max(abs(g.shift[0]), abs(g.shift[1])) for g in gs):.1f} px, |turn| max '
                   f'{max(abs(g.turn) for g in gs):.1f}, |k1| max {max(abs(g.k1) for g in gs):.3f}')
    return 'level 0 draws Geometry(); 500 draws each within range, varied, k1 drawn -- ' + '; '.join(out)


def t_m_frame_identity() -> str:
    photo: int = CFG.data.photo_px
    ref = texture(read_px(CFG.data, photo))
    assert np.array_equal(frame(ref, Geometry(), photo), centre(ref, photo))
    return f'frame at the identity is the scan\'s own centre {photo} px, pixel for pixel'


def t_n_versions() -> str:
    photo: int = CFG.data.photo_px
    ref = texture(read_px(CFG.data, photo))
    sources = ('ref', 'bench-real')
    labs = sim_labs()
    for lab in labs.values():
        assert tuple(lab.base.distortion_k1_range) == (0.0, 0.0) and lab.base.stage_shift_max == 0, lab
    geos, vs = plans_of(5, CFG.data, sources, labs)
    assert len(vs) == 4 and [k for _, k, _, _ in vs] == [0, 1, 0, 1], [k for _, k, _, _ in vs]
    assert len(geos) == groups_of(CFG.data, len(sources)) == 2, len(geos)
    assert vs[0][0] == Geometry() and vs[1][0] == Geometry(), 'slot 0 is canonical'
    assert vs[0][2] == vs[1][2] == 0 and vs[2][2] == vs[3][2] == 1, 'a group\'s views share its frame'
    assert vs[2][0] == vs[3][0] and vs[2][0] != Geometry(), 'slot 1: twins, one perturbation (k1 too)'
    for _, _, g, _ in vs:
        assert 0 <= g < len(geos)
    for geo, _, g, _ in vs:
        assert geos[g] == geo, 'a view\'s geometry is its frame\'s'
    assert vs[0][3] is None and vs[2][3] is None and isinstance(vs[1][3], Shot) and isinstance(vs[3][3], Shot), 'the ref has no shot, the simulated source has'
    ind_geos, ind = plans_of(5, replace(CFG.data, mechanism='ind'), sources, labs)
    assert len(ind) == 4 and len(ind_geos) == groups_of(replace(CFG.data, mechanism='ind'), len(sources)) == 3
    assert ind[2][0] != ind[3][0] and ind[2][2] != ind[3][2], 'ind: every source its own perturbation and frame'
    assert ind[0][0] == Geometry() and ind[1][0] == Geometry()
    zero_geos, zero = plans_of(5, replace(CFG.data, level='0'), sources, labs)
    assert len(zero) == 2 and len(zero_geos) == 1 and all(g == Geometry() for g, _, _, _ in zero)
    assert (geos, vs) == plans_of(5, CFG.data, sources, labs), 'the same seed, the same plan'
    return ('mix/large: 4 views on 2 frames (slot 0 canonical, slot 1 twins with one shared geometry), the ref without a shot; ind: 3 frames, slot 1 '
            'sources differ; level 0: 2 views on 1 frame; labs photometric only; deterministic')


def t_o_field_maps() -> str:
    c = CFG.cofig
    grey = _grey(2)
    m = field_maps(grey, c, 224)
    assert m.shape == (2, 10, 28, 28), m.shape
    hand = TF.avg_pool2d(signal_pixels(grey, c)[..., 208:432, 208:432], 8)
    assert torch.allclose(m, hand, atol=1e-5), float((m - hand).abs().max())
    c32 = replace(c, out_cells=32)                                                   # uni2's field: 256 px = 32 cells of 8 px (256 % 28 != 0)
    m256 = field_maps(grey, c32, 256)
    old = observations(signal_maps(grey, c), c)[..., 24:56, 24:56]                   # the same 8 px cell grid, the central 32 x 32 cells
    err = float((m256[:, :9] - old[:, :9]).abs().max())
    assert err < 1e-5, err
    assert _raises(lambda: field_maps(grey, c, 250)), 'a field that is not a multiple of out_cells'
    return f'(2, 10, 28, 28) at 224 px, 8 px a cell; equals the hand-made crop and average; at 256 px the phase and congruency channels match the old central cells (max diff {err:.1e})'


def t_p_view_targets() -> str:
    own = torch.randn(24, 10, 32, 32)                                                # 3 places x 2 slots x 4 sources
    x = own.view(3, 2, 4, 10, 32, 32)
    tg = view_targets(own, 2, 4, True, 'vref')
    assert tg.obs.shape == (24, 1, 10, 32, 32) and bool(tg.valid.all())
    assert torch.equal(tg.mean.view(3, 2, 4, 10, 32, 32), x[:, :, :1].expand_as(x)), 'twins + vref: the slot\'s ref map'
    assert torch.equal(tg.obs[:, 0], tg.mean), 'vref: the one observation is the target itself'
    tg = view_targets(own, 2, 4, True, 'vavg')
    assert tg.obs.shape == (24, 4, 10, 32, 32)
    assert torch.equal(tg.obs.view(3, 2, 4, 4, 10, 32, 32)[:, :, 1], x), 'twins: the observations are the slot\'s four maps'
    assert torch.allclose(tg.mean.view(3, 2, 4, 10, 32, 32), x.mean(2, keepdim=True).expand_as(x), atol=1e-6)
    tg = view_targets(own, 2, 4, False, 'vref')
    assert tg.obs.shape == (24, 1, 10, 32, 32) and bool(tg.valid.all())
    assert torch.equal(tg.mean.view(3, 8, 10, 32, 32), own.view(3, 8, 10, 32, 32)[:, :1].expand(3, 8, 10, 32, 32)), 'not twins + vref: (slot 0, ref)'
    assert torch.equal(tg.obs[:, 0], tg.mean), 'vref: the one observation is the target itself'
    tg = view_targets(own, 2, 4, False, 'vavg')
    assert tg.obs.shape == (24, 8, 10, 32, 32)
    assert torch.allclose(tg.mean.view(3, 8, 10, 32, 32), own.view(3, 8, 10, 32, 32).mean(1, keepdim=True).expand(3, 8, 10, 32, 32), atol=1e-6)
    return 'twins share the slot\'s ref map / mean, others the place\'s (slot 0, ref) map / mean of all 8; obs (24, 1, ...) under vref, (24, 4, ...) and (24, 8, ...) under vavg; valid all True'


def t_x_targets_of_views() -> str:
    c = CFG.cofig
    grey = _grey(12)                                                                 # 2 places x 2 slots x 3 sources, each canvas its own
    for twins in (True, False):
        full = view_targets(field_maps(grey, c, 224), 2, 3, twins, 'vref')
        part = targets_of_views(grey, c, 224, 2, 3, twins, 'vref', 3)
        err = float((part.mean - full.mean).abs().max())
        assert err < 1e-5, (twins, err)
        assert part.obs.shape == (12, 1, n_channels(c), c.out_cells, c.out_cells) and bool(part.valid.all()) and torch.equal(part.obs[:, 0], part.mean)
        ref_only = grey.view(2, 2, 3, 640, 640)[:, :, :1].reshape(-1, 640, 640)       # each slot's ref canvas alone (kept 1)
        assert torch.equal(targets_of_views(ref_only, c, 224, 2, 3, twins, 'vref', 1).mean, part.mean), f'kept 1 reads the same canvases (twins {twins})'
    for twins in (True, False):
        full = view_targets(field_maps(grey, c, 224), 2, 3, twins, 'vavg')
        part = targets_of_views(grey, c, 224, 2, 3, twins, 'vavg', 3)
        assert torch.equal(part.mean, full.mean) and torch.equal(part.obs, full.obs), f'vavg is view_targets of the full maps (twins {twins})'
        assert _raises(lambda: targets_of_views(ref_only, c, 224, 2, 3, twins, 'vavg', 1)), 'vavg needs every source\'s canvas'
    return 'vref: the maps of the ref canvases alone equal the full maps\' target (twins and not, < 1e-5), obs is the target; vavg: identical to view_targets'


def t_y2_probe_plans() -> str:
    labs = probe_labs(ProbeConfig().ood)
    geos, views = probe_plans(7, labs)
    k_n: int = 4 + 2 * len(labs)
    kinds: List[str] = [k for k, _, _ in views]
    assert len(views) == k_n and kinds[0] == PROBE_ANCHOR and len(set(kinds)) == k_n, kinds
    assert len(geos) == 4 + len(labs), len(geos)
    assert geos[0] == Geometry() and geos[1] == Geometry(shift=(0.5, 0.5)), 'the anchor and sib:interp frames'
    for kind, g, shot in views:
        assert 0 <= g < len(geos), (kind, g)
        if kind.startswith('twin:'):
            assert g == 0 and isinstance(shot, Shot), kind
        elif kind.startswith('both:'):
            assert g >= 4 and isinstance(shot, Shot) and geos[g] != Geometry(), kind
        else:
            assert shot is None, kind
    assert len({g for k, g, _ in views if k.startswith('both:')}) == len(labs), 'each both:* has its own frame'
    assert (geos, views) == probe_plans(7, labs), 'the same seed, the same plan'
    assert (geos, views) != probe_plans(8, labs), 'another seed, another plan'
    return f'K = {k_n} views on {len(geos)} frames: anchor / sib:interp / twin:* on fixed frames, sib:light / sib:large / both:* each its own; twin:* and both:* carry a Shot; deterministic'


def t_z_board() -> str:
    shared = make_board((2, 4, 4, 3), shared=True)
    ref = shared.ref
    shared.arr[:] = np.arange(shared.arr.size, dtype=np.uint8).reshape(shared.arr.shape)
    other = shared_memory.SharedMemory(name=ref[0])
    seen = np.ndarray(ref[1], dtype=np.uint8, buffer=other.buf)
    assert np.array_equal(seen, shared.arr), 'an attached view reads what the creator wrote'
    del seen
    other.close()
    release_board(shared)
    try:
        shared_memory.SharedMemory(name=ref[0])
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('the segment still exists after release_board')
    plain = make_board((2, 4, 4, 3), shared=False)
    assert plain.shm is None and plain.arr.shape == (2, 4, 4, 3)
    release_board(plain)
    return 'a shared board is read by name from another attach and gone after release_board; a plain board has no segment'


def t_q_place_moments() -> str:
    c = CFG.cofig
    grey = _grey(2)
    m0 = place_moments(grey, c, 224, 4, 0.0)
    assert m0.shape == (2, 29, 4, 4), m0.shape
    x = signal_pixels(grey, c)[..., 208:432, 208:432]
    hand = TF.avg_pool2d(torch.cat([x, second_moments(x, c)], 1), 56)
    assert torch.allclose(m0, hand, atol=1e-5), float((m0 - hand).abs().max())
    for lo in (10, 16):                                                              # the squares of a unit vector signal add to 1
        assert float((m0[:, lo:lo + 3].sum(1) - 1).abs().max()) < 1e-3, lo
    m64 = place_moments(grey, c, 224, 4, 64.0)
    assert m64.shape == m0.shape and not torch.allclose(m64, m0)
    v0, v64 = float(m0.flatten(2).var(2).mean()), float(m64.flatten(2).var(2).mean())
    assert v64 <= v0, (v64, v0)
    assert _raises(lambda: place_moments(grey, c, 224, 4, 400.0)), 'the widened field reaches the taper'
    return f'(2, 29, 4, 4); dilate 0 equals the hand-made moments; unit squares add to 1; dilate 64 differs, between-cell variance {v64:.4f} <= {v0:.4f}'


def t_r_shared_moments() -> str:
    c = CFG.cofig
    grey = _grey(12)                                                                 # 2 places x 2 slots x 3 sources
    m = shared_moments(grey, 2, 3, True, c, 224, 4, 64.0, 3)
    assert m.shape == (12, 29, 4, 4)
    ref_only = grey.view(2, 2, 3, 640, 640)[:, :, :1].reshape(-1, 640, 640)           # the canvases of each slot's ref alone (kept 1)
    for shared in (True, False):
        assert torch.equal(shared_moments(ref_only, 2, 3, shared, c, 224, 4, 64.0 if shared else 0.0, 1),
                           shared_moments(grey, 2, 3, shared, c, 224, 4, 64.0 if shared else 0.0, 3)), f'kept 1 reads the same ref canvases (shared {shared})'
    ref0 = place_moments(grey[0:1], c, 224, 4, 64.0)[0]
    ref1 = place_moments(grey[6:7], c, 224, 4, 64.0)[0]
    assert all(torch.allclose(m[i], ref0, atol=1e-5) for i in range(6)), 'shared: one target for the place, the canonical ref\'s'
    assert all(torch.allclose(m[i], ref1, atol=1e-5) for i in range(6, 12))
    d = shared_moments(grey, 2, 3, False, c, 224, 4, 0.0, 3)
    assert all(torch.equal(d[0], d[i]) for i in (1, 2)) and all(torch.equal(d[3], d[i]) for i in (4, 5)), 'dep: one a slot'
    assert not torch.allclose(d[0], d[3]), 'dep: the two slots differ'
    assert torch.allclose(d[3], place_moments(grey[3:4], c, 224, 4, 0.0)[0], atol=1e-5)
    assert _raises(lambda: shared_moments(grey, 2, 3, False, c, 224, 4, 64.0, 3)), 'dep does not widen'
    return 'shared: the 6 views of a place carry (slot 0, ref)\'s descriptor; dep: a slot\'s 3 views carry its ref\'s, the slots differ'


def t_s_probe_scores() -> str:
    torch.manual_seed(0)
    p, d = 16, 64
    kinds = ['anchor', 'twin:x', 'both:x']
    cell = torch.tensor([0] * 8 + [1] * 8)
    a = torch.randn(p, d)
    f = torch.stack([a, a + 0.01 * torch.randn(p, d), torch.randn(p, d)], 1)         # [P, K, d]
    s = probe_scores([f], cell, kinds, False)
    assert set(s) == {'twin:x', 'both:x'}
    assert s['twin:x']['hit'] == 1.0 and s['twin:x']['consumed'] < 0.05, s['twin:x']
    assert s['both:x']['hit'] < 0.5 and s['both:x']['consumed'] > 0.5, s['both:x']
    both = probe_scores([f, f[..., :32]], cell, kinds, False)
    assert both['twin:x']['hit'] == 1.0
    assert all(s[kd]['spread'] > 0 for kd in s), {kd: s[kd]['spread'] for kd in s}
    by_cell = probe_cell_scores([f], cell, kinds)
    assert set(by_cell) == {'twin:x', 'both:x'}
    assert all(len(v) == 2 for kd in by_cell for v in by_cell[kd].values()), 'a number a cell (2 cells)'
    assert by_cell['twin:x']['hit'] == [1.0, 1.0], by_cell['twin:x']['hit']
    exact = probe_cell_scores([torch.stack([a, a], 1)], cell, ['anchor', 'twin:x'])                 # every view is its anchor: all right
    assert exact['twin:x']['hit'] == [1.0, 1.0] and all(c < 1e-6 for c in exact['twin:x']['consumed']), exact
    base = probe_scores([f], cell, kinds, True)
    shifted = probe_scores([f + 1000.0 * torch.randn(1, 1, d)], cell, kinds, True)
    for kd in ('twin:x', 'both:x'):
        assert base[kd]['hit'] == shifted[kd]['hit'] and abs(base[kd]['consumed'] - shifted[kd]['consumed']) < 1e-3, (base[kd], shifted[kd])
    spread = anchor_spread([f], cell, False)
    assert spread['nn'] < 0.9 and spread['n'] == 16.0, spread
    dup = f.clone()
    dup[1, 0] = dup[0, 0]
    pair = anchor_spread([dup[:2]], torch.tensor([0, 0]), False)                    # a cell of n = 2: each anchor's only other anchor
    assert abs(pair['nn'] - 1.0) < 1e-5 and abs(pair['neg'] - 1.0) < 1e-5 and pair['n'] == 2.0, pair
    return (f'twin: hit {s["twin:x"]["hit"]:.2f} consumed {s["twin:x"]["consumed"]:.3f}; unrelated: hit {s["both:x"]["hit"]:.2f} '
            f'consumed {s["both:x"]["consumed"]:.2f}; a composite runs; centred scores ignore a large constant; anchor spread of random '
            f'vectors nn {spread["nn"]:.2f}, of a copied pair {pair["nn"]:.2f}')


def t_u_noisy() -> str:
    torch.manual_seed(0)
    x = torch.randn(4096, 256)
    y = noisy(x, 0.5, True)
    assert not torch.equal(y, x)
    ratio = float((y - x).pow(2).mean().sqrt() / x.pow(2).mean().sqrt())
    assert 0.4 < ratio < 0.6, ratio
    ratio100 = float((noisy(100 * x, 0.5, True) - 100 * x).pow(2).mean().sqrt() / (100 * x).pow(2).mean().sqrt())
    assert abs(ratio100 - ratio) < 0.02, (ratio, ratio100)
    assert noisy(x, 0.5, False) is x and noisy(x, 0.0, True) is x
    z = torch.zeros(4096, 256)
    z[:, 0] = 10.0
    n = noisy(z, 0.5, True) - z
    sd = n.std(0)
    iso = float(sd[0] / sd[1:].mean())
    assert 0.8 < iso < 1.25, iso
    return f'noise / signal rms {ratio:.2f} (x100: {ratio100:.2f}); off in eval and at sigma 0; isotropic (sd ratio of a 10-valued dimension to the rest {iso:.2f})'


def t_cfg_cells() -> str:
    cfg = CoaeConfig()
    assert cfg.token_grid == 14 and cfg.latent_side == 7 and cfg.cofig.out_cells == 28
    assert _raises(lambda: replace(CFG, cofig=replace(CFG.cofig, out_cells=32))), 'gigapath: out_cells is 2 x 14'
    uni2 = replace(CFG, fm=replace(CFG.fm, encoder='uni2'), cofig=replace(CFG.cofig, out_cells=32))
    assert uni2.token_grid == 16 and uni2.latent_side == 8
    assert _raises(lambda: replace(CFG, fm=replace(CFG.fm, encoder='uni2'))), 'uni2 with 28 cells'
    assert _raises(lambda: replace(CFG, fm=replace(CFG.fm, encoder='nope'))), 'an encoder without a token grid'
    assert _raises(lambda: replace(CFG, model=replace(CFG.model, decoder_dim=(384, 192)))), 'two widths reach 2s, not 4s'
    assert _raises(lambda: replace(CFG, model=replace(CFG.model, swin_dim=100))), 'swin_dim not a multiple of the heads'
    return 'gigapath: grid 14, latent 7, 28 cells; uni2: 16, 8, 32 cells; a mismatch of out_cells, a wrong decoder depth or swin_dim raise'


def t_v_config() -> str:
    cfg = CoaeConfig()
    assert cfg.arm == 'RP3' and cfg.data.slots == 2 and cfg.data.twins and cfg.data.place_shared and cfg.data.dilate_px == 64.0
    assert replace(cfg, data=replace(cfg.data, mechanism='dep')).data.dilate_px == 0.0
    assert replace(cfg, data=replace(cfg.data, level='0')).data.slots == 1
    assert cfg.head.output == 'point' and replace(cfg, head=replace(cfg.head, variant='vdist')).head.output == 'dist'
    replace(cfg, head=replace(cfg.head, w_pool=0.0))                   # the descriptor-loss ablation builds
    assert _raises(lambda: replace(cfg, head=replace(cfg.head, w_pool=-1.0)))
    xp: CoaeConfig = replace(cfg, head=replace(cfg.head, w_recon=0.0))   # w_recon 0 with no contrast builds: no local branch (xP2)
    assert xp.arm == 'xP3', xp.arm
    replace(cfg, head=replace(cfg.head, w_recon=0.0, w_vnce=1.0))
    replace(cfg, head=replace(cfg.head, w_recon=0.0, w_dnce=0.5))
    replace(cfg, head=replace(cfg.head, w_recon=0.0, w_lnce=0.5))
    assert _raises(lambda: replace(cfg, data=replace(cfg.data, mechanism='ind'), head=replace(cfg.head, w_dnce=0.5))), 'w_dnce needs twins'
    assert _raises(lambda: replace(cfg, data=replace(cfg.data, sims=()), head=replace(cfg.head, w_dnce=0.5))), 'w_dnce needs twins of two sources'
    assert _raises(lambda: replace(cfg, head=replace(cfg.head, w_dnce=0.5, variant='vdist'))), 'w_dnce needs the point output'
    assert _raises(lambda: replace(cfg, data=replace(cfg.data, mechanism='ind'), head=replace(cfg.head, w_lnce=0.5))), 'w_lnce needs twins'
    assert _raises(lambda: replace(cfg, head=replace(cfg.head, twin_cells=0))) and _raises(lambda: replace(cfg, head=replace(cfg.head, twin_cells=50)))
    replace(cfg, head=replace(cfg.head, w_hnce=0.5, w_pool=0.0))                       # the head contrast in the place of the descriptor loss
    assert all(_raises(lambda k=k: replace(cfg, head=replace(cfg.head, **{k: -0.5}))) for k in ('w_hnce', 'w_dnce', 'w_lnce')), 'a negative weight'
    replace(cfg, head=replace(cfg.head, w_nce=0.5))
    assert _raises(lambda: replace(cfg, data=replace(cfg.data, mechanism='dep'), head=replace(cfg.head, w_nce=0.5))), 'w_nce needs one descriptor a place'
    assert _raises(lambda: replace(cfg, head=replace(cfg.head, w_nce=0.5, w_pool=0.0))), 'w_nce needs the descriptor loss'
    assert _raises(lambda: replace(cfg, head=replace(cfg.head, tau=0.0))), 'tau > 0'
    assert _raises(lambda: replace(cfg, head=replace(cfg.head, w_nce=-0.5))) and _raises(lambda: replace(cfg, head=replace(cfg.head, w_vnce=-1.0)))
    assert _raises(lambda: replace(cfg, data=replace(cfg.data, level='huge')))
    assert _raises(lambda: replace(cfg, data=replace(cfg.data, mechanism='x')))
    assert _raises(lambda: replace(cfg, head=replace(cfg.head, variant='x')))
    assert _raises(lambda: replace(cfg, data=replace(cfg.data, sims=()), head=replace(cfg.head, variant='vdist')))
    assert cfg.data.pool_cap == 128 and cfg.data.cell_parts == 1 and cfg.data.places == 128, (cfg.data.pool_cap, cfg.data.cell_parts, cfg.data.places)
    assert _raises(lambda: replace(cfg, data=replace(cfg.data, pool_cap=-1)))
    assert _raises(lambda: replace(cfg, data=replace(cfg.data, cell_parts=0)))
    assert not hasattr(cfg.data, 'spare'), 'a step lists places: the pool is screened at start'
    assert _raises(lambda: replace(cfg, data=replace(cfg.data, processes=0))), 'the workers read and make every view'
    assert all('jpeg' not in s for s in ProbeConfig().ood), 'the probe sources have no codec'
    assert cfg.canvases == 1 and replace(cfg, head=replace(cfg.head, variant='vavg')).canvases == 1 + len(cfg.data.sims) == 4
    assert cfg.train.epochs == 3, cfg.train.epochs
    assert cfg.fm.layers == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), cfg.fm.layers
    assert resolve_encoder_layers(cfg.fm.layers, 40) == (3, 7, 11, 15, 19, 23, 27, 31, 35, 39), resolve_encoder_layers(cfg.fm.layers, 40)
    assert resolve_encoder_layers(cfg.fm.layers, 24) == (1, 4, 6, 9, 11, 13, 16, 18, 21, 23), resolve_encoder_layers(cfg.fm.layers, 24)
    assert _raises(lambda: replace(cfg, fm=replace(cfg.fm, layers=()))), 'no layer to mix'
    assert _raises(lambda: replace(cfg, fm=replace(cfg.fm, init=(1.0, 1.0, 1.0)))), 'fm.init needs one value a layer'
    assert cfg.fm.weights_dtype == 'fp32', cfg.fm.weights_dtype
    replace(cfg, fm=replace(cfg.fm, weights_dtype='bf16'))
    replace(cfg, fm=replace(cfg.fm, weights_dtype='fp32', lora=True))
    assert _raises(lambda: replace(cfg, fm=replace(cfg.fm, weights_dtype='fp16'))), 'an unknown FM weight dtype'
    assert _raises(lambda: replace(cfg, fm=replace(cfg.fm, weights_dtype='bf16', lora=True))), 'LoRA needs float32 weights'
    assert _raises(lambda: replace(cfg, train=replace(cfg.train, epochs=4, warmup_epochs=5.0))), 'a warm-up longer than the run'
    replace(cfg, train=replace(cfg.train, epochs=4, warmup_epochs=0.5))
    replace(cfg, train=replace(cfg.train, epochs=0, warmup_epochs=1.0))
    return ('defaults give RP3 (2 slots, twins, shared place target, dilate 64), pool_cap 128, cell_parts 1, places 128, 3 epochs, '
            'layers as depth shares (gigapath -> 3, 7 .. 39; uni2 -> 1, 4, 6, 9, 11, 13, 16, 18, 21, 23); dep does not dilate; level 0 has '
            '1 slot; canvases 1 under vref, 4 under vavg; bad settings raise')


def _cfg_of(flags: List[str]) -> CoaeConfig:
    """The config of train.py's flags (what `CoaeTrain.sh` gives `--print-arm`)."""
    saved: List[str] = sys.argv
    try:
        sys.argv = ['train.py', '--out', 'x'] + flags
        return make_cfg(parse())
    finally:
        sys.argv = saved


def t_arm_names() -> str:
    from locascope.training.CoAE.Config import LOSS_TERMS, NAME_PARTS, UNNAMED                       # noqa: PLC0415
    sims2: List[str] = ['--sims', 'bench,bench-real']
    cases: List[Tuple[str, List[str]]] = [           # (the arm's name, the flags that make it): the names the saved arms get (spec.md)
        ('RP3', []), ('RP3', ['--sims', 'bench,bench-real,ad-chroma4']), ('RP0', ['--sims', 'none']),
        ('RP3.dep', ['--mechanism', 'dep']), ('RP3.ind', ['--mechanism', 'ind']), ('RP3.vdist', ['--variant', 'vdist']),
        ('Rx3', ['--w-pool', '0']), ('RP3.fmresidual', ['--cls-residual']), ('RN3', ['--w-nce', '0.5']),
        ('RP2', sims2), ('RP2.seed1', sims2 + ['--seed', '1']),
        ('xV2.weightV1', sims2 + ['--w-vnce', '1', '--w-recon', '0', '--w-pool', '0']),
        ('RPV2', sims2 + ['--w-vnce', '0.5']), ('RPV2.seed1', sims2 + ['--w-vnce', '0.5', '--seed', '1']),
        ('RH2', sims2 + ['--w-pool', '0', '--w-hnce', '0.5']), ('DP2', sims2 + ['--w-recon', '0', '--w-dnce', '0.5']),
        ('DH2', sims2 + ['--w-recon', '0', '--w-pool', '0', '--w-dnce', '0.5', '--w-hnce', '0.5']),
        ('LH2', sims2 + ['--w-recon', '0', '--w-pool', '0', '--w-lnce', '0.5', '--w-hnce', '0.5']),
        ('RV2', sims2 + ['--w-pool', '0', '--w-vnce', '0.5']),
        ('RP3.pool512.epochs12', ['--pool-cap', '512', '--epochs', '12']), ('RP3.pool16.epochs24', ['--pool-cap', '16', '--epochs', '24']),
        ('RP3.slides50.epochs6', ['--slides', '50', '--epochs', '6']),
        ('RP3.noperturb', ['--level', '0']), ('RP3.lightperturb', ['--level', 'light']), ('RP3.desc7', ['--pool-cells', '7']),
        ('RP3.noise0.25', ['--noise', '0.25']), ('RP3.lora', ['--lora']),
        ('RP1.sims-bench', ['--sims', 'bench']), ('RP2.sims-bench+ad-chroma4', ['--sims', 'bench,ad-chroma4']),
        ('RN3.weightN1', ['--w-nce', '1.0']), ('RP3.weightR0.5', ['--w-recon', '0.5']), ('xP2', sims2 + ['--w-recon', '0']),
        ('RP3.ind.vdist.slides50.epochs6.seed2',
         ['--mechanism', 'ind', '--variant', 'vdist', '--slides', '50', '--epochs', '6', '--seed', '2']),
    ]
    BASE_S: str = 'large-mix-vref-sims-bench+bench-real'
    dirs: Dict[str, str] = {                         # the directory a code's run has: the names the saved runs have (result/CoaeTrain/<dir>), and the new ones
        'RP3': 'large-mix-vref', 'RP0': 'large-mix-vref-refonly', 'RP3.dep': 'large-dep-vref', 'RP3.ind': 'large-ind-vref',
        'RP3.vdist': 'large-mix-vdist', 'Rx3': 'large-mix-vref-wpool0', 'RP3.fmresidual': 'large-mix-vref-fmcls', 'RN3': 'large-mix-vref-nce',
        'RP2': BASE_S, 'RP2.seed1': BASE_S + '-seed1', 'xV2.weightV1': BASE_S + '-vnceonly', 'RPV2': BASE_S + '-vnce',
        'RPV2.seed1': BASE_S + '-vnce-seed1', 'RH2': BASE_S + '-wpool0-hnce', 'DP2': BASE_S + '-wrecon0-dnce',
        'DH2': BASE_S + '-wrecon0-wpool0-dnce-hnce', 'LH2': BASE_S + '-wrecon0-wpool0-lat-hnce', 'RV2': BASE_S + '-wpool0-vnce',
        'RP3.pool512.epochs12': 'large-mix-vref-cap512-epochs12', 'RP3.pool16.epochs24': 'large-mix-vref-cap16-epochs24',
        'RP3.slides50.epochs6': 'large-mix-vref-slides50-epochs6', 'RP3.noperturb': '0-mix-vref', 'RP3.lightperturb': 'light-mix-vref',
        'RP3.desc7': 'large-mix-vref-pool7', 'RP3.noise0.25': 'large-mix-vref-noise0.25',
        'RP3.lora': 'large-mix-vref-lora', 'RP1.sims-bench': 'large-mix-vref-sims-bench',
        'RP2.sims-bench+ad-chroma4': 'large-mix-vref-sims-bench+ad-chroma4', 'RP3.weightR0.5': 'large-mix-vref-wrecon0.5',
        'xP2': BASE_S + '-wrecon0',
        'RP3.ind.vdist.slides50.epochs6.seed2': 'large-ind-vdist-seed2-slides50-epochs6'}
    by_name: Dict[str, str] = {}                     # a code -> the config it names
    by_dir: Dict[str, str] = {}                      # a directory -> the config it names
    for want, flags in cases:
        cfg: CoaeConfig = _cfg_of(flags)
        assert cfg.arm == want, (flags, cfg.arm, want)
        assert by_name.setdefault(want, str(asdict(cfg))) == str(asdict(cfg)), f'{want}: two configs, one code'
        if want in dirs:                             # a code and its directory: both made by the config, both read back to it, round trip code -> directory -> code
            assert cfg.dir_name == dirs[want], (want, cfg.dir_name, dirs[want])
            assert by_dir.setdefault(dirs[want], str(asdict(cfg))) == str(asdict(cfg)), f'{dirs[want]}: two configs, one directory'
            assert config_of_arm(want) == cfg and config_of_dir(dirs[want]) == cfg, want
            assert config_of_dir(config_of_arm(want).dir_name).arm == want and config_of_arm(config_of_dir(dirs[want]).arm).dir_name == dirs[want]
    uni2: CoaeConfig = _cfg_of(['--encoder', 'uni2'])           # the FM is not in the code (the same RP3 as gigapath's), only in the directory
    assert uni2.arm == 'RP3' and uni2.dir_name == 'large-mix-vref-uni2', (uni2.arm, uni2.dir_name)
    assert config_of_arm('RP3', 'uni2') == uni2 and config_of_dir('large-mix-vref-uni2') == uni2 and uni2 != config_of_arm('RP3')
    for d in ('large-mix-vref-uni2', 'large-mix-vref-uni2-nce'):
        assert config_of_arm(config_of_dir(d).arm, config_of_dir(d).fm.encoder).dir_name == d, d
    assert _raises(lambda: config_of_arm('RP3', 'nosuch')), 'an FM that is not in FM_GRIDS'
    assert set(dirs) <= set(by_name), 'a directory of a code that is not in the cases'
    assert len(set(by_name.values())) == len(by_name), 'two codes, one config'
    assert len(set(by_dir.values())) == len(by_dir), 'two directories, one config'
    assert _raises(lambda: _cfg_of(['--w-nce', '1.0']).dir_name), 'a contrast off its standard weight has a code (weightN1) but no directory name'
    for bad in ('', 'RP', 'RP3.nosuch', 'RP3.seed1.dep', 'RP3.weightR1', 'xx3', 'QP3', 'RP3.', 'RP3.fm-uni2'):
        assert _raises(lambda b=bad: config_of_arm(b)), f'{bad!r} is not the code of a config'
    for bad in ('', 'large-mix', 'large-mix-vref-nosuch', 'large-mix-vref-nce-lora', 'large-mix-vref-cap128', 'xlarge-mix-vref'):
        assert _raises(lambda b=bad: config_of_dir(b)), f'{bad!r} is not the directory of a config'
    # the tables are the one source: every loss term is a weight of the head, a letter once; a setting the name cannot say has no name
    fields: Tuple[str, ...] = tuple(asdict(CFG.head))
    assert all(f'w_{t.key}' in fields for t in LOSS_TERMS) and len({t.letter for t in LOSS_TERMS}) == len(LOSS_TERMS)
    assert all(t.rides_on in {u.key for u in LOSS_TERMS} for t in LOSS_TERMS if t.rides_on)
    assert len(NAME_PARTS) == 8 + len(LOSS_TERMS) + 4, len(NAME_PARTS)    # the FM is not a part of the code (the <Fm>CoAE prefix says it)
    assert {u.what for u in UNNAMED} == {'head.tau', 'head.twin_cells', 'head.w_koleo', 'head.w_adv'}
    for head_change in ({'tau': 0.2}, {'twin_cells': 4}, {'w_koleo': 0.1}, {'w_adv': 0.1}, {'w_dnce': 0.5}):
        assert _raises(lambda c=head_change: replace(CFG, head=replace(CFG.head, **c)).arm), f'{head_change}: no name'
    return f'{len(cases)} flag sets -> their names (RP3, RP0, Rx3, RN3, xV2.weightV1, RPV2, RH2, DP2, DH2, LH2, RV2, RP3.pool512.epochs12 ...); one name a config; ' \
           f'tau / twin_cells / w_koleo / w_adv off their default and two local losses have no name'


def t_t5_warmup_flag() -> str:
    saved: List[str] = sys.argv
    try:
        sys.argv = ['train.py', '--out', 'x', '--warmup-epochs', '0.5']
        assert make_cfg(parse()).train.warmup_epochs == 0.5
        sys.argv = ['train.py', '--out', 'x']
        assert make_cfg(parse()).train.warmup_epochs == 1.0
    finally:
        sys.argv = saved
    return '--warmup-epochs 0.5 sets TrainConfig.warmup_epochs; without the flag it is 1.0'


def t_t6_places_read() -> str:
    assert places_read(7, 1, torch.device('cpu')) == [7]
    return 'places_read(7, 1) == [7] (the many-process path: TestCoae)'


def _lattice() -> Tuple[List[Place], List[int]]:
    """A 40 x 30 lattice of places (pitch 384 px) with the lower right corner cut away (10 x 8 places): 1,120 places, not a rectangle."""
    places: List[Place] = []
    for iy in range(30):
        for ix in range(40):
            if ix >= 30 and iy >= 22:
                continue
            cx: float = 192.0 + 384.0 * ix
            cy: float = 192.0 + 384.0 * iy
            places.append(Place(0, 0, 1.0, cx, cy, (cx - 192.0, cy - 192.0, cx + 192.0, cy + 192.0)))
    return places, list(range(len(places)))


def _firsts(runs: List[List[int]]) -> List[int]:
    """The first candidate of each run, sorted: what a pool without screening holds."""
    return sorted(r[0] for r in runs)


def t_t_pool_runs() -> str:
    places, idx = _lattice()
    n: int = len(idx)
    for cap in (0, n, n + 5):
        runs0: List[List[int]] = pool_runs(places, idx, cap, 'k')
        assert all(len(r) == 1 for r in runs0) and _firsts(runs0) == idx, cap
    runs: List[List[int]] = pool_runs(places, idx, 128, 'k')
    pool: List[int] = _firsts(runs)
    assert len(pool) == 128 == len(set(pool)) and set(pool) <= set(idx), len(pool)
    flat: List[int] = [i for r in runs for i in r]
    assert sorted(flat) == idx, 'the runs are disjoint and their union is the cell'
    assert all(n // 128 <= len(r) <= n // 128 + 1 for r in runs), sorted({len(r) for r in runs})
    assert _firsts(pool_runs(places, idx, 128, 'k')) == pool, 'the same key, the same pool'
    assert _firsts(pool_runs(places, idx, 128, 'other')) != pool, 'another key, another pool'
    shuffled: List[int] = list(idx)
    random.Random(3).shuffle(shuffled)
    assert _firsts(pool_runs(places, shuffled, 128, 'k')) == pool, 'only the coordinates decide, not the order of idx'
    count: np.ndarray = np.zeros((4, 4), dtype=np.int64)
    total: np.ndarray = np.zeros((4, 4), dtype=np.int64)
    in_pool: set = set(pool)
    for i in idx:
        bx: int = min(3, int(places[i].cx // 384) * 4 // 40)
        by: int = min(3, int(places[i].cy // 384) * 4 // 30)
        total[by, bx] += 1
        count[by, bx] += i in in_pool
    off: float = float(np.abs(count - total * 128 / n).max())
    assert off <= 6, f'a block of the 4 x 4 split holds {off:.1f} places more or fewer than its share'
    codes: List[int] = [int(_morton(np.array([x]), np.array([y]))[0]) for x, y in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0))]
    assert codes == [0, 1, 2, 3, 4], codes
    return f'{n} places: cap 0 and cap >= n give a run a place; cap 128 gives 128 disjoint runs covering the cell (n // 128 to n // 128 + 1 '\
           f'long), their first candidates distinct, the same for one key, another for another, whatever the order of idx; spread over the 16 '\
           f'blocks (largest deviation {off:.1f}); Z-order codes 0 1 2 3 4'


def t_t7_screened_pool() -> str:
    runs: List[List[List[int]]] = [[[0, 1, 2], [3, 4], [5]], [[10, 11], [12]], [[20, 21]]]    # three cells
    asked: List[int] = []

    def screen_with(bad: set) -> Callable[[List[int]], List[int]]:
        def screen(idx: List[int]) -> List[int]:
            asked.extend(idx)
            return [BLANK if i in bad else USABLE for i in idx]
        return screen

    pools, tally, rounds = screened_pool(runs, screen_with(set()))
    assert pools == [[0, 3, 5], [10, 12], [20]] and rounds == 1 and tally == {USABLE: 6, BLANK: 0, 2: 0}, (pools, rounds, tally)
    asked.clear()
    pools, tally, rounds = screened_pool(runs, screen_with({0}))
    assert pools == [[1, 3, 5], [10, 12], [20]] and rounds == 2, (pools, rounds)
    asked.clear()
    pools, tally, rounds = screened_pool(runs, screen_with({3, 4}))
    assert pools == [[0, 5], [10, 12], [20]], pools                                       # a run with none adds nothing
    assert sum(tally.values()) == len(asked), (tally, len(asked))
    asked.clear()
    pools, tally, rounds = screened_pool(runs, screen_with({20, 21}))
    assert pools == [[0, 3, 5], [10, 12], []], pools                                      # a cell with none is []
    assert sum(tally.values()) == len(asked) and tally[BLANK] == 2, (tally, len(asked))
    again = screened_pool(runs, screen_with({3, 4}))
    assert again[0] == screened_pool(runs, screen_with({3, 4}))[0], 'the same input, the same pools'
    return 'screened_pool: no blank -> each run\'s first, one round; a blank first -> the next; a run all blank adds nothing; a cell all blank is empty; the tally counts what was asked'


def t_t2_epoch_seed() -> str:
    seeds: set = {epoch_seed(0, r, e) for r in range(8) for e in range(100)}
    assert len(seeds) == 800, len(seeds)
    return 'epoch_seed differs over rank 0-7 x epoch 0-99'


def t_t3_keep_rows() -> str:
    with tempfile.TemporaryDirectory() as d:
        path: Path = Path(d) / 'rows.tsv'
        path.write_text(''.join(f'row\t{i}\n' for i in range(5)))
        keep_rows(path, lambda l: int(l.split('\t')[1]) < 3)
        assert path.read_text() == 'row\t0\nrow\t1\nrow\t2\n', path.read_text()
        assert sorted(p.name for p in Path(d).iterdir()) == ['rows.tsv'], 'no .tmp left'
        gone: Path = Path(d) / 'none.tsv'
        keep_rows(gone, lambda l: True)
        assert gone.read_text() == '' and sorted(p.name for p in Path(d).iterdir()) == ['none.tsv', 'rows.tsv']
        log: Path = Path(d) / 'log.tsv'
        log.write_text('{"step": 1}\n{"step": 2}\n{"step": 3}\n{"step": 4')                  # the last row cut: no newline, half a JSON
        keep_rows(log, lambda l: log_row_kept(l, 4))
        assert log.read_text() == '{"step": 1}\n{"step": 2}\n{"step": 3}\n', log.read_text()
        log.write_text('{"step": 1}\n{"step": 2')                                           # a newline but half a JSON is not a row either
        keep_rows(log, lambda l: log_row_kept(l, 4))
        assert log.read_text() == '{"step": 1}\n', log.read_text()
        val: Path = Path(d) / 'val.tsv'
        val.write_text('probe\t-1\t{"a": 1}\nprobe\t0\t{"a": 2}\nprobe\t1\t{"a"')            # the last row cut
        keep_rows(val, lambda l: val_row_kept(l, 1))
        assert val.read_text() == 'probe\t-1\t{"a": 1}\nprobe\t0\t{"a": 2}\n', val.read_text()
        val.write_text('probe\t0\t{"a": 1}\nprobe\t1\t{"a": 2}\nprobe\t2\t{"a": 3}\n')
        keep_rows(val, lambda l: val_row_kept(l, 1))
        assert val.read_text() == 'probe\t0\t{"a": 1}\nprobe\t1\t{"a": 2}\n', val.read_text()
    return 'keep_rows keeps the accepted lines, leaves no .tmp; a missing file becomes an empty one; a cut last row (no newline / half JSON) is dropped'


def t_t4_checkpoint_roundtrip() -> str:
    with tempfile.TemporaryDirectory() as d:
        path: Path = Path(d) / 'epoch0.pt'
        torch.save({'cfg': asdict(CFG), 'world': 1, 'epoch': 0, 'step': 3}, path)
        ck: Dict[str, Any] = load_checkpoint(path, CFG, 1)
        assert ck['step'] == 3
        other: CoaeConfig = replace(CFG, model=replace(CFG.model, noise=0.3))
        try:
            load_checkpoint(path, other, 1)
        except ValueError as exc:
            assert 'model.noise' in str(exc) and 'the config differs' in str(exc), str(exc)
        else:
            raise AssertionError('another config was accepted')
        assert _raises(lambda: load_checkpoint(path, CFG, 2)), 'another process count'
        try:
            load_checkpoint(Path(d) / 'none.pt', CFG, 1)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('a missing checkpoint was accepted')
    return 'a saved asdict(config) equals the new one after torch.save / load (tuples stay tuples); another config (named field), another world, a missing file are refused'


def t_w_probe_labs() -> str:
    labs = probe_labs(ProbeConfig().ood)
    assert set(labs) == {'bench', 'bench-real', 'ad-chroma4', 'ood-heavy', 'ood-soft'}, set(labs)
    for n, lab in labs.items():
        assert tuple(lab.base.distortion_k1_range) == (0.0, 0.0) and lab.base.distortion_k2 == 0.0 and lab.base.stage_shift_max == 0, n
    assert tuple(labs['ood-soft'].blur_factor) == (1.5, 2.5), labs['ood-soft'].blur_factor
    assert all(lab.base.jpeg is False for lab in labs.values()), 'no source has a codec'
    assert 'jpeg' not in AD_CHROMA4
    base = sim_labs()['bench'].base
    for text in ('x|jpeg_quality=60', 'x|jpeg=1'):
        assert _raises(lambda: parse_variant(text, base)), text
    return '5 sources (3 simulators + ood-heavy + ood-soft), none with a lens, a stage shift or a codec; ood-soft blurs 1.5..2.5; a variant cannot set the codec'


def t_n2_no_codec() -> str:
    labs = sim_labs()
    soft = probe_labs(ProbeConfig().ood)['ood-soft']
    img = texture(256, 3)
    wh: Tuple[int, int] = (256, 256)
    for s in range(3):
        assert draw_shot(soft, random.Random(s)) == draw_shot(soft, random.Random(s)), 'ood-soft: the same seed is the same shot'
        lab = labs['bench']
        assert lab.base.jpeg is False, 'bench: no codec'
        shot = draw_shot(lab, random.Random(s))
        _, p = simulate_with_gt(img, cfg=lab.base, rng=random.Random(s), output_wh=wh)
        assert (shot.saturation, shot.brightness, shot.contrast, shot.color_temp, shot.defocus, shot.chromatic, shot.vignette) == (
            float(p['saturation']), float(p['brightness']), float(p['contrast']), float(p['color_temp']), int(p['defocus_radius']),
            int(p['chromatic_shift']), float(p['vignette_strength'])), 'bench: the shot is simulate_with_gt\'s draw (no codec)'
    return 'ood-soft\'s shot is reproducible; bench\'s shot is simulate_with_gt\'s own draw'


# ── CoAE as stage 2's tile feature (the real validation: core/stage2_retrieval/RetrieverCoae.py) ───────────────────────────────

def t_stage2_arm() -> str:
    from locascope.core.backbones.TileEncoder import encoder_config                               # noqa: PLC0415
    from locascope.core.stage2_retrieval import retriever_class                                    # noqa: PLC0415
    from locascope.core.stage2_retrieval.ArmSpec import (COAE_PARTS, ENCODER_FAMILIES, ArmSyntaxError, coae_encoder,   # noqa: PLC0415
                                                         coae_encoder_name, coae_weights, family_of, parse_arm, phase_configs)
    from locascope.core.stage2_retrieval.RetrieverCoae import RetrieverCoaeConfig                  # noqa: PLC0415
    from locascope.core.stage2_retrieval.Naming import arm_label, config_label                    # noqa: PLC0415
    text: str = 'GigaCoAE+RP3:SlidingWindow::cls_latent_avg+mean'
    spec: Any = parse_arm(text)
    assert spec.text() == text, spec.text()
    cfgs: List[Any] = phase_configs(spec)
    assert len(cfgs) == 1 and type(cfgs[0]).__name__ == 'RetrieverCoaeConfig', cfgs
    assert cfgs[0].weights == coae_weights('RP3', 'gigapath') == 'CoaeTrain/large-mix-vref/weights/epoch2.pt', cfgs[0].weights
    assert cfgs[0].feature == 'cls_latent_avg' and cfgs[0].score == 'mean' and cfgs[0].raw_topk == 1000 and cfgs[0].overlap is True
    assert cfgs[0].encoder.REGISTERED_AS == 'gigapath' and cfgs[0].encoder.pooling == 'cls'
    assert not hasattr(cfgs[0], 'centre'), 'no centre file: every cosine is the plain one'
    # the hash does not change with the grammar: the config is the one the old grammar built, field by field
    old: RetrieverCoaeConfig = RetrieverCoaeConfig(
        encoder=encoder_config('gigapath', pooling='cls').with_model(dtype='fp16'), tile_size=256, overlap=True, raw_topk=1000,
        min_sep_tiles=1.0, weights='CoaeTrain/large-mix-vref/weights/epoch2.pt', feature='cls_latent_avg', score='mean')
    assert cfgs[0].identity_id() == old.identity_id(), 'the new grammar must make the old config'
    variants: Dict[str, str] = {'GigaCoAE+RN3:SlidingWindow::fm_head+mean': 'CoaeTrain/large-mix-vref-nce/weights/epoch2.pt',
                                'GigaCoAE+RP3.pool512.epochs12:SlidingWindow::fm_head+mean': 'CoaeTrain/large-mix-vref-cap512-epochs12/weights/epoch11.pt',
                                'GigaCoAE+RP2.sims-bench+ad-chroma4:SlidingWindow::head+mean':
                                    'CoaeTrain/large-mix-vref-sims-bench+ad-chroma4/weights/epoch2.pt'}
    for v_text, v_weights in variants.items():
        v_spec: Any = parse_arm(v_text)
        assert v_spec.text() == v_text and phase_configs(v_spec)[0].weights == v_weights, (v_text, phase_configs(v_spec)[0].weights)
    label_a: str = 'slidewin-GigaCoAE-RP3-cls-latent-avg-mean'
    assert config_label(cfgs[0]) == label_a == arm_label(spec), config_label(cfgs[0])
    assert arm_label(parse_arm('GigaCoAE+RN3:SlidingWindow::fm_head+mean')) == 'slidewin-GigaCoAE-RN3-fm-head-mean'
    geo: Any = phase_configs(parse_arm('GigaCoAE+RP3:SlidingWindow::cls_latent_avg+geomean'))[0]
    assert config_label(geo).endswith('-geomean') and geo.identity_id() != cfgs[0].identity_id()
    assert retriever_class(cfgs[0]).__name__ == 'RetrieverCoae'
    assert coae_encoder('gigapath') is None and coae_encoder('GigaCoAE+RP3') == ('gigapath', 'RP3')
    assert coae_encoder('GigaCoAE+RP2.sims-bench+ad-chroma4') == ('gigapath', 'RP2.sims-bench+ad-chroma4')
    assert _raises(lambda: coae_encoder('GigaCoAE')), 'a CoAE encoder names its arm'
    assert coae_encoder_name('gigapath', coae_weights('RN3', 'gigapath')) == 'GigaCoAE+RN3', 'the inverse'
    assert coae_encoder_name('gigapath', coae_weights('RP2.sims-bench+ad-chroma4', 'gigapath')) == 'GigaCoAE+RP2.sims-bench+ad-chroma4'
    assert _raises(lambda: coae_encoder_name('gigapath', 'CoaeTrain/large-mix-vref/weights/last.pt')), 'another file'
    assert _raises(lambda: coae_encoder_name('gigapath', 'CoaeTrain/nosuch/weights/epoch2.pt')), 'a directory that is no config\'s'
    assert coae_encoder_name('gigapath', 'CoaeTrain/large-mix-vref-sims-bench+bench-real-wpool0-hnce/weights/epoch2.pt') == 'GigaCoAE+RH2', 'a saved run'
    assert coae_weights('RH2', 'gigapath') == 'CoaeTrain/large-mix-vref-sims-bench+bench-real-wpool0-hnce/weights/epoch2.pt'
    assert _raises(lambda: coae_encoder_name('gigapath', 'Other/RP3/weights/epoch2.pt')), 'another directory'
    assert _raises(lambda: coae_encoder_name('uni2', coae_weights('RN3', 'gigapath'))), 'another FM'
    assert coae_encoder('Uni2CoAE+RN3') == ('uni2', 'RN3'), 'the FM is the prefix'
    assert coae_weights('RN3', 'uni2') == 'CoaeTrain/large-mix-vref-uni2-nce/weights/epoch2.pt'
    assert coae_encoder_name('uni2', coae_weights('RN3', 'uni2')) == 'Uni2CoAE+RN3', 'the inverse, uni2'
    assert _raises(lambda: coae_encoder_name('gigapath', coae_weights('RN3', 'uni2'))), 'another FM (a uni2 run named as gigapath)'
    assert _raises(lambda: coae_encoder('Uni2CoAE+RN3.fm-uni2')), 'the FM is not in the arm code'
    long_run: str = coae_weights('RP3.pool16.epochs24', 'gigapath')
    assert long_run.endswith('/weights/epoch23.pt'), long_run                                      # the last epoch of the run, not epoch2
    assert _raises(lambda: coae_encoder_name('gigapath', 'CoaeTrain/large-mix-vref-cap16-epochs24/weights/epoch2.pt')), 'not the last epoch'
    assert coae_encoder_name('gigapath', long_run) == 'GigaCoAE+RP3.pool16.epochs24', 'the inverse of a longer run'
    # the grammar asks the family registry, and knows no encoder by name
    assert tuple(ENCODER_FAMILIES) == ('CoAE',) and family_of('gigapath') is None and family_of('') is None
    fam: Any = family_of('GigaCoAE+RP3')
    assert fam is not None and fam[1] == ('gigapath', 'RP3') and fam[0] is ENCODER_FAMILIES['CoAE']
    for bad in ('gigapath:CoAE::cls+fm_head', 'GigaCoAE:SlidingWindow::fm_head+mean', 'GigaCoAE+RP3:SlidingWindow::fm_head',
                'GigaCoAE+RP3:SlidingWindow::nosuch+mean', 'GigaCoAE+RP3:BagOfTiles::fm_head+gap', 'UniCoAE+RP3:SlidingWindow::fm_head+mean',
                'GigaCoAE+:SlidingWindow::fm_head+mean', 'GigaCoAE+RP3:SlidingWindow::fm_head+mean::Dual::cls_avg+dual',
                'GigaCoAE+RP3:SlidingWindow::place_m+mean'):
        try:
            parse_arm(bad)
        except ArmSyntaxError:
            continue
        raise AssertionError(f'{bad!r} is not an arm of the grammar')
    after: str = 'GigaCoAE+RP3:SlidingWindow::fm_head+mean::gigapath:Dual::cls_avg+dual'
    assert parse_arm(after).text() == after
    ids: List[str] = [phase_configs(parse_arm(f'GigaCoAE+RP3:SlidingWindow::{op}+mean'))[0].identity_id() for op in ('place', 'latent', 'fm_cls_latent')]
    assert len(set(ids)) == 3, ids
    assert len(COAE_PARTS) == 18 and COAE_PARTS['fm'] == ('cls_fm',) and 'latent_dev' in COAE_PARTS
    assert COAE_PARTS['latent4_head'] == ('latent4', 'head') and COAE_PARTS['fm_latent4_head'] == ('cls_fm', 'latent4', 'head')
    return f'{config_label(cfgs[0])}; {len(COAE_PARTS)} usages; same id as the old grammar; three usages, three ids'


class _FakeNet:
    """What `CoaeTileEncoder` and `tile_parts` read of a CoaeNet: `cfg`, `latent_side`, `cls_dim`, `fm.dim`, `pool_shape`, `pool_sd`, `pool_mu`, and a
    call that answers a CoaeOut whose tensors are a fixed base plus the tile's mean times a fixed gain (no FM)."""

    def __init__(self) -> None:
        g: torch.Generator = torch.Generator().manual_seed(0)
        s, L, t = CFG.latent_side, CFG.model.latent_dim, CFG.cofig.out_cells
        n_out: int = n_channels(CFG.cofig) if CFG.head.output == 'point' else n_dist_channels(CFG.cofig)
        self.cfg: CoaeConfig = CFG
        self.latent_side: int = s
        self.cls_dim: int = 16
        self.fm: SimpleNamespace = SimpleNamespace(dim=16)
        self.pool_shape: Tuple[int, int] = (3, CFG.head.pool_cells)
        self.pool_sd: torch.Tensor = (torch.rand(3, generator=g) + 0.5).to(DEV)
        self.pool_mu: torch.Tensor = torch.randn(3, generator=g).to(DEV)          # what `tile_parts` takes off for the head (place less pool_mu / pool_sd)
        shapes: Dict[str, Tuple[int, ...]] = {'latent': (s, s, L), 'cls': (16,), 'cofig': (n_out, t, t), 'pooled': (3, CFG.head.pool_cells, CFG.head.pool_cells),
                                              'cls_fm': (16,), 'avg_fm': (16,)}
        self.base: Dict[str, torch.Tensor] = {k: torch.randn((1,) + v, generator=g).to(DEV) for k, v in shapes.items()}
        self.gain: Dict[str, torch.Tensor] = {k: torch.randn((1,) + v, generator=g).to(DEV) for k, v in shapes.items()}

    def __call__(self, tiles: torch.Tensor) -> CoaeOut:
        x: torch.Tensor = tiles.float().mean(dim=(1, 2, 3)) / 255.0
        made: Dict[str, torch.Tensor] = {k: self.base[k] + x.view(-1, *([1] * (self.base[k].dim() - 1))) * self.gain[k] for k in self.base}
        return CoaeOut(latent=made['latent'], cls=made['cls'], cofig=made['cofig'], pooled=made['pooled'], cls_fm=made['cls_fm'],
                       avg_fm=made['avg_fm'])


def _pool_by_hand(latent: torch.Tensor, side: int) -> torch.Tensor:
    """[B, s, s, L] -> [B, side * side * L]: bin i of s cells is [floor(i s / side), ceil((i + 1) s / side)), each bin's mean, cells first."""
    s: int = latent.shape[1]
    rows: List[torch.Tensor] = []
    for i in range(side):
        a, b = (i * s) // side, -((-(i + 1) * s) // side)
        cols: List[torch.Tensor] = []
        for j in range(side):
            c, d = (j * s) // side, -((-(j + 1) * s) // side)
            cols.append(latent[:, a:b, c:d].mean(dim=(1, 2)))
        rows.append(torch.stack(cols, 1))
    return torch.stack(rows, 1).flatten(1)


def t_tile_encoder() -> str:
    from locascope.core.stage2_retrieval.ArmSpec import COAE_PARTS                                 # noqa: PLC0415
    net: Any = _FakeNet()
    scale: ChannelScale = ChannelScale(n_channels(CFG.cofig)).to(DEV)
    scale.sd.copy_(torch.rand(n_channels(CFG.cofig)) + 0.5)
    tiles: torch.Tensor = torch.randint(0, 256, (5, 256, 256, 3), dtype=torch.uint8, generator=torch.Generator().manual_seed(1))
    o: CoaeOut = net(tiles.to(DEV))
    parts: Dict[str, torch.Tensor] = tile_parts(o, net, scale)
    widths: Dict[str, int] = part_widths(net)
    assert {p for ps in COAE_PARTS.values() for p in ps} <= set(parts), set(parts)
    for name, v in parts.items():
        assert v.dtype == torch.float32 and v.shape == (5, widths[name]), (name, v.shape, widths[name])
    lat: torch.Tensor = o.latent.float()
    place: torch.Tensor = (o.pooled / net.pool_sd.view(1, -1, 1, 1)).flatten(1)
    assert torch.allclose(parts['place'], place), 'place: the pooled descriptor over the network\'s running sd'
    assert torch.allclose(parts['latent_avg'], lat.mean(dim=(1, 2))), 'latent_avg: the mean cell'
    assert parts['latent_dev'].view(5, *lat.shape[1:]).mean(dim=(1, 2)).abs().max() < 1e-5, 'latent_dev: the cells minus their mean'
    assert torch.allclose(parts['latent_dev'] + parts['latent_avg'].repeat(1, CFG.latent_side ** 2), parts['latent'], atol=1e-5)
    assert torch.allclose(pool_latent(lat, 4), _pool_by_hand(lat, 4), atol=1e-6) and torch.allclose(parts['latent2'], _pool_by_hand(lat, 2), atol=1e-6)
    cofig: torch.Tensor = (cofig_point(o.cofig.float(), CFG.cofig, scale, CFG.head.output) / scale.sd.view(1, -1, 1, 1)).flatten(1)
    assert torch.allclose(parts['cofig'], cofig), 'cofig: the Co-figure over the targets\' running sd'
    want: Dict[str, torch.Tensor] = {
        'place': TF.normalize(place, dim=1),
        'cls_latent': TF.normalize(torch.cat([TF.normalize(o.cls, dim=1), TF.normalize(lat.flatten(1), dim=1)], 1), dim=1),
        'latent4': TF.normalize(_pool_by_hand(lat, 4), dim=1)}
    for op, w in want.items():
        enc: CoaeTileEncoder = CoaeTileEncoder(net, scale, DEV, op, batch=2)               # a batch of 2: the loop runs three times
        got: torch.Tensor = enc(tiles)
        assert got.dtype == torch.float32 and got.shape == w.shape, (op, got.dtype, got.shape)
        assert (got.norm(dim=1) - 1).abs().max() < 1e-3, op
        assert torch.allclose(got, w, atol=1e-5), (op, float((got - w).abs().max()))
        assert torch.equal(got, enc([t.numpy() for t in tiles])), f'{op}: a list of arrays is a tensor'
        assert enc.feature_pooling == f'coae-{op}'
    assert unit_feature(parts, ('cls_fm',)).shape == (5, 16) and torch.allclose(unit_feature(parts, ('cls_fm',)), TF.normalize(o.cls_fm, dim=1), atol=1e-6)
    enc = CoaeTileEncoder(net, scale, DEV, 'fm_cls_latent')
    assert enc(tiles[:0]).shape == (0, 16 + 16 + widths['latent']), 'an empty input: no row, the usage\'s width'
    assert _raises(lambda: enc(tiles[:, :128])) and _raises(lambda: enc(tiles[0])), 'a tile of another side / not a stack of tiles'
    assert _raises(lambda: CoaeTileEncoder(net, scale, DEV, 'place_m')) and _raises(lambda: CoaeTileEncoder(net, scale, DEV, 'nosuch'))
    enc = CoaeTileEncoder(net, scale, DEV, 'latent', batch=2)
    cells: torch.Tensor = enc.cells(tiles)
    side, dim = CFG.latent_side, CFG.model.latent_dim
    assert cells.dtype == torch.float32 and tuple(cells.shape) == (5, side, side, dim), (cells.dtype, tuple(cells.shape))
    assert float((TF.normalize(cells.flatten(1), dim=1) - enc(tiles)).abs().max()) < 1e-6, 'cells made unit and flat are the `latent` usage'
    assert tuple(enc.cells(tiles[:0]).shape) == (0, side, side, dim), 'an empty input: no cell, the usage\'s shape'
    return 'place, cls_latent, latent4 equal the hand-made vectors (float32, unit); every part has its width; a list is a tensor; bad sides and ops are refused; cells are the latent as it is'


_TESTS = [t for n, t in sorted(globals().items()) if n.startswith('t_')]


def main() -> int:
    print(f'device {DEV}', flush=True)
    for fn in _TESTS:
        check(fn.__name__[4:].replace('_', ' '), fn)
    failed: List[str] = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
