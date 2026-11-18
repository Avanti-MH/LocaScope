'''Validation (spec.md §6).

  probe  `probe_measures` (the training flow's measure, once before training and after each epoch): on the fixed probe (`Supply.Probe`:
         places of slides that are never trained on and have no real photo, each as the anchor and its fixed views), every feature
         of `PROBE_FEATURES` as a query of each view kind against the anchors of its (slide, rung) cell: hit, consumed, z; each
         feature directly and anchors'-mean-removed (`<name>-m`); the decoder's Co-figure of the anchor and the twins against the
         anchor's own; the effective rank of the anchors' cls'. `probe_features`, `probe_scores` are its two halves. The targets' own
         features (`direct`, `cofig-direct*`) are computed from each view's pixels, no network.
  real   the chosen checkpoint as stage 2's tile feature goes through RealTest (`core/stage2_retrieval/RetrieverCoae.py`); here are
         its tile encoder (`CoaeTileEncoder`: `unit_feature` of the parts of `tile_parts`, a usage of `ArmSpec.COAE_PARTS`) and the
         checkpoint's reader (`coae_of_checkpoint`). NOT called by the training flow (real photos only validate).
  cofig  `real_cofig`: the predicted Co-figure against the one computed from the real photo itself (`field_maps` on the FM's
         field), a signal (cos, scaled MSE) -- validation only, a one-off.
'''
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn.functional as TF
from PIL import Image

from locascope.core.stage2_retrieval.ArmSpec import COAE_PARTS
from locascope.training.CoAE.CoFigure import field_maps, layout, n_channels, part_names
from locascope.training.CoAE.Config import (CoaeConfig, CoFigureConfig, DataConfig, FmConfig, HeadConfig, ModelConfig, ProbeConfig,
                                            TrainConfig)
from locascope.training.CoAE.Losses import ChannelScale, agreement, cofig_point
from locascope.training.CoAE.Model import CoaeNet, CoaeOut
from locascope.training.CoAE.Sims import GREY_WEIGHTS
from locascope.training.CoAE.Supply import PROBE_ANCHOR, Probe

#: the dataset of the validation photos (never trained on: DataConfig.forbidden)
VAL_DATASET: str = 'ki67_with_photo'
#: the pooled sides of 'latent2' / 'latent4'
LATENT_SIDES: Dict[str, int] = {'latent2': 2, 'latent4': 4}


def pool_latent(latent: torch.Tensor, side: int) -> torch.Tensor:
    """[B, s, s, L] -> [B, side * side * L]: the cells averaged down to side x side (`TF.adaptive_avg_pool2d`: s = 7 does not divide by 2
    or 4, its bins overlap), cells first, channels last."""
    return TF.adaptive_avg_pool2d(latent.permute(0, 3, 1, 2), side).permute(0, 2, 3, 1).flatten(1)


def tile_parts(o: CoaeOut, net: CoaeNet, scale: ChannelScale) -> Dict[str, torch.Tensor]:
    """Every part a usage may name (`ArmSpec.COAE_PARTS`), float32 [B, d]: ONE definition for the stage-2 tile encoder and the tile-level
    search. The descriptor parts are brought to one scale a channel by the CHECKPOINT's own running sd (no sample is taken):
        cls_fm      o.cls_fm                                                              [B, D]
        cls         o.cls                                                                 [B, cls_dim]
        place       (o.pooled / net.pool_sd).flatten(1)                                   [B, M c c]
        head        ((o.pooled - net.pool_mu) / net.pool_sd).flatten(1): place less the
                    common vector pool_mu / pool_sd (the training's own constant)          [B, M c c]
        latent      o.latent.flatten(1)                                                   [B, s s L]
        latent_avg  the mean over the s x s cells                                         [B, L]
        latent_dev  (o.latent - latent_avg).flatten(1): the arrangement alone             [B, s s L]
        latent2, latent4   pool_latent(o.latent, 2 | 4)                                   [B, 4 L], [B, 16 L]
        cofig       (cofig_point(o.cofig) / scale.sd).flatten(1)                          [B, N t t]"""
    cfg: CoaeConfig = net.cfg
    latent: torch.Tensor = o.latent.float()
    avg: torch.Tensor = latent.mean(dim=(1, 2), keepdim=True)
    parts: Dict[str, torch.Tensor] = {
        'cls_fm': o.cls_fm.float(),
        'cls': o.cls.float(),
        'place': (o.pooled.float() / net.pool_sd.view(1, -1, 1, 1)).flatten(1),
        'head': ((o.pooled.float() - net.pool_mu.view(1, -1, 1, 1)) / net.pool_sd.view(1, -1, 1, 1)).flatten(1),
        'latent': latent.flatten(1),
        'latent_avg': avg.flatten(1),
        'latent_dev': (latent - avg).flatten(1),
        'cofig': (cofig_point(o.cofig.float(), cfg.cofig, scale, cfg.head.output) / scale.sd.view(1, -1, 1, 1)).flatten(1)}
    for name, side in LATENT_SIDES.items():
        parts[name] = pool_latent(latent, side)
    return parts


def part_widths(net: CoaeNet) -> Dict[str, int]:
    """The width d of every part of `tile_parts` for this network."""
    cfg: CoaeConfig = net.cfg
    s: int = net.latent_side
    dim: int = cfg.model.latent_dim
    return {'cls_fm': net.fm.dim, 'cls': net.cls_dim, 'place': net.pool_shape[0] * net.pool_shape[1] ** 2,
            'head': net.pool_shape[0] * net.pool_shape[1] ** 2, 'latent': s * s * dim,
            'latent_avg': dim, 'latent_dev': s * s * dim, 'latent2': 4 * dim, 'latent4': 16 * dim,
            'cofig': n_channels(cfg.cofig) * cfg.cofig.out_cells ** 2}


def unit_feature(raw: Dict[str, torch.Tensor], parts: Tuple[str, ...]) -> torch.Tensor:
    """[B, d] unit: each part of `parts` made unit, joined, made unit again (the plain cosine: no mean is taken off)."""
    units: List[torch.Tensor] = [TF.normalize(raw[p].float(), dim=-1) for p in parts]
    return TF.normalize(torch.cat(units, -1), dim=-1)


def coae_of_checkpoint(path: str, device: torch.device) -> Tuple[CoaeNet, CoaeConfig, ChannelScale]:
    """The network, its config and the targets' running scale of a train.py checkpoint (`epoch<k>.pt`), net.eval(). A key the config
    classes no longer have raises TypeError; a checkpoint that does not fit the network raises ValueError (train.py's --resume rule:
    only the frozen FM's own weights may be absent)."""
    ck: Dict[str, Any] = torch.load(path, map_location='cpu', weights_only=False)
    d: Dict[str, Any] = ck['cfg']
    cfg: CoaeConfig = CoaeConfig(fm=FmConfig(**d['fm']), model=ModelConfig(**d['model']), cofig=CoFigureConfig(**d['cofig']),
                                 data=DataConfig(**d['data']), head=HeadConfig(**d['head']), train=TrainConfig(**d['train']),
                                 probe=ProbeConfig(**d['probe']))
    net: CoaeNet = CoaeNet(cfg, device, n_sources=1 + len(cfg.data.sims)).to(device)
    missing, unexpected = net.load_state_dict(ck['state'], strict=False)
    if unexpected or [k for k in missing if not k.startswith('fm.vit.')]:
        raise ValueError(f'{path} does not fit the network (missing {missing[:3]}, unexpected {unexpected[:3]})')
    scale: ChannelScale = ChannelScale(n_channels(cfg.cofig)).to(device)
    scale.load_state_dict(ck['scale'])
    net.eval()
    return net, cfg, scale


class CoaeTileEncoder:
    """A stage-2 tile encoder: uint8 tiles [N, T, T, 3] (a tensor, an array, or a list of arrays) -> unit vectors [N, d] float32 on the
    device: `unit_feature` of the usage `op` (`ArmSpec.COAE_PARTS`). float32 on purpose: the plain cosine of 'place' between places is
    1.00 / 0.99, all its information in the last two digits. The interface of `PhaseQueryEncoder`: `Context.encode_level` and
    `encode_query` call it."""

    def __init__(self, net: CoaeNet, scale: ChannelScale, device: torch.device, op: str, batch: int = 256) -> None:
        if op not in COAE_PARTS:
            raise ValueError(f'no CoAE usage {op!r}; they are {", ".join(COAE_PARTS)}')
        self.parts: Tuple[str, ...] = COAE_PARTS[op]
        self.net: CoaeNet = net
        self.scale: ChannelScale = scale
        self.device: torch.device = device
        self.batch: int = int(batch)
        self.feature_pooling: str = f'coae-{op}'

    def _outs(self, tiles: Any) -> Iterator[CoaeOut]:
        """The network's output for each batch of `tiles` (a tensor, an array, or a list of arrays: [N, T, T, 3] uint8), under autocast."""
        arr: torch.Tensor = tiles if isinstance(tiles, torch.Tensor) else torch.from_numpy(np.stack([np.ascontiguousarray(t) for t in tiles]))
        side: int = int(self.net.cfg.cofig.tile)
        if arr.ndim != 4 or tuple(arr.shape[1:]) != (side, side, 3):
            raise ValueError(f'CoaeTileEncoder takes tiles [N, {side}, {side}, 3], got {tuple(arr.shape)}')
        for s in range(0, len(arr), self.batch):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o: CoaeOut = self.net(arr[s:s + self.batch].to(self.device))
            yield o

    @torch.no_grad()
    def __call__(self, tiles: Any) -> torch.Tensor:
        out: List[torch.Tensor] = [unit_feature(tile_parts(o, self.net, self.scale), self.parts) for o in self._outs(tiles)]
        if out:
            return torch.cat(out)
        widths: Dict[str, int] = part_widths(self.net)
        return torch.zeros(0, sum(widths[p] for p in self.parts), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def cells(self, tiles: Any) -> torch.Tensor:
        """[N, s, s, L] float32 on the device: the latent cells as the network gives them (NOT made unit: a map is stitched from them, and a
        cell must weigh the same whichever tile it is read from)."""
        out: List[torch.Tensor] = [o.latent.float() for o in self._outs(tiles)]
        if out:
            return torch.cat(out)
        s: int = int(self.net.latent_side)
        return torch.zeros(0, s, s, int(self.net.cfg.model.latent_dim), dtype=torch.float32, device=self.device)

    features = __call__


@dataclass
class ValFov:
    slide: str
    photo: str
    level: int
    x0: float
    y0: float


def read_val_fovs(path: Path, limit: int) -> List[ValFov]:
    with open(path) as f:
        rows: List[Dict[str, str]] = list(csv.DictReader(f))
    return [ValFov(r['slide'], r['photo'], int(r['level']), float(r['x0']), float(r['y0'])) for r in rows][:limit]


@torch.no_grad()
def real_cofig(net: CoaeNet, fovs: List[ValFov], cfg: CoaeConfig, scale: ChannelScale, device: torch.device,
               batch: int = 64) -> Dict[str, Any]:
    """On the real validation photos (validation only: nothing is fitted or chosen here): the Co-figure the network predicts from a
    photo tile against the Co-figure computed from the photo itself, a signal (`Losses.agreement`: cos, scaled MSE). A photo is cut
    as a training view is: canvases of `photo_px` on a lattice a tile apart, the input the canvas's centre tile, the target the
    canvas's own Co-figure on the FM's field of that tile (`field_maps`); blank tiles (grey sd < `blank_sd`) left out.
    The photo's own px are used as they are (its scale to the scan is not known here, as it is not at search time)."""
    net.eval()
    c: CoFigureConfig = cfg.cofig
    px, tile = cfg.data.photo_px, c.tile
    t: int = c.out_cells
    grey_w: torch.Tensor = torch.tensor(GREY_WEIGHTS, device=device)
    points: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    photos: int = 0
    for v in fovs:
        img: np.ndarray = np.asarray(Image.open(v.photo).convert('RGB'))
        h, w = img.shape[:2]
        canv: List[np.ndarray] = [img[y:y + px, x:x + px] for y in range(0, h - px + 1, tile) for x in range(0, w - px + 1, tile)]
        photos += bool(canv)
        for s in range(0, len(canv), batch):
            rgb: torch.Tensor = torch.as_tensor(np.stack(canv[s:s + batch]), device=device)
            grey: torch.Tensor = rgb.float() @ grey_w
            o0: int = (px - tile) // 2
            tiles: torch.Tensor = rgb[:, o0:o0 + tile, o0:o0 + tile]
            keep: torch.Tensor = grey[:, o0:o0 + tile, o0:o0 + tile].flatten(1).std(1) >= cfg.data.blank_sd
            if not bool(keep.any()):
                continue
            target: torch.Tensor = field_maps(grey[keep], c, net.field_px)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = net(tiles[keep])
            points.append(cofig_point(o.cofig.float(), c, scale, cfg.head.output))
            targets.append(target)
    if not points:
        return {'tiles': 0, 'photos': photos}
    point: torch.Tensor = torch.cat(points)
    return {'tiles': len(point), 'photos': photos} | agreement(point, torch.cat(targets), torch.ones(len(point), t, t, dtype=torch.bool, device=device), scale, c)


# ── the probe (Supply.Probe): places of slides never trained on and without a real photo, each as fixed views ────────────────

#: the probe's CENTRED variant of every feature: the anchors' mean per channel taken off each part before the cosine
CENTRED: str = '-m'
#: a feature = one or two PARTS; a composite's parts are centred and made unit each on its own, then joined and made unit.
#: `direct` and the `cofig-direct*` features are computed from a view's own pixels by the TARGETS' functions, no network: what a
#: target is worth before any training
PROBE_FEATURES: Dict[str, Tuple[str, ...]] = {
    'fm': ('cls_fm',), 'fmavg': ('cls_fm', 'avg_fm'), 'cls': ('cls',), 'place': ('place',), 'fm+cls': ('cls_fm', 'cls'),
    'fm+place': ('cls_fm', 'place'), 'head': ('head',), 'fm+head': ('cls_fm', 'head'), 'latent': ('latent',), 'latent-avg': ('latent_avg',), 'cofig': ('cofig',),
    'direct': ('direct',), 'fm+direct': ('cls_fm', 'direct'),
    'cofig-direct': ('cofig_direct',), 'fm+cofig-direct': ('cls_fm', 'cofig_direct')}
#: a signal's own share of `cofig-direct`: the feature f'{SIGNAL_FEATURE}{part}' is the part f'cofig_direct:{part}' (part: CoFigure.part_names)
SIGNAL_FEATURE: str = 'cofig-direct:'
#: the features whose scores are also kept a cell (`probe_measures`' 'by_cell')
CELL_FEATURES: Tuple[str, ...] = ('fm', 'cls', 'head', 'place')


def pooled_sides(cfg: CoaeConfig) -> Tuple[int, ...]:
    """The sides the Co-figure is averaged down to, f'cofig-direct{g}': 2 and 4 (the coarse forms; 4 is `direct`'s cells) and the
    latent's own side (gigapath 7, uni2 8: the target as the latent's cells see it)."""
    return tuple(sorted({2, 4, cfg.latent_side}))


def probe_names(cfg: CoaeConfig) -> Dict[str, Tuple[str, ...]]:
    """PROBE_FEATURES, the pooled forms `pooled_sides` names and a feature a signal of the Co-figure: {name: its parts}."""
    return (dict(PROBE_FEATURES) | {f'cofig-direct{g}': (f'cofig_direct{g}',) for g in pooled_sides(cfg)}
            | {SIGNAL_FEATURE + n: (f'cofig_direct:{n}',) for n in part_names(cfg.cofig)})


def probe_groups(cfg: CoaeConfig) -> List[Tuple[str, List[str]]]:
    """The probe table's blocks (title, the features in column order); `fm` leads both branches as their reference."""
    return [('cls\' branch', ['fm', 'fmavg', 'cls', 'place', 'head', 'direct', 'fm+cls', 'fm+place', 'fm+head', 'fm+direct']),
            ('latent branch', ['fm', 'latent', 'latent-avg', 'cofig', 'cofig-direct'] + [f'cofig-direct{g}' for g in pooled_sides(cfg)]
             + ['fm+cofig-direct']),
            ('the decoder\'s target a signal, from the view\'s own pixels', [SIGNAL_FEATURE + n for n in part_names(cfg.cofig)])]


@torch.no_grad()
def probe_features(net: CoaeNet, probe: Probe, cfg: CoaeConfig, scale: ChannelScale, device: torch.device, batch: int = 256
                   ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """net.eval(); every view through the network (autocast bf16), kept on the device as float16:
    parts {'cls_fm', 'avg_fm', 'cls', 'latent' (the s x s x latent_dim cells flattened: gigapath 6272 values, uni2 8192, compared cell
    by cell; a cell is 32 px, so a 64 px shift is two cells), 'latent_avg' (the mean over the s x s cells: latent_dim values, no
    arrangement, tolerant of shifts but only latent_dim directions), 'place', 'cofig', 'direct', 'cofig_direct',
    'head' (the linear head's output in the checkpoint's own units, (o.pooled - net.pool_mu) / net.pool_sd, flattened: the same
    definition as the real-photo tile level's `coae head`; it differs from 'place' by a vector all places share),
    'cofig_direct:<signal>' (a signal's channels of it), 'cofig_direct<g>' (g in `pooled_sides`; the one at the latent's own side s is
    the untrained counterpart of 'latent'; neither latent column is the decoder's output, so `cofig-direct*` is no ceiling for them)}: [P, K, d]
    and the decoder's Co-figure in point form (Losses.cofig_point, `scale` the decoder's channel scale) [P, K, N, t, t].
    The descriptor parts are brought to one scale a channel by the PROBE's own sd, never by a running one:
        place_sd [M] = the sd of probe.direct[:, 0] a channel (over the anchors and the pool cells), clamp(min=1e-4)
        map_sd   [N] = the sd of probe.maps[:, 0] a channel (the anchors' Co-figures)
        cell_sd  [N] = the same over the anchors' g x g averaged cells, a side g of its own
        'place'  = (o.pooled / place_sd).flatten ; 'direct' = (probe.direct / place_sd).flatten ; 'cofig' = (point / map_sd).flatten
        'cofig_direct' = (probe.maps / map_sd).flatten (every view's own Co-figure, the scale of 'cofig');
        'cofig_direct<g>' = (the g x g cell mean of probe.maps / cell_sd).flatten"""
    net.eval()
    p, k = probe.tiles.shape[:2]
    flat: torch.Tensor = probe.tiles.flatten(0, 1)                                           # [P * K, tile, tile, 3]
    place_sd: torch.Tensor = probe.direct[:, 0].permute(1, 0, 2, 3).flatten(1).std(1).clamp(min=1e-4).to(device)       # [M]
    map_sd: torch.Tensor = probe.maps[:, 0].float().permute(1, 0, 2, 3).flatten(1).std(1).clamp(min=1e-6).to(device)  # [N]
    got: Dict[str, List[torch.Tensor]] = {n: [] for n in ('cls_fm', 'avg_fm', 'cls', 'latent', 'latent_avg', 'place', 'head', 'cofig')}
    points: List[torch.Tensor] = []
    for s in range(0, len(flat), batch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = net(flat[s:s + batch].to(device))
        pt: torch.Tensor = cofig_point(o.cofig.float(), cfg.cofig, scale, cfg.head.output)
        got['cls_fm'].append(o.cls_fm.half())
        got['avg_fm'].append(o.avg_fm.half())
        got['cls'].append(o.cls.float().half())
        got['latent'].append(o.latent.float().flatten(1).half())
        got['latent_avg'].append(o.latent.float().flatten(1, 2).mean(1).half())
        got['place'].append((o.pooled.float() / place_sd.view(1, -1, 1, 1)).flatten(1).half())
        got['head'].append(((o.pooled.float() - net.pool_mu.view(1, -1, 1, 1)) / net.pool_sd.view(1, -1, 1, 1)).flatten(1).half())
        got['cofig'].append((pt / map_sd.view(1, -1, 1, 1)).flatten(1).half())
        points.append(pt.half())
    parts: Dict[str, torch.Tensor] = {n: torch.cat(v).view(p, k, -1) for n, v in got.items()}
    parts['direct'] = (probe.direct.to(device) / place_sd.view(1, 1, -1, 1, 1)).flatten(2).half()
    maps: torch.Tensor = probe.maps.to(device)                                               # half [P, K, N, t, t]
    n, t = maps.shape[2], maps.shape[-1]
    own: torch.Tensor = maps.float() / map_sd.view(1, 1, -1, 1, 1)                           # the scale of 'cofig'
    parts['cofig_direct'] = own.flatten(2).half()
    for name, pt_ in zip(part_names(cfg.cofig), layout(cfg.cofig)):
        parts[f'cofig_direct:{name}'] = own[:, :, pt_.start:pt_.start + pt_.size].flatten(2).half()
    for g in pooled_sides(cfg):
        if t % g:
            raise ValueError(f'the Co-figure side {t} is not a multiple of the pooled side {g}')
        cells: torch.Tensor = TF.avg_pool2d(maps.float().flatten(0, 1), t // g).view(p, k, n, g, g)    # the UNscaled map averaged a g x g cell
        cell_sd: torch.Tensor = cells[:, 0].permute(1, 0, 2, 3).flatten(1).std(1).clamp(min=1e-4)      # [N]: a channel's sd over the anchors' cells
        parts[f'cofig_direct{g}'] = (cells / cell_sd.view(1, 1, -1, 1, 1)).flatten(2).half()
    return parts, torch.cat(points).view(p, k, *points[0].shape[1:])


def _unit(parts: List[torch.Tensor], centred: bool) -> torch.Tensor:
    """A feature's vectors [P, K, D]: the parts, each (centred: minus its mean over the ANCHORS; then) unit, joined, unit."""
    units: List[torch.Tensor] = []
    for x in parts:
        y: torch.Tensor = x.float()
        if centred:
            y = y - y[:, 0].mean(0)
        units.append(TF.normalize(y, dim=-1))
    return TF.normalize(torch.cat(units, -1), dim=-1)


def anchor_spread(parts: List[torch.Tensor], cell: torch.Tensor, centred: bool) -> Dict[str, float]:
    """How far apart the ANCHORS of a cell are in one feature (f as in probe_scores): for each anchor, over the OTHER anchors of its cell
        nn    the largest cosine  -> the median over the anchors   (1 - nn: the cosine a query may lose before another place wins)
        neg   the mean cosine     -> the median                     (probe_scores' neg_a)
    Returns {'nn', 'neg', 'n'}; cells of fewer than 2 places left out."""
    f: torch.Tensor = _unit(parts, centred)
    nn: List[torch.Tensor] = []
    neg: List[torch.Tensor] = []
    for c in torch.unique(cell).tolist():
        idx: torch.Tensor = (cell == c).nonzero().flatten().to(f.device)
        n: int = len(idx)
        if n < 2:
            continue
        a: torch.Tensor = f[idx, 0]                                                           # [n, D]
        off: torch.Tensor = (a @ a.t())[~torch.eye(n, dtype=torch.bool, device=f.device)].view(n, n - 1)
        nn.append(off.max(1).values)
        neg.append(off.mean(1))
    return {'nn': float(torch.cat(nn).median()), 'neg': float(torch.cat(neg).median()), 'n': float(len(torch.cat(nn)))}


def _query_scores(parts: List[torch.Tensor], cell: torch.Tensor, kinds: List[str], centred: bool) -> Dict[str, Dict[str, torch.Tensor]]:
    """{kind: {'hit', 'consumed', 'z', 'spread', 'cell'}} for every kind but the anchor: a value a query (the cells of >= 2 places, in
    the order of torch.unique(cell); 'cell' = the query's cell, its ordinal among those cells). As probe_scores' doc;
    spread = (1 - mean(off)) / sd(off).clamp(min=1e-9): how many sd the wrong anchors lie from identical
    (z = (1 - consumed') x spread a query, consumed' against the query's own off-mean)."""
    f: torch.Tensor = _unit(parts, centred)                                                   # [P, K, D]
    out: Dict[str, Dict[str, List[torch.Tensor]]] = {kd: {'hit': [], 'consumed': [], 'z': [], 'spread': [], 'cell': []} for kd in kinds[1:]}
    ordinal: int = 0
    for c in torch.unique(cell).tolist():
        idx: torch.Tensor = (cell == c).nonzero().flatten().to(f.device)
        n: int = len(idx)
        if n < 2:
            continue
        a: torch.Tensor = f[idx, 0]                                                           # [n, D]
        g: torch.Tensor = a @ a.t()
        neg_a: torch.Tensor = (g.sum(1) - g.diagonal()) / (n - 1)
        eye: torch.Tensor = torch.eye(n, dtype=torch.bool, device=f.device)
        for j in range(1, len(kinds)):
            s: torch.Tensor = f[idx, j] @ a.t()                                               # [n, n]: a query (row) against every anchor
            pos: torch.Tensor = s.diagonal()
            off: torch.Tensor = s[~eye].view(n, n - 1)
            sd: torch.Tensor = off.std(1).clamp(min=1e-9)
            q: Dict[str, List[torch.Tensor]] = out[kinds[j]]
            q['hit'].append((pos > off.max(1).values).float())
            q['consumed'].append((1 - pos) / (1 - neg_a).clamp(min=1e-9))
            q['z'].append((pos - off.mean(1)) / sd)
            q['spread'].append((1 - off.mean(1)) / sd)
            q['cell'].append(torch.full((n,), ordinal, dtype=torch.long, device=f.device))
        ordinal += 1
    return {kd: {k: torch.cat(v) for k, v in q.items()} for kd, q in out.items() if q['hit']}


def probe_scores(parts: List[torch.Tensor], cell: torch.Tensor, kinds: List[str], centred: bool) -> Dict[str, Dict[str, float]]:
    """One feature's scores a view kind. f = the parts, each (centred: minus its mean over the ANCHORS; then) unit, joined, unit.
    In each cell (n places): a = f[cell, anchor] [n, D]; neg_a = the mean of a @ a.T off its diagonal, a row.
    For a kind j: s = f[cell, j] @ a.T [n, n]; pos = its diagonal; off = the rest of each row
        hit       pos > max(off)                                    -> the mean over all queries
        consumed  (1 - pos) / (1 - neg_a).clamp(min=1e-9)           -> the median
        z         (pos - mean(off)) / sd(off).clamp(min=1e-9)       -> the median
        spread    (1 - mean(off)) / sd(off).clamp(min=1e-9)         -> the median: how many sd the wrong anchors lie from identical
    Returns {kind: {'hit', 'consumed', 'z', 'spread', 'n'}} for every kind but the anchor."""
    out: Dict[str, Dict[str, float]] = {}
    for kd, q in _query_scores(parts, cell, kinds, centred).items():
        out[kd] = {'hit': float(q['hit'].mean()), 'consumed': float(q['consumed'].median()), 'z': float(q['z'].median()),
                   'spread': float(q['spread'].median()), 'n': float(len(q['hit']))}
    return out


def probe_cell_scores(parts: List[torch.Tensor], cell: torch.Tensor, kinds: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """One feature's scores a cell, as it is (not centred): {kind: {'hit', 'consumed', 'z', 'spread': [a number a cell]}} for every
    kind but the anchor; hit the mean over the cell's queries, the rest the median; the cells of >= 2 places, in the order of
    torch.unique(cell) (for resampling the cells)."""
    out: Dict[str, Dict[str, List[float]]] = {}
    for kd, q in _query_scores(parts, cell, kinds, False).items():
        n_cells: int = int(q['cell'].max()) + 1
        out[kd] = {k: [float(q[k][q['cell'] == i].mean() if k == 'hit' else q[k][q['cell'] == i].median()) for i in range(n_cells)]
                   for k in ('hit', 'consumed', 'z', 'spread')}
    return out


def probe_cell_names(probe: Probe) -> List[str]:
    """The names of the cells `probe_cell_scores` has a number for (the cells of >= 2 places), in its order (torch.unique(cell))."""
    ids, counts = torch.unique(probe.cell, return_counts=True)
    return [probe.cells[c] for c, n in zip(ids.tolist(), counts.tolist()) if n >= 2]


@torch.no_grad()
def probe_measures(net: CoaeNet, probe: Probe, cfg: CoaeConfig, scale: ChannelScale, device: torch.device) -> Dict[str, Any]:
    """{'features': {name: {kind: {...}}} for every `probe_names` name and name + CENTRED,
        'anchors': {name: {'nn', 'neg', 'n'}} for the same names (`anchor_spread`),
        'map':  {kind: {part: cos}} for the anchor and every 'twin:*' kind -- the decoder's Co-figure of that view against the
                ANCHOR's own (Losses.agreement's cos, every cell valid),
        'map_direct': {kind: {part: cos}} for every 'twin:*' kind -- the view's OWN Co-figure (probe.maps, no network) against the
                anchor's own: the untrained baseline of 'map',
        'by_cell': {name: {kind: {'hit', 'consumed', 'z', 'spread': [a number a cell]}}} for the `CELL_FEATURES`, as it is (not
                centred; `probe_cell_scores`: for resampling the cells), 'cell_names': the cells' names in that order,
        'groups': [[title, [feature, ...]], ...] the probe table's blocks (`probe_groups`),
        'rank': the effective rank of the anchors' unit cls (exp of the entropy of the normalised singular values),
        'places', 'cells', 'seconds'}"""
    t0: float = time.perf_counter()
    parts, point = probe_features(net, probe, cfg, scale, device)
    names: Dict[str, Tuple[str, ...]] = probe_names(cfg)
    groups: List[Tuple[str, List[str]]] = probe_groups(cfg)
    if {x for _, g_ in groups for x in g_} != set(names):
        raise ValueError(f'the probe table\'s blocks {sorted({x for _, g_ in groups for x in g_})} are not the probe\'s features {sorted(names)}')
    features: Dict[str, Any] = {}
    anchors: Dict[str, Dict[str, float]] = {}
    for name, keys in names.items():
        features[name] = probe_scores([parts[kd] for kd in keys], probe.cell, probe.kinds, False)
        features[name + CENTRED] = probe_scores([parts[kd] for kd in keys], probe.cell, probe.kinds, True)
        anchors[name] = anchor_spread([parts[kd] for kd in keys], probe.cell, False)
        anchors[name + CENTRED] = anchor_spread([parts[kd] for kd in keys], probe.cell, True)
    target: torch.Tensor = probe.maps[:, 0].float().to(device)
    valid: torch.Tensor = torch.ones(len(target), target.shape[-2], target.shape[-1], dtype=torch.bool, device=device)
    maps: Dict[str, Dict[str, float]] = {}
    map_direct: Dict[str, Dict[str, float]] = {}
    for j, kd in enumerate(probe.kinds):
        if kd == PROBE_ANCHOR or kd.startswith('twin:'):
            maps[kd] = {part: v['cos'] for part, v in agreement(point[:, j].float(), target, valid, scale, cfg.cofig).items()}
        if kd.startswith('twin:'):
            map_direct[kd] = {part: v['cos'] for part, v in agreement(probe.maps[:, j].float().to(device), target, valid, scale, cfg.cofig).items()}
    u: torch.Tensor = TF.normalize(parts['cls'][:, 0].float(), dim=1)
    sv: torch.Tensor = torch.linalg.svdvals(u - u.mean(0))
    pr: torch.Tensor = sv / sv.sum().clamp(min=1e-12)
    rank: float = float(torch.exp(-(pr * torch.log(pr.clamp(min=1e-12))).sum()))
    return {'features': features, 'anchors': anchors, 'map': maps, 'map_direct': map_direct,
            'by_cell': {name: probe_cell_scores([parts[kd] for kd in names[name]], probe.cell, probe.kinds) for name in CELL_FEATURES},
            'cell_names': probe_cell_names(probe),
            'groups': [[title, g_] for title, g_ in groups], 'rank': rank, 'places': len(probe.cell), 'cells': len(probe.cells),
            'seconds': time.perf_counter() - t0}
