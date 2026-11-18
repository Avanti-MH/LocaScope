"""The chosen checkpoint's one-off checks, before and after its real validation (RealTest: `--stage2 arm:GigaCoAE+RP3:SlidingWindow::<usage>+mean`).

    python -m locascope.training.CoAE.final probe --checkpoint <epoch2.pt> --out <dir> [--probe-slides N] [--probe-places N] [--workers 4]
    python -m locascope.training.CoAE.final cofig --checkpoint <epoch2.pt> --fovs <val_fovs.csv> --out <dir>

probe  reads NO real photo. The checkpoint's own probe (its config's: places of slides never trained on, each as the anchor and its
       fixed views), the plain cosine only:
         (a) the probe's rows of every usage of `ArmSpec.COAE_PARTS` ('head', 'fm_head' among them) (hit/consumed a view
             kind, `anchors nn/neg` on top), printed and written to <out>/probe.txt; also <out>/probe.json: the usages' `probe_scores`
             ('features'), the cell-wise scores of `CELL_FEATURES` ('by_cell', `probe_cell_scores`) and the cells' names ('cell_names'); 'place' and 'cofig' are scaled by the PROBE's own sd, as in training's probe;
         (b) the stage-2 tile encoder (`Validate.CoaeTileEncoder`) against the probe's own parts: for the usages that need no sd of
             their own, the cosine of its vectors with `Validate.unit_feature` of the probe's parts must be >= 0.999;
         (c) 'place' and 'cofig' as the tile encoder makes them (scaled by the checkpoint's running sd, `net.pool_sd` and
             `scale.sd`), their rows beside the probe's own columns; the hit of a view kind of 'place' may differ by `--place-tol` at
             most ('cofig' is only printed).
       A failed check exits non-zero.
cofig  the real validation photos (`Validate.read_val_fovs`): the predicted Co-figure against the one computed from the photo itself;
       <out>/real_cofig.json. Reads real photos: only after the RealTest arms.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from locascope.core.stage2_retrieval.ArmSpec import COAE_PARTS
from locascope.training.CoAE import LogLayout as LL
from locascope.training.CoAE.Losses import ChannelScale
from locascope.training.CoAE.Model import CoaeNet
from locascope.training.CoAE.Supply import Probe, Supply
from locascope.training.CoAE.Validate import (CELL_FEATURES, LATENT_SIDES, CoaeTileEncoder, anchor_spread, coae_of_checkpoint, pool_latent,
                                              probe_cell_names, probe_cell_scores, probe_features, probe_names, probe_scores, read_val_fovs,
                                              real_cofig, unit_feature)
from locascope.training.CoAE.train import given

#: the usages (b) compares: each part of them is scaled alike in the probe and in the tile encoder
CHECKED_OPS: Tuple[str, ...] = ('cls', 'latent', 'latent_avg', 'cls_latent', 'fm_cls')
#: the cosine of the tile encoder's vectors with the probe's own, at least
MIN_COS: float = 0.999
#: the places (b) runs, at most
CHECK_PLACES: int = 256
Scores = Dict[str, Dict[str, Dict[str, float]]]


def say(text: str) -> None:
    print(text, flush=True)


def probe_parts(parts: Dict[str, torch.Tensor], net: CoaeNet) -> Dict[str, torch.Tensor]:
    """The probe's parts [P, K, d] of every part a usage names: those `probe_features` made (its own sd for 'place' and 'cofig') and the
    three it does not make, from its 'latent' (the s x s x latent_dim cells flat): 'latent_dev', 'latent2', 'latent4'."""
    p, k = parts['latent'].shape[:2]
    lat: torch.Tensor = parts['latent'].float().view(p * k, net.latent_side, net.latent_side, net.cfg.model.latent_dim)
    out: Dict[str, torch.Tensor] = {n: parts[n] for n in ('cls_fm', 'cls', 'place', 'head', 'latent', 'latent_avg', 'cofig')}
    out['latent_dev'] = (lat - lat.mean(dim=(1, 2), keepdim=True)).flatten(1).view(p, k, -1)
    for name, side in LATENT_SIDES.items():
        out[name] = pool_latent(lat, side).view(p, k, -1)
    return out


def usage_scores(pp: Dict[str, torch.Tensor], probe: Probe) -> Tuple[Scores, Dict[str, Dict[str, float]]]:
    """(the probe's `probe_scores`, `anchor_spread`) of every usage of COAE_PARTS ('head' and 'fm_head' among them), the plain cosine."""
    scores: Scores = {}
    spread: Dict[str, Dict[str, float]] = {}
    for op, names in COAE_PARTS.items():
        scores[op] = probe_scores([pp[n] for n in names], probe.cell, probe.kinds, False)
        spread[op] = anchor_spread([pp[n] for n in names], probe.cell, False)
    return scores, spread


def table(title: str, scores: Scores, spread: Dict[str, Dict[str, float]], kinds: List[str]) -> str:
    """A block: a column a feature of `scores`, a row a view kind (the anchor's left out), a cell hit/consumed; the first row `anchors nn/neg`."""
    cols: List[str] = list(scores)
    w: int = max(13, 2 + max(len(c) for c in cols))
    lines: List[str] = [f'{title}: hit/consumed (first row: the anchors among themselves, nearest/mean cosine)',
                        f'{"":<15}' + ''.join(f'{c:>{w}}' for c in cols)]
    if spread:
        lines.append(f'{"anchors nn/neg":<15}' + ''.join(f'{spread[c]["nn"]:.2f}/{spread[c]["neg"]:.2f}'.rjust(w) for c in cols))
    for kd in kinds[1:]:
        lines.append(f'{kd:<15}' + ''.join((f'{scores[c][kd]["hit"]:.2f}/{scores[c][kd]["consumed"]:.2f}' if kd in scores[c] else '-').rjust(w)
                                           for c in cols))
    return '\n'.join(lines)


def check_encoder(net: CoaeNet, scale: ChannelScale, pp: Dict[str, torch.Tensor], probe: Probe, device: torch.device) -> Tuple[List[str], float]:
    """(b): for each of CHECKED_OPS, the first `CHECK_PLACES` places and the views (the anchor, 'both:ood-heavy' when the probe has it, else
    its last): the cosine of CoaeTileEncoder's vectors with `unit_feature` of the probe's parts. (the lines, the smallest cosine)."""
    n: int = min(CHECK_PLACES, len(probe.cell))
    views: List[int] = sorted({0, probe.kinds.index('both:ood-heavy') if 'both:ood-heavy' in probe.kinds else len(probe.kinds) - 1})
    lines: List[str] = []
    worst: float = 1.0
    for op in CHECKED_OPS:
        enc: CoaeTileEncoder = CoaeTileEncoder(net, scale, device, op)
        for j in views:
            got: torch.Tensor = enc(probe.tiles[:n, j]).float()
            want: torch.Tensor = unit_feature({k: pp[k][:n, j].float().to(device) for k in COAE_PARTS[op]}, COAE_PARTS[op])
            cos: float = float((got * want).sum(1).min())
            worst = min(worst, cos)
            lines.append(f'  tile encoder {op:<12} view {probe.kinds[j]:<16} smallest cosine to the probe\'s own {cos:.5f}')
    return lines, worst


def encoder_scores(net: CoaeNet, scale: ChannelScale, probe: Probe, device: torch.device, ops: Tuple[str, ...]) -> Scores:
    """The probe's `probe_scores` of `ops` as the tile encoder makes them (the checkpoint's running sd, no probe sample)."""
    p, k = probe.tiles.shape[:2]
    out: Scores = {}
    for op in ops:
        f: torch.Tensor = CoaeTileEncoder(net, scale, device, op)(probe.tiles.flatten(0, 1)).view(p, k, -1)
        out[op] = probe_scores([f], probe.cell, probe.kinds, False)
    return out


def run_probe(a: argparse.Namespace, device: torch.device) -> None:
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError('torch.backends.cuda.matmul.allow_tf32 is on: the cosines of a CoAE feature would keep 10 mantissa bits, '
                           'too few for the places\' differences (float32 is the contract)')
    net, cfg, scale = coae_of_checkpoint(a.checkpoint, device)
    say(f'checkpoint {a.checkpoint}: FM weights {cfg.fm.weights_dtype}')
    cfg = replace(cfg, probe=replace(cfg.probe, **given(slides=a.probe_slides, places=a.probe_places)))   # only a Smoke job gives them
    val: Supply = Supply(replace(cfg, data=replace(cfg.data, dataset=cfg.probe.dataset, slides=cfg.probe.slides)), 0, 1, device,
                         workers=a.workers, say=say, pool=False)                                           # as train.py's probe
    probe: Probe = val.probe_set(cfg, net.field_px, device)
    val.close()
    say(f'probe: {len(probe.cell)} places in {len(probe.cells)} cells, views {probe.kinds}')
    parts, _ = probe_features(net, probe, cfg, scale, device)                                              # the probe's own parts, its own scales
    pp: Dict[str, torch.Tensor] = probe_parts(parts, net)
    scores, spread = usage_scores(pp, probe)
    enc: Scores = encoder_scores(net, scale, probe, device, ('place', 'cofig'))
    beside: Scores = {}
    for op in ('place', 'cofig'):
        beside[f'{op} (probe sd)'] = scores[op]
        beside[f'{op} (net sd)'] = enc[op]
    check_lines, worst = check_encoder(net, scale, pp, probe, device)
    gap: float = max((abs(scores['place'][kd]['hit'] - enc['place'][kd]['hit']) for kd in scores['place'] if kd in enc['place']), default=0.0)
    text: str = '\n'.join([table('(a) probe, the cosine as it is, every usage of COAE_PARTS', scores, spread, probe.kinds), '',
                           table('(c) place and cofig: the probe\'s sd against the checkpoint\'s running sd (what the tile encoder uses)',
                                 beside, {}, probe.kinds), f'  largest |hit difference| between the two sds: {gap:.3f} (limit {a.place_tol})', '',
                           '(b) the tile encoder against the probe\'s own parts:'] + check_lines)
    say(text)
    out: Path = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'probe.txt').write_text(text + '\n')
    names: Dict[str, Tuple[str, ...]] = probe_names(cfg)
    (out / 'probe.json').write_text(json.dumps({
        'features': scores,
        'by_cell': {name: probe_cell_scores([parts[kd] for kd in names[name]], probe.cell, probe.kinds) for name in CELL_FEATURES},
        'cell_names': probe_cell_names(probe)}))
    if worst < MIN_COS:
        raise RuntimeError(f'(b) the tile encoder is not the probed feature: smallest cosine {worst:.5f} < {MIN_COS}')
    if gap > a.place_tol:
        raise RuntimeError(f'(c) place / cofig by the checkpoint\'s running sd differ from the probe\'s by {gap:.3f} in hit (> {a.place_tol}): '
                           f'the stage-2 feature is not the one the probe measured')
    say(f'(b) smallest cosine {worst:.5f} >= {MIN_COS}; (c) largest hit difference {gap:.3f} <= {a.place_tol}')


def run_cofig(a: argparse.Namespace, device: torch.device) -> None:
    net, cfg, scale = coae_of_checkpoint(a.checkpoint, device)
    say(f'checkpoint {a.checkpoint}: FM weights {cfg.fm.weights_dtype}')
    res: Dict[str, Any] = real_cofig(net, read_val_fovs(Path(a.fovs), 10 ** 9), cfg, scale, device)
    say(LL.cofig_line('real photos', res))
    out: Path = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'real_cofig.json').write_text(json.dumps(res))


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('probe', help='the checkpoint\'s probe: every usage, the tile encoder against it (no real photo)')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True, help='the directory probe.txt is written to')
    p.add_argument('--probe-slides', type=int, default=None, help='only a Smoke job gives it (the checkpoint\'s own probe config otherwise)')
    p.add_argument('--probe-places', type=int, default=None, help='only a Smoke job gives it')
    p.add_argument('--workers', type=int, default=4, help='reader threads of the probe\'s Supply')
    p.add_argument('--place-tol', type=float, default=0.03, help='(c): the largest hit difference between the two sds (a Smoke probe is too small for 0.03)')
    c = sub.add_parser('cofig', help='the Co-figure on the real validation photos')
    c.add_argument('--checkpoint', required=True)
    c.add_argument('--fovs', required=True, help='the val_fovs CSV (slide, photo, level, x0, y0)')
    c.add_argument('--out', required=True, help='the directory real_cofig.json is written to')
    return ap.parse_args()


def main() -> None:
    a = parse()
    torch.cuda.set_device(0)
    device: torch.device = torch.device('cuda', 0)
    {'probe': run_probe, 'cofig': run_cofig}[a.cmd](a, device)


if __name__ == '__main__':
    sys.exit(main())
