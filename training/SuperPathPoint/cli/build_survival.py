#!/usr/bin/env python3
"""spec.md 3.2: build the survival table, both axes, from the chains.

    python training/SuperPathPoint/cli/build_survival.py --checkpoint <.pt>

Outputs (in result/<SLURM_JOB_NAME or BuildSurvival>/):
    <slide>__F__t256.safetensors     one file per (slide, axis)
    <slide>__R__t256.safetensors

argparse, wiring and printed progress. Everything that decides anything is in
`PointsAnalysisByMpp/`.

THE FIRST RUN IS A CALIBRATION RUN, NOT A RESULT
==================================================
`tau_alpha` starts at 1.5 and nobody has measured it (spec.md 3.2 配對,
ClaudeRules 8). What comes out of run one is the match-rate-against-tau curve
and its decoy, and the value is chosen from the knee. An attribution split
quoted from run one is quoting a guess.

WHAT THIS WRITES AND WHAT IT DELIBERATELY DOES NOT
====================================================
It writes `score`, `dist` and `suppressed_by_*`. It does NOT write `alive` or
`born_rung`, because both are functions of a threshold and a tau that the
reader chooses: storing them freezes two guesses into a file that costs hours to
rebuild, when re-cutting is a comparison on arrays already in memory
(`Patterns.alive_from`).

BOTH AXES COME OUT OF ONE PASS OVER THE CHAINS
================================================
The 'R' stack is DERIVED from each chain's ds 1 tile rather than extracted
(`MppStack.r_stack`), so the two axes share a centre by construction. That is
not a saving, it is the requirement: 新生歸因 asks whether a point born late on
'F' is also born late on 'R', and that question only has an answer if the two
axes are about the same physical point.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..', '..', 'utilities'),
           os.path.join(_HERE, '..')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _paths import RESULT_DIR, job_result_dir, setup_import_paths  # noqa: E402

setup_import_paths()

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

from DsLadder import DEFAULT_RUNGS                            # noqa: E402
from PointsAnalysisByMpp import MppStack                      # noqa: E402
from PointsAnalysisByMpp import SurvivalProcess               # noqa: E402
from PointsAnalysisByMpp import SurvivalTable                 # noqa: E402

# The three shape readers and the strict load already exist and are tested
# (`test_reeval_density`, section `rebuild`). Importing them beats a third copy:
# a checkpoint rebuilt by two rules is two models that answer to one name.
sys.path.insert(0, _HERE)
from reeval_density import load_arm                           # noqa: E402

#: THE CHAIN CORPUS, not the training one. `cache/tiles` holds the 2026-08-27
#: extraction, which has `inherit_id = -1` on every row -- no chains, nothing
#: for a stack to be built from. The chain corpus is a separate root so that
#: the two can never be read as one (`Datasets` refuses two stores for a
#: (slide, rung), and a separate root means it never has to).
DEFAULT_TILE_ROOT = os.path.join(RESULT_DIR, 'cache', 'tiles_chains')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tiles-root', default=DEFAULT_TILE_ROOT)
    ap.add_argument('--checkpoint', required=True,
                    help='the detector. Its identity_id goes in every table, '
                         'because a table that cannot name its detector is '
                         'indistinguishable from the round before it')
    ap.add_argument('--wsi-stem', nargs='*', default=None,
                    help='default: every slide with complete chains')
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--ds', type=float, nargs='+', default=list(DEFAULT_RUNGS))
    ap.add_argument('--stack-kind', nargs='+', default=['F', 'R'],
                    choices=('F', 'R'))
    ap.add_argument('--score-threshold', type=float, default=0.001,
                    help='the PERMISSIVE cut the store is written at. Every '
                         'later question re-cuts at some higher value on these '
                         'arrays; a question below it needs a re-run')
    ap.add_argument('--decoy-alpha', type=float, default=8.0,
                    help='how far the decoy probe sits from the anchor, as a '
                         'multiple of ds. Far enough to be a different place '
                         '-- outside any tau worth choosing -- and near enough '
                         'to be the same tissue, because a decoy on glass is '
                         'trivially beaten and says nothing')
    ap.add_argument('--tau-alpha', type=float, default=1.5,
                    help='tau = alpha * ds level-0 px. CALIBRATION, not a '
                         'setting: run one measures the curve, run two picks '
                         'from its knee (ClaudeRules 8)')
    ap.add_argument('--tau-floor-um', type=float, default=0.0)
    ap.add_argument('--mpp-0', type=float, default=0.0)
    ap.add_argument('--limit-chains', type=int, default=0,
                    help='0 = all. A small number makes this a smoke run')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    out_dir = args.out or job_result_dir('BuildSurvival')
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net, identity, _ = load_arm(args.checkpoint, device)
    print(f'detector {args.checkpoint}\n  {net.identity_id()}  '
          f'{net.cfg.backbone.in_channels} ch  init '
          f'{identity.get("init", "?")}', flush=True)
    print(f'  max_keypoints is OFF for this run by construction: '
          f'SurvivalProcess never passes one. A cap is global competition we '
          f'imposed, and it is indistinguishable from a neighbourhood killing '
          f'the point (spec.md 3.2).', flush=True)

    stems = args.wsi_stem or _stems_under(args.tiles_root, args.tile)
    rungs = sorted(float(d) for d in args.ds)

    for stem in stems:
        found = MppStack.chains(args.tiles_root, stem, tile=args.tile,
                                rungs=rungs)
        if not found:
            print(f'\n[{stem}] no complete chain over {rungs}; skipped',
                  flush=True)
            continue
        ids = sorted(found)
        if args.limit_chains:
            ids = ids[:int(args.limit_chains)]
        print(f'\n[{stem}] {len(ids)} chains x {len(rungs)} rungs', flush=True)

        for kind in args.stack_kind:
            batch = _one_axis(found, ids, net, kind, rungs, args)
            meta = SurvivalTable.SurvivalMeta(
                wsi_stem=stem, stack_kind=kind, tile=int(args.tile),
                rungs=tuple(rungs), detector_id=net.identity_id(),
                detector_path=args.checkpoint,
                score_threshold=float(args.score_threshold),
                tau_alpha=float(args.tau_alpha),
                decoy_alpha=float(args.decoy_alpha),
                tau_floor_um=float(args.tau_floor_um),
                mpp_0=float(args.mpp_0), n_chains=len(ids))
            path = SurvivalTable.save(out_dir, batch, meta)
            print(f'  {kind}  {len(batch):6d} points  -> {os.path.basename(path)}',
                  flush=True)

    print(f'\nSaved to {out_dir}')
    print('  Run one is a CALIBRATION run: read the tau curve, not the '
          'attribution split.')
    return 0


def _one_axis(found, ids, net, kind, rungs, args):
    """Every chain of one slide on one axis, concatenated into one batch."""
    decoy = np.array([args.decoy_alpha * MppStack.rung_shrink(d, kind)
                      for d in rungs], np.float64)

    parts, chain_ids = [], []
    for k, cid in enumerate(ids):
        chain = found[cid]
        if kind == 'F':
            stack = MppStack.f_stack(chain, tile=args.tile)
            origins = {d: (float(chain.members[d][1].x),
                           float(chain.members[d][1].y)) for d in rungs}
        else:
            stack = MppStack.r_stack(chain, rungs, tile=args.tile)
            # ONE origin for every rung: an 'R' rung's footprint is the ds 1
            # tile's, degraded in place. The frame never moves, so the corner
            # never moves either -- see `MppStack.rung_scale`.
            base = chain.members[1.0][1]
            origins = {d: (float(base.x), float(base.y)) for d in rungs}

        # TWO DIFFERENT QUANTITIES, both derived from `ds` and equal on one of
        # the two axes: `rung_scale` is level-0 px per output PIXEL (the
        # mapping) and `rung_shrink` is how far a position can be off (the
        # tolerance, above). They agree on 'F' and disagree on 'R'.
        scales = {d: MppStack.rung_scale(d, kind) for d in rungs}
        columns = SurvivalProcess.run(
            stack, net, rungs=rungs, origins=origins, scales=scales,
            decoy_offset=decoy, score_threshold=args.score_threshold)
        parts.append(columns)
        chain_ids.append(np.full(len(columns['x0']), cid, np.int32))
        if (k + 1) % 50 == 0 or k + 1 == len(ids):
            print(f'    {kind}  {k + 1}/{len(ids)} chains', flush=True)

    return SurvivalTable.SurvivalBatch(
        x0=np.concatenate([p['x0'] for p in parts]),
        y0=np.concatenate([p['y0'] for p in parts]),
        chain=np.concatenate(chain_ids) if chain_ids else np.zeros(0, np.int32),
        score=np.concatenate([p['score'] for p in parts]),
        dist=np.concatenate([p['dist'] for p in parts]),
        suppressed_by_score=np.concatenate(
            [p['suppressed_by_score'] for p in parts]),
        suppressed_by_dist=np.concatenate(
            [p['suppressed_by_dist'] for p in parts]),
        decoy_score=np.concatenate([p['decoy_score'] for p in parts]),
        decoy_dist=np.concatenate([p['decoy_dist'] for p in parts]))


def _stems_under(tiles_root, tile):
    import PreTileStore                                       # noqa: PLC0415
    stems = set()
    for folder in PreTileStore.find(tiles_root, tile=int(tile)):
        stems.add(PreTileStore.load_meta(folder).wsi_stem)
    return sorted(stems)


if __name__ == '__main__':
    sys.exit(main())
