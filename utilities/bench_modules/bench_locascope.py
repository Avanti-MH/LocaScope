#!/usr/bin/env python3
"""End-to-end LocaScope bench: synthetic FoVs with known positions through the
three stages, every stage's output and by-products kept as cache entries.

    python utilities/bench_modules/bench_locascope.py \\
        --datasets bracs/test ki67_with_photo --split val --n-wsi 10 \\
        --stage1 knn:gigapath --stage2 slidewin:gigapath --stage3 sift:default \\
        --route both

SHOTS: the first --n-wsi slides of each dataset's recorded --split; per slide
one draw over the --fov recipe's levels on the hest masks of
--fov-mask-cache-job, photographed through `FovSupply.cached` (draw and
render in --draw-cache-job / --render-cache-job, default this job).

STAGES: each one is `<method>:<recipe>` from its package's METHODS table, and
every field of the recipe can be replaced by a `--stageN-<field>` flag (the
encoder, the masks and the bank draw are not flags; they are the recipe's or
--seg's). A stage's results for a slide are ONE cache entry, written once the
slide is done (`Cache.Entry.writing`), under this job's tree:

    .../render=<gap>/stage1/{output,neighbours,probs,votes,prototypes_index}_<s1>.csv
    .../render=<gap>/stage1=<s1|oracle>/stage2/
            {output,tile_sims,truth,truth_sim}_<s2>.csv
    .../render=<gap>/stage1=<s1|oracle>/stage2=<s2>/stage3/
            {output,matches}_<s3>.csv

`output` is each stage's output interface, row for row: stage 1's
`EstMppResult`, stage 2's `CandidateSet` (one row per window, with its level
and ds), stage 3's `SiftRansacResult` per verified rank. The rest is what the
stage has beside it: a KNN's neighbours, a voting method's probs and votes,
a prototype method's support tiles (`prototypes_index`, one row per tile as the
bank draw's index holds it -- not per FoV),
the per-tile cosines of every output window (`tile_sims`), the truth window
and its per-tile cosines (`truth`, `truth_sim`), stage 3's point pairs.

`<sN>` is `<method>-<recipe>-<16 hex>`: the hex is the config's identity (for
stage 2 together with the mask it searches), so an overridden field is another
hex under the same label. Every table has an `index` column, the FoV's row in
the draw and in the render CSV, which holds its ground truth. A stage whose
entry is a hit is read back and its model never loaded; a slide whose entries
all hit is not rendered.

--route stage1 routes stage 2 to stage 1's level, oracle to the level the FoV
was placed at (`stage1=oracle/`), both runs stages 2 and 3 once per route.

Nothing here scores or draws. `utilities/cli/plot/plot_locascope.py` joins the
render CSV and these tables and computes every error from them.

The tables, the loop that fills them and the helpers other tools import are in
BenchCommon.py; this file is the synthetic-FoV entry point (the shots, their
truth, the two routes).
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # utilities/
import _paths                                                       # noqa: E402
_paths.setup_import_paths()
sys.path.insert(0, str(Path(__file__).resolve().parent))

import Cache                                                           # noqa: E402
from CpuBudget         import CpuBudget                                 # noqa: E402
from TissueMaskConfig  import MASK_RECIPES, MaskMaker                   # noqa: E402
from FovSupply         import FovSupply                                 # noqa: E402
from BenchCommon       import (ROLES, RunClock, Shot, Stage, Truth,     # noqa: E402
                               add_run_args, fmt_time, matching_rotation,
                               parse_run_args, require_gpu_if_allocated,
                               query_grid_centre, run_from_args,
                               run_slide_stages, slide_supplies)


def synthetic_shots(supply: FovSupply, workers: int) -> Iterator[Shot]:
    """A supply's photos as shots, each with where it was placed. Closing this
    closes the supply's generator, which drops a staged render entry."""
    gen = supply.shots(workers=workers)
    try:
        for index, meta, img, params in gen:
            geom: Dict[str, Any] = supply.geometry(meta)
            yield Shot(
                index=index, img=img, label=f'L{geom["level"]}',
                truth=Truth(
                    level=int(geom['level']), ds=float(meta.ds),
                    rot=matching_rotation(int(params['rot_deg'])),
                    centre=partial(query_grid_centre, supply.camera_for(meta.ds),
                                   geom['x0'], geom['y0'], params, img.shape)))
    finally:
        gen.close()


def bench_slide(path: str, supply: FovSupply, stages: Tuple[Stage, Stage, Stage],
                routes: List[str], masks: MaskMaker, args: argparse.Namespace,
                own: str, run_clock: RunClock) -> None:
    """One slide: its synthetic FoVs through the stages, on the stage-1 route
    and the oracle's, under the address of its renders."""
    run_slide_stages(path, synthetic_shots(supply, args.workers), stages, routes,
                     masks, supply.render_address.on(own), ROLES, own,
                     limit=args.limit, run_clock=run_clock)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    add_run_args(ap)
    ap.add_argument('--save-photos', action='store_true',
                    help='keep every photo beside its record')
    ap.add_argument('--batch-size', type=int, default=None,
                    help="stage 2 encoder's batch (not identity)")
    ap.add_argument('--multi-gpu', action='store_true')
    ap.add_argument('--device', default='auto')
    args = parse_run_args(ap)

    import torch
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    require_gpu_if_allocated(device, args.device)
    budget = CpuBudget.for_job(processes=1).apply()
    args.workers = budget.workers
    print(f'device     : {device}   {budget.line()}', flush=True)

    stages, routes, mask_cfg = run_from_args(
        args, device, batch_size=args.batch_size, multi_gpu=args.multi_gpu,
        read_workers=budget.workers)
    for s in stages:
        print(f'stage {s.n}    : {s.id}', flush=True)
    print(f'routes     : {" ".join(routes)}   mask {args.seg} '
          f'{mask_cfg.seg_id()}/{mask_cfg.region_id()}', flush=True)

    own = Cache.job_name('BenchLocaScope')
    fov_masks = MaskMaker(MASK_RECIPES['hest'], args.fov_mask_cache_job, device)
    masks = MaskMaker(mask_cfg, args.mask_cache_job, device)
    print(f'shots      : {" ".join(args.datasets)}  #{args.split}  '
          f'n_wsi={args.n_wsi}  cache job {own}', flush=True)

    t_start = time.time()
    run_clock = RunClock()
    for _, _, path, supply in slide_supplies(args, fov_masks, own,
                                             save_photos=args.save_photos):
        bench_slide(path, supply, stages, routes, masks, args, own, run_clock)
    if run_clock.slides:
        print('\n' + run_clock.report(routes), flush=True)
    print(f'\nTotal wall time: {fmt_time(time.time() - t_start)} '
          f'(everything: models loaded, photos rendered, cache written)', flush=True)


if __name__ == '__main__':
    main()
