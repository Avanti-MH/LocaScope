#!/usr/bin/env python3
"""Stage-1 mpp estimators compared head to head on the same FoVs, every
method's results a cache entry.

    python utilities/bench_modules/bench_stage1_mpp.py \\
        --stage1 knn:gigapath knn:uni2 classifier:gigapath-arcface classic:default \\
        --datasets bracs/test ki67_with_photo --split test --n-wsi 5

FOVS: the first --n-wsi slides of each dataset's recorded --split, one draw
per slide over the --fov recipe's levels on the hest masks of
--fov-mask-cache-job, photographed through `FovSupply.cached` -- the draws and
photos bench_locascope uses, in --draw-cache-job / --render-cache-job (default
this job). The photos are kept beside their record (`photos_<gap>/`), so every
method after the first reads them instead of rendering them.

METHODS: each `--stage1` is `<method>:<recipe>` from `stage1_estimation.METHODS`
(KNN_RECIPES, CLASSIFIER_RECIPES, PROTOTYPE_RECIPES, CLASSIC_RECIPES). A recipe
naming a mask recipe is given --seg's (`bench_locascope.stage1_of`). One
method is resident at a time: methods are the outer loop.

WRITES, per (slide, method), the stage-1 entry bench_locascope writes -- the
same address and the same tables, so either bench reads what the other made:

    .../render=<gap>/stage1/{output,neighbours,probs,votes,prototypes_index}_<s1>.csv

`output` is the method's own answer; `neighbours` a KNN's vote, `ref_index`
the reference tile's row in the bank's draw (`.../plan=native-.../draw/
index_<sampler>.csv`, under the same job); `probs` a voting method's per-patch
probabilities and `votes` every rule of `bench_locascope.RULES` over them;
`prototypes_index` a prototype method's support tiles, one row per tile with
the bank draw's index columns (`index` its row in the draw) and `prototype`,
the level it went into.
Each reference bank is a draw entry, its encoded features beside it.

Nothing is scored here: `utilities/cli/metrics/analyze_stage1_metrics.py`
takes the same flags, reads these entries and the render's ground truth.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))                                # utilities/
import _paths                                                       # noqa: E402
_paths.setup_import_paths()
sys.path.insert(0, str(_HERE))

import Cache                                                        # noqa: E402
from CpuBudget import CpuBudget                                     # noqa: E402
from FovSupply import add_fov_args                                  # noqa: E402
from SafeSlide import SafeSlide                                     # noqa: E402
from TissueMaskConfig import (MASK_RECIPES, MaskMaker, add_mask_args,  # noqa: E402
                              mask_cfg_from_args)
import stage1_estimation                                            # noqa: E402
from bench_locascope import (ROLES, Tables, run_stage1, slide_supplies,  # noqa: E402
                             stage1_of, stage1_record, status, support_rows)

JOB_NAME = 'Stage1MppBench'

#: What runs when no --stage1 is given.
DEFAULT_METHODS = ('knn:gigapath', 'knn:uni2', 'classic:default')


def add_args(ap) -> None:
    """The flags that decide which entries a run writes. The analysis takes
    the same ones and computes the same addresses."""
    ap.add_argument('--datasets', nargs='+', default=['bracs/test', 'ki67_with_photo'])
    ap.add_argument('--split', default='test', choices=['val', 'test'],
                    help='the recorded split slides are taken from. FoV_Vote.md '
                         'fixes every risk threshold on val before test is '
                         'looked at: run val first')
    ap.add_argument('--n-wsi', type=int, default=5, help='slides per dataset')
    ap.add_argument('--stage1', nargs='+', default=list(DEFAULT_METHODS),
                    help='<method>:<recipe>, one or more')
    # --fov, --max-ds, --sampler-*, --camera-*
    add_fov_args(ap)
    ap.add_argument('--fov-mask-cache-job', default='MppRoutingHead',
                    help='hest masks the FoVs are placed on')
    # --seg: the mask the reference banks are drawn on
    add_mask_args(ap)
    ap.add_argument('--mask-cache-job', default='MppRoutingHead',
                    help="whose mask cache the --seg mask is read from and "
                         'written to')
    ap.add_argument('--draw-cache-job', default=None,
                    help='whose cache the FoV draws are in. Default: this job')
    ap.add_argument('--render-cache-job', default=None,
                    help='whose cache the photos are in. Default: this job')
    ap.add_argument('--limit', type=int, default=0,
                    help='only the first N FoVs of each slide (a smoke run); '
                         'recorded in the entries')


def stages_of(args, device=None) -> list:
    """One `bench_locascope.Stage` per --stage1, models not built."""
    mask_cfg = mask_cfg_from_args(args)
    return [stage1_of(*stage1_estimation.recipe(spec), mask_cfg, args.seg, device)
            for spec in args.stage1]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False)
    add_args(ap)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    workers = CpuBudget.for_job(processes=1).apply().workers
    own = Cache.job_name(JOB_NAME)
    stages = stages_of(args, device)
    for s in stages:
        print(f'stage 1    : {s.id}', flush=True)
    mask_cfg = mask_cfg_from_args(args)
    fov_masks = MaskMaker(MASK_RECIPES['hest'], args.fov_mask_cache_job, device)
    masks = MaskMaker(mask_cfg, args.mask_cache_job, device)

    # Which entries are missing, per slide. Nothing is rendered or built yet.
    slides = []
    for _, name, path, supply in slide_supplies(args, fov_masks, own,
                                                save_photos=True):
        base = supply.render_address.on(own)
        todo = {}
        for s in stages:
            entry, rec = base.entry('stage1'), stage1_record(s, base, args.limit)
            if status(entry, s.id, rec) != 'hit':
                todo[s.id] = Tables(entry, s.id, rec, ROLES[1])
        print(f'  {len(stages) - len(todo)} of {len(stages)} methods hit',
              flush=True)
        if todo:
            slides.append((name, path, supply, todo))

    # One method at a time over every slide that misses it.
    for s in stages:
        work = [(name, path, supply, todo[s.id])
                for name, path, supply, todo in slides if s.id in todo]
        if not work:
            continue
        print(f'\n-- {s.id} --', flush=True)
        est = s.obj()
        for name, path, supply, tables in work:
            t0 = time.perf_counter()
            wsi = SafeSlide(path)
            est.build(wsi, mask=None, masks=masks, cache_job=own)
            tables.add_slide('prototypes_index', support_rows(est))
            t_build = time.perf_counter() - t0
            n, t0 = 0, time.perf_counter()
            shots = supply.shots(workers=workers)
            for index, _, img, _ in shots:
                if args.limit and n >= args.limit:
                    break
                n += 1
                run_stage1(est, img, tables, index)
            shots.close()
            tables.write()
            wsi.close()
            print(f'  {name}: {n} FoVs  build {t_build:.0f}s  '
                  f'estimate {time.perf_counter() - t0:.0f}s  -> '
                  f'{tables.entry.dir}', flush=True)
        s._obj = None
        del est
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    masks.close()
    print('\nscore with utilities/cli/metrics/analyze_stage1_metrics.py and the '
          'same flags', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
