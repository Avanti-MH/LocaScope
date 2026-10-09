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
this job). With --save-photos the photos are kept beside their record
(`photos_<gap>/`), so every method after the first reads them instead of rendering
them; without it (the default) a render entry that has them is read from, and one that
has not is rendered from, each time, and its parameters checked against the record.

METHODS: each `--stage1` is `<method>:<recipe>` from `stage1_estimation.METHODS`
(KNN_RECIPES, CLASSIFIER_RECIPES, PROTOTYPE_RECIPES, CLASSIC_RECIPES). A recipe
naming a mask recipe is given --seg's (`BenchCommon.stage1_of`).

CHECKPOINTS: each `--checkpoints` is a directory (its *.pt), a glob or a file, and
every trained checkpoint it names is one more method -- built from what the file
records about itself (`<Method>Config.from_checkpoint`), not from a recipe, so a
checkpoint nobody registered is run all the same. A prototype checkpoint (it holds
`routing_head_name`) is a prototype method, a classifier checkpoint (it holds
`head_name`) a classifier method; its stage name is the file's stem
(`..._best`, `..._last`, `..._native`), which is what tells the variants of one run
apart. `--checkpoints` adds to `--stage1`; give `--stage1` nothing but `none` to run
only the checkpoints.

One method is resident at a time: methods are the outer loop.

WRITES, per (slide, method), the stage-1 entry bench_locascope writes -- the
same address and the same tables, so either bench reads what the other made:

    .../render=<gap>/stage1/{output,neighbours,probs,votes,prototypes_index}_<s1>.csv

`output` is the method's own answer; `neighbours` a KNN's vote, `ref_index`
the reference tile's row in the bank's draw (`.../plan=native-.../draw/
index_<sampler>.csv`, under the same job); `probs` a voting method's per-patch
probabilities and `votes` every rule of `BenchCommon.RULES` over them;
`prototypes_index` a prototype method's support tiles, one row per tile with
the bank draw's index columns (`index` its row in the draw) and `prototype`,
the level it went into.
Each reference bank is a draw entry, its encoded features beside it.

Nothing is scored here: `utilities/cli/metrics/analyze_stage1_metrics.py`
takes the same flags, reads these entries and the render's ground truth.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from BenchCommon import (ROLES, Tables, run_stage1, slide_supplies,  # noqa: E402
                             stage1_of, stage1_record, status, support_rows)

JOB_NAME = 'Stage1MppBench'

#: What runs when no --stage1 is given.
DEFAULT_METHODS = ('knn:gigapath', 'knn:uni2', 'classic:default')

#: `--stage1 none`: no recipe method, only what --checkpoints names.
NO_RECIPES = 'none'


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
                    help=f'<method>:<recipe>, one or more; `{NO_RECIPES}` for none')
    ap.add_argument('--checkpoints', nargs='*', default=[],
                    help='trained checkpoints, each run as its own method built from '
                         'what the file records: a directory (its *.pt), a glob or a '
                         'file. Added to --stage1')
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


def checkpoint_files(patterns: List[str]) -> List[Path]:
    """The .pt files `--checkpoints` names, sorted, each once."""
    found: Dict[Path, None] = {}
    for pattern in patterns:
        path: Path = Path(pattern)
        hits: List[Path] = (sorted(path.glob('*.pt')) if path.is_dir()
                            else sorted(Path(m) for m in glob.glob(pattern)))
        if not hits:
            raise FileNotFoundError(f'--checkpoints {pattern!r} names no .pt file')
        for hit in hits:
            found[hit.resolve()] = None
    return list(found)


def checkpoint_method(weights: Path) -> Tuple[str, str, Any, type]:
    """`(method, name, config, estimator class)` for a trained checkpoint, built from
    the file's own record. The kind is read off the keys the two trainers write:
    `routing_head_name` only a prototype checkpoint has, `head_name` only a
    classifier's."""
    keys = torch.load(weights, map_location='cpu', mmap=True, weights_only=False)
    if 'routing_head_name' in keys:
        method: str = 'prototype'
    elif 'head_name' in keys:
        method = 'classifier'
    else:
        raise ValueError(f'{weights} is neither a prototype nor a classifier '
                         f'checkpoint; it holds {sorted(keys)}')
    module_name, _, cls_name = stage1_estimation.METHODS[method]
    module = importlib.import_module(module_name)
    cls: type = getattr(module, cls_name)
    config_cls: type = getattr(module, f'{cls_name}Config')
    return method, weights.stem, config_cls.from_checkpoint(str(weights)), cls


def stages_of(args, device=None) -> list:
    """One `BenchCommon.Stage` per --stage1 and per --checkpoints file, models not
    built."""
    mask_cfg = mask_cfg_from_args(args)
    stages: list = [stage1_of(*stage1_estimation.recipe(spec), mask_cfg, args.seg, device)
                    for spec in args.stage1 if spec != NO_RECIPES]
    for weights in checkpoint_files(args.checkpoints):
        stages.append(stage1_of(*checkpoint_method(weights), mask_cfg, args.seg, device))
    return stages


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False)
    add_args(ap)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save-photos', action='store_true',
                    help='keep each FoV photo beside its render record (photos_<gap>/<i>.png, '
                         '~250 MB a slide) so every method after the first reads it instead '
                         'of rendering it. Default: off, a photo is rendered each time it is '
                         'needed (a render entry that already has them is still read from)')
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
                                                save_photos=args.save_photos):
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
