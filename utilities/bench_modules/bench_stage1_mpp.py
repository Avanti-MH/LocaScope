#!/usr/bin/env python3
"""Stage-1 mpp estimators compared head to head on the same drawn FoVs.

    python utilities/bench_modules/bench_stage1_mpp.py \
        --knn-encoder gigapath uni2 --classifier-weights <ckpt> ... \
        --datasets bracs/test ki67_with_photo --n-wsi 9

One KnnEstMpp per --knn-encoder and one ClassifierEstMpp per
--classifier-weights checkpoint, each run on the SAME synthetic FoVs (drawn
once per slide, from the RECORDED test split) across --n-wsi slides of every
--datasets. Writes one row per (FoV, method) to

    result/<SLURM_JOB_NAME or Stage1MppBench>/<sampler_id>_<seg_id>_<region_id>.csv

which utilities/cli/metrics/analyze_stage1_metrics.py scores. The scoring
lives there, not here: this bench only produces rows.

Split out of bench_mpp_feature_decomposition.py (2026-09-29), where it was the
`stage1_compare` part. That file's other three parts (axes, subspace_knn,
sampler_routing) investigate the feature space and read cached stores; this
one measures the production estimators and reads no store, which is why it
has its own job and result directory. Its cache job defaults to
Stage1MppBench; `--mask-cache-job` / `--sampler-cache-job` name an existing
one instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / 'utilities'))
import _paths                                                       # noqa: E402
_paths.setup_import_paths()

from KnnEstMpp import (KnnEstMpp, KnnEstMppConfig,                  # noqa: E402
                       REFERENCE_BANK_RICHNESS)
from ClassifierEstMpp import ClassifierEstMpp, ClassifierEstMppConfig  # noqa: E402
from StageInterface import EstMppResult                             # noqa: E402
from _paths import job_result_dir                                   # noqa: E402
from AccessDatasets import locate                                    # noqa: E402
from training.MppRoutingHead.Datasets import (                      # noqa: E402
    add_cache_args, open_caches)
from WsiSplit import SPLIT_JOB, read_split, split_path              # noqa: E402
from SafeSlide import SafeSlide                                     # noqa: E402
from TissueMaskConfig import MASK_RECIPES                           # noqa: E402
from TileSampler import (OverlapConfig, PlanSpec, RichnessConfig,     # noqa: E402
                         SamplerConfig, TileSampler)
from DsLadder import DEFAULT_RUNGS                                  # noqa: E402
from QueryFromWSI import QueryFromWSI                                # noqa: E402
from simulate_microscope_photo import simulate_microscope_photo       # noqa: E402

JOB_NAME = 'Stage1MppBench'


def write_csv(rows, path) -> None:
    if not rows:
        print(f'  (nothing to write to {Path(path).name})')
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, restval='')
        writer.writeheader()
        writer.writerows(rows)
    print(f'  {path}  ({len(rows)} rows)')


# ══════════════════════════════════════════════════════════════════════════════
#  PART "stage1_compare"  --  NEW, 2026-09-17
# ══════════════════════════════════════════════════════════════════════════════
#
# Compares whichever stage-1 mpp estimators are named on the command line
# (one KnnEstMpp per --knn-encoder, one ClassifierEstMpp per
# --classifier-weights checkpoint) on the SAME drawn FoVs, across --n-wsi
# slides from each of --datasets. Writes one row per (FoV, method) to a csv
# that utilities/cli/metrics/analyze_stage1_metrics.py reads -- see that
# file's own docstring for the schema and for why the scoring lives there,
# not here: this part's only job is to produce rows in the shape that file
# expects.

def _sampling_recipe_id(args) -> str:
    """`<sampler_id>_<seg_id>` -- the filename analyze_stage1_metrics.py's
    own docstring names. Two hashes because they answer two different
    questions: sampler_id is "which FoVs, from which slides, how"; seg_id is
    "what counted as tissue" -- a caller who only changed the mask recipe
    should not silently collide with a run that sampled different FoVs, and
    the reverse.
    """
    import hashlib
    parts = '|'.join([
        f'native_only={args.native_only}', f'tile={args.tile}',
        f'n_per_rung={args.n_per_rung}', f'seed={args.seed}',
        f'mpixels={args.mpixels}', f'ratio={args.ratio}',
        f'datasets={",".join(sorted(args.datasets))}', f'n_wsi={args.n_wsi}'])
    sampler_id = hashlib.sha256(parts.encode()).hexdigest()[:8]
    mask_cfg = MASK_RECIPES[args.seg]
    return f'{sampler_id}_{mask_cfg.seg_id()}_{mask_cfg.region_id()}'


def _overlap_cfg(enabled: bool) -> OverlapConfig:
    """Disjoint lattice (default) or, with `--overlap`, the same lattice plus
    JITTER TOP-UP allowed. A coarse rung's footprint is huge, so a disjoint
    lattice can run out of room inside a tissue region long before
    `n_per_rung` is reached -- and `OverlapConfig`'s own default
    (`jitter_cap=0`) is deliberately dead under a disjoint lattice, by its
    own docstring: "under a disjoint lattice the top-up is provably dead...
    it means something as soon as overlap is allowed." Raising
    `max_overlap_ratio`/`overlapping_share`/`jitter_cap` is what makes that
    existing top-up mechanism reachable, without changing `grid_step` (the
    main lattice stays disjoint; only the shortfall gets topped up).
    """
    if not enabled:
        return OverlapConfig()
    return OverlapConfig(max_overlap_ratio=0.5, overlapping_share=1.0,
                         jitter_cap=1.0)


def _method_specs(args) -> list:
    """`[{'kind', 'encoder', 'classifier', 'reduction', 'weights',
    'weights_path', 'needs_mask'}, ...]` -- DESCRIPTORS only, nothing built.

    Building is deferred to `_instantiate` and happens ONE METHOD AT A TIME
    in `run_stage1_compare`, not all up front: every method here loads a
    full encoder (a `ClassifierEstMpp` loads a head on top of one too), and
    2 KnnEstMpp + N ClassifierEstMpp all resident for the whole multi-slide
    run is what OOM-killed job 346494 -- 13 encoders alive at once against a
    64G --mem request. Only the CHEAP part (reading a checkpoint's own
    header) happens here, so a bad checkpoint is skipped before the run
    spends a build on it.

    A checkpoint that fails to even load its header is SKIPPED with a
    printed warning, not raised -- one stale file among
    `--classifier-weights all`'s glob should not sink a multi-hour
    comparison over every other method that loads fine.
    """
    specs = []
    for name in args.knn_encoder:
        specs.append(dict(kind='knn', encoder=name, classifier='',
                          reduction='', loss='', weights='', weights_path=None,
                          needs_mask=True))
    for weights in args.classifier_weights:
        try:
            ckpt = torch.load(weights, map_location='cpu')
        except Exception as exc:                             # noqa: BLE001
            print(f'  [SKIP] {weights}: {type(exc).__name__}: {exc}')
            continue
        # ckpt['head_name'], NOT ckpt['classifier']. 'classifier' is the
        # CLASS's own registered name (Checkpoints.save_checkpoint's own
        # `classifier_name(type(head.classify))`) -- 'mlp' for every one of
        # mlp/mlp_deep/mlp_wide/mlp_deep_residual/mlp_deep_wide, since all
        # five build the SAME MlpHead class at different mlp_depth/
        # mlp_width_mult/mlp_residual. 'head_name' is the HEAD_CHOICES key
        # cli/train.py trained it under (`for name, head in heads.items():
        # save_tagged(..., name, ...)`), already stored in every checkpoint
        # -- so this needs no Checkpoints.py change, only reading the field
        # that was already there. Using 'classifier' here made
        # analyze_stage1_metrics.py's method_of() collapse all five mlp
        # variants into one 'uni2+mlp+fixed' label, averaging five actually-
        # different trained models into one number.
        #
        # `loss` (2026-09-22): the SAME class of gap -- ckpt['args']['loss']
        # (train.py's own `--loss`, default 'bal') is not part of `head_name`
        # either, so a bal- and an ord_a-trained arcface head would collapse
        # into one 'gigapath+arcface' method_of() label without this,
        # averaging two differently-trained models the same way the mlp
        # variants did before 'classifier' switched to head_name. `.get`,
        # not `[...]`: a checkpoint trained before `--loss` existed has no
        # `'loss'` key in `args` at all, and 'bal' was every run's behaviour
        # before that flag was added, so it is the correct default, not a
        # guess.
        specs.append(dict(
            kind='classifier', encoder=ckpt['encoder'],
            classifier=ckpt['head_name'], reduction=ckpt['reduction'],
            loss=ckpt.get('args', {}).get('loss', 'bal'),
            weights=os.path.basename(weights), weights_path=weights,
            needs_mask=False))
    if not specs:
        raise ValueError(
            'stage1_compare needs at least one method: pass --knn-encoder '
            'and/or --classifier-weights')
    return specs


def _instantiate(spec: dict, args, device):
    """The one estimator `spec` describes, built now (not before). Caller
    frees it (`del`, `torch.cuda.empty_cache()`) once every WSI has been run
    against it, before calling this again for the next spec."""
    if spec['kind'] == 'knn':
        cfg = KnnEstMppConfig(
            encoder=spec['encoder'],
            sampler_cfg=SamplerConfig(tile=args.tile, n_per_rung=args.knn_samples,
                                      seed=args.seed,
                                      richness=REFERENCE_BANK_RICHNESS,
                                      overlap=_overlap_cfg(args.overlap)),
            k=args.knn_k)
        return KnnEstMpp(cfg, device)
    cfg = ClassifierEstMppConfig.from_checkpoint(spec['weights_path'])
    return ClassifierEstMpp(cfg, device)


def _extra_fields(result) -> dict:
    """Whatever a specific `EstMppResult` subclass adds beyond the five base
    fields -- generic over any future estimator, no hardcoded field names."""
    import dataclasses
    base = {f.name for f in dataclasses.fields(EstMppResult)}
    return {k: v for k, v in dataclasses.asdict(result).items() if k not in base}


def _mem_snapshot(device) -> str:
    '''Host RSS (current + this process's lifetime peak) plus GPU allocated,
    for printing between methods in `run_stage1_compare`. `--mem=64G` on
    `jobscripts/Benchmarks/Stage1MppBench.sh` was sized for a
    DIFFERENT part of that jobscript entirely ("one token store in flight
    per level" -- its own comment) and job 346717 OOM-killed (exit 137, a
    HOST-memory cgroup kill, not a GPU one) running `stage1_compare`'s 13
    sequential methods against it. This is what tells us, from real numbers
    rather than a guess, whether host RAM climbs method over method (a
    leak/fragmentation worth chasing) or drops back near baseline after each
    `del estimator` (this job just needs a bigger --mem for this part).

    `ru_maxrss` is KB on Linux (this cluster), not bytes -- it would be bytes
    on macOS, but nothing here runs there.
    '''
    rss_now_gb = None
    try:
        with open('/proc/self/status') as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    rss_now_gb = int(line.split()[1]) / (1024 ** 2)
                    break
    except OSError:
        pass
    rss_peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    gpu_gb = (torch.cuda.memory_allocated(device) / 1e9
             if device.type == 'cuda' else 0.0)
    now = f'{rss_now_gb:.2f} GB' if rss_now_gb is not None else '?'
    return (f'host_rss_now={now}  host_rss_peak={rss_peak_gb:.2f} GB  '
           f'gpu_alloc={gpu_gb:.2f} GB')


def run_stage1_compare(args, out_dir: Path) -> int:
    device = torch.device(args.device)
    specs = _method_specs(args)
    print('methods: ' + ', '.join(
        f'{s["encoder"]}' + (f'+{s["classifier"]}' if s['classifier'] else '')
        for s in specs))

    # WSI lists resolved ONCE -- cheap (no model), and every method has to
    # run against the SAME slides for the comparison to be paired.
    slides_by_dataset = {}
    for dataset_id in args.datasets:
        # READ, never written: the split make_split.py recorded
        # (--split-cache-job, default MakeSplit) -- the same file the
        # checkpoints were selected on, so this bench never scores on a slide
        # a checkpoint saw in val.
        test_names = read_split(split_path(args.split_cache_job or SPLIT_JOB,
                                           dataset_id))[1]
        slides_by_dataset[dataset_id] = test_names[:args.n_wsi]
        print(f'{dataset_id}: {len(slides_by_dataset[dataset_id])} slide(s) '
             f'from the recorded test split '
             f'({", ".join(slides_by_dataset[dataset_id])})')

    # Segmentation + position sampling, ONCE per slide, shared by every
    # method below -- nothing about either varies across methods, so neither
    # is redone 17 methods x 18 slides times. Both go through the sampler
    # cache: `TileSampler.cached` for the positions (a slide already drawn
    # under this recipe/config/plan is not even opened), and the mask cache
    # for the mask the KNN methods need (a hit after the draw, since the draw
    # had to make it). The segmenter is loaded once, on the first miss, and
    # released before any method is built.
    print(f'\n======== segmenting ({args.seg}) + sampling every slide once ========')
    caches = open_caches(args, JOB_NAME, device)
    sampler_cfg = SamplerConfig(tile=args.tile, n_per_rung=args.n_per_rung,
                                seed=args.seed, richness=RichnessConfig(),
                                overlap=_overlap_cfg(args.overlap))
    plan = (PlanSpec('native') if args.native_only
            else PlanSpec('ladder', tuple(DEFAULT_RUNGS)))
    slide_cache = {}
    for dataset_id, names in slides_by_dataset.items():
        for wsi_name in names:
            entry = locate(wsi_name, dataset=dataset_id)
            sampler = TileSampler.cached(
                entry.path, sampler_cfg, plan, caches.sampler_root,
                masks=caches.masks,
                report_dir=out_dir / 'sampler_reports' / dataset_id.replace('/', '_'))
            wsi = SafeSlide(entry.path)
            mask, _ = caches.masks.mask(wsi)
            # `RungPlan.shrink` says which rungs are native (1.0) -- read off
            # this slide's own plans rather than re-derived.
            native_by_rung = {float(p.rung_ds): float(p.shrink) == 1.0
                              for p in plan.plans_for(wsi, args.tile)}
            positions = [dict(x=int(s.meta.x), y=int(s.meta.y),
                              rung=float(s.meta.ds),
                              native=native_by_rung.get(float(s.meta.ds), False))
                         for s in sampler]
            print(f'  {wsi_name}: {len(positions)} positions   (mask '
                 f'{"reused" if sampler.cache_info["mask_hit"] else "segmented"}, '
                 f'draw {"reused" if sampler.cache_info["samples_hit"] else "drawn"})')
            slide_cache[(dataset_id, wsi_name)] = (mask, positions)
            wsi.close()
    caches.masks.close()
    print(f'  [after segmentation] {_mem_snapshot(device)}')

    rows = []
    for spec in specs:
        label = spec['encoder'] + (f'+{spec["classifier"]}'
                                   if spec['classifier'] else '')
        print(f'\n-- {label} --  [before build] {_mem_snapshot(device)}')
        try:
            estimator = _instantiate(spec, args, device)
        except Exception as exc:                             # noqa: BLE001
            print(f'  [SKIP] failed to build: {type(exc).__name__}: {exc}  '
                 f'[{_mem_snapshot(device)}]')
            continue

        for dataset_id, names in slides_by_dataset.items():
            for wsi_name in names:
                entry = locate(wsi_name, dataset=dataset_id)
                mask, positions = slide_cache[(dataset_id, wsi_name)]
                wsi = SafeSlide(entry.path)

                if spec['needs_mask']:
                    estimator.build(wsi, mask=mask)
                else:
                    estimator.build(wsi)

                for pos in positions:
                    gt_ds = pos['rung']
                    gt_mpp = wsi.base_mpp * gt_ds
                    # `rung` is the CANONICAL bin (nearest DEFAULT_RUNGS
                    # value in log space), not `gt_ds` itself -- under
                    # --native-only, `gt_ds` is this WSI's own native
                    # downsample (e.g. 4.00003), and grouping by that exact
                    # float would give every slide its own private "rung 4"
                    # instead of letting analyze_stage1_metrics.py aggregate
                    # them as the same one. `gt_ds`/`gt_mpp` keep the exact
                    # value for mpp_error_relative.
                    rung = min(DEFAULT_RUNGS,
                              key=lambda r: abs(np.log(gt_ds) - np.log(r)))
                    qwsi = QueryFromWSI(entry.path, MPixels=args.mpixels,
                                        wh_ratio=args.ratio, mpp=gt_mpp)
                    image = qwsi.crop(pos['x'], pos['y'])
                    if image is None:
                        continue
                    query = simulate_microscope_photo(image)

                    result = estimator.estimate(query)
                    rows.append(dict(
                        dataset=dataset_id, wsi_name=wsi_name,
                        x=pos['x'], y=pos['y'],
                        h=query.shape[0], w=query.shape[1],
                        rung=rung, native=pos['native'],
                        gt_mpp=gt_mpp, gt_ds=gt_ds,
                        encoder=spec['encoder'], classifier=spec['classifier'],
                        reduction=spec['reduction'], loss=spec['loss'],
                        weights=spec['weights'],
                        estimated_ds=result.estimated_ds,
                        estimated_mpp=result.estimated_mpp,
                        chosen_ds=result.chosen_ds, chosen_mpp=result.chosen_mpp,
                        chosen_level=result.chosen_level,
                        extra_json=json.dumps(_extra_fields(result))))
                wsi.close()

        # Freed before the NEXT spec's build -- this is the whole point of
        # the method-outer loop: only one method's encoder(s) are ever
        # resident at once.
        del estimator
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'  [after free] {_mem_snapshot(device)}')

    path = out_dir / f'{_sampling_recipe_id(args)}.csv'
    write_csv(rows, path)
    print(f'\n{path}  ({len(rows)} rows) -- read with '
         f'utilities/cli/metrics/analyze_stage1_metrics.py')
    return 0


# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tile', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mpixels', type=float, default=1.475,
                        help='query size -- 1.475 MPixels at 45:32 matches '
                             "CLAUDE.md's real-photo spec (1440x1024), not "
                             "query_sim's 4:3/12MP default")
    parser.add_argument('--ratio', default='45:32',
                        help='query W:H ratio, passed to '
                             'QueryFromWSI. Defaulted explicitly rather than '
                             "left off -- an omitted wh_ratio silently falls "
                             "back to QueryFromWSI's own 4:3 default instead "
                             "of CLAUDE.md's real-photo spec, which is "
                             "exactly the bug run_sampler_routing's own "
                             "QueryFromWSI call above still has (flagged, "
                             "not fixed, 2026-09-16)")
    parser.add_argument('--datasets', nargs='+',
                        default=['bracs/test', 'ki67_with_photo'],
                        help='which datasets to draw slides '
                             'from')
    parser.add_argument('--n-wsi', type=int, default=9,
                        help='slides per dataset, taken from '
                             'the RECORDED test split '
                             '(utilities/cli/build_cache/make_split.py) '
                             'so this never scores on a slide a checkpoint '
                             'was selected on')
    parser.add_argument('--native-only', action='store_true',
                        help='sample only the ds values '
                             "this WSI's own pyramid actually has "
                             '(native_plans) instead of every DsLadder rung '
                             '-- see analyze_stage1_metrics.py\'s docstring '
                             'for what this changes about the output tables')
    parser.add_argument('--n-per-rung', type=int, default=20,
                        help='query positions per rung per '
                             'slide')
    # --seg / --mask-cache-job / --sampler-cache-job / --split-cache-job.
    add_cache_args(parser)
    parser.add_argument('--overlap', action='store_true',
                        help='allow jitter top-up (both the '
                             'query positions and each KnnEstMpp\'s own '
                             'reference bank) instead of a strictly disjoint '
                             'lattice -- see _overlap_cfg for why the '
                             'disjoint default can come up short at coarse '
                             'rungs, where one tile\'s footprint is most of '
                             'a tissue region')
    parser.add_argument('--knn-encoder', nargs='+', default=[],
                        help='one KnnEstMpp per encoder name '
                             '(TileEncoderFunc registry)')
    parser.add_argument('--knn-samples', type=int, default=40,
                        help='reference tiles per level for '
                             'EVERY KnnEstMpp (was hardcoded to 40, '
                             'KnnEstMppConfig\'s own default)')
    parser.add_argument('--knn-k', type=int, default=5,
                        help='k for EVERY KnnEstMpp (was '
                             'never exposed, silently stuck at '
                             'KnnEstMppConfig\'s own default)')
    parser.add_argument('--classifier-weights', nargs='+', default=[],
                        help='one ClassifierEstMpp per '
                             'trained checkpoint path')

    parser.add_argument(
        '--out', default=None,
        help='output directory, used verbatim. Default: '
             f'result/<SLURM_JOB_NAME or {JOB_NAME}>/ (no encoder tag: one '
             'CSV can span several encoders)')
    args = parser.parse_args()
    if not (args.knn_encoder or args.classifier_weights):
        parser.error('needs --knn-encoder and/or --classifier-weights')

    out_dir = Path(args.out or job_result_dir(JOB_NAME))
    out_dir.mkdir(parents=True, exist_ok=True)
    return run_stage1_compare(args, out_dir)


if __name__ == '__main__':
    sys.exit(main())
