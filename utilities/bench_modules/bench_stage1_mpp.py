#!/usr/bin/env python3
"""Stage-1 mpp estimators compared head to head on the same drawn FoVs.

    python utilities/bench_modules/bench_stage1_mpp.py \
        --knn-encoder gigapath uni2 --classifier-weights <ckpt> ... \
        --prototype-weights <ckpt> ... --classic \
        --datasets bracs/test ki67_with_photo --n-wsi 9

Every stage-1 method, each run on the SAME synthetic FoVs (drawn once per
slide, from the RECORDED test split) across --n-wsi slides of every
--datasets, and shown the same photo of each (`FovSupply.photo`):

    KnnEstMpp          one per --knn-encoder
    ClassifierEstMpp   one per --classifier-weights checkpoint (MppRoutingHead)
    PrototypeEstMpp    one per --prototype-weights checkpoint
                       (PrototypicalRoutingHead)
    ClassicEstMpp      --classic, the fingerprint baseline

The two that vote over per-patch probabilities (classifier, prototype) run
once per FoV and are scored under every --rules rule (RULES), one row per
rule, with the FoV's distribution statistics (`fov_stats`) on each row. Writes

    result/<SLURM_JOB_NAME or Stage1MppBench>/<sampler_id>_<seg_id>_<region_id>_<split>.csv
    result/<...>/<sampler_id>_<seg_id>_<region_id>_<split>_probs.jsonl

-- one row per (FoV, method, vote), and the raw per-patch probabilities once
per (FoV, voting method). utilities/cli/metrics/analyze_stage1_metrics.py
scores them, the vote diagnosis included. The scoring lives there, not here:
this bench only produces rows.

bench_mpp_feature_decomposition.py's parts (axes, subspace_knn,
sampler_routing) investigate the feature space and read cached stores; this
one measures the production estimators and reads no store, which is why it
has its own job and result directory. Its cache job defaults to
Stage1MppBench; `--mask-cache-job` / `--draw-cache-job` name an existing
one instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / 'utilities'))
import _paths                                                       # noqa: E402
_paths.setup_import_paths()

from stage1_estimation.KnnEstMpp import KnnEstMpp, KnnEstMppConfig                   # noqa: E402
from stage1_estimation.ClassifierEstMpp import ClassifierEstMpp, ClassifierEstMppConfig  # noqa: E402
from stage1_estimation.PrototypeEstMpp import PrototypeEstMpp, PrototypeEstMppConfig  # noqa: E402
from stage1_estimation.estimate_mpp_classic import ClassicEstMpp, ClassicEstMppConfig  # noqa: E402
from stage1_estimation.FoVVote import QUALITY_SIGNALS, diagnose                      # noqa: E402
from stage1_estimation.StageInterface import EstMppResult                             # noqa: E402
from _paths import job_name, job_result_dir                         # noqa: E402
from CpuBudget import CpuBudget                                     # noqa: E402
from AccessDatasets import list_names, locate                        # noqa: E402
from training.MppRoutingHead.Datasets import (                      # noqa: E402
    add_cache_args, open_caches, read_label_of)
from SafeSlide import SafeSlide                                     # noqa: E402
from TissueMaskConfig import MASK_RECIPES                           # noqa: E402
from ConfigIdentity import enc, short_id                            # noqa: E402
from TileSampler import SAMPLER_RECIPES, PlanSpec                   # noqa: E402
from DsLadder import DEFAULT_RUNGS                                  # noqa: E402
from camera import Render, render_spec                               # noqa: E402
from FovSupply import FovSupply, add_fov_args, fov_from_args         # noqa: E402
from SlideReader import SlideReader                                 # noqa: E402

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
#  PART "stage1_compare"
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

#: The code that turns a draw into FoV rows (ConfigIdentity rule 3); bump it
#: when the same configs would produce other rows, and say why in TODO.log.
FOV_VERSION = 0


def _sampling_recipe_id(args) -> str:
    """`<fov id>_<seg_id>_<region_id>_<split>` -- the filename
    analyze_stage1_metrics.py's own docstring names. The fov id is composed
    from the ids of what decides which photos there are: the recipe's draw,
    domain gap, sensor and levels (`fov_from_args`), and the slides asked for.
    The mask is its own two ids, so a caller who only changed the mask recipe
    cannot collide with a run that drew other FoVs. The split is last and
    outside the hash: a val and a test run of one recipe differ only in it,
    which is how analyze_stage1_metrics pairs them."""
    fov = fov_from_args(args)
    fov_id = short_id([
        f'sampler={enc(fov.sampler.identity_id())}',
        f'gap={enc(fov.gap.identity_id())}',
        f'sensor={enc(tuple(fov.sensor))}',
        f'rungs={enc(fov.rungs)}', f'max_ds={enc(fov.max_ds)}',
        f'tile={enc(args.tile)}', f'datasets={enc(sorted(args.datasets))}',
        f'n_wsi={enc(args.n_wsi)}', f'fov_per_rung={enc(args.fov_per_rung)}',
        f'version={enc(FOV_VERSION)}'])
    mask_cfg = MASK_RECIPES[args.seg]
    return f'{fov_id}_{mask_cfg.seg_id()}_{mask_cfg.region_id()}_{args.split}'


def _method_specs(args) -> list:
    """`[{'kind', 'encoder', 'classifier', 'reduction', 'weights',
    'weights_path'}, ...]` -- DESCRIPTORS only, nothing built.

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
                          reduction='', loss='', read_level='', weights='', weights_path=None))
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
        # -- so analyze_stage1_metrics.py's method_of() does not collapse the
        # five mlp variants into one 'uni2+mlp+fixed' label.
        #
        # `loss`: the SAME class of gap -- ckpt['args']['loss'] (train.py's
        # own `--loss`, default 'bal') is not part of `head_name` either, so
        # a bal- and an ord_a-trained arcface head would share one label.
        # `.get` with 'bal': a checkpoint without the key was trained bal.
        specs.append(dict(
            kind='classifier', encoder=ckpt['encoder'],
            classifier=ckpt['head_name'], reduction=ckpt['reduction'],
            loss=ckpt.get('args', {}).get('loss', 'bal'),
            # how the head was TRAINED, the same gap one level further: a pyramid- and a resampled-trained head of one loss
            # would share a label. `read_label_of` reads `pyramid` off a
            # checkpoint saved before the read mode existed.
            read_level=read_label_of(ckpt.get('args')),
            weights=os.path.basename(weights), weights_path=weights))
    for weights in args.prototype_weights:
        try:
            ckpt = torch.load(weights, map_location='cpu')
        except Exception as exc:                             # noqa: BLE001
            print(f'  [SKIP] {weights}: {type(exc).__name__}: {exc}')
            continue
        # The arm is the filename: train.py names a run by every axis that
        # changes it (contexts, collapse, head, episode reuse, K), so the stem
        # without the encoder prefix and the checkpoint tag is the label --
        # two arms can never share one.
        stem = os.path.basename(weights).rsplit('_best.pt', 1)[0]
        arm = stem[len(ckpt['encoder']) + 1:] if stem.startswith(
            ckpt['encoder'] + '_') else stem
        specs.append(dict(
            kind='prototype', encoder=ckpt['encoder'],
            classifier=f'proto:{arm}', reduction='', loss='', read_level='',
            weights=os.path.basename(weights), weights_path=weights))
    if args.classic:
        specs.append(dict(kind='classic', encoder='classic', classifier='',
                          reduction='', loss='', read_level='', weights='',
                          weights_path=None))
    if not specs:
        raise ValueError(
            'stage1_compare needs at least one method: pass --knn-encoder, '
            '--classifier-weights, --prototype-weights and/or --classic')
    return specs


def _instantiate(spec: dict, args, device):
    """The one estimator `spec` describes, built now (not before). Caller
    frees it (`del`, `torch.cuda.empty_cache()`) once every WSI has been run
    against it, before calling this again for the next spec."""
    if spec['kind'] == 'knn':
        cfg = KnnEstMppConfig(
            encoder=spec['encoder'],
            sampler_cfg=replace(SAMPLER_RECIPES['reference-bank'],
                                n_per_rung=args.knn_samples, seed=args.seed),
            k=args.knn_k, tile_size=args.tile)
        return KnnEstMpp(cfg, device)
    if spec['kind'] == 'prototype':
        return PrototypeEstMpp(
            PrototypeEstMppConfig.from_checkpoint(spec['weights_path']), device)
    if spec['kind'] == 'classic':
        return ClassicEstMpp(ClassicEstMppConfig(
            tile=args.tile, samples=args.knn_samples, k=args.classic_k,
            seed=args.seed), device)
    cfg = ClassifierEstMppConfig.from_checkpoint(spec['weights_path'])
    return ClassifierEstMpp(cfg, device)


#: Methods whose answer is a vote over per-patch class probabilities, so every
#: `--rules` rule can be applied to ONE forward pass (`patch_probs`).
VOTING_KINDS = ('classifier', 'prototype')

#: The rules compared, FoV_Vote.md's six with the variants it asks for: both
#: tie rules of the patch class median ("分別報告 lower／upper tie rule"), and
#: the quality-weighted mean under each quality signal in
#: `FoVVote.QUALITY_SIGNALS`. label -> (FoVVote rule, doc section, options).
#: Plain `quality_weighted` (no signal) is not a rule here: it IS
#: mean_probability, and is used only as a check on every FoV.
RULES = {
    'mean_probability':         ('mean_probability', 1, {}),
    'hard_majority':            ('hard_majority', 2, {}),
    'patch_class_median:lower': ('patch_class_median', 3, {'tie_rule': 'lower'}),
    'patch_class_median:upper': ('patch_class_median', 3, {'tie_rule': 'upper'}),
    'median_log_rung':          ('median_log_rung', 4, {}),
    'sum_log_probability':      ('sum_log_probability', 5, {}),
    'quality_weighted:tissue':  ('quality_weighted', 6, {'signal': 'tissue'}),
    'quality_weighted:agree':   ('quality_weighted', 6, {'signal': 'agree'}),
}


def _class_of(result, classes) -> int:
    """The class index a result chose -- `estimated_ds` is classes[i] exactly."""
    return min(range(len(classes)),
               key=lambda i: abs(math.log(classes[i] / result.estimated_ds)))


def fov_stats(probs: torch.Tensor, classes_ds, gt_ds: float) -> dict:
    """What one FoV's per-patch distribution looked like -- the columns the
    vote diagnosis in analyze_stage1_metrics.py stratifies by. Independent of
    the vote rule, so every vote row of one FoV carries the same values.

        fov_agree_frac        share of patches whose argmax is the plurality
                              class: 1.0 = every patch agrees
        fov_n_distinct        how many classes some patch picked
        fov_patch_entropy     mean per-patch entropy / log C: how sure each
                              patch is on its own
        fov_pooled_entropy    entropy of the mean distribution / log C
        fov_pooled_margin     top-1 minus top-2 of the mean distribution
        fov_argmax_log2_spread  std of log2(ds) of the patches' argmaxes: how
                              far apart, in octaves, the patches disagree
        fov_gt_reachable      a class within 1% of the true ds exists. A
                              prototype head's classes are the slide's own
                              levels, so ds 2 on a 4x pyramid has none
        fov_gt_prob / fov_gt_rank / fov_gt_patch_frac
                              mean probability on the true class, its rank
                              (0 = top), share of patches whose argmax it is
    """
    p = probs.detach().float().cpu()
    m, c = p.shape
    arg = p.argmax(dim=1)
    counts = torch.bincount(arg, minlength=c)
    pooled = p.mean(dim=0)
    log_c = math.log(c) if c > 1 else 1.0

    def entropy(q):
        return -(q.clamp_min(1e-12).log() * q).sum(dim=-1)

    log2_ds = torch.log2(torch.tensor([float(d) for d in classes_ds]))[arg]
    top = pooled.topk(min(2, c)).values
    gt = min(range(c), key=lambda i: abs(math.log(float(classes_ds[i]) / gt_ds)))
    reachable = abs(math.log(float(classes_ds[gt]) / gt_ds)) < math.log(1.01)
    out = dict(
        fov_n_patches=m, fov_n_classes=c,
        fov_agree_frac=float(counts.max()) / m,
        fov_n_distinct=int((counts > 0).sum()),
        fov_patch_entropy=float(entropy(p).mean()) / log_c,
        fov_pooled_entropy=float(entropy(pooled)) / log_c,
        fov_pooled_margin=float(top[0] - top[1]) if c > 1 else 1.0,
        fov_argmax_log2_spread=float(log2_ds.std(unbiased=False)) if m > 1 else 0.0,
        fov_gt_reachable=reachable)
    if reachable:
        out.update(fov_gt_prob=float(pooled[gt]),
                   fov_gt_rank=int((pooled > pooled[gt]).sum()),
                   fov_gt_patch_frac=float(counts[gt]) / m)
    return out


def result_row(fov: dict, result, vote: str) -> dict:
    """One csv row: the FoV and method columns, the vote rule ('' for a method
    that does not vote), and the EstMppResult."""
    return dict(fov, vote=vote,
                estimated_ds=result.estimated_ds,
                estimated_mpp=result.estimated_mpp,
                chosen_ds=result.chosen_ds, chosen_mpp=result.chosen_mpp,
                chosen_level=result.chosen_level,
                extra_json=json.dumps(_extra_fields(result)))


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
        # checkpoints were selected on. Under --split test this never scores
        # on a slide a checkpoint saw in val; under --split val it does, on
        # purpose: val is where the risk thresholds are fixed, and its
        # accuracies are not results.
        test_names = list_names(dataset=f'{dataset_id}#{args.split}',
                                split_job=args.split_cache_job)
        if args.n_wsi > len(test_names):
            raise ValueError(
                f'--n-wsi {args.n_wsi} but {dataset_id}#{args.split} holds '
                f'{len(test_names)} slide(s); a shorter list than asked for '
                f"would be read as the run's size")
        slides_by_dataset[dataset_id] = test_names[:args.n_wsi]
        print(f'{dataset_id}: {len(slides_by_dataset[dataset_id])} slide(s) '
             f'from the recorded {args.split} split '
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
    fov = fov_from_args(args)
    # The positions are placed for the camera that photographs them (the
    # recipe's gap on its sensor): its sensor, and -- since it rotates -- the
    # bounding square and sensor margin it reads, so the rotated read stays on the
    # tissue the sampler scored. The rungs are each slide's own (`rungs_for`).
    camera = render_spec(fov.gap, fov.sensor)
    slide_cache = {}
    for dataset_id, names in slides_by_dataset.items():
        for wsi_name in names:
            entry = locate(wsi_name, dataset=dataset_id)
            # one microscope per slide, every rung through its own objective;
            # the draw through the sampler cache (FovSupply.cached)
            wsi = SafeSlide(entry.path)
            plan = PlanSpec('ladder', fov.rungs_for(wsi.level_downsamples),
                            camera=camera)
            supply = FovSupply.cached(
                Render(SlideReader(wsi), fov.sensor, fov.gap, ds=1.0), plan,
                fov.sampler,
                masks=caches.masks, draw_job=caches.draw_job,
                render_job=args.render_cache_job or job_name(JOB_NAME),
                save_photos=args.save_photos,
                report_dir=out_dir / 'sampler_reports' / dataset_id.replace('/', '_'))
            sampler = supply.sampler
            mask, _ = caches.masks.mask(wsi)
            # `RungPlan.is_native`: the pyramid has the rung within
            # LEVEL_REL_TOL, so BRACS's 4.00003 / 16.0017 / 32.006 levels are
            # native.
            native_by_rung = {float(p.rung_ds): p.is_native
                              for p in plan.plans_for(wsi)}
            # x, y are the FoV's own top-left (`fov_rect`): what is cropped,
            # centred in the footprint the sampler placed
            positions = [dict(x=int(s.meta.fov_rect[0]), y=int(s.meta.fov_rect[1]),
                              rung=float(s.meta.ds),
                              native=native_by_rung.get(float(s.meta.ds), False),
                              meta=s.meta)
                         for s in sampler]
            if args.fov_per_rung:
                kept, seen = [], {}
                for p in positions:
                    seen[p['rung']] = seen.get(p['rung'], 0) + 1
                    if seen[p['rung']] <= args.fov_per_rung:
                        kept.append(p)
                positions = kept
            print(f'  {wsi_name}: {len(positions)} positions   (mask '
                 f'{"reused" if sampler.cache_info["mask_hit"] else "segmented"}, '
                 f'draw {"reused" if sampler.cache_info["samples_hit"] else "drawn"})')
            slide_cache[(dataset_id, wsi_name)] = (mask, positions, supply)
    caches.masks.close()
    print(f'  [after segmentation] {_mem_snapshot(device)}')

    # Every photo, made ONCE. The read and the domain-gap simulation depend on
    # the FoV alone -- its rng is its position's (`FovSupply.photo`) -- so every method
    # is shown the same photos either way; made inside the method loop they
    # would be ~60% of a method's time. 816 photos of 1440x1024 are ~3.6 GB.
    #
    # Rendered by `Render` (ARCHITECTURE.md: position -> read -> render), one
    # objective per rung, so a 90/270 turn and a scale < 1 read real
    # tissue and the lens ops have their margin.
    t0 = time.perf_counter()
    photos_by_slide = {}
    workers = CpuBudget.for_job(processes=1).workers
    for (dataset_id, wsi_name), (_, positions, supply) in slide_cache.items():
        # the positions to score, by their place in the draw; any other is
        # skipped before it is rendered
        wanted = {id(p['meta']): p for p in positions}
        skip = None if len(wanted) == len(supply.sampler) else (
            lambda m, w=wanted: id(m) not in w)
        photos_by_slide[(dataset_id, wsi_name)] = [
            (wanted[id(meta)], image)
            for _, meta, image, _ in supply.shots(workers=workers, skip=skip)]
        supply.microscope.wsi.close()
    print(f'  [photos] {sum(len(v) for v in photos_by_slide.values())} made once '
          f'in {time.perf_counter() - t0:.0f}s  [{_mem_snapshot(device)}]',
          flush=True)

    rows = []
    n_vote_mismatch = 0
    probs_path = out_dir / f'{_sampling_recipe_id(args)}_probs.jsonl'
    probs_out = open(probs_path, 'w')
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
        # The estimator's own identity, weights included: two checkpoints
        # under one label are two estimators, and analyze_stage1_metrics
        # refuses to carry a val threshold across them.
        estimator_id = estimator.identity_id()
        # Where a method's time goes, summed over its FoVs.
        t_build = t_fwd = t_vote = 0.0
        n_fov = 0

        for dataset_id, names in slides_by_dataset.items():
            for wsi_name in names:
                entry = locate(wsi_name, dataset=dataset_id)
                mask = slide_cache[(dataset_id, wsi_name)][0]
                wsi = SafeSlide(entry.path)

                t0 = time.perf_counter()
                estimator.build(wsi, mask=mask)
                t_build += time.perf_counter() - t0

                for pos, query in photos_by_slide[(dataset_id, wsi_name)]:
                    gt_ds = pos['rung']
                    gt_mpp = wsi.base_mpp * gt_ds
                    # `rung` is the CANONICAL bin (nearest DEFAULT_RUNGS
                    # value in log space), not `gt_ds` itself -- `gt_ds` is
                    # this WSI's own native downsample (e.g. 4.00003), and
                    # grouping by that exact float would give every slide its
                    # own private "rung 4"
                    # instead of letting analyze_stage1_metrics.py aggregate
                    # them as the same one. `gt_ds`/`gt_mpp` keep the exact
                    # value for mpp_error_relative.
                    rung = min(DEFAULT_RUNGS,
                              key=lambda r: abs(np.log(gt_ds) - np.log(r)))
                    n_fov += 1
                    fov = dict(
                        dataset=dataset_id, wsi_name=wsi_name,
                        x=pos['x'], y=pos['y'],
                        h=query.shape[0], w=query.shape[1],
                        rung=rung, native=pos['native'],
                        gt_mpp=gt_mpp, gt_ds=gt_ds,
                        split=args.split, kind=spec['kind'],
                        encoder=spec['encoder'], classifier=spec['classifier'],
                        reduction=spec['reduction'], loss=spec['loss'],
                        read_level=spec['read_level'],
                        weights=spec['weights'], estimator_id=estimator_id)

                    if spec['kind'] not in VOTING_KINDS:
                        t0 = time.perf_counter()
                        rows.append(result_row(fov, estimator.estimate(query), ''))
                        t_fwd += time.perf_counter() - t0
                        continue
                    # One forward pass, every vote rule over it.
                    t0 = time.perf_counter()
                    patches = estimator.query_patches(query)
                    probs = estimator.patch_probs(query)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t_fwd += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    classes = [float(d) for d in estimator.classes_ds]
                    stats = fov_stats(probs, classes, gt_ds)
                    weights = {name: QUALITY_SIGNALS[name](probs, patches)
                               for name in {RULES[r][2]['signal'] for r in args.rules
                                            if 'signal' in RULES[r][2]}}
                    for label in args.rules:
                        vote_name, _, opts = RULES[label]
                        w = weights.get(opts.get('signal'))
                        result = estimator.from_probs(
                            probs, vote_name, weights=w,
                            tie_rule=opts.get('tie_rule', 'lower'))
                        risk = diagnose(vote_name, probs, _class_of(result, classes),
                                        rungs=classes, weights=w)
                        if w is not None:
                            risk['risk_w_mean'] = float(w.float().mean())
                        rows.append({**result_row(fov, result, label), **stats, **risk})
                    # quality_weighted with NO signal is mean_probability by
                    # construction; a disagreement means the dispatch is
                    # broken, not that one rule is better. Checked on every
                    # FoV, never written as a row.
                    if (estimator.from_probs(probs, 'quality_weighted').estimated_ds
                            != estimator.from_probs(probs, 'mean_probability').estimated_ds):
                        n_vote_mismatch += 1
                    probs_out.write(json.dumps(dict(
                        fov, classes_ds=classes,
                        probs=[[round(float(v), 5) for v in row]
                               for row in probs.detach().float().cpu()])) + '\n')
                    t_vote += time.perf_counter() - t0
                wsi.close()

        # Freed before the NEXT spec's build -- this is the whole point of
        # the method-outer loop: only one method's encoder(s) are ever
        # resident at once.
        total = t_build + t_fwd + t_vote
        print(f'  [time] build {t_build:.0f}s (all slides)  {n_fov} FoVs  '
              f'forward {t_fwd:.0f}s  vote+write {t_vote:.0f}s  '
              f'({total / max(n_fov, 1):.2f} s/FoV)', flush=True)
        del estimator
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'  [after free] {_mem_snapshot(device)}')

    probs_out.close()
    print(f'  {probs_path}  (per-patch probabilities, one line per FoV x voting method)')
    if n_vote_mismatch:
        print(f'  [FAIL] quality_weighted and mean_probability disagreed on '
              f'{n_vote_mismatch} FoV(s); with no quality signal they are the '
              f'same rule, so the vote dispatch is broken')
    path = out_dir / f'{_sampling_recipe_id(args)}.csv'
    write_csv(rows, path)
    print(f'\n{path}  ({len(rows)} rows) -- read with '
         f'utilities/cli/metrics/analyze_stage1_metrics.py')
    return 0


# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False)
    parser.add_argument('--tile', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42,
                        help='the estimators\' own: the reference banks and the '
                             'classic estimator. Where the FoVs go is the '
                             'recipe\'s (--sampler-seed)')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
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
    parser.add_argument('--fov-per-rung', type=int, default=0,
                        help='score only the first k positions of every rung '
                             'of the draw (0: all). A smoke run reads the SAME '
                             'cached draw as the full run and scores a subset '
                             'of it, instead of drawing other FoVs with a '
                             'smaller --sampler-n-per-rung')
    # --fov, --max-ds, --sampler-*, --camera-*: the FoV recipe and its overrides.
    add_fov_args(parser)
    # --seg / --mask-cache-job / --draw-cache-job / --split-cache-job.
    add_cache_args(parser)
    parser.add_argument('--render-cache-job', default=None,
                        help='whose cache the photo record (and, with '
                             '--save-photos, the photos) is read from and '
                             'written to. Default: this job')
    parser.add_argument('--save-photos', action='store_true',
                        help='keep every photo beside its record, so a later '
                             'run reads it instead of rendering it')
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
    parser.add_argument('--prototype-weights', nargs='+', default=[],
                        help='one PrototypeEstMpp per PrototypicalRoutingHead '
                             'checkpoint path')
    parser.add_argument('--classic', action='store_true',
                        help='also run ClassicEstMpp, the fingerprint baseline '
                             '(--knn-samples tiles per level, --classic-k)')
    parser.add_argument('--classic-k', type=int, default=3)
    parser.add_argument('--rules', nargs='+', default=list(RULES),
                        choices=list(RULES),
                        help='vote rules applied to every classifier and '
                             'prototype method, all from one forward pass; '
                             'one row per rule. Default: every rule (RULES)')
    parser.add_argument('--split', choices=('val', 'test'), default='test',
                        help='which recorded split to draw slides from. '
                             'FoV_Vote.md fixes every risk threshold on val '
                             'before test is looked at: run val first, '
                             'analyze it with --fit-thresholds, then test')

    parser.add_argument(
        '--out', default=None,
        help='output directory, used verbatim. Default: '
             f'result/<SLURM_JOB_NAME or {JOB_NAME}>/ (no encoder tag: one '
             'CSV can span several encoders)')
    args = parser.parse_args()
    if not (args.knn_encoder or args.classifier_weights or args.prototype_weights
            or args.classic):
        parser.error('needs at least one of --knn-encoder, --classifier-weights, '
                     '--prototype-weights, --classic')

    out_dir = Path(args.out or job_result_dir(JOB_NAME))
    out_dir.mkdir(parents=True, exist_ok=True)
    return run_stage1_compare(args, out_dir)


if __name__ == '__main__':
    sys.exit(main())
