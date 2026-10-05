#!/usr/bin/env python3
"""Stage-1 mpp estimators compared head to head on the same drawn FoVs.

    python utilities/bench_modules/bench_stage1_mpp.py \
        --knn-encoder gigapath uni2 --classifier-weights <ckpt> ... \
        --prototype-weights <ckpt> ... --classic \
        --datasets bracs/test ki67_with_photo --n-wsi 9

Every stage-1 method, each run on the SAME synthetic FoVs (drawn once per
slide, from the RECORDED test split) across --n-wsi slides of every
--datasets, and shown the same photo of each (`fov_rng`):

    KnnEstMpp          one per --knn-encoder
    ClassifierEstMpp   one per --classifier-weights checkpoint (MppRoutingHead)
    PrototypeEstMpp    one per --prototype-weights checkpoint
                       (PrototypicalRoutingHead)
    ClassicEstMpp      --classic, the fingerprint baseline

The two that vote over per-patch probabilities (classifier, prototype) run
once per FoV and are scored under every --votes rule (FoVVote), one row per
rule, with the FoV's distribution statistics (`fov_stats`) on each row. Writes

    result/<SLURM_JOB_NAME or Stage1MppBench>/<sampler_id>_<seg_id>_<region_id>.csv
    result/<...>/<sampler_id>_<seg_id>_<region_id>_probs.jsonl

-- one row per (FoV, method, vote), and the raw per-patch probabilities once
per (FoV, voting method). utilities/cli/metrics/analyze_stage1_metrics.py
scores them, the vote diagnosis included. The scoring lives there, not here:
this bench only produces rows.

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
import hashlib
import json
import math
import random
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
from PrototypeEstMpp import PrototypeEstMpp, PrototypeEstMppConfig  # noqa: E402
from estimate_mpp_classic import ClassicEstMpp, ClassicEstMppConfig  # noqa: E402
from FoVVote import VOTE_CHOICES                                    # noqa: E402
from StageInterface import EstMppResult                             # noqa: E402
from _paths import job_result_dir                                   # noqa: E402
from AccessDatasets import list_names, locate                        # noqa: E402
from training.MppRoutingHead.Datasets import (                      # noqa: E402
    add_cache_args, open_caches, read_label_of)
from SafeSlide import SafeSlide                                     # noqa: E402
from TissueMaskConfig import MASK_RECIPES                           # noqa: E402
from TileSampler import (OverlapConfig, PlanSpec, RichnessConfig,     # noqa: E402
                         SamplerConfig, TileSampler)
from DsLadder import DEFAULT_RUNGS                                  # noqa: E402
from camera import sensor_size                                       # noqa: E402
from SlideReader import SlideReader                                 # noqa: E402
from ReadGeometry import ReadSpec                                 # noqa: E402
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
    existing top-up mechanism reachable, without changing `step` (the
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
                          reduction='', loss='', read_level='', weights='', weights_path=None,
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
            # how the head was TRAINED (2026-10-02), the same gap one level
            # further: a pyramid- and a resampled-trained head of one loss
            # would share a label. `read_label_of` reads `pyramid` off a
            # checkpoint saved before the read mode existed.
            read_level=read_label_of(ckpt.get('args')),
            weights=os.path.basename(weights), weights_path=weights,
            needs_mask=False))
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
            weights=os.path.basename(weights), weights_path=weights,
            needs_mask=True))
    if args.classic:
        specs.append(dict(kind='classic', encoder='classic', classifier='',
                          reduction='', loss='', read_level='', weights='',
                          weights_path=None, needs_mask=True))
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
            sampler_cfg=SamplerConfig(n_per_rung=args.knn_samples,
                                      seed=args.seed,
                                      richness=REFERENCE_BANK_RICHNESS,
                                      overlap=_overlap_cfg(args.overlap)),
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
#: `--votes` rule can be applied to ONE forward pass (`patch_probs`).
VOTING_KINDS = ('classifier', 'prototype')


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


def fov_rng(seed: int, dataset: str, wsi_name: str, pos: dict) -> random.Random:
    """The photo simulation's rng for one FoV, the same for every method:
    seeded by the bench seed and the FoV's identity, through sha256 because
    Python's hash() of a str changes per process."""
    key = f'{seed}|{dataset}|{wsi_name}|{pos["x"]}|{pos["y"]}|{pos["rung"]!r}'
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


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
        # checkpoints were selected on, so this bench never scores on a slide
        # a checkpoint saw in val.
        test_names = list_names(dataset=f'{dataset_id}#test',
                                split_job=args.split_cache_job)
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
    sampler_cfg = SamplerConfig(n_per_rung=args.n_per_rung,
                                seed=args.seed, richness=RichnessConfig(),
                                overlap=_overlap_cfg(args.overlap))
    # The positions are placed for the camera that photographs them: this
    # bench's query FoV (--ratio, --mpixels), cropped without rotation --
    # `simulate_microscope_photo` turns the crop itself. Placing a 256 px tile
    # and then reading a 1440 px FoV there, as it did before 2026-10-03, put
    # the photographed rectangle partly outside the tissue the sampler scored.
    camera = ReadSpec(*sensor_size(args.ratio, args.mpixels))
    plan = (PlanSpec('native', camera=camera) if args.native_only
            else PlanSpec('ladder', tuple(DEFAULT_RUNGS), camera=camera))
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
                              for p in plan.plans_for(wsi)}
            # x, y are the FoV's own top-left (`fov_rect`): what is cropped,
            # centred in the footprint the sampler placed
            positions = [dict(x=int(s.meta.fov_rect[0]), y=int(s.meta.fov_rect[1]),
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

        for dataset_id, names in slides_by_dataset.items():
            for wsi_name in names:
                entry = locate(wsi_name, dataset=dataset_id)
                mask, positions = slide_cache[(dataset_id, wsi_name)]
                wsi = SafeSlide(entry.path)
                # the photo, straight off the slide (lanczos, as the
                # microscope simulation always read), then the domain gap
                reader = SlideReader(wsi)
                photo = ReadSpec(*sensor_size(args.ratio, args.mpixels))

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
                    image = reader.read(pos['x'], pos['y'], photo, gt_ds)
                    if image is None:
                        continue
                    # The rng is the FoV's own, so every method is shown the
                    # SAME photo. It was the global `random` until 2026-10-05:
                    # each method then got its own augmentation and rotation
                    # of one FoV, and the comparison was paired on the
                    # position only.
                    query = simulate_microscope_photo(
                        image, rng=fov_rng(args.seed, dataset_id, wsi_name, pos))
                    fov = dict(
                        dataset=dataset_id, wsi_name=wsi_name,
                        x=pos['x'], y=pos['y'],
                        h=query.shape[0], w=query.shape[1],
                        rung=rung, native=pos['native'],
                        gt_mpp=gt_mpp, gt_ds=gt_ds,
                        kind=spec['kind'],
                        encoder=spec['encoder'], classifier=spec['classifier'],
                        reduction=spec['reduction'], loss=spec['loss'],
                        read_level=spec['read_level'],
                        weights=spec['weights'])

                    if spec['kind'] not in VOTING_KINDS:
                        rows.append(result_row(fov, estimator.estimate(query), ''))
                        continue
                    # One forward pass, every vote rule over it.
                    probs = estimator.patch_probs(query)
                    classes = [float(d) for d in estimator.classes_ds]
                    stats = fov_stats(probs, classes, gt_ds)
                    picked = {}
                    for vote_name in args.votes:
                        result = estimator.from_probs(probs, vote_name)
                        picked[vote_name] = result.estimated_ds
                        rows.append({**result_row(fov, result, vote_name), **stats})
                    # quality_weighted with no quality signal IS
                    # mean_probability; a disagreement means the dispatch is
                    # broken, not that one rule is better.
                    if ('quality_weighted' in picked and 'mean_probability' in picked
                            and picked['quality_weighted'] != picked['mean_probability']):
                        n_vote_mismatch += 1
                    probs_out.write(json.dumps(dict(
                        fov, classes_ds=classes,
                        probs=[[round(float(v), 5) for v in row]
                               for row in probs.detach().float().cpu()])) + '\n')
                wsi.close()

        # Freed before the NEXT spec's build -- this is the whole point of
        # the method-outer loop: only one method's encoder(s) are ever
        # resident at once.
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
                        help='query W:H ratio (camera.sensor_size). Defaulted '
                             "explicitly to CLAUDE.md's real-photo spec; "
                             "bench_mpp_feature_decomposition's run_sampler_"
                             "routing still reads 4:3 (flagged, not fixed, "
                             "2026-09-16)")
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
    parser.add_argument('--prototype-weights', nargs='+', default=[],
                        help='one PrototypeEstMpp per PrototypicalRoutingHead '
                             'checkpoint path')
    parser.add_argument('--classic', action='store_true',
                        help='also run ClassicEstMpp, the fingerprint baseline '
                             '(--knn-samples tiles per level, --classic-k)')
    parser.add_argument('--classic-k', type=int, default=3)
    parser.add_argument('--votes', nargs='+', default=list(VOTE_CHOICES),
                        choices=list(VOTE_CHOICES),
                        help='FoVVote rules applied to every classifier and '
                             'prototype method, all from one forward pass; '
                             'one row per rule. Default: every rule')

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
