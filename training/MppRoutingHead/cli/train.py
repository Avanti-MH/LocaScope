#!/usr/bin/env python3
'''Train both baselines, select on a held-out val split, save weights.

    python training/MppRoutingHead/cli/train.py            # --baseline all
    python .../train.py --baseline 2 --encoders gigapath
    python .../train.py --baseline 3 --heads linear
    python .../train.py --baseline 2 --heads arcface    # 2-3, baseline 2 only

    baseline 2   the encoder is FROZEN, and WHICH one is a variable
                 (`--encoders`, default gigapath + uni2 -- one run each).
                 Every head trains off that encoder's one forward pass.
    baseline 3   the trunk is FINE-TUNED and is ConvNeXt V2 by definition.
                 One independent run per head, because each head's gradient
                 moves the trunk a different way.

    baseline 1   the KNN (`KnnEstMpp`). Not here: it trains nothing.
                 `bench_stage1_mpp.py` scores it.

THREE SPLITS, SPLIT BY WSI
---------------------------
    train   ki67_pure, every WSI
    val     10 WSIs from EACH of the eval datasets (make_split.py --val-n-wsi)
    test    the rest of those datasets -- NOT touched here at all

The val half is drawn per dataset by `utilities/cli/build_cache/make_split.py`
(run it first) and recorded in `result/cache/<job>_split/<dataset>/
wsi_split.csv`, so which slides
were held out is a recorded fact rather than a consequence of `--seed` -- and
the same fact for every package that reads it. Val spans BOTH eval datasets on purpose: selecting on
it selects for cross-dataset generalisation, which is what this package is for.

Scoring the test split is `cli/evaluate.py`'s job, off a saved checkpoint.
Keeping it out of this file is what stops the test set from being looked at
once per training run.

WHAT THIS WRITES
-----------------
    weights/<encoder>_<frozen|finetuned>_<head>[_<loss>]_<last|best>.pt
    val_scores.csv      one row per (head, epoch)
    sampler_reports/<dataset>_<split>/sampler_report_<slide>.md + samples_<slide>.csv

Positions are drawn per slide through `TileSampler.cached`: masks into
result/cache/<--mask-cache-job>_mask/, draws into <--sampler-cache-job>_sampler/
(`--seg`, default hest; `Datasets.add_cache_args`).

Optionally also one wandb run per MODEL (one per baseline-2 encoder, one per
baseline-3 head -- see `Runtime.wandb_init`), logging the SAME rows that go
into `val_scores.csv`. `--wandb-mode` defaults to the `WANDB_MODE` environment
variable (`online` if unset), so a smoke run stays off the server with
`WANDB_MODE=offline` in the environment rather than a flag here -- see
`jobscripts/_env.sh`.

`best` is by val `level_accuracy` on the val split.

WHY ONE ENCODER PASS FEEDS EVERY HEAD (baseline 2)
----------------------------------------------------
The encoder is frozen, so running it once per batch and handing the result to
every head computes exactly what running it once per head would. What it
produces once is the UN-REDUCED exit (`tokens()`/`spatial()`); `pooled_view`
and `grid_view` then derive the CLS/GAP vector and the patch grid from that
same tensor, so the heads that want a vector and the head that wants a grid
still cost one forward pass between them.

That sharing stops at the epoch boundary and cannot be extended past it: with
per-epoch camera augmentation (spec.md, "Camera: train vs eval") the pixels
differ every epoch, so the encoder genuinely has new input each time. Caching
features across epochs would be caching the augmentation away.

NOT `--arms`/`Arm`/`ARMS`. This file compares heads, but the underlying class (`common.Head.Head`) and its per-task registry
(`Runtime.HEAD_CHOICES`) are also used by `stage1_estimation/
ClassifierEstMpp.py`, which loads exactly ONE trained head to run inference
with -- no comparison happening there, so "arm" would misname what it is
doing. "Head" is the standard word for "the task-specific top of a network"
and does not carry that baggage either way.
'''
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

# _paths holds the one definition of every package's sys.path entry
# (setup_import_paths) -- utilities/ goes on the path here, by hand, because
# that function is INSIDE it and this is the one step nothing else can do
# for this file. Same idiom every test_modules/cli entry point uses.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'utilities'))
from _paths import setup_import_paths                                  # noqa: E402
setup_import_paths()

import torch                                                        # noqa: E402
import torch.nn.functional as F                                     # noqa: E402

import _paths                                                       # noqa: E402
from training.MppRoutingHead.Datasets import (                      # noqa: E402
    NUM_CLASSES, READ_LEVELS, RESAMPLE_FROM, RUNGS, RenderConfig,
    add_cache_args, build_manifest, cache_jobs, class_weights, data_record,
    iterate_epoch,
    open_caches)
from AccessDatasets import list_names                                # noqa: E402
from Heads import HeadConfig, resolve_encoder_layers                # noqa: E402
from Features import check_layer_tokens                              # noqa: E402
from training.MppRoutingHead.Runtime import (                       # noqa: E402
    BASELINE3_ENCODER, HEAD_CHOICES, encode_raw, heads_for, predict,
    rescore, rescore_by_rung, save_checkpoint, trunk_raw,
    wandb_epoch_metrics, wandb_finish, wandb_init, wandb_log, weight_filename)
from Head import Head                                                # noqa: E402
from Resume import ResumeFile, resume_identity                       # noqa: E402

#: Arguments that change where output goes, how many epochs in total, how it is
#: logged, or which models this invocation covers -- not what one model IS.
#: `resume_identity` leaves them out.
_NOT_IDENTITY = ('epochs', 'out', 'device', 'resume_dir', 'wandb_project',
                 'wandb_mode', 'run_name', 'merge', 'num_workers', 'cpu_processes',
                 'encoders',
                 'heads', 'baseline', 'encode_batch', 'mask_cache_job',
                 'sampler_cache_job', 'split_cache_job',
                 # the read mode enters as ONE key, `read_tag`, and only when it
                 # is not the default -- see `_identity`
                 'read_level', 'resample_from', 'max_resample_factor',
                 'resampled_share')


def train_render_cfg(args) -> RenderConfig:
    '''The `RenderConfig` TRAINING renders with: the tile and the read mode.
    Val and test build their own at the defaults, so they always read
    `pyramid`.'''
    return RenderConfig(tile_size=args.tile, read_level=args.read_level,
                        resample_from=args.resample_from,
                        max_resample_factor=args.max_resample_factor,
                        resampled_share=args.resampled_share)


def _identity(args, **more) -> Dict:
    '''The resume identity: the arguments that decide what is trained, plus
    `more`. The read mode is in it as `read_tag` only when it is not the
    default, so a resume file without it is this run's identity at the
    default.'''
    identity = dict(resume_identity(args, _NOT_IDENTITY), **more)
    tag = train_render_cfg(args).read_tag
    if tag:
        identity['read_tag'] = tag
    return identity


def _resume_name(baseline: int, encoder_name: str, loss: str, head: str = '',
                 read_tag: str = '') -> str:
    '''The resume unit: baseline 2 is one ENCODER with every head it feeds
    (one shared forward, one shared epoch loop), baseline 3 one HEAD with its
    own fine-tuned trunk. The loss and the read mode each add a segment only
    when they are not the default, so a default run's file keeps its name.'''
    seg = ('' if loss == 'bal' else f'_{loss}') + (f'_{read_tag}' if read_tag else '')
    return (f'b2_{encoder_name}_frozen{seg}' if baseline == 2 else
            f'b3_{encoder_name}_finetuned_{head}{seg}')


# ══════════════════════════════════════════════════════════════════════════
#  positions -- drawn per slide through TileSampler.cached, which is the only
#  thing this package persists, and it holds no pixels
# ══════════════════════════════════════════════════════════════════════════

def _report_dir(out_dir, dataset_id: str, split: str) -> Path:
    return Path(out_dir) / 'sampler_reports' / f'{dataset_id.replace("/", "_")}_{split}'


def train_rows(args, caches, out_dir) -> List:
    '''Positions only -- the pixels are rendered fresh every epoch (spec.md,
    "Camera: train vs eval"), so there is nothing here to go stale.'''
    rows = build_manifest(
        args.train_dataset, masks=caches.masks,
        sampler_root=caches.sampler_root,
        report_dir=_report_dir(out_dir, args.train_dataset, 'train'),
        tile_size=args.tile, n_per_rung=args.n_per_rung, seed=args.seed,
        max_wsi=args.max_wsi)
    print(f'[train] {args.train_dataset}: {len(rows)} positions', flush=True)
    return rows


def val_rows(args, caches, out_dir) -> List:
    '''The val halves of every eval dataset, concatenated. The split is READ
    (the datasets `<id>#val` and `<id>#test`), never derived here -- `make_split.py` is the
    one place that writes it.'''
    rows: List = []
    for dataset_id in args.eval_datasets:
        val_names = list_names(dataset=f'{dataset_id}#val',
                               split_job=caches.split_job)
        test_names = list_names(dataset=f'{dataset_id}#test',
                                split_job=caches.split_job)
        part = build_manifest(
            dataset_id, masks=caches.masks, sampler_root=caches.sampler_root,
            report_dir=_report_dir(out_dir, dataset_id, 'val'),
            tile_size=args.tile, n_per_rung=args.val_n_per_rung,
            seed=args.seed, wsi_names=val_names)
        print(f'[val]   {dataset_id}: {len(part)} positions from '
              f'{len(val_names)} WSIs ({len(test_names)} held back for test)',
              flush=True)
        rows += part
    return rows


# ══════════════════════════════════════════════════════════════════════════
#  reporting
# ══════════════════════════════════════════════════════════════════════════

def print_composition(rows, seen_class, native_class) -> None:
    '''What the epoch was actually MADE OF -- neither number is in the
    manifest, because both are decided at render time. `trained on` counts what
    survived the off-slide drop (`render_row` returns None near a region edge
    and `collate_routing_batch` drops it, silently); `native` says how much came
    off a pyramid level at the requested mpp rather than being resampled down
    (see `Runtime.score`). Printed rather than assumed: the training set being
    all-native is a property of Ki67's 2x pyramid, not a guarantee.'''
    print(f'    rung        {"  ".join(f"{r:>7g}" for r in RUNGS)}\n'
          f'    trained on  {"  ".join(f"{int(n):>7d}" for n in seen_class)}'
          f'   (of {len(rows)} positions)\n'
          f'    native      {"  ".join(f"{int(n):>7d}" for n in native_class)}',
          flush=True)


def print_class_weights(weights) -> None:
    '''What the manifest's own imbalance turned into -- printed once per
    baseline run, not per epoch, since the weight is fixed for the whole run
    (`Datasets.class_weights`). A weight of 0 means that rung had zero
    training positions; everything else says how many times harder that
    rung's gradient now counts relative to an even split.'''
    print(f'    rung    {"  ".join(f"{r:>7g}" for r in RUNGS)}\n'
          f'    weight  {"  ".join(f"{float(w):>7.3f}" for w in weights)}',
          flush=True)


#: Field width for a head NAME column in the prints below -- wide enough
#: for the longest registered `HEAD_CHOICES` key (`mix_clsattn_mlp_deep`,
#: 20 chars) plus a margin, so a name never runs longer than its field.
#: STILL followed by an explicit literal space at every call site, never
#: relied on alone -- `f'{name:_NAME_W}s'` does not TRUNCATE a name that
#: somehow exceeds this width, it just stops padding, so a name and the
#: field after it would glue together with no separator at all if the
#: field boundary were the only thing keeping them apart
#: (`mlp_deep_residualbracs/test`).
_NAME_W = 22


def val_report(name: str, combined: Dict, rows: List[Dict],
               datasets: List[str], epoch: int, epochs: int, loss: float,
               **identity) -> List[Dict]:
    '''Print the val line and return the CSV rows -- one per dataset plus an
    `all` row, for ONE head. Called once per head, from a loop the CALLER
    owns (`run_baseline2`'s `for name in heads`, `run_baseline3`'s own
    `for name in names`) -- NOT looped over every head internally, so the
    caller calls `val_report` THEN `rung_report` for the SAME head before
    moving to the next one and a head's val and rung blocks print together.

    MUTATES `combined` to add `level_accuracy_unweighted` -- `save_tagged`
    reads it back off the same dict `run_baseline2`/`run_baseline3` already
    hold, rather than this function returning a second value every call
    site would have to thread through.

    PER DATASET, not only combined. `level_accuracy_native` against
    `level_accuracy_resampled` is the check for whether a head is scoring on
    the LANCZOS signature instead of on scale, and it only means that inside
    one dataset: Ki67's 2x pyramid makes every rung native, so over the two
    datasets together the "native" side is BRACS's native rungs plus all of
    Ki67 while the "resampled" side is BRACS alone.

    TWO `all` NUMBERS, not one. Each dataset's OWN `level_accuracy` here is
    the plain mean of its six rungs' own accuracies (not pooled over its
    tiles -- `per_rung`'s mean, not `rescore(dataset_rows)`'s pooled one: RICHNESS's
    coarse-rung shortfall means a tile-pooled average is dominated by
    whichever rungs happen to have the most val tiles). `level_accuracy` ("all, n-weighted") then combines those SIX-
    RUNG-AVERAGED per-dataset numbers weighted by each dataset's own total n
    -- so a dataset with more val positions gets proportionally more say,
    but not at the tile level. `level_accuracy_unweighted` ("all, dataset-avg") is the
    plain mean of the two dataset numbers instead, so both datasets get
    equal say regardless of n. `save_tagged` selects `_best.pt` on the first
    and `_best_unweighted.pt` on the second -- see `Checkpoints.
    weight_filename`'s own docstring for why two files rather than a knob
    (that docstring's own "pooled" wording is stale the same way this one
    was; not fixed here since it lives in the generic layer, shared with
    `PrototypicalRoutingHead`'s checkpoint code).
    '''
    per_dataset = []
    for dataset_id in datasets:
        dataset_rows = [row for row in rows if row['dataset'] == dataset_id]
        result = rescore(dataset_rows)
        per_rung = rescore_by_rung(dataset_rows)
        # 單一 dataset 的 acc = 六個 rung accuracy 直接平均
        result['level_accuracy'] = sum(
            per_rung[rung]['level_accuracy'] for rung in RUNGS) / len(RUNGS)
        per_dataset.append((dataset_id, result))

    # all, n-weighted：dataset accuracy（六個 rung 的平均）根據該 dataset 的
    # sample 數量加權 -- 不是「所有 tile 攤平在一起」的那種 pooled 了
    total_n = sum(r['n'] for _, r in per_dataset)
    combined['level_accuracy'] = (
        sum(r['n'] * r['level_accuracy'] for _, r in per_dataset) / total_n
        if total_n else float('nan'))

    # all, dataset-avg：dataset accuracy 直接平均
    combined['level_accuracy_unweighted'] = (
        sum(r['level_accuracy'] for _, r in per_dataset) / len(per_dataset)
        if per_dataset else float('nan'))

    print(f'    val  epoch {epoch}/{epochs}  {name:{_NAME_W}s} '
         f'acc {combined["level_accuracy"]:.4f} (all, n-weighted)  '
         f'{combined["level_accuracy_unweighted"]:.4f} (all, dataset-avg)',
         flush=True)
    out = []
    for dataset_id, r in per_dataset:
        print(f'         {"":{_NAME_W}s} {dataset_id:17s} '
              f'acc {r["level_accuracy"]:.4f}  '
              f'native {r["level_accuracy_native"]:.4f} (n={r["n_native"]})  '
              f'resampled {r["level_accuracy_resampled"]:.4f} '
              f'(n={r["n_resampled"]})', flush=True)
        # `level_accuracy_unweighted` is an 'all'-row-only property (the
        # mean ACROSS datasets), meaningless for one dataset's own row --
        # still given a value here (NaN) so every row this function
        # emits carries the SAME key set. csv.DictWriter derives its
        # fieldnames from the first row it sees (`main()`'s out_rows[0]`,
        # a per-dataset row): a later row with a key that first one
        # lacks is a ValueError, not a wider table.
        out.append(dict(**identity, head=name, epoch=epoch,
                        train_loss=loss, val_dataset=dataset_id,
                        level_accuracy_unweighted=float('nan'), **r))
    out.append(dict(**identity, head=name, epoch=epoch,
                    train_loss=loss, val_dataset='all', **combined))
    return out


def rung_report(name: str, rows: List[Dict], datasets: List[str],
                epoch: int, **identity) -> List[Dict]:
    '''Print the per-rung val line and return `val_scores_per_rung.csv`'s
    rows: one per (head, val_dataset, rung). Called once per head, right
    after that SAME head's own `val_report` call -- see `val_report`'s own
    docstring for why the caller interleaves the two rather than calling
    each once over every head.

    `val_report`'s own `level_accuracy` pools every rung of a dataset into
    one number, which is exactly what hides whether a specific rung is
    improving (`rescore_by_rung`'s own docstring). This is what answers "did
    rung 32 get better", which no column of `val_scores.csv` can -- written
    every epoch, same as that file, so a training curve per rung is
    possible, not just a final number.

    Per dataset, never pooled across them, same reasoning as `val_report`'s
    own per-dataset loop: which rungs are native differs by pyramid, so a
    rung's accuracy on bracs/test and on ki67_with_photo are not the same
    quantity even when they carry the same rung label.
    '''
    out = []
    for dataset_id in datasets:
        per_rung = rescore_by_rung(
            [r for r in rows if r['dataset'] == dataset_id])
        print(f'         {name:{_NAME_W}s} {dataset_id:17s} rung   '
             + '  '.join(f'{r:>7g}' for r in RUNGS), flush=True)
        print(f'         {"":{_NAME_W}s} {"":17s} acc    '
             + '  '.join(f'{per_rung[r]["level_accuracy"]:>7.4f}'
                         for r in RUNGS), flush=True)
        for rung, r in per_rung.items():
            out.append(dict(**identity, head=name, epoch=epoch,
                            val_dataset=dataset_id, rung=rung, **r))
    return out


# ══════════════════════════════════════════════════════════════════════════
#  baseline 2: one frozen encoder, every head off its one forward pass
# ══════════════════════════════════════════════════════════════════════════

def _head_cfg_for(name: str, in_dim: int, args, depth: int = 0) -> HeadConfig:
    '''`HeadConfig` for one head NAME -- reads `HEAD_CHOICES[name]`'s
    optional 4th element (a dict of `HeadConfig` field overrides, e.g.
    `mlp_deep`'s `{'mlp_depth': 2}`) on top of the CLI's own `--mlp-*`
    defaults, so `mlp`/`mlp_deep`/`mlp_wide`/`mlp_deep_wide` can all be
    trained in the SAME run, each with its own architecture -- see
    `Runtime.HEAD_CHOICES`'s own comment for why this has to be resolved per
    name rather than built once and shared.

    `encoder_layers` is resolved here, against the encoder's `depth`, the same
    way `mlp_width_mult` is resolved against `in_dim`: the registry writes one
    spread for every encoder, the config records the blocks it meant.
    '''
    overrides = HEAD_CHOICES[name][3] if len(HEAD_CHOICES[name]) > 3 else {}
    layers = overrides.get('encoder_layers', ())
    if layers:
        if depth < 1:
            raise ValueError(f'{name} mixes encoder layers and this encoder has '
                             f'no block depth to resolve them against')
        layers = resolve_encoder_layers(layers, depth)
    mlp_width = (int(in_dim * overrides['mlp_width_mult'])
                if 'mlp_width_mult' in overrides else
                overrides.get('mlp_width', args.mlp_width))
    return HeadConfig(
        in_dim=in_dim, num_classes=NUM_CLASSES, dtype='fp32',
        mlp_depth=overrides.get('mlp_depth', args.mlp_depth),
        mlp_width=mlp_width,
        mlp_residual=overrides.get('mlp_residual', args.mlp_residual),
        mlp_dropout=args.mlp_dropout,
        encoder_layers=layers)


def _compute_loss(logits: torch.Tensor, target: torch.Tensor, weights,
                  loss_kind: str, ordinal_weight: float,
                  ordinal_sigma: float) -> torch.Tensor:
    '''`spec.md`'s "Ordinal-aware loss" section, formulas 2/3a/3b, as one
    switch so both baselines' training loops share it rather than each
    growing their own copy.

        L_bal   = -w_y log p_y                              (today's default)
        L_ord_a = L_bal + lambda * sum_i w_i (E_c[ell_c] - ell_y)^2 / sum_i w_i
                                                     (regression term, class-weighted)
        L_ord_b = w_y * (-sum_c q_c(y) log p_c)              (soft target)

    `ell_c = log2(rung_c)`, matching `analyze_stage1_metrics.nearest_rung`'s
    own reasoning that pyramid scales are geometric, not linear. `ord_b`
    REPLACES the CE target rather than adding to it -- `q_c(y)` already
    carries `w_y` outside the sum (the class weight scales the whole
    per-sample loss, the same normalisation `F.cross_entropy(weight=...)`
    uses: divide by the batch's SUMMED weight, not by N), so there is no
    separate `L_bal` term added underneath it the way `ord_a` does.

    `loss_kind='bal'` ignores `ordinal_weight`/`ordinal_sigma` entirely and
    is exactly the `F.cross_entropy(logits, target, weight=weights)` call
    this replaces -- a run that never passes `--loss` is byte-for-byte the
    old behaviour.
    '''
    if loss_kind == 'bal':
        return F.cross_entropy(logits, target, weight=weights)

    log2_rungs = torch.log2(
        torch.tensor(RUNGS, dtype=torch.float32, device=logits.device))

    if loss_kind == 'ord_a':
        l_bal = F.cross_entropy(logits, target, weight=weights)
        probs = F.softmax(logits.float(), dim=-1)
        expected_log_rung = (probs * log2_rungs).sum(dim=-1)
        true_log_rung = log2_rungs[target]
        # Class-weighted like L_bal. Unweighted, the regression
        # term is a plain mean over the batch, and rung 32 -- 0.3 per cent of
        # the training manifest -- would barely move it.
        sq = (expected_log_rung - true_log_rung).pow(2)
        if weights is None:
            l_ord = sq.mean()
        else:
            w = weights[target]
            l_ord = (w * sq).sum() / w.sum()
        return l_bal + ordinal_weight * l_ord

    if loss_kind == 'ord_b':
        # [N, K] squared log2-rung distance from each sample's true rung to
        # every rung, then a softmax turns "closer" into "more soft-target
        # mass" -- sigma->0 collapses this to one-hot, i.e. L_bal's own
        # target, which is why 'bal' (not sigma=0) is the off switch.
        dist2 = (log2_rungs.unsqueeze(0) - log2_rungs[target].unsqueeze(1)).pow(2)
        soft_target = F.softmax(-dist2 / (2 * ordinal_sigma ** 2), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        per_sample = -(soft_target * log_probs).sum(dim=-1)
        if weights is None:
            return per_sample.mean()
        w = weights[target]
        return (w * per_sample).sum() / w.sum()

    raise ValueError(f'unknown --loss {loss_kind!r}')


def _clip_grad(opt, max_norm: float) -> None:
    '''No-op when `max_norm` is falsy (0 -- the default, i.e. clipping off).
    Otherwise clips every param THIS optimizer owns, read off its own
    `param_groups` rather than passed in separately -- the same call works
    whether `opt` owns only a head's params (baseline 2, frozen trunk) or a
    head's AND a trunk's together (baseline 3, one optimizer, two LR
    groups), without the caller needing to know which. Call AFTER
    `loss.backward()`, BEFORE `opt.step()` -- clipping acts on the
    gradients backward just computed, not on the parameters themselves.
    '''
    if not max_norm:
        return
    torch.nn.utils.clip_grad_norm_(
        [p for g in opt.param_groups for p in g['params']], max_norm)


def _apply_warmup(opt, epoch: int, warmup_epochs: int, base_lrs: List[float]) -> None:
    '''No-op when `warmup_epochs` is falsy (0 -- the default, i.e. warmup
    off) or once `epoch` has passed it. Otherwise linearly ramps every param
    group's LR from 0 up to its own `base_lrs[i]` over the first
    `warmup_epochs` epochs -- lets a freshly-initialised head settle before
    it is pushing a pretrained trunk at full strength.

    `base_lrs` is captured ONCE, right after the optimizer is built
    (`[g['lr'] for g in opt.param_groups]`) -- this function overwrites
    `param_group['lr']` every call, so reading it back from the optimizer on
    a later epoch would read last epoch's SCALED value, not the true
    target, and the ramp would decay toward zero instead of climbing to it.

    Call at the TOP of each epoch, before that epoch's batch loop -- one
    call per epoch, not per batch; the ramp is epoch-grained. A RESUMED run
    continues from the epoch after the saved one, so it ramps exactly where the
    uninterrupted run would have been.
    '''
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return
    scale = epoch / warmup_epochs
    for group, base_lr in zip(opt.param_groups, base_lrs):
        group['lr'] = base_lr * scale


def run_baseline2(args, encoder_name: str, caches, out_dir,
                  device) -> Tuple[List[Dict], List[Dict]]:
    from TileEncoderFunc import encoder_config                   # noqa: PLC0415
    names = heads_for(args.heads, 2)
    if not names:
        return [], []
    # `--dtype` governs the FROZEN ENCODER's inference precision here --
    # fp16 forward, no gradient, the same case `TileEncoderFunc` was built for
    # (cos=0.99995 against fp32, log/TODO.log). It does NOT reach `head_cfg`
    # below: see the comment there for why that is a different question with
    # a fixed answer, not a knob.
    #
    # `dtype` is NOT a top-level field of `TileEncoderConfig` -- it lives on
    # the nested `ModelConfig`, so `encoder_config(name, dtype=...)` fails with
    # `TypeError: __init__() got an unexpected keyword argument 'dtype'`
    # (`config_from` forwards `**over` straight into the dataclass's own
    # `__init__`, which has no `dtype` field to catch). `variant(dtype=...)`
    # does the nested-replace, but only on an already-BUILT `TileEncoder`
    # instance -- here the config still needs building, so the replace has to
    # happen on the CONFIG directly, the same pattern `run_baseline3` already
    # uses for the trunk below.
    base_cfg = encoder_config(encoder_name)
    cfg = replace(base_cfg, model=replace(base_cfg.model, dtype=args.dtype))
    encoder = cfg.build(device)
    spec = encoder.model_spec
    num_prefix = int(spec.num_prefix)
    # ALWAYS fp32, never `args.dtype`. This head is trained by Adam with no
    # GradScaler, and fp16 there is not a slower-but-safe choice, it is a
    # guaranteed eventual NaN -- `HeadConfig.dtype`'s own docstring has the
    # mechanism. `--dtype` defaults to 'fp16', so passing it through would
    # override the dataclass default at every construction site.
    # PER HEAD, not one shared `head_cfg` -- see `_head_cfg_for`'s own
    # docstring: mlp/mlp_deep/mlp_wide/mlp_deep_wide each need their own
    # depth/width, and a single shared config could only ever give them all
    # the same one.
    depth = encoder.depth if spec.kind == 'tokens' else 0
    head_cfgs = {n: _head_cfg_for(n, int(spec.dim), args, depth) for n in names}
    heads = {n: Head(head_cfgs[n], *HEAD_CHOICES[n][:2]).to(device) for n in names}
    # Every block any head mixes, read in the one forward they share. Empty --
    # no mix_ head in the run -- keeps encode_raw's plain tensor.
    mix_blocks = tuple(sorted({b for c in head_cfgs.values()
                               for b in c.encoder_layers}))
    for n in names:
        if head_cfgs[n].encoder_layers:
            print(f'  {n}: mixes encoder blocks {head_cfgs[n].encoder_layers} '
                  f'of {depth}', flush=True)
    print(f'baseline 2  {encoder_name}  dim={spec.dim} kind={spec.kind} '
          f'num_prefix={num_prefix}  (frozen)', flush=True)

    rows = train_rows(args, caches, out_dir)
    vrows = val_rows(args, caches, out_dir)
    caches.masks.close()        # the segmenter is not needed past here
    # Fixed for the whole run -- see Datasets.class_weights. `weights=None`
    # under --class-weight none restores plain unweighted cross_entropy.
    weights = class_weights(rows, device) if args.class_weight == 'balanced' else None
    if weights is not None:
        print_class_weights(weights)
    # (best_n_weighted, best_unweighted) per head -- see save_tagged.
    best = {n: (-1.0, -1.0) for n in names}
    opts = {n: torch.optim.Adam(h.parameters(), lr=args.lr)
            for n, h in heads.items()}
    # Captured BEFORE any warmup call or resume overwrites `param_group['lr']`
    # -- see `_apply_warmup`'s own docstring for why.
    base_lrs = {n: [g['lr'] for g in opt.param_groups] for n, opt in opts.items()}
    out: List[Dict] = []
    out_rung: List[Dict] = []

    # RESUME: one file for the encoder and every head it feeds, since they
    # share one forward and one epoch loop (Resume.py has the rule).
    resume = ResumeFile.for_model(args.resume_dir,
                                  _resume_name(2, encoder_name, args.loss,
                                               read_tag=train_render_cfg(args).read_tag))
    identity = _identity(args, baseline=2, encoder=encoder_name,
                         heads=list(names))
    # Which blocks a mix_ head read is not in its NAME, so a registry edit that
    # kept the name would otherwise resume onto weights for other blocks. Only
    # heads that mix are listed, and the key only exists when one does.
    mixed = {n: list(c.encoder_layers) for n, c in head_cfgs.items()
             if c.encoder_layers}
    if mixed:
        identity['encoder_layers'] = mixed
    start_epoch = 0
    state = resume.load(identity)
    if state is not None:
        ResumeFile.restore(state, modules=heads, optimizers=opts)
        start_epoch = int(state['epoch'])
        best = state['best']
        out, out_rung = state['extra']
        print(f'  [resume] {resume.path}: continuing after epoch {start_epoch} '
              f'of {args.epochs}', flush=True)
    elif resume.enabled:
        print(f'  [resume] no {resume.path.name} yet -- training from scratch, '
              f'writing it every epoch', flush=True)
    end_epoch = args.epochs
    raw_of = lambda p: encode_raw(encoder, p, args.encode_batch, device,  # noqa: E731
                                  layers=mix_blocks)
    layers_checked = False

    # ONE run for this encoder, covering every head trained off it -- see
    # `Runtime.wandb_init`'s docstring for why the boundary is per encoder
    # here and per head in `run_baseline3`.
    slurm_job_name = os.environ.get('SLURM_JOB_NAME', '')
    prefix = args.run_name or slurm_job_name
    run_name = f'{prefix}-b2-{encoder_name}' if prefix else f'b2-{encoder_name}'
    read_cfg = train_render_cfg(args)
    run_name += f'-{read_cfg.read_tag}' if read_cfg.read_tag else ''
    # No epoch left: no run. It would log nothing, and an empty run is a row in
    # every chart's legend with no line.
    wb = None if start_epoch >= end_epoch else wandb_init(
        args.wandb_project, args.wandb_mode, run_name, config=dict(
        baseline=2, encoder=encoder_name, heads=names,
        encoder_dtype=args.dtype, head_dtype='fp32',
        class_weight=args.class_weight, loss=args.loss, seg=args.seg,
        ordinal_weight=args.ordinal_weight, ordinal_sigma=args.ordinal_sigma,
        lr=args.lr, epochs=args.epochs, n_per_rung=args.n_per_rung,
        tile=args.tile, batch_size=args.batch_size, seed=args.seed,
        train_dataset=args.train_dataset, eval_datasets=args.eval_datasets,
        slurm_job_id=os.environ.get('SLURM_JOB_ID', ''),
        slurm_job_name=slurm_job_name, read_level_label=read_cfg.read_label,
        **cache_jobs(args, 'MppRoutingHead', caches)),
        run_id=resume.wandb_run_id(state is not None))

    for epoch in range(start_epoch + 1, end_epoch + 1):
        for n, opt in opts.items():
            _apply_warmup(opt, epoch, args.warmup_epochs, base_lrs[n])
        for head in heads.values():
            head.train()
        totals = {n: [0.0, 0] for n in heads}
        seen_class = torch.zeros(NUM_CLASSES, dtype=torch.int64)
        native_class = torch.zeros(NUM_CLASSES, dtype=torch.int64)
        for batch in iterate_epoch(
                rows, wsi_group_size=args.wsi_group_size,
                batch_size=args.batch_size, num_workers=args.num_workers,
                cfg=train_render_cfg(args), epoch_seed=epoch):
            # ONE pass, every head. Moved to the device once here rather than
            # per head: the heads differ in how they reduce it, not in which
            # copy of it they read.
            if mix_blocks and not layers_checked:
                (same, decoy), ok = check_layer_tokens(encoder, batch['patches'])
                print(f'  layer tokens: last block vs tokens() min cos '
                      f'{same:.6f}, block before it {decoy:.6f}  '
                      f'{"OK" if ok else "FAIL"}', flush=True)
                if not ok:
                    raise RuntimeError(
                        'layer_tokens does not reproduce tokens() on its last '
                        'block, so every mix_ head would train on something '
                        'else than its name says')
                layers_checked = True
            raw = raw_of(batch['patches'])
            target = batch['labels'].to(device)
            seen_class += torch.bincount(batch['labels'], minlength=NUM_CLASSES)
            native_class += torch.bincount(batch['labels'][batch['native']],
                                           minlength=NUM_CLASSES)
            for name, head in heads.items():
                loss = _compute_loss(head(raw, num_prefix, target), target,
                                     weights, args.loss, args.ordinal_weight,
                                     args.ordinal_sigma)
                opts[name].zero_grad()
                loss.backward()
                _clip_grad(opts[name], args.clip_grad_norm)
                opts[name].step()
                totals[name][0] += float(loss) * len(target)
                totals[name][1] += len(target)

        print(f'  epoch {epoch}/{end_epoch}  loss  ' + '  '.join(
            f'{n} {t[0] / max(t[1], 1):.4f}' for n, t in totals.items()),
            flush=True)
        print_composition(rows, seen_class, native_class)

        # The per-tile detail is used for the per-dataset breakdown and then
        # dropped; only the TEST split's rows are written out (evaluate.py),
        # since ten epochs of val rows would be ten CSVs nobody opens.
        val, detail = predict(vrows, heads, raw_of, num_prefix, tile=args.tile,
                              wsi_group_size=args.wsi_group_size,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)
        # PER HEAD, val_report then rung_report for the SAME head before
        # moving to the next one -- see val_report's own docstring for why.
        epoch_rows: List[Dict] = []
        epoch_rung_rows: List[Dict] = []
        for name in heads:
            epoch_rows += val_report(
                name, val[name], detail[name], args.eval_datasets, epoch,
                end_epoch, totals[name][0] / max(totals[name][1], 1),
                baseline=2, encoder=encoder_name, loss_kind=args.loss,
                read_level=train_render_cfg(args).read_label, seg=args.seg)
            epoch_rung_rows += rung_report(
                name, detail[name], args.eval_datasets, epoch,
                baseline=2, encoder=encoder_name, loss_kind=args.loss,
                read_level=train_render_cfg(args).read_label, seg=args.seg)
        out += epoch_rows
        out_rung += epoch_rung_rows
        for name, head in heads.items():
            best[name] = save_tagged(out_dir, head, encoder, encoder_name, True,
                                     name, head_cfgs[name], args, epoch, val[name],
                                     best[name], opts[name])
        resume.save(identity, epoch=epoch, modules=heads, optimizers=opts,
                    best=best, extra=(out, out_rung))
        # Logged AFTER the resume file is written. Before it, a job killed between
        # the two leaves wandb holding an epoch the resume file does not: the rerun
        # trains that epoch again, wandb refuses its step as already logged, and
        # the curve keeps the dead job's numbers.
        wandb_log(wb, epoch, wandb_epoch_metrics(epoch_rows))
    wandb_finish(wb)
    return out, out_rung


# ══════════════════════════════════════════════════════════════════════════
#  baseline 3: the trunk is fine-tuned, so every head gets its OWN copy
# ══════════════════════════════════════════════════════════════════════════

def run_baseline3(args, caches, out_dir, device) -> Tuple[List[Dict], List[Dict]]:
    '''ONE INDEPENDENT RUN PER HEAD, not one loop over several heads sharing a
    trunk. Baseline 2's heads share an encoder pass because the encoder is
    frozen; here each head's gradient moves the trunk a different way, so
    sharing one trunk would make the heads' training interfere. Each gets its
    own encoder -- which means loading the checkpoint once per head, the cost
    of the independence.'''
    from TileEncoderFunc import encoder_config                   # noqa: PLC0415
    names = heads_for(args.heads, 3)
    if not names:
        return [], []
    rows = train_rows(args, caches, out_dir)
    vrows = val_rows(args, caches, out_dir)
    caches.masks.close()        # the segmenter is not needed past here
    # Fixed for the whole run and the SAME across heads (all trained off the
    # same `rows`) -- computed once here rather than per head.
    weights = class_weights(rows, device) if args.class_weight == 'balanced' else None
    if weights is not None:
        print_class_weights(weights)
    out: List[Dict] = []
    out_rung: List[Dict] = []

    for name in names:
        reduction, classifier = HEAD_CHOICES[name][:2]
        # THE TRUNK IS TRAINED, SO IT IS FP32. `ConvNeXtV2EncoderConfig` sets
        # dtype='fp16', which is right for the frozen use it was written for
        # and wrong here for the same reason `HeadConfig.dtype` is fp32: Adam
        # updating fp16 parameters without a GradScaler produces NaN.
        # 28M parameters in fp32 is ~112 MB of weights plus optimizer state --
        # nothing, on this card.
        base = encoder_config(BASELINE3_ENCODER)
        cfg = replace(base, model=replace(base.model, dtype='fp32'))
        encoder = cfg.build(device)
        # ALWAYS fp32, never `args.dtype` -- same reasoning as `run_baseline2`'s
        # `head_cfg`.
        head_cfg = _head_cfg_for(name, int(encoder.model_spec.dim), args)
        head = Head(head_cfg, reduction, classifier).to(device)
        print(f'baseline 3  head {name} (reduction={reduction}, '
              f'classifier={classifier.__name__}): fine-tuning '
              f'{BASELINE3_ENCODER} dim={head_cfg.in_dim} '
              f'trunk_lr={args.trunk_lr} head_lr={args.lr}', flush=True)
        best = (-1.0, -1.0)     # (best_n_weighted, best_unweighted)
        end_epoch = args.epochs

        # Two param groups: a trunk pretrained on natural images and a head
        # initialised at random do not want the same step size -- the standard
        # fine-tuning split, and the reason --trunk-lr defaults below --lr.
        opt = torch.optim.Adam([
            {'params': encoder.model.parameters(), 'lr': args.trunk_lr},
            {'params': head.parameters(), 'lr': args.lr},
        ])
        # Captured BEFORE any warmup call or resume overwrites
        # `param_group['lr']` -- see `_apply_warmup`'s own docstring for why.
        base_lrs = [g['lr'] for g in opt.param_groups]

        modules = {'head': head, 'trunk': encoder.model}
        resume = ResumeFile.for_model(
            args.resume_dir, _resume_name(3, BASELINE3_ENCODER, args.loss, name,
                                          read_tag=train_render_cfg(args).read_tag))
        identity = _identity(args, baseline=3, encoder=BASELINE3_ENCODER,
                             head=name)
        start_epoch = 0
        head_rows: List[Dict] = []
        head_rung_rows: List[Dict] = []
        state = resume.load(identity)
        if state is not None:
            ResumeFile.restore(state, modules=modules, optimizers={'adam': opt})
            start_epoch = int(state['epoch'])
            best = tuple(state['best'])
            head_rows, head_rung_rows = state['extra']
            print(f'  [resume] {resume.path}: continuing after epoch '
                  f'{start_epoch} of {args.epochs}', flush=True)
        elif resume.enabled:
            print(f'  [resume] no {resume.path.name} yet -- training from '
                  f'scratch, writing it every epoch', flush=True)
        raw_of = lambda p: trunk_raw(encoder, p, device)          # noqa: E731

        # ONE run for this head -- it owns its own trunk/optimizer/epoch loop,
        # so it is one experiment on its own (contrast baseline 2's per-
        # ENCODER boundary, where several heads share one epoch axis).
        slurm_job_name = os.environ.get('SLURM_JOB_NAME', '')
        prefix = args.run_name or slurm_job_name
        run_label = f'{prefix}-b3-{name}' if prefix else f'b3-{name}'
        read_cfg = train_render_cfg(args)
        run_label += f'-{read_cfg.read_tag}' if read_cfg.read_tag else ''
        # No `dtype=args.dtype` here -- baseline 3 has nothing left for it to
        # control (trunk and head are both hardcoded fp32, above), so logging
        # it would imply a knob that does not exist for this baseline.
        wb = None if start_epoch >= end_epoch else wandb_init(
            args.wandb_project, args.wandb_mode, run_label, config=dict(
            baseline=3, encoder=BASELINE3_ENCODER, head=name, reduction=reduction,
            classifier=classifier.__name__, trunk_dtype='fp32', head_dtype='fp32',
            class_weight=args.class_weight, loss=args.loss, seg=args.seg,
            ordinal_weight=args.ordinal_weight, ordinal_sigma=args.ordinal_sigma,
            lr=args.lr, trunk_lr=args.trunk_lr, epochs=args.epochs,
            n_per_rung=args.n_per_rung, tile=args.tile, batch_size=args.batch_size,
            seed=args.seed, train_dataset=args.train_dataset,
            eval_datasets=args.eval_datasets,
            slurm_job_id=os.environ.get('SLURM_JOB_ID', ''),
            slurm_job_name=slurm_job_name, read_level_label=read_cfg.read_label,
            **cache_jobs(args, 'MppRoutingHead', caches)),
            run_id=resume.wandb_run_id(state is not None))

        for epoch in range(start_epoch + 1, end_epoch + 1):
            _apply_warmup(opt, epoch, args.warmup_epochs, base_lrs)
            encoder.train()     # see TileEncoder.train: the reflex, reaching
            head.train()        # past `encoder` to the trunk it holds
            total, seen = 0.0, 0
            seen_class = torch.zeros(NUM_CLASSES, dtype=torch.int64)
            native_class = torch.zeros(NUM_CLASSES, dtype=torch.int64)
            for batch in iterate_epoch(
                    rows, wsi_group_size=args.wsi_group_size,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    cfg=train_render_cfg(args), epoch_seed=epoch):
                target = batch['labels'].to(device)
                seen_class += torch.bincount(batch['labels'], minlength=NUM_CLASSES)
                native_class += torch.bincount(batch['labels'][batch['native']],
                                               minlength=NUM_CLASSES)
                loss = _compute_loss(head(raw_of(batch['patches']), 0, target),
                                     target, weights, args.loss,
                                     args.ordinal_weight, args.ordinal_sigma)
                opt.zero_grad()
                loss.backward()
                _clip_grad(opt, args.clip_grad_norm)
                opt.step()
                total += float(loss) * len(target)
                seen += len(target)

            print(f'  epoch {epoch}/{end_epoch}  loss {total / max(seen, 1):.4f}',
                  flush=True)
            print_composition(rows, seen_class, native_class)

            encoder.eval()      # dropout / drop_path off while scoring
            val, detail = predict(vrows, {name: head}, raw_of, 0, tile=args.tile,
                                  wsi_group_size=args.wsi_group_size,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
            epoch_rows = val_report(
                name, val[name], detail[name], args.eval_datasets, epoch,
                end_epoch, total / max(seen, 1),
                baseline=3, encoder=BASELINE3_ENCODER, loss_kind=args.loss,
                read_level=train_render_cfg(args).read_label, seg=args.seg)
            head_rows += epoch_rows
            head_rung_rows += rung_report(
                name, detail[name], args.eval_datasets, epoch,
                baseline=3, encoder=BASELINE3_ENCODER, loss_kind=args.loss,
                read_level=train_render_cfg(args).read_label, seg=args.seg)
            best = save_tagged(out_dir, head, encoder, BASELINE3_ENCODER, False,
                               name, head_cfg, args, epoch, val[name], best, opt)
            resume.save(identity, epoch=epoch, modules=modules,
                        optimizers={'adam': opt}, best=best,
                        extra=(head_rows, head_rung_rows))
            # after the resume file, as in run_baseline2
            wandb_log(wb, epoch, wandb_epoch_metrics(epoch_rows))
        wandb_finish(wb)
        out += head_rows
        out_rung += head_rung_rows
    return out, out_rung


# ══════════════════════════════════════════════════════════════════════════

def save_tagged(out_dir, head: Head, encoder, encoder_name: str, frozen: bool,
                head_name: str, head_cfg, args, epoch: int, val: Dict,
                best, optimizer):
    '''Write `_last` every epoch, `_best` when the POOLED val accuracy
    improves, and `_best_unweighted` when the per-dataset-AVERAGED val
    accuracy improves -- see `val_report`'s and `Checkpoints.weight_filename`'s
    own docstrings for why these two can pick different epochs. `best` is
    `(best_n_weighted, best_unweighted)` in, the same tuple (possibly updated)
    out.

    THREE files rather than one per epoch: with `--epochs 10` and several
    heads, per-epoch checkpoints are tens of GB of things nobody will open,
    and the only ones anyone asks for are "the one training ended on" and
    "the one that scored best" -- now two answers to "scored best", since
    the two weightings do not always agree.

    `optimizer.state_dict()` still travels with every file, for whoever
    wants to fine-tune from one; RESUMING a run is `--resume-dir`'s job
    (`aiNNModel/models/common/Resume.py`), not these files'.'''
    best_n_weighted, best_unweighted = best
    wdir = Path(out_dir) / 'weights'
    # `extra`: this package's own label-space facts, which the generic
    # checkpoint format (`aiNNModel/models/common/Checkpoints.py`) has no
    # opinion about -- see `save_checkpoint`'s own docstring.
    read_tag = train_render_cfg(args).read_tag
    common = dict(head=head, encoder=encoder, encoder_name=encoder_name,
                  frozen=frozen, head_name=head_name, head_cfg=head_cfg,
                  epoch=epoch, val=val, run_args=vars(args),
                  optimizer_state=optimizer.state_dict(),
                  # `rungs`: the label space this head classifies into --
                  # `stage1_estimation/ClassifierEstMpp.py` reads it back
                  # to turn a predicted class into a ds value, so it has to
                  # travel WITH the checkpoint rather than be re-imported from
                  # this package's own `Datasets.RUNGS` at inference time,
                  # which could drift out of sync with what a specific
                  # checkpoint was actually trained against.
                  extra=dict(tile_size=args.tile, rungs=RUNGS,
                             # what the tiles were made of; evaluate.py
                             # refuses test data whose record differs
                             data=data_record(args.tile, args.seg)))
    save_checkpoint(wdir / weight_filename(encoder_name, frozen, head_name,
                                           'last', loss=args.loss,
                                           read_tag=read_tag), **common)

    acc = val['level_accuracy']
    if acc > best_n_weighted:
        save_checkpoint(wdir / weight_filename(encoder_name, frozen, head_name,
                                               'best', loss=args.loss,
                                               read_tag=read_tag), **common)
        print(f'      new best, n-weighted ({acc:.4f}) -> '
              f'{weight_filename(encoder_name, frozen, head_name, "best", loss=args.loss, read_tag=read_tag)}',
              flush=True)
        best_n_weighted = acc

    acc_u = val['level_accuracy_unweighted']
    if acc_u > best_unweighted:
        save_checkpoint(wdir / weight_filename(encoder_name, frozen, head_name,
                                               'best_unweighted', loss=args.loss,
                                               read_tag=read_tag), **common)
        print(f'      new best, unweighted ({acc_u:.4f}) -> '
              f'{weight_filename(encoder_name, frozen, head_name, "best_unweighted", loss=args.loss, read_tag=read_tag)}',
              flush=True)
        best_unweighted = acc_u

    return best_n_weighted, best_unweighted


def _merge_val_scores(path: Path, out_rows: List[Dict]) -> List[Dict]:
    '''Under `--merge`: `out_rows`' own `(baseline, encoder, head, loss_kind)`
    keys REPLACE the matching rows already in `path` -- a rerun of one head
    overwrites that head's history, one epoch row at a time, same as a plain
    overwrite would if it were the only head trained. Every OTHER head's
    rows already on disk are kept untouched, and a key `path` has never seen
    is a plain append. `path` not existing yet is just `out_rows` alone --
    the first run has nothing to merge with.

    `loss_kind` is in the key: without it, a `bal` run and an `ord_a` run of
    the SAME (baseline, encoder, head) would --merge into ONE row per epoch,
    one silently overwriting the other's history. `.get(..., 'bal')`: a row
    without a `loss_kind` column was trained bal.

    `read_level` is in the key for the same reason: a `resampled` run of a
    (baseline, encoder, head, loss) would otherwise replace the `pyramid`
    run's rows. A row without it, or with it empty, is `pyramid`.

    `csv.DictReader` reads every value as a str (`baseline` is `'2'`/`'3'`,
    not `2`/`3`) -- the key comparison normalizes `out_rows`' own values to
    str on that side rather than parsing the file's ints back, since the
    only thing this needs from either side is string equality, not the
    values themselves.
    '''
    if not path.exists():
        return out_rows
    def key(r):
        return (str(r['baseline']), r['encoder'], r['head'],
                r.get('loss_kind') or 'bal', r.get('read_level') or 'pyramid')
    new_keys = {key(r) for r in out_rows}
    with open(path, newline='') as fh:
        kept = [r for r in csv.DictReader(fh) if key(r) not in new_keys]
    return kept + out_rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Names as `TileEncoderFunc._IMPLEMENTATIONS` spells them, so a typo is a
    # refusal here rather than a KeyError after the weights start loading.
    # baseline 3's encoder is NOT among these -- it is bound, see
    # BASELINE3_ENCODER.
    ap.add_argument('--encoders', nargs='+',
                    choices=['gigapath', 'uni2', 'conch_vit'],
                    default=['gigapath', 'uni2'],
                    help='baseline 2 only; one frozen run per name')
    ap.add_argument('--heads', nargs='+', choices=sorted(HEAD_CHOICES),
                    default=sorted(HEAD_CHOICES))
    ap.add_argument('--train-dataset', default='ki67_pure')
    ap.add_argument('--eval-datasets', nargs='+',
                    default=['bracs/test', 'ki67_with_photo'],
                    help='val is drawn from these; the rest is test, which '
                         'only cli/evaluate.py ever touches')
    ap.add_argument('--val-n-per-rung', type=int, default=20)
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--n-per-rung', type=int, default=100)
    ap.add_argument('--wsi-group-size', type=int, default=8,
                    help='WSIs open at once per worker; see spec.md "Handle count"')
    ap.add_argument('--batch-size', type=int, default=16,
                    help='PATCHES per batch; one manifest row is one patch')
    ap.add_argument('--encode-batch', type=int, default=256,
                    help='PATCHES per frozen-encoder call (baseline 2)')
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--cpu-processes', type=int, default=1,
                    help='training processes this job runs side by side (the '
                         'jobscript runs baselines 2 and 3 as two). Each takes '
                         'cpus / this for its workers and threads -- CpuBudget')
    ap.add_argument('--baseline', choices=['2', '3', 'all'], default='all',
                    help='2 = frozen encoder, every head off one pass; '
                         '3 = fine-tuned trunk, one independent run per head; '
                         'all (default) = both. baseline 1 (the KNN, '
                         'KnnEstMpp) is not here: it trains nothing')
    ap.add_argument('--class-weight', choices=('balanced', 'none'),
                    default='balanced',
                    help="'balanced' (default): F.cross_entropy(weight=...) "
                         "from the training manifest's own rung distribution "
                         "(Datasets.class_weights) -- RICHNESS's coarse-rung "
                         "shortfall means rung 32 supplies roughly 1 percent "
                         "of what rung 1 does, and an unweighted loss lets "
                         "the common rungs' gradient drown the rare ones out. "
                         "'none' restores plain unweighted cross_entropy, for "
                         "comparison")
    ap.add_argument('--loss', choices=('bal', 'ord_a', 'ord_b'), default='bal',
                    help="spec.md's \"Ordinal-aware loss\" section, formulas "
                         "2/3a/3b. 'bal' (default): today's weighted CE alone "
                         "(--class-weight decides the weight -- this flag is "
                         "orthogonal to that one, not a replacement for it). "
                         "'ord_a': + a regression penalty on the softmax's "
                         "own expected log2-rung against the true one "
                         "(--ordinal-weight). 'ord_b': REPLACES the one-hot "
                         "target with a Gaussian kernel over log2-rung "
                         "distance (--ordinal-sigma) -- 'bal' is what sigma->0 "
                         "would recover, so there is no off-by-default value "
                         "for sigma the way --ordinal-weight=0 is for ord_a")
    ap.add_argument('--read-level', choices=READ_LEVELS, default='pyramid',
                    help="which pyramid level a TRAINING tile is read from "
                         "(spec.md 詞彙). 'pyramid' (default): the nearest-level "
                         "rule every run before this used. 'resampled': a finer "
                         "level, drawn uniformly, then resampled down. 'mixed': "
                         "'resampled' for --resampled-share of the tiles. Val "
                         "and test always read 'pyramid'")
    ap.add_argument('--resample-from', choices=RESAMPLE_FROM, default='finer',
                    help="'finer' (default): any level finer than the rung. "
                         "'l0': level 0 only")
    ap.add_argument('--max-resample-factor', type=float, default=None,
                    help='drop a candidate level that needs more than this '
                         'much downsampling (a read cost bound: the read side '
                         'is tile x factor). Default: no bound')
    ap.add_argument('--resampled-share', type=float, default=0.5,
                    help="--read-level mixed only: the share of tiles read "
                         "'resampled' (default 0.5)")
    ap.add_argument('--ordinal-weight', type=float, default=1.0,
                    help="ord_a's lambda: (E_c[log2 rung] - log2 rung_true)^2, "
                         "weight on top of the weighted-CE term. Unvalidated "
                         "starting value, same status as the routing recipes' "
                         "optics probability 0.5 -- sweep it")
    ap.add_argument('--ordinal-sigma', type=float, default=1.0,
                    help="ord_b's kernel bandwidth in log2-rung units. 1.0 = "
                         "one rung-step gets meaningful soft-target mass, two "
                         "steps away much less. Unvalidated starting value")
    ap.add_argument('--mlp-depth', type=int, default=1,
                    help="MlpHead only ('mlp'/'attn_linear' do NOT use "
                         "this -- 'attn_linear' is LinearHead behind an attn "
                         'reduction, not MlpHead; only --heads mlp is '
                         'affected). Number of hidden layers, each '
                         '--mlp-width wide. 1 (default) reproduces the '
                         'original single-hidden-layer MlpHead.')
    ap.add_argument('--mlp-width', type=int, default=None,
                    help='MlpHead only. Hidden width, the SAME for every '
                         'hidden layer. Default (unset) is in_dim, same as '
                         'before this flag existed.')
    ap.add_argument('--mlp-residual', action='store_true',
                    help='MlpHead only. Residual connection around each '
                         'width-to-width hidden layer (the 2nd through '
                         '--mlp-depth-th) -- see Heads.MlpHead\'s own '
                         'docstring for why the first and last layers are '
                         'excluded. Nothing to wrap at --mlp-depth 1.')
    ap.add_argument('--mlp-dropout', type=float, default=0.1,
                    help='MlpHead only.')
    ap.add_argument('--epochs', type=int, default=10,
                    help='TOTAL epochs, resumed or not')
    ap.add_argument('--lr', type=float, default=1e-3, help='the HEAD')
    ap.add_argument('--trunk-lr', type=float, default=1e-4,
                    help='baseline 3 only. Below --lr on purpose: the trunk is '
                         'pretrained and the head is random')
    ap.add_argument('--clip-grad-norm', type=float, default=0.0,
                    help='0 (default) = off. Clips every optimizer this '
                         'file builds to this L2 norm right before each '
                         'opt.step() -- caps how far one batch can move any '
                         'parameter, trunk or head. See _clip_grad\'s own '
                         'docstring.')
    ap.add_argument('--warmup-epochs', type=int, default=0,
                    help='0 (default) = off. Linearly ramps every '
                         "optimizer's own LR from 0 up to its target over "
                         'the first this-many epochs -- lets a freshly-'
                         'initialised head settle before it is pushing a '
                         'pretrained trunk at full strength. See '
                         '_apply_warmup\'s own docstring.')
    ap.add_argument('--dtype', choices=['fp16', 'fp32'], default='fp32',
                    help='baseline 2 ONLY, and only the FROZEN encoder\'s '
                         'inference precision. Every trained parameter in '
                         'this file (every head, and baseline 3\'s trunk) is '
                         'hardcoded fp32 regardless of this flag -- Adam '
                         'without a GradScaler makes fp16 training an '
                         'eventual NaN, not a speed/accuracy trade-off. '
                         'Defaults to fp32 (not the fp16 the encoder itself '
                         'defaults to elsewhere) so this file never NEEDS '
                         'the flag to avoid a NaN -- pass --dtype fp16 for '
                         'the faster, still-safe frozen-inference path '
                         '(cos=0.99995 against fp32, log/TODO.log) once the '
                         'pipeline itself is trusted')
    add_cache_args(ap)
    ap.add_argument('--seed', type=int, default=42,
                    help='sampling AND the val/test WSI split')
    ap.add_argument('--max-wsi', type=int, default=None,
                    help='smoke run: N training WSIs, chosen at random with --seed')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None)
    ap.add_argument('--resume-dir', default=None,
                    help='write each model\'s full state here every epoch '
                         '(<model>_resume.pt: weights, optimizer, epoch, best '
                         'scores, RNG, the val rows so far), and continue from '
                         'it when it is already there. Unset: train from '
                         'scratch and write nothing. --epochs is the TOTAL, so '
                         'a finished model resumes into nothing. A model is an '
                         'encoder with all its heads (baseline 2) or one head '
                         'with its trunk (baseline 3)')
    ap.add_argument('--wandb-project', default='mpp-routing-head')
    # Reads WANDB_MODE, same convention as SuperPathPoint/FewShotEoMT's own
    # --wandb-mode: `wandb.init(mode=...)` takes an EXPLICIT argument, which
    # beats the env var, so the default has to read it rather than hardcode a
    # literal (jobscripts/_env.sh has the full note).
    ap.add_argument('--wandb-mode', default=os.environ.get('WANDB_MODE', 'online'),
                    choices=('online', 'offline', 'disabled'))
    ap.add_argument('--run-name', default='',
                    help='prefixed onto the per-model run name '
                         '(b2-<encoder> / b3-<head>). Empty means no prefix, '
                         "NOT wandb's own random name generator -- every run "
                         'gets an explicit, deterministic name either way')
    ap.add_argument('--merge', action='store_true',
                    help="val_scores.csv: keep every OTHER (baseline, "
                         "encoder, head)'s rows already on disk and only "
                         "replace this run's own -- e.g. retraining just "
                         "convnext_v2/attn_linear (--baseline 3 --heads "
                         "attn_linear) keeps the other heads' rows. Off by "
                         'default: a plain overwrite is what a full run wants.')
    args = ap.parse_args()
    try:
        read_cfg = train_render_cfg(args)       # refused here, not mid-run
    except ValueError as exc:
        ap.error(str(exc))
    print(f'  training read mode: {read_cfg.read_label}', flush=True)
    # The render runs in --num-workers DataLoader workers; this process keeps
    # what is left of the cpus. With torch's default it ran 8 threads beside 8
    # workers and its encode took 3x as long (CpuBudget's docstring).
    from CpuBudget import CpuBudget                                 # noqa: PLC0415
    print(f'  {CpuBudget.for_job(processes=args.cpu_processes, workers=args.num_workers).apply().line()}',
          flush=True)

    device = torch.device(args.device)
    caches = open_caches(args, 'MppRoutingHead', device)
    out_dir = Path(args.out or _paths.job_result_dir('MppRoutingHead'))
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = ('2', '3') if args.baseline == 'all' else (args.baseline,)
    print(f'baseline {" + ".join(wanted)}   '
          f'heads: {", ".join(f"{n} ({HEAD_CHOICES[n][0]})" for n in args.heads)}')
    if '2' in wanted:
        print(f'  baseline 2 encoders: {", ".join(args.encoders)}')
    if '3' in wanted:
        print(f'  baseline 3 encoder:  {BASELINE3_ENCODER} (bound)')
    print(f'  class_weight: {args.class_weight}')
    print(f'  loss: {args.loss}' + (
        f'  ordinal_weight={args.ordinal_weight}' if args.loss == 'ord_a' else
        f'  ordinal_sigma={args.ordinal_sigma}' if args.loss == 'ord_b' else ''))
    # NOT "wandb auto-names it" when --run-name is unset -- run_baseline2/3
    # build a real name either way (`b2-<encoder>`, `b3-<head>`, prefixed by
    # --run-name, else by the job's name). wandb's own random name generator
    # ("electric-unicorn-42") never fires here.
    prefix = (args.run_name or os.environ.get('SLURM_JOB_NAME', '')
              or '(none -- named b2-<encoder> / b3-<head>)')
    print(f'  wandb project={args.wandb_project}   mode={args.wandb_mode}   '
         f'run_name prefix={prefix}')

    out_rows: List[Dict] = []
    out_rung_rows: List[Dict] = []
    if '2' in wanted:
        for encoder_name in args.encoders:
            rows, rung_rows = run_baseline2(args, encoder_name, caches,
                                            out_dir, device)
            out_rows += rows
            out_rung_rows += rung_rows
    if '3' in wanted:
        rows, rung_rows = run_baseline3(args, caches, out_dir, device)
        out_rows += rows
        out_rung_rows += rung_rows

    if not out_rows:
        # Reachable without a bug: every requested head can be one the
        # requested baseline does not have (`--baseline 3 --heads arcface`).
        # Say so -- writing a header-only CSV would read as "all heads scored
        # zero".
        print('\nno head ran: every requested head belongs to another baseline')
        return 1

    # NO encoder level in the path. CLAUDE.md's rule is there so two encoders'
    # outputs cannot overwrite each other; one run covering several of them
    # writes ONE file that names the encoder in a COLUMN instead, which answers
    # the same question and keeps the comparison in one table. The weights
    # underneath DO carry the encoder, in their filenames.
    path = out_dir / 'val_scores.csv'
    write_rows = _merge_val_scores(path, out_rows) if args.merge else out_rows
    with open(path, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        wr.writeheader()
        wr.writerows(write_rows)
    print(f'\n{path}  ({len(write_rows)} rows'
         f'{f", merged with existing (kept {len(write_rows) - len(out_rows)})" if args.merge else ""})')

    # Per-rung breakdown -- see rung_report's own docstring for why this is
    # a SEPARATE file rather than a `rung` column added to val_scores.csv:
    # that file's one row per (head, val_dataset) would become six, and
    # every existing reader of it (save_tagged does not, but a dashboard or
    # a by-hand `awk` over it might) would see rows multiply under it
    # without warning. Same merge key, same reasoning, reused as-is.
    if out_rung_rows:
        rung_path = out_dir / 'val_scores_per_rung.csv'
        write_rung_rows = (_merge_val_scores(rung_path, out_rung_rows)
                          if args.merge else out_rung_rows)
        with open(rung_path, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(out_rung_rows[0].keys()))
            wr.writeheader()
            wr.writerows(write_rung_rows)
        print(f'{rung_path}  ({len(write_rung_rows)} rows)')

    print(f'{out_dir / "weights"}  '
          f'(*_last.pt and *_best.pt; cli/evaluate.py scores them on test)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
