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
                 `bench_mpp_feature_decomposition.py` scores it.

THREE SPLITS, SPLIT BY WSI
---------------------------
    train   ki67_pure, every WSI
    val     `--val-n-wsi` (10) WSIs from EACH of the eval datasets
    test    the rest of those datasets -- NOT touched here at all

The val half is drawn per dataset by `Datasets.split_wsi_names` and written to
`wsi_split.csv`, so which slides were held out is a recorded fact rather than a
consequence of `--seed`. Val spans BOTH eval datasets on purpose: selecting on
it selects for cross-dataset generalisation, which is what this package is for.

Scoring the test split is `cli/evaluate.py`'s job, off a saved checkpoint.
Keeping it out of this file is what stops the test set from being looked at
once per training run.

WHAT THIS WRITES
-----------------
    weights/<encoder>_<frozen|finetuned>_<head>_<last|best>.pt
    val_scores.csv      one row per (head, epoch)

Optionally also one wandb run per MODEL (one per baseline-2 encoder, one per
baseline-3 head -- see `Runtime.wandb_init`), logging the SAME rows that go
into `val_scores.csv`. `--wandb-mode` defaults to the `WANDB_MODE` environment
variable (`online` if unset), so a smoke run stays off the server with
`WANDB_MODE=offline` in the environment rather than a flag here -- see
`jobscripts/_env.sh`.

`best` is by val `level_accuracy`, which is a meaningful quantity only because
there is a val split; before there was one, every run's number came from
whatever the last epoch happened to be.

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

NOT `--arms`/`Arm`/`ARMS` (2026-09-17). Those names came from a context where
several configurations were trained SIDE BY SIDE for comparison -- "arm" the
way a clinical trial has arms. This file still compares (that is what it is
for), but the underlying class (`common.Head.Head`) and its per-task registry
(`Runtime.HEAD_CHOICES`) are also used by `1_estimate_query_mpp/
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
from typing import Dict, List

# _paths holds the one definition of every package's sys.path entry
# (setup_import_paths) -- utilities/ goes on the path here, by hand, because
# that function is INSIDE it and this is the one step nothing else can do
# for this file. Same idiom every test_modules/cli entry point uses.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'utilities'))
from _paths import setup_import_paths                                   # noqa: E402
setup_import_paths()

import torch                                                        # noqa: E402
import torch.nn.functional as F                                     # noqa: E402

import _paths                                                       # noqa: E402
from Datasets import (NUM_CLASSES, RUNGS, RenderConfig,             # noqa: E402
                      build_manifest, class_weights, iterate_epoch,
                      manifest_parts, manifest_path, read_manifest,
                      write_manifest, write_manifest_key, wsi_split)
from Heads import HeadConfig                                        # noqa: E402
from Runtime import (BASELINE3_ENCODER, HEAD_CHOICES, encode_raw,   # noqa: E402
                     heads_for, predict, rescore, save_checkpoint,
                     trunk_raw, wandb_epoch_metrics, wandb_finish,
                     wandb_init, wandb_log, weight_filename)
from Head import Head                                                # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
#  manifests -- the only thing this package persists, and it holds no pixels
# ══════════════════════════════════════════════════════════════════════════

def _manifest(ddir, split: str, parts, build) -> List:
    '''Read the manifest this key names, or build and write it. The key is a
    hash of everything the content depends on (`manifest_parts`), so a changed
    `--tile`/`--seed`/`--n-per-rung` cannot land on a stale file.'''
    path = manifest_path(ddir, split, parts)
    if path.exists():
        return read_manifest(path)
    rows = build()
    write_manifest(rows, path)
    write_manifest_key(path, parts)
    return rows


def train_rows(args, cache_root) -> List:
    '''Positions only -- the pixels are rendered fresh every epoch (spec.md,
    "Camera: train vs eval"), so there is nothing here to go stale.'''
    ddir = cache_root / args.train_dataset.replace('/', '_')
    parts = manifest_parts(dataset_id=args.train_dataset, tile_size=args.tile,
                           n_per_rung=args.n_per_rung, seed=args.seed,
                           max_wsi=args.max_wsi)
    rows = _manifest(ddir, 'train', parts, lambda: build_manifest(
        args.train_dataset, tile_size=args.tile, n_per_rung=args.n_per_rung,
        seed=args.seed, max_wsi=args.max_wsi))
    print(f'[train] {args.train_dataset}: {len(rows)} positions', flush=True)
    return rows


def val_rows(args, cache_root) -> List:
    '''The val halves of every eval dataset, concatenated. The split itself is
    recorded next to the manifest (`wsi_split.csv`) rather than left implicit
    in `--seed`, and an EXISTING record is reused rather than re-derived --
    see `Datasets.wsi_split`.'''
    rows: List = []
    for dataset_id in args.eval_datasets:
        ddir = cache_root / dataset_id.replace('/', '_')
        val_names, test_names = wsi_split(dataset_id, args.val_n_wsi,
                                          ddir / 'wsi_split.csv', seed=args.seed)
        parts = manifest_parts(dataset_id=dataset_id, tile_size=args.tile,
                               n_per_rung=args.val_n_per_rung, seed=args.seed,
                               n_wsi=args.val_n_wsi, wsi_names=val_names)
        part = _manifest(ddir, 'val', parts, lambda: build_manifest(
            dataset_id, tile_size=args.tile, n_per_rung=args.val_n_per_rung,
            seed=args.seed, wsi_names=val_names))
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
    survived the off-slide drop (`_render_row` returns None near a region edge
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


def val_report(val: Dict[str, Dict], detail: Dict[str, List[Dict]],
               datasets: List[str], epoch: int, epochs: int,
               losses: Dict[str, float], **identity) -> List[Dict]:
    '''Print the val line and return the CSV rows -- one per dataset plus an
    `all` row, per head. MUTATES `combined` (i.e. `val[name]`) to add
    `level_accuracy_unweighted` -- `save_tagged` reads it back off the same
    dict `run_baseline2`/`run_baseline3` already hold, rather than this
    function returning a second value every call site would have to thread
    through.

    PER DATASET, not only combined. `level_accuracy_native` against
    `level_accuracy_resampled` is the check for whether a head is scoring on
    the LANCZOS signature instead of on scale, and it only means that inside
    one dataset: Ki67's 2x pyramid makes every rung native, so over the two
    datasets together the "native" side is BRACS's native rungs plus all of
    Ki67 while the "resampled" side is BRACS alone.

    TWO `all` NUMBERS, not one -- `level_accuracy` (pooled: every val example
    from both datasets scored together, so a dataset with more val positions
    gets proportionally more say -- 2026-09-18, BRACS had 1094 against Ki67's
    847) and `level_accuracy_unweighted` (the plain mean of each dataset's
    OWN accuracy, so both count equally regardless of how many positions
    either happened to contribute). `save_tagged` selects `_best.pt` on the
    first and `_best_unweighted.pt` on the second -- see
    `Checkpoints.weight_filename`'s own docstring for why two files rather
    than a knob.
    '''
    out = []
    for name, combined in val.items():
        rows = detail[name]
        per_dataset = [(d, rescore([r for r in rows if r['dataset'] == d]))
                       for d in datasets]
        combined['level_accuracy_unweighted'] = (
            sum(r['level_accuracy'] for _, r in per_dataset) / len(per_dataset)
            if per_dataset else float('nan'))
        print(f'    val  epoch {epoch}/{epochs}  {name:12s} '
              f'acc {combined["level_accuracy"]:.4f} (all, pooled)  '
              f'{combined["level_accuracy_unweighted"]:.4f} (all, dataset-avg)',
              flush=True)
        for dataset_id, r in per_dataset:
            print(f'         {"":21s}{dataset_id:17s} '
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
                            train_loss=losses[name], val_dataset=dataset_id,
                            level_accuracy_unweighted=float('nan'), **r))
        out.append(dict(**identity, head=name, epoch=epoch,
                        train_loss=losses[name], val_dataset='all', **combined))
    return out


# ══════════════════════════════════════════════════════════════════════════
#  baseline 2: one frozen encoder, every head off its one forward pass
# ══════════════════════════════════════════════════════════════════════════

def _maybe_resume(resume_dir, encoder_name: str, frozen: bool, head_name: str,
                  head: Head, encoder):
    '''Loads `head` (and, for a fine-tuned run, `encoder`'s trunk) from an
    existing `weight_filename(..., 'best')` checkpoint under `resume_dir`, if
    one is there. Returns `(start_epoch, best, optimizer_state)`:

      start_epoch       the checkpoint's own `epoch` -- callers run
                        `range(start_epoch + 1, start_epoch + args.epochs + 1)`
                        so `--epochs` always means "this many MORE epochs",
                        resumed or not, and a from-scratch run (start_epoch=0)
                        gets the exact range it always has.
      best              the checkpoint's own val accuracy, not -1.0, so an
                        early epoch that dips before the optimizer re-settles
                        cannot overwrite `_best` with something worse than
                        what was already loaded in.
      optimizer_state   `ckpt['optimizer_state']` for the caller to
                        `opt.load_state_dict(...)` AFTER building `opt` --
                        this function never builds one itself. `None` for a
                        checkpoint saved before that key existed, or when
                        there is nothing to resume from -- either way the
                        caller just skips the load and Adam starts fresh.

    `(0, -1.0, None)` when `resume_dir` is None or has no matching file,
    which is exactly training from scratch.
    '''
    if not resume_dir:
        return 0, -1.0, None
    path = Path(resume_dir) / weight_filename(encoder_name, frozen, head_name, 'best')
    if not path.exists():
        print(f'  [resume] no {path.name} in {resume_dir} -- training from scratch')
        return 0, -1.0, None
    ckpt = torch.load(path, map_location='cpu')
    head.load_state_dict(ckpt['head_state'])
    if not frozen:
        encoder.model.load_state_dict(ckpt['trunk_state'])
    acc = ckpt['val']['level_accuracy']
    start_epoch = int(ckpt['epoch'])
    opt_state = ckpt.get('optimizer_state')
    print(f'  [resume] {head_name}: resumed from {path} (epoch {start_epoch}, '
         f'val level_accuracy={acc:.4f}, optimizer_state='
         f'{"yes" if opt_state is not None else "no (pre-optimizer-state checkpoint)"})',
         flush=True)
    return start_epoch, acc, opt_state


def _head_cfg_for(name: str, in_dim: int, args) -> HeadConfig:
    '''`HeadConfig` for one head NAME -- reads `HEAD_CHOICES[name]`'s
    optional 4th element (a dict of `HeadConfig` field overrides, e.g.
    `mlp_deep`'s `{'mlp_depth': 2}`) on top of the CLI's own `--mlp-*`
    defaults, so `mlp`/`mlp_deep`/`mlp_wide`/`mlp_deep_wide` can all be
    trained in the SAME run, each with its own architecture -- see
    `Runtime.HEAD_CHOICES`'s own comment for why this has to be resolved per
    name rather than built once and shared.
    '''
    overrides = HEAD_CHOICES[name][3] if len(HEAD_CHOICES[name]) > 3 else {}
    mlp_width = (int(in_dim * overrides['mlp_width_mult'])
                if 'mlp_width_mult' in overrides else
                overrides.get('mlp_width', args.mlp_width))
    return HeadConfig(
        in_dim=in_dim, num_classes=NUM_CLASSES, dtype='fp32',
        mlp_depth=overrides.get('mlp_depth', args.mlp_depth),
        mlp_width=mlp_width,
        mlp_residual=overrides.get('mlp_residual', args.mlp_residual),
        mlp_dropout=args.mlp_dropout)


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
    it is pushing a pretrained trunk at full strength (the mechanism behind
    the 2026-09-17 convnext_v2/attn_linear collapse, epoch 6 of that run).

    `base_lrs` is captured ONCE, right after the optimizer is built
    (`[g['lr'] for g in opt.param_groups]`) -- this function overwrites
    `param_group['lr']` every call, so reading it back from the optimizer on
    a later epoch would read last epoch's SCALED value, not the true
    target, and the ramp would decay toward zero instead of climbing to it.

    Call at the TOP of each epoch, before that epoch's batch loop -- one
    call per epoch, not per batch; the ramp is epoch-grained; and on a
    RESUMED run `epoch` starts above `warmup_epochs` already (`_maybe_resume`
    returns the checkpoint's own `epoch`, and training continues from
    `epoch + 1`), so this is naturally a no-op for a resumed head, which is
    correct -- it is not freshly initialised, so there is nothing for
    warmup to protect it from.
    '''
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return
    scale = epoch / warmup_epochs
    for group, base_lr in zip(opt.param_groups, base_lrs):
        group['lr'] = base_lr * scale


def run_baseline2(args, encoder_name: str, cache_root, out_dir,
                  device) -> List[Dict]:
    from TileEncoderFunc import encoder_config                   # noqa: PLC0415
    names = heads_for(args.heads, 2)
    if not names:
        return []
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
    # mechanism. The 2026-09-16 full run hit exactly this: `--dtype` defaults
    # to 'fp16' and this line used to pass that straight through to
    # `head_cfg`, so the dataclass default fixed nothing -- every construction
    # site overrode it right back to fp16. Hardcoding it here is what makes
    # that impossible to regress on again by way of a CLI default.
    # PER HEAD, not one shared `head_cfg` -- see `_head_cfg_for`'s own
    # docstring: mlp/mlp_deep/mlp_wide/mlp_deep_wide each need their own
    # depth/width, and a single shared config could only ever give them all
    # the same one.
    head_cfgs = {n: _head_cfg_for(n, int(spec.dim), args) for n in names}
    heads = {n: Head(head_cfgs[n], *HEAD_CHOICES[n][:2]).to(device) for n in names}
    print(f'baseline 2  {encoder_name}  dim={spec.dim} kind={spec.kind} '
          f'num_prefix={num_prefix}  (frozen)', flush=True)

    rows, vrows = train_rows(args, cache_root), val_rows(args, cache_root)
    # Fixed for the whole run -- see Datasets.class_weights. `weights=None`
    # under --class-weight none restores plain unweighted cross_entropy.
    weights = class_weights(rows, device) if args.class_weight == 'balanced' else None
    if weights is not None:
        print_class_weights(weights)
    resumed = {n: _maybe_resume(args.resume_dir, encoder_name, True, n, h, encoder)
              for n, h in heads.items()}
    # (best_pooled, best_unweighted) -- a resumed checkpoint only recorded
    # the pooled val accuracy (`_maybe_resume` reads `ckpt['val']
    # ['level_accuracy']`), so best_unweighted always restarts at -1.0 even
    # on resume; see save_tagged's own docstring for what the two track.
    best = {n: (r[1], -1.0) for n, r in resumed.items()}
    opts = {n: torch.optim.Adam(h.parameters(), lr=args.lr)
            for n, h in heads.items()}
    for n, (_, _, opt_state) in resumed.items():
        if opt_state is not None:
            opts[n].load_state_dict(opt_state)
    # Captured BEFORE any warmup call ever overwrites `param_group['lr']` --
    # see `_apply_warmup`'s own docstring for why.
    base_lrs = {n: [g['lr'] for g in opt.param_groups] for n, opt in opts.items()}
    # ONE shared epoch loop below covers every head trained off this
    # encoder, so a single starting point has to serve all of them -- the
    # FURTHEST-ALONG head's own epoch, so a head resumed from an earlier
    # checkpoint gets a few epochs of harmless catch-up rather than rolling
    # the whole loop (and every other head) back to its own earlier point.
    start_epoch = max((r[0] for r in resumed.values()), default=0)
    end_epoch = start_epoch + args.epochs   # "epoch N/end_epoch" reads right
                                            # whether this run started at 0
                                            # or resumed partway in
    raw_of = lambda p: encode_raw(encoder, p, args.encode_batch, device)  # noqa: E731
    out: List[Dict] = []

    # ONE run for this encoder, covering every head trained off it -- see
    # `Runtime.wandb_init`'s docstring for why the boundary is per encoder
    # here and per head in `run_baseline3`.
    run_name = f'{args.run_name}-b2-{encoder_name}' if args.run_name else f'b2-{encoder_name}'
    wb = wandb_init(args.wandb_project, args.wandb_mode, run_name, config=dict(
        baseline=2, encoder=encoder_name, heads=names,
        encoder_dtype=args.dtype, head_dtype='fp32',
        class_weight=args.class_weight,
        lr=args.lr, epochs=args.epochs, n_per_rung=args.n_per_rung,
        tile=args.tile, batch_size=args.batch_size, seed=args.seed,
        train_dataset=args.train_dataset, eval_datasets=args.eval_datasets))

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
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
                cfg=RenderConfig(tile_size=args.tile), epoch_seed=epoch):
            # ONE pass, every head. Moved to the device once here rather than
            # per head: the heads differ in how they reduce it, not in which
            # copy of it they read.
            raw = raw_of(batch['patches'])
            target = batch['labels'].to(device)
            seen_class += torch.bincount(batch['labels'], minlength=NUM_CLASSES)
            native_class += torch.bincount(batch['labels'][batch['native']],
                                           minlength=NUM_CLASSES)
            for name, head in heads.items():
                loss = F.cross_entropy(head(raw, num_prefix, target), target,
                                       weight=weights)
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
        epoch_rows = val_report(val, detail, args.eval_datasets, epoch,
                                end_epoch,
                                losses={n: t[0] / max(t[1], 1)
                                        for n, t in totals.items()},
                                baseline=2, encoder=encoder_name)
        out += epoch_rows
        wandb_log(wb, epoch, wandb_epoch_metrics(epoch_rows))
        for name, head in heads.items():
            best[name] = save_tagged(out_dir, head, encoder, encoder_name, True,
                                     name, head_cfgs[name], args, epoch, val[name],
                                     best[name], opts[name])
    wandb_finish(wb)
    return out


# ══════════════════════════════════════════════════════════════════════════
#  baseline 3: the trunk is fine-tuned, so every head gets its OWN copy
# ══════════════════════════════════════════════════════════════════════════

def run_baseline3(args, cache_root, out_dir, device) -> List[Dict]:
    '''ONE INDEPENDENT RUN PER HEAD, not one loop over several heads sharing a
    trunk. Baseline 2's heads share an encoder pass because the encoder is
    frozen; here each head's gradient moves the trunk a different way, so
    sharing one trunk would make the heads' training interfere. Each gets its
    own encoder -- which means loading the checkpoint once per head, the cost
    of the independence.'''
    from TileEncoderFunc import encoder_config                   # noqa: PLC0415
    names = heads_for(args.heads, 3)
    if not names:
        return []
    rows, vrows = train_rows(args, cache_root), val_rows(args, cache_root)
    # Fixed for the whole run and the SAME across heads (all trained off the
    # same `rows`) -- computed once here rather than per head.
    weights = class_weights(rows, device) if args.class_weight == 'balanced' else None
    if weights is not None:
        print_class_weights(weights)
    out: List[Dict] = []

    for name in names:
        reduction, classifier = HEAD_CHOICES[name][:2]
        # THE TRUNK IS TRAINED, SO IT IS FP32. `ConvNeXtV2EncoderConfig` sets
        # dtype='fp16', which is right for the frozen use it was written for
        # and wrong here for the same reason `HeadConfig.dtype` is fp32: Adam
        # updating fp16 parameters without a GradScaler produces NaN, and the
        # 2026-09-16 smoke run showed exactly that on the (much smaller) head.
        # 28M parameters in fp32 is ~112 MB of weights plus optimizer state --
        # nothing, on this card.
        base = encoder_config(BASELINE3_ENCODER)
        cfg = replace(base, model=replace(base.model, dtype='fp32'))
        encoder = cfg.build(device)
        # ALWAYS fp32, never `args.dtype` -- same reasoning as `run_baseline2`'s
        # `head_cfg`, and the same bug this used to have: passing `args.dtype`
        # through here silently reintroduced fp16-under-Adam the moment
        # `--dtype`'s default (fp16) was left in place, regardless of what
        # `HeadConfig`'s own dataclass default said.
        head_cfg = _head_cfg_for(name, int(encoder.model_spec.dim), args)
        head = Head(head_cfg, reduction, classifier).to(device)
        print(f'baseline 3  head {name} (reduction={reduction}, '
              f'classifier={classifier.__name__}): fine-tuning '
              f'{BASELINE3_ENCODER} dim={head_cfg.in_dim} '
              f'trunk_lr={args.trunk_lr} head_lr={args.lr}', flush=True)
        start_epoch, best_pooled, opt_state = _maybe_resume(
            args.resume_dir, BASELINE3_ENCODER, False, name, head, encoder)
        # (best_pooled, best_unweighted) -- see run_baseline2's identical
        # comment for why best_unweighted always restarts at -1.0.
        best = (best_pooled, -1.0)
        end_epoch = start_epoch + args.epochs

        # Two param groups: a trunk pretrained on natural images and a head
        # initialised at random do not want the same step size -- the standard
        # fine-tuning split, and the reason --trunk-lr defaults below --lr.
        opt = torch.optim.Adam([
            {'params': encoder.model.parameters(), 'lr': args.trunk_lr},
            {'params': head.parameters(), 'lr': args.lr},
        ])
        if opt_state is not None:
            opt.load_state_dict(opt_state)
        # Captured BEFORE any warmup call ever overwrites `param_group['lr']`
        # -- see `_apply_warmup`'s own docstring for why.
        base_lrs = [g['lr'] for g in opt.param_groups]
        raw_of = lambda p: trunk_raw(encoder, p, device)          # noqa: E731

        # ONE run for this head -- it owns its own trunk/optimizer/epoch loop,
        # so it is one experiment on its own (contrast baseline 2's per-
        # ENCODER boundary, where several heads share one epoch axis).
        run_label = f'{args.run_name}-b3-{name}' if args.run_name else f'b3-{name}'
        # No `dtype=args.dtype` here -- baseline 3 has nothing left for it to
        # control (trunk and head are both hardcoded fp32, above), so logging
        # it would imply a knob that does not exist for this baseline.
        wb = wandb_init(args.wandb_project, args.wandb_mode, run_label, config=dict(
            baseline=3, encoder=BASELINE3_ENCODER, head=name, reduction=reduction,
            classifier=classifier.__name__, trunk_dtype='fp32', head_dtype='fp32',
            class_weight=args.class_weight,
            lr=args.lr, trunk_lr=args.trunk_lr, epochs=args.epochs,
            n_per_rung=args.n_per_rung, tile=args.tile, batch_size=args.batch_size,
            seed=args.seed, train_dataset=args.train_dataset,
            eval_datasets=args.eval_datasets))

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
                    cfg=RenderConfig(tile_size=args.tile), epoch_seed=epoch):
                target = batch['labels'].to(device)
                seen_class += torch.bincount(batch['labels'], minlength=NUM_CLASSES)
                native_class += torch.bincount(batch['labels'][batch['native']],
                                               minlength=NUM_CLASSES)
                loss = F.cross_entropy(head(raw_of(batch['patches']), 0, target),
                                       target, weight=weights)
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
            epoch_rows = val_report(val, detail, args.eval_datasets, epoch,
                                    end_epoch,
                                    losses={name: total / max(seen, 1)},
                                    baseline=3, encoder=BASELINE3_ENCODER)
            out += epoch_rows
            wandb_log(wb, epoch, wandb_epoch_metrics(epoch_rows))
            best = save_tagged(out_dir, head, encoder, BASELINE3_ENCODER, False,
                               name, head_cfg, args, epoch, val[name], best, opt)
        wandb_finish(wb)
    return out


# ══════════════════════════════════════════════════════════════════════════

def save_tagged(out_dir, head: Head, encoder, encoder_name: str, frozen: bool,
                head_name: str, head_cfg, args, epoch: int, val: Dict,
                best, optimizer):
    '''Write `_last` every epoch, `_best` when the POOLED val accuracy
    improves, and `_best_unweighted` when the per-dataset-AVERAGED val
    accuracy improves -- see `val_report`'s and `Checkpoints.weight_filename`'s
    own docstrings for why these two can pick different epochs. `best` is
    `(best_pooled, best_unweighted)` in, the same tuple (possibly updated)
    out.

    THREE files rather than one per epoch: with `--epochs 10` and several
    heads, per-epoch checkpoints are tens of GB of things nobody will open,
    and the only ones anyone asks for are "the one training ended on" and
    "the one that scored best" -- now two answers to "scored best", since
    the two weightings do not always agree.

    `optimizer.state_dict()` travels with every file (`_maybe_resume` is the
    reader) so a resumed run continues Adam's own momentum/variance instead
    of restarting them on top of loaded weights -- the standard PyTorch
    resume recipe (model + optimizer + epoch, saved and loaded together).'''
    best_pooled, best_unweighted = best
    wdir = Path(out_dir) / 'weights'
    # `extra`: this package's own label-space facts, which the generic
    # checkpoint format (`aiNNModel/models/common/Checkpoints.py`) has no
    # opinion about -- see `save_checkpoint`'s own docstring.
    common = dict(head=head, encoder=encoder, encoder_name=encoder_name,
                  frozen=frozen, head_name=head_name, head_cfg=head_cfg,
                  epoch=epoch, val=val, run_args=vars(args),
                  optimizer_state=optimizer.state_dict(),
                  # `rungs`: the label space this head classifies into --
                  # `1_estimate_query_mpp/ClassifierEstMpp.py` reads it back
                  # to turn a predicted class into a ds value, so it has to
                  # travel WITH the checkpoint rather than be re-imported from
                  # this package's own `Datasets.RUNGS` at inference time,
                  # which could drift out of sync with what a specific
                  # checkpoint was actually trained against.
                  extra=dict(tile_size=args.tile, rungs=RUNGS))
    save_checkpoint(wdir / weight_filename(encoder_name, frozen, head_name,
                                           'last'), **common)

    acc = val['level_accuracy']
    if acc > best_pooled:
        save_checkpoint(wdir / weight_filename(encoder_name, frozen, head_name,
                                               'best'), **common)
        print(f'      new best, pooled ({acc:.4f}) -> '
              f'{weight_filename(encoder_name, frozen, head_name, "best")}',
              flush=True)
        best_pooled = acc

    acc_u = val['level_accuracy_unweighted']
    if acc_u > best_unweighted:
        save_checkpoint(wdir / weight_filename(encoder_name, frozen, head_name,
                                               'best_unweighted'), **common)
        print(f'      new best, unweighted ({acc_u:.4f}) -> '
              f'{weight_filename(encoder_name, frozen, head_name, "best_unweighted")}',
              flush=True)
        best_unweighted = acc_u

    return best_pooled, best_unweighted


def _merge_val_scores(path: Path, out_rows: List[Dict]) -> List[Dict]:
    '''Under `--merge`: `out_rows`' own `(baseline, encoder, head)` keys
    REPLACE the matching rows already in `path` -- a rerun of one head
    overwrites that head's history, one epoch row at a time, same as a plain
    overwrite would if it were the only head trained. Every OTHER head's
    rows already on disk are kept untouched, and a key `path` has never seen
    is a plain append. `path` not existing yet is just `out_rows` alone --
    the first run has nothing to merge with.

    `csv.DictReader` reads every value as a str (`baseline` is `'2'`/`'3'`,
    not `2`/`3`) -- the key comparison normalizes `out_rows`' own values to
    str on that side rather than parsing the file's ints back, since the
    only thing this needs from either side is string equality, not the
    values themselves.
    '''
    if not path.exists():
        return out_rows
    new_keys = {(str(r['baseline']), r['encoder'], r['head']) for r in out_rows}
    with open(path, newline='') as fh:
        kept = [r for r in csv.DictReader(fh)
               if (r['baseline'], r['encoder'], r['head']) not in new_keys]
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
    ap.add_argument('--val-n-wsi', type=int, default=10,
                    help='WSIs held out per eval dataset for val')
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
    ap.add_argument('--epochs', type=int, default=10)
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
    ap.add_argument('--seed', type=int, default=42,
                    help='sampling AND the val/test WSI split')
    ap.add_argument('--max-wsi', type=int, default=None, help='smoke run')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None)
    ap.add_argument('--resume-dir', default=None,
                    help='directory of existing weight_filename()-named '
                         'checkpoints (e.g. an old --out/weights) to warm-'
                         'start each arm/encoder from before training. '
                         'WEIGHTS ONLY -- save_checkpoint has never stored '
                         'optimizer state, so this restarts Adam\'s own '
                         'momentum/variance from zero on top of the loaded '
                         'weights; see _maybe_resume\'s docstring')
    ap.add_argument('--wandb-project', default='mpp-routing-head')
    # Reads WANDB_MODE, same convention as SuperPathPoint/FewShotEoMT's own
    # --wandb-mode: `wandb.init(mode=...)` takes an EXPLICIT argument, which
    # beats the env var, so the default has to read it rather than hardcode a
    # literal -- a literal here would be the exact bug fixed elsewhere on
    # 2026-09-16 (jobscripts/_env.sh has the full note).
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
                         "attn_linear) no longer wipes out the other 10 "
                         "heads' history. Off by default: a plain overwrite "
                         'is still what a full run wants.')
    args = ap.parse_args()

    device = torch.device(args.device)
    cache_root = Path(_paths.RESULT_DIR) / 'cache' / 'mpp_routing_head'
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
    # NOT "wandb auto-names it" when --run-name is unset -- run_baseline2/3
    # build a real name either way (`b2-<encoder>`, `b3-<head>`, optionally
    # prefixed by --run-name). wandb's own random name generator
    # ("electric-unicorn-42") never fires here.
    prefix = args.run_name or '(none -- still named b2-<encoder> / b3-<head>)'
    print(f'  wandb project={args.wandb_project}   mode={args.wandb_mode}   '
         f'run_name prefix={prefix}')

    out_rows: List[Dict] = []
    if '2' in wanted:
        for encoder_name in args.encoders:
            out_rows += run_baseline2(args, encoder_name, cache_root, out_dir,
                                      device)
    if '3' in wanted:
        out_rows += run_baseline3(args, cache_root, out_dir, device)

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
    print(f'{out_dir / "weights"}  '
          f'(*_last.pt and *_best.pt; cli/evaluate.py scores them on test)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
