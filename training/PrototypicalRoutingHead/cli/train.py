#!/usr/bin/env python3
'''Trains the metric-based meta-learning line: `Pooling` -> Stage 2
(support context G, query context F, Collapse) -> Stage 3 routing head,
episodic, on `ki67_pure` -- optionally with `bracs/train` as a second staining
domain a draw may put on either side (`--cross-domain-dataset`).

    python training/PrototypicalRoutingHead/cli/train.py --encoder uni2 \
        --episode-reuse hold_q --reuse-k auto

EPISODES (`Episodes.py` has the mechanics). Every epoch runs each of the 39
training rung combinations (3-, 4- and 5-of-6 minus `HELD_OUT_COMBOS`), in a
fresh order, from the MIXED pool under the overlap rule. One draw is `ks`
support batches x `kq` query batches and trains one optimizer step per
(support, query) pair:

    --episode-reuse none     1 x 1 per draw, 39 x K draws an epoch
    --episode-reuse hold_s   1 x K per draw, 39 draws an epoch
    --episode-reuse hold_q   K x 1 per draw, 39 draws an epoch

so all three take 39 x K optimizer steps an epoch. `--reuse-k auto` is the
largest K the pool can actually supply for BOTH held shapes on every
combination (`Episodes.max_feasible_k`); an integer is used as given and
refused if it cannot be drawn. `--episodes-per-epoch auto` is the count
above. Both are printed at start-up as "given -> used" so a log says which
was the default and which was chosen.

The frozen encoder runs ONCE per batch; the pairs reuse its tokens and only
the trainable part (pooling, G, F, Collapse, head) runs per pair.

VALIDATION, K x K (user, 2026-09-24). For each eval dataset and each held-out
combination: K support batches and K query batches, every one of the K x K
pairs scored. A rung's accuracy in a combination is its accuracy averaged over
the pairs; a combination's accuracy is the mean of its rungs'; a dataset's is
the mean of its combinations'; the TOTAL is the mean of the datasets'. Every
epoch prints each combination's accuracy and its per-rung accuracies.
`--val-reuse-k auto` is the largest K every dataset can supply for every
held-out combination. The val draw uses a fresh `random.Random(--seed)` every
epoch, so the same questions are asked every epoch and only the model changes.

CHECKPOINTS: `_best.pt` on the TOTAL, `_6rung.pt` on the full 6-way
combination's accuracy (mean over datasets), `_native.pt` on the mean over the
combinations whose every query rendered natively.

RESUME (`aiNNModel/models/common/Resume.py`): with `--resume-dir`, the run's
full state is written there every epoch and picked up from on the next start;
without it nothing is written and training starts from scratch. `--epochs` is
the total, so a finished run resumes into nothing.

`--supply-only` builds the manifests (filling the shared caches), prints the
per-rung supply and the K / episode numbers `auto` resolves to, and exits.
'''
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', 'utilities'))
from _paths import setup_import_paths, job_result_dir              # noqa: E402
setup_import_paths()

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

from training.MppRoutingHead.Datasets import (                      # noqa: E402
    add_cache_args, build_manifest, CameraBank, open_caches, RenderConfig,
    RUNGS)
from WsiSplit import (native_bracs_rung_wsi_names, read_split,       # noqa: E402
                      split_path)
from training.PrototypicalRoutingHead.Episodes import (              # noqa: E402
    HELD_OUT_COMBOS, REUSE_MODES, batch_shape, draw, epoch_schedule,
    episodes_per_epoch,
    group_by_wsi_name, max_feasible_k, pool_by_rung, render_draw, reuse_pairs,
    training_combos, training_pools)
from WsiSplit import BRACS_RUNGS                                       # noqa: E402
from training.PrototypicalRoutingHead.Losses import compute_loss     # noqa: E402
from Pooling import Pooling, PoolingConfig                            # noqa: E402
from training.PrototypicalRoutingHead.Runtime import (               # noqa: E402
    COLLAPSE_CHOICES, ROUTING_HEAD_CHOICES, SUPPORT_CONTEXT_CHOICES,
    QUERY_CONTEXT_CHOICES, wandb_init, wandb_log, wandb_finish, rescore)
from Checkpoints import save_prototype_checkpoint                    # noqa: E402
from Features import encode_raw                                      # noqa: E402
from Resume import ResumeFile, resume_identity                       # noqa: E402


def build_encoder(name: str, dtype: str, device):
    '''Same pattern `MppRoutingHead/cli/train.py`'s `run_baseline2` uses --
    `dtype` lives on the nested `ModelConfig`, so it has to go in as a
    `dataclasses.replace` on the CONFIG, not a kwarg `encoder_config` itself
    would raise on.'''
    from dataclasses import replace                                 # noqa: PLC0415
    from TileEncoderFunc import encoder_config                       # noqa: PLC0415
    base_cfg = encoder_config(name)
    cfg = replace(base_cfg, model=replace(base_cfg.model, dtype=dtype))
    return cfg.build(device)


def encode_group(group, encoder, batch_size: int, device):
    """One rendered batch `{rung: [(patch, native), ...]}` through the FROZEN
    encoder, once: `{rung: raw tokens [n, L, D]}`, plus `native` per rung.
    Every pair a batch takes part in reuses these tokens -- only what trains
    runs again per pair."""
    raw, native = {}, {}
    for rung, items in group.items():
        stacked = torch.from_numpy(np.stack([p for p, _n in items]))
        raw[rung] = encode_raw(encoder, stacked, batch_size, device)
        native[rung] = [bool(n) for _p, n in items]
    return raw, native


def pair_forward(rungs, support_raw, query_raw, query_native, *, num_prefix: int,
                 pooling, support_context, query_context, collapse, head, device):
    """One (support batch, query batch) pair -> `(logits, target, native)`.
    `target` is the LOCAL index into `rungs` (0..N-1).

    Stage 2 is `pool -> G -> {F, Collapse}`: G acts on one rung's support at a
    time (support of different rungs never attend to each other); F and
    Collapse are PARALLEL consumers of G's output, so F always reads the
    un-collapsed support whatever `--collapse` is. `collapse.COLLAPSES` says
    whether Collapse returned one vector per rung (stacked `[K, D]` for the
    head) or left the members (`{rung: [K_r, D]}`)."""
    collapses = getattr(collapse, 'COLLAPSES', True)
    g_output_by_rung, support_by_rung, prototypes = {}, {}, []
    for rung in rungs:
        ctx = support_context(pooling(support_raw[rung], num_prefix))
        g_output_by_rung[rung] = ctx
        out = collapse(ctx)
        if collapses:
            prototypes.append(out)
        else:
            support_by_rung[rung] = out
    prototypes = torch.stack(prototypes) if collapses else support_by_rung

    q_raw, target, native = [], [], []
    for local_idx, rung in enumerate(rungs):
        q_raw.append(query_raw[rung])
        target += [local_idx] * len(query_raw[rung])
        native += query_native[rung]
    query_vecs = query_context(pooling(torch.cat(q_raw), num_prefix),
                               g_output_by_rung)
    logits = head(query_vecs, prototypes)
    return (logits, torch.tensor(target, dtype=torch.long, device=device),
            torch.tensor(native, dtype=torch.bool))


def episode_forward(rendered, *, encoder, num_prefix: int, pooling,
                    support_context, query_context, collapse, head,
                    batch_size: int, device):
    """The original one-support-one-query episode (`Episodes.RenderedEpisode`)
    -> `(logits, target, native)`. Kept for `cli/evaluate.py`'s original test
    scoring; it is `encode_group` + `pair_forward` on the one pair."""
    s_raw, _s_native = encode_group(rendered.support, encoder, batch_size, device)
    q_raw, q_native = encode_group(rendered.query, encoder, batch_size, device)
    return pair_forward(rendered.rungs, s_raw, q_raw, q_native,
                        num_prefix=num_prefix, pooling=pooling,
                        support_context=support_context,
                        query_context=query_context, collapse=collapse,
                        head=head, device=device)


# ══════════════════════════════════════════════════════════════════════════
#  held-out validation -- held-out WSI (same split MppRoutingHead/cli/
#  train.py reads) x held-out rung combination (Episodes.HELD_OUT_COMBOS)
# ══════════════════════════════════════════════════════════════════════════

def val_manifests(args, caches, out_dir: Path):
    '''`{dataset_id: {rung: [row, ...]}}` -- the MIXED pool of every
    `--eval-datasets` entry's held-out WSIs.

    The held-out names are READ from the split `utilities/cli/build_cache/
    make_split.py` wrote (`--split-cache-job`, default MakeSplit) --
    the same file `MppRoutingHead/cli/train.py`'s `val_rows` reads, so both
    packages hold out the IDENTICAL slides by construction rather than by two
    derivations happening to agree. The BRACS native filter is applied there,
    by the one writer.
    '''
    out = {}
    for dataset_id in args.eval_datasets:
        val_names, test_names = read_split(
            split_path(caches.split_job, dataset_id))
        part = build_manifest(
            dataset_id, masks=caches.masks, sampler_root=caches.sampler_root,
            report_dir=(out_dir / 'sampler_reports'
                        / f'{dataset_id.replace("/", "_")}_val'),
            tile_size=args.tile, n_per_rung=args.val_n_per_rung,
            seed=args.seed, wsi_names=val_names)
        print(f'[val]   {dataset_id}: {len(part)} positions from '
             f'{len(val_names)} WSIs ({len(test_names)} held back for test)',
             flush=True)
        # Manifest SUPPLY: what every val WSI together offers per rung, before
        # any draw. The coarsest rung here is what bounds --val-reuse-k.
        by_rung = Counter(r.rung for r in part)
        print(f'          rung  {"  ".join(f"{r:>7g}" for r in RUNGS)}\n'
             f'          n     {"  ".join(f"{by_rung.get(r, 0):>7d}" for r in RUNGS)}',
             flush=True)
        out[dataset_id] = pool_by_rung(part)
    return out


def _combo_label(rungs) -> str:
    '''`(1.0, 4.0, 16.0, 32.0)` -> `'1+4+16+32'` -- a compact, CSV-safe,
    human-readable label for which `HELD_OUT_COMBOS` entry (or, in
    principle, any rung tuple) an episode drew. Sorted ascending already
    (every `Episode.rungs` is, `Episodes._rung_subsets`' own docstring),
    not re-sorted here.
    '''
    return '+'.join(f'{r:g}' for r in rungs)


def val_episode_detail(rendered, logits: torch.Tensor, target: torch.Tensor,
                       native: torch.Tensor, dataset_id: str):
    '''One val episode's query predictions -> detail rows `rescore`/
    `combo_report` can read. GLOBAL rung/class (via
    `RUNGS.index`), not the episode's own LOCAL N-way index -- `rescore`'s
    own `score()` looks up `RUNGS[pred_class]`/`RUNGS[true_class]`
    directly, so a local index from a 3-way episode would silently name
    the wrong rung. `combo` (2026-09-21) records WHICH `HELD_OUT_COMBOS`
    entry this episode drew -- see `combo_report`'s own docstring for why
    this has to be tracked per row rather than assumed from the rung alone
    (one rung appears in more than one combo).
    '''
    pred = logits.detach().argmax(-1)
    combo = _combo_label(rendered.rungs)
    out = []
    for local_true, local_pred, nat in zip(target.tolist(), pred.tolist(),
                                           native.tolist()):
        gt_rung, pred_rung = rendered.rungs[local_true], rendered.rungs[local_pred]
        out.append(dict(dataset=dataset_id, combo=combo, native=bool(nat),
                        gt_class=RUNGS.index(gt_rung), gt_rung=gt_rung,
                        pred_class=RUNGS.index(pred_rung), pred_rung=pred_rung,
                        correct=local_true == local_pred))
    return out


def combo_report(detail_by_dataset, epoch: int, **identity):
    '''Per-COMBO (3-way / 4-way / 6-way) val breakdown -- for EACH combo,
    its own overall accuracy (with native/resampled) AND its own per-rung
    breakdown, BOTH computed within that combo's own query examples only.
    Returns `(combo_rows, rung_rows)`: `combo_rows` for `val_scores_
    per_combo.csv` (one row per dataset+combo), `rung_rows` for
    `val_scores_per_rung.csv` (one row per dataset+combo+rung).

    THE ONLY val report (2026-09-21) -- there used to be a separate
    `val_report` producing an "all, n-weighted"/"all, dataset-avg" summary
    ABOVE the per-dataset numbers, and a `rung_report` pooling every combo
    together per rung. Both were REMOVED: a summary that blends 3-way,
    4-way and the full 6-way DEPLOYMENT task into one number answers no
    single question cleanly (a rung scored alongside 2 other candidates
    and the same rung scored alongside 5 are different questions, and the
    SAME is true one level up -- "accuracy" blended across combos is not
    one thing either). Every number this file now reports is scoped to
    ONE combo, which is a question with an actual, specific answer.
    Used by `cli/evaluate.py`'s ORIGINAL test table (random full 6-way
    episodes, pooled per combo). Training's own val is `kxk_report`.

    The per-rung breakdown here is NOT `rescore_by_rung` (which always
    iterates the full 6-rung `RUNGS` ladder, filling in NaN for rungs this
    combo cannot contain at all -- a 3-way combo has exactly 3 rungs, so
    the other 3 NaN rows would be noise, not information): computed
    directly over the rungs actually present in THIS combo's own rows.

    Per-combo's OWN overall accuracy is POOLED within the combo, not
    rung-averaged: a combo's query examples are already BALANCED across
    its own rungs by construction (`--n-query` is the same for every rung
    an episode drew), so there is no tile-count skew here for rung-
    averaging to correct -- that skew came from `RICHNESS`'s coarse-rung
    TRAINING shortfall, which has no equivalent at the per-episode level.
    '''
    combo_rows, rung_rows = [], []
    for dataset_id, rows in detail_by_dataset.items():
        combos = sorted({r['combo'] for r in rows},
                        key=lambda c: (c.count('+'), c))
        for combo in combos:
            combo_detail = [r for r in rows if r['combo'] == combo]
            result = rescore(combo_detail)
            print(f'         {dataset_id:17s} combo {combo:20s} '
                 f'acc {result["level_accuracy"]:.4f}  '
                 f'native {result["level_accuracy_native"]:.4f} '
                 f'(n={result["n_native"]})  '
                 f'resampled {result["level_accuracy_resampled"]:.4f} '
                 f'(n={result["n_resampled"]})', flush=True)
            combo_rows.append(dict(**identity, epoch=epoch, val_dataset=dataset_id,
                                   combo=combo, **result))

            combo_rungs = sorted({r['gt_rung'] for r in combo_detail})
            per_rung = {rung: rescore([r for r in combo_detail if r['gt_rung'] == rung])
                       for rung in combo_rungs}
            print(f'         {"":17s}       rung   '
                 + '  '.join(f'{r:>7g}' for r in combo_rungs), flush=True)
            print(f'         {"":17s}       acc    '
                 + '  '.join(f'{per_rung[r]["level_accuracy"]:>7.4f}'
                             for r in combo_rungs), flush=True)
            for rung, r in per_rung.items():
                rung_rows.append(dict(**identity, epoch=epoch, val_dataset=dataset_id,
                                      combo=combo, rung=rung, **r))
    return combo_rows, rung_rows


# ══════════════════════════════════════════════════════════════════════════
#  K x K scoring -- val every epoch, and cli/evaluate.py's second test table
# ══════════════════════════════════════════════════════════════════════════

def draw_until(sp, qp, rng, rungs, *, tries: int, **kw):
    """`Episodes.draw`, retried up to `tries` times with the same `rng` (so a
    fixed seed gives a fixed sequence of attempts). None if every try failed."""
    for _ in range(tries):
        d = draw(sp, qp, rng, rungs, **kw)
        if d is not None:
            return d
    return None


def score_kxk(pools_by_dataset, combos, *, k: int, n_support: int, n_query: int,
              max_overlap: float, seed: int, tries: int, bank, render_cfg,
              support_native: bool, encoder, num_prefix: int, pooling,
              support_context, query_context, collapse, head, batch_size: int,
              device) -> Dict[str, Dict[str, Dict]]:
    """`{dataset: {combo label: result}}` for every dataset and combination:
    K support batches x K query batches drawn once (fixed `seed`, so the same
    questions every call), every pair scored.

    `result`: `acc_by_pair_rung` -- `[K*K][rung]` accuracy of that rung's
    queries in that pair -- plus `n_resampled` (queries rendered off a finer
    level) and `ok` (False when the draw or render failed `tries` times, which
    `max_feasible_k` is there to make impossible, so it is printed loudly)."""
    out: Dict[str, Dict[str, Dict]] = {}
    for dataset_id, pool in pools_by_dataset.items():
        rng = random.Random(seed)
        per_combo: Dict[str, Dict] = {}
        for combo in combos:
            label = _combo_label(combo)
            # rendered = None
            # for _ in range(tries):
            #     d = draw_until(pool, pool, rng, combo, tries=tries,
            #                    n_support=n_support, n_query=n_query, ks=k, kq=k,
            #                    max_overlap=max_overlap)
            #     if d is None:
            #         break
            #     rendered = render_draw(d, bank, render_cfg, deterministic=True,
            #                            support_native=support_native)
            #     if rendered is not None:
            #         break
            # if rendered is None:
            #     print(f'    [kxk] {dataset_id} {label}: could not draw/render '
            #           f'K={k} in {tries} tries -- scored as missing', flush=True)
            #     per_combo[label] = dict(ok=False, rungs=combo,
            #                             acc_by_pair_rung=[], n_resampled=0)
            #     continue
            # ////////////////////////////////////////////////////////////////// add by me
            rendered = None
            n_drawn = 0
            n_render_failed = 0
            draw_exhausted = False

            for _ in range(tries):
                d = draw_until(
                    pool, pool, rng, combo, tries=tries,
                    n_support=n_support, n_query=n_query,
                    ks=k, kq=k, max_overlap=max_overlap,
                )

                if d is None:
                    draw_exhausted = True
                    break

                n_drawn += 1
                rendered = render_draw(
                    d, bank, render_cfg, deterministic=True,
                    support_native=support_native,
                )

                if rendered is not None:
                    break

                n_render_failed += 1

            if rendered is None:
                if n_drawn == 0:
                    print(
                        f'    [kxk:draw-failed] {dataset_id} {label}: '
                        f'no valid K={k} draw in {tries} attempts '
                        f'-- scored as missing',
                        flush=True,
                    )
                elif draw_exhausted:
                    print(
                        f'    [kxk:mixed-failed] {dataset_id} {label}: '
                        f'{n_render_failed} valid draw(s) failed rendering; '
                        f'then no valid K={k} draw in the next {tries} attempts '
                        f'-- scored as missing',
                        flush=True,
                    )
                else:
                    print(
                        f'    [kxk:render-failed] {dataset_id} {label}: '
                        f'all {n_render_failed} valid K={k} draw(s) contained '
                        f'at least one unrenderable tile '
                        f'-- scored as missing',
                        flush=True,
                    )

                per_combo[label] = dict(
                    ok=False,
                    rungs=combo,
                    acc_by_pair_rung=[],
                    n_resampled=0,
                )
                continue
            # ////////////////////////////////////////////////////////////////// end add by me
            s_enc = [encode_group(g, encoder, batch_size, device) for g in rendered.supports]
            q_enc = [encode_group(g, encoder, batch_size, device) for g in rendered.queries]
            accs, n_resampled = [], 0
            for si, qj in reuse_pairs(k, k):
                with torch.no_grad():
                    logits, target, native = pair_forward(
                        rendered.rungs, s_enc[si][0], q_enc[qj][0], q_enc[qj][1],
                        num_prefix=num_prefix, pooling=pooling,
                        support_context=support_context,
                        query_context=query_context, collapse=collapse,
                        head=head, device=device)
                correct = (logits.argmax(-1) == target).cpu()
                target_c = target.cpu()
                accs.append({rung: float(correct[target_c == i].double().mean())
                             for i, rung in enumerate(rendered.rungs)})
                n_resampled += int((~native).sum())
            per_combo[label] = dict(ok=True, rungs=rendered.rungs,
                                    acc_by_pair_rung=accs,
                                    n_resampled=n_resampled)
        out[dataset_id] = per_combo
    return out


def _nanmean(values) -> float:
    vals = [v for v in values if v == v]
    return sum(vals) / len(vals) if vals else float('nan')


def kxk_report(results, epoch: int, *, k: int, **identity):
    """The user's definition (2026-09-24), in order:

        rung in a combo   mean over the K x K pairs of that rung's accuracy
        combo             mean of its rungs
        dataset           mean of its combos
        total             mean of the datasets

    Prints every combination's accuracy and its per-rung row, then each
    dataset's and the total. Returns `(combo_rows, rung_rows, summary)`;
    `summary` has `total`, `by_dataset`, `six_rung` (the full 6-way combo,
    mean over datasets) and `native` (mean over combos with no resampled
    query)."""
    combo_rows, rung_rows = [], []
    by_dataset: Dict[str, float] = {}
    six_label = _combo_label(RUNGS)
    six, native_accs = [], []
    for dataset_id, per_combo in results.items():
        combo_accs = []
        for label, res in per_combo.items():
            rungs = res['rungs']
            per_rung = {r: _nanmean(p[r] for p in res['acc_by_pair_rung'])
                        if res['ok'] else float('nan') for r in rungs}
            acc = _nanmean(per_rung.values())
            combo_accs.append(acc)
            if label == six_label:
                six.append(acc)
            if res['ok'] and res['n_resampled'] == 0:
                native_accs.append(acc)
            print(f'         {dataset_id:17s} combo {label:20s} acc {acc:.4f}   '
                  f'({k}x{k} pairs, {res["n_resampled"]} resampled queries)',
                  flush=True)
            print(f'         {"":17s}       rung   '
                  + '  '.join(f'{r:>7g}' for r in rungs), flush=True)
            print(f'         {"":17s}       acc    '
                  + '  '.join(f'{per_rung[r]:>7.4f}' for r in rungs), flush=True)
            combo_rows.append(dict(**identity, epoch=epoch, val_dataset=dataset_id,
                                   combo=label, k=k, n_pairs=len(res['acc_by_pair_rung']),
                                   n_resampled=res['n_resampled'], level_accuracy=acc))
            for r in rungs:
                rung_rows.append(dict(**identity, epoch=epoch, val_dataset=dataset_id,
                                      combo=label, k=k, rung=r,
                                      level_accuracy=per_rung[r]))
        by_dataset[dataset_id] = _nanmean(combo_accs)
        print(f'         {dataset_id:17s} dataset acc {by_dataset[dataset_id]:.4f}',
              flush=True)
    total = _nanmean(by_dataset.values())
    print(f'         total acc {total:.4f}   (mean of '
          f'{len(by_dataset)} dataset(s))', flush=True)
    return combo_rows, rung_rows, dict(total=total, by_dataset=by_dataset,
                                       six_rung=_nanmean(six),
                                       native=_nanmean(native_accs))


def _cross_domain_tag(cross_domain_dataset: str) -> str:
    '''`args.cross_domain_dataset`, filename/CSV-safe -- `'bracs/train'` ->
    `'bracs_train'`, `''` (disabled, `--cross-domain-dataset ""`) ->
    `'none'` so an empty CSV cell never silently compares equal to a
    genuinely missing column the way `''` would.'''
    return cross_domain_dataset.replace('/', '_') if cross_domain_dataset else 'none'


def _prototype_weight_filename(encoder_name: str, pooling: str,
                               support_context: str, query_context: str,
                               collapse: str, routing_head: str,
                               cross_domain_dataset: str, loss: str,
                               tag: str, episode_reuse: str = 'none',
                               reuse_k: int = 1) -> str:
    '''`<encoder>_frozen_<pooling>_<support_context>_<query_context>_
    <collapse>_<routing_head>_<cross_domain_dataset>_<loss>_<tag>.pt` (the
    `<loss>` segment omitted when `loss` is `'bal'`) -- the prototype-
    checkpoint analogue of `Checkpoints.weight_filename`. Not that function
    itself: it is shaped for ONE `Head` (`<encoder>_<frozen|finetuned>_
    <head>_<tag>.pt`), and this checkpoint holds FIVE modules (pooling/
    support_context/query_context/collapse/head) plus axes (`pooling`,
    `cross_domain_dataset`, `loss`) that function's own world has no name
    for. `support_context`/`query_context` joined 2026-09-22, same day
    `ContextEncoders.py` was -- `identity` (both, by default) trains
    nothing, but `bilstm`/`attnlstm` do, and a run with one of those
    enabled vs one without is a different experiment even when every other
    flag matches; before this they would have silently shared one
    `_best.pt`, same class of gap `cross_domain_dataset` was added here to
    close a day earlier.

    `loss` (2026-09-22, same day `Checkpoints.weight_filename` gained the
    identical segment for `MppRoutingHead`): `bal` gets NO segment (every
    checkpoint before `--loss` existed, and every `bal` run since, keeps
    its exact filename), `ord_a`/`ord_b` get a real, distinct one -- two
    runs differing ONLY in `--loss` used to silently share one `_best.pt`
    the moment they matched on every other axis, exactly the bug this
    file's own `_IDENTITY_FIELDS`/`combo_report` gap for `loss` was closed
    for a day earlier on the CSV side; the checkpoint FILE itself had the
    same gap and is closed here.

    `<episode_reuse>-k<K>` (2026-09-24): the three reuse modes and their K
    are three experiments over the same modules and would otherwise share one
    `_best.pt`. The RESOLVED K, never `auto`: two `auto` runs over different
    supplies are different experiments.
    '''
    tag_cd = _cross_domain_tag(cross_domain_dataset)
    loss_seg = '' if loss == 'bal' else f'{loss}_'
    # The mask recipe is recorded in the checkpoint's args, not named here --
    # `Checkpoints.weight_filename`'s rule.
    return (f'{encoder_name}_frozen_{pooling}_{support_context}_{query_context}_'
           f'{collapse}_{routing_head}_{tag_cd}_{episode_reuse}-k{int(reuse_k)}_'
           f'{loss_seg}{tag}.pt')


#: `_merge_val_scores`' merge key -- serves the role `MppRoutingHead/
#: cli/train.py`'s own `(baseline, encoder, head)` serves for ITS
#: `_merge_val_scores` (that codebase's own term is "head"/"baseline",
#: never "arm" -- MppRoutingHead deliberately renamed away from "arm" to
#: those, 2026-09-17, its own module docstring). `training_framework`
#: (renamed from `arm` 2026-09-21) identifies WHICH of spec.md's Stage 4
#: three-way comparison produced a row -- `'metric_based'` here, a CONSTANT
#: (this file only ever trains that one regime). `cli/train_baseline.py`
#: (`'baseline_prototype'`, Stage 4's own Baseline arm) existed 2026-09-20
#: through 2026-09-22 and was deleted, its own inference-time positioning
#: never having settled -- kept as a field here anyway, since whatever
#: redesign replaces it should be able to write rows into the SAME
#: `val_scores_per_combo.csv`/`val_scores_per_rung.csv` for a Stage-4
#: comparison without a fieldname mismatch. The
#: rest of the tuple (`--collapse`/`--routing-head`/`--pooling`/`--encoder`/
#: `--train-dataset`/`--loss`/`--support-native`, `train_dataset`/`loss`
#: added 2026-09-22 -- previously absent from both this key AND `combo_
#: report`'s own call, which meant two runs differing ONLY in `--loss` or
#: `--train-dataset` would silently share one row instead of accumulating
#: side by side, the same class of gap `support_context`/`query_context`
#: closed a day earlier; `support_native` joined the same day it was
#: added -- it changes what pixels the model was actually trained to
#: compare, exactly like a `--loss`/`--train-dataset` difference does, so
#: it gets the same treatment) is fixed for a whole run too (spec.md's own
#: "which arm is the main line, one at a time" discipline), so the whole
#: tuple is constant across every row a single run produces -- the KEY
#: still has to be here, though, because a SECOND run with different
#: values is exactly what `--merge` exists to not overwrite.
_IDENTITY_FIELDS = ('training_framework', 'encoder', 'pooling',
                    'support_context', 'query_context', 'collapse',
                    'routing_head', 'cross_domain_dataset',
                    'train_dataset', 'loss', 'support_native',
                    'episode_reuse', 'reuse_k')


def _merge_val_scores(path: Path, out_rows):
    '''Under `--merge`: `out_rows`' own `_IDENTITY_FIELDS` key REPLACES the
    matching rows already in `path` -- a rerun of one arm overwrites that
    arm's history, every OTHER arm's rows already on disk are kept. Same
    shape as `MppRoutingHead/cli/train.py`'s own `_merge_val_scores` --
    ALL of `PrototypicalRoutingHead`'s runs share one job name (`--out`
    defaults to `job_result_dir('PrototypicalRoutingHead')`, always the
    same directory), so training a second arm (a different `--collapse`,
    say) without `--merge` would silently truncate the first arm's rows
    away instead of accumulating next to them -- exactly the shape that
    lets `bench_stage1_mpp.py`-style tooling compare every
    arm the same way it already compares `MppRoutingHead`'s several heads.

    `.get(k, '')`, not `r[k]` (2026-09-22, fixing a real crash): a row
    written by an OLDER run, before `_IDENTITY_FIELDS` grew its most
    recent field, has no column for it at all -- `csv.DictReader` then has
    no key to raise on. Missing means "compares as `''`" for this key,
    which is only ever used to tell two IDENTITY tuples apart; it does not
    need to reconstruct that old row's true value, unlike `MppRoutingHead/
    cli/train.py`'s own `.get('loss_kind', 'bal')`, which restores the
    exact value every pre-`--loss` row actually had. This file has no
    such backward-compatibility need -- delete the old CSV and rerun.
    '''
    if not path.exists():
        return out_rows
    new_keys = {tuple(str(r.get(k, '')) for k in _IDENTITY_FIELDS)
               for r in out_rows}
    with open(path, newline='') as fh:
        kept = [r for r in csv.DictReader(fh)
               if tuple(r.get(k, '') for k in _IDENTITY_FIELDS) not in new_keys]
    return kept + out_rows


def save_tagged(weights_dir: Path, modules, cfgs, encoder, args, in_dim: int,
                epoch: int, summary: Dict, best: Dict, run: Dict) -> Dict:
    """Three files, three criteria, because they do not always pick the same
    epoch:

        _best.pt    the val TOTAL (mean of datasets of combos of rungs)
        _6rung.pt   the full 6-way combination, mean over datasets -- the
                    deployment task, never rehearsed in training
        _native.pt  the mean over combinations whose every query rendered
                    natively (no resampled pixels)

    `run` carries the resolved episode settings (`episode_reuse`, `reuse_k`,
    `val_reuse_k`, `episodes_per_epoch`), recorded in `extra` because
    `run_args` holds what was TYPED, `auto` included. Returns the updated
    `best`."""
    pooling, support_context, query_context, collapse, head = modules
    support_context_cfg, query_context_cfg, collapse_cfg = cfgs
    common = dict(pooling=pooling, support_context=support_context,
                  query_context=query_context, collapse=collapse, head=head,
                  encoder=encoder, encoder_name=args.encoder, frozen=True,
                  pooling_cfg=PoolingConfig(kind=args.pooling, in_dim=in_dim),
                  support_context_cfg=support_context_cfg,
                  query_context_cfg=query_context_cfg,
                  collapse_cfg=collapse_cfg,
                  support_context_name=args.support_context,
                  query_context_name=args.query_context,
                  collapse_name=args.collapse,
                  routing_head_name=args.routing_head, epoch=epoch,
                  val=dict(total_accuracy=summary['total'],
                           six_rung_accuracy=summary['six_rung'],
                           native_accuracy=summary['native'],
                           by_dataset=summary['by_dataset']),
                  run_args=vars(args),
                  extra=dict(cross_domain_dataset=args.cross_domain_dataset, **run))

    def fname(tag: str) -> str:
        return _prototype_weight_filename(
            args.encoder, args.pooling, args.support_context, args.query_context,
            args.collapse, args.routing_head, args.cross_domain_dataset,
            args.loss, tag, run['episode_reuse'], run['reuse_k'])

    best = dict(best)
    for tag, key in (('best', 'total'), ('6rung', 'six_rung'), ('native', 'native')):
        value = summary[key]
        if value == value and value > best[tag]:
            save_prototype_checkpoint(weights_dir / fname(tag), **common)
            print(f'      new best, {tag} ({key} {value:.4f}) -> {fname(tag)}',
                  flush=True)
            best[tag] = value
    return best


def _auto_int(value: str):
    """argparse type for `auto` or a positive integer."""
    if value == 'auto':
        return 'auto'
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f'{value!r}: must be auto or >= 1')
    return n


def _supply_line(label: str, pool) -> str:
    return (f'  {label:28s}' + '  '.join(f'{len(pool.get(float(r), ())):>7d}'
                                         for r in RUNGS))


#: `resume_identity` leaves these out: they change where output goes, how long
#: the run is in total, or how it is logged -- not what is being trained.
_NOT_IDENTITY = ('epochs', 'out', 'device', 'resume_dir', 'wandb_project',
                 'wandb_mode', 'run_name', 'merge', 'encode_batch', 'supply_only',
                 'mask_cache_job', 'sampler_cache_job', 'split_cache_job',
                 'feasibility_tries')


def resolve_episodes(args, train_pool, cross_pool, val_pools):
    """`auto` / given -> used, for K, val K and draws per epoch. Returns
    `(run, combos)`; `run` holds the resolved values. An integer that the
    pool cannot supply is refused rather than quietly shrunk."""
    combos = training_combos(args.n_choices)

    def train_pairs(combo):
        if cross_pool is not None and set(combo) <= BRACS_RUNGS:
            return [(cross_pool, train_pool), (train_pool, cross_pool)]
        return [(train_pool, train_pool)]

    held_shapes = [lambda k: batch_shape('hold_s', k), lambda k: batch_shape('hold_q', k)]
    common = dict(n_support=args.n_support, n_query=args.n_query,
                  max_overlap=args.max_overlap, tries=args.feasibility_tries,
                  seed=args.seed)
    k_max = max_feasible_k(train_pairs, combos, held_shapes, **common)
    if k_max < 1:
        raise SystemExit(
            f'not even K=1 can be drawn for every training combination with '
            f'--n-support {args.n_support} --n-query {args.n_query}: the '
            f'coarsest rung is short. Lower them, or raise --n-per-rung')
    val_max = {}
    for dataset_id, pool in val_pools.items():
        val_max[dataset_id] = max_feasible_k(
            lambda combo, _p=pool: [(_p, _p)], HELD_OUT_COMBOS,
            [lambda k: (k, k)], **common)
    val_k_max = min(val_max.values()) if val_max else 0
    if val_k_max < 1:
        raise SystemExit(
            f'val cannot draw even 1 x 1 for every held-out combination: '
            f'{val_max}. Lower --n-support/--n-query or raise --val-n-per-rung')

    k = k_max if args.reuse_k == 'auto' else args.reuse_k
    if k > k_max:
        raise SystemExit(f'--reuse-k {k}: the pool supplies at most K={k_max} '
                         f'for hold_s/hold_q on every combination')
    val_k = val_k_max if args.val_reuse_k == 'auto' else args.val_reuse_k
    if val_k > val_k_max:
        raise SystemExit(f'--val-reuse-k {val_k}: the val pools supply at most '
                         f'K={val_k_max} ({val_max})')
    n_draws = (episodes_per_epoch(args.episode_reuse, k, len(combos))
               if args.episodes_per_epoch == 'auto' else args.episodes_per_epoch)
    ks, kq = batch_shape(args.episode_reuse, k)
    run = dict(episode_reuse=args.episode_reuse, reuse_k=int(k),
               val_reuse_k=int(val_k), episodes_per_epoch=int(n_draws),
               k_max=int(k_max), val_k_max=int(val_k_max))

    print('\nepisode settings  (given -> used)', flush=True)
    print(f'  --episode-reuse       {args.episode_reuse}   '
          f'({ks} support x {kq} query batches per draw)', flush=True)
    print(f'  --reuse-k             {args.reuse_k} -> K = {k}   '
          f'(max drawable for hold_s/hold_q: {k_max})', flush=True)
    print(f'  --val-reuse-k         {args.val_reuse_k} -> val K = {val_k}   '
          f'(max drawable: {val_k_max}; per dataset {val_max})', flush=True)
    print(f'  --episodes-per-epoch  {args.episodes_per_epoch} -> {n_draws} draws '
          f'x {ks * kq} pair(s) = {n_draws * ks * kq} optimizer steps per epoch   '
          f'({len(combos)} training combinations)', flush=True)
    return run, combos


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train-dataset', default='ki67_pure')
    ap.add_argument('--cross-domain-dataset', default='bracs/train',
                    help="a SECOND staining domain: a draw whose rung "
                         "combination fits Episodes.BRACS_RUNGS puts it on "
                         "one side, --train-dataset on the other. Empty "
                         "string disables it")
    ap.add_argument('--encoder', default='uni2',
                    choices=('gigapath', 'uni2', 'conch_vit'))
    ap.add_argument('--dtype', choices=('fp16', 'fp32'), default='fp16',
                    help='frozen encoder inference precision; everything '
                         'that trains is fp32')
    ap.add_argument('--pooling',
                    choices=('cls', 'avg', 'max', 'attn', 'passthrough'),
                    default='cls',
                    help="Stage 2's shared support/query pooling. "
                         "'passthrough' is refused: Collapse needs one vector "
                         "per tile")
    ap.add_argument('--support-context', choices=sorted(SUPPORT_CONTEXT_CHOICES),
                    default='identity', help='Stage 2, G')
    ap.add_argument('--query-context', choices=sorted(QUERY_CONTEXT_CHOICES),
                    default='identity', help='Stage 2, F')
    ap.add_argument('--collapse', choices=sorted(COLLAPSE_CHOICES),
                    default='set_transformer', help='Stage 2, Collapse')
    ap.add_argument('--routing-head', choices=sorted(ROUTING_HEAD_CHOICES),
                    default='cosine_tau', help='Stage 3')
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--n-per-rung', type=int, default=100,
                    help='TileSampler budget per WSI when the manifest is built '
                         '-- the POOL, not a per-episode count')
    ap.add_argument('--n-support', type=int, default=5,
                    help='support positions per rung per batch')
    ap.add_argument('--n-query', type=int, default=10,
                    help='query positions per rung per batch')
    ap.add_argument('--support-native', action='store_true',
                    help='render support with CAMERA_GEOMETRY_ONLY (rotation '
                         'only); query always renders CAMERA_FULL')
    ap.add_argument('--n-choices', type=int, nargs='+', default=[3, 4, 5],
                    help='way counts whose rung combinations train (minus '
                         'Episodes.HELD_OUT_COMBOS)')
    ap.add_argument('--episode-reuse', choices=REUSE_MODES, default='hold_q',
                    help="none: 1 support x 1 query batch per draw, K times as "
                         "many draws. hold_s: 1 x K. hold_q: K x 1")
    ap.add_argument('--reuse-k', type=_auto_int, default='auto',
                    help='K, or auto = the largest K every training combination '
                         'can be drawn at, for both hold_s and hold_q')
    ap.add_argument('--val-reuse-k', type=_auto_int, default='auto',
                    help='val K (K x K pairs per held-out combination), or auto '
                         '= the largest every eval dataset supplies')
    ap.add_argument('--episodes-per-epoch', type=_auto_int, default='auto',
                    help='draws per epoch, or auto = every training combination '
                         'once (x K under --episode-reuse none)')
    ap.add_argument('--max-overlap', type=float, default=0.0,
                    help='support/query overlap rule: two positions on one WSI '
                         'sharing more than this fraction of the smaller '
                         'footprint never land on opposite sides. 0 = any '
                         'shared area')
    ap.add_argument('--feasibility-tries', type=int, default=100,
                    help='attempts per combination when measuring the largest '
                         'drawable K, and per draw in val')
    ap.add_argument('--eval-datasets', nargs='+',
                    default=['bracs/test', 'ki67_with_photo'],
                    help='held-out val -- the same datasets and held-out WSIs '
                         'as MppRoutingHead/cli/train.py')
    ap.add_argument('--val-n-per-rung', type=int, default=20)
    ap.add_argument('--merge', action='store_true',
                    help="val_scores_per_combo.csv / _per_rung.csv: this run's "
                         "_IDENTITY_FIELDS rows replace their old ones, every "
                         "other run's rows are kept")
    ap.add_argument('--epochs', type=int, default=20,
                    help='TOTAL epochs, resumed or not')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr-patience', type=int, default=3,
                    help='ReduceLROnPlateau on the val 6-way accuracy')
    ap.add_argument('--lr-factor', type=float, default=0.5)
    ap.add_argument('--loss', choices=('bal', 'ord_a', 'ord_b'), default='bal')
    ap.add_argument('--ordinal-weight', type=float, default=1.0)
    ap.add_argument('--ordinal-sigma', type=float, default=1.0)
    ap.add_argument('--encode-batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    add_cache_args(ap)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None)
    ap.add_argument('--resume-dir', default=None,
                    help='write this run\'s full state here every epoch, and '
                         'continue from it when it is already there. Unset: '
                         'train from scratch, write nothing')
    ap.add_argument('--supply-only', action='store_true',
                    help='build the manifests, print supply and what auto '
                         'resolves to, and exit')
    ap.add_argument('--max-wsi', type=int, default=None,
                    help='cap on the WSIs each training manifest (--train-'
                         'dataset AND --cross-domain-dataset) draws from: N '
                         'of them, chosen at random with --seed, so the same '
                         'N every time. At or above a dataset\'s size it '
                         'changes nothing. Val and test slides are named by '
                         'the split and ignore it')
    ap.add_argument('--wandb-project', default='prototypical-routing-head')
    ap.add_argument('--wandb-mode', default=os.environ.get('WANDB_MODE', 'online'))
    ap.add_argument('--run-name', default='')
    args = ap.parse_args()

    if args.pooling == 'passthrough':
        ap.error("--pooling passthrough gives Collapse an unpooled grid, not one "
                 "vector per tile -- pick cls/avg/max/attn")

    device = torch.device(args.device)
    out_dir = Path(args.out or job_result_dir('PrototypicalRoutingHead'))
    weights_dir = out_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    caches = open_caches(args, 'PrototypicalRoutingHead', device)
    reports = out_dir / 'sampler_reports'

    print(f'building manifest ({args.train_dataset})...', flush=True)
    rows = build_manifest(
        args.train_dataset, masks=caches.masks, sampler_root=caches.sampler_root,
        report_dir=reports / f'{args.train_dataset.replace("/", "_")}_train',
        tile_size=args.tile, n_per_rung=args.n_per_rung, seed=args.seed,
        max_wsi=args.max_wsi)
    train_pool = pool_by_rung(rows)
    print(f'  {len(rows)} positions across '
          f'{len(group_by_wsi_name(rows))} WSIs', flush=True)

    cross_pool = None
    if args.cross_domain_dataset:
        print(f'building cross-domain manifest ({args.cross_domain_dataset})...',
              flush=True)
        # Only the BRACS WSIs native at every BRACS_RUNGS rung, so a
        # cross-domain draw never pairs a resampled bracs position with a
        # native ki67 one.
        cd_wsi_names = (native_bracs_rung_wsi_names(args.cross_domain_dataset)
                        if args.cross_domain_dataset.startswith('bracs') else None)
        cd_rows = build_manifest(
            args.cross_domain_dataset, masks=caches.masks,
            sampler_root=caches.sampler_root,
            report_dir=(reports / f'{args.cross_domain_dataset.replace("/", "_")}'
                                  f'_cross_domain'),
            tile_size=args.tile, n_per_rung=args.n_per_rung, seed=args.seed,
            max_wsi=args.max_wsi, wsi_names=cd_wsi_names)
        cross_pool = pool_by_rung(cd_rows)

    val_pools = val_manifests(args, caches, out_dir)
    caches.masks.close()        # the segmenter is not needed past here

    print('\nsupply (positions per rung)', flush=True)
    print(f'  {"":28s}' + '  '.join(f'{r:>7g}' for r in RUNGS), flush=True)
    print(_supply_line(f'train {args.train_dataset}', train_pool), flush=True)
    if cross_pool is not None:
        print(_supply_line(f'cross {args.cross_domain_dataset}', cross_pool), flush=True)
    for dataset_id, pool in val_pools.items():
        print(_supply_line(f'val {dataset_id}', pool), flush=True)

    if args.supply_only and cross_pool is not None:
        # The cross-domain pool can lower K (bracs supplies a side of some
        # draws), so a run WITH it and a run without can resolve `auto`
        # differently. Both are shown; the second is what a run with
        # --cross-domain-dataset "" would use.
        print(f'\n[with cross-domain {args.cross_domain_dataset}]', flush=True)
        resolve_episodes(args, train_pool, cross_pool, val_pools)
        print('\n[without cross-domain]', flush=True)
        resolve_episodes(args, train_pool, None, val_pools)
        print('\n--supply-only: done', flush=True)
        return 0
    run, combos = resolve_episodes(args, train_pool, cross_pool, val_pools)
    if args.supply_only:
        print('\n--supply-only: done', flush=True)
        return 0
    k, val_k = run['reuse_k'], run['val_reuse_k']
    ks, kq = batch_shape(args.episode_reuse, k)
    pairs = reuse_pairs(ks, kq)

    encoder = build_encoder(args.encoder, args.dtype, device)
    spec = encoder.model_spec
    in_dim, num_prefix = int(spec.dim), int(spec.num_prefix)
    print(f'\nencoder {args.encoder}  dim={in_dim}  num_prefix={num_prefix}  '
          f'kind={spec.kind}  (frozen)', flush=True)

    pooling = Pooling(PoolingConfig(kind=args.pooling, in_dim=in_dim)).to(device)
    support_context, support_context_cfg = SUPPORT_CONTEXT_CHOICES[
        args.support_context](in_dim, device)
    query_context, query_context_cfg = QUERY_CONTEXT_CHOICES[
        args.query_context](in_dim, device)
    collapse, collapse_cfg = COLLAPSE_CHOICES[args.collapse](in_dim, device)
    head = ROUTING_HEAD_CHOICES[args.routing_head](in_dim, device)

    # The head has to take the shape Collapse hands it -- checked once here,
    # not discovered as a tensor error in the first draw.
    collapses = getattr(collapse, 'COLLAPSES', True)
    head_needs_raw = getattr(head, 'NEEDS_RAW_SUPPORT', False)
    if not getattr(head, 'ACCEPTS_EITHER', False) and collapses == head_needs_raw:
        ap.error(
            f'--collapse {args.collapse!r} '
            f'({"collapses" if collapses else "does NOT collapse"}) is not '
            f'compatible with --routing-head {args.routing_head!r} '
            f'({"needs raw per-rung support" if head_needs_raw else "needs a collapsed prototype stack"})')
    print(f'  stage 2: support-context={args.support_context}  '
          f'query-context={args.query_context}  collapse={args.collapse}   '
          f'stage 3 routing head: {args.routing_head}', flush=True)

    trainable = dict(pooling=pooling, support_context=support_context,
                     query_context=query_context, collapse=collapse, head=head)
    params = [p for m in trainable.values() for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=args.lr_factor, patience=args.lr_patience)

    render_cfg = RenderConfig(tile_size=args.tile)
    bank = CameraBank(render_cfg)
    rng = random.Random(args.seed)

    best = {'best': -1.0, '6rung': -1.0, 'native': -1.0}
    out_rung_rows: List[Dict] = []
    out_combo_rows: List[Dict] = []
    start_epoch = 0

    run_stem = _prototype_weight_filename(
        args.encoder, args.pooling, args.support_context, args.query_context,
        args.collapse, args.routing_head, args.cross_domain_dataset, args.loss,
        'x', run['episode_reuse'], run['reuse_k'])[:-len('_x.pt')]
    resume = ResumeFile.for_model(args.resume_dir, run_stem)
    identity = {**resume_identity(args, _NOT_IDENTITY), **run}
    state = resume.load(identity)
    if state is not None:
        ResumeFile.restore(state, modules=trainable, optimizers={'adam': opt},
                           schedulers={'plateau': lr_scheduler},
                           named_rngs={'episodes': rng})
        start_epoch = int(state['epoch'])
        best = state['best']
        out_rung_rows, out_combo_rows = state['extra']
        print(f'[resume] {resume.path}: continuing after epoch {start_epoch} '
              f'of {args.epochs}', flush=True)
    elif resume.enabled:
        print(f'[resume] no {resume.path.name} yet -- training from scratch, '
              f'writing it every epoch', flush=True)

    run_name = (f'{args.run_name}-{args.encoder}-{args.collapse}-'
                f'{args.routing_head}-{args.episode_reuse}' if args.run_name else
                f'{args.encoder}-{args.collapse}-{args.routing_head}-{args.episode_reuse}')
    wb = wandb_init(args.wandb_project, args.wandb_mode, run_name, config=dict(
        vars(args), **run))

    identity_cols = dict(
        training_framework='metric_based', encoder=args.encoder,
        pooling=args.pooling, support_context=args.support_context,
        query_context=args.query_context, collapse=args.collapse,
        routing_head=args.routing_head,
        cross_domain_dataset=_cross_domain_tag(args.cross_domain_dataset),
        train_dataset=args.train_dataset, loss=args.loss,
        support_native=args.support_native, seg=args.seg,
        episode_reuse=args.episode_reuse, reuse_k=k)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        for m in trainable.values():
            m.train()
        total_loss, n_steps, n_draws, n_redraws = 0.0, 0, 0, 0
        n_by_rung = {r: 0 for r in RUNGS}
        correct_by_rung = {r: 0 for r in RUNGS}

        # Every combination once per round, a fresh order each round; `none`
        # runs K rounds so its step count matches the held modes'.
        schedule = epoch_schedule(combos, run['episodes_per_epoch'], rng)

        for combo in schedule:
            rendered = None
            for _attempt in range(max(args.feasibility_tries, 1) * 4):
                sp, qp = training_pools(train_pool, cross_pool, rng, combo)
                d = draw(sp, qp, rng, combo, n_support=args.n_support,
                         n_query=args.n_query, ks=ks, kq=kq,
                         max_overlap=args.max_overlap)
                if d is not None:
                    rendered = render_draw(d, bank, render_cfg, deterministic=False,
                                           support_native=args.support_native,
                                           rng=rng)
                if rendered is not None:
                    break
                n_redraws += 1
            if rendered is None:
                raise RuntimeError(
                    f'combination {_combo_label(combo)} could not be drawn and '
                    f'rendered in {max(args.feasibility_tries, 1) * 4} attempts at '
                    f'K={k}, although K was measured drawable -- the render is '
                    f'failing, not the pool')
            n_draws += 1
            s_enc = [encode_group(g, encoder, args.encode_batch, device)
                     for g in rendered.supports]
            q_enc = [encode_group(g, encoder, args.encode_batch, device)
                     for g in rendered.queries]
            for si, qj in pairs:
                logits, target, _native = pair_forward(
                    rendered.rungs, s_enc[si][0], q_enc[qj][0], q_enc[qj][1],
                    num_prefix=num_prefix, device=device, **trainable)
                loss = compute_loss(logits, target, rendered.rungs, args.loss,
                                    args.ordinal_weight, args.ordinal_sigma)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss)
                n_steps += 1
                pred = logits.detach().argmax(-1)
                for local_true, local_pred in zip(target.tolist(), pred.tolist()):
                    rung = rendered.rungs[local_true]
                    n_by_rung[rung] += 1
                    correct_by_rung[rung] += int(local_true == local_pred)

        mean_loss = total_loss / max(n_steps, 1)
        acc_by_rung = {r: (correct_by_rung[r] / n_by_rung[r] if n_by_rung[r]
                           else float('nan')) for r in RUNGS}
        print(f'  epoch {epoch}/{args.epochs}  loss {mean_loss:.4f}  '
              f'({n_draws} draws x {ks}x{kq} = {n_steps} steps, {n_redraws} '
              f'redraws; episode-reuse {args.episode_reuse}, K {k})', flush=True)
        print(f'    rung        {"  ".join(f"{r:>7g}" for r in RUNGS)}\n'
              f'    query n     {"  ".join(f"{n_by_rung[r]:>7d}" for r in RUNGS)}'
              f'   (of {sum(n_by_rung.values())} query examples)\n'
              f'    train acc   {"  ".join(f"{acc_by_rung[r]:>7.4f}" for r in RUNGS)}',
              flush=True)

        # Held-out val, K x K -- same seed every epoch, so the same questions.
        for m in trainable.values():
            m.eval()
        print(f'    val  K={val_k} ({val_k}x{val_k} pairs per combination)', flush=True)
        results = score_kxk(
            val_pools, HELD_OUT_COMBOS, k=val_k, n_support=args.n_support,
            n_query=args.n_query, max_overlap=args.max_overlap, seed=args.seed,
            tries=args.feasibility_tries, bank=bank, render_cfg=render_cfg,
            support_native=args.support_native, encoder=encoder,
            num_prefix=num_prefix, batch_size=args.encode_batch, device=device,
            **trainable)
        epoch_combo_rows, epoch_rung_rows, summary = kxk_report(
            results, epoch, k=val_k, **identity_cols)
        out_rung_rows += epoch_rung_rows
        out_combo_rows += epoch_combo_rows

        lr_before = opt.param_groups[0]['lr']
        if not math.isnan(summary['six_rung']):
            lr_scheduler.step(summary['six_rung'])
        lr_after = opt.param_groups[0]['lr']
        if lr_after != lr_before:
            print(f'    lr {lr_before:.2e} -> {lr_after:.2e}  (val 6rung accuracy '
                  f'stalled {args.lr_patience} epochs)', flush=True)

        metrics = {
            'train_loss': mean_loss, 'n_draws': n_draws, 'n_steps': n_steps,
            'n_redraws': n_redraws, 'lr': lr_after,
            'val_total_acc': summary['total'], 'val_6rung_acc': summary['six_rung'],
            'val_native_acc': summary['native'],
            **{f'train_acc_rung_{r:g}': acc_by_rung[r] for r in RUNGS},
            **{f'val_{d}/acc': a for d, a in summary['by_dataset'].items()},
            **{f'val_{r["val_dataset"]}/combo_{r["combo"]}/level_accuracy':
               r['level_accuracy'] for r in epoch_combo_rows},
        }
        wandb_log(wb, epoch, metrics)

        best = save_tagged(weights_dir,
                           (pooling, support_context, query_context, collapse, head),
                           (support_context_cfg, query_context_cfg, collapse_cfg),
                           encoder, args, in_dim, epoch, summary, best, run)
        resume.save(identity, epoch=epoch, modules=trainable,
                    optimizers={'adam': opt}, schedulers={'plateau': lr_scheduler},
                    best=best, extra=(out_rung_rows, out_combo_rows),
                    named_rngs={'episodes': rng})

    wandb_finish(wb)

    if not out_combo_rows:
        print('no val rows (no epoch ran) -- nothing to write', flush=True)
        return 0
    for name, rows_ in (('val_scores_per_rung.csv', out_rung_rows),
                        ('val_scores_per_combo.csv', out_combo_rows)):
        path = out_dir / name
        write_rows = _merge_val_scores(path, rows_) if args.merge else rows_
        fields = list(dict.fromkeys(k_ for r in write_rows for k_ in r))
        with open(path, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            wr.writerows(write_rows)
        print(f'{path}  ({len(write_rows)} rows)', flush=True)
    print(f'{weights_dir}  (*_best.pt / *_6rung.pt / *_native.pt)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
