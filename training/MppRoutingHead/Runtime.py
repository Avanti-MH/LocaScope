'''What THIS package needs beyond the generic encoder/head plumbing --
`cli/train.py`'s val/test scoring loop and its wandb logging. The generic
pieces (head classes, the `Head` assembly, patches->raw-features, checkpoint
save/load) moved to `aiNNModel/models/` on 2026-09-17: nothing about them was
specific to routing a tile to an mpp rung, and `stage1_estimation/
ClassifierEstMpp.py` needs the same pieces a training loop does, without
importing a training package to get them. This file re-exports what
`cli/train.py`/`cli/evaluate.py` still need from there, so their own imports
do not have to know which pieces live where.

`HEAD_CHOICES` (was `ARMS`) stays HERE, not in the generic layer: it is a
REGISTRY OF THIS TASK'S OWN CHOICES ('linear'/'mlp'/'attn_linear'/'arcface',
spec.md's 2-1/2-5/2-6/2-3), including which BASELINE (2 = frozen, 3 =
fine-tuned) each is valid for -- vocabulary that means nothing outside this
package. `common.Head.Head` (the mechanism) is generic; which named
combinations exist for THIS task is not.
'''
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

# _paths holds the one definition of every package's sys.path entry
# (setup_import_paths) -- utilities/ goes on the path here, by hand, because
# that function is INSIDE it and this is the one step nothing else can do
# for this file. Same idiom every test_modules/cli entry point uses.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'utilities'))
from _paths import setup_import_paths                                  # noqa: E402
setup_import_paths()

import torch                                                        # noqa: E402

from training.MppRoutingHead.Datasets import (                      # noqa: E402
    RUNGS, RenderConfig, iterate_epoch)
from Heads import ArcFaceHead, HeadConfig, LinearHead, MlpHead       # noqa: E402
from Head import Head, grid_view, pooled_view                       # noqa: E402
from Features import encode_raw, normalise_patches, trunk_raw       # noqa: E402
from Checkpoints import (build_from_checkpoint, save_checkpoint,    # noqa: E402
                         weight_filename)

__all__ = [
    'ArcFaceHead', 'HeadConfig', 'LinearHead', 'MlpHead',
    'Head', 'grid_view', 'pooled_view',
    'encode_raw', 'normalise_patches', 'trunk_raw',
    'build_from_checkpoint', 'save_checkpoint', 'weight_filename',
    'BASELINE3_ENCODER', 'HEAD_CHOICES', 'MIX_LAYERS', 'head_parts', 'heads_for',
    'wandb_init', 'wandb_log', 'wandb_finish', 'wandb_epoch_metrics',
    'score', 'rescore', 'predict',
]


#: baseline 3 is ConvNeXt V2 by DEFINITION (spec.md's "Two baselines"), not by
#: default -- it is the branch of the comparison that asks what a fine-tuned
#: natural-image CNN can do, so there is no encoder to choose. baseline 2's
#: encoder IS a choice (`--encoders`), because "which frozen pathology
#: foundation model" is one of the things that baseline is measuring.
BASELINE3_ENCODER = 'convnext_v2'

#: name -> (how one tile's tokens are reduced, which classifier produces the
#:          logits, which baselines this choice is valid for, HeadConfig
#:          field overrides on top of the CLI's own --mlp-* defaults).
#:
#: TWO AXES, not a list of alternatives -- see `common.Head.Head`'s own
#: docstring. The third field is spec.md's Baseline column, not a default to
#: override: 'arcface' is 2-3 and has no 3-x twin there, so asking for
#: `--baseline 3 --heads arcface` gets a skip and a printed reason rather than
#: a run that quietly invents a row the spec does not name.
#:
#: The fourth field is EMPTY for every entry except the MlpHead architecture
#: variants below -- 2026-09-17. `mlp_deep`/`mlp_wide`/`mlp_deep_wide`/
#: `mlp_deep_residual` are the SAME MlpHead class as `mlp`, just built from a
#: `HeadConfig` with different `mlp_depth`/`mlp_width`/`mlp_residual`; giving
#: each its own registered NAME (rather than sweeping --mlp-depth/--mlp-
#: width/--mlp-residual across separate runs) is what lets all five train
#: and score side by side in one `val_scores.csv`, each under its own
#: `weight_filename()` -- sharing one head_name would have them overwrite
#: each other's checkpoint and CSV rows. `mlp_width_mult`, not a literal
#: `mlp_width`, is how a variant asks for "wider than THIS encoder's
#: in_dim" -- in_dim differs by encoder (1536 gigapath/uni2, 768
#: convnext_v2) and one registry entry is defined once for every encoder it
#: might run against, so it cannot hardcode a literal width; `cli/train.py`'s
#: per-head `HeadConfig` builder resolves it once in_dim is actually known.
#:
#: spec.md's 2-2 (NCM) and 2-4 (Mahalanobis) are still absent: neither is
#: gradient-trained, so both need a fit-once path the training loop lacks.
#: What a `mix_` head mixes: the blocks at a quarter, half, three quarters and
#: the whole of the encoder's depth (`Heads.resolve_encoder_layers`), so one
#: entry means the same spread on UNI2's 24 blocks (5, 11, 17, 23) and
#: GigaPath's 40 (9, 19, 29, 39).
MIX_LAYERS = (0.25, 0.5, 0.75, 1.0)

#: The head NAME is `[mix_]` + `[attn_ | clsattn_ | nothing]` + classifier:
#: nothing before the classifier is the `fixed` reduction on the last block.
#: The name is only a key -- what a head is lives in its entry -- and
#: `head_parts` reads the parts back off the entry, never off the string.
HEAD_CHOICES = {
    'linear':        ('fixed', LinearHead,  (2, 3), {}),  # 2-1/3-1  main line
    'mlp':           ('fixed', MlpHead,     (2, 3), {}),  # 2-5/3-2  observation
    'mlp_deep':      ('fixed', MlpHead,     (2, 3),
                      dict(mlp_depth=2)),
    'mlp_wide':      ('fixed', MlpHead,     (2, 3),
                      dict(mlp_width_mult=2)),
    'mlp_deep_wide': ('fixed', MlpHead,     (2, 3),
                      dict(mlp_depth=2, mlp_width_mult=2)),
    'mlp_deep_residual': ('fixed', MlpHead, (2, 3),
                      dict(mlp_depth=2, mlp_residual=True)),
    'mlp_narrow':    ('fixed', MlpHead,     (2, 3),
                      dict(mlp_width_mult=0.5)),  # narrower than in_dim, not wider
    'attn_linear':   ('attn',  LinearHead,  (2, 3), {}),  # 2-6/3-3  learnable pool
    # The learnable pool under the two deep MLPs that lead on the CLS view.
    # attn_linear alone could not say whether the pool helps: it paired the
    # pool with the weakest classifier, and classifier depth moved val more
    # than the reduction did.
    'attn_mlp_deep':      ('attn', MlpHead, (2, 3), dict(mlp_depth=2)),
    'attn_mlp_deep_wide': ('attn', MlpHead, (2, 3),
                           dict(mlp_depth=2, mlp_width_mult=2)),
    # clsattn needs a CLS and a mix needs blocks of one width, so neither runs
    # on baseline 3's ConvNeXt; `common.Head.Head` refuses both there as well.
    'clsattn_mlp_deep':      ('clsattn', MlpHead, (2,), dict(mlp_depth=2)),
    'mix_attn_mlp_deep':     ('attn', MlpHead, (2,),
                              dict(mlp_depth=2, encoder_layers=MIX_LAYERS)),
    'mix_clsattn_mlp_deep':  ('clsattn', MlpHead, (2,),
                              dict(mlp_depth=2, encoder_layers=MIX_LAYERS)),
    'arcface':       ('fixed', ArcFaceHead, (2,),   {}),  # 2-3      angular margin
}


def head_parts(name: str) -> Tuple[str, str, bool]:
    """`(classifier, reduction, mixes)` for a registered head name: the
    classifier family it is drawn in (`mlp_deep` for `mix_attn_mlp_deep`), its
    reduction, and whether it mixes encoder layers. Read off `HEAD_CHOICES`;
    a name that is not registered (an old CSV) is taken as `fixed`, unmixed."""
    entry = HEAD_CHOICES.get(name)
    if entry is None:
        return name, 'fixed', False
    reduction = entry[0]
    mixes = bool(len(entry) > 3 and entry[3].get('encoder_layers'))
    prefix = ('mix_' if mixes else '') + ('' if reduction == 'fixed'
                                          else f'{reduction}_')
    return name[len(prefix):] if name.startswith(prefix) else name, reduction, mixes


def heads_for(requested: List[str], baseline: int) -> List[str]:
    '''The requested heads this baseline actually has, with a line for each
    one it does not -- silence here would read as "it ran and tied".'''
    keep = [n for n in requested if baseline in HEAD_CHOICES[n][2]]
    for name in requested:
        if name not in keep:
            print(f'  baseline {baseline}: skipping head {name} -- spec.md '
                  f'lists it under baseline '
                  f'{"/".join(map(str, HEAD_CHOICES[name][2]))} only',
                  flush=True)
    return keep


# ══════════════════════════════════════════════════════════════════════════
#  wandb -- same idiom as SuperPathPoint/SuperPoint/Trainer.py's
#  _start_wandb/_log/_finish_wandb: optional dependency, one run per MODEL,
#  and WANDB_MODE (read as the CLI default, not read here) is what a smoke
#  run sets to keep this off the server -- see jobscripts/_env.sh.
# ══════════════════════════════════════════════════════════════════════════

def wandb_init(project: str, mode: str, name: str, config: Dict,
               run_id: str = None):
    '''Returns a run, or `None` if wandb is not installed or `mode='disabled'`
    -- every other function here takes that `None` and no-ops, so a caller
    never has to branch on whether wandb exists.

    ONE RUN PER MODEL, not one per invocation of `cli/train.py`. A single
    `--baseline all` run trains up to 2 encoders x 4 heads (baseline 2) + 3
    heads (baseline 3) -- up to 11 independently-optimised models -- and
    cramming all of them into one wandb run would mean one run's charts hold
    11 unrelated loss curves under manufactured key prefixes.
    `run_baseline2` inits one run per ENCODER (its several heads share one
    epoch axis and one encoder forward pass, so they are one experiment with
    several lines); `run_baseline3` inits one run per HEAD (each owns its own
    trunk, optimizer and epoch loop, so each is its own experiment).

    `run_id` makes the run CONTINUABLE: with an id, `resume='allow'` appends to
    the run of that id when it exists and starts it when it does not, so a model
    resumed from its checkpoint keeps drawing on the same curves
    (`ResumeFile.wandb_run_id`). None starts a fresh run each time.

    `config` goes in through `config.update(..., allow_val_change=True)` rather
    than `init(config=...)`: a continued run already holds a config, and a value
    that differs from last time (`slurm_job_id` always does) is the new job, not
    an error.
    '''
    try:
        import wandb                                                # noqa: PLC0415
    except ImportError:
        print('wandb not installed; logging to stdout and the CSVs only',
              flush=True)
        return None
    run = wandb.init(project=project, mode=mode, name=name or None,
                     id=run_id, resume='allow' if run_id else None)
    run.config.update(config, allow_val_change=True)
    return run


def wandb_log(run, step: int, metrics: Dict[str, float]) -> None:
    '''NaN values are dropped, not sent. `level_accuracy_resampled` is `nan`
    whenever a dataset's rungs are all native (Ki67's own 2x pyramid, every
    epoch) -- wandb accepts a NaN point, but a line that is NaN every epoch on
    one dataset and a real number on the other reads as a bug in the chart
    rather than as the mathematical certainty `score`'s docstring explains it
    to be.'''
    if run is None:
        return
    run.log({k: v for k, v in metrics.items() if v == v}, step=step)


def wandb_finish(run) -> None:
    if run is not None:
        run.finish()


def wandb_epoch_metrics(rows: List[Dict]) -> Dict[str, float]:
    '''Flattens `cli/train.py`'s `val_report` output for ONE epoch into
    `{head}/{val_dataset}/{metric}` keys, plus `{head}/train_loss` off the
    `all` row -- the SAME rows that become `val_scores.csv`, read differently
    rather than recomputed, so the chart and the CSV cannot disagree about
    what happened in a given epoch.'''
    keys = ('level_accuracy', 'mpp_error_relative_p50', 'coarse_share_of_errors',
           'level_accuracy_native', 'level_accuracy_resampled')
    out: Dict[str, float] = {}
    for row in rows:
        prefix = f'{row["head"]}/{row["val_dataset"]}'
        for k in keys:
            out[f'{prefix}/{k}'] = row[k]
        if row['val_dataset'] == 'all':
            out[f'{row["head"]}/train_loss'] = row['train_loss']
    return out


# ══════════════════════════════════════════════════════════════════════════
#  scoring -- the same numbers bench_mpp_feature_decomposition.py's score()
#  reports, so the two are comparable without a conversion
# ══════════════════════════════════════════════════════════════════════════

def score(pred_class: torch.Tensor, true_class: torch.Tensor,
          native: torch.Tensor | None = None) -> Dict:
    '''`mpp_error_relative` needs no per-slide `base_mpp`: predicted and true
    mpp for one example share that slide's base_mpp, and it cancels in the
    ratio, leaving `|RUNGS[pred] - RUNGS[true]| / RUNGS[true]`. That is the
    same quantity the KNN bench reports, restricted to a discrete ladder.

    `native` (`Datasets.render_row`'s third element,
    `Render.reads_natively`) splits `level_accuracy` a second way, and
    the split is a CONFOUND CHECK rather than a breakdown for its own sake.

    A rung whose mpp is not on the slide's pyramid is read one level finer and
    LANCZOS-resampled down, which leaves its own high-frequency signature. That
    signature does not average out, because which rungs it lands on is fixed by
    the pyramid: on Ki67's 2x pyramid (the TRAINING set) every rung is native,
    while on BRACS's 4x pyramid rungs 2 and 8 are resampled (rung 32 varies per
    slide -- see spec.md). So there the extra resampling can be aligned with
    the CLASS, and a head that has learnt to detect resampling scores as if it
    had learnt scale. A large gap between these two numbers is that; a small
    one says the concern did not materialise.
    '''
    rungs = torch.tensor(RUNGS, dtype=torch.float64)
    pred_r, true_r = rungs[pred_class], rungs[true_class]
    correct = pred_class == true_class
    wrong = ~correct
    out = dict(
        n=int(len(pred_class)),
        level_accuracy=float(correct.double().mean()),
        mpp_error_relative_p50=float(((pred_r - true_r).abs() / true_r).median()),
        coarse_share_of_errors=(float((pred_r[wrong] > true_r[wrong]).double().mean())
                                if wrong.any() else float('nan')),
    )
    # Flat keys, always present: one `csv.DictWriter` writes every row of a
    # scores file off the FIRST row's keys, so a column that appears only on
    # some rows is a silently truncated table.
    for label, keep in (('native', native),
                        ('resampled', None if native is None else ~native)):
        n = 0 if keep is None else int(keep.sum())
        out[f'n_{label}'] = n
        out[f'level_accuracy_{label}'] = (float(correct[keep].double().mean())
                                          if n else float('nan'))
    return out


def rescore(detail_rows: List[Dict]) -> Dict:
    '''`score()` again over a SUBSET of `predict`'s per-tile detail -- the rows
    already carry `gt_class`, `pred_class` and `native`, so a breakdown costs a
    list comprehension rather than a second forward pass.

    Exists because `level_accuracy_native` vs `level_accuracy_resampled` is
    only meaningful WITHIN one dataset. Ki67 is a 2x pyramid, so every one of
    its rungs is native; BRACS is 4x, so rungs 2/8 (and part of 32) are
    resampled. Scored over the two together, the "native" side is BRACS's
    native rungs plus the whole of Ki67 while the "resampled" side is BRACS
    alone -- the comparison then measures the dataset, not the resampling.
    Filtering by `dataset` first is what makes it measure what it is named
    after.
    '''
    if not detail_rows:
        # `score` cannot take an empty tensor -- `.median()` raises on one --
        # and an empty subset is reachable without a bug (a dataset all of
        # whose rungs are native has no resampled rows at all).
        nan = float('nan')
        return dict(n=0, level_accuracy=nan, mpp_error_relative_p50=nan,
                    coarse_share_of_errors=nan, n_native=0,
                    level_accuracy_native=nan, n_resampled=0,
                    level_accuracy_resampled=nan)
    pick = lambda k, dt: torch.tensor([r[k] for r in detail_rows], dtype=dt)  # noqa: E731
    return score(pick('pred_class', torch.int64), pick('gt_class', torch.int64),
                 pick('native', torch.bool))


def rescore_by_rung(detail_rows: List[Dict]) -> Dict[float, Dict]:
    '''`rescore` again, split by `gt_rung` -- one `score()` per rung rather
    than one pooled over all of them.

    `level_accuracy`'s single pooled number cannot show whether a SPECIFIC
    rung is improving: `RICHNESS`'s own coarse-rung shortfall means fine
    rungs are oversupplied relative to coarse ones, so a tile-pooled average
    is dominated by whichever rungs happen to have the most val tiles --
    2026-09-19's own finding was a run posting ~90% pooled while its worst
    rung sat under 10% the entire time, invisible in that one number. This
    is the one place a rung's own trajectory across epochs can be read at
    all, which is what decides whether `--loss ord_a`/`ord_b` are actually
    helping the rungs they target rather than moving the pooled average by
    accident.

    Caller filters by dataset FIRST, same reasoning `rescore`'s own
    docstring gives for native/resampled: a rung's accuracy is only
    comparable within one dataset, since which rungs are native differs by
    pyramid and a resampled rung is a harder question than a native one.
    '''
    return {rung: rescore([r for r in detail_rows if r['gt_rung'] == rung])
            for rung in RUNGS}


# ══════════════════════════════════════════════════════════════════════════
#  running a split
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(rows, heads: Dict[str, Head], raw_of, num_prefix: int, *,
            tile: int, wsi_group_size: int, batch_size: int,
            num_workers: int):
    '''`(scores, detail)`: `{head: score dict}` and `{head: [one dict per
    tile]}`.

    Scored through the SAME `iterate_epoch` the training loop uses. `raw_of`
    is `encode_raw`-or-`trunk_raw` already bound to an encoder, which is the
    only thing that differs between the two baselines.

    `detail` is one row per TILE -- where it came from (`wsi_name`, `x`, `y`),
    what it is (`rung`, `bucket`, `native`), and what happened (`gt_class`,
    `pred_class`, and the same two as rung values). An aggregate accuracy
    cannot answer "is it only wrong on the near-blank tiles" or "is it only
    wrong on one slide"; these rows can, and they cost one dict per example.
    The caller decides whether to write them -- the training loop discards its
    copy, `cli/evaluate.py` writes them out.

    NOTHING IS CACHED. `split='eval'` makes `render_row` seed its rng from the
    row's own identity, so the same position renders the same pixels on every
    call -- which is what a val curve and a reported test number both need, and
    it is a SEED rather than a stored corpus.

    `epoch_seed=0` fixes the WSI grouping too, so two runs see the same
    positions in the same order.

    Every head is put in `eval()` and left there -- the caller is the one that
    knows whether training continues afterwards (ArcFace in particular gives
    different logits in the two modes, deliberately).
    '''
    for head in heads.values():
        head.eval()
    preds = {name: [] for name in heads}
    meta: List = []
    labels, native = [], []
    for batch in iterate_epoch(rows, wsi_group_size=wsi_group_size,
                               batch_size=batch_size, num_workers=num_workers,
                               split='eval', cfg=RenderConfig(tile_size=tile),
                               epoch_seed=0):
        raw = raw_of(batch['patches'])
        meta += batch['rows']
        labels.append(batch['labels'])
        native.append(batch['native'])
        for name, head in heads.items():
            preds[name].append(head(raw, num_prefix).argmax(-1).cpu())
    labels, native = torch.cat(labels), torch.cat(native)

    scores, detail = {}, {}
    for name, chunks in preds.items():
        pred = torch.cat(chunks)
        scores[name] = score(pred, labels, native)
        detail[name] = [
            dict(dataset=row.dataset, wsi_name=row.wsi_name, x=row.x, y=row.y,
                 rung=row.rung, bucket=row.bucket, native=bool(nat),
                 gt_class=int(g), gt_rung=RUNGS[int(g)],
                 pred_class=int(p), pred_rung=RUNGS[int(p)],
                 correct=int(g) == int(p))
            for row, nat, g, p in zip(meta, native, labels, pred)
        ]
    return scores, detail
