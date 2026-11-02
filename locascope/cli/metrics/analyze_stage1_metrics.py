#!/usr/bin/env python3
"""Score the stage-1 entries bench_stage1_mpp.py wrote and say which
mpp-estimation method actually wins.

Takes bench_stage1_mpp's own flags and computes the same addresses: for
every slide its FoV draw, for every --stage1 method its stage-1 entry under
--stage-cache-job. No model and no GPU; the slide is opened only for its
pyramid. The entries are joined with the ground truth -- where each FoV was
placed (`FovSupply.geometry`) -- into one table, written as
`<out>/stage1_<set>.csv` (`set_name`), and everything below reads that table.

    table columns (one row per (query, method[, vote rule])):
        dataset, wsi_name, x, y, h, w        -- FoV identity
        rung, native, gt_mpp, gt_ds          -- ground truth
        encoder, classifier, reduction, loss,
        weights                               -- which method/config;
                                              KnnEstMpp leaves classifier/
                                              reduction/loss/weights blank.
                                              `loss`: bal/ord_a/
                                              ord_b, blank on a checkpoint
                                              trained before --loss existed
                                              (treated as 'bal' by
                                              method_of(), never as blank --
                                              see that function's own
                                              docstring)
        estimated_ds, estimated_mpp,
        chosen_ds, chosen_mpp, chosen_level   -- EstMppResult's five
        extra_json                            -- method-specific secondary
                                              fields (predicted_class,
                                              confidence, ...), JSON-encoded
                                              so a new method never needs a
                                              new column
        kind                                  -- knn / classifier / prototype /
                                              classic (absent in some files,
                                              see kind_of)
        vote                                  -- the FoVVote rule, on
                                              classifier and prototype rows
                                              only: one row per rule, all
                                              from one forward pass
        fov_*                                 -- that FoV's per-patch
                                              distribution, on voting rows
                                              (fov_stats, from the probs table)

        set                                   -- the --datasets ids (`@legacy-val`)
        estimator_id                          -- the estimator's identity_id,
                                              weights included
        risk_*                              -- FoVVote.diagnose: the raw
                                              quantities FoV_Vote.md's danger
                                              section names for the row's rule

View 3 carries a 95% FoV-bootstrap interval (resampled inside each rung) and
the log2-rung MAE. Section 6 is FoV_Vote.md's risk flags: prevalence,
accuracy flagged / unflagged and the risk ratio, overall and per GT rung.
Their thresholds are fitted on a VAL run (`--fit-thresholds`, which refuses a
test file) and read back for test (`--thresholds auto` finds the val file of
the same recipe), refused when the test file's estimators -- per kind, by
`estimator_id` -- are not the ones the val file was fitted on. Section 7 is risk-coverage per rule; `*_confusion.csv`
holds every method's confusion matrix.

Section 5 is the vote diagnosis: which rule wins, where two rules disagree
which is right, how much any rule could gain (some-rule-right vs best rule),
and every rule's accuracy inside strata of the FoV distribution -- so the
answer can be "rule X when the patches agree, rule Y when they scatter",
not only one winner. Its tables also go to `<csv stem>_vote_*.csv` and
`<csv stem>_fov_distribution.csv` next to the input.

Every method is scored on the same drawn FoVs -- one draw per slide, read by
address -- so the comparison is paired. The analysis outputs (the table, the
vote and risk tables, the thresholds, the figures) go to --out, default
`result/<job>/`: they are results, not reusable computation, so not the
cache. A test run finds the thresholds its val run fitted in the same --out
(`stage1_legacy-val_thresholds.json` for `@legacy-test`, `val_set_of`).

Scored on `estimated_*`, NOT `chosen_*`. `chosen_ds`/`chosen_level` already
went through the shared "snap to this WSI's own pyramid" step
(`StageInterface.routed_level`) that every method shares, so
scoring on it would measure that shared step as much as the method.

THREE VIEWS, not one table:
    per (wsi_name, rung, method)   -- does a method fail on one slide, or
                                    everywhere at that scale?
    per (rung, method), cross-slide -- with n: native levels mean not
                                    every slide contributes every rung (BRACS
                                    has no native ds=2), so a rung's number
                                    can rest on one slide's worth of shots
                                    and the table has to say so rather than
                                    let a reader assume equal weight.
    one overall number per method  -- the MEAN of the per-rung accuracies
                                    above, not the pooled accuracy over every
                                    shot: rungs are not sampled equally
                                    (native levels leave some rungs thin), and
                                    pooling would let whichever rung has the
                                    most shots decide the winner instead of
                                    every scale counting equally.

native vs resampled, same reason `training/ClassifyingRoutingHead/Runtime.score`
splits it: a rung whose mpp is not on a slide's own pyramid is read one level
finer and resampled down, which leaves a resampling signature a method could
be scoring on instead of genuine scale. Meaningless pooled across datasets
with different pyramid steps (BRACS 4x, Ki67 2x) -- always split by dataset
first.

FIGURES (`make_figures`, `<out>/<csv stem>_fig_<name>.png`), each titled with what it shows:
    methods                 which method is best: accuracy and its 95% interval, best on top
    by_rung                 where each method fails: accuracy at each true rung
    levels_off              how wrong when wrong, in the dataset's own pyramid levels (BRACS steps x4 of ds,
                            Ki67 x2, so one level is one level on both)
    native_vs_resampled     whether resampling flatters a method
    confusion_<dataset>     what each method calls each true rung
    per_slide_<dataset>     does a method fail on one slide or everywhere
    votes, votes_pairwise   which vote rule, and who is right when two rules differ
    strata_<dataset>_<kind> which rule when, by how the patches look (agreement, entropy, spread, margin)
    patch_stats_<...>       what a wrong FoV looks like, statistic by statistic, scale by scale
    risk_flags_<...>        every risk flag: the number it is made of for the right and wrong FoVs with its
                            threshold, or the accuracy with and without it
    risk_coverage           what refusing the least sure FoVs buys, per rule
`--top N` is how many voting checkpoints the figures draw: the best by their mean over the datasets, the
same ones on every dataset (a method not scored on a dataset is simply absent from its panel); every knn
and classic baseline is always drawn.

Usage:
    python utilities/cli/metrics/analyze_stage1_metrics.py <bench_stage1_mpp flags> \\
        --datasets @legacy-val --fit-thresholds
    python utilities/cli/metrics/analyze_stage1_metrics.py <bench_stage1_mpp flags> \\
        --datasets @legacy-test --thresholds auto
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import sys
from typing import List, Optional, Sequence

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import _paths                                                       # noqa: E402
_paths.setup_import_paths()



# ── the table, from the cache ─────────────────────────────────────────────────

#: The five EstMppResult fields every row carries; the rest of an output or
#: vote row is the method's own, kept in `extra_json`.
_RESULT = ('estimated_ds', 'estimated_mpp', 'chosen_ds', 'chosen_mpp',
           'chosen_level')


def fov_stats(probs, classes_ds, gt_ds: float) -> dict:
    """What one FoV's per-patch distribution looked like -- the columns the
    vote diagnosis stratifies by. Independent of the vote rule, so every vote
    row of one FoV carries the same values. `probs` is [patches, classes].

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
    import numpy as np                                            # noqa: PLC0415
    p = np.asarray(probs, dtype=np.float64)
    m, c = p.shape
    arg = p.argmax(axis=1)
    counts = np.bincount(arg, minlength=c)
    pooled = p.mean(axis=0)
    log_c = math.log(c) if c > 1 else 1.0

    def entropy(q):
        return -(np.log(np.clip(q, 1e-12, None)) * q).sum(axis=-1)

    log2_ds = np.log2(np.asarray(classes_ds, dtype=np.float64))[arg]
    top = np.sort(pooled)[::-1][:2]
    gt = min(range(c), key=lambda i: abs(math.log(float(classes_ds[i]) / gt_ds)))
    reachable = abs(math.log(float(classes_ds[gt]) / gt_ds)) < math.log(1.01)
    out = dict(
        fov_n_patches=m, fov_n_classes=c,
        fov_agree_frac=float(counts.max()) / m,
        fov_n_distinct=int((counts > 0).sum()),
        fov_patch_entropy=float(entropy(p).mean()) / log_c,
        fov_pooled_entropy=float(entropy(pooled)) / log_c,
        fov_pooled_margin=float(top[0] - top[1]) if c > 1 else 1.0,
        fov_argmax_log2_spread=float(log2_ds.std()) if m > 1 else 0.0,
        fov_gt_reachable=reachable)
    if reachable:
        out.update(fov_gt_prob=float(pooled[gt]),
                   fov_gt_rank=int((pooled > pooled[gt]).sum()),
                   fov_gt_patch_frac=float(counts[gt]) / m)
    return out


def _describe(stage) -> dict:
    """The method columns `method_of` labels a row by: the kind, the encoder,
    and for a checkpoint method its recipe name in `classifier`."""
    cfg = stage.cfg
    head = {'classifier': stage.name, 'prototype': f'proto:{stage.name}'}
    return dict(kind=stage.method,
                encoder=getattr(cfg, 'encoder', '') or stage.method,
                classifier=head.get(stage.method, ''), reduction='', loss='',
                read_level='', weights=getattr(cfg, 'weights', '') or '')


def _probs_matrix(rows):
    """[patches, classes] and the classes' ds from one FoV's `probs` rows."""
    import numpy as np                                            # noqa: PLC0415
    classes = sorted({float(r['class_ds']) for r in rows})
    n = 1 + max(int(r['patch']) for r in rows)
    p = np.zeros((n, len(classes)))
    for r in rows:
        p[int(r['patch']), classes.index(float(r['class_ds']))] = float(r['p'])
    return p, classes


def rows_from_cache(args) -> list:
    """One row per (FoV, method[, vote rule]) of every hit stage-1 entry the
    flags address, joined with where the FoV was placed."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', 'bench_modules'))
    from BenchCommon import (by_index, read_rows, slide_supplies,  # noqa: PLC0415
                                 stage1_record)
    from bench_stage1_mpp import stages_of                        # noqa: PLC0415
    from DsLadder import DEFAULT_RUNGS                            # noqa: PLC0415
    from SafeSlide import SafeSlide                               # noqa: PLC0415
    from TissueMaskConfig import MASK_RECIPES, MaskMaker          # noqa: PLC0415

    job = args.stage_cache_job
    stages = stages_of(args)
    fov_masks = MaskMaker(MASK_RECIPES['hest'], args.fov_mask_cache_job, 'cpu')
    rows, skipped = [], 0
    for dataset, name, path, supply in slide_supplies(args, fov_masks, job):
        base = supply.render_address.on(job)
        wsi = SafeSlide(path)
        base_mpp, lds = float(wsi.base_mpp), list(wsi.level_downsamples)
        wsi.close()
        geo = [supply.geometry(s.meta) for s in supply.sampler]
        for s in stages:
            entry = base.entry('stage1')
            state, diff = entry.status(s.id, stage1_record(s, base, args.limit))
            if state != 'hit':
                print(f'  {name} {s.id}: {state}' + ''.join(f'\n    {d}' for d in diff))
                continue
            table = {role: by_index(read_rows(entry.path(role, s.id, '.csv')))
                     for role in ('output', 'votes', 'probs')}
            desc = _describe(s)
            for index, (out,) in sorted(table['output'].items()):
                if out.get('error'):
                    skipped += 1
                    continue
                g = geo[index]
                gt_ds = float(g['ds'])
                fov = dict(
                    dataset=dataset, wsi_name=name, index=index,
                    x=int(g['x0']), y=int(g['y0']),
                    h=int(g['sensor_h']), w=int(g['sensor_w']),
                    rung=min(DEFAULT_RUNGS,
                             key=lambda r: abs(math.log(gt_ds) - math.log(r))),
                    native=abs(math.log(gt_ds / float(lds[int(g['level'])])))
                    < math.log(1.01),
                    gt_mpp=base_mpp * gt_ds, gt_ds=gt_ds, set=' '.join(args.datasets),
                    estimator_id=s.id, **desc)
                votes = table['votes'].get(index)
                if not votes:
                    rows.append(dict(
                        fov, vote='', **{k: out[k] for k in _RESULT},
                        extra_json=json.dumps(
                            {k: v for k, v in out.items()
                             if k not in _RESULT + ('index', 'error', 't_s')})))
                    continue
                probs, classes = _probs_matrix(table['probs'][index])
                stats = fov_stats(probs, classes, gt_ds)
                for v in votes:
                    risk = {k: v[k] for k in v if k.startswith('risk_')}
                    extra = {k: x for k, x in v.items()
                             if k not in _RESULT and k not in risk
                             and k not in ('index', 'rule')}
                    rows.append(dict(fov, vote=v['rule'],
                                     **{k: v[k] for k in _RESULT},
                                     extra_json=json.dumps(extra), **stats, **risk))
    if skipped:
        print(f'  {skipped} FoV x method with an error in stage 1 left out')
    return rows


BAR = '=' * 78

#: Keyed by "head recipe" (`<head_name>+<reduction>`, or 'baseline' for a
#: bare KnnEstMpp with neither) so `mlp` is the SAME colour in every
#: encoder's subplot -- colour follows the entity, not its position in
#: whichever list happened to be drawn that run. LOSS-INDEPENDENT: see
#: `head_recipe_of`'s own docstring for why loss changes linestyle instead
#: of colour.
#:
#: Keyed by `training/ClassifyingRoutingHead/Runtime.HEAD_CHOICES`'s registered names
#: (a CLASSIFIER_RECIPES name not keyed here draws in the secondary ink). Its
#: own table: `training/ClassifyingRoutingHead/cli/evaluate.py` colours by classifier
#: and marks the reduction by shape (`CLASSIFIER_COLORS`, `head_style`), so
#: the two reports do not use the same colour for the same head.
HEAD_RECIPE_COLORS = {
    'baseline':              '#2a78d6',   # raw KnnEstMpp
    'linear+fixed':          '#eb6834',
    'attn_linear+attn':      '#1baf7a',
    'mlp+fixed':             '#eda100',
    'mlp_deep+fixed':        '#e87ba4',
    'mlp_wide+fixed':        '#008300',
    'mlp_deep_wide+fixed':   '#4a3aa7',
    'mlp_deep_residual+fixed': '#e34948',
    'mlp_narrow+fixed':      '#17a2b8',
    'arcface+fixed':         '#8b5a2b',
}

#: Loss changes the LINE, not the colour (`head_recipe_of`'s own docstring)
#: -- 'bal' solid (every checkpoint before `--loss` existed was this,
#: unlabelled), 'ord_a' dashed, 'ord_b' dotted. `.get(loss, '-')` covers a
#: blank `loss` (KnnEstMpp rows, which have no classifier/reduction either)
#: the same way solid covers 'bal'.
LOSS_LINESTYLES = {'bal': '-', 'ord_a': '--', 'ord_b': ':'}
_INK = '#0b0b0b'
_INK_SECONDARY = '#52514e'
_GRIDLINE = '#e1e0d9'
_AXIS = '#c3c2b7'
_SURFACE = '#fcfcfb'


def head_recipe_of(classifier: str, reduction: str) -> str:
    """`<classifier>+<reduction>` (e.g. `'arcface+fixed'`), 'baseline' for a
    bare KnnEstMpp (blank classifier) -- LOSS-INDEPENDENT on purpose:
    colour follows which HEAD this is, the same entity
    `HEAD_RECIPE_COLORS` keys on; which --loss trained it changes the
    LINESTYLE instead (`plot_dataset`'s own `LOSS_LINESTYLES`), not the
    colour -- giving every (head, loss) pair its own colour would need
    9 heads x 3 losses = 27 slots, well past any categorical palette this
    codebase's dataviz skill validates. Takes the two fields directly
    (`cross_slide_rung`'s own `classifier`/`reduction` columns), not a
    composed `method` string, so it cannot accidentally include `loss` by
    parsing past it."""
    return f'{classifier}+{reduction}' if classifier else 'baseline'


def encoder_of(method: str) -> str:
    """The encoder a method label runs on: `knn:uni2` and `uni2+mlp+fixed`
    are both uni2, so the figure draws them in one subplot."""
    return method.split("+", 1)[0].split(":")[-1]

#: The rung vocabulary every row's `rung` column is drawn from -- DsLadder's
#: own default, duplicated here (not imported) because DsLadder.py has no
#: torch and this file has none either, and importing it just for one tuple
#: would be the first import that could ever pull torch in behind it.
RUNGS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


# ── csv access ────────────────────────────────────────────────────────────────

def cell(row: dict, key: str):
    v = row.get(key, '')
    return None if v in ('', 'None', None) else v


def num(row: dict, key: str):
    v = cell(row, key)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def boolean(row: dict, key: str):
    v = cell(row, key)
    return None if v is None else v.strip().lower() in ('true', '1', 'yes')


def method_of(row: dict) -> str:
    """One string identifying which method+config a row belongs to.

    `classifier`/`reduction`/`loss`/`weights` are blank for `KnnEstMpp` (see
    the module docstring), so the label collapses to just the encoder there
    -- the same rule `_paths.encoder_tag` uses for an encoder with no head.

    `loss` is appended ONLY when it is present AND not 'bal', so a bal
    run keeps the bare label (no '+bal' suffix appearing
    everywhere) -- same reasoning `bench_stage1_mpp.py`'s own
    `_prototype_weight_filename`-style filenames only tag a NON-default
    loss. Without this, a bal- and an ord_a-trained checkpoint of the same
    encoder+head+reduction would collapse into one label and get averaged
    together -- the same bug `classifier` switching from the class name to
    `head_name` already fixed once for the mlp variants (see the module
    docstring's csv columns note).
    """
    # KnnEstMpp has no head; 'knn:' says which method it is, so the label
    # does not read like the encoder itself
    encoder = cell(row, 'encoder') or '?'
    parts = [f'knn:{encoder}' if kind_of(row) == 'knn' else encoder]
    for k in ('classifier', 'reduction'):
        v = cell(row, k)
        if v:
            parts.append(v)
    loss = cell(row, 'loss')
    if loss and loss != 'bal':
        parts.append(loss)
    # `read_level`, the same rule one column on: only off the
    # default, so a pyramid-trained head keeps its label
    read = cell(row, 'read_level')
    if read and read != 'pyramid':
        parts.append(read)
    label = '+'.join(parts)
    # `vote`: a classifier or prototype row is one FoVVote rule
    # over the method's patch probabilities, and two rules of one checkpoint
    # are two methods here. '@' rather than '+', so `base_method_of` can
    # strip it without parsing the head recipe.
    vote = cell(row, 'vote')
    return f'{label}@{vote}' if vote else label


def base_method_of(row: dict) -> str:
    """`method_of` without the vote rule: the checkpoint, every vote of it."""
    return method_of(row).split('@', 1)[0]


def kind_of(row: dict) -> str:
    """knn / classifier / prototype / classic. A csv from before the `kind`
    column has only knn (blank classifier) and classifier rows."""
    return cell(row, 'kind') or ('classifier' if cell(row, 'classifier') else 'knn')


#: The rule both voting estimators use by default (`cfg.vote`), i.e. what the
#: pipeline would run. Views 1-2 and the figure show only this rule.
DEFAULT_VOTE = 'mean_probability'


def is_default_vote(row: dict) -> bool:
    return cell(row, 'vote') in (None, DEFAULT_VOTE)


def nearest_rung(ds: float, rungs=RUNGS) -> float:
    """Which rung `ds` is closest to, in LOG space -- pyramid scales are
    geometric, so a linear nearest would call 1-vs-4 nearer than 16-vs-64
    though both are one rung apart."""
    return min(rungs, key=lambda r: abs(math.log(ds) - math.log(r)))


def is_correct(row: dict) -> bool:
    """The method's OWN estimate lands on the true rung -- see the module
    docstring for why `estimated_ds`, not `chosen_ds`."""
    ds = num(row, 'estimated_ds')
    rung = num(row, 'rung')
    return ds is not None and rung is not None and nearest_rung(ds) == rung


def log2_rung_error(row: dict) -> float:
    """How many octaves the estimate's rung is from the true one -- 0 when
    right, 1 one rung off on the 2x ladder, 2 for ds 4 called 16."""
    ds, rung = num(row, 'estimated_ds'), num(row, 'rung')
    if ds is None or rung is None:
        return float('nan')
    return abs(math.log2(nearest_rung(ds)) - math.log2(rung))


def mpp_error_relative(row: dict):
    est, gt = num(row, 'estimated_mpp'), num(row, 'gt_mpp')
    return None if not est or not gt else abs(est - gt) / gt


def pctl(values: list, p: int) -> float:
    v = sorted(x for x in values if x is not None)
    if not v:
        return float('nan')
    return v[min(len(v) - 1, int(len(v) * p / 100))]


# ── the three views ──────────────────────────────────────────────────────────

def score_group(rows: list) -> dict:
    """One group's numbers -- shared by all three views, so a slide-level row
    and a cross-slide row read the same way."""
    correct = [is_correct(r) for r in rows]
    err = [e for e in (mpp_error_relative(r) for r in rows) if e is not None]
    native = [boolean(r, 'native') for r in rows]
    out = dict(n=len(rows),
              level_accuracy=sum(correct) / len(rows) if rows else float('nan'),
              mpp_error_relative_p50=pctl(err, 50),
              log2_rung_mae=(sum(log2_rung_error(r) for r in rows) / len(rows)
                             if rows else float('nan')))
    for label, keep in (('native', [n is True for n in native]),
                        ('resampled', [n is False for n in native])):
        sub = [c for c, k in zip(correct, keep) if k]
        out[f'n_{label}'] = len(sub)
        out[f'level_accuracy_{label}'] = (sum(sub) / len(sub) if sub
                                          else float('nan'))
    return out


def per_slide_rung(rows: list) -> list:
    """View 1: one row per (dataset, wsi_name, rung, method)."""
    g = collections.defaultdict(list)
    for r in rows:
        g[(cell(r, 'dataset'), cell(r, 'wsi_name'), num(r, 'rung'),
           method_of(r))].append(r)
    out = []
    for (dataset, wsi_name, rung, method), grp in sorted(
            g.items(), key=lambda kv: (kv[0][0] or '', kv[0][1] or '', kv[0][2] or 0)):
        out.append(dict(dataset=dataset, wsi_name=wsi_name, rung=rung,
                        method=method, **score_group(grp)))
    return out


def cross_slide_rung(rows: list) -> list:
    """View 2: one row per (dataset, rung, method), pooled across every slide
    that contributed one. `n` says how many shots that actually is -- read it
    before trusting the accuracy next to it, especially under
    native levels, where a rung some slides lack natively rests on fewer
    slides' worth of shots than one that every slide has.

    `classifier`/`reduction`/`loss` are carried through
    alongside the composed `method` string, read off `grp[0]` since every
    row a `method` groups together shares the same three values by
    construction (`method_of` is a deterministic function of them) --
    `plot_dataset` needs them SEPARATELY (colour follows classifier+
    reduction, loss only changes the linestyle), not re-parsed back out of
    the composed string.
    """
    g = collections.defaultdict(list)
    for r in rows:
        g[(cell(r, 'dataset'), num(r, 'rung'), method_of(r))].append(r)
    out = []
    for (dataset, rung, method), grp in sorted(
            g.items(), key=lambda kv: (kv[0][0] or '', kv[0][1] or 0)):
        n_slides = len({cell(r, 'wsi_name') for r in grp})
        out.append(dict(dataset=dataset, rung=rung, method=method,
                        classifier=cell(grp[0], 'classifier') or '',
                        reduction=cell(grp[0], 'reduction') or '',
                        loss=cell(grp[0], 'loss') or '',
                        read_level=cell(grp[0], 'read_level') or '',
                        n_slides=n_slides, **score_group(grp)))
    return out


def overall(rows: list) -> list:
    """View 3: one row per (dataset, method) -- the MEAN of that method's own
    per-rung accuracies (view 2, same dataset), not the pooled accuracy over
    every shot. See the module docstring for why."""
    by_rung = cross_slide_rung(rows)
    g = collections.defaultdict(list)
    for r in by_rung:
        g[(r['dataset'], r['method'])].append(r)
    ci = bootstrap_ci(rows)

    def mean_of(grp, key):
        v = [r[key] for r in grp if not math.isnan(r[key])]
        return sum(v) / len(v) if v else float('nan')

    out = []
    for (dataset, method), grp in sorted(g.items()):
        lo, hi = ci.get((dataset, method), (float('nan'), float('nan')))
        out.append(dict(
            dataset=dataset, method=method, n_rungs=len(grp),
            mean_level_accuracy=mean_of(grp, 'level_accuracy'),
            ci95_lo=lo, ci95_hi=hi,
            mean_log2_rung_mae=mean_of(grp, 'log2_rung_mae'),
            mean_mpp_error_relative_p50=mean_of(grp, 'mpp_error_relative_p50')))
    return out


#: Bootstrap resamples, and the seed, for every interval this file reports.
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 0


def bootstrap_ci(rows: list) -> dict:
    """{(dataset, method): (lo, hi)} -- 95% interval of the equal-rung
    accuracy, resampling FoVs with replacement INSIDE each rung so every
    resample keeps the per-rung weighting the point estimate has. FoV is the
    unit, as FoV_Vote.md asks: one FoV's rows of one method are one draw."""
    import numpy as np                                            # noqa: PLC0415
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        g[(cell(r, 'dataset'), method_of(r))][num(r, 'rung')].append(
            1.0 if is_correct(r) else 0.0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {}
    for key, by_rung in g.items():
        per_rung = []
        for vals in by_rung.values():
            v = np.asarray(vals)
            idx = rng.integers(0, len(v), size=(BOOTSTRAP_N, len(v)))
            per_rung.append(v[idx].mean(axis=1))
        means = np.mean(np.stack(per_rung), axis=0)
        out[key] = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    return out


# ── printing ─────────────────────────────────────────────────────────────────

def _col_width(rows: list, key: str, minimum: int, cap: int = None) -> int:
    """`max(actual string length across rows, minimum) + 2` -- computed from
    THIS call's own rows rather than a hardcoded guess, so a label longer
    than any fixed width (`convnext_v2+LinearHead+attn` is 28 chars,
    `ki67_with_photo` is 16) still gets a gap after it instead of running
    straight into the next column -- the exact misalignment a fixed-width
    `:16s` produces the moment one value exceeds it (Python pads a SHORTER
    string to width, but does not truncate a LONGER one, so every column
    after it drifts by however much that one value overshot).

    `cap`, when given, bounds the width so one absurdly long value (there is
    no enforced limit on a WSI name) cannot blow up every row's width --
    values longer than `cap` are the caller's to truncate, this function
    only sizes the column.
    """
    longest = max((len(str(r.get(key) or '')) for r in rows), default=0)
    width = max(minimum, longest) + 2
    return min(width, cap) if cap else width


def print_per_slide_rung(rows: list) -> None:
    print(BAR)
    print('1. PER SLIDE, PER RUNG')
    print(BAR)
    dw = _col_width(rows, 'dataset', 10)
    ww = _col_width(rows, 'wsi_name', 12, cap=24)
    mw = _col_width(rows, 'method', 10)
    print(f'  {"dataset":{dw}s}{"wsi_name":{ww}s}{"rung":>6s}{"method":{mw}s}'
         f'{"n":>5s}{"acc":>7s}{"mpp_err_p50":>12s}{"n_nat":>7s}{"acc_nat":>9s}'
         f'{"n_res":>7s}{"acc_res":>9s}')
    for r in rows:
        print(f'  {r["dataset"] or "":{dw}s}'
             f'{(r["wsi_name"] or "")[:ww - 2]:{ww}s}'
             f'{r["rung"]:>6g}{r["method"]:{mw}s}{r["n"]:>5d}'
             f'{r["level_accuracy"]:>7.2f}{r["mpp_error_relative_p50"]:>12.3f}'
             f'{r["n_native"]:>7d}{r["level_accuracy_native"]:>9.2f}'
             f'{r["n_resampled"]:>7d}{r["level_accuracy_resampled"]:>9.2f}')


def print_cross_slide_rung(rows: list) -> None:
    print('\n' + BAR)
    print('2. CROSS-SLIDE, PER RUNG   (n_slides: how many slides contributed)')
    print(BAR)
    dw = _col_width(rows, 'dataset', 10)
    mw = _col_width(rows, 'method', 10)
    print(f'  {"dataset":{dw}s}{"rung":>6s}{"method":{mw}s}{"n_slides":>9s}'
         f'{"n":>6s}{"acc":>7s}{"mpp_err_p50":>12s}')
    for r in rows:
        print(f'  {r["dataset"] or "":{dw}s}{r["rung"]:>6g}{r["method"]:{mw}s}'
             f'{r["n_slides"]:>9d}{r["n"]:>6d}{r["level_accuracy"]:>7.2f}'
             f'{r["mpp_error_relative_p50"]:>12.3f}')


def print_overall(rows: list) -> None:
    print('\n' + BAR)
    print('3. OVERALL   (mean of the per-rung accuracies -- every rung weighted '
         'equally)')
    print(BAR)
    dw = _col_width(rows, 'dataset', 10)
    mw = _col_width(rows, 'method', 10)
    print(f'  {"dataset":{dw}s}{"method":{mw}s}{"n_rungs":>8s}'
         f'{"mean_acc":>10s}{"ci95":>16s}{"log2_mae":>10s}{"mean_mpp_err_p50":>18s}')
    for r in rows:
        print(f'  {r["dataset"] or "":{dw}s}{r["method"]:{mw}s}{r["n_rungs"]:>8d}'
             f'{r["mean_level_accuracy"]:>10.3f}'
             f'{"[%.3f, %.3f]" % (r["ci95_lo"], r["ci95_hi"]):>16s}'
             f'{r["mean_log2_rung_mae"]:>10.3f}{r["mean_mpp_error_relative_p50"]:>18.3f}')


# ── 5. vote diagnosis ────────────────────────────────────────────────────────
#
# Every classifier/prototype checkpoint is scored under every FoVVote rule
# from ONE forward pass, so two rules of one checkpoint saw identical
# probabilities and differ only in how they aggregate them. Every comparison
# below is paired that way: per FoV, per checkpoint, rule against rule.
#
# The question is not only "which rule wins" but "which rule suits which kind
# of FoV", so the FoV is described by its per-patch distribution (`fov_*`,
# computed by fov_stats from the probs table) and every rule is scored inside
# strata of it.

#: (column, [(label, lo, hi)]) -- lo inclusive, hi exclusive, except a bin
#: whose lo == hi, which matches that value exactly.
STRATA = (
    ('fov_agree_frac', [('<0.5', 0.0, 0.5), ('0.5-0.75', 0.5, 0.75),
                        ('0.75-1', 0.75, 1.0), ('=1', 1.0, 1.0)]),
    ('fov_pooled_entropy', [('<0.25', 0.0, 0.25), ('0.25-0.5', 0.25, 0.5),
                            ('0.5-0.75', 0.5, 0.75), ('>=0.75', 0.75, 9.0)]),
    ('fov_argmax_log2_spread', [('=0', 0.0, 0.0), ('0-0.5', 1e-9, 0.5),
                                ('0.5-1', 0.5, 1.0), ('>=1', 1.0, 99.0)]),
    # the share of patches backing the chosen class -- FoV_Vote.md #1 asks
    # for the error rate against it
    ('risk_winner_support', [('<0.25', 0.0, 0.25), ('0.25-0.5', 0.25, 0.5),
                             ('0.5-0.75', 0.5, 0.75), ('>=0.75', 0.75, 9.0)]),
    ('fov_pooled_margin', [('<0.1', 0.0, 0.1), ('0.1-0.3', 0.1, 0.3),
                           ('0.3-0.6', 0.3, 0.6), ('>=0.6', 0.6, 9.0)]),
)

#: The distribution columns summarised per rung by `fov_distribution`.
FOV_COLUMNS = ('fov_agree_frac', 'fov_n_distinct', 'fov_patch_entropy',
               'fov_pooled_entropy', 'fov_pooled_margin',
               'fov_argmax_log2_spread', 'fov_gt_prob', 'fov_gt_patch_frac')


def _bin_of(value, bins):
    if value is None:
        return None
    for label, lo, hi in bins:
        if (lo == hi and value == lo) or (lo != hi and lo <= value < hi):
            return label
    return None


def fov_key(row: dict) -> tuple:
    return (cell(row, 'dataset'), cell(row, 'wsi_name'), cell(row, 'x'),
            cell(row, 'y'), num(row, 'gt_ds'))


def _paired(rows: list) -> dict:
    """{(dataset, kind, base_method, fov_key): {vote: row}} over voting rows."""
    out = collections.defaultdict(dict)
    for r in rows:
        if cell(r, 'vote'):
            out[(cell(r, 'dataset'), kind_of(r), base_method_of(r),
                 fov_key(r))][cell(r, 'vote')] = r
    return out


def vote_accuracy(rows: list) -> list:
    """(dataset, kind, base_method, vote) -> the view-3 number (mean of the
    per-rung accuracies) for that rule, plus `rank` among that checkpoint's
    rules (1 = best)."""
    voting = [r for r in rows if cell(r, 'vote')]
    kinds = {base_method_of(r): kind_of(r) for r in voting}
    out = []
    for o in overall(voting):
        base, vote = o['method'].split('@', 1)
        out.append(dict(dataset=o['dataset'], kind=kinds[base], method=base,
                        vote=vote, n_rungs=o['n_rungs'],
                        mean_level_accuracy=o['mean_level_accuracy']))
    groups = collections.defaultdict(list)
    for r in out:
        groups[(r['dataset'], r['method'])].append(r)
    # Competition ranking: 1 + how many rules are STRICTLY better, so rules
    # that tie share a rank and all count as best. Ranking by position in a
    # sorted list split ties by name, and the alphabetically first rule was
    # reported best on every checkpoint where all six scored the same.
    for grp in groups.values():
        for r in grp:
            r['rank'] = 1 + sum(o['mean_level_accuracy'] > r['mean_level_accuracy']
                                for o in grp)
    return out


def vote_summary(acc_rows: list) -> list:
    """(dataset, kind, vote): mean accuracy over that kind's checkpoints,
    how many checkpoints the rule is best on, and its mean rank."""
    g = collections.defaultdict(list)
    for r in acc_rows:
        g[(r['dataset'], r['kind'], r['vote'])].append(r)
    out = []
    for (dataset, kind, vote), grp in sorted(g.items()):
        accs = [r['mean_level_accuracy'] for r in grp
                if not math.isnan(r['mean_level_accuracy'])]
        out.append(dict(dataset=dataset, kind=kind, vote=vote,
                        n_methods=len(grp),
                        mean_acc=sum(accs) / len(accs) if accs else float('nan'),
                        n_best=sum(r['rank'] == 1 for r in grp),
                        mean_rank=sum(r['rank'] for r in grp) / len(grp)))
    return out


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test on the discordant pairs (McNemar exact):
    the chance of a split at least this lopsided if both rules were equally
    good. Ties (both right, both wrong) carry no information and are out."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def vote_pairwise(paired: dict, by_method: bool = False) -> list:
    """(dataset, kind, a, b) -- or per checkpoint when `by_method` -- paired
    on the FoV: both right, only a, only b, both wrong; how often the two
    answer differently at all; and the sign test on only-a vs only-b."""
    g = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for (dataset, kind, base, _), by_vote in paired.items():
        votes = sorted(by_vote)
        for i, a in enumerate(votes):
            for b in votes[i + 1:]:
                ra, rb = by_vote[a], by_vote[b]
                ca, cb = is_correct(ra), is_correct(rb)
                s = g[(dataset, kind, base if by_method else '', a, b)]
                s[0 if ca and cb else 1 if ca else 2 if cb else 3] += 1
                s[4] += (nearest_rung(num(ra, 'estimated_ds'))
                         != nearest_rung(num(rb, 'estimated_ds')))
    out = []
    for (d, k, m, a, b), (both, a_only, b_only, neither, differ) in sorted(g.items()):
        n = both + a_only + b_only + neither
        out.append(dict(dataset=d, kind=k, method=m, vote_a=a, vote_b=b, n_fov=n,
                        both_right=both, a_only=a_only, b_only=b_only,
                        both_wrong=neither, n_disagree=differ,
                        disagree_rate=differ / n if n else float('nan'),
                        sign_p=sign_test_p(a_only, b_only)))
    return out


def vote_oracle(paired: dict) -> list:
    """(dataset, kind, base_method): the share of FoVs SOME rule gets right,
    EVERY rule gets right, and the best single rule's share -- the room any
    choice of rule has, and how much of it the best one takes."""
    g = collections.defaultdict(list)
    for (dataset, kind, base, _), by_vote in paired.items():
        g[(dataset, kind, base)].append({v: is_correct(r) for v, r in by_vote.items()})
    out = []
    for (dataset, kind, base), fovs in sorted(g.items()):
        votes = sorted({v for f in fovs for v in f})
        per_vote = {v: sum(f.get(v, False) for f in fovs) / len(fovs) for v in votes}
        best = max(per_vote, key=per_vote.get)
        out.append(dict(dataset=dataset, kind=kind, method=base, n_fov=len(fovs),
                        any_right=sum(any(f.values()) for f in fovs) / len(fovs),
                        all_right=sum(all(f.values()) for f in fovs) / len(fovs),
                        best_vote=best, best_vote_acc=per_vote[best]))
    return out


def vote_strata(rows: list, by_method: bool) -> list:
    """Accuracy of every rule inside each stratum of each `STRATA` column --
    pooled per (dataset, kind), or per checkpoint when `by_method`. FoVs whose
    true ds is no class of the method (`fov_gt_reachable` false) are their own
    stratum: no rule can be right there, and mixing them in would lower
    every rule equally and hide nothing but the signal."""
    g = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        vote = cell(r, 'vote')
        if not vote:
            continue
        who = base_method_of(r) if by_method else ''
        head = (cell(r, 'dataset'), kind_of(r), who)
        if boolean(r, 'fov_gt_reachable') is False:
            s = g[head + ('fov_gt_reachable', 'false', vote)]
            s[0] += 1
            s[1] += is_correct(r)
            continue
        for column, bins in STRATA:
            label = _bin_of(num(r, column), bins)
            if label is None:
                continue
            s = g[head + (column, label, vote)]
            s[0] += 1
            s[1] += is_correct(r)
    order = {c: [b[0] for b in bins] for c, bins in STRATA}
    order['fov_gt_reachable'] = ['false']
    out = [dict(dataset=d, kind=k, method=m, stat=c, bin=b, vote=v, n=n,
                accuracy=right / n if n else float('nan'))
           for (d, k, m, c, b, v), (n, right) in g.items()]
    return sorted(out, key=lambda r: (r['dataset'] or '', r['kind'], r['method'],
                                      r['stat'], order[r['stat']].index(r['bin']),
                                      r['vote']))


def fov_distribution(rows: list) -> list:
    """(dataset, kind, rung, outcome): the median of every `FOV_COLUMNS`
    column, outcome being right/wrong under DEFAULT_VOTE -- what a FoV the
    method gets wrong looks like next to one it gets right, scale by scale.
    One row per FoV and checkpoint (the distribution does not depend on the
    rule)."""
    g = collections.defaultdict(list)
    for r in rows:
        if cell(r, 'vote') != DEFAULT_VOTE:
            continue
        outcome = 'right' if is_correct(r) else 'wrong'
        g[(cell(r, 'dataset'), kind_of(r), num(r, 'rung'), outcome)].append(r)
    out = []
    for (dataset, kind, rung, outcome), grp in sorted(
            g.items(), key=lambda kv: (kv[0][0] or '', kv[0][1], kv[0][2] or 0, kv[0][3])):
        row = dict(dataset=dataset, kind=kind, rung=rung, outcome=outcome, n=len(grp))
        for c in FOV_COLUMNS:
            row[f'{c}_p50'] = pctl([num(r, c) for r in grp], 50)
        row['gt_reachable_share'] = (sum(boolean(r, 'fov_gt_reachable') is True
                                         for r in grp) / len(grp))
        out.append(row)
    return out


def write_rows(rows: list, path: str) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'  {path}  ({len(rows)} rows)')


def print_vote_diagnosis(summary, pairwise, oracle, strata, distribution) -> None:
    print('\n' + BAR)
    print('5. VOTE RULES   (every classifier / prototype checkpoint, one forward '
          'pass, every rule)')
    print(BAR)
    print('\n  5a. per (dataset, kind, rule): mean accuracy over checkpoints, '
          'checkpoints where it is best, mean rank')
    vw = _col_width(summary, 'vote', 10)
    for r in summary:
        print(f'  {r["dataset"]:18s}{r["kind"]:11s}{r["vote"]:{vw}s}'
              f'n={r["n_methods"]:<4d}acc {r["mean_acc"]:.3f}   best on '
              f'{r["n_best"]:>3d}   mean rank {r["mean_rank"]:.2f}')
    print('\n  5b. paired, per FoV: only a right / only b right, how often they '
          'answer differently, sign test   (pooled per kind; per checkpoint in the csv)')
    for p in pairwise:
        if not p['n_disagree']:
            continue
        print(f'  {p["dataset"]:18s}{p["kind"]:11s}{p["vote_a"]:>24s} vs '
              f'{p["vote_b"]:<24s} only {p["a_only"]:>5d} / {p["b_only"]:<5d}  '
              f'differ {p["disagree_rate"]:6.1%}  p={p["sign_p"]:.3g}')
    print('\n  5c. headroom per checkpoint: some rule right / every rule right / '
          'best single rule')
    mw = _col_width(oracle, 'method', 10, cap=60)
    for o in oracle:
        print(f'  {o["dataset"]:18s}{o["method"][:mw - 2]:{mw}s}any {o["any_right"]:.3f}  '
              f'all {o["all_right"]:.3f}  best {o["best_vote"]} {o["best_vote_acc"]:.3f}')
    print('\n  5d. accuracy per rule inside strata of the FoV distribution  '
          '(pooled per kind; per checkpoint in the csv)')
    cells = collections.defaultdict(dict)
    for s in strata:
        cells[(s['dataset'], s['kind'], s['stat'], s['bin'])][s['vote']] = s
    votes = sorted({s['vote'] for s in strata})
    print(f'  {"":18s}{"":11s}{"stat":24s}{"bin":10s}{"n":>6s}'
          + ''.join(f'{v[:12]:>13s}' for v in votes))
    for (dataset, kind, stat, b), by_vote in cells.items():
        n = max(s['n'] for s in by_vote.values())
        print(f'  {dataset:18s}{kind:11s}{stat:24s}{b:10s}{n:>6d}'
              + ''.join(f'{by_vote[v]["accuracy"]:>13.3f}' if v in by_vote
                        else f'{"":>13s}' for v in votes))
    print('\n  5e. FoV distribution, right vs wrong under '
          f'{DEFAULT_VOTE}  (medians; per rung)')
    for d in distribution:
        print(f'  {d["dataset"]:18s}{d["kind"]:11s}rung {d["rung"]:>4g} '
              f'{d["outcome"]:6s}n={d["n"]:<6d}agree {d["fov_agree_frac_p50"]:.2f}  '
              f'pooled_H {d["fov_pooled_entropy_p50"]:.2f}  '
              f'margin {d["fov_pooled_margin_p50"]:.2f}  '
              f'spread {d["fov_argmax_log2_spread_p50"]:.2f}  '
              f'gt_p {d["fov_gt_prob_p50"]:.2f}  reachable {d["gt_reachable_share"]:.2f}')


# ── 6. risk flags (FoV_Vote.md, 危險分布) ─────────────────────────────────────
#
# One flag per rule, the doc's own definition, over the raw `risk_*` columns
# FoVVote.diagnose wrote. Every threshold is FITTED ON VAL and read back for
# test: the doc forbids choosing one after looking at test. The quantile each
# is fitted at is fixed here, before either split is scored -- moving it
# after seeing test is the same mistake one step removed.

def rule_of(row: dict) -> str:
    """The FoVVote rule of a row's vote label: 'patch_class_median:upper' ->
    'patch_class_median'."""
    return (cell(row, 'vote') or '').split(':', 1)[0]


#: tau name -> (rule, column, quantile in percent, fit on |value|)
THRESHOLD_SPECS = {
    'hm_margin_majority_low': ('hard_majority', 'risk_margin_majority_p50', 25, False),
    'hm_margin_dissent_high': ('hard_majority', 'risk_margin_dissent_p50', 75, False),
    'pcm_top1_low':           ('patch_class_median', 'risk_top1_p50', 25, False),
    'mlr_support_low':        ('median_log_rung', 'risk_winner_prob', 25, False),
    'slp_veto':               ('sum_log_probability', 'risk_min_p_winner', 10, False),
    'qw_corr_high':           ('quality_weighted', 'risk_corr_w_e', 75, True),
    'qw_ess_low':             ('quality_weighted', 'risk_ess_frac', 25, False),
}


def _finite(v) -> bool:
    return v is not None and not math.isnan(v)


def estimators_by_kind(rows: list) -> dict:
    """`{kind: sorted estimator ids}` -- which estimators a file's taus are
    taken over. A threshold is per kind, so it is only valid for the same set
    of estimators: a retrained checkpoint under the same label is another."""
    out = collections.defaultdict(set)
    for r in rows:
        if cell(r, 'vote'):
            out[kind_of(r)].add(cell(r, 'estimator_id') or '')
    return {k: sorted(v) for k, v in sorted(out.items())}


def check_estimators(fitted: dict, rows: list, path: str) -> None:
    """Refuse taus fitted on other estimators than the ones in `rows`."""
    stored, now = fitted.get('estimators'), estimators_by_kind(rows)
    if stored is None:
        sys.exit(f'{path} records no estimator ids; refit it on the val run '
                 f'(--fit-thresholds)')
    bad = [f'{k}: val {stored.get(k)}, this file {now.get(k)}'
           for k in sorted(set(stored) | set(now)) if stored.get(k) != now.get(k)]
    if bad:
        sys.exit(f'{path} was fitted on other estimators than this file holds -- '
                 + '; '.join(bad) + '. Rerun the val set with the same '
                 'estimators and refit')


def is_val_set(datasets: str) -> bool:
    """Is a row's `set` (the --datasets ids, space-separated) one a threshold may be fitted on: every id a val
    set, named `...-val` (`@legacy-val`, `@pure-val`). An older table's `split` column says `val`."""
    ids: List[str] = datasets.split()
    return bool(ids) and all(i == 'val' or i.endswith('-val') for i in ids)


def set_name(datasets: Sequence[str]) -> str:
    """The --datasets ids as one file-name word: `@legacy-val` -> `legacy-val`, joined by `+`."""
    return '+'.join(d.lstrip('@').replace('/', '_') for d in datasets)


def val_set_of(datasets: Sequence[str]) -> Optional[List[str]]:
    """The val set a test set's thresholds were fitted on: each `...-test` id as its `...-val`
    (`@legacy-test` -> `@legacy-val`); None when an id is not a `-test` one."""
    if not datasets or not all(d.endswith('-test') for d in datasets):
        return None
    return [d[:-len('-test')] + '-val' for d in datasets]


def fit_thresholds(rows: list, source: str) -> dict:
    """Every tau of THRESHOLD_SPECS, per (kind, vote label) -- a prototype's
    near-uniform probabilities and a classifier's sharp ones are different
    scales, so one tau across both would flag one of them almost always.
    Refuses a file with rows of anything but a val set in it (`is_val_set`)."""
    sets = {cell(r, 'set') or cell(r, 'split') for r in rows}
    if not all(s is not None and is_val_set(s) for s in sets):
        sys.exit(f'--fit-thresholds needs a val run (bench --datasets @<name>-val); this file '
                 f'has {sorted(s or "unrecorded" for s in sets)}')
    groups = collections.defaultdict(list)
    for r in rows:
        if cell(r, 'vote'):
            groups[(kind_of(r), cell(r, 'vote'))].append(r)
    values = {}
    for (kind, label), grp in sorted(groups.items()):
        for tau, (rule, column, q, use_abs) in THRESHOLD_SPECS.items():
            if rule_of(grp[0]) != rule:
                continue
            v = [num(r, column) for r in grp]
            v = [abs(x) if use_abs else x for x in v if _finite(x)]
            if v:
                values[f'{kind}|{label}|{tau}'] = dict(value=pctl(v, q), n=len(v))
    return dict(source=os.path.abspath(source),
                estimators=estimators_by_kind(rows),
                specs={k: dict(rule=r, column=c, quantile=q, abs=a)
                       for k, (r, c, q, a) in THRESHOLD_SPECS.items()},
                values=values)


def flags_of(row: dict, taus: dict) -> dict:
    """{flag name: True/False} for this row's rule; a flag whose tau was not
    fitted is left out, never guessed."""
    rule, v = rule_of(row), (lambda c: num(row, c))
    key = f'{kind_of(row)}|{cell(row, "vote")}|'
    tau = lambda name: (taus.get(key + name) or {}).get('value')     # noqa: E731
    out = {}
    if rule == 'mean_probability':
        out['few_confident_decide'] = (v('risk_mean_not_mode') == 1
                                       and (v('risk_winner_support') or 0) < 0.5)
    elif rule == 'hard_majority':
        out['vote_tie'] = v('risk_vote_tie') == 1
        lo, hi = tau('hm_margin_majority_low'), tau('hm_margin_dissent_high')
        if lo is not None and hi is not None:
            a, b = v('risk_margin_majority_p50'), v('risk_margin_dissent_p50')
            out['weak_majority_strong_dissent'] = (_finite(a) and _finite(b)
                                                   and a < lo and b > hi)
    elif rule == 'patch_class_median':
        out['median_split'] = (v('risk_median_split') or 0) >= 1
        t = tau('pcm_top1_low')
        if t is not None:
            out['unsure_median_or_split'] = (out['median_split']
                                             or (v('risk_top1_p50') or 0) < t)
    elif rule == 'median_log_rung':
        out['unsupported_snap'] = v('risk_support_count') == 0
        t = tau('mlr_support_low')
        if t is not None:
            out['weak_or_unsupported_snap'] = (out['unsupported_snap']
                                               or (v('risk_winner_prob') or 0) < t)
    elif rule == 'sum_log_probability':
        out['loo_unstable'] = v('risk_loo_unstable') == 1
        t = tau('slp_veto')
        if t is not None:
            out['single_patch_veto'] = (out['loo_unstable']
                                        and (v('risk_min_p_winner') or 0) < t)
    elif rule == 'quality_weighted':
        out['weighting_flipped'] = v('risk_flip') == 1
        c, e = tau('qw_corr_high'), tau('qw_ess_low')
        if c is not None and e is not None:
            corr = v('risk_corr_w_e')
            out['weight_tracks_scale_or_collapses'] = (
                (_finite(corr) and abs(corr) > c) or (v('risk_ess_frac') or 1) < e)
    return out


def flag_report(rows: list, taus: dict) -> list:
    """Per (dataset, kind, rule label, flag) overall and per GT rung: n,
    prevalence, accuracy flagged / unflagged, and the error risk ratio
    (error rate flagged / unflagged) -- the doc's five numbers."""
    g = collections.defaultdict(lambda: [0, 0, 0, 0])     # n_f, right_f, n_u, right_u
    for r in rows:
        if not cell(r, 'vote'):
            continue
        right = is_correct(r)
        for name, hit in flags_of(r, taus).items():
            for rung in ('all', num(r, 'rung')):
                s = g[(cell(r, 'dataset'), kind_of(r), cell(r, 'vote'), name, rung)]
                if hit:
                    s[0] += 1
                    s[1] += right
                else:
                    s[2] += 1
                    s[3] += right
    out = []
    for (d, k, label, name, rung), (nf, rf, nu, ru) in g.items():
        acc_f = rf / nf if nf else float('nan')
        acc_u = ru / nu if nu else float('nan')
        err_u = 1 - acc_u
        out.append(dict(dataset=d, kind=k, vote=label, flag=name, rung=rung,
                        n=nf + nu, n_flagged=nf, prevalence=nf / (nf + nu),
                        acc_flagged=acc_f, acc_unflagged=acc_u,
                        risk_ratio=((1 - acc_f) / err_u) if nf and nu and err_u > 0
                        else float('nan')))
    return sorted(out, key=lambda r: (r['dataset'] or '', r['kind'], r['vote'], r['flag'],
                                      -1 if r['rung'] == 'all' else r['rung']))


def print_flags(report: list, have_taus: bool) -> None:
    print('\n' + BAR)
    print('6. RISK FLAGS   (FoV_Vote.md; pooled over checkpoints of a kind, '
          'all rungs -- per rung in the csv)')
    print(BAR)
    if not have_taus:
        print('  thresholds not loaded: only the flags that need none are shown. '
              'Fit them on a val run (--fit-thresholds), then pass --thresholds.')
    for r in report:
        if r['rung'] != 'all':
            continue
        print(f'  {r["dataset"]:18s}{r["kind"]:11s}{r["vote"]:26s}{r["flag"]:34s}'
              f'prev {r["prevalence"]:6.1%}  acc flagged {r["acc_flagged"]:.3f} '
              f'/ unflagged {r["acc_unflagged"]:.3f}  risk x{r["risk_ratio"]:.2f}')


# ── 7. risk-coverage and confusion ───────────────────────────────────────────

def risk_coverage(rows: list) -> list:
    """Per (dataset, method label) of a voting method: answer only the most
    confident share of FoVs and report the error among those. Confidence is
    `risk_winner_prob` (mean probability on the chosen class), the one score
    every rule has. AURC is the mean error over every coverage -- lower is
    better; pooled over rungs, unlike the equal-rung accuracy."""
    g = collections.defaultdict(list)
    for r in rows:
        conf = num(r, 'risk_winner_prob')
        if cell(r, 'vote') and _finite(conf):
            g[(cell(r, 'dataset'), kind_of(r), base_method_of(r),
               cell(r, 'vote'))].append((conf, not is_correct(r)))
    out = []
    for (d, k, m, label), pts in sorted(g.items()):
        pts.sort(key=lambda p: -p[0])
        errs, cum = [], 0
        for i, (_, wrong) in enumerate(pts, 1):
            cum += wrong
            errs.append(cum / i)
        row = dict(dataset=d, kind=k, method=m, vote=label, n=len(pts),
                   aurc=sum(errs) / len(errs))
        for cov in (25, 50, 75, 100):
            row[f'err_at_{cov}'] = errs[max(0, math.ceil(len(errs) * cov / 100) - 1)]
        out.append(row)
    return out


def print_risk_coverage(rc: list) -> None:
    print('\n' + BAR)
    print('7. RISK-COVERAGE   (mean over checkpoints of a kind; AURC lower is '
          'better; per checkpoint in the csv)')
    print(BAR)
    g = collections.defaultdict(list)
    for r in rc:
        g[(r['dataset'], r['kind'], r['vote'])].append(r)
    for (d, k, label), grp in sorted(g.items()):
        mean = lambda key: sum(r[key] for r in grp) / len(grp)       # noqa: E731
        print(f'  {d:18s}{k:11s}{label:26s}n={len(grp):<4d}AURC {mean("aurc"):.3f}   '
              f'error at 25/50/75/100% coverage  {mean("err_at_25"):.3f} '
              f'{mean("err_at_50"):.3f} {mean("err_at_75"):.3f} {mean("err_at_100"):.3f}')


def confusion(rows: list) -> list:
    """Per (dataset, method label): how many FoVs of true rung X were called
    rung Y. Long format, one row per non-empty cell."""
    g = collections.Counter()
    for r in rows:
        ds = num(r, 'estimated_ds')
        if ds is not None:
            g[(cell(r, 'dataset'), method_of(r), num(r, 'rung'), nearest_rung(ds))] += 1
    return [dict(dataset=d, method=m, gt_rung=gt, pred_rung=pr, n=n)
            for (d, m, gt, pr), n in sorted(g.items(), key=lambda kv: (
                kv[0][0] or '', kv[0][1], kv[0][2] or 0, kv[0][3] or 0))]


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_dataset(view2_rows: list, dataset: str, out_path) -> None:
    """One PNG: accuracy vs rung, one subplot per ENCODER (small multiples --
    faceting by encoder keeps each subplot's line count down to that
    encoder's own heads x losses, not every encoder's at once), one line per
    (head recipe, loss) pair -- colour follows the HEAD (`HEAD_RECIPE_
    COLORS`, the SAME colour for `mlp` in every subplot regardless of loss),
    linestyle follows the LOSS (`LOSS_LINESTYLES` -- solid/dashed/dotted for
    bal/ord_a/ord_b) -- see `head_recipe_of`'s own docstring for why loss
    is a second visual channel instead of a 27th colour. Never pooled
    across datasets: BRACS steps 4x per pyramid level and Ki67 steps 2x, so
    one dataset's rung axis does not mean the same magnification jump as
    the other's -- see the module docstring.

    matplotlib is imported HERE, not at module level -- this file's whole
    point (its own module docstring) is running on a bare login node with no
    conda env and no heavy deps, and the login node's system python has no
    matplotlib. A missing import here costs this one dataset's PNG, not the
    text tables above, which is why `main()` calls this only after they have
    already printed.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f'  [skip] matplotlib not available in this python -- no PNG '
             f'for {dataset} (the text tables above are unaffected; run '
             f'this under an env that has it, e.g. `conda activate '
             f'locascope`, for the plot)')
        return

    rows = [r for r in view2_rows if r['dataset'] == dataset]
    if not rows:
        return
    encoders = sorted({encoder_of(r['method']) for r in rows})

    fig, axes = plt.subplots(1, len(encoders), figsize=(4.2 * len(encoders), 3.6),
                             sharey=True, facecolor=_SURFACE)
    axes = [axes] if len(encoders) == 1 else list(axes)
    #: `{(recipe, loss): label}`, filled as lines are drawn -- the legend is
    #: built from this AFTER the loop rather than from `HEAD_RECIPE_COLORS`'
    #: own keys, since only the (recipe, loss) pairs actually PRESENT in
    #: this dataset's rows should appear in it.
    seen: dict = {}

    for ax, encoder in zip(axes, encoders):
        ax.set_facecolor(_SURFACE)
        by_line = collections.defaultdict(list)
        for r in rows:
            if encoder_of(r['method']) == encoder:
                recipe = head_recipe_of(r['classifier'], r['reduction'])
                by_line[(recipe, r['loss'], r.get('read_level') or '')].append(r)
        for (recipe, loss, read), line_rows in sorted(by_line.items()):
            line_rows.sort(key=lambda r: r['rung'])
            color = HEAD_RECIPE_COLORS.get(recipe, _INK_SECONDARY)
            # baseline (KnnEstMpp) has no loss at all -- dashed regardless;
            # every classifier row's linestyle
            # comes from ITS OWN loss instead (`head_recipe_of`'s own
            # docstring: colour is head identity, loss is the line).
            style = '--' if recipe == 'baseline' else LOSS_LINESTYLES.get(loss, '-')
            label = recipe if loss in ('', 'bal') else f'{recipe} ({loss})'
            # a head trained off the default read mode: same colour and
            # line as its pyramid twin, faded, and named in the legend
            off = read not in ('', 'pyramid')
            if off:
                label = f'{label} [{read}]'
            ax.plot([r['rung'] for r in line_rows],
                    [r['level_accuracy'] for r in line_rows],
                    color=color, linewidth=2, marker='o', markersize=8,
                    linestyle=style, label=label, alpha=0.5 if off else 1.0)
            seen[(recipe, loss, read)] = (color, style, label, 0.5 if off else 1.0)
        ax.set_xscale('log', base=2)
        ax.set_xticks(RUNGS)
        ax.set_xticklabels([f'{int(r)}' for r in RUNGS], color=_INK_SECONDARY)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(encoder, color=_INK, fontsize=11)
        ax.set_xlabel('rung (ds multiplier)', color=_INK_SECONDARY, fontsize=9)
        ax.grid(axis='y', color=_GRIDLINE, linewidth=0.8, zorder=0)
        for spine in ax.spines.values():
            spine.set_color(_AXIS)
        ax.tick_params(colors=_AXIS, labelcolor=_INK_SECONDARY)

    axes[0].set_ylabel('level accuracy', color=_INK_SECONDARY, fontsize=9)
    handles = [plt.Line2D([0], [0], color=color, linewidth=2, marker='o',
                          markersize=6, linestyle=style, label=label, alpha=alpha)
              for color, style, label, alpha in seen.values()]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 5),
              bbox_to_anchor=(0.5, 0.0), frameon=False,
              fontsize=9, labelcolor=_INK_SECONDARY)
    fig.suptitle(f'stage1_compare -- {dataset}', color=_INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.14, 1, 0.92))
    fig.savefig(out_path, dpi=130, facecolor=_SURFACE)
    plt.close(fig)
    print(f'  {out_path}')


# ── main ─────────────────────────────────────────────────────────────────────

# ── figures: each one says what it shows in its title ────────────────────────
#
# The tables above are for looking things up; these are for reading. Every figure has one question,
# and its title is the answer, written from the numbers (`_best_line`), so a figure can be read
# without the caption. matplotlib is imported inside `make_figures`, as `plot_dataset` does.

#: One colour per kind of method, the same in every figure (Okabe-Ito).
KIND_COLOURS = {'classifier': '#0072B2', 'prototype': '#D55E00', 'knn': '#009E73', 'classic': '#8A8A8A'}
RIGHT_COLOUR = '#009E73'
WRONG_COLOUR = '#D55E00'
#: levels off, most finer first: the estimate is too fine (small ds) ... right ... too coarse
LEVEL_ORDER = ('<=-2', '-1', 'right', '+1', '>=+2')
LEVEL_COLOURS = {'<=-2': '#B35806', '-1': '#E69F00', 'right': '#009E73', '+1': '#56B4E9', '>=+2': '#0B5394'}
LEVEL_LABELS = {'<=-2': '2+ levels too fine', '-1': '1 level too fine', 'right': 'right',
                '+1': '1 level too coarse', '>=+2': '2+ levels too coarse'}


def short_name(method: str) -> str:
    """A label that fits on a figure: the vote rule left off, the encoder not repeated, no `best`. A checkpoint
    named in segments (`<data>--<encoder>--...--best`, `Checkpoints.weights_name`) also loses its data segment
    and keeps its own `_`."""
    parts = method.split('@', 1)[0].split('+')
    if len(parts) >= 2 and '--' in parts[1]:
        segments: List[str] = parts[1].split('--')[1:]
        kept: List[str] = [s for i, s in enumerate(segments)
                           if not (i == 0 and s == parts[0]) and not (i == len(segments) - 1 and s == 'best')]
        return ' '.join([parts[0]] + kept + parts[2:])[:60]
    if len(parts) >= 2 and parts[1].startswith(parts[0] + '_'):
        parts[1] = parts[1][len(parts[0]) + 1:]
    return ' '.join(parts).replace('_best', '').replace('_', ' ')[:60]


def pick_methods(view3: list, kinds: dict, dataset: str, top: int) -> list:
    """The methods one figure draws, best first: the `top` voting checkpoints by their accuracy under the
    default rule (what the pipeline would run), then every knn and classic row (the baselines, always
    there). Each is `dict(base, kind, acc, lo, hi, best_acc, best_rule, mae)`; `best_*` is the best rule
    of that checkpoint, which is a choice made on the same FoVs and so a little optimistic."""
    by_base = collections.defaultdict(list)
    for o in view3:
        if o['dataset'] == dataset:
            by_base[o['method'].split('@', 1)[0]].append(o)
    out = []
    for base, grp in by_base.items():
        kind = kinds.get(base, 'knn')
        default = [o for o in grp if o['method'].endswith('@' + DEFAULT_VOTE)] or grp
        d = default[0]
        best = max(grp, key=lambda o: o['mean_level_accuracy'] if not math.isnan(o['mean_level_accuracy']) else -1)
        out.append(dict(base=base, kind=kind, acc=d['mean_level_accuracy'], lo=d['ci95_lo'], hi=d['ci95_hi'],
                        best_acc=best['mean_level_accuracy'],
                        best_rule=best['method'].split('@', 1)[1] if '@' in best['method'] else '',
                        mae=d['mean_log2_rung_mae']))
    out.sort(key=lambda m: -(m['acc'] if not math.isnan(m['acc']) else -1))
    voting = [m for m in out if m['kind'] in ('classifier', 'prototype')][:top]
    baselines = [m for m in out if m['kind'] not in ('classifier', 'prototype')]
    return sorted(voting + baselines, key=lambda m: -(m['acc'] if not math.isnan(m['acc']) else -1))


def pick_shared(view3: list, kinds: dict, datasets: list, top: int) -> dict:
    """`pick_methods` for several datasets, but with ONE set of methods: the `top` voting checkpoints by their
    mean accuracy over the datasets they were scored on (so a checkpoint that is good on one dataset only
    does not push out one that is good on both), and every knn and classic baseline. Each dataset lists the
    chosen methods it has, best first on that dataset, so the panels of a figure compare the same methods."""
    full = {d: pick_methods(view3, kinds, d, top=10 ** 6) for d in datasets}
    scores = collections.defaultdict(list)
    for ms in full.values():
        for m in ms:
            if m['kind'] in ('classifier', 'prototype') and not math.isnan(m['acc']):
                scores[m['base']].append(m['acc'])
    chosen = set(sorted(scores, key=lambda b: -sum(scores[b]) / len(scores[b]))[:top])
    return {d: [m for m in ms if m['base'] in chosen or m['kind'] not in ('classifier', 'prototype')]
            for d, ms in full.items()}


def _best_line(picked: list, dataset: str) -> str:
    """The sentence a methods figure is titled with: the best method, and the baselines next to it."""
    if not picked:
        return dataset
    best = picked[0]
    base = [m for m in picked if m['kind'] in ('knn', 'classic')]
    tail = '; '.join(f'{m["base"]} {m["acc"]:.0%}' for m in base[:3])
    return f'{dataset}: best is {short_name(best["base"])} at {best["acc"]:.0%}' + (f'   ({tail})' if tail else '')


def ladder_step(rows: list, dataset: str) -> float:
    """The scale factor between neighbouring true rungs of a dataset (BRACS 4, Ki67 2): what one pyramid
    level is. An error in octaves means one level on one dataset and two on the other; divided by this
    step it is the same thing on both."""
    rungs = sorted({num(r, 'rung') for r in rows if cell(r, 'dataset') == dataset and num(r, 'rung')})
    ratios = [b / a for a, b in zip(rungs, rungs[1:]) if a > 0]
    return min(ratios) if ratios else 2.0


def level_bucket(row: dict, step: float) -> str:
    """Where one estimate stands against the true rung, in the dataset's own levels: `right`, or how many
    levels too fine (`-1`, `<=-2`) or too coarse (`+1`, `>=+2`), a wrong answer counting at least one level."""
    if is_correct(row):
        return 'right'
    ds, rung = num(row, 'estimated_ds'), num(row, 'rung')
    if ds is None or rung is None or ds <= 0:
        return 'right'
    octaves = math.log2(ds) - math.log2(rung)
    levels = max(1, int(round(abs(octaves) / math.log2(step))))
    if octaves < 0:
        return '-1' if levels == 1 else '<=-2'
    return '+1' if levels == 1 else '>=+2'


def _clean(ax) -> None:
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=_INK)


def fig_methods(picked_by_dataset: dict, out_path: str, plt) -> None:
    """Which method is best: a dot and its 95% interval per method, best at the top. The dot is the
    default rule (what the pipeline runs); the hollow diamond is the best rule of that checkpoint. Rows
    whose interval overlaps the best one's are shaded -- they are not shown to be worse."""
    datasets = list(picked_by_dataset)
    height = 0.34 * max(len(p) for p in picked_by_dataset.values()) + 2.2
    fig, axes = plt.subplots(1, len(datasets), figsize=(8.2 * len(datasets), height), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        picked = picked_by_dataset[dataset]
        y = list(range(len(picked)))[::-1]
        best_lo = picked[0]['lo'] if picked and not math.isnan(picked[0]['lo']) else float('nan')
        for yi, m in zip(y, picked):
            tied = not math.isnan(best_lo) and not math.isnan(m['hi']) and m['hi'] >= best_lo
            if tied:
                ax.axhspan(yi - 0.5, yi + 0.5, color='#eef3f8', zorder=0)
            colour = KIND_COLOURS.get(m['kind'], '#555555')
            if not math.isnan(m['lo']):
                ax.plot([m['lo'], m['hi']], [yi, yi], color=colour, lw=2, zorder=2)
            ax.scatter([m['acc']], [yi], color=colour, s=46, zorder=3)
            if m['kind'] in ('classifier', 'prototype') and m['best_acc'] > m['acc'] + 1e-9:
                ax.scatter([m['best_acc']], [yi], facecolors='none', edgecolors=colour, marker='D', s=40,
                           zorder=3, linewidths=1.2)
            ax.text(1.01, yi, f'{m["acc"]:.0%}', va='center', fontsize=8, color=_INK,
                    transform=ax.get_yaxis_transform())
        labels = [short_name(m['base']) + ('   <- pipeline' if m['base'] == 'knn:gigapath' else '') for m in picked]
        ax.set_yticks(y, labels, fontsize=8)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
        ax.set_xlabel('level accuracy, mean over rungs (%)   dot: default rule, diamond: best rule')
        ax.set_title(_best_line(picked, dataset), fontsize=10, loc='left', color=_INK)
        ax.grid(axis='x', alpha=0.25)
        _clean(ax)
    handles = [plt.Line2D([], [], marker='o', color=c, ls='', label=k) for k, c in KIND_COLOURS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.text(0.01, 0.01, 'shaded rows: the 95% interval overlaps the best method\'s, so they are not shown to be worse.',
             fontsize=8, color=_INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_by_rung(view2: list, picked_by_dataset: dict, out_path: str, plt) -> None:
    """Where each method fails: accuracy at each true rung, a line per method."""
    datasets = list(picked_by_dataset)
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.6 * len(datasets), 4.4), squeeze=False, sharey=True)
    for ax, dataset in zip(axes[0], datasets):
        picked = picked_by_dataset[dataset][:6] + [m for m in picked_by_dataset[dataset][6:]
                                                   if m['kind'] in ('knn', 'classic')]
        names = {m['base'] for m in picked}
        for m in picked:
            pts = sorted((r['rung'], r['level_accuracy']) for r in view2
                         if r['dataset'] == dataset and r['method'].split('@', 1)[0] == m['base'])
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', ms=4, lw=1.6,
                    color=KIND_COLOURS.get(m['kind'], '#555555'),
                    ls='-' if m['kind'] in ('classifier', 'knn') else '--', alpha=0.9,
                    label=short_name(m['base'])[:34])
        rungs = sorted({r['rung'] for r in view2 if r['dataset'] == dataset and r['method'].split('@', 1)[0] in names})
        ax.set_xscale('log', base=2)
        ax.set_xticks(rungs, [f'{x:g}' for x in rungs])
        ax.minorticks_off()
        ax.set_ylim(0, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
        ax.set_xlabel('true rung (ds)')
        ax.set_ylabel('% right')
        ax.set_title(f'{dataset}: accuracy at each scale, best {len(picked)} methods and the baselines',
                     fontsize=10, loc='left', color=_INK)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=6.5, loc='lower left')
        _clean(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_levels_off(rows: list, picked_by_dataset: dict, out_path: str, plt) -> None:
    """How wrong when wrong, in the dataset's own levels: a 100% bar per method, right in the middle,
    too fine to its left, too coarse to its right. One level off is one level on every dataset."""
    datasets = list(picked_by_dataset)
    height = 0.34 * max(len(p) for p in picked_by_dataset.values()) + 2.4
    fig, axes = plt.subplots(1, len(datasets), figsize=(8.2 * len(datasets), height), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        picked = picked_by_dataset[dataset]
        step = ladder_step(rows, dataset)
        counts = {m['base']: collections.Counter() for m in picked}
        for r in rows:
            if cell(r, 'dataset') != dataset:
                continue
            base = base_method_of(r)
            if base in counts and (not cell(r, 'vote') or cell(r, 'vote') == DEFAULT_VOTE):
                counts[base][level_bucket(r, step)] += 1
        y = list(range(len(picked)))[::-1]
        for yi, m in zip(y, picked):
            c = counts[m['base']]
            total = sum(c.values()) or 1
            left = 0.0
            for key in LEVEL_ORDER:
                share = c[key] / total
                ax.barh(yi, share, left=left, color=LEVEL_COLOURS[key], height=0.72, edgecolor='white', linewidth=0.5)
                if share >= 0.06:
                    ax.text(left + share / 2, yi, f'{100 * share:.0f}', ha='center', va='center', fontsize=7,
                            color='white' if key in ('<=-2', '>=+2', 'right') else _INK)
                left += share
        ax.set_yticks(y, [short_name(m['base']) for m in picked], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
        ax.set_xlabel(f'% of FoVs   (one level = a x{step:g} step of ds on this dataset)')
        ax.set_title(f'{dataset}: when a method is wrong, by how many levels', fontsize=10, loc='left', color=_INK)
        _clean(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=LEVEL_COLOURS[k]) for k in LEVEL_ORDER]
    fig.legend(handles, [LEVEL_LABELS[k] for k in LEVEL_ORDER], loc='lower center', ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_votes(acc: list, summary: list, out_path: str, plt) -> None:
    """Which vote rule: for each checkpoint, a rule's accuracy minus the mean of that checkpoint over its
    rules, so checkpoints of different strength sit on one axis. A dot per checkpoint, a diamond for the
    mean; a rule whose dots stay above zero is better for every checkpoint, not for a few."""
    panels = sorted({(r['dataset'], r['kind']) for r in acc})
    n_cols = max(len({k for d, k in panels if d == dataset}) for dataset in {d for d, _ in panels})
    datasets = sorted({d for d, _ in panels})
    fig, axes = plt.subplots(len(datasets), n_cols, figsize=(6.4 * n_cols, 3.9 * len(datasets)), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, dataset in enumerate(datasets):
        for j, kind in enumerate(sorted({k for d, k in panels if d == dataset})):
            ax = axes[i][j]
            ax.set_visible(True)
            grp = [r for r in acc if r['dataset'] == dataset and r['kind'] == kind
                   and not math.isnan(r['mean_level_accuracy'])]
            by_method = collections.defaultdict(dict)
            for r in grp:
                by_method[r['method']][r['vote']] = r['mean_level_accuracy']
            delta = collections.defaultdict(list)
            for votes in by_method.values():
                mean = sum(votes.values()) / len(votes)
                for vote, a in votes.items():
                    delta[vote].append(a - mean)
            order = sorted(delta, key=lambda v: -(sum(delta[v]) / len(delta[v])))
            for x, vote in enumerate(order):
                vals = delta[vote]
                jitter = [(-0.18 + 0.36 * (k % 7) / 6) for k in range(len(vals))]
                ax.scatter([x + jj for jj in jitter], vals, s=14, color=KIND_COLOURS.get(kind, '#555555'),
                           alpha=0.55, zorder=2)
                ax.scatter([x], [sum(vals) / len(vals)], marker='D', s=70, color='white',
                           edgecolors=_INK, linewidths=1.3, zorder=3)
            ax.axhline(0, color=_INK, lw=0.8)
            ax.set_xticks(range(len(order)), order, rotation=35, ha='right', fontsize=7)
            ax.set_ylabel('accuracy vs the checkpoint\'s mean rule (points)')
            best = order[0] if order else '?'
            ax.set_title(f'{dataset}, {kind}: best rule {best}  ({len(by_method)} checkpoints)', fontsize=9,
                         loc='left', color=_INK)
            ax.grid(axis='y', alpha=0.25)
            _clean(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_votes_pairwise(summary: list, pairwise: list, out_path: str, plt) -> None:
    """Who is right when two rules disagree: the best rule against each other rule, FoVs where only one
    of the two is right, as a PERCENT of the FoVs. Right of the line: only the best rule is right. A star
    where the sign test says the lean is not chance (with this many pairs a tiny lean gets stars too: read
    the length, not the stars). Every panel is drawn on the same axis, so the lengths compare across them."""
    panels = sorted({(r['dataset'], r['kind']) for r in summary})
    datasets = sorted({d for d, _ in panels})
    n_cols = max(len({k for d, k in panels if d == dataset}) for dataset in datasets)
    fig, axes = plt.subplots(len(datasets), n_cols, figsize=(6.4 * n_cols, 3.9 * len(datasets)), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    drawn: list = []
    for i, dataset in enumerate(datasets):
        for j, kind in enumerate(sorted({k for d, k in panels if d == dataset})):
            rules = [r for r in summary if r['dataset'] == dataset and r['kind'] == kind]
            if not rules:
                continue
            best = max(rules, key=lambda r: r['mean_acc'] if not math.isnan(r['mean_acc']) else -1)['vote']
            bars = []
            for p in pairwise:
                if p['dataset'] != dataset or p['kind'] != kind or best not in (p['vote_a'], p['vote_b']):
                    continue
                best_first = p['vote_a'] == best
                other = p['vote_b'] if best_first else p['vote_a']
                mine, theirs = (p['a_only'], p['b_only']) if best_first else (p['b_only'], p['a_only'])
                n = p['n_fov'] or 1
                bars.append((other, 100.0 * mine / n, 100.0 * theirs / n, p['sign_p'], p['n_disagree']))
            bars.sort(key=lambda b: b[1] - b[2])
            drawn.append((axes[i][j], dataset, kind, best, bars))
    longest_right = max((b[1] for _, _, _, _, bars in drawn for b in bars), default=1.0)
    longest_left = max((b[2] for _, _, _, _, bars in drawn for b in bars), default=1.0)
    for ax, dataset, kind, best, bars in drawn:
        ax.set_visible(True)
        y = list(range(len(bars)))
        for yi, (other, mine, theirs, pval, nd) in zip(y, bars):
            ax.barh(yi, mine, color=RIGHT_COLOUR, height=0.7)
            ax.barh(yi, -theirs, color=WRONG_COLOUR, height=0.7)
            star = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            ax.text(mine + 0.01 * longest_right, yi, star, va='center', fontsize=9, color=_INK)
        ax.axvline(0, color=_INK, lw=0.8)
        ax.set_xlim(-1.15 * longest_left, 1.2 * longest_right)
        ax.set_yticks(y, [b[0] for b in bars], fontsize=7)
        ax.set_xlabel(f'% of FoVs where only one of the two is right   (right: only {best}; left: only the other)',
                      fontsize=8)
        ax.set_title(f'{dataset}, {kind}: {best} against the others   (* sign test)', fontsize=9,
                     loc='left', color=_INK)
        ax.grid(axis='x', alpha=0.25)
        _clean(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


#: A risk flag drawn as a distribution of the number it is made from, with its threshold:
#: (title, rule, column, direction, tau name or a fixed value). '<' lights below the threshold, '>' above,
#: '|>' above in absolute value.
RISK_HISTOGRAMS = (
    ('few_confident_decide: share of patches backing the winner', 'mean_probability', 'risk_winner_support', '<', 0.5),
    ('weak_majority_strong_dissent: majority\'s margin', 'hard_majority', 'risk_margin_majority_p50', '<',
     'hm_margin_majority_low'),
    ('weak_majority_strong_dissent: dissenters\' margin', 'hard_majority', 'risk_margin_dissent_p50', '>',
     'hm_margin_dissent_high'),
    ('unsure_median_or_split: top-1 probability, median', 'patch_class_median', 'risk_top1_p50', '<', 'pcm_top1_low'),
    ('weak_or_unsupported_snap: probability of the chosen level', 'median_log_rung', 'risk_winner_prob', '<',
     'mlr_support_low'),
    ('single_patch_veto: smallest probability a patch gives', 'sum_log_probability', 'risk_min_p_winner', '<',
     'slp_veto'),
    ('weight_tracks_scale: |corr(weight, e)|', 'quality_weighted', 'risk_corr_w_e', '|>', 'qw_corr_high'),
    ('weights collapse: effective sample size', 'quality_weighted', 'risk_ess_frac', '<', 'qw_ess_low'))
#: The flags that are only on or off: (flag, rule)
RISK_BINARY = (('vote_tie', 'hard_majority'), ('median_split', 'patch_class_median'),
               ('unsupported_snap', 'median_log_rung'), ('loo_unstable', 'sum_log_probability'),
               ('weighting_flipped', 'quality_weighted'))
#: Of a rule that has several labels, the one a risk figure draws.
RISK_LABEL_PREFERENCE = ('patch_class_median:lower', 'quality_weighted:tissue')


def _rule_label(rows: list, rule: str) -> str:
    labels = sorted({cell(r, 'vote') for r in rows if rule_of(r) == rule})
    for preferred in RISK_LABEL_PREFERENCE:
        if preferred in labels:
            return preferred
    return labels[0] if labels else ''


def fig_risk_flags(rows: list, dataset: str, kind: str, taus: dict, out_path: str, plt) -> bool:
    """The risk flags of one kind of method on one dataset, one small picture each. A flag made from a
    number: that number's distribution for FoVs the rule got right (green) and wrong (orange), each as a
    share of its own group, with the threshold line and the flagged side shaded -- the picture says whether
    the wrong FoVs sit on the flagged side. A flag that is on or off: the accuracy with and without it."""
    sub = [r for r in rows if cell(r, 'dataset') == dataset and kind_of(r) == kind and cell(r, 'vote')]
    if not sub:
        return False
    import numpy as np                                                    # noqa: PLC0415
    n_panels = len(RISK_HISTOGRAMS) + len(RISK_BINARY)
    n_cols = 4
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.3 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    k = 0
    for title, rule, column, direction, tau in RISK_HISTOGRAMS:
        ax = axes.flat[k]
        k += 1
        label = _rule_label(sub, rule)
        pts = [(num(r, column), is_correct(r)) for r in sub if cell(r, 'vote') == label and _finite(num(r, column))]
        if not pts:
            continue
        ax.set_visible(True)
        threshold = tau if isinstance(tau, float) else (taus.get(f'{kind}|{label}|{tau}') or {}).get('value')
        vals = np.array([abs(v) if direction == '|>' else v for v, _ in pts], dtype=float)
        right = np.array([c for _, c in pts], dtype=bool)
        edges = np.linspace(float(vals.min()), float(vals.max()) + 1e-9, 31)
        for flag, colour, name in ((True, RIGHT_COLOUR, 'right'), (False, WRONG_COLOUR, 'wrong')):
            v = vals[right == flag]
            if len(v):
                ax.hist(v, bins=edges, weights=np.full(len(v), 1.0 / len(v)), histtype='stepfilled', alpha=0.45,
                        color=colour, edgecolor=colour, label=f'{name} (n={len(v)})')
        note = ''
        if threshold is not None and _finite(threshold):
            ax.axvline(threshold, color=_INK, ls='--', lw=1)
            lit = vals > threshold if direction in ('>', '|>') else vals < threshold
            if direction in ('>', '|>'):
                ax.axvspan(threshold, float(vals.max()), color='#f3d9cc', alpha=0.35, zorder=0)
            else:
                ax.axvspan(float(vals.min()), threshold, color='#f3d9cc', alpha=0.35, zorder=0)
            share_wrong = float(lit[~right].mean()) if (~right).any() else float('nan')
            share_right = float(lit[right].mean()) if right.any() else float('nan')
            note = f'flagged: {share_wrong:.0%} of the wrong, {share_right:.0%} of the right'
        elif not isinstance(tau, float):
            note = 'threshold not loaded (fit it on val)'
        ax.set_title(f'{title}\n{label}   {note}', fontsize=7.5, loc='left', color=_INK)
        ax.set_ylabel('share of its group', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(frameon=False, fontsize=7)
        _clean(ax)
    for flag, rule in RISK_BINARY:
        ax = axes.flat[k]
        k += 1
        label = _rule_label(sub, rule)
        pairs = [(flags_of(r, taus).get(flag), is_correct(r)) for r in sub if cell(r, 'vote') == label]
        pairs = [(f, c) for f, c in pairs if f is not None]
        if not pairs:
            continue
        ax.set_visible(True)
        on = [c for f, c in pairs if f]
        off = [c for f, c in pairs if not f]
        acc_on = sum(on) / len(on) if on else float('nan')
        acc_off = sum(off) / len(off) if off else float('nan')
        ax.bar([0, 1], [acc_off if off else 0, acc_on if on else 0], color=[RIGHT_COLOUR, WRONG_COLOUR], width=0.6)
        for x, a, n in ((0, acc_off, len(off)), (1, acc_on, len(on))):
            ax.text(x, (a if _finite(a) else 0) + 0.02, f'{a:.0%}  n={n}' if _finite(a) else f'n={n}',
                    ha='center', fontsize=7.5, color=_INK)
        risk = ((1 - acc_on) / (1 - acc_off)) if on and off and acc_off < 1 else float('nan')
        ax.set_xticks([0, 1], ['not flagged', 'flagged'], fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 0.5, 1], ['0', '50', '100'], fontsize=7)
        ax.set_ylabel('% right', fontsize=7)
        prev = len(on) / len(pairs) if pairs else float('nan')
        ax.set_title(f'{flag}: lit on {prev:.1%} of FoVs\n{label}   flagged FoVs err x{risk:.1f}' if _finite(risk)
                     else f'{flag}: lit on {prev:.1%} of FoVs\n{label}', fontsize=7.5, loc='left', color=_INK)
        _clean(ax)
    fig.suptitle(f'{dataset}, {kind}: risk flags -- is the flagged side where the wrong FoVs are?', x=0.01,
                 ha='left', fontsize=11, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def fig_strata(strata: list, dataset: str, kind: str, out_path: str, plt) -> bool:
    """Which rule when: accuracy of every rule inside each slice of the FoV's patch distribution (rows:
    rules, columns: the slices of one statistic). The box marks the best rule of each column; where it
    moves along a row of cells, the best rule depends on what the patches look like."""
    sub = [r for r in strata if r['dataset'] == dataset and r['kind'] == kind and not r['method']]
    if not sub:
        return False
    import numpy as np                                                    # noqa: PLC0415
    stats = [c for c, _ in STRATA if any(r['stat'] == c for r in sub)]
    rules = sorted({r['vote'] for r in sub})
    fig, axes = plt.subplots(1, len(stats), figsize=(3.7 * len(stats), 0.5 * len(rules) + 2.6), squeeze=False)
    for ax, stat in zip(axes[0], stats):
        bins = [b[0] for b in dict(STRATA)[stat] if any(r['stat'] == stat and r['bin'] == b[0] for r in sub)]
        grid = np.full((len(rules), len(bins)), np.nan)
        counts = {}
        for r in sub:
            if r['stat'] == stat and r['bin'] in bins and r['vote'] in rules:
                grid[rules.index(r['vote']), bins.index(r['bin'])] = r['accuracy']
                counts[r['bin']] = r['n']
        ax.imshow(grid, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        for a in range(len(rules)):
            for b in range(len(bins)):
                if not math.isnan(grid[a, b]):
                    ax.text(b, a, f'{100 * grid[a, b]:.0f}', ha='center', va='center', fontsize=7,
                            color='white' if grid[a, b] > 0.6 else _INK)
        for b in range(len(bins)):
            col = grid[:, b]
            if not np.isnan(col).all():
                best = int(np.nanargmax(col))
                ax.add_patch(plt.Rectangle((b - 0.5, best - 0.5), 1, 1, fill=False, edgecolor='#D55E00', lw=2))
        ax.set_xticks(range(len(bins)), [f'{b}\nn={counts.get(b, 0)}' for b in bins], fontsize=7)
        ax.set_yticks(range(len(rules)), rules if ax is axes[0][0] else [], fontsize=7)
        ax.set_title(stat, fontsize=8, loc='left', color=_INK)
    fig.suptitle(f'{dataset}, {kind}: accuracy (%) of each vote rule by how the patches look; box = best rule '
                 f'in that slice', x=0.01, ha='left', fontsize=10, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def fig_distribution(distribution: list, dataset: str, kind: str, out_path: str, plt) -> bool:
    """What a wrong FoV looks like: the median of every patch-distribution statistic, per rung, for FoVs
    the method got right (green) and wrong (orange). Lines that part are statistics that tell them apart."""
    sub = [r for r in distribution if r['dataset'] == dataset and r['kind'] == kind]
    if not sub:
        return False
    n_cols = 4
    n_rows = math.ceil(len(FOV_COLUMNS) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.9 * n_rows), squeeze=False)
    for ax, column in zip(axes.flat, FOV_COLUMNS):
        for outcome, colour in (('right', RIGHT_COLOUR), ('wrong', WRONG_COLOUR)):
            pts = sorted((r['rung'], r[f'{column}_p50']) for r in sub if r['outcome'] == outcome
                         and r['rung'] is not None and _finite(r[f'{column}_p50']))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', ms=4, color=colour, label=outcome)
        ax.set_xscale('log', base=2)
        rungs = sorted({r['rung'] for r in sub if r['rung'] is not None})
        ax.set_xticks(rungs, [f'{x:g}' for x in rungs], fontsize=7)
        ax.minorticks_off()
        ax.set_title(column, fontsize=8, loc='left', color=_INK)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        _clean(ax)
    for ax in list(axes.flat)[len(FOV_COLUMNS):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f'{dataset}, {kind}: median of each patch statistic by true rung, right vs wrong FoVs', x=0.01,
                 ha='left', fontsize=10, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def fig_risk_coverage(rc: list, out_path: str, plt) -> None:
    """What refusing the least sure FoVs buys: the error left among the FoVs answered, at 25 / 50 / 75 /
    100% coverage, per vote rule (mean over checkpoints). A lower line is a rule whose confidence can be
    trusted to say when not to answer."""
    panels = sorted({(r['dataset'], r['kind']) for r in rc})
    datasets = sorted({d for d, _ in panels})
    n_cols = max(len({k for d, k in panels if d == dataset}) for dataset in datasets)
    fig, axes = plt.subplots(len(datasets), n_cols, figsize=(6.0 * n_cols, 3.9 * len(datasets)), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    rules = sorted({r['vote'] for r in rc})
    palette = plt.get_cmap('tab10')
    colour = {v: palette(i % 10) for i, v in enumerate(rules)}
    for i, dataset in enumerate(datasets):
        for j, kind in enumerate(sorted({k for d, k in panels if d == dataset})):
            ax = axes[i][j]
            ax.set_visible(True)
            sub = [r for r in rc if r['dataset'] == dataset and r['kind'] == kind]
            by_rule = collections.defaultdict(list)
            for r in sub:
                by_rule[r['vote']].append(r)
            aurc = {v: sum(r['aurc'] for r in g) / len(g) for v, g in by_rule.items()}
            for v in sorted(by_rule, key=aurc.get):
                g = by_rule[v]
                ys = [sum(r[f'err_at_{c}'] for r in g) / len(g) for c in (25, 50, 75, 100)]
                ax.plot([25, 50, 75, 100], ys, marker='o', ms=3, lw=1.4, color=colour[v],
                        label=f'{v}  AURC {aurc[v]:.3f}')
            best = min(aurc, key=aurc.get) if aurc else '?'
            ax.set_xticks([25, 50, 75, 100])
            ax.set_xlabel('coverage (% of FoVs answered, most confident first)')
            ax.set_ylabel('error among those answered')
            ax.set_title(f'{dataset}, {kind}: lowest AURC is {best}', fontsize=9, loc='left', color=_INK)
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=6)
            _clean(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_confusion(conf: list, picked_by_dataset: dict, dataset: str, out_path: str, plt, top: int = 6) -> bool:
    """What each method calls each true rung: rows are the true rung, columns the rung it called, each row
    a share of its FoVs; the diagonal is right."""
    import numpy as np                                                    # noqa: PLC0415
    picked = picked_by_dataset[dataset]
    chosen = [m for m in picked if m['kind'] in ('classifier', 'prototype')][:top - 2] + \
             [m for m in picked if m['kind'] in ('knn', 'classic')][:2]
    rungs = sorted({c['gt_rung'] for c in conf if c['dataset'] == dataset and c['gt_rung'] is not None})
    if not chosen or not rungs:
        return False
    preds = sorted({c['pred_rung'] for c in conf if c['dataset'] == dataset and c['pred_rung'] is not None}
                   | set(rungs))
    n_cols = min(3, len(chosen))
    n_rows = math.ceil(len(chosen) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.1 * n_cols, 3.7 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, m in zip(axes.flat, chosen):
        ax.set_visible(True)
        grid = np.zeros((len(rungs), len(preds)))
        for c in conf:
            label = c['method']
            right_vote = label == m['base'] or label == m['base'] + '@' + DEFAULT_VOTE
            if c['dataset'] == dataset and right_vote and c['gt_rung'] in rungs and c['pred_rung'] in preds:
                grid[rungs.index(c['gt_rung']), preds.index(c['pred_rung'])] += c['n']
        norm = grid / np.maximum(grid.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        for a in range(len(rungs)):
            for b in range(len(preds)):
                if norm[a, b] >= 0.01:
                    ax.text(b, a, f'{100 * norm[a, b]:.0f}', ha='center', va='center', fontsize=7,
                            color='white' if norm[a, b] > 0.6 else _INK)
        ax.set_xticks(range(len(preds)), [f'{p:g}' for p in preds], fontsize=7)
        ax.set_yticks(range(len(rungs)), [f'{r:g}' for r in rungs], fontsize=7)
        ax.set_xlabel('called rung (ds)', fontsize=7)
        ax.set_ylabel('true rung (ds)', fontsize=7)
        ax.set_title(f'{short_name(m["base"])[:34]}\n{m["acc"]:.0%} right', fontsize=8, loc='left', color=_INK)
    fig.suptitle(f'{dataset}: what each method calls each true rung (% of the true rung\'s FoVs)', x=0.01,
                 ha='left', fontsize=10, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def fig_per_slide(view1: list, picked_by_dataset: dict, dataset: str, out_path: str, plt) -> bool:
    """Whether a method fails on one slide or everywhere: accuracy of the best method and of the pipeline's
    current one (knn:gigapath) on every slide at every rung, slides down, rungs across."""
    import numpy as np                                                    # noqa: PLC0415
    picked = picked_by_dataset[dataset]
    shown = [picked[0]] + [m for m in picked if m['base'] == 'knn:gigapath' and m is not picked[0]]
    sub = [r for r in view1 if r['dataset'] == dataset]
    slides = sorted({r['wsi_name'] for r in sub})
    rungs = sorted({r['rung'] for r in sub if r['rung'] is not None})
    if not slides or not rungs:
        return False
    fig, axes = plt.subplots(1, len(shown), figsize=(4.4 * len(shown) + 1.5, 0.34 * len(slides) + 2.2), squeeze=False)
    for ax, m in zip(axes[0], shown):
        grid = np.full((len(slides), len(rungs)), np.nan)
        for r in sub:
            if r['method'].split('@', 1)[0] == m['base'] and r['rung'] in rungs:
                grid[slides.index(r['wsi_name']), rungs.index(r['rung'])] = r['level_accuracy']
        ax.imshow(grid, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        for a in range(len(slides)):
            for b in range(len(rungs)):
                if not math.isnan(grid[a, b]):
                    ax.text(b, a, f'{100 * grid[a, b]:.0f}', ha='center', va='center', fontsize=7,
                            color='white' if grid[a, b] > 0.6 else _INK)
        ax.set_xticks(range(len(rungs)), [f'{r:g}' for r in rungs], fontsize=7)
        ax.set_yticks(range(len(slides)), [s[:22] for s in slides] if ax is axes[0][0] else [], fontsize=7)
        ax.set_xlabel('true rung (ds)', fontsize=7)
        ax.set_title(f'{short_name(m["base"])[:36]}', fontsize=8, loc='left', color=_INK)
    fig.suptitle(f'{dataset}: % right on each slide and rung -- does a method fail on one slide or everywhere?',
                 x=0.01, ha='left', fontsize=10, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def fig_native(view2: list, picked_by_dataset: dict, out_path: str, plt) -> None:
    """Whether resampling flatters a method: accuracy on rungs the slide has natively (filled) and on rungs
    read one level finer and shrunk (hollow), per method. A long line means the method may be reading the
    resampling, not the scale."""
    datasets = list(picked_by_dataset)
    height = 0.3 * max(len(p) for p in picked_by_dataset.values()) + 2.0
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.6 * len(datasets), height), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        picked = picked_by_dataset[dataset]
        y = list(range(len(picked)))[::-1]
        for yi, m in zip(y, picked):
            rs = [r for r in view2 if r['dataset'] == dataset and r['method'].split('@', 1)[0] == m['base']]
            nat = [r['level_accuracy_native'] for r in rs if _finite(r['level_accuracy_native'])]
            res = [r['level_accuracy_resampled'] for r in rs if _finite(r['level_accuracy_resampled'])]
            colour = KIND_COLOURS.get(m['kind'], '#555555')
            if nat and res:
                a, b = sum(nat) / len(nat), sum(res) / len(res)
                ax.plot([a, b], [yi, yi], color='#c3c2b7', lw=1.5, zorder=1)
            if nat:
                ax.scatter([sum(nat) / len(nat)], [yi], color=colour, s=40, zorder=3)
            if res:
                ax.scatter([sum(res) / len(res)], [yi], facecolors='none', edgecolors=colour, s=44, zorder=3,
                           linewidths=1.3)
        ax.set_yticks(y, [short_name(m['base']) for m in picked], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
        ax.set_xlabel('% right   filled: native rungs   hollow: resampled rungs')
        ax.set_title(f'{dataset}: native vs resampled', fontsize=10, loc='left', color=_INK)
        ax.grid(axis='x', alpha=0.25)
        _clean(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_figures(rows: list, view1: list, view2: list, view3: list, tables: dict, taus: dict, out_dir: str,
                 stem: str, top: int) -> list:
    """Every figure of this file, from the tables `main` made; the paths written. `tables` carries
    `acc`, `summary`, `pairwise`, `strata`, `distribution`, `rc` and `conf` (a missing one leaves its
    figures out). A missing matplotlib leaves out all of them, never the tables."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [skip] matplotlib not available in this python -- no figures (the tables are unaffected)')
        return []
    kinds = {base_method_of(r): kind_of(r) for r in rows}
    datasets = sorted({cell(r, 'dataset') for r in rows if cell(r, 'dataset')})
    picked = {d: p for d, p in pick_shared(view3, kinds, datasets, top).items() if p}
    written: list = []

    def path(name: str) -> str:
        p = os.path.join(out_dir, f'{stem}_fig_{name}.png')
        written.append(p)
        return p

    if picked:
        fig_methods(picked, path('methods'), plt)
        fig_by_rung(view2, picked, path('by_rung'), plt)
        fig_levels_off(rows, picked, path('levels_off'), plt)
        fig_native(view2, picked, path('native_vs_resampled'), plt)
        for d in picked:
            tag = d.replace('/', '_')
            if not fig_confusion(tables.get('conf', []), picked, d, path(f'confusion_{tag}'), plt):
                written.pop()
            if not fig_per_slide(view1, picked, d, path(f'per_slide_{tag}'), plt):
                written.pop()
    if tables.get('acc'):
        fig_votes(tables['acc'], tables['summary'], path('votes'), plt)
        fig_votes_pairwise(tables['summary'], tables['pairwise'], path('votes_pairwise'), plt)
    if tables.get('rc'):
        fig_risk_coverage(tables['rc'], path('risk_coverage'), plt)
    for d in datasets:
        for kind in sorted({kind_of(r) for r in rows if cell(r, 'vote') and cell(r, 'dataset') == d}):
            tag = f'{d.replace("/", "_")}_{kind}'
            if not fig_risk_flags(rows, d, kind, taus, path(f'risk_flags_{tag}'), plt):
                written.pop()
            if not fig_strata(tables.get('strata', []), d, kind, path(f'strata_{tag}'), plt):
                written.pop()
            if not fig_distribution(tables.get('distribution', []), d, kind, path(f'patch_stats_{tag}'), plt):
                written.pop()
    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', 'bench_modules'))
    from bench_stage1_mpp import JOB_NAME, add_args               # noqa: PLC0415
    add_args(ap)
    ap.add_argument('--stage-cache-job', default=_paths.job_name(JOB_NAME),
                    help='the bench job whose stage-1 entries are scored. '
                         'Default: this job (SLURM_JOB_NAME first), the one '
                         'bench_stage1_mpp wrote under in the same jobscript')
    ap.add_argument('--out', default=None,
                    help='where the table, the analysis tables and the '
                         'figures go. Default: result/<job>/')
    ap.add_argument('--method', nargs='+', default=None,
                    help='keep only these method labels (see method_of)')
    ap.add_argument('--dataset', nargs='+', default=None)
    ap.add_argument('--all-votes', action='store_true',
                    help='views 1-2 and the figure for every vote rule, not '
                         f'only {DEFAULT_VOTE}')
    ap.add_argument('--top', type=int, default=12,
                    help='the figures draw this many voting checkpoints, best first, and always every '
                         'knn and classic baseline')
    ap.add_argument('--fit-thresholds', action='store_true',
                    help='fit every risk threshold on THIS file, which must be '
                         'a val run, and write <csv stem>_thresholds.json')
    ap.add_argument('--thresholds', default=None,
                    help="a thresholds json fitted on val, or 'auto' for the "
                         "val set of this test set (@legacy-test -> "
                         "stage1_legacy-val_thresholds.json)")
    args = ap.parse_args()

    out_dir = args.out or _paths.job_result_dir(JOB_NAME)
    os.makedirs(out_dir, exist_ok=True)
    rows = rows_from_cache(args)
    if not rows:
        sys.exit('no stage-1 entry in the cache for these flags')
    args.csv_path = os.path.join(out_dir, f'stage1_{set_name(args.datasets)}.csv')
    write_rows(rows, args.csv_path)
    # read back, so every cell is the string the scoring parses, as from any
    # table on disk
    with open(args.csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    print(f'{args.csv_path}\n{len(rows)} rows')

    if args.dataset:
        rows = [r for r in rows if cell(r, 'dataset') in args.dataset]
    if args.method:
        rows = [r for r in rows if method_of(r) in args.method]
    if not rows:
        sys.exit('no rows left after filtering')

    # Views 1-2 and the figure: each voting checkpoint under its default rule
    # only, unless --all-votes -- six rules per checkpoint would bury them.
    # View 3 always carries every rule; section 5 compares them.
    shown = rows if args.all_votes else [r for r in rows if is_default_vote(r)]
    view1 = per_slide_rung(shown)
    view2 = cross_slide_rung(shown)
    view3 = overall(rows)
    print_per_slide_rung(view1)
    print_cross_slide_rung(view2)
    print_overall(view3)

    csv_dir = os.path.dirname(os.path.abspath(args.csv_path)) or '.'
    csv_stem = os.path.splitext(os.path.basename(args.csv_path))[0]
    taus: dict = {}
    tables: dict = {}                       # what the figures read besides the views
    if any(cell(r, 'vote') for r in rows):
        paired = _paired(rows)
        acc = vote_accuracy(rows)
        summary = vote_summary(acc)
        pairwise = vote_pairwise(paired)
        oracle = vote_oracle(paired)
        strata_kind = vote_strata(rows, by_method=False)
        distribution = fov_distribution(rows)
        print_vote_diagnosis(summary, pairwise, oracle, strata_kind, distribution)
        print()
        for name, table in (('vote_accuracy', acc), ('vote_summary', summary),
                            ('vote_pairwise', pairwise), ('vote_oracle', oracle),
                            ('vote_strata', strata_kind),
                            ('vote_strata_per_method', vote_strata(rows, by_method=True)),
                            ('fov_distribution', distribution)):
            write_rows(table, os.path.join(csv_dir, f'{csv_stem}_{name}.csv'))
        if args.fit_thresholds:
            fitted = fit_thresholds(rows, args.csv_path)
            out = os.path.join(csv_dir, f'{csv_stem}_thresholds.json')
            with open(out, 'w') as fh:
                json.dump(fitted, fh, indent=1)
            print(f'\n  thresholds fitted on this val file: {out}  '
                  f'({len(fitted["values"])} values)')
            taus = fitted['values']
        elif args.thresholds:
            path = args.thresholds
            if path == 'auto':
                val_ids: Optional[List[str]] = val_set_of(args.datasets)
                path = (os.path.join(csv_dir, f'stage1_{set_name(val_ids)}_thresholds.json')
                        if val_ids else '')
            if path and os.path.exists(path):
                with open(path) as fh:
                    fitted = json.load(fh)
                check_estimators(fitted, rows, path)
                taus = fitted['values']
                print(f'\n  thresholds from {path}')
            else:
                print(f'\n  [warn] no thresholds file ({path or "--datasets is not a -test set"}); '
                      f'run the val set with --fit-thresholds first')
        flags = flag_report(rows, taus)
        print_flags(flags, bool(taus))
        rc = risk_coverage(rows)
        print_risk_coverage(rc)
        print()
        tables.update(acc=acc, summary=summary, pairwise=pairwise, strata=strata_kind, distribution=distribution,
                      rc=rc)
        write_rows(flags, os.path.join(csv_dir, f'{csv_stem}_risk_flags.csv'))
        write_rows(rc, os.path.join(csv_dir, f'{csv_stem}_risk_coverage.csv'))
        write_rows(vote_pairwise(paired, by_method=True),
                   os.path.join(csv_dir, f'{csv_stem}_vote_pairwise_per_method.csv'))
    tables['conf'] = confusion(rows)
    write_rows(tables['conf'], os.path.join(csv_dir, f'{csv_stem}_confusion.csv'))
    write_rows(view3, os.path.join(csv_dir, f'{csv_stem}_overall.csv'))

    print('\n' + BAR)
    print('4. PNG (accuracy vs rung, one file per dataset -- see plot_dataset)')
    print(BAR)
    for dataset in sorted({r['dataset'] for r in view2}):
        stem = dataset.replace('/', '_')
        plot_dataset(view2, dataset,
                    os.path.join(csv_dir, f'{csv_stem}_{stem}.png'))

    print('\n' + BAR)
    print('FIGURES   (make_figures: each one is titled with what it shows)')
    print(BAR)
    for p in make_figures(rows, view1, view2, view3, tables, taus, csv_dir, csv_stem, args.top):
        print(f'  {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
