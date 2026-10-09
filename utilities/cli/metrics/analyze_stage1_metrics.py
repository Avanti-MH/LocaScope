#!/usr/bin/env python3
"""Score the stage-1 entries bench_stage1_mpp.py wrote and say which
mpp-estimation method actually wins.

Takes bench_stage1_mpp's own flags and computes the same addresses: for
every slide its FoV draw, for every --stage1 method its stage-1 entry under
--stage-cache-job. No model and no GPU; the slide is opened only for its
pyramid. The entries are joined with the ground truth -- where each FoV was
placed (`FovSupply.geometry`) -- into one table, written as
`<out>/stage1_<split>.csv`, and everything below reads that table.

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

        split                                 -- val / test
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
(`stage1_val_thresholds.json`).

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

native vs resampled, same reason `training/MppRoutingHead/Runtime.score`
splits it: a rung whose mpp is not on a slide's own pyramid is read one level
finer and resampled down, which leaves a resampling signature a method could
be scoring on instead of genuine scale. Meaningless pooled across datasets
with different pyramid steps (BRACS 4x, Ki67 2x) -- always split by dataset
first.

Usage:
    python utilities/cli/metrics/analyze_stage1_metrics.py <bench_stage1_mpp flags> \\
        --split val --fit-thresholds
    python utilities/cli/metrics/analyze_stage1_metrics.py <bench_stage1_mpp flags> \\
        --split test --thresholds auto
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import sys

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
                    gt_mpp=base_mpp * gt_ds, gt_ds=gt_ds, split=args.split,
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
#: Keyed by `training/MppRoutingHead/Runtime.HEAD_CHOICES`'s registered names
#: (a CLASSIFIER_RECIPES name not keyed here draws in the secondary ink). Its
#: own table: `training/MppRoutingHead/cli/evaluate.py` colours by classifier
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
                 + '; '.join(bad) + '. Rerun the val split with the same '
                 'estimators and refit')


def fit_thresholds(rows: list, source: str) -> dict:
    """Every tau of THRESHOLD_SPECS, per (kind, vote label) -- a prototype's
    near-uniform probabilities and a classifier's sharp ones are different
    scales, so one tau across both would flag one of them almost always.
    Refuses a file with test rows in it."""
    splits = {cell(r, 'split') for r in rows}
    if 'test' in splits or None in splits:
        sys.exit(f'--fit-thresholds needs a val run (bench --split val); this file '
                 f'has split {sorted(s or "unrecorded" for s in splits)}')
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
    ap.add_argument('--fit-thresholds', action='store_true',
                    help='fit every risk threshold on THIS file, which must be '
                         'a val run, and write <csv stem>_thresholds.json')
    ap.add_argument('--thresholds', default=None,
                    help="a thresholds json fitted on val, or 'auto' for the "
                         "val file of this same recipe (<..>_test.csv -> "
                         "<..>_val_thresholds.json)")
    args = ap.parse_args()

    out_dir = args.out or _paths.job_result_dir(JOB_NAME)
    os.makedirs(out_dir, exist_ok=True)
    rows = rows_from_cache(args)
    if not rows:
        sys.exit('no stage-1 entry in the cache for these flags')
    args.csv_path = os.path.join(out_dir, f'stage1_{args.split}.csv')
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
        taus = {}
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
                path = (args.csv_path[:-len('_test.csv')] + '_val_thresholds.json'
                        if args.csv_path.endswith('_test.csv') else '')
            if path and os.path.exists(path):
                with open(path) as fh:
                    fitted = json.load(fh)
                check_estimators(fitted, rows, path)
                taus = fitted['values']
                print(f'\n  thresholds from {path}')
            else:
                print(f'\n  [warn] no thresholds file ({path or "not a _test.csv"}); '
                      f'run the val split with --fit-thresholds first')
        flags = flag_report(rows, taus)
        print_flags(flags, bool(taus))
        rc = risk_coverage(rows)
        print_risk_coverage(rc)
        print()
        write_rows(flags, os.path.join(csv_dir, f'{csv_stem}_risk_flags.csv'))
        write_rows(rc, os.path.join(csv_dir, f'{csv_stem}_risk_coverage.csv'))
        write_rows(vote_pairwise(paired, by_method=True),
                   os.path.join(csv_dir, f'{csv_stem}_vote_pairwise_per_method.csv'))
    write_rows(confusion(rows), os.path.join(csv_dir, f'{csv_stem}_confusion.csv'))
    write_rows(view3, os.path.join(csv_dir, f'{csv_stem}_overall.csv'))

    print('\n' + BAR)
    print('4. PNG (accuracy vs rung, one file per dataset -- see plot_dataset)')
    print(BAR)
    for dataset in sorted({r['dataset'] for r in view2}):
        stem = dataset.replace('/', '_')
        plot_dataset(view2, dataset,
                    os.path.join(csv_dir, f'{csv_stem}_{stem}.png'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
