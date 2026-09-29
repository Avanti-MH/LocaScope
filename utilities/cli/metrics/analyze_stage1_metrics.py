#!/usr/bin/env python3
"""Read a stage1_compare per-shot CSV and say which mpp-estimation method
actually wins.

No GPU, no WSI, no torch -- it only reads the csv, so it runs on a login node
and takes under a second. `bench_stage1_mpp.py` is the writer; THIS FILE decides the schema that writer
has to produce (below), and everything the writer collects exists because a
question here needs it -- not the other way round.

    csv columns (one row per (query, method)):
        dataset, wsi_name, x, y, h, w        -- FoV identity
        rung, native, gt_mpp, gt_ds          -- ground truth
        encoder, classifier, reduction, loss,
        weights                               -- which method/config;
                                              KnnEstMpp leaves classifier/
                                              reduction/loss/weights blank.
                                              `loss` (2026-09-22): bal/ord_a/
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

Filename: `<sampler_id>_<seg_id>.csv` -- the sampling recipe's own hash
(rungs, n per rung, seed, native_only, tile size, wh_ratio, mpixels) plus the
tissue-mask recipe's hash, so one file can hold EVERY method's rows over the
exact same drawn FoVs (a paired comparison, same premise as
`bench_subspace_knn.py`'s arm A/B) and a changed sampling recipe cannot land
on a stale file.

Scored on `estimated_*`, NOT `chosen_*`. `chosen_ds`/`chosen_level` already
went through the shared "snap to this WSI's own pyramid" step
(`SafeSlide.coarser_level_for_downsample`) that every method shares, so
scoring on it would measure that shared step as much as the method. Same
convention `bench_mpp_feature_decomposition.py`'s own `score()` uses.

THREE VIEWS, not one table:
    per (wsi_name, rung, method)   -- does a method fail on one slide, or
                                    everywhere at that scale?
    per (rung, method), cross-slide -- with n: `--native-only` means not
                                    every slide contributes every rung (BRACS
                                    has no native ds=2), so a rung's number
                                    can rest on one slide's worth of shots
                                    and the table has to say so rather than
                                    let a reader assume equal weight.
    one overall number per method  -- the MEAN of the per-rung accuracies
                                    above, not the pooled accuracy over every
                                    shot: rungs are not sampled equally
                                    (native_only leaves some rungs thin), and
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
    python utilities/cli/metrics/analyze_stage1_metrics.py \\
        result/Stage1MppBench/<sampler_id>_<seg_id>_<region_id>.csv

    python utilities/cli/metrics/analyze_stage1_metrics.py \\
        result/Stage1MppBench/<sampler_id>_<seg_id>_<region_id>.csv --native-only
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

BAR = '=' * 78

#: Keyed by "head recipe" (`<head_name>+<reduction>`, or 'baseline' for a
#: bare KnnEstMpp with neither) so `mlp` is the SAME colour in every
#: encoder's subplot -- colour follows the entity, not its position in
#: whichever list happened to be drawn that run. LOSS-INDEPENDENT: see
#: `head_recipe_of`'s own docstring for why loss changes linestyle instead
#: of colour.
#:
#: 2026-09-22: rebuilt against `training/MppRoutingHead/Runtime.
#: HEAD_CHOICES`'s actual registered names -- the previous version's keys
#: (`'LinearHead+fixed'`, `'ArcFaceHead+fixed'`, ...) were the CLASS names
#: `classifier` used to hold before `bench_stage1_mpp.py`'s
#: own `_method_specs` switched it to `head_name` (that switch's own
#: comment explains why: 'classifier' collapsed all five mlp variants into
#: one label). Nothing here was ever updated to match, so every lookup has
#: been falling through to the `_INK_SECONDARY` grey fallback since that
#: switch -- this was found and fixed the same day as the `loss` column,
#: not a regression it introduced. Colours reused from `training/
#: MppRoutingHead/cli/evaluate.py`'s own `HEAD_COLORS` (same head_name
#: keys, one extra file keying off the same 8) for consistency across the
#: two reports; `mlp_narrow` (present in `HEAD_CHOICES`, absent from that
#: file's own dict too) gets a 9th colour here, not yet backfilled there.
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
    bare KnnEstMpp (blank classifier) -- LOSS-INDEPENDENT on purpose
    (2026-09-22): colour follows which HEAD this is, the same entity
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
    return method.split('+', 1)[0]

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

    `loss` (2026-09-22) is appended ONLY when it is present AND not 'bal':
    every checkpoint trained before `--loss` existed, and every 'bal' run
    since, keeps the exact label it already had (no '+bal' suffix appearing
    everywhere) -- same reasoning `bench_stage1_mpp.py`'s own
    `_prototype_weight_filename`-style filenames only tag a NON-default
    loss. Without this, a bal- and an ord_a-trained checkpoint of the same
    encoder+head+reduction would collapse into one label and get averaged
    together -- the same bug `classifier` switching from the class name to
    `head_name` already fixed once for the mlp variants (see the module
    docstring's csv columns note).
    """
    parts = [cell(row, 'encoder') or '?']
    for k in ('classifier', 'reduction'):
        v = cell(row, k)
        if v:
            parts.append(v)
    loss = cell(row, 'loss')
    if loss and loss != 'bal':
        parts.append(loss)
    return '+'.join(parts)


def nearest_rung(ds: float, rungs=RUNGS) -> float:
    """Which rung `ds` is closest to, in LOG space -- pyramid scales are
    geometric, so a linear nearest would call 1-vs-4 nearer than 16-vs-64
    though both are one rung apart. Same formula as
    `bench_mpp_feature_decomposition.py`'s own `nearest_level`."""
    return min(rungs, key=lambda r: abs(math.log(ds) - math.log(r)))


def is_correct(row: dict) -> bool:
    """The method's OWN estimate lands on the true rung -- see the module
    docstring for why `estimated_ds`, not `chosen_ds`."""
    ds = num(row, 'estimated_ds')
    rung = num(row, 'rung')
    return ds is not None and rung is not None and nearest_rung(ds) == rung


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
              mpp_error_relative_p50=pctl(err, 50))
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
    `--native-only` where a rung some slides lack natively rests on fewer
    slides' worth of shots than one that every slide has.

    `classifier`/`reduction`/`loss` (2026-09-22) are carried through
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
    out = []
    for (dataset, method), grp in sorted(g.items()):
        accs = [r['level_accuracy'] for r in grp if not math.isnan(r['level_accuracy'])]
        errs = [r['mpp_error_relative_p50'] for r in grp
               if not math.isnan(r['mpp_error_relative_p50'])]
        out.append(dict(
            dataset=dataset, method=method, n_rungs=len(grp),
            mean_level_accuracy=sum(accs) / len(accs) if accs else float('nan'),
            mean_mpp_error_relative_p50=sum(errs) / len(errs) if errs else float('nan')))
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
         f'{"mean_acc":>10s}{"mean_mpp_err_p50":>18s}')
    for r in rows:
        print(f'  {r["dataset"] or "":{dw}s}{r["method"]:{mw}s}{r["n_rungs"]:>8d}'
             f'{r["mean_level_accuracy"]:>10.3f}{r["mean_mpp_error_relative_p50"]:>18.3f}')


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
             f'gigapath`, for the plot)')
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
                by_line[(recipe, r['loss'])].append(r)
        for (recipe, loss), line_rows in sorted(by_line.items()):
            line_rows.sort(key=lambda r: r['rung'])
            color = HEAD_RECIPE_COLORS.get(recipe, _INK_SECONDARY)
            # baseline (KnnEstMpp) has no loss at all -- dashed regardless,
            # same as before this change; every classifier row's linestyle
            # comes from ITS OWN loss instead (`head_recipe_of`'s own
            # docstring: colour is head identity, loss is the line).
            style = '--' if recipe == 'baseline' else LOSS_LINESTYLES.get(loss, '-')
            label = recipe if loss in ('', 'bal') else f'{recipe} ({loss})'
            ax.plot([r['rung'] for r in line_rows],
                    [r['level_accuracy'] for r in line_rows],
                    color=color, linewidth=2, marker='o', markersize=8,
                    linestyle=style, label=label)
            seen[(recipe, loss)] = (color, style, label)
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
                          markersize=6, linestyle=style, label=label)
              for color, style, label in seen.values()]
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
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv_path')
    ap.add_argument('--method', nargs='+', default=None,
                    help='keep only these method labels (see method_of)')
    ap.add_argument('--dataset', nargs='+', default=None)
    args = ap.parse_args()

    with open(args.csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    print(f'{args.csv_path}\n{len(rows)} rows')

    if args.dataset:
        rows = [r for r in rows if cell(r, 'dataset') in args.dataset]
    if args.method:
        rows = [r for r in rows if method_of(r) in args.method]
    if not rows:
        sys.exit('no rows left after filtering')

    view1 = per_slide_rung(rows)
    view2 = cross_slide_rung(rows)
    view3 = overall(rows)
    print_per_slide_rung(view1)
    print_cross_slide_rung(view2)
    print_overall(view3)

    print('\n' + BAR)
    print('4. PNG (accuracy vs rung, one file per dataset -- see plot_dataset)')
    print(BAR)
    csv_dir = os.path.dirname(os.path.abspath(args.csv_path)) or '.'
    csv_stem = os.path.splitext(os.path.basename(args.csv_path))[0]
    for dataset in sorted({r['dataset'] for r in view2}):
        stem = dataset.replace('/', '_')
        plot_dataset(view2, dataset,
                    os.path.join(csv_dir, f'{csv_stem}_{stem}.png'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
