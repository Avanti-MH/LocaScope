#!/usr/bin/env python3
'''Score saved checkpoints on the TEST split.

    python training/MppRoutingHead/cli/evaluate.py                  # every *_best.pt
    python .../evaluate.py --tag last
    python .../evaluate.py --weights .../gigapath_frozen_arcface_best.pt

Separate from `train.py` on purpose. The test split is every WSI of the eval
datasets that `train.py` did NOT hold out for val, and it is meant to be looked
at once a question is settled -- not once per training run. Putting it behind
its own entry point is what makes "did anyone score on test today" answerable.

A checkpoint carries everything needed to rebuild its model (see
`Checkpoints.save_checkpoint`), so this file takes no `--encoder`, no `--head`
and no head geometry: passing any of those would be a second source of truth
able to disagree with the weights.

The split comes from the recorded split, written once by
`utilities/cli/build_cache/make_split.py`.
Read rather than re-derived: a split recomputed from
`--seed` is one library version away from silently becoming a different
experiment, and then the "test" set contains slides the model selected on.

`--n-wsi` (5) slides from each dataset's test half, `--n-per-rung` (50)
positions on each rung of each -- so 5 x 6 x 50 = 1500 positions per dataset,
deeper per slide than val is and over fewer slides.

Two CSVs come out: `test_scores_<tag>.csv`, one row per (checkpoint, dataset),
and `test_predictions_<tag>.csv`, ONE ROW PER TILE with its `wsi_name`, `x`,
`y`, `rung`, richness `bucket`, `native` flag, ground truth and prediction. The
summary cannot say whether the errors are one bad slide, one richness bucket or
the resampled rungs; the per-tile file can.

NOTHING IS CACHED. The test corpus renders through the same `iterate_epoch`
training uses, with `split='eval'` so each position's pixels are seeded from
its own identity -- so two runs of this file on one checkpoint give the same
number without a corpus on disk.
'''
from __future__ import annotations

import argparse
import collections
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

# _paths holds the one definition of every package's sys.path entry
# (setup_import_paths) -- utilities/ goes on the path here, by hand, because
# that function is INSIDE it and this is the one step nothing else can do
# for this file. Same idiom every test_modules/cli entry point uses.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'utilities'))
from _paths import setup_import_paths                                  # noqa: E402
setup_import_paths()

import torch                                                        # noqa: E402

import _paths                                                       # noqa: E402
from training.MppRoutingHead.Datasets import (                      # noqa: E402
    RUNGS, add_cache_args, build_manifest, data_record, open_caches,
    read_label_of)
from AccessDatasets import list_names                                # noqa: E402
from training.MppRoutingHead.Runtime import (                       # noqa: E402
    build_from_checkpoint, encode_raw, head_parts, predict, rescore_by_rung,
    trunk_raw)


def test_rows(args, caches, out_dir, dataset_id: str) -> List:
    '''`--n-wsi` slides from the test half, `--n-per-rung` positions on each
    rung of each.

    THE FIRST `n_wsi` OF THE RECORDED ORDER, not a fresh draw.
    `split_wsi_names` shuffles before it splits and the recorded split keeps that
    order, so the test rows in the file are already randomised -- taking a
    prefix is a sample, and it is the SAME sample every time without a second
    seed to keep in step with the first. Which five were used is then readable
    off the rows' own `wsi_name` column and the sampler reports.

    The split is READ (`<dataset>#test`; it refuses when the split is missing): a split
    derived here could disagree with the one that selected the checkpoints,
    and the disagreement would show up as a good score.
    '''
    names = list_names(dataset=f'{dataset_id}#test',
                       split_job=caches.split_job)[:args.n_wsi]
    print(f'[test]  {dataset_id}: {len(names)} WSIs ({", ".join(names)})',
          flush=True)
    return build_manifest(
        dataset_id, masks=caches.masks, draw_job=caches.draw_job,
        report_dir=(out_dir / 'sampler_reports'
                    / f'{dataset_id.replace("/", "_")}_test'),
        tile_size=args.tile, n_per_rung=args.n_per_rung, seed=args.seed,
        wsi_names=names)


def find_weights(args, out_dir: Path) -> List[Path]:
    if args.weights:
        return [Path(w) for w in args.weights]
    found = sorted((out_dir / 'weights').glob(f'*_{args.tag}.pt'))
    if not found:
        raise FileNotFoundError(
            f'no *_{args.tag}.pt under {out_dir / "weights"} -- name one with '
            f'--weights, or point --out at the run that wrote them')
    return found


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', nargs='+', default=None,
                    help='checkpoint paths; default is every *_<tag>.pt under '
                         'the run directory')
    ap.add_argument('--tag', choices=['best', 'last', 'best_unweighted'],
                    default='best',
                    help="which checkpoint of each head, when --weights is "
                         "not given. 'best': highest POOLED val accuracy "
                         "(both eval datasets scored together, so the "
                         "larger one has more say). 'best_unweighted': "
                         "highest val accuracy averaged EQUALLY across "
                         "datasets -- see cli/train.py's val_report/"
                         "save_tagged for why these can pick different "
                         "epochs")
    ap.add_argument('--eval-datasets', nargs='+',
                    default=['bracs/test', 'ki67_with_photo'])
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--n-wsi', type=int, default=5,
                    help='slides taken from each dataset\'s test half')
    ap.add_argument('--n-per-rung', type=int, default=50,
                    help='positions per rung per slide')
    ap.add_argument('--wsi-group-size', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--encode-batch', type=int, default=256)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    # --seg: every checkpoint scored must have been trained under the same one
    add_cache_args(ap)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    from CpuBudget import CpuBudget                                 # noqa: PLC0415
    print(f'  {CpuBudget.for_job(workers=args.num_workers).apply().line()}', flush=True)

    device = torch.device(args.device)
    caches = open_caches(args, 'MppRoutingHead', device)
    out_dir = Path(args.out or _paths.job_result_dir('MppRoutingHead'))
    weights = find_weights(args, out_dir)
    print(f'{len(weights)} checkpoint(s), test split of '
          f'{", ".join(args.eval_datasets)}', flush=True)

    rows = {d: test_rows(args, caches, out_dir, d) for d in args.eval_datasets}
    caches.masks.close()
    out_rows: List[Dict] = []
    tile_rows: List[Dict] = []
    for path in weights:
        head, encoder, ckpt = build_from_checkpoint(path, device)
        frozen = bool(ckpt['frozen'])
        # num_prefix is 0 for the fine-tuned route because its spatial exit has
        # already dropped the prefix -- see `Features.trunk_raw`.
        num_prefix = 0 if not frozen else int(encoder.model_spec.num_prefix)
        # A mix_ head reads the blocks its checkpoint recorded; every other
        # head's `encoder_layers` is () and gets the plain tensor.
        raw_of = ((lambda p: encode_raw(encoder, p, args.encode_batch, device,
                                        layers=head.layers))
                  if frozen else (lambda p: trunk_raw(encoder, p, device)))
        # `tile_size` is task-specific, not part of the generic checkpoint
        # format -- `save_tagged` (cli/train.py) puts it under `extra`.
        tile = int(ckpt['extra']['tile_size'])
        if tile != args.tile:
            raise ValueError(
                f'{path.name} was trained at tile_size={tile} and --tile is '
                f'{args.tile}: scoring it at a different tile size measures a '
                f'different question, so this is a refusal rather than a warning')
        # Same refusal for the mask recipe: the test positions would come from
        # a different mask than the ones it learned on.
        seg = ckpt['args']['seg']
        if seg != args.seg:
            raise ValueError(
                f'{path.name} was trained with seg={seg!r} and --seg is '
                f'{args.seg!r}: the test positions would come from a different '
                f'mask than the ones it learned on')
        # Everything else the tiles are made of: sampler distribution, camera
        # templates, rungs, the code versions and the environment.
        from ConfigIdentity import record_diff                    # noqa: PLC0415
        stale = record_diff(ckpt['extra'].get('data'), data_record(args.tile, args.seg))
        if stale:
            raise ValueError(
                f'{path.name} was trained on other data than the test rows '
                f'would be made of: ' + '; '.join(stale) + '. Retrain it, or '
                f'score it with the code and environment it was trained under')

        head_name = ckpt['head_name']
        for dataset_id, drows in rows.items():
            scores, detail = predict(
                drows, {head_name: head}, raw_of, num_prefix, tile=tile,
                wsi_group_size=args.wsi_group_size,
                batch_size=args.batch_size, num_workers=args.num_workers)
            result = scores[head_name]
            # `level_accuracy` OVERWRITTEN with the mean of the six rungs'
            # own accuracies, NOT the pooled per-tile value `scores[...]`
            # still carries in every other key -- the same rule `cli/
            # train.py`'s `val_report` applies to VAL: RICHNESS's own coarse-rung
            # supply shortfall means a tile-pooled average is dominated by
            # whichever rungs happen to have the most test tiles, the same
            # `rescore_by_rung` finding val_report's own docstring cites.
            # Before this fix, `print_and_plot` below computed this SAME
            # corrected number via its own `overall_view` and only printed
            # it -- the number that got saved to `test_scores_<tag>.csv`
            # was the uncorrected one.
            per_rung = rescore_by_rung(detail[head_name])
            result['level_accuracy'] = sum(
                per_rung[rung]['level_accuracy'] for rung in RUNGS) / len(RUNGS)
            print(f'  {ckpt["encoder"]:12s} '
                  f'{"frozen" if frozen else "finetuned":10s} '
                  f'{head_name:12s} {dataset_id:17s} {result}', flush=True)
            # identity = dict(weights=path.name, encoder=ckpt['encoder'],
            #                 trunk='frozen' if frozen else 'finetuned',
            #                 head=head_name)
            # out_rows.append(dict(
            #     **identity, train_epoch=ckpt['epoch'],
            #     val_accuracy=ckpt['val']['level_accuracy'],
            #     test_dataset=dataset_id, **result))
            loss_kind = ckpt.get('args', {}).get('loss', 'bal')
            identity = dict(
                weights=path.name,
                encoder=ckpt['encoder'],
                trunk='frozen' if frozen else 'finetuned',
                head=head_name,
                loss_kind=loss_kind,
                # how the HEAD was trained; the test tiles themselves are
                # always read `pyramid` (the per-tile `native` column says
                # which of them came off a finer level)
                read_level=read_label_of(ckpt.get('args')),
            )
            out_rows.append(dict(
                **identity,
                train_epoch=ckpt['epoch'],
                val_accuracy=ckpt['val']['level_accuracy'],
                test_dataset=dataset_id,
                **result,
            ))
            tile_rows += [dict(**identity, **d) for d in detail[head_name]]

    status = write_csvs(out_dir, args.tag, out_rows, tile_rows)
    print_and_plot(out_dir, args.tag, tile_rows)
    return status


def write_csvs(out_dir: Path, tag: str, out_rows: List[Dict],
               tile_rows: List[Dict]) -> int:
    '''Two files, because they answer two questions.

    `test_scores_<tag>.csv` is one row per (checkpoint, dataset) -- the summary
    that goes next to `bench_stage1_mpp.py`'s numbers.

    `test_predictions_<tag>.csv` is one row per TILE: where it came from
    (`wsi_name`, `x`, `y`), what it is (`rung`, `bucket`, `native`) and what
    happened (`gt_class`/`gt_rung`, `pred_class`/`pred_rung`, `correct`). The
    summary cannot say whether the errors are one bad slide, one richness
    bucket, or the resampled rungs; this can, and at 1500 positions per dataset
    it is a few MB.
    '''
    written = []
    for name, table in (('scores', out_rows), ('predictions', tile_rows)):
        if not table:
            continue
        path = out_dir / f'test_{name}_{tag}.csv'
        with open(path, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
            wr.writeheader()
            wr.writerows(table)
        written.append((path, len(table)))
    for path, n in written:
        print(f'\n{path}  ({n} rows)')
    return 0 if written else 1


# ── analysis: per_slide_rung / cross_slide_rung / overall, printed + PNG ────
#
# Mirrors utilities/cli/metrics/analyze_stage1_metrics.py's THREE VIEWS on
# purpose -- same shape of question (does a checkpoint fail on one slide, on
# one rung, or overall), same reason to split by dataset (BRACS steps 4x per
# pyramid level, Ki67 steps 2x -- pooled numbers would average two different
# things) -- but NOT that file's code: this reads `tile_rows` (already in
# memory here, ONE ROW PER TILE, no `chosen_ds`/`estimated_mpp` at all) while
# that one reads a stage1_compare per-SHOT csv, and the two are owned
# separately on purpose (see the module docstring's "separate from train.py"
# reasoning, one level up: test scoring answers a different question than a
# method bake-off, on its own schedule). No new CSV -- these are read
# straight off `tile_rows`/`out_rows`, the exact rows `write_csvs` just wrote.

BAR = '=' * 78

#: A head is drawn by its THREE parts, one visual channel each, so that 14
#: heads never need 14 colours -- past eight, categorical hues stop being
#: tellable apart and start to hurt:
#:     colour   the classifier (`Runtime.head_parts`)  -- CLASSIFIER_COLORS
#:     marker   the reduction                           -- REDUCTION_MARKERS
#:     fill     hollow when the head mixes encoder layers (`mix_`)
#:     line     the loss                                -- LOSS_STYLES
#: A pair that differs in one part (`mlp_deep` / `attn_mlp_deep`) then shares
#: a colour and differs in shape, which is the comparison being made. Shape
#: and line also carry what colour carries, for a reader who cannot rely on
#: hue. The eight hues are the dataviz skill's adjacent-safe categorical
#: order (references/palette.md); `mlp_narrow` takes the orange that
#: `attn_linear` held when every head had its own colour.
CLASSIFIER_COLORS = {
    'linear':             '#2a78d6',   # slot 1 blue
    'mlp_narrow':         '#eb6834',   # slot 2 orange
    'mlp':                '#1baf7a',   # slot 3 aqua
    'mlp_deep':           '#eda100',   # slot 4 yellow
    'mlp_wide':           '#e87ba4',   # slot 5 magenta
    'mlp_deep_wide':      '#008300',   # slot 6 green
    'mlp_deep_residual':  '#4a3aa7',   # slot 7 violet
    'arcface':            '#e34948',   # slot 8 red
}
REDUCTION_MARKERS = {'fixed': 'o', 'attn': '^', 'clsattn': 's'}


def head_style(head: str) -> dict:
    """`plot` keyword arguments for one head NAME: colour, marker and fill."""
    classifier, reduction, mixes = head_parts(head)
    color = CLASSIFIER_COLORS.get(classifier, _INK_SECONDARY)
    return dict(color=color, marker=REDUCTION_MARKERS.get(reduction, 'o'),
                markerfacecolor=_SURFACE if mixes else color,
                markeredgecolor=color)


_INK, _INK_SECONDARY = '#0b0b0b', '#52514e'
_GRIDLINE, _AXIS, _SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'
LOSS_STYLES = {
    'bal': '-',
    'ord_a': '--',
    'ord_b': ':',
}

def method_of(row: dict) -> str:
    """`<encoder>+<head>` -- `test_predictions` has no separate `reduction`
    column (unlike stage1_compare's per-shot csv) because `head` already
    names it: 'attn_linear' IS reduction='attn', 'linear' IS reduction='fixed'
    -- see `Runtime.HEAD_CHOICES`."""
    # return f'{row["encoder"]}+{row["head"]}'
    # `+<read mode>` only off the default, so a pyramid method keeps the label
    # it had; a row from before the column existed is pyramid
    read = row.get('read_level') or 'pyramid'
    return (f'{row["encoder"]}+{row["head"]}+{row["loss_kind"]}'
            + ('' if read == 'pyramid' else f'+{read}'))


def _levels_off(row: dict):
    """abs(log2(pred_rung / gt_rung)) -- how many pyramid levels off the
    prediction is, the closest thing to stage1_compare's
    `mpp_error_relative` this schema supports: `test_predictions` has no
    continuous `estimated_mpp` at all (it is a discrete rung classification,
    not a continuous ds estimate), so there is no relative-mpp-error to
    compute -- this is an honest substitute, not the same number, and reads
    the same way CLAUDE.md's own note does: 0 = correct, 1 = one level off."""
    pred, gt = float(row['pred_rung']), float(row['gt_rung'])
    return abs(math.log2(pred) - math.log2(gt))


def score_group(rows: list) -> dict:
    correct = [int(r['correct']) for r in rows]
    levels = [_levels_off(r) for r in rows]
    native = [r['native'] in (True, 'True', '1') for r in rows]
    out = dict(n=len(rows),
              level_accuracy=sum(correct) / len(rows) if rows else float('nan'),
              mean_levels_off=sum(levels) / len(levels) if levels else float('nan'))
    for label, keep in (('native', native), ('resampled', [not n for n in native])):
        sub = [c for c, k in zip(correct, keep) if k]
        out[f'n_{label}'] = len(sub)
        out[f'level_accuracy_{label}'] = sum(sub) / len(sub) if sub else float('nan')
    return out


def per_slide_rung(tile_rows: list) -> list:
    g = collections.defaultdict(list)
    for r in tile_rows:
        g[(r['dataset'], r['wsi_name'], r['gt_rung'], method_of(r))].append(r)
    out = []
    for (dataset, wsi_name, rung, method), grp in sorted(
            g.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        out.append(dict(dataset=dataset, wsi_name=wsi_name, rung=rung,
                        method=method, **score_group(grp)))
    return out


def cross_slide_rung(tile_rows: list) -> list:
    g = collections.defaultdict(list)
    for r in tile_rows:
        g[(r['dataset'], r['gt_rung'], method_of(r))].append(r)
    out = []
    for (dataset, rung, method), grp in sorted(
            g.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n_slides = len({r['wsi_name'] for r in grp})
        out.append(dict(dataset=dataset, rung=rung, method=method,
                        n_slides=n_slides, **score_group(grp)))
    return out


def overall_view(tile_rows: list) -> list:
    """Mean of the per-rung accuracies (view 2, same dataset), not the
    pooled accuracy over every tile -- same reasoning as
    `analyze_stage1_metrics.py`'s `overall`: rungs are not sampled equally,
    and pooling would let the most-sampled rung decide the winner."""
    by_rung = cross_slide_rung(tile_rows)
    g = collections.defaultdict(list)
    for r in by_rung:
        g[(r['dataset'], r['method'])].append(r)
    out = []
    for (dataset, method), grp in sorted(g.items()):
        accs = [r['level_accuracy'] for r in grp if not math.isnan(r['level_accuracy'])]
        lvls = [r['mean_levels_off'] for r in grp if not math.isnan(r['mean_levels_off'])]
        out.append(dict(
            dataset=dataset, method=method, n_rungs=len(grp),
            mean_level_accuracy=sum(accs) / len(accs) if accs else float('nan'),
            mean_levels_off=sum(lvls) / len(lvls) if lvls else float('nan')))
    return out


def _col_width(rows: list, key: str, minimum: int, cap: int = None) -> int:
    """See analyze_stage1_metrics.py's `_col_width` for why this is computed
    per call rather than a hardcoded guess."""
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
    print(f'  {"dataset":{dw}s}{"wsi_name":{ww}s}{"rung":>6s}  {"method":{mw}s}'
         f'{"n":>5s}{"acc":>7s}{"levels_off":>11s}{"n_nat":>7s}{"acc_nat":>9s}'
         f'{"n_res":>7s}{"acc_res":>9s}')
    for r in rows:
        print(f'  {r["dataset"] or "":{dw}s}'
             f'{(r["wsi_name"] or "")[:ww - 2]:{ww}s}'
             f'{r["rung"]:>6g}  {r["method"]:{mw}s}{r["n"]:>5d}'
             f'{r["level_accuracy"]:>7.2f}{r["mean_levels_off"]:>11.3f}'
             f'{r["n_native"]:>7d}{r["level_accuracy_native"]:>9.2f}'
             f'{r["n_resampled"]:>7d}{r["level_accuracy_resampled"]:>9.2f}')


def print_cross_slide_rung(rows: list) -> None:
    print('\n' + BAR)
    print('2. CROSS-SLIDE, PER RUNG   (n_slides: how many slides contributed)')
    print(BAR)
    dw = _col_width(rows, 'dataset', 10)
    mw = _col_width(rows, 'method', 10)
    print(f'  {"dataset":{dw}s}{"rung":>6s}  {"method":{mw}s}{"n_slides":>9s}'
         f'{"n":>6s}{"acc":>7s}{"levels_off":>11s}')
    for r in rows:
        print(f'  {r["dataset"] or "":{dw}s}{r["rung"]:>6g}  {r["method"]:{mw}s}'
             f'{r["n_slides"]:>9d}{r["n"]:>6d}{r["level_accuracy"]:>7.2f}'
             f'{r["mean_levels_off"]:>11.3f}')


def print_overall(rows: list) -> None:
    print('\n' + BAR)
    print('3. OVERALL   (mean of the per-rung accuracies -- every rung weighted '
         'equally)')
    print(BAR)
    dw = _col_width(rows, 'dataset', 10)
    mw = _col_width(rows, 'method', 10)
    print(f'  {"dataset":{dw}s}{"method":{mw}s}{"n_rungs":>8s}'
         f'{"mean_acc":>10s}{"mean_levels_off":>17s}')
    for r in rows:
        print(f'  {r["dataset"] or "":{dw}s}{r["method"]:{mw}s}{r["n_rungs"]:>8d}'
             f'{r["mean_level_accuracy"]:>10.3f}{r["mean_levels_off"]:>17.3f}')


def encoder_of(method: str) -> str:
    return method.split('+', 1)[0]


def head_of(method: str) -> str:
    # return method.split('+', 1)[1]
    return method.split('+')[1]
    
def loss_of(method: str) -> str:
    return method.split('+')[2]


def read_of(method: str) -> str:
    """The training read mode a method label carries, '' for pyramid."""
    parts = method.split('+')
    return parts[3] if len(parts) > 3 else ''


#: A head trained off the default read mode is drawn at this opacity, so it
#: sits next to its pyramid twin -- same colour, marker, fill and line --
#: without being mistaken for it. The legend names the mode.
READ_ALPHA = 0.5

def plot_dataset(view2_rows: list, dataset: str, out_path) -> None:
    """One PNG: accuracy vs rung, one subplot per encoder -- see
    analyze_stage1_metrics.py's `plot_dataset` for the small-multiples/color-
    follows-the-entity reasoning, identical here. matplotlib is imported
    HERE, not at module level, for the same login-node reason that file
    gives; a missing import here costs only the plot, never the CSVs or the
    text tables above, which is why this is called last."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f'  [skip] matplotlib not available in this python -- no PNG '
             f'for {dataset} (the text tables above are unaffected)')
        return

    rows = [r for r in view2_rows if r['dataset'] == dataset]
    if not rows:
        return
    encoders = sorted({encoder_of(r['method']) for r in rows})

    fig, axes = plt.subplots(1, len(encoders), figsize=(4.2 * len(encoders), 3.6),
                             sharey=True, facecolor=_SURFACE)
    axes = [axes] if len(encoders) == 1 else list(axes)
    # seen_heads = []

    # for ax, encoder in zip(axes, encoders):
    #     ax.set_facecolor(_SURFACE)
    #     by_head = collections.defaultdict(list)
    #     for r in rows:
    #         if encoder_of(r['method']) == encoder:
    #             by_head[head_of(r['method'])].append(r)
    #     for head, head_rows in sorted(by_head.items()):
    #         head_rows.sort(key=lambda r: r['rung'])
    #         color = CLASSIFIER_COLORS.get(head, _INK_SECONDARY)
    #         ax.plot([r['rung'] for r in head_rows],
    #                 [r['level_accuracy'] for r in head_rows],
    #                 color=color, linewidth=2, marker='o', markersize=8, label=head)
    #         if head not in seen_heads:
    #             seen_heads.append(head)
    seen_methods = []

    for ax, encoder in zip(axes, encoders):
        ax.set_facecolor(_SURFACE)
        by_method = collections.defaultdict(list)

        for r in rows:
            if encoder_of(r['method']) == encoder:
                key = (head_of(r['method']), loss_of(r['method']),
                       read_of(r['method']))
                by_method[key].append(r)

        for (head, loss, read), method_rows in sorted(by_method.items()):
            method_rows.sort(key=lambda r: r['rung'])
            label = f'{head}+{loss}' + (f'+{read}' if read else '')
            ax.plot(
                [r['rung'] for r in method_rows],
                [r['level_accuracy'] for r in method_rows],
                **head_style(head),
                linestyle=LOSS_STYLES.get(loss, '-'),
                alpha=READ_ALPHA if read else 1.0,
                linewidth=2,
                markersize=8,
                label=label,
            )
            if (head, loss, read) not in seen_methods:
                seen_methods.append((head, loss, read))
        ax.set_xscale('log', base=2)
        ticks = sorted({r['rung'] for r in rows})
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{int(t)}' for t in ticks], color=_INK_SECONDARY)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(encoder, color=_INK, fontsize=11)
        ax.set_xlabel('rung (ds multiplier)', color=_INK_SECONDARY, fontsize=9)
        ax.grid(axis='y', color=_GRIDLINE, linewidth=0.8, zorder=0)
        for spine in ax.spines.values():
            spine.set_color(_AXIS)
        ax.tick_params(colors=_AXIS, labelcolor=_INK_SECONDARY)

    axes[0].set_ylabel('level accuracy', color=_INK_SECONDARY, fontsize=9)
    # handles = [plt.Line2D([0], [0], color=CLASSIFIER_COLORS.get(h, _INK_SECONDARY),
    #                       linewidth=2, marker='o', markersize=6, label=h)
    #           for h in seen_heads]
    handles = []
    for head, loss, read in seen_methods:
        handles.append(plt.Line2D(
            [0], [0],
            **head_style(head),
            linestyle=LOSS_STYLES.get(loss, '-'),
            alpha=READ_ALPHA if read else 1.0,
            linewidth=2,
            markersize=6,
            label=f'{head}+{loss}' + (f'+{read}' if read else ''),
        ))
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 4),
              bbox_to_anchor=(0.5, 0.0), frameon=False,
              fontsize=9, labelcolor=_INK_SECONDARY)
    fig.suptitle(f'evaluate.py test scores -- {dataset}', color=_INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.16, 1, 0.92))
    fig.savefig(out_path, dpi=130, facecolor=_SURFACE)
    plt.close(fig)
    print(f'  {out_path}')


def print_and_plot(out_dir: Path, tag: str, tile_rows: List[Dict]) -> None:
    if not tile_rows:
        print('\nno tile rows to analyze (every checkpoint failed before scoring)')
        return
    view1 = per_slide_rung(tile_rows)
    view2 = cross_slide_rung(tile_rows)
    view3 = overall_view(tile_rows)
    print()
    print_per_slide_rung(view1)
    print_cross_slide_rung(view2)
    print_overall(view3)

    print('\n' + BAR)
    print('4. PNG (accuracy vs rung, one file per dataset)')
    print(BAR)
    for dataset in sorted({r['dataset'] for r in view2}):
        stem = dataset.replace('/', '_')
        plot_dataset(view2, dataset,
                    out_dir / f'test_predictions_{tag}_{stem}.png')


if __name__ == '__main__':
    raise SystemExit(main())
