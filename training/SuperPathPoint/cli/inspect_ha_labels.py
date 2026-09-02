#!/usr/bin/env python3
"""spec.md 12 step 5: the decision point. Are these labels worth training on?

    python training/SuperPathPoint/cli/inspect_ha_labels.py
    python training/SuperPathPoint/cli/inspect_ha_labels.py --with-model --tiles 8

Outputs (in result/<SLURM_JOB_NAME or InspectHaLabels>/):
    ha_labels__<slide>_ds<d>.png
    ha_labels.csv
    ha_label_definitions.csv

NOT A TEST. It asserts nothing and cannot fail; it draws a figure for a human
and prints how to read it. `TestSuperPathPoint.sh` is where the assertions live.

THE QUESTION
-------------
The teacher is COCO-trained on natural images. spec.md 12 step 5 says out loud
that it may simply not work on H&E, and that the answer decides between
continuing, swapping in SIFT/Harris as the teacher, and doing upstream's stage 1
(MagicPoint on synthetic shapes) after all. Those three are not choosable now --
this is what produces the information.

FOUR OUTCOMES, AND ONLY ONE OF THEM IS "CONTINUE"
--------------------------------------------------
  A  points land on nuclei, gland boundaries and stromal texture, `n_kp` has a
     knee well below the cap, and a tile's two independent HA runs agree far
     better than the shifted decoy  ->  continue
  B  `n_kp` is near zero  ->  the teacher sees nothing here. Threshold first
     (it is 0.005 and the map may simply be flat), then the teacher
  C  `n_kp` sits at the cap on every tile  ->  it fires everywhere, which is
     what a detector does on texture it has no model for. The cap is then what
     selected, not the threshold, and the label is a grid of noise
  D  the two runs agree no better than the decoy  ->  the points are
     view-specific noise that N=100 did not average away. More views will not
     fix that; a different teacher might

THE AGREEMENT NUMBER IS SCORED AGAINST A DECOY, NOT A TOLERANCE
----------------------------------------------------------------
Two independent HA runs of the SAME tile differ only in which 99 homographies
were drawn. Their top-M point sets are compared, and the same comparison is made
against one of the sets shifted by `2 * nms_radius + 1` px -- far enough that no
point can match itself.

A margin over that decoy is meaningful with no ground truth at all, which is the
situation here: nobody has labelled keypoints on a WSI. An absolute repeatability
of "0.62" would need a reference to mean anything (CLAUDE.md, "prefer scoring
against deliberately wrong alternatives over scoring against a tolerance").

The cheap half of this file needs no model and no GPU: it reads the store and
plots `n_kp`, the score distribution and where the points sit. `--with-model` is
what adds the two-run agreement, and it costs `2 * N` forwards per inspected
tile -- minutes for eight tiles, which is why the default is eight.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..', '..', 'utilities'),
           os.path.join(_HERE, '..')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                                                 # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

from _paths import RESULT_DIR, job_result_dir, setup_import_paths  # noqa: E402

setup_import_paths()

from common import KeypointLabelStore                # noqa: E402
import PreTileStore
from common.KeypointLabelStore import points_from_prob             # noqa: E402
from PreTileStore import centre_crop                        # noqa: E402

DEFAULT_TILE_ROOT = os.path.join(RESULT_DIR, 'cache', 'tiles')
DEFAULT_LABEL_ROOT = os.path.join(RESULT_DIR, 'cache', 'keypoint_labels')

#: Every string a reader will see. Rewritten on every run so the definitions
#: cannot drift from the code that produced them (ClaudeRules section 12).
DEFINITIONS = [
    ('n_kp', 'points kept for one tile: above the score threshold, after NMS '
             'and the border cut, capped at points_per_megapixel * area'),
    ('cap', 'the per-tile maximum. A MEMORY bound, not a decision about what a '
            'keypoint is -- see at_cap'),
    ('at_cap', 'tiles whose n_kp reached the cap. Every one of them was cut by '
               'the cap rather than by the threshold, which is not recoverable '
               'without re-running HA'),
    ('score', 'the aggregated probability at a kept point: mean over the views '
              'that could see it, weighted by coverage (spec.md 3.1)'),
    ('kp_count', 'how many of the N views could see that pixel at all. Small '
                 'near the tile edge, and the reason a zero there is not the '
                 'same fact as a zero in the middle'),
    ('agreement', 'fraction of run A\'s top-M points that have a run-B point '
                  'within nms_radius. Two independent draws of the same N '
                  'homographies on the same tile'),
    ('decoy agreement', 'the same fraction against run B shifted by '
                        '2*nms_radius+1 px. The floor that agreement has to '
                        'beat for the points to be about the tissue'),
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels-root', default=DEFAULT_LABEL_ROOT)
    ap.add_argument('--tiles-root', default=DEFAULT_TILE_ROOT)
    ap.add_argument('--wsi-stem', nargs='*', default=None)
    ap.add_argument('--ds', type=float, nargs='*', default=None)
    ap.add_argument('--examples', type=int, default=3,
                    help='tiles drawn with their points, per label set')
    ap.add_argument('--with-model', action='store_true',
                    help='also run HA twice per tile for the agreement number. '
                         'Loads the teacher; costs 2N forwards per tile')
    ap.add_argument('--tiles', type=int, default=8,
                    help='tiles for the agreement measure')
    ap.add_argument('--num', type=int, default=100,
                    help='N for those runs. Match what made the labels')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    out_dir = args.out or job_result_dir('InspectHaLabels')
    os.makedirs(out_dir, exist_ok=True)

    paths = KeypointLabelStore.find(args.labels_root)
    if not paths:
        print(f'no labels under {args.labels_root}. Run cli/make_ha_labels.py '
              f'first.')
        return 1

    rows = []
    for path in paths:
        meta = KeypointLabelStore.load_meta(path)
        if args.wsi_stem and meta.wsi_stem not in args.wsi_stem:
            continue
        if args.ds and not any(abs(meta.ds - d) < 1e-6 for d in args.ds):
            continue

        batch, meta = KeypointLabelStore.load(path)
        print(f'\n{meta.wsi_stem}  ds {meta.ds:g}  tile {meta.tile}', flush=True)
        print(f'  {len(batch)} tiles   n_kp mean {meta.mean_n_kp:.0f}  '
              f'min {batch.n_kp.min()}  max {batch.n_kp.max()}  '
              f'cap {batch.cap}  at-cap {batch.at_cap}', flush=True)

        folder = _pretiles_for(args.tiles_root, meta)
        agreement = (_agreement(folder, meta, args) if args.with_model and folder
                     else {})
        figure = _draw(batch, meta, folder, agreement, out_dir, args)
        print(f'  figure -> {os.path.basename(figure)}', flush=True)

        row = {'wsi_stem': meta.wsi_stem, 'ds': meta.ds, 'tile': meta.tile,
               'n_tiles': len(batch), 'mean_n_kp': round(meta.mean_n_kp, 1),
               'median_n_kp': float(np.median(batch.n_kp)),
               'min_n_kp': int(batch.n_kp.min()),
               'max_n_kp': int(batch.n_kp.max()),
               'cap': batch.cap, 'at_cap': batch.at_cap,
               'score_threshold': meta.score_threshold,
               'ha_id': meta.ha_id}
        row.update(agreement)
        rows.append(row)

    if not rows:
        print('nothing matched the filters')
        return 1

    summary = os.path.join(out_dir, 'ha_labels.csv')
    with open(summary, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, 'ha_label_definitions.csv'), 'w',
              newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['term', 'means'])
        writer.writerows(DEFINITIONS)
    print(f'\nSaved {summary}   ({len(rows)} label sets)')
    _verdict(rows, args)
    return 0


def _pretiles_for(root, meta):
    """The pre-tile store these labels were made from, or None.

    Matched on `pretile_id`, not on (slide, rung): two extractions of the same
    slide and rung differ in seed or sampler_id, and drawing the labels of one
    over the images of the other would look almost right.
    """
    for folder in PreTileStore.find(root, wsi_stem=meta.wsi_stem,
                                    tile=int(meta.tile)):
        if PreTileStore.load_meta(folder).cfg_hash() == meta.pretile_id:
            return folder
    print(f'  no pre-tile store with id {meta.pretile_id}; drawing without '
          f'images', flush=True)
    return None


def _agreement(folder, meta, args):
    """Two independent HA runs per tile, scored against a shifted decoy."""
    import torch                                                  # noqa: PLC0415

    from SuperPoint.HomographicAdaptation import HaConfig         # noqa: PLC0415
    from SuperPoint.Teacher import TeacherConfig                  # noqa: PLC0415

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ha = HaConfig(num=args.num).build(TeacherConfig().build(device))

    records = PreTileStore.load_index(folder)[:int(args.tiles)]
    shift = 2 * int(meta.nms_radius) + 1
    hits, decoys = [], []
    for record in records:
        pre = PreTileStore.read_tile(folder, record)
        sets = []
        for seed in (11, 22):        # two draws, not two seeds of one draw
            result = ha.run(pre, meta.tile, rng=np.random.default_rng(seed),
                            factor=None)
            xy, _, _ = points_from_prob(
                result.prob, result.counts,
                score_threshold=meta.score_threshold,
                nms_radius=meta.nms_radius, border=meta.border,
                max_points=_cap_of(meta))
            sets.append(xy)
        hits.append(_match_rate(sets[0], sets[1], meta.nms_radius))
        decoys.append(_match_rate(sets[0], sets[1] + shift, meta.nms_radius))

    if not hits:
        return {}
    agreement, decoy = float(np.mean(hits)), float(np.mean(decoys))
    print(f'  agreement {agreement:.3f}  vs decoy {decoy:.3f}   '
          f'({agreement / max(decoy, 1e-6):.1f}x)', flush=True)
    return {'agreement': round(agreement, 4), 'decoy_agreement': round(decoy, 4),
            'agreement_tiles': len(hits)}


def _cap_of(meta):
    """The cap these labels were written with, so the re-run is cut the same way.
    A store from before `cap` was recorded has 0 there, which means no cap."""
    return int(meta.cap) if meta.cap else None


def _match_rate(a: np.ndarray, b: np.ndarray, radius: int) -> float:
    """Fraction of `a` with a `b` point within `radius`, in the max-norm.

    Max-norm and not Euclidean because that is the neighbourhood NMS itself
    suppresses over -- two points closer than the NMS radius cannot both survive
    one run, so treating them as the same point across runs is the consistent
    reading.
    """
    if len(a) == 0 or len(b) == 0:
        return 0.0
    delta = np.abs(a[:, None, :].astype(np.int32) - b[None, :, :].astype(np.int32))
    return float((delta.max(axis=2).min(axis=1) <= radius).mean())


def _draw(batch, meta, folder, agreement, out_dir, args):
    """One figure per label set: examples, n_kp, scores, coverage."""
    n_examples = min(int(args.examples), len(batch))
    fig, axes = plt.subplots(2, max(n_examples, 3),
                             figsize=(4.2 * max(n_examples, 3), 8.6),
                             squeeze=False)

    records = PreTileStore.load_index(folder) if folder else []
    by_position = {(r.x, r.y): r for r in records}

    for i in range(max(n_examples, 3)):
        ax = axes[0][i]
        if i >= n_examples:
            ax.axis('off')
            continue
        record = by_position.get((int(batch.tile_x[i]), int(batch.tile_y[i])))
        if folder and record is not None:
            image = centre_crop(PreTileStore.read_tile(folder, record),
                                int(meta.tile))
            ax.imshow(image)
        else:
            ax.set_facecolor('0.9')
        xy = batch.points_of(i)
        score = batch.scores_of(i).astype(np.float32)
        if len(xy):
            ax.scatter(xy[:, 0], xy[:, 1], s=6, c=score, cmap='autumn',
                       edgecolors='none')
        ax.set_title(f'tile {i}  n_kp {len(xy)}', fontsize=9)
        ax.set_xlim(0, meta.tile)
        ax.set_ylim(meta.tile, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    # n_kp: the shape that separates outcome A from C.
    ax = axes[1][0]
    ax.hist(batch.n_kp, bins=40, color='0.35')
    ax.axvline(batch.cap, color='crimson', lw=1.5,
               label=f'cap {batch.cap}')
    ax.set_xlabel('n_kp per tile')
    ax.set_ylabel('tiles')
    ax.set_title('a knee below the cap is a threshold selecting;\n'
                 'a pile at the cap is the cap selecting', fontsize=9)
    ax.legend(fontsize=8)

    # Scores: where the threshold sits in the distribution it is cutting.
    ax = axes[1][1]
    scores = np.concatenate([batch.scores_of(i).astype(np.float32)
                             for i in range(len(batch))]) if len(batch) else \
        np.zeros(0)
    if len(scores):
        ax.hist(scores, bins=50, color='0.35', log=True)
    ax.axvline(meta.score_threshold, color='crimson', lw=1.5,
               label=f'threshold {meta.score_threshold:g}')
    ax.set_xlabel('score of a kept point')
    ax.set_title('everything below the line was already cut', fontsize=9)
    ax.legend(fontsize=8)

    # Coverage, and the agreement bars if they were measured.
    ax = axes[1][2]
    if agreement:
        ax.bar(['two runs', 'shifted decoy'],
               [agreement['agreement'], agreement['decoy_agreement']],
               color=['seagreen', '0.6'])
        ax.set_ylim(0, 1)
        ax.set_title(f'agreement over {agreement["agreement_tiles"]} tiles\n'
                     'the bar on the left has to be far higher', fontsize=9)
    else:
        counts = np.concatenate([batch.counts_of(i).astype(np.float32)
                                 for i in range(len(batch))]) if len(batch) \
            else np.zeros(0)
        if len(counts):
            ax.hist(counts, bins=40, color='0.35')
        ax.set_xlabel('kp_count (views that could see the point)')
        ax.set_title('points seen by very few views are edge points,\n'
                     'and their score is an opinion not a consensus', fontsize=9)

    for extra in range(3, max(n_examples, 3)):
        axes[1][extra].axis('off')

    fig.suptitle(f'{meta.wsi_stem}  ds {meta.ds:g}  tile {meta.tile}   '
                 f'HA {meta.ha_id}   N views aggregated, threshold '
                 f'{meta.score_threshold:g}', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    path = os.path.join(out_dir,
                        f'ha_labels__{meta.wsi_stem}_ds{meta.ds:g}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _verdict(rows, args):
    """Print which of the four outcomes the numbers point at. Not a decision --
    the figure is what decides, and this is what to look at first."""
    print('\nRead this against the four outcomes in the file docstring:')
    for row in rows:
        flags = []
        if row['median_n_kp'] < 5:
            flags.append('B: almost no points')
        if row['at_cap'] > 0.1 * row['n_tiles']:
            flags.append('C: the cap is selecting, not the threshold')
        if 'agreement' in row and row['agreement'] < 2 * row['decoy_agreement']:
            flags.append('D: two runs agree barely better than the decoy')
        state = '; '.join(flags) if flags else 'A: nothing here says stop'
        print(f'  {row["wsi_stem"]:<16} ds {row["ds"]:<5g} {state}')
    if not args.with_model:
        print('\nOutcome D cannot be seen without --with-model: it is the one '
              'that needs the teacher run twice.')


if __name__ == '__main__':
    sys.exit(main())
