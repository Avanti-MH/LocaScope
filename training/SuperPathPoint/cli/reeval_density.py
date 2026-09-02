#!/usr/bin/env python3
"""Re-score trained checkpoints at a MATCHED point budget, so margins compare.

    python training/SuperPathPoint/cli/reeval_density.py
    python training/SuperPathPoint/cli/reeval_density.py --budgets 160 --arms gray gray_pre

Outputs (in result/<SLURM_JOB_NAME or ReEvalSuperPathPoint>/):
    matched_density.csv     one row per (arm, budget, slide, rung)

No training, no weights change. This reads `superpathpoint_last.pt` out of each
arm's directory, runs the SAME validation pairs through it once, and cuts the
resulting probability maps several ways.

WHY THIS EXISTS: A MARGIN AT AN UNMATCHED DENSITY IS NOT A COMPARISON
----------------------------------------------------------------------
`margin = repeatability / decoy`, and the decoy asks what point DENSITY alone
buys -- the same points shifted past the NMS radius, matched anyway. Uniform
density over the `(2*nms_radius+1)^2 = 81` px match box in a `tile^2` image
gives

    decoy ~ 1 - exp(-81 N / tile^2)

so the decoy, and with it the ceiling `1/decoy`, is a function of N. The
2026-08-31 run came back with

    arm        pts/view   decoy   margin   ceiling
    gray            420   0.442     1.50      2.26
    rgb             420   0.420     1.57      2.38
    gray_pre        159   0.221     3.59      4.52
    rgb_pre         161   0.222     3.56      4.51

and those two groups are not two results, they are two operating points. The
`_pre` arms did not win by 2.4x; they were scored at a third of the density,
where the ceiling is twice as high.

WORSE, THE 420 IS NOT A POINT COUNT. It is `max_keypoints`, exactly, on every
tile -- and a count that lands on the cap to the integer is the cap selecting,
not the model. The reason is arithmetic: the detector is a 65-way softmax per
cell, so a model that has learnt nothing puts 1/65 = 0.015385 on every class,
and `detection_threshold` was 0.015. The threshold sits BELOW the value of
total ignorance. For an undertrained detector every cell passes it, NMS thins
the field to a few thousand, and the cap picks 420 of them by score.

SO THE FIX IS TO STOP USING A THRESHOLD HERE. `--budgets` cuts each view to
exactly the top N by score (`score_threshold=0` and `max_points=N`), which pins
the density by construction and makes the decoy the same quantity for every
arm. `--native` keeps the trained rule as one more row, so the number the
training run printed is still in the table rather than replaced by it.

WHAT THE BUDGET LADDER IS FOR
-------------------------------
One matched N would settle the four-arm comparison at one density and say
nothing about any other. The ladder is there because the interesting failure is
a crossing: an arm that wins at 40 points and loses at 320 is a model that
found a few good points and nothing else, which is a different object from one
that is uniformly better. A single N cannot tell those apart.

The rungs of the ladder are the LABEL densities, not round numbers -- the
per-slide, per-rung `n_kp` means of the corpus these students were trained on
run 3 to 527, and the two held-out slides sit at about 295 (BRACS_1598) and 40
(the Ki67 slide). A budget far outside that range scores the model somewhere it
was never asked to work.

`decoy_uniform` is in the CSV beside the measured decoy for the same reason a
decoy is there at all. It is the value the arithmetic above predicts for N
points spread evenly. Measured well ABOVE it means the points are clustered
(one region hoarding the budget); well BELOW means they are more evenly spread
than chance, which is what NMS at a hard radius produces once the budget
approaches the geometric limit. Either way the gap is a property of the model,
and it is a number that only means something because it has a prediction to sit
next to.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..', '..', 'utilities'),
           os.path.join(_HERE, '..')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _paths import RESULT_DIR, job_result_dir, setup_import_paths  # noqa: E402

setup_import_paths()

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

from ConfigIdentity import config_from_json                       # noqa: E402

import PreTileStore                                               # noqa: E402

from DsLadder import DEFAULT_RUNGS                                # noqa: E402
from SuperPoint.Datasets import PairDatasetConfig                 # noqa: E402
from SuperPoint.KeypointNet import KeypointNetConfig              # noqa: E402
from SuperPoint.Trainer import (_repeatability,                   # noqa: E402
                                _repeatability_row)

DEFAULT_TILE_ROOT = os.path.join(RESULT_DIR, 'cache', 'tiles')
DEFAULT_LABEL_ROOT = os.path.join(RESULT_DIR, 'cache', 'keypoint_labels')
#: NOT `job_result_dir('TrainSuperPathPoint')`. That function returns
#: `$SLURM_JOB_NAME or default_name`, which is right for an OUTPUT directory and
#: wrong for an input one: submitted as ReEvalSuperPathPoint it would look for
#: the checkpoints inside this job's own output directory, find nothing, and
#: exit saying no arm has a checkpoint. The training run's name is a literal
#: here because that is what it is -- a path someone else wrote.
DEFAULT_MODEL_ROOT = os.path.join(RESULT_DIR, 'TrainSuperPathPoint')

#: The four the training jobscript produces. Missing ones are skipped with a
#: line saying so rather than an error: re-scoring two arms is a normal thing to
#: want, and `--arms gray gray_pre` should not have to also delete a directory.
DEFAULT_ARMS = ('gray', 'rgb', 'gray_pre', 'rgb_pre')

#: The label corpus's own densities, rounded. Per-rung `n_kp` means over the 72
#: (slide, rung) stores run 3 to 527 with an overall mean of 146; the two
#: held-out slides average about 295 and 40. 40 and 320 are the ends of where
#: the labels actually live, 160 is near the middle, and 420 is carried because
#: it is the value the training run reported at -- dropping it would leave the
#: old numbers with nothing to be compared against.
DEFAULT_BUDGETS = (40, 80, 160, 320, 420)


def _channels_of(state) -> int:
    """How many input channels this checkpoint's trunk expects.

    Read off the FIRST CONV rather than out of `identity_json`, because the
    weights are what got built and the json is what was asked for. A checkpoint
    whose two answers disagree is exactly the case worth catching, and this way
    the shapes win.
    """
    key = 'backbone.stages.0.0.conv.weight'
    if key not in state:
        raise SystemExit(
            f'{key} is not in this checkpoint, so its input width cannot be '
            f'read. Keys start with: {sorted(state)[:3]}')
    return int(state[key].shape[1])


def _last_conv(state, prefix):
    """The deepest 4-d weight under `prefix`, by MODULE INDEX and not by name.

    `sorted()` on the key strings would put `head.10` before `head.2`, so the
    "last" layer of any head with ten modules would be the third one. Both
    heads are two modules deep today, which is exactly the kind of fact that
    makes the wrong sort survive until someone widens a head.
    """
    keys = [k for k in state if k.startswith(prefix)
            and k.endswith('.weight') and state[k].dim() == 4]
    if not keys:
        return None

    def depth(key):
        return [int(p) for p in key[len(prefix):].split('.') if p.isdigit()]

    return max(keys, key=depth)


def _cell_of(state) -> int:
    """`cell` from the detector's output width, which is `cell**2 + 1`."""
    key = _last_conv(state, 'detector.head.')
    if key is None:
        raise SystemExit(
            'this checkpoint has no detector.head convolution, so there is no '
            'cell to read and nothing to extract points from')
    out = int(state[key].shape[0])
    cell = int(round((out - 1) ** 0.5))
    if cell * cell + 1 != out:
        raise SystemExit(
            f'the detector emits {out} channels, which is not cell**2 + 1 for '
            f'any integer cell. This checkpoint is not a depth-to-space '
            f'detector and the extraction rule below does not apply to it')
    return cell


def _descriptor_dim_of(state) -> int:
    """0 when there is no descriptor head, which is MagicPoint and is loadable.

    Reading a missing head as the default 256 would build a descriptor the
    checkpoint has no weights for, and `strict=True` would then refuse a
    checkpoint that is perfectly fine.
    """
    key = _last_conv(state, 'descriptor.head.')
    return int(state[key].shape[0]) if key is not None else 0


def load_arm(path, device):
    """A checkpoint back into a `KeypointNet`, STRICT, plus what it was trained on.

    Strict for the reason `load_upstream` is strict: a partial load leaves the
    unmatched layers at their random init, the forward runs, the metric comes
    out low, and the run reads as "this arm is worse" rather than as "this arm
    was not loaded".
    """
    blob = torch.load(path, map_location='cpu', weights_only=False)
    state = blob['state_dict']
    net = KeypointNetConfig.wired(
        in_channels=_channels_of(state), cell=_cell_of(state),
        descriptor_dim=_descriptor_dim_of(state)).build(device)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net, blob.get('identity_json', {}) or {}, blob.get('history', [])


def _check_val_size(val_set, history, arm):
    """The set being re-scored must not be SMALLER than the one that trained.

    A cheap assertion in front of an expensive run, and it exists because the
    failure it catches produces a table rather than an error. When the held-out
    slides are recovered wrongly -- one slide instead of two, say -- everything
    downstream still works: the loader loads, the model runs, the margins come
    out, and the only sign is a pair count nobody was looking at.

    One-directional on purpose. The recorded `val/n_pairs` counts pairs that
    yielded points at all, so it can be LOWER than the dataset; it can never be
    higher unless this run is looking at less data than the training run did.
    """
    recorded = max((float(row.get('val/n_pairs', 0)) for row in history),
                   default=0.0)
    if recorded and len(val_set) < recorded - 0.5:
        raise SystemExit(
            f'{arm} validated on {recorded:.0f} pairs while training and this '
            f'run built only {len(val_set)}. Some of the held-out set is '
            f'missing -- most likely the slides were recovered wrongly -- and '
            f'every number below would be about the part that survived. Pass '
            f'--val-slides explicitly')


def _slides_from(recorded, args):
    """The held-out stems a checkpoint recorded, in either of the two formats.

    Since 2026-08-31 the writer uses `json.dumps`, which needs no undoing. What
    follows is for the checkpoints written BEFORE that -- including every arm of
    the 2026-08-31 run, which are the only detectors Stage B can currently use,
    so dropping the old path would strand them.

    UNDOING `','.join(stems)` WHEN A STEM CAN ITSELF CONTAIN COMMAS.

    `S1103627,G7E,110127` is one Ki67 slide, not three. `split(',')` turns the
    recorded pair into four fragments of which exactly one -- `BRACS_1598` --
    matches a real store, so the run scores ONE slide, prints `515 pairs`
    against the 1044 the training run validated on, and produces a table that
    is silently about half the held-out set. It does not error anywhere. The
    same comma cost an awk parse of the Ki67 CSV earlier in this project.

    So the fragments are not trusted. The candidates are the stems that
    actually exist under `--tiles-root`, and the reconstruction is VERIFIED:
    the chosen stems joined back the way the writer joined them must reproduce
    the recorded string exactly. A substring match alone could pick a stem that
    is a prefix of another; the round trip is what makes that impossible.
    """
    try:
        decoded = json.loads(recorded)
    except ValueError:
        decoded = None
    if isinstance(decoded, list) and all(isinstance(x, str) for x in decoded):
        return decoded

    stems = sorted({PreTileStore.load_meta(folder).wsi_stem
                    for folder in PreTileStore.find(args.tiles_root,
                                                    tile=int(args.tile))})
    picked = sorted(stem for stem in stems if stem in recorded)
    if ','.join(picked) != recorded:
        raise SystemExit(
            f'cannot recover the held-out slides from {recorded!r}. Rebuilding '
            f'it from the stores under {args.tiles_root} gave {picked}, which '
            f'joins back to {",".join(picked)!r}. A stem here contains commas, '
            f'so the recorded string is ambiguous -- pass --val-slides')
    return picked


def val_set_of(identity, args, channels):
    """The SAME held-out pairs the arm was validated on while it trained.

    Taken from the checkpoint's own `data` and `val_slides` when they are there,
    because the homography config decides what a pair IS -- re-scoring against
    a different set of warps would be a second experiment wearing the first
    one's name. `--val-slides` overrides, and a checkpoint without the fields
    falls back to the CLI defaults with a line saying so.
    """
    cfg = None
    if 'data' in identity:
        cfg = config_from_json(identity['data'])
        cfg = dataclasses.replace(cfg, in_channels=int(channels),
                                  balance='none', workers=args.workers)
    if cfg is None:
        print('  this checkpoint carries no `data` identity; falling back to '
              'PairDatasetConfig defaults, which may not be what it trained on',
              flush=True)
        cfg = PairDatasetConfig(tile=args.tile, in_channels=int(channels),
                                balance='none', workers=args.workers)

    slides = args.val_slides
    if slides is None:
        recorded = identity.get('val_slides', '')
        if not recorded:
            raise SystemExit(
                'this checkpoint does not record its held-out slides, so there '
                'is no way to re-score it on the same ones. Pass --val-slides')
        slides = _slides_from(recorded, args)
    return cfg.build(args.tiles_root, args.labels_root, wsi_stems=slides,
                     rungs=args.ds, ha_id=args.ha_id)


@torch.no_grad()
def score_arm(net, val_set, budgets, *, native, batch_size, workers):
    """Every budget on ONE pass of the data.

    The forward runs once per batch and each budget cuts the SAME probability
    maps. That is not only cheaper -- it means a difference between two budgets
    cannot be a difference between two draws of the augmentation.
    """
    from torch.utils.data import DataLoader                       # noqa: PLC0415

    loader = DataLoader(val_set, batch_size=int(batch_size), shuffle=False,
                        num_workers=int(workers), pin_memory=True,
                        persistent_workers=bool(workers))
    cfg = net.cfg
    shift = 2 * int(cfg.nms_radius) + 1

    # `_repeatability` takes the budget as an argument, and 0 means "the
    # config's own rule": threshold, then `max_keypoints`. So the native row is
    # not a second implementation, it is this one with the budget switched off.
    rules = [(f'top{int(n)}', int(n)) for n in budgets]
    if native:
        rules.append((f'native_thr{cfg.detection_threshold:g}', 0))

    # (rule, slide, ds) -> three parallel lists. The slide and the rung are
    # both here because the label density varies more BETWEEN rungs of one
    # slide (20 to 80 on the Ki67 slide) than between the slides at one rung.
    bucket = defaultdict(lambda: ([], [], [], []))

    for batch in loader:
        on_device = {k: v.to(net.device, non_blocking=True)
                     for k, v in batch.items() if torch.is_tensor(v)}
        prob = net(on_device['image']).prob_map.float().cpu().numpy()
        warped = net(on_device['warped_image']).prob_map.float().cpu().numpy()
        homography = batch['homography'].float().numpy()
        slide = batch['slide_index'].numpy()
        rung = batch['rung_index'].numpy()

        for name, budget in rules:
            hits, decoys, counts, avails, which = _repeatability(
                prob, warped, homography, cfg, shift, budget)
            for i, index in enumerate(which):
                for key in ((name, int(slide[index]), int(rung[index])),
                            (name, int(slide[index]), -1),
                            (name, -1, -1)):
                    got = bucket[key]
                    got[0].append(hits[i])
                    got[1].append(decoys[i])
                    got[2].append(counts[i])
                    got[3].append(avails[i])
    return bucket, [name for name, _ in rules]


def rows_from(bucket, names, *, arm, path, channels, init, val_set, tile):
    slides = list(getattr(val_set, 'slides', []))
    rungs = list(getattr(val_set, 'rungs', []))
    box = 81.0                                    # (2 * nms_radius + 1) ** 2

    out = []
    for name in names:
        for (rule, slide, rung), (hits, decoys, counts, avails) in sorted(
                bucket.items(), key=lambda kv: (kv[0][1], kv[0][2])):
            if rule != name:
                continue
            row = _repeatability_row(hits, decoys, counts, avails, prefix='')
            if not row:
                continue
            n = row['points_per_view']
            decoy = row['repeatability_decoy']
            out.append({
                'arm': arm, 'checkpoint': os.path.basename(path),
                'channels': channels, 'init': init, 'rule': rule,
                'slide': slides[slide] if slide >= 0 else '(all)',
                'ds': f'{rungs[rung]:g}' if rung >= 0 and rung < len(rungs) else '',
                'n_pairs': int(row['n_pairs']),
                'points_per_view': round(n, 1),
                'points_available': round(row['points_available'], 1),
                'repeatability': round(row['repeatability'], 4),
                'repeatability_decoy': round(decoy, 4),
                # What N points spread evenly would match by density alone.
                # Beside the measured value because a decoy with no prediction
                # next to it says whether the points repeat, not whether they
                # are anywhere in particular.
                'decoy_uniform': round(1.0 - np.exp(-box * n / (tile * tile)), 4),
                'repeatability_margin': round(row['repeatability_margin'], 4),
                'ceiling': round(1.0 / max(decoy, 1e-6), 3),
            })
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models-root', default=DEFAULT_MODEL_ROOT,
                    help='holds model_<tile>_<arm>/ directories')
    ap.add_argument('--arms', nargs='+', default=list(DEFAULT_ARMS))
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--epoch-tag', default='last',
                    help="which checkpoint: 'last', or e.g. 'epoch049'")
    ap.add_argument('--budgets', type=int, nargs='+', default=list(DEFAULT_BUDGETS),
                    help='matched point budgets. Each view is cut to exactly '
                         'this many points by score, so the decoy is the same '
                         'quantity for every arm')
    ap.add_argument('--no-native', dest='native', action='store_false',
                    help="drop the trained threshold's row from the table")
    ap.add_argument('--tiles-root', default=DEFAULT_TILE_ROOT)
    ap.add_argument('--labels-root', default=DEFAULT_LABEL_ROOT)
    ap.add_argument('--val-slides', nargs='+', default=None,
                    help='default: whichever the checkpoint says it held out')
    ap.add_argument('--ds', type=float, nargs='+', default=list(DEFAULT_RUNGS))
    ap.add_argument('--ha-id', default=None)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    bad = [n for n in args.budgets if n < 1]
    if bad:
        raise SystemExit(f'a budget must be at least 1 point, got {bad}')

    out_dir = args.out or job_result_dir('ReEvalSuperPathPoint')
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rows, pairs_seen = [], {}
    for arm in args.arms:
        path = os.path.join(args.models_root, f'model_{args.tile}_{arm}',
                            f'superpathpoint_{args.epoch_tag}.pt')
        if not os.path.exists(path):
            print(f'\n[{arm}] no {path}; skipped', flush=True)
            continue

        print(f'\n======== {arm} ========', flush=True)
        net, identity, history = load_arm(path, device)
        channels = net.cfg.backbone.in_channels
        init = identity.get('init', '?')
        print(f'  {path}\n  {channels} ch, init {init}, '
              f'{net.identity_id()}', flush=True)

        val_set = val_set_of(identity, args, channels)
        print(f'  val  {val_set.summary()}', flush=True)
        _check_val_size(val_set, history, arm)
        bucket, names = score_arm(net, val_set, args.budgets,
                                  native=args.native,
                                  batch_size=args.batch_size,
                                  workers=args.workers)
        rows += rows_from(bucket, names, arm=arm, path=path,
                          channels=channels, init=init, val_set=val_set,
                          tile=args.tile)
        pairs_seen[arm] = len(val_set)
        del net
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit(
            f'nothing scored. No arm of {args.arms} has a '
            f'superpathpoint_{args.epoch_tag}.pt under {args.models_root}')

    path = os.path.join(out_dir, 'matched_density.csv')
    keys = list(rows[0])
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved {path}', flush=True)

    _report(rows, pairs_seen, args)
    return 0


def _report(rows, pairs_seen, args):
    """The table, plus the three ways this run can be meaningless.

    Printed rather than left to the CSV because each of the three turns a
    number that looks fine into one that is not about what it says it is, and
    the CSV has no way to say so.
    """
    if len(set(pairs_seen.values())) > 1:
        print('\nWARNING  the arms saw different numbers of pairs: '
              f'{pairs_seen}. They were scored on different data, so the '
              'columns below are not a comparison.', flush=True)

    for rule in dict.fromkeys(r['rule'] for r in rows):
        print(f'\n{rule}')
        print(f"  {'arm':10s} {'slide':24s} {'pairs':>6s} {'pts':>6s} "
              f"{'avail':>7s} {'repeat':>7s} {'decoy':>7s} {'unif':>6s} "
              f"{'margin':>7s} {'ceiling':>8s}")
        for row in rows:
            if row['rule'] != rule or row['ds']:
                continue
            print(f"  {row['arm']:10s} {row['slide']:24s} {row['n_pairs']:6d} "
                  f"{row['points_per_view']:6.0f} "
                  f"{row['points_available']:7.0f} {row['repeatability']:7.3f} "
                  f"{row['repeatability_decoy']:7.3f} "
                  f"{row['decoy_uniform']:6.3f} "
                  f"{row['repeatability_margin']:7.3f} {row['ceiling']:8.2f}")

    # A budget that did not bind is the failure this whole run exists to avoid,
    # and it is silent: the row still prints a margin. `points_per_view` is the
    # instrument -- at a matched budget it must equal N exactly, because the
    # cut is "the top N by score" with the threshold at zero. Less than N means
    # NMS left fewer than N survivors on some tile, and those tiles were scored
    # at a lower density than the rest of the column.
    short = [r for r in rows if r['rule'].startswith('top') and not r['ds']
             and r['points_per_view'] < int(r['rule'][3:]) - 0.5]
    if short:
        print('\nWARNING  the budget did not bind on these rows -- NMS left '
              'fewer survivors than the budget asked for, so the density is '
              'NOT matched and the margins are not comparable:', flush=True)
        for row in short:
            print(f"    {row['arm']:10s} {row['slide']:24s} {row['rule']} "
                  f"got {row['points_per_view']:.1f}", flush=True)

    print('\nHOW TO READ THIS.  Compare arms DOWN a rule block, never across '
          'blocks: the ceiling is 1/decoy and the decoy is a function of the '
          'density, so a margin at top40 and one at top420 are two different '
          'scales.\n  One arm ahead on one slide and behind on the other is '
          'UNDECIDED (spec.md 1, fourth row), and two held-out slides cannot '
          'settle six pairwise comparisons.\n  `unif` is what N points spread '
          'evenly would score by density alone; measured far above it means '
          'the points are clustered.')
    print(f'\n  per (slide, rung) rows are in the CSV -- {len(args.ds)} rungs, '
          'and the label density varies more between rungs of one slide than '
          'between the slides.')


if __name__ == '__main__':
    sys.exit(main())
