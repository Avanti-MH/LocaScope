#!/usr/bin/env python3
"""spec.md 12 step 6: train a student on the HA labels.

    python training/SuperPathPoint/cli/train_superpathpoint.py --channels 1
    python training/SuperPathPoint/cli/train_superpathpoint.py --channels 3

Outputs (in result/<SLURM_JOB_NAME or TrainSuperPathPoint>/):
    superpathpoint_epoch###.pt, superpathpoint_last.pt
    train_history.csv

argparse, wiring, and printed progress. Everything that decides anything is in
`SuperPoint/KeypointNet.py`, `SuperPoint/Losses.py`, `SuperPoint/Datasets.py`
and `SuperPoint/Trainer.py`.

TWO STUDENTS, ONE SET OF LABELS
---------------------------------
`--channels 1` and `--channels 3` are `model_256_gray` and `model_256_rgb`
(spec.md 13). They share the HA labels exactly, because HA produces COORDINATES
and a coordinate does not care how many channels the image had. What differs is
one number in the backbone and everything the model can learn to use.

What the pair can measure is repeatability on the held-out slides, reported per
slide. What it CANNOT measure is cross-stain generalisation: both stains are in
the training split, so the held-out number says "an unseen SLIDE", not "an
unseen stain" (spec.md 6.5). Until the leave-one-stain-out arm exists, the words
"cross-stain" do not belong in a conclusion drawn from this run.

THE SPLIT IS BY SLIDE AND IT IS NOT A FLAG WITH A DEFAULT
-----------------------------------------------------------
`--train-slides` and `--val-slides` both default to spec.md 6.5's split, and the
held-out pair is `BRACS_1936` and `S1151088`. Tiles of one slide share a stain
batch, a scanner, a section thickness and a tissue donor, so a random split by
TILE would leave every validation tile with thousands of same-slide neighbours
in training -- the standard leakage in this field, and it inflates every number
without raising anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..', '..', 'utilities'),
           os.path.join(_HERE, '..')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _paths import RESULT_DIR, job_result_dir, setup_import_paths  # noqa: E402

setup_import_paths()

import torch                                                      # noqa: E402

from ConfigIdentity import config_json                            # noqa: E402

from DsLadder import DEFAULT_RUNGS                          # noqa: E402
from common.HomographyConfig import HomographyConfig               # noqa: E402
from SuperPoint.Datasets import BALANCE_MODES, PairDatasetConfig   # noqa: E402
from SuperPoint.KeypointNet import (KeypointNetConfig,            # noqa: E402
                                    load_upstream)
from SuperPoint.Teacher import SuperPointTeacher                  # noqa: E402
from SuperPoint.Losses import SuperPointLossConfig                 # noqa: E402
from SuperPoint.Trainer import TrainerConfig                       # noqa: E402

DEFAULT_TILE_ROOT = os.path.join(RESULT_DIR, 'cache', 'tiles')
DEFAULT_LABEL_ROOT = os.path.join(RESULT_DIR, 'cache', 'keypoint_labels')

#: spec.md 6.5. `BRACS_1228` is deliberately in TRAIN: it is the slide
#: `SlideWinTest`, `BenchMarkV2` and `SlidewinPooling` all ran on, so the
#: existing SIFT and retrieval numbers are about it -- which makes it a sanity
#: check that can be asked at any time without touching the held-out pair.
#: Five per stain, chosen so that each addition answers something the set
#: could not:
#:   BRACS_1579  Group_BT/Type_N   -- Group_BT was absent entirely, and normal
#:                                    tissue is the furthest in architecture
#:                                    from ADH / FEA / DCIS
#:   BRACS_1284  Group_MT/Type_IC  -- the largest type, and invasive carcinoma
#:                                    is unlike the other four
#:   S1103520 (110126), S1140701 (111018) -- the three Ki67 slides already here
#:                                    are 110208, 110208 and 111220, i.e. two
#:                                    batches. Scanner and batch drift is a real
#:                                    axis and it was almost unsampled.
#:
#: Five and five is also what first makes the leave-one-stain-out arm possible
#: (spec.md 13): with both stains in TRAIN the held-out number says "an unseen
#: slide", never "an unseen stain".
TRAIN_SLIDES = ('BRACS_1228', 'BRACS_1476', 'BRACS_1936',
                'BRACS_1579', 'BRACS_1284',
                'S1104233,G7E,110208', 'S1104360,G7E,110208',
                'S1151088,G7E,111220', 'S1103520,G7E,110126',
                'S1140701,G7E,111018')

#: Held out from OUTSIDE the ten, one per stain. `BRACS_1598` is the second
#: slide of a type that is in TRAIN, so what it measures is "a new slide of a
#: seen type" rather than confounding slide and type in one number.
#: `S1103627` is the first of the Ki67 remainder once the two slides with
#: scanner holes are excluded -- `S1137178` (spec.md 6.5) and `S1103037`
#: (SafeSlide.py:33). A hole is filled with the background colour, and the
#: straight edge that leaves is a perfect corner: high-scoring false points
#: that look exactly like a model having learned something.
VAL_SLIDES = ('BRACS_1598', 'S1103627,G7E,110127')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tiles-root', default=DEFAULT_TILE_ROOT)
    ap.add_argument('--labels-root', default=DEFAULT_LABEL_ROOT)
    ap.add_argument('--tile', type=int, default=256,
                    help='v1 is 256. 512 and 1024 are separate models')
    ap.add_argument('--channels', type=int, default=1, choices=(1, 3),
                    help='1 = model_256_gray, 3 = model_256_rgb. Same labels')
    ap.add_argument('--cell', type=int, default=8,
                    help='detector cell in input pixels. NOT the backbone '
                         'stride -- see common/Interfaces.py')
    ap.add_argument('--descriptor-dim', type=int, default=256,
                    help='0 trains the detector alone, which is MagicPoint and '
                         'is not what spec.md 3.1 asks for')
    ap.add_argument('--train-slides', nargs='+', default=list(TRAIN_SLIDES))
    ap.add_argument('--val-slides', nargs='+', default=list(VAL_SLIDES))
    ap.add_argument('--ds', type=float, nargs='+', default=list(DEFAULT_RUNGS))
    ap.add_argument('--ha-id', default=None,
                    help='which round of labels. Default: whichever is the only '
                         'one present, and an error if there are two')
    ap.add_argument('--balance', default='none', choices=BALANCE_MODES,
                    help="rung balance. 'none' since the 3b probe of "
                         '2026-08-27: the worst cell is 18 disjoint positions '
                         'at ds 32, so align-min would truncate every rung to '
                         '18 and delete the ladder rather than flatten it. The '
                         "earlier 1784-of-2000 that made align-min look cheap "
                         'was the tissue gate exhausting a budget, not a count '
                         'of positions')
    ap.add_argument('--homography', default='rotation',
                    choices=('rotation', 'full'),
                    help="which augmentation the training pairs are drawn "
                         "with. 'rotation' since 2026-08-31: the query is a "
                         "microscope photograph of a flat slide at an "
                         "arbitrary angle, stage 1 has already fixed the "
                         "scale, and perspective on a coverslip is small. "
                         "'full' is upstream's thirteen options, which is what "
                         "the LABELS were voted on -- so 'rotation' asks the "
                         "student to be invariant to LESS than its teacher "
                         "was, which is the safe direction")
    ap.add_argument('--val-budget', type=int, default=200,
                    help='points per view the repeatability is measured at, '
                         'top-N by score. A budget and not a threshold: the '
                         'decoy rises with density, so two models cut by one '
                         'threshold are on two different scales')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4,
                    help="upstream's SuperPoint LR, constant, no schedule")
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-amp', dest='amp', action='store_false')
    ap.add_argument('--wandb-project', default='superpathpoint')
    ap.add_argument('--wandb-mode', default='online',
                    choices=('online', 'offline', 'disabled'))
    ap.add_argument('--pretrained', action='store_true',
                    help='initialise from upstream SuperPoint v6 instead of at '
                         'random. For the grayscale student this is SELF-'
                         'DISTILLATION -- the labels were produced by those '
                         'same weights (spec.md 13) -- so the arm answers '
                         '"does a second round on its own HA labels beat '
                         'learning them from scratch", NOT "does pretraining '
                         'help"')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    overlap = set(args.train_slides) & set(args.val_slides)
    if overlap:
        raise SystemExit(
            f'{sorted(overlap)} is in both splits. Held-out means it takes no '
            f'part in training or in tuning; a slide in both makes every '
            f'validation number an inflated one, with nothing to say so')

    tag = 'gray' if args.channels == 1 else 'rgb'
    if args.pretrained:
        tag += '_pre'
    out_dir = args.out or os.path.join(
        job_result_dir('TrainSuperPathPoint'), f'model_{args.tile}_{tag}')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = KeypointNetConfig.wired(
        in_channels=args.channels, cell=args.cell,
        descriptor_dim=args.descriptor_dim).build(device)
    if args.pretrained:
        # STRICT, inside `load_upstream`. A partial load leaves the unmatched
        # layers random, trains, converges, and reads as "pretraining did not
        # help" rather than as "pretraining did not happen".
        state, weights = SuperPointTeacher.weights_state_dict()
        load_upstream(net, state)
        print(f'initialised from {weights}', flush=True)
    loss = SuperPointLossConfig().build()
    print(net.summary(), flush=True)

    # ROTATION ONLY, and it is narrower than the sampler that made the labels.
    # That direction is the safe one: `HomographyConfig`'s docstring warns
    # against the two configs drifting because a student drawn from a WIDER
    # distribution than its teacher is asked to be invariant to transforms the
    # teacher never voted on. Here the teacher voted on thirteen options and
    # the student is asked about one of them.
    #
    # `PRE_TILE_FACTOR = 3` still holds with room to spare: the 2000-draw
    # calibration needed 2.702 for the full sampler, and rotation alone needs
    # sqrt(2) / patch_ratio = 1.66.
    homography = (HomographyConfig() if args.homography == 'full' else
                  HomographyConfig(perspective=False, scaling=False,
                                   translation=False, rotation=True))
    data_cfg = PairDatasetConfig(tile=args.tile, in_channels=args.channels,
                                 balance=args.balance, seed=args.seed,
                                 workers=args.workers, homography=homography)
    train_set = data_cfg.build(args.tiles_root, args.labels_root,
                               wsi_stems=args.train_slides, rungs=args.ds,
                               ha_id=args.ha_id)
    # The validation set takes `balance='none'`: balancing the held-out set
    # would change what the reported number is a number ABOUT, and the point of
    # it is to describe the data as it is.
    val_set = PairDatasetConfig(
        tile=args.tile, in_channels=args.channels, balance='none',
        seed=args.seed, workers=args.workers,
        homography=data_cfg.homography).build(
            args.tiles_root, args.labels_root, wsi_stems=args.val_slides,
            rungs=args.ds, ha_id=args.ha_id)
    print(f'train  {train_set.summary()}', flush=True)
    print(f'val    {val_set.summary()}', flush=True)

    trainer = TrainerConfig(
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        workers=args.workers, amp=args.amp, val_budget=args.val_budget,
        wandb_project=args.wandb_project, wandb_mode=args.wandb_mode,
        run_name=f'model_{args.tile}_{tag}').build(
            net, loss, train_set, val_set, out_dir,
            # Everything the checkpoint would otherwise not be able to say
            # about itself. `ha_id` is the important one: round 2 turns this
            # checkpoint into a teacher, and a teacher that cannot name the
            # labels it was trained on is indistinguishable from the round
            # before it -- which is what LabelMeta.ha_id exists to separate.
            extra_identity={
                'data': config_json(data_cfg),
                'loss': config_json(loss.cfg),
                # json.dumps AND NOT ','.join. `S1103627,G7E,110127` is ONE
                # Ki67 slide -- the comma is part of the name, because the stem
                # is the WSI's filename. Joining stems with a comma produces a
                # string that cannot be split back, and the failure is silent:
                # `reeval_density` read one of these, recovered a single slide
                # of the two, and printed a complete table over 515 pairs
                # instead of 1044 with nothing raising. The same comma had
                # already broken an awk parse of the Ki67 CSV.
                #
                # `extra_identity` is not hashed into `identity_id`
                # (`save_checkpoint` takes that from the net), so changing the
                # format here re-hashes nothing.
                'train_slides': json.dumps(sorted(args.train_slides)),
                'val_slides': json.dumps(sorted(args.val_slides)),
                'ha_id': args.ha_id or '(the only one present)',
                'init': 'superpoint-v6' if args.pretrained else 'random',
            })

    history = trainer.run()

    summary = os.path.join(out_dir, 'train_history.csv')
    if history:
        keys = sorted({k for row in history for k in row})
        with open(summary, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)
        print(f'\nSaved {summary}')
    print(f'checkpoints -> {out_dir}')

    last = history[-1] if history else {}
    if 'val/repeatability_margin' in last:
        # PER SLIDE, because the criterion in spec.md 1 is "one of them wins on
        # BOTH slides" and an averaged number cannot answer it. The point count
        # is printed beside the margin because the decoy's rate rises with
        # density: without N, a margin that moved cannot be told from a model
        # that changed how many points it emits.
        print('\nval, per slide -- the MARGIN is the number, and it is '
              'bounded above by 1/decoy:')
        print(f"  {'slide':24s} {'pairs':>6s} {'pts/view':>9s} "
              f"{'avail':>7s} {'repeat':>7s} {'decoy':>7s} {'margin':>7s} "
              f"{'ceiling':>8s}")
        for stem in sorted(args.val_slides) + ['']:
            prefix = f'val/{stem}/' if stem else 'val/'
            if f'{prefix}repeatability_margin' not in last:
                continue
            decoy = last[f'{prefix}repeatability_decoy']
            print(f"  {stem or '(both)':24s} "
                  f"{last[f'{prefix}n_pairs']:6.0f} "
                  f"{last[f'{prefix}points_per_view']:9.0f} "
                  f"{last.get(f'{prefix}points_available', 0.0):7.0f} "
                  f"{last[f'{prefix}repeatability']:7.3f} {decoy:7.3f} "
                  f"{last[f'{prefix}repeatability_margin']:7.3f} "
                  f"{1.0 / max(decoy, 1e-6):8.2f}")
        print('  One slide up and the other down is UNDECIDED, not a small '
              'win (spec.md 1, fourth row).')
        print('  `avail` is the UNCAPPED count above the threshold and is the '
              'convergence gauge: of order 1000 while the softmax is still '
              'flat, falling towards the label density as the dustbin learns. '
              'The margin cannot show that -- the budget holds pts/view fixed '
              'on purpose.')
        for key, label in (('val/dustbin_mean', 'dustbin'),
                           ('val/hit_score_mean', 'hit')):
            if key in last:
                print(f'  {label:8s} {last[key]:.4f}   (both climb to 1; '
                      f'dustbin alone climbing is a model that learnt to say '
                      f'"nothing here" and not to localise)')
        rungs = sorted(k for k in last if k.startswith('val/ds')
                       and k.endswith('/detector'))
        if rungs:
            print('  detector CE per rung -- all falling together means the '
                  'label-density spread is not the problem:')
            print('    ' + '  '.join(
                f"{k[len('val/'):-len('/detector')]}={last[k]:.3f}"
                for k in rungs))
    return 0


if __name__ == '__main__':
    sys.exit(main())
