"""The training loop: Adam, a constant LR, wandb, and a checkpoint that says
what it is.

    trainer = TrainerConfig().build(net, loss, train_ds, val_ds, out_dir)
    trainer.run()

UPSTREAM'S OPTIMISER, AND WHY THERE IS NO SCHEDULE
----------------------------------------------------
Adam at a constant learning rate, no warmup, no decay (`base_model.py:212`),
1e-4 for SuperPoint and 1e-3 for MagicPoint (spec.md 9). Adding a schedule would
be a change nobody here has measured, on top of a change (the data) that is the
thing being measured -- and the first run's job is to make the pipeline produce
a number, not to be the best model.

WHAT IS LOGGED, AND THE ONE UPSTREAM DOES NOT LOG
---------------------------------------------------
Upstream tracks precision and recall during training (`super_point.py:94-101`)
and computes repeatability offline in `evaluations/`. Precision and recall here
are measured against the model's OWN teacher's labels, so a teacher with a
systematic bias produces a precision that climbs happily while the detector gets
worse at the actual task.

So repeatability is pulled into the loop (spec.md 9): it is the number that
decides whether to run another round of Homographic Adaptation, and a number
that decides something has to be visible while it is being decided.

It is computed on the VALIDATION set only, and against a decoy -- the same
points shifted by `2 * nms_radius + 1` px -- because there is no keypoint ground
truth on a WSI and an absolute repeatability of 0.62 would need a reference to
mean anything (spec.md 1: every criterion is a margin over a decoy).

THE CHECKPOINT CARRIES `identity_json`
----------------------------------------
Not just the state dict: the config that built it, the loss config, the dataset
config, and the label store's `ha_id`. Round 2 of Stage A turns this checkpoint
into a teacher, and a teacher that cannot say which labels it was trained on
cannot be told apart from the one before it -- which is precisely the thing
`LabelMeta.ha_id` exists to separate.

<!-- PENDING-MEASUREMENT: the three loss magnitudes after the first epoch
     (spec.md 12 step 6). `lambda_loss = 10000` compensates for the descriptor
     term's double normalisation, and whether it lands at the detector term's
     scale on THIS data is a fact about the data. `parts` is logged every step
     for exactly this; fill the ratio in here once the first epoch has run. -->
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ConfigIdentity import IdentifiedConfig, register

from common.KeypointLabelStore import points_from_prob

#: The zero point. ConfigIdentity rule 1.
_TRAINER_BASELINE = {
    'method': 'superpathpoint-trainer',
    'lr': 1e-4,
    'batch_size': 16,
    'epochs': 10,
    'optimizer': 'adam',
}


@register('superpathpoint-trainer')
@dataclass(frozen=True)
class TrainerConfig(IdentifiedConfig):
    method: str = 'superpathpoint-trainer'

    #: Upstream's SuperPoint LR (`configs/superpoint_coco.yaml`). MagicPoint's
    #: 1e-3 is for the stage this project skips.
    lr: float = 1e-4
    optimizer: str = 'adam'

    #: Upstream counts ITERATIONS (600000 for SuperPoint), not epochs, because
    #: COCO is effectively infinite. v1's corpus is 18000 tiles at most, so an
    #: epoch is a meaningful unit here and the number of them is small.
    epochs: int = 10
    batch_size: int = 16

    #: Throughput, checkpointing and logging. None of it changes a weight.
    workers: int = 4
    log_every: int = 20
    val_every: int = 1
    amp: bool = True
    wandb_project: str = 'superpathpoint'
    wandb_mode: str = 'online'
    run_name: str = ''

    #: HOW MANY POINTS THE REPEATABILITY IS MEASURED AT, per view, taken as the
    #: top N by score with no threshold. A FIXED BUDGET AND NOT A THRESHOLD,
    #: because `margin = repeatability / decoy` and the decoy is a function of
    #: the density: two models cut by one threshold land at two densities, on
    #: two different scales, and the 2026-08-31 run reported 1.50 against 3.59
    #: that way with nothing to say they were not comparable.
    #:
    #: 200 is inside the label corpus's own range (per-rung `n_kp` means run 3
    #: to 527, overall 146; the two held-out slides average about 295 and 40).
    #: The ladder over several budgets is `cli/reeval_density.py`, run once on
    #: the finished checkpoints; in the loop one budget is enough because what
    #: it has to compare is this epoch against the last.
    val_budget: int = 200

    #: `val_budget` is NOT identity: it changes what is REPORTED, never a
    #: weight. Listing it here also keeps every existing checkpoint's id
    #: (ConfigIdentity rule 1 -- a new hashed field re-hashes all of them).
    NOT_IDENTITY = ('workers', 'log_every', 'val_every', 'amp',
                    'wandb_project', 'wandb_mode', 'run_name', 'val_budget')

    def build(self, net, loss, train_set, val_set=None, out_dir='.',
              extra_identity: Optional[Dict[str, str]] = None) -> 'Trainer':
        return Trainer(self, net, loss, train_set, val_set, out_dir,
                       extra_identity)


class Trainer:
    def __init__(self, cfg: TrainerConfig, net, loss, train_set, val_set,
                 out_dir, extra_identity=None):
        self.cfg = cfg
        self.net = net
        self.loss = loss
        self.train_set = train_set
        self.val_set = val_set
        self.out_dir = str(out_dir)
        self.extra_identity = dict(extra_identity or {})
        os.makedirs(self.out_dir, exist_ok=True)

        self.device = net.device
        # `trainable` is read, not assumed: a frozen foundation-model trunk
        # hands the optimiser nothing to do, and an optimiser over frozen
        # tensors is a loss that never moves with no error anywhere
        # (common/Interfaces.py).
        params = [p for p in net.parameters() if p.requires_grad]
        if not params:
            raise ValueError(
                'no trainable parameters. Every part of this model reports '
                'trainable=False, so there is nothing for the optimiser to do')
        if cfg.optimizer != 'adam':
            raise ValueError(f'unknown optimizer {cfg.optimizer!r}; upstream is adam')
        self.optimizer = torch.optim.Adam(params, lr=cfg.lr)

        self.scaler = torch.amp.GradScaler(
            'cuda', enabled=cfg.amp and self.device.type == 'cuda')
        self.rung_weight = getattr(train_set, 'rung_weight', None)
        self.wandb_run = None
        self.history: List[dict] = []

    # ── loaders ──

    def _loader(self, dataset, shuffle: bool, persistent: bool = False
                ) -> DataLoader:
        """Fork-safe because a worker only opens PNGs (`Datasets`).

        `persistent=False` FOR THE TRAINING SET, and it is not a throughput
        oversight. A worker is a forked copy of the dataset, so
        `set_epoch` on this process's copy never reaches a worker that was
        forked once and kept -- every epoch would silently redraw epoch 0's
        warps. Re-forking per epoch costs a second; the alternative costs the
        augmentation.
        """
        return DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=shuffle,
            num_workers=self.cfg.workers, pin_memory=True, drop_last=shuffle,
            persistent_workers=bool(self.cfg.workers) and persistent)

    # ── the loop ──

    def run(self) -> List[dict]:
        """Train for `cfg.epochs` and return the per-epoch rows."""
        self._start_wandb()
        steps = max(len(self.train_set) // int(self.cfg.batch_size), 1)
        print(f'train {len(self.train_set)} pairs, {steps} steps/epoch, '
              f'{steps * int(self.cfg.epochs)} total', flush=True)

        step = 0
        for epoch in range(int(self.cfg.epochs)):
            # BEFORE the loader is built, so the workers forked from this
            # dataset carry the new epoch. See `_loader` and
            # `HomographyPairDataset.set_epoch`.
            if hasattr(self.train_set, 'set_epoch'):
                self.train_set.set_epoch(epoch)
            loader = self._loader(self.train_set, shuffle=True)
            self.net.train()
            started = time.time()
            running: Dict[str, float] = {}
            for batch in loader:
                parts = self._step(batch)
                step += 1
                for key, value in parts.items():
                    running[key] = running.get(key, 0.0) + value
                if step % int(self.cfg.log_every) == 0:
                    self._log({f'train/{k}': v for k, v in parts.items()},
                              step=step, epoch=epoch)

            row = {'epoch': epoch, 'seconds': round(time.time() - started, 1)}
            row.update({f'train/{k}': v / max(len(loader), 1)
                        for k, v in running.items()})
            if self.val_set is not None and (epoch + 1) % int(self.cfg.val_every) == 0:
                row.update(self.validate())
            self.history.append(row)
            self._log(row, step=step, epoch=epoch)
            print(f'[epoch {epoch}] ' + '  '.join(
                f'{k}={v:.4g}' for k, v in row.items() if k != 'epoch'),
                flush=True)
            self.save_checkpoint(f'epoch{epoch:03d}')

        self.save_checkpoint('last')
        self._finish_wandb()
        return self.history

    def _step(self, batch) -> Dict[str, float]:
        batch = self._to_device(batch)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', torch.float16,
                            enabled=self.cfg.amp and self.device.type == 'cuda'):
            output = self.net(batch['image'])
            warped_output = self.net(batch['warped_image'])
            total, parts = self.loss(output, warped_output, batch,
                                     self.net.cell)
        self.scaler.scale(total).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return parts

    def _to_device(self, batch) -> Dict[str, torch.Tensor]:
        out = {k: v.to(self.device, non_blocking=True)
               for k, v in batch.items() if torch.is_tensor(v)}
        if self.rung_weight is not None and 'rung_index' in out:
            # Looked up per sample rather than carried in the item, so that a
            # changed balance policy does not require re-reading the dataset.
            out['rung_weight'] = self.rung_weight.to(self.device)[
                out['rung_index'].long()]
        return out

    # ── validation ──

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Loss on the held-out pairs, plus repeatability against a decoy.

        REPORTED PER SLIDE AS WELL AS OVERALL (spec.md 13). Tiles of one slide
        share a staining batch, a scanner, a section thickness and a tissue
        source, so two slides averaged into one number can hide a difference
        that is entirely one slide's personality -- and the criterion in
        spec.md 1 is "one of them wins on BOTH slides", which a single averaged
        number cannot answer. One slide up and the other down is undecided, not
        a small win.

        The per-slide keys are `val/<stem>/...`. They are extra columns on the
        same row rather than a second file, so a history CSV stays one table.
        """
        from SuperPoint.Losses import (detector_ce_per_sample,  # noqa: PLC0415
                                       dustbin_and_hit)

        self.net.eval()
        # Built here and not cached: `validate` is called once per `val_every`
        # epochs and the workers are gone by the time it returns. Persistent
        # workers on a loader that is iterated once would leak eight processes
        # per validation, which over 250 epochs is not a rounding error.
        loader = self._loader(self.val_set, shuffle=False)

        totals: Dict[str, float] = {}
        hits, decoys, counts, avails, slides = [], [], [], [], []
        cell = self.net.cell
        cfg = self.net.cfg
        shift = 2 * int(cfg.nms_radius) + 1
        budget = int(self.cfg.val_budget)

        # Sums, not means of means. A per-rung mean over batches would weight a
        # batch of 5 the same as a batch of 64.
        by_rung: Dict[int, List[float]] = {}
        dustbin_sum = dustbin_n = hit_sum = hit_n = 0.0

        for batch in loader:
            batch = self._to_device(batch)
            output = self.net(batch['image'])
            warped_output = self.net(batch['warped_image'])
            _, parts = self.loss(output, warped_output, batch, self.net.cell)
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value

            # (f) per-rung detector CE. The label density varies thirty-fold
            # across the rungs (2 per cent of cells occupied at Ki67 ds 1, 62
            # at BRACS ds 32), and one averaged CE cannot tell "learning
            # everywhere" from "learning only where the points are dense".
            per_sample = detector_ce_per_sample(
                output.cell_logits, batch['keypoint_map'], cell,
                batch.get('valid_mask')).detach().cpu().numpy()
            if 'rung_index' in batch:
                rungs = batch['rung_index'].detach().cpu().numpy()
                for i, value in enumerate(per_sample):
                    by_rung.setdefault(int(rungs[i]), []).append(float(value))

            # (e) the two halves the CE mixes together.
            d_sum, d_n, h_sum, h_n = dustbin_and_hit(
                output.cell_logits, batch['keypoint_map'], cell)
            dustbin_sum += float(d_sum)
            dustbin_n += float(d_n)
            hit_sum += float(h_sum)
            hit_n += float(h_n)

            hit, decoy, count, avail, which = _repeatability(
                output.prob_map.detach().float().cpu().numpy(),
                warped_output.prob_map.detach().float().cpu().numpy(),
                batch['homography'].detach().float().cpu().numpy(),
                cfg, shift, budget)
            hits += hit
            decoys += decoy
            counts += count
            avails += avail
            if 'slide_index' in batch:
                index = batch['slide_index'].detach().cpu().numpy()
                slides += [int(index[i]) for i in which]
            else:
                slides += [-1] * len(which)

        steps = max(len(loader), 1)
        row = {f'val/{k}': v / steps for k, v in totals.items()}
        row.update(_repeatability_row(hits, decoys, counts, avails,
                                      prefix='val/'))

        if dustbin_n:
            row['val/dustbin_mean'] = dustbin_sum / dustbin_n
        if hit_n:
            row['val/hit_score_mean'] = hit_sum / hit_n

        rung_values = getattr(self.val_set, 'rungs', [])
        for index, values in sorted(by_rung.items()):
            name = (f'{rung_values[index]:g}' if index < len(rung_values)
                    else str(index))
            row[f'val/ds{name}/detector'] = float(np.mean(values))
            row[f'val/ds{name}/n_tiles'] = float(len(values))

        names = getattr(self.val_set, 'slides', None)
        if names and hits:
            for index, name in enumerate(names):
                keep = [i for i, s in enumerate(slides) if s == index]
                if not keep:
                    continue
                row.update(_repeatability_row(
                    [hits[i] for i in keep], [decoys[i] for i in keep],
                    [counts[i] for i in keep], [avails[i] for i in keep],
                    prefix=f'val/{name}/'))
        return row

    # ── checkpoint ──

    def save_checkpoint(self, tag: str) -> str:
        path = os.path.join(self.out_dir, f'superpathpoint_{tag}.pt')
        identity = {'net': self.net.identity_json(),
                    'trainer': json.dumps(
                        {f.name: getattr(self.cfg, f.name)
                         for f in dataclasses.fields(self.cfg)},
                        sort_keys=True, default=str)}
        identity.update(self.extra_identity)
        torch.save({'state_dict': self.net.state_dict(),
                    'identity_json': identity,
                    'identity_id': self.net.identity_id(),
                    'history': self.history}, path)
        return path

    # ── wandb ──

    def _start_wandb(self) -> None:
        try:
            import wandb                                          # noqa: PLC0415
        except ImportError:
            print('wandb not installed; logging to stdout and the CSV only',
                  flush=True)
            return
        self.wandb_run = wandb.init(
            project=self.cfg.wandb_project, mode=self.cfg.wandb_mode,
            name=self.cfg.run_name or None,
            # The identity, not just the config: `identity_id` is what a label
            # store points at, so a run and the labels it produced can be lined
            # up afterwards without guessing from timestamps.
            config={'identity_id': self.net.identity_id(),
                    'net': self.net.identity_json(),
                    **self.extra_identity})

    def _log(self, row: Dict[str, float], *, step: int, epoch: int) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log({**row, 'epoch': epoch}, step=step)

    def _finish_wandb(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()


def _cut(prob, cfg, budget):
    """One NMS pass -> `(top-budget points, how many passed the threshold)`.

    TWO NUMBERS FROM ONE EXTRACTION, and they answer different questions:

      the points    the top `budget` by score, threshold at zero. A FIXED
                    BUDGET, so the density is pinned and `decoy` is the same
                    quantity for every epoch and every arm.
      the count     how many survivors clear `cfg.detection_threshold`, with no
                    cap at all. This is the convergence gauge, and it is the
                    one the old code could not report: `points_per_view` was
                    measured after the cap, so an undertrained detector emitting
                    thousands of points read as exactly `max_keypoints`.

    `points_from_prob` does NMS, then the border cut, then the threshold, then
    the cap. Asking for both numbers separately would run NMS twice; asking
    once at threshold 0 with no cap gives every survivor with its score, and
    both answers are then array operations on that.
    """
    xy, score, _ = points_from_prob(prob, None, score_threshold=0.0,
                                    nms_radius=cfg.nms_radius,
                                    border=cfg.border, max_points=None)
    available = int((score > float(cfg.detection_threshold)).sum())

    if not budget:
        # `budget=0` is the CONFIG'S OWN RULE -- threshold then cap -- kept so
        # that one implementation serves both the loop (a budget) and
        # `cli/reeval_density.py`'s `native` row (what the training run
        # actually reported). Two spellings of an extraction is how a
        # repeatability turns into a comparison of two NMS implementations.
        keep = score > float(cfg.detection_threshold)
        xy, score = xy[keep], score[keep]
        budget = int(cfg.max_keypoints or 0)

    if budget and len(xy) > int(budget):
        keep = np.argpartition(-score, int(budget) - 1)[:int(budget)]
        xy = xy[keep]
    return xy, available


def _repeatability(prob, warped_prob, homography, cfg, shift, budget):
    """Fraction of the warped view's points that the identity view also found.

    Both point sets are cut by `points_from_prob`, the same function that cut
    the teacher's labels -- otherwise this number would be measuring the
    difference between two implementations of NMS rather than between two views
    (spec.md 14).

    The decoy is the same comparison against one set shifted by more than the
    NMS radius, so no point can match itself. What is reported is the pair.

    CUT TO A FIXED BUDGET, NOT TO A THRESHOLD. `margin = repeatability / decoy`
    and `repeatability <= 1`, so the decoy IS the ceiling -- and the decoy rises
    with density, because it asks whether a set shifted past the NMS radius
    matches anyway and a dense enough set matches anything. Two models cut by
    one threshold land at two densities and their margins are on two different
    scales: the 2026-08-31 run reported 1.50 for one arm and 3.59 for another
    at 420 and 159 points per view, and nothing in the table said they could not
    be compared.
    """
    from common.Homography import inside, points_input_to_output  # noqa: PLC0415

    hits, decoys, counts, avails, which = [], [], [], [], []
    shape = prob.shape[-2:]
    for i in range(prob.shape[0]):
        a, avail_a = _cut(prob[i], cfg, budget)
        b, avail_b = _cut(warped_prob[i], cfg, budget)
        if len(a) == 0 or len(b) == 0:
            continue
        projected = points_input_to_output(a.astype(np.float64), homography[i])
        projected = projected[inside(projected, shape)]
        if len(projected) == 0:
            continue
        hits.append(_match_rate(projected, b, cfg.nms_radius))
        decoys.append(_match_rate(projected, b + shift, cfg.nms_radius))
        # THE POINT COUNT IS NOT DECORATION. Under a fixed budget it is the
        # budget, which is exactly the claim being made -- a column that comes
        # back lower means NMS left fewer survivors than the budget asked for
        # on some tile, and those tiles were scored at a lower density than the
        # rest.
        counts.append(0.5 * (len(a) + len(b)))
        avails.append(0.5 * (avail_a + avail_b))
        which.append(i)
    return hits, decoys, counts, avails, which


def _repeatability_row(hits, decoys, counts, avails, *, prefix: str
                       ) -> Dict[str, float]:
    """The five numbers that have to be read together, under one prefix.

    `n_pairs` and `points_per_view` are here because the other two cannot be
    read without them: a margin over 6 pairs is not a measurement, and a margin
    at an unrecorded density cannot be compared with the next epoch's.

    `points_available` is the one that is NOT part of the margin. It is how many
    points the model would emit if nothing capped it, and it is the only column
    that moves when the detector converges: a model whose softmax is still flat
    puts 1/65 = 0.0154 on every class, above `detection_threshold`, so every
    cell passes and NMS geometry alone decides the count -- of order a thousand.
    A model that has learnt to say "nothing here" emits its own density. Watch
    this fall towards the label density; the margin cannot show it because the
    budget holds the density fixed on purpose.
    """
    if not hits:
        return {}
    hit = float(np.mean(hits))
    decoy = float(np.mean(decoys))
    return {
        f'{prefix}n_pairs': float(len(hits)),
        f'{prefix}points_per_view': float(np.mean(counts)) if counts else 0.0,
        f'{prefix}points_available': float(np.mean(avails)) if avails else 0.0,
        f'{prefix}repeatability': hit,
        f'{prefix}repeatability_decoy': decoy,
        f'{prefix}repeatability_margin': hit / max(decoy, 1e-6),
    }


def _match_rate(a: np.ndarray, b: np.ndarray, radius: int) -> float:
    """Fraction of `a` with a `b` point within `radius`, in the max-norm.

    Max-norm because that is the neighbourhood NMS suppresses over: two points
    closer than the radius cannot both survive one extraction, so treating them
    as the same point across views is the consistent reading.
    """
    delta = np.abs(a[:, None, :] - b[None, :, :])
    return float((delta.max(axis=2).min(axis=1) <= radius).mean())
