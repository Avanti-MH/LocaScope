"""Upstream's two losses, transcribed, with the parts that look wrong explained.

    loss = SuperPointLossConfig().build()
    total, parts = loss(output, warped_output, batch)

    detector    sparse cross-entropy over `cell**2 + 1` channels, on BOTH the
                identity and the warped view, weight 1 each
    descriptor  a dense hinge over every pair of cells, weighted by whether the
                homography maps one onto the other

THREE THINGS THAT LOOK LIKE MISTAKES AND ARE NOT
--------------------------------------------------
1. `lambda_loss = 10000`. The dot-product tensor is L2-normalised AGAIN along
   each of the two flattened cell axes (`models/utils.py:102-119`), which drives
   its magnitude down by roughly `Hc * Wc`, and 10000 is what brings the
   descriptor term back to the detector's scale. **Both halves have to be copied
   together.** Taking the 10000 without the double normalisation makes the
   descriptor term dominate by four orders of magnitude; taking the
   normalisation without the 10000 makes it vanish. spec.md 9 says so and this
   is the file where it matters.

2. The random tie-break in the detector labels. `labels + uniform(0, 0.1)`
   before the argmax (`models/utils.py:14-16`): a cell holding two keypoints has
   to pick one, and picking the first would make the choice a function of raster
   order -- a systematic bias toward the top-left of every crowded cell. The
   noise makes it a coin flip instead. It is 0.1 against a gap of 1, so it can
   never turn a keypoint into a dustbin.

3. The valid mask is ANDed down a cell, not averaged. `reduce_prod` along the
   channel axis after `space_to_depth`: a cell counts only if EVERY one of its
   `cell**2` pixels is valid. Averaging would let a cell that is one pixel
   inside the warp contribute a fully-weighted label built from mostly-invalid
   pixels.

THE POSITIVE SET IS A GEOMETRIC FACT, NOT A LEARNED ONE
---------------------------------------------------------
`s[b, h, w, h', w']` is 1 when cell (h, w) of the original, pushed through the
homography, lands within `cell - 0.5` pixels of cell (h', w') of the warped
image. It is computed from the matrix, so it is right by construction -- and it
is also where a direction error hides: with `H` reversed the positives are the
pairs that do NOT correspond, the loss still decreases, and the descriptors
learn to match the wrong thing.

`points_input_to_output` is used rather than a bare matrix multiply for exactly
that reason: it says which direction it is in its name, and its docstring says
that upstream's `warp_points` folds the inversion in silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ConfigIdentity import IdentifiedConfig, register

#: The zero point. Every value is upstream's (`configs/superpoint_coco.yaml:
#: 42-45`, `super_point.py:73-92`), spec.md 9.
_LOSS_BASELINE = {
    'method': 'superpoint-loss',
    'positive_margin': 1.0,
    'negative_margin': 0.2,
    'lambda_d': 0.05,
    'lambda_loss': 10000.0,
    'detector_weight': 1.0,
    'tie_break': 0.1,
}


@register('superpoint-loss')
@dataclass(frozen=True)
class SuperPointLossConfig(IdentifiedConfig):
    method: str = 'superpoint-loss'

    #: The hinge. `positive_margin` and `negative_margin` are cosines, which is
    #: why the descriptor head has to L2-normalise: on unnormalised vectors
    #: these numbers mean whatever the current scale makes them mean.
    positive_margin: float = 1.0
    negative_margin: float = 0.2

    #: Positives are rare -- one cell matches one cell out of `Hc * Wc` -- so
    #: upstream down-weights them rather than the far more numerous negatives.
    lambda_d: float = 0.05

    #: See the module docstring. Copied together with the double normalisation
    #: or not at all.
    lambda_loss: float = 10000.0

    #: Identity and warped detector terms, weight 1 each (`super_point.py:73-92`).
    detector_weight: float = 1.0

    #: Amplitude of the argmax tie-break. 0.1 against a gap of 1.
    tie_break: float = 0.1

    def build(self) -> 'SuperPointLoss':
        return SuperPointLoss(self)


# ── the pieces ───────────────────────────────────────────────────────────────

def space_to_depth(x: torch.Tensor, cell: int) -> torch.Tensor:
    """`[N, H, W]` -> `[N, Hc, Wc, cell**2]`. The exact inverse of the decode.

    Channel `c` holds the pixel at `(dy, dx) = (c // cell, c % cell)` of its
    cell, which is what `Decoders.depth_to_space_prob` reads back out. The two
    have to agree or the labels are transposed inside every cell against the
    predictions -- and both are self-consistent, so nothing raises.
    """
    n, h, w = x.shape
    if h % cell or w % cell:
        raise ValueError(f'{h}x{w} does not divide into {cell} px cells')
    x = x.reshape(n, h // cell, cell, w // cell, cell)
    return x.permute(0, 1, 3, 2, 4).reshape(n, h // cell, w // cell, cell * cell)


def cell_labels(keypoint_map: torch.Tensor, cell: int,
                tie_break: float = 0.1) -> torch.Tensor:
    """`[N, H, W]` of 0/1 -> `[N, Hc, Wc]` int64 in `[0, cell**2]`.

    `cell**2` is the dustbin. Transcribed from `models/utils.py:9-16`, including
    the `2 * labels` against a column of ones -- a keypoint scores 2 and beats
    the dustbin's 1, an empty cell scores 0 and loses to it.
    """
    labels = space_to_depth(keypoint_map.float(), cell) * 2.0
    dustbin = torch.ones_like(labels[..., :1])
    stacked = torch.cat([labels, dustbin], dim=-1)
    if tie_break:
        stacked = stacked + torch.rand_like(stacked) * float(tie_break)
    return stacked.argmax(dim=-1)


def cell_valid(valid_mask: Optional[torch.Tensor], cell: int,
               shape: Tuple[int, int, int], device) -> torch.Tensor:
    """`[N, H, W]` of 0/1 -> `[N, Hc, Wc]` float, ANDed down each cell."""
    if valid_mask is None:
        n, h, w = shape
        return torch.ones(n, h // cell, w // cell, device=device)
    return space_to_depth(valid_mask.float(), cell).prod(dim=-1)


def detector_loss(cell_logits: torch.Tensor, keypoint_map: torch.Tensor,
                  cell: int, valid_mask: Optional[torch.Tensor] = None,
                  tie_break: float = 0.1,
                  sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Weighted sparse cross-entropy over the `cell**2 + 1` channels.

    `weights=valid_mask` in `tf.losses.sparse_softmax_cross_entropy` is a MEAN
    weighted by the mask, not a sum -- so an image whose warp leaves half its
    cells invalid contributes the same total as one that is fully valid, rather
    than half. Reproduced here by dividing by the mask's sum.

    `sample_weight` is `[N]` and is upstream-less: it is the `loss-weight` half
    of the rung-balance switch (`Datasets.rung_weight`). It multiplies into the
    same weighted mean, so a rung with a tenth of the tiles can be given ten
    times the per-tile weight and the term keeps its scale. Passing None is the
    `none` and `align-min` modes, where the balance was handled by the sampling.
    """
    labels = cell_labels(keypoint_map, cell, tie_break)
    weights = cell_valid(valid_mask, cell, keypoint_map.shape,
                         cell_logits.device)
    if sample_weight is not None:
        weights = weights * sample_weight.to(weights.device).view(-1, 1, 1)
    per_cell = F.cross_entropy(cell_logits, labels, reduction='none')
    return (per_cell * weights).sum() / weights.sum().clamp_min(1.0)


def detector_ce_per_sample(cell_logits: torch.Tensor,
                           keypoint_map: torch.Tensor, cell: int,
                           valid_mask: Optional[torch.Tensor] = None
                           ) -> torch.Tensor:
    """`[N]` -- each image's own detector cross-entropy. A METRIC, not the loss.

    NOT `detector_loss` WITH A DIFFERENT REDUCTION, and the difference is not
    cosmetic. `detector_loss` divides by the mask's sum over the WHOLE batch, so
    an image whose warp left few valid cells contributes proportionally less.
    Here each image is normalised by its own mask, because the question this
    answers is "how well is this rung learnt", and a rung whose tiles happen to
    warp badly must not read as a rung the model is failing at.

    `tie_break=0` on purpose. The 0.1 of noise in the training labels is there
    so a cell holding two keypoints does not always pick the raster-first one;
    a metric wants the same cell to give the same number twice.
    """
    labels = cell_labels(keypoint_map, cell, tie_break=0.0)
    weights = cell_valid(valid_mask, cell, keypoint_map.shape,
                         cell_logits.device)
    per_cell = F.cross_entropy(cell_logits, labels, reduction='none')
    return ((per_cell * weights).sum(dim=(1, 2))
            / weights.sum(dim=(1, 2)).clamp_min(1.0))


def dustbin_and_hit(cell_logits: torch.Tensor, keypoint_map: torch.Tensor,
                    cell: int) -> Tuple[torch.Tensor, torch.Tensor,
                                        torch.Tensor, torch.Tensor]:
    """The two halves the detector CE mixes into one number.

        dustbin  mean p(dustbin) over the cells whose label IS the dustbin
        hit      mean p(label) over the cells that hold a keypoint

    Both should climb to 1. THE POINT OF SPLITTING THEM is that a single CE
    cannot say which half is stuck, and the two failures want opposite fixes: a
    model that learns to say "nothing here" and never localises needs the
    keypoint cells upweighted, while a model that does neither needs steps or a
    learning rate. On this corpus 86 per cent of cells are dustbin (146 label
    points over 1024 cells), so the CE is mostly reporting the first half.

    Returns `(dustbin_sum, dustbin_count, hit_sum, hit_count)` rather than two
    means, so a caller accumulating over batches averages over CELLS and not
    over batches -- the counts differ per tile by a factor of thirty across the
    rungs, and a mean of means would weight a Ki67 ds1 tile the same as a BRACS
    ds32 one.
    """
    labels = cell_labels(keypoint_map, cell, tie_break=0.0)
    probs = F.softmax(cell_logits, dim=1)
    chosen = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

    is_dustbin = labels == (cell * cell)
    dustbin = chosen[is_dustbin]
    hit = chosen[~is_dustbin]
    return (dustbin.sum(), torch.tensor(float(dustbin.numel())),
            hit.sum(), torch.tensor(float(hit.numel())))


def correspondence_mask(homography: torch.Tensor, pitch: int,
                        grid_hw: Tuple[int, int], device) -> torch.Tensor:
    """`[N, Hc, Wc, Hc, Wc]`: which original grid square maps onto which warped one.

    A square's position is its CENTRE pixel, `h * pitch + pitch // 2`
    (`models/utils.py:79-82`), and the threshold is `pitch - 0.5` -- just under
    one square, so a centre that lands exactly on a boundary claims one side
    rather than both.

    `pitch` IS THE GRID `grid_hw` DESCRIBES, WHICH IS NOT ALWAYS THE CELL.
    Its only caller is `descriptor_loss`, and the grid there is the DESCRIPTOR
    map -- which lives on the backbone's stride, not on the detector's cell.
    Upstream has both at 8 so the distinction never arose; at stride 16 against
    cell 8 the centres would be computed over the top-left quarter of the tile
    and every correspondence would be wrong, with the loss still going down.

    `homography` is the matrix `warp_image` was called with, i.e. OUTPUT ->
    INPUT. Pushing an ORIGINAL point into the WARPED frame therefore applies its
    INVERSE, which is what `points_input_to_output` does and says. Reversing it
    makes the positives the pairs that do not correspond; the loss still goes
    down, and the descriptors learn to match the wrong cells.
    """
    hc, wc = grid_hw
    ys, xs = torch.meshgrid(torch.arange(hc, device=device),
                            torch.arange(wc, device=device), indexing='ij')
    centres = torch.stack([xs, ys], dim=-1).float() * pitch + pitch // 2  # (x, y)

    flat = centres.reshape(-1, 2)
    ones = torch.ones(flat.shape[0], 1, device=device)
    points = torch.cat([flat, ones], dim=1)                    # [Hc*Wc, 3]
    inverse = torch.linalg.inv(homography.float())             # OUT -> IN, so
    warped = points @ inverse.transpose(-1, -2)                # IN -> OUT here
    warped = warped[..., :2] / warped[..., 2:3].clamp(min=1e-8)

    n = warped.shape[0]
    warped = warped.reshape(n, hc, wc, 1, 1, 2)
    grid = centres.reshape(1, 1, 1, hc, wc, 2)
    distance = torch.linalg.norm(grid - warped, dim=-1)
    return (distance <= pitch - 0.5).float()


def descriptor_loss(descriptors: torch.Tensor, warped_descriptors: torch.Tensor,
                    homography: torch.Tensor, stride: int,
                    valid_mask: Optional[torch.Tensor] = None, *,
                    positive_margin: float = 1.0, negative_margin: float = 0.2,
                    lambda_d: float = 0.05) -> torch.Tensor:
    """The dense hinge. `[N, D, Hf, Wf]` twice, plus the homography between them.

    Transcribed from `models/utils.py:70-125`. The two extra `l2_normalize`
    calls on the dot-product tensor are the ones the module docstring is about:
    they are not tidying, they set the scale that `lambda_loss = 10000`
    compensates for.

    `stride`, NOT `cell`. The descriptor map comes off the BACKBONE, so one of
    its pixels is `stride` input pixels wide; the detector's cell grid is a
    different grid whenever a decoder changes the resolution between them.
    Upstream's VGG has stride 8 and cell 8, so this argument was `cell` here
    until the first backbone where they differ -- and at stride 16 against cell
    8 it put every correspondence in the top-left quarter of the tile, silently.
    """
    n, d, hc, wc = descriptors.shape
    s = correspondence_mask(homography, stride, (hc, wc), descriptors.device)

    a = F.normalize(descriptors, p=2, dim=1).permute(0, 2, 3, 1)
    b = F.normalize(warped_descriptors, p=2, dim=1).permute(0, 2, 3, 1)
    dot = torch.einsum('nhwd,nijd->nhwij', a, b)
    dot = F.relu(dot)
    # The double normalisation, in the two orders upstream uses: over the
    # WARPED grid flattened, then over the ORIGINAL grid flattened.
    dot = F.normalize(dot.reshape(n, hc, wc, hc * wc), p=2, dim=3).reshape(
        n, hc, wc, hc, wc)
    dot = F.normalize(dot.reshape(n, hc * wc, hc, wc), p=2, dim=1).reshape(
        n, hc, wc, hc, wc)

    positive = (positive_margin - dot).clamp_min(0.0)
    negative = (dot - negative_margin).clamp_min(0.0)
    loss = lambda_d * s * positive + (1.0 - s) * negative

    # `stride` again, for the same reason and in a second place: `hc, wc` is
    # the DESCRIPTOR grid, so the full-resolution mask folds down onto it by
    # the stride. `cell_valid` is named for its usual caller (the detector,
    # where the grid IS the cell grid); what it actually takes is the pitch of
    # whatever grid the caller is folding onto.
    weights = cell_valid(valid_mask, stride,
                         (n, hc * stride, wc * stride), descriptors.device)
    weights = weights.reshape(n, 1, 1, hc, wc)
    # `normalization` is the number of valid PAIRS, which is why it multiplies
    # by Hc*Wc rather than being the sum of a [N,Hc,Wc,Hc,Wc] mask: only the
    # warped side is masked, and each masked cell pairs with every original one.
    normalization = weights.sum() * float(hc * wc)
    return (weights * loss).sum() / normalization.clamp_min(1.0)


# ── the sum ──────────────────────────────────────────────────────────────────

class SuperPointLoss(nn.Module):
    """detector(identity) + detector(warped) + lambda_loss * descriptor.

    Returns the total and the parts. The parts are not decoration: spec.md 12
    step 6 has the three magnitudes as a thing to look at after the first epoch,
    and a total alone cannot say which term is doing anything.
    """

    def __init__(self, cfg: SuperPointLossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, output, warped_output, batch: Dict[str, torch.Tensor],
                cell: int, stride: Optional[int] = None
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """`cell` drives the detector terms, `stride` the descriptor one.

        Two arguments and not one, because they are two grids: `cell_logits`
        is on the cell grid by construction and `descriptors` is on the
        backbone's. `stride` defaults to whatever the output carries, which is
        the only place that knows -- a foundation trunk's stride is its patch
        size, a fact about the loaded model rather than about any config.
        """
        cfg = self.cfg
        sample_weight = batch.get('rung_weight')
        detector = cfg.detector_weight * detector_loss(
            output.cell_logits, batch['keypoint_map'], cell,
            batch.get('valid_mask'), cfg.tie_break, sample_weight)
        warped_detector = cfg.detector_weight * detector_loss(
            warped_output.cell_logits, batch['warped_keypoint_map'], cell,
            batch.get('warped_valid_mask'), cfg.tie_break, sample_weight)

        total = detector + warped_detector
        parts = {'detector': float(detector), 'warped_detector':
                 float(warped_detector)}

        if output.descriptors is not None and warped_output.descriptors is not None:
            stride = int(stride or getattr(output, 'stride', 0) or 0)
            if stride <= 0:
                raise ValueError(
                    'descriptor_loss needs the BACKBONE stride and neither the '
                    'argument nor output.stride supplied one. Defaulting it to '
                    'cell is right only when they are equal, which is upstream '
                    'and nothing else')
            descriptor = descriptor_loss(
                output.descriptors, warped_output.descriptors,
                batch['homography'], stride, batch.get('warped_valid_mask'),
                positive_margin=cfg.positive_margin,
                negative_margin=cfg.negative_margin, lambda_d=cfg.lambda_d)
            total = total + cfg.lambda_loss * descriptor
            parts['descriptor'] = float(descriptor)
            parts['descriptor_scaled'] = float(cfg.lambda_loss * descriptor)

        parts['total'] = float(total)
        return total, parts
