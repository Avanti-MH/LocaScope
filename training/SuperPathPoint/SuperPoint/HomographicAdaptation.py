"""N random homographies, one aggregated keypoint map. Upstream's stage 2.

    ha = HaConfig().build(teacher)
    result = ha.run(pre_tile, tile=256, rng=np.random.default_rng(0))
    result.mean_prob   # [tile, tile] float32, the label to threshold
    result.counts      # [tile, tile] float32, how many views could see it

Transcribed from `superpoint/models/homographies.py:28-114`, with the three
differences below stated rather than absorbed.

1. THE SOURCE IS A PRE-TILE, THE FRAME IS THE TILE
---------------------------------------------------
Upstream warps an image into its own frame, so a third of every warped view is
the black `BORDER_CONSTANT` fill and the detector fires on its perfectly
straight edges (spec.md 6.6, measured: `valid 67.8%`).

Here the homography is sampled for the TILE shape, and the warp reads from the
PRE-TILE, which extends `margin` px past the tile on every side:

    H          output (tile frame) -> input (tile frame)      -- what is sampled
    T @ H      output (tile frame) -> pre-tile pixel coords   -- what cv2 reads

`T` is a pure translation by the margin. It never touches the geometry a
keypoint lands in: every coordinate this module records, warps or inverts is in
the TILE frame, and `T` exists only so that the pixels come from real tissue.
The network still sees `tile x tile`, so the cost is unchanged -- warping the
whole pre-tile and cropping afterwards would have cost 9x per view.

The visible consequence, and the evidence it is working: `mask` goes to nearly
all-True while `counts` does not. Upstream's `mask` is "which output pixels came
from inside the source" and with a 3x source that is everything but the eroded
rim; `counts` is "how many views covered this pixel of the ORIGINAL frame",
which a zoomed-in view still fails to do. Two masks that used to be symmetric
and now are not.

2. `aggregation: 'sum'` IS A MEAN, SO IT IS CALLED ONE
-------------------------------------------------------
`homographies.py:102-107` computes `mean_prob = sum(probs) / counts` inside the
branch named `'sum'` -- a coverage-weighted mean. The options here are
`'mean' | 'max'` and the field is `mean_prob`. A name is the name of the
operation (ClaudeRules section 12).

3. NO NMS, NO THRESHOLD, ANYWHERE IN HERE
------------------------------------------
Upstream runs both AFTER adaptation (`magic_point.py:39-46`), on the aggregate.
Thinning inside the loop would suppress a peak in one view that a hundred views
agree on -- which is the thing being aggregated. `Teacher.dense_prob` is
deliberately the pre-NMS map for the same reason, and turning the aggregate into
points is `KeypointLabelStore`'s caller's job.

WHAT WOULD RUN AND BE WRONG
----------------------------
Warping the probability back with `H` instead of `H_inv`. The result is a smooth,
plausible map, offset by the homography -- and averaged over 100 draws the
offsets partly cancel, so it looks like a slightly blurry label rather than a
bug. `test_homographic_adaptation` (spec.md 10) pins it with a fake detector that
answers one fixed point and a decoy shifted by one cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ConfigIdentity import IdentifiedBuild, IdentifiedConfig, register

from common.Homography import erode_valid, invert, sample_homography, warp_image
from common.HomographyConfig import HOMOGRAPHY_BASELINE, HomographyConfig
from PreTileStore import (centre_crop, centre_margin,
                                pretile_valid_mask, warp_from_pretile)

#: The zero point. ConfigIdentity rule 1: editing this re-hashes every label
#: ever written; editing a dataclass default splits new from old.
#:
#: Every value is upstream's, from `configs/magic-point_coco_export.yaml:12-26`
#: and `models/homographies.py:117-230` -- the table in spec.md 9.
_HA_BASELINE = {
    'method': 'ha-superpoint',
    'num': 100,
    'aggregation': 'mean',
    'valid_border_margin': 3,
    'filter_counts': 0,
    'homography': HOMOGRAPHY_BASELINE,
}

@register('ha-superpoint')
@dataclass(frozen=True)
class HaConfig(IdentifiedConfig):
    """Upstream's HA config, spelled out field by field.

    Flat rather than a nested dict because these ARE the identity of a label:
    `parts_against` walks fields, and a dict field would hash as whatever
    `str(dict)` happens to produce -- insertion order included. Twelve fields is
    also the spec.md 9 table, so the two can be diffed by eye.
    """
    method: str = 'ha-superpoint'

    #: How many views, the identity one included. What it decides is in
    #: spec.md 13: label noise, the wall clock (strictly linear in it), and the
    #: coverage at the tile edge. Upstream's 100.
    num: int = 100

    #: 'mean' is upstream's `'sum'` branch, which divides by counts. 'max' is
    #: upstream's `'max'`.
    aggregation: str = 'mean'

    #: Erosion radius for both masks, in pixels. Upstream's 3, applied with an
    #: EVEN 6 px elliptical kernel and TF's anchor -- see `Homography.
    #: erode_valid`, where getting this wrong costs one pixel on two sides of
    #: every view and does not average out over 100 draws.
    valid_border_margin: int = 3

    #: Zero out pixels seen by fewer than this many views. Upstream's default is
    #: 0, i.e. off -- and it stays off here because `counts` is stored, so a
    #: reader can apply any threshold later without re-running. Baking it in is
    #: the lossy version of the same thing.
    filter_counts: int = 0

    #: The thirteen sampler options, nested rather than spelled out here.
    #: `PairDatasetConfig` needs the same thirteen, and the two must not drift:
    #: a student trained on pairs drawn from a wider distribution than the one
    #: that produced its labels is being asked to be invariant to transforms its
    #: teacher never voted on. See `common/HomographyConfig.py`.
    homography: HomographyConfig = field(default_factory=HomographyConfig)

    #: Throughput only. How many warped views go through the teacher at once.
    batch: int = 16

    NOT_IDENTITY = ('batch',)

    def homography_kwargs(self) -> Dict[str, object]:
        """Kept as a method on this config so callers do not have to know the
        options moved into a nested one."""
        return self.homography.kwargs()

    def build(self, teacher) -> 'HomographicAdaptation':
        return HomographicAdaptation(self, teacher)


@dataclass
class HaResult:
    """One tile's aggregated label, in the TILE frame.

    `counts` is not a diagnostic -- it is the difference between "no view
    detected anything here" and "no view could see here", and those are opposite
    facts about the same zero. They diverge near the tile edge, where a label
    built from `mean_prob` alone would read as a genuinely featureless border.
    spec.md 3.1 says to store it, so it is returned rather than folded in.
    """
    mean_prob: np.ndarray        # [tile, tile] float32
    counts:    np.ndarray        # [tile, tile] float32, >= 1 (identity view)
    max_prob:  np.ndarray        # [tile, tile] float32
    n_views:   int
    aggregation: str = 'mean'   # which of the two the config asked for
    n_identity_only: int = 0     # draws that came back as the identity matrix

    @property
    def prob(self) -> np.ndarray:
        """Whichever aggregation the config asked for.

        Both are returned regardless, because they cost nothing to keep and
        re-running HA to see the other one costs `num` forwards per tile.
        """
        return self.select(self.aggregation)

    def select(self, aggregation: str) -> np.ndarray:
        if aggregation == 'mean':
            return self.mean_prob
        if aggregation == 'max':
            return self.max_prob
        raise ValueError(f'unknown aggregation {aggregation!r}; known: mean, max')


class HomographicAdaptation(IdentifiedBuild):
    """The loop. One instance per (teacher, config); `run` is per tile."""

    BASELINE = _HA_BASELINE

    def __init__(self, cfg: HaConfig, teacher):
        self.cfg = cfg
        self.teacher = teacher
        self.device = getattr(teacher, 'device', None)
        # The teacher's weights ARE part of this label's identity -- round 2 of
        # Stage A runs the same config against a different teacher, and the two
        # sets of labels must not share a store file.
        self.model = getattr(teacher, 'model', None)

    def identity_parts(self) -> List[str]:
        """HA's own fields, plus the whole teacher as one part.

        Not `super().identity_parts()`: that would hash the teacher's WEIGHTS
        (via `IdentifiedBuild.weights_id`) and drop its config. The teacher's own
        `identity_id()` already folds both, and using it keeps one definition of
        what a teacher is.
        """
        return (self.cfg.identity_parts(self.BASELINE) +
                [f'teacher={self.teacher.identity_id()}'])

    # ── one tile ──

    def run(self, pre_tile: np.ndarray, tile: int, *,
            rng: Optional[np.random.Generator] = None,
            factor: Optional[int] = None) -> HaResult:
        """Aggregate `cfg.num` views of one pre-tile into one label.

        Args:
            pre_tile: HxWx3 (or HxW) as stored by `PreTileStore`. Its side must
                be `tile * factor`.
            tile: the frame everything is expressed in, and what the teacher
                sees.
            factor: the pre-tile factor; inferred from the array when omitted.
        """
        rng = np.random.default_rng() if rng is None else rng
        pre = np.asarray(pre_tile)
        side = pre.shape[0]
        if pre.shape[0] != pre.shape[1]:
            raise ValueError(f'pre-tile must be square, got {pre.shape[:2]}')
        if factor is None:
            factor, remainder = divmod(side, int(tile))
            if remainder:
                raise ValueError(
                    f'a {side} px pre-tile is not a whole multiple of a {tile} '
                    f'px tile, so the tile has no exact centre in it')
        margin = centre_margin(int(tile), int(factor))

        shape = (int(tile), int(tile))
        prob_sum = np.zeros(shape, np.float32)
        prob_max = np.zeros(shape, np.float32)
        count_sum = np.zeros(shape, np.float32)

        # View 0 is the identity, with full coverage and no erosion -- upstream
        # `homographies.py:42-46`, where `counts = ones_like(probs)` and no mask
        # is applied. It is the only view whose pixels are the tile as stored.
        identity_view = centre_crop(pre, int(tile))
        prob0 = self._detect([identity_view])[0]
        prob_sum += prob0
        prob_max = np.maximum(prob_max, prob0)
        count_sum += 1.0

        n_identity_only = 0
        pending: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for _ in range(int(self.cfg.num) - 1):
            sample = sample_homography(shape, rng=rng,
                                       **self.cfg.homography_kwargs())
            if sample.is_identity:
                n_identity_only += 1
            matrix = sample.matrix
            inverse = invert(matrix)

            warped = self._warp_from_pre(pre, matrix, margin, shape)
            mask = self._source_mask(pre, matrix, margin, shape)
            count = erode_valid(
                warp_image(np.ones(shape, np.uint8), inverse,
                           interpolation='nearest', border_value=0),
                self.cfg.valid_border_margin).astype(np.float32)

            pending.append((warped, mask, count, inverse))
            if len(pending) >= int(self.cfg.batch):
                prob_sum, prob_max, count_sum = self._drain(
                    pending, prob_sum, prob_max, count_sum)
        prob_sum, prob_max, count_sum = self._drain(
            pending, prob_sum, prob_max, count_sum)

        mean_prob = prob_sum / np.maximum(count_sum, 1e-6)
        if self.cfg.filter_counts:
            keep = count_sum >= self.cfg.filter_counts
            mean_prob = np.where(keep, mean_prob, 0.0)
            prob_max = np.where(keep, prob_max, 0.0)

        result = HaResult(mean_prob=mean_prob.astype(np.float32),
                          counts=count_sum.astype(np.float32),
                          max_prob=prob_max.astype(np.float32),
                          n_views=int(self.cfg.num),
                          aggregation=self.cfg.aggregation,
                          n_identity_only=n_identity_only)
        return result

    # ── the pieces ──

    def _warp_from_pre(self, pre: np.ndarray, matrix: np.ndarray,
                       margin: int, shape: Tuple[int, int]) -> np.ndarray:
        """One warped view, sampled out of the pre-tile.

        `PreTileStore.warp_from_pretile` owns the composition, because the
        training pair dataset needs the identical one -- a student trained on
        pairs composed the other way round would be learning a different
        correspondence than its labels describe.
        """
        return warp_from_pretile(pre, matrix, margin, shape)

    def _source_mask(self, pre: np.ndarray, matrix: np.ndarray, margin: int,
                     shape: Tuple[int, int]) -> np.ndarray:
        """Which output pixels came from inside the PRE-tile, as float.

        See `PreTileStore.pretile_valid_mask` for why this is not
        `valid_mask(shape, matrix)`. Float rather than bool because it is
        multiplied into a probability map on the next line.
        """
        return pretile_valid_mask(pre, matrix, margin, shape,
                                  self.cfg.valid_border_margin).astype(np.float32)

    def _drain(self, pending, prob_sum, prob_max, count_sum):
        """Run the teacher on the queued views and accumulate them.

        Batched because a forward on one 256 px tile leaves a GPU idle, and HA
        does `num - 1` of them per tile and hundreds of thousands per rung. The
        accumulation is unchanged by it: sums are associative, and each view's
        contribution depends only on itself.
        """
        if not pending:
            return prob_sum, prob_max, count_sum
        probs = self._detect([view for view, _, _, _ in pending])
        for prob, (_, mask, count, inverse) in zip(probs, pending):
            projected = warp_image(prob * mask, inverse, interpolation='linear',
                                   border_value=0) * count
            prob_sum += projected
            prob_max = np.maximum(prob_max, projected)
            count_sum += count
        pending.clear()
        return prob_sum, prob_max, count_sum

    def _detect(self, views: List[np.ndarray]) -> np.ndarray:
        """[B, H, W] float32 out of the teacher, on the CPU."""
        prob = self.teacher.dense_prob(views)
        return prob.detach().float().cpu().numpy()

