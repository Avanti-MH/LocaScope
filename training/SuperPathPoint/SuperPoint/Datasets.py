"""The training pair: one tile, one warped view of it, and both label maps.

    ds = PairDatasetConfig().build(tiles_root, labels_root, wsi_stems, rungs)
    batch = ds[0]
        image, warped_image        [C, tile, tile] float in [0, 1]
        keypoint_map, warped_...   [tile, tile] float 0/1
        valid_mask, warped_...     [tile, tile] float 0/1
        homography                 [3, 3] float, OUTPUT -> INPUT in TILE pixels

THE LABEL IS WARPED AS POINTS AND RE-SPLATTED, NOT AS AN IMAGE
----------------------------------------------------------------
`warp_points` -> `filter_points` -> `add_keypoint_map`
(`datasets/utils/pipeline.py:40-56`, `models/homographies.py:308-311`). Warping
the keypoint MAP instead is the mistake with a delay on it: a keypoint is one
pixel, bilinear interpolation spreads it over four at a quarter the height, and
`> 0.5` then deletes most of them. Training proceeds against a label set that
loses points every epoch, in a pattern that depends on the sampled homography,
and nothing raises. Nearest-neighbour warping is not a fix either -- it drops
points wherever the warp is expanding.

The points go through the geometry; only the IMAGE is resampled.

THE PAIR IS TWO CROPS OF ONE PRE-TILE
---------------------------------------
The identity view is `centre_crop(pre)`. The warped view is
`warp_from_pretile(pre, H, margin)`, the same composition Homographic
Adaptation used to produce the labels (spec.md 6.6). Composing it differently
here would train the student on a correspondence its labels do not describe --
and both compositions produce a perfectly ordinary-looking warped tile.

So `valid_mask` on the warped side is nearly all ones, and that is expected:
with a 3x source there is nothing outside to sample. What it still marks is the
eroded rim, which is upstream's `valid_border_margin` and is about the
detector's receptive field rather than about the border fill.

RUNG BALANCE IS A SWITCH, AND THE PROBE DECIDES IT
----------------------------------------------------
ds 1 may yield 500 tiles per slide while ds 32 yields 40 (spec.md 6.5). Doing
nothing is the worst option available, because the fine rungs then get an order
of magnitude more training and NOTHING SHOWS IT.

    'none'         take what there is. Recorded, so at least the CSV says so
    'align-min'    every rung truncated to the smallest rung's count. Perfectly
                   balanced, and it can throw away nine tenths of the data
    'loss-weight'  keep everything; `rung_weight` is proportional to 1/count and
                   the detector CE is weighted by it. All the data, and the
                   noise of the sparse coarse rungs is amplified with it

The probe of spec.md 12 step 3b decides which -- worst cell at 300 tiles makes
`align-min` cheap, worst cell at 40 makes it unaffordable. This class implements
all three and picks none of them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ConfigIdentity import IdentifiedConfig, register

from common import KeypointLabelStore
import PreTileStore
from common.Homography import inside, points_input_to_output, sample_homography
from common.HomographyConfig import HOMOGRAPHY_BASELINE, HomographyConfig
from PreTileStore import (centre_crop, centre_margin,
                                 pretile_valid_mask, warp_from_pretile)

BALANCE_MODES = ('none', 'align-min', 'loss-weight')

#: The zero point. ConfigIdentity rule 1.
_PAIR_BASELINE = {
    'method': 'superpathpoint-pairs',
    'tile': 256,
    'in_channels': 1,
    'valid_border_margin': 3,
    'balance': 'loss-weight',
    'homography': HOMOGRAPHY_BASELINE,
}


@register('superpathpoint-pairs')
@dataclass(frozen=True)
class PairDatasetConfig(IdentifiedConfig):
    method: str = 'superpathpoint-pairs'

    tile: int = 256

    #: 1 for the grayscale student, 3 for the RGB one. Two students, two
    #: checkpoints, ONE set of labels -- HA produces coordinates, and a
    #: coordinate does not care how many channels the image had (spec.md 13).
    in_channels: int = 1

    #: Upstream's 3. Applied to the warped side's valid mask.
    valid_border_margin: int = 3

    #: RE-SETTLED 2026-08-27, THE OTHER WAY (spec.md 6.5). The rule was written
    #: before the numbers: align-min if the worst rung still holds hundreds,
    #: loss-weight if it holds forty. The 2026-08-26 probe answered "hundreds"
    #: -- ds 32 came back 1784 of 2000 over four slides -- and that number was
    #: the TISSUE GATE, not the slides. The gate admitted only background <= 50
    #: per cent and the sampler then asked for 500 positions it had already been
    #: handed, so 1784 measured how fast a rejection budget ran out.
    #:
    #: The 12-slide probe measures the candidate pool instead: ds 32 offers 18
    #: disjoint positions on the worst slide and 583 over all twelve. align-min
    #: truncates every rung to the worst cell, so 6 rungs x 12 slides x 18 =
    #: 1,296 tiles against 8,465 available at ds 8 alone. That is not balancing
    #: a ladder, it is deleting one.
    #:
    #: So 'none' -- and not yet 'loss-weight' either, because the rung weights
    #: are a second decision and the corpus had to exist first. The imbalance is
    #: a recorded property of the v1 run rather than a later discovery. What
    #: would change it is more slides, not this switch.
    #:
    #: The DEFAULT moves with the decision and `_PAIR_BASELINE` does not. The
    #: baseline is the zero point an identity is measured against and it is
    #: append-only (ConfigIdentity rule 1); the default is what runs.
    balance: str = 'none'

    #: The same thirteen options Homographic Adaptation drew its views with.
    #: Nested so the two cannot drift: see `common/HomographyConfig.py`.
    homography: HomographyConfig = field(default_factory=HomographyConfig)

    #: Throughput and reproducibility, not identity. The seed is NOT identity
    #: because a dataset is a distribution, not a fixed list -- two seeds draw
    #: two samples of the same pairs from the same tiles.
    seed: int = 0
    workers: int = 4

    NOT_IDENTITY = ('seed', 'workers')

    def build(self, tiles_root, labels_root, *, wsi_stems: Sequence[str],
              rungs: Optional[Sequence[float]] = None,
              ha_id: Optional[str] = None) -> 'HomographyPairDataset':
        return HomographyPairDataset(self, tiles_root, labels_root,
                                     wsi_stems=wsi_stems, rungs=rungs,
                                     ha_id=ha_id)


@dataclass
class PairItem:
    """One index entry. Deliberately small: this list is held per worker."""
    folder: Path
    record: PreTileStore.PreTileRecord
    points: np.ndarray            # [n, 2] int16, tile coordinates
    rung: float
    rung_index: int
    pre_tile_factor: int
    #: Which slide this tile came from. Carried per item because validation is
    #: reported PER SLIDE (spec.md 13): tiles of one slide share a staining
    #: batch, a scanner, a section thickness and a tissue source, so two slides
    #: averaged into one number can hide a difference that is entirely one
    #: slide's personality. `slide_index` is what `Trainer.validate` groups on;
    #: `slide` is what the row is labelled with.
    slide: str = ''
    slide_index: int = -1


class HomographyPairDataset(Dataset):
    """Pre-tiles plus HA labels in, `(I, I', H, label, label')` out.

    FORK-SAFE BY CONSTRUCTION, which is the whole reason step 3c exists: the
    only files a worker opens are PNGs. No OpenSlide handle crosses a fork, and
    the MIRAX reopen storm that `log/TODO.log` measured at 752 reopens in one
    read cannot happen here (spec.md 6.5).
    """

    def __init__(self, cfg: PairDatasetConfig, tiles_root, labels_root, *,
                 wsi_stems: Sequence[str],
                 rungs: Optional[Sequence[float]] = None,
                 ha_id: Optional[str] = None):
        if cfg.balance not in BALANCE_MODES:
            raise ValueError(
                f'balance must be one of {BALANCE_MODES}, got {cfg.balance!r}')
        self.cfg = cfg
        self.items: List[PairItem] = []

        rung_values = sorted({float(r) for r in rungs}) if rungs else None
        found_rungs: List[float] = []

        # ONE STORE PER (slide, rung), REFUSED RATHER THAN UNIONED. A root
        # legitimately holds two corpora of the same slides -- `sampler_id` is
        # in the directory name, so re-extracting at another setting adds a
        # directory rather than replacing one -- and this loop would then read
        # BOTH and call the union a corpus. It would not error: it would report
        # twice the tiles and a bucket distribution that is neither corpus's.
        #
        # `KeypointLabelStore.find_one` refuses the same shape for the same
        # reason ("a labels root legitimately holds round-1 and round-2 labels
        # of the same tiles"); this side had no guard until 2026-09-01.
        seen: Dict[Tuple[str, float], Path] = {}
        for folder in sorted(PreTileStore.find(tiles_root, tile=int(cfg.tile))):
            meta = PreTileStore.load_meta(folder)
            if meta.wsi_stem not in set(wsi_stems):
                continue
            if rung_values is not None and not any(
                    abs(meta.ds - r) < 1e-6 for r in rung_values):
                continue
            key = (meta.wsi_stem, float(meta.ds))
            if key in seen:
                raise ValueError(
                    f'two pre-tile stores for {meta.wsi_stem} at ds '
                    f'{meta.ds:g} under {tiles_root}:\n'
                    f'  {seen[key].name}\n  {folder.name}\n'
                    f'They were cut at different sampler settings, so their '
                    f'union is not a corpus -- it is two corpora with one '
                    f'name. Point --tiles-root at one of them, or move the '
                    f'other aside')
            seen[key] = folder

            query = dict(wsi_stem=meta.wsi_stem, ds=meta.ds,
                         pretile_id=meta.cfg_hash())
            if ha_id:
                query['ha_id'] = ha_id
            # find_one and not find()[0]: a labels root legitimately holds
            # round-1 and round-2 labels of the same tiles, and picking whichever
            # sorted first would train round 3 on round 1.
            batch, _ = KeypointLabelStore.load(
                KeypointLabelStore.find_one(labels_root, **query))

            records = {(r.x, r.y): r for r in PreTileStore.load_index(folder)}
            if meta.ds not in found_rungs:
                found_rungs.append(meta.ds)
            for i in range(len(batch)):
                key = (int(batch.tile_x[i]), int(batch.tile_y[i]))
                record = records.get(key)
                if record is None:
                    # The label store names a position the pre-tile store does
                    # not have. Skipping would hide a store pair that does not
                    # belong together, which is exactly what `pretile_id` is
                    # supposed to make impossible.
                    raise KeyError(
                        f'{folder.name} has no pre-tile at {key}, but its '
                        f'labels list one. The two stores do not belong '
                        f'together despite matching pretile_id')
                self.items.append(PairItem(
                    folder=folder, record=record, points=batch.points_of(i),
                    rung=meta.ds, rung_index=-1,
                    pre_tile_factor=int(meta.pre_tile_factor),
                    slide=meta.wsi_stem))

        self.rungs = sorted(found_rungs)
        index_of = {r: i for i, r in enumerate(self.rungs)}
        # Numbered off `wsi_stems` and not off what was found, so that the index
        # of a slide does not move when a rung filter happens to exclude every
        # tile of one -- a per-slide row keyed by a shifting integer is a row
        # that silently changes which slide it describes.
        self.slides = [str(s) for s in wsi_stems]
        slide_of = {s: i for i, s in enumerate(self.slides)}
        for item in self.items:
            item.rung_index = index_of[item.rung]
            item.slide_index = slide_of[item.slide]

        self.counts = Counter(item.rung for item in self.items)
        self._apply_balance()
        self.rung_weight = self._weights()

        #: Which epoch's draw `__getitem__` is making. See `set_epoch`.
        self.epoch = 0

    # ── balance ──

    def _apply_balance(self) -> None:
        if self.cfg.balance != 'align-min' or not self.items:
            return
        smallest = min(self.counts.values())
        rng = np.random.default_rng(self.cfg.seed)
        kept: List[PairItem] = []
        for rung in self.rungs:
            pool = [i for i in self.items if i.rung == rung]
            # A random subset, not the first `smallest`. The index is built in
            # position order, so taking a prefix would take one region of every
            # slide and call it a balanced sample.
            choice = rng.choice(len(pool), size=smallest, replace=False)
            kept += [pool[int(k)] for k in sorted(choice)]
        self.items = kept
        self.counts = Counter(item.rung for item in self.items)

    def _weights(self) -> torch.Tensor:
        """One weight per rung, in `self.rungs` order, mean 1.

        Mean 1 rather than sum 1 so that turning weighting on does not also
        change the overall size of the detector term -- otherwise the three loss
        magnitudes that spec.md 12 step 6 asks to be read after the first epoch
        would move for two reasons at once.
        """
        weights = torch.ones(len(self.rungs))
        if self.cfg.balance != 'loss-weight' or not self.items:
            return weights
        counts = torch.tensor([float(self.counts[r]) for r in self.rungs])
        weights = counts.mean() / counts.clamp_min(1.0)
        return weights / weights.mean()

    def summary(self) -> str:
        parts = ', '.join(f'ds{r:g}:{self.counts[r]}' for r in self.rungs)
        return (f'{len(self.items)} pairs   {parts}   balance='
                f'{self.cfg.balance}   weights='
                f'{[round(float(w), 2) for w in self.rung_weight]}')

    # ── one pair ──

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch: int) -> None:
        """Which pass this is, so the augmentation draws a NEW warp each time.

        MUST BE CALLED BEFORE THE EPOCH'S LOADER IS ITERATED, and the loader
        must not hold persistent workers. A DataLoader worker is a forked copy
        of this object: with `persistent_workers=True` the copy is made once and
        an assignment here never reaches it, so every epoch would silently reuse
        epoch 0's warps -- the exact defect this method exists to fix, wearing
        the disguise of a fix. `Trainer._loader` therefore passes
        `persistent_workers=False` for anything it calls this on.

        The validation set is NEVER told an epoch. Its warps are fixed by
        construction, because a metric whose input changes every epoch cannot say
        whether the model moved or the data did.
        """
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        item = self.items[index]
        tile = int(cfg.tile)
        shape = (tile, tile)
        margin = centre_margin(tile, item.pre_tile_factor)

        pre = PreTileStore.read_tile(item.folder, item.record)
        image = centre_crop(pre, tile)

        # Seeded per (seed, ITEM, EPOCH). The index term is what makes a
        # worker's draw independent of how many items it happened to see first,
        # so two workers reading the same index get the same pair and a run
        # reproduces under a changed `workers`.
        #
        # THE EPOCH TERM WAS MISSING UNTIL 2026-08-31 AND IT COST A RUN. Without
        # it the seed is the same on every pass, so a tile is shown the SAME
        # warp for the whole run: 50 epochs over 5,344 pairs is 5,344 distinct
        # pairs seen 50 times, not 267,200. Homographic Adaptation's entire
        # premise is that the same content under different geometry is what
        # teaches invariance, and the augmentation was delivering one geometry
        # per tile. It showed as textbook overfitting -- `train/detector` fell
        # monotonically to the last epoch while `val/detector` bottomed at 3.12
        # on epoch 42 and rose to 3.27.
        rng = np.random.default_rng((int(cfg.seed), index, int(self.epoch)))
        sample = sample_homography(shape, rng=rng, **cfg.homography.kwargs())

        warped = warp_from_pretile(pre, sample.matrix, margin, shape)
        warped_valid = pretile_valid_mask(pre, sample.matrix, margin, shape,
                                          cfg.valid_border_margin)

        points = np.asarray(item.points, np.float32).reshape(-1, 2)
        warped_points = (points_input_to_output(points, sample.matrix)
                         if len(points) else points)
        if len(warped_points):
            warped_points = warped_points[inside(warped_points, shape)]

        return {
            'image': _to_tensor(image, cfg.in_channels),
            'warped_image': _to_tensor(warped, cfg.in_channels),
            'homography': torch.from_numpy(
                np.ascontiguousarray(sample.matrix, np.float32)),
            'keypoint_map': torch.from_numpy(splat(points, shape)),
            'warped_keypoint_map': torch.from_numpy(splat(warped_points, shape)),
            'valid_mask': torch.ones(shape, dtype=torch.float32),
            'warped_valid_mask': torch.from_numpy(
                warped_valid.astype(np.float32)),
            'rung': torch.tensor(float(item.rung)),
            'rung_index': torch.tensor(int(item.rung_index)),
            'slide_index': torch.tensor(int(item.slide_index)),
        }


def splat(points_xy: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """`[n, 2]` (x, y) -> `[H, W]` float 0/1. Upstream's `add_keypoint_map`.

    Rounds and then CLAMPS to the last pixel (`pipeline.py:68`), so a point at
    exactly the edge lands on the edge instead of raising or wrapping. Note the
    axis order: upstream's `keypoints` are (row, col) because `scatter_nd`
    indexes that way, and everything in this project is (x, y) -- the swap
    happens here, once, in the line below.
    """
    out = np.zeros(shape, np.float32)
    if len(points_xy) == 0:
        return out
    xy = np.rint(np.asarray(points_xy, np.float64)).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, shape[0] - 1)
    out[xy[:, 1], xy[:, 0]] = 1.0
    return out


def _to_tensor(image: np.ndarray, in_channels: int) -> torch.Tensor:
    """HxWx3 uint8 -> `[C, H, W]` float in [0, 1], with upstream's luma.

    The same three coefficients as `Teacher._to_gray` and
    `Backbones.to_model_channels`. They are three copies of one constant and
    that is deliberate: folding them into a shared helper would put a torch
    import into whichever module held it, and two of the three are in modules
    that are careful about that. The number to keep in step is 0.299/0.587/0.114
    and it is upstream's.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    tensor = tensor.float() / 255.0
    if tensor.shape[0] == in_channels:
        return tensor
    if tensor.shape[0] == 3 and in_channels == 1:
        scale = tensor.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
        return (tensor * scale).sum(0, keepdim=True)
    raise ValueError(
        f'cannot turn {tensor.shape[0]} channels into {in_channels}')
