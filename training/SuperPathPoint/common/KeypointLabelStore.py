"""Homographic Adaptation's output on disk: one sparse point set per tile.

    points = points_from_prob(result.mean_prob, result.counts, cfg)
    LabelStore.save(root, batch, LabelMeta.of(...))

    batch, meta = LabelStore.load(path, require={'wsi_stem': 'BRACS_1228'})
    xy = batch.points_of(i)          # [n_kp[i], 2] int16, tile coordinates

SPARSE, NOT DENSE, AND THE ARITHMETIC THAT DECIDES IT
-------------------------------------------------------
A 256x256 fp16 probability map is 128 KB per tile. One slide, one rung, ten
thousand tiles is 1.3 GB, and a 1024 tile is sixteen times that. The points that
survive a threshold are a few hundred per tile: two int16 and two more small
numbers each, about 2 KB. spec.md 6.3.

What is given up is the ability to re-threshold later, which is why the
threshold is an identity field and `n_kp` is stored: a store that is at its cap
is a store that was truncated, and only the recorded count can say so.

`kp_count` IS STORED, AND IT IS NOT A DIAGNOSTIC
-------------------------------------------------
It separates "no view detected anything at this pixel" from "no view could SEE
this pixel", and those are opposite facts wearing the same zero (spec.md 3.1).
They come apart at the tile edge, where coverage is thin: without it, a label
built from the score alone reads the edge as genuinely featureless, and the
student learns to have nothing to say there.

M IS A DENSITY, NOT A COUNT
----------------------------
Three tile sizes are three separate models (spec.md 6.5), and a fixed M would
store the same number of points in a 1024 tile as in a 256 one -- a sixteenfold
difference in density, straight into any comparison between the three. So the
cap is `points_per_megapixel * H * W`, and it is a MEMORY cap: the threshold is
what selects, and `n_kp` is what happened.

That second role is the one that is easy to miss. Every kept point becomes a
positive cell for the detector cross-entropy and everything else becomes
dustbin, so M is not a storage parameter -- it is the density of the target the
student is trained to reproduce.

THE ARROW POINTS THIS WAY ON PURPOSE
-------------------------------------
`points_from_prob` lives here rather than in `SuperPoint/KeypointNet.py`, and
that module's `extract_keypoints` calls it. The teacher's labels and the
student's predictions have to be turned into points by the SAME rule -- an NMS
radius that differs between them makes every repeatability number a comparison
of two conventions. `common/` is the layer both stages already depend on.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors.numpy import save_file
from safetensors import safe_open

#: Bumped when a field changes MEANING. Adding one with a default does not need
#: it. `MaskStore.SCHEMA_VERSION`'s rule, which is ConfigIdentity's first.
SCHEMA_VERSION = '1'

#: What makes two label sets different. `ha_id` folds the whole adaptation --
#: its config AND the teacher's config AND the teacher's weights -- so nothing
#: here has to enumerate `num` or `patch_ratio` and then fail to be updated when
#: a field appears. `pretile_id` is in because the POSITIONS come from a
#: pre-tile store: same slide, same rung, different seed is a different set of
#: tiles and therefore a different set of labels.
_IDENTITY_FIELDS = ('wsi_stem', 'ds', 'tile', 'ha_id', 'pretile_id',
                    'score_threshold', 'points_per_megapixel', 'nms_radius',
                    'border')


class LabelMismatch(RuntimeError):
    """`load(require=...)` was handed a store that is not the one asked for."""


# ── turning a map into points ────────────────────────────────────────────────

def nms_max_pool(scores: torch.Tensor, radius: int) -> torch.Tensor:
    """Upstream's `batched_nms`, transcribed (`superpoint_pytorch.py:25-40`).

    Not a plain "keep local maxima": the loop runs twice, and the second pass is
    what lets a point that lost to a suppressed neighbour come back. Dropping
    the loop gives a result that is still a set of sparse peaks -- fewer of them,
    in a pattern that depends on how ties fell -- so it looks like a stricter
    NMS rather than a different algorithm.
    """
    if radius < 0:
        raise ValueError(f'nms radius must be >= 0, got {radius}')
    if radius == 0:
        return scores

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=radius * 2 + 1, stride=1, padding=radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def points_from_prob(prob: np.ndarray, counts: Optional[np.ndarray] = None, *,
                     score_threshold: float = 0.005,
                     nms_radius: int = 4,
                     border: int = 4,
                     max_points: Optional[int] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`[H, W]` map -> `(xy int16 [n, 2], score float32 [n], count uint8 [n])`.

    The order is upstream's and it is not interchangeable: NMS first, then the
    border cut, then the threshold, then the cap (`superpoint_pytorch.py:
    122-137`). Thresholding before NMS would let a wide low plateau contribute
    nothing while a sharp peak just under the threshold is also dropped;
    cutting the border after the cap would spend the budget on points that are
    then thrown away.

    `max_points` truncates by SCORE, keeping the highest. Returning fewer than
    `max_points` is the normal case and is what `n_kp` records; returning
    exactly `max_points` is the case to be suspicious of.
    """
    scores = torch.from_numpy(np.ascontiguousarray(prob, dtype=np.float32))
    scores = nms_max_pool(scores[None, None], int(nms_radius))[0, 0]

    if border:
        pad = int(border)
        # -1, not 0: upstream's own value (`superpoint_pytorch.py:126-130`), and
        # it matters because the threshold below is a `>`, so a zeroed border
        # would still pass a threshold of exactly 0.
        scores[:pad] = -1
        scores[-pad:] = -1
        scores[:, :pad] = -1
        scores[:, -pad:] = -1

    rows, cols = torch.where(scores > float(score_threshold))
    values = scores[rows, cols]

    if max_points is not None and values.numel() > int(max_points):
        values, order = torch.topk(values, int(max_points), sorted=True)
        rows, cols = rows[order], cols[order]

    xy = torch.stack([cols, rows], dim=-1).numpy().astype(np.int16)   # (x, y)
    score = values.numpy().astype(np.float32)

    if counts is None:
        count = np.zeros(len(score), np.uint8)
    else:
        raw = np.asarray(counts)[xy[:, 1], xy[:, 0]] if len(xy) else np.zeros(0)
        # uint8 saturates at 255, and num can exceed that in principle. Clipping
        # is stated rather than wrapped: 255 reads as "seen by everything",
        # while a wrap would read as "seen by four".
        count = np.clip(raw, 0, 255).astype(np.uint8)
    return xy, score, count


def cap_for(tile: int, points_per_megapixel: float) -> int:
    """The per-tile point cap, as a density. See the module docstring."""
    return max(1, int(round(points_per_megapixel * tile * tile / 1e6)))


# ── what a batch of labels is ────────────────────────────────────────────────

@dataclass
class LabelBatch:
    """Every tile of one (slide, rung), padded to a rectangle.

    Padded rather than ragged because safetensors holds tensors, not lists of
    them, and because the training loader wants a fixed shape anyway. `n_kp` is
    the only thing that says where the padding starts -- reading `kp_xy` without
    it gives a tile full of points at (0, 0), which is a plausible-looking
    corner rather than an error.
    """
    tile_x:   np.ndarray      # int32 [T]   level-0 top-left of the TILE
    tile_y:   np.ndarray      # int32 [T]
    kp_xy:    np.ndarray      # int16 [T, M, 2]  in-tile (x, y)
    kp_score: np.ndarray      # float16 [T, M]   the aggregated probability
    kp_count: np.ndarray      # uint8 [T, M]     views that could see it
    n_kp:     np.ndarray      # int16 [T]        real points; the rest is padding

    def __len__(self) -> int:
        return int(len(self.n_kp))

    def points_of(self, index: int) -> np.ndarray:
        return self.kp_xy[index, :int(self.n_kp[index])]

    def scores_of(self, index: int) -> np.ndarray:
        return self.kp_score[index, :int(self.n_kp[index])]

    def counts_of(self, index: int) -> np.ndarray:
        return self.kp_count[index, :int(self.n_kp[index])]

    @property
    def cap(self) -> int:
        return int(self.kp_xy.shape[1])

    @property
    def at_cap(self) -> int:
        """Tiles whose point list was truncated. The number to look at first.

        If it is not near zero the store is lossy in the one way that cannot be
        undone: the points that fell off were the lowest-scoring survivors of
        the threshold, and getting them back means re-running HA.
        """
        return int((self.n_kp >= self.cap).sum())



def batch_from_lists(tile_xy: List[Tuple[int, int]],
                     points: List[np.ndarray],
                     scores: List[np.ndarray],
                     counts: List[np.ndarray],
                     cap: int) -> LabelBatch:
    """Pad per-tile point lists into one rectangle.

    The cap is passed rather than taken as `max(len(p))`, so that a rung where
    every tile happened to yield few points does not silently write a store with
    a smaller M than the config asked for -- two stores of the same identity
    with different second dimensions.
    """
    count = len(tile_xy)
    if not (len(points) == len(scores) == len(counts) == count):
        raise ValueError(
            f'{count} positions but {len(points)} point lists, {len(scores)} '
            f'score lists and {len(counts)} count lists')

    kp_xy = np.zeros((count, cap, 2), np.int16)
    kp_score = np.zeros((count, cap), np.float16)
    kp_count = np.zeros((count, cap), np.uint8)
    n_kp = np.zeros(count, np.int16)

    for i, (xy, score, seen) in enumerate(zip(points, scores, counts)):
        keep = min(len(xy), cap)
        kp_xy[i, :keep] = np.asarray(xy[:keep], np.int16)
        kp_score[i, :keep] = np.asarray(score[:keep], np.float16)
        kp_count[i, :keep] = np.asarray(seen[:keep], np.uint8)
        n_kp[i] = keep

    xy = np.asarray(tile_xy, np.int64).reshape(count, 2)
    return LabelBatch(tile_x=xy[:, 0].astype(np.int32),
                      tile_y=xy[:, 1].astype(np.int32),
                      kp_xy=kp_xy, kp_score=kp_score, kp_count=kp_count,
                      n_kp=n_kp)


# ── what makes one label set not another ─────────────────────────────────────

@dataclass(frozen=True)
class LabelMeta:
    wsi_stem:  str
    ds:        float
    tile:      int
    ha_id:     str        # HomographicAdaptation.identity_id(): HA cfg + teacher
    pretile_id: str       # PreTileMeta.cfg_hash(): which positions

    #: The two knobs that decide what a point IS. Identity, because a store cut
    #: at one threshold cannot be re-cut at another.
    score_threshold:      float = 0.005
    points_per_megapixel: float = 0.0
    nms_radius:           int = 4
    border:               int = 4

    #: Provenance and facts about the arrays.
    wsi_path:  str = ''
    n_tiles:   int = 0
    cap:       int = 0
    n_at_cap:  int = 0
    mean_n_kp: float = 0.0
    aggregation: str = 'mean'

    created_at:     str = ''
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def of(cls, batch: LabelBatch, *, wsi_stem: str, ds: float, tile: int,
           ha, pretile_meta, score_threshold: float,
           points_per_megapixel: float, nms_radius: int = 4, border: int = 4,
           aggregation: str = 'mean', wsi_path: str = '') -> 'LabelMeta':
        """Read the identity off the objects that determine it.

        `ha` is the built `HomographicAdaptation` and not its config: only the
        built one knows which teacher it holds, and round 2 of Stage A differs
        from round 1 in exactly that.
        """
        return cls(
            wsi_stem=wsi_stem, ds=float(ds), tile=int(tile),
            ha_id=ha.identity_id(), pretile_id=pretile_meta.cfg_hash(),
            score_threshold=float(score_threshold),
            points_per_megapixel=float(points_per_megapixel),
            nms_radius=int(nms_radius), border=int(border),
            wsi_path=str(wsi_path), aggregation=str(aggregation),
            n_tiles=len(batch), cap=batch.cap, n_at_cap=batch.at_cap,
            mean_n_kp=float(np.mean(batch.n_kp)) if len(batch) else 0.0)

    def cfg_hash(self) -> str:
        parts = [f'{n}={getattr(self, n)}' for n in sorted(_IDENTITY_FIELDS)]
        return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:8]

    def filename(self) -> str:
        return f'{self.wsi_stem}__ds{self.ds:g}__{self.cfg_hash()}.safetensors'

    def to_strings(self) -> Dict[str, str]:
        return {f.name: str(getattr(self, f.name))
                for f in dataclasses.fields(self)}

    @classmethod
    def from_strings(cls, d: Dict[str, str]) -> 'LabelMeta':
        kwargs = {}
        for field in dataclasses.fields(cls):
            if field.name not in d:
                continue          # a field this build no longer has, or gained
            raw = d[field.name]
            kwargs[field.name] = (int(raw) if field.type == 'int' else
                                  float(raw) if field.type == 'float' else raw)
        return cls(**kwargs)


# ── write ────────────────────────────────────────────────────────────────────

def save(root, batch: LabelBatch, meta: LabelMeta) -> Path:
    """Validate, then write atomically to root/<meta.filename()>."""
    if len(batch) != meta.n_tiles:
        raise ValueError(
            f'meta says {meta.n_tiles} tiles, the batch has {len(batch)}. '
            f'Build the meta with LabelMeta.of(batch, ...)')
    if not meta.ha_id:
        raise ValueError(
            'ha_id is empty, so these labels cannot say which teacher and which '
            'adaptation produced them -- and round 2 of Stage A writes labels '
            'that differ in nothing else')
    if batch.kp_xy.shape[:2] != batch.kp_score.shape:
        raise ValueError(
            f'kp_xy is {batch.kp_xy.shape[:2]} and kp_score is '
            f'{batch.kp_score.shape}; they index the same points')

    meta = dataclasses.replace(
        meta, created_at=meta.created_at or time.strftime('%Y-%m-%dT%H:%M:%S'))

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / meta.filename()
    tmp = path.with_suffix('.tmp')
    save_file({'tile_x': np.ascontiguousarray(batch.tile_x),
               'tile_y': np.ascontiguousarray(batch.tile_y),
               'kp_xy': np.ascontiguousarray(batch.kp_xy),
               'kp_score': np.ascontiguousarray(batch.kp_score),
               'kp_count': np.ascontiguousarray(batch.kp_count),
               'n_kp': np.ascontiguousarray(batch.n_kp)},
              str(tmp), metadata=meta.to_strings())
    os.replace(tmp, path)
    return path


# ── read ─────────────────────────────────────────────────────────────────────

def load_meta(path) -> LabelMeta:
    with safe_open(str(path), framework='numpy') as handle:
        md = handle.metadata()
    if md is None:
        raise LabelMismatch(f'{path} has no metadata -- not a label store')
    return LabelMeta.from_strings(md)


def load(path, *, require: Optional[Dict[str, object]] = None
         ) -> Tuple[LabelBatch, LabelMeta]:
    """Return (LabelBatch, LabelMeta), refusing outright when `require` misses.

    Never falls back, for `FeatureStore.load`'s reason: labels that are not the
    ones asked for train a model that converges on the wrong target, and the
    only symptom is a number that is worse than expected.
    """
    meta = load_meta(path)
    if require:
        bad = {k: (v, getattr(meta, k, '<no such field>'))
               for k, v in require.items() if getattr(meta, k, None) != v}
        if bad:
            lines = '\n'.join(f'  {k}: wanted {w!r}, store has {g!r}'
                              for k, (w, g) in sorted(bad.items()))
            raise LabelMismatch(f'{path}\n{lines}')

    with safe_open(str(path), framework='numpy') as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    return LabelBatch(**tensors), meta


def find(root, **eq) -> List[Path]:
    """Label files under `root` matching the given fields, by metadata."""
    hits = []
    for candidate in sorted(Path(root).glob('*.safetensors')):
        try:
            meta = load_meta(candidate)
        except Exception:                                        # noqa: BLE001
            continue                      # not ours; leave other files alone
        if all(getattr(meta, k, None) == v for k, v in eq.items()):
            hits.append(candidate)
    return hits


def find_one(root, **eq) -> Path:
    """The single label file matching `eq`, or an error naming the ones that did.

    Never `find()[0]`: a root legitimately holds round-1 and round-2 labels of
    the same slide and rung -- they differ in `ha_id` and coexist -- and picking
    whichever sorted first would train round 3 on round 1.
    """
    hits = find(root, **eq)
    if len(hits) == 1:
        return hits[0]
    query = ', '.join(f'{k}={v!r}' for k, v in sorted(eq.items())) or '(no filter)'
    if not hits:
        raise LabelMismatch(
            f'no labels under {root} matching {query}. Present: '
            f'{[p.name for p in sorted(Path(root).glob("*.safetensors"))]}')
    raise LabelMismatch(
        f'{len(hits)} label sets under {root} match {query}, and this call '
        f'needs one. Narrow it with ha_id=... :\n' +
        '\n'.join(f'  {p.name}   ha {load_meta(p).ha_id}' for p in hits))
