'''What a candidate window IS, relative to where a synthetic FoV was placed:
the labels every loss and every recall figure of the second phase reads. They are the bench's answer
(`BenchCommon.answers`), so what training calls right is what the benches count as found.

    pos      the window is an answer -- the main-lattice window or the offset-lattice window nearest the
             centre of the photo's tile grid, at the rotation that sets the photo upright -- or, in a list
             a suppression cut, the window that kept the answer's place (the same rotation, nearer to it
             than the suppression radius: `BenchCommon.listed_places`)
    near     the window is at the right rotation, is not an answer, and its centre is within `near_tiles`
             tiles of the truth's. It is no negative -- a window one tile over holds most of the FoV -- but
             it is not found either, so the truth has to beat it, and by more the farther it is: a soft
             target `near_alpha * exp(-d^2 / 2 sigma^2)` (d in tiles) against the answers' 1
    neg      every other window: far from the truth, or at the wrong rotation (stage 3 needs it right)

Boxes are in level-0 px and all areas, so nothing here knows how many tiles a FoV has.
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from locascope.core.stage2_retrieval.Context import iou_matrix

#: A window's identity and the level-0 point it starts at (`BenchCommon.Answer.key`, `.origin`).
Key = Tuple[int, str, int, int, int]


@dataclass(frozen=True)
class LabelConfig:
    near_tiles: float = 2.0     # centre distance (tiles) up to which a non-answer window is `near`
    near_alpha: float = 0.3     # the soft target of a near window at distance 0, against the answers' 1
    near_sigma: float = 1.0     # tiles; the soft target falls as exp(-d^2 / 2 sigma^2)
    loc_min_iou: float = 0.1    # IoU with the truth box for a window that has an offset to learn
    fov_margin: float = 1.2     # the FoV box is the photo's circumscribed square times this


@dataclass
class Boxes:
    """Where a FoV was placed, as boxes `(x0, y0, x1, y1)` in level-0 px, and its answers."""
    truth: np.ndarray           # [4]
    fov: np.ndarray             # [4]
    rotation: int               # the stage-2 rotation that sets the photo upright
    centre: Tuple[float, float]
    answers: Tuple[Tuple[Key, Tuple[int, int]], ...] = ()    # (key, level-0 origin) of the main and offset answers


def truth_boxes(centre: Tuple[float, float], rotation: int, rows: int, cols: int, tile: int, ds: float,
                fov_centre: Tuple[float, float], fov_w0: float, fov_h0: float,
                cfg: LabelConfig, answers: Sequence[Tuple[Key, Tuple[int, int]]] = ()) -> Boxes:
    """The truth box (the query's `rows x cols` tile grid at `rotation`, centred on `centre`), the FoV box
    (the square around the photo's centre whose side is its diagonal, times `fov_margin`) and the answers."""
    cx, cy = centre
    hw, hh = cols * tile * ds / 2.0, rows * tile * ds / 2.0
    side: float = float(np.hypot(fov_w0, fov_h0)) * cfg.fov_margin / 2.0
    fx, fy = fov_centre
    return Boxes(truth=np.array([cx - hw, cy - hh, cx + hw, cy + hh]),
                 fov=np.array([fx - side, fy - side, fx + side, fy + side]),
                 rotation=int(rotation), centre=(float(cx), float(cy)), answers=tuple(answers))


@dataclass
class Labels:
    """Per candidate, `[K]`. `soft` is a near window's target (0 elsewhere), `covers` where the window is within
    `near_tiles` of the truth at any rotation (where "right rotation" is worth learning), `loc` the truth's
    centre minus the window's, in tiles, and `loc_mask` where it is worth learning. `hit` is `pos`."""
    iou: np.ndarray
    pos: np.ndarray
    near: np.ndarray
    neg: np.ndarray
    rot_ok: np.ndarray
    loc: np.ndarray             # [K, 2]
    loc_mask: np.ndarray
    hit: np.ndarray
    soft: np.ndarray
    covers: np.ndarray


def soft_target(distance_tiles: np.ndarray, cfg: LabelConfig) -> np.ndarray:
    """The soft target of a near window at `distance_tiles` from the truth."""
    return cfg.near_alpha * np.exp(-np.square(distance_tiles) / (2.0 * cfg.near_sigma ** 2))


def label_windows(boxes: np.ndarray, rotation: np.ndarray, keys: Sequence[Key], where: Boxes, tile_px: float,
                  cfg: LabelConfig, tol_px: float = 0.0) -> Labels:
    """Labels of windows with level-0 `boxes` `[K, 4]`, `rotation` `[K]` and identities `keys`. `tile_px` is a
    tile's side in level-0 px (the unit of distance). `tol_px` is the radius within which a window of the
    answer's rotation stands for it (a list's suppression radius; 0: only the answer itself)."""
    centres: np.ndarray = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)
    offset: np.ndarray = np.array(where.centre)[None, :] - centres
    distance: np.ndarray = np.hypot(offset[:, 0], offset[:, 1]) / tile_px
    rot_ok: np.ndarray = rotation == where.rotation
    pos: np.ndarray = np.zeros(len(boxes), dtype=bool)
    for key, (ox, oy) in where.answers:
        is_it: np.ndarray = np.array([k == key for k in keys], dtype=bool)
        stands_in: np.ndarray = (rotation == key[4]) & (np.hypot(boxes[:, 0] - ox, boxes[:, 1] - oy) < tol_px)
        pos |= is_it | stands_in
    near: np.ndarray = rot_ok & ~pos & (distance <= cfg.near_tiles)
    neg: np.ndarray = ~(pos | near)
    iou: np.ndarray = iou_matrix(boxes, where.truth[None, :])[:, 0]
    return Labels(iou=iou, pos=pos, near=near, neg=neg, rot_ok=rot_ok, loc=offset / tile_px,
                  loc_mask=iou >= cfg.loc_min_iou, hit=pos.copy(),
                  soft=np.where(near, soft_target(distance, cfg), 0.0), covers=distance <= cfg.near_tiles)


def choose_subset(labels: Labels, n_cand: int, hard_share: float, rng: np.random.Generator,
                  n_loc: int = 16) -> np.ndarray:
    """Indices (sorted: first-phase order) of the candidates a training step uses: every positive, every near
    window (the nearest `n_cand // 4` of them), the hardest negatives -- the earliest in the list, which the
    first phase ranked highest -- and random ones from the rest, up to `n_cand`, plus at most `n_loc` negatives
    that have an offset to learn."""
    pos: np.ndarray = np.flatnonzero(labels.pos)
    near_all: np.ndarray = np.flatnonzero(labels.near)
    near: np.ndarray = near_all[np.argsort(-labels.soft[near_all], kind='stable')[:max(n_cand // 4, 1)]]
    neg: np.ndarray = np.flatnonzero(labels.neg)
    room: int = max(n_cand - len(pos) - len(near) - n_loc, 0)
    n_hard: int = min(int(round(room * hard_share)), len(neg))
    hard: np.ndarray = neg[:n_hard]
    rest: np.ndarray = neg[n_hard:]
    easy: np.ndarray = rng.choice(rest, size=min(room - n_hard, len(rest)), replace=False) if len(rest) else rest
    loc_pool: np.ndarray = np.flatnonzero(labels.neg & labels.loc_mask)
    loc: np.ndarray = rng.choice(loc_pool, size=min(n_loc, len(loc_pool)), replace=False) if len(loc_pool) else loc_pool
    return np.unique(np.concatenate([pos, near, hard, easy, loc]).astype(np.int64))
