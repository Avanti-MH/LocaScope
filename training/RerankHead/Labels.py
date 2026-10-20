'''What a candidate window IS, relative to where a synthetic FoV was placed:
the labels every loss and every recall figure of the second phase reads.

    pos      the window covers the truth (IoU with the truth box >= pos_iou) AND is
             at the rotation that sets the photo upright
    neg      the window is disjoint from the photo -- or it covers the truth at the
             wrong rotation, which stage 3 needs right
    ignore   everything else: it overlaps the photo but is not the truth. It is NOT
             a negative -- a window one tile over holds most of the FoV -- so no
             ranking loss sees it. It still knows where the truth is relative to it
             (`loc`), and `loc` learns from it.

Three boxes decide, all in level-0 px and all areas, so nothing here knows how many
tiles a FoV has: the TRUTH box (the query's tile grid, turned upright, centred on
where it was taken), the FoV box (a square that contains the photo whatever its
rotation or scale) and each candidate's box.
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from stage2_retrieval.Context import intersection, iou_matrix


@dataclass(frozen=True)
class LabelConfig:
    pos_iou: float = 0.5        # IoU with the truth box for a positive
    hit_iou: float = 0.5        # ... for a hit when recall is counted
    loc_min_iou: float = 0.1    # ... for a window that has an offset to learn
    fov_margin: float = 1.2     # the FoV box is the photo's circumscribed square times this


@dataclass
class Boxes:
    """Where a FoV was placed, as boxes `(x0, y0, x1, y1)` in level-0 px."""
    truth: np.ndarray           # [4]
    fov: np.ndarray             # [4]
    rotation: int               # the stage-2 rotation that sets the photo upright
    centre: Tuple[float, float]


def truth_boxes(centre: Tuple[float, float], rotation: int, rows: int, cols: int, tile: int, ds: float,
                fov_centre: Tuple[float, float], fov_w0: float, fov_h0: float,
                cfg: LabelConfig) -> Boxes:
    """The truth box (the query's `rows x cols` tile grid at `rotation`, centred on
    `centre`) and the FoV box (the square around the photo's centre whose side is
    its diagonal, times `fov_margin`)."""
    cx, cy = centre
    hw, hh = cols * tile * ds / 2.0, rows * tile * ds / 2.0
    side: float = float(np.hypot(fov_w0, fov_h0)) * cfg.fov_margin / 2.0
    fx, fy = fov_centre
    return Boxes(truth=np.array([cx - hw, cy - hh, cx + hw, cy + hh]),
                 fov=np.array([fx - side, fy - side, fx + side, fy + side]),
                 rotation=int(rotation), centre=(float(cx), float(cy)))


@dataclass
class Labels:
    """Per candidate, `[K]`. `loc` is the truth's centre minus the window's, in
    tiles, and `loc_mask` where it is worth learning."""
    iou: np.ndarray
    pos: np.ndarray
    neg: np.ndarray
    ignore: np.ndarray
    rot_ok: np.ndarray
    loc: np.ndarray             # [K, 2]
    loc_mask: np.ndarray
    hit: np.ndarray             # IoU >= hit_iou and rotation right


def label_windows(boxes: np.ndarray, rotation: np.ndarray, where: Boxes, tile_px: float,
                  cfg: LabelConfig) -> Labels:
    """Labels of windows with level-0 `boxes` `[K, 4]` and `rotation` `[K]`.
    `tile_px` is a tile's side in level-0 px (the unit of `loc`)."""
    iou: np.ndarray = iou_matrix(boxes, where.truth[None, :])[:, 0]
    rot_ok: np.ndarray = rotation == where.rotation
    disjoint: np.ndarray = intersection(boxes, where.fov[None, :])[:, 0] <= 0.0
    covers: np.ndarray = iou >= cfg.pos_iou
    pos: np.ndarray = covers & rot_ok
    neg: np.ndarray = disjoint | (covers & ~rot_ok)
    neg &= ~pos
    ignore: np.ndarray = ~(pos | neg)
    centres: np.ndarray = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)
    loc: np.ndarray = (np.array(where.centre)[None, :] - centres) / tile_px
    return Labels(iou=iou, pos=pos, neg=neg, ignore=ignore, rot_ok=rot_ok, loc=loc,
                  loc_mask=iou >= cfg.loc_min_iou, hit=(iou >= cfg.hit_iou) & rot_ok)


def choose_subset(labels: Labels, n_cand: int, hard_share: float, rng: np.random.Generator,
                  n_loc: int = 16) -> np.ndarray:
    """Indices (sorted: first-phase order) of the candidates a training step uses:
    every positive, the hardest negatives -- the earliest in the list, which the
    first phase ranked highest -- and random ones from the rest, up to `n_cand`,
    plus at most `n_loc` ignored windows that have an offset to learn."""
    pos: np.ndarray = np.flatnonzero(labels.pos)
    neg: np.ndarray = np.flatnonzero(labels.neg)
    room: int = max(n_cand - len(pos) - n_loc, 0)
    n_hard: int = min(int(round(room * hard_share)), len(neg))
    hard: np.ndarray = neg[:n_hard]
    rest: np.ndarray = neg[n_hard:]
    easy: np.ndarray = rng.choice(rest, size=min(room - n_hard, len(rest)), replace=False) if len(rest) else rest
    loc_pool: np.ndarray = np.flatnonzero(labels.ignore & labels.loc_mask)
    loc: np.ndarray = rng.choice(loc_pool, size=min(n_loc, len(loc_pool)), replace=False) if len(loc_pool) else loc_pool
    return np.unique(np.concatenate([pos, hard, easy, loc]).astype(np.int64))
