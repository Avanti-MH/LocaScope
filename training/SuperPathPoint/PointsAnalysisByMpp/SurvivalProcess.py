"""One stack -> the rows of a survival table. spec.md 3.2.

    rows = run(stack, net, rungs=rungs, stack_kind='F', meta=meta)

THREE STEPS, AND EACH ONE HAS A WAY OF BEING SILENTLY WRONG
=============================================================
    1. detect      per rung, NMS -> border -> a PERMISSIVE threshold, NO CAP
    2. map         tile pixels -> level 0, using that rung's own scale
    3. match       a point at one rung is the same point as one at another
                   when the level-0 distance is within tau

`max_keypoints` IS OFF AND THAT IS LOAD-BEARING (spec.md 3.2). With a cap, a
point that drops out of the top N looks exactly like one killed by its
neighbourhood, and no column can tell them apart -- the cap is global
competition that we imposed. With it off, competition is only NMS-local, radius
4 output px = `4 * ds` level-0 px, and it is recorded in `suppressed_by`.

The threshold is PERMISSIVE for the same reason `alive` is not stored: every
later question re-cuts at some higher value on the arrays this wrote, and a
question below the stored cut cannot be answered without re-running the model
over every stack.

THE BIRTH RUNG IS NOT DECIDED HERE
====================================
This module writes `score` and `dist` and stops. Which rung a point was born at
depends on the threshold, and the threshold is a reader's choice -- so `born` is
computed in `Attribution.born_rung_of` from a freshly cut `alive`, and this
file's only job is to make that possible.

WHY THE ANCHOR RUNG IS THE FINEST
===================================
A point's level-0 position is taken at the finest rung it appears in, because
that is where the position is most precisely known: one pixel there is one
level-0 pixel, against `ds` at rung d. Anchoring at a coarse rung would give
every point a position with `ds` px of slop and then measure whether other rungs
agree with it to within tau -- which would be measuring the anchor's own error.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..', '..', 'utilities'),
           os.path.join(_HERE, '..')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.KeypointLabelStore import points_from_prob        # noqa: E402

# The sentinel this file WRITES and `Attribution` READS, imported from the
# reader rather than declared here. `Attribution` is pure -- numpy and nothing
# else -- so it cannot depend on this module, which imports torch; the
# dependency therefore runs the other way. One definition, two users.
from PointsAnalysisByMpp.Attribution import NONE             # noqa: E402


def detect(prob: np.ndarray, cfg, *, score_threshold: float
           ) -> Tuple[np.ndarray, np.ndarray]:
    """`(xy, score)` -- one rung's NMS survivors above the permissive cut.

    Uncapped. This produces the ANCHOR SET only: which locations the detector
    called points at some rung. What each rung then says about each anchor is
    `probe`'s job, and the two are separate because a point's absence from this
    list is not a measurement -- it is the absence of one.
    """
    xy, score, _ = points_from_prob(
        prob, None, score_threshold=float(score_threshold),
        nms_radius=cfg.nms_radius, border=cfg.border, max_points=None)
    return xy, score.astype(np.float32)


def rival_at(prob: np.ndarray, x: float, y: float, *, nms_radius: int
             ) -> float:
    """The strongest response within `nms_radius` of (x, y), excluding its peak.

    `rival > score` means the location was outranked locally, which -- with
    `max_keypoints` off -- is the only form competition takes. It reads the RAW
    map rather than the detection list because a suppressed location is by
    definition absent from that list, and "was it suppressed" is exactly the
    question (`Attribution.outranked`).

    THIS IS ALL THAT IS LEFT OF THE OLD `probe`. That function also returned a
    score and an offset, taken as the argmax over a window whose radius was a
    parameter -- and that parameter was wrong twice: bound to tau it made the
    store re-cuttable downward only, and opened up it made the offset a
    function of the window (a wider window finds a stronger peak FURTHER away,
    so the match rate collapsed). Distance to a detection is not a windowed
    question and is no longer asked as one; see `nearest_detection`.
    """
    height, width = prob.shape[-2:]
    r = int(nms_radius)
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = max(0, cx - r), min(width, cx + r + 1)
    y0, y1 = max(0, cy - r), min(height, cy + r + 1)
    if x1 <= x0 or y1 <= y0:
        return NONE
    window = np.array(prob[y0:y1, x0:x1], np.float32, copy=True)
    flat = int(window.argmax())
    window[divmod(flat, window.shape[1])] = -1.0
    best = float(window.max()) if window.size > 1 else NONE
    return best if best > 0.0 else NONE


def nearest_detection(points: np.ndarray, score: np.ndarray,
                      xy0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """`(distance, score)` of the nearest detected point to each query, level 0.

    `NONE` where the rung detected nothing at all.

    NO WINDOW AND THEREFORE NO CEILING. The distance to the nearest detection
    is a property of the point set, so every tau is answerable from one stored
    number and the table is re-cuttable in both directions -- which the windowed
    probe it replaced was not.

    THE NEAREST IS THE ONLY ONE, in the regime this is read at. NMS leaves no
    two survivors within `nms_radius` output pixels, which is `nms_radius * ds`
    level-0 px; tau is `alpha * ds`. So for `alpha < nms_radius` -- 4 here, and
    the offsets say the useful alphas are near 1 -- there is at most one
    detection inside tau, and "the nearest" and "the nearest above any
    threshold" are the same point. Above that alpha the stored score is the
    nearest one's and a threshold sweep is approximate; the sweep that matters
    runs well below it.
    """
    if not len(points) or not len(xy0):
        return (np.full(len(xy0), NONE, np.float64),
                np.zeros(len(xy0), np.float64))
    delta = xy0[:, None, :] - points[None, :, :]
    distance = np.sqrt((delta ** 2).sum(axis=2))
    nearest = distance.argmin(axis=1)
    return (distance[np.arange(len(xy0)), nearest],
            np.asarray(score, np.float64)[nearest])


def to_level0(xy: np.ndarray, *, origin: Tuple[float, float], ds: float
              ) -> np.ndarray:
    """Tile pixels -> level 0. `X = x0 + u * scale` (spec.md 3.2 配對).

    `ds` here is the SCALE -- level-0 px per output pixel -- which equals the
    rung's ds on the 'F' axis and 1.0 on the 'R' axis. The caller supplies it;
    see `MppStack.rung_scale`.

    `origin` is the TILE's top-left in level-0 coordinates, which differs per
    rung because the footprints differ -- the shared quantity is the CENTRE.
    Passing the centre and deriving the corner here would put the same rounding
    in every caller; passing the corner keeps `PreTileRecord.x, y` the single
    source, which is what they already are.
    """
    if not len(xy):
        return np.zeros((0, 2), np.float64)
    return np.asarray(origin, np.float64) + np.asarray(xy, np.float64) * float(ds)


def to_tile(xy0: np.ndarray, *, origin: Tuple[float, float], ds: float
            ) -> np.ndarray:
    """Level 0 -> tile pixels. The inverse of `to_level0`, spelled once."""
    return (np.asarray(xy0, np.float64)
            - np.asarray(origin, np.float64)) / float(ds)


def anchors_of(per_rung: Dict[float, np.ndarray], order: Sequence[float],
               merge_radius_l0: float) -> np.ndarray:
    """The union of every rung's detections, deduplicated. `[N, 2]` level 0.

    Finest rung first, so a point's level-0 position is the one measured where
    it is most precisely knowable: one pixel there is one level-0 pixel against
    `ds` at rung d. Anchoring on a coarse rung would give every point `ds` px of
    slop and then measure whether the other rungs agree with it to within tau --
    measuring the anchor's own error.

    A coarse-rung detection that matches nothing finer BECOMES an anchor. That
    is not a detail: those are the late-born points, and they are the whole of
    新生歸因. A set anchored once at ds 1 would not contain them.

    `merge_radius_l0` IS NOT `tau`, AND IT USED TO BE. Merging at tau makes the
    anchor set a function of tau, which makes the whole store un-recuttable:
    every threshold sweep and every tau the calibration run suggests would need
    the GPU again -- against a store whose entire design (no `alive` column, no
    `born_rung`) exists so that re-cutting is a comparison on arrays already in
    memory. So this radius answers a different question: "is this the same
    LOCATION", which is about pixel identity and not about how far two rungs
    may disagree. `nms_radius` level-0 px is the natural value -- two
    detections closer than that could not both have survived NMS at the finest
    rung.

    What tau merges is then done at READ time (`Report.merge_anchors`), where
    it costs milliseconds and can be redone at another tau.
    """
    radius = float(merge_radius_l0)
    out: List[np.ndarray] = []
    for ds in order:
        xy0 = per_rung[ds]
        for k in range(len(xy0)):
            if out:
                near = np.linalg.norm(np.asarray(out) - xy0[k], axis=1)
                if float(near.min()) <= radius:
                    continue
            out.append(xy0[k])
    return (np.asarray(out, np.float64) if out else np.zeros((0, 2), np.float64))


def run(stack: Dict[float, np.ndarray], net, *, rungs: Sequence[float],
        origins: Dict[float, Tuple[float, float]],
        scales: Dict[float, float], decoy_offset: Sequence[float],
        score_threshold: float = 0.001) -> Dict[str, np.ndarray]:
    """One stack -> the `[N, L]` columns of a survival table.

    `stack` is `{ds: image}`, `origins` is `{ds: (x, y)}` level-0 top-left,
    `scales` is `{ds: level-0 px per output pixel}` and `decoy_offset` is how
    far the decoy probe sits from each anchor, in level-0 px.

    THERE IS NO SEARCH RADIUS ANY MORE AND THAT IS THE POINT. `dist` is the
    distance to the nearest DETECTION, which is a property of the point set --
    so every tau is answerable and the table re-cuts in both directions. The
    windowed probe it replaced took the argmax over a radius, and that radius
    was wrong in both directions on 2026-09-01: set to tau it bounded `dist` by
    tau (the curve went flat at the window and read as saturation), opened up
    it made `dist` the distance to whatever the widest peak in the window was
    (the match rate at ds 1 fell from 0.090 to 0.005). A distance to a
    detection has no window in it.

    `scales` IS A PARAMETER AND NOT `ds`, which is the whole reason it is here.
    On the 'F' axis one output pixel spans `ds` level-0 px; on the 'R' axis it
    spans 1.0 at every rung, because the footprint is the ds 1 tile's and the
    degradation happens inside a frame that never moves. Deriving it from `ds`
    here -- which this function did until it was caught -- scatters every
    coarse-rung 'R' point `ds` times too far from the centre, and the table
    fills, and the survival numbers become a picture of the bug.
    `MppStack.rung_scale` is where the two answers live.

    TWO PASSES, AND THEY ANSWER DIFFERENT QUESTIONS. The first detects, to find
    out WHICH locations are worth asking about. The second probes every rung at
    every one of those locations, including the rungs that did not detect it --
    because "not detected" is not a measurement, and the table needs one at
    every cell or the threshold sweep has nothing to sweep.
    """
    import torch                                             # noqa: PLC0415

    order = sorted(float(r) for r in rungs)
    length = len(order)
    cfg = net.cfg
    device = net.device

    maps: Dict[float, np.ndarray] = {}
    detections: Dict[float, np.ndarray] = {}
    detected_score: Dict[float, np.ndarray] = {}
    with torch.no_grad():
        for ds in order:
            tensor = _to_tensor(stack[ds], cfg.backbone.in_channels).to(device)
            prob = net(tensor[None]).prob_map[0].float().cpu().numpy()
            maps[ds] = prob
            xy, sc = detect(prob, cfg, score_threshold=score_threshold)
            detections[ds] = to_level0(xy, origin=origins[ds],
                                       ds=float(scales[ds]))
            detected_score[ds] = sc

    # The merge radius is `nms_radius` LEVEL-0 px and not `tau`: see
    # `anchors_of`. tau is applied when the table is read, which is what makes
    # a threshold or tau sweep free.
    anchors = anchors_of(detections, order, float(cfg.nms_radius))
    n = len(anchors)
    score = np.zeros((n, length), np.float32)
    dist = np.full((n, length), NONE, np.float32)
    rival = np.full((n, length), NONE, np.float32)
    rival_dist = np.full((n, length), NONE, np.float32)
    decoy_score = np.zeros((n, length), np.float32)
    decoy_dist = np.full((n, length), NONE, np.float32)
    if not n:
        return {'x0': np.zeros(0, np.float32), 'y0': np.zeros(0, np.float32),
                'score': score, 'dist': dist,
                'suppressed_by_score': rival, 'suppressed_by_dist': rival_dist,
                'decoy_score': decoy_score, 'decoy_dist': decoy_dist}

    for j, ds in enumerate(order):
        scale = float(scales[ds])
        found, found_score = detections[ds], detected_score[ds]

        near, near_score = nearest_detection(found, found_score, anchors)
        dist[:, j] = near
        score[:, j] = near_score

        # The decoy: the same question asked at a place the point is NOT.
        # Diagonal so it is not a shift along a scan line, where a tissue edge
        # would make it systematically easier or harder than a real
        # neighbourhood.
        step = float(decoy_offset[j]) / np.sqrt(2.0)
        shifted = anchors + step
        far, far_score = nearest_detection(found, found_score, shifted)
        decoy_dist[:, j] = far
        decoy_score[:, j] = far_score

        # `rival` still reads the raw map, because a suppressed location is by
        # definition absent from the detection list and "was it suppressed" is
        # the question.
        tile_xy = to_tile(anchors, origin=origins[ds], ds=scale)
        for i in range(n):
            rival[i, j] = rival_at(maps[ds], tile_xy[i, 0], tile_xy[i, 1],
                                   nms_radius=cfg.nms_radius)
            rival_dist[i, j] = float(cfg.nms_radius) * scale

    return {'x0': anchors[:, 0].astype(np.float32),
            'y0': anchors[:, 1].astype(np.float32),
            'score': score, 'dist': dist,
            'suppressed_by_score': rival, 'suppressed_by_dist': rival_dist,
            'decoy_score': decoy_score, 'decoy_dist': decoy_dist}


def _to_tensor(image: np.ndarray, channels: int):
    """`[H, W, 3] uint8` -> `[C, H, W] float32` in 0..1, the loader's rule.

    Grayscale by the same weights `Datasets._to_tensor` uses, because the
    student was trained on that conversion and a second definition of "gray"
    would show up as a keypoint that moved.
    """
    import torch                                             # noqa: PLC0415

    array = np.asarray(image, np.float32) / 255.0
    if int(channels) == 1:
        array = (0.299 * array[..., 0] + 0.587 * array[..., 1]
                 + 0.114 * array[..., 2])[..., None]
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
