"""One stack -> real measurements only. spec.md 3.2.

    xy, score = detect(prob, cfg, score_threshold=0.001)
    anchors = anchors_of({ds: xy_per_rung}, order, merge_radius_l0=0)
    dist, score, rival, rival_dist = probe_real(anchors, per_rung_detections,
                                                order, maps, origins, scales,
                                                nms_radius=cfg.nms_radius)

NO DECOY HERE. A decoy exists only to calibrate alpha (`AlphaCalibration.py`);
once alpha is fixed, classifying a point's six-pattern membership only ever
needs `probe_real`'s output. Keeping decoy machinery out of this file means
alpha's calibration logic can be rewritten without touching a line here.

THE BIRTH RUNG IS NOT DECIDED HERE. `alive`/`born` are a reader's threshold
and tau applied to `dist`/`score` (`Patterns.alive_from`,
`Attribution.born_rung_of`) -- this file stops at the raw measurement.

WHY THE ANCHOR RUNG IS THE FINEST. A point's level-0 position is taken at the
finest rung it appears in: one pixel there is one level-0 pixel, against `ds`
at rung d. Anchoring at a coarse rung would give every point `ds` px of slop
and then measure whether other rungs agree with it to within tau -- which
would be measuring the anchor's own error.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..', '..', 'utilities'),
           os.path.join(_HERE, '..')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.KeypointLabelStore import points_from_prob        # noqa: E402
from SurvivalAnalysis.Attribution import NONE                    # noqa: E402


def detect(prob: np.ndarray, cfg, *, score_threshold: float
           ) -> Tuple[np.ndarray, np.ndarray]:
    """`(xy, score)` -- one rung's NMS survivors above a permissive cut.
    Tile pixels, not level 0. Uncapped (`max_points=None`): a point that
    drops out of a top-N cap looks identical to one killed by its
    neighbourhood, and no column could tell the two apart.
    """
    xy, score, _ = points_from_prob(
        prob, None, score_threshold=float(score_threshold),
        nms_radius=cfg.nms_radius, border=cfg.border, max_points=None)
    return xy, score.astype(np.float32)


def rival_at(prob: np.ndarray, x: float, y: float, *, nms_radius: int
             ) -> float:
    """The strongest response within `nms_radius` of `(x, y)`, excluding its
    own peak. Reads the raw map rather than the detection list, because a
    suppressed location is by definition absent from that list.
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
    """`(distance, score)` of the nearest detected point to each query,
    level 0. `NONE` where the rung detected nothing at all.

    No window, no ceiling: the distance to the nearest detection is a
    property of the point set, so any tau can be applied to it afterward --
    this is why it cannot just reuse `_merge_within_radius`'s fixed-radius
    grid: there, a candidate farther than `radius` is irrelevant and never
    has to be found; here, the TRUE global nearest point has to be found
    even when it is very far away (a large but real number, not a sentinel
    -- `Attribution.NONE` means "this rung found nothing AT ALL", a
    different thing from "found something, far").

    GRID-HASH EXPANDING-RING SEARCH, not the O(N*M) all-pairs matrix this
    used to be (`delta = xy0[:, None, :] - points[None, :, :]`, an
    `[N, M, 2]` array): fine at a few thousand points a side, an OOM crash
    on a real C-tree's ds=1 rung, which routinely has tens of thousands of
    both anchors and raw detections (2026-09-12: ~70,000 either side would
    ask for ~78 GB). `points` is bucketed once into a grid sized for a
    handful of points per cell; each query then searches OUTWARD ring by
    ring (its own cell, the 8 around it, the 16 around those, ...) and
    stops the moment the NEXT ring cannot possibly hold anything closer
    than the best candidate already found -- a cell at Chebyshev ring `r`
    is at least `(r-1)*cell` from the query, so once `best_d <= r*cell`
    after finishing ring `r`, ring `r+1` and beyond are provably no closer.
    This is EXACT, not an approximation: the DISTANCE always matches the
    O(N*M) version (`test_survival_process.py`'s `t_nearest_detection_
    matches_brute_force_on_random_points` checks it) -- the one thing that
    can differ is WHICH point wins an exact tie (two points at the
    identical distance), since this visits candidates in grid/ring order,
    not array order the way `argmin` does. Real detector scores are
    continuous floats; an exact tie has probability zero on real data, so
    this is not a behaviour change worth engineering around.
    """
    n = len(xy0)
    if not len(points) or not n:
        return np.full(n, NONE, np.float64), np.zeros(n, np.float64)
    nearest = _nearest_index_grid(points, xy0)
    delta = xy0 - points[nearest]
    distance = np.sqrt((delta ** 2).sum(axis=1))
    return distance, np.asarray(score, np.float64)[nearest]


def _nearest_index_grid(points: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """`[Q]` -- for each query, the INDEX into `points` of its nearest
    neighbour (exact). `points` must be non-empty; `nearest_detection` is
    the only caller and already guards that.

    Cell size aims for roughly one point per cell on average
    (`span / sqrt(len(points))`) -- small enough that a typical query
    resolves within the first ring or two, large enough that a ring is not
    almost-always empty (which would just mean more rings, not a wrong
    answer, but a slower one).

    THE RING SEARCH'S OWN STOPPING RULE (`best_d <= ring * cell`) IS WHAT
    MAKES THIS EXACT, AND IT NEVER NEEDS A QUERY-INDEPENDENT CAP: a decoy
    query can sit far outside `points`' own bounding box (`decoy_shift`
    moves anchors by real, unbounded-in-principle amounts) and still needs
    its TRUE nearest point, however many rings out that is -- a cap sized
    only from `points`' own span would cut such a search off too early and
    return a wrong (or missing) answer for exactly the queries furthest
    from the cloud. `max_ring` below is sized per query instead, from that
    query's own distance to the cloud's centre plus the cloud's own
    radius (triangle inequality: no point in `points` can be farther from
    the query than that), so it is generous enough to always contain the
    true answer -- a defensive bound against a runaway loop, not a limit
    that can ever change the result.
    """
    n_points = len(points)
    span = float(max(points[:, 0].max() - points[:, 0].min(),
                     points[:, 1].max() - points[:, 1].min(), 1.0))
    cell = max(span / np.sqrt(n_points), 1.0)
    cloud_cx = 0.5 * float(points[:, 0].min() + points[:, 0].max())
    cloud_cy = 0.5 * float(points[:, 1].min() + points[:, 1].max())
    cloud_radius = float(np.hypot(points[:, 0].max() - cloud_cx,
                                 points[:, 1].max() - cloud_cy)) + cell

    grid: Dict[Tuple[int, int], List[int]] = {}
    cell_x = np.floor(points[:, 0] / cell).astype(np.int64)
    cell_y = np.floor(points[:, 1] / cell).astype(np.int64)
    for i in range(n_points):
        grid.setdefault((int(cell_x[i]), int(cell_y[i])), []).append(i)

    n_queries = len(queries)
    nearest = np.empty(n_queries, dtype=np.int64)
    for qi in range(n_queries):
        qx, qy = queries[qi]
        qcx = int(np.floor(qx / cell))
        qcy = int(np.floor(qy / cell))
        query_to_cloud = float(np.hypot(qx - cloud_cx, qy - cloud_cy))
        max_ring = int(np.ceil((query_to_cloud + cloud_radius) / cell)) + 2
        best_d = np.inf
        best_j = -1
        ring = 0
        while True:
            for dx in range(-ring, ring + 1):
                for dy in range(-ring, ring + 1):
                    if max(abs(dx), abs(dy)) != ring:
                        continue    # only this ring's NEW cells -- inner
                                    # ones were already searched last pass
                    cand = grid.get((qcx + dx, qcy + dy))
                    if not cand:
                        continue
                    for j in cand:
                        d = float(np.hypot(points[j, 0] - qx, points[j, 1] - qy))
                        if d < best_d:
                            best_d = d
                            best_j = j
            if (best_j != -1 and best_d <= ring * cell) or ring >= max_ring:
                break
            ring += 1
        nearest[qi] = best_j
    return nearest


def to_level0(xy: np.ndarray, *, origin: Tuple[float, float], ds: float
              ) -> np.ndarray:
    """Tile pixels -> level 0. `X = x0 + u * scale`.

    `ds` here is the SCALE (level-0 px per output pixel) -- equal to the
    rung's `ds` on the 'F'/'C' axes, 1.0 on 'R' (`ChainStack.rung_scale`).
    `origin` is the tile's top-left in level-0 coordinates.
    """
    if not len(xy):
        return np.zeros((0, 2), np.float64)
    return np.asarray(origin, np.float64) + np.asarray(xy, np.float64) * float(ds)


def to_tile(xy0: np.ndarray, *, origin: Tuple[float, float], ds: float
            ) -> np.ndarray:
    """Level 0 -> tile pixels. The inverse of `to_level0`."""
    return (np.asarray(xy0, np.float64)
            - np.asarray(origin, np.float64)) / float(ds)


def _merge_within_radius(points: np.ndarray, radius: float,
                         priority: Optional[np.ndarray] = None,
                         rung_id: Optional[np.ndarray] = None,
                         rung_scale: Optional[np.ndarray] = None
                         ) -> np.ndarray:
    """Indices to KEEP after merging points within `radius` of each other --
    the 網格雜湊合併法 (grid hash merge).

    Visits points in `priority` order, highest first (input order if
    `priority` is None), keeping a point only if no already-kept point is
    within `radius` of it -- the survivor of a cluster is whichever was
    visited first. Shared by `anchors_of` (F and R), `anchors_of_generations`
    (C, both its within-generation and cross-generation merges) and
    `AlphaCalibration.merge_anchors` (the `--merge-radius-2nd` read-time
    re-merge) so all four are one definition.

    A candidate compares against the already-kept points in its own
    `radius`-sized grid cell and the surrounding 3x3, not every already-kept
    point: any point within `radius` of a cell of side `radius` must fall in
    that 3x3 neighbourhood, so nothing is missed. That turns the walk from
    O(n*k) (k = points kept so far) into roughly O(n) once points aren't
    pathologically clustered inside one cell -- the difference that matters
    at a C generation's finest rung, where hundreds of tiles' raw detections
    concatenate into tens of thousands of points before this runs.

    Replaced two prior versions, both O(n*k): a Python list rebuilt into a
    fresh array every iteration, then a pre-allocated index buffer sliced in
    place (removed the rebuild's extra O(n^2), not the O(n*k) itself). Both
    retired to `cli/demo_survival_analysis.py` as reference implementations -- that
    file is where this one was checked against them before being adopted
    here: identical kept-index sets on synthetic data, on a real C-tree's
    raw ds=1 detections (34503 points, 131x faster than the array-buffer
    version), and deployed end-to-end in `cli/survival_alpha_analysis.py`'s
    own C-axis flow (0% anchor mismatch and bit-identical aggregated curves
    across 2 real ChainStacks, 1.5x faster overall).

    `rung_id`/`rung_scale`, if given (both together, `[N]` each) -- turns on
    CROSS-RUNG mode (spec.md "同一個點的定義"). Two
    DIFFERENT questions, kept as two arrays on purpose:

        rung_id     which rung a point came from, for the "is this even a
                    different rung" test. Just a label -- equal labels mean
                    same rung, unequal means different, nothing about the
                    label's VALUE is used.
        rung_scale  that rung's actual level-0-per-output-pixel scale (same
                    units as `to_level0`'s `ds`), for how far apart two
                    DIFFERENT rungs are still allowed to be. On F and C
                    these are the same number (the rung label IS the
                    scale); on R they are not -- R's own footprint never
                    grows (`ChainStack.rung_scale('R', ds) == 1.0` at every
                    rung, `to_level0`'s `ds` is 1.0 there regardless of the
                    label), so `rung_id` still separates R's own rungs
                    while `rung_scale` correctly contributes nothing to the
                    quantisation term.

    The rule:

        same rung_id        never the same point, at ANY distance -- that
                            rung's own NMS already told two of its own
                            survivors apart (`nms_max_pool` separates by
                            Chebyshev radius, which lower-bounds their
                            Euclidean distance too, so two same-rung
                            survivors are never closer than that rung's own
                            nms_radius)
        different rung_id    the same point only within
                            `radius + max(scale_i, scale_j) // 2` --
                            `radius` here is the FLOOR (0 for the real
                            build), `max(scale_i, scale_j) // 2` is the
                            coarser point's own quantisation half-width from
                            `cv2.INTER_AREA` block-average downsampling
                            (`ChainStack.py:153`), 0 on R. Integer floor
                            division because level-0 coordinates are integer
                            pixel positions, not because `radius`'s own
                            units need it.

    `rung_id=None` (default) is the ORIGINAL single-radius, no-exclusion
    behaviour: what `merge_anchors`'s 2nd-pass re-merge needs (it merges the
    FINAL anchor list, which carries no rung of its own to compare), and
    what `anchors_of_generations`'s within-generation, cross-TILE merge
    needs (every point there already shares one ds, so cross-rung mode
    would exclude every pair and merge nothing -- that merge's radius is
    about tile-crop boundary/receptive-field effects, not downsampling
    quantisation, and is a different constant kept out of this mode).
    """
    n = len(points)
    if not n:
        return np.arange(0, dtype=np.int64)
    if rung_id is None and float(radius) <= 0.0:
        return np.arange(n, dtype=np.int64)
    order = (np.argsort(-np.asarray(priority)) if priority is not None
             else np.arange(n))
    if rung_id is None:
        cell = float(radius)
    else:
        rung_id = np.asarray(rung_id)
        rung_scale = np.asarray(rung_scale, dtype=np.float64)
        cell = max(float(radius) + float(np.max(rung_scale)) // 2.0, 1.0)
    grid: Dict[Tuple[int, int], List[int]] = {}
    kept: List[int] = []
    for i in order:
        i = int(i)
        cx = int(np.floor(points[i, 0] / cell))
        cy = int(np.floor(points[i, 1] / cell))
        near = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), ()):
                    if rung_id is not None:
                        if rung_id[j] == rung_id[i]:
                            continue
                        r = (float(radius)
                            + max(rung_scale[i], rung_scale[j]) // 2.0)
                    else:
                        r = float(radius)
                    if np.linalg.norm(points[j] - points[i]) <= r:
                        near = True
                        break
                if near:
                    break
            if near:
                break
        if near:
            continue
        kept.append(i)
        grid.setdefault((cx, cy), []).append(i)
    return np.sort(np.asarray(kept, dtype=np.int64))


def anchors_of(per_rung: Dict[float, np.ndarray], order: Sequence[float],
              merge_radius_l0: float, *,
              rung_scale: Optional[Callable[[float], float]] = None
              ) -> np.ndarray:
    """The union of every rung's detections, deduplicated. `[N, 2]` level 0.

    `order` finest first: a point kept from a finer rung's list is visited
    before the same physical point's coarser-rung duplicate, so its
    level-0 position is the one measured most precisely.

    A coarse-rung detection that matches nothing finer BECOMES an anchor --
    those are the late-born points (新生歸因). `merge_radius_l0` is the
    CROSS-RUNG floor added to each pair's own quantisation half-width
    (`_merge_within_radius`'s `rung_id`/`rung_scale` mode, spec.md
    "同一個點的定義") -- 0 on F/C, where the ONLY systematic disagreement
    between two rungs' reports of the same feature is the coarser one's own
    downsampling quantisation, now carried by `rung_scale` instead: not tau,
    so the anchor set is not a function of tau and every later tau sweep
    does not need to rebuild it; not a single fixed radius either, because
    that ignored the quantisation slop and could both wrongly fuse two
    distinct same-rung points at the finest rung and wrongly miss a genuine
    coarse-rung duplicate (same section). NOT 0 on R (`_one_r_tile` passes
    `net.cfg.nms_radius`): R's quantisation term is always 0 (below), so if
    the floor were also 0 there would be nothing left to absorb R's actual
    disagreement source -- `degrade_resolution`'s own blur shifting a peak,
    not quantisation -- and its cross-rung merge would need near-exact
    pixel coincidence to ever fire.

    `rung_scale`, if given, maps an `order` entry to its ACTUAL level-0
    scale for that quantisation term -- defaults to the identity (`order`'s
    own value IS the scale), true on F, but NOT on R: `order`'s entries
    there are `degrade_resolution` labels (how much degraded relative to
    whichever real WSI pyramid level the base image happened to come from),
    not scale factors (`ChainStack.rung_scale('R', ds) == 1.0` at every
    rung -- R's own footprint never grows, so there is no downsampling
    quantisation to add regardless of which label two points came from).
    `_one_r_tile` passes `rung_scale=lambda ds: 1.0`; the labels still
    separate R's own rungs (`rung_id` in `_merge_within_radius` is
    unaffected), only the added
    slop collapses to 0.
    """
    groups = [(ds, per_rung[ds]) for ds in order if len(per_rung[ds])]
    all_pts = (np.concatenate([p for _, p in groups], axis=0) if groups
              else np.zeros((0, 2), np.float64))
    scale_of = rung_scale if rung_scale is not None else (lambda ds: ds)
    rung_id = (np.concatenate([np.full(len(p), ds, np.float64)
                              for ds, p in groups])
              if groups else np.zeros(0, np.float64))
    scale = (np.concatenate([np.full(len(p), scale_of(ds), np.float64)
                            for ds, p in groups])
            if groups else np.zeros(0, np.float64))
    keep = _merge_within_radius(all_pts, merge_radius_l0,
                                rung_id=rung_id, rung_scale=scale)
    return all_pts[keep]


def anchors_of_generations(
        per_rung_tiles: Dict[float, Tuple[List[np.ndarray], np.ndarray]],
        order: Sequence[float], tile_merge_radius: float,
        cross_rung_base: float = 0.0) -> np.ndarray:
    """The 'C' axis equivalent of `anchors_of`: one rung is many tiles, not
    one. `[N, 2]` level 0.

    `per_rung_tiles[ds] = (main_tiles, overlap_tile)` -- every main
    descendant's own level-0 detections at that rung, and the overlap
    tile's. Built in two passes, EACH WITH ITS OWN RADIUS
    (spec.md "同一個點的定義" -- the two are different
    physical questions, not one constant reused twice):

        within a generation   `tile_merge_radius` (`cfg.nms_radius` for the
        (same ds, cross-TILE) real build) -- the main tiles' points are
                              unioned and deduplicated
                              (`_merge_within_radius`, `ds=None`: every
                              point here already shares one ds, so
                              cross-rung mode would exclude every pair and
                              merge nothing) into that generation's
                              consensus set; the overlap tile's points then
                              join it. This radius answers "how far can two
                              INDEPENDENT crops' detections of the same
                              near-seam feature disagree" (a
                              receptive-field/boundary effect) -- main tiles
                              do not overlap and each excludes its own
                              `border` margin (`KeypointLabelStore.py`), so
                              this step rarely has a true duplicate to
                              catch; the overlap tile exists to cover
                              exactly that `border` dead zone, which is
                              where one actually occurs.
        across generations    `_merge_within_radius(..., rung_id=..., rung_
        (cross ds)            scale=...)`, same cross-rung mode `anchors_of`
                              uses: `cross_rung_base + max(ds_i, ds_j) // 2`
                              per pair, same-ds pairs never merged, finest rung
                              first.
    """
    per_generation: List[np.ndarray] = []
    for ds in order:
        main_tiles, overlap_pts = per_rung_tiles[ds]
        main_pts = ([m for m in main_tiles if len(m)])
        main_pts = (np.concatenate(main_pts, axis=0) if main_pts
                   else np.zeros((0, 2), np.float64))
        generation = main_pts[_merge_within_radius(main_pts, tile_merge_radius)]

        overlap_pts = np.asarray(overlap_pts, np.float64)
        if len(overlap_pts):
            combined = np.concatenate([generation, overlap_pts], axis=0)
            generation = combined[_merge_within_radius(combined,
                                                        tile_merge_radius)]
        per_generation.append(generation)

    all_pts = (np.concatenate(per_generation, axis=0) if per_generation
              else np.zeros((0, 2), np.float64))
    rung_ds = (np.concatenate([np.full(len(g), ds, np.float64)
                              for ds, g in zip(order, per_generation)])
              if per_generation else np.zeros(0, np.float64))
    keep = _merge_within_radius(all_pts, cross_rung_base,
                                rung_id=rung_ds, rung_scale=rung_ds)
    return all_pts[keep]


def probe_real(anchors: np.ndarray,
              per_rung_detections: Dict[float, Tuple[np.ndarray, np.ndarray]],
              order: Sequence[float], maps: Optional[Dict[float, np.ndarray]] = None, *,
              origins: Optional[Dict[float, Tuple[float, float]]] = None,
              scales: Optional[Dict[float, float]] = None,
              nms_radius: Optional[int] = None
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(dist, score, rival, rival_dist)`, each `[N, len(order)]`.

    `per_rung_detections[ds] = (points, score)` -- that rung's own detected
    points to probe against, level 0. `maps[ds]` is the rung's own
    probability map, tile pixels, needed only for `rival_at`.

    `maps=None` leaves `rival`/`rival_dist` at `NONE` throughout. 'C' has
    several candidate tiles per rung (`detect_all_generations` never builds
    a single `maps[ds]`), and resolving which one an anchor's coordinate
    falls inside needs the pyramid's own geometry -- deferred to whichever
    round actually reads `rival` (round 1, alpha calibration, never does).
    """
    n = len(anchors)
    length = len(order)
    dist = np.full((n, length), NONE, np.float32)
    score = np.zeros((n, length), np.float32)
    rival = np.full((n, length), NONE, np.float32)
    rival_dist = np.full((n, length), NONE, np.float32)
    if not n:
        return dist, score, rival, rival_dist

    for j, ds in enumerate(order):
        found, found_score = per_rung_detections[ds]
        near, near_score = nearest_detection(found, found_score, anchors)
        dist[:, j] = near
        score[:, j] = near_score

        if maps is not None:
            scale = float(scales[ds])
            tile_xy = to_tile(anchors, origin=origins[ds], ds=scale)
            for i in range(n):
                rival[i, j] = rival_at(maps[ds], tile_xy[i, 0], tile_xy[i, 1],
                                       nms_radius=nms_radius)
                rival_dist[i, j] = float(nms_radius) * scale
    return dist, score, rival, rival_dist


def detect_all_rungs(stack: Dict[float, np.ndarray], net, *,
                     rungs: Sequence[float],
                     origins: Dict[float, Tuple[float, float]],
                     scales: Dict[float, float], score_threshold: float = 0.001
                     ) -> Tuple[Dict[float, np.ndarray],
                                Dict[float, Tuple[np.ndarray, np.ndarray]],
                                Dict[float, np.ndarray]]:
    """Run the detector once per rung. `stack` is `{ds: image}` -- ONE tile
    per rung ('F'/'R' shape; 'C' calls `detect` per tile itself and builds
    the three returned dicts by hand instead of through this function).

    Returns `(detections, per_rung_detections, maps)`:
    `detections[ds]` is that rung's own `(xy, score)` in TILE pixels (kept
    for callers that build an anchor list from it, e.g. `anchors_of` wants
    LEVEL-0 points, so see below); `per_rung_detections[ds]` is the same
    points already mapped to level 0 -- `probe_real`'s input;
    `maps[ds]` is the raw probability map, `rival_at`'s input.
    """
    import torch                                             # noqa: PLC0415

    order = sorted(float(r) for r in rungs)
    cfg = net.cfg
    device = net.device

    import time                                              # noqa: PLC0415

    detections: Dict[float, np.ndarray] = {}
    per_rung_detections: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    maps: Dict[float, np.ndarray] = {}
    with torch.no_grad():
        for ds in order:
            t0 = time.perf_counter()
            tensor = _to_tensor(stack[ds], cfg.backbone.in_channels).to(device)
            prob = net(tensor[None]).prob_map[0].float().cpu().numpy()
            maps[ds] = prob
            xy, sc = detect(prob, cfg, score_threshold=score_threshold)
            xy0 = to_level0(xy, origin=origins[ds], ds=float(scales[ds]))
            detections[ds] = xy0
            per_rung_detections[ds] = (xy0, sc)
            print(f'      ds {ds:g}: {len(xy0)} detections '
                 f'({time.perf_counter() - t0:.2f}s)', flush=True)
    return detections, per_rung_detections, maps


def detect_all_generations(mother, mother_image: np.ndarray, groups_by_ds,
                           images_by_ds, net, *, score_threshold: float = 0.001,
                           keep_maps: bool = False
                           ) -> Union[
        Tuple[Dict[float, Tuple[List[np.ndarray], np.ndarray]],
             Dict[float, Tuple[np.ndarray, np.ndarray]]],
        Tuple[Dict[float, Tuple[List[np.ndarray], np.ndarray]],
             Dict[float, Tuple[np.ndarray, np.ndarray]],
             Dict[float, Tuple[List[np.ndarray], List[Tuple[float, float]],
                               List[np.ndarray], List[Tuple[float, float]]]]]]:
    """The 'C' axis's `detect_all_rungs`: one rung is every `TileGroup`'s
    main + overlap tiles, not one image. `groups_by_ds`/`images_by_ds` are
    `ChainStack.CStack.pyramid`'s/`read_tree`'s own shapes -- `mother`/
    `mother_image` are the coarsest generation, one tile, no overlap.

    Returns `(per_rung_tiles, per_rung_detections)`, or
    `(per_rung_tiles, per_rung_detections, per_rung_maps)` when
    `keep_maps=True`:
    `per_rung_tiles[ds] = (main_tiles, overlap)`, level-0 points --
    `anchors_of_generations`'s input. `per_rung_detections[ds]` is every
    tile's detections at that rung CONCATENATED, unmerged -- the probe
    target list only needs the closest point, which duplicates cannot
    change. `per_rung_maps[ds] = (main_maps, main_origins, overlap_maps,
    overlap_origins)` -- `AliveCandidates.assemble_generation_map`'s input
    (candidate 1); a full probability map per tile instead of a handful of
    peak coordinates, which is why this is opt-in (`keep_maps=False` by
    default) rather than always kept -- every other caller of this function
    pays nothing for a candidate it is not using.
    """
    import time                                              # noqa: PLC0415
    import torch                                             # noqa: PLC0415

    cfg = net.cfg
    device = net.device
    t_start = time.perf_counter()

    def _detect_one(info, image):
        tensor = _to_tensor(image, cfg.backbone.in_channels).to(device)
        with torch.no_grad():
            prob = net(tensor[None]).prob_map[0].float().cpu().numpy()
        xy, sc = detect(prob, cfg, score_threshold=score_threshold)
        xy0 = to_level0(xy, origin=(info.x, info.y), ds=float(info.ds))
        return xy0, sc, prob

    mother_xy0, mother_sc, mother_prob = _detect_one(mother, mother_image)
    per_rung_tiles: Dict[float, Tuple[List[np.ndarray], np.ndarray]] = {
        float(mother.ds): ([mother_xy0], np.zeros((0, 2), np.float64))}
    per_rung_detections: Dict[float, Tuple[np.ndarray, np.ndarray]] = {
        float(mother.ds): (mother_xy0, mother_sc)}
    per_rung_maps: Dict[float, Tuple[List[np.ndarray], List[Tuple[float, float]],
                                    List[np.ndarray], List[Tuple[float, float]]]] = {}
    if keep_maps:
        per_rung_maps[float(mother.ds)] = (
            [mother_prob], [(float(mother.x), float(mother.y))], [], [])
    print(f'      mother ds {mother.ds:g}: 1 tile ({time.perf_counter() - t_start:.2f}s)',
         flush=True)

    n_done_total = 1
    for ds, groups in groups_by_ds.items():
        n_main = sum(len(g.main) for g in groups)
        infos = ([m for g in groups for m in g.main]
                + [g.overlap for g in groups])
        images = images_by_ds[ds]
        n_tiles = len(infos)
        main_pts: List[np.ndarray] = []
        overlap_pts: List[np.ndarray] = []
        all_xy: List[np.ndarray] = []
        all_sc: List[np.ndarray] = []
        main_maps: List[np.ndarray] = []
        main_origins: List[Tuple[float, float]] = []
        overlap_maps: List[np.ndarray] = []
        overlap_origins: List[Tuple[float, float]] = []
        t_rung = time.perf_counter()
        for k, (info, image) in enumerate(zip(infos, images)):
            xy0, sc, prob = _detect_one(info, image)
            all_xy.append(xy0)
            all_sc.append(sc)
            is_main = k < n_main
            (main_pts if is_main else overlap_pts).append(xy0)
            if keep_maps:
                origin = (float(info.x), float(info.y))
                if is_main:
                    main_maps.append(prob)
                    main_origins.append(origin)
                else:
                    overlap_maps.append(prob)
                    overlap_origins.append(origin)
            n_done_total += 1
            if (k + 1) % 50 == 0 or k + 1 == n_tiles:
                elapsed = time.perf_counter() - t_rung
                print(f'      ds {ds:g}: {k + 1}/{n_tiles} tiles '
                     f'({elapsed:.1f}s, {elapsed / (k + 1) * 1000:.0f} ms/tile, '
                     f'{n_done_total} total so far, '
                     f'{time.perf_counter() - t_start:.1f}s elapsed)', flush=True)
        if keep_maps:
            per_rung_maps[float(ds)] = (main_maps, main_origins,
                                       overlap_maps, overlap_origins)
        per_rung_tiles[float(ds)] = (
            main_pts,
            np.concatenate(overlap_pts, axis=0) if overlap_pts
            else np.zeros((0, 2), np.float64))
        per_rung_detections[float(ds)] = (
            np.concatenate(all_xy, axis=0) if all_xy
            else np.zeros((0, 2), np.float64),
            np.concatenate(all_sc, axis=0) if all_sc
            else np.zeros(0, np.float32))
    if keep_maps:
        return per_rung_tiles, per_rung_detections, per_rung_maps
    return per_rung_tiles, per_rung_detections


def _to_tensor(image: np.ndarray, channels: int):
    """`[H, W, 3] uint8` -> `[C, H, W] float32` in 0..1. Grayscale by the
    same weights `Datasets._to_tensor` uses, since the student was trained
    on that conversion.
    """
    import torch                                             # noqa: PLC0415

    array = np.asarray(image, np.float32) / 255.0
    if int(channels) == 1:
        array = (0.299 * array[..., 0] + 0.587 * array[..., 1]
                 + 0.114 * array[..., 2])[..., None]
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
