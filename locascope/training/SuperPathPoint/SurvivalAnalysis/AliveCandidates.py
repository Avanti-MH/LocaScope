"""Candidate replacements for `Patterns.alive_from`. Named, not adopted.

`AlphaSelectionNotes.md` section 14 has the comparison table (what each one
solves, what it costs) that produced these names and signatures. NONE OF
THESE ARE WIRED IN -- `Patterns.alive_from` (the absolute dual-threshold
method) is what `SurvivalProcess`/`AlphaCalibration` actually call. This file
exists so the candidates have a real name and a real signature to argue
about, not just a row in a table -- "later" needed a place to land.

Most candidates keep `alive_from`'s own shape contract: `score`/`dist` are
`[N, L]` (N points, L rungs, finest first), `tau` is `[L]`, the return is
`alive[N, L]` bool. `alive_probability_map` is the exception -- it reads the
raw probability maps directly instead of an already-thresholded peak list,
so its signature looks like `SurvivalProcess.probe_real`'s, not
`alive_from`'s; see its own docstring for why.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def alive_absolute_dual_threshold(score: np.ndarray, dist: np.ndarray, *,
                                  score_threshold: float, tau: np.ndarray
                                  ) -> np.ndarray:
    """絕對雙門檻法 (current production method) -- `Patterns.alive_from`
    itself. Kept here as a same-signature reference point for comparing the
    other four against, not a second implementation to maintain: delegates.

        alive[i,j] = (score[i,j] > score_threshold)
                    & (0 <= dist[i,j] <= tau[j])
    """
    from SurvivalAnalysis.Patterns import alive_from  # noqa: PLC0415
    return alive_from(score, dist, score_threshold=score_threshold, tau=tau)


def assemble_generation_map(main_maps: Sequence[np.ndarray],
                            main_origins: Sequence[Tuple[float, float]],
                            overlap_maps: Sequence[np.ndarray],
                            overlap_origins: Sequence[Tuple[float, float]],
                            scale: float, footprint_origin: Tuple[float, float],
                            footprint_shape: Tuple[int, int],
                            overlap_mode: str = 'union') -> np.ndarray:
    """Step A -- one rung's main tiles' raw probability maps stitched into a
    single dense field covering the whole generation's footprint, with the
    overlap tile folded in where it spatially coincides.

    TWO STAGES, because main tiles and the overlap tile relate to each other
    differently:

        main tiles among themselves    do not overlap (they partition the
                                       generation's footprint), so combining
                                       them is placement, not arithmetic --
                                       each pixel is covered by exactly one
                                       main tile.
        overlap tile against that      DOES spatially coincide with the
                                       already-placed main stitch, but only
                                       across the seams BETWEEN main tiles --
                                       the outermost half-tile-size border of
                                       the whole footprint has no overlap
                                       contribution at all, by construction
                                       (the overlap tile straddles an
                                       INTERNAL boundary, not the generation's
                                       outer edge).

    `overlap_mode` (same vocabulary as `SurvivalProcess.anchors_of_
    generations`'s parameter of the same name, translated from set
    union/intersection to a per-pixel combination of two continuous
    probabilities):

        union         combined = max(main, overlap) at each pixel the
                      overlap tile covers -- "either source seeing a strong
                      response is enough", matching why the overlap tile
                      exists (a real feature near a seam can look weaker in
                      whichever tile's receptive field it was truncated by;
                      max keeps the untruncated reading rather than
                      averaging it down).
        intersection  combined = min(main, overlap) -- both sources have to
                      independently agree it is strong.

    `main_maps`/`overlap_maps` are FLAT lists (already gathered across every
    `TileGroup` in this generation, the same flattening `detect_all_
    generations` already does for `infos` -- a generation with several
    `TileGroup`s has one overlap tile PER GROUP, not one for the whole
    generation, so this takes a list, not a single tile, on that side too.

    `footprint_origin`/`footprint_shape` are the WHOLE generation's own
    level-0 origin and this map's pixel shape (`(rows, cols)`) -- the space
    every tile gets placed into. A tile whose placement would fall outside
    that shape raises rather than silently clipping or wrapping: that is a
    coordinate-conversion bug upstream, not a legitimate edge case (C-tree
    generations all cover the SAME footprint as the mother tile by
    construction, AlphaSelectionNotes.md section 15/Q on footprint
    equality).
    """
    if overlap_mode not in ('union', 'intersection'):
        raise ValueError(f"overlap_mode must be 'union' or 'intersection', "
                         f"got {overlap_mode!r}")
    combine_fn = np.maximum if overlap_mode == 'union' else np.minimum

    rows, cols = footprint_shape
    combined = np.zeros((rows, cols), dtype=np.float64)
    covered = np.zeros((rows, cols), dtype=bool)
    fx0, fy0 = footprint_origin

    def _place(tile_map: np.ndarray, origin: Tuple[float, float]
              ) -> Tuple[int, int, int, int]:
        h, w = tile_map.shape
        px = int(round((origin[0] - fx0) / scale))
        py = int(round((origin[1] - fy0) / scale))
        if px < 0 or py < 0 or px + w > cols or py + h > rows:
            raise ValueError(
                f'tile at origin {origin} (pixel {(px, py)}, size {(w, h)}) '
                f'falls outside the generation footprint {(cols, rows)} -- '
                f'a coordinate-conversion bug, not a legitimate edge tile')
        return px, py, w, h

    # stage 1: main tiles -- pure placement, no arithmetic (they do not
    # overlap each other).
    for tile_map, origin in zip(main_maps, main_origins):
        px, py, w, h = _place(tile_map, origin)
        combined[py:py + h, px:px + w] = tile_map
        covered[py:py + h, px:px + w] = True

    # stage 2: overlap tiles -- combine only where they coincide with an
    # already-placed main tile; elsewhere (should not happen for a real
    # overlap tile, which straddles an INTERNAL seam) fall back to the
    # overlap's own value rather than raising, since arithmetic against an
    # uncovered region is undefined, not wrong.
    for tile_map, origin in zip(overlap_maps, overlap_origins):
        px, py, w, h = _place(tile_map, origin)
        region = combined[py:py + h, px:px + w]
        region_covered = covered[py:py + h, px:px + w]
        combined[py:py + h, px:px + w] = np.where(
            region_covered, combine_fn(region, tile_map), tile_map)
        covered[py:py + h, px:px + w] = True

    return combined


def probe_via_probability_map(combined_map: np.ndarray, anchors: np.ndarray,
                              *, map_origin: Tuple[float, float], scale: float,
                              tau: float, sample_step: float = 0.5
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Step C -- for one rung, one anchor at a time: the peak of the
    bilinearly-interpolated probability field within a CIRCLE of radius
    `tau` (converted to this rung's own pixel units) around the anchor's own
    fixed coordinate. Returns `(peak_value, peak_dist)`, each `[N]` --
    `peak_dist` is the level-0 distance from the anchor to where the peak
    was found, playing `dist`'s role; `peak_value` plays `score`'s role.
    This REPLACES `SurvivalProcess.nearest_detection` for this candidate --
    there is no separate point list to search, `combined_map` (step A's
    output) is read directly.

        cx, cy      = (anchor - map_origin) / scale          (this map's own
                     continuous pixel coordinates)
        r           = tau / scale
        candidates  = points on a `sample_step`-pixel grid with
                     euclidean_dist(candidate, (cx,cy)) <= r      (a CIRCLE,
                     not the bounding square -- consistent with `dist<=tau`
                     everywhere else in this project using euclidean
                     distance)
        value(p)    = bilinear interpolation of combined_map at p, for each
                     candidate p
        peak        = argmax_p value(p)
        peak_value  = value(peak)
        peak_dist   = euclidean_dist(peak, (cx,cy)) * scale     (back to
                     level-0 px)

    `sample_step` (sub-pixel grid spacing, in this rung's own pixel units)
    is a real accuracy/speed knob, not yet tuned -- finer catches a sharper
    peak more precisely at more compute per anchor per rung.

    NO SEPARATE "NOT FOUND" CASE (accepted, not a defect to fix): a dense
    field always has SOME maximum in any non-empty window, so there is no
    analogue of `nearest_detection`'s empty-`points` `NONE` sentinel here --
    "nothing real nearby" and "something weak nearby" both come back as a
    low `peak_value`, told apart only by `score_threshold`, same as anywhere
    else this project relies on that threshold rather than a sentinel.

    `SurvivalProcess.rival_at` LIKELY BECOMES REDUNDANT once this exists: it
    answers "excluding the anchor's own peak, what is the strongest nearby
    responder", which is what a peak-of-the-window read already finds
    whenever the anchor's own peak is not the window's maximum. Whether
    `Attribution`'s suppression-release cause (the one existing consumer of
    `rival`/`rival_dist`) can be re-derived from this instead is a separate
    question, not resolved here.

    Out-of-bounds sample points (the window straddles `combined_map`'s own
    edge) read as `-inf`, not a clamped edge value -- an edge pixel's value
    repeated across the missing part of the window would bias the peak
    towards the boundary, which is worse than just excluding those samples
    from contention.
    """
    anchors = np.asarray(anchors, dtype=np.float64)
    n = len(anchors)
    peak_value = np.full(n, -np.inf, dtype=np.float64)
    peak_dist = np.zeros(n, dtype=np.float64)
    if not n:
        return peak_value, peak_dist

    r = float(tau) / float(scale)
    steps = np.arange(-r, r + sample_step, sample_step)
    dx, dy = np.meshgrid(steps, steps)
    within_circle = (dx ** 2 + dy ** 2) <= r ** 2
    dx, dy = dx[within_circle], dy[within_circle]
    offset_dist = np.hypot(dx, dy)

    map_origin_arr = np.asarray(map_origin, dtype=np.float64)
    h, w = combined_map.shape
    for i in range(n):
        cx, cy = (anchors[i] - map_origin_arr) / float(scale)
        xs, ys = cx + dx, cy + dy
        values = _bilinear_sample(combined_map, xs, ys, h, w)
        best = int(np.argmax(values))
        peak_value[i] = values[best]
        peak_dist[i] = offset_dist[best] * float(scale)
    return peak_value, peak_dist


def _bilinear_sample(map_: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                     h: int, w: int) -> np.ndarray:
    """Bilinear interpolation of `map_` at continuous coordinates
    `(xs, ys)`; points outside `[0, w-1] x [0, h-1]` come back `-inf` (see
    `probe_via_probability_map`'s docstring on why not a clamped edge
    value).
    """
    out_of_bounds = (xs < 0) | (xs > w - 1) | (ys < 0) | (ys > h - 1)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, w - 1)
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = np.clip(xs - x0, 0.0, 1.0)
    wy = np.clip(ys - y0, 0.0, 1.0)
    top = map_[y0, x0] * (1 - wx) + map_[y0, x1] * wx
    bottom = map_[y1, x0] * (1 - wx) + map_[y1, x1] * wx
    values = top * (1 - wy) + bottom * wy
    return np.where(out_of_bounds, -np.inf, values)


def alive_probability_map(peak_value: np.ndarray, peak_dist: np.ndarray, *,
                          score_threshold: float, tau: np.ndarray
                          ) -> np.ndarray:
    """機率圖法 (candidate 1) -- the actual alive decision, once
    `probe_via_probability_map` (step C, run once per rung and stacked into
    `[N, L]` arrays the same shape `alive_from` expects) has supplied
    `peak_value`/`peak_dist` in place of `score`/`dist`.

        alive[i,j] = (peak_value[i,j] > score_threshold)
                    & (peak_dist[i,j] <= tau[j])

    Deliberately NOT `alive_absolute_dual_threshold`'s formula with renamed
    inputs -- there is no `0 <= dist` clause here because `probe_via_
    probability_map` never returns a NONE sentinel (see its own docstring).
    """
    return ((peak_value > float(score_threshold))
           & (peak_dist <= tau[None, :]))


def alive_scale_local_extremum(peak_value: np.ndarray, peak_dist: np.ndarray,
                               *, score_threshold: float, tau: np.ndarray,
                               margin: float = 0.0) -> np.ndarray:
    """尺度局部極值法 (candidate 2) -- the closest translation of SIFT's own
    26-neighbour DoG extremum test to this pipeline's scale axis. NEEDS
    candidate 1's `probe_via_probability_map` to supply `peak_value`/
    `peak_dist` -- comparing `alive_absolute_dual_threshold`'s `score`
    (nearest INDEPENDENTLY-DETECTED point's score) across rungs is circular,
    because whether rung j-1's nearest detection is even the SAME feature as
    rung j's is exactly the question being asked. Reading `peak_value` at
    the anchor's own FIXED coordinate has no such dependency -- it compares
    the same location's response at neighbouring scales, which is what SIFT
    actually does.

        alive[i,j] = (peak_value[i,j] > score_threshold)
                    & (peak_dist[i,j] <= tau[j])
                    & (peak_value[i,j] >= peak_value[i,j-1] - margin)   (j>0)
                    & (peak_value[i,j] >= peak_value[i,j+1] - margin)   (j<L-1)

    Edge rungs (j=0, j=L-1) only have one neighbour to compare against.

    `peak_value`/`peak_dist` are `[N, L]`, finest rung first, `tau` is `[L]`
    -- same shapes `alive_probability_map` takes (both are stacked once per
    alpha from repeated `probe_via_probability_map` calls, one per rung).
    """
    peak_value = np.asarray(peak_value, dtype=np.float64)
    peak_dist = np.asarray(peak_dist, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    length = peak_value.shape[1]

    base = (peak_value > float(score_threshold)) & (peak_dist <= tau[None, :])
    is_extremum = np.ones_like(base)
    if length > 1:
        is_extremum[:, :-1] &= peak_value[:, :-1] >= peak_value[:, 1:] - margin
        is_extremum[:, 1:] &= peak_value[:, 1:] >= peak_value[:, :-1] - margin
    return base & is_extremum


def alive_exp_decay_joint_score(score: np.ndarray, dist: np.ndarray, *,
                                tau: np.ndarray, combined_threshold: float
                                ) -> np.ndarray:
    """指數衰減聯合分數法 (candidate 3). Blends the two independent gates
    into one soft score instead of requiring both to pass separately.

        combined[i,j] = score[i,j] * exp(-dist[i,j] / tau[j])
        alive[i,j]    = (0 <= dist[i,j]) & (combined[i,j] > combined_threshold)

    IMPLEMENTED, despite being flagged as the weakest candidate: `combined`
    conflates "no detection at all" (`dist < 0`) with "detected but far"
    into one continuous number, which breaks the NONE-sentinel separation
    `SurvivalProcess`/`Attribution` are built around on purpose (see
    `Attribution.py`'s own comment on why `NONE` is not 0 and not merged
    into any real value) -- but it needs no upstream pipeline change (same
    `score`/`dist` inputs as the baseline), so it is runnable now, weakness
    and all, while candidates 1/2/4 are still being built.

    `exp(-dist/tau)` is 1.0 at `dist=0` and decays smoothly, so `combined`
    is on `score`'s own scale at zero offset but strictly smaller everywhere
    else -- `combined_threshold` therefore has to be well below
    `score_threshold` to accept anything beyond an exact hit; the two are
    not interchangeable numbers (ClaudeRules 8: this is exactly why the CLI
    requires `--combined-threshold` explicitly rather than defaulting it to
    `--score-threshold`).
    """
    score = np.asarray(score, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    valid = dist >= 0.0
    combined = np.where(valid, score * np.exp(-dist / np.maximum(tau, 1e-9)),
                       -np.inf)
    return valid & (combined > float(combined_threshold))


def _rayleigh_pdf(x: np.ndarray, sigma: float, eps: float = 1e-9) -> np.ndarray:
    """`x/sigma^2 * exp(-x^2/(2*sigma^2))`, `x>=0` -- the distribution of
    distance-to-origin for a 2D isotropic Gaussian error of scale `sigma`.
    Chosen for both mixture components (not, say, Gaussian+exponential)
    because `dist` is a 2D radial distance either way: a genuine match's
    offset from isotropic localisation noise IS Rayleigh by construction,
    and a background/no-match distance (nearest point of a roughly uniform
    2D scatter) is ALSO Rayleigh-shaped (`P(R>r) = exp(-lambda*pi*r^2)` for
    a Poisson process of intensity `lambda`) -- just with a larger scale.
    Same functional form, different sigma, is the honest model here, not an
    arbitrary convenience.
    """
    sigma = max(float(sigma), eps)
    return (x / sigma ** 2) * np.exp(-(x ** 2) / (2.0 * sigma ** 2))


def _fit_rayleigh_mixture(x: np.ndarray, n_iter: int = 50, eps: float = 1e-9
                          ) -> Tuple[float, float, float]:
    """EM fit of a 2-component Rayleigh mixture to non-negative `x`.
    Returns `(pi_genuine, sigma_genuine, sigma_background)` -- `pi_genuine`
    is the weight of whichever fitted component has the SMALLER sigma
    (there is no guarantee EM's two components come out in that order, so
    they are relabelled after fitting, not assumed by initialisation).

    Standard two-step EM, both steps closed-form for a Rayleigh mixture:
        E: responsibility r_i = pi*f1(x_i) / (pi*f1(x_i) + (1-pi)*f2(x_i))
        M: pi        = mean(r)
           sigma_c^2 = sum(r_c * x^2) / (2 * sum(r_c))     (weighted Rayleigh
                                                            MLE, c in {1,2})
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[x >= 0.0]
    n = len(x)
    if n < 8:
        # Too little data for two components to be identifiable at all --
        # one Rayleigh fit to everything, called "genuine" by default so
        # this degrades to something closer to the baseline than to
        # arbitrarily rejecting everything.
        sigma = float(np.sqrt(np.mean(x ** 2) / 2.0)) if n else 1.0
        return 1.0, max(sigma, eps), max(sigma, eps) * 2.0

    sigma1 = max(float(np.percentile(x, 25)), eps)
    sigma2 = max(float(np.percentile(x, 90)), sigma1 * 2.0)
    pi = 0.5

    for _ in range(n_iter):
        f1 = _rayleigh_pdf(x, sigma1, eps)
        f2 = _rayleigh_pdf(x, sigma2, eps)
        num = pi * f1
        den = num + (1.0 - pi) * f2
        r = np.where(den > eps, num / np.maximum(den, eps), 0.5)

        pi = float(np.clip(r.mean(), eps, 1.0 - eps))
        w1, w2 = r.sum(), (1.0 - r).sum()
        sigma1 = float(np.sqrt(max((r * x ** 2).sum() / (2.0 * max(w1, eps)), eps)))
        sigma2 = float(np.sqrt(max(((1.0 - r) * x ** 2).sum() / (2.0 * max(w2, eps)), eps)))

    if sigma1 > sigma2:
        sigma1, sigma2 = sigma2, sigma1
        pi = 1.0 - pi
    return pi, sigma1, sigma2


def alive_two_component_mixture(score: np.ndarray, dist: np.ndarray, *,
                                score_threshold: float,
                                posterior_threshold: float = 0.5
                                ) -> np.ndarray:
    """雙成分混合模型法 (candidate 4). Fits `dist`'s own empirical
    distribution, PER RUNG (one fit per column of `dist`, pooling across all
    N anchors -- not per anchor), as a two-component Rayleigh mixture -- a
    "genuine match" component (small sigma) and a "background/no-match"
    component (large sigma, AlphaSelectionNotes.md section 9's "nearest
    thing happened to be however far" case) -- and calls a point alive by
    posterior probability rather than a hard distance cutoff.

        alive[i,j] = (score[i,j] > score_threshold)
                    & (P(genuine | dist[i,j]) > posterior_threshold)

    Unlike the other four candidates, this one is NOT a per-(anchor,rung)
    pure function -- fitting needs the WHOLE column `dist[:,j]` before any
    single anchor's posterior can be computed, which is why it loops over
    rungs internally instead of taking a scalar `tau` the way the others do.

    Data volume caveat (still open, not resolved here): C axis's 10
    ChainStacks means this fits INDEPENDENTLY per ChainStack per rung on
    whatever N that ChainStack contributes -- a tree with too few anchors
    at some rung falls into `_fit_rayleigh_mixture`'s n<8 fallback (one
    component, called all "genuine"), which is a documented degradation,
    not a crash, but is exactly the low-data instability
    AlphaSelectionNotes.md flagged as this candidate's biggest risk.
    """
    score = np.asarray(score, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)
    n, length = dist.shape
    posterior = np.zeros((n, length), dtype=np.float64)
    for j in range(length):
        column = dist[:, j]
        valid = column >= 0.0
        pi, sigma_genuine, sigma_background = _fit_rayleigh_mixture(column[valid])
        f_genuine = _rayleigh_pdf(column, sigma_genuine)
        f_background = _rayleigh_pdf(column, sigma_background)
        num = pi * f_genuine
        den = num + (1.0 - pi) * f_background
        posterior[:, j] = np.where(valid & (den > 0.0),
                                  num / np.maximum(den, 1e-9), 0.0)
    return (score > float(score_threshold)) & (posterior > float(posterior_threshold))
