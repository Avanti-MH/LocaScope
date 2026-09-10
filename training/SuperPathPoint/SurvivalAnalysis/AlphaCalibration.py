"""Calibrate alpha (`tau = max(tau_floor, alpha * ds)`) against a decoy.
plan.md 2.2①.

    curve = alpha_curve(dist, score, decoy_dist, decoy_score,
                        rungs=rungs, alphas=alphas, tau_floor=tau_floor,
                        threshold=threshold)
    bucket = aggregate_curves([alpha_curve(...) for each ChainStack])

SIDE-QUEST, NOT THE SIX-PATTERN ANALYSIS ITSELF. This module only answers
"how wide should tau be" -- classifying a point's own six-pattern membership
(`Patterns.classify`) never needs a decoy. Kept in its own file, separate
from `SurvivalProcess.py` (real measurement only) and `Report.py` (single
already-built table -> numbers), because it is the one piece of this package
that both (a) needs a decoy at all and (b) aggregates across many
ChainStacks before it means anything -- neither existing file's shape fits.

margin (`match_rate / decoy_rate`) IS AGGREGATED IN LOG SPACE. It is a ratio
whose denominator can sit near 0 for a single ChainStack, and a handful of
those blow the linear mean up past anything the rest report; log turns the
ratio into a difference, which averages the way a bounded quantity does.
`match_rate`/`decoy_rate`/`gap` are bounded and are not.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if os.path.join(_HERE, '..') not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, '..'))

from SurvivalAnalysis import SurvivalProcess                      # noqa: E402
from SurvivalAnalysis.Attribution import NONE                      # noqa: E402
from SurvivalAnalysis.Patterns import PATTERNS, alive_from, classify  # noqa: E402
from SurvivalAnalysis.NullModel import alive_rate_of, null_patterns  # noqa: E402
from SurvivalAnalysis.AliveCandidates import (                     # noqa: E402
    probe_via_probability_map, alive_probability_map,
    alive_scale_local_extremum)

DecoyShift = Callable[[np.ndarray], np.ndarray]


def merge_anchors(anchors: np.ndarray, radius: float,
                  priority: Optional[np.ndarray] = None) -> np.ndarray:
    """Indices to KEEP after a SECOND, read-time merge of the anchor list.
    `radius=0` returns every anchor -- the baseline a sensitivity check
    compares against (module docstring: this radius is not tuned to a
    value, it is swept to see whether the headline numbers move).

    Same primitive `SurvivalProcess.anchors_of` uses for its own (build-time,
    `nms_radius`) merge. `priority` lets the caller keep the strongest
    anchor of a cluster (e.g. `score.max(axis=1)` once probed) instead of
    whichever was encountered first.
    """
    return SurvivalProcess._merge_within_radius(anchors, radius,
                                                priority=priority)


def probe_decoy(anchors: np.ndarray,
                per_rung_detections: Dict[float, Tuple[np.ndarray, np.ndarray]],
                order: Sequence[float],
                decoy_shift: Union[DecoyShift, Dict[float, DecoyShift]]
                ) -> Tuple[np.ndarray, np.ndarray]:
    """`(decoy_dist, decoy_score)`, each `[N, len(order)]` -- the same
    question `SurvivalProcess.probe_real` asks, at a place each anchor is
    NOT.

    `decoy_shift` is either one `callable(anchors) -> shifted` used at every
    rung, or `{ds: callable}` for a per-rung magnitude (e.g. one that scales
    with that rung's own `ds`, matching how `tau` itself scales).
    """
    n = len(anchors)
    length = len(order)
    decoy_dist = np.full((n, length), NONE, np.float32)
    decoy_score = np.zeros((n, length), np.float32)
    if not n:
        return decoy_dist, decoy_score

    per_rung = decoy_shift if isinstance(decoy_shift, dict) else None
    for j, ds in enumerate(order):
        shift = per_rung[ds] if per_rung is not None else decoy_shift
        shifted = shift(anchors)
        found, found_score = per_rung_detections[ds]
        far, far_score = SurvivalProcess.nearest_detection(
            found, found_score, shifted)
        decoy_dist[:, j] = far
        decoy_score[:, j] = far_score
    return decoy_dist, decoy_score


def alpha_curve(dist: np.ndarray, score: np.ndarray, decoy_dist: np.ndarray,
                decoy_score: np.ndarray, *, rungs: Sequence[float],
                alphas: Sequence[float], tau_floor: float, threshold: float,
                alive_fn: Optional[Callable[..., np.ndarray]] = None
                ) -> Dict[str, np.ndarray]:
    """One ChainStack's `match_rate`/`decoy_rate`/`gap`/`margin`, each
    `[L, len(alphas)]` -- rungs down the rows, alpha across the columns.

        tau[j, k]        = max(tau_floor, alphas[k] * rungs[j])
        alive[i, j, k]   = (score[i,j] > threshold) & (dist[i,j] >= 0)
                          & (dist[i,j] <= tau[j, k])
        match_rate[j, k] = mean_i alive[i, j, k]
        gap              = match_rate - decoy_rate       (signed, no abs())
        margin           = match_rate / max(decoy_rate, eps)

    `alive_fn`, if given, REPLACES the formula above with
    `alive_fn(score, dist, tau=tau[:, k]) -> alive[N, L]`, called once per
    alpha column `k` -- the uniform signature every `AliveCandidates.py`
    method is wrapped down to before being passed in here (this module does
    not know about `score_threshold` vs `combined_threshold` vs whatever the
    next candidate needs; the caller binds that). `None` (the default) keeps
    the fast, fully-vectorised baseline path -- no behaviour change for
    every existing caller.
    """
    rungs_arr = np.asarray(list(rungs), np.float64)
    alphas_arr = np.asarray(list(alphas), np.float64)
    tau = np.maximum(float(tau_floor),
                     alphas_arr[None, :] * rungs_arr[:, None])      # [L, K]

    def _rate(d: np.ndarray, s: np.ndarray) -> np.ndarray:
        if not len(d):
            return np.full(tau.shape, np.nan)
        if alive_fn is None:
            alive = ((s[:, :, None] > float(threshold))
                    & (d[:, :, None] >= 0.0)
                    & (d[:, :, None] <= tau[None, :, :]))
            return alive.mean(axis=0)
        out = np.empty(tau.shape, np.float64)
        for k in range(tau.shape[1]):
            out[:, k] = alive_fn(s, d, tau=tau[:, k]).mean(axis=0)
        return out

    match_rate = _rate(dist, score)
    decoy_rate = _rate(decoy_dist, decoy_score)
    gap = match_rate - decoy_rate
    margin = match_rate / np.maximum(decoy_rate, 1e-9)
    return {'match_rate': match_rate, 'decoy_rate': decoy_rate,
            'gap': gap, 'margin': margin,
            'rungs': rungs_arr, 'alphas': alphas_arr}


def offset_quantiles_of(dist: np.ndarray, *, rungs: Sequence[float],
                        quantiles: Sequence[float] = (0.5, 0.9, 0.99)
                        ) -> np.ndarray:
    """`[L, len(quantiles)]` -- per-rung quantiles of the REAL offset
    distribution, among anchors that found something (`dist >= 0`).
    Independent of tau/alpha/threshold: a property of the data, checked
    against whichever alpha the gap curve picks (tau below the 90th
    percentile declares a tenth of real matches dead by construction).
    """
    rungs_arr = np.asarray(list(rungs), np.float64)
    q = np.asarray(list(quantiles), np.float64)
    out = np.full((len(rungs_arr), len(q)), np.nan, np.float64)
    for j in range(len(rungs_arr)):
        valid = dist[:, j][dist[:, j] >= 0.0]
        if len(valid):
            out[j] = np.quantile(valid, q)
    return out


def _aggregate(stacked_by_key: Dict[str, np.ndarray],
               log_keys: Sequence[str] = ()) -> Dict[str, np.ndarray]:
    """Mean + std across axis 0 (ChainStacks), per key. Keys in `log_keys`
    are aggregated in log space and exponentiated back.
    """
    out: Dict[str, np.ndarray] = {}
    for key, arr in stacked_by_key.items():
        if key in log_keys:
            logged = np.log(np.maximum(arr, 1e-9))
            out[f'{key}_mean'] = np.exp(np.nanmean(logged, axis=0))
            out[f'{key}_std'] = np.exp(np.nanstd(logged, axis=0))
        else:
            out[f'{key}_mean'] = np.nanmean(arr, axis=0)
            out[f'{key}_std'] = np.nanstd(arr, axis=0)
    return out


def aggregate_curves(curves: Sequence[Dict[str, np.ndarray]]
                     ) -> Dict[str, np.ndarray]:
    """Mean + std of `alpha_curve`'s output across every ChainStack of one
    bucket -- never across F/R/C. Each ChainStack's own curve is computed
    first; this only combines the finished curves, so one anchor-rich
    ChainStack cannot dominate the bucket's average.
    """
    stacked = {k: np.stack([c[k] for c in curves], axis=0)
              for k in ('match_rate', 'decoy_rate', 'gap', 'margin')}
    out = _aggregate(stacked, log_keys=('margin',))
    out['rungs'] = curves[0]['rungs']
    out['alphas'] = curves[0]['alphas']
    out['n_chainstacks'] = len(curves)
    return out


def aggregate_offset_quantiles(per_chainstack: Sequence[np.ndarray]
                               ) -> Dict[str, np.ndarray]:
    """Mean + std of `offset_quantiles_of`'s output across every ChainStack
    of one bucket. Same aggregation primitive `aggregate_curves` uses.
    """
    out = _aggregate({'offset_quantile': np.stack(per_chainstack, axis=0)})
    out['n_chainstacks'] = len(per_chainstack)
    return out


def _pattern_fractions(alive: np.ndarray) -> np.ndarray:
    """`[len(PATTERNS)]` -- fraction of rows classified into each pattern.
    `EMPTY` rows cannot occur by construction (every anchor is alive at its
    own `born_rung`, AlphaSelectionNotes.md section on how the six patterns
    are computed) but are excluded from both the count and the denominator
    defensively rather than assumed away.
    """
    counts = {p: 0 for p in PATTERNS}
    total = 0
    for row in alive:
        name, _lo, _hi = classify(row)
        if name in counts:
            counts[name] += 1
            total += 1
    return np.array([counts[p] / total if total else np.nan for p in PATTERNS])


def _null_fractions(alive: np.ndarray) -> np.ndarray:
    """`[len(PATTERNS)]` -- `NullModel.null_patterns` evaluated at this
    `alive` matrix's own measured per-rung rates, read back in `PATTERNS`
    order.
    """
    rate = alive_rate_of(alive)
    null = null_patterns(rate)
    return np.array([null.get(p, np.nan) for p in PATTERNS])


def pattern_curve(dist: np.ndarray, score: np.ndarray, decoy_dist: np.ndarray,
                  decoy_score: np.ndarray, *, rungs: Sequence[float],
                  alphas: Sequence[float], tau_floor: float, threshold: float,
                  alive_fn: Optional[Callable[..., np.ndarray]] = None
                  ) -> Dict[str, np.ndarray]:
    """One ChainStack's six-pattern composition and `NullModel` excess, for
    both real anchors and decoy anchors, as a function of alpha. Each output
    key is `[len(PATTERNS), len(alphas)]` -- patterns down the rows (`
    PATTERNS` order), alpha across the columns, the same `[rows, alpha]`
    shape `alpha_curve` uses with rung replaced by pattern.

    `gap` (from `alpha_curve`) stays the primary way this project picks
    alpha, cross-checked against the keypoint-precision / coarse-rung-error
    estimate (physical evidence, not this file's decoy); whether
    `excess`/`excess_decoy` below ever becomes a THIRD criterion is still
    open -- AlphaSelectionNotes.md's Q7 and section 8.

        measured_real[p, k]  = _pattern_fractions(alive_from(real))
        measured_decoy[p, k] = _pattern_fractions(alive_from(decoy))
        null_real[p, k]      = _null_fractions(alive_from(real))
        null_decoy[p, k]     = _null_fractions(alive_from(decoy))
        excess_*             = measured_* - null_*             (signed)

    `alive_fn`, if given, replaces `Patterns.alive_from` with
    `alive_fn(score, dist, tau=tau) -> alive[N, L]`, called once per alpha --
    same uniform signature `alpha_curve` takes, same reason (this module
    stays ignorant of which threshold name the selected method needs).
    """
    order = np.asarray(sorted(rungs), dtype=np.float64)
    alphas_arr = np.asarray(list(alphas), dtype=np.float64)
    n_patterns = len(PATTERNS)
    shape = (n_patterns, len(alphas_arr))
    measured_real = np.zeros(shape)
    measured_decoy = np.zeros(shape)
    null_real = np.zeros(shape)
    null_decoy = np.zeros(shape)

    for k, alpha in enumerate(alphas_arr):
        tau = np.maximum(float(tau_floor), float(alpha) * order)
        if alive_fn is None:
            alive_real = alive_from(score, dist, score_threshold=threshold,
                                    tau=tau)
            alive_decoy = alive_from(decoy_score, decoy_dist,
                                     score_threshold=threshold, tau=tau)
        else:
            alive_real = alive_fn(score, dist, tau=tau)
            alive_decoy = alive_fn(decoy_score, decoy_dist, tau=tau)
        measured_real[:, k] = _pattern_fractions(alive_real)
        measured_decoy[:, k] = _pattern_fractions(alive_decoy)
        null_real[:, k] = _null_fractions(alive_real)
        null_decoy[:, k] = _null_fractions(alive_decoy)

    return {'measured_real': measured_real, 'measured_decoy': measured_decoy,
            'null_real': null_real, 'null_decoy': null_decoy,
            'excess_real': measured_real - null_real,
            'excess_decoy': measured_decoy - null_decoy,
            'alphas': alphas_arr}


def aggregate_pattern_curves(curves: Sequence[Dict[str, np.ndarray]]
                             ) -> Dict[str, np.ndarray]:
    """Mean + std of `pattern_curve`'s output across every ChainStack of one
    bucket. Same aggregation primitive `aggregate_curves` uses; no key here
    is a ratio prone to blowing up near a zero denominator (fractions and
    signed differences of fractions are all bounded), so nothing is
    log-aggregated.
    """
    keys = ('measured_real', 'measured_decoy', 'null_real', 'null_decoy',
           'excess_real', 'excess_decoy')
    stacked = {k: np.stack([c[k] for c in curves], axis=0) for k in keys}
    out = _aggregate(stacked)
    out['alphas'] = curves[0]['alphas']
    out['n_chainstacks'] = len(curves)
    return out


# ── candidates 1/2 (probability-map probe): SEPARATE from alpha_curve/ ────────
# pattern_curve on purpose, not routed through their alive_fn hook -- spec.md
# "同一個點的定義" section's neighbour explains why: `alive_fn(score, dist,
# tau)` is called once for real and once for decoy with no way to tell which,
# because every OTHER method reads that distinction off already-computed
# `score`/`dist` (computed against the right anchor set by the caller before
# `alive_fn` runs). Candidates 1/2 ignore `score`/`dist` entirely -- they
# probe the probability field fresh, at `tau`-dependent radius, every alpha --
# so they have nothing else to tell real and decoy apart with. These two
# functions match `alpha_curve`/`pattern_curve`'s OUTPUT shape and keys
# exactly (so `aggregate_curves`/`aggregate_pattern_curves` and every CSV
# writer/reader/plotter downstream need no changes at all), just not their
# internal loop.

def _probability_map_alive(combined_maps: Dict[float, np.ndarray],
                           anchors_by_rung: Dict[float, np.ndarray],
                           order: np.ndarray, *,
                           footprint_origin: Dict[float, Tuple[float, float]],
                           rung_scale: Dict[float, float],
                           tau: np.ndarray, score_threshold: float,
                           kind: str, extremum_margin: float = 0.0,
                           sample_step: float = 0.5
                           ) -> np.ndarray:
    """One alpha's `alive[N, L]` for candidate 1 ('probability_map') or
    candidate 2 ('scale_extremum') -- the one place `probe_via_probability_
    map` actually runs, shared by `probability_map_curve` (match/gap) and
    `probability_map_pattern_curve` (six-pattern) so the expensive per-rung
    probe is written once, not duplicated between the two views.

    `anchors_by_rung[ds]` is which coordinates to probe AT rung `ds` -- the
    same `[N, 2]` array at every rung for real anchors (they do not move),
    a DIFFERENT per-rung array for decoy (`_decoy_shift_per_rung`'s shift
    scales with that rung's own `rung_shrink`) -- this is the one thing a
    plain `[N, 2]` argument could not express, which is why it is a dict
    keyed by `ds` rather than a single array.

    `footprint_origin`/`rung_scale` are ALSO per-rung dicts, not one shared
    value -- true on C (every generation covers the mother tile's own
    rectangle, so both happen to be constant across rungs there) but NOT on
    F (the footprint GROWS with `ds`, so each rung's own tile has a
    DIFFERENT origin) or R (`ChainStack.rung_scale('R', ds) == 1.0` at
    EVERY rung -- the `ds` label there is a `degrade_resolution` amount,
    not a real geometric scale, spec.md "同一個點的定義" makes the same
    point about `anchors_of`'s own `rung_scale` override). Passing a single
    constant here would have been silently wrong the moment this got
    reused for F/R instead of only C.

    `sample_step` passes straight through to `probe_via_probability_map` --
    that function's own docstring calls it "a real accuracy/speed knob, not
    yet tuned"; the default here matches its own default (0.5) rather than
    silently picking a different one.
    """
    n = len(next(iter(anchors_by_rung.values()))) if anchors_by_rung else 0
    length = len(order)
    peak_value = np.zeros((n, length), np.float64)
    peak_dist = np.zeros((n, length), np.float64)
    for j, ds in enumerate(order):
        ds = float(ds)
        pv, pd = probe_via_probability_map(
            combined_maps[ds], anchors_by_rung[ds],
            map_origin=footprint_origin[ds], scale=rung_scale[ds],
            tau=float(tau[j]), sample_step=sample_step)
        peak_value[:, j] = pv
        peak_dist[:, j] = pd
    if kind == 'probability_map':
        return alive_probability_map(peak_value, peak_dist,
                                     score_threshold=score_threshold, tau=tau)
    if kind == 'scale_extremum':
        return alive_scale_local_extremum(
            peak_value, peak_dist, score_threshold=score_threshold, tau=tau,
            margin=extremum_margin)
    raise ValueError(f"kind must be 'probability_map' or 'scale_extremum', "
                     f"got {kind!r}")


def _probe_positions(anchors: np.ndarray, order: np.ndarray,
                     decoy_shift: Dict[float, Callable[[np.ndarray], np.ndarray]]
                     ) -> Tuple[Dict[float, np.ndarray], Dict[float, np.ndarray]]:
    """`(real_by_rung, decoy_by_rung)` -- real anchors repeated at every
    rung (they are fixed coordinates), decoy anchors shifted per rung by
    that rung's own `decoy_shift[ds]`. Computed once, alpha-independent
    (`_decoy_shift_per_rung`'s magnitude scales with `rung_shrink`, not
    with alpha) -- shared by both `probability_map_curve` and
    `probability_map_pattern_curve` so the shift itself is not redrawn
    (`random`/`rotate` decoy kinds draw fresh randomness per call) between
    the two views on the same alpha sweep.
    """
    real_by_rung = {float(ds): anchors for ds in order}
    decoy_by_rung = {float(ds): decoy_shift[float(ds)](anchors) for ds in order}
    return real_by_rung, decoy_by_rung


def probability_map_curve(combined_maps: Dict[float, np.ndarray],
                          anchors: np.ndarray,
                          decoy_shift: Dict[float, Callable[[np.ndarray], np.ndarray]],
                          *, rungs: Sequence[float], alphas: Sequence[float],
                          tau_floor: float, threshold: float,
                          footprint_origin: Dict[float, Tuple[float, float]],
                          rung_scale: Dict[float, float], kind: str,
                          extremum_margin: float = 0.0, sample_step: float = 0.5
                          ) -> Dict[str, np.ndarray]:
    """Candidate 1/2's `alpha_curve` equivalent -- identical output keys and
    `[L, len(alphas)]` shapes (`match_rate`/`decoy_rate`/`gap`/`margin`,
    `rungs`/`alphas`), computed by re-probing the probability field fresh
    every alpha instead of thresholding a `dist`/`score` computed once.
    `footprint_origin`/`rung_scale`: see `_probability_map_alive`'s own
    docstring -- per-rung dicts, F/R/C each build them differently.
    """
    order = np.asarray(sorted(rungs), dtype=np.float64)
    alphas_arr = np.asarray(list(alphas), dtype=np.float64)
    shape = (len(order), len(alphas_arr))
    match_rate = np.full(shape, np.nan)
    decoy_rate = np.full(shape, np.nan)

    if len(anchors):
        real_by_rung, decoy_by_rung = _probe_positions(anchors, order, decoy_shift)
        for k, alpha in enumerate(alphas_arr):
            tau = np.maximum(float(tau_floor), float(alpha) * order)
            alive_real = _probability_map_alive(
                combined_maps, real_by_rung, order,
                footprint_origin=footprint_origin, rung_scale=rung_scale,
                tau=tau, score_threshold=threshold, kind=kind,
                extremum_margin=extremum_margin, sample_step=sample_step)
            alive_decoy = _probability_map_alive(
                combined_maps, decoy_by_rung, order,
                footprint_origin=footprint_origin, rung_scale=rung_scale,
                tau=tau, score_threshold=threshold, kind=kind,
                extremum_margin=extremum_margin, sample_step=sample_step)
            match_rate[:, k] = alive_real.mean(axis=0)
            decoy_rate[:, k] = alive_decoy.mean(axis=0)

    gap = match_rate - decoy_rate
    margin = match_rate / np.maximum(decoy_rate, 1e-9)
    return {'match_rate': match_rate, 'decoy_rate': decoy_rate,
            'gap': gap, 'margin': margin,
            'rungs': order, 'alphas': alphas_arr}


def probability_map_pattern_curve(
        combined_maps: Dict[float, np.ndarray], anchors: np.ndarray,
        decoy_shift: Dict[float, Callable[[np.ndarray], np.ndarray]], *,
        rungs: Sequence[float], alphas: Sequence[float], tau_floor: float,
        threshold: float, footprint_origin: Dict[float, Tuple[float, float]],
        rung_scale: Dict[float, float], kind: str,
        extremum_margin: float = 0.0, sample_step: float = 0.5
        ) -> Dict[str, np.ndarray]:
    """Candidate 1/2's `pattern_curve` equivalent -- identical output keys
    and `[len(PATTERNS), len(alphas)]` shapes (measured/null/excess, real
    and decoy), same re-probe-every-alpha loop `probability_map_curve` uses.
    `footprint_origin`/`rung_scale`: see `_probability_map_alive`.
    """
    order = np.asarray(sorted(rungs), dtype=np.float64)
    alphas_arr = np.asarray(list(alphas), dtype=np.float64)
    shape = (len(PATTERNS), len(alphas_arr))
    measured_real = np.zeros(shape)
    measured_decoy = np.zeros(shape)
    null_real = np.zeros(shape)
    null_decoy = np.zeros(shape)

    if len(anchors):
        real_by_rung, decoy_by_rung = _probe_positions(anchors, order, decoy_shift)
        for k, alpha in enumerate(alphas_arr):
            tau = np.maximum(float(tau_floor), float(alpha) * order)
            alive_real = _probability_map_alive(
                combined_maps, real_by_rung, order,
                footprint_origin=footprint_origin, rung_scale=rung_scale,
                tau=tau, score_threshold=threshold, kind=kind,
                extremum_margin=extremum_margin, sample_step=sample_step)
            alive_decoy = _probability_map_alive(
                combined_maps, decoy_by_rung, order,
                footprint_origin=footprint_origin, rung_scale=rung_scale,
                tau=tau, score_threshold=threshold, kind=kind,
                extremum_margin=extremum_margin, sample_step=sample_step)
            measured_real[:, k] = _pattern_fractions(alive_real)
            measured_decoy[:, k] = _pattern_fractions(alive_decoy)
            null_real[:, k] = _null_fractions(alive_real)
            null_decoy[:, k] = _null_fractions(alive_decoy)

    return {'measured_real': measured_real, 'measured_decoy': measured_decoy,
            'null_real': null_real, 'null_decoy': null_decoy,
            'excess_real': measured_real - null_real,
            'excess_decoy': measured_decoy - null_decoy,
            'alphas': alphas_arr}
