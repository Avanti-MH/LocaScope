"""Survival tables -> the numbers. spec.md 3.2, plan.md P1.

    rows = pattern_table(batch, meta, thresholds=(0.005, 0.015, 0.03))
    rows = attribution_table(f_batch, f_meta, r_batch, r_meta, thresholds=...)

PURE: NO print, NO plot, NO file. Every function takes arrays and returns
dicts or arrays, because the three numbers this produces decide Stage C's
design (plan.md P1) and a number that needs a GPU to check is a number nobody
checks. `cli/report_survival.py` does the printing, the plotting and the CSVs.

EVERY TABLE IS SWEPT OVER THE THRESHOLD, AND THAT IS NOT AN OPTION
===================================================================
`alive` is derived (`Patterns.alive_from`), so a pattern fraction is a function
of the detection threshold. One threshold's answer is not a finding: the finding
is either that the fraction barely moves across a plausible range, or that it
does -- and in the second case there is no finding. So `thresholds` is a
required argument of every table here rather than a default someone can forget.

`只在一階` IS THE MOST FRAGILE AND THE SWEEP MATTERS MOST FOR IT
=================================================================
It is defined by two negative decisions, one on each side of the live rung.
`一直存活` survives one flip as `細部存活`; `只在一階` does not survive one as
anything adjacent. If any column in the sweep is going to move, it is that one.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from PointsAnalysisByMpp import Attribution, NullModel
from PointsAnalysisByMpp.Patterns import (EMPTY, PATTERNS, alive_from,
                                          band_fraction, classify)


def alive_of(batch, meta, threshold: float) -> np.ndarray:
    """`[N, L]` bool, from the stored columns and one threshold.

    The single place the two cuts are applied to a whole table, so that every
    number in this module is downstream of one spelling of "alive".
    """
    tau = meta.tau()
    return np.asarray([alive_from(batch.score[i], batch.dist[i],
                                  score_threshold=float(threshold), tau=tau)
                       for i in range(len(batch))], bool)


def merge_anchors(batch, radius_l0: float) -> np.ndarray:
    """Indices to KEEP after merging rows within `radius_l0` level-0 px.

    THE OTHER HALF OF `SurvivalProcess.anchors_of`. The store deduplicates at a
    fixed radius so that the anchor set does not depend on tau; anything tau
    would additionally have merged is merged here, at read time, where redoing
    it at another tau costs milliseconds instead of the GPU.

    Greedy from the strongest: the survivor of a cluster is the row with the
    highest peak score, so a duplicate never displaces the better-measured copy
    of the same location. Order-independent in the only way that matters -- the
    result does not depend on the order rows happen to sit in the file.

    `radius_l0 = 0` returns everything, which is how the sensitivity to this
    number gets checked: if a fraction moves a lot between radius 0 and the
    coarsest tau, the fraction is partly a statement about duplicates.
    """
    n = len(batch)
    if n == 0 or float(radius_l0) <= 0.0:
        return np.arange(n)
    xy = np.stack([batch.x0, batch.y0], axis=1).astype(np.float64)
    strength = batch.score.max(axis=1)
    order = np.argsort(-strength)

    keep: List[int] = []
    kept_xy: List[np.ndarray] = []
    for i in order:
        if kept_xy:
            near = np.linalg.norm(np.asarray(kept_xy) - xy[i], axis=1)
            if float(near.min()) <= float(radius_l0):
                continue
        keep.append(int(i))
        kept_xy.append(xy[i])
    return np.sort(np.asarray(keep, np.int64))


def pattern_table(batch, meta, *, thresholds: Sequence[float],
                  merge_radius_l0: float = 0.0) -> List[dict]:
    """One row per (threshold, pattern), with the null beside every fraction.

    `n_alive_somewhere` is on every row because it is the denominator: a
    fraction of a hundred points and a fraction of a hundred thousand are the
    same number and not the same evidence.
    """
    rows: List[dict] = []
    keep = merge_anchors(batch, merge_radius_l0)
    for threshold in thresholds:
        alive = alive_of(batch, meta, threshold)[keep]
        rate = NullModel.alive_rate_of(alive)
        null = NullModel.null_patterns(rate) if len(rate) else {}
        counted = {name: 0 for name in PATTERNS}
        counted[EMPTY] = 0
        for row in alive:
            counted[classify(row)[0]] += 1
        total = int(sum(counted[name] for name in PATTERNS))

        for name in PATTERNS:
            frac = (counted[name] / total) if total else float('nan')
            rows.append({
                'wsi_stem': meta.wsi_stem, 'stack_kind': meta.stack_kind,
                'threshold': float(threshold), 'pattern': name,
                'n': counted[name], 'frac': frac,
                'null_frac': float(null.get(name, float('nan'))),
                'excess': frac - float(null.get(name, float('nan'))),
                'n_alive_somewhere': total, 'n_rows': int(len(alive)),
                'band_fraction': band_fraction(alive),
                'band_fraction_multi': band_fraction(alive, multi_only=True),
                'n_multi': int((alive.sum(axis=1) > 1).sum()),
                'merge_radius_l0': float(merge_radius_l0)})
    return rows


def pair_axes(f_batch, r_batch, radius_l0: float) -> np.ndarray:
    """`[N_f]` -- for each 'F' row, the index of the 'R' row at that location.

    -1 where the 'R' axis has no row there. THE TWO AXES SHARE A CENTRE BUT NOT
    AN ANCHOR SET: the tiles are co-registered by construction (the 'R' stack is
    derived from the chain's ds 1 tile), but each axis anchors on its own
    detections, so a point present on one and not the other has no counterpart
    and must be reported as -1 rather than matched to whatever is nearest.

    A -1 is not a failure. 新生歸因 asks whether an 'F' birth is also an 'R'
    birth, and "the 'R' axis never detected anything there" is a real answer to
    that -- it is the answer that makes the birth a neighbourhood one.
    """
    if not len(f_batch) or not len(r_batch):
        return np.full(len(f_batch), -1, np.int64)
    f_xy = np.stack([f_batch.x0, f_batch.y0], axis=1).astype(np.float64)
    r_xy = np.stack([r_batch.x0, r_batch.y0], axis=1).astype(np.float64)
    delta = f_xy[:, None, :] - r_xy[None, :, :]
    distance = np.sqrt((delta ** 2).sum(axis=2))
    nearest = distance.argmin(axis=1)
    best = distance[np.arange(len(f_xy)), nearest]
    return np.where(best <= float(radius_l0), nearest, -1)


def attribution_table(f_batch, f_meta, r_batch, r_meta, *,
                      thresholds: Sequence[float],
                      pair_radius_l0: Optional[float] = None,
                      score_rise: float = Attribution.DEFAULT_SCORE_RISE
                      ) -> List[dict]:
    """One row per (threshold, cause). 'F' is the subject, 'R' is the control.

    `pair_radius_l0` defaults to the finest rung's tau, which is the tightest
    the two axes are ever asked to agree to -- and the right one, because both
    axes locate a point to one level-0 pixel at ds 1 and the pairing is about
    identity of location, not about cross-rung tolerance.
    """
    if f_meta.stack_kind != 'F' or r_meta.stack_kind != 'R':
        raise ValueError(
            f"attribution takes ('F', 'R') and got ({f_meta.stack_kind!r}, "
            f"{r_meta.stack_kind!r}). The axes are not interchangeable: 'F' is "
            f'the subject and R is the control that subtracts blur')
    if tuple(f_meta.rungs) != tuple(r_meta.rungs):
        raise ValueError(
            f'{f_meta.rungs} against {r_meta.rungs}; the [N, L] columns of the '
            f'two axes are indexed by the same rung order or the comparison is '
            f'between different magnifications')

    radius = (float(pair_radius_l0) if pair_radius_l0 is not None
              else float(f_meta.tau()[0]))
    partner = pair_axes(f_batch, r_batch, radius)
    dead = np.zeros(len(f_meta.rungs), bool)

    rows: List[dict] = []
    for threshold in thresholds:
        f_alive = alive_of(f_batch, f_meta, threshold)
        r_alive = alive_of(r_batch, r_meta, threshold)
        causes = []
        for i in range(len(f_batch)):
            other = r_alive[partner[i]] if partner[i] >= 0 else dead
            causes.append(Attribution.attribute(
                f_alive[i], f_batch.score[i], other,
                f_batch.suppressed_by_score[i], score_rise=score_rise))
        counted = Attribution.summarise(causes)
        late = int(sum(counted[c] for c in Attribution.CAUSES))

        for name in Attribution.CAUSES:
            rows.append({
                'wsi_stem': f_meta.wsi_stem, 'threshold': float(threshold),
                'cause': name, 'n': counted[name],
                'frac_of_late': (counted[name] / late) if late else float('nan'),
                'n_late': late, 'n_rows': int(len(f_batch)),
                'n_unpaired': int((partner < 0).sum()),
                'pair_radius_l0': radius, 'score_rise': float(score_rise)})
    return rows


def cross_table(f_batch, f_meta, r_batch, r_meta, *, threshold: float,
                score_rise: float = Attribution.DEFAULT_SCORE_RISE
                ) -> List[dict]:
    """樣態 x 歸因. ONE threshold, because this is a two-way table already.

    The cell this exists for is `只在一階` x `鄰域新生`: a point that appears at
    exactly one rung AND appears because the receptive field widened is the
    strongest available candidate for scale information, and neither table alone
    identifies it (plan.md P1, 分析三).
    """
    radius = float(f_meta.tau()[0])
    partner = pair_axes(f_batch, r_batch, radius)
    f_alive = alive_of(f_batch, f_meta, threshold)
    r_alive = alive_of(r_batch, r_meta, threshold)
    dead = np.zeros(len(f_meta.rungs), bool)

    grid: Dict[Tuple[str, str], int] = {}
    for i in range(len(f_batch)):
        other = r_alive[partner[i]] if partner[i] >= 0 else dead
        cause = Attribution.attribute(
            f_alive[i], f_batch.score[i], other,
            f_batch.suppressed_by_score[i], score_rise=score_rise)
        pattern = classify(f_alive[i])[0]
        grid[(pattern, cause)] = grid.get((pattern, cause), 0) + 1

    total = sum(grid.values())
    causes = list(Attribution.CAUSES) + [Attribution.NOT_LATE]
    return [{'wsi_stem': f_meta.wsi_stem, 'threshold': float(threshold),
             'pattern': pattern, 'cause': cause,
             'n': grid.get((pattern, cause), 0),
             'frac': (grid.get((pattern, cause), 0) / total) if total else
                     float('nan')}
            for pattern in list(PATTERNS) + [EMPTY] for cause in causes]


def tau_curve(batch, meta, *, alphas: Sequence[float],
              threshold: float) -> List[dict]:
    """Match rate against tau, per rung, against a probed decoy. THE CALIBRATION.

    `alpha` scales tau as `alpha * ds` level-0 px. What run one produces is this
    curve and nothing else: the pattern and attribution tables are downstream of
    tau, so quoting them before tau is chosen is quoting the default (spec.md
    3.2, ClaudeRules 8).

    THE DECOY IS A SHIFT AND NOT A NULL MODEL, and that is the one place in this
    module where a decoy is still the right instrument: whether a displaced set
    matches anyway has no closed form -- it depends on how the points are laid
    out, which is exactly the question.

    THE DECOY IS A SECOND PROBE, STORED AT BUILD TIME (`decoy_score`,
    `decoy_dist`), NOT A SHIFT APPLIED TO `dist` HERE. It was the latter for one
    day and that was not a decoy at all: `dist + shift <= tau` is exactly the
    match rate at `alpha - shift/ds`, so the "gap" was a finite difference of
    one curve with itself and read `margin 1.1` at every rung. A decoy has to be
    the same measurement somewhere the answer should be NO, which means probing
    a different place -- and that needs the images, so it happens where the
    images are.

    EVERY TAU IS ANSWERABLE. `dist` is the distance to the nearest detection,
    which has no window in it, so the sweep is bounded by nothing the build
    chose. It was not always so -- see `SurvivalProcess.run`.

    Read the KNEE, not the peak: the match rate rises with tau until tau is
    wide enough to catch anything, and the useful value is where it stops
    rising faster than the decoy.
    """
    rungs = np.asarray(meta.rungs, np.float64)
    floor = (meta.tau_floor_um / meta.mpp_0) if meta.mpp_0 else 0.0
    passes = batch.score > float(threshold)
    decoy_passes = batch.decoy_score > float(threshold)

    rows: List[dict] = []
    for alpha in alphas:
        tau = np.maximum(floor, float(alpha) * rungs)
        real = passes & (batch.dist >= 0.0) & (batch.dist <= tau)
        # The decoy asks the same question at a place the point is not.
        decoy = (decoy_passes & (batch.decoy_dist >= 0.0)
                 & (batch.decoy_dist <= tau))
        for j, ds in enumerate(meta.rungs):
            rows.append({
                'wsi_stem': meta.wsi_stem, 'stack_kind': meta.stack_kind,
                'alpha': float(alpha), 'ds': float(ds),
                'tau_l0': float(tau[j]), 'threshold': float(threshold),
                'match_rate': float(real[:, j].mean()) if len(batch) else
                              float('nan'),
                'decoy_rate': float(decoy[:, j].mean()) if len(batch) else
                              float('nan'),
                'decoy_l0': float(meta.decoy_alpha * float(ds)),
                'n_rows': int(len(batch))})
    return rows


def offset_quantiles(batch, meta, *, quantiles: Sequence[float] = (
        0.5, 0.9, 0.99)) -> List[dict]:
    """Where the probed peak actually sits, per rung. The other half of tau.

    tau has to be at least as wide as the offsets a real point produces, and
    this is that distribution. A tau chosen below the 90th percentile is
    declaring a tenth of the real matches dead by construction.
    """
    rows = []
    for j, ds in enumerate(meta.rungs):
        column = batch.dist[:, j]
        valid = column[column >= 0.0]
        for q in quantiles:
            rows.append({
                'wsi_stem': meta.wsi_stem, 'stack_kind': meta.stack_kind,
                'ds': float(ds), 'quantile': float(q),
                'offset_l0': float(np.quantile(valid, q)) if len(valid) else
                             float('nan'),
                'n_valid': int(len(valid))})
    return rows
