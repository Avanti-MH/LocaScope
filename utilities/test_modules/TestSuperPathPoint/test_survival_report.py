#!/usr/bin/env python3
"""Tests for Stage B's reporting arithmetic. plan.md P1 ①②⑤.

    python utilities/test_modules/TestSuperPathPoint/test_survival_report.py

WHAT THIS BLOCKS
==================
`Report` and `NullModel` turn a table into the three numbers that decide Stage
C's design (plan.md P1): the band fraction, the late-born fraction, and the
one-rung-only fraction. Every one of them is a ratio, and a ratio is the shape
that fails silently -- a denominator counted over the wrong set, a null that
conditions on something the measurement does not, a merge that drops rows.
Nothing raises. The number just means something else than it says.

So every test here is either a hand-computable value or a decoy that must NOT
be produced.

Sections:
  null      the closed form against hand arithmetic, and its conditioning
  merge     read-time merging, its monotonicity, and the radius-0 identity
  patterns  the table's denominators, and the sweep it forces
  pair      the two axes joined, including the -1 that is a real answer
  tau       the calibration curve and its decoy
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                        # noqa: E402

setup_import_paths()

import numpy as np                                           # noqa: E402

from PointsAnalysisByMpp import NullModel, Report            # noqa: E402
from PointsAnalysisByMpp.Patterns import EMPTY, PATTERNS     # noqa: E402
from PointsAnalysisByMpp.SurvivalTable import (SurvivalBatch,  # noqa: E402
                                               SurvivalMeta)

_RESULTS = []
_RUNGS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                   # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def _meta(kind='F', alpha=1.5):
    return SurvivalMeta(wsi_stem='S1103627,G7E,110127', stack_kind=kind,
                        tile=256, rungs=_RUNGS, tau_alpha=alpha)


def _batch(score, dist=None, rival=None, xy=None, decoy=None):
    """A table from hand-written `[N, L]` arrays. `dist=None` means 0 -- every
    point sits exactly where it was probed, so tau never decides anything and
    a test about the score cut is only about the score cut."""
    score = np.asarray(score, np.float32)
    n, length = score.shape
    xy = (np.asarray(xy, np.float32) if xy is not None
          else np.stack([np.arange(n) * 1000.0, np.zeros(n)], axis=1))
    return SurvivalBatch(
        x0=xy[:, 0].astype(np.float32), y0=xy[:, 1].astype(np.float32),
        chain=np.zeros(n, np.int32), score=score,
        dist=(np.asarray(dist, np.float32) if dist is not None
              else np.zeros((n, length), np.float32)),
        suppressed_by_score=(np.asarray(rival, np.float32) if rival is not None
                             else np.zeros((n, length), np.float32)),
        suppressed_by_dist=np.zeros((n, length), np.float32),
        # The decoy defaults to "nothing found there" -- score 0, offset -1 --
        # so a test about the real match is only about the real match.
        decoy_score=np.zeros((n, length), np.float32),
        decoy_dist=(np.asarray(decoy, np.float32) if decoy is not None
                    else np.full((n, length), -1.0, np.float32)))


# ── 1. the null model ────────────────────────────────────────────────────────

def t_the_null_enumerates_every_vector_and_sums_to_one():
    out = NullModel.null_patterns([0.3, 0.6, 0.5, 0.2, 0.9, 0.4])
    total = sum(out[name] for name in PATTERNS)
    if abs(total - 1.0) > 1e-9:
        raise AssertionError(
            f'the six patterns sum to {total}, not 1. A vector fell through '
            f'the classifier into no class, and every fraction below is then '
            f'over a denominator nobody named')
    if out[EMPTY] != 0.0:
        raise AssertionError(
            'the all-dead vector has non-zero mass. The table cannot contain '
            'it -- a row exists because the point was detected somewhere -- so '
            'leaving it in the denominator deflates every fraction')
    return f'{2 ** 6} vectors, six classes, sum 1.0'


def t_always_alive_is_p_to_the_L_conditioned_on_being_alive_at_all():
    """The one value that can be written down without the enumeration.

    All rungs at rate p: `一直存活` is `p**L`, and the null is conditioned on
    being alive somewhere, so the denominator is `1 - (1-p)**L`. Getting the
    conditioning wrong is invisible at p = 0.5 on a short ladder and grows with
    sparsity, which is the direction this corpus actually goes.
    """
    p, length = 0.5, len(_RUNGS)
    out = NullModel.null_patterns([p] * length)
    want = p ** length / (1.0 - (1.0 - p) ** length)
    if abs(out['一直存活'] - want) > 1e-12:
        raise AssertionError(
            f"一直存活 is {out['一直存活']:.6f}, hand arithmetic says "
            f'{want:.6f}. Either the class is wrong or the conditioning is')
    # The decoy: the UNconditioned value, which a null that forgot to divide
    # would produce. It must not be what came back.
    if abs(out['一直存活'] - p ** length) < 1e-12:
        raise AssertionError(
            'the value equals the unconditioned p**L, so the all-dead vector '
            'is still in the denominator')
    return f'{want:.6f} conditioned, {p ** length:.6f} unconditioned'


def t_a_ladder_where_everything_survives_is_all_one_pattern():
    out = NullModel.null_patterns([1.0] * len(_RUNGS))
    if abs(out['一直存活'] - 1.0) > 1e-12:
        raise AssertionError(f'{out}; every rung alive is one pattern')


def t_alive_rate_is_over_every_row_and_not_the_surviving_subset():
    """The rate feeds the null, so conditioning it would double-count.

    A rate computed over "alive somewhere" is inflated by exactly the selection
    the null is then asked to model, and the excess over null would come back
    negative on a corpus with no structure at all.
    """
    rows = np.array([[True, False], [False, False]])
    got = NullModel.alive_rate_of(rows)
    if not np.allclose(got, [0.5, 0.0]):
        raise AssertionError(
            f'{got}; expected [0.5, 0.0] over BOTH rows, including the dead one')


def t_the_null_refuses_a_rate_outside_zero_to_one():
    try:
        NullModel.null_patterns([0.5, 1.2])
    except ValueError:
        return 'refused'
    raise AssertionError('a rate of 1.2 was accepted as a probability')


# ── 2. merging ───────────────────────────────────────────────────────────────

def t_radius_zero_keeps_everything():
    """The identity that makes the sensitivity check possible.

    Radius 0 is how "how much of this fraction is duplicates" gets answered: run
    the table at 0 and at the coarsest tau and compare. If radius 0 quietly
    merged anything, the comparison would have no baseline.
    """
    batch = _batch(np.full((5, 6), 0.5), xy=np.zeros((5, 2)))
    keep = Report.merge_anchors(batch, 0.0)
    if len(keep) != 5:
        raise AssertionError(
            f'radius 0 kept {len(keep)} of 5 rows that are all at the origin')


def t_merging_is_monotone_in_the_radius():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, (60, 2))
    batch = _batch(rng.uniform(0, 1, (60, 6)), xy=xy)
    counts = [len(Report.merge_anchors(batch, r)) for r in (0, 2, 5, 10, 40)]
    if counts != sorted(counts, reverse=True):
        raise AssertionError(
            f'{counts} is not non-increasing in the radius; a wider merge kept '
            f'more rows than a narrower one')
    if counts[-1] >= counts[0]:
        raise AssertionError(
            f'{counts}: a radius of 40 on points spread over 100 px merged '
            f'nothing, so this fixture cannot tell merging from not merging')
    return f'kept {counts} at radius 0/2/5/10/40'


def t_the_survivor_of_a_cluster_is_the_strongest_row():
    """A duplicate must never displace the better-measured copy.

    Three rows at the same place with different peak scores: the one that
    survives has to be the strongest, or the table reports the weaker
    measurement of a location it measured twice.
    """
    score = np.zeros((3, 6), np.float32)
    score[0, 0], score[1, 0], score[2, 0] = 0.1, 0.9, 0.4
    batch = _batch(score, xy=np.zeros((3, 2)))
    keep = Report.merge_anchors(batch, 5.0)
    if list(keep) != [1]:
        raise AssertionError(
            f'kept {list(keep)}, expected [1] -- the row whose peak is 0.9')


# ── 3. the pattern table ─────────────────────────────────────────────────────

def t_the_pattern_counts_sum_to_the_alive_somewhere_denominator():
    rng = np.random.default_rng(1)
    batch = _batch(rng.uniform(0, 1, (200, 6)))
    rows = Report.pattern_table(batch, _meta(), thresholds=(0.5,))
    total = sum(r['n'] for r in rows)
    named = rows[0]['n_alive_somewhere']
    if total != named:
        raise AssertionError(
            f'the six counts sum to {total} while the row says the denominator '
            f'is {named}. Points went missing between the classifier and the '
            f'table, and every fraction is over the wrong base')
    return f'{total} points over six classes'


def t_the_sweep_produces_one_block_per_threshold():
    batch = _batch(np.full((10, 6), 0.5))
    rows = Report.pattern_table(batch, _meta(), thresholds=(0.1, 0.4, 0.9))
    if len({r['threshold'] for r in rows}) != 3:
        raise AssertionError('the sweep collapsed to fewer than three blocks')
    high = [r for r in rows if r['threshold'] == 0.9]
    if sum(r['n'] for r in high) != 0:
        raise AssertionError(
            'a threshold above every score still counted points alive')
    low = [r for r in rows if r['threshold'] == 0.1]
    everywhere = [r for r in low if r['pattern'] == '一直存活'][0]
    if everywhere['n'] != 10:
        raise AssertionError(
            f'a threshold below every score put {everywhere["n"]} of 10 points '
            f'in 一直存活')
    return 'threshold 0.1 -> 10 alive everywhere, 0.9 -> 0 alive'


def t_tau_kills_a_match_that_the_score_would_have_kept():
    """Both cuts reach the table, not just the one being swept.

    A point with a strong score whose peak is 400 level-0 px away is not the
    same point. If only the score cut reached the table, tau would be
    decorative and the coarse rungs would report matches from the next feature
    over.
    """
    score = np.full((1, 6), 0.9, np.float32)
    dist = np.zeros((1, 6), np.float32)
    dist[0, 5] = 400.0                      # tau at ds 32 is 1.5 * 32 = 48
    rows = Report.pattern_table(_batch(score, dist=dist), _meta(),
                                thresholds=(0.5,))
    got = [r['pattern'] for r in rows if r['n'] == 1]
    if got != ['細部存活']:
        raise AssertionError(
            f'{got}; the ds 32 rung should be dead on distance, leaving a band '
            f'that stops short of the coarsest rung')
    return 'a 400 px offset at ds 32 dies against tau 48'


# ── 4. pairing the axes ──────────────────────────────────────────────────────

def t_pairing_finds_the_nearest_and_reports_minus_one_beyond_the_radius():
    f = _batch(np.zeros((2, 6)), xy=[[0.0, 0.0], [500.0, 0.0]])
    r = _batch(np.zeros((1, 6)), xy=[[1.0, 0.0]])
    got = Report.pair_axes(f, r, 5.0)
    if list(got) != [0, -1]:
        raise AssertionError(
            f'{list(got)}, expected [0, -1]. The far F row has no R counterpart '
            f'and -1 is the answer -- "the R axis never detected anything '
            f'there" is what makes a birth a neighbourhood one, not a failure '
            f'to be papered over by matching the nearest thing')
    return 'nearest within 5 px, -1 at 500'


def t_attribution_refuses_the_axes_the_wrong_way_round():
    batch = _batch(np.zeros((1, 6)))
    try:
        Report.attribution_table(batch, _meta('R'), batch, _meta('F'),
                                 thresholds=(0.5,))
    except ValueError:
        return 'refused'
    raise AssertionError(
        "('R', 'F') was accepted; the axes are not interchangeable -- F is the "
        'subject and R is the control that subtracts blur')


def t_attribution_refuses_two_axes_on_different_rungs():
    batch = _batch(np.zeros((1, 6)))
    other = SurvivalMeta(wsi_stem='A', stack_kind='R', tile=256,
                         rungs=(1.0, 2.0, 4.0, 8.0, 16.0, 64.0))
    try:
        Report.attribution_table(batch, _meta('F'), batch, other,
                                 thresholds=(0.5,))
    except ValueError:
        return 'refused'
    raise AssertionError(
        'two axes indexed by different rung tuples were compared; every column '
        'would be a different magnification on each side')


# ── 5. the calibration curve ─────────────────────────────────────────────────

def t_the_match_rate_rises_with_alpha_and_the_decoy_lags():
    """The curve the whole first run exists to produce.

    Points sit 3 level-0 px from where they were probed at every rung, and the
    decoy probe found NOTHING (score 0, offset -1) -- so the decoy rate is 0 at
    every alpha and the gap is the match rate.

    THE DECOY IS A STORED SECOND PROBE, not `dist` plus a constant. The
    difference is the whole test: `dist + shift <= tau` is the match rate at a
    shifted alpha, so a "decoy" built that way tracks the real curve by
    construction and reads margin 1.1 everywhere (2026-09-01).
    """
    dist = np.full((20, 6), 3.0, np.float32)
    batch = _batch(np.full((20, 6), 0.9), dist=dist)
    rows = Report.tau_curve(batch, _meta(), alphas=(0.5, 2.0, 8.0),
                            threshold=0.5)
    at = {(r['alpha'], r['ds']): r for r in rows}
    if at[(0.5, 1.0)]['match_rate'] != 0.0:
        raise AssertionError('tau 0.5 matched a point 3 px away')
    if at[(8.0, 1.0)]['match_rate'] != 1.0:
        raise AssertionError('tau 8 failed to match a point 3 px away')
    for row in rows:
        if row['decoy_rate'] > row['match_rate']:
            raise AssertionError(
                f'the decoy beat the real rate at alpha {row["alpha"]}, '
                f'ds {row["ds"]}: {row["decoy_rate"]} > {row["match_rate"]}')
    return 'ds 1: 0.0 at alpha 0.5, 1.0 at alpha 8, decoy never ahead'


def t_the_nearest_detection_has_no_window_in_it():
    """`dist` must depend on the point set and on nothing the build chose.

    THE FAILURE THIS BLOCKS HAPPENED TWICE IN ONE DAY, in both directions.
    `dist` used to be the offset of the argmax over a window of radius r:

        r bound to tau     -> dist <= tau, so the curve went flat AT the window
                              and read as the data saturating
        r opened up to 64  -> the argmax over a wide window is a stronger peak
                              FURTHER away, so the ds 1 match rate fell from
                              0.090 to 0.005

    Neither raised. Distance to the nearest detection has no radius in it, and
    this is the assertion that it stays that way: the same points, queried from
    the same place, must give the same distance whatever else is in the tile.
    """
    from PointsAnalysisByMpp.SurvivalProcess import nearest_detection

    points = np.array([[10.0, 0.0], [200.0, 0.0]])
    score = np.array([0.2, 0.9])
    query = np.array([[0.0, 0.0]])

    dist, sc = nearest_detection(points, score, query)
    if abs(dist[0] - 10.0) > 1e-9:
        raise AssertionError(
            f'{dist[0]}: the nearest point is at 10, not the strongest at 200. '
            f'An argmax over a window would have returned the second')
    if abs(sc[0] - 0.2) > 1e-9:
        raise AssertionError(
            f'{sc[0]}: the score reported must be the NEAREST detection\'s, '
            f'not the strongest in the neighbourhood')

    # A far-away stronger point must not move the answer at all -- that is what
    # "no window" means.
    more = np.array([[10.0, 0.0], [200.0, 0.0], [1000.0, 0.0]])
    again, _ = nearest_detection(more, np.array([0.2, 0.9, 1.0]), query)
    if abs(again[0] - dist[0]) > 1e-9:
        raise AssertionError(
            'adding a detection 1000 px away changed the answer; something '
            'about this is still windowed')

    empty, empty_score = nearest_detection(np.zeros((0, 2)), np.zeros(0), query)
    if empty[0] >= 0:
        raise AssertionError(
            f'{empty[0]}: a rung that detected nothing must report the -1 '
            f'sentinel, not a distance')
    return 'nearest is nearest, and a point 1000 px away changes nothing'


def t_the_decoy_is_a_second_probe_and_not_the_same_curve_shifted():
    """The decoy must be able to differ from the match at EVERY alpha.

    THE FAILURE THIS BLOCKS PRODUCED A NUMBER, NOT AN ERROR. When the decoy was
    derived as `dist + shift <= tau` it was exactly the match rate at
    `alpha - shift/ds` -- one curve compared with itself -- so the gap was a
    finite difference and `margin` sat at 1.1 for every rung and every axis
    (2026-09-01). Nothing in the output said the control was not a control.

    Here the decoy probe found something very close (offset 1) while the real
    probe found something far (offset 40). A derived decoy CANNOT produce that
    ordering: it is the real distance plus a positive constant, so it is always
    the harder test. A probed one can, because it is a different measurement.
    """
    dist = np.full((10, 6), 40.0, np.float32)
    decoy = np.full((10, 6), 1.0, np.float32)
    batch = _batch(np.full((10, 6), 0.9), dist=dist, decoy=decoy)
    batch.decoy_score = np.full((10, 6), 0.9, np.float32)

    rows = Report.tau_curve(batch, _meta(), alphas=(2.0,), threshold=0.5)
    at_ds1 = [r for r in rows if r['ds'] == 1.0][0]
    if at_ds1['match_rate'] != 0.0:
        raise AssertionError(
            f"match {at_ds1['match_rate']}: an offset of 40 must not match a "
            f'tau of 2')
    if at_ds1['decoy_rate'] != 1.0:
        raise AssertionError(
            f"decoy {at_ds1['decoy_rate']}: the decoy probe found something at "
            f'offset 1, which is inside a tau of 2. A decoy derived from '
            f'`dist + shift` could never exceed the match here, and that it '
            f'can is the whole difference')
    return 'the decoy beat the match, which a derived one cannot do'


def t_offset_quantiles_ignore_the_unmatched_sentinel():
    dist = np.full((4, 6), -1.0, np.float32)
    dist[:2, 0] = [2.0, 4.0]
    rows = Report.offset_quantiles(_batch(np.zeros((4, 6)), dist=dist),
                                   _meta(), quantiles=(0.5,))
    first = [r for r in rows if r['ds'] == 1.0][0]
    if first['n_valid'] != 2:
        raise AssertionError(
            f'{first["n_valid"]} valid offsets at ds 1, expected 2; the -1 '
            f'sentinel was counted as an offset of -1 px')
    if not (2.0 <= first['offset_l0'] <= 4.0):
        raise AssertionError(f'median {first["offset_l0"]} outside [2, 4]')
    return 'the -1 sentinel stays out of the quantiles'


_SECTIONS = {
    'null':     ['t_the_null_enumerates_every_vector_and_sums_to_one',
                 't_always_alive_is_p_to_the_L_conditioned_on_being_alive_at_all',
                 't_a_ladder_where_everything_survives_is_all_one_pattern',
                 't_alive_rate_is_over_every_row_and_not_the_surviving_subset',
                 't_the_null_refuses_a_rate_outside_zero_to_one'],
    'merge':    ['t_radius_zero_keeps_everything',
                 't_merging_is_monotone_in_the_radius',
                 't_the_survivor_of_a_cluster_is_the_strongest_row'],
    'patterns': ['t_the_pattern_counts_sum_to_the_alive_somewhere_denominator',
                 't_the_sweep_produces_one_block_per_threshold',
                 't_tau_kills_a_match_that_the_score_would_have_kept'],
    'pair':     ['t_pairing_finds_the_nearest_and_reports_minus_one_beyond_the_radius',
                 't_attribution_refuses_the_axes_the_wrong_way_round',
                 't_attribution_refuses_two_axes_on_different_rungs'],
    'tau':      ['t_the_nearest_detection_has_no_window_in_it',
                 't_the_match_rate_rises_with_alpha_and_the_decoy_lags',
                 't_the_decoy_is_a_second_probe_and_not_the_same_curve_shifted',
                 't_offset_quantiles_ignore_the_unmatched_sentinel'],
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(_SECTIONS))
    args = ap.parse_args()

    for section in (args.only or list(_SECTIONS)):
        print(f'\n[{section}]')
        for name in _SECTIONS[section]:
            check(name[2:].replace('_', ' '), globals()[name])

    failed = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
