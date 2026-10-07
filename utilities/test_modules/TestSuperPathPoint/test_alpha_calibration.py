#!/usr/bin/env python3
"""Tests for `SurvivalAnalysis/AlphaCalibration.py`. plan.md 2.2①.

    python utilities/test_modules/TestSuperPathPoint/test_alpha_calibration.py

PURE, NO GPU, NO NET -- every function here takes arrays a human can type by
hand. That is the whole point of splitting alpha calibration out of
`SurvivalProcess.py`: this file runs in seconds and is checked before any
real ChainStack ever reaches it (ClaudeRules 8).
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                           # noqa: E402

setup_import_paths('SuperPathPoint')

import numpy as np                                            # noqa: E402

from SurvivalAnalysis import AlphaCalibration as AC            # noqa: E402
from SurvivalAnalysis.Attribution import NONE                   # noqa: E402

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                   # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ── 1. merge_anchors ─────────────────────────────────────────────────────────

def t_merge_anchors_is_the_survivalprocess_primitive():
    """Thin wrapper, not a second definition -- radius 0 keeps everything,
    matching `SurvivalProcess._merge_within_radius`'s own contract."""
    anchors = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    keep = AC.merge_anchors(anchors, radius=0.0)
    if list(keep) != [0, 1, 2]:
        raise AssertionError(f'radius 0 dropped points: kept {list(keep)}')
    return 'radius 0 is a no-op, as documented'


# ── 2. probe_decoy ───────────────────────────────────────────────────────────

def t_probe_decoy_single_callable_used_at_every_rung():
    """One `decoy_shift` (not a per-rung dict) applies the SAME shift at
    every rung."""
    anchors = np.array([[0.0, 0.0]])
    per_rung = {1.0: (np.array([[5.0, 0.0]]), np.array([0.9])),
               2.0: (np.array([[0.0, 5.0]]), np.array([0.9]))}

    def shift(a):
        return a + np.array([5.0, 0.0])

    decoy_dist, decoy_score = AC.probe_decoy(anchors, per_rung, [1.0, 2.0], shift)
    # shifted anchor is (5, 0): exactly the ds-1 rung's own detection (dist 0)
    if not np.isclose(decoy_dist[0, 0], 0.0):
        raise AssertionError(f'ds 1 decoy_dist {decoy_dist[0, 0]} != 0')
    # ... and sqrt(5^2+5^2) from the ds-2 rung's detection at (0, 5)
    if not np.isclose(decoy_dist[0, 1], np.sqrt(50)):
        raise AssertionError(f'ds 2 decoy_dist {decoy_dist[0, 1]} != sqrt(50)')
    return f'decoy_dist {decoy_dist.tolist()}'


def t_probe_decoy_per_rung_dict_uses_a_different_shift_each_rung():
    anchors = np.array([[0.0, 0.0]])
    per_rung_det = {1.0: (np.array([[5.0, 0.0]]), np.array([0.9])),
                   2.0: (np.array([[5.0, 0.0]]), np.array([0.9]))}
    shifts = {1.0: lambda a: a + np.array([5.0, 0.0]),      # lands exactly on it
             2.0: lambda a: a + np.array([0.0, 0.0])}       # does not move at all
    decoy_dist, _ = AC.probe_decoy(anchors, per_rung_det, [1.0, 2.0], shifts)
    if not np.isclose(decoy_dist[0, 0], 0.0):
        raise AssertionError(f'ds 1 (shifted onto the point) dist '
                             f'{decoy_dist[0, 0]} != 0')
    if not np.isclose(decoy_dist[0, 1], 5.0):
        raise AssertionError(f'ds 2 (not shifted) dist {decoy_dist[0, 1]} != 5')
    return f'decoy_dist {decoy_dist.tolist()} -- two different shifts, two answers'


def t_probe_decoy_empty_anchors_is_empty_not_a_crash():
    decoy_dist, decoy_score = AC.probe_decoy(
        np.zeros((0, 2)), {1.0: (np.zeros((0, 2)), np.zeros(0))}, [1.0],
        lambda a: a)
    if decoy_dist.shape != (0, 1):
        raise AssertionError(f'expected shape (0, 1), got {decoy_dist.shape}')
    return 'no anchors -> empty arrays, no exception'


# ── 3. alpha_curve ───────────────────────────────────────────────────────────

def t_alpha_curve_match_rate_crosses_at_the_right_alpha():
    """One anchor, one rung (ds=2), dist=3. tau=alpha*2, so match starts the
    moment alpha >= 1.5 (tau=3) -- hand-computable, not measured off a plot."""
    dist = np.array([[3.0]])
    score = np.array([[0.9]])
    decoy_dist = np.full((1, 1), NONE)
    decoy_score = np.zeros((1, 1))
    curve = AC.alpha_curve(dist, score, decoy_dist, decoy_score, rungs=[2.0],
                          alphas=[1.0, 1.5, 2.0], tau_floor=0.0, threshold=0.5)
    match = curve['match_rate'][0]
    if not np.allclose(match, [0.0, 1.0, 1.0]):
        raise AssertionError(f'match_rate {match} != [0, 1, 1] -- tau crosses '
                             f'3 exactly at alpha 1.5 (tau=alpha*2)')
    return f'match_rate {match.tolist()} at alphas [1.0, 1.5, 2.0]'


def t_alpha_curve_gap_is_signed_not_absolute():
    """decoy_rate > match_rate at some alpha must leave `gap` NEGATIVE --
    `abs()` anywhere in the pipeline would hide exactly this."""
    dist = np.array([[5.0]])           # never matches within the alphas tried
    score = np.array([[0.9]])
    decoy_dist = np.array([[0.5]])     # decoy matches immediately
    decoy_score = np.array([[0.9]])
    curve = AC.alpha_curve(dist, score, decoy_dist, decoy_score, rungs=[1.0],
                          alphas=[1.0], tau_floor=0.0, threshold=0.5)
    gap = curve['gap'][0, 0]
    if not gap < 0:
        raise AssertionError(f'gap {gap} should be negative (decoy beat match)')
    return f'gap {gap:.3f} -- negative and untouched by abs()'


def t_alpha_curve_threshold_gates_independently_of_distance():
    """score below threshold kills the match even at dist=0."""
    dist = np.array([[0.0]])
    score = np.array([[0.1]])           # below the 0.5 threshold
    decoy_dist = np.full((1, 1), NONE)
    decoy_score = np.zeros((1, 1))
    curve = AC.alpha_curve(dist, score, decoy_dist, decoy_score, rungs=[1.0],
                          alphas=[10.0], tau_floor=0.0, threshold=0.5)
    if curve['match_rate'][0, 0] != 0.0:
        raise AssertionError('a low score should not count as a match even '
                             'at dist 0 and a huge alpha')
    return 'low score gates the match regardless of distance'


def t_alpha_curve_tau_floor_wins_over_a_tiny_alpha():
    dist = np.array([[3.0]])
    score = np.array([[0.9]])
    decoy_dist = np.full((1, 1), NONE)
    decoy_score = np.zeros((1, 1))
    curve = AC.alpha_curve(dist, score, decoy_dist, decoy_score, rungs=[1.0],
                          alphas=[0.001], tau_floor=5.0, threshold=0.5)
    if curve['match_rate'][0, 0] != 1.0:
        raise AssertionError('tau_floor=5 should have admitted dist=3 even '
                             'though alpha*ds=0.001 alone would not')
    return 'tau_floor overrides an alpha too small to matter on its own'


# ── 4. aggregate_curves ──────────────────────────────────────────────────────

def t_aggregate_curves_margin_uses_log_space_not_linear():
    """Two ChainStacks, same match_rate, wildly different decoy_rate (one
    near 0). A LINEAR mean of the resulting margins would be dominated by
    the near-0 decoy's huge ratio; log-space aggregation should not be."""
    def _curve(decoy_rate):
        return {'match_rate': np.array([[0.5]]), 'decoy_rate': np.array([[decoy_rate]]),
               'gap': np.array([[0.5 - decoy_rate]]),
               'margin': np.array([[0.5 / max(decoy_rate, 1e-9)]]),
               'rungs': np.array([1.0]), 'alphas': np.array([1.0])}

    curves = [_curve(0.5), _curve(1e-6)]     # margins: 1.0 and 500,000
    bucket = AC.aggregate_curves(curves)
    linear_mean = np.mean([1.0, 0.5 / 1e-6])
    if bucket['margin_mean'][0, 0] >= linear_mean / 2:
        raise AssertionError(f"margin_mean {bucket['margin_mean'][0, 0]:.1f} "
                             f"is too close to the linear mean {linear_mean:.1f} "
                             f"-- log-space aggregation should have tamed it")
    # geometric mean of 1.0 and 500,000 is sqrt(500,000) ~= 707
    if not np.isclose(bucket['margin_mean'][0, 0], np.sqrt(0.5 / 1e-6), rtol=0.05):
        raise AssertionError(f"margin_mean {bucket['margin_mean'][0, 0]:.1f} "
                             f"!= geometric mean ~{np.sqrt(0.5 / 1e-6):.1f}")
    return f"margin_mean {bucket['margin_mean'][0, 0]:.1f} (geometric, not " \
          f"linear {linear_mean:.1f})"


def t_aggregate_curves_match_rate_is_a_plain_linear_mean():
    curves = [{'match_rate': np.array([[0.2]]), 'decoy_rate': np.array([[0.1]]),
              'gap': np.array([[0.1]]), 'margin': np.array([[2.0]]),
              'rungs': np.array([1.0]), 'alphas': np.array([1.0])},
             {'match_rate': np.array([[0.8]]), 'decoy_rate': np.array([[0.1]]),
              'gap': np.array([[0.7]]), 'margin': np.array([[8.0]]),
              'rungs': np.array([1.0]), 'alphas': np.array([1.0])}]
    bucket = AC.aggregate_curves(curves)
    if not np.isclose(bucket['match_rate_mean'][0, 0], 0.5):
        raise AssertionError(f"match_rate_mean {bucket['match_rate_mean'][0, 0]} "
                             f"!= 0.5")
    if bucket['n_chainstacks'] != 2:
        raise AssertionError(f"n_chainstacks {bucket['n_chainstacks']} != 2")
    return f"match_rate_mean {bucket['match_rate_mean'][0, 0]}, " \
          f"n_chainstacks {bucket['n_chainstacks']}"


# ── 5. offset_quantiles ──────────────────────────────────────────────────────

def t_offset_quantiles_of_matches_numpy_on_filtered_data():
    """`dist < 0` (NONE) rows are excluded before the quantile, not counted
    as offset 0 or -1 -- either would understate the real spread."""
    dist = np.array([[0.0], [2.0], [4.0], [NONE], [8.0]])
    out = AC.offset_quantiles_of(dist, rungs=[1.0], quantiles=[0.5])
    expected = np.quantile([0.0, 2.0, 4.0, 8.0], 0.5)
    if not np.isclose(out[0, 0], expected):
        raise AssertionError(f'{out[0, 0]} != {expected} -- the NONE row '
                             f'should have been excluded, not counted as 0')
    return f'median {out[0, 0]} over the 4 valid rows, NONE row excluded'


def t_offset_quantiles_of_excludes_the_self_match_column():
    """`source_rung[i] == rungs[j]` means anchor i's own coordinates were
    copied from rung j (AlphaSelectionNotes.md §9) -- probing rung j with
    it always finds itself at distance 0, which is not a real cross-rung
    offset. Two anchors born at ds=1 (rows 0/1, dist=0.0 at column ds=1 by
    construction) and one born at ds=2 (row 2, a real 6.0 offset measured
    back at ds=1) -- the ds=1 column must drop rows 0/1 and keep only the
    real 6.0, and the ds=2 column must drop row 2, its own 0.0, and keep
    rows 0/1's real 3.0 and 5.0.
    """
    dist = np.array([[0.0, 3.0],
                     [0.0, 5.0],
                     [6.0, 0.0]])
    source_rung = np.array([1.0, 1.0, 2.0])
    out = AC.offset_quantiles_of(dist, rungs=[1.0, 2.0], source_rung=source_rung,
                                 quantiles=[0.5])
    if not np.isclose(out[0, 0], 6.0):
        raise AssertionError(f'ds=1 column: {out[0, 0]} != 6.0 -- the two '
                             f'self-matched rows (born at ds=1) should have '
                             f'been excluded, leaving only the real offset')
    expected_ds2 = np.quantile([3.0, 5.0], 0.5)
    if not np.isclose(out[1, 0], expected_ds2):
        raise AssertionError(f'ds=2 column: {out[1, 0]} != {expected_ds2} -- '
                             f'row 2 was born at ds=2, so its own 0.0 there '
                             f'should have been excluded')
    return f'ds=1 median {out[0, 0]} (self-matches excluded), ds=2 median {out[1, 0]}'


def t_offset_quantiles_of_without_source_rung_keeps_old_contaminated_behaviour():
    """`source_rung=None` (the default) keeps every row, self-matches
    included."""
    dist = np.array([[0.0], [0.0], [6.0]])
    out = AC.offset_quantiles_of(dist, rungs=[1.0], quantiles=[0.5])
    expected = np.quantile([0.0, 0.0, 6.0], 0.5)
    if not np.isclose(out[0, 0], expected):
        raise AssertionError(f'{out[0, 0]} != {expected} -- source_rung=None '
                             f'should not exclude anything')
    return f'median {out[0, 0]}, no exclusion applied'


def t_aggregate_offset_quantiles_shares_the_curve_aggregator():
    per_chainstack = [np.array([[1.0]]), np.array([[3.0]])]
    out = AC.aggregate_offset_quantiles(per_chainstack)
    if not np.isclose(out['offset_quantile_mean'][0, 0], 2.0):
        raise AssertionError(f"mean {out['offset_quantile_mean'][0, 0]} != 2.0")
    return f"offset_quantile_mean {out['offset_quantile_mean'][0, 0]}"


# ── 6. probability_map_curve / probability_map_pattern_curve (candidates 1/2) ─
# One rung's `combined_map` is a 10x10 field, all zero except a
# single pixel spike at (5,5) -- placing an anchor exactly on an integer
# pixel makes `_bilinear_sample` read that pixel's value with no blending,
# so every number below is hand-computable, not just "ran and looked right".

def _single_peak_map(peak=1.0, size=10):
    m = np.zeros((size, size))
    m[5, 5] = peak
    return {1.0: m}


def t_probe_positions_real_fixed_decoy_shifts_per_rung():
    anchors = np.array([[5.0, 5.0]])
    decoy_shift = {1.0: lambda a: a + np.array([1.0, 0.0]),
                  2.0: lambda a: a + np.array([0.0, 2.0])}
    real_by_rung, decoy_by_rung = AC._probe_positions(
        anchors, np.array([1.0, 2.0]), decoy_shift)
    if real_by_rung[1.0] is not anchors or real_by_rung[2.0] is not anchors:
        raise AssertionError('real_by_rung should be the SAME anchors array '
                             'at every rung (anchors do not move)')
    if not np.allclose(decoy_by_rung[1.0], [[6.0, 5.0]]):
        raise AssertionError(f'ds 1 decoy {decoy_by_rung[1.0]} != [[6, 5]]')
    if not np.allclose(decoy_by_rung[2.0], [[5.0, 7.0]]):
        raise AssertionError(f'ds 2 decoy {decoy_by_rung[2.0]} != [[5, 7]]')
    return 'real fixed across rungs, decoy independently shifted per rung'


def t_probability_map_alive_true_at_the_peak_false_off_the_map():
    combined_maps = _single_peak_map()
    tau = np.array([2.0])
    alive_at_peak = AC._probability_map_alive(
        combined_maps, {1.0: np.array([[5.0, 5.0]])}, np.array([1.0]),
        footprint_origin={1.0: (0.0, 0.0)}, rung_scale={1.0: 1.0}, tau=tau,
        score_threshold=0.5, kind='probability_map')
    alive_off_map = AC._probability_map_alive(
        combined_maps, {1.0: np.array([[55.0, 55.0]])}, np.array([1.0]),
        footprint_origin={1.0: (0.0, 0.0)}, rung_scale={1.0: 1.0}, tau=tau,
        score_threshold=0.5, kind='probability_map')
    if not alive_at_peak[0, 0]:
        raise AssertionError('anchor exactly on the peak pixel should be alive')
    if alive_off_map[0, 0]:
        raise AssertionError('anchor far outside the map bounds should not '
                             'be alive (-inf peak_value)')
    return 'alive at the peak, dead off the map edge'


def t_probability_map_alive_sample_step_is_actually_threaded_through():
    """`sample_step` reaches `probe_via_probability_map` rather than its own
    default.
    r=tau/scale=2.0: a step of 0.5 divides 2.0 exactly, so the search grid
    lands exactly on offset (0, 0) -- the anchor sits exactly on the peak,
    so peak_value is exactly 1.0. A step of 0.3 does NOT divide 2.0 exactly
    (`arange(-2, 2.3, 0.3)` never hits 0.0), so the closest sample is offset
    (0.1, 0.1) from the peak -- bilinear interpolation of a lone spike at
    that offset is `(1-0.1)*(1-0.1) = 0.81`. A threshold of 0.9 sits between
    the two, so this is not just "the number changed", it flips the alive
    decision -- proof `sample_step` reaches the actual probe, not just
    accepted and ignored.
    """
    combined_maps = _single_peak_map()
    tau = np.array([2.0])
    anchors = {1.0: np.array([[5.0, 5.0]])}
    fine = AC._probability_map_alive(
        combined_maps, anchors, np.array([1.0]),
        footprint_origin={1.0: (0.0, 0.0)}, rung_scale={1.0: 1.0}, tau=tau,
        score_threshold=0.9, kind='probability_map', sample_step=0.5)
    coarse = AC._probability_map_alive(
        combined_maps, anchors, np.array([1.0]),
        footprint_origin={1.0: (0.0, 0.0)}, rung_scale={1.0: 1.0}, tau=tau,
        score_threshold=0.9, kind='probability_map', sample_step=0.3)
    if not fine[0, 0]:
        raise AssertionError('sample_step=0.5 should hit the peak exactly '
                             '(peak_value=1.0 > threshold 0.9)')
    if coarse[0, 0]:
        raise AssertionError('sample_step=0.3 should NOT hit the peak '
                             'exactly (best offset (0.1,0.1) -> peak_value '
                             '~0.81 < threshold 0.9) -- if this is alive, '
                             'sample_step is being ignored')
    return 'sample_step=0.5 finds the exact peak, 0.3 does not -- threaded through'


def t_probability_map_curve_output_keys_and_shapes_match_alpha_curve():
    """Downstream (`aggregate_curves`, every CSV writer/reader/plotter)
    reads this by key/shape, not by which function produced it -- this is
    the contract check that makes that reuse safe."""
    dummy = np.zeros((1, 1))
    reference = AC.alpha_curve(dummy, dummy, dummy, dummy, rungs=[1.0],
                               alphas=[1.0, 2.0], tau_floor=0.0, threshold=0.5)
    combined_maps = _single_peak_map()
    got = AC.probability_map_curve(
        combined_maps, np.array([[5.0, 5.0]]), {1.0: lambda a: a},
        rungs=[1.0], alphas=[1.0, 2.0], tau_floor=0.0, threshold=0.5,
        footprint_origin={1.0: (0.0, 0.0)}, rung_scale={1.0: 1.0},
        kind='probability_map')
    if set(got) != set(reference):
        raise AssertionError(f'keys {sorted(got)} != alpha_curve\'s '
                             f'{sorted(reference)}')
    mismatched = [k for k in reference if got[k].shape != reference[k].shape]
    if mismatched:
        raise AssertionError(f'shape mismatch on {mismatched}: '
                             f'{[(k, got[k].shape, reference[k].shape) for k in mismatched]}')
    return f'keys {sorted(got)}, all shapes match alpha_curve\'s own'


def t_probability_map_curve_match_one_decoy_zero_at_the_peak():
    combined_maps = _single_peak_map()
    anchors = np.array([[5.0, 5.0]])
    decoy_shift = {1.0: lambda a: a + np.array([50.0, 50.0])}  # off the map
    curve = AC.probability_map_curve(
        combined_maps, anchors, decoy_shift, rungs=[1.0], alphas=[2.0],
        tau_floor=0.0, threshold=0.5, footprint_origin={1.0: (0.0, 0.0)},
        rung_scale={1.0: 1.0}, kind='probability_map')
    if curve['match_rate'][0, 0] != 1.0:
        raise AssertionError(f"match_rate {curve['match_rate'][0, 0]} != 1.0 "
                             f"-- anchor sits exactly on the peak")
    if curve['decoy_rate'][0, 0] != 0.0:
        raise AssertionError(f"decoy_rate {curve['decoy_rate'][0, 0]} != 0.0 "
                             f"-- decoy shifted off the map entirely")
    return f"match_rate {curve['match_rate'][0, 0]}, decoy_rate {curve['decoy_rate'][0, 0]}"


def t_probability_map_curve_empty_anchors_is_nan_not_a_crash():
    curve = AC.probability_map_curve(
        _single_peak_map(), np.zeros((0, 2)), {1.0: lambda a: a},
        rungs=[1.0], alphas=[1.0], tau_floor=0.0, threshold=0.5,
        footprint_origin={1.0: (0.0, 0.0)}, rung_scale={1.0: 1.0},
        kind='probability_map')
    if not np.all(np.isnan(curve['match_rate'])):
        raise AssertionError(f"expected all-NaN match_rate for 0 anchors, "
                             f"got {curve['match_rate']}")
    return 'no anchors -> NaN curve, no exception (matches alpha_curve)'


def t_probability_map_pattern_curve_output_keys_and_shapes_match_pattern_curve():
    dummy = np.zeros((1, 1))
    reference = AC.pattern_curve(dummy, dummy, dummy, dummy, rungs=[1.0],
                                 alphas=[1.0, 2.0], tau_floor=0.0, threshold=0.5)
    got = AC.probability_map_pattern_curve(
        _single_peak_map(), np.array([[5.0, 5.0]]), {1.0: lambda a: a},
        rungs=[1.0], alphas=[1.0, 2.0], tau_floor=0.0, threshold=0.5,
        footprint_origin={1.0: (0.0, 0.0)}, rung_scale={1.0: 1.0},
        kind='probability_map')
    if set(got) != set(reference):
        raise AssertionError(f'keys {sorted(got)} != pattern_curve\'s '
                             f'{sorted(reference)}')
    mismatched = [k for k in reference if got[k].shape != reference[k].shape]
    if mismatched:
        raise AssertionError(f'shape mismatch on {mismatched}')
    return f'keys {sorted(got)}, all shapes match pattern_curve\'s own'


def t_probability_map_pattern_curve_alive_everywhere_at_a_shared_peak():
    """Two rungs, both with a peak at the SAME LEVEL-0 anchor position --
    `probe_via_probability_map` reads map pixels as `(anchor-origin)/scale`,
    so the SAME level-0 point (8, 8) is pixel (8, 8) on the ds=1 map but
    pixel (4, 4) on the ds=2 map (scale=2); putting the peak at each map's
    own corresponding pixel keeps this hand-checkable rather than
    coincidentally right. Classifies as 一直存活 (alive-everywhere), fraction
    1.0 -- the same hand-checkable shape `Patterns.py`'s own tests use.
    """
    map_ds1 = np.zeros((10, 10))
    map_ds1[8, 8] = 1.0
    map_ds2 = np.zeros((10, 10))
    map_ds2[4, 4] = 1.0
    combined_maps = {1.0: map_ds1, 2.0: map_ds2}
    curve = AC.probability_map_pattern_curve(
        combined_maps, np.array([[8.0, 8.0]]),
        {1.0: lambda a: a, 2.0: lambda a: a},
        rungs=[1.0, 2.0], alphas=[1.0], tau_floor=0.0, threshold=0.5,
        footprint_origin={1.0: (0.0, 0.0), 2.0: (0.0, 0.0)},
        rung_scale={1.0: 1.0, 2.0: 2.0}, kind='probability_map')
    p = list(AC.PATTERNS).index('一直存活')
    if curve['measured_real'][p, 0] != 1.0:
        raise AssertionError(f"measured_real[一直存活] "
                             f"{curve['measured_real'][p, 0]} != 1.0")
    return f"measured_real[一直存活] = {curve['measured_real'][p, 0]}"


_SECTIONS = {
    'merge':     ['t_merge_anchors_is_the_survivalprocess_primitive'],
    'probe':     ['t_probe_decoy_single_callable_used_at_every_rung',
                 't_probe_decoy_per_rung_dict_uses_a_different_shift_each_rung',
                 't_probe_decoy_empty_anchors_is_empty_not_a_crash'],
    'curve':     ['t_alpha_curve_match_rate_crosses_at_the_right_alpha',
                 't_alpha_curve_gap_is_signed_not_absolute',
                 't_alpha_curve_threshold_gates_independently_of_distance',
                 't_alpha_curve_tau_floor_wins_over_a_tiny_alpha'],
    'aggregate': ['t_aggregate_curves_margin_uses_log_space_not_linear',
                 't_aggregate_curves_match_rate_is_a_plain_linear_mean'],
    'offset':    ['t_offset_quantiles_of_matches_numpy_on_filtered_data',
                 't_offset_quantiles_of_excludes_the_self_match_column',
                 't_offset_quantiles_of_without_source_rung_keeps_old_contaminated_behaviour',
                 't_aggregate_offset_quantiles_shares_the_curve_aggregator'],
    'probmap':   ['t_probe_positions_real_fixed_decoy_shifts_per_rung',
                 't_probability_map_alive_true_at_the_peak_false_off_the_map',
                 't_probability_map_alive_sample_step_is_actually_threaded_through',
                 't_probability_map_curve_output_keys_and_shapes_match_alpha_curve',
                 't_probability_map_curve_match_one_decoy_zero_at_the_peak',
                 't_probability_map_curve_empty_anchors_is_nan_not_a_crash',
                 't_probability_map_pattern_curve_output_keys_and_shapes_match_pattern_curve',
                 't_probability_map_pattern_curve_alive_everywhere_at_a_shared_peak'],
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
