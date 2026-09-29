#!/usr/bin/env python3
"""Tests for `SurvivalAnalysis/AliveCandidates.py`. AlphaSelectionNotes.md
section 14/15.

    python utilities/test_modules/TestSuperPathPoint/test_alive_candidates.py

PURE, NO GPU, NO NET, NO REAL SLIDE -- every function here takes arrays a
human can construct by hand. This is the ONLY verification these four
candidates have had: they were written without ever being run (ClaudeRules
2 -- only `python -m py_compile` runs directly; everything that executes
code is handed to the user), so this file is what stands between "the
algorithm as reasoned through" and "the algorithm as it actually behaves".
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths, add_training_package  # noqa: E402

setup_import_paths()
add_training_package('SuperPathPoint')

import numpy as np                                            # noqa: E402

from SurvivalAnalysis import AliveCandidates as AL             # noqa: E402

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                   # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ── 0. baseline (delegation, not a second implementation) ───────────────────

def t_baseline_delegates_to_patterns_alive_from():
    from SurvivalAnalysis.Patterns import alive_from
    score = np.array([[0.5, 0.02]])
    dist = np.array([[1.0, 5.0]])
    tau = np.array([2.0, 2.0])
    a = AL.alive_absolute_dual_threshold(score, dist, score_threshold=0.1, tau=tau)
    b = alive_from(score, dist, score_threshold=0.1, tau=tau)
    if not np.array_equal(a, b):
        raise AssertionError(f'{a} != {b}')
    return f'{a.tolist()}'


# ── 1. assemble_generation_map (candidate 1, step A) ─────────────────────────

def t_assemble_two_main_tiles_pure_placement():
    """Two 2x2 tiles side by side, no overlap -- stitching is placement, no
    arithmetic, so the output is exactly the two tiles concatenated.
    """
    left = np.array([[1.0, 2.0], [3.0, 4.0]])
    right = np.array([[5.0, 6.0], [7.0, 8.0]])
    combined = AL.assemble_generation_map(
        main_maps=[left, right], main_origins=[(0.0, 0.0), (2.0, 0.0)],
        overlap_maps=[], overlap_origins=[], scale=1.0,
        footprint_origin=(0.0, 0.0), footprint_shape=(2, 4))
    expected = np.array([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]])
    if not np.array_equal(combined, expected):
        raise AssertionError(f'{combined} != {expected}')
    return 'left|right stitched exactly'


def t_assemble_overlap_union_takes_the_max():
    """Overlap tile coincides with the RIGHT HALF of a single main tile;
    union must keep whichever of the two is larger at each pixel there.
    """
    main = np.array([[1.0, 1.0, 1.0, 1.0]])
    overlap = np.array([[9.0, 0.0]])   # covers columns 2..3 of the main tile
    combined = AL.assemble_generation_map(
        main_maps=[main], main_origins=[(0.0, 0.0)],
        overlap_maps=[overlap], overlap_origins=[(2.0, 0.0)], scale=1.0,
        footprint_origin=(0.0, 0.0), footprint_shape=(1, 4),
        overlap_mode='union')
    expected = np.array([[1.0, 1.0, 9.0, 1.0]])  # max(1,9)=9, max(1,0)=1
    if not np.array_equal(combined, expected):
        raise AssertionError(f'{combined} != {expected}')
    return f'{combined.tolist()}'


def t_assemble_overlap_intersection_takes_the_min():
    main = np.array([[1.0, 1.0, 1.0, 1.0]])
    overlap = np.array([[9.0, 0.0]])
    combined = AL.assemble_generation_map(
        main_maps=[main], main_origins=[(0.0, 0.0)],
        overlap_maps=[overlap], overlap_origins=[(2.0, 0.0)], scale=1.0,
        footprint_origin=(0.0, 0.0), footprint_shape=(1, 4),
        overlap_mode='intersection')
    expected = np.array([[1.0, 1.0, 1.0, 0.0]])  # min(1,9)=1, min(1,0)=0
    if not np.array_equal(combined, expected):
        raise AssertionError(f'{combined} != {expected}')
    return f'{combined.tolist()}'


def t_assemble_out_of_bounds_tile_raises():
    tile = np.zeros((2, 2))
    try:
        AL.assemble_generation_map(
            main_maps=[tile], main_origins=[(10.0, 10.0)],
            overlap_maps=[], overlap_origins=[], scale=1.0,
            footprint_origin=(0.0, 0.0), footprint_shape=(4, 4))
    except ValueError as e:
        return f'raised: {e}'
    raise AssertionError('expected ValueError, got no exception')


def t_assemble_unknown_overlap_mode_raises():
    tile = np.zeros((2, 2))
    try:
        AL.assemble_generation_map(
            main_maps=[tile], main_origins=[(0.0, 0.0)],
            overlap_maps=[], overlap_origins=[], scale=1.0,
            footprint_origin=(0.0, 0.0), footprint_shape=(2, 2),
            overlap_mode='bogus')
    except ValueError:
        return 'raised'
    raise AssertionError('expected ValueError, got no exception')


# ── 2. probe_via_probability_map (candidate 1, step C) ───────────────────────

def _peak_map(size=21, peak_xy=(10, 10), sigma=2.0):
    """A clean single-peak map: a 2D Gaussian bump, for probe tests where
    "where is the maximum" has one unambiguous right answer.
    """
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    px, py = peak_xy
    return np.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma ** 2))


def t_probe_finds_the_peak_when_it_is_within_radius():
    m = _peak_map(peak_xy=(10, 10))
    anchor = np.array([[9.0, 10.0]])  # 1 px off the true peak, level-0 units
    value, dist = AL.probe_via_probability_map(
        m, anchor, map_origin=(0.0, 0.0), scale=1.0, tau=5.0, sample_step=0.25)
    if value[0] < 0.99:
        raise AssertionError(f'peak value {value[0]} should be ~1.0 (found the bump)')
    if dist[0] > 1.5:
        raise AssertionError(f'peak_dist {dist[0]} should be close to the 1px offset')
    return f'value={value[0]:.4f} dist={dist[0]:.3f}'


def t_probe_window_too_small_misses_the_peak():
    """Same map, but the anchor is far enough that a small tau CANNOT reach
    the true peak -- the found max must be lower than the true peak's value,
    proving the circular window is actually being enforced, not ignored.
    """
    m = _peak_map(peak_xy=(10, 10), sigma=1.0)
    anchor = np.array([[10.0, 10.0 + 8.0]])  # 8px away, true peak far outside a small tau
    value, _dist = AL.probe_via_probability_map(
        m, anchor, map_origin=(0.0, 0.0), scale=1.0, tau=1.0, sample_step=0.25)
    if value[0] > 0.5:
        raise AssertionError(
            f'value {value[0]} should be well below the true peak (~1.0) -- '
            f'the window radius is not being respected')
    return f'value={value[0]:.4f} (correctly cannot see the far peak)'


def t_probe_anchor_entirely_outside_the_map_is_negative_infinity():
    m = _peak_map()
    anchor = np.array([[1000.0, 1000.0]])
    value, _dist = AL.probe_via_probability_map(
        m, anchor, map_origin=(0.0, 0.0), scale=1.0, tau=2.0, sample_step=0.5)
    if not np.isneginf(value[0]):
        raise AssertionError(f'expected -inf, got {value[0]}')
    return '-inf, as documented (not a clamped edge value)'


def t_probe_bilinear_interpolation_is_exact_on_a_linear_ramp():
    """Bilinear interpolation of a perfectly planar function is exact --
    checked directly against `_bilinear_sample`, NOT through the disk-search
    `probe_via_probability_map` wrapper.

    The wrapper's own grid search has a SEPARATE, direction-dependent
    discretisation error: the best `sample_step`-spaced point inside a disk
    of radius `tau` is not exactly the true continuum maximum, and how far
    off depends on which way the ramp's gradient happens to point relative
    to the grid. An earlier version of this test tried to check both at
    once -- comparing the wrapper's output against a hand-picked closed form
    assumed to sit on the (1,1)/sqrt(2) diagonal -- which is wrong for any
    ramp whose gradient (a, b) is not itself along that diagonal (this one
    is (2, 3)), and conflated grid coarseness with interpolation error even
    when the direction is fixed. Calling `_bilinear_sample` directly at an
    off-grid point isolates exactly the thing this test's name promises.
    """
    size = 10
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    a, b, c = 2.0, 3.0, 1.0
    ramp = a * xs + b * ys + c        # f(x,y) = 2x + 3y + 1, exactly planar
    px, py = np.array([4.3]), np.array([6.7])   # off-grid, non-integer
    value = AL._bilinear_sample(ramp, px, py, size, size)
    expected = a * px[0] + b * py[0] + c
    if abs(value[0] - expected) > 1e-9:
        raise AssertionError(
            f'value {value[0]:.9f} != closed-form {expected:.9f} -- bilinear '
            f'interpolation of an exactly planar function should be exact '
            f'everywhere, not approximate')
    return f'value={value[0]:.6f} == closed-form {expected:.6f} (off-grid point)'


def t_probe_empty_anchors_is_empty_not_a_crash():
    m = _peak_map()
    value, dist = AL.probe_via_probability_map(
        m, np.zeros((0, 2)), map_origin=(0.0, 0.0), scale=1.0, tau=2.0)
    if len(value) or len(dist):
        raise AssertionError(f'expected empty outputs, got {value}, {dist}')
    return 'empty in, empty out'


# ── 3. alive_probability_map (candidate 1, the decision) ─────────────────────

def t_alive_probability_map_matches_its_own_formula():
    peak_value = np.array([[0.5, 0.02]])
    peak_dist = np.array([[1.0, 5.0]])
    tau = np.array([2.0, 2.0])
    alive = AL.alive_probability_map(peak_value, peak_dist,
                                     score_threshold=0.1, tau=tau)
    expected = np.array([[True, False]])   # col0: 0.5>0.1 & 1<=2; col1: 0.02<0.1
    if not np.array_equal(alive, expected):
        raise AssertionError(f'{alive} != {expected}')
    return f'{alive.tolist()}'


# ── 4. alive_scale_local_extremum (candidate 2) ──────────────────────────────

def t_scale_extremum_accepts_a_true_peak_in_the_middle_rung():
    """peak_value rises then falls across rungs -- the middle rung (the
    actual local max) must pass; the strictly-rising or strictly-falling
    neighbours should not both also pass with margin=0 unless they too are
    non-decreasing on one side (edge rungs only have one neighbour).
    """
    peak_value = np.array([[0.1, 0.9, 0.2]])   # rung 1 is the clear peak
    peak_dist = np.zeros((1, 3))
    tau = np.array([10.0, 10.0, 10.0])
    alive = AL.alive_scale_local_extremum(
        peak_value, peak_dist, score_threshold=0.05, tau=tau, margin=0.0)
    if not alive[0, 1]:
        raise AssertionError('the true middle peak should be alive')
    return f'{alive.tolist()}'


def t_scale_extremum_rejects_a_dip_between_two_stronger_neighbours():
    peak_value = np.array([[0.9, 0.1, 0.9]])   # rung 1 is a dip, not a peak
    peak_dist = np.zeros((1, 3))
    tau = np.array([10.0, 10.0, 10.0])
    alive = AL.alive_scale_local_extremum(
        peak_value, peak_dist, score_threshold=0.05, tau=tau, margin=0.0)
    if alive[0, 1]:
        raise AssertionError('a dip between two stronger neighbours should not be alive')
    return f'{alive.tolist()}'


def t_scale_extremum_edge_rungs_only_compare_their_one_neighbour():
    """The finest rung (index 0) has no j-1 to compare against; a rising
    sequence must not falsely reject it for "not being higher than a
    neighbour that does not exist on that side".
    """
    peak_value = np.array([[0.3, 0.6, 0.9]])   # monotonically rising
    peak_dist = np.zeros((1, 3))
    tau = np.array([10.0, 10.0, 10.0])
    alive = AL.alive_scale_local_extremum(
        peak_value, peak_dist, score_threshold=0.05, tau=tau, margin=0.0)
    # rung 0: only compared against rung 1 (0.3 < 0.6) -> NOT alive
    # rung 2: only compared against rung 1 (0.9 >= 0.6) -> alive
    if alive[0, 0]:
        raise AssertionError('rung 0 is lower than its only neighbour, should not be alive')
    if not alive[0, 2]:
        raise AssertionError('rung 2 (the coarsest, monotone max) should be alive')
    return f'{alive.tolist()}'


# ── 5. alive_exp_decay_joint_score (candidate 3) ─────────────────────────────

def t_exp_decay_at_zero_distance_combined_equals_score():
    score = np.array([[0.5]])
    dist = np.array([[0.0]])
    tau = np.array([2.0])
    alive = AL.alive_exp_decay_joint_score(score, dist, tau=tau,
                                           combined_threshold=0.4)
    if not alive[0, 0]:
        raise AssertionError('combined at dist=0 equals score (0.5), should pass 0.4')
    return 'combined(dist=0) == score, as documented'


def t_exp_decay_negative_distance_is_always_dead():
    score = np.array([[100.0]])   # absurdly high score, should not matter
    dist = np.array([[-1.0]])     # NONE sentinel
    tau = np.array([2.0])
    alive = AL.alive_exp_decay_joint_score(score, dist, tau=tau,
                                           combined_threshold=0.0)
    if alive[0, 0]:
        raise AssertionError('dist<0 (NONE) must never be alive regardless of score')
    return 'NONE sentinel respected'


def t_exp_decay_matches_hand_computed_value():
    score = np.array([[1.0]])
    dist = np.array([[2.0]])
    tau = np.array([2.0])
    expected = 1.0 * np.exp(-2.0 / 2.0)   # = exp(-1) ~= 0.3679
    alive_below = AL.alive_exp_decay_joint_score(
        score, dist, tau=tau, combined_threshold=expected + 0.01)
    alive_above = AL.alive_exp_decay_joint_score(
        score, dist, tau=tau, combined_threshold=expected - 0.01)
    if alive_below[0, 0] or not alive_above[0, 0]:
        raise AssertionError(
            f'combined should be exactly exp(-1)={expected:.4f}, '
            f'threshold straddling it did not behave as expected')
    return f'combined == exp(-1) == {expected:.4f}, confirmed from both sides'


# ── 6. alive_two_component_mixture (candidate 4) ─────────────────────────────

def t_mixture_gives_near_points_higher_posterior_than_far_points():
    """Synthetic dist column: half genuinely-Rayleigh(sigma=2) (a tight
    cluster near 0), half Rayleigh(sigma=50) (a diffuse background) -- the
    fit does not need to recover the exact sigmas, only to end up scoring a
    dist=1 point as more likely genuine than a dist=80 point.
    """
    rng = np.random.default_rng(0)
    n = 400
    genuine = rng.rayleigh(scale=2.0, size=n // 2)
    background = rng.rayleigh(scale=50.0, size=n // 2)
    dist_col = np.concatenate([genuine, background])
    # Fit on the pooled column, then read the two extreme points' posteriors
    # directly via the same internal fit `alive_two_component_mixture` uses.
    pi, sigma_g, sigma_b = AL._fit_rayleigh_mixture(dist_col)
    if sigma_g >= sigma_b:
        raise AssertionError(f'expected sigma_genuine < sigma_background, '
                             f'got {sigma_g:.2f} >= {sigma_b:.2f}')
    f_near_g, f_near_b = AL._rayleigh_pdf(np.array([1.0]), sigma_g), AL._rayleigh_pdf(np.array([1.0]), sigma_b)
    f_far_g, f_far_b = AL._rayleigh_pdf(np.array([80.0]), sigma_g), AL._rayleigh_pdf(np.array([80.0]), sigma_b)
    post_near = (pi * f_near_g) / (pi * f_near_g + (1 - pi) * f_near_b)
    post_far = (pi * f_far_g) / (pi * f_far_g + (1 - pi) * f_far_b)
    if not (post_near[0] > post_far[0]):
        raise AssertionError(f'post_near={post_near[0]:.3f} should exceed '
                             f'post_far={post_far[0]:.3f}')
    return (f'sigma_genuine={sigma_g:.2f} sigma_background={sigma_b:.2f} '
           f'post(dist=1)={post_near[0]:.3f} post(dist=80)={post_far[0]:.3f}')


def t_mixture_too_few_points_falls_back_without_crashing():
    tiny = np.array([1.0, 2.0, 1.5])
    pi, sigma_g, sigma_b = AL._fit_rayleigh_mixture(tiny)
    if not (0.0 <= pi <= 1.0 and sigma_g > 0 and sigma_b > 0):
        raise AssertionError(f'degenerate fallback produced invalid params: '
                             f'{pi}, {sigma_g}, {sigma_b}')
    return f'pi={pi:.2f} sigma_g={sigma_g:.2f} sigma_b={sigma_b:.2f} (n<8 fallback)'


def t_mixture_end_to_end_alive_shape_and_gating():
    rng = np.random.default_rng(1)
    n = 200
    genuine = rng.rayleigh(scale=2.0, size=n)
    dist = genuine[:, None]
    score_pass = np.full_like(dist, 1.0)
    score_fail = np.full_like(dist, -1.0)
    alive_pass = AL.alive_two_component_mixture(
        score_pass, dist, score_threshold=0.5, posterior_threshold=0.5)
    alive_fail = AL.alive_two_component_mixture(
        score_fail, dist, score_threshold=0.5, posterior_threshold=0.5)
    if alive_pass.shape != dist.shape:
        raise AssertionError(f'shape mismatch: {alive_pass.shape} != {dist.shape}')
    if alive_fail.any():
        raise AssertionError('score_threshold gate should override a favourable posterior')
    return f'{alive_pass.sum()}/{n} alive when score passes, 0 when it does not'


_SECTIONS = {
    'baseline':  ['t_baseline_delegates_to_patterns_alive_from'],
    'assemble':  ['t_assemble_two_main_tiles_pure_placement',
                 't_assemble_overlap_union_takes_the_max',
                 't_assemble_overlap_intersection_takes_the_min',
                 't_assemble_out_of_bounds_tile_raises',
                 't_assemble_unknown_overlap_mode_raises'],
    'probe':     ['t_probe_finds_the_peak_when_it_is_within_radius',
                 't_probe_window_too_small_misses_the_peak',
                 't_probe_anchor_entirely_outside_the_map_is_negative_infinity',
                 't_probe_bilinear_interpolation_is_exact_on_a_linear_ramp',
                 't_probe_empty_anchors_is_empty_not_a_crash'],
    'alive-map': ['t_alive_probability_map_matches_its_own_formula'],
    'extremum':  ['t_scale_extremum_accepts_a_true_peak_in_the_middle_rung',
                 't_scale_extremum_rejects_a_dip_between_two_stronger_neighbours',
                 't_scale_extremum_edge_rungs_only_compare_their_one_neighbour'],
    'exp-decay': ['t_exp_decay_at_zero_distance_combined_equals_score',
                 't_exp_decay_negative_distance_is_always_dead',
                 't_exp_decay_matches_hand_computed_value'],
    'mixture':   ['t_mixture_gives_near_points_higher_posterior_than_far_points',
                 't_mixture_too_few_points_falls_back_without_crashing',
                 't_mixture_end_to_end_alive_shape_and_gating'],
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
