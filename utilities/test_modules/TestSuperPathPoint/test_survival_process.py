#!/usr/bin/env python3
"""Tests for `SurvivalAnalysis/SurvivalProcess.py`: the core, GPU-free half.
plan.md 2.2①.

    python utilities/test_modules/TestSuperPathPoint/test_survival_process.py

WHAT THIS DOES NOT COVER. `detect`, `detect_all_rungs`, `detect_all_generations`
and `rival_at` need a real `net`/prob map and are not exercised here -- this
file is exactly the part that runs in seconds without a GPU: the geometry and
set-merging arithmetic every other function is built on. A wrong merge or a
wrong nearest-point pick would still run to completion against real data and
produce a plausible, wrong survival table (ClaudeRules 8).
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

from SurvivalAnalysis import SurvivalProcess as SP             # noqa: E402

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                   # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ── 1. merge ─────────────────────────────────────────────────────────────────

def t_merge_within_radius_no_priority_keeps_first_seen():
    """No `priority` -> input order decides who a cluster's survivor is."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    keep = SP._merge_within_radius(points, radius=2.0)
    if list(keep) != [0, 2]:
        raise AssertionError(f'kept {list(keep)}, expected [0, 2] -- '
                             f'point 1 is within radius 2 of point 0')
    return f'kept indices {list(keep)}'


def t_merge_within_radius_zero_keeps_everything():
    """`radius <= 0` is a no-op -- the baseline a sensitivity sweep starts
    from (module docstring: 0 must return every point untouched)."""
    points = np.zeros((5, 2))
    keep = SP._merge_within_radius(points, radius=0.0)
    if list(keep) != [0, 1, 2, 3, 4]:
        raise AssertionError(f'radius 0 dropped points: kept {list(keep)}')
    return 'all 5 coincident points survive at radius 0'


def t_merge_within_radius_priority_picks_the_higher_scored_survivor():
    """`priority` -- not input order -- decides which of a cluster survives."""
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    keep = SP._merge_within_radius(points, radius=2.0,
                                   priority=np.array([1.0, 5.0]))
    if list(keep) != [1]:
        raise AssertionError(f'kept {list(keep)}, expected [1] (higher '
                             f'priority) even though it is index 1, not 0')
    return 'higher-priority point 1 survives over lower-priority point 0'


def t_merge_within_radius_cross_rung_same_rung_id_never_merges():
    """Cross-rung mode (`rung_id`/`rung_scale` given): two points sharing a
    `rung_id` are never the same point, at ANY distance -- that rung's own
    NMS already told them apart (SurvivalProcess.py header, spec.md "同一個
    點的定義"). 0.1 apart, both `rung_id=1.0` -- radius=2.0 alone would
    merge them in rung-blind mode; must NOT here.
    """
    points = np.array([[0.0, 0.0], [0.1, 0.0]])
    keep = SP._merge_within_radius(points, radius=2.0,
                                   rung_id=np.array([1.0, 1.0]),
                                   rung_scale=np.array([1.0, 1.0]))
    if list(keep) != [0, 1]:
        raise AssertionError(f'kept {list(keep)}, expected [0, 1] -- same '
                             f'rung_id must never merge regardless of distance')
    return ('same-rung points 0.1 apart both survive (radius=2.0 alone '
           'would have merged them)')


def t_merge_within_radius_cross_rung_additive_quantisation_boundary():
    """Different `rung_id`: radius = `radius` (the floor) + `max(scale_i,
    scale_j) // 2`. floor=0, scale=[1.0, 16.0, 16.0] -> threshold =
    0 + 16//2 = 8. Point exactly 8.0 from the ds=1 point merges (`<=`);
    8.1 away does not.
    """
    points = np.array([[0.0, 0.0], [8.0, 0.0], [8.1, 0.0]])
    keep = SP._merge_within_radius(
        points, radius=0.0,
        rung_id=np.array([1.0, 16.0, 16.0]),
        rung_scale=np.array([1.0, 16.0, 16.0]))
    if list(keep) != [0, 2]:
        raise AssertionError(f'kept {list(keep)}, expected [0, 2] -- point '
                             f'at distance 8.0 should merge into point 0 '
                             f'(threshold 8.0), point at 8.1 should not')
    return 'boundary point (8.0) merged, just-over point (8.1) survived'


# ── 2. anchors_of ────────────────────────────────────────────────────────────

def t_anchors_of_unions_rungs_and_dedupes_across_them():
    """A coarse-rung point close to a fine-rung one is dropped as a
    duplicate; a coarse-rung point with nothing nearby BECOMES an anchor
    (晚生型 depends on exactly this)."""
    per_rung = {1.0: np.array([[0.0, 0.0], [100.0, 100.0]]),
               2.0: np.array([[0.5, 0.5], [200.0, 200.0]])}
    anchors, _ = SP.anchors_of(per_rung, order=[1.0, 2.0], merge_radius_l0=2.0)
    pts = {tuple(p) for p in np.round(anchors, 1)}
    expected = {(0.0, 0.0), (100.0, 100.0), (200.0, 200.0)}
    if pts != expected:
        raise AssertionError(f'anchors {pts} != expected {expected} -- '
                             f'(0.5,0.5) should have merged into (0,0), '
                             f'(200,200) should have survived as new')
    return f'{len(anchors)} anchors: {sorted(pts)}'


def t_anchors_of_empty_rungs_produce_no_anchors():
    per_rung = {1.0: np.zeros((0, 2)), 2.0: np.zeros((0, 2))}
    anchors, _ = SP.anchors_of(per_rung, order=[1.0, 2.0], merge_radius_l0=2.0)
    if len(anchors):
        raise AssertionError(f'expected 0 anchors, got {len(anchors)}')
    return 'empty in, empty out'


def t_anchors_of_same_rung_points_never_merge_even_within_radius():
    """The bug this redesign closes (spec.md "同一個點的定義"): two points
    from the SAME rung, close enough that a rung-blind radius would have
    merged them, must both survive as their own anchors -- that rung's own
    NMS already told them apart.
    """
    per_rung = {1.0: np.array([[0.0, 0.0], [0.1, 0.0]])}
    anchors, _ = SP.anchors_of(per_rung, order=[1.0], merge_radius_l0=5.0)
    if len(anchors) != 2:
        raise AssertionError(f'expected both same-rung points to survive, '
                             f'got {len(anchors)} anchors: {anchors.tolist()}')
    return f'{len(anchors)} same-rung points survive despite radius=5.0'


def t_anchors_of_rung_scale_override_collapses_quantisation_term():
    """R-axis style (`_one_r_tile`): `rung_scale=lambda ds: 1.0` makes the
    quantisation term 0 regardless of the `order` label's own value, while
    `order`'s labels still separate rungs for the same-rung exclusion. Two
    points 8 apart, labels 1.0 and 16.0: under the DEFAULT identity
    rung_scale they merge (`max(1,16)//2=8`, distance<=8); under the
    override they must not (`max(1,1)//2=0`).
    """
    per_rung = {1.0: np.array([[0.0, 0.0]]), 16.0: np.array([[8.0, 0.0]])}
    identity, _ = SP.anchors_of(per_rung, order=[1.0, 16.0], merge_radius_l0=0.0)
    overridden, _ = SP.anchors_of(per_rung, order=[1.0, 16.0], merge_radius_l0=0.0,
                                  rung_scale=lambda ds: 1.0)
    if len(identity) != 1:
        raise AssertionError(f'default (identity) rung_scale should merge '
                             f'(radius 0+16//2=8, distance=8.0) -- got '
                             f'{len(identity)} anchors, expected 1')
    if len(overridden) != 2:
        raise AssertionError(f'rung_scale=1.0 override should NOT merge '
                             f'(radius 0+1//2=0, distance=8.0) -- got '
                             f'{len(overridden)} anchors, expected 2')
    return (f'identity scale -> {len(identity)} anchor(s), '
           f'override -> {len(overridden)} anchor(s)')


def t_anchors_of_source_rung_names_which_rung_each_anchor_survived_in():
    """The second return value (added 2026-09-13 for `offset_quantiles_of`'s
    self-match exclusion): a fine-rung point that survives names its OWN
    rung; a coarse-rung point that gets deduped away contributes nothing;
    a coarse-rung point with nothing nearby (a late-born anchor) names
    ITS OWN rung, not the fine one it failed to merge with."""
    per_rung = {1.0: np.array([[0.0, 0.0]]),
               2.0: np.array([[0.5, 0.5], [200.0, 200.0]])}
    anchors, source_rung = SP.anchors_of(per_rung, order=[1.0, 2.0],
                                         merge_radius_l0=2.0)
    by_point = {tuple(np.round(a, 1)): float(s)
               for a, s in zip(anchors, source_rung)}
    expected = {(0.0, 0.0): 1.0, (200.0, 200.0): 2.0}
    if by_point != expected:
        raise AssertionError(f'source_rung {by_point} != expected {expected} '
                             f'-- (0,0) survived from ds=1.0 (the (0.5,0.5) '
                             f'ds=2.0 duplicate merged into it), (200,200) '
                             f'is its own late-born ds=2.0 anchor')
    return f'source_rung correctly names {by_point}'


# ── 3. anchors_of_generations ────────────────────────────────────────────────

def t_anchors_of_generations_keeps_overlap_points_not_near_main():
    """The overlap tile's points join the generation's consensus set like
    any other candidate -- one far from every main point survives as its
    own anchor after the closing dedup."""
    per_rung_tiles = {16.0: (
        [np.array([[0.0, 0.0]]), np.array([[100.0, 100.0]])],
        np.array([[0.5, 0.5], [50.0, 50.0]]))}
    anchors, _ = SP.anchors_of_generations(
        per_rung_tiles, order=[16.0], tile_merge_radius=2.0)
    pts = {tuple(p) for p in np.round(anchors, 1)}
    expected = {(0.0, 0.0), (100.0, 100.0), (50.0, 50.0)}
    if pts != expected:
        raise AssertionError(f'anchors {pts} != expected {expected}')
    return f'{len(anchors)} anchors: {sorted(pts)}'


def t_anchors_of_generations_cross_rung_formula_catches_a_coarse_duplicate():
    """The whole point of the redesign (spec.md "同一個點的定義"): a genuine
    coarse-rung duplicate, offset from its finer sibling by MORE than a
    small fixed radius but within the coarse rung's OWN quantisation
    half-width, must merge instead of becoming a spurious extra (晚生型)
    anchor. Fine rung ds=1.0 at (0,0); coarse rung ds=16.0 at (6,0), 6px
    away -- more than a typical fixed `tile_merge_radius` (4 here) would
    tolerate, but within 16's own quantisation half-width (16//2=8).
    """
    per_rung_tiles = {
        1.0: ([np.array([[0.0, 0.0]])], np.zeros((0, 2))),
        16.0: ([np.array([[6.0, 0.0]])], np.zeros((0, 2))),
    }
    anchors, _ = SP.anchors_of_generations(
        per_rung_tiles, order=[1.0, 16.0], tile_merge_radius=4.0)
    if len(anchors) != 1:
        raise AssertionError(f'expected the ds=16 point to merge into the '
                             f'finer one, got {len(anchors)} anchors: '
                             f'{anchors.tolist()}')
    if not np.allclose(anchors[0], [0.0, 0.0]):
        raise AssertionError(f"surviving anchor should be the FINER rung's "
                             f'own coordinate (0,0), got {anchors[0].tolist()}')
    return f'coarse duplicate at 6px merged into finer anchor at {anchors[0].tolist()}'


# ── 4. nearest_detection ─────────────────────────────────────────────────────

def t_nearest_detection_picks_the_closest_and_its_score():
    points = np.array([[0.0, 0.0], [10.0, 0.0]])
    score = np.array([0.3, 0.9])
    query = np.array([[1.0, 0.0], [9.0, 0.0]])
    dist, sc = SP.nearest_detection(points, score, query)
    if not np.allclose(dist, [1.0, 1.0]):
        raise AssertionError(f'dist {dist} != [1.0, 1.0]')
    if not np.allclose(sc, [0.3, 0.9]):
        raise AssertionError(f'score {sc} != [0.3, 0.9] -- picked the wrong '
                             f'neighbour\'s score')
    return f'dist {dist.tolist()}, score {sc.tolist()}'


def t_nearest_detection_empty_points_is_NONE_not_a_crash():
    dist, sc = SP.nearest_detection(np.zeros((0, 2)), np.zeros(0),
                                    np.array([[0.0, 0.0]]))
    from SurvivalAnalysis.Attribution import NONE
    if dist[0] != NONE:
        raise AssertionError(f'expected NONE sentinel, got {dist[0]}')
    return 'no detections at all -> NONE, not an exception'


def t_nearest_detection_matches_brute_force_on_random_points():
    """`nearest_detection`'s `cKDTree`-based lookup has to give the exact
    same DISTANCE as a brute-force `O(N*M)` scan -- this is the test that
    function's own docstring promises exists. Ties are the one thing allowed
    to differ (cKDTree's internal traversal order vs `argmin`'s array
    order), so this only asserts on `dist`/`score`, not on which index won.
    """
    rng = np.random.default_rng(0)
    points = rng.uniform(-500.0, 500.0, size=(300, 2))
    score = rng.uniform(0.0, 1.0, size=300)
    queries = rng.uniform(-500.0, 500.0, size=(50, 2))

    dist, sc = SP.nearest_detection(points, score, queries)

    brute_dist = np.empty(len(queries))
    brute_score = np.empty(len(queries))
    for qi, q in enumerate(queries):
        d = np.linalg.norm(points - q[None, :], axis=1)
        j = int(np.argmin(d))
        brute_dist[qi] = d[j]
        brute_score[qi] = score[j]

    if not np.allclose(dist, brute_dist, atol=1e-9):
        raise AssertionError(f'cKDTree dist diverges from brute force: '
                             f'max |diff| = {np.abs(dist - brute_dist).max()}')
    if not np.allclose(sc, brute_score, atol=1e-9):
        raise AssertionError('cKDTree score diverges from brute force '
                             '(same distance but different score -> picked '
                             'a different point at an exact tie, or a bug)')
    return f'300 points, 50 queries, max |diff| = ' \
           f'{np.abs(dist - brute_dist).max():.2e}'


def t_nearest_detection_far_outside_query_still_finds_true_nearest():
    """A query can sit far outside `points`' own bounding box (the decoy-
    shifted-anchor case, or a coarse C rung's small cloud probed by anchors
    spanning the whole tree) -- `cKDTree` finds the true nearest point
    regardless of that distance, unlike the retired grid-hash expanding-ring
    search, whose per-query cost grew with (distance to the cloud) / (the
    cloud's own density-derived cell size).
    """
    points = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 1.0], [0.5, -0.5]])
    score = np.array([0.1, 0.2, 0.3, 0.4])
    query = np.array([[100_000.0, 100_000.0]])

    dist, sc = SP.nearest_detection(points, score, query)

    expected_j = int(np.argmin(np.linalg.norm(points - query, axis=1)))
    expected_dist = np.linalg.norm(points[expected_j] - query[0])
    if not np.isclose(dist[0], expected_dist):
        raise AssertionError(f'dist {dist[0]} != expected {expected_dist} -- '
                             f'did not find the true nearest point')
    if sc[0] != score[expected_j]:
        raise AssertionError(f'score {sc[0]} != point {expected_j}\'s score '
                             f'{score[expected_j]}')
    return f'query 1e5 away from a tiny cloud -> dist {dist[0]:.1f}, ' \
           f'still the true nearest'


# ── 5. coords ────────────────────────────────────────────────────────────────

def t_to_level0_and_to_tile_round_trip():
    xy = np.array([[10.0, 20.0], [0.0, 0.0]])
    origin = (500.0, 700.0)
    xy0 = SP.to_level0(xy, origin=origin, ds=4.0)
    back = SP.to_tile(xy0, origin=origin, ds=4.0)
    if not np.allclose(back, xy):
        raise AssertionError(f'{back} != {xy} after round-trip')
    return f'level0 {xy0.tolist()} -> tile {back.tolist()}'


_SECTIONS = {
    'merge':    ['t_merge_within_radius_no_priority_keeps_first_seen',
                't_merge_within_radius_zero_keeps_everything',
                't_merge_within_radius_priority_picks_the_higher_scored_survivor',
                't_merge_within_radius_cross_rung_same_rung_id_never_merges',
                't_merge_within_radius_cross_rung_additive_quantisation_boundary'],
    'anchors':  ['t_anchors_of_unions_rungs_and_dedupes_across_them',
                't_anchors_of_empty_rungs_produce_no_anchors',
                't_anchors_of_same_rung_points_never_merge_even_within_radius',
                't_anchors_of_rung_scale_override_collapses_quantisation_term',
                't_anchors_of_source_rung_names_which_rung_each_anchor_survived_in'],
    'generations': [
        't_anchors_of_generations_keeps_overlap_points_not_near_main',
        't_anchors_of_generations_cross_rung_formula_catches_a_coarse_duplicate'],
    'nearest':  ['t_nearest_detection_picks_the_closest_and_its_score',
                't_nearest_detection_empty_points_is_NONE_not_a_crash',
                't_nearest_detection_matches_brute_force_on_random_points',
                't_nearest_detection_far_outside_query_still_finds_true_nearest'],
    'coords':   ['t_to_level0_and_to_tile_round_trip'],
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
