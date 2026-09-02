#!/usr/bin/env python3
"""Tests for training/SuperPathPoint/common/KeypointLabelStore.py.

    python utilities/test_modules/test_keypoint_label_store.py

No weights, no GPU, no WSI. Temporary directories only.

WHAT THIS IS DEFENDING AGAINST
-------------------------------
Three failures, none of which raises:

    padding read as data      `kp_xy` is a rectangle and most of it is zeros.
                              Reading it without `n_kp` gives every tile a
                              crowd of points at (0, 0) -- a plausible corner,
                              in the corner, on every tile
    the cap doing the cutting  `points_per_megapixel` is a MEMORY bound and the
                              threshold is what should select (spec.md 6.3). A
                              store at its cap is lossy and looks identical to
                              one that is not
    two rounds colliding      round 2 of Stage A writes labels for the same
                              slide and rung from a different teacher. Same
                              filename would mean round 3 trains on round 1

and one that does: an aggregated map turned into points by a different rule than
the student's predictions. That one is structural rather than testable here --
`points_from_prob` is the single rule, and `KeypointNet.extract_keypoints` calls
it -- so what this file pins is the rule itself, in the order upstream applies
it.

Sections:
  1. points   -- threshold, NMS, border and cap, in that order
  2. padding  -- the rectangle and what says where it ends
  3. identity -- what makes two label sets two files
  4. store    -- write, read, refuse
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

# `utilities/` holds the one definition of the output roots (`_paths.py`), and
# `setup_import_paths` -- which puts every other package on the path -- is
# inside it, so it has to be reachable before anything else is imported.
#
# BOTH parents are inserted, and that is deliberate rather than sloppy: this
# file runs from `utilities/test_modules/` and from
# `utilities/test_modules/TestSuperPathPoint/`, one level deeper, and inserting
# both means the move needs no edit here. The one that is not `utilities/` is
# either the repo root or `test_modules/`; neither holds a `_paths.py`, and
# `setup_import_paths` puts the repo root on the path anyway.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import numpy as np                                               # noqa: E402

from common import KeypointLabelStore                             # noqa: E402
from common.KeypointLabelStore import (LabelMeta, LabelMismatch,   # noqa: E402
                                       batch_from_lists, cap_for,
                                       points_from_prob)

_RESULTS = []
TILE = 64


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def _map_with(peaks) -> np.ndarray:
    """A probability map that is zero except at the given (x, y, value)."""
    prob = np.zeros((TILE, TILE), np.float32)
    for x, y, value in peaks:
        prob[y, x] = value
    return prob


# ══════════════════════════════════════════════════════════════════════════════
#  1. points
# ══════════════════════════════════════════════════════════════════════════════

def t_isolated_peaks_come_back_at_their_own_coordinates():
    """And in (x, y), not (row, col).

    The one that has bitten this project before (CLAUDE.md, the rotated-GT
    episode): an (x, y)/(row, col) swap is invisible on a symmetric example, so
    the peaks here are deliberately off-diagonal.
    """
    xy, score, _ = points_from_prob(
        _map_with([(10, 30, 0.9), (40, 12, 0.5)]),
        score_threshold=0.1, nms_radius=4, border=4)
    got = {tuple(int(v) for v in point) for point in xy}
    assert got == {(10, 30), (40, 12)}, got
    assert score.max() > 0.8, score
    return f'{len(xy)} points, (x, y)'


def t_nms_keeps_the_higher_of_two_close_peaks():
    """Within the radius, one survives; outside it, both do.

    Both halves are asserted, because "NMS is on" and "NMS radius is right" are
    different claims and only the pair separates them.
    """
    near = points_from_prob(_map_with([(20, 20, 0.9), (22, 20, 0.4)]),
                            score_threshold=0.1, nms_radius=4, border=4)[0]
    far = points_from_prob(_map_with([(20, 20, 0.9), (30, 20, 0.4)]),
                           score_threshold=0.1, nms_radius=4, border=4)[0]
    assert len(near) == 1, f'{len(near)} points survived at distance 2'
    assert tuple(near[0]) == (20, 20), tuple(near[0])
    assert len(far) == 2, f'{len(far)} points survived at distance 10'
    return 'distance 2 -> 1 point, distance 10 -> 2'


def t_the_border_is_cut_and_the_first_kept_row_is_the_margin():
    """Upstream removes rows 0..pad-1, so row `pad` survives. Pinning the exact
    boundary and not just "the corner is gone": an off-by-one here silently
    changes the fraction of every tile that can hold a label."""
    inside = points_from_prob(_map_with([(4, 4, 0.9)]),
                              score_threshold=0.1, nms_radius=0, border=4)[0]
    outside = points_from_prob(_map_with([(3, 3, 0.9)]),
                               score_threshold=0.1, nms_radius=0, border=4)[0]
    assert len(inside) == 1 and tuple(inside[0]) == (4, 4), inside
    assert len(outside) == 0, outside
    return 'row 4 kept, row 3 cut, border=4'


def t_the_border_value_is_negative_not_zero():
    """Upstream writes -1 into the border (`superpoint_pytorch.py:126-130`).

    Zeroing it instead is indistinguishable at any positive threshold, so the
    check has to run at a NEGATIVE one -- which is not a contrived case: a
    threshold sweep looking for where the map starts producing points goes
    there, and a zeroed border would hand back the whole rim as keypoints.
    """
    points = points_from_prob(_map_with([(20, 20, 0.5)]),
                              score_threshold=-0.5, nms_radius=0, border=4)[0]
    assert len(points), 'nothing passed a negative threshold'
    xs, ys = points[:, 0], points[:, 1]
    assert xs.min() >= 4 and ys.min() >= 4, (int(xs.min()), int(ys.min()))
    assert xs.max() < TILE - 4 and ys.max() < TILE - 4, (int(xs.max()),
                                                         int(ys.max()))
    assert len(points) == (TILE - 8) ** 2, (
        f'{len(points)} points where the interior holds {(TILE - 8) ** 2}; the '
        f'border was zeroed rather than set to -1')
    return f'threshold -0.5 keeps the {(TILE - 8) ** 2} interior pixels only'


def t_the_cap_keeps_the_highest_scores():
    peaks = [(8 * i + 8, 8 * j + 8, 0.1 * (i + 3 * j) + 0.1)
             for i in range(3) for j in range(2)]
    xy, score, _ = points_from_prob(_map_with(peaks), score_threshold=0.05,
                                    nms_radius=4, border=4, max_points=2)
    assert len(xy) == 2, len(xy)
    assert score[0] >= score[1], score
    assert score.min() > 0.4, f'the cap kept low scores: {score}'
    return f'6 peaks -> 2, top scores {np.round(score, 2)}'


def t_counts_are_read_at_the_kept_points():
    """`kp_count` has to be sampled at (y, x) of each point, not at (x, y).

    A transposed read gives a count from a different pixel -- still a plausible
    small integer, and the whole reason the field exists is to tell a coverage
    zero from a detection zero.
    """
    counts = np.zeros((TILE, TILE), np.uint8)
    counts[30, 10] = 77           # (y=30, x=10)
    xy, _, got = points_from_prob(_map_with([(10, 30, 0.9)]), counts,
                                  score_threshold=0.1, nms_radius=4, border=4)
    assert tuple(xy[0]) == (10, 30), xy
    assert int(got[0]) == 77, (
        f'count came back {int(got[0])}, so it was read at ({xy[0][1]}, '
        f'{xy[0][0]}) instead of ({xy[0][0]}, {xy[0][1]})')
    return 'count 77 at the point'


def t_cap_is_a_density_not_a_count():
    """256 and 1024 must get the same points per megapixel, not the same M.

    spec.md 6.3: three tile sizes are three models, and a fixed M would make
    their label densities differ by 16x -- straight into any comparison between
    them.
    """
    want = 30000.0
    small, large = cap_for(256, want), cap_for(1024, want)

    # The DENSITY is the invariant. The cap is a whole number of points, so
    # rounding moves it by at most half a point -- and half a point is worth
    # `0.5 / area` in density, which is 7.6 per Mpx at tile 256 and 0.48 at
    # tile 1024. The bound below is that quantity and not a number: this test
    # has now been wrong twice by asserting a constant, first `large == 16 *
    # small` (31456 against 31457) and then `|density - 30000| < 1` (the true
    # error at tile 256 is 1.22). A tolerance that does not scale with the
    # thing it is bounding is a guess, and it fails on the small tile because
    # that is where one point is worth the most.
    for tile, cap in ((256, small), (1024, large)):
        area_mpx = tile * tile / 1e6
        density = cap / area_mpx
        assert abs(density - want) <= 0.5 / area_mpx, (tile, cap, density)

    # And the multiple, to within the same rounding on both ends.
    assert abs(large - 16 * small) <= 16, (small, large)
    return f'256 -> {small}, 1024 -> {large}, density within half a point'


# ══════════════════════════════════════════════════════════════════════════════
#  2. padding
# ══════════════════════════════════════════════════════════════════════════════

def t_points_of_returns_exactly_what_went_in():
    points = [np.array([[1, 2], [3, 4], [5, 6]], np.int16),
              np.zeros((0, 2), np.int16),
              np.array([[7, 8]], np.int16)]
    batch = batch_from_lists([(0, 0), (10, 10), (20, 20)], points,
                             [np.array([0.9, 0.5, 0.1], np.float32),
                              np.zeros(0, np.float32),
                              np.array([0.3], np.float32)],
                             [np.array([9, 9, 9], np.uint8),
                              np.zeros(0, np.uint8),
                              np.array([4], np.uint8)],
                             cap=8)
    for i, want in enumerate(points):
        assert np.array_equal(batch.points_of(i), want), (i, batch.points_of(i))
    assert list(batch.n_kp) == [3, 0, 1], batch.n_kp
    # The failure this guards: reading the rectangle instead of points_of.
    assert batch.kp_xy.shape == (3, 8, 2), batch.kp_xy.shape
    assert (batch.kp_xy[1] == 0).all(), 'the empty tile is not zero-padded'
    return 'ragged in, rectangle out, n_kp says where'


def t_at_cap_counts_the_truncated_tiles():
    points = [np.arange(2 * n, dtype=np.int16).reshape(n, 2) for n in (2, 4, 4)]
    batch = batch_from_lists([(0, 0)] * 3, points,
                             [np.zeros(len(p), np.float32) for p in points],
                             [np.zeros(len(p), np.uint8) for p in points],
                             cap=4)
    assert batch.at_cap == 2, batch.at_cap
    return '2 of 3 tiles at the cap'


def t_the_cap_is_the_configs_and_not_the_data_maximum():
    """Otherwise a rung where every tile happened to yield few points writes a
    store with a smaller M than the config asked for -- two files of the same
    identity with different second dimensions."""
    batch = batch_from_lists([(0, 0)], [np.zeros((1, 2), np.int16)],
                             [np.zeros(1, np.float32)],
                             [np.zeros(1, np.uint8)], cap=500)
    assert batch.cap == 500, batch.cap
    assert batch.at_cap == 0, batch.at_cap
    return 'one point, cap stays 500'


# ══════════════════════════════════════════════════════════════════════════════
#  3. identity
# ══════════════════════════════════════════════════════════════════════════════

def _meta(**over) -> LabelMeta:
    base = dict(wsi_stem='SLIDE_A', ds=4.0, tile=TILE, ha_id='aaaa1111',
                pretile_id='bbbb2222', score_threshold=0.005,
                points_per_megapixel=30000.0, nms_radius=4, border=4)
    base.update(over)
    return LabelMeta(**base)


def t_every_identity_field_changes_the_filename():
    base = _meta()
    moved = {'wsi_stem': 'SLIDE_B', 'ds': 8.0, 'tile': 128,
             'ha_id': 'cccc3333', 'pretile_id': 'dddd4444',
             'score_threshold': 0.01, 'points_per_megapixel': 1000.0,
             'nms_radius': 3, 'border': 8}
    assert set(moved) == set(KeypointLabelStore._IDENTITY_FIELDS), (
        f'this test covers {sorted(moved)} but the module identifies on '
        f'{sorted(KeypointLabelStore._IDENTITY_FIELDS)}')
    for field, value in moved.items():
        assert _meta(**{field: value}).cfg_hash() != base.cfg_hash(), field
    return f'{len(moved)} fields'


def t_two_rounds_of_stage_a_do_not_collide():
    """The one this file exists for. Round 2 differs in `ha_id` alone -- same
    slide, same rung, same threshold, a different teacher."""
    round1, round2 = _meta(ha_id='aaaa1111'), _meta(ha_id='eeee5555')
    assert round1.filename() != round2.filename(), round1.filename()
    assert round1.filename().startswith('SLIDE_A__ds4__'), round1.filename()
    return f'{round1.filename()} vs {round2.filename()}'


def t_provenance_does_not_change_the_filename():
    base = _meta()
    for field, value in (('wsi_path', '/other/mount.svs'), ('n_tiles', 500),
                         ('cap', 999), ('mean_n_kp', 12.5),
                         ('created_at', '2020-01-01T00:00:00')):
        assert _meta(**{field: value}).cfg_hash() == base.cfg_hash(), field
    return 'wsi_path, counts, created_at'


# ══════════════════════════════════════════════════════════════════════════════
#  4. store
# ══════════════════════════════════════════════════════════════════════════════

def _batch(n=3, cap=8):
    points = [np.array([[i + 1, i + 2]], np.int16) for i in range(n)]
    return batch_from_lists([(100 * i, 200 * i) for i in range(n)], points,
                            [np.array([0.5], np.float32)] * n,
                            [np.array([7], np.uint8)] * n, cap=cap)


def t_write_then_read_returns_the_same_arrays():
    with tempfile.TemporaryDirectory() as root:
        batch = _batch()
        meta = _meta(n_tiles=len(batch), cap=batch.cap)
        path = KeypointLabelStore.save(root, batch, meta)
        back, got = KeypointLabelStore.load(path)

        for name in ('tile_x', 'tile_y', 'kp_xy', 'kp_score', 'kp_count', 'n_kp'):
            assert np.array_equal(getattr(back, name), getattr(batch, name)), name
        assert got.cfg_hash() == meta.cfg_hash()
        # The MaskStore bug, pinned here too: `from __future__ import
        # annotations` makes every field annotation a STRING, so a decoder that
        # compares `field.type is float` hands back str and the first caller to
        # format it with :.2f raises.
        assert isinstance(got.ds, float), type(got.ds)
        assert isinstance(got.tile, int), type(got.tile)
        assert f'{got.ds:.1f}' == '4.0'
    return 'six tensors, and the meta comes back typed'


def t_require_refuses_rather_than_falls_back():
    with tempfile.TemporaryDirectory() as root:
        batch = _batch()
        path = KeypointLabelStore.save(root, batch,
                                       _meta(n_tiles=len(batch), cap=batch.cap))
        KeypointLabelStore.load(path, require={'wsi_stem': 'SLIDE_A'})
        try:
            KeypointLabelStore.load(path, require={'ha_id': 'not-this-one'})
        except LabelMismatch as e:
            assert 'ha_id' in str(e), str(e)
            return 'refused, and named the field'
    raise AssertionError('a mismatched require was accepted')


def t_find_one_refuses_two_rounds_and_takes_ha_id():
    with tempfile.TemporaryDirectory() as root:
        batch = _batch()
        for ha_id in ('aaaa1111', 'eeee5555'):
            KeypointLabelStore.save(
                root, batch, _meta(ha_id=ha_id, n_tiles=len(batch),
                                   cap=batch.cap))
        try:
            KeypointLabelStore.find_one(root, wsi_stem='SLIDE_A')
        except LabelMismatch as e:
            assert '2 label sets' in str(e), str(e)
        else:
            raise AssertionError('find_one picked one of two rounds')
        assert KeypointLabelStore.find_one(root, ha_id='eeee5555').exists()
    return 'ambiguous refused, ha_id resolves it'


def t_save_validates_the_meta_against_the_batch():
    with tempfile.TemporaryDirectory() as root:
        batch = _batch(n=3)
        for meta, why in ((_meta(n_tiles=99, cap=batch.cap), 'wrong n_tiles'),
                          (_meta(n_tiles=3, cap=batch.cap, ha_id=''),
                           'empty ha_id')):
            try:
                KeypointLabelStore.save(root, batch, meta)
            except ValueError:
                continue
            raise AssertionError(f'{why} was accepted')
    return 'n_tiles and ha_id both checked'


# ══════════════════════════════════════════════════════════════════════════════

_SECTIONS = {
    'points':   ['t_isolated_peaks_come_back_at_their_own_coordinates',
                 't_nms_keeps_the_higher_of_two_close_peaks',
                 't_the_border_is_cut_and_the_first_kept_row_is_the_margin',
                 't_the_border_value_is_negative_not_zero',
                 't_the_cap_keeps_the_highest_scores',
                 't_counts_are_read_at_the_kept_points',
                 't_cap_is_a_density_not_a_count'],
    'padding':  ['t_points_of_returns_exactly_what_went_in',
                 't_at_cap_counts_the_truncated_tiles',
                 't_the_cap_is_the_configs_and_not_the_data_maximum'],
    'identity': ['t_every_identity_field_changes_the_filename',
                 't_two_rounds_of_stage_a_do_not_collide',
                 't_provenance_does_not_change_the_filename'],
    'store':    ['t_write_then_read_returns_the_same_arrays',
                 't_require_refuses_rather_than_falls_back',
                 't_find_one_refuses_two_rounds_and_takes_ha_id',
                 't_save_validates_the_meta_against_the_batch'],
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(_SECTIONS))
    args = ap.parse_args()

    for section in (args.only or sorted(_SECTIONS)):
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
