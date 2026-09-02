#!/usr/bin/env python3
"""Tests for cli/reeval_density.py -- the matched-budget re-scoring.

    python utilities/test_modules/TestSuperPathPoint/test_reeval_density.py

THIS TEST EXISTS BECAUSE THE THING IT CHECKS ALREADY WENT WRONG ONCE. The
2026-08-31 training run reported margins of 1.50 and 3.59 for two arms and they
were not comparable: the decoy rises with point density, `margin <= 1/decoy`, so
the second arm was scored where the ceiling was twice as high. Nothing errored.
The table looked like a result.

The re-eval fixes that by cutting every view to exactly N points instead of
thresholding. So the three claims it rests on are the three things here:

  1. budget    the cut returns EXACTLY N points, which is what "matched
               density" means. If NMS leaves fewer than N survivors the density
               is not matched and the margin is not comparable -- the CLI warns
               on that, and the warning is only worth anything if the normal
               case really is exact.
  2. uniform   `decoy_uniform = 1 - exp(-(2r+1)^2 N / tile^2)` is the arithmetic
               the whole diagnosis rests on -- it is what says 420 points force
               a ceiling of 2.47 and 1966 force 1.10. Checked against actual
               random point sets, and scored against two deliberately wrong box
               sizes so that being close is a margin over a decoy rather than a
               tolerance.
  3. rebuild   the CLI reconstructs a net from a checkpoint's SHAPES, not from
               its json. Wrong channels, wrong cell or wrong descriptor width
               and `load_state_dict(strict=True)` refuses -- but only if the
               three readers are right. Round-tripped over every combination
               the four arms use.

WHAT WOULD RUN AND BE WRONG
-----------------------------
* `score_threshold=0.0` with a `>` comparison against a map that has exact
  zeros in it. Every suppressed pixel is exactly 0 after NMS, so the wrong
  comparison would admit the whole suppressed field and the budget would select
  from noise -- which is the very failure being fixed.
* the uniform formula with `nms_radius` where `2*nms_radius+1` belongs. That is
  off by 9x in the exponent and still produces a plausible-looking column.
* `_cell_of` reading the FIRST detector conv instead of the last. Both are
  4-d weights under the same prefix, so nothing would raise; the rebuilt net
  would just have the wrong cell.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402

sys.path.insert(0, os.path.join(_HERE, '..', '..', '..',
                                'training', 'SuperPathPoint', 'cli'))

from common.KeypointLabelStore import (nms_max_pool,             # noqa: E402
                                       points_from_prob)
from SuperPoint.KeypointNet import KeypointNetConfig             # noqa: E402
from SuperPoint.Trainer import _match_rate                       # noqa: E402

from reeval_density import (_cell_of, _channels_of,              # noqa: E402
                            _descriptor_dim_of)

TILE = 256
RADIUS = 4
BORDER = 4

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ── the fixture, and its own preconditions ───────────────────────────────────

def _bumpy_map(seed=0, tile=TILE, peaks=4000):
    """A probability map with more local maxima than any budget asks for.

    Not `rand`: an i.i.d. uniform field has its maxima wherever noise put them,
    which is a fine map but not one whose peak COUNT is known. This places a
    known number of gaussian bumps, so "NMS left fewer survivors than the
    budget" is a statement the fixture can be held to rather than a surprise.
    """
    rng = np.random.default_rng(seed)
    prob = np.full((tile, tile), 1e-4, np.float32)
    xy = rng.integers(BORDER + 1, tile - BORDER - 1, size=(peaks, 2))
    prob[xy[:, 1], xy[:, 0]] = rng.uniform(0.01, 1.0, size=peaks)
    return prob


def check_fixture():
    """A fixture that cannot satisfy the test's premise fails SILENTLY here.

    Both directions matter, and the second one is about the map AFTER NMS, not
    before it. A real probability map has no exact zeros in it -- a softmax
    cannot emit one -- so asking the raw fixture for a zero floor would be
    asking it to be unlike the thing it stands in for. The zeros that make
    `> 0.0` different from `>= 0.0` are the ones `nms_max_pool` WRITES:
    everything it suppresses becomes exactly 0, and that is most of the field.
    Checking the raw map instead is how the first version of this failed --
    against a fixture whose 1e-4 floor was deliberate.
    """
    prob = _bumpy_map()
    xy, _, _ = points_from_prob(prob, None, score_threshold=0.0,
                                nms_radius=RADIUS, border=BORDER,
                                max_points=None)
    if len(xy) < 600:
        raise AssertionError(
            f'the fixture map yields only {len(xy)} NMS survivors, so a budget '
            f'of 420 could never bind and the budget tests would be vacuous')

    suppressed = nms_max_pool(
        torch.from_numpy(prob)[None, None], RADIUS)[0, 0].numpy()
    zeros = int((suppressed == 0.0).sum())
    if zeros < prob.size // 2:
        raise AssertionError(
            f'only {zeros} of {prob.size} pixels are exactly zero after NMS, '
            f'so a `>= 0` comparison where `> 0` belongs would admit almost '
            f'nothing extra and the border test below would prove nothing')
    return f'{len(xy)} survivors, {100 * zeros / prob.size:.0f}% zero after NMS'


# ── 1. budget ────────────────────────────────────────────────────────────────

def t_a_budget_returns_exactly_that_many_points():
    prob = _bumpy_map()
    for n in (40, 80, 160, 320, 420):
        xy, _, _ = points_from_prob(prob, None, score_threshold=0.0,
                                    nms_radius=RADIUS, border=BORDER,
                                    max_points=n)
        if len(xy) != n:
            raise AssertionError(
                f'asked for the top {n} and got {len(xy)}. The density is then '
                f'not matched, and every margin in that column is on its own '
                f'scale')


def t_two_different_maps_give_the_same_count_at_one_budget():
    """The property the whole re-eval rests on, stated directly.

    A threshold gives two models two densities; a budget gives them one. This
    is the difference between the 2026-08-31 table and this one.
    """
    counts = set()
    for seed in range(4):
        xy, _, _ = points_from_prob(_bumpy_map(seed), None, score_threshold=0.0,
                                    nms_radius=RADIUS, border=BORDER,
                                    max_points=160)
        counts.add(len(xy))
    if counts != {160}:
        raise AssertionError(
            f'four different maps gave point counts {sorted(counts)} at one '
            f'budget. Matched density is exactly the claim that this set has '
            f'one element')


def t_a_threshold_gives_them_different_counts():
    """The decoy for the test above: without a budget the counts DO diverge.

    Without this, `t_two_different_maps...` would also pass on a fixture where
    all four maps happen to be identical.
    """
    counts = {len(points_from_prob(_bumpy_map(seed), None, score_threshold=0.5,
                                   nms_radius=RADIUS, border=BORDER,
                                   max_points=None)[0])
              for seed in range(4)}
    if len(counts) < 2:
        raise AssertionError(
            f'thresholding gave the same count {counts} on four different '
            f'maps, so this fixture cannot tell a budget apart from a '
            f'threshold and the test above proves nothing')
    return f'threshold: {sorted(counts)}   budget: 160 every time'


def t_the_border_is_not_admitted_by_a_zero_threshold():
    """`points_from_prob` writes -1 into the border, and `> 0.0` must drop it.

    A `>=` where the `>` belongs would admit the entire border ring -- 4064
    pixels of it on a 256 tile -- and a budget would then spend itself on the
    frame.
    """
    xy, _, _ = points_from_prob(_bumpy_map(), None, score_threshold=0.0,
                                nms_radius=RADIUS, border=BORDER,
                                max_points=None)
    edge = ((xy[:, 0] < BORDER) | (xy[:, 1] < BORDER) |
            (xy[:, 0] >= TILE - BORDER) | (xy[:, 1] >= TILE - BORDER))
    if edge.any():
        raise AssertionError(
            f'{int(edge.sum())} of {len(xy)} points are inside the border, '
            f'which a threshold of 0.0 was supposed to exclude')


# ── 2. uniform ───────────────────────────────────────────────────────────────

def _uniform_decoy(n, radius=RADIUS, tile=TILE):
    box = (2 * radius + 1) ** 2
    return 1.0 - np.exp(-box * n / float(tile * tile))


def _measured_decoy(n, seed, radius=RADIUS, tile=TILE):
    """What two INDEPENDENT uniform point sets actually match at density n.

    Two independent draws, not one set against a shifted copy of itself: a
    shifted copy carries the first set's spatial structure, and for uniform
    points there is none to carry -- so the two agree here, and independence is
    the simpler thing to be right about.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(0, tile, size=(n, 2)).astype(np.float64)
    b = rng.integers(0, tile, size=(n, 2)).astype(np.float64)
    return _match_rate(a, b, radius)


def t_the_uniform_decoy_formula_predicts_random_points():
    for n in (40, 160, 420):
        measured = float(np.mean([_measured_decoy(n, s) for s in range(8)]))
        predicted = _uniform_decoy(n)
        if abs(measured - predicted) > 0.05:
            raise AssertionError(
                f'at N={n} the formula says {predicted:.3f} and uniform points '
                f'match at {measured:.3f}. `decoy_uniform` would then be a '
                f'column nobody can read against')
    return ('N=420 predicted %.3f measured %.3f'
            % (_uniform_decoy(420),
               float(np.mean([_measured_decoy(420, s) for s in range(8)]))))


def t_the_right_box_beats_two_wrong_ones():
    """A tolerance is a guess; a margin over a decoy is evidence.

    The two decoys are the mistakes that would actually get made: the radius
    where the diameter belongs, and the area of a disc where NMS suppresses a
    SQUARE (the match is in the max-norm -- `Trainer._match_rate`).
    """
    n = 160
    measured = float(np.mean([_measured_decoy(n, s) for s in range(8)]))
    right = abs(measured - _uniform_decoy(n))
    for label, box in (('radius**2', RADIUS ** 2),
                       ('pi*radius**2', np.pi * RADIUS ** 2)):
        wrong = abs(measured - (1.0 - np.exp(-box * n / float(TILE * TILE))))
        if wrong < 3 * right:
            raise AssertionError(
                f'the wrong box {label} is off by {wrong:.4f} against the '
                f'right one\'s {right:.4f}. This data cannot tell them apart, '
                f'so passing says nothing about which is in the code')
    return f'err {right:.4f} vs radius**2 and pi*radius**2'


# ── 3. rebuild ───────────────────────────────────────────────────────────────

def t_a_checkpoint_says_its_own_shape():
    """Every combination the four arms use, round-tripped through a state dict.

    `strict=True` is the assertion: the rebuilt net has to accept the weights
    with no missing and no unexpected key, which one wrong reader breaks.
    """
    # `cell` is not swept and that is a fact about the model, not a gap: the
    # VGG trunk has stride 8, and `DepthToSpaceDecoder` REFUSES a cell the
    # stride does not divide rather than rounding the grid off the labels
    # (Decoders.py:189-196). Asking for cell=4 here raised ShapeMismatch at
    # build -- the code being right about a config that cannot exist.
    for channels in (1, 3):
        for cell in (8,):
            for dim in (256, 128):
                cfg = KeypointNetConfig.wired(in_channels=channels, cell=cell,
                                              descriptor_dim=dim)
                state = cfg.build('cpu').state_dict()
                got = (_channels_of(state), _cell_of(state),
                       _descriptor_dim_of(state))
                if got != (channels, cell, dim):
                    raise AssertionError(
                        f'built ({channels}, {cell}, {dim}) and read back '
                        f'{got}. The re-eval would rebuild the wrong network '
                        f'and load it into the wrong weights')
                rebuilt = KeypointNetConfig.wired(
                    in_channels=got[0], cell=got[1],
                    descriptor_dim=got[2]).build('cpu')
                rebuilt.load_state_dict(state, strict=True)


def t_a_detector_only_checkpoint_reads_as_zero_width():
    """`descriptor_dim=0` is MagicPoint, and it must not read as 256.

    Reading a missing head as the default width would build a descriptor the
    checkpoint has no weights for, and `strict=True` would then refuse a
    checkpoint that is perfectly loadable.
    """
    state = KeypointNetConfig.wired(descriptor_dim=0).build('cpu').state_dict()
    if _descriptor_dim_of(state) != 0:
        raise AssertionError(
            f'a checkpoint with no descriptor head read as width '
            f'{_descriptor_dim_of(state)}')


def t_a_detector_that_is_not_depth_to_space_is_refused():
    """The failure this reader has to make loud rather than absorb.

    `cell**2 + 1` is the only width the extraction rule below it applies to. A
    checkpoint with any other width is a different decoder, and guessing a cell
    for it would produce points on the wrong grid with nothing to say so.
    """
    # 60, NOT 50. `50 = 7**2 + 1` is a perfectly good cell-7 detector, so the
    # first version of this test asserted that a VALID width be refused and
    # failed for the right reason. 59 is not a square: sqrt(59) = 7.68, and the
    # nearest cell 8 gives 65.
    state = {'detector.head.9.conv.weight': torch.zeros(60, 8, 3, 3)}
    try:
        _cell_of(state)
    except SystemExit:
        return
    raise AssertionError('a 60-channel detector was accepted as cell**2 + 1')


_SECTIONS = {
    'fixture': ['check_fixture'],
    'budget': ['t_a_budget_returns_exactly_that_many_points',
               't_two_different_maps_give_the_same_count_at_one_budget',
               't_a_threshold_gives_them_different_counts',
               't_the_border_is_not_admitted_by_a_zero_threshold'],
    'uniform': ['t_the_uniform_decoy_formula_predicts_random_points',
                't_the_right_box_beats_two_wrong_ones'],
    'rebuild': ['t_a_checkpoint_says_its_own_shape',
                't_a_detector_only_checkpoint_reads_as_zero_width',
                't_a_detector_that_is_not_depth_to_space_is_refused'],
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(_SECTIONS))
    args = ap.parse_args()

    torch.manual_seed(0)
    for section in (args.only or list(_SECTIONS)):
        print(f'\n[{section}]')
        for name in _SECTIONS[section]:
            label = name[2:] if name.startswith('t_') else name
            check(label.replace('_', ' '), globals()[name])

    failed = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
