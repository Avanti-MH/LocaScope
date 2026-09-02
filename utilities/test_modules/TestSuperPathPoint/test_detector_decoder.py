#!/usr/bin/env python3
"""Tests for training/SuperPathPoint/SuperPoint/Decoders.py.

    python utilities/test_modules/test_detector_decoder.py

No weights, no GPU, seconds. Every tensor here is made up.

WHAT THIS IS DEFENDING AGAINST
-------------------------------
One thing, and it is invisible: **the two permutes in the wrong order transpose
every keypoint inside its own 8x8 cell**. The output has the right shape, the
right number of peaks, and peaks in plausible places -- each one is simply at
`(dx, dy)` where it should be at `(dy, dx)` within its cell. Every label is then
0 to 7 px off in a pattern that depends on position, no assertion about counts
or shapes can see it, and the student trains happily on it.

So the checks here are all about WHICH PIXEL, and they are scored against the
transposed convention as an explicit decoy rather than against a tolerance.

The second thing is smaller and also silent: the dustbin is dropped and NOT
renormalised (`models/utils.py:21-26`). After dropping it, the values are "the
probability that there is a keypoint at this pixel", and a cell with nothing in
it correctly sums to near zero. Renormalising would push every empty cell back
up to one, which reads as a detector that fires everywhere.

Sections:
  1. placement  -- which pixel a channel decodes to, against the decoy
  2. dustbin    -- dropped, and not renormalised
  3. guards     -- what has to raise
"""

from __future__ import annotations

import argparse
import os
import sys

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

import torch                                                     # noqa: E402

from SuperPoint.Decoders import depth_to_space_prob               # noqa: E402

_RESULTS = []

#: The CELL, which is `Decoders`' name for it and not `stride`: the two are
#: the same 8 here and are two different numbers in general (spec.md 5.1). A
#: test that called it stride would be pinning the coincidence.
CELL = 8
GRID = 3           # a 3x3 grid of cells -> a 24x24 decoded map


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def _logits(channel: int, cell_row: int, cell_col: int,
            value: float = 20.0) -> torch.Tensor:
    """Logits whose softmax is ~1 on `channel` at one cell and ~uniform else.

    A large logit rather than a one-hot probability, because the function under
    test starts with a softmax -- handing it probabilities would test a
    different function.
    """
    logits = torch.zeros(1, CELL * CELL + 1, GRID, GRID)
    logits[0, channel, cell_row, cell_col] = value
    return logits


# ══════════════════════════════════════════════════════════════════════════════
#  1. placement
# ══════════════════════════════════════════════════════════════════════════════

def t_channel_decodes_to_row_major_within_the_cell():
    """Channel c lands at (dy, dx) = (c // stride, c % stride).

    Read off the transcription: after `reshape(b, h, w, s, s)` the 4th axis is
    the channel's high digit and the 5th is its low digit, and the second permute
    interleaves the high one with `h`. So the high digit is the ROW.
    """
    for channel in (0, 1, CELL, CELL + 1, 13, 63):
        prob = depth_to_space_prob(_logits(channel, 1, 2), CELL)[0]
        flat = int(prob.argmax())
        row, col = divmod(flat, prob.shape[1])
        assert (row, col) == (1 * CELL + channel // CELL,
                              2 * CELL + channel % CELL), \
            f'channel {channel} decoded to ({row}, {col})'
    return '6 channels, cell (1, 2)'


def t_the_transposed_convention_is_a_different_pixel():
    """THE DECOY. Swapping the two permutes must move the peak.

    Only an ASYMMETRIC channel can see this: channel 9 is (1, 1) under both
    conventions, and a test that happened to use it would pass on either. So the
    check picks channels whose two digits differ and states that requirement.
    """
    for channel in (1, CELL, 13, 62):
        assert channel // CELL != channel % CELL, (
            f'channel {channel} is diagonal, so it decodes to the same pixel '
            f'under both conventions and cannot separate them')
        prob = depth_to_space_prob(_logits(channel, 0, 0), CELL)[0]
        row, col = divmod(int(prob.argmax()), prob.shape[1])
        transposed = (channel % CELL, channel // CELL)
        assert (row, col) != transposed, (
            f'channel {channel} decoded to {(row, col)}, which is the '
            f'TRANSPOSED convention -- every point is mirrored inside its cell')
    return '4 asymmetric channels'


def t_cells_do_not_leak_into_each_other():
    """One hot cell leaves every other cell near uniform.

    A reshape that mixes the cell axes with the within-cell axes would spread
    one cell's mass across the map -- which looks like a smooth detector rather
    than a broken decode.
    """
    prob = depth_to_space_prob(_logits(0, 1, 1), CELL)[0]
    peak_cell = prob[CELL:2 * CELL, CELL:2 * CELL]
    assert peak_cell.max() > 0.9, float(peak_cell.max())
    outside = prob.clone()
    outside[CELL:2 * CELL, CELL:2 * CELL] = 0
    assert float(outside.max()) < 0.05, float(outside.max())
    return f'peak {float(peak_cell.max()):.3f} vs elsewhere {float(outside.max()):.3f}'


def t_shape_is_the_cell_grid_times_the_stride():
    prob = depth_to_space_prob(torch.zeros(2, CELL * CELL + 1, 5, 7), CELL)
    assert tuple(prob.shape) == (2, 5 * CELL, 7 * CELL), tuple(prob.shape)
    return f'{tuple(prob.shape)}'


# ══════════════════════════════════════════════════════════════════════════════
#  2. dustbin
# ══════════════════════════════════════════════════════════════════════════════

def t_dustbin_is_dropped_and_not_renormalised():
    """All the mass in the dustbin -> a near-zero map, not a uniform one.

    The renormalising version would divide by a vanishing remainder and hand
    back a cell summing to 1 -- a detector that fires confidently on empty
    glass, with no error anywhere.
    """
    prob = depth_to_space_prob(_logits(CELL * CELL, 0, 0), CELL)[0]
    cell = prob[:CELL, :CELL]
    assert float(cell.sum()) < 0.05, (
        f'the dustbin cell sums to {float(cell.sum()):.3f}; dropping the '
        f'dustbin must not be followed by a renormalisation')
    return f'dustbin cell sums to {float(cell.sum()):.4f}'


def t_a_cell_and_its_dustbin_sum_to_one():
    """The softmax is over the channel axis, so each CELL is one distribution.

    Stated from the other side of the same fact: what is dropped is a
    probability, so the decoded cell plus the dustbin has to come to 1. If the
    softmax were taken over the wrong axis this is what would move.
    """
    logits = torch.randn(1, CELL * CELL + 1, GRID, GRID)
    prob = depth_to_space_prob(logits, CELL)[0]
    dustbin = torch.softmax(logits, 1)[0, -1]
    for row in range(GRID):
        for col in range(GRID):
            cell = prob[row * CELL:(row + 1) * CELL,
                        col * CELL:(col + 1) * CELL]
            total = float(cell.sum()) + float(dustbin[row, col])
            assert abs(total - 1.0) < 1e-5, (row, col, total)
    return f'{GRID * GRID} cells within 1e-5 of 1'


# ══════════════════════════════════════════════════════════════════════════════
#  3. guards
# ══════════════════════════════════════════════════════════════════════════════

def t_wrong_channel_count_raises():
    """`stride**2 + 1` and not `stride**2`. Forgetting the dustbin gives a
    tensor that reshapes cleanly and decodes every value one channel early."""
    try:
        depth_to_space_prob(torch.zeros(1, CELL * CELL, GRID, GRID), CELL)
    except ValueError as e:
        return type(e).__name__ + f': {str(e)[:40]}...'
    raise AssertionError('a dustbin-less tensor did not raise')


# ══════════════════════════════════════════════════════════════════════════════

_SECTIONS = {
    'placement': ['t_channel_decodes_to_row_major_within_the_cell',
                  't_the_transposed_convention_is_a_different_pixel',
                  't_cells_do_not_leak_into_each_other',
                  't_shape_is_the_cell_grid_times_the_stride'],
    'dustbin':   ['t_dustbin_is_dropped_and_not_renormalised',
                  't_a_cell_and_its_dustbin_sum_to_one'],
    'guards':    ['t_wrong_channel_count_raises'],
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(_SECTIONS))
    args = ap.parse_args()

    torch.manual_seed(0)
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
