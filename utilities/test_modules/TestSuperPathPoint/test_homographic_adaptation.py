#!/usr/bin/env python3
"""Tests for training/SuperPathPoint/SuperPoint/HomographicAdaptation.py.

    python utilities/test_modules/test_homographic_adaptation.py

No weights, no GPU, no WSI. The teacher is a dozen lines at the top of this file
that answers "wherever this view is brightest", so the whole aggregation runs
in seconds on a 64 px tile.

WHY A FAKE DETECTOR AND NOT THE REAL ONE
-----------------------------------------
The property under test is a COORDINATE property: a point detected in a warped
view has to come back to where it was in the original frame. A real detector
answers something different in every view, so nothing about the answer says
where it should have landed. A detector that always answers "the bright dot"
turns the whole loop into a statement with a known truth: the dot was planted at
a known place, and the aggregate has to peak there.

WHAT WOULD RUN AND BE WRONG
----------------------------
    projecting with H instead of H_inv
        The result is a smooth, plausible probability map, offset by each
        homography. Averaged over N draws the offsets partly cancel, so it looks
        like a slightly blurry label rather than a bug. This is the failure that
        `Camera.output_to_level0` shipped with in a different form, invisible at
        0 and 180 degrees and fatal at 90 and 270.
    composing the pre-tile translation on the right
        `T @ H` changes the INPUT frame; `H @ T` shifts the OUTPUT frame, which
        warps a different part of the tile and looks entirely reasonable.
    building the mask with valid_mask(tile_shape, H)
        Correct when the source is the tile, and it is not: with a 3x pre-tile
        that call declares two thirds of a legitimate view invalid, and the only
        symptom is labels that thin out toward the edges.

Everything below is scored against a decoy rather than a tolerance, because a
tolerance on "is the peak in the right place" is a guess and the margin over a
shifted window is not.

Sections:
  1. direction -- the aggregate peaks where the dot was planted
  2. pretile   -- what the 3x source changes about the two masks
  3. counts    -- coverage is not detection
  4. identity  -- the teacher is part of the label's identity
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

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402

from common.Homography import invert, sample_homography, valid_mask  # noqa: E402
from PreTileStore import centre_margin, pre_tile_px        # noqa: E402
from SuperPoint.HomographicAdaptation import HaConfig             # noqa: E402

_RESULTS = []

TILE = 64
FACTOR = 3
NUM = 24            # enough for the aggregate to be a consensus, fast enough
                    # for this to stay a seconds-long test
DOT = (22, 41)      # (x, y) in the TILE frame. Off-diagonal on purpose: an
                    # (x, y)/(row, col) swap is invisible on the diagonal.


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ── stand-ins ────────────────────────────────────────────────────────────────

class BrightestPointTeacher:
    """Answers a Gaussian blob at the brightest pixel of each view.

    A blob and not a single hot pixel, because warping back is bilinear: a
    one-pixel answer spreads its mass over four pixels in the original frame,
    and after averaging that is a peak of a quarter the height sitting between
    pixels. The blob makes the aggregate's argmax a stable statement about
    where the dot was rather than about how the interpolation rounded.
    """
    device = torch.device('cpu')
    sigma = 1.5

    def identity_id(self):
        return 'faketeacher01'

    def dense_prob(self, views):
        maps = []
        for view in views:
            image = np.asarray(view)
            grey = image.mean(axis=2) if image.ndim == 3 else image
            flat = int(np.argmax(grey))
            y, x = divmod(flat, grey.shape[1])
            maps.append(_blob(grey.shape, x, y, self.sigma))
        return torch.from_numpy(np.stack(maps))


def _blob(shape, x, y, sigma):
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    return np.exp(-(((xs - x) ** 2 + (ys - y) ** 2) /
                    (2.0 * sigma ** 2))).astype(np.float32)


def _pre_tile_with_dot() -> np.ndarray:
    """A mid-grey pre-tile with one bright 3x3 square at `DOT` of the TILE.

    Mid-grey and not black so that "brightest pixel" is a statement about the
    dot rather than about the `BORDER_CONSTANT` fill -- and the fill is what
    the pre-tile exists to keep out of the frame, so a test that could not tell
    them apart would pass for the wrong reason.
    """
    pre = pre_tile_px(TILE, FACTOR)
    margin = centre_margin(TILE, FACTOR)
    image = np.full((pre, pre, 3), 110, np.uint8)
    # A little texture, so the argmax is not a tie between thousands of pixels.
    rng = np.random.default_rng(0)
    image = np.clip(image.astype(np.int16) +
                    rng.integers(-8, 9, image.shape), 0, 255).astype(np.uint8)
    x, y = margin + DOT[0], margin + DOT[1]
    image[y - 1:y + 2, x - 1:x + 2] = 255
    return image


def _run(num=NUM, seed=0):
    ha = HaConfig(num=num, batch=8).build(BrightestPointTeacher())
    return ha, ha.run(_pre_tile_with_dot(), TILE,
                      rng=np.random.default_rng(seed), factor=FACTOR)


def _window_mass(prob, x, y, half=3):
    """Mass of `prob` in a square window, clipped to the frame."""
    y0, y1 = max(0, y - half), min(prob.shape[0], y + half + 1)
    x0, x1 = max(0, x - half), min(prob.shape[1], x + half + 1)
    return float(prob[y0:y1, x0:x1].sum())


# ══════════════════════════════════════════════════════════════════════════════
#  1. direction
# ══════════════════════════════════════════════════════════════════════════════

def t_the_aggregate_peaks_where_the_dot_was_planted():
    _, result = _run()
    y, x = divmod(int(np.argmax(result.mean_prob)), TILE)
    distance = max(abs(x - DOT[0]), abs(y - DOT[1]))
    assert distance <= 2, (
        f'peak at ({x}, {y}), dot at {DOT} -- {distance} px away. A peak far '
        f'from the dot with a smooth map around it is what projecting with H '
        f'instead of H_inv looks like')
    return f'peak ({x}, {y}) vs dot {DOT}'


def t_it_beats_every_shifted_decoy_by_a_margin():
    """THE ONE THAT CARRIES THE SECTION.

    "The peak is within 2 px" would also pass on a map that is 2 px of signal in
    a sea of comparable noise. The margin over four windows one cell away is
    what says the aggregate concentrated rather than merely peaked.
    """
    _, result = _run()
    truth = _window_mass(result.mean_prob, *DOT)
    decoys = {(dx, dy): _window_mass(result.mean_prob, DOT[0] + dx, DOT[1] + dy)
              for dx, dy in ((8, 0), (-8, 0), (0, 8), (0, -8), (8, 8))}
    worst = max(decoys.values())
    assert truth > 5 * worst, (
        f'mass at the dot {truth:.2f}, best decoy {worst:.2f} -- only '
        f'{truth / max(worst, 1e-9):.1f}x. The aggregate did not concentrate')
    return f'{truth / max(worst, 1e-9):.0f}x the best of 5 decoys'


def t_two_seeds_agree_about_the_peak():
    """Different homographies, same answer. If the peak moved with the draw,
    the aggregate would be describing the sampler rather than the image."""
    peaks = []
    for seed in (0, 7):
        _, result = _run(seed=seed)
        peaks.append(divmod(int(np.argmax(result.mean_prob)), TILE))
    (ay, ax), (by, bx) = peaks
    assert max(abs(ax - bx), abs(ay - by)) <= 2, peaks
    return f'({ax}, {ay}) and ({bx}, {by})'


def t_mean_and_max_are_both_returned_and_differ():
    """Both aggregations cost nothing to keep and `num` forwards to recompute,
    so the result carries both. They must not be the same array -- if they are,
    the division by counts is not happening."""
    _, result = _run()
    assert result.mean_prob.shape == result.max_prob.shape
    assert not np.allclose(result.mean_prob, result.max_prob), (
        'mean and max are identical, so mean_prob was not divided by counts')
    assert result.aggregation == 'mean'
    assert np.array_equal(result.prob, result.mean_prob)
    return 'mean and max differ; prob follows the config'


# ══════════════════════════════════════════════════════════════════════════════
#  2. pretile
# ══════════════════════════════════════════════════════════════════════════════

def t_the_pretile_mask_is_full_where_the_tile_sized_one_is_not():
    """spec.md 6.6's free assertion, and its decoy in the same check.

    `_source_mask` warps the PRE-TILE's extent, so with factor 3 it is True
    everywhere but the eroded rim. `valid_mask(tile_shape, H)` -- the call that
    would be right if the source were the tile -- is not, and the gap between
    the two is the measurement of what the pre-tile bought.
    """
    ha, _ = _run(num=2)
    pre = _pre_tile_with_dot()
    margin = centre_margin(TILE, FACTOR)
    rng = np.random.default_rng(3)
    rim = int(ha.cfg.valid_border_margin)

    pre_full, tile_full = [], []
    for _ in range(12):
        sample = sample_homography((TILE, TILE), rng=rng,
                                   **ha.cfg.homography_kwargs())
        interior = slice(rim, TILE - rim)
        pre_mask = ha._source_mask(pre, sample.matrix, margin, (TILE, TILE))
        tile_mask = valid_mask((TILE, TILE), sample.matrix,
                               erosion_radius=rim)
        pre_full.append(float(pre_mask[interior, interior].mean()))
        tile_full.append(float(tile_mask[interior, interior].mean()))

    assert min(pre_full) > 0.999, (
        f'the worst pre-tile mask is {min(pre_full):.3f} valid inside the rim, '
        f'so at least one draw ran off a 3x pre-tile -- the factor is too small')
    assert np.mean(tile_full) < 0.9, (
        f'the tile-sized mask averages {np.mean(tile_full):.3f}, so this test '
        f'cannot see the difference the pre-tile makes')
    return (f'pre-tile {np.mean(pre_full):.3f} valid, tile-sized '
            f'{np.mean(tile_full):.3f}')


def t_a_non_square_pre_tile_raises():
    ha, _ = _run(num=2)
    try:
        ha.run(np.zeros((100, 120, 3), np.uint8), TILE)
    except ValueError as e:
        assert 'square' in str(e), str(e)
    else:
        raise AssertionError('a non-square pre-tile was accepted')

    try:
        ha.run(np.zeros((100, 100, 3), np.uint8), TILE)
    except ValueError as e:
        assert 'multiple' in str(e), str(e)
        return 'square-but-not-a-multiple refused too'
    raise AssertionError('a pre-tile that is not a whole multiple was accepted')


# ══════════════════════════════════════════════════════════════════════════════
#  3. counts
# ══════════════════════════════════════════════════════════════════════════════

def t_counts_start_at_one_and_thin_toward_the_edge():
    """The identity view covers everything, so nothing is ever zero; and a
    zoomed or rotated view does not reach the corners, so the edge is thinner
    than the middle. Those two together are why a zero in `mean_prob` needs
    `counts` beside it to be readable at all (spec.md 3.1)."""
    _, result = _run()
    assert result.counts.min() >= 1.0, float(result.counts.min())
    centre = result.counts[TILE // 4:3 * TILE // 4, TILE // 4:3 * TILE // 4]
    edge = np.concatenate([result.counts[0, :], result.counts[-1, :],
                           result.counts[:, 0], result.counts[:, -1]])
    assert centre.mean() > edge.mean(), (centre.mean(), edge.mean())
    return f'centre {centre.mean():.1f} views, edge {edge.mean():.1f}'


def t_filter_counts_zeroes_the_thin_places_and_is_off_by_default():
    """Off by default because `counts` is stored: any threshold can be applied
    later, and baking one in is the lossy version of the same thing."""
    assert HaConfig().filter_counts == 0
    ha = HaConfig(num=NUM, batch=8, filter_counts=NUM).build(
        BrightestPointTeacher())
    result = ha.run(_pre_tile_with_dot(), TILE,
                    rng=np.random.default_rng(0), factor=FACTOR)
    assert float(result.mean_prob[0, 0]) == 0.0, result.mean_prob[0, 0]
    return f'filter_counts={NUM} zeroes the corner'


# ══════════════════════════════════════════════════════════════════════════════
#  4. identity
# ══════════════════════════════════════════════════════════════════════════════

def t_the_teacher_is_part_of_the_labels_identity():
    """Round 2 of Stage A runs the same config against a different teacher. If
    that did not move the id, round 2's labels would overwrite round 1's."""
    teacher = BrightestPointTeacher()
    first = HaConfig(num=NUM).build(teacher).identity_id()

    other = BrightestPointTeacher()
    other.identity_id = lambda: 'faketeacher02'
    second = HaConfig(num=NUM).build(other).identity_id()
    assert first != second, 'a different teacher gave the same HA identity'

    assert 'teacher=faketeacher01' in HaConfig(num=NUM).build(
        teacher).identity_parts(), 'the teacher is not in the parts at all'
    return f'{first} vs {second}'


def t_num_is_identity_and_batch_is_not():
    teacher = BrightestPointTeacher()
    base = HaConfig(num=100).build(teacher).identity_id()
    assert HaConfig(num=50).build(teacher).identity_id() != base
    assert HaConfig(num=100, batch=1).build(teacher).identity_id() == base, (
        'batch moved the identity, so a throughput knob would split the cache')
    return 'num splits, batch does not'


def t_the_config_and_the_sampler_agree_about_their_options():
    """The import-time check in the module, restated so a failure here names it.

    `sample_homography` takes `**overrides`, so an option this config stopped
    naming would simply not be passed -- the sampler would run on its own
    default while the config said otherwise, silently.
    """
    from common.Homography import HOMOGRAPHY_DEFAULTS             # noqa: PLC0415
    kwargs = HaConfig().homography_kwargs()
    assert set(kwargs) == set(HOMOGRAPHY_DEFAULTS), (
        set(kwargs) ^ set(HOMOGRAPHY_DEFAULTS))
    sample_homography((TILE, TILE), rng=np.random.default_rng(0), **kwargs)
    return f'{len(kwargs)} options, accepted by the sampler'


# ══════════════════════════════════════════════════════════════════════════════

_SECTIONS = {
    'direction': ['t_the_aggregate_peaks_where_the_dot_was_planted',
                  't_it_beats_every_shifted_decoy_by_a_margin',
                  't_two_seeds_agree_about_the_peak',
                  't_mean_and_max_are_both_returned_and_differ'],
    'pretile':   ['t_the_pretile_mask_is_full_where_the_tile_sized_one_is_not',
                  't_a_non_square_pre_tile_raises'],
    'counts':    ['t_counts_start_at_one_and_thin_toward_the_edge',
                  't_filter_counts_zeroes_the_thin_places_and_is_off_by_default'],
    'identity':  ['t_the_teacher_is_part_of_the_labels_identity',
                  't_num_is_identity_and_batch_is_not',
                  't_the_config_and_the_sampler_agree_about_their_options'],
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
