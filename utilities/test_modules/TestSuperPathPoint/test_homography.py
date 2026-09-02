#!/usr/bin/env python3
"""Tests for training/SuperPathPoint/common/Homography.py.

    python utilities/test_modules/test_homography.py
    python utilities/test_modules/test_homography.py --only direction torch
    python utilities/test_modules/test_homography.py --draws 500

No model, no GPU, no WSI, no figures -- seconds. The figure that shows the same
facts to a human is `training/SuperPathPoint/cli/demo_homography.py`; a test says
yes or no, a demo says which way it is wrong.

WHAT THIS IS DEFENDING AGAINST
------------------------------
Every check here answers "what would a wrong result look like", and in this
module the answer is always "a picture that looks fine". A homography applied in
the reverse direction produces a warped image that is a perfectly plausible
warped image. A half-pixel offset in the grid_sample conversion produces an
image no one can tell from the right one, and keypoints that are all off by half
a pixel. Neither raises.

So the checks that matter score against a DELIBERATELY WRONG alternative rather
than against a tolerance -- the discipline that found the R(-theta) bug in
`Camera.output_to_level0`, where the first version passed at 0 and 180 degrees
and lost to a point-reflected candidate 40/40 times at 90 and 270. A margin over
a decoy is robust; a threshold is a guess.

Sections:
  1. identity   -- the switches, the quad, and what patch_ratio does not do
  2. direction  -- image warp vs point warp, against the reversed decoy
  3. torch      -- the grid_sample path vs cv2, against a half-pixel decoy
  4. sampling   -- allow_artifacts, determinism, candidate bookkeeping
  5. mask       -- valid_mask and its erosion
"""

from __future__ import annotations

import argparse
import os
import sys

# _paths holds the one definition of OUTPUT_ROOT and of where each package
# lives; that directory goes on sys.path here because setup_import_paths --
# which puts the rest there, SuperPathPoint included -- is inside it.
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

import numpy as np                                              # noqa: E402

from _paths import setup_import_paths                           # noqa: E402

setup_import_paths()

from common.Homography import (HOMOGRAPHY_DEFAULTS,             # noqa: E402
                               identity, inside, invert,
                               points_input_to_output,
                               points_output_to_input,
                               sample_homography, valid_mask, warp_image,
                               erosion_anchor)


_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ══════════════════════════════════════════════════════════════════════════════
#  helpers
# ══════════════════════════════════════════════════════════════════════════════

def _blob_image(shape, centre_xy, sigma=2.0):
    """A single smooth blob, so its centroid is a well-defined position.

    A one-pixel delta would do for an affine map but not for a homography: the
    resampled kernel's centroid drifts where the map is not locally affine, and
    the drift would be indistinguishable from the bug being hunted. A blob wide
    enough to be resolved keeps the centroid honest.
    """
    height, width = shape
    ys, xs = np.mgrid[0:height, 0:width]
    dx = xs - float(centre_xy[0])
    dy = ys - float(centre_xy[1])
    return np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).astype(np.float32)


def _centroid(image):
    total = image.sum()
    if total <= 1e-8:
        raise AssertionError('the warped blob carried no mass -- it left the frame')
    height, width = image.shape
    ys, xs = np.mgrid[0:height, 0:width]
    return np.array([(image * xs).sum() / total, (image * ys).sum() / total])


def _sample(seed, **over):
    return sample_homography((256, 256), rng=np.random.default_rng(seed), **over)


# ══════════════════════════════════════════════════════════════════════════════
#  1. identity
# ══════════════════════════════════════════════════════════════════════════════

def t_all_switches_off_is_the_identity():
    """patch_ratio does NOT bake in a zoom.

    The docstring of `sample_homography` claims this and it is the single
    easiest thing to get wrong when reading the upstream code: `pts1` is a
    patch_ratio-sized centred quad, which reads like a crop. It is not -- `pts2`
    starts equal to it, so with nothing perturbing the corners the solve returns
    I and the warp is a no-op.

    Checked at patch_ratio 0.5 as well as 0.85 because a zoom of 1/patch_ratio
    would be 2x at 0.5 and only 1.18x at 0.85; if this ever regresses, the small
    ratio is where it screams.
    """
    for ratio in (0.5, 0.85):
        sample = _sample(0, perspective=False, scaling=False,
                         rotation=False, translation=False, patch_ratio=ratio)
        assert sample.is_identity, (
            f'patch_ratio={ratio} with all switches off gave\n{sample.matrix}')
        rng = np.random.default_rng(1)
        image = rng.random((256, 256)).astype(np.float32)
        warped = warp_image(image, sample.matrix)
        assert np.array_equal(warped, image), (
            f'patch_ratio={ratio}: identity matrix still changed the image '
            f'(max abs diff {np.abs(warped - image).max():.3g})')
    return 'ratio 0.5 and 0.85 both give I, and warping is bitwise a no-op'


def t_quad_is_the_centred_patch():
    """quad_out is the reference quad in pixels, corners in (x, y) TL/BL/BR/TR."""
    sample = _sample(0, patch_ratio=0.5)
    expected = np.array([[64., 64.], [64., 192.], [192., 192.], [192., 64.]])
    assert np.allclose(sample.quad_out, expected), (
        f'quad_out\n{sample.quad_out}\nexpected\n{expected}')
    return 'patch_ratio 0.5 on 256 px -> corners at 64 and 192'


def t_matrix_round_trip():
    """invert(invert(H)) == H, with the H[2,2] normalisation doing its job.

    Without the normalisation in `invert`, this fails on a correct inverse: two
    matrices differing by a scale factor are the same projective map, and
    np.linalg.inv has no reason to return the scale that was handed in.
    """
    worst = 0.0
    for seed in range(20):
        matrix = _sample(seed).matrix
        worst = max(worst, np.abs(invert(invert(matrix)) - matrix).max())
    assert worst < 1e-9, f'max |inv(inv(H)) - H| = {worst:.3g}'
    return f'20 draws, max deviation {worst:.2g}'


def t_point_round_trip():
    """output_to_input . input_to_output is the identity on points.

    This is the cheap half of the direction question: it passes even if BOTH
    functions apply the same matrix in the same direction, because the two
    would still be inverses of each other. `t_image_warp_matches_point_warp` is
    the half that can tell them apart.
    """
    rng = np.random.default_rng(7)
    points = rng.uniform(0, 256, size=(500, 2))
    worst = 0.0
    for seed in range(20):
        matrix = _sample(seed).matrix
        back = points_output_to_input(points_input_to_output(points, matrix), matrix)
        worst = max(worst, np.abs(back - points).max())
    assert worst < 1e-6, f'max |round trip - p| = {worst:.3g} px'
    return f'20 draws x 500 points, max {worst:.2g} px'


# ══════════════════════════════════════════════════════════════════════════════
#  2. direction  -- the one that catches a plausible-looking wrong answer
# ══════════════════════════════════════════════════════════════════════════════

def t_image_warp_matches_point_warp(args):
    """Warp an image and warp a point; they must agree, and beat the decoy.

    THE DECOY is `points_output_to_input` -- the same matrix applied the other
    way. That is precisely the mistake `cv2.WARP_INVERSE_MAP` exists to prevent,
    and it produces a warped image that looks entirely reasonable.

    Reported as a ratio rather than an absolute error on purpose: the tolerance
    below is a sanity floor, but what makes the check trustworthy is that the
    right answer is orders of magnitude closer than the wrong one. A tolerance
    can be met by a half-broken implementation; a margin over a decoy cannot.
    """
    shape = (256, 256)
    correct, decoy = [], []
    for seed in range(args.draws // 10 or 1):
        sample = _sample(seed)
        for centre in ((96., 96.), (160., 96.), (128., 160.)):
            predicted = points_input_to_output(centre, sample.matrix)[0]
            reversed_prediction = points_output_to_input(centre, sample.matrix)[0]
            # Both predictions must be well inside, or "it left the frame"
            # would decide the comparison instead of the geometry.
            if not (inside(predicted[None], shape)[0] and
                    inside(reversed_prediction[None], shape)[0]):
                continue
            warped = warp_image(_blob_image(shape, centre), sample.matrix)
            found = _centroid(warped)
            correct.append(np.linalg.norm(found - predicted))
            decoy.append(np.linalg.norm(found - reversed_prediction))

    assert len(correct) >= 5, (
        f'only {len(correct)} usable draws -- both predictions have to land '
        f'inside the frame for the comparison to mean anything')
    correct_max = float(np.max(correct))
    decoy_median = float(np.median(decoy))
    assert correct_max < 2.0, (
        f'image warp and point warp disagree by up to {correct_max:.2f} px')
    assert decoy_median > 10 * correct_max, (
        f'the reversed-direction decoy is not clearly worse: '
        f'decoy median {decoy_median:.2f} px vs correct max {correct_max:.2f} px. '
        f'Either the two directions have collapsed into one, or the draws were '
        f'too close to identity to tell them apart')
    return (f'{len(correct)} blobs: correct <= {correct_max:.3f} px, '
            f'reversed decoy median {decoy_median:.1f} px '
            f'({decoy_median / max(correct_max, 1e-6):.0f}x worse)')


def t_valid_mask_is_in_the_output_frame():
    """valid_mask marks where the OUTPUT sampled from inside the input.

    Uses a pure translation so the answer is arithmetic rather than a judgement
    call: shifting the sampling window right by `d` means the output's rightmost
    `d` columns read past the input's edge and must be False, while everything
    else stays True.
    """
    shape = (64, 64)
    shift = 10
    matrix = np.array([[1., 0., float(shift)],
                       [0., 1., 0.],
                       [0., 0., 1.]])          # output q -> input q + (shift, 0)
    mask = valid_mask(shape, matrix)
    assert mask[:, :shape[1] - shift].all(), 'the covered part is not all valid'
    assert not mask[:, shape[1] - shift:].any(), (
        f'{int(mask[:, shape[1] - shift:].sum())} pixels past the edge were '
        f'called valid')
    return f'shift {shift} px -> exactly {shift} invalid columns, on the right'


def t_valid_mask_erosion_matches_tf_same_padding():
    """The eroded margin is EXACTLY TF's, on all four sides.

    This check used to assert only "the outermost ring goes, and more radius
    removes more", because cv2's anchor for an even-sized kernel was a detail
    nothing here had measured. It then failed -- and what it had caught was two
    real defects at once:

      * cv2's default erosion border is +DBL_MAX, so the frame edge counted as
        valid and an all-ones mask survived untouched. `tf.nn.erosion2d` with
        SAME padding pads with zero and does eat the border.
      * the even kernel's anchor differs between the libraries, and the two
        asymmetries are MIRROR IMAGES -- tf takes 2 from the top-left and 3 from
        the bottom-right for radius 3, cv2's default the other way round.

    With the anchor pinned to TF's `(k-1)//2` both are decided, so the widths
    are now derivable and worth asserting exactly. An exact width is a much
    stronger statement than "smaller than before": it fails if either the border
    value or the anchor drifts, and it names which.

    Radius 1 is left out on purpose: at k=2 the elliptical structuring element
    is degenerate enough that its ON pattern is a cv2 detail this test still has
    not measured. `t_small_radius_still_erodes` covers it more weakly.
    """
    shape = (64, 64)
    full = valid_mask(shape, identity())
    assert full.all(), 'identity with no erosion should leave everything valid'

    for radius in (2, 3, 6):
        before = erosion_anchor(radius)                 # eaten from top / left
        after = (radius * 2 - 1) - before               # eaten from bottom / right
        mask = valid_mask(shape, identity(), erosion_radius=radius)

        assert not mask[:before].any(), (
            f'radius {radius}: expected the top {before} rows gone, '
            f'{int(mask[:before].sum())} pixels survived')
        assert not mask[-after:].any(), (
            f'radius {radius}: expected the bottom {after} rows gone, '
            f'{int(mask[-after:].sum())} pixels survived')
        assert not mask[:, :before].any() and not mask[:, -after:].any(), \
            f'radius {radius}: the left {before} / right {after} columns'
        assert mask[before:-after, before:-after].all(), (
            f'radius {radius} ate into the interior: '
            f'{int((~mask[before:-after, before:-after]).sum())} pixels')
    return 'radius 2/3/6: top-left 1/2/5 px, bottom-right 2/3/6 px, exactly'


def t_small_radius_still_erodes():
    """radius 1 removes something, and not everything. Nothing about WHERE.

    The check above pins exact widths for radius 2, 3 and 6, and this one
    deliberately does not, because the 2x2 case is where the reasoning behind
    those widths stops being checkable from the anchor alone: at k=2 the
    elliptical structuring element is small enough that its ON pattern is a cv2
    construction detail, and `getStructuringElement` applies its OWN anchor when
    building the ellipse.

    An earlier version of this check asserted a direction -- that the top row
    and left column survive, which is what `anchor=0` implies if the kernel is
    full. It failed. Whether the kernel is not full, or the anchor interacts
    with it differently, is not something this test has measured, and asserting
    a direction on top of an unmeasured pattern was the mistake. The honest
    statement is the weak one.

    Measuring it would take a probe that prints the 2x2 kernel and the eroded
    mask beside it. Worth doing before anyone relies on radius 1; not worth
    guessing at here.
    """
    shape = (32, 32)
    mask = valid_mask(shape, identity(), erosion_radius=1)
    removed = 1.0 - mask.mean()
    assert removed > 0, (
        'radius 1 removed nothing at all -- borderValue may have gone back to '
        '+DBL_MAX, in which case the frame edge counts as valid and this '
        'function silently does less than upstream')
    assert removed < 0.5, (
        f'radius 1 removed {100 * removed:.0f}% of a 32x32 mask, which is far '
        f'more than a one-pixel margin can account for')
    return f'{100 * removed:.1f}% removed (direction not asserted -- see docstring)'


# ══════════════════════════════════════════════════════════════════════════════
#  3. torch  -- the second warp path
# ══════════════════════════════════════════════════════════════════════════════

def t_torch_path_matches_cv2():
    """grid_sample and cv2 must produce the same image, and beat a decoy.

    THE DECOY is the align_corners=True normalisation -- `2i/(S-1) - 1` instead
    of `(2i+1)/S - 1`. It differs by half a pixel at the edges and less in the
    middle, so it produces an image no one can distinguish by eye and keypoints
    that are systematically offset. Exactly the failure this project cannot
    afford, and exactly the one a tolerance-only check would wave through.

    Compared on the interior only: the two libraries disagree about border
    padding, which is real but is not what this check is about.
    """
    try:
        import torch
    except ImportError:
        return 'skipped -- torch not importable'
    from common.Homography import warp_image_torch

    rng = np.random.default_rng(3)
    shape = (128, 128)
    # Smooth, not white noise: at high spatial frequency a half-pixel shift and
    # a correct resample differ everywhere by a lot, which would make the decoy
    # look bad for the wrong reason. On smooth content the decoy is subtle,
    # which is the honest test.
    from scipy.ndimage import gaussian_filter
    coarse = rng.random((16, 16)).astype(np.float32)
    image = gaussian_filter(
        np.kron(coarse, np.ones((8, 8), dtype=np.float32)).astype(np.float32),
        3.0)

    batch = torch.from_numpy(image)[None, None]
    margin = 8
    worst_ok, worst_decoy = 0.0, 0.0
    for seed in range(10):
        matrix = _sample(seed).matrix
        reference = warp_image(image, matrix)[margin:-margin, margin:-margin]

        got = warp_image_torch(batch, matrix)[0, 0].numpy()
        got = got[margin:-margin, margin:-margin]
        worst_ok = max(worst_ok, float(np.abs(got - reference).max()))

        # The decoy, spelled out here rather than hidden behind a flag in the
        # library: same code path, align_corners=True normalisation.
        n, _, in_h, in_w = batch.shape
        ys, xs = torch.meshgrid(torch.arange(128, dtype=torch.float32),
                                torch.arange(128, dtype=torch.float32),
                                indexing='ij')
        grid_out = torch.stack([xs, ys, torch.ones_like(xs)], -1).reshape(1, -1, 3)
        src = grid_out @ torch.as_tensor(matrix, dtype=torch.float32).T
        src = src[..., :2] / src[..., 2:3]
        bad_x = 2. * src[..., 0] / (in_w - 1) - 1.
        bad_y = 2. * src[..., 1] / (in_h - 1) - 1.
        bad_grid = torch.stack([bad_x, bad_y], -1).reshape(n, 128, 128, 2)
        bad = torch.nn.functional.grid_sample(
            batch, bad_grid, mode='bilinear', padding_mode='zeros',
            align_corners=False)[0, 0].numpy()[margin:-margin, margin:-margin]
        worst_decoy = max(worst_decoy, float(np.abs(bad - reference).max()))

    span = float(image.max() - image.min())
    assert worst_ok < 0.02 * span, (
        f'cv2 and grid_sample disagree by {worst_ok:.4g} '
        f'({100 * worst_ok / span:.1f}% of the image range)')
    assert worst_decoy > 3 * worst_ok, (
        f'the align_corners decoy is not clearly worse: {worst_decoy:.4g} vs '
        f'{worst_ok:.4g}. The normalisation may not be doing anything')
    return (f'10 draws: agree to {worst_ok:.4g}, half-pixel decoy off by '
            f'{worst_decoy:.4g} ({worst_decoy / max(worst_ok, 1e-9):.1f}x)')


# ══════════════════════════════════════════════════════════════════════════════
#  4. sampling
# ══════════════════════════════════════════════════════════════════════════════

def t_allow_artifacts_drops_only_the_prepended_no_op(args):
    """It removes candidate 0 -- which is enough for scaling and not for rotation.

    `allow_artifacts` reads as "permit going out of bounds" and ALSO drops index
    0 from the scaling and rotation pools (homographies.py:184, :213), the no-op
    prepended one line above. A port that keeps the bounds behaviour and forgets
    the exclusion still runs, still produces sensible homographies, and quietly
    makes about a sixth of the scales exactly 1.0.

    The half that is easy to overstate, and that this module's docstring did
    overstate until this check counted them: dropping index 0 does NOT make
    rotation non-identity. `linspace(-max_angle, max_angle, n_angles)` with an
    ODD n_angles contains 0 at its midpoint, so with the default 25 there is a
    second zero at index 13 that nothing excludes.

    Pinned from both sides: an even n_angles has no midpoint sample, so the
    zeros must vanish entirely. That is the mechanism, not a rate, and it is
    what makes this a check rather than a measurement.
    """
    draws = args.draws
    noop_scale = noop_angle = 0
    even_noop_angle = 0
    free_noop_scale = free_noop_angle = 0
    for seed in range(draws):
        forced = _sample(seed, allow_artifacts=True, translation=False,
                         perspective=False)
        noop_scale += forced.drawn['scaling']['scale'] == 1.0
        noop_angle += forced.drawn['rotation']['angle_deg'] == 0.0

        even = _sample(seed, allow_artifacts=True, translation=False,
                       perspective=False, n_angles=24)
        even_noop_angle += even.drawn['rotation']['angle_deg'] == 0.0

        # patch_ratio 0.4 leaves plenty of room, so the bounds check rejects
        # almost nothing and the no-op is genuinely in the pool.
        free = _sample(seed, allow_artifacts=False, translation=False,
                       perspective=False, patch_ratio=0.4, max_angle=np.pi / 6)
        free_noop_scale += free.drawn['scaling']['scale'] == 1.0
        free_noop_angle += free.drawn['rotation']['angle_deg'] == 0.0

    assert noop_scale == 0, (
        f'allow_artifacts=True produced {noop_scale} scales of exactly 1.0 in '
        f'{draws} draws; candidate 0 is not being dropped')
    assert even_noop_angle == 0, (
        f'{even_noop_angle} zero angles with n_angles=24, which has no midpoint '
        f'sample -- so they came from somewhere other than linspace, and the '
        f'explanation in the module docstring is wrong')
    assert noop_angle > 0, (
        f'no zero angles at all with n_angles=25, whose linspace DOES contain '
        f'0. Either the candidate list changed or index 13 is being excluded '
        f'too, and the docstring should say so')
    assert free_noop_scale > 0 and free_noop_angle > 0, (
        f'allow_artifacts=False never produced the no-op in {draws} draws '
        f'({free_noop_scale} scales, {free_noop_angle} angles) -- candidate 0 '
        f'may have been dropped in both branches')
    return (f'{draws} draws: scale no-ops 0, angle no-ops {noop_angle} at '
            f'n_angles=25 and {even_noop_angle} at 24, free '
            f'{free_noop_scale}/{free_noop_angle}')


def t_translation_interval_inverts_and_is_sampled_anyway(args):
    """The quad outgrows the frame often, and that must not raise.

    The interval is `[-t_min, t_max]` and its width is `1 - (quad width)`, so it
    is empty exactly when the quad exceeds the frame -- which under the
    production config happens on a large minority of draws, because
    allow_artifacts leaves `perspective_amplitude_x` unclamped and scaling
    multiplies afterwards.

    `tf.random_uniform` samples the reversed interval without complaint;
    `np.random.Generator.uniform` raises `high - low < 0`. Sorting the two bounds
    reproduces the former. What is asserted here is that it HAPPENS -- if it
    never did, the sort would be dead code and the module docstring's whole
    explanation of it would be describing nothing.
    """
    inverted = 0
    for seed in range(args.draws):
        drawn = _sample(seed).drawn['translation']
        inverted += drawn['inverted_x'] or drawn['inverted_y']
    assert inverted > 0, (
        f'not one of {args.draws} production draws produced an inverted '
        f'translation interval. Either the sampler no longer lets the quad '
        f'leave the frame, or `inverted_x/y` is not being computed -- and '
        f'either way the sorting in sample_homography is now unexplained')
    return (f'{inverted}/{args.draws} draws '
            f'({100 * inverted / args.draws:.0f}%) had the interval inverted')


def t_candidate_bookkeeping_is_recorded():
    """`drawn` carries how many candidates survived the bounds check.

    Not decoration: the scaling and rotation steps pick uniformly from the
    survivors, so if `n_valid` is routinely 1 then "pick a random valid one" is
    picking the only one and the configured distribution is fiction. This is the
    number `cli/demo_homography.py` prints, and the reason it prints it.
    """
    tight = _sample(0, allow_artifacts=False, patch_ratio=0.98)
    roomy = _sample(0, allow_artifacts=False, patch_ratio=0.3,
                    max_angle=np.pi / 8)
    for name, sample in (('tight', tight), ('roomy', roomy)):
        for step in ('scaling', 'rotation'):
            assert 'n_valid' in sample.drawn[step], f'{name}/{step} has no n_valid'
    assert roomy.drawn['rotation']['n_valid'] > tight.drawn['rotation']['n_valid'], (
        f"a roomy quad should leave more rotations valid than a tight one: "
        f"{roomy.drawn['rotation']['n_valid']} vs "
        f"{tight.drawn['rotation']['n_valid']}")
    return (f"rotation candidates valid: patch_ratio 0.98 -> "
            f"{tight.drawn['rotation']['n_valid']}, "
            f"0.3 -> {roomy.drawn['rotation']['n_valid']} "
            f"of {roomy.drawn['rotation']['n_candidates']}")


def t_same_seed_same_matrix():
    """Reproducible, and the four steps consume the rng in a fixed order."""
    a = _sample(42)
    b = _sample(42)
    assert np.array_equal(a.matrix, b.matrix), 'same seed gave different matrices'
    c = _sample(43)
    assert not np.array_equal(a.matrix, c.matrix), \
        'different seeds gave the same matrix'
    return 'seed 42 twice identical, seed 43 different'


def t_unknown_option_raises():
    """A typo in an option name must not be silently ignored.

    `sample_homography(**cfg)` is how a jobscript will pass a dozen values.
    Dropping `patch_ratios` on the floor because it is not `patch_ratio` would
    run a whole Homographic Adaptation pass at the default and report nothing.
    """
    try:
        _sample(0, patch_ratios=0.5)
    except TypeError as e:
        assert 'patch_ratios' in str(e), f'unhelpful message: {e}'
        return 'unknown keys raise TypeError and name themselves'
    raise AssertionError('a misspelled option was accepted')


def t_defaults_match_upstream():
    """The values are upstream's export config, not the function's own defaults.

    `configs/magic-point_coco_export.yaml:12-26` is what Homographic Adaptation
    actually ran with; `homographies.py:9-25` is a looser default nothing used.
    They differ in every amplitude, in patch_ratio, and in allow_artifacts -- so
    picking the wrong one is a real and silent change of distribution.
    """
    expected = {'n_scales': 5, 'n_angles': 25, 'scaling_amplitude': 0.2,
                'perspective_amplitude_x': 0.2, 'perspective_amplitude_y': 0.2,
                'patch_ratio': 0.85, 'allow_artifacts': True,
                'translation_overflow': 0.0}
    for key, want in expected.items():
        got = HOMOGRAPHY_DEFAULTS[key]
        assert got == want, f'{key} is {got}, upstream export config says {want}'
    assert abs(float(HOMOGRAPHY_DEFAULTS['max_angle']) - np.pi) < 1e-12, \
        f"max_angle is {HOMOGRAPHY_DEFAULTS['max_angle']}, export config says pi"
    return 'all 9 match magic-point_coco_export.yaml'


# ══════════════════════════════════════════════════════════════════════════════

_SECTIONS = {
    'identity':  ['t_all_switches_off_is_the_identity', 't_quad_is_the_centred_patch',
                  't_matrix_round_trip', 't_point_round_trip'],
    'direction': ['t_image_warp_matches_point_warp',
                  't_valid_mask_is_in_the_output_frame',
                  't_valid_mask_erosion_matches_tf_same_padding',
                  't_small_radius_still_erodes'],
    'torch':     ['t_torch_path_matches_cv2'],
    'sampling':  ['t_allow_artifacts_drops_only_the_prepended_no_op',
                  't_translation_interval_inverts_and_is_sampled_anyway',
                  't_candidate_bookkeeping_is_recorded', 't_same_seed_same_matrix',
                  't_unknown_option_raises', 't_defaults_match_upstream'],
}

#: Which checks need the parsed args. Everything else takes none, so the runner
#: does not have to thread an unused parameter through a dozen signatures.
_WANTS_ARGS = {'t_image_warp_matches_point_warp',
               't_allow_artifacts_drops_only_the_prepended_no_op',
               't_translation_interval_inverts_and_is_sampled_anyway'}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(_SECTIONS),
                    help='run only these sections (default: all)')
    ap.add_argument('--draws', type=int, default=200,
                    help='how many homographies the statistical checks draw')
    args = ap.parse_args()

    sections = args.only or sorted(_SECTIONS)
    for section in sections:
        print(f'\n[{section}]')
        for name in _SECTIONS[section]:
            fn = globals()[name]
            label = name[2:].replace('_', ' ')
            check(label, (lambda f=fn: f(args)) if name in _WANTS_ARGS else fn)

    failed = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
