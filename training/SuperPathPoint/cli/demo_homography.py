#!/usr/bin/env python3
"""Every homography operation on its own, in one figure.

    python training/SuperPathPoint/cli/demo_homography.py
    python training/SuperPathPoint/cli/demo_homography.py --wsi slide.svs --ds 4
    python training/SuperPathPoint/cli/demo_homography.py --seed 7 --tile-size 512

No model, no GPU, seconds. Runs with no WSI at all -- it falls back to a
synthetic pattern -- so it is available before any data has been prepared.

Outputs (in result/<SLURM_JOB_NAME or SuperPathPointDemo>/):
    homography_operations.png

WHY THIS IS NOT SHAPED LIKE query_sim/cli/demo.py
--------------------------------------------------
That demo has one panel per augment because `query_sim/augment/` really is a
list of independent image functions: `apply_rotation`, `apply_vignette` and the
rest each take an image and return one.

A homography is not a chain of image operations. It is four perturbations
applied to the four corners of a quad, solved once into a single 3x3 matrix
(`common/Homography.py`). So "show one operation alone" means turning the other
three booleans off and sampling again -- which is what the panels below do, and
why each of them is a DIFFERENT random draw rather than a step in one sequence.

THE TWO PANELS THAT ARE NOT DECORATION
---------------------------------------
`all off` must be pixel-identical to `original`. It is there because
`patch_ratio` reads like a crop and is not one; anyone who expects a 1/0.85 zoom
should see nothing happen and go read the module docstring.

`point-warp check` is the only panel that can show a wrong answer. Bright
markers are drawn into the image, the image is warped, and the crosses are
plotted where `points_input_to_output` says those markers went. If crosses and
markers coincide, the image path and the point path agree. If they are
systematically displaced, the two disagree -- and the displacement pattern says
which way: a mirror-like scatter means the direction is reversed, a uniform
half-pixel drift means the grid_sample normalisation.

`test_homography.py` asserts both facts. This gives the human the direction of
the error, which a passing or failing assertion cannot.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # training/SuperPathPoint/

import numpy as np                                              # noqa: E402
import matplotlib                                               # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                 # noqa: E402

from cli import job_result_dir, setup_import_paths              # noqa: E402

setup_import_paths()

from common.Homography import (invert, points_input_to_output,  # noqa: E402
                               points_output_to_input,
                               sample_homography, valid_mask, warp_image)
from common.HomographyConfig import HomographyConfig             # noqa: E402
from PreTileStore import (PRE_TILE_FACTOR, centre_margin,  # noqa: E402
                                 pre_tile_px, pretile_valid_mask)

#: The four switches, in the order `sample_homography` applies them. Order
#: matters and is not commutative, so the panels are laid out in it.
_STEPS = ('perspective', 'scaling', 'translation', 'rotation')


# ── the image to warp ─────────────────────────────────────────────────────────

def synthetic_tile(size: int) -> np.ndarray:
    """A pattern with structure at several scales and no rotational symmetry.

    Symmetry is the enemy of a geometry demo: on a checkerboard a 90-degree
    rotation and the identity look the same, and a sign error hides. The
    diagonal gradient and the off-centre disc break both mirror and rotational
    symmetry, so every panel is distinguishable from every other by eye.
    """
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    u, v = xs / size, ys / size

    board = ((xs // (size // 16) + ys // (size // 16)) % 2).astype(np.float32)
    fine = ((xs // (size // 64) + ys // (size // 64)) % 2).astype(np.float32)
    image = 0.35 * board + 0.15 * fine + 0.30 * (u + v) / 2.0

    disc = ((u - 0.32) ** 2 + (v - 0.62) ** 2) < 0.14 ** 2
    image[disc] = 0.95
    bar = (np.abs(v - 0.25 * u - 0.20) < 0.012)
    image[bar] = 0.05

    rgb = np.stack([image, image * 0.85 + 0.10, image * 0.70 + 0.20], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def wsi_tile(path: str, ds: float, size: int, seed: int) -> np.ndarray:
    """One tile at the requested rung, through the ladder that will feed training.

    Uses `read_region_rgb`, never `.convert('RGB')`: the latter merely drops
    alpha, and unphotographed pixels carry RGB 0, so every scanner hole becomes
    a black rectangle. A black rectangle's border is a perfect corner, which is
    the last thing a keypoint demo should be showing (spec.md 6.1).
    """
    from SafeSlide import SafeSlide
    from DsLadder import DsLadder

    wsi = SafeSlide(path)
    try:
        plan = next(p for p in DsLadder(rungs=(float(ds),)).plan_for(wsi, size))
        print(f'  {plan.summary()}')
        width, height = wsi.level_dimensions[0]
        span = plan.footprint_l0
        rng = np.random.default_rng(seed)
        origin = (int(rng.integers(0, max(1, int(width - span)))),
                  int(rng.integers(0, max(1, int(height - span)))))
        tile = wsi.read_region_rgb(origin, plan.level,
                                   (plan.read_size, plan.read_size))
        print(f'  read at level-0 {origin}, {plan.read_size} px of level '
              f'{plan.level}')
    finally:
        wsi.close()

    if plan.read_size != size:
        import cv2
        # INTER_AREA, not the default bilinear: this is a downsample, and area
        # averaging is the one that does not alias. Aliasing here would put
        # high-frequency artefacts into the very texture the detector learns on.
        tile = cv2.resize(tile, (size, size), interpolation=cv2.INTER_AREA)
    return tile


# ── panels ────────────────────────────────────────────────────────────────────

def _only(step: str, shape, seed: int):
    """A sample with exactly one of the four switches on."""
    off = {other: False for other in _STEPS if other != step}
    return sample_homography(shape, rng=np.random.default_rng(seed), **off)


def _draw_quad(axis, quad, colour):
    """Close the (4, 2) quad and draw it. `quad_polygon` in the library takes a
    HomographySample and picks one of its two quads; here the caller has already
    chosen, so closing four points is the whole job."""
    closed = np.concatenate([quad, quad[:1]], axis=0)
    axis.plot(closed[:, 0], closed[:, 1], color=colour, linewidth=1.6,
              linestyle='--')


def _marker_grid(shape, spacing: int) -> np.ndarray:
    """(N, 2) grid of (x, y), inset so no marker starts on the border."""
    height, width = shape
    xs = np.arange(spacing, width - spacing + 1, spacing)
    ys = np.arange(spacing, height - spacing + 1, spacing)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return grid.astype(np.float64)


def _stamp_markers(image: np.ndarray, points, radius: int = 3) -> np.ndarray:
    """Bright dots burned into a copy, so the warp carries them along."""
    import cv2
    out = image.copy()
    for x, y in points:
        cv2.circle(out, (int(round(x)), int(round(y))), radius,
                   (255, 30, 30), thickness=-1, lineType=cv2.LINE_AA)
    return out


# ── the parameter printout ────────────────────────────────────────────────────

def print_drawn(sample) -> None:
    """What was sampled, beside the range it came from.

    Modelled on `query_sim/cli/demo.py::_print_capture_params`, with one
    addition that matters here: the CANDIDATE COUNTS. Scaling and rotation pick
    uniformly from whichever candidates survived a bounds check, so `n_valid` is
    the only visible evidence that the pick was a pick. If it reads 1 draw after
    draw, "random valid scale" is returning the only scale there was, and the
    configured distribution is not the one running.
    """
    drawn = sample.drawn
    print('\n  sampled this draw            | drawn from')
    print(f"  patch_ratio  {drawn['patch_ratio']:.3f}            "
          f"| margin {drawn['margin']:.3f} on each side")
    print(f"  allow_artifacts  {str(drawn['allow_artifacts']):<5}       "
          f"| when True, the no-op candidate is excluded from scaling+rotation")

    if 'perspective' in drawn:
        p = drawn['perspective']
        print(f"  perspective  h_left {p['h_left']:+.4f}  h_right "
              f"{p['h_right']:+.4f}  v {p['v_disp']:+.4f}"
              f"   | TN(0, amp/2), amp_x {p['amp_x']}, amp_y {p['amp_y']}")
    if 'scaling' in drawn:
        s = drawn['scaling']
        print(f"  scaling      {s['scale']:.4f}              "
              f"| TN(1, {s['stddev']:.3f}), {s['n_valid']} of "
              f"{s['n_candidates']} candidates valid")
    if 'translation' in drawn:
        t = drawn['translation']
        inverted = [axis for axis in ('x', 'y') if t[f'inverted_{axis}']]
        note = f"  [interval INVERTED on {'+'.join(inverted)}]" if inverted else ''
        print(f"  translation  dx {t['dx']:+.4f}  dy {t['dy']:+.4f}   "
              f"| slack to edges "
              f"{[round(v, 3) for v in t['slack_min']]} / "
              f"{[round(v, 3) for v in t['slack_max']]}{note}")
        if inverted:
            print(f"      the quad is wider than the frame on {'+'.join(inverted)}, "
                  f"so the two bounds are the wrong way round and the sampled "
                  f"interval runs between them. tf.random_uniform does the same "
                  f"without saying so; numpy raises unless they are sorted.")
    if 'rotation' in drawn:
        r = drawn['rotation']
        print(f"  rotation     {r['angle_deg']:+.2f} deg           "
              f"| linspace(+/-{r['max_angle_deg']:.0f} deg, "
              f"{r['n_candidates'] - 1}), {r['n_valid']} valid")

    for step in ('scaling', 'rotation'):
        if step in drawn and drawn[step]['n_valid'] <= 1:
            print(f"  [!] {step}: only {drawn[step]['n_valid']} candidate "
                  f"survived the bounds check -- the random pick had no choice. "
                  f"Lower patch_ratio or turn allow_artifacts on.")


# ── main ──────────────────────────────────────────────────────────────────────

def calibrate(tile: int, draws: int, seed: int,
              factor: int = PRE_TILE_FACTOR) -> int:
    """Is `PRE_TILE_FACTOR` big enough? A CHECK, not a calibration. spec.md 6.6.

    The worst case is DERIVED, not measured -- 0.85 -> perspective 1.25 ->
    scaling 1.50 -> rotation 2.12 -> frame/quad 2.49, and 3 is that with the
    slack the derivation's one approximation needs (a projective map is not
    exactly a scaling about the centre). So there is no calibration run to do.
    This turns the derivation into a measured statement, at a cost of seconds,
    and that is worth having because the failure it guards is silent: a draw
    that runs off the pre-tile produces a warped view with a black wedge, and a
    black wedge is a straight maximum-contrast edge with two right angles --
    exactly what a corner detector fires on. The label would be confident and
    wrong.

    TWO COMPUTATIONS OF ONE FACT, WHICH IS THE POINT
    --------------------------------------------------
    `needed` is analytic: push the output frame's four corners back through the
    matrix and ask how far from the centre they reach. A homography maps lines
    to lines, so the image of the square IS the quadrilateral through those four
    points and its extremes are at the vertices -- exact, not a bound.

    `ran_off` is empirical: warp an all-ones array of the pre-tile's own extent
    through `translate(margin) @ H` and look for a False in the output. That is
    what `HomographicAdaptation` actually does, through the same function.

    They are computed by different lines from different quantities, so their
    AGREEING is evidence that both are right. A tolerance would not be: this is
    the same discipline as `inspect_feature_store --pairs`.
    """
    margin = centre_margin(tile, factor)
    pre = pre_tile_px(tile, factor)
    dummy = np.empty((pre, pre), np.uint8)
    corners = np.array([[0, 0], [tile, 0], [tile, tile], [0, tile]], np.float64)
    centre = tile / 2.0

    print(f'--calibrate  tile {tile}  factor {factor}  pre-tile {pre}  '
          f'margin {margin}  draws {draws}')
    print(f'  the same {len(HomographyConfig().kwargs())} sampler options '
          f'Homographic Adaptation draws its views with')

    rng = np.random.default_rng(seed)
    needed = np.empty(draws, np.float64)
    ran_off = np.zeros(draws, bool)
    worst = None
    for i in range(draws):
        sample = sample_homography((tile, tile), rng=rng,
                                   **HomographyConfig().kwargs())
        source = points_output_to_input(corners, sample.matrix)
        reach = float(np.abs(source - centre).max())
        needed[i] = 2.0 * reach / tile
        ran_off[i] = not pretile_valid_mask(
            dummy, sample.matrix, margin, (tile, tile), erosion_radius=0).all()
        if worst is None or needed[i] > needed[worst[0]]:
            worst = (i, sample)

    q = np.percentile(needed, [50, 90, 99, 100])
    print(f'\n  factor needed:  median {q[0]:.3f}   p90 {q[1]:.3f}   '
          f'p99 {q[2]:.3f}   max {q[3]:.3f}')
    _histogram(needed, factor)

    # The two computations against each other. `needed > factor` and "the warp
    # ran off the pre-tile" are the same event, reached two different ways.
    predicted = needed > factor
    agree = int((predicted == ran_off).sum())
    print(f'\n  analytic vs measured: {agree}/{draws} draws agree '
          f'({int(predicted.sum())} predicted over, {int(ran_off.sum())} '
          f'actually ran off the pre-tile)')

    ok = True
    if agree != draws:
        ok = False
        bad = int(np.argmax(predicted != ran_off))
        print(f'  FAIL  they disagree, first at draw {bad}: needed '
              f'{needed[bad]:.3f}, ran_off {bool(ran_off[bad])}. One of the two '
              f'is wrong and the analytic one is the cheaper to re-derive')
    if q[3] > factor:
        ok = False
        print(f'  FAIL  the worst draw needs {q[3]:.3f} and the store is cut '
              f'at {factor}. Raise PRE_TILE_FACTOR (it is an identity field, so '
              f'the pre-tiles have to be re-cut) or narrow the sampler')
        print('\n  the worst draw:')
        print_drawn(worst[1])
    else:
        print(f'  ok    the worst of {draws} draws needs {q[3]:.3f}, under '
              f'{factor}, with {factor - q[3]:.3f} to spare')

    # The margin over the decoy, rather than over the threshold: how much
    # smaller a pre-tile the SAME draws would have survived.
    tight = float(np.ceil(q[3] * 100) / 100)
    print(f'  the smallest factor these {draws} draws would have survived is '
          f'{tight:g}; {factor} is the derivation, not this number')
    return 0 if ok else 1


def _histogram(values: np.ndarray, factor: int, bins: int = 20,
               width: int = 48) -> None:
    """A text histogram, because this runs in a SLURM log and not in a notebook."""
    lo, hi = float(values.min()), max(float(values.max()), float(factor))
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    peak = max(int(counts.max()), 1)
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        bar = '#' * int(round(width * count / peak))
        flag = '  <- factor' if left <= factor < right else ''
        print(f'    {left:5.3f}-{right:5.3f} {count:5d} |{bar}{flag}')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wsi', help='WSI to take a tile from; omit for a '
                                  'synthetic pattern')
    ap.add_argument('--ds', type=float, default=4.0,
                    help='ladder rung to read the tile at (--wsi only)')
    ap.add_argument('--tile-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--erosion', type=int,
                    default=3, help='valid_border_margin; upstream export '
                                    'config uses 3')
    ap.add_argument('--marker-spacing', type=int, default=48)
    ap.add_argument('--out', help='output PNG (default: under result/<job>/)')
    ap.add_argument('--calibrate', type=int, metavar='N',
                    help='draw N homographies, print the distribution of the '
                         'pre-tile factor they need, and assert none exceeds '
                         'PRE_TILE_FACTOR. Prints no figure and reads no slide')
    ap.add_argument('--factor', type=int, default=PRE_TILE_FACTOR,
                    help='the factor --calibrate checks against')
    args = ap.parse_args()

    size = args.tile_size
    if args.calibrate:
        return calibrate(size, args.calibrate, args.seed, args.factor)
    if size % 8:
        print(f'[warn] tile_size {size} is not a multiple of 8; the detector '
              f'head decodes on an 8 px cell grid and will refuse it later')

    if args.wsi:
        print(f'Reading a tile from {os.path.basename(args.wsi)} at ds {args.ds}:')
        image = wsi_tile(args.wsi, args.ds, size, args.seed)
        source = f'{os.path.basename(args.wsi)}  ds {args.ds:g}'
    else:
        image = synthetic_tile(size)
        source = 'synthetic pattern'
    shape = image.shape[:2]

    production = sample_homography(shape, rng=np.random.default_rng(args.seed))
    all_off = sample_homography(shape, rng=np.random.default_rng(args.seed),
                                **{s: False for s in _STEPS})

    print(f'\nproduction draw (seed {args.seed}), all four steps on:')
    print_drawn(production)

    # ── the marker panel's three ingredients ────────────────────────────────
    markers = _marker_grid(shape, args.marker_spacing)
    stamped = _stamp_markers(image, markers)
    stamped_warped = warp_image(stamped, production.matrix)
    predicted = points_input_to_output(markers, production.matrix)

    panels = [
        ('original', image, production.quad_in, 'tab:orange'),
        ('all off  (identical to original)', warp_image(image, all_off.matrix),
         None, None),
        ('perspective only', warp_image(image, _only('perspective', shape,
                                                     args.seed).matrix), None, None),
        ('scaling only', warp_image(image, _only('scaling', shape,
                                                 args.seed).matrix), None, None),
        ('translation only', warp_image(image, _only('translation', shape,
                                                     args.seed).matrix), None, None),
        ('rotation only', warp_image(image, _only('rotation', shape,
                                                  args.seed).matrix), None, None),
        ('all on  (production)', warp_image(image, production.matrix),
         production.quad_out, 'tab:cyan'),
    ]

    identical = np.array_equal(panels[1][1], image)
    if not identical:
        print('\n[!] "all off" is NOT identical to the original. The four '
              'switches off must give the identity matrix -- see '
              'common/Homography.py, "WHAT patch_ratio DOES NOT DO".')

    mask = valid_mask(shape, production.matrix, erosion_radius=args.erosion)
    coverage = valid_mask(shape, invert(production.matrix),
                          erosion_radius=args.erosion)

    n_cols = 5
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.6, n_rows * 3.9))
    flat = np.atleast_2d(axes).flatten()

    for axis, (title, panel, quad, colour) in zip(flat, panels):
        axis.imshow(panel)
        if quad is not None:
            _draw_quad(axis, quad, colour)
        axis.set_title(title, fontsize=9)
        axis.axis('off')

    axis = flat[7]
    axis.imshow(mask, cmap='gray', vmin=0, vmax=1)
    axis.set_title(f'valid mask  (erosion {args.erosion})\n'
                   f'{100 * mask.mean():.1f}% valid', fontsize=9)
    axis.axis('off')

    axis = flat[8]
    axis.imshow(coverage, cmap='gray', vmin=0, vmax=1)
    axis.set_title(f'coverage of the warp-back\n{100 * coverage.mean():.1f}% covered',
                   fontsize=9)
    axis.axis('off')

    axis = flat[9]
    axis.imshow(stamped_warped)
    keep = ((predicted[:, 0] >= 0) & (predicted[:, 0] < shape[1]) &
            (predicted[:, 1] >= 0) & (predicted[:, 1] < shape[0]))
    axis.plot(predicted[keep, 0], predicted[keep, 1], 'x', color='lime',
              markersize=7, markeredgewidth=1.4)
    axis.set_xlim(0, shape[1])
    axis.set_ylim(shape[0], 0)
    axis.set_title(f'point-warp check\ncrosses must sit on the dots '
                   f'({int(keep.sum())}/{len(markers)} in frame)', fontsize=9)
    axis.axis('off')

    fig.suptitle(
        f'Homography sampling, one operation per panel  --  {source}, '
        f'{size} px, seed {args.seed}\n'
        f'each panel is its own draw: a homography is four corner '
        f'perturbations solved into one matrix, not a chain of image ops',
        fontsize=11)
    fig.text(0.5, 0.005,
             'matrix maps OUTPUT -> INPUT (upstream convention); '
             'warp_image sets cv2.WARP_INVERSE_MAP for that reason.   '
             'allow_artifacts also DROPS the no-op candidate from scaling and '
             'rotation, so both are forced to be non-identity.',
             ha='center', fontsize=8, color='0.35')
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))

    out_path = args.out or os.path.join(
        job_result_dir('SuperPathPointDemo'), 'homography_operations.png')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved  {out_path}')
    print(f'  all-off identical to original: {identical}')
    print(f'  valid {100 * mask.mean():.1f}%, coverage {100 * coverage.mean():.1f}%')
    return 0


if __name__ == '__main__':
    sys.exit(main())
