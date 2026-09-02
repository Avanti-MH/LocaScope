"""Random homographies, and the two direction conventions that decide everything.

Ported from rpautrat/SuperPoint `superpoint/models/homographies.py`, which is
TensorFlow. The numbers are the upstream ones and are cited line by line; what
changes here is the coordinate order and the naming of directions, both
deliberately, both explained below.

TWO CONVENTIONS, AND WHY THEY ARE SPELLED OUT IN THE FUNCTION NAMES
-------------------------------------------------------------------
1. THE MATRIX MAPS OUTPUT TO INPUT.

   Upstream's docstring (homographies.py:124-126): "As in
   `tf.contrib.image.transform`, it maps the output point (warped patch) to a
   transformed input point (original patch)."

   `cv2.warpPerspective` is the other way round by default: it treats M as
   input -> output and inverts it internally. Passing this matrix without
   `cv2.WARP_INVERSE_MAP` therefore applies the inverse warp, silently. That is
   why `warp_image` sets the flag and `test_homography` pins it against a decoy.

   This project has been bitten by exactly this class of error before:
   `Camera.output_to_level0` used R(-theta) where R(+theta) was correct, which
   is invisible at 0 and 180 degrees and fatal at 90 and 270. The fix there came
   from scoring against a point-reflected candidate, not from reading docs. Same
   discipline here.

2. POINTS ARE (x, y), NOT (y, x).

   Upstream's `warp_points` (homographies.py:280) takes (y, x) -- it does
   `points[:, ::-1]` on the way in and again on the way out, because its
   keypoint maps come from `tf.scatter_nd` in image-index order.

   LocaScope is (x, y) everywhere: `utilities/README.md`'s coordinate table,
   `PatchInfo.x/y`, `TissueRegion.x/y`, `Camera.output_to_level0(x, y, u, v)`.
   Carrying upstream's order into this repo would put two conventions in one
   codebase, so the order is converted HERE, once, and never again.

   Upstream also folds the inversion into `warp_points` ("Warp a list of points
   with the INVERSE of the given homography"), so one function name covers one
   direction and the other direction has no name at all. Here both directions
   are named, after `Camera.output_to_level0`:

       points_input_to_output(pts, H)   where does this original pixel land?
       points_output_to_input(pts, H)   where did this warped pixel come from?

   `points_output_to_input` is the one that applies H directly; the other
   inverts. If a call site cannot say which of the two it wants, it does not yet
   know what it is computing.

WHAT `patch_ratio` DOES NOT DO
------------------------------
It does not bake in a zoom. With all four switches off, `quad_in == quad_out`
and the matrix is the identity -- `sample_homography(..., perspective=False,
scaling=False, rotation=False, translation=False)` returns I and warping is a
no-op. What it controls is how much room the four corners have to move inside
[0, 1]: `margin = (1 - patch_ratio) / 2`. Upstream's export config uses 0.85,
which leaves almost none, which is why it must also set `allow_artifacts`.

WHAT `allow_artifacts` DOES BESIDES ALLOWING ARTIFACTS
------------------------------------------------------
In the scaling and rotation branches it also drops candidate 0 from the list
(homographies.py:184, :213) -- and candidate 0 is the no-op that was prepended
on the line above. So it does not merely permit going out of bounds, it also
removes the option of not moving.

BUT ONLY THE PREPENDED ONE, which matters for rotation and not for scaling:

    scaling    the other candidates are TN(1, amp/2) draws, never exactly 1.0,
               so dropping candidate 0 does leave every scale non-identity.
    rotation   `linspace(-max_angle, max_angle, n_angles)` with an ODD n_angles
               contains 0 at its midpoint. n_angles is 25, so index 13 of the
               26 candidates is another 0 degrees, and dropping index 0 does
               not remove it. About 1 in 25 draws still comes out unrotated --
               measured at 9 in 200.

This module said "FORCES ... a non-zero angle on every draw" until
`test_homography` counted them. Upstream has the same property and the same
`tf.range(1, n_angles + 1)`; what was wrong was the sentence, not the port.
Setting n_angles EVEN removes the midpoint zero, which is the mechanism the
test now pins from both sides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

#: Upstream's export configuration, `configs/magic-point_coco_export.yaml:12-26`.
#: These are the values Homographic Adaptation runs with, not the looser
#: `homography_adaptation_default_config` at homographies.py:9-25.
#:
#: `max_angle` is pi in that config; the function signature's own default is
#: pi/2 (homographies.py:119). The config wins, because the config is what ran.
HOMOGRAPHY_DEFAULTS: Dict[str, object] = {
    'perspective':             True,
    'scaling':                 True,
    'rotation':                True,
    'translation':             True,
    'n_scales':                5,
    'n_angles':                25,
    'scaling_amplitude':       0.2,
    'perspective_amplitude_x': 0.2,
    'perspective_amplitude_y': 0.2,
    'patch_ratio':             0.85,
    'max_angle':               np.pi,
    'allow_artifacts':         True,
    'translation_overflow':    0.0,
}

#: Corner order, in (x, y) and in units of patch_ratio before the margin is
#: added: top-left, bottom-left, bottom-right, top-right. Upstream builds the
#: same four at homographies.py:153-155; the order matters because the
#: perspective step below addresses corners by index.
_CORNERS = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], dtype=np.float64)

#: Anything this far outside the [0, 1] box still counts as inside it. Upstream
#: compares exactly (`(scaled >= 0.) & (scaled <= 1.)`, homographies.py:186); a
#: corner that lands on the boundary is then accepted or not depending on the
#: last bit of a float, which is not a decision anyone made.
#:
#: The tolerance can only ADD candidates to the pool, never change the geometry
#: of one, so setting it to 0 changes which draws were available and nothing
#: about any individual homography.
_INSIDE_TOL = 1e-9


# ── sampling ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HomographySample:
    """One sampled homography, plus everything that went into it.

    `drawn` is not decoration. Two of the four steps pick uniformly from a list
    of candidates that survived a bounds check, so the number that survived is
    the only visible evidence that the choice was a choice: if `n_valid` is
    routinely 1, "pick a random valid one" is picking the only one, and the
    distribution is not what the config says it is. `cli/demo_homography.py`
    prints it for that reason.
    """
    matrix:   np.ndarray                      # (3, 3) float64, OUTPUT -> INPUT
    quad_out: np.ndarray                      # (4, 2) (x, y) px, the reference quad
    quad_in:  np.ndarray                      # (4, 2) (x, y) px, the perturbed quad
    drawn:    Dict[str, object] = field(default_factory=dict)

    @property
    def is_identity(self) -> bool:
        return bool(np.allclose(self.matrix, np.eye(3), atol=1e-9))


def _truncated_normal(rng: np.random.Generator, mean: float, stddev: float,
                      size: Optional[int] = None) -> np.ndarray:
    """`tf.truncated_normal`: normal, resampled until within 2 stddev.

    Not `rng.normal` clipped -- clipping piles mass onto the two bounds, and the
    scaling step then picks those piled-up values disproportionately often.
    """
    shape = () if size is None else (size,)
    out = rng.normal(mean, stddev, shape)
    if stddev == 0:
        return out
    bad = np.abs(out - mean) > 2 * stddev
    while np.any(bad):
        out = np.where(bad, rng.normal(mean, stddev, shape), out)
        bad = np.abs(out - mean) > 2 * stddev
    return out


def _pick_valid(rng: np.random.Generator, candidates: np.ndarray,
                allow_artifacts: bool) -> Tuple[int, int, int]:
    """Index of a candidate quad that stays inside [0, 1], and how many did.

    Returns (chosen_index, n_valid, n_candidates).

    `candidates` is (K, 4, 2) with candidate 0 the no-op. When artifacts are
    allowed every candidate is valid EXCEPT that no-op, which is dropped -- see
    the module docstring. When they are not, the no-op stays in the pool and the
    bounds check decides the rest.
    """
    n = len(candidates)
    if allow_artifacts:
        valid = np.arange(1, n)
    else:
        inside = np.all(
            (candidates >= -_INSIDE_TOL) & (candidates <= 1. + _INSIDE_TOL),
            axis=(1, 2))
        valid = np.flatnonzero(inside)
    if len(valid) == 0:
        # Upstream would index an empty tensor here and die. Falling back to the
        # no-op is the only choice that keeps the sampler total, and it is
        # visible: n_valid == 0 lands in `drawn` and the demo prints it.
        return 0, 0, n
    return int(valid[rng.integers(len(valid))]), len(valid), n


def sample_homography(shape: Tuple[int, int], *,
                      rng: Optional[np.random.Generator] = None,
                      **overrides) -> HomographySample:
    """Sample one random homography for an image of `shape`.

    Args:
        shape: (height, width) in pixels -- the same order as `img.shape[:2]`.
               Upstream takes [H, W] too and flips it internally with
               `shape[::-1]` (homographies.py:219); that flip happens here at
               `_scale`, once.
        rng:   a `np.random.Generator`. Required for reproducibility in practice;
               defaults to a fresh default_rng so a demo can leave it out.
        **overrides: any key of HOMOGRAPHY_DEFAULTS.

    Returns:
        HomographySample whose `.matrix` maps OUTPUT coordinates to INPUT
        coordinates. See the module docstring before using it with cv2.

    The four perturbations are applied to the four corners of a centred quad, in
    this order and not commutatively: perspective, scaling, translation,
    rotation (homographies.py:159-215). Every step works in normalised [0, 1]
    coordinates; the scale to pixels happens once at the end.
    """
    if rng is None:
        rng = np.random.default_rng()
    cfg = dict(HOMOGRAPHY_DEFAULTS)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise TypeError(
            f'unknown homography options: {sorted(unknown)}. '
            f'Known: {sorted(cfg)}')
    cfg.update(overrides)

    height, width = int(shape[0]), int(shape[1])
    patch_ratio = float(cfg['patch_ratio'])
    margin = (1. - patch_ratio) / 2.

    quad_out = margin + _CORNERS * patch_ratio      # pts1 upstream
    quad_in = quad_out.copy()                       # pts2 upstream
    drawn: Dict[str, object] = {
        'patch_ratio':     patch_ratio,
        'margin':          margin,
        'allow_artifacts': bool(cfg['allow_artifacts']),
    }

    # ── 1. perspective (and the affine shear that rides along with it) ───────
    #
    # homographies.py:161-176. Three draws, applied so that the LEFT edge's two
    # corners move by +/- the same vertical displacement and the RIGHT edge's
    # two move by the opposite pair -- a trapezoid. The horizontal draws move
    # the left pair and the right pair independently, which is where the shear
    # comes from.
    if cfg['perspective']:
        amp_x = float(cfg['perspective_amplitude_x'])
        amp_y = float(cfg['perspective_amplitude_y'])
        if not cfg['allow_artifacts']:
            # homographies.py:163-165: clamp so the perturbed corner cannot
            # leave the image. With artifacts allowed the amplitude stands.
            amp_x = min(amp_x, margin)
            amp_y = min(amp_y, margin)
        disp_y = float(_truncated_normal(rng, 0., amp_y / 2.))
        disp_left = float(_truncated_normal(rng, 0., amp_x / 2.))
        disp_right = float(_truncated_normal(rng, 0., amp_x / 2.))
        quad_in = quad_in + np.array([
            [disp_left,  +disp_y],      # TL
            [disp_left,  -disp_y],      # BL
            [disp_right, +disp_y],      # BR
            [disp_right, -disp_y],      # TR
        ])
        drawn['perspective'] = {'h_left': disp_left, 'h_right': disp_right,
                                'v_disp': disp_y, 'amp_x': amp_x, 'amp_y': amp_y}

    # ── 2. scaling, about the quad's own centroid ────────────────────────────
    #
    # homographies.py:180-189. Candidate 0 is scale 1.0, prepended so that "do
    # not scale" is reachable; allow_artifacts drops it (see module docstring).
    if cfg['scaling']:
        amp = float(cfg['scaling_amplitude'])
        scales = np.concatenate(
            [[1.], _truncated_normal(rng, 1., amp / 2., int(cfg['n_scales']))])
        centre = quad_in.mean(axis=0, keepdims=True)
        candidates = (quad_in - centre)[None, ...] * scales[:, None, None] + centre
        idx, n_valid, n_cand = _pick_valid(rng, candidates, cfg['allow_artifacts'])
        quad_in = candidates[idx]
        drawn['scaling'] = {'scale': float(scales[idx]), 'n_valid': n_valid,
                            'n_candidates': n_cand, 'stddev': amp / 2.}

    # ── 3. translation ───────────────────────────────────────────────────────
    #
    # homographies.py:193-200. The slack to each edge is measured from the quad
    # as it stands after scaling, so this can never be the step that pushes a
    # corner out -- unless translation_overflow is set, which is what it is for.
    if cfg['translation']:
        t_min = quad_in.min(axis=0)             # slack to left / top
        t_max = (1. - quad_in).min(axis=0)      # slack to right / bottom
        if cfg['allow_artifacts']:
            overflow = float(cfg['translation_overflow'])
            t_min = t_min + overflow
            t_max = t_max + overflow

        # THE INTERVAL INVERTS WHEN THE QUAD IS WIDER THAN THE FRAME, and under
        # the production config it often is. Its width is
        #
        #     t_max + t_min = (1 - x_hi) + x_lo = 1 - (quad width)
        #
        # so it is empty exactly when the quad exceeds 1. With allow_artifacts
        # on, `perspective_amplitude_x` is NOT clamped to the margin, and the
        # left and right pairs of corners take independent TN(0, amp/2) draws --
        # so the width is 0.85 + (h_right - h_left), up to 1.25 before scaling
        # multiplies it by as much as 1.2 again.
        #
        # Upstream does not raise here because `tf.random_uniform`'s float path
        # is `minval + rnd * (maxval - minval)` with no validation of the order
        # (tensorflow/python/ops/random_ops.py, the `math_ops.add` line). With
        # maxval < minval that samples the REVERSED interval -- well defined,
        # and not what anyone wrote down. numpy's Generator.uniform checks, and
        # raises `high - low < 0`.
        #
        # Sorting reproduces what upstream's runs actually did rather than
        # inventing a rule. The alternative considered was "no valid
        # translation, so shift by 0", which would silently remove translation
        # from about a fifth of production draws -- a change of distribution
        # dressed as a bug fix.
        low_x, high_x = sorted((-t_min[0], t_max[0]))
        low_y, high_y = sorted((-t_min[1], t_max[1]))
        shift = np.array([rng.uniform(low_x, high_x),
                          rng.uniform(low_y, high_y)])
        quad_in = quad_in + shift[None, :]
        drawn['translation'] = {'dx': float(shift[0]), 'dy': float(shift[1]),
                                'slack_min': t_min.tolist(),
                                'slack_max': t_max.tolist(),
                                #: True when the quad was already wider than the
                                #: frame on that axis, so the bound that reads as
                                #: a lower limit is above the one that reads as
                                #: an upper limit. Recorded because how OFTEN it
                                #: happens is a fact about upstream's config, and
                                #: cli/demo_homography.py prints it.
                                'inverted_x': bool(t_max[0] + t_min[0] < 0),
                                'inverted_y': bool(t_max[1] + t_min[1] < 0)}

    # ── 4. rotation, about the quad's own centroid ───────────────────────────
    #
    # homographies.py:204-215. Angle 0 is prepended for the same reason scale 1
    # is, and dropped by allow_artifacts for the same reason.
    if cfg['rotation']:
        max_angle = float(cfg['max_angle'])
        n_angles = int(cfg['n_angles'])
        angles = np.concatenate([[0.], np.linspace(-max_angle, max_angle, n_angles)])
        centre = quad_in.mean(axis=0, keepdims=True)
        cos, sin = np.cos(angles), np.sin(angles)
        # Row-vector convention, matching upstream's
        # `matmul(pts - centre, rot_mat)` with rot_mat = [[cos, -sin],
        # [sin, cos]]. Written out rather than assembled so the transpose is
        # not a thing a reader has to hold in their head.
        centred = quad_in - centre
        candidates = np.stack([
            np.stack([centred[:, 0] * c + centred[:, 1] * s,
                      -centred[:, 0] * s + centred[:, 1] * c], axis=1) + centre
            for c, s in zip(cos, sin)])
        idx, n_valid, n_cand = _pick_valid(rng, candidates, cfg['allow_artifacts'])
        quad_in = candidates[idx]
        drawn['rotation'] = {'angle_deg': float(np.degrees(angles[idx])),
                             'n_valid': n_valid, 'n_candidates': n_cand,
                             'max_angle_deg': float(np.degrees(max_angle))}

    # ── 5. to pixels, then solve ─────────────────────────────────────────────
    scale_xy = np.array([width, height], dtype=np.float64)
    quad_out_px = quad_out * scale_xy
    quad_in_px = quad_in * scale_xy

    # Upstream builds the 8x8 system by hand and calls matrix_solve_ls
    # (homographies.py:222-230). For exactly four correspondences that system is
    # square, so cv2's exact solve gives the same matrix; `getPerspectiveTransform
    # (src, dst)` returns M with dst = M . src, and upstream solves for
    # pts2 = H . pts1, so src is quad_out and dst is quad_in.
    matrix = cv2.getPerspectiveTransform(quad_out_px.astype(np.float32),
                                         quad_in_px.astype(np.float32))
    return HomographySample(matrix=np.asarray(matrix, dtype=np.float64),
                            quad_out=quad_out_px, quad_in=quad_in_px, drawn=drawn)


def identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def invert(matrix: np.ndarray) -> np.ndarray:
    """The inverse homography, normalised so H[2, 2] == 1.

    The normalisation is not cosmetic: two matrices differing by a scale factor
    are the same projective map, so `allclose(inv(inv(H)), H)` can fail on a
    correct inverse without it.
    """
    inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    return inverse / inverse[2, 2]


# ── points ────────────────────────────────────────────────────────────────────

def _apply(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    out = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    return out[:, :2] / out[:, 2:3]


def points_output_to_input(points_xy: np.ndarray,
                           matrix: np.ndarray) -> np.ndarray:
    """Where an OUTPUT pixel was sampled from in the INPUT. Applies H directly.

    `points_xy` is (N, 2) in (x, y). See the module docstring on both counts.
    """
    return _apply(points_xy, matrix)


def points_input_to_output(points_xy: np.ndarray,
                           matrix: np.ndarray) -> np.ndarray:
    """Where an INPUT pixel lands in the OUTPUT. Applies H inverse.

    This is the direction almost every caller wants -- "I have a keypoint in the
    original, where is it in the warped image" -- and it is the one upstream
    spells `warp_points`, inversion folded in and undocumented at the call site.
    """
    return _apply(points_xy, invert(matrix))


def inside(points_xy: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Boolean mask of points within [0, w-1] x [0, h-1]. Upstream's
    `filter_points` (homographies.py:308-311), as a mask rather than a filter so
    the caller can keep scores and labels aligned with the points."""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    height, width = int(shape[0]), int(shape[1])
    return ((pts[:, 0] >= 0) & (pts[:, 0] <= width - 1) &
            (pts[:, 1] >= 0) & (pts[:, 1] <= height - 1))


# ── images ────────────────────────────────────────────────────────────────────

_INTERP = {'linear': cv2.INTER_LINEAR, 'nearest': cv2.INTER_NEAREST}


def warp_image(image: np.ndarray, matrix: np.ndarray, *,
               out_size: Optional[Tuple[int, int]] = None,
               interpolation: str = 'linear',
               border_value: float = 0) -> np.ndarray:
    """Warp `image` by a matrix that maps OUTPUT to INPUT.

    Args:
        out_size: (height, width) of the result; defaults to the input's.

    `cv2.WARP_INVERSE_MAP` is the whole point of this wrapper. Without it cv2
    treats the matrix as input -> output and inverts it first, so the warp comes
    out backwards -- correct-looking, wrong. See the module docstring.
    """
    if interpolation not in _INTERP:
        raise ValueError(
            f"interpolation must be one of {sorted(_INTERP)}, got {interpolation!r}")
    height, width = (image.shape[:2] if out_size is None
                     else (int(out_size[0]), int(out_size[1])))
    return cv2.warpPerspective(
        image, np.asarray(matrix, dtype=np.float64), (width, height),
        flags=_INTERP[interpolation] | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def warp_image_torch(batch, matrices, *,
                     out_size: Optional[Tuple[int, int]] = None,
                     mode: str = 'bilinear'):
    """The GPU path: same convention, `F.grid_sample` instead of cv2.

    Args:
        batch:    [N, C, H, W] float tensor.
        matrices: [N, 3, 3] or [3, 3], OUTPUT -> INPUT, in PIXEL coordinates.
        out_size: (height, width) of the result; defaults to the input's.

    torch imports lazily so that `cli/demo_homography.py` and the geometry half
    of `test_homography` stay runnable without it.

    `grid_sample` wants normalised coordinates in [-1, 1] with `align_corners
    =False`, i.e. -1 and +1 sit on the OUTER edges of the border pixels while
    the pixel centres sit at (2i + 1)/S - 1. The two conversions below are that
    formula and its inverse; getting them wrong shifts everything by half a
    pixel, which is under the tolerance of most tests and over the tolerance of
    a keypoint. `test_homography` scores the cv2 and torch paths against each
    other for exactly this reason.
    """
    import torch

    if batch.ndim != 4:
        raise ValueError(f'batch must be [N, C, H, W], got {tuple(batch.shape)}')
    n, _, in_h, in_w = batch.shape
    out_h, out_w = (in_h, in_w) if out_size is None else (int(out_size[0]),
                                                          int(out_size[1]))

    mat = torch.as_tensor(np.asarray(matrices, dtype=np.float64),
                          dtype=batch.dtype, device=batch.device)
    if mat.ndim == 2:
        mat = mat.expand(n, 3, 3)
    if mat.shape != (n, 3, 3):
        raise ValueError(f'matrices must be [N, 3, 3] or [3, 3], '
                         f'got {tuple(mat.shape)} for batch of {n}')

    ys, xs = torch.meshgrid(
        torch.arange(out_h, dtype=batch.dtype, device=batch.device),
        torch.arange(out_w, dtype=batch.dtype, device=batch.device),
        indexing='ij')
    ones = torch.ones_like(xs)
    grid_out = torch.stack([xs, ys, ones], dim=-1).reshape(1, -1, 3)

    src = grid_out @ mat.transpose(1, 2)                  # [N, out_h*out_w, 3]
    src = src[..., :2] / src[..., 2:3]

    # pixel index -> normalised, align_corners=False
    norm_x = (2. * src[..., 0] + 1.) / in_w - 1.
    norm_y = (2. * src[..., 1] + 1.) / in_h - 1.
    grid = torch.stack([norm_x, norm_y], dim=-1).reshape(n, out_h, out_w, 2)

    return torch.nn.functional.grid_sample(
        batch, grid, mode=mode, padding_mode='zeros', align_corners=False)


def valid_mask(shape: Tuple[int, int], matrix: np.ndarray, *,
               erosion_radius: int = 0) -> np.ndarray:
    """Where the warped result actually came from inside the source image.

    Args:
        shape: (height, width) of both source and result.

    Returns a bool array in the OUTPUT frame: True where the output pixel was
    sampled from within the input, False on the bordering artefacts.

    `erosion_radius` shrinks the True region with an elliptical kernel of
    diameter `2 * radius`, which is upstream's `compute_valid_mask`
    (homographies.py:257-277). Two things about that are NOT a straight
    transcription, and both were found by a test rather than by reading:

    NO `+ 1.`, AND THAT ONE IS FREE. Upstream adds it because `tf.nn.erosion2d`
    subtracts the kernel from the input; `cv2.erode` takes a plain minimum. A
    reader diffing the two files should not go looking for the missing line.

    `borderValue=0` AND AN EXPLICIT ANCHOR, AND THOSE ARE NOT FREE:

      * cv2's default border for erosion is `morphologyDefaultBorderValue()` =
        +DBL_MAX, so the frame edge is treated as "everything outside is valid"
        and an all-ones mask survives erosion untouched. `tf.nn.erosion2d` with
        SAME padding pads with ZERO, which does pull the border down. Without
        `borderValue=0` this function silently does less than upstream's --
        the failure mode that only an assertion catches, because a slightly
        larger valid region looks exactly like a correct one.

      * The kernel is EVEN-sized, so the anchor is not the centre and the
        erosion is asymmetric. The two libraries choose opposite sides:

            tf SAME   pad_before = (k-1)//2 = 2, pad_after = 3   for k=6
                      -> eats 2 px from top/left, 3 from bottom/right
            cv2       anchor = k//2 = 3
                      -> eats 3 px from top/left, 2 from bottom/right

        Mirror images. So "keep upstream's kernel" is not the same as "keep
        upstream's behaviour", and the anchor below is set explicitly to TF's
        `(k-1)//2` to make them agree pixel for pixel.

        A symmetric `2*radius + 1` kernel was the alternative, and it would make
        `valid_border_margin=3` mean 3 in every direction, which is what the
        name says. Rejected deliberately: the numbers here are meant to
        reproduce upstream's, and a one-pixel systematic difference on two sides
        of every tile of every homography is exactly the kind of thing that does
        not average out over the 100 draws of Homographic Adaptation.

    Homographic Adaptation needs this in both directions and gets them from the
    same function: `valid_mask(shape, H)` is the valid region inside the warped
    frame, `valid_mask(shape, invert(H))` is the coverage of the warped-back
    result in the original frame -- upstream's `mask` and `count` respectively
    (homographies.py:59-62).
    """
    height, width = int(shape[0]), int(shape[1])
    ones = np.ones((height, width), dtype=np.uint8)
    mask = warp_image(ones, matrix, interpolation='nearest', border_value=0)
    return erode_valid(mask, erosion_radius)


def erode_valid(mask: np.ndarray, erosion_radius: int) -> np.ndarray:
    """Upstream's border erosion, on a mask the caller already has.

    Split out of `valid_mask` because Homographic Adaptation cannot always use
    that function to BUILD its mask: with a pre-tile the source is larger than
    the output frame, so validity has to be warped from the pre-tile's own
    extent, and only the erosion is shared. Everything delicate is in here --
    the even kernel, TF's anchor, `borderValue=0` -- so there is one copy of it
    and not two. See `valid_mask` for why each of those is what it is.

    Takes and returns bool; a radius of 0 is a no-op, which is upstream's
    `valid_border_margin: 0`.
    """
    mask = np.asarray(mask)
    if erosion_radius <= 0:
        return mask.astype(bool)
    size = int(erosion_radius) * 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    anchor = erosion_anchor(int(erosion_radius))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, anchor=(anchor, anchor),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return eroded.astype(bool)


def erosion_anchor(erosion_radius: int) -> int:
    """Where TF's SAME padding puts the anchor in a `2 * radius` kernel.

    `pad_before = (k - 1) // 2` for stride 1, which for k = 6 is 2 -- so the
    window at output i covers input [i-2, i+3] and the erosion eats 2 px from
    the top and left and 3 from the bottom and right. cv2's own default is
    `k // 2 = 3`, the mirror of that.

    Public so the test can assert the exact eroded widths instead of settling
    for "smaller than before". Before the anchor was pinned, those widths were a
    cv2 implementation detail nothing here had measured, and the test said so
    and checked less.
    """
    return (erosion_radius * 2 - 1) // 2


def quad_polygon(sample: HomographySample, which: str = 'in') -> np.ndarray:
    """The quad as a closed (5, 2) polyline in (x, y), for plotting."""
    quad = sample.quad_in if which == 'in' else sample.quad_out
    return np.concatenate([quad, quad[:1]], axis=0)
