#!/usr/bin/env python3
"""Tests for the student half of training/SuperPathPoint/SuperPoint/.

    python utilities/test_modules/test_superpathpoint.py
    python utilities/test_modules/test_superpathpoint.py --only losses

Files 11-19 of spec.md 14 in one place: Interfaces, Backbones, Decoders, Heads,
KeypointNet, Losses, Datasets, Trainer. No weights, no GPU, no WSI -- the model
is built from scratch at tile 64 and the data is a temporary store this file
writes.

ONE FILE, NOT EIGHT
--------------------
CLAUDE.md's rule is that a test is named after the module it tests, and this
breaks it deliberately: what is being tested here is not eight modules but the
CONTRACT BETWEEN them. Almost every check below crosses a file boundary --
`space_to_depth` against `depth_to_space_prob`, `check_shapes` against a
backbone, the dataset's warp against the loss's correspondence mask -- and eight
files would each hold half of a statement.

THE FIVE THAT WOULD RUN AND BE WRONG
--------------------------------------
Ranked, because that is the order the sections are in:

1. `space_to_depth` not being the exact inverse of `depth_to_space_prob`. The
   labels are then transposed inside every cell against the predictions. Both
   sides are self-consistent, the loss still falls, and the model is merely
   worse. Nothing anywhere raises.
2. `correspondence_mask` built from the wrong direction of the homography. The
   descriptor's positives become the pairs that do NOT correspond; the loss
   still falls, and the descriptors learn to match the wrong cells.
3. `sample_descriptors` alignment -- the `+ 0.5`, the division by the INPUT
   extent, `align_corners=False`. Any one of them wrong shifts every descriptor
   by up to half a cell. The matches are still matches, to slightly the wrong
   place.
4. The dataset warping the keypoint MAP instead of the points. A keypoint is one
   pixel; bilinear interpolation spreads it over four at a quarter the height
   and the threshold deletes most of them, differently every epoch.
5. A backbone whose declared `stride` is not its real one. The cell grid is then
   a different size than the labels were splatted onto.

Every one of them is scored against a decoy -- the transposed convention, the
reversed direction, a neighbouring cell -- rather than against a tolerance,
because a margin over a decoy is robust and a threshold is a guess.

Sections:
  1. losses     -- the cell round trip, the labels, the correspondence direction
  2. heads      -- descriptor sampling, against a neighbouring cell
  3. interfaces -- what check_shapes catches
  4. backbone   -- the widths and the stride
  5. decoder    -- channels, grid, and what UpsampleDecoder refuses
  6. net        -- wiring, forward shapes, extraction
  7. dataset    -- the pair, the label warp, the balance switch
  8. train      -- one batch, overfitted, as the only end-to-end evidence
"""

from __future__ import annotations

import argparse
import collections
import itertools
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
import torch                                                     # noqa: E402
import torch.nn as nn                                            # noqa: E402

from common import KeypointLabelStore              # noqa: E402
import PreTileStore
from common.Homography import invert, points_input_to_output     # noqa: E402
from common.Interfaces import ShapeMismatch, check_shapes        # noqa: E402
from common.KeypointLabelStore import (LabelMeta,                # noqa: E402
                                       batch_from_lists)
from PreTileStore import (PreTileMeta, PreTileRecord,     # noqa: E402
                                 centre_crop, centre_margin,
                                 pre_tile_px)
from common.KeypointLabelStore import points_from_prob        # noqa: E402
from SuperPoint.Backbones import VggBackbone, VggBackboneConfig  # noqa: E402
from SuperPoint.Datasets import PairDatasetConfig, splat         # noqa: E402
from SuperPoint.Decoders import (DepthToSpaceDecoderConfig,      # noqa: E402
                                 UpsampleDecoderConfig,
                                 depth_to_space_prob)
from SuperPoint.Heads import DescriptorHeadConfig, sample_descriptors  # noqa: E402
from SuperPoint.KeypointNet import (KeypointNetConfig,           # noqa: E402
                                    KeypointOutput,
                                    extract_keypoints)
from SuperPoint.Losses import (SuperPointLossConfig, cell_labels,  # noqa: E402
                               cell_valid, correspondence_mask,
                               descriptor_loss, detector_loss,
                               space_to_depth)
from SuperPoint.Trainer import (TrainerConfig, _cut, _match_rate,      # noqa: E402
                                _repeatability_row)

_RESULTS = []

CELL = 8
TILE = 64          # 8 cells of 8. Big enough to have an interior, small enough
                   # that every forward below is instant.
GRID = TILE // CELL

#: The VGG's stride, which happens to equal the cell -- and that coincidence is
#: exactly what hid a bug in `sample_descriptors` and `descriptor_loss` until a
#: stride-16 backbone arrived. Spelled as its own name so a reader can see which
#: of the two each call site means.
STRIDE = CELL


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ══════════════════════════════════════════════════════════════════════════════
#  1. losses
# ══════════════════════════════════════════════════════════════════════════════

def t_space_to_depth_is_the_exact_inverse_of_the_decode():
    """THE ONE THIS FILE EXISTS FOR.

    The loss splats labels into cells with `space_to_depth`; the model decodes
    predictions out of cells with `depth_to_space_prob`. If the two disagree
    about which channel is which pixel, every label is transposed inside its own
    cell against every prediction -- and both halves are internally consistent,
    so the loss falls, the model trains, and nothing raises.

    Checked by making a map where every pixel is unique, so a transposition
    cannot hide behind a symmetry.
    """
    pixels = torch.arange(TILE * TILE, dtype=torch.float32).reshape(1, TILE, TILE)
    cells = space_to_depth(pixels, CELL)                # [1, GRID, GRID, 64]
    assert tuple(cells.shape) == (1, GRID, GRID, CELL * CELL), cells.shape

    # Put the cells back through the decode's own arithmetic. Softmax would
    # destroy the values, so the decode is applied to a one-hot instead: for
    # every channel, the decode must place it where space_to_depth read it from.
    for channel in (0, 1, CELL, 13, 63):
        logits = torch.full((1, CELL * CELL + 1, GRID, GRID), -20.0)
        logits[0, channel] = 20.0
        prob = depth_to_space_prob(logits, CELL)[0]
        for row in range(GRID):
            for col in range(GRID):
                block = prob[row * CELL:(row + 1) * CELL,
                             col * CELL:(col + 1) * CELL]
                flat = int(block.argmax())
                placed = pixels[0, row * CELL:(row + 1) * CELL,
                                col * CELL:(col + 1) * CELL].reshape(-1)[flat]
                assert float(placed) == float(cells[0, row, col, channel]), (
                    f'channel {channel} at cell ({row}, {col}): the decode put '
                    f'it at pixel {float(placed)} and space_to_depth read it '
                    f'from {float(cells[0, row, col, channel])}')
    return f'{GRID}x{GRID} cells, 5 channels, exact'


def t_the_transposed_packing_is_a_different_answer():
    """The decoy for the check above. Without it, two consistent-but-wrong
    implementations would pass together."""
    pixels = torch.arange(TILE * TILE, dtype=torch.float32).reshape(1, TILE, TILE)
    cells = space_to_depth(pixels, CELL)
    # The mirror convention: swap the two within-cell axes.
    n, h, w = pixels.shape
    other = pixels.reshape(n, h // CELL, CELL, w // CELL, CELL)
    other = other.permute(0, 1, 3, 4, 2).reshape(n, h // CELL, w // CELL,
                                                 CELL * CELL)
    assert not torch.equal(cells, other), (
        'the transposed packing is identical to the real one, so this test '
        'cannot see the failure it is written for')
    return 'the two packings differ'


def t_cell_labels_put_an_empty_cell_in_the_dustbin():
    keypoints = torch.zeros(1, TILE, TILE)
    keypoints[0, 9, 17] = 1.0                       # cell (1, 2), offset (1, 1)
    labels = cell_labels(keypoints, CELL, tie_break=0.0)
    assert int(labels[0, 1, 2]) == 1 * CELL + 1, int(labels[0, 1, 2])
    empty = [int(labels[0, r, c]) for r in range(GRID) for c in range(GRID)
             if (r, c) != (1, 2)]
    assert set(empty) == {CELL * CELL}, sorted(set(empty))
    return f'dustbin is {CELL * CELL}, the one keypoint is {int(labels[0, 1, 2])}'


def t_the_tie_break_can_never_promote_a_dustbin():
    """0.1 of noise against a gap of 1. A cell with two keypoints has to pick
    one, and picking the first would bias every crowded cell toward its
    top-left; the noise makes it a coin flip. What it must NOT do is turn an
    empty cell into a keypoint."""
    keypoints = torch.zeros(64, TILE, TILE)
    labels = cell_labels(keypoints, CELL, tie_break=0.1)
    assert int(labels.min()) == CELL * CELL == int(labels.max()), (
        f'an empty batch produced labels in [{int(labels.min())}, '
        f'{int(labels.max())}]; the tie-break promoted a dustbin')

    both = torch.zeros(400, TILE, TILE)
    both[:, 0, 0] = 1.0
    both[:, 7, 7] = 1.0                              # same cell, two corners
    picked = cell_labels(both, CELL, tie_break=0.1)[:, 0, 0]
    share = float((picked == 0).float().mean())
    assert 0.3 < share < 0.7, (
        f'the first of two keypoints in a cell won {share:.0%} of the time; '
        f'the tie-break is meant to make that a coin flip')
    return f'two-in-a-cell splits {share:.0%}/{1 - share:.0%}'


def t_the_valid_mask_is_anded_down_a_cell_not_averaged():
    """A cell counts only if EVERY one of its 64 pixels is valid
    (`reduce_prod`, `models/utils.py:22`). Averaging would let a cell that is
    one pixel inside the warp contribute a fully-weighted label built from
    mostly-invalid pixels."""
    mask = torch.ones(1, TILE, TILE)
    mask[0, 0, 0] = 0.0
    cells = cell_valid(mask, CELL, mask.shape, torch.device('cpu'))
    assert float(cells[0, 0, 0]) == 0.0, float(cells[0, 0, 0])
    assert float(cells.sum()) == GRID * GRID - 1, float(cells.sum())
    return 'one invalid pixel kills its whole cell'


def t_correspondence_is_the_diagonal_under_the_identity():
    s = correspondence_mask(torch.eye(3)[None], STRIDE, (GRID, GRID),
                            torch.device('cpu'))[0]
    eye = torch.eye(GRID * GRID).reshape(GRID, GRID, GRID, GRID)
    assert torch.equal(s, eye), 'the identity homography is not the diagonal'
    return f'{GRID * GRID} cells, exactly the diagonal'


def t_correspondence_moves_the_way_the_warp_moves():
    """THE DIRECTION CHECK, against the reversed matrix as a decoy.

    `matrix` is OUTPUT -> INPUT, so an original cell centre reaches the warped
    frame through its INVERSE. A translation that shifts the warped image right
    by one cell must make cell (r, c) of the original correspond to cell
    (r, c+1) of the warped one. Reversing it moves the other way -- and the loss
    would still fall, on the pairs that do not correspond.
    """
    shift = torch.eye(3)
    shift[0, 2] = -float(CELL)          # OUT -> IN: output x = input x + CELL
    s = correspondence_mask(shift[None], STRIDE, (GRID, GRID),
                            torch.device('cpu'))[0]

    right, wrong = 0, 0
    for row in range(GRID):
        for col in range(GRID - 1):
            right += int(s[row, col, row, col + 1])
            wrong += int(s[row, col, row, max(col - 1, 0)])
    assert right == GRID * (GRID - 1), (
        f'{right} of {GRID * (GRID - 1)} cells matched one to the right')
    assert wrong == 0, (
        f'{wrong} cells matched one to the LEFT, which is the reversed '
        f'direction -- the positives would be the pairs that do not correspond')
    return f'{right} right, {wrong} left'


def t_the_descriptor_loss_prefers_agreeing_descriptors():
    """Matching descriptors under the identity must cost less than mismatched
    ones. Not a tolerance -- a comparison between two arrangements of the same
    tensors."""
    torch.manual_seed(0)
    desc = torch.nn.functional.normalize(torch.randn(1, 16, GRID, GRID), dim=1)
    eye = torch.eye(3)[None]
    agreeing = float(descriptor_loss(desc, desc.clone(), eye, STRIDE))
    # The decoy: the same descriptors, rolled by one cell, so every positive
    # pair now holds two unrelated vectors.
    rolled = torch.roll(desc, shifts=1, dims=3)
    mismatched = float(descriptor_loss(desc, rolled, eye, STRIDE))
    assert agreeing < mismatched, (agreeing, mismatched)
    return f'{agreeing:.4f} agreeing vs {mismatched:.4f} rolled'


def t_the_detector_loss_prefers_the_right_labels():
    torch.manual_seed(0)
    keypoints = torch.zeros(2, TILE, TILE)
    keypoints[:, 9, 17] = 1.0
    labels = cell_labels(keypoints, CELL, tie_break=0.0)
    logits = torch.full((2, CELL * CELL + 1, GRID, GRID), -5.0)
    logits.scatter_(1, labels[:, None], 5.0)

    good = float(detector_loss(logits, keypoints, CELL, tie_break=0.0))
    shifted = torch.roll(keypoints, shifts=CELL, dims=2)
    bad = float(detector_loss(logits, shifted, CELL, tie_break=0.0))
    assert good < bad, (good, bad)
    return f'{good:.4f} on its own labels vs {bad:.4f} one cell over'


def t_the_rung_weight_reaches_the_loss():
    """The `loss-weight` half of the balance switch. A weight of zero on every
    sample has to zero the term; anything else means the switch is wired to
    nothing."""
    torch.manual_seed(0)
    keypoints = torch.zeros(2, TILE, TILE)
    keypoints[0, 9, 17] = 1.0
    logits = torch.randn(2, CELL * CELL + 1, GRID, GRID)
    plain = float(detector_loss(logits, keypoints, CELL, tie_break=0.0))
    lopsided = float(detector_loss(logits, keypoints, CELL, tie_break=0.0,
                                   sample_weight=torch.tensor([1.0, 0.0])))
    first_only = float(detector_loss(logits[:1], keypoints[:1], CELL,
                                     tie_break=0.0))
    assert abs(lopsided - first_only) < 1e-5, (lopsided, first_only)
    assert abs(plain - lopsided) > 1e-6, 'the weight changed nothing'
    return f'weighting to one sample reproduces that sample alone'


# ══════════════════════════════════════════════════════════════════════════════
#  2. heads
# ══════════════════════════════════════════════════════════════════════════════

def t_a_descriptor_is_sampled_from_its_own_cell():
    """A point at a cell's centre must return that cell's vector, and a point at
    the neighbouring centre a different one.

    This is what pins the `+ 0.5` and the division by the INPUT extent. Getting
    either wrong shifts the sampling by up to half a cell; the returned vectors
    are still unit vectors from a plausible neighbourhood, and the matching they
    produce is still matching -- to slightly the wrong place.
    """
    dense = torch.zeros(1, 4, GRID, GRID)
    for row in range(GRID):
        for col in range(GRID):
            dense[0, :, row, col] = torch.tensor(
                [1.0, float(row), float(col), 0.0])
    dense = torch.nn.functional.normalize(dense, dim=1)

    # The centre of feature pixel `col`, in INPUT PIXEL INDICES, is
    # `col*s + (s-1)/2` -- 3.5 at s=8, not 4. Index i covers the continuous
    # interval [i, i+1), so the block col*s .. col*s+s-1 has its centre half a
    # pixel below `col*s + s//2`. `sample_descriptors` adds the `+0.5` that
    # turns an index into a continuous coordinate (upstream's line verbatim),
    # so the two land on each other only at the half.
    #
    # `Losses.correspondence_mask` DOES use `h*cell + cell//2`, and that is not
    # a contradiction: it is upstream's as well (`models/utils.py:79-82`) and it
    # feeds a match with a `cell - 0.5` tolerance, where half a pixel cannot
    # change the answer. Here it is the whole answer -- the first run of this
    # check failed with `col*s + s//2`, which blends 1/(2s) = 6.25 per cent of
    # the neighbouring vector into every sample.
    half = (STRIDE - 1) / 2
    centres = torch.tensor([[[col * STRIDE + half, row * STRIDE + half]
                             for row in range(GRID) for col in range(GRID)]],
                           dtype=torch.float32)
    got = sample_descriptors(centres, dense, STRIDE)[0]

    want = dense[0].permute(1, 2, 0).reshape(-1, 4)
    assert torch.allclose(got, want, atol=1e-5), (
        'a descriptor sampled at a cell centre is not that cell\'s vector; the '
        'alignment (+0.5, the input-extent scale, align_corners=False) is off')

    # The decoy: one cell to the right must be a DIFFERENT vector, so the check
    # above is not passing on a map that is constant.
    off = centres.clone()
    off[..., 0] += STRIDE
    shifted = sample_descriptors(off, dense, STRIDE)[0]
    moved = (shifted - got).abs().max(dim=1).values > 1e-3
    assert bool(moved[:-1].all()), 'sampling one cell over returned the same vectors'
    return f'{GRID * GRID} cell centres exact, neighbours differ'


def t_descriptors_are_sampled_on_the_stride_and_not_on_the_cell():
    """THE ONE THAT WAS ACTUALLY WRONG.

    `sample_descriptors` normalises by `feature_size * s`, and `s` has to be the
    pitch of the map it is sampling -- the BACKBONE's stride. The call site
    passed `cfg.detector.cell`, which is a different number the moment a decoder
    changes resolution between the two grids. Upstream's VGG is stride 8 against
    cell 8, so every existing check above passes under either reading; gigapath
    is stride 16 against cell 8, and there the scale came out half the real
    extent, every normalised coordinate came out twice too large, and
    `grid_sample` returned border-clamped vectors -- unit norm, right shape,
    sampled from the wrong place.

    So: a map at stride 16 with a planted vector per feature pixel, read at the
    centre of each. Passing the stride must return exactly what was planted;
    passing the cell must not.
    """
    fg = TILE // (2 * CELL)                      # stride 16 on a 64 px tile
    dense = torch.zeros(1, 4, fg, fg)
    for row in range(fg):
        for col in range(fg):
            dense[0, :, row, col] = torch.tensor(
                [1.0, float(row + 1), float(col + 1), 0.0])
    dense = torch.nn.functional.normalize(dense, dim=1)
    want = dense[0].permute(1, 2, 0).reshape(-1, 4)

    half = (2 * CELL - 1) / 2                    # 7.5 at stride 16, not 8
    centres = torch.tensor(
        [[[col * 2 * CELL + half, row * 2 * CELL + half]
          for row in range(fg) for col in range(fg)]], dtype=torch.float32)

    right = sample_descriptors(centres, dense, 2 * CELL)[0]
    assert torch.allclose(right, want, atol=1e-5), \
        'sampling with the stride did not return the planted vectors'

    wrong = sample_descriptors(centres, dense, CELL)[0]
    assert not torch.allclose(wrong, want, atol=1e-3), \
        ('sampling with the cell gave the same answer as sampling with the '
         'stride, so this fixture cannot tell the two apart -- pick a stride '
         'that is not the cell')
    differ = int(((wrong - want).abs().max(dim=1).values > 1e-3).sum())
    return f'stride 16 vs cell 8: {differ}/{fg * fg} points land elsewhere'


def t_sampled_descriptors_are_unit_vectors():
    """Interpolating between unit vectors does not give a unit vector, which is
    why upstream normalises again after `grid_sample`."""
    torch.manual_seed(0)
    dense = torch.nn.functional.normalize(torch.randn(1, 8, GRID, GRID), dim=1)
    points = torch.rand(1, 20, 2) * TILE
    got = sample_descriptors(points, dense, STRIDE)
    norms = got.pow(2).sum(-1).sqrt()
    assert float((norms - 1).abs().max()) < 1e-5, float(norms.min())
    return f'20 interpolated points, norm 1 to 1e-5'


def t_the_head_normalises_its_output():
    head = DescriptorHeadConfig(dim=16, hidden=8, in_channels=4).build().eval()
    with torch.no_grad():
        dense = head(torch.randn(2, 4, GRID, GRID))
    norms = dense.pow(2).sum(1).sqrt()
    assert float((norms - 1).abs().max()) < 1e-4, float(norms.max())
    return 'unit norm along dim'


# ══════════════════════════════════════════════════════════════════════════════
#  3. interfaces
# ══════════════════════════════════════════════════════════════════════════════

class _LyingBackbone(nn.Module):
    """Declares one thing and does another. `check_shapes` exists for this."""

    trainable = True

    def __init__(self, out_channels=8, stride=8, real_stride=4, real_c=8):
        super().__init__()
        self.out_channels, self.stride = out_channels, stride
        self.pool = nn.AvgPool2d(real_stride, real_stride)
        self.conv = nn.Conv2d(3, real_c, 1)

    def forward(self, images):
        return self.pool(self.conv(images))


def t_check_shapes_passes_a_consistent_stack():
    cfg = KeypointNetConfig.wired(in_channels=3, cell=CELL, descriptor_dim=16)
    backbone = cfg.backbone.build()
    shape = check_shapes(backbone, cfg.detector.build(), cfg.descriptor.build(),
                         image_size=TILE, channels=3)
    assert shape == (1, backbone.out_channels, GRID, GRID), shape
    return f'{shape}'


def t_check_shapes_catches_a_lying_stride():
    """The failure it is for: everything runs, and the cell grid is a different
    size than the labels were splatted onto."""
    try:
        check_shapes(_LyingBackbone(stride=8, real_stride=4), image_size=TILE)
    except ShapeMismatch as e:
        assert 'stride' in str(e), str(e)
        return 'declared 8, measured 4'
    raise AssertionError('a backbone with the wrong declared stride passed')


def t_check_shapes_catches_a_lying_width_and_an_unnormalised_head():
    try:
        check_shapes(_LyingBackbone(out_channels=99, real_c=8, real_stride=8),
                     image_size=TILE)
    except ShapeMismatch as e:
        assert 'out_channels' in str(e), str(e)
    else:
        raise AssertionError('a backbone with the wrong declared width passed')

    class Raw(nn.Module):
        dim = 8

        def forward(self, features):
            return torch.randn(features.shape[0], 8, *features.shape[2:]) * 5

    try:
        check_shapes(_LyingBackbone(real_stride=8), descriptor=Raw(),
                     image_size=TILE)
    except ShapeMismatch as e:
        assert 'normalise' in str(e), str(e)
        return 'both refused'
    raise AssertionError('an unnormalised descriptor head passed')


# ══════════════════════════════════════════════════════════════════════════════
#  4. backbone
# ══════════════════════════════════════════════════════════════════════════════

def t_the_trunk_ends_at_128_and_not_at_256():
    """`channels[:-1]` drops the head's hidden width. Reading the list the
    obvious way builds a trunk one stage deeper and twice as wide -- which
    trains, on a different architecture (spec.md 9)."""
    cfg = VggBackboneConfig()
    assert cfg.out_channels == 128, cfg.out_channels
    assert cfg.channels[-1] == 256, cfg.channels
    assert cfg.stride == 8, cfg.stride

    backbone = VggBackboneConfig(in_channels=1).build().eval()
    with torch.no_grad():
        features = backbone(torch.zeros(1, 1, TILE, TILE))
    assert tuple(features.shape) == (1, 128, GRID, GRID), features.shape
    return f'128 channels at stride 8, {tuple(features.shape)}'


def t_the_backbone_refuses_the_wrong_channel_count_and_size():
    backbone = VggBackboneConfig(in_channels=1).build()
    for images, why in ((torch.zeros(1, 3, TILE, TILE), 'RGB into a gray model'),
                        (torch.zeros(1, 1, TILE + 1, TILE), 'an odd size')):
        try:
            backbone(images)
        except ValueError:
            continue
        raise AssertionError(f'{why} was accepted')
    return 'channels and stride multiple both checked'


# ══════════════════════════════════════════════════════════════════════════════
#  5. decoder
# ══════════════════════════════════════════════════════════════════════════════

def t_the_decoder_emits_the_dustbin_and_keeps_the_grid():
    decoder = DepthToSpaceDecoderConfig(cell=CELL, hidden=8,
                                        in_channels=4).build().eval()
    with torch.no_grad():
        logits = decoder(torch.randn(2, 4, GRID, GRID))
    assert tuple(logits.shape) == (2, CELL * CELL + 1, GRID, GRID), logits.shape
    assert tuple(decoder.decode(logits).shape) == (2, TILE, TILE)
    return f'{CELL}^2 + 1 = {CELL * CELL + 1} channels, grid unchanged'


def t_the_upsample_decoder_reaches_the_cell_and_refuses_what_it_cannot():
    decoder = UpsampleDecoderConfig(cell=CELL, stride=16, hidden=8,
                                    in_channels=16).build().eval()
    with torch.no_grad():
        logits = decoder(torch.randn(1, 16, 4, 4))
    assert tuple(logits.shape) == (1, CELL * CELL + 1, 8, 8), logits.shape

    # 14 over 8 is not a power of two, and rounding it would offset the cell
    # grid from the labels everywhere and uniformly.
    try:
        UpsampleDecoderConfig(cell=CELL, stride=14, in_channels=16).build()
    except ValueError as e:
        assert 'transposed convolutions' in str(e), str(e)
        return 'stride 16 -> cell 8 works, stride 14 refused'
    raise AssertionError('a stride that is not a power-of-two multiple was accepted')


# ══════════════════════════════════════════════════════════════════════════════
#  6. net
# ══════════════════════════════════════════════════════════════════════════════

def t_wired_builds_and_forward_has_the_shapes_the_loss_wants():
    net = _net()
    with torch.no_grad():
        out = net(torch.zeros(2, 1, TILE, TILE))
    assert tuple(out.cell_logits.shape) == (2, CELL * CELL + 1, GRID, GRID)
    assert tuple(out.prob_map.shape) == (2, TILE, TILE)
    assert tuple(out.descriptors.shape) == (2, 16, GRID, GRID)
    return f'{net.summary().split("  ")[-2]}'


def t_build_validates_the_widths_rather_than_repairing_them():
    """A repaired config is not the config that was hashed, so the identity
    would name a model nobody built."""
    import dataclasses
    cfg = KeypointNetConfig.wired(in_channels=1, cell=CELL, descriptor_dim=16)
    broken = dataclasses.replace(
        cfg, detector=dataclasses.replace(cfg.detector, in_channels=999))
    try:
        broken.build()
    except ValueError as e:
        assert 'wired' in str(e), str(e)
        return 'refused, and named the constructor that gets it right'
    raise AssertionError('a mismatched width was silently repaired')


def t_extraction_uses_the_same_rule_as_the_labels():
    """`extract_keypoints` goes through `points_from_prob`, the function that cut
    the teacher's labels. Two implementations of NMS would make every
    repeatability number a comparison of two conventions (spec.md 14)."""
    net = _net()
    prob = torch.zeros(1, TILE, TILE)
    prob[0, 30, 10] = 0.9
    prob[0, 31, 11] = 0.4                     # inside the NMS radius, must lose
    output = KeypointOutput(cell_logits=torch.zeros(1, CELL * CELL + 1,
                                                    GRID, GRID),
                            prob_map=prob, descriptors=None)
    points = extract_keypoints(output, net.cfg)[0]
    assert len(points) == 1, len(points)
    assert tuple(points.xy[0]) == (10, 30), tuple(points.xy[0])
    return f'1 point at (x=10, y=30), the neighbour suppressed'


def t_identity_moves_with_the_parts_and_not_with_the_extraction_alone():
    import dataclasses
    net = _net()
    base = net.identity_id()
    other = dataclasses.replace(
        net.cfg, backbone=dataclasses.replace(net.cfg.backbone, in_channels=3))
    assert other.build().identity_id() != base, 'the channel count did not move it'
    # The extraction fields ARE hashed, on purpose: a repeatability number is a
    # property of the pair (model, extraction rule), and the labels carry a rule
    # of their own.
    nms = dataclasses.replace(net.cfg, nms_radius=2)
    assert nms.build().identity_id() != base, 'nms_radius did not move it'
    return f'{base} moves with both'


def _net(channels: int = 1):
    return KeypointNetConfig.wired(in_channels=channels, cell=CELL,
                                   descriptor_dim=16).build()


# ══════════════════════════════════════════════════════════════════════════════
#  7. dataset
# ══════════════════════════════════════════════════════════════════════════════

def t_splat_is_xy_and_clamps_to_the_frame():
    out = splat(np.array([[10, 30]]), (TILE, TILE))
    assert float(out[30, 10]) == 1.0, 'splat wrote (row, col) instead of (x, y)'
    assert float(out.sum()) == 1.0
    edge = splat(np.array([[TILE + 5, -3]]), (TILE, TILE))
    assert float(edge[0, TILE - 1]) == 1.0, 'an out-of-frame point was not clamped'
    return 'one point at (x=10, y=30), out-of-frame clamped'


def t_the_pair_warps_points_and_not_the_map():
    """The label of the warped view must be the warped POINTS re-splatted.

    Checked by rebuilding the expected map from the recorded homography: if the
    dataset had warped the keypoint map as an image, the result would be blurred
    across four pixels and mostly gone, and it would not agree with this.
    """
    with tempfile.TemporaryDirectory() as root:
        tiles_root, labels_root, points = _make_stores(root)
        cfg = PairDatasetConfig(tile=TILE, in_channels=1, seed=0)
        dataset = cfg.build(tiles_root, labels_root, wsi_stems=[_STEM_A])
        assert len(dataset) == 2, len(dataset)

        item = dataset[0]
        assert tuple(item['image'].shape) == (1, TILE, TILE), item['image'].shape
        assert float(item['keypoint_map'].sum()) == len(points), (
            'the identity view lost points on the way in')

        matrix = item['homography'].numpy().astype(np.float64)
        warped = points_input_to_output(points.astype(np.float64), matrix)
        keep = ((warped >= 0) & (warped <= TILE - 1)).all(axis=1)
        want = splat(warped[keep], (TILE, TILE))
        got = item['warped_keypoint_map'].numpy()
        assert np.array_equal(got, want), (
            f'{int(got.sum())} points in the warped label, {int(want.sum())} '
            f'expected from warping the points. A warped keypoint MAP would be '
            f'blurred across four pixels and mostly thresholded away')
        return f'{int(want.sum())} of {len(points)} points survived the warp'


def t_the_warped_valid_mask_is_nearly_full_because_of_the_pre_tile():
    """spec.md 6.6's evidence, as a pair. With a 3x source there is nothing
    outside to sample, so the only False is the eroded rim -- and it must NOT be
    the two thirds a tile-sized source would give."""
    with tempfile.TemporaryDirectory() as root:
        tiles_root, labels_root, _ = _make_stores(root)
        dataset = PairDatasetConfig(tile=TILE, in_channels=1, seed=0).build(
            tiles_root, labels_root, wsi_stems=[_STEM_A])
        rim = 3
        fractions = []
        for i in range(len(dataset)):
            mask = dataset[i]['warped_valid_mask'].numpy()
            fractions.append(float(mask[rim:-rim, rim:-rim].mean()))
        assert min(fractions) > 0.999, (
            f'the worst interior is {min(fractions):.3f} valid, so a draw ran '
            f'off the 3x pre-tile')
        return f'interior {min(fractions):.4f} valid'


def t_align_min_truncates_and_loss_weight_does_not():
    """The switch, both ways. `none` and `loss-weight` keep every tile;
    `align-min` cuts every rung to the smallest."""
    with tempfile.TemporaryDirectory() as root:
        tiles_root, labels_root, _ = _make_stores(root, rungs=((1.0, 4), (4.0, 1)))
        sizes = {}
        for mode in ('none', 'align-min', 'loss-weight'):
            dataset = PairDatasetConfig(tile=TILE, balance=mode, seed=0).build(
                tiles_root, labels_root, wsi_stems=[_STEM_A])
            sizes[mode] = (len(dataset),
                           [round(float(w), 2) for w in dataset.rung_weight])
        assert sizes['none'][0] == 5, sizes
        assert sizes['align-min'][0] == 2, sizes
        assert sizes['loss-weight'][0] == 5, sizes
        assert sizes['none'][1] == [1.0, 1.0], sizes
        # 4 and 1 tiles -> the sparse rung is weighted up, mean 1.
        assert sizes['loss-weight'][1][1] > sizes['loss-weight'][1][0], sizes
        assert abs(sum(sizes['loss-weight'][1]) / 2 - 1.0) < 0.01, sizes
        return (f"none {sizes['none'][0]}, align-min {sizes['align-min'][0]}, "
                f"weights {sizes['loss-weight'][1]}")


def t_every_item_says_which_slide_it_came_from():
    """`slide_index` indexes `wsi_stems`, NOT the slides that happened to have
    tiles -- otherwise a rung filter that empties one slide silently shifts
    every other slide's index and a per-slide row describes the wrong slide."""
    with tempfile.TemporaryDirectory() as root:
        tiles_root, labels_root, _ = _make_stores(
            root, rungs=((1.0, 2), (4.0, 3)), stems=(_STEM_A, _STEM_B))
        stems = [_STEM_B, _STEM_A]              # deliberately not sorted
        dataset = PairDatasetConfig(tile=TILE, balance='none').build(
            tiles_root, labels_root, wsi_stems=stems)
        assert dataset.slides == stems, dataset.slides
        seen = collections.Counter()
        for i in range(len(dataset)):
            item = dataset.items[i]
            index = int(dataset[i]['slide_index'])
            assert stems[index] == item.slide, (index, item.slide)
            seen[item.slide] += 1
        assert seen == {_STEM_A: 5, _STEM_B: 5}, seen

        # the decoy: drop one slide's rungs and the OTHER slide's index must
        # not move. A dict built from what was found would renumber here.
        one = PairDatasetConfig(tile=TILE, balance='none').build(
            tiles_root, labels_root, wsi_stems=stems, rungs=[4.0])
        assert one.slides == stems, one.slides
        return (f'{seen[_STEM_A]} + {seen[_STEM_B]}, indices stable; '
                f'the second stem holds {_STEM_B.count(",")} commas')


def t_validation_splits_by_slide_and_the_parts_sum_to_the_whole():
    """The per-slide rows spec.md 1's fourth row needs.

    The silent failure is every pair landing in one group: the overall number
    is unchanged, `val/<one stem>/...` looks complete, and the other slide is
    simply absent -- which reads as "that slide had no pairs".
    """
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as root:
        tiles_root, labels_root, _ = _make_stores(
            root, rungs=((1.0, 3),), stems=(_STEM_A, _STEM_B))
        val = PairDatasetConfig(tile=TILE, balance='none', workers=0).build(
            tiles_root, labels_root, wsi_stems=[_STEM_A, _STEM_B])
        net = _net()
        trainer = TrainerConfig(batch_size=2, workers=0, amp=False,
                                wandb_mode='disabled').build(
            net, SuperPointLossConfig().build(), val, val, root)
        trainer.wandb_run = None
        row = trainer.validate()

    for stem in (_STEM_A, _STEM_B):
        assert f'val/{stem}/repeatability_margin' in row, sorted(row)
        assert f'val/{stem}/points_per_view' in row, sorted(row)
    parts = row[f'val/{_STEM_A}/n_pairs'] + row[f'val/{_STEM_B}/n_pairs']
    assert parts == row['val/n_pairs'], (parts, row['val/n_pairs'])
    assert row['val/n_pairs'] > 0, 'no pairs scored at all; this is vacuous'
    return (f"{row['val/n_pairs']:.0f} pairs = "
            f"{row[f'val/{_STEM_A}/n_pairs']:.0f} + "
            f"{row[f'val/{_STEM_B}/n_pairs']:.0f}   "
            f"(the second key has commas in it)")


def t_the_margin_ceiling_is_one_over_the_decoy():
    """Not arithmetic for its own sake: `margin = repeatability / decoy` and
    `repeatability <= 1`, so the decoy IS the ceiling -- at decoy 0.92 the whole
    measurable range is 1.10, which is what the 2026-08-28 run ran into. It is
    why the loop cuts to a fixed BUDGET rather than a threshold: a budget pins
    the density, and with it the ceiling, so this epoch and the last are on one
    scale."""
    row = _repeatability_row([0.9, 1.0], [0.5, 0.5], [100.0, 200.0],
                             [900.0, 1100.0], prefix='v/')
    assert abs(row['v/repeatability'] - 0.95) < 1e-9, row
    assert abs(row['v/repeatability_margin'] - 1.9) < 1e-9, row
    assert row['v/points_per_view'] == 150.0, row
    assert row['v/n_pairs'] == 2.0, row
    assert row['v/repeatability_margin'] <= 1.0 / row['v/repeatability_decoy']
    # `points_available` is the UNCAPPED count and is deliberately unrelated to
    # `points_per_view`: 1000 available against 150 measured is the normal
    # shape while a detector is still flat, and a row that forced them equal
    # would hide exactly that.
    assert row['v/points_available'] == 1000.0, row
    assert _repeatability_row([], [], [], [], prefix='v/') == {}
    return 'margin 1.9 under a ceiling of 2.0, 1000 available at a budget of 150'


def t_a_new_epoch_draws_a_new_warp_and_the_same_epoch_repeats():
    """The augmentation was frozen until 2026-08-31 and NOTHING SAID SO.

    `rng = default_rng((seed, index))` has no epoch term, so a tile is shown
    the SAME warp on every pass. 50 epochs over 5,344 pairs is 5,344 distinct
    pairs seen 50 times, and Homographic Adaptation's premise -- the same
    content under different geometry -- was being delivered as one geometry per
    tile. It surfaced only as textbook overfitting: `train/detector` fell to the
    last epoch while `val/detector` bottomed on epoch 42 and rose.

    Both directions are asserted. A fix that made every read random would also
    pass "the warps differ", and would throw away the reproducibility the seed
    is for -- so the same epoch read twice must give the same warp.
    """
    with tempfile.TemporaryDirectory() as root:
        tiles_root, labels_root, _ = _make_stores(root)
        data = PairDatasetConfig(tile=TILE, in_channels=1, balance='none',
                                 seed=0, workers=0).build(
                                     tiles_root, labels_root,
                                     wsi_stems=[_STEM_A])

        data.set_epoch(0)
        first = data[0]['warped_image']
        again = data[0]['warped_image']
        data.set_epoch(1)
        second = data[0]['warped_image']

    if not torch.equal(first, again):
        raise AssertionError(
            'the same item at the same epoch gave two different warps, so the '
            'seed is not doing its job and a run cannot be reproduced')
    if torch.equal(first, second):
        raise AssertionError(
            'epoch 0 and epoch 1 gave the SAME warp for the same tile. The '
            'epoch is not reaching the draw, and 250 epochs would be 250 '
            'passes over one augmentation')
    return 'same epoch identical, next epoch different'


def t_the_budget_and_the_available_count_are_different_numbers():
    """`points_per_view` is what was MEASURED, `points_available` is what the
    model would emit uncapped. Conflating them is what hid the 2026-08-31 run:
    the capped column read 420 on every tile -- the cap, to the integer -- while
    the real count was of order a thousand, and a count that lands exactly on
    the cap is the cap selecting rather than the model.

    Scored on a map with a KNOWN answer: a flat field just above the threshold,
    where every NMS survivor is available and the budget must still bind.
    """
    # 256 and not this file's TILE=64: the claim is about a budget of 200 and a
    # cap of 420 BINDING, and a 64 px tile holds only about forty NMS survivors
    # -- neither would bind and both assertions would pass vacuously.
    side = 256
    cfg = KeypointNetConfig.wired()
    budget = 200
    rng = np.random.default_rng(0)
    prob = (float(cfg.detection_threshold) * 2
            + rng.uniform(0, 1e-6, (side, side))).astype(np.float32)

    # The fixture asserts its own premise. A flat field just above the
    # threshold makes every survivor available by construction, but how MANY
    # survive is NMS geometry, and if that is under the cap then "the cap
    # bound" is untestable here.
    survivors, _, _ = points_from_prob(prob, None, score_threshold=0.0,
                                       nms_radius=cfg.nms_radius,
                                       border=cfg.border, max_points=None)
    if len(survivors) <= int(cfg.max_keypoints):
        raise AssertionError(
            f'NMS left {len(survivors)} survivors on a {side} px field, which '
            f'is not more than the cap of {cfg.max_keypoints}. Neither the '
            f'budget nor the cap would bind and this test would prove nothing')

    xy, available = _cut(prob, cfg, budget=budget)
    if len(xy) != budget:
        raise AssertionError(
            f'the budget asked for {budget} and got {len(xy)}; the density is '
            f'then not pinned and this epoch cannot be compared with the last')
    if available != len(survivors):
        raise AssertionError(
            f'{available} of {len(survivors)} survivors counted as available '
            f'on a field that is entirely above the threshold')

    # budget=0 falls back to the config's own rule, and `max_keypoints` is a
    # DIFFERENT number from the budget -- which is the whole reason both exist.
    capped, available_again = _cut(prob, cfg, budget=0)
    if available_again != available:
        raise AssertionError(
            f'the available count moved with the budget ({available} -> '
            f'{available_again}); it is supposed to be independent of it')
    if len(capped) != int(cfg.max_keypoints):
        raise AssertionError(
            f'budget=0 should fall back to the config rule and cap at '
            f'{cfg.max_keypoints}, got {len(capped)}')
    return (f'budget {budget}, cap {len(capped)}, available {available} '
            f'of {len(survivors)} survivors')


#: THE SECOND SLIDE'S NAME HAS COMMAS IN IT, AND THAT IS THE POINT.
#: `S1103627,G7E,110127` is a real Ki67 stem -- the comma is part of the NAME,
#: because the stem is `Path(wsi_path).stem` and the file is called that. Any
#: code that writes a list of stems by joining them with a delimiter produces a
#: string that cannot be split back, and the failure is silent: it has already
#: cost an awk parse of the Ki67 CSV and a re-eval run that scored 515 pairs
#: instead of 1044 and printed a complete table about half the held-out set.
#:
#: A fixture named `SLIDE_A` / `SLIDE_B` CANNOT REACH THAT BUG. Putting the
#: hazard in the default fixture turns "will someone remember" into "does the
#: test pass", which is the only version of that question with an answer.
#: `test_feature_store.py:35` already does this; these tests did not.
_STEM_A = 'SLIDE_A'
_STEM_B = 'S1103627,G7E,110127'


def _make_stores(root, rungs=((1.0, 2),), stems=(_STEM_A,)):
    """A pre-tile store and a matching label store, in a temp directory.

    Written through the real `PreTileStore` and `KeypointLabelStore` rather than
    by hand, so that a change to either format breaks this test instead of
    letting it test a shape nothing produces any more.
    """
    tiles_root = os.path.join(root, 'tiles')
    labels_root = os.path.join(root, 'labels')
    factor = 3
    pre_px = pre_tile_px(TILE, factor)
    margin = centre_margin(TILE, factor)
    rng = np.random.default_rng(0)
    points = np.array([[12, 40], [33, 20], [50, 51]], np.int16)

    for stem, (ds, count) in itertools.product(stems, rungs):
        meta = PreTileMeta(wsi_stem=stem, ds=float(ds), tile=TILE,
                           sampler_id='aaaa1111', seed=0, segmenter_id='seg0000',
                           pre_tile_factor=factor, level=0, level_ds=1.0,
                           shrink=1.0, read_size=pre_px)
        folder = PreTileStore.create(tiles_root, meta)
        records = []
        for i in range(count):
            image = rng.integers(0, 256, (pre_px, pre_px, 3), dtype=np.uint8)
            record = PreTileRecord(index=i, x=1000 * i, y=2000 * i)
            PreTileStore.save_tile(folder, record, image, meta)
            records.append(record)
        PreTileStore.write_index(folder, records, meta)

        batch = batch_from_lists(
            [(r.x, r.y) for r in records], [points] * count,
            [np.full(len(points), 0.5, np.float32)] * count,
            [np.full(len(points), 9, np.uint8)] * count, cap=16)
        KeypointLabelStore.save(labels_root, batch, LabelMeta(
            wsi_stem=stem, ds=float(ds), tile=TILE, ha_id='ha000000',
            pretile_id=meta.cfg_hash(), n_tiles=len(batch), cap=batch.cap))
    return tiles_root, labels_root, points


# ══════════════════════════════════════════════════════════════════════════════
#  8. train
# ══════════════════════════════════════════════════════════════════════════════

def t_one_batch_can_be_overfitted():
    """The only end-to-end evidence that the wiring works.

    Every check above is local: shapes, directions, one function against
    another. None of them would notice a loss that is computed correctly and
    connected to nothing -- a detached tensor, an optimiser over the wrong
    parameters, a frozen module. Twenty steps on ONE batch is what does.

    Scored against its own starting point rather than a target, because the
    number a real run reaches is what spec.md 12 step 6 is for.
    """
    torch.manual_seed(0)
    net = _net()
    loss = SuperPointLossConfig().build()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    keypoints = torch.zeros(2, TILE, TILE)
    keypoints[:, 9, 17] = 1.0
    keypoints[:, 40, 12] = 1.0
    batch = {'image': torch.rand(2, 1, TILE, TILE),
             'warped_image': torch.rand(2, 1, TILE, TILE),
             'keypoint_map': keypoints,
             'warped_keypoint_map': keypoints.clone(),
             'homography': torch.eye(3)[None].repeat(2, 1, 1)}

    first = last = None
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        total, parts = loss(net(batch['image']), net(batch['warped_image']),
                            batch, net.cell)
        total.backward()
        optimizer.step()
        if step == 0:
            first = parts
        last = parts

    assert last['detector'] < first['detector'], (first['detector'],
                                                  last['detector'])
    assert last['total'] < first['total'], (first['total'], last['total'])
    return (f"detector {first['detector']:.3f} -> {last['detector']:.3f}, "
            f"total {first['total']:.1f} -> {last['total']:.1f}")


def t_the_three_loss_magnitudes_are_reported_separately():
    """spec.md 12 step 6 asks for them after the first epoch, and a total alone
    cannot say which term is doing anything. `lambda_loss = 10000` compensates
    for the descriptor term's double normalisation, so the raw and the scaled
    descriptor numbers are both kept."""
    torch.manual_seed(0)
    net = _net()
    loss = SuperPointLossConfig().build()
    batch = {'image': torch.rand(1, 1, TILE, TILE),
             'warped_image': torch.rand(1, 1, TILE, TILE),
             'keypoint_map': torch.zeros(1, TILE, TILE),
             'warped_keypoint_map': torch.zeros(1, TILE, TILE),
             'homography': torch.eye(3)[None]}
    with torch.no_grad():
        _, parts = loss(net(batch['image']), net(batch['warped_image']),
                        batch, net.cell)
    for key in ('detector', 'warped_detector', 'descriptor',
                'descriptor_scaled', 'total'):
        assert key in parts, sorted(parts)
    ratio = parts['descriptor_scaled'] / max(parts['descriptor'], 1e-12)
    assert abs(ratio - 10000) < 1.0, ratio
    return (f"detector {parts['detector']:.3f}  descriptor "
            f"{parts['descriptor']:.2e} x10000 = {parts['descriptor_scaled']:.3f}")


def t_match_rate_is_the_max_norm_within_the_radius():
    a = np.array([[10, 10], [50, 50]], np.float64)
    assert _match_rate(a, a + 2, 4) == 1.0
    assert _match_rate(a, a + 9, 4) == 0.0
    assert _match_rate(a, np.array([[10, 14]], np.float64), 4) == 0.5
    return 'max-norm, radius 4'


# ══════════════════════════════════════════════════════════════════════════════

_SECTIONS = {
    'losses':     ['t_space_to_depth_is_the_exact_inverse_of_the_decode',
                   't_the_transposed_packing_is_a_different_answer',
                   't_cell_labels_put_an_empty_cell_in_the_dustbin',
                   't_the_tie_break_can_never_promote_a_dustbin',
                   't_the_valid_mask_is_anded_down_a_cell_not_averaged',
                   't_correspondence_is_the_diagonal_under_the_identity',
                   't_correspondence_moves_the_way_the_warp_moves',
                   't_the_descriptor_loss_prefers_agreeing_descriptors',
                   't_the_detector_loss_prefers_the_right_labels',
                   't_the_rung_weight_reaches_the_loss'],
    'heads':      ['t_descriptors_are_sampled_on_the_stride_and_not_on_the_cell',
                   't_a_descriptor_is_sampled_from_its_own_cell',
                   't_sampled_descriptors_are_unit_vectors',
                   't_the_head_normalises_its_output'],
    'interfaces': ['t_check_shapes_passes_a_consistent_stack',
                   't_check_shapes_catches_a_lying_stride',
                   't_check_shapes_catches_a_lying_width_and_an_unnormalised_head'],
    'backbone':   ['t_the_trunk_ends_at_128_and_not_at_256',
                   't_the_backbone_refuses_the_wrong_channel_count_and_size'],
    'decoder':    ['t_the_decoder_emits_the_dustbin_and_keeps_the_grid',
                   't_the_upsample_decoder_reaches_the_cell_and_refuses_what_it_cannot'],
    'net':        ['t_wired_builds_and_forward_has_the_shapes_the_loss_wants',
                   't_build_validates_the_widths_rather_than_repairing_them',
                   't_extraction_uses_the_same_rule_as_the_labels',
                   't_identity_moves_with_the_parts_and_not_with_the_extraction_alone'],
    'dataset':    ['t_splat_is_xy_and_clamps_to_the_frame',
                   't_the_pair_warps_points_and_not_the_map',
                   't_the_warped_valid_mask_is_nearly_full_because_of_the_pre_tile',
                   't_align_min_truncates_and_loss_weight_does_not',
                   't_every_item_says_which_slide_it_came_from',
                   't_a_new_epoch_draws_a_new_warp_and_the_same_epoch_repeats'],
    'train':      ['t_validation_splits_by_slide_and_the_parts_sum_to_the_whole',
                   't_the_margin_ceiling_is_one_over_the_decoy',
                   't_the_budget_and_the_available_count_are_different_numbers',
                   't_one_batch_can_be_overfitted',
                   't_the_three_loss_magnitudes_are_reported_separately',
                   't_match_rate_is_the_max_norm_within_the_radius'],
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
            check(name[2:].replace('_', ' '), globals()[name])

    failed = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
