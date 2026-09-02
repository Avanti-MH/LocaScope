"""Feature map -> detector cell logits, for a backbone of any stride.

    decoder = DepthToSpaceDecoderConfig(in_channels=128).build()
    logits = decoder(features)                  # [N, cell**2 + 1, Hc, Wc]
    prob = depth_to_space_prob(logits, decoder.cell)   # [N, H, W]

TWO NUMBERS, AND THIS FILE IS WHERE THEY ARE ALLOWED TO DIFFER
----------------------------------------------------------------
`stride` belongs to the backbone; `cell` belongs to the labels. Upstream ties
them at 8 and never has to say so (spec.md 5.1). A ViT has a stride of 14 or 16,
and a decoder that inherited it would predict a 14 px cell while the labels were
splatted onto an 8 px grid -- self-consistent, comparable to nothing, and silent.

    DepthToSpaceDecoder   stride == cell. Upstream's head, transcribed.
    UpsampleDecoder       stride  > cell. A small transposed-conv tower brings
                          the map down to `cell`, then the same head.

THE DUSTBIN IS IN BOTH, AND IT IS NOT DECORATION
--------------------------------------------------
`cell**2 + 1` channels: one per pixel of the cell, plus one for "no keypoint in
this cell". That extra class is what lets a softmax say "nothing here" instead of
a threshold carving it out afterwards, and upstream's detector loss is a plain
cross-entropy over exactly these channels (spec.md 5.2). Dropping it is not a
simplification of the loss, it is a different loss.

It is also why `depth_to_space_prob` drops the dustbin WITHOUT renormalising
(`models/utils.py:21-26`): after the drop the values are "the probability of a
keypoint at this pixel", and an empty cell correctly sums to near zero.
Renormalising would push every empty cell back to one -- a detector that fires
confidently on blank glass, with nothing raising anywhere.

WHY THE DECODE LIVES HERE AND THE TEACHER IMPORTS IT
------------------------------------------------------
Three callers: the teacher's dense map, the student's `prob_map`, and
`test_detector_decoder`. The decode IS the detector decoder, so this is its
module, and the alternative -- a copy in each -- is two chances for the two
permutes to be written in opposite orders. The teacher would then be scored
against a student whose points are transposed inside every cell, and the only
symptom is a repeatability number that is worse than it should be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from ConfigIdentity import IdentifiedConfig, register

from SuperPoint.Backbones import VggBlock


def depth_to_space_prob(logits: torch.Tensor, cell: int) -> torch.Tensor:
    """`[N, cell**2+1, Hc, Wc]` logits -> `[N, Hc*cell, Wc*cell]` probabilities.

    Softmax over the channel axis, drop the last channel, then depth-to-space.
    Transcribed from `superpoint_pytorch.py:114-121`, itself the PyTorch
    spelling of `tf.depth_to_space` in NHWC (`models/utils.py:21-26`).

    Channel `c` lands at `(dy, dx) = (c // cell, c % cell)` within its cell: the
    reshape puts the channel's high digit on the 4th axis and the second permute
    interleaves that one with the ROW axis.

    THE TWO PERMUTES IN THE OTHER ORDER RUN AND ARE WRONG. They transpose every
    keypoint inside its own cell -- right shape, right count, plausible
    positions, every label 0 to `cell-1` px off in a pattern that depends on
    where it is. `test_detector_decoder` pins it against that exact decoy, which
    is the only way to see it: no assertion about shapes or counts can.
    """
    if logits.shape[1] != cell * cell + 1:
        raise ValueError(
            f'{logits.shape[1]} channels for cell {cell}; expected '
            f'{cell * cell + 1} = {cell}^2 cells plus the dustbin')
    scores = torch.nn.functional.softmax(logits, 1)[:, :-1]
    b, _, h, w = scores.shape
    scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, cell, cell)
    scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * cell, w * cell)
    return scores


# ── stride == cell ───────────────────────────────────────────────────────────

_D2S_BASELINE = {
    'method': 'depth-to-space',
    'cell': 8,
    'hidden': 256,
    'in_channels': 128,
}


@register('depth-to-space')
@dataclass(frozen=True)
class DepthToSpaceDecoderConfig(IdentifiedConfig):
    """Upstream's detector head: `VggBlock(C, hidden, 3)` then a 1x1 to the cells.

    `hidden` is `conf.channels[-1]` upstream -- the 256 that the backbone does
    NOT end at (spec.md 9). It is a field here rather than read off the
    backbone's config, because a decoder has to be constructible against any
    backbone; `in_channels` is the coupling, and it is one number.
    """
    method: str = 'depth-to-space'
    cell: int = 8
    hidden: int = 256
    in_channels: int = 128

    def build(self) -> 'DepthToSpaceDecoder':
        return DepthToSpaceDecoder(self)


class DepthToSpaceDecoder(nn.Module):
    """`[N, C, Hf, Wf]` -> `[N, cell**2 + 1, Hf, Wf]`. The grid is unchanged.

    Which is the point: this decoder is for a backbone whose stride already IS
    the cell, so there is nothing to resample and the head is two convolutions.
    """

    def __init__(self, cfg: DepthToSpaceDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.cell = int(cfg.cell)
        self.head = nn.Sequential(
            VggBlock(int(cfg.in_channels), int(cfg.hidden), 3),
            VggBlock(int(cfg.hidden), self.cell ** 2 + 1, 1, relu=False))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        return depth_to_space_prob(logits, self.cell)


# ── stride > cell ────────────────────────────────────────────────────────────

_UPSAMPLE_BASELINE = {
    'method': 'upsample-d2s',
    'cell': 8,
    'stride': 16,
    'hidden': 256,
    'in_channels': 1536,
    'norm_groups': 8,
}


@register('upsample-d2s')
@dataclass(frozen=True)
class UpsampleDecoderConfig(IdentifiedConfig):
    """A transposed-conv tower down to `cell`, then the same head.

    NO UPSTREAM TO COPY. This is the one genuinely new piece in the
    foundation-model path (spec.md 5.2) and the most likely thing to need a
    second version. The first version is deliberately dull: `ConvTranspose2d`
    with stride 2, `GroupNorm`, `GELU`, channels halving each rung.

    `stride / cell` must be a power of two, because that is what a stack of
    stride-2 transposed convolutions can express. A ViT at stride 14 does not
    satisfy that against cell 8 -- 14/8 is not a power of two -- and NO tile size
    repairs it, because the tile cancels out of `(tile/cell) / (tile/stride)`.
    An earlier version of this paragraph said such a backbone could be 'fed a
    crop whose stride divides evenly', which is wrong: a crop changes the tile
    and the tile is not in the ratio. The three things that would work are in
    `SuperPoint/EncoderBackbone.py` -- resize the image, resize the features
    here, or give that one backbone its own cell -- and none is chosen, because
    the patch-16 encoders need none of them. Refused rather than rounded:
    rounding here would put the cell grid a fraction off the labels everywhere,
    uniformly, invisibly.
    """
    method: str = 'upsample-d2s'
    cell: int = 8
    stride: int = 16
    hidden: int = 256
    in_channels: int = 1536
    norm_groups: int = 8

    def build(self) -> 'UpsampleDecoder':
        return UpsampleDecoder(self)


class UpsampleDecoder(nn.Module):
    """`[N, C, H/stride, W/stride]` -> `[N, cell**2 + 1, H/cell, W/cell]`."""

    def __init__(self, cfg: UpsampleDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.cell = int(cfg.cell)

        factor, remainder = divmod(int(cfg.stride), self.cell)
        if remainder or factor < 1 or (factor & (factor - 1)):
            raise ValueError(
                f'stride {cfg.stride} over cell {cfg.cell} is {cfg.stride / cfg.cell:g}, '
                f'which a stack of stride-2 transposed convolutions cannot '
                f'express. Feed the backbone a size whose stride divides the '
                f'cell evenly, or give this decoder an explicit resize -- '
                f'rounding it would offset the cell grid from the labels '
                f'everywhere and uniformly')

        width = int(cfg.in_channels)
        rungs = []
        while factor > 1:
            out = max(int(cfg.hidden), width // 2)
            rungs += [nn.ConvTranspose2d(width, out, kernel_size=2, stride=2),
                      nn.GroupNorm(_groups(out, int(cfg.norm_groups)), out),
                      nn.GELU()]
            width, factor = out, factor // 2
        self.up = nn.Sequential(*rungs)
        self.head = nn.Sequential(
            VggBlock(width, int(cfg.hidden), 3),
            VggBlock(int(cfg.hidden), self.cell ** 2 + 1, 1, relu=False))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.up(features))

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        return depth_to_space_prob(logits, self.cell)


def _groups(channels: int, wanted: int) -> int:
    """The largest divisor of `channels` that is at most `wanted`.

    `GroupNorm` raises when the groups do not divide the channels, and the
    channel counts here come from halving a foundation model's width -- 1536,
    768, 384 divide by 8, but a config that set `hidden=200` would not. Falling
    back to a divisor keeps the failure out of the middle of a training run.
    """
    for groups in range(min(wanted, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1
