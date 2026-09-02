"""The descriptor head, and the one correct way to read a vector out of it.

    head = DescriptorHeadConfig(in_channels=128).build()
    dense = head(features)                       # [N, 256, Hf, Wf], L2-normed
    vectors = sample_descriptors(xy, dense, stride)  # [N, K, 256], at the points

L2-NORMALISED INSIDE THE HEAD
------------------------------
Every consumer assumes it. The hinge loss is on dot products with
`positive_margin=1` and `negative_margin=0.2` (spec.md 9), and those numbers are
cosines -- on unnormalised vectors they mean whatever the current scale happens
to make them mean. Upstream normalises in the forward
(`superpoint_pytorch.py:108-110`) and again after interpolation
(`:19-21`), and both are here for the same reason: bilinear interpolation of
unit vectors does not produce a unit vector.

A head that skipped it would train. To something else.

SAMPLING, NOT UPSAMPLING
-------------------------
The dense map is at CELL resolution, and a keypoint is at pixel resolution.
Upstream's PyTorch version interpolates at the exact keypoint with
`grid_sample`; the TF version upsamples the whole map first. spec.md 9 takes the
PyTorch one -- cheaper, and the alignment is stated rather than implied.

That alignment is the part worth reading twice:

    (xy + 0.5) / (size * cell) * 2 - 1

`+ 0.5` moves from "pixel index" to "pixel centre", the division is by the
INPUT extent rather than the feature extent, and `align_corners=False` is what
makes those two conventions agree. Getting any of the three wrong shifts every
descriptor by up to half a cell -- four pixels at cell 8 -- and the matches it
produces are still matches, just to slightly the wrong place.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ConfigIdentity import IdentifiedConfig, register

from SuperPoint.Backbones import VggBlock

#: The zero point. ConfigIdentity rule 1.
_DESCRIPTOR_BASELINE = {
    'method': 'descriptor-vgg',
    'dim': 256,
    'hidden': 256,
    'in_channels': 128,
}


@register('descriptor-vgg')
@dataclass(frozen=True)
class DescriptorHeadConfig(IdentifiedConfig):
    """Upstream's descriptor head: 3x3 to `hidden`, 1x1 to `dim`, normalise.

    `dim` is 256 upstream. It is a config field and not a constant because
    spec.md 5.5 lists "descriptor dimension" as a thing that must be changeable
    without touching anything else -- and it is: nothing downstream reads a
    literal 256, the loss works on whatever `dim` is, and the store holds
    whatever the loss produced.
    """
    method: str = 'descriptor-vgg'
    dim: int = 256
    hidden: int = 256
    in_channels: int = 128

    def build(self) -> 'DescriptorHead':
        return DescriptorHead(self)


class DescriptorHead(nn.Module):
    """`[N, C, Hf, Wf]` -> `[N, dim, Hf, Wf]`, unit norm along `dim`."""

    def __init__(self, cfg: DescriptorHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.dim = int(cfg.dim)
        self.head = nn.Sequential(
            VggBlock(int(cfg.in_channels), int(cfg.hidden), 3),
            VggBlock(int(cfg.hidden), self.dim, 1, relu=False))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.head(features), p=2, dim=1)


def sample_descriptors(keypoints: torch.Tensor, dense: torch.Tensor,
                       stride: int) -> torch.Tensor:
    """Descriptors at `keypoints`. `[N, K, 2]` and `[N, D, Hf, Wf]` -> `[N, K, D]`.

    Transcribed from `superpoint_pytorch.py:11-22`, with the axis order flipped
    at the end so a caller gets `[N, K, D]` -- points along the middle axis,
    which is what every loss and every store here indexes by.

    `keypoints` are in INPUT pixels, not feature pixels. That is the whole
    reason the normalisation divides by `size * stride`.

    STRIDE, NOT CELL, AND UPSTREAM CANNOT TELL YOU WHICH
    -----------------------------------------------------
    `superpoint_pytorch.py:14` writes `s = self.conf.descriptor_dim`... no --
    it writes the SAME number for both, because upstream's VGG has stride 8 and
    upstream's detector has cell 8, so `w * 8` is right under either reading.
    They are two different quantities and this file is where they come apart:
    `dense` lives on the BACKBONE's grid, so the input extent it covers is
    `w * stride`. Dividing by `w * cell` instead is exact only when the two are
    equal.

    This was a live bug for the first backbone where they are not. GigaPath is
    stride 16 against cell 8, so the scale came out half of the real extent,
    every normalised coordinate came out twice too large, and `grid_sample`
    with the default padding sampled the whole descriptor set from a corner --
    silently, with unit-norm vectors of the right shape coming back. The
    matching would simply have been bad.

    The parameter is named `stride` so a caller reading the signature cannot
    reach for `cfg.detector.cell`, which is what the call site did.
    """
    b, c, h, w = dense.shape
    scale = keypoints.new_tensor([w, h]) * stride
    grid = (keypoints + 0.5) / scale
    grid = grid * 2 - 1                                    # to (-1, 1)
    sampled = torch.nn.functional.grid_sample(
        dense, grid.view(b, 1, -1, 2), mode='bilinear', align_corners=False)
    sampled = torch.nn.functional.normalize(
        sampled.reshape(b, c, -1), p=2, dim=1)
    return sampled.transpose(1, 2)
