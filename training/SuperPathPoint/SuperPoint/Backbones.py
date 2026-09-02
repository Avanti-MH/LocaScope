"""The student's feature extractor. v1 is upstream's VGG, trained from scratch.

    backbone = VggBackboneConfig().build()
    features = backbone(images)     # [N, 3, H, W] -> [N, 128, H/8, W/8]

WHY 128 AND NOT 256
--------------------
`conf.channels` is `[64, 64, 128, 128, 256]` and the last entry is the HEAD's
hidden width, not the backbone's output:

    channels = [1, *conf.channels[:-1]]        superpoint_pytorch.py:83

so the trunk is four stages ending at 128, and each head lifts 128 to 256 on its
own (spec.md 9). Reading the list the obvious way builds a backbone one stage
deeper and twice as wide. That misreading does not raise -- the model trains, on
a different architecture -- which is why the number is spelled out in
`out_channels` and checked by `common.Interfaces.check_shapes`.

`stride` COMES FROM THE SAME LIST, WHICH IS WHY IT IS DERIVED
--------------------------------------------------------------
`2 ** (len(channels) - 2)` = 8: three max-pools between four stages. Changing the
LENGTH of `channels` therefore changes the stride and, upstream, the number of
detector channels with it. Here it changes only the stride -- `cell` belongs to
the decoder (spec.md 5.1) -- so a deeper trunk stays compatible with the same
labels as long as a decoder bridges the difference.

CONV -> RELU -> BATCHNORM, WHICH IS NOT THE USUAL ORDER
--------------------------------------------------------
`superpoint_pytorch.py:50-65` puts the activation BEFORE the norm, with
`eps=0.001`. Kept, deliberately: the teacher's released weights are in this
architecture, and the repeatability numbers of spec.md 1 compare a student
against that teacher on the same tiles. Swapping in the conventional order would
add an unmeasured architectural difference to a comparison whose whole point is
to isolate a different one.

It is one line to change and nobody has measured it here, so it is not a
finding -- it is a decision to keep the comparison clean.

WHERE THE FOUNDATION-MODEL BACKBONE GOES
-----------------------------------------
`TileEncoderBackbone` (spec.md 5.3) is step 8 of spec.md 12, after the 256 line
works end to end, and it is deliberately not in this file yet. There is an open
question in front of it that writing it now would answer by accident:
`TileEncoder.spatial()` is an INFERENCE api -- `_run` applies the config's
transform (resize and centre-crop to `crop_size`, 224) and returns to the host --
so it cannot be dropped into a training loop that feeds 256 px tiles and wants
the map on the device. Reaching past it to `_spatial_forward` works and touches
nothing in `aiNNModel/`, but it is a private method; the alternative is a change
in `aiNNModel/TileEncoderFunc.py`. That is a decision, not an implementation
detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from ConfigIdentity import IdentifiedConfig, register


class VggBlock(nn.Sequential):
    """conv -> activation -> BatchNorm, upstream's order and upstream's eps.

    Shared by the backbone, the detector decoder and the descriptor head, which
    is how upstream has it: `VGGBlock` is the only conv unit in the file
    (`superpoint_pytorch.py:50-65`). Named parts, because a state dict keyed on
    `0`, `1`, `2` is a state dict nobody can read.
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int,
                 relu: bool = True):
        padding = (kernel_size - 1) // 2
        super().__init__()
        self.add_module('conv', nn.Conv2d(c_in, c_out, kernel_size=kernel_size,
                                          stride=1, padding=padding))
        self.add_module('activation',
                        nn.ReLU(inplace=True) if relu else nn.Identity())
        self.add_module('bn', nn.BatchNorm2d(c_out, eps=0.001))


#: The zero point. ConfigIdentity rule 1: editing this re-hashes every
#: checkpoint identity ever written; editing a dataclass default splits new from
#: old. The values are upstream's (spec.md 9).
_VGG_BASELINE = {
    'method': 'vgg',
    'channels': (64, 64, 128, 128, 256),
    'in_channels': 1,
}


@register('vgg')
@dataclass(frozen=True)
class VggBackboneConfig(IdentifiedConfig):
    """Upstream's trunk. `channels` is upstream's whole list, head width and all.

    Kept whole rather than trimmed to the four the trunk uses, because it is the
    number `stride` is derived from and because the trimming is the exact
    misreading this file warns about -- a config that had already dropped the
    256 would make `channels[:-1]` wrong in a second place.
    """
    method: str = 'vgg'
    channels: Tuple[int, ...] = (64, 64, 128, 128, 256)

    #: 1 for the grayscale student, 3 for the RGB one. Both are trained
    #: (spec.md 13): H&E's nuclei boundaries survive a luma conversion and
    #: Ki67's DAB brown against a blue counterstain may not, and which of those
    #: dominates is a measurement nobody here has made.
    in_channels: int = 1

    def build(self) -> 'VggBackbone':
        return VggBackbone(self)

    @property
    def stride(self) -> int:
        return 2 ** (len(self.channels) - 2)

    @property
    def out_channels(self) -> int:
        return int(self.channels[-2])


class VggBackbone(nn.Module):
    """`[N, in_channels, H, W]` -> `[N, out_channels, H/stride, W/stride]`.

    A plain `nn.Module` and not an `IdentifiedBuild`: identity belongs to the
    assembled `KeypointNet`, which is what a checkpoint holds. A backbone that
    named itself would give two answers to "which model is this".
    """

    trainable = True

    def __init__(self, cfg: VggBackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.stride = cfg.stride
        self.out_channels = cfg.out_channels

        widths = [int(cfg.in_channels), *[int(c) for c in cfg.channels[:-1]]]
        stages = []
        for i, width in enumerate(widths[1:], 1):
            layers = [VggBlock(widths[i - 1], width, 3),
                      VggBlock(width, width, 3)]
            if i < len(widths) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] != self.cfg.in_channels:
            raise ValueError(
                f'this backbone takes {self.cfg.in_channels} channel(s) and got '
                f'{images.shape[1]}. The grayscale and RGB students are two '
                f'configs and two checkpoints, not one model taking either')
        h, w = images.shape[-2:]
        if h % self.stride or w % self.stride:
            raise ValueError(
                f'{h}x{w} is not a multiple of the stride {self.stride}. The '
                f'max-pools would floor the odd row and column away, and the '
                f'cell grid would then cover slightly less than the image -- '
                f'with every label still splatted onto the full extent')
        return self.stages(images)


def to_model_channels(images: torch.Tensor, in_channels: int) -> torch.Tensor:
    """RGB or gray in, `in_channels` out, with upstream's luma.

    One definition, because the grayscale student and the teacher both need it
    and they must agree: a mean of the three channels is a different input
    distribution from 0.299/0.587/0.114, and H&E's pink and purple are exactly
    where the two differ most. `Teacher._to_gray` is the same three numbers, on
    the other side of a package boundary it does not import across.
    """
    if images.shape[1] == in_channels:
        return images
    if images.shape[1] == 3 and in_channels == 1:
        scale = images.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (images * scale).sum(1, keepdim=True)
    raise ValueError(
        f'cannot turn {images.shape[1]} channels into {in_channels}; the only '
        f'conversion defined here is RGB to luma')
