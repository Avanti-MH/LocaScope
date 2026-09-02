"""The three protocols that make the encoder and the decoder swappable.

    class Backbone(Protocol):        images -> a feature map
    class DetectorDecoder(Protocol): a feature map -> cell logits
    class DescriptorHead(Protocol):  a feature map -> dense descriptors

Runtime-checkable Protocols rather than base classes: `TileEncoderBackbone`
wraps `aiNNModel/TileEncoderFunc.py`, which is not ours to give a base class to,
and inheritance would put this module in `aiNNModel`'s import path for no gain.
`check_shapes` below is what actually enforces anything.

`stride` AND `cell` ARE TWO NUMBERS
------------------------------------
This is the whole of the swappability, in one line. Upstream ties them together
at 8 -- three max-pools in the VGG make the stride 8, and the detector head
emits `1 + 8**2 = 65` channels -- so it never had to name them separately.

    stride   input pixels per FEATURE pixel. A property of the backbone.
    cell     input pixels per detector PREDICTION. The head emits `cell**2 + 1`
             channels on a `H/cell` grid and `depth_to_space` unpacks them, so
             `cell` is the side of the square block that one softmax covers --
             at most one keypoint per block, anywhere inside it to the pixel.

It is NOT the upsampling kernel. `UpsampleDecoder` uses `kernel_size=2,
stride=2` throughout and the only thing `cell` decides there is how many rungs:
`log2(stride / cell)`, which is 1 for stride 16 and 0 for stride 8.

CELL IS PER MODEL, NOT PER PROJECT
------------------------------------
An earlier version of this paragraph said `cell` was "a property of the labels
and therefore fixed across every backbone that is to be compared". The first
half is wrong: the labels are POINTS (`KeypointLabelStore` holds `kp_xy`), and
`Losses.cell_labels` folds them into cells at loss time, so changing `cell`
re-discretises nothing that is stored. Only the second half survives, as a
caveat rather than a constraint -- two models with different `cell` differ in
one more way than the thing being compared, and that has to be SAID rather than
prevented, because preventing it is what forces a resize somewhere else.

What actually changes with `cell`: the ceiling on keypoint density. One
prediction per `cell**2` pixels, and `cell_labels`' argmax drops the rest of a
crowded cell (a random one survives each step, so it is a stochastic subsample
of the supervision, not a deletion). That is why uni2 takes cell 7 and not
cell 14 -- both give `UpsampleDecoder` a power-of-two climb from stride 14, but
14 is 3.06x coarser than cell 8 while 7 is 1.31x FINER, and a comparison about
keypoints should not hand one arm a lower ceiling on keypoints:

    backbone   stride  cell   climb   px per prediction   vs cell 8
    VGG            8      8     1x                   64      1.00x
    gigapath      16      8     2x                   64      1.00x
    uni2          14      7     2x                   49      1.31x
    uni2          14     14     1x                  196      0.33x   <- rejected

Whether the ceiling binds at all is a measurement -- `cli/inspect_ha_labels.py`'s
`n_kp` histogram against the cell count -- and it has not been made.

Keeping them apart is what `UpsampleDecoder` exists for (spec.md 5.2): it climbs
from a stride-16 feature map to a stride-8 cell grid, and then the depth-to-space
is the same operation for every backbone.

STRIDE 16, NOT 14. An earlier version of this paragraph said 14, and that is the
one thing `UpsampleDecoder` cannot do. The climb is `stride / cell`, which a
stack of stride-2 transposed convolutions can express only when it is a power of
two: 16/8 = 2 is, 14/8 = 1.75 is not, and the tile size cancels out of that ratio
so no choice of tile fixes it. `SuperPoint/EncoderBackbone.py` has the three
things that would, none of them chosen, and `UpsampleDecoderConfig` refuses the
ratio rather than rounding it.

WHAT IS NOT IN HERE
--------------------
NMS, top-k and border removal. They belong to `extract_keypoints`, not to any
forward pass: training consumes the dense `cell_logits`, and folding a
non-differentiable threshold into the model would make the loss depend on it
(spec.md 5.4).
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable

import torch


@runtime_checkable
class Backbone(Protocol):
    """`[N, C_in, H, W]` -> `[N, out_channels, H/stride, W/stride]`.

    `trainable` is a fact about the implementation, not a request: a foundation
    model's trunk is frozen (spec.md 5.3), and 1.1B parameters of GigaPath is
    not a thing this machine trains. The Trainer reads it to decide which
    parameters to hand the optimiser, so an implementation that lies here gets
    an optimiser over frozen tensors and a loss that never moves.
    """
    out_channels: int
    stride: int
    trainable: bool

    def forward(self, images: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class DetectorDecoder(Protocol):
    """`[N, C, Hf, Wf]` -> `[N, cell**2 + 1, Hc, Wc]`.

    The `+ 1` is the dustbin and it is not optional. It makes "no keypoint in
    this cell" a class the softmax can predict rather than something a threshold
    carves out afterwards, and upstream's whole detector loss is a
    cross-entropy over those `cell**2 + 1` channels (spec.md 5.2). Removing it
    means designing a different loss, not simplifying this one.

    `Hc, Wc` need not equal `Hf, Wf`: an upsampling decoder changes the grid on
    the way through, which is how a stride-16 backbone reaches a stride-8 cell.
    Only by a power of two -- see the note above about 14.
    """
    cell: int

    def forward(self, features: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class DescriptorHead(Protocol):
    """`[N, C, Hf, Wf]` -> `[N, dim, Hf, Wf]`, L2-normalised along `dim`.

    Normalised INSIDE the head, because every consumer assumes it: the hinge
    loss is on dot products, and `sample_descriptors` normalises again after
    interpolating (`superpoint_pytorch.py:11-22`). A head that returned raw
    vectors would train -- to a different scale, with the margins meaning
    something else.
    """
    dim: int

    def forward(self, features: torch.Tensor) -> torch.Tensor: ...


class ShapeMismatch(RuntimeError):
    """An implementation whose numbers do not match what it declares."""


def check_shapes(backbone: Backbone,
                 detector: Optional[DetectorDecoder] = None,
                 descriptor: Optional[DescriptorHead] = None,
                 *, image_size: int = 64, channels: int = 3,
                 device=None) -> Tuple[int, ...]:
    """Run one tiny batch and check every declared number against reality.

    A Protocol is a promise; this is the part that collects on it. The three
    things it catches are all silent:

        `out_channels` that disagrees with the tensor
            the detector's first conv is built from the declared number, so a
            wrong one raises at CONSTRUCTION -- but only if the two are built
            together, and a bench that builds them separately gets a shape error
            somewhere else entirely
        `stride` that disagrees with the tensor
            everything still runs. The cell grid is a different size than the
            labels were splatted onto, and every keypoint is off by a factor
        a descriptor head that forgot to normalise
            the hinge loss still trains, to a different scale, with
            `positive_margin=1` meaning something other than what it means
            upstream

    Cheap enough to run at construction: one forward on a `image_size` square.
    """
    device = device or torch.device('cpu')
    images = torch.zeros(1, channels, image_size, image_size, device=device)

    features = backbone.forward(images)
    if features.ndim != 4:
        raise ShapeMismatch(
            f'backbone returned {features.ndim} dims; a feature MAP is 4')
    if features.shape[1] != backbone.out_channels:
        raise ShapeMismatch(
            f'backbone declares out_channels={backbone.out_channels} and '
            f'returned {features.shape[1]}')
    got_stride = image_size / features.shape[-1]
    if abs(got_stride - backbone.stride) > 1e-6:
        raise ShapeMismatch(
            f'backbone declares stride={backbone.stride} but {image_size} px '
            f'came out {features.shape[-1]} px wide, i.e. stride {got_stride:g}. '
            f'Nothing downstream would raise: the cell grid would simply be a '
            f'different size than the labels were splatted onto')

    if detector is not None:
        logits = detector.forward(features)
        want = detector.cell ** 2 + 1
        if logits.shape[1] != want:
            raise ShapeMismatch(
                f'detector declares cell={detector.cell} so it must emit '
                f'{want} channels ({detector.cell}^2 cells plus the dustbin); '
                f'it emitted {logits.shape[1]}')
        expected = image_size // detector.cell
        if logits.shape[-1] != expected:
            raise ShapeMismatch(
                f'detector emitted a {logits.shape[-1]} wide cell grid; '
                f'cell={detector.cell} on a {image_size} px input means '
                f'{expected}')

    if descriptor is not None:
        dense = descriptor.forward(features)
        if dense.shape[1] != descriptor.dim:
            raise ShapeMismatch(
                f'descriptor declares dim={descriptor.dim} and returned '
                f'{dense.shape[1]}')
        norms = dense.float().pow(2).sum(dim=1).sqrt()
        if float((norms - 1.0).abs().max()) > 1e-3:
            raise ShapeMismatch(
                f'descriptor vectors have norms in '
                f'[{float(norms.min()):.3f}, {float(norms.max()):.3f}]; the '
                f'head must L2-normalise, because the hinge margins are '
                f'defined on unit vectors')

    return tuple(features.shape)
