"""The assembled student: backbone + detector decoder + descriptor head.

    net = KeypointNetConfig.wired().build(device)
    out = net(images)                       # KeypointOutput, dense, differentiable
    points = extract_keypoints(out, net.cfg)   # inference only

WHAT IS IN forward AND WHAT IS NOT
-----------------------------------
`forward` returns dense tensors and nothing else: `cell_logits` for the loss,
`prob_map` for looking at, `descriptors` for the hinge. NMS, top-k and border
removal are `extract_keypoints`, a separate function.

The reason is upstream's (spec.md 5.4) and it is not stylistic: the detector
loss is a cross-entropy over `cell_logits`, and if the forward thinned its own
output the loss would depend on a non-differentiable threshold. Training would
still run. The gradient would be describing a different function than the one
being evaluated.

IDENTITY LIVES HERE, NOT IN THE PARTS
--------------------------------------
`VggBackbone`, `DepthToSpaceDecoder` and `DescriptorHead` are plain
`nn.Module`s. This class is the `IdentifiedBuild`, because this is what a
checkpoint holds and what a label store's `ha_id` will point at in round 2. A
backbone that named itself would give a second answer to "which model is this",
and the two would drift the first time a head changed.

`identity_id()` folds the config AND the loaded weights (`IdentifiedBuild`),
which is what makes a fine-tuned copy a different model from the config it was
built from -- the case the base class docstring calls out.

THE TWO NUMBERS, WIRED ONCE
----------------------------
`backbone.out_channels` has to equal `detector.in_channels` and
`descriptor.in_channels`. `build` VALIDATES that rather than fixing it: silently
replacing the field would mean the config that was hashed is not the config that
ran, and the identity would name a model nobody built. `wired()` is the
constructor that gets it right in the first place.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ConfigIdentity import IdentifiedBuild, IdentifiedConfig, register

from common.Interfaces import check_shapes
from common.KeypointLabelStore import points_from_prob
from SuperPoint.Backbones import VggBackbone, VggBackboneConfig
from SuperPoint.Decoders import (DepthToSpaceDecoder, DepthToSpaceDecoderConfig,
                                 depth_to_space_prob)
from SuperPoint.Heads import (DescriptorHead, DescriptorHeadConfig,
                              sample_descriptors)


@dataclass
class KeypointOutput:
    """What one forward produces. All dense, all differentiable.

    `prob_map` is derived from `cell_logits` and is carried anyway, because
    deriving it twice -- once for a figure, once for a metric -- is two places
    for the decode to be spelled differently.
    """
    cell_logits: torch.Tensor              # [N, cell**2+1, Hc, Wc]  the loss eats this
    prob_map:    torch.Tensor              # [N, H, W]               decoded
    descriptors: Optional[torch.Tensor] = None   # [N, D, Hf, Wf], L2-normed
    survival:    Optional[torch.Tensor] = None   # [N, J, Hc, Wc], Stage C

    #: The BACKBONE's stride, carried because `descriptors` lives on the
    #: backbone's grid and sampling from it needs to know what one of its
    #: pixels is worth in input pixels. It is here rather than read off the
    #: config at the call site because `KeypointNetConfig` does not have it --
    #: a foundation trunk's stride is its patch size, which is a fact about the
    #: loaded model. The call site that DID read it off the config reached for
    #: `cfg.detector.cell`, which is a different number the moment a backbone
    #: has stride != cell.
    stride: int = 0


@dataclass
class Keypoints:
    """One image's points, after NMS and the cuts. Inference only."""
    xy:       np.ndarray                   # [K, 2] int16, input pixels
    score:    np.ndarray                   # [K] float32
    desc:     Optional[np.ndarray] = None  # [K, D] float32
    survival: Optional[np.ndarray] = None  # [K, J] float32

    def __len__(self) -> int:
        return int(len(self.xy))


#: The zero point. Nested configs take theirs from here, so the values below are
#: the ones that reproduce upstream (spec.md 9). ConfigIdentity rule 1: adding a
#: field needs a baseline entry that reproduces the old behaviour, or every
#: checkpoint identity re-hashes -- a recompute, not a wrong answer, but one to
#: know you are paying for.
_NET_BASELINE = {
    'method': 'superpathpoint',
    'backbone': VggBackboneConfig(),
    'detector': DepthToSpaceDecoderConfig(),
    'descriptor': DescriptorHeadConfig(),
    'nms_radius': 4,
    'detection_threshold': 0.005,
    'border': 4,
    'max_keypoints': 0,
}


@register('superpathpoint')
@dataclass(frozen=True)
class KeypointNetConfig(IdentifiedConfig):
    """Which parts, and how a map becomes points.

    The four extraction fields live here rather than in the decoder because they
    describe INFERENCE, not architecture -- two checkpoints that differ only in
    `nms_radius` are the same model read two ways. They are still hashed: a
    repeatability number is a property of the pair (model, extraction rule), and
    the labels were cut with a rule of their own (`LabelMeta.nms_radius`). Those
    two rules being the same number is what makes teacher and student
    comparable, and a field nobody hashed is a field that quietly stops
    matching.
    """
    method: str = 'superpathpoint'

    backbone:   VggBackboneConfig = field(default_factory=VggBackboneConfig)
    detector:   DepthToSpaceDecoderConfig = field(
        default_factory=DepthToSpaceDecoderConfig)
    descriptor: Optional[DescriptorHeadConfig] = field(
        default_factory=DescriptorHeadConfig)

    #: Upstream's four (spec.md 9). The BASELINE keeps upstream's values and is
    #: append-only (ConfigIdentity rule 1); these DEFAULTS are what runs, and
    #: two of them left upstream on 2026-08-29 for reasons measured here.
    nms_radius: int = 4
    border: int = 4

    #: 0.015, NOT upstream's 0.005 -- it is the value the LABELS were cut at
    #: (`make_ha_labels.DEFAULT_SCORE_THRESHOLD`, and `LabelMeta.score_threshold`
    #: records it per store). Student and teacher cutting a map by two different
    #: rules makes every repeatability number a comparison of two extractions as
    #: well as of two views.
    #:
    #: AND IT COLLIDES WITH THE VALUE OF TOTAL IGNORANCE, measured 2026-08-31.
    #: The detector is a `cell**2 + 1` = 65-way softmax per cell, so a model
    #: that has learnt nothing puts
    #:
    #:     1 / 65 = 0.015385
    #:
    #: on every class -- ABOVE this threshold. For an undertrained detector
    #: every cell therefore passes, NMS thins the field to a couple of thousand,
    #: and `max_keypoints` picks the top ones out of noise. That is exactly what
    #: the from-scratch arms did: `points_per_view` came back as the integer
    #: 420, the cap, on every tile, while their `val/detector` was 3.27 against
    #: `ln(65) = 4.174`.
    #:
    #: A well-trained detector is unaffected -- the `_pre` arms sat at CE 0.19
    #: and emitted 159 points, their own density. So this value is right for
    #: what it was chosen for (matching the labels) and unusable as an
    #: instrument on a model that has not converged. `cli/reeval_density.py`
    #: is the answer for measurement: it cuts to a fixed budget with the
    #: threshold at zero, which does not depend on where the softmax floor sits.
    detection_threshold: float = 0.015

    #: 420, NOT upstream's uncapped 0.
    #:
    #: WHERE THE NUMBER CAME FROM, CORRECTED 2026-08-31. It was taken from
    #: `n_kp mean 221 min 85 max 420` in the MakeHaLabels log and written up
    #: here as "the label corpus's own maximum". It is not. That line is
    #: `BRACS_1228 ds 4` -- one rung of one slide, and at the time the only
    #: store that existed (`1 (slide, rung) stores`, two lines above it in the
    #: same log). Over all 72 (slide, rung) stores the per-rung `n_kp` means
    #: run 3 to 527, the overall mean is 146, and the corpus maximum is 906.
    #:
    #: So 420 is neither the maximum nor a typical density; it is one cell of a
    #: 72-cell table, read before the other 71 existed. The cap never bound on
    #: the label side either way (`at-cap 0` in every cell), so the labels are
    #: unaffected -- what it bound was the STUDENT, and only because
    #: `detection_threshold` sits under 1/65 (above). Keeping the value pending
    #: the matched-budget table rather than replacing one guess with another:
    #: `cli/reeval_density.py` scores every arm at several fixed budgets, and a
    #: cap chosen from that is an argument about numbers.
    #:
    #: WHY A CAP AT ALL, AND WHY NOT 1966. The decoy's match rate rises with
    #: point density: it asks whether a point set shifted past the NMS radius
    #: matches anyway, and a dense enough set matches anything. Uniform-density
    #: arithmetic over the (2*nms_radius+1)^2 = 81 px match box in a 256^2 tile:
    #:
    #:     N/tile    pts/MP    decoy     margin ceiling = 1/decoy
    #:        146     2,228    0.163                6.13   corpus mean
    #:        221     3,372    0.239                4.18
    #:        420     6,409    0.405                2.47
    #:      1,966    29,999    0.912                1.10
    #:
    #: `margin = repeatability / decoy` and `repeatability <= 1`, so the decoy
    #: IS the ceiling of the metric. The 2026-08-28 run came back with decoy
    #: 0.92-0.94 and margin 1.03 against a ceiling of 1.10 -- a third of a ten
    #: per cent range, which is why epoch 0 and epoch 9 were indistinguishable.
    #: Copying the label side's 1966 would reproduce exactly that. 420 buys a
    #: ceiling of 2.47 while still capping nothing the teacher actually did.
    max_keypoints: int = 420

    @classmethod
    def wired(cls, *, in_channels: int = 1, cell: int = 8,
              descriptor_dim: int = 256, **over) -> 'KeypointNetConfig':
        """A consistent triple, so the widths cannot be set inconsistently.

        The alternative -- letting `build` overwrite `in_channels` -- would mean
        the config that was hashed is not the config that ran.
        """
        backbone = VggBackboneConfig(in_channels=int(in_channels))
        width = backbone.out_channels
        return cls(backbone=backbone,
                   detector=DepthToSpaceDecoderConfig(cell=int(cell),
                                                      in_channels=width),
                   descriptor=(None if descriptor_dim <= 0 else
                               DescriptorHeadConfig(dim=int(descriptor_dim),
                                                    in_channels=width)),
                   **over)

    def build(self, device=None) -> 'KeypointNet':
        return KeypointNet(self, device)


#: How an upstream key becomes a student key. The two models are the SAME
#: architecture -- `VggBackbone`'s widths are `[in, *channels[:-1]]`, its
#: `VggBlock` is conv -> ReLU -> BatchNorm(eps=0.001) with the module names
#: upstream uses, the detector emits `cell**2 + 1` and the descriptor 256 -- so
#: only the containers are named differently:
#:
#:     upstream  backbone.0.0.conv.weight     student  backbone.stages.0.0...
#:     upstream  detector.0.conv.weight       student  detector.head.0...
#:     upstream  descriptor.0.conv.weight     student  descriptor.head.0...
_UPSTREAM_PREFIX = (('backbone.', 'backbone.stages.'),
                    ('detector.', 'detector.head.'),
                    ('descriptor.', 'descriptor.head.'))

#: The one weight whose SHAPE differs, and only for the RGB student.
_FIRST_CONV = 'backbone.stages.0.0.conv.weight'


def upstream_state_dict(state: Dict[str, torch.Tensor], in_channels: int
                        ) -> Dict[str, torch.Tensor]:
    """Upstream SuperPoint v6's tensors, keyed and shaped for `KeypointNet`.

    RGB TAKES THE FIRST CONV REPEATED AND DIVIDED BY THREE. The released
    weights are 1 channel. `w/3` on each of three copies makes the 3-channel
    network's response to a luma image EXACTLY the 1-channel network's response
    to that image -- which is not a convenience, it is the assertion: the two
    students then differ only in what RGB adds, not in how their trunks were
    initialised. Initialising the first layer randomly instead would leave
    `rgb+pretrained` and `gray+pretrained` two different kinds of pretraining,
    and "RGB is worse" would be inseparable from "RGB's transfer was weaker".
    """
    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for old, new in _UPSTREAM_PREFIX:
            if key.startswith(old):
                key = new + key[len(old):]
                break
        out[key] = value

    first = out.get(_FIRST_CONV)
    if first is not None and int(in_channels) != int(first.shape[1]):
        if int(first.shape[1]) != 1 or int(in_channels) != 3:
            raise ValueError(
                f'cannot reshape {_FIRST_CONV} from {tuple(first.shape)} to '
                f'{in_channels} input channels; the only inflation defined '
                f'here is 1 -> 3 by repeat-and-divide')
        out[_FIRST_CONV] = first.repeat(1, 3, 1, 1) / 3.0
    return out


def load_upstream(net: 'KeypointNet', state: Dict[str, torch.Tensor]) -> None:
    """Load the released weights into a student. STRICT, and never partial.

    `strict=True` is the whole point. Under `strict=False` the keys that did not
    match stay randomly initialised, the network trains, the loss falls, and the
    result reads as "pretraining did not help" rather than as "pretraining did
    not happen". `test_pretrained_load` pins the same thing from the other side
    by comparing this student's `prob_map` with the teacher's `dense_prob`.
    """
    net.load_state_dict(
        upstream_state_dict(state, net.cfg.backbone.in_channels), strict=True)


class KeypointNet(nn.Module, IdentifiedBuild):
    """The model. `nn.Module` first, so `to()`, `train()` and `parameters()` win."""

    BASELINE = _NET_BASELINE

    def __init__(self, cfg: KeypointNetConfig, device=None):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device or 'cpu')

        self.backbone = cfg.backbone.build()
        width = self.backbone.out_channels
        _require(cfg.detector.in_channels == width,
                 f'detector.in_channels is {cfg.detector.in_channels} and the '
                 f'backbone emits {width}')
        self.detector = cfg.detector.build()
        self.descriptor = (cfg.descriptor.build() if cfg.descriptor else None)
        if cfg.descriptor:
            _require(cfg.descriptor.in_channels == width,
                     f'descriptor.in_channels is {cfg.descriptor.in_channels} '
                     f'and the backbone emits {width}')
        self.survival = None            # Stage C, spec.md 12 step 10

        self.cell = self.detector.cell
        self.stride = self.backbone.stride
        self.to(self.device)

        # The cheap assertion in front of the expensive run: one forward on a
        # tiny square, at construction. It catches a declared stride that is not
        # the real one -- which changes nothing that raises, and makes the cell
        # grid a different size than the labels were splatted onto.
        check_shapes(self.backbone, self.detector, self.descriptor,
                     image_size=max(self.cell, self.stride) * 2,
                     channels=cfg.backbone.in_channels, device=self.device)

    # `IdentifiedBuild.weights_id` reads `self.model`; for this class the model
    # IS self. Named rather than special-cased there, because that base class is
    # shared with the encoders and must not learn about this one.
    @property
    def model(self) -> nn.Module:
        return self

    def forward(self, images: torch.Tensor) -> KeypointOutput:
        features = self.backbone(images)
        logits = self.detector(features)
        return KeypointOutput(
            cell_logits=logits,
            prob_map=depth_to_space_prob(logits, self.cell),
            descriptors=(self.descriptor(features) if self.descriptor else None),
            stride=self.stride)

    def summary(self) -> str:
        params = sum(p.numel() for p in self.parameters())
        return (f'superpathpoint  stride {self.stride}  cell {self.cell}  '
                f'{params / 1e6:.2f}M params  {self.identity_id()}')


def extract_keypoints(output: KeypointOutput, cfg: KeypointNetConfig
                      ) -> List[Keypoints]:
    """Dense output -> one `Keypoints` per image. INFERENCE ONLY.

    Goes through `common.KeypointLabelStore.points_from_prob`, which is the same
    function that cut the teacher's labels. That is the point of the arrow
    (spec.md 14): a repeatability number compares a student's points with a
    teacher's, and if the two were thinned by two implementations of NMS the
    number would be measuring the difference between the implementations.

    The round trip through numpy is deliberate and is why this is not in
    `forward`: it costs a device-to-host copy per image, which is nothing at
    inference and would be the whole budget in a training loop.
    """
    prob = output.prob_map.detach().float().cpu().numpy()
    dense = (output.descriptors.detach() if output.descriptors is not None
             else None)
    stride = int(output.stride)
    if dense is not None and stride <= 0:
        raise ValueError(
            'KeypointOutput.stride is unset and there are descriptors to '
            'sample. It is the backbone grid the dense map lives on, and '
            'guessing it from cell is right only when stride == cell')

    out: List[Keypoints] = []
    for i in range(prob.shape[0]):
        xy, score, _ = points_from_prob(
            prob[i], None,
            score_threshold=cfg.detection_threshold,
            nms_radius=cfg.nms_radius, border=cfg.border,
            max_points=cfg.max_keypoints or None)

        desc = None
        if dense is not None and len(xy):
            points = torch.from_numpy(xy.astype(np.float32))[None].to(dense.device)
            desc = sample_descriptors(points, dense[i:i + 1],
                                      stride)[0].cpu().numpy()
        out.append(Keypoints(xy=xy, score=score, desc=desc))
    return out


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(
            f'{message}. Build the config with KeypointNetConfig.wired(), which '
            f'derives the widths from the backbone -- build() validates rather '
            f'than repairs, because a repaired config is not the config that '
            f'was hashed')
