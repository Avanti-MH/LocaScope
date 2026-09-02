"""The bootstrap detector: upstream SuperPoint v6, loaded and asked for a DENSE map.

    teacher = TeacherConfig().build(device)
    prob = teacher.dense_prob(images)      # [B, H, W] float32, no NMS, no cut

WHY A TEACHER MODULE AT ALL
----------------------------
spec.md 3.1 skips upstream's stage 1 (MagicPoint on synthetic shapes) and starts
from the released weights. So exactly one thing in this project loads
`superpoint_v6_from_tf.pth`, and it is not the student: the student is built by
`KeypointNet.py` out of parts this repo owns, and it never shares a checkpoint
with the teacher. Keeping them in one class would make round 2 of Stage A --
where the student BECOMES the teacher -- look like a state change on one object
rather than what it is.

WHY THE MODULE IS BORROWED AND THE DECODE IS NOT
-------------------------------------------------
The network itself is imported from `/work/u26130998/SuperPoint/
superpoint_pytorch.py` rather than re-declared here. Re-declaring it means
re-typing `VGGBlock`'s conv -> ReLU -> BatchNorm order (which is unusual: the
activation comes BEFORE the norm, `superpoint_pytorch.py:50-65`), `eps=0.001`,
and `channels = [1, *conf.channels[:-1]]` -- and every one of those has to be
byte-exact or the state dict silently fails to load, or worse, loads into a
model that is a different model. Borrowing makes that impossible.

What is NOT borrowed is `SuperPoint.forward`. It returns keypoints: softmax,
depth-to-space, NMS at radius 4, border removal at 4 px, then a threshold at
0.005. Homographic Adaptation needs the map from BEFORE all four of those:

    upstream TF          `net(image)['prob']` -- `models/utils.py:21-26`,
                         softmax and depth_to_space and nothing else
    used by HA at        `models/homographies.py:42` and `:74`
    NMS happens          in `magic_point.py:39-42`, on the AGGREGATED prob,
                         after homography_adaptation has returned

Running NMS inside HA would suppress a peak in one warped view that a hundred
other views agree on -- the aggregate is exactly where the evidence gets
combined, so thinning before it throws away what HA is for. `dense_prob` is
therefore assembled here out of `model.backbone` and `model.detector`, and the
decode itself comes from `Decoders.depth_to_space_prob` -- the student uses the
same function, and two transcriptions of those two permutes would eventually be
written in opposite orders.

TWO THINGS THAT WOULD RUN AND BE WRONG
---------------------------------------
1. The decode. Getting the two permutes the wrong way round transposes every
   keypoint inside its 8x8 cell -- see `Decoders.depth_to_space_prob`, which is
   where that lives and where `test_detector_decoder` pins it.
2. Grayscale. The released weights are 1 channel (`superpoint_pytorch.py:83`),
   and feeding them a mean of the three channels rather than upstream's
   0.299/0.587/0.114 luma is a different input distribution -- with no error, and
   with H&E's pink and purple exactly where the difference is largest. The
   conversion here is upstream's line (`superpoint_pytorch.py:104-106`), used
   for the teacher whatever the student's channel count turns out to be.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from ConfigIdentity import IdentifiedBuild, IdentifiedConfig, register

from SuperPoint.Decoders import depth_to_space_prob

#: Where the upstream checkout is. A sibling of this repo by default, the same
#: relation `_paths.OUTPUT_ROOT` uses -- and overridable, because a checkout is
#: not a fact about this project. Resolved lazily (inside `build`) so that
#: importing this module costs nothing and fails nowhere.
UPSTREAM_ENV = 'SUPERPOINT_ROOT'

#: The released weights, relative to that root. `pretrained_models/sp_v6.tgz` is
#: the TensorFlow original; this is the converted state dict, and the notebook
#: that produced it (`convert_to_pytorch.ipynb:74-100`) asserts every key and
#: every shape against the PyTorch model before saving, which is why loading it
#: with strict=True below is a real check and not a formality.
WEIGHTS_REL = os.path.join('weights', 'superpoint_v6_from_tf.pth')

#: The zero point. Editing this invalidates every label id ever written, on
#: purpose; editing a dataclass DEFAULT does not -- it splits new from old.
#: ConfigIdentity rule 1.
_TEACHER_BASELINE = {
    'method': 'superpoint-v6',
    'channels': (64, 64, 128, 128, 256),
    'descriptor_dim': 256,
    'grayscale': 'luma',
    'fp16': False,
}


@register('superpoint-v6')
@dataclass(frozen=True)
class TeacherConfig(IdentifiedConfig):
    """What to load, and what makes one teacher not another.

    Deliberately SMALL. `nms_radius`, `detection_threshold` and `remove_borders`
    are upstream config fields and are absent here, because this class only ever
    produces a dense map -- they belong to whoever turns a map into points, and
    a field that nothing reads is a field that will be set and have no effect.
    """
    method: str = 'superpoint-v6'

    #: The backbone widths. `channels[-1]` is the HEAD's hidden width, not the
    #: backbone's output: `channels = [1, *conf.channels[:-1]]` makes the
    #: backbone four stages ending at 128, and each head lifts 128 to 256
    #: (`superpoint_pytorch.py:83-100`, spec.md 9). Reading the list the obvious
    #: way builds a model one stage deeper and twice as wide -- which loads
    #: nothing and raises, but the same misreading in the student would train.
    channels: Tuple[int, ...] = (64, 64, 128, 128, 256)
    descriptor_dim: int = 256

    #: How RGB becomes the 1 channel the weights want. 'luma' is upstream's
    #: 0.299/0.587/0.114; the name is a field rather than a hardcoded line
    #: because it IS an identity-bearing choice on stained tissue, and a config
    #: that cannot express the alternative cannot record which one was used.
    grayscale: str = 'luma'

    fp16: bool = False

    #: Provenance and throughput, not identity.
    weights: Optional[str] = None      # None = the sibling checkout's
    root: Optional[str] = None         # None = $SUPERPOINT_ROOT or the sibling
    batch: int = 16

    NOT_IDENTITY = ('weights', 'root', 'batch')

    def build(self, device) -> 'SuperPointTeacher':
        return SuperPointTeacher(self, device)


class SuperPointTeacher(IdentifiedBuild):
    """Upstream's network, exposed as the two dense maps this project needs."""

    BASELINE = _TEACHER_BASELINE

    @staticmethod
    def weights_state_dict(cfg: Optional['TeacherConfig'] = None
                           ) -> Tuple[dict, str]:
        """The released state dict, WITHOUT building or moving a model.

        `train_superpathpoint --pretrained` needs the tensors and nothing else:
        the student is a different object with different attribute names, and
        instantiating the upstream module only to throw it away would make the
        loader depend on the upstream import for no reason.
        """
        cfg = cfg or TeacherConfig()
        root = _upstream_root(cfg.root)
        weights = cfg.weights or os.path.join(root, WEIGHTS_REL)
        state = torch.load(weights, map_location='cpu')
        state = state.get('state_dict', state) if isinstance(state, dict) else state
        return state, weights

    def __init__(self, cfg: TeacherConfig, device):
        self.cfg = cfg
        self.device = torch.device(device)

        root = _upstream_root(cfg.root)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from superpoint_pytorch import SuperPoint       # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                f'cannot import superpoint_pytorch from {root}. Set '
                f'${UPSTREAM_ENV} to the upstream checkout, or pass '
                f'TeacherConfig(root=...)') from e

        model = SuperPoint(channels=list(cfg.channels),
                           descriptor_dim=cfg.descriptor_dim)
        weights = cfg.weights or os.path.join(root, WEIGHTS_REL)
        state = torch.load(weights, map_location='cpu')
        state = state.get('state_dict', state) if isinstance(state, dict) else state
        # strict, and never a partial load. A teacher with a randomly
        # initialised head produces a plausible map of noise, HA averages a
        # hundred views of that noise into something smooth, and the result
        # reads as a difficult slide rather than as an unloaded checkpoint --
        # which is the failure `test_EoMT`'s own docstring records for EoMT.
        model.load_state_dict(state, strict=True)

        self.model = model.eval().to(self.device)
        self.stride = int(model.stride)
        self.weights_path = weights

    # ── the dense maps ──

    @torch.no_grad()
    def dense_prob(self, images) -> torch.Tensor:
        """[B, H, W] float32 keypoint probability. No NMS, no border, no cut.

        The HA entry point. Upstream's equivalent is `net(image)['prob']`
        (`models/utils.py:21-26`), and the three things it does NOT do are the
        three things `SuperPoint.forward` does after it.
        """
        logits = self._logits(self._prepare(images))
        return depth_to_space_prob(logits, self.stride)

    @torch.no_grad()
    def dense_descriptors(self, images) -> torch.Tensor:
        """[B, 256, H/8, W/8], L2-normalised along the channel axis.

        At CELL resolution, not pixel: upstream samples this at the exact
        keypoint positions with `grid_sample` (`superpoint_pytorch.py:11-22`)
        rather than upsampling the map, and that is the version spec.md 9
        adopts. Returning it dense keeps the sampling with the caller who knows
        where the points are.
        """
        images = self._prepare(images)
        with torch.autocast('cuda', torch.float16,
                            enabled=self.cfg.fp16 and self.device.type == 'cuda'):
            features = self.model.backbone(images)
            dense = self.model.descriptor(features)
        return torch.nn.functional.normalize(dense.float(), p=2, dim=1)

    def _logits(self, images: torch.Tensor) -> torch.Tensor:
        with torch.autocast('cuda', torch.float16,
                            enabled=self.cfg.fp16 and self.device.type == 'cuda'):
            features = self.model.backbone(images)
            logits = self.model.detector(features)
        return logits.float()

    # ── input ──

    def _prepare(self, images) -> torch.Tensor:
        """Anything reasonable in, `[B, 1, H, W]` float in [0, 1] out.

        Accepts a single HxWx3 uint8 array (what `PreTileStore.read_tile`
        returns), a list of them, or a tensor already in NCHW. The conversion is
        upstream's luma, and the /255 is upstream's scale -- both are stated in
        the module docstring because both are silent when wrong.
        """
        tensor = _as_nchw(images).to(self.device)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        else:
            tensor = tensor.float()
            if float(tensor.max()) > 1.5:
                raise ValueError(
                    'float input outside [0, 1]: the weights were trained on '
                    'image/255, and feeding 0-255 floats gives a saturated map '
                    'that still looks like a detection')

        if tensor.shape[1] == 3:
            tensor = self._to_gray(tensor)
        elif tensor.shape[1] != 1:
            raise ValueError(f'expected 1 or 3 channels, got {tensor.shape[1]}')

        h, w = tensor.shape[-2:]
        if h % self.stride or w % self.stride:
            raise ValueError(
                f'{h}x{w} is not a multiple of the stride {self.stride}. The '
                f'decode reshapes by exactly that factor, so a remainder would '
                f'silently drop the last partial cell row and column')
        return tensor

    def _to_gray(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.cfg.grayscale == 'luma':
            scale = tensor.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            return (tensor * scale).sum(1, keepdim=True)
        if self.cfg.grayscale == 'mean':
            return tensor.mean(1, keepdim=True)
        raise ValueError(
            f'unknown grayscale {self.cfg.grayscale!r}; known: luma, mean')

    def summary(self) -> str:
        return (f'superpoint-v6  stride {self.stride}  '
                f'{self.identity_id()}  weights {self.weights_path}')


# ── helpers ───────────────────────────────────────────────────────────────────

def _upstream_root(explicit: Optional[str]) -> str:
    """The upstream checkout, and an error that says what to do when it is not
    there -- an ImportError on `superpoint_pytorch` names the module, not the
    directory it was looked for in."""
    if explicit:
        root = explicit
    elif os.environ.get(UPSTREAM_ENV):
        root = os.environ[UPSTREAM_ENV]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.abspath(os.path.join(here, '..', '..', '..'))
        root = os.path.join(os.path.dirname(repo), 'SuperPoint')
    if not os.path.isfile(os.path.join(root, 'superpoint_pytorch.py')):
        raise FileNotFoundError(
            f'{root} does not hold superpoint_pytorch.py. Point '
            f'${UPSTREAM_ENV} at the upstream checkout')
    return os.path.abspath(root)


def _as_nchw(images) -> torch.Tensor:
    """One place that decides what a batch of images is.

    HxWx3 and HxW are single images; BxHxWx3 is a batch; anything already
    channel-first is left alone. The rule is that the CHANNEL axis is the one of
    size 1 or 3, which is unambiguous for every shape this project produces --
    a 3-pixel-wide tile is not a thing.
    """
    if isinstance(images, (list, tuple)):
        return torch.stack([_as_nchw(im)[0] for im in images])
    tensor = images if torch.is_tensor(images) else torch.from_numpy(
        np.ascontiguousarray(images))
    if tensor.ndim == 2:                                   # H W
        return tensor[None, None]
    if tensor.ndim == 3:
        if tensor.shape[-1] in (1, 3):                     # H W C
            return tensor.permute(2, 0, 1)[None]
        return tensor[None]                                # C H W
    if tensor.ndim == 4:
        if tensor.shape[-1] in (1, 3) and tensor.shape[1] not in (1, 3):
            return tensor.permute(0, 3, 1, 2)              # B H W C
        return tensor                                      # B C H W
    raise ValueError(f'cannot read a batch of images from shape {tuple(tensor.shape)}')
