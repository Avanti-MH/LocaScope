"""HEST tissue segmentation (MahmoodLab/hest-tissue-seg) as a plane segmenter.

    seg  = MASK_RECIPES['hest'].seg.build(device)   # TissueMaskConfig's recipes
    mask = seg.segment_slide(wsi)        # SlideMask, via PlaneSegmenter's read
    binary = seg(rgb)                    # one image: [H, W] uint8, 1 = tissue

The contract, the plane read and the model-free methods are TissueSeg's. What is here is
what is HEST's: the frozen baseline its numbers form, the DeepLabV3 the
checkpoint fits, the Lightning prefix that checkpoint was saved with, and the
class index that means tissue.

Fully convolutional, so any input size is accepted and tiling is sound -- unlike
Otsu, whose threshold depends on the histogram of whatever it is shown.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

_HERE = Path(__file__).resolve().parent

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from PIL import Image                                       # noqa: E402
from torchvision import transforms                          # noqa: E402

from locascope.core.config.ConfigIdentity import ModelConfig, file_fingerprint, register  # noqa: E402
from locascope.core.backbones.TissueSeg import PlaneSegConfig, PlaneSegmenter    # noqa: E402


from locascope.paths import MODEL_WEIGHTS_DIR                    # noqa: E402

#: The released weights, next to the other model weights, not inside the package: a checkout
#: (or a second one, on another OS) then finds the same file.
_CKPT_DIR = Path(MODEL_WEIGHTS_DIR) / 'hest_tissue_seg'
_HF_REPO  = 'MahmoodLab/hest-tissue-seg'
_HF_FILE  = 'deeplabv3_seg_v4.ckpt'

#: class 0 = background, class 1 = tissue
_TISSUE_CLASS = 1

HEST_ARCH = 'segmentation.deeplabv3_resnet50'

#: Fixed rather than configurable: nothing here varies it, and the ImageNet
#: constants are the ones the checkpoint was trained under. It becomes a config
#: the day a second preprocessing is needed.
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])


def _download_ckpt() -> Path:
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE,
                    local_dir=str(_CKPT_DIR))
    return _CKPT_DIR / _HF_FILE


@register('hest')
@dataclass(frozen=True)
class HestSegConfig(PlaneSegConfig):
    """DeepLabV3 + ResNet-50, two classes.

    `model.arch` names the ARCHITECTURE and not the checkpoint, so a finetune of
    the same network keeps it honest: what changed is the parameters, and
    weights_id hashes those. `model.weights` points at a local checkpoint or is
    None for the published one; it is in ModelConfig's NOT_IDENTITY because the
    content hash already covers what the file holds, and a checkpoint that moved
    between mounts should not invalidate a cache.
    """
    method: str = 'hest'
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(source='torchvision',
                                            arch=HEST_ARCH, dtype='fp32'))

    BASELINE = {'method': 'hest', 'limit_bounds': True, 'ds': 4.0,
                'seg_chunk_px': 4_000_000, 'read_chunk_px': 4_000_000,
                'stitch_overlap': 128, 'model': 'ModelConfig'}

    def build(self, device: Optional[torch.device] = None) -> 'HestSegmenter':
        return HestSegmenter(self, device or torch.device('cpu'))

    def weights_key(self) -> str:
        """The CONTENT of a local checkpoint, for a key that cannot build the
        model to hash its parameters. `model.weights` is NOT_IDENTITY because
        the built segmenter hashes what it loaded -- but a cache hit is exactly
        the case where nothing is built, so without this a finetune overwritten
        in place under the same path would keep hitting the old masks. The
        published checkpoint is fixed by the repo and file name, so it adds
        nothing."""
        return file_fingerprint(self.model.weights) if self.model.weights else ''


class HestSegmenter(PlaneSegmenter):
    def __init__(self, cfg: HestSegConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._weights_id = None

        # num_classes=2 is what makes this a tissue segmenter, the way
        # num_classes=0 is what makes a TileEncoder an encoder. Both are
        # constants of the domain rather than settings a caller varies, so they
        # are passed here and not hashed.
        model = cfg.model.build(weights=None, weights_backbone=None,
                                num_classes=2)

        if cfg.model.weights is None:
            ckpt = _CKPT_DIR / _HF_FILE
            if not ckpt.exists():
                print('Downloading HEST seg checkpoint...')
                ckpt = _download_ckpt()
            _load_published(model, ckpt)

        # channels_last: the layout cuDNN's convolutions run fastest in on a tensor-core GPU. Speed only.
        self.model = model.to(device).eval().to(memory_format=torch.channels_last)
        # Every interior tile of a tiled plane is the same shape, so the convolution
        # algorithm is searched once and reused. Speed only: no field of the config.
        if torch.device(device).type == 'cuda':
            torch.backends.cudnn.benchmark = True
        self._mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
        self._std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)

    #: tiles of one shape sent through the network together (TissueSeg.tiled_apply)
    #: and threads reading the next tiles meanwhile. Speed only, not identity: a
    #: tile sees the same pixels either way. Set on the built segmenter to change.
    batch_tiles: int = 16
    read_workers: int = 4

    def _autocast(self):
        """fp16 when the config says so (`model.dtype`, which IS identity: a
        different seg_id, a mask beside the fp32 one), on a GPU only."""
        import contextlib                                     # noqa: PLC0415
        if self.cfg.model.dtype == 'fp16' and torch.device(self.device).type == 'cuda':
            return torch.autocast('cuda', dtype=torch.float16)
        return contextlib.nullcontext()

    @torch.no_grad()
    def segment_batch(self, images) -> list:
        """RGB uint8 images of ONE shape -> one [H, W] uint8 mask each, 1 = tissue.

        The same normalisation as `_TRANSFORM` (x / 255, ImageNet mean and std),
        done on the GPU for the whole batch rather than per image on the CPU."""
        if not images:
            return []
        stack = torch.from_numpy(np.ascontiguousarray(np.stack(images)))   # [B, H, W, 3] uint8
        x = stack.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
        x = ((x - self._mean) / self._std).contiguous(memory_format=torch.channels_last)
        with self._autocast():
            out = self.model(x)['out']                          # [B, 2, H, W]
        masks = (out.argmax(dim=1) == _TISSUE_CLASS).to(torch.uint8).cpu().numpy()
        return list(masks)

    @property
    def runs(self) -> bool:
        return True

    @torch.no_grad()
    def __call__(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """One RGB image -> [H, W] uint8, 1 = tissue.

        Called once per tile by TissueMask's tile-and-stitch pass. The
        model is fully convolutional, so the tile size is the caller's.
        """
        pil = Image.fromarray(image) if isinstance(image, np.ndarray) \
            else image.convert('RGB')
        tensor = _TRANSFORM(pil).unsqueeze(0).to(self.device)   # [1, 3, H, W]
        with self._autocast():
            out = self.model(tensor)['out']                     # [1, 2, H, W]
        mask = out.argmax(dim=1).squeeze(0).cpu().numpy()       # [H, W] 0 or 1
        return (mask == _TISSUE_CLASS).astype(np.uint8)


def _load_published(model: torch.nn.Module, ckpt_path: Path) -> None:
    """Load the published checkpoint, which is not in torchvision's own shape.

    It was saved through a Lightning wrapper, so every key is prefixed with
    "model.". The aux_classifier head carries 21 VOC classes from pretraining
    and is dropped -- it is not used at inference, and loading it into a
    two-class head would fail on shape.

    strict=False for exactly that reason and no other: the keys that go missing
    are the aux ones just removed. A finetune goes through ModelConfig.weights,
    which loads STRICTLY, so a checkpoint that does not fit the architecture is
    an error there rather than a half-random model here.
    """
    raw = torch.load(ckpt_path, map_location='cpu')
    sd = raw.get('state_dict', raw)
    stripped = {
        k[len('model.'):]: v
        for k, v in sd.items()
        if k.startswith('model.') and not k.startswith('model.aux_classifier')
    }
    model.load_state_dict(stripped, strict=False)
