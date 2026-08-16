"""HEST tissue segmentation (MahmoodLab/hest-tissue-seg) as a TissueSegmenter.

    seg = HestSegConfig().build(device)
    binary = seg(rgb)                    # [H, W] uint8, 1 = tissue

The contract and the model-free methods are TissueSegFunc's. What is here is
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
for _d in (_HERE, _HERE.parent / 'utilities'):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from PIL import Image                                       # noqa: E402
from torchvision import transforms                          # noqa: E402

from ConfigIdentity import ModelConfig, register            # noqa: E402
from TissueSegFunc import TissueSegConfig, TissueSegmenter  # noqa: E402


_CKPT_DIR = _HERE / 'ckpt'
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


#: The zero point. Editing this invalidates every mask id ever written, on
#: purpose; editing a dataclass DEFAULT does not -- it splits new from old.
_HEST_BASELINE = {
    'method': 'hest',
    'model': ModelConfig(source='torchvision', arch=HEST_ARCH, dtype='fp32'),
}


def _download_ckpt() -> Path:
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE,
                    local_dir=str(_CKPT_DIR))
    return _CKPT_DIR / _HF_FILE


@register('hest')
@dataclass(frozen=True)
class HestSegConfig(TissueSegConfig):
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

    def build(self, device: Optional[torch.device] = None) -> 'HestSegmenter':
        return HestSegmenter(self, device or torch.device('cpu'))


class HestSegmenter(TissueSegmenter):
    BASELINE = _HEST_BASELINE

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

        self.model = model.to(device).eval()

    @property
    def runs(self) -> bool:
        return True

    @torch.no_grad()
    def __call__(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """One RGB image -> [H, W] uint8, 1 = tissue.

        Called once per tile by TissuesRegionsMask's tile-and-stitch pass. The
        model is fully convolutional, so the tile size is the caller's.
        """
        pil = Image.fromarray(image) if isinstance(image, np.ndarray) \
            else image.convert('RGB')
        tensor = _TRANSFORM(pil).unsqueeze(0).to(self.device)   # [1, 3, H, W]
        out = self.model(tensor)['out']                         # [1, 2, H, W]
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
