from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


_CKPT_PATH = Path(__file__).resolve().parent / 'ckpt' / 'deeplabv3_seg_v4.ckpt'
_HF_REPO   = 'MahmoodLab/hest-tissue-seg'
_HF_FILE   = 'deeplabv3_seg_v4.ckpt'

# class 0 = background, class 1 = tissue
_TISSUE_CLASS = 1


def _download_ckpt() -> Path:
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id=_HF_REPO,
        filename=_HF_FILE,
        local_dir=str(_CKPT_PATH.parent),
    )
    return _CKPT_PATH


def hest_seg_model(device: torch.device) -> torch.nn.Module:
    '''
    Load HEST tissue segmentation model.

    Architecture: DeepLabV3 + ResNet-50 backbone, num_classes=2 (bg / tissue).
    Checkpoint: MahmoodLab/hest-tissue-seg  deeplabv3_seg_v4.ckpt
    Stored at:  aiNNModel/ckpt/deeplabv3_seg_v4.ckpt

    The checkpoint was saved with a Lightning wrapper (keys prefixed with "model.").
    The aux_classifier head has 21 VOC classes (pretrain artifact) and is skipped;
    it is not used at inference time.
    '''
    ckpt_path = _CKPT_PATH
    if not ckpt_path.exists():
        print('Downloading HEST seg checkpoint...')
        ckpt_path = _download_ckpt()

    model = models.segmentation.deeplabv3_resnet50(
        weights=None, weights_backbone=None, num_classes=2,
    )

    raw = torch.load(ckpt_path, map_location='cpu')
    sd  = raw.get('state_dict', raw)
    # strip 'model.' prefix; skip aux_classifier (VOC 21-class pretrain artifact)
    new_sd = {
        k[len('model.'):]: v
        for k, v in sd.items()
        if k.startswith('model.') and not k.startswith('model.aux_classifier')
    }
    model.load_state_dict(new_sd, strict=False)

    return model.to(device).eval()


_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std =(0.229, 0.224, 0.225)),
])


def make_hest_method(
    model: torch.nn.Module,
    device: torch.device,
) -> callable:
    '''
    Return a method(img) callable for use with TissuesRegionsMask.from_wsi.

    Example:
        trm = TissuesRegionsMask.from_wsi(
            wsi, ds=8.0, method=make_hest_method(model, device)
        )
    '''
    def _method(img: np.ndarray) -> np.ndarray:
        return hest_seg_predict(img, model, device)
    return _method


@torch.no_grad()
def hest_seg_predict(
    image: Union[np.ndarray, Image.Image],
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    '''
    Run tissue segmentation on a single RGB image.

    Returns a binary uint8 mask (1 = tissue, 0 = background),
    same spatial size as the input image.
    The model is fully convolutional and accepts any input size.
    '''
    if isinstance(image, np.ndarray):
        pil = Image.fromarray(image)
    else:
        pil = image.convert('RGB')

    tensor = _TRANSFORM(pil).unsqueeze(0).to(device)   # [1, 3, H, W]
    out    = model(tensor)['out']                        # [1, 2, H, W]
    mask   = out.argmax(dim=1).squeeze(0).cpu().numpy() # [H, W]  0 or 1
    return (mask == _TISSUE_CLASS).astype(np.uint8)
