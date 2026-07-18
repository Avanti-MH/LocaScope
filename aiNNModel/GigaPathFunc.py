import os

import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def gigapath_model(device: torch.device):
    os.environ.setdefault('HF_HOME', '/work/u26130998/prov-gigapath/model_weights')
    os.environ.setdefault('HF_TOKEN', '')
    model = timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=True)
    return model.to(device).eval()


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def gigapath_encode(
    images,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    '''
    Encode a list of PIL Images or numpy arrays.

    Returns [N, D] L2-normalized features, or [D] for a single input.
    '''
    transform = build_transform()
    single_input = not isinstance(images, (list, tuple))
    if single_input:
        images = [images]

    outputs = []
    for start in range(0, len(images), batch_size):
        batch_imgs = images[start:start + batch_size]
        batch = torch.stack([
            transform(img if isinstance(img, Image.Image) else Image.fromarray(img))
            for img in batch_imgs
        ]).to(device)
        feats = model(batch)
        feats = F.normalize(feats, dim=-1)
        outputs.append(feats.cpu())

    stacked = torch.cat(outputs, dim=0)
    return stacked[0] if single_input else stacked
