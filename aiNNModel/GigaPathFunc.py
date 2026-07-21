import os
from contextlib import nullcontext
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_DOTENV = Path(__file__).resolve().parent.parent / '.env'


def gigapath_model(device: torch.device, multi_gpu: bool = False) -> torch.nn.Module:
    os.environ.setdefault('HF_HOME', '/work/u26130998/prov-gigapath/model_weights')
    if not os.environ.get('HF_TOKEN'):
        if _DOTENV.exists():
            from dotenv import load_dotenv
            load_dotenv(_DOTENV)
    model = timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=True)
    model = model.to(device).eval()
    if multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def gigapath_compile(
    model: torch.nn.Module,
    mode: str = 'reduce-overhead',
) -> torch.nn.Module:
    return torch.compile(model, mode=mode)


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
    batch_size: int = 128,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    '''
    Encode a list of PIL Images or numpy arrays.

    Returns [N, D] L2-normalized features (fp32), or [D] for a single input.
    dtype controls autocast precision during forward; output is always fp32.
    '''
    transform = build_transform()
    single_input = not isinstance(images, (list, tuple))
    if single_input:
        images = [images]

    ctx = (torch.autocast(device_type=device.type, dtype=dtype)
           if dtype != torch.float32 else nullcontext())

    outputs = []
    for start in range(0, len(images), batch_size):
        batch_imgs = images[start:start + batch_size]
        batch = torch.stack([
            transform(img if isinstance(img, Image.Image) else Image.fromarray(img))
            for img in batch_imgs
        ]).to(device)
        with ctx:
            feats = model(batch)
        feats = F.normalize(feats.float(), dim=-1)
        outputs.append(feats.cpu())

    stacked = torch.cat(outputs, dim=0)
    return stacked[0] if single_input else stacked


def gigapath_apply_tome(model: torch.nn.Module, r: int = 8) -> torch.nn.Module:
    '''
    Apply Token Merging (ToMe) to the model in-place.

    r: tokens merged per layer. r=8 ≈ 30% speedup with minimal accuracy loss.
    Must be called before gigapath_compile() if both are used.

    Install: pip install git+https://github.com/facebookresearch/ToMe.git
             pip install "timm>=1.0.3"   # must come after tome
    '''
    import tome
    tome.patch.timm(model)
    model.r = r
    return model


def make_gigapath_encoder(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 128,
    dtype: torch.dtype = torch.float32,
):
    '''
    Return an encoder(images) callable for use wherever a patch encoder is expected.

    Mirrors make_hest_method: callers do not need to write a lambda or pass
    model/device/batch_size at each call site.

    Example:
        model   = gigapath_model(device)
        model   = gigapath_compile(model)          # optional
        encoder = make_gigapath_encoder(model, device, batch_size=128,
                                        dtype=torch.float16)
        feats   = encoder(patches)
    '''
    def _encoder(images):
        return gigapath_encode(images, model, device,
                               batch_size=batch_size, dtype=dtype)
    return _encoder
