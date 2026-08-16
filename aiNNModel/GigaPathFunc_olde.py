'''SUPERSEDED. Kept only so the refactor of 2026-08-16 can be diffed against
what it replaced; nothing imports this and nothing should.

This is aiNNModel/GigaPathFunc.py as it stood at commit b5530ec -- free
functions taking (model, device, batch_size, dtype) at every call site, a
module-level _TRANSFORM built at import, and Token Merging applied by mutating
a model after construction. The replacement is EncoderConfig / TransformConfig /
GigaPathEncoder in GigaPathFunc.py; the equivalence between the two is measured
in test_gigapath_pooling.py, not assumed.

Delete once nobody needs to read the old shape.
'''
import os
import functools
from contextlib import nullcontext
from pathlib import Path

# MUST stay above `import timm`. huggingface_hub freezes its cache location into
# module-level constants the moment it is imported
# (huggingface_hub/constants.py: HF_HOME and HF_HUB_CACHE are os.getenv calls at
# import time, not at download time), and timm imports huggingface_hub. Setting
# HF_HOME after that point is read by nobody: the 4.5 GB weights are silently
# re-downloaded to ~/.cache/huggingface even though a complete copy already sits
# in the path below. The only symptom is a progress bar.
os.environ.setdefault('HF_HOME', '/work/u26130998/prov-gigapath/model_weights')

import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_DOTENV = Path(__file__).resolve().parent.parent / '.env'


# ── Model ────────────────────────────────────────────────────────────────────

def gigapath_model(device: torch.device, multi_gpu: bool = False) -> torch.nn.Module:
    # HF_HOME is set at the top of this module, not here: by the time this runs,
    # huggingface_hub has long since read it. See the note above the imports.
    if not os.environ.get('HF_TOKEN') and _DOTENV.exists():
        from dotenv import load_dotenv
        load_dotenv(_DOTENV)
    model = timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=True)
    model = model.to(device).eval()
    if multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def gigapath_apply_tome(model: torch.nn.Module, r: int = 8) -> torch.nn.Module:
    '''
    Apply Token Merging (ToMe) to the model in-place.

    r: tokens merged per layer. r=8 ~ 30% speedup with minimal accuracy loss.
    Must be called before gigapath_compile() if both are used.

    Install: pip install git+https://github.com/facebookresearch/ToMe.git
             pip install "timm>=1.0.3"   # must come after tome

    LayerScale fix
    --------------
    Upstream tome (0.1.x) ToMeBlock.forward omits self.ls1 / self.ls2.
    Modern timm ViT-g / DINOv2-style models (GigaPath included) rely on
    LayerScale with gamma ~ 1e-4 to keep residuals stable; skipping it
    scales the attention branch ~1e4x too large and the forward collapses
    to a near-random embedding within a few blocks. We patch the class
    forward here so any ToMeBlock instance created by tome.patch.timm uses
    the LayerScale-aware version.
    '''
    import tome
    from tome.patch.timm import ToMeBlock
    from tome.merge import bipartite_soft_matching, merge_source, merge_wavg

    tome.patch.timm(model)

    def _forward_with_ls(self, x):
        attn_size = self._tome_info['size'] if self._tome_info['prop_attn'] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(self.ls1(x_attn))

        r_ = self._tome_info['r'].pop(0)
        if r_ > 0:
            merge, _ = bipartite_soft_matching(
                metric, r_,
                self._tome_info['class_token'],
                self._tome_info['distill_token'],
            )
            if self._tome_info['trace_source']:
                self._tome_info['source'] = merge_source(
                    merge, x, self._tome_info['source']
                )
            x, self._tome_info['size'] = merge_wavg(
                merge, x, self._tome_info['size']
            )

        x = x + self._drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

    ToMeBlock.forward = _forward_with_ls

    model.r = r
    return model


def gigapath_compile(
    model: torch.nn.Module,
    mode: str = 'reduce-overhead',
) -> torch.nn.Module:
    return torch.compile(model, mode=mode)


# ── Encode ───────────────────────────────────────────────────────────────────

def build_transform(preprocess: str = 'none') -> transforms.Compose:
    '''The tile -> tensor pipeline. `preprocess` is inserted before ToTensor.

        'none'  RGB exactly as given. What production encodes, and the default,
                so nothing moves unless a caller asks.
        'grey'  luminance only, replicated back to three channels because the
                model's patch embedding takes three. Colour is removed from the
                problem entirely -- both sides of a comparison have to use it or
                the two are not describing the same thing.

    Note what Normalize does NOT do: its mean and std are fixed ImageNet
    constants, the same numbers for every image, so it is a global affine map
    and the colour DIFFERENCE between two tiles survives it untouched. It puts
    the input where the model expects it; it does not align two images to each
    other. Anything that wants to remove a colour shift between a photograph and
    a slide has to happen here, above this line.
    '''
    steps = [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ]
    if preprocess == 'grey':
        steps.append(transforms.Grayscale(num_output_channels=3))
    elif preprocess != 'none':
        raise ValueError(f"preprocess must be 'none' or 'grey', got {preprocess!r}")
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(steps)


_TRANSFORM = build_transform()


def _to_pil(img) -> Image.Image:
    return img if isinstance(img, Image.Image) else Image.fromarray(img)


@torch.no_grad()
def gigapath_encode(
    images,
    model: torch.nn.Module,
    device: torch.device,
    transform=None,
    batch_size: int = 128,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    '''
    Encode a list of PIL Images or numpy arrays.

    Returns [N, D] L2-normalized features (fp32), or [D] for a single input.
    dtype controls autocast precision during forward; output is always fp32.
    '''
    single_input = not isinstance(images, (list, tuple))
    if single_input:
        images = [images]

    ctx = (torch.autocast(device_type=device.type, dtype=dtype)
           if dtype != torch.float32 else nullcontext())

    tf = transform if transform is not None else _TRANSFORM

    outputs = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack(
            [tf(_to_pil(img)) for img in images[start:start + batch_size]]
        ).to(device)
        with ctx:
            feats = model(batch)
        feats = F.normalize(feats.float(), dim=-1)
        outputs.append(feats.cpu())

    stacked = torch.cat(outputs, dim=0)
    return stacked[0] if single_input else stacked


# ── Tokens and pooling ───────────────────────────────────────────────────────
#
# gigapath_encode keeps one vector per tile: timm pools with global_pool='token',
# which is `x[:, 0]` (timm/layers/pool1d.py:global_pool_nlc), so 196 of the 197
# tokens are discarded. These three let a caller keep them and try other
# reductions offline, without re-encoding and without editing the model.

def model_token_spec(model: torch.nn.Module) -> dict:
    '''The three numbers any pooling needs, read off the model, never assumed.

    Unwraps DataParallel first. nn.DataParallel defines no __getattr__ of its
    own, so nn.Module's is used, and that searches only _parameters / _buffers /
    _modules. `embed_dim` and `num_prefix_tokens` are plain ints on the wrapped
    module, and `patch_embed` is one of ITS submodules -- so all three raise
    AttributeError through the wrapper, and gigapath_model(multi_gpu=True) is the
    default in the benches.

    No defaults, deliberately. `getattr(m, 'num_prefix_tokens', 1)` is right for
    GigaPath and wrong for the DINOv2 _reg4 variants, where four register tokens
    would be averaged in as if they were patches -- silently, showing up only as
    a slightly worse score. Crashing is the better failure.
    '''
    m = getattr(model, 'module', model)
    if not isinstance(m.head, torch.nn.Identity):
        raise ValueError(
            f'pooling needs a feature extractor, but model.head is '
            f'{type(m.head).__name__}. Build the model with num_classes=0.')
    return {'dim':        int(m.embed_dim),
            'token_grid': tuple(int(v) for v in m.patch_embed.grid_size),
            'num_prefix': int(m.num_prefix_tokens)}


@torch.no_grad()
def gigapath_encode_tokens(
    images,
    model: torch.nn.Module,
    device: torch.device,
    transform=None,
    batch_size: int = 128,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    '''Every token, not just the pooled one. Returns [N, T, D] fp32.

    Deliberately NOT L2-normalized: pooling comes first and each slot is
    normalized afterwards, because mean-then-normalize and normalize-then-mean
    are different vectors and only the first is what a pooled descriptor means.

    fc_norm is applied here even though forward_features looks finished, and that
    is the subtlety worth reading twice. timm builds `norm` and `fc_norm` as a
    pair in which exactly one is real and the other is Identity:

        vision_transformer.py:800  use_fc_norm = global_pool in ('avg','avgmax','max')
                             :879  self.norm    = LayerNorm if not use_fc_norm else Identity
                             :902  self.fc_norm = LayerNorm if     use_fc_norm else Identity

    At global_pool='token' -- GigaPath today -- `norm` is the real one, already
    applied inside forward_features, and fc_norm is a no-op. At global_pool='avg'
    (one of the pooling variants this experiment exists to try) the two swap:
    forward_features returns un-normed tokens and skipping fc_norm would change
    the scale of every token without raising anything.

    Applying it unconditionally is correct in both cases. And because LayerNorm
    acts per token along D, fc_norm(tokens)[:, 0] == fc_norm(tokens[:, 0])
    exactly, so slot 0 IS the production feature rather than an approximation of
    it -- test_gigapath_pooling.py holds that equality down.

    Note this calls the unwrapped module, so it runs on one GPU even under
    DataParallel. That is the intended trade: the pooling dump is ~68k tiles,
    which is minutes on a single card.
    '''
    m = getattr(model, 'module', model)

    single_input = not isinstance(images, (list, tuple))
    if single_input:
        images = [images]

    ctx = (torch.autocast(device_type=device.type, dtype=dtype)
           if dtype != torch.float32 else nullcontext())

    tf = transform if transform is not None else _TRANSFORM

    outputs = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack(
            [tf(_to_pil(img)) for img in images[start:start + batch_size]]
        ).to(device)
        with ctx:
            toks = m.fc_norm(m.forward_features(batch))
        outputs.append(toks.float().cpu())

    stacked = torch.cat(outputs, dim=0)
    return stacked[0] if single_input else stacked


def _ring_bins(gh: int, gw: int, n_rings: int) -> torch.Tensor:
    '''Ring id per patch cell, split so every ring holds the same COUNT.

    Equal count rather than equal radius: with equal radii the outer ring covers
    most of the grid and the inner one a handful of cells, so the slots would
    average over wildly different support and their norms would not be
    comparable. Rotation-invariance is the point of rings, and equal-count keeps
    it while making the slots comparable to each other.
    '''
    yy, xx = torch.meshgrid(torch.arange(gh, dtype=torch.float32),
                            torch.arange(gw, dtype=torch.float32),
                            indexing='ij')
    r = torch.hypot(yy - (gh - 1) / 2.0, xx - (gw - 1) / 2.0).flatten()
    order = torch.argsort(r)
    bins = torch.empty(gh * gw, dtype=torch.long)
    n = gh * gw
    for k in range(n_rings):
        lo, hi = round(n * k / n_rings), round(n * (k + 1) / n_rings)
        bins[order[lo:hi]] = k
    return bins


def pool_tokens(tokens: torch.Tensor, mode: str, spec: dict):
    '''Reduce [N, T, D] tokens to [N, n, D] slots.

    Returns (features, slots, slot_layout) where `slots` names each of the n
    entries and `slot_layout` says how they permute under a 90-degree rotation.
    Both go into the store's metadata: without slot_layout, a reader matching
    rotated queries against a grid pooling would compare slot (0,1) with slot
    (1,0) and see only a lower score.

    Every slot is L2-normalized on its own rather than the flattened [n*D]
    vector. Flattening would fix the weight between slots at write time, and how
    to weight them is exactly the open question the experiment is for.
    '''
    p = int(spec['num_prefix'])
    gh, gw = (int(v) for v in spec['token_grid'])
    n_patch = gh * gw

    if tokens.ndim != 3:
        raise ValueError(f'tokens must be [N, T, D], got {tuple(tokens.shape)}')
    if tokens.shape[1] != p + n_patch:
        raise ValueError(
            f'spec says {p} prefix + {gh}x{gw} patches = {p + n_patch} tokens, '
            f'but tokens have T={tokens.shape[1]}. spec and model disagree.')

    cls     = tokens[:, 0]                       # registers, if any, are dropped
    patches = tokens[:, p:]                      # [N, gh*gw, D]

    if mode == 'cls':
        parts, slots, layout = [cls], ('cls',), 'none'

    elif mode == 'cls_avg':
        parts, slots, layout = [cls, patches.mean(1)], ('cls', 'avg'), 'none'

    elif mode == 'cls_std':
        # Heterogeneity inside the tile: uniform stroma and a tissue boundary
        # have similar means and very different spreads. Rotation-invariant,
        # since it is a statistic over the token set and not over its layout.
        parts, slots, layout = [cls, patches.std(1)], ('cls', 'std'), 'none'

    elif mode.startswith('rings'):
        n_rings = int(mode[5:] or 3)
        bins = _ring_bins(gh, gw, n_rings)
        parts = [cls] + [patches[:, bins == k].mean(1) for k in range(n_rings)]
        slots = ('cls',) + tuple(f'r{k}' for k in range(n_rings))
        layout = f'ring:{n_rings}'

    elif mode.startswith('grid'):
        bh, bw = (int(v) for v in mode[4:].split('x'))
        if gh % bh or gw % bw:
            raise ValueError(
                f'{gh}x{gw} patch grid does not divide into {bh}x{bw} blocks')
        g = patches.reshape(patches.shape[0], gh, gw, -1)
        parts = [cls]
        slots = ['cls']
        for i in range(bh):
            for j in range(bw):
                blk = g[:, i * (gh // bh):(i + 1) * (gh // bh),
                        j * (gw // bw):(j + 1) * (gw // bw), :]
                parts.append(blk.mean(dim=(1, 2)))
                slots.append(f'g{i}{j}')
        slots, layout = tuple(slots), f'grid:{bh}x{bw}'

    elif mode == 'tokens':
        parts = [cls] + [patches[:, k] for k in range(n_patch)]
        slots = ('cls',) + tuple(f'p{k:03d}' for k in range(n_patch))
        layout = f'grid:{gh}x{gw}'

    else:
        raise ValueError(f'unknown pooling mode {mode!r}')

    feats = torch.stack(parts, dim=1)            # [N, n, D]
    return F.normalize(feats, dim=-1), slots, layout


# ── Factory ──────────────────────────────────────────────────────────────────

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
    return functools.partial(gigapath_encode,
                             model=model, device=device,
                             batch_size=batch_size, dtype=dtype)
