'''GigaPath (prov-gigapath/prov-gigapath) as a TileEncoder.

    encoder = GigaPathEncoderConfig().build(device)
    feats   = encoder(tiles)                  # [N, 1536]
    toks    = encoder.tokens(tiles)           # [N, 197, 1536]
    slots   = encoder.pooled(tiles, 'grid2x2')

The contract, the batch loop, the identity surface and `variant` are
TileEncoderFunc's. What is here is what is GigaPath's: the frozen baseline its
numbers form, the CLS as the answer to "which token is the feature", the token
poolings, and Token Merging.

_GIGAPATH_BASELINE is the zero point every id is measured against. Its transform
is 256 -> 224, which is crop_pct 0.875 and NOT what the checkpoint declares:
prov-gigapath's config.json says "crop_pct": 1.0. Their own code disagrees with
their own metadata, in four places and consistently --

    prov-gigapath/demo/3_load_tile_encoder.py:15-16
    prov-gigapath/gigapath/pipeline.py:110-111       (the WSI inference path)
    prov-gigapath/myApp/inference_baseline.py:68-69
    prov-gigapath/myApp/WSIdataset.py:37-38

all four Resize(256) then CenterCrop(224), and the demo then asserts its output
against a stored tensor. So 256/224 is the preprocessing those weights were
validated with and crop_pct is an upstream slip. test_gigapath_pooling holds the
baseline down against that same stored tensor, so a changed default fails there
rather than in a number nobody can trace.

flash-attn is auto-detected by timm; no extra code needed.
'''
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple

_HERE = Path(__file__).resolve().parent
for _d in (_HERE, _HERE.parent / 'utilities'):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

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
import torch.nn.functional as F  # noqa: E402

from ConfigIdentity import ModelConfig, register  # noqa: E402
from TileEncoderFunc import (OutputSpec, TileEncoder,  # noqa: E402
                             TileEncoderConfig, TransformConfig)

_DOTENV = _HERE.parent / '.env'

GIGAPATH_ARCH = 'hf_hub:prov-gigapath/prov-gigapath'


# ── Configuration ─────────────────────────────────────────────────────────────

#: The zero point. Editing this invalidates every id ever written, on purpose;
#: editing a dataclass DEFAULT does not -- it splits new from old instead, which
#: is why the two are separate. See ConfigIdentity for the four cases.
_GIGAPATH_BASELINE = {
    'model': ModelConfig(source='timm', arch=GIGAPATH_ARCH, dtype='fp16'),
    'transform': TransformConfig(scale_size=256, crop_size=224,
                                 interpolation='bicubic',
                                 mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225),
                                 preprocess='none'),
    'tome_r': 0,
}


@register('gigapath')
@dataclass(frozen=True)
class GigaPathEncoderConfig(TileEncoderConfig):
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(source='timm', arch=GIGAPATH_ARCH,
                                            dtype='fp16'))

    #: Token Merging. A FIELD and not a function applied afterwards, because it
    #: changes the vectors -- r=8 dropped top-5 retrieval overlap to 0.26
    #: (log/TODO.log) -- and a mutation performed after construction leaves
    #: nothing for an id to record. 0 is off and is the baseline, so adding it
    #: moved no existing hash.
    tome_r: int = 0

    #: Excluded for a different reason from batch_size: torch.compile reorders
    #: reductions, worth about 1e-7, two orders below what fp16 storage in a
    #: FeatureStore already discards. Splitting a cache on it would separate
    #: files nobody could tell apart afterwards.
    compile: bool = False

    NOT_IDENTITY = ('batch_size', 'compile')

    def build(self, device: torch.device, multi_gpu: bool = False):
        return GigaPathEncoder(self, device, multi_gpu=multi_gpu)

# ── Model ────────────────────────────────────────────────────────────────────

def _apply_tome(model: torch.nn.Module, r: int = 8) -> torch.nn.Module:
    '''
    Apply Token Merging (ToMe) to the model in-place.

    r: tokens merged per layer. r=8 ~ 30% speedup with minimal accuracy loss.
    Applied by GigaPathEncoder before torch.compile; see EncoderConfig.tome_r.

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


def _compile(model: torch.nn.Module,
             mode: str = 'reduce-overhead') -> torch.nn.Module:
    '''torch.compile. Private, and reached through EncoderConfig.compile, so the
    "tome first, then compile" ordering is enforced by build() rather than by a
    sentence in a docstring that a caller has to have read.'''
    return torch.compile(model, mode=mode)


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


def _ring_bins(gh: int, gw: int, n_rings: int,
               device: torch.device = None) -> torch.Tensor:
    '''Ring id per patch cell, split so every ring holds the same COUNT.

    Equal count rather than equal radius: with equal radii the outer ring covers
    most of the grid and the inner one a handful of cells, so the slots would
    average over wildly different support and their norms would not be
    comparable. Rotation-invariance is the point of rings, and equal-count keeps
    it while making the slots comparable to each other.

    `device` exists because the result is used as an index into the tokens, and
    a CPU mask cannot index a CUDA tensor. Built on CPU either way -- gh*gw is
    196 and argsort of 196 floats on a GPU is slower than the transfer -- then
    moved once. pool_tokens passes tokens.device, so ring pooling follows the
    tokens wherever they are.
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
    return bins if device is None else bins.to(device)


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
        bins = _ring_bins(gh, gw, n_rings, tokens.device)
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


# ── Encoder ──────────────────────────────────────────────────────────────────

class GigaPathEncoder(TileEncoder):
    '''prov-gigapath, built as a token producer.

    global_pool='' and num_classes=0 are the two halves of "this is an encoder":
    no classifier, and no reduction the caller did not ask for. Stated here and
    not inherited -- prov-gigapath's config.json happens to say num_classes 0,
    so leaving it out worked for exactly one architecture and gave
    `model.head is Linear` on the first other one tried.

    That the empty setting returns exactly what this module used to assemble by
    hand -- fc_norm(forward_features(x)) -- is measured, not assumed:
    test_gigapath_pooling checks it at max|delta| == 0 and asserts the four
    preconditions the equality rests on. Going through model() rather than
    around it is what lets DataParallel work and removes the private attribute
    reach.
    '''

    BASELINE = _GIGAPATH_BASELINE

    def __init__(self, cfg: GigaPathEncoderConfig, device: torch.device,
                 multi_gpu: bool = False):
        self.cfg = cfg
        self.device = device
        if not os.environ.get('HF_TOKEN') and _DOTENV.exists():
            from dotenv import load_dotenv
            load_dotenv(_DOTENV)

        model = cfg.model.build(num_classes=0, global_pool='')
        model = model.to(device).eval()

        # Order, and the reason it is here rather than in a docstring: ToMe
        # replaces block forwards, torch.compile traces them. Compiling first
        # traces the unmerged blocks and the merge never runs -- silently, at
        # full speed, with the wrong answer.
        if cfg.tome_r:
            model = _apply_tome(model, cfg.tome_r)
        if cfg.compile:
            model = _compile(model)
        if multi_gpu and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        self.model = model
        token = model_token_spec(model)
        self.spec = OutputSpec(kind='tokens', dim=token['dim'],
                               grid=token['token_grid'],
                               num_prefix=token['num_prefix'])
        self._transform = cfg.transform.build()
        self._weights_id = None

    def _vector_from(self, raw: torch.Tensor) -> torch.Tensor:
        '''The CLS, which is GigaPath's own answer and not a default.

        timm's pool() takes the slice point from the model's own
        num_prefix_tokens rather than from an assumption about how many prefix
        tokens this architecture has -- 1 here, 5 for the DINOv2 _reg4 variants.
        '''
        m = getattr(self.model, 'module', self.model)
        return F.normalize(m.pool(raw, pool_type='token'), dim=-1)

    def pooled(self, images, mode: str):
        '''(features, slots, slot_layout) for one of pool_tokens' modes.

        Reduces INSIDE the batch loop, so what crosses to the host is the slots
        and not the 197 tokens -- 86 KB per tile against 1.21 MB.
        '''
        self._require('tokens', 'pooled()')
        holder = {}

        def reduce(t):
            f, slots, layout = pool_tokens(t, mode, self.token_spec())
            holder['slots'], holder['layout'] = slots, layout
            return f.cpu()

        feats = self._run(images, reduce)
        return feats, holder['slots'], holder['layout']

    def token_spec(self) -> dict:
        '''pool_tokens' dict form of the same three numbers.'''
        return {'dim': self.spec.dim,
                'token_grid': self.spec.grid,
                'num_prefix': self.spec.num_prefix}
