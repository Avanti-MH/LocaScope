'''ConvNeXt V2 (timm's `convnextv2_*` family) as a TileEncoder.

    encoder = ConvNeXtV2EncoderConfig().build(device)
    feats   = encoder(tiles)                  # [N, 768]  (Tiny)
    spatial = encoder.spatial(tiles)          # [N, 768, 7, 7]
    slots   = encoder.pooled(tiles, 'rings3')

The contract, the batch loop, the identity surface, `variant`, the poolings and
every exit are TileEncoderFunc's -- same as GigaPath and UNI2. What is
different here is what is different about a CNN: there is no CLS token and no
prefix, so the model's own answer to "what is the one vector for this tile" is
the GLOBAL AVERAGE of its last feature map, not a selected token.

WHY THIS IS 'spatial', NOT 'vector' -- built with `global_pool=''` rather than
`global_pool='avg'`, so `model(x)` returns the last stage's un-pooled feature
map ([N, 768, 7, 7] at 224x224 input for Tiny) rather than a flattened vector.
Measured, not assumed: `NormMlpClassifierHead(pool_type='')` sets
`global_pool` to a no-op and `fc` to `nn.Identity()` the moment `num_classes=0`
is also passed -- confirmed by constructing the model and reading
`m.head.global_pool` / `type(m.head.fc)` directly, the same way
`test_gigapath_equivalence` (removed 2026-10-05) held GigaPath's baseline down against a
stored tensor rather than trusting the docs. Keeping the map intact rather
than pooling it inside the model is what lets `TileEncoderFunc._pool`'s own
rank-4 branch ("[B, C, H, W] -> flatten -> pooling_kinds") handle GAP / rings
/ grid for this CNN with the SAME code that already does it for a ViT's
feature map -- nothing CNN-specific had to be written for that part.

7x7 IS TOO SMALL FOR A USEFUL GRID. GigaPath's patch grid is 14x14 and UNI2's
is 16x16, both of which admit a handful of block sizes; Tiny's last feature
map at the model's own validated 224 crop is 7x7, and 7 is prime, so the only
block sizes that divide it are 1 (== gap, redundant) and 7 (== 'tokens', no
reduction at all). POOLINGS below lists exactly what is legal instead of
including a 'grid' family that would raise "does not divide" on the first
forward -- the same discipline GigaPath's and UNI2's own POOLINGS tables
follow, just with a shorter answer because the model's own grid is smaller.

FROZEN FEATURE EXTRACTION ONLY. `TileEncoder._run` -- the method every exit on
this class goes through -- is `@torch.no_grad()`, which is right for GigaPath
and UNI2 (always used as a frozen encoder here) and is ALSO right for this
class whenever it is used the same way: as a frozen, natural-image-pretrained
CNN feature extractor to compare against the frozen pathology foundation
models. It is NOT what `training/MppRoutingHead`'s baseline 3 wants --
end-to-end fine-tuning needs gradients flowing through the trunk, which this
frozen-inference class structurally cannot give (no code path here ever calls
the model outside `torch.no_grad()`). That trainable arm builds its own
`nn.Module` directly from `timm.create_model(cfg.model.arch, ...)` using this
config's `model`/`transform` fields for the checkpoint name and preprocessing,
and trains it with an ordinary grad-enabled forward pass -- see
`training/MppRoutingHead/Models.py`. This class exists so ConvNeXt V2 can ALSO
be dropped into the exact same frozen-encoder-plus-head comparison GigaPath
and UNI2 already sit in (a natural-image prior instead of a pathology one),
not because baseline 3 is built on top of it.
'''
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

_HERE = Path(__file__).resolve().parent
for _d in (_HERE, _HERE.parent / 'utilities'):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

# Same reasoning and same ordering constraint as GigaPathFunc.py /
# Uni2Func.py: this MUST stay above `import timm`, because huggingface_hub
# freezes HF_HOME into module constants the moment IT is imported, and timm
# imports huggingface_hub. Repeated here rather than shared for the same
# reason those two repeat it: a helper would have to run before THIS module's
# `import timm`, and nothing enforces that ordering across files.
os.environ.setdefault('HF_HOME', '/work/u26130998/model_weights')

_DOTENV = _HERE.parent / '.env'
if _DOTENV.exists():
    from dotenv import load_dotenv
    load_dotenv(_DOTENV)          # override=False: an exported value still wins

# No HF_TOKEN_* swap here: unlike prov-gigapath and UNI2-h, timm's own
# convnextv2_* checkpoints are not gated -- a plain, ungated hub repo, so
# there is no per-model token to promote to HF_TOKEN.

import timm  # noqa: E402
import torch  # noqa: E402

from ConfigIdentity import ModelConfig, register  # noqa: E402
from TileEncoderFunc import (ModelOutputSpec, TileEncoder,  # noqa: E402
                             TileEncoderConfig, TransformConfig)


#: `convnextv2_tiny` (27.9M params, `depths=(3,3,9,3)`, `dims=(96,192,384,768)`
#: -- measured off timm's own model_args, not the paper's prose) with the
#: strongest checkpoint timm ships for this tier: self-supervised FCMAE
#: pretraining, fine-tuned on ImageNet-22k (~14.2M images, ~21.8k classes)
#: then ImageNet-1k (1,281,167 images, 1000 classes). Tiers below Nano only
#: ship the plain `fcmae_ft_in1k` tag -- no `_in22k_in1k` -- so 'tiny' is also
#: the smallest tier that gets this particular recipe.
CONVNEXTV2_ARCH = 'convnextv2_tiny.fcmae_ft_in22k_in1k'


# ── Configuration ─────────────────────────────────────────────────────────────

#: The zero point. `scale_size=256, crop_size=224` is not copied from
#: GigaPath's baseline by habit -- it is this checkpoint's OWN validated
#: recipe: `timm.create_model(CONVNEXTV2_ARCH).pretrained_cfg['crop_pct']` is
#: 0.875, and 224 / 0.875 = 256. mean/std are plain ImageNet stats (measured
#: off the same `pretrained_cfg`), not CONCH's OpenAI-CLIP pair -- this
#: checkpoint was never touched by CLIP's training run.
_CONVNEXTV2_BASELINE = {
    'model': ModelConfig(source='timm', arch=CONVNEXTV2_ARCH, dtype='fp16'),
    'transform': TransformConfig(scale_size=256, crop_size=224,
                                 interpolation='bicubic',
                                 mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225),
                                 preprocess='none'),
    'head': '',
    'pooling': 'gap',
}


@register('convnext_v2')
@dataclass(frozen=True)
class ConvNeXtV2EncoderConfig(TileEncoderConfig):
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(source='timm', arch=CONVNEXTV2_ARCH,
                                            dtype='fp16'))

    NOT_IDENTITY = ('batch_size',)

    #: One trunk, one exit -- same reason GigaPath's 'trunk' is an alias for
    #: '' rather than a second computation: there is no separate pooler head
    #: here the way CONCH has one.
    HEADS = {'': '', 'trunk': ''}

    #: '' -> 'gap' is this model's own answer, not a default: with no prefix
    #: token, slot 0 -- the tile's single summary vector -- IS the global
    #: average of the last feature map (SUMMARY_SLOT[False] == 'gap'; see
    #: TileEncoderFunc.pool_slots). 'cls'/'cls_avg'/'cls_std' are absent on
    #: purpose: those need a prefix token distinct from the patch average, and
    #: this model has none.
    #:
    #: No grid family: the last feature map is 7x7 at this checkpoint's own
    #: 224 crop, and 7 is prime -- the only block sizes that divide it are 1
    #: (== gap) and 7 (== 'tokens'), so pooling_kinds' grid path has nothing
    #: to offer here that 'gap' and 'tokens' do not already give.
    POOLINGS = {'': 'gap', 'gap': 'gap', 'rings3': 'rings3', 'tokens': 'tokens'}

    def build(self, device: torch.device, multi_gpu: bool = False):
        return ConvNeXtV2Encoder(self, device, multi_gpu=multi_gpu)


# ── Encoder ──────────────────────────────────────────────────────────────────

class ConvNeXtV2Encoder(TileEncoder):
    '''ConvNeXt V2, built as a frozen feature extractor.

    `global_pool='' , num_classes=0` are the CNN equivalent of GigaPath's
    `global_pool='', num_classes=0`: no classifier, and the reduction this
    class's OWN pooling machinery controls rather than one baked into the
    model. Unlike a bare `global_pool='avg'` build, this keeps the last stage's
    spatial map intact ([N, 768, 7, 7] for Tiny at 224x224), which is what
    lets `rings3`/`tokens` reach real spatial structure instead of a single
    already-collapsed vector.
    '''

    BASELINE = _CONVNEXTV2_BASELINE

    def __init__(self, cfg: ConvNeXtV2EncoderConfig, device: torch.device,
                multi_gpu: bool = False):
        self.cfg = cfg
        self.device = device
        model = cfg.model.build(num_classes=0, global_pool='')
        model = model.to(device).eval()

        if multi_gpu and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        self.model = model
        self._transform = cfg.transform.build()
        self._weights_id = None
        self._feat_hw: Tuple[int, int] = self._measure_feat_hw()

    def _measure_feat_hw(self) -> Tuple[int, int]:
        '''Run one dummy tile through the trunk to read off H, W.

        Not derived from crop_size // 32: total stride is stem(4) *
        stage1..3 downsample(2 each) = 32 for every convnextv2_* variant
        TODAY, but that is an assumption about the architecture rather than a
        fact this file has checked, and it is exactly the kind of assumption
        _vit_model_spec's own docstring warns against (deriving feat_hw from
        `patch_embed.grid_size` instead of measuring it is what made Token
        Merging's broken output invisible). One forward pass at construction
        costs nothing worth avoiding.
        '''
        m = getattr(self.model, 'module', self.model)
        crop = int(self.cfg.transform.crop_size)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, crop, crop, device=self.device,
                                dtype=next(m.parameters()).dtype)
            out = m(dummy)
        if out.ndim != 4:
            raise RuntimeError(
                f'{type(self).__name__}: expected a [1, C, H, W] feature map '
                f'with global_pool=\'\', got {tuple(out.shape)}. Was the model '
                f'built with num_classes=0, global_pool=\'\'?')
        return (int(out.shape[2]), int(out.shape[3]))

    def _compute_model_spec(self):
        '''A CNN feature map, not a ViT's tokens: kind='spatial', no prefix.

        `dim` is `self.model.num_features` (the last stage's channel width --
        768 for Tiny), read off the model rather than hardcoded so a config
        pointed at a different tier still answers correctly.
        '''
        m = getattr(self.model, 'module', self.model)
        return ModelOutputSpec(kind='spatial', dim=int(m.num_features),
                               feat_hw=self._feat_hw, num_prefix=0)

    def _spatial_forward(self, batch):
        '''The model's own forward IS the spatial map already -- global_pool
        is disabled and fc is Identity (num_classes=0), so there is no
        separate "ask for intermediates" path the way a ViT needs
        forward_intermediates. Unlike `_vit_spatial_forward`, this is not an
        override of what `_run` calls by default -- `_run`'s default forward
        (`self.model`) already returns this, so `_pool` reaches it through the
        ordinary rank-4 branch without `spatial()` ever being called. Provided
        anyway so `spatial()`/`spatial_spec()` work as a named exit too.
        '''
        m = getattr(self.model, 'module', self.model)
        return m(batch)
