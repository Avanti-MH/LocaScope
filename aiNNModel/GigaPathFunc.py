'''GigaPath (prov-gigapath/prov-gigapath) as a TileEncoder.

    encoder = GigaPathEncoderConfig().build(device)
    feats   = encoder(tiles)                  # [N, 1536]
    toks    = encoder.tokens(tiles)           # [N, 197, 1536]
    slots   = encoder.pooled(tiles, 'grid2x2')

The contract, the batch loop, the identity surface, `variant`, the poolings and
every exit are TileEncoderFunc's. What is left here is what is GigaPath's: the
frozen baseline its numbers form, the CLS as the answer to "which token is the
feature", and which poolings its 14x14 grid admits.

Token Merging used to live here and is gone. It was rejected on the numbers
(log/MILESTONE.log M3, log/TODO.log), and the new base is why it could not
simply be left in place: it merged tokens without changing patch_embed.grid_size,
so model_spec claimed 197 tokens over an output that carried 101. Nothing
noticed while features() was raw[:, 0], which does not care how many tokens
follow. pooling_kinds does care, and said so. GigaPathFunc_old.py keeps the code
for the equivalence tests that compare the two APIs.

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
validated with and crop_pct is an upstream slip. test_gigapath_equivalence holds
the baseline down against that same stored tensor, so a changed default fails
there rather than in a number nobody can trace.

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
#
# ONE cache for every encoder, not one per model. Freezing happens at the FIRST
# huggingface_hub import in the process, and that is usually not this module --
# bench_slidewin_pooling reaches transformers through TissueSegFunc before any
# encoder is named -- so with three different defaults the winner was whichever
# module happened to be imported first. Agreeing on one value makes that race
# harmless: whoever wins, the answer is the same directory. `setdefault`, so an
# exported HF_HOME still overrides all three; see jobscripts/_env.sh, which
# exports it before python starts and is the part that actually decides.
os.environ.setdefault('HF_HOME', '/work/u26130998/model_weights')

_DOTENV = _HERE.parent / '.env'
if _DOTENV.exists():
    from dotenv import load_dotenv
    load_dotenv(_DOTENV)          # override=False: an exported value still wins

# Which ACCOUNT's token this model needs, which is not always the same one.
# Every gated repo here is approved per HuggingFace account, and approval for
# prov-gigapath says nothing about MahmoodLab's -- so HF_TOKEN_<NAME> names a
# token for this model and HF_TOKEN is the shared fallback. Absent, nothing
# changes.
#
# It is written back into HF_TOKEN because that is the only name timm and
# huggingface_hub read: this module never passes token= anywhere, the download
# happens inside timm.create_model. Overwriting is safe for the same reason
# TileEncoderFunc._IMPLEMENTATIONS imports one module at a time -- one encoder
# per process, so there is no second reader to confuse.
#
# Repeated in each implementation rather than shared, for the reason the block
# above gives: a helper would have to be called before THIS module's timm
# import, and a rule of that shape has no enforcement.
if os.environ.get('HF_TOKEN_GIGAPATH'):
    os.environ['HF_TOKEN'] = os.environ['HF_TOKEN_GIGAPATH']

import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ConfigIdentity import ModelConfig, register  # noqa: E402
from TileEncoderFunc import (ModelOutputSpec, TileEncoder,  # noqa: E402
                             TileEncoderConfig, TransformConfig)
# Re-exported, not owned -- see the "Tokens and pooling" section below for why
# they moved and why this line has to stay.
from TileEncoderFunc import (SUMMARY_SLOT, _ring_bins,  # noqa: E402,F401
                             model_token_spec, pool_slots, pooling_kinds)


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
    # '' is what this config already did before `head` existed, so every id
    # written before it stays valid. Omitting this line is not the smaller
    # change but the bigger one: parts_against always emits a field the baseline
    # has no entry for, so all 73 stores under result/cache/ would take a new
    # encoder_id and miss.
    'head': '',
    # Same reason as 'head' above: a field the baseline has no entry for is
    # emitted always, so leaving it out would re-hash all 73 stores.
    'pooling': 'cls',
}


@register('gigapath')
@dataclass(frozen=True)
class GigaPathEncoderConfig(TileEncoderConfig):
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(source='timm', arch=GIGAPATH_ARCH,
                                            dtype='fp16'))

    #: Excluded for a different reason from batch_size: torch.compile reorders
    #: reductions, worth about 1e-7, two orders below what fp16 storage in a
    #: FeatureStore already discards. Splitting a cache on it would separate
    #: files nobody could tell apart afterwards.
    compile: bool = False

    NOT_IDENTITY = ('batch_size', 'compile')

    #: 'trunk' is not a second head here, it is a second name for the only one:
    #: prov-gigapath is a bare ViT and its own answer IS the trunk's CLS. Listed
    #: so a sweep spelling head='trunk' across encoders reaches this one, and
    #: mapped to '' so it cannot fork the cache.
    HEADS = {'': '', 'trunk': ''}

    #: Which reductions this model can be asked for, and what '' means for it.
    #:
    #: '' -> 'cls' is the only line here that a base class could not have
    #: written. It is GigaPath's own answer and not a default, and 'cls' rather
    #: than 'token' because it is the CLS specifically -- pooling_kinds' slot 0
    #: is tokens[:, 0] and timm's pool(pool_type='token') is x[:, 0], the same
    #: arithmetic under two names, which is why the _vector_from override that
    #: used to sit in this file could be deleted rather than rewritten.
    #:
    #: 'cls' appears twice because it is both the canonical value and a spelling
    #: of '', for the reason 'trunk' is above: one computation, one id.
    #:
    #: The grid family stops at 7x7 because the patch grid is 14x14 and
    #: pooling_kinds refuses a block size that does not divide it. Listing them
    #: says so at config time rather than after the weights have loaded.
    POOLINGS = {'': 'cls', 'cls': 'cls',
                'cls_avg': 'cls_avg', 'cls_std': 'cls_std',
                'rings3': 'rings3',
                'grid2x2': 'grid2x2', 'grid7x7': 'grid7x7',
                'tokens': 'tokens'}

    def build(self, device: torch.device, multi_gpu: bool = False):
        return GigaPathEncoder(self, device, multi_gpu=multi_gpu)

# ── Model ────────────────────────────────────────────────────────────────────

def _compile(model: torch.nn.Module,
             mode: str = 'reduce-overhead') -> torch.nn.Module:
    '''torch.compile. Private, and reached through EncoderConfig.compile, so a
    caller cannot apply it out of order with whatever else build() does to the
    model.'''
    return torch.compile(model, mode=mode)


# ── Tokens and pooling ───────────────────────────────────────────────────────
#
# The production feature keeps one vector per tile: pool(pool_type='token') is
# `x[:, 0]` (timm/layers/pool1d.py:global_pool_nlc), so 196 of the 197 tokens are
# discarded. `pooling_kinds` lets a caller keep them and try other reductions
# offline, without re-encoding and without editing the model.
#
# It is defined in TileEncoderFunc and re-exported here. It reads a spec and a
# tensor and nothing else, so it was never GigaPath's -- it only lived here
# while GigaPath was the only encoder. UNI2 is what made that visible: reaching
# these four names would have meant importing one implementation module from
# another, dragging this file's HF_HOME default, its timm import and its Token
# Merging patcher along for two functions about tensor shapes.
#
# The re-export is not a courtesy to old code, it is the point: every caller
# spells `from GigaPathFunc import pooling_kinds` and the move was meant to change
# where the definition lives, not what any of them import.


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
    test_gigapath_equivalence checks it at max|delta| == 0 and asserts the four
    preconditions the equality rests on. Going through model() rather than
    around it is what lets DataParallel work and removes the private attribute
    reach.
    '''

    BASELINE = _GIGAPATH_BASELINE

    def __init__(self, cfg: GigaPathEncoderConfig, device: torch.device,
                 multi_gpu: bool = False):
        self.cfg = cfg
        self.device = device
        # .env and the token are resolved at IMPORT, not here -- see the block
        # above `import timm`. Doing it at construction was too late for
        # HF_TOKEN_GIGAPATH to be visible when that block ran.
        model = cfg.model.build(num_classes=0, global_pool='')
        model = model.to(device).eval()

        if cfg.compile:
            model = _compile(model)
        if multi_gpu and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        self.model = model
        token = model_token_spec(model)
        self.model_spec = ModelOutputSpec(kind='tokens', dim=token['dim'],
                               feat_hw=token['feat_hw'],
                               num_prefix=token['num_prefix'])
        self._transform = cfg.transform.build()
        self._weights_id = None
