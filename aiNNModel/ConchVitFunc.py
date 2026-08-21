'''CONCH's vision tower (MahmoodLab/conch) as a TileEncoder.

    encoder = ConchVitEncoderConfig().build(device)          # head='attn_pool'
    feats   = encoder(tiles)                  # [N, 512]
    toks    = encoder.tokens(tiles)           # [N, 785, 768]  -- the trunk
    slots   = encoder.pooled(tiles, 'grid2x2')

The first model here with a HEAD. GigaPath and UNI2 are bare ViTs: their trunk
output is all there is, and cfg.head is '' because there is nothing attached.
CONCH's image tower is a ViT under an attentional pooler -- 0.5 M trained
parameters that map [N, 785, 768] to [N, 512] -- so the two exits are vectors of
different widths in different spaces, and which one you get has to be a config
field rather than a call-site argument.

    head='attn_pool'   the tower's own answer. 512-d.  pooling='identity',
                       because the head IS the pooling and counting it twice
                       would put a reduction where none happens.
    head='trunk'       the ViT alone, 768-d, the same shape GigaPath and UNI2
                       have. This is the arm that goes into the pooling bench:
                       cls / rings3 / grid2x2 mean here exactly what they mean
                       there.

Why there is no `import conch`
------------------------------
The package would buy nothing this file cannot do in fifty lines, and it would
cost the two things below. What is copied from it is copied verbatim, with the
line numbers of the original, so a reader can diff rather than trust.

    strict     `factory.load_checkpoint` calls load_state_dict(strict=False) and
               DISCARDS `missing, unexpected` (factory.py:27-30). A key that
               does not land is silent there. Here the state dict is split by
               prefix and each part is loaded strict=True, so a checkpoint that
               does not fit the tower says so.
    decoder    CoCa builds a text decoder whose weights were REMOVED from the
               public release (README:145). Constructing it leaves a randomly
               initialised module in the model, and a random module makes
               weights_id different in every process -- so no store filename
               would ever repeat. Only the vision half is built here.

Upstream, verified in /work/u26130998/CONCH
-------------------------------------------
    coca_model.py:76-83      the trunk's constructor arguments
    coca_model.py:86         trunk.forward = trunk.forward_features (and why
                             global_pool='' makes it unnecessary here)
    vision_tower.py:47-49    the pooler and ln_contrast
    vision_tower.py:121-130  forward_no_head: trunk -> attn_pool[:, 0] -> ln
    README.md:150-157        "as a vision encoder for histopathology images",
                             which is proj_contrast=False, normalize=False --
                             i.e. WITHOUT proj_contrast. The projected path is
                             for image-TEXT retrieval, which this is not.
    transform.py:31-38       Resize(448, BICUBIC) -> CenterCrop(448)
    factory.py:71-72         mean/std overridden to OpenAI CLIP's, NOT ImageNet
    factory.py:57-63         hf_hub_download(repo, 'pytorch_model.bin')

Two deliberate deviations
-------------------------
The head runs in fp32. _run calls .float() before the reduce (and _apply_head
is part of the reduce), while upstream runs the whole tower under one autocast.
That is why conch's LayerNorm subclass -- whose only job is casting back to the
input dtype -- is not copied: at fp32 in, it and nn.LayerNorm are the same
function. Ours is the more precise of the two, for the reason _run gives.

`proj_contrast` is not built. It is a 512x512 matrix that only matters when
comparing against text embeddings, and leaving it out keeps weights_id honest
about what actually runs.

Gated repo: HF_TOKEN_CONCH or HF_TOKEN, exported or in the .env this loads at
import time, AND the account
must have been granted access on the model page. A 401 here is an account
problem, not a code one.
'''
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _d in (_HERE, _HERE.parent / 'utilities'):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

# MUST stay above `import timm`, for the reason GigaPathFunc spells out in full:
# huggingface_hub freezes HF_HOME into module-level constants at import time,
# and timm imports huggingface_hub.
#
# The same value the other two set, for the reason GigaPathFunc spells out: the
# first huggingface_hub import in the process wins, and it is rarely this one.
# This names the root; the hub CACHE is the `hub/` under it, content-addressed
# blobs plus snapshot symlinks. CONCH_LOCAL is a sibling of that `hub/`, holding
# the files under their own names. Same root, two layouts, and only the first
# one huggingface_hub knows how to read.
os.environ.setdefault('HF_HOME', '/work/u26130998/model_weights')

_DOTENV = _HERE.parent / '.env'
if _DOTENV.exists():
    from dotenv import load_dotenv
    load_dotenv(_DOTENV)          # override=False: an exported value still wins

# HF_TOKEN_CONCH if it is set, else the shared HF_TOKEN. GigaPathFunc carries
# the full reasoning: gated approval is per ACCOUNT, and MahmoodLab's says
# nothing about prov-gigapath's. Written back into HF_TOKEN so that _download's
# explicit token= and anything else reading the environment agree -- one name,
# one value, rather than a caller having to know which of the two applies here.
if os.environ.get('HF_TOKEN_CONCH'):
    os.environ['HF_TOKEN'] = os.environ['HF_TOKEN_CONCH']

import torch  # noqa: E402
from torch import nn  # noqa: E402

from ConfigIdentity import ModelConfig, register, weights_id  # noqa: E402
from TileEncoderFunc import (ModelOutputSpec, TileEncoder,  # noqa: E402
                             TileEncoderConfig, TransformConfig,
                             model_token_spec)


CONCH_REPO = 'MahmoodLab/conch'
CONCH_FILE = 'pytorch_model.bin'

#: Fetched and thrown away, exactly as factory.py:57-60 does. See _download.
CONCH_META = 'meta.yaml'

#: The weights on disk, tried before the hub API. Not a mirror and not a
#: convenience: the hub path needs a token whose fine-grained scope covers
#: public gated repos, and this directory needs only the account's approval,
#: which the `git clone https://huggingface.co/MahmoodLab/CONCH` that produced
#: it already used.
#:
#: It WAS that clone; its .git was deleted to reclaim the second copy git-lfs
#: keeps, so it is now a plain directory of files. The difference is not
#: cosmetic -- there is no `git pull` to repair it any more, and the hub
#: fallback below is the only remedy left. That fallback is unchanged, so
#: deleting the directory still costs a download and nothing else, but the
#: download is the one that has failed on gating before.
#:
#: Override with CONCH_WEIGHTS_DIR. An absolute path is deliberate: this is
#: outside the checkout, like everything else a run needs.
CONCH_LOCAL = Path(os.environ.get(
    'CONCH_WEIGHTS_DIR', '/work/u26130998/model_weights/CONCH'))

#: source='local' and this class, because there is no timm hub entry to name.
#: ModelConfig documents 'package.module:Class' as exactly what that source
#: means, so this is the contract rather than a way around it.
CONCH_CLASS = 'timm.models.vision_transformer:VisionTransformer'

#: The trunk's seven constructor arguments, from coca_model.py:76-83 with
#: CoCaVisionCfg's defaults (:39-51) for what the JSON leaves out. Keyed by
#: nothing: there is one CONCH.
#:
#: `num_classes=0` and `global_pool=''` are NOT here -- they go to build(), the
#: same two arguments GigaPath and UNI2 pass, on ModelConfig's rule that
#: construction constants of the domain belong at the call site.
#:
#: They are also why upstream's `trunk.forward = trunk.forward_features`
#: (coca_model.py:86) is not reproduced. That line exists because CoCa builds
#: the trunk with timm's default global_pool='token', which would pool. At
#: global_pool='' timm's forward_head pools nothing, fc_norm is Identity and
#: head is Identity, so forward() IS forward_features. Copying the assignment
#: anyway would have cost something real: an instance-level `forward` is a
#: method bound to the ORIGINAL module, and DataParallel replicates by copying
#: __dict__ -- so every replica would run the original's parameters, on the
#: wrong device, without raising.
_TRUNK_KWARGS = {
    'embed_dim':        768,
    'depth':            12,
    'num_heads':        12,
    'mlp_ratio':        4,
    'img_size':         448,
    'patch_size':       16,
    'dynamic_img_size': True,
}

#: The head, from vision_tower.py:47-49 with conch_ViT-B-16.json's embed_dim 512
#: and n_queries_contrast 1.
_POOL_KWARGS = {
    'd_model':     512,
    'context_dim': 768,
    'n_head':      8,
    'n_queries':   1,
}

#: OpenAI CLIP's, copied from constants.py:3-4. NOT ImageNet's: factory.py:71-72
#: overwrites visual.image_mean/std with these after building, and
#: create_model_from_pretrained:104 is what reads them back into the transform.
#: Copying GigaPath's block here would be the easy mistake and would show up
#: only as slightly worse numbers.
_OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
_OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)


# ── the head ──────────────────────────────────────────────────────────────────

class AttentionalPooler(nn.Module):
    '''Copied from conch/open_clip_custom/transformer.py, unchanged.

    Verbatim on purpose. Every parameter name here is a key in the released
    checkpoint, so a "tidier" rewrite -- self.q instead of self.query, one
    LayerNorm instead of ln_q and ln_k -- is a load_state_dict failure at best
    and a silently unloaded module at worst. The one thing not copied is the
    norm_layer default: upstream's LayerNorm subclass exists to cast back to the
    input dtype, and this runs at fp32 where it and nn.LayerNorm agree.

    n_queries=1 for CONCH's contrastive pooler, so the [N, 1, 512] it returns is
    sliced to [N, 512] by the caller -- vision_tower.py:123 does the same.
    '''

    def __init__(self, d_model: int, context_dim: int, n_head: int = 8,
                 n_queries: int = 256, norm_layer=nn.LayerNorm):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim,
                                          vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        if attn_mask is not None:
            attn_mask = ~attn_mask.bool()
        out = self.attn(self._repeat(q, N), x, x, need_weights=False,
                        key_padding_mask=attn_mask)[0]
        return out.permute(1, 0, 2)  # LND -> NLD

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


# ── weights ───────────────────────────────────────────────────────────────────

def _download() -> Path:
    '''The checkpoint, from the hub cache under HF_HOME. factory.py:57-63.

    The meta.yaml fetch is upstream's and its result is discarded there too --
    `_ = hf_hub_download(...)`. It is not decoration: this repo is GATED, and
    the request that discovers that should be the one for a few kilobytes
    rather than the one for a gigabyte. A 401 or a GatedRepoError arrives here,
    named, before anything long starts.

    Separate from the split below for the same reason: the likely failure is
    access, and it should happen on its own line with the repo in it rather
    than inside a state-dict comprehension.
    '''
    local = CONCH_LOCAL / CONCH_FILE
    if local.exists():
        # The meta.yaml probe is skipped with it. Its only job is to discover a
        # permission problem for a few kilobytes instead of a gigabyte, and a
        # file already on disk has no permission to discover.
        return local

    from huggingface_hub import hf_hub_download
    token = os.environ.get('HF_TOKEN')
    hf_hub_download(CONCH_REPO, filename=CONCH_META, token=token)
    return Path(hf_hub_download(CONCH_REPO, filename=CONCH_FILE, token=token))


def _split(state: dict, prefix: str) -> dict:
    '''The keys under one prefix, with the prefix removed.

    Raises on empty rather than handing back {} for load_state_dict to accept
    under strict=False: a prefix that matches nothing means the checkpoint is
    laid out differently from what this file assumes, and that is exactly the
    case upstream's strict=False turns into a randomly initialised module.
    '''
    out = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not out:
        raise KeyError(
            f'no keys under {prefix!r} in the checkpoint. It holds '
            f'{len(state)} keys beginning: '
            f'{sorted({k.split(".")[0] for k in state})}')
    return out


# ── configuration ─────────────────────────────────────────────────────────────

#: The zero point. head is 'attn_pool' and not '' because that is what '' means
#: for this model, and __post_init__ has already resolved it by the time
#: identity is computed -- a baseline holding '' would never compare equal.
_CONCHVIT_BASELINE = {
    'model': ModelConfig(source='local', arch=CONCH_CLASS, dtype='fp16'),
    'transform': TransformConfig(scale_size=448, crop_size=448,
                                 interpolation='bicubic',
                                 mean=_OPENAI_MEAN, std=_OPENAI_STD,
                                 preprocess='none'),
    'repo': CONCH_REPO,
    'head': 'attn_pool',
    'pooling': 'identity',
}


@register('conch_vit')
@dataclass(frozen=True)
class ConchVitEncoderConfig(TileEncoderConfig):
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(source='local', arch=CONCH_CLASS,
                                            dtype='fp16'))
    transform: TransformConfig = field(
        default_factory=lambda: TransformConfig(
            scale_size=448, crop_size=448, interpolation='bicubic',
            mean=_OPENAI_MEAN, std=_OPENAI_STD, preprocess='none'))

    #: Which repository the weights come from. A field and not a constant
    #: because it is the one thing a caller might legitimately vary -- a
    #: finetuned copy -- and because `arch` cannot carry it: under
    #: source='local' arch names the CLASS, and the class is timm's ViT for
    #: every CONCH there will ever be.
    #:
    #: weights_id already hashes what the file CONTAINS, so this is not what
    #: makes two finetunes distinguishable. It is what makes the id READABLE,
    #: the same role ModelConfig.arch plays for the other two encoders.
    repo: str = CONCH_REPO

    #: '' is the tower's own answer, which is the attentional pooler. Named
    #: rather than left blank: cfg.head then always points at a head that
    #: exists, and 'trunk' is a genuinely different vector rather than an alias
    #: -- 768-d out of the ViT against 512-d out of the pooler.
    HEADS = {'': 'attn_pool', 'attn_pool': 'attn_pool', 'trunk': 'trunk'}

    #: Both heads' vocabularies in one table, because POOLINGS is a class
    #: attribute and cannot depend on a field. __post_init__ rejects the
    #: combinations this over-admits.
    #:
    #: The grid family is the divisors of 28: 2, 4, 7, 14. 28 divides too and is
    #: left out because grid28x28 is one slot per cell, which is what 'tokens'
    #: already is -- two names for one reduction would be two ids over one set
    #: of vectors.
    #:
    #: 'identity' is the whole of what head='attn_pool' allows: the pooler
    #: already produced the vector, and nothing here reduces one vector further.
    #: When something does, it goes in this table and in the pair rule below --
    #: not before.
    #: No '' entry, unlike every other encoder's table. This is the first config
    #: with TWO heads, and "this model's own answer" is a different word for
    #: each of them -- 'identity' behind the pooler, 'cls' for the bare trunk.
    #: A single table cannot say that, so __post_init__ resolves '' from
    #: _BY_HEAD before the base ever looks the value up here.
    POOLINGS = {'identity': 'identity',
                'cls': 'cls', 'cls_avg': 'cls_avg', 'cls_std': 'cls_std',
                'rings3': 'rings3',
                'grid2x2': 'grid2x2', 'grid4x4': 'grid4x4',
                'grid7x7': 'grid7x7', 'grid14x14': 'grid14x14',
                'tokens': 'tokens'}

    #: Which poolings each head admits, FIRST ONE FIRST: the leading entry is
    #: what '' resolves to for that head. The base's __post_init__ checks each
    #: field against its own table and cannot see a pair; this is the pair, and
    #: it doubles as the per-head default because those are the same knowledge.
    _BY_HEAD = {'attn_pool': ('identity',),
                'trunk': ('cls', 'cls_avg', 'cls_std', 'rings3',
                          'grid2x2', 'grid4x4', 'grid7x7', 'grid14x14',
                          'tokens')}

    def __post_init__(self):
        """Resolve '' per head, then the base's checks, then the pair.

        The order matters. '' has to become concrete BEFORE the base validates,
        because the base looks it up in POOLINGS and POOLINGS has no entry for
        it -- there cannot be one, since what '' means here depends on which
        head is configured. Reading HEADS directly is how this sees the
        canonical head one line before the base sets it; an unknown head falls
        through untouched and the base raises on it, which is the right error to
        report first.

        The pair check afterwards is the part the base structurally cannot do:
        head and pooling are each legal alone and still wrong together --
        head='attn_pool' hands back one 512-d vector, so 'grid2x2' has no token
        axis to work on. Caught here rather than at the first forward, which is
        after a gigabyte of weights has loaded.
        """
        head = self.HEADS.get(self.head, self.head)
        if not self.pooling and head in self._BY_HEAD:
            object.__setattr__(self, 'pooling', self._BY_HEAD[head][0])
        super().__post_init__()
        ok = self._BY_HEAD[self.head]
        if self.pooling not in ok:
            raise ValueError(
                f'head={self.head!r} admits pooling {", ".join(repr(p) for p in ok)}'
                f', got {self.pooling!r}. '
                + ("The attentional pooler already produced the vector; there is "
                   "no reduction of one vector yet. Add it to POOLINGS and to "
                   "_BY_HEAD when there is."
                   if self.head == 'attn_pool' else
                   'The trunk hands back tokens, so it takes the token '
                   'poolings.'))

    def build(self, device: torch.device, multi_gpu: bool = False):
        return ConchVitEncoder(self, device, multi_gpu=multi_gpu)


# ── encoder ───────────────────────────────────────────────────────────────────

class ConchVitEncoder(TileEncoder):
    '''CONCH's vision tower, built from the checkpoint alone.

    self.model is the TRUNK, as it is for every encoder here, so tokens() hands
    back the ViT's own 785 tokens whichever head is configured. The head lives
    in _apply_head and only exists when cfg.head asks for it -- which is also
    what keeps weights_id from covering parameters that never run.
    '''

    BASELINE = _CONCHVIT_BASELINE

    def __init__(self, cfg: ConchVitEncoderConfig, device: torch.device,
                 multi_gpu: bool = False):
        self.cfg = cfg
        self.device = device

        state = torch.load(_download(), map_location='cpu')
        state = state.get('state_dict', state)

        model = cfg.model.build(num_classes=0, global_pool='', **_TRUNK_KWARGS)
        model.load_state_dict(_split(state, 'visual.trunk.'), strict=True)
        model = model.to(device).eval()

        if cfg.head == 'trunk':
            self.head = None
        else:
            pool = AttentionalPooler(**_POOL_KWARGS)
            pool.load_state_dict(_split(state, 'visual.attn_pool_contrast.'),
                                 strict=True)
            ln = nn.LayerNorm(_POOL_KWARGS['d_model'])
            ln.load_state_dict(_split(state, 'visual.ln_contrast.'), strict=True)
            # A bare Module and not a Sequential: the two are applied with a
            # slice between them, so a container that is itself callable and
            # would run neither is a trap. This one only exists to hold them,
            # so that .to(device), .eval() and weights_id see both at once.
            self.head = nn.Module()
            self.head.pool, self.head.ln = pool, ln
            self.head = self.head.to(device).eval()

        del state
        if multi_gpu and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        self.model = model
        token = model_token_spec(model)
        self.model_spec = ModelOutputSpec(kind='tokens', dim=token['dim'],
                                          feat_hw=token['feat_hw'],
                                          num_prefix=token['num_prefix'])
        self._transform = cfg.transform.build()
        self._weights_id = None

    def identity_parts(self):
        '''Config, trunk weights, and the head's weights as well.

        IdentifiedBuild.weights_id hashes self.model, and self.model is only the
        trunk. The pooler is half a million trained parameters that decide every
        vector features() returns, so leaving it out would let two encoders with
        different heads share an id -- exactly what cfg.head being in identity is
        meant to prevent, undone one layer down.
        '''
        parts = super().identity_parts()
        if self.head is not None:
            parts.append(f'head_weights={weights_id(self.head)}')
        return parts

    def _apply_head(self, raw: torch.Tensor) -> torch.Tensor:
        '''The attentional pooler, or nothing.

        vision_tower.py:121-130 in three lines: pool, take the single query,
        normalise. Without proj_contrast -- README:150-157 says the unprojected
        vector is the one to use as an image encoder, and the projected one is
        for image-text retrieval.

        Runs on the device and in fp32: _run hands the reduce a device tensor it
        has already called .float() on, so nothing crosses to the host and back
        in the middle of a batch.
        '''
        if self.head is None:
            return raw
        return self.head.ln(self.head.pool(raw)[:, 0])
