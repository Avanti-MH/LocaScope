'''UNI2-h (MahmoodLab/UNI2-h) as a TileEncoder.

    encoder = Uni2EncoderConfig().build(device)
    feats   = encoder(tiles)                  # [N, 1536]
    toks    = encoder.tokens(tiles)           # [N, 265, 1536]
    slots   = encoder.pooled(tiles, 'grid2x2')

The contract, the batch loop, the identity surface, `variant`, `pooling_kinds`
and `model_token_spec` are all TileEncoderFunc's -- the last two moved there
when this file was written, because they read a spec and a tensor and know
nothing about whichever ViT produced them.

What is here is what is UNI2's.

Eight register tokens
---------------------
UNI2-h is built with `reg_tokens=8`, so timm lays the sequence out as

    [CLS] [reg x 8] [patch x 256]        num_prefix_tokens = 9,  T = 265

and this is the first model in the project where `num_prefix` is neither 0 nor
1. Nothing here hardcodes it: `model_token_spec` reads `num_prefix_tokens` off
the model, and pooling_kinds slices `tokens[:, p:]`. That is not defensive
programming, it is the whole reason the field exists -- averaging eight
registers in as if they were image content lowers every pooled score by a
little and raises nothing, which reads as "patch pooling does not help".

The constructor arguments are not settings
------------------------------------------
`_TIMM_KWARGS` is keyed by arch and not exposed as config fields, on the rule
ModelConfig states: construction kwargs are constants of the domain. These
particular ones are more than that -- they ARE the architecture, and upstream
passes all twelve at every one of its own call sites:

    UNI/README.md:362-377                  hub path, timm_kwargs
    UNI/README.md:395-411                  local-checkpoint path, same twelve
                                           plus model_name vit_giant_patch14_224
    UNI/uni/get_encoder/get_encoder.py:128 the library path, same twelve
    UNI/README_old.md:50                   UNI v1: init_values + dynamic_img_size

Hand UNI2-h's twelve to the v1 checkpoint and `load_state_dict(strict=True)`
does not fit; the dict keyed by arch is what keeps the pairing from being a
call-site habit.

They are not config fields but they DO enter identity, through
Uni2Encoder.identity_parts -- three of them change the forward pass without
changing a parameter, so weights_id cannot see them. _recipe_parts says which
three.

Preprocessing
-------------
UNI ships two preprocessing paths and they do not agree, exactly as GigaPath
does:

    README.md:377    create_transform(**resolve_data_config(pretrained_cfg))
    get_encoder.py:40  Resize(224) -> CenterCrop(224) -> ToTensor -> Normalize

The second is torchvision's `Resize`, whose default interpolation is BILINEAR;
timm's resolve_data_config path takes whatever the hub's pretrained_cfg
declares, which for these checkpoints is usually bicubic. On a 256px tile the
two agree on everything else -- Resize(224) makes the shorter side 224 and the
CenterCrop is then a no-op, so crop_pct is 1.0 either way -- and differ only in
the resampling filter.

The baseline takes the LIBRARY path, bilinear, for the reason GigaPathFunc
gives for taking prov-gigapath's code over its config.json: the code is what
produced the published numbers, and the metadata is a claim about the weights
rather than a measurement of them. Pin it against the shipped pipeline before
trusting a number that came out of it -- same tensor on a real 256px tile, not
the same repr.

Gated repo: HF_TOKEN_UNI2 or HF_TOKEN, exported or in the .env this loads at
import time. Approval is per HuggingFace ACCOUNT, so having prov-gigapath does
not imply having this.
'''
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _d in (_HERE, _HERE.parent / 'utilities'):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

# MUST stay above `import timm`, for the reason GigaPathFunc spells out in
# full: huggingface_hub freezes HF_HOME into module-level constants at import
# time, and timm imports huggingface_hub. Repeated here rather than shared,
# because the constraint is "before THIS module's timm import" and a module
# that has to be imported first to be effective is a rule with no enforcement.
#
# The same value GigaPathFunc and ConchVitFunc set, and that is the point: the
# freeze happens at the first huggingface_hub import in the process, which is
# often neither of them. GigaPathFunc carries the full argument.
os.environ.setdefault('HF_HOME', '/work/u26130998/model_weights')

_DOTENV = _HERE.parent / '.env'
if _DOTENV.exists():
    from dotenv import load_dotenv
    load_dotenv(_DOTENV)          # override=False: an exported value still wins

# HF_TOKEN_UNI2 if it is set, else the shared HF_TOKEN. GigaPathFunc carries the
# full reasoning: gated approval is per ACCOUNT, and this one is MahmoodLab's
# while prov-gigapath's is not. Written back into HF_TOKEN because timm's hub
# path is what downloads here and HF_TOKEN is the only name it reads.
if os.environ.get('HF_TOKEN_UNI2'):
    os.environ['HF_TOKEN'] = os.environ['HF_TOKEN_UNI2']

import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ConfigIdentity import ModelConfig, enc, register  # noqa: E402
from TileEncoderFunc import (ModelOutputSpec, TileEncoder,  # noqa: E402
                             TileEncoderConfig, TransformConfig,
                             model_token_spec, pooling_kinds)


UNI2_ARCH = 'hf-hub:MahmoodLab/UNI2-h'
UNI_ARCH = 'hf-hub:MahmoodLab/UNI'


#: What each checkpoint has to be built as. Keyed by arch so the pairing cannot
#: come apart: UNI2-h's kwargs applied to UNI-v1 build a tower its state dict
#: does not fit, and the reverse builds a ViT-L where a ViT-H was meant.
#:
#: Transcribed from UNI/README.md:362-377 and cross-checked against
#: UNI/uni/get_encoder/get_encoder.py:128-143, which spell the same twelve.
#: `num_classes: 0` is upstream's thirteenth and is passed at build() instead,
#: with global_pool='', on ModelConfig's rule about domain constants.
#:
#: They are NOT config fields, and they DO enter identity -- see
#: `_recipe_parts` and Uni2Encoder.identity_parts. Editing a value here
#: changes every vector, so it has to change the id, and "treat this dict as
#: frozen" written in a comment is not a mechanism.
#:
#: Every value is a scalar, layer types included: they are NAMES looked up in
#: _LAYER_CLASSES, not the classes themselves. See that table for why.
_TIMM_KWARGS = {
    UNI2_ARCH: {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'no_embed_class': True,
        'mlp_layer': 'SwiGLUPacked',
        'act_layer': 'SiLU',
        'reg_tokens': 8,
        'dynamic_img_size': True,
    },
    UNI_ARCH: {
        'init_values': 1e-5,
        'dynamic_img_size': True,
    },
}

#: Layer types a build recipe may name, and what each name resolves to.
#:
#: The table exists so a recipe can be written in strings. A class object in
#: there would be three separate problems, and only the first is obvious:
#:
#:   json      `config_json` runs the fields through json.dumps, and a class is
#:             not serialisable. The registry is what the repo already does for
#:             configs (ConfigIdentity._REGISTRY) and for the same reason:
#:             something has to survive the round trip through a file.
#:   identity  a class encodes as its import path, so timm moving it between
#:             modules would re-hash every store for a refactor that changed no
#:             arithmetic. The NAME is ours; the path is timm's. Only ours is a
#:             stable quantity (ConfigIdentity's rule 2).
#:   reach     a caller could pass any callable at all, and the first sign
#:             would be a tower that trains nothing and encodes fine.
#:
#: A recipe that needs a type not listed here adds it, deliberately, rather than
#: reaching into timm from the call site. An unknown name raises and lists what
#: is known -- the same shape as `config_from`, and for the same reason: the
#: usual failure is not a typo but an entry nobody added.
_LAYER_CLASSES = {
    'SwiGLUPacked': timm.layers.SwiGLUPacked,
    'Mlp':          timm.layers.Mlp,
    'SiLU':         torch.nn.SiLU,
    'GELU':         torch.nn.GELU,
}

#: Which recipe keys hold a layer name. Two, because two are used; a recipe that
#: needs norm_layer adds it here and to the table above in the same edit.
_LAYER_KEYS = ('mlp_layer', 'act_layer')


def layer_class(name: str):
    '''The class a recipe's layer name stands for.'''
    try:
        return _LAYER_CLASSES[name]
    except KeyError:
        known = ', '.join(sorted(_LAYER_CLASSES))
        raise KeyError(
            f'no layer class registered as {name!r}. Registered: {known}. Add '
            f'an entry to _LAYER_CLASSES rather than putting the class in a '
            f'recipe -- see the note there.') from None


def _recipe_parts(recipe: dict) -> list:
    '''`arch_kwargs.name=value` for a build recipe.

    Appended to identity because most of this dict, but NOT all of it, is
    already covered by weights_id -- and the part that is not covered is the
    silent part. weights_id hashes name, shape, dtype and bytes of the state
    dict, so anything that changes a parameter moves it: embed_dim, depth,
    patch_size, reg_tokens, mlp_ratio, mlp_layer, img_size and no_embed_class
    all resize something. Three do not:

        act_layer        SiLU or GELU, no parameters either way
        num_heads        qkv is (3*dim, dim) whatever the head count; heads
                         change the reshape, not the tensor
        dynamic_img_size a forward-path flag

    Change one of those three and every vector changes while the state dict is
    byte-identical, so the id would not move and every store written by the old
    value would still be a cache hit. That is ConfigIdentity's rule 1 -- an
    existing name given a new meaning -- arriving through a dict instead of
    through a field.

    All twelve go in rather than only those three. Which kwargs happen to be
    covered by the weights is a fact about timm's parameterisation, not about
    this project, and keeping a list of them in sync with timm is a derivation
    that can go stale silently. Twelve short strings cost nothing.

    Takes the RECIPE and not the resolved kwargs, so what identity records is
    what was written -- 'SiLU', a name this module owns -- rather than what it
    resolved to. That is the whole point of _LAYER_CLASSES: the name is stable
    even when timm moves the class.
    '''
    return [f'arch_kwargs.{key}={enc(value)}'
            for key, value in sorted(recipe.items())]


# ── Configuration ─────────────────────────────────────────────────────────────

#: The zero point. Editing this invalidates every id ever written, on purpose;
#: editing a dataclass DEFAULT does not -- it splits new from old instead, which
#: is why the two are separate. See ConfigIdentity for the four cases.
_UNI2_BASELINE = {
    'model': ModelConfig(source='timm', arch=UNI2_ARCH, dtype='fp16'),
    # bilinear, not bicubic: torchvision's Resize default, which is what
    # get_encoder.py:40 applies. See the module docstring -- upstream's two
    # paths differ here and only here.
    'transform': TransformConfig(scale_size=224, crop_size=224,
                                 interpolation='bilinear',
                                 mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225),
                                 preprocess='none'),
    # UNI2 has one output and '' is it. Present anyway: a baseline missing a
    # field the config has is the one mistake in this scheme that costs a
    # recompute, and a module that omits it teaches the omission.
    'head': '',
    # A baseline missing a field the config has is the one mistake in this
    # scheme that costs a recompute; both new fields are here for that reason.
    'pooling': 'cls',
}


@register('uni2')
@dataclass(frozen=True)
class Uni2EncoderConfig(TileEncoderConfig):
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(source='timm', arch=UNI2_ARCH,
                                            dtype='fp16'))

    #: Same reasoning as GigaPath's: torch.compile reorders reductions, worth
    #: about 1e-7, two orders below what fp16 storage in a FeatureStore already
    #: discards. Splitting a cache on it would separate files nobody could tell
    #: apart afterwards.
    compile: bool = False

    NOT_IDENTITY = ('batch_size', 'compile')

    #: Same as GigaPath's: UNI2-h is a bare ViT, so its own answer is the
    #: trunk's CLS and the two spellings name one computation. Note this is the
    #: trunk of a plain timm ViT, not of a CoCa tower -- CONCH is where 'trunk'
    #: stops being an alias and becomes a different vector in a different space.
    HEADS = {'': '', 'trunk': ''}

    #: Which reductions this model can be asked for, and what '' means for it.
    #:
    #: '' -> 'cls' is UNI's own answer, not a default. Slot 0 is tokens[:, 0],
    #: which is the CLS AHEAD of the eight registers rather than an assumption
    #: about where the image content starts.
    #:
    #: The grid family goes further than GigaPath's because the patch grid is
    #: 224/14 = 16x16 rather than 14x14, so 2, 4, 8 and 16 all divide it. Which
    #: grids are legal was always a property of the model; listing them says so
    #: at config time instead of after a gigabyte of weights has loaded.
    POOLINGS = {'': 'cls', 'cls': 'cls',
                'cls_avg': 'cls_avg', 'cls_std': 'cls_std',
                'rings3': 'rings3',
                'grid2x2': 'grid2x2', 'grid4x4': 'grid4x4',
                'grid8x8': 'grid8x8', 'grid16x16': 'grid16x16',
                'tokens': 'tokens'}

    #: The recipe is not a field and does not need to be: `arch` is one, and the
    #: recipe is a function of it. These two properties say so, so that the
    #: encoder and identity_parts both read it OFF THE CONFIG rather than each
    #: re-deriving it -- one place that knows which recipe this config means,
    #: which is what a field would have bought without also inviting a caller to
    #: build a tower the checkpoint does not fit.

    @property
    def recipe(self) -> dict:
        """What this config's arch has to be built with, as written.

        All scalars: layer types appear as the names _LAYER_CLASSES resolves,
        which is what `timm_kwargs` turns into classes. The two are separate
        because identity reads this one and timm reads that one, and mixing
        them would put an import path into a hash.

        Raises rather than falling back to {}. An empty dict builds *a* model
        from the hub config and loads *some* of the weights, and timm's
        pretrained loader filters shape-mismatched keys rather than refusing
        them -- so the failure would be a randomly-initialised tower that
        encodes tiles into plausible-looking vectors.
        """
        try:
            return dict(_TIMM_KWARGS[self.model.arch])
        except KeyError:
            known = ', '.join(sorted(_TIMM_KWARGS))
            raise KeyError(
                f'no build recipe recorded for arch {self.model.arch!r}. This '
                f'module knows: {known}. Add an entry rather than building with '
                f'none -- see the docstring on _TIMM_KWARGS.') from None

    @property
    def timm_kwargs(self) -> dict:
        """`recipe`, with every layer name resolved to its class.

        The only place the classes are reached, and the last step before
        create_model -- so a name that is not in the table fails at build,
        naming the table, rather than reaching timm from a call site.
        """
        kwargs = self.recipe
        for key in _LAYER_KEYS:
            if key in kwargs:
                kwargs[key] = layer_class(kwargs[key])
        return kwargs

    def build(self, device: torch.device, multi_gpu: bool = False):
        return Uni2Encoder(self, device, multi_gpu=multi_gpu)


# ── Encoder ──────────────────────────────────────────────────────────────────

class Uni2Encoder(TileEncoder):
    '''UNI2-h, built as a token producer.

    global_pool='' and num_classes=0 are the two halves of "this is an
    encoder": no classifier, and no reduction the caller did not ask for. They
    are passed at build() rather than carried in the config, on ModelConfig's
    rule -- and stated here rather than left to the hub config, which is the
    mistake that gave `model.head is Linear` the first time a second
    architecture was tried.
    '''

    BASELINE = _UNI2_BASELINE

    def __init__(self, cfg: Uni2EncoderConfig, device: torch.device,
                 multi_gpu: bool = False):
        self.cfg = cfg
        self.device = device
        # .env and the token are resolved at IMPORT, not here -- see the block
        # above `import timm`. Doing it at construction was too late for
        # HF_TOKEN_UNI2 to be visible when that block ran.
        model = cfg.model.build(num_classes=0, global_pool='',
                                **cfg.timm_kwargs)
        model = model.to(device).eval()

        if cfg.compile:
            model = torch.compile(model, mode='reduce-overhead')
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
        '''Config, weights, and the recipe the weights were built into.

        The third is not redundant: three of the twelve kwargs change the
        forward pass without changing a single parameter, so weights_id cannot
        see them. _recipe_parts names which three and why.
        '''
        return super().identity_parts() + _recipe_parts(self.cfg.recipe)
