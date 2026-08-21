"""What every tile encoder in this project has to be, and what it may also be.

    encoder = SomeEncoderConfig().build(device)
    feats   = encoder(tiles)                 # [N, D] -- always
    toks    = encoder.tokens(tiles)          # [N, T, D] -- only some models

A tile encoder turns a list of 256 px images into one vector each. That much is
universal. Everything past it is not, and pretending otherwise is how a base
class stops being usable by the second implementation:

    kind        model output          who
    'vector'    [N, D]                a CNN with num_classes=0
    'tokens'    [N, T, D]             a ViT with global_pool=''
    'spatial'   [N, C, H, W]          a U-Net, or a CNN before its pooling

So `features()` is required and `tokens()` / `spatial()` / `pooled()` are
capabilities. A model that lacks one says so with a sentence naming what it IS,
rather than an AttributeError about embed_dim from four frames down.

What features() returns is the config's to say
----------------------------------------------
    tile -> transform -> trunk -> [head] -> pooling -> normalise -> [N, D]

`self.model` is ALWAYS the trunk. cfg.head names what is attached after it and
cfg.pooling names how what comes out is reduced, so `encoder(patches)` -- the
bare EncodeFn call PatchingLib makes, with nowhere to put an argument -- honours
both. That is why they are fields: before them, a multi-slot pooling could reach
retrieval only by a caller assembling its own FeaturesMap, and nothing recorded
which reduction a store held.

Each config maps '' to its own answer in POOLINGS ('cls' for a ViT, 'gap' for a
feature map, 'identity' where a head already produced the vector). That mapping
is the one thing a base class cannot supply: choose the wrong reduction and
nothing raises, the score is merely worse, and that reads as a property of the
model rather than a mistake.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple

_HERE = Path(__file__).resolve().parent
for _d in (_HERE, _HERE.parent / 'utilities'):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn.functional as F                             # noqa: E402
from PIL import Image                                       # noqa: E402
from torchvision import transforms                          # noqa: E402

from ConfigIdentity import (IdentifiedBuild, IdentifiedConfig,  # noqa: E402
                            config_from,
                            ModelConfig)


_INTERPOLATION = {
    'bicubic': transforms.InterpolationMode.BICUBIC,
    'bilinear': transforms.InterpolationMode.BILINEAR,
    'nearest': transforms.InterpolationMode.NEAREST,
}

KINDS = ('vector', 'tokens', 'spatial')


def _to_pil(img) -> Image.Image:
    return img if isinstance(img, Image.Image) else Image.fromarray(img)


# ── what a model produces ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelOutputSpec:
    """The shape of what a model hands back, read off the model, never assumed.

    `feat_hw` is the spatial extent of the output: the patch grid for 'tokens',
    the feature-map size for 'spatial', None for 'vector'. It is NOT called
    `grid` and not called `token_grid`, for two separate reasons.

    `grid` is already taken. In this project it means the lattice of TILES
    across a slide -- PatchGrid, grid_rc, region_grids -- and that lattice sits
    one scale outside this one. StoreMeta would otherwise carry `grid` and
    `grid_rc` side by side meaning opposite things.

    `token_grid` was the old name and it is wrong twice over: a CNN's output is
    a feature map and has no tokens, and even for a ViT the grid is something
    the MODEL imposes, not a property of the tile -- the tile is only pixels.
    `feat_hw` says what the value is: the H and W of the features. It pairs with
    `dim`, which is their channel axis.

    `num_prefix` is the slice point before the patch tokens -- 1 for a CLS, 5
    for the DINOv2 _reg4 variants, and getting it wrong averages four register
    tokens in as if they were image content, silently, showing up only as a
    slightly worse score. It is 0 for every model with no prefix at all, which
    is what tells pooling_kinds that slot 0 has to be a pooled summary rather than
    tokens[:, 0].
    """
    kind: str
    dim: int
    feat_hw: Optional[Tuple[int, int]] = None
    num_prefix: int = 0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}, got {self.kind!r}')
        if self.kind == 'tokens' and self.feat_hw is None:
            raise ValueError("kind='tokens' needs a patch grid")
        if self.kind == 'vector' and self.feat_hw is not None:
            raise ValueError("kind='vector' has no feat_hw")

    def __getitem__(self, key: str):
        """Field access by name, so one protocol serves both suppliers.

        pooling_kinds is fed from two places: a live encoder (this class) and a
        StoreMeta read back off disk. Before the rename those two spelled the
        same value differently, so every consumer built a three-key dict to
        bridge them -- four such bridges for one architecture, and one more per
        architecture after that. With the names unified, subscripting is all
        either side needs, and the bridges are gone rather than moved.

        It is a method and not a dict base class: inheriting dict would give
        this frozen dataclass a __eq__ and __hash__ that disagree with dict's,
        so it would look like a mapping and not compare like one.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def n_tokens(self) -> int:
        if self.kind != 'tokens':
            raise ValueError(f'{self.kind} output has no token count')
        return self.num_prefix + self.feat_hw[0] * self.feat_hw[1]


@dataclass(frozen=True)
class EncoderOutputSpec:
    """What ONE of an encoder's exits hands back, as ModelOutputSpec says what
    the model hands back.

    One instance describes one exit. features(), tokens() and pooled(mode) are
    three exits and a config normally has exactly one that matters -- more than
    one appears together only in an ablation, which is why this is a shape that
    any exit can be described BY rather than a record that has to cover all of
    them at once.

    `shape` is a ModelOutputSpec and not four re-declared fields. The question
    is the same one -- how to read the axes -- and it does not become a
    different KIND of question because a reduction happened in between. Reusing
    it also reuses its __post_init__, which already refuses kind='tokens'
    without a grid and kind='vector' with one; a second copy of those rules
    would be a second place for them to drift.

    When there IS a slot axis, `shape` describes ONE slot and `slots` describes
    the axis -- the same split StoreMeta uses, where `dim` sits beside `slots`
    and means the width of one of them. Describing the whole tensor instead
    would need a feat_hw for a slot axis that frequently has no grid behind it.
    An unreduced exit has no slot axis, so there `shape` is the model's own spec
    and `slots` is empty.

    The other three name what a tensor cannot say. `pooling` is the reduction's
    name, `slots` names each entry of the slot axis, `slot_layout` says how
    those entries permute under a 90-degree rotation. All three go into
    StoreMeta under the same names (FeatureStore:198-201), so writing a store is
    a copy and not an assembly -- assembly at the call site is what let
    WsiFeaturesMapStore write pooling='cls' for a CNN.

    What is measured and what is declared is not a style choice:

        dim, slot count   measured -- they are axes of a tensor
        pooling, names,   declared -- nothing in a tensor says what a vector
        slot_layout                  means, or where slot 3 sat

    ModelOutputSpec.dim and this one's dim are equal for every reduction that
    selects or averages along one axis (CLS, GAP, rings, grids all keep the
    channel width) and come apart the moment a reduction has weights: CONCH's
    attentional pooler maps 768 to 512. Reading model_spec.dim for the width of
    a feature is right today and wrong then, which is the whole reason this
    class exists.
    """
    shape:       ModelOutputSpec
    pooling:     str
    slots:       Tuple[str, ...] = ()
    slot_layout: str = 'none'

    @property
    def dim(self) -> int:
        """The channel width of one slot. Delegated so callers never have to
        decide between fs.dim and fs.shape.dim -- there is one number."""
        return self.shape.dim


# ── reading a spec off a model, and pooling what it describes ─────────────────
#
# Everything in this section reads a spec and a tensor and nothing else, so it
# serves every token model rather than the one it was written for. It lived in
# GigaPathFunc while GigaPath was the only encoder, where it read as GigaPath's
# pooling; the second ViT is what made the placement wrong, because reaching it
# would have meant importing a sibling implementation -- and with it that
# module's HF_HOME default, its timm import and its Token Merging patcher -- for
# two functions about tensor shapes. GigaPathFunc re-exports these names, so
# every existing importer is unaffected.

def model_token_spec(model: torch.nn.Module) -> dict:
    '''The three numbers any pooling needs, read off a timm ViT, never assumed.

    Named for what it reads and not for who calls it: GigaPath, UNI and UNI2 are
    all timm VisionTransformers and all answer here. A model that is NOT one --
    an image tower behind an attentional pooler, a CNN -- builds its
    ModelOutputSpec another way, and the right way is whatever can be
    observed rather than recalled.

    Unwraps DataParallel first. nn.DataParallel defines no __getattr__ of its
    own, so nn.Module's is used, and that searches only _parameters / _buffers /
    _modules. `embed_dim` and `num_prefix_tokens` are plain ints on the wrapped
    module, and `patch_embed` is one of ITS submodules -- so all three raise
    AttributeError through the wrapper, and build(multi_gpu=True) is the
    default in the benches.

    No defaults, deliberately. `getattr(m, 'num_prefix_tokens', 1)` is right for
    GigaPath and wrong for every model that carries registers -- 5 for the
    DINOv2 _reg4 variants, 9 for UNI2-h, where eight register tokens would be
    averaged in as if they were patches, silently, showing up only as a slightly
    worse score. Crashing is the better failure.
    '''
    m = getattr(model, 'module', model)
    if not isinstance(m.head, torch.nn.Identity):
        raise ValueError(
            f'pooling needs a feature extractor, but model.head is '
            f'{type(m.head).__name__}. Build the model with num_classes=0.')
    return {'dim':        int(m.embed_dim),
            'feat_hw':    tuple(int(v) for v in m.patch_embed.grid_size),
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
    moved once. pooling_kinds passes tokens.device, so ring pooling follows the
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


#: What slot 0 is called, keyed on whether the model has a prefix token.
#: Every mode below puts the tile's single summary vector there, but which
#: vector that is -- and so what to call it -- is the model's property, not the
#: mode's. FeatureStore knows both names: they are the ones that do not count
#: towards a slot_layout's cell count.
SUMMARY_SLOT = {True: 'cls', False: 'gap'}


def pool_slots(mode: str, spec):
    '''(slots, slot_layout) for a pooling mode. NAMES ONLY -- no tensor.

    `slots` names each of the n entries `pooling_kinds` will produce, and
    `slot_layout` says how they permute under a 90-degree rotation. Both go into
    the store's metadata: without slot_layout, a reader matching rotated queries
    against a grid pooling would compare slot (0,1) with slot (1,0) and see only
    a lower score.

    Separate from the reduction because it is a function of the MODE and the
    SPEC and of nothing else -- no tensor is consulted, and none is needed. Two
    things follow. A caller that wants to describe an output it already has does
    not re-run a reduction to learn what to call the parts, which is what the
    *_spec methods do. And pooling_kinds calls this rather than assembling names
    beside its arithmetic, so the two cannot disagree about how many there are.

    `spec` is anything subscriptable by 'dim' / 'feat_hw' / 'num_prefix': a
    ModelOutputSpec off a live encoder, or a StoreMeta off disk.
    '''
    # The two modes a model with no token axis can be asked for. Answered
    # before feat_hw is read, because a 'vector' model has none -- and this is
    # the reason to answer them here rather than in each encoder: one place
    # knows every mode's name, so *_spec never has to branch on kind.
    if mode in ('identity', 'gap'):
        return (mode,), 'none'

    p = int(spec['num_prefix'])
    gh, gw = (int(v) for v in spec['feat_hw'])

    # The NAME of slot 0 follows the model, not the mode. Every mode puts the
    # tile's single summary vector there, but a ViT's is its CLS token and a
    # CNN's is the global average of its feature map -- labelling the second
    # 'cls' would put a word for a token that does not exist into
    # StoreMeta.slots, where it is the only record of what slot 0 holds.
    summary = SUMMARY_SLOT[bool(p)]

    if mode == 'cls':
        return (summary,), 'none'
    if mode == 'cls_avg':
        return (summary, 'avg'), 'none'
    if mode == 'cls_std':
        return (summary, 'std'), 'none'
    if mode.startswith('rings'):
        n_rings = int(mode[5:] or 3)
        return (summary,) + tuple(f'r{k}' for k in range(n_rings)), \
               f'ring:{n_rings}'
    if mode.startswith('grid'):
        bh, bw = (int(v) for v in mode[4:].split('x'))
        return ((summary,) + tuple(f'g{i}{j}' for i in range(bh)
                                   for j in range(bw)), f'grid:{bh}x{bw}')
    if mode == 'tokens':
        return ((summary,) + tuple(f'p{k:03d}' for k in range(gh * gw)),
                f'grid:{gh}x{gw}')
    raise ValueError(f'unknown pooling mode {mode!r}')


def pooling_kinds(tokens: torch.Tensor, mode: str, spec) -> torch.Tensor:
    '''PERFORMS one of the pooling kinds: [N, T, D] tokens -> [N, n, D] slots.

    The name is a category, not a query -- this runs the reduction and hands
    back the tensor. What the n entries are CALLED comes from pool_slots, which
    needs no tensor to answer.

    Every slot is L2-normalized on its own rather than the flattened [n*D]
    vector. Flattening would fix the weight between slots here, at the point
    where nothing records that it happened; a caller that wants one vector
    flattens afterwards and, if it is an encoder, records the choice in its
    config so the id carries it.

    `spec` is anything subscriptable by 'dim' / 'feat_hw' / 'num_prefix': a
    ModelOutputSpec off a live encoder, or a StoreMeta off disk. Those two are
    the only suppliers and they spell the fields the same, so neither side
    builds a dict to get here.
    '''
    p = int(spec['num_prefix'])
    gh, gw = (int(v) for v in spec['feat_hw'])
    n_patch = gh * gw

    if tokens.ndim != 3:
        raise ValueError(f'tokens must be [N, T, D], got {tuple(tokens.shape)}')
    if tokens.shape[1] != p + n_patch:
        raise ValueError(
            f'spec says {p} prefix + {gh}x{gw} patches = {p + n_patch} tokens, '
            f'but tokens have T={tokens.shape[1]}. spec and model disagree.')

    patches = tokens[:, p:]                      # [N, gh*gw, D]

    # Slot 0 is "this model's single vector for the whole tile", and which
    # tensor that is depends on whether the model HAS one. A ViT does: the CLS,
    # at index 0, with any registers between it and the patches. A CNN or a
    # U-Net does not -- its num_prefix is 0, and its summary is the global
    # average, which is also exactly what its features() returns.
    #
    # Reading tokens[:, 0] unconditionally would, for num_prefix == 0, put the
    # TOP-LEFT CELL in slot 0 and label it 'cls'. Nothing downstream would
    # raise: the count check above passes (0 + gh*gw == T), the store validates,
    # the vectors look like vectors. It would surface only as a pooling that
    # scores worse than it should, which reads as a finding.
    cls = tokens[:, 0] if p else patches.mean(1)

    if mode == 'cls':
        parts = [cls]

    elif mode == 'cls_avg':
        # Refused rather than silently duplicated: with no prefix the summary IS
        # patches.mean(1), so the two slots would be the same vector -- and
        # after per-slot normalisation, bit-identical. Every similarity would be
        # exactly doubled and carry no more information than 'cls', which reads
        # as "cls_avg performs like cls" rather than as a broken configuration.
        if not p:
            raise ValueError(
                "'cls_avg' needs a prefix token to be different from 'cls': "
                f'this model has num_prefix=0, so its summary is already '
                f"patches.mean(1) and both slots would be identical. Use 'cls'.")
        parts = [cls, patches.mean(1)]

    elif mode == 'cls_std':
        # Heterogeneity inside the tile: uniform stroma and a tissue boundary
        # have similar means and very different spreads. Rotation-invariant,
        # since it is a statistic over the token set and not over its layout.
        # Safe without a prefix: std is not mean, so the two slots differ.
        parts = [cls, patches.std(1)]

    elif mode.startswith('rings'):
        n_rings = int(mode[5:] or 3)
        bins = _ring_bins(gh, gw, n_rings, tokens.device)
        parts = [cls] + [patches[:, bins == k].mean(1) for k in range(n_rings)]

    elif mode.startswith('grid'):
        bh, bw = (int(v) for v in mode[4:].split('x'))
        if gh % bh or gw % bw:
            raise ValueError(
                f'{gh}x{gw} patch grid does not divide into {bh}x{bw} blocks')
        g = patches.reshape(patches.shape[0], gh, gw, -1)
        parts = [cls]
        for i in range(bh):
            for j in range(bw):
                blk = g[:, i * (gh // bh):(i + 1) * (gh // bh),
                        j * (gw // bw):(j + 1) * (gw // bw), :]
                parts.append(blk.mean(dim=(1, 2)))

    elif mode == 'tokens':
        parts = [cls] + [patches[:, k] for k in range(n_patch)]

    else:
        raise ValueError(f'unknown pooling mode {mode!r}')

    feats = torch.stack(parts, dim=1)            # [N, n, D]

    # Declared against measured, and free: pool_slots named n entries without
    # looking at a tensor, the arithmetic above produced some number of them.
    # They are written by different code, so agreement is evidence -- and the
    # names are what a store keeps, so a mode that grew a part and not a name
    # would put every later slot under the wrong label.
    n_named = len(pool_slots(mode, spec)[0])
    if n_named != feats.shape[1]:
        raise ValueError(
            f'{mode!r}: pool_slots names {n_named} slots, the reduction '
            f'produced {feats.shape[1]}')
    return F.normalize(feats, dim=-1)


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TransformConfig(IdentifiedConfig):
    """The tile -> tensor pipeline, as parameters rather than as code.

    scale_size and crop_size are two numbers rather than timm's single crop_pct,
    because crop_pct is a ratio you have to divide to recover them and the
    division is where the mistakes live. crop_pct() reports it for comparison
    with other people's configs; nothing here consumes it.

    No defaults that mean anything: the numbers belong to whichever model was
    validated with them, so an implementation puts them in ITS baseline. A
    shared shape with a shared zero point would quietly impose one model's
    preprocessing on every other.
    """
    scale_size:    int = 256
    crop_size:     int = 224
    interpolation: str = 'bicubic'
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

    #: 'none' keeps RGB as given. 'grey' is luminance replicated to three
    #: channels, because a patch embedding takes three; colour leaves the
    #: problem entirely, and both sides of a comparison have to use it or the
    #: two are not describing the same thing.
    preprocess: str = 'none'

    def crop_pct(self) -> float:
        return self.crop_size / self.scale_size

    def build(self) -> transforms.Compose:
        if self.preprocess not in ('none', 'grey'):
            raise ValueError(
                f"preprocess must be 'none' or 'grey', got {self.preprocess!r}")
        if self.interpolation not in _INTERPOLATION:
            raise ValueError(
                f'interpolation must be one of {sorted(_INTERPOLATION)}, '
                f'got {self.interpolation!r}')
        steps = [
            transforms.Resize(self.scale_size,
                              interpolation=_INTERPOLATION[self.interpolation]),
            transforms.CenterCrop(self.crop_size),
        ]
        if self.preprocess == 'grey':
            steps.append(transforms.Grayscale(num_output_channels=3))
        steps += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]
        return transforms.Compose(steps)


@dataclass(frozen=True)
class TileEncoderConfig(IdentifiedConfig):
    """Base for every encoder config. Implementations add their own fields.

    batch_size is excluded from identity and that is not a convenience: a ViT
    normalises per sample -- LayerNorm, no cross-sample coupling -- so batching
    cannot change a single vector. It is also the most-tuned knob in the repo,
    and hashing it would throw away every cached result on a throughput
    experiment.

    `head` names which of the model's outputs is the feature. '' is the one its
    authors decided on, and for a bare ViT that is the only one there is. CONCH
    is why the field exists: its image tower is a ViT under an attentional
    pooler, and those are two vectors in two spaces, 768-d and 512-d.

    It is on the base and not on the one config that needs it, because a field
    added to the base LATER cannot be added compatibly -- parts_against emits a
    field the baseline has no entry for ALWAYS, so every id would move. Adding
    it now, with 'head': '' in every baseline, moves nothing.

    It is NOT in NOT_IDENTITY (two heads are two different vectors, usually of
    two different widths) and not in TileEncoder._VARIABLE (it decides what gets
    built, not how it is run).

    `pooling` names the reduction applied after the head. '' is again the
    model's own answer -- 'cls' for a ViT, 'gap' for a CNN. It is a FIELD and
    not an argument to features(), and that is the whole point: what an encoder
    produces has to be a property of the encoder, or `encoder(patches)` cannot
    honour it. Before it existed, a multi-slot pooling could reach retrieval
    only by a caller assembling its own FeaturesMap, because to_features(encoder)
    speaks EncodeFn and EncodeFn has nowhere to put a mode.

    The rule it follows is the one Token Merging was added under and then
    removed under: anything that changes the vectors is a field, because a
    mutation performed after construction leaves nothing for an id to record.
    """
    model:     ModelConfig     = field(default_factory=ModelConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    batch_size: int = 128
    head: str = ''
    pooling: str = ''

    NOT_IDENTITY = ('batch_size',)

    #: Accepted spellings of `head` -> the value this config stores. A model
    #: with one output still lists the aliases that name it, so a sweep over
    #: head='trunk' reaches every encoder for which that is true instead of
    #: raising on the ones that never needed the word.
    #:
    #: No type annotation, and that is load-bearing rather than an omission: an
    #: annotated name with a default in a dataclass body is a FIELD. It would
    #: reach dataclasses.fields(), no baseline would hold an entry for it, and
    #: parts_against emits an absent-from-baseline field always -- re-hashing
    #: every store in result/cache/. It would also be settable per instance,
    #: which would let a caller widen the value domain from the outside.
    #: NOT_IDENTITY above is unannotated for the same two reasons.
    HEADS = {'': ''}

    #: The same table for `pooling`, and closed for the same reason. It looked
    #: at first as though it could not be, because 'rings{n}' and 'grid{H}x{W}'
    #: are parameterised families with no finite spelling -- but the grid family
    #: has to DIVIDE the patch grid (pooling_kinds raises 'does not divide'), so
    #: which grids are legal was always a property of the model. Enumerating
    #: them per class writes that down instead of discovering it on the first
    #: forward, after a gigabyte of weights has loaded.
    POOLINGS = {'': ''}

    def __post_init__(self):
        """Reject an unknown head or pooling; collapse the synonyms of a known one.

        Collapsing is the point, not tidiness. If 'trunk' and '' name the same
        computation on this model and both survived into the config, they would
        produce two encoder_ids over one set of vectors -- two byte-identical
        stores nobody can tell apart afterwards, which is what ConfigIdentity
        exists to prevent. The alias is accepted at the door and does not get
        past it. 'cls' and '' are the same pair for GigaPath.

        Rejecting matters too: head='trunk' handed to a config with no trunk
        head would otherwise be ignored, do nothing, and still be recorded in
        the id as though it had.

        self.HEADS resolves on the instance's actual class, so this one method
        serves every subclass -- the same shape as identity_parts reading
        self.NOT_IDENTITY.
        """
        for name, table in (('head', self.HEADS), ('pooling', self.POOLINGS)):
            value = getattr(self, name)
            try:
                canonical = table[value]
            except KeyError:
                raise ValueError(
                    f'{type(self).__name__} has no {name} {value!r}. Accepts: '
                    f'{", ".join(repr(k) for k in table)}   '
                    f"('' is the model's own answer)") from None
            object.__setattr__(self, name, canonical)  # frozen; the only way in

    def with_model(self, **over) -> 'TileEncoderConfig':
        """A copy with some ModelConfig fields changed.

        `dtype` lives on the nested ModelConfig, so changing it would otherwise
        mean respelling `arch` at every call site -- and a caller who has to
        retype the architecture to switch precision will eventually retype it
        wrong. One source of truth for what the model is, one short way to vary
        the parts that are not the model.

            GigaPathEncoderConfig(batch_size=1024).with_model(dtype='fp32')
        """
        import dataclasses as _dc
        return _dc.replace(self, model=_dc.replace(self.model, **over))

    def build(self, device: torch.device, **kw) -> 'TileEncoder':
        raise NotImplementedError(
            f'{type(self).__name__} must implement build()')


# ── the encoder ───────────────────────────────────────────────────────────────

class TileEncoder(IdentifiedBuild):
    """Base for a tile encoder. Subclasses set cfg, device, model, model_spec.

    Provides the batch loop, the identity surface, `variant`, and features() for
    the two kinds whose reduction is arithmetic. A token model must say which
    token, because nothing here can know.
    """

    #: Fields a variant may change. Everything else decides what gets BUILT, so
    #: changing one has to go through a real construction or the object would
    #: describe a model it does not have.
    _VARIABLE = ('dtype', 'batch_size', 'transform')

    cfg: TileEncoderConfig
    model_spec: ModelOutputSpec

    # ── capability gates ─────────────────────────────────────────────────────

    def _require(self, kind: str, what: str) -> None:
        if self.model_spec.kind != kind:
            raise TypeError(
                f'{what} needs a {kind!r} model and this one is '
                f'{self.model_spec.kind!r} (dim {self.model_spec.dim}'
                + (f', feat_hw {self.model_spec.feat_hw}'
                   if self.model_spec.feat_hw else '')
                + f'). features() works for every kind.')

    # ── encoding ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run(self, images, reduce: Optional[Callable]) -> torch.Tensor:
        """One batch loop. `reduce` decides what crosses to the host.

        Order is fixed here and not left to the callback: .float() runs BEFORE
        reduce. Under autocast the output arrives fp16, and a reduction means
        .mean() or .std() over a couple of hundred vectors -- in fp16 that
        quietly degrades the averaged parts while leaving a single selected
        token exactly right. The symptom is "averaging poolings do not help",
        which reads as a result rather than an error.

        reduce=None moves everything to the host, which is the only safe
        default: [N, 197, 1536] fp32 is 1.21 MB per tile, so 68k tiles would be
        82 GB held on the device. A caller who wants it to stay there passes
        reduce=lambda t: t and owns that decision. The choice can only be made
        HERE -- by the time _run returns, torch.cat has already built the whole
        thing wherever it was going to live.
        """
        dtype = self.cfg.model.torch_dtype()
        ctx = (torch.autocast(device_type=self.device.type, dtype=dtype)
               if dtype is not torch.float32 else nullcontext())
        out = []
        for start in range(0, len(images), self.cfg.batch_size):
            batch = torch.stack([
                self._transform(_to_pil(img))
                for img in images[start:start + self.cfg.batch_size]
            ]).to(self.device)
            with ctx:
                raw = self.model(batch)
            raw = raw.float()
            out.append(raw.cpu() if reduce is None else reduce(raw))
        return torch.cat(out, dim=0)

    @property
    def feature_pooling(self) -> str:
        """What features() reduced by: 'cls', 'gap', 'rings3', 'identity', ...

        One line, and it used to be five. The four it replaces read
        model_spec.kind to decide what '' meant, which is the same mistake _pool
        had: consulting the MODEL's shape to describe a reduction that may have
        happened after a head. POOLINGS resolves '' now -- at construction, in
        a table, per config -- so by the time anything asks, cfg.pooling names
        a concrete mode and there is nothing left to infer.

        A store of features() output is labelled with this rather than with a
        literal, because the caller writing the label takes any encoder while
        the label is only ever true of one. WsiFeaturesMapStore used to write
        pooling='cls' for whatever it was handed -- correct for GigaPath, and a
        false claim about every CNN, whose features() is a global average.

        It is also what _vector_from reduces BY, which is the point: the label
        and the arithmetic read the same attribute, so a store cannot say one
        thing while holding another.
        """
        if not self.cfg.pooling:
            raise NotImplementedError(
                f"{type(self.cfg).__name__} has not said what '' means for it. "
                f"Map '' to a concrete mode in its POOLINGS -- 'cls' for a ViT "
                f"whose answer is the CLS, 'gap' for a feature map, 'identity' "
                f"where a head already produced the vector. Guessing here would "
                f"answer a question only the model's authors can.")
        return self.cfg.pooling

    # ── head and pooling ─────────────────────────────────────────────────────

    def _apply_head(self, raw: torch.Tensor) -> torch.Tensor:
        """cfg.head, applied to the trunk's output. Base: there is no head.

        self.model is ALWAYS the trunk, never the trunk with something bolted
        on, so that tokens() and spatial() can hand back what the model itself
        produced. Anything after it lives here, and runs on the device: _run
        hands this a device tensor and only the result of the reduce crosses to
        the host.
        """
        return raw

    def _pool(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """[B, ...] -> [B, n, D] slots, each L2-normalised. Runs on the device.

        A step, not an exit. pooled() is the exit and calls this inside the
        batch loop; _vector_from calls it in the same place. The distinction
        matters because _vector_from is handed a TENSOR and has no images to
        start a second batch loop with.

        It branches on the RANK of what it was handed, and not on
        model_spec.kind. That is the whole difference between this version and
        the one before it. model_spec describes the MODEL's output; by the time
        this runs, _apply_head may have replaced it with something else --
        CONCH's attentional pooler turns [B, 785, 768] into [B, 512], so a
        'tokens' kind would send a two-dimensional tensor into pooling_kinds and
        get told it is not [N, T, D]. Reading the tensor cannot go stale that
        way, and it is the same rule the rest of this module now follows:
        measure what is in hand, do not consult a claim about something else.

        Rank settles it completely:

            2   already one vector per tile -- a head reduced it, or the model
                never had an axis to reduce. Nothing left to pool.
            3   [B, T, D] channel-last, which is what pooling_kinds reads.
            4   [B, C, H, W] channel-first -- a feature map. Permuted rather
                than refused, which hands rings and grids to a CNN, where they
                mean exactly what they mean for a ViT.

        `spec` is still model_spec, and that is the part this does not fix: a
        head that RESHAPES without collapsing -- [B, T, D] -> [B, T', D'] --
        would leave num_prefix and feat_hw describing the wrong grid, and
        pooling_kinds' token-count check is what would catch it. No such head
        exists here yet. When one does, it has to hand out its own spec, and
        this is the line that will need it.

        Override to add a pooling this encoder alone has.
        """
        if x.ndim == 2:
            if mode not in ('identity', 'gap'):
                raise ValueError(
                    f'{type(self).__name__} was handed one vector per tile '
                    f'({tuple(x.shape)}), so there is nothing for {mode!r} to '
                    f"reduce. Legal here: 'identity'.")
            return F.normalize(x, dim=-1).unsqueeze(1)
        if x.ndim == 4:
            if mode == 'gap':
                return F.normalize(x.mean(dim=(-2, -1)), dim=-1).unsqueeze(1)
            # [B, C, H, W] -> [B, H*W, C]. num_prefix is 0 for a feature map, so
            # slot 0 comes out as the global average and is named 'gap' rather
            # than 'cls' -- pooling_kinds and SUMMARY_SLOT already do that,
            # without a branch here.
            x = x.flatten(2).transpose(1, 2)
        return pooling_kinds(x, mode, self.model_spec)

    def _vector_from(self, raw: torch.Tensor) -> torch.Tensor:
        """One [B, ...] batch of model output -> [B, D], L2-normalised.

        head, then pooling, then one vector. Not a branch in sight: which head
        and which pooling are both cfg, and every encoder in the project now
        goes through this same line. It used to be an override point and the
        overrides were all the same three lines with one word changed, which is
        how GigaPath's version came to ignore cfg.pooling for a revision.

        What each config still has to say is what '' means for it -- 'cls' for
        a ViT whose answer is the CLS, 'gap' for a feature map, 'identity' where
        a head already produced the vector. That is one entry in POOLINGS, and
        it is a NAME rather than arithmetic, which is the only part a base class
        genuinely cannot know: pick the wrong token out of 197 and nothing
        fails, the retrieval score is merely worse, and that reads as a property
        of the model.

        The flatten is where n slots become one vector, and it fixes their
        relative weight at equal. That is concat_slots' recipe, so the cosine
        of the result is the mean of the per-slot cosines. Weighting them
        differently is a third axis and nothing in the repo opens it; when
        something does, it is a config field, because it changes the vectors and
        the id has to carry it.
        """
        return F.normalize(
            self._pool(self._apply_head(raw), self.feature_pooling).flatten(1),
            dim=-1)

    # ── exits: five ways out, each handing back ONE tensor ───────────────────
    #
    # One tensor and not (tensor, spec) so that __call__ keeps satisfying
    # EncodeFn -- a bare Callable[[List], Any] that PatchingLib, the mpp
    # estimator and several plain functions in the tests all speak. The
    # description is available from the *_spec method beside each exit, which
    # reads the tensor the caller already has and needs no second forward.

    def features(self, images) -> torch.Tensor:
        """[N, D] L2-normalised, reduced the way cfg says. Every encoder has it.

        The reduction runs inside the batch loop, so what crosses to the host is
        a few KB per tile rather than the full output.
        """
        return self._run(images, lambda t: self._vector_from(t).cpu())

    def tokens(self, images, reduce: Optional[Callable] = None) -> torch.Tensor:
        """[N, T, D] fp32, NOT normalised, NO head. Token models only.

        The trunk's own output: self.model is always the trunk, so nothing here
        has been through cfg.head or cfg.pooling. That is what this exit is for.

        Normalising is left to whoever pools: mean-then-normalise and
        normalise-then-mean are different vectors, and only the first is what a
        pooled descriptor means.
        """
        self._require('tokens', 'tokens()')
        return self._run(images, reduce)

    def spatial(self, images, reduce: Optional[Callable] = None) -> torch.Tensor:
        """[N, C, H, W] fp32, NO head. Spatial models only."""
        self._require('spatial', 'spatial()')
        return self._run(images, reduce)

    def pooled(self, images, mode: str) -> torch.Tensor:
        """[N, n, D] slots for one reduction, chosen HERE rather than by cfg.

        The sweep tool. One config is one arm of an experiment and gets one
        encoder_id, which is what production wants; a bench comparing five
        poolings would otherwise load the weights five times to vary a string.

        What the n entries are called comes from pool_slots(mode, model_spec) --
        no tensor needed, so a caller writing a store does not run the reduction
        twice to learn the names.
        """
        return self._run(images,
                         lambda t: self._pool(self._apply_head(t), mode).cpu())

    def __call__(self, images) -> torch.Tensor:
        """features(), under the name EncodeFn expects.

        PatchingLib's FeaturesMap.from_patch_container, GigaPathKnnEstiMpp and
        several plain functions in the tests all speak `Callable[[List], Any]`,
        which has nowhere to put a mode and nothing to receive a second return
        value. That is exactly why pooling is a CONFIG field: this call honours
        it without the protocol having to know it exists.
        """
        return self.features(images)

    # ── describing what an exit handed back ──────────────────────────────────

    def _slot_spec(self, t: torch.Tensor, pooling: str,
                   slots, layout: str) -> EncoderOutputSpec:
        """Shared tail: check the slot count, then describe.

        `shape` describes ONE slot, which is how StoreMeta reads its own dim
        beside its own slots (FeatureStore:198-201). The alternative -- shape
        describing the whole tensor -- would need a feat_hw for a slot axis that
        often has no grid behind it at all.
        """
        n = 1 if t.ndim == 2 else t.shape[1]
        if len(slots) != n:
            raise ValueError(
                f'{type(self).__name__}: {pooling!r} names {len(slots)} slots '
                f'{slots} but the tensor carries {n} ({tuple(t.shape)})')
        # The last axis IS one slot's width, in both layouts, which is why there
        # is no arithmetic here. pooled() hands back [N, n, D] with the slot
        # axis intact; features() hands back [N, n*D] but names ONE slot, so n
        # is 1 and the flattened width is that slot. A `// n` looks right for
        # the second and is wrong for the first -- it shipped for one revision
        # and turned pooled()'s D=6 into 1.
        return EncoderOutputSpec(
            shape=ModelOutputSpec('vector', int(t.shape[-1])),
            pooling=pooling, slots=tuple(slots), slot_layout=layout)

    def feature_spec(self, features: torch.Tensor) -> EncoderOutputSpec:
        """Describe what features() returned. One slot, whatever cfg.pooling was.

        ONE slot even for a multi-slot pooling, because _vector_from flattened
        them: the file holds one vector per tile and `pooling` is the only
        record of what went into it. pool_slots(pooling, model_spec) recovers
        the internal structure for anyone who wants it.

        Takes the tensor rather than deriving the width from cfg. Deriving it
        would mean n_slots * model_spec.dim, which holds only while the
        reduction keeps the channel axis -- true of every selection and average,
        false the moment a head has weights.
        """
        name = self.feature_pooling
        return self._slot_spec(features, name, (name,), 'none')

    def pooled_spec(self, pooled: torch.Tensor, mode: str) -> EncoderOutputSpec:
        """Describe what pooled(images, mode) returned."""
        slots, layout = pool_slots(mode, self.model_spec)
        return self._slot_spec(pooled, mode, slots, layout)

    def tokens_spec(self, tokens: torch.Tensor) -> EncoderOutputSpec:
        """Describe what tokens() returned: the model's own output, unreduced.

        `slots` is empty and that is not an omission. A slot name says what one
        entry of a REDUCTION holds, and nothing has been reduced -- naming these
        entries is the reducer's job, done differently per mode and not always
        in the model's order (pooling_kinds' 'tokens' mode drops UNI2's eight
        registers). model_spec already says kind, dim, feat_hw and num_prefix,
        which is everything there is to say about an output nobody has touched.
        """
        self._require('tokens', 'tokens_spec()')
        if tokens.shape[1] != self.model_spec.n_tokens():
            raise ValueError(
                f'{type(self).__name__}: model_spec says T='
                f'{self.model_spec.n_tokens()}, tokens() produced '
                f'{tokens.shape[1]}')
        return EncoderOutputSpec(shape=self.model_spec, pooling='tokens')

    def spatial_spec(self, spatial: torch.Tensor) -> EncoderOutputSpec:
        """Describe what spatial() returned: the model's own feature map."""
        self._require('spatial', 'spatial_spec()')
        want = self.model_spec.feat_hw
        if want is not None and tuple(spatial.shape[2:]) != tuple(want):
            raise ValueError(
                f'{type(self).__name__}: model_spec says feat_hw {tuple(want)}, '
                f'spatial() produced {tuple(spatial.shape[2:])}')
        return EncoderOutputSpec(shape=self.model_spec, pooling='spatial')

    # ── variants ─────────────────────────────────────────────────────────────

    def variant(self, **over) -> 'TileEncoder':
        """A second encoder over the SAME loaded model, configured differently.

        Sweeps are what this is for: batch size x precision, or two
        preprocessing arms. Building a second config would reload the weights to
        change a number, which in a throughput benchmark is most of what is
        being measured.

        Only _VARIABLE may change; anything else raises rather than handing back
        an encoder whose config disagrees with its weights. `dtype` lives on the
        nested ModelConfig, so it is rewritten there.
        """
        import dataclasses as _dc
        bad = set(over) - set(self._VARIABLE)
        if bad:
            raise ValueError(
                f'variant() cannot change {sorted(bad)} -- those decide what is '
                f'built. Construct a config instead. Changeable: '
                f'{list(self._VARIABLE)}')

        cfg = self.cfg
        if 'dtype' in over:
            cfg = _dc.replace(cfg, model=_dc.replace(cfg.model,
                                                     dtype=over.pop('dtype')))
        cfg = _dc.replace(cfg, **over) if over else cfg

        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.cfg = cfg
        if 'transform' in over:
            clone._transform = cfg.transform.build()
        return clone


# ── choosing an implementation by name ────────────────────────────────────────

#: Registered name -> the module whose import registers it. ConfigIdentity's
#: registry fills by import side effect, so config_from('uni2') on its own says
#: "not registered" for a module nobody imported -- the failure its docstring
#: names. This table is what turns a --encoder flag into the right import.
#:
#: They are imported ONE AT A TIME, and that is not laziness. Every
#: implementation module sets os.environ.setdefault('HF_HOME', ...) above its
#: own `import timm`, and huggingface_hub freezes HF_HOME into module constants
#: when IT is imported. setdefault is first-one-wins, so importing all three
#: would let whichever landed first decide where the other two look for weights:
#: CONCH's checkpoint downloaded into prov-gigapath/model_weights, several
#: gigabytes re-fetched, and a directory name that no longer says what is in it.
#: Nothing raises. Importing only the one asked for is the whole fix.
_IMPLEMENTATIONS = {
    'gigapath':  'GigaPathFunc',
    'uni2':      'Uni2Func',
    'conch_vit': 'ConchVitFunc',
}


def encoder_names() -> list:
    """The names --encoder accepts. For argparse `choices`."""
    return sorted(_IMPLEMENTATIONS)


def encoder_config(name: str, **over) -> 'TileEncoderConfig':
    """The config registered as `name`, with its module imported first.

    Raises on an unknown name with the known ones listed, rather than on the
    ImportError or the empty registry that would otherwise follow.
    """
    import importlib
    try:
        module = _IMPLEMENTATIONS[name]
    except KeyError:
        raise KeyError(
            f'no encoder called {name!r}. Known: '
            f'{", ".join(encoder_names())}') from None
    importlib.import_module(module)
    return config_from(name, **over)


def admissible_poolings(cfg, wanted) -> tuple:
    """(kept, dropped) out of `wanted`, by what this config admits.

    A sweep names the arms it wants compared; which of them a given model can
    actually do is the MODEL's property. GigaPath's 14x14 patch grid divides by
    7 and UNI2's 16x16 does not, so 'grid7x7' is an arm for one and not the
    other -- and pooling_kinds only says so after the weights have loaded.

    Returns the dropped ones rather than filtering silently, because a table
    with four arms where five were asked for still reads as a complete
    comparison unless something says otherwise.
    """
    keep = [m for m in wanted if m in cfg.POOLINGS]
    drop = [m for m in wanted if m not in cfg.POOLINGS]
    return keep, drop
