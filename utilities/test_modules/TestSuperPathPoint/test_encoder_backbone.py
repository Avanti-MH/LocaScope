#!/usr/bin/env python3
"""Tests for training/SuperPathPoint/SuperPoint/EncoderBackbone.py. spec.md 5.3.

    python utilities/test_modules/test_encoder_backbone.py
    python utilities/test_modules/test_encoder_backbone.py --with-model
    python utilities/test_modules/test_encoder_backbone.py --only refusals

NO WEIGHTS unless --with-model. The trunk here is fifteen lines of fake: a conv
with a stride, wearing the three attributes the backbone reads off a timm ViT
(`patch_embed.patch_size`, `dynamic_img_size`, `embed_dim`). That is enough to
test everything this file decides, because everything it decides is about the
NUMBERS -- and a fake trunk can lie about them on purpose, which the real one
cannot be made to do.

WHAT WOULD RUN AND BE WRONG
-----------------------------
1. `stride` taken from `model_spec.feat_hw` instead of from `patch_size`. That
   is `crop_size // patch_size` -- the grid at INFERENCE, on a 224 px centre
   crop -- and this backbone deliberately does not centre-crop. At crop 224,
   patch 16, tile 256 the two are 14 and 16: the cell grid comes out a
   different size than the labels were splatted onto, and nothing raises.
2. The normalisation dropped, or the geometry kept. `TransformConfig.build()`
   is `Resize -> CenterCrop -> ToTensor -> Normalize`; the first two must NOT
   happen here and the last two must. Keeping the resize trains on a 224 px
   view of a 256 px tile whose labels are at 256; dropping the Normalize feeds
   a distribution the trunk never saw. Both train.
3. The trunk registered as an `nn.Module` child. Then `parameters()` hands the
   optimiser 1.1B frozen tensors and every checkpoint carries them.
4. A tile size the patch grid does not divide. That is the reason `tile_size`
   is a config field at all, and it is refused at construction rather than at
   the first forward of a training run. A model whose input size is FIXED is
   not that -- it is one `set_input_size` call, and skipping it looks identical
   on a `dynamic_img_size` model and asserts deep inside timm on the other.

Sections:
  1. adopt     -- SpatialTrunk.over: the MRO, the shared weights, the refusal
  2. shapes    -- stride and width, against the inference-grid decoy
  3. refusals  -- what cannot be built, and why each one is silent otherwise
  4. forward   -- normalisation, the frozen trunk, and where the gradient goes
  5. identity  -- what moves the hash
  6. model     -- --with-model only: a real encoder at a real tile size
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field, replace

# `utilities/` holds the one definition of the output roots (`_paths.py`), and
# `setup_import_paths` -- which puts every other package on the path -- is
# inside it, so it has to be reachable before anything else is imported.
#
# BOTH parents are inserted, and that is deliberate rather than sloppy: this
# file runs from `utilities/test_modules/` and from
# `utilities/test_modules/TestSuperPathPoint/`, one level deeper, and inserting
# both means the move needs no edit here. The one that is not `utilities/` is
# either the repo root or `test_modules/`; neither holds a `_paths.py`, and
# `setup_import_paths` puts the repo root on the path anyway.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import torch                                                     # noqa: E402
import torch.nn as nn                                            # noqa: E402

from ConfigIdentity import ModelConfig, register                 # noqa: E402
from common.Interfaces import Backbone, ShapeMismatch            # noqa: E402
from SuperPoint.EncoderBackbone import (SpatialTrunk,            # noqa: E402
                                        TileEncoderBackbone,
                                        TileEncoderBackboneConfig,
                                        _TILE_ENCODER_BASELINE)
from TileEncoderFunc import (ModelOutputSpec, TileEncoder,       # noqa: E402
                             TileEncoderConfig, TransformConfig)

_RESULTS = []

TILE = 256
PATCH = 16
DIM = 64            # a real trunk is 768 or 1536; the arithmetic is the same
CROP = 224          # so crop // patch = 14 and tile // patch = 16 differ


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ══════════════════════════════════════════════════════════════════════════════
#  A trunk that can lie
# ══════════════════════════════════════════════════════════════════════════════

class _FakeVit(nn.Module):
    """What `TileEncoderBackbone` reads off a timm ViT, plus a real conv.

    Three of the four knobs exist to make it LIE, which is the whole reason a
    fake is the stronger test here:

        conv_stride  a declared stride that is not the real one. The one
                     failure `check_shapes` exists for, and the one a real ViT
                     cannot be talked into.
        dynamic      False reproduces `PatchEmbed`'s `strict_img_size` assert
                     (`timm/layers/patch_embed.py:120`) -- the guard that makes
                     prov-gigapath refuse anything but 224 until it is told
                     otherwise.
        resizable    False is a trunk with no `set_input_size` at all.
    """

    def __init__(self, dim=DIM, patch=PATCH, dynamic=True, conv_stride=None,
                 resizable=True):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.patch_size = (patch, patch)
        self.dynamic_img_size = dynamic
        self.img_size = (CROP, CROP)
        self.resized_to = None
        self.embed_dim = int(dim)
        self.num_prefix_tokens = 1
        self.head = nn.Identity()
        s = int(conv_stride or patch)
        self.proj = nn.Conv2d(3, int(dim), kernel_size=s, stride=s)
        if not resizable:
            # getattr(model, 'set_input_size', None) -> None, which is a trunk
            # that is not a timm ViT. An instance attribute shadows the method.
            self.set_input_size = None

    def set_input_size(self, img_size=None, patch_size=None):
        if img_size is not None:
            self.img_size = (int(img_size[0]), int(img_size[1]))
            self.resized_to = self.img_size

    def forward(self, x):
        if not self.dynamic_img_size:
            h, w = x.shape[-2:]
            if (h, w) != self.img_size:
                raise AssertionError(
                    f"Input height ({h}) doesn't match model "
                    f"({self.img_size[0]}).")
        return self.proj(x)


@register('fake_trunk')
@dataclass(frozen=True)
class _FakeEncoderConfig(TileEncoderConfig):
    method: str = 'fake_trunk'
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(source='local', arch='fake',
                                            dtype='fp32'))
    transform: TransformConfig = field(
        default_factory=lambda: TransformConfig(scale_size=CROP, crop_size=CROP))

    def build(self, device=None, **kw) -> '_FakeEncoder':
        return _FakeEncoder(self, torch.device(device or 'cpu'), **kw)


class _FakeEncoder(TileEncoder):
    """A `TileEncoder` in the two places that matter: it answers
    `_compute_model_spec` and `_spatial_forward`, which are the two the base
    refuses. Everything `SpatialTrunk.over` needs is here and nothing else is.
    """

    def __init__(self, cfg, device, **kw):
        self.cfg = cfg
        self.device = device
        self.model = _FakeVit(**kw).to(device).eval()
        self._transform = cfg.transform.build()

    def _compute_model_spec(self) -> ModelOutputSpec:
        crop = int(self.cfg.transform.crop_size)
        patch = int(self.model.patch_embed.patch_size[0])
        return ModelOutputSpec(kind='tokens', dim=int(self.model.embed_dim),
                               feat_hw=(crop // patch, crop // patch),
                               num_prefix=1)

    def _spatial_forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)


def _trunk(**kw) -> SpatialTrunk:
    preprocess = kw.pop('preprocess', 'none')
    cfg = _FakeEncoderConfig()
    cfg = replace(cfg, transform=replace(cfg.transform, preprocess=preprocess))
    return SpatialTrunk.over(cfg.build('cpu', **kw))


def _backbone(cfg_over=None, **kw) -> TileEncoderBackbone:
    over = dict(encoder='fake_trunk', tile_size=TILE)
    over.update(cfg_over or {})
    trunk = _trunk(preprocess=over.get('preprocess', 'none'), **kw)
    return TileEncoderBackbone(TileEncoderBackboneConfig(**over), trunk)


# ══════════════════════════════════════════════════════════════════════════════
#  1. adopt
# ══════════════════════════════════════════════════════════════════════════════

def t_over_reparents_without_rebuilding_the_model():
    """The whole reason `over()` exists rather than a second `build()`.

    A foundation trunk is gigabytes. Building it twice to change its class is
    the cost this replaces, and the check is object identity: the SAME model
    and the SAME config, under a class that now has `trunk_forward`.
    """
    enc = _FakeEncoderConfig().build('cpu')
    trunk = SpatialTrunk.over(enc)
    assert trunk.model is enc.model, 'the weights were rebuilt'
    assert trunk.cfg is enc.cfg
    assert isinstance(trunk, SpatialTrunk)
    assert isinstance(trunk, _FakeEncoder), \
        'the concrete class left the MRO; _spatial_forward would raise'
    assert isinstance(trunk, TileEncoder)
    return f'MRO {" -> ".join(c.__name__ for c in type(trunk).__mro__[:4])}'


def t_over_resolves_the_concrete_spatial_forward_and_not_the_base_refusal():
    """`SpatialTrunk(TileEncoder)` alone inherits the base's `_spatial_forward`,
    which raises `NotImplementedError` naming the subclass. That is the decoy:
    if the dynamic class were dropped and `SpatialTrunk` used directly, this is
    what would happen instead of a feature map.
    """
    trunk = _trunk()
    out = trunk.trunk_forward(torch.zeros(1, 3, TILE, TILE))
    assert out.shape == (1, DIM, TILE // PATCH, TILE // PATCH), out.shape
    assert out.dtype is torch.float32, 'trunk_forward must hand back fp32'

    bare = object.__new__(SpatialTrunk)
    bare.__dict__.update(trunk.__dict__)
    try:
        bare.trunk_forward(torch.zeros(1, 3, TILE, TILE))
    except NotImplementedError:
        pass
    else:
        raise AssertionError(
            'a bare SpatialTrunk produced a map; then the dynamic class in '
            'over() is doing nothing and the concrete encoder is not in the MRO')


def t_over_refuses_something_that_is_not_an_encoder():
    for bad in (_FakeVit(), object(), None):
        try:
            SpatialTrunk.over(bad)
        except TypeError:
            continue
        raise AssertionError(f'over() accepted {type(bad).__name__}')


# ══════════════════════════════════════════════════════════════════════════════
#  2. shapes
# ══════════════════════════════════════════════════════════════════════════════

def t_stride_is_the_patch_and_not_the_inference_grid():
    """THE ONE THIS FILE EXISTS FOR.

    `model_spec.feat_hw` is `crop_size // patch_size` = 224 // 16 = 14: the
    grid the encoder produces at inference, after a resize and a centre crop
    this backbone deliberately does not do. The real stride is the patch, 16,
    and 256 // 16 = 16 cells.

    Both numbers are plausible, both are self-consistent, and using the wrong
    one makes the cell grid a different size than the labels were splatted
    onto. Nothing raises -- which is why the decoy is spelled out here rather
    than left implicit.
    """
    b = _backbone()
    assert b.stride == PATCH, b.stride
    assert b.out_channels == DIM, b.out_channels

    decoy = b.trunk.model_spec.feat_hw[0]
    assert decoy == CROP // PATCH == 14
    assert TILE // b.stride == 16
    assert TILE // b.stride != decoy, \
        'the inference grid and the training grid are the same number here, ' \
        'so this check proves nothing -- pick a CROP that is not the TILE'

    out = b(torch.zeros(1, 3, TILE, TILE))
    assert out.shape == (1, DIM, 16, 16), out.shape
    return f'stride {b.stride}, grid {out.shape[-1]}, inference grid {decoy}'


def t_the_backbone_satisfies_the_protocol_and_its_own_check():
    """`check_shapes` already ran inside `__init__`; this is the protocol half.

    `Backbone` is `runtime_checkable`, so isinstance only checks that the three
    names exist -- which is exactly why `check_shapes` exists and why the
    construction above is the real assertion.
    """
    b = _backbone()
    assert isinstance(b, Backbone)
    assert b.trainable is False


def t_a_lying_stride_is_caught_at_construction():
    """The trunk declares patch 16 and strides 8. The map is then twice as wide
    as the declared stride says, every keypoint is off by a factor of two, and
    the only thing that would ever notice is this.
    """
    try:
        _backbone(conv_stride=PATCH // 2)
    except ShapeMismatch as e:
        assert 'stride' in str(e)
        return str(e).splitlines()[0][:70]
    raise AssertionError('a backbone whose real stride is half its declared '
                         'one was built without complaint')


def t_a_lying_width_is_caught_at_construction():
    """`out_channels` comes from `model_spec.dim` and the tensor comes from the
    conv. Making them disagree needs a spec that lies, which is the fake's
    other job.
    """
    trunk = _trunk()
    trunk._compute_model_spec = lambda: ModelOutputSpec(
        kind='tokens', dim=DIM * 2, feat_hw=(14, 14), num_prefix=1)
    try:
        TileEncoderBackbone(
            TileEncoderBackboneConfig(encoder='fake_trunk', tile_size=TILE),
            trunk)
    except ShapeMismatch:
        return 'caught'
    raise AssertionError('out_channels and the tensor disagreed silently')


# ══════════════════════════════════════════════════════════════════════════════
#  3. refusals
# ══════════════════════════════════════════════════════════════════════════════

def t_a_tile_the_patch_does_not_divide_is_refused():
    """UNI2 is patch 14 and 256 % 14 = 4.

    The refusal names the multiples of the PATCH and stops there, because a
    backbone does not know the cell. An earlier version computed
    `lcm(patch, 8)` and told the reader to use multiples of 56 -- which bakes
    cell 8 into a class that has no business assuming it, and is wrong the
    moment uni2 takes cell 7 and every multiple of 14 becomes legal.

    So this checks two things: the patch multiples ARE named, and the cell
    constraint is handed to the decoder rather than decided here.
    """
    try:
        _backbone(patch=14)
    except ValueError as e:
        msg = str(e)
        assert '14' in msg and '256' in msg
        assert '252' in msg and '266' in msg, \
            f'the refusal must name the nearest tiles the patch divides: {msg}'
        # NOT `'56' not in msg`: the message says "tile_size is 256", and
        # '56' is a substring of '256'. The first version of this check failed
        # for exactly that, which is the argument against matching on bare
        # numbers -- assert the PHRASE the wrong version would have used.
        assert 'multiples of 56' not in msg and 'lcm' not in msg, \
            f'the refusal is assuming cell 8; a backbone does not know the ' \
            f'cell: {msg}'
        assert 'multiples of 14' in msg, \
            f'the refusal must name the patch it does know: {msg}'
        assert 'UpsampleDecoderConfig' in msg, \
            f'the refusal must hand the cell constraint to the decoder: {msg}'
        return '252 / 266 named, cell left to the decoder'
    raise AssertionError('a 256 px tile on a patch-14 trunk was accepted')


def t_a_fixed_size_trunk_is_resized_rather_than_refused():
    """prov-gigapath is built with no `dynamic_img_size` and its
    `pretrained_cfg` says `fixed_input_size: true`, so `PatchEmbed.forward`
    asserts the input is exactly 224. That is a property of how the model was
    CONSTRUCTED, not of the weights, and one public timm call changes it.

    The decoy is the same trunk without the call: it must assert at 256. That
    is what makes the passing case evidence rather than a coincidence -- a
    backbone that silently skipped `set_input_size` would look identical on a
    `dynamic_img_size` model and only fail on this one.
    """
    raw = _FakeVit(dynamic=False)
    try:
        raw(torch.zeros(1, 3, TILE, TILE))
    except AssertionError:
        pass
    else:
        raise AssertionError('the fake is not reproducing strict_img_size, so '
                             'this check cannot fail for the right reason')

    b = _backbone(dynamic=False)
    assert b.trunk.model.resized_to == (TILE, TILE), \
        'a fixed-size trunk was accepted without being told its new size'
    assert b(torch.zeros(1, 3, TILE, TILE)).shape[-1] == TILE // PATCH

    # A dynamic trunk gets the same call, so the two paths agree by
    # construction: the resample runs once here instead of on every forward.
    assert _backbone().trunk.model.resized_to == (TILE, TILE)


def t_a_trunk_that_cannot_be_resized_is_refused():
    try:
        _backbone(resizable=False)
    except TypeError as e:
        assert 'set_input_size' in str(e)
        return 'refused'
    raise AssertionError('a trunk with no set_input_size was accepted; on a '
                         'fixed-size model that asserts deep inside timm')


def t_a_non_square_patch_is_refused():
    trunk = _trunk()
    trunk.model.patch_embed.patch_size = (16, 8)
    try:
        TileEncoderBackbone(
            TileEncoderBackboneConfig(encoder='fake_trunk', tile_size=TILE),
            trunk)
    except ValueError as e:
        assert 'one number' in str(e)
        return 'refused'
    raise AssertionError('a non-square patch became a single stride')


def t_preprocess_must_agree_with_the_trunk_it_was_built_over():
    """The identity records `cfg.preprocess`; the forward does whatever the
    trunk's transform says. Two names for one fact.
    """
    try:
        TileEncoderBackbone(
            TileEncoderBackboneConfig(encoder='fake_trunk', tile_size=TILE,
                                      preprocess='grey'),
            _trunk(preprocess='none'))
    except ValueError as e:
        assert 'preprocess' in str(e)
    else:
        raise AssertionError('the identity and the forward were allowed to '
                             'disagree about the preprocessing')
    # and the consistent pair builds
    TileEncoderBackbone(
        TileEncoderBackboneConfig(encoder='fake_trunk', tile_size=TILE,
                                  preprocess='grey'),
        _trunk(preprocess='grey'))


def t_an_unknown_encoder_name_is_refused_before_anything_loads():
    try:
        TileEncoderBackboneConfig(encoder='not_an_encoder').build('cpu')
    except ValueError as e:
        assert 'Known' in str(e)
        return 'refused'
    raise AssertionError('an unregistered encoder name got as far as a build')


# ══════════════════════════════════════════════════════════════════════════════
#  4. forward
# ══════════════════════════════════════════════════════════════════════════════

def t_normalise_is_the_transform_minus_its_geometry():
    """`Resize -> CenterCrop -> ToTensor -> Normalize`. The first two must not
    happen and the last two must, and the check is arithmetic rather than a
    tolerance: a constant image has one exactly known answer.
    """
    b = _backbone()
    t = b.cfg_transform
    x = torch.full((2, 3, TILE, TILE), 0.5)
    out = b.normalise(x)
    assert out.shape == x.shape, \
        f'normalise changed the geometry: {tuple(out.shape)} from ' \
        f'{tuple(x.shape)} -- the resize and the centre crop leaked in'
    for c in range(3):
        want = (0.5 - t.mean[c]) / t.std[c]
        got = float(out[0, c].mean())
        assert abs(got - want) < 1e-5, f'channel {c}: {got} vs {want}'


def t_grey_is_luma_replicated_and_not_a_one_channel_input():
    """spec.md 13's grayscale arm for a trunk that takes three channels. The
    coefficients are upstream's 0.299/0.587/0.114 -- the same three as
    `Teacher._to_gray` and `Backbones.to_model_channels` -- and a plain mean of
    the three channels is the decoy, because it is what a careless
    implementation writes and H&E's pink and purple are exactly where the two
    differ most.
    """
    b = _backbone(cfg_over={'preprocess': 'grey'})
    t = b.cfg_transform
    x = torch.zeros(1, 3, 32, 32)
    x[:, 0] = 1.0                                   # pure red
    out = b.normalise(x)
    for c in range(3):
        want = (0.299 - t.mean[c]) / t.std[c]
        assert abs(float(out[0, c].mean()) - want) < 1e-5, \
            f'channel {c} is not luma; a plain mean would give {1 / 3:.3f}'
    decoy = (1 / 3 - t.mean[0]) / t.std[0]
    assert abs(float(out[0, 0].mean()) - decoy) > 1e-3


def t_one_channel_in_is_refused_and_says_where_the_grey_arm_lives():
    b = _backbone()
    try:
        b(torch.zeros(1, 1, TILE, TILE))
    except ValueError as e:
        assert 'grey' in str(e)
        return 'refused'
    raise AssertionError('a 1-channel batch reached a 3-channel patch embedding')


def t_a_size_the_stride_does_not_divide_is_refused_at_the_forward():
    b = _backbone()
    try:
        b(torch.zeros(1, 3, TILE, TILE - 1))
    except ValueError as e:
        assert 'stride' in str(e)
        return 'refused'
    raise AssertionError('a size the patch grid does not cover was accepted')


def t_the_frozen_trunk_is_not_in_the_checkpoint_and_not_in_the_optimiser():
    """`self._trunk = [trunk]`, so `nn.Module.__setattr__` does not register it.

    Both halves are silent when wrong: the optimiser would be handed frozen
    tensors and step them nowhere, and every checkpoint would carry gigabytes
    of weights the encoder can reload from its own store.
    """
    b = _backbone()
    assert list(b.parameters()) == [], \
        f'{sum(p.numel() for p in b.parameters())} trunk parameters reached ' \
        f'the optimiser'
    assert b.state_dict() == {}, list(b.state_dict())
    assert b.trunk.model is not None, 'and yet the trunk is still reachable'


def t_the_gradient_stops_at_the_trunk_and_flows_in_the_head():
    """What `trainable = False` has to mean in practice: the map arrives
    detached, and a decoder built on it still trains. A backbone that forgot
    the `no_grad` would keep the trunk's activation graph -- gigabytes -- and
    a backbone that detached too late would train nothing.
    """
    b = _backbone()
    feats = b(torch.zeros(2, 3, TILE, TILE))
    assert not feats.requires_grad, 'the trunk kept its graph'

    head = nn.Conv2d(DIM, 4, 1)
    loss = head(feats).pow(2).mean()
    loss.backward()
    assert head.weight.grad is not None, 'nothing downstream can train'
    assert float(head.weight.grad.abs().sum()) >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  5. identity
# ══════════════════════════════════════════════════════════════════════════════

def t_identity_moves_with_what_changes_the_features():
    """Every field here changes what comes out of the trunk, so every one of
    them has to appear when it differs from the baseline and stay quiet when it
    does not.
    """
    base = TileEncoderBackboneConfig()
    assert base.identity_parts(_TILE_ENCODER_BASELINE) == [], \
        base.identity_parts(_TILE_ENCODER_BASELINE)
    for fieldname, value in (('encoder', 'uni2'), ('head', 'trunk'),
                             ('dtype', 'fp32'), ('preprocess', 'grey'),
                             ('tile_size', 512)):
        parts = replace(base, **{fieldname: value}).identity_parts(
            _TILE_ENCODER_BASELINE)
        assert any(p.startswith(f'{fieldname}=') for p in parts), \
            f'{fieldname} changed and the identity did not move: {parts}'


def t_encoder_id_reports_the_trunk_and_not_this_config():
    """Two facts, two places: the config says which encoder and how it is read,
    `encoder_id` says which weights got loaded. A checkpoint wants both, and
    only the first can be a dataclass field.
    """
    b = _backbone()
    eid = b.encoder_id
    assert isinstance(eid, str) and eid
    assert eid == b.trunk.identity_id()


# ══════════════════════════════════════════════════════════════════════════════
#  6. model  (--with-model)
# ══════════════════════════════════════════════════════════════════════════════

def with_model(encoder: str, tile: int, device: str):
    """One real encoder at a real tile size. Minutes, and a weights download the
    first time.

    What this adds over the fake is the only thing the fake cannot have: a
    position embedding that actually has to interpolate. `dynamic_img_size` is
    a flag the fake can set; whether timm then produces a 16x16 grid from a 256
    px input is a fact about the real model.
    """
    dev = torch.device(device)
    cfg = TileEncoderBackboneConfig(encoder=encoder, tile_size=tile)
    print(f'\n[model] building {encoder} at tile {tile} on {dev} ...')
    b = cfg.build(dev)
    print(f'  stride {b.stride}   out_channels {b.out_channels}')
    print(f'  encoder_id {b.encoder_id}')

    x = torch.rand(2, 3, tile, tile, device=dev)
    out = b(x)
    print(f'  {tuple(x.shape)} -> {tuple(out.shape)}')
    assert out.shape[:2] == (2, b.out_channels)
    assert out.shape[-1] == tile // b.stride
    assert out.dtype is torch.float32
    assert not out.requires_grad

    # The inference exit, for comparison. It resizes to crop_size and returns
    # to the host, which is the whole reason this backbone does not use it.
    grid_here = out.shape[-1]
    grid_there = b.trunk.model_spec.feat_hw[0]
    print(f'  training grid {grid_here}  vs  inference grid {grid_there} '
          f'(crop {b.cfg_transform.crop_size} // patch {b.stride})')
    return f'{encoder}: {tuple(out.shape)}'


_SECTIONS = {
    'adopt':    ['t_over_reparents_without_rebuilding_the_model',
                 't_over_resolves_the_concrete_spatial_forward_and_not_the_base_refusal',
                 't_over_refuses_something_that_is_not_an_encoder'],
    'shapes':   ['t_stride_is_the_patch_and_not_the_inference_grid',
                 't_the_backbone_satisfies_the_protocol_and_its_own_check',
                 't_a_lying_stride_is_caught_at_construction',
                 't_a_lying_width_is_caught_at_construction'],
    'refusals': ['t_a_tile_the_patch_does_not_divide_is_refused',
                 't_a_fixed_size_trunk_is_resized_rather_than_refused',
                 't_a_trunk_that_cannot_be_resized_is_refused',
                 't_a_non_square_patch_is_refused',
                 't_preprocess_must_agree_with_the_trunk_it_was_built_over',
                 't_an_unknown_encoder_name_is_refused_before_anything_loads'],
    'forward':  ['t_normalise_is_the_transform_minus_its_geometry',
                 't_grey_is_luma_replicated_and_not_a_one_channel_input',
                 't_one_channel_in_is_refused_and_says_where_the_grey_arm_lives',
                 't_a_size_the_stride_does_not_divide_is_refused_at_the_forward',
                 't_the_frozen_trunk_is_not_in_the_checkpoint_and_not_in_the_optimiser',
                 't_the_gradient_stops_at_the_trunk_and_flows_in_the_head'],
    'identity': ['t_identity_moves_with_what_changes_the_features',
                 't_encoder_id_reports_the_trunk_and_not_this_config'],
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(_SECTIONS))
    ap.add_argument('--with-model', action='store_true',
                    help='build a real encoder and read its map. Minutes.')
    ap.add_argument('--encoder', default='gigapath',
                    help='which one. gigapath and conch_vit are patch 16 and '
                         'take 256; uni2 is patch 14 and takes 224/252/266')
    ap.add_argument('--tile-size', type=int, default=TILE)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available()
                    else 'cpu')
    args = ap.parse_args()

    torch.manual_seed(0)
    for section in (args.only or list(_SECTIONS)):
        print(f'\n[{section}]')
        for name in _SECTIONS[section]:
            check(name[2:].replace('_', ' '), globals()[name])

    if args.with_model:
        print('\n[model]')
        check(f'{args.encoder} at tile {args.tile_size}',
              lambda: with_model(args.encoder, args.tile_size, args.device))

    failed = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
