#!/usr/bin/env python3
"""Unit test for the TileEncoderFunc template.

    python utilities/test_modules/test_tile_encoder.py

No GPU. About a second, and one 5 MB download the first time -- vit_tiny, for
the last section only; everything else runs on fakes. Three fake models stand
in for the three output kinds, which is the whole point of the exercise: a base
class written
against one implementation looks correct until the second one arrives, and this
repo has watched that happen twice in a day -- num_classes=0 was GigaPath's
config rather than the encoder's, and a one-tile region turned a decoy into the
identity.

So the template is checked against a vector model, a token model and a spatial
model before any of them exist for real.

The pooling and model_token_spec sections at the bottom came from
what is now test_gigapath_equivalence, and they are here because their SUBJECT
moved: 923e9e6
took those functions out of GigaPathFunc into TileEncoderFunc, and the tests
stayed behind under a filename naming one encoder. They fit the description
above unchanged -- fake specs, no weights, a second to run -- and several of
them describe models this repo does not have: num_prefix=0 is a CNN, and the
prefix-free branch has no other coverage because all three real encoders carry
a CLS.

The last section is different again: it uses a REAL config, because "which of
my fields reach the hash" is a question a fake would be both asking and
answering. It could as well have gone in test_encoders; it is here because it
costs 5 MB rather than 4.5 GB, and that is the line this file is on.

The real encoders are checked in test_encoders.py, which needs the weights.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: BEFORE `import torch`. huggingface_hub freezes HF_HOME into module constants
#: at its own import, and something here reaches it during `import torch`, so
#: the setdefault inside each encoder module runs too late to decide anything.
#: This file downloads vit_tiny for its last section; without this line it goes
#: to ~/.cache/huggingface, where it does not join the 16 GB already in /work.
#: setdefault, so jobscripts/_env.sh and an exported HF_HOME both still win.
os.environ.setdefault(
    'HF_HOME', os.environ.get('LOCASCOPE_MODEL_WEIGHTS',
                              '/work/u26130998/model_weights'))

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _d in ('aiNNModel', 'utilities'):
    p = str(_ROOT / _d)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn.functional as F                             # noqa: E402

from ConfigIdentity import ModelConfig                      # noqa: E402
from TileEncoderFunc import (ModelOutputSpec, TileEncoder,       # noqa: E402
                             TileEncoderConfig, TransformConfig,
                             _ring_bins, model_token_spec,
                             pool_slots, pooling_kinds)

_RESULTS = []


def check(name, fn):
    try:
        fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}')
    except Exception as e:                                   # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n        {type(e).__name__}: {e}')


def rejects(fn, needle=''):
    try:
        fn()
    except Exception as e:                                   # noqa: BLE001
        if needle and needle not in str(e):
            raise AssertionError(
                f'raised, but the message never mentions {needle!r}: {e}') from None
        return
    raise AssertionError('should have raised, returned normally')


# ── fakes ─────────────────────────────────────────────────────────────────────

DIM, GRID, PREFIX = 6, (2, 2), 1
N_TOK = PREFIX + GRID[0] * GRID[1]      # 5


class _Vector(torch.nn.Module):
    """[B, 3, H, W] -> [B, D]. A CNN with its classifier removed."""
    def forward(self, x):
        return x.mean(dim=(-2, -1)) @ torch.ones(3, DIM)


class _Tokens(torch.nn.Module):
    """-> [B, T, D]. Every token of every sample is distinguishable.

    Keyed on the sample's CONTENT and not on its position in the batch. The
    first version used arange(batch_size), which made the output depend on where
    a tile landed in a batch -- the exact thing a real model cannot do, and the
    exact thing t_batching_does_not_reorder exists to detect. It reported a
    difference of 400 and it was right to.
    """
    def forward(self, x):
        who = x.mean(dim=(-3, -2, -1)).view(-1, 1, 1) * 1000.0
        base = torch.arange(N_TOK, dtype=torch.float32).view(1, N_TOK, 1)
        chan = torch.arange(DIM, dtype=torch.float32).view(1, 1, DIM) * 0.01
        return who + base + chan


class _Spatial(torch.nn.Module):
    """-> [B, C, H, W]. A U-Net-ish feature map, keyed on content."""
    def forward(self, x):
        who = x.mean(dim=(-3, -2, -1)).view(-1, 1, 1, 1) * 1000.0
        m = torch.arange(DIM * 4, dtype=torch.float32).view(1, DIM, 2, 2)
        return who + m


@dataclass(frozen=True)
class FakeConfig(TileEncoderConfig):
    kind: str = 'vector'
    transform: TransformConfig = field(
        default_factory=lambda: TransformConfig(scale_size=8, crop_size=8))

    #: '' is NOT mapped, and that is the fake's job here: it stands for a config
    #: whose author never said what their model's own answer is. feature_pooling
    #: raises for it, which is what t_features_refuses_to_guess_for_tokens
    #: checks. A real config maps it -- see TokenConfig below.
    POOLINGS = {'': '', 'cls': 'cls', 'gap': 'gap', 'identity': 'identity',
                'cls_avg': 'cls_avg', 'grid2x2': 'grid2x2'}

    def build(self, device, **kw):
        return FakeEncoder(self, device)


class FakeEncoder(TileEncoder):
    BASELINE = {'kind': 'vector',
                'model': ModelConfig(source='local', arch='x:Y', dtype='fp32'),
                'transform': TransformConfig(scale_size=8, crop_size=8),
                'head': '', 'pooling': ''}

    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        self.model = {'vector': _Vector, 'tokens': _Tokens,
                      'spatial': _Spatial}[cfg.kind]().eval()
        self.model_spec = {
            'vector':  ModelOutputSpec('vector', DIM),
            'tokens':  ModelOutputSpec('tokens', DIM, GRID, PREFIX),
            'spatial': ModelOutputSpec('spatial', DIM, (2, 2)),
        }[cfg.kind]
        self._transform = cfg.transform.build()
        self._weights_id = None


@dataclass(frozen=True)
class TokenConfig(FakeConfig):
    """A token model that HAS decided which token, the way a real one must.

    One table entry is the whole of it: '' -> 'cls'. There is no _vector_from
    override here and none in GigaPathFunc or Uni2Func either -- the base
    reduces by cfg.pooling, so an override would be a second copy of that
    decision and, worse, one that ignores cfg.pooling. This fake kept such an
    override for one revision and two of the tests below caught it, which is
    what they are for.
    """
    POOLINGS = {**FakeConfig.POOLINGS, '': 'cls'}

    def build(self, device, **kw):
        return TokenEncoder(self, device)


@dataclass(frozen=True)
class HeadConfig(FakeConfig):
    """A model whose head already produced the vector, so nothing pools it."""
    POOLINGS = {'': 'identity', 'identity': 'identity'}

    def build(self, device, **kw):
        return HeadEncoder(self, device)


class TokenEncoder(FakeEncoder):
    """Built by TokenConfig; the decision lives in the config, not here."""


#: What a head that collapses the token axis maps DIM to. Narrower than DIM on
#: purpose: if the two were equal, a _pool that still consulted model_spec.dim
#: would pass by coincidence.
HEAD_DIM = DIM - 2


class HeadEncoder(FakeEncoder):
    """A token model behind a head that reduces, the shape CONCH's tower has.

    The trunk hands back [B, T, DIM]; the head hands back [B, HEAD_DIM], which
    is both a different rank and a different width. model_spec still describes
    the TRUNK, because self.model is always the trunk -- so anything downstream
    that reads model_spec to decide how to treat the head's output is wrong, and
    this fake is here to say so before CONCH exists.
    """

    def _apply_head(self, raw):
        return raw[:, 0, :HEAD_DIM]     # stands in for an attentional pooler


def tiles(n=5):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(n)]


#: Which config builds which fake. A real encoder has exactly one config class
#: and the pairing is build(); here the tests name the ENCODER, so this is the
#: reverse lookup. It exists because '' now resolves in POOLINGS, which is a
#: property of the config -- so three fakes need three configs.
_CONFIG_FOR = {FakeEncoder: FakeConfig, TokenEncoder: TokenConfig,
               HeadEncoder: HeadConfig}


def enc(kind, cls=FakeEncoder, **over):
    cfg = _CONFIG_FOR[cls](
        kind=kind,
        model=ModelConfig(source='local', arch='x:Y', dtype='fp32'),
        **over)
    return cls(cfg, torch.device('cpu'))


# ── features works for every kind ─────────────────────────────────────────────

def t_features_for_vector_and_spatial():
    """The two kinds whose reduction is arithmetic, once the config names it.

    Named here rather than inferred from kind: the base used to read
    model_spec.kind to decide that a 'spatial' model means 'gap', which is only
    sound while nothing sits between the model and the reduction. A head breaks
    it, and CONCH has one.
    """
    for kind, pooling in (('vector', 'identity'), ('spatial', 'gap')):
        f = enc(kind, pooling=pooling).features(tiles())
        assert f.shape == (5, DIM), f'{kind}: {tuple(f.shape)}'
        norms = f.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f'{kind}: features are not unit norm'


def t_features_refuses_to_guess_for_tokens():
    """A config that never said what '' means for it must not have one picked.

    Guessing would be answering a question only the model's authors can --
    GigaPath validated the CLS against a stored tensor, another model may not
    have -- and the guess would be invisible in every number downstream.
    """
    rejects(lambda: enc('tokens').features(tiles()), 'POOLINGS')


def t_features_for_a_token_model_that_decided():
    """TokenConfig maps '' to 'cls', and that one table entry is the difference."""
    f = enc('tokens', TokenEncoder).features(tiles())
    assert f.shape == (5, DIM), tuple(f.shape)


# ── capabilities are gated, and say what the model is ─────────────────────────

def t_tokens_only_for_token_models():
    """tokens() is gated on kind. pooled() is NOT -- see below."""
    for kind in ('vector', 'spatial'):
        e = enc(kind)
        rejects(lambda: e.tokens(tiles()), kind)


def t_pooled_reaches_a_feature_map():
    """pooled() works on a CNN, because _pool permutes rather than refusing.

    [B, C, H, W] is channel-first and pooling_kinds reads channel-last, which is
    a layout difference and not a reason a feature map cannot be pooled into
    rings or blocks -- those mean the same thing over cells as over patches.

    'vector' output has genuinely nothing to reduce, so that one still refuses,
    and the message says what the only legal mode is rather than what is wrong.
    """
    got = enc('spatial').pooled(tiles(), 'grid2x2')
    assert got.shape == (5, 5, DIM), tuple(got.shape)     # summary + 2x2 blocks
    fs = enc('spatial').pooled_spec(got, 'grid2x2')
    assert fs.slots[0] == 'gap', fs.slots     # no prefix -> not 'cls'
    assert fs.slot_layout == 'grid:2x2', fs.slot_layout
    rejects(lambda: enc('vector').pooled(tiles(), 'cls'), "'identity'")


def t_spatial_only_for_spatial_models():
    for kind in ('vector', 'tokens'):
        e = enc(kind)
        rejects(lambda: e.spatial(tiles()), kind)


def t_the_refusal_names_what_works():
    """The message has to point somewhere, or it is an AttributeError with
    better spelling."""
    try:
        enc('vector').tokens(tiles())
    except TypeError as e:
        assert 'features()' in str(e), f'no way forward offered: {e}'


def t_shapes_come_through_unreduced():
    toks = enc('tokens').tokens(tiles())
    assert toks.shape == (5, N_TOK, DIM), tuple(toks.shape)
    maps = enc('spatial').spatial(tiles())
    assert maps.shape == (5, DIM, 2, 2), tuple(maps.shape)


# ── the batch loop ────────────────────────────────────────────────────────────

def t_batching_does_not_reorder():
    """batch_size below the tile count on purpose: the loop has to run more than
    once or an off-by-one in the concatenation never shows."""
    whole = enc('tokens', batch_size=99).tokens(tiles(5))
    split = enc('tokens', batch_size=2).tokens(tiles(5))
    gap = float((whole - split).abs().max())
    decoy = float((whole - split.roll(1, dims=0)).abs().max())
    print(f'        batch 99 vs 2   max|Δ| {gap:.3e}   roll decoy {decoy:.2e}')
    assert gap == 0.0, f'batching changed the result by {gap:.3e}'
    assert decoy > 0, 'the decoy is empty, so this comparison proves nothing'


def t_reduce_runs_inside_the_loop():
    """reduce is the only place the device decision can be made, so it has to
    see one batch at a time, not the concatenated whole."""
    seen = []

    def watch(t):
        seen.append(t.shape[0])
        return t.cpu()

    enc('tokens', batch_size=2).tokens(tiles(5), reduce=watch)
    assert seen == [2, 2, 1], seen


def t_reduce_gets_fp32():
    """Under autocast the model output is fp16, and .float() has to happen
    BEFORE reduce or every averaged reduction is computed in fp16."""
    got = []
    enc('tokens').tokens(tiles(3), reduce=lambda t: got.append(t.dtype) or t.cpu())
    assert got and all(d is torch.float32 for d in got), got


# ── variant ───────────────────────────────────────────────────────────────────

def t_variant_shares_the_model():
    a = enc('tokens')
    b = a.variant(batch_size=1)
    assert b.model is a.model, 'variant reloaded the model'
    assert b.cfg.batch_size == 1 and a.cfg.batch_size == 128


def t_variant_rewrites_dtype_on_the_nested_config():
    a = enc('tokens')
    b = a.variant(dtype='fp16')
    assert b.cfg.model.dtype == 'fp16' and a.cfg.model.dtype == 'fp32'


def t_variant_refuses_what_rebuilds():
    a = enc('tokens')
    rejects(lambda: a.variant(kind='vector'), 'kind')
    rejects(lambda: a.variant(model=ModelConfig()), 'model')


# ── ModelOutputSpec ──────────────────────────────────────────────────────────

def t_output_spec_refuses_nonsense():
    rejects(lambda: ModelOutputSpec('token', 8), 'kind must be')
    rejects(lambda: ModelOutputSpec('tokens', 8), 'patch grid')
    rejects(lambda: ModelOutputSpec('vector', 8, (2, 2)), 'no feat_hw')
    assert ModelOutputSpec('tokens', 8, (14, 14), 1).n_tokens() == 197
    rejects(lambda: ModelOutputSpec('vector', 8).n_tokens(), 'no token count')

# ── cfg.pooling decides what __call__ hands back ──────────────────────────────

def t_call_follows_the_configured_pooling():
    """The whole reason pooling is a FIELD rather than an argument.

    encoder(patches) is the EncodeFn PatchingLib speaks -- one list in, one
    tensor out, nowhere to put a mode. Before this, a multi-slot pooling could
    only reach retrieval by a caller assembling its own FeaturesMap. Here the
    config says grid2x2 and the plain call honours it.

    Widths, not just "different": 5 slots of DIM concatenated, which is also
    what makes the cosine the mean of the per-slot cosines.
    """
    plain = enc('tokens', TokenEncoder)
    grid = enc('tokens', TokenEncoder, pooling='grid2x2')
    assert plain(tiles()).shape == (5, DIM), tuple(plain(tiles()).shape)
    assert grid(tiles()).shape == (5, 5 * DIM), tuple(grid(tiles()).shape)


def t_pooling_alias_does_not_fork_the_id():
    """'' and 'cls' are one computation on a model whose answer IS the CLS.

    Both land on 'cls', because POOLINGS resolves in __post_init__ and the
    canonical side of the table is the concrete mode. So nothing downstream ever
    sees '' -- not feature_pooling, not the store's filename, not the id.
    """
    a = enc('tokens', TokenEncoder)
    b = enc('tokens', TokenEncoder, pooling='cls')
    assert a.cfg.pooling == 'cls', f"'' stayed {a.cfg.pooling!r}"
    assert b.cfg.pooling == 'cls', f'alias survived as {b.cfg.pooling!r}'
    assert a.identity_id() == b.identity_id()


def t_unknown_pooling_dies_at_config_time():
    """Before the weights load, not after -- POOLINGS is closed for that.

    The grid family looked unenumerable until the divisibility rule made it
    obvious that which grids are legal was always a property of the model.
    """
    rejects(lambda: FakeConfig(kind='tokens', pooling='grid3x3'), 'grid3x3')
    rejects(lambda: FakeConfig(kind='tokens', pooling='grid3x3'), 'Accepts')


def t_feature_spec_describes_what_features_returned():
    """One slot whatever the pooling, because _vector_from flattened them.

    The file holds one vector per tile and `pooling` is the only record of what
    went into it; pool_slots(pooling, model_spec) recovers the rest.
    """
    for pooling, width in (('', DIM), ('grid2x2', 5 * DIM)):
        e = enc('tokens', TokenEncoder, pooling=pooling)
        f = e.features(tiles())
        fs = e.feature_spec(f)
        assert fs.dim == width == f.shape[-1], (pooling, fs.dim, f.shape)
        assert fs.slots == (e.feature_pooling,), fs.slots
        assert fs.pooling == ('cls' if pooling == '' else pooling), fs.pooling


def t_a_head_that_reduces_does_not_confuse_the_pooling():
    """_pool reads the RANK of what it is handed, not model_spec.kind.

    model_spec says 'tokens' because self.model is the trunk. After the head
    there is no token axis and no DIM-wide channel left, so a _pool that
    branched on kind would push a two-dimensional tensor into pooling_kinds and
    be told it is not [N, T, D]. This is CONCH's shape, written down before
    CONCH is, because the version of _pool that shipped an hour ago failed it.
    """
    e = enc('tokens', HeadEncoder)
    f = e.features(tiles())
    assert f.shape == (5, HEAD_DIM), tuple(f.shape)
    fs = e.feature_spec(f)
    assert fs.dim == HEAD_DIM, fs.dim         # NOT model_spec.dim
    assert e.model_spec.dim == DIM, 'model_spec must still describe the trunk'
    # tokens() is upstream of the head, so it is unaffected by it.
    assert e.tokens(tiles()).shape == (5, N_TOK, DIM), tuple(e.tokens(tiles()).shape)


def t_tokens_spec_is_the_model_s_own():
    """tokens() went through no head and no pooling, so its spec is model_spec
    and its slots are empty -- naming entries is a reducer's job."""
    e = enc('tokens', TokenEncoder)
    fs = e.tokens_spec(e.tokens(tiles()))
    assert fs.shape == e.model_spec, fs.shape
    assert fs.slots == (), fs.slots


# ── identity ──────────────────────────────────────────────────────────────────

def t_identity_ignores_batch_size():
    a = enc('tokens')
    assert a.identity_id() == a.variant(batch_size=7).identity_id(), \
        'batch size split the identity; a throughput sweep would drop the cache'


def t_identity_moves_on_transform():
    a = enc('tokens')
    b = a.variant(transform=TransformConfig(scale_size=8, crop_size=4))
    assert a.identity_id() != b.identity_id()


# ══ pooling and model_token_spec ══════════════════════════════════════════════
#
# These came from what is now test_gigapath_equivalence, and are here because
# the code
# they test moved. 923e9e6 took model_token_spec, _ring_bins and pool_tokens
# out of GigaPathFunc -- which now holds none of them -- and put them in
# TileEncoderFunc as model_token_spec, _ring_bins, pool_slots and
# pooling_kinds. The tests did not follow, so for one release the only test of
# this module's pooling arithmetic lived in a file named after one encoder.
#
# They were never GigaPath's. Every one reads a spec and a tensor, and several
# describe models this repo does not have yet: num_prefix=0 is a CNN, and
# num_prefix=5 is DINOv2 with four register tokens. UNI2-h has eight, which is
# what turned that from a hypothetical into the reason model_token_spec takes
# no default.

SPEC = {'dim': 8, 'feat_hw': (14, 14), 'num_prefix': 1}


def toks(n=3, spec=SPEC, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = spec['num_prefix'] + spec['feat_hw'][0] * spec['feat_hw'][1]
    return torch.randn(n, t, spec['dim'], generator=g)


def t_shapes_and_slots():
    x = toks()
    for mode, n, layout in (('cls', 1, 'none'),
                            ('cls_avg', 2, 'none'),
                            ('cls_std', 2, 'none'),
                            ('rings3', 4, 'ring:3'),
                            ('grid2x2', 5, 'grid:2x2'),
                            ('tokens', 197, 'grid:14x14')):
        f = pooling_kinds(x, mode, SPEC)
        slots, lay = pool_slots(mode, SPEC)
        assert f.shape == (3, n, SPEC['dim']), f'{mode}: shape {tuple(f.shape)}'
        assert len(slots) == n, f'{mode}: {len(slots)} names for {n} slots'
        assert lay == layout, f'{mode}: layout {lay!r} != {layout!r}'
        assert slots[0] == 'cls', f'{mode}: slot 0 must be cls'


def t_every_slot_is_unit_norm():
    # Each slot separately, not the flattened vector -- otherwise the weight
    # between slots is fixed at write time, and that weighting is the open
    # question the experiment exists to answer.
    for mode in ('cls', 'cls_avg', 'cls_std', 'rings3', 'grid2x2', 'tokens'):
        f = pooling_kinds(toks(), mode, SPEC)
        n = f.norm(dim=-1)
        assert torch.allclose(n, torch.ones_like(n), atol=1e-5), \
            f'{mode}: slot norms range {n.min():.4f}..{n.max():.4f}'


def t_cls_slot_is_just_the_cls_token():
    x = toks()
    f = pooling_kinds(x, 'cls_avg', SPEC)
    assert torch.allclose(f[:, 0], F.normalize(x[:, 0], dim=-1), atol=1e-6)


def t_registers_are_dropped():
    # DINOv2 _reg4 has num_prefix=5 and UNI2-h has 9. Patches must start after
    # the registers; the failure mode otherwise is junk vectors averaged into
    # every pooling, with no error and only a slightly worse score to show.
    spec = {'dim': 8, 'feat_hw': (4, 4), 'num_prefix': 5}
    x = torch.zeros(1, 5 + 16, 8)
    x[0, 0] = 1.0                       # cls
    x[0, 1:5] = 99.0                    # registers -- must not reach any slot
    x[0, 5:] = 2.0                      # patches
    f = pooling_kinds(x, 'cls_avg', spec)
    assert torch.allclose(f[0, 1], F.normalize(torch.full((8,), 2.0), dim=-1)), \
        'the average picked up the register tokens'


def t_no_prefix_makes_slot_0_the_average():
    # A CNN or a U-Net has no CLS: num_prefix is 0 and its summary vector is the
    # global average, which is also what its features() returns. Reading
    # tokens[:, 0] regardless would put the TOP-LEFT CELL in slot 0 and label it
    # 'cls'. Nothing downstream raises -- the token-count check passes because
    # 0 + gh*gw == T, the store validates, the vectors look like vectors. It
    # would show up only as a pooling that scores worse than it should.
    #
    # Scored against that exact decoy rather than a tolerance: the corner is
    # what the old code returned, so if it ever comes back this fails at the
    # decoy and not at the margin.
    spec = {'dim': 4, 'feat_hw': (2, 2), 'num_prefix': 0}
    x = torch.zeros(1, 4, 4)
    x[0, 0] = torch.tensor([9.0, 0.0, 0.0, 0.0])        # the corner cell
    x[0, 1:] = 1.0
    f = pooling_kinds(x, 'cls', spec)
    slots, _ = pool_slots('cls', spec)
    want = F.normalize(x[0].mean(0), dim=-1)
    corner = F.normalize(x[0, 0], dim=-1)
    d_want = float((f[0, 0] - want).abs().max())
    d_corner = float((f[0, 0] - corner).abs().max())
    assert d_want * 1000 < d_corner, (
        f'slot 0 is {d_want:.3e} from the average and {d_corner:.3e} from the '
        f'corner cell -- it took tokens[:, 0] with no prefix to take')
    # 'gap' and not 'cls': StoreMeta.slots is the only record of what slot 0
    # holds, so naming a global average after a token this model does not have
    # would make the store say something false about itself.
    assert slots[0] == 'gap', f"slot 0 of a prefix-free model is {slots[0]!r}"


def t_no_prefix_refuses_the_mode_that_would_duplicate():
    # cls_avg pairs the summary with the patch mean. With no prefix the summary
    # IS the patch mean, so both slots would be the same vector -- bit-identical
    # after per-slot normalisation. Nothing raises on its own: the store holds
    # two slots, every similarity is exactly doubled, and the result reads as
    # "cls_avg performs like cls" rather than as a broken configuration.
    #
    # cls_std is the control. std is not mean, so it stays legal, and if this
    # ever starts raising the refusal has grown past what it was for.
    spec = {'dim': 4, 'feat_hw': (2, 2), 'num_prefix': 0}
    x = torch.randn(2, 4, 4, generator=torch.Generator().manual_seed(1))
    rejects(lambda: pooling_kinds(x, 'cls_avg', spec), 'num_prefix=0')
    f = pooling_kinds(x, 'cls_std', spec)
    slots, _ = pool_slots('cls_std', spec)
    assert slots == ('gap', 'std'), f'slots {slots}'
    assert not torch.allclose(f[:, 0], f[:, 1]), 'std collapsed onto the summary'

    # Every remaining mode names slot 0 for what it actually holds.
    for mode in ('cls', 'cls_std', 'rings2', 'grid2x2', 'tokens'):
        pooling_kinds(x, mode, spec)
        slots, _ = pool_slots(mode, spec)
        assert slots[0] == 'gap', f'{mode}: slot 0 is {slots[0]!r}'


def t_no_prefix_keeps_every_cell_as_a_patch():
    # The other half of the same rule: with num_prefix 0 the summary is computed
    # FROM the cells, so no cell may be consumed by it. A slice of tokens[:, 1:]
    # would drop one and still produce a plausible answer for every mode.
    spec = {'dim': 2, 'feat_hw': (2, 2), 'num_prefix': 0}
    x = torch.zeros(1, 4, 2)
    x[0, :, 0] = torch.arange(4, dtype=torch.float32)
    f = pooling_kinds(x, 'tokens', spec)
    slots, layout = pool_slots('tokens', spec)
    assert layout == 'grid:2x2'
    assert len(slots) == 5, f'1 summary + 4 cells, got {len(slots)}'
    for k in range(4):
        assert torch.allclose(f[0, 1 + k], F.normalize(x[0, k], dim=-1),
                              atol=1e-5), f'cell {k} is not slot {1 + k}'


def t_outputspec_and_storemeta_subscript_the_same():
    # pooling_kinds takes either supplier by name. The point of the test is that
    # nothing converts between them: the same three keys reach it whether it was
    # handed a live encoder's model_spec or a StoreMeta off disk, so a rename on
    # one side fails here rather than at some consumer months later.
    spec = ModelOutputSpec(kind='tokens', dim=8, feat_hw=(4, 4), num_prefix=1)
    for k, want in (('dim', 8), ('feat_hw', (4, 4)), ('num_prefix', 1),
                    ('kind', 'tokens')):
        assert spec[k] == want, f'spec[{k!r}] == {spec[k]!r}, wanted {want!r}'
    x = toks(spec={'dim': 8, 'feat_hw': (4, 4), 'num_prefix': 1})
    a = pooling_kinds(x, 'rings3', spec)
    b = pooling_kinds(x, 'rings3', {'dim': 8, 'feat_hw': (4, 4),
                                    'num_prefix': 1})
    assert torch.equal(a, b), 'ModelOutputSpec and the plain dict disagree'
    rejects(lambda: spec['token_grid'], 'token_grid')


def t_grid_blocks_average_the_right_cells():
    spec = {'dim': 1, 'feat_hw': (4, 4), 'num_prefix': 1}
    x = torch.zeros(1, 17, 1)
    x[0, 1:] = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    # 4x4 grid, 2x2 blocks -> top-left block holds cells 0,1,4,5 -> mean 2.5
    g = x[0, 1:].reshape(4, 4)
    want = [g[:2, :2].mean(), g[:2, 2:].mean(), g[2:, :2].mean(), g[2:, 2:].mean()]
    slots, _ = pool_slots('grid2x2', spec)
    assert slots[1:] == ('g00', 'g01', 'g10', 'g11')
    # after per-slot normalization a 1-D vector is just its sign, so compare the
    # pre-normalized means through a second call on a widened dim
    x2 = x.repeat(1, 1, 2)
    f2 = pooling_kinds(x2, 'grid2x2', {**spec, 'dim': 2})
    for k, w in enumerate(want):
        got = f2[0, 1 + k]
        assert torch.allclose(got, F.normalize(torch.tensor([w, w]), dim=-1),
                              atol=1e-5), f'block {slots[1+k]}: expected mean {w}'


def t_rings_hold_equal_counts():
    bins = _ring_bins(14, 14, 3)
    counts = [int((bins == k).sum()) for k in range(3)]
    assert max(counts) - min(counts) <= 1, f'ring sizes {counts} are not balanced'
    assert sorted(set(bins.tolist())) == [0, 1, 2]


def t_rings_follow_the_tokens_to_the_gpu():
    """A CPU ring mask cannot index CUDA tokens, and rings3 is the only mode
    that indexes at all -- the others are reshape and slice, which never notice.

    So pooling moved onto the GPU (bench_slidewin_pooling's `reduce`) breaks
    exactly one of the five and leaves four working. It raises rather than
    corrupting, but it raises deep inside a run that has already read a whole
    tissue region, which is minutes and tens of GB from here.
    """
    assert _ring_bins(14, 14, 3).device.type == 'cpu'
    if not torch.cuda.is_available():
        return                                   # nothing to check without one
    dev = torch.device('cuda')
    assert _ring_bins(14, 14, 3, dev).device.type == 'cuda'
    assert torch.equal(_ring_bins(14, 14, 3, dev).cpu(), _ring_bins(14, 14, 3))
    f = pooling_kinds(toks().to(dev), 'rings3', SPEC)
    assert f.device.type == 'cuda', "ring pooling left the tokens' device"


def t_bad_inputs_are_refused():
    rejects(lambda: pooling_kinds(torch.zeros(3, 8), 'cls', SPEC), '[N, T, D]')
    rejects(lambda: pooling_kinds(torch.zeros(3, 5, 8), 'cls', SPEC), 'disagree')
    rejects(lambda: pooling_kinds(toks(), 'nonsense', SPEC), 'unknown pooling mode')
    rejects(lambda: pooling_kinds(toks(), 'grid3x3', SPEC), 'does not divide')


# ── model_token_spec ──────────────────────────────────────────────────────────

class _Fake(torch.nn.Module):
    """Enough of a timm ViT for model_token_spec to read, and nothing else."""

    def __init__(self, head=None):
        super().__init__()
        self.embed_dim = 8
        self.num_prefix_tokens = 1
        self.head = head if head is not None else torch.nn.Identity()
        self.patch_embed = torch.nn.Module()
        self.patch_embed.grid_size = (4, 4)


def t_spec_unwraps_dataparallel():
    m = _Fake()
    dp = torch.nn.DataParallel(m)
    # The pit itself: DataParallel defines no __getattr__, so nn.Module's is
    # used and it searches only _parameters / _buffers / _modules.
    rejects(lambda: dp.embed_dim, 'embed_dim')
    assert model_token_spec(dp) == model_token_spec(m) == {
        'dim': 8, 'feat_hw': (4, 4), 'num_prefix': 1}


def t_spec_refuses_a_classifier():
    rejects(lambda: model_token_spec(_Fake(head=torch.nn.Linear(8, 5))),
            'num_classes=0')


# ── config fields are honoured ────────────────────────────────────────────────
#
# A config field that is declared, documented and hashed but never read is the
# failure this section exists for. It happened here: weights, tome_r and compile
# were added to EncoderConfig with comments explaining their effect and none of
# them reached __init__. Setting weights='ft.ckpt' loaded the HF weights.
#
# The shape of that failure is worse than doing nothing. An inert field that
# differs from its baseline still MOVES the hash -- two stores, two filenames,
# identical vectors, and every reason to believe they hold different things.
#
# tome_r was one of the three and is gone; `pooling` took its place in the
# checks below, and it is the better example because reading it is what
# features() now does rather than something build() does once.
#
# A REAL config, unlike everything above. The fakes in this file declare their
# own baselines, so a fake asking "which of my fields reach the hash" would be
# asking itself a question it also answers. GigaPathEncoderConfig is the oldest
# concrete answer to ConfigIdentity's contract, which is why it is the fixture.
#
# vit_tiny rather than GigaPath: the loading path does not depend on the
# architecture, and 5 MB keeps this in the half that runs on a login node.

_TINY = 'vit_tiny_patch16_224'


def _tiny_encoder(**over):
    from GigaPathFunc import GigaPathEncoderConfig
    # arch, dtype and weights live on the nested ModelConfig now, so they go
    # through with_model rather than being spelled at the top level.
    model_over = {k: over.pop(k) for k in ('weights',) if k in over}
    return (GigaPathEncoderConfig(batch_size=2, **over)
            .with_model(arch=_TINY, dtype='fp32', **model_over))


def t_weights_field_is_actually_loaded():
    """GigaPathEncoderConfig(weights=...) must change the model, not just the hash.

    The second checkpoint is the first one with a single tensor perturbed. A
    real finetune is not needed and would not test anything more: what is under
    test is whether load_state_dict is reached, and any two different parameter
    sets answer that.

    weights_id is the probe because it is derived from the parameters actually
    resident. If cfg.weights were ignored, the second encoder would hold the
    pretrained values, the two ids would match, and this fails.
    """
    import tempfile

    dev = torch.device('cpu')
    base = _tiny_encoder().build(dev)
    id_a = base.weights_id

    state = {k: v.clone() for k, v in base.model.state_dict().items()}
    # A matrix, and noise rather than a constant. The first attempt added 0.25
    # to sorted(state)[0], which was blocks.0.attn.proj.bias -- a constant added
    # to a bias shifts every token by the same vector, and every LayerNorm
    # downstream subtracts the per-token mean, so 0.25 arrived at the output as
    # 3.6e-07. The id moved and the features did not, which says something true
    # about both: weights_id is strictly more sensitive than behaviour, and a
    # perturbation has to survive normalisation to test anything.
    victim = next(k for k in sorted(state)
                  if state[k].ndim >= 2 and state[k].is_floating_point())
    g = torch.Generator().manual_seed(0)
    t = state[victim]
    state[victim] = t + torch.randn(t.shape, generator=g) * 0.1 * float(t.std())

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'perturbed.pt'
        torch.save(state, path)
        other = _tiny_encoder(weights=str(path)).build(dev)
        id_b = other.weights_id

        assert id_a != id_b, (
            f'weights_id is {id_a} both with and without weights= -- the '
            f'checkpoint was not loaded, so the field is inert')

        # And the vectors move, which is the thing the id is standing in for.
        rng = np.random.default_rng(0)
        pixels = [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
                  for _ in range(2)]
        gap = float((base.features(pixels) - other.features(pixels)).abs().max())
        print(f'        {victim}  +10% noise   ids {id_a[:8]} -> {id_b[:8]}   '
              f'max|dfeat| {gap:.3e}')
        assert gap > 1e-4, f'weights changed but features did not: {gap:.3e}'

    # The path is provenance, not identity: the same content under another name
    # must not split the cache.
    parts = other.identity_parts()
    assert not any(p.startswith('weights=/') or 'perturbed' in p for p in parts), \
        f'the checkpoint PATH leaked into identity: {parts}'


def t_identity_moves_only_where_it_should():
    """Which fields reach the hash, checked without loading anything."""
    from GigaPathFunc import GigaPathEncoderConfig, _GIGAPATH_BASELINE

    B = _GIGAPATH_BASELINE
    base = GigaPathEncoderConfig().identity_parts(B)
    assert base == [], f'a default config should be all-baseline, got {base}'

    M = B['model']
    import dataclasses as _dc
    for over, why in ((dict(batch_size=999), 'batch_size'),
                      (dict(compile=True), 'compile'),
                      (dict(model=_dc.replace(M, weights='/tmp/x.ckpt')), 'weights'),
                      (dict(pooling='cls'), "pooling at its baseline, spelled")):
        assert GigaPathEncoderConfig(**over).identity_parts(B) == [], \
            f'{why} must not reach the hash: ' \
            f'{GigaPathEncoderConfig(**over).identity_parts(B)}'

    for over, why in ((dict(pooling='grid2x2'), 'pooling away from baseline'),
                      (dict(model=_dc.replace(M, dtype='fp32')), 'dtype'),
                      (dict(model=_dc.replace(M, arch='timm:other')), 'arch')):
        assert GigaPathEncoderConfig(**over).identity_parts(B) != [], \
            f'{why} must reach the hash'

    # The transform rides in under its own prefix, and only when it differs.
    assert GigaPathEncoderConfig(transform=B['transform']).identity_parts(B) == []
    grey = GigaPathEncoderConfig(
        transform=_dc.replace(B['transform'], preprocess='grey')).identity_parts(B)
    assert grey == ['transform.preprocess=grey'], grey


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    argparse.ArgumentParser().parse_args()

    print('features()')
    check('works where the reduction is arithmetic', t_features_for_vector_and_spatial)
    check('refuses to guess for tokens',      t_features_refuses_to_guess_for_tokens)
    check('works once a token model decides', t_features_for_a_token_model_that_decided)

    print('capabilities')
    check('tokens() gated on kind',           t_tokens_only_for_token_models)
    check('pooled() reaches a feature map',   t_pooled_reaches_a_feature_map)
    check('spatial() gated on kind',          t_spatial_only_for_spatial_models)
    check('the refusal names a way forward',  t_the_refusal_names_what_works)
    check('unreduced shapes come through',    t_shapes_come_through_unreduced)

    print('batch loop')
    check('batching does not reorder',        t_batching_does_not_reorder)
    check('reduce sees one batch at a time',  t_reduce_runs_inside_the_loop)
    check('reduce gets fp32',                 t_reduce_gets_fp32)

    print('variant')
    check('shares the loaded model',          t_variant_shares_the_model)
    check('dtype lands on ModelConfig',       t_variant_rewrites_dtype_on_the_nested_config)
    check('refuses what would rebuild',       t_variant_refuses_what_rebuilds)

    print('ModelOutputSpec')
    check('refuses impossible shapes',        t_output_spec_refuses_nonsense)

    print('cfg.pooling')
    check('__call__ follows it',              t_call_follows_the_configured_pooling)
    check('an alias does not fork the id',    t_pooling_alias_does_not_fork_the_id)
    check('an unknown one dies at config',    t_unknown_pooling_dies_at_config_time)
    check('feature_spec describes features',  t_feature_spec_describes_what_features_returned)
    check('a reducing head does not confuse it',
          t_a_head_that_reduces_does_not_confuse_the_pooling)
    check("tokens_spec is the model's own",   t_tokens_spec_is_the_model_s_own)

    print('identity')
    check('batch size does not split it',     t_identity_ignores_batch_size)
    check('a transform change does',          t_identity_moves_on_transform)

    print('pooling')
    check('shapes and slot names agree',      t_shapes_and_slots)
    check('every slot is unit norm',          t_every_slot_is_unit_norm)
    check('cls slot is the cls token',        t_cls_slot_is_just_the_cls_token)
    check('register tokens are dropped',      t_registers_are_dropped)
    check('grid blocks average the right cells', t_grid_blocks_average_the_right_cells)
    check('rings hold equal counts',          t_rings_hold_equal_counts)
    check('ring mask follows the tokens',     t_rings_follow_the_tokens_to_the_gpu)
    check('bad inputs are refused',           t_bad_inputs_are_refused)
    check('spec subscripts like the dict',    t_outputspec_and_storemeta_subscript_the_same)

    print('pooling with no prefix token')
    check('slot 0 is the average',            t_no_prefix_makes_slot_0_the_average)
    check('cls_avg is refused',               t_no_prefix_refuses_the_mode_that_would_duplicate)
    check('every cell is kept',               t_no_prefix_keeps_every_cell_as_a_patch)

    print('model_token_spec')
    check('unwraps DataParallel',             t_spec_unwraps_dataparallel)
    check('refuses a classifier head',        t_spec_refuses_a_classifier)

    print('config fields are honoured  (builds vit_tiny, 5 MB)')
    check('identity moves only where it should', t_identity_moves_only_where_it_should)
    check('weights= is actually loaded',      t_weights_field_is_actually_loaded)

    bad = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(bad)}/{len(_RESULTS)} passed')
    if bad:
        print('failed: ' + ', '.join(bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
