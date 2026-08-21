#!/usr/bin/env python3
"""Unit test for the TileEncoderFunc template.

    python utilities/test_modules/test_tile_encoder.py

No GPU, no download, about a second. Three fake models stand in for the three
output kinds, which is the whole point of the exercise: a base class written
against one implementation looks correct until the second one arrives, and this
repo has watched that happen twice in a day -- num_classes=0 was GigaPath's
config rather than the encoder's, and a one-tile region turned a decoy into the
identity.

So the template is checked against a vector model, a token model and a spatial
model before any of them exist for real.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _d in ('aiNNModel', 'utilities'):
    p = str(_ROOT / _d)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

from ConfigIdentity import ModelConfig                      # noqa: E402
from TileEncoderFunc import (ModelOutputSpec, TileEncoder,       # noqa: E402
                             TileEncoderConfig, TransformConfig)

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

    bad = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(bad)}/{len(_RESULTS)} passed')
    if bad:
        print('failed: ' + ', '.join(bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
