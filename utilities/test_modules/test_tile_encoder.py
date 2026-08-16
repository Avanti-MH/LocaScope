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
import torch.nn.functional as F                             # noqa: E402

from ConfigIdentity import ModelConfig                      # noqa: E402
from TileEncoderFunc import (OutputSpec, TileEncoder,       # noqa: E402
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

    def build(self, device, **kw):
        return FakeEncoder(self, device)


class FakeEncoder(TileEncoder):
    BASELINE = {'kind': 'vector',
                'model': ModelConfig(source='local', arch='x:Y', dtype='fp32'),
                'transform': TransformConfig(scale_size=8, crop_size=8)}

    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        self.model = {'vector': _Vector, 'tokens': _Tokens,
                      'spatial': _Spatial}[cfg.kind]().eval()
        self.spec = {
            'vector':  OutputSpec('vector', DIM),
            'tokens':  OutputSpec('tokens', DIM, GRID, PREFIX),
            'spatial': OutputSpec('spatial', DIM, (2, 2)),
        }[cfg.kind]
        self._transform = cfg.transform.build()
        self._weights_id = None


class TokenEncoder(FakeEncoder):
    """A token model that HAS decided which token, the way a real one must."""
    def _vector_from(self, raw):
        return F.normalize(raw[:, 0], dim=-1)


def tiles(n=5):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(n)]


def enc(kind, cls=FakeEncoder, **over):
    cfg = FakeConfig(kind=kind,
                     model=ModelConfig(source='local', arch='x:Y', dtype='fp32'),
                     **over)
    e = cls(cfg, torch.device('cpu'))
    return e


# ── features works for every kind ─────────────────────────────────────────────

def t_features_for_vector_and_spatial():
    """The two kinds whose reduction is arithmetic, so the base can do them."""
    for kind in ('vector', 'spatial'):
        f = enc(kind).features(tiles())
        assert f.shape == (5, DIM), f'{kind}: {tuple(f.shape)}'
        norms = f.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f'{kind}: features are not unit norm'


def t_features_refuses_to_guess_for_tokens():
    """A token model with no reduction declared must not silently pick one.

    Guessing here would be answering a question the base cannot know -- GigaPath
    validated the CLS against a stored tensor, and another model may not have --
    and the guess would be invisible in every number downstream.
    """
    rejects(lambda: enc('tokens').features(tiles()), '_vector_from')


def t_features_for_a_token_model_that_decided():
    f = TokenEncoder(FakeConfig(kind='tokens',
                                model=ModelConfig(source='local', arch='x:Y',
                                                  dtype='fp32')),
                     torch.device('cpu')).features(tiles())
    assert f.shape == (5, DIM), tuple(f.shape)


# ── capabilities are gated, and say what the model is ─────────────────────────

def t_tokens_only_for_token_models():
    for kind in ('vector', 'spatial'):
        e = enc(kind)
        rejects(lambda: e.tokens(tiles()), kind)
        rejects(lambda: e.pooled(tiles(), 'cls'), kind)


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


# ── OutputSpec ────────────────────────────────────────────────────────────────

def t_output_spec_refuses_nonsense():
    rejects(lambda: OutputSpec('token', 8), 'kind must be')
    rejects(lambda: OutputSpec('tokens', 8), 'patch grid')
    rejects(lambda: OutputSpec('vector', 8, (2, 2)), 'no grid')
    assert OutputSpec('tokens', 8, (14, 14), 1).n_tokens() == 197
    rejects(lambda: OutputSpec('vector', 8).n_tokens(), 'no token count')


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

    print('OutputSpec')
    check('refuses impossible shapes',        t_output_spec_refuses_nonsense)

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
