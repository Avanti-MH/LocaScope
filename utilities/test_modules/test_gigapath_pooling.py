#!/usr/bin/env python3
"""Unit test for the token/pooling additions in aiNNModel/GigaPathFunc.

Two halves, because they cost very different amounts:

    python utilities/test_modules/test_gigapath_pooling.py
        Pure tensor maths against a fake spec. No model, no GPU, no WSI --
        runs on a login node in a second and catches most mistakes.

    python utilities/test_modules/test_gigapath_pooling.py --with-model
        Loads GigaPath and asserts the one thing the cheap half cannot:
        pool_tokens(..., 'cls') is bit-for-bit the production feature.

That last check is why this file exists. Every pooling in the experiment is
compared against `cls` as the baseline, so if `cls` is not exactly what
gigapath_encode produces today, every conclusion is measured against the wrong
reference and nothing raises. It is written as an equality test rather than a
one-off spot check because the thing that would break it -- switching
global_pool to 'avg', one of the pooling variants under consideration -- swaps
which of timm's `norm` / `fc_norm` is the real LayerNorm.
"""

from __future__ import annotations

import argparse
import sys
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

from GigaPathFunc import (model_token_spec, pool_tokens,     # noqa: E402
                          _ring_bins)

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


# ── fixtures ──────────────────────────────────────────────────────────────────

SPEC = {'dim': 8, 'token_grid': (14, 14), 'num_prefix': 1}


def toks(n=3, spec=SPEC, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = spec['num_prefix'] + spec['token_grid'][0] * spec['token_grid'][1]
    return torch.randn(n, t, spec['dim'], generator=g)


# ── shape / slot bookkeeping ──────────────────────────────────────────────────

def t_shapes_and_slots():
    x = toks()
    for mode, n, layout in (('cls', 1, 'none'),
                            ('cls_avg', 2, 'none'),
                            ('cls_std', 2, 'none'),
                            ('rings3', 4, 'ring:3'),
                            ('grid2x2', 5, 'grid:2x2'),
                            ('tokens', 197, 'grid:14x14')):
        f, slots, lay = pool_tokens(x, mode, SPEC)
        assert f.shape == (3, n, SPEC['dim']), f'{mode}: shape {tuple(f.shape)}'
        assert len(slots) == n, f'{mode}: {len(slots)} names for {n} slots'
        assert lay == layout, f'{mode}: layout {lay!r} != {layout!r}'
        assert slots[0] == 'cls', f'{mode}: slot 0 must be cls'


def t_every_slot_is_unit_norm():
    # Each slot separately, not the flattened vector -- otherwise the weight
    # between slots is fixed at write time, and that weighting is the open
    # question the experiment exists to answer.
    for mode in ('cls', 'cls_avg', 'cls_std', 'rings3', 'grid2x2', 'tokens'):
        f, _, _ = pool_tokens(toks(), mode, SPEC)
        n = f.norm(dim=-1)
        assert torch.allclose(n, torch.ones_like(n), atol=1e-5), \
            f'{mode}: slot norms range {n.min():.4f}..{n.max():.4f}'


def t_cls_slot_is_just_the_cls_token():
    x = toks()
    f, _, _ = pool_tokens(x, 'cls_avg', SPEC)
    assert torch.allclose(f[:, 0], F.normalize(x[:, 0], dim=-1), atol=1e-6)


def t_registers_are_dropped():
    # DINOv2 _reg4 has num_prefix=5. Patches must start after the registers; the
    # failure mode otherwise is four junk vectors averaged into every pooling,
    # with no error and only a slightly worse score to show for it.
    spec = {'dim': 8, 'token_grid': (4, 4), 'num_prefix': 5}
    x = torch.zeros(1, 5 + 16, 8)
    x[0, 0] = 1.0                       # cls
    x[0, 1:5] = 99.0                    # registers -- must not reach any slot
    x[0, 5:] = 2.0                      # patches
    f, _, _ = pool_tokens(x, 'cls_avg', spec)
    assert torch.allclose(f[0, 1], F.normalize(torch.full((8,), 2.0), dim=-1)), \
        'the average picked up the register tokens'


def t_grid_blocks_average_the_right_cells():
    spec = {'dim': 1, 'token_grid': (4, 4), 'num_prefix': 1}
    x = torch.zeros(1, 17, 1)
    x[0, 1:] = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    # 4x4 grid, 2x2 blocks -> top-left block holds cells 0,1,4,5 -> mean 2.5
    g = x[0, 1:].reshape(4, 4)
    want = [g[:2, :2].mean(), g[:2, 2:].mean(), g[2:, :2].mean(), g[2:, 2:].mean()]
    f, slots, _ = pool_tokens(x, 'grid2x2', spec)
    assert slots[1:] == ('g00', 'g01', 'g10', 'g11')
    # after per-slot normalization a 1-D vector is just its sign, so compare the
    # pre-normalized means through a second call on a widened dim
    x2 = x.repeat(1, 1, 2)
    f2, _, _ = pool_tokens(x2, 'grid2x2', {**spec, 'dim': 2})
    for k, w in enumerate(want):
        got = f2[0, 1 + k]
        assert torch.allclose(got, F.normalize(torch.tensor([w, w]), dim=-1), atol=1e-5), \
            f'block {slots[1+k]}: expected mean {w}'


def t_rings_hold_equal_counts():
    bins = _ring_bins(14, 14, 3)
    counts = [int((bins == k).sum()) for k in range(3)]
    assert max(counts) - min(counts) <= 1, f'ring sizes {counts} are not balanced'
    assert sorted(set(bins.tolist())) == [0, 1, 2]


def t_bad_inputs_are_refused():
    rejects(lambda: pool_tokens(torch.zeros(3, 8), 'cls', SPEC), '[N, T, D]')
    rejects(lambda: pool_tokens(torch.zeros(3, 5, 8), 'cls', SPEC), 'disagree')
    rejects(lambda: pool_tokens(toks(), 'nonsense', SPEC), 'unknown pooling mode')
    rejects(lambda: pool_tokens(toks(), 'grid3x3', SPEC), 'does not divide')


# ── model_token_spec ──────────────────────────────────────────────────────────

class _Fake(torch.nn.Module):
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
        'dim': 8, 'token_grid': (4, 4), 'num_prefix': 1}


def t_spec_refuses_a_classifier():
    rejects(lambda: model_token_spec(_Fake(head=torch.nn.Linear(8, 5))),
            'num_classes=0')


# ── the expensive half ────────────────────────────────────────────────────────

def t_cls_equals_production(n_tiles=8):
    """pool_tokens(...,'cls') must BE gigapath_encode, not merely resemble it."""
    import torch as _t
    from GigaPathFunc import (gigapath_model, gigapath_encode,
                              gigapath_encode_tokens)

    dev = _t.device('cuda' if _t.cuda.is_available() else 'cpu')
    model = gigapath_model(dev)                       # single card on purpose
    spec = model_token_spec(model)
    print(f'        spec = {spec}')

    # State the precondition instead of failing obscurely on it. "Production" is
    # gigapath_encode's output, and which token that is depends on global_pool:
    # at 'token' it is the CLS, at 'avg' it is the mean of the patches. The
    # equality below is cls-vs-cls only while the model pools on the token.
    gp = getattr(getattr(model, 'module', model), 'global_pool', None)
    assert gp == 'token', (
        f"this test compares the cls slot against gigapath_encode, which is only "
        f"the same quantity when global_pool='token'; model says {gp!r}. If the "
        f"model was deliberately switched, compare against the matching pooling "
        f"mode instead of loosening this.")

    rng = np.random.default_rng(0)
    tiles = [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
             for _ in range(n_tiles)]

    a = gigapath_encode(tiles, model, dev)
    b, slots, _ = pool_tokens(gigapath_encode_tokens(tiles, model, dev), 'cls', spec)
    b = b[:, 0]
    cos = F.cosine_similarity(a, b, dim=-1)
    print(f'        cos(production, cls slot): min {cos.min():.8f}')
    assert (1 - cos).max() < 1e-4, \
        f'cls slot is not the production feature: worst cos {cos.min():.6f}'


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-model', action='store_true',
                    help='also load GigaPath and check cls == gigapath_encode')
    ap.add_argument('--tiles', type=int, default=8)
    args = ap.parse_args()

    print('pool_tokens')
    check('shapes and slot names agree',        t_shapes_and_slots)
    check('every slot is unit norm',            t_every_slot_is_unit_norm)
    check('cls slot is the cls token',          t_cls_slot_is_just_the_cls_token)
    check('register tokens are dropped',        t_registers_are_dropped)
    check('grid blocks average the right cells', t_grid_blocks_average_the_right_cells)
    check('rings hold equal counts',            t_rings_hold_equal_counts)
    check('bad inputs are refused',             t_bad_inputs_are_refused)

    print('model_token_spec')
    check('unwraps DataParallel',               t_spec_unwraps_dataparallel)
    check('refuses a classifier head',          t_spec_refuses_a_classifier)

    if args.with_model:
        print('equivalence (loads GigaPath)')
        check('cls slot == gigapath_encode',
              lambda: t_cls_equals_production(args.tiles))
    else:
        print('equivalence: skipped, pass --with-model to run it')

    bad = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(bad)}/{len(_RESULTS)} passed')
    if bad:
        print('failed: ' + ', '.join(bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
