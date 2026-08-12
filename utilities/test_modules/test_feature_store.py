#!/usr/bin/env python3
"""Unit test for utilities/FeatureStore.

No GPU, no WSI, no model -- it writes a few small stores to a temp directory and
reads them back, so it runs on a login node in about a second:

    python utilities/test_modules/test_feature_store.py

Most of what is checked here is refusal rather than function. A feature store
that loads cleanly but means something else is the failure this module exists to
prevent, so the tests that matter are the ones asserting that save() and
load(require=...) say no.
"""

from __future__ import annotations

import dataclasses
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import torch                                              # noqa: E402
import FeatureStore as FS                                 # noqa: E402


# ── fixtures ──────────────────────────────────────────────────────────────────

def a_meta(**over) -> FS.StoreMeta:
    base = dict(
        wsi_stem='S1104360,G7E,110208', wsi_path='/data/S1104360.mrxs', level=1,
        ds=2.0, mpp=0.4851, base_mpp=0.24255, tile_size=256, overlap=True,
        pooling='cls_avg', slots=('cls', 'avg'), slot_layout='none',
        dim=8, token_grid=(14, 14), num_prefix=1,
        encoder_id='prov-gigapath@fp16', mask_id='hest@ds4',
        coverage='sample', n_available=1000, sample_seed=7, n_tiles=4,
    )
    base.update(over)
    return FS.StoreMeta(**base)


def tensors_for(meta: FS.StoreMeta, **over):
    n = meta.n_tiles
    t = dict(
        features=torch.zeros(n, len(meta.slots), meta.dim, dtype=torch.float16),
        x=torch.arange(n, dtype=torch.int32),
        y=torch.arange(n, dtype=torch.int32),
        region=torch.zeros(n, dtype=torch.int16),
        grid_rc=torch.zeros(n, 2, dtype=torch.int32),
    )
    t.update(over)
    return t


# ── harness ───────────────────────────────────────────────────────────────────

_RESULTS = []


def check(name: str, fn) -> None:
    try:
        fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}')
    except Exception as e:                                # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n        {type(e).__name__}: {e}')


def rejects(fn, needle: str = '') -> None:
    """Assert fn() raises, and that the message names the problem."""
    try:
        fn()
    except Exception as e:                                # noqa: BLE001
        if needle and needle not in str(e):
            raise AssertionError(
                f'raised, but the message does not mention {needle!r}: {e}') from None
        return
    raise AssertionError('should have raised, returned normally')


# ── metadata ──────────────────────────────────────────────────────────────────

def t_roundtrip():
    m = a_meta()
    s = m.to_strings()
    assert FS.StoreMeta.from_strings(s) == m, 'value changed across the round trip'
    # A field added to the dataclass but not to the codec would vanish here, and
    # then require= could never check it.
    assert set(s) == {f.name for f in dataclasses.fields(m)}, 'codec missed a field'


def t_unknown_annotation_is_loud():
    rejects(lambda: FS._codec('SomeTypeWeNeverHandled'), 'no metadata codec')


def t_cfg_hash_identity_only():
    m = a_meta()
    h = m.cfg_hash()
    for over in (dict(n_tiles=999), dict(sample_seed=1), dict(created_at='2030'),
                 dict(level=0), dict(pooling='tokens'), dict(ds=99.0)):
        assert dataclasses.replace(m, **over).cfg_hash() == h, \
            f'{over} should not change the config hash'
    for over in (dict(encoder_id='other'), dict(mask_id='mask_all'),
                 dict(tile_size=512), dict(overlap=False), dict(base_mpp=0.5)):
        assert dataclasses.replace(m, **over).cfg_hash() != h, \
            f'{over} should change the config hash'


def t_float_format_is_stable():
    # cfg_hash feeds on the encoded strings, so two spellings of one value must
    # not produce two hashes.
    assert a_meta(base_mpp=0.24255).cfg_hash() == a_meta(base_mpp=0.242550).cfg_hash()


def t_slot_name_with_comma_refused():
    rejects(lambda: a_meta(slots=('cls', 'a,b')).to_strings(), 'comma')


# ── save validation ───────────────────────────────────────────────────────────

def t_save_load_roundtrip(root):
    m = a_meta()
    p = FS.save(root, meta=m, **tensors_for(m))
    assert p.name == m.filename()
    got, meta2 = FS.load(p)
    assert meta2.created_at, 'created_at should be stamped on save'
    assert dataclasses.replace(meta2, created_at='') == dataclasses.replace(m, created_at='')
    for k in FS.CORE_TENSORS:
        assert k in got, f'{k} missing after load'
    assert got['features'].shape == (4, 2, 8)
    assert not list(Path(root).glob('*.tmp')), 'a .tmp file survived the write'


def t_save_rejects_bad_shapes(root):
    m = a_meta()
    rejects(lambda: FS.save(root, meta=m,
                            **tensors_for(m, features=torch.zeros(4, 8, dtype=torch.float16))),
            '[N, n, D]')
    rejects(lambda: FS.save(root, meta=m,
                            **tensors_for(m, features=torch.zeros(4, 2, 8))),
            'fp16')
    rejects(lambda: FS.save(root, meta=a_meta(slots=('cls',)), **tensors_for(m)),
            'slot names')
    rejects(lambda: FS.save(root, meta=a_meta(dim=99), **tensors_for(m)), 'meta.dim')
    rejects(lambda: FS.save(root, meta=m,
                            **tensors_for(m, x=torch.arange(4, dtype=torch.int64))),
            'x must be')


def t_save_rejects_bad_coverage(root):
    m = a_meta()
    rejects(lambda: FS.save(root, meta=a_meta(coverage='partial'), **tensors_for(m)),
            'coverage must be')
    rejects(lambda: FS.save(root, meta=a_meta(sample_seed=None), **tensors_for(m)),
            'sample_seed')
    # coverage='all' has to mean all of them, or a reader searches a fraction of
    # the slide and reports nothing found.
    rejects(lambda: FS.save(root, meta=a_meta(coverage='all', n_available=1000),
                            **tensors_for(m)),
            'n_available')
    ok = a_meta(coverage='all', n_available=4, sample_seed=None)
    FS.save(root, meta=ok, **tensors_for(ok))


def t_save_rejects_slot_layout_mismatch(root):
    m = a_meta(pooling='grid2x2', slots=('cls', 'g00', 'g01', 'g10', 'g11'),
               slot_layout='grid:2x2')
    FS.save(root, meta=m, **tensors_for(m))                     # 4 + cls == 5, fine
    bad = dataclasses.replace(m, slot_layout='grid:3x3')
    rejects(lambda: FS.save(root, meta=bad, **tensors_for(m)), 'implies')


def t_extra_tensors(root):
    m = a_meta(pooling='queries')
    extra = {'ans_main': torch.arange(4, dtype=torch.int32),
             'delta': torch.zeros(4, 2, dtype=torch.int16)}
    p = FS.save(root, meta=m, extra=extra, **tensors_for(m))
    got, _ = FS.load(p)
    assert got['ans_main'].tolist() == [0, 1, 2, 3]
    rejects(lambda: FS.save(root, meta=m, extra={'x': torch.zeros(4)},
                            **tensors_for(m)),
            'collides')


# ── read ──────────────────────────────────────────────────────────────────────

def t_require_refuses(root):
    m = a_meta(pooling='tokens', slots=('cls', 'p0'), coverage='sample')
    p = FS.save(root, meta=m, **tensors_for(m))
    FS.load(p, require={'pooling': 'tokens', 'coverage': 'sample'})   # matches
    try:
        FS.load(p, require={'coverage': 'all', 'encoder_id': 'nope'})
    except FS.StoreMismatch as e:
        msg = str(e)
        assert 'coverage' in msg and 'encoder_id' in msg, \
            f'the error should name every mismatch, got: {msg}'
        return
    raise AssertionError('load should have refused a store that does not match')


def t_load_meta_and_keys(root):
    m = a_meta(pooling='cls', slots=('cls',))
    p = FS.save(root, meta=m, **tensors_for(m))
    assert FS.load_meta(p).pooling == 'cls'
    got, _ = FS.load(p, keys=('x', 'y'))
    assert set(got) == {'x', 'y'}, 'keys= should not read the rest'


def t_find(root):
    for lvl in (0, 1):
        for pool in ('cls', 'tokens'):
            m = a_meta(level=lvl, pooling=pool,
                       slots=('cls',) if pool == 'cls' else ('cls', 'p0'))
            FS.save(root, meta=m, **tensors_for(m))
    def n(**q):
        return len(FS.find(root, **q))
    assert n(level=1) == 2, f'level=1 should match 2, got {n(level=1)}'
    assert n(pooling='tokens') == 2, f'pooling=tokens should match 2, got {n(pooling="tokens")}'
    assert n(level=0, pooling='cls') == 1, 'level and pooling should intersect'
    assert FS.find(root, wsi_stem='nope') == [], 'unknown stem should match nothing'
    (Path(root) / 'unrelated.safetensors').write_bytes(b'not ours')
    assert n(level=1) == 2, 'a foreign .safetensors should be skipped, not counted'


def t_find_one(root):
    """Exactly one, or an error that names the candidates.

    The failure this guards is silent: a root can hold two stores for the same
    slide and level that differ only in how their tiles were chosen, and
    hits[0] would pick whichever sorted first. The test therefore checks the
    MESSAGE too -- an exception that does not say which files matched leaves the
    caller no way to disambiguate, which is barely better than the wrong pick.
    """
    for sid in ('', 'aaaa1111'):
        m = a_meta(level=0, pooling='tokens', slots=('cls', 'p0'),
                   sampler_id=sid)
        FS.save(root, meta=m, **tensors_for(m))

    one = FS.find_one(root, level=0, pooling='tokens', sampler_id='aaaa1111')
    assert FS.load_meta(one).sampler_id == 'aaaa1111', 'picked the wrong store'

    try:
        FS.find_one(root, level=0, pooling='tokens')
    except FS.StoreMismatch as e:
        msg = str(e)
        assert '2 stores' in msg, f'the count is not in the message: {msg}'
        for h in FS.find(root, level=0, pooling='tokens'):
            assert h.name in msg, f'{h.name} is not named in the message'
        assert 'sampler_id' in msg, 'the message does not say how to separate them'
    else:
        raise AssertionError('two matches should refuse, not pick one')

    rejects(lambda: FS.find_one(root, level=9), 'no store')


def t_on_grid_count_is_what_is_bounded(root):
    """A store may hold more tiles than the grid offers, but not more that CLAIM
    to come from it.

    The guard exists for the two ways the tiles and the count come from
    different places -- a stale level index, an accumulator not cleared between
    levels. Displacement made the naive form fire on normal operation, so it is
    scoped by origin: 0 is a grid position, anything else was synthesised.
    """
    import torch
    m = a_meta(coverage='sample', n_available=4, sample_seed=1)
    t = tensors_for(m)                                   # 4 tiles

    # three from the grid, one displaced: fine against a grid of four, and
    # fine even against a grid of three.
    origin = torch.tensor([0, 0, 0, 1], dtype=torch.int8)
    FS.save(root, meta=a_meta(coverage='sample', n_available=3, sample_seed=1),
            extra={'origin': origin}, **t)

    # all four claim the grid, but it only offers three
    rejects(lambda: FS.save(
        root, meta=a_meta(coverage='sample', n_available=3, sample_seed=1),
        extra={'origin': torch.zeros(4, dtype=torch.int8)}, **t),
        'came from the grid')

    # without origin the old rule stands: every tile is assumed on-grid
    rejects(lambda: FS.save(
        root, meta=a_meta(coverage='sample', n_available=3, sample_seed=1), **t),
        'sampled 4 of 3')


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print('FeatureStore')
    print('  metadata')
    check('round-trip covers every field', t_roundtrip)
    check('unknown annotation is loud', t_unknown_annotation_is_loud)
    check('cfg_hash tracks identity fields only', t_cfg_hash_identity_only)
    check('float encoding is stable', t_float_format_is_stable)
    check('slot name with a comma is refused', t_slot_name_with_comma_refused)

    root = Path(tempfile.mkdtemp(prefix='featurestore_test_'))
    try:
        # A directory per test. Sharing one made t_find count the stores every
        # earlier test had left behind -- the kind of cross-talk that makes a
        # passing suite mean less than it looks.
        def in_own_dir(fn, name):
            d = root / name.replace(' ', '_')[:40]
            d.mkdir(parents=True, exist_ok=True)
            return lambda: fn(d)

        print(f'  save/load   ({root})')
        for name, fn in (
            ('save then load returns what went in', t_save_load_roundtrip),
            ('bad shapes and dtypes are refused',   t_save_rejects_bad_shapes),
            ('coverage must agree with n_available', t_save_rejects_bad_coverage),
            ('only on-grid tiles are bounded by the grid',
             t_on_grid_count_is_what_is_bounded),
            ('slot_layout must agree with n',       t_save_rejects_slot_layout_mismatch),
            ('extra tensors, and name collisions',  t_extra_tensors),
        ):
            check(name, in_own_dir(fn, name))

        print('  read')
        for name, fn in (
            ('require refuses and names every mismatch', t_require_refuses),
            ('load_meta and partial keys',               t_load_meta_and_keys),
            ('find matches on metadata',                 t_find),
            ('find_one refuses an ambiguous match',      t_find_one),
        ):
            check(name, in_own_dir(fn, name))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    bad = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(bad)}/{len(_RESULTS)} passed')
    if bad:
        print('failed: ' + ', '.join(bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
