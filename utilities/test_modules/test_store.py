#!/usr/bin/env python3
"""Tests for utilities/Store.py -- feature stores, the feature-map cache, and
pre-tile rungs.

    python utilities/test_modules/test_store.py

No slide, no model, no GPU: every store lives in a temporary directory, the
encoder and the container are stand-ins, and the features are ramps whose value
says where they came from.

WHAT THIS DEFENDS
-----------------
    addressing    a writer and a reader computing the same path from the same
                  values -- and two draws, two masks or two grids landing in
                  two places, never one
    validation    the shapes and labels that load cleanly and mean something
                  else: slots out of step with the tensor, a tokens store that
                  cannot say where its cells sat, a draw passed off as the slide
    the gate      a feature map restored onto a mask it was not built from --
                  the one error the address cannot catch alone
    pre-tiles     a rung is a dataset only once its index is written, and a
                  finished one is never appended to
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402

from PatchingLib import FeaturesMap, WsiFeaturesMap              # noqa: E402
from Store import (FeatureMapCache, FeatureStore as FS,          # noqa: E402
                   PreTileCorpus, PreTileStore, StoreMismatch,
                   from_store_tensors, geometry_mismatch, region_grids,
                   to_store_tensors)
from TissueMask import TissueRegion                              # noqa: E402

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def rejects(fn, needle: str = '') -> str:
    try:
        fn()
    except Exception as e:                                       # noqa: BLE001
        if needle and needle not in str(e):
            raise AssertionError(f'raised, but never mentions {needle!r}: {e}') from None
        return 'refused'
    raise AssertionError('should have raised, returned normally')


# ══════════════════════════════════════════════════════════════════════════════
#  feature store
# ══════════════════════════════════════════════════════════════════════════════

def a_meta(**over) -> FS.Meta:
    """A four-tile, two-slot sample store."""
    base = dict(wsi_stem='SLIDE_A', wsi_path='/data/g/SLIDE_A.svs', level=1,
                ds=4.00003, mpp=1.0, base_mpp=0.25, tile_size=256, overlap=False,
                pooling='cls_avg', slots=('cls', 'avg'), slot_layout='none', dim=8,
                feat_hw=(14, 14), num_prefix=1, encoder_id='enc12345',
                seg_id='hest-e3b0c442', region_id='abcd1234', coverage='sample',
                n_available=100, n_tiles=4, sampler_id='samp0001', plan='native',
                sample_seed=7, buckets=('bg00_15', 'bg15_30'))
    base.update(over)
    return FS.Meta(**base)


def tensors_for(meta: FS.Meta, **over) -> dict:
    n, k = meta.n_tiles, len(meta.slots)
    t = dict(features=(torch.arange(n * k * meta.dim, dtype=torch.float32)
                       .reshape(n, k, meta.dim) / 100).half(),
             x=torch.arange(n, dtype=torch.int32) * 1024,
             y=torch.zeros(n, dtype=torch.int32))
    t.update(over)
    return t


def t_meta_round_trips_through_strings():
    m = a_meta()
    s = m.to_strings()
    assert set(s) == {f.name for f in dataclasses.fields(m)}, 'a field has no codec'
    assert FS.Meta.from_strings(s) == m, 'a value changed across the round trip'
    assert a_meta(ds=4.00003).to_strings()['ds'] != a_meta(ds=4.0).to_strings()['ds']
    rejects(lambda: a_meta(slots=('cls', 'a,b')).to_strings(), 'comma')
    return 'every field, 4.00003 kept apart from 4'


def t_path_is_the_key():
    """Writer and reader compute one path from the same values; every value
    that makes a different store moves it."""
    root = Path('/r')
    m = a_meta()
    want = root / 'hest-e3b0c442/SLIDE_A/abcd1234/samp0001_native/ds4.00003_cls_avg.safetensors'
    assert FS.path(root, m) == want, FS.path(root, m)
    base = FS.path(root, m)
    for over in (dict(seg_id='hsv-1'), dict(region_id='ffff'), dict(sampler_id='s2'),
                 dict(plan='native-L0'), dict(ds=4.0), dict(pooling='tokens'),
                 dict(wsi_stem='SLIDE_B')):
        assert FS.path(root, dataclasses.replace(m, **over)) != base, over
    grid = dataclasses.replace(m, coverage='grid')
    assert FS.path(root, grid).parent.name == 'grid-t256-o0'
    return 'each identity field moves it'


def t_save_load_round_trip():
    with tempfile.TemporaryDirectory() as root:
        m = a_meta()
        path = FS.save(root, meta=m, extra={'fov_id': torch.arange(4, dtype=torch.int32)},
                       **tensors_for(m))
        got, meta = FS.load(path)
        assert meta.created_at, 'created_at not stamped'
        assert dataclasses.replace(meta, created_at='') == m
        assert got['features'].shape == (4, 2, 8) and got['fov_id'].tolist() == [0, 1, 2, 3]
        assert 'region' not in got, 'a draw was given grid columns it does not have'
        part, _ = FS.load(path, keys=('x',))
        assert set(part) == {'x'}
        assert not [p for p in Path(root).rglob('*') if p.name.endswith('.tmp')]
        assert FS.levels(path.parent, 'cls_avg') == {1: path}
    return 'core, extra, keys=, no temp left'


def t_require_refuses_and_names_the_field():
    with tempfile.TemporaryDirectory() as root:
        m = a_meta()
        path = FS.save(root, meta=m, **tensors_for(m))
        FS.load(path, require={'encoder_id': 'enc12345', 'coverage': 'sample'})
        try:
            FS.load(path, require={'encoder_id': 'other'})
        except StoreMismatch as e:
            assert 'encoder_id' in str(e), str(e)
            return 'refused, field named'
    raise AssertionError('a mismatched encoder_id loaded')


def t_save_refuses_what_would_load_wrong():
    with tempfile.TemporaryDirectory() as root:
        m = a_meta()
        t = tensors_for(m)
        rejects(lambda: FS.save(root, meta=m, **dict(t, features=t['features'].float())), 'fp16')
        rejects(lambda: FS.save(root, meta=a_meta(slots=('cls',)), **t), 'slot names')
        rejects(lambda: FS.save(root, meta=a_meta(dim=9), **t), 'meta.dim')
        rejects(lambda: FS.save(root, meta=m, **dict(t, x=t['x'].long())), 'int32')
        rejects(lambda: FS.save(root, meta=a_meta(coverage='all'), **t), 'coverage')
        rejects(lambda: FS.save(root, meta=a_meta(sample_seed=None), **t), 'sample_seed')
        rejects(lambda: FS.save(root, meta=a_meta(sampler_id=''), **t), 'sampler_id')
        rejects(lambda: FS.save(root, meta=m, extra={'x': torch.zeros(4)}, **t), 'collides')
        # A grid store IS the slide: it must carry the grid columns and every tile.
        g = a_meta(coverage='grid', n_available=4, sampler_id='', plan='', sample_seed=None)
        rejects(lambda: FS.save(root, meta=g, **t), 'region')
        cols = dict(region=torch.zeros(4, dtype=torch.int16),
                    grid_rc=torch.zeros(4, 2, dtype=torch.int32))
        FS.save(root, meta=g, **t, **cols)
        rejects(lambda: FS.save(root, meta=dataclasses.replace(g, n_available=5), **t, **cols),
                'n_available')
    return 'dtype, slots, dim, coverage, collisions, a partial grid'


def t_a_tokens_store_says_where_its_cells_sat():
    with tempfile.TemporaryDirectory() as root:
        tok = a_meta(pooling='tokens', slots=tuple(f'c{i}' for i in range(4)),
                     slot_layout='grid:2x2', feat_hw=(2, 2))
        FS.save(root, meta=tok, **tensors_for(tok))
        rejects(lambda: FS.save(root, meta=dataclasses.replace(tok, feat_hw=None),
                                **tensors_for(tok)), 'feat_hw')
        rejects(lambda: FS.save(root, meta=dataclasses.replace(tok, feat_hw=(4, 1)),
                                **tensors_for(tok)), 'implies')
        q = dataclasses.replace(tok, pooling='query_tokens')
        rejects(lambda: FS.save(root, meta=dataclasses.replace(q, feat_hw=None),
                                **tensors_for(q)), 'feat_hw')
    return 'tokens and query_tokens both held to feat_hw'


# ══════════════════════════════════════════════════════════════════════════════
#  the feature-map cache: WsiFeaturesMap <-> a grid store, and the gate
# ══════════════════════════════════════════════════════════════════════════════

DS, LEVEL, TILE, DIM = 4.0, 1, 256, 8

#: Three regions sized so their grids differ -- equal sizes would let a swap
#: of two pass. The small one is 2x1 tiles: a 1x1 region has no order to get
#: wrong, and the round trip's "roll the rows" decoy would be the identity.
REGIONS = [
    TissueRegion(x=0,     y=0,     w=int(TILE * DS * 4), h=int(TILE * DS * 2), index=0),
    TissueRegion(x=8192,  y=4096,  w=int(TILE * DS * 2), h=int(TILE * DS * 3), index=1),
    TissueRegion(x=20480, y=12288, w=int(TILE * DS * 2), h=int(TILE * DS * 1), index=2),
]


def a_wfm(regions=REGIONS, ds=DS, level=LEVEL, overlap=True) -> WsiFeaturesMap:
    """Features are a ramp: region r, patch i, channel c -> 1000 r + i + c/1000,
    so a mis-ordered restore prints as a recognisable difference."""
    maps = []
    for r, g in enumerate(region_grids(regions, ds=ds, level=level,
                                       tile_size=TILE, overlap=overlap)):
        base = torch.arange(len(g), dtype=torch.float32).unsqueeze(1) + 1000.0 * r
        maps.append(FeaturesMap(g, base + torch.arange(DIM).float() * 0.001))
    return WsiFeaturesMap(regions, maps, ds=ds, level=level, tile_size=TILE,
                          overlap=overlap)


def t_map_round_trip_beats_a_decoy():
    """fp16 moves a value by ~1e-3; the check is against rolling the rows by
    one, which a mis-ordered restore would look like."""
    wfm = a_wfm()
    tensors = to_store_tensors(wfm)
    back = from_store_tensors(tensors, REGIONS, ds=DS, level=LEVEL, tile_size=TILE,
                              overlap=True)
    a = torch.cat([m.features for m in wfm])
    b = torch.cat([m.features for m in back])
    gap = float((a - b).abs().max())
    decoy = float((a - torch.roll(b, 1, dims=0)).abs().max())
    assert gap * 100 < decoy, (gap, decoy)
    return f'gap {gap:.2g} against decoy {decoy:.2g}'


def t_gate_accepts_the_same_geometry_and_nothing_else():
    tensors = to_store_tensors(a_wfm())
    same = region_grids(REGIONS, ds=DS, level=LEVEL, tile_size=TILE, overlap=True)
    assert geometry_mismatch(tensors, same) == []
    moved = [TissueRegion(r.x + 1024, r.y, r.w, r.h, r.index) for r in REGIONS]
    for name, grids in (
            ('one fewer region', region_grids(REGIONS[:2], ds=DS, level=LEVEL, tile_size=TILE, overlap=True)),
            ('a moved region', region_grids(moved, ds=DS, level=LEVEL, tile_size=TILE, overlap=True)),
            ('another scale', region_grids(REGIONS, ds=DS * 4, level=LEVEL + 1, tile_size=TILE, overlap=True)),
            ('no overlap grid', region_grids(REGIONS, ds=DS, level=LEVEL, tile_size=TILE, overlap=False))):
        assert geometry_mismatch(tensors, grids), f'{name} was accepted'
    return 'fewer, moved, rescaled and un-overlapped all refused'


class _Encoder:
    feature_pooling = 'cls'
    model_spec = SimpleNamespace(feat_hw=(14, 14), num_prefix=1)

    def __init__(self, ident='enc12345'):
        self.ident = ident

    def identity_id(self):
        return self.ident


class _Recipe:
    def __init__(self, seg='hest-e3b0c442', region='abcd1234'):
        self.seg, self.region = seg, region

    def seg_id(self):
        return self.seg

    def region_id(self):
        return self.region


def _container(regions=REGIONS):
    return SimpleNamespace(tile_size=TILE, overlap=True, ds=DS, level=LEVEL,
                           tissue_regions=regions)


def t_map_cache_hits_and_misses_for_the_right_reasons():
    with tempfile.TemporaryDirectory() as root:
        cache = FeatureMapCache(root, '/data/g/SLIDE_A.svs', _Encoder(), _Recipe(),
                                verbose=False)
        assert cache.load(_container()) is None, 'a hit before any write'
        path = cache.save(a_wfm())
        assert path == cache.path(_container()), (path, cache.path(_container()))
        assert path.parts[-5:-1] == ('hest-e3b0c442', 'SLIDE_A', 'abcd1234',
                                     'grid-t256-o1'), path
        back = cache.load(_container())
        assert back is not None and back.n_patches() == a_wfm().n_patches()
        # the address is right, the encoder is not: a miss
        other = FeatureMapCache(root, '/data/g/SLIDE_A.svs', _Encoder('enc99999'),
                                _Recipe(), verbose=False)
        assert other.load(_container()) is None, 'another encoder was served'
        # the address is right, the regions are not: the gate
        assert cache.load(_container(REGIONS[:2])) is None, 'narrowed regions served'
        # another recipe is another address
        moved = FeatureMapCache(root, '/data/g/SLIDE_A.svs', _Encoder(),
                                _Recipe(region='ffff0000'), verbose=False)
        assert moved.path(_container()) != path
        ro = FeatureMapCache(root, '/data/g/SLIDE_A.svs', _Encoder(), _Recipe(),
                             mode='r', verbose=False)
        assert ro.save(a_wfm()) is None, "mode='r' wrote"
    return 'hit, encoder miss, gate miss, recipe moves the address'


# ══════════════════════════════════════════════════════════════════════════════
#  pre-tiles
# ══════════════════════════════════════════════════════════════════════════════

TILE_P, FACTOR = 64, 3


def a_corpus(root, **over) -> PreTileCorpus:
    base = dict(root=Path(root), seg_id='hest-e3b0c442', region_id='abcd1234',
                sampler_id='samp0001', plan='ladder-1-2', factor=FACTOR)
    base.update(over)
    return PreTileCorpus(**base)


def a_pre_meta(corpus, ds=1.0, stem='SLIDE_A') -> PreTileStore.Meta:
    return PreTileStore.Meta(wsi_stem=stem, ds=ds, tile=TILE_P, seg_id=corpus.seg_id,
                             region_id=corpus.region_id, sampler_id=corpus.sampler_id,
                             plan=corpus.plan, pre_tile_factor=corpus.factor)


def a_pre(value: int) -> np.ndarray:
    side = TILE_P * FACTOR
    img = np.zeros((side, side, 3), np.uint8)
    img[..., 0] = value
    return img


def t_corpus_addresses_by_every_key():
    base = a_corpus('/r')
    want = Path('/r/hest-e3b0c442/SLIDE_A/abcd1234_samp0001_ladder-1-2/f3/ds4')
    assert base.rung_dir('SLIDE_A', 4.0) == want, base.rung_dir('SLIDE_A', 4.0)
    for over in (dict(seg_id='hsv-1'), dict(region_id='ffff'), dict(sampler_id='s2'),
                 dict(plan='ladder-1'), dict(factor=5)):
        assert a_corpus('/r', **over).set_dir('SLIDE_A') != base.set_dir('SLIDE_A'), over
    return 'mask, draw, plan and factor each move it'


def t_rung_round_trip_and_the_index_is_last():
    with tempfile.TemporaryDirectory() as root:
        corpus = a_corpus(root)
        meta = a_pre_meta(corpus)
        folder = PreTileStore.create(corpus, meta)
        records = [PreTileStore.Record(index=i, x=i * 100, y=5, bucket='bg00_15',
                                       inherit_id=i % 2) for i in range(3)]
        for r in records:
            PreTileStore.save_tile(folder, r, a_pre(10 + r.index), meta)
        assert corpus.rung_dirs('SLIDE_A') == [], 'a rung counted before its index'
        PreTileStore.write_index(folder, records)
        assert corpus.rung_dirs('SLIDE_A') == [folder]
        assert corpus.slides() == ['SLIDE_A']
        back = PreTileStore.load_meta(folder, require={'sampler_id': 'samp0001'})
        assert back.n_tiles == 3 and back.created_at, back
        assert PreTileStore.load_index(folder) == records
        assert int(PreTileStore.read_tile(folder, records[2])[0, 0, 0]) == 12
        rejects(lambda: PreTileStore.create(corpus, meta), 'finished')
        rejects(lambda: PreTileStore.load_meta(folder, require={'plan': 'other'}), 'plan')
    return '3 tiles; unfinished invisible; finished refuses a second write'


def t_rungs_sort_by_ds_and_a_wrong_size_is_refused():
    with tempfile.TemporaryDirectory() as root:
        corpus = a_corpus(root)
        for ds in (16.0, 1.0, 4.0):
            meta = a_pre_meta(corpus, ds=ds)
            folder = PreTileStore.create(corpus, meta)
            PreTileStore.write_index(folder, [])
        assert [d.name for d in corpus.rung_dirs('SLIDE_A')] == ['ds1', 'ds4', 'ds16']
        meta = a_pre_meta(corpus, ds=2.0)
        folder = PreTileStore.create(corpus, meta)
        rejects(lambda: PreTileStore.save_tile(
            folder, PreTileStore.Record(index=0, x=0, y=0),
            np.zeros((TILE_P * FACTOR - 2,) * 2 + (3,), np.uint8), meta), 'meta says')
    return 'ds1 < ds4 < ds16, not lexical; a short pre-tile refused'


def t_pixels_come_back_byte_exact_in_rgb_order():
    """Exact, not close: a lossy codec passes any tolerance and still rings at
    every 8x8 block edge, which is a corner. Noise, because an R/B swap (save
    writes BGR, read reverses it) survives an all-equal check on a grey tile."""
    image = np.random.default_rng(0).integers(
        0, 256, (TILE_P * FACTOR,) * 2 + (3,), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as root:
        corpus = a_corpus(root)
        meta = a_pre_meta(corpus)
        folder = PreTileStore.create(corpus, meta)
        record = PreTileStore.Record(index=3, x=1024, y=2048)
        PreTileStore.save_tile(folder, record, image, meta)
        PreTileStore.write_index(folder, [record])
        assert (PreTileStore.read_tile(folder, record) == image).all(), (
            'pixels changed: a lossy codec, or the channel order reversed once')
    return f'{image.shape[0]} px noise, exact'


def t_every_record_axis_survives_the_index():
    """bucket / score / overlap_max / inherit_id / origin / parent: properties
    of the RUN that placed the tile, none recoverable from (x, y) afterwards."""
    want = [PreTileStore.Record(index=0, x=100, y=200, clip_px=37, bucket='bg50_70',
                                score=0.83, overlap_max=0.25, inherit_id=7,
                                origin='jitter', parent_x=36, parent_y=200),
            PreTileStore.Record(index=1, x=300, y=400, bucket='bg00_15', score=0.02)]
    with tempfile.TemporaryDirectory() as root:
        corpus = a_corpus(root)
        folder = PreTileStore.create(corpus, a_pre_meta(corpus))
        PreTileStore.write_index(folder, want)
        assert PreTileStore.load_index(folder) == want
        assert PreTileStore.load_meta(folder).n_clipped == 1
    return 'every column exact, n_clipped counted'


def t_level0_geometry_is_self_consistent():
    """`x`/`y` are the TILE's top-left, the pre-tile's is `margin_l0` up-left,
    the centre half a footprint in -- three expressions, one point."""
    meta = dataclasses.replace(a_pre_meta(a_corpus('/r')), ds=4.0)
    record = PreTileStore.Record(index=0, x=10_000, y=20_000)
    assert meta.tile_footprint_l0 == TILE_P * 4.0
    px, py = record.pre_origin_l0(meta)
    cx, cy = record.centre_l0(meta)
    half = meta.tile_footprint_l0 * FACTOR / 2
    assert abs(px + half - cx) < 1e-6 and abs(py + half - cy) < 1e-6, (px, cx)
    return f'margin {meta.margin_l0:g} L0'


def t_a_corpus_key_round_trips_and_the_meta_spells_the_same():
    corpus = a_corpus('/r')
    assert PreTileCorpus.from_key('/r', corpus.key) == corpus
    assert a_pre_meta(corpus).corpus_key == corpus.key
    rejects(lambda: PreTileCorpus.from_key('/r', 'not-a-key'), 'not a pre-tile')
    return corpus.key


def t_unfinished_is_refused_and_overwrite_needs_saying():
    with tempfile.TemporaryDirectory() as root:
        corpus = a_corpus(root)
        meta = a_pre_meta(corpus)
        folder = PreTileStore.create(corpus, meta)
        rejects(lambda: PreTileStore.load_index(folder), 'did not finish')
        PreTileStore.write_index(folder, [])
        assert PreTileStore.create(corpus, meta, overwrite=True) == folder
    return 'no index refused; overwrite=True reopens the same rung'


# ══════════════════════════════════════════════════════════════════════════════

_TESTS = [t for n, t in sorted(globals().items()) if n.startswith('t_')]


def main() -> int:
    for fn in _TESTS:
        check(fn.__name__[2:].replace('_', ' '), fn)
    failed = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
