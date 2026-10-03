#!/usr/bin/env python3
"""Tests for utilities/Store.py -- feature stores, the feature-map cache, and
pre-tile rungs.

    python utilities/test_modules/test_store.py
    python utilities/test_modules/test_store.py --with-model [--only-model]

The unit tests need no slide, no model, no GPU: every store lives in a
temporary directory, the encoder and the container are stand-ins, and the
features are ramps whose value says where they came from.

`--with-model` adds one measurement that does need them (see `run_precision`):
how far a pooling computed live is from the same pooling taken from a stored
raw output. It prints a distribution and fails only on a value fp16 cannot hold.

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
    raw           the model's own output stored whole -- prefix slots and cells,
                  in the model's order -- and told apart from the reduced
                  'tokens'; the dtype the tensor has is the dtype it keeps; a
                  chunked read is the whole read, cut up; several poolings of
                  one slide sit side by side and neither overwrites the other
"""

from __future__ import annotations

import argparse
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
                   PooledFeatures, PreTileCorpus, PreTileStore, StoreMismatch,
                   from_store_tensors, geometry_mismatch, raw_layout,
                   region_grids, to_store_tensors)
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
        rejects(lambda: FS.save(root, meta=m, **dict(t, features=t['features'].double())), 'fp16')
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
    return 'dtype (fp64), slots, dim, coverage, collisions, a partial grid'


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
#  raw outputs, dtype, chunked reads, several poolings
# ══════════════════════════════════════════════════════════════════════════════

def a_raw_meta(**over) -> FS.Meta:
    """3 prefix slots (cls + 2 registers) and a 2x2 cell grid: 7 slots."""
    base = dict(pooling='raw', slots=('cls', 'r0', 'r1', 'p0', 'p1', 'p2', 'p3'),
                slot_layout=raw_layout(3, (2, 2)), feat_hw=(2, 2), num_prefix=3)
    base.update(over)
    return a_meta(**base)


def slot_ramp(meta: FS.Meta, dtype=torch.float16) -> dict:
    """features[:, k, :] == k, so a slot in the wrong place prints as itself."""
    n, k = meta.n_tiles, len(meta.slots)
    f = torch.arange(k, dtype=torch.float32).reshape(1, k, 1).expand(n, k, meta.dim)
    return tensors_for(meta, features=f.to(dtype).contiguous())


def t_a_raw_store_keeps_every_slot_in_the_models_order():
    with tempfile.TemporaryDirectory() as root:
        m = a_raw_meta()
        path = FS.save(root, meta=m, **slot_ramp(m))
        got, meta = FS.load(path)
        assert meta.slots == m.slots and meta.slot_layout == 'raw:3+2x2', meta
        assert got['features'].shape == (4, 7, 8)
        assert [int(got['features'][0, k, 0]) for k in range(7)] == list(range(7)), (
            'a slot moved: registers and cells are in the model\'s order')
        assert path.name == 'ds4.00003_raw.safetensors', path.name
        # the three values written by three lines have to agree
        t = slot_ramp(m)
        rejects(lambda: FS.save(root, meta=a_raw_meta(slot_layout=raw_layout(1, (2, 2))), **t),
                'imply')
        rejects(lambda: FS.save(root, meta=a_raw_meta(feat_hw=None), **t), 'feat_hw')
        rejects(lambda: FS.save(root, meta=a_raw_meta(slot_layout='none'), **t), 'raw store says')
        rejects(lambda: FS.save(root, meta=a_meta(slot_layout=raw_layout(3, (2, 2))),
                                **tensors_for(a_meta())), 'raw store says')
        six = a_raw_meta(slots=('cls', 'r0', 'r1', 'p0', 'p1', 'p2'))
        rejects(lambda: FS.save(root, meta=six, **slot_ramp(six)), 'has 7 slots, got 6')
    return '7 slots in order; layout, feat_hw, prefix and count must agree'


def t_the_dtype_a_tensor_has_is_the_dtype_it_keeps():
    with tempfile.TemporaryDirectory() as root:
        m = a_meta()
        t16 = tensors_for(m)
        f32 = (torch.arange(4 * 2 * 8, dtype=torch.float32).reshape(4, 2, 8) / 3.0)
        p16 = FS.save(root, meta=m, **t16)
        assert FS.load(p16)[0]['features'].dtype == torch.float16
        p32 = FS.save(root, meta=a_meta(pooling='cls_std'),
                      **dict(t16, features=f32))
        back = FS.load(p32)[0]['features']
        assert back.dtype == torch.float32 and torch.equal(back, f32), (
            'an fp32 store was rounded on its way to disk')
        assert f32.half().float().sub(f32).abs().max() > 0, (
            'the values are fp16-exact, so this proved nothing about fp32')
        # nothing about the dtype is in the metadata, so files written before
        # this existed read as they did
        assert set(m.to_strings()) == {f.name for f in dataclasses.fields(m)}
        assert not any('dtype' in k for k in m.to_strings())
    return 'fp16 stays fp16; fp32 comes back bit-exact and differs from its fp16 rounding'


def t_a_chunked_read_is_the_whole_read_cut_up():
    with tempfile.TemporaryDirectory() as root:
        m = a_meta(n_tiles=10)
        path = FS.save(root, meta=m, **tensors_for(m))
        whole = FS.load(path)[0]['features']
        assert FS.n_rows(path) == 10
        for rows in (1, 3, 4, 10, 100):
            got = list(FS.iter_chunks(path, rows=rows))
            assert [a for a, _ in got] == list(range(0, 10, rows)), rows
            assert torch.equal(torch.cat([c for _, c in got]), whole), rows
        assert torch.equal(torch.cat([c for _, c in FS.iter_chunks(path, key='x', rows=4)]),
                           FS.load(path)[0]['x'])
        rejects(lambda: list(FS.iter_chunks(path, rows=0)), 'positive')
        rejects(lambda: list(FS.iter_chunks(path, require={'encoder_id': 'other'})),
                'encoder_id')
    return 'rows 1, 3, 4, 10, 100 all give the same tensor; uneven tail included'


def _pooled_for(container=None, spec=(14, 14), prefix=1, dim=DIM):
    """cls and raw of the whole slide, sized from the grid."""
    n = a_wfm().n_patches()
    cells = spec[0] * spec[1] + prefix
    raw = (torch.arange(n * cells * dim, dtype=torch.float32)
           .reshape(n, cells, dim) / 7.0).half()
    cls = raw[:, :1].clone()
    return {'cls': PooledFeatures(cls, ('cls',), 'none'),
            'raw': PooledFeatures(raw, tuple(f's{i}' for i in range(cells)),
                                  raw_layout(prefix, spec))}, n


def t_several_poolings_of_one_slide_sit_side_by_side():
    pooled, n = _pooled_for()
    with tempfile.TemporaryDirectory() as root:
        cache = FeatureMapCache(root, '/data/g/SLIDE_A.svs', _Encoder(), _Recipe(),
                                verbose=False)
        paths = cache.save_pooled(_container(), pooled)
        assert set(paths) == {'cls', 'raw'} and paths['cls'] != paths['raw']
        assert paths['raw'].name == 'ds4_raw.safetensors', paths['raw'].name
        assert paths['raw'].parent == paths['cls'].parent, 'not in one key directory'
        assert cache.check(_container(), 'raw').slot_layout == 'raw:1+14x14'
        got, meta = cache.load_pooled(_container(), 'raw')
        assert torch.equal(got['features'], pooled['raw'].features)
        assert meta.pooling == 'raw' and meta.n_tiles == n
        chunks = cache.iter_pooled(_container(), 'raw', rows=7)
        assert torch.equal(torch.cat([c for _, c in chunks]), pooled['raw'].features)
        got_cls, _ = cache.load_pooled(_container(), 'cls')
        assert torch.equal(got_cls['features'], pooled['cls'].features), (
            'writing raw overwrote cls')
        # the one-vector API reads what save_pooled wrote for the same pooling
        assert cache.load(_container()).n_patches() == n
        # the checks every read makes
        assert cache.check(_container(REGIONS[:2]), 'raw') is None, 'narrowed regions served'
        other = FeatureMapCache(root, '/data/g/SLIDE_A.svs', _Encoder('enc99999'),
                                _Recipe(), verbose=False)
        assert other.check(_container(), 'raw') is None, 'another encoder was served'
        assert cache.check(_container(), 'rings3') is None, 'a pooling nobody wrote'
        assert cache.iter_pooled(_container(), 'rings3') is None
        ro = FeatureMapCache(root, '/data/g/SLIDE_A.svs', _Encoder(), _Recipe(),
                             mode='r', verbose=False)
        assert ro.save_pooled(_container(), pooled) == {}, "mode='r' wrote"
        bad = PooledFeatures(pooled['cls'].features[:-1], ('cls',), 'none')
        rejects(lambda: cache.save_pooled(_container(), {'cls': bad}), 'grid')
    return 'cls and raw coexist; encoder, regions and pooling are checked'


def t_an_fp32_encoder_writes_fp32_and_anything_else_writes_fp16():
    def encoder_at(dtype):
        e = _Encoder()
        e.cfg = SimpleNamespace(model=SimpleNamespace(torch_dtype=lambda: dtype))
        return e

    seen = {}
    for name, enc in (('fp32', encoder_at(torch.float32)),
                      ('fp16', encoder_at(torch.float16)),
                      ('unsaid', _Encoder())):
        with tempfile.TemporaryDirectory() as root:
            cache = FeatureMapCache(root, '/data/g/SLIDE_A.svs', enc, _Recipe(),
                                    verbose=False)
            path = cache.save(a_wfm())
            seen[name] = FS.load(path)[0]['features'].dtype
            assert cache.load(_container()) is not None, name
    assert seen == {'fp32': torch.float32, 'fp16': torch.float16,
                    'unsaid': torch.float16}, seen
    return 'fp32 encoder -> fp32; fp16 and an encoder that does not say -> fp16'


# ══════════════════════════════════════════════════════════════════════════════
#  --with-model: live against cached, on real tiles
# ══════════════════════════════════════════════════════════════════════════════
#
# A pooling computed live comes from the encoder's fp32 output. The same pooling
# taken from a stored raw output comes from that output rounded to the storage
# dtype -- through the REAL Store: FeatureStore.save, then load. This measures
# how far apart the two land, per pooling, against two yardsticks:
#
#     G_store   live  vs  cached-from-raw      what the file costs
#     G_direct  live  vs  the arm stored at fp16 itself (normalised, then cast)
#     G_run     live  vs  the SAME tiles encoded a second time
#
# G_run is the floor. A GPU does not promise the same bits twice, so a gap no
# larger than G_run is not a difference at all. Gaps are 1 - cos over every
# (tile, slot) pair; the tail matters more than the median, because a rank flip
# is a tail event.
#
# It also reports the largest |value| by token type (cls, registers, patches)
# against fp16's 65504 -- registers are where large activations live -- and
# FAILS only if the cast to fp16 produced an inf or a nan, which no threshold
# could excuse. Everything else is printed for a human: no distribution has been
# seen yet, so nothing is asserted about it.

POOL_ARMS = ('cls', 'cls_avg', 'cls_std', 'rings3', 'grid2x2')


def _sample_tiles(slide, mask, level, n, rng, tile=256):
    """`n` tissue tiles of a slide at `level`, random places inside its regions,
    background under half. Fewer come back if the slide has no room."""
    ds = float(slide.level_downsamples[level])
    side = int(tile * ds)
    regions = [r for r in mask.tissue_regions if r.w > side and r.h > side]
    if not regions:
        return []
    xy = []
    for _ in range(20 * n):
        r = regions[int(rng.integers(len(regions)))]
        xy.append((int(r.x + rng.integers(0, r.w - side)),
                   int(r.y + rng.integers(0, r.h - side))))
    xy = np.array(xy, dtype=np.int64)
    keep = xy[np.asarray(mask.white_fractions(xy, level, tile)) < 0.5][:n]
    return [np.asarray(slide.read_region_rgb((int(x), int(y)), level, (tile, tile)))
            for x, y in keep]


def _gap(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """1 - cos over every (tile, slot); both are unit slots."""
    return (1.0 - (a.float() * b.float()).sum(-1)).clamp_min(0.0).flatten().numpy()


def _row(name, comparison, gap):
    q = np.percentile(gap, [50, 95, 99]) if len(gap) else [float('nan')] * 3
    return dict(group=name, comparison=comparison, n=len(gap), median=float(q[0]),
                p95=float(q[1]), p99=float(q[2]),
                max=float(gap.max()) if len(gap) else float('nan'))


def run_precision(args) -> int:
    import csv
    from AccessDatasets import locate
    from SafeSlide import SafeSlide
    from TileEncoderFunc import admissible_poolings, encoder_config, pooling_kinds
    import Cache
    from TissueMaskConfig import MASK_RECIPES, MaskMaker
    from _paths import job_result_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = encoder_config(args.encoder, batch_size=args.batch_size).with_model(dtype='fp16')
    arms, dropped = admissible_poolings(cfg, POOL_ARMS)
    encoder = cfg.build(device)
    spec = encoder.model_spec
    print(f'encoder {args.encoder} on {device}: T={spec.n_tokens()} D={spec.dim} '
          f'prefix={spec.num_prefix} cells={spec.feat_hw}   arms {list(arms)}'
          + (f'  (cannot do {dropped})' if dropped else ''))
    p = int(spec.num_prefix)
    slots = (('cls',) + tuple(f'r{k}' for k in range(p - 1)) if p else ()) + tuple(
        f'p{k:03d}' for k in range(spec.feat_hw[0] * spec.feat_hw[1]))

    rng = np.random.default_rng(args.seed)
    rows, ranges, bad_values = [], [], 0
    pooled = {a: {'store': [], 'direct': [], 'run': []} for a in arms}
    # The mask is read from a cache when there is one: a hit builds no
    # segmenter and reads no pixels. `--mask-cache-job` names whose cache.
    masks_root = Cache.cache_root(
        args.mask_cache_job or Cache.job_name('StoreTest'), 'mask')
    print(f'masks: --seg {args.seg}, cache {masks_root}')
    with MaskMaker(MASK_RECIPES[args.seg], masks_root, device) as masks:
        for name in args.slides:
            slide = SafeSlide(locate(name).path)
            mask, hit = masks.mask(slide)
            print(f'  {name}: mask {"from the cache" if hit else "segmented now"}')
            for level in args.levels:
                if level >= slide.level_count:
                    continue
                tiles = _sample_tiles(slide, mask, level, args.tiles, rng)
                tag = f'{name} L{level}'
                if not tiles:
                    print(f'  {tag}: no tissue tile fits -- skipped')
                    continue
                raw = encoder.tokens(tiles)                       # [N, T, D] fp32
                again = encoder.tokens(tiles)                     # the floor
                half = raw.to(torch.float16)
                bad = int((~torch.isfinite(half)).sum())
                bad_values += bad
                for kind, sl in (('cls', slice(0, 1)),
                                 ('registers', slice(1, p)),
                                 ('patches', slice(p, None))):
                    part = raw[:, sl].abs()
                    if part.numel():
                        ranges.append(dict(group=tag, token=kind,
                                           max_abs=float(part.max()),
                                           over_60000=int((part > 60000).sum())))
                # through the real Store, exactly as a cache would keep it
                with tempfile.TemporaryDirectory() as root:
                    meta = FS.Meta(
                        wsi_stem=name, wsi_path='', level=level, ds=1.0, mpp=0.0,
                        base_mpp=0.0, tile_size=256, overlap=False, pooling='raw',
                        slots=slots, slot_layout=raw_layout(p, spec.feat_hw),
                        dim=int(spec.dim), feat_hw=tuple(spec.feat_hw), num_prefix=p,
                        encoder_id=encoder.identity_id(), seg_id='precision',
                        region_id='precision', coverage='sample',
                        n_available=len(tiles), n_tiles=len(tiles),
                        sampler_id='precision', plan='precision', sample_seed=0)
                    # two tensors, not one passed twice: safetensors refuses
                    # tensors that share memory
                    path = FS.save(root, meta=meta, features=half,
                                   x=torch.zeros(len(tiles), dtype=torch.int32),
                                   y=torch.zeros(len(tiles), dtype=torch.int32))
                    stored = FS.load(path)[0]['features']
                    chunked = torch.cat([c for _, c in FS.iter_chunks(path, rows=97)])
                    assert torch.equal(stored, chunked), 'chunked read differs on real data'
                stored = stored.float()
                for a in arms:
                    live = pooling_kinds(raw, a, spec)
                    g_store = _gap(live, pooling_kinds(stored, a, spec))
                    g_direct = _gap(live, live.to(torch.float16).float())
                    g_run = _gap(live, pooling_kinds(again, a, spec))
                    pooled[a]['store'].append(g_store)
                    pooled[a]['direct'].append(g_direct)
                    pooled[a]['run'].append(g_run)
                    for cmp_name, g in (('store', g_store), ('direct', g_direct),
                                        ('run', g_run)):
                        rows.append(_row(f'{tag} {a}', cmp_name, g))
                print(f'  {tag}: {len(tiles)} tiles, non-finite after fp16: {bad}',
                      flush=True)
            slide.close()

    print('\n1 - cos to the live pooling, every (tile, slot); smaller is closer')
    print(f'{"arm":<10}{"vs":<8}{"n":>8}{"median":>12}{"p95":>12}{"p99":>12}{"max":>12}')
    for a in arms:
        for cmp_name, label in (('run', 'G_run'), ('direct', 'G_direct'),
                                ('store', 'G_store')):
            g = np.concatenate(pooled[a][cmp_name]) if pooled[a][cmp_name] else np.zeros(0)
            r = _row(a, cmp_name, g)
            print(f'{a:<10}{label:<8}{r["n"]:>8}{r["median"]:>12.2e}{r["p95"]:>12.2e}'
                  f'{r["p99"]:>12.2e}{r["max"]:>12.2e}')
            rows.append(dict(r, group=f'ALL {a}'))
        print()
    print('largest |value| in the raw output, by token type (fp16 holds 65504)')
    worst = {}
    for r in ranges:
        w = worst.setdefault(r['token'], dict(max_abs=0.0, over=0))
        w['max_abs'] = max(w['max_abs'], r['max_abs'])
        w['over'] += r['over_60000']
    for kind, w in worst.items():
        print(f'  {kind:<10}max {w["max_abs"]:>10.1f}   values above 60000: {w["over"]}')

    out = job_result_dir('StoreTest')
    path = os.path.join(out, f'precision_{args.encoder}.csv')
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['group'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n  {path}')
    if bad_values:
        print(f'\nFAIL: {bad_values} value(s) are not finite once cast to fp16 -- '
              f'a raw store at fp16 cannot hold this encoder\'s output')
        return 1
    print('\nno inf or nan after the cast. Read G_store against G_run: a gap no '
          'larger than the floor is not a difference.')
    return 0


# ══════════════════════════════════════════════════════════════════════════════

_TESTS = [t for n, t in sorted(globals().items()) if n.startswith('t_')]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--with-model', action='store_true',
                    help='also measure live against cached on real tiles')
    ap.add_argument('--only-model', action='store_true',
                    help='skip the unit tests (implies --with-model)')
    ap.add_argument('--slides', nargs='+', default=['BRACS_1411', 'S1104233,G7E,110208'],
                    help='slide NAMES, resolved by AccessDatasets. The defaults '
                         'have a hest mask in result/cache/MppRoutingHead_mask')
    ap.add_argument('--seg', default='hest', help='mask recipe (MASK_RECIPES)')
    ap.add_argument('--mask-cache-job', default=None,
                    help='whose mask cache to read and fill: result/cache/<this>_'
                         'mask/. Default: this job (StoreTest)')
    ap.add_argument('--encoder', default='uni2')
    ap.add_argument('--levels', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--tiles', type=int, default=200, help='tiles per (slide, level)')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    status = 0
    if not args.only_model:
        for fn in _TESTS:
            check(fn.__name__[2:].replace('_', ' '), fn)
        failed = [n for n, e in _RESULTS if e is not None]
        print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
        if failed:
            print('failed: ' + ', '.join(failed))
        status = 1 if failed else 0
    if args.with_model or args.only_model:
        print('\n======== live against cached ========')
        status = max(status, run_precision(args))
    return status


if __name__ == '__main__':
    sys.exit(main())
