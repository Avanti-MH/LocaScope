"""The stores: encoded tile features, and pre-tiles. One module, one rule.

    from Store import FeatureStore as FS, PreTileStore

Both follow `Cache.py`'s layout -- `<root>/<one level per config the content
depends on>/`, upstream above downstream -- and both are addressed BY KEY: a
writer computes the path from what made the content, and a reader computes the
same path from the same values. Neither searches its root by metadata. The
metadata is still written, and still checked on every read (`require=`), but
it is the check, not the address.

    features   <root>/<seg_id>/<slide>/<region_id>/<key>/ds<d>_<pooling>.safetensors
                 root = Cache.cache_root(<job>, 'features') / encoder_tag
                 key  = grid-t<tile>-o<0|1>        every tile of every region
                        <sampler_id>_<plan>        a TileSampler draw
    pretiles   <root>/<seg_id>/<slide>/<region_id>_<sampler_id>_<plan>/f<factor>/ds<d>/
                 root = Cache.cache_root(<job>, 'pretiles')
                 meta.json (written first) + PNGs + index.csv (written LAST)

Every directory level is named by what that level's content depends on, and
the mask's keys come first because everything below was cut from it: dropping a
recipe is one `rm -rf <seg_id>/`.

WHY ONE MODULE
--------------
FeatureStore, WsiFeaturesMapStore and PreTileStore were three modules with three
copies of the same machinery -- tmp + rename, a hand-rolled identity hash in the
filename, a `find()` that opened every file under a root to read its header,
and a `find_one()` that existed because `find()` could return two. The
addressing is what made them need all that: a hashed name nobody could
recompute. Addressed by key, the hash, the search and the "which of these two
did you mean" error all go, and what is left is small enough to share a file.

WsiFeaturesMapStore is not a third store. It is the grid-coverage case of the
feature store plus the conversion to and from `WsiFeaturesMap` -- which is what
`FeatureMapCache` below is.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from Cache import atomic_file
from TileSampler import PRE_TILE_FACTOR, centre_margin, pre_tile_px


class StoreMismatch(RuntimeError):
    """A store exists but is not the one asked for, or is incomplete."""


def _stamp() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S')


def _require(meta, require: Optional[Dict[str, object]], where) -> None:
    """Refuse, naming every field, when `meta` is not what was asked for. Never
    falls back: the alternative is the wrong data and a number that merely
    looks disappointing."""
    if not require:
        return
    bad = {k: (v, getattr(meta, k, '<no such field>'))
           for k, v in require.items() if getattr(meta, k, None) != v}
    if bad:
        lines = '\n'.join(f'  {k}: wanted {w!r}, store has {g!r}'
                          for k, (w, g) in sorted(bad.items()))
        raise StoreMismatch(f'{where}\n{lines}')


def _ds_name(ds: float) -> str:
    """`ds<d>`: `:g` keeps a level's own 4.00003 apart from a rung's 4 -- they
    are not the same pixels -- and prints a rung as the number it is."""
    return f'ds{float(ds):g}'


# ══════════════════════════════════════════════════════════════════════════════
#  features
# ══════════════════════════════════════════════════════════════════════════════
#
#   features   [N, n, D] fp16    n = len(slots); n = 1 still uses 3 dims so
#                                readers never branch on shape
#   x, y       [N] int32         level-0 top-left of each tile
#   region     [N] int16         grid coverage only: which region
#   grid_rc    [N, 2] int32      grid coverage only: (row, col) in its grid
#
# plus any `extra` tensors the writer passes -- a query store carries its
# answer indices into the reference store beside it this way.

#: Slot names that hold the tile's single summary vector rather than a cell.
#: Which one appears is the model's property: a ViT's summary is its CLS token,
#: a CNN's the global average of its feature map.
SUMMARY_SLOTS = frozenset({'cls', 'gap'})

_CORE = ('features', 'x', 'y', 'region', 'grid_rc')


def _enc_float(v: float) -> str:
    return f'{v:.12g}'


def _enc_slots(v: Tuple[str, ...]) -> str:
    for s in v:
        if ',' in s:
            raise ValueError(f'slot name may not contain a comma: {s!r}')
    return ','.join(v)


#: safetensors metadata is str -> str, so every field has an explicit codec,
#: dispatched on the annotation. A field whose type has none raises on first
#: use rather than being silently dropped from what `require=` can check.
_CODECS = {
    'str':                      (lambda v: v, lambda s: s),
    'int':                      (str, int),
    'float':                    (_enc_float, float),
    'bool':                     (lambda v: '1' if v else '0', lambda s: s == '1'),
    'Optional[int]':            (lambda v: '' if v is None else str(v),
                                 lambda s: None if s == '' else int(s)),
    'Tuple[str,...]':           (_enc_slots,
                                 lambda s: tuple(s.split(',')) if s else ()),
    'Optional[Tuple[int,int]]': (lambda v: '' if v is None else f'{v[0]}x{v[1]}',
                                 lambda s: None if s == ''
                                 else tuple(int(p) for p in s.split('x'))),
}


def _codec(annotation: str):
    try:
        return _CODECS[annotation.replace(' ', '')]
    except KeyError:
        raise TypeError(f'FeatureMeta field annotated {annotation!r} has no '
                        f'metadata codec; add one to _CODECS') from None


@dataclass(frozen=True)
class FeatureMeta:
    """Everything needed to decide whether a feature file is the one wanted.

    The address holds the identity (see the module docstring); these fields
    repeat it so a read can check it, and add what the address does not say.
    """
    wsi_stem:    str
    wsi_path:    str
    level:       int
    ds:          float
    mpp:         float
    base_mpp:    float
    tile_size:   int
    overlap:     bool

    # what is in `features`
    pooling:     str                        # 'tokens' | 'cls' | 'query_tokens' | ...
    slots:       Tuple[str, ...]            # len == n
    slot_layout: str                        # 'none' | 'grid:2x2' | 'ring:3' | ...
    dim:         int                        # D

    #: The ENCODER that wrote this, not the slots above: a pooling='cls' file
    #: still carries (14, 14) -- one vector, from a model with a 14x14 grid.
    #: feat_hw rather than token_grid because a CNN's output has no tokens.
    feat_hw:     Optional[Tuple[int, int]]
    num_prefix:  int                        # CLS + registers; 0 if none

    # what produced it
    encoder_id:  str                        # TileEncoder.identity_id()
    seg_id:      str                        # TissueMaskConfig.seg_id()
    region_id:   str                        # TissueMaskConfig.region_id()

    # which tiles
    coverage:    str                        # 'grid' | 'sample'
    n_available: int                        # grid: tiles the grid offers
    n_tiles:     int                        # N
    sampler_id:  str = ''                   # sample: SamplerConfig.sampler_id()
    plan:        str = ''                   # sample: PlanSpec.key()
    sample_seed: Optional[int] = None
    #: sample: the richness bucket names, in the order an int8 `bucket` extra
    #: column indexes them -- the names live in the sampler's config, which
    #: the file does not carry.
    buckets:     Tuple[str, ...] = ()

    created_at:  str = ''

    @property
    def key(self) -> str:
        """The coverage directory: `grid-t<tile>-o<0|1>` or `<sampler_id>_<plan>`."""
        if self.coverage == 'grid':
            return FeatureStore.grid_key(self.tile_size, self.overlap)
        return FeatureStore.sample_key(self.sampler_id, self.plan)

    def __getitem__(self, key: str):
        """Field access by name, so `pooling_kinds` takes a FeatureMeta as-is --
        it reads dim / feat_hw / num_prefix off a live ModelOutputSpec or off
        this, spelled the same."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def to_strings(self) -> Dict[str, str]:
        return {f.name: _codec(f.type)[0](getattr(self, f.name))
                for f in dataclasses.fields(self)}

    @classmethod
    def from_strings(cls, d: Dict[str, str]) -> 'FeatureMeta':
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in d:
                raise StoreMismatch(f'feature metadata has no {f.name!r} -- '
                                    f'not written by this module')
            kwargs[f.name] = _codec(f.type)[1](d[f.name])
        return cls(**kwargs)


class FeatureStore:
    """Encoded tile features, one safetensors file per (slide, ds, pooling)."""

    Meta = FeatureMeta

    # ── addressing ──────────────────────────────────────────────────────────

    @staticmethod
    def grid_key(tile_size: int, overlap: bool) -> str:
        return f'grid-t{int(tile_size)}-o{int(bool(overlap))}'

    @staticmethod
    def sample_key(sampler_id: str, plan: str) -> str:
        if not sampler_id or not plan:
            raise ValueError('a sample key needs both a sampler_id and a plan')
        return f'{sampler_id}_{plan}'

    @staticmethod
    def key_dir(root, *, seg_id: str, slide: str, region_id: str, key: str) -> Path:
        """The directory one draw (or one grid) of one slide lives in -- every
        ds and pooling of it are files inside."""
        return Path(root) / seg_id / slide / region_id / key

    @classmethod
    def path(cls, root, meta: FeatureMeta) -> Path:
        return (cls.key_dir(root, seg_id=meta.seg_id, slide=meta.wsi_stem,
                            region_id=meta.region_id, key=meta.key)
                / f'{_ds_name(meta.ds)}_{meta.pooling}.safetensors')

    @classmethod
    def files(cls, key_dir, pooling: Optional[str] = None) -> List[Path]:
        """The files one key directory holds, sorted -- the levels of one draw.
        A listing of a KNOWN directory, not a search of a root."""
        pattern = f'ds*_{pooling}.safetensors' if pooling else 'ds*.safetensors'
        return sorted(Path(key_dir).glob(pattern))

    @classmethod
    def levels(cls, key_dir, pooling: str) -> Dict[int, Path]:
        """`{level: path}` for one pooling in one key directory -- one draw's
        stores, one per pyramid level. Refuses two files claiming one level."""
        out: Dict[int, Path] = {}
        for path in cls.files(key_dir, pooling):
            level = cls.load_meta(path).level
            if level in out:
                raise StoreMismatch(f'two {pooling} stores for level {level} '
                                    f'in {key_dir}: {out[level].name}, {path.name}')
            out[level] = path
        if not out:
            raise FileNotFoundError(f'no {pooling} store in {key_dir}')
        return out

    # ── write ───────────────────────────────────────────────────────────────

    @classmethod
    def save(cls, root, *, meta: FeatureMeta, features, x, y, region=None,
             grid_rc=None, extra: Optional[Dict[str, object]] = None) -> Path:
        """Validate, then write atomically to `path(root, meta)`. A truncated
        store that still loads is worse than none: it drops half the
        distractors and every recall comes out high."""
        import torch                                             # noqa: PLC0415
        from safetensors.torch import save_file                  # noqa: PLC0415

        _validate(features, x, y, region, grid_rc, meta, extra)
        meta = dataclasses.replace(meta, created_at=meta.created_at or _stamp())
        tensors = {'features': features.contiguous(), 'x': x.contiguous(),
                   'y': y.contiguous()}
        if region is not None:
            tensors['region'] = region.contiguous()
            tensors['grid_rc'] = grid_rc.contiguous()
        for k, v in (extra or {}).items():
            tensors[k] = v.contiguous() if isinstance(v, torch.Tensor) else v
        path = cls.path(root, meta)
        with atomic_file(path) as tmp:
            save_file(tensors, str(tmp), metadata=meta.to_strings())
        return path

    # ── read ────────────────────────────────────────────────────────────────

    @staticmethod
    def load_meta(path) -> FeatureMeta:
        """Header only -- instant on a 30 GB file."""
        from safetensors import safe_open                         # noqa: PLC0415
        with safe_open(str(path), framework='pt') as f:
            md = f.metadata()
        if md is None:
            raise StoreMismatch(f'{path} has no metadata -- not a feature store')
        return FeatureMeta.from_strings(md)

    @classmethod
    def load(cls, path, *, require: Optional[Dict[str, object]] = None,
             keys: Optional[Iterable[str]] = None):
        """`(tensors, meta)`, refusing outright when `require` does not match."""
        from safetensors import safe_open                         # noqa: PLC0415
        from safetensors.torch import load_file                   # noqa: PLC0415
        meta = cls.load_meta(path)
        _require(meta, require, path)
        if keys is None:
            return load_file(str(path)), meta
        tensors = {}
        with safe_open(str(path), framework='pt') as f:
            for k in keys:
                tensors[k] = f.get_tensor(k)
        return tensors, meta

    @classmethod
    def has(cls, path, key: str) -> bool:
        from safetensors import safe_open                         # noqa: PLC0415
        with safe_open(str(path), framework='pt') as f:
            return key in f.keys()


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f'FeatureStore.save: {msg}')


def _validate(features, x, y, region, grid_rc, meta: FeatureMeta, extra) -> None:
    """Refuse at write time. Every one of these is silent if it reaches a reader."""
    import torch                                                  # noqa: PLC0415
    _check(features.ndim == 3, f'features must be [N, n, D], got {tuple(features.shape)}')
    n_tiles, n_slots, dim = features.shape
    _check(features.dtype == torch.float16, f'features must be fp16, got {features.dtype}')
    _check(len(meta.slots) == n_slots,
           f'{len(meta.slots)} slot names for {n_slots} slots: {meta.slots}')
    _check(meta.dim == dim, f'meta.dim={meta.dim} but features have D={dim}')
    _check(meta.n_tiles == n_tiles, f'meta.n_tiles={meta.n_tiles} but features have N={n_tiles}')
    columns = [('x', x, torch.int32, (n_tiles,)), ('y', y, torch.int32, (n_tiles,))]

    _check(meta.coverage in ('grid', 'sample'),
           f"coverage must be 'grid' or 'sample', got {meta.coverage!r}")
    if meta.coverage == 'grid':
        # Every tile of every region, so a reader may treat the file as the
        # slide: the count must be the grid's, and the tiles must say where in
        # it they sit -- which is what FeatureMapCache's geometry check reads.
        _check(region is not None and grid_rc is not None,
               'grid coverage needs region and grid_rc')
        _check(n_tiles == meta.n_available,
               f'grid coverage but n_tiles={n_tiles} != n_available={meta.n_available}')
        columns += [('region', region, torch.int16, (n_tiles,)),
                    ('grid_rc', grid_rc, torch.int32, (n_tiles, 2))]
    else:
        # A draw: nobody may read it as the slide. It is named by the rule
        # that drew it and needs the seed to be redone. region/grid_rc are
        # the grid's words; a lattice draw has no grid.
        _check(bool(meta.sampler_id) and bool(meta.plan),
               'sample coverage needs sampler_id and plan -- they are its address')
        _check(meta.sample_seed is not None,
               'sample coverage needs a sample_seed, or the draw cannot be redone')
        if region is not None:
            columns += [('region', region, torch.int16, (n_tiles,)),
                        ('grid_rc', grid_rc, torch.int32, (n_tiles, 2))]
    for name, t, want_dtype, want_shape in columns:
        _check(tuple(t.shape) == want_shape, f'{name} must be {want_shape}, got {tuple(t.shape)}')
        _check(t.dtype == want_dtype, f'{name} must be {want_dtype}, got {t.dtype}')

    n_summary = 1 if set(meta.slots) & SUMMARY_SLOTS else 0
    if meta.slot_layout.startswith('grid:'):
        gh, gw = (int(v) for v in meta.slot_layout[5:].split('x'))
        _check(gh * gw + n_summary == n_slots,
               f'slot_layout {meta.slot_layout!r} implies {gh * gw + n_summary} slots, got {n_slots}')
    elif meta.slot_layout.startswith('ring:'):
        want = int(meta.slot_layout[5:]) + n_summary
        _check(want == n_slots,
               f'slot_layout {meta.slot_layout!r} implies {want} slots, got {n_slots}')

    # A tokens store keeps every cell, so its reader has to know where each
    # sat: slot k is at (k // W, k % W) and nothing else says what W is. The
    # two values are written by different lines -- feat_hw off the encoder,
    # slot_layout out of pooling_kinds -- so agreement is evidence both came
    # from one model. endswith: 'query_tokens' keeps every cell too.
    if meta.pooling.endswith('tokens'):
        _check(meta.feat_hw is not None,
               "pooling='tokens' keeps one slot per cell, so feat_hw must say where")
        want = f'grid:{meta.feat_hw[0]}x{meta.feat_hw[1]}'
        _check(meta.slot_layout == want,
               f'feat_hw {meta.feat_hw} implies slot_layout {want!r}, got {meta.slot_layout!r}')

    for k in (extra or {}):
        _check(k not in _CORE, f'extra tensor {k!r} collides with a core tensor name')


# ── WsiFeaturesMap <-> the grid-coverage file ─────────────────────────────────
#
# GigaPathSlidingWinSimRot holds one FeaturesMap per tissue region, each with
# its own PatchGrid; the store holds one tensor per column over every tile of
# the slide. The conversion both ways, plus the check that decides whether a
# stored grid still describes the mask in hand.
#
# THE CHECK IS GEOMETRIC. The address names the mask by seg_id / region_id, and
# a region_id is exactly as good as the recipe that made it. So nothing trusts
# it alone: `region_grids` recomputes what the mask implies at this scale and
# `geometry_mismatch` compares that with the stored columns -- milliseconds,
# no slide opened, in front of minutes of reading and encoding.
#
# THE ROUND TRIP IS LOSSY: fp16 storage moves a feature by about 1e-3, far
# below anything a ranking notices, but a cached run is not bit-identical to an
# uncached one.

#: region is int16: a slide has hundreds of regions (2,988 unfiltered, once).
_MAX_REGIONS = 32767


def region_grids(regions, *, ds: float, level: int, tile_size: int,
                 overlap: bool) -> list:
    """One PatchGrid per region, from geometry alone -- the same call
    WsiTissuesContainer makes when it builds for real. `regions` must already
    be the `patchable` view at this ds."""
    from PatchingLib import PatchGrid                             # noqa: PLC0415
    return [PatchGrid.from_size(int(r.w / ds), int(r.h / ds), tile_size,
                                overlap=overlap, x_offset=int(r.x / ds),
                                y_offset=int(r.y / ds), ds=ds, level=level)
            for r in regions]


def _columns(grids) -> dict:
    import torch                                                  # noqa: PLC0415
    if len(grids) > _MAX_REGIONS:
        raise ValueError(f'{len(grids)} regions exceeds the int16 region column')
    xs, ys, rs, rc = [], [], [], []
    for r, grid in enumerate(grids):
        for info in grid.iter_infos():
            xs.append(info.x)
            ys.append(info.y)
            rs.append(r)
            rc.append((info.row, info.col))
    return {'x': torch.tensor(xs, dtype=torch.int32),
            'y': torch.tensor(ys, dtype=torch.int32),
            'region': torch.tensor(rs, dtype=torch.int16),
            'grid_rc': torch.tensor(rc, dtype=torch.int32).reshape(-1, 2)}


def to_store_tensors(wfm) -> dict:
    """The columns `FeatureStore.save` takes, from one slide's WsiFeaturesMap.
    features come out [N, 1, D]: this path carries one vector per tile, which
    is whatever the encoder's features() reduced to."""
    import torch                                                  # noqa: PLC0415
    parts = [m.features for m in wfm]
    features = torch.cat(parts, dim=0) if parts else torch.empty(0, 0)
    out = _columns(wfm.grids())
    if features.shape[0] != out['x'].numel():
        raise ValueError(f'{features.shape[0]} feature rows against '
                         f'{out["x"].numel()} grid positions')
    out['features'] = features.unsqueeze(1).half()
    return out


def from_store_tensors(tensors: dict, regions, *, ds: float, level: int,
                       tile_size: int, overlap: bool):
    """Split the flat features back into one FeaturesMap per grid -- by the
    grids' own lengths, which agree with the stored `region` column only if
    the geometry check passed; if they disagree this raises."""
    from PatchingLib import FeaturesMap, WsiFeaturesMap          # noqa: PLC0415
    features = tensors['features']
    if features.ndim != 3 or features.shape[1] != 1:
        raise ValueError(f'expected [N, 1, D] features, got {tuple(features.shape)}')
    grids = region_grids(regions, ds=ds, level=level, tile_size=tile_size,
                         overlap=overlap)
    want = sum(len(g) for g in grids)
    if features.shape[0] != want:
        raise ValueError(f'store holds {features.shape[0]} rows and these grids '
                         f'need {want} -- run geometry_mismatch before restoring')
    flat = features[:, 0].float()
    out, at = [], 0
    for grid in grids:
        out.append(FeaturesMap(grid, flat[at:at + len(grid)], source='FeatureStore'))
        at += len(grid)
    return WsiFeaturesMap(regions, out, ds=ds, level=level,
                          tile_size=tile_size, overlap=overlap)


def geometry_mismatch(tensors: dict, grids) -> List[str]:
    """Everything about the stored columns these grids contradict; empty when
    the file describes them. Strings that name a column and a row, because a
    cache that recomputes silently reads the same as one that never hits."""
    import torch                                                  # noqa: PLC0415
    bad: List[str] = []
    want = _columns(grids)
    n_store, n_want = int(tensors['x'].numel()), int(want['x'].numel())
    if n_store != n_want:
        n_regions = int(tensors['region'].max()) + 1 if n_store else 0
        return [f'tile count: store has {n_store} over {n_regions} regions, '
                f'these grids offer {n_want} over {len(grids)}']
    for name in ('region', 'x', 'y', 'grid_rc'):
        a, b = tensors[name].to(torch.int64), want[name].to(torch.int64)
        if a.shape != b.shape:
            bad.append(f'{name}: store {tuple(a.shape)}, grids {tuple(b.shape)}')
            continue
        differs = (a != b).any(dim=-1) if a.ndim > 1 else (a != b)
        if bool(differs.any()):
            first = int(differs.nonzero()[0])
            bad.append(f'{name}: {int(differs.sum())} of {n_store} rows differ, '
                       f'first at row {first} -- store {a[first].tolist()}, '
                       f'grids {b[first].tolist()}')
    return bad


class FeatureMapCache:
    """Cached WsiFeaturesMaps for one slide: the grid-coverage feature files.

        cache = FeatureMapCache(root, wsi_path, encoder, mask_cfg)
        wfm   = cache.load(container)     # None on a miss, with a reason printed
        cache.save(wfm)

    `root` is `Cache.cache_root(<job>, 'features') / encoder_tag`. The file is
    addressed by the mask recipe's seg_id / region_id and the container's tile,
    overlap and ds; the encoder's full identity and the geometry are checked on
    every read. `load` returns None rather than raising, because the caller's
    policy is to rebuild on a miss -- so every None says why.
    """

    #: ONE slot: a WsiFeaturesMap holds one vector per tile by construction.
    #: WHAT that vector is -- CLS for a ViT, a global average for a CNN -- is
    #: the encoder's `feature_pooling`, which is this store's pooling label.
    SLOT_LAYOUT = 'none'

    def __init__(self, root, wsi_path, encoder, mask_cfg, mode: str = 'rw',
                 verbose: bool = True):
        if mode not in ('r', 'w', 'rw'):
            raise ValueError(f"mode must be 'r', 'w' or 'rw', got {mode!r}")
        from Cache import wsi_stem_of                             # noqa: PLC0415
        self.root = Path(root)
        self.wsi_path = str(wsi_path)
        self.wsi_stem = wsi_stem_of(wsi_path)
        self.encoder = encoder
        self.seg_id = mask_cfg.seg_id()
        self.region_id = mask_cfg.region_id()
        self.mode = mode
        self.verbose = verbose

    @property
    def pooling(self) -> str:
        return self.encoder.feature_pooling

    def _say(self, *lines) -> None:
        if self.verbose:
            for line in lines:
                print(f'  [features] {line}', flush=True)

    def path(self, container) -> Path:
        return (FeatureStore.key_dir(
            self.root, seg_id=self.seg_id, slide=self.wsi_stem,
            region_id=self.region_id,
            key=FeatureStore.grid_key(container.tile_size, container.overlap))
            / f'{_ds_name(container.ds)}_{self.pooling}.safetensors')

    def load(self, container):
        if 'r' not in self.mode:
            return None
        path = self.path(container)
        if not path.exists():
            self._say(f'no store at {path} -- encoding')
            return None
        want = {'wsi_stem': self.wsi_stem, 'level': container.level,
                'ds': float(container.ds), 'encoder_id': self.encoder.identity_id(),
                'seg_id': self.seg_id, 'region_id': self.region_id,
                'coverage': 'grid'}
        meta = FeatureStore.load_meta(path)
        differs = {k: (v, getattr(meta, k)) for k, v in want.items()
                   if getattr(meta, k) != v}
        if differs:
            self._say(f'{path.name} does not match:',
                      *[f'    {k}: store {g!r}, now {w!r}'
                        for k, (w, g) in sorted(differs.items())])
            return None
        tensors, _ = FeatureStore.load(path)
        grids = region_grids(container.tissue_regions, ds=container.ds,
                             level=container.level, tile_size=container.tile_size,
                             overlap=container.overlap)
        bad = geometry_mismatch(tensors, grids)
        if bad:
            self._say(f'{path.name} has the right address and the wrong regions:',
                      *[f'    {b}' for b in bad])
            return None
        wfm = from_store_tensors(tensors, container.tissue_regions,
                                 ds=container.ds, level=container.level,
                                 tile_size=container.tile_size,
                                 overlap=container.overlap)
        self._say(f'{path}  {wfm.n_patches():,} tiles, {len(wfm)} regions -- '
                  f'no encoding needed')
        return wfm

    def save(self, wfm) -> Optional[Path]:
        if 'w' not in self.mode:
            return None
        base_mpp = float(getattr(self.encoder, 'base_mpp', 0.0)) or 0.0
        meta = FeatureMeta(
            wsi_stem=self.wsi_stem, wsi_path=self.wsi_path, level=wfm.level,
            ds=float(wfm.ds), mpp=base_mpp * wfm.ds, base_mpp=base_mpp,
            tile_size=wfm.tile_size, overlap=wfm.overlap,
            pooling=self.pooling, slots=(self.pooling,),
            slot_layout=self.SLOT_LAYOUT, dim=wfm.feat_dim,
            feat_hw=self.encoder.model_spec.feat_hw,
            num_prefix=self.encoder.model_spec.num_prefix,
            encoder_id=self.encoder.identity_id(), seg_id=self.seg_id,
            region_id=self.region_id, coverage='grid',
            n_available=wfm.n_patches(), n_tiles=wfm.n_patches())
        path = FeatureStore.save(self.root, meta=meta, **to_store_tensors(wfm))
        self._say(f'wrote {path}  {wfm.n_patches():,} tiles')
        return path


# ══════════════════════════════════════════════════════════════════════════════
#  pre-tiles
# ══════════════════════════════════════════════════════════════════════════════
#
# WHAT A PRE-TILE IS. `warp_image` fills anything sampled from outside its
# input with pure black, and a production homography needs 1.78x the source it
# is given (spec.md 6.6, `valid 67.8%`). Black is not a blank: it is a straight
# maximum-contrast edge with two right angles, which is what a corner detector
# fires on. A WSI has an answer a photograph does not -- tissue continues past
# the tile -- so the store holds the tile with `PRE_TILE_FACTOR` times its side
# of context (TileSampler's reserve, read back), the warp runs on that, and the
# centre tile is cropped out: every output pixel is real tissue.
#
# THE TILE IS ALWAYS THE CENTRE CROP. `centre_crop(pre, tile)` is the tile,
# everywhere; nothing slides a pre-tile inward at a slide edge.
#
# WHY PNG FILES AND NOT ONE PACKED TENSOR. A training epoch wants one pre-tile
# at a time, in the sampler's order, from several worker processes: one file
# per record is random access for free. PNG because lossless -- JPEG ringing at
# an 8x8 block edge is a corner.

_INDEX_NAME = 'index.csv'
_META_NAME = 'meta.json'


@dataclass(frozen=True)
class PreTileCorpus:
    """Which pre-tiles: one extraction's key, slide-independent.

        corpus = PreTileCorpus.of(root, mask_cfg, sampler_cfg, plan)
        for folder in corpus.rung_dirs(slide): ...

    Everything that decides which tiles were cut: the mask they were drawn
    through (`seg_id`, `region_id`), the draw (`sampler_id` covers all three
    sampling axes, the tile size and the seed) and the rung plan the draw was
    made over (a chain is only a chain over the rungs sampled TOGETHER), and
    the context factor. Two corpora in one root differ in at least one of
    these, so they sit in two directories and a reader cannot mix them.
    """
    root: Path
    seg_id: str
    region_id: str
    sampler_id: str
    plan: str
    factor: int = PRE_TILE_FACTOR

    @classmethod
    def of(cls, root, mask_cfg, sampler_cfg, plan, factor: int = PRE_TILE_FACTOR
           ) -> 'PreTileCorpus':
        """From the objects that made it -- `plan` is a `TileSampler.PlanSpec`."""
        return cls(Path(root), mask_cfg.seg_id(), mask_cfg.region_id(),
                   sampler_cfg.sampler_id(), plan.key(), int(factor))

    @property
    def key(self) -> str:
        """`<seg_id>/<region_id>_<sampler_id>_<plan>/f<factor>` -- what a label
        store records as the pre-tiles it was made from (`PreTileMeta.corpus_key`
        spells the same string)."""
        return f'{self.seg_id}/{self.region_id}_{self.sampler_id}_{self.plan}/f{self.factor}'

    @classmethod
    def from_key(cls, root, key: str) -> 'PreTileCorpus':
        """The inverse of `key`: a corpus from the string a label store kept."""
        try:
            seg_id, middle, factor = key.split('/')
            region_id, sampler_id, plan = middle.split('_', 2)
            return cls(Path(root), seg_id, region_id, sampler_id, plan,
                       int(factor.lstrip('f')))
        except ValueError:
            raise StoreMismatch(f'{key!r} is not a pre-tile corpus key') from None

    def set_dir(self, slide: str) -> Path:
        return (self.root / self.seg_id / slide
                / f'{self.region_id}_{self.sampler_id}_{self.plan}' / f'f{self.factor}')

    def rung_dir(self, slide: str, ds: float) -> Path:
        return self.set_dir(slide) / _ds_name(ds)

    def rung_dirs(self, slide: str) -> List[Path]:
        """The FINISHED rungs of one slide (index.csv present), finest first.
        An interrupted rung has no index and is not a dataset yet."""
        base = self.set_dir(slide)
        found = [d for d in base.glob('ds*') if (d / _INDEX_NAME).exists()]
        return sorted(found, key=lambda d: float(d.name[2:]))

    def slides(self) -> List[str]:
        """Slides this corpus has at least one finished rung of."""
        return sorted(p.name for p in (self.root / self.seg_id).glob('*')
                      if self.rung_dirs(p.name))


@dataclass(frozen=True)
class PreTileMeta:
    """One rung of one slide's extraction. `meta.json`.

    The reading facts (`level`, `shrink`, `read_size`) are carried rather than
    re-derived so a reader sees what was actually read without rebuilding a
    RungPlan from a pyramid that may have been remounted.
    """
    wsi_stem:        str
    ds:              float        # the ladder rung, not a native level
    tile:            int          # what the model sees
    seg_id:          str
    region_id:       str
    sampler_id:      str
    plan:            str
    pre_tile_factor: int = PRE_TILE_FACTOR
    seed:            int = 0      # provenance: already inside sampler_id
    stack_kind:      str = 'F'
    segmenter_id:    str = ''     # the built segmenter's identity, provenance
    wsi_path:        str = ''

    level:           int = 0
    level_ds:        float = 1.0
    shrink:          float = 1.0
    read_size:       int = 0      # LEVEL px, of the pre-tile

    n_tiles:         int = 0
    n_clipped:       int = 0
    n_requested:     int = 0
    created_at:      str = ''

    @property
    def pre_px(self) -> int:
        return pre_tile_px(self.tile, self.pre_tile_factor)

    @property
    def margin_px(self) -> int:
        """Tile-resolution pixels from the pre-tile edge to the tile edge."""
        return centre_margin(self.tile, self.pre_tile_factor)

    @property
    def tile_footprint_l0(self) -> float:
        """Level-0 px one tile covers -- what the richness scorer measured. The
        pre-tile is warp CONTEXT: it has to be readable, not tissue."""
        return float(self.tile) * float(self.ds)

    @property
    def margin_l0(self) -> float:
        """Level-0 px from the pre-tile's top-left to the tile's."""
        return self.tile_footprint_l0 * (self.pre_tile_factor - 1) / 2.0

    @property
    def corpus_key(self) -> str:
        """`<seg_id>/<region_id>_<sampler_id>_<plan>/f<factor>` -- what a label
        store records as the pre-tiles it was made from."""
        return (f'{self.seg_id}/{self.region_id}_{self.sampler_id}_{self.plan}'
                f'/f{self.pre_tile_factor}')

    def to_json(self) -> Dict[str, object]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: Dict[str, object]) -> 'PreTileMeta':
        known = {f.name for f in dataclasses.fields(cls)}
        missing = known - set(d) - {f.name for f in dataclasses.fields(cls)
                                    if f.default is not dataclasses.MISSING}
        if missing:
            raise StoreMismatch(f'pre-tile meta lacks {sorted(missing)}')
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def of(cls, wsi, plan, corpus: PreTileCorpus, *, tile: int, seed: int,
           stack_kind: str = 'F', segmenter_id: str = '', **counts) -> 'PreTileMeta':
        """`plan` is the RungPlan for the PRE-TILE (`tile_size=pre_px`), because
        the pre-tile is what gets read. Passing the tile's plan would record a
        read_size three times too small."""
        from Cache import wsi_stem_of                             # noqa: PLC0415
        pre = pre_tile_px(tile, corpus.factor)
        if plan.tile_size != pre:
            raise ValueError(
                f'plan is for a {plan.tile_size} px output but the pre-tile is '
                f'{pre} px. Build it with tile_size=pre_tile_px({tile}) -- the '
                f'PRE-tile is what is read')
        return cls(wsi_stem=wsi_stem_of(wsi), ds=float(plan.rung_ds), tile=int(tile),
                   seg_id=corpus.seg_id, region_id=corpus.region_id,
                   sampler_id=corpus.sampler_id, plan=corpus.plan,
                   pre_tile_factor=corpus.factor, seed=int(seed),
                   stack_kind=str(stack_kind), segmenter_id=str(segmenter_id),
                   wsi_path=str(getattr(wsi, '_filename', '') or ''),
                   level=int(plan.level), level_ds=float(plan.level_ds),
                   shrink=float(plan.shrink), read_size=int(plan.read_size),
                   **counts)


@dataclass(frozen=True)
class PreTileRecord:
    """One stored pre-tile. `x`, `y` are the TILE's top-left in level 0 -- the
    sampler's own output; the pre-tile's top-left is derived (`pre_origin_l0`)
    so the two can never disagree about where the centre is."""
    index:   int
    x:       int
    y:       int
    #: How far the pre-tile ran past the scanned rectangle, level-0. 0 on
    #: every record: the sampler is handed the pre-tile as its reserve, and
    #: extract_pretiles asserts rather than fills this.
    clip_px: int = 0

    # The three sampling axes, carried from TileSampler.SampleMeta -- each a
    # property of the RUN that placed the tile, not derivable from (x, y).
    bucket:      str = 'mid'
    score:       float = 0.0
    overlap_max: float = 0.0
    inherit_id:  int = -1
    origin:      str = 'grid'           # 'grid' | 'jitter' | 'inherit'
    parent_x:    int = -1
    parent_y:    int = -1

    @property
    def filename(self) -> str:
        return f'{self.index:06d}.png'

    def centre_l0(self, meta: PreTileMeta) -> Tuple[float, float]:
        """The tile's centre in level 0 -- what Stage B matches across rungs.
        The centre, not a corner: under rotation corners are different corners
        of one footprint."""
        half = meta.tile_footprint_l0 / 2.0
        return (self.x + half, self.y + half)

    def pre_origin_l0(self, meta: PreTileMeta) -> Tuple[float, float]:
        return (self.x - meta.margin_l0, self.y - meta.margin_l0)


#: index.csv's columns, `file` last. Read by name, so the reader needs no copy.
INDEX_COLUMNS = ('index', 'x', 'y', 'clip_px', 'bucket', 'score',
                 'overlap_max', 'inherit_id', 'origin', 'parent_x', 'parent_y',
                 'file')


class PreTileStore:
    """One rung directory: `meta.json`, one PNG per record, `index.csv` last."""

    Corpus = PreTileCorpus
    Meta = PreTileMeta
    Record = PreTileRecord

    # ── write ───────────────────────────────────────────────────────────────

    @staticmethod
    def create(corpus: PreTileCorpus, meta: PreTileMeta, *,
               overwrite: bool = False) -> Path:
        """Make the rung directory and write `meta.json`.

        Refuses a FINISHED rung (index.csv present) rather than adding to it: a
        training run may already have read it. An interrupted one is resumed."""
        folder = corpus.rung_dir(meta.wsi_stem, meta.ds)
        if (folder / _INDEX_NAME).exists() and not overwrite:
            raise StoreMismatch(
                f'{folder} already holds a finished extraction; pass '
                f'overwrite=True to replace it')
        folder.mkdir(parents=True, exist_ok=True)
        meta = dataclasses.replace(meta, created_at=meta.created_at or _stamp())
        with atomic_file(folder / _META_NAME) as tmp:
            tmp.write_text(json.dumps(meta.to_json(), indent=2, sort_keys=True))
        return folder

    @staticmethod
    def save_tile(folder, record: PreTileRecord, image: np.ndarray,
                  meta: Optional[PreTileMeta] = None) -> Path:
        """One pre-tile as PNG, atomically. Pass `meta`: a pre-tile of the wrong
        side survives everything downstream -- `centre_crop` would still return
        a square, of the wrong place."""
        import cv2                                                # noqa: PLC0415
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'expected HxWx3 RGB, got shape {image.shape}')
        if image.shape[0] != image.shape[1]:
            raise ValueError(f'pre-tile must be square, got {image.shape[:2]}')
        if meta is not None and image.shape[0] != meta.pre_px:
            raise ValueError(f'pre-tile is {image.shape[0]} px, meta says '
                             f'{meta.pre_px} ({meta.tile} x {meta.pre_tile_factor})')
        path = Path(folder) / record.filename
        ok, buf = cv2.imencode('.png', image[:, :, ::-1])
        if not ok:
            raise RuntimeError(f'cv2 failed to encode {path}')
        with atomic_file(path) as tmp:
            tmp.write_bytes(buf.tobytes())
        return path

    @classmethod
    def write_index(cls, folder, records: Iterable[PreTileRecord]) -> Path:
        """`index.csv`, LAST: its presence is what marks the rung finished, so
        a job killed at walltime leaves no index rather than a short one. The
        two counts are merged onto the meta.json `create` wrote -- not onto a
        caller's copy, which would lack `created_at`."""
        records = list(records)
        folder = Path(folder)
        meta = dataclasses.replace(
            cls.load_meta(folder), n_tiles=len(records),
            n_clipped=sum(1 for r in records if r.clip_px > 0))
        with atomic_file(folder / _META_NAME) as tmp:
            tmp.write_text(json.dumps(meta.to_json(), indent=2, sort_keys=True))
        lines = [','.join(INDEX_COLUMNS)]
        lines += [','.join(str(getattr(r, c)) for c in INDEX_COLUMNS[:-1])
                  + f',{r.filename}' for r in records]
        with atomic_file(folder / _INDEX_NAME) as tmp:
            tmp.write_text('\n'.join(lines) + '\n')
        return folder / _INDEX_NAME

    # ── read ────────────────────────────────────────────────────────────────

    @staticmethod
    def load_meta(folder, require: Optional[Dict[str, object]] = None) -> PreTileMeta:
        path = Path(folder) / _META_NAME
        if not path.exists():
            raise StoreMismatch(f'{folder} has no {_META_NAME} -- not a pre-tile rung')
        meta = PreTileMeta.from_json(json.loads(path.read_text()))
        _require(meta, require, folder)
        return meta

    @staticmethod
    def load_index(folder) -> List[PreTileRecord]:
        path = Path(folder) / _INDEX_NAME
        if not path.exists():
            raise StoreMismatch(f'{folder} has no {_INDEX_NAME}: the extraction '
                                f'did not finish. Re-run it; create() resumes')
        with open(path, newline='') as handle:
            return [PreTileRecord(
                index=int(row['index']), x=int(row['x']), y=int(row['y']),
                clip_px=int(row['clip_px']), bucket=row['bucket'],
                score=float(row['score']), overlap_max=float(row['overlap_max']),
                inherit_id=int(row['inherit_id']), origin=row['origin'],
                parent_x=int(row['parent_x']), parent_y=int(row['parent_y']))
                for row in csv.DictReader(handle)]

    @staticmethod
    def read_tile(folder, record) -> np.ndarray:
        """One pre-tile as HxWx3 RGB uint8. `record` may be a record or an index."""
        import cv2                                                # noqa: PLC0415
        if not isinstance(record, PreTileRecord):
            record = PreTileRecord(index=int(record), x=0, y=0)
        path = Path(folder) / record.filename
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise StoreMismatch(f'{path} is missing or unreadable, though the '
                                f'index lists it')
        return image[:, :, ::-1]
