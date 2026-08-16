"""Between `list[FeaturesMap]` and FeatureStore's five flat tensors.

GigaPathSlidingWinSimRot holds one FeaturesMap per tissue region, each carrying
its own PatchGrid. FeatureStore holds one tensor per column over every tile of
the slide. This module is the conversion in both directions, plus the check that
decides whether a stored one still describes the mask in hand.

It knows nothing about paths, filenames or ids -- the caller picks a root and
composes StoreMeta. What lives here is only the part that has to agree with
PatchingLib's geometry.

Why the check is geometric
--------------------------
StoreMeta.mask_id is a string a caller writes. 'hest@ds4' is the same string
whether min_region_ratio was 0.10 or 0.30, and a store written under one loaded
under the other is a region's worth of features missing with every later index
shifted by one -- silently, because the shapes still line up.

So nothing here trusts an id. `region_grids` recomputes what the mask implies at
this scale, and `geometry_mismatch` compares that against the stored columns.
PatchGrid.from_size is pure arithmetic and TissueRegion is a bounding box, so
the whole check runs without opening the slide: milliseconds, in front of the
minutes of reading and encoding it is there to protect.

Why the round trip is lossy
---------------------------
FeatureStore refuses anything but fp16 (FeatureStore._validate). Features come
back about 1e-3 different on values of unit scale, which is below the 1e-5 that
fp16 storage costs a similarity and far below anything a ranking notices -- but
it does mean a cached run is not bit-identical to an uncached one, and that the
test scores the round trip against decoys rather than against zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch

import FeatureStore as FS
from PatchingLib import FeaturesMap, PatchGrid, WsiFeaturesMap


#: What FeatureStore.save demands, mirrored here so a mismatch is caught while
#: building the tensors rather than at the write.
_DTYPES = {'x': torch.int32, 'y': torch.int32,
           'region': torch.int16, 'grid_rc': torch.int32}

#: region is int16 because a slide has hundreds of regions, not thousands of
#: thousands. It has held 2,988 unfiltered; this says where the ceiling is.
_MAX_REGIONS = 32767


def region_grids(regions, *, ds: float, level: int, tile_size: int,
                 overlap: bool) -> List[PatchGrid]:
    """One PatchGrid per region, from geometry alone -- no pixels are read.

    The same call WsiTissuesContainer makes when it builds for real, so a grid
    from here and a grid from there describe the same patches. That is what
    makes the gate meaningful: it is not comparing against a recollection, it is
    recomputing the thing itself.

    `regions` must already be narrowed to what can host a tile at this ds --
    WsiTissuesContainer.from_ds does that with mask.regions_view().
    filter_patchable. A region too small to tile yields an empty grid here and
    an empty batch at the encoder later, which is the crash from_ds exists to
    prevent.
    """
    out = []
    for region in regions:
        out.append(PatchGrid.from_size(
            int(region.w / ds), int(region.h / ds), tile_size,
            overlap=overlap,
            x_offset=int(region.x / ds), y_offset=int(region.y / ds),
            ds=ds, level=level))
    return out


def _columns(grids: List[PatchGrid]) -> dict:
    """x / y / region / grid_rc for a list of grids, in flat order."""
    if len(grids) > _MAX_REGIONS:
        raise ValueError(
            f'{len(grids)} regions exceeds the int16 region column '
            f'({_MAX_REGIONS}); widen the dtype in FeatureStore first')
    xs, ys, rs, rc = [], [], [], []
    for r, grid in enumerate(grids):
        for info in grid.iter_infos():
            xs.append(info.x)
            ys.append(info.y)
            rs.append(r)
            rc.append((info.row, info.col))
    return {
        'x': torch.tensor(xs, dtype=_DTYPES['x']),
        'y': torch.tensor(ys, dtype=_DTYPES['y']),
        'region': torch.tensor(rs, dtype=_DTYPES['region']),
        'grid_rc': torch.tensor(rc, dtype=_DTYPES['grid_rc']).reshape(-1, 2),
    }


def to_store_tensors(wfm: WsiFeaturesMap) -> dict:
    """The five tensors FeatureStore.save takes, from one slide's features.

    Takes a WsiFeaturesMap and not (grids, maps): the pairing between the two
    was the argument this function used to check on every call, and a type that
    checks it once at construction is a better place for that than a length
    comparison here.

    features come out [N, 1, D] because a store's second axis is its slots and
    this path carries one -- the production CLS. A pooling with more of them
    goes through pool_tokens and does not come here.
    """
    grids = wfm.grids()
    parts = [m.features for m in wfm]
    features = torch.cat(parts, dim=0) if parts else torch.empty(0, 0)
    out = _columns(grids)
    if features.shape[0] != out['x'].numel():
        raise ValueError(
            f'{features.shape[0]} feature rows against {out["x"].numel()} grid '
            f'positions')
    out['features'] = features.unsqueeze(1).half()
    return out


def from_store_tensors(tensors: dict, regions, *, ds: float, level: int,
                       tile_size: int, overlap: bool) -> WsiFeaturesMap:
    """Split the flat features back into one FeaturesMap per grid.

    Split by the grids' own lengths rather than by the stored `region` column.
    The two agree only if the gate passed, and if they disagree the caller has
    the wrong store -- so this raises on the length rather than quietly
    honouring whichever of the two it happened to read.
    """
    features = tensors['features']
    if features.ndim != 3:
        raise ValueError(f'features must be [N, n, D], got {tuple(features.shape)}')
    if features.shape[1] != 1:
        raise ValueError(
            f'features carry {features.shape[1]} slots; this path restores the '
            f'single-slot store a FeaturesMap can hold')

    grids = region_grids(regions, ds=ds, level=level,
                         tile_size=tile_size, overlap=overlap)
    want = sum(len(g) for g in grids)
    if features.shape[0] != want:
        raise ValueError(
            f'store holds {features.shape[0]} rows and these grids need {want} '
            f'-- run geometry_mismatch before restoring')

    flat = features[:, 0].float()
    out, at = [], 0
    for grid in grids:
        n = len(grid)
        out.append(FeaturesMap(grid, flat[at:at + n], source='FeatureStore'))
        at += n
    return WsiFeaturesMap(regions, out, ds=ds, level=level,
                          tile_size=tile_size, overlap=overlap)


def geometry_mismatch(tensors: dict, grids: List[PatchGrid]) -> List[str]:
    """Everything about the stored columns that these grids contradict.

    Returns the differences rather than raising, because the caller's policy is
    to rebuild on a miss and say why. A cache that recomputes silently is
    indistinguishable from one that is permanently cold, so the strings here are
    the thing that tells those apart -- they name a column and a row, not just
    'mismatch'.

    Empty list means the store describes these grids.
    """
    bad: List[str] = []
    want = _columns(grids)

    n_store = int(tensors['x'].numel())
    n_want = int(want['x'].numel())
    if n_store != n_want:
        n_regions = int(tensors['region'].max()) + 1 if n_store else 0
        bad.append(f'tile count: store has {n_store} over {n_regions} regions, '
                   f'these grids offer {n_want} over {len(grids)}')
        return bad          # nothing below can line up; one message is enough

    for name in ('region', 'x', 'y', 'grid_rc'):
        a = tensors[name].to(torch.int64)
        b = want[name].to(torch.int64)
        if a.shape != b.shape:
            bad.append(f'{name}: store {tuple(a.shape)}, grids {tuple(b.shape)}')
            continue
        differs = (a != b).any(dim=-1) if a.ndim > 1 else (a != b)
        if bool(differs.any()):
            first = int(differs.nonzero()[0])
            bad.append(
                f'{name}: {int(differs.sum())} of {n_store} rows differ, first '
                f'at row {first} -- store {a[first].tolist()}, '
                f'grids {b[first].tolist()}')
    return bad


# ── the store ─────────────────────────────────────────────────────────────────

class WsiFeaturesMapStore:
    """A directory of cached WsiFeaturesMaps for one slide.

    The conversion above knows nothing about paths or ids. This does: it picks
    a filename, composes StoreMeta, and decides whether a file on disk still
    describes the mask in hand.

        store = WsiFeaturesMapStore(root, wsi_path, encoder, mask_id)
        wfm   = store.load(container)     # None on a miss, with a reason printed
        store.save(wfm)

    load() returns None rather than raising, because the caller's policy is to
    rebuild on a miss. That makes the printed reason the only thing separating
    "the configuration changed, so of course it missed" from "this cache has
    never hit and nobody noticed", so every return of None says why in terms of
    a field or a row -- never a bare 'miss'.

    Correctness does not rest on the filename. Two masks can collide on one
    mask_id -- it is a string a caller composed -- and the geometry check is
    what catches it: the region coordinates are recomputed from the mask in hand
    and compared against the stored columns, which costs milliseconds and can
    only agree if the mask and the scale agree.
    """

    #: What this path stores. A pooling with more slots goes through pool_tokens
    #: and does not come here; see to_store_tensors.
    POOLING = 'cls'
    SLOTS = ('cls',)

    def __init__(self, root, wsi_path, encoder, mask_id: str,
                 mode: str = 'rw', verbose: bool = True):
        if mode not in ('r', 'w', 'rw'):
            raise ValueError(f"mode must be 'r', 'w' or 'rw', got {mode!r}")
        self.root = Path(root)
        self.wsi_path = str(wsi_path)
        self.wsi_stem = Path(wsi_path).stem
        self.encoder = encoder
        self.mask_id = mask_id
        self.mode = mode
        self.verbose = verbose

    def _say(self, *lines) -> None:
        if self.verbose:
            for line in lines:
                print(f'  [features] {line}', flush=True)

    def _require(self, container) -> dict:
        """The metadata fields a stored file has to agree with."""
        return {'wsi_stem': self.wsi_stem,
                'level': container.level,
                'pooling': self.POOLING,
                'ds': float(container.ds),
                'tile_size': container.tile_size,
                'overlap': container.overlap,
                'encoder_id': self.encoder.identity_id(),
                'mask_id': self.mask_id}

    def load(self, container) -> Optional[WsiFeaturesMap]:
        """A cached WsiFeaturesMap for this container, or None with a reason."""
        if 'r' not in self.mode:
            return None

        want = self._require(container)
        try:
            hits = FS.find(self.root, wsi_stem=self.wsi_stem,
                           level=container.level, pooling=self.POOLING)
        except FileNotFoundError:
            hits = []
        if not hits:
            self._say(f'no store for {self.wsi_stem} L{container.level} '
                      f'under {self.root} -- encoding')
            return None

        for path in hits:
            meta = FS.load_meta(path)
            differs = {k: (v, getattr(meta, k, '<absent>'))
                       for k, v in want.items() if getattr(meta, k, None) != v}
            if differs:
                # Named, not counted: "encoder_id differs" and "this cache is
                # permanently cold" read identically otherwise.
                self._say(f'{path.name} does not match:',
                          *[f'    {k}: store {g!r}, now {w!r}'
                            for k, (w, g) in sorted(differs.items())])
                continue

            tensors, _ = FS.load(path)
            grids = region_grids(container.tissue_regions, ds=container.ds,
                                 level=container.level,
                                 tile_size=container.tile_size,
                                 overlap=container.overlap)
            bad = geometry_mismatch(tensors, grids)
            if bad:
                # The metadata agreed and the geometry did not, which is exactly
                # what mask_id cannot catch on its own.
                self._say(f'{path.name} has the right name and the wrong '
                          f'regions:', *[f'    {b}' for b in bad])
                continue

            wfm = from_store_tensors(tensors, container.tissue_regions,
                                     ds=container.ds, level=container.level,
                                     tile_size=container.tile_size,
                                     overlap=container.overlap)
            self._say(f'{path.name}  {wfm.n_patches():,} tiles, '
                      f'{len(wfm)} regions -- no encoding needed')
            return wfm

        return None

    def save(self, wfm: WsiFeaturesMap) -> Optional[Path]:
        if 'w' not in self.mode:
            return None
        base_mpp = float(getattr(self.encoder, 'base_mpp', 0.0)) or 0.0
        meta = FS.StoreMeta(
            wsi_stem=self.wsi_stem, wsi_path=self.wsi_path,
            level=wfm.level, ds=wfm.ds,
            mpp=base_mpp * wfm.ds, base_mpp=base_mpp,
            tile_size=wfm.tile_size, overlap=wfm.overlap,
            pooling=self.POOLING, slots=self.SLOTS, slot_layout='none',
            dim=wfm.feat_dim, token_grid=None, num_prefix=0,
            encoder_id=self.encoder.identity_id(), mask_id=self.mask_id,
            coverage='all', n_available=wfm.n_patches(), sample_seed=None,
            n_tiles=wfm.n_patches())
        path = FS.save(self.root, meta=meta, **to_store_tensors(wfm))
        self._say(f'wrote {Path(path).name}  {wfm.n_patches():,} tiles')
        return path
