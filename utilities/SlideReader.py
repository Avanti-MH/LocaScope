'''The one place pixels leave a slide.

    reader = SlideReader(wsi, resize='lanczos' | 'area', workers=budget.workers)
    tile   = reader.read(x, y, ReadSpec(256, 256), ds)              # one position
    for block in reader.read_grid(regions, grids, ds, tile=256):     # a region grid
        block.main, block.offset                                     # uint8 tensors
    for tiles, index in reader.read_points(points, spec, ds):       # scattered points

Coordinates are level-0, magnifications are ds (downsample from level 0). An
mpp enters at the caller, once (`ds = mpp / base_mpp`).

WHAT A READ IS (`read`, `plan`, `_read`)
-----------------------------------------
`ReadSpec` says how big: the sensor's output px, the bounding square if it
rotates, and the margin read around either. `ReadGeometry` turns that into a
level-0 rectangle (`FovGeometry.read_rect`), the level (`level_for`: never
upsample) and the level px (`level_px`: rounded, from output px, so a native
read needs no resize). `_read` is the one `read_region_rgb` and the one
resample (`resample`, the reader's filter). An arbitrary rectangle is
`read(x0, y0, ReadSpec(w_out, h_out), ds)`; a read off the slide is None.

`stack='R'` reads the same rectangle at ds 1 and degrades it to ds
(`degrade_resolution`); `level=` forces a finer level (the routing heads'
resampled read modes).

WHAT A GRID READ IS (`read_grid`)
----------------------------------
Every tile of the regions' `PatchGrid`s, at a pyramid level's own ds, in
blocks, in DataLoader workers. At an integer downsample (level 0; a 2x
pyramid's levels) a block is `block_rows` main rows of one region read in ONE
call, with the offset rows inside it cut from the same pixels. At any other
(BRACS's 4.00014, 16.001) each region is read once and cut into the same
blocks: a read's level origin is `origin / ds`, and reads with their own
origins would each sit at their own sub-pixel phase -- features up to 0.079
apart at BRACS level 1. Regions are the unit of work there, so different
regions of one level are read by different workers in parallel; one region
is one call, which is what bounds a level made of a few large regions
(BRACS_1228 level 1: 313 tiles/s against 3,780 at level 0).

A grid read is NOT `read` in a loop, for both reasons above: speed (one call
per tile was ~270 tiles/s, blocks over 7 workers ~1,060) and phase.
'''
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ReadGeometry import (LEVEL_REL_TOL, FovGeometry, ReadRect, ReadSpec,
                          is_native, level_for, level_px, nearest_level)
from SafeSlide import SafeSlide

#: The resampling filters a reader can be built with.
#:   lanczos  PIL LANCZOS -- the microscope simulation's filter
#:   area     cv2 INTER_AREA -- the ladder's filter; anything else invents
#:            high-frequency texture, and a keypoint detector learns to fire
#:            on it (reference tiles, pre-tiles)
RESIZE = ('lanczos', 'area')

#: Where a read's pixels come from (`RungPlan.stack_kind`).
#:   F  the footprint grows with ds, read off the pyramid
#:   R  the ds 1 footprint, degraded to ds and restored
STACKS = ('F', 'R')


def resample(img: np.ndarray, w: int, h: int, method: str) -> np.ndarray:
    """`img` at (w, h) px; returned as is when it already is that size, so a
    native read is never filtered."""
    if img.shape[1] == w and img.shape[0] == h:
        return img
    if method not in RESIZE:
        raise ValueError(f'resize must be one of {RESIZE}, got {method!r}')
    if method == 'lanczos':
        from PIL import Image                                    # noqa: PLC0415
        return np.asarray(Image.fromarray(img).resize((int(w), int(h)),
                                                      Image.LANCZOS))
    import cv2                                                   # noqa: PLC0415
    return cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)


def degrade_resolution(img, ds: float, out_side: Optional[int] = None):
    """An 'R' rung's degradation: shrink by `ds`, grow back.

    INTER_AREA down, INTER_LINEAR up. Area-averaging is the non-aliasing
    downsample; coming back up with it would be a second box filter rather than
    the interpolation a real coarser level would have gone through.

    `read(stack='R')` applies it, and Stage B applies it to derive an 'R' stack
    from a chain's ds 1 tile without a second extraction. One definition, so a
    survival number does not depend on which filter each half used.
    """
    import cv2                                                  # noqa: PLC0415
    side = int(out_side or img.shape[0])
    if float(ds) <= 1.0:
        return (img if img.shape[0] == side else
                cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA))
    small = max(1, int(round(side / float(ds))))
    img = cv2.resize(img, (small, small), interpolation=cv2.INTER_AREA)
    return cv2.resize(img, (side, side), interpolation=cv2.INTER_LINEAR)


@dataclass(frozen=True)
class ReadPlan:
    """Everything one read decides, before any IO."""
    rect: ReadRect                  # level-0: the origin, and what must be on the slide
    level: int
    read_wh: Tuple[int, int]        # level px
    out_wh: Tuple[int, int]         # output px


# ── the grid read ────────────────────────────────────────────────────────────

@dataclass
class GridBlock:
    """Rows `row0 ..` of one region: its main tiles and offset tiles."""
    region: int                 # index into the regions the read was given
    row0: int                   # first row, the same index on both lattices
    cols: int                   # main tiles per row; the offset lattice has cols - 1
    main_rows: int
    offset_rows: int
    main: torch.Tensor          # uint8 [main_rows * cols, T, T, 3], row-major
    offset: torch.Tensor        # uint8 [offset_rows * (cols - 1), T, T, 3]


def integer_downsample(ds: float) -> bool:
    """Does every block read at this downsample land on the same sub-pixel
    phase as the whole-region read? Only when level-0 rows map to whole level
    rows, i.e. the downsample is an integer."""
    return abs(float(ds) - round(float(ds))) < 1e-9


def _offset_rows(grid, row0: int, main_rows: int, offset: bool) -> int:
    """Offset rows starting at `row0` that a block of `main_rows` main rows holds."""
    if not offset:
        return 0
    off_rows_total, off_cols = grid.lattice_dims('offset')
    return max(0, min(main_rows, off_rows_total - row0)) if off_cols else 0


def _first_tile(grid, lattice: str) -> Tuple[int, int]:
    """Where a lattice's tile (0, 0) sits, in level px from the region's own
    origin -- `PatchGrid.tile_origin` with the grid's offset taken off."""
    x, y = grid.tile_origin(lattice, 0, 0)
    return x - grid.x_offset, y - grid.y_offset


def _cut(arr: np.ndarray, y0: int, grid, tile: int, main_rows: int,
         n_off: int) -> Tuple[np.ndarray, np.ndarray]:
    """Main and offset tiles of `main_rows` rows whose first row starts `y0`
    level px down `arr`, which starts at the region's left edge. Where each
    lattice starts is the grid's (`PatchGrid.tile_origin`); the reshape then
    steps by `tile`, which holds because `PatchGrid._full_grid_starts` is
    range(0, L, T) -- tiles abut, by construction."""
    _, cols = grid.lattice_dims('main')
    mx, my = _first_tile(grid, 'main')
    main = (arr[y0 + my:y0 + my + main_rows * tile, mx:mx + cols * tile]
            .reshape(main_rows, tile, cols, tile, 3)
            .transpose(0, 2, 1, 3, 4)
            .reshape(main_rows * cols, tile, tile, 3))
    if not n_off:
        return np.ascontiguousarray(main), np.zeros((0, tile, tile, 3), np.uint8)
    _, off_cols = grid.lattice_dims('offset')
    ox, oy = _first_tile(grid, 'offset')
    off = (arr[y0 + oy:y0 + oy + n_off * tile, ox:ox + off_cols * tile]
           .reshape(n_off, tile, off_cols, tile, 3)
           .transpose(0, 2, 1, 3, 4)
           .reshape(n_off * off_cols, tile, tile, 3))
    return np.ascontiguousarray(main), np.ascontiguousarray(off)


def read_block(slide, level: int, grid, tile: int, row0: int,
               main_rows: int, offset: bool) -> Tuple[np.ndarray, np.ndarray]:
    """One region's rows `row0 .. row0 + main_rows - 1` in one level read, with
    the offset rows the read covers (it is made tall enough for them). The
    read starts at main tile (row0, 0)'s level-0 origin, `tile_origin_l0`, so
    `grid` must come from `PatchGrid.for_region`."""
    half = tile // 2
    _, cols = grid.lattice_dims('main')
    n_off = _offset_rows(grid, row0, main_rows, offset)
    height = max(main_rows * tile, n_off * tile + half if n_off else 0)
    x_l0, y_l0 = grid.tile_origin_l0('main', row0, 0)
    arr = slide.read_region_rgb((x_l0, y_l0), level, (cols * tile, height))
    return _cut(arr, 0, grid, tile, main_rows, n_off)


def read_region_whole(slide, region, ds: float, level: int) -> np.ndarray:
    """The whole region in ONE read, at its level-0 origin and size: the
    reference read_grid is tested against."""
    size = (int(region.w / ds), int(region.h / ds))
    return slide.read_region_rgb((region.x, region.y), level, size)


class _Blocks(torch.utils.data.Dataset):
    """The unit of work a worker takes; each worker opens its own SafeSlide (a
    handle is not shared across processes). Integer ds: an item is one block.
    Otherwise: an item is a whole region, read once and returned as all its
    blocks, and the workers split the regions between them."""

    def __init__(self, path, regions, grids, ds, level, tile, block_rows, offset):
        self.path, self.regions, self.grids = str(path), list(regions), list(grids)
        self.ds, self.level, self.tile = float(ds), int(level), int(tile)
        self.offset = bool(offset)
        self.whole = not integer_downsample(self.ds)
        self.plan: List[Tuple[int, int, int]] = []      # every block, in order
        for r, grid in enumerate(self.grids):
            rows, cols = grid.lattice_dims('main')
            if rows == 0 or cols == 0:
                continue
            for row0 in range(0, rows, block_rows):
                self.plan.append((r, row0, min(block_rows, rows - row0)))
        if self.whole:
            in_plan = list(dict.fromkeys(r for r, _, _ in self.plan))
            self.items = [[p for p in self.plan if p[0] == r] for r in in_plan]
        else:
            self.items = [[p] for p in self.plan]
        self._slide = None

    def __len__(self):
        return len(self.items)

    def _block(self, r, row0, n, main, off) -> GridBlock:
        cols = self.grids[r].lattice_dims('main')[1]
        return GridBlock(region=r, row0=row0, cols=cols, main_rows=n,
                         offset_rows=len(off) // max(1, cols - 1) if len(off) else 0,
                         main=torch.from_numpy(main), offset=torch.from_numpy(off))

    def __getitem__(self, i) -> List[GridBlock]:
        if self._slide is None:
            self._slide = SafeSlide(self.path)
        blocks = self.items[i]
        r = blocks[0][0]
        grid = self.grids[r]
        if not self.whole:
            _, row0, n = blocks[0]
            main, off = read_block(self._slide, self.level,
                                   grid, self.tile, row0, n, self.offset)
            return [self._block(r, row0, n, main, off)]
        img = read_region_whole(self._slide, self.regions[r], self.ds, self.level)
        out = []
        for _, row0, n in blocks:
            main, off = _cut(img, row0 * self.tile, grid, self.tile, n,
                             _offset_rows(grid, row0, n, self.offset))
            out.append(self._block(r, row0, n, main, off))
        return out


def _identity(item):
    return item


class GridRead:
    """What `read_grid` returns: iterate it for the blocks, in order (region,
    then block of rows); `len` (blocks), `n_tiles` and `one_read_per_region`
    are known before reading."""

    def __init__(self, blocks: _Blocks, workers: int):
        self._blocks, self.workers = blocks, int(workers)

    def __len__(self) -> int:
        return len(self._blocks.plan)

    @property
    def one_read_per_region(self) -> bool:
        return self._blocks.whole

    @property
    def n_tiles(self) -> int:
        """Main + offset tiles this read yields."""
        n = 0
        for r, row0, rows in self._blocks.plan:
            grid = self._blocks.grids[r]
            cols = grid.lattice_dims('main')[1]
            n += rows * cols
            n += _offset_rows(grid, row0, rows, self._blocks.offset) * max(0, cols - 1)
        return n

    def __iter__(self) -> Iterator[GridBlock]:
        loader = torch.utils.data.DataLoader(
            self._blocks, batch_size=None, num_workers=self.workers,
            collate_fn=_identity, pin_memory=False)
        for blocks in loader:
            yield from blocks


class _Points(torch.utils.data.Dataset):
    """One `SlideReader.read` per item, on a worker. The worker opens its own
    SafeSlide on first use -- a handle carried across a fork returns another
    process's pixels without raising, which is why only the path travels."""

    def __init__(self, path, points, spec, ds, level, resize):
        self.path, self.points = str(path), list(points)
        self.spec, self.ds, self.level, self.resize = spec, float(ds), level, resize
        self._reader = None

    def __len__(self):
        return len(self.points)

    def __getitem__(self, k):
        if self._reader is None:
            self._reader = SlideReader(SafeSlide(self.path), resize=self.resize)
        x, y = self.points[k]
        img = self._reader.read(int(x), int(y), self.spec, self.ds, level=self.level)
        if img is None:
            raise RuntimeError(f'{self.path} ({x}, {y}) ds {self.ds:g}: the read '
                               f'runs off the slide')
        return torch.from_numpy(np.ascontiguousarray(img)), k


# ── the reader ───────────────────────────────────────────────────────────────

class SlideReader:
    """See the module docstring. One per slide per process; `Render` and its
    `at(ds)` objectives share one, as do the rungs of a training bank."""

    def __init__(self, slide: Union[SafeSlide, str, Path], *,
                 resize: str = 'lanczos', workers: int = 0):
        if resize not in RESIZE:
            raise ValueError(f'resize must be one of {RESIZE}, got {resize!r}')
        if isinstance(slide, (str, Path)):
            slide = SafeSlide(str(slide))
        if not hasattr(slide, 'read_region_rgb'):
            raise TypeError(
                f'SlideReader needs a SafeSlide, not a {type(slide).__name__}: it '
                f'reads `base_mpp` and `read_region_rgb` off the handle, and a '
                f'plain openslide handle turns a scanner hole into black. '
                f'SafeSlide(path) subclasses OpenSlide, so nothing else changes.')
        self.slide = slide
        self.path = getattr(slide, '_filename', None)
        self.resize = resize
        self.workers = int(workers)
        self.base_mpp = float(slide.base_mpp)
        self.level_downsamples = [float(d) for d in slide.level_downsamples]

    # ── the level ──────────────────────────────────────────────────────────

    def native_scale(self, *, mpp: Optional[float] = None,
                     ds: Optional[float] = None) -> Tuple[int, float]:
        """A requested scale -> (level, that level's own downsample): the
        level nearest by ratio (`ReadGeometry.nearest_level`).

        Not `level_of`, and the two answer different questions. `level_of`
        keeps `ds` and picks where to read it from. This REPLACES `ds` with a
        pyramid level's, for a caller that only works at native scales --
        retrieval, whose stored features are one level's tiles. The result
        is native, so `level_of` of it returns the same level.

        Exactly one of `mpp` / `ds`. Accepting both would mean deciding which
        wins, and a caller who passes two is telling you they are not sure --
        that is a bug to surface, not an ambiguity to resolve. No I/O."""
        if (mpp is None) == (ds is None):
            raise ValueError('give exactly one of mpp / ds')
        if ds is None:
            ds = mpp / self.base_mpp
        level = nearest_level(self.level_downsamples, ds)
        return level, self.level_downsamples[level]

    def level_of(self, ds: float, level: Optional[int] = None) -> int:
        """`level_for(ds)`, or `level` checked: a forced level must not be
        coarser than `ds` -- that would be a blow-up, not a resample."""
        if level is None:
            return level_for(self.level_downsamples, ds)
        n = len(self.level_downsamples)
        if not 0 <= level < n:
            raise ValueError(f'level {level}: this slide has levels 0..{n - 1}')
        if self.level_downsamples[level] > float(ds) * (1.0 + LEVEL_REL_TOL):
            raise ValueError(
                f'level {level} is ds {self.level_downsamples[level]:.4g}, coarser '
                f'than the ds {ds:.4g} asked for: it would be blown up, not '
                f'resampled down')
        return int(level)

    def native(self, ds: float, level: Optional[int] = None) -> bool:
        """Does `ds` come off its level with no resampling (within
        `LEVEL_REL_TOL`)? A property of the slide's pyramid: on a 2x pyramid
        every power of two is native, on a 4x one the odd ones are not, so a
        resampling signature correlates with the rung -- the routing heads
        split their accuracy on it."""
        return is_native(self.level_downsamples[self.level_of(ds, level)], ds)

    # ── one position ───────────────────────────────────────────────────────

    def plan(self, x: int, y: int, spec: ReadSpec, ds: float, *,
             level: Optional[int] = None) -> ReadPlan:
        """What `read` would read, with no IO. (x, y) is the sensor
        rectangle's level-0 top-left. A rotating spec reads the bounding
        square, centred on the rectangle, otherwise the rectangle; either grown
        by `margin_out` output px on every side (`FovGeometry.read_rect`)."""
        geo = FovGeometry.of(spec.sensor_w, spec.sensor_h, ds)
        rect = geo.read_rect(x, y, spec.rotates, margin_out=spec.margin_out)
        out = geo.read_out(spec.rotates, spec.margin_out)
        lv = self.level_of(ds, level)
        lds = self.level_downsamples[lv]
        return ReadPlan(rect=rect, level=lv,
                        read_wh=(level_px(out[0], ds, lds), level_px(out[1], ds, lds)),
                        out_wh=out)

    def fits(self, rect: ReadRect) -> bool:
        """Is this level-0 rectangle inside the canvas `read_region` addresses?"""
        w, h = self.slide.dimensions
        return rect.inside(0, 0, w, h)

    def read(self, x: int, y: int, spec: ReadSpec, ds: float, *,
             stack: str = 'F', level: Optional[int] = None) -> Optional[np.ndarray]:
        """uint8 [H, W, 3], or None when the read runs off the slide.
        `stack='R'` reads the same rectangle at ds 1 (level 0; a forced `level`
        does not apply) and degrades it to `ds`."""
        if stack not in STACKS:
            raise ValueError(f'stack must be one of {STACKS}, got {stack!r}')
        if stack == 'R':
            if spec.sensor_w != spec.sensor_h:
                raise ValueError(f"an 'R' read needs a square sensor, got "
                                 f'{spec.sensor_w}x{spec.sensor_h}')
            img = self.read(x, y, spec, 1.0)
            return None if img is None else degrade_resolution(img, ds, img.shape[0])
        p = self.plan(x, y, spec, ds, level=level)
        if not self.fits(p.rect):
            return None
        return self._read(p)

    def read_samples(self, samples, spec: ReadSpec) -> List[np.ndarray]:
        """`read(meta.x, meta.y, spec, meta.ds, stack=meta.stack_kind)` for each
        `SampleMeta` (or `Sample`) a TileSampler drew. A sampler only offers
        positions whose read fits, so one off the slide is an error here, not
        a tile to drop."""
        out = []
        for s in samples:
            m = getattr(s, 'meta', s)
            img = self.read(m.x, m.y, spec, m.ds, stack=m.stack_kind)
            if img is None:
                raise RuntimeError(
                    f'{m.slide} ds {m.ds:g} ({m.x}, {m.y}): the read runs off the '
                    f'slide, which the sampler should never have offered')
            out.append(img)
        return out

    def read_points(self, points: Sequence[Tuple[int, int]], spec: ReadSpec,
                    ds: float, *, level: Optional[int] = None,
                    batch: int = 64) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """`read(x, y, spec, ds, level=level)` at every level-0 point, on the
        reader's workers, in batches: `(uint8 [B, H, W, 3], index [B])`. The
        index is each tile's position in `points` -- place results by it, not
        by arrival. For scattered positions; a region's whole lattice is
        `read_grid`, which reads a block of rows per call instead of a tile."""
        if self.path is None:
            raise ValueError('read_points needs a slide opened from a path '
                             '(each worker reopens it)')
        loader = torch.utils.data.DataLoader(
            _Points(self.path, points, spec, ds, level, self.resize),
            batch_size=batch, shuffle=False, num_workers=self.workers,
            pin_memory=False, drop_last=False)
        yield from loader

    def _read(self, p: ReadPlan) -> np.ndarray:
        """THE read: `read_region_rgb` (a scanner hole is the background colour,
        not a black rectangle with a perfect corner) and the reader's filter."""
        img = self.slide.read_region_rgb((p.rect.x0, p.rect.y0), p.level, p.read_wh)
        return resample(img, p.out_wh[0], p.out_wh[1], self.resize)

    # ── a region grid ──────────────────────────────────────────────────────

    def read_grid(self, regions: Sequence, grids: Sequence, ds: float, *,
                  tile: int, offset: bool = True, block_rows: int = 8,
                  level: Optional[int] = None) -> GridRead:
        """Every tile of the regions' `PatchGrid`s (`PatchingLib.region_grids`), in
        blocks. A grid tile is `tile` LEVEL px, so `ds` must be its level's
        own: a grid at a ds the pyramid does not have would need a resample
        per tile, which is what a block read exists to avoid. `offset=False`
        reads the main lattice only."""
        if len(regions) != len(grids):
            raise ValueError(f'{len(regions)} regions against {len(grids)} grids')
        if block_rows < 1:
            raise ValueError(f'block_rows must be >= 1, got {block_rows}')
        lv = self.level_of(ds, level)
        if not self.native(ds, lv):
            raise ValueError(f'read_grid at ds {ds:g}: level {lv} is ds '
                             f'{self.level_downsamples[lv]:g}; a grid is read at '
                             f"a level's own ds")
        if self.path is None:
            raise ValueError('read_grid needs a slide opened from a path '
                             '(each worker reopens it)')
        return GridRead(_Blocks(self.path, regions, grids, ds, lv, tile,
                                block_rows, offset), self.workers)
