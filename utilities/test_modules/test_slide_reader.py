"""test_slide_reader -- SlideReader against the slide itself.

    python utilities/test_modules/test_slide_reader.py --wsi <path> [--level 1]

Run through `jobscripts/TestReadPath.sh`. Needs a slide; no GPU, no model.

    read   one position. A native read is `read_region_rgb` of the same
           rectangle, untouched; a read off the slide is None; a forced level
           coarser than ds is refused; the output is the spec's shape
           (sensor + margin, or the bounding square); `stack='R'` is the ds 1
           read degraded; `read_samples` refuses a sample off the slide. The
           decoy for "untouched" is the same read one level px over.
    grid   `read_grid` against what the retired `WsiTissuesContainer` did -- ONE
           `read_region_rgb` of the region, main tile (r, c) cut at
           (c*T, r*T), offset tile at (c*T + T/2, r*T + T/2). Blocks at an
           integer ds, one read per region otherwise; every tile must equal
           the container's to the pixel. The decoy is the reference read one
           level px to the right: it must differ, or the region is blank glass.
    scale  `native_scale`: each level, asked 0.04% off, gives back itself and
           its own downsample, which `level_of` maps to the same level; neither
           or both of mpp / ds is refused.

(Was test_grid_reader.py, the grid half alone, until GridReader became
SlideReader on 2026-10-03. The geometry is test_read_geometry.py.)
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import numpy as np                                               # noqa: E402

from ReadGeometry import ReadSpec                                # noqa: E402
from SafeSlide import SafeSlide                                  # noqa: E402
from SlideReader import SlideReader, degrade_resolution          # noqa: E402
from PatchingLib import region_grids                             # noqa: E402

TILE = 256
TEXTURE = 2.0       # the decoy's mean |diff| that says "not blank glass"
POSITIONS = [(fx, fy) for fy in (0.5, 0.35, 0.65, 0.2, 0.8)
             for fx in (0.5, 0.35, 0.65, 0.2, 0.8)]


def textured_positions(slide, rw: int, rh: int, n: int = 5) -> list:
    """Up to `n` (fx, fy) fractions of the slide where a region of rw x rh
    level-0 px holds the most edges, found on the smallest level. A MIRAX
    canvas is mostly blank around the tissue, so fixed positions can all
    miss it."""
    top = slide.level_count - 1
    ds = float(slide.level_downsamples[top])
    tw, th = slide.level_dimensions[top]
    grey = slide.read_region_rgb((0, 0), top, (tw, th)).astype(np.float32).mean(-1)
    edges = np.zeros_like(grey)
    edges[:, 1:] += np.abs(np.diff(grey, axis=1))
    edges[1:, :] += np.abs(np.diff(grey, axis=0))
    ww, wh = max(1, int(rw / ds)), max(1, int(rh / ds))
    if ww >= tw or wh >= th:
        return []
    integral = np.pad(edges, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    score = (integral[wh:, ww:] - integral[:-wh, ww:] - integral[wh:, :-ww]
             + integral[:-wh, :-ww])
    out = []
    for flat in np.argsort(score, axis=None)[::-1][:n * 50]:
        y, x = divmod(int(flat), score.shape[1])
        fx, fy = x / max(1, tw - ww), y / max(1, th - wh)
        if all(abs(fx - a) > 0.05 or abs(fy - b) > 0.05 for a, b in out):
            out.append((fx, fy))
        if len(out) == n:
            break
    return out


def _region(slide, ds, rw, rh):
    """A textured region of rw x rh level-0 px, and the decoy distance there."""
    w0, h0 = slide.dimensions
    for fx, fy in textured_positions(slide, rw, rh) + POSITIONS:
        region = SimpleNamespace(x=int(fx * (w0 - rw)), y=int(fy * (h0 - rh)),
                                 w=rw, h=rh)
        yield fx, fy, region


def _expect(failures, ok, what):
    print(f'  {"ok  " if ok else "FAIL"}  {what}', flush=True)
    if not ok:
        failures.append(what)


# ── read ─────────────────────────────────────────────────────────────────────

def run_read(slide, level) -> list:
    failures = []
    ds = float(slide.level_downsamples[level])
    reader = SlideReader(slide)
    spec = ReadSpec(TILE, TILE)
    for fx, fy, region in _region(slide, ds, int(TILE * ds), int(TILE * ds)):
        x, y = region.x, region.y
        native = reader.read(x, y, spec, ds)
        raw = slide.read_region_rgb((x, y), level, (TILE, TILE))
        shifted = slide.read_region_rgb((int(x + ds), y), level, (TILE, TILE))
        far = float(np.abs(raw.astype(np.int16) - shifted.astype(np.int16)).mean())
        if far >= TEXTURE:
            break
    print(f'  level {level} (ds {ds:g}) at ({fx:g}, {fy:g}); decoy |diff| {far:.2f}')
    _expect(failures, native is not None and np.array_equal(native, raw),
            'a native read is read_region_rgb of the same rectangle, untouched')
    _expect(failures, far >= TEXTURE, f'and the decoy one px over differs ({far:.2f})')

    w0, h0 = slide.dimensions
    _expect(failures, reader.read(w0 - 10, h0 - 10, spec, ds) is None,
            'a read off the slide is None')
    if level + 1 < slide.level_count:
        try:
            reader.read(x, y, spec, ds, level=level + 1)
            _expect(failures, False, 'a forced level coarser than ds is refused')
        except ValueError:
            _expect(failures, True, 'a forced level coarser than ds is refused')

    pad = reader.read(x, y, ReadSpec(TILE, TILE, margin_out=32), ds)
    rot = reader.plan(x, y, ReadSpec(TILE, TILE, rotates=True), ds)
    _expect(failures, pad is not None and pad.shape == (TILE + 64, TILE + 64, 3)
            and rot.out_wh[0] == rot.out_wh[1] > TILE,
            f'the output is the spec\'s shape: margin {None if pad is None else pad.shape[:2]}, '
            f'rotating square {rot.out_wh}')

    one = reader.read(x, y, spec, 1.0)
    r_read = reader.read(x, y, spec, 4.0, stack='R')
    _expect(failures, one is not None and r_read is not None
            and np.array_equal(r_read, degrade_resolution(one, 4.0, TILE)),
            "stack='R' is the ds 1 read, degraded")

    off = SimpleNamespace(meta=SimpleNamespace(slide='x', x=w0 - 10, y=h0 - 10,
                                               ds=ds, stack_kind='F'))
    try:
        reader.read_samples([off], spec)
        _expect(failures, False, 'read_samples refuses a sample off the slide')
    except RuntimeError:
        _expect(failures, True, 'read_samples refuses a sample off the slide')
    return failures


# ── grid ─────────────────────────────────────────────────────────────────────

def reference_tiles(slide, region, ds, level, grid, shift=0):
    """{(lattice, r, c): tile} from one whole-region read, cut the container's way."""
    size = (int(region.w / ds), int(region.h / ds))
    img = slide.read_region_rgb((int(region.x + shift * ds), int(region.y)), level, size)
    out = {}
    for lattice, off in (('main', 0), ('offset', TILE // 2)):
        rows, cols = grid.lattice_dims(lattice)
        for r in range(rows):
            for c in range(cols):
                y, x = r * TILE + off, c * TILE + off
                out[(lattice, r, c)] = img[y:y + TILE, x:x + TILE]
    return out


def run_grid(slide, level, workers) -> list:
    failures = []
    ds = float(slide.level_downsamples[level])
    # 5 x 7 main tiles: several blocks at block_rows=2, a short last block,
    # and an offset lattice of 4 x 6
    rw, rh = int(7 * TILE * ds), int(5 * TILE * ds)
    for fx, fy, region in _region(slide, ds, rw, rh):
        (grid,) = region_grids([region], ds=ds, level=level, tile_size=TILE,
                               overlap=True)
        ref = reference_tiles(slide, region, ds, level, grid)
        decoy = reference_tiles(slide, region, ds, level, grid, shift=1)
        far = float(np.mean([np.abs(decoy[k].astype(np.int16)
                                    - ref[k].astype(np.int16)).mean() for k in ref]))
        if far >= TEXTURE:
            break
    print(f'  level {level} (ds {ds:g}) region at ({fx:g}, {fy:g}), '
          f'{grid.lattice_dims("main")} main, {grid.lattice_dims("offset")} offset')
    got = {}
    read = SlideReader(slide, workers=workers).read_grid(
        [region], [grid], ds, tile=TILE, block_rows=2, level=level)
    for block in read:
        main, off = block.main.numpy(), block.offset.numpy()
        for i in range(block.main_rows):
            for c in range(block.cols):
                got[('main', block.row0 + i, c)] = main[i * block.cols + c]
        for i in range(block.offset_rows):
            for c in range(block.cols - 1):
                got[('offset', block.row0 + i, c)] = off[i * (block.cols - 1) + c]
    _expect(failures, set(got) == set(ref),
            f'every tile present: {len(got)} ({read.n_tiles} counted, '
            f'{len(ref)} in the reference)')
    diffs = [np.abs(got[k].astype(np.int16) - ref[k].astype(np.int16))
             for k in ref if k in got]
    worst = max(int(d.max()) for d in diffs) if diffs else -1
    how = 'one read per region' if read.one_read_per_region else 'blocks'
    _expect(failures, worst == 0,
            f'pixels equal to the whole-region read, max |diff| {worst} ({how})')
    _expect(failures, far >= TEXTURE, f'decoy: one pixel off differs by {far:.2f}')
    try:
        SlideReader(slide).read_grid([region], [grid], ds * 1.5, tile=TILE)
        _expect(failures, False, "a grid at a ds that is not a level's own is refused")
    except ValueError:
        _expect(failures, True, "a grid at a ds that is not a level's own is refused")
    return failures


# ── scale ────────────────────────────────────────────────────────────────────

def run_scale(slide) -> list:
    """native_scale must return a real level and THAT level's own downsample,
    and refuse a call that gives neither or both of mpp / ds. (Was
    test_patching_lib's validate_resolve_scale, when this was
    WsiTissuesContainer.resolve_scale.)"""
    failures = []
    reader = SlideReader(slide)
    for bad in ({}, {'mpp': 0.5, 'ds': 2.0}):
        try:
            reader.native_scale(**bad)
            _expect(failures, False, f'native_scale({bad}) refused')
        except ValueError:
            _expect(failures, True, f'native_scale({bad}) refused')
    for level, ds_true in enumerate(reader.level_downsamples):
        # 0.04% off -- the size of the gap that broke the patchable filter
        # against from_mpp -- and the level's own value must come back.
        got_level, got_ds = reader.native_scale(ds=ds_true * 1.0004)
        _expect(failures, got_level == level and got_ds == ds_true,
                f'ds {ds_true * 1.0004:.5f} -> level {got_level} ds {got_ds!r} '
                f'(want {level}, {ds_true!r})')
        # native means level_of agrees, so retrieval reads the level it chose
        _expect(failures, reader.level_of(got_ds) == got_level,
                f'level_of({got_ds:.5f}) == {got_level}')
    return failures


SECTIONS = ('read', 'grid', 'scale')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--wsi', required=True)
    ap.add_argument('--level', type=int, default=None,
                    help='pyramid level to read (default: the second, or 0)')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--only', nargs='+', choices=SECTIONS, default=None)
    args = ap.parse_args()

    slide = SafeSlide(args.wsi)
    level = args.level if args.level is not None else min(1, slide.level_count - 1)
    print(f'slide {os.path.basename(args.wsi)}  level {level}')
    failures = []
    for name in (args.only or SECTIONS):
        print(f'\n[{name}]', flush=True)
        failures += (run_read(slide, level) if name == 'read' else
                     run_scale(slide) if name == 'scale'
                     else run_grid(slide, level, args.workers))
    slide.close()
    print(f'\n{"PASS" if not failures else f"{len(failures)} FAILED"}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
