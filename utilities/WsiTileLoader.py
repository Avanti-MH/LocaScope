"""Read WSI tiles on worker processes, in a fixed order, without a plane in RAM.

    for tiles, small, index in wsi_tile_loader(path, origin, 224, positions,
                                              level=1, cells=(16, 16),
                                              batch=64, workers=8):
        features = encode(tiles)          # [B, 3, 224, 224] uint8 on the host

Ported from `utilities/test_modules/test_EoMT.py:358-411`, which is where it was
written and where it is still the only user until this file existed. Two callers
need it now -- the PCA segmenter's fit, and the whole-slide projection that
replaces reading a plane into one array -- and a third is coming: spec.md 12
step 3c extracts pre-tiles for every slide and rung.

WHY NOT JUST LOOP AND read_region
----------------------------------
Because the cost is the read, not the model. A whole slide at level 1 is tens of
thousands of tiles and tens of GB off disk, and a MIRAX decodes each rect on the
way out. `bench_slidewin_pooling` reaches 650 tiles/s through UNI2; a serial
reader in the parent process does not come close to feeding that.

THE ONE THING THAT MUST NOT BE SIMPLIFIED
------------------------------------------
The slide is opened LAZILY, inside `__getitem__`, and never in the parent.

DataLoader workers are forked, and an OpenSlide handle carried across a fork is
one handle used by several processes. It does not raise. It returns pixels --
from whatever region the other process last asked for. Opening in `__init__`
looks identical, passes any smoke test, and corrupts reads under load.

`test_EoMT.TileSet` says the same thing in its own docstring. It is repeated here
because this is now the shared copy, and the next person to "clean up" the lazy
handle will be reading this file rather than that one.

ORDER
-----
`shuffle=False`, and the index rides along in the batch anyway. Results are
placed by the index the worker returns, never by arrival order -- so a caller
that later turns shuffling on, or switches to a sampler, does not silently
scatter its output. The index is not an assumption; it is data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402


class WsiTileSet(torch.utils.data.Dataset):
    """Tiles at given grid positions, read on a worker.

    Args:
        path:      the slide. A PATH, not a handle -- see the module docstring.
        origin:    (x, y) in LEVEL-0 pixels, where the grid starts. Usually
                   `openslide.bounds-*`, because a MIRAX canvas outside it is
                   stage travel range holding no image data.
        tile:      tile side in LEVEL pixels.
        positions: (row, col) grid coordinates, in the order results are wanted.
        level:     pyramid level to read.
        cells:     (gh, gw) for the per-cell mean colour, or None to skip it.

    Yields `(tile_u8, small, index)`. `small` is the per-cell mean colour,
    computed HERE because the worker already holds the pixels and the parent
    would otherwise keep the whole tile alive just to shrink it. It is what a
    figure draws, so the slide thumbnail and the feature grid come out the same
    shape and cannot slip.
    """

    def __init__(self, path: str, origin: Tuple[int, int], tile: int,
                 positions: Sequence[Tuple[int, int]], level: int = 0,
                 cells: Optional[Tuple[int, int]] = None):
        self.path = str(path)
        self.origin = origin
        self.tile = int(tile)
        self.positions = list(positions)
        self.level = int(level)
        self.cells = cells
        self._wsi = None
        self._stride_l0 = None

    def _slide(self):
        # Lazy on purpose. See the module docstring; this is the line.
        if self._wsi is None:
            from SafeSlide import SafeSlide
            self._wsi = SafeSlide(self.path, warn=False)
            self._stride_l0 = int(round(
                self.tile * float(self._wsi.level_downsamples[self.level])))
        return self._wsi

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, k: int):
        wsi = self._slide()
        row, col = self.positions[k]
        origin_x, origin_y = self.origin
        # read_region always takes LEVEL-0 coordinates whatever level it reads,
        # which is why the stride is in level-0 pixels and the size is not.
        # utilities/README.md's coordinate table is the standing statement.
        rgb = wsi.read_region_rgb(
            (origin_x + col * self._stride_l0, origin_y + row * self._stride_l0),
            self.level, (self.tile, self.tile))

        if self.cells is None:
            small = np.zeros((1, 1, 3), dtype=np.uint8)
        else:
            gh, gw = self.cells
            ph, pw = self.tile // gh, self.tile // gw
            small = (rgb.reshape(gh, ph, gw, pw, 3).mean(axis=(1, 3))
                     .astype(np.uint8))

        return (torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1),
                torch.from_numpy(small), k)


def wsi_tile_loader(path: str, origin: Tuple[int, int], tile: int,
                    positions: Sequence[Tuple[int, int]], *, level: int = 0,
                    cells: Optional[Tuple[int, int]] = None,
                    batch: int = 64, workers: int = 8) -> Iterable:
    """A DataLoader over `WsiTileSet`, with the settings that make it correct.

    `workers=0` runs in the parent, which is what a test wants and what a login
    node should get: forking eight processes to read a hundred tiles costs more
    than it saves, and `persistent_workers` is off so nothing outlives the loop.
    """
    return torch.utils.data.DataLoader(
        WsiTileSet(path, origin, tile, positions, level, cells),
        batch_size=batch, shuffle=False, num_workers=workers,
        pin_memory=True, drop_last=False,
        prefetch_factor=2 if workers else None,
        persistent_workers=False)


def grid_positions(span: Tuple[int, int], tile: int, level_ds: float):
    """Every whole tile of a level-0 span, as (row, col), and the grid shape.

    Partial tiles at the right and bottom edge are DROPPED, which is what
    `test_EoMT.slide_pca_mask:512` does (`nx, ny = sw // tile, sh // tile`).
    Keeping them would mean a ragged last row and column, and every consumer
    would have to carry the special case into its own indexing.

    Returns `(positions, (n_rows, n_cols))`.
    """
    stride_l0 = tile * float(level_ds)
    n_cols = int(span[0] // stride_l0)
    n_rows = int(span[1] // stride_l0)
    return ([(row, col) for row in range(n_rows) for col in range(n_cols)],
            (n_rows, n_cols))
