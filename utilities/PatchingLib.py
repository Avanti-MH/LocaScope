"""
PatchingLib — shared patch grid layout for query and WSI pipelines.

    PatchInfo            — per-patch metadata
    PatchGrid            — layout + indexing only
    PatchContainerBase   — shared patch-container API (ABC)
    FeaturesMap          — feature vectors aligned to PatchGrid
    QueryPatchContainer  — a query image cut into a PatchGrid
    WsiFeaturesMap       — every region's FeaturesMap at one scale

A slide's tiles are read by SlideReader.read_grid, block by block.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
import numpy as np
from PIL import Image
import openslide


PatchIndex = Union[int, Tuple[int, int]]
EncodeFn = Callable[List[Any], Any]

def _source_label(source: Any) -> str:
    return source if isinstance(source, str) else ''
    
def as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    '''Normalize image to (H, W, 3) uint8 RGB (same rules as QueryPreprocessor).'''
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f'expected HxW or HxWx3/4 image, got shape {arr.shape}')
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


@dataclass
class PatchInfo:
    '''Location of one patch in a grid (query pixels or WSI level-n coords).

    For a WSI grid, x and y are level-n integers counted from
    int(region.x / ds) -- the region's origin TRUNCATED. openslide samples
    level n at the fractional x / ds, so x * ds is not where the patch's
    pixels are, by up to one level pixel. The level-0 point a read of the
    patch starts at is PatchGrid.tile_origin_l0; there is deliberately no
    level-0 conversion here.'''
    row: int
    col: int
    y: int          # top-left row or y at ds coords.
    x: int          # top-left col or x at ds coords.
    size_px: int    # patch size in pixels at ds coords.
    kind: str       # 'main' | 'overlap'
    ds: float = 1.0      # ds
    level: Optional[int] = None      # level-n

    @classmethod
    def for_query(cls, row, col, x, y, size_px, kind):
        return cls(row=row, col=col, x=x, y=y, size_px=size_px, kind=kind, ds=1.0)

    @classmethod
    def for_wsi(cls, row, col, x, y, size_px, kind, ds=1.0, level=None):
        return cls(row=row, col=col, x=x, y=y, size_px=size_px, kind=kind, ds=ds, level=level)
    
class PatchGrid:
    '''
    Patch layout and indexing for a width x height region (no pixels, no features).

    這個 class 只回答三件事（不碰影像、不碰模型）:
    - **切法**: 主格 (main) + 內角重疊格 (overlap) 的 PatchInfo 產生規則
    - **索引**: flat index / unified 2D index 的互轉與合法性檢查
    - **順序**: flat 掃描順序（有 overlap 時 m,o,m,o,... 最後幾行只有 main）

    典型用法:
    - **切 patch**（由 container 負責切像素）:
        `for info in grid.iter_infos(): patch = image[info.y:info.y+s, info.x:info.x+s]`
    - **用 index 取同一格**（container / FeaturesMap 都靠這套規則）:
        `flat = grid.flat_index_at(index)`；`info = grid.patch_info_at(index)`

    Helper / method 關係（誰用誰、用來做什麼）:
    - **from_size()**: layout 工廠
        - uses `_full_grid_starts()` 產生 main row/col 起點
        - produces `main_patch_infos` / `overlap_patch_infos`（`PatchInfo.x/y` 已含 offset）
    - **_row_scan_width() / _flat_prefix()**: flat 掃描的行寬與前綴累積
        - used by `__len__()`（計算總 slot 數）
        - used by `flat_to_unified()`（把 flat i 反推 unified (r,c)）
        - used by `flat_index_for_main/overlap()`（把 main/overlap 座標推回 flat i）
    - **flat_to_unified(flat_i)**: flat → unified
        - uses `_row_scan_width()` 逐行扣 remaining
        - used by `patch_info_at(int)`（有 overlap 時把 flat i 轉 tuple 再查）
    - **patch_info_at(index)**: 任意 index → `PatchInfo`
        - unified tuple: 做合法性檢查（mixed parity / out-of-range）
        - flat int: 若 has_overlap 會走 `flat_to_unified()` 再回到 tuple 分支
        - used by `flat_index_at()` / `iter_infos()`
    - **flat_index_at(index)**: 任意 index → flat i（給 container/features 的 `__getitem__` 用）
        - uses `patch_info_at(tuple|int)`
        - uses `flat_index_for_main()` / `flat_index_for_overlap()`
    - **iter_main_infos / iter_overlap_infos / iter_infos**
        - `iter_infos()` uses `patch_info_at(i)`（照 flat 順序產出對應的 PatchInfo）

    依賴關係表（helper / method 的用途與上下游；純文字對齊版）:

        +------------------------------+---------------------------+------------------------------+----------------------------------------------+
        | helper / method              | 主要用途                  | 依賴（calls）                 | 被誰用到（used by）                           |
        +------------------------------+---------------------------+------------------------------+----------------------------------------------+
        | from_size(...)               | 產生 layout / PatchInfo   | _full_grid_starts()          | QueryPatchContainer.extract_all()            |
        | _full_grid_starts(...)       | full-tile 起點 list        | -                            | from_size()                                  |
        | has_overlap                  | 是否有 overlap             | -                            | 多數 index/iter/len 分支                      |
        | _row_scan_width(r)           | main row 的 flat 行寬      | has_overlap                  | _flat_prefix, __len__, flat_to_unified       |
        | _flat_prefix(r)              | row r 之前 flat 累積       | _row_scan_width              | flat_index_for_main/overlap                  |
        | __len__()                    | flat slot 總數             | _row_scan_width              | len(grid), iter_infos, flat_index_at(int)    |
        | flat_to_unified(flat_i)      | flat → unified (r,c)       | _row_scan_width              | patch_info_at(int) (has_overlap 時)          |
        | patch_info_at(index)         | index → PatchInfo          | flat_to_unified (必要時)     | flat_index_at, iter_infos                    |
        | flat_index_for_main(r,c)     | main (r,c) → flat i        | _flat_prefix (必要時)        | flat_index_at, container.iter_main()         |
        | flat_index_for_overlap(r,c)  | overlap (r,c) → flat i     | _flat_prefix                 | flat_index_at, container.iter_overlap()      |
        | flat_index_at(index)         | index → flat i             | patch_info_at + flat_index_* | container/features __getitem__               |
        | iter_main_infos()            | main PatchInfo 迭代         | -                            | container.iter_main()                        |
        | iter_overlap_infos()         | overlap PatchInfo 迭代      | -                            | container.iter_overlap()                     |
        | iter_infos()                 | flat 順序 PatchInfo 迭代    | __len__ + patch_info_at(i)   | QueryPatchContainer.extract_all()            |
        +------------------------------+---------------------------+------------------------------+----------------------------------------------+

    QueryPreprocessor 對照（PatchGrid 只管 layout，不回傳影像）:
        QP.from_path / from_array / from_pil + extract_*  →  from_size（只含切法，不含讀圖）
        QP.__getitem__(index)                               →  patch_info_at(index) 取 PatchInfo；
                                                              容器用 flat_index_at(index) 取 patches[i]
        QP.__iter__() (main only)                           →  iter_main_infos()
        QP.iter_all()                                       →  iter_infos()（container 用 __iter__ 拿到同序 patches）
        QP._flat_to_unified(i)                              →  flat_to_unified(i)（此處為 public）
        QP.patch_info_at / flat_index_for_* / __len__       →  同名或等價方法

    Indexing rules (same as QueryPreprocessor):
        Without overlap:
            (r, c)  -> main patch at grid (r, c)
            i       -> i-th main patch (row-major flat)

        With overlap:
            (2*r, 2*c)       -> main (r, c)
            (2*r+1, 2*c+1)   -> overlap (r, c) at interior main corners
            i                -> row-major flat (m,o,m,o,... then tail mains)
            mixed parity     -> IndexError
    '''

    def __init__(
        self,
        width: int,
        height: int,
        tile_size: int,
        grid_rows: int,
        grid_cols: int,
        overlap_rows: int,
        overlap_cols: int,
        main_patch_infos: List[PatchInfo],
        overlap_patch_infos: List[PatchInfo],
        main_row_starts: List[int],
        main_col_starts: List[int],
        x_offset: int = 0,
        y_offset: int = 0,
    ):
        '''
        低階建構子；一般請用 from_size()。

        QueryPreprocessor 對照:
            無單一 __init__ 對應；QP 在 extract_* 後逐欄位填入等價狀態。
        '''
        self.width = width          # width in pixels (query size or WSI level-n span)
        self.height = height        # height in pixels (query size or WSI level-n span)
        self.tile_size = tile_size  # tile size in pixels
        self.x_offset = x_offset    # level-n top-left X offset of WSI level-n
        self.y_offset = y_offset    # level-n top-left Y offset of WSI level-n

        self.grid_rows = grid_rows  # number of rows in the main grid
        self.grid_cols = grid_cols  # number of columns in the main grid
        self.overlap_rows = overlap_rows  # number of rows in the overlap grid
        self.overlap_cols = overlap_cols  # number of columns in the overlap grid
        self.main_patch_infos = main_patch_infos  # list of PatchInfo for main patches
        self.overlap_patch_infos = overlap_patch_infos  # list of PatchInfo for overlap patches
        self._main_row_starts = main_row_starts  # list of row starts for main patches
        self._main_col_starts = main_col_starts  # list of column starts for main patches
        #: The region's level-0 top-left and the grid's downsample, set by
        #: `for_region` and None otherwise. x_offset is int(x / ds), truncated,
        #: so it cannot give back the level-0 point a read is anchored at.
        self.origin_l0: Optional[Tuple[float, float]] = None
        self.ds: Optional[float] = None
    # ── Factory ───────────────────────────────────────────────────────────────

    @staticmethod
    def _full_grid_starts(length: int, tile_size: int) -> List[int]:
        '''
        QueryPreprocessor._full_grid_starts — 可放 full tile 的起始座標。

        回傳一串起點，使得 `[start, start + tile_size)` 完全落在 `[0, length)`。
        '''
        return [
            start for start in range(0, length, tile_size)
            if start + tile_size <= length
        ]

    @classmethod
    def from_size(
        cls,
        width: int,
        height: int,
        tile_size: int,
        overlap: bool = True,
        x_offset: int = 0,
        y_offset: int = 0,
        ds: float = 1.0,
        level: Optional[int] = None,
    ) -> PatchGrid:
        '''
        由區域尺寸建立 grid layout（不切 pixel、不讀檔）。

        QueryPreprocessor 對照:
            extract_sub_query(tile) + extract_overlap_sub_query(tile)
            內部算 row/col starts 與 PatchInfo 的步驟，濃縮成這一個入口。
            QP 還會在此之後裁切 sub_query[]；PatchGrid 只產出「該怎麼切」。

        用途:
            - query: width/height = 圖寬高，x_offset/y_offset = 0
            - WSI: width/height = region 大小，x_offset/y_offset = level-n 左上角

        回傳:
            `PatchGrid`，其 `main_patch_infos` / `overlap_patch_infos` 的 `x,y` 已包含 offset。
            （因此對 WSI 來說 `x,y` 直接就是 level-n 座標）
        '''
        row_starts = cls._full_grid_starts(height, tile_size)
        col_starts = cls._full_grid_starts(width, tile_size)

        main_infos: List[PatchInfo] = []
        for ri, i in enumerate(row_starts):
            for ci, j in enumerate(col_starts):
                main_infos.append(PatchInfo(
                    row=ri, col=ci,
                    y=y_offset + i, x=x_offset + j,
                    size_px=tile_size, kind='main',
                    ds=ds, level=level,
                ))

        overlap_infos: List[PatchInfo] = []
        overlap_rows = overlap_cols = 0
        if overlap and len(row_starts) >= 2 and len(col_starts) >= 2:
            half = tile_size // 2
            overlap_rows = len(row_starts) - 1
            overlap_cols = len(col_starts) - 1
            for ri in range(overlap_rows):
                for ci in range(overlap_cols):
                    i = row_starts[ri] + half
                    j = col_starts[ci] + half
                    overlap_infos.append(PatchInfo(
                        row=ri, col=ci,
                        y=y_offset + i, x=x_offset + j,
                        size_px=tile_size, kind='overlap',
                        ds=ds, level=level,
                    ))

        return cls(
            width=width,
            height=height,
            tile_size=tile_size,
            grid_rows=len(row_starts),
            grid_cols=len(col_starts),
            overlap_rows=overlap_rows, 
            overlap_cols=overlap_cols, 
            main_patch_infos=main_infos,
            overlap_patch_infos=overlap_infos,
            main_row_starts=row_starts,
            main_col_starts=col_starts,
            x_offset=x_offset,
            y_offset=y_offset,
        )

    @classmethod
    def for_region(cls, region, ds: float, tile_size: int, overlap: bool = True,
                   level: Optional[int] = None,
                   size: Optional[Tuple[int, int]] = None) -> PatchGrid:
        '''The grid of one tissue region at downsample `ds` -- THE one place the
        region's level-0 box becomes level-n size and offset. `region_grids`
        and every reader of a slide's tiles go through here, so the stored
        grid and the reader's grid cannot drift apart.

        `size` overrides the level-n (w, h) when the pixels in hand are not
        exactly int(w / ds) x int(h / ds) -- a pre-cut crop. Also records
        `origin_l0` and `ds`, which `tile_origin_l0` needs.'''
        w, h = size if size is not None else (int(region.w / ds), int(region.h / ds))
        grid = cls.from_size(w, h, tile_size,
                             overlap=overlap, x_offset=int(region.x / ds),
                             y_offset=int(region.y / ds), ds=ds, level=level)
        grid.origin_l0 = (region.x, region.y)
        grid.ds = float(ds)
        return grid

    # ── Lattices ──────────────────────────────────────────────────────────────
    #
    # 'main' and 'offset' name the two lattices for the readers and the window
    # scorers; 'offset' is what PatchInfo calls kind='overlap'.

    def _half(self, lattice: str) -> int:
        if lattice == 'main':
            return 0
        if lattice == 'offset':
            return self.tile_size // 2
        raise ValueError(f"lattice must be 'main' or 'offset', got {lattice!r}")

    def lattice_dims(self, lattice: str) -> Tuple[int, int]:
        '''(rows, cols) of one lattice. The offset lattice is one row and one
        column smaller: its tiles sit between the main ones.'''
        self._half(lattice)
        if lattice == 'main':
            return self.grid_rows, self.grid_cols
        return self.overlap_rows, self.overlap_cols

    def tile_origin(self, lattice: str, row: int, col: int) -> Tuple[int, int]:
        '''Level-n (x, y) of a tile's top-left -- the PatchInfo it would get.'''
        half = self._half(lattice)
        return (self.x_offset + self._main_col_starts[col] + half,
                self.y_offset + self._main_row_starts[row] + half)

    def tile_origin_l0(self, lattice: str, row: int, col: int) -> Tuple[int, int]:
        '''Level-0 (x, y) a read of this tile starts at: the region's level-0
        origin plus the tile's level-n offset inside the region, times ds.
        Not `tile_origin * ds` -- that carries x_offset's truncation, a whole
        level pixel off the phase the region's own read has.'''
        half = self._half(lattice)
        return self.local_to_l0(self._main_col_starts[col] + half,
                                self._main_row_starts[row] + half)

    def local_to_l0(self, lx: float, ly: float) -> Tuple[int, int]:
        '''Level-0 (x, y) of a point `(lx, ly)` level px from the region's
        own top-left -- where a read of anything inside the region starts so
        it lands on the phase the region's own read has. THE formula: a tile's
        read (`tile_origin_l0`) and stage 3's crop both come through here.'''
        if self.origin_l0 is None:
            raise ValueError('local_to_l0 needs a grid made by for_region')
        return (int(round(self.origin_l0[0] + lx * self.ds)),
                int(round(self.origin_l0[1] + ly * self.ds)))

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def has_overlap(self) -> bool:
        '''是否啟用 overlap（即 overlap_patch_infos 非空）。'''
        return bool(self.overlap_patch_infos)

    @property
    def unified_rows(self) -> int:
        '''Unified 2D index 的行數；有 overlap 時為 \(2*grid_rows-1\)。'''
        if not self.has_overlap:
            return self.grid_rows
        return max(0, 2 * self.grid_rows - 1)

    @property
    def unified_cols(self) -> int:
        '''Unified 2D index 的列數；有 overlap 時為 \(2*grid_cols-1\)。'''
        if not self.has_overlap:
            return self.grid_cols
        return max(0, 2 * self.grid_cols - 1)

    # ── Flat / unified index helpers ──────────────────────────────────────────

    def _row_scan_width(self, r: int) -> int:
        '''QueryPreprocessor._row_scan_width — flat 掃描時 main row r 佔幾個 slot。'''
        if not self.has_overlap:
            return self.grid_cols
        if r < self.overlap_rows:
            return self.grid_cols + self.overlap_cols
        return self.grid_cols

    def _flat_prefix(self, r: int) -> int:
        '''QueryPreprocessor._flat_prefix — main row r 之前的 flat slot 總數。

        Closed form of `sum(_row_scan_width(i) for i in range(r))`: the first
        overlap_rows main rows each carry an overlap row beside them. O(1), because
        every tile lookup makes one call.'''
        if not self.has_overlap:
            return r * self.grid_cols
        with_overlap = min(r, self.overlap_rows)
        return (with_overlap * (self.grid_cols + self.overlap_cols)
                + (r - with_overlap) * self.grid_cols)

    def flat_to_unified(self, flat_idx: int) -> Tuple[int, int]:
        '''
        flat index → unified (row, col)。

        QueryPreprocessor 對照:
            _flat_to_unified（QP 為 private；此處公開給 container / debug 用）

        用途:
            已知 flat 順序第 i 個，反查 unified 2D index（例如畫圖、除錯）。
            QP.__getitem__(int) 有 overlap 時內部也是先走這步再取 patch。

        回傳:
            unified (row, col)，其中：
            - main: (even, even) = (2*r, 2*c)
            - overlap: (odd, odd) = (2*r+1, 2*c+1)
        '''
        remaining = flat_idx
        for r in range(self.grid_rows):
            width = self._row_scan_width(r)
            if remaining >= width:
                remaining -= width
                continue

            j = remaining
            if not self.has_overlap:
                return (r, j)
            if r < self.overlap_rows:
                if j % 2 == 0:
                    return (2 * r, 2 * (j // 2))
                return (2 * r + 1, 2 * (j // 2) + 1)
            return (2 * r, 2 * j)

        raise IndexError(f'flat index {flat_idx} out of range')

    def flat_index_for_main(self, r: int, c: int) -> int:
        '''
        main grid (r,c) → flat index。

        有 overlap 時，同一個 main row 會被 overlap slot 交錯插入，因此 flat index 不是簡單 r*cols+c。
        '''
        if not self.has_overlap:
            return r * self.grid_cols + c
        if r < self.overlap_rows:
            return self._flat_prefix(r) + 2 * c
        return self._flat_prefix(r) + c

    def flat_index_for_overlap(self, r: int, c: int) -> int:
        '''overlap grid (r,c) → flat index（只在 has_overlap=True 時有效）。'''
        return self._flat_prefix(r) + 2 * c + 1

    def flat_index_at(self, index: Union[int, Tuple[int, int]]) -> int:
        '''
        任意合法 index → flat int（`iter_infos()` / container `__iter__` 的 flat 順序位置）。

        QueryPreprocessor 對照:
            QP 沒有此 public method；QP.__getitem__(index) 內部等價於
            patches[flat_index_at(index)]（若把 index 換算成 flat）。

        用途:
            PatchContainer / FeaturesMap 的 __getitem__ 統一委派此方法，
            避免在容器層重複 unified index 邏輯。
            輸入可為 flat int，或 unified tuple (2r,2c)/(2r+1,2c+1)。

        注意:
            這裡的 (row,col) 是 unified 2D index（有 overlap 時只允許 even-even 或 odd-odd）。
        '''
        if isinstance(index, int):
            if not 0 <= index < len(self):
                raise IndexError(f'flat index {index} out of range')
            return index

        info = self.patch_info_at(index)
        if info.kind == 'main':
            return self.flat_index_for_main(info.row, info.col)
        return self.flat_index_for_overlap(info.row, info.col)

    # ── PatchInfo lookup ──────────────────────────────────────────────────────

    def patch_info_at(self, index: Union[int, Tuple[int, int]]) -> PatchInfo:
        '''
        回傳該 index 對應格的 PatchInfo（位置 metadata，不是影像）。

        QueryPreprocessor 對照:
            patch_info_at — 邏輯相同。
            QP.__getitem__(index) 取的是同一格的 pixel；這裡只取「哪一格」。

        輸入:
            - int: flat index（0..len(self)-1）
            - tuple: unified (row,col)
        '''
        if isinstance(index, tuple):
            row, col = index
            if not self.has_overlap:
                if not (0 <= row < self.grid_rows and 0 <= col < self.grid_cols):
                    raise IndexError(
                        f'main grid index ({row}, {col}) out of range '
                        f'({self.grid_rows}, {self.grid_cols})'
                    )
                return self.main_patch_infos[row * self.grid_cols + col]

            row_even = row % 2 == 0
            col_even = col % 2 == 0
            if row_even != col_even:
                raise IndexError(
                    f'unified index ({row}, {col}) is invalid: '
                    'both coordinates must be even (main) or both odd (overlap)'
                )

            if row_even:
                r, c = row // 2, col // 2
                if not (0 <= r < self.grid_rows and 0 <= c < self.grid_cols):
                    raise IndexError(
                        f'main grid index ({row}, {col}) -> ({r}, {c}) out of range '
                        f'({self.grid_rows}, {self.grid_cols})'
                    )
                return self.main_patch_infos[r * self.grid_cols + c]

            r, c = (row - 1) // 2, (col - 1) // 2
            if not (0 <= r < self.overlap_rows and 0 <= c < self.overlap_cols):
                raise IndexError(
                    f'overlap grid index ({row}, {col}) -> ({r}, {c}) out of range '
                    f'({self.overlap_rows}, {self.overlap_cols})'
                )
            return self.overlap_patch_infos[r * self.overlap_cols + c]

        if not self.has_overlap:
            return self.main_patch_infos[index]

        return self.patch_info_at(self.flat_to_unified(index))

    def __len__(self) -> int:
        '''可索引的 patch slot 總數（main + overlap，以 flat 順序計）。'''
        if not self.has_overlap:
            return len(self.main_patch_infos)
        return self._flat_prefix(self.grid_rows)

    # ── Pixel ↔ grid ──────────────────────────────────────────────────────────

    def pixel_to_grid(self, y: int, x: int, round_up: bool = False) -> Tuple[int, int]:
        '''
        Global level-N pixel (y, x) → grid tile index (r, c).

        round_up=False (default, floor): 給 top-left；pixel 落在 tile (r,c) 內就算該 tile
        round_up=True   (ceil):         給 bottom-right / size；餘數補齊一個 tile

        注意: 不做 clamp — 允許回傳超出 grid 邊界的值（由 caller 處理）。
        '''
        ts = self.tile_size
        dy = y - self.y_offset
        dx = x - self.x_offset
        if round_up:
            return math.ceil(dy / ts), math.ceil(dx / ts)
        return dy // ts, dx // ts

    # ── Iteration ─────────────────────────────────────────────────────────────

    def iter_main_infos(self) -> Iterator[PatchInfo]:
        '''
        只遍歷 main grid 的 PatchInfo。

        QueryPreprocessor 對照:
            __iter__() — QP  yield 的是 main patch 影像；這裡 yield 對應的 PatchInfo。
            順序與 QP.__iter__ 一致（main row-major）。
        '''
        yield from self.main_patch_infos

    def iter_overlap_infos(self) -> Iterator[PatchInfo]:
        '''
        只遍歷 overlap grid 的 PatchInfo。

        QueryPreprocessor 對照:
            無直接對應（QP 沒有單獨的 overlap iterator）。
            QP 的 overlap 存在 overlap_sub_query[]，需自行依 overlap_patch_infos 索引。

        用途:
            只處理 corner overlap 格（例如單獨 encode / 視覺化 overlap）。
        '''
        yield from self.overlap_patch_infos

    def iter_infos(self) -> Iterator[PatchInfo]:
        '''
        依 flat / container `__iter__` 順序遍歷全部 PatchInfo。

        QueryPreprocessor 對照:
            iter_all() — QP yield patch 影像；這裡 yield 同序的 PatchInfo。
            有 overlap 時順序為 m,o,m,o,...，最後幾行僅 main。

        用途:
            extract 時 for info in grid.iter_infos(): cut/read patch at (info.y, info.x)
        '''
        if not self.has_overlap:
            yield from self.main_patch_infos
            return
        for idx in range(len(self)):
            yield self.patch_info_at(idx)

    def summary(self) -> PatchGrid:
        '''QueryPreprocessor.summary — 印 grid 資訊（不含原圖路徑與 pixel）。'''
        print(f'Region       : {self.width} x {self.height} (W x H)')
        print(f'Offset       : ({self.x_offset}, {self.y_offset})')
        print(f'Tile size    : {self.tile_size}')
        print(
            f'Main grid    : {self.grid_rows} x {self.grid_cols} '
            f'= {len(self.main_patch_infos)}'
        )
        print(
            f'Overlap grid : {self.overlap_rows} x {self.overlap_cols} '
            f'= {len(self.overlap_patch_infos)}'
        )
        print(f'Total slots  : {len(self)}')
        if self.has_overlap:
            print(f'Unified grid : {self.unified_rows} x {self.unified_cols}')
        return self


def region_grids(regions, *, ds: float, level: int, tile_size: int,
                 overlap: bool) -> List[PatchGrid]:
    '''One `PatchGrid.for_region` per region, from geometry alone. `regions`
    must already be the `patchable` view at this ds.'''
    return [PatchGrid.for_region(r, ds, tile_size, overlap=overlap, level=level)
            for r in regions]


class FeaturesMap:
    '''
    Feature vectors aligned to a PatchGrid.

    features[i] corresponds to container[i] / grid.patch_info_at(i) in flat order.
    '''

    def __init__(
        self,
        grid: PatchGrid,
        features: Any,
        source: str = '',
    ):
        if torch is not None and not isinstance(features, torch.Tensor):
            raise TypeError('features must be a torch.Tensor')
        if features.ndim != 2:
            raise ValueError(f'features must be [N, D], got {features.shape}')
        if features.shape[0] != len(grid):
            raise ValueError(
                f'feature count {features.shape[0]} != grid length {len(grid)}'
            )

        self.grid = grid
        self.features = features
        self.source = source

    @property
    def feat_dim(self) -> int:
        return int(self.features.shape[1])

    @classmethod
    def from_patch_container(
        cls,
        container: PatchContainerBase,
        encoder: EncodeFn,
    ) -> FeaturesMap:
        container._require_extracted()
        patches = list(container)
        features = encoder(patches)
        if torch is not None and features.ndim == 1:
            features = features.unsqueeze(0)
        return cls(container.grid, features, source=_source_label(container.source))

    def _flat_index(self, index: PatchIndex) -> int:
        return self.grid.flat_index_at(index)

    def __getitem__(self, index: PatchIndex) -> Any:
        return self.features[self._flat_index(index)]

    def __len__(self) -> int:
        return len(self.grid)
    
    def __iter__(self) -> Iterator[Any]:
        for idx in range(len(self)):
            yield self[idx]

    def patch_info_at(self, index: PatchIndex) -> PatchInfo:
        return self.grid.patch_info_at(index)

    def iter_main_features(self) -> Iterator[Any]:
        for info in self.grid.main_patch_infos:
            yield self[self.grid.flat_index_for_main(info.row, info.col)]
    
    def iter_overlap_features(self) -> Iterator[Any]:
        for info in self.grid.overlap_patch_infos:
            yield self[self.grid.flat_index_for_overlap(info.row, info.col)]

    def iter_all_features(self) -> Iterator[Any]:
        for idx in range(len(self)):
            yield self.features[idx]

    def _lattice_grid(self, rows: int, cols: int, flat_index) -> Any:
        '''`[rows, cols, D]`: one lattice's features in grid order, gathered
        in ONE index_select on the device the features live on -- not a copy
        per cell, which on a slide's grid was hundreds of thousands of them.'''
        idx = torch.tensor([flat_index(r, c) for r in range(rows) for c in range(cols)],
                           dtype=torch.long, device=self.features.device)
        return self.features.index_select(0, idx).view(rows, cols, self.feat_dim)

    def main_feature_grid(self) -> Any:
        return self._lattice_grid(self.grid.grid_rows, self.grid.grid_cols,
                                  self.grid.flat_index_for_main)

    def overlap_feature_grid(self) -> Any:
        return self._lattice_grid(self.grid.overlap_rows, self.grid.overlap_cols,
                                  self.grid.flat_index_for_overlap)

    def summary(self) -> FeaturesMap:
        print(f'Source       : {self.source or "<unknown>"}')
        print(f'Feature dim  : {self.feat_dim}')
        print(f'Total feats  : {len(self)}')
        self.grid.summary()
        return self


class WsiFeaturesMap:
    '''Every tissue region's FeaturesMap for ONE slide at ONE scale.

    The WSI-level counterpart of FeaturesMap: one FeaturesMap per region, with
    the regions they belong to. Built by `from_grid_read` from what
    `SlideReader.read_grid` yields, or loaded from the feature cache.

    What it is for is a bug this repo has paid for more than once. A bare
    `list[FeaturesMap]` does not carry which regions it belongs to, so the
    pairing lives in whichever variable a caller happened to zip it with:

        GigaPathSlidingWinSim.py     (since removed) zipped mask.tissue_regions (unfiltered)
                                     with sim_maps (filtered). Every window
                                     landed on a neighbouring region's
                                     coordinates. Nothing raised.
        SlidingWinSimRot     had to publish `self.regions` with a
                                     comment saying which of the two lists it
                                     is, because the features line up with one
                                     and not the other.
        build_sim_canvas             takes mask AND regions, for the same
                                     reason.

    Here the pairing is the object. regions[i] and maps[i] are checked against
    each other once, at construction, and there are no two lists left for a
    caller to zip the wrong way round.

    It deliberately holds NO pixels and knows nothing about files. The scale it
    was built at travels with it -- ds, level, tile_size, overlap -- because
    that is what a cached copy has to be checked against, and because a
    FeaturesMap alone cannot say which downsample produced it.
    '''

    def __init__(self, regions: list, maps: List[FeaturesMap], *,
                 ds: float, level: int, tile_size: int, overlap: bool):
        if len(regions) != len(maps):
            raise ValueError(
                f'{len(maps)} FeaturesMaps for {len(regions)} regions -- these '
                f'are paired by position, so a length difference means the '
                f'features were built over a different region list')
        for i, (region, fmap) in enumerate(zip(regions, maps)):
            want = len(PatchGrid.for_region(region, ds, tile_size,
                                            overlap=overlap, level=level))
            if len(fmap) != want:
                raise ValueError(
                    f'region {i} at ds {ds} offers {want} patches and its '
                    f'FeaturesMap holds {len(fmap)} -- the two were built at '
                    f'different scales')

        self.regions = regions
        self.maps = maps
        self.ds = float(ds)
        self.level = level
        self.tile_size = tile_size
        self.overlap = overlap

    @classmethod
    def from_grid_read(cls, blocks, regions: list, grids: List[PatchGrid],
                       encoder: EncodeFn, *, ds: float, level: int,
                       tile_size: int, overlap: bool) -> 'WsiFeaturesMap':
        '''Encode what `SlideReader.read_grid` yields, block by block, into
        every region's FeaturesMap -- the slide never held whole.

        A block carries main rows `row0 ..` and the offset rows between
        them, each row-major; tile (r, c) of a lattice goes to the flat slot
        `grid.flat_index_for_main/overlap(r, c)` names, so the result is in
        exactly the grid's flat order. Every slot
        must be written exactly once, or this raises: a block lost, or read
        twice, would otherwise leave zeros that score like a real tile.

        The features stay on the device `encoder` returns them on -- a
        TileEncoder's -- because
        the similarity runs there; nothing here moves them to the host. A cache
        write moves them, once, when it writes (`Store.to_store_tensors`).'''
        feats: List[Any] = [None] * len(grids)
        hits = [torch.zeros(len(g), dtype=torch.int32) for g in grids]
        for b in blocks:
            grid = grids[b.region]
            lattices = [(b.main, b.cols, grid.flat_index_for_main)]
            if b.offset_rows:
                lattices.append((b.offset, b.cols - 1, grid.flat_index_for_overlap))
            for tiles, cols, slot in lattices:
                if not len(tiles):
                    continue
                idx = torch.tensor([slot(b.row0 + i // cols, i % cols)
                                    for i in range(len(tiles))])
                f = encoder(tiles)
                f = f.unsqueeze(0) if f.ndim == 1 else f
                if feats[b.region] is None:
                    feats[b.region] = torch.empty(len(grid), f.shape[1],
                                                  dtype=f.dtype, device=f.device)
                feats[b.region][idx.to(f.device)] = f.detach()
                hits[b.region][idx] += 1
        dim = next((f.shape[1] for f in feats if f is not None), 0)
        device = next((f.device for f in feats if f is not None), None)
        for r, (grid, h) in enumerate(zip(grids, hits)):
            if len(grid) and not bool((h == 1).all()):
                raise RuntimeError(
                    f'region {r}: {int((h == 0).sum())} of {len(grid)} tiles never '
                    f'read and {int((h > 1).sum())} read more than once')
            if feats[r] is None:
                feats[r] = torch.empty(0, dim, device=device)
        return cls(regions, [FeaturesMap(g, f) for g, f in zip(grids, feats)],
                   ds=ds, level=level, tile_size=tile_size, overlap=overlap)

    def to(self, device) -> 'WsiFeaturesMap':
        '''The same maps with their features on `device` -- one move per
        region, for features read back from a cache onto the host.'''
        return WsiFeaturesMap(
            self.regions,
            [FeaturesMap(m.grid, m.features.to(device), source=m.source)
             for m in self.maps],
            ds=self.ds, level=self.level, tile_size=self.tile_size,
            overlap=self.overlap)

    @property
    def feat_dim(self) -> int:
        return self.maps[0].feat_dim if self.maps else 0

    def n_patches(self) -> int:
        return sum(len(m) for m in self.maps)

    def grids(self) -> List[PatchGrid]:
        return [m.grid for m in self.maps]

    def __len__(self) -> int:
        return len(self.maps)

    def __getitem__(self, index: int) -> FeaturesMap:
        return self.maps[index]

    def __iter__(self) -> Iterator[FeaturesMap]:
        return iter(self.maps)

    def items(self) -> Iterator[Any]:
        '''(region, FeaturesMap) pairs.'''
        return zip(self.regions, self.maps)

    def summary(self) -> WsiFeaturesMap:
        print(f'Regions      : {len(self)}')
        print(f'Patches      : {self.n_patches()}')
        print(f'Feature dim  : {self.feat_dim}')
        print(f'Scale        : level {self.level}  ds {self.ds:g}  '
              f'tile {self.tile_size}  overlap {self.overlap}')
        return self


class PatchContainerBase(ABC):
    '''
    Shared patch-container API for query and WSI sources.

    Subclasses implement extract_all(); indexing and iteration are handled here.
    '''

    def __init__(self, source: Optional[Union[str, openslide.OpenSlide, Image.Image, np.ndarray]] = None):
        self.source = source
        self.grid: Optional[PatchGrid] = None
        self.patches: List[Any] = []
        # img_origin: 目前 self.img[0,0] 在 level-N global 座標的位置
        # QPC / TPC 未 crop 時預設 (0, 0)；TPC(is_crop=True) 會 override
        self.img_origin_x: int = 0
        self.img_origin_y: int = 0

    @property
    @abstractmethod
    def source_type(self) -> str:
        '''Return ``'query'`` or ``'wsi'``.'''

    def _bind(self, grid: PatchGrid, patches: List[Any]) -> PatchContainerBase:
        if len(patches) != len(grid):
            raise ValueError(
                f'patch count {len(patches)} != grid length {len(grid)}'
            )
        self.grid = grid
        self.patches = patches
        return self

    def _require_extracted(self) -> PatchGrid:
        if self.grid is None:
            raise RuntimeError('call extract_all() before using patch container')
        return self.grid

    def _flat_index(self, index: PatchIndex) -> int:
        return self._require_extracted().flat_index_at(index)

    def patch_info_at(self, index: PatchIndex) -> PatchInfo:
        return self._require_extracted().patch_info_at(index)

    def __getitem__(self, index: PatchIndex) -> Any:
        return self.patches[self._flat_index(index)]

    def __len__(self) -> int:
        return len(self._require_extracted())

    def __iter__(self) -> Iterator[Any]:
        '''All patches in flat order (same as qc[i] for i in range(len(qc))).'''
        for idx in range(len(self)):
            yield self.patches[idx]

    def iter_main(self) -> Iterator[Any]:
        '''Main-grid patches only (row-major).'''
        grid = self._require_extracted()
        for info in grid.main_patch_infos:
            yield self.patches[grid.flat_index_for_main(info.row, info.col)]

    def iter_overlap(self) -> Iterator[Any]:
        '''Overlap corner patches only (row-major).'''
        grid = self._require_extracted()
        for info in grid.overlap_patch_infos:
            yield self.patches[grid.flat_index_for_overlap(info.row, info.col)]

    def iter_batches(self, batch_size: int = 32) -> Iterator[List[Any]]:
        batch: List[Any] = []
        for patch in self:
            batch.append(patch)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    @abstractmethod
    def extract_all(
        self,
        tile_size: int,
        overlap: bool = True,
    ) -> PatchContainerBase:
        '''Cut/read all patches and bind grid + patches.'''

    def to_features(self, encoder: EncodeFn) -> FeaturesMap:
        return FeaturesMap.from_patch_container(self, encoder)

    def summary(self) -> PatchContainerBase:
        print(f'Source type  : {self.source_type}')
        print(f'Source       : {_source_label(self.source) or "<in-memory>"}')
        print(f'Patch count  : {len(self.patches)}')
        if self.grid is not None:
            self.grid.summary()
        else:
            print('Grid         : not extracted yet')
        return self


class QueryPatchContainer(PatchContainerBase):
    '''Container for a single query image as RGB uint8 numpy array.'''

    def __init__(self, source: Optional[Union[str, Image.Image, np.ndarray]] = None):
        super().__init__(source)
        if source is None:
            raise ValueError('source must be provided')

        if isinstance(source, str):
            self.img = as_rgb_uint8(np.array(Image.open(source).convert('RGB')))
        elif isinstance(source, Image.Image):
            self.img = as_rgb_uint8(np.array(source.convert('RGB')))
        elif isinstance(source, np.ndarray):
            self.img = as_rgb_uint8(source)
        else:
            raise ValueError(f'Unsupported source type: {type(source)}')

        self.height, self.width = self.img.shape[:2]

    @classmethod
    def from_path(cls, query_path: str) -> QueryPatchContainer:
        return cls(query_path)

    @classmethod
    def from_pil(cls, image: Image.Image) -> QueryPatchContainer:
        return cls(image)

    @classmethod
    def from_array(cls, image: np.ndarray) -> QueryPatchContainer:
        return cls(image)

    @property
    def source_type(self) -> str:
        return 'query'

    def _cut_patch(self, info: PatchInfo) -> np.ndarray:
        s = info.size_px
        return self.img[info.y:info.y + s, info.x:info.x + s].copy()

    def extract_all(self, tile_size: int, overlap: bool = True) -> QueryPatchContainer:
        grid = PatchGrid.from_size(self.width, self.height, tile_size, overlap=overlap)
        patches = [self._cut_patch(info) for info in grid.iter_infos()]
        return self._bind(grid, patches)
