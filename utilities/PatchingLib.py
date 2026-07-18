"""
PatchingLib — shared patch grid layout for query and WSI pipelines.

    PatchInfo            — per-patch metadata
    PatchGrid            — layout + indexing only
    PatchContainerBase   — shared patch-container API (ABC)
    FeaturesMap          — feature vectors aligned to PatchGrid

Later:
    QueryPatchContainer / WsiPatchContainer
"""

from __future__ import annotations

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

from TissuesRegionsMask import TissueRegion

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
    '''Location of one patch in a grid (query pixels or WSI level-n or mpp coords).'''
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
    
    def to_level0(self) -> PatchInfo:
        return PatchInfo(
            row=self.row, col=self.col,
            y=int(self.y * self.ds), x=int(self.x * self.ds),
            size_px=int(self.size_px * self.ds),
            kind=self.kind,
            ds=1.0,
            level=0,
        )
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
        '''QueryPreprocessor._flat_prefix — main row r 之前的 flat slot 總數。'''
        return sum(self._row_scan_width(i) for i in range(r))

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
        return sum(self._row_scan_width(r) for r in range(self.grid_rows))

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

    def main_feature_grid(self) -> Any:
        rows, cols = self.grid.grid_rows, self.grid.grid_cols
        out = self.features.new_empty(rows, cols, self.feat_dim)
        for r in range(rows):
            for c in range(cols):
                out[r, c] = self[self.grid.flat_index_for_main(r, c)]
        return out
    def overlap_feature_grid(self) -> Any:
        rows, cols = self.grid.overlap_rows, self.grid.overlap_cols
        out = self.features.new_empty(rows, cols, self.feat_dim)
        for r in range(rows):
            for c in range(cols):
                out[r, c] = self[self.grid.flat_index_for_overlap(r, c)]
        return out

    def summary(self) -> FeaturesMap:
        print(f'Source       : {self.source or "<unknown>"}')
        print(f'Feature dim  : {self.feat_dim}')
        print(f'Total feats  : {len(self)}')
        self.grid.summary()
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

    # TODO crop_grid — 以 tile 為單位 crop，回傳同型別的新 container
    #
    #   目的：讓 SiftRansacLocalizer / 其他模組不需要手動算像素座標與 clamp，
    #         crop 後的 img_origin_x/y 自動反映偏移，外部座標轉換不需另記 crop_origin。
    #
    #   介面（subclass 各自 override）：
    #     def crop_grid(self, r0, c0, r1, c1, pad=0) -> '<SameType>':
    #
    #   三種實作方案（擇一或分層組合）：
    #
    #   方案A — 直接在 TissuePatchContainer 實作（主要動機）
    #     ts    = self.grid.tile_size
    #     r0c, c0c = max(0, r0-pad), max(0, c0-pad)
    #     r1c, c1c = min(grid_rows, r1+pad), min(grid_cols, c1+pad)
    #     px0 = grid.x_offset + c0c*ts;  py0 = grid.y_offset + r0c*ts
    #     px1 = grid.x_offset + c1c*ts;  py1 = grid.y_offset + r1c*ts
    #     new_img       = self.img[py0:py1, px0:px1].copy()
    #     new_origin_x  = self.img_origin_x + px0
    #     new_origin_y  = self.img_origin_y + py0
    #     → 需新增 TPC classmethod 接受 (img, origin_x, origin_y, img_ds)
    #     注意：QPC 無 img_origin，實作較簡單（只需 crop img + re-bind grid）。
    #
    #   方案B — TPC + QPC 都加，介面對稱
    #     與方案A相同邏輯，另在 QPC 加對稱方法（QPC 無 global 座標）。
    #     已有 patches 時應 subset 而非重新 extract，避免重複切割。
    #
    #   方案C — 先在 PatchGrid 抽 sub-grid 索引邏輯，container 組合使用
    #     PatchGrid.crop_grid(r0,c0,r1,c1) → 新 PatchGrid（含 offset 更新）
    #     TPC/QPC 用其 offset 計算 img slicing，再 re-bind。
    #     優點：邏輯集中；缺點：多一層間接，caller 用起來差異不大。


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


class TissuePatchContainer(PatchContainerBase):
    '''
    Patch container for tissue regions from a WSI level image.

    Three usage patterns — patch pixels are identical in case 2 and 3:

    Case 1 — full image, no region:
        img[0,0] = level-N (0, 0); grid covers entire image.
        TissuePatchContainer(arr, img_ds=ds)

    Case 2 — full image + region (is_crop=False):
        img is the complete level-N image; region is a sub-bbox within it.
        img_origin = (0, 0); grid starts at (region.x/ds, region.y/ds).
        TissuePatchContainer(arr, region=r, img_ds=ds, is_crop=False)

    Case 3 — pre-cropped image + region (is_crop=True):
        img is already cropped to the region bbox (memory-efficient WSI path).
        img_origin = (region.x/ds, region.y/ds); grid starts at same point.
        TissuePatchContainer(crop, region=r, img_ds=ds, is_crop=True)

    PatchInfo.x/y always holds level-N global coordinates in all three cases.
    '''
    def __init__(self, source: Optional[Union[str, openslide.OpenSlide, Image.Image, np.ndarray]] = None,
                 region: Optional[TissueRegion] = None,
                 img_ds: float = 1.0,
                 is_crop: bool = False,
                 at_level: Optional[int] = None,
                 ):
        super().__init__(source)

        if is_crop and region is None:
            raise ValueError('region must be provided when is_crop=True')

        self.tissue_region = region
        self.img_ds   = img_ds
        self.is_crop  = is_crop
        self.at_level = at_level

        if isinstance(source, str):
            self.img = as_rgb_uint8(np.array(Image.open(source).convert('RGB')))
        elif isinstance(source, Image.Image):
            self.img = as_rgb_uint8(np.array(source.convert('RGB')))
        elif isinstance(source, np.ndarray):
            self.img = as_rgb_uint8(source)
        elif isinstance(source, openslide.OpenSlide):
            if at_level is None:
                raise ValueError('at_level must be provided when source is openslide.OpenSlide')
            self.img = as_rgb_uint8(np.array(
                source.read_region((0, 0), at_level, source.level_dimensions[at_level])
            ))
        else:
            raise ValueError(f'Unsupported source type: {type(source)}')

        self.height, self.width = self.img.shape[:2]

        # img_origin: where self.img[0,0] is in level-N global space
        # local crop: crop starts at region position → origin = region.x / img_ds
        # full image: image starts at (0, 0) globally → origin = 0
        if self.is_crop and self.tissue_region is not None:
            self.img_origin_x = int(self.tissue_region.x / self.img_ds)
            self.img_origin_y = int(self.tissue_region.y / self.img_ds)
        else:
            self.img_origin_x = 0
            self.img_origin_y = 0

    @property
    def source_type(self) -> str:
        return 'tissue'

    @classmethod
    def from_path(cls, source: str, region: Optional[TissueRegion] = None,
                  img_ds: float = 1.0, is_crop: bool = False,
                  at_level: Optional[int] = None) -> TissuePatchContainer:
        return cls(source, region, img_ds, is_crop, at_level)

    @classmethod
    def from_openslide(cls, source: openslide.OpenSlide, at_level: int,
                       region: Optional[TissueRegion] = None) -> TissuePatchContainer:
        img_ds = source.level_downsamples[at_level]
        return cls(source, region, img_ds, False, at_level)

    @classmethod
    def from_pil(cls, image: Image.Image, region: Optional[TissueRegion] = None,
                 img_ds: float = 1.0, is_crop: bool = False,
                 at_level: Optional[int] = None) -> TissuePatchContainer:
        return cls(image, region, img_ds, is_crop, at_level)

    @classmethod
    def from_array(cls, array: np.ndarray, region: Optional[TissueRegion] = None,
                   img_ds: float = 1.0, is_crop: bool = False,
                   at_level: Optional[int] = None) -> TissuePatchContainer:
        return cls(array, region, img_ds, is_crop, at_level)

    def _cut_patch(self, info: PatchInfo) -> np.ndarray:
        lx = info.x - self.img_origin_x
        ly = info.y - self.img_origin_y
        s  = info.size_px
        return self.img[ly:ly+s, lx:lx+s].copy()

    def extract_all(self, tile_size: int, overlap: bool = True) -> TissuePatchContainer:
        # grid_x/y_offset: where PatchGrid starts in level-N global space
        # separate from img_origin_x/y (where self.img[0,0] is)
        if self.tissue_region is not None:
            grid_x_offset = int(self.tissue_region.x / self.img_ds)
            grid_y_offset = int(self.tissue_region.y / self.img_ds)
            if self.is_crop:
                w, h = self.width, self.height
            else:
                w = int(self.tissue_region.w / self.img_ds)
                h = int(self.tissue_region.h / self.img_ds)
        else:
            grid_x_offset = 0
            grid_y_offset = 0
            w, h = self.width, self.height

        grid = PatchGrid.from_size(
            w, h, tile_size, overlap=overlap,
            x_offset=grid_x_offset,
            y_offset=grid_y_offset,
            ds=self.img_ds, level=self.at_level
        )
        patches = [self._cut_patch(info) for info in grid.iter_infos()]
        return self._bind(grid, patches)


class WsiTissuesContainer():
    def __init__(self, wsi: openslide.OpenSlide, ds: float = 1.0, level: int = None, tile_size: int = 256, overlap: bool = True, mask: Optional[TissuesRegionsMask] = None):
        self.wsi: openslide.OpenSlide = wsi
        self.ds: float = ds
        self.mask: Optional[TissuesRegionsMask] = mask
        self.tile_size: int = tile_size
        
        if level is not None:
            self.level = level
            found = WsiTissuesContainer._find_level(wsi, ds)
            if found is not None and found != level:
                raise ValueError(f'Level {level} does not match DS {ds} (found level {found})')
        else:
            found = WsiTissuesContainer._find_level(wsi, ds)
            if found is None:
                raise ValueError(f'No level found for the given DS: {ds}')
            self.level = found
        
        self.tissue_regions: list[TissueRegion] = []
        if mask is None:
            self.tissue_regions = [
                TissueRegion(x=0, y=0, w=wsi.level_dimensions[0][0], h=wsi.level_dimensions[0][1], index=0)
            ]
        else:
            self.tissue_regions = mask.tissue_regions
        
        self.tissue_patches: list[TissuePatchContainer] = []
        for region in self.tissue_regions:
            tissue_img = wsi.read_region((region.x, region.y), self.level, (int(region.w / self.ds), int(region.h / self.ds)))
            tpc = TissuePatchContainer.from_pil(tissue_img, region=region, img_ds=self.ds, is_crop=True, at_level=self.level)
            self.tissue_patches.append(tpc.extract_all(tile_size=self.tile_size, overlap=overlap))

    @staticmethod
    def _find_level(wsi: openslide.OpenSlide, ds: float, tol: float = 1e-3) -> Optional[int]:
        for i, d in enumerate(wsi.level_downsamples):
            if abs(d - ds) / max(d, 1e-9) < tol:
                return i
        return None

    @classmethod
    def from_mpp(cls, wsi: openslide.OpenSlide, mpp: float, tile_size: int = 256, overlap: bool = True, mask: Optional[TissuesRegionsMask] = None):
        # TODO: 目前要求 mpp 必須精確對應到某個 WSI level（透過 _find_level），不然丟 ValueError。
        #   這使 pipeline 依賴 MPP 估測器輸出 level-bound 的離散值（如 GigaPathKnnEstiMpp）。
        #   若未來使用能輸出連續 MPP 的估測器，需新增 from_mpp_continuous classmethod：
        #
        #     ds_target = mpp / base_mpp
        #     level = wsi.get_best_level_for_downsample(ds_target)
        #     ds_level = wsi.level_downsamples[level]
        #     scale = ds_level / ds_target          # resize 比例：level-n px → target px
        #
        #     # 對每個 tissue region：
        #     #   1. wsi.read_region((r.x, r.y), level, (int(r.w/ds_level), int(r.h/ds_level)))
        #     #   2. cv2.resize(img, new_size, interpolation=INTER_AREA if scale<1 else INTER_LINEAR)
        #     #   3. TissuePatchContainer(resized_img, region=r, img_ds=ds_target, is_crop=True)
        #
        #     resize 後 img_ds = ds_target，PatchGrid / SiftRansacLocalizer 的座標計算不需另外
        #     校正，因為 img 已精確對應 target resolution。
        base_mpp = float(wsi.properties.get('openslide.mpp-x', 0))
        if base_mpp == 0:
            raise ValueError('WSI has no openslide.mpp-x metadata.')
        ds = mpp / base_mpp
        level = cls._find_level(wsi, ds)
        if level is None:
            raise ValueError(f'No level found for the given MPP: {mpp}')
        return cls(wsi, ds=ds, level=level, tile_size=tile_size, overlap=overlap, mask=mask)


    
    def __len__(self):
        return len(self.tissue_patches)

    def __getitem__(self, index):
        return self.tissue_patches[index]

    def __iter__(self):
        return iter(self.tissue_patches)