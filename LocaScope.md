# LocaScope.py — v1 Design

## Pipeline

```
query image (np.ndarray / PIL)
    │
    ├─ QueryPatchContainer.extract_all(tile_size, overlap)
    │
    ├─ [mpp=None] GigaPathKnnEstiMpp.estimate(qpc)  →  mpp_est
    │  [mpp=float] skip, use directly
    │
    ├─ TissuesRegionsMask.from_wsi(wsi, ds=seg_ds)
    │    .filter_regions(min_region_ratio)
    │    .filter_patchable(tile_size, ds_est)
    │
    ├─ GigaPathSlidingWinSim(wsi, encoder, mask, mpp, tile_size, overlap)
    │    .build_wsi_features()
    │    .build_query_features(qpc)
    │    .compute_sim_maps()
    │    .find_best()                               →  SlideWinSimResult
    │
    └─ SiftRansacLocalizer(wsi_container, qpc, retrieval)
         .read_wsi_crop()
         .detect_and_match()
         .estimate_homography()                     →  SiftRansacResult
                                                    →  LocaScopeResult
```

---

## Result

```python
@dataclass
class LocaScopeResult:
    x: int                      # level-0 top-left X  (sift.x0 if success, else retrieval.best_x0)
    y: int                      # level-0 top-left Y
    mpp: float                  # mpp used (estimated or provided)
    retrieval: SlideWinSimResult
    sift: SiftRansacResult

    @property
    def sift_success(self) -> bool:
        return self.sift.success
```

---

## Class

```python
class LocaScope:
    def __init__(
        self,
        wsi: openslide.OpenSlide | str,
        device: str | torch.device = 'auto',
        tile_size: int = 256,
        overlap: bool = True,
        seg_ds: float = 32.0,
        min_region_ratio: float = 0.10,
        batch_size: int = 1024,
    )

    def build(self) -> None
        """建 tissue mask + WSI patch features，多 query 只需呼叫一次"""

    def locate(
        self,
        query: np.ndarray | PIL.Image.Image,
        mpp: float | None = None,
        sift_padding: int = 2,
        min_inliers: int = 10,
    ) -> LocaScopeResult
```

### `__init__` 參數

| 參數 | 說明 |
|---|---|
| `wsi` | path 或已開啟的 OpenSlide object |
| `device` | `'auto'` = cuda if available |
| `tile_size` | patch 邊長（px @ target level） |
| `overlap` | sliding window 是否做 half-tile overlap |
| `seg_ds` | 組織分割用的 downsample（傳給 `from_wsi`） |
| `min_region_ratio` | filter_regions 閾值 |
| `batch_size` | GigaPath encode batch size |

### `build()`

1. load GigaPath model（若尚未載入）
2. `TissuesRegionsMask.from_wsi(wsi, ds=seg_ds)`
3. `mask.filter_regions(min_region_ratio)`
4. `GigaPathSlidingWinSim.build_wsi_features()`（feature cache 存在 self）

### `locate(query, mpp, ...)`

1. `QueryPatchContainer.extract_all(tile_size, overlap)`
2. `mpp=None` → `GigaPathKnnEstiMpp.estimate(qpc)` → `mpp_est`
3. `mask.filter_patchable(tile_size, ds_est)`（每次 locate 用當次 mpp 算）
4. `GigaPathSlidingWinSim.build_query_features(qpc)`
5. `.compute_sim_maps()` → `.find_best()` → `retrieval`
6. `SiftRansacLocalizer(...).estimate_homography()` → `sift`
7. 回傳 `LocaScopeResult`

---

## 開放問題

- `filter_patchable` 每次 locate 重跑 → 可能改變 `mask.tissue_regions`，多 query 時有副作用，考慮 copy 或在 locate 內用 temp list
- `wsi` 由呼叫者管理 lifetime（LocaScope 不 close），需在 docstring 說明
- MPP 估算失敗時的 fallback 策略（目前 GigaPathKnnEstiMpp 一定給結果）
- 是否暴露 `wsi_container`、`localizer` 供外部 debug 用
