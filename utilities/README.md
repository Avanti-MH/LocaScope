# Utilities — Design Document

這份文件記錄 `utilities/` 下各 class 的設計決策，供下次對話繼續討論用。

---

## 座標系統

| 座標系 | 原點 | 單位 | 誰在用 |
|--------|------|------|--------|
| **Query local** | 圖左上角 | pixel | `QueryPatchContainer._cut_patch` |
| **Level-0** | WSI 左上角，最高解析度 | level-0 px | `wsi.read_region` 永遠吃 level-0；`TissueRegion.x/y`；`PatchInfo.x/y` |
| **Level-N** | WSI 左上角，level N | level-N px | 只作為工作空間（TileSampler 內部）；**不往外傳** |
| **Mask/thumbnail** | thumbnail 左上角 | thumb px | `TissuesRegionsMask` 內部；`ds_x/ds_y` 橋接到 level-0 |
| **Grid local** | region 左上角 | 格數 | `PatchInfo.row/col` |

### 規則
- WSI 所有 spatial 座標統一 **level-0**
- Level-N 位置在邊界乘 `ds` 轉成 level-0，不進入 PatchInfo / TissueRegion
- `wsi.read_region((x, y), level, size)` — x, y 永遠 level-0，這是 OpenSlide 固定行為
- Mask 座標只在 `has_tissue` 內部出現，外部全程 level-0

---

## Class 設計

### TissueRegion（dataclass）
```
index : int    # unique region ID
x     : int    # level-0
y     : int    # level-0
w     : int    # level-0
h     : int    # level-0
```
bbox 全部 level-0，直接給 `wsi.read_region` 使用。

---

### TissuesRegionsMask（取代並淘汰 TissueMask）

| 欄位 | 說明 |
|------|------|
| `mask` | `(H_t, W_t)` bool array，thumbnail 解析度 |
| `ds_x` | `W0 / W_t`，分開存因為 get_thumbnail 整數取整使 ds_x ≠ ds_y |
| `ds_y` | `H0 / H_t` |
| `regions` | `List[TissueRegion]`，由 connected component 或 segmentation 產生 |

**為什麼 ds_x ≠ ds_y**：`get_thumbnail(2048, 2048)` 等比縮放後輸出整數尺寸，取整造成 x/y 縮放比例微小差異，必須分開儲存。

**segmenter 介面**：
```
segmenter: Callable[[np.ndarray], np.ndarray]
  輸入：RGB uint8 (H, W, 3)
  輸出：bool（binary）或 int labeled（每個非零值 = 一個 region）
```
`from_wsi(wsi, segmenter=None)` — `None` 預設用 HSV 閾值，有傳就呼叫它。

**`has_tissue(x, y, w, h, ds=1.0)`**：
- 只給 `TileSampler` 用，**不給 WsiPatchContainer 用**
- 呼叫方傳自己的座標空間和 ds，內部算 `ds / ds_x` 轉 mask index
- 檢查 tile bbox 內 tissue pixel 佔比 ≥ threshold

**淘汰 TissueMask 的原因**：`TissuesRegionsMask` 是嚴格超集，mask + ds_x/ds_y + has_tissue 全部保留，只多了 `regions`，沒有理由保留兩個 class。

---

### PatchInfo
```python
@dataclass
class PatchInfo:
    row:  int
    col:  int
    x:    int
    y:    int
    kind: str    # 'main' | 'overlap'
    ds:   float = 1.0
```

**移除 `size`**：`size` 在 query（pixel 數）和 WSI（level-0 footprint）有雙重語義，改由容器自己記 `tile_size`。

**加入 `ds`**：PatchInfo 自描述座標空間。容器讀圖時：
```
read_region((int(info.x * info.ds), int(info.y * info.ds)), level, (tile_size, tile_size))
```
level-0 的 PatchInfo ds=1.0，level-N 的 ds=wsi.level_downsamples[N]。

---

### PatchGrid
設計良好，維持現狀。只管 layout，不碰像素。flat/unified index 系統完整。

---

### RegionPatchMap（繼承 PatchContainerBase）

| 欄位 | 說明 |
|------|------|
| `region` | `TissueRegion`（has-a，不是繼承） |
| `grid` | `PatchGrid`，level-0 空間建立 |
| `patches` | dense `List[ndarray]`，region bbox 內**全切**，不做 has_tissue 過濾 |

**為什麼全切不過濾**：`TissueRegion` 本身就是由 segmentation/connected component 確認的組織區域，bbox 內的 patch 全部有效，不需要二次過濾。

繼承 `PatchContainerBase` 可以免費取得 `iter_batches`、`to_features`、`iter_main`、`iter_overlap`。

`extract_all` 由 `WsiPatchContainer` 呼叫 `_bind` 填入，`RegionPatchMap` 自身不負責提取。

---

### WsiPatchContainer（**不**繼承 PatchContainerBase）

**為什麼不繼承**：`PatchContainerBase.__getitem__` 回傳單一 patch，但 `WsiPatchContainer.__getitem__(region_id)` 回傳 `RegionPatchMap`，語義不同，繼承會誤導。

| 欄位 | 說明 |
|------|------|
| `wsi` | `openslide.OpenSlide` |
| `level` | 讀取的 WSI level |
| `tile_size` | level-N pixel 數（讀圖用，不存在 PatchInfo） |
| `mask` | `Optional[TissuesRegionsMask]` |
| `_maps` | `Dict[int, RegionPatchMap]`，key = region_id |

**有 mask**：對每個 `TissueRegion` 建一個 `RegionPatchMap`，全切 bbox 內 patch。

**無 mask**：整張 WSI 當一個 region（id=0），建一個 `RegionPatchMap`，介面一致，不做 has_tissue 過濾。

```
container[region_id]              → RegionPatchMap
container[region_id][flat_i]      → np.ndarray
for patch in container[region_id] → 迭代該 region 所有 patch
```

---

### QueryPatchContainer
設計良好，維持現狀。

---

### FeaturesMap
待處理：torch 耦合不一致。需決定：
- **torch-only**：`__init__` 硬性要求 `isinstance(features, torch.Tensor)`，不管 torch 裝沒裝
- **agnostic**：所有 torch-specific call（`.new_empty()`、`.unsqueeze()`）改用 numpy/generic 版本

---

## 現有問題（待修）

### TissueMask.py（整個檔案需重寫）
- `TissueMask` → 淘汰，功能併入 `TissuesRegionsMask`
- `Region` class → 淘汰，改用 `TissueRegion`（dataclass，level-0）
  - 問題：`ds` 欄位座標系不明
  - 問題：`__le__`/`__ge__` 語義錯（面積比較 ≠ subset）
  - 問題：`__neq__` 應為 `__ne__`
- `TissuesRegionsMask` → 半成品，重寫
  - `main_mask` vs `tissue_mask` 不清
  - `mask_ds` 應為 `ds_x/ds_y`
  - `_to_region` SyntaxError（default 參數後接 non-default）
  - `has_tissue` 用了未傳入的 `tissue_ratio`，`self.mask` 不存在，return type 語法錯

### PatchingLib.py（局部修改）
- `PatchInfo` 移除 `size`，加入 `ds: float = 1.0`
- `FeaturesMap` torch 耦合問題（決定策略後修）
- `PatchContainerBase.extract_all` abstract 簽名太窄（WsiPatchContainer 不繼承故暫不影響）
- `WsiPatchContainer` stub 改為獨立實作

---

## 整體資料流

```
wsi.get_thumbnail()
    ↓ segmenter(img)
TissuesRegionsMask
    regions: [TissueRegion(0,...), TissueRegion(1,...), ...]
    ↓
WsiPatchContainer(wsi, level=N, mask=trm).extract_all(tile_size=256)
    _maps = {
        0: RegionPatchMap(region=TissueRegion(0), grid, patches=[...]),
        1: RegionPatchMap(region=TissueRegion(1), grid, patches=[...]),
    }
    ↓
container[0]                      → RegionPatchMap
list(container[0])                → all patches in region 0
container[0].to_features(encoder) → FeaturesMap
```

`TileSampler` 獨立使用 `TissuesRegionsMask.has_tissue(x, y, w, h, ds)` 做隨機取樣過濾，和 `WsiPatchContainer` 無關。
