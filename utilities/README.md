# Utilities — Design Document

這份文件記錄 `utilities/` 裡 **mask 與 patch 兩塊**的設計決策，供下次對話繼續討論用。
2026-11-02 依現在的程式重寫過：原本寫的 `TissuesRegionsMask`、`RegionPatchMap`、`WsiPatchContainer` 沒有照那個設計實作（見最後一節）。
細節以程式為準：`TissueMask.py`、`TissueMaskConfig.py`、`PatchingLib.py` 的開頭都有說明。

其他模組這份沒有涵蓋：`SlideReader`（含 `read_grid`）、`SafeSlide`、`ReadGeometry`、`DsLadder`、`TileSampler`、`Store`、`Cache`、`ConfigIdentity` / `ConfigArgs`、`AccessDatasets`、`SlideWinSift`、`LocaScopePipeline`、`cli/`。

---

## 座標系統

| 座標系 | 原點 | 單位 | 誰在用 |
|--------|------|------|--------|
| **Query local** | 圖左上角 | pixel | `QueryPatchContainer`；`PatchInfo` 的 `ds=1.0` |
| **Level-0** | WSI 左上角，最高解析度 | level-0 px | `wsi.read_region` 永遠吃 level-0；`TissueRegion.x/y/w/h`；`TissueMask` 的 public 方法；`PatchGrid.origin_l0`、`tile_origin_l0` |
| **Level-N（ds 座標）** | region 左上角往外量 | level-N px | `PatchInfo.x/y/size_px`、`PatchGrid` 的 `x_offset/y_offset`：**grid 的工作空間，會往外傳**，但不能當 level-0 用 |
| **Mask** | mask 左上角，對應 level-0 的 `(origin_x, origin_y)` | mask px | `TissueMask` 內部；`mask_ds_x/mask_ds_y` 換算到 level-0 |
| **Grid local** | region 左上角 | 格數 | `PatchInfo.row/col` |

### 規則
- `TissueRegion` 與 `TissueMask` 的 public 方法一律收發 **level-0 絕對座標**。MIRAX 的 mask 只涵蓋掃描範圍（`openslide.bounds-*`），`origin_x/origin_y` 不是 0，但 `TissueRegion.x/y` 仍是絕對座標，外部不受裁切影響。
- mask 座標只在 `TissueMask` 內部出現：`mask_col = floor((x0 - origin_x) / mask_ds_x)`，用 `to_mask_xy()` / `region_box()` 換算。
- `PatchInfo.x/y` 是 level-N 整數，從 `int(region.x / ds)`（**截斷**）起算。openslide 在 level N 是從小數的 `x / ds` 取樣，所以 `x * ds` 與 patch 實際位置最多差一個 level pixel。
- 要讀圖或換回 level-0，一律走 `PatchGrid.tile_origin_l0(lattice, row, col)` / `local_to_l0(lx, ly)`；`PatchInfo` 刻意不提供 level-0 轉換。
- `wsi.read_region((x, y), level, size)` 的 x, y 永遠 level-0，這是 OpenSlide 固定行為。

---

## Class 設計

### TissueRegion
```
index : int    # region ID，預設 -1
x, y, w, h : int    # level-0 絕對座標
```
單純的 class（不是 dataclass）。bbox 全部 level-0，直接給 `wsi.read_region` 使用。

---

### SlideMask（frozen dataclass）：segmenter 的產出，也是 cache 存的東西

| 欄位 | 說明 |
|------|------|
| `mask` | `[rows, cols]` bool 或 uint8，1 = 組織 |
| `origin` | mask 左上角在 level-0 的位置；SVS 是 0，MIRAX 是 `openslide.bounds-*` |
| `span` | mask **實際涵蓋**的 level-0 範圍 `(w, h)` |
| `mask_ds` | 每個 mask px 對應的 level-0 px |
| `report` | segmenter 對這次擬合的說明 |
| `components` | 可選，`[rows, cols, k]` float16：mask 是它的閾值，留著可以快速掃閾值 |

**為什麼和 `TissueMask` 分開**：切 region 很便宜，且依賴 segmentation 不依賴的參數；segmentation 每張 slide 要幾分鐘 GPU。所以貴的那半存起來，便宜的那半重做。

`save` / `load` 寫成一個 safetensors（mask 與 components 同一格網，放同一檔才不會讀到不同 slide 的組合）；mask 預設 `packbits + zlib` 壓縮（`MASK_PACKED`），讀回來是同樣的 uint8 陣列，舊的未壓縮檔（`MASK_RAW`）仍可讀。

---

### TissueMask（`TissueMask(wsi, slide_mask)`）

`SlideMask` 放到 slide 上，加上由 connected component 找出的 `tissue_regions`。純資料，**不做 segmentation**。

| 欄位 | 說明 |
|------|------|
| `main_mask` | `(rows, cols)` bool |
| `mask_ds_x` / `mask_ds_y` | 各軸 `span / 格數`，分開存 |
| `mask_mpp` | 一個 mask px 的 um，由 wsi mpp 與 ds 算出，不另外傳入 |
| `origin_x` / `origin_y` | `main_mask[0, 0]` 的 level-0 座標 |
| `tissue_regions` | `List[TissueRegion]`（可 `len()`、`[]`、迭代） |

**為什麼 ds_x ≠ ds_y**：segmenter 可能涵蓋的範圍比要求的小（例如 tiler 丟掉右、下邊的不完整 tile：276 個 224 px 的 tile = 61,824 px，掃描矩形是 61,879 px），用要求的範圍除以 mask 寬會得到 14.012 而不是 14，遠端邊緣漂 55 px。所以 ds 由實際涵蓋的 span 逐軸算。

**region 是 view，不是狀態**：`raw()`、`filtered(min_ratio)`、`merged()`、`patchable(side_l0)` 各回傳一個**新的** `TissueMask`，共用 raster，自己擁有 region list。view 不會漏回原來的 mask，所以沒有東西要還原。

**查詢**：
- `has_tissue_l0(x, y, w, h, tissue_ratio=0.5)`：一個 level-0 矩形的組織比例是否 ≥ 門檻；會裁到 mask 範圍內，mask 外的面積算背景（分母是完整的要求面積）。
- `white_fractions(xy, level, tile)`：每個 tile footprint 的背景比例，只用 mask 算；背後是 summed-area table，超過 `1 << 28` 個 mask px 時用間隔取樣的版本。`TileSampler` 用的是這個。
- 另有 `to_mask_xy`、`region_box`、`get_local_mask`、`read_matching_rgb`。

**通常不手動建立**：`TissueMaskConfig` 是產出 `TissueMask` 的 recipe（`cfg.build(wsi, device)`；一批 slide 用 `MaskMaker(cfg, ...).mask(wsi)`，segmentation 有 cache）。三個模組一個方向：

```
TissueMaskConfig  ->  TissueSegFunc（producer）  ->  TissueMask（product）
recipe + cache        slide -> SlideMask             SlideMask + regions
```
product 不 import 另外兩個。兩個 identity：`seg_id` 管 segmentation（決定 raw mask 的 cache 目錄），`region_id` 管之後的 region 處理；改 `min_region_ratio` 只換 `region_id`，不會重切。

---

### PatchInfo（dataclass）
```python
row, col : int
y, x     : int     # 左上角，ds 座標（level-N，截斷）
size_px  : int     # patch 大小，ds 座標下的 pixel 數
kind     : str     # 'main' | 'overlap'
ds       : float = 1.0
level    : Optional[int] = None
```
工廠方法 `for_query(...)`（ds=1.0）與 `for_wsi(..., ds, level)`。

**`ds` 與 `level` 讓 PatchInfo 自描述座標空間**。`size_px` 還在（舊版 README 寫的是移除它）；它的單位是 ds 座標下的 pixel，query 時就是 query pixel，WSI 時是 level-N pixel，沒有「query 是 pixel、WSI 是 level-0 footprint」的雙重語義。

---

### PatchGrid
只管 layout 與 index，不碰像素也不碰模型。flat / unified index 系統完整（有 overlap 時 flat 順序是 m,o,m,o,...，最後幾行只有 main）。

- `from_size(...)`：給大小與 offset 的 layout 工廠，`QueryPatchContainer.extract_all()` 用。
- `for_region(region, ds, tile_size, overlap=True, level=...)`：**region 的 level-0 box 變成 level-N 大小與 offset 的唯一地方**。會設定 `origin_l0` 與 `ds`，`tile_origin_l0` / `local_to_l0` 靠這兩個。stage 3 的 crop 與 tile 的讀取起點都走 `local_to_l0`，所以兩邊在同一個相位上。
- `region_grids(regions, ds=, level=, tile_size=)`：每個 region 一個 `for_region`，只靠幾何，不讀像素。

---

### PatchContainerBase / QueryPatchContainer
`PatchContainerBase`（ABC）：`extract_all()` 由子類實作；索引與迭代（`__getitem__`、`iter_main`、`iter_overlap`、`iter_batches`）在基底。`QueryPatchContainer` 是單張 query 圖（RGB uint8）切成 `PatchGrid`。

---

### FeaturesMap
`FeaturesMap(grid, features, source='')`：`features[i]` 對應 `grid` 的 flat 順序第 i 格，形狀 `[N, D]`，`N == len(grid)`。

**待決（torch 耦合仍不一致）**：建構子在 torch 有裝時要求 `torch.Tensor`、沒裝時不檢查；但 `main_feature_grid` / `overlap_feature_grid`（`index_select`）與 `from_patch_container` 的 `unsqueeze` 直接用 torch。實質上是 torch-only，只差建構子的檢查條件沒對齊。要嘛明文 torch-only，要嘛把這些 call 改成通用版本。

### WsiFeaturesMap
一張 slide、一個 scale 下，**每個 region 一個 `FeaturesMap`**，並帶著它們所屬的 regions：`regions[i]` 與 `maps[i]` 配對，建構時檢查個數與每個 region 的 patch 數。

**為什麼要有它**：裸的 `list[FeaturesMap]` 不帶「它屬於哪些 region」，配對就落在呼叫端隨手 zip 的變數上（用未過濾的 `mask.tissue_regions` 去配過濾後的 maps，區域就錯位到鄰居身上）；這個 bug 這個 repo 付過不止一次代價。這裡配對本身就是物件。

`from_grid_read(blocks, regions, grids, encoder, ds=, level=, tile_size=, overlap=)`：把 `SlideReader.read_grid` 吐出的 block 一塊一塊編碼進各 region 的 `FeaturesMap`，slide 不會整張放在記憶體裡。每個 slot 必須恰好寫一次，否則 raise（漏讀或重讀會留下分數像真 tile 的零向量）。特徵留在 encoder 回傳的 device 上，只有寫 cache 時才搬到 host（`Store.to_store_tensors`）。

---

## 整體資料流

```
segmenter.segment_slide(wsi)                 # TissueSegFunc（aiNNModel/*SegFunc.py）
    ↓ SlideMask（mask, origin, span, mask_ds）   ← cache 存這一層
TissueMask(wsi, slide_mask)                  # connected components -> tissue_regions
    .filtered(r).merged() / .patchable(side)     # region 的 view
    ↓ regions（level-0）
region_grids(regions, ds=, level=, tile_size=)   # 每個 region 一個 PatchGrid.for_region
    ↓ grids（level-N 的 layout，帶 origin_l0 與 ds）
SlideReader.read_grid(regions, grids, ds, tile=) # 分 block 讀，每 block 若干列
    ↓ blocks
WsiFeaturesMap.from_grid_read(blocks, regions, grids, encoder, ...)
    maps[i] = FeaturesMap(grids[i], features)    # 與 regions[i] 配對
```
query 端：`QueryPatchContainer(image)` -> `extract_all` -> `FeaturesMap.from_patch_container(container, encoder)`。
`TileSampler` 另外用 `mask.white_fractions(...)` 給取樣的 tile 評分，和上面這條資料流無關。
