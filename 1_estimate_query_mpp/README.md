## 1_estimate_query_mpp

### 原則（stage module = 可重用純邏輯）

- **資料流**：Input（data + config）→ Output（結果 + metadata）
- **不做的事**：CLI 解析、print/plot、讀寫結果檔、挑最後候選人的 heuristics（這些放外層 main / orchestrator）
- **輸出要明確**：不要只回傳 tensor/array；請回傳結構化結果（dataclass）

### 方法一覽

#### `estimate_mpp_gigapath.py`

- **目的**：用 GigaPath embedding 對 WSI level tiles 做 KNN，估計 query 的 MPP
- **主要入口**：`estimate_mpp_gigapath(wsi, query, tile_size=256, samples_per_level=40, batch_size=32, device='auto')`
- **Input**：
  - `wsi`: `openslide.OpenSlide`
  - `query`: `QueryPatchContainer` 或可用 `QueryPatchContainer(query)` 建立的來源（路徑 / PIL / ndarray）
  - `tile_size`, `samples_per_level`, `batch_size`, `device`: 設定
- **Output**：`EstimateMppGigapathResult`
  - `estimated_mpp`: 估計 MPP
  - `base_mpp`: WSI `openslide.mpp-x`
  - `tile_size`, `samples_per_level`, `k`, `query_patch_count`: metadata

