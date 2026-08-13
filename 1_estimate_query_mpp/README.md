## 1_estimate_query_mpp

### 原則（stage module = 可重用純邏輯）

- **資料流**：Input（data + config）→ Output（結果 + metadata）
- **不做的事**：CLI 解析、print/plot、讀寫結果檔、挑最後候選人的 heuristics（這些放外層 main / orchestrator）
- **輸出要明確**：不要只回傳 tensor/array；請回傳結構化結果（dataclass）

### 檔案狀態標籤

每個模組標一個狀態，判準是「這是誰在用」，不是主觀印象：

- **canonical** — 這個 stage 目前真正被呼叫的實作。一個 stage 最多一個；
  `LocaScopePipeline`（或它委派的呼叫端）import 的就是它。
- **primitive** — 沒有自己的呼叫端，是被別的實作（canonical 或 legacy）拆出來
  共用的邏輯，通常是一個函式而非整個模組。
- **legacy** — 曾經負責跟 canonical 一樣的責任、現在被取代了，但還沒被移除
  ——通常是因為還有測試在測它，或它的一部分被 canonical 當 primitive 用。
  目標是被替換掉，不是被擴充。
- **baseline** — 從一開始就不是 canonical 的候選，是刻意獨立的對照方法，存在
  的價值是拿來比較，不是要接手 production。
- **diagnostic** — 拿來回答「系統現在的狀態／行為是什麼」的工具，沒有任何
  模組 import 它；通常在 `*/cli/` 或 `utilities/test_modules/` 底下。

legacy 跟 baseline 最容易混——判準是「它曾不曾是跟 canonical 同一個責任的候
選」：legacy 曾經是（或就是被取代前的 canonical），baseline 從來就不是。

### 方法一覽

#### `GigaPathKnnEstiMpp.py` — canonical

`LocaScopePipeline` 唯一呼叫的估計器（`utilities/LocaScopePipeline.py`）。

- **主要入口**：`GigaPathKnnEstiMpp(wsi, encoder, mask=None, tile_size=256, samples_per_level=40, k=5, seed=42, tissue_ratio=0.5)`
- **四階段**（依序，晚一步呼叫會自動補跑前面漏掉的）：
  1. `build_samples()` — 從 WSI 金字塔各層抽參考 tile
  2. `build_ref_features()` — 編碼參考 tile → `KnnClassifier`
  3. `build_query_features(query)` — 編碼 query patch → `FeaturesMap`
  4. `estimate()` — KNN 投票 → `GigapathKnnEstiMppResult`
- **Output**：`GigapathKnnEstiMppResult`
  - `estimated_mpp`：估計 MPP
  - `base_mpp`：WSI `openslide.mpp-x`
  - `tile_size`, `samples_per_level`, `k`, `query_patch_count`：metadata

#### `estimate_mpp_classic.py` — baseline

沒有任何檔案 import 它；只有自己的 `if __name__ == '__main__'` CLI（`python estimate_mpp_classic.py slide.svs query.jpg`）。不看位置，比對「倍率指紋」（頻率重心 + 自相關長度），跟 GigaPath 的表徵完全獨立——這正是 baseline 的判準：它從來不是 canonical 的候選，存在的價值是拿來比較。
