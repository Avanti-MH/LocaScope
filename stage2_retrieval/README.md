## 2_retrieval

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

#### `GigaPathSlidingWinSimRot.py` — canonical

`LocaScopePipeline` 唯一呼叫的檢索器（`utilities/LocaScopePipeline.py`）。試 4 個
cardinal 旋轉（0/90/180/270 度），挑分數最高的方向，因此下游 `SiftRansacLocalizer`
才拿得到 `best_rotation` 去對齊 query。

- **主要入口**：`GigaPathSlidingWinSimRot(wsi, encoder, mask=None, mpp=None, tile_size=256, overlap=True)`
- **四階段**：
  1. `build_wsi_features(mpp)` — 對 WSI tile 化並編碼，只做一次
  2. `build_query_features(query)` — 對 query 的 4 個旋轉各自抽 patch + 編碼
  3. `compute_sim_maps()` — 4 組相似度圖
  4. `find_best()` → `SlideWinSimRotResult`（`best_x/y`、`best_rotation`、`scores_by_rotation`）
  - `top_k(k)` — 同一批分數的完整排名，用於驗證候選（`SiftRansacLocalizer` 可直接吃 `SlideWinSimCandidate`，鴨子定型）
- **測試**：`utilities/test_modules/test_gigapath_slide_win_sim.py` 的第 5 步
  （旋轉恢復四個角度 + 對照非旋轉版；步驟 1-4 是同一個檔案測非旋轉版的原始
  內容，兩者合併進同一支腳本，因為分開跑會把同一次裁切、同一次模型載入的
  GPU 成本付兩次）

#### `GigaPathSlidingWinSim.py` — 一個檔案，兩種狀態

`SlidingWindowSimilarity` 是 **primitive**：真正共用的相似度核心，
`GigaPathSlidingWinSimRot` 在內部 import 它，自己沒有獨立呼叫端。

`GigaPathSlidingWinSim` class 與 `compute_gigapath_sliding_win_similarity` 是
**legacy**：只搜尋 query 給定的單一方向，不會找旋轉——這正是 Rot 版取代它的
理由。`LocaScopePipeline` 不呼叫它們，但還沒被移除，因為
`utilities/test_modules/test_gigapath_slide_win_sim.py`（步驟 1-4）和
`test_sift_ransac.py` 還在測這條路徑，尚未依畫布審查的結論重新分類——現在兩
條路徑的測試結果會一起跑出來，決定去留時兩邊的證據都在同一份輸出裡。
