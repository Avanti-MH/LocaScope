## stage2_retrieval

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

#### `StageInterface.py` — 介面（設計見 `spec.md`）

- `Candidate(region_index, lattice, row, col, rotation, score)`：只放檢索自己找到的東西。
- `CandidateSet`：候選依分數排好（位置就是名次），加上它的座標框架——`level`、`ds`、
  每個 region 的 `PatchGrid`（tile 大小也從它拿）。視窗大小是 query 的 tile 格，由方法吃 query 算出。下一站從自己的輸入就拿得到的
  東西不放進來。
- 位置只從 `CandidateSet` 的方法來：`origin_l0`（視窗讀圖的 level-0 整數起點，
  `PatchGrid.tile_origin_l0`）、`window_l0`、`centre_l0`、`window_local`。不經過截斷的
  level-n 座標——openslide 以 bilinear 取樣，`int(region.x / ds)` 會偏一個 level 像素以內。
- `Retriever`、`Reranker` 兩個 Protocol；`Reranker` 還沒有實作。

#### `SlidingWinSimRot.py` — canonical

`LocaScopePipeline` 唯一呼叫的檢索器。試 4 個 cardinal 旋轉（0/90/180/270 度），
所有旋轉、兩組格點、所有 region 的視窗一起排名。視窗相似度核心
`SlidingWindowSimilarity` 也在這個檔案（2026-10-05 從已刪除的
`GigaPathSlidingWinSim.py` 搬進來），window bench 與 off-grid bench 都從這裡 import。

- **入口**：`SlidingWinSimRot(SlidingWinSimRotConfig(encoder_cfg, tile_size=256,
  overlap=True, k=20, min_sep_tiles=1.0), device, multi_gpu=False).build(wsi, mask, feature_store=None)`，
  encoder 由 `encoder_cfg`（`TileEncoderConfig`，含精度與 batch）自己建，和 stage 1 一樣不收外部的；然後
  `retrieve(query, EstMppResult) → CandidateSet`：在 stage 1 的 `chosen_level` 上檢索，
  該層的格點與特徵第一次用到時才建（cache 命中不讀圖），之後快取。
- **步驟**（bench 自己控制尺度時用）：`build_wsi_features(mpp= | ds= | level=)`、
  `build_query_features(query)`、`compute_sim_maps()`、`candidate_set(k)`。
- `min_sep_tiles`：同一旋轉下，視窗起點彼此距離小於這麼多 tile 的候選只留分數高的
  （一個強峰會從 main 和 offset 兩組格點重複出現）；不同旋轉不互相抑制。
