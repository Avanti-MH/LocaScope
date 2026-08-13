## 3_localization

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

#### `SIFT_RANSAC.py` — canonical，唯一實作

`LocaScopePipeline` 唯一呼叫的定位器（`utilities/LocaScopePipeline.py`）。把
retrieval 給的 tile 級精度（誤差 ≤ 1 tile）用 SIFT keypoint + RANSAC homography
細化到 sub-pixel。

- **主要入口**：`SiftRansacLocalizer(wsi_container, query, location, min_inliers=10, ...)`
  - `location` 接受 `SlideWinSimRotResult` 或 `SlideWinSimCandidate`（鴨子定型：
    只讀 `best_region_index`、`best_x`、`best_y`、`ds`、`best_rotation` 五個屬性）
- **三階段**：
  1. `read_wsi_crop(padding)` — 以 retrieval 給的位置為中心，取 `±padding` tile 的 crop
  2. `detect_and_match()` — query 與 crop 的 SIFT keypoint + BFMatcher(knn=2) + Lowe ratio
  3. `estimate_homography()` → `SiftRansacResult`（`H`、`inlier_count`、`success`，
     RANSAC 失敗或 inlier 不足時 fallback 回 retrieval 的 `best_x/y`）
- **Output** `SiftRansacResult` 的 `center_x/y`（連同 `center_x0/y0`）優先於
  `x/y`：query 是繞自己中心旋轉的，中心點不受旋轉影響，跟未知方向的 ground
  truth 比較時才站得住。

> ⚠ `location: SlideWinSimResult` 這個型別標註（`SIFT_RANSAC.py:15`）指向
> `2_retrieval/GigaPathSlidingWinSim.py` 的非旋轉版本，但 production 實際傳入
> 的是 `SlideWinSimRotResult`——能動是因為鴨子定型，標註本身在說謊。抽出一個
> Stage 2/3 共用的 result protocol 是畫布審查排定的下一步，這裡先誠實記下來。
