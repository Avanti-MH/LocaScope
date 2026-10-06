## stage3_localization

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

- **入口**：`SiftRansacLocalizer(min_inliers=10, padding=2).build(wsi)`，然後
  `localize(query, CandidateSet, rank=0) → SiftRansacResult`：吃 stage 2 的輸出，用自己
  `build` 建的 `SlideReader` 讀圖，不向 retriever 借。
- **三步**（畫圖時可逐步跑，先 `prepare(query, cs, rank)`）：
  1. `read_wsi_crop(padding)` — 候選視窗 `±padding` tile，夾在它的 region 內，在候選集的
     level 上讀；`crop_origin_l0` 是這次讀圖真正的 level-0 整數起點
  2. `detect_and_match()` — query 與 crop 的 SIFT keypoint + BFMatcher(knn=2) + Lowe ratio
  3. `estimate_homography()` — RANSAC H（query px → crop px）；crop 的像素 (u, v) 在 level-0
     的 `crop_origin_l0 + (u, v) * ds`，所以 query 的 (0, 0) 和中心經 H 後直接得到 level-0
     位置，保留小數。inlier 不足或 H 退化時 fallback 回候選視窗本身的位置。
- **Output** `SiftRansacResult`：`x0/y0`、`center_x0/y0`（level-0，float）、`H`、
  `inlier_count`、`match_count`、`success`、`rank`、`candidate`、`ds`、`level`。
  中心點優先於左上角：query 繞自己中心旋轉，中心不受旋轉影響。
- **2026-10-05 前的記帳**把 crop 原點記成 `int(region.x / ds) + x0`，截掉 region 原點的
  小數；openslide 以 bilinear 取樣，所以回報位置偏 `-frac(region.x / ds) * ds` 個 level-0
  像素（BRACS L1 約 1 µm、L2 約 4 µm）。現在直接記讀圖起點；新舊記帳在已知位置上的
  比對（舊的誤差等於 -frac，新的 < 0.025 level 像素）記在 `log/TODO.log`。
