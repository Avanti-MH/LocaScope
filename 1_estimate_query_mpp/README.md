## 1_estimate_query_mpp

### 原則（stage module = 可重用純邏輯）

- **資料流**：Input（data + config）→ Output（結果 + metadata）
- **不做的事**：CLI 解析、print/plot、讀寫結果檔、挑最後候選人的 heuristics（這些放外層 main / orchestrator）
- **輸出要明確**：不要只回傳 tensor/array；請回傳結構化結果（dataclass）

### 檔案狀態標籤

每個模組標一個狀態，判準是「這是誰在用」，不是主觀印象：

- **canonical** — 這個 stage 目前真正被呼叫的實作。一個 stage 最多一個；
  `LocaScopePipeline`（或它委派的呼叫端）import 的就是它。
- **challenger** — 還沒被任何呼叫端採用，但有明確的評分場（scorecard）要打贏
  canonical 才能取代它。跟 canonical 的差別只在「還沒贏」，不是永遠的對照組。
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
challenger 跟 legacy 的差別是時間方向：legacy 從 canonical 退下來，
challenger 還沒升上去。

### 共用介面：`StageInterface.py`

每個估計器的 `.estimate(query)` 都回傳一個繼承 `EstMppResult` 的 dataclass，
統一有五個欄位：

| 欄位 | 意義 |
|---|---|
| `estimated_ds` / `estimated_mpp` | 方法自己算出來的原始答案（ds、mpp 兩種單位都給） |
| `chosen_ds` / `chosen_mpp` / `chosen_level` | 貼到目標 WSI 實際存在的 pyramid level 後的答案，`SafeSlide.coarser_level_for_downsample()` 算的 |

沒有 `method: str` 欄位——`type(result)` 本身就是方法識別。`MppEstimator`
（一個 `typing.Protocol`）只規定 `estimate(query: np.ndarray) -> EstMppResult`
這一個形狀；每個方法自己的 build 階段形狀不強行統一——一個從目標 WSI 現場建參
考庫（貴），一個載入離線訓練好的權重（便宜），硬統一介面只會掩蓋這個真實差異。

### 方法一覽

#### `KnnEstMpp.py` — canonical

`LocaScopePipeline` 目前唯一呼叫的估計器（`utilities/LocaScopePipeline.py`）。
K-近鄰投票：從目標 WSI 自己的 pyramid 現場抽參考 tile、編碼，query 也切成
patch 編碼後去投票。

- **Config**：`KnnEstMppConfig(encoder, mask_cfg=TissueMaskConfig(), sampler_cfg=SamplerConfig(...), k=5)`
  - `encoder`：`TileEncoderFunc` 的註冊名稱（如 `'gigapath'`），建構時自己
    用這個名稱 build 一個 encoder，不吃外部傳入的物件
  - `mask_cfg`：`TissueMaskConfig`，決定哪些區域算組織
  - `sampler_cfg`：`TileSampler.SamplerConfig`，決定參考庫怎麼抽（tile
    大小、每層抽幾張、seed、richness caps/floors、overlap）；query patch
    切多大也讀這裡的 `sampler_cfg.tile`，跟參考 tile 用同一個尺寸
  - 預設 richness 是 `REFERENCE_BANK_RICHNESS`（只收背景比例 <50% 的
    tile，三個桶等權，不偏好任何一桶）
- **建置**：`KnnEstMpp(cfg, device)` 建構時就把 encoder 建好；`.build(wsi, mask=None)`
  才是真正貴的一步——對「這個 WSI」現場抽參考 tile、編碼（`mask=None`
  時用 `cfg.mask_cfg` 自己 segment；呼叫端想跨 stage 共用同一個 mask 就自
  己傳進來）
- **每次 query**：`.estimate(query, overlap=True)` — 切 patch、編碼、KNN
  median-of-medians 投票 → `KnnEstMppResult`
- **Output**：`KnnEstMppResult(EstMppResult)`，沒有額外欄位

#### `ClassifierEstMpp.py` — challenger

用 `training/MppRoutingHead/` 訓練出來的分類頭做 mpp 估計。要在
`bench_mpp_feature_decomposition.py` 的 `sampler_routing` scorecard
（`level_accuracy`、`mpp_error_relative_p50`）上打贏 `KnnEstMpp` 才能取代
它成為 canonical；`LocaScopePipeline` 目前不呼叫它。

模型離線訓練好、跟任何一張 WSI 無關，所以跟 `KnnEstMpp` 正好相反：建構貴
（要載入 encoder+head），綁定 WSI 便宜（只需要 `wsi.base_mpp`）。

- **Config**：`ClassifierEstMppConfig(encoder, classifier, reduction, tile_size, weights)`
  - `encoder`：`TileEncoderFunc` 的註冊名稱
  - `classifier`：`aiNNModel/models/Heads.py` 的註冊名稱（`'linear'`/`'mlp'`/`'arcface'`）
  - `reduction`：`'fixed'`（用 encoder 自己的 CLS/GAP）或 `'attn'`（學一個
    attention pooling）
  - `weights`：訓練好的 checkpoint 路徑
  - `ClassifierEstMppConfig.from_checkpoint(weights)` 直接從 checkpoint
    讀出 `encoder`/`classifier`/`reduction`/`tile_size` 四個欄位，不用手填
- **建置**：`ClassifierEstMpp(cfg, device)` 建構時載入 encoder+head，並檢查
  `cfg` 跟 checkpoint 實際內容是否吻合（手動建的 config 若跟檔案內容不符
  會直接報錯）；`.build(wsi)` 只是綁定這個 WSI 的 `base_mpp`
- **每次 query**：`.estimate(query)` — 切成 tile，每張 patch 算 softmax
  機率、加總成 soft-vote（票的權重就是各自的 probability），贏家 class 換
  算成 `estimated_ds`（訓練時的 rung 是相對 WSI 自己 base_mpp 的 ds 倍
  數，`rungs` 隨 checkpoint 一起存，不重新 import 訓練套件）
- **Output**：`ClassifierEstMppResult(EstMppResult)` 加兩個欄位：
  `predicted_class`（贏家 class index）、`confidence`（贏家拿到的平均票
  權重，衡量票有多集中）

#### `estimate_mpp_classic.py` — baseline

沒有任何檔案 import 它；只有自己的 `if __name__ == '__main__'` CLI（`python estimate_mpp_classic.py slide.svs query.jpg`）。不看位置，比對「倍率指紋」（頻率重心 + 自相關長度），跟任何深度學習表徵完全獨立——這正是 baseline 的判準：它從來不是 canonical 的候選，存在的價值是拿來比較。
