## stage1_estimation

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
  模組 import 它；通常在 `*/cli/` 或 `tests/` 底下。

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
| `chosen_ds` / `chosen_mpp` / `chosen_level` | 貼到目標 WSI 實際存在的 pyramid level 後的答案，`StageInterface.routed_level`（`ReadGeometry.coarser_level`）算的 |

沒有 `method: str` 欄位——`type(result)` 本身就是方法識別。`MppEstimator`
（一個 `typing.Protocol`）只規定 `estimate(query: np.ndarray) -> EstMppResult`
這一個形狀；每個方法自己的 build 階段形狀不強行統一——一個從目標 WSI 現場建參
考庫（貴），一個載入離線訓練好的權重（便宜），硬統一介面只會掩蓋這個真實差異。

### 方法一覽

#### `KnnEstMpp.py` — canonical

CLI 與 bench 的預設估計器（`locate_photo.py`、`BenchCommon.py` 的 `--stage1` 預設
`knn:gigapath`）。`LocaScopePipeline` 本身不綁方法：stage 1 以 spec（`"method:recipe"`）或已建好的
物件傳入，`stage1_estimation/__init__.py` 的 `METHODS` 登記了 `knn`、`classifier`、`prototype`、
`classic` 四種，沒給 spec 就不跑 stage 1。「canonical」在這裡指預設與基準，不是唯一可呼叫。
K-近鄰投票：從目標 WSI 自己的 pyramid 現場抽參考 tile、編碼，query 也切成
patch 編碼後去投票。

- **Config**：`KnnEstMppConfig(encoder, mask_cfg, sampler_cfg, k=5, tile_size=256, n_per_level=None)`；
  recipe 表 `KNN_RECIPES`（`gigapath`、`uni2`：`n_per_rung=40, seed=42`）
  - `encoder`：`TileEncoder` 的註冊名稱（如 `'gigapath'`），建構時自己
    用這個名稱 build 一個 encoder，不吃外部傳入的物件
  - `mask_cfg`：`TissueMaskConfig`，預設 `MASK_RECIPES['hest']`，決定哪些區域算組織
  - `sampler_cfg`：`TileSampler.SamplerConfig`，決定參考庫怎麼抽（每層抽幾張、seed、
    richness caps/floors、overlap）；預設 `SAMPLER_RECIPES['reference-bank']`（只收背景比例
    <50% 的 tile，三個桶等權，不偏好任何一桶）。`SamplerConfig` 沒有 tile 大小
  - `tile_size`：參考 tile 與 query patch 共用的唯一尺寸（兩邊必須同大小，投票才比得起來）
  - `n_per_level`：投票用每層抽樣的前幾張；`None` 為全用。快取的是整份抽樣
    （`sampler_cfg.n_per_rung`），這個欄位只從中取，所以不同的 bank 大小共用同一份抽樣
- **建置**：`KnnEstMpp(cfg, device)` 建構時就把 encoder 建好；`.build(wsi, mask=None, *, masks=None,
  cache_job=None)` 才是真正貴的一步——對「這個 WSI」現場抽參考 tile、編碼（`mask=None`
  時用 `cfg.mask_cfg` 自己 segment；呼叫端想跨 stage 共用同一個 mask 就自
  己傳進來；`cache_job` 連同 `masks`（`mask` 出自的 MaskMaker）一起給，抽樣與編碼後的特徵
  各存成一筆 cache，同一張片第二次 build 不再編碼）
- **每次 query**：`.estimate(query, overlap=True)` — 切 patch、編碼、KNN
  median-of-medians 投票 → `KnnEstMppResult`
- **Output**：`KnnEstMppResult(EstMppResult)`，沒有額外欄位

#### `ClassifierEstMpp.py` — challenger

用 `training/ClassifyingRoutingHead/` 訓練出來的分類頭做 mpp 估計。要在 stage 1 bench
（`bench_stage1_mpp.py`、`analyze_stage1_metrics.py`）上打贏 `KnnEstMpp` 才能取代
它成為預設；pipeline 以 `--stage1 classifier:<recipe>` 就能呼叫它，只是預設不是它。

模型離線訓練好、跟任何一張 WSI 無關，所以跟 `KnnEstMpp` 正好相反：建構貴
（要載入 encoder+head），綁定 WSI 便宜（只需要 `wsi.base_mpp`）。

- **Config**：`ClassifierEstMppConfig(encoder, classifier, reduction, tile_size, weights, vote='mean_probability')`；
  recipe 表 `CLASSIFIER_RECIPES`（`uni2-mlp-ord-b`，`vote='quality_weighted:tissue'`）
  - `encoder`：`TileEncoder` 的註冊名稱
  - `classifier`：`locascope/core/networks/Heads.py` 的註冊名稱（`'linear'`/`'mlp'`/`'arcface'`）
  - `reduction`：`'fixed'`（用 encoder 自己的 CLS/GAP）、`'attn'`（學一個
    attention pooling）或 `'clsattn'`（`common.Head.REDUCTIONS`）
  - `weights`：訓練好的 checkpoint 路徑（不進 identity；權重變了由 `weights_id` 反映）
  - `vote`：`FoVVote.VOTE_CHOICES` 的 key（可帶選項，如 `quality_weighted:tissue`、
    `patch_class_median:lower`）。這是推論時的選擇，不存在 checkpoint 裡——訓練只看單張 tile，
    沒有「整個 FoV」可記；進 identity，只差 `vote` 的兩次跑是兩個實驗
  - `ClassifierEstMppConfig.from_checkpoint(weights, vote=...)` 直接從 checkpoint
    讀出 `encoder`/`classifier`/`reduction`/`tile_size`，不用手填；
    該 head 混哪幾個 encoder block（`encoder_layers`）也隨 checkpoint 的 `head_cfg` 走，
    不是 config 欄位
- **建置**：`ClassifierEstMpp(cfg, device)` 建構時載入 encoder+head，並檢查
  `cfg` 跟 checkpoint 實際內容是否吻合（手動建的 config 若跟檔案內容不符
  會直接報錯）；`.build(wsi, mask=None, *, masks=None, cache_job=None)` 只是綁定這個 WSI 的
  `base_mpp`（`mask`／`masks`／`cache_job` 收了不用，因為不從片子上抽樣）
- **每次 query**：`.estimate(query)` — 切成 tile，每張 patch 算 softmax
  機率，再由 `cfg.vote` 指名的 `FoVVote` 函式把整個 FoV 的機率合成一票，贏家 class 換
  算成 `estimated_ds`（訓練時的 rung 是相對 WSI 自己 base_mpp 的 ds 倍
  數，`rungs` 隨 checkpoint 一起存，不重新 import 訓練套件）
- **Output**：`ClassifierEstMppResult(EstMppResult)` 加兩個欄位：
  `predicted_class`（贏家 class index）、`vote_extra`（`Dict[str, float]`，該 vote 函式自己
  回報的東西：`mean_probability` 給 `confidence`，`quality_weighted` 另有
  `effective_sample_size` 等；形狀隨 vote 而異，所以不是每種 vote 一個欄位）

#### `PrototypeEstMpp.py` — challenger

用 `training/PrototypicalRoutingHead/`（metric-based／episodic）訓練出來的 checkpoint 做 mpp 估計。
建置成本落在 `build(wsi)`，跟 `KnnEstMpp` 同理、與 `ClassifierEstMpp` 相反：它沒有訓練好的固定
class anchor，要在目標 WSI 上現場抽並編碼 reference bank（相當於訓練時的 support 那一邊），再
過一次 checkpoint 自己的 Stage 2（support_context／collapse），之後每次 `estimate(query)` 就便宜。
跟 `KnnEstMpp` 互相獨立，各自擁有自己的 build 設定；參考庫的取樣 recipe 都按名稱取
`SAMPLER_RECIPES['reference-bank']`，要調其中一個是給它自己一個新 recipe，不動另一個。

- **Config**：`PrototypeEstMppConfig(encoder, pooling, support_context, query_context, collapse,
  routing_head, tile_size, weights, mask_cfg, sampler_cfg, levels=None, vote='mean_probability',
  n_per_level=None)`；前七個架構欄位用 `PrototypeEstMppConfig.from_checkpoint(weights, ...)`
  從 checkpoint 讀，`mask_cfg`／`sampler_cfg` 是一般預設，不從 checkpoint 推。`levels=None` 為
  取樣每一個原生 pyramid level。recipe 表 `PROTOTYPE_RECIPES`：
  `uni2-attn-set-transformer-attn-score-cross-on`（`vote='sum_log_probability'`，`n_per_rung=5`）
- **Output**：`PrototypeEstMppResult(EstMppResult)` 加 `predicted_level`（索引的是這次 `build` 對該 WSI
  實際有的原生 level 所排出的 `level_order`，不是 `Datasets.RUNGS` 的 class index）與 `vote_extra`

#### `estimate_mpp_classic.py` — baseline

不看位置，比對「倍率指紋」（頻率重心 + 自相關長度），跟任何深度學習表徵完全獨立——這正是 baseline 的判準：它從來不是 canonical 的候選，存在的價值是拿來比較。

- **介面**：`ClassicEstMpp(CLASSIC_RECIPES['default']).build(wsi, mask=None).estimate(query)`（`ClassicEstMppConfig(tile, samples, k, min_std, seed, seg, sampler_cfg)`；pipeline 用 `--stage1 classic:default`），回傳 `ClassicEstMppResult(EstMppResult)`，多兩個欄位 `query_fingerprint`、`neighbour_levels`。`bench_stage1_mpp --classic` 用它和其他方法在同一批 FoV 上比較。
- **建置**：每個原生層由 TileSampler 在 tissue mask 內放 `samples` 個位置、SlideReader 讀出，灰階標準差低於 `min_std` 的不計，兩個指紋各取中位數。
- **估計**：query 中央裁切 `tile`（不縮放）算指紋，兩維 z-score 後做距離加權 KNN，在 log-mpp 上平均。
- 2026-10-05 前是只有 CLI 的腳本：在整層上無種子地隨機抽位置、自己 read_region，單一指紋的版本只 print 不回傳。
