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

#### `Retriever.py` — 介面（設計見 `blueprint.md`）

- 每個 stage 2 方法都是 `Retriever`：`retrieve(upstream, query, topk=None) → RetrievalOutput`。
  第一階段方法的上游是 stage 1 的 `EstMppResult`，reranker 的上游是另一個 retriever 的 `RetrievalOutput`；
  除此之外沒有差別。`RetrievalOutput` = 候選名單（`CandidateSet`）＋自己定義的副產物（`extras`）＋上游（`parent`）。
- 所有參數在 config 決定（id 在建構前就固定）；名單長度是 config 的 `raw_topk`（reranker 讀的是 `topk_in`），
  消費者呼叫時選的 `topk` 只取前綴，不進 id。
- `EncoderPool`：encoder 依 trunk 共用（uni2 的 `cls` 與 `cls_avg` 是同一個載入的模型），trunk 或權重不同就 raise。
- 橋接鎖（`take`）：config 的 `reuse` 宣告要用上游的 `level_features`；上游提供但 key 對不上 → `BridgeError`；
  上游沒提供（名單從 cache 讀回）→ 下游自己做。

#### `StageInterface.py` — 候選與座標框架

- `Candidate(region_index, lattice, row, col, rotation, score)`：只放檢索自己找到的東西；方法專屬的欄位用子類別
  （`RerankedCandidate`、`ShiftedCandidate`、`TokenMatchCandidate`、`VoteCandidate`）。
- `CandidateSet`：候選依分數排好（位置就是名次），加上它的座標框架——`level`、`ds`、每個 region 的 `PatchGrid`。
- 位置只從 `CandidateSet` 的方法來：`origin_l0`、`window_l0`、`centre_l0`、`window_local`。不經過截斷的
  level-n 座標——openslide 以 bilinear 取樣，`int(region.x / ds)` 會偏一個 level 像素以內。

#### 第一階段 retriever（各自編碼，彼此對等）

| 檔案 | `--stage2` | 打分 |
|---|---|---|
| `RetrieverSlidingWinSimRot.py` — canonical | `slidewin:<recipe>` | 逐 tile 餘弦，`mean` / `geomean` / `min` |
| `RetrieverBagOfTiles.py` | `bagoftiles:<recipe>` | 視窗平均向量（`gap`）或四象限（`quad`） |
| `RetrieverTileVote.py` | `tilevote:<recipe>` | query tile 找最近的 tile 投票（`exact` / `hnsw` / `ivf`） |
| `RetrieverClusterBow.py` | `clusterbow:<recipe>` | tile 分群的直方圖交集 |
| `RetrieverWinEmbed.py` | `winembed:<recipe>` | 學習出的視窗向量 |

共用的走法（從 routed level 往細的 level 退、編碼 level 與 query、top-k 去重）在 `Context.py`
（`SlideLevels`、`WindowSearch`、`top_candidates`）。`SlidingWindowSimilarity` 也在 `RetrieverSlidingWinSimRot.py`。

#### reranker（吃上游名單的 retriever）

- `RerankerScorers.py`：Dual、CSLS、Corr、CandGraph（免訓練，讀特徵）；`ColumnRanker` 是它們共用的做法
  （讀前 `topk_in`、打分、依 config 的欄位排序、欄位當 `columns` 副產物往下傳）。
- `RerankerEncode.py`：Shift、TokenMatch、XEnc（現場讀圖編碼）。
- `RerankerNeural.py`：TileGT、CorrNet、Stack（載入權重）；`neural_inputs` 等訓練共用的輸入也在這裡。

#### `MultiPhases.py` — 串接

`MultiPhasesConfig(phases=(...))`：一個接一個，前一個的輸出是下一個的上游。`--stage2 multiphases:<recipe>`
或 `--stage2 arm:<arm>`（arm 文法見 `ArmSpec.py`，`config_of_arm` 轉成 config）。每一段是自己的 cache entry
（`Naming.py`，`_after-<上游 hash>`），pipeline 與 bench 同一個 config 就是同一個檔。

- 網路在 `aiNNModel/models/RerankHeads.py`，訓練在 `training/RerankHead/`，評估是
  `utilities/bench_modules/bench_stage2_retriever.py`（arm 是任何 retriever 串接）。
