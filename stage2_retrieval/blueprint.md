# Stage 2 重構藍圖

本文件是 stage 2 重構的施工依據：目標、介面、串接方式、每個檔案的架構與施工順序。附錄 A 是各方法的
資料流（原 `dataflow.md`），算法在這次重構中不變。

---

## 〇、目標

1. **所有 stage 2 方法都是對等的 `Retriever`。** 第一階段方法（SlidingWinSimRot、BagOfTiles、TileVote、
   ClusterBow、WinEmbed）吃 stage 1 的輸出；reranker（Dual、CSLS、Corr、CandGraph、Shift、TokenMatch、
   XEnc、TileGT、CorrNet、Stack）吃另一個 retriever 的輸出。除此之外沒有差別，誰也不寄生在誰身上。
   SlidingWinSimRot 的角色如同 `GigaPathFunc`，其他第一階段方法是和它抗衡的對等實作；reranker 堆疊在上面。
2. **`Retriever.py` 只放最小的介面和定義**（對應 `aiNNModel/TileEncoderFunc.py`）。參數、attribute 由子類自己
   定義。廢除 `CoRetriever`（`CoRetrieve.py`）、`CoStage.py`、`Union.py`、`Rerank.py` 的多 scorer 框架；
   `TwoPhase.py` 改為 `MultiPhases.py`。不需要 `Reranker.py`：reranker 就是 Retriever。
3. **config 決定一切。** 所有影響算法的參數在 config 階段決定，id 在建構前就固定。例外是消費者在呼叫時
   選的 `topk`（只取前綴，不影響算法）。
4. **encoder 共用接口**：可以接收上游（或呼叫端）已建好的 encoder，省下建構與記憶體；也可以自己建。接收時
   和自己 config 的設定比對，不合就 raise，讓 pipeline 停下來，絕不改用別人的。
5. **副產物橋接鎖**：下游宣告要用上游的副產物、上游有提供但 key 對不上 → raise。上游沒有提供（例如上游的
   名單是從 cache 讀回的）→ 下游自己做出它要的東西。下游沒宣告 → 完全不看上游的副產物。
6. **pipeline 和 bench 用同一個 config，就是同一個 id、同一個 cache entry。** 每一段一個 entry，以
   `_after-<上游 hash>` 串起來。
7. **算法結果與重構前完全一致**，只有 id 與名稱改變；用重構前保留的 cache 驗證。

---

## 一、名詞與規則

| 名詞 | 意思 |
|---|---|
| 第一階段 retriever | 上游是 stage 1 的 `EstMppResult`，自己找出整層的候選名單 |
| 下游 retriever（reranker） | 上游是另一個 retriever 的 `RetrievalOutput`，重排它收到的名單 |
| `raw_topk` | config 欄位，進 id：這個 retriever 產生的名單長度（每張分數圖的預選量也用它）。預設 **1000** |
| `topk` | 消費者呼叫時選的長度，不進 id：從 `raw_topk` 的名單取前 `topk` 個；不給就是全部 |
| `topk_in` | reranker 的 config 欄位，進 id：讀上游名單的前幾個。reranker 的輸出長度就是 `topk_in` |
| `@K`（arm 文法） | **消費者使用的 topk**。`A@100::B@10`：B 讀 A 的前 100（B 的 `topk_in = 100`），B 的消費者（下一段、報表或 stage 3）讀 B 的前 10 |

- level 上的視窗不足 `raw_topk` 時，內部函式輸出所有可用的，**config 不修正**；實際列出幾個記在產物裡
  （`output` 表每列的 `n_listed`）。
- `topk > raw_topk`（或 reranker 的 `topk_in` 大於上游 `raw_topk`）→ raise。
- K 的前綴性質：每張分數圖的預選量用 `raw_topk`，不再用消費者的 K，所以任何 `topk` 都是同一份名單的前綴。
  重構前 k=1000 的結果 ≡ `raw_topk = 1000`。

---

## 二、`Retriever.py`：最小介面

```
Retriever.py
├── RetrieverConfig(IdentifiedConfig)     沒有欄位；慣例：名單長度欄位叫 raw_topk
├── Extra            (frozen)  kind: str, key: tuple, value: Any       一份副產物
├── RetrievalOutput            candidates: CandidateSet                第一層：候選名單（必備）
│                              extras: Dict[str, Extra]                第三層：自己定義的副產物
│                              parent: Optional[RetrievalOutput]       上游的輸出（整條鏈可回溯）
│                              producer: str                           產出者（entry 名）
│   └── find(kind) -> Optional[Extra]     沿 parent 往上找
├── Uses             (frozen)  kind, key                               下游宣告要用的上游副產物
├── BridgeError(RuntimeError)
├── EncoderPool                                                        ← 由 Context.py 搬來並改寫
│   ├── get(cfg) -> TileEncoder           依 trunk（去掉 pooling 的 encoder config + weights_id）共用，
│   │                                     回傳套上 cfg.pooling 的 view
│   └── offer(built)                      登記外部建好的 encoder，以它自己的 cfg 為 key
├── check_encoder(built, wanted_cfg)      trunk 或 weights_id 不同 → raise
└── Retriever(IdentifiedBuild)            [抽象]
    ├── UPSTREAM: Tuple[type, ...]        可接受的上游型別：(EstMppResult,) 或 (RetrievalOutput,)
    ├── __init__(cfg, device=None, *, encoders: Optional[EncoderPool] = None)
    │                                     沒給就自己建；有給就經由它取得並比對
    ├── encoder_needs() -> Dict[str, TileEncoderConfig]   要用哪些 encoder（讓呼叫端知道能共用什麼）
    ├── uses() -> Tuple[Uses, ...]        橋接鎖宣告，預設 ()；由 config 的 reuse 欄位決定
    ├── build(wsi, mask) -> self          綁定 slide（第二層：wsi、mask）
    ├── retrieve(upstream, query, *, topk=None) -> RetrievalOutput    [基類實作，子類不覆寫]
    │     檢查 upstream 型別 → 檢查橋接鎖 → self.run(...) → 取前 topk
    ├── run(upstream, query) -> RetrievalOutput           [抽象，子類實作]
    └── tables(output, qc) -> Dict[str, List[dict]]        可選：副產物寫成表，預設 {}
```

**三層輸入**

| 層級 | 內容 |
|---|---|
| 第一層 | `upstream`：`EstMppResult` 或 `RetrievalOutput` |
| 第二層 | `build` 綁定的 wsi、mask；`query`（影像 / `QueryPatchContainer`） |
| 第三層 | `upstream.find(kind)`：上游（或更上游）的副產物 |

**encoder 規則**：比對 trunk（model、transform、head）與實際載入權重的 `weights_id`；pooling 不在比對範圍，
由各 retriever 依自己的 config 在共用的 trunk 上套用。uni2 的 `cls` 與 `cls_avg` 共用一個載入的模型。
`TileEncoderConfig.pooling` 是欄位、預設 `''`（模型自己的，ViT 為 `cls`），並沒有強制設定。

**橋接鎖**

- 宣告是 config 的 `reuse: Tuple[str, ...]` 欄位，`NOT_IDENTITY`：有了鎖的保證，取用上游的特徵和自己算的
  結果相同，它只是省時間。
- 下游宣告了，上游有提供，但 key（trunk id、pooling、tile、overlap、level、mask）對不上 → `BridgeError`。
- 下游宣告了，上游沒有提供（上游名單是從 cache 讀回的，cache 不存記憶體裡的特徵）→ **下游自己負責**，
  只為它收到的名單做出需要的特徵（候選 tile 現場編碼、query 自己編碼、CSLS 自己抽樣參考 tile）。
  **不回頭叫上游補算**：這本來就是下游的任務，取用上游的副產物只是下游的捷徑。
- 下游沒宣告 → 不看上游的副產物。

---

## 三、stage 2 內部的串接

```
stage 1  EstMppResult
           │ upstream
           ▼
   ┌──────────────────────────┐   extras：level_features、query_features、window_scores（ranked 模式）、
   │ RetrieverSlidingWinSimRot │          tile_sims
   │ （或任一第一階段 retriever）│   candidates：raw_topk 個（不足就全部，n_listed 記錄）
   └────────────┬─────────────┘
                │ RetrievalOutput(parent=None)
                ▼
   ┌──────────────────────────┐   讀上游前 topk_in 個，重排
   │ RerankerDual             │   reuse=('level_features',)：上游有就經鎖取用（key 不合 → raise），
   └────────────┬─────────────┘                               上游沒有就自己做
                │ RetrievalOutput(parent=上一段)   extras：columns（以 Candidate.key() 對應）
                ▼
   ┌──────────────────────────┐   Stack 要讀 Dual、CSLS 的分數欄：find('columns') 沿 parent 往上找
   │ RerankerStack            │
   └────────────┬─────────────┘
                ▼
          CandidateSet → stage 3（讀前 topk）
```

### `MultiPhases.py`（由 `TwoPhase.py` 改名）

```
MultiPhases.py
├── MultiPhasesConfig(RetrieverConfig)   phases: Tuple[RetrieverConfig, ...]
│     __post_init__：第 0 段的 UPSTREAM 必須收 EstMppResult，其餘必須收 RetrievalOutput；
│                   每段的 topk_in ≤ 上一段的 raw_topk
├── MultiPhases(Retriever)
│     __init__：一個 EncoderPool 給所有段
│     run(upstream, query, cached=None)：依序 retrieve；cached(i) 是 cache 查詢的 hook，
│           命中就直接用讀回的名單（沒有副產物，下游照第二節的規則自己負責）
└── MULTIPHASES_RECIPES
```

arm `A::B::C` ≡ `MultiPhasesConfig(phases=(A, B, C))`。

### `Context.py`：共用工具箱（不是基類）

```
Context.py
├── 既有：LATTICES、OnDemandTiles、MemorySource、WindowBatch、window_boxes、iou_matrix、separated、
│         top_candidates（預選改用 raw_topk）、offset_pair_cosines、rotated_query_tiles
├── 由 SlidingWinSimRot 搬出（第一階段 retriever 共用）：
│   ├── level_geometry(wsi, mask, level, tile_size, overlap) -> (regions, grids)
│   ├── fitting_rotations(grids, dims)
│   ├── route_levels(routed, n_levels, fits, search) -> CandidateSet    退到更細 level 的邏輯
│   ├── LevelFeatures   一個 level 的 region grids（encoder view、tile、overlap；可接 feature_store）
│   └── QueryFeatures   query 各旋轉的特徵
├── frame_at(wsi, mask, level, tile_size, overlap) -> CandidateSet      讀回 cache 用，不需要 retriever
├── nearest_window(frame, dims, x, y, rotation)                         純幾何，truth 用
└── 移出：EncoderPool（到 Retriever.py）、LiveSource（由 extras 取代）
```

---

## 四、各檔案的架構

### 第一階段 retriever（各自是 `Retriever` 子類，`UPSTREAM = (EstMppResult,)`，自己編碼）

```
RetrieverSlidingWinSimRot.py
├── _score_mean / _score_geomean / _score_min、WINDOW_SCORES、window_score、SlidingWindowSimilarity、
│   _sim_tensors*（搬過來，內容不變）
├── RetrieverSlidingWinSimRotConfig   encoder, tile_size, overlap, raw_topk, min_sep_tiles, score
├── RetrieverSlidingWinSimRot(Retriever)
│   ├── run：route_levels → LevelFeatures / QueryFeatures → 分數圖 → top_candidates
│   ├── extras：level_features、query_features、window_scores、tile_sims
│   ├── tables：tile_sims
│   └── window_tile_sims(c)            truth_sim 用
└── SLIDEWIN_RECIPES

RetrieverBagOfTiles.py / RetrieverTileVote.py / RetrieverClusterBow.py / RetrieverWinEmbed.py
├── 各自的 Config：encoder, tile_size, overlap, raw_topk, min_sep_tiles, 自己的參數
├── Retriever 子類：run = route_levels → LevelFeatures → 自己的 window_maps → top_candidates
│                  extras：level_features、window_scores
└── 各自的 RECIPES
```

### reranker（各自是 `Retriever` 子類，`UPSTREAM = (RetrievalOutput,)`；檔案照現在分，只改名）

```
RerankerScorers.py   RerankerDual / RerankerCsls / RerankerCorr / RerankerCandGraph
RerankerEncode.py    RerankerShift / RerankerTokenMatch / RerankerXEnc
RerankerNeural.py    RerankerTileGT / RerankerCorrNet / RerankerStack
                     NeuralInputs、neural_inputs、pair_features、stack_features（training 用，保留）
├── 每個 Config：encoder（含 pooling，明確寫出，不再有 '' = 上游的）、topk_in、方法參數、reuse（NOT_IDENTITY）
├── run：取上游前 topk_in 個 → 特徵（經鎖取上游的，或自己做）→ 算分數欄 → 依自己的欄排序
│        → RerankedCandidate（first_score / first_rank 沿用）
├── extras：columns
└── tables：rerank_scores
```

`OutputPolicyConfig`（`max_rot_per_place`）刪除，不移植：repo 內每個使用處都設 0（全部保留，等於沒作用）。
要做「限制同一地點的旋轉數」的實驗時，寫成一個 reranker 接在鏈上。

### 其他

```
StageInterface.py    Candidate 系列、CandidateSet：不變
                     移除 Retriever / Reranker Protocol、RetrievalContext、TileFeatureSource、RerankNeeds
                     （由 Retriever.py 取代）
Naming.py            stage_name 通用化：由 config 註冊的 slug 與 operation 欄位產生，slidewin 不再特判
ArmSpec.py           config_of_arm(spec, ...) -> MultiPhasesConfig（由 bench 的 RetrieverArms 搬來）；
                     各方法 Config 提供 from_segment(seg, ...)
__init__.py          METHODS：slidewin、bagoftiles、tilevote、clusterbow、winembed、各 reranker、multiphases；
                     recipe('arm:<arm 文字>') 經 ArmSpec.config_of_arm
```

arm 文字轉 config 放在 stage 2 套件：pipeline 只經由各 stage 套件的 `recipe()` 把文字轉成 config
（stage 1 的 `knn:gigapath` 也是 stage1_estimation 自己解析），不需要知道 stage 2 的文法；bench 呼叫同一個
函式，所以兩邊的 config 由同一份程式產生，id 一定相同。

---

## 五、和其他 stage、pipeline、bench 的串接

### stage 3

吃 `RetrievalOutput.candidates`（`CandidateSet`）的前 `topk`，不變。

### pipeline（`utilities/LocaScopePipeline.py`）

- 移除 `rerankers` 參數；stage 2 是一個 Retriever 或一個 MultiPhases（`--stage2 <method>:<recipe>` 或
  `--stage2 arm:<arm 文字>`）。
- `stage2()` 回傳 `RetrievalOutput`；`LocaScopeQueryResult.stage2` 仍是 `CandidateSet`，另加 `stage2_output`。
- `utilities/cli/driver/locate_photo.py`：副產物表由 `retriever.tables()` 產生。

### pipeline bench（`utilities/bench_modules/BenchCommon.py`、`bench_locascope.py`）

- cache：stage 2 的每一段各一個 entry，`stage1=<s1|oracle>/stage2/<該段 entry 名>`；每段 id =
  `stage_hexid(段 config, (mask, 上一段 id))`，名稱帶 `_after-<上一段 hash>`；stage 3 在
  `stage2=<最後一段 entry 名>/stage3/`。
- `run_stage2`：每段寫 `output`、`retriever.tables()`、truth 表。
- 讀回：`frame_from_rows` 改用 `Context.frame_at`，不需要 retriever 物件。
- `Stage`：stage 2 持有多段；`stage_hexid` 照舊（權重檔內容進 id）。
- flag：`--stage2-k` 改為 `--stage2-raw-topk`（由 config 欄位自動產生）。

### stage 2 bench（`bench_stage2_retriever.py`、`RetrieverArms.py`、`ArmTables.py`）

```
arm 文字 ──ArmSpec.config_of_arm──▶ MultiPhasesConfig      （與 pipeline 同一個函式 → 同一個 id）
   │
   ├── SlidingWindow arm：串流快速路徑（保留）
   │     一次掃 level，所有 pooling × score，特徵不留
   │     gate：結果必須等於 RetrieverSlidingWinSimRot（更新 gate_window_stream / gate_scored_windows）
   │     寫自己的 entry（id = 對應 RetrieverSlidingWinSimRotConfig 的 stage_hexid）
   │
   └── 其他 arm：ArmEngine = 共用前綴的 phase 樹
         每個節點 = 一個 Retriever 物件，共用一個 EncoderPool
         level store（bench 專用）：以 key（trunk id、pooling、tile、overlap、level、mask）嚴格比對，
           經由 retriever 的正式 feature_store 接口取用；一個 level 編碼一次，所有 arm 做完，換 level 就清
         下游需要的 sliding window 節點：在 engine 內即時算，不重寫串流已寫的 entry，只比對兩份名單是否相同
         ranked 模式：讀第一階段節點的 extras['window_scores']
         FeatureHosts、host._encoder 直接塞 encoder 的做法：刪除
```

bench 為了比較所有方法、快速看到結果，做法可以和 pipeline 不同，但算法結果必須相同；每條快速路徑都有
gate 證明這點。pipeline 不開 level store / feature store。

### training（`training/RerankHead/*`）

改 import 路徑與類別名，行為不變。`cli/fit_stack.py` 改為跑 MultiPhases 鏈，分數欄由各 reranker 段的
`columns` 副產物（沿 parent）收集。

---

## 六、產物檔案與誰負責

### cache entry（`result/cache/<job>/.../stage1=<s1|oracle>/stage2/`，檔名 `<kind>_<entry 名>.<ext>`）

| kind | 內容 | 現在誰寫 | 誰讀 | 重構後 |
|---|---|---|---|---|
| `record` (.json) | config、各 VERSION、上游、環境、權重 fingerprint、`limit` | `BenchCommon.Stage.record`、`ArmTables.arm_entry` | `Cache`（是否命中、是否完整） | 每段一份；記 `raw_topk` |
| `output` (.csv) | 每個 FoV 的候選名單：rank、region、lattice、row、col、rotation、score、x0/y0/w0/h0、候選的額外欄、level、ds、irretrievable_lvl、alter_lvl、error、t_s | `BenchCommon.run_stage2`（pipeline bench）、`stage2_tables`（bench 串流）、`ArmTables.add_level`（bench engine） | stage 3（`stage2_for_stage3`）、分析、bench 報表、demo | 每段一份；每列加 `n_listed` |
| `tile_sims` (.csv) | 每個候選、每個 query tile 對其下 WSI tile 的餘弦 | sliding window 的 `tile_sim_rows` | `plot_locascope.py`、`locate_photo.py` | `RetrieverSlidingWinSimRot.tables()` |
| `truth` (.csv) | 每個 FoV 的兩個答案（只有合成 FoV、只有第一段有）：`truth_main_*` 與 `truth_offset_*` 各是 region / row / col、level-0 框、到 query 格中心的距離、分數；名次 `truth_rank_main`（M 在 main）、`truth_rank_main_in_all`、`truth_rank_offset_in_offset`、`truth_rank_offset_in_all`；`pool_main`、`pool_all`；`truth_rotation`、`truth_level`。搜到別層時只有位置、框、距離 | `BenchCommon.truth_rows`、`stage2_tables`（bench 串流） | bench 報表、分析、plot（都從第一段讀） | `TRUTH_VERSION` 4；reranker 段不寫（它只對名單打分，答案就是第一段的） |
| `truth_sim` (.csv) | 兩個答案的逐 tile 餘弦，`lattice` 欄分 main / offset | `BenchCommon.truth_rows`（需要 sliding window 的 sim maps） | `plot_locascope.py` | 只有提供 `tile_sims` 的 retriever |
| `first_phase` (.csv) | 第一階段提出的名單 | `TwoPhase.phase_rows` | 分析 | 刪除：第一階段本身就是一個 entry 的 `output` |
| `rerank_scores` (.csv) | 每個 reranker 收到的候選（依讀入順序）：視窗 key（region / lattice / row / col / rotation）與它自己寫的欄。進出名次、分數、框、`in` / `first` / `first_rank` 由上游與本段的 `output` 依 key join 得到 | `Context.score_rows`（`ColumnRanker.tables()`） | 分析 | 每個 reranker 段的 `tables()` 寫自己的欄 |

一個 entry 只寫它那一段有的表（`BenchCommon.phase_roles`：`output` + retriever 的 `TABLES` + 第一段有真值時的 `truth`、`truth_sim`），命中也只要求這些表。各欄存不存的準則見 `ARCHITECTURE.md`「entry 存什麼」。

### bench 結果（`result/Stage2RetrieverBench/<encoder>/`）

| 檔案 | 內容 | 誰寫 / 誰讀 |
|---|---|---|
| `stage2_retriever.csv` | 每個 FoV、每個 arm 的評分列（rank_main、pool、mode = ranked（NMS off）/ listed（NMS on）等；舊 CSV 的 whole / production 讀成這兩個） | 每張 slide 從 cache 的 entry 讀出（`slide_rows`）：ranked 列就是第一段 `truth` 的名次，listed 列是答案在最後一段 `output` 裡的位置（`BenchCommon.listed_places`），以第一段的答案為準；`--assemble` 不跑模型、只讀 cache 重組；`--report-only` 讀它出報表 |

bench 不另存評分列：每個 arm 只留它的 stage 2 entry，續跑看 entry 是否涵蓋每個有 FoV 的 level（從 render 表放置 FoV，不必 render 照片）。答案是哪格、排第幾、名單裡的位置，只有 `BenchCommon` 一份定義（`answers`、`ranked_ranks`、`listed_places`），bench、pipeline 的 truth、analyze、plot 都用它。

---

## 七、施工順序（working list）

施工期間舊檔全部保留，測試可以同時 import 新舊兩版、直接比對結果相同；驗證通過後舊檔交給使用者刪。
不新增 slurm script；測試併入既有的測試檔。測試由 `jobscripts/RerankHead/TestRerank.sh`、
`jobscripts/TestConfig.sh` 送 Slurm。

| 步驟 | 檔案 | 動作 | 驗證 |
|---|---|---|---|
| 1 | `stage2_retrieval/Retriever.py` | 新增 | `test_context.py` 擴充：encoder 比對會 raise、橋接鎖（key 不合 raise；上游沒有時下游自己做）、`topk > raw_topk` 會 raise |
| 1 | `stage2_retrieval/Context.py` | 搬入 level / query 共用工具、`frame_at`、`nearest_window`；移出 EncoderPool、LiveSource | 同上 |
| 2 | `stage2_retrieval/RetrieverSlidingWinSimRot.py` | 新增（移植） | 與舊 `SlidingWinSimRot` 在合成資料上名單、分數完全相同 |
| 3 | `RetrieverBagOfTiles.py`、`RetrieverTileVote.py`、`RetrieverClusterBow.py`、`RetrieverWinEmbed.py` | 新增（移植） | 與舊 CoRetriever 路徑結果相同 |
| 4 | `RerankerScorers.py`、`RerankerEncode.py`、`RerankerNeural.py` | 新增（移植） | 與舊 `ScoredReranker` 單 scorer 結果相同（`test_rerank.py`、`test_neural.py` 改寫） |
| 5 | `MultiPhases.py`、`__init__.py`、`Naming.py`、`ArmSpec.py` | 新增 / 修改 | `test_naming.py`、`test_arm_spec.py` |
| 5 | `StageInterface.py` | 移除被取代的 Protocol | 同上 |
| 6 | `utilities/LocaScopePipeline.py`、`utilities/bench_modules/BenchCommon.py`、`utilities/cli/driver/locate_photo.py`、`utilities/cli/plot/plot_locascope.py` | 修改 | `test_locascope_stages.py` |
| 7 | `utilities/bench_modules/RetrieverArms.py`、`ArmTables.py`、`bench_stage2_retriever.py` | 改寫 / 修改 | `test_retriever_arms.py`、`test_arm_tables.py`、bench 內建 gate |
| 8 | `training/RerankHead/{Steps,Visits,Embed}.py`、`cli/train.py`、`cli/fit_stack.py`、`utilities/bench_modules/bench_offgrid_score.py` | 改 import / fit_stack 改走 MultiPhases | `test_config_identity.py`（模組清單、recipe lint） |
| 8 | `realtest.sh`、`jobscripts/BenchLocaScope.sh`、`jobscripts/PlotLocaScope.sh` | `--stage2-k` → `--stage2-raw-topk` | — |
| 9 | Smoke | `-J SmokeStage2RetrieverBench`，`raw_topk = 1000` | 依 arm 文字與重構前的 cache 逐一比對 `output` 表 |
| 10 | `stage2_retrieval/README.md`、`spec.md`、本文件、`ARCHITECTURE.md` | 更新文件 | — |
| 11 | 交給使用者刪 | `SlidingWinSimRot.py`、`BagOfTiles.py`、`TileVote.py`、`ClusterBow.py`、`WinEmbed.py`、`CoRetrieve.py`、`CoStage.py`、`Union.py`、`TwoPhase.py`、`Rerank.py`、`RerankScorers.py`、`RerankEncode.py`、`RerankNeural.py`、`dataflow.md` | — |

### 驗證時要注意

- 下游自己做特徵（候選 tile 現場編碼）與取用上游整層編碼的特徵，讀取路徑與 batch 組成不同，fp16 下最後幾位
  可能有微小差異，同分時排序可能不同。比對時分開回報「完全相同」與「差在容許誤差內」。
- stage 2 的 id 全部會變；重構前的 cache 不刪，只用來依 arm 文字比對結果。

---

## 八、施工紀錄（與藍圖的差異）

施工到第九步之前完成；以下是實作時定下、藍圖沒寫到或改了的地方。

- **下游自己做特徵的方式**：reranker 沒有可取用的上游特徵時（例如上游名單從 cache 讀回），用和第一階段
  同一個 `encode_level` 編碼名單所在的整個 level，而不是只編候選 tile。這樣「上游現算」與「從 cache 讀回」
  兩種情況的數字逐位元相同；只編候選 tile 的讀取路徑不同，fp16 下最後幾位可能不同。
- **reranker 的共用做法** `RerankerScorers.ColumnRanker`：讀前 `topk_in`、建 incoming table、取特徵、
  寫欄位、依欄位排序、欄位當 `columns` 副產物往下傳。它是 `Retriever` 子類，不是另一種介面；
  `RerankerEncode`、`RerankerNeural` 的方法也繼承它。
- **sibling 共用一次打分**（bench）：只差排序欄位的 reranker（例如 Corr 的四個欄位）打分一次，其他欄位用
  `Context.reordered` 重排；結果與各自打分相同（`test_rerank`）。
- **bench 的 level store**：`RetrieverArms.LevelStore`，key 是 `Context.feature_key`，只留目前的 level；
  經由 retriever 的 `feature_store`（`shared_level`）取用，所以同一份特徵的所有 arm 共用一份。
- **entry 的寫入只有一處**：`BenchCommon.write_phase`，pipeline 與 bench 都用它；記錄由
  `BenchCommon.phase_record` 產生；名稱與 hash 由 `BenchCommon.phase_ids` 產生。
- **`tile_sims` 只寫前 10 個**（`TILE_SIM_ROWS`）：名單 1000 個時全寫是每個 FoV 兩萬列。
- **`Stage` 的 stage 2 一律是 chain**：單一 retriever 是一段的 `MultiPhases`（`BenchCommon.as_chain`），
  所以每個 stage 2 都按段建 entry；`stage_entries()[2]` 是最後一段（stage 3 讀的名單）。
- **class 是 identity 的一部分**：`RetrieverConfig.identity_parts` 前面加上 `retriever=<class>`。原本 id
  只記與自己 BASELINE 不同的欄位，各自在預設 op 的 BagOfTiles+gap、TileVote+exact、ClusterBow、Dual+dual
  都只差 `encoder`，因此 id 相同；bench 把它們當成同一個物件執行，接在它們後面的 reranker 也會共用同一個
  entry 與 `_after-`（smoke 572081 發現；`test_retriever_arms` 的 q 項）。
- **驗證（smoke 572456 對舊 uni2 run，BRACS_310 + S1128171）**：SlidingWindow 15 個、BagOfTiles、TileVote(exact)、
  CSLS、各 reranker 的第一個欄位逐列相同。其餘 reranker 欄位不同，是因為舊 run 的每個方法只建一個 reranker，所有欄位
  都用第一個排序（舊 CSV 中 Corr 四欄、CandGraph 三欄、Dual 兩欄的 rank 逐列相同）；新的是對的。ClusterBow 有
  少數列名次小動（cls_avg 2/518、rings3 約 14/518）：k-means 的 `index_add_` 在 CUDA 上是 atomic add，每次跑
  都可能不同；使用者接受，不改成決定性。
- **刪除**（交使用者刪檔）：`Union.py`、`OutputPolicyConfig`、`UnionCandidate`、舊的
  `Retriever` / `Reranker` Protocol、`RetrievalContext`、`RerankNeeds`、`LiveSource`、舊的 `EncoderPool`。

---

## 附錄 A：各方法的資料流（原 `dataflow.md`）

算法在這次重構中不變。其中「`ctx = RetrievalContext`」「bench 把 host 的 ctx 交給 reranker」的接法由本文件
第二、三節取代：特徵由各 retriever 自己做，或經橋接鎖取用上游的副產物。

符號：`R, C` 一個區域的 tile 格點列、欄數；`rq, cq` query 的 tile 格點（約 4×5，轉 90 / 270 度後是 5×4）；
`n = rq·cq`（約 20）；`F` 特徵維度（`cls` 1536、`cls_avg` / `cls_std` 3072、`rings3` 4608、`grid2x2` 7680，UNI2 與 GigaPath 的基底都是 1536）；
`H = R − rq + 1`、`W = C − cq + 1`（視窗的位置數）。一個「視窗」是 query 的 `rq×cq` 個 tile 排在 WSI 格點上的某個位置。

### A.0 共同的前端（每個方法都從這裡開始）

```
WSI ──切 256 px tile──► 兩組格點（main：[R, C]；offset：往右下偏半個 tile，[R−1, C−1]）
      encoder + pooling          │
                                 ▼
            wsi_grids[region] = (main [R, C, F], offset [R−1, C−1, F])    一層編碼一次，整張 slide 保留

FoV 照片 ─轉 0 / 90 / 180 / 270 度─► 每個方向切成整數個 tile ─ encoder + pooling ─►
            query_grid[rot]      = [rq, cq, F]        main 格點（打分用）
            query_offset[rot]    = [rq−1, cq−1, F]    offset 格點（Dual 用）

                    ↓ 兩者合起來，retriever 交給 reranker 的把手
              ctx = RetrievalContext(層、ds、grids、source、reader、encoders)
                    source：reference_grid / query_grid / encode_tiles（現場編碼任意 tile）
```

每個 retriever 的輸出都是同一種東西：**對每一個視窗位置一個分數**（分數圖），再取 top-K 並做 NMS：

```
分數圖 maps[(旋轉, 區域, 格點)] = [H, W]
   │ ranked 模式（NMS off）：整張圖排名（沒有 K、沒有 NMS）
   ▼
每張圖取 top-K → 全部依分數排序 → 同旋轉內太近（< min_sep_tiles）的只留最好的 → 候選名單（K 個視窗）   ← listed 模式（NMS on）
```

### A.1 第一階段（retriever）

#### SlidingWindow（`SlidingWinSimRot`）

```
query_grid[rot]  [rq, cq, F]  ──當 kernel──┐
wsi_grids[region].main / .offset [R, C, F] ─┴─► 每個視窗位置，query tile (i, j) 對 WSI tile (r+i, c+j) 的餘弦
                                                 sim = [H, W, rq, cq]          （每個 (旋轉, 區域, 格點) 一張）
sim ──window_score 合成最後兩個維度──► [H, W]      +mean 平均｜+geomean 幾何平均（下限 1e-3）｜+min 最小值
    ──► 分數圖 ──► top-K + NMS ──► 候選名單
```
- 沒有訓練的部分。一個 FoV 的成本是 4 個旋轉 × 區域 × 2 個格點的一次批次矩陣運算。

#### BagOfTiles（`+gap` / `+quad`）

```
wsi_grids[region].main / .offset [R, C, F]
   │ 視窗平均向量 = avg_pool2d(grid, (rq, cq), stride 1)       每次切一段 channel，不複製整層
   ▼
+gap :  cos( 視窗的平均向量 [F] , query 的平均向量 [F] )                  → [H, W]
+quad:  視窗切成四象限（rq//2, cq//2），每個象限各取平均向量，各自和 query 對應象限算餘弦，
        四個取平均（保留版面，所以轉過方向的不會配）                    → [H, W]
   ▼
分數圖 → top-K + NMS → 候選名單
```
- 沒有訓練。與 SlidingWindow 的差別：SlidingWindow 逐格對齊，BagOfTiles 先把視窗平均掉，所以真值在兩個格點之間時分數掉得比較少。

#### TileVote（`+exact` / `+hnsw` / `+ivf`）

```
整層所有 tile（所有區域、兩個格點）攤平 → TileIndex（以內積找最近鄰）
     exact：GPU 矩陣乘；hnsw / ivf：faiss（CPU）
query_grid[rot] 攤平 [n, F]
   │ 每個 query tile 找整層最像的 m = 64 個 tile → (相似度, 哪個格點, 平面索引) 各 [n, m]
   ▼
權重 = softmax(相似度 / tau)                                          [n, m]
一個命中在格點的 (row, col)，query tile (i, j) → 它投票給視窗起點 (row − i, col − j)
票數累加成每個格點一張 [H, W] 的票數圖，除以 n，再用 (2·smooth+1) 的框平滑一下
   ▼
分數圖 → top-K + NMS（候選帶 `votes` 欄位）→ 候選名單
```
- 沒有訓練。它是 SlidingWindow 的「反過來看」：不是問每個視窗和 query 像不像，而是問每個 query tile 的匹配在哪裡、讓它們為對應的視窗投票。

#### ClusterBow

```
【每層做一次】整層的 tile 抽樣 ≤ 200,000 個
   → spherical k-means：k1 = 32 個大群，每個大群再分（≤ 64 個，且 ≤ 成員數 / 32）→ 葉子 centroids [n_leaf, F]
   → 每個 tile 指派到最近的葉子 → id 格點 [R, C]（每個區域、格點各一張）
   → 一元詞：每個葉子的出現次數 → idf；二元詞：相鄰（左右、上下）tile 的 id 對，雜湊到 4096 個桶 → idf
query_grid[rot] ──指派到葉子──► query 的 id 格點 → 一元直方圖 q_uni，二元直方圖 q_h、q_v
視窗直方圖 = one-hot 的框和（avg_pool2d × 視窗面積）
   ▼
分數 = (1 − w)·一元交集 + w·½·(q_h 交集 + q_v 交集)         w = bigram_weight = 0.5
        交集 = Σ min(視窗計數, query 計數) × idf ／ query 自己的 idf 加權總數
   ▼
分數圖 → top-K + NMS → 候選名單
```
- 沒有學習式的訓練，但每層要現場做一次分群（成本在這裡）。它不做 tile 對 tile 的餘弦，所以能補「每個 tile 都找不到相同的 tile、但組織類型的組成一樣」的情況。

#### WinEmbed（要訓練）

```
【每層做一次】wsi_grids[region].main / .offset [R, C, F]
   │ WinEmbedder.tiles：每個 tile 過 Linear(F→512) + LayerNorm → [1, 512, R, C]
   │ 每種視窗形狀（4×5、5×4）一個 depthwise 卷積核 [512, a, b]（視窗內每個位置、每個 channel 一個權重）
   │ → [H, W, 512] → LayerNorm + MLP → L2 正規化                  整層每個視窗一個單位向量
query_grid[rot] [a, b, F] ──同一個函式──► [512]                   query 一個單位向量
   ▼
分數 = 視窗向量 · query 向量（一次矩陣乘向量）→ [H, W]
   ▼
分數圖 → top-K + NMS → 候選名單
```
- 設計上刻意**不看視窗邊緣以外**（卷積核就是視窗），所以「slide 上的一個視窗」與「排成同樣版面的 query」是同一個函式作用在同樣的 tile 上。

### A.2 第二階段（reranker）：共同的輸入

```
候選名單（K 個視窗）+ query 照片 + ctx
   │ WindowBatch：依視窗形狀分組，每個候選取
   │     q = query 轉到該視窗旋轉後的 tile  [G, rq, cq, F]
   │     w = 該視窗位置下 WSI 的 tile        [G, rq, cq, F]        （透過 ctx.source.reference_grid 一次 gather）
   │     cos = Σ_F q·w  → [G, rq, cq]        （就是 SlidingWindow 的逐 tile 餘弦）
   ▼
各 scorer 寫出「[K] 的欄位」到 ScoreTable → 依選定的欄位排序 → 重排後的名單
```
reranker 要的 pooling 或 encoder 和 retriever 不同時，bench 把「那個 (encoder, pooling) 的 host 的 ctx」交給它，scorer 不需要知道。
（重構後：reranker 的 config 明確寫出自己的 encoder 與 pooling，特徵自己做或經橋接鎖取用上游的。）

#### Dual（`+dual` / `+cos_main`）
```
主格點：cos [G, rq, cq] → 總和 main_sum（n 對）
offset 格點：query 的 offset tile [rq−1, cq−1, F] 對上 WSI 另一組格點的同位置 tile → 總和 off_sum（(rq−1)(cq−1) 對）
dual = (main_sum + off_sum) / (n_main + n_off)     cos_main = main_sum / n_main
```
#### CSLS（`+csls` / `+rank`）
```
抽樣整層 n_ref = 20,000 個 tile（固定種子）
r_t  = 候選視窗下每個 WSI tile 對該樣本的 top-k 餘弦平均（扣掉自己）           「這個 tile 對什麼都像」
rho_i = 每個 query tile 對整層所有 tile 的 top-k 餘弦平均                         「這個 query tile 對什麼都像」
+csls : mean_i(2·cos_i − rho_i − r_i)
+rank : 1 − （整層裡比「候選自己那個 tile」更像 query tile i 的 tile 所佔比例），取平均
```
#### Corr（`+diag` / `+contrast` / `+mnn` / `+sinkhorn`）
```
q [G, n, F]、w [G, n, F]  →  C = q·wᵀ  [G, n, n]     C[i, j] = query tile i 對 視窗 tile j 的餘弦
+diag     : C 對角線平均（等於 retriever 的分數）
+contrast : 平均( C[i,i] − 同一列最大的其他 C[i,j] )
+mnn      : 同位置互為最近鄰的 tile 所佔比例
+sinkhorn : exp(C / tau) 做 20 次列、欄正規化 → 傳輸計畫落在對角線上的質量 / n
```
#### CandGraph（`raw`，`+nbr_support` / `+prominence` / `+rot_margin`）
```
節點 = 候選（不看特徵），中心用候選的框；兩個候選的中心距離 ≤ 2 個 tile 就連邊（不管旋轉）
輸入分數 = 它收到的名單的排序分數（`in` 欄）
nbr_support = 鄰居的平均分數         prominence = 自己 − nbr_support          n_nbr = 鄰居數
rot_margin  = 自己 − 同一地點（距離 < ½ tile）其他旋轉中最高的分數（沒有就減最低分）
```
#### Shift
```
每個候選：位置往旁邊偏移 stride_px(64) × {−1, 0, 1}² 去掉 (0, 0) → 8 個偏移視窗
偏移視窗的 tile → ctx.source.encode_tiles（讀玻片、現場編碼；同一個 tile 只編一次，LRU 快取）→ [G, 8, rq, cq, F]
與 query tile 做餘弦 → 每個視窗取 tile 平均 → 在「原位置 + 8 個偏移」中取最高
輸出 shift（最高分）、shift_gain、shift_dx、shift_dy（候選變成位置被修正的 ShiftedCandidate）
```
#### TokenMatch（`raw`）
```
query tile（轉到視窗旋轉）→ 模型的 patch token（去掉 prefix）→ 正規化 [n·P, D]
候選視窗的 tile → encode_tiles(raw=True) → 同樣處理 [n·P, D]
sim = Tq · Twᵀ [n·P, n·P]；互為最近鄰且 ≥ sim_min(0.5) 的配對，最多留 top_matches(1500)
每個 token 的位置 = 所在 tile 的角落 + patch 格的中心（level px）
RANSAC：512 組假設，每組由兩個配對解出「縮放（0.8 到 1.25）＋平移」，位置誤差 ≤ 24 px 的算內點 → 取最大內點數
tokinl = 內點數 / (n·P)（候選變成 TokenMatchCandidate）
```
#### TileGT（要訓練）
```
輸入（每個候選一份，NeuralInputs）：q、w [K, n, F]；pos [K, n, 2]（tile 在視窗內的 row、col）；
       scalars [K, n, 1]（第 i 對的餘弦）；corr [K, n, n]（同 Corr 的 C）；
       glob [K, 8] = [retriever 分數在名單內的 z 分數, log1p(名次)/7, 是不是 offset 格點, 旋轉 one-hot（4 個）, 平均 tile 餘弦的 z 分數]
每個節點 = 一對 (query tile i, 視窗 tile i)：
   投影 LN + Linear(F→256)，融合 [q, w, q·w, |q−w|] → Linear → 256
   + corr 的第 i 列與第 i 欄（它和視窗內「所有」tile 的比較：偏一格會表現成隔壁 tile 更配）
   + 自己的 scalar + row / col embedding；前面加一個摘要節點（帶 glob）
4 層 transformer，注意力加上「相對位置」的學習式偏置（max_side 8，所有視窗形狀共用）
摘要節點 → 四個輸出：logit（是不是這個地方）、rot（旋轉對不對）、loc（真值相對視窗中心的位移，單位 tile）、embed
`tilegt_ctx`：再讓同一個 query 的 K 個候選的 embed 互相注意（配對特徵 [K, K, 4]：偏移、同旋轉、是否在 2 tile 內）
```
#### CorrNet（要訓練）
```
C [B, n, n] → 影像 [B, n, rq, cq]（每個 query tile 一個 channel，空間是視窗的格點）
              + 「各 query tile 自己應該在哪」的單位矩陣 [B, n, rq, cq]
→ 三層 3×3 卷積（width 64）→ 平均池化加最大池化 → MLP → logit [B]
```
#### Stack（要訓練，最後才能訓練）
```
ScoreTable 的各欄（first、dual、csls、sinkhorn、contrast、mnn、nbr_support、prominence、rot_margin、tilegt、corrnet…）
→ 每一欄取「在這個 query 名單內的名次（0 到 1）」和「z 分數」→ [K, 2·欄數] → MLP(64) → 一個分數
```
（重構後：各欄由鏈上各 reranker 段的 `columns` 副產物沿 parent 收集。）

### A.3 訓練的資料流

不論訓練哪一個網路，資料都是**現做現用、不存磁碟**，encoder 凍結：

```
訓練 slide（bracs/train、ki67_pure）→ mask → 每個 level 抽 FoV 位置（--fov-seed）
   │ 一個 level 一個 visit：
   ▼
retriever 編碼這個 level（一個 visit 只做一次，約 2 到 3 分鐘）
   │ 取 --fov-per-visit 個 FoV：渲染照片 → retriever.retrieve → 候選名單（K = 500，來自 uni2-avg-k500）
   ▼
標籤（Labels.py）  正例：與真值框 IoU ≥ pos_iou 且旋轉對   負例：與 FoV 不相交，或蓋住真值但旋轉錯   其餘：忽略
   ▼
prepare_fov：把這個 FoV 訓練要用的張量複製出來（下一個 FoV 一來，retriever 裡的特徵就被覆蓋）
   ▼
訓練 --steps-per-visit 次，每次 --fovs-per-step 個 FoV；level 丟掉，換下一個
   ▼
驗證（val slide 的 FoV）：算「第一個命中在前 k 名」的比例，同時算 retriever 原本的順序 → 進步看得到
   ▼
存權重：_best（--select-k 上最好的 epoch）、_last
```

#### WinEmbed 的訓練（`Embed.py`）
```
一個 visit 的整層特徵格點 ─► model.tiles 一次（所有視窗形狀共用）
每個 FoV：query 向量 = model.query(query 的 tile)；整層每個視窗向量 = 卷積
logit = 視窗向量 · query 向量 / 溫度            對象是「整層所有視窗」，不需要候選名單
正例 = 與真值框 IoU ≥ pos_iou 且旋轉對的視窗；負例 = 與 FoV 框不相交的視窗；其餘重疊的不進損失
損失 = 整層 softmax（pool softmax）
驗證 = 用向量對整層排名，看第一個命中的名次
```

#### TileGT、CorrNet 的訓練（`Steps.py`、`Losses.py`）
```
候選名單（K = 500）→ choose_subset：留下「所有正例」加「最難的負例」，湊成 n_cand 個
                         （hard_share 從 0.2 逐步增加到 1.0：先學簡單的，再學難的）
→ NeuralInputs（q、w、pos、scalars、corr、glob；tilegt_ctx 另加配對特徵）
→ 網路輸出 {logit, rot, loc}
損失（一個 FoV 的名單一個損失）：
   listwise softmax（主要）+ top-k hinge（正例要贏過第 k′ 高的負例）+ BCE + rot BCE（旋轉）+ loc Huber（位置修正；只用 IoU ≥ 0.1 的）
優化：AdamW，梯度裁剪 --clip
驗證：K 個候選全部打分（分批），看第一個命中的名次，對比 retriever 的順序
```

#### Stack 的訓練（`fit_stack.py`）
```
先要有其他 scorer 的權重（TileGT、CorrNet…）
在 val slide（不是訓練 slide）上，用 bench 的方式跑 reranker，得到「每個候選一列、每個 scorer 一欄」的表
欄 → 名單內名次 + z 分數 → [K, 2·欄數]
損失 = listwise softmax + top-k hinge；val slide 的三分之一（依 slide 分）拿來挑 epoch
用 --split test 確認（訓練和擬合時完全不碰）
```
之所以在 val 擬合：TileGT、CorrNet 在訓練 slide 上的分數特別好，在那裡擬合的 stacker 會太信任它們。

### A.4 要記得的兩個限制

- reranker 的**訓練候選只來自 sliding window 的名單**（`uni2-avg-k500`，`cls_avg+mean`）。新 bench 允許把 TileGT 接在別的 retriever 的名單後面，那是訓練與使用的候選分布不同。
- `train.py` 的 pooling 由 `--retriever` 指定的 recipe 決定，目前只有 `cls`、`cls_avg`；`rings3`、`grid2x2` 的版本要先加 recipe。
