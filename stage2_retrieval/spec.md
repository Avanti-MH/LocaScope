# Stage 2 — 檢索

Stage 2 把一張 query 對到 WSI 上的候選視窗，分兩階段：

    第一階段  粗檢索   整張 slide → top-K 候選窗
    第二階段  重排序   query + top-K → top-K′ 候選窗      K′ ≤ K

檢索階段不產生更細的物理位置；位置細化是 stage 3 的事。

狀態（2026-10-19）：程式的**結構**照 `blueprint.md` 重構過：每個方法都是一個 `Retriever`
（`Retriever.py`），reranker 是吃上游名單的 retriever，`MultiPhases` 串接，pipeline 與 bench 同一個
config 就是同一個 cache entry。本文件以下講的是**方法本身**（演算法、損失、訓練、評估），仍然有效；
講介面的段落（`RetrievalContext`、`ScoredReranker`、`TwoPhase`、`OutputPolicy`、`UnionRetriever`）
是重構前的接法，已由 `blueprint.md` 第二到五節取代，留在下面只為了說明當初的設計理由。測試用
`sbatch jobscripts/RerankHead/TestRerank.sh`。標〔待驗證〕的是還沒用資料或程式確認的假設。

| 部分 | 在哪裡 | 測試 |
|---|---|---|
| `Retriever`、`EncoderPool`、橋接鎖 | `Retriever.py` | `test_context` |
| Candidate 家族、`CandidateSet` | `StageInterface.py` | `test_context` |
| level 路由、編碼、sources、`WindowBatch`、框與 IoU、`top_candidates`、reranker 的共用做法 | `Context.py` | `test_context`、`test_rerank` |
| 第一階段：sliding window、F1 到 F4 | `RetrieverSlidingWinSimRot.py`、`RetrieverBagOfTiles.py`、`RetrieverTileVote.py`、`RetrieverClusterBow.py`、`RetrieverWinEmbed.py` | `test_context`、`test_rerank`、`test_neural` |
| S0 Dual、S1 CSLS、S2 Corr、S6 CandGraph（`ColumnRanker`） | `RerankerScorers.py` | `test_rerank` |
| S3 Shift、S4 TokenMatch、`xenc` | `RerankerEncode.py` | Shift 有；TokenMatch、`xenc` 需要 slide 與 encoder，由 bench 驗證 |
| S5 TileGT、S2net、S7；網路與 checkpoint | `RerankerNeural.py`、`aiNNModel/models/RerankHeads.py`、`RerankChoices.py`、`common/Checkpoints.py` | `test_neural` |
| 串接與 recipe | `MultiPhases.py`、`ArmSpec.config_of_arm` | `test_arm_spec`、`test_config_identity` |
| 命名 | `Naming.py` | `test_naming` |
| 標籤、損失、指標、visit、訓練、S7 擬合 | `training/RerankHead/` | `test_neural` |
| 評估 | `utilities/bench_modules/RetrieverArms.py`、`ArmTables.py`、`bench_stage2_retriever.py` | `test_retriever_arms`、`test_arm_tables` |
| mask 瘦身 | `TissueMask.py`、`utilities/cli/build_cache/slim_masks.py` | `test_slim_masks` |
| 巢狀 config 的 tuple | `ConfigIdentity.py` | `test_config_identity` |

**還沒做**：`TileGT` 的窗口外圍 `margin` 版本；`raw`（不 pool 的 patch token）的 sliding window
與 reranker 讀法；`--cache ro|rw`（P1 到 P3 的 cache 模式）；訓練的 crash-resume；S4 沒有單元測試。
與前面設計的小差異：位置偏移的方法叫 `shift_px()`（level px）；`WinEmbedder` 沒有鄰居混合層（見 F3）。

## 詞彙

query、shot 沿用 `CLAUDE.md`；pool、rank、答案視窗、arm、pooling 見
`bench_stage2_retriever.py` 開頭的 VOCABULARY（原 window bench）。

| 中文 | English | 定義 |
|---|---|---|
| 粗檢索 | coarse retrieval | 第一階段：整張 slide → top-K 候選窗 |
| 重排序 | reranking | 第二階段：query + top-K → top-K′ 候選窗 |
| 第一階段候選數 K | K | 粗檢索輸出的候選數，設定值；**實驗軸**（100 / 200 / 300 / 500），不是固定 500 |
| 第二階段候選數 K′ | K′ | 重排序輸出的候選數，K′ ≤ K，設定值（30 / 100） |
| 圖塊 | tile | 256×256 px，encoder 的輸入單位 |
| 組織區域 | region | 組織 mask 的一個 bbox；格點以它的原點為準 |
| 視窗 | window | query 的 tile 格（約 20 個 tile，例如 5×4，旋轉 90/270 度時 4×5）落在參考格點上的位置。FoV 大小會變，**所有門檻都用視窗面積的比例，不用 tile 數** |
| 格點 | lattice | 參考 tile 的一組排列，有 main 與 offset 兩組 |
| 主格點 | main lattice | 從 region 原點起、間距 256 px |
| 偏移格點 | offset lattice | main 平移 (128, 128)；取代舊稱 overlap（程式與 CSV 欄位仍叫 overlap） |
| 候選 | candidate | `(region_index, lattice, row, col, rotation, score)`，子類別可加欄位 |
| 候選集 | candidate set | 依 score 由高到低排好的候選清單（位置就是名次），加上它的座標框架：level、ds、每個 region 的格點 |
| 主格點召回率 | main recall, `recall_main@k` | 最近的 main 視窗，在只有 main 視窗的池裡排名 ≤ k 的比例 |
| 偏移格點召回率 | offset recall, `recall_offset@k` | 最近的 offset 視窗，在只有 offset 視窗的池裡排名 ≤ k 的比例 |
| 混合召回率 | combined recall, `recall_all@k` | main + offset 混合池裡，上面兩個視窗較前者的排名 ≤ k 的比例 |
| 命中召回率 | recall, `R@k` | 訓練與標籤用，報表不用：候選窗與真值框的 IoU ≥ `hit_iou`（預設 0.5）且旋轉正確（Stage 3 要求旋轉正確），名次 ≤ k 的 FoV 比例 |
| 位置召回率 | location recall, `R_loc@k` | 同上但旋轉不限；只當診斷 |
| FoV 框 | FoV box | 照片在 level-0 的外接軸對齊框；比 query 的 tile 格框大（照片 1440×1024，tile 格 1280×1024） |
| 原始 token | raw token | encoder 的原始輸出，每個 tile 265 個（UNI2：9 個 prefix + 16×16 個 patch） |
| 圖方法 | graph method | 以 tile 為節點、關係為邊的檢索方式 |

## 起點資料（已量測）

**uni2 的 pipeline**（`result/AnalyzeLocaScopeUni2`；只算 stage 2 搜了真值所在層的 FoV；stage 2 回 100 個候選）

| | hit@1 | hit@10 | hit@100 | stage 2 漏掉（占全部 FoV） |
|---|---|---|---|---|
| bracs（n=644） | 91.3% | 95.0% | 98.3% | 0.8% |
| ki67（n=769） | 70.2% | 81.3% | 91.3% | 3.4%（ds=1 為 10.6%） |

**window bench**（uni2、`cls_avg+mean`，`recall_all@k`，每個資料集的所有 FoV）

| | @1 | @10 | @30 | @100 | @500 | @1000 |
|---|---|---|---|---|---|---|
| bracs 全部 | 95.6 | 98.9 | 99.2 | 99.3 | 99.7 | 99.7 |
| ki67 全部 | 75.1 | 90.1 | 93.5 | 95.7 | 97.9 | 98.1 |
| ki67 ds=1 | 78.4 | 87.4 | 90.6 | 92.3 | 95.5 | 95.7 |

- ki67 ds=1 在 K=1000 仍有 4.3% 沒被提出：這一段只有**第一階段**能解，rerank 無能為力。
- ki67 的 @1 → @100 差 20 個點：這是**排序**的空間，rerank 能解。
- bench 的 rank 是原始排名；pipeline 的候選有 `min_sep_tiles` 去重，同樣的 K 實際走得更遠〔待驗證〕。
- gigapath（舊 stage 2，兩個資料集合計）：@100 只有 70.0%，@500 為 81.0%，空間大得多；但現在用 uni2。

## 目標與指標

**主要目標：提高 top-100 的 recall。** K 與 K′ 是兩個獨立的軸，部署時才可能用到 K=500。

| 設定 | 說明 |
|---|---|
| K = K′ = 100 | rerank 不改變召回，只改順序；能提高 R@100 的只有第一階段（更好的檢索器、聯集） |
| K ∈ {200, 300, 500}、K′ = 100 | rerank 把落在名次 100 以外的真值撈進 100 |
| K ∈ {100…500}、K′ = 30 | 同上，且 stage 3 只對 30 個候選做 SIFT（約省 70% 時間） |

實驗是 (K, K′) 的格子，不預設其中一格。

**指標：只用 stage 2 retriever bench 的那一套，不另外發明。**
所有第一階段方法（sliding window 與其他）與第二階段的 reranker 鏈是同一個 bench（`bench_stage2_retriever.py`）的 arm，
用同樣的 FoV、同樣的答案、同樣的表格（細節見該檔 VOCABULARY 與 `RetrieverArms.py`）：

| 指標 | 定義 |
|---|---|
| 答案 | 離真位置最近的 main 視窗與 offset 視窗，旋轉要對（`matching_rotation`） |
| `rank_truth`（記錄用） | 兩個答案中幾何上較近者的名次 |
| `top@f%`（跨片、跨層用） | `rank_truth ≤ k@f%`，`k@f% = max(1, ceil(f·pool))`，f = 0.01 / 0.1 / 1 / 10 %；隨機排序的基準固定是 f%，所以粗層（pool 只有幾百）與細層（幾十萬）可以放在一起 |
| `truth@k`、`gap@k`（單片單層、同層跨片） | 固定 k 的命中率；`gap@k` = 較近那個沒進、另一個進了 |
| `recall_main / offset / all @k` | 兩個答案各自、以及較前者在 k 內的比例；`random` 一行是同 pool 隨機排序的期望值 |
| 配對比較 | 對 `cls+mean` 逐 FoV 的 wins / losses / ties / decided / win%、ratio Q1 / med / Q3 |
| 聚合 | 單片單層、同層跨片（PRIMARY）、單片跨層、全部 |

**每個 arm 有兩種模式**，由同一組分數圖得到，所以兩者的差就是 top-K 與 NMS 的代價：

| 模式 | 名次的算法 |
|---|---|
| `whole` | 整層所有視窗（四個旋轉、兩個格點、所有區域）依 arm 的分數排名，沒有 top-K、沒有 NMS。`1 + 嚴格較高的視窗數`；`rank_main` 只比 main 視窗，offset 答案只比 offset 視窗，`_in_all` 比全部。看 pooling 與算分本身好不好 |
| `production` | stage 實際交給下游的：最好的 K 個視窗（`--k`，預設 1000），同旋轉內離更好的視窗不到 `min_sep_tiles` 的被丟掉。**答案的名次 = 名單裡第一個「就是它、或旋轉相同且離它不到一個 tile（NMS 的半徑）」的視窗的位置**：答案的視窗若被 NMS 擠掉，擠掉它的視窗就在這個半徑內，由它代表那個地點；範圍內都沒有才是 `pool + 1`（對任何 k、任何 f% 都算沒中）。半徑 0（`--min-sep-tiles 0`）時只認視窗本身；被 Shift 移動過的視窗以移動後的位置計。記錄用的是離 shot 較近的那個答案的名次，`recall_all` 用兩個答案中較好者。因此 production 的名次可以比同一個 arm 的 whole 還好（代理人排得比 GT 前面），兩個模式的定義不同；production 沒有 main / offset 之分，所以不印 `recall_main`、`recall_offset`、`gap@k`，改看 `recall_all@K` 曲線 |

報表每個模式一個區塊，各自對同模式的 `cls+mean` 配對；production 區塊後面有 `production lists` 表，列出「答案有沒有被列出來」（天花板）與名單長度。
reranker 鏈只重排給定的名單，名單外的視窗沒有分數，所以只有 production 模式。
sliding window 同時有兩種 arm：全排序（`cls_avg+mean`，pooling arm）與 top-K 名單（`slidewin:cls_avg+mean:k500`，含 NMS，是 stage 實際交給下游的）。

**IoU 命中只用在訓練標籤**（`training/RerankHead/Labels.py`：IoU ≥ `pos_iou` **且**旋轉對才是正例）。報表不用。

**旋轉是硬要求。** Stage 3 要求旋轉正確，所以答案視窗取的是 `matching_rotation` 那個旋轉的視窗，名次也只認這個旋轉；位置對、旋轉錯不算命中。
- 訓練時，旋轉錯的位置是負例（旋轉難負例），並有輔助的 `L_rot`；不需要 `rot_weight`，也不需要先量測 stage 3 對錯誤旋轉的容忍度。
- 輸出時見 `OutputPolicy`：不跨旋轉合併。

## 現況與目標

| | 現況 | 目標（未實作） |
|---|---|---|
| 第一階段 | `SlidingWinSimRot`（canonical，滑窗 cosine，試 4 個旋轉） | 同左，加 F1 到 F4 與 `UnionRetriever` |
| 第二階段 | 無 | S0 到 S7 與 `xenc`，鏈式 |
| 候選 | `Candidate` / `CandidateSet`（`StageInterface.py`） | 基礎類別不變，子類別帶方法專屬欄位 |

## 候選

視窗 = query 的 tile 格（約 20 個 tile，旋轉 90/270 度時長寬對調）落在參考格點上的位置。
參考有兩組格點：`main`（region 原點起，間距 256 px）與 `offset`（main 平移
(128, 128)）。

    Candidate    = (region_index, lattice, row, col, rotation, score)
    CandidateSet = (candidates, level, ds, grids)

- `lattice` 是 `'main'` 或 `'offset'`（取代 `from_overlap`）。
- 基礎 `Candidate` 只放 retrieval 自己找出來的東西。可從 input 推論的（像素座標、視窗大小、名次）不放。
- **方法專屬的資訊用子類別加欄位**，不塞進基礎類別，也不用 nan 當「沒有」的哨兵：

```python
@dataclass(frozen=True)
class Candidate:                       # 基礎，欄位與現在完全相同
    region_index: int
    lattice: str
    row: int
    col: int
    rotation: int
    score: float                       # 這個集合最後一個 phase 的分數；集合依它排序
    def key(self) -> Tuple[int, str, int, int, int]: ...      # 前五欄，不變
    def shift_px(self) -> Tuple[int, int]: ...                # 預設 (0, 0)，level px；精修的子類別覆寫
    def extra(self) -> Dict[str, Any]: ...                    # 預設 {}；子類別回傳要寫進表的欄位

@dataclass(frozen=True)
class RerankedCandidate(Candidate):    # 任何 reranker 的輸出
    first_score: float = 0.0           # 第一階段給這個窗口的分數
    first_rank: int = 0                # 它在第一階段名單的名次（1 起）

@dataclass(frozen=True)
class ShiftedCandidate(RerankedCandidate):     # S3 / S5 的次格點精修
    shift: Tuple[int, int] = (0, 0)    # (dx, dy)，level px，加在格點原點上

@dataclass(frozen=True)
class TokenMatchCandidate(RerankedCandidate):  # S4
    n_inliers: int = 0
    inlier_ratio: float = 0.0

@dataclass(frozen=True)
class VoteCandidate(Candidate):        # F2
    votes: float = 0.0

@dataclass(frozen=True)
class UnionCandidate(Candidate):       # UnionRetriever
    src_ranks: Tuple[Tuple[str, int], ...] = ()    # (來源名, 該來源的名次)
```

- 身分 = 前五個欄位（不含子類別欄位）。兩個方法的候選是不是同一個視窗，就是這五個欄位相等。
- `CandidateSet.origin_l0` 把 `c.shift_px()`（level px）乘上 `ds` 加進那條唯一的公式；`rows()` 把 `c.extra()` 併成額外欄位（不同子類別的欄位取聯集，缺的留空）。其餘派生量仍只從 `CandidateSet` 的方法來。
- 第一階段的集合裡是基礎 `Candidate`（或 `VoteCandidate` 等）；rerank 的輸出一律是 `RerankedCandidate` 或其子類別，`first_*` 永遠指 retriever（含 `UnionRetriever`）輸出的那一份，不是 rerank 鏈中間的。
- `CandidateSet` 是候選清單加上它的座標框架。`candidates` 依 score 由高到低，位置就是名次。`region_index`、`row`、`col` 只對一份 regions、一個層才有意義，所以那份 regions 的格點（`grids`，每個 region 一個 `PatchGrid`，帶 level-0 原點）、`level`、`ds` 跟著清單走，不讓呼叫端拿兩個變數自己配對——與 `WsiFeaturesMap` 存在的理由相同。推論的輸入跟著輸出走，推論的結果不存。
- 視窗大小不在 `CandidateSet` 裡。它是 query 的 tile 格，只由 query 決定；需要它的方法（`window_tiles`、`window_l0`、`centre_l0`）直接吃未旋轉的 `QueryPatchContainer`，並檢查它的 tile 大小與 `grids` 的相同。
- 位置只有一條公式：視窗左上角的 level-0 讀圖點 =
  `grids[c.region_index].tile_origin_l0(c.lattice, c.row, c.col)` 加上 `c.shift_px() * ds`。
  bench、視覺化、pipeline、stage 3 都用這一條。

## 介面（重構前；現在的介面見 `blueprint.md`）

> 下面到「OutputPolicy」為止是重構前的接法：第一階段交出 `RetrievalContext`、reranker 是另一種
> Protocol、`TwoPhase` 把它們包成一個 stage、`OutputPolicy` 裁名單。現在 reranker 就是吃上游名單的
> `Retriever`，特徵由各 retriever 自己做或經橋接鎖取用上游的，串接是 `MultiPhases`，`OutputPolicy`
> 與 `UnionRetriever` 已刪除。

每一階段吃前一階段的輸出，pipeline 只是串接：

    stage 1   build(wsi, mask)                              每張 WSI 一次
              estimate(query)               → EstMppResult
    stage 2   第一階段  build(wsi, mask)                    每張 WSI 一次
                        retrieve(query, EstMppResult) → CandidateSet   長度 K
                        context()             → RetrievalContext       （選用）
              第二階段  build(wsi, mask)                    每張 WSI 一次
                        rerank(query, CandidateSet, ctx) → CandidateSet   長度 K′
    stage 3   build(wsi)                                    每張 WSI 一次
              localize(query, CandidateSet, rank=0)  → 定位結果

- `retrieve` 在 `EstMppResult.chosen_level` 上檢索。層由 stage 1 選一次，stage 2 不再自己選；每一層的格點與特徵第一次用到時才建，之後快取。
- K、K′ 是設定，不寫死。`rerank` 輸入輸出同型，第二階段可以整個略過。每個方法自有 config 與 build 產物（快取以設定 id 為鍵），沿用 stage 1 的樹幹加分支。
- `build(wsi, mask)` 以 pipeline 目前實際呼叫的為準（原 Protocol 寫 `build(wsi, encoder)`）；encoder 改由 `ctx.encoders` 取，見下。

### RetrievalContext 與 TileFeatureSource

rerank 要拿第一階段已算好的特徵，也可以有自己的 encoder。兩條路用同一個介面：

```python
@dataclass
class RetrievalContext:
    level: int
    ds: float
    tile_size: int
    overlap: bool
    grids: Tuple[PatchGrid, ...]
    source: 'TileFeatureSource'    # 參考與 query 的 tile 特徵（retriever 保留的 pooling）
    reader: SlideReader            # 重新讀候選窗像素
    encoders: 'EncoderPool'        # get(cfg)：identity_id 相同就沿用 retriever 的 encoder，否則建立並快取

class TileFeatureSource(Protocol):
    def has(self, pooling: str) -> bool: ...
    def reference_grid(self, region: int, lattice: str, pooling: str) -> torch.Tensor: ...   # [R, C, F]
    def query_grid(self, rotation: int, pooling: str, lattice: str = 'main') -> torch.Tensor: ...   # [qr, qc, F]
    def encode_tiles(self, origins_l0: torch.Tensor, pooling: str, raw: bool = False) -> torch.Tensor: ...

class Reranker(Protocol):
    k_in: int
    k_out: int
    needs: 'RerankNeeds'           # 需要的 pooling / raw token / 自己的 encoder
    def build(self, wsi: SafeSlide, mask: Optional[TissueMask]) -> 'Reranker': ...
    def rerank(self, query: np.ndarray, cs: CandidateSet, ctx: RetrievalContext) -> CandidateSet: ...
```

| `TileFeatureSource` 實作 | 來源 | 用在 |
|---|---|---|
| `LiveSource` | `SlidingWinSimRot` 持有的 `wsi_grids` / `query_features_by_rot` | 推論、bench |
| `MemorySource` | 一個 level 的特徵放在記憶體（測試與離線評估用；訓練直接讀 retriever 的 `LiveSource`，見「訓練資料」） | 測試、離線評估 |
| `OnDemandSource` | 從 `reader` 讀候選 tile 像素、用 encoder 現場編碼；LRU 以 `(x0, y0, ds, pooling)` 為鍵 | S3、S4、`xenc` |

同一個 reranker 模組在三種 source 上跑同一份程式，訓練與推論不會分歧。`needs` 讓組裝時就檢查
retriever 是否保留了 reranker 要的 pooling，不要到跑了才發現。

### TwoPhase：有 cache 身分的 composite stage

`--stage2 twophase:<recipe>` 在 `stage2_retrieval/__init__.py` 的 `METHODS` 登記。它的 config
同時包含 retrievers、`k_first`、rerank 鏈與輸出政策，所以 `s2.id` 自然包含所有 phase；
`BenchCommon` 的 cache 位址與 `stage_entries` 不必改。

- 它轉發 `nearest_window`、`tile_sim_rows`、`sim_maps_by_rot`、`frame`，bench 的 truth 表照舊算「第一階段名次」。
- 它額外提供 `phase_rows()`：第一階段名單與各 scorer 的分數表，bench 寫成兩個選用的表（`first_phase`、`rerank_scores`）。
- pipeline 原有的 `rerankers=` 參數保留給臨時組合；composite 才是有 cache 身分的路徑。

### OutputPolicy

rerank 鏈的最後一步，決定 K′ 個候選怎麼選：
- **旋轉是候選身分的一部分**：Stage 3 要求旋轉正確，所以同一地點的不同旋轉是不同的假設（只有一個是對的），**不跨旋轉合併**。`min_sep` 去重仍只在同一旋轉內（現行行為）。
- 可選 `max_rot_per_place`（預設不限）：同一地點（窗口框 IoU ≥ 0.5）最多保留幾個旋轉假設，供實驗「旋轉假設是否吃掉 K′ 名額」時使用；預設不啟用。
- 取前 `k_out`，把 `first_score`、`first_rank`、`shift` 寫回。

## 方法統整：16 個名稱 → 11 個單元

兩份清單裡同一個演算法有兩個名字，統整如下（各自的變體保留為設定選項）：

| 原編號 | 統一單元 | 說明 |
|---|---|---|
| A1 | **F1 BagOfTiles** | 視窗均值描述子（積分圖） |
| A2 = G1 | **F2 TileVote** | tile kNN 圖 + 位置投票；索引後端 `exact` / `hnsw` / `ivf` |
| A3 = G3 | **F3 WinEmbed** | 學習式視窗 embedding（grid graph 聚合） |
| G2 | **F4 ClusterBoW** | 階層分群 + 群 ID 袋 / 雙連詞 |
| B1 = R4 | **S1 CSLS** | hubness 校正（tile kNN 圖的度數） |
| B2 + R1 | **S2 Corr** | n×n 對應矩陣：免訓練分數 + 可訓練 `CorrNet` |
| B3 | **S3 Shift** | 次格點偏移精修 |
| B4 | **S4 TokenMatch** | patch token 互為最近鄰 + 批次 RANSAC |
| B5 = R2 | **S5 TileGT** | tile graph transformer（可訓練） |
| R3 | **S6 CandGraph** | 候選圖上的特徵傳播 / 突出度 |
| （新） | **S7 Stack** | 把各分數融合的 listwise 堆疊器 |
| （新、近乎免費） | **S0 Dual** | 同時使用 query 的 offset 格點 tile |
| （新） | **xenc** | 用另一個 encoder 重算候選分數 |
| （共用） | `UnionRetriever` | 把多個第一階段合併成 K |

## 共用模組

寫一次，所有方法重用。

**WindowBatch**：候選 → 對齊好的 tile 張量，向量化 gather。
- 輸入：`CandidateSet`（K 個）+ `TileFeatureSource` + pooling + `margin`（0 或 1，窗口外圍多帶一圈 tile，帶遮罩）。
- 輸出：`q` `[K, qr, qc, F]`（旋轉後的 query tile）、`w` `[K, qr, qc, F]`（窗口下的參考 tile，依 `(region, lattice)` 分組 gather）、`cos` `[K, qr, qc]`（逐 tile cosine）、`lattice`、`rotation`、`first_score`、`first_rank`、窗口框。
- `cos` 的平均必須與第一階段 `window_score` 逐位元一致（單元測試）。
- 旋轉 90 / 270 的窗口形狀是 `(qc, qr)`，張量依形狀分兩組，不補 padding。FoV 大小改變時，形狀由 query 決定，不寫死。

**ScoreTable**：`Dict[str, Tensor[K]]`，每個 scorer 回寫一欄（`first`、`dual`、`csls`、`diag`、`contrast`、`mnn`、`sinkhorn`、`corrnet`、`tokinl`、`shift_gain`、`tilegt`、`nbr_support`、`prominence`、`rot_margin`、`src_rank_<來源>`）。S7 吃整張表；bench 把整張表輸出成 csv。

**Scorer 與 registry**：`Scorer.score(batch, ctx, table) → Tensor[K]` 加 `needs`。`ScoredReranker`（一個 stage 類別）依設定逐一執行 scorers，再接 S7 與 `OutputPolicy`。所有 reranker 方法都是 registry 裡的 scorer，不是各自一個 stage 類別。

**config 與 identity**：`IdentifiedConfig` + `@register`；有訓練權重的 scorer，config 帶 checkpoint 路徑，identity 用 `file_fingerprint`。`TwoPhase` 的 config 需要「巢狀 config 的 tuple」，見「基礎模組改動」。

## 第一階段方法（推高召回與天花板）

所有第一階段方法都輸出 `Candidate`（或其子類別），由 `UnionRetriever` 合併。

**F1 BagOfTiles（A1）**
- 對 `[R, C, F]` 特徵做 fp32 積分圖 `S`；窗口均值 `m_w = (S[r+qr,c+qc] − S[r,c+qc] − S[r+qr,c] + S[r,c]) / (qr·qc)`，分數 `cos(m_w, m_q)`。
- `gap` 模式與旋轉無關，只需 2 個形狀的 pass；`quad` 模式用 2×2 象限均值，保留排列，需要 4 個旋轉。
- 對天花板有幫助的理由：窗口均值對 ±半個 tile 的偏移不敏感，而逐 tile cosine 在偏離格點時衰減；ki67 ds=1 的 4.3% 缺口可能與此有關〔待驗證〕。
- `gap` 找到的位置，旋轉在 2 個相容旋轉中取 slidewin 精算分數較高者。
- 無訓練；成本是一次 cumsum 加矩陣乘法。

**F2 TileVote（A2 / G1）**
- 每個 query tile `(i, j)`（旋轉 θ）對參考 tile 找 top-m 個鄰居（`m=64` 起），每個鄰居 `(ri, rj, lattice)` 對窗口原點 `(ri−i, rj−j)` 投票，權重 `softmax(sim/τ)`。投票圖做 3×3 box 平滑後取 top-K。輸出 `VoteCandidate`。
- 索引後端：`exact`（GPU 矩陣乘法 + topk，預設）、`hnsw`、`ivf`（`faiss-cpu`，已安裝；**只有 CPU 版**，所以 `hnsw` / `ivf` 在 CPU 上跑，`exact` 仍在 GPU）。
- 單張 slide 約百萬 tile 時 `exact` 比 HNSW 更快也更準；`hnsw` / `ivf` 是為了「圖索引」這個研究面向與多 slide 資料庫的情境，不是效能需要。
- 無訓練。它是 sliding window 的另一個方向：sliding window 問「每個窗口的 tile 跟 query 像不像」，這裡問「每個 query tile 的匹配在哪裡」，讓匹配為它們所屬的窗口投票。單元測試以「參考裡植入一塊 query」驗證投票落在那個窗口。

**F3 WinEmbed（A3 / G3）**
- 窗口 embedding：`e = MLP(depthwise_conv_{kernel=(qr,qc)}(Linear+LayerNorm(F→512)(tiles)))`。**沒有**鄰居混合層：卷積若讓每個 tile 看到窗口外的鄰居，參考窗口看得到周圍而 query 看不到，兩邊就不是同一個函數；現在 `windows(grid)[r, c]` 等於 `query(grid[r:r+a, c:c+b])`（單元測試）。
- 每個位置有可學的位置權重；90° / 270° 的 kernel 是 0° kernel 的轉置，**權重綁定**，只有一組參數。
- 全 slide 的 `e_ref` 一次算完：對所有 tile 先做線性投影，再用 depthwise conv 滑過整張特徵圖，不用逐窗口 gather。
- 推論：`e_q` 在 4 個旋轉各算一次，與對應形狀的 `e_ref` 做 cosine，取 top-K。
- 訓練：整個 slide 窗口池的 full softmax，見「標籤與損失」。

**F4 ClusterBoW（G2）**
- 每個 slide-level 對 tile 特徵做 k-means（第一層 k1=32，每群內 k2=64；GPU 上約 10 次迭代）。
- 窗口描述子：葉群 ID 的直方圖（積分直方圖 `[R, C, k]`）+ 水平 / 垂直雙連詞（群對 ID 雜湊進 4096 桶）。群間與群內各是一個 graph：葉群鄰接圖（相鄰 tile 的群共現）就是「群間 graph」。
- 分數：TF-IDF 加權的直方圖交集；query 在 4 個旋轉各算一次（重新 encode 後以最近質心指派群）。
- 無訓練。預期對 uni2 的邊際效益最低，但失敗模式與其他方法不同，有利於聯集。

**UnionRetriever**
- 各來源提供 raw 候選 → 以「位置距離 ≤ 0.5 tile 且旋轉相容」去重 → 全部用 slidewin 的精確分數重算 `score`（任意窗口的逐 tile cosine 取自 grids，成本低）→ 以 reciprocal rank fusion（k=60）排序後截到 K，讓每個來源都有名額。輸出 `UnionCandidate`，保留各來源的名次。
- **K=100 時聯集仍有意義**：它直接提高 R@100。評估必看「只有該來源找到、slidewin 同 K 沒找到」的真值數。

## 第二階段方法

鏈式：便宜的先對全部 K 個，貴的只對留下來的少數。重新 encode 的方法（S3、S4、`xenc`）只能放在後半；uni2 重新讀 tile 並編碼約 100 到 200 tile/s（由 bench log 的小批次推估，瓶頸是讀圖〔待驗證〕），100 個候選 × 約 20 tile 約十幾秒。

    [S0 S1 S2 S6 → S7]   K → K₂（只用已算好的特徵）
    [S4 S3 S5 xenc → S7] K₂ → K′（可重新 encode 候選）
    → OutputPolicy

| 編號 | 輸入 | 演算法 | 成本 |
|---|---|---|---|
| **S0 Dual** | 已有特徵 | 窗口在 main 格點：query main tile ↔ 參考同格點 tile；query offset tile ↔ 參考 offset 格點 tile，位置 `(row+i, col+j)`。窗口在 offset 格點：query main ↔ 參考 offset；query offset ↔ 參考 main `(row+i+1, col+j+1)`。約 20 對變成約 32 對（5×4 的 query：20 + 12）。分數是這些 cosine 的平均 | 免 encode。`PatchGrid` 的 offset 定義（平移 tile/2、少一行一列）已讀過，**資料驗證〔待驗證〕**：上線前先單元測試退化情況等於第一階段 |
| **S1 CSLS** | 已有特徵 | 對每個參考 tile `t` 算 `r_t` = 對 slide 隨機 2 萬 tile 取 top-10 的平均 cosine（slide-level 一次性）；query tile `i` 算 `ρ_i` = 對全參考取 top-10 的平均。`csls = 2cos − ρ_i − r_t`，窗口分數取平均。另有 `rank-norm`：把 cos 換成該 query tile 對所有參考 tile 的百分位 | 幾乎免費 |
| **S2 Corr（免訓練）** | 已有特徵 | `C = Q Wᵀ ∈ R^{n×n}`。`diag`=對角平均（等於第一階段）；`contrast`=`C_ii − max_{j≠i} C_ij` 的平均；`mnn`=互為最近鄰且在同位置的比例；`sinkhorn`=熵正則 OT（20 次迭代）後對角線質量 | 免 encode，批次 `K×n×n` |
| **S2net** | 已有特徵 | `CorrNet`：把 `C` 當成 `n` 通道、窗口格為空間的影像，加位置通道，3 層 3×3 conv（n→64→64→64）+ 全域池化 + MLP → logit，約 0.1M 參數 | 小 |
| **S3 Shift** | 重新 encode | 對候選在 `{−64, 0, +64}²` px 的位置重新讀 tile 並 encode，取最高分（輸出 `ShiftedCandidate`）。只對 K_in ≤ 30，`exact` 模式；約 30 × 8 × 20 = 4800 tile，先在小子集評估 | 昂貴 |
| **S4 TokenMatch** | 重新 encode raw token | 候選窗口與 query 的 raw token（uni2：每 tile 256 個 patch token）；互為最近鄰且 sim > τ 的配對；每個候選用 GPU 批次 RANSAC（2 點解 `(s, tx, ty)`，每候選 512 個假設，內點門檻 20 px）。分數 = 內點比例（輸出 `TokenMatchCandidate`） | 昂貴 |
| **S5 TileGT** | 已有特徵 | 見「神經模組」 | 小，可對全部 K |
| **S6 CandGraph** | 分數表 | 對 K 個候選建圖，邊 = 中心距離 ≤ 2 tile（任何旋轉）。特徵：`nbr_support`（鄰居分數的加權和）、`prominence`（自己 − 鄰居均值）、`rot_margin`（同位置其他旋轉的最高分與自己的差）、`n_nbr`。**不寫死方向**，由 S7 學正負號 | 免費 |
| **xenc** | 重新 encode | 用另一個 encoder（例如 gigapath）對 K_in ≤ 100 的候選窗口重算 cosine | 昂貴 |
| **S7 Stack** | 分數表 | 見「神經模組」 | 小 |

S6 的警告：通用紋理區會形成一整片高分平台，真位置反而是尖峰，所以平滑可能有害；`rot_margin` 的理由是真位置在旋轉上有特異性，各向同性的紋理則四個旋轉都高。這是推論，由 S7 的學習結果驗證。

## 神經模組

**S5 TileGT**
- 每個 tile 位置 `i` 的 token：`t_i = LN(Linear(1024→256)([q'_i ; w'_i ; q'_i⊙w'_i ; |q'_i−w'_i|]))`，`q' = P(q)`、`w' = P(w)`，`P` 是共用的 `Linear(F→256)`（F=3072 時約 0.79M 參數）。
- 另加：`Linear(n→256)` 對應 `C[i, :]` 與 `C[:, i]`（讓它含有 S2 的驗證資訊，也看得出偏移）；逐 tile 的 scalar（`cos_ii`、`r_t`、`ρ_i`、`log first_rank`、`first_score`、lattice 旗標）。
- 位置：可學的 2D 位置 embedding（窗口格座標）；窗口外圍一圈 tile 的 `margin` 版本**尚未實作**（`WindowBatch` 的 gather 已支援 `margin`，網路端還沒用）。
- 主幹：4 層 pre-LN Transformer，D=256、8 heads、FF 1024、dropout 0.1，每個 head 加相對位置偏置（Swin 式），即 tile 圖上的圖注意力。約 4M 參數。
- 輸出頭：`s`（CLS → logit，排序分數）、`rot_ok`（旋轉是否正確，輔助）、`loc`（預測窗口中心到真值中心的偏移 `(dx, dy)`，單位為窗口邊長的比例，Huber 回歸）。`loc` 讓 S5 輸出 `shift`，不必重新 encode，等於 S3 的學習版。
- 候選脈絡（可選，消融）：對 K 個候選的 CLS embedding 再做 2 層自注意力，輸入含候選之間的相對位置（Δx、Δy、是否同旋轉）與名次，學習 S6。約 1.6M 參數。

**S7 Stack**
- 輸入：`ScoreTable` 逐 FoV 做名次正規化，丟進 3 層 MLP，listwise softmax 訓練；同時提供 `sklearn` logistic regression 當簡單基準（`sklearn` 已在環境）。
- **防洩漏**：有訓練的 scorer（S5、S2net）餵給 S7 的分數必須是 out-of-fold（slide 分成兩折，A 折訓練的模型替 B 折打分，反之亦然）。
- 鏈式使用時每一段各有一個 S7。

## 標籤與損失

### 標籤：候選窗與 FoV 有重疊時，不當負例

視窗約 20 個 tile，與 FoV 重疊的鄰近窗口（例如移一個 tile）其實也包含了 FoV 的大部分內容，
所以**不能**用中心距離切出負例。標籤一律由三個框決定，門檻是面積比例，與 FoV 的 tile 數無關：

| 框 | 定義 |
|---|---|
| 真值框 `B_t` | query 的 tile 格在真值旋轉下的 level-0 框（`BenchCommon.truth_rows`：搜到別層時只放答案的位置與框） |
| FoV 框 `B_f` | 照片在 level-0 的外接軸對齊框（含旋轉 / 縮放 / 角度抖動；比 `B_t` 大） |
| 候選框 `B_c` | 候選窗的 level-0 框（含 `shift`），`CandidateSet.rows` 給出 |

| 標籤 | 條件 | 在損失中 |
|---|---|---|
| `pos` | `IoU(B_c, B_t) ≥ pos_iou`（預設 0.5）**且** `rotation` = 真值旋轉（`matching_rotation`） | 正例 |
| `neg` | `B_c ∩ B_f = ∅`（完全不相交） | 負例 |
| `neg`（旋轉難負例） | `IoU(B_c, B_t) ≥ pos_iou`，但 `rotation` 不是真值旋轉 | 負例：地點對、方向錯，Stage 3 要求旋轉正確，所以它不算命中 |
| `ignore` | 其餘（與 FoV 有重疊但不是正例，也不是旋轉難負例） | **不進任何排序損失**：不在 listwise softmax 的分母、不是 `L_topk` 的負例、不進 BCE |

- `ignore` 並非沒用：只要 `IoU(B_c, B_t) ≥ loc_min_iou`（預設 0.1），它仍有明確的偏移目標，進 `L_loc`（學習對齊）。
- 旋轉是標籤的一部分（Stage 3 要求旋轉正確）；標籤的 pos 用這組框：`IoU(B_c, B_t) ≥ pos_iou`（預設 0.5）**且**旋轉等於真值旋轉；報表看的是 retriever bench 的指標（最近的 main 與 offset 視窗），不是這個 hit。
- **tile 粒度的標籤**（選項 `granularity: tile`）：query tile `(i, j)` 有自己的真值位置；參考 tile 與它的真值框 IoU ≥ 0.5 為正，與 `B_f` 不相交為負，其餘 ignore。適用於逐 tile 的監督（`CorrNet` 的對應矩陣輔助損失、S5 的逐 token 頭）。預設用窗口粒度。

### 損失（S5 / S2net）

| 項 | 公式 | 作用 |
|---|---|---|
| `L_list` | `−log Σ_{p∈P} softmax_k(s_k / T)`，分母只含 pos 與 neg，`T` 可學 | 多正例 listwise，直接優化正例拿到的機率質量 |
| `L_topk` | `Σ_p max(0, m + s_neg^{(K′)} − s_p)`，`s_neg^{(K′)}` 是負例中第 K′ 高的分數 | 讓正例勝過第 K′ 高的負例，直接對應 recall@K′；K′ 取評估的 K′ |
| `L_bce` | 逐點 BCE（pos_weight），只含 pos / neg | 穩定校準（RRT / CVNet 作法） |
| `L_rot` | 對位置正例預測旋轉是否正確 | 輔助 |
| `L_loc` | Huber，預測 `(dx, dy)`，只含 `IoU ≥ loc_min_iou` 者 | 學習式次格點偏移 |

總損失：`L_list + λ₁L_topk + λ₂L_bce + λ₃L_rot + λ₄L_loc`，起始 `λ = (1, 0.5, 0.1, 0.5)`，在驗證集調。

### F3 的損失

- 整個 slide 的窗口池（兩個形狀、約百萬個窗口）做 full softmax：`−log [exp(cos(e_q, e_p)/T) / Σ_{w∈池} exp(cos(e_q, e_w)/T)]`，**分母排除所有與 `B_f` 有重疊的非正例窗口**（同上的 ignore 規則）。這就是檢索指標本身，不需要負例挖掘；計算量與「抽樣約 2 萬窗口」大致相當。
- 旋轉負例項：同一個真窗口，錯誤旋轉的 query embedding 要比正確旋轉的低 `m`。
- 照片增強的 anchor 由 `FovSupply` 的 gap 提供。

### 課程與抽樣

- 每個 FoV 取最多 128 個候選：所有 pos + 64 個最難的 neg（取自第一階段 top-100 內不相交者）+ 隨機 neg 補滿。
- 難負例比例從 0.2 線性升到 1.0（CVNet 式課程）。
- 沒有 pos 的 FoV（真值不在 K 內）不進訓練，但保留在評估的分母。

## 訓練框架與資料準備（`training/RerankHead`）

### P0：先瘦身 mask（前置，所有後續工作的前提）

- 現況：`SlideMask.save` 把 mask 存成每格一個 byte 的 uint8（`result/cache/*/slide=*/mask/mask_*.safetensors`）。273 個檔案共 195.6 GB（平均 716 MB，最大 1.24 GB）；例如 BRACS_1489 是 12376×32370 = 400 MB（`mask_ds=4`）。當時 `MppRoutingHead` 的 cache 目錄 183 GB 幾乎都是它（這些 mask 現在在 `Masks`）。
- 做法：同一個 safetensors 檔內改存 `np.packbits` 後再 `zlib` 壓縮的位元組，metadata 加 `mask_encoding`；`load` 依 metadata 解碼，沒有該欄位就視為舊格式（uint8 原樣），**新舊檔案都能讀**。tissue mask 是大片連續區塊，預期壓到數 MB 以內〔待量測：先在 3 張 slide 上量〕。
- 讀回的陣列與原來逐位元相同，所以 `record_*.json`、mask 的 id、`VERSION` 都不變；測試是「所有現有 mask 轉換後 `np.array_equal`」。
- 轉換工具 `utilities/cli/build_cache/slim_masks.py` 走 Slurm（CPU array）；先寫暫存檔、驗證逐位元相同後才原子替換，並需明確的 `--replace` 才覆蓋原檔。`components`（uni2-pca mask 才有，581 到 814 MB）不在這次範圍。
- 影響的程式只有 `utilities/TissueMask.py` 的 `SlideMask.save` / `load`（讀取端 `TissueMaskConfig.py` 只呼叫 `SlideMask.load`）。

### P1 到 P3：現做現用，不存本地

本機沒有空間，P1 到 P3 的產物**不寫到磁碟**。每次訓練、評估都在記憶體中現做，用完丟棄；
同時保留一個 cache 開關，**目前不使用**：

| 旗標 | 行為 | 狀態 |
|---|---|---|
| `--cache off`（預設） | 全部現做現用；`/work` 只寫 checkpoint、log、結果 csv、wandb | **目前採用** |
| `--cache ro` / `--cache rw` | 讀 / 讀寫 Store（`FeatureMapCache` 的 slide-level 特徵、photos 類特徵、候選名單） | 之後再開，介面不變 |

| 步驟 | `off` 模式的做法 |
|---|---|
| P1 slide-level 特徵 | 由 retriever 在記憶體編碼 (slide, level)（main + offset，`cls_avg`），level 切換時由下一個 visit 取代；retriever 自己的 `_by_ds` 保留這張 slide 看過的 level，下一張 slide 才釋放 |
| P2 FoV 與 query 特徵 | 用**非 cached** 的 `FovSupply(microscope, plan, cfg, mask)` 現場抽位置與渲染照片（seed 與評估用的不同）；query tile（main + offset）在 4 個旋轉各編碼一次；真值（level、ds、matching_rotation、中心、`B_t`、`B_f`）現場算 |
| P3 候選名單 | 用第一階段在 retriever 上現算（`retrieve`），每個 FoV 約 0.2 秒 |

成本估算（粗估，非量測）：ds=1 的一個 level 約 50 萬 tile（BRACS_1411：main 253,934 + offset 251,922 個窗口）；bench log 裡 ki67 的第一張照片（含該 level 編碼）stage 2 為 136 秒，所以一個 ds=1 level 約 2 到 3 分鐘，一張 slide 所有 level 約 4 分鐘。query 特徵每個 FoV 約 1 秒（128 tile），渲染另計。

### 訓練迴圈（visit 為單位）

因為編碼是最貴的一步，一個 **visit** = (slide, level)：編碼一次，抽 `--fov-per-visit`（預設 64）個 FoV，現算候選，然後在這批資料上跑 `--steps-per-visit` 個梯度步（每步 8 個 FoV）。一個 epoch = 走過所有 visit 一次。

- **多個 head 共用同一批資料**：`train.py --heads tilegt corrnet winembed` 在一個行程內對同一批 visit 資料訓練多個 head，各有自己的 optimizer 與 checkpoint；不會因為多個 run 各自重編碼而浪費。
- 平行度：訓練不用 DDP。多 GPU 節點上用 `--shard I/N` 把 visit 切給 N 個行程，各自訓練獨立的 run（沿用 `PrototypicalRoutingHead.sh` 的 `PARALLEL` 做法）；slide 的讀圖與渲染用 CPU worker。
- 最佳化：AdamW，lr 3e-4（S5）/ 1e-3（CorrNet）/ 1e-4（F3），weight decay 1e-2，cosine 排程，bf16 autocast，梯度裁剪 1.0；epoch 30。
- 驗證與選擇：`val` slide，選 `R@K′`（對訓練設定的 (K, K′)）最高者，另存 `last`。
- 檔名沿用 `weight_filename` 的慣例：`uni2_frozen_cls_avg_tilegt_list_best.pt`；超參數（含 `pos_iou`、`hit_iou`）記在 `ckpt['args']`，評估時不符就拒絕。
- 紀錄：wandb 與 `Runtime.py` 沿用 `PrototypicalRoutingHead`。

### 資料切分

- 訓練 slide：沿用 stage-1 訓練的語料（`bracs/train`、`ki67_pure`），與評估用的 `bracs/test`、`ki67_with_photo` 沒有重疊。各資料集的 slide 數我還沒量；`train.py --dry-run` 只印出 slide 數、level 數、tile 數與預估編碼時間，不寫任何東西。
- 選 checkpoint 與超參數用 `val`；`test` 保留到最後。encoder 凍結、維度固定、不用 test 調參（Reality Check 的協定）。
- 另做跨資料集測試：只用 bracs 訓練、在 ki67 評估，反之亦然。
- S7 吃 out-of-fold 分數（見 S7）。

## 評估（`utilities/bench_modules/bench_stage2_retriever.py`）

原來的 window bench 改名為 stage 2 retriever bench：它當時只比 sliding window 的 encoder × pooling × 算分，
因為當時只有 sliding window。現在 **arm 是一個 retriever，後面可以接任意多個 reranker**，用一套文法命名
（`stage2_retrieval/ArmSpec.py`）：

```
[encoder:]Method::pooling[+操作][@K]   用 :: 接起來，第一段是 retriever，其後都是 reranker
uni2:BagOfTiles::cls_avg+gap::gigapath:CSLS::grid2x2+csls@100
```

- `encoder:` 單冒號，沒寫就用 retriever 的 encoder；reranker 可以另外指定（它自己編碼需要的東西）。
- `+操作` 是這個方法內部可選的一個操作；沒有可選操作的方法不寫 `+`。不接 pooling 的方法 pooling 填 `raw`。
- `@K` 是「這一段處理完只留前 K 個」；retriever 沒寫就是 `--k`（名單最長長度），reranker 沒寫就保留收到的全部；粗層名單不到 K 就用全部。

| 方法 | 位置 | `+` 可選 |
|---|---|---|
| SlidingWindow | retriever | `mean`、`geomean`、`min`（逐 tile 餘弦怎麼合成窗分數） |
| BagOfTiles | retriever | `gap`（窗內平均向量）、`quad`（四象限平均） |
| TileVote | retriever | `exact`、`hnsw`、`ivf`（最近鄰索引） |
| ClusterBow、WinEmbed | retriever | 無（WinEmbed 要有該 encoder 與 pooling 訓練的權重） |
| Dual | reranker | `dual`、`cos_main`（依哪一欄排序） |
| CSLS | reranker | `csls`、`rank`（校正的方式） |
| Corr | reranker | `diag`、`contrast`、`mnn`、`sinkhorn` |
| CandGraph | reranker，pooling 填 `raw` | `nbr_support`、`prominence`、`rot_margin`（依輸入名單的排序分數建圖） |
| Shift、CorrNet | reranker | 無 |
| TokenMatch | reranker，pooling 填 `raw` | 無 |
| TileGT | reranker | `tilegt`、`tilegt_rot`（要權重） |

**每個 retriever arm 有兩種模式**（見上）：`whole` 與 `production`，由同一份分數圖得到。reranker 只重排給定的名單，只有 production 模式。
sliding window 用一次串流掃描算所有 pooling 與算分；其他 retriever 與所有 reranker 由 `ArmEngine` 跑。每個方法把 `propose` 拆成
`window_maps`（每個視窗的分數）與 `top_k`（top-K 加 NMS），whole 與 production 從同一份 `window_maps` 得到。

**production 名單與 K**：每個 retriever 只做一份最長的名單（`--k`，預設 1000）；較短的 K 是它的前綴，所以報表用一份名單就給出
`recall_all@K`（K = 30、100、200、300、500、1000），看「涵蓋真值的比率」隨長度增加到哪裡就不值得。
答案不在名單內的名次是 `pool + 1`，任何 K 都不算命中（K 超過一層的視窗總數時也不會把它算進去）。

**reranker 怎麼拿特徵**：retriever 交出候選名單和 `context()`（`RetrievalContext`：層、ds、格點、`source`、`reader`、`encoders`，是「怎麼拿到特徵」的把手，
不複製任何東西）。reranker 要的 pooling 或 encoder 和 retriever 不同時，bench 為它建一個 `FeatureHosts` 裡的 host（每個 (encoder, pooling) 一個，共用已載入的權重，
一層只編碼一次、整張 slide 保留），把那個 host 的 context 交給它；scorer 不需要知道。`raw` 的方法不需要整層特徵，拿到只會現場編碼 tile 的 `LeanSource`。
`extra_poolings` 這個想法取消，等正式部署再討論。

**k × h**：`--rerank-on <retriever arm...> --rerankers <reranker...>` 產生 k 個 retriever 乘 h 個 reranker 的 arm；同一個 retriever 的名單只算一次，
同一個 reranker 的同一次打分用不同操作切出不同排序。報表另外把每個 `A::R` 對 `A`（它收到的名單）配對。

每個 arm 一個目錄（`run-<id>/<entry 名稱>/<slide>__L<k>.csv`，名稱規則見 `Naming.py`），retriever 與 reranker 的產物不共用檔案；改一個 arm 只重跑那個 arm；
同一份 `stage2_retriever.csv`、同一張報表。arm 名字帶 encoder，所以 `--encoder uni2` 與 `--encoder gigapath` 兩次 run 的 CSV 可以一起 `--report-only`。
FoV 從 `supply_for` 取（與 pipeline bench 同一批照片）。job 名是 `Stage2RetrieverBench`；舊名 `WindowRetrievalBench` 的結果與 cache 不動，舊 CSV 當作 whole 模式的 sliding window 讀。

**不再有打包的 recipe**（`cheap`、`heavy`、`union`）：`TWOPHASE_RECIPES` 只留給 pipeline 的 `--stage2 twophase:<recipe>`，bench 不用。

**端到端**：選出最佳鏈後跑 `BenchLocaScope --stage2 twophase:<recipe> --stage3-topk K′`，再用 `AnalyzeLocaScope` 看 stage 2 漏掉率與 funnel。

**單元測試**（`utilities/test_modules/TestRerank/`）：
- `WindowBatch.cos` 的平均等於第一階段分數（逐位元一致）。
- F1 積分圖等於暴力平均；F2 在 `m = ∞` 時等於 slidewin 分數的和。
- S0 在只有 main 時退化成第一階段。
- 標籤：`pos` / `neg` / `ignore` 在已知幾何下的分類；`ignore` 不進分母。
- `TwoPhase` 的 id 隨任何子設定改變（接入 `test_config_identity`）；巢狀 tuple 的 JSON 來回。
- 損失的形狀與梯度；`slim_masks` 的逐位元來回；`faiss` 能 import 且 `hnsw` 的最近鄰與 `exact` 在小例子上一致。

## 檔案配置

沿用「訓練放 `training/`、stage 只載入權重」。

| 位置 | 內容 |
|---|---|
| `stage2_retrieval/Retriever.py` | `Retriever` 介面、`RetrievalOutput`、`EncoderPool`、橋接鎖 |
| `stage2_retrieval/StageInterface.py` | `Candidate` 家族、`CandidateSet` |
| `stage2_retrieval/Context.py` | level 路由與編碼（`SlideLevels`、`WindowSearch`）、`GridSource` / `MemorySource` / `OnDemandTiles`、`WindowBatch`、框與 IoU、reranker 的共用做法（`ScoreTable`、`RerankFeatures`） |
| `stage2_retrieval/ArmSpec.py` | bench 的 arm 文法：`[encoder:]Method::pooling[+op][@K]` 用 `::` 接起來，retriever 在前、reranker 在後；`config_of_arm` |
| `stage2_retrieval/blueprint.md` | 重構藍圖；附錄是每個方法的詳細資料流（推論與訓練），含張量形狀 |
| `stage2_retrieval/Retriever{SlidingWinSimRot,BagOfTiles,TileVote,ClusterBow,WinEmbed}.py` | 第一階段（各有 config 與 recipe） |
| `stage2_retrieval/RerankerScorers.py` | S0、S1、S2（免訓練）、S6 |
| `stage2_retrieval/RerankerEncode.py` | S3、S4、`xenc` |
| `stage2_retrieval/RerankerNeural.py` | 載入權重的 S2net、S5、S7 |
| `stage2_retrieval/MultiPhases.py` | 串接與 recipe 表 |
| `stage2_retrieval/Naming.py` | entry 的名稱 |
| `aiNNModel/models/RerankHeads.py` | `TileGraphTransformer`、`CorrNet`、`CandidateContext`、`WindowAggregator` |
| `aiNNModel/models/RerankChoices.py` | 架構 registry |
| `aiNNModel/models/common/Checkpoints.py` | 新增 `save_rerank_checkpoint` / `build_rerank_from_checkpoint`（只新增函式） |
| `training/RerankHead/` | `spec.md`、`plan.md`、`Datasets.py`、`Episodes.py`（visit）、`Losses.py`、`Runtime.py`、`cli/{train,evaluate}.py` |
| `utilities/bench_modules/RetrieverArms.py`、`bench_stage2_retriever.py` | 評估（每個 arm 有 whole 與 production 兩種模式） |
| `utilities/cli/build_cache/slim_masks.py` | P0 轉換工具 |
| `jobscripts/RerankHead/*.sh`、`jobscripts/SlimMasks.sh` | Train、Eval、SlimMasks（沒有 Prepare job：P1 到 P3 現做現用） |
| `utilities/test_modules/TestRerank/` | 測試 |

## 基礎模組改動

能不改就不改；必要的改動如下，**每一項動手前需要明確批准**。

| 檔案 | 改動 | 原因 | 狀態 |
|---|---|---|---|
| `environment.yml`、`locascope_environment.yml` | 加 `faiss-cpu==1.13.0` | F2 的 `hnsw` / `ivf` | **已完成**（env 已裝，只新增這一個套件；只有 CPU 版；import 尚未驗證，見單元測試） |
| `utilities/TissueMask.py` | `SlideMask.save` / `load` 的 mask 編碼（P0） | 瘦身 | 已完成（未測） |
| `utilities/ConfigIdentity.py` | 支援「巢狀 config 的 tuple」，見下 | `TwoPhase` 的 config | 已完成（未測） |
| `stage2_retrieval/StageInterface.py` | 上面的 `Candidate` 家族、`origin_l0`、`rows()`、`Reranker`、`Retriever.context` | 介面 | 已完成（未測） |
| `stage2_retrieval/SlidingWinSimRot.py` | 新增 `context()`、`extra_poolings`（NOT_IDENTITY）、K 的 recipe；`candidate_set` 的 NMS 向量化（輸出不變，只改效能） | 介面、K 變大時的效能〔待量測〕 | 部分完成（未測；`extra_poolings` 未做） |
| `utilities/LocaScopePipeline.py` | `stage2()` 把 `ctx` 傳給 reranker；reranker 的 recipe 表（可選） | 介面 | 部分完成（未測；recipe 表未做） |
| `utilities/bench_modules/BenchCommon.py` | `run_stage2` 增加兩個選用 role（`first_phase`、`rerank_scores`）並登記 | 讓 composite 的中間分數落地 | 已完成（未測） |
| `utilities/bench_modules/bench_locascope.py` | 用 `roles_of(stages)` 取代 `ROLES` | stage 2 的額外 table 進 entry | 已完成（未測） |
| `utilities/test_modules/test_config_identity.py` | 新 config 的 recipe lint 與 id 測試 | 慣例 | 已完成（未測） |
| `utilities/_paths.py` | rerank 權重與結果目錄 | 慣例 | 已完成（未測） |

**不改**：`Cache`、`Store`、`PatchingLib`、`TileEncoderFunc`、`FovSupply`、`TileSampler`。
`Truth` / `query_grid_centre` / `matching_rotation` 目前在 `BenchCommon`，訓練要用；先從那裡 import（訓練 import bench 模組在 Prototype 訓練有先例），日後可考慮搬到中性位置。

### 巢狀 config 的 tuple（決議：支援）

現況：`parts_of` 遇到「值是 tuple 且元素是 config」的欄位，會走 `enc(value)` → `_canon` →
`str(config)`，也就是用 dataclass 的 repr 當 identity：它包含 `NOT_IDENTITY` 的欄位（例如
`batch_size`），也不看各 config 自己的 `BASELINE`。這是一個靜默的錯誤。現有的 config 沒有這種欄位
（已 grep `Tuple/Sequence/List[...Config]`），所以支援它**不會改變任何現有 id**。

改法（只在值是非空、元素全是 config 的 tuple / list 時才走新路徑）：
- `parts_of`：每個元素寫 `name[i]=<ClassName>`，再接該元素自己的 `identity_parts()`，前綴 `name[i].`；順序（鏈的順序）因此進 id。
- `provenance`、`versions_of`：同樣逐元素遞迴。
- `_as_plain` / `_from_plain`：元素寫成 `{'name': REGISTERED_AS, 'fields': ...}`，回讀時查 registry（元素的類別可以不同）。
- `BASELINE` 對這種欄位寫 `()`。
- `test_config_identity` 的 registry 測試要能為這種欄位產生一個「改了就該改 id」的變體。

## 邊界

- Stage 3（`SiftRansacLocalizer`）直接吃 `CandidateSet`：crop 的 level-0 讀圖點由上面那條公式算，加上 padding，記帳全程用 level-0。它用自己 `build(wsi)` 建的 reader 讀，不向 retriever 借。
- 以前的作法（以鴨子定型讀候選的 `best_x`、`best_y`、`best_region_index`、`best_rotation`、`ds`，crop 原點記成 `int(region.x / ds) + x0`）把 region 原點的小數截掉。openslide 以 bilinear 取樣（位置的小數會被內插出來），所以回報的位置偏 `-frac(region.x / ds) * ds` 個 level-0 像素，方向固定。實測（當時 `MppRoutingHead` cache 的 mask，現在在 `Masks`；`diag_read_exp.py` 的 phase / origins flow）：BRACS 幾乎每個 region 都偏將近一個 level 像素，L1 約 1 µm、L2 約 4 µm；Ki67 在 L1、L2 為 0，L3 起約 1 µm 以上。
- Stage 3 的 `topk` 跟著 K′ 設定；Stage 3 要求旋轉正確（硬要求，見「目標與指標」）。

## 決議與待確認

**已決議**
1. 候選窗與 FoV 有重疊者不當負例；標籤由 `pos` / `neg` / `ignore` 三態決定，門檻為面積比例。
2. `Candidate` 基礎類別不變，方法專屬資訊用子類別加欄位。
3. K 是實驗軸，主要目標是 top-100 的 recall；部署時才可能用 K=500。
4. P1 到 P3 現做現用，不存本地；cache 模式保留為旗標，目前不使用；先瘦身 mask。
5. Stage 3 要求旋轉正確（硬要求）：旋轉不對的候選視為未命中；位置對但旋轉錯的窗口是負例（旋轉難負例）。
6. 支援巢狀 config 的 tuple（不影響現有 id）。
7. 安裝 `faiss-cpu` 並更新 yml（已完成）。

**待量測 / 待驗證**
1. `pos_iou`、`hit_iou`、`loc_min_iou` 的預設值（0.5 / 0.5 / 0.1）是起始值，要用資料看正例數量與 stage 3 的實際成功率後再定。
2. S0 Dual 的格點對位（讀程式推得，未用資料驗證）。
3. 壓縮後 mask 的實際大小與載入時間；`faiss` 在計算節點上能否 import。
4. K=500 時 `candidate_set` 的 NMS（Python 迴圈）耗時。
5. 重新 encode 候選的吞吐量（bench log 的 100 到 200 tile/s 是小批次的推估）。
6. 訓練 slide 的數量與預估編碼時間（`--dry-run`）。

**範圍外**
- 目前最大的損失其實是 stage 1 路由錯（ki67 19.5%、bracs 5.5%），而 `CandidateSet` 是單一 level。若之後要讓 reranker 在多個 level 的候選之間選，需要擴充 `CandidateSet`。這次不做。
