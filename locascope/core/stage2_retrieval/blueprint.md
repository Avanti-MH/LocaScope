# Stage 2 重構藍圖

> **狀態：重構已完成（2026-11-02 核對並瘦身）。** 這份原本是施工依據：目標、`Retriever.py` 的介面草稿、每個檔案的架構、
> pipeline／bench 的串接、產物表、施工順序。完成之後，目標與介面草稿的內容已經在 `Retriever.py` 開頭的 docstring 裡；
> 檔案架構與產物表隨程式改了名（草稿裡的 `Context.frame_at`、`ArmTables.arm_entry`、`ArmTables.add_level`、
> `truth_rank_offset_in_all`、`stage2_output`、`max_rot_per_place` 現在都不存在）；施工順序與驗證紀錄過時了。這些都刪掉了
> （原文：`git show HEAD:stage2_retrieval/blueprint.md`，在 commit 這次瘦身之前 HEAD 就是原文）。
> 留下：名詞與規則、串接與 id／cache 的規則、實作時與草稿不同的決定、附錄 A 各方法的資料流。
> 介面的現況看 `Retriever.py`、`MultiPhases.py`、`Context.py`、`StageInterface.py`。GloRet／LogLoRet（`RetrieverGloRet.py`）見 `proposal.md`。

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

## 二、stage 2 內部的串接

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

`MultiPhases`（`MultiPhasesConfig(phases=(...))`）把它們串起來；arm `A::B::C` ≡ `MultiPhasesConfig(phases=(A, B, C))`。
每段 `topk_in` 不能超過上一段的 `raw_topk`；方法放在不能工作的位置時，串接被拒絕（`Retriever.py` 的 `takes_wsi`／`takes_candidates`、
`ArmSpec.METHODS`）。`OutputPolicyConfig`（`max_rot_per_place`）已刪除、不移植：當時 repo 內每個使用處都設 0，等於沒作用；
要做「限制同一地點的旋轉數」的實驗時，寫成一個接在鏈上的 reranker。

---

## 三、pipeline 與 bench 怎麼對上同一份 entry

- **每一段一個 cache entry**：`result/cache/<job>/…/stage1=<s1|oracle>/stage2/`（真實照片鏈則在 `photos=<id>/` 底下）。每段 id =
  `stage_hexid(段 config, (mask, 上一段 id))`，名稱帶 `_after-<上一段 hash>`；stage 3 在 `stage2=<最後一段 entry 名>/stage3/`。
  所以 pipeline 與 bench 只要 config 相同，就是同一個 id、同一個檔。
- **arm 文字轉 config 放在 stage 2 套件**（`ArmSpec.config_of_arm`；各 stage 套件的 `recipe()`）：pipeline 只經由 `recipe()` 把文字轉成 config，
  不需要知道 stage 2 的文法；bench 呼叫同一個函式，兩邊的 config 由同一份程式產生，id 一定相同。
- **bench 可以走快速路徑**（SlidingWindow 的串流、level store），為了比較所有方法、快速看到結果；但算法結果必須相同，每條快速路徑都有 gate
  證明這點。pipeline 不開 level store／feature store。
- **答案怎麼算只有一份定義**：答案是哪格、排第幾、名單裡的位置，只在 `BenchCommon`（`answers`、`ranked_ranks`、`listed_places`），
  bench、pipeline 的 truth、analyze、plot 都用它。bench 不另存評分列：`result/Stage2RetrieverBench/<encoder>/stage2_retriever.csv` 每張 slide
  從 cache 的 entry 讀出（`cli/metrics/analyze_stage2_metrics.py` 用 bench 的同一組旗標讀 cache 重組出 rows，寫 csv 與報表；`--csv` 則直接讀現成的 csv）。一個 entry 存什麼，準則見 `ARCHITECTURE.md`「entry 存什麼」。

---

## 四、實作時與草稿不同的決定

- **下游自己做特徵的方式**：reranker 沒有可取用的上游特徵時（例如上游名單從 cache 讀回），用和第一階段同一個 `encode_level` 編碼名單所在的
  整個 level，而不是只編候選 tile。這樣「上游現算」與「從 cache 讀回」兩種情況的數字逐位元相同；只編候選 tile 的讀取路徑不同，fp16 下最後幾位可能不同。
- **reranker 的共用做法** `RerankerScorers.ColumnRanker`：讀前 `topk_in`、建 incoming table、取特徵、寫欄位、依欄位排序、欄位當 `columns`
  副產物往下傳。它是 `Retriever` 子類，不是另一種介面；`RerankerEncode`、`RerankerNeural` 的方法也繼承它。
- **sibling 共用一次打分**（bench）：只差排序欄位的 reranker（例如 Corr 的四個欄位）打分一次，其他欄位用 `Context.reordered` 重排；結果與各自打分相同。
- **bench 的 level store**：`RetrieverArms.LevelStore`，key 是 `Context.feature_key`，只留目前的 level；同一份特徵的所有 arm 共用一份。
- **entry 的寫入只有一處**：`BenchCommon.write_phase`，pipeline 與 bench 都用它；記錄由 `BenchCommon.phase_record` 產生，名稱與 hash 由
  `BenchCommon.phase_ids` 產生。
- **`tile_sims` 只寫前 10 個**（`RetrieverSlidingWinSimRot.TILE_SIM_ROWS`）：名單 1000 個時全寫是每個 FoV 兩萬列。
- **`Stage` 的 stage 2 一律是 chain**：單一 retriever 是一段的 `MultiPhases`（`BenchCommon.as_chain`），所以每個 stage 2 都按段建 entry；
  `stage_entries()[2]` 是最後一段（stage 3 讀的名單）。
- **class 是 identity 的一部分**：`RetrieverConfig.identity_parts` 前面有 `retriever=<class>`（理由在 `RetrieverConfig` 的 docstring：各自在預設 op 的
  BagOfTiles+gap、TileVote+exact、ClusterBow、Dual+dual 都只差 `encoder`，id 會相同）。
- **ClusterBow 不是決定性的**：k-means 的 `index_add_` 在 CUDA 上是 atomic add，每次跑少數列的名次可能小動；使用者接受，不改成決定性。

---

## 附錄 A：各方法的資料流

算法在這次重構中不變。其中「`ctx = RetrievalContext`」「bench 把 host 的 ctx 交給 reranker」的接法已被取代（`RetrievalContext` 已不存在）：
特徵由各 retriever 自己做，或經橋接鎖取用上游的副產物（`Retriever.py` 開頭的 docstring）。

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
- `locascope/training/RerankHead/train.py` 的 encoder 與 pooling 由 `--encoder`（`uni2`、`gigapath`）和 `--pooling`（預設 `cls_avg`）指定，heads 讀同一個 pooling；`--k`（預設 500）是訓練用的名單長度。
