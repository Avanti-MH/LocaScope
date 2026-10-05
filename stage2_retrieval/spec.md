# Stage 2 — 檢索（草稿）

Stage 2 把一張 query 對到 WSI 上的候選視窗，分兩階段：

    第一階段  粗檢索   整張 slide → top-K 候選窗
    第二階段  重排序   query + top-K → top-K′ 候選窗      K′ ≤ K

檢索階段不產生更細的物理位置；位置細化是 stage 3 的事。

## 詞彙

query、shot 沿用 `CLAUDE.md`；pool、rank、答案視窗、arm、pooling 見
`bench_window_retrieval.py` 開頭的 VOCABULARY。

| 中文 | English | 定義 |
|---|---|---|
| 粗檢索 | coarse retrieval | 第一階段：整張 slide → top-K 候選窗 |
| 重排序 | reranking | 第二階段：query + top-K → top-K′ 候選窗 |
| 第一階段候選數 K | K | 粗檢索輸出的候選數，設定值 |
| 第二階段候選數 K′ | K′ | 重排序輸出的候選數，K′ ≤ K，設定值 |
| 圖塊 | tile | 256×256 px，encoder 的輸入單位 |
| 組織區域 | region | 組織 mask 的一個 bbox；格點以它的原點為準 |
| 視窗 | window | query 的 tile 格（5×4，旋轉 90/270 度時 4×5）落在參考格點上的位置 |
| 格點 | lattice | 參考 tile 的一組排列，有 main 與 offset 兩組 |
| 主格點 | main lattice | 從 region 原點起、間距 256 px |
| 偏移格點 | offset lattice | main 平移 (128, 128)；取代舊稱 overlap（程式與 CSV 欄位仍叫 overlap） |
| 候選 | candidate | `(region_index, lattice, row, col, rotation, score)` |
| 候選集 | candidate set | 依 score 由高到低排好的候選清單（位置就是名次），加上它的座標框架：level、ds、每個 region 的格點 |
| 主格點召回率 | main recall, `recall_main@k` | 最近的 main 視窗，在只有 main 視窗的池裡排名 ≤ k 的比例 |
| 偏移格點召回率 | offset recall, `recall_offset@k` | 最近的 offset 視窗，在只有 offset 視窗的池裡排名 ≤ k 的比例 |
| 混合召回率 | combined recall, `recall_all@k` | main + offset 混合池裡，上面兩個視窗較前者的排名 ≤ k 的比例 |
| 原始 token | raw token | encoder 的原始輸出，每個 tile 265 個（UNI2：9 個 prefix + 16×16 個 patch） |
| 圖方法 | graph method | 以 tile 為節點、關係為邊的檢索方式；類型未收斂 |

## 現況與目標

| | 現況 | 目標（未實作） |
|---|---|---|
| 第一階段 | `SlidingWinSimRot`（canonical，滑窗 cosine，試 4 個旋轉） | 同左，加 graph 方法 |
| 第二階段 | 無 | raw cosine、transformer |
| 候選 | `Candidate` / `CandidateSet`（`StageInterface.py`，見下） | 同左 |

## 候選

視窗 = query 的 tile 格（5×4，旋轉 90/270 度時 4×5）落在參考格點上的位置。
參考有兩組格點：`main`（region 原點起，間距 256 px）與 `offset`（main 平移
(128, 128)）。

    Candidate    = (region_index, lattice, row, col, rotation, score)
    CandidateSet = (candidates, level, ds, grids)

- `lattice` 是 `'main'` 或 `'offset'`（取代 `from_overlap`）。
- `Candidate` 只放 retrieval 自己找出來的東西。可從 input 推論的（像素座標、
  視窗大小、名次、第一階段的分數）不放。
- 身分 = 前五個欄位。兩個方法的候選是不是同一個視窗，就是這五個欄位相等。
- 所有第一階段方法都輸出這種格點視窗，評估指標因此共用。
- `CandidateSet` 是候選清單加上它的座標框架。`candidates` 依 score 由高到低，
  位置就是名次。`region_index`、`row`、`col` 只對一份 regions、一個層才有意
  義，所以那份 regions 的格點（`grids`，每個 region 一個 `PatchGrid`，帶
  level-0 原點）、`level`、`ds` 跟著清單走，不讓呼叫端拿兩個變數自己配對——
  與 `WsiFeaturesMap` 存在的理由相同。推論的輸入跟著輸出走，推論的結果不存。
- 視窗大小不在 `CandidateSet` 裡。它是 query 的 tile 格，只由 query 決定，而
  query 本來就沿著副 flow 傳到 stage 3；需要它的方法（`window_tiles`、
  `window_l0`、`centre_l0`）直接吃未旋轉的 `QueryPatchContainer`，並檢查它的
  tile 大小與 `grids` 的相同。
- 位置只有一條公式：視窗左上角的 level-0 讀圖點 =
  `grids[c.region_index].tile_origin_l0(c.lattice, c.row, c.col)`。不經過截斷
  的 level-n 整數；bench、視覺化、pipeline、stage 3 都用這一條。

## 介面

每一階段吃前一階段的輸出，pipeline 只是串接：

    stage 1   build(wsi, mask)                              每張 WSI 一次
              estimate(query)               → EstMppResult
    stage 2   第一階段  build(wsi, mask)                    每張 WSI 一次
                        retrieve(query, EstMppResult) → CandidateSet   長度 K
              第二階段  build(wsi, encoder)                 每張 WSI 一次
                        rerank(query, CandidateSet)  → CandidateSet   長度 K′
    stage 3   build(wsi)                                    每張 WSI 一次
              localize(query, CandidateSet, rank=0)  → 定位結果

    r1 = est.estimate(q);  r2 = ret.retrieve(q, r1);  r3 = loc.localize(q, r2)

- `retrieve` 在 `EstMppResult.chosen_level` 上檢索。層由 stage 1 選一次，stage 2
  不再自己選；每一層的格點與特徵第一次用到時才建，之後快取。
- K、K′ 是設定，不寫死。`rerank` 輸入輸出同型，第二階段可以整個略過。每個方
  法自有 config 與 build 產物（快取以設定 id 為鍵），沿用 stage 1 的樹幹加分支。

## 第一階段方法

- **滑窗 cosine**（現況）：query 的每個 tile 對位置對應的參考 tile 算 cosine，
  以 `mean` 合成視窗分數。
- **graph**（未收斂）：想法有三：模型學出 recall 最高的 graph（要訓練）；
  分群再分群、群間與群內各是 graph；FAISS 建 graph。訓練程式放
  `training/<名稱>/`，stage 2 只載入權重。

## 第二階段方法

- **raw cosine**：候選窗與 query 都用 raw token 算 cosine。只對 K 個候選窗
  現場 encode，不需要快取整張 slide 的 raw。
- **transformer**（未收斂）：輸入表示與是否訓練未定。

## 邊界

- Stage 3（`SiftRansacLocalizer`）直接吃 `CandidateSet`：crop 的 level-0 讀圖點
  由上面那條公式算，加上 padding，記帳全程用 level-0。它用自己 `build(wsi)`
  建的 reader 讀，不向 retriever 借。
- 以前的作法（以鴨子定型讀候選的 `best_x`、`best_y`、`best_region_index`、
  `best_rotation`、`ds`，crop 原點記成 `int(region.x / ds) + x0`）把 region 原
  點的小數截掉。openslide 以 bilinear 取樣（位置的小數會被內插出來），所以回報
  的位置偏 `-frac(region.x / ds) * ds` 個 level-0 像素，方向固定。實測（MppRoutingHead
  的 mask，`diag_container_retire.py` phase / origins 段）：BRACS 幾乎每個 region
  都偏將近一個 level 像素，L1 約 1 µm、L2 約 4 µm；Ki67 在 L1、L2 為 0，L3 起
  約 1 µm 以上。

## 評估

- 第一階段：`utilities/bench_modules/bench_window_retrieval.py`。答案是離
  query 最近的 main 視窗與 offset 視窗，量 `recall_main`、`recall_offset`、
  `recall_all`@k，用來選 K。
- 第二階段：以第一階段 top-K 為輸入，量重排後真值的名次與 top-K′；bench 尚未寫。
