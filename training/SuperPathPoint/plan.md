# SuperPathPoint — 2026-08-31 之後的計畫

`spec.md` 是規格，這份是**現在要做什麼、按什麼順序**。決定的理由留在這裡，
量測的結論進 `log/TODO.log`。

---

## A. 現況：ReEvalSuperPathPoint 的讀數

四個 arm 的 checkpoint（2026-08-31，50 epoch、batch 128、2,050 步）在**對齊過的
點數**下重新評分。對齊的方式是 `score_threshold=0` + `max_points=N`，也就是取分數
最高的 N 個，密度被構造性地釘死，`decoy` 因此對每個 arm 是同一個量。

### 欄位

| 欄位 | 意思 |
|---|---|
| `repeat` | 兩個 view 的點對得上的比例。越高越好 |
| `decoy` | 把一組整體平移超過 NMS 半徑之後還「對得上」的比例 = **光靠密度就能拿到的分數** |
| `unif` | N 個點均勻散開時 `decoy` 的理論值，`1 - exp(-81N/256²)` |
| `margin` | `repeat / decoy` —— 扣掉密度紅利之後剩下的，**這才是分數** |
| `ceiling` | `1/decoy`，`margin` 的上限。`repeat ≤ 1` 所以 decoy 就是天花板 |

### A.1 有一個乾淨的交叉

`margin`（全部 1044 對 pair）：

| N | gray（從頭） | gray_pre | 逐片判定 |
|---|---|---|---|
| 40 | **6.82** | 5.92 | **gray 贏兩片** |
| 80 | **4.54** | 4.29 | 未定（各贏一片） |
| 160 | 2.95 | **3.05** | 未定（各贏一片） |
| 320 | 1.78 | **2.07** | **pre 贏兩片** |
| 420 | 1.50 | **1.74** | **pre 贏兩片** |

判定用 spec.md 1 第四列：一片贏一片輸算未定，不是小勝。

**結論：從頭那個模型最好的 40 個點不輸 pre，它只是沒有更多好點。**

原始 `repeat` 說同一件事：N=40 是 0.575 對 0.563（打平），之後 pre 拉開到
+0.07 ~ +0.08。從頭那組的問題不是「點不準」，是「準的點只有那麼多」。

一個 N 看不到這件事。ladder 是為了這個存在的。

### A.2 RGB 不是空結果 —— 先前的判斷被推翻

**10 次比較，RGB 全部小贏：**

| N | gray → rgb | gray_pre → rgb_pre |
|---|---|---|
| 40 | 6.824 → 6.832 | 5.924 → **6.046** |
| 80 | 4.539 → **4.702** | 4.289 → **4.385** |
| 160 | 2.948 → **3.106** | 3.047 → **3.131** |
| 320 | 1.775 → **1.886** | 2.066 → **2.143** |
| 420 | 1.498 → **1.566** | 1.742 → **1.813** |

在 top320 和 top420，RGB 在兩個 init 下都**兩片都贏**。

幅度 2-5%，但方向一致到不像雜訊。先前判為「空結果」是因為兩個 arm 都被
`max_keypoints=420` 釘在同一點上，只看得到 ladder 的一格。

**不改變「先用 gray」的決定**（省一半算力換 2-5% 不划算），但這是一個 OPEN 的
發現，不是關掉的問題。

### A.3 點是聚集的，而且聚集程度隨 N 變

| N | `unif` | gray | gray_pre |
|---|---|---|---|
| 40 | 0.048 | 0.084 (1.8x) | 0.095 (2.0x) |
| 160 | 0.179 | 0.210 (1.2x) | 0.228 (1.3x) |
| 420 | 0.405 | 0.442 (1.1x) | 0.419 (1.0x) |

小 N 兩個都明顯聚集（點集中在組織區而非散在整張 tile，合理）。大 N 時 pre 反而
比 gray 散。

### A.4 從頭那組為什麼點數會飆到 NMS 的幾何上限

detector 是每個 cell 的 65 類 softmax：64 個位置 + 1 個 **dustbin**（「這裡沒有
點」）。`Decoders.py:76` 做完 softmax 丟掉 dustbin **且不重新正規化**，所以一個
cell 的 64 個像素值加起來是 `1 - d`，平均像素值是

    v̄ = (1 - d) / 64

而多數 cell 的正確答案就是 dustbin，所以 `CE ≈ -ln(d)`。兩者接起來：

| | CE | dustbin `d` | 平均像素 `v̄` | 對閾值 0.015 |
|---|---|---|---|---|
| 什麼都沒學 | 4.174 = ln 65 | 0.0154 | 0.01538 | 幾乎正好等於 |
| 從頭 ep49 | 3.27 | 0.038 | 0.01503 | 還在閾值**之上** |
| pre | 0.19 | 0.827 | 0.00270 | 閾值的 1/5.6 |

**同一個 0.015 落在兩個完全不同的位置。** 對 pre 它設得住（只有尖峰過得去，
約 150 點，內容決定）；對從頭它壓在自己圖的平均值上，等於沒過濾，剩下全靠 NMS 砍
—— 那是幾何決定的，約 10³ 個。

閾值要開始有作用需要 `v̄ < 0.015`，即 `d > 0.04`，即 **CE < 3.22**。從頭那組最後是
3.27、中途最低 3.12，**整輪都在這條線上來回穿**。

一句話：**它不是想要更多點，它是還沒學會拒絕。**

### A.5 從頭那組連「不看圖」的水準都還沒到

CE 有三個算得出來的刻度：

    4.17 ──────────── 3.27 ─────────── 1.0 ────── 0.19
    均勻            從頭(ep49)      不看圖的      pre
                                     基準線

**「不看圖的基準線」**：label 平均每 tile 146 點、1024 個 cell，所以有點的 cell 佔
`f = 0.143`。一個永遠輸出 `d = 0.857`、其餘平均攤開、完全不看影像的模型：

    CE = 0.857 x (-ln 0.857) + 0.143 x (-ln(0.143/64)) = 0.13 + 0.87 ≈ 1.0

從頭那組是 3.27，離這條線都還很遠。**這不是模型學不會，是還沒學到。**
2,050 步，上游 SuperPoint 是 600,000 步。

而 pre 的 0.19 遠低於 1.0，證明它不是靠猜基本比例過關 —— 也證明**這個架構在這批
資料上做得到 0.19**，不是容量或 label 噪音的限制。

---

## B. 方向：先做 pretrain，但不關掉另一條路

**決定：接受。理由：保留。**

資料支持的是「在我們在乎的密度（N ≥ 160）pre 比較好，而且兩片都贏」。

資料**不**支持「隨機初始化是錯誤方向」—— 那個 arm 沒有被公平跑過：2,050 步、
augmentation 凍住、CE 3.27。**用一個沒跑完的實驗否定一條路，是這個專案自己定的
規矩要避免的事（ClaudeRules §8：先量再校準）。**

但決定不需要那個強命題。先做 pre 的三個理由都成立：

1. 它現在就比較好
2. 它現在就能用（CE 0.19，足以當 Stage B 的 detector）
3. 它便宜 —— 不用等收斂

所以 P0 裡保留一個**只改一個變數**的從頭消融（見 P0-c），把問題留著開口。

### 名詞：sp_v6 是這個專案的 MagicPoint

| 上游 | 這裡 |
|---|---|
| MagicPoint（合成形狀）→ HA → COCO label | **sp_v6** → HA → WSI label |
| 在那些 label 上訓 SuperPoint | 在這些 label 上訓 SuperPathPoint |

**要保留的結論限制**：`gray_pre` 是「從 bootstrap detector 熱啟動」，所以它贏過
`gray` 只能說「跳過冷啟動有代價」，**不能說「預訓練有幫助」**。

---

## C. 優先順序

### P0 — 下一次重訓，六項一次改完

| | 改什麼 | 為什麼擋路 |
|---|---|---|
| **a** | augmentation 解凍：`rng = default_rng((seed, index, epoch))`，trainer 每個 epoch 呼叫 `set_epoch`，驗證集不呼叫 | 不解凍，加步數只會過擬合 —— 上一輪 `val/detector` 在 ep42 觸底 3.12 後回升到 3.27，而 train 一路降，那就是過擬合開始 |
| **b** | homography 只留旋轉：`perspective/scaling/translation = False` | 縮窄是安全方向（老師 13 個選項投過的票比學生被要求的多）。`PRE_TILE_FACTOR=3` 不用重抽：只有旋轉需要 1.66，全選項才 2.702 |
| **c** | 兩個 arm：`gray_pre`（batch 64 / 250 epoch = 20,750 步）+ `gray`（batch 128 / 50 epoch = 2,050 步，**只改 augmentation 的乾淨消融**） | `gray` 那個回答「解凍有沒有用」，不回答「從頭行不行」—— 步數沒變，別期待收斂 |
| **d** | `points_available`（過閾值、不設限、不配對的存活點**數量**）+ 驗證 budget **top-200** | 現有的 `points_per_view` 是 cap 之後的數，永遠顯示 420。`points_available` 是唯一能看出收斂的讀數，而且和 budget 從同一次 NMS 取出，成本接近零 |
| **e** | `dustbin_mean` + `hit_score_mean` | 「沒點的分數夠不夠低、有點的夠不夠高」的直接讀數。CE 把兩者混成一個數 |
| **f** | 逐 rung detector CE | 決定「要不要動 loss」的儀器 |

其他維持：`detection_threshold = 0.015`（對 pre 設得住）、LR 1e-4。驗證 budget 是
**top-200** —— 落在 label 語料自己的範圍內（逐 rung `n_kp` 平均 3 到 527，全體
146），而多個 budget 的 ladder 是 `cli/reeval_density.py` 的事，跑完之後做一次。

`max_keypoints` 的預設值**先不動** —— 換一個猜測沒有意義，等這一輪的
`points_available` 說話。但它的註解已經改正：420 的出處是 `BRACS_1228 ds4` 一個
rung，不是語料最大值（72 格的 `n_kp` 平均 3 到 527，最大 906）。

#### 三個讀數的預期，以及各自指向什麼

| 看到什麼 | 意思 | 下一步 |
|---|---|---|
| `dustbin_mean` 和 `hit_score_mean` 都慢慢上升 | 就是步數 | 繼續跑 |
| dustbin 上去、hit 卡住 | 學會說「沒有」，學不會定位 | 稀釋是真問題 → focal / 加權 |
| 兩個都不動 | LR 或最佳化 | 不是資料的問題 |
| 逐 rung CE 一起降 | 稀釋不是問題 | 不動 loss |
| 稠密降、稀疏卡住 | 稀釋是真的 | 才輪到 rung weight |

### P1 — Stage B（重訓在跑的時候寫）

用現有的 `gray_pre` checkpoint 當 detector。spec.md 1700 說 Stage B「需要一個堪用
的 detector」—— CE 0.19、repeatability 0.79，夠用。

**第一張存亡表當拋棄式的。** 目的是把 τ 校準完（`alpha` 起始 1.5，第一次跑當校準
跑不當結果）、把 `SurvivalTable` 的格式定下來、把 Stage C 的頭接上去。
`identity_id` 會讓過期的表自己說出來，所以重跑是安全的，而重跑是這條線上最便宜
的一步。

#### P1 的目標：三個決定，不是一張表

Stage B 存在的理由是**替 Stage C 的設計做決定**。三個：

| | 由哪個數字決定 | 決定什麼 |
|---|---|---|
| **A** | 連續帶比例 | Stage C 的頭能不能簡化成 `(j_lo, j_hi)` 兩個輸出。0.97 可以，0.6 不行（spec.md 3.3） |
| **B** | 晚生型比例 | 這個頭**有沒有東西可學**。接近零的話 Stage C 不用做 |
| **C** | 只在一階比例 | Stage 1 的 mpp 估計用不用得上尺度簽章 |

**「做完」的定義**：這三個數字各自附著一個**閾值敏感度**和一個**虛無基準線**。
沒有那兩樣，數字不能用來做決定。

前置目標：

| **D** | tau 的校準曲線 | A/B/C 全部在 tau 下游。tau 沒定，三個都是在報 1.5 |

#### 已經寫好的（2026-09-01，test_survival 18/18）

| 檔案 | 內容 |
|---|---|
| `PointsAnalysisByMpp/Patterns.py` | 六種樣態、`alive_from`、`band_fraction`。**純函式** |
| `PointsAnalysisByMpp/Attribution.py` | 四種歸因、`outranked`、`NONE` 哨兵。**純函式** |
| `PointsAnalysisByMpp/MppStack.py` | 讀 'F'、推導 'R'、`rung_scale` / `rung_shrink` |
| `PointsAnalysisByMpp/SurvivalTable.py` | safetensors store，無 `alive` 欄位 |
| `PointsAnalysisByMpp/SurvivalProcess.py` | `detect`（找位置）+ `probe`（量數值） |
| `cli/build_survival.py`、`jobscripts/.../BuildSurvival.sh` | |
| `test_modules/TestSuperPathPoint/test_survival.py` | 18 個，每個歸因分支各有誘餌 |

寫的過程中改變計畫的三件事，記在這裡因為它們都會被重新提出：

1. **'R' 軸不用抽取。** 一個 'R' rung 是 `tile` 個 level-0 px 縮小再放大，而 'F'
   鏈的 ds 1 那張**就是** `tile` 個 level-0 px。所以 'R' 從它推導。這不是省事：
   新生歸因要求兩軸講同一個實體點，兩次獨立抽取會各自按可容納性挑中心，事後只能
   空間 join，而那個容差正是分析要量的東西。`StackCentres` 這個部件因此不存在。
2. **`rung_scale` 和 `rung_shrink` 是兩個量。** 都從 `ds` 來，在 'F' 上相等、在
   'R' 上不等（1.0 對 `ds`）。用錯的話粗階的 'R' 點被撒到 `ds` 倍遠，表照樣填滿。
3. **偵測和量測要分開。** `score` 原本只在配對到的階有值，把「機率低」和「沒被
   偵測到」記成同一件事 —— 而那正是 (i)/(ii) 要分的。改成兩趟：`detect` 找哪些
   位置值得問，`probe` 對每個位置每一階量 `(score, offset, rival)`。

#### 要先修的設計缺陷

**`anchors_of` 用 tau 合併錨點，所以換 tau 錨點集就變 —— store 不是真的可以重切。**
這違背整個 store 的設計前提（不存 `alive`、事後重切）。

| | 現在 | 改成 |
|---|---|---|
| 合併半徑 | `tau[j]` | **固定 `nms_radius` 個 level-0 px** —— 那是「同一個位置」的定義，和跨階容差無關 |
| tau 在哪裡作用 | 建表時 | **只在讀表時**（`Patterns.alive_from` 的 `dist <= tau`） |

代價是錨點變多（tau 本來會合併的近似重複各自成列），那些改在讀的時候合併 —— 幾
毫秒，而且可以換 tau 重做。**這一項先做，否則每次換 tau 都要重跑 GPU。**

#### 還要寫的

| # | 檔案 | 職責 | 完成判準 |
|---|---|---|---|
| ① | `PointsAnalysisByMpp/Report.py` | 表 → 指標。`merge_anchors`、`pattern_table`、`attribution_table`、`cross_table`、`tau_curve`。**回傳資料結構，不 print 不 plot 不寫檔** | 每個函式能用手寫的小表測，不需要 GPU 或 store |
| ② | `PointsAnalysisByMpp/NullModel.py` | `null_patterns(p_per_rung)` → 六種樣態的期望比例 | 64 個機率加總為 1；全 `p=0.5` 對上手算 |
| ③ | `cli/inspect_survival.py` | **第一輪：校準**。配對率 vs tau 曲線（含平移誘餌）、`offset` 與 `score` 的逐階分布 | 曲線上有 knee，且 knee 在誘餌之上 |
| ④ | `cli/report_survival.py` | **第二輪：分析**。`patterns.csv`、`attribution.csv`、`cross.csv` + 三張圖 | A/B/C 各有敏感度、基準線、逐片 |
| ⑤ | `test_modules/TestSuperPathPoint/test_survival_report.py` | ①② 的測試 | `merge_anchors` 隨 tau 單調；tau=0 一個都不合併 |
| ⑥ | `jobscripts/SuperPathPointJobs/ReportSurvival.sh` | | |

**③ 不產出六種樣態和歸因。報了就是在報 1.5。**

#### 虛無基準線用窮舉，不用誘餌

L=6 只有 64 個存活向量。給定每一階**實測**的存活率 `p_j`，把 64 個向量的機率算
出來、按樣態加總，就得到「各階獨立擲硬幣的話六種樣態該是多少」。

它比誘餌好，因為它是**精確的**而不是抽樣的；而且它保留了實測的每階存活率，所以
「帶比隨機多多少」問的是結構而不是密度。

誘餌仍然留著 —— 用在**配對**上（平移超過 tau 之後還配得到多少），那個沒有封閉解。

#### 三張必要的圖，其餘是輔助

| 圖 | 回答什麼 |
|---|---|
| **存活矩陣熱圖** | 列=點（按樣態排序）、行=6 階、色=`score`。一眼看出帶不帶 |
| **閾值敏感度** | 三條線（連續帶／晚生型／只在一階）vs 閾值。**線陡就代表結論不成立** |
| **貼圖範例** | 每種樣態抽點，把它在各階的 tile 排成一列 |

第三張是**唯一能證偽「晚生型 = 腺體導管這類大結構」的東西** —— 前面所有數字都做
不到。挑點的規則要寫死在 docstring 裡：**固定隨機種子，每種樣態按 `score` 的
10/50/90 分位數各抽一個**。「作者挑的例子」和「規則挑的例子」在圖上長得一樣。

#### 放置

| 東西 | 位置 | 依據 |
|---|---|---|
| 純邏輯（`Report`、`NullModel`） | `PointsAnalysisByMpp/` | stage 模組：輸入資料+設定，輸出結構化的東西 |
| 兩個 CLI | `training/SuperPathPoint/cli/` | 驅動與診斷都在該套件的 `cli/` |
| 測試 | `test_modules/TestSuperPathPoint/` | 一個 jobscript 擁有一整組測試 |
| 輸出 | `/work/u26130998/result/<job>/` | 產出全部在 repo 外 |

**不進 `bench_modules/`** —— 那是端到端的品質量測，而 Stage B 量的是資料的性質。

#### P1 的順序

```
修 anchors_of（脫鉤 tau）        ← 先做
      ↓
① Report.py   ② NullModel.py     ← 純邏輯，語料還沒好就能寫能測
      ↓
⑤ test_survival_report.py
      ↓
③ inspect_survival.py  →  第一輪，定 tau
      ↓
④ report_survival.py   →  第二輪，A/B/C
```

**整條卡在語料**：現在 0 條 chain，要先重抽（開 `InheritConfig`）+ 重跑 HA label。
純邏輯那三項不受影響。

### P2 — Stage C

硬依賴：label 就是 B 的輸出。

**P1 + P2 是「先鋒隊」—— 把 SuperPathPoint 的整體框架打通。框架的價值不取決於
detector 有多好。**

### P3 — encoder 搜尋：先 CNN，再 ViT

理由是 stride：

| | patch / stride | cell = 8 |
|---|---|---|
| CNN（VGG / ResNet / MobileNet / ConvNeXt / NFNet） | 可設計成 8 | 直接接 |
| ViT patch-16（`gigapath`, `conch_vit`） | 16 | 要多一層**隨機的** upsample 堆疊 |
| ViT patch-14（`uni2`） | 14 | **接不了**，256 不是 14 的倍數 |

ViT 那條多一個要學的隨機模組，結果會混進「upsample 學不學得起來」。
**CNN 先跑，它是乾淨的對照。**

**trunk 一律先凍住。** 那是最便宜的版本，直接回答「特徵是不是瓶頸」。凍住撐不
起來，fine-tune 也救不回 —— 至少不是用負擔得起的步數。

不要用 HEST：它是組織/玻璃**分割**模型（DeepLabV3-ResNet50），被訓練成對組織
內部細節不敏感，而那恰好是 keypoint 唯一要的東西。

**還有一個限制不會因為換 encoder 而消失**：label 是 sp_v6 挑的點，所以任何學生的
天花板都是「同意 sp_v6」，不是「找到好點」。

### P4 — soft label（teacher-student）

先講清楚：**我們現在已經是 teacher-student** —— sp_v6 是 teacher，HA label 是它的
輸出。

要做的是把**硬標籤換成軟標籤**：

| | 現在 | 軟標籤 |
|---|---|---|
| detector 目標 | 一個 cell 裡「哪個位置有點」的 one-hot 整數 | 老師那張完整機率圖，KL 散度 |
| 每個 cell 的梯度 | 只有 argmax 那一個 | 65 個都有 |

**這是偏離上游，不是修正上游。** 上游 SuperPoint 的 detector loss 就是硬標籤的
sparse cross-entropy（`F.cross_entropy(cell_logits, labels)`），descriptor 是稠密
hinge。軟標籤是實驗。

代價：`KeypointLabelStore` 只存**點**不存圖，真做要重跑 HA 並存機率圖 —— 44 GB
等級的問題。

**中間版本比較便宜**：`kp_score` 已經存了。把 one-hot 換成「用該點的分數當目標
信心」，不用重跑 HA 就能拿到一部分好處。

---

## 順序圖

    P0 重訓（gray_pre 250ep + gray 消融 50ep）
          │
          ├──並行──> P1 Stage B ──> P2 Stage C     ← 框架先鋒隊
          │
          └──之後──> P3 CNN encoder（凍住）──> ViT
                            │
                            └──> P4 soft label

---

## 待處理的舊帳

- ~~`train_superpathpoint.py` 的 `val_slides` 逗號 join~~ —— 2026-08-31 修掉。寫端
  改 `json.dumps`，讀端先試 JSON、失敗才走 store 重建（08-31 那批 checkpoint 是舊
  格式，而它們是 Stage B 目前唯一能用的 detector）。`extra_identity` 不進
  `identity_id`，所以沒有重新雜湊。
  **並且把逗號放進 `test_superpathpoint.py` 的預設 fixture**（`_STEM_B =
  'S1103627,G7E,110127'`），讓「會不會有人記得」變成「測試會不會過」。
- **stem 正規化**（逗號 → 底線），從源頭終結這個類別。`wsi_stem = Path(wsi_path).stem`
  而檔名帶逗號，這已經咬過兩次不同的工具（awk、`val_slides`）。**現在不做**：179 處
  引用、stem 是目錄名的一部分、而且進了 `PreTileMeta` / `LabelMeta` / `StoreMeta` 的
  雜湊 —— 等於整批 cache 作廢重建。等下次有理由重建 cache 時一起。
- `utilities/test_modules/test_config_identity.py` 沒有任何 jobscript 在跑它。
- spec.md 還有四段過時：`tile_size`/`tissue_ratio 0.75` 那節、`17,784 / 14.2 GB`
  的落地大小表、「`tissue_ratio` 套在 tile 的 footprint 上」、「探針要回答的三件
  事」；split 那節還寫 6 片而程式是 12 片。
- `log/TODO.log` 要補 reference bank 的純玻璃表（刪掉 44 GB 之前要留的五個數字）。
- ReferenceSampler 退役進 TileSampler：bucket 換成新的七個、jitter 換成
  `OverlapConfig` 的比例、`over` 不帶過去、`ConfigIdentity` 用自我驗證的改名腳本
  遷移。
