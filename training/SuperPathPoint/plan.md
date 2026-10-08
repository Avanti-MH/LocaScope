# SuperPathPoint — 計畫

`spec.md` 是規格：名詞、每個階段的程序、code 的 interface、code 對應哪個階段的哪個
步驟。這份是**現在要做什麼、按什麼順序、做到哪**。決定的理由留在這裡，量測的結論
進 `log/TODO.log`。

**三個階段，這是現況（2026-09-03 跟使用者對過）：**

```
1. 訓自己的 wsi superpoint
   1.1 先模仿上游 backbone + detector + descriptor
   1.2 換 backbone
2. 分析六種 survival 樣態、相鄰響應、新生死亡歸因
   2.1 建構語料：F/R/C 三軸的 stack 要先能建構起來（unit test → demo smoke
       test → 正式建語料，順序寫死，2026-09-05 定案）
   2.2 分析六種 survival 樣態（tau、alpha）—— 語料建好才做
   2.3 相鄰響應 + 相依型 keypoint 比對 —— 2.2 有數字才做，不是現在
   2.4 新生死亡歸因
   2.5 再決定要訓練哪一種語意 —— 還沒決定，等 2.2-2.4 有數字再選
3. 訓練有語意的 keypoint
```

下面每一節底下都標了它屬於哪一步。

---

## 1. 訓自己的 wsi superpoint

### 1.1 先模仿上游 backbone + detector + descriptor

#### 現況：ReEvalSuperPathPoint 的讀數

四個 arm 的 checkpoint（2026-08-31，50 epoch、batch 128、2,050 步）在**對齊過的
點數**下重新評分。對齊的方式是 `score_threshold=0` + `max_points=N`，也就是取分數
最高的 N 個，密度被構造性地釘死，`decoy` 因此對每個 arm 是同一個量。

**欄位**

| 欄位 | 意思 |
|---|---|
| `repeat` | 兩個 view 的點對得上的比例。越高越好 |
| `decoy` | 把一組整體平移超過 NMS 半徑之後還「對得上」的比例 = **光靠密度就能拿到的分數** |
| `unif` | N 個點均勻散開時 `decoy` 的理論值，`1 - exp(-81N/256²)` |
| `margin` | `repeat / decoy` —— 扣掉密度紅利之後剩下的，**這才是分數** |
| `ceiling` | `1/decoy`，`margin` 的上限。`repeat ≤ 1` 所以 decoy 就是天花板 |

**A.1 有一個乾淨的交叉**

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

**A.2 RGB 不是空結果 —— 先前的判斷被推翻**

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

**A.3 點是聚集的，而且聚集程度隨 N 變**

| N | `unif` | gray | gray_pre |
|---|---|---|---|
| 40 | 0.048 | 0.084 (1.8x) | 0.095 (2.0x) |
| 160 | 0.179 | 0.210 (1.2x) | 0.228 (1.3x) |
| 420 | 0.405 | 0.442 (1.1x) | 0.419 (1.0x) |

小 N 兩個都明顯聚集（點集中在組織區而非散在整張 tile，合理）。大 N 時 pre 反而
比 gray 散。

**A.4 從頭那組為什麼點數會飆到 NMS 的幾何上限**

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

一句話：**它不是想要更多點，它是還沒學會拒絕。**

**A.5 從頭那組連「不看圖」的水準都還沒到**

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

#### 方向：先做 pretrain，但不關掉另一條路

**決定：接受。理由：保留。**

資料支持的是「在我們在乎的密度（N ≥ 160）pre 比較好，而且兩片都贏」。

資料**不**支持「隨機初始化是錯誤方向」—— 那個 arm 沒有被公平跑過：2,050 步、
augmentation 凍住、CE 3.27。**用一個沒跑完的實驗否定一條路，是這個專案自己定的
規矩要避免的事（ClaudeRules §8：先量再校準）。**

但決定不需要那個強命題。先做 pre 的三個理由都成立：

1. 它現在就比較好
2. 它現在就能用（CE 0.19，足以當 Stage B 的 detector）
3. 它便宜 —— 不用等收斂



**名詞：sp_v6 是這個專案的 MagicPoint**

| 上游 | 這裡 |
|---|---|
| MagicPoint（合成形狀）→ HA → COCO label | **sp_v6** → HA → WSI label |
| 在那些 label 上訓 SuperPoint | 在這些 label 上訓 SuperPathPoint |

**要保留的結論限制**：`gray_pre` 是「從 bootstrap detector 熱啟動」，所以它贏過
`gray` 只能說「跳過冷啟動有代價」，**不能說「預訓練有幫助」**。

#### 下一次重訓，六項一次改完

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

**三個讀數的預期，以及各自指向什麼**

| 看到什麼 | 意思 | 下一步 |
|---|---|---|
| `dustbin_mean` 和 `hit_score_mean` 都慢慢上升 | 就是步數 | 繼續跑 |
| dustbin 上去、hit 卡住 | 學會說「沒有」，學不會定位 | 稀釋是真問題 → focal / 加權 |
| 兩個都不動 | LR 或最佳化 | 不是資料的問題 |
| 逐 rung CE 一起降 | 稀釋不是問題 | 不動 loss |
| 稠密降、稀疏卡住 | 稀釋是真的 | 才輪到 rung weight |

### 1.2 換 backbone

#### encoder 搜尋：先 CNN，再 ViT

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

換成 pretrain 在 WSI 的 encoder 會撞到的六件事，已經想過一輪寫進 `spec.md` §13
「換成 pretrain 在 WSI 的 encoder」——那是介面/程序層級的風險清單，屬於 spec，不
重複列在這裡。

---

## 2. 分析六種 survival 樣態、相鄰響應、新生死亡歸因

**Stage B與 Stage C是「先鋒隊」—— 把 SuperPathPoint 的整體框架打通。
框架的價值不取決於 detector 有多好，可以在 1.1 的重訓跑的時候並行寫。**

### 2.1 建構語料：F/R/C 三軸的 stack 要先能建構起來

**這是 2.2 以下全部的前提，順序寫死在這裡，不要跳著做（2026-09-05 定案）：**

```
① unit test（純幾何 + 本地快取，不用真的語料、不用 GPU）
      ↓
② demo_survival_analysis.py（chains_stack 部分）當 smoke real test（對著一片真的 WSI 跑，不是只驗證合成座標）
      ↓
③ 正式把語料建出來（F/R/C 三種形式都要能建）
      ↓
   （語料建好才進 2.2——分析 tau/alpha/survival；相依型比對邏輯更晚，見 2.3）
```

**① 做完（2026-09-06）**：`FStack`/`RStack`/`CStack` 三個 class 都在
`SurvivalAnalysis/ChainStack.py`，各自拆「純幾何」（`footprint`/`pyramid`/
`mother`/`nearest`）跟「IO」（`read`/`derive`/`read_one`/`read_tree`）。

| 檔案 | 內容 |
|---|---|
| `SurvivalAnalysis/ChainStack.py` | 見下 |
| `test_modules/TestSuperPathPoint/test_chain_stack.py` | 21 個測試：幾何/本地快取（原本在 `test_survival.py` 的部分）+ `own`（`from_tile`/`base_rung`/三個 `from_own`，都對著假 corpus fixture：draw 給定、像素用畫的） |
| `jobscripts/SuperPathPointJobs/TestSuperPathPoint.sh` | `chain-stack` stage |
| `cli/prepare_chain_stack.py` / `jobscripts/.../PrepareChainStack.sh` | 一條龍入口，見下 |

**三軸的原料怎麼來：**

- **F**：唯一需要「繼承」的軸，本質上也是一種 own（`inherit.share=1.0`，R/C
  的 own 是 `share=0`）。`FStack.read(chain)` 是 store-backed，需要一條真的
  chain（own 語料 `stageB-fOwn`，12 片，還沒抽）。
- **R**：不需要一整條 chain，只要一張真實 tile 當底，三選一：
  1. `source='F'`——重用 F 的某一階，`base_rung` 參數決定哪一階（不侷限 ds=1）
  2. `source='C'`——重用 C 的母 tile 或任一子嗣，同一個 `base_rung` 機制
  3. `own`——直接指向既有的 `stageA`（2026-08-27，獨立多階 `share=0` 的真實
     tile，正是 R 需要的東西），不另外抽

  `base_rung` 的降解語意：`degrade_resolution` 只在 `ds<=1.0` 跳過降解，
  `base_rung=B` 時每一階 `1<X<=B` 都是在已經模糊的底上**再**跑一次絕對 `ds`
  的降解——比 `base_rung=1` 更模糊，不是「差不多模糊」，確認過是刻意的。
  `footprint(..., base_rung=...)` 要跟著回報真正的窗口大小。（`SurvivalMeta`
  還沒有 `base_rung` 欄位，等真的有非 1.0 的呼叫端再補。）
- **C**：子嗣純幾何算出來（`CStack.pyramid`），不需要 corpus/inherit。母 tile 兩個來源：重用 F 讀過的同中心同階，或 own（`stageB-cOwn`，
  單一階、只抽 5 張，還沒抽）。

**own 的兩層架構**：`RStack.from_tile(image, base_rung, rungs, tile=...)` 是
三個來源共用的底層原語——一張已讀進來的圖 + 它自己的 ds → 一個 RStack，純函式，
不吃 `Chain`/`wsi`/store。`derive(chain, ..., source=...)` 是 F/C 來源的單條
chain 便利包裝，內部呼叫 `from_tile`；**`source='own'` 故意不是這裡的分支**——
own 的 tile 是 corpus 裡獨立的一個位置，沒有 `Chain` 可餵，傳
`source='own'` 進 `derive` 直接 `ValueError`。

列舉層 `from_own`（`FStack`/`RStack`/`CStack` 各自一個，留在 `ChainStack.py`
裡，不是集中放進 `prepare_chain_stack.py`——三軸的 own store 長相不同，列舉
邏輯跟著各自的 class 走）都接受 `sampler_id`，都是 LAZY（仿 `TileSampler.Sample`：
metadata 一直都在，pixel 只有 `__getitem__` 才讀）：

- `FStack.from_own` → `OwnChains`，`x[inherit_id]` 呼叫 `FStack.read`
- `RStack.from_own` → `OwnTiles`，位置索引（own 的批次可能橫跨好幾階），`x[i]`
  用 `Corpus.read` 讀 pre-tile、裁中心，再呼叫 `from_tile`。`cache_root` 預設關——
  `degrade_resolution` 只是記憶體裡的 resize，不像 WSI 讀取那麼貴，大規模跑
  的時候開它只是白花磁碟 IO，只有 demo/小量重複讀才需要
- `CStack.from_own` → `OwnForest`，跟 F/R 不一樣的地方：整片森林的幾何在
  `from_own()` 當下就全部建好，`x[i]` 才讀像素——母 tile 直接是 corpus 這個
  位置的像素，子嗣仍要 `wsi`（子嗣的位置是算出來的，不在任何 draw 裡）。回傳
  `(mother, mother_image, groups_by_ds, images_by_ds)`，母子的圖都在

其他順手做的：`CStack.read` 改名 `read_tree`（跟 `FStack.read`/`RStack.derive`
同名不同形狀的問題，`OwnForest.__getitem__` 開始呼叫它之後不再是死碼）；
讀 pre-tile + `centre_crop` 抽成共用的 `_read_store_tile`（原本在
`FStack.read`/`OwnTiles`/`OwnForest` 三處各寫一次）；`chains()`/
`Datasets.py`（`HomographyPairDataset.build()`）都加了可選的 `sampler_id`
參數，`result/cache/tiles/` 現在一個根目錄裝下 stageA 跟全部 own 語料，不用
`tiles_chains` 這種另開目錄的方式分開。

**`cli/prepare_chain_stack.py`**：三軸各自的 corpus 從 `Corpora.RECIPES`
（`AXIS_RECIPE`）算出來，不猜磁碟上哪個屬於誰。corpus 就是一個 draw：第一次讀
那張 slide 時由 `TileSampler.cached` 抽（mask 走 `MaskMaker`），之後讀回；像素
在 `from_own(...)[i]` 時才讀。R 讀 `stageA`。

寫的過程中抓到的真的 bug（供以後參考）：`OwnTiles.__getitem__` 一度少了
`centre_crop`（會把整張 pre-tile 硬縮成 tile，不是裁中心那塊）；重構
`extract_pretiles.py` 的關鍵字參數時漏改一行 `args.n`（`_write_rung` 最後的
print，runtime `NameError`）；第一版 `base_rung` 測試餵了同一張沒模糊過的圖
兩次，驗證不到任何東西，改成先把圖真的降到目標 ds 畫質再餵。

**② 做完（2026-09-06）**：`_pick_record`（`find_one` 不傳 `sampler_id`，一旦
`stageB-fOwn` 抽出來會因兩個 store 都符合 ds=1 而報錯）整個拿掉，改成跟
`prepare_chain_stack.py` 一樣直接算 `sampler_id`
（`_sampler_config_for('stageB-fOwn'/'stageA'/'stageB-cOwn', tile).sampler_id()`），
不猜、不查磁碟內容——上面那條風險現在不存在了。

demo 現在跑五條路徑，每條都是真正的類別入口，不是 workaround：

| 路徑 | 入口 |
|---|---|
| F | `FStack.from_own` |
| R own | `RStack.from_own` |
| R reuse-F | `RStack.derive(chain, source='F')` |
| C own | `CStack.from_own` |
| C reuse-F | `CStack.from_mother`（母 tile 吃 `FStack.read(chain)` 的 ds16，不重讀） |

own 跟 reuse-F 對 R/C 而言中心點本來就不同（own 是獨立抽樣落點，reuse-F 是 F
own chain 的中心），兩條路徑不互比，但都各自完整跑完＋各自留下圖，不再是「reuse-F
畫圖、own 只印數字就丟」——六張圖：`r_stack_{own,reuseF}.png`、
`pyramid_{lineage,overview}_{own,reuseF}.png`。C 的 mother-crop vs 真子嗣的
decoy 相關性檢定（幾何/像素一致性，不是 own vs reuse-F 互比）現在兩棵樹都做。

**真的跑過了（2026-09-06，`AXES=F R C` 之後，`BRACS_1228`，exit 0）**：五條
路徑全部對著真實 slide 跑完。`F own` chain 0 六階 99.3ms；`R own`/`R reuse-F`
都 OK；`C own`/`C reuse-F` 的 mother-crop vs 真子嗣 decoy 檢定都是 real 明顯贏
decoy（0.902 vs 0.104、0.923 vs 0.020）。六張圖都產出。

**③ 做完（2026-09-06，`BRACS_1228`，`AXES=F R C` 一次跑完，exit 0）**：三軸都
真的建出語料了。

| 軸 | corpus | sampler_id | 結果 |
|---|---|---|---|
| F | `stageB-fOwn`（現抽） | `578e0d1b` | 1095 tiles / 6 階，145 條 chain 湊到 inherit，其中 84 條六階都齊（完整） |
| R | `stageA`（既有，直接命中未現抽） | `d4366c49` | 543 筆 own tiles |
| C | `stageB-cOwn`（現抽） | `42e55094` | 5 棵樹，母 tile ds 16 |

「三軸都能建出語料」這句話現在成立。下一步是 2.2（分析六種 survival 樣態）；
目前只在 `BRACS_1228` 一片 slide 上跑過，`stageB-fOwn`/`stageB-cOwn` 真正的
12 片語料還沒批次抽——這是 2.2 開工前的最後一件事。

`_RECIPES`（F/R/C 唯一的 `SamplerConfig` 定義）跟 `--rungs` 預設之後又動過，上表的
`sampler_id`/`578e0d1b`/`d4366c49` 已經不是現在的值；12 片批次會用當時的
`_RECIPES`/`--rungs` 統一重抽。

### 2.2 分析六種 survival 樣態

**目標：六種樣態分類 + 歸因，前提是 alpha 先定案——tau 沒定，下游每個比例都是
在報 1.5。這一節先只展開 alpha 校準這一步，其餘（正式建表、樣態統計、報告
圖）等 alpha 定案後再寫。**

**① alpha 校準（2026-09-06 定案）**：核心是「F/R/C stack → 存活分析 → 歸因」
本身；alpha 校準是為了知道 tau 該多寬另外做的支線分析，兩者分屬不同抽象層
次，分成兩個檔案。`[新]` = 還沒寫，`[留]` = 沿用現有的：

```
設計
│
├── 0. 桶(bucket)= F/R/C
│       cli/survival_alpha_analysis.py:main()，--alive-method 選 alive 判準
│         (baseline/exp_decay/probability_map/scale_extremum，見下方分岔)
│
├── 1. 每個 ChainStack
│   │
│   ├── [C 專屬子流程]
│   │   SurvivalAnalysis/SurvivalProcess.py:
│   │     anchors_of_generations(per_rung_tiles, order, tile_merge_radius, cross_rung_base)   [新]
│   │     _merge_within_radius(points, radius) -> keep_idx        (私有,anchors_of/
│   │                                              anchors_of_generations/
│   │                                              AlphaCalibration.merge_anchors 共用)
│   │       演算法:網格雜湊合併法——依 priority(高分先)排序訪問,一個點只跟
│   │       它自己 radius 網格格的 3x3 鄰域裡「已保留」的點比,不是全體已保留
│   │       點,倖存者是同一堆裡最先被訪問到的那個                     [新]
│   │
│   ├── 建錨點清單(核心,不管有沒有要校準 alpha 都要做)
│   │     F/R:  SurvivalProcess.py:anchors_of(...)                 [留]
│   │     C:    SurvivalProcess.py:anchors_of_generations(...)     [新]
│   │
│   ├── alive 判準:兩條資料路徑,由 --alive-method 二選一(2026-09-11 定案,
│   │   AlphaSelectionNotes.md §14 有候選 1-4 完整取捨)
│   │   │
│   │   ├── 路徑 A -- (score, dist, tau) 判準:baseline / exp_decay
│   │   │   對每一階:真實探測(核心)
│   │   │     SurvivalProcess.py:
│   │   │       detect / nearest_detection / rival_at                 [留]
│   │   │       演算法(nearest_detection):scipy.spatial.cKDTree,精確全域
│   │   │       最近鄰——2026-09-11 從手刻網格雜湊環狀搜尋換過來，因為後者
│   │   │       per-query 成本隨「離點雲多遠」增長，C 軸粗階小而密的偵測點
│   │   │       雲被跨全樹的錨點探測時會卡住(production 實測 20-40 分鐘
│   │   │       still running)；先前更早的版本是 O(N*M) 全對全距離矩陣，
│   │   │       在 C 軸 ds=1 一階上會 OOM(~70,000 點雙邊時要價 ~78 GB)
│   │   │       probe_real(anchors, per_rung_detections)
│   │   │         -> dist, score, rival                                [新,取代 run() 的真實那半]
│   │   │   對每一階:誘餌探測(支線,只有校準 alpha 才做這步)
│   │   │     AlphaCalibration.py:
│   │   │       probe_decoy(anchors, per_rung_detections, decoy_shift)
│   │   │         -> decoy_dist, decoy_score
│   │   │         (內部呼叫 SurvivalProcess.nearest_detection,decoy_shift 決定 shifted 座標)  [新]
│   │   │     cli/survival_alpha_analysis.py:
│   │   │       decoy_shift_fixed(offset_xy) -> Callable
│   │   │       decoy_shift_random(min_mag, max_mag, rng) -> Callable
│   │   │       decoy_shift_rotate(angle_range, rng) -> Callable       [新,傳進 probe_decoy]
│   │   │       _make_alive_fn(method)：'baseline' -> None（沿用 alpha_curve/
│   │   │         pattern_curve 自帶的 alive_absolute_dual_threshold 快速路徑）；
│   │   │         'exp_decay' -> 包一層呼叫 AliveCandidates.
│   │   │         alive_exp_decay_joint_score                          [新]
│   │   │   -> 吃 AlphaCalibration.py:alpha_curve(dist, score, decoy_dist,
│   │   │        decoy_score, *, rungs, alphas, tau_floor, threshold, alive_fn)
│   │   │
│   │   └── 路徑 B -- 機率圖判準:probability_map(候選1) / scale_extremum(候選2)
│   │       不吃 dist/score，直接重探機率場；候選4(雙成分混合模型)函式已寫出
│   │       但還沒接進這裡，見 AlphaSelectionNotes.md §14
│   │         SurvivalAnalysis/AliveCandidates.py:
│   │           assemble_generation_map(...) -> combined_map            [新]
│   │           probe_via_probability_map(combined_map, anchors, ...)
│   │             -> peak_value, peak_dist                              [新]
│   │           alive_probability_map(peak_value, peak_dist, ...)   候選1  [新]
│   │           alive_scale_local_extremum(peak_value, peak_dist, ...) 候選2 [新]
│   │         AlphaCalibration.py:
│   │           _probe_positions(anchors, order, decoy_shift)
│   │             -> real_by_rung, decoy_by_rung(每階各自的 decoy 位移)     [新]
│   │           _probability_map_alive(combined_maps, anchors_by_rung, *, kind=...)
│   │             -> alive[N, L]，kind 切候選1/候選2                       [新]
│   │         cli/survival_alpha_analysis.py 每軸各多一份 worker(跟原本
│   │         _one_f_chain/_one_r_tile/_one_c_tree 並存,不是取代):
│   │           _one_f_chain_probability_map / _one_r_tile_probability_map /
│   │           _one_c_tree_probability_map                               [新]
│   │       -> 吃 AlphaCalibration.py:probability_map_curve(combined_maps,
│   │            anchors, decoy_shift, *, kind=...)——輸出 shape/keys 跟
│   │            alpha_curve 完全一樣，下游彙總/畫圖不用分流
│   │
│   ├── merge_radius_2nd(迴圈外,一次,支線專用旋鈕,兩條路徑共用)
│   │     AlphaCalibration.py:merge_anchors(anchors, merge_radius_2nd)
│   │       (內部呼叫 SurvivalProcess._merge_within_radius)        [新,從 Report.py 搬過來]
│   │
│   └── alphas_sweep + tau_floor(支線,兩條路徑各自算,輸出同一種 shape)
│         路徑 A：AlphaCalibration.py:alpha_curve(dist, score, decoy_dist, decoy_score, *,
│                                        rungs, alphas, tau_floor, threshold, alive_fn)
│         路徑 B：AlphaCalibration.py:probability_map_curve(...)
│           -> 這個 ChainStack 的 match_rate/decoy_rate/gap/margin,[L, len(alphas)]   [新]
│
├── 2. 桶內彙總(支線,兩條路徑共用同一個彙總函式)
│       AlphaCalibration.py:aggregate_curves(list_of_每ChainStack結果)
│         -> 桶內平均矩陣 + 標準差矩陣(margin 取 log 再平均)         [新]
│
├── 3. 圖:1D 三格(match+decoy、gap、margin 對 alpha)
│       cli/survival_alpha_analysis.py:_plot_alpha_curves(...)      [新]
│
└── 4. offset_quantiles 等價物(支線,只有路徑 A 有 dist 可用;路徑 B 目前沒有
    等價的 offset 診斷)
        AlphaCalibration.py:offset_quantiles_of(dist, *, rungs, quantiles)
          (彙總邏輯重用 aggregate_curves)                          [新]
        cli/survival_alpha_analysis.py:_plot_heatmaps(...)
          gap/margin：y=ds x=alpha 深淺圖
          offset_quantiles：offset_quantiles 不吃 alpha，畫成「ds 對分位數值」
            的曲線圖，不是熱圖                                     [新]
```

其餘核心檔案（`ChainStack.py`/`SurvivalTable.py`/`Patterns.py`/`Attribution.py`/
`Report.py`/`NullModel.py`）不受這次重寫影響，見「alpha 定案之後才要做的事」。
`SurvivalProcess.py` 不知道「誘餌」這個概念——誘餌只有校準 alpha 才需要，alpha
定案後核心流程只用 `probe_real`。

**C 軸的錨點怎麼建**：每一世代（=每一階）先把該世代所有 tile（main + overlap）
的偵測合併出該世代的共識錨點（overlap 一律併入，`overlap_mode='intersection'`
2026-09-11 移除——曾經是個死分支，寫測試時抓到的，overlap 收進來的點必然會被
收尾合併判成重複而刪掉，從來沒有真的貢獻過任何點）；世代之間再聯集成跨世代
錨點清單，之後每一階都用同一份清單探測。

**F 軸不用**：footprint 隨階數變大（`tile × ds`），粗階邊緣的 anchor 在細階根
本沒被讀過，`dist` 永遠 `NONE`（spec.md 327-332）——這正是 C 軸存在的理由，不
是新發現。

**其他釐清（供以後參考）**：

- `match_rate` 分母是錨點數，不是這一階自己的偵測點數。
- 誘餌位移的是錨點座標，不是這一階的偵測清單；`decoy_shift` 是
  `callable(anchors) -> shifted`，可換固定方向/隨機/繞中心旋轉。
- `gap = match_rate - decoy_rate` 本來就有號，不用 `abs()`。
- `nms_radius` 沿用偵測器自己的 NMS 設定（`KeypointNetConfig.nms_radius=4`），
  不是另外選的——這句對 Step A/B（`tile_merge_radius`，同 ds 跨 tile）跟 R 軸的
  跨 rung 合併仍然成立；F/C 軸的跨 rung 合併（`cross_rung_base`）2026-09-11 改成
  0，理由見 spec.md「同一個點的定義」。
- `merge_radius_2nd` 是合併半徑的敏感度檢查，不是要調到某個「對」的值。
- `margin` 是乘性量，跨 ChainStack 平均前先取 log；`match_rate`/`decoy_rate`/
  `gap` 有界，不取 log。

**alpha 定案之後才要做的事，先不展開**：建正式的六樣態分類 + 歸因表、
`NullModel`/`Report.py` 的樣態統計、三張報告圖。

**alive 用 `probability_map` 候選法時，alpha ~ 2.5。**

**production（baseline）方法：alpha ≈ 3。** 理由：

- 落在 C 軸兩種收斂讀法的 bracket（2.5–3.5）裡，也落在 R 軸的 bracket（1.5–4.5）
  裡——兩軸交集的中心值。
- 跟 probability_map 候選已經定的 2.5 同一個數量級，兩個獨立方法互相印證，不是
  各說各話。
- 對 rotate/C 那個異常值保持謹慎：如果你在意 C 軸粗階（ds=16 那種）的可靠度多
  一點，可以考慮往 3.5–4.5 靠；如果你比較在意細階不要被稀釋，3 附近的原值就夠。

**TODO：統計「真實 offset 分布」的方法還沒實作，以後有空再討論怎麼做。**
2026-09-13 討論過一個方向——`offset_quantiles_of` 再對 `decoy_dist`（誘餌，
`probe_decoy` 早就算好，CSV 目前沒有這欄）也算一次分位數，兩條曲線疊在一起
看某個 rung 的真錨點 offset 像訊號還是像背景。**但這只是軟性診斷，跟
excess/NullModel 同一個等級**——它是分布層級的比較，沒辦法逐點判斷哪一筆是
真配對、哪一筆是雜訊，所以不會產生新的硬性規則，也不會讓「不能拿 p90 offset
反推 alpha」（第 9 節的結論）變得可以用。真的要修，得先想清楚這張圖到底要
回答什麼問題，不是現在就動手接。

### 2.3 相鄰響應 + 相依型 keypoint 比對

**在 2.2 有 tau/alpha/六種樣態的數字之後才開始寫這一節——不是現在，2026-09-05
跟使用者對過的順序。** 理由：相依型比對本身也要吃 tau（同一個位置算不算「配對
上」），tau 沒校準之前寫這裡的比對邏輯，跟 2.2 沒校準前就分析六種樣態是同一種錯誤。

`spec.md` §3.2「第三個軸：C（子嗣／組合 stack）」已經把切法定案——沿用
`utilities/PatchingLib.py` 的 `PatchGrid.from_size(..., overlap=True)`，main 格
精確不重疊密鋪，overlap 格是內角格，跟周圍 4 個 main 格各共用 1/4 面積。子嗣的
幾何跟抽取本身已經在 2.1 做完（`ChainStack.CStack`），這裡要做的只剩比對邏輯：

**這一節能不能證偽 spec.md §3.2「(i) 尺度結構 / (ii) 上下文」現有靠 NMS 分支的
判準**——同一階、只換周圍框住的組織，理論上乾淨地只測得到 (ii)。下面的設計
就是在回答這個問題。

**主程式（2026-09-13 定案）**：`training/SuperPathPoint/cli/
dependent_keypoint_analysis.py`——跟 `demo_survival_analysis.py` 一樣，可以
只分析一個 case、也可以吃很多 case 統計。**不另外開 `SurvivalAnalysis/` 模
組**——跟 `AlphaCalibration.py` 那種有大量可重用純函式的情況不同，這裡看不
出有分出去的必要，比對邏輯直接寫在 CLI 檔案裡就好。

**單一 case 的呈現（demo 模式，一張 keypoint 圖 + 一張指標圖）**：一組 4 張
main tile + 1 張 overlap tile 稱一個 case。用 4 張 main tile 拼出母 tile；5
張 sub-tile 的 keypoint 中，挑出落在 overlap 象限範圍內的——**overlap 的一個
象限只跟它對應的那一個 main tile 比，不是跟 4 張全部比**（`PatchGrid` 的內角
格跟周圍每個 main 格只共用 1/4 面積，見上）。keypoint 圖：黑 X = 兩邊都測
到，紅 = 只有 overlap 測到，綠 = 只有 main 測到。

**「同一個點」的容忍度：1 個 tile 輸出解析度的 px**（這一階等於 `ds` 個
level-0 px，換算成 level-0 座標時要乘 `ds`，跟 `tile_merge_radius`/
`nms_radius` 現有的 level-0 單位慣例一致）——不是 2.2 的 `tau=alpha*ds`：
overlap/main 是同一個 ds，沒有降採樣量化誤差要容忍，同一個真實特徵理論上要
落在幾乎完全相同的位置，容忍度該遠比跨 rung 的 tau 緊。

**兩段式統計/分析（2026-09-13 定案）**：

1. **`mismatch_rate`（存在跟周圍資訊有沒有關係的直接量測，主要指標）**——
   `mismatch_rate = (紅+綠) / (黑+紅+綠)`，`red_rate`/`green_rate` 分開報，
   不只報一個合併數字（紅綠不對稱本身可能是訊號，例如 overlap 格更靠內角、
   跟 main 格的邊界效應不對稱）。逐階（逐 ds）分開算；彙總沿用
   `aggregate_curves` 的既有慣例——每個 case 先算完自己的 rate，再跨 case 取
   mean/std，不把所有 case 的點混在一起算一個全域比例。同一階出現紅/綠點，
   本身就是「上下文（而非尺度結構）影響了 keypoint 存不存在」的直接證據，不
   需要額外證明。
2. **descriptor 相似度分析（這些紅/綠點是「有意義、受周圍內容影響」還是
   「純雜訊/detector 邊界抖動」，次要、補強指標，回報有意義/雜訊各佔的比
   例）**——對每個紅/綠點，拿它的 descriptor，跟 (a) 它對應的那一個 main
   tile 全部的點算相似度、(b) 其他 3 個象限的點算相似度，各自組成一個分
   布（一個點對很多點 = 一個分布），一個 case 裡同一類（紅/綠）的點再 pool
   成一個 case 級的正/負分布。用 AUC 量兩組分布可不可分（Mann-Whitney U，
   不需要密度估計，小樣本穩）——**負類（其他象限）取樣時要用距離配對取樣
   法**：只抽跟正類集合距離範圍相近的點，讓兩組的空間距離分布先對齊，避免
   「只是離得近所以像」這個混淆解釋掉分開的結果。AUC 高（可分）→ 判定這批
   紅/綠點「有意義，受周圍內容影響」；AUC 低（分不開，像隨機）→「雜訊/邊
   界抖動」。
   （2026-09-13 考慮過、否決的做法：改用「同一位置、兩種裁切各自的 dense
   feature 直接比」——detector head 跟 descriptor head 共用 backbone，紅綠
   點的定義本身就代表兩邊 feature 已經不同，這樣比接近套套邏輯，沒有帶來新
   的獨立證據，所以維持「跟其他點比」這個間接但非循環的設計。）

**先決條件（2026-09-13 定案：要做）**：`SurvivalProcess.detect`/
`detect_all_generations` 現在只回傳 `(xy, score)`，從來沒有算過或存過
descriptor，整個 descriptor 相似度分析目前完全沒有資料可用——要接上
`Heads.py` 的 `sample_descriptors`，讓兩個函式也回傳每個偵測點的 descriptor。
這不是實作到那步才會發現的小事，是動工前就存在、而且已經決定要填的缺口。

**還沒定案的細節**：AUC 多高算「可分/有意義」的門檻還沒選；距離配對取樣的
具體做法（怎麼定義「距離範圍相近」）還沒寫清楚；`SurvivalTable`/
`SurvivalBatch` 目前在 production code 從沒被真的呼叫過（只在自己 docstring
的 usage example 出現）——2.3 這裡到底需不需要靠它對齊 chain/anchor id，還是
直接吃 `anchors_of_generations`/`probe_real` 的 in-memory 結果就夠，還沒決
定。這些等真的要動手實作時再逐一決定。

### 2.4 新生死亡歸因

**新生歸因已經有：** `Attribution.py`（四種歸因、`outranked`、`NONE` 哨兵，見
2.2「已經寫好的」）——模糊新生 / 鄰域新生（分數）/ 鄰域新生（壓制解除）/ 未定，
spec.md §3.2「歸因」一節。2.3 的相依型比對能替「鄰域新生（分數）」提供獨立證據
（同階不同框的自然對照組），屬於補強不是重寫。

**死亡歸因還沒有對應的邏輯。** 一個點在某一階消失，現有欄位（`suppressed_by`）
只回答「是不是被鄰居壓過去」，跟新生那邊的四分法不是對稱的——這裡列成待做，還沒
拆解。

### 2.5 再決定要訓練哪一種語意

**還沒決定，等 2.2-2.4 有數字再選。** 候選：

- 相依型 keypoint（2.3）
- 只在一階、帶尺度資訊的 keypoint（2.2 的 C 判準）
- 晚生型（原本 spec.md 假設它是頭最有機會學到東西的地方；這個假設本身要不要
  留著，也是這裡才決定，不在 spec 裡先寫死）

這一步的產物是「Stage C 的 label 定義」，往下接 3.

---

## 3. 訓練有語意的 keypoint

### 3.0 pretile 抽取

沒有抽取這一步了：corpus 是一個 draw，像素在 dataset 的 `__getitem__` 裡讀
（`Corpora.Corpus.read`，每個 worker 自己開 reader）。

硬依賴：label 就是 2.5 的決定 + 2.2-2.4 的輸出。

在 2.5 定案之前，這裡沒有可以動工的東西。`spec.md` §3.3 的 multi-label sigmoid /
相對階梯 label / K 讀出 vs K 學出 三個介面決定已經寫在 spec 裡，是通用的頭部設計，
跟訓練目標選哪個無關，可以先讀。

---

## 側支：soft label（teacher-student）

**不在上面三個階段的主線上，先擱著，還沒決定要不要留。**

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

    1.1 重訓（gray_pre 250ep + gray 消融 50ep）
          │
          ├──並行──> 2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 3     ← 框架先鋒隊
          │
          └──之後──> 1.2 CNN encoder（凍住）──> ViT
                            │
                            └──> 側支 soft label

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
- `log/TODO.log` 要補 reference bank 的純玻璃表（刪掉 44 GB 之前要留的五個數字）。
- ReferenceSampler 退役進 TileSampler：bucket 換成新的七個、jitter 換成
  `OverlapConfig` 的比例、`over` 不帶過去、`ConfigIdentity` 用自我驗證的改名腳本
  遷移。
- ~~spec.md 還有四段過時：`tile_size`/`tissue_ratio 0.75` 那節、`17,784 / 14.2 GB`
  的落地大小表、「`tissue_ratio` 套在 tile 的 footprint 上」、「探針要回答的三件
  事」；split 那節還寫 6 片而程式是 12 片~~ —— 2026-09-03 修掉，見 spec.md §6.5/§6.6/§13。
- `spec.md` §13「四個 arm」底下的執行細節（載入斷言、RGB 複製除以三、`gray+pretrain`
  是自蒸餾、兩片撐不起四個 arm）還留在 spec 裡，內容跟 1.1 現在的「現況讀數」與
  「下一次重訓」兩節重疊——之後要對一次要不要搬到這裡，這次沒動它。
- ~~`MppStack.py`/`CompositeStack.py` 兩個檔案~~ —— 2026-09-05 整併成
  `ChainStack.py`（`FStack`/`RStack`/`CStack` 三個 class），`test_survival.py`
  裡跟 F/R 幾何有關的兩節（`rung_scale`/`rung_shrink`、'R' 的降解）搬進新的
  `test_chain_stack.py`，一起補了 `CStack` 原本完全沒有的單元測試。見 2.1。

---

## 六樣態與特徵尺度的關聯性檢驗（2026-09-14）

假說：細部存活/晚生型/中間帶這幾類存活模式，是否可以用這個點自身的「特徵尺度」
（scale-space 意義下）解釋——衰減得快的點存活範圍窄，衰減得慢的點存活範圍廣。

方法：`cli/demo_survival_analysis.py --parts scale_diagnostic` 對每個 anchor 算一個
古典（非網路、避免跟 `alive_from` 用的 `score` 循環論證）衰減速率——
`classical_blur_stack` 產生的 Gaussian 模糊圖上 `|Laplacian|` 響應，`log(R)` 對
`log(ds)` 做 Theil-Sen 回歸取斜率負值——跟 `_survival_breadth`（該點最粗撐到哪個
ds，不是 alive 幾階的計數）做 Spearman 相關性，加上打散配對的 permutation test
（`--decay-permutations` 次，margin over 誘餌，不是裸 p 值）。

先用 `--parts scale_synthetic_control` 驗證這把尺本身可信：已知半徑（1~16px，log-
uniform）的合成高斯斑點，同一套流程（`classical_blur_stack`/`_laplacian_maps`/
`_decay_rate` 完全共用，不是另外一份實作）量出來 **rho=-0.987、p=0.000
（n=2917）**——尺是準的，不是量測工具本身量不到已知的尺度差異。

**目前站得住腳的結論**：在這批資料、這個古典衰減速率定義下，「六樣態存活模式」
（至少細部存活/晚生型/中間帶這三類）看不出跟特徵尺度有系統性關聯——這是一個真的
空結果，不是量測工具的問題。

重跑驗證：

    PARTS=scale_synthetic_control sbatch jobscripts/SuperPathPointJobs/DemoSurvivalAnalysis.sh
