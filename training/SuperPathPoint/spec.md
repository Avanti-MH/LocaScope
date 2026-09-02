# SuperPathPoint — 在 WSI 上訓練 SuperPoint 的規劃

`training/SuperPathPoint/`

這份文件規劃訓練程序，不是實作紀錄。它要回答的是：資料怎麼準備、模型怎麼組、
loss 是什麼、訓練迴圈長什麼樣、介面切在哪裡才換得掉 encoder 與 decoder，以及
——最重要的——**什麼數字會決定這整件事值不值得做**。

上游參照：`/work/u26130998/SuperPoint`（rpautrat 的 TF 實作 + 根目錄的
`superpoint_pytorch.py`）。訓練程序的形狀參照 `/work/u26130998/GMR-Conv`，
WSI 套件的分層參照 `/work/u26130998/prov-gigapath`。所有從上游抄過來的常數，
在第十節有一張表附出處，不憑記憶寫。

---

## 0. 為什麼做這個

LocaScope 的 stage 3 目前是 `3_localization/SIFT_RANSAC.py`：retrieval 給一個
tile 級的位置，SIFT keypoint + RANSAC homography 把它細化到次像素。它有效
（`log/TODO.log:1683`：一旦拿到含真值的視窗就收斂到微米級），但 SIFT 的點是
手工設計的 blob，它不懂 H&E 紋理，也不知道自己在哪個解析度會消失。

第三件事是這個專案特有的。查詢照片的 mpp 是 stage 1 **估**出來的，不是已知的；
retrieval 選出的 ds 也可能差一層。一個「在 ±1 層 ds 都還在」的 keypoint，和一個
「只在 level-0 存在、放大倍率一降就沒了」的 keypoint，對跨解析度比對的價值差
一個量級——而現在沒有任何東西分辨得出這兩種點。

所以這裡要的不只是「一個學出來的 SIFT」，而是**一個會說自己撐得住多少解析度
落差的 keypoint detector**。

---

## 1. 驗收：什麼數字決定成敗

先寫這個，因為一個講不出兩種結果長什麼樣的訓練程序不值得任何人的 GPU
（ClaudeRules §2、§12）。

| 判準 | 在哪裡量 | 通過長什麼樣 | 失敗長什麼樣 |
|---|---|---|---|
| **每一輪 HA 值不值得再跑一輪** | held-out tile 的 homography pair，repeatability@τ | 第 r+1 輪贏第 r 輪，且贏的幅度大於「同密度隨機點」誘餌與第 r 輪的差距 | 兩輪差距落在誘餌噪音內 → 停，記進 TODO.log |
| **detector 有沒有勝過 SIFT** | `utilities/bench_modules/bench_locascope.py`，同一批 189 張真實 query | centre error 的中位數與成功率都不輸 SIFT | 任一項輸 → 這是負面結果，帶著數字進 TODO.log |
| **解析度頭學到東西沒有** | held-out 存亡表 | 每一階的 AP 贏過「照該階的整體存活率猜」這個常數基線 | 贏不過基線 → 存亡是位置的性質不是外觀的性質，這個頭該砍 |
| **灰階與 RGB 哪個好** | held-out 兩片，**逐片分開報** | 其中一個在兩片上都贏 | 一片贏一片輸 → 差異是片子的性質，兩片不夠判，記成未決 |
| **三個 `tile_size` 哪個好** | 固定評估協定（見 6.5）：同一批位置、同一個 τ、真實照片走 1440x1024 | 某一個尺寸在同一協定下贏 | 各自在自己的 tile 上量 → 這個數字不能用，見下 |

第三列是最容易自我欺騙的一列。若某一階的實際存活率是 0.83，一個永遠輸出 0.83 的
頭 accuracy 就有 83%，看起來像學會了。所以基線是**常數預測**，不是隨機猜。

第五列有一個一定要擋掉的作法：**三個 model 各自在自己尺寸的 tile 上量 repeatability，
然後比大小。** 那是無效的——`valid_border_margin=3` 在 256 上侵蝕掉 4.6% 的面積、
在 1024 上只有 1.2%，所以大 tile 的分數天生比較高，而那和 detector 的好壞無關
（6.5 末段）。三個 model 必須在同一個協定下評估。

第四列的「跨染色泛化」**不在這張表裡，因為這份資料計畫量不到它**——兩種染色都在
訓練集裡。見 6.5 末段。

**每一個判準都是對誘餌評分，不是對容忍度評分。** 誘餌的差距是穩健的，閾值是
猜的——CLAUDE.md 的 `DELTA_MAX` 案例就是閾值算錯而資料是對的。

---

## 2. 詞彙

沿用 CLAUDE.md 的既有詞彙（query / 真實 query / 合成 query / shot），另加：

| 詞 | 意思 |
|---|---|
| **rung**（階） | ds 階梯上的一階。`ds=4` 是一個 rung |
| **ladder**（階梯） | 一組 rung，例如 `{1, 2, 4, 8}` |
| **co-registered stack** | 同一個 level-0 中心、在每個 rung 各讀一張的 tile 集合 |
| **teacher / student** | HA 迴圈中產生 label 的網路 / 被 label 訓練的網路 |
| **survival**（存亡） | 一個 keypoint 在鄰近 rung 是否仍被偵測到 |
| **relative rung** | 相對於本 tile 自己的 ds 的位移，記作 `j`。`j=+1` 是「再粗一倍」 |

`ds` 一律指相對於該片 slide 的 level-0 的降採樣倍率，和 `PatchInfo.ds`、
`WsiTissuesContainer.from_ds` 同義。**注意它跨 slide 不可比**：BRACS 的 SVS 每層
4x、Ki67 的 MRXS 每層 2x，而且兩者的 level-0 mpp 本來就不同。第 3.4 節說明這為
什麼在這裡不構成問題。

---

## 3. 三個階段

### 3.1 Stage A — 逐解析度的 Homographic Adaptation（`SuperPoint/`）

照做上游 SuperPoint 的第二與第三階段。第一階段（合成幾何圖形訓 MagicPoint）
**跳過**，改用上游釋出的權重當起點。

```
teacher_0 = SuperPoint/weights/superpoint_v6_from_tf.pth
for round r in 1..R:
    for rung d in ladder:
        for tile t sampled at rung d:
            label[t] = homographic_adaptation(teacher_{r-1}, t, N=100)
    student_r  = train(backbone + detector + descriptor, 所有 rung 的 label)
    teacher_r  = student_r
```

**約束一：homography 不改變解析度。** 一個 rung 的圖只和自己同 rung 的
homography keypoint 做比對。homography 只 warp *tile 影像*，它從來不去改 WSI
讀取的 level。label 不跨 rung 聚合。

**約束二：只有一個網路。** 權重在所有 rung 之間共享，不是每個 rung 一個模型。
理由有二：(a) 每 rung 一個模型的話 Stage C 沒有東西可學，因為「這個點在哪些
解析度存活」在那個世界裡是由「你呼叫了哪個模型」決定的；(b) 真實 query 的 mpp
是估的，推論時根本挑不出該用哪個模型。

**約束三：ds 不進網路。** 沒有 ds embedding、沒有 FiLM 條件化、沒有多一個
channel 放 ds。網路只看畫素。這是你定的約束，而第 3.3 節會說明它如何決定了
Stage C 的 label 形狀。

#### HA 的演算法（照抄上游 `superpoint/models/homographies.py:28-114`）

對一張 tile：

1. 先跑 identity view：`probs[0] = net(image)`，`counts[0] = ones`。
2. 重複 `N-1` 次：
   - `H = sample_homography(...)`，`H_inv = invert(H)`
   - `warped = warp(image, H)`
   - `count = warp(ones, H_inv, NEAREST)` — warp 回來的結果在**原座標系**的覆蓋範圍
   - `mask  = warp(ones, H,     NEAREST)` — **warped 座標系**裡的有效區
   - 兩者都用 `valid_border_margin` 的橢圓核侵蝕，把 homography 的邊界人工物切掉
   - `prob = net(warped) * mask`，再 `prob_proj = warp(prob, H_inv, BILINEAR) * count`
3. `counts = sum(counts)`，`mean_prob = sum(probs) / counts`

**上游的 `aggregation: 'sum'` 這個名字在說謊。** `homographies.py:102-107` 在
`'sum'` 這個分支裡用的是 `mean_prob = sum(probs) / counts`，也就是
coverage-weighted mean。我們這邊的欄位就叫 `mean_prob`，選項就叫
`'mean' | 'max'`。名字要是運算的名字（ClaudeRules §12）。

`counts` 是必須存下來的東西：它分辨「這個像素是 0，因為沒有 homography 在那裡
偵測到點」和「這個像素是 0，因為根本沒有 homography 看得到它」。這兩件事在
tile 邊緣附近完全不同，而且錯了不會有人報錯。

### 3.2 Stage B — 存亡分析（`PointsAnalysisByMpp/`）

#### 詞彙表——這一節用到的每一個名詞

新名詞在這裡定義一次，別處只引用。一個沒有定義就開始用的名詞，讀的人只能猜，
而猜錯的成本是他做出來的東西答的是另一個問題。

**幾何**

| 名詞 | 意思 |
|---|---|
| `level 0` | WSI 金字塔最底層，最高解析度。所有座標的共同單位 |
| `tile` | 餵給網路的方形影像，v1 是 256x256 像素 |
| `ds` | downsample，縮小倍率。階梯是 1, 2, 4, 8, 16, 32 |
| `rung`（階） | ds 階梯的一格 |
| `footprint` | 一張 tile 蓋住多少 level-0 像素 |
| **感受野**（receptive field） | 輸出上一個位置的值由輸入上多大一塊決定。**這個 trunk 是 84 個輸出像素**，在 ds=d 就是 `84d` 個 level-0 px。超出這個範圍的組織對這個點沒有影響 |

**堆疊**

| 名詞 | 意思 |
|---|---|
| **stack**（同心堆疊） | 同一個 level-0 中心，每一階各讀一張 tile |
| **chain**（鏈） | stack 從取樣器那邊看的名字。`inherit_id` 把同一條的成員標在一起 |
| `stack_kind='R'` | resolution stack。footprint 固定 `tile` 個 level-0 px，從 level 0 讀、降 ds 倍、升回 `tile` 像素。**同一塊組織，細節變少** |
| `stack_kind='F'` | FoV stack。tile 像素數固定，footprint 隨 ds 變大，讀該階的金字塔層。**更多組織，細節更粗** |
| **參照讀取**（reference read） | 把一張 'F' tile 的 footprint 在 level 0 全解析度一次讀成 `tile x ds` 的大圖。**不是第三個 `stack_kind`**——它是「這個 footprint 在 ds=1 的樣子」，只是一次讀完而不切成子塊 |

**過程與產物**

| 名詞 | 意思 |
|---|---|
| **resolution survival process** | 'R' 堆疊 -> 存亡表。操作 |
| **FoV survival process** | 'F' 堆疊 -> 存亡表。操作 |
| `SurvivalTable` | 產物。一列一個 keypoint |
| `tau`（τ） | 兩個 rung 的點要多近才算「同一個點」，單位 level-0 像素 |

**存亡表的欄位**

| 欄位 | 意思 |
|---|---|
| `born_rung` | 這個點在哪一階**第一次**被偵測到 |
| `alive[L]` | 每一階在不在。**是事後由 `score` 和 `tau` 算出來的，不是原始資料** |
| `score[L]` | 每一階的 detector 機率 |
| `dist[L]` | 配到的最近點的 level-0 距離；配不到記 -1 |
| `suppressed_by[L]` | 被 NMS 壓掉時，壓它那個點的分數與 level-0 距離；沒被壓記 -1 |

**存活樣態**

| 名詞 | 意思 |
|---|---|
| **存活向量** | 一個點在各階的 `alive`，一串布林值 |
| **連續帶** | 存活向量裡活著的部分是連續的一段。用 `j_lo`（最細那端）和 `j_hi`（最粗那端）描述 |
| 一直存活 | `j_lo` 最細、`j_hi` 最粗。**攜帶位置資訊**，檢索與定位要的 |
| 細部存活 | `j_lo` 最細、`j_hi` 在中間。小的紋理細節 |
| **晚生型** | `j_lo` 在中間、`j_hi` 最粗。只有縮小才看得見的大結構 |
| **只在一階** | `j_lo == j_hi`。**攜帶尺度資訊**——它的存在本身就說了「你在這一階」，這是 Stage 1 的 mpp 估計要的 |
| 中間帶 | `j_lo` 和 `j_hi` 都在中間。有特徵尺度的結構 |
| 不連續（閃爍） | 活、死、活。**不是連續帶** |

**歸因**

| 名詞 | 意思 |
|---|---|
| **新生**（born late） | `born_rung` 不是最細的那一階 |
| **新生歸因**（birth attribution） | 把一個新生指派給一個成因。**是推論不是量測**，所以每一條分支都必須有資料上的判準 |
| 模糊新生 | 'R' 軸在同一個 ds 也新生 -> 細節劣化造成的 |
| 鄰域新生（分數） | 只有 'F' 新生，且它自己的分數升高 -> 感受野在 level-0 上變大造成的 |
| 鄰域新生（壓制解除） | 只有 'F' 新生，分數沒動，`suppressed_by` 顯示原本壓它的點死了 |
| 未定 | 以上都不符 |

「大結構進入尺度」和「周圍組織改變響應」**不是兩個標籤**：兩者都是「感受野裡的內容
變了、分數升高」，幾何上完全一樣。要分辨得知道感受野裡「是什麼」，那是語意問題，
沒有任何一種堆疊方式分得開。

**一個「點 x 階」格子裡存的三個數，以及 `alive` 怎麼從它們算出來**

    alive[階] = ( score[階] > 閾值 )  AND  ( offset[階] <= tau[階] )
                  夠不夠亮                     夠不夠近

| 名詞 | 意思 |
|---|---|
| **錨點**（anchor） | 一個要追蹤的 level-0 位置。所有階的偵測結果聯集起來、在固定半徑內去重複而成，所以「只在粗階出現」的點也是錨點——那些正是晚生型 |
| **探測**（probe） | 在某一階的圖上，看錨點附近有什麼。回傳 score、offset、rival 三個數。**它和「偵測」是兩回事**：偵測決定哪些位置值得問，探測回答每一階對每個位置說了什麼——包括沒偵測到它的那些階，因為「沒偵測到」不是一個量測 |
| **score**（分數） | 探測到的最高機率，0 到 1。回答「這裡有沒有東西」 |
| **offset**（偏移） | 那個最高峰離錨點多遠，單位 level-0 像素。存成 `dist`。回答「這是不是同一個點」。粗階的一個像素等於 `ds` 個 level-0 像素，所以粗階的偏移天生就大 |
| **rival** | 那個峰的 NMS 半徑內、排除它自己的最高值。`rival > score` 就是「這個位置被旁邊壓過去了」。存成 `suppressed_by_score` |
| **alpha** | `tau = alpha * ds`。**要選的就是這個數**——「容忍幾個粗階像素」。它是無單位的，所以一個值適用於所有階 |
| **探測窗口**（probe window） | 探測搜尋的半徑，`probe_alpha * ds`。**它是這張表能回答的 tau 的天花板**：窗口外從來沒看過，所以 tau 超過它時報的是窗口不是資料，而症狀是曲線變平——看起來像資料飽和 |

**校準用的量**

| 名詞 | 意思 |
|---|---|
| **配對率**（match rate） | 一階上有多少比例的錨點同時通過兩個判準。tau 愈寬愈高 |
| **誘餌率**（decoy rate） | 同樣的問題，但把配對整體推遠 `shift_alpha * ds` 再問一次。**推遠了還配得到，表示 tau 寬到什麼都吃**——那是密度給的，不是訊號 |
| **gap** | `配對率 - 誘餌率`。扣掉密度紅利之後剩下的 |
| **margin** | `配對率 / 誘餌率`。**它沒有拐點**：誘餌為 0 的地方它是無限大，然後隨 tau 單調衰減到 1，所以它說的是「比亂猜好多少」，不是「該選哪裡」 |
| **knee**（拐點） | 曲線「不再明顯上升」的那個位置。這裡的定義是 **gap 最大的那個 alpha**。它是真的極大值不是啟發式：tau 變大時配對率和誘餌率**收斂到同一個極限**（所有有配對的錨點），所以 gap 從 0 升起、必然再回到 0。**峰落在掃描範圍的兩端就不是峰**——落在上端表示掃得不夠遠，落在下端表示最緊的 tau 誘餌就已經贏了 |

**方法**

| 名詞 | 意思 |
|---|---|
| **誘餌**（decoy） | 把點集整體平移超過 `tau` 之後再算一次同樣的數字。任何比例都要贏過它才算數（第 1 節） |
| **閾值敏感度** | 換一個偵測閾值重算，看結論移動多少。**「只在一階」對它最敏感**：那一類要求兩側都死，是兩個否定判斷 |

#### 兩個軸，兩個 process，一張表

中心固定之後，`footprint_l0 = tile * ds` 這條式子還剩一個自由度。
`TileSampler` 已經把兩種取法命名了，而且 `stack_kind` 進了 identity：

```
stack_kind = 'R'   resolution stack   footprint 固定 tile 個 level-0 px，
                                      從 level 0 讀、降採樣 ds 倍、再升回 tile 像素。
                                      同一塊組織、同樣輸出尺寸、真實細節變少。
stack_kind = 'F'   FoV stack          tile 像素數固定，footprint 隨 ds 變大。
                                      讀該階自己的金字塔層。ds 32 的 tile 包含
                                      ds 1 的 tile：同樣像素數、更多組織、細節更粗。
```

三個層次要分得開，因為它們是三種東西：

| 層次 | R 軸 | F 軸 |
|---|---|---|
| **操作**（process） | `resolution survival process` | `FoV survival process` |
| **產物** | `SurvivalTable`，欄位 `stack_kind='R'` | 同一張表，`stack_kind='F'` |
| **問題** | 這個角點禁不禁得起失去細節 | 這個角點禁不禁得起被縮小 |

**產物是一張表不是兩張。** `stack_kind` 本來就是欄位而且在 identity 裡；拆成兩張
會讓「同一個點在兩個軸上的行為」變成要 join 才看得到的東西。而**不說是哪個軸的
存活數字沒有意義**——這正是 `stack_kind` 不是讀取時旗標的原因。

不要為這兩個 process 另取名字。`stack_kind='R'` 已經是這個軸的名字，
「Resolution Down Sample Survival」會是同一件事的第二個名稱（CLAUDE.md 的詞彙
規則）。

#### 那段拒絕已經過時了

這一節原本寫：

> 另一種做法是「固定 level-0 footprint、讓像素數隨 ds 變小」。不採用，因為粗階
> 會小到只剩幾個 cell（tile 256 在 ds=8 只剩 32 px = 4x4 個 cell），而且那不是
> 真實 pipeline 會餵給網路的東西。

**被拒絕的是第三種做法，不是 'R'。** 'R' 升採樣回 `tile` 像素，所以網路拿到的
永遠是固定尺寸——`TileSampler` 的註解寫明「Both are trainable」。那段拒絕成立的
是「footprint 固定 **且** 像素數縮小」，而 'R' 只固定前者。

'R' 有它自己的極限，而且更早到：真實取樣點是 `tile / ds`，

| ds | 真實像素（tile 256） | 有真實細節的 cell（cell 8） |
|---|---|---|
| 1 | 256 | 32x32 |
| 4 | 64 | 8x8 |
| 8 | 32 | 4x4 |
| 32 | **8** | **1x1** |

ds 32 是 8 個真實像素被撐開成 256。**'R' 的可用範圍大約到 ds 8**，而 'F' 的牆
在 footprint（ds 64 是 16384 level-0 px，實測零個可用位置）。兩個軸在不同的地方
用完，這是選擇跨幾階時要看的東西。

#### co-registered stack 的定義（'F'）

取一個 level-0 位置 `(cx, cy)`，在每個 rung 讀一張 tile：**同中心、同 tile 像素
數、footprint 隨 ds 變大**。

```
rung d 的 tile：level-0 上一塊 (tile_size * d) x (tile_size * d) 的方形，
                以 (cx, cy) 為中心，讀成 tile_size x tile_size 像素
```

採用的做法有一個代價：粗階的 tile 看到的組織比細階多，所以**存亡只在最細那階的
footprint 內有定義**——那是所有 rung 的交集。這個限制寫進 `SurvivalTable` 的
docstring，不是留給讀者推。

**'R' 沒有這個代價**：每一階的 footprint 都是 `tile` 個 level-0 px，交集就是整張。

#### sub-tile 的對應是精確的，但「看起來的大小」不是

一張 'F' ds=4 的 tile 輸出 256 像素、覆蓋 level-0 1024 px。它的 ds=1 子塊每塊覆蓋
256 px，所以一邊 4 塊、共 16 塊，每塊在 'F' 圖裡佔 64x64 像素。

| | 覆蓋 level-0 | 真實取樣點 | 輸出成幾像素 |
|---|---|---|---|
| 子塊的 'R' ds=4 | 256 px | 64 | **256**（升採樣回去） |
| 同一塊在 'F' ds=4 裡 | 256 px | 64 | **64** |

**細節等級一樣，但 'R' 把它撐開成 d 倍大。** 一個 level-0 上 8 px 寬的結構在 'R'
裡是 8 個輸出像素寬，在 'F' 裡是 2 個。

#### 感受野是 84 像素，而這決定了「上下文」是什麼

VGG trunk 是四個 stage、每個 stage 兩層 3x3、中間三個 2x2 池化，加上 detector
head 的一層 3x3：

```
stage1  3 -> 5     pool  j=2
stage2  10 -> 14   pool  j=4
stage3  24 -> 32   pool  j=8
stage4  52 -> 68
head    -> 84
```

**一個 keypoint 被不被偵測到，只由它周圍 84 個輸出像素以內的東西決定。**

| | 一個輸出像素 = 幾個 level-0 px | 感受野蓋住多少 level-0 |
|---|---|---|
| 'R' ds=4 | 1 | 84 |
| 'F' ds=4 | 4 | 336 |
| 'F' ds=32 | 32 | 2,688 |

於是一個結論，它推翻了一個看起來很自然的設計：**「看到更多組織」和「東西變小了」
是同一件事**。對感受野固定的 CNN，把圖縮小 d 倍就等於讓感受野蓋住 d 倍的組織。
那不是兩個機制，是一個機制的兩種說法。

#### 被考慮並否決的第三個 arm

那個 2x2 少了一格，而少的那格看起來很吸引人：

| | footprint | 渲染 |
|---|---|---|
| 'R' | 窄（tile 個 level-0 px） | 原尺寸 |
| 'F' | 寬（tile x ds） | 縮小 |
| **第三個** | **寬** | **原尺寸** |
| （第四個） | 窄 | 縮小 —— 退化，圖太小 |

提出它的理由是「第三個 vs 'R' 只差視野、第三個 vs 'F' 只差表觀大小，所以可以把
上下文和尺度結構拆開」。**那是錯的。** 第三個 arm 渲染成原尺寸，所以它的感受野
還是蓋 84 個 level-0 px——和窄的 'R' 一模一樣。多出來的 footprint 全部在感受野
外面，對任何內部的點都沒有影響。

它唯一真正消掉的是子塊邊界剪裁那 6.2%（見下），而那有成本為零的解法。它自己的
成本是 `d^2` 倍計算：ds 32 是 8192x8192 的圖，要切塊加 halo 才跑得動。

**否決。記在這裡是因為它是一個會被重新提出的想法，而否決它的是一個算得出來的
數字（84），不是品味。**

#### 那 (i) 尺度結構 和 (ii) 上下文 要怎麼分

分得開，而且是免費的——只要看偵測規則本身。`points_from_prob` 的順序是
**NMS -> 邊界 -> 絕對閾值 -> 上限**。上限關掉之後**沒有任何全域競爭**，一個點的
生死只剩兩條路：

1. 它自己的分數，由 84 像素的感受野決定
2. NMS 半徑（4 個輸出像素 = `4 * ds` 個 level-0 px）內有沒有分數更高的鄰居

兩條都記得下來：

| | 判準 |
|---|---|
| **(ii) 上下文 / 競爭** | 點死了，而 NMS 半徑內有一個分數更高的存活點 |
| **(i) 尺度結構** | 點自己的分數變了，NMS 半徑內沒有壓過它的鄰居 |

「該死卻活著」同理：分數升高 -> 支撐它的結構大於 84 個 level-0 px -> (i)；分數
沒動、只是原本壓著它的鄰居不見了 -> (ii)。

**所以要多存一個欄位 `suppressed_by`，不是多跑一個 arm。**

**而 `max_keypoints` 在存亡分析時必須關掉。** 上限製造全域競爭：一個點掉出前 N
名看起來就像被上下文殺死，但那是我們自己訂的名額。它會偽裝成 (ii) 而且無法從表
裡分辨。

#### 配對

rung d 的 tile 上偵測到的點 `(u, v)` 映回 level-0：

```
X = x_d + u * d        Y = y_d + v * d
```

兩個 rung 的點是「同一個點」當且僅當 level-0 距離 <= tau。

**tau 必須是物理量，而且它的下限由粗階自己決定。** 粗階的一個像素在 level-0 是
`d_coarse` 個 level-0 px；一個真實存在的點在粗階最多只能被定位到那個精度。所以

```
tau(d_fine, d_coarse) = max(tau_floor_um / mpp_0, alpha * d_coarse)
```

`alpha` 是幾個粗階像素（起始值 1.5，第一次跑當校準跑，不當結果——ClaudeRules §8）。
若把 tau 寫成固定的 level-0 像素數，粗階在定義上就永遠配不到，而輸出會是
「keypoint 在粗解析度全部消失」——一個看起來像發現的錯誤答案。

**'R' 軸的 tau 不一樣。** 'R' 的每一階都是從 level 0 讀的，位置精度是
`shrink = ds` 個 level-0 px（降採樣再升回來），所以同一條式子成立但 `d_coarse`
換成 `shrink`。這件事要在程式裡由 `RungPlan` 回答，不是由呼叫端猜。

#### 輸出

`SurvivalTable`，一列一個 keypoint：

| 欄位 | 意思 |
|---|---|
| `wsi_stem` | 哪一片 |
| `stack_kind` | `'R'` 或 `'F'`——沒有這欄的存活數字沒有意義 |
| `x0, y0` | level-0 次像素座標（在 `born_rung` 上偵測到的位置映回來） |
| `born_rung` | 這個點是在哪一階第一次被偵測到 |
| `alive[L]` | 每一階在不在，bool |
| `score[L]` | 每一階的 detector 機率，float16 |
| `dist[L]` | 配到的最近點的 level-0 距離，配不到記 -1 |
| `suppressed_by[L]` | 被 NMS 壓掉時，壓它那個點的分數與 level-0 距離；沒被壓記 -1 |

`score` 和 `dist` 一定要一起存。`alive` 是 `score` 過閾值又過 tau 的產物，只存
`alive` 等於把兩個閾值凍進資料裡，之後想換就得重跑整批（ClaudeRules §8）。
`suppressed_by` 的理由在上一節：它是 (i) 和 (ii) 唯一的分界線。

#### 六種存活樣態，全部要觀察

每個點的存活是一個布林向量。所有樣態只分兩種：**連續的一段（帶）**，或**不連續**。
連續帶用 `j_lo`（最細那端）和 `j_hi`（最粗那端）兩個數字描述。

| 樣態 | `j_lo` | `j_hi` | 物理意義 |
|---|---|---|---|
| 一直存活 | 最細 | 最粗 | 跨尺度穩定，檢索最想要的 |
| 細部存活，粗了就死 | 最細 | 中間 | 小的紋理細節 |
| **晚生型** | **中間** | **最粗** | **只有縮小才看得見的大結構** |
| 只在一階 | d | d | 帶寬 1，上面的特例 |
| 中間帶 | 中間 | 中間 | 有特徵尺度的結構 |
| **不連續 / 閃爍** | — | — | 活、死、活 |

**晚生型就是「FoV 全域 keypoint」。** 結構太大，在細階它超出 84 個 level-0 px 的
感受野，detector 看不出它是個角；縮小之後才進得了感受野。腺體、導管、組織邊界的
轉角都是幾百到幾千 level-0 px 的東西。它不需要第三個 arm 就找得到——它直接寫在
存活向量裡，而且 `born_rung` 已經把它和「一直存活」分開了。

**不連續的比例是 Stage B 要交出的第一個數字**（3.3 節：0.97 和 0.6 導向相反的
結論）。而閃爍有兩種，要分開：真的閃爍，和閾值抖動造成的假閃爍——後者靠「換個
閾值，閃爍比例變多少」量出來，這就是 `score` 不能提早二值化的第二個理由。

#### 噪音：兩類，處理方式不同

**A. 系統性假象——會偽裝成要找的訊號**

**① 子塊的邊界剪裁。** `border=4` 把距離 tile 邊緣 4 像素內的點一律丟掉。在子塊
（256 像素蓋 256 個 level-0 px）剪掉的是 4 個 level-0 px 的框；在 'F' ds=4 裡，那
16 個子塊的**內部接縫根本不是邊界**。所以落在子塊邊緣 4 個 level-0 px 內的點，在
'F' 裡偵測得到、在子塊裡永遠偵測不到：

```
1 - (248/256)^2 = 6.2%
```

**6.2% 的點會無條件呈現「該死卻活著」**，而那和要找的效應量級相當。處理：比較只在
「距離每個子塊邊界 >= 4 個 level-0 px」的區域內做。成本零。

**② 重採樣來源不同。** 'F' 讀的是金字塔的某一層——掃描機當初做的降採樣；'R' 是
軟體降採樣。兩種濾波器的高頻內容不一樣。**這是 'F' 和 'R' 之間無法消除的系統差，
只能記錄，不能修掉**，而它正是「不說是哪個軸的存活數字沒有意義」的第二個理由。

**B. 真噪音——閾值附近的抖動**

分數剛好在閾值邊緣，兩張圖一邊過一邊不過。處理就是上面的：存分數不要提早二值化，
然後量結論對閾值有多敏感。若「該死卻活著」的比例在閾值 0.01 和 0.02 之間翻倍，
那個發現不成立。

**C. 誘餌**

「該死卻活著」的比例沒有參照就沒有意義。誘餌：**把該階的點整體平移超過 tau，再
算一次同樣的比例。** 真效應要贏過誘餌。這和 repeatability 的 decoy 是同一個手法
（第 1 節：每個判準都是勝過誘餌，不是超過某個絕對值）。

#### 一個免費的檢查，寫 `MppStack` 時第一個寫

**ds=1 的時候 'F' 和 'R' 是同一張圖。** 所以 'R' 在 ds=1 的偵測結果和 'F' 在 ds=1
的必須**逐點完全一致**。

這個斷言不花錢，而且它抓的是座標換算錯誤——「圖看起來很正常、數字全錯」的那一類，
正是 `test_camera_output_to_level0` 找到真 bug 的那一類（第 14 節）。

### 3.3 Stage C — 解析度語意頭（`SemanticPoints/`）

在共享 backbone 上多接一個頭，用 Stage B 的存亡表當 label。

#### label 是相對的，不是絕對的

這是 Stage A 約束三的直接後果，也是這個階段唯一真正需要想清楚的設計點。

若 label 用**絕對** rung 編號（channel k = 「在 ds=2^k 存活」），網路必須先知道
自己看的是哪一階才可能預測——那等於把 ds 從後門送回輸入端。同一張影像，如果它
其實是 ds=1 的 tile，答案是一組；如果它是 ds=4 的 tile，答案是另一組。函數不是
良定義的。

所以 label 是**相對**的：

```
channel j = 「這個點在 (本 tile 的 ds) x 2^j 還在不在」
j 屬於 {+1, +2, +3}，可選加上 {-1, -2}
```

網路學的是「這個角點禁不禁得起再縮一半」，純從外觀。這對 LocaScope 本身也更
有用：真實 query 的 mpp 是估的，我們能問的問題本來就是相對的。

副作用：**ds 跨 slide 不可比這件事在這裡自動消失了**。相對階梯問的是倍率，而
倍率沒有單位。BRACS 的 4x 金字塔和 Ki67 的 2x 金字塔在 `j=+1`（縮一半）這件事
上意思完全一樣。這是選相對編號換到的第二個好處，不是設計目標。

#### 為什麼是 multi-label sigmoid，不是 L-way 分類

存活不是 one-hot。一個點在 `{j=1,2}` 活、`{j=3}` 死，這是一個「帶」。`2^J` 種
樣態裡只有大約 `J(J+1)/2` 種是連續的帶。強行做成分類，要嘛丟掉非帶樣態的資訊，
要嘛先假設所有樣態都是帶——而那個假設沒有人驗證過。

所以：學完整樣態（每階一個 sigmoid），然後**量**它有多常是連續帶。那個比例本身
就是要寫進 `log/TODO.log` 的發現。如果它是 0.97，之後把頭簡化成預測
`(j_lo, j_hi)` 兩個數字是有根據的；如果它是 0.6，那簡化會是一個安靜的錯誤。

**那個比例是 Stage B 量的，不是 Stage C**（3.2 節「六種存活樣態」）。六種裡五種
是連續帶、一種不是，而其中一種——**晚生型**（`j_lo` 在中間、`j_hi` 在最粗）——是
「只有縮小才看得見的大結構」，也就是這個頭最有機會學到東西的地方。若晚生型的比例
接近零，這個頭沒有東西可學，而那是 Stage B 就該發現的事，不是訓練完才發現。

而閃爍要先扣掉閾值抖動造成的那一半再報（3.2 節噪音 B），否則「不是連續帶」的比例
會被高估，把一個可以簡化的頭判成不能簡化。

#### 監督在哪裡

頭輸出 `[N, J, Hc, Wc]` 的 logits，和 descriptor 同一個 cell 網格。loss 只在
**存亡表有 label 的那些 keypoint 位置**算（用 `grid_sample` 在該位置雙線性取樣），
其餘位置遮掉。沒有 label 的位置不是負樣本，它只是沒被問過。

#### 推論時的形狀（初步想法，前面做完再定案）

目前的想法是：**輸入一個分類數 K，對某一個 rung 的圖推論，得到 keypoints 與
descriptor，然後依 K 把這些 keypoint / descriptor 分到 K 個類。**

這和上面的頭相容，但有一個要在設計時決定的分岔，先記在這裡：

| 作法 | K 換掉要不要重訓 | 說明 |
|---|---|---|
| **K 是讀出（建議）** | 不用 | 頭學的永遠是 J 維存活向量；K 個類是在那個向量上做的一次分桶。K 是推論時的引數 |
| K 是學出來的頭 | 要 | 頭直接輸出 K 路 softmax。K 一改，label 的定義就改了，整個 Stage C 重訓 |

建議前者，理由就是那一欄：K 是「我現在想把點分成幾堆」，那是使用端的問題，不是
模型的性質。存活向量是模型量到的東西，分桶是拿它來做的事。把 K 焊進模型等於每
問一次不同的問題就要重訓一次。

若採讀出，還要決定分桶的規則（等寬？照 `j` 的最大存活階？照存活向量的分群？），
而那要等看過真實的存活向量分布——特別是 3.3 節那個「有多常是連續帶」的比例
——才有得選。**這一節在 Stage A 與 Stage B 有產出之前不定案。**

---

### 3.4 三個階段之間傳什麼

```
Stage A  --> KeypointLabelStore（HA label）+ 一個 checkpoint
              checkpoint 帶 identity_json，所以 Stage B 說得出自己用了哪個 detector
Stage B  --> SurvivalTable（parquet / CSV）
              帶產生它的 checkpoint 的 identity_id，以及 stack_kind：
              'R' 和 'F' 是兩個不同的問題，一張沒說是哪個的表沒有意義
Stage C  --> 一個多了 resolution head 的 checkpoint
```

Stage B 的表必須記下 detector 的 `identity_id`。否則換一個 detector 重跑，兩張
表混在同一個目錄裡，而它們量的是不同的東西——這正是 `FeatureStore` 存在的理由
（`utilities/FeatureStore.py` 開頭那段）。

---

## 4. 目錄佈局

```
training/SuperPathPoint/
  README.md                    # 照 3_localization/README.md：原則 + 檔案狀態標籤
  common/
    Interfaces.py              # Backbone / DetectorDecoder / Head protocol + 輸出 dataclass
    Homography.py              # sample_homography / warp_points / warp_image / valid_mask
    DsLadder.py                # ds 階梯 -> 每片 slide 的 (level, resize)
    KeypointLabelStore.py      # HA label 落地
  SuperPoint/                  # Stage A
    Backbones.py               # VggBackbone（上游的 VGG，從頭訓）
    EncoderBackbone.py         # TileEncoderBackbone：aiNNModel 的 TileEncoder
                               #   當 Backbone 用。獨立一支，因為 import
                               #   aiNNModel 會設 HF_HOME，而 Backbones.py 在
                               #   每一支 CPU 測試的 import 路徑上
    Decoders.py                # DepthToSpaceDecoder / UpsampleDecoder
    Heads.py                   # DescriptorHead
    KeypointNet.py             # backbone + decoder + heads 的組合體
    Losses.py                  # detector CE / descriptor hinge
    Datasets.py                # WsiTileDataset / HomographyPairDataset
    HomographicAdaptation.py   # label 產生器
    Trainer.py
  PointsAnalysisByMpp/         # Stage B
    MppStack.py                # co-registered stack 的取樣與讀取
    SurvivalTable.py           # 配對、落表
  SemanticPoints/              # Stage C
    ResolutionHead.py
    Losses.py
    Trainer.py
  cli/
    demo_homography.py         # diagnostic：每個 homography 操作單獨一格，不需模型
    demo_ha.py                 # diagnostic：3 個 view 各自的點 vs 聚合後的
                               #   label，三個閾值同框，每個 num 一格
    make_ha_labels.py
    train_superpathpoint.py
    build_survival_table.py
    train_semantic_points.py
    inspect_ha_labels.py       # diagnostic：把產出的 label 畫回 tile 上
```

遵守 CLAUDE.md 既有的分工，不另立規矩：

| 東西 | 去哪 |
|---|---|
| 模組本身（純邏輯，不 print / 不 plot / 不讀寫結果檔） | 上面那棵樹 |
| CLI 與診斷工具 | `training/SuperPathPoint/cli/` |
| 單元測試 | `utilities/test_modules/TestSuperPathPoint/test_*.py`（這個專案的九支；共用模組的測試留在上一層） |
| 端到端品質量測 | `utilities/bench_modules/bench_*.py` |
| SLURM jobscript | `jobscripts/` |
| 跑出來的任何東西 | `/work/u26130998/result/`，repo 之外 |

測試放 `utilities/test_modules/` 而不是套件內，是因為 CLAUDE.md 已經定了那條規則，
而且它有理由：測試依模組命名、模組搬家測試跟著搬，一個獨立的測試目錄是唯一能讓
「commit 前把這些全跑一遍」變成一個指令的形狀。

---

## 5. 介面（可換 encoder / decoder 的那一層）

### 5.1 protocol

```python
class Backbone(Protocol):
    out_channels: int      # C
    stride:       int      # 輸入 px / 特徵 px
    trainable:    bool     # foundation model 為 False
    def forward(self, images: Tensor) -> Tensor:   # [N, 3, H, W] -> [N, C, H/s, W/s]
        ...

class DetectorDecoder(Protocol):
    cell: int              # 輸出 cell 的邊長，以輸入像素計
    def forward(self, feat: Tensor) -> Tensor:     # [N, C, Hc, Wc] -> [N, cell**2+1, Hc', Wc']
        ...

class DescriptorHead(Protocol):
    dim: int
    def forward(self, feat: Tensor) -> Tensor:     # [N, C, Hc, Wc] -> [N, dim, Hc, Wc]，已 L2 正規化
        ...
```

`stride` 和 `cell` 是兩個不同的數，這是整個可換性的關鍵。上游把它們綁死成 8
（`vgg_backbone` 三個 maxpool = 2^3，detector head 輸出 `1 + 8^2 = 65` 通道），
所以它從來不需要區分。ViT 的 stride 是 14 或 16，綁死就換不動了。

### 5.2 兩族 decoder

| 實作 | 適用 | 做法 |
|---|---|---|
| `DepthToSpaceDecoder` | `cell == stride`，即 stride 8 的 CNN | 1x1 conv 到 `cell**2+1` 通道 -> softmax -> 丟掉 dustbin -> depth-to-space。上游那個，照抄 |
| `UpsampleDecoder` | `stride > cell`，即 stride 14/16 的 ViT | 小的 conv 上採樣塔把 stride 降到 `cell`，再接 depth-to-space |

**`UpsampleDecoder` 沒有上游可抄，是新的。** 這是接 foundation model 時唯一真正
要設計的東西，也是第一版最可能需要迭代的地方。它的第一版：
`ConvTranspose2d(stride=2)` + `GroupNorm` + `GELU` 疊到 stride 8，通道數逐層減半。

**dustbin 在兩族都保留。** 它讓「這個 cell 裡沒有 keypoint」成為一個可以被預測
的類別，而不是靠閾值切出來的。上游的 detector loss 整個建立在
`cell**2 + 1` 路的 softmax 上，拿掉 dustbin 就得重新設計 loss。

### 5.3 兩族 backbone

| 實作 | 說明 |
|---|---|
| `VggBackbone` | 上游的 VGG：4 個 conv stage、前 3 個後面各一個 maxpool、stride 8、輸出 **128** 通道。`trainable=True`，從頭訓 |
| `TileEncoderBackbone` | 包 `aiNNModel/TileEncoderFunc.py` 的 trunk，回傳 `[N,C,H,W]`。`trainable=False`，**只訓 decoder 與 head**。在 `SuperPoint/EncoderBackbone.py`，不在 `Backbones.py` |

`TileEncoder.spatial()` 在 commit df3a09e 才對 token 模型打開，所以 GigaPath /
UNI2 / CONCH 現在都給得出空間特徵圖。

**但 `spatial()` 是推論 API，不能直接拿來訓練。** 它走 `_run`：每張圖轉 PIL、套
config 的 transform（`Resize(256)` 然後 `CenterCrop(224)`）、`torch.no_grad()`、結果
搬回 host。訓練迴圈餵的是 256 px 的 tile、要的是留在裝置上的特徵圖，這四件事沒有
一件用得上——光是那個 resize 就會把特徵格點放到和 label 不同的像素格上，而且不會
報錯。

要的是它下面一層的 `_spatial_forward(batch)`。**走法是繼承**：`SpatialTrunk`
繼承 `TileEncoder`，`SpatialTrunk.over(enc)` 用 `object.__new__` 加 `__dict__`
複製（和 `TileEncoder.variant()` 同一招，`:1103`）把已經建好的 encoder 換到
`(SpatialTrunk, type(enc))` 這個動態類別底下。子類別呼叫父類別的 protected 方法
不是「伸手進去拿私有東西」——`_spatial_forward` 的 base 實作就是拋
`NotImplementedError` 並點名子類別，它本來就是給子類別覆寫的掛勾。**`aiNNModel/`
一行都不動。**

**哪一個 encoder 做得到，以及在什麼 tile 尺寸下**——只有**一個**數字決定，而且它是
從模型身上讀出來、不是從 arch 名字推的：`patch_embed.patch_size`。

| encoder | patch | 原生輸入 | tile 256 |
|---|---|---|---|
| `gigapath` | **16** | 224，**固定** | 可以，16x16 格 |
| `conch_vit` | 16 | 448，dynamic | 可以，16x16 格 |
| `uni2` | 14 | 224，dynamic | **沒有任何 tile 尺寸可以**，見下 |

**`gigapath` 是那個陷阱，值得寫下來。** 它的 arch 叫 `vit_giant_patch14_dinov2`，
而 prov-gigapath 自己的 `config.json` 把 `model_args.patch_size` 覆蓋成 **16**——
名字說 14，模型是 16，它在 224 下的 14x14 格是 `224/16` 而不是 patch 14。從 arch
**名字**讀 patch size，在唯一一個名字裡有 patch size 的 encoder 上剛好會讀反。
`_patch_stride` 讀的是 `patch_embed.patch_size`，所以程式碼一直是對的，錯的是描述。

**輸入尺寸固定不是一道拒絕，是一次呼叫。** `GigaPathFunc` 建的時候沒給
`dynamic_img_size`，而 prov-gigapath 的 `pretrained_cfg` 寫著
`fixed_input_size: true`，所以 `PatchEmbed.forward` 會斷言輸入正好是 224
（`timm/layers/patch_embed.py:120`）。那是**怎麼建的**的性質，不是權重的性質，而
timm 有公開的呼叫可以改它——`VisionTransformer.set_input_size(img_size=...)`
（`timm/models/vision_transformer.py:1013`），它更新 `patch_embed.img_size` 與
`grid_size`，並把 `pos_embed` 重採樣到新格上，一次。

所以 `__init__` 對**每一個** trunk 都在建構時說一次它的輸入尺寸。對
`dynamic_img_size` 的模型，那是本來每個 forward 都會跑的同一段算術（`_pos_embed`
呼叫同一個 `resample_abs_pos_embed`），改成只跑一次；對固定的那個，那是「能用」與
「斷言」的差別。

它**會改動已載入的模型**。這件事有被記下來：`tile_size` 是上面那個 config 的雜湊
欄位，而重採樣過的 `pos_embed` 是 `weights_id` 雜湊的東西的一部分——所以「在 256
下讀的 trunk」和「檢索管線裡在 224 下讀的同一個 encoder」正確地拿到不同的 id。

剩下唯一的拒絕是 `tile % patch`，在建構時發生、把出路寫在訊息裡（patch 14 在
256 附近是 252 或 266，而 224 是它預訓練的尺寸），而不是等到訓練跑起來的第一個
forward。padding 不是出路：它把 cell 格點推離 label 所在的像素格，而且不會報錯。

#### tile 就是餵進 trunk 的那塊。不 resize，也不 crop

`TransformConfig.build()` 是 `Resize -> CenterCrop -> ToTensor -> Normalize`，
訓練這條路只拿後兩個。兩個都不拿，但理由不同。

**crop 是會安靜出錯的那一個。** 對檢索它不痛不癢——出來一個向量，gigapath 的
`crop_pct=0.875` 切掉外圈只是少一點上下文。對 dense 任務，被切掉的那圈**有
label 而沒有預測**：loss 根本看不到它們，照樣下降，而模型只在每塊 tile 中間
76% 的面積上被訓練。（`uni2` 和 `conch_vit` 的 `crop_size == scale_size`，它們的
crop 本來就是 no-op；三支裡只有 gigapath 真的切。）

**crop 拿掉之後，resize 就不需要了。** tile 尺寸不是從任何地方傳下來的——tile 是
768 px pre-tile 的中心裁切（6.6），換一個尺寸不用重抽。所以直接把 tile 餵進去，
`stride` 就精確等於 patch。

#### patch 14 到不了 cell 8，而且沒有任何 tile 尺寸能改變這件事

decoder 要從特徵格爬到 cell 格，那個倍率是

```
(tile / cell) / (tile / patch) = patch / cell        ← tile 被約掉了
```

所以 gigapath 與 conch 是 `16/8 = 2`——正好是一疊 stride-2 轉置卷積表達得出來的
——而 uni2 是 `14/8 = 1.75`，在**任何** tile 尺寸下都是。tile 還必須同時被兩者
整除，所以 uni2 合法的 tile 是 `lcm(14, 8) = 56` 的倍數：112、168、224、280…
每一個都是 1.75。

（我先前寫的「uni2 就用 224 / 252 / 266」錯了兩層：252 和 266 根本不能被 cell 8
整除，而 224 雖然可以，倍率照樣是 1.75。）

出路剛好三條，沒有一條是 tile 尺寸：

| 出路 | 代價 |
|---|---|
| **resize 影像** 256 → 224 | 特徵格 16 對 cell 格 32，倍率 2。trunk 看到的線性解析度先掉 12.5% |
| **resize 特徵圖** 16×16 → 28×28（在 decoder 裡） | 影像解析度不掉，但把內插搬到特徵空間。`UpsampleDecoderConfig` 自己的 docstring 已經點名這條 |
| **uni2 自己用 cell = 14** | 哪裡都不用 resize，`DepthToSpaceDecoder` 直接接。但 uni2 的 detector 變成 14 px cell 上的 197 路 softmax，別人是 8 px 上的 65 路——把一個架構差異塞進一個目的是隔離另一個架構差異的比較裡 |

**選第三條，而且 `cell = 7`，不是 14。** 14 有兩個因數能讓 `stride/cell` 是 2 的
冪：14（比值 1）和 7（比值 2）。兩個都過得了 decoder 的檢查，而 7 在每一條有意義的
軸上都比較好：

| cell | 比值 | 爬幾階 | 每個預測涵蓋 | 對 cell 8 的密度上限 |
|---|---|---|---|---|
| 14 | 1 | 0，完全不用 upsample | 196 px² | **0.33x**（粗 3.06 倍） |
| **7** | **2** | **1 階，和 gigapath 的 16/8 同一套機制** | **49 px²** | **1.31x（還更細）** |

真正的理由是形狀對得上：

```
gigapath  tile 256  stride 16 -> 16x16 特徵 -> 32x32 cell   head 65 通道
uni2      tile 224  stride 14 -> 16x16 特徵 -> 32x32 cell   head 50 通道
```

**張量形狀從頭到尾一樣**，預測數一樣，用 cell 數算的邊界比例也一樣。只有像素
**間距**不同——每個特徵 14 對 16、每個 cell 7 對 8——而那正是兩個 trunk 的差別，
也正是這個比較要問的東西。224 同時是 uni2 的原生輸入尺寸，位置編碼一次都不用重採樣。

**不用寫任何新程式碼**：`KeypointNetConfig.wired(cell=7)` 建得出來，
`UpsampleDecoderConfig(cell=7, stride=14)` 自己的 2 的冪檢查也過。

（cell 14 被否掉的理由值得留著：它把 uni2 的 keypoint 密度上限壓成 cell 8 的三分之一
——在一個**關於 keypoint** 的比較裡給其中一臂比較低的上限。cell 7 反而略高，這個
問題就不存在了。）

**唯一真正的代價是 label store，而且躲得掉。** HA 的 label 是按 tile 尺寸切的，
所以 tile 224 和 tile 256 是兩份 store，而 `make_ha_labels` 是以 GPU-hours 計的
那一步。但它們不必是兩次**執行**：224 的 tile 就是 256 的中心裁切，所以點座標
各減 16、落在框外的丟掉、在新框上重切一次 `border=4` 就得到了。內部完全精確，
只有最外圈 4 px 和原生跑 224 會不一樣。

#### `cell` 不是 upsample 的 kernel size

三個數字要分清楚：

| | 是什麼 | 誰的性質 |
|---|---|---|
| `stride` | 每個**特徵**像素等於幾個輸入像素 | backbone |
| `cell` | 每個**預測**涵蓋幾個輸入像素見方。head 出 `cell**2 + 1` 通道，`depth_to_space` 攤開它，所以一個 cell 裡最多一個 keypoint，但位置精確到像素 | detector |
| upsample kernel | 固定是 2（`ConvTranspose2d(kernel_size=2, stride=2)`） | `UpsampleDecoder` 的實作細節 |

`cell` 在 `UpsampleDecoder` 裡唯一決定的是**爬幾階**：`log2(stride/cell)`，
stride 16 是一階，stride 8 是零階。

而且——**`cell` 不是 label 的性質。** label 存的是**點**（`KeypointLabelStore`
的 `kp_xy`），`Losses.cell_labels` 是在算 loss 的時候才把點折進 cell 的。所以改
`cell` 不會讓任何已經存下來的東西需要重切。原本 `Interfaces.py` 寫「cell 是 label
的性質，因此每個要互相比較的 backbone 都必須一樣」——前半句是錯的，後半句只剩下
一句**告誡**：cell 不同的兩個模型多差一個地方，那件事要**說出來**，而不是用設計
擋掉，因為擋掉它就等於把 resize 逼到別的地方去。

`cell` 真正決定的是 keypoint **密度上限**：每 `cell**2` 個像素一個預測。
`cell_labels` 的 argmax 會把一個 cell 裡多出來的點丟掉（`tie_break` 讓每一步隨機
留一個，所以是對監督訊號的隨機抽樣、不是刪除）。這個上限會不會真的卡到，是一個
**還沒量的量**——`cli/inspect_ha_labels.py` 的 `n_kp` 直方圖對上 cell 數就是答案。

#### stride 和 cell 曾經在三個地方被當成同一個數

順著「cell 到底是什麼」查下去翻出來的，**對 gigapath 是活的 bug**。上游的 VGG
兩者都是 8，所以既有的每一支檢查在任一種讀法下都會過；三處都不會拋例外，三處都
只在 `stride != cell` 時錯——也就是 gigapath（16/8）和 uni2（14/7）。

| 位置 | 錯在哪 |
|---|---|
| `Heads.sample_descriptors(kp, dense, s)` | `s` 必須是 `dense` 那張圖的間距，也就是 backbone 的 stride。呼叫端 `KeypointNet.extract_keypoints` 傳的是 `cfg.detector.cell` → gigapath 的 scale 只有真實範圍的一半 → 每個正規化座標大兩倍 → `grid_sample` 把**整批 descriptor 從角落取樣回來**。單位長度、形狀正確、內容錯 |
| `Losses.descriptor_loss(..., cell)` | `hc, wc` 來自 `descriptors.shape`，那是 stride 的格；卻用 `cell` 去算中心點座標 → 所有對應關係都算在 tile 左上四分之一裡 |
| `Losses.correspondence_mask` | 參數本身：它算的是格子中心，格距是**呼叫端那張圖**的間距，不是 detector 的 cell |

修法：兩處參數改名 `stride`（`correspondence_mask` 內部叫 `pitch`），
`KeypointOutput` 多帶一個 `stride` 欄位——**只有建好的模型知道它是多少**
（foundation trunk 的 stride 是它的 patch size，config 裡沒有這個數）。有
descriptor 而 `stride` 未設就拋例外，不預設成 `cell`：那個預設只有在兩者相等時
才對，而那正是上游，不是這裡。

誘餌測試 `t_descriptors_are_sampled_on_the_stride_and_not_on_the_cell`：stride 16
的圖、每個特徵像素種一個向量、在各自中心取樣。**傳 stride 必須原封不動取回，傳
cell 必須取不回來。** 既有那幾支呼叫改用 `STRIDE` 這個名字（值等於 `CELL`），讓
讀的人看得出每個呼叫點指的是哪一個——那個巧合正是這個 bug 藏了這麼久的原因。

#### 「格子中心」有兩個慣例，差半個像素

`test_superpathpoint.py` 第一次真的執行（job 317136）就抓到一支 FAIL，而且**錯的是
測試不是程式**——值得寫下來，因為它是同一類的東西。

特徵像素 `col` 的中心，用**輸入像素索引**表示，是 `col*s + (s-1)/2`：s=8 時是 3.5，
不是 4。索引 i 涵蓋連續區間 `[i, i+1)`，所以 `col*s … col*s+s-1` 這一塊的中心比
`col*s + s//2` 低半個像素。`sample_descriptors` 裡那個 `+0.5` 就是把索引轉成連續
座標的那一步（上游原文照抄），兩者只有在那個半上才對得起來。

而 `Losses.correspondence_mask` **確實**用 `h*cell + cell//2`——那不矛盾：它也是
上游的（`models/utils.py:79-82`），而且餵進去的是一個容忍度 `cell - 0.5` 的比對，
半個像素改變不了答案。在取樣這裡它就是全部的答案：s=8 時那半個像素把
`1/(2s) = 6.25%` 的鄰居向量混進每一次取樣，這正是第一次執行報出來的東西。

兩個慣例都留著、都標明用途，比統一成一個好——統一就得改掉一處上游原文，而那正是
這份 spec 第 9 節說「照抄」的地方。

`stride` 取的是 **patch_size**，不是 `model_spec.feat_hw`。後者是
`crop_size // patch_size`，是**推論時**在 224 中心裁切上的格數——crop 224 / patch 16
是 14，tile 256 / patch 16 是 16。兩個都講得通、都自洽，用錯的那個會讓 cell 格數和
label splat 的格數對不上，而沒有任何東西會拋例外。這是 `check_shapes` 在建構時用
**真正的 tile 尺寸**跑一次 forward 的理由。

用 foundation model 時凍結 trunk 是你的決定，也有一個獨立理由：GigaPath 的
trunk 是 1.1B 參數，解凍它就不是這台機器上一張卡的訓練了。

### 5.4 組合體

```python
class KeypointNet(nn.Module, IdentifiedBuild):
    backbone:   Backbone
    detector:   DetectorDecoder
    descriptor: Optional[DescriptorHead]
    resolution: Optional[ResolutionHead]   # Stage C 才有

    def forward(self, images) -> KeypointOutput: ...

@dataclass(frozen=True)
class KeypointOutput:
    cell_logits: Tensor           # [N, cell**2+1, Hc, Wc]   loss 吃這個
    prob_map:    Tensor           # [N, H, W]                解碼後的全解析度機率
    descriptors: Optional[Tensor] # [N, D, Hc, Wc]           已 L2 正規化的稠密圖
    survival:    Optional[Tensor] # [N, J, Hc, Wc]           logits
```

**NMS / top-k / 去邊界不在 forward 裡。** 它們是 `extract_keypoints(output, cfg)`
這個獨立函式，回傳 `Keypoints(xy, score, desc, survival)`。理由和上游一致：訓練
吃的是稠密的 `cell_logits`，後處理只在推論時發生；混在一起的話 loss 會意外地
依賴一個不可微的閾值。

### 5.5 registry

**重用 `utilities/ConfigIdentity.py`，不新發明。**

| 要的東西 | 現成的 |
|---|---|
| 名字 -> config 類別 | `@register(name)`（`:359`）、`config_from(name, **over)`（`:378`）、`registered()`（`:374`） |
| config -> 物件 | `ModelConfig.build()`（`:308`），`source='local'` 吃 `'package.module:Class'` |
| 「這個模型是什麼」的短 id | `IdentifiedBuild.identity_id()`（`:253`）、`identity_json()`（`:262`） |
| 哪些欄位不進雜湊 | `NOT_IDENTITY`（`batch_size` 那類） |

範本是 `aiNNModel/TileEncoderFunc.py:1111` 開始那段「choosing an implementation
by name」。照著寫一份就好。

換掉一個東西要改哪個檔案，這是這一節的驗收：

| 想換的 | 改哪裡 | 其他地方要不要動 |
|---|---|---|
| backbone | `SuperPoint/Backbones.py` 加一個 `@register` 的 config 類別（包 foundation model 的那種放 `SuperPoint/EncoderBackbone.py`） | 不用 |
| detector decoder | `SuperPoint/Decoders.py` 同上 | 不用，只要 `cell` 對得上 |
| descriptor 維度 | config 欄位 | 不用 |
| 加第三個 head | `KeypointNet` 加一個 Optional 欄位 + 自己的 loss | Trainer 的 loss 加總要加一項 |

`ConfigIdentity` 的規則一（baseline 只增不改、名字不得改義）在這裡照樣適用：
`BASELINE` 加欄位時要給一個能重現舊行為的值，否則所有既有的 label store 會重新
雜湊——那是重算，不是錯答案，但要知道自己在付什麼。

---

## 6. 資料模組

### 6.1 現成的，不重寫

| 要的 | 用什麼 |
|---|---|
| 組織遮罩、region | `utilities/TissuesRegionsMask.py` |
| 在組織內取 tile | `utilities/TileSampler.py:58`，`TileInfo(level, x, y, tile_size, mpp)`，x/y 是 level-0 |
| 安全讀圖 | `utilities/SafeSlide.py`。**讀圖一律走 `read_region_rgb`（`:371`）**，不要 `.convert('RGB')` |

最後一條不是風格問題。`.convert('RGB')` 只是丟掉 alpha，而沒拍到的像素 RGB 是 0，
於是每個掃描空洞都變成純黑——`read_region_rgb` 的 docstring 記著這件事讓一個
分割模型把掃描跳過的區域判成組織。對 keypoint detector 而言，一塊純黑方塊的邊界
是完美的角點。這個錯誤會產生大量高分的假 keypoint，而且看起來很像模型學會了。

#### 組織分割用哪一個：`test_EoMT.py` 的 PCA 那一半

遮罩不用預設的 HSV，改用 `utilities/test_modules/test_EoMT.py` 裡那套，暫時包成一個
可用的 SegFunc。有兩件事必須先講清楚，否則會包錯。

**那個檔案裡有兩條 mask 路徑，只有一條能用。** 檔案自己的 docstring（`:40-42`）寫著
「The head is randomly initialised. Nothing here trains anything, so **every mask
is noise**」——講的是 `segment()`（`:218`），EoMT 的頭是隨機初始化的。能用的是
`slide_pca_mask()`（`:504`）：UNI2 的 patch 特徵做 PCA，
`mask = pc1 > background_threshold`（`:594-596`），來自 prov-gigapath 自己的 demo。

包錯的後果是雜訊遮罩，而下游沒有任何東西會報錯：region 照樣切得出來、tile 照樣抽
得到、HA 照樣跑完，只是全部長在背景上。**所以要包的是 PCA 那條，不是 EoMT 那條**
——即使檔案叫 EoMT。

**它套不進 `method=` 的介面，不要硬塞。** `TissuesRegionsMask.from_wsi(method=...)`
（`TissuesRegionsMask.py:545-565`）要的是 `callable(img: np.ndarray) -> np.ndarray`，
一張圖進、一張遮罩出。`slide_pca_mask` 不是那個形狀：它自己讀 slide、在分層抽樣的
tile 上擬合一個 PCA、再投影每一格。

而且 **PCA 必須整片擬合一次**，這不是實作偏好。`:516-519` 的 docstring 講了為什麼
不能逐格：`MinMaxScaler` 把每一格自己的極值映到 0 和 1，所以一格全是組織時，它的
閾值會落在組織內部。分割器因此天生是 slide-scoped 的，做不成無狀態的逐圖 callable。

作法是繞過那個便利建構子：`slide_pca_mask` 產出遮罩與它的 ds（= UNI2 的
`patch_size` = 14，比 `from_wsi` 預設的 32 細得多），拿它直接建構
`TissuesRegionsMask`（該類別持有的就是 `mask` / `ds_x` / `ds_y` / `regions`）。
遮罩已經在手上時不需要 `from_wsi`。

包裝的形狀照 `aiNNModel/TissueSegFunc.py`：`@register('...')` 的
`IdentifiedConfig` + `IdentifiedBuild`，於是遮罩的來源自動進 `identity_id`，
換分割器就換 label store 的路徑。這是必要的——HSV 遮罩和 PCA 遮罩會選出不同的
region，因此是不同的 tile 集合，因此是不同的 label。

**一個尚未解決的落差**：遮罩在 ds=14，而 `tissue_ratio` 的檢查
（`has_tissue(x, y, w, h, ds)`）是在 level-0 座標上換算過去的。ds=14 的遮罩比 ds=32
的細，`tissue_ratio=0.75` 在細遮罩上比在粗遮罩上**更難通過**——粗遮罩會把小空隙
一起算成組織。所以 6.5 的那支探針必須用**實際要用的遮罩**跑，不能拿舊 log 裡
HSV 遮罩的數字外推。舊 log 的數字只說明機制，不說明我們會拿到幾張。

### 6.2 新的

**`DsLadder`** — 把一個 ds 階梯解析成每片 slide 的讀取計畫。

```python
@dataclass(frozen=True)
class DsLadder(IdentifiedConfig):
    rungs: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)

    def plan(self, wsi) -> List[RungPlan]:   # RungPlan(rung_ds, level, level_ds, resize)
```

規則：對每個 rung，挑**「ds ≤ 目標的最粗那一層」**，讀進來之後再降採樣到目標。
永遠不上採樣——上採樣不會憑空生出解析度，只會讓網路學到插值產生的假紋理。

`SafeSlide` 現有的兩個選擇器都不是這個：

- `nearest_level_for_downsample`（`SafeSlide.py:333`）取比值最近的，兩邊都可能
- `coarser_level_for_downsample`（`:349`）取「≥ 目標 ds 的最細層」，也就是**偏粗**
  那一側。在 4x 金字塔上問 ds=2 會拿到 level 1（ds=4），比想要的粗，得上採樣

兩者都有各自的理由（後者的 docstring 附了 1398 shots 的量測，說明對 retrieval
而言偏粗和偏細不對稱），但它們回答的是別的問題。`DsLadder` 需要自己的
`finer_level_for_downsample`。**這是新寫的，不是重用**，寫的時候把上面這段理由
放進 docstring，否則下一個人會以為漏用了現成的東西。

**`MppStack`** — co-registered stack 的取樣與讀取（Stage B）。`TileSampler` 各
level 獨立取樣，沒有東西在做「同一個 level-0 中心、每階各一張」。

**`WsiTileDataset` / `HomographyPairDataset`** — 前者一個 rung 的 tile + 它的 HA
label；後者把前者包起來，每次吐 `(img, warp(img, H), H, valid_mask)` 給
descriptor loss 用。

### 6.3 HA label 怎麼存

路徑與規矩照 `utilities/FeatureStore.py`：

```
result/cache/keypoint_labels/<wsi_stem>__ds<d>__<cfg8>.safetensors
```

`result/cache/` 而不是 `result/<job>/`：這是好幾個 job 都會讀的昂貴中間產物，
不是任何一次執行的副產品，而且 `make clean-job JOB=cache` 就是它的清除方式
（ClaudeRules §6）。

**存 sparse，不存 dense。** 256x256 的 fp16 機率圖是 128 KB/tile；一片一階一萬張
tile 就是 1.3 GB，而 1024 的 tile 是 16 倍。改存每張 tile 的稀疏點集（M 見下）：

| 張量 | 型別 | 說明 |
|---|---|---|
| `tile_x, tile_y` | int32 `[T]` | level-0 左上角 |
| `kp_xy` | int16 `[T, M, 2]` | tile 內座標 |
| `kp_score` | float16 `[T, M]` | `mean_prob` |
| `kp_count` | uint8 `[T, M]` | 有幾個 homography 看得到這個位置 |
| `n_kp` | int16 `[T]` | 實際點數，不足 M 的部分是 padding |

`kp_count` 一定要存，理由見 3.1 節末。

**M 是密度，不是計數。** 6.5 決定了三個 tile_size 各訓一個 model、各自抽、各自跑
HA。若 M 是固定的 1000，`model_1024` 的 tile 和 `model_256` 的 tile 存一樣多的點，
密度就差 16 倍——而三個 model 的比較會直接被這個差異污染。所以存的量由
`points_per_megapixel` 決定，`M = density * H * W`，固定計數只當記憶體上限。

**M 決定的是兩件事，第二件比較容易漏：**

1. **store 對閾值是不是無損的。** M 一旦截斷，事後就不能再調高或調低分數閾值，
   只能整批重跑。
2. **student 被訓練成多密的 detector。** 每個留下來的點都成為 detector CE 的一個
   正樣本 cell，其餘全是 dustbin。M 太大，`mean_prob` 尾巴上的雜訊點變成正樣本，
   student 學會到處開火；M 太小則學到一個人為稀疏的 detector。M 不是一個儲存參數，
   它是**訓練目標的密度**。

所以它跟上游的 `detection_threshold`（0.005）是切同一份清單的兩把刀。作法：**用
閾值切，記下 `n_kp`，M 只當上限**。第一次跑把 `n_kp` 的分布畫出來——分布若在遠低於
M 處有膝點，那就是真實密度；若一路貼著 M，就是在截斷（ClaudeRules §8）。

載入時照 `FeatureStore.load(require=...)` 的規矩：對不上就拒絕，不 fallback。

### 6.4 homography 取樣做了哪些事，以及怎麼單獨看

#### 它不是一串影像運算

`query_sim/augment/` 底下每個 augment 都是一個獨立的影像函式，`apply_rotation`、
`apply_vignette` 各做各的，`cli/demo.py` 因此可以一格畫一個。**homography 不是
那樣。** 它對一個四邊形的四個角做四次擾動，最後解出**一個** 3x3 矩陣
（`superpoint/models/homographies.py:152-230`）。所以「單獨 demo 一個操作」的意思
是把另外三個布林開關關掉，不是抽出一個函式來單獨呼叫。

順序固定且不可交換：

| # | 操作 | 做什麼 | 參數 |
|---|---|---|---|
| 0 | 基準四邊形 | 置中、邊長 `patch_ratio` 的正方形。`margin = (1 - patch_ratio) / 2` 是四個角能動的餘裕 | `patch_ratio` |
| 1 | `perspective` | 三個 truncated normal：左兩角共用一個 x 位移、右兩角共用另一個、上下一個 y 位移且**左右邊符號相反** -> 梯形。affine 剪切也在這一項裡 | `perspective_amplitude_x/y` |
| 2 | `scaling` | 抽 `n_scales` 個 `TN(1, amp/2)`，前面插一個 1.0；以四邊形**自己的重心**縮放；濾掉出界的候選，**隨機挑一個** | `scaling_amplitude`, `n_scales` |
| 3 | `translation` | 由四角到邊界的餘裕算出範圍，x/y 各 uniform 抽一個位移 | `translation_overflow` |
| 4 | `rotation` | `n_angles` 個角度線性分布在 `[-max_angle, max_angle]`，前面插一個 0；繞重心轉；同樣濾掉出界的，**隨機挑一個** | `max_angle`, `n_angles` |
| 5 | 解 | 四點對應 -> `matrix_solve_ls` -> 8 參數 | — |

#### 三個會安靜抄錯的地方

**`allow_artifacts` 不只是「允許超出邊界」。** 在 scaling 與 rotation 兩個分支裡，
它同時把候選清單的第 0 項排除掉（`homographies.py:184`、`:213`），而第 0 項正是
前面插進去的「不縮放」與「不旋轉」。所以開了它就**強制每一次都縮放且旋轉**。
關掉它則是「濾掉會出界的候選，剩下的隨機挑」，其中包含不動。兩者不是強弱之分，
是不同的分布。

**H 的方向是 output -> input。** docstring 明寫「As in `tf.contrib.image.transform`,
it maps the output point (warped patch) to a transformed input point (original
patch)」。`cv2.warpPerspective` 的預設方向相反。這是 `Camera.output_to_level0`
被咬過的那一類錯誤——在對稱的情況下看不出來，不對稱時整批資料靜靜地錯。
`common/Homography.py` 的每個公開函式都必須在簽名旁邊寫明方向，且
`test_homography` 對兩個方向各驗一次。

**`patch_ratio` 不會內建一個縮放。** 四個開關全關時 `pts2 == pts1`，H 是單位
矩陣。它控制的是四個角有多少餘裕可以動：0.85 幾乎沒有餘裕，所以上游 export
config 必須同時開 `allow_artifacts: true`（`magic-point_coco_export.yaml:25-26`），
否則有效候選會常常是空的。

#### 對這個專案的一個後果

scaling 的實際幅度是 `TN(1, scaling_amplitude/2)` = `TN(1, 0.1)`，也就是大約
±10~20%，**遠小於一階 ds（2x）**。所以「同 rung 的 HA」確實留在該 rung 之內，
Stage B「一個 keypoint 在鄰近 rung 的存亡」這個問題才有意義。

這句話不是自明的，而且它是可以被打破的：把 `scaling_amplitude` 調到 0.7 以上，
HA 的尺度抖動就會跨過半個 rung，Stage A 的 label 開始混入相鄰解析度的答案，
而 Stage B 量到的存活率會無來由地上升——一個看起來像好消息的錯誤。所以
`scaling_amplitude` 的上限由 ds 階梯的間距決定，這條關係寫進
`common/Homography.py` 的 docstring。

#### `cli/demo_homography.py`

形狀照 `query_sim/cli/demo.py`：一張多格圖，每格一個操作，加上參數值與其抽樣
範圍的並排列印（`_print_capture_params` 那個作法）。

不需要模型，不需要 GPU，秒級。這是「昂貴的執行之前那個可以先看一眼的東西」。

```
python training/SuperPathPoint/cli/demo_homography.py <wsi> --x X --y Y --ds D [--seed S]
```

格子：

| 格 | 內容 |
|---|---|
| `original` | 原 tile，畫上基準四邊形 `pts1` |
| `perspective only` | 只開 `perspective`，其餘三個關 |
| `scaling only` | 只開 `scaling` |
| `translation only` | 只開 `translation` |
| `rotation only` | 只開 `rotation` |
| `all off` | 四個全關。**應該和 original 完全相同**——這一格是給 `patch_ratio` 的誤解用的 |
| `all on (production)` | 生產設定，四個全開 |
| `valid mask` | 侵蝕後的 `mask` 與 `count`，看 `valid_border_margin` 切掉了多少 |
| `point-warp check` | 在原圖撒一個規則格點，影像 warp 一次、點 warp 一次，把 warp 過的點畫在 warp 過的影像上 |

最後一格是整張圖裡唯一會抓到錯誤的一格。點若不落在對應的紋理上，就是兩條路徑
（cv2 與 `grid_sample`）不一致或方向反了。這是 `test_homography` 的斷言的
可視版本——測試給是非，這張圖給的是「錯在哪個方向」。

同時列印這次抽到的值與其範圍：

```
patch_ratio        = 0.85
perspective        h_left=-0.021  h_right=+0.014  disp=+0.033  | amp_x/y = 0.2
scaling            picked 1.0873   from 6 candidates, 4 valid  | TN(1, 0.10)
translation        dx=+0.012 dy=-0.008                          | overflow = 0.0
rotation           picked -37.5 deg from 26 candidates, 18 valid| max_angle = 180 deg
allow_artifacts    True   (so candidate 0 -- no-op -- was excluded from scaling and rotation)
```

「6 個候選裡 4 個有效」這種數字要印出來，因為它是 `allow_artifacts` 與
`patch_ratio` 交互作用的唯一可見證據。有效候選數若常態是 1，那麼「隨機挑一個」
其實沒有在隨機。

#### `cli/demo_ha.py`

需要 teacher，所以走 jobscript。畫一張 tile 的 HA 疊加過程：原圖 / 6 個抽樣的
warp 與各自的偵測 / 累積的 `mean_prob` / `counts` 圖 / 最後 top-M 的點畫回原圖。

`counts` 那一格是重點：它應該在中央高、邊緣低，而且邊緣的低是連續漸變不是斷崖。
若是斷崖，`valid_border_margin` 的侵蝕做錯了。

### 6.5 訓練資料

#### 片子

3 片 HE + 3 片 Ki67，取自這個 repo 其他地方已經在用的那一組：

| 資料集 | 片 |
|---|---|
| BRACS（HE, SVS） | `BRACS_1228`、`BRACS_1476`、`BRACS_1936` |
| Ki67（MRXS） | `S1104233`、`S1104360`、`S1151088` |

Ki67 現有四片，這裡要三片。**捨棄 `S1137178`**，理由是它是目前僅知的兩片有掃描
空洞的玻片之一（`log/TODO.log` 2026-08-22 那條）。空洞在 `read_region_rgb` 之下
會被填成背景色，形成一塊高對比的人工邊界——對 keypoint detector 而言那是完美的
角點，會產生大量高分假點，而且看起來很像模型學會了。這是刻意的取捨，不是隨手挑：
空洞是真實存在的，之後要處理它就用 `SafeSlide.read_region_valid`（`:386`）拿有效
遮罩，把無效區當 HA 的 invalid 一起遮掉。那是另一次的事。

#### 階梯：按 ds 取樣，不按 native level

實測的金字塔（從既有 SLURM log 撈出，不是估的）：

| 資料集 | levels | mpp | 等效 ds |
|---|---|---|---|
| BRACS | 4 | 0.252 / 1.008 / 4.031 / 8.062 | **1, 4, 16, 32** |
| Ki67 | 10（可用 0-5） | 0.243 / 0.485 / 0.970 / 1.941 / 3.881 / 7.763 | **1, 2, 4, 8, 16, 32** |

Ki67 的 level 6 以上取不到樣（`[SKIP] Level 7: no region fits tile_size=256`，
level 6 是 `0/100 after 500 tries`），所以可用階數的上限是 ds=32。

**BRACS 沒有 ds=2 這一階。** 若照 native level 各取 500，mpp 約 0.5 與約 1.9 這兩階
**只有 Ki67 供料**——而 Ki67 是 DAB 棕色、BRACS 是 H&E 粉紫。解析度軸和染色軸在
訓練集裡變成相關的，模型在那兩階學到的任何東西都和棕色糾纏在一起，而且沒有任何
東西會報錯。

所以取樣的軸是 **ds 階梯**，不是 native level：

```
ladder = (1, 2, 4, 8, 16, 32)
```

缺的階（BRACS 的 2 與 8）由 `DsLadder` 讀細一層再降採樣（見 6.2）。兩個資料集因此
在每一階都供料。

這同時是 Stage B 需要的形狀。若用 native level 當階梯，`j=+1` 在 4x 金字塔是 x4、
在 2x 金字塔是 x2，兩個資料集問的不是同一個問題——相對階梯的整個好處就沒了
（見 3.3）。

#### `tile_size` 三種，`tissue_ratio` 0.75，而這張表不是矩形

`tile_size` 用 256 / 512 / 1024 三種，`tissue_ratio` 取 0.75。

**`tile_size` 和 `ds` 是相乘的。** mask 要容納的是 level-0 footprint =
`tile_size * ds`，而這正是 `TileSampler._sample_level` 的拒絕取樣在檢查的東西。

| tile | ds=1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 256 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
| 512 | 512 | 1024 | 2048 | 4096 | 8192 | **16384** |
| 1024 | 1024 | 2048 | 4096 | 8192 | **16384** | **32768** |

實測的牆（`tissue_ratio=0.5`，`max_tries=500`，tile 256）：

| footprint | 觀察 | 出處 |
|---|---|---|
| 4096 | BRACS L2 `only sampled 90/100 after 500 tries` | BenchMarkV2 log |
| 8192 | BRACS L3、Ki67 L5 都 100/100 | 同上 |
| 16384 | Ki67 L6 `0/100 after 500 tries` | RealTest_uni2 log |
| 32768 | Ki67 L7 `[SKIP] no region fits tile_size=256` | 同上 |

所以粗體那三格在 `tissue_ratio=0.5` 就已經是空的，18 格裡先天只有 15 格有資料。
把 `tissue_ratio` 拉到 0.75 只會讓牆往左移，而且視窗越大移得越多——大視窗更容易
跨到組織邊緣。BRACS 在 footprint 4096 燒完 500 次只拿到 90 張，那還是 0.5 的成績。

順帶一提：4096 拿 90 而 8192 拿 100 是**非單調的**，這份 spec 不解釋它。`_sample_level`
每層都重跑一次 `filter_regions -> merge_overlapping -> filter_patchable` 再撤銷，
所以每層存活的 region 集合不同，這是最可能的原因——但沒有量過，而
「兩個觀察共用一個表面特徵」不等於機制（ClaudeRules §10）。要判定就印出每層存活
的 region 數與面積。

**每格拿得到幾張是量出來的，不是排出來的。** 所以第一件事是一支探針：對每個
`(片, tissue_ratio, tile_size, ds)` 用固定預算跑一次拒絕取樣，只記「拿到幾張、
花了幾次」。它只碰遮罩，不讀 tile、不進模型；格數與它決定的事見下一節。

#### 三個尺寸各訓一個 model，各自抽、各自跑 HA

`tile_size` 不是一個 batch 內的軸，而是**三個獨立的 model**：`model_256`、
`model_512`、`model_1024`。**v1 只做 256**，另外兩個排在 256 的整條線走完之後。

於是上面那張 footprint 表的用途不是「哪一格抽哪個尺寸」，而是**哪個 model 有哪些
rung**——每個 model 拿到它自己的 footprint 容許的最大階數：

| model | 可用 rung | 格數 |
|---|---|---|
| `model_256` | ds 1, 2, 4, 8, 16, 32 | 6 |
| `model_512` | ds 1, 2, 4, 8, 16 | 5 |
| `model_1024` | ds 1, 2, 4, 8 | 4 |

**這就是為什麼不從 1024 裁出 256。** 巢狀裁切看起來省 HA（跑一次 16 個單位，
而不是 1 + 4 + 16 = 21），但從 1024 裁出來的 256 只覆蓋 ds 1-8——獨立抽的 256
覆蓋 ds 1-32。少掉的是最粗的兩階，而那正是 Stage C 的相對存活 label 最有資訊的
地方。省下的 5 個成本單位買不到那兩階。

代價要記下來：三份 label 互相不一致。不同尺寸的邊界侵蝕比例不同（見下），
homography 相對於內容的尺度也不同。**所以三個 model 之間的比較必須在固定的評估
協定下做**，否則「哪個 tile_size 好」和「label 噪音」分不開：

```
同一批 held-out 位置、同一個 tau、真實照片走 1440x1024 原尺寸
（1440/8 = 180、1024/8 = 128，兩邊都是 8 的倍數，全卷積網路直接吃）
```

#### rung 平衡是一個開關，不是一個現在要做的決定

探針（12 節第 3b 步）跑 `tissue_ratio` 兩個值：

```
2 個 ratio (0.5, 0.75) x 3 個 tile x 6 個 ds x 6 片 = 216 格
```

只碰遮罩，秒級。產出一張 `(片, ratio, tile, ds) -> 拿到幾張 / 花了幾次` 的表。

兩個值都跑而不是先挑一個，理由在 6.1 末段：0.75 在 ds=14 的 PCA 遮罩上比在舊 log
那種 ds=32 的 HSV 遮罩上更難通過，而那個差距沒有量過。舊 log 的數字說明機制，
不預測我們會拿到幾張。

平衡政策做成 config 開關，兩個模式：

| 模式 | 做什麼 | 代價 |
|---|---|---|
| `align-min` | 每格都只取 `min(各格張數)` | 完全平衡、最乾淨，但可能丟掉九成資料 |
| `loss-weight` | 拿多少用多少，detector CE 按 rung 加權，權重正比於 `1/張數` | 資料全用，但粗階少數樣本的噪音被同步放大 |

**探針的結果決定選哪個，不是反過來。** 最差的格若有 300 張，`align-min` 便宜；
若只有 40 張，那就得是 `loss-weight`。開關存在的理由就是「先跑再決定」——現在挑
一個等於在還沒看到分布之前就挑了閾值（ClaudeRules §8）。

不做任何處理也是一個選項，但它是最壞的一個：ds=1 若有 500 張而 ds=32 只有 40 張，
細階的訓練量就是粗階的十幾倍，而**沒有任何東西會顯示這件事**。

#### 探針跑了兩次，答案相反，而第二次是對的

**2026-08-26，六片、216 格**（`2 ratio x 3 tile x 6 ds x 6 片`）。tile 256、每格
要 500 張，六片裡最差那片：

| ratio | ds 1 | ds 2 | ds 4 | ds 8 | ds 16 | ds 32 |
|---|---|---|---|---|---|---|
| 0.50 | 500 | 500 | 500 | 500 | 500 | **353** |
| 0.75 | 500 | 414 | 320 | 224 | **87** | **27** |

照這張表，四片訓練片的 ds 32 有 1784 張、其餘各 2000。上面那條**事先寫好的規則**
說「最差的格有幾百張就用 `align-min`」，於是選了 `align-min`——砍齊丟 9%。

**那個 1784 不是玻片說的，是預算說的。** `tissue_ratio` 的閘門只放行背景 <= 50%，
取樣器接著對一批已經被過濾過的位置要 500 個。1784 量到的是**拒絕預算多快用完**，
不是一個 rung 上有幾個互不重疊的位置。這兩件事在細階幾乎一樣，在粗階差一個數量級。

**2026-08-27，十二片、216 格**（`12 片 x 3 tile x 6 ds`，ratio 軸拆掉了）。這次記的
是 `n_admissible`——整個候選池本身，而不是預算的殘骸。tile 256、每片：

| | ds 1 | ds 2 | ds 4 | ds 8 | ds 16 | ds 32 |
|---|---|---|---|---|---|---|
| 最好的片 | 69,011 | 18,853 | 4,908 | 1,257 | 317 | 80 |
| 最差的片 | 17,807 | 4,718 | 1,262 | 318 | 86 | **18** |
| 十二片合計 | 461,141 | 123,624 | 32,474 | 8,465 | 2,242 | **583** |

**最差的格是 18，不是 1784。** `align-min` 把每一格截到最差的格，`6 x 12 x 18 =
1,296` 張——而光是 ds 8 一階就有 8,465 張可用。那不是把階梯弄平，是把階梯刪掉。

**所以：不用 `align-min`，2026-08-27 定案。** 也還不是 `loss-weight`——rung 的權重
是第二個決定，而它要等語料存在之後才問得出口。v1 用 `BALANCE=none`，每一階供給多少
就用多少；不平衡是這次執行**已知並記錄在案的性質**，不是之後的發現。

改變它的是**更多玻片**，不是那個開關：ds 32 大約要 25 片才能達到 12 片在 ds 16 上
的 1,200 張。

#### `tissue_ratio` 同一天退役，因為它和「豐富度桶」是同一個機制

閘門和桶量的是同一個量。`score_background` 就是 `white_fractions`，而
`tissue >= tissue_ratio` 就是 `background <= 1 - tissue_ratio`。在定案的 0.5 上，
閘門在任何一個桶看到候選之前，就已經把背景高於 50% 的全部刪光——於是最粗的兩桶
永遠是 0%，而那個 0% 讀起來像「玻片上沒有這種組織」，其實是「閘門不讓它進來」。
2026-08-26 的語料回來是 475/500，而那個數字被當成玻片的性質讀了。**兩個機制量同一
件事而互相不知道，就是那個 bug 的形狀。**

取代它的是**七個桶，各帶一個下限與一個上限**（`utilities/TileSampler.RichnessConfig`）：

| 桶 | 背景比 | 下限 | 上限 | 配額 |
|---|---|---|---|---|
| `bg00_15` | < 15% | 5% | 15% | 15% |
| `bg15_30` | 15 - 30% | 15% | 25% | 25% |
| `bg30_50` | 30 - 50% | 50% | 60% | 60% |
| `bg50_70` | 50 - 70% | — | 20% | 0 |
| `bg70_85` | 70 - 85% | — | 20% | 0 |
| `bg85_95` | 85 - 95% | — | 0 | 0 |
| `bg95_100` | > 95% | — | 0 | 0 |

**下限與上限是兩種不同性質的東西，所以是兩個 tuple 不是一個。** 上限一定達得到
——不拿就是了；下限達不達得到，取決於玻片上真的有沒有那麼多。未指派的 30% 平均分給
三個**有下限**的桶，於是配額剛好加總到 1.0。後兩桶的上限存在的意義是讓**填不滿的
缺口有地方去**，而缺口是唯一會抵達它們的東西。

零上限說的正是閘門說的話，而且只說一次：`bg85_95` 與 `bg95_100` 是 0，就是一道
「背景 85%」的閘門。

三條算術護欄，中間那條就是 475/500 本身：

```
sum(floors) <= 1      否則沒有任何一階能同時滿足所有下限
sum(caps)   >= 1      否則這一階「由建構決定」就是短的
floors <= caps        逐元素
```

舊合約通過中間那條時一分餘裕都沒有——`0.85 + 0.15 = 1.00`——這就是細階停在 85:15
不動、讀起來卻像一次供給量測的原因。

`floor_frame` 是一個開關。`'ask'`（預設）把下限當成 `n_per_rung` 的比例：留住張數，
讓混合比例漂移。`'taken'` 把下限當成「做得到的量」的比例，整階縮到
`n_goal = min(supply_b / target_b)`：混合比例準，張數少。v1 用 `'ask'`。

**這些值寫在程式碼的 `RichnessConfig` 預設裡，jobscript 只把它印出來當表頭。**
`ExtractPreTiles.sh` 已經沒有 `TISSUE_RATIO` 這個變數，`--tissue-ratio` 被
`ap.error` 指名拒收；`TrainSuperPathPoint.sh` 的 `BALANCE=none` 帶著上一節第二張表
當理由。

`tissue_ratio` 這個名字在別處還活著，而那些是別的東西：
`TissuesRegionsMask.has_tissue(...)` 是一個對矩形的述詞（`query_sim` 的 Camera 用
0.3），`TileSampler.caps_for_tissue_ratio()` 是翻譯層，給 `GigaPathKnnEstiMpp` 的
參考庫與 `bench_gigapath_accuracy` 把舊閘門原封不動地表示成一組上限，行為逐位元
相同。**退役的是取樣策略裡的那一個。**

#### 3c 實際切出來的語料

2026-08-27，`result/cache/tiles/`，12 片 x 6 階 = 72 格，每格要 100 張：

```
ds 1  1200 | ds 2  1200 | ds 4  1200 | ds 8  1200 | ds 16  1117 | ds 32   471
```

合計 **6,388 張、4.4 GB**。ds 32 的 471/1200 是上面那道 583 的牆，不是預算——
探針說 12 片合計 583 個可入位置，取樣器拿走了幾乎全部。

落在七個桶上：

| 桶 | 張數 | 佔比 | 配額 |
|---|---|---|---|
| `bg00_15` | 890 | 13.9% | 15% |
| `bg15_30` | 1,490 | 23.3% | 25% |
| `bg30_50` | 3,485 | 54.6% | 60% |
| `bg50_70` | 308 | 4.8% | 0 |
| `bg70_85` | 215 | 3.4% | 0 |

前三桶各差配額 1 到 5 個百分點，差額 8.2% 全部流進兩個溢流桶——**那正是溢流桶存在
的理由**，而且它精確地指出缺口在粗階。這裡要注意的是它**不再是一個沈默的短缺**：
`bg50_70` 和 `bg70_85` 有數字，就是「前三桶填不滿、缺口按 `spill_order` 找到有非零
上限的桶」的可讀證據。舊閘門下這兩格永遠是 0，短缺只會表現成 475/500。


#### `valid_border_margin` 在不同尺寸下切掉的比例不同

`valid_border_margin=3` 在 256 tile 上侵蝕掉約 4.6% 的面積，在 1024 上約 1.2%。
所以大尺寸的 HA label 邊界損失比例較小。

這不是錯誤，但它是**三個 model 的 label 不等價的具體來源之一**：`model_256` 的
label 有 4.6% 的面積是被侵蝕掉的，`model_1024` 只有 1.2%。比較三個 model 時若在
各自的 tile 上量 repeatability，這個差異會直接進到數字裡並被讀成「大 tile 比較好」。
上面那個固定評估協定就是為了把它擋掉——**在同一批位置、同一個尺寸上評估，而不是
各自在自己的 tile 上評估**。

#### 切分：按片，不按 tile

| 用途 | 片 |
|---|---|
| train | `BRACS_1228`、`BRACS_1476`、`S1104233`、`S1104360` |
| held-out | `BRACS_1936`、`S1151088` |

**held-out 是完全不參與訓練、也不參與任何調參的片子，只在量測時打開。** 它回答的
是「模型在沒看過的**玻片**上還行不行」。第 1 節的 repeatability 判準都在它上面量。

**為什麼按片切而不按 tile 切**：同一片的 tile 共享染色批次、掃描機、切片厚度與組織
來源。隨機切 tile 的話，held-out 的每一張都有幾千個「同一片的鄰居」躺在訓練集裡，
量到的數字會遠高於真實泛化。這在病理影像是標準的洩漏形式。

**代價：1+1 兩片的估計很吵。** held-out 的 repeatability 差 5% 可能只是那片組織長得
比較特別。**這正是第 1 節所有判準都是「贏過誘餌」而不是「超過某個絕對值」的原因**
——誘餌和模型看同一片、同一批位置，片子的個性在相減時被抵消掉。絕對值會隨片子
飄，差距不會。

**`BRACS_1228` 刻意留在 train。** 它是 `SlideWinTest`、`BenchMarkV2`、
`SlidewinPooling` 都跑過的那片，既有的 SIFT 與 retrieval 數字全在它身上。放在 train
才能拿它當 sanity check——「新 detector 在這片上的行為和舊數字對不對得起來」是一個
隨時可以問的問題，而問它不會污染 held-out。

#### 這個切分量得到什麼、量不到什麼

**量得到：新玻片、已見染色。** 兩種染色都在 train 裡，所以 held-out 上的數字說的是
「換一片沒看過的玻片」。

**量不到：跨染色泛化。** 6.6 要比灰階與 RGB 兩個 student，其中一個動機是
「RGB detector 可能靠染色認點，所以跨染色會更差」。**這個 split 決定不了那件事**
——決定它的那個條件（測試集的染色沒出現在訓練集裡）在這份資料計畫裡不存在
（ClaudeRules §10）。

要量就得加一組 **leave-one-stain-out** arm：3 片 HE 訓、3 片 Ki67 測，以及反向。
它用同一批已抽好的 tile，不必重抽，是加法不是改建。列為具名的後續 arm，v1 不做。
在那之前，任何關於「跨染色」的句子都不准出現在結論裡。

#### tile 先落地，不在訓練時讀 WSI

MRXS 在 DataLoader 的多個 worker 裡邊訓練邊讀是一件會咬人的事：`SafeSlide` 遇到
破損會重開 handle，而重開 MIRAX 要重新解析它的索引（`log/TODO.log` 2026-08-22
量到一次讀取 752 次 reopen）。

所以照 prov-gigapath 的作法：**離線切一次 tile 落地，訓練完全不碰 WSI**
（`gigapath/preprocessing/` 與 `finetune/` 是分開的兩件事，訓練迴圈只讀
預先算好的東西）。落地位置
`result/cache/tiles/<wsi_stem>__ds<d>__t<tile>__<cfg8>/`，和
`KeypointLabelStore` 一對一對齊，同一組 `(wsi_stem, ds, tile)` 索引。目錄名帶
cfg hash 的理由見 `PreTileStore.PreTileMeta.dirname`：`tissue_ratio`、seed 與
**遮罩的 `segmenter_id`** 也決定內容，而那三個在名字裡看不見。

#### 落地的是 pre-tile，所以是 9 倍，而這件事會決定 512/1024 做不做得起

**存的是 pre-tile 不是 tile**（6.6）。`tile` 是訓練時從 pre-tile 中心裁出來的，
不另外存——但 pre-tile 每邊 3 倍，面積就是 **9 倍**。這不是 6.6 的附帶成本，
它是 6.6 的**主要**成本，而先前這張表算的是 tile。

上界（每格 500 張，探針還沒說哪幾格真的拿得到）：

| model | rung 數 | 張數上界 | 存的邊長 | 像素 | 未壓縮 |
|---|---|---|---|---|---|
| **256（v1）** | 6 | 6 x 500 x 6 片 = 18000 | 768 | 10.6 G px | **31.9 GB** |
| 512 | 5 | 15000 | 1536 | 35.4 G px | 106 GB |
| 1024 | 4 | 12000 | 3072 | 113 G px | 340 GB |
| 合計 | | 45000 | | 159 G px | 477 GB |

**量到了，2026-08-27（第 3c 步跑完）：PNG 是原始的 45.1%。**

| | 張數 | 未壓縮 | 落地 |
|---|---|---|---|
| **256（v1，實際）** | **17,784** | 31.5 G | **14.2 GB** |
| 512（按 45.1% 推） | 15000 | 106 G | 約 48 GB |
| 1024（同上） | 12000 | 340 G | 約 153 GB |

比先前猜的「二十幾 GB」好 —— 玻璃背景大片同色，PNG 吃得很乾淨。三個尺寸全做完
約 215 GB，`result/cache/` 承受得住（那裡已經有過 60 GB 的 feature store），但
1024 那一族單獨就 153 GB，不是零頭。

推算用同一個 45.1% 有一個保留：粗階的 tile 組織佔比較低、glass 較多，壓得更兇
（實測 ds 32 那一格 191 MB / 500 張，ds 4 是 402 MB / 500 張，差一倍）。512 和
1024 的 rung 分布和 256 不同，所以上表的推算是上界而不是預測。

**512 和 1024 這樣做不起來。** 340 GB 一個 model 不是「大一點」，是不同的量級。
在抽 1024 之前必須先決定其中一條，而不是抽到一半才發現：

| 出路 | 做什麼 | 代價 |
|---|---|---|
| 減張數 | 粗階本來就抽不到 500，探針會說實際是多少 | 最省事，但省不到一個量級 |
| 降倍率 | 3 -> 2，面積 9 -> 4 倍 | 超出的 draw 有黑邊；6.6 那條斷言會逐張指出是哪些，於是「有多少張真的超出」變成量得到的 |
| 不落地 | 512/1024 改成訓練時從 WSI 讀 | 撞上 6.5 開頭那個 MRXS reopen 的問題，正是這裡先落地的原因 |

v1 不必現在決定——256 這條線走完之前 1024 一張都不會抽。**但它必須在
`extract_pretiles.py` 支援 `--pre-tile-factor` 這件事上先留好位置**，而不是把 3
寫死在抽取程式裡。`PreTileStore` 因此把 `pre_tile_factor` 當成 identity 欄位：
倍率不同的兩批 tile 是兩個資料集，不是同一批的兩個版本。

探針（12 節第 3b 步）跑完之後這些上界會往下修。

---

### 6.6 pre-tile 與中心裁切：訓練資料不能有黑邊

#### 問題，以及它的量測值

`warp_image` 用 `BORDER_CONSTANT, borderValue=0`，所以輸出裡凡是取樣到輸入影像外面的
像素都是純黑。`cli/demo_homography.py` 印出來的第一個百分比就是這件事：

```
seed 0 的生產 draw：  valid 67.8%   coverage 95.1%
```

`valid` 是 warped 畫框裡「真的取樣自輸入」的比例。**三分之一是假的黑色。**

那不是意外，是幾何的必然。對一下同一 draw 的參數：

| 來源 | 這一 draw | 倍率 |
|---|---|---|
| perspective | `h_left -0.0132  h_right +0.0640` -> 寬 0.927 | 1.091 |
| scaling | 1.1304（> 1 是 zoom out，要更多來源） | 1.130 |
| 畫框 / 四邊形 | `1 / patch_ratio` | 1.176 |
| rotation | +15 度，方形 bbox 係數 `cos + sin` | 1.225 |
| | **合計** | **1.78** |

需要 1.78 倍的來源、只給了 1.0 倍，缺 32%。和量到的 67.8% 對得起來。

#### 為什麼黑邊對 keypoint detector 比「浪費面積」嚴重一階

黑邊不是資訊為零的區域，是**一條筆直、最高對比的邊，還帶兩個完美的直角**——角點偵測器
最喜歡的刺激。那些假 keypoint 會拿到高分，經 `H_inv` warp 回原圖，疊進 HA 的平均裡。

上游的緩解是 `mask = valid_mask(shape, H)`：把黑區的 prob 歸零再 warp 回去。但偵測器有
感受野，一個落在有效區內 5 px 的 keypoint 仍然可能是在對 5 px 外的黑邊起反應。
`valid_border_margin=3` 就是那個補償，而 3 px 遠小於 SuperPoint 的感受野。**上游是容忍
這件事，不是解決它。**

#### 為什麼不是改成 BORDER_REFLECT

repo 裡有這個選項的先例：`query_sim/augment/geometry.py:29` 就是 `BORDER_REFLECT`。

但反射會造出**鏡像對稱的假結構**，對角點偵測器可能更糟——接縫上每個特徵都完美對稱，而
對稱正是很多角點度量的極值條件。

`query_sim` 選反射不是選錯：**一張照片的畫框外面什麼都沒有**，它沒有第三個選項。WSI 有。
tile 之外還有真實組織，而那正是 pre-tile 這條路可行的唯一原因。

#### 做法：讀 pre-tile、warp、中心裁切

```
讀   pre_tile x pre_tile 的方塊，以目標 tile 的中心為中心
warp 整個 pre-tile
裁   中央的 tile x tile
```

輸出的每個像素都來自真實組織。

#### 倍率是 3，推導出來的界是 2.49，而且不需要校準跑

「tile 的外接圓直徑」= `sqrt(2)` = 1.414 只涵蓋**旋轉**那一項。四個 op 是相乘的，
以 tile 邊長為 1：

| 來源 | 累積 | 為什麼 |
|---|---|---|
| 四邊形起點 | 0.85 | `patch_ratio` |
| perspective | **1.25** | 寬度 `0.85 + (h_right - h_left)`，兩個獨立 `TN(0, 0.1)` 各在 +-0.2 |
| scaling | **1.50** | `TN(1, 0.1)` 2 sigma 截斷，最大 1.2 |
| rotation | **2.12** | 旋轉後方形的 bounding box，最大 sqrt(2) |
| 畫框 / 四邊形 | **2.49** | `1 / patch_ratio` = 1.176 |

**所以 sqrt(2) 不夠，3 夠。** 它漏掉的是 perspective 撐寬、scaling、以及「畫框比
四邊形大 1.176 倍」——最後這項是 `patch_ratio = 0.85` 的直接後果，最容易被忽略。

#### 為什麼不是取分位數

這裡原本寫著「跑一批 draw、量 bbox 分布、取 99 分位」，那是錯的，而錯在把成本算成
每個 draw 一次讀圖。

**pre-tile 每個位置只讀一次，N = 100 個 homography 全部從它 warp。** 所以它要涵蓋的
是那 100 個裡最大的那個，不是典型的那個。而 100 個抽樣的最大值大致就落在單一 draw
分布的 99 分位——在 N = 100 之下，「取 99 分位」和「取最壞界」是同一件事。

最壞界是推導得出來的，所以校準跑不存在。3 對 2.49 留的餘裕，是給那個推導裡的近似
用的：投影變換嚴格說不是繞中心的等比縮放，`1.176` 那一項只是近似。

`cli/demo_homography.py --calibrate N` 仍然存在，但它是**檢查**不是校準：抽 N 個、
印出需要的倍率分布、斷言沒有一個超過 3。把紙上的推導變成量到的敘述，成本是秒。

#### 精確範圍算得出來，不必猜

需要的來源區域就是輸出畫框四角的原像，而那個函式已經在庫裡：

```python
points_output_to_input([[0, 0], [0, tile], [tile, tile], [tile, 0]], H)  ->  bbox
```

`--calibrate` 就是對這個取分布。逐 draw 精確讀取也是可行的形狀，但 HA 的成本結構
否決了它：讀圖是每張 tile 一次，warp 是 100 次。

#### 一條免費的斷言掉出來

pre-tile 夠不夠大，不必用看的驗證：

```
valid_mask(shape, H) 在中央 tile x tile 的區域上必須全部為 True
```

不是的話就是這一 draw 超出了 pre-tile。這把「倍率選對了嗎」從一個要事後查圖的疑問，變成
一個**每次 draw 都在跑、成本為零**的檢查。

順帶一個可觀察的後果：pre-tile 生效之後 HA 的兩張遮罩會不對稱——`mask`（warped 畫框內
的有效區）趨近全 1，而 `count`（warp 回來在原畫框的覆蓋）**仍然有意義且仍然需要**。上面
那組 67.8% / 95.1% 應該變成大約 100% / 95%，而那個不對稱本身就是它生效的證據。

#### 它會動到 rung 的可達性，除非把兩件事分開

6.5 那道實測的牆是 footprint = `tile x ds`：8192 可以、16384 是 0/100、32768 連 region
都放不下。

若 `tissue_ratio` 的檢查套在 **pre-tile** 上，footprint 變成 `2.9 x tile x ds`：

```
tile 256 @ ds 32  ->  256 x 2.9 x 32 = 23757     過牆，死
tile 256 的上限   ->  8192 / (256 x 2.9) = ds 11  ->  只剩 rung 1, 2, 4, 8
```

**掉最粗的兩階**——正是 Stage C 的相對存活 label 最有資訊的地方。

所以兩件事分開：

- **`tissue_ratio` 套在 tile 的 footprint 上。** 我們訓練的是那個 tile，要求它是組織是對的。
- **pre-tile 只是 warp 的上下文**，不需要是組織，只需要讀得到。

這樣可達性表完全不變。代價是：靠近組織邊緣或玻片邊緣的位置，pre-tile 會被裁到，那些 draw
仍有黑邊——那是少數，而且上面那條斷言會逐張指出是哪些。

#### 落實的形狀：`T @ H`，而且網路只看 tile

寫 `HomographicAdaptation` 的時候有兩個做法，成本差 9 倍：

| 做法 | 網路吃到的 | 每個 view 的成本 |
|---|---|---|
| warp 整個 pre-tile，之後裁中心 | 768 | **9x** |
| 對 tile 取樣 H，從 pre-tile 讀來源 | 256 | 1x |

採第二個。`sample_homography` 拿 tile 的 shape，所以每一個被記錄、warp、求逆的座標
都在 **tile 座標系**；讀圖時才左乘一個平移：

```
H          output(tile) -> input(tile)        取樣出來的那個
T @ H      output(tile) -> pre-tile 像素座標   cv2 實際讀的那個
```

`T` 只換輸入的座標系，不動幾何。**左乘**是關鍵：右乘會變成平移輸出畫框，warp 到
tile 的另一個位置，而且看起來完全合理。

同一個 pre-tile 也讓 `mask` 的算法變了：不能再用 `valid_mask(shape, H)`，那是在問
「來源和輸出一樣大」時的有效性，會把三分之二的合法視角判成無效。改成把 pre-tile
自己那麼大的一張全 1 warp 過同一個 `T @ H`——於是問的是「實際拿得到的那些像素」。
`erode_valid` 就是為此從 `valid_mask` 裡拆出來的（第 14 節）。

#### 這改的是哪裡，不改的是哪裡

| | |
|---|---|
| **不改** `common/Homography.py` | `warp_image` 的 `BORDER_CONSTANT` 是原語該有的行為，黑邊是它誠實的輸出 |
| **不改** `cli/demo_homography.py` | demo 的工作是展示這個 op 實際做什麼，黑邊看得見才對 |
| **改** 資料模組 | `Datasets.py` 的 `HomographyPairDataset`、`HomographicAdaptation.py` |
| **改** 落地格式 | `result/cache/tiles/` 存 **pre-tile**，第 12 節 3c 抽的是 pre-tile |

現在是最便宜的時機：12 節才走到第 3 步，**一張 tile 都還沒抽**。這件事現在只花一次 spec
編輯；抽完之後才發現，就是整批重抽。

## 7. Loss 模組

### 7.1 detector

上游 `superpoint/models/utils.py:54-72`，逐步照抄，因為每一步都有理由而且都不會
報錯：

1. 稠密的 0/1 keypoint map 用 `space_to_depth(grid_size)` 折成 `[N, cell**2, Hc, Wc]`
2. `labels = concat([2 * labels, ones], axis=channel)` — 真 keypoint 乘 2，再接一個
   全 1 的 dustbin 通道，於是任何真 keypoint 都贏過 dustbin
3. 加 `U(0, 0.1)` 的雜訊 — 空 cell 裡 64 個通道全是 0，argmax 會固定挑 0 號；
   雜訊讓它隨機挑，否則模型會學到「沒有點時就報左上角」
4. `argmax` 取出 65 路的類別索引
5. `cross_entropy(logits, target, weight=valid_mask)`

`valid_mask` 同樣用 `space_to_depth` 折疊後**取 AND**（每個 cell 的 8x8 像素全部
有效才算有效），不是取 OR、也不是取平均。

第 3 步的雜訊是最容易在重寫時掉的一步，掉了不會報錯，只會讓空白區域長出一堆
規律的假點——看起來像模型的性質。

### 7.2 descriptor

上游 `models/utils.py:75-145`。cell 中心經 homography 映射，`s = 1[距離 <= cell - 0.5]`
當正樣本指示，對所有 cell 對算 hinge：

```
positive_dist = max(0, positive_margin - dot)
negative_dist = max(0, dot - negative_margin)
loss = lambda_d * s * positive_dist + (1 - s) * negative_dist
```

上游用的是 **dense** 版本（全部 `Hc*Wc` x `Hc*Wc` 對），不是抽樣版。在 256x256 /
cell 8 上是 32x32 = 1024 個 cell，1024^2 = 1M 對，可以接受。若 tile 加大到 512
就是 16M 對，那時要換 sparse 版——先量再換，不要預先最佳化。

### 7.3 resolution（Stage C，新的）

```
loss = BCEWithLogits(pred[labeled], alive[labeled], pos_weight=w)
```

`w` 每一階各一個，由該階實測的存活率決定，不是猜的。遮罩只留存亡表有 label 的
位置——沒被問過的位置不是負樣本。

### 7.4 加總

```
L = detector(I) + detector(warp(I)) + lambda_loss * descriptor + lambda_res * resolution
```

前三項的形狀與權重照上游 `super_point.py:73-92`。`lambda_res` 是新的，第一版設成
讓兩項 loss 在第一個 epoch 結束時量級相當，並把那個量級記進 TODO.log。

---

## 8. 模型與訓練程序

### 8.1 run 的落地

形狀照 GMR-Conv（`main.py:368-395`、`674-676`、`790-795`），身份層照 LocaScope：

```
result/<SLURM_JOB_NAME>/<identity_id>/
    config/train_config.json     # cfg_json(identity_parts, provenance)
    ckpt/last.ckpt
    ckpt/best.ckpt
    logs.json
```

目錄名用 `identity_id` 而不是時間戳。時間戳讓兩次相同設定的執行分開，
`identity_id` 讓兩次**不同**設定的執行分開——後者才是會安靜出錯的那一種。時間戳
進 `provenance`。

checkpoint 的內容：

```python
{'state_dict', 'optimizer', 'scheduler', 'cur_ep', 'identity_json'}
```

前四個照 GMR-Conv。`identity_json` 是加的，而且是關鍵：Stage B 的存亡表必須說得
出「我是哪個 detector 產的」，否則換了 detector 之後兩張表混在一起，量的是不同
的東西。這是把 `ConfigIdentity` 的紀律套到一個**訓練出來的**模型上——config 說
要建什麼，權重說建出了什麼，而 finetune 正是這兩者分岔的地方
（`ConfigIdentity.py:253` 的 docstring 已經寫了這句）。

### 8.2 wandb

**這會是本 repo 第一個用 wandb 的東西**，目前它只躺在 `environment.yml:413`
（0.26.0）。

```python
if not args.dev and rank == 0:
    logger = wandb.init(project='SuperPathPoint',
                        name=f'{job_name}_{identity_id}',
                        config=json.loads(model.identity_json()))
```

`--dev` 關掉，照 GMR-Conv `main.py:396-405`。**不設 offline mode**，理由和
`jobscripts/_env.sh` 不設 `HF_HUB_OFFLINE` 一樣：安靜的離線比大聲的連線失敗難查。
計算節點連不出去的話，`WANDB_MODE=offline` 加進 `jobscripts/_env.sh`，一個地方
改完全部。

記什麼：

| 頻率 | 內容 |
|---|---|
| 每個 step | `train/loss` 與它的每一個分項、`train/lr` |
| 每個 epoch | 驗證集的 repeatability@τ、localisation error、每張圖的平均 keypoint 數 |
| 每個 epoch | 誘餌：同密度隨機點的 repeatability |

誘餌逐 epoch 記，是因為「repeatability 上升」在點數同時上升時毫無意義——點越多
越容易重複命中。兩條線一起看才讀得出東西。

### 8.3 優化器與排程

上游是常數 LR 的 Adam（`base_model.py:212`），SuperPoint 階段 `1e-4`，
50000 / 18000 / 600000 iter。那些數字是自然影像的，抄過來當起點，不當結論。

AMP 照 prov-gigapath 的做法（`finetune/training.py:56,125,181`）：手動
`GradScaler` + `autocast`。DDP 第一版不做——prov-gigapath 的 finetune 也沒有，
而這裡的瓶頸更可能在 WSI 讀圖而不是在卡上。

### 8.4 環境限制

torch 2.3.0+cu121、opencv 4.13、timm 1.0.28、safetensors 0.7.0、wandb 0.26.0。
**沒有 kornia**。所以：

- HA 的 label 生成走 cv2（CPU），和 `query_sim/augment/geometry.py` 同一套
- 訓練時的 homography warp 走 `F.grid_sample`（GPU）

兩條路徑必須一致，這是第 11 節第一個測試要檢查的東西。

---

## 9. 上游常數對照表

每一格都能對回 `/work/u26130998/SuperPoint/` 的 file:line。憑記憶抄常數是這個
專案吃過虧的地方（ClaudeRules §13 的 CONCH mean/std 案例），所以這張表就是規格
本身。

### HA（`configs/magic-point_coco_export.yaml:12-26`）

| 參數 | 值 |
|---|---|
| `num` | 100 |
| `aggregation` | `'sum'`（實際是 coverage-weighted mean，見 3.1） |
| `valid_border_margin` | 3 |
| `filter_counts` | 0 |
| `patch_ratio` | 0.85 |
| `perspective_amplitude_x/y` | 0.2 |
| `scaling_amplitude` | 0.2 |
| `allow_artifacts` | true |
| `translation/rotation/scaling/perspective` | 全 true |

### `sample_homography`（`models/homographies.py:117-230`）

| 參數 | 預設 |
|---|---|
| `n_scales` | 5 |
| `n_angles` | 25 |
| `max_angle` | pi（export config），函式簽名預設是 pi/2 |

演算法與三個會安靜抄錯的地方見 **6.4**。這裡只放常數，那裡放語意——一份規格裡
同一件事描述兩次，遲早會有一份先被改。

### 模型（`superpoint_pytorch.py:70-102`）

| 參數 | 值 |
|---|---|
| `channels` | `[64, 64, 128, 128, 256]` |
| `stride` | `2 ** (len(channels) - 2)` = 8 |
| backbone 實際層數 | **`channels = [1, *conf.channels[:-1]]` = `[1,64,64,128,128]`**（`superpoint_pytorch.py:83`），所以是 4 個 stage、輸出 **128** 通道 |
| detector head | `VGGBlock(128, 256, 3)` -> `VGGBlock(256, stride**2+1 = 65, 1, relu=False)` |
| descriptor head | `VGGBlock(128, 256, 3)` -> `VGGBlock(256, 256, 1, relu=False)`，L2 正規化 |
| `descriptor_dim` | 256 |
| `nms_radius` | 4 |
| `detection_threshold` | 0.005 |
| `remove_borders` | 4 |

**`conf.channels` 的最後一個 256 不是 backbone 的輸出寬度，是 head 的隱藏寬度。**
`channels[:-1]` 把它切掉了，backbone 只到 128，而兩個 head 各自把 128 升到 256 再
出去。照著 `channels` 直觀讀會做出一個多一層、寬一倍的 backbone——不會報錯，只會
是另一個模型。`stride` 也是從 `len(conf.channels)` 算的，所以動 `channels` 的長度
會同時動 stride 與 dustbin 的通道數。

**輸入是 1 channel 灰階。** 見第 13 節。

descriptor 的取樣方式：PyTorch 版把稠密圖正規化一次，然後在 keypoint 的確切位置
用 `grid_sample` 雙線性取樣（`superpoint_pytorch.py:11-22`，座標映到 `[-1,1]` 並
`+0.5` 對齊像素中心，`align_corners=False`）。TF 版是整張圖上採樣。**採 PyTorch
版**，比較省而且對齊語意明確。

### Loss（`super_point.py`、`configs/superpoint_coco.yaml`）

| 參數 | 值 | 出處 |
|---|---|---|
| `positive_margin` | 1 | `superpoint_coco.yaml:43` |
| `negative_margin` | 0.2 | `superpoint_coco.yaml:44` |
| `lambda_d` | 0.05 | `superpoint_coco.yaml:42` |
| `lambda_loss` | 10000 | `superpoint_coco.yaml:45` |
| detector 權重 | identity 與 warped 各 1 | `super_point.py:73-92` |

`lambda_loss` 是 10000 這件事看起來荒謬，原因是 `dot_product_desc` 被沿兩個
攤平維度**再正規化了一次**（`models/utils.py:102-119`），量級極小。照抄的時候
兩件事要一起抄，只抄一半會炸。

### 訓練

| 參數 | 值 | 出處 |
|---|---|---|
| optimizer | Adam，常數 LR，無排程 | `base_model.py:212` |
| LR | MagicPoint 1e-3 / SuperPoint 1e-4 | 各 config |
| `train_iter` | 合成 50000 / COCO MagicPoint 18000 / SuperPoint 600000 | 各 config |
| 訓練中的 metric | 只有 precision / recall | `super_point.py:94-101` |

最後一列值得注意：上游訓練中**不算 repeatability**，那些在離線的 `evaluations/`
才有。我們把 repeatability 拉進訓練迴圈，因為它是第 1 節裡決定「要不要再跑一輪
HA」的那個數，而 precision/recall 是對著自己產的 label 算的——teacher 有系統性
偏差時它會一路上升。

---

## 10. 便宜的斷言

CLAUDE.md 的規則：在任何以小時或數十 GB 計的執行之前，先寫那個幾秒鐘、而且只有
在執行有意義時才會通過的檢查。問「錯的結果會長什麼樣」，如果答案是「一個我會
相信的數字」，那裡就是斷言該放的地方。

**優先對故意錯的替代品評分，而不是對容忍度評分。**

| 測試 | 檢查什麼 | 錯的時候會看到什麼 |
|---|---|---|
| `test_homography` | `warp_points(H_inv) . warp_points(H)` = identity 到 1e-6；影像 warp 與點 warp 一致（單一亮點的 argmax 落在點 warp 說的位置）；cv2 路徑與 `grid_sample` 路徑一致 | (x,y)/(row,col) 互換、正負號錯。在 0 和 180 度看不出來，90 和 270 度致命——`Camera.output_to_level0` 就是被這個咬過的，而且是靠對誘餌評分才抓到 |
| `test_homographic_adaptation` | 塞一個只回固定點的假 detector：aggregate 的峰值在那個點、`counts` 等於 valid mask 蓋到它的 homography 數；且贏過「位移一個 cell」的誘餌 | aggregate 的座標系反了（用 H 而不是 H_inv warp 回來）。結果會是一張看起來合理但整體偏移的機率圖 |
| `test_detector_decoder` | depth-to-space 來回：已知 argmax 的 cell 張量，解碼後最大值落在對應像素；dustbin 是被丟掉而不是被算進去 | 通道排列錯（`(cell,cell)` 的 row-major/col-major），keypoint 會轉置 |
| `test_mpp_stack` | 同中心兩階 tile，細的降採樣後與粗的算正規化互相關 > 0.9，且贏過位移一 tile 的誘餌 | co-registration 的中心算錯。這是 `test_camera_output_to_level0` 在 Stage B 的對應物，那個測試找到過真 bug |
| `test_keypoint_label_store` | 存讀來回；`require=` 對不上時拒絕；`n_kp` 與 `kp_xy` 的 padding 一致 | 兩個設定的 label 互相覆蓋。照 `test_feature_store` |
| `test_ds_ladder` | 每個 rung 挑到的 level 的 ds ≤ 目標；在 4x 與 2x 兩種金字塔上各驗一次 | 挑到偏粗的一側 -> 靜靜地上採樣。這正是不能重用 `coarser_level_for_downsample` 的原因 |
| `test_pre_tile_store` | 中心裁切取回植入的方塊，而偏 ±1 格的裁切取不回（誘餌）；PNG 來回逐位元相同；七個 identity 欄位各自改動都會換目錄，而 `wsi_path`／層／計數都不會 | 裁切偏一格 -> 每張圖對每個 label 都偏一像素，訓練照樣收斂，模型只是「差一點」。這條是擋在 3c 那 32 GB 前面的秒級斷言 |

再加一個不是單元測試但同等重要的：**第一輪 HA 的 label 產出來之後，先跑
`cli/inspect_ha_labels.py` 把機率圖畫出來看**，再決定要不要進第二輪。第一輪的
teacher 是 COCO domain 的權重，H&E 是 out-of-domain，label 的品質是完全的未知數。
直接進第二輪的話，得到的會是一個對著壞 label 收斂得很好的模型——訓練曲線漂亮，
而且沒有任何東西會報錯。

---

## 11. 和 LocaScope 的接點

### 11.1 接回 stage 3

`3_localization/` 多一個 `SuperPathPointLocalizer`，和 `SIFT_RANSAC.py` 平行。
鴨子定型讀同樣那五個屬性——`best_region_index`、`best_x`、`best_y`、`ds`、
`best_rotation`（`3_localization/README.md` 記著這件事）——回傳
`SiftRansacResult` 形狀的結果，含 `center_x0/center_y0`。

`3_localization/README.md` 已經標記了一個待辦：那個 `location: SlideWinSimResult`
的型別標註在說謊，production 傳的是 `SlideWinSimRotResult`。加第二個 localizer
是抽出共用 result protocol 的自然時機——**但那是另一次的事**，不要順手做。

### 11.2 repo 層級

| 檔案 | 改什麼 |
|---|---|
| `utilities/_paths.py` | 加 `TRAINING_DIR`，納入 `setup_import_paths()` |
| `jobscripts/` | `SuperPathPointLabels.sh`（array over slides）、`SuperPathPointTrain.sh`（GPU） |
| `utilities/test_modules/` | 第 10 節的六個測試 |
| `utilities/bench_modules/bench_superpathpoint.py` | repeatability 對 SIFT |
| `log/TODO.log` | 一筆，含這份 spec 的決定與它們的理由 |

---

## 12. 執行順序

刻意把最便宜的未知數排在最前面。

| # | 做什麼 | 產出 | 為什麼在這個位置 |
|---|---|---|---|
| 1 | `common/Homography.py` + `test_homography` + `cli/demo_homography.py` | 兩條 warp 路徑一致，以及一張看得懂的圖 | 後面每一件事都建立在它上面，而它的錯誤是隱形的。demo 不需要模型，秒級 |
| 2 | `DsLadder` + `test_ds_ladder` | 6 片各自的讀取計畫 | 幾分鐘。金字塔已量（6.5），這步是驗證解析器挑對了層 |
| 3a | 把 `test_EoMT.py` 的 PCA 那一半包成 SegFunc，產出 6 片的遮罩 | 6 張 ds=14 的遮罩 + region | 後面兩步的輸入。**不是** EoMT 的頭，見 6.1 |
| 3b | 取樣探針：`(片, tile_size, ds)` 各跑一次，記整個候選池 | **216 格**（12 片 x 3 tile x 6 ds），每格 `n_admissible` 與每桶的供給 / 取得 | **跑了兩次，答案相反。** 2026-08-26 的六片版量到的是拒絕預算不是候選池；2026-08-27 的十二片版才是。結論：`tissue_ratio` 退役、`BALANCE=none`，兩張表在 6.5 |
| 3c | 離線切 **pre-tile**，只切 256 這一族 | 12 片 x 6 個 rung，每格要 100，實得 **6,388 張 / 4.4 GB**（3b 說的牆） | v1 只做 `model_256`。落地的是 pre-tile 不是 tile（6.6）——豐富度的桶判在 tile 的 footprint 上，pre-tile 只是 warp 的上下文 |
| 4 | 包上游權重成 teacher + HA + label store | 一片 slide、一個 rung 的 label | 第一個真實的未知數。同時量 HA 的 wall clock（§13） |
| 5 | `inspect_ha_labels` 看圖 | 決定：繼續，還是換 teacher | **決策點。** COCO 權重在 H&E 上可能根本不work |
| 6 | `KeypointNet` + loss + Trainer，VGG backbone | **`model_256_gray` 與 `model_256_rgb` 兩個 student** | 先把管線跑通，用最小的 backbone。兩個 student 共用同一批 HA label（座標不在乎通道數） |
| 7 | repeatability bench + 誘餌 | 第 1 節第一、四列的數 | 決定要不要 round 2，以及灰階 / RGB 哪個進下一步 |
| 8 | `TileEncoderBackbone` + `UpsampleDecoder` | foundation model 那條路 | 類別已經寫好（`SuperPoint/EncoderBackbone.py`，5.3），**還沒接進 `KeypointNet`**：`KeypointNetConfig.backbone` 標的是 `VggBackboneConfig`，而 `wired()` 沒有載權重就算不出 `out_channels`。接法本身是一個決定（`KeypointNet` 收一個建好的 backbone，還是這支帶一張寬度表），留到管線跑通再說 |
| 9 | Stage B：`MppStack` + `SurvivalTable` | 存亡表 + 連續帶的比例 | 需要一個堪用的 detector 才能開始 |
| 10 | Stage C：resolution head | 第 1 節第三列的數 | 需要存亡表 |
| 11 | `SuperPathPointLocalizer` + `bench_locascope` | 對 SIFT 的比較 | 最終判準 |

排在這條線**之後**、不進 v1 的：

| 做什麼 | 為什麼在後面 |
|---|---|
| `model_512` / `model_1024` 的抽取與訓練 | 第 1 節第五列要固定評估協定才比得了；256 那條線沒走完之前，協定本身還沒被驗證過 |
| leave-one-stain-out arm（3 HE 訓 / 3 Ki67 測，反向亦然） | 這是唯一能量到跨染色泛化的形狀（6.5 末段）。用同一批 tile，不必重抽 |

第 5 步是真正的決策點，不是形式。它可能的結論是「上游權重在 H&E 上產不出可用的
label」，那時的選項是換 SIFT/Harris 當 teacher，或補做 SuperPoint 的第一階段
（合成幾何圖形訓 MagicPoint）。這兩條路在那個時候才有資訊可以選，現在選是猜的。

---

## 13. 待決與已知風險

以下都**沒有**填進猜測值。每一項先說它在決定什麼，再說怎麼定——一個講不出自己
在決定什麼的參數，調它只是在改數字。

### `N`（HA 的 homography 數，上游 100）

**它決定三件事：**

1. **label 的雜訊。** `mean_prob` 是「看得到這個像素的那些 homography」的平均。
   N 小的時候，主導結果的是這次剛好抽到哪幾個 homography，於是 top-M 的排名有一
   部分是 RNG 的性質而不是組織的性質。
2. **整個 Stage A 的 wall clock。** 成本嚴格線性於 N：每張 tile `N` 次前向 + `2N`
   次 warp。它同時決定 `R`（輪數）付不付得起。
3. **邊緣的覆蓋。** tile 邊緣的 `counts` 本來就小；N 小時邊緣可能只有 1-2，那裡的
   `mean_prob` 是一個 detector 的意見而不是共識。上游的 `filter_counts` 就是為此
   存在的，而它預設關著（0）。

**怎麼定：跑一次 N=200，然後用前 k 個（k = 10/25/50/100/200）重算 aggregate，畫
「top-M 點集與 k=200 的重疊率」隨 k 的曲線，取膝點。** 一次執行，不是五次。

WSI 的紋理是準週期的（腺體、細胞核陣列），homography 之下的重複性可能和自然影像
的角點很不一樣——可能更好（結構豐富），也可能更糟（自相似讓到處都像角點）。上游的
100 是自然影像的數字，沒有理由直接繼承。

### `alpha`（Stage B 的配對半徑，`tau = max(tau_floor, alpha * d_coarse)`）

**它決定 Stage C 的 label 本身。** 存活率整條都是它切出來的：

- alpha 太小：一個真的存活、但在粗階被定位到 2 個粗像素外的點被判為死。存活率
  系統性偏低，Stage C 學到「幾乎沒有東西存活」——那會被讀成關於組織的發現，而不是
  關於閾值的人工物。
- alpha 太大：什麼都配得到，存活率逼近 1，頭沒有東西可學。第 1 節第三列的
  「贏不贏得過常數基線」會正確地說沒有——但要等整個 Stage B 跑完才知道。

所以 alpha 決定的是**Stage C 的 label 帶不帶資訊**。

**怎麼定：畫出跨階最近鄰距離的分布。** 眾數要明顯落在 tau 之內、尾巴落在外面。
若那個分布是單峰而且沒有分離，那就表示「同一個點」在那個階差上根本不可判定——
**那句話本身就是發現**，不是一個要用數字填掉的缺口（ClaudeRules §10）。
起始值 1.5 個粗階像素，第一次跑當校準跑。

### `M` / 點密度

見 6.3：它同時決定 store 對閾值是否無損，以及 **student 被訓練成多密的 detector**。
第二件是實質的——留下來的每個點都是 detector CE 的一個正樣本 cell。作法是用閾值切、
記下 `n_kp`、M 只當記憶體上限。

### HA 的 wall clock

**它決定 Stage A 是一個 job 還是一個專案**，也就是 `R > 1` 付不付得起，以及
`model_512` / `model_1024` 那兩條線值不值得開。

成本 = tile 數 x N x forward(tile_size)，而 forward 隨 tile **面積**走。v1 全部跑在
256 上，所以它是三個 model 裡最便宜的一個——1024 的每張 tile 貴 16 倍，而它的 rung
只少兩階。換句話說 v1 的 wall clock 不只是自己的數字，它同時是另外兩條線的下界。

**怎麼量：一片、一階、100 張 tile，`N=100`，先只量 256。** 十分鐘，對 N 和 tile 數
線性外推、對 tile_size 平方外推，所以 512 / 1024 不必另外量就估得出來。第 12 節
第 4 步的第一件事。

### ~~pre-tile 的倍率~~ —— 已決定：3

不再是待決。推導的界是 **2.49**（6.6 有逐項的表），取 3。

刪掉的理由值得留著，因為它是一個推理錯誤：這裡原本要「跑一批 draw、取 99 分位」，而
那把成本算成了每個 draw 一次讀圖。**pre-tile 每個位置只讀一次、N = 100 個 homography
共用它**，所以要涵蓋的是 100 個裡的最大值——在 N = 100 之下，99 分位和最壞界是同一
件事，而最壞界推導得出來。

我先前給的「最壞 2.9」也是錯的：把 perspective 的 1.47 當成對 tile 的倍率，而它是對
四邊形（0.85）的倍率。

`--calibrate` 留著當**檢查**：抽 N 個、斷言沒有一個超過 3。

### 灰階 vs RGB —— 已決定：兩個都訓

這其實是兩個問題，被一個詞黏在一起。

**teacher 沒有選擇。** 上游權重是 1 channel（`superpoint_pytorch.py:83` 的
`channels = [1, *conf.channels[:-1]]`）。餵它灰階，否則就別用它。

**student 兩個都訓**，`model_256_gray` 與 `model_256_rgb`，共用同一批 HA label
（HA 的產出是座標，不在乎通道數）。成本是訓練時間 x2，管線完全相同。

它決定的是「染色顏色能不能造出或抹掉一個 keypoint」，而兩個方向都是真的：

- H&E 的細胞核是紫的、細胞質是粉的，那些邊界大致和亮度邊界重合，轉灰損失不大。
- **Ki67 不是。** DAB 陽性的細胞核是棕的、陰性的是藍的，兩者的亮度可以幾乎一樣。
  轉成灰階會把一個真實的生物邊界抹掉。
- 反方向：RGB 的 detector 可以靠染色來認點，跨染色的泛化可能更差。

**量得到的**：兩個 student 在 held-out 兩片上的 repeatability，**逐片分開報**。
一片贏一片輸就是「差異是片子的性質」，兩片不夠判，記成未決（§1 第四列）。

**量不到的**：上面第三點。**兩種染色都在訓練集裡**，所以 held-out 上的數字說的是
「換一片沒看過的玻片」，不是「換一種沒看過的染色」。要量得加 leave-one-stain-out
arm（6.5 末段），用同一批 tile，v1 不做。在那之前，結論裡不准出現「跨染色」這三
個字。

### 逐片驗證：為什麼記、怎麼記、記什麼

**為什麼。** 同一片的 tile 共用染色批次、掃描機、切片厚度與組織來源。兩片合併成
一個數之後，「A 比 B 好 0.002」可能整個來自其中一片的個性——**分開報是唯一能發現
這件事的方法**。這不是「順便多記一點」，它是 §1 第四列那條判準能不能執行的前提：
判準寫的是「其中一個在兩片上都贏」，而合併後的單一數字**回答不了那句話**。

**記什麼。** 每個 arm x 每片一列：

```
slide, n_pairs, repeatability, repeatability_decoy, repeatability_margin,
mean_points_per_view, detector, warped_detector, descriptor_scaled, total
```

`mean_points_per_view` 是新增的，而且它不是裝飾：**沒有它，margin 掉了就分不出是
模型變差還是點變多**。誘餌的匹配率隨點密度單調上升（匹配框 `(2*nms_radius+1)^2`
對 tile 面積），所以一個吐更多點的模型會同時推高 repeatability 和 decoy，而 margin
可以完全不動。點數是讀 margin 的必要背景，不是額外的好奇心。

**怎麼記。** `Trainer._repeatability` 已經回傳逐 pair 的 `hits` / `decoys`，只是被
`np.mean` 一次壓平。要分片只需要 item 帶一個 `slide_index`——
`Datasets.__getitem__` 現在回傳 `rung` 和 `rung_index`，**沒有 slide**——然後
`validate()` 依它分組。所以這是一個欄位加一次 groupby，不是新的一遍計算。

**「兩片都贏」贏的是 `repeatability_margin`。** arm A 在 `BRACS_1598` 和
`S1103627,G7E,110127` 上的 margin **都**高於 arm B 才算贏。一片贏一片輸 → 差異是
片子的性質，兩片不夠判 → **記成未決**（§1 第四列），不是取平均。

### 四個 arm：RGB / 灰階 x 有預訓練 / 無預訓練

student 的 backbone 和上游 v6 **逐層相同**——`VggBackbone` 的
`widths = [in, *channels[:-1]]` 給出 `1->64->64->128->128`，`VggBlock` 是
conv -> ReLU -> BatchNorm(eps=0.001)，模組名 `conv`/`activation`/`bn`，detector 出
`stride^2+1 = 65`，descriptor 出 256。全部對得上
（`superpoint_pytorch.py:50-100`）。所以上游權重可以整包轉移，唯一差別是屬性名：
上游 `backbone.`，這裡 `stages.`。

#### 載入必須有一條會擋住整個 arm 的斷言

**斷言**：載入之後，對同一張輸入，`student.prob_map` 和 `teacher.dense_prob`
**逐元素相等**（fp32 容差）。

**理由**：key rename 是機械的，但**載錯一部分不會報錯**。`strict=False` 之下沒對上
的層保持隨機，網路照樣訓練、照樣收斂——只是變成一個好一點的隨機初始化。而那會被
讀成「預訓練沒什麼用」，也就是 CLAUDE.md 那條「問一個錯的結果長什麼樣子；如果答案
是『一個我會相信的數字』，那就是斷言該放的位置」。

**加一個誘餌**：把某一層的權重打亂，相等**必須破**。否則這條斷言可能在比較兩個都
是空的東西。

#### RGB 的第一層用複製除以三

上游權重是 1 channel。conv1 的 kernel 複製三份除以 3，於是 **3 channel 網路吃
luma 圖的輸出，和 1 channel 網路吃同一張灰階圖完全相同**——這直接就是第二條斷言。

第一層改用隨機初始化也可以，但那樣 `rgb+pretrain` 和 `gray+pretrain` 就不是同一種
預訓練，四個 arm 的兩條軸不再正交，而「RGB 比灰階差」會和「RGB 的預訓練比較弱」
混在一起。

#### `gray + pretrain` 是自蒸餾，結論要照這個寫

HA label 是**上游 v6** 產的（第 4 節 Teacher）。gray student 從上游 v6 初始化，就是
從**產生它訓練標籤的那個網路**開始。

這是一個合理的 arm——它正是 SuperPoint 的 round 2——但它回答的問題是
**「在自己的 HA label 上再訓一輪，比從零學同一批 label 好嗎」**，不是「預訓練有沒有
幫助」。後者的對照組必須是**別的**預訓練來源。寫進 `log/TODO.log` 和任何結論時用
前一句話，否則這個數字會被誤引。

#### 兩片 held-out 撐不起四個 arm

四個 arm 是六組兩兩比較，而片數沒有變。§1 第四列的「未決」在這裡不是例外而是常態，
**加 arm 不能解決它**——要嘛加片子，要嘛把這一輪的目標寫成「排除明顯壞的組合」而
不是「宣告贏家」。

### `background_threshold`（UNI2-PCA 遮罩的閾值，notebook 的 0.5）

**它決定的是「哪些位置有機會被抽到」，而且它的兩種錯不對稱——但 2026-08-27 之後
只剩下一種。**

`tissue_ratio` 退役之前，遮罩不是最後一關：`TileSampler` 之後還有一道逐窗的閘門，
會把多進來的玻璃位置丟掉，代價只是拒絕取樣多跑幾輪。**那道關已經沒有了**，而接手的
豐富度桶不是它的替代品——`score_background` 就是 `white_fractions`，讀的是**同一張
遮罩**。一個被鬆閾值放進來的玻璃位置，會被那張同意放它進來的遮罩評成低背景，落進
`bg00_15`，然後被留下來。

- **閾值太鬆**：多進來的玻璃**不再有第二個機會被攔下**。它會變成訓練 tile，而 HA
  在一片空白上找到的點是壓縮雜訊與掃描條紋。**現在也不可回收了。**
- **閾值太緊**：被排除的組織不在 `tissue_regions` 裡，取樣器永遠看不到它，也沒有
  任何下游步驟能把它找回來。**不可回收**，而且它看起來不像錯——只是抽到的 tile
  少了一點。

**兩側都不可回收了，所以判準換人。** 不再是「往鬆的一側錯」，而是**那兩桶的張數就是
儀表**：`bg50_70` 與 `bg70_85` 在 3c 拿到 308 + 215 張（6.5），閾值太鬆會讓這兩個數
字往上跑，因為多出來的玻璃位置正好落在高背景那一側。它們原本只是溢流的去處，現在
同時是遮罩閾值的讀數。

**為什麼不能直接挑一個數。** `InspectPcaSeg`（2026-08-26）畫出來的 PC1 直方圖在兩
種染色上是不同形狀的：Ki67 雙峰，谷底約 0.2；BRACS 單峰，眾數約 0.27、尾巴拖到 0.8。
單峰的直方圖沒有谷可取——在它上面挑一個數就是在「還沒看到分離之前挑閾值」
（ClaudeRules §8）。而且 Ki67 那個谷分開的是**整片空白的 tile 和含組織的 tile**，
不是玻璃 cell 和組織 cell，所以它甚至不是同一個問題的答案。

`agree_hsv` 也不能當裁判：HSV 遮罩對淡染的切片本來就少算（0.5 閾值下 Ki67 的 HSV
只給 1.4%，PCA 給 14.7%），拿它當基準等於把 PCA 校準到一個已知會漏的東西上。

**怎麼定：對存下來的成分做掃描，不要重跑編碼。** `MaskStore` 的每個檔案帶第二個
tensor `components`（`[rows, cols, k]` float16，每片 581-814 MB），就是這個 bit 被
閾值切出來之前的那個場。於是

    slide_mask, meta = MaskStore.load(path, with_components=True)
    pc1 = slide_mask.components[..., 0]
    # 對每個候選閾值：mask = pc1 > t，close-then-open，量 fraction 與區域數

一片一個候選是**秒**；不存成分的話，一片一個候選是 3.5 到 6 分鐘的 GPU。這就是
成分要進 store 的理由，也是 store 先建、閾值後定的理由。

要看的量有三個，順序是固定的：

1. **`fraction` 對 `t` 的曲線。** 組織/玻璃如果真的被 PC1 分開，曲線上會有一段
   平台——閾值在平台裡怎麼移都不改變答案。**沒有平台就是沒有分離**，那句話本身是
   發現，該換的是成分數或 `feature_norm`，不是這個數字。
2. **平台的鬆側邊緣**，不是中點。理由是上面的不對稱。
3. **和已量過的組織比例對照**：BRACS 20.8-38.2%、Ki67 3.5-9.2%（HSV，ds 32）。
   對照不是校準——PCA 比 HSV 更會抓淡染，所以 PCA 略高是預期的，PCA 低於它才是
   警訊。

**在那之前，store 用的是 0.5。** 這不是把一個猜測凍起來——0.5 是 notebook 的值，而
這一輪建 store 的目的正是把那個場放到磁碟上，好讓這個問題問得成。掃描完若改了值，
`background_threshold` 是 hashed 欄位，新的遮罩會是**另一個檔**而不是覆蓋，第 3c 步
讀哪一個由 `require={'segmenter_id': ...}` 說了算。

### 第一輪 label 的品質

完全的未知數。已在第 12 節第 5 步設成決策點。

### 每格實際取得到幾張 tile —— 探針要回答的三件事

第 12 節第 3b 步的探針跑 216 格（2 個 `tissue_ratio` x 3 個 tile_size x 6 個 ds
x 6 片）。它一次決定三件事，這是它值得先跑的原因：

1. **`tissue_ratio` 取 0.5 還是 0.75。** 0.75 在 ds=14 的 PCA 遮罩上比在舊 log 那種
   ds=32 的 HSV 遮罩上更難通過，而那個差距沒有量過（6.1 末段）。舊 log 的數字說明
   機制，不預測我們會拿到幾張。
2. **rung 平衡用 `align-min` 還是 `loss-weight`。** 最差的格有 300 張就用前者，
   只有 40 張就得用後者（6.5）。
3. **`model_512` / `model_1024` 各自剩幾個 rung。** footprint 表說有 3 格先天是空的，
   但那是 `tissue_ratio=0.5` 的成績；0.75 可能再吃掉幾格。

三件事都不是用挑的。探針的成本是幾秒，而挑錯任何一件的成本是整批重抽。

**2026-08-26 跑完，三件都有答案**（`result/BuildMaskStore/tile_yield.csv`）：

1. **0.5。** 0.75 把 ds 16 / ds 32 壓到 87 / 27 張。表在 6.5。
2. **`align-min`。** 四片訓練片的最差階有 1784 張，砍齊丟 9%。
3. **那道牆是分片的，不是一致的**——這是預測沒說中的地方。footprint 16384 與
   32768 兩格在 BRACS 上滿額，在 Ki67 上塌掉：

   | tile / ds | footprint | BRACS x3 | Ki67 x3 |
   |---|---|---|---|
   | 512 / 32、1024 / 16 | 16384 | 500 / 500 / 500 | **0** / 281 / 500 |
   | 1024 / 32 | 32768 | 500 / 500 / 500 | **0 / 0 / 0** |

   所以 `model_512` 的 ds 32 少一片、`model_1024` 的 ds 32 是 BRACS-only。那不是
   「這一格沒有資料」，是「這一格只有一種染色的資料」——拿它訓出來的粗階行為會和
   染色綁在一起，而**沒有任何東西會顯示這件事**。512 / 1024 真的要做的時候，這
   一格要嘛丟掉，要嘛在結論裡明講。

---

## 14. 建構清單：SuperPoint 第二到第三階段

第 12 節排的是**做什麼**的順序，這一節是**寫哪些檔**。29 個，逐項列出而不是含糊地說
「加一個訓練模組」，因為一個講不出自己要幾個檔的計畫，是還沒想清楚的計畫。

順序就是相依順序：上面的不完成，下面的沒有輸入。

### 資料（第 12 節的 3a / 3b / 3c）

| # | 檔案 | 做什麼 |
|---|---|---|
| 1 | `utilities/MaskStore.py` | `SlideMask` + 落地格式 + `build_one(wsi, segmenter)`。遮罩之外還存 `components` |
| 2 | `utilities/cli/build_mask_store.py` | 對 6 片跑分割器、寫 store |
| 3 | `utilities/cli/probe_tile_yield.py` | 3b 探針，`(片, ratio, tile, ds)` 216 格 |
| 4 | `training/SuperPathPoint/common/PreTileStore.py` | pre-tile 的落地格式與索引 |
| 5 | `training/SuperPathPoint/cli/extract_pretiles.py` | 3c 抽取 |

`SlideMask` 從 `aiNNModel/Uni2PcaSegFunc.py` **搬到** `utilities/MaskStore.py`。層次
問題：store 在 utilities，而 utilities 不該 import aiNNModel。搬過去之後 store 也就與
分割器無關——存 hsv 或 hest 的遮罩用同一個。

第 2 支薄到只剩 argparse、迴圈、印進度；建構邏輯在 `MaskStore.build_one`。`FeatureStore`
與 `cli/build_reference_store.py` 是同一個分法，而 CLAUDE.md 說庫層不放 CLI 解析與 print。

**每個遮罩檔存兩個 tensor**：`mask`（一片一個 bit / cell）與 `components`
（`[rows, cols, k]` float16，581-814 MB）。理由在 §13 的 `background_threshold`：
遮罩回答不了「這個閾值對不對」，因為問題問的正是產生它的那個場。同一個檔而不是
sidecar——sidecar 可以被單獨刪掉、搬走或重建，接著掃描就會拿 A 片的成分配 B 片的
遮罩。`load()` 預設**不**讀成分（`safe_open` 逐 tensor 取），只有掃描要它。

### 第二階段：Homographic Adaptation

| # | 檔案 | 做什麼 |
|---|---|---|
| 6 | `SuperPoint/Teacher.py` | 載 `superpoint_v6_from_tf.pth`，`detect(images) -> prob` |
| 7 | `SuperPoint/HomographicAdaptation.py` | N = 100 的疊加、`mean_prob`、`counts` |
| 8 | `common/KeypointLabelStore.py` | 稀疏點集落地（6.3） |
| 9 | `cli/make_ha_labels.py` | 跑 HA、寫 store |
| 10 | `cli/inspect_ha_labels.py` | **gate**（第 12 節第 5 步） |

### 第三階段：聯合訓練

上游的第三階段**就是聯合的**（`super_point.py:73-92`）。只訓 detector 是 MagicPoint，
那是第一階段，不是這裡要的東西。

| # | 檔案 | 做什麼 |
|---|---|---|
| 11 | `common/Interfaces.py` | Backbone / DetectorDecoder / DescriptorHead protocol + `check_shapes` |
| 12 | `SuperPoint/Backbones.py` | VGG，4 stage、stride 8、輸出 128 通道 |
| 13 | `SuperPoint/Decoders.py` | `DepthToSpaceDecoder` + `UpsampleDecoder` + `depth_to_space_prob`（`cell**2 + 1` 通道，含 dustbin） |
| 14 | `SuperPoint/Heads.py` | `DescriptorHead`，256 維、L2 正規化 |
| 15 | `SuperPoint/KeypointNet.py` | 組合體 + `extract_keypoints`（NMS / top-k / 去邊） |
| 16 | `SuperPoint/Losses.py` | detector CE x2 + descriptor dense hinge |
| 17 | `SuperPoint/Datasets.py` | 吐 `(I, I', H, label, label')` 的 pair dataset |
| 18 | `SuperPoint/Trainer.py` | 迴圈、wandb、checkpoint 帶 `identity_json`，驗證時算 repeatability 對誘餌 |
| 19 | `cli/train_superpathpoint.py` | 進入點 |

**這一輪寫檔時定下的三件事，寫在這裡而不是散在程式碼註解裡：**

1. **`points_from_prob` 放在 `common/KeypointLabelStore.py`，第 15 支的
   `extract_keypoints` 呼叫它。** teacher 的 label 與 student 的預測必須用
   **同一條規則**變成點——NMS 半徑在兩邊不一樣的話，每一個 repeatability 數字量
   的都是兩種慣例的差。`common/` 是兩個階段本來就都依賴的那一層。
2. **`erode_valid` 從 `valid_mask` 裡拆出來。** HA 在 pre-tile 之下不能用
   `valid_mask` 造遮罩：來源比輸出畫框大，有效性要從 pre-tile 自己的範圍 warp
   過來，只有侵蝕那一段是共用的。偶數核、TF 的 anchor、`borderValue=0` 三個難處
   因此只有一份。
3. **HA 的 identity 不繼承 `IdentifiedBuild.identity_parts`。** 那個預設會去雜湊
   teacher 的**權重**而丟掉 teacher 的**設定**；`HomographicAdaptation.
   identity_parts` 改成放 `teacher.identity_id()` 一個字串，兩者都在裡面，而且
   「teacher 是什麼」只有一個定義。

第 17 支有一個會安靜出錯的地方：**label 的 warp 是把點 warp 過去再重新 splat**
（`pipeline.py:38-53` 的 `warp_points` + `filter_points` + `add_keypoint_map`），不是
把 keypoint map 當影像 warp。當影像 warp 會把單像素的點插值糊掉，等於一邊訓練一邊丟
label，而且不會報錯。

### 測試與 jobscript

| # | 檔案 |
|---|---|
| 20 | `utilities/test_modules/TestSuperPathPoint/test_mask_store.py` |
| 20b | `utilities/test_modules/TestSuperPathPoint/test_pre_tile_store.py` |
| 21 | `utilities/test_modules/TestSuperPathPoint/test_homographic_adaptation.py`（第 10 節那條誘餌檢查） |
| 22 | `utilities/test_modules/TestSuperPathPoint/test_keypoint_label_store.py` |
| 23 | `utilities/test_modules/TestSuperPathPoint/test_detector_decoder.py`（depth-to-space 來回） |
| 23b | `jobscripts/ExtractPreTiles.sh`（第 12 節 3c） |
| 23c | `utilities/test_modules/TestSuperPathPoint/test_superpathpoint.py`（第 11-19 支） |
| 23d | `utilities/test_modules/TestSuperPathPoint/test_encoder_backbone.py`（`EncoderBackbone.py`） |
| 24 | `jobscripts/MakeHaLabels.sh` |
| 25 | `jobscripts/TrainSuperPathPoint.sh` |

這八支測試全部掛在 `jobscripts/TestSuperPathPoint.sh` 底下，分成六個 stage：
`mask`（`from_mask`）、`store`（三個 store）、`decoder`、`ha`、`student`、
`backbone`。**沒有一支需要資料、權重或 GPU**——金字塔、玻片、偵測器、ViT 全是檔案
自己捏的，所以它們可以在 login node 上秒級跑完，而它們擋在前面的那些執行是數小時
與數十 GB。（另有 `backbone-model` 一個 stage 用真權重跑同一支，那是唯一需要 GPU
的一個，理由見 5.3：假 trunk 可以宣告 `dynamic_img_size`，但「timm 會不會真的把
位置編碼內插到 256」是關於真模型的事實。）

`cli/demo_homography.py` 加一個 `--calibrate` 模式，不算新檔。

**寫的過程中多出來的六個檔**（原本 25 個沒有它們，理由都不是「順手」）：

| 檔 | 為什麼 |
|---|---|
| `utilities/test_modules/TestSuperPathPoint/test_pre_tile_store.py` | CLAUDE.md 那條「在昂貴的執行之前放一條便宜的斷言」。3c 讀六片、寫約 32 GB、跑數小時，而它會壞的四種方式沒有一種會拋例外——中心裁切偏一格、pre-tile 存錯尺寸、兩批抽取撞同一個目錄、換成有損編碼。四種都是秒級可釘，其中三種釘的是**誘餌**而不是容忍度 |
| `common/HomographyConfig.py` | 13 個 sampler 選項被 `HaConfig` 與 `PairDatasetConfig` 各要一次。放不進 `common/Homography.py`：那支刻意不在 import 時碰 torch（`warp_image_torch` 自己 lazy import），而 `ConfigIdentity` 會拉 torch 進來，登入節點上跑得動的 demo 就跑不動了 |
| `jobscripts/ExtractPreTiles.sh` | 3c 要在叢集上跑，而原本的清單沒有給它一支 |
| `utilities/test_modules/TestSuperPathPoint/test_superpathpoint.py` | 第 11-19 支**一支測試都沒有**。它刻意違反「測試以被測 module 命名」那條——被測的不是八個 module，是它們**之間的合約**：`space_to_depth` 對 `depth_to_space_prob`、`check_shapes` 對 backbone、dataset 的 warp 對 loss 的對應遮罩。拆成八個檔，每個檔只會拿到半句話 |
| `SuperPoint/EncoderBackbone.py` | 原本規劃在 `Backbones.py`（第 12 支）裡。分出來的理由不是整潔：`import aiNNModel` 會執行 `os.environ.setdefault('HF_HOME', ...)`，而 huggingface_hub 是先到先得——把它放進 `Backbones.py` 等於把那個副作用放上 `KeypointNet.py` 的 import 路徑，也就是每一支 CPU 測試的 import 路徑。這支檔案頂層只 import `TileEncoderFunc`（沒有 HF），實作模組是在 `build()` 裡才 import 的 |
| `utilities/test_modules/TestSuperPathPoint/test_encoder_backbone.py` | 上面那支的測試。假 trunk 而不是真權重，而且假的在這裡是**更強**的測試：這支檔案決定的全是數字，而假 trunk 可以被指使去謊報它們——宣告 stride 16 實際 stride 8——真的 ViT 做不到 |

---

## 15. 寫這 31 個檔時遵守的規則

### 抽取的門檻是三次

**同一件事寫到第三次就抽出來。** 兩次是巧合，三次是模式。

**被抽出來的東西超過三個而且是同一性質，就開一個 module。** 四個散落的 helper 各自
待在用它的檔案旁邊，和四個放在一起、有一個共同名字的 helper，差別是後者能被找到。

### 動到既有的 utilities 或 encoder module 就停下來問

`utilities/` 和 `aiNNModel/` 底下的東西有既有的呼叫端，而這個專案的失效模式一貫是
「安靜地改變一個別人依賴的行為」。所以：**要改它們，先停止寫程式，問過再動。**

新增檔案不受此限；改既有檔案受。

### 因為合併或刪除而變動的程式碼，要回來更新這份 spec

這份文件是規格不是紀錄。如果實作時發現兩個檔該合併、或某個模組不需要存在，改完程式
之後回來改這裡——否則下一個人讀到的是一份描述不存在的程式的規格。

### 還沒量到的常數，用 PENDING-MEASUREMENT 標記，不要猜

有三個值必須等執行結果才知道，而它們擋在寫程式的路上。作法是照
`utilities/_tempmeasure.py` 的 `TEMP-MEASURE` 慣例：一個可以被 grep 和 sed 掃掉的
標籤，加上一份說明它在等什麼的檔案。

    grep -rn PENDING-MEASUREMENT --include=*.py .    # 還在等什麼
    sed -i '/PENDING-MEASUREMENT/d' <那些檔案>       # 填完之後掃掉

三個是（第一個已經填掉了）：

| 標籤 | 在等什麼 | 誰跑 |
|---|---|---|
| ~~`PENDING-MEASUREMENT: tile-yield`~~ | **已填兩次，第二次推翻第一次**：2026-08-26 的 `tissue_ratio=0.5` / `align-min` / 17,784 張量錯了對象；2026-08-27 的十二片探針給的是 `tissue_ratio` 退役、`BALANCE=none`、**6,388 張**。兩張表在 6.5 | 使用者 |
| ~~`PENDING-MEASUREMENT: png-ratio`~~ | **已填，2026-08-27**：45.1%。v1 實際落地 **4.4 GB**（14.2 GB 是 17,784 張那一版的推算，作廢）。表在 6.5 | 使用者 |
| ~~`PENDING-MEASUREMENT: ha-label-quality`~~ | **已填，2026-08-27**：72 格全數產出，889,022 點。第 12 節第 5 步的 gate 過了——8 格判 D 是誘餌在高點密度下飽和，不是 teacher 失敗 | 使用者 |
| `PENDING-MEASUREMENT: loss-scale` | **epoch 0 有了，但不夠。** 2026-08-28 的 `model_256_gray`：`descriptor 1.156e-4 x 10000 = 1.156` 對 `detector 4.485`，約 1:4，同一個數量級——補償轉移過來了。**還在等的是十個 epoch 之後那個比例。**起跑點對只說明縮放係數挑得對，不說明 descriptor 在整段訓練裡都拿得到梯度：若 detector 一路降而 descriptor 卡在 1.16 不動，descriptor 就是被固定住的一項而不是在學的一項，而 `total` 照樣會下降。填的時候要填**兩個 epoch 的比例**，不是一個 | 使用者 |

標記的地方寫下：**在等什麼數字、拿到之後要填哪裡、以及暫時用什麼值跑得動**。暫用值
必須明說是暫用的——一個沒有標記的猜測值，和一個量出來的值在程式碼裡長得一模一樣，
而那正是 ClaudeRules 第 8 節在講的事。

## 附：這份 spec 沒有處理的

- 多卡 / DDP。第一版單卡，理由見 8.3
- keypoint 的旋轉不變性。LocaScope 的 retrieval 已經在處理旋轉
  （`GigaPathSlidingWinSimRot`），這裡的 homography 取樣含旋轉所以 detector 會學到
  一些，但沒有明確的設計或量測
- 真實 query 上的直接訓練。真實照片沒有 keypoint 的真值，而 HA 需要能對整張圖做
  homography——一張 1440x1024 的照片可以，但它不在任何 ds 階梯上。這是之後的事
