# CoAE 規格

本檔描述方法的現況（程式在 `locascope/training/CoAE/*.py`），不含歷史；變更只留在最後的變更紀錄。代號的定義見 `locascope/training/NAMES.md`；文件與程式不一致時以程式為準。

## 1. 目的與研究條件

目的：用已分析過、兼具區分性與不變性的多通道訊號（共通圖，Co-figure）當目標，自監督地學一個表示，量這些訊號學進表示之後能不能提升搜尋能力。

研究條件（不可違反）：
1. 真實照片（`Data.real`）不訓練、不用來選任何設定（層混合、權重、先驗、切分都不行）；訓練流程裡沒有真實照片，只在選定 checkpoint 之後驗證一次。
2. FM 不丟：FM 的 token 是網路唯一的輸入，FM 自己的 CLS 與 patch 平均並列輸出。
3. 不變性與區分性來自訊號目標，沒有以來源為對象的損失（判別器與 KoLeo 的程式保留、預設權重 0）。實驗 arm 的例外見「損失」節。
4. 結論只用實際條件的結果：層級已知、尺度未知（raw）。尺度對齊、oracle、只比一個錯誤位置的結果是天花板，不是發現。
5. 網路只吃 tile：slot、來源、擾動值都不是輸入。

預先寫死的規則在「量測與判準」節；規則的門檻由 thinker 提出、使用者同意，使用者可改。

## 2. 架構 CoAE.Arch1

```
輸入 tile 256 px（一個 view 的畫布的中心 tile）→ FM 自己的 transform（gigapath：中心裁 224 px，視野 field_px = 224；uni2：整張縮成 224，field_px = 256）
→ FM（frozen 或 LoRA），取多層 patch tokens，層混合（可學權重，有初始值與上限）→ tokens g×g×D（FM 自己的格，不重採樣：gigapath 14×14、uni2 16×16；gigapath 的 D = 1536）
   同時輸出 FM 自己的 CLS（cls_fm）與 patch 平均（avg_fm）
   ├─ AE encoder：Swin transformer 一段（g×g 上 `swin_depth` = 2 個 block：窗口 g/2，W-MSA 加 SW-MSA，寬 `swin_dim` 384）→ 2×2 patch merging，加 tokens 2×2 平均的殘差
   │    → latent s×s×128（s = g/2：gigapath 7×7×128、uni2 8×8×128；輸出的 latent 不含雜訊）
   │       → z = latent + 雜訊（只在訓練）
   │          ├─ decoder：ConvNeXt，卷積上採樣 s→2s→4s，不接 skip connection → Co-figure out_cells×out_cells（= 2g：gigapath 28×28、uni2 32×32），鋪滿 FM 的視野，每格 8 px；一個 token = 2×2 格，一個 latent 格 = 4×4 格
   │          │    vref、vavg：N = 10 通道；vdist：每種訊號一組 μ、一組離散度
   │          └─ K、V（z 的 s² 個 cell）→ cross-attention（不從 q 加殘差）→ LayerNorm → cls′（維度 = FM 的維度；不做投影；輸出的 cls′ 不含雜訊）
   │               → + 雜訊（只在訓練）→ 一層線性（線性頭）→ 位置描述子 29×4×4
   └─ attn_pool（學習的 query）→ q 1×d → cross-attention 的 Q
```

- FM 的精度與編譯：`FmConfig.weights_dtype` 預設 fp32（fp32 權重加 bf16 autocast）；實驗 arm 由 `CoaeTrain.sh` 以 `--fm-dtype bf16` 明示（FM 權重 bf16，forward 不包 autocast），正式 arm（`COAE_CAP` 有給）與 LoRA 一律 fp32。`torch.compile` 包 `FmTokens.forward`，預設開（`--no-compile` 關；不是 config 欄位，不影響 arm 的身分）。續跑時 `weights_dtype` 與 checkpoint 不同就拒絕。
- 瓶頸雜訊（`ModelConfig.noise` σ = 0.5）：各向同性的加性高斯雜訊，sd = σ × 該向量自己（未中心化）的 RMS（RMS 不回傳梯度；latent 逐格算、cls′ 逐向量算），只在訓練時；latent 與 cls′ 共用一個 σ，沒有 warm-up。decoder、cross-attention 與線性頭吃加過雜訊的；輸出的特徵（`latent`、`cls`）沒有雜訊。它是功率配置的正則，不是資訊量上的瓶頸：放在讀出用不到的方向上的能量只會加大雜訊，所以被擠掉，cos 才會跟著目標走。
- 梯度路徑：`latent → z → cross-attention → cls′ → 線性頭` 沒有 detach；cls′ 的內容損失的梯度到得了 encoder。重建損失對 cls′ 沒有梯度。
- 線性頭的單位：線性頭在標準化單位上輸出，`pooled = head × pool_sd + pool_mu`（`pool_mu`、`pool_sd` 是描述子每個通道的 running 平均與 sd，模組的 buffer，每步更新）。描述子目標仍是原始動差。
- 殘差版本（`--cls-residual`，arm 修飾 `fmresidual`）：cls′ = FM 自己的 CLS（原封不動）加 gain × attention 分支，gain 從 0 開始。
- 判別器只在 `head.w_adv > 0` 時建立（預設不建）。

## 3. 資料與來源

名詞：
| 名詞 | 意思 | 程式 |
|---|---|---|
| 同位置 | 同一個位置（一個 footprint）的所有 view | 一個 `Place` |
| 來源 | scan 本身（`ref`）或一種模擬（只做光度） | `sources = ('ref',) + data.sims` |
| 擾動程度 0、輕、大 | 幾何擾動的範圍（位移、偏轉、縮放、畸變 k1），每個量在 ± 範圍內均勻抽、任意實數：0 = 不擾動；輕 = 8 px、±3°、3 %、0.05；大 = 64 px、±6°、±20 %、0.1 | `LEVELS` |
| twins | 同一個擾動（同一個 `Geometry`）、不同來源的 view | 同一個 slot 的全部來源 |
| siblings | 同一個位置、同一個來源、不同擾動的 view | 不同 slot |
| dependence | 目標只在 twins 之間共用，跟著 view 自己的幾何走 | 機制 `dep` |
| independence | 整個位置的 view 共用一個目標，view 之間不對齊 | 機制 `ind` |
| `mix` | 共通圖照 `dep`，cls′ 的目標照 `ind` | 機制 `mix`（預設） |
| vref、vavg、vdist | 共用者怎麼合成目標：ref 的共通圖、共用者的平均、對共用者的每張圖算 NLL | `HeadConfig.variant` |

預設是 `large`、`mix`、`vref`，理由：
- 大：格點搜尋要吸收的錯位（最近格點在 |dx|+|dy| ≤ 128 px 內）與照片相對層級約 12 % 的放大是「大」的量級；dependence 之下擾動程度只決定輸入的涵蓋範圍，不讓目標變模糊。
- `mix`：密集輸出（共通圖、latent）要跟著幾何走，全域輸出（cls′）要在同位置內不變，兩者用不同的目標、共用同一個 encoder。不對齊的地圖目標在「大」之下把一階內容幾乎抹光，所以 independence 只用在逐格的二階動差（cls′ 的目標）。
- `vref`：光度模擬幾乎不動相位，twins 的共通圖彼此幾乎相同，`vavg`、`vdist` 在來源軸上多不出什麼。

FM 目前用 gigapath（凍結）；LoRA 與 `fmresidual` 的開關仍在，沒有現役 arm。

### 3.1 slide 與位置
- 訓練用 slide（只用 scan）：`Data.train`（`@pure-train`，100 張 Ki67）。`@pure-test` 不碰；`Data.probe`（`@pure-half-val`，10 張 Ki67）只給探針，與訓練 slide 相交就拒絕啟動。
- `Data.real` 的 16 張 slide 一律不進訓練、不進探針（`DataConfig.forbidden`，遇到即報錯）。
- 取樣層級 ds 1、2、4 … 32（`max_ds`）。
- 位置：開始時列一次。每張 slide、每個層級，用 TileSampler（`data.sampler`，預設 `reference-bank`：背景 < 50 %）在 `data.mask`（預設 `uni2_pca`）遮罩的組織上取不重疊格子的全部位置。一個位置的 footprint = tile 加兩邊最大位移 = 256 + 2 × 64 = 384 px（乘該層 ds），背景比例也以 384 計：位移後的 view 不會跑進鄰格。抽樣結果存在 `data.draw_cache`（`CoaeTrain`；探針的 draw 也在這裡）。
- `data.scale`（0.20）、`turn_deg`（6.0）、`shift_px`（64）是外框：footprint 與讀取大小（`read_px` = 1056 px）依它們算，任何擾動程度都不得超過。
- 池子（`pool_cap`）：每個 (slide, 層級) 格只留 `pool_cap` 個位置（不足就全部；實驗 arm 128，正式 arm 512），整個訓練只從這個池子取。挑法是空間分層：格子排成 Z 序，每 n ÷ cap 個相鄰位置取一個（起點由 `pool_seed`、slide 名稱、ds 決定）。開始時每個位置檢查一次：參考方塊讀得到、中心 tile 不空白（灰階 sd ≥ `blank_sd`）；一格留下通過的，沒有任何通過就報錯。池子與篩選不寫 cache。探針的 Supply 固定以 `pool_cap = 0` 建，不受池子影響。

### 3.2 一個位置的 view
- canonical 組（slot 0）：擾動 0，每個來源一張。ref 是 scan 的原生像素（直接切中心，不經內插）。
- 擾動組（slot 1；擾動程度不是 0 才有）：`dep`、`mix` 抽一次擾動、全部來源共用（twins）；`ind` 每個來源各抽一次。取景用 cubic 內插（縮放、偏轉、位移），之後套鏡頭畸變 k1。
- 同一個來源的 canonical 與擾動兩張是 siblings。
- 模擬來源：`bench`、`bench-real`、`ad-chroma4`，只做光度（色彩、模糊、雜訊；不做 JPEG，因為真實照片存成 bmp 或 png；鏡頭畸變與載台平移一律關掉）。幾何只來自 view 的 `Geometry`；畸變是擾動程度的一部分。來源組見 `Sources.*`。
- 一個位置有 n_view = slots × 來源數 個 view（預設 2 × 4 = 8），連續排列，slot 在外、來源在內（view 的位置內索引 = `slot × n_src + k`）。每個 view 是一張 640 px 的畫布（`photo_px`），網路的輸入是它的中心 tile，目標由整張畫布算。

### 3.3 epoch 與一步
- 一個 epoch：每個 (slide, 層級) 格列池子的 1 ÷ `cell_parts`（至少一個、至多 `cell_epoch`），優先挑跨 epoch 被看過最少次的（同次數先挑最久沒看的，再隨機）。128 的池子用 `cell_parts` 1，512 用 4（`CoaeTrain.sh`：`cell_parts` = cap ÷ 128）。一個 epoch 至多 448 步（池子 128）或 359 步（池子 512）；篩選丟掉位置的格更少。
- 一步：每個 GPU 128 個位置（`DataConfig.places`），每個位置 n_view 個 view，不做梯度累積；一步內同 slide 的 footprint 不相交（粗層先列，相交的退回該格隊尾）。湊不出一步就換 epoch。層級的組成是穩態的：每格的 epoch 預算以累積 credit 均勻攤在每一步；取樣層面不做層級平均（ds 32 全部不到一步），平衡由損失權重負責。
- 層級權重（每步、每個 GPU 自己算）：w(ds) = 這一步的位置數 ÷ (√該層在這一步的位置數 × Σ 各出現層 √位置數)；沒出現的層不參與；每層份量與 √其份額成正比。所有損失都用它。
- 訓練長度 `epochs`（預設 3；正式 arm 12），lr 3e-4（LoRA 1e-4）、AdamW、weight decay 0.05、warm-up `warmup_epochs`（預設 1 個 epoch；`--warmup-epochs`）後 cosine 降到 0、梯度裁剪 1.0。所有損失從第一步一起開，沒有課程。job script 用 1 GPU、12 CPU、200 G、上限 12 小時。
- `--epochs 0`：只建探針、量一次就結束（審核用，不存權重）。`--probe-every N`（預設 1）：每 N 個 epoch 與最後一個 epoch 量一次；checkpoint 每個 epoch 存。
- 續跑（`--resume <k>`）：從 `weights/epoch<k>.pt` 接著跑第 k+1 個 epoch；同一個輸出目錄、同一組旗標、同樣的 process 數，config 任何一欄不同就拒絕；給了才接續 `log.tsv`、`val.tsv`（先裁掉第 k 個 epoch 之後的列），沒給就覆寫；不能用來加 epoch。checkpoint 另存 optimizer、Supply 的 `seen`／`last_seen`、process 數，先寫暫存檔再改名；每個 epoch 開頭以 (seed, rank, epoch) 重設 torch 的種子，不存亂數狀態。續跑不是逐位元重現。目標庫（`nce` 用）不進 checkpoint，續跑後從空的開始填。

### 3.4 供應的實作
- worker process（`data.processes` 10）只讀參考方塊、做空白檢查、抽每個 view 的參數（幾何 `Geometry`、光度 `Shot`），不做像素。參考方塊放在共享記憶體（`Board`，`/dev/shm`），worker 依索引寫入，只回傳判定、各 frame 的幾何與各 view 的計畫。取景與光度模擬在訓練 process 的 GPU 上做（`train.views_of`：`GpuSims.frames`、`GpuSims.simulate`）。`Supply.close` 釋放。
- 同一個 config 的 batch 可重現：view 的種子是 (seed, epoch, step, rank) 的函數。一個 Supply 做過 batch 之後 rank 與 world 固定，不符就報錯。
- log 表的 `sim` 欄是在 GPU 上做 view 的時間；`make:` 欄（cand、wall、pack、read、work、cpu、pids）量供應：供應是瓶頸時 `work` ÷ s/step ≈ 同時在忙的 worker 數，`cpu` ÷ `work` ≈ 1 表示每個 worker 有自己的核心（`cpu` 欄只在 worker 是 process 時有意義）。

## 4. 目標

### 4.1 訊號
參數與 phasetiles 一致：局部相位（log-Gabor 8、24 週期 / 256 px，lg_sigma 0.55，taper 32；每個頻率是一個 3 維單位向量，monogenic）、相位一致圖（尺度 4、8、16、32；3 通道）、T weighted 1-8（1 通道；每張畫布在內部區域 z-score）。共 N = 10 通道。

### 4.2 共通圖（decoder 的目標）
每個 view 的共通圖在它自己的畫布上算，範圍是 FM 實際看到的視野（`field_px`）：訊號在像素級算完、裁出中央的視野、最後才平均成 `out_cells` × `out_cells` 格，每格 8 px（gigapath 224 px → 28 格，uni2 256 px → 32 格；`CoFigure.field_maps`）。沒有 warp、沒有跨幾何的平均，每一格都有效。誰共用目標由機制決定，怎麼合由變體決定（`CoFigure.view_targets`）：
| | vref | vavg | vdist |
|---|---|---|---|
| `dep`、`mix`（twins 共用） | 同 slot 的 ref 的圖 | 同 slot 各來源的圖的平均 | 對同 slot 每個來源的圖算 NLL |
| `ind`（整個位置共用，不對齊） | canonical ref 的圖 | 位置全部 view 的圖的平均 | 對位置每個 view 的圖算 NLL |

`vref` 只算會被讀到的 ref 畫布（`targets_of_views`），結果與全部算再取 ref 相同。

### 4.3 描述子（cls′ 的目標）
- 內容：ref 的共通圖的逐格一階加二階原始動差：10 個通道的平均，加上每個訊號內部的平方與交叉乘積（lp8 6、lp24 6、相位一致 6、T 1，共 19），M = 29 個通道；乘積在像素級算、之後才平均；4 × 4 格（`head.pool_cells`），共 29 × 4 × 4 = 464 個數（`CoFigure.place_moments`）。
- `mix`、`ind`：取 canonical 的 ref，每格的平均範圍向外擴「該擾動程度的位移」（大 = ±64 px；像素級、兩軸獨立的 box 平均），整個位置的 view（canonical 組與擾動組、所有來源）共用。這是「位移 siblings 各自的描述子」的期望的封閉式，不必真的抽 n 個 siblings。`dep`：取同 slot 的 ref、不外擴，同 slot 的 view 共用（`CoFigure.shared_moments`）。所以位置的描述子是位置的固定函數，不隨訓練變。
- 為什麼是二階：帶通訊號在 64 px 內轉好幾圈，一階平均接近 0；二階（能量、方向張量）留得住。為什麼取 canonical：gallery 的 tile 永遠是 canonical，擾動組的 view 去還原它，等於要求對尺度、偏轉、畸變也不變。
- 寫死的形式：原始動差（不中心化、不做 power 或 L2 正規化、不用 NLL；MSE 回歸得到條件期望，而原始動差對「把 siblings 的分布混在一起」是線性的）；外擴的 box 必須與 `LEVELS` 的位移抽法是同一個分布（改一個，另一個要跟著改）；目標只由 ref 的畫布算、不回傳梯度、不經任何可學的模組。4 × 4 格在外擴 ±64 px 之後，實際的空間解析度約 2 × 2。
- `direct`（由該 view 的像素直接算的描述子，不訓練就有）與學習版的關係：`direct` 只對輸出就是目標的特徵（`place`、`cofig`）在沒有光度模擬的 `sib:*` 列是上限（超過即量測有錯）；在 `twin`、`both` 列，`direct` 套在擾動 view 上算，學習版的輸出可以超過它（學到的去擾動來自模擬，要由 `ood-*` 列與真實照片看管）。對中間特徵（`cls`、`latent`）它不是上限。

## 5. 損失

下表是損失的清單，每一列對應 `LOSS_TERMS` 的一個權重鍵（`koleo`、`adv` 沒有 arm 字母）；權重寫在 `HeadConfig.w_<鍵>`。下限指損失可達的最小值，由定義推得；n_view = 一個位置的 view 數（slots × 來源數），n_src = 一個 (位置, slot) 的來源數。損失目前是 `train.py` 裡各一段（尚無註冊制，見「已知的限制」）。
| 損失（權重鍵） | arm 字母 | 作用的輸出 | 標準權重 | 下限 | 內容 | 限制 |
|---|---|---|---|---|---|---|
| 重建（`recon`） | `R` | decoder 輸出 `cofig` | 1 | 0 | vref、vavg：對共用目標的 MSE，每個通道先除以它的 running sd；vdist：vMF（方向向量）與高斯（β-NLL，β 0.5）的 NLL，對共用者的每張圖；log 另外逐訊號列出（lp8、lp24、pc、tw） | 為 0 只在帶對比的實驗 arm |
| 描述子（`pool`） | `P` | 線性頭 `head` | 1 | 0 | cls′（訓練時加雜訊）經一層線性預測位置描述子（「目標」節）；MSE，每個通道除以描述子自己的 running sd；這是唯一直接告訴 cls′ 要裝什麼的損失 | 為 0 是消融：算並記錄、乘 0 |
| 目標對比（`nce`） | `N` | 線性頭 `head`（標準化單位） | 0.5 | 0 | 一個 view 的輸出對各位置的 ref 描述子做 InfoNCE（cos、τ 0.1）：正例是自己位置的描述子，負例是同一步其他位置的、以及目標庫裡同 (slide, 層級) 格其他位置的；庫放在 GPU、記原始單位的描述子，不會過期；沒有 view 對 view 的比較 | 需要每位置一個描述子（`mix`、`ind`）與 `w_pool` > 0 |
| view 對比（`vnce`） | `V` | cls′ `cls` | 0.5 | log(n_view − 1) | 同位置的其他 view 為正例、同一步其他位置的 view 為負例的多正例 InfoNCE | 只給實驗 arm |
| head 對比（`hnce`） | `H` | 線性頭 `head` | 0.5 | log(n_view − 1) | 同 `vnce`，作用在線性頭的輸出；用來取代描述子損失 | 只給實驗 arm |
| decoder 區塊的 twin 對比（`dnce`） | `D` | decoder 輸出 `cofig`，一個 latent 格對應的區塊（gigapath：4 × 4 格 × 10 通道 = 160 維） | 0.5 | log(n_src − 1) | 同一個 (位置, slot) 的不同來源（twins）互為正例，負例只取其他 (位置, slot) 在同一個格位置的向量；每步抽 `twin_cells` = 8 個格位置（全批次相同）；用來取代重建 | 只給實驗 arm；需要 twins、點輸出 |
| latent 的 twin 對比（`lnce`） | `L` | latent | 0.5 | log(n_src − 1) | 同 `dnce`，直接加在 latent 上（`dnce` 的對照） | 只給實驗 arm；需要 twins |
| KoLeo（`koleo`） | 無 | cls′ | 0 | 無 | 同來源內的 KoLeo | 預設不算；沒有名字 |
| 對抗（`adv`） | 無 | 來源判別器 | 0 | 無 | 判別器分辨來源，作用在 latent 與 cls′ | 預設不建判別器；沒有名字 |

- 每個輸出都要有內容損失：`w_recon` 為 0 只在帶對比的實驗 arm（`vnce`、`dnce`、`lnce`）允許；`w_pool` 為 0 是消融，實驗 arm 可用 `hnce` 取代。`nce` 的標準權重 0.5；`tau` 的預設 0.1（兩種對比共用，不進 arm 名）。
- 所有損失都依位置的層級權重加權。`w_recon`、`w_pool` 為 0 時兩項照算、乘 0 後照加，所以 DDP 沒有無梯度的參數，log 照有這兩欄。
- 主線的不變性與區分性只來自訊號目標：不同來源、不同擾動的 view 之間不直接算相似度損失，也沒有以來源為對象的損失；來源不變性來自「twins 的目標是同一張」，幾何不變性來自「同位置的 cls′ 目標是同一個」。
- 實驗 arm 的例外（使用者，2026-11-16）：上一條的本意是，直接在模型的輸出上做跨來源的對比，可能只學到看過的來源、沒看過的來源就失效。為了量這件事，允許帶 `V`、`H`、`D`、`L` 的實驗 arm 加 cls′ 或線性頭上的 view 對比、decoder 輸出或 latent 上 twins 之間的對比，並可把 `w_recon`、`w_pool` 設為 0。這些 arm 只用來比較來源泛化（訓練時留出 `ad-chroma4`）：不得當預設 arm 或正式 arm，不得當 `M-realtest` 的被測 arm。本條是否放寬，等實驗結果由使用者決定。`nce`（`N`）不是例外：它的正例是位置的描述子，不是別的 view。

## 6. arm

arm 有兩個名字，語法與修飾表在 `locascope/training/NAMES.md`（arm 代號 `CoaeConfig.arm`，結果目錄與 log 的長名 `CoaeConfig.dir_name`；兩者由 `Config.py` 的表產生、`config_of_arm` 與 `config_of_dir` 互轉）。預設 arm 是 `RP3`（長名 `large-mix-vref`）。結果在 `result/<job>/<長名>/`，log 在 `log/<job>_<長名>`。

job script `deploy/slurm/CoAE/CoaeTrain.sh` 一個 arm 一個 job，以環境變數選 arm，`train.py --print-arm` 向 `CoaeConfig.arm` 要名字：
| 環境變數 | 旗標 | 效果（arm 代號中的位置） |
|---|---|---|
| `COAE_LEVEL`、`COAE_MECH`、`COAE_VARIANT` | `--level`、`--mechanism`、`--variant` | `noperturb`／`lightperturb`、`dep`／`ind`、`vdist`／`vavg` |
| `COAE_SIMS=<a,b\|none>` | `--sims` | 來源個數 3、2、0；非固定組加 `sims-<a+b>` |
| `COAE_NOISE`、`COAE_POOL` | `--noise`、`--pool-cells` | `noise<σ>`、`desc<c>`（描述子的格數） |
| `COAE_ENCODER`、`COAE_LORA`、`COAE_FMCLS` | `--encoder`、`--lora`、`--cls-residual` | `fm-<名>`、`lora`、`fmresidual` |
| `COAE_WRECON`、`COAE_WPOOL` | `--w-recon`、`--w-pool` | `R`、`P` 在核心中去留；權重非標準時 `weight<字母><值>` |
| `COAE_NCE=1` | `--w-nce` | `N` |
| `COAE_VNCE=1\|only` | `--w-vnce` | `V`；`only` = 權重 1、沒有 `recon` 與 `pool`（`xV…weightV1`） |
| `COAE_HNCE=1`、`COAE_DNCE=1`、`COAE_LNCE=1` | `--w-hnce`、`--w-dnce`、`--w-lnce` | `H`、`D`、`L`（`COAE_DNCE` 與 `COAE_LNCE` 不能同時） |
| `COAE_CAP`、`COAE_SLIDES`、`COAE_EPOCHS`、`COAE_SEED` | `--pool-cap`、`--slides`、`--epochs`、`--seed` | `pool<n>`、`slides<n>`、`epochs<n>`、`seed<n>` |
| `COAE_FM_DTYPE`、`COAE_COMPILE`、`COAE_WARMUP`、`COAE_RESUME`、`COAE_ARGS` | 見 script | 不進名字 |

`COAE_POOL`（`desc<c>`，描述子的格數，必須整除 Co-figure 的格數；gigapath 是 2、4、7）與 `COAE_CAP`（`pool<n>`，池子）是兩件事。`desc8`（8 × 8 格的描述子）只用在不訓練的審核；要訓練它時 σ 要一起重新考慮。

現有 arm（長名在 NAMES.md 的 arm 表）：
| arm | 組成 | 回答 |
|---|---|---|
| `RP3` | 預設：重建加描述子，三個來源 | `CoAE.Q1` |
| `RP0` | 只有 ref | `CoAE.Q2` |
| `RP3.dep`、`RP3.ind`、`RP3.vdist`、`RP3.fmresidual`、`Rx3` | 各改一項；`Rx3` 沒有描述子 | `CoAE.Q3` |
| `RN3` | 加目標對比 | `CoAE.Q4` |
| `RP2`、`RP2.seed1` | 留出 `ad-chroma4`；兩個種子 | `CoAE.Q5`、`CoAE.Q6` |
| `xV2.weightV1`、`RPV2`、`RPV2.seed1` | 只有 cls′ 的 view 對比；重建加描述子加 cls′ 的 view 對比 | `CoAE.Q5` |
| `RH2`、`DP2`、`DH2` | 全域支、局部支、兩支換成對比 | `CoAE.Q6` |
| `LH2`、`RV2` | 對比加在 latent；對比加在 cls′ | `CoAE.Q6` |
| `RP3.pool16.epochs24`、`RP3.slides50.epochs6` | 每格位置數 1/8、slide 減半，總步數相同 | `CoAE.Q7` |
| `RP3.pool512.epochs12`（核心依 `CoAE.Q6`） | 正式的版本：池子 512、12 個 epoch、FM 全精度 | `CoAE.Q8` |
| `RP3.epochs0`、`RP3.noperturb.epochs0`、`RP3.lightperturb.epochs0`、`RP3.desc8.epochs0` | 訓練前審核（`CoaeAudit.sh`，job `ExpCoaeAudit`）：只建探針、量一次 | `CoAE.Q1` 的前置 |

## 7. 量測與判準

### 7.1 探針（訓練流程裡唯一的量測；`Data.probe`，沒有真實照片）
- slide：`Data.probe` 的 10 張 Ki67，沒訓練過、也沒有真實照片。
- 位置：層級 ds 1、2、4、8，每個 (slide, 層級) 格取 64 個不重疊的位置（不足 64 個的格不用），共約 2560 個。開始時建一次（固定種子），之後每次量測都是同一組；tile 與特徵只在記憶體，不寫檔。
- 每個位置 14 個 view（`Supply.probe_plans`；像素在 GPU 上做）：
  - `anchor`：ref、擾動 0，也是 gallery。
  - `twin:<來源>` × 5：擾動 0、只差來源：三個訓練用的模擬，加兩個任何 arm 都不訓練的來源 `ood-heavy`（defocus 3、noise 6、chromatic 5、色溫 ±0.25；沒有 JPEG）與 `ood-soft`（解析度損失 1.5–2.5 倍、相關雜訊）。
  - `sib:interp`、`sib:light`、`sib:large`：來源 ref、只差幾何（只偏 0.5 px 的內插、輕、大）。
  - `both:<來源>` × 5：來源與大擾動同時，最接近搜尋時的一對（它是模擬，不是真實照片）。
- 三類 view 各量一軸：`twin` 是來源軸（dependence 的成果），`sib` 是幾何軸（independence 的成果），`both` 是兩者同時。
- 特徵（`Validate.PROBE_FEATURES`；每個量兩次：直接 cos，與先減掉 anchor 的平均再 cos 的 `<名稱>-m`）：
  - `fm`（FM CLS）、`fmavg`（CLS 與 patch 平均相接）、`cls`（cls′）、`place`（cls′ 的線性讀出）、`fm+cls`、`fm+place`。
  - `head`（線性頭的輸出，以 checkpoint 的 `pool_mu`、`pool_sd` 為單位；與 `place` 差一個所有位置共用的向量）與 `fm+head`。帶描述子損失的 arm 讀 `head`，不讀 `place`。
  - `latent`（s×s×128 攤平：gigapath 6272 維、uni2 8192 維，逐格比較；一格 = 32 px，所以 64 px 的位移是 2 格，`sib:large`、`both:*` 在這幾列對位移更敏感是形狀造成的；不訓練的對照欄是同一個格子的 `cofig-direct<s>`）、`latent-avg`（s² 格的平均，128 維：不看排列、耐位移）、`cofig`（decoder 輸出）。
  - 不訓練的對照：`direct`、`fm+direct`、`cofig-direct`（該 view 自己的畫布，光度模擬之後，算出的 Co-figure，攤平：gigapath 10×28×28）、`cofig-direct<g>`（平均到 g×g 格，g 取 `Validate.pooled_sides`：2、4 與 latent 的邊長 s）、`fm+cofig-direct`、`cofig-direct:<訊號>`（只取一個訊號的通道：`lp8`、`lp24`、`pc`、`tw`）。
  - 合成特徵的每個部分各自單位化後串接；描述子類的特徵用探針自己的 sd 做尺度（`cofig-direct`、`cofig-direct<g>` 各用自己的 sd，來自 anchor 的圖）。
- `cofig-direct` 是「decoder 的目標函數套在該 view 自己的像素上」。在 `vref` 加 twins（`dep`、`mix`）下：
  - `sib:*` 列：`cofig-direct` 是完美 decoder 的輸出，逐值相同，這三列是 `cofig` 的 oracle。
  - `twin:*` 列：完美 decoder 的輸出等於 anchor 的圖（hit 1、consumed 0）；`cofig-direct` 量的是訊號本身對來源的不變性，`cofig` 高出它的部分是從模擬學來的。
  - `both:*` 列：完美 decoder 的輸出與 `sib:large` 列同分布，所以 `both:*` 的 oracle 讀 `sib:large` 列。
  - `latent`、`latent-avg` 不是 decoder 的輸出，`cofig-direct*` 對它們不是上限。
  - `ind`：完美 decoder 每列 hit 1，這個上限沒有資訊；`cofig-direct` 仍是像素基準。
- `fm`、`fmavg`、`direct`、`fm+direct` 與所有 `cofig-direct*` 不隨訓練變（FM 凍結時），每個 epoch 應一模一樣，這是探針沒壞的檢查。`cofig-direct*` 與 `map-direct` 另外不隨 arm 變。`direct` 會隨 arm 的同位置範圍（`dilate_px`）變，跨機制比較時不是同一個基準。
- 輸出的表分三個 block（cls′ 支、latent 支、逐訊號），`fm` 在前兩個 block 都當參考；每個 block 的第一列 `anchors nn/neg` 是 anchor 之間的間距。

### 7.2 指標（`Validate.probe_scores`、`probe_measures`）
| 指標 | 定義 | 用途 |
|---|---|---|
| hit（主要） | 查詢 view 對同 (slide, 層級) 格的 64 個 anchor 做 cos，自己的 anchor 排第一的比例 | 格點搜尋對一張 tile 做的事的縮小版；一致性與區分性合成一個數 |
| consumed | (1 − cos(view, 自己的 anchor)) ÷ (1 − anchor 對其他 anchor 的平均 cos)，中位數 | 0 = 沒被動到，1 = 和別的位置一樣遠 |
| z | (正確位置的 cos − 錯誤位置平均) ÷ 錯誤位置的 sd，中位數 | 只當描述，不用來選 |
| spread | (1 − 錯誤位置的平均 cos) ÷ 錯誤位置的 sd，中位數；z = (1 − consumed′) × spread（consumed′ 對該查詢自己的錯誤位置平均） | 錯誤的 anchor 離「完全相同」有幾個 sd：特徵把位置展開多開 |
| map | decoder 的共通圖對 anchor 自己的共通圖的 cos，逐訊號；anchor 與每個 `twin:*` 各一列 | 讀取在訊號空間的來源不變性 |
| map-direct | 每個 `twin:*` view 自己的 Co-figure（不經網路）對 anchor 自己的 Co-figure 的 cos，逐訊號 | `map` 的不訓練基準 |
| anchors nn/neg | 每個 anchor 對同格其他 anchor 的最大與平均 cos，中位數 | 1 − nn 是查詢還能損失多少 cos 而不被別的位置超過 |
| rank | anchor 的 cls′ 的有效秩 | 塌縮檢查 |

`val.tsv` 的每一列另有 `by_cell`（`fm`、`cls`、`head`、`place` 四個特徵、原樣 cos、每個 view 種類逐格的 hit、consumed、z、spread，供格重抽）與 `cell_names`。

### 7.3 頻率與輸出
- 訓練前量一次（epoch −1：基準欄與未訓練的網路），之後每個 epoch 量一次；每個 epoch 存 `weights/epoch<k>.pt` 與 `last.pt`。
- `result/<job>/<長名>/`：`log.tsv`（訓練表）、`val.tsv`（每次量測一列 `probe<TAB>epoch<TAB>json`）、`config.json`、`weights/`。
- 選 checkpoint：`both:ood-*` 兩列 hit 的平均最高的 epoch，平手取較晚的；選 `fm+place`、`place`、`cls` 三者（帶描述子損失的 arm 以 `head` 與 `fm+head` 代 `place`、`fm+place`）裡同一個數最高的特徵做最後驗證。

### 7.4 判準（`Gate.CoAE.*`；都只用模擬）
模擬上的量測只能證明必要條件成立，不能證明真實照片上有效。
| 判準 | 條件 |
|---|---|
| `Gate.CoAE.signal-untrained` | 下界式檢查，量「不必學就有用」，不是上限；不過不能推出「讀得完美也幫不了 FM」，只能推出這個描述子的不變性要靠網路從模擬學來。cls′ 支：`fm+direct` 在 `both:ood-*` 兩列的 hit 平均 ≥ `fm` 的 + 0.05，且每一列 ≥ `fm` − 0.02；不過就不訓練，回來改目標 |
| `Gate.CoAE.map-untrained` | latent 支的停止條件：(a) `cofig-direct` 在 `sib:interp` 的 hit ≥ 0.95；(b) `twin:ood-*` 兩列，至少 `cofig-direct:lp8` 與 `cofig-direct:pc` 的 hit ≥ `fm` 同列 − 0.02；不過就不訓練，回來改訊號 |
| `Gate.CoAE.map-with-fm` | latent 支，只當訊號：`fm+cofig-direct` 在 `both:ood-*` 兩列的 hit 平均 ≥ `fm` 的 + 0.05，且每一列 ≥ `fm` − 0.02；不過不停止 |
| `Gate.CoAE.trained` | 最後一個 epoch：`head`（線性頭的輸出，扣掉共用向量）或 `cls`（cross-attention 輸出的 cls′）在 `both:ood-*` 的 hit ≥ `fm` 同一列 + 0.05；`map` 的 lp8 在三個訓練用來源的 `twin` 列 ≥ 0.6 且逐 epoch 不降 |
| `Gate.CoAE.fm-kept` | `fm+place` 在每一列的 hit ≥ `fm` − 0.02 |
| `Gate.CoAE.from-target` | `RP0` 的 `place` 在五個 `twin` 列的 consumed 都比 `fm` 低三成以上 |
| `Gate.CoAE.source-gap` | 見下 |
| `Gate.CoAE.branch-swap` | 見下 |

`Gate.CoAE.source-gap`（`coae_gap.py` 照下列算；輸入是各 arm 的 `val.tsv` 最後一列的 `by_cell`）：
- 記號：格 j = 1..40；看過的來源集合 seen = {`bench`、`bench-real`}，沒看過的 unseen = {`ad-chroma4`、`ood-heavy`、`ood-soft`}。C(arm, f, s) = 格的 `twin:s` consumed 平均；r = C(arm, f, s) ÷ C(arm, `fm`, s)（同一個 `val.tsv` 的 FM：來源本身的難度）；GAP(arm, f) = unseen 上 r 的平均 − seen 上 r 的平均；Z_unseen、Z_seen = unseen、seen 上各來源的 `both:s` 的 z 的格平均。
- 特徵：`xV2.weightV1`、`RPV2` 用 `cls`；`RP2` 用 `head`；`RPV2` 的 `head` 並列、不進規則。比的是各 arm 內的落差，不直接比不同幾何的 z。
- 量：D_gap = GAP(`xV2.weightV1`, `cls`) − GAP(`RPV2`, `cls`)；D_z = Z_unseen(`RPV2`, `cls`) − Z_unseen(`xV2.weightV1`, `cls`)。區間：格重抽 1000 次（種子 0；同一組格同時用在兩個 arm 與 `fm` 上），取 2.5–97.5 百分位。run 間雜訊 ε：`RP2` 與 `RP2.seed1`、`RPV2` 與 `RPV2.seed1` 兩對重複各算 |GAP 差| 與 |Z_unseen 差|，各取兩對的大者。
- 飽和檢查（先判；觸發就只印數字、結論寫「負例太容易，本輪不下結論」）：`xV2.weightV1` 的 epoch 0 最後一列的 `vnce` 超出下限 log 5 不到 0.2，或其 `features.cls['sib:large'].hit` < 0.5。
- 一致（約束讓泛化較好）：D_gap 區間下緣 > 0；D_gap ≥ max(0.05, 2 ε_gap)；D_z 區間下緣 > 0；Z_seen(`RPV2`, `cls`) ≥ Z_seen(`xV2.weightV1`, `cls`) − 0.3。不一致（無約束的不比有約束的差）：D_gap 區間上緣 < 0.05；Z_unseen(`xV2.weightV1`, `cls`) ≥ Z_unseen(`RPV2`, `cls`) − max(0.3, 2 ε_z)；Z_unseen(`xV2.weightV1`, `cls`) ≥ Z_unseen(`RP2`, `head`)；沒有觸發飽和檢查。其餘為未定。把 `RPV2` 換成 `RPV2.seed1` 重算，兩個結論不同則寫未定。
- 後續：一致 → 主線保留約束；不一致 → 主線改做對比，但要真實照片同向（`xV2.weightV1` 的 `coae cls` raw z 區間不低於 `RP2` 的 `coae head` 與 `RPV2` 的 `coae cls`）；未定 → 補重複與其他組合。

`Gate.CoAE.branch-swap`（2×2：局部支用重建或對比 × 全域支用描述子或對比；格 = `RP2`（重建、描述子）、`RH2`（重建、對比）、`DP2`（對比、描述子）、`DH2`（對比、對比）；`coae_gap.py` 照下列算，特徵以 `head` 為主、`cls` 並列）：
- 效果（m = 格的某個量，鍵 (局部支用重建, 全域支用描述子)）：cofig_alone = m[1,0] − m[0,0]；desc_alone = m[0,1] − m[0,0]；mixed = m[1,1] − m[0,0]；main_A、main_B 為兩支的主效果；inter = m[1,1] − m[1,0] − m[0,1] + m[0,0]。量與最小效果 m0：Z_unseen 0.3、Z_seen 0.3、GAP 0.05（越小越好，取負號）。
- 門檻：T = max(m0, 2 ε)（`inter` 用 max(m0, 3 ε)）；up = 效果 ≥ T 且重抽區間下緣 > 0，down 同理反向。較好 = Z_unseen 或 GAP 有 up 且沒有 down；較差 = 有 down 且沒有 up；兩面 = 一個 up 另一個 down；其餘看不出。`inter` 的較好稱互補，較差稱替代。
- 對照只列差、不下結論：`RH2` 對 `RV2`（head 對比對 cls′ 對比）、`DH2` 對 `LH2`（decoder 區塊對 latent）。
- 真實照片同向才談是否放寬「損失」節的例外：`coae_gap.py real` 讀各格的 `M-tile` 結果，raw z 依同一組效果式組合。

### 7.5 真實照片：只在最後驗證一次
- 選定 checkpoint 與特徵之後才跑；訓練中不記錄任何真實照片的量，結果不回頭改選擇。`train.py` 不匯入這些函式。
- CoAE 是 stage 2 的方法，只換 stage 2 的 tile 特徵（當滑動視窗的特徵用）；stage 1 與 stage 3 照 `M-realtest` 的現行流程不動。`Validate.CoaeTileEncoder` 把選定的特徵當 tile 特徵；`Validate.real_cofig`（預測的共通圖對照片自己算的共通圖，目標用 FM 的視野）留給這一次用。
- `Gate.shared.tile`（`M-tile`，tile 層級，只看 raw z；消耗與展開只當診斷）：z 的區間下緣 > 2.33 才送 `M-realtest`；中位數 ≥ 3.4 且下緣 ≥ 3.1 另標「正常」。比較兩個 arm 時，區間不重疊且較高才算有幫助。
- `Gate.shared.realtest`（`M-realtest`，全部 2009 張；找到 = 非 retrieval-only 且 ncc ≥ 0.6）：
  | 找到的張數 | 結論 |
  |---|---|
  | ≥ 1805 | 追平 phasetiles |
  | ≥ 1708 | 通過（85 %；使用者 2026-11-15） |
  | 1419–1707 | 有幫助、未通過；另需 slidewin gigapath 找到的 1319 張裡遺失 ≤ 66、真值命中 ≥ 1305，遺失條件不符時降為「互有得失」 |
  | 1269–1418 | 持平 |
  | < 1269 | 變差 |
- 基準（找到的張數）：phasetiles 1805、slidewin uni2 1690、slidewin gigapath 1319、winembed gigapath 1145。
- 報告並列：用過（`Photos.sampled`）與沒用過（`Photos.unsampled`）兩組、與三個基準成對的「只有一方找到」、真值命中（答案在 checklist `correct == 1` 位置的 24 個 level-0 px 內）、逐 slide 的找到數。凡是看過 `Photos.sampled` 才選出的用法，結論只在 `Photos.unsampled` 內判，而且只讀對 slidewin gigapath 的差：多 ≥ 2.5 個百分點 = 趨勢重現；兩組的絕對找到率不能互比。

## 8. 程式組成

`locascope/training/CoAE/`：
| 檔案 | 內容 |
|---|---|
| `Config.py` | 所有超參數（config 欄位）；`LEVELS`、`MECHANISMS`、`VARIANTS`；`ProbeConfig`；arm 代號與長名的表（`LOSS_TERMS`、`NAME_PARTS`、`DIR_PARTS`、`UNNAMED`）與 `config_of_arm`、`config_of_dir` |
| `Sims.py` | lab（`query_sim` 的 gap 加 scratch `LabGap` 的移植，只做光度、沒有 codec）、`Shot`、`draw_shot`（抽光度參數，不做像素）、`Geometry`、`draw_geometry`（依擾動程度抽）、`frame`（取景加畸變的 CPU 定義：測試的參考） |
| `GpuSims.py` | GPU 上的取景（`frames`）與光度模擬（`simulate`）；與 CPU 版的對照在 `tests/TestCoAE/test_sim_gpu.py` |
| `CoFigure.py` | 訊號、`field_maps`、`view_targets`、`targets_of_views`、`place_moments`、`shared_moments`；舊的對齊路徑（`place_targets` 等）留著，訓練不呼叫 |
| `Supply.py` | 位置清單與池子、epoch 的列單、讀 scan 與抽 view 的參數（worker process 加共享記憶體）、組 batch；探針（`probe_plans`、`probe_set`） |
| `Model.py` | FM token 與視野、層混合、LoRA、Swin encoder、瓶頸雜訊、ConvNeXt decoder、attn_pool、cross-attn、線性頭（標準化單位）、判別器（只在 `w_adv > 0`） |
| `Losses.py` | 重建（point、dist，逐訊號）、描述子的 MSE、`target_nce`、`view_nce`、`twin_nce`、`decoder_blocks`、`agreement`；KoLeo、對抗（預設不呼叫） |
| `Validate.py` | 探針的特徵與分數（`probe_features`、`probe_scores`、`probe_cell_scores`、`probe_measures`）；`CoaeTileEncoder`、`real_cofig`（最後一次真實驗證用） |
| `LogLayout.py` | 訓練表、探針表的排版 |
| `train.py` | 主程式（DDP，一個 GPU 一個 process；`--print-arm`） |
| `final.py` | 選定 checkpoint 的一次性檢查：`probe` 子命令（多算 `head`、`fm_head`，寫 `probe.json`） |

stage 2 的接法：`core/stage2_retrieval/ArmSpec.py`（arm 文法與 `COAE_PARTS`）、`RetrieverCoae.py`。

job script（`deploy/slurm/CoAE/`）：`CoaeTrain.sh`（一個 arm 一個 job）、`CoaeAudit.sh`（訓練前審核）、`CoaeProbeCheck.sh`（對既有 checkpoint 重量探針）、`CoaeRealCofig.sh`、`TestCoae.sh`（單元測試 `tests/TestCoAE/test_coae.py`、`test_coae_slide.py`，加小訓練）。

## 9. 已知的限制
1. 層混合的層、初始權重、上限不能用任何真實照片決定；目前預設每 4 個 block 取一層（4–40）、初始權重相同、不設上限。要設先驗只能從 scan 與模擬量。
2. 內插捷徑：canonical 的 ref 是原生像素，擾動組都被內插過；網路可能學成「原生 = canonical」，真實照片是原生像素，尺度不變性那部分可能不轉移。`sib:interp` 看各特徵對內插有多敏感；實際影響只有真實照片答得了。
3. 「FM token 在真實照片上讀得出訊號」目前只能靠最後的真實驗證確認。
4. 畸變在視野內很小（k1 = 0.10 時 tile 邊緣約移 2 px），`sib:*` 的結果不能讀成「對畸變穩健」。
5. independence 的「對齊」版本沒有接線（它等同 `dep` 加 `vref`，只多了 warp 的內插）；`ind` 的目標是不對齊的。
6. 開 LoRA 時 FM 本身會變，`fm`、`fmavg` 不再是跨 epoch 不變的基準。
7. `--encoder uni2`：`fm.layers` 預設到第 40 層，換 encoder 時沒有跟著換，也沒有守門檢查；跑 uni2 之前要先處理。
8. `--cls-residual` 時，梯度診斷列的 pool → cls′ 印成 0（參數收到的梯度沒有錯）。
9. 網路的初始權重沒有固定種子；同一個 arm 送兩次起點不同，兩次訓練不逐位相同。
10. `place` 含所有位置共用的固定向量 `pool_mu / pool_sd`，位置之間的 cos 全在 0.99 以上，所以帶描述子損失的 arm 讀 `head`（`place` 只當對照）。
11. 損失還沒有註冊制：每個損失在 `train.py` 裡各一段；新增損失要同時改 `train.py` 與 `LOSS_TERMS`。
12. 降低光度模擬的 CPU 成本暫不做（1 GPU 時 GPU 約有六成時間在等資料）。
13. LoRA rank 8（作用在 attention 的 qkv）、Swin 每階段 2 個 block、ConvNeXt 每個解析度 2 個 block，都依建議，沒有實驗依據。

## 10. 變更紀錄
| 日期 | 改了什麼 | 為什麼 | 出處 |
|---|---|---|---|
| 2026-11-07 | 規格定稿：重建加 cls′ 的內容損失、瓶頸雜訊、探針 | 不變性與區分性只來自訊號目標 | 本檔前身 |
| 2026-11-15 | 改成 twins 與 siblings 的設計；共通圖鋪滿 FM 的視野；描述子 29×4×4；探針改在 `Data.probe` | 舊設計的 view 沒有只差一個因素的配對；目標與輸入的格對不齊 | 本檔前身 |
| 2026-11-15 | 固定池子、1 GPU、每 GPU 128 個位置、續跑、`--probe-every` | 實驗 arm 要在 1.6 GPU-h 級的成本下跑完 | 本檔前身 |
| 2026-11-16 | 探針加 `head`、`fm+head`、`spread`、`by_cell`；損失加 `nce` | 缺的是展開，不是消耗；結果要能格重抽 | 變更書 |
| 2026-11-16 | 損失加 `vnce`、`hnce`、`dnce`、`lnce`（只給實驗 arm）；加實驗 arm 的例外條款 | 量來源泛化與兩支各用訊號目標或對比 | 使用者 2026-11-16 的決定 |
| 2026-11-16 | arm 改用代號與長名兩套名字，由 `Config.py` 的表產生；代號字典搬到 `NAMES.md` | 一個名字只有一個來源；舊代號停用 | 名稱體系 v2 與文件規範 |
| 2026-11-16 | 判準改名為 `Gate.CoAE.*`、`Gate.shared.*`；`M-realtest` 的分段合成一套；PhaseQuery 放棄 | 判準的名字要說出它檢查的事 | NAMES.md |
| 2026-11-16 | 全文依文件規範重排：只寫現況，歷史移到本表 | 舊內容混有已取代的設計與狀態 | 文件規範 |
