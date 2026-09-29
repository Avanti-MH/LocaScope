# PrototypicalRoutingHead — 計畫

`spec.md` 是規格：架構、每個 stage 的選項、模組沿用/新增的決定與理由。這份是
**現在要做什麼、按什麼順序、做到哪**，還沒開始動工（2026-09-19 訂）。

```
0. 三個 open design decision 先拍板（spec.md「Open design decisions」）
1. Stage 1+2+3 主線打通一條完整路徑（不含任何支線/觀察臂）—— 已完成三個模組
   1.1 共用的 Pooling 模組（Avg/Max/Attn/passthrough），support 跟 query 共用
   1.2 Set Transformer prototype generator（主線）
   1.3 Cosine + learnable tau 路由頭（主線）
   1.4 Checkpoint 存讀（Open design decision 1 拍板後才能做）
2. Loss + episodic 訓練迴圈 —— 這一步訓練的是「Metric-based meta-learning」
   這個 arm（Stage 4 三選一之一，spec.md 2026-09-20 新增的分類）
   2.1 L_bal（沿用 MppRoutingHead 的 class_weights 公式，先跑通訓練迴圈本身）
   2.2 L_ord_A / L_ord_B 補上（spec.md 的 3a/3b，這次是從頭就有，不是後補）
   2.3 Support/query WSI 同一張 vs 不同張的機率 p_same_wsi（spec.md 新增段落）
3. Metric-based 主線完整跑一次，上 stage1_compare 計分板跟 KnnEstMpp/
   MppRoutingHead 比
4. Baseline prototype learning arm（Stage 4 三選一之二）—— 便宜，跟
   MppRoutingHead 自己的訓練形狀幾乎一樣（一般 supervised 訓練 embedding，
   prototype 是推論時事後平均出來的），當作下限對照 —— 已完成（架構+訓練
   迴圈，還沒跑過；不等 metric-based 有分數才動工，因為兩邊訓練互不依賴）
5. 主線+baseline 分數都出來後，逐一加觀察臂（一次只加一個，跟主線分開比）
   5.1 Stage 2：mean/median、shared MLP（已完成架構，還沒跑過分數）、
       Matching Net（跟 5.2 共用同一個 cross-attention 模組，還沒動工）
   5.2 Stage 3：非學習路由頭（負歐氏距離、固定溫度 cosine、multi-prototype
       soft-min）、可學習 Arm 1（bias routing）、可學習 Arm 2（跟 5.1 的
       Matching Net 共用模組）
6. Optimization-based meta-learning arm（Stage 4 三選一之三）—— 最大的一塊,
   MAML 式 inner loop,full second-order 還是 FOMAML/Reptile 還沒拍板
   （spec.md「Open, not yet decided」）,排在 3/4 都有分數之後才動工
7. Stage 1 的兩個支線（混合 foundation models、fine-tuned CNN）—— 排最後，
   主線分數穩定之後才考慮
```

**已完成（2026-09-20 追加）：多 arm 訓練骨架**——使用者確認的最終形狀是三條軸
疊在一起：Stage 4 訓練方式（baseline / metric-based / optimization-based）
x Stage 2 generator（Set Transformer / shared MLP / Matching Net）x Stage 3
routing head（Cosine/τ / 交叉注意力度量路由）。這次只搭骨架，沒有動工任何一個
還沒存在的 arm（5.1/5.2/6 仍然是「一次加一個」，沒有改變）：
- `Runtime.py`（新建）：`GENERATOR_CHOICES`/`ROUTING_HEAD_CHOICES` 兩個獨立
  registry，現在各只有一個 entry（`set_transformer`、`cosine_tau`）。每個
  entry 是一個 builder function（`(in_dim, device) -> module` 或
  `-> (module, cfg)`），不是 `(class, config_class)` tuple——因為還沒有第二個
  entry 可以驗證這個 tuple 該長什麼樣子。
- `cli/train.py` 加 `--generator`/`--routing-head`，從 registry 查表建構；
  checkpoint 檔名跟 `extra` 都記錄了這兩個選擇。
- `cli/train_baseline.py` 加 `--classifier`（讀 `Heads._CLASSIFIER_REGISTRY`，
  不是讀上面那個新 registry——兩邊的呼叫形狀不一樣，`LearnedPrototypeClassifier`
  自己的 docstring 早就講過這是「例外」）。`_VALID_CLASSIFIERS` 現在只有
  `prototype_cosine` 一個，之後 Stage 3 多一個 arm 就要同時加進兩邊的 registry。
- Stage 4 這條軸維持三個獨立 CLI 檔案，沒有合併——跟 MppRoutingHead 自己
  `run_baseline2`/`run_baseline3` 用兩個函式而非一個函式裡 if-else 是同一個
  原則,只是這裡連 CLI 參數集合都幾乎不重疊,所以拆到檔案這一層。

**已完成（2026-09-20 追加）：5.1 的 mean/median + shared MLP**——使用者確認
「Set Transformer x shared MLP」這個比較值得做，兩個都建好了，還沒跑過分數：
- `mean`/`median`（`aiNNModel/models/TrivialPrototype.py`）：零參數，這條軸的
  下限，跟 baseline arm 在 Stage 4 那條軸上的角色一樣。
- `shared_mlp`（`aiNNModel/models/SharedMlpPrototype.py`）：Deep Sets 風格,
  每個 support member 各自過同一個 MLP（不互相 attend），再 mean pool——這是
  「support member 之間互相 attend 到底有沒有用」這個問題的直接對照組。
  **設計決定**：depth=2、width_mult=1.0（不是 0.5）、outer+inner residual都
  開，三者都是為了跟 SetTransformerPrototype 的容量/inductive bias 儘量對齊，
  避免比較結果混進「參數變少」或「沒有 residual 錨定」這些跟 attention 本身
  無關的干擾項。更窄（width_mult=0.5）是合理的第二個問題,等這個 config 有
  分數之後再開一個新的 registry entry,不是同一個 entry 開關就好——跟
  MppRoutingHead 的 `mlp`/`mlp_narrow` 是兩個獨立命名而非一個 flag 同一個
  道理。
- Matching Net（5.1 剩下的部分）還沒動工，需要 `CrossAttentionMatch.py`。
- Pooling（avg/max/attn）本來就是既有的獨立軸（`--pooling`），跟這次的
  generator 軸互相獨立，不需要額外動工。

**已完成（2026-09-20 追加）：`cli/train.py` 的 log 補齊到跟
`cli/train_baseline.py`/`MppRoutingHead` 一樣的樣式，兩邊都加了 wandb**——
使用者發現兩個正在跑的 job 印出來的東西完全不一樣，追問 episodic 訓練內部
到底能怎麼呈現：
- `cli/train.py` 現在每個 epoch 印一個 composition 風格的區塊：這個 epoch
  每個 rung 實際拿到多少 query 樣本（episodic N-way 抽樣不保證每個 rung
  每次都出現）,以及每個 rung 在**這個 epoch 自己的 episode**上的準確率。
  **這不是驗證分數**——是拿剛剛更新過 optimizer 的同一批 episode 算出來的,
  回答的問題是「loss 有沒有在教模型分辨 rung」（episode 內部的學習訊號），
  不是「有沒有 generalize」。真正的 held-out 驗證還是要等 held-out-WSI +
  held-out-rung-combo 這個設計（module docstring 裡的「NO VAL LOOP YET」
  段落還在,沒有解決,只是現在不再是唯一有訊號可看的東西了)。
- `cli/train.py`/`cli/train_baseline.py` 都加了 wandb（`--wandb-project`/
  `--wandb-mode`/`--run-name`,跟 MppRoutingHead 同一套 flag,jobscript 也都
  補上 `WANDB_PROJECT`/`RUN_NAME` env var)。`cli/train.py` 的
  `wandb_init`/`wandb_log`/`wandb_finish` 放進這個專案自己的
  `Runtime.py`（全是通用函式，跟 `MppRoutingHead/Runtime.py` 裡的完全等價,
  照抄一份是因為 import 會撞名);`cli/train_baseline.py` 沒有 import 任何一個
  `Runtime.py`（它自己的理由本來就寫在 module docstring 裡了),所以是再抄
  一份局部的 `_wandb_init`/`_wandb_log`/`_wandb_finish`。

**已完成（2026-09-21 追加）：`cli/train.py` 的 held-out val 迴圈——上面那句
「NO VAL LOOP YET 沒有解決」現在解決了。** 兩條軸都是沿用現成的東西,不是
重新設計：
- **Held-out WSI**：`--eval-datasets`（`bracs/test`/`ki67_with_photo`，跟
  `MppRoutingHead`/`train_baseline.py` 同預設值）x `wsi_split` 指到
  `MppRoutingHead` 自己的 cache——跟 `train_baseline.py` 的 `val_rows`
  完全同一套機制，三個腳本讀到的 held-out WSI 是**同一批**。訓練用的
  `ki67_pure` 完全不切，維持使用者說的「train 還是整個 ki67_pure」。
- **Held-out rung 組合**：`Episodes.HELD_OUT_COMBOS` 加了第三組——全 6 個
  rung `(1.0, 2.0, 4.0, 8.0, 16.0, 32.0)`，這是 spec.md 自己講的實際
  DEPLOYMENT task，訓練從來沒有練過（`n_choices` 也不包含 6）,驗證是這個
  arm 第一次真正被拿去跟它實際要做的工作對照。新的 `Episodes.
  sample_val_episode` 從 `HELD_OUT_COMBOS`（現在 3 組)裡面**直接抽**,不是
  排除它們——跟 `sample_episode` 共用同一個 `_draw_episode` 做 WSI/position
  的抽法，只有「rung 組合怎麼選」這一步不一樣。
- Val episode 每個 epoch 用**新建的** `random.Random(args.seed)` 抽，不是
  訓練迴圈自己一直往前走的那個 `rng`——這樣每個 epoch 的驗證題目完全一樣,
  分數變化才是模型在變,不是題目在變（跟 `MppRoutingHead.predict` 自己
  `epoch_seed=0` 的理由一樣）。
- 算法跟印法跟另外兩個腳本同步（n-weighted/dataset-avg/per-rung、native/
  resampled）——`score`/`rescore`/`rescore_by_rung` 這次加進了這個專案自己
  的 `Runtime.py`（一樣是照抄 `MppRoutingHead/Runtime.py` 的版本，同一個
  撞名理由，不是 import）。
- Checkpoint 現在有 `_best`/`_best_unweighted` 了——因為 prototype checkpoint
  是三個模組（pooling/generator/head），`Checkpoints.weight_filename` 那個
  現成的命名慣例是給單一 `Head` 用的，所以在 `cli/train.py` 自己裡面寫了一個
  對應版本（`_prototype_weight_filename`/`save_tagged`）。
- **還沒解決的**：這次做的是「能不能量」，不是「量出來好不好」——模型在
  held-out 組合/WSI 上實際表現如何，還沒有真正跑過一次才知道。`evaluate.py`
  （test split）也還沒動工。

**已完成（2026-09-21 追加）：`--merge`，兩個訓練腳本都加了。** 使用者指出
`PrototypicalRoutingHead`（跟 `...Baseline`）的每次執行都用同一個 job
name,結果一定寫進同一個目錄——這是刻意的,為的是讓不同 arm（不同
`--generator`/`--routing-head`/`--pooling`/`--classifier`）的結果能疊在
一起比,像 `bench_stage1_mpp.py` 現在比較 `MppRoutingHead`
好幾個 head 一樣。但原本 `val_scores.csv`/`val_scores_per_rung.csv` 每次都是
整份覆寫,換一個 arm 重跑就會把前一個 arm 的歷史洗掉。現在照抄
`MppRoutingHead/cli/train.py` 自己的 `--merge`/`_merge_val_scores` 慣例：
`cli/train.py` 用 `(encoder, pooling, generator, routing_head)` 當 key,
`cli/train_baseline.py` 用 `(arm, encoder, classifier)`——同一個 key 的舊
row 被這次跑的結果取代,其他 arm 的 row 保留。順便把 `pooling`/`classifier`
補進 CSV 的 identity 欄位（之前漏掉,不然同一個 generator/routing_head 換
pooling 重跑會分不出來是哪一次)。Jobscript 都加了 `MERGE` env var,跟
`MppRoutingHead.sh` 同一個 pattern。

**已完成（2026-09-21 追加）：修正 `arm` 這個詞的說法。** 使用者問「MppRoutingHead
有 arm 嗎」——沒有，那個專案 2026-09-17 就刻意把 `arm` 改名成 `head`/
`baseline`（`cli/train.py` 自己的 module docstring 講過理由：「arm」暗示
side-by-side 比較，`ClassifierEstMpp.py` 單一 head 推論的場景沒有這個意思）。
`arm` 是**這個專案自己**的詞（spec.md Stage 4 的三選一：baseline prototype
learning / metric-based / optimization-based meta-learning）。之前
`cli/train_baseline.py` 的 `_IDENTITY_FIELDS` 已經有 `arm='baseline_prototype'`
但 comment 寫成「跟 cli/train.py 的 _IDENTITY_FIELDS 對得起來」——其實
`cli/train.py` 那時候根本沒有 `arm` 欄位，comment 本身是錯的。這次一起修：
`cli/train.py` 的 `_IDENTITY_FIELDS` 也補上 `arm='metric_based'`（常數,這個
腳本只訓練這一個 Stage 4 regime),兩邊 comment 也改成準確描述——這樣兩個
`val_scores.csv` 之後才真的能拿去接在一起做 Stage 4 的比較,不會欄位對不上。

下面每一節底下標了它屬於哪一步。

---

## 0. Open design decision（動工前先拍板）

三個都寫在 spec.md 自己的「Open design decisions」一節：

1. Checkpoint 格式要不要在 `aiNNModel/models/common/Checkpoints.py` 新增函式
   （不改現有的）——目前提案是這樣做，等你確認。
2. `CrossAttentionMatch.py` 放 `aiNNModel/models/`，Stage 2 的 Matching Net
   跟 Stage 3 的可學習 Arm 2 共用同一個類別——這個決定比較篤定，主要是等你
   看過同意。
3. `SetTransformerPrototype.py` 同樣放 `aiNNModel/models/`，理由跟 2 一樣。

## 1. 主線打通（1.1 → 1.4，缺一個都不能算「主線跑得動」）

### 1.1 共用 Pooling 模組 —— 已完成

`Pooling.py`。Attn 選項直接包一層 `Heads.AttentionPoolHead`（沿用,不重寫）；
Avg/Max/passthrough 是新的。**位置：`aiNNModel/models/`,不是
`training/PrototypicalRoutingHead/`**——寫 checkpoint 存讀（1.4）時發現放
在 training 資料夾會讓共用層的 `Checkpoints.py` 要 import 任務層的東西,
方向反了,搬過去才對,跟 `SetTransformerPrototype.py` 同一層。

**還沒做的驗證**：support 跟 query 過同一個 `Pooling` 實例、同一組權重
這件事,目前只是「呼叫端要記得傳同一個物件」，還沒有測試斷言兩者是同一個
物件 id——等 episode 迴圈（step 2）寫出來、有真正的呼叫端之後再補這個測試,
現在寫測試沒有東西可以呼叫。

### 1.2 Set Transformer prototype generator（主線）—— 已完成

`aiNNModel/models/SetTransformerPrototype.py`。輸入 `[K, D]`（一個 rung 的
support tile 池，已過 1.1 的 pooling），輸出 `[D]`（那個 rung 的 prototype）。
一次只吃一個 rung 的 support——呼叫端要自己對 6 個 rung 各呼叫一次,這個
類別本身不強制,是呼叫慣例,跟 Pooling 的「同一實例」是同一種要小心的地方。
collapse 用 mean pooling,不是第二層學出來的 pooling——先用最簡單的版本。

### 1.3 Cosine + learnable tau 路由頭（主線）—— 已完成

`aiNNModel/models/PrototypeRoutingHeads.py`：`CosineTauHead`。`tau` 用
`log_tau` 這個 `nn.Parameter` 存（取 exp 才是真正的 tau),保證梯度不會把
tau 推成負的。初始值 `tau_init=10.0`，未驗證，之後可調。

只寫了這一個 head——非學習路由頭那個 block、bias arm、cross-attention arm
是 4.2 的範圍,故意還沒動,等主線有分數才輪到它們。

### 1.4 Checkpoint 存讀 —— 已完成

`aiNNModel/models/common/Checkpoints.py` 新增 `save_prototype_checkpoint`/
`build_prototype_from_checkpoint`，現有的 `save_checkpoint`/
`build_from_checkpoint` 完全沒動。過程中發現 `Pooling.py` 放錯層——已搬移,
見 spec.md「Open design decisions」4。

## 2. Loss + episodic 訓練迴圈（訓練 Metric-based meta-learning 這個 arm）

### 2.1 `Losses.py` —— 已完成

`compute_loss(logits, target, episode_rungs, loss_kind, ...)`，`L_bal`/
`L_ord_a`/`L_ord_b` 三個都用同一個進入點,`episode_class_weights` 每個
episode 自己重新算 `w_c`（sklearn balanced 公式，scope 是這個 episode 的
query batch，不是固定 6-way 的 table——N-way 設計下沒有一個固定分佈可以套,
細節見 spec.md/`Losses.py` 自己的 docstring)。之後 `cli/train.py` 會用
`--loss {bal,ord_a,ord_b}` 選,跟 `MppRoutingHead` 的
`--class-weight {balanced,none}` 同一個慣例。

### 2.2 `Episodes.py` —— 已完成（position 抽樣 + render 都接好了）

`sample_episode(manifest_by_wsi, rng, p_same_wsi=, n_support=, n_query=, ...)`
——抽 support/query WSI、抽 N-way rung 子集（排除 `HELD_OUT_COMBOS`）、從
`MppRoutingHead.Datasets.build_manifest` 建好的 manifest 裡切位置。
`render_episode(episode, bank, cfg, deterministic=)` 接著把位置 render 成
`RenderedEpisode`（`{rung: [(patch, native), ...]}`），沿用
`MppRoutingHead.Datasets.render_row`/`CameraBank` 不變。`RenderedEpisode`
故意不存 label——LOCAL index（0..N-1）用 `rungs.index(rung)` 現算，不重複
存一個可能跟 `rungs` 兜不起來的數字。

**改名**：這個檔案原本規劃叫 `Datasets.py`，寫的時候發現會撞名——
`MppRoutingHead` 自己就有一個同名的 top-level 模組,兩個目錄都在
`sys.path` 上,Python 用模組名稱快取,先載入的那個會被兩邊共用,包括
MppRoutingHead 自己的程式碼都可能拿錯。改叫 `Episodes.py`,spec.md 已同步
更新。

**`_CameraBank`/`_render_row` 已改成公開**（`CameraBank`/`render_row`,
2026-09-20，使用者選了「聯動改動」這個選項）——`MppRoutingHead/Datasets.py`
自己內部的呼叫、`cli/train.py`/`Runtime.py`/
`test_camera_output_to_level0.py` 裡提到這兩個名字的註解,全部同步改名,
`py_compile` 都過。

### 2.3 support/query WSI 抽樣（2026-09-20 定案，已寫進 2.2）

每個 episode 各自獨立抽 support WSI、query WSI，`p_same_wsi` 的機率抽到
同一張（同張 WSI 內部切 disjoint position，排練部署情境）,`1-p_same_wsi`
抽到不同兩張（強迫單一 episode 內就要跨 WSI 泛化，不能靠同張 slide 的
色調/批次線索走捷徑）。不用 k-fold 那種明確的 fold 輪替——訓練跑的 episode
數夠多，獨立抽樣統計上就等價於全覆蓋，不用管理 fold 邊界。`p_same_wsi`
是新的超參數，起始值待定、未驗證。

### 2.4 N-way rung 子集抽樣（2026-09-20 定案，使用者自己的設計）

每個 episode 先抽 `N∈{3,4,5}`（刻意排除 6——訓練從不看完整 6-way 任務,
只有 `stage1_compare` 的評分才是完整 6-way），再抽是哪 N 個 rung（排除
保留給驗證用的組合,例如 `{1,4,16,32}`、`{1,2,4}`,名單內容待定）。
`SetTransformerPrototype`/`CosineTauHead` 不用改（K 本來就是動態的）。
loss 要改：`log2_rungs` 要用這個 episode 自己選中的 N 個 rung 算，`target`
要對應到 episode 內的 local index（0..N-1）,不是全域 rung index。

`episodes_per_epoch ≈ 126`（訓練用 WSI 數量,粗略對齊傳統 epoch「每個樣本
看過一次」的精神），不追求窮舉每一種 N-way 組合。

### epoch 的定義（2026-09-20 確認保留）

保留 epoch，當作記帳單位：`episodes_per_epoch` 個 episode 算一個 epoch,
epoch 邊界不影響 episode 抽樣邏輯本身,只決定多久印一次進度、跑一次 val、
存一次 checkpoint——跟 `MppRoutingHead` 現有的節奏一致。

### `cli/train.py` —— 主線訓練迴圈已寫完，還沒跑過

把 1.1-1.4 + 2.1-2.4 全部串起來：建 manifest（一次）→ 每個 epoch
`episodes_per_epoch` 個 episode → 每個 episode 抽樣（`sample_episode`）→
render（`render_episode`，失敗就重抽,不算進 episode 計數)→ forward（support
過 Pooling+SetTransformerPrototype 變 prototype,query 過 Pooling 變向量,
`CosineTauHead` 算 logits)→ `compute_loss` → backward → 存
`_last.pt`（`save_prototype_checkpoint`)。

**還沒做**：val 迴圈（見檔案自己的 module docstring——需要同時處理「沒見過
的 WSI」跟「沒見過的 rung 組合」兩個軸，不是隨手幾行）、`_best.pt` 因此也
還沒有。`--pooling passthrough` 明確擋掉（`SetTransformerPrototype` 需要
一個向量一個 tile，不是沒 pool 過的 grid）。

**沒有實際跑過**——只做到 `py_compile` + 逐行核對呼叫的參數/型別對不對,
沒有真的用 GPU 跑一次 forward/backward。建議先用很小的參數跑一次 smoke
test 再放大規模。

## 3. Metric-based 主線分數

上 `stage1_compare` 那個計分板（`utilities/bench_modules/
bench_stage1_mpp.py`），跟 `KnnEstMpp`、`MppRoutingHead` 現有
checkpoint 站在同一批 slide 上比——需要一個新的 `PrototypicalEstMpp`
（`1_estimate_query_mpp/`，`StageInterface.MppEstimator` 形狀),但這是
之後的事,先不寫。

## 4. Baseline prototype learning arm

跟 metric-based 用同一套 Stage 3 距離函式（`CosineTauHead`），但訓練方式換成
一般 supervised（不切 episode，每個訓練 tile 直接對一個 FIXED 學出來的權重
矩陣分類，形狀跟 `MppRoutingHead` 自己的訓練迴圈幾乎一樣）。Prototype
只在推論時才算——對一批 support tile 的 embedding 事後平均，那時 `self.
weight` 整個丟棄不用。這是 Chen et al. 2019 few-shot literature 的
「Baseline」分類，也是 `MppRoutingHead` spec.md 還沒做的 2-2（NCM）的加強版
（embedding 是這個專案自己訓練出來的，不是原始 frozen 特徵）。便宜，可以當
「episodic 訓練到底有沒有比一般訓練好」這個問題的下限對照。

**已完成（2026-09-20）**：`LearnedPrototypeClassifier`（`aiNNModel/models/
PrototypeRoutingHeads.py`，註冊進 `Heads._CLASSIFIER_REGISTRY` 當
`'prototype_cosine'`）+ `cli/train_baseline.py`（完全沿用
`MppRoutingHead.Datasets` 的 `build_manifest`/`iterate_epoch`/
`class_weights`、`common.Head.Head`、**原始未改動的**
`Checkpoints.save_checkpoint`/`build_from_checkpoint`——這個 arm 的訓練形狀
（一次 encoder pass、一次 classify pass、對整個 manifest 跑一般 epoch）正好
就是那兩個函式當初設計的形狀，不需要 episodic arm 另外寫的
`save_prototype_checkpoint`/`build_prototype_from_checkpoint`）。

**這個注意事項已經解決了（2026-09-21，見檔案最下面的追加記錄）**：原本這條路
走的是 `common.Head.Head` 自己的 `pooled_view`，跟 metric-based 主線用的
`Pooling(kind='avg')` 不是同一個東西。現在 `--pooling` 已經改成跟主線共用
同一個 `Pooling` 模組，這個差異不存在了。

還沒跑過（py_compile 過、每個 import 對照過現有程式碼確認簽名一致，沒有真的
執行）——`jobscripts/PrototypicalRoutingHead/PrototypicalRoutingHeadBaseline.
sh` 的 `SMOKE=1` 是第一次真的跑。

**已完成（2026-09-20 追加）**：補上 val 迴圈 + per-rung 印出，log 樣式對齊
`MppRoutingHead/cli/train.py`（composition/class weights/per-epoch val
n-weighted+per-dataset/per-rung breakdown 都印，`_best`/`_best_unweighted`
存檔）。
Val 用的 WSI 跟 `MppRoutingHead` 自己 held out 的**同一批**——`val_rows`
把 `wsi_split` 指到 `result/cache/mpp_routing_head/`（`MppRoutingHead` 自己的
cache 目錄，不是這個專案自己的），靠 `wsi_split`「EXISTING WINS」的規則讀到
同一個檔案，兩邊的比較才站得住腳。順便修了 `MppRoutingHead/cli/train.py`
自己的 `rung_report`：那個函式從一開始就只寫 CSV，從沒真的印到 log 過
（這次一起補上，兩邊算是雙修）。

`predict`/`score`/`rescore`/`rescore_by_rung` 沒有直接 import
`MppRoutingHead/Runtime.py` 的同名函式，是照抄一份簡化版（單一 head，不是
dict of heads）——原因跟 `Datasets.py` 沒有沿用同名是同一個：spec.md 自己
規劃了這個專案未來要有一個 `Runtime.py`（Stage-2 x Stage-3 arm 組合的
registry），跟 `MppRoutingHead/Runtime.py` 撞名，現在 import 沒事，那個檔案
一造出來就會靜默出錯。

**已完成（2026-09-21 追加）：`val_report` 的 per-dataset acc 公式跟
`MppRoutingHead/cli/train.py` 同步**——使用者把 `MppRoutingHead` 那邊的
per-dataset `level_accuracy` 改成六個 rung 直接平均（不是 pooled-per-tile），
`(all, pooled)` 這個印出來的標籤也順便改成 `(all, n-weighted)`（更準確描述
現在的算法：先各 dataset 算六個 rung 的平均，再用 dataset 的 n 加權合併，
不是所有 tile 攤平算一個數字）。這邊也照樣改了同一套公式跟標籤——不改的話,
兩邊 `val_scores.csv` 裡同一個欄位名稱 `level_accuracy` 實際上是兩種不同的
算法,拿來比較會在不知不覺間比錯東西,而這個 arm 存在的整個目的就是要跟
`MppRoutingHead` 站在同一個量尺上比。`/work/u26130998/log/PrototypicalRoutingHead`
（episodic 主線）不需要改——那個腳本現在沒有 held-out val,沒有這一整套
pooled/n-weighted/dataset-avg 的東西存在。

**已完成（2026-09-21 追加）**：`best_pooled`（追蹤要不要存 `_best.pt` 的那個
變數,`save_tagged`/`run_baseline2`/`run_baseline3`/`_maybe_resume` 裡到處
都是）改名成 `best_n_weighted`,`'new best, pooled (...)'` 印出來的字也改成
`'new best, n-weighted (...)'`。純粹是命名一致性,不影響任何 CSV 欄位或
checkpoint 存的 key（`best_pooled`/`best_n_weighted` 從來都只是 in-memory
的暫存變數,從來沒有以這個名字被寫進 CSV 或 ckpt 裡過,舊檔案完全不受影響）。
`aiNNModel/models/common/Checkpoints.py` 裡 `weight_filename` 自己的
docstring（原本講「POOLED val accuracy (torch.cat before scoring)」）也
一起改了,現在講的是 n-weighted 的算法,並且說明這個 generic 檔案本身不算
公式,公式在 `MppRoutingHead`/`PrototypicalRoutingHead` 各自的 `val_report`
裡,兩邊刻意保持同步。

## 5. 觀察臂，一次一個

Matching Net（5.1）跟 Stage 3 可學習 Arm 2（5.2）共用
`CrossAttentionMatch.py`,只做一次,兩處呼叫。

## 6. Optimization-based meta-learning arm

MAML 式：episode 裡先對 support set 的 loss 做 K 步真正的梯度下降（複製一份
參數），再用「適應過」的那份參數算 query loss，梯度穿回原始參數更新。

**現在的傾向（2026-09-20，還不是最終決定）：先做 Reptile。** 三者 inner
loop 一樣，差別只在 outer 更新怎麼算：full MAML 要對 inner loop 的梯度再
微分一次（Hessian-vector product，最貴、最不穩定）；FOMAML 假裝適應後的
參數跟原始參數無關，直接把 outer loss 的梯度當更新方向（丟掉二階項，原始
論文自己的 ablation 顯示影響不大）；Reptile 完全不算 outer 梯度，就是
inner loop 跑完之後把參數往 `theta' - theta` 的方向挪一點，不需要任何
穿過 inner loop 的微分機制。Reptile 實作風險最低（這個 codebase 已經有
fp16-under-Adam NaN 的前科），先用它拿到「optimization-based 到底有沒有用」
的答案，有需要再往 FOMAML、full MAML 加精度。

比 metric-based 貴（每個 episode K+1 次 forward/backward），排在 3、4 都
有分數之後才動工。

## 7. Stage 1 支線

主線分數穩定、觀察臂都比過一輪之後才排——現在不動。

---

**已完成（2026-09-21 追加）：Pooling/RoutingHead 兩邊共用，`arm` 改名成
`training_framework`。** 使用者這輪的結論：Pooling/Generator/RoutingHead
是模型本身的架構，理論上該獨立於 Stage 4（哪個 regime 訓練它）——generator
在 baseline arm 缺席不是漏做，是 Chen et al. 對「Baseline」的定義本身
（訓練時沒有 support set，沒東西餵給 generator）。

- `cli/train_baseline.py` 加了 `--pooling`/`--routing-head`，讀跟
  `cli/train.py` 完全同一套（`Pooling`/`Runtime.ROUTING_HEAD_CHOICES`）。
  沒有 `--generator`，以後也不會有。
- 因此離開了 `common.Head.Head`/`Heads._CLASSIFIER_REGISTRY`——`Head` 的
  classifier 合約是 `classifier(cfg)`，一個位置參數，塞不進「選哪個
  routing head」這件事。`LearnedPrototypeClassifier` 改成建構子直接注入
  `routing_head` 模組（不再寫死 `CosineTauHead`），`Heads._CLASSIFIER_
  REGISTRY` 的 `'prototype_cosine'` entry 跟著拿掉（沒人再用了）。
- `Checkpoints.save_prototype_checkpoint` 的 `generator`/`generator_cfg`
  改成可選（預設 `None`）——比另外寫第四種存檔格式簡單，`train_baseline.py`
  兩個都傳 `None`。`build_prototype_from_checkpoint`（讀的那一半）還沒跟著
  改，一樣是「還沒 registry-aware」那個已知缺口的一部分。
- `train_baseline.py` 原本自己抄的 `_score`/`_rescore`/`_rescore_by_rung`/
  `_wandb_init`/`_wandb_log`/`_wandb_finish` 六個本地複本拿掉了，改成從
  這個專案自己的 `Runtime.py` import——當初抄一份是因為 `Runtime.py` 還
  不存在，這個理由早就不成立了，只是沒人回頭處理。
- `arm` 改名成 `training_framework`：`arm` 是這個專案自己的詞沒錯，但
  spec.md 裡到處都在用（Stage 2/3 的觀察臂也叫 arm），拿來當「哪個 Stage 4
  regime」這一個特定欄位的名字會混淆。取名 `training_framework` 不是
  `training_arch`——「architecture」現在就是指 Pooling/Generator/
  RoutingHead 本身，Stage 4 這條軸叫 arch 會跟這個區分互相矛盾。
- 過程中順便抓到一個真的 bug：`val_report` 少了 `out = []` 初始化，會
  `NameError`——這次重寫一起修掉了。

還沒真的跑過。

**已完成（2026-09-21 追加）：Stage 2 三個 generator 檔案合併成一個。** 使用者
發現 `PrototypeRoutingHeads.py` 早就把 Stage 3 所有 arm 都放同一個檔案，
Stage 2 卻是三個獨立檔案（`SetTransformerPrototype.py`/`TrivialPrototype.py`/
`SharedMlpPrototype.py`）——這個不一致不是刻意的，只是三個 arm 陸續加的時候
順手各開一個檔案。合併成 `aiNNModel/models/PrototypeGenerators.py`，class
名字都沒變，只是搬了檔案。`Runtime.py`、`Checkpoints.
build_prototype_from_checkpoint` 的 import 跟著改。以後 Matching Net
（plan.md 5.1）也會加進這個檔案。

**已完成（2026-09-21 追加）：held-out val 補上 per-combo（3/4/6-way）拆解。**
使用者發現 `rung_report` 現在的 per-rung 數字把「這個 rung 是在 3-way
episode 裡被考」跟「在 6-way episode 裡被考」混在一起算——但 6-way 才是
spec.md 講的真正 deployment task，3/4-way 只是泛化探針，混著算會看不出模型
在真正的任務規模上表現如何。新增 `combo_report`（`val_scores_per_combo.csv`），
每個 (dataset, combo) 一行，combo 內部直接 pooled（不用像 dataset 那樣先
rung 平均——因為一個 combo 裡每個 rung 的 query 數量本來就一樣多，沒有
`RICHNESS` 那種不均勻的問題）。`val_episode_detail` 現在會記錄每個 query
屬於哪個 combo（`_combo_label`，例如 `"1+4+16+32"`）。

**已完成（2026-09-21 追加）：拿掉「統合 acc」，checkpoint 選擇標準改成
6-way deployment accuracy vs 全 combo 診斷平均。** 使用者確認：3-way/4-way/
6-way 難度不同，equal-weight 混在一起的分數不該拿來選 checkpoint。這次把
`val_report`（dataset 層級的 n-weighted/dataset-avg 總結）整個拿掉，
`rung_report`（pooled-across-combo 的版本）也一起拿掉——`combo_report` 現在
是唯一的 val report，每個 combo 印自己的 acc（含 native/resampled）+ 自己的
per-rung 分解，彼此不混。

Checkpoint 選擇：
- `_best.pt` 現在看 `_deployment_accuracy`——全 6-way combo 自己的 accuracy，
  跨兩個 dataset n-weighted 合併。這是真正的 deployment task，訓練從沒練過。
- `_best_unweighted.pt` 改名成 `_best_diagnostic.pt`，看 `_diagnostic_
  accuracy`——所有 (dataset, combo) 的 accuracy 直接平均，equal weight，當
  「泛化能力有沒有全面撐住」的診斷用第二個 checkpoint，刻意跟 deployment 那個
  分開存,不要混成一個數字。

`val_scores.csv`（dataset 層級的整份 CSV）也跟著拿掉了——`val_scores_
per_combo.csv`/`val_scores_per_rung.csv` 現在是僅有的兩份 val 輸出。
