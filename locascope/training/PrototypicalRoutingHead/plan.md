# PrototypicalRoutingHead — 計畫

`spec.md` 是規格：架構、每個 stage 的選項、模組沿用/新增的決定與理由。這份是**做到哪、還沒做什麼**。

> **2026-11-02 瘦身。** 原本是 2026-09 的施工順序與逐日追加記錄，主線早就做完、訓練過、接進 stage 1，
> 那些記錄裡大半在寫已經改名或刪掉的檔案。原文：`git show HEAD:training/PrototypicalRoutingHead/plan.md`。
> 下面只留程式裡查得到的現況、仍然有效的決定、還沒做的項目。每個機制的細節看 `cli/train.py` 開頭的 docstring、
> `Episodes.py`、`aiNNModel/models/PrototypeChoices.py` 開頭的 docstring。

## 現況

| 部分 | 位置 |
|---|---|
| Pooling（support 與 query 共用同一個模組） | `aiNNModel/models/Pooling.py` |
| Stage 2 的 collapse：Set Transformer、attn pool、`mean`（零參數下限）、`shared_mlp`（Deep Sets 對照組） | `aiNNModel/models/PrototypeGenerators.py` |
| Stage 2 的 support／query context（`bilstm`、`attnlstm`） | `aiNNModel/models/ContextEncoders.py` |
| Stage 3 路由頭：`cosine_tau`、`cosine_logsumexp`、`attn_score` | `aiNNModel/models/PrototypeRoutingHeads.py` |
| 以上的 registry（`--support-context`／`--query-context`／`--collapse`／`--routing-head`） | `aiNNModel/models/PrototypeChoices.py`，`Runtime.py` 重新匯出 |
| Matching Net | 不是 registry 的名字，是一個組合：`--support-context bilstm --query-context attnlstm --collapse off --routing-head cosine_logsumexp` |
| checkpoint 存讀 | `aiNNModel/models/common/Checkpoints.py`（`save_prototype_checkpoint`／`build_prototype_from_checkpoint`） |
| 損失 | `Losses.py`：`L_bal`／`L_ord_a`／`L_ord_b` |
| episode | `Episodes.py`：混合池 `pool_by_rung`、`draw`、`training_combos`（39 個 rung 組合）、`reuse_pairs`（none／hold_s／hold_q／val） |
| 訓練與評估 | `cli/train.py`（K×K 的 held-out val）、`cli/evaluate.py`（test split，隨機 6-way 與 K×K 兩套分數） |
| 接進 stage 1 | `stage1_estimation/PrototypeEstMpp.py`；權重在 `result/PrototypicalRoutingHead/weights/`；已有 `result/Stage1MppBench` 的圖 |
| job | `jobscripts/PrototypicalRoutingHead/`：`PrototypicalRoutingHead.sh`、`TestRoutingHeads.sh` |

## 仍然有效的決定

- **三條軸疊在一起**：Stage 2（pooling → G → F → collapse）× Stage 3 路由頭 × Stage 4 訓練方式
  （`training_framework`：metric-based／baseline／optimization-based）。Pooling／generator／路由頭是模型本身，
  獨立於用哪種方式訓練它；baseline 沒有 generator 不是漏做，是 Chen et al. 2019 對 Baseline 的定義（訓練時沒有 support set）。
- **Stage 4 那個欄位叫 `training_framework`**，不叫 `arm`（Stage 2／3 的觀察臂也叫 arm，會混淆）、不叫 `training_arch`
  （「architecture」指的是 Pooling／generator／路由頭本身）。
- **registry 的每個 entry 是 builder function**，不是 `(class, config_class)` tuple：每個 entry 要哪些 config 欄位，
  在有第二個 entry 可以對照之前不猜。
- **`shared_mlp` 刻意跟 Set Transformer 同深度、同寬度、同樣的 residual**（depth=2、width_mult=1.0）：這個比較只該隔離
  「support 成員之間互相 attend 有沒有用」，不要混進「參數變少」。更窄的版本是之後另開一個 entry，不是同一個 entry 的旗標。
- **epoch 只是記帳單位**，不影響 episode 怎麼抽；多久印一次進度、跑一次 val、存一次 checkpoint 依它。
- **選 checkpoint 不用等權混合 3／4／6-way 的分數**：訓練只用 3、4、5-way（排除 `HELD_OUT_COMBOS`），完整 6-way 才是部署任務，
  訓練從不練；難度不同的 combo 混成一個數字無法拿來選。各項 checkpoint 與 CSV 的定義看 `spec.md` 與 `cli/train.py`。

## 還沒做

1. **Optimization-based meta-learning**（Stage 4 三選一之三，MAML 式 inner loop）。傾向先做 Reptile（還不是最終決定）：
   三者 inner loop 相同，差在 outer 更新——full MAML 要對 inner loop 再微分一次（最貴、最不穩）；FOMAML 丟掉二階項；
   Reptile 完全不算 outer 梯度，把參數往 `theta' - theta` 挪一點，實作風險最低（這個 codebase 有 fp16 under Adam 出 NaN 的前科）。
   比 metric-based 貴（每個 episode K+1 次 forward／backward）。
2. **Baseline prototype learning**（Stage 4 三選一之二）：2026-09-22 入口已刪，推論時一個 baseline checkpoint 面對全新 WSI
   該怎麼用一直沒定案，要從頭重新設計，不是接著舊版做。使用者決定。
3. **Stage 3 的其他觀察臂**：非學習路由頭（負歐氏距離、固定溫度 cosine）、bias routing。沒做，也沒有決定要做。
4. **Stage 1 的兩個支線**（混合 foundation models、fine-tuned CNN）：主線分數穩定、觀察臂都比過一輪之後才排，現在不動。
