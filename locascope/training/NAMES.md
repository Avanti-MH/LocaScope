# NAMES：代號字典

本檔是代號的唯一存放處：其他文件第一次用一個代號時給一句定義，之後只寫代號，完整定義只在這裡。名字的產生與讀回由程式的表負責：CoAE 在 `locascope/training/CoAE/Config.py`（`LOSS_TERMS`、`NAME_PARTS`、`DIR_PARTS`），arm 文法在 `locascope/core/stage2_retrieval/ArmSpec.py`；文件與程式不一致時以程式為準。

## 1. 原則
1. 一個前綴只代表一類東西：看到前綴就知道它是問題、回答、架構、訓練版本、量測、判準、資料、照片組、還是條件。
2. 屬於某個方法的代號帶方法名（`CoAE`、`PQ` = PhaseQuery）；兩個方法共用的代號不帶。
3. 數字只用在兩處：研究問題的編號（`Q<n>`，與同號的 `A<n>`）與架構的版本（`Arch<n>`）。其餘一律用說得出內容的詞。
4. 一個訓練版本（arm）有兩個名字，都由它的組成拼出，都只由 `CoaeConfig` 的表產生、互相可讀回：arm 代號（`CoaeConfig.arm`，短，用在 arm 文法與 stage 2 cache 的 label）與結果目錄的長名（`CoaeConfig.dir_name`，用在 `result/<job>/`、`log/<job>_` 與 checkpoint 路徑）。
5. 已停用的代號：A0、A1、A2、B、E1、E2、T0、T1、C2、V2、CV2、SS、SC、CS、CC、VC、P、S、M1、M2、H<n>、「步驟 0／1／2」、「第 1 步／第 2 步」、「(a)(b)(c)」式的子項代號、沒有方法名的 G<n>。實驗名（如 `ExpA1Full`）不在此限。
6. 一個代號在文件裡第一次出現時用一句話定義（組成加出處）；之後只寫代號。

## 2. 代號字典

### 2.1 前綴
| 類別 | 寫法 | 組成規則 | 例 | 屬於 |
|---|---|---|---|---|
| 研究問題 | `<方法>.Q<n>` | n 固定、不重用 | `CoAE.Q4` | 各方法 |
| 回答 | `<方法>.A<n>` | 與問題同號；沒有結論時寫「未答」 | `CoAE.A2` | 各方法 |
| 架構 | `<方法>.Arch<n>` | 網路形狀的版本；一個方法現役的只有一個 | `CoAE.Arch1`、`PQ.Arch1` | 各方法 |
| 訓練版本（arm） | `<方法>:<arm 代號>` | 語法見 §3 | `CoAE:RN3` | 各方法 |
| 量測 | `M-<種類>(<對象>[; <條件>])` | 種類見 §4；對象是 arm（可加 `:<用法>`）或 FM 名 | `M-tile(CoAE:RN3)` | 共用 |
| 判準 | `Gate.<方法或 shared>.<檢查什麼>` | 詞說明它檢查的事（§5） | `Gate.CoAE.trained` | 見 §5 |
| 資料集 | `Data.<用途>` | `train`、`probe`、`real` | `Data.probe` | 共用 |
| 模擬來源組 | `Sources.<名>` | `all3`、`holdout-chroma`、`refonly`；arm 代號裡的 3、2、0 對應這三組 | `Sources.holdout-chroma` | CoAE |
| 探針的來源與 view | 原名 | 來源 `ood-heavy`、`ood-soft`；view `anchor`、`twin:<來源>`、`sib:<程度>`、`both:<來源>` | `both:ood-soft` | CoAE |
| 合成照片的條件 | `Cond.<名>` | `ident`（不做幾何與外觀）、`geom`（只做幾何）、`full`（幾何加外觀） | `Cond.geom` | PQ（已放棄） |
| 真實照片組 | `Photos.<名>` | 見 §6 | `Photos.unsampled` | 共用 |
| 管線的階段 | `stage1`、`stage2`、`stage3` | 既有；只指管線的三段，不指研究的先後 | `stage2` | 共用 |
| 搜尋時讀的特徵 | 原名 | CoAE：`ArmSpec.COAE_PARTS` 的鍵；PQ：查詢塔的輸出 `query` | `fm_head` | 各方法 |

研究的先後不另設「步驟」的代號：以「先回答哪個 `Q`、過哪個 `Gate`」表達。

### 2.2 研究問題
| 問題 | 內容 | 用到的 arm |
|---|---|---|
| `CoAE.Q1` | 以共通圖為目標的訓練，在模擬上有效嗎 | `RP3` |
| `CoAE.Q2` | 不變性來自目標，還是來自模擬來源 | `RP0` |
| `CoAE.Q3` | 哪個設計是必要的 | `RP3.dep`、`RP3.ind`、`RP3.vdist`、`RP3.fmresidual`、`Rx3` |
| `CoAE.Q4` | 降低消耗的同時，能不能把展開補回來 | `RN3` |
| `CoAE.Q5` | 跨來源的對比是否只學到看過的來源；約束是否讓泛化較好 | `RP2`、`xV2.weightV1`、`RPV2` |
| `CoAE.Q6` | 描述子與 cofig 單獨或合起來貢獻多少；比單純的對比好嗎；對比該加在哪 | `RP2`、`RH2`、`DP2`、`DH2`、`LH2`、`RV2` |
| `CoAE.Q7` | 每格的位置數、slide 數夠不夠 | `RP3.pool16.epochs24`、`RP3.slides50.epochs6` |
| `CoAE.Q8` | 最後的模型在真實照片上到哪裡 | 正式 arm |
| `CoAE.Q9` | 特徵對位移的反應是什麼形狀（64 px 附近有沒有斷崖） | `M-offset` |
| `CoAE.Q10` | 描述子與 cofig 是否解決源頭問題；cofig 怎樣才有貢獻 | `RP3`、`RN3`、四格 |
| `CoAE.Q11` | 怎樣的探針能預測召回（因子卡） | `M-card`、`M-dense` |
| `CoAE.Q12` | winembed 改讀 CoAE 的 latent，能否勝過 slidewin gigapath | `GigaCoAE+RN3:WinEmbed::latent` |
| `PQ.Q1` | gigapath 自己在合成照片各條件下的水準，與顏色佔多少 | `M-synth(gigapath; …)`、`M-grey` |
| `PQ.Q2` | 只憑結構能決定多少 FM 的 cls | `PQ:ident…`；`Gate.PQ.ceiling` |
| `PQ.Q3` | 加上幾何與外觀後的小規模訓練到哪裡 | `PQ:full.slides20`；`Gate.PQ.window-pilot` |
| `PQ.Q4` | 正式訓練 | `PQ:full`；`Gate.PQ.window-full`、`Gate.PQ.no-leak` |
| `PQ.Q5` | 真實照片 | `M-tile`、`M-realtest`；`Gate.shared.*` |

PhaseQuery 已放棄（網路 `PQ.Arch1`）。`PQ.Q1`–`PQ.Q5` 的切法、`Cond.*`、`PQ:*`、`Gate.PQ.*` 只留作紀錄，不再有新的 arm 或量測。

## 3. 各方法的 arm 語法

### 3.1 CoAE：arm 代號
一個 arm 代號由三位核心與零到多個修飾組成，是 `CoaeConfig.arm` 的輸出；`config_of_arm` 讀回，讀回的 config 再算出的代號與輸入不同就拒絕（一個 config 只有一個代號）。

```
arm      ::= core { "." modifier }
core     ::= local global sources
local    ::= "R" | "D" | "L" | "x"                                   # 局部支（latent → decoder）的損失
global   ::= ( "P" | "N" ) [ "H" ] [ "V" ] | "H" [ "V" ] | "V" | "x"   # 全域支（cls′ → 線性頭）的損失，依此順序
sources  ::= digit                                                   # 模擬來源的個數；3、2、0 是固定的三組
```

| core 的字母 | 意義 | 權重鍵 | 標準權重 |
|---|---|---|---|
| `R` | cofig 重建 | `w_recon` | 1 |
| `D` | decoder 輸出區塊的 twin 對比 | `w_dnce` | 0.5 |
| `L` | latent 的 twin 對比 | `w_lnce` | 0.5 |
| `P` | 描述子回歸 | `w_pool` | 1 |
| `N` | 描述子回歸加目標對比（`nce` 附在 `pool` 上） | `w_pool` 加 `w_nce` | 1、0.5 |
| `H` | 線性頭輸出的 view 對比 | `w_hnce` | 0.5 |
| `V` | cls′ 的 view 對比 | `w_vnce` | 0.5 |
| `x` | 這一支沒有損失 | — | — |

局部支只能有一個損失；全域支的字母依 `LOSS_TERMS` 的順序。`sources` 的 3、2、0 是固定的三組：`Sources.all3`（`bench`、`bench-real`、`ad-chroma4`）、`Sources.holdout-chroma`（`bench`、`bench-real`）、`Sources.refonly`（沒有）；其他個數或其他組合要加修飾 `sims-<a+b>`。

修飾依下表的順序出現，與預設相同就不寫（`NAME_PARTS`）：
| 順序 | 修飾 | config 欄位 | 預設（不寫） | 合法值 |
|---|---|---|---|---|
| 1 | `dep`、`ind` | `data.mechanism` | `mix` | `dep`、`ind` |
| 2 | `vdist`、`vavg` | `head.variant` | `vref` | `vdist`、`vavg` |
| 3 | `fmresidual` | `model.cls_residual` | 關 | — |
| 4 | `noperturb`、`lightperturb` | `data.level` | `large` | `0`、`light` |
| 5 | `desc<c>` | `head.pool_cells`（描述子的格數） | 4 | 整除 `cofig.out_cells` 的整數 |
| 6 | `noise<σ>` | `model.noise` | 0.5 | 非負數 |
| 7 | `lora` | `fm.lora` | 關 | — |
| 8 | `sims-<a+b>` | `data.sims` | 個數 3、2、0 的固定組 | 模擬器名以 `+` 連接 |
| 9 | `weight<字母><值>` | 某個損失的權重 | 上表的標準權重 | 每個非標準的損失一項 |
| 10 | `pool<n>` | `data.pool_cap`（每格留幾個位置） | 128 | 非負整數 |
| 11 | `slides<n>` | `data.slides`（只用前 n 張） | 全部 | 正整數 |
| 12 | `epochs<n>` | `train.epochs` | 3 | 非負整數；0 = 只量探針 |
| 13 | `seed<n>` | `train.seed` | 0 | 非負整數 |

不進名字、也不能偏離預設的設定（偏離就拒絕命名，`UNNAMED`）：`head.tau`、`head.twin_cells`、`head.w_koleo`、`head.w_adv`。不進名字、可以不同的設定：FM 權重的精度、compile、每步的位置數、worker 數。

### 3.2 CoAE：結果目錄與 log 的長名
結果目錄與 log 用長名，由 `CoaeConfig.dir_name` 產生，`config_of_dir` 讀回；一個 arm 代號加一個 FM 對應一個長名（`config_of_arm(<代號>, <FM>).dir_name`；長名含 FM，代號不含）。結果在 `result/<job>/<長名>/`，log 在 `log/<job>_<長名>`，checkpoint 在 `result/CoaeTrain/<長名>/weights/epoch<k>.pt`。

長名的形式是 `<level>-<mechanism>-<variant>` 接 `DIR_PARTS` 中偏離預設的部分，各以 `-` 連接：
| 順序 | 部分 | 預設（不寫） | 備註 |
|---|---|---|---|
| 1 | `sims-<a+b>`、`refonly` | 三個來源 | 兩個來源寫 `sims-bench+bench-real` |
| 2 | `noise<σ>` | 0.5 | |
| 3 | `pool<c>` | 4 | 描述子的格數，等於 arm 代號的 `desc<c>` |
| 4 | `gigapath`、`uni2` | `gigapath` | |
| 5 | `lora`、`fmcls` | 關 | `fmcls` = arm 代號的 `fmresidual` |
| 6 | `wrecon<w>`、`wpool<w>` | 1 | 權重為 0 也寫（`wpool0`、`wrecon0`） |
| 7 | `nce` | 關 | |
| 8 | `vnce`、`vnceonly` | 關 | `vnceonly` = 只有 cls′ 的 view 對比，權重 1，沒有 `recon` 與 `pool` |
| 9 | `dnce`、`lat`、`hnce` | 關 | `lat` = latent 的 twin 對比 |
| 10 | `seed<n>` | 0 | |
| 11 | `cap<n>` | 128 | 池子，等於 arm 代號的 `pool<n>` |
| 11 | `slides<n>`、`epochs<n>` | 全部、3 | 非預設時才寫，寫在最後 |

長名與代號有兩處同字不同義：長名的 `pool<c>` 是描述子的格數（代號的 `desc<c>`），代號的 `pool<n>` 是池子（長名的 `cap<n>`）。對比類損失只能以標準權重寫進長名（`vnceonly` 例外）；非標準的權重沒有長名，`dir_name` 拒絕。

### 3.3 CoAE：arm 文法與 stage 2 cache 的 label
CoAE 的 checkpoint 在 stage 2 的 arm 裡是滑動視窗的 encoder，pooling 位置寫搜尋時讀的特徵：

```
arm:<Fm>CoAE+<arm 代號>:SlidingWindow::<特徵>+<mean|geomean|min>
```

- `<arm 代號>` 必填，字元限於 `[A-Za-z0-9._+-]`，必須是某個 config 的代號（`config_of_arm(<代號>, <FM>)`）。編碼器名稱是 `<Fm>CoAE+<arm 代號>`，`<Fm>` 是 `Giga`（gigapath）或 `Uni2`（uni2）（`ArmSpec.COAE_FMS`）；arm 代號不含 FM（`RN3` 對每個 FM 是同一個代號），FM 只由前綴說一次；目錄名含 FM（`Uni2CoAE+RN3` 的目錄是 `large-mix-vref-uni2-nce`）。
- checkpoint 是 run 的最後一個 epoch：`CoaeTrain/<長名>/weights/epoch<k>.pt`，k = epochs − 1（`CoaeConfig.last_checkpoint`、`ArmSpec.coae_weights`）；反向的 `coae_encoder_name` 由長名讀回代號，路徑不是最後一個 epoch 就報錯。
- CoAE encoder 只能是 arm 的第一段，可由 `SlidingWindow` 或 `WinEmbed` 讀；其後的段要自己指定 encoder（例 `::gigapath:Dual::cls_avg+dual`）。
- `WinEmbed` 讀 CoAE：`arm:<Fm>CoAE+<arm 代號>:WinEmbed::<特徵>`（不加 `+op`；CoAE 凍結，只當逐 tile 的輸入）。權重 `RerankHead/weights/<data>--<Fm>CoAE+<arm 代號>--<特徵>--winembed--best.pt`。例：`GigaCoAE+RN3:WinEmbed::latent`。
- `<特徵>` 是 `ArmSpec.COAE_PARTS` 的鍵：`fm`、`cls`、`place`、`head`、`latent`、`latent_avg`、`latent_dev`、`latent2`、`latent4`、`cofig`、`fm_cls`、`cls_latent`、`cls_latent_avg`、`fm_cls_latent`、`fm_cls_latent_avg`、`fm_head`、`latent4_head`、`fm_latent4_head`；特徵後可加 `-M`（只搜主格點）。
- stage 2 cache 的 label：`slidewin-<Fm>CoAE-<arm 代號>-<特徵>-<mean|geomean|min>`，特徵裡的 `_` 寫成 `-`。例：`GigaCoAE+RN3:SlidingWindow::fm_head+mean` 的 label 是 `slidewin-GigaCoAE-RN3-fm-head-mean`。

### 3.4 CoAE：現有與已排的 arm
定義句式：「`CoAE:<代號>` = <組成>」。長名是 `config_of_arm(<代號>, <FM>).dir_name` 的結果（表列的是 gigapath；其他 FM 的長名加 `-<FM>`）。
| arm 代號 | 組成 | 長名 | 回答 |
|---|---|---|---|
| `RP3` | 重建加描述子，三個模擬來源；其餘預設 | `large-mix-vref` | `CoAE.Q1` |
| `RP0` | 同上，只有 ref | `large-mix-vref-refonly` | `CoAE.Q2` |
| `RP3.dep` | `RP3` 的機制改 `dep` | `large-dep-vref` | `CoAE.Q3` |
| `RP3.ind` | `RP3` 的機制改 `ind` | `large-ind-vref` | `CoAE.Q3` |
| `RP3.vdist` | `RP3` 的變體改 `vdist` | `large-mix-vdist` | `CoAE.Q3` |
| `RP3.fmresidual` | `RP3` 的 cls′ 帶 FM 的 CLS 當殘差 | `large-mix-vref-fmcls` | `CoAE.Q3` |
| `Rx3` | `RP3` 沒有描述子損失 | `large-mix-vref-wpool0` | `CoAE.Q3` |
| `RN3` | 重建加描述子與目標對比 | `large-mix-vref-nce` | `CoAE.Q4` |
| `RP2` | `RP3` 留出 `ad-chroma4` | `large-mix-vref-sims-bench+bench-real` | `CoAE.Q5`、`CoAE.Q6` |
| `RP2.seed1` | `RP2` 的第二個種子 | `…-sims-bench+bench-real-seed1` | `CoAE.Q5`、`CoAE.Q6` |
| `xV2.weightV1` | 只有 cls′ 的 view 對比，權重 1 | `…-sims-bench+bench-real-vnceonly` | `CoAE.Q5` |
| `RPV2` | 重建加描述子加 cls′ 的 view 對比 | `…-sims-bench+bench-real-vnce` | `CoAE.Q5` |
| `RPV2.seed1` | `RPV2` 的第二個種子 | `…-sims-bench+bench-real-vnce-seed1` | `CoAE.Q5` |
| `RH2` | 重建加 head 對比（全域支換成對比） | `…-sims-bench+bench-real-wpool0-hnce` | `CoAE.Q6` |
| `DP2` | decoder 對比加描述子（局部支換成對比） | `…-sims-bench+bench-real-wrecon0-dnce` | `CoAE.Q6` |
| `DH2` | 兩支都是對比 | `…-sims-bench+bench-real-wrecon0-wpool0-dnce-hnce` | `CoAE.Q6` |
| `LH2` | 同 `DH2`，局部支的對比加在 latent | `…-sims-bench+bench-real-wrecon0-wpool0-lat-hnce` | `CoAE.Q6` |
| `RV2` | 重建加 cls′ 對比（對照 `RH2`） | `…-sims-bench+bench-real-wpool0-vnce` | `CoAE.Q6` |
| `RP3.pool16.epochs24` | 每格位置數 1/8、總步數相同 | `large-mix-vref-cap16-epochs24` | `CoAE.Q7` |
| `RP3.slides50.epochs6` | slide 減半、總步數相同 | `large-mix-vref-slides50-epochs6` | `CoAE.Q7` |
| `RP3.pool512.epochs12` | 正式的版本（核心依 `CoAE.Q6` 選出的為準） | `large-mix-vref-cap512-epochs12` | `CoAE.Q8` |
| `RP3.epochs0`、`RP3.noperturb.epochs0`、`RP3.lightperturb.epochs0`、`RP3.desc8.epochs0` | 訓練前審核：只建探針、量一次 | `large-mix-vref-epochs0`、`0-mix-vref-epochs0`、`light-mix-vref-epochs0`、`large-mix-vref-pool8-epochs0` | `CoAE.Q1` 的前置 |

`RP2` 以下各列的 `…` 代表 `large-mix-vref`。實驗用的對比 arm（含 `V`、`H`、`D`、`L` 的）只用來比較來源泛化，不當預設 arm、正式 arm 或 `M-realtest` 的被測 arm（見 CoAE `spec.md` 的損失節）。

### 3.5 PhaseQuery：arm 語法（已放棄，只留紀錄）
```
arm  ::= cond { "." modifier }
cond ::= "ident" | "geom" | "full"                                    # 訓練用的合成照片條件（Cond.*）
modifier（依序）::= "slides<n>" | "levels<a+b>" | "steps<n>" | "bank<n>" | "seed<n>"     # 與預設相同就不寫
```
| arm | 定義 |
|---|---|
| `PQ:full.slides20` | 完整幾何加外觀、20 張 slide 的小規模訓練 |
| `PQ:ident.slides20` | 不做幾何與外觀的上限訓練 |
| `PQ:full` | 全部 slide 的正式訓練 |

這份修飾清單依 `stage2fm` 文件裡出現的設定列寫成，沒有逐欄對過 PhaseQuery 的 config；PhaseQuery 已放棄，不再核對。程式已於 2026-11-18 刪除（`ArmSpec.METHODS` 不再有 `PhaseQuery`），封存在 `/work/u26130998/scratch/backups_repo/PhaseQuery_removed_2026-11-18.tgz`。

## 4. 量測
| 名字 | 量什麼 | 對象與條件 | job |
|---|---|---|---|
| `M-synth` | 合成照片的 tile 層級：tile R@K、z、展開、消耗 | PQ arm 或 FM；條件 `Cond.*`（已放棄） | `ExpPhaseQueryBaseline` |
| `M-grey` | FM 讀灰階 scan tile 能否找回 RGB tile | FM（已放棄） | `ExpPhaseQueryBaseline` |
| `M-bench` | 合成 FoV 的視窗層級 R@K | PQ arm（已放棄） | `ExpPhaseQueryBench` |
| `M-tile` | 真實照片 tile 對其區域全部 tile：raw 與 H 的 R@K（主讀數 R@100）、z、t、e、消耗、展開、第一名干擾的類別 | arm`:`用法、FM、封閉式訊號 | `ExpTileConfusion` |
| `M-card` | 因子卡（真實照片版，只驗證）：① 區分性、② 干擾尾端 t、③ 跨 domain、④ 位移、⑤ 尺度與旋轉、⑥ 視窗聚合，各以 e 的變化（Δe = Δz + Δt）表示，預測 R@100 | arm`:`用法、FM、相位類 | `ExpPhaseSeparation`（`PS_OFF_SCHEDULE=card`，位置 `realphoto`） |
| `M-dense` | 因子卡（模擬查詢版，設計用的探針），與 `M-card` 同一份程式、同一套因子 | 同上 | `ExpPhaseSeparation`（`PS_OFF_SCHEDULE=card`，位置 `dense`） |
| `M-offset` | 相似度隨候選中心偏離的衰減 | 位置來源、半徑排程、特徵 | `ExpPhaseSeparation` |
| `M-realtest` | 全部真實照片的端到端：找到的張數 | arm`:`用法 | `RealTestGroups.sh` |

「raw」= 層級已知、尺度未知的實際條件；「H」= 尺度對齊的天花板。結論只用 raw。

每個 query：z =（正解分數 − 干擾平均）÷ 干擾 sd；t =（第 100 名干擾 − 干擾平均）÷ 干擾 sd（常態約 3.12）；e = z − t。R@100 = 比例(e > 0)，是等式。

## 5. 判準
| 名字 | 檢查什麼 | 條件的出處 |
|---|---|---|
| `Gate.CoAE.signal-untrained` | 訓練前：不訓練的描述子接上 FM 是否有用（下界式） | CoAE `spec.md` 的量測與判準節 |
| `Gate.CoAE.map-untrained` | 訓練前：共通圖本身可分、對來源穩（停止條件） | 同上 |
| `Gate.CoAE.map-with-fm` | 訓練前：共通圖接上 FM（只當訊號） | 同上 |
| `Gate.CoAE.trained` | 訓練有效：沒看過的來源上勝過 FM、重建達標 | 同上 |
| `Gate.CoAE.fm-kept` | 接上 FM 不拖累 FM | 同上 |
| `Gate.CoAE.from-target` | 不變性來自目標（只用 ref 訓練仍不變） | 同上 |
| `Gate.CoAE.source-gap` | 來源泛化：一致、不一致、未定 | `coae_gap.py`、CoAE `spec.md` |
| `Gate.CoAE.branch-swap` | 兩支各用訊號目標或對比：較好、較差、兩面、看不出 | `coae_gap.py`、CoAE `spec.md` |
| `Gate.PQ.ceiling` | 上限訓練（`PQ:ident…`）夠不夠高（已放棄） | `stage2fm/spec.md` 的判準節 |
| `Gate.PQ.window-pilot` | 小規模訓練後的視窗 R@K（已放棄） | `PhaseQueryReport.md` 的門檻節 |
| `Gate.PQ.window-full` | 正式訓練後的視窗 R@K、尺度與外觀族的穩定（已放棄） | 同上 |
| `Gate.PQ.no-leak` | 來源探針接近隨機、換訊號就失效（已放棄） | 同上 |
| `Gate.shared.tile` | `M-tile` 的 raw R@100 ≥ 0.60 才送 `M-realtest`（以 uni2 cls_avg 0.605、找到 1690 校準；z 只列，不判） | `.claude/changespec/coae/answer.md` 的 `CoAE.Q11` |
| `Gate.shared.realtest` | `M-realtest` 的找到張數（全部 2009 張；找到 = 非 retrieval-only 且 ncc ≥ 0.6） | 下表 |

`Gate.shared.realtest` 的分段（一套，CoAE 與其他方法共用）：
| 找到的張數 | 結論 |
|---|---|
| ≥ 1805 | 追平 phasetiles |
| ≥ 1708 | 通過（85 %） |
| 1419–1707 | 有幫助、未通過 |
| 1269–1418 | 持平 |
| < 1269 | 變差 |

「有幫助」另有附帶條件（遺失數與真值命中），寫在 CoAE `spec.md`。

## 6. 資料與照片組
| 名字 | 定義 |
|---|---|
| `Data.train` | `@pure-train`：ki67_pure 的 train 分割，Ki67 100 張 |
| `Data.probe` | `@pure-half-val`：Ki67 10 張，不訓練、沒有真實照片 |
| `Data.real` | ki67_with_photo：16 張 slide、2009 張真實照片；只驗證，不訓練、不選設定 |
| `Sources.all3` | `ref` 加 `bench`、`bench-real`、`ad-chroma4` |
| `Sources.holdout-chroma` | 去掉 `ad-chroma4`：`ref` 加 `bench`、`bench-real` |
| `Sources.refonly` | 只有 `ref` |
| `Photos.all` | `Data.real` 的全部 2009 張 |
| `Photos.truth` | checklist `correct == 1` 的 1332 張 |
| `Photos.sampled` | tile 層級量測用過的 691 張（`ExpPhaseSeparation/samples.csv`） |
| `Photos.unsampled` | 其餘 1318 張；其中有真值的 641 張是 `Photos.unsampled-truth` |

## 7. 變更紀錄
| 日期 | 改了什麼 | 為什麼 | 出處 |
|---|---|---|---|
| 2026-11-16 | 新檔：名稱體系 v2 的唯一存放處 | 一個代號一份定義；CoAE 的 arm 代號與長名由 `Config.py` 的表產生 | 變更書的名稱體系 v2 節與文件規範節 |
| 2026-11-17 | 加 `CoAE.Q10`–`Q12`、`M-card`、`M-dense`、z/t/e 的定義、`WinEmbed` 讀 CoAE 的文法；checkpoint 改讀最後一個 epoch；`Gate.shared.tile` 改以 raw R@100 ≥ 0.60 判；`M-probe`、`M-audit`、`M-gap` 停用 | z 與召回不一致（干擾尾端非常態）；舊探針量不到召回；使用者決定 | `.claude/changespec/coae/answer.md` `CoAE.Q10`–`Q12` |
| 2026-11-17 | CoAE 的 FM 只寫在編碼器名稱的前綴 `<Fm>CoAE+<arm>`（`Giga`、`Uni2`）；arm 代號不再含 `.fm-<名>`（`NAME_PARTS` 刪該條，修飾順序 8–14 前移為 7–13）；`config_of_arm` 與 `coae_weights` 加 FM 參數；目錄名仍含 FM | 一個 FM 只說一次；gigapath 的名稱、目錄、cache id 不變 | 使用者 2026-11-17 同意；`.claude/changespec/coae-fm/spec.md` |
