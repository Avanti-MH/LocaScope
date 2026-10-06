# 架構檢視筆記(2026-09-22)

還沒處理的架構觀察與建議,依「難度低→高」排序、同難度內再依「嚴重度低→
高」排序。不是進度追蹤表,不重複 `log/TODO.log` 已經記過的決策。已經修掉
的部分(`Runtime.py` 撞名、`_merge_val_scores` 的 `KeyError`)不重述,要查
看 git log。

**怎麼看待這份表:** 這是一人研究型 repo,不是要上線的產品。表裡偏「教科書
式好設計」的項目(checkpoint schema 版本、拆大檔案、統一命名)報酬遞減——
值不值得做取決於這個 repo 打算活多久、以後有沒有別人接手。動手前先問這兩
題,比從上做到下更省力。

| # | 項目 | 難度 | 嚴重度 | 現況 | 現象/原因 | 處理方向 |
|---|---|---|---|---|---|---|
| 1 | `LocaScope.py` 刪除 | 低 | 低 | 卡在使用者手上 | 0 bytes,舊名字殘骸,repo 裡沒人 import 它 | `git rm LocaScope.py`——Claude Code 權限分類器擋下我直接刪,需你自己跑 |
| 2 | `realtest.sh` 歸位 | 低 | 低 | 使用者決定 | 在根目錄,不符 `jobscripts/<Group>/` 慣例,但 `locate_photo.py` 還提到它 | 搬進 `jobscripts/` 前先確認沒有你自己習慣的 `sbatch realtest.sh` 之類路徑依賴 |
| 3 | `DomainGapConfig` 完整性斷言 | 低 | 低 | 延後(等 sweep 結束) | `CAMERA_FULL`/`CAMERA_GEOMETRY_ONLY` 手寫全部欄位,沒機制檢查清單跟得上 dataclass 本身 | 加一段 `dataclasses.fields()` 比對斷言,`MppRoutingHead/Datasets.py` 單檔案 |
| 4 | 命名/詞彙先定案的習慣 | 低 | 中 | 未要求動手 | "arm"→"head"/"baseline"、"arm"→"training_framework" 都是程式碼寫完才改名,兩個姊妹套件各發生一次 | 每個新 training 套件開工前,在 `spec.md` 開一節詞彙表定案,仿 `CLAUDE.md` 自己 Vocabulary 那節的做法,CSV/checkpoint/CLI 欄位名照著抄 |
| 5 | `PrototypeEstMpp.py` 補文件+驗證 | 低 | 中 | 未要求動手 | 硬 bracket 讀 checkpoint top-level 欄位,假設沒寫成文件;寫完後從沒真的餵過一個真實 checkpoint | 補一行假設說明(仿 `cli/evaluate.py` 已有的寫法),找一個真 checkpoint 跑一次 `estimate()` 驗證 |
| 6 | Checkpoint `schema_version` | 中 | 中 | 延後(等 bench 結束) | `Checkpoints.py` 沒版本標記,全 repo 對 checkpoint 硬 bracket 讀取 78 次、`.get()` 防禦僅 6 次;`Store.py` 的 meta 與 mask 快取的 sidecar 都已有這套紀律 | 存檔時加一個版本欄位,讀取端逐步補 `.get()` 防禦,不用一次補齊 |
| 7 | 大檔案拆分 | 中 | 中 | 未要求動手 | `demo_survival_analysis.py`(2986 行)、`bench_mpp_feature_decomposition.py`(2167 行)各塞了好幾個不相關工具,靠字串前綴分辨子命令 | 改成 argparse subparsers,一個工具一個函式、各自參數群組;先盤點 `jobscripts/` 裡的呼叫點 |
| 8 | `FewShotEoMT` 撞名修復 | 中 | 高 | 未要求動手 | 自己手刻 `sys.path.insert`,從沒進過 `_paths.py`;`EncoderBackbone.py`/`Trainer.py`/`Datasets.py` 跟其他套件撞名,目前沒 process 同時 import 兩邊所以還沒炸 | 套用今天對 `MppRoutingHead`/`PrototypicalRoutingHead` 做過的同一招:轉真正的 package、改成完全限定 import |
| 10 | 命名/大小寫慣例統一(`aiNNModel` 等) | 高 | 低 | 未要求動手,不建議做 | 頂層目錄三套大小寫規則並存,`aiNNModel` 誰都不符;純美觀 | 重新命名會牽動所有 import 它的地方,風險/報酬比最差 |
| 11 | Stage 4 Baseline arm 重新設計 | 高 | 中 | 使用者決定 | `cli/train_baseline.py` 已刪,推論時怎麼用學好的 support 這個定位一直沒定案 | 使用者說要之後重新設計來 fit 現在的情況,不是現在 |
| 12 | Flat sys.path / 套件化重構 | 高 | 高 | 未要求動手 | 沒有 `pyproject.toml`,40+ entry point 各自手刻 `sys.path.insert`;`1_`/`2_`/`3_` 數字開頭目錄名本身不合法 package 名稱,是今天兩個真 bug 的根因 | 換成 `src/` layout + 可安裝套件,順便決定要不要把三個數字開頭目錄改名——範圍最大,建議先確認值得投資才動 |

## 快取重構之後新增的項目(2026-09-23)

| # | 項目 | 難度 | 嚴重度 | 現況 | 現象/原因 | 處理方向 |
|---|---|---|---|---|---|---|
| 13 | FewShotEoMT import 會失敗 | 低 | 中 | 使用者決定不處理 | `MaskStore.py` 已刪,`FewShotEoMT/Dataset.py`、`cli/infer.py` 還在 `import MaskStore`;也還 import 已刪的 `Uni2PcaSegFunc.scanned_bounds`(現在是 `TissueSegFunc.scanned_rect`);`InferPMT.sh` 寫死舊的 `result/cache/tiles/<slide>__ds..__<hash>/` 路徑;它自己的 per-WSI 取樣快取 key 也沒有 mask 身分 | 要用它時改走 `TissueMaskConfig.MaskMaker` / `TileSampler.cached`,pre-tile 用 `Store.PreTileCorpus` 的位址 |
| 14 | `KeypointLabelStore` 搬上 `Cache.py` | 中 | 低 | 未要求動手 | features 與 pre-tile 已併進 `Store.py`、按位址讀;只剩 `KeypointLabelStore` 還是自己一套 tmp+replace / cfg_hash / `find`,寫在 `result/cache/keypoint_labels/`。它的 `pretile_id` 已經是 corpus key,所以 label 與 pre-tile 的配對不受影響 | 改用 `Cache.cache_root/atomic_file`,目錄 `<made_by>_labels/<corpus key>/<slide>/ds<d>/<ha_id>/`,讀端從 corpus 算位址而不是 `find` |
| 16 | PARALLEL=1 時兩個 train process 可能重複分割 | 低 | 低 | 未要求動手 | `MppRoutingHead.sh` 的 PARALLEL 模式兩個 process 共用同一個 `<job>_mask/` 快取;第一次同時 miss 同一張 slide 會各跑一次 hest,原子寫入保證結果正確,只是浪費一次 GPU | 先跑 `build_mask_store.py --seg hest --cache-job <train 的 --mask-cache-job>` 預熱 |
