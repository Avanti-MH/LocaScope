# LocaScope 重構計畫：Pipeline / Stage 介面 / locate_photo / Cache `photos` 層

**圖例**：`【新增】` `【修改】` `【刪除】` `【不動】`；沒標的就是原有、不變。
**狀態**：**【定】** 已決定　**【待】** 待確認。
尚未改任何程式。動手前以這份為準；需要動到這裡沒列的檔案，先問使用者。

## 規範

1. **型別**：所有新增或修改的程式，參數、回傳、dataclass 欄位、屬性、**局部變數**都標 `名稱: 型別`
   （含有預設值的參數，例如 `batch_size: Optional[int] = None`）。
   Python 無法內嵌標註的地方（comprehension 變數、lambda 參數、tuple 解構的目標）先宣告再用；做不到的地方保持原樣並說明。
   沒有動到的舊程式不整份補標。
2. **不限定具體方法**【定】：之後會設計更好的 retriever 與 localizer，所以
   - `LocaScopePipeline`、bench、`locate_photo` **只依賴 Protocol 與共用資料型別**
     （`EstMppResult`、`Candidate` / `CandidateSet`、`LocalizationResult` / `LocalizationResultSet`、
     `MppEstimator` / `Retriever` / `Reranker` / `Localizer`），
     不 import `SlidingWinSimRot`、`SiftRansacLocalizer`。具體類別只出現在各方法自己的模組，以及各 stage 套件的 `METHODS` 表。
   - 方法專屬的診斷資料一律是**可選**的：呼叫端用 `getattr` 檢查，方法沒提供就略過那張表（跟 stage 1 的 `neighbour_rows` 同一種做法）。
     - retriever：`tile_sim_rows`、`rotations_searched`、`last_level_seconds`；bench 的 truth 分析另用 `nearest_window` / `window_tile_sims`；
       `frame(level)`（stage 3 單獨重跑時補座標框架；沒有的方法，bench 改成重跑 stage 2）。
     - localizer 結果：`pair_rows()`（`matches` 表）。
3. 只改使用者點名的檔案；不刪檔；git 只看不改。

---

## 1. 全貌（檔案層級）

```
LocaScope/
├─ stage1_estimation/                          【不動】
├─ stage2_retrieval/
│   ├─ StageInterface.py                       【修改】CandidateSet +2 欄位、grids 補型別、Retriever Protocol 縮小並補型別
│   └─ SlidingWinSimRot.py                     【修改】目前唯一的 retriever：換層、預判、延後建 encoder、診斷方法
├─ stage3_localization/
│   ├─ StageInterface.py                       【新增】LocalizationResult、LocalizationResultSet、Localizer
│   └─ SIFT_RANSAC.py                          【修改】SiftRansacResult 繼承 LocalizationResult；localize 簽名改變
├─ utilities/
│   ├─ LocaScopePipeline.py                    【修改】重寫（目前是空檔，原版在 git HEAD）
│   ├─ bench_modules/bench_locascope.py        【修改】synthesis 專用，配合新介面
│   ├─ Cache.py                                【修改】新增 photos 層（見第 9 節）
│   ├─ test_modules/test_cache.py              【修改】photos 鏈的測試
│   ├─ test_modules/test_locascope_stages.py   【修改】跟著 SIFT 的 localize 簽名（約第 296、424 行呼叫處）
│   └─ cli/driver/locate_photo.py              【修改：重新設計】內容已清空，檔案還在；主流、沒有 GT、bench 式 Cache 結構
├─ jobscripts/BenchLocaScope.sh                【修改】拿掉 feature cache 變數
├─ realtest.sh                                 【修改】改呼叫新的 locate_photo
└─ plan.md
```

---

## 2. Pipeline

```
LocaScopePipeline
├─ 建構 __init__(
│       self,
│       wsi: Union[str, SafeSlide],
│       masks: MaskMaker,
│       stage1: StageSpec,                         【修改】原本是已建好的 estimator
│       stage2: StageSpec,                         【修改】原本是已建好的 retriever
│       stage3: StageSpec,                         【修改】原本是已建好的 localizer
│       rerankers: Sequence[StageSpec] = (),       【新增】
│       topk: int = 10,                            【新增】stage 3 驗證幾個候選（= 目前 SIFT recipe 的 topk）【定】
│       bank_cache_job: Optional[str] = None,
│       device: Optional[torch.device] = None,     【新增】建構 stage 用的執行期參數
│       multi_gpu: bool = False,                   【新增】
│       read_workers: int = 0,                     【新增】
│   ) -> None
│     StageSpec = Union[str, Tuple[IdentifiedConfig, type]]   【新增】"method:recipe" 字串，或已解析的 (config, 類別)
│   ├─ feature_cache_job: Optional[str]            【刪除】暫時不用 feature cache（以後可能再用）
│   └─ feature_store_mode: str                     【刪除】
│
├─ 屬性
│   ├─ self.wsi: SafeSlide
│   ├─ self.masks: MaskMaker
│   ├─ self.mask_cfg: TissueMaskConfig
│   ├─ self.base_mpp: float
│   ├─ self.mask: Optional[TissueMask]
│   ├─ self.estimator: Optional[MppEstimator]        延後建構
│   ├─ self.retriever: Optional[Retriever]           【修改】型別是 Protocol，不是 SlidingWinSimRot
│   ├─ self.rerankers: Tuple[Reranker, ...]          【新增】
│   ├─ self.localizer: Optional[Localizer]           【修改】型別是 Protocol，不是 SiftRansacLocalizer
│   ├─ self.topk: int                                【新增】
│   ├─ self.bank_cache_job: Optional[str]
│   ├─ self.tile_size: Optional[int]                 【刪除】pipeline 不再切 query
│   └─ self._level_reason: Dict[int, Optional[str]]  【刪除】失敗記憶搬進 retriever
│
├─ build(self) -> 'LocaScopePipeline'                每張 slide 一次
│   ├─ masks.mask(wsi) → mask
│   ├─ estimator.build(wsi, mask, masks, cache_job)
│   ├─ retriever.build(wsi, mask)                    【修改】不再傳 feature_store
│   ├─ reranker.build(wsi, mask) × n                 【新增】
│   └─ localizer.build(wsi)
│
├─ _feature_store(self)                              【刪除】
├─ _level_ready(self, level: int)                    【刪除】層的處理搬進 retriever
├─ stage1(self, img: np.ndarray) -> EstMppResult
├─ stage2(self, img: np.ndarray, level: int) -> Tuple[QueryPatchContainer, CandidateSet]
│        【修改】→ stage2(self, img: np.ndarray, r1: EstMppResult) -> CandidateSet
│               經過 retriever，再依序經過 rerankers；不再回傳切好的 query
├─ candidates_at(self, level: int, candidates)       【刪除】搬到 bench（用 retriever 的可選 frame(level)）
├─ stage3(self, qc: QueryPatchContainer, cs: CandidateSet)
│        【修改】→ stage3(self, img: np.ndarray, cs: CandidateSet) -> LocalizationResultSet
├─ run(self, img: np.ndarray, keep_objects: bool = False) -> LocaScopeQueryResult
│        【修改】保留；→ run(self, img: np.ndarray) -> LocaScopeQueryResult，由 stage1/2/3 串起來
└─ UnusableLevel（例外類別）                          【刪除】改看 stage2.candidates 是否為空

LocaScopeQueryResult（@dataclass，欄位全部標型別）
├─ stage1: Optional[EstMppResult]                    原本 stage1: object
├─ stage2: Optional[CandidateSet]                    【修改】原本叫 retrieval
├─ stage3: Optional[LocalizationResultSet]           【修改】原本叫 ranks: Optional[list]
├─ error: Optional[str]
├─ t_stage1_s: Optional[float]
├─ t_stage2_s: Optional[float]
├─ t_stage3_s: Optional[float]
├─ est_mpp, routed_level, refine                     【刪除】重複
├─ unusable_level: bool                              【刪除】stage2.candidates 為空即是
├─ t_level_s                                         【刪除】改由 retriever 的可選屬性 last_level_seconds 提供
└─ retriever / localizer / query_qc                  【刪除】
```

---

## 3. 一張 shot 的流程圖

```
img: np.ndarray
 │
 ▼
① stage 1  estimator.estimate(img) ──────────────────────► r1: EstMppResult（chosen_level = L）
 │   失敗 → error='stage1 failed'，結束
 ▼
┄┄┄┄ 【刪除】pipeline 原本在這裡做的事 ┄┄┄┄
   _level_ready(L)：先 build 該層 feature map，記住失敗
   不能用 → unusable_level=True，結束
┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
 ▼
② stage 2 第一階段  retriever.retrieve(img, r1)                    【修改】
 │     原：pipeline 逐步呼叫 build_wsi_features → build_query_features → compute_sim_maps → candidate_set
 │     新：換層流程在 retriever 內部（各方法自己決定；SlidingWinSimRot 見第 4 節）
 ▼   cs: CandidateSet(candidates, level, ds, grids,
 │                    irretrievable_lvl 【新增】, alter_lvl 【新增】)
 │   candidates 為空（所有層都搜不了）→ 結束，stage3 = None          【修改】原本是 unusable_level
 ▼
③ stage 2 第二階段  rerankers 依序（沒有就略過）                       【新增】
 ▼
④ stage 3  localizer.localize(img, cs, topk)                        【修改】原 localize_top(qc, cs)；現在傳 img 與 topk
 │   → LocalizationResultSet（results 依名次排序）
 ▼
LocaScopeQueryResult(stage1, stage2, stage3, error, t_stage1_s, t_stage2_s, t_stage3_s)
```

---

## 4. Stage 2

### 4.1 介面（`stage2_retrieval/StageInterface.py`）【修改】
```
Candidate（region_index: int, lattice: str, row: int, col: int, rotation: int, score: float）   [不動]
CandidateSet（@dataclass(frozen=True)）
├─ candidates: Tuple[Candidate, ...] / level: int / ds: float             [原有]
├─ grids: Tuple[PatchGrid, ...]                                            【修改】原本 grids: tuple
├─ irretrievable_lvl: Tuple[int, ...]                                      【新增】必填；搜不了的層
└─ alter_lvl: Tuple[int, ...]                                              【新增】必填；依序試過的替代層，最後成功的 = level
Retriever（Protocol）                                                      【修改】縮小並補型別
├─ encoder: Optional[TileEncoder]                  沒有 tile encoder 的方法為 None【待】
├─ tile_size: int / overlap: bool                  query 的切法，bench / locate_photo 要用它算視窗的 level-0 方框（cs.rows）
├─ build(self, wsi: Union[str, SafeSlide], mask: Optional[TissueMask], feature_store: Optional[Any] = None) -> 'Retriever'
└─ retrieve(self, query: np.ndarray, estimate: EstMppResult) -> CandidateSet
Reranker（Protocol）                                                       [不動]
```
- 判斷有沒有換層：比較 `cs.level` 與 stage 1 的 `chosen_level`。
- 所有層都不行：`candidates` 為空、`irretrievable_lvl` 為全部層、`level` 填路由的那層。

### 4.2 目前唯一的實作：`SlidingWinSimRot`（別人只透過上面的 Protocol 使用它）
```
retrieve(self, query: np.ndarray, estimate: EstMppResult) -> CandidateSet                 【修改】
│ 【新增】0. 取得 query 的格數（4 個旋轉各自的行列數）
│ 【新增】1. 要試的層：從 estimate.chosen_level 開始，往細的方向，最近的先試
│       ├─ 已知更粗的層 → 直接記入 irretrievable_lvl，不試
│       ├─ 預判（不讀像素）：_fitting_rotations(self, level: int, query_shape: Tuple[int, int]) -> Tuple[int, ...]
│       │       回傳放得下的旋轉；空 tuple → 記入 irretrievable_lvl，換下一層（記入 alter_lvl）
│       └─ build 時例外 → 同上
│ 2. build_wsi_features(level)       encode 這層，存 _by_ds                   [原有]
│ 3. build_query_features(query)     4 個旋轉 encode；換層重試只做一次         [原有]
│ 4. compute_sim_maps()              simmap 在實例上（GPU，不落硬碟）          [原有]
│ 5. candidate_set()                 每個格點 topk → 排序 → min_sep → 前 k     [原有]
│        沒有任何視窗時丟 ValueError                                          【刪除】由預判與換層取代
│ 【新增】6. 全部層都不行 → candidates 為空
│ 【新增】7. 記下準備層耗時 → self.last_level_seconds
└─ 回傳 CandidateSet

建構與屬性
├─ __init__(self, cfg: SlidingWinSimRotConfig, device: Optional[torch.device] = None,
│           multi_gpu: bool = False, read_workers: int = 0) -> None          【修改】不再在這裡建 encoder
├─ encoder（@property）-> TileEncoder               【修改】第一次用到才建
├─ model（@property）-> torch.nn.Module             【修改】原本 self.model = self.encoder.model（IdentifiedBuild.weights_id 要用）
├─ self._by_ds: Dict[float, tuple]                   原有，跨 query 保留
├─ self.sim_maps_by_rot: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]   原有；simmap 只在實例記憶體，不落硬碟
├─ self.last_level_seconds: float                    【新增】診斷
└─ self._unusable: Dict[int, str]                    【新增】只記與 query 無關的失敗（層沒有可用 region、build 例外）

方法
├─ build(...) -> 'SlidingWinSimRot'                  [原有，feature_store 參數保留]
├─ build_wsi_features / build_query_features / compute_sim_maps / candidate_set   [原有，public]
├─ window_tile_sims(self, c: Candidate) -> torch.Tensor                    [原有]
├─ nearest_window(self, x_l0: float, y_l0: float, rotation: int,
│                 lattices: Tuple[str, ...] = ('main', 'offset')) -> Tuple[Candidate, int, float]   [原有，bench 專用，需要 GT]
├─ tile_sim_rows(self, candidates: Sequence[Candidate]) -> List[Dict[str, Union[int, float]]]       【新增】診斷：原本 bench 的 _tile_rows，bench 與 locate_photo 共用
└─ rotations_searched(self) -> Tuple[int, ...]                             【新增】診斷
```
- 換層規則：往細的找；region 格數隨 ds 增加單調遞減而 query 格數固定，所以路由層失敗時更粗的層一定也失敗。
  region 過濾（`mask.patchable`）註解明說單調；「整個 query 放得下」是推論【待驗證】。
- 預判用 `region_grids`（不讀像素）比 region 格數與 query 格數，放不下就不 encode（否則整層 encode 完才發現）。`PatchGrid` 行列屬性名稱動手時確認。

---

## 5. Stage 3

```
stage3_localization/StageInterface.py                                         【新增】（效仿 stage 2 的 Candidate / CandidateSet）
├─ LocalizationResult（@dataclass(frozen=True)）     共用的結果型別，方法的結果繼承它【定】
│                                                    stage 3 也有「候選」：像 stage 2 的 Candidate / CandidateSet，是候選與候選名單
│     x0: float / y0: float / center_x0: float / center_y0: float
│     confidence: float                              越大越有把握（0～1）
│     rank: int / candidate: Candidate / ds: float / level: int
│     row(self) -> Dict[str, Optional[Union[int, float, bool, str]]]      一列表格（寫 cache entry 的 CSV；每種方法決定自己的欄位）
├─ LocalizationResultSet（@dataclass(frozen=True）  依名次排序
│     results: Tuple[LocalizationResult, ...]         【定】對應 stage 2 的 candidates（原稿寫 LocalizationResults，欄位名稱用小寫）
└─ Localizer（Protocol）
      build(self, wsi: Union[str, SafeSlide]) -> 'Localizer'
      localize(self, query: np.ndarray, cs: CandidateSet, topk: int) -> LocalizationResultSet
```
- **約定**：`confidence` 為 0.0 代表沒有定位成功；此時 `x0 / y0 / center_x0 / center_y0` 是該候選視窗本身的位置（退回值）。
  新的 localizer 也要遵守。因此 locate_photo 不需要知道「成功與否」，取 confidence 最大的結果即可。

```
stage3_localization/SIFT_RANSAC.py                                            【修改】
├─ SiftRansacResult 繼承 LocalizationResult
│     唯一建構點 SIFT_RANSAC.py:379（全部關鍵字參數）多傳 confidence：
│       success 為 False 或 match_count <= 0 → 0.0；否則 inlier_count / match_count
│       （依 log/TODO.log 2026-08-08：佔比比絕對 inlier 數更能分開對錯）
├─ SIFT 專屬欄位留在子類別：H、inlier_count、match_count、success、crop_l0、pairs
├─ row() 覆寫：現有欄位名稱不變（n_good、n_inliers、h00…h22、crop_*），加上 confidence
├─ pair_rows()                                                                  [不動，可選的 matches 表]
└─ SiftRansacLocalizer
      localize(self, query, cs, topk) -> LocalizationResultSet                  【修改】就是原 localize_top 的內容包成 Set
      localize_one(self, query, cs, rank) -> SiftRansacResult                   【修改】原本單一名次的 localize 改名
      prepare / read_wsi_crop / detect_and_match / estimate_homography           [不動]
```

---

## 6. bench（synthesis）

```
bench main
├─ 解析 --stage1/2/3                                    [原有]
├─ 建 pipeline：原本 bench 自己的 Stage 類別（id、record、延後建構）+ 手動組裝
│                                                        【修改】把規格交給 pipeline；id/record 與 pipeline 的規格解析重複，改的時候合併
├─ --features-cache-job / --feature-store-mode           【刪除】
└─ 每張 slide
    ├─ FovSupply 合成照片                                 [原有]
    ├─ 逐 route 查 stage1/2/3 entry 的 hit / miss          [原有]
    ├─ run_stage1                                         [原有]
    ├─ run_stage2
    │     ├─ stage1 route：pl.stage2(img, r1)             【修改】簽名
    │     ├─ oracle route：placed_estimate(true_ds: float, true_level: int, base_mpp: float) -> EstMppResult   【新增】用放置層的真實值建完整結果
    │     ├─ output 表：cs.rows(qc)（qc 由 bench 用 retriever.tile_size / overlap 切）
    │     │       + irretrievable_lvl、alter_lvl 欄位；candidates 為空時也寫一列      【新增】
    │     ├─ tile_sims：getattr(retriever, 'tile_sim_rows')   【修改】原本是 bench 的 _tile_rows；沒有就略過
    │     └─ truth / truth_sim：getattr(retriever, 'nearest_window')   [原有] 沒有就略過
    ├─ run_stage3：pl.stage3(img, cs)                     【修改】簽名；表格用 results 的 row()，matches 用可選的 pair_rows()
    ├─ stage 3 單獨重跑
    │     candidates_at → frame_from_rows(retriever: Retriever, rows: List[Dict[str, str]]) -> CandidateSet   【修改】bench 自己的 helper，用可選的 retriever.frame(level)；沒有就重跑 stage 2
    └─ 寫 entry（原子寫入）                                [原有]
```

---

## 7. locate_photo.py（主流，沒有 GT）

```
locate_photo.py                                                          【修改：重新設計】
├─ 參數：--stage1/2/3 <method>:<recipe>、--seg、--mask-cache-job、--limit …（跟 bench 同款）
├─ slide 範圍：list_names(dataset='ki67_with_photo') 全部 16 張（沒有 split）
└─ 每張 slide
    ├─ 照片資料夾 = entry.related['photos']
    ├─ 照片集 id = 「檔名:檔案大小」排序後 hash 取 16 碼
    ├─ Address(slide, seg, region, photos=<id>)
    │     shots/ entry：index, photo, bytes, width, height           【新增】
    ├─ stage1 / stage2 / stage3 entry：hit 略過，miss 才載入模型
    │     record 不放 tables；用 members 判斷表是否齊全
    ├─ 每張照片：pl.stage1 → pl.stage2 → pl.stage3
    ├─ 寫表：output、stage 3 的 row()、可選表（matches、neighbours / probs / votes、tile_sims 全部 K 個候選）、耗時
    │        沒有 truth / truth_sim；方法沒提供的可選表略過
    ├─ 最終答案：stage3.results 中 confidence 最大者的 (center_x0, center_y0)
    │        全為 0.0 時就是退回值（rank 0 的候選視窗位置），不需要另外算 stage 2 的視窗中心
    ├─ 信心分級：不分級，只輸出原始 confidence 數值（門檻待用真實照片校準，見 TODO.log 2026-08-08）
    └─ 一張 slide 做完才寫 entry（Resume 先不做，預估整批約 2 小時）
```

## 8. 主流 / 支流

```
主流：locate_photo（真實照片，沒有 GT）            支流：bench（synthesis，有 GT）
   呼叫 pipeline 逐段                                  呼叫 pipeline 逐段
   表：output、stage 3 row()、可選表                    表：同左 + truth、truth_sim
   cache：photos 鏈                                     cache：plan/draw/render 鏈
        └──────────── 共用：pipeline + stage 介面 + 遮罩 cache + stage id（方法-recipe-hash）────────────┘
real 結果怎麼讀：先不管
```

---

## 9. Cache：新增 `photos` 層

真實照片（`ki67_with_photo` 的 `entry.related['photos']`）沒有 sampler 抽位置、也沒有相機，放不進現有的 `plan → draw → render` 鏈。
stage 1 / 2 / 3 的結果要像 bench 一樣是 cache entry，所以需要一個正式的層來代表「一組真實照片」。

### 目標路徑【定】

```
result/cache/<job>/
└─ slide=<slide>/
   └─ seg=<seg id>/
      └─ region=<region id>/
         └─ photos=<照片集 id>/                     ← 新的層
            ├─ shots/                                ← 新的 entry：index → 檔名、尺寸
            │    index_<id>.csv   record_<id>.json
            ├─ stage1/                               ← stage1 entry 可放在 photos 層
            ├─ stage1=<s1>/
            │   ├─ stage2/
            │   └─ stage2=<s2>/stage3/
```
- 沒有 `stage1=oracle`（真實照片沒有「放置的層」）。
- 現有的合成鏈完全不變，現有 cache 的路徑不受影響。

### `Cache.py` 要改的【定】
現況：
```
TREE:    dataset, slide, seg←slide, region←seg, grid←region, plan←region,
         draw←plan, render←draw, stage1←render, stage2←stage1
ENTRIES: split(dataset) mask(slide) features(grid,draw,render) draw(plan)
         render(draw) stage1(render) stage2(stage1) stage3(stage2)
         labels(draw) chainstack(slide)
```
1. `TREE` 新增 `photos`，父層是 `region`。
2. `stage1` 的父層從單一的 `render` 改成「`render` 或 `photos`」：
   - `TREE` 現在是「一個 kind → 一個父層」的字典，要允許多個父層。
   - `_chain(kind)`、`Address.__init__` 的鏈驗證（現在用 `max(_chain(k) …)` 並要求 levels 剛好是一條鏈）、
     `Address.children()`（現在比較 `TREE.get(kind) != self.leaf`）都要跟著改。
   - 兩個父層同時出現在一個 address 裡必須報錯。
3. `ENTRIES` 新增 `shots`，放在 `photos` 層：`'shots': ('photos',)`。
4. `ENTRIES['stage1']` 改成 `('render', 'photos')`。`stage2`、`stage3` 的位置不變。

連帶要檢查：`test_cache.py`（補 `photos` 鏈、兩個父層不能並存、舊鏈行為不變）、
`utilities/cli/inspect_cache_store/purge_cache.py`（依 `TREE` 走層）、
「只有 `Cache.py` 才寫 `<kind>=` 路徑」的 lint（`test_config_identity`）、其他用到 `Address` / `ENTRIES` 的地方（已查：`Store.py`、`stage1_estimation/StageInterface.py` 的註解）。

### 照片集 id【定】
一個 id 代表一張 slide 的整個照片資料夾（不是一張照片一個）；每張照片在集合裡靠 `index` 區分（取檔名結尾的數字）。
算法：把資料夾裡每個檔案寫成 `檔名:檔案大小`，依檔名排序後整串做 hash，取 16 碼。
- 增加、刪除、改名、換大小的照片都會產生新的 id；搬動或重新複製資料夾不影響 id（不含路徑、日期）。
- 不會發現「大小不變、內容被改」。內容 hash 要讀約 9 GB（2009 張 BMP），不採用。

### 主流與支流的表【定：不分 entry】
每個 stage 的 entry 一律寫完整的表，跟 bench 現在一樣；`ENTRIES` 不為此新增 kind。

**record 的寫法【定】**（bench 與 locate_photo 都照做）：
`Entry.status` 是整個 variant 一起判斷，比對 `id`、`upstream`、`versions`、`env`、`parts`，再加上呼叫端放進 record 的**每個欄位**（`ConfigIdentity.record_diff`）。
bench 現在把 `tables=[…]` 放進 record，所以「先跑主流再補支流」與「先跑完整再只要主流」兩個方向都會判成過期並重寫（後者還會把支流表刪掉）。
- 身分（放進 record）：設定、上游 id、版本、環境，以及會改變結果的選項（例如 `limit`）。**record 裡不放 `tables`。**
- 有哪些表是內容，不是身分：用 Cache 本來就記的 `members` 判斷。`status` 為 `hit` 且需要的表都在 `members` 裡才算命中；缺表當 `miss`。
- 不需要改 `Cache.py`。

### `shots` entry 的欄位【定】
`index, photo, bytes, width, height`。

### Resume【定：先不做】
locate_photo 採 bench 式「一張 slide 做完才寫 entry」，被殺掉就整張 slide 重來；預估整批約 2 小時，所以先不做區塊切分或 parts。
（若以後要做：照片按 `index // N` 切成區塊，每個區塊各自一個 `photos=<id>`，用 Cache 本來的 hit / miss 當 resume；
代價是被殺重跑時要重新 encode 用到的層，因為目前不用 feature cache。）

---

## 10. 要動手的程式碼

### 第一批（Stage 2 + Stage 3 + pipeline + bench）
| 順序 | 檔案 | 動作 |
|---|---|---|
| 1 | `stage2_retrieval/StageInterface.py` | `CandidateSet` 加兩個必填欄位、`grids: Tuple[PatchGrid, ...]`；`Retriever` Protocol 縮小並補型別 |
| 2 | `stage3_localization/StageInterface.py` | **新增**：`LocalizationResult`、`LocalizationResultSet`、`Localizer` |
| 3 | `stage3_localization/SIFT_RANSAC.py` | `SiftRansacResult` 繼承 `LocalizationResult` 並傳 `confidence`；`row()` 覆寫；`localize` / `localize_one` 簽名 |
| 4 | `stage2_retrieval/SlidingWinSimRot.py` | 換層與預判；encoder / model 延後建構；新增 `tile_sim_rows`、`rotations_searched`、`last_level_seconds`；建構 `CandidateSet` 補新欄位 |
| 5 | `utilities/LocaScopePipeline.py` | 重寫（第 2 節） |
| 6 | `utilities/bench_modules/bench_locascope.py` | 第 6 節 |
| 7 | `utilities/test_modules/test_locascope_stages.py` | 跟著 `localize` 簽名 |
| 8 | `jobscripts/BenchLocaScope.sh` | 拿掉 `FEATURES_CACHE_JOB`、`FEATURE_STORE_MODE` |
| 9 | 其他建構 `CandidateSet(…)` 的地方 | 先 grep 列全再補 |

### 第二批
| 順序 | 檔案 | 動作 |
|---|---|---|
| 10 | `utilities/Cache.py` + `utilities/test_modules/test_cache.py` | `photos` 層（第 9 節）；連帶檢查 `purge_cache.py`、lint |
| 11 | `utilities/cli/driver/locate_photo.py` | 第 7 節 |
| 12 | `realtest.sh` | 改呼叫新的 locate_photo，`--array` 涵蓋 16 張 |

### 不動
stage 1 全部、文件（程式定案後再更新）。

---

## 11. 待確認與風險

已確認【定】：
1. `LocalizationResult` / `LocalizationResultSet` 是共用 dataclass（仿 stage 2 的候選與候選名單），集合欄位叫 `results`。
2. `topk` 由 pipeline 建構參數提供，預設 10。
3. `SiftRansacLocalizer.localize` 簽名改變，連動 `test_locascope_stages.py`（同意）。

尚待確認：
4. `Retriever.encoder` 改成 `Optional`；Protocol 保留 `tile_size`、`overlap`（bench / locate_photo 算視窗方框要用）。
5. 「整個 query 放得下」對 ds 單調，是推論，需驗證。
6. 驗證需要舊版的參考輸出：pipeline 現在是空檔，無法直接跑舊版；要比對改動前後的 stage 2 表格，得先在別的目錄還原舊版跑一次 smoke（例如使用者自己 `git worktree add`）。
7. `CandidateSet` 兩個新欄位是必填（跟 stage 1 一致），所以要補全部建構點。
8. 信心分級的門檻待用真實照片校準（TODO.log 2026-08-08）；locate_photo 先只輸出原始數值。

---

# 收斂（已完成）：共用部分搬出 bench_locascope

bench_locascope.py 同時是執行檔，也是 `bench_stage1_mpp`、`bench_window_retrieval`、`plot_locascope`、
`analyze_stage1_metrics` 借用函式的函式庫，所以簽名動不了。已拆開：

- **新增 `utilities/bench_modules/BenchCommon.py`**：`Stage`、`Tables`、`status`、表格讀寫、`stage_entries`、
  `stage1_record`、`ROLES`、`run_stage1`（回傳完整 `EstMppResult`，原本的 `estimate_stage1` 與回傳層數的外殼合一）、
  `run_stage2`、`run_stage3`、`Truth`、`Shot`、`run_from_args`、`run_slides`、`supply_for`、`slide_supplies` 等，
  以及**唯一的一條迴圈 `run_slide_stages`**（算 hit/miss → 建 Tables → 逐張照片跑 stage → 寫 entry）。
- **`bench_locascope.py`**：只剩合成照片的來源 `synthetic_shots`、`bench_slide`（呼叫 `run_slide_stages`）和 `main`。
- **`locate_photo.py`**：同一條 `run_slide_stages`，`truth=False`、只有 stage1 route、多一張 `answer` 表。
- 四個借用檔案只改了 import 那一行（與 docstring 裡的名稱）：`bench_stage1_mpp.py`、`bench_window_retrieval.py`、
  `plot_locascope.py`、`analyze_stage1_metrics.py`。
- `Stage` 現在用 `LocaScopePipeline.construct_stage` 建物件（pipeline 與 bench 只有一份建構邏輯）。
- `stage_entries(..., truth=True)`：real 的 stage 2 record 不再帶 `truth`。
- 未動：`bench_stage1_mpp` 自己「方法在外層、slide 在內層」的迴圈（只共用 `Tables` / `status` / `run_stage1`）。
