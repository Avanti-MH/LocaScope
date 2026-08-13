# query_sim — Microscope FOV Simulator（整合設計）

> **這是 M4 當初的整合計畫，不是現在的目錄結構。** `synth_fov_generator.py` 已
> 不存在，`source/tissue_filter.py`（`is_tissue`/`classify_region`）從未落地——
> 那條路線被 M4.2「`region_type` 被實測否決」推翻，見 `log/MILESTONE.log` 與
> `log/TODO.log`。`camera.py`（M4.1 的 Camera 抽象）也是這份文件寫完之後才加的,
> 不在下面的目錄結構裡。保留原文是因為被推翻的路線本身是負面結果,不是要重寫
> 成看起來從一開始就對；要看現在實際長什麼樣,直接看 `query_sim/` 底下的檔案。

整合 `query_sim/`（現有模組化 augmentation）與 `synth_fov_generator.py`（批量 + GT）為一個
統一 package，同時保留兩者最強的部份：

- **query_sim**：模組化 augment（cv2 精度）、MPP → level 自動選擇、單張互動 demo
- **synth_fov_generator**：批量生成、Ground Truth 記錄、tissue 過濾、region 分類、rotation + scale

---

## 動機

目前兩套工具能力互補但重複：

| | `query_sim/` | `synth_fov_generator.py` |
|---|---|---|
| 用途 | 互動 demo、單張 query + 效果圖 | 批量產 dataset + GT，給 pipeline benchmark |
| 架構 | 4 檔案 modular | 單檔 all-in-one |
| Rotation / Scale | ✗ | ✓ |
| GT / Tissue filter / Region 分類 | ✗ | ✓ |
| Field mask / Chromatic / JPEG / Stage shift | ✓ | ✗ |
| Lens distortion 精度 | ✓ sub-pixel (`cv2.remap`) | ⚠ nearest-neighbor int index |
| Vignette 平滑度 | ✓ Gaussian | r² polynomial |
| 依賴 | cv2 + PIL + numpy + openslide | PIL + numpy + openslide（無 cv2） |

**目標**：合成一個 package，同時支援 demo + batch，augmentation 統一取兩邊的較佳實作。

---

## 目錄結構

```
query_sim/
├── __init__.py
│
├── config.py                  ← DomainGapConfig（所有 augment / 取像參數）
├── record.py                  ← FOVRecord（GT dataclass）
│
├── source/                    ← 從 WSI 取「原始」patch（無 augment）
│   ├── wsi_query.py           ← QueryFromWSI（保留 MPP → level 邏輯）
│   └── tissue_filter.py       ← is_tissue、classify_region
│
├── augment/                   ← 個別 augmentation function 集合
│   ├── color.py               ← color、color_temp、brightness / contrast、jpeg
│   ├── field.py               ← vignette、stage_shift
│   ├── lens.py                ← distortion、defocus、chromatic
│   ├── geometry.py            ← rotation (0/90/180/270 + jitter)、scale     ← 新
│   └── noise.py               ← gaussian noise
│
├── pipeline.py                ← simulate_microscope_photo(img, cfg) 串接所有 augment
│                                simulate_with_gt(cfg, output_wh) → (img, FOVRecord)
│                                兩段式：場景階段決定什麼落到感測器上，裁切，
│                                然後感測器階段。見下方「op 的順序」
│
├── generator.py               ← 批量生成 loop（tissue retry、stratify、CSV 寫入）
│
├── cli/                       ← 兩個入口對應原本兩支 script
│   ├── demo.py                ← 舊 simulate_microscope_photo.py（單張 + effects grid）
│   └── batch.py               ← 舊 synth_fov_generator.py（N 張 + gt.csv）
│
└── result/                    ← 輸出（gitignored）
```

---

## 三層 API（清楚分責）

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3 — cli/                                             │
│    demo.py:  1 張 → effects panel figure                    │
│    batch.py: N 張 → images/ + gt.csv                        │
├─────────────────────────────────────────────────────────────┤
│  Layer 2 — generator.py                                     │
│    generate(cfg, n, out_dir):                               │
│      迴圈: source → tissue filter → pipeline → save + GT    │
│    generate_one(cfg) → (img, FOVRecord)                     │
├─────────────────────────────────────────────────────────────┤
│  Layer 1 — pipeline.py                                      │
│    simulate_microscope_photo(img, cfg) → img                │
│    simulate_with_gt(img, cfg) → (img, params_dict)          │
├─────────────────────────────────────────────────────────────┤
│  Layer 0 — augment/*                                        │
│    apply_vignette(img, strength) …                          │
│    apply_rotation(img, angle) … 每個獨立可測                │
└─────────────────────────────────────────────────────────────┘

source/wsi_query.py 是獨立子系統：拿 WSI + 位置 → raw PIL query
```

每一層都能單獨呼叫：

- **Layer 0** — 論文寫 methodology 時可以單獨挑幾個 augment 展示
- **Layer 1** — pipeline 直接餵 image + cfg，適合寫 unit test
- **Layer 2** — generator 給 batch loop 或 notebook 呼叫
- **Layer 3** — 使用者 CLI 入口

---

## DomainGapConfig 統一 spec

```python
@dataclass
class DomainGapConfig:
    # Source
    wh_ratio: str = '4:3'
    MPixels: float = 12
    query_mpp: float = 0.25
    fov_size: Optional[int] = None      # 若指定則 bypass MPixels 算法

    # Rotation (from synth_fov)
    rotation_choices: Tuple[int, ...] = (0, 90, 180, 270)
    angle_jitter_deg: float = 3.0

    # Scale (from synth_fov)
    scale_range: Tuple[float, float] = (0.90, 1.15)

    # Color
    brightness_range: Tuple[float, float] = (-0.08, 0.08)
    contrast_range:   Tuple[float, float] = (-0.08, 0.08)
    saturation:       float               = 1.0
    color_temp_range: Tuple[float, float] = (-0.12, 0.12)

    # Field
    vignette_range:  Tuple[float, float] = (0.15, 0.45)
    stage_shift_max: int                 = 3

    # Lens
    distortion_k1_range: Tuple[float, float] = (-0.04, 0.04)
    distortion_k2:       float               = 0.0
    defocus_radius:      int                 = 2
    chromatic_shift:     int                 = 2

    # Noise + JPEG
    noise_sigma:  float = 3.0
    jpeg_quality: int   = 85
```

**每個都是 `range` 而不是 fixed value** → batch 生成隨機採樣；demo 模式可設 `(v, v)` 得固定值。

---

## Augmentation 合併決策

| 效果 | 用哪邊實作 | 原因 |
|---|---|---|
| color / brightness / contrast | **query_sim** (`cv2` HSV) | HSV 空間合理 |
| color_temp | **synth_fov** | query_sim 沒有 |
| vignette | **query_sim** (Gaussian) | 比 r² polynomial 平滑 |
| stage_shift | **query_sim** | synth_fov 沒有 |
| lens distortion | **query_sim** (`cv2.remap`) | sub-pixel accurate |
| defocus | **query_sim** (disk kernel) | 更真實 |
| chromatic | **query_sim** | synth_fov 沒有 |
| jpeg | **query_sim** | synth_fov 沒有 |
| noise | 兩邊等價 | 隨便 |
| **rotation (90x + jitter)** | **synth_fov** | query_sim 沒有 |
| **scale** | **synth_fov** | query_sim 沒有 |

**依賴**：合併版統一用 `cv2` + PIL + numpy + openslide（synth_fov 純 PIL/numpy 的部分改成 cv2）。

---

## Rotation 特別處理（連動 retrieval TODO）

`geometry.py` 的 `apply_rotation` 有兩個介面：

```python
apply_rotation(img, angle=None, cfg=None) → (img, angle_used)
    angle=None 時從 cfg.rotation_choices 隨機選 + jitter
    angle=int 時強制使用（測試 / benchmark 用）
```

`FOVRecord.rot_deg` 記錄實際套用的角度。這樣：

1. **`--rotation-only` 模式**：只旋轉不套 photometric augment
   → 給 [rotation-aware retrieval TODO](../log/TODO.log) 產 benchmark 資料
2. **完整模式**：所有 augment 都套 → real-world dataset

**未來也可用作 rotation classifier 的 training set**（若走 rotation-invariant embedding 路線）。

---

## Ground Truth Record

```python
@dataclass
class FOVRecord:
    filename:    str
    wsi:         str
    level:       int
    fov_size:    int

    # Position (level-0 座標)
    gt_x:        int
    gt_y:        int
    region_type: str          # feature_rich / moderate / sparse

    # Geometry
    rot_deg:      int          # 0 / 90 / 180 / 270
    angle_jitter: float
    scale:        float

    # Photometric（全部記錄，方便反推 / debug）
    vignette_strength: float
    color_temp:        float
    brightness:        float
    contrast:          float
    distortion_k1:     float
    defocus_radius:    int
    chromatic_shift:   int
    noise_sigma:       float
    jpeg_quality:      int
```

每 row = 一張 FOV。CSV 用 `csv.DictWriter(fieldnames=asdict(rec).keys())` 寫入。

---

## CLI 對照

| 舊 | 新 |
|---|---|
| `python simulate_microscope_photo.py <wsi> --x 0 --y 0` | `python -m query_sim.cli.demo <wsi> --x 0 --y 0` |
| `python synth_fov_generator.py --wsi ... --n 300` | `python -m query_sim.cli.batch <wsi> --n 300 --out ./synth_fovs` |

- `demo.py` — import `pipeline.simulate_microscope_photo`，印 effects panel（單張比對用）
- `batch.py` — 呼叫 `generator.generate(...)`，產生 dataset + gt.csv

---

## 遷移建議順序

```
Step 1  augment/ 集中             ← 把 capture / field / lens 搬過來
                                    + 新增 geometry.py（rotation + scale）
                                    + 補上 color_temp / brightness_contrast
Step 2  config.py                 ← 合併兩邊參數為一個 DomainGapConfig
Step 3  source/wsi_query.py       ← rename QueryFromWSI，保留 MPP 邏輯
Step 4  source/tissue_filter.py   ← is_tissue、classify_region 搬進來
Step 5  pipeline.py               ← 抽 simulate_microscope_photo(img, cfg)
Step 6  generator.py              ← 抽批量 loop + tissue retry + stratify + CSV
Step 7  cli/demo.py, cli/batch.py ← 兩支 CLI 入口
Step 8  __init__.py re-export     ← 舊 import 路徑不壞掉（backward compat）
Step 9  刪除 synth_fov_generator.py（若存在於 repo 內）
```

---

## 命名說明

`query_sim` 的 **query = 整個 LocaScope 專案的 FoV Picture**（顯微鏡下拍到的一張視野）。
這裡的 query 跟 retrieval 語境的 query 是**同一件事**，不是兩個概念 → 名稱不改，
`query_sim` = 「query（FoV picture）的 simulator」，語意正確。

---

## 可討論的取捨

1. **`generator.py` 用 iterator 還是 list？**
   iterator 省記憶體、可以 pipeline 串接；list 簡單直觀。batch 場景兩者都 OK。

2. **`source/wsi_query.py` 要不要跟 `PatchingLib.WsiTissuesContainer` 整合？**
   後者已有完整 WSI 讀取 + patching 邏輯，可能重複實作 WSI region 讀取。
   風險：兩邊 use case 不同（一個要 raw crop、一個要 grid patches），強行整合可能反而複雜。

3. **`--seed` 統一入口**
   `random.seed / np.random.seed / torch.manual_seed` 統一設定，方便 reproduce。
   Config 也可加 `seed: Optional[int]` 欄位。

4. **要不要支援多 WSI 混合輸出？**
   例如 `--wsi wsi1.svs wsi2.svs --n 300` 平均產出。`FOVRecord.wsi` 已支援。

---

## 相關 TODO

- `PatchingLib` crop() TODO(A/B) — sub-container 語義、overlap 對齊
- retrieval rotation-aware TODO — 需要本 package 產出旋轉 GT 資料集才能 benchmark


---

## op 的順序，以及它為什麼是兩段

`_apply_params` 把 12 個 op 分成兩段，分界是**裁切到感測器尺寸**。分在哪一段由
一件事決定：這個 op 的幾何是以哪個畫面為基準量出來的。

```
── 場景階段：決定什麼落到感測器上 ────────────────
   crop_bounding_square        讀 1767² = FoV 對角線見方
   rotation                    旋轉需要畫面外的像素轉進來
   scale
   stage_shift                 載物台抖動 = 重新取景
   ↓
   裁切到 output + SENSOR_MARGIN
   ↓
── 感測器階段：光學與感測器對這張影像做了什麼 ──────
   color / brightness / color_temp
   distortion                  ┐ 讀鄰域，需要 margin
   defocus                     │
   chromatic                   ┘
   ↓
   裁切到精確的 output（丟掉 margin）
   ↓
   vignette                    逐像素，要拿到精確的感測器畫面
   noise
   jpeg                        8×8 區塊對齊交付的影像
```

**為什麼裁切要提前。** 以前裁切在最後一步，所有 op 都跑在 1767² 上。但
`vignette` 的 σ、`distortion` 的正規化座標、`jpeg` 的區塊都是從「手上這個畫面」
算出來的 —— 跑在過大的畫面上，等於用錯誤的基準。實際後果：輸出只看到暈影曲線
的中央 53%，所以實際暈影比 `vignette_range` 說的弱。

裁切提前同時讓 11 個 op 少處理 2.12 倍的像素，但那是附帶效果，不是動機。

**`field_mask` 已移除。** 它的圓半徑是 `min(w,h)//2 = 883`，而 1440×1024 輸出
的半對角線是 883.48 —— 正方形的內切圓恆等於矩形的外接圓，因為正方形的邊長就是
矩形的對角線。所以它畫的圓正好把整個輸出框住，裁切之後貢獻 0 個像素。真實照片
的視野本來就是矩形的。

**`SENSOR_MARGIN = 64`，這個數字是推導出來的。** `defocus` 讀 `radius` px、
`chromatic` 讀 `shift` px（兩者預設都是 2，可忽略）。真正決定 margin 的是
`distortion`：枕形（k1 < 0）時 `remap` 往畫面外取樣，而 `src_x` 被 `np.clip`
夾在畫面內，所以畫面太窄會把自己的角落抹開。

輸出角落到中心的距離**與 margin 無關**，恆為 (719.5, 511.5) —— margin 移動的是
中心，不是角落。不被夾的條件是

```
719.5 / factor + cx ≤ 2·cx      即   factor ≥ 719.5 / (719.5 + M)
factor = 1 + k1·r2，最壞 k1 = -0.04（distortion_k1_range 的下界）
```

| M | factor | 需要 ≥ | x 餘裕 | 畫面 |
|---|---|---|---|---|
| 0 | 0.9200 | 1.0000 | −62.6 px | 1440×1024，1.47 Mpx |
| 32 | 0.9279 | 0.9574 | −23.9 px | 1504×1088，1.64 Mpx |
| 56 | 0.9331 | 0.9278 | +4.4 px | 1552×1136，1.76 Mpx |
| **64** | **0.9347** | **0.9183** | **+13.7 px** | **1568×1152，1.81 Mpx** |

舊順序從來不會踩到：畸變跑在 1767² 上，輸出角落的取樣點落在 1632，畫面邊界是
1766，完全沒被夾。**是「裁切提前」讓 margin 變成承重結構的**，所以它必須撐得住
正方形以前免費吸收掉的同一個最壞情況。
