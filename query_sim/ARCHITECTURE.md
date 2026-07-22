# query_sim — Microscope FOV Simulator（整合設計）

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
│   ├── field.py               ← vignette、field_mask、stage_shift
│   ├── lens.py                ← distortion、defocus、chromatic
│   ├── geometry.py            ← rotation (0/90/180/270 + jitter)、scale     ← 新
│   └── noise.py               ← gaussian noise
│
├── pipeline.py                ← simulate_microscope_photo(img, cfg) 串接所有 augment
│                                simulate_with_gt(cfg) → (img, FOVRecord)
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
    field_mask:      bool                = False
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
| field_mask | **query_sim** | synth_fov 沒有 |
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

## 可討論的取捨

1. **要不要 rename `query_sim/` → 更精確的名字？**
   e.g. `microscope_sim/`、`fov_sim/`（現有名稱偏「query」，容易和 retrieval 的 query 混淆）

2. **`generator.py` 用 iterator 還是 list？**
   iterator 省記憶體、可以 pipeline 串接；list 簡單直觀。batch 場景兩者都 OK。

3. **`source/wsi_query.py` 要不要跟 `PatchingLib.WsiTissuesContainer` 整合？**
   後者已有完整 WSI 讀取 + patching 邏輯，可能重複實作 WSI region 讀取。
   風險：兩邊 use case 不同（一個要 raw crop、一個要 grid patches），強行整合可能反而複雜。

4. **`--seed` 統一入口**
   `random.seed / np.random.seed / torch.manual_seed` 統一設定，方便 reproduce。
   Config 也可加 `seed: Optional[int]` 欄位。

5. **要不要支援多 WSI 混合輸出？**
   例如 `--wsi wsi1.svs wsi2.svs --n 300` 平均產出。`FOVRecord.wsi` 已支援。

---

## 相關 TODO

- `PatchingLib` crop() TODO(A/B) — sub-container 語義、overlap 對齊
- retrieval rotation-aware TODO — 需要本 package 產出旋轉 GT 資料集才能 benchmark
