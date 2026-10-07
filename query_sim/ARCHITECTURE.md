# query_sim — Microscope FOV Simulator（整合設計）

> **這是 M4 當初的整合計畫，不是現在的目錄結構。** `synth_fov_generator.py` 已
> 不存在，`source/tissue_filter.py`（`is_tissue`/`classify_region`）從未落地——
> 那條路線被 M4.2「`region_type` 被實測否決」推翻，見 `log/MILESTONE.log` 與
> `log/TODO.log`。`camera.py`（M4.1 的 Camera 抽象）也是這份文件寫完之後才加的,
> 不在下面的目錄結構裡。保留原文是因為被推翻的路線本身是負面結果,不是要重寫
> 成看起來從一開始就對；要看現在實際長什麼樣,直接看 `query_sim/` 底下的檔案。
> 2026-10-03 起 `source/`（`QueryFromWSI`）已刪除:讀取是 `utilities/SlideReader`,
> `camera.py` 的 Camera 改名 `Render` 只負責效果和 GT,見專案根目錄的 `ARCHITECTURE.md`。

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
├── pipeline.py                ← 串接所有 augment
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

## 命名說明

`query_sim` 的 **query = 整個 LocaScope 專案的 FoV Picture**（顯微鏡下拍到的一張視野）。
這裡的 query 跟 retrieval 語境的 query 是**同一件事**，不是兩個概念 → 名稱不改，
`query_sim` = 「query（FoV picture）的 simulator」，語意正確。

---

## 可討論的取捨

1. **`generator.py` 用 iterator 還是 list？**
   iterator 省記憶體、可以 pipeline 串接；list 簡單直觀。batch 場景兩者都 OK。

2. **`source/wsi_query.py` 要不要跟 `PatchingLib.WsiTissuesContainer` 整合？**
   已不成立：`source/` 於 2026-10-03 刪除，`WsiTissuesContainer` 於 2026-10-06 淘汰，
   兩邊的讀圖都改由 `SlideReader` 負責。

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
── 讀取：camera.render_spec(cfg, sensor) ─────────
   旋轉 → 外接正方形 + 2·m；否則 sensor 長方形 + 2·m
   m = camera.read_margin：這次曝光所有取樣點都必須落在讀進來的範圍內
   ↓
── 場景階段：決定什麼落到感測器上 ────────────────
   rotation                    旋轉需要畫面外的像素轉進來
   scale
   stage_shift                 載物台抖動 = 重新取景
   ↓
   裁切到 sensor + lens_margin（感測器階段會取樣到的範圍）
   ↓
── 感測器階段：光學與感測器對這張影像做了什麼 ──────
   color / brightness / color_temp
   distortion                  ┐ 讀鄰域；畸變以 sensor 半寬正規化
   defocus                     │
   chromatic                   ┘
   ↓
   裁切到精確的 sensor
   ↓
   vignette                    逐像素，要拿到精確的感測器畫面
   noise
   jpeg                        8×8 區塊對齊交付的影像
```

**每一個框都量自 sensor。** `vignette` 在裁到 sensor 之後才跑；`distortion` 以
sensor 的半寬正規化（`augment.lens.apply_distortion(..., sensor=)`），所以 k1 =
-0.04 就是 sensor 角落 4% 的形變，不論框外多讀了多少。

**margin 是推導出來的，不是常數。** `pipeline.lens_margin(cfg, sensor)` 從 sensor
角落往回走感測器階段：色差、失焦，再以最往外的 k1 做畸變，每次重取樣加 1 px。
`pipeline.read_reach(cfg, sensor, rotation)` 接著往回走場景階段：stage shift、最小
的 scale、cfg 會抽到的每個角度。`camera.read_margin` 把它換成讀取要多讀的 px：

| 相機 | lens margin (x, y) | 讀取 |
|---|---|---|
| 256 tile，CAMERA_FULL 幾何 | 17, 15 | 363² (m = 0) |
| 1440×1024，預設 gap | 69, 49 | 1827² (m = 30) |
| 1440×1024，不轉不縮放 | 69, 49 | 1578×1162 (m = 69) |

`test_camera.py` 的 `reach` 拿最壞的一次抽樣，比較依 spec 讀的照片和讀寬 400 px
的照片，角落要一致；不加 margin 的讀取是對照組，角落必須不一致。

**`field_mask` 已移除。** 它的圓半徑是 `min(w,h)//2 = 883`，而 1440×1024 輸出
的半對角線是 883.48 —— 正方形的內切圓恆等於矩形的外接圓，因為正方形的邊長就是
矩形的對角線。所以它畫的圓正好把整個輸出框住，裁切之後貢獻 0 個像素。真實照片
的視野本來就是矩形的。
