# query_sim — Microscope FOV Simulator

> **2026-11-02：這份是 M4 當初的整合設計，下面只留現在還有對應程式的部分。** `region_type` 被 M4.2 實測否決，見 `log/MILESTONE.log` 與 `log/TODO.log`。
> 讀取是 `locascope/core/slides/SlideReader`；`camera.py` 的 `Render` 只負責效果和 GT，見專案根目錄的 `ARCHITECTURE.md`。
>
> **現在 `query_sim/` 實際有的**：`config.py`（`DomainGapConfig`）、`record.py`（`FOVRecord`）、`augment/{color,field,geometry,lens,noise}.py`、
> `pipeline.py`（`simulate_with_gt`、`lens_margin`、`read_reach`）、`camera.py`（`Render`、`render_spec`、`read_margin`、`photo_rng`）、
> `FovSupply.py`（`FovSupply`、`FOV_RECIPES`），沒有 `__init__.py`。批量生成是 `FovSupply` 加 `locascope/cli/bench`，單張 demo 是
> `locascope/cli/demo/show_camera_effects.py`。

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

    # Rotation
    rotation_choices: Tuple[int, ...] = (0, 90, 180, 270)
    angle_jitter_deg: float = 3.0

    # Scale
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

## Rotation 特別處理（連動 retrieval TODO）

`geometry.py` 的 `apply_rotation(img, angle=0.0)` 繞中心轉 `angle` 度（逆時針為正），輸出維持 (H, W)；90／180／270 走 lossless `cv2.rotate`。
`FOVRecord.rot_deg` 記錄實際套用的角度。只旋轉、不套 photometric augment 的資料用 recipe 表達（例如 `FOV_RECIPES['routing-support-native']`），
給 [rotation-aware retrieval TODO](../log/TODO.log) 產 benchmark 資料；完整 gap 的資料就是 real-world dataset。

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
| 256 tile，完整 gap 幾何(`DomainGapConfig(stage_shift_max=0)`,routing head 的 tile) | 17, 15 | 363² (m = 0) |
| 1440×1024，預設 gap | 69, 49 | 1827² (m = 30) |
| 1440×1024，不轉不縮放 | 69, 49 | 1578×1162 (m = 69) |

`test_camera.py` 的 `reach` 拿最壞的一次抽樣，比較依 spec 讀的照片和讀寬 400 px
的照片，角落要一致；不加 margin 的讀取是對照組，角落必須不一致。

**`field_mask` 已移除。** 它的圓半徑是 `min(w,h)//2 = 883`，而 1440×1024 輸出
的半對角線是 883.48 —— 正方形的內切圓恆等於矩形的外接圓，因為正方形的邊長就是
矩形的對角線。所以它畫的圓正好把整個輸出框住，裁切之後貢獻 0 個像素。真實照片
的視野本來就是矩形的。
