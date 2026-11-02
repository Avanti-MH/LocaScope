# GigaPath／Stage 2 Inference 加速

**狀態（2026-10-26）。** 這份文件原本是 2026-07 的「怎麼加速 GigaPath encode」清單（起點：一張 WSI 的 retrieval 約 30 分鐘，瓶頸是 GigaPath ViT-g 對所有 tissue patch 做 encode）。之後真正讓時間下降的，除了 fp16，是**讀圖**（reader 統一與整塊讀取）、**CPU 預算**、**不要重算的快取**；原清單裡的 ToMe、flash-attn、`torch.compile` 沒有帶來實質的收益。所以下面先寫「現在是什麼樣」和「怎麼演進到這裡」，原本的逐項說明保留在後面（每項加了一行現況）。

數字的出處：`log/TODO.log`、`log/MILESTONE.log`、`ARCHITECTURE.md`（讀取與加速一節）、`utilities/SlideReader.py` 的 docstring，以及 2026-10 訓練 job 的 log。標「（量級）」的是不同 job、不同資源下的對照，只能看量級；標「（推測）」的是還沒用資料確認。

---

## 現況總表

| 項目 | 現況 | 量到的效果 | 在哪裡 |
|---|---|---|---|
| fp16 autocast | **採用**（所有 production encoder `dtype='fp16'`） | GigaPath 5.4–5.5×（H200，bs ≥ 16）；cos = 0.99995；建 feature 67,348 tiles 97 s = 694 tiles/s，fp32 為 97 tiles/s（約 7×，job 374401） | `encoder_config(...).with_model(dtype='fp16')` |
| batch size | **採用**：預設 32 → 128；整層 encode 用 1024（`RealTest.sh` 的 `STAGE2_BATCH`） | 吞吐在 bs 128 以上飽和（fp32 97 p/s） | `GigaPathFunc.gigapath_encode` |
| `torch.compile` | 保留成選項（`TileEncoder._set_model(compile=)`），**沒有帶來收益** | fp32 單獨 1.0–1.1×，bs 大時幾乎為 1 | `TileEncoder` |
| Flash Attention | 環境裝了（flash-attn 2.5.8），**沒有收益** | H200 上 1.00–1.01× | — |
| ToMe | **否決，程式碼已刪**  | 見第 6 節 | — |
| 讀圖：整塊讀取 + worker | **採用**（`SlideReader.read_grid`，2026-10-03 統一） | 一個 tile 讀一次約 270 tiles/s → 整塊讀再切加 worker 約 1,060 tiles/s；Ki67／BRACS L0 約 3,570 tiles/s | `utilities/SlideReader.py` |
| CPU 預算 | **採用** | thread 數超過 CPU 數：bench 慢 7 倍、訓練 encode 慢 3 倍（36 → 251 tiles/s 每 process） | `CpuBudget` |
| uint8 進 encoder、GPU 前處理、結果留 GPU | **採用** | 訓練輸入約 25 → 151 tiles/s | `TileEncoder`、`GridBlock` |
| 多 GPU | **採用**（`DataParallel`，gigapath／uni2 再加 `compile`） | 約 N×（量級） | `TileEncoder._set_model` |
| Feature cache | **採用**（`FeatureStore`，內容雜湊 id）；訓練用記憶體版 `MemoryFeatureStore`（`--keep-levels`） | 重跑時整個 encode 省掉；**讀圖省不掉** | `utilities/Store.py`（`FeatureStore`）、`training/GloRet/Supply.py` |
| 其餘（INT8、TensorRT、Dynamic ViT／EViT、pruning、distillation） | **沒做** | — | — |

---

## 演進（歷史）

1. **2026-07-21／22　encode 本身。** 加 `dtype` + autocast、normalize 前 `.float()`、`batch_size` 32 → 128、`torch.compile` 封裝、ToMe 封裝；裝 flash-attn 2.5.8。`BenchMarkV2`（H200，n = 40,960 個合成 patch）：

   | config \ bs | 8 | 16 | 128 | 1K | 4K |
   |---|---|---|---|---|---|
   | baseline fp32（patch/s） | 75 | 81 | 97 | 97 | 97 |
   | fp16 | 4.74× | 5.48× | 5.45× | 5.48× | 5.42× |
   | flash-attn（fp32） | 1.00× | 1.01× | 1.01× | 1.01× | 1.01× |
   | ToMe r=8（fp32） | 2.39× | 2.62× | 2.70× | 2.81× | 2.82× |
   | compile（fp32） | 1.11× | 1.10× | 1.03× | 1.02× | 1.01× |
   | fp16 + flash | 4.68× | 5.37× | 5.47× | 5.50× | 5.43× |
   | 全開（fp16 + flash + ToMe + compile） | 7.94× | 8.93× | 9.06× | 9.38× | 9.21× |

   fp16 是最大單一貢獻；flash-attn 在 H200 沒感；compile 小。**但這張表裡有 ToMe 的列是假的**：2026-07-23 的精度檢查（`MILESTONE.log` M3）發現當時 ToMe 的輸出和 baseline 的 cos 只有 0.015（隨機），所以「9.4×」其實包含一個壞掉的分支。

2. **2026-07-23　ToMe 查到底。** 原因是 upstream tome 的 `ToMeBlock.forward` 略過 `ls1` / `ls2`（LayerScale），GigaPath 的 residual 分支因此放大約 2000 倍。補上之後 r=8 的 cos 回到 0.779（損失約 22%）。ToMe 的決議仍是**出局**（第 6 節記錄的理由：top-5 retrieval overlap 0.26、與 fp16 併用 NaN），實作已刪除（等效測試於 2026-10-05 移除）。

3. **2026-10-02／03　CPU 與讀圖才是大頭。**
   - **CpuBudget**：torch 預設把 thread pool 開到看得見的所有 CPU，8 CPU 上兩個 shard 跑 16 thread，每個 process 只有 36 tiles/s（單跑 251）；訓練 8 thread 配 8 個 render worker，encode 從 10.6 s 變 33 s。依 CPU 切份之後每個 shard 4 thread 251 tiles/s，訓練 1 thread 配 8 worker 151 tiles/s。
   - **讀取統一**（`SlideReader` 讀、`Render` 畫；六個讀取模組變兩個）：選層與像素數各只有一個定義。`read_grid` 整塊讀取再切：一個 tile 讀一次約 270 tiles/s → 整塊讀 + worker 約 1,060 tiles/s。非整數 ds（BRACS 4.00014）為了次像素相位一致，每個 region 只能一次讀完，所以單一大 region 只能 300 tiles/s 左右（BRACS_1228 L1 313 tiles/s，L0 3,780）。
   - 量到的前後（不同資源，量級）：window bench 每 process 41.7 → ~410 tiles/s（window 分數差值 0）；pipeline 的 level build（BRACS_310 L0，98k tiles）362 → 118 s；BRACS_1952 L0 74,853 個 tile 6768 → 93 s、L1 4,511 個 448 → 21 s；訓練輸入（含 encode）99.5／25.3 → 151 tiles/s。每項都用逐像素比對保證結果不變（566 次讀取逐像素相同）。

4. **2026-10-06　pipeline 的每張照片 ~60 s 拆開。** BRACS_1228 L0 三張照片 profile：時間幾乎全在 `compute_sim_maps`，每次都重排整層特徵網格（4 個旋轉就 4 次），而且排法是 tile 數的平方（`PatchGrid.__len__` 逐列加總）。改成 O(1) 的封閉式、特徵留在 GPU（`TileEncoder` 的出口不再 `.cpu()`）、每層網格建一次。第一張照片多出的 570 s 是建 L0 特徵：`read_region` 248 s（`read_workers` 為 0，主程序串行）+ 編碼 261 s。改後 retrieval 796.4 → 690.8 s（`LocaScopeStagesTest`，只有一張照片，主要是建 L0 特徵，每張照片省的量不到）。同日：DataParallel 蓋在 `self.model` 上造成 encoder id 隨 GPU 數變（4 卡寫的 cache，1 卡 miss 並重新編碼 314 s），改成 `self.model` 永遠是網路本身、`runner` 才是怎麼跑。

5. **2026-10-07　讀像素一律走 `SlideReader.read`。** 其餘各處自己叫 `read_region_rgb` 的全部收進來（洞不再被 `.convert('RGB')` 悄悄變黑）；`read_points` 取代 PCA 切割的逐 tile 讀。

6. **2026-10-22 到 26　GloRet／LogLoRet 的訓練與 inference 路徑（`training/GloRet`）。**
   - 一個 level 的 encode 約 670 tiles/s，一個 epoch 的時間由它決定，所以訓練以 **visit**（一個 slide 的一個 level）為單位：level 編碼一次、多個 head 共用，切換 level 時丟掉；`--keep-levels` 把編碼好的 level 留在 host 記憶體，之後的 epoch 讀回而不重編。
   - **Gradient Cache**（Gao et al. 2021）：整層（29 萬到 38 萬個窗）的 softmax 不能保留 activation，窗向量先不帶圖算、對向量求梯度、再分塊帶圖回傳。ds1 的 level：一張卡約 13–15 s/visit，GPU 峰值 16 GB；2 張一組 9–11 s；3 張一組不加 cache 在 38 萬個窗 OOM。
   - attnpool 的 patch token 每個 tile 約 0.6 MB：階段 B 用 `LazyLevel`，窗被用到才讀並編碼它底下的 tile，不讀整個細層。
   - **讀取 worker 與 `/dev/shm`**（2026-10-24）：16 張卡（4 節點各 4 卡）、每個 process 11 個讀取 worker，CPU 平均只用 15–23%（peak 26–51%），但 worker 交接 tile 用的 `/dev/shm` 滿到 ENOSPC，一個 block 丟失；`DataLoader` 沒有 timeout，主程式永遠等不到它，其他 15 個 rank 在 NCCL 等了 180 分鐘（job 605745）。對策：`DataLoader(timeout=600)`（`SlideReader.READ_TIMEOUT_S`），`RerankHead.sh` 的 `READ_WORKERS` 預設 5。
   - 檢索端的 inference：GloRet 的 window 向量 $k$ 與 query 無關，同一 slide、level、window 形狀可 cache；目前只有 exact 搜尋，`hnsw`／`ivf` 還沒實作（`RetrieverGloRet.run_wsi` 有 TODO）。

---

## 方法總覽

| 方法 | 預估加速（2026-07 的估計） | 難度 | 需改模型 | 現況 |
|---|---|---|---|---|
| fp16 autocast | ~2× | ★ | 否 | **採用**，實測 5.4–5.5× |
| 加大 batch size | 1.2–2× | ★ | 否 | **採用**（128；整層 1024） |
| `torch.compile` | 10–30% | ★ | 否 | 選項，實測 0–10% |
| DataLoader preprocessing | 5–20% | ★★ | 否 | **採用，而且是大頭**：讀圖 worker 與整塊讀取，270 → 1,060 tiles/s |
| Flash Attention | 20–50% | ★★ | 否 | 無收益（1.00–1.01×） |
| Token Merging (ToMe) | 30–50% | ★★ | 微調 | **否決**，已刪 |
| Multi-GPU DataParallel | ~N× | ★★ | 否 | **採用** |
| Feature cache | 99%+ (重跑) | ★★ | 否 | **採用**（重跑省 encode，省不掉讀圖） |
| INT8 quantization | ~2× | ★★★ | 否 | 沒做 |
| TensorRT | 3–5× | ★★★ | 否 | 沒做 |
| Dynamic ViT / EViT | 30–50% | ★★★★ | 需 fine-tune | 沒做 |
| Pruning | 1.5–3× | ★★★★ | 需 fine-tune | 沒做 |
| Knowledge distillation | 5–10× | ★★★★ | 重訓 | 沒做 |

以下各節是 2026-07 的原始說明，只在標題後加了一行「現況」。

---

## 1. fp16 autocast（立即可做）

> **現況：採用。** GigaPath 5.4–5.5×（H200，bs ≥ 16）；建 feature 694 tiles/s（fp16）對 97 tiles/s（fp32）。所有 production encoder 都是 `dtype='fp16'`。

GigaPath 特徵不需要 fp32 精度，H100 的 Tensor Core 在 fp16 下約快 2×。

```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    feats = model(batch)
feats = F.normalize(feats.float(), dim=-1)   # normalize 回 fp32
```

**注意**：L2 normalize 前先轉回 fp32 避免精度問題。

---

## 2. 加大 batch size

> **現況：採用。** 預設 128；整層 encode 用 1024（`RealTest.sh` 的 `STAGE2_BATCH`）。吞吐在 128 以上飽和。

H100 有 80GB HBM，可以大幅提高 batch size。  
吞吐量通常在 batch 128–512 之間飽和。

---

## 3. `torch.compile`

> **現況：保留成選項，沒有收益。** 實測 fp32 下 1.0–1.1×；`TileEncoder._set_model(compile=)` 是唯一把網路包起來的地方。

PyTorch 2.0+ 的 graph compile，fuse ops，H100 上對 ViT 有明顯幫助。

```python
model = torch.compile(model, mode='reduce-overhead')
```

第一次 forward 需要幾分鐘 compile，之後每次 call 都用 compiled graph。  
**建議**：搭配 fp16 一起用，效果更好。

---

## 4. DataLoader preprocessing

> **現況：採用，而且比預期重要。** 後來的瓶頸是**讀圖**，不是 transform：`SlideReader.read_grid` 整塊讀取 + DataLoader worker（270 → 1,060 tiles/s），encoder 直接吃 uint8、GPU 前處理（訓練輸入 25 → 151 tiles/s）。見上面「演進」第 3、4 項。worker 太多會把 `/dev/shm` 撐滿，並要設 `DataLoader` timeout（演進第 6 項）。

目前 CPU transform（resize → crop → normalize）和 GPU forward 是串行的。  
用 DataLoader 的 worker 讓兩者重疊：

```python
loader = DataLoader(
    PatchDataset(patches, transform),
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
for batch in loader:
    batch = batch.to(device, non_blocking=True)
    feats = model(batch)
```

---

## 5. Flash Attention

> **現況：沒有收益。** H200 上 flash-attn 單獨 1.00–1.01×（memory bandwidth 已充足）。

GigaPath (ViT-g) 的 self-attention 是主要計算量，Flash Attention 2 用 fused kernel 減少 memory bandwidth：

```python
# timm 新版支援，載入時傳 attn_implementation
model = timm.create_model(
    'hf_hub:prov-gigapath/prov-gigapath',
    pretrained=True,
    attn_implementation='flash_attention_2',
)
```

或安裝 `flash-attn` 套件後 timm 自動偵測。  
**先確認 timm 版本支援**：`timm.__version__`

---

## 6. Token Merging (ToMe) — 已否決，程式碼已移除

> **試過，失敗，不要再試。** 決定過程與數字見 `log/MILESTONE.log` M3 和
> `log/TODO.log`：對 GigaPath（ViT-g，靠 LayerScale 穩住殘差）套 ToMe 之後
> top-5 retrieval overlap 掉到 0.26，fp16 併用時直接 NaN。同一輪測到 fp16
> 單獨用是乾淨的（cos=0.99995、5.5× 加速），所以 fp16 進生產、ToMe 出局。
>
> 還有第二個理由，是後來才看見的：ToMe 合併 token 但不動
> `patch_embed.grid_size`，所以 `model_spec` 會宣稱 197 個 token 而輸出只有
> 101 個。在 `features()` 還是 `raw[:, 0]` 的年代沒有任何東西會發現這件事。
>
> `aiNNModel/GigaPathFunc.py` 裡的實作已刪除；等效測試已於
> 2026-10-05 移除。下面這段留著，是為了記錄「當初為什麼想用」——
> `MILESTONE.log` 記的是失敗過程，不是動機。

ViT 各層合併相似 token，減少 30–50% 計算量，對病理圖影響小。

```python
import tome
tome.patch.timm(model)
model.r = 8   # 每層 merge 8 個 token，可調
```

- `r=8`：約減少 30% 計算，精度幾乎不變
- `r=16`：約減少 50% 計算，需驗證 retrieval 精度

安裝：`pip install tome`

---

## 7. Multi-GPU DataParallel

> **現況：採用。** 在 `TileEncoder._set_model(multi_gpu=)`；網路的 identity 不受 `DataParallel`／`compile` 影響（2026-10-06）。

SLURM 改 `--gpus-per-node=2`（或 4），model 自動切分 batch：

```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

---

## 8. Feature Cache

> **現況：採用。** `FeatureStore`（`utilities/Store.py`，內容雜湊 id，pipeline 與 bench 共用）；訓練用記憶體版 `MemoryFeatureStore`。省得掉 encode，省不掉讀圖（BRACS_1228 L0：讀 278 s／編碼 285 s），所以讀圖要另外加速。

同一張 WSI 在不同 query 下只需 encode 一次。  
將 `WsiRegionsFeatures` 序列化到磁碟，下次直接 load：

```python
cache_path = Path(f'cache/{wsi_stem}_level{level}.pt')
if cache_path.exists():
    features = torch.load(cache_path)
else:
    features = [tp.to_features(encoder) for tp in wtc]
    torch.save(features, cache_path)
```

對「多個 query 搜同一張 WSI」的場景效果最大（實際上是 100% 省去 encode 時間）。

---

## 9. Dynamic ViT / EViT

> **現況：沒做。**

在 forward pass 中動態跳過重要性低的 token，計算量隨被跳過的比例線性下降。

- **DynamicViT**：每層用一個輕量 predictor 決定哪些 token 可以丟棄
- **EViT**：根據 [CLS] token 的 attention 分數篩選 informative token

```python
# 概念示意（需針對 GigaPath 的 ViT-g 架構改寫）
# 每層保留 top-k token，k 可設為原數量的 50–70%
```

與 ToMe 的差異：ToMe 是**合併**相似 token（無資訊損失），Dynamic ViT 是**丟棄** token（有輕微資訊損失）。兩者都需要對 GigaPath 做 fine-tune 才能維持 retrieval 精度，成本較高。

---

## 10. Pruning

> **現況：沒做。**

剪掉 ViT-g 中貢獻少的 attention head 或整個 transformer layer：

- **Head pruning**：GigaPath ViT-g 有 16 heads/layer，移除低重要性的 head
- **Layer pruning**：移除整層（ViT-g 有 40 層），直接線性減少計算量

```python
# 移除 attention head 示意
for layer in model.blocks:
    layer.attn.num_heads = 12   # 從 16 剪到 12
```

**流程**：fine-tune → measure importance → prune → fine-tune 恢復精度  
GigaPath 是 pretrained foundation model，pruning 後需要用病理資料 fine-tune，成本較高。

---

## 11. INT8 Quantization

> **現況：沒做。**

```python
import bitsandbytes as bnb
# 或 torch.ao.quantization
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

速度約 2×，記憶體減半，但需驗證 retrieval accuracy。

---

## 12. TensorRT

> **現況：沒做。**

最高加速（3–5×），但 export 流程較複雜：

```bash
pip install tensorrt
python -c "import torch; torch.onnx.export(model, dummy_input, 'gigapath.onnx')"
trtexec --onnx=gigapath.onnx --fp16 --saveEngine=gigapath.trt
```

---

## 實際走過的順序，與之後

```
2026-07  fp16 → 5.4×；bs 128；compile／flash-attn 試過沒有收益；ToMe 試過，出局
2026-10  CpuBudget（thread 不超過 CPU）→ 讀取統一與整塊讀取（270 → 1,060 tiles/s）
         → encoder 吃 uint8、結果留 GPU → feature cache（省 encode）→ 每張照片的重排與搬運
         → 訓練端：visit、Gradient Cache、lazy 讀 token、讀取 worker 與 /dev/shm 的上限
```

還可以做、尚未做（沒有量過，標〔推測〕）：

1. **GloRet 的 exact 搜尋換 `hnsw`／`ivf`**：只在窗口數百萬以上或跨多張 slide 才有意義〔推測〕；目前是 exact matmul。
2. **非整數 ds 的大 region 讀取**：單一 region 只能一次讀完（相位），BRACS L1 只有 300 tiles/s 左右；要拆就得改像素，所以這一項要先決定可不可以改像素。
3. **pipeline 改用 `read_grid`**：原型量到 362 → 118 s（BRACS_310 L0），正式 pipeline 的第一次 level build 仍是舊路徑〔以 ARCHITECTURE.md 的記載為準〕。
4. **Stage 1 參考庫與 pre-tile 平行讀取**、**PrototypicalRoutingHead 的渲染移到 DataLoader worker**（`ARCHITECTURE.md` 列為「這一輪不做」）。
5. **lazy 層的 tile 讀取與訓練重疊**：目前 attnpool 階段 B 的 tile 是在訓練 thread 內讀，以前（固定 pool）是在背景 thread 預讀；沒有實測差多少。

---

## 相關檔案

- `aiNNModel/GigaPathFunc.py` — `GigaPathEncoderConfig`, `GigaPathEncoder`（ToMe 已刪）
- `utilities/SlideReader.py` — `read`、`read_grid`、`read_points`；docstring 有 270 → 1,060 tiles/s 的來源
- `utilities/CpuBudget.py` — worker 與 thread 的分配
- `utilities/Store.py`（`FeatureStore`）、`training/GloRet/Supply.py`（`MemoryFeatureStore`）— 快取
- `aiNNModel/TileEncoderFunc.py` — `_set_model(multi_gpu=, compile=)`、結果留在 GPU
- `training/GloRet/Lazy.py`、`training/GloRet/Full.py` — lazy token 與 Gradient Cache
- `utilities/bench_modules/bench_gigapath_configs.py` — 速度與精度 benchmark
- `ARCHITECTURE.md`（讀取與加速）、`log/TODO.log`、`log/MILESTONE.log`（M3）— 數字與決策的出處
