# GigaPath Inference 加速方案

目前 retrieval pipeline 跑一張 WSI 約 **30 分鐘**，瓶頸在 GigaPath (ViT-g, ~1B params) 對所有 tissue patches 做 encode。

---

## 方法總覽

| 方法 | 預估加速 | 難度 | 需改模型 |
|---|---|---|---|
| fp16 autocast | ~2× | ★ | 否 |
| 加大 batch size | 1.2–2× | ★ | 否 |
| `torch.compile` | 10–30% | ★ | 否 |
| DataLoader preprocessing | 5–20% | ★★ | 否 |
| Flash Attention | 20–50% | ★★ | 否 |
| Token Merging (ToMe) | 30–50% | ★★ | 微調 |
| Multi-GPU DataParallel | ~N× | ★★ | 否 |
| Feature cache | 99%+ (重跑) | ★★ | 否 |
| INT8 quantization | ~2× | ★★★ | 否 |
| TensorRT | 3–5× | ★★★ | 否 |
| Dynamic ViT / EViT | 30–50% | ★★★★ | 需 fine-tune |
| Pruning | 1.5–3× | ★★★★ | 需 fine-tune |
| Knowledge distillation | 5–10× | ★★★★ | 重訓 |

---

## 1. fp16 autocast（立即可做）

GigaPath 特徵不需要 fp32 精度，H100 的 Tensor Core 在 fp16 下約快 2×。

```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    feats = model(batch)
feats = F.normalize(feats.float(), dim=-1)   # normalize 回 fp32
```

**注意**：L2 normalize 前先轉回 fp32 避免精度問題。

---

## 2. 加大 batch size

H100 有 80GB HBM，可以大幅提高 batch size。  
吞吐量通常在 batch 128–512 之間飽和，用 `bench_gigapath_infer.py` 找最優值。

---

## 3. `torch.compile`

PyTorch 2.0+ 的 graph compile，fuse ops，H100 上對 ViT 有明顯幫助。

```python
model = torch.compile(model, mode='reduce-overhead')
```

第一次 forward 需要幾分鐘 compile，之後每次 call 都用 compiled graph。  
**建議**：搭配 fp16 一起用，效果更好。

---

## 4. DataLoader preprocessing

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

**先跑 `bench_gigapath_infer.py` Part 0** 確認 CPU/GPU ratio，若 ratio < 0.3 則跳過此項。

---

## 5. Flash Attention

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

## 6. Token Merging (ToMe)

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

SLURM 改 `--gpus-per-node=2`（或 4），model 自動切分 batch：

```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

`bench_gigapath_infer.py` 已自動偵測並啟用。

---

## 8. Feature Cache

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

```python
import bitsandbytes as bnb
# 或 torch.ao.quantization
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

速度約 2×，記憶體減半，但需驗證 retrieval accuracy。

---

## 12. TensorRT

最高加速（3–5×），但 export 流程較複雜：

```bash
pip install tensorrt
python -c "import torch; torch.onnx.export(model, dummy_input, 'gigapath.onnx')"
trtexec --onnx=gigapath.onnx --fp16 --saveEngine=gigapath.trt
```

---

## 建議執行順序

```
1. 先跑 bench_gigapath_infer.py 找最優 batch size
2. 加 fp16 autocast               → 目標：15 min
3. 加 torch.compile               → 目標：12 min
4. 看 Part 0 bottleneck ratio，決定要不要加 DataLoader
5. 試 ToMe r=8                    → 目標：8 min
6. 多 GPU（申請 2 GPU）            → 目標：4 min
7. 實作 feature cache             → 多 query 場景免費
```

---

## 相關檔案

- `aiNNModel/GigaPathFunc.py` — `gigapath_encode`, `gigapath_model`
- `utilities/test_modules/bench_gigapath_infer.py` — 速度 benchmark
- `2_retrieval/GigaPathSlideWinSim.py` — encoder 傳遞點
