# MppRoutingHead — spec

Trains a classifier head that routes an unknown tile/photo to the DsLadder
rung (equivalently, the mpp) it was taken at. Sits next to `KnnEstMpp`
(stage 1's production estimator) as a candidate that has to beat it on the
stage 1 bench (`bench_stage1_mpp.py`, `analyze_stage1_metrics.py`) before it
can replace it.

Ultimate goal (not this package): a model that, given a handful of prototype
tiles each labelled with their own mpp/ds, tells an unknown tile which
prototype it is closest to in SCALE and can hand back an mpp/ds number, not
just a class id — Prototypical / Relation / Siamese-networks territory. That
line is not started as code; the design is being worked out in discussion
with AI (2026-09-18: how a prototype is generated -- per-WSI support tiles
through a learned embedding, averaged per rung -- and how it is used --
nearest-prototype classification, `StageInterface.MppEstimator`-shaped so it
can sit beside `KnnEstMpp`/`ClassifierEstMpp` on the same `stage1_compare`
scorecard). This package is the classifier-head baseline that any of those
has to beat, not a step towards them.

## 詞彙

| 中文 | English | 定義 |
|---|---|---|
| 區塊輸出 | block output | encoder 第 k 個 transformer block 的輸出 `[N, T, D]`，還沒經過模型最後的 LayerNorm |
| 終端 norm | final norm | 模型最後那個 LayerNorm（timm 的 `model.norm`） |
| 層 token | layer tokens | 區塊輸出再經過終端 norm，prefix token 在前。最後一個 block 的層 token 就是 `TileEncoder.tokens()` |
| 描述子 | descriptor | 層 token 縮成的一個向量：CLS、`cls_avg`、attention pool 的輸出 |
| L2 單位化 | L2 normalise | 把向量長度縮放成 1。`tokens()` 沒有做；`fixed` 對 CLS 做 |
| 縮減方式 | reduction | head 把 token 縮成一個向量的方法，`common.Head.REDUCTIONS`：`fixed`、`attn`、`clsattn` |
| 固定縮減 | fixed | 最後一層的 CLS（CNN 為空間格平均），L2 單位化；沒有可學參數 |
| 注意力池化 | attn | 丟掉 prefix token，對 patch 做一個可學 query 的 attention pool（`AttentionPoolHead`） |
| CLS 加注意力池化 | clsattn | 最後一層 CLS 與 attn 的輸出各自 L2 單位化後串接，再接 LayerNorm；分類器輸入為 `2 × D`。需要 CLS，只用於 ViT |
| encoder 層 | `encoder_layers` | head 取哪幾個 block 的層 token。整數是絕對位置，從 0 開始、可為負（同 timm 的 `indices`）；小數是相對位置 (0, 1]，換成 block `round(r × 深度) − 1`。一律寫成 list / tuple。空的 `()` 是只用最後一層，即這個欄位出現以前的行為 |
| 層權重 | layer weights | 混合多層時每層一個可學純量，softmax 後加權；每層先過自己的 LayerNorm |
| 多層混合 | `mix_` | head 名稱前綴：`encoder_layers` 為 `MIX_LAYERS = (0.25, 0.5, 0.75, 1.0)`，UNI2 為 block 5/11/17/23，GigaPath 為 9/19/29/39。只配 `attn` / `clsattn`，只用於 ViT |
| 讀取模式 | `read_level` | 訓練 tile 從金字塔哪一層讀：`pyramid`（最近層規則，有原生層就讀原生層，否則讀較細一層再縮小；這個欄位出現以前唯一的行為）、`resampled`（從比 rung 細的層中均勻抽一層，讀進來再縮小）、`mixed`（`resampled_share` 比例走 resampled，其餘走 pyramid）。只用於訓練；val、test 一律 `pyramid` |
| 縮小來源 | `resample_from` | `finer`：任何比 rung 細的層；`l0`：只有 level 0 |
| 最大縮小倍數 | `max_resample_factor` | 候選層需要的縮小倍數（rung ÷ 該層 ds）超過它就不列入；控制讀取成本（讀取邊長 = tile × 倍數）。預設不設 |
| 重新取樣比例 | `resampled_share` | 只用於 `mixed`：走 resampled 的比例，預設 0.5 |
| 讀取模式標籤 | read tag | 讀取模式寫成一個字串，用在權重與 resume 檔名、wandb run 名稱、CSV 的 `read_level` 欄、評估標籤：`resampled-finer`、`resampled-finer-x8`、`mixed-l0-p0.3`。預設模式為空字串，CSV 寫 `pyramid`。四個預設值固定不改，舊的檔名與 resume 檔才繼續代表 pyramid |
| 原生 / 重新取樣（tile） | native / resampled (tile) | 一張 tile 是否直接讀自目標 mpp 的金字塔層（`QueryFromWSI.reads_natively`），CSV 的 `n_native`、`level_accuracy_resampled`、`native` 欄。與 Camera 風格參數 `native`（'routing-support-native' recipe）無關，也與讀取模式不同：讀取模式描述模型怎麼訓練，這個描述一張 tile 怎麼讀出來 |
| head 名稱 | head name | `[mix_]` + `[attn_ / clsattn_ / 無]` + 分類器名；沒有前綴即 `fixed`、只用最後一層。名稱只是 `HEAD_CHOICES` 的 key，`Runtime.head_parts` 從登記內容讀回各部分 |

## Task

6-way classification over `DsLadder.DEFAULT_RUNGS = (1, 2, 4, 8, 16, 32)`.
Not mpp regression — argmax already IS the answer (`rung * wsi.base_mpp`).
Not ordinal regression — that is on the deferred list with Prototypical /
Relation / Siamese, for the same "still reading" reason.

## Two baselines

| | Encoder | Trainable |
|---|---|---|
| **baseline 2** | GigaPath or UNI2 (`aiNNModel/{GigaPathFunc,Uni2Func}.py`), CLS pooling | **frozen** |
| **baseline 3** | ConvNeXt V2 Tiny (`aiNNModel/ConvNeXtV2Func.py`, NEW), `convnextv2_tiny.fcmae_ft_in22k_in1k` init | **fine-tuned end-to-end** |

No subspace projection on either arm — raw encoder features straight into the
head: ranking-by-correlation subspaces lose to the full space, so there is
nothing to gain from adding that step here.

### Head candidates (settled 2026-09-15)

| # | Head | Baseline | Trained? | Status / why |
|---|---|---|---|---|
| 2-1 | Linear (on frozen features) | 2 | head only | **main line** — matches the shape ConvNeXt V2's own released head reduces to (`NormMlpClassifierHead` with `hidden_size` unset, i.e. Norm + one Linear, no hidden layer) and the standard linear-probe convention (SimCLR/CLIP) for measuring representation quality without letting head capacity mask it |
| 2-2 | NCM (nearest class mean, cosine) | 2 | no training — one averaging pass | free second arm, floor for "how separable are the frozen features already" |
| 2-3 | ArcFace-margin head (`ArcFaceHead`, arm `arcface`) | 2 | head only | **implemented.** Weight rows and features both L2-normalised, so each row is a learned prototype direction and the logit is `s*cos(theta)` — the SAME geometry as 2-2, with the prototypes learned rather than averaged, which is what makes 2-2/2-3 a pair worth running together. ArcFace over CosFace was the user's call; `s=64`, `m=0.5`, the paper's own values. `s` and `m` are CONSTANTS, not parameters: the margin's only effect is to raise the training loss (`dL/dm > 0` everywhere), so learning it drives it to zero unless a counter-reward term is added whose weight is then the hyperparameter (AdaptiveFace) — and `s`, having no such contradiction, instead drifts up until it saturates. Sweep them, do not learn them. Caveat to read the result with: on baseline 2 the encoder is frozen, so the margin cannot compact the feature clusters the way it does in ArcFace's own setting — what remains is the re-weighting of samples near a boundary, which is aimed at this task's actual failure mode (adjacent rungs on a 2x pyramid, CLAUDE.md "Pyramid spacing"). A `3-4` twin on the fine-tuned trunk, where the margin COULD reshape features, is the obvious follow-up and is deliberately not implemented yet |
| 2-4 | Gaussian / Mahalanobis classifier | 2 | fits mean+covariance, no deep training | named, not yet detailed — NCM's covariance-aware generalisation |
| 2-5 | MLP (Linear→GELU→Dropout→Linear) on frozen features | 2 | head only | **observation arm**, parallel to 3-2 — same question (does extra head capacity help or hurt), asked on the frozen-feature side instead of the fine-tuned side |
| 3-1 | trunk (fine-tuned) → Linear | 3 | trunk + head | **main line** — matches the checkpoint's own validated head shape; more head capacity does not help bridge the natural→pathology domain gap (that is what fine-tuning the trunk itself already does) and this dataset's scale (10-50k tiles fine-tuning a 28M-param trunk, not training it from scratch) argues for fewer trainable head params, not more |
| 3-2 | trunk (fine-tuned) → MLP → Linear | 3 | trunk + head | **observation arm**, not a default — exists to empirically check the 3-1 reasoning rather than assert it: does an extra projection layer actually hurt cross-dataset generalisation (train=ki67_pure, eval=bracs/test+ki67_with_photo) the way the capacity argument predicts, or not |
| 2-6 / 3-3 | AttentionPoolHead (learnable pooling over the UN-pooled grid, CLS excluded) → LinearHead | 2 and 3 | head only (2) / trunk+head (3) | **genuine learnable pooling**, not just added capacity — takes `[N, L, D]` (L = 196 patch tokens for GigaPath/UNI2 with the CLS token excluded, or 49 spatial cells for ConvNeXt V2's 7x7 map), learns one query that attends over the L positions to produce `[N, D]`, then classifies with the SAME `LinearHead` class 2-1/3-1 use. CLS excluded on purpose: CLS is already a pooled summary the encoder's OWN pretraining trained it to be, and mixing it into a pooler meant to learn a TASK-specific aggregation would muddy which one the comparison is actually testing. Parameter count depends on `in_dim`/`n_head`, not on L — unlike flattening the grid into one giant Linear, which for GigaPath's 14x14x1536 grid would be a ~231M-parameter first layer alone (measured 2026-09-15), several orders above what a 10-50k-tile fine-tuning set can support without overfitting |
| 2-7 | AttentionPoolHead → MLP depth 2 (`attn_mlp_deep`, `attn_mlp_deep_wide`) | 2 and 3 | head only (2) / trunk+head (3) | the pool under the classifier that leads on the CLS view. `attn_linear` alone could not separate "the pool does not help" from "the pool was paired with the weakest classifier" |
| 2-8 | CLS + AttentionPoolHead → MLP depth 2 (`clsattn_mlp_deep`) | 2 | head only | keeps the encoder's own summary and adds a task-specific one, so it should not lose to `mlp_deep` unless the two halves are scaled wrongly (each is L2-normalised before the concatenation for that reason) |
| 2-9 | multi-layer mix → `attn` / `clsattn` → MLP depth 2 (`mix_attn_mlp_deep`, `mix_clsattn_mlp_deep`) | 2 | head only | asks whether scale is better read from earlier blocks, which keep more texture than the last one. The learned layer weights are themselves a result |
| — | Prototypical Networks | independent | yes (episodic) | **deferred — design in discussion with AI, not yet code** |
| — | Relation Networks | independent | yes | **deferred**, same reason |
| — | Siamese / Triplet | independent | yes | **deferred**, same reason |
| — | Ordinal regression head | stacks on any encoder | yes | **deferred**, same reason |

### Ordinal-aware loss (`--loss bal|ord_a|ord_b`)

Motivated by `test_scores_best.csv`'s own native/resampled split on bracs/test
(BRACS's 4x pyramid: rungs 1/4/16 native, 2/8/32 resampled). The gap is
encoder-dependent, not universal:

| weights | native | resampled | gap |
|---|---|---|---|
| `uni2_frozen_mlp_wide_best.pt` | 0.667 | 0.669 | 0.002 |
| `uni2_frozen_arcface_best.pt` | 0.411 | 0.414 | 0.003 |
| `gigapath_frozen_mlp_best.pt` | 0.624 | 0.388 | 0.236 |
| `gigapath_frozen_linear_best.pt` | 0.603 | 0.279 | 0.324 |
| `gigapath_frozen_arcface_best.pt` | 0.581 | 0.169 | 0.412 |

Training is 100% `ki67_pure`, whose 2x pyramid makes every rung native — so
no run has ever backpropagated through a resampled tile. A per-rung
breakdown of `test_predictions_best.csv` on bracs/test rules out the
simplest alternative explanation (some rungs are just intrinsically harder,
independent of native/resampled status): gigapath's native rungs {1,4,16}
beat their resampled counterparts {2,8,32} at every one of the three ranks
(0.692>0.333, 0.687>0.215, 0.454>0.411 for `linear`; 0.892>0.217,
0.390>0.122, 0.523>0.263 for `arcface`) — a consistent ordering, not one
rung dragging an average. uni2's `mlp_wide` shows no such ordering (its
resampled rung 8 outscores its own native rung 4 and 16). So the gap is real
for gigapath and not an artifact of which specific rungs happen to be
native on BRACS — the open question is WHY a model that never saw a
resampled tile in training is nonetheless vulnerable to one at test time:
the working hypothesis is that gigapath's frozen features let a head key
off fine-texture/sharpness as a correlate of scale, which genuinely does
track scale on Ki67's native reads but breaks under BRACS's LANCZOS
resampling — a shortcut, not a memorised artifact. Unverified; the
ordinal-aware loss below does not depend on this explanation being right,
it only depends on rung 1..32 being ordered, which they are regardless.

**Notation.** `K=6`, rungs `r_1..r_K = DsLadder.DEFAULT_RUNGS = (1,2,4,8,16,32)`,
`ℓ_c = log2(r_c) = (0,1,2,3,4,5)` (log-space, matching
`analyze_stage1_metrics.nearest_rung`'s own reasoning: pyramid scales are
geometric, so linear rung-index distance is not the right metric either).
`z` = one sample's `K` logits, `p_c = softmax(z)_c`, `y` = true class index,
`w_c = N / (K · n_c)` (`Datasets.class_weights`'s existing balanced formula,
`0` where `n_c=0`).

**1. Current loss** (`cli/train.py:477`, `:611` — plain, unweighted case):

```
L_CE(z, y) = -log p_y
```

**2. + class imbalance** (already implemented, `--class-weight balanced`,
the default): reweights by how UNDER-supplied `y`'s class is, says nothing
about which WRONG class was predicted.

```
L_bal(z, y) = -w_y · log p_y
```
batched as PyTorch's own `weight=` convention: `Σ_i w_{y_i} L_i / Σ_i w_{y_i}`
(divides by the batch's summed weight, not `N`).

**3a. + class imbalance + ordinal regression term (A, minimal diff).**
Penalises HOW FAR the prediction's own expected rung is from the true one.
Adds one soft-regression term on top of the weighted CE, and weights it the
same way (user, 2026-09-24) -- unweighted, it is a plain batch mean, and
rung 32 (0.3 per cent of the training manifest) barely moves it:

```
ℓ̂(z) = Σ_c p_c · ℓ_c                    (expected log2-rung under the softmax)
L_ord = Σ_i w_{y_i} (ℓ̂(z_i) - ℓ_{y_i})² / Σ_i w_{y_i}    (class-weighted MSE)

L = L_bal + λ · L_ord
```
`λ` is `--ordinal-weight`.

**3b. + class imbalance + soft ordinal target (B, label-smoothing version).**
Replaces the one-hot target `F.cross_entropy` uses internally with a
Gaussian kernel over log-rung distance from `y`, so a near-miss costs less
than a far one BY CONSTRUCTION rather than through an added penalty term:

```
q_c(y) = exp(-(ℓ_c - ℓ_y)² / 2σ²) / Σ_j exp(-(ℓ_j - ℓ_y)² / 2σ²)   (q_y largest, K-1 others share the rest)
L_soft(z, y) = -Σ_c q_c(y) · log p_c

L = w_y · L_soft(z, y)                  (same weighted-mean batching as 2.)
```
`σ` a new hyperparameter controlling how much mass bleeds to neighbouring
rungs (`σ→0` recovers plain one-hot CE, i.e. line 2 above, exactly).

**In code.** `cli/train.py`'s `_compute_loss` is the one switch both
baselines call: `bal` is `F.cross_entropy(weight=w)`, `ord_a` adds the
weighted regression term above (`--ordinal-weight`), `ord_b` replaces the
target with the Gaussian kernel (`--ordinal-sigma`). The loss is a segment
of every checkpoint name (`_ord_a`/`_ord_b`; `bal` has none) and part of
the `--merge` key, so the three never overwrite each other.
`PrototypicalRoutingHead/Losses.compute_loss` is the same three formulas per
episode; `test_ordinal_loss.py` holds the two to the same numbers.

**Resume** (`--resume-dir`, `aiNNModel/models/common/Resume.py`). Unset:
train from scratch, write nothing. Set: every epoch writes
`<model>_resume.pt` there -- weights, optimizer, epoch, best scores, the val
rows so far and every RNG -- and a run that finds one continues after its
epoch. A model is one encoder with every head it feeds (baseline 2, one
shared forward) or one head with its trunk (baseline 3). `--epochs` is the
total, so a finished model resumes into nothing; a file from a different
identity is refused by name. (The augmentations rendered in the DataLoader
workers are not seeded, before or after a resume.)

ConvNeXt V2 size/init tiers considered and the reasoning: see the session log
around 2026-09-14; landed on Tiny (28M params, `convnextv2_tiny.
fcmae_ft_in22k_in1k`) over Nano (15M, no `_in22k` tag available below Nano) —
final call was Tiny.

### Why `ConvNeXtV2Func.py` is a frozen-only `TileEncoder` and baseline 3 is not built on it

`TileEncoder._run` (`aiNNModel/TileEncoderFunc.py`) is `@torch.no_grad()` —
right for GigaPath/UNI2 (always frozen here) and right for ConvNeXtV2Encoder
whenever IT is used frozen (e.g. as a 4th arm next to GigaPath/UNI2: a
natural-image CNN prior instead of a pathology-ViT prior). It cannot be what
baseline 3 fine-tunes through. Baseline 3's trainable module builds its own
`nn.Module` from `timm.create_model(ConvNeXtV2EncoderConfig().model.arch,
pretrained=True, num_classes=0, global_pool='avg')` directly (see
`Models.py`), reusing the config only for the checkpoint name and the
`TransformConfig` (so preprocessing matches the frozen arm), and trains with
an ordinary grad-enabled forward pass.

## Data

**Train**: every WSI in `AccessDatasets.list_names(dataset='ki67_pure')`
(126 slides, measured 2026-09-14).
**Eval**: `list_names(dataset='bracs/test')` (87 slides) +
`list_names(dataset='ki67_with_photo')` (19 slides) — a genuine
cross-dataset generalisation test (different stain/scanner/pyramid spacing
for BRACS; same stain/scanner as train for Ki67_with_photo, which is why it
does not need its own in-domain held-out val: same distribution as train, so
it already reads as an in-domain sanity check sitting inside the eval report
rather than needing a separate carve-out from Ki67_pure).

No held-out in-domain val slice of Ki67_pure — decided explicitly (see
session log): Ki67_with_photo already plays that role.

Ki67_with_photo's real photographed FoV companions (`entry.related['photos']`)
are NOT used, on both eval legs: there is no trustworthy mpp ground truth for
them. Every leg (train AND both eval legs) is rendered through `Camera`
(`query_sim/camera.py`, CLAUDE.md's own "central abstraction" — NOT the raw
`QueryFromWSI`+`simulate_microscope_photo` pair the first draft of this
package used), so the eval numbers measure cross-DATASET generalisation, not a
photo-vs-tile domain gap on top of it. Train and eval differ in whether that
render is reproducible (see "Camera: train vs eval" below) — that difference
is deliberate, not an inconsistency.

### The camera's sensor is ONE TILE (settled 2026-09-16)

`RenderConfig.tile_size` is the camera's sensor side, the sampler's window,
and the encoder's input — one number, three roles that must agree. It renders
`Render(reader, (tile_size, tile_size), cfg, ds=rung)`, so:

| | window |
|---|---|
| `TileSampler` (via `DsLadder.plan`) | `footprint_l0 = tile_size * rung` |
| `Render` | `rect_w_l0 = output_w * ds = tile_size * rung` |

**What this replaced, and why.** The first draft rendered CLAUDE.md's
real-photo frame (1440x1024, 45:32, 1.475 MPixels) and cut it into 256 patches
with `QueryPatchContainer`. That made the two windows above disagree by
**5.6x**: `build_manifest` asked `TileSampler` to certify a `256*rung` window
as tissue, and then `Camera` read `1440*rung`. Consequences, in order of
severity:

1. Every coarse rung drew positions `Camera.capture` then refused as
   off-slide — **silently in training**, because `collate_routing_batch` drops
   a `None`. `DsLadder.reachable`'s own measured numbers say where that lands:
   footprint 8192 sampled 100/100, 16384 sampled 0/100, 32768 had no region
   that could hold it. At 1440 the ladder's top three rungs are
   11520/23040/46080, i.e. classes 3/4/5 were near-empty and the CE loss was
   being taken over a label distribution nobody had looked at. At 256 the top
   rung is 8192 — the case measured 100/100 — so **all six rungs exist**.
2. `TileSampler`'s `tissue_ratio` guarantee covered `256*rung`; the rendered
   frame covered 31.6x that area, most of it never checked.
3. Cost: 20 patches per position instead of 1. 1.51M encoder forwards per
   epoch became 75.6k; `bracs/test`'s cached token tensor went from ~126 GB to
   ~6.3 GB.

**What it is not.** Not a claim that a 256-px photograph is realistic. The
encoder never consumes a frame — a 1440x1024 photograph reaches it only as 20
separate 256 patches — so rendering the tile directly is the same input by a
shorter route. CLAUDE.md's 1440x1024 spec still governs anything that compares
against real photographs as *frames*; this package's unit of comparison is the
patch.

**The two exceptions** are the two FRAME-REFERENCED optics: the vignette and
the lens distortion. Every other op in `query_sim/augment/` is per-pixel or
local (colour, colour temperature, brightness/contrast, defocus, chromatic
shift, noise, JPEG) or a scene transform (rotation, scale) and means the same
thing at either frame size. These two are normalised to the sensor's own
half-width — as `pipeline._apply_params` states, and as the recorded episode
there shows (taking them from the oversized read left the photo seeing only
the central 53% of the falloff curve).

They are not switched off; they are made **present only sometimes**.
`DomainGapConfig` gained `vignette_p` / `distortion_p` (default `1.0` =
always, so every existing config and every already-generated corpus keeps its
exact behaviour — `pipeline._coin` does not even draw at 1.0, so the rng
sequence is byte-identical), and the 'routing-query' recipe sets both to 0.5.

Why a probability rather than a wider strength range: a real photograph's
tiles are slices of ONE vignette, and a tile from the lit centre of the frame
has none at all. Drawing strength from `(0, 0.45)` models only the darkened-
edge kind, ever more weakly; a coin models both kinds. 0.5 says a
tile is as likely to have come from the centre as from the edge — a starting
value, not a measured one.

`distortion_k2` rides on the same coin as `k1`, drawn once as `lens_on` in
`_sample_params`: k2 is a cfg constant rather than a drawn value, so deciding
it from `k1 == 0` would read the coin off a value that can legitimately be
zero on its own. And the strength is not drawn when the coin says absent, so
the RECORDED parameter is the value actually applied — the shape the
`stage_shift_dx/dy` bug got wrong.

**Left open.** `Camera` reads the rotation bounding square,
`ceil(sqrt(2)) * tile * rung`, which is wider than the window `TileSampler`
certified, so positions near a region edge still fail. That is now a rare
per-position drop rather than a whole class, and the counters report it
(`materialise_eval_corpus` prints the total). If one rung's drop rate turns
out high, the fix is `DsLadder.plan` growing a `reserve_l0` factor —
`TileSampler._reserve_fits`/`SampleMeta.margin` already honour one, the ladder
path simply never sets it.

### Camera: train wants fresh augmentation, eval wants reproducibility (2026-09-15)

`Camera`'s own module docstring already names what it is: "a different `cfg`
is a different microscope, a different `seed` is a different set of
exposures" — i.e. the augmentation IS the point, not an implementation detail
to route around. Confirmed by reading `pipeline.simulate_with_gt`:
`rng = rng or random` falls back to Python's GLOBAL `random` module when no
`rng` is given, so every un-seeded call draws genuinely fresh parameters
(rotation, defocus, chromatic shift, vignette, noise, JPEG quality).

This splits train and eval, because they want opposite things from that:

- **Train** wants FRESH augmentation every epoch — the regularising effect is
  exactly what a 10-50k-tile fine-tune, being asked to also generalise across
  datasets, needs (same reasoning that ruled out extra head capacity in
  2-5/3-2 — augmentation is the tool that actually earns its keep here).
  Caching a rendered photo (or, worse, a frozen encoder's output on it) and
  reusing it across epochs would freeze ONE random draw for the whole run,
  which is not a smaller version of augmentation — it is no augmentation,
  wearing its parameters.
- **Eval** wants the OPPOSITE: the same query must render the same photo
  every time a head gets scored, or level_accuracy/mpp_error_relative_p50
  are not comparable between runs.

Both are served by one small, additive change to `Camera` rather than two
different code paths: `capture`/`capture_with_gt` gain an optional `rng`
parameter (`capture(self, x, y, rotation=None, rng=None)`), used INSTEAD of
`self._py_rng` for that one call when given, falling back to the existing
behaviour when it is not — every one of `Camera`'s nine existing callers
(`bench_offgrid_score.py`, `bench_tile_retrieval.py`, `query_sim/
generator.py`, ... — see the session log's usage survey) is unaffected.
`RoutingHeadDataset` then reuses ONE `Camera` per WSI (cheap: the WSI handle
opens once) for both splits, and only EVAL passes a `rng` derived from the
sample's own identity (`(dataset, wsi_name, x, y, rung)`, not from worker id
or call order — a `DataLoader` may hand the same query to any worker in any
order, and the render must not depend on which).

`Camera` also has to be constructed lazily, ONE PER WORKER PROCESS, not once
in the main process and shared: `DataLoader(num_workers>0)` runs
`__getitem__` in separate processes, and an `openslide` handle opened in the
parent is not safely shared across that boundary. `Camera(wsi_or_path, ...)`
already accepts a path rather than a live handle, so this needs no change to
`Camera` itself — only to how `RoutingHeadDataset` builds one (a lazy
`{wsi_name: Camera}` cache on `self`, populated on first use inside whichever
worker happens to touch that WSI first).

### Position sampling ("disjoint lattice")

Per WSI, per rung: `DsLadder(rungs=DEFAULT_RUNGS).plan_for(wsi, tile_size)` +
`TileSampler` with plain default `RichnessConfig()` (SuperPoint stageA's own
recipe — caps `bg85_95`/`bg95_100` at 0) and `OverlapConfig()` (0 overlap).
"Disjoint" means one draw per WSI never repeats a position. It does NOT mean a
ref/query split — there is no KNN reference bank here, every drawn position
is directly a training example. Train and eval never share a WSI (different
`AccessDatasets` ids), so there is no cross-split leakage to guard against
beyond that.

Each drawn position → **one** rendered patch (`Render(reader, (tile, tile), cfg,
ds=rung).capture(x, y)` —
see "The camera's sensor is ONE TILE" above, and see below for why `Camera`
and not the raw `QueryFromWSI`+`simulate_microscope_photo` pair the first
draft used) carrying that position's rung label. Nothing is cut up afterwards:
the sensor is already the tile.

The constraint this satisfies: the encoder must see the query at the
reference tiles' own effective scale, so a whole frame must never be encoded
as one vector. Rendering at tile size reaches that
from the other direction: instead of rendering a frame and splitting it, the
frame is never rendered.

### Caching — train never persists, eval always does

Not `FeatureStore` (`utilities/FeatureStore.py`) — that module's schema
(`x, y, region, grid_rc` per WSI per level) is built for a dense per-level
grid over ONE WSI, read back by coordinate; training data here is a flat,
order-independent pool of labelled patches, and forcing it through the grid
schema would mean one degenerate "store" per WSI per rung for no benefit
anything downstream reads.

**Manifest** (`build_manifest`, unchanged from the first draft): position
identity only — no pixels, no `Camera` involved at all (`TileSampler`'s own
seed, a different thing from `Camera`'s). `(dataset_id, wsi_name, x, y,
rung)` rows, one CSV per (dataset, split). Cheap to build, so it is rebuilt
whenever the sampler recipe changes rather than invalidated by hand — this
part is identical for train and eval.

**Train — no persisted cache past the manifest.** `RoutingHeadDataset`
(a `torch.utils.data.Dataset`) reads manifest rows and renders ON EVERY
ACCESS via a lazily-built, per-worker `Camera` (see above) — `Camera.
capture(x, y)`, no `rng`, so every epoch (every access, really) gets fresh
augmentation. `DataLoader(num_workers=N, collate_fn=collate_routing_batch)`
is what hides the WSI-read + augmentation latency behind GPU compute — the
lever that makes NOT caching affordable. Baseline 2's per-step saving still
holds WITHIN one batch (augment once, encode once with the frozen encoder,
feed every 2-x head that batch) — it just cannot extend ACROSS epochs the
way the first (pre-augmentation-aware) draft of this design assumed, because
the input itself is different every epoch by design.

**Eval — materialised and cached, same shape the first draft had, rebuilt on
`Camera`.** `materialise_eval_corpus` renders every eval-manifest row through
`Camera.capture(x, y, rng=derived_from((dataset, wsi_name, x, y, rung)))` —
deterministic regardless of which `DataLoader` worker handles it or in what
order — and writes the same uint8 patch shards the first draft's
`materialise_patches` did. `encode_eval_features` then runs a frozen encoder
once over those shards and caches the RAW, UN-REDUCED output — `.tokens()`
for GigaPath/UNI2 (`[N, 197, 1536]`, CLS + 196 patches) or `.spatial()` for
ConvNeXt V2 flattened to `[N, 49, 768]` — NOT the already-pooled `[N, D]`
vector, so every 2-x head can derive its own view from ONE cached tensor:
`LinearHead`/`MlpHead`/`NCM`/etc. read `tokens[:, 0]` (CLS) or
`spatial.mean(dim=1)` (ConvNeXt's GAP); `AttentionPoolHead` reads
`tokens[:, 1:]` (patches, CLS excluded — GigaPath's `num_prefix=1` makes this
right, but UNI2's `num_prefix=9` means the SAME slice would silently fold 8
register tokens in as if they were patches; the real code must read
`model_spec.num_prefix` per encoder, not hardcode `[:, 1:]`) or the full
`spatial` grid. Caching the pooled vector separately would duplicate data
that already sits inside the raw tensor — one cached tensor, multiple cheap
views. `.tokens()`/`.spatial()` return fp32 and are NOT L2-normalised (their
own docstrings say so) — the view-derivation step must cast to fp16 before
caching and must call `F.normalize` itself when deriving the CLS/GAP view,
since `.features()`'s automatic normalisation is bypassed by reading the raw
exit instead.

Baseline 3 (trainable) reads eval's cached pixel shards for scoring, and
`RoutingHeadDataset` for training — never a feature cache either way, the
trunk is not frozen. Its three ConvNeXt siblings (3-1 Linear / 3-2 MLP / 3-3
AttentionPool) do NOT share a trunk forward pass the way baseline 2's heads
share one encoder pass: each fine-tunes its OWN trunk copy (the trunk's
weights move differently under each head's gradient), so they are three
independent training runs, not one loop over three heads.

### Handle count: WSI-batching, not LRU eviction (2026-09-15)

`RoutingHeadDataset` holds open `Camera`/`SafeSlide` handles for whatever
WSIs its `rows` touch (one `_CameraBank` per worker, per spec.md's earlier
"Camera: train vs eval" section). Left unbounded across a full training run,
a worker would accumulate one handle per DISTINCT WSI it has ever touched —
up to all 126 of Ki67_pure by the end of an epoch, times `num_workers`, with
no eviction. Considered and rejected: an LRU cache with a fixed size `K` —
it bounds memory but turns "K too small" into thrashing (evict, then
immediately reopen the same WSI) with no principled way to pick `K` other
than guessing at how many distinct WSIs a worker's shuffled index stream
touches at once.

(A related worry — MRXS's own hole-recovery reopen storm,
`log/TODO.log` 2026-08-22, 752 reopens on one 2816x2608 region read — was
raised and then RULED OUT for this design: that number came from a
large-area tissue-region scan on one of only two WSIs (out of ten sampled)
known to have holes at all; `Camera`'s own read is a single small
tile-footprint bounding square on an already tissue-gated position, a much
smaller and much less hole-prone read. It is not the reason for the design
below.)

The fix is structural instead: `group_by_wsi(rows, group_size, seed=)` chunks
manifest rows into groups of `group_size` WSIs (every row of one WSI stays in
one group), and `iterate_epoch(rows, wsi_group_size=, ...)` gives each group
its OWN `RoutingHeadDataset` + `DataLoader`, exhausts it, and lets it go out
of scope (workers exit, handles close) before building the next group's.
At any moment, at most `wsi_group_size` WSIs are open per worker — a
structural bound, not a cache-size guess, and no eviction logic anywhere.

Trade-off, not free: mini-batches within one WSI-group's DataLoader draw only
from that group's small set of WSIs, so within-batch cross-WSI diversity is
lower than one global shuffle over the whole manifest would give.
`iterate_epoch`'s `epoch_seed` (pass a different value — e.g. the epoch
number — each epoch) reshuffles which WSIs land in which group and the
groups' own order, so the same WSIs are not always batched together across
the run, recovering diversity over many epochs even though any single
mini-batch's diversity stays bounded by `wsi_group_size`.

`n_per_rung` (`build_manifest`) and `wsi_group_size` (`iterate_epoch`) are
independent knobs: the first decides how much data exists per WSI per rung
(richness/balance of the manifest itself), the second decides only the ORDER
or GROUPING the training loop consumes an already-built manifest in (handle
count). Changing one does not require touching the other.

### Training loop: train several arms per forward pass where they truly share one

Baseline 2's arms (any of 2-1..2-6, however many end up built) all read from
the SAME cached raw tensor and the SAME frozen encoder — nothing about
running five heads instead of one changes what the encoder computes. So the
loop batches through the cached tensor ONCE per step and updates every head's
optimizer off that one batch, rather than five separate passes over the same
data. This is the concrete form of "don't duplicate what can be shared" for
this package: not a hardware constraint (compute headroom is not the limit
here), but a design discipline — redundant encoder passes over data that
never changes is waste regardless of whether the cluster can absorb it.

Baseline 3's three siblings cannot share this way (see above) — they run as
three separate training loops, each its own trunk+head+optimizer. As more
baselines/backbones join this package later, whichever of them are frozen
extend baseline 2's shared-pass loop; whichever fine-tune a trunk each get
their own loop, the same split that already holds between baseline 2 and 3.

### Precision

fp16 throughout, training and inference both — same default `TileEncoderConfig`
already uses for every registered encoder (`ModelConfig.dtype = 'fp16'`).
`aiNNModel/models/Heads.py`'s `HeadConfig.dtype` (`'fp32'` default since the
2026-09-16 NaN regression, `'fp16'` still accepted) and its `torch_dtype()`
helper mirror `ConfigIdentity.ModelConfig.dtype`/`torch_dtype()` exactly —
duplicated rather than imported, because a head has no `arch`/`source`/
`weights` identity to hang a `ModelConfig` off, but the two must never
disagree about what `'fp16'` means. (Every trained parameter in `cli/train.py`
is hardcoded fp32 regardless of this default — see the 2026-09-16 status
entries below.)

## Encoders available (measured against the installed `timm==1.0.28` /
`aiNNModel` registry, 2026-09-14 — see session log for the full survey)

| name | `aiNNModel` module | output | dim | notes |
|---|---|---|---|---|
| `gigapath` | `GigaPathFunc.py` | CLS | 1536 | ViT-giant, num_prefix=1 |
| `uni2` | `Uni2Func.py` | CLS | 1536 | ViT-huge, num_prefix=9 (1 CLS + 8 register) |
| `conch_vit` | `ConchVitFunc.py` | `attn_pool`(512) or `trunk`(768) | 512/768 | two genuinely different vector spaces; not yet decided whether/which arm joins this comparison |
| `convnext_v2` | `ConvNeXtV2Func.py` (NEW) | GAP over `[N,768,7,7]` | 768 | Tiny, `fcmae_ft_in22k_in1k` init; frozen-encoder use only, see above |

## Files

```
aiNNModel/models/           generic encoder+head plumbing, shared by any task
                            built on an encoder+head pair (this package, and
                            the planned retrieval work: GraphNN / tree /
                            reranking NN)
    Heads.py                torch_dtype/HeadConfig (precision), LinearHead,
                            MlpHead, ArcFaceHead, AttentionPoolHead — the head
                            CLASSIFIER components, reused unchanged on top of
                            a frozen encoder or a fine-tuned trunk; plus
                            CLASSIFIER_TYPES (name -> class, for
                            Checkpoints.build_from_checkpoint)
    common/
        Head.py             the `Head` assembly class ((reduction, classifier)
                            -> runnable module) and pooled_view/grid_view (the
                            two views one encoder pass is read through).
                            NOT named `Arm` — see its own docstring: that name
                            came from a context of several variants trained
                            SIDE BY SIDE for comparison, and this class is also
                            used where exactly ONE trained head is loaded to
                            run inference, where "arm" would misname it
        Features.py         encode_raw (frozen route, chunked, no graph) /
                            trunk_raw + normalise_patches (fine-tuned route,
                            graph attached, not chunked)
        Checkpoints.py      weight_filename/save_checkpoint/
                            build_from_checkpoint — generic checkpoint format;
                            `extra: dict` is where a task-specific caller
                            (e.g. this package's `cli/train.py`) records
                            anything the format itself has no opinion about

training/MppRoutingHead/
    spec.md              this file
    Datasets.py           split_wsi_names/write_wsi_split (the WSI-level
                            val/test split, recorded not re-derived);
                            build_manifest/write_manifest/read_manifest
                            (positions — the ONLY thing persisted);
                            RenderConfig + _CameraBank + _render_row (one row
                            renders one patch, the camera's sensor IS the tile);
                            RoutingHeadDataset + collate_routing_batch;
                            group_by_wsi + iterate_epoch (WSI-batched epoch
                            loop — bounds open WSI handles structurally)
    Runtime.py            what train.py and evaluate.py BOTH need, so the two
                            cannot drift: BASELINE3_ENCODER/HEAD_CHOICES/
                            heads_for (assembly, task-specific so it stays
                            here rather than moving to aiNNModel/models/),
                            score (incl. the native/resampled split),
                            wandb_init/wandb_log/wandb_finish/
                            wandb_epoch_metrics, predict (one pass over a
                            split); re-exports Head/HeadConfig/encode_raw/
                            trunk_raw/save_checkpoint/build_from_checkpoint/
                            weight_filename from aiNNModel/models/(common/)
                            so callers need one import line, not two
    cli/
        train.py            trains, selects on val, writes weights/ and
                            val_scores.csv. Never touches test.
        evaluate.py          rebuilds a model from a checkpoint alone and
                            scores it on the test split
jobscripts/MppRoutingHead/
    MppRoutingHead.sh

query_sim/camera.py    Camera.capture/capture_with_gt gain an optional `rng`
                        parameter (see "Camera: train vs eval" above) —
                        additive, every existing caller unaffected
query_sim/config.py    DomainGapConfig gains vignette_p / distortion_p
query_sim/pipeline.py  _coin/_maybe — no draw at p=1.0, so every existing
                        corpus reproduces byte-identically
query_sim/source/wsi_query.py
                       QueryFromWSI.reads_natively — whether this rung came off
                        a pyramid level or was LANCZOS-resampled down
```

### Splits, and what may look at which (settled 2026-09-16)

| split | slides | positions | who reads it |
|---|---|---|---|
| train | ki67_pure, all 126 | 100/rung | `cli/train.py` |
| val | `--val-n-wsi` (10) from EACH eval dataset | 20/rung | `cli/train.py`, every epoch |
| test | `--n-wsi` (5) of what is left, per dataset | 50/rung | `cli/evaluate.py` only |

Test is FEWER slides and DEEPER on each (5 x 6 x 50 = 1500 positions per
dataset) than val, which is the other way round (10 slides, 20/rung). The five
are the first five of the recorded test order, which `split_wsi_names` already
shuffled before splitting -- a prefix of a shuffled list is a sample, and it is
the same sample every run without a second seed to keep in step with the first.
Which five were used is readable off the manifest's `wsi_name` column.

### The manifest cache is keyed by a hash, not by a name

```
ki67_pure/train_3f9a1c2b.csv     the positions
ki67_pure/train_3f9a1c2b.json    what the eight characters stand for
```

A manifest is reused whenever it exists, so its filename has to be a complete
key or it is a trap: a re-run with a changed `--n-per-rung` would silently
score the old manifest — the flag appearing to have done nothing, and the
number it produces being real but not the one that was asked for.

The first attempt put the two flags that happened to be on the command line
into the name (`test_w5_r50.csv`) and was worse than either alternative,
because `--tile` and `--seed` are just as much part of what the positions are
and it looked like it was protecting you. `manifest_parts` now lists all
eleven facts the content depends on and `ConfigIdentity.short_id` hashes them,
the same idiom the feature stores use for `StoreMeta.encoder_id`.

The sidecar JSON is what keeps that hash from being the opaque kind CLAUDE.md
objects to: `ls` shows eight characters, and the file beside it says exactly
which eleven facts they stand for. Two of those eleven are recorded by NAME
rather than by value — `richness=RichnessConfig()` and `overlap=OverlapConfig()`
— because this package only ever uses the defaults and spelling out seven
buckets' floors and caps would be a second copy of a contract that lives in
`TileSampler`. That is a stated limit: a caller passing a non-default one has
to grow that line.

### The native/resampled split only means anything inside one dataset

`score()` reports `level_accuracy_native` beside `level_accuracy_resampled` to
catch a head scoring on the LANCZOS signature instead of on scale. That
comparison is valid **within one dataset only**: Ki67's 2x pyramid makes every
rung native, BRACS's 4x pyramid makes rungs 2/8/32 resampled. Taken over the
two together, the "native" side is BRACS's native rungs PLUS the whole of Ki67
while the "resampled" side is BRACS alone — so the number measures the dataset,
not the resampling.

`cli/evaluate.py` was already clean, because it loops per dataset. `cli/train.py`
was not: `val_rows` concatenates both eval datasets into one `predict` call.
Fixed by `Runtime.rescore`, which re-runs `score()` over a filtered slice of
`predict`'s per-tile detail — the rows already carry `gt_class`, `pred_class`,
`native` and `dataset`, so a breakdown costs a list comprehension rather than a
second forward pass.

`val_scores.csv` therefore has a `val_dataset` column with one row per dataset
plus an `all` row per (arm, epoch). **`best` is selected on `all`**, because
selecting on one dataset would be selecting against the other; the per-dataset
rows are there to be read, not to choose.

### Per-tile predictions

`cli/evaluate.py` writes two CSVs. `test_scores_<tag>.csv` is one row per
(checkpoint, dataset). `test_predictions_<tag>.csv` is **one row per tile**:

```
weights, encoder, trunk, arm, dataset, wsi_name, x, y, rung, bucket, native,
gt_class, gt_rung, pred_class, pred_rung, correct
```

An aggregate accuracy cannot say whether the errors are one bad slide, one
richness bucket, or the resampled rungs. These rows can, and at 1500 positions
per dataset they are a few MB. `bucket` is `SampleMeta.bucket`, carried on
`ManifestRow` from the sampler rather than recomputed later: the sampler's
quotas are per bucket, so it is the number that actually placed the tile, and
a mask re-read could disagree with it.

Carrying it required the row itself to survive the DataLoader —
`iterate_epoch` shuffles and regroups, so after that a batch position says
nothing about which manifest line produced an example. `RoutingHeadDataset`
returns `(row, patch, label, native)` and `collate_routing_batch` keeps `rows`
as a plain list, since it is identity rather than a quantity.

The split is at WSI level, so no position from one slide can land on both
sides. `split_wsi_names` sorts before its seeded shuffle (`list_names` returns
registry order, which nobody controls) and `write_wsi_split` records the answer
in the recorded split; `evaluate.py` READS it rather than re-deriving the
split, because a split recomputed from `--seed` is one library version away
from quietly putting val slides into test.

Val spans both eval datasets deliberately: selecting on it selects for
cross-dataset generalisation, which is the thing this package exists to
measure. There is still no in-domain val, per the user's own call —
ki67_with_photo already plays that role.

`best` is by val `level_accuracy`, and it is a meaningful quantity only because
this split exists. Before it did, a run's reported number was whatever the last
epoch happened to be.

### Weight files

```
<run>/weights/<encoder>_<frozen|finetuned>_<head>_<last|best>.pt
              gigapath_frozen_arcface_best.pt
              convnext_v2_finetuned_linear_best.pt
```

- **Encoder first**, because `_paths.encoder_tag(encoder, head)` already spells
  an encoder-plus-head name that way and CLAUDE.md's result paths follow it
  (`conch_vit_attn_pool`). A second ordering for the same pair of facts would
  be a second convention.
- **`encoder` is the registry name** (`convnext_v2`, never `ConvNeXt`) — the
  same rule that governs `--encoders`.
- **`frozen|finetuned` is the field that says what weights are inside.** A
  frozen run trains the head alone, so the file holds `head_state` and
  `trunk_state=None`: the trunk is the published checkpoint, reachable by name,
  and storing a copy per head would be several GB saying nothing. A fine-tuned
  run holds both — in ONE file rather than two, because they were trained
  together and a mismatched pair of files is a failure mode worth making
  impossible. It also keeps a future FROZEN `convnext_v2` baseline-2 head from
  colliding with baseline 3's fine-tuned one.
- **Two tags, not one file per epoch.** With several heads and ten epochs,
  per-epoch checkpoints are tens of GB nobody opens; the only two ever asked
  for are "the one that scored best" and "the one training ended on".

A checkpoint carries enough to REBUILD the model, not merely to reload into one
the reader guessed correctly: encoder name, frozen flag, head name, reduction
and classifier class name (both read off the LIVE `Head` object by
`Checkpoints.save_checkpoint`, not looked up from `HEAD_CHOICES` by name — the
object already knows what it is), `head_cfg`, the epoch, the val result and the
full `args`. `evaluate.py` therefore takes no `--encoder` and no `--head` —
passing either would be a second source of truth able to disagree with the
weights, and it refuses outright if `--tile` differs from what the checkpoint
was trained at (`ckpt['extra']['tile_size']` — see below).

`Checkpoints.save_checkpoint` has no `render_cfg` parameter — `RenderConfig`
is this package's own type and the generic layer has no business knowing it
exists — and takes an open `extra: Optional[Dict]` instead of hardcoded
fields. `cli/train.py`'s `save_tagged` passes `extra=dict(tile_size=args.tile,
rungs=RUNGS)`: `tile_size` is read back by `evaluate.py`'s tile-size guard,
`rungs` by `stage1_estimation/ClassifierEstMpp.py` to turn a predicted
class back into a ds value without importing this package's own `Datasets.py`
at inference time. `num_classes` is not carried — `len(rungs)` already says
it, and a stored copy could disagree with the tuple sitting right next to it.

### No rendered corpus on disk (2026-09-16)

An earlier draft materialised the eval corpus into `.pt` patch shards and
cached each frozen encoder's output on top of them (`_ShardWriter`,
`materialise_eval_corpus`, `encode_eval_features`, `load_eval_features`,
`load_patch_shards`). All of it is deleted. Val and test now run the same
`iterate_epoch` training does, with a different manifest.

Two reasons. The feature cache stored the UN-REDUCED exit of every patch —
197 x 1536 fp16 each — to save one forward pass; at the frame-rendering scale
that was ~126 GB for one eval leg, and the whole tensor was then moved to the
GPU at once. And a row is one patch now rather than twenty, so re-rendering
costs a twentieth of what it did when the cache was designed.

`split='eval'` survives all of this and is NOT a cache: `_render_row` seeds its
rng from the row's own identity, so a position renders the same pixels on every
call, with nothing stored. That is what makes a val curve comparable across
epochs and a test number reproducible across runs.

The manifest is still written, and is the exception that proves the rule: it
holds positions and no pixels, rebuilding it means HSV segmentation over 126
slides, and nothing about it can freeze an augmentation.

## Status (2026-09-16)

Done: `aiNNModel/ConvNeXtV2Func.py` (registered as `convnext_v2`).
`AccessDatasets.list_names(dataset=)`. `Models.py` — `HeadConfig`/
`torch_dtype`, `LinearHead`, `MlpHead`, `ArcFaceHead` (2-3),
`AttentionPoolHead`. `cli/train.py` — `ARMS`/`Arm`/`arms_for`, both
baselines, one `scores.csv`. `query_sim/camera.py` — `Camera.capture`/
`capture_with_gt` gained the optional `rng` parameter (additive, every
existing caller unaffected — verified by reading all nine, not just by the
diff being small). `Datasets.py` — `build_manifest`/`write_manifest`/
`read_manifest` (position identity, unchanged in spirit from the first
draft); `_CameraBank`/`RoutingHeadDataset`/`collate_routing_batch` (train,
live Camera render per access via a per-worker-lazy `_CameraBank`, no
persisted cache); `group_by_wsi`/`iterate_epoch` (WSI-batched epoch loop,
bounds handle count structurally — see "Handle count" above);
`materialise_eval_corpus`/`load_patch_shards`/
`encode_eval_features`/`load_eval_features`/`pooled_view`/`grid_view` (eval,
deterministic Camera render + cached raw `tokens()`/`spatial()`, `num_prefix`
-aware view derivation). `Runtime.py` and `cli/evaluate.py`; the val/test
split, checkpointing, and the removal of every rendered-corpus cache.
`Runtime.wandb_init`/`wandb_log`/`wandb_finish`/`wandb_epoch_metrics` and the
matching `--wandb-project`/`--wandb-mode`/`--run-name` in `cli/train.py` — same
optional-dependency idiom as `SuperPathPoint/SuperPoint/Trainer.py`, one run
per MODEL (per encoder in baseline 2, per arm in baseline 3, since that is the
boundary each already trains at), logging exactly the rows that become
`val_scores.csv` rather than a second computation of the same numbers.
`--wandb-mode` defaults to `WANDB_MODE` (`jobscripts/_env.sh`), so a smoke run
goes offline via the environment, matching the other two training loops.
This package had NO wandb code before 2026-09-16 — a request to turn wandb off
during a smoke run had nothing here to act on, which is what the request
actually meant until this landed.

**The fp16-head NaN regressed on the first full run (2026-09-16), after
already being "fixed."** `HeadConfig.dtype`'s dataclass default was changed to
`'fp32'`, but both `cli/train.py` construction sites still called
`HeadConfig(..., dtype=args.dtype)` and `--dtype`'s own CLI default was still
`'fp16'` — an explicit keyword always wins over a dataclass default, so the
smoke run (which happened not to exercise that path the same way) looked
fixed while the full run reproduced `loss ... nan` on every arm. Fixed at the
actual call sites: both `HeadConfig(...)` constructions now hardcode
`dtype='fp32'`, and `--dtype` is repointed to the one thing it can safely mean
— baseline 2's FROZEN encoder's inference precision — with its help text
saying so explicitly. `Models.py`'s `HeadConfig` docstring now says up front
that the dataclass default alone fixes nothing if a caller forwards a CLI
flag on top of it.

**That repointing crashed the very next run**: `encoder_config(encoder_name,
dtype=args.dtype)` raised `TypeError: __init__() got an unexpected keyword
argument 'dtype'`. `dtype` is not a top-level field of `TileEncoderConfig` —
it lives on the nested `ModelConfig`, and `TileEncoder.variant(dtype=...)`'s
nested-replace only exists for an already-built `TileEncoder` instance, not
for `encoder_config`/`config_from`'s plain `cls(**over)`. Fixed to build the
config first and `dataclasses.replace` the nested `model` field directly —
the exact pattern `run_baseline3` already used for the trunk, now used
symmetrically for baseline 2's frozen encoder.

**A fourth precision bug, found on review rather than by a failed run:**
`Runtime.build_from_checkpoint` built the encoder at its registry's own
default dtype and then, for a fine-tuned checkpoint, called
`encoder.model.load_state_dict(ckpt['trunk_state'])` on top of it.
`ConvNeXtV2EncoderConfig`'s own default is `dtype='fp16'`, while
`run_baseline3` trains the trunk at `fp32` — so every baseline-3 test-set
score `cli/evaluate.py` had ever produced would have been scoring a trunk
silently downcast to fp16 on load (`Tensor.copy_`, which `load_state_dict`
uses internally, casts on a dtype mismatch with no error and no warning).
Fixed by forcing `dtype='fp32'` unconditionally for a fine-tuned checkpoint
(never anything read back from `args` — a fine-tuned trunk here is always
fp32, full stop) and, for a frozen checkpoint, by building at the exact
`--dtype` training used (`ckpt['args']['dtype']`) rather than the registry
default, for exactness even though that particular gap's measured effect is
negligible (cos=0.99995, and `tokens()`/`spatial()` always cast their output
back to fp32 regardless of the internal compute dtype).

**`QueryFromWSI.reads_natively` was too strict to mean what it claims,** found
from the FIRST full run's real val numbers: BRACS's own docstring ("four
levels: ds 1, 4, 16, 32") turned out not to hold literally across its slides
(measured level_downsamples include a 4th level near 64 on some, near 32 on
others, and one held-out slide with only three levels at all) — but the
larger issue was `reads_natively`'s own arithmetic. It compared
`(self._read_w, self._read_h) == (self.output_w, self.output_h)`, and
`_read_w = int(w_um / self.chosen_mpp)` TRUNCATES; a measured level downsample
is essentially never a bit-exact integer (the reason `utilities/SafeSlide.py`
already carries `_LEVEL_REL_TOL = 1e-3` for the identical problem). A level at
ds=4.0000226 — six PARTS PER MILLION off the target — truncated `256*4/
4.0000226 = 255.9985` down to 255, one pixel short of 256, and got called
"resampled": indistinguishable, under the old code, from the genuinely
2x-different case one rung over. Effect on the numbers: bracs/test's val
showed `n_native=220` (≈ rung 1 alone) against `n_resampled=870` (rungs
2/4/8/16/32 all lumped in), which is why the observed "resampled accuracy" was
so low — it was mostly measuring the ALREADY-DIAGNOSED class-imbalance problem
(rung 16/32 have almost no training supply) rather than anything about
LANCZOS. Fixed by comparing `chosen_mpp` to the requested `mpp` with the same
`1e-3` relative tolerance instead of a post-truncation pixel count. Re-derived
by hand against the run's own printed diffs: rung 4 and 16 become native on
essentially every BRACS slide (diffs of 1e-5–1e-4), rung 32 becomes a genuine
per-slide split (native on the ~6/10 slides whose 4th level lands near 32,
resampled on the rest), and rungs 2/8 stay resampled (diff ≈ 1.0, a real 2x).
A currently-running job does not pick this up until restarted (Python modules
load once); `cli/evaluate.py`, invoked as a fresh process after training
finishes, does.

**Class-weighted loss, added to fix the SAME class-imbalance problem at its
actual source rather than only diagnose it.** `Datasets.class_weights(rows,
device)` computes sklearn's 'balanced' weights (`w_c = N / (K * n_c)`) from
the training manifest's own rung counts, once per run and reused every epoch;
`cli/train.py` passes it to every `F.cross_entropy(..., weight=...)` call in
both baselines. `--class-weight {balanced,none}`, default `balanced`; `none`
restores the old unweighted behaviour for comparison. Val/test scoring is
unaffected — `Runtime.predict` never computes a loss, only accuracy, so
nothing there needed to change.

Every touched file `py_compile`s clean; NOT run
(no `python` execution on the login node — see `who-runs-commands` — so the
actual `Camera`/encoder/DataLoader-worker wiring is unverified at runtime).

Not yet written: the jobscript (`jobscripts/MppRoutingHeadJobs/Train.sh`), and
arms **2-2 (NCM)** and **2-4 (Mahalanobis)**. Those two are not blocked on
design but on plumbing: neither is trained by gradient descent — they fit
statistics in one pass (per-class means; means plus a pooled covariance) — so
they need a fit-once path beside `train()`'s `backward()`/`step()`, and
because training photos are rendered fresh every epoch that accumulation has
to stream (per-class sums and counts, plus `sum(x x^T)` for 2-4) rather than
hold a feature matrix. 2-4 has exactly one viable form at D=1536: a POOLED
covariance with shrinkage toward `(tr/D)*I`, i.e. LDA — per-class covariance
is 2.4M parameters six times over, far past what the sample count supports,
and inverting a badly estimated one amplifies precisely the noise directions.
Worth noting where that lands it: with equal priors and a shared covariance,
`-d^2` collapses to a LINEAR classifier, so 2-4 against 2-1 is "the same
decision family, boundary computed from statistics vs learned by SGD".
