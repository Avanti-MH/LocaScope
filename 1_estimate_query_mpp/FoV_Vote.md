# FoV Patch-voting Strategies

比較六種把一張 FoV 中 \(M\) 個 patch 預測合成一個 MPP class 的方法。所有方法都應使用完全相同的 FoV、patch logits 與 rung scoring 做 paired evaluation。

## Notation

- \(p_{i,c}\)：patch \(i\) 對 class \(c\) 的 softmax probability。
- \(M\)：一張 FoV 的 patch 數。
- Rung：\(r=[1,2,4,8,16,32]\)。
- Log-rung：\(\ell_c=\log_2(r_c)=[0,1,2,3,4,5]\)。

「Patch class 中位數」是對每個 patch 的 hard argmax class 取 median；「Median log-rung」則先使用完整 probability distribution 計算每個 patch 的 expected log-rung。兩者不是同一種方法。

## 方法總覽

| 方法 | 核心聚合方式 | 主要特性 |
|---|---|---|
| Mean probability | 平均每個 class 的 softmax probability | 保留 confidence，但容易被少數過度自信 patch 拉動 |
| Hard majority | 每個 patch 的 argmax 投一票 | 不受 probability calibration 影響，但完全丟棄 confidence |
| Patch class 中位數 | 對 argmax class index 取 median | 使用 class 順序，對少數遠距 outlier 穩健 |
| Median log-rung | 對 expected log-rung 取 median | 同時保留 soft distribution 與 robust aggregation |
| Sum log-probability | 累加每個 class 的 log probability | 要求所有 patch 持續支持，容易產生單 patch veto |
| Trimmed / quality-weighted | 依 patch 品質加權或移除 patch | 可排除 artifact，但品質權重可能與尺度產生偏差 |

## 1. Mean probability

目前 `ClassifierEstMpp` 使用的方法。

### 公式與做法

\[
\bar P_c=\frac{1}{M}\sum_i p_{i,c},
\qquad
\hat c=\arg\max_c \bar P_c
\]

逐 class 平均所有 patch 的 softmax probability，選擇平均值最大的 class。

### 成功案例

GT = \(c_2\)（ds=4）：

```text
p1=(.05,.10,.65,.15,.03,.02) → c2
p2=(.04,.11,.60,.18,.05,.02) → c2
p3=(.08,.12,.55,.18,.05,.02) → c2
p4=(.05,.10,.58,.20,.05,.02) → c2
p5=(.02,.03,.10,.15,.20,.50) → c5
```

```text
mean=(.048,.092,.496,.172,.076,.116) → c2，正確
```

四個一致的 \(c_2\) patch 可以吸收一個 confident coarse outlier。

### 失敗案例

GT = \(c_2\)（ds=4）：

```text
p1–p4=(.16,.16,.20,.16,.16,.16) → 各自 c2
p5=(.002,.002,.002,.002,.002,.990) → c5
```

```text
mean=(.128,.128,.160,.128,.128,.326) → c5，錯 3 rungs
```

四票都微弱支持正確的 \(c_2\)，但一個未校準的 0.99 將 arithmetic mean 拉到 \(c_5\)。

### 解決的問題

相較 Hard majority 與 Patch class median，它不會完全丟棄 confidence。少數高信心且資訊清楚的 patch，可以勝過大量近乎平票的 patch。

### 危險分布與統計

危險型態：少數 patch 對某 class 極高信心，但多數 patch 的 argmax 是另一個 class；mean winner 被少數 patch 主導。

```text
flag = (mean winner ≠ mode(argmax))
       AND (winner-supporting patches < M/2)
```

應統計 flag 比例、flag subset accuracy，以及 winner-supporting patch 數與錯誤率的關係。

## 2. Hard majority

### 公式與做法

\[
z_i=\arg\max_c p_{i,c},
\qquad
\hat c=\operatorname{mode}(z_1,\ldots,z_M)
\]

每個 patch 只有一票，票數最多的 class 勝出。

### 成功案例

GT = \(c_2\)（ds=4）：

```text
p1–p4=(.16,.16,.20,.16,.16,.16) → 各投 c2
p5=(.002,.002,.002,.002,.002,.990) → 投 c5
```

```text
votes: c2=4, c5=1 → c2，正確
```

每個 patch 只有一票，\(c_5=.99\) 不會取得比 \(c_2=.20\) 更大的票權。

### 失敗案例

GT = \(c_2\)（ds=4）：

```text
p1–p4=(.16,.20,.19,.15,.15,.15) → 各投 c1
p5–p6=(.01,.01,.94,.02,.01,.01) → 各投 c2
```

```text
votes: c1=4, c2=2 → c1，錯 1 rung
```

四個 top-1 僅 0.20 的錯票，擊敗兩個 0.94 的明確正確票；confidence 被完全丟棄。

### 解決的問題

相較 Mean probability，它對未校準的高信心 outlier 較穩健，也不受 temperature scaling 改變。

### 危險分布與統計

危險型態：多數票 patch 都是 low-margin／high-entropy，少數反對票卻是高信心，票數與證據強度相反。

```text
flag = median(margin of majority patches) < τ_low
       AND median(margin of dissenters) > τ_high
```

應統計 flag 比例與 conditional accuracy，另外報告平票 FoV 比例及不同 tie-break rule 的結果翻轉率。

## 3. Patch class 中位數

亦即 hard ordinal median。

### 公式與做法

\[
z_i=\arg\max_c p_{i,c},
\qquad
\hat c=\operatorname{median}(z_1,\ldots,z_M)
\]

依尺度排序 class index，取中位數後映回 rung。若 patch 數為偶數，必須事先定義 lower median、upper median 或其他固定 tie rule。

### 成功案例

GT = \(c_2\)（ds=4）：

```text
p1–p4=(.03,.08,.72,.10,.04,.03) → classes 2,2,2,2
p5=(.002,.002,.002,.002,.002,.990) → class 5
```

```text
median([2,2,2,2,5])=2 → ds=4，正確
```

class 5 距離再遠也只是一個排序樣本，無法拉動中位數。

### 失敗案例

GT = \(c_2\)（ds=4）：

```text
p1–p4=(.22,.18,.20,.14,.13,.13) → classes 0,0,0,0
p5–p7=(.01,.01,.94,.02,.01,.01) → classes 2,2,2
```

```text
median([0,0,0,0,2,2,2])=0 → ds=1，錯 2 rungs
```

低品質 patch 一旦過半，即使 top-1 只有 0.22，也會擊敗三個 0.94 的正確 patch。

### 解決的問題

相較 Hard majority，它使用 class 的 ordinal 順序，分散投票時也不依賴唯一 mode；相較 Mean probability，它較不受少數極端 class outlier 影響。

### 危險分布與統計

危險型態：超過半數 patch 以很低 margin 投向同一側或同一錯誤 class，少數高信心正確 patch 位於另一側。

```text
flag = median(top-1 probability) < τ_low
       OR |lower-median − upper-median| ≥ 1
```

應統計 low-confidence median 與偶數中央值分裂的 FoV 比例，並分別報告 lower／upper tie rule accuracy。

## 4. Median log-rung

亦即 soft ordinal median。

### 公式與做法

\[
e_i=\sum_c p_{i,c}\ell_c,
\qquad
\hat e=\operatorname{median}(e_1,\ldots,e_M),
\qquad
\hat c=\operatorname{nearest}_{\ell}(\hat e)
\]

每個 patch 先利用完整 probability distribution 計算 expected log-rung，再跨 patch 取 median，最後 snap 到最近的 rung。

### 成功案例

GT = \(c_2\)（ds=4）：

```text
p1=(.02,.08,.72,.12,.04,.02) → e1=2.14
p2=(.03,.10,.68,.13,.04,.02) → e2=2.11
p3=(.04,.10,.60,.18,.06,.02) → e3=2.18
p4=(.02,.06,.70,.15,.05,.02) → e4=2.21
p5=(.01,.01,.03,.05,.10,.80) → e5=4.62
```

```text
median(e)=2.18 → nearest ℓ=2 → c2，正確
```

完整 distribution 被保留，但 \(e_5\) 的 coarse outlier 被 median 隔離。

### 失敗案例

GT = \(c_0\)（ds=1）：

```text
p1=(.50,0,0,0,0,.50) → e1=2.50
p2=(.55,0,0,0,0,.45) → e2=2.25
p3=(.45,0,0,0,0,.55) → e3=2.75
p4=(.52,0,0,0,0,.48) → e4=2.40
p5=(.48,0,0,0,0,.52) → e5=2.60
```

```text
median(e)=2.50 → tie snap c2/c3，兩者都錯
```

所有 probability mass 都只在 \(c_0/c_5\)，expectation 卻製造出完全沒有被任何 patch 支持的中尺度。

### 解決的問題

相較 Patch class median 與 Hard majority，它不會丟棄 soft distribution；相較 Mean probability，它用 median 提供對少數 patch outlier 的 robust aggregation。

### 危險分布與統計

危險型態：patch probability 呈現遠距雙峰，expected log-rung 落在中間，但中間 class 幾乎沒有 probability 或 argmax 支持。

```text
flag = mean_i p[i, ĉ] < τ_support
       OR count(argmax_i = ĉ) = 0
```

應統計 unsupported-interpolation 比例，並對 flagged FoV 報告 bimodality、snap class 與 GT 的距離。

## 5. Sum log-probability

亦即 product of evidence。

### 公式與做法

\[
S_c=\sum_i\log(\max(p_{i,c},\epsilon)),
\qquad
\hat c=\arg\max_c S_c
\]

等價於最大化各 patch probability 的乘積；\(\epsilon\) 用來避免 \(\log(0)\)。

### 成功案例

GT = \(c_2\)（ds=4）：

```text
p1=(.05,.10,.60,.15,.07,.03)
p2=(.05,.10,.55,.18,.08,.04)
p3=(.06,.10,.50,.20,.10,.04)
```

```text
Πp(c2)=.165
次高 Πp(c3)=.0054
→ c2，正確
```

只有每個 patch 都持續支持的 class 才會保有高 product。

### 失敗案例

GT = \(c_2\)（ds=4）：

```text
p1–p4=(.04,.08,.70,.10,.05,.03)
p5=(.01,.01,.000001,.01,.019999,.95)
```

```text
Πp(c2)=2.40×10⁻⁷
Πp(c5)=7.70×10⁻⁷
→ c5，錯 3 rungs
```

單一 patch 對正確 class 給 \(10^{-6}\)，形成近似 veto，推翻另外四個 0.70。

### 解決的問題

相較 Mean probability，它不容易讓只被少數 patch 強烈支持的 class 勝出；winner 必須避免被任何 patch 強烈否定。

### 危險分布與統計

危險型態：單一 patch 對候選 class 給接近零的 probability，形成 veto；移除該 patch 後答案立即翻轉。

```text
flag = min_i p[i, ĉ_without_i] < ε_veto
       AND leave-one-out winner changes
```

應統計 leave-one-out unstable FoV 比例、每張 FoV 的最小 probability，以及單 patch veto 導致的錯誤率。

## 6. Trimmed / quality-weighted vote

### 公式與做法

\[
\bar P_c=\frac{\sum_i w_i p_{i,c}}{\sum_i w_i},
\qquad
w_i\in[0,1]
\]

根據 tissue amount、blur、artifact 或預測一致性給予每個 patch 權重。Trimmed vote 是令部分 \(w_i=0\) 的特殊情況。

### 成功案例

GT = \(c_2\)（ds=4）：

```text
p1–p4=(.16,.16,.20,.16,.16,.16), w=1
p5=(.002,.002,.002,.002,.002,.990), w=.05
```

```text
weighted c2=.198, c5=.170 → c2，正確
```

quality model 正確辨識 \(p_5\) 是 artifact，讓四個正常 patch 恢復主導。

### 失敗案例

GT = \(c_5\)（ds=32）：

```text
p1–p3=(.02,.03,.05,.08,.12,.70), w=.1（被誤判為 blur）
p4–p6=(.10,.65,.10,.05,.05,.05), w=1
```

```text
weighted c1=.594, c5=.109 → c1，錯 4 rungs
```

quality 與尺度相關：平滑的 coarse tissue 被錯誤降權。未加權時 \(c_5=.375>c_1=.340\)，原本會答對。

### 解決的問題

它解決所有 unweighted 方法將空白、模糊、反光、筆跡 patch 與正常組織等權處理的問題。

### 危險分布與統計

危險型態：quality weight 與預測尺度系統性相關，或者權重集中在少數 patch；某些 rung 的正常外觀被 quality model 當作低品質。

```text
flag = |corr(w_i, e_i)| > τ_corr
       OR ESS=(Σw)²/Σ(w²) < τ_ESS
```

應統計 flag 比例、各 predicted/GT rung 的平均 weight、effective sample size，以及 weighted 與 unweighted 結果的翻轉率。

## 建議的第一輪比較

第一輪應同時比較：

1. Mean probability。
2. Hard majority。
3. Patch class median。
4. Median expected log-rung。
5. Quality-weighted mean。

Sum log-probability 可保留作為壓力測試，但相鄰 FoV patches 通常不是統計獨立樣本，因此 product-of-evidence 的獨立性假設並不可靠。

若重視對少數極端錯誤的穩健性，優先觀察 Patch class median 與 Median log-rung。若重視 confidence 輸出，使用 calibrated mean 或 quality-weighted mean，並另外保存 top-1/top-2 margin、entropy 與 patch disagreement。平均 winning probability 本身不是 FoV 正確率。

## 危險分布的統計方式

每一種 risk flag 都應同時報告：

- Prevalence：flagged FoV / all FoV。
- Flagged accuracy。
- Unflagged accuracy。
- Flagged 與 unflagged error risk ratio。
- 按 dataset 與 GT rung 分層後的結果。

其中：

```text
margin = top-1 probability − top-2 probability
```

所有 threshold \(\tau\) 必須預先在 validation set 固定，不能查看 test 結果後再調整。Flag 代表某種風險型態，不代表該 FoV 必然分類錯誤。

## 最低限度的評估指標

- Equal-rung accuracy，即每個 rung accuracy 等權平均。
- Log₂-rung MAE。
- 每個 rung 的 confusion matrix。
- Risk–coverage curve。
- 以 FoV 為單位的 bootstrap confidence interval。
- 每一對 aggregation strategy 的 paired 勝負統計。

不要只看單一 pooled overall accuracy。目前 `stage1_compare` 在 coarse rungs 有 collapse；新 aggregation 若只改善常見的 ds=1/2，卻使 ds=16/32 更差，pooled accuracy 仍可能看似上升。因此應維持每個 rung 等權，並分 dataset 報告。
