# Stage 2 Retrieval Proposal — 架構規格

## 0. 符號

| 符號 | 意義 |
|---|---|
| $F$ / $W$ | FoV（query）/ candidate Window（reference） |
| $H\times W$ | FoV tile grid；$N=HW$ |
| $P$ | 每個 tile 的 patch token 數 |
| $d_0$ | foundation feature 維度 |
| $d$ | tile model hidden 維度（config 指定）；$h$ heads，$d_h=d/h$ |
| $d_g$ | general token 維度 |
| $\theta\in\{0°,90°,180°,270°\}$ | query 旋轉 hypothesis |
| $m_i\in\{0,1\}$ | 有效 tile mask（padding 或無效 tile = 0），資料給定，不可學習 |

所有 tensor shape 不含 batch 維度。

---

## 1. 統一介面

所有方法都是同一個型別：

$$
\mathrm{Scorer}:\ \mathrm{CandidateSet}_{\text{in}}\times\mathrm{Query}\ \rightarrow\ \mathrm{CandidateSet}_{\text{out}}
$$

- Candidate = (region, lattice, row, col, $\theta$, score)；identity 永遠是「位置 × 旋轉」。
- Phase 1 的 $\mathrm{CandidateSet}_{\text{in}}$ = 由 WSI region fold 出的全部 windows × 4 rotations。
- Phase $n>1$ 的 $\mathrm{CandidateSet}_{\text{in}}$ = 上游 Top-K。
- 任何 scorer 都可放在任意 phase；限制只有計算量與 reference 端能否預先計算。
- Scorer 吃哪一個 CandidateSet 是同一個函數：吃整張 slide 的全部 windows（可共用計算，如 SlideWin 的 box filter、`GloRet` 預先算好的 $k$），或吃上游的子集（逐 candidate gather）；同一個 candidate 兩種吃法的分數必須一致。
- 可攜條件：$s(c)=f(\text{Query},c)$，只依賴 query 與該 candidate。跨 candidates 做 attention / normalization 的 scorer（如 CSLS、CandGraph）在不同大小的 CandidateSet 上，同一 candidate 會得到不同分數，只能用於訓練時的同一個 phase。

### 目標

| Phase | 輸入 | 輸出 | 指標 |
|---|---|---|---|
| 1 | 全部 windows × 4θ | Top-100 | $R@100\ge 0.9$；top-1～50 不要求 |
| 2..n | 上游 Top-K | Top-10 | $R@10 \rightarrow R_{\text{Phase1}}@100$ |

評估同時報告 Phase 1 $R@100$、final $R@10$、conditional recall $=R_{\text{final}}@10\,/\,R_{\text{Phase1}}@100$，以及 rotation accuracy（位置正確條件下）。

---

## 2. 旋轉協定

- Reference windows 不旋轉。
- Query RGB pixels 以 `np.rot90` 旋轉 4 次 → 每次重新切 tile → 重新跑 foundation encoder。
- $\theta\in\{0°,180°\}$：query grid $H\times W$，對應 window $H\times W$。
- $\theta\in\{90°,270°\}$：query grid $W\times H$，對應 window $W\times H$。
- 每個 $\theta$ 的 query 已轉到 window 座標方向，因此 expected tile mapping 一律是 identity：$\pi_\theta(i)=i$。
- 4 個 rotations 放同一個 encoder batch。
- 不採用「只 encode 0°/90°、以 token 反轉代替 180°/270°」：token 反轉不會旋轉 tile 內 pixels。

---

## 3. 共用前段：Tile Tokenizer（可替換）

### 3.1 資料流

$$
\text{tile RGB}\ \xrightarrow{\ \text{Foundation (frozen)}\ }\ [\,t_{\text{CLS}};\ T_{\text{patch}}\,]\in\mathbb{R}^{(1+P)\times d_0}
\ \xrightarrow{\ \text{TilePool}\ }\ x_i^{0}\in\mathbb{R}^{d_0}
$$

$$
X^{0}=[x_1^{0};\dots;x_N^{0}]\in\mathbb{R}^{N\times d_0}
\ \xrightarrow{\ \text{LN}\rightarrow W_{\text{in}}\ }\ X\in\mathbb{R}^{N\times d}
$$

### 3.2 TilePool 選項（一次選一種）

| 選項 | 公式 | 可學習 |
|---|---|---|
| CLS | $x_i^0=t_{\text{CLS}}$ | 否 |
| CLS‖mean（`cls_avg`） | $x_i^0=[\,t_{\text{CLS}};\ \overline{T_{\text{patch}}}\,]$ | 否 |
| Patch AttnPool（`attnpool`） | $e_p=w_t^\top\tanh(W_tT_{\text{patch}}[p])$，$\beta=\mathrm{softmax}_p(e)$，$x_i^0=\sum_p\beta_pT_{\text{patch}}[p]$ | 是（$W_t,w_t$） |

- TilePool 選項為 CLS / CLS‖mean / Patch AttnPool。
- Patch AttnPool：推論時權重固定，reference tile 過 foundation 與 AttnPool 後只存池化後的一個向量，不需存 patch token。訓練時每步需要 patch token（現跑 foundation，或快取抽樣到的 tile）。其 reference 特徵隨 checkpoint 走，不與 CLS 系共用 foundation 層級的特徵快取。

不同 phase 可以使用不同 foundation 與不同 TilePool。

### 3.3 位置編碼（任意 $H\times W$）

禁止固定大小的 learned absolute table（綁定 grid 大小）。可選：

| 選項 | 公式 | 可學習 |
|---|---|---|
| Relative bias（預設） | $B_{ij}=f_{\text{pos}}(\Delta r_{ij},\Delta c_{ij})$，$f_{\text{pos}}$ 為小型 MLP，輸入以 tile 為單位 | 是（MLP） |
| 2D sinusoidal | $x_i\leftarrow x_i+\phi(r_i,c_i)$ | 否 |
| 2D RoPE | 對 $Q,K$ 依 $(r_i,c_i)$ 旋轉 | 否 |

Mask $m_i$：

- 同一 batch 內 shape 相同（如 Phase 1 同一 $\theta$ 的全部 windows）：不需 padding，$m_i\equiv1$。
- 同一 batch 混合不同 $H\times W$（訓練）：padding 至最大長度，padding 位置 $m_i=0$。
- 可選：空白 / FoV 視野外 tile 設 $m_i=0$。
- 所有 attention 對 $m_j=0$ 的 key 設 logit $=-\infty$。

---

## 4. Tile Context Block（可替換，一次選一種）

輸入 $X\in\mathbb{R}^{N\times d}$，輸出 $X'\in\mathbb{R}^{N\times d}$。

ContextEn 的名稱：`SelfAttn`（4.1）、`GNN`（4.2）、`SptSelfAttn`（4.3）、`Identity`（4.4）、`CLSTrans`（4.5）。本文其他處的 ContextBlock 即 ContextEn，Pool 即 GlobalEn（§5.2）。

### 4.1 SelfAttn（Dense Self-Attention，預設）

單一 head $m$：

$$
Q^{(m)}=\mathrm{LN}(X)\,W_Q^{(m)},\quad
K^{(m)}=\mathrm{LN}(X)\,W_K^{(m)},\quad
V^{(m)}=\mathrm{LN}(X)\,W_V^{(m)}\qquad\in\mathbb{R}^{N\times d_h}
$$

$$
S^{(m)}=\frac{Q^{(m)}K^{(m)\top}}{\sqrt{d_h}}+B\ \in\mathbb{R}^{N\times N}
\qquad\text{attention logits}
$$

$$
A^{(m)}=\mathrm{softmax}_{\text{key}}\big(S^{(m)}\big)\ \in\mathbb{R}^{N\times N}
\qquad\text{attention map，}\textstyle\sum_jA^{(m)}_{ij}=1
$$

$$
O^{(m)}=A^{(m)}V^{(m)}\ \in\mathbb{R}^{N\times d_h}
\qquad\text{attention output}
$$

一層 Transformer block：

$$
Y=X+\mathrm{Concat}\big(O^{(1)},\dots,O^{(h)}\big)W_O
$$

$$
X_{\text{next}}=Y+\mathrm{FFN}\big(\mathrm{LN}(Y)\big)
$$

重複 $L$ 層：$X'=\mathrm{TransformerBlocks}_L(X)$。

| 參數 | 意義 | 初始值 |
|---|---|---|
| $L$ | Transformer block 層數 | 2 |
| $d$ | $W_{\text{in}}$ 投影後的 tile token 維度 | config 指定；須被 $h$ 整除。$d=d_0$ 時 $W_{\text{in}}$ 可省略 |
| $h$ | attention head 數（$d_h=d/h$） | 4 |

### 4.2 GNN（Explicit GNN）

$$
X\in\mathbb{R}^{N\times d},\quad
\text{edge\_index}\in\mathbb{N}^{2\times E},\quad
\text{edge\_attr}\in\mathbb{R}^{E\times d_e}
\ \xrightarrow{\ \text{GAT / GINE / Graph Transformer}\ }\ X'\in\mathbb{R}^{N\times d}
$$

- Edges：4-鄰接或 8-鄰接。
- edge\_attr：$(\Delta r,\Delta c,\ \lVert\Delta\rVert,\ \text{方向 one-hot})$。

### 4.3 SptSelfAttn（Spatially Masked Self-Attention）

與 4.1 相同，但：

$$
S^{(m)}_{ij}=
\begin{cases}
\dfrac{Q^{(m)}_iK^{(m)\top}_j}{\sqrt{d_h}}+B_{ij}, & (i,j)\in\mathcal{E}_{\text{spatial}}\\[4pt]
-\infty, & \text{otherwise}
\end{cases}
$$

### 4.4 Identity

$X'=X$。作為 ablation baseline。

### 4.5 CLSTrans

$$
Z^{(0)}=[\,c_{\text{CLS}};\,x_1;\dots;x_N\,],\qquad Z^{(L)}=\mathrm{TransformerBlocks}_L(Z^{(0)}),\qquad g=Z^{(L)}[0,:]
$$

- $c_{\text{CLS}}\in\mathbb{R}^d$ 可學習。自帶 tile 之間的 dense self-attention 與一個全域輸出 $g$，所以它取代 4.1 + GlobalEn，GlobalEn 只能是 `Identity`（直接取 $g$），不與 AttnPool 等串接，也不與 `SelfAttn` 串接。
- Method 名稱為 `CLSTrans-Identity`。

### 4.6 禁止

Dense self-attention 後再把同一張 $A$ 當 edge weights 送入 GNN。$AV$ 已是 dense message passing，兩者重複。

---

## 5. Architecture 1 — General-Token Retriever

### 5.1 資料流

```
FoV tiles(θ)  → TileTokenizer → X_F [N,d] → ContextEn → X'_F [N,d] → GlobalEn → g_F [d_g] → W_q, L2 → q [d_r]
Window tiles  → TileTokenizer → X_W [N,d] → ContextEn → X'_W [N,d] → GlobalEn → g_W [d_g] → W_k, L2 → k [d_r]
                                                                                                   ↓
                                                                           s = qᵀk / τ  → rank → Top-K
```

每個 $\theta$ 產生一個 $q^{(\theta)}$；$\theta\in\{0°,180°\}$ 只與 $H\times W$ windows 打分，$\theta\in\{90°,270°\}$ 只與 $W\times H$ windows 打分。

### 5.2 GlobalEn 選項（一次選一種）

| 選項 | 公式 | 可學習 |
|---|---|---|
| AttnPool（預設） | $e_i=w_p^\top\tanh(W_px'_i)$，$\alpha=\mathrm{softmax}_i(e)$（$m_i=0$ 設 $-\infty$），$g=\sum_i\alpha_ix'_i$ | $W_p,w_p$ |
| MaskedMean | $g=\dfrac{\sum_im_ix'_i}{\sum_im_i}$ | 否 |
| CNN | $G=\mathrm{reshape}(X')\in\mathbb{R}^{d\times H\times W}$，$g=\mathrm{AdaptiveAvgPool}(\mathrm{Conv2DBlocks}(G))$ | conv |
| GraphPool | 僅配 `GNN`（4.2）：graph attention pooling / mean | 依選項 |
| Align | 僅 Arch 1。不壓縮：不產生 $g$，逐 tile 對位比較，$s=\overline{\cos(x'^{F}_i,\,x'^{W}_i)}\,/\,\tau$（對 $m_i=1$ 的 tile 取平均）。只能窮舉，不能用 ANN | 否 |
| Identity | 僅配 `CLSTrans`（4.5）：直接取 $g=Z^{(L)}[0,:]$，不再 pooling | 否 |

- `CLSTrans` 屬於 ContextEn（4.5），不在 GlobalEn；與它搭配的 GlobalEn 只能是 `Identity`，不與 AttnPool 等串接。
- 禁止 sum pooling（分數會隨 $N$ 增大）。
- Method 名稱 `GloRet-{ContextEn}-{GlobalEn}`，例如 `GloRet-SelfAttn-AttnPool`、`GloRet-CLSTrans-Identity`。

### 5.3 打分

$$
q=\frac{W_q\,g_F}{\lVert W_q\,g_F\rVert},\qquad
k=\frac{W_k\,g_W}{\lVert W_k\,g_W\rVert}
$$

$$
s(F,W,\theta)=\frac{q^{(\theta)\top}k}{\tau}
$$

- 這一段是 bi-encoder（雙塔）cosine head：query 與 window 各自編碼，只經最後的內積互動，所以 $k$ 與 query 無關，可預先算好，並能吃整張 slide 的 window。
- $\tau$：可學習 scalar 或固定。
- 若不 L2 normalize，改用 $q^\top k/\sqrt{d_r}$。
- Inference：直接以 $s$ 排序取 Top-K；softmax over windows 只在訓練使用。

### 5.4 檢索方式

$$
\mathrm{Top\text{-}K}=\operatorname{argsort}_{c\in\mathrm{CandidateSet}_{\text{in}}}\ s(F,W_c,\theta_c)
$$

對 $\mathrm{CandidateSet}_{\text{in}}$ 每個 candidate 打分 → 排序 → Top-K（同 SlideWin）。Phase 1 的 $\mathrm{CandidateSet}_{\text{in}}$ 為全部 windows × 4θ。

計算 $k$ 的方式：

| ContextBlock / Pool | 計算全部 windows 的 $k$ |
|---|---|
| Identity + masked mean | box filter / conv 一次算完（同 SlideWin） |
| Self-Attention / GNN / CNN / AttnPool | 每個 window 各跑一次 |

可選加速：

- $k$ 只依賴 window，不依賴 query → 同一 slide / level / window shape 的 $k$ 可 cache，供多個 query 重用。
- ANN（HNSW / IVF / PQ）：只用於有壓縮的 GlobalEn（不含 `Align`）；只在 windows 數百萬以上或跨多張 slide 時使用；預設 exact matmul。`Align` 只能窮舉。

### 5.5 成本

- Reference：每個 window 需各自跑 ContextBlock（重疊 windows 的 context 不同，不能共用）→ $O(\#\text{windows}\cdot L\cdot N^2d)$，一次性，可 cache。
- Query：$4$ 次 $O(LN^2d)$。
- 打分：$O(\#\text{windows}\cdot d_r)$。

ContextBlock = Identity 時退化為 BagOfTiles 類方法；可作 Phase 1 下限 baseline。

---

## 6. Architecture 2 — Global + Tile Correspondence Scorer

### 6.1 資料流

```
FoV tiles(θ) → TileTokenizer → X_F [N,d] → ContextEn(shared) → X'_F [N,d] ─┬─→ GlobalEn → g_F ─┐
Window tiles → TileTokenizer → X_W [N,d] → ContextEn(shared) → X'_W [N,d] ─┼─→ GlobalEn → g_W ─┴→ s_global
                                                                              │
                                                                              └─→ Bidirectional Cross-Attention
                                                                                    F→W: S_FW, A_FW [h,N,N(+1)], C_FW [N,d]
                                                                                    W→F: S_WF, A_WF [h,N,N(+1)], C_WF [N,d]
                                                                                              ↓
                                                         Evidence Extractor → e_pair → MLP → s(F,W,θ)
```

Window 端也經過與 FoV 相同的 ContextBlock（共用權重），兩邊都先取得窗口內 context 才比較。

### 6.2 Global branch

$$
g_F=\mathrm{GlobalEn}(X'_F),\qquad g_W=\mathrm{GlobalEn}(X'_W),\qquad
s_{\text{global}}=\cos(g_F,g_W)\in\mathbb{R}
$$

- GlobalEn 為 §5.2 的選項，不含 `Align`（`Align` 不產生 $g$）。BiCrossAttn 與 GlobalEn 是平行的兩條分支，都接到 Evidence extractor。
- Method 名稱 `LogLoRet-{ContextEn}-{GlobalEn}`，例如 `LogLoRet-SelfAttn-AttnPool`。

### 6.3 Bidirectional Cross-Attention

FoV → Window（head $m$）：

$$
Q_F^{(m)}=\mathrm{LN}(X'_F)W_{Q,x}^{(m)},\quad
K_W^{(m)}=\mathrm{LN}(X'_W)W_{K,x}^{(m)},\quad
V_W^{(m)}=\mathrm{LN}(X'_W)W_{V,x}^{(m)}
$$

$$
S_{F\to W}^{(m)}=\frac{Q_F^{(m)}K_W^{(m)\top}}{\sqrt{d_h}}+B^{\times}\ \in\mathbb{R}^{N\times N}
$$

$$
A_{F\to W}^{(m)}=\mathrm{softmax}_{W}\big(S_{F\to W}^{(m)}\big)
$$

$$
C_{F\leftarrow W}=\mathrm{Concat}_m\big(A_{F\to W}^{(m)}V_W^{(m)}\big)W_{O,x}\ \in\mathbb{R}^{N\times d}
$$

- $S_{F\to W}[i,j]$：FoV tile $i$ 與 Window tile $j$ 的未正規化匹配分數。
- $A_{F\to W}[i,:]$：FoV tile $i$ 對所有 Window tiles 的 attention map（每個 FoV tile 一列，每個 head 一張）。
- $C_{F\leftarrow W}[i]$：FoV tile $i$ 從 Window 讀回的 context vector。
- $B^{\times}_{ij}=f_{\text{pos}}\big(\mathrm{pos}_W(j)-\mathrm{pos}_F(i)\big)$；query 已旋轉，座標已對齊。

Window → FoV：交換 $X'_F,X'_W$，以同一組權重 $W_{\cdot,x}$ 再算一次，得 $S_{W\to F},A_{W\to F},C_{W\leftarrow F}$。

### 6.4 No-match 處理（一次選一種）

| 選項 | 定義 |
|---|---|
| Dustbin（預設） | $\tilde S=[\,S\ \|\ z\mathbf{1}\,]\in\mathbb{R}^{N\times(N+1)}$，$z$ 為可學習 scalar；$\tilde A=\mathrm{softmax}_{\text{row}}(\tilde S)$ |
| Sinkhorn + dustbin | $\tilde S\in\mathbb{R}^{(N+1)\times(N+1)}$，以 Sinkhorn 迭代使 row/column 邊際約束成立，得近似一對一 assignment |
| Sigmoid pair | $P_{ij}=\sigma(S_{ij})$，每對獨立 0～1；需另加 coverage / calibration loss |

### 6.5 Evidence（每個 head 分別計算，再 concat）

以 $\pi(i)=i$，$\bar{\cdot}$ 表示對 $m_i=1$ 的 tiles 取平均。

| Evidence | 來源 | 公式 |
|---|---|---|
| Global cosine | 6.2 | $s_{\text{global}}$ |
| Aligned logit | $S_{F\to W}$ | $\mu_{\text{align}}=\overline{S_{F\to W}[i,i]}$ |
| Row margin | $S_{F\to W}$ | $\mu_{\text{margin}}=\overline{S[i,i]-\log\sum_{j\ne i}e^{S[i,j]}}$ |
| Bidirectional agreement | $A_{F\to W},A_{W\to F}$ | $\mu_{\text{bi}}=\overline{\tfrac12\big(\log A_{F\to W}[i,i]+\log A_{W\to F}[i,i]\big)}$ |
| Coverage | $A_{F\to W}$ | $c_j=\sum_iA_{F\to W}[i,j]$；$\text{cov}=\dfrac{\lvert\{\arg\max_jA[i,j]\}_i\rvert}{N}$，$\text{collapse}=\dfrac{\max_jc_j}{N}$ |
| Dustbin rate | $\tilde A_{F\to W},\tilde A_{W\to F}$ | $\rho=\overline{\tilde A[i,N{+}1]}$（兩方向各一） |
| Entropy | $A_{F\to W}$ | $\overline{-\sum_jA[i,j]\log A[i,j]}\,/\log N$ |
| Match feature | $C_{F\leftarrow W}$ | $u_i=\mathrm{MLP}_u\big([x'_i;\,C_i;\,x'_i\odot C_i;\,\lvert x'_i-C_i\rvert]\big)$，$u=\overline{u_i}\in\mathbb{R}^{d_u}$ |
| Upstream score（可選） | 上游 CandidateSet | $s_{\text{up}}$；Phase 1 無此項 |

低 entropy 只代表集中，不代表正確；必須與 aligned / margin / bi 一起使用。

上式的 overline（對 tile 取平均）是聚合的基本形；實作的 `evidence_agg` 有 `mean`／`attn`／`robust`，預設 `attn`：逐 tile 的證據與配對特徵各用一組可學的 softmax 權重（初始均勻，等於 `mean`）合成每個窗一個值。Head 這一軸不聚合，直接 concat。

### 6.6 打分

$$
e_{\text{pair}}=\mathrm{Concat}\big(s_{\text{global}},\ \mu_{\text{align}},\ \mu_{\text{margin}},\ \mu_{\text{bi}},\ \text{cov},\ \text{collapse},\ \rho,\ \text{entropy},\ u,\ [s_{\text{up}}]\big)
$$

$$
s(F,W,\theta)=\mathrm{MLP}_{\text{score}}(e_{\text{pair}})\in\mathbb{R}
$$

不直接 flatten $N\times N$ map 進 MLP（$N$ 可變且易 overfit）。

### 6.7 旋轉

- Phase 2 只對候選帶的那個 $\theta$ 打一次分；不為同一位置另算其他 $\theta$，也不比較不同 $\theta$ 的 map。
- 該 $\theta$ 是否可信由這一次的 $(S,A)$ 本身顯示：多數 rows 同時 diagonal logit 高、margin 高、雙向一致、dustbin 低、coverage 高 → 信心高；高值偏離 diagonal 或分散、collapse 高 → 信心低，分數由 §6.5 的 evidence 拉低。
- 位置對而 $\theta$ 錯的候選，與位置錯的候選一樣是 negative（§9.1）。

### 6.8 成本

- Query-dependent，reference 端無法 cache。
- 每個 candidate：ContextBlock $O(LN^2d)$ + cross-attention $O(2N^2d)$。
- 適合 Phase $n>1$（Top-100）；放 Phase 1 需窗口數 × $N^2$ 成本可承受。

---

## 7. 權重共享

### 7.1 FoV / Window 之間

| 模組 | 共享 | 規則 |
|---|---|---|
| Foundation encoder | 必須共享 | frozen；同一 feature space |
| TilePool 定義與權重 | 必須共享 | 兩邊同一種 tile token |
| Input projection $W_{\text{in}}$ | 共享 | 同一 token space |
| Domain LayerNorm / adapter | 可分開 | 低容量；query 由 WSI 合成時先共享 |
| 位置函數 $f_{\text{pos}}$ / $\phi$ | 必須共享 | 同一座標定義 |
| ContextBlock（$W_Q,W_K,W_V,W_O$、FFN 或 GNN） | 共享 | 分開易學到 domain shortcut |
| Pool（AttnPool / CNN / CLS） | 共享 | 保持 $g_F,g_W$ 可比較 |
| Arch 1 retrieval $W_q,W_k$ | 可分開 | 吸收小幅 domain asymmetry；無 $V$ |
| Cross-attention $W_{Q,x},W_{K,x},W_{V,x},W_{O,x}$ | 先共享（交換輸入跑兩次） | 有證據時再拆 F→W / W→F |
| Dustbin scalar $z$ | 兩方向可分開 | 兩個 scalar |
| Evidence MLP $\mathrm{MLP}_u$ | 共享 | 只用於 F→W；若加 W→F 用同一個 |
| $\mathrm{MLP}_{\text{score}}$ | 不適用 | 吃 pair evidence |

### 7.2 Phase 之間

| 模組 | 共享 |
|---|---|
| Foundation / TilePool | 可不同；Phase $n$ 可換 encoder、patch level 或 TilePool |
| ContextBlock / Pool | 可不同；Phase 2 可自帶 AttnPool，不必沿用 Phase 1 |
| 位置與旋轉協定（§2） | 必須一致 |
| Candidate identity | 必須一致 |

---

## 8. 可替換點總表

| 階段 | 預設 | 可替換 |
|---|---|---|
| Foundation | 現行 tile encoder（frozen） | 其他 pathology foundation model |
| TilePool | CLS | CLS‖mean（`cls_avg`）、patch AttnPool（`attnpool`） |
| 位置編碼 | relative bias MLP | 2D sinusoidal、2D RoPE、無 |
| ContextEn | SelfAttn ×2 | GNN、SptSelfAttn、Identity、CLSTrans |
| GlobalEn（Arch 1、Arch 2 的 global 分支） | AttnPool | MaskedMean、CNN、GraphPool（僅 GNN）、Align（僅 Arch 1）、Identity（僅 CLSTrans） |
| Arch 1 similarity | $\cos/\tau$ | $q^\top k/\sqrt{d_r}$ |
| Arch 1 search | exact matmul | HNSW、IVF、PQ |
| Arch 2 no-match | dustbin | Sinkhorn + dustbin、sigmoid pair |
| Arch 2 evidence 聚合 | masked mean over tiles | attention pool over tiles、robust mean |
| Phase 1 scorer | SlideWin（canonical）/ Arch 1 | 任何可負擔全 windows 的 scorer |
| Phase 2..n scorer | Arch 2 | Arch 1（不同權重）、現有 rerankers |

---

## 9. 訓練

### 9.1 樣本

- Candidate lists 取自實際上游輸出（Phase $n$ 用 Phase $n-1$ 的 Top-K）。
- FoV 會被轉四個旋轉（$0/90/180/270°$）去搜索；**其中恰有一個是正確旋轉**。標籤只由正確旋轉的結果決定，錯誤旋轉不產生 positive。
- 標籤只有三種（沒有 easy）：
  - **Positive**：正確旋轉下的 GT 窗（main 與 offset 兩個答案都算；名單裡沒有答案本身時，容許範圍 `tol_px` 內的窗代替）。
  - **Ignore**：以每個答案窗的原點為中心，往 row／col 各擴 `--ignore-expand` $\times$ FoV 窗的高／寬（預設 $0.5$，即半個 FoV 窗；窗越大範圍越大，邊界含在內）的方框內，**不是 positive** 的窗。不進 softmax、不進 loss、不拿到梯度。
  - **Hard negative**：其餘任何窗（不是 positive、不在 ignore 方框內）。
- **錯誤旋轉的窗**（包含 GT 窗本身，以及落在 ignore 方框內的窗）：`--wrong-rotation {ignore,negative}`，預設 `ignore`（超參數）。設成 `negative` 時，它們是 negative，不再被忽略。
- Negative：random、一般、semi-hard 混合；不只挑最難的。Phase 2 對每個候選只打一次分，錯的 $\theta$ 靠 evidence 被壓低。
- Slide-level split；另做跨資料集測試。

### 9.2 Loss

| Loss | 定義 | 用於 |
|---|---|---|
| Multi-positive listwise | $\mathcal{L}=-\log\dfrac{\sum_{c\in\mathcal{P}}e^{s_c}}{\sum_{c\in\mathcal{P}\cup\mathcal{N}}e^{s_c}}$ | Arch 1、Arch 2 |
| 近鄰軟目標（`--w-geo` 大於 0；`--soft-target`；`--sigma` 1 tile） | 開啟時，ignore 方框內、正確旋轉、不是 positive 的窗不再被排除，而是 near 窗，進入 softmax 的分母，答案的目標為 1。`gaussian`（`--near-alpha` 0.3）：近鄰窗的目標 $t_c=w\cdot\alpha e^{-d_c^2/2\sigma^2}$（$d_c$：窗中心到答案中心的距離，單位 tile），不是機率分佈，$\mathcal{L}=\big(\mathcal{L}_{\text{list}}-\sum_c t_c\log p_c\big)/\big(1+\sum_c t_c\big)$（RerankHead 的 WinEmbed 用的形式）。`exponential`：目標是機率分佈 $\propto e^{-d/\sigma}$（沒有平方），在正確旋轉、在 softmax 裡的窗上正規化，$\mathcal{L}=\mathcal{L}_{\text{list}}+w\,\mathrm{CE}(\text{target},\mathrm{softmax})$。錯旋轉的窗不拿軟目標 | Arch 1、Arch 2 |
| Correspondence | positive pair 上 $-\overline{\log A_{F\to W}[i,i]}$（雙向） | Arch 2 |
| Cycle consistency | $-\overline{\log\,(A_{F\to W}A_{W\to F})[i,i]}$ | Arch 2 |

- 不使用嚴格 triplet + hardest negative mining 作主 loss。
- Multi-positive listwise 在只有一個答案類別時就是 one-hot 的 cross-entropy：對 $\mathrm{softmax}_c\{\cos(q,k_c)/\tau\}$ 取 $-\log$ 答案窗的機率（InfoNCE／listwise）。幾何項關掉（`--w-geo 0`，預設）時，loss 就是這個 cross-entropy，在非 ignore 的候選上算；ignore 的窗不在 softmax 裡，也沒有梯度。
- Foundation frozen；可訓練部分限於 TilePool（若選 AttnPool）、projection、ContextBlock、Pool、cross-attention、MLPs。

### 9.3 選用輔助 loss（消融用；預設權重 0，不影響 9.2）

兩者都是 9.2 之外的額外項，只在要檢驗「壓縮是否丟資訊」「輸出空間是否擠在一起」時開。

**(a) 還原（AE／VAE）**：每個**壓縮維度**的投影（$d_{\text{in}}>d_{\text{out}}$，例如 $W_{\text{in}}$ 1536→512、$W_Q/W_K$ $d\to d_r$）都配一個同架構、只在訓練時使用的 decoder $D$（鏡像的線性層 $d_{\text{out}}\to d_{\text{in}}$），還原該投影的（detached）輸入：

$$
\mathcal{L}_{\text{recon}}=\overline{\,1-\cos\big(D(z_i),\ x_i\big)\,},\qquad z_i=W x_i
$$

- `--recon {'',ae,vae}`；`--w-recon`、`--w-kl` 都是超參數，預設 0（不開）。
- `ae`：$D$ 吃 $z$。`vae`：投影另輸出 $\log\sigma^2$，$D$ 吃 $z+\sigma\epsilon$，並加 $\mathcal{L}_{\text{KL}}=\mathrm{KL}\big(\mathcal{N}(z,\sigma^2)\,\|\,\mathcal{N}(0,I)\big)$。網路往後傳的永遠是 $z$，所以輸出與模式無關。
- **沒有壓縮的地方（$d_{\text{in}}\le d_{\text{out}}$，如 $d_r=d$）或沒開模式，decoder 與這項 loss 自動不存在**；評估時（eval／no-grad）也不產生。
- 全網路的值是各壓縮點的加總；權重取小（約 0.1）：還原與辨別方向不完全一致，權重大會犧牲 R@k。
- 注意：`TileEmbed.proj` 現在是 `Decoded` 模組，state-dict 的鍵由 `embed.proj.weight` 變成 `embed.proj.proj.weight`，舊 checkpoint 不相容（沒有正式 checkpoint 受影響）。
- 對照基線：固定的 PCA／白化投影（不訓練）。先看它與學出來的 $W_{\text{in}}$ 差多少，再決定是否值得加 decoder。
- 目標：TilePool 為 `attnpool` 時，$x_i$ 是池化後的 tile 向量。

**(b) 均勻性（uniformity）**：搜索要認出獨一無二的 input，所以不同的 input 在輸出空間應該盡量攤開；但近似的 input（同樣是脂肪的 tile、彼此重疊的 window）可以靠在一起：群內要均勻、群與群之間要分開，而不是全部壓成一點。

最接近的既有 loss（已讀原文）：

| Loss | 定義 | 出處 |
|---|---|---|
| KoLeo | $\mathcal{L}_{\text{KoLeo}}=-\frac{1}{n}\sum_i\log d_{n,i}$，$d_{n,i}=\min_{j\ne i}\lVert x_i-x_j\rVert$；先 $\ell_2$ 正規化；由 Kozachenko–Leonenko 微分熵估計而來，目的是在 batch 內均勻展開特徵 | Sablayrolles et al. 2019 (arXiv 1806.03198)，為相似度搜尋設計；DINOv2 (arXiv 2304.07193, §4) 沿用 |
| Uniformity | $\mathcal{L}_{\text{unif}}=\log\mathbb{E}_{x,y}\,e^{-t\lVert f(x)-f(y)\rVert_2^2}$（超球面上的 Gaussian potential 平均的 log） | Wang & Isola 2020 (arXiv 2005.10242)，與 alignment 項 $\mathbb{E}\lVert f(x)-f(y)\rVert^\alpha$ 成對 |

兩者都有的問題：它們把**所有**點推開，包括本來就該重疊的近似 input（相鄰 window 共用大部分 tile，它們的向量本該接近，這與 9.2 的 soft geometry 衝突）。KoLeo 只看最近鄰、是局部的，所以不會強迫群聚散開到全域均勻，這點符合「群內均勻」；但它在 $d\to 0$ 時懲罰趨近無限，對近似 input 的衝突較大。本 proposal 的做法是 **KoLeo 加上 input 感知的排除集**：

$$
\mathcal{L}_{\text{unif}}=-\frac{1}{n}\sum_i\log\min_{j\notin\mathcal{E}(i)}\lVert z_i-z_j\rVert,\qquad
\mathcal{E}(i)=\{\,i\}\cup\{\,j:\ \text{window }j\text{ 與 }i\text{ 共用 tile，或 }\cos(\bar f_i,\bar f_j)\ge\tau_{\text{in}}\,\}
$$

- $z_i$：Arch 1 的 window 向量（$\ell_2$ 正規化後），在一個 step 抽到的 window 之間算；$\bar f_i$：window 內 tile 特徵的平均，$\tau_{\text{in}}$（約 0.95）以上視為「近似 input」，可以重疊。
- 群與群之間的分離不由這項保證，仍由 9.2 的 listwise 負例項負責；這項只防止群內塌縮。
- 權重 $\lambda_{\text{unif}}$ 取小；以 R@500 與訓練／驗證差距檢驗，不預設開啟。

### 9.4 TilePool = `attnpool` 的訓練流程

patch token 約 0.6 MB 一個 tile（196 × 1536 × fp16），不能像 `cls`／`cls_avg` 那樣把整個 level 快取，所以與其他 arm 的流程不同，獨立成三個階段（`deploy/slurm/GloRet/GloRetAttnPool.sh`）：

1. **階段 A**：只在粗 level（ds $\ge 4$）訓練，patch token 放 host 記憶體；GloRet 與 LogLoRet。
2. **階段 B**：目的是**訓練不用看整個細 level**。細 level（ds $\le 2$）上，LogLoRet 以 SlideWin 的 top-10000 名單、GloRet 以 `@items=list` 當 reranker 繼續訓練（起點為 A 的權重）；池化層一併微調。level 是 lazy 的（`Lazy.LazyLevel`）：只在某個窗被用到時才讀並編碼它底下的 tile。pool 的選法與其他階段**完全相同**（§9.5）：warm-up 期間由 teacher（名單順序）選，之後（或沒有 warm-up）由網路自己對名單條目評分來選；評分時讀到的 tile 只用一次就丟（不保留），被選中的 pool 的 tile 讀一次並保留到這個 visit 結束。名單裡沒有正例的 FoV 在 lazy level 上不補答案（答案的 token 沒有讀），整個跳過。
3. **階段 C**：凍結 B 學到的池化層，每個 tile 池化成一個向量，所有 level 編碼一次並快取，其餘各 head 像 `cls_avg` 一樣在所有 level 上訓練。

### 9.5 Softmax 的候選：`--top-p`、候選池、episode 與 hard-negative 記憶表

**候選（item）**：一個 FoV 的 softmax 可能看的東西，由 head 的 `@items=` 決定（預設：GloRet `level`、LogLoRet `list`）。
- `level`：這個 level 的**所有窗**（四個旋轉、各形狀，每個窗帶第一階段 SlideWin 的分數 `window_scores` 當 teacher）；
- `list`：別的 retriever 給的**名單**的條目（`--raw-topk` 個，依該 retriever 的順序，順序當 teacher 分數）。
- **名單裡沒有任何正例**（沒有答案本身，也沒有占住它位置的窗）時，**把答案的窗加進 item**（排在最後，teacher 最差），FoV 不跳過；log 的 `injected` 是被補進答案的 FoV 比例（＝名單涵蓋不到的比例）。驗證不補：沒有命中的 FoV 仍算 miss。`--on-demand`（階段 B）讀不到補進去的窗的 token，不補，沒有正例的 FoV 仍跳過。
- 打分器與 item 來源是兩件事：GloRet（向量）和 LogLoRet（pair）都可以對 `level` 或 `list` 訓練。pair 打分器對 `level` 排不了名（每個窗一對，約 5×10^5 對），只能用 `episodes`、pool 永遠由 teacher 選（`--top-p` ≤ 0.5）。
- 角色由訓練方式（`@items`）決定，不看架構：`@items=level` 是 retriever，驗證讓網路自己列出整個 level 的窗（R@500）；`@items=list` 是 reranker，驗證重排第一階段名單的前 `--val-keep` 個（R@10）。兩個架構都可以當任一種；pair 打分器當 retriever 時對 level 每個窗各算一對，池在 warmup epoch 用第一階段的分數選，之後用自己的分數選。

**`--top-p`**：softmax 實際看多少個 item，答案與近鄰窗另外加上（K+1）。
- 帶小數點的數（`0.05`、`1.0`）是**比例**，`1.0` 就是全部 item；整數（`2000`）是**個數**；兩種都可以給任何 head（`Name@top=…`），job 的 `--top-p` 是預設：比例給 GloRet、個數給 LogLoRet（兩種 head 都有時各給一個）。沒有預設值，必須指定。
- `--min-candidates`（預設 256）：比例太小的小層級，至少選這麼多 negative（個數不受影響）。

**三種情形**（依比例；整數取 K/名單長度）：

| 比例 | 候選池 |
|---|---|
| `1.0`（或 K 涵蓋全部 item） | 全部；沒有排名、episode、記憶表 |
| `> 0.5` | 每一步用網路**當下**的分數取前 K（每步對所有 item 打分）；不用 teacher、episode、記憶表 |
| `≤ 0.5` | 每個 **episode** 開頭選一次，在這個 episode 內固定：前 `--warmup-epochs` 個 epoch 用 teacher（第一階段 SlideWin）的分數，之後用網路自己的分數；有記憶表 |

一個 visit（一個 level、一批 FoV）的 `--steps-per-visit` 步分給 `--episodes` 個 episode，每個 episode 內 FoV 不變。

**候選池怎麼選**（`Pool.py`，對每個 FoV）：排在**答案前面**的 negative（分數高過最好的答案）都是 hard negative，共 N' 個，由近到遠取前 K 個；不足 K 時，先補記憶表裡不重複的成員，再補其餘分數最高的。答案與（開啟軟目標時的）近鄰窗一定在池裡。ignore 的窗不會被選。

**記憶表**：每個 FoV 一份，記錄贏過答案的 negative 與差多少；答案打敗某個 negative 就從記錄刪除，它再贏過答案時再記回來。注意：當每個 episode 開頭對**所有** item 重新排名時，記憶表的成員剛好就是「排在答案前面的 negative」，候選池不需要它；它在排名漏掉某些 item 時才有作用。

**成本**：要排名就必須對所有 item 打一次分（不建圖），這一步的成本與 P 無關；P 小只省第二趟（帶圖）。所以 episode 才省時間：整個 level 的排名只在 episode 開頭做一次，之後的步驟只編碼候選池。`0.5<P<1` 每一步都排名，幾乎沒有好處。

**實測**（`ExpGloRetFull`，`GloRet-SelfAttn-AttnPool`、gigapath cls、ds1 約 29 萬到 38 萬個窗）：一張卡加 Gradient Cache、`--top-p 1.0`，每個 visit 約 13 到 15 秒、GPU 峰值 16 GB；2 張一組約 9 到 11 秒；3 張一組不加 cache 在 38 萬個窗的 level OOM（每張卡約 12.7 萬個窗，134 GB 已分配）。一個 epoch 的時間由編碼 level（約 670 tiles/s）決定。

**已移除**：`--candidates pool`（隨機池挖掘）、每步抽樣的 `--n-cand`、`--hard-share`、`--mem-share`、`--mem-cap`、`--keep`、LogLoRet 的 `--n-extra`（名單以外的隨機 negative）。

**尚未實作**：KoLeo 均勻性（§9.3 b）與 effective rank 診斷；LogLoRet 的多個名單來源（bench 的各種第一階段）。

### 9.6 Gradient Cache 與多卡拆窗

候選池裡的窗（GloRet）或配對（LogLoRet）可能有幾十萬個，反向傳播留不下所有中間結果（每窗約 1.3 MB）。兩個辦法，可以併用：

- **Gradient Cache**（Gao et al. 2021，arXiv 2101.06983；預設）：
  1. 候選分批、不建圖地編碼（GloRet）或打分（LogLoRet），只留向量 $K\in\mathbb{R}^{W\times d_r}$ 或分數 $s_i$；
  2. 用它們與 query 算 loss，只對 $K$ 或 $s$ 反向，得到 $\partial\mathcal{L}/\partial K$ 或 $\partial\mathcal{L}/\partial s_i$；
  3. 每一批再編碼一次（這次建圖），帶著那一批的梯度反傳；一步更新一次。
  - 梯度與「一次算完」相同（dropout 在兩次用同一個種子）；額外成本是多一次不建圖的前向。
  - LogLoRet 的 correspondence 與 cycle loss 只作用在答案配對（很少），在第二趟直接帶圖算。
- **多卡拆窗**（`--window-split G`）：G 張卡為一組處理**同一個 level**（同一批 FoV、同一份候選池），各負責 $1/G$ 的候選，all-gather 向量（GloRet）或分數（LogLoRet，每個一個數）後各算同樣的 loss，各自只對自己那份反傳，組內把參數梯度相加；不同組處理不同的 slide。`--no-gradcache` 讓每張卡的那份一次帶圖編碼（少一次前向，記憶體約為整份的 $1/G$）。
- **一個 step 的成本是它的候選數的成本，與 FoV 數無關**，所以每個 visit 取很多 FoV（`--fovs-per-step`）、很少的步數（`--steps-per-visit`）。
- **TilePool = `attnpool`**：每個 tile 的 patch token 只池化一次（葉節點），窗從池化後的網格取；窗的梯度先累加在這些葉節點上，再逐 tile 重算池化並反傳到池化層（兩層的 Gradient Cache）。
- **`Align`**：窗保留逐 tile 的向量（每 tile 正規化後展平），分數為一次內積除以 tile 數。
- **近鄰軟目標**（`--w-geo`、`--soft-target {gaussian,exponential}`）見 9.2：ignore 方框內、正確旋轉的窗拿到軟目標並進入 softmax 的分母。

---

## 10. 實驗順序

1. 固定 CandidateSet protocol：同一 scorer 可接全 windows 或上游 Top-K。
2. Arch 1：ContextBlock = Identity vs Dense Self-Attention；Pool = masked mean vs AttnPool vs CNN；TilePool = CLS vs CLS‖mean。
3. Arch 1 作 Phase 1，對照 SlideWin 的 $R@100$。
4. Arch 2 作 Phase 2：逐項加入 evidence（global → aligned/margin → bi → coverage/dustbin → match feature）。
5. Exact vs ANN（僅 Arch 1）。
6. 基本組合（每個 job 的指令記在 `deploy/slurm/GloRet/GloRet.sh` 開頭）：
   - attnpool 單獨（gigapath，`GloRet-SelfAttn-AttnPool`，三階段）。
   - $d$ 消融（gigapath `cls`，`GloRet-SelfAttn-AttnPool`）：$d=d_r=1024$ 與不壓縮（$d=d_r=1536$，AE／VAE 自動失效）。
   - {uni2, gigapath} × {`cls`, `cls_avg`, `grid2x2`}：每格 `GloRet-SelfAttn-AttnPool`（搜尋後端 `exact`／`hnsw`／`ivf` 不重訓）與 `LogLoRet-SelfAttn-AttnPool+dustbin`。
   - gigapath `cls`、`exact`：GloRet 的 `SelfAttn`×{`MaskedMean`, `CNN`, `Align`}、{`SptSelfAttn`, `Identity`}×{`AttnPool`, `CNN`, `Align`}、`GNN-GraphPool`、`CLSTrans-Identity`；LogLoRet 的 `SelfAttn-AttnPool`×{`sinkhorn`, `sigmoid`}。
   - LogLoRet 的其他 ContextEn／GlobalEn 與 `evidence_agg` 的其他選項（`mean`／`robust`）：先不做。`evidence_agg` 預設 `attn`（每個 channel 學一組在 tile 上的 softmax 權重，初始為均勻，即從 `mean` 起步）。
7. 輔助 loss 與標籤的消融（在 6 的最佳組合上，一次改一項，其餘用預設；每項是一個超參數）：

   | 項目 | 旗標 | 比較 |
   |---|---|---|
   | 還原（§9.3 a） | `--recon {'',ae,vae}`、`--w-recon`、`--w-kl` | 關（預設）／AE／VAE；權重約 0.01／0.1／1；只在有壓縮處有效，$d=d_r=1536$ 時自動失效 |
   | 均勻性（§9.3 b，KoLeo 加排除集） | 尚未實作（`--w-unif`） | 先看 effective rank 與最近鄰距離確認有攤縮，再比 關／小權重 |
   | 錯旋轉的窗（§9.1） | `--wrong-rotation {ignore,negative}` | ignore（預設）／negative |
   | ignore 範圍（§9.1） | `--ignore-expand` | 0（只排除答案本身）／0.25／0.5（預設）／1.0 |
   | 一次看的窗（§9.5） | `--top-p`、`--min-candidates` | GloRet：0.01／0.05／0.5／1.0（所有窗）；LogLoRet：256／2000／名單全部 |
   | episode 與 warm-up（§9.5） | `--episodes`、`--warmup-epochs` | 1／3（預設）／6；0／2（預設）／5 |
   | 近鄰軟目標（§9.2） | `--w-geo`、`--soft-target {gaussian,exponential}`、`--near-alpha`、`--sigma` | 0（關，預設，ignore）／1；gaussian 與 exponential；α 0.1／0.3／0.6 |

   每項要報 R@1／R@10／R@500，並依 ds 層級拆開（ds≥4 的 R@500 飽和，以 R@1／R@10 比較）。

   **怎麼比**：一個設定就是一個 head（`Name@w-geo=1.0@sigma=2`，旗標見 `train.py` 的 `HEAD_FLAGS`；§9.5 的 `@items`、`@top`），同一個 job 的所有 head 吃同一批 FoV、同一份 level 編碼、同一個驗證集，所以是配對比較。`deploy/slurm/GloRet/ExpGloRetScreen.sh` 的 `ROUND=1` 是候選與 `--top-p`，`2` 是標籤，`3` 是溫度、學習率、寬度、decoder 與 LogLoRet 的輔助 loss，`4` 是上面 gigapath `cls` 的十五個 arm；規模：每個資料集 20 張 slide、4 個 epoch、驗證每個 visit 32 個 FoV（n 夠大才分得出幾個百分點的差）。2 到 4 輪用第 1 輪贏的 `--top-p`（`G_TOP`、`L_TOP`）。attnpool 不在這些 job 裡（token 放不進），它的實驗是 `GloRetAttnPool.sh` 的縮小版（`JOB=ExpGloRetAttnPool`），階段 C 的設定與第 4 輪相同，才有可比性。

---

## 11. Arm 名稱

`Method::TilePool[+op]`，與 `stage2_retrieval/ArmSpec.py` 的文法相同。

| 架構 | Method | `TilePool`（pooling 欄位） | `+op` |
|---|---|---|---|
| Arch 1 | `GloRet-{ContextEn}-{GlobalEn}` | `cls` / `cls_avg` / `cls_std` / `grid2x2` / `attnpool` | 搜尋後端 `exact` / `hnsw` / `ivf`（僅有壓縮的 GlobalEn；`Align` 沒有 `+op`） |
| Arch 2 | `LogLoRet-{ContextEn}-{GlobalEn}` | 同上 | no-match：`dustbin` / `sinkhorn` / `sigmoid` |

合法的 (ContextEn, GlobalEn)：

| ContextEn | GlobalEn |
|---|---|
| `SelfAttn`、`GNN`、`SptSelfAttn`、`Identity` | `AttnPool`、`MaskedMean`、`CNN`、（Arch 1 另加）`Align` |
| `GNN` | 另加 `GraphPool` |
| `CLSTrans` | 只能 `Identity` |

- 位置編碼、相似度（$\cos/\tau$ 或 $q^\top k/\sqrt{d_r}$）、evidence 聚合、$W_{\text{in}}$、domain LN 等可替換點不進名稱，記在 checkpoint 設定與 identity。
- 搜尋後端與 no-match 以外的 `+op` 不存在；兩者都不影響訓練出的權重的，搜尋後端不需重訓。
- 兩個架構都可站在 retriever 或 reranker 的位置：`ArmSpec.MethodInfo` 要能同時標記兩種身分。
