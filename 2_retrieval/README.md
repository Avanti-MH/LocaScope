## 2_retrieval

### 原則（stage module = 可重用純邏輯）

- **資料流**：Input（data + config）→ Output（結果 + metadata）
- **不做的事**：CLI 解析、print/plot、讀寫結果檔、挑最後候選人的 heuristics（這些放外層 main / orchestrator）
- **輸出要明確**：不要只回傳 tensor/array；請回傳結構化結果（dataclass）

### 目標（這個階段要產出什麼）

- **Similarity / heatmap**：query 對 reference grid 的相似度圖
- **Top‑K ranking**：從相似度圖導出的 top‑k 結果（含座標、分數、level 等 metadata）

> 這個資料夾目前尚未整理出正式 API；後續新增 retrieval 方法時請遵守以上原則。

