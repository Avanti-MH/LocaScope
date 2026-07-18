## 3_localization

### 原則（stage module = 可重用純邏輯）

- **資料流**：Input（data + config）→ Output（結果 + metadata）
- **不做的事**：CLI 解析、print/plot、讀寫結果檔、挑最後候選人的 heuristics（這些放外層 main / orchestrator）
- **輸出要明確**：不要只回傳 tensor/array；請回傳結構化結果（dataclass）

### 目標（這個階段要產出什麼）

- 把 retrieval 的 heatmap / top‑k 結果轉成 **可定位的候選 ROI**（boxes/points）與分數
- （可選）做 refinement（例如多尺度融合、NMS、mask 過濾等），但 **策略/heuristics 建議留在外層 orchestrator**

> 這個資料夾目前尚未整理出正式 API；後續新增 localization 方法時請遵守以上原則。

