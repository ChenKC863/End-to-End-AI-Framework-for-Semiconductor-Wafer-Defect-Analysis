# Query limits and summaries / 查詢上限與摘要

The query API permits restricted read-only SELECT statements on wafers. It returns at most 200 rows. LIMIT 5 selects five rows at the SQL layer. truncated: false means the backend did not additionally truncate that SQL result at its return limit. It does not mean only five rows match the filters.

摘要僅使用回傳筆數、分數範圍及最多前五筆精簡樣本。完整影像路徑仍保留於結果表格。摘要失敗時，原查詢 API 保留 SQL 與資料並顯示 warning。資料集統計不能稱為產線良率。

Source: wafer_llm_query/react_api.py, read_query, summary_evidence, and query.
