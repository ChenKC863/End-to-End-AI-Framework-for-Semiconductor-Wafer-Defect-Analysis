# Wafer database fields / 資料欄位

The wafers table contains id, image_path, split, true_label, pred_label, pred_prob, anomaly_score, and is_anomaly. true_label is the ground-truth category. pred_label is the classifier prediction. split identifies train, valid, or test. pred_prob is classifier confidence, not the anomaly score. Dataset S and M are separate SQLite files.

true_label 是真實類別，pred_label 是預測類別。問題同時要求兩種條件時，必須分別篩選兩個欄位。最低異常分數使用 ASC 排序。只要求分數排名時，不額外加入 is_anomaly = 1。統計筆數或影像排名必須查詢 SQLite，文件檢索不提供即時筆數。

Source: wafer_llm_query/react_api.py, SYSTEM_PROMPT and DATABASES. Retrieve actual counts through the SQL tool instead of treating documentation as current data.
