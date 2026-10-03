# Anomaly score interpretation / 異常分數

This project stores class-specific Isolation Forest decision_function scores as anomaly_score. Within the same trained class-specific model, lower scores indicate greater model-assessed abnormality. A score below zero indicates an anomaly. Scores are not probabilities or measurements of physical defect severity. Do not compare severity directly across predicted classes or EfficientNetV2-S and M models.

本專案 anomaly_score 是各類別 Isolation Forest 的 decision_function 分數。同一個已訓練類別模型內，分數越低越異常，零為判定門檻。分數不是異常機率，也不能直接比較不同類別或 S/M 模型的實體缺陷嚴重程度。

Source: Doc1.docx, Slide 46A score interpretation; the-defect-analysis-of-wafer.ipynb anomaly scoring pipeline. This note describes the project, not a manufacturing diagnosis.
