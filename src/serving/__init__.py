"""ЛР5 — обробка поза ONNX-графом: від полів запиту API до входу ONNX.

Межа графа (README, «Межа ONNX-графа»):
  поля запиту → [Python: src/serving] DFS-ознаки клієнта → препроцесор ЛР1–3 (конструктори, AdaptiveImputer,
  OOF-WoE, вінсоризація, one-hot) → SelectKBest → обрані SFS колонки → [ONNX] RandomForest → P(відтік).
"""
