# План робіт ML_Labs

Робочий файл для послідовного виконання 5 лабораторних. ТЗ — локально в `docs/labs/` (не в git).

## Ключові рішення (незворотні після ЛР1)

| Параметр | Значення | Статус |
|---|---|---|
| Домен | E-commerce (прогноз відтоку) | ✅ |
| Датасет | `data/raw/E Commerce Dataset.xlsx`, аркуш `E Comm`, 5630 × 20, відтік 16.8% | ✅ |
| Захищені ознаки | `[CityTier, Gender]`: CityTier — основна (метрики справедливості), Gender — архітектурно виключена; PreferredLoginDevice — слабкий проксі (у features); **PreferredPaymentMode — сильний проксі (V=0.44)**, лишився, під моніторингом | ✅ |
| Схема `model_family` | `additive` / `partitioning` | ✅ |
| Штучні пропуски | MCAR seed 42: CashbackAmount 20%, PreferedOrderCat 15% (`config/data.yaml`) | ✅ |
| Евристика | `Tenure <= 1 OR Complain == 1` → Acc 0.699 / Rec 0.848 / Prec 0.341 | ✅ |
| ML Metric | Recall ≥ 0.80 при Precision ≥ 0.50; ROC-AUC ≥ 0.85 | ✅ |
| Трек ЛР4 | **4A** (semi-supervised + active learning); 4B відпав — немає `make_prod_windows.py`; 4D (LLM/RAG, Ollama) — можливо після ЛР5 | ✅ |
| Python | 3.11.9 у `.venv`, kernel `ml-labs`; pandas 2.3.3 (ЛР2: featuretools не працює з pandas 3) | ✅ |

## Git

- Кожна ЛР — окрема гілка (`lab1` … `lab5`).
- ЛР1: тег `lab1-v1.0` (згадано в README, REFORMS 2b).
- ЛР2: тег `lab2-v1.0`.
- ЛР3: тег `lab3-v1.0` — еталонний DVC-знімок для ЛР4 (`dvc.api.open('data/processed/train.csv', rev='lab3-v1.0')`).
- ЛР4: тег `lab4-v1.0`; ЛР5: тег `lab5-v1.0`.
- Звіти: `docs/labN_report.md`; README — «обкладинка» з таблицею «артефакт → файл».

## Прогрес

- [x] ЛР1 — здано (тег lab1-v1.0)
- [x] ЛР2 — здано (тег lab2-v1.0)
- [x] ЛР3 — здано (тег lab3-v1.0)
- [x] ЛР4 (трек 4A) — здано (тег lab4-v1.0)
- [x] ЛР5 — готово, чекає коміту/тегу `lab5-v1.0` і захисту

## Нотатки для наступних ЛР

- ✅ ЛР2: дублікати прибрано (`config/data.yaml: deduplicate`), 5 074 рядки; OOF-WoE реалізовано.
- ЛР3: прибрати витік Експ. 4 (TF-IDF/SVD, WoE, Filter/SFS — лише у внутрішніх фолдах, imblearn pipeline, Nested CV); чесний поріг; довірчий DPR/EOR;
  перевірити проксі `PreferredPaymentMode` і кешбек-ознаки; вирішити долю мультимодального блоку (неінформативний).
- ЛР3: природні пропуски залежать від класу (Tenure 4% vs 8.7%) — MAR/MNAR, індикатори можуть бути корисні.
- Мультимодальний корпус і фото — кеш у `data/raw/amazon` (37 МБ), прив'язка — `data/processed/reviews_linked.csv`.
- Особисті чеклісти до захисту — `notes/` (не в git).

## ЛР3 → ЛР4/ЛР5
- Дані: `data/processed/train.csv` (4059, розробка) і `holdout.csv` (1015, недоторканний) — виходи стадії DVC `prepare`.
- Фінальна модель: `models/lab3_final_pipeline.joblib` (не в git), метадані й поріг — `reports/lab3_final_model.json`.
- Матриця втрат — `config/costs.yaml`. Вкладена CV — `python -m src.run_nested --model LR|RF`, `--final`.
- DVC remote не налаштовано (локальний кеш). Для іншої машини — налаштувати remote або завантажити дані заново.
- ONNX (ЛР5): кастомні трансформери (OOF-WoE, ReviewTextSVD, конструктори) не мають конвертерів skl2onnx — граф ONNX
  ймовірно покриє лише частину конвеєра, решта — препроцесинг поза ONNX (ТЗ ЛР5 це допускає: «межу ONNX-графа задокументовано»).

## ЛР4A → ЛР5 (трековий модуль «Weak & Active Learning»)
- Міра uncertainty — ентропія; стратегія черги — entropy + diversity (`reports/lab4a_strategy.json`).
- Пул дорозмітки — `data/processed/lab4a_split.csv` (role == pool), не перетинається з test ЛР4A і holdout ЛР3.
- Self-Training на цих даних не допоміг (негативний результат, задокументовано); AL дає +0.025 ROC-AUC до random.

## ЛР5 — рішення (не прямо з ТЗ)
- Стартового шаблону від викладача немає → структура за розділом 11 ТЗ; `src/serving/` — препроцесинг поза ONNX (замість `src/features/`,
  бо `src/features.py` уже існує з ЛР2).
- Train v1 = pool ЛР4A (3 044), пул дорозмітки = test ЛР4A (1 015), test ЛР5 = holdout ЛР3 (1 015) — без перетинів.
- Ознаки відгуків ЛР2 у сервісі не використовуються (у CRM на момент запиту їх немає) → 15 табличних ознак, гіперпараметри ЛР3.
- У ONNX лише RandomForest; препроцесор — `models/preprocessor.skops` (кастомні трансформери без конвертерів skl2onnx).
- OOD: PCA за правилом ТЗ (≥ 0.9 дисперсії, ≤ 10 компонент) → KDE, поріг — 1-й перцентиль на відкладених 20% train.
- Фінальні артефакти `models/` комітяться (≈ 24 МБ), `mlflow.db`, `mlruns/`, `feedback.db`, `models/challenger|random_baseline` — ні.
- Ліміт пакета 256 (як у ТЗ і для заміру p50/p95).
- Docker Compose: окремий MLflow-сервер, реєстр наповнює `deploy/seed_registry.py` з `models/` (номер версії там 1).
- Слайди — PowerPoint (`docs/slides/ChurnGuard_defense.pptx`) + PDF (`reports/defense.pdf`).
