# ML_Labs — прогнозування відтоку клієнтів (E-commerce)

Наскрізний навчальний проєкт з дисципліни "Прикладне Машинне Навчання": 5 лабораторних робіт за методологією **CRISP-ML(Q)** і стандартом **REFORMS**.
Кожна наступна робота продовжує попередню.

- **Домен:** E-commerce — прогнозування відтоку (churn)
- **Датасет:** [Ecommerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) (Kaggle) — 5 630 клієнтів (5 074 після видалення дублікатів), 20 колонок, відтік ≈ 17%
- **Захищені ознаки:** `CityTier` (основна — географія, рівень міста) і `Gender` (архітектурно виключена з моделі)
- **Неструктуровані дані (з ЛР2):** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io) — тексти відгуків і фото покупців

## Лабораторні роботи

| ЛР | Тема | Гілка / тег | Звіт | Статус |
|---|---|---|---|---|
| 1 | Системний дизайн за CRISP-ML(Q) та динамічний препроцесинг-конфігуратор | `lab1` / `lab1-v1.0` | [docs/lab1_report.md](docs/lab1_report.md) | ✅ |
| 2 | Мультимодальна інженерія та відбір ознак | `lab2` / `lab2-v1.0` | [docs/lab2_report.md](docs/lab2_report.md) | ✅ |
| 3 | Герметизація пайплайну: ліквідація витоку, аномалії, дисбаланс, Nested CV | `lab3` / `lab3-v1.0` | [docs/lab3_report.md](docs/lab3_report.md) | ✅ |
| 4 | Трек 4A: semi-supervised і active learning | `lab4` / `lab4-v1.0` | [docs/lab4_report.md](docs/lab4_report.md) | ✅ |
| 5 | Реліз PoC-системи ChurnGuard: ONNX, MLflow, FastAPI, XAI, документація (трек 4A) | `lab5` / `lab5-v1.0` | [docs/lab5_report.md](docs/lab5_report.md) | ✅ |

Кожна лабораторна робота виконується в окремій гілці. Звіт кожної ЛР — окремий файл у `docs/`.

### Де що лежить

| Артефакт | ЛР | Файл |
|---|---|---|
| ML Canvas, Non-ML бейзлайн, захищені ознаки, схема `model_family`, таблиця метаданих, REFORMS 1–2 | 1 | [docs/lab1_report.md](docs/lab1_report.md) |
| `DynamicPipelineBuilder`, адаптивний імпутер, валідація конфігурації | 1 | [src/dynamic_pipeline_builder.py](src/dynamic_pipeline_builder.py) |
| Конфігурації препроцесингу (`config_lr`, `config_rf`) | 1 | [config/](config/) |
| Кастомні трансформери (3 категорії) і їх обґрунтування | 2 | [src/features.py](src/features.py), [звіт §1](docs/lab2_report.md#1-крок-1--ручний-дизайн-ознак-п-41) |
| Екстрактори (текст, зображення), перевірка інформативності | 2 | [src/extraction.py](src/extraction.py), [звіт §2](docs/lab2_report.md#2-крок-2--мультимодальна-екстракція-п-42) |
| EntitySet, DFS, графік «до/після», повторна перевірка захищених ознак | 2 | [notebooks/02_multimodal_leakage.ipynb](notebooks/02_multimodal_leakage.ipynb), [звіт §3](docs/lab2_report.md#3-крок-3--autofe-featuretools-dfs-п-43) |
| Таблиця п'яти експериментів, рядок з витоком, DPR/EOR | 2 | [звіт §4](docs/lab2_report.md#4-крок-4--експерименти-і-пастка-витоку-п-44) |
| Трирівневий відбір ознак | 2 | [src/selection.py](src/selection.py), [звіт §5](docs/lab2_report.md#5-крок-5--статистика-потім-відбір-п-45) |
| REFORMS, модулі 3, 4, 6 | 2 | [звіт §6](docs/lab2_report.md#6-reforms-модулі-3-4-і-6) |
| DVC: `.dvc`-покажчики, стадія `prepare`, читання лише через `dvc.api` | 3 | [data/raw/*.dvc](data/raw/), [dvc.yaml](dvc.yaml), [src/data.py](src/data.py), [звіт §1](docs/lab3_report.md) |
| Механізм витоку ЛР2, McNemar | 3 | [звіт §2, §6](docs/lab3_report.md), [notebooks/lab3_robust_validation.ipynb](notebooks/lab3_robust_validation.ipynb) |
| Герметичний конвеєр (imblearn), вибір ланки, вкладена CV + Optuna | 3 | [src/evaluation.py](src/evaluation.py), [звіт §3–4](docs/lab3_report.md) |
| Матриця бізнес-втрат, три підходи, графік «втрати vs поріг», оновлений ML Canvas | 3 | [config/costs.yaml](config/costs.yaml), [звіт §5](docs/lab3_report.md) |
| DPR/EOR з трьома порівняннями, вплив SMOTE на захищену ознаку | 3 | [notebooks/lab3_fairness.ipynb](notebooks/lab3_fairness.ipynb), [звіт §7](docs/lab3_report.md) |
| REFORMS, модулі 5 і 6 | 3 | [звіт §8](docs/lab3_report.md) |
| Частково розмічений набір (pool / test під DVC) | 4A | [src/semi_supervised.py](src/semi_supervised.py), [dvc.yaml](dvc.yaml) (`lab4a_split`), [звіт §1](docs/lab4_report.md) |
| Self-Training, Label Spreading vs лише розмічені; якість псевдоміток | 4A | [notebooks/lab4a_semi_supervised.ipynb](notebooks/lab4a_semi_supervised.ipynb), [звіт §2–3](docs/lab4_report.md) |
| Active learning (стратегії запитів, оракул, random), міра невизначеності для ЛР5 | 4A | [звіт §4–5](docs/lab4_report.md), `reports/lab4a_strategy.json` |
| Експорт в ONNX, межа графа, skops, `model_meta.json`, тест паритету | 5 | [src/export/export_onnx.py](src/export/export_onnx.py), [src/serving/](src/serving/), [tests/test_onnx_parity.py](tests/test_onnx_parity.py) |
| MLflow Registry: v1 → v2 @champion, v3 @challenger, метрики з інтервалами, DVC-хеші | 5 | [src/export/register_model.py](src/export/register_model.py), [deploy/requirements.txt](deploy/requirements.txt) |
| FastAPI-сервіс, контракт Pydantic v2, перевірка контракту в lifespan | 5 | [src/app/main.py](src/app/main.py), [src/app/schemas.py](src/app/schemas.py) |
| OOD-детектор (PCA + KDE) і fallback на правило ЛР1 | 5 | [src/app/ood.py](src/app/ood.py), `reports/ood_report.json` |
| Трековий модуль 4A: черга розмітки, раунд дорозмітки, парний бутстреп | 5 | [src/observability/](src/observability/), `reports/label_queue.csv`, `reports/label_round.json` |
| Автотести сервісу (8 сценаріїв + трек) | 5 | [tests/test_api.py](tests/test_api.py), [tests/test_track.py](tests/test_track.py) |
| `/v1/explain` (SHAP + LIME), офлайн-звіт пояснень, HITL-дашборд | 5 | [src/xai/](src/xai/), `reports/xai/`, [src/dashboard/app.py](src/dashboard/app.py) |
| Оцінка на test: система vs правило, fairness, ROI | 5 | [src/export/evaluate_test.py](src/export/evaluate_test.py), `reports/test_evaluation.json` |
| Data Card, Model Card, REFORMS Audit (+ REFORMS-Ops) | 5 | [reports/DATA_CARD.md](reports/DATA_CARD.md), [reports/MODEL_CARD.md](reports/MODEL_CARD.md), [reports/REFORMS_Audit.md](reports/REFORMS_Audit.md) |
| Слайди захисту (5) | 5 | [docs/slides/ChurnGuard_defense.pptx](docs/slides/ChurnGuard_defense.pptx), [reports/defense.pdf](reports/defense.pdf) |
| Docker Compose (MLflow + API + дашборд) | 5 | [docker-compose.yml](docker-compose.yml), [deploy/](deploy/) |

## Структура репозиторію

```
ML_Labs/
├── config/
│   ├── config_lr.yaml, config_rf.yaml          # ЛР1: препроцесинг (additive / partitioning)
│   ├── config_lr_fe.yaml, config_rf_fe.yaml    # ЛР2: + ручні ознаки (Експеримент 2)
│   ├── generated/                              # згенеровані конфігурації: ЛР2 (full / leaky), ЛР3 (hermetic)
│   ├── data.yaml                               # джерело даних, дедуплікація, інжектування пропусків
│   ├── multimodal.yaml                         # ЛР2: корпус Amazon, прив'язка, екстракція, DFS
│   ├── selection.yaml                          # ЛР2: параметри відбору ознак
│   ├── hermetic.yaml                           # ЛР3: розбиття, SMOTE, відбір у фолдах, вкладена CV, простір Optuna
│   ├── costs.yaml                              # ЛР3: матриця бізнес-втрат і class_weight
│   ├── lab4a.yaml                              # ЛР4A: частки міток, Self-Training, Label Spreading, active learning
│   └── generated/config_rf_serving.yaml        # ЛР5: препроцесор сервісу (без ознак відгуків)
├── data/                                       # під DVC (не в git): raw/*.dvc — покажчики; processed/ — виходи стадії prepare
├── dvc.yaml, dvc.lock                          # стадії prepare (ЛР3) і lab4a_split (ЛР4A), хеші входів/виходів
├── deploy/                                     # ЛР5: requirements (сервіс, трек), Dockerfile, seed_registry.py
├── docker-compose.yml                          # ЛР5: mlflow + seed + api + dashboard
├── docs/                                       # звіти лабораторних; slides/ — презентація захисту ЛР5
├── notebooks/
│   ├── 00_eda.ipynb, 01_quality_gate.ipynb     # ЛР1
│   ├── 02_multimodal_leakage.ipynb             # ЛР2
│   ├── lab3_robust_validation.ipynb            # ЛР3: DVC, герметизація, вкладена CV, cost-sensitive, McNemar
│   ├── lab3_fairness.ipynb                     # ЛР3: довірчі DPR/EOR, вплив SMOTE
│   └── lab4a_semi_supervised.ipynb             # ЛР4A: semi-supervised, псевдомітки, active learning
├── reports/                                    # графіки й таблиці; ЛР5: *.json (з run_id і model_version), xai/, картки, REFORMS, defense.pdf
├── src/
│   ├── dynamic_pipeline_builder.py             # ЛР1: validate_config, AdaptiveImputer, DynamicPipelineBuilder
│   ├── transformers.py                         # реєстр кроків transform_policy, вінсоризація, синоніми
│   ├── data.py                                 # завантаження, дедуплікація, MCAR
│   ├── features.py                             # ЛР2, Крок 1: конструктори ознак, WoE / OOF-WoE
│   ├── pipeline.py                             # ЛР2: розширений препроцесор (конструктори → ColumnTransformer)
│   ├── extraction.py                           # ЛР2, Кроки 2–3: корпус, прив'язка, TextExtractor, ImageExtractor, DFS
│   ├── selection.py                            # ЛР2, Крок 5: статистика, Filter, Hybrid (SFS), Embedded, Tree
│   ├── evaluation.py                           # евристика, DPR/EOR; ЛР3: герметичний конвеєр, Nested CV, Optuna, втрати, McNemar
│   ├── prepare_data.py                         # ЛР3: стадія DVC prepare, генерація герметичних конфігурацій
│   ├── preprocessor.py                         # ЛР3: точка входу препроцесора (реекспорт класів ЛР1–2)
│   ├── legacy_lab2.py                          # ЛР3: відтворення прогнозів Експ. 4 ЛР2 для McNemar
│   ├── run_nested.py                           # ЛР3: вкладена CV і фінальна модель (CLI)
│   ├── semi_supervised.py                      # ЛР4A: Self-Training, Label Spreading, стратегії запитів, AL
│   ├── run_lab4a.py                            # ЛР4A: CLI (--split / --ssl / --al)
│   ├── serving/                                # ЛР5: препроцесинг поза ONNX, Predictor (препроцесор + ONNX + OOD + правило)
│   ├── export/                                 # ЛР5: export_onnx, register_model, evaluate_test
│   ├── app/                                    # ЛР5: FastAPI (main, schemas, db), ood, benchmark
│   ├── observability/                          # ЛР5 (4A): label_queue, label_round
│   ├── xai/                                    # ЛР5: LocalExplainer (SHAP, LIME), offline_report
│   └── dashboard/app.py                        # ЛР5: Streamlit
├── models/                                     # ЛР5 (у git): pipeline.onnx, pipeline.skops, preprocessor.skops, shap_background.npy,
│                                               #   model_meta.json, ood/ood.skops; joblib ЛР3 — не в git
├── tests/                                      # pytest: test_lab1.py … test_lab4.py; ЛР5: conftest, test_onnx_parity, test_api, test_track
├── run_verification.py                         # протокол верифікації (з ТЗ ЛР1) — на всіх конфігураціях
├── PLAN.md
└── requirements.txt
```

## Запуск

Потрібен Python 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name ml-labs     # ядро, на яке посилаються ноутбуки
```

**Дані.** Завантажте `E Commerce Dataset.xlsx` з Kaggle —
[Ecommerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
(Ankit Verma, 2021) — і покладіть у `data/raw/`. Дані не комітяться в git — їх версіонує **DVC** (з ЛР3).
Корпус Amazon Reviews 2023 і фото завантажує код ЛР2 (`src/extraction.py::download_reviews`) при першому запуску (≈ 2 хв).
Віддаленого DVC-сховища немає (кеш `.dvc/cache` локальний), тож на новій машині дані треба завантажити заново і перевірити
збіг хешів з `.dvc`-покажчиками (`dvc status`).

```powershell
dvc status                          # хеші даних збігаються з .dvc / dvc.lock
dvc repro                           # (пере)будувати data/processed/* (стадія prepare)
python run_verification.py          # протокол верифікації: усі конфігурації ЛР1–ЛР3
python -m pytest -q                 # автотести
jupyter nbconvert --to notebook --execute --inplace notebooks/lab3_robust_validation.ipynb notebooks/lab3_fairness.ipynb
python -m src.run_lab4a --ssl; python -m src.run_lab4a --al     # ЛР4A (≈ 6 + 6 хв), потім notebooks/lab4a_semi_supervised.ipynb
```

## ЛР5 — ChurnGuard: PoC-сервіс підтримки рішень (трек 4A)

Повний звіт — [docs/lab5_report.md](docs/lab5_report.md). Усі цифри нижче взято з JSON-звітів у `reports/`. Кожен звіт містить
`model_version` і `run_id`, для яких його пораховано; для @champion це **v2**, run `ed310e5fcd524d249510833a3211ab2b`.

### Додаток (6 рядків)

| | |
|---|---|
| Користувач | менеджер з утримання клієнтів інтернет-магазину |
| Рішення, яке підтримує додаток | зробити клієнтові утримувальну пропозицію цього місяця чи ні; остаточно вирішує людина |
| Що вводиться | 16 полів поведінки за місяць (стаж, замовлення, купони, скарга, кешбек…), без `CityTier` і `Gender` |
| Що отримує користувач | ризик відтоку, рекомендацію (поріг 0.46), невизначеність, пояснення SHAP waterfall + LIME |
| Нетиповий випадок (fallback) | рішення за Non-ML правилом ЛР1 `Tenure ≤ 1 OR Complain = 1`, у відповіді `fallback_applied: true` |
| Ціна помилок (матриця ЛР3) | FN — втрачений клієнт (480 у.о.); FP — знижка тому, хто не пішов би (50 у.о.); TP — 362 у.о. |

**Трек 4A.** Модуль додає здатність, якої немає в правилі: він показує, **яких клієнтів передати експертові першими** (черга
розмітки), і перенавчає модель на отриманих мітках у новій версії реєстру. Розмітку симульовано оракулом: це приховані
мітки пулу.

### Запуск сервісу й дашборду

```powershell
uvicorn src.app.main:app --port 8000            # Swagger: http://localhost:8000/docs (модель @champion з mlflow.db)
streamlit run src/dashboard/app.py              # http://localhost:8501, ходить в API (змінна API_URL)
mlflow ui --backend-store-uri sqlite:///mlflow.db   # реєстр: http://localhost:5000
# або все разом у Docker:
docker compose up --build                       # mlflow :5000, api :8000, dashboard :8501
```

Без `mlflow.db` (свіжий клон) сервіс бере артефакти з `models/` (`MODEL_SOURCE=local`). У Docker реєстр наповнює
`deploy/seed_registry.py` з тих самих файлів `models/`, тому там @champion має номер версії 1 на сервері, а не 2.

### Ключові рішення

* **Межа ONNX-графа.** У графі лише `RandomForestClassifier`: 15 ознак → `label`, `probabilities` (opset 17, zipmap вимкнено).
  Решта препроцесингу працює **поза графом**: `models/preprocessor.skops` (DFS-ознаки клієнта з `src/serving/features.py`,
  AdaptiveImputer, OOF-WoE, вінсоризація, one-hot, SelectKBest, обрані SFS колонки). Причина: кастомні трансформери ЛР1–2 не мають
  конвертерів skl2onnx. SMOTE (семплер) з експорту прибрано: він працює лише під час навчання. Ознаки відгуків ЛР2
  у сервісі не використовуються, бо в CRM на момент запиту їх немає. v1 перенавчено на train ЛР5 з гіперпараметрами ЛР3.
* **Паритет ONNX ↔ sklearn** на test (1 015): max |Δp| = 2.2·10⁻⁷ (atol 10⁻⁴). **Після порогу 0.46 клас змінили 0 об'єктів**,
  тож переліку близьких до порогу немає (`reports/onnx_parity_v1.json`).
* **skops: довірені типи** переглянуто вручну, завантаження — лише з allowlist префіксів (`src/serving/artifacts.py`):
  * `pipeline.skops`: `sklearn.tree._tree.Tree`;
  * `preprocessor.skops`: `src.dynamic_pipeline_builder.AdaptiveImputer`, `src.features.{CategoryCrosser, IntervalBinner, OOFWoEEncoder,
    SafeRatio}`, `src.serving.features.IndexSelector`, `src.transformers.{ValueAliasMapper, Winsorizer}`,
    `sklearn.feature_selection._univariate_selection.f_classif`;
  * `ood/ood.skops`: `sklearn.neighbors._kd_tree.KDTree`, `sklearn.metrics._dist_metrics.EuclideanDistance64`.
* **Стратегія релізу.** Аліаси замість Stages: v1 → v2 (той самий ONNX + OOD-артефакти) → @champion; v1 має тег `superseded_by = 2`.
  Відкат — перепризначення аліасу, без зміни коду. Виведення в продакшен: спершу **shadow** — challenger отримує копію запитів,
  а його прогнози лише логуються. Потім **канарка** на частині менеджерів. **A/B на клієнтах не застосовуємо**: рішення ухвалює
  людина, а ефект пропозиції видно лише через місяць, тож помилкова версія встигла б вплинути на тисячі клієнтів раніше, ніж
  це стане видно.
* **Невизначеність** — ентропія прогнозу (міра стратегії запитів ЛР4A): H(p) = −p·log₂p − (1−p)·log₂(1−p) ∈ [0, 1].
  Максимум 1 досягається при p = 0.5.
* **OOD.** StandardScaler → PCA(whiten) → KDE на 80% train (простір входу ONNX). Компонент **9**: найменше k, що пояснює ≥ 0.9
  дисперсії (**0.919**). Bandwidth 0.233 підібрано за log-likelihood на 5-fold CV. Поріг −36.2 — 1-й перцентиль log-густини
  на відкладених 20% train.
  * **Fallback на test: 1 з 1 015 (0.1%).** Це нижче орієнтиру 1–3%: детектор консервативний. Частка більша за 3% вимагала б перегляду.
  * Штучні OOD (test, зсунутий на 3σ по PC1 / PC1+PC2) виявлено у **76% / 87%** випадків.
  * На реальних нетипових запитах якість детектора **не оцінено**: таких міток немає.
* **SHAP:** `TreeExplainer` на `pipeline.skops` з того самого run, що й ONNX. Функція цінності — **interventional**
  (`feature_perturbation="interventional"`), фон — `models/shap_background.npy` (300 рядків train), вихід — імовірність класу 1,
  base value 0.231. LIME: `LimeTabularExplainer(training_data=фон)`, 1 500 збурень, на сервері в `/v1/explain`.
  Стабільність SHAP (Спірмен 0.95, топ-5 0.91) і збіг із LIME — діагностика надійності, а не доказ правильності пояснень.

### Трек 4A: пул, черга, рішення про @challenger

* **Пул** — `role = test` стадії DVC `lab4a_split` (1 015 клієнтів train ЛР3, у навчання v2 не входили). Його пропущено через `/v1/predict`
  без міток. З train v1 (3 044) і з test ЛР5 (holdout ЛР3, 1 015) пул не перетинається (перевіряє `src/serving/data.py`).
* **Черга** (`src/observability/label_queue.py` → `reports/label_queue.csv`): 20 запитів без фідбеку за стратегією **entropy + diversity**
  (топ-100 за ентропією → KMeans на 20). 3 OOD-запити пулу в чергу не потрапили — їх виведено окремо (`reports/label_queue_ood.csv`).
* **Раунд:** 20 міток оракула → `/v1/feedback` → перенавчання (ті самі гіперпараметри, експорт, паритет: 0 змін класу) →
  **v3 @challenger** з тегами `strategy`, `n_new_labels`. Парний бутстреп на test (1 000 повторень, однакові індекси для всіх моделей):

  | Різниця | ROC-AUC | Втрати, у.о./клієнта |
  |---|---|---|
  | challenger − champion | +0.0019 [−0.0043; 0.0083] | +0.58 [−0.28; 1.43] |
  | random − champion | +0.0038 [−0.0033; 0.0126] | +0.63 [−0.20; 1.51] |

  **Рішення: @challenger НЕ просунуто.** Обидва інтервали містять 0, тож перевагу не доведено: 20 міток — це 0.7% train.
  @champion автоматично не змінюється.

### Тести ↔ ML Test Score

| Категорія | Тести |
|---|---|
| Дані | `test_3_protected_attribute_rejected` (CityTier/Gender → 422), `test_4_string_instead_of_number`, `test_5_out_of_field_bounds`, `test_meta_contract`; ЛР1–3: захищені ознаки поза конфігураціями, ідемпотентність препроцесора |
| Модель | `test_parity_on_test`, `test_parity_on_domain_boundaries_and_categories`, `test_no_label_flips_far_from_threshold`, `test_explain_matches_prediction`; ЛР3: герметичність конвеєра |
| Інфраструктура | `test_1_health_after_start`, `test_2_valid_predict`, `test_7_batch_over_limit`, `test_8_feedback_unknown_request` (404), `test_broken_contract_keeps_503` |
| Моніторинг | `test_6_ood_fallback`, `test_queue_limited_sorted_without_feedback_and_ood`, `test_feedback_removes_request_from_next_queue`, `test_explain_without_log_does_not_duplicate_queue`; журнал запитів і фідбеку в SQLite |

**Час відповіді** (`python -m src.app.benchmark` → `reports/latency.json`; 100 послідовних запитів через HTTP; i5-13420H, 12 логічних ядер,
16 ГБ, Windows 10, @champion v2):

| Ендпоінт | p50, мс | p95, мс |
|---|---|---|
| `/v1/predict` | 212 | 253 |
| `/v1/predict_batch` (256 рядків, ліміт пакета 256) | 1 821 | 2 168 |
| `/v1/explain` (SHAP + LIME 1 500) | 646 | 780 |

### Відтворення ЛР5

```powershell
dvc status                                        # дані збігаються з lab3-v1.0
python -m src.export.export_onnx                  # v1: models/*, reports/onnx_parity_v1.json
python -m src.export.register_model --release v1  # run release-v1, @champion = v1
python -m src.app.ood                             # OOD: models/ood/, model_meta.json["ood"], reports/ood_report.json
python -m src.export.register_model --release v2  # run release-v2, @champion = v2, v1: superseded_by = 2
python -m src.export.evaluate_test                # reports/test_evaluation.json: система vs правило, fairness, ROI
python -m src.observability.label_round --stream  # пул → /v1/predict (журнал feedback.db)
python -m src.observability.label_queue           # reports/label_queue.csv
python -m src.observability.label_round --label   # 20 міток оракула → /v1/feedback
python -m src.observability.label_round --retrain # v3 @challenger, reports/label_round.json
python -m src.xai.offline_report                  # reports/xai/, лог у run @champion
python -m src.app.benchmark                       # reports/latency.json
python -m pytest -q                               # 55 тестів
```

`feedback.db`, `mlflow.db` і `mlruns/` не комітяться. Фінальні артефакти моделі (`models/`) — у git, щоб сервіс і тести
запускалися зі свіжого клону.
