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
| 3 | Герметизація пайплайну: ліквідація витоку, аномалії, дисбаланс, Nested CV | `lab3` | [docs/lab3_report.md](docs/lab3_report.md) | 🔄 |
| 4 | Трек Модуля 4 (на вибір) | `lab4` | — | ⏳ |
| 5 | Реліз PoC-системи | `lab5` | — | ⏳ |

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
│   └── costs.yaml                              # ЛР3: матриця бізнес-втрат і class_weight
├── data/                                       # під DVC (не в git): raw/*.dvc — покажчики; processed/ — виходи стадії prepare
├── dvc.yaml, dvc.lock                          # ЛР3: стадія prepare і хеші її входів/виходів
├── docs/                                       # звіти лабораторних
├── notebooks/
│   ├── 00_eda.ipynb, 01_quality_gate.ipynb     # ЛР1
│   ├── 02_multimodal_leakage.ipynb             # ЛР2
│   ├── lab3_robust_validation.ipynb            # ЛР3: DVC, герметизація, вкладена CV, cost-sensitive, McNemar
│   └── lab3_fairness.ipynb                     # ЛР3: довірчі DPR/EOR, вплив SMOTE
├── reports/                                    # графіки й таблиці результатів
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
│   └── legacy_lab2.py                          # ЛР3: відтворення прогнозів Експ. 4 ЛР2 для McNemar
├── models/                                     # не в git: фінальний конвеєр ЛР3 (joblib)
├── tests/                                      # pytest: test_lab1.py, test_lab2.py, test_lab3.py
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
```
