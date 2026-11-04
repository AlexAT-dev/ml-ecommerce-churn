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
| 2 | Мультимодальна інженерія та відбір ознак | `lab2` | [docs/lab2_report.md](docs/lab2_report.md) | 🔄 |
| 3 | Герметизація пайплайну: ліквідація витоку, аномалії, дисбаланс, Nested CV | `lab3` | — | ⏳ |
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

## Структура репозиторію

```
ML_Labs/
├── config/
│   ├── config_lr.yaml, config_rf.yaml          # ЛР1: препроцесинг (additive / partitioning)
│   ├── config_lr_fe.yaml, config_rf_fe.yaml    # ЛР2: + ручні ознаки (Експеримент 2)
│   ├── generated/                              # ЛР2: + DFS-ознаки (Експ. 3) і «дірява» версія (Експ. 4) — генерується ноутбуком
│   ├── data.yaml                               # джерело даних, дедуплікація, інжектування пропусків
│   ├── multimodal.yaml                         # ЛР2: корпус Amazon, прив'язка, екстракція, DFS
│   └── selection.yaml                          # ЛР2: параметри відбору ознак
├── data/                                       # не в git: raw/ (xlsx, корпус Amazon, фото), processed/
├── docs/                                       # звіти лабораторних
├── notebooks/
│   ├── 00_eda.ipynb, 01_quality_gate.ipynb     # ЛР1
│   └── 02_multimodal_leakage.ipynb             # ЛР2
├── reports/                                    # графіки й таблиці результатів
├── src/
│   ├── dynamic_pipeline_builder.py             # ЛР1: validate_config, AdaptiveImputer, DynamicPipelineBuilder
│   ├── transformers.py                         # реєстр кроків transform_policy, вінсоризація, синоніми
│   ├── data.py                                 # завантаження, дедуплікація, MCAR
│   ├── features.py                             # ЛР2, Крок 1: конструктори ознак, WoE / OOF-WoE
│   ├── pipeline.py                             # ЛР2: розширений препроцесор (конструктори → ColumnTransformer)
│   ├── extraction.py                           # ЛР2, Кроки 2–3: корпус, прив'язка, TextExtractor, ImageExtractor, DFS
│   ├── selection.py                            # ЛР2, Крок 5: статистика, Filter, Hybrid (SFS), Embedded, Tree
│   └── evaluation.py                           # евристика, DPR / EOR
├── tests/                                      # pytest: test_lab1.py, test_lab2.py
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
(Ankit Verma, 2021) — і покладіть у `data/raw/`. Дані не комітяться в репозиторій.
Корпус Amazon Reviews 2023 і фото ноутбук ЛР2 завантажує сам при першому запуску (≈ 2 хв, кеш у `data/`).

```powershell
python run_verification.py          # протокол верифікації: усі конфігурації ЛР1–ЛР2
python -m pytest -q                 # автотести
jupyter nbconvert --to notebook --execute --inplace notebooks/02_multimodal_leakage.ipynb
```
