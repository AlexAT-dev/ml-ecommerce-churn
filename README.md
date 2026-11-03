# ML_Labs — прогнозування відтоку клієнтів (E-commerce)

Наскрізний навчальний проєкт з дисципліни "Прикладне Машинне Навчання": 5 лабораторних робіт за методологією **CRISP-ML(Q)** і стандартом **REFORMS**.
Кожна наступна робота продовжує попередню.

- **Домен:** E-commerce — прогнозування відтоку (churn)
- **Датасет:** [Ecommerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) (Kaggle) — ~5.6 тис. клієнтів, 20 ознак, відтік ~17%
- **Захищена ознака:** _буде визначено в ЛР1_

## Лабораторні роботи

| ЛР | Тема | Гілка | Статус |
|---|---|---|---|
| 1 | Системний дизайн за CRISP-ML(Q) та динамічний препроцесинг-конфігуратор | `lab1` | ⏳ |
| 2 | Мультимодальна інженерія та відбір ознак | `lab2` | ⏳ |
| 3 | Герметизація пайплайну: ліквідація витоку, аномалії, дисбаланс, Nested CV | `lab3` | ⏳ |
| 4 | Трек Модуля 4 (на вибір) | `lab4` | ⏳ |
| 5 | Реліз PoC-системи | `lab5` | ⏳ |

Кожна лабораторна робота виконується в окремій гілці.

## Структура репозиторію

```
ML_Labs/
├── config/          # YAML-конфігурації
├── data/
│   ├── raw/         # сирі дані (не в git)
│   └── processed/
├── docs/            # документація проєкту
├── notebooks/       # Jupyter-ноутбуки
├── reports/         # графіки, звіти
├── src/             # вихідний код
├── PLAN.md          # робочий план
└── requirements.txt
```

## Запуск

Потрібен Python 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Дані.** Завантажте `E Commerce Dataset.xlsx` з Kaggle —
[Ecommerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
(Ankit Verma, 2021) — і покладіть у `data/raw/`. Дані не комітяться в репозиторій.
