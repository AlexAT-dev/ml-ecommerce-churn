# REFORMS Audit — ChurnGuard

32 пункти REFORMS (Kapoor et al., 2024, *Science Advances* 10(18), eadk3452; формулювання — офіційний `checklist.docx`).
REFORMS — чекліст звітності, а не тест з вердиктом: **статус** показує, чи пункт розкрито в документації.
Скорочення: L1…L5 — звіти `docs/lab1_report.md` … `docs/lab4_report.md` і цей проєкт (ЛР5).
MLflow: модель `ChurnGuard`, v2 = @champion, run `ed310e5fcd524d249510833a3211ab2b`.

## Модуль 1. Study goals

| № | Пункт | Відповідь | Доказ | Статус |
|---|---|---|---|---|
| 1a | Population or distribution about which the scientific claim is made | Клієнти одного анонімного онлайн-рітейлера (імовірно Індія), активні на момент знімка | L1 §6 (1a); DATA_CARD §5 | розкрито |
| 1b | Motivation for choosing this population | Доступність відкритого набору з ціллю відтоку і захищеними ознаками; перенесення на інші компанії не обґрунтоване | L1 §6 (1b) | розкрито |
| 1c | Motivation for the use of ML methods | Прогноз для окремого клієнта (алгоритмічна культура Бреймана); поріг обґрунтування — Non-ML правило (Recall 0.85 / Precision 0.32 / ROC-AUC 0.74 на test); модель його перевершує (ROC-AUC 0.907, втрати −11%) | L1 §1, §6; MODEL_CARD §7 | розкрито |

## Модуль 2. Computational reproducibility

| № | Пункт | Відповідь | Доказ | Статус |
|---|---|---|---|---|
| 2a | Dataset with link/DOI | Kaggle-посилання + DVC-хеші сирого файла й похідних наборів; тег `lab3-v1.0` | DATA_CARD §2; `data/raw/*.dvc`; `dvc.lock`; теги MLflow `dvc_*_md5` | розкрито |
| 2b | Code with link/DOI to the version | Git-репозиторій, теги `lab1-v1.0` … `lab4-v1.0`; ЛР5 — тег `lab5-v1.0` | README | розкрито |
| 2c | Computing infrastructure | Windows 10, Python 3.11.9, CPU; версії — `requirements.txt`, `deploy/requirements.txt`, `model_meta.json["library_versions"]`; час: вкладена CV RF ≈ 26 хв, LR ≈ 6 хв, експорт ≈ 1 хв, офлайн-XAI ≈ 8 хв | README; L3 §4 | розкрито |
| 2d | README with instructions | README: встановлення, DVC, запуск сервісу, дашборду, тестів, Docker Compose | README | розкрито |
| 2e | Reproduction script | `dvc repro` → `python -m src.run_nested` → `python -m src.export.export_onnx` → `register_model` → `src.app.ood` → `label_round` → `offline_report` → `evaluate_test`; ноутбуки ЛР1–4 | README «Відтворення» | розкрито |

## Модуль 3. Data quality

| № | Пункт | Відповідь | Доказ | Статус |
|---|---|---|---|---|
| 3a | Sources of data, collection time, annotations | Kaggle-набір (мітка — прапорець компанії, процес не задокументовано; дата збору невідома); корпус Amazon — лише в ЛР2–3 | DATA_CARD §2, §4; L2 §6 | розкрито |
| 3b | Sampling frame | Клієнти рітейлера в момент знімка; невипадкова вибірка компанії | L2 §6 (3b) | частково — рамку компанія не описує |
| 3c | Why dataset is useful | Природна бінарна ціль відтоку, поведінкові ознаки, кандидати захищених ознак | L1 §1 | розкрито |
| 3d | Outcome definition + statistics | `Churn` = 1 — клієнт пішов; 16.6% (5 074 після дедуплікації) | L2 §6 (3d); DATA_CARD §1 | розкрито |
| 3e | Number of samples | 5 630 → 5 074 (дедуплікація); train 3 044 / пул 1 015 / test 1 015 | DATA_CARD §3 | розкрито |
| 3f | Missing data by class | Tenure 8.7% (відтік) vs 4.0%; WarehouseToHome 7.6 vs 3.8%; MCAR-інжектовані — рівномірно | L2 §6 (3f) | розкрито |
| 3g | Representativeness | Висновки — лише для бази цього рітейлера; Tier 2 недопредставлений (4%) | L2 §6 (3g); DATA_CARD §5 | розкрито |

## Модуль 4. Data preprocessing

| № | Пункт | Відповідь | Доказ | Статус |
|---|---|---|---|---|
| 4a | Excluded samples | 556 точних дублікатів (без конфліктних міток) — щоб копії не потрапили в train і test | L2 §6 (4a) | розкрито |
| 4b | Impossible / corrupt samples | Ділення на 0 і log(0) → NaN → AdaptiveImputer; синоніми категорій зведено; API відхиляє значення поза доменними межами (422) | L2 §6 (4b); `src/app/schemas.py` | розкрито |
| 4c | All transformations raw → model | Дедуплікація → MCAR → DFS-ознаки клієнта → конструктори ЛР2 → AdaptiveImputer → вінсоризація / OOF-WoE / one-hot → SelectKBest → SFS-колонки → RF (з позначкою «залежна від даних») | L2 §6 (4c); `src/serving/__init__.py` | розкрито |

## Модуль 5. Modeling

| № | Пункт | Відповідь | Доказ | Статус |
|---|---|---|---|---|
| 5a | Models, features, loss | RF (281, depth 11), зважена нечистота Джині (class_weight 50/118), SMOTE у навчанні; 15 ознак — `model_meta.json["features"]` | L3 §8 (5a); MODEL_CARD §1 | розкрито |
| 5b | Justification of model types | Родини `additive` (LR) і `partitioning` (RF) з ЛР1; RF стабільно кращий (ROC-AUC 0.918 vs 0.876) | L3 §8 (5b) | розкрито |
| 5c | Evaluation method, splits/folds | Вкладена CV 5×3 StratifiedKFold (ЛР3) + незалежний test ЛР5 (holdout) з бутстреп-інтервалами | L3 §4; MODEL_CARD §5, §7 | розкрито |
| 5d | Model selection method | Мінімум сумарних втрат на зовнішніх фолдах (RF 55 976 vs LR 57 151) | L3 §8 (5d) | розкрито |
| 5e | Hyperparameters: range, method, final | Optuna TPE, 30 спроб; діапазони й обґрунтування; фінальні — `model_meta.json["hyperparameters"]` | L3 §4, §8 (5e) | розкрито |
| 5f | Appropriate baselines | Non-ML правило ЛР1, «нікого не чіпати», «всім»; на test ROC-AUC 0.907 vs 0.744, втрати 69.0 vs 77.8 у.о./клієнта | L3 §5; MODEL_CARD §7 | розкрито |

## Модуль 6. Data leakage

| № | Пункт | Відповідь | Доказ | Статус |
|---|---|---|---|---|
| 6a | Preprocessing/modeling use only training data | Усі залежні кроки — ланки одного `imblearn.Pipeline`, що вчиться на train; OOD — на 80% train, поріг — на решті 20% train; test ЛР5 не брав участі ні в чому, крім фінальних метрик; витік ЛР2 доведено/усунено | L3 §2, §6; `src/app/ood.py` | розкрито |
| 6b | Dependencies / duplicates between train and test | Дедуплікація до розбиття; один рядок — один клієнт; пул розмітки 4A не перетинається ні з train, ні з test (assert у `src/serving/data.py`) | L3 §4; `src/serving/data.py` | розкрито |
| 6c | Legitimacy of each feature | Захищені ознаки відсутні в моделі (перевірка контракту, тест 422); ознаки — поведінка клієнта; ризик: пропуски стажу можуть бути наслідком відтоку (прихований витік цілі) — задокументовано | L3 §8 (6c); MODEL_CARD §8 | розкрито |

## Модуль 7. Metrics and uncertainty

| № | Пункт | Відповідь | Доказ | Статус |
|---|---|---|---|---|
| 7a | All metrics; suitability of selection metric | Сумарні втрати (вибір), Recall/Precision/F1 @T, ROC-AUC; Accuracy не використовується через дисбаланс 17/83 | MODEL_CARD §4 | розкрито |
| 7b | Uncertainty estimates and how computed | 95% t-інтервали по 5 зовнішніх фолдах (ЛР3); 95% бутстреп-інтервали на test (1 000 повторень) для метрик і DPR/EOR; парний бутстреп для champion/challenger/random | MLflow `*_ci_low/high`; `reports/test_evaluation.json`; `reports/label_round.json` | розкрито |
| 7c | Statistical tests and assumptions | McNemar (парні прогнози, χ² з поправкою — припущення: ті самі об'єкти, b + c ≥ 25 виконано); парний бутстреп (незалежність об'єктів test) | L3 §6; L5 `label_round.json` | розкрито |

## Модуль 8. Generalizability and limitations

| № | Пункт | Відповідь | Доказ | Статус |
|---|---|---|---|---|
| 8a | Evidence of external validity | Зовнішньої валідації (інша компанія/період) немає; є лише внутрішній незалежний test (holdout) — його метрики збігаються з CV (0.907 vs 0.918) | MODEL_CARD §7 | частково — немає даних іншої компанії/періоду |
| 8b | Contexts where findings won't hold | Інші ринки й компанії; нові типи клієнтів поза зоною OOD; Tier 2 (мала вибірка); зміна економіки пропозицій (матриця втрат); симульована розмітка | MODEL_CARD §2, §9 | розкрито |

---

# REFORMS-Ops (розширення курсу)

## OOD і fallback
KDE у просторі входу ONNX (StandardScaler → PCA(9, whiten) — мінімум компонент для ≥ 0.9 дисперсії, bandwidth 0.233 за CV), поріг — 1-й перцентиль log-густини на відкладених 20% train.
Хибні спрацювання на test — 0.1% (1 клієнт); штучні OOD (3σ) — 76% (PC1) / 87% (PC1+PC2); на реальних нетипових запитах якість не оцінено (немає міток). Для OOD-запиту рішення дає Non-ML правило,
`meta.fallback_applied = true`; модель і пояснення — лише довідково. Звіт: `reports/ood_report.json`.

## Цикл фідбеку і правило @challenger
Кожен `/v1/predict` логується в SQLite (`predictions`), `/v1/feedback` — у `feedback` (невідомий `request_id` → 404). Перенавчена модель
реєструється **лише** з аліасом `@challenger`; `@champion` автоматично не змінюється. Просування — тільки якщо парний бутстреп на test
доводить перевагу (інтервал різниці не містить 0) і після ручної перевірки.

## Стратегія релізу і перехід v1 → v2
Аліаси замість Stages. v1 (без OOD) → v2 (+ OOD-артефакти, той самий ONNX) — @champion перепризначено на v2, v1 має тег `superseded_by = 2`:
відкат = перепризначення аліасу без зміни коду (сервіс резолвить аліас один раз у lifespan). Стратегія виведення: спершу **shadow**
(challenger отримує копію запитів, прогнози лише логуються), далі — канарка на частині менеджерів; A/B на клієнтах не застосовуємо,
бо рішення ухвалює людина й вплив пропозиції затримується на місяць.

## H.E.A.R.T. (з ЛР1)
* **H** — шкода: нерівний TPR → клієнти групи частіше йдуть без допомоги; контроль EOR за CityTier і Gender (MODEL_CARD §7).
* **E** — стейкхолдери: менеджери (навантаження чергою), клієнти без пропозицій (субсидують знижки іншим).
* **A** — відповідальність: власник моделі ≠ власник рішення; менеджер затверджує кожну пропозицію.
* **R** — EOR як пріоритетна метрика (різні базові рівні відтоку); інтервали включають 1.
* **T** — проксі `PreferredPaymentMode` (V = 0.44); захист не лише вилученням ознаки, а й вимірюванням на кожній версії.

## Трек 4A — цикл розмітки
* **Міра невизначеності:** ентропія прогнозу (ЛР4A); для двох класів еквівалентна least confidence і margin.
* **Стратегія черги:** entropy + diversity (топ-100 за ентропією → KMeans(20) → найближчі до центрів), без запитів з фідбеком; OOD — окремим списком.
* **Пул:** 1 015 клієнтів (25% train), відкладено до етапу 1, не перетинається з train і test; пропущено через `/v1/predict` без міток.
* **Оракул:** приховані мітки пулу (розмітку симульовано — DATA_CARD §4); бюджет — 20 міток.
* **Результат** (парний бутстреп, 1 000 повторень, test n = 1 015): challenger − champion: ROC-AUC +0.002 [−0.004; +0.008],
  втрати +0.58 [−0.28; +1.43] у.о./клієнта; random − champion: ROC-AUC +0.004 [−0.003; +0.013]. **Усі інтервали містять 0 → перевагу
  не доведено, @challenger (v3) не просунуто.** Пояснення: 20 нових міток поверх 3 044 — 0.7% даних; ефект AL з ЛР4A проявлявся при
  малій кількості міток (150–450), а не поверх повної вибірки.
