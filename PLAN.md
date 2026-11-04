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
| Трек ЛР4 | 4A / 4B / 4C | не обрано |
| Python | 3.11.9 у `.venv`, kernel `ml-labs`; pandas 2.3.3 (ЛР2: featuretools не працює з pandas 3) | ✅ |

## Git

- Кожна ЛР — окрема гілка (`lab1` … `lab5`).
- ЛР1: тег `lab1-v1.0` (згадано в README, REFORMS 2b).
- ЛР2: тег `lab2-v1.0`.
- Звіти: `docs/labN_report.md`; README — «обкладинка» з таблицею «артефакт → файл».

## Прогрес

- [x] ЛР1 — здано (тег lab1-v1.0)
- [x] ЛР2 — готово, чекає коміту/тегу і захисту (`docs/lab2_report.md`, `notebooks/02_multimodal_leakage.ipynb`)
- [ ] ЛР3
- [ ] ЛР4 (трек: ?)
- [ ] ЛР5

## Нотатки для наступних ЛР

- ✅ ЛР2: дублікати прибрано (`config/data.yaml: deduplicate`), 5 074 рядки; OOF-WoE реалізовано.
- ЛР3: прибрати витік Експ. 4 (TF-IDF/SVD, WoE, Filter/SFS — лише у внутрішніх фолдах, imblearn pipeline, Nested CV); чесний поріг; довірчий DPR/EOR;
  перевірити проксі `PreferredPaymentMode` і кешбек-ознаки; вирішити долю мультимодального блоку (неінформативний).
- ЛР3: природні пропуски залежать від класу (Tenure 4% vs 8.7%) — MAR/MNAR, індикатори можуть бути корисні.
- Мультимодальний корпус і фото — кеш у `data/raw/amazon` (37 МБ), прив'язка — `data/processed/reviews_linked.csv`.
- Особисті чеклісти до захисту — `notes/` (не в git).
