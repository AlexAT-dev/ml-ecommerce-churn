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
| Python | 3.11.9 у `.venv`, kernel `ml-labs` | ✅ |

## Git

- Кожна ЛР — окрема гілка (`lab1` … `lab5`).
- ЛР1: тег `lab1-v1.0` (згадано в README, REFORMS 2b).

## Прогрес

- [x] ЛР1 — код, конфіги, ноутбуки, README готові; чекає коміту/тегу і захисту
- [ ] ЛР2
- [ ] ЛР3
- [ ] ЛР4 (трек: ?)
- [ ] ЛР5

## Нотатки для наступних ЛР

- 556 повних дублікатів (без CustomerID) → при розбитті train/test ризик витоку; вирішити в ЛР2 (дедуплікація або групове розбиття).
- WoEEncoder без внутрішньої CV — у ЛР3 має вчитися всередині фолдів.
- pandas 3.1 / numpy 2.4 / sklearn 1.9 — свіжі; featuretools / imblearn можуть вимагати даунгрейду.
- Особисті чеклісти до захисту — `notes/` (не в git).
