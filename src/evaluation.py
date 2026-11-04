"""Оцінювання: ROC-AUC, евристичний бейзлайн, метрики справедливості (DPR, EOR)."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def heuristic_predict(X: pd.DataFrame) -> pd.Series:
    """Правило ЛР1 (Non-ML Benchmark): Tenure ≤ 1 міс. АБО скарга минулого місяця → відтік.

    Правило — частина постановки задачі (ML Canvas), а не крок препроцесингу, тому його
    умови записані тут явно. Порівняння з NaN дає False.
    """
    return ((X["Tenure"] <= 1) | (X["Complain"] == 1)).astype(int)


def selection_rate(pred: np.ndarray) -> float:
    return float(np.mean(pred)) if len(pred) else np.nan


def true_positive_rate(y: np.ndarray, pred: np.ndarray) -> float:
    pos = y == 1
    return float(pred[pos].mean()) if pos.any() else np.nan


def fairness_report(y_true, y_pred, group, privileged, groups: Iterable | None = None) -> pd.DataFrame:
    """DPR і EOR кожної групи відносно привілейованої.

    DPR = P(ŷ=1 | група) / P(ŷ=1 | привілейована)        — паритет частки позитивних рішень
    EOR = TPR(група) / TPR(привілейована)                  — рівність можливостей серед справжніх позитивних
    Поріг-орієнтир 0.80 («правило 4/5») — евристика, а не математична межа.
    """
    y_true, y_pred, group = np.asarray(y_true), np.asarray(y_pred), np.asarray(group)
    groups = list(groups) if groups is not None else sorted(pd.unique(group))
    priv = group == privileged
    sr_p, tpr_p = selection_rate(y_pred[priv]), true_positive_rate(y_true[priv], y_pred[priv])
    rows = []
    for g in groups:
        m = group == g
        sr, tpr = selection_rate(y_pred[m]), true_positive_rate(y_true[m], y_pred[m])
        rows.append({"group": g, "n": int(m.sum()), "positives": int((y_true[m] == 1).sum()),
                     "selection_rate": sr, "TPR": tpr,
                     "DPR": sr / sr_p if sr_p else np.nan, "EOR": tpr / tpr_p if tpr_p else np.nan})
    return pd.DataFrame(rows)


def auc(y_true, scores) -> float:
    return float(roc_auc_score(y_true, scores))
