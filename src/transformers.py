"""Атомарні трансформери та реєстр «назва в конфігурації → фабрика трансформера».

Реєстр — це *механізм*: перелік операцій, які система вміє виконувати.
Яку операцію застосувати до якої групи ознак — *політика*, вона живе в YAML
(секція ``transform_policy``). Тому назви операцій тут є, а назв колонок,
порогів і ключів ``model_family`` — немає.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted


def _as_frame(X, columns=None) -> pd.DataFrame:
    """Приводить вхід до DataFrame, зберігаючи назви колонок, якщо вони є."""
    if isinstance(X, pd.DataFrame):
        return X
    X = np.asarray(X)
    if columns is None:
        columns = [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=list(columns))


class _NamedColumnsMixin:
    """Спільна логіка запам'ятовування назв вхідних колонок (one-to-one трансформери)."""

    def _remember_columns(self, X) -> pd.DataFrame:
        X = _as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(self.feature_names_in_, dtype=object)


class ValueAliasMapper(_NamedColumnsMixin, BaseEstimator, TransformerMixin):
    """Зводить синоніми категорій до канонічного значення (напр. "CC" → "Credit Card").

    Перетворення безстанове: таблиця замін задається в конфігурації, з даних
    нічого не оцінюється, тому витоку бути не може. Застосовується до імпутації,
    інакше мода рахувалася б по «розщеплених» категоріях.
    """

    def __init__(self, mapping: Mapping[str, Mapping[Any, Any]] | None = None):
        self.mapping = mapping

    def fit(self, X, y=None):
        self._remember_columns(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X = _as_frame(X, self.feature_names_in_).copy()
        for col, aliases in (self.mapping or {}).items():
            if col in X.columns:
                X[col] = X[col].replace(dict(aliases))
        return X


class Winsorizer(_NamedColumnsMixin, BaseEstimator, TransformerMixin):
    """Вінсоризація: обрізання значень за перцентилями, оціненими на train.

    Межі (``lower_``/``upper_``) обчислюються лише у ``fit`` — на тестовій
    підвибірці вони тільки застосовуються.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = self._remember_columns(X)
        if not 0.0 <= self.lower < self.upper <= 1.0:
            raise ValueError(
                f"Winsorizer: потрібно 0 <= lower < upper <= 1, отримано "
                f"lower={self.lower}, upper={self.upper}"
            )
        values = X.to_numpy(dtype=float)
        self.lower_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, ("lower_", "upper_"))
        X = _as_frame(X, self.feature_names_in_)
        clipped = np.clip(X.to_numpy(dtype=float), self.lower_, self.upper_)
        return pd.DataFrame(clipped, columns=X.columns, index=X.index)


class WoEEncoder(_NamedColumnsMixin, BaseEstimator, TransformerMixin):
    """Weight of Evidence для бінарної цілі: WoE = ln(%подій / %не-подій) у категорії.

    * ``smoothing`` — адитивне згладжування (Лапласа), щоб рідкісна категорія
      без подій не давала ±∞.
    * Невідома на train категорія отримує WoE = 0 (нейтральний внесок).
    * Потребує ``y`` у ``fit``: це кероване ціллю кодування, тому навчати його
      можна тільки на тренувальній підвибірці / фолді.
    """

    def __init__(self, smoothing: float = 0.5):
        self.smoothing = smoothing

    def fit(self, X, y=None):
        X = self._remember_columns(X)
        if y is None:
            raise ValueError("WoEEncoder потребує y у fit(): кодування кероване ціллю")
        y = np.asarray(y).astype(int)
        if set(np.unique(y)) - {0, 1}:
            raise ValueError("WoEEncoder підтримує лише бінарну ціль {0, 1}")
        a = float(self.smoothing)
        total_pos, total_neg = y.sum(), (1 - y).sum()
        self.mapping_: Dict[str, Dict[Any, float]] = {}
        for col in X.columns:
            stats = pd.DataFrame({"x": X[col].to_numpy(), "y": y}).groupby("x")["y"].agg(["sum", "count"])
            k = len(stats)
            pos = (stats["sum"] + a) / (total_pos + a * k)
            neg = (stats["count"] - stats["sum"] + a) / (total_neg + a * k)
            self.mapping_[col] = np.log(pos / neg).to_dict()
        return self

    def transform(self, X):
        check_is_fitted(self, "mapping_")
        X = _as_frame(X, self.feature_names_in_)
        # невідома категорія → NaN після map → нейтральне WoE = 0
        out = {
            col: np.nan_to_num(X[col].map(self.mapping_[col]).to_numpy(dtype=float), nan=0.0)
            for col in X.columns
        }
        return pd.DataFrame(out, index=X.index)


def _log1p(**kwargs) -> FunctionTransformer:
    return FunctionTransformer(np.log1p, feature_names_out="one-to-one", **kwargs)


def _one_hot(**kwargs) -> OneHotEncoder:
    params = {"handle_unknown": "ignore", "sparse_output": False}
    params.update(kwargs)
    return OneHotEncoder(**params)


def _ordinal(**kwargs) -> OrdinalEncoder:
    params = {"handle_unknown": "use_encoded_value", "unknown_value": -1}
    params.update(kwargs)
    return OrdinalEncoder(**params)


# Назва кроку в transform_policy → фабрика (kwargs з конфігурації передаються як є).
TRANSFORMER_REGISTRY: Dict[str, Callable[..., BaseEstimator]] = {
    "log1p": _log1p,
    "standard_scaler": StandardScaler,
    "winsorize": Winsorizer,
    "woe": WoEEncoder,
    "one_hot": _one_hot,
    "ordinal_encoder": _ordinal,
}
