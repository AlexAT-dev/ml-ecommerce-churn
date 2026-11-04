"""Завантаження датасету та контрольоване інжектування пропусків.

Тут немає жодних статистик по даних: лише читання файла і детерміноване
(за seed) маскування випадково обраних клітинок (MCAR). Інжектування
імітує властивість джерела даних, а не крок препроцесингу, тому живе
поза конвеєром і описане в config/data.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_data_config(path: str | Path = "config/data.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(data_config: dict) -> pd.DataFrame:
    """Сирий датасет без жодних змін."""
    src = data_config["source"]
    path = Path(src["path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"Не знайдено {path}. Завантажте датасет за посиланням {src.get('url')} "
            f"і покладіть у data/raw/"
        )
    return pd.read_excel(path, sheet_name=src.get("sheet", 0))


def inject_mcar(df: pd.DataFrame, columns: Dict[str, float], seed: int) -> pd.DataFrame:
    """Додає пропуски типу MCAR: у колонці маскується рівно ``share·n`` випадкових клітинок.

    Клітинки обираються з усіх рядків рівноймовірно (незалежно від значень),
    тож підсумкова частка = природні пропуски + інжектовані (без перетину
    з уже пропущеними).
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    for col, share in columns.items():
        if col not in out.columns:
            raise KeyError(f"inject_mcar: колонки {col!r} немає в датасеті")
        candidates = np.flatnonzero(out[col].notna().to_numpy())
        k = int(round(share * len(out)))
        k = min(k, len(candidates))
        idx = rng.choice(candidates, size=k, replace=False)
        out.loc[out.index[idx], col] = np.nan
    return out


def load_dataset(data_config: dict | None = None, inject: bool = True) -> pd.DataFrame:
    """Датасет, з яким працює конвеєр: сирі дані + інжектовані за config/data.yaml пропуски."""
    data_config = data_config or load_data_config()
    df = load_raw(data_config)
    spec = data_config.get("missing_injection") or {}
    if inject and spec.get("columns"):
        df = inject_mcar(df, spec["columns"], seed=int(spec["seed"]))
    return df


def split_xy(df: pd.DataFrame, data_config: dict) -> Tuple[pd.DataFrame, pd.Series]:
    """X — усі колонки, крім цілі (ID і захищена ознака лишаються: їх відкидає ColumnTransformer)."""
    target = data_config["target"]
    return df.drop(columns=[target]), df[target].astype(int)
