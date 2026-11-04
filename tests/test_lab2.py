"""Перевірки ЛР2: нові трансформери, екстрактори, розширені конфігурації.

Запуск: python -m pytest -q
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.extraction import ImageExtractor, TextExtractor
from src.features import (CategoryCrosser, CyclicalEncoder, IntervalBinner, OOFWoEEncoder, SafeRatio,
                          WoEEncoder)
from src.pipeline import ExtendedPipelineBuilder

ROOT = Path(__file__).resolve().parents[1]
FE_CONFIGS = [ROOT / "config" / "config_lr_fe.yaml", ROOT / "config" / "config_rf_fe.yaml"]


# --- Категорія Б: захищене відношення ------------------------------------------------ #
def test_safe_ratio_handles_zero_and_nan():
    X = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [0.0, 4.0, 1.0]})
    out = SafeRatio("r", "a", "b", epsilon=1e-5).fit(X).transform(X)
    assert np.isfinite(out.loc[0, "r"])                 # ділення на 0 не дає inf
    assert out.loc[1, "r"] == pytest.approx(0.5, rel=1e-4)
    assert np.isnan(out.loc[2, "r"])                     # пропуск лишається пропуском (його заповнить імпутер)
    zero_eps = SafeRatio("r", "a", "b", epsilon=0.0).fit(X).transform(X)
    assert np.isnan(zero_eps.loc[0, "r"])                # inf примусово → NaN


# --- Категорія А ---------------------------------------------------------------------- #
def test_interval_binner_and_cyclical():
    X = pd.DataFrame({"t": [0, 1, 3, 12, 40, np.nan]})
    codes = IntervalBinner("s", "t", edges=[0, 1, 5, 12, 120]).fit(X).transform(X)["s"]
    assert codes.tolist()[:5] == [0, 0, 1, 2, 3] and np.isnan(codes.iloc[5])
    cyc = CyclicalEncoder({"h": 24}).fit_transform(pd.DataFrame({"h": [0, 23]}))
    # 23:00 і 00:00 — сусіди на колі
    assert np.hypot(*(cyc.iloc[0] - cyc.iloc[1])) < 0.3


# --- Категорія В ---------------------------------------------------------------------- #
def test_crosser_keeps_missing_as_token():
    X = pd.DataFrame({"a": ["x", None], "b": [1, 2]})
    seg = CategoryCrosser("seg", ["a", "b"]).fit(X).transform(X)["seg"]
    assert seg.tolist() == ["x | 1", "NA | 2"]


def test_oof_woe_differs_from_plain_on_train():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"c": rng.integers(0, 400, 2000).astype(str)})   # дуже висока кардинальність
    y = rng.integers(0, 2, 2000)                                        # ціль — чистий шум
    plain = WoEEncoder().fit(X, y).transform(X)["c"]
    oof = OOFWoEEncoder().fit_transform(X, y)["c"]
    corr_plain = np.corrcoef(plain, y)[0, 1]
    corr_oof = np.corrcoef(oof, y)[0, 1]
    assert corr_plain > 0.3          # звичайний WoE «вгадує» шумову ціль на train — перенавчання
    assert abs(corr_oof) < 0.1       # OOF не бачить власної мітки → зв'язку з шумом немає


def test_oof_woe_transform_uses_full_train_mapping():
    X = pd.DataFrame({"c": list("aabbbcc") * 20})
    y = np.array([1, 0, 1, 1, 0, 0, 0] * 20)
    enc = OOFWoEEncoder()
    enc.fit_transform(X, y)
    ref = WoEEncoder().fit(X, y)
    assert np.allclose(enc.transform(X), ref.transform(X))


def test_woe_requires_y():
    with pytest.raises(ValueError, match="потребує y"):
        WoEEncoder().fit(pd.DataFrame({"c": ["a"]}))


# --- Екстрактори: порожні входи не ламають ---------------------------------------------- #
def test_text_extractor_empty_inputs():
    texts = pd.DataFrame({"t": ["good phone fast delivery", "", None, "battery broke after a week",
                                "great value", "terrible support never again", "love it"]})
    ex = TextExtractor(max_features=50, n_components=2).fit(texts)
    out = ex.transform(texts)
    assert out.shape == (7, 2)
    assert out.iloc[1].isna().all() and out.iloc[2].isna().all()
    assert ex.stats_["transform_empty"] == 2


def test_image_extractor_missing_and_broken(tmp_path):
    from skimage import io

    good = tmp_path / "ok.png"
    io.imsave(good, (np.random.default_rng(0).random((16, 16)) * 255).astype(np.uint8))
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    paths = pd.DataFrame({"p": [str(good), None, str(tmp_path / "nope.jpg"), str(broken)]})
    ex = ImageExtractor(levels=8).fit(paths)
    out = ex.transform(paths)
    assert out.iloc[0].notna().all() and out.iloc[1:].isna().all().all()
    assert ex.stats_ == {"rows": 4, "no_image": 2, "failed": 1, "ok": 1}


# --- Розширені конфігурації ------------------------------------------------------------- #
@pytest.mark.parametrize("path", FE_CONFIGS)
def test_fe_config_builds_and_protected_not_used(path):
    builder = ExtendedPipelineBuilder(path)
    prep = builder.build()
    ct = prep.named_steps["columns"]
    used = {c for _, _, cols in ct.transformers for c in cols}
    assert not used & set(builder.config["protected_attributes"])


def test_constructor_on_protected_is_blocked(tmp_path):
    cfg = yaml.safe_load(FE_CONFIGS[0].read_text(encoding="utf-8"))
    cfg["feature_engineering"].append({"type": "cross", "name": "Leak", "sources": ["CityTier", "MaritalStatus"]})
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="конструкторів"):
        ExtendedPipelineBuilder(p)


def test_fe_configs_differ_by_one_line():
    import difflib
    a, b = (p.read_text(encoding="utf-8").splitlines() for p in FE_CONFIGS)
    changed = [l for l in difflib.unified_diff(a, b, lineterm="", n=0)
               if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    assert len(changed) == 2 and all("model_family" in l for l in changed)
