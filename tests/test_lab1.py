"""Додаткові перевірки ЛР1 (поверх обов'язкового run_verification.py).

Запуск: python -m pytest -q
"""
import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.dynamic_pipeline_builder import AdaptiveImputer, DynamicPipelineBuilder

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [ROOT / "config" / "config_lr.yaml", ROOT / "config" / "config_rf.yaml"]
POLICY = [
    {"max_share": 0.10, "strategy": "median"},
    {"max_share": 0.40, "strategy": "median", "add_indicator": True},
    {"max_share": 1.00, "strategy": "constant", "fill_value": 0, "add_indicator": True},
]


def _load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write(tmp_path, cfg, name="cfg.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return p


# --- конфігурації ---------------------------------------------------------- #
def test_configs_differ_by_one_line():
    a, b = (p.read_text(encoding="utf-8").splitlines() for p in CONFIGS)
    changed = [l for l in difflib.unified_diff(a, b, lineterm="", n=0)
               if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    assert len(changed) == 2, changed  # один рядок видалено, один додано
    assert all("model_family" in l for l in changed)


def test_protected_in_features_is_blocked(tmp_path):
    cfg = _load(CONFIGS[0])
    cfg["features"]["categorical_ordinal"].append(cfg["protected_attributes"][0])
    with pytest.raises(ValueError, match="Захищені ознаки"):
        DynamicPipelineBuilder(_write(tmp_path, cfg))


def test_unknown_step_is_blocked(tmp_path):
    cfg = _load(CONFIGS[0])
    family = cfg["model_family"]
    cfg["transform_policy"][family]["numeric_continuous"] = ["no_such_step"]
    with pytest.raises(ValueError, match="невідомий крок"):
        DynamicPipelineBuilder(_write(tmp_path, cfg))


def test_unknown_model_family_is_blocked(tmp_path):
    cfg = _load(CONFIGS[0])
    cfg["model_family"] = "simple"
    with pytest.raises(ValueError, match="model_family"):
        DynamicPipelineBuilder(_write(tmp_path, cfg))


@pytest.mark.parametrize("path", CONFIGS)
def test_build_returns_unfitted(path):
    ct = DynamicPipelineBuilder(path).build()
    with pytest.raises(NotFittedError):
        check_is_fitted(ct)


def test_no_dataset_literals_in_code():
    """Умова автопровалу: у коді немає назв колонок і значень model_family."""
    code = "\n".join((ROOT / "src" / f).read_text(encoding="utf-8")
                     for f in ("dynamic_pipeline_builder.py", "transformers.py"))
    cfg = _load(CONFIGS[0])
    literals = set(cfg["transform_policy"]) | set(cfg["protected_attributes"])
    for cols in cfg["features"].values():
        literals |= set(cols)
    found = {w for w in literals if re.search(rf"\b{re.escape(str(w))}\b", code)}
    assert not found, f"Літерали з конфігурації в коді: {found}"
    for rules in cfg["imputation_policy"].values():
        for r in rules[:-1]:  # останній поріг = 1.0 — межа області значень частки, не політика
            assert str(r["max_share"]) not in code


# --- адаптивний імпутер ---------------------------------------------------- #
def _frame(n, share, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(10, 2, n)
    x[rng.random(n) < share] = np.nan
    return pd.DataFrame({"a": x})


def test_imputer_routes_by_share():
    X = pd.concat([_frame(1000, 0.05)["a"].rename("low"),
                   _frame(1000, 0.25, 1)["a"].rename("mid"),
                   _frame(1000, 0.60, 2)["a"].rename("high")], axis=1)
    imp = AdaptiveImputer(POLICY).fit(X)
    assert imp.audit_["low"]["rule_max_share"] == 0.10 and not imp.audit_["low"]["add_indicator"]
    assert imp.audit_["mid"]["rule_max_share"] == 0.40 and imp.audit_["mid"]["add_indicator"]
    assert imp.audit_["high"]["strategy"] == "constant"
    out = imp.transform(X)
    assert list(out.columns) == list(imp.get_feature_names_out())
    assert not out.isna().any().any()


def test_transform_does_not_recompute():
    train, test = _frame(1000, 0.05), _frame(1000, 0.50, 7)
    imp = AdaptiveImputer(POLICY).fit(train)
    before = dict(imp.audit_["a"])
    out = imp.transform(test)
    assert imp.audit_["a"] == before                 # стан не змінився
    assert out.shape[1] == 1                          # індикатор не з'явився
    filled = out["a"][test["a"].isna()]
    assert np.allclose(filled, train["a"].median())   # медіана train, не test


def test_indicator_present_even_without_test_missing():
    train, test = _frame(1000, 0.25), _frame(200, 0.0)
    imp = AdaptiveImputer(POLICY).fit(train)
    out = imp.transform(test)
    assert out.shape[1] == 2 and out.iloc[:, 1].sum() == 0
