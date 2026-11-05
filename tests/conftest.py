"""Фікстури ЛР5: артефакти моделі, вхід ONNX на test і синтетичних межах, клієнт API з тимчасовою БД."""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def _models_ready():
    return (MODELS / "pipeline.onnx").exists() and (MODELS / "model_meta.json").exists()


requires_models = pytest.mark.skipif(not _models_ready(), reason="немає models/ (python -m src.export.export_onnx)")


@pytest.fixture(scope="session")
def meta():
    return json.loads((MODELS / "model_meta.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def export_pipe():
    from src.serving.artifacts import load_skops   # білий список типів — README, «Довірені типи skops»
    return load_skops(MODELS / "pipeline.skops")


@pytest.fixture(scope="session")
def predictor():
    from src.serving.predictor import Predictor
    return Predictor.from_dir(MODELS, with_explain=False)


@pytest.fixture(scope="session")
def valid_payload():
    from src.app.schemas import PredictionRequest
    return dict(PredictionRequest.model_config["json_schema_extra"]["example"])


@pytest.fixture(scope="session")
def ood_payload(valid_payload):
    """Усі значення в межах Field, але комбінація нетипова (перевірено: log-густина ≪ порогу)."""
    return {**valid_payload, "Tenure": 200.0, "WarehouseToHome": 280.0, "HourSpendOnApp": 20.0,
            "NumberOfDeviceRegistered": 15, "NumberOfAddress": 45, "OrderAmountHikeFromlastYear": 900.0,
            "CouponUsed": 150.0, "OrderCount": 180.0, "DaySinceLastOrder": 3000.0, "CashbackAmount": 9000.0, "Complain": 0}


@pytest.fixture(scope="session")
def X_test(predictor):
    """Test ЛР5 (holdout ЛР3, через DVC) після препроцесингу поза графом — тобто вхід ONNX (float32)."""
    try:
        from src.serving.data import load_lab5
        from src.serving.features import REQUEST_FIELDS
        test = load_lab5()["test"]
    except Exception as exc:  # дані DVC недоступні
        pytest.skip(f"test-набір недоступний через DVC: {exc}")
    return predictor.features(test[REQUEST_FIELDS].to_dict("records"))


@pytest.fixture(scope="session")
def X_synthetic(predictor, valid_payload):
    """Межі доменних діапазонів (ті самі, що Field) і кожне допустиме значення категоріальних ознак."""
    import typing

    from src.app.schemas import PredictionRequest

    rows = []
    for name, field in PredictionRequest.model_fields.items():
        lo = next((m.ge for m in field.metadata if hasattr(m, "ge")), None)
        hi = next((m.le for m in field.metadata if hasattr(m, "le")), None)
        if lo is not None:
            for v in (lo, hi):
                row = {**valid_payload, name: v}
                if name == "CouponUsed":
                    row["OrderCount"] = max(float(v), row["OrderCount"])
                if name == "OrderCount":
                    row["CouponUsed"] = min(float(v), row["CouponUsed"])
                rows.append(row)
        args = [a for a in typing.get_args(field.annotation) if a is not type(None)]
        lits = [x for a in args for x in typing.get_args(a)] if args else list(typing.get_args(field.annotation))
        for v in lits:
            rows.append({**valid_payload, name: v})
        if type(None) in typing.get_args(field.annotation):
            rows.append({**valid_payload, name: None})
    for r in rows:                      # кожен синтетичний рядок проходить контракт API
        PredictionRequest(**r)
    return predictor.features(rows)


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    os.environ["MODEL_SOURCE"] = "local"
    os.environ["MODEL_DIR"] = str(MODELS)
    os.environ["FEEDBACK_DB"] = str(tmp_path_factory.mktemp("db") / "feedback_test.db")
    from src.app.main import app
    with TestClient(app) as c:          # без with lifespan не запуститься
        yield c


@pytest.fixture()
def broken_model_dir(tmp_path):
    d = tmp_path / "models"
    shutil.copytree(MODELS, d, ignore=shutil.ignore_patterns("challenger", "random_baseline", "*.joblib"))
    meta = json.loads((d / "model_meta.json").read_text(encoding="utf-8"))
    meta["features"] = list(reversed(meta["features"]))             # переставлені ознаки
    (d / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return d
