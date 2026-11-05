"""ЛР5, Етап 4 — 8 обов'язкових сценаріїв API (+ стартовий контракт).

ML Test Score (Breck et al., 2017): дані — 3, 4, 5, 6; модель — паритет ONNX (test_onnx_parity.py), 2;
інфраструктура — 1, 7, 8, test_broken_contract_*; моніторинг — логування прогнозів і фідбеку (test_track.py), 6 (OOD).
"""
import os

from tests.conftest import requires_models

pytestmark = requires_models


def test_1_health_after_start(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_2_valid_predict(client, valid_payload):
    r = client.post("/v1/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    for key in ("request_id", "score", "label", "uncertainty", "meta"):
        assert key in body
    assert 0.0 <= body["uncertainty"] <= 1.0 and {"model_version", "fallback_applied", "ood_log_density"} <= set(body["meta"])


def test_3_protected_attribute_rejected(client, valid_payload):
    for extra in ({"Gender": "Male"}, {"CityTier": 1}):
        r = client.post("/v1/predict", json={**valid_payload, **extra})
        assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_4_string_instead_of_number(client, valid_payload):
    assert client.post("/v1/predict", json={**valid_payload, "Tenure": "35"}).status_code == 422


def test_5_out_of_field_bounds(client, valid_payload):
    assert client.post("/v1/predict", json={**valid_payload, "SatisfactionScore": 9}).status_code == 422
    assert client.post("/v1/predict", json={**valid_payload, "Tenure": -1.0}).status_code == 422


def test_6_ood_fallback(client, ood_payload):
    r = client.post("/v1/predict", json=ood_payload)
    assert r.status_code == 200
    meta = r.json()["meta"]
    assert meta["fallback_applied"] is True and meta["ood_log_density"] is not None


def test_7_batch_over_limit(client, valid_payload):
    from src.app.schemas import BATCH_LIMIT
    assert client.post("/v1/predict_batch", json={"items": [valid_payload] * (BATCH_LIMIT + 1)}).status_code == 422
    assert client.post("/v1/predict_batch", json={"items": [valid_payload] * 3}).status_code == 200


def test_8_feedback_unknown_request(client):
    r = client.post("/v1/feedback", json={"request_id": "00000000-0000-4000-8000-000000000000", "y_true": 1})
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_explain_matches_prediction(client, valid_payload):
    r = client.post("/v1/explain", json=valid_payload)
    assert r.status_code == 200
    j = r.json()
    total = j["shap"]["base_value"] + sum(j["shap"]["values"])
    assert abs(total - j["prediction"]["score"]) < 1e-2             # адитивність SHAP (одиниці ймовірності)
    assert 0 <= j["agreement"]["top5_overlap"] <= 1


def test_broken_contract_keeps_503(broken_model_dir, tmp_path):
    from fastapi.testclient import TestClient

    from src.app.main import create_app
    app = create_app()
    old = {k: os.environ.get(k) for k in ("MODEL_SOURCE", "MODEL_DIR", "FEEDBACK_DB")}
    os.environ.update({"MODEL_SOURCE": "local", "MODEL_DIR": str(broken_model_dir), "FEEDBACK_DB": str(tmp_path / "b.db")})
    try:
        with TestClient(app) as c:
            r = c.get("/health")
            assert r.status_code == 503
            assert c.post("/v1/predict", json={}).status_code in (422, 503)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
