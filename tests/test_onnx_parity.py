"""ЛР5, Етап 1 — паритет sklearn-артефакту (skops) і ONNX-графа на тому самому float32-вході."""
import numpy as np
import onnxruntime as ort

from tests.conftest import MODELS, requires_models

pytestmark = requires_models


def _onnx(X, meta):
    sess = ort.InferenceSession(str(MODELS / "pipeline.onnx"), providers=["CPUExecutionProvider"])
    return sess.run(None, {meta["onnx_inputs"][0]["name"]: X.astype(np.float32)})[1]


def _atol(meta):
    return 1e-4 if meta["model_family"] == "tree_ensemble" else 1e-5


def test_parity_on_test(export_pipe, X_test, meta):
    np.testing.assert_allclose(export_pipe.predict_proba(X_test), _onnx(X_test, meta), atol=_atol(meta))


def test_parity_on_domain_boundaries_and_categories(export_pipe, X_synthetic, meta):
    np.testing.assert_allclose(export_pipe.predict_proba(X_synthetic), _onnx(X_synthetic, meta), atol=_atol(meta))


def test_no_label_flips_far_from_threshold(export_pipe, X_test, meta):
    """Клас може змінитися лише для |score − поріг| < atol (наслідок float32); інакше — помилка експорту."""
    thr = meta["threshold"]
    p_skl, p_onx = export_pipe.predict_proba(X_test)[:, 1], _onnx(X_test, meta)[:, 1]
    flips = (p_skl >= thr) != (p_onx >= thr)
    assert np.all(np.abs(p_skl[flips] - thr) < _atol(meta))


def test_meta_contract(meta):
    for key in ["request_fields", "features", "dtypes", "onnx_inputs", "classes", "positive_class", "threshold",
                "model_family", "opset", "preprocessing_version", "library_versions"]:
        assert key in meta
    assert meta["onnx_inputs"][0]["shape"][1] == len(meta["features"])
    assert not set(meta["protected_attributes"]) & set(meta["request_fields"])
