"""ЛР5, трек 4A — 2 тести черги розмітки."""
import json
import sqlite3

import numpy as np

from tests.conftest import requires_models


def _fill(db, n=40, seed=0):
    from src.app.db import SCHEMA

    rng = np.random.default_rng(seed)
    with sqlite3.connect(db) as con:
        con.executescript(SCHEMA)
        for i in range(n):
            p = float(rng.uniform())
            con.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?)",
                        (f"r{i}", "2026-01-01", json.dumps({}), p, int(p > 0.5), float(rng.uniform()), "t",
                         int(i % 7 == 0), -1.0))
        for i in range(0, n, 5):                                   # частина вже має фідбек
            con.execute("INSERT INTO feedback VALUES (?,?,?)", (f"r{i}", "2026-01-02", 1))


def test_queue_limited_sorted_without_feedback_and_ood(tmp_path):
    from src.observability.label_queue import build_queue

    db = tmp_path / "q.db"
    _fill(db)
    q = build_queue(db, size=20, predictor=None)
    ids = [r["request_id"] for r in q["queue"]]
    unc = [r["uncertainty"] for r in q["queue"]]
    with sqlite3.connect(db) as con:
        labeled = {r[0] for r in con.execute("SELECT request_id FROM feedback")}
        ood = {r[0] for r in con.execute("SELECT request_id FROM predictions WHERE fallback_applied = 1")}
    assert len(ids) <= 20
    assert unc == sorted(unc, reverse=True)
    assert not set(ids) & labeled and not set(ids) & ood
    assert all(r["request_id"] in ood for r in q["ood"])


@requires_models
def test_feedback_removes_request_from_next_queue(client, valid_payload):
    from src.app.db import db_path
    from src.observability.label_queue import build_queue

    for t in (0.0, 1.0, 2.0, 5.0, 12.0):
        assert client.post("/v1/predict", json={**valid_payload, "Tenure": t}).status_code == 200
    first = build_queue(db_path(), size=20)["queue"]
    target = first[0]["request_id"]
    assert client.post("/v1/feedback", json={"request_id": target, "y_true": 1}).status_code == 200
    after = [r["request_id"] for r in client.get("/v1/label_queue").json()["queue"]]
    assert target not in after


@requires_models
def test_explain_without_log_does_not_duplicate_queue(client, valid_payload):
    """Дашборд пояснює запит з черги через /v1/explain?log=false: копія клієнта не має з'являтися в журналі (і в черзі)."""
    from src.app.db import db_path

    def n_logged():
        with sqlite3.connect(db_path()) as con:
            return con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    before = n_logged()
    r = client.post("/v1/explain", params={"log": "false"}, json=valid_payload)
    assert r.status_code == 200 and n_logged() == before
    rid = r.json()["prediction"]["request_id"]
    assert client.post("/v1/feedback", json={"request_id": rid, "y_true": 1}).status_code == 404   # тимчасовий id
    assert client.post("/v1/explain", json=valid_payload).status_code == 200 and n_logged() == before + 1
