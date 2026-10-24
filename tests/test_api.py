from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ml.config import ARTIFACTS_DIR


@pytest.fixture
def client():
    from backend.app.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "models_loaded" in body


@pytest.mark.skipif(
    not (ARTIFACTS_DIR / "dt_model.joblib").exists(),
    reason="Artifacts not trained yet",
)
def test_predict_valid_frame(client):
    payload = {
        "can_id": "0350",
        "dlc": 8,
        "data": ["05", "28", "84", "66", "6d", "00", "00", "a2"],
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "predicted_class" in data
    assert "detection_method" in data


def test_predict_invalid_can_id(client):
    payload = {"can_id": "ZZZZ", "dlc": 8, "data": ["00"] * 8}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_traffic_sample_encodes_inactive_bytes_as_null(client):
    r = client.get("/traffic/sample", params={"source": "fuzzy", "skip": 1, "limit": 1})
    assert r.status_code == 200
    frame = r.json()["frames"][0]
    assert frame["dlc"] == 5
    assert frame["data"][:5] == ["ff", "7f", "00", "05", "49"]
    assert frame["data"][5:] == [None, None, None]


def test_traffic_sample_uses_safe_normal_demo_when_csv_is_missing(client, tmp_path, monkeypatch):
    import backend.app.main as main

    monkeypatch.setitem(main.TRAFFIC_SOURCES, "normal", tmp_path / "missing.csv")
    r = client.get("/traffic/sample", params={"source": "normal", "limit": 20})
    assert r.status_code == 200
    body = r.json()
    assert body["demo"] is True
    assert body["count"] == 1
    assert body["frames"][0]["can_id"] == "0350"


def test_traffic_sample_reports_missing_non_normal_csv(client, tmp_path, monkeypatch):
    import backend.app.main as main

    monkeypatch.setitem(main.TRAFFIC_SOURCES, "dos", tmp_path / "missing.csv")
    r = client.get("/traffic/sample", params={"source": "dos"})
    assert r.status_code == 404
    assert "safe built-in normal demo frame" in r.json()["detail"]
