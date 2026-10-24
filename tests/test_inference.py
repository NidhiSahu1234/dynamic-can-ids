from pathlib import Path

import pytest

from ml.config import ARTIFACTS_DIR


@pytest.mark.skipif(
    not (ARTIFACTS_DIR / "dt_model.joblib").exists(),
    reason="Artifacts not trained yet",
)
def test_scaler_model_pairing():
    from backend.app.ml.inference import ModelArtifacts

    art = ModelArtifacts()
    art.load()
    data = ["05", "28", "84", "66", "6d", "00", "00", "a2"]
    X = art.raw_to_selected("0350", data, dlc=8)
    pred, prob = art.predict_dt(X)
    assert pred in (0, 1)
    assert 0.0 <= prob <= 1.0
    score = art.score_ae(X)
    assert score >= 0.0


@pytest.mark.skipif(
    not (ARTIFACTS_DIR / "dt_model.joblib").exists(),
    reason="Artifacts not trained yet",
)
def test_classify_frame_normal_sample():
    from backend.app.ml.inference import classify_frame, get_artifacts

    art = get_artifacts()
    # First row from normal_processed.csv
    data = ["05", "28", "84", "66", "6d", "00", "00", "a2"]
    result = classify_frame("0350", 8, data, bloom_contains=False, artifacts=art)
    assert result.predicted_class in ("normal", "attack", "anomaly")
