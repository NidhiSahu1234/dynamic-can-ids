"""
Model inference for Dynamic CAN IDS.

Detection flow
--------------
1. Validate frame fields (CAN ID, DLC, hex payload).
2. Bloom filter: if signature matches a *confirmed* rule → attack (confirmed, bloom).
3. Decision Tree (known R/T patterns from pre-training): if predicted attack → alert (decision_tree).
4. If DT predicts normal → Autoencoder MAE vs threshold on normal-trained scaler:
   - error > threshold → *suspected anomaly* (autoencoder), not a confirmed attack.
5. Otherwise → normal.

The Autoencoder is consulted only when the Bloom filter misses and the DT classifies the frame as normal.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml.config import ARTIFACTS_DIR, SELECTED_FEATURES
from ml.features import row_to_feature_vector, select_features

ARTIFACT_NAMES = (
    "dt_model.joblib",
    "dt_scaler.joblib",
    "ae_model.keras",
    "ae_scaler.joblib",
    "metadata.json",
)


class Severity(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionMethod(str, Enum):
    NORMAL = "normal"
    BLOOM_FILTER = "bloom_filter"
    DECISION_TREE = "decision_tree"
    AUTOENCODER = "autoencoder"


@dataclass
class PredictionResult:
    predicted_class: str
    detection_method: DetectionMethod
    anomaly_score: float | None
    threshold: float | None
    severity: Severity
    dt_probability_attack: float | None
    bloom_match: bool
    details: dict[str, Any]


class ModelArtifacts:
    def __init__(self, artifact_dir: Path | None = None) -> None:
        self.artifact_dir = artifact_dir or ARTIFACTS_DIR
        self._loaded = False
        self.dt_model = None
        self.dt_scaler = None
        self.ae_model = None
        self.ae_scaler = None
        self.metadata: dict[str, Any] = {}
        self.ae_threshold: float = 0.0

    def _path(self, name: str) -> Path:
        return self.artifact_dir / name

    def validate_present(self) -> None:
        missing = [n for n in ARTIFACT_NAMES if not self._path(n).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model artifacts in {self.artifact_dir}: {', '.join(missing)}. "
                "Run: python -m ml.train_models"
            )

    def load(self) -> None:
        if self._loaded:
            return
        self.validate_present()
        import json
        # TensorFlow is loaded only after the artifact check so the degraded API
        # can still start on machines without artifacts or TensorFlow runtime.
        from tensorflow import keras

        self.dt_model = joblib.load(self._path("dt_model.joblib"))
        self.dt_scaler = joblib.load(self._path("dt_scaler.joblib"))
        self.ae_scaler = joblib.load(self._path("ae_scaler.joblib"))
        self.ae_model = keras.models.load_model(self._path("ae_model.keras"))
        self.metadata = json.loads(self._path("metadata.json").read_text())
        self.ae_threshold = float(self.metadata["ae_threshold"])
        expected = len(SELECTED_FEATURES)
        if self.metadata.get("selected_features") != SELECTED_FEATURES:
            raise ValueError("Artifact feature order does not match ml.config.SELECTED_FEATURES")
        for name, scaler in (("Decision Tree", self.dt_scaler), ("Autoencoder", self.ae_scaler)):
            if int(getattr(scaler, "n_features_in_", -1)) != expected:
                raise ValueError(f"{name} scaler does not expect {expected} selected features")
        if int(getattr(self.dt_model, "n_features_in_", -1)) != expected:
            raise ValueError("Decision Tree artifact feature count does not match its scaler")
        input_shape = getattr(self.ae_model, "input_shape", None)
        if not input_shape or int(input_shape[-1]) != expected:
            raise ValueError("Autoencoder artifact feature count does not match its scaler")
        if not np.isfinite(self.ae_threshold) or self.ae_threshold < 0:
            raise ValueError("Autoencoder threshold in metadata must be a finite nonnegative number")
        self._loaded = True

    def raw_to_selected(self, can_id: str, data_bytes: list[Any], dlc: int) -> np.ndarray:
        full = row_to_feature_vector(can_id, data_bytes, dlc=dlc)
        return select_features(full.reshape(1, -1))

    def predict_dt(self, X_raw: np.ndarray) -> tuple[int, float]:
        X_scaled = self.dt_scaler.transform(X_raw)
        pred = int(self.dt_model.predict(X_scaled)[0])
        proba = self.dt_model.predict_proba(X_scaled)[0]
        classes = list(self.dt_model.classes_)
        attack_prob = float(proba[classes.index(1)]) if 1 in classes else 0.0
        return pred, attack_prob

    def score_ae(self, X_raw: np.ndarray) -> float:
        X_scaled = self.ae_scaler.transform(X_raw)
        recon = self.ae_model.predict(X_scaled, verbose=0)
        return float(np.mean(np.abs(X_scaled - recon)))


_artifacts: ModelArtifacts | None = None


def get_artifacts(artifact_dir: Path | None = None) -> ModelArtifacts:
    global _artifacts
    if _artifacts is None or (artifact_dir and artifact_dir != _artifacts.artifact_dir):
        _artifacts = ModelArtifacts(artifact_dir)
        _artifacts.load()
    elif not _artifacts._loaded:
        _artifacts.load()
    return _artifacts


def reset_artifacts_cache() -> None:
    global _artifacts
    _artifacts = None


def classify_frame(
    can_id: str,
    dlc: int,
    data_bytes: list[Any],
    *,
    bloom_contains: bool = False,
    artifacts: ModelArtifacts | None = None,
) -> PredictionResult:
    art = artifacts or get_artifacts()
    X_raw = art.raw_to_selected(can_id, data_bytes, dlc)

    if bloom_contains:
        return PredictionResult(
            predicted_class="attack",
            detection_method=DetectionMethod.BLOOM_FILTER,
            anomaly_score=None,
            threshold=None,
            severity=Severity.CRITICAL,
            dt_probability_attack=None,
            bloom_match=True,
            details={"reason": "Confirmed attack signature in Bloom filter"},
        )

    dt_pred, attack_prob = art.predict_dt(X_raw)
    if dt_pred == 1:
        sev = Severity.HIGH if attack_prob >= 0.9 else Severity.MEDIUM
        return PredictionResult(
            predicted_class="attack",
            detection_method=DetectionMethod.DECISION_TREE,
            anomaly_score=None,
            threshold=None,
            severity=sev,
            dt_probability_attack=attack_prob,
            bloom_match=False,
            details={"reason": "Known attack pattern (supervised DT)"},
        )

    ae_score = art.score_ae(X_raw)
    threshold = art.ae_threshold
    if ae_score > threshold:
        return PredictionResult(
            predicted_class="anomaly",
            detection_method=DetectionMethod.AUTOENCODER,
            anomaly_score=ae_score,
            threshold=threshold,
            severity=Severity.MEDIUM,
            dt_probability_attack=attack_prob,
            bloom_match=False,
            details={
                "reason": "Reconstruction error exceeds normal validation threshold; suspected unknown attack"
            },
        )

    return PredictionResult(
        predicted_class="normal",
        detection_method=DetectionMethod.NORMAL,
        anomaly_score=ae_score,
        threshold=threshold,
        severity=Severity.NORMAL,
        dt_probability_attack=attack_prob,
        bloom_match=False,
        details={},
    )
