from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.app.ml.inference import ModelArtifacts, classify_frame, get_artifacts
from backend.app.schemas import (
    AlertResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    CANFrameRequest,
    PredictionResponse,
    RuleCreateRequest,
    RuleResponse,
)
from backend.app.services.alerts import AlertStore
from backend.app.services.rules import RuleService
import pandas as pd

from ml.config import ARTIFACTS_DIR, IDS_DB_PATH, PROCESSED_DIR
from ml.features import build_attack_signature

DB_PATH = IDS_DB_PATH
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts = ModelArtifacts()
    try:
        artifacts.load()
        models_ok = True
        model_error = None
    except Exception as exc:
        models_ok = False
        model_error = str(exc)
        artifacts = None

    _state["artifacts"] = artifacts
    _state["models_ok"] = models_ok
    _state["model_error"] = model_error
    _state["rules"] = RuleService(DB_PATH)
    _state["alerts"] = AlertStore(DB_PATH)
    _state["frames_processed"] = 0
    _state["normal_processed"] = 0
    _state["attack_processed"] = 0
    _state["anomaly_processed"] = 0
    yield


app = FastAPI(title="Dynamic CAN IDS API", version="1.0.0", lifespan=lifespan)

cors_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_models() -> ModelArtifacts:
    if not _state.get("models_ok"):
        raise HTTPException(
            status_code=503,
            detail=_state.get("model_error") or "Models not loaded",
        )
    return _state["artifacts"]


def _predict_frame(frame: CANFrameRequest) -> PredictionResponse:
    art = _require_models()
    rules: RuleService = _state["rules"]
    signature = build_attack_signature(frame.can_id, frame.dlc, frame.data)
    bloom_maybe = rules.bloom_maybe_contains(signature)
    bloom_hit = bloom_maybe and rules.is_confirmed_attack(signature)

    result = classify_frame(
        frame.can_id,
        frame.dlc,
        frame.data,
        bloom_contains=bloom_hit,
        artifacts=art,
    )
    if bloom_maybe and not bloom_hit:
        result.details["bloom_filter"] = (
            "Possible Bloom membership was not confirmed in the rule store; "
            "the frame was classified by the ML models."
        )

    if result.predicted_class != "normal":
        _state["alerts"].add(
            can_id=frame.can_id,
            predicted_class=result.predicted_class,
            detection_method=result.detection_method.value,
            severity=result.severity.value,
            payload={"signature": signature, "data": frame.data, "dlc": frame.dlc},
            anomaly_score=result.anomaly_score,
            threshold=result.threshold,
        )

    _state["frames_processed"] = _state.get("frames_processed", 0) + 1
    if result.predicted_class == "normal":
        _state["normal_processed"] = _state.get("normal_processed", 0) + 1
    elif result.predicted_class == "anomaly":
        _state["anomaly_processed"] = _state.get("anomaly_processed", 0) + 1
    else:
        _state["attack_processed"] = _state.get("attack_processed", 0) + 1

    return PredictionResponse(
        predicted_class=result.predicted_class,
        detection_method=result.detection_method.value,
        severity=result.severity.value,
        anomaly_score=result.anomaly_score,
        threshold=result.threshold,
        dt_probability_attack=result.dt_probability_attack,
        bloom_match=result.bloom_match,
        signature=signature,
        details=result.details,
    )


@app.get("/health")
def health():
    return {
        "status": "ok" if _state.get("models_ok") else "degraded",
        "models_loaded": bool(_state.get("models_ok")),
        "model_error": _state.get("model_error"),
        "artifacts_dir": str(ARTIFACTS_DIR),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(frame: CANFrameRequest):
    return _predict_frame(frame)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(body: BatchPredictRequest):
    results = [_predict_frame(f) for f in body.frames]
    return BatchPredictResponse(results=results, processed=len(results))


@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="metrics.json not found. Run python -m ml.train_models",
        )
    return json.loads(METRICS_PATH.read_text())


@app.get("/alerts", response_model=list[AlertResponse])
def alerts(limit: int = Query(default=100, ge=1, le=500)):
    store: AlertStore = _state["alerts"]
    out = []
    for a in store.list_recent(limit=limit):
        out.append(
            AlertResponse(
                alert_id=a.alert_id,
                timestamp=a.timestamp,
                can_id=a.can_id,
                predicted_class=a.predicted_class,
                detection_method=a.detection_method,
                severity=a.severity,
                anomaly_score=a.anomaly_score,
                threshold=a.threshold,
                payload=json.loads(a.payload_json),
            )
        )
    return out


@app.get("/stats")
def stats():
    alerts: AlertStore = _state["alerts"]
    recent = alerts.list_recent(limit=500)
    attack_count = sum(1 for a in recent if a.predicted_class in ("attack", "anomaly"))
    normal_count = sum(1 for a in recent if a.predicted_class == "normal")
    return {
        "frames_processed_session": _state.get("frames_processed", 0),
        "normal_processed_session": _state.get("normal_processed", 0),
        "attack_processed_session": _state.get("attack_processed", 0),
        "anomaly_processed_session": _state.get("anomaly_processed", 0),
        "alert_count": alerts.count(),
        "recent_attack_or_anomaly": attack_count,
        "recent_normal": normal_count,
        "rules_count": len(_state["rules"].list_rules()),
    }


@app.get("/rules", response_model=list[RuleResponse])
def list_rules():
    rules: RuleService = _state["rules"]
    return [RuleResponse(**r.__dict__) for r in rules.list_rules()]


@app.post("/rules", response_model=RuleResponse)
def create_rule(body: RuleCreateRequest):
    rules: RuleService = _state["rules"]
    try:
        rule = rules.add_rule(
            body.signature,
            body.attack_type,
            status=body.status,
            source="api",
            notes=body.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RuleResponse(**rule.__dict__)


TRAFFIC_SOURCES = {
    "normal": PROCESSED_DIR / "normal_processed.csv",
    "dos": PROCESSED_DIR / "DoS_processed.csv",
    "fuzzy": PROCESSED_DIR / "Fuzzy_processed.csv",
    "gear": PROCESSED_DIR / "gear_processed.csv",
    "rpm": PROCESSED_DIR / "RPM_processed.csv",
}
NORMAL_DEMO_FRAME = {
    "timestamp": 0.0,
    "can_id": "0350",
    "dlc": 8,
    "data": ["05", "28", "84", "66", "6d", "00", "00", "a2"],
    "label": "normal",
}


@app.get("/traffic/sample")
def traffic_sample(
    source: str = "normal",
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=200),
):
    path = TRAFFIC_SOURCES.get(source.lower())
    if path is None:
        raise HTTPException(status_code=404, detail="Unknown traffic source")
    if not path.exists():
        if source.lower() == "normal":
            frames = [] if skip else [{**NORMAL_DEMO_FRAME, "data": NORMAL_DEMO_FRAME["data"].copy()}]
            return {
                "source": source,
                "skip": skip,
                "count": len(frames),
                "frames": frames[:limit],
                "demo": True,
            }
        raise HTTPException(
            status_code=404,
            detail=(
                f"Processed dataset '{source}' is not deployed. Only the safe built-in normal "
                "demo frame is available without publishing the processed CSVs."
            ),
        )
    try:
        df = pd.read_csv(
            path,
            skiprows=range(1, skip + 1),
            nrows=limit,
            dtype={"CAN_ID": "string", **{f"Data_{i}": "string" for i in range(8)}},
        )
    except (OSError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read traffic sample: {exc}") from exc
    frames = []
    for row in df.itertuples(index=False):
        data = []
        for i in range(8):
            value = getattr(row, f"Data_{i}")
            data.append(None if pd.isna(value) else str(value))
        frames.append(
            {
                "timestamp": float(row.Timestamp),
                "can_id": str(row.CAN_ID),
                "dlc": int(row.DLC),
                "data": data,
                "label": str(row.Label),
            }
        )
    return {"source": source, "skip": skip, "count": len(frames), "frames": frames}


@app.delete("/rules/{rule_id}")
def delete_rule(rule_id: str):
    rules: RuleService = _state["rules"]
    if not rules.remove_rule(rule_id):
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"deleted": rule_id}
