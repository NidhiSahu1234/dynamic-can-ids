from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from ml.features import validate_can_id, validate_dlc, validate_payload_bytes


class CANFrameRequest(BaseModel):
    timestamp: float | None = None
    can_id: str
    dlc: int = Field(ge=0, le=8)
    data: list[str | None] = Field(min_length=8, max_length=8)

    @field_validator("can_id")
    @classmethod
    def check_can_id(cls, v: str) -> str:
        validate_can_id(v)
        return v

    @model_validator(mode="after")
    def check_payload(self) -> "CANFrameRequest":
        validate_payload_bytes(self.data, self.dlc)
        return self


class PredictionResponse(BaseModel):
    predicted_class: str
    detection_method: str
    severity: str
    anomaly_score: float | None = None
    threshold: float | None = None
    dt_probability_attack: float | None = None
    bloom_match: bool = False
    signature: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class BatchPredictRequest(BaseModel):
    frames: list[CANFrameRequest] = Field(min_length=1, max_length=500)


class BatchPredictResponse(BaseModel):
    results: list[PredictionResponse]
    processed: int


class RuleCreateRequest(BaseModel):
    signature: str = Field(min_length=1, max_length=128)
    attack_type: str = Field(min_length=1, max_length=32)
    status: str = "confirmed"
    notes: str | None = Field(default=None, max_length=500)


class RuleResponse(BaseModel):
    rule_id: str
    signature: str
    attack_type: str
    status: str
    source: str
    created_at: str
    notes: str | None = None


class AlertResponse(BaseModel):
    alert_id: str
    timestamp: str
    can_id: str
    predicted_class: str
    detection_method: str
    severity: str
    anomaly_score: float | None
    threshold: float | None
    payload: dict[str, Any]
