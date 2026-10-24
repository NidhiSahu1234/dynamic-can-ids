from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class AlertRecord:
    alert_id: str
    timestamp: str
    can_id: str
    predicted_class: str
    detection_method: str
    severity: str
    anomaly_score: float | None
    threshold: float | None
    payload_json: str


class AlertStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    can_id TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    detection_method TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    anomaly_score REAL,
                    threshold REAL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def add(
        self,
        can_id: str,
        predicted_class: str,
        detection_method: str,
        severity: str,
        payload: dict,
        anomaly_score: float | None = None,
        threshold: float | None = None,
    ) -> AlertRecord:
        record = AlertRecord(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            can_id=can_id,
            predicted_class=predicted_class,
            detection_method=detection_method,
            severity=severity,
            anomaly_score=anomaly_score,
            threshold=threshold,
            payload_json=json.dumps(payload),
        )
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO alerts (
                    alert_id, timestamp, can_id, predicted_class, detection_method,
                    severity, anomaly_score, threshold, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.alert_id,
                    record.timestamp,
                    record.can_id,
                    record.predicted_class,
                    record.detection_method,
                    record.severity,
                    record.anomaly_score,
                    record.threshold,
                    record.payload_json,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return record

    def list_recent(self, limit: int = 100) -> list[AlertRecord]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [AlertRecord(**dict(r)) for r in rows]

    def count(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) AS c FROM alerts").fetchone()
        finally:
            conn.close()
        return int(row["c"])
