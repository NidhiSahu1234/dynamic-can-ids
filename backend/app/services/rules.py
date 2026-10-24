"""
Dynamic rule management with Bloom filter sync.

Rule promotion
--------------
- Anomalies detected by the Autoencoder are *suspected*, not added to the Bloom filter automatically.
- Operators (or API clients) promote a validated composite signature to `confirmed` via POST /rules.
- Confirmed signatures are inserted into the Bloom filter for O(1)-ish membership checks.

Deletion
--------
- Rules are removed from the authoritative SQLite store.
- Bloom filter bits are NOT cleared (standard Bloom limitation); deleted rules may still cause
  false-positive Bloom hits. Inference must cross-check rule IDs in the store when handling Bloom hits.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backend.app.services.bloom_filter import BloomFilter, BloomFilterConfig
from ml.features import normalize_attack_signature


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Rule:
    rule_id: str
    signature: str
    attack_type: str
    status: str
    source: str
    created_at: str
    notes: str | None = None


class RuleService:
    def __init__(self, db_path: Path, bloom: BloomFilter | None = None) -> None:
        self.db_path = db_path
        self.bloom = bloom or BloomFilter(BloomFilterConfig())
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rules (
                    rule_id TEXT PRIMARY KEY,
                    signature TEXT NOT NULL UNIQUE,
                    attack_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    notes TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bloom_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload TEXT NOT NULL
                )
                """
            )
            row = conn.execute("SELECT payload FROM bloom_state WHERE id = 1").fetchone()
            if row:
                try:
                    stored = BloomFilter.from_dict(json.loads(row["payload"]))
                    self.bloom = BloomFilter(stored.config)
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    # The SQL rule table is authoritative; rebuild a bad cache from it.
                    self.bloom = BloomFilter(self.bloom.config)
            confirmed = conn.execute(
                "SELECT signature FROM rules WHERE status = 'confirmed'"
            ).fetchall()
            for item in confirmed:
                self.bloom.add(item["signature"])
            self._persist_bloom(conn)
            conn.commit()
        finally:
            conn.close()

    def _persist_bloom(self, conn: sqlite3.Connection | None = None) -> None:
        payload = json.dumps(self.bloom.to_dict())
        if conn is None:
            c = self._connect()
            try:
                c.execute(
                    "INSERT OR REPLACE INTO bloom_state (id, payload) VALUES (1, ?)",
                    (payload,),
                )
                c.commit()
            finally:
                c.close()
        else:
            conn.execute(
                "INSERT OR REPLACE INTO bloom_state (id, payload) VALUES (1, ?)",
                (payload,),
            )

    def list_rules(self) -> list[Rule]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM rules ORDER BY created_at DESC").fetchall()
        finally:
            conn.close()
        return [Rule(**dict(r)) for r in rows]

    def get_confirmed_signatures(self) -> set[str]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT signature FROM rules WHERE status = 'confirmed'"
                ).fetchall()
            finally:
                conn.close()
        return {r["signature"] for r in rows}

    def bloom_maybe_contains(self, signature: str) -> bool:
        with self._lock:
            return signature in self.bloom

    def is_confirmed_attack(self, signature: str) -> bool:
        with self._lock:
            if signature not in self.bloom:
                return False
            return signature in self.get_confirmed_signatures()

    def add_rule(
        self,
        signature: str,
        attack_type: str,
        *,
        status: str = "confirmed",
        source: str = "manual",
        notes: str | None = None,
    ) -> Rule:
        signature = normalize_attack_signature(signature)
        attack_type = attack_type.strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", attack_type):
            raise ValueError("attack_type must be a short lowercase identifier")
        if status not in ("confirmed", "suspected"):
            raise ValueError("status must be 'confirmed' or 'suspected'")
        if notes is not None and len(notes) > 500:
            raise ValueError("notes must be at most 500 characters")
        rule = Rule(
            rule_id=str(uuid.uuid4()),
            signature=signature,
            attack_type=attack_type,
            status=status,
            source=source,
            created_at=_utc_now(),
            notes=notes,
        )
        with self._lock:
            previous_bloom = self.bloom.to_dict()
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO rules (rule_id, signature, attack_type, status, source, created_at, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rule.rule_id,
                        rule.signature,
                        rule.attack_type,
                        rule.status,
                        rule.source,
                        rule.created_at,
                        rule.notes,
                    ),
                )
                if status == "confirmed":
                    self.bloom.add(signature)
                    self._persist_bloom(conn)
                conn.commit()
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                self.bloom = BloomFilter.from_dict(previous_bloom)
                raise ValueError("A rule with this signature already exists") from exc
            except Exception:
                conn.rollback()
                self.bloom = BloomFilter.from_dict(previous_bloom)
                raise
            finally:
                conn.close()
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        with self._lock:
            previous_bloom = self.bloom
            conn = self._connect()
            try:
                cur = conn.execute("DELETE FROM rules WHERE rule_id = ?", (rule_id,))
                removed = cur.rowcount > 0
                if removed:
                    rebuilt = BloomFilter(self.bloom.config)
                    rows = conn.execute(
                        "SELECT signature FROM rules WHERE status = 'confirmed'"
                    ).fetchall()
                    for item in rows:
                        rebuilt.add(item["signature"])
                    self.bloom = rebuilt
                    self._persist_bloom(conn)
                conn.commit()
                return removed
            except Exception:
                conn.rollback()
                self.bloom = previous_bloom
                raise
            finally:
                conn.close()
