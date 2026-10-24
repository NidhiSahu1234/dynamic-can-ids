"""Small SQLite/PostgreSQL connection adapter for IDS persistence."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

try:
    from psycopg import IntegrityError as _PostgresIntegrityError
except ImportError:  # psycopg is only required when DATABASE_URL is configured.
    class _PostgresIntegrityError(Exception):
        pass


INTEGRITY_ERRORS = (sqlite3.IntegrityError, _PostgresIntegrityError)


def uses_postgres() -> bool:
    return bool(os.environ.get("DATABASE_URL", "").strip())


def connect(db_path: Path) -> Any:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if database_url:
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(database_url, row_factory=dict_row, connect_timeout=10)

    connection = sqlite3.connect(db_path, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_sqlite_parent(db_path: Path) -> None:
    if not uses_postgres():
        db_path.parent.mkdir(parents=True, exist_ok=True)


def execute(connection: Any, sql: str, parameters: tuple = ()) -> Any:
    if uses_postgres():
        sql = sql.replace("?", "%s")
    return connection.execute(sql, parameters)
