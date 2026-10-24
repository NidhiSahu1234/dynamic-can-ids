"""CAN frame validation and numerical feature extraction."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ml.config import FEATURE_COLUMNS, PAYLOAD_COLUMNS, SELECTED_FEATURES

_HEX_BYTE = re.compile(r"^[0-9a-fA-F]{1,2}$")


def normalize_can_id(can_id: str) -> str:
    return str(can_id).strip().lower()


def validate_can_id(can_id: str) -> None:
    cid = normalize_can_id(can_id)
    if not re.fullmatch(r"[0-9a-f]{1,8}", cid):
        raise ValueError(f"Invalid CAN ID: {can_id!r}")


def validate_dlc(dlc: int) -> None:
    if type(dlc) is not int or dlc < 0 or dlc > 8:
        raise ValueError(f"DLC must be an integer 0–8, got {dlc!r}")


def validate_payload_bytes(data_bytes: list[str | None], dlc: int) -> None:
    validate_dlc(dlc)
    if len(data_bytes) != 8:
        raise ValueError("Payload must contain exactly 8 byte slots (Data_0 … Data_7)")
    for i in range(dlc):
        cell = data_bytes[i]
        if (
            cell is None
            or (isinstance(cell, float) and np.isnan(cell))
            or str(cell).strip().upper() in ("", "<NA>", "NAN", "NONE")
        ):
            continue
        text = str(cell).strip()
        if not _HEX_BYTE.fullmatch(text):
            raise ValueError(f"Invalid hex byte at Data_{i}: {cell!r}")


def hex_byte_to_int(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return -1.0
    text = str(value).strip()
    if not text or text.upper() in ("<NA>", "NAN", "NONE"):
        return -1.0
    return float(int(text, 16))


def series_hex_to_float(series: pd.Series) -> np.ndarray:
    """Hex payload column → float32; missing / invalid → -1."""
    out = np.empty(len(series), dtype=np.float32)
    for i, value in enumerate(series.to_list()):
        out[i] = hex_byte_to_int(value)
    return out


def can_id_to_numeric(can_id: str) -> float:
    validate_can_id(can_id)
    return float(int(normalize_can_id(can_id), 16))


def row_to_feature_vector(
    can_id: str,
    data_bytes: list[Any],
    dlc: int = 8,
) -> np.ndarray:
    """Return 9-feature raw vector (all payload columns + CAN ID)."""
    validate_payload_bytes(data_bytes, dlc)
    vector = np.empty(len(FEATURE_COLUMNS), dtype=np.float32)
    vector[0] = can_id_to_numeric(can_id)
    for i, col in enumerate(PAYLOAD_COLUMNS, start=1):
        vector[i] = hex_byte_to_int(data_bytes[i - 1]) if i - 1 < dlc else -1.0
    return vector


def dataframe_to_raw_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert processed CSV chunk to float32 feature matrix."""
    n = len(df)
    out = np.empty((n, len(FEATURE_COLUMNS)), dtype=np.float32)
    out[:, 0] = df["CAN_ID"].apply(lambda x: int(str(x), 16)).to_numpy(dtype=np.float32)
    dlcs: np.ndarray | None = None
    if "DLC" in df.columns:
        parsed_dlcs = pd.to_numeric(df["DLC"], errors="coerce").to_numpy(dtype=np.float32)
        if (
            not np.isfinite(parsed_dlcs).all()
            or np.any(parsed_dlcs < 0)
            or np.any(parsed_dlcs > 8)
            or np.any(parsed_dlcs != np.floor(parsed_dlcs))
        ):
            raise ValueError("DLC column must contain integer values from 0 through 8")
        dlcs = parsed_dlcs.astype(np.int8)
    for i, col in enumerate(PAYLOAD_COLUMNS, start=1):
        values = series_hex_to_float(df[col])
        if dlcs is not None:
            values[dlcs <= i - 1] = -1.0
        out[:, i] = values
    return out


def select_features(raw: np.ndarray) -> np.ndarray:
    indices = [FEATURE_COLUMNS.index(name) for name in SELECTED_FEATURES]
    return raw[:, indices]


def label_to_target(label: str) -> int:
    """R → 0 (normal), T → 1 (attack)."""
    normalized = str(label).strip().upper()
    if normalized not in ("R", "T"):
        raise ValueError(f"Unsupported CAN traffic label: {label!r}")
    return 1 if normalized == "T" else 0


def build_attack_signature(can_id: str, dlc: int, data_bytes: list[Any]) -> str:
    """
    Composite signature: CAN ID + DLC + active payload bytes.
    CAN ID alone is insufficient; payload distinguishes message content.
    """
    validate_can_id(can_id)
    validate_dlc(dlc)
    validate_payload_bytes(data_bytes, dlc)
    cid = normalize_can_id(can_id)
    active = []
    for i in range(dlc):
        cell = data_bytes[i]
        if (
            cell is None
            or (isinstance(cell, float) and np.isnan(cell))
            or str(cell).strip().upper() in ("", "<NA>", "NAN", "NONE")
        ):
            active.append("--")
        else:
            active.append(f"{int(str(cell).strip(), 16):02x}")
    return f"{cid}|dlc={dlc}|" + ":".join(active)


def normalize_attack_signature(signature: str) -> str:
    """Validate and canonicalize a persisted/API composite attack signature."""
    if not isinstance(signature, str):
        raise ValueError("Attack signature must be a string")
    parts = signature.strip().split("|", 2)
    if len(parts) != 3:
        raise ValueError("Signature format must be CAN_ID|dlc=N|byte:byte:…")
    can_id, dlc_field, payload = parts
    validate_can_id(can_id)
    match = re.fullmatch(r"dlc=(\d+)", dlc_field.lower())
    if not match:
        raise ValueError("Signature DLC must use the form dlc=N")
    dlc = int(match.group(1))
    validate_dlc(dlc)
    tokens = payload.split(":") if payload else []
    if len(tokens) != dlc:
        raise ValueError(f"Signature must contain exactly {dlc} active payload bytes")
    normalized: list[str | None] = []
    for token in tokens:
        if token == "--":
            normalized.append(None)
        elif _HEX_BYTE.fullmatch(token):
            normalized.append(token)
        else:
            raise ValueError(f"Invalid byte in attack signature: {token!r}")
    return build_attack_signature(can_id, dlc, normalized + [None] * (8 - dlc))
