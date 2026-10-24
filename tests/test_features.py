import numpy as np
import pandas as pd
import pytest

from ml.config import SELECTED_FEATURES
from ml.features import (
    build_attack_signature,
    dataframe_to_raw_matrix,
    hex_byte_to_int,
    row_to_feature_vector,
    select_features,
)


def test_hex_byte_missing_is_minus_one():
    assert hex_byte_to_int(None) == -1.0
    assert hex_byte_to_int("ff") == 255.0


def test_row_to_feature_vector_shape():
    data = ["05", "28", "84", "66", "6d", "00", "00", "a2"]
    vec = row_to_feature_vector("0350", data, dlc=8)
    assert vec.shape == (9,)
    selected = select_features(vec.reshape(1, -1))
    assert selected.shape == (1, len(SELECTED_FEATURES))


def test_invalid_can_id_raises():
    with pytest.raises(ValueError):
        row_to_feature_vector("not_hex", ["00"] * 8)


def test_attack_signature_includes_payload():
    sig = build_attack_signature("0350", 8, ["05", "28", "84", "66", "6d", "00", "00", "a2"])
    assert sig.startswith("0350|dlc=8|")
    assert "05" in sig


def test_dataframe_uses_missing_encoding_after_dlc():
    row = {"CAN_ID": "0350", "DLC": 2}
    row.update({f"Data_{i}": f"{i + 1:02x}" for i in range(8)})
    raw = dataframe_to_raw_matrix(pd.DataFrame([row]))
    assert raw[0, 1] == 1
    assert raw[0, 2] == 2
    assert np.all(raw[0, 3:] == -1)


def test_dataframe_rejects_invalid_dlc():
    row = {"CAN_ID": "0350", "DLC": 9}
    row.update({f"Data_{i}": "00" for i in range(8)})
    with pytest.raises(ValueError, match="DLC"):
        dataframe_to_raw_matrix(pd.DataFrame([row]))
