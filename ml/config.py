"""Shared ML configuration aligned with the audited notebook."""
from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _project_path(env_name: str, default: str) -> Path:
    configured = Path(os.environ.get(env_name, default)).expanduser()
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


PROCESSED_DIR = PROJECT_ROOT / "Processed-Datasets"
ARTIFACTS_DIR = _project_path("ARTIFACTS_DIR", "artifacts")
IDS_DB_PATH = _project_path("IDS_DB_PATH", "data/ids.db")

PRETRAINING_FILES = {
    "Normal": PROCESSED_DIR / "normal_processed.csv",
    "DoS": PROCESSED_DIR / "DoS_processed.csv",
    "Gear": PROCESSED_DIR / "gear_processed.csv",
}

UNSEEN_ATTACK_FILES = {
    "Fuzzy": PROCESSED_DIR / "Fuzzy_processed.csv",
    "RPM": PROCESSED_DIR / "RPM_processed.csv",
}

NORMAL_FILE = PROCESSED_DIR / "normal_processed.csv"

FEATURE_COLUMNS = [
    "CAN_ID_numeric",
    "Data_0",
    "Data_1",
    "Data_2",
    "Data_3",
    "Data_4",
    "Data_5",
    "Data_6",
    "Data_7",
]

SELECTED_FEATURES = [
    "CAN_ID_numeric",
    "Data_2",
    "Data_3",
    "Data_4",
    "Data_5",
    "Data_6",
    "Data_7",
]

PAYLOAD_COLUMNS = [f"Data_{i}" for i in range(8)]

DT_PARAMS = {
    "criterion": "entropy",
    "max_depth": 8,
    "min_samples_split": 10,
    "min_samples_leaf": 22,
    "random_state": 42,
}

RANDOM_STATE = 42
TRAIN_TEST_SPLIT = 0.20
CHUNK_SIZE = 200_000

AE_EPOCHS = 50
AE_BATCH_SIZE = 64
AE_LEARNING_RATE = 0.001
AE_THRESHOLD_PERCENTILE = 99.0

ARTIFACT_VERSION = "1.0.0"
