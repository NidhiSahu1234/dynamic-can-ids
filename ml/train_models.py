"""
Train Decision Tree and Autoencoder; save artifacts and metrics.

Run from the project root:
    python -m ml.train_models

By default, training uses deterministic, uniform row samples from each processed
CSV. Metrics identify their sample sizes and must not be interpreted as full-file
metrics. Increase the caps only after checking available memory.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from tensorflow.keras import layers

from ml.config import (
    AE_BATCH_SIZE,
    AE_EPOCHS,
    AE_LEARNING_RATE,
    AE_THRESHOLD_PERCENTILE,
    ARTIFACTS_DIR,
    ARTIFACT_VERSION,
    CHUNK_SIZE,
    DT_PARAMS,
    NORMAL_FILE,
    PRETRAINING_FILES,
    RANDOM_STATE,
    SELECTED_FEATURES,
    TRAIN_TEST_SPLIT,
    UNSEEN_ATTACK_FILES,
)
from ml.features import dataframe_to_raw_matrix, select_features

CSV_DTYPES = {
    "CAN_ID": "string",
    "Label": "string",
    **{f"Data_{i}": "string" for i in range(8)},
}


def count_rows(path: Path) -> int:
    """Count rows using only the small label column; do not retain the CSV."""
    return sum(
        len(chunk)
        for chunk in pd.read_csv(
            path, usecols=["Label"], dtype={"Label": "string"}, chunksize=CHUNK_SIZE
        )
    )


def load_uniform_sample(
    path: Path, max_rows: int, seed: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Select an exact deterministic uniform sample without retaining the file.

    A count pass determines the sample positions; a second chunked pass converts
    only selected rows to features. Peak feature memory is bounded by max_rows.
    """
    if max_rows < 1:
        raise ValueError("Sample size must be at least 1")
    total_rows = count_rows(path)
    if total_rows == 0:
        raise ValueError(f"No records found in {path}")

    sample_size = min(total_rows, max_rows)
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(total_rows, size=sample_size, replace=False))

    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    position_cursor = 0
    offset = 0

    for chunk in pd.read_csv(path, dtype=CSV_DTYPES, chunksize=CHUNK_SIZE):
        end = offset + len(chunk)
        next_cursor = int(np.searchsorted(positions, end, side="left"))
        if next_cursor > position_cursor:
            local_positions = positions[position_cursor:next_cursor] - offset
            selected = chunk.iloc[local_positions]
            raw = dataframe_to_raw_matrix(selected)
            feature_parts.append(select_features(raw))
            normalized_labels = selected["Label"].str.strip().str.upper()
            if not normalized_labels.isin(("R", "T")).all():
                raise ValueError(f"Unsupported label found while sampling {path}")
            label_parts.append((normalized_labels == "T").to_numpy(dtype=np.int8))
        position_cursor = next_cursor
        offset = end

    if position_cursor != sample_size:
        raise RuntimeError(f"CSV row count changed while sampling {path}")
    return (
        np.concatenate(feature_parts, axis=0),
        np.concatenate(label_parts, axis=0),
        total_rows,
        sample_size,
    )


def batched_reconstruction_errors(
    model: keras.Model, X_scaled: np.ndarray, batch_size: int
) -> np.ndarray:
    # Keras performs bounded internal batches; keeping the reconstructed matrix
    # is inexpensive for these seven features and avoids one predict() call per batch.
    reconstruction = model.predict(X_scaled, verbose=0, batch_size=batch_size)
    return np.mean(np.abs(X_scaled - reconstruction), axis=1).astype(np.float32)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Binary metrics with a stable [normal, attack] matrix, including one-class data."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["normal", "attack"],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }


def build_autoencoder(input_dim: int = 7) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(5, activation="relu"),
            layers.Dense(3, activation="relu"),
            layers.Dense(5, activation="relu"),
            layers.Dense(input_dim, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=AE_LEARNING_RATE),
        loss="mae",
    )
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-train-rows-per-file",
        type=int,
        default=200_000,
        help="Maximum deterministic sample size from each Normal/DoS/Gear CSV (default: 200000).",
    )
    parser.add_argument(
        "--max-eval-rows-per-file",
        type=int,
        default=100_000,
        help="Maximum deterministic sample size from each Fuzzy/RPM CSV (default: 100000).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=AE_EPOCHS,
        help=f"Autoencoder epochs (default: {AE_EPOCHS}).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_train_rows_per_file < 1 or args.max_eval_rows_per_file < 1:
        raise SystemExit("Sample-size arguments must be at least 1")
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")

    started = time.time()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    runtime_tmp = ARTIFACTS_DIR.parent / "data" / ".training-tmp"
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    # Keras stages .keras archives in the system temp directory before replacing
    # the target. Keep this write inside the project for restricted environments.
    tempfile.tempdir = str(runtime_tmp.resolve())
    try:
        keras.utils.set_random_seed(RANDOM_STATE)
    except AttributeError:
        pass

    sampled: dict[str, dict] = {}
    samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    print("=" * 60)
    print("Decision Tree — deterministic processed-data samples")
    print("=" * 60)
    for index, (name, path) in enumerate(PRETRAINING_FILES.items()):
        if not path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {path}")
        print(f"Sampling {name} from {path.name} …")
        X_part, y_part, total_rows, sample_rows = load_uniform_sample(
            path, args.max_train_rows_per_file, RANDOM_STATE + index
        )
        samples[name] = (X_part, y_part)
        sampled[name] = {
            "file": str(path),
            "dataset_rows": total_rows,
            "sample_rows": sample_rows,
            "sample_label_counts": {
                "normal": int(np.sum(y_part == 0)),
                "attack": int(np.sum(y_part == 1)),
            },
        }
        print(f"  sampled {sample_rows:,} of {total_rows:,} rows")

    X_full = np.concatenate([samples[name][0] for name in PRETRAINING_FILES], axis=0)
    y_full = np.concatenate([samples[name][1] for name in PRETRAINING_FILES], axis=0)
    if len(np.unique(y_full)) != 2:
        raise ValueError("Training sample must contain both normal and attack labels")

    X_train_raw, X_heldout_raw, y_train, y_heldout = train_test_split(
        X_full,
        y_full,
        test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y_full,
    )
    del X_full, y_full

    dt_scaler = MinMaxScaler()
    X_train_scaled = dt_scaler.fit_transform(X_train_raw).astype(np.float32)
    X_heldout_scaled = dt_scaler.transform(X_heldout_raw).astype(np.float32)

    dt_model = DecisionTreeClassifier(**DT_PARAMS)
    print(f"Training DT on {len(y_train):,} sampled rows …")
    dt_model.fit(X_train_scaled, y_train)
    y_heldout_pred = dt_model.predict(X_heldout_scaled)
    dt_heldout_metrics = classification_metrics(y_heldout, y_heldout_pred)

    print("=" * 60)
    print("Autoencoder — sampled normal traffic only")
    print("=" * 60)
    X_normal_sample, y_normal_sample = samples["Normal"]
    X_normal = X_normal_sample[y_normal_sample == 0]
    if len(X_normal) < 2:
        raise ValueError("At least two normal rows are required for AE train/validation split")

    X_normal_train, X_normal_val = train_test_split(
        X_normal,
        test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    del X_normal_sample, y_normal_sample, X_normal

    ae_scaler = MinMaxScaler()
    X_normal_train_scaled = ae_scaler.fit_transform(X_normal_train).astype(np.float32)
    X_normal_val_scaled = ae_scaler.transform(X_normal_val).astype(np.float32)
    del X_normal_train, X_normal_val

    ae_model = build_autoencoder(len(SELECTED_FEATURES))
    print(f"Training AE on {len(X_normal_train_scaled):,} normal rows, {args.epochs} epochs …")
    ae_model.fit(
        X_normal_train_scaled,
        X_normal_train_scaled,
        epochs=args.epochs,
        batch_size=AE_BATCH_SIZE,
        shuffle=True,
        verbose=2,
    )

    normal_val_errors = batched_reconstruction_errors(
        ae_model, X_normal_val_scaled, AE_BATCH_SIZE
    )
    ae_threshold = float(np.percentile(normal_val_errors, AE_THRESHOLD_PERCENTILE))
    normal_val_predictions = (normal_val_errors > ae_threshold).astype(np.int8)
    ae_eval: dict[str, dict] = {
        "normal_validation_sample": {
            **classification_metrics(
                np.zeros(len(normal_val_errors), dtype=np.int8), normal_val_predictions
            ),
            "mean_error": float(normal_val_errors.mean()),
            "threshold_percentile": AE_THRESHOLD_PERCENTILE,
            "threshold": ae_threshold,
            "flagged_count": int(np.sum(normal_val_predictions)),
        }
    }

    dt_unseen: dict[str, dict] = {}
    unseen_sampling: dict[str, dict] = {}
    for index, (attack_name, path) in enumerate(UNSEEN_ATTACK_FILES.items()):
        if not path.exists():
            raise FileNotFoundError(f"Unseen-attack dataset not found: {path}")
        print(f"Evaluating {attack_name} on a uniform sample …")
        X_eval, y_eval, total_rows, sample_rows = load_uniform_sample(
            path,
            args.max_eval_rows_per_file,
            RANDOM_STATE + 100 + index,
        )
        X_eval_dt = dt_scaler.transform(X_eval).astype(np.float32)
        dt_unseen[attack_name] = {
            **classification_metrics(y_eval, dt_model.predict(X_eval_dt)),
            "dataset_rows": total_rows,
            "evaluation_sample_rows": sample_rows,
            "sampling": "deterministic uniform sample without replacement",
        }
        unseen_sampling[attack_name] = {
            "file": str(path),
            "dataset_rows": total_rows,
            "evaluation_sample_rows": sample_rows,
            "label_counts": {
                "normal": int(np.sum(y_eval == 0)),
                "attack": int(np.sum(y_eval == 1)),
            },
        }

        # The AE has its own scaler. Do not reuse the Decision Tree scaler.
        X_eval_ae = ae_scaler.transform(X_eval).astype(np.float32)
        errors = batched_reconstruction_errors(ae_model, X_eval_ae, AE_BATCH_SIZE)
        ae_predictions = (errors > ae_threshold).astype(np.int8)
        ae_eval[attack_name] = {
            **classification_metrics(y_eval, ae_predictions),
            "dataset_rows": total_rows,
            "evaluation_sample_rows": sample_rows,
            "sampling": "deterministic uniform sample without replacement",
            "mean_error": float(errors.mean()),
            "threshold": ae_threshold,
            "detected_above_threshold": int(np.sum(ae_predictions == 1)),
        }
        del X_eval, y_eval, X_eval_dt, X_eval_ae, errors, ae_predictions

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "selected_features": SELECTED_FEATURES,
        "missing_payload_encoding": "-1 for missing bytes and bytes at indexes >= DLC",
        "dt_params": DT_PARAMS,
        "ae_architecture": "7 -> 5 -> 3 -> 5 -> 7 (relu/sigmoid)",
        "ae_training": {
            "epochs": args.epochs,
            "batch_size": AE_BATCH_SIZE,
            "learning_rate": AE_LEARNING_RATE,
            "normal_train_samples": int(len(X_normal_train_scaled)),
            "normal_validation_samples": int(len(X_normal_val_scaled)),
        },
        "ae_threshold": ae_threshold,
        "ae_threshold_percentile": AE_THRESHOLD_PERCENTILE,
        "random_state": RANDOM_STATE,
        "train_test_split": TRAIN_TEST_SPLIT,
        "sampling": {
            "method": "deterministic uniform sample without replacement",
            "max_train_rows_per_file": args.max_train_rows_per_file,
            "max_eval_rows_per_file": args.max_eval_rows_per_file,
            "pretraining_datasets": sampled,
            "unseen_attack_datasets": unseen_sampling,
        },
    }
    metrics = {
        "evaluation_scope": (
            "Sampled metrics only. Training and validation use deterministic samples; "
            "Fuzzy and RPM are evaluated separately on deterministic uniform samples. "
            "These values are not full-dataset metrics."
        ),
        "decision_tree": {
            "heldout_pretraining_sample": dt_heldout_metrics,
            "unseen_attack_categories": dt_unseen,
        },
        "autoencoder": ae_eval,
        "training_duration_seconds": round(time.time() - started, 2),
    }

    joblib.dump(dt_model, ARTIFACTS_DIR / "dt_model.joblib")
    joblib.dump(dt_scaler, ARTIFACTS_DIR / "dt_scaler.joblib")
    ae_model.save(ARTIFACTS_DIR / "ae_model.keras")
    joblib.dump(ae_scaler, ARTIFACTS_DIR / "ae_scaler.joblib")
    (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\nArtifacts saved to", ARTIFACTS_DIR)
    print("DT held-out sampled F1:", dt_heldout_metrics["f1"])
    print("AE validation threshold (p{}): {:.6f}".format(AE_THRESHOLD_PERCENTILE, ae_threshold))
    print("Metrics are sampled and are not full-dataset metrics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
