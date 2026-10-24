# Dynamic CAN Intrusion Detection System

A research prototype inspired by **ML-BF: Responsive and Dynamic Intrusion Detection towards Intelligent Connected Vehicles**. It combines a Decision Tree for known attack patterns, a normal-traffic Autoencoder for anomaly detection, a Bloom filter backed by a persistent dynamic-rule store, a FastAPI service, and a React/Vite dashboard.

The original notebook, `Can_Intrusion_Detection_System.ipynb`, is preserved. Processed data is read from `Processed-Datasets/`; raw data in `Car-Hacking-Dataset/` is not reparsed by the training pipeline.

## Completion checklist

### Completed
- [x] Feature extraction, frame validation, and deterministic resource-conscious training.
- [x] Decision Tree and Autoencoder artifacts, with independent scalers, held-out threshold selection, and sampled confusion-matrix metrics.
- [x] FastAPI inference, batch prediction, metrics, alerts, dynamic rules, health, and dataset sampling endpoints.
- [x] SQLite-backed alert/rule persistence and Bloom-filter confirmation with rule-store cross-checks.
- [x] React dashboard build with real API data, dataset playback, filtering, alerts, metrics, and rule management.
- [x] Existing test suite, model inference checks, API integration checks, and production frontend build.

### Partially complete
- [ ] Training and evaluation use deterministic samples to respect available memory. Metrics are not full-dataset results.
- [ ] The dashboard replays processed CSV rows; it has no live vehicle connection.

### Remaining
- [ ] Optional live CAN hardware/interface integration, outside the current research prototype.

## Setup

Use a Python environment with the dependencies in `requirements.txt` and Node.js/npm for the dashboard.

From the project root:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The `.env` file is optional. The default artifact and SQLite paths are under the project root.

Install the frontend dependencies once, then use `npm ci` for later installs from the generated lockfile:

```powershell
Set-Location frontend
npm ci
```

## Train models

From the project root:

```powershell
python -m ml.train_models
```

Defaults use deterministic uniform samples without replacement:

- Up to 200,000 rows from each Normal, DoS, and Gear CSV for Decision Tree training and held-out evaluation.
- Up to 100,000 rows from each Fuzzy and RPM CSV for separate unseen-category evaluation.
- Autoencoder training uses only the sampled Normal rows, with a held-out normal validation split. The p99 validation reconstruction error sets the threshold.
- Both models use their own scaler. Scalers are fitted only on the corresponding training data.

Metrics are sample metrics. They must not be interpreted as full-dataset results. The JSON records sample counts, total dataset row counts, and the sampling method. Sample sizes and Autoencoder epochs can be adjusted:

```powershell
python -m ml.train_models --max-train-rows-per-file 200000 --max-eval-rows-per-file 100000 --epochs 50
```

Artifacts are written to `artifacts/`:

- `dt_model.joblib` and `dt_scaler.joblib`
- `ae_model.keras` and `ae_scaler.joblib`
- `metadata.json` and `metrics.json`

The API reports a degraded health state and returns HTTP 503 for inference until valid model artifacts exist. It does not substitute mock predictions.

## Start the backend

From the project root:

```powershell
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive API docs are at `http://127.0.0.1:8000/docs`.

Available endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | API and model-artifact status |
| POST | `/predict` | Detect one CAN frame |
| POST | `/predict/batch` | Detect 1–500 frames |
| GET | `/metrics` | Read measured training/evaluation metrics |
| GET | `/stats` | Session frame counts and persisted alert/rule counts |
| GET | `/alerts?limit=40` | Recent persisted non-normal detections |
| GET | `/rules` | List suspected and confirmed rules |
| POST | `/rules` | Add a validated rule |
| DELETE | `/rules/{rule_id}` | Remove a rule |
| GET | `/traffic/sample?source=normal&skip=0&limit=20` | Read rows for dashboard playback |

Single-frame request:

```json
{
  "can_id": "0350",
  "dlc": 8,
  "data": ["05", "28", "84", "66", "6d", "00", "00", "a2"]
}
```

Batch request:

```json
{
  "frames": [
    {
      "can_id": "0350",
      "dlc": 8,
      "data": ["05", "28", "84", "66", "6d", "00", "00", "a2"]
    }
  ]
}
```

A rule uses the canonical composite format `CAN_ID|dlc=N|byte:byte:…`. Only `confirmed` rules are inserted into the Bloom filter. `suspected` rules remain review-only. A possible Bloom hit is cross-checked against SQLite before it can confirm an attack.

Example:

```json
{
  "signature": "0350|dlc=8|05:28:84:66:6d:00:00:a2",
  "attack_type": "custom",
  "status": "suspected"
}
```

## Start the dashboard

In another terminal:

```powershell
Set-Location frontend
npm run dev
```

Open `http://localhost:5173`. The default API URL is `http://127.0.0.1:8000`; override it with `VITE_API_BASE_URL` in the project-root `.env` file if needed.

The dashboard displays live API health, session counts, dataset playback results, persisted alerts, measured model metrics, and rule management. Its playback controls replay rows from the processed CSV files.

## Deploy

The deployment keeps the trained model artifacts but excludes the raw and processed datasets, local SQLite database, research notebook/PDF, caches, and environment secrets. As a result, online frame prediction, rules, alerts, and metrics use the real saved models, while dataset playback is unavailable in the hosted version.

The root `render.yaml` configures the FastAPI service, the saved model directory, and a 1 GB persistent disk mounted at `/var/data` for SQLite. The selected Render plan provides 1 CPU and 2 GB RAM; Render currently lists it at $25/month plus $0.25/month for the 1 GB disk, before tax or any usage overages. Do not select a free backend plan: free instances have 512 MB RAM, spin down, and cannot attach persistent disks.

For Vercel, set the project root to `frontend`, use `npm ci` and `npm run build`, and publish `dist`. Set the production environment variable `VITE_API_BASE_URL` to the full FastAPI URL (for example, `https://<your-service>.onrender.com`). After Vercel assigns the frontend domain, set Render's `CORS_ORIGINS` to that HTTPS origin. The Vercel Hobby plan is free for personal, non-commercial use.

## Tests and build

From the project root:

```powershell
python -m pytest
```

Build the frontend:

```powershell
Set-Location frontend
npm run build
```

Inference and API prediction tests skip when trained artifacts are unavailable.

## Detection flow

1. Create a composite signature from CAN ID, DLC, and active payload bytes.
2. Check the Bloom filter and confirm a possible hit in the exact SQLite rule store.
3. Use the Decision Tree for known attack/normal classification.
4. If the Decision Tree predicts normal, use the Autoencoder's own scaler and compare reconstruction MAE against its normal-validation threshold.
5. Keep Autoencoder anomalies as suspected detections; they are not automatically promoted into confirmed rules.

Missing and inactive payload bytes use `-1` for model features. Missing active bytes use `--` in signatures; bytes beyond DLC are excluded.

## Known limitations

- The reported metrics come from deterministic samples of the processed datasets, not a full scan for model evaluation. Re-run with larger caps only after checking available memory and runtime.
- The Decision Tree held-out metrics are from a split of the sampled Normal/DoS/Gear records.
- On this run's samples, DT held-out F1 was 0.9967 (120,000 rows), while DT recall was 0.0006 on Fuzzy and 0.0000 on RPM. Autoencoder F1 was 0.6698 on Fuzzy and 0.9606 on RPM (100,000 rows per category). These are sample results, not full-dataset estimates.
- Fuzzy and RPM use separate deterministic labeled samples; the metrics JSON records sample and total row counts.
- Dataset playback is simulated. There is no in-vehicle CAN adapter or live bus connection.
- Confirmed signatures depend on exact CAN ID, DLC, and active payload bytes; the Bloom filter can have false positives, which are checked against SQLite.
- The saved model and metrics artifacts are selectively included for deployment; the local SQLite database and datasets remain ignored by Git.
