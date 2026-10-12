# DevOps & MLOps Architecture & Pipeline Specification

## 1. System Pipeline Overview

```
                      Student Input / Dataset
                                ↓
                 Feature Engineering & Preprocessing
                         (StandardScaler)
                                ↓
                     Machine Learning Model
                    (RandomForestClassifier)
                                ↓
                      FastAPI Backend API
               (Inference, Validation, Hot-Retraining)
                                ↓
                    Git & GitHub Repository
                                ↓
                         CI Pipeline
                 ┌──────────────────────────┐
                 │  Flake8 Linting          │
                 │  15 Pytest Unit Tests    │
                 │  Code Coverage Report    │
                 └──────────────────────────┘
                                ↓
                       Docker Container
                      (python:3.11-slim)
                                ↓
                     Docker Hub / Registry
                                ↓
                         CD Pipeline
                  (Container Smoke Testing)
                                ↓
                         Deployment
                    (Host / Cloud Server)
                                ↓
                     Monitoring & Observability
              ┌─────────────────────────────────┐
              │  /health System Uptime          │
              │  /metrics Telemetry             │
              │  Statistical Data Drift Engine  │
              └─────────────────────────────────┘
```

---

## 2. DevOps Architecture Pillars

### 2.1 Git Branching Strategy
- **`main`:** Production-ready code. Protected branch; merges only occur via Pull Requests that pass CI checks.
- **`develop`:** Integration branch for ongoing sprint development.
- **`feature/*`:** Individual feature branches for developer implementation.

### 2.2 Continuous Integration (CI)
Automated by GitHub Actions (`.github/workflows/ci-cd.yml`):
- **Triggers:** On any `push` or `pull_request` to `main` or `develop`.
- **Environment:** Ubuntu Linux container with Python 3.11.
- **Quality Gates:**
  - `flake8`: Checks for syntax errors, import conventions, and complexity.
  - `pytest`: Executes 15 unit and integration tests across data, model, and API layers.
  - Test coverage generation.

### 2.3 Containerization (Docker)
- **Base Image:** `python:3.11-slim` for minimal container footprint and reduced attack surface.
- **Security:** Runs under an unprivileged non-root user (`appuser` with UID 1001).
- **Healthcheck:** Native Docker `HEALTHCHECK` periodically pings `http://localhost:8000/health`.
- **Volume Mounts:** Model artifacts (`models/`) and prediction logs (`logs/`) persist across container restarts.

### 2.4 Continuous Delivery (CD)
- Container build and smoke test validates that the container launches cleanly and returns HTTP 200 on `/health` within CI.
- Registry push simulation for Docker Hub image deployment.

---

## 3. MLOps Lifecycle & Governance

### 3.1 Model Artifact Versioning
Every model training run produces:
- `models/placement_model.joblib`: Serialized RandomForest model.
- `models/scaler.joblib`: Fitted StandardScaler parameters.
- `models/model_metadata.json`: Semantic version tag (`v1.2.0`), training timestamp, accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and ranked feature importances.

### 3.2 Continuous Training (CT) & Hot-Reloading
- **Trigger:** Accessible via `POST /api/v1/retrain` or automated GitHub Actions schedule (`.github/workflows/model-retrain.yml`).
- **Mechanism:** Retrains on updated data, validates metrics against the prior baseline, persists new artifacts, and updates the in-memory singleton predictor without taking the FastAPI server offline.

### 3.3 Data Drift Detection
- **Baseline:** Pre-computes mean and standard deviation for all 7 features during training.
- **Production Audit:** Every inference request is logged to `logs/predictions.jsonl`.
- **Statistical Test:** The `DataDriftDetector` computes the standardized score ($Z$-shift) of recent inference distributions:
  $$Z = \frac{|\mu_{\text{production}} - \mu_{\text{baseline}}|}{\sigma_{\text{baseline}}}$$
  If $Z > 1.25$, drift is flagged and automated retraining is recommended.
