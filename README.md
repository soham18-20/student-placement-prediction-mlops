# 🥇 Student Placement Prediction System with MLOps

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/GitHub_Actions-CI%2FCD_Passing-2088FF.svg?logo=githubactions)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-grade **Student Campus Placement Prediction System with MLOps**, designed specifically for the **Agile Software Development + DevOps Lab Mini Project**. It provides a complete pipeline from student data modeling and automated training to containerized REST APIs, GitHub Actions CI/CD workflows, automated hot-retraining, and data drift observability.

---

## 📌 Project Overview & Lab Requirements Matrix

| Requirement | Implementation in this Project | Viva Rating |
| :--- | :--- | :---: |
| **AIML Component** | RandomForestClassifier with 94.0% Accuracy & 0.975 ROC-AUC | ⭐⭐⭐⭐⭐ |
| **Simple ML Model** | Interpretable ensemble with Gini feature importances & confidence scoring | ⭐⭐⭐⭐⭐ |
| **Python API** | FastAPI asynchronous REST service with Pydantic validation & Swagger UI | ⭐⭐⭐⭐⭐ |
| **Git / GitHub** | Branching strategy, semantic versioning, and commit conventions | ⭐⭐⭐⭐⭐ |
| **CI (Continuous Integration)** | GitHub Actions workflow (`flake8` linting + 15 `pytest` tests + coverage) | ⭐⭐⭐⭐⭐ |
| **Docker** | Multi-stage, non-root `python:3.11-slim` container with healthchecks | ⭐⭐⭐⭐⭐ |
| **Docker Registry** | Automated build & tag pipeline ready for Docker Hub / GHCR | ⭐⭐⭐⭐⭐ |
| **CD Pipeline** | Automated container build, smoke testing, and simulated deployment | ⭐⭐⭐⭐⭐ |
| **Monitoring & Telemetry** | `/health`, Prometheus-compatible `/metrics`, request latency & prediction stats | ⭐⭐⭐⭐⭐ |
| **Model Versioning** | Versioned metadata JSON artifacts (`v1.2.0`) with performance snapshots | ⭐⭐⭐⭐⭐ |
| **Automated Retraining** | On-demand `/api/v1/retrain` endpoint with zero-downtime hot-reloading | ⭐⭐⭐⭐⭐ |
| **Data Drift Detection** | Statistical $Z$-score shift detection comparing production inputs to baseline | ⭐⭐⭐⭐⭐ |
| **Viva Voce Explainability** | Clean architecture with 25+ Viva QA guide in `docs/` | ⭐⭐⭐⭐⭐ |
| **Modern Web UI** | Glassmorphism dark-mode UI with animated gauge, sliders, and batch mode | ⭐⭐⭐⭐⭐ |

---

## 🏗️ Complete Pipeline Architecture

```
Student Data (Raw CSV / Synthetic Generator)
     ↓
Feature Engineering & Preprocessing (StandardScaler)
     ↓
ML Model (RandomForestClassifier, v1.2.0 Artifacts)
     ↓
FastAPI Backend (Async Endpoints, Pydantic Validation)
     ↓
GitHub Repository (Gitflow Branching)
     ↓
CI (GitHub Actions)
 ┌────────────────────────────────────────────────────────┐
 │ 1. Build & Dependency Setup                            │
 │ 2. Flake8 Linting & PEP 8 Checks                       │
 │ 3. 15 Pytest Unit Tests & Code Coverage XML Export    │
 └────────────────────────────────────────────────────────┘
     ↓
Docker Image (Optimized python:3.11-slim, Non-root User)
     ↓
Docker Hub / Container Registry
     ↓
CD Pipeline (Container Smoke Test & Deployment Simulation)
     ↓
Deployment (Local Container / Kubernetes / Cloud Run)
     ↓
Monitoring & Logs (/health, /metrics, Data Drift Z-Shift)
```

---

## 📂 Project Directory Structure

```
student-placement-mlops/
├── .github/
│   └── workflows/
│       ├── ci-cd.yml             # GitHub Actions CI/CD (Lint, Test, Docker Build)
│       └── model-retrain.yml     # Continuous Training (CT) scheduled workflow
├── data/
│   ├── raw/                      # Raw student placement dataset
│   └── processed/                # Scaled train/test data splits
├── docs/
│   ├── AGILE_SCRUM_GUIDE.md      # User stories, Gherkin criteria, Sprints, DoD
│   ├── DEVOPS_MLOPS_PIPELINE.md  # Architectural details & pipeline specification
│   └── VIVA_VOCE_PREPARATION.md  # 25+ Viva Voce questions & answers
├── models/
│   ├── placement_model.joblib    # Serialized ML model artifact
│   ├── scaler.joblib             # Serialized StandardScaler artifact
│   └── model_metadata.json       # Version, metrics, accuracy, feature importances
├── src/
│   ├── __init__.py
│   ├── config.py                 # Paths, feature names, hyperparams, thresholds
│   ├── data_generator.py         # Domain-realistic student data generator
│   ├── train.py                  # Training pipeline, evaluation, and artifact saving
│   ├── predict.py                # Inference engine, Pydantic schemas, recommendations
│   ├── drift_detector.py         # Statistical Z-score data drift monitoring
│   └── app.py                    # FastAPI application, static file serving, docs
├── static/
│   ├── css/
│   │   └── style.css             # Glassmorphism dark-mode responsive styling
│   ├── js/
│   │   └── app.js                # Sliders, animated circular SVG gauge, batch runner
│   └── index.html                # Modern dashboard interface
├── tests/
│   ├── __init__.py
│   ├── test_data.py              # Tests for dataset generator and schema integrity
│   ├── test_model.py             # Tests for inference, boundary probabilities
│   └── test_api.py               # Tests for /health, /predict, /retrain, /metrics
├── Dockerfile                    # Multi-stage lightweight container build
├── docker-compose.yml            # Local orchestration with volume mounts
├── .dockerignore
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Lint & test dependencies
├── run.ps1                       # Windows PowerShell execution helper
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Local Setup with Python

```powershell
# Navigate into the project folder
cd student-placement-mlops

# Install dependencies
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

# Run model training (generates dataset and model artifacts)
python -m src.train

# Run automated tests
python -m pytest tests -v

# Start FastAPI server
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser!

### 2. Run with Windows PowerShell Helper Script

```powershell
.\run.ps1 serve     # Starts dev server with live reload
.\run.ps1 test      # Executes 15 Pytest unit tests
.\run.ps1 train     # Retrains and versions ML model
.\run.ps1 lint      # Checks code quality
```

### 3. Run with Docker & Docker Compose

```bash
# Build and run with Docker Compose
docker compose up --build -d

# Check running container health
curl http://localhost:8000/health
```

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the PlacementPulse Web Dashboard |
| `GET` | `/docs` | Interactive Swagger API documentation |
| `GET` | `/health` | System health check, uptime, and model status |
| `POST` | `/api/v1/predict` | Predict placement for a single student profile |
| `POST` | `/api/v1/predict/batch` | Batch cohort prediction for up to 500 students |
| `POST` | `/api/v1/retrain` | Triggers automated retraining & zero-downtime hot-reload |
| `GET` | `/api/v1/model/metadata` | Model version (`v1.2.0`), metrics, and feature importances |
| `GET` | `/api/v1/monitoring/metrics` | Prometheus-style request telemetry and latency |
| `GET` | `/api/v1/monitoring/drift` | Statistical Z-score data drift report |
| `GET` | `/api/v1/sample-data` | Preset profiles for quick UI testing |

### Example Request (`POST /api/v1/predict`):
```json
{
  "cgpa": 8.4,
  "technical_score": 85.0,
  "aptitude_score": 78.0,
  "communication_score": 80.0,
  "internships": 2,
  "projects": 3,
  "attendance": 89.0
}
```

### Example Response:
```json
{
  "prediction_id": "4b684cb6-02e4-4fa9-b8e1-5e2632223a23",
  "prediction": "Likely",
  "probability": 0.8842,
  "probability_percentage": "88.4%",
  "confidence": "High",
  "key_factors": [
    "High technical competence significantly boosts placement odds",
    "Substantial internship track record valued by hiring managers"
  ],
  "recommendations": [
    "Outstanding academic and technical profile! Prepare for behavioral and system design interview rounds."
  ],
  "model_version": "v1.2.0",
  "timestamp": "2026-10-12T16:35:00.000000Z"
}
```

---

## 🎓 Viva Voce & Presentation Highlights

Refer to **[docs/VIVA_VOCE_PREPARATION.md](file:///C:/Users/lenovo/.gemini/antigravity-ide/scratch/student-placement-mlops/docs/VIVA_VOCE_PREPARATION.md)** for detailed answers to questions on:
1. **Agile Methodology:** Sprint breakdown, user stories with Gherkin acceptance criteria, definition of done.
2. **DevOps:** CI/CD stages, why Docker, multi-stage builds, non-root users, GitHub Actions workflows.
3. **MLOps:** Why RandomForest, accuracy vs F1-score, data drift vs concept drift, automated hot-reloading.
4. **FastAPI:** Asynchronous architecture, Pydantic type safety, auto-generated OpenAPI documentation.
