# Agile Software Development & Scrum Documentation

## 1. Project Vision
**Product Name:** Student Placement Prediction System with MLOps (PlacementPulse)  
**Vision Statement:** Deliver an automated, reliable, and interpretable machine learning system to help university placement cells and students predict campus recruitment outcomes, identify skill deficiencies, and track readiness using modern DevOps and MLOps practices.

---

## 2. Personas & Stakeholders
1. **Final Year Student (User):** Wants to assess their placement odds, view key strengths, and receive actionable tips to prepare for interviews.
2. **Placement Officer / Admin (Stakeholder):** Wants to evaluate cohorts in batches, spot students requiring mentorship, and track overall batch employability.
3. **MLOps / DevOps Engineer (Developer):** Wants automated CI/CD testing, containerized packaging, versioned model deployments, zero-downtime hot-reloading, and data drift observability.

---

## 3. Scrum Sprint Breakdown (4 Sprints)

### 🏃 Sprint 1: Data Modeling & Machine Learning Core (2 Weeks)
- **Goal:** Create domain-realistic student dataset, establish baseline classification models, and serialize model artifacts.
- **Deliverables:**
  - Synthetic data generation script (`src/data_generator.py`)
  - Exploratory Data Analysis & Feature selection
  - RandomForestClassifier training and evaluation pipeline (`src/train.py`)
  - Model serialization with metadata (`models/model_metadata.json`)

### 🏃 Sprint 2: FastAPI Backend & Automated Testing (2 Weeks)
- **Goal:** Build robust REST API with input validation, batch inference, and unit test suite.
- **Deliverables:**
  - FastAPI service with Pydantic validation (`src/predict.py`, `src/app.py`)
  - `/health`, `/api/v1/predict`, `/api/v1/predict/batch`, and `/api/v1/model/metadata` endpoints
  - 15 automated Pytest unit and integration tests (`tests/`)

### 🏃 Sprint 3: Containerization & CI/CD Pipeline (2 Weeks)
- **Goal:** Implement Docker packaging and automated GitHub Actions CI/CD workflows.
- **Deliverables:**
  - Production `Dockerfile`, `.dockerignore`, `docker-compose.yml`
  - GitHub Actions CI workflow (linting with Flake8, Pytest with coverage)
  - Docker build and container smoke test in CI

### 🏃 Sprint 4: MLOps Telemetry, Web UI & Viva Voce Polish (2 Weeks)
- **Goal:** Develop dark-theme glassmorphism frontend, automated retraining endpoint, statistical data drift detector, and lab presentation materials.
- **Deliverables:**
  - Interactive Web Dashboard with animated probability gauge and sliders (`static/`)
  - Hot-reloading retraining endpoint (`/api/v1/retrain`)
  - Data drift monitoring module (`src/drift_detector.py`)
  - Lab documentation and Viva Voce preparation guide

---

## 4. User Stories & Acceptance Criteria

### US-01: Individual Placement Likelihood Prediction
- **As a** student,
- **I want** to input my CGPA, technical skills, aptitude, communication scores, internships, projects, and attendance,
- **So that** I can instantly see my predicted placement outcome and probability percentage.
- **Acceptance Criteria (Gherkin):**
  ```gherkin
  Scenario: Successful Placement Prediction
    Given the student provides valid inputs (CGPA: 8.5, Technical: 85, Internships: 2)
    When the prediction request is submitted
    Then the system returns HTTP 200
    And the response contains "prediction" ("Likely")
    And the probability is between 0% and 100%
    And tailored recommendations are included
  ```

### US-02: Input Validation & Boundary Enforcement
- **As a** system engineer,
- **I want** invalid student values (e.g., CGPA > 10.0 or negative scores) to be rejected,
- **So that** bad data does not corrupt inference or telemetry.
- **Acceptance Criteria:**
  ```gherkin
  Scenario: Invalid CGPA submitted
    Given a student profile with CGPA 14.5
    When submitted to /api/v1/predict
    Then the API returns HTTP 422 Unprocessable Entity
  ```

### US-03: Continuous Integration (CI) Enforcement
- **As a** developer,
- **I want** every Git commit and pull request to automatically run code linting and unit tests,
- **So that** regressions are detected before merging to main.
- **Acceptance Criteria:**
  ```gherkin
  Scenario: GitHub Actions CI Execution
    Given a commit pushed to branch main
    When the CI pipeline executes
    Then Flake8 confirms no syntax errors
    And all 15 Pytest tests pass with 0 failures
    And a Docker image build succeeds
  ```

### US-04: Automated Retraining & Zero Downtime Hot-Reload
- **As an** MLOps engineer,
- **I want** an endpoint to retrain the ML model on demand or when data drift occurs,
- **So that** the production model stays accurate without restarting the server.
- **Acceptance Criteria:**
  ```gherkin
  Scenario: Trigger Hot Retrain
    Given the FastAPI server is running
    When a POST request is sent to /api/v1/retrain
    Then the model is retrained and validated
    And in-memory predictor is reloaded with the new version tag
    And the API returns the updated metrics and version delta
  ```

---

## 5. Definition of Done (DoD)
A task or user story is considered **Done** only when:
1. Code follows PEP 8 standards and passes Flake8 linting.
2. All relevant unit tests pass with `pytest`.
3. API endpoints are documented with Pydantic schemas and visible on `/docs`.
4. Feature functions correctly inside the Docker container.
5. Changes are committed with clear Git commit messages.
6. Documentation is updated.
