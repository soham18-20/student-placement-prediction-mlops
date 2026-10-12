# Viva Voce Examination Preparation Guide
### Agile Software Development + DevOps Lab Mini Project
**Project:** Student Placement Prediction System with MLOps (PlacementPulse)

---

## Section A: Agile Software Development & Scrum

### Q1: What is the Agile methodology, and why did you choose it over Waterfall?
**Answer:** Agile is an iterative, customer-centric software development methodology focusing on continuous delivery and adaptability. In this project, requirements (like incorporating drift monitoring or batch predictions) evolved iteratively across 4 sprints. Waterfall would have delayed feedback until the very end, whereas Agile allowed us to test each increment (data model → API → Docker → CI/CD) continuously.

### Q2: What were your Sprints and how did you organize your team?
**Answer:** We conducted 4 two-week sprints:
- **Sprint 1:** Data generation, feature engineering, and RandomForest baseline training.
- **Sprint 2:** FastAPI REST backend, Pydantic schemas, and Pytest unit test suite.
- **Sprint 3:** Docker containerization and GitHub Actions CI/CD pipeline automation.
- **Sprint 4:** Interactive Glassmorphism UI, continuous retraining endpoint, and drift telemetry.

### Q3: What is a User Story? Give an example from your project.
**Answer:** A User Story is an informal explanation of a feature from the perspective of an end user:
> *"As a student, I want to input my CGPA and technical scores, so that I can see my placement probability and skill recommendations."*  
It includes Acceptance Criteria written in Gherkin syntax (*Given/When/Then*).

### Q4: What is the "Definition of Done" (DoD) in your project?
**Answer:** A feature is only considered "Done" when:
1. Code passes PEP 8 and Flake8 linter checks.
2. Unit tests pass with zero failures in Pytest.
3. Endpoints are documented in Swagger OpenAPI docs.
4. Container builds successfully in Docker.
5. All code is merged via a passing GitHub Actions CI workflow.

---

## Section B: DevOps & CI/CD Pipeline

### Q5: What is CI/CD and what stages exist in your pipeline?
**Answer:** 
- **CI (Continuous Integration):** Automatically validates code changes. Our CI workflow triggers on commit/PR, installs dependencies, runs Flake8 linting, trains the baseline model, and executes 15 Pytest tests with code coverage.
- **CD (Continuous Delivery):** Builds the Docker image, conducts a container smoke test (pings `/health`), and prepares the image for deployment to Docker Hub or cloud environments.

### Q6: Why did you containerize the application with Docker?
**Answer:** Docker solves the "it works on my machine" problem by bundling the Python runtime, dependencies (`requirements.txt`), pre-trained model artifacts, and web server into an immutable image. It guarantees consistent behavior across developer machines, CI runners, and production servers.

### Q7: Explain your Dockerfile optimization techniques.
**Answer:**
1. **Lightweight Base:** Uses `python:3.11-slim` rather than the bloated full image, reducing image size by ~70%.
2. **Layer Caching:** Copies `requirements.txt` and runs `pip install` before copying application code, so dependency layers are cached when source code changes.
3. **Non-Root User:** Runs under an unprivileged `appuser` (UID 1001) for container security best practices.
4. **Native Healthcheck:** Includes `HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health || exit 1`.

### Q8: What is the role of GitHub Actions in your project?
**Answer:** GitHub Actions acts as our CI/CD orchestration engine. Whenever code is pushed to `main`, GitHub spins up an isolated Ubuntu virtual environment, executes our automated test suite, builds the Docker image, and uploads code coverage artifacts.

---

## Section C: AIML & MLOps

### Q9: Why did you choose RandomForestClassifier over Logistic Regression or Deep Learning?
**Answer:** 
1. **Non-linear Relationships:** Campus placement depends on non-linear thresholds (e.g. high CGPA compensating for moderate aptitude).
2. **Robustness to Overfitting:** Random forest ensembles multiple decision trees via bootstrap aggregation (bagging), reducing variance.
3. **Explainability & Feature Importance:** Random forest provides native Gini impurity feature importances, showing which attributes drive placement most (e.g., CGPA and Technical Score).
4. **Low Latency:** Inference completes in ~2ms, making it ideal for real-time web applications without needing GPU infrastructure.

### Q10: How is MLOps different from standard DevOps?
**Answer:** DevOps focuses on the lifecycle of **Code and Binaries** (build, test, deploy). MLOps extends DevOps to handle **Code + Data + Models**:
- Handling **Data Drift** and **Concept Drift**.
- **Model Versioning** and artifact tracking.
- **Continuous Training (CT)** pipelines that re-fit models when new data arrives.
- Monitoring inference probabilities and model degradation over time.

### Q11: Why is Accuracy alone not enough to evaluate this model?
**Answer:** If a dataset has class imbalance (e.g., 85% of students get placed), a naive model predicting "Placed" for everyone would achieve 85% accuracy but be useless. Therefore, we track:
- **Precision:** Of all students predicted "Likely", how many actually were placed?
- **Recall:** Of all placed students, what proportion did the model correctly identify?
- **F1-Score:** The harmonic mean of Precision and Recall (our model scores ~0.966).
- **ROC-AUC:** Area under the ROC curve (our model scores ~0.975), proving high discriminative ability.

### Q12: How does your system detect and handle Data Drift?
**Answer:** During training, we store the baseline mean and standard deviation for each feature. The system logs all production inferences to `logs/predictions.jsonl`. The `DataDriftDetector` computes the statistical $Z$-score shift of recent predictions against baseline parameters. If $Z > 1.25$, drift is flagged on the UI, and the user/system triggers automated retraining via `/api/v1/retrain`.

### Q13: How does your `/api/v1/retrain` endpoint work without server downtime?
**Answer:** The endpoint executes the training pipeline in a separate execution flow, writes new model artifacts with an incremented version tag (`v1.2.1`), and atomically reloads the in-memory singleton `PlacementPredictor` reference. New requests immediately use the updated model without restarting the FastAPI process.

---

## Section D: FastAPI & Architecture

### Q14: Why FastAPI instead of Flask or Django?
**Answer:** 
1. **Asynchronous Performance:** Native `async/await` support on Starlette and Uvicorn.
2. **Automatic OpenAPI/Swagger Docs:** Generates interactive API docs at `/docs` out of the box.
3. **Type Safety & Data Validation:** Pydantic automatically validates student input ranges and returns structured HTTP 422 errors for invalid data.

### Q15: What are the input features and output format of your API?
**Answer:**
- **Inputs (7):** CGPA (0-10), Technical Score (0-100), Aptitude Score (0-100), Communication Score (0-100), Internships (count), Projects (count), Attendance (percentage).
- **Outputs:** Prediction label (`Likely` / `Unlikely`), Probability (`86.4%`), Confidence (`High`), Key Factors, and Personalized Actionable Recommendations.
