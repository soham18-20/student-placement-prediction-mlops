// PlacementPulse - Frontend Application Logic
document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const inputs = {
    cgpa: document.getElementById("input-cgpa"),
    tech: document.getElementById("input-tech"),
    aptitude: document.getElementById("input-aptitude"),
    comm: document.getElementById("input-comm"),
    internships: document.getElementById("input-internships"),
    projects: document.getElementById("input-projects"),
    attendance: document.getElementById("input-attendance"),
  };

  const valPills = {
    cgpa: document.getElementById("val-cgpa"),
    tech: document.getElementById("val-tech"),
    aptitude: document.getElementById("val-aptitude"),
    comm: document.getElementById("val-comm"),
    internships: document.getElementById("val-internships"),
    projects: document.getElementById("val-projects"),
    attendance: document.getElementById("val-attendance"),
  };

  // Prediction Form & CTA
  const form = document.getElementById("prediction-form");
  const gaugeCircle = document.getElementById("gauge-circle");
  const resultPercent = document.getElementById("result-percent");
  const resultBadge = document.getElementById("result-badge");
  const resultPrediction = document.getElementById("result-prediction");
  const resultIcon = document.getElementById("result-icon");
  const resultConfidence = document.getElementById("result-confidence");
  const resultSummary = document.getElementById("result-summary");
  const recommendationList = document.getElementById("recommendation-list");
  const factorsList = document.getElementById("factors-list");
  const modelVersionTag = document.getElementById("model-version-tag");

  // Competency Bars
  const bars = {
    cgpa: { val: document.getElementById("bar-val-cgpa"), fill: document.getElementById("bar-fill-cgpa") },
    tech: { val: document.getElementById("bar-val-tech"), fill: document.getElementById("bar-fill-tech") },
    apt: { val: document.getElementById("bar-val-apt"), fill: document.getElementById("bar-fill-apt") },
    comm: { val: document.getElementById("bar-val-comm"), fill: document.getElementById("bar-fill-comm") },
  };

  // System Status
  const statusPill = document.getElementById("system-status-pill");
  const statusText = document.getElementById("system-status-text");

  // Tab Navigation
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const target = btn.getAttribute("data-tab");
      const targetPane = document.getElementById(target);
      if (targetPane) targetPane.classList.add("active");

      if (target === "mlops-view") {
        fetchMetadataAndMetrics();
      }
    });
  });

  // Sync Slider values with pill labels
  function syncValueDisplay() {
    valPills.cgpa.textContent = parseFloat(inputs.cgpa.value).toFixed(2);
    valPills.tech.textContent = `${inputs.tech.value} / 100`;
    valPills.aptitude.textContent = `${inputs.aptitude.value} / 100`;
    valPills.comm.textContent = `${inputs.comm.value} / 100`;
    valPills.internships.textContent = inputs.internships.value;
    valPills.projects.textContent = inputs.projects.value;
    valPills.attendance.textContent = `${parseFloat(inputs.attendance.value).toFixed(1)}%`;

    // Update benchmark bar previews
    const cgpaPct = Math.round((parseFloat(inputs.cgpa.value) / 10.0) * 100);
    bars.cgpa.val.textContent = `${cgpaPct}%`;
    bars.cgpa.fill.style.width = `${cgpaPct}%`;

    bars.tech.val.textContent = `${inputs.tech.value}%`;
    bars.tech.fill.style.width = `${inputs.tech.value}%`;

    bars.apt.val.textContent = `${inputs.aptitude.value}%`;
    bars.apt.fill.style.width = `${inputs.aptitude.value}%`;

    bars.comm.val.textContent = `${inputs.comm.value}%`;
    bars.comm.fill.style.width = `${inputs.comm.value}%`;
  }

  Object.values(inputs).forEach(input => {
    input.addEventListener("input", syncValueDisplay);
  });

  // Preset Buttons
  document.getElementById("btn-preset-high").addEventListener("click", () => {
    setProfile(9.2, 92, 88, 85, 2, 4, 92.0);
    doPredict();
  });

  document.getElementById("btn-preset-borderline").addEventListener("click", () => {
    setProfile(7.1, 68, 64, 65, 1, 2, 78.0);
    doPredict();
  });

  document.getElementById("btn-preset-risk").addEventListener("click", () => {
    setProfile(5.8, 45, 48, 52, 0, 1, 64.0);
    doPredict();
  });

  function setProfile(cgpa, tech, apt, comm, intern, proj, att) {
    inputs.cgpa.value = cgpa;
    inputs.tech.value = tech;
    inputs.aptitude.value = apt;
    inputs.comm.value = comm;
    inputs.internships.value = intern;
    inputs.projects.value = proj;
    inputs.attendance.value = att;
    syncValueDisplay();
  }

  // Update Animated Gauge
  // Circumference = 2 * PI * 68 ≈ 427.25
  const CIRCUMFERENCE = 2 * Math.PI * 68;
  function updateGauge(prob) {
    const clamped = Math.max(0, Math.min(1, prob));
    const offset = CIRCUMFERENCE - (clamped * CIRCUMFERENCE);
    gaugeCircle.style.strokeDashoffset = offset;

    // Color gradient based on probability
    if (clamped >= 0.70) {
      gaugeCircle.style.stroke = "#10b981"; // Emerald
    } else if (clamped >= 0.45) {
      gaugeCircle.style.stroke = "#f59e0b"; // Amber
    } else {
      gaugeCircle.style.stroke = "#f43f5e"; // Rose
    }
  }

  // Handle Predict Form
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    doPredict();
  });

  async function doPredict() {
    const payload = {
      cgpa: parseFloat(inputs.cgpa.value),
      technical_score: parseFloat(inputs.tech.value),
      aptitude_score: parseFloat(inputs.aptitude.value),
      communication_score: parseFloat(inputs.comm.value),
      internships: parseInt(inputs.internships.value, 10),
      projects: parseInt(inputs.projects.value, 10),
      attendance: parseFloat(inputs.attendance.value),
    };

    try {
      const response = await fetch("/api/v1/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Inference error: ${response.statusText}`);
      }

      const data = await response.json();
      renderPrediction(data);
    } catch (err) {
      console.error(err);
      alert("Failed to fetch prediction from API. Check server console.");
    }
  }

  function renderPrediction(data) {
    // 1. Gauge
    resultPercent.textContent = data.probability_percentage;
    updateGauge(data.probability);

    // 2. Verdict Badge
    if (data.prediction === "Likely") {
      resultBadge.className = "prediction-badge badge-likely";
      resultIcon.textContent = "✓";
      resultPrediction.textContent = "Likely to be Placed";
      resultSummary.textContent = "Student demonstrates strong core competencies exceeding standard campus recruitment benchmarks.";
    } else {
      resultBadge.className = "prediction-badge badge-unlikely";
      resultIcon.textContent = "✕";
      resultPrediction.textContent = "Unlikely to be Placed";
      resultSummary.textContent = "Key qualification parameters fall below competitive campus placement cutoffs.";
    }

    resultConfidence.textContent = `${data.confidence} Confidence (${data.model_version})`;
    modelVersionTag.textContent = `Model: ${data.model_version}`;

    // 3. Recommendations
    recommendationList.innerHTML = "";
    data.recommendations.forEach(tip => {
      const li = document.createElement("li");
      li.textContent = tip;
      recommendationList.appendChild(li);
    });

    // 4. Key Factors
    factorsList.innerHTML = "";
    data.key_factors.forEach(fac => {
      const li = document.createElement("li");
      li.textContent = fac;
      factorsList.appendChild(li);
    });
  }

  // Health Check
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      const health = await res.json();
      if (health.status === "healthy") {
        statusPill.className = "status-pill status-online";
        statusText.textContent = `API: Online (${health.version})`;
      } else {
        statusPill.className = "status-pill status-degraded";
        statusText.textContent = `API: Degraded`;
      }
    } catch {
      statusPill.className = "status-pill status-degraded";
      statusText.textContent = `API: Offline`;
    }
  }

  // Fetch Metadata & Metrics for MLOps tab
  async function fetchMetadataAndMetrics() {
    try {
      const [metaRes, metricRes, driftRes] = await Promise.all([
        fetch("/api/v1/model/metadata"),
        fetch("/api/v1/monitoring/metrics"),
        fetch("/api/v1/monitoring/drift"),
      ]);

      const meta = await metaRes.json();
      const metrics = await metricRes.json();
      const drift = await driftRes.json();

      if (meta && meta.metrics) {
        document.getElementById("mlops-acc").textContent = `${(meta.metrics.accuracy * 100).toFixed(1)}%`;
        document.getElementById("mlops-f1").textContent = meta.metrics.f1_score;
        document.getElementById("mlops-auc").textContent = meta.metrics.roc_auc;

        // Confusion Matrix
        const cm = meta.metrics.confusion_matrix;
        if (cm) {
          document.getElementById("cm-tn").textContent = `TN: ${cm.true_negatives}`;
          document.getElementById("cm-fp").textContent = `FP: ${cm.false_positives}`;
          document.getElementById("cm-fn").textContent = `FN: ${cm.false_negatives}`;
          document.getElementById("cm-tp").textContent = `TP: ${cm.true_positives}`;
        }

        // Feature Importance Bars
        const impContainer = document.getElementById("feature-importance-bars");
        impContainer.innerHTML = "";
        if (meta.feature_importances) {
          Object.entries(meta.feature_importances).forEach(([feat, weight]) => {
            const pct = (weight * 100).toFixed(1);
            const item = document.createElement("div");
            item.className = "importance-bar-item";
            item.innerHTML = `
              <div class="importance-bar-head">
                <span>${feat.replace("_", " ").toUpperCase()}</span>
                <span class="font-mono">${pct}%</span>
              </div>
              <div class="bar-track">
                <div class="bar-fill" style="width: ${pct}%;"></div>
              </div>
            `;
            impContainer.appendChild(item);
          });
        }
      }

      // Drift
      if (drift) {
        const driftBadge = document.getElementById("mlops-drift-status");
        driftBadge.textContent = drift.status;
        driftBadge.className = drift.drift_detected ? "kpi-val text-rose" : "kpi-val text-success";
        document.getElementById("mlops-drift-sub").textContent = drift.recommendation;
      }

      // Telemetry
      if (metrics) {
        document.getElementById("tel-version").textContent = metrics.model_version;
        document.getElementById("tel-total").textContent = metrics.total_requests;
        document.getElementById("tel-latency").textContent = `${metrics.average_latency_seconds}s`;
      }
      if (meta.training_timestamp) {
        document.getElementById("tel-timestamp").textContent = new Date(meta.training_timestamp).toLocaleTimeString();
      }

    } catch (e) {
      console.warn("Could not load MLOps telemetry", e);
    }
  }

  // Trigger Retraining
  const retrainBtn = document.getElementById("btn-trigger-retrain");
  const retrainSpinner = document.getElementById("retrain-spinner");
  const retrainBtnText = document.getElementById("retrain-btn-text");
  const retrainMsg = document.getElementById("retrain-status-msg");

  retrainBtn.addEventListener("click", async () => {
    retrainBtn.disabled = true;
    retrainSpinner.style.display = "inline-block";
    retrainBtnText.textContent = "Retraining Model & Re-evaluating...";
    retrainMsg.style.display = "none";

    try {
      const res = await fetch("/api/v1/retrain", { method: "POST" });
      const data = await res.json();

      retrainMsg.style.display = "block";
      retrainMsg.innerHTML = `
        <strong>Retraining Successful!</strong><br>
        Version: <code>${data.new_version}</code><br>
        New Accuracy: ${(data.metrics.accuracy * 100).toFixed(2)}% (Delta: ${data.accuracy_change.delta >= 0 ? '+' : ''}${data.accuracy_change.delta})<br>
        F1 Score: ${data.metrics.f1_score}
      `;

      // Refresh display
      fetchMetadataAndMetrics();
    } catch (err) {
      alert("Retraining failed: " + err);
    } finally {
      retrainBtn.disabled = false;
      retrainSpinner.style.display = "none";
      retrainBtnText.textContent = "Trigger Automated Retraining";
    }
  });

  // Batch Cohort Simulation
  const sampleCohort = [
    { cgpa: 9.4, technical_score: 95.0, aptitude_score: 92.0, communication_score: 88.0, internships: 3, projects: 5, attendance: 95.0 },
    { cgpa: 8.8, technical_score: 86.0, aptitude_score: 82.0, communication_score: 84.0, internships: 2, projects: 4, attendance: 91.0 },
    { cgpa: 8.1, technical_score: 79.0, aptitude_score: 75.0, communication_score: 78.0, internships: 1, projects: 3, attendance: 86.0 },
    { cgpa: 7.7, technical_score: 74.0, aptitude_score: 70.0, communication_score: 72.0, internships: 1, projects: 2, attendance: 82.0 },
    { cgpa: 7.2, technical_score: 66.0, aptitude_score: 64.0, communication_score: 68.0, internships: 1, projects: 2, attendance: 79.0 },
    { cgpa: 6.8, technical_score: 60.0, aptitude_score: 58.0, communication_score: 62.0, internships: 0, projects: 2, attendance: 76.0 },
    { cgpa: 6.2, technical_score: 54.0, aptitude_score: 52.0, communication_score: 58.0, internships: 0, projects: 1, attendance: 71.0 },
    { cgpa: 5.7, technical_score: 46.0, aptitude_score: 48.0, communication_score: 50.0, internships: 0, projects: 1, attendance: 65.0 },
    { cgpa: 8.5, technical_score: 88.0, aptitude_score: 80.0, communication_score: 82.0, internships: 2, projects: 3, attendance: 89.0 },
    { cgpa: 7.9, technical_score: 76.0, aptitude_score: 72.0, communication_score: 75.0, internships: 2, projects: 3, attendance: 84.0 },
  ];

  document.getElementById("btn-load-sample-cohort").addEventListener("click", async () => {
    const tbody = document.getElementById("batch-tbody");
    tbody.innerHTML = `<tr><td colspan="10" class="empty-table-msg">Executing batch inference for 10 students...</td></tr>`;

    try {
      const res = await fetch("/api/v1/predict/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(sampleCohort),
      });

      const results = await res.json();
      tbody.innerHTML = "";

      results.forEach((res, i) => {
        const student = sampleCohort[i];
        const tr = document.createElement("tr");
        const isLikely = res.prediction === "Likely";
        tr.innerHTML = `
          <td class="font-mono">#STU-${1001 + i}</td>
          <td>${student.cgpa}</td>
          <td>${student.technical_score}</td>
          <td>${student.aptitude_score}</td>
          <td>${student.communication_score}</td>
          <td>${student.internships}</td>
          <td>${student.projects}</td>
          <td>${student.attendance}%</td>
          <td>
            <span class="badge-pill ${isLikely ? 'badge-likely' : 'badge-unlikely'}">
              ${res.prediction}
            </span>
          </td>
          <td class="font-mono"><strong>${res.probability_percentage}</strong></td>
        `;
        tbody.appendChild(tr);
      });
    } catch (e) {
      alert("Batch inference failed: " + e);
    }
  });

  // Initial Boot
  syncValueDisplay();
  checkHealth();
  doPredict();
});
