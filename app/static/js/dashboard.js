/**
 * Minimalist Placement Analytics Dashboard JS
 * Vanilla JavaScript implementation for Agile ASD & DevOps Lab Mini Project
 */

document.addEventListener("DOMContentLoaded", () => {
  // Form elements
  const form = document.getElementById("prediction-form");
  const submitBtn = document.getElementById("submit-btn");
  const errorAlert = document.getElementById("error-alert");
  const errorList = document.getElementById("error-list");

  // Internship Pill Toggle
  const internOptions = document.querySelectorAll(".intern-pill");
  const internInput = document.getElementById("internship-val");

  internOptions.forEach((option) => {
    option.addEventListener("click", () => {
      internOptions.forEach((opt) => opt.classList.remove("active"));
      option.classList.add("active");
      internInput.value = option.getAttribute("data-value");
    });
  });

  // Presets
  const presets = {
    top: {
      cgpa: 9.2,
      tech: 92.0,
      apt: 88.0,
      comm: 86.0,
      intern: "yes",
      proj: 4,
      att: 94.0,
    },
    borderline: {
      cgpa: 7.2,
      tech: 70.0,
      apt: 68.0,
      comm: 66.0,
      intern: "no",
      proj: 2,
      att: 80.0,
    },
    atrisk: {
      cgpa: 5.6,
      tech: 45.0,
      apt: 48.0,
      comm: 52.0,
      intern: "no",
      proj: 0,
      att: 62.0,
    },
  };

  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const presetKey = btn.getAttribute("data-preset");
      const p = presets[presetKey];
      if (!p) return;

      document.getElementById("cgpa").value = p.cgpa;
      document.getElementById("technical_score").value = p.tech;
      document.getElementById("aptitude_score").value = p.apt;
      document.getElementById("communication_score").value = p.comm;
      document.getElementById("projects_completed").value = p.proj;
      document.getElementById("attendance").value = p.att;

      // Set internship
      internOptions.forEach((opt) => {
        if (opt.getAttribute("data-value") === p.intern) {
          opt.classList.add("active");
        } else {
          opt.classList.remove("active");
        }
      });
      internInput.value = p.intern;

      // Automatically trigger prediction for convenience
      form.dispatchEvent(new Event("submit"));
    });
  });

  // Handle Form Submission
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearErrors();

    const payload = {
      cgpa: parseFloat(document.getElementById("cgpa").value),
      technical_score: parseFloat(document.getElementById("technical_score").value),
      aptitude_score: parseFloat(document.getElementById("aptitude_score").value),
      communication_score: parseFloat(document.getElementById("communication_score").value),
      internship_experience: internInput.value,
      projects_completed: parseInt(document.getElementById("projects_completed").value, 10),
      attendance: parseFloat(document.getElementById("attendance").value),
    };

    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span>Computing Probability...</span>`;

    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const json = await resp.json();

      if (!resp.ok) {
        if (json.errors && Array.isArray(json.errors)) {
          showErrors(json.errors.map((err) => `${err.field}: ${err.message}`));
        } else {
          showErrors([json.message || "Failed to compute prediction."]);
        }
        return;
      }

      displayPrediction(json.data);
    } catch (err) {
      showErrors([`Network error: ${err.message}`]);
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `<span>Analyze Placement Probability</span> <span>→</span>`;
    }
  });

  function clearErrors() {
    errorAlert.style.display = "none";
    errorList.innerHTML = "";
  }

  function showErrors(messages) {
    errorList.innerHTML = messages.map((m) => `<li>${m}</li>`).join("");
    errorAlert.style.display = "block";
    errorAlert.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function displayPrediction(data) {
    const placeholder = document.getElementById("result-placeholder");
    const display = document.getElementById("result-display");
    const banner = document.getElementById("outcome-banner");
    const statusText = document.getElementById("outcome-status-text");
    const confTag = document.getElementById("confidence-tag");
    const probVal = document.getElementById("prob-val");
    const fill = document.getElementById("progress-fill");
    const factorsList = document.getElementById("factors-list");
    const recsList = document.getElementById("recs-list");

    placeholder.style.display = "none";
    display.style.display = "flex";

    const isLikely = data.prediction === "Likely";
    banner.className = `outcome-banner ${isLikely ? "likely" : "unlikely"}`;
    statusText.textContent = isLikely ? "Likely Placed" : "Unlikely Placed";
    confTag.textContent = `${data.confidence} Confidence`;

    probVal.textContent = data.probability_percentage;
    fill.className = `progress-fill ${isLikely ? "fill-likely" : "fill-unlikely"}`;
    fill.style.width = `${Math.min(Math.max(data.probability * 100, 2), 100)}%`;

    // Populate Key Factors
    factorsList.innerHTML = (data.key_factors || [])
      .map((f) => `<div class="factor-item"><span>✦</span> <span>${f}</span></div>`)
      .join("");

    // Populate Recommendations
    recsList.innerHTML = (data.recommendations || [])
      .map((r) => `<div class="rec-item"><span>→</span> <span>${r}</span></div>`)
      .join("");
  }

  // Load System Health & Model Analytics
  async function loadInitialData() {
    // 1. Health check
    try {
      const hResp = await fetch("/health");
      if (hResp.ok) {
        const hData = await hResp.json();
        document.getElementById("api-status-text").textContent = `API: ${hData.status.toUpperCase()} (${hData.uptime_seconds}s)`;
        document.getElementById("status-dot").style.background = "#10b981";
      }
    } catch {
      document.getElementById("api-status-text").textContent = "API: Offline";
      document.getElementById("status-dot").style.background = "#f43f5e";
    }

    // 2. Metrics & Baseline Comparison
    try {
      const mResp = await fetch("/api/metrics");
      if (mResp.ok) {
        const result = await mResp.json();
        renderMetrics(result.data);
      }
    } catch (e) {
      console.warn("Could not load initial metrics:", e);
    }
  }

  function renderMetrics(report) {
    const primary = report.primary_metrics || {};
    const cm = primary.confusion_matrix || {};
    const baselines = report.baseline_comparison || {};

    // Top Stats
    document.getElementById("stat-accuracy").textContent = primary.accuracy
      ? `${(primary.accuracy * 100).toFixed(1)}%`
      : "--";
    document.getElementById("stat-f1").textContent = primary.f1_score
      ? (primary.f1_score * 100).toFixed(1) + "%"
      : "--";
    document.getElementById("stat-recall").textContent = primary.recall
      ? `${(primary.recall * 100).toFixed(1)}%`
      : "--";
    document.getElementById("stat-roc").textContent = primary.roc_auc
      ? `${(primary.roc_auc * 100).toFixed(1)}%`
      : "--";

    // Confusion Matrix
    document.getElementById("cm-tp").textContent = cm.true_positives ?? "--";
    document.getElementById("cm-tn").textContent = cm.true_negatives ?? "--";
    document.getElementById("cm-fp").textContent = cm.false_positives ?? "--";
    document.getElementById("cm-fn").textContent = cm.false_negatives ?? "--";

    // Baseline Table
    const tbody = document.getElementById("baseline-tbody");
    if (tbody) {
      let rows = `
        <tr>
          <td><strong>Primary (Random Forest Pipeline)</strong></td>
          <td>${(primary.accuracy * 100).toFixed(1)}%</td>
          <td>${(primary.f1_score * 100).toFixed(1)}%</td>
          <td><span class="tag-delta delta-neutral">Reference</span></td>
        </tr>
      `;

      for (const [key, b] of Object.entries(baselines)) {
        const m = b.metrics || {};
        const delta = b.accuracy_delta || 0;
        const deltaStr = delta >= 0 ? `+${(delta * 100).toFixed(1)}%` : `${(delta * 100).toFixed(1)}%`;
        const deltaClass = delta >= 0 ? "delta-positive" : "delta-neutral";
        rows += `
          <tr>
            <td>${b.name || key}</td>
            <td>${(m.accuracy * 100).toFixed(1)}%</td>
            <td>${(m.f1_score * 100).toFixed(1)}%</td>
            <td><span class="tag-delta ${deltaClass}">${deltaStr}</span></td>
          </tr>
        `;
      }
      tbody.innerHTML = rows;
    }

    // Demo Mode notice check
    if (report.dataset && report.dataset.is_synthetic) {
      const banner = document.getElementById("demo-banner");
      if (banner) banner.style.display = "flex";
    }
  }

  loadInitialData();
});
