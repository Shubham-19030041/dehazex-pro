// ---- Tab switching ----
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("tab-" + tab.dataset.tab).classList.add("active");

    if (tab.dataset.tab === "compare") {
      loadCompareResults();
    }
  });
});

async function loadCompareResults() {
  const container = document.getElementById("compareContent");
  container.innerHTML = `<p class="hint">Loading comparison results...</p>`;

  try {
    const res = await fetch("/api/compare");
    const data = await res.json();

    if (!data.available) {
      container.innerHTML = `<p class="hint">${data.message}</p>`;
      return;
    }

    const columns = Object.keys(data.rows[0]);
    let html = `<table class="compare-table"><thead><tr>`;
    columns.forEach(col => { html += `<th>${col}</th>`; });
    html += `</tr></thead><tbody>`;
    data.rows.forEach(row => {
      html += `<tr>`;
      columns.forEach(col => {
        html += `<td>${row[col]}</td>`;
      });
      html += `</tr>`;
    });
    html += `</tbody></table>`;

    if (data.chart_available) {
      html += `<img class="compare-chart" src="/api/compare/chart?t=${Date.now()}" alt="comparison bar chart">`;
    }

    container.innerHTML = html;
  } catch (err) {
    container.innerHTML = `<p class="hint">Could not load comparison results.</p>`;
    console.error(err);
  }
}

document.getElementById("refreshCompareBtn").addEventListener("click", loadCompareResults);

// ---- Elements ----
const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const dehazeBtn = document.getElementById("dehazeBtn");
const allModelsBtn = document.getElementById("allModelsBtn");
const saveBtn = document.getElementById("saveBtn");
const autoSelect = document.getElementById("autoSelect");
const modelSelect = document.getElementById("modelSelect");
const statusPill = document.getElementById("statusPill");
const inputBox = document.getElementById("inputBox");
const outputBox = document.getElementById("outputBox");
const densityBadge = document.getElementById("densityBadge");
const weatherBadge = document.getElementById("weatherBadge");
const allModelsPanel = document.getElementById("allModelsPanel");
const allModelsGrid = document.getElementById("allModelsGrid");

let currentFile = null;
let lastOutputDataUrl = null;

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (!fileInput.files.length) return;
  currentFile = fileInput.files[0];
  const reader = new FileReader();
  reader.onload = e => {
    inputBox.innerHTML = `<img src="${e.target.result}" alt="hazy input">`;
  };
  reader.readAsDataURL(currentFile);
  outputBox.innerHTML = `<span class="placeholder">Click Dehaze to process</span>`;
  allModelsPanel.hidden = true;
  saveBtn.disabled = true;
  setStatus("Image loaded", "green");
});

function setStatus(text, color) {
  statusPill.textContent = text;
  statusPill.style.color = `var(--accent-${color})`;
  statusPill.style.borderColor = `var(--accent-${color})`;
}

function setMetric(id, value) {
  document.getElementById(id).textContent = value;
}

dehazeBtn.addEventListener("click", async () => {
  if (!currentFile) { setStatus("Upload an image first", "red"); return; }
  setStatus("Processing...", "orange");

  const formData = new FormData();
  formData.append("image", currentFile);
  formData.append("model", modelSelect.value);
  formData.append("auto_select", autoSelect.checked ? "true" : "false");

  try {
    const res = await fetch("/api/dehaze", { method: "POST", body: formData });
    const data = await res.json();
    if (data.error) { setStatus(data.error, "red"); return; }

    outputBox.innerHTML = `<img src="${data.output_image}" alt="dehazed output">`;
    lastOutputDataUrl = data.output_image;
    saveBtn.disabled = false;

    densityBadge.textContent = data.haze_density;
    weatherBadge.textContent = data.weather;

    setMetric("m-psnr", data.metrics.psnr + " dB");
    setMetric("m-ssim", data.metrics.ssim);
    setMetric("m-entropy", data.metrics.entropy);
    setMetric("m-edge", data.metrics.edge);
    setMetric("m-time", data.metrics.time_ms + " ms");
    setMetric("m-haze", data.haze_density);
    setMetric("m-weather", data.weather);
    setMetric("m-model", data.model_used);

    setStatus("Done", "green");
  } catch (err) {
    setStatus("Request failed", "red");
    console.error(err);
  }
});

allModelsBtn.addEventListener("click", async () => {
  if (!currentFile) { setStatus("Upload an image first", "red"); return; }
  setStatus("Running all models...", "orange");

  const formData = new FormData();
  formData.append("image", currentFile);

  try {
    const res = await fetch("/api/dehaze_all", { method: "POST", body: formData });
    const data = await res.json();

    // Find the best-scoring result (highest PSNR among models that succeeded)
    let bestName = null;
    let bestResult = null;
    for (const [name, result] of Object.entries(data.results)) {
      if (result.error) continue;
      if (!bestResult || result.metrics.psnr > bestResult.metrics.psnr) {
        bestName = name;
        bestResult = result;
      }
    }

    // Populate the main Output box + Live Metrics with the best result
    if (bestResult) {
      outputBox.innerHTML = `<img src="${bestResult.output_image}" alt="best dehazed output">`;
      lastOutputDataUrl = bestResult.output_image;
      saveBtn.disabled = false;

      weatherBadge.textContent = data.weather;

      setMetric("m-psnr", bestResult.metrics.psnr + " dB");
      setMetric("m-ssim", bestResult.metrics.ssim);
      setMetric("m-entropy", bestResult.metrics.entropy);
      setMetric("m-edge", bestResult.metrics.edge);
      setMetric("m-time", bestResult.metrics.time_ms + " ms");
      setMetric("m-weather", data.weather);
      setMetric("m-model", bestName + " (best of " + Object.keys(data.results).length + ")");
    }

    // Render all cards below, marking the best one
    allModelsGrid.innerHTML = "";
    for (const [name, result] of Object.entries(data.results)) {
      const card = document.createElement("div");
      card.className = "model-card" + (name === bestName ? " model-card-best" : "");
      if (result.error) {
        card.innerHTML = `<div class="model-card-title">${name}</div><p class="hint" style="padding:10px">${result.error}</p>`;
      } else {
        const bestTag = name === bestName ? `<span class="best-tag">BEST</span>` : "";
        card.innerHTML = `
          <div class="model-card-title">${name} ${bestTag}</div>
          <img src="${result.output_image}" alt="${name} output">
          <div class="psnr-tag">PSNR: ${result.metrics.psnr} dB</div>`;
      }
      allModelsGrid.appendChild(card);
    }
    allModelsPanel.hidden = false;
    setStatus("Done", "green");
  } catch (err) {
    setStatus("Request failed", "red");
    console.error(err);
  }
});

saveBtn.addEventListener("click", () => {
  if (!lastOutputDataUrl) return;
  const a = document.createElement("a");
  a.href = lastOutputDataUrl;
  a.download = "dehazed_output.png";
  a.click();
  setStatus("Downloaded!", "green");
});