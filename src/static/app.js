const featureColumns = ["accel_x", "accel_y", "accel_z"];
const fileInput = document.getElementById("file");
const result = document.getElementById("result");
const sendButton = document.getElementById("send");
const chartPanel = document.getElementById("chartPanel");
const windowSelect = document.getElementById("windowSelect");
const channelSelect = document.getElementById("channelSelect");
const chartCanvas = document.getElementById("vibrationChart");
const chartNote = document.getElementById("chartNote");

let currentRows = [];
let currentWindows = [];
let currentAverageWindow = null;
let currentPayload = null;

function formatNumber(value) {
  return Number(value).toLocaleString("es-ES", {
    minimumFractionDigits: 6,
    maximumFractionDigits: 6
  });
}

function renderPending(message) {
  result.className = "status pending";
  result.innerHTML = `
    <div class="status-title">Analizando vibraciones</div>
    <div>${message}</div>
  `;
  chartPanel.className = "panel chart-panel";
}

function renderError(message) {
  result.className = "status error";
  result.innerHTML = `
    <div class="status-title">No se pudo analizar el archivo</div>
    <div>${message}</div>
  `;
  chartPanel.className = "panel chart-panel";
}

function renderResult(payload) {
  const isAlert = Boolean(payload.anomaly);
  const statusClass = isAlert ? "alert" : "ok";
  const title = isAlert ? "Via posiblemente defectuosa" : "Via en estado normal";
  const windows = payload.anomaly_windows && payload.anomaly_windows.length
    ? payload.anomaly_windows.slice(0, 30).join(", ") + (payload.anomaly_windows.length > 30 ? "..." : "")
    : "Sin ventanas criticas";

  result.className = `status ${statusClass}`;
  result.innerHTML = `
    <div class="status-title">${title}</div>
    <div>${payload.message}</div>
    <div class="metric-grid">
      <div class="metric">
        <span>Ventanas analizadas</span>
        <strong>${payload.window_count}</strong>
      </div>
      <div class="metric">
        <span>MSE maximo</span>
        <strong>${formatNumber(payload.max_mse_loss)}</strong>
      </div>
      <div class="metric">
        <span>MSE medio</span>
        <strong>${formatNumber(payload.mean_mse_loss)}</strong>
      </div>
      <div class="metric">
        <span>Umbral</span>
        <strong>${formatNumber(payload.threshold)}</strong>
      </div>
      <div class="metric">
        <span>Ratio sospechoso</span>
        <strong>${(payload.anomaly_window_ratio * 100).toFixed(2)}%</strong>
      </div>
    </div>
    <div class="windows"><strong>Ventanas sospechosas:</strong> ${windows}</div>
  `;
}

function parseCsv(csv) {
  const lines = csv.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) {
    throw new Error("El CSV debe contener cabecera y al menos una fila de datos.");
  }

  const headers = lines[0].split(",").map((header) => header.trim());
  const indexes = featureColumns.map((column) => headers.indexOf(column));
  const missing = featureColumns.filter((_, idx) => indexes[idx] === -1);
  if (missing.length) {
    throw new Error(`Faltan columnas en el CSV: ${missing.join(", ")}`);
  }

  return lines.slice(1).map((line, rowIndex) => {
    const values = line.split(",");
    const row = {};
    indexes.forEach((columnIndex, featureIndex) => {
      const value = Number(values[columnIndex]);
      if (!Number.isFinite(value)) {
        throw new Error(`Valor no numerico en fila ${rowIndex + 2}, columna ${featureColumns[featureIndex]}.`);
      }
      row[featureColumns[featureIndex]] = value;
    });
    return row;
  });
}

function buildRawWindows(rows, windowSize, step) {
  const windows = [];
  for (let start = 0; start + windowSize <= rows.length; start += step) {
    const window = {};
    featureColumns.forEach((column) => {
      window[column] = rows.slice(start, start + windowSize).map((row) => row[column]);
    });
    windows.push(window);
  }
  return windows;
}

function computeAverageWindow(windows, windowSize) {
  const average = {};
  featureColumns.forEach((column) => {
    average[column] = Array(windowSize).fill(0);
  });

  windows.forEach((window) => {
    featureColumns.forEach((column) => {
      window[column].forEach((value, index) => {
        average[column][index] += value;
      });
    });
  });

  featureColumns.forEach((column) => {
    average[column] = average[column].map((value) => value / windows.length);
  });

  return average;
}

function getCandidateWindows(payload) {
  const suspicious = payload.anomaly_windows && payload.anomaly_windows.length
    ? payload.anomaly_windows
    : [];

  if (suspicious.length) {
    return suspicious;
  }

  const losses = payload.window_mse_losses || [];
  if (!losses.length) {
    return [0];
  }

  return losses
    .map((loss, index) => ({ loss, index }))
    .sort((a, b) => b.loss - a.loss)
    .slice(0, Math.min(5, losses.length))
    .map((item) => item.index);
}

function renderChartControls(payload) {
  const candidates = getCandidateWindows(payload);
  windowSelect.innerHTML = candidates.map((index) => {
    const loss = payload.window_mse_losses ? payload.window_mse_losses[index] : null;
    const label = loss === null
      ? `Ventana ${index}`
      : `Ventana ${index} - MSE ${formatNumber(loss)}`;
    return `<option value="${index}">${label}</option>`;
  }).join("");

  chartPanel.className = "panel chart-panel visible";
  drawSelectedWindow();
}

function scalePoint(value, min, max, height, padding) {
  if (max === min) {
    return height / 2;
  }
  return height - padding - ((value - min) / (max - min)) * (height - padding * 2);
}

function drawLine(context, values, min, max, color, width, padding, plotWidth, plotHeight) {
  context.beginPath();
  values.forEach((value, index) => {
    const x = padding + (index / Math.max(values.length - 1, 1)) * plotWidth;
    const y = scalePoint(value, min, max, plotHeight, padding);
    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  });
  context.strokeStyle = color;
  context.lineWidth = width;
  context.stroke();
}

function drawSelectedWindow() {
  if (!currentPayload || !currentWindows.length || !currentAverageWindow) {
    return;
  }

  const selectedIndex = Number(windowSelect.value);
  const channel = channelSelect.value;
  const selectedWindow = currentWindows[selectedIndex];
  if (!selectedWindow) {
    return;
  }

  const selectedValues = selectedWindow[channel];
  const averageValues = currentAverageWindow[channel];
  const allValues = selectedValues.concat(averageValues);
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const context = chartCanvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = chartCanvas.clientWidth;
  const cssHeight = chartCanvas.clientHeight;

  chartCanvas.width = cssWidth * dpr;
  chartCanvas.height = cssHeight * dpr;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, cssWidth, cssHeight);

  const padding = 38;
  const plotWidth = cssWidth - padding * 2;
  const plotHeight = cssHeight;

  context.strokeStyle = "#e6ece8";
  context.lineWidth = 1;
  for (let i = 0; i < 5; i += 1) {
    const y = padding + (i / 4) * (cssHeight - padding * 2);
    context.beginPath();
    context.moveTo(padding, y);
    context.lineTo(cssWidth - padding, y);
    context.stroke();
  }

  drawLine(context, averageValues, min, max, "#245c3b", 2, padding, plotWidth, plotHeight);
  drawLine(context, selectedValues, min, max, "#c43d3d", 2.5, padding, plotWidth, plotHeight);

  const loss = currentPayload.window_mse_losses
    ? currentPayload.window_mse_losses[selectedIndex]
    : null;
  const startSample = selectedIndex * currentPayload.step;
  const endSample = startSample + currentPayload.window_size - 1;
  const difference = selectedValues.reduce((acc, value, index) => (
    acc + Math.abs(value - averageValues[index])
  ), 0) / selectedValues.length;

  chartNote.innerHTML = `
    <strong>${channel}</strong> · muestras ${startSample}-${endSample}
    · MSE ${loss === null ? "n/a" : formatNumber(loss)}
    · diferencia media absoluta frente al promedio ${formatNumber(difference)}.
  `;
}

sendButton.addEventListener("click", async () => {
  if (!fileInput.files.length) {
    renderError("Selecciona un archivo CSV antes de analizar.");
    return;
  }

  sendButton.disabled = true;
  renderPending("Procesando archivo y calculando error de reconstruccion...");

  try {
    const csv = await fileInput.files[0].text();
    currentRows = parseCsv(csv);
    const response = await fetch("/predict/csv", {
      method: "POST",
      headers: { "Content-Type": "text/csv" },
      body: csv
    });
    const payload = await response.json();

    if (!response.ok) {
      renderError(payload.detail || "Respuesta no valida del servidor.");
      return;
    }

    renderResult(payload);
    currentPayload = payload;
    currentWindows = buildRawWindows(currentRows, payload.window_size, payload.step);
    currentAverageWindow = computeAverageWindow(currentWindows, payload.window_size);
    renderChartControls(payload);
  } catch (error) {
    renderError(error.message || "Error inesperado durante el analisis.");
  } finally {
    sendButton.disabled = false;
  }
});

windowSelect.addEventListener("change", drawSelectedWindow);
channelSelect.addEventListener("change", drawSelectedWindow);
window.addEventListener("resize", drawSelectedWindow);
