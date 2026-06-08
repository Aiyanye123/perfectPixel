const $ = (id) => document.getElementById(id);
const state = {
  files: [], original: "", overlay: "", result: "", batchZip: "", activeView: "original",
  sourceSize: null,
  viewer: { scale: 1, x: 0, y: 0, dragging: false, startX: 0, startY: 0 }
};

function validImages(files) {
  return [...files].filter((file) => file.type.startsWith("image/"));
}

function setFiles(files) {
  const images = validImages(files);
  if (!images.length) return showError("没有找到可处理的图片。");
  if (images.length > 200) return showError("单次最多处理 200 张图片。");
  clearObjectUrls();
  state.files = images;
  state.original = URL.createObjectURL(images[0]);
  state.sourceSize = null;
  state.overlay = ""; state.result = ""; state.batchZip = "";
  loadSourceSize(state.original);

  $("file-thumb").src = state.original;
  $("file-name").textContent = images.length === 1 ? images[0].name : `已选择 ${images.length} 张图片`;
  const totalMb = images.reduce((sum, file) => sum + file.size, 0) / 1024 / 1024;
  $("file-meta").textContent = images.length === 1
    ? `${totalMb.toFixed(2)} MB`
    : `${images[0].webkitRelativePath?.split("/")[0] || "批量图片"} · 共 ${totalMb.toFixed(2)} MB`;
  $("file-row").classList.remove("hidden");
  $("process-button").disabled = false;
  $("process-button").querySelector("span").textContent = images.length === 1 ? "开始像素修复" : `批量修复 ${images.length} 张图片`;
  $("download-button").textContent = images.length === 1 ? "下载 PNG ↓" : "下载 ZIP ↓";
  $("download-button").disabled = true;
  $("batch-status").classList.add("hidden");
  resetResultViews();
  showView("original");
  resetDiagnostics();
  hideError();
}

function clearObjectUrls() {
  [state.original, state.batchZip].forEach((url) => { if (url) URL.revokeObjectURL(url); });
}

function clearFiles() {
  clearObjectUrls();
  state.files = []; state.original = ""; state.overlay = ""; state.result = ""; state.batchZip = ""; state.sourceSize = null;
  $("file-input").value = ""; $("folder-input").value = "";
  $("file-row").classList.add("hidden");
  $("process-button").disabled = true;
  $("process-button").querySelector("span").textContent = "开始像素修复";
  $("download-button").disabled = true;
  $("download-button").textContent = "下载 PNG ↓";
  $("viewport-content").classList.add("hidden");
  $("viewer-tools").classList.add("hidden");
  $("batch-status").classList.add("hidden");
  $("empty-state").classList.remove("hidden");
  resetResultViews();
  resetDiagnostics();
  updateManualGridHint();
}

function loadSourceSize(source) {
  const image = new Image();
  image.onload = () => {
    if (state.original !== source) return;
    state.sourceSize = { width: image.naturalWidth, height: image.naturalHeight };
    updateManualGridHint();
  };
  image.src = source;
}

function updateManualGridHint() {
  if ($("auto-grid").checked) {
    $("help-grid-status").textContent = "当前使用自动识别。";
    return false;
  }

  const gridWidth = Number($("grid-width").value);
  const gridHeight = Number($("grid-height").value);
  if (!Number.isInteger(gridWidth) || !Number.isInteger(gridHeight)
      || gridWidth < 2 || gridHeight < 2 || gridWidth > 4096 || gridHeight > 4096) {
    $("help-grid-status").textContent = "当前网格无效：宽高必须是 2 到 4096 之间的整数。";
    return true;
  }
  if (gridWidth * gridHeight > 1024 * 1024) {
    $("help-grid-status").textContent = "当前网格无效：输出超过 1,048,576 个像素。";
    return true;
  }
  if (!state.sourceSize || state.files.length > 1) {
    $("help-grid-status").textContent = `当前手动网格：${gridWidth} × ${gridHeight}。`;
    return false;
  }

  const { width, height } = state.sourceSize;
  if (gridWidth > width || gridHeight > height) {
    $("help-grid-status").textContent = `当前网格无效：不能超过原图 ${width} × ${height}。`;
    return true;
  }

  const cellWidth = width / gridWidth;
  const cellHeight = height / gridHeight;
  const cellRatio = Math.max(cellWidth / cellHeight, cellHeight / cellWidth);
  const outputRatio = Math.max(gridWidth / gridHeight, gridHeight / gridWidth);
  if (cellRatio > 1.5) {
    $("help-grid-status").textContent = `当前单格约 ${cellWidth.toFixed(2)} × ${cellHeight.toFixed(2)} px，比例偏离正方形。`;
  } else if (outputRatio >= 8) {
    $("help-grid-status").textContent = `当前将输出 ${gridWidth} × ${gridHeight} 长条图。`;
  } else {
    $("help-grid-status").textContent = `当前单格约 ${cellWidth.toFixed(2)} × ${cellHeight.toFixed(2)} px。`;
  }
  return false;
}

function resetResultViews() {
  document.querySelectorAll(".view-tabs button").forEach((button, index) => {
    if (index > 0) button.disabled = true;
  });
}

function showView(view) {
  state.activeView = view;
  document.querySelectorAll(".view-tabs button").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  const source = state[view];
  if (source) {
    $("preview-image").src = source;
    $("viewport-content").classList.remove("hidden");
    $("viewer-tools").classList.remove("hidden");
    $("empty-state").classList.add("hidden");
    $("batch-status").classList.add("hidden");
    resetViewer();
  }
}

function appendOptions(form) {
  form.append("sample_method", $("sample-method").value);
  form.append("backend", $("backend").value);
  form.append("auto_grid", $("auto-grid").checked);
  form.append("grid_width", $("grid-width").value);
  form.append("grid_height", $("grid-height").value);
  form.append("refine_intensity", $("refine").value);
  form.append("min_size", $("min-size").value);
  form.append("peak_width", $("peak-width").value);
  form.append("fix_square", $("fix-square").checked);
}

async function processImages() {
  if (!state.files.length) return;
  if (!$("auto-grid").checked && updateManualGridHint()) return showError("请先修正手动网格尺寸。");
  setProcessing(true);
  const isBatch = state.files.length > 1;
  try {
    if (isBatch) await processBatch();
    else await processSingle();
  } catch (error) {
    showError(error.message);
    showView("original");
  } finally {
    setProcessing(false);
  }
}

async function processSingle() {
  const form = new FormData();
  form.append("image", state.files[0]);
  appendOptions(form);
  const response = await fetch("/api/process", { method: "POST", body: form });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "处理失败。");
  state.overlay = data.overlay; state.result = data.result;
  document.querySelector('[data-view="overlay"]').disabled = false;
  document.querySelector('[data-view="result"]').disabled = false;
  $("download-button").disabled = false;
  updateDiagnostics(data.diagnostics);
  showView("result");
}

async function processBatch() {
  const form = new FormData();
  state.files.forEach((file) => form.append("images", file, file.webkitRelativePath || file.name));
  form.append("export_scale", $("export-scale").value);
  appendOptions(form);
  const response = await fetch("/api/process-batch", { method: "POST", body: form });
  if (!response.ok) {
    const data = await response.json();
    throw new Error(data.error || "批量处理失败。");
  }
  const zip = await response.blob();
  if (state.batchZip) URL.revokeObjectURL(state.batchZip);
  state.batchZip = URL.createObjectURL(zip);
  const total = Number(response.headers.get("X-Batch-Total"));
  const success = Number(response.headers.get("X-Batch-Success"));
  const failed = Number(response.headers.get("X-Batch-Failed"));
  const elapsed = Number(response.headers.get("X-Batch-Elapsed-Ms"));
  $("viewport-content").classList.add("hidden");
  $("viewer-tools").classList.add("hidden");
  $("empty-state").classList.add("hidden");
  $("batch-title").textContent = failed ? "批量处理已完成，部分失败" : "批量处理完成";
  $("batch-summary").textContent = `成功 ${success} 张 · 失败 ${failed} 张 · 共 ${total} 张`;
  $("batch-status").classList.remove("hidden");
  $("download-button").disabled = false;
  updateBatchDiagnostics(total, success, failed, elapsed);
}

function setProcessing(active) {
  hideError();
  $("loading").classList.toggle("hidden", !active);
  if (active) {
    $("viewport-content").classList.add("hidden");
    $("viewer-tools").classList.add("hidden");
  }
  $("empty-state").classList.add("hidden");
  $("batch-status").classList.add("hidden");
  $("process-button").disabled = active;
  $("loading").querySelector("strong").textContent = state.files.length > 1 ? `正在批量处理 ${state.files.length} 张图片` : "正在识别网格";
}

function updateDiagnostics(d) {
  $("metric-input").textContent = `${d.input_width} × ${d.input_height}`;
  $("metric-grid").textContent = `${d.detected_grid_width} × ${d.detected_grid_height}`;
  $("metric-cell").textContent = `单格约 ${d.cell_width} × ${d.cell_height} px`;
  $("metric-output").textContent = `${d.output_width} × ${d.output_height}`;
  $("metric-time").textContent = `${d.elapsed_ms} ms`;
  $("metric-backend").textContent = d.has_alpha ? `${d.backend} · 保留透明` : d.backend;
  if (d.grid_candidates?.length) {
    const confidence = Math.round(d.grid_confidence * 100);
    $("metric-confidence").textContent = `${confidence}%`;
    $("confidence-bar").style.transform = `scaleX(${confidence / 100})`;
    renderCandidates(d.grid_candidates);
  } else {
    $("metric-confidence").textContent = "手动";
    $("confidence-bar").style.transform = "scaleX(1)";
    $("candidate-list").innerHTML = '<div class="candidate-empty">手动模式</div>';
  }
}

function renderCandidates(candidates) {
  $("candidate-list").innerHTML = candidates.map((candidate, index) => {
    const alignment = Math.round(candidate.boundary_alignment * 100);
    const detail = Math.round((1 - candidate.detail_loss) * 100);
    return `<button class="candidate-item${index === 0 ? " selected" : ""}" type="button"
      data-grid-width="${candidate.grid_width}" data-grid-height="${candidate.grid_height}">
      <span class="candidate-rank">${index === 0 ? "已采用" : `#${index + 1}`}</span>
      <strong>${candidate.grid_width} × ${candidate.grid_height}</strong>
      <div><span>边界 ${alignment}%</span><span>细节 ${detail}%</span></div>
    </button>`;
  }).join("");
}

function useCandidateGrid(button) {
  const width = button.dataset.gridWidth;
  const height = button.dataset.gridHeight;
  if (!width || !height || !state.files.length) return;
  $("auto-grid").checked = false;
  $("manual-grid").classList.remove("hidden");
  $("grid-mode-label").textContent = "手动";
  $("grid-width").value = width;
  $("grid-height").value = height;
  processImages();
}

function updateBatchDiagnostics(total, success, failed, elapsed) {
  $("metric-input").textContent = `${total} 张`;
  $("metric-grid").textContent = "逐张检测";
  $("metric-cell").textContent = "每张图片独立识别";
  $("metric-output").textContent = `${success} 成功`;
  $("metric-time").textContent = `${elapsed} ms`;
  $("metric-backend").textContent = failed ? `${failed} 张失败` : "全部成功";
  $("metric-confidence").textContent = "批量";
  $("confidence-bar").style.transform = `scaleX(${success / Math.max(total, 1)})`;
  $("candidate-list").innerHTML = '<div class="candidate-empty">批量模式</div>';
}

function resetDiagnostics() {
  ["metric-input", "metric-grid", "metric-output", "metric-time", "metric-confidence"].forEach((id) => $(id).textContent = "—");
  $("metric-cell").textContent = "等待自动识别";
  $("metric-backend").textContent = "等待处理";
  $("confidence-bar").style.transform = "scaleX(0)";
  $("candidate-list").innerHTML = '<div class="candidate-empty">—</div>';
}

function toggleHelp(force) {
  const layer = $("help-layer");
  const shouldOpen = force === undefined ? layer.classList.contains("hidden") : force;
  if (shouldOpen) updateManualGridHint();
  layer.classList.toggle("hidden", !shouldOpen);
  document.body.style.overflow = shouldOpen ? "hidden" : "";
  if (shouldOpen) $("help-close").focus();
  else $("help-trigger").focus();
}

function downloadResult() {
  if (state.files.length > 1) {
    if (!state.batchZip) return;
    triggerDownload(state.batchZip, "perfect-pixel-batch.zip");
    return;
  }
  if (!state.result) return;
  const image = new Image();
  image.onload = () => {
    const scale = Number($("export-scale").value);
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth * scale; canvas.height = image.naturalHeight * scale;
    const context = canvas.getContext("2d");
    context.imageSmoothingEnabled = false;
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    triggerDownload(canvas.toDataURL("image/png"), `perfect-pixel-${image.naturalWidth}x${image.naturalHeight}-${scale}x.png`);
  };
  image.src = state.result;
}

function triggerDownload(url, filename) {
  const link = document.createElement("a"); link.download = filename; link.href = url; link.click();
}

function resetViewer() {
  state.viewer.scale = 1;
  state.viewer.x = 0;
  state.viewer.y = 0;
  updateViewer();
}

function setViewerScale(nextScale, clientX, clientY) {
  const canvas = $("canvas-wrap");
  const rect = canvas.getBoundingClientRect();
  const oldScale = state.viewer.scale;
  const scale = Math.min(24, Math.max(0.25, nextScale));
  const anchorX = clientX === undefined ? 0 : clientX - rect.left - rect.width / 2;
  const anchorY = clientY === undefined ? 0 : clientY - rect.top - rect.height / 2;
  state.viewer.x = anchorX - (anchorX - state.viewer.x) * (scale / oldScale);
  state.viewer.y = anchorY - (anchorY - state.viewer.y) * (scale / oldScale);
  state.viewer.scale = scale;
  updateViewer();
}

function updateViewer() {
  const { scale, x, y } = state.viewer;
  $("viewport-content").style.transform = `translate3d(${x}px, ${y}px, 0) scale(${scale})`;
  $("zoom-value").textContent = `${Math.round(scale * 100)}%`;
}

function startViewerDrag(event) {
  if ($("viewport-content").classList.contains("hidden") || event.button !== 0 || event.target.closest(".viewer-tools")) return;
  event.preventDefault();
  state.viewer.dragging = true;
  state.viewer.startX = event.clientX - state.viewer.x;
  state.viewer.startY = event.clientY - state.viewer.y;
  $("canvas-wrap").classList.add("dragging-view");
  $("canvas-wrap").setPointerCapture(event.pointerId);
}

function moveViewer(event) {
  if (!state.viewer.dragging) return;
  state.viewer.x = event.clientX - state.viewer.startX;
  state.viewer.y = event.clientY - state.viewer.startY;
  updateViewer();
}

function stopViewerDrag(event) {
  if (!state.viewer.dragging) return;
  state.viewer.dragging = false;
  $("canvas-wrap").classList.remove("dragging-view");
  if ($("canvas-wrap").hasPointerCapture(event.pointerId)) $("canvas-wrap").releasePointerCapture(event.pointerId);
}

function showError(message) { $("error-message").textContent = message; $("error-message").classList.remove("hidden"); }
function hideError() { $("error-message").classList.add("hidden"); }

$("choose-files").addEventListener("click", () => $("file-input").click());
$("choose-folder").addEventListener("click", () => $("folder-input").click());
$("file-input").addEventListener("change", (event) => setFiles(event.target.files));
$("folder-input").addEventListener("change", (event) => setFiles(event.target.files));
$("remove-file").addEventListener("click", clearFiles);
$("process-button").addEventListener("click", processImages);
$("download-button").addEventListener("click", downloadResult);
$("auto-grid").addEventListener("change", (event) => {
  $("manual-grid").classList.toggle("hidden", event.target.checked);
  $("grid-mode-label").textContent = event.target.checked ? "自动" : "手动";
  updateManualGridHint();
});
$("grid-width").addEventListener("input", updateManualGridHint);
$("grid-height").addEventListener("input", updateManualGridHint);
$("refine").addEventListener("input", (event) => $("refine-value").textContent = Number(event.target.value).toFixed(2));
$("export-scale").addEventListener("input", (event) => $("scale-value").textContent = `${event.target.value}×`);
document.querySelectorAll(".view-tabs button").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
$("zoom-out").addEventListener("click", () => setViewerScale(state.viewer.scale / 1.25));
$("zoom-in").addEventListener("click", () => setViewerScale(state.viewer.scale * 1.25));
$("reset-view").addEventListener("click", resetViewer);
$("canvas-wrap").addEventListener("wheel", (event) => {
  if ($("viewport-content").classList.contains("hidden")) return;
  event.preventDefault();
  setViewerScale(state.viewer.scale * (event.deltaY < 0 ? 1.12 : 1 / 1.12), event.clientX, event.clientY);
}, { passive: false });
$("canvas-wrap").addEventListener("pointerdown", startViewerDrag);
$("canvas-wrap").addEventListener("pointermove", moveViewer);
$("canvas-wrap").addEventListener("pointerup", stopViewerDrag);
$("canvas-wrap").addEventListener("pointercancel", stopViewerDrag);
$("canvas-wrap").addEventListener("dblclick", resetViewer);
$("candidate-list").addEventListener("click", (event) => {
  const candidate = event.target.closest(".candidate-item");
  if (candidate) useCandidateGrid(candidate);
});
$("help-trigger").addEventListener("click", () => toggleHelp(true));
$("help-close").addEventListener("click", () => toggleHelp(false));
$("help-backdrop").addEventListener("click", () => toggleHelp(false));
document.addEventListener("keydown", (event) => {
  const target = event.target;
  const editing = target instanceof HTMLTextAreaElement || target.isContentEditable;
  if (!editing && event.key.toLowerCase() === "p") {
    event.preventDefault();
    toggleHelp();
  } else if (event.key === "Escape" && !$("help-layer").classList.contains("hidden")) {
    toggleHelp(false);
  }
});

const dropzone = $("dropzone");
["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); }));
dropzone.addEventListener("drop", (event) => setFiles(event.dataTransfer.files));
dropzone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") $("file-input").click(); });
