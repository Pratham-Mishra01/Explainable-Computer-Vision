/* ── TAB SWITCHING ── */
function switchTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.querySelector(`[data-tab="${name}"]`).classList.add('active');
}

/* ── ANALYZE TAB ── */
function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('uploadZone').classList.add('dragover');
}
function handleDragLeave() {
  document.getElementById('uploadZone').classList.remove('dragover');
}
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('uploadZone').classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) runAnalysis(file);
}
function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) runAnalysis(file);
}

async function runAnalysis(file) {
  const loading = document.getElementById('loadingState');
  const results = document.getElementById('resultsArea');
  const errMsg  = document.getElementById('errorMsg');

  loading.classList.add('visible');
  results.classList.remove('visible');
  errMsg.classList.remove('visible');

  const formData = new FormData();
  formData.append('image', file);

  try {
    const res  = await fetch('/analyze', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderAnalysis(data);
  } catch (err) {
    errMsg.textContent = 'Error: ' + err.message;
    errMsg.classList.add('visible');
  } finally {
    loading.classList.remove('visible');
  }
}

function renderAnalysis(d) {
  document.getElementById('predClass').textContent = d.predicted_class;
  document.getElementById('confVal').textContent   = d.confidence.toFixed(3) + ' %';
  document.getElementById('imgOriginal').src = 'data:image/png;base64,' + d.original;
  document.getElementById('imgOverlay').src  = 'data:image/png;base64,' + d.overlay;

  const note = d.confidence > 99.9
    ? 'Near-saturated confidence. As documented in research, this value offers no predictive power over explanation quality on this dataset.'
    : 'Confidence slightly below saturation. Still within the near-constant range observed across the validation set.';
  document.getElementById('confNote').textContent = note;

  const fv = document.getElementById('focusVal');
  fv.textContent = d.focus_score.toFixed(4);
  fv.className   = 'focus-score-val ' + d.focus_label_class;

  const badge = document.getElementById('focusBadge');
  badge.textContent = d.focus_label;
  badge.className   = 'focus-badge ' + d.focus_label_class;

  const pct = Math.max(2, Math.min(98, d.focus_percentile));
  setTimeout(() => {
    document.getElementById('distMarker').style.left = pct + '%';
  }, 80);
  document.getElementById('distCaption').textContent =
    `${pct}th percentile of observed distribution (n=50, range 0.30–0.77)`;

  document.getElementById('resultsArea').classList.add('visible');
}

/* ── CONSISTENCY TAB ── */
function handleMultiDragOver(e) {
  e.preventDefault();
  document.getElementById('multiUploadZone').classList.add('dragover');
}
function handleMultiDragLeave() {
  document.getElementById('multiUploadZone').classList.remove('dragover');
}
function handleMultiDrop(e) {
  e.preventDefault();
  document.getElementById('multiUploadZone').classList.remove('dragover');
  const files = Array.from(e.dataTransfer.files).slice(0, 6);
  if (files.length >= 2) runConsistency(files);
  else showMultiError('Please drop at least 2 images.');
}
function handleMultiSelect(e) {
  const files = Array.from(e.target.files).slice(0, 6);
  if (files.length >= 2) runConsistency(files);
  else showMultiError('Please select at least 2 images.');
}
function showMultiError(msg) {
  const el = document.getElementById('multiErrorMsg');
  el.textContent = msg;
  el.classList.add('visible');
}

async function runConsistency(files) {
  const loading = document.getElementById('multiLoadingState');
  const results = document.getElementById('consistencyResults');
  const errMsg  = document.getElementById('multiErrorMsg');

  loading.classList.add('visible');
  results.classList.remove('visible');
  errMsg.classList.remove('visible');

  const formData = new FormData();
  files.forEach(f => formData.append('images', f));

  try {
    const res  = await fetch('/consistency', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderConsistency(data);
  } catch (err) {
    showMultiError('Error: ' + err.message);
  } finally {
    loading.classList.remove('visible');
  }
}

function renderConsistency(d) {
  document.getElementById('consistencyMeta').innerHTML = `
    <div class="meta-card">
      <div class="meta-card-label">Std. Deviation</div>
      <div class="meta-card-val">${d.std_dev.toFixed(4)}</div>
      <div class="meta-card-note">Focus Score across images</div>
    </div>
    <div class="meta-card">
      <div class="meta-card-label">Variance</div>
      <div class="meta-card-val">${d.variance.toFixed(4)}</div>
      <div class="meta-card-note">Lower = more stable</div>
    </div>
    <div class="meta-card">
      <div class="meta-card-label">Range</div>
      <div class="meta-card-val">${d.range.toFixed(4)}</div>
      <div class="meta-card-note">Max − Min focus score</div>
    </div>
  `;

  const grid = document.getElementById('consistencyGrid');
  grid.innerHTML = '';
  d.results.forEach((r, i) => {
    const item = document.createElement('div');
    item.className = 'consistency-item';
    item.innerHTML = `
      <div class="consistency-item-imgs">
        <img src="data:image/png;base64,${r.original}" alt="Original ${i + 1}">
        <img src="data:image/png;base64,${r.overlay}"  alt="Grad-CAM ${i + 1}">
      </div>
      <div class="consistency-item-meta">
        <div class="ci-class">${r.predicted_class}</div>
        <div class="ci-row">
          <span>Conf: ${r.confidence.toFixed(3)}%</span>
          <span>Focus: ${r.focus_score.toFixed(4)}</span>
        </div>
      </div>
    `;
    grid.appendChild(item);
  });

  document.getElementById('consistencyResults').classList.add('visible');
}
