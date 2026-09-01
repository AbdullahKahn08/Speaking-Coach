// English Speaking Coach — frontend

requestAnimationFrame(() => {
  document.querySelectorAll('[data-reveal]').forEach((el, i) => {
    setTimeout(() => el.classList.add('on'), 60 * i);
  });
});

// ─── Tabs ───
const tabs = document.querySelectorAll('.tab');
tabs.forEach(t => {
  t.addEventListener('click', () => {
    tabs.forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    const tab = t.dataset.tab;
    document.getElementById('view-record').style.display  = tab === 'record'  ? '' : 'none';
    document.getElementById('view-history').style.display = tab === 'history' ? '' : 'none';
    if (tab === 'history') loadStudents();
  });
});

// ─── Upload ───
const uploader   = document.getElementById('uploader');
const audioInput = document.getElementById('audioInput');
const audioPrev  = document.getElementById('audioPreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const selectedEl = document.getElementById('selectedFile');
let currentFile = null;

function handleFile(file) {
  currentFile = file;
  selectedEl.textContent = `Selected: ${file.name}`;
  audioPrev.src = URL.createObjectURL(file);
  document.getElementById('audioPreviewWrap').style.display = 'flex';
  analyzeBtn.disabled = false;
}

audioInput.addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

['dragover', 'dragenter'].forEach(ev =>
  uploader.addEventListener(ev, e => { e.preventDefault(); uploader.classList.add('dragging'); }));
['dragleave', 'drop'].forEach(ev =>
  uploader.addEventListener(ev, e => { e.preventDefault(); uploader.classList.remove('dragging'); }));
uploader.addEventListener('drop', e => {
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

// ─── Source toggle (upload vs record) ───
const sourceToggle = document.getElementById('sourceToggle');
const recorderPanel = document.getElementById('recorder');
sourceToggle.querySelectorAll('button').forEach(btn => {
  btn.addEventListener('click', () => {
    sourceToggle.querySelectorAll('button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const src = btn.dataset.source;
    uploader.style.display        = src === 'upload' ? 'block' : 'none';
    recorderPanel.style.display   = src === 'record' ? 'block' : 'none';
    // reset state
    currentFile = null;
    selectedEl.textContent = '';
    analyzeBtn.disabled = true;
    document.getElementById('audioPreviewWrap').style.display = 'none';
    stopRecording(false);
  });
});

// ─── Recorder ───
const recBtn = document.getElementById('recBtn');
const recStatus = document.getElementById('recStatus');
const recTimer = document.getElementById('recTimer');
const waveCanvas = document.getElementById('waveform');
const waveCtx = waveCanvas.getContext('2d');
let mediaRecorder = null;
let recordedChunks = [];
let recStart = 0;
let recInterval = null;
let audioCtx = null;
let analyser = null;
let waveRAF = null;

function fmtTime(s) {
  const m = Math.floor(s / 60), sec = s % 60;
  return String(m).padStart(2, '0') + ':' + String(sec).padStart(2, '0');
}

// Fit canvas to actual pixel size for a crisp waveform on hi-dpi screens
function sizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const w = waveCanvas.clientWidth, h = waveCanvas.clientHeight;
  waveCanvas.width = w * dpr;
  waveCanvas.height = h * dpr;
  waveCtx.scale(dpr, dpr);
}

function drawIdleWaveform() {
  const w = waveCanvas.clientWidth, h = waveCanvas.clientHeight;
  waveCtx.clearRect(0, 0, w, h);
  const barCount = 60, gap = 3;
  const barW = (w - gap * (barCount - 1)) / barCount;
  waveCtx.fillStyle = '#d8d3cc';
  for (let i = 0; i < barCount; i++) {
    const barH = 2;
    waveCtx.fillRect(i * (barW + gap), (h - barH) / 2, barW, barH);
  }
}

function drawLiveWaveform() {
  if (!analyser) return;
  const w = waveCanvas.clientWidth, h = waveCanvas.clientHeight;
  const bufferLen = analyser.frequencyBinCount;
  const data = new Uint8Array(bufferLen);
  analyser.getByteFrequencyData(data);

  waveCtx.clearRect(0, 0, w, h);
  const barCount = 60, gap = 3;
  const barW = (w - gap * (barCount - 1)) / barCount;
  // Sample the frequency data down to our bar count, weighting toward vocal range
  const step = Math.floor(bufferLen / barCount / 2);   // use lower half of spectrum (voice sits here)
  for (let i = 0; i < barCount; i++) {
    const v = data[i * step] || 0;
    const barH = Math.max(2, (v / 255) * h * 0.95);
    // gradient from center outwards for a nice mirrored feel
    waveCtx.fillStyle = '#CF4647';
    waveCtx.fillRect(i * (barW + gap), (h - barH) / 2, barW, barH);
  }
  waveRAF = requestAnimationFrame(drawLiveWaveform);
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // Audio graph for live visualization
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    sizeCanvas();
    drawLiveWaveform();

    const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm';
    mediaRecorder = new MediaRecorder(stream, { mimeType: mime });
    recordedChunks = [];
    mediaRecorder.ondataavailable = e => { if (e.data.size) recordedChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      if (audioCtx) { audioCtx.close(); audioCtx = null; analyser = null; }
      cancelAnimationFrame(waveRAF);
      drawIdleWaveform();
      const blob = new Blob(recordedChunks, { type: mime });
      const file = new File([blob], `recording_${Date.now()}.webm`, { type: mime });
      handleFile(file);
      recStatus.textContent = 'Recording ready. Tap again to re-record.';
    };
    mediaRecorder.start();
    recBtn.classList.add('recording');
    recBtn.textContent = '■';
    recStatus.textContent = 'Recording…';
    recStart = Date.now();
    recInterval = setInterval(() => {
      recTimer.textContent = fmtTime(Math.floor((Date.now() - recStart) / 1000));
    }, 200);
  } catch (e) {
    recStatus.textContent = 'Mic permission denied or unavailable.';
    console.error(e);
  }
}

function stopRecording(reset = true) {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
  }
  clearInterval(recInterval);
  cancelAnimationFrame(waveRAF);
  recBtn.classList.remove('recording');
  recBtn.textContent = '●';
  if (reset) {
    recTimer.textContent = '00:00';
    recStatus.textContent = 'Tap to start recording';
    drawIdleWaveform();
  }
}

recBtn.addEventListener('click', () => {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    stopRecording(false);
  } else {
    startRecording();
  }
});

// Draw idle bars whenever the recorder panel becomes visible
new MutationObserver(() => {
  if (recorderPanel.style.display !== 'none') {
    sizeCanvas();
    drawIdleWaveform();
  }
}).observe(recorderPanel, { attributes: true, attributeFilter: ['style'] });
window.addEventListener('resize', () => {
  if (recorderPanel.style.display !== 'none') { sizeCanvas(); drawIdleWaveform(); }
});

// ─── Analyze ───
analyzeBtn.addEventListener('click', async () => {
  const name  = document.getElementById('studentName').value.trim();
  const grade = document.getElementById('studentGrade').value.trim();
  if (!name)        { alert('Please enter a student name.'); return; }
  if (!currentFile) return;

  const results = document.getElementById('results');
  results.innerHTML = `
    <div class="loader-wrap">
      <div class="loader"></div>
      <div class="loader-caption">Listening carefully</div>
    </div>`;
  analyzeBtn.disabled = true;

  const form = new FormData();
  form.append('student_name', name);
  form.append('grade', grade);
  form.append('audio', currentFile);

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: form });
    if (!res.ok) throw new Error(await res.text());
    renderResults(await res.json());
  } catch (e) {
    results.innerHTML = `<div class="empty">Analysis failed: ${e.message}</div>`;
  } finally {
    analyzeBtn.disabled = false;
  }
});

// ─── Render results ───
function renderResults(d) {
  const results = document.getElementById('results');

  const mistakes = d.grammar_mistakes.length
    ? d.grammar_mistakes.map(m => `
        <div class="mistake-card" data-reveal>
          <span class="rule-tag">${escape(m.rule_id)}</span>
          <div class="mistake-body">
            <div class="diff">
              <span class="wrong">${escape(m.wrong)}</span>
              <span class="arrow">→</span>
              <span class="fix">${escape(m.correction)}</span>
            </div>
            <div class="msg">${escape(m.message)}</div>
          </div>
        </div>`).join('')
    : `<div class="empty">No grammar issues detected. Nicely spoken.</div>`;

  const fillerBlock = d.fillers.length ? `
    <div class="section-head" data-reveal>
      <span class="num">04</span>
      <h2>Filler Words</h2>
    </div>
    <div class="fillers">
      ${d.fillers.map(f => `<span class="filler-pill">${escape(f)}</span>`).join('')}
    </div>` : '';

  results.innerHTML = `
    <div style="margin-top: 48px;">
      <div class="success" data-reveal>
        <span class="dot"></span>
        Session #${d.session_id} saved
      </div>
    </div>

    <div class="section-head" data-reveal>
      <span class="num">01</span>
      <h2>Transcript</h2>
    </div>
    <div class="transcript-card" data-reveal>
      <div class="quote-mark">"</div>
      <div class="transcript-text">${escape(d.transcript || '(no speech detected)')}</div>
    </div>
    ${d.corrected && d.corrected !== d.transcript ? `
      <div class="corrected-card" data-reveal>
        <div class="label">Corrected Version</div>
        <div class="corrected-text">${highlightCorrections(d.corrected, d.grammar_mistakes)}</div>
      </div>` : ''}

    <div class="section-head" data-reveal>
      <span class="num">02</span>
      <h2>Fluency Metrics</h2>
    </div>
    <div class="metrics">
      ${metric('Words / min', Math.round(d.wpm), '140–160 = fluent')}
      ${metric('Duration', d.duration.toFixed(1) + 's', '')}
      ${metric('Filler words', d.fillers.length, 'total')}
      ${metric('Long pauses', d.long_pauses, 'over 1.5s')}
      ${metric('Short pauses', d.short_pauses, '0.25–0.75s')}
      ${metric('Medium pauses', d.medium_pauses, '0.75–1.5s')}
      ${metric('Avg pause', Math.round(d.avg_pause_ms) + 'ms', '')}
      ${metric('Word count', d.word_count, 'total')}
    </div>

    <div class="section-head" data-reveal>
      <span class="num">03</span>
      <h2>Grammar &amp; Structure</h2>
    </div>
    <div class="mistakes-list">${mistakes}</div>

    ${fillerBlock}
  `;

  requestAnimationFrame(() => {
    results.querySelectorAll('[data-reveal]').forEach((el, i) => {
      setTimeout(() => el.classList.add('on'), 70 * i);
    });
  });
}

function metric(label, value, unit) {
  return `
    <div class="metric" data-reveal>
      <div class="label">${label}</div>
      <div class="value">${value}</div>
      ${unit ? `<div class="unit">${unit}</div>` : ''}
    </div>`;
}

// ─── History ───
async function loadStudents() {
  const picker = document.getElementById('studentPicker');
  const detail = document.getElementById('studentDetail');
  detail.innerHTML = '';
  picker.innerHTML = `<div class="loader-wrap"><div class="loader"></div></div>`;

  try {
    const students = await (await fetch('/api/students')).json();
    if (!students.length) {
      picker.innerHTML = '';
      detail.innerHTML = '<div class="empty">No students yet. Record something first.</div>';
      return;
    }
    picker.innerHTML = students.map(s =>
      `<button class="student-chip" data-name="${escape(s.name)}">${escape(s.name)} · ${s.session_count}</button>`
    ).join('');
    picker.querySelectorAll('.student-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        picker.querySelectorAll('.student-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        loadStudentDetail(chip.dataset.name);
      });
    });
    picker.querySelector('.student-chip').click();
  } catch (e) {
    picker.innerHTML = `<div class="empty">Failed to load: ${e.message}</div>`;
  }
}

async function loadStudentDetail(name) {
  const detail = document.getElementById('studentDetail');
  detail.innerHTML = `<div class="loader-wrap"><div class="loader"></div></div>`;
  try {
    const d = await (await fetch(`/api/students/${encodeURIComponent(name)}`)).json();

    const sessionRows = d.sessions.map(s => `
      <tr>
        <td>${escape(s.date)}</td>
        <td>${s.wpm}</td>
        <td>${s.duration}s</td>
        <td>${s.fillers}</td>
        <td>${s.long_pauses}</td>
        <td>${s.grammar_mistakes}</td>
      </tr>`).join('');

    const max = d.top_rules[0]?.count || 1;
    const bars = d.top_rules.length
      ? d.top_rules.map(r => `
          <div class="bar-row">
            <span class="rule-tag">${escape(r.rule)}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${(r.count/max*100).toFixed(0)}%"></div></div>
            <div class="bar-count">${r.count}</div>
          </div>`).join('')
      : '<div class="empty" style="margin:0">No grammar mistakes logged yet.</div>';

    detail.innerHTML = `
      <div class="history-header">
        <h1 class="name">${escape(d.name)}</h1>
        <div class="meta">${escape(d.grade || 'No grade')} · ${d.sessions.length} session(s)</div>
      </div>

      <div class="section-head">
        <span class="num">02</span>
        <h2>Session History</h2>
      </div>
      <table class="session-table">
        <thead><tr><th>Date</th><th>WPM</th><th>Duration</th><th>Fillers</th><th>Long pauses</th><th>Grammar</th></tr></thead>
        <tbody>${sessionRows}</tbody>
      </table>

      <div class="section-head">
        <span class="num">03</span>
        <h2>Recurring Mistakes</h2>
      </div>
      <div class="bar-list">${bars}</div>
    `;
  } catch (e) {
    detail.innerHTML = `<div class="empty">Failed to load: ${e.message}</div>`;
  }
}

function escape(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

// Wrap each corrected phrase in <mark> so the fixes stand out
function highlightCorrections(correctedText, mistakes) {
  let out = escape(correctedText);
  // sort corrections by length desc so longer phrases match before shorter ones
  const fixes = [...new Set(mistakes.map(m => m.correction).filter(Boolean))]
    .sort((a, b) => b.length - a.length);
  for (const fix of fixes) {
    const safe = escape(fix);
    // escape regex-special chars in the fix
    const pattern = safe.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    // \b works for word boundaries; won't match inside existing <mark> because HTML
    const re = new RegExp(`\\b(${pattern})\\b`, 'g');
    out = out.replace(re, '<mark>$1</mark>');
  }
  return out;
}