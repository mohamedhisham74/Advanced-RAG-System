'use strict';

// ─── DOM ──────────────────────────────────────────────────────────────────────
const uploadZone    = document.getElementById('uploadZone');
const fileInput     = document.getElementById('fileInput');
const uploadName    = document.getElementById('uploadName');
const uploadBtn     = document.getElementById('uploadBtn');
const uploadBtnText = document.getElementById('uploadBtnText');
const uploadSpinner = document.getElementById('uploadSpinner');
const uploadMsg     = document.getElementById('uploadMsg');
const chunkCount    = document.getElementById('chunkCount');
const refreshBtn    = document.getElementById('refreshBtn');
const topK          = document.getElementById('topK');
const threshold     = document.getElementById('threshold');
const rerankTopN    = document.getElementById('rerankTopN');
const messages      = document.getElementById('messages');
const queryInput    = document.getElementById('queryInput');
const sendBtn       = document.getElementById('sendBtn');
const welcome       = document.getElementById('welcome');

let pendingFile = null;
let busy        = false;

// ─── Stats ────────────────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const res  = await fetch('/api/stats');
    const data = await res.json();
    chunkCount.textContent = data.indexed_chunks ?? '—';
  } catch {
    chunkCount.textContent = '?';
  }
}

refreshBtn.addEventListener('click', loadStats);
loadStats();

// ─── File pick / drag-drop ────────────────────────────────────────────────────
uploadZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => fileInput.files[0] && pickFile(fileInput.files[0]));

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.name.toLowerCase().endsWith('.pdf')) pickFile(f);
  else showAlert('Only PDF files are accepted.', 'error');
});

function pickFile(f) {
  pendingFile = f;
  uploadZone.classList.add('ready');
  uploadName.textContent = f.name;
  uploadName.classList.remove('hidden');
  uploadBtn.disabled = false;
  clearAlert();
}

// ─── Upload ───────────────────────────────────────────────────────────────────
uploadBtn.addEventListener('click', async () => {
  if (!pendingFile || busy) return;
  busy = true;
  uploadBtn.disabled    = true;
  uploadBtnText.textContent = 'Uploading…';
  uploadSpinner.classList.remove('hidden');
  clearAlert();

  const form = new FormData();
  form.append('file', pendingFile);

  try {
    const res  = await fetch('/api/ingest', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) {
      showAlert(`Error: ${data.detail || res.statusText}`, 'error');
    } else {
      showAlert(
        `✓ ${data.filename}\n${data.chunks_stored} chunks indexed · ${data.pages} pages`,
        'success',
      );
      loadStats();
    }
  } catch (err) {
    showAlert(`Network error: ${err.message}`, 'error');
  } finally {
    busy = false;
    uploadBtn.disabled = false;
    uploadBtnText.textContent = 'Upload & Index';
    uploadSpinner.classList.add('hidden');
    pendingFile = null;
    fileInput.value = '';
    uploadZone.classList.remove('ready');
    uploadName.classList.add('hidden');
  }
});

function showAlert(msg, type) {
  uploadMsg.textContent = msg;
  uploadMsg.className   = `alert ${type}`;
}
function clearAlert() {
  uploadMsg.textContent = '';
  uploadMsg.className   = 'alert hidden';
}

// ─── Chat ─────────────────────────────────────────────────────────────────────
queryInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
queryInput.addEventListener('input', () => {
  queryInput.style.height = 'auto';
  queryInput.style.height = queryInput.scrollHeight + 'px';
});
sendBtn.addEventListener('click', sendMessage);

function fillQuery(text) {
  queryInput.value = text;
  queryInput.dispatchEvent(new Event('input'));
  queryInput.focus();
}
window.fillQuery = fillQuery;

async function sendMessage() {
  const query = queryInput.value.trim();
  if (!query || busy) return;

  welcome?.remove();

  appendUser(query);
  queryInput.value = '';
  queryInput.style.height = 'auto';
  busy = true;
  sendBtn.disabled = true;

  const typingId = appendTyping();

  try {
    const res = await fetch('/api/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        top_k:        parseInt(topK.value),
        threshold:    parseFloat(threshold.value),
        rerank_top_n: parseInt(rerankTopN.value),
      }),
    });

    removeEl(typingId);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      appendError(err.detail || 'Request failed.');
    } else {
      appendAnswer(await res.json());
    }
  } catch (err) {
    removeEl(typingId);
    appendError(`Network error: ${err.message}`);
  } finally {
    busy = false;
    sendBtn.disabled = false;
    scrollEnd();
  }
}

// ─── Message builders ─────────────────────────────────────────────────────────
function appendUser(text) {
  const el = div('msg user');
  el.innerHTML = `<div class="avatar">👤</div><div class="bubble">${esc(text)}</div>`;
  messages.appendChild(el);
  scrollEnd();
}

let _tid = 0;
function appendTyping() {
  const id = `t${++_tid}`;
  const el = div('msg ai');
  el.id = id;
  el.innerHTML = `
    <div class="avatar">⚖️</div>
    <div class="bubble">
      <div class="typing"><div class="t-dot"></div><div class="t-dot"></div><div class="t-dot"></div></div>
    </div>`;
  messages.appendChild(el);
  scrollEnd();
  return id;
}

function appendAnswer(data) {
  const sid = `s${Date.now()}`;
  const hasSrc = data.sources && data.sources.length > 0;

  const el = div('msg ai');
  el.innerHTML = `
    <div class="avatar">⚖️</div>
    <div class="bubble">
      <div class="answer-text">${esc(data.answer)}</div>
      <div class="pipeline-tags">
        <span class="ptag">🔍 ${data.embeddings_used} embeddings</span>
        <span class="ptag">📄 ${data.chunks_retrieved} retrieved</span>
        <span class="ptag">🏆 ${data.chunks_after_rerank} reranked</span>
      </div>
      ${hasSrc ? `
        <button class="sources-btn" onclick="toggleSrc('${sid}',this)">
          <span class="s-arrow">▶</span> ${data.sources.length} source${data.sources.length > 1 ? 's' : ''}
        </button>
        <div class="sources-list" id="${sid}">
          ${data.sources.map(buildSource).join('')}
        </div>` : ''}
    </div>`;

  messages.appendChild(el);
  scrollEnd();
}

function buildSource(src) {
  const title = [src.article, src.law_number, src.document_name].filter(Boolean).join(' · ') || src.chunk_id;
  return `
    <div class="source-card">
      <div class="sc-header">
        <div class="sc-title">${esc(title)}</div>
        <div class="sc-badges">
          <span class="badge badge-sim">sim ${src.similarity.toFixed(2)}</span>
          <span class="badge badge-rank">rank ${src.rerank_score.toFixed(1)}</span>
        </div>
      </div>
      <div class="sc-excerpt">${esc(src.excerpt)}…</div>
    </div>`;
}

function appendError(msg) {
  const el = div('msg ai');
  el.innerHTML = `<div class="avatar">⚖️</div><div class="error-bubble">⚠️ ${esc(msg)}</div>`;
  messages.appendChild(el);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function toggleSrc(id, btn) {
  document.getElementById(id).classList.toggle('open');
  btn.querySelector('.s-arrow').classList.toggle('open');
}
window.toggleSrc = toggleSrc;

function div(cls) {
  const el = document.createElement('div');
  el.className = cls;
  return el;
}

function removeEl(id) {
  document.getElementById(id)?.remove();
}

function scrollEnd() {
  messages.scrollTop = messages.scrollHeight;
}

function esc(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
