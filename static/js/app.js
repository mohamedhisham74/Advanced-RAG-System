'use strict';

const API = '';  // same origin

// ── DOM refs ──────────────────────────────────────────────────────────────────
const messagesEl   = document.getElementById('messages');
const queryInput   = document.getElementById('queryInput');
const sendBtn      = document.getElementById('sendBtn');
const uploadArea   = document.getElementById('uploadArea');
const fileInput    = document.getElementById('fileInput');
const uploadBtn    = document.getElementById('uploadBtn');
const uploadStatus = document.getElementById('uploadStatus');
const uploadSpinner = document.getElementById('uploadSpinner');
const chunkCount   = document.getElementById('chunkCount');
const refreshStats = document.getElementById('refreshStats');
const topK         = document.getElementById('topK');
const threshold    = document.getElementById('threshold');
const rerankTopN   = document.getElementById('rerankTopN');

let selectedFile = null;
let isLoading    = false;

// ── Stats ─────────────────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const res  = await fetch(`${API}/api/stats`);
    const data = await res.json();
    chunkCount.textContent = data.indexed_chunks ?? '—';
  } catch {
    chunkCount.textContent = 'ERR';
  }
}

refreshStats.addEventListener('click', loadStats);
loadStats();

// ── File upload ───────────────────────────────────────────────────────────────
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadArea.classList.add('drag-over');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.name.endsWith('.pdf')) setFile(f);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(f) {
  selectedFile = f;
  uploadArea.classList.add('has-file');
  uploadArea.querySelector('.upload-text').textContent = f.name;
  uploadBtn.disabled = false;
}

uploadBtn.addEventListener('click', async () => {
  if (!selectedFile || isLoading) return;

  uploadBtn.disabled = true;
  uploadBtn.querySelector('.btn-text').textContent = 'Uploading…';
  uploadSpinner.classList.remove('hidden');
  showStatus('', '');

  const form = new FormData();
  form.append('file', selectedFile);

  try {
    const res  = await fetch(`${API}/api/ingest`, { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) {
      showStatus(`Error: ${data.detail || res.statusText}`, 'error');
    } else {
      showStatus(
        `✓ ${data.filename}\n${data.chunks_stored} chunks indexed (${data.pages} pages)`,
        'success'
      );
      loadStats();
    }
  } catch (err) {
    showStatus(`Network error: ${err.message}`, 'error');
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.querySelector('.btn-text').textContent = 'Upload & Index';
    uploadSpinner.classList.add('hidden');
    selectedFile = null;
    fileInput.value = '';
    uploadArea.classList.remove('has-file');
    uploadArea.querySelector('.upload-text').textContent = 'Click or drag a PDF here';
  }
});

function showStatus(msg, type) {
  if (!msg) { uploadStatus.classList.add('hidden'); return; }
  uploadStatus.textContent = msg;
  uploadStatus.className   = `status-msg ${type}`;
}

// ── Chat ──────────────────────────────────────────────────────────────────────
queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);

queryInput.addEventListener('input', () => {
  queryInput.style.height = 'auto';
  queryInput.style.height = queryInput.scrollHeight + 'px';
});

function setQuery(text) {
  queryInput.value = text;
  queryInput.dispatchEvent(new Event('input'));
  queryInput.focus();
}
window.setQuery = setQuery;

async function sendMessage() {
  const query = queryInput.value.trim();
  if (!query || isLoading) return;

  // Remove welcome screen
  const welcome = messagesEl.querySelector('.welcome-msg');
  if (welcome) welcome.remove();

  appendUserMessage(query);
  queryInput.value = '';
  queryInput.style.height = 'auto';
  sendBtn.disabled = true;
  isLoading        = true;

  const typingId = appendTyping();

  try {
    const res = await fetch(`${API}/api/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        top_k:        parseInt(topK.value),
        threshold:    parseFloat(threshold.value),
        rerank_top_n: parseInt(rerankTopN.value),
      }),
    });

    removeTyping(typingId);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      appendErrorMessage(err.detail || 'Request failed.');
    } else {
      const data = await res.json();
      appendAIMessage(data);
    }
  } catch (err) {
    removeTyping(typingId);
    appendErrorMessage(`Network error: ${err.message}`);
  } finally {
    sendBtn.disabled = false;
    isLoading        = false;
    scrollToBottom();
  }
}

// ── Message builders ──────────────────────────────────────────────────────────
function appendUserMessage(text) {
  const el = document.createElement('div');
  el.className = 'message user';
  el.innerHTML = `
    <div class="avatar">👤</div>
    <div class="bubble">${escapeHtml(text)}</div>
  `;
  messagesEl.appendChild(el);
  scrollToBottom();
}

let typingCounter = 0;

function appendTyping() {
  const id = `typing-${++typingCounter}`;
  const el = document.createElement('div');
  el.className = 'message ai';
  el.id = id;
  el.innerHTML = `
    <div class="avatar">⚖️</div>
    <div class="bubble">
      <div class="typing">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
  `;
  messagesEl.appendChild(el);
  scrollToBottom();
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendAIMessage(data) {
  const el = document.createElement('div');
  el.className = 'message ai';

  const sourcesId = `src-${Date.now()}`;
  const hasSources = data.sources && data.sources.length > 0;

  el.innerHTML = `
    <div class="avatar">⚖️</div>
    <div class="bubble">
      <div class="answer-text">${escapeHtml(data.answer)}</div>
      <div class="pipeline-meta">
        <span class="meta-tag">🔍 ${data.embeddings_used} embeddings</span>
        <span class="meta-tag">📄 ${data.chunks_retrieved} retrieved</span>
        <span class="meta-tag">🏆 ${data.chunks_after_rerank} reranked</span>
      </div>
      ${hasSources ? `
        <div class="sources-toggle" onclick="toggleSources('${sourcesId}', this)">
          <span class="toggle-arrow">▶</span>
          ${data.sources.length} source${data.sources.length > 1 ? 's' : ''}
        </div>
        <div class="sources-list" id="${sourcesId}">
          ${data.sources.map(buildSourceCard).join('')}
        </div>
      ` : ''}
    </div>
  `;

  messagesEl.appendChild(el);
  scrollToBottom();
}

function buildSourceCard(src) {
  const title = [src.article, src.law_number, src.document_name]
    .filter(Boolean).join(' · ') || src.chunk_id;

  return `
    <div class="source-card">
      <div class="source-header">
        <div class="source-title">${escapeHtml(title)}</div>
        <div class="source-badges">
          <span class="badge badge-sim">sim ${src.similarity.toFixed(2)}</span>
          <span class="badge badge-rerank">rank ${src.rerank_score.toFixed(1)}</span>
        </div>
      </div>
      <div class="source-excerpt">${escapeHtml(src.excerpt)}…</div>
    </div>
  `;
}

function appendErrorMessage(msg) {
  const el = document.createElement('div');
  el.className = 'message ai';
  el.innerHTML = `
    <div class="avatar">⚖️</div>
    <div class="error-bubble">⚠️ ${escapeHtml(msg)}</div>
  `;
  messagesEl.appendChild(el);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function toggleSources(id, btn) {
  const list  = document.getElementById(id);
  const arrow = btn.querySelector('.toggle-arrow');
  list.classList.toggle('visible');
  arrow.classList.toggle('open');
}
window.toggleSources = toggleSources;

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
