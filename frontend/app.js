const state = {
  currentResult: null,
  currentAbortController: null,
  currentView: 'home',
  debugVisible: false,
};

const elements = {
  queryForm: document.getElementById('query-form'),
  queryInput: document.getElementById('query-input'),
  maxIterations: document.getElementById('max-iterations'),
  maxIterationsValue: document.getElementById('max-iterations-value'),
  modelInput: document.getElementById('model-input'),
  depthMode: document.getElementById('depth-mode'),
  queryBtn: document.getElementById('query-btn'),
  streamBtn: document.getElementById('stream-btn'),
  clearBtn: document.getElementById('clear-btn'),
  loadExampleBtn: document.getElementById('load-example-btn'),
  refreshHealthBtn: document.getElementById('refresh-health-btn'),
  currentModel: document.getElementById('current-model'),
  currentLimitLabel: document.getElementById('current-limit-label'),
  healthDot: document.getElementById('health-dot'),
  healthStatus: document.getElementById('health-status'),
  groqStatus: document.getElementById('groq-status'),
  tavilyStatus: document.getElementById('tavily-status'),
  toolsCount: document.getElementById('tools-count'),
  healthJson: document.getElementById('health-json'),
  answerCard: document.getElementById('answer-card'),
  validationCard: document.getElementById('validation-card'),
  sourcesList: document.getElementById('sources-list'),
  traceList: document.getElementById('trace-list'),
  streamOutput: document.getElementById('stream-output'),
  rawJson: document.getElementById('raw-json'),
  resultMeta: document.getElementById('result-meta'),
  exampleButtons: document.querySelectorAll('.example-chip'),
  navHomeBtn: document.getElementById('nav-home-btn'),
  navAboutBtn: document.getElementById('nav-about-btn'),
  viewHome: document.getElementById('view-home'),
  viewAbout: document.getElementById('view-about'),
  toggleDebugBtn: document.getElementById('toggle-debug-btn'),
  debugSection: document.getElementById('debug-section'),
};

const SAMPLE_QUERY = 'What is RAG in AI?';

function setLoading(isLoading) {
  elements.queryBtn.disabled = isLoading;
  elements.streamBtn.disabled = isLoading;
  elements.loadExampleBtn.disabled = isLoading;
  elements.refreshHealthBtn.disabled = isLoading;
  document.body.classList.toggle('is-loading', isLoading);
}

function setStatusTone(tone, message) {
  elements.healthDot.className = `status-dot ${tone}`;
  elements.healthStatus.textContent = message;
}

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function truncateText(text, maxLength = 220) {
  const normalized = String(text || '').replace(/\s+/g, ' ').trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, maxLength - 3).trimEnd()}...`;
}

function formatInlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}

function isSectionLine(line) {
  return /^(\d+)\.\s+(?:\*\*(.+?)\*\*|([^:]+?)):\s*(.*)$/.exec(line);
}

function isHeadingLine(line) {
  return /^(#{1,3})\s+(.+)$/.exec(line);
}

function isBulletLine(line) {
  return /^[-*]\s+(.+)$/.exec(line);
}

function isOrderedListLine(line) {
  return /^(\d+)\.\s+(?!\*\*)(.+)$/.exec(line);
}

function isTableLine(line) {
  return line.trim().startsWith('|') && line.includes('|');
}

function isTableSeparatorLine(line) {
  const cells = splitTableCells(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function splitTableCells(line) {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim());
}

function renderMarkdownTable(rows) {
  if (rows.length === 0) {
    return '';
  }

  const headerCells = splitTableCells(rows[0]);
  const bodyRows = [];

  for (let index = 1; index < rows.length; index += 1) {
    if (isTableSeparatorLine(rows[index])) {
      continue;
    }
    bodyRows.push(splitTableCells(rows[index]));
  }

  const thead = `<thead><tr>${headerCells.map((cell) => `<th>${formatInlineMarkdown(cell)}</th>`).join('')}</tr></thead>`;
  const tbody = bodyRows.length
    ? `<tbody>${bodyRows
        .map((row) => `<tr>${row.map((cell) => `<td>${formatInlineMarkdown(cell)}</td>`).join('')}</tr>`)
        .join('')}</tbody>`
    : '';

  return `<table class="answer-table">${thead}${tbody}</table>`;
}

function renderMarkdownContent(text) {
  const lines = String(text || '').replaceAll('\r\n', '\n').split('\n');
  const blocks = [];
  let index = 0;

  while (index < lines.length) {
    const rawLine = lines[index];
    const line = rawLine.trim();

    if (!line) {
      index += 1;
      continue;
    }

    if (line.startsWith('```')) {
      const codeLines = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith('```')) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) {
        index += 1;
      }
      blocks.push(`<pre class="answer-code"><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
      continue;
    }

    const sectionMatch = isSectionLine(line);
    if (sectionMatch) {
      const sectionNumber = sectionMatch[1];
      const sectionTitle = sectionMatch[2] || sectionMatch[3] || '';
      const sectionBody = [];
      if (sectionMatch[4]) {
        sectionBody.push(sectionMatch[4]);
      }

      index += 1;
      while (index < lines.length) {
        const nextLine = lines[index].trim();
        if (!nextLine) {
          index += 1;
          break;
        }
        if (isSectionLine(nextLine) || isHeadingLine(nextLine) || isTableLine(nextLine) || isBulletLine(nextLine) || isOrderedListLine(nextLine) || nextLine.startsWith('```')) {
          break;
        }
        sectionBody.push(nextLine);
        index += 1;
      }

      const bodyHtml = sectionBody.length ? `<p class="answer-paragraph">${formatInlineMarkdown(sectionBody.join(' '))}</p>` : '';
      blocks.push(
        `<section class="answer-section"><h3 class="answer-heading"><span class="answer-index">${sectionNumber}.</span> ${formatInlineMarkdown(sectionTitle)}</h3>${bodyHtml}</section>`,
      );
      continue;
    }

    const headingMatch = isHeadingLine(line);
    if (headingMatch) {
      const headingText = headingMatch[2];
      blocks.push(`<h3 class="answer-heading">${formatInlineMarkdown(headingText)}</h3>`);
      index += 1;
      continue;
    }

    if (isTableLine(line)) {
      const tableRows = [];
      while (index < lines.length && isTableLine(lines[index].trim())) {
        tableRows.push(lines[index].trim());
        index += 1;
      }
      blocks.push(renderMarkdownTable(tableRows));
      continue;
    }

    const bulletMatch = isBulletLine(line);
    if (bulletMatch) {
      const items = [];
      while (index < lines.length) {
        const currentLine = lines[index].trim();
        const currentBullet = isBulletLine(currentLine);
        if (!currentBullet) {
          break;
        }
        items.push(currentBullet[1]);
        index += 1;
      }
      blocks.push(`<ul class="answer-list">${items.map((item) => `<li>${formatInlineMarkdown(item)}</li>`).join('')}</ul>`);
      continue;
    }

    const orderedMatch = isOrderedListLine(line);
    if (orderedMatch) {
      const items = [];
      while (index < lines.length) {
        const currentLine = lines[index].trim();
        const currentOrdered = isOrderedListLine(currentLine);
        if (!currentOrdered) {
          break;
        }
        items.push(currentOrdered[2]);
        index += 1;
      }
      blocks.push(`<ol class="answer-list">${items.map((item) => `<li>${formatInlineMarkdown(item)}</li>`).join('')}</ol>`);
      continue;
    }

    const paragraphLines = [line];
    index += 1;
    while (index < lines.length) {
      const nextLine = lines[index].trim();
      if (!nextLine) {
        index += 1;
        break;
      }
      if (isSectionLine(nextLine) || isHeadingLine(nextLine) || isTableLine(nextLine) || isBulletLine(nextLine) || isOrderedListLine(nextLine) || nextLine.startsWith('```')) {
        break;
      }
      paragraphLines.push(nextLine);
      index += 1;
    }
    blocks.push(`<p class="answer-paragraph">${formatInlineMarkdown(paragraphLines.join(' '))}</p>`);
  }

  return blocks.join('');
}

function renderMeta(result) {
  const chips = [
    ['Intent', result.intent],
    ['Answer type', result.answer_type || 'Unknown'],
    ['Iterations', String(result.iterations)],
    ['Latency', `${result.latency_ms} ms`],
    ['Tools', result.tools_used?.length ? result.tools_used.join(', ') : 'None'],
  ];

  elements.resultMeta.innerHTML = chips
    .map(([label, value]) => `<span class="meta-chip"><strong>${label}:</strong> ${escapeHtml(value)}</span>`)
    .join('');

  if (result.confidence) {
    elements.resultMeta.innerHTML += ` <span class="meta-chip confidence-chip confidence-${escapeHtml(result.confidence)}"><strong>Confidence:</strong> ${escapeHtml(result.confidence)}</span>`;
  }

  if (Array.isArray(result.plan) && result.plan.length > 0) {
    elements.resultMeta.innerHTML += ` <span class="meta-chip"><strong>Plan:</strong> ${escapeHtml(truncateText(result.plan.join(' → '), 140))}</span>`;
  }

  if (Array.isArray(result.validation_errors) && result.validation_errors.length > 0) {
    elements.resultMeta.innerHTML += ` <span class="meta-chip meta-chip-warning"><strong>Validation:</strong> ${escapeHtml(String(result.validation_errors.length))} issue(s)</span>`;
  }
}

function renderSources(sources) {
  if (!Array.isArray(sources) || sources.length === 0) {
    elements.sourcesList.innerHTML = '<div class="empty-state">No sources captured for this answer.</div>';
    return;
  }

  elements.sourcesList.innerHTML = sources
    .map((source) => {
      const url = String(source.url || '').trim();
      const title = String(source.title || '').trim();
      const snippet = String(source.snippet || '').trim();
      let host = url;
      try {
        host = new URL(url).hostname.replace(/^www\./, '');
      } catch (error) {
        host = url;
      }
      const label = title || host || url || 'Source';
      return `
        <a class="source-item" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">
          <span class="source-title">${escapeHtml(label)}</span>
          <span class="source-url">${escapeHtml(host)}</span>
          ${snippet ? `<span class="source-snippet">${escapeHtml(snippet)}</span>` : ''}
        </a>
      `;
    })
    .join('');
}

function renderAnswer(answer) {
  const rendered = renderMarkdownContent(answer);
  elements.answerCard.innerHTML = rendered
    ? `<div class="answer-content">${rendered}</div>`
    : '<div class="answer-placeholder">Run a question to see the answer.</div>';
}

function renderValidationErrors(validationErrors) {
  if (!Array.isArray(validationErrors) || validationErrors.length === 0) {
    elements.validationCard.innerHTML = `
      <div class="card-title">Validation</div>
      <div class="empty-state">No validation issues were reported.</div>
    `;
    return;
  }

  elements.validationCard.innerHTML = `
    <div class="card-title">Validation</div>
    <div class="validation-list">
      ${validationErrors.map((item) => `<div class="validation-item">${escapeHtml(item)}</div>`).join('')}
    </div>
  `;
}

function renderTrace(trace) {
  if (!Array.isArray(trace) || trace.length === 0) {
    elements.traceList.innerHTML = '<div class="empty-state">No trace steps were captured.</div>';
    return;
  }

  elements.traceList.innerHTML = trace
    .map((step) => {
      const actionClass = step.action === 'FINISH' ? ' action-finish' : '';
      const thoughtText = truncateText(step.thought, 220);
      const observation = step.observation
        ? `<div class="trace-label">Observation</div><p class="trace-text">${escapeHtml(truncateText(step.observation, 260))}</p>`
        : '<div class="trace-label">Observation</div><p class="trace-text">None</p>';
      return `
        <article class="trace-step${actionClass}">
          <div class="trace-meta">
            <span class="trace-pill">${escapeHtml(step.action)}</span>
          </div>
          <div class="trace-label">Thought</div>
          <p class="trace-text">${escapeHtml(thoughtText)}</p>
          ${observation}
        </article>
      `;
    })
    .join('');
}

function renderResult(result) {
  state.currentResult = result;
  renderMeta(result);
  renderAnswer(result.answer);
  renderValidationErrors(result.validation_errors || []);
  renderSources(result.sources || []);
  renderTrace(result.trace);
  elements.rawJson.textContent = prettyJson(result);
}

function renderHealth(payload) {
  const dependencyTexts = payload.dependencies || {};
  const groqState = dependencyTexts.groq_api_key ? 'Ready' : 'Missing';
  const tavilyState = dependencyTexts.tavily_api_key ? 'Ready' : 'Missing';

  elements.currentModel.textContent = payload.model || 'llama-3.3-70b-versatile';
  elements.toolsCount.textContent = String((payload.tools || []).length);
  elements.groqStatus.textContent = groqState;
  elements.tavilyStatus.textContent = tavilyState;
  elements.healthJson.textContent = prettyJson(payload);

  if (payload.status === 'ok') {
    setStatusTone('ok', 'Online');
  } else if (payload.status) {
    setStatusTone('warn', payload.status);
  } else {
    setStatusTone('bad', 'Unavailable');
  }
}

function getPayload() {
  return {
    query: elements.queryInput.value.trim(),
    max_iterations: Number(elements.maxIterations.value),
    model_name: elements.modelInput.value.trim() || null,
    depth_mode: elements.depthMode.value,
  };
}

async function fetchJson(url, options) {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...(options?.headers || {}),
    },
    ...options,
  });

  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (error) {
    data = { detail: text };
  }

  if (!response.ok) {
    const message = data?.detail || data?.message || `Request failed with ${response.status}`;
    throw new Error(message);
  }

  return data;
}

async function refreshHealth() {
  setStatusTone('warn', 'Checking...');
  try {
    const payload = await fetchJson('/health', { method: 'GET' });
    renderHealth(payload);
  } catch (error) {
    setStatusTone('bad', 'Offline');
    elements.healthJson.textContent = String(error.message || error);
    elements.groqStatus.textContent = 'Unknown';
    elements.tavilyStatus.textContent = 'Unknown';
  }
}

function setExampleQuery(query) {
  elements.queryInput.value = query;
  elements.queryInput.focus();
}

function setDefaultSample() {
  setExampleQuery(SAMPLE_QUERY);
  elements.depthMode.value = 'learning_ml';
}

function clearOutput() {
  state.currentResult = null;
  elements.answerCard.innerHTML = '<div class="answer-placeholder">Run a question to see the answer.</div>';
  elements.validationCard.innerHTML = `
    <div class="card-title">Validation</div>
    <div class="empty-state">Validation feedback appears when the backend flags formatting issues.</div>
  `;
  elements.sourcesList.innerHTML = '<div class="empty-state">No sources captured for this answer.</div>';
  elements.traceList.innerHTML = '<div class="empty-state">Steps show up here after a request.</div>';
  elements.streamOutput.textContent = 'Waiting for stream...';
  elements.rawJson.textContent = 'No data yet.';
  elements.resultMeta.innerHTML = '<span class="meta-chip">No result</span>';
}

function renderRequestError(message) {
  const safeMessage = escapeHtml(message);
  elements.answerCard.innerHTML = `<div class="answer-placeholder">${safeMessage}</div>`;
  elements.validationCard.innerHTML = `
    <div class="card-title">Validation</div>
    <div class="validation-list">
      <div class="validation-item validation-item-error">${safeMessage}</div>
    </div>
  `;
  elements.sourcesList.innerHTML = '<div class="empty-state">No sources captured for this answer.</div>';
  elements.traceList.innerHTML = '<div class="empty-state">No trace available for the failed request.</div>';
}

async function runQuery(endpoint) {
  const payload = getPayload();
  if (!payload.query) {
    setExampleQuery(SAMPLE_QUERY);
    return;
  }

  setLoading(true);
  elements.rawJson.textContent = 'Waiting...';
  try {
    const result = await fetchJson(endpoint, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    renderResult(result);
  } catch (error) {
    elements.rawJson.textContent = String(error.message || error);
    renderRequestError(String(error.message || error));
  } finally {
    setLoading(false);
  }
}

function readEventBlock(block) {
  const lines = block
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);

  const dataLines = lines.filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trim());
  if (dataLines.length === 0) {
    return null;
  }

  const payloadText = dataLines.join('\n');
  if (payloadText === '[DONE]') {
    return { type: 'done' };
  }

  try {
    return JSON.parse(payloadText);
  } catch (error) {
    return { type: 'raw', text: payloadText };
  }
}

async function streamQuery() {
  const payload = getPayload();
  if (!payload.query) {
    setExampleQuery(SAMPLE_QUERY);
    return;
  }

  if (state.currentAbortController) {
    state.currentAbortController.abort();
  }

  const controller = new AbortController();
  state.currentAbortController = controller;
  setLoading(true);
  elements.streamOutput.textContent = '';
  elements.rawJson.textContent = 'Waiting...';
  elements.answerCard.innerHTML = '<div class="answer-placeholder">Streaming...</div>';
  elements.validationCard.innerHTML = `
    <div class="card-title">Validation</div>
    <div class="empty-state">Validation feedback will appear with the final streamed answer.</div>
  `;
  elements.traceList.innerHTML = '<div class="empty-state">Steps will appear when the stream ends.</div>';

  try {
    const response = await fetch('/agent/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok || !response.body) {
      throw new Error(`Stream failed with ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let streamedText = '';
    let finalResult = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split('\n\n');
      buffer = chunks.pop() || '';

      for (const chunk of chunks) {
        const event = readEventBlock(chunk);
        if (!event) {
          continue;
        }

        if (event.type === 'token') {
          streamedText += event.text || '';
          elements.streamOutput.textContent = streamedText;
        } else if (event.type === 'final' && event.result) {
          finalResult = event.result;
        } else if (event.type === 'error') {
          throw new Error(event.message || 'Stream error');
        } else if (event.type === 'done') {
          break;
        }
      }
    }

    if (finalResult) {
      renderResult(finalResult);
      elements.streamOutput.textContent = finalResult.answer || streamedText || 'Stream completed.';
    } else if (streamedText.trim()) {
      elements.streamOutput.textContent = streamedText;
    }
  } catch (error) {
    const message = error.name === 'AbortError' ? 'Stream cancelled.' : String(error.message || error);
    elements.streamOutput.textContent = message;
    elements.rawJson.textContent = message;
    renderRequestError(message);
  } finally {
    state.currentAbortController = null;
    setLoading(false);
  }
}

function wireEvents() {
  elements.maxIterations.addEventListener('input', () => {
    elements.maxIterationsValue.textContent = elements.maxIterations.value;
    elements.currentLimitLabel.textContent = `${elements.maxIterations.value} configurable steps`;
  });

  elements.queryForm.addEventListener('submit', (event) => {
    event.preventDefault();
    runQuery('/agent/query');
  });

  elements.streamBtn.addEventListener('click', () => {
    streamQuery();
  });

  elements.clearBtn.addEventListener('click', clearOutput);

  elements.loadExampleBtn.addEventListener('click', () => {
    setDefaultSample();
    elements.maxIterations.value = '5';
    elements.maxIterationsValue.textContent = '5';
  });

  elements.refreshHealthBtn.addEventListener('click', refreshHealth);

  elements.exampleButtons.forEach((button) => {
    button.addEventListener('click', () => {
      const query = button.getAttribute('data-query') || SAMPLE_QUERY;
      setExampleQuery(query);
    });
  });

  elements.navHomeBtn?.addEventListener('click', () => {
    state.currentView = 'home';
    elements.navHomeBtn.classList.add('active');
    elements.navAboutBtn.classList.remove('active');
    elements.viewHome.style.display = '';
    elements.viewAbout.style.display = 'none';
  });

  elements.navAboutBtn?.addEventListener('click', () => {
    state.currentView = 'about';
    elements.navAboutBtn.classList.add('active');
    elements.navHomeBtn.classList.remove('active');
    elements.viewAbout.style.display = '';
    elements.viewHome.style.display = 'none';
  });

  elements.toggleDebugBtn?.addEventListener('click', () => {
    state.debugVisible = !state.debugVisible;
    elements.debugSection.style.display = state.debugVisible ? '' : 'none';
    elements.toggleDebugBtn.textContent = state.debugVisible ? 'Hide Developer View' : 'Show Developer View';
  });
}

function initialize() {
  elements.queryInput.value = '';
  elements.modelInput.value = '';
  elements.depthMode.value = 'learning_ml';
  elements.maxIterationsValue.textContent = elements.maxIterations.value;
  elements.currentLimitLabel.textContent = `${elements.maxIterations.value} configurable steps`;
  wireEvents();
  refreshHealth();
  clearOutput();
  elements.queryInput.focus();
}

initialize();
