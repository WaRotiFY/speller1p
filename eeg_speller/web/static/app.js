'use strict';
// Participant window: follows the session started in the experimenter window,
// shows the flashes, collects space presses and reports progress live.

const $ = (id) => document.getElementById(id);
const ui = {
  matrix: $('matrix'), state: $('state-pill'), targetChar: $('target-char'), targetProgress: $('target-progress'),
  flashLabel: $('flash-label'), flashHelp: $('flash-help'), flashIcon: $('flash-icon'),
  hits: $('hit-count'), progress: $('progress-fill'), epoch: $('epoch-label'),
  count: $('flash-count'), primary: $('primary-text'), secondary: $('secondary-text'),
  symbol: $('decision-symbol'), decision: $('decision-detail'), note: $('session-note'),
};

let latest = null;        // latest participant snapshot from the server
let runningId = null;     // session whose epochs this window is running
let runToken = 0;
let activeFlash = null;
let activeCells = [];
let epochHits = [];
let epochId = null;
let wake = null;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const show = (s) => (s === ' ' ? '␣' : s || '—');

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

// Fire-and-forget progress report; never delays the flash schedule.
function report(event, flashId) {
  if (!latest || epochId === null) return;
  fetch('/api/progress', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({session_id: latest.session_id, epoch_id: epochId, event, flash_id: flashId}),
  }).catch(() => {});
}

function setStatus(text, kind = '') {
  ui.state.textContent = text;
  ui.state.className = 'state-pill' + (kind ? ` ${kind}` : '');
}

function setMessage(label, help, icon = '✧') {
  ui.flashLabel.textContent = label;
  ui.flashHelp.textContent = help;
  ui.flashIcon.textContent = icon;
}

function renderMatrix(symbols) {
  ui.matrix.replaceChildren();
  symbols.forEach((symbol, index) => {
    const cell = document.createElement('div');
    cell.className = 'cell' + (symbol === ' ' || symbol === '⌫' ? ' special' : '');
    cell.textContent = symbol === ' ' ? '␣' : symbol;
    cell.setAttribute('role', 'gridcell');
    cell.setAttribute('aria-label', symbol === ' ' ? 'пробел' : symbol === '⌫' ? 'удалить' : symbol);
    cell.dataset.index = String(index);
    ui.matrix.append(cell);
  });
}

function clearFlash() {
  activeFlash = null;
  activeCells.forEach((i) => ui.matrix.children[i]?.classList.remove('flash', 'hit'));
  activeCells = [];
}

function showFlash(flash) {
  clearFlash();
  activeFlash = flash;
  activeCells = flash.indices;
  activeCells.forEach((i) => ui.matrix.children[i]?.classList.add('flash'));
  const name = flash.kind === 'row' ? `Строка ${flash.index + 1}` :
    flash.kind === 'column' ? `Столбец ${flash.index + 1}` : `Символ ${flash.index + 1}`;
  setMessage(`${name} · повтор ${flash.repetition + 1}`, 'Если здесь ваш символ — нажмите пробел сейчас', '✳');
}

function renderText(state) {
  if (ui.matrix.children.length !== state.symbols.length) renderMatrix(state.symbols);
  ui.primary.textContent = state.text || '';
  if (!state.text) {
    const placeholder = document.createElement('span');
    placeholder.className = 'placeholder';
    placeholder.textContent = 'Текст появится здесь';
    ui.primary.append(placeholder);
  }
  ui.secondary.textContent = state.secondary || '—';
  const next = state.target && state.target_index < state.target.length ? state.target[state.target_index] : null;
  ui.targetChar.textContent = next === ' ' ? '␣' : next || '—';
  ui.targetProgress.textContent = state.target ? `${state.target_index} / ${state.target.length} символов` : 'Свободный набор';
  ui.epoch.textContent = state.finished ? 'Эпоха —' :
    `Эпоха ${state.epoch_id + 1}` + (state.retry ? ` · повторная попытка ${state.retry}` : '');
  if (state.last) {
    ui.symbol.textContent = show(state.last.symbol);
    ui.decision.textContent = state.last.symbol ? 'Символ выбран' :
      `Недостаточно сигнала · ${state.last.hits} сигналов, повторяем`;
  } else {
    ui.symbol.textContent = '—';
    ui.decision.textContent = 'Ожидается первая эпоха';
  }
}

function waitForChange() {
  return new Promise((resolve) => { wake = resolve; });
}

function onLive(data) {
  const state = data.session;
  latest = state;
  if (wake) { const resolve = wake; wake = null; resolve(); }
  if (!state) {
    runToken++; runningId = null; clearFlash();
    setStatus('Ожидание');
    setMessage('Ожидание', 'Экспериментатор настраивает и запускает сессию в своём окне.');
    return;
  }
  if (state.finished) {
    if (runningId === state.session_id || runningId === null) {
      runToken++; runningId = null; clearFlash();
    }
    renderText(state);
    ui.count.textContent = '0 / 0 вспышек';
    ui.progress.style.width = '0%';
    setStatus('Завершена');
    setMessage('Сессия завершена', state.end_reason === 'target_finished' ? 'Фраза набрана. Спасибо!' :
      'Экспериментатор остановил сессию.');
    ui.note.textContent = 'Нажатия и решения записаны на этом компьютере.';
    return;
  }
  if (runningId !== state.session_id) {
    runToken++;
    runningId = state.session_id;
    clearFlash();
    renderText(state);
    runEpochs(runToken, state.session_id);
  }
}

function stillValid(token, sessionId) {
  return token === runToken && latest && latest.session_id === sessionId && !latest.finished;
}

async function runEpochs(token, sessionId) {
  try {
    while (stillValid(token, sessionId)) {
      if (latest.paused) {
        clearFlash();
        ui.count.textContent = '—';
        ui.progress.style.width = '0%';
        setStatus('Пауза', 'paused');
        setMessage('Пауза', 'Экспериментатор поставил паузу. Можно отдохнуть.', '❚❚');
        await waitForChange();
        continue;
      }
      const state = latest;
      const plan = state.plan;
      epochId = state.epoch_id;
      epochHits = Array(plan.length).fill(false);
      ui.hits.textContent = '0';
      ui.count.textContent = `0 / ${plan.length} вспышек`;
      ui.progress.style.width = '0%';
      setStatus('Идёт сессия', 'active');
      setMessage('Приготовьтесь', 'Найдите глазами свой символ — подсветка скоро начнётся');
      await sleep(state.timing.pause_ms);
      if (!stillValid(token, sessionId) || latest.paused || latest.epoch_id !== epochId) continue;
      report('epoch_begin');
      for (let i = 0; i < plan.length; i++) {
        if (!stillValid(token, sessionId)) return;
        showFlash(plan[i]);
        report('flash', i);
        ui.count.textContent = `${i + 1} / ${plan.length} вспышек`;
        ui.progress.style.width = `${100 * (i + 1) / plan.length}%`;
        await sleep(state.timing.flash_ms);
        clearFlash();
        if (i + 1 < plan.length) await sleep(state.timing.isi_ms);
      }
      if (!stillValid(token, sessionId)) return;
      report('flashes_done');
      setMessage('Обрабатываю сигналы…', 'Декодер объединяет ваши нажатия с языковой моделью');
      setStatus('Декодирование', 'active');
      let next;
      try {
        next = await api('/api/epoch', {session_id: sessionId, epoch_id: epochId, hits: epochHits});
      } catch (error) {
        if (!stillValid(token, sessionId)) return;   // stopped while decoding
        throw error;
      }
      if (token !== runToken) return;
      latest = next;
      renderText(next);
      if (next.finished) { onLive({session: next}); return; }
    }
  } catch (error) {
    clearFlash();
    setStatus('Ошибка', 'error');
    setMessage(error.message, 'Сообщите экспериментатору');
  } finally {
    epochId = null;
  }
}

async function liveLoop() {
  let since = -1;
  for (;;) {
    try {
      const data = await api(`/api/live?role=participant&since=${since}&timeout=15`);
      since = data.control_version;
      onLive(data);
    } catch (error) {
      setStatus('Нет связи', 'error');
      await sleep(1500);
    }
  }
}

document.addEventListener('keydown', (event) => {
  if (event.code !== 'Space') return;
  event.preventDefault();
  if (!latest || latest.finished || !activeFlash || event.repeat || epochHits[activeFlash.id]) return;
  epochHits[activeFlash.id] = true;
  report('hit', activeFlash.id);
  ui.hits.textContent = String(epochHits.filter(Boolean).length);
  activeCells.forEach((i) => ui.matrix.children[i]?.classList.add('hit'));
  ui.flashIcon.textContent = '✓';
  ui.flashHelp.textContent = 'Сигнал принят';
});

async function init() {
  try {
    const data = await api('/api/configs');
    renderMatrix(data.symbols);
    setStatus('Ожидание');
  } catch (error) {
    setStatus('Нет связи', 'error');
  }
  liveLoop();
}
init();
