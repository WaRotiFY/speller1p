'use strict';
// Experimenter window: configures the experimental blocks, controls the session
// and mirrors the participant window through /api/live (long polling).

const $ = (id) => document.getElementById(id);
const ui = {
  presence: $('presence'), presenceText: $('presence-text'), pill: $('session-pill'),
  openParticipant: $('open-participant'), participant: $('participant'), experiment: $('experiment'),
  task: $('task'), customRow: $('custom-row'), custom: $('custom'), profile: $('profile'),
  profileHint: $('profile-hint'), start: $('start'), pause: $('pause'), stop: $('stop'), error: $('error'),
  groups: $('groups'), blocksSub: $('blocks-sub'), presetName: $('preset-name'), savePreset: $('save-preset'),
  presetHint: $('preset-hint'), stats: $('stats'), epochSub: $('epoch-sub'), mirror: $('mirror'),
  curTarget: $('cur-target'), curFlash: $('cur-flash'), curHits: $('cur-hits'), curProgress: $('cur-progress'),
  leaderSymbol: $('leader-symbol'), leaderFill: $('leader-fill'), leaderTau: $('leader-tau'), leaderText: $('leader-text'),
  words: $('words'), predSub: $('pred-sub'), decision: $('decision'), qwen: $('qwen'), qwenText: $('qwen-text'),
  distSub: $('dist-sub'), priorCap: $('prior-cap'), likCap: $('lik-cap'), postCap: $('post-cap'),
  heatPrior: $('heat-prior'), heatLik: $('heat-lik'), heatPost: $('heat-post'),
  topPrior: $('top-prior'), topLik: $('top-lik'), topPost: $('top-post'),
  tTarget: $('t-target'), tPrimary: $('t-primary'), tSecondary: $('t-secondary'), textSub: $('text-sub'),
  corrections: $('corrections'), log: $('log'), logSub: $('log-sub'), dirNote: $('dir-note'),
};

let schema = null;
let symbols = [];
let baseline = {};          // values of the selected profile
let inputs = {};            // key -> input element
let session = null;         // latest monitor() payload
let live = null;            // latest /api/live payload
let view = 'live';
let filledFrom = null;      // session id whose settings are shown in the locked form
let logKey = '';
let qwenReady = false;

const fmt = new Intl.NumberFormat('ru-RU', {maximumFractionDigits: 1});
const fmt2 = new Intl.NumberFormat('ru-RU', {maximumFractionDigits: 2});
const show = (s) => (s === ' ' ? '␣' : s === null || s === undefined ? '—' : s);
const pct = (p, digits = 1) => (p === null || p === undefined ? '—' :
  `${new Intl.NumberFormat('ru-RU', {maximumFractionDigits: digits}).format(p * 100)} %`);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function showError(message) {
  ui.error.hidden = !message;
  ui.error.textContent = message || '';
}

function store(key, value) { try { localStorage.setItem(key, value); } catch (_) { /* optional */ } }
function recall(key) { try { return localStorage.getItem(key); } catch (_) { return null; } }

// ------------------------------------------------------------------ settings form
function buildForm() {
  ui.groups.replaceChildren();
  inputs = {};
  for (const group of schema.groups) {
    const box = el('details', 'group');
    box.open = ['mode', 'stimulus', 'fusion', 'detector'].includes(group.id);
    box.dataset.group = group.id;
    const summary = el('summary');
    summary.append(el('span', 'group-title', group.title), el('span', 'group-refs', group.refs));
    const changed = el('span', 'group-dot');
    changed.title = 'Есть изменения относительно профиля';
    summary.append(changed);
    box.append(summary);
    for (const field of schema.fields.filter((f) => f.group === group.id)) {
      const row = el('div', 'field-row');
      row.dataset.key = field.key;
      const id = `f-${field.key.replaceAll('.', '-')}`;
      const label = el('label', '', field.label);
      label.htmlFor = id;
      let input;
      if (field.kind === 'choice') {
        input = el('select');
        for (const choice of field.choices) {
          const option = el('option', '', choice.label);
          option.value = choice.value;
          input.append(option);
        }
      } else if (field.kind === 'bool') {
        input = el('input');
        input.type = 'checkbox';
        row.classList.add('check');
      } else {
        input = el('input');
        input.type = field.kind === 'text' ? 'text' : 'number';
        if (field.kind !== 'text') {
          input.min = field.min; input.max = field.max; input.step = field.step;
          input.inputMode = field.kind === 'int' ? 'numeric' : 'decimal';
        } else {
          input.maxLength = 200;
          input.spellcheck = false;
        }
      }
      input.id = id;
      input.dataset.key = field.key;
      input.dataset.kind = field.kind;
      input.addEventListener('input', onFormChange);
      input.addEventListener('change', onFormChange);
      const base = el('span', 'base-value');
      label.append(base);
      row.append(label, input);
      if (field.help) row.append(el('p', 'hint', field.help));
      box.append(row);
      inputs[field.key] = input;
    }
    ui.groups.append(box);
  }
}

function readValue(input) {
  const kind = input.dataset.kind;
  if (kind === 'bool') return input.checked;
  if (kind === 'int' || kind === 'float') {
    const value = input.value.trim() === '' ? NaN : Number(input.value.replace(',', '.'));
    return Number.isFinite(value) ? value : input.value;
  }
  return input.value;
}

function writeValue(input, value) {
  if (input.dataset.kind === 'bool') input.checked = Boolean(value);
  else input.value = value === null || value === undefined ? '' : String(value);
}

function readValues() {
  return Object.fromEntries(Object.entries(inputs).map(([key, input]) => [key, readValue(input)]));
}

function same(a, b) {
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) < 1e-9;
  return a === b;
}

function changedSettings() {
  const values = readValues();
  return Object.fromEntries(Object.entries(values).filter(([key, value]) => !same(value, baseline[key])));
}

function describe(key, value) {
  const field = schema.fields.find((f) => f.key === key);
  if (field?.kind === 'choice') return field.choices.find((c) => c.value === value)?.value ?? String(value);
  if (field?.kind === 'bool') return value ? 'да' : 'нет';
  return String(value);
}

function onFormChange() {
  const values = readValues();
  const changed = new Set(Object.keys(changedSettings()));
  for (const [key, input] of Object.entries(inputs)) {
    const row = input.closest('.field-row');
    row.classList.toggle('changed', changed.has(key));
    row.querySelector('.base-value').textContent = changed.has(key) ? `профиль: ${describe(key, baseline[key])}` : '';
  }
  for (const box of ui.groups.children) {
    box.classList.toggle('has-changes', [...box.querySelectorAll('.field-row.changed')].length > 0);
  }
  // Dependencies between blocks: show what a setting does not affect.
  const llmOff = values['mode.llm'] === 'off';
  const linear = values['fusion.strategy'] === 'linear_pool';
  const inactive = {
    'fusion.alpha': llmOff || linear, 'fusion.beta': linear, 'fusion.lambda': llmOff || !linear,
    'slow_llm.every_k_symbols': values['slow_llm.trigger'] !== 'every_k_symbols',
    'detector.max_retries': values['detector.mode'] !== 'abstain',
  };
  for (const box of ui.groups.children) {
    const off = llmOff && ['fast', 'slow'].includes(box.dataset.group);
    box.classList.toggle('inactive', off);
  }
  for (const [key, input] of Object.entries(inputs)) {
    input.closest('.field-row').classList.toggle('inactive', Boolean(inactive[key]));
  }
  ui.profileHint.textContent = changed.size ?
    `Изменено полей: ${changed.size}. Будут записаны в config.yaml сессии.` :
    'Поля ниже заполняются из профиля. Изменённые помечены точкой.';
}

function lockForm(locked) {
  for (const input of Object.values(inputs)) input.disabled = locked;
  for (const node of [ui.participant, ui.experiment, ui.task, ui.custom, ui.profile, ui.presetName, ui.savePreset]) {
    node.disabled = locked;
  }
  ui.blocksSub.textContent = locked ? 'идёт сессия — настройки зафиксированы' : 'меняются между сессиями';
}

async function loadProfile(name) {
  const data = await api(`/api/profile?name=${encodeURIComponent(name)}`);
  baseline = data.values;
  for (const [key, input] of Object.entries(inputs)) writeValue(input, baseline[key]);
  onFormChange();
}

function taskChoice() {
  const value = ui.task.value;
  if (value === '__custom') return {kind: 'custom', text: ui.custom.value};
  if (value === '__free') return {kind: 'free'};
  return {kind: 'task', id: value};
}

// ------------------------------------------------------------------ heatmaps
function buildGrid(container, small = false) {
  container.replaceChildren();
  symbols.forEach((symbol, index) => {
    const cell = el('div', small ? 'mcell' : 'hcell');
    cell.dataset.index = String(index);
    const name = el('b', '', show(symbol));
    cell.append(name);
    if (!small) cell.append(el('small', '', ''));
    container.append(cell);
  });
}

const LOW = [243, 248, 247];
const HIGH = [8, 72, 80];
function shade(t) {
  const mix = LOW.map((low, i) => Math.round(low + (HIGH[i] - low) * t));
  return `rgb(${mix[0]}, ${mix[1]}, ${mix[2]})`;
}

function paintHeat(container, list, probs, marks) {
  const cells = container.children;
  list.replaceChildren();
  if (!probs) {
    for (const cell of cells) {
      cell.style.background = '';
      cell.className = 'hcell empty';
      cell.querySelector('small').textContent = '';
      cell.title = '';
    }
    list.append(el('li', 'muted', 'нет данных'));
    return;
  }
  const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
  const rank = new Map(order.map(([, i], r) => [i, r + 1]));
  const leader = order[0][1];
  probs.forEach((p, i) => {
    const cell = cells[i];
    const t = Math.sqrt(Math.max(0, Math.min(1, p)));
    cell.style.background = shade(t);
    cell.className = 'hcell' + (t > 0.55 ? ' dark' : '') + (i === marks.target ? ' target' : '') +
      (i === leader ? ' leader' : '') + (i === marks.decided ? ' decided' : '');
    cell.querySelector('small').textContent = p >= 0.005 ? fmt.format(p * 100) : '';
    cell.title = `${show(symbols[i])}: ${pct(p, 2)} · место ${rank.get(i)}`;
  });
  for (const [p, i] of order.slice(0, 5)) {
    const item = el('li', i === marks.target ? 'is-target' : '');
    item.append(el('b', '', show(symbols[i])), el('span', '', pct(p)));
    list.append(item);
  }
}

// ------------------------------------------------------------------ rendering
function statTile(label, value, note) {
  const tile = el('div', 'stat');
  tile.append(el('span', '', label), el('strong', '', value));
  if (note) tile.append(el('small', '', note));
  return tile;
}

function clock(seconds) {
  const s = Math.round(seconds || 0);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function renderStats(s) {
  const st = s.stats;
  const tiles = [
    statTile('Эпох', String(st.epochs), st.repeats ? `повторов: ${st.repeats}` : 'без повторов'),
    statTile('Выбрано символов', String(st.selected), s.target ? `цель: ${s.target.length}` : 'свободный набор'),
    statTile('Точность выбора', st.accuracy === null ? '—' : pct(st.accuracy, 0),
      s.target ? `${st.correct} из ${st.selected}` : 'нет цели'),
    statTile('Совпало с целью', st.matching_chars === null ? '—' : `${st.matching_chars} / ${s.target.length}`, 'посимвольно'),
    statTile('Активное время', clock(st.active_s), 'паузы не учитываются'),
    statTile('Скорость, симв/мин', st.speed_cpm === null ? '—' : fmt2.format(st.speed_cpm),
      st.selected_cpm === null ? 'по совпавшим с целью' : `выборов в минуту: ${fmt2.format(st.selected_cpm)}`),
  ];
  ui.stats.replaceChildren(...tiles);
}

function renderMirror(s) {
  const cur = s.current;
  const cells = ui.mirror.children;
  const counts = new Array(36).fill(0);
  const flashing = new Set(cur && cur.phase === 'flashing' && cur.flash_group ? cur.flash_group.indices : []);
  if (cur) for (const group of cur.hit_groups) for (const i of group) counts[i] += 1;
  const max = Math.max(1, ...counts);
  const target = cur && cur.target !== null ? symbols.indexOf(cur.target) : -1;
  for (let i = 0; i < cells.length; i++) {
    const cell = cells[i];
    cell.className = 'mcell' + (flashing.has(i) ? ' flash' : '') + (i === target ? ' target' : '') +
      (counts[i] ? ' hit' : '');
    cell.style.background = counts[i] && !flashing.has(i) ? shade(0.25 + 0.6 * counts[i] / max) : '';
    cell.title = counts[i] ? `${show(symbols[i])}: нажатий в группах с символом — ${counts[i]}` : show(symbols[i]);
  }
}

function renderEpoch(s) {
  const cur = s.current;
  if (!cur) {
    ui.epochSub.textContent = s.finished ? 'сессия завершена' : '—';
    ui.curTarget.textContent = '—'; ui.curFlash.textContent = '—'; ui.curHits.textContent = '—';
    ui.curProgress.style.width = '0%';
    ui.leaderSymbol.textContent = '—'; ui.leaderFill.style.width = '0%';
    ui.leaderText.textContent = s.finished ? 'Сессия завершена' : 'Ожидание данных';
    renderMirror({current: null});
    return;
  }
  const phase = {ready: 'ждём начала вспышек', flashing: 'идут вспышки', decoding: 'декодирование'}[cur.phase] || cur.phase;
  ui.epochSub.textContent = `эпоха ${cur.epoch_id + 1} · позиция ${cur.position + 1}` +
    (cur.retry ? ` · повтор ${cur.retry}` : '') + ` · ${s.paused ? 'пауза' : phase}`;
  ui.curTarget.textContent = cur.target === null ? 'свободный набор' : show(cur.target);
  const shown = Math.max(0, cur.flash + 1);
  ui.curFlash.textContent = `${shown} / ${cur.n_flashes}`;
  ui.curHits.textContent = String(cur.hit_flashes.length);
  ui.curProgress.style.width = `${100 * shown / Math.max(1, cur.n_flashes)}%`;
  const lead = cur.leader;
  ui.leaderSymbol.textContent = show(lead.symbol);
  ui.leaderFill.style.width = `${100 * lead.confidence}%`;
  ui.leaderFill.classList.toggle('low', lead.below_tau);
  ui.leaderTau.style.left = `${100 * s.tau}%`;
  ui.leaderTau.title = `τ = ${s.tau}`;
  const fate = lead.below_tau ?
    (s.detector_mode === 'abstain' ? 'ниже τ: если так и останется, эпоха повторится' :
      'ниже τ: если так и останется, символ выберется с меткой «неуверенно»') :
    `выше τ = ${fmt2.format(s.tau)}`;
  ui.leaderText.textContent = cur.evidence_flashes ?
    `${pct(lead.confidence)} · отрыв ${pct(lead.margin)} · ${fate}` :
    `Пока только prior: ${pct(lead.confidence)}. Нажатий ещё нет.`;
  renderMirror(s);
}

function renderDistributions(s) {
  const llmOff = s.llm === 'off';
  ui.priorCap.textContent = llmOff ? 'равномерный · без БЯМ' : `быстрая БЯМ · ${s.settings['fast_llm.backend']}`;
  let data = null;
  let marks = {target: -1, decided: -1};
  const useLast = s.last_epoch && (view === 'last' || !s.current);
  if (useLast) {
    const e = s.last_epoch;
    data = {prior: e.prior, likelihood: e.likelihood, posterior: e.posterior};
    marks = {target: e.target === null ? -1 : symbols.indexOf(e.target),
      decided: e.decided === null ? -1 : symbols.indexOf(e.decided)};
    ui.distSub.textContent = `эпоха ${e.epoch_id + 1} завершена · выбор: ${e.decided === null ? 'повтор' : show(e.decided)}`;
    ui.likCap.textContent = `нажатий: ${e.hits.length}`;
    ui.postCap.textContent = `${s.settings['fusion.strategy']} · уверенность ${pct(e.confidence)}`;
  } else if (s.current) {
    const cur = s.current;
    data = {prior: cur.prior, likelihood: cur.likelihood, posterior: cur.posterior};
    marks.target = cur.target === null ? -1 : symbols.indexOf(cur.target);
    ui.distSub.textContent = `эпоха ${cur.epoch_id + 1} · учтено вспышек: ${cur.evidence_flashes} из ${cur.n_flashes}`;
    ui.likCap.textContent = cur.likelihood ? `нажатий: ${cur.hit_flashes.length}` : 'ждём вспышек';
    ui.postCap.textContent = `${s.settings['fusion.strategy']} · обновляется на каждой вспышке`;
  } else {
    ui.distSub.textContent = '—';
  }
  paintHeat(ui.heatPrior, ui.topPrior, data?.prior, marks);
  paintHeat(ui.heatLik, ui.topLik, data?.likelihood, marks);
  paintHeat(ui.heatPost, ui.topPost, data?.posterior, marks);
}

function renderWords(s) {
  const words = s.last_epoch && (view === 'last' || !s.current) ? s.last_epoch.words : s.current?.words;
  ui.words.replaceChildren();
  if (s.llm === 'off') {
    ui.predSub.textContent = 'без БЯМ — выключено';
    ui.words.append(el('li', 'muted', 'В режиме без БЯМ кандидаты не считаются'));
    return;
  }
  ui.predSub.textContent = s.current ? `контекст: «${s.current.context_tail || ''}» · ${fmt2.format(s.current.words_ms)} мс` : 'быстрая БЯМ';
  if (!words || !words.length) {
    ui.words.append(el('li', 'muted', 'Нет кандидатов для этого префикса'));
    return;
  }
  const max = Math.max(...words.map((w) => w[1]));
  for (const [word, p] of words) {
    const item = el('li');
    const bar = el('span', 'bar');
    const fill = el('i');
    fill.style.width = `${100 * p / max}%`;
    bar.append(fill);
    item.append(el('b', '', word), bar, el('span', 'num', pct(p, 0)));
    ui.words.append(item);
  }
}

function renderDecision(s) {
  const row = s.log.length ? s.log[s.log.length - 1] : null;
  ui.decision.replaceChildren();
  if (!row) { ui.decision.textContent = '—'; return; }
  const badge = el('span', 'badge');
  if (row.decided === null) {
    badge.textContent = row.hits ? 'повтор: неуверенно' : 'повтор: нет нажатий';
    badge.classList.add('warn');
  } else if (row.correct === true) { badge.textContent = '✓ совпало с целью'; badge.classList.add('ok'); }
  else if (row.correct === false) { badge.textContent = `✗ цель «${show(row.target)}»`; badge.classList.add('bad'); }
  else badge.textContent = row.uncertain ? 'неуверенно' : 'выбрано';
  ui.decision.append(el('strong', '', show(row.decided ?? row.leader)),
    el('span', '', `эпоха ${row.epoch_id + 1} · уверенность ${pct(row.confidence)} · отрыв ${pct(row.margin)}`), badge);
}

function renderTexts(s) {
  ui.tTarget.replaceChildren();
  if (s.target) {
    [...s.target].forEach((ch, i) => {
      ui.tTarget.append(el('span', i < s.target_index ? 'done' : i === s.target_index ? 'cur' : '', ch === ' ' ? '␣' : ch));
    });
  } else ui.tTarget.textContent = 'Свободный набор';
  ui.tPrimary.replaceChildren();
  if (s.text) {
    [...s.text].forEach((ch, i) => {
      const bad = s.target && i < s.target.length && s.target[i] !== ch;
      ui.tPrimary.append(el('span', bad ? 'bad' : '', ch === ' ' && bad ? '␣' : ch));
    });
  } else ui.tPrimary.append(el('span', 'muted', 'пока пусто'));
  ui.tSecondary.textContent = s.llm === 'off' ? 'коррекция выключена (без БЯМ)' : (s.secondary || '—');
  ui.textSub.textContent = `${s.participant_id} · задание ${s.task_id}`;
}

function renderCorrections(s) {
  ui.corrections.replaceChildren();
  if (!s.corrections.length) {
    ui.corrections.append(el('li', 'muted', s.llm === 'off' ? 'Коррекция выключена (без БЯМ)' : 'Коррекций пока не было'));
    return;
  }
  for (const c of [...s.corrections].reverse()) {
    const item = el('li');
    const head = el('div', 'c-head');
    const badge = el('span', 'badge');
    if (!c.accepted) { badge.textContent = `отклонено: ${c.reject_reason}`; badge.classList.add('bad'); }
    else if (c.changed) { badge.textContent = 'текст изменён'; badge.classList.add('info'); }
    else badge.textContent = 'без изменений';
    head.append(el('span', '', `#${c.correction_id} · эпоха ${c.epoch_id + 1} · ${c.trigger} · ${fmt2.format(c.latency_ms)} мс`), badge);
    item.append(head, el('div', 'c-body', `${c.words_in.join(' ')} → ${c.words_out.join(' ')}`));
    ui.corrections.append(item);
  }
}

function renderLog(s) {
  const key = `${s.session_id}:${s.log.length ? s.log[s.log.length - 1].epoch_id : -1}`;
  if (key === logKey) return;
  logKey = key;
  ui.log.replaceChildren();
  for (const row of [...s.log].reverse()) {
    const tr = el('tr', row.decided === null ? 'retry' : row.correct === false ? 'wrong' : '');
    const mark = row.decided === null ? '↻' : row.correct === true ? '✓' : row.correct === false ? '✗' : '';
    const cells = [row.epoch_id + 1, row.position + 1, show(row.target), row.decided === null ? '—' : show(row.decided), mark,
      pct(row.confidence), pct(row.margin), pct(row.p_prior_target), pct(row.p_post_target),
      row.rank_target ?? '—', `${row.hits}/${row.n_flashes}`, fmt2.format(row.prior_ms)];
    for (const value of cells) tr.append(el('td', '', String(value)));
    ui.log.append(tr);
  }
  if (!s.log.length) {
    const tr = el('tr');
    const td = el('td', 'muted', 'Эпох ещё не было');
    td.colSpan = 12;
    tr.append(td);
    ui.log.append(tr);
  }
  ui.logSub.textContent = `последние ${Math.min(60, s.log.length)} · полный журнал в events.jsonl`;
}

function fillFormFromSession(s) {
  if (filledFrom === s.session_id) return;
  filledFrom = s.session_id;
  ui.participant.value = s.participant_id;
  ui.experiment.value = s.experiment_id || '';
  if ([...ui.profile.options].some((o) => o.value === s.profile)) ui.profile.value = s.profile;
  if (s.task_id === 'FREE') ui.task.value = '__free';
  else if (s.task_id === 'CUSTOM') { ui.task.value = '__custom'; ui.custom.value = s.target; }
  else if ([...ui.task.options].some((o) => o.value === s.task_id)) ui.task.value = s.task_id;
  ui.customRow.hidden = ui.task.value !== '__custom';
  for (const [key, input] of Object.entries(inputs)) writeValue(input, s.settings[key]);
  api(`/api/profile?name=${encodeURIComponent(s.profile)}`)
    .then((data) => { baseline = data.values; onFormChange(); })
    .catch(() => onFormChange());
}

function renderPresence(data) {
  const connected = data.participant_connected;
  ui.presence.querySelector('.dot').className = 'dot' + (connected ? '' : ' off');
  ui.presenceText.textContent = connected ?
    (data.participant_windows > 1 ? `Окон участника: ${data.participant_windows} — оставьте одно` : 'Окно участника подключено') :
    'Окно участника не открыто';
  ui.presence.classList.toggle('warn', data.participant_windows > 1);
}

function setPill(text, kind) {
  ui.pill.textContent = text;
  ui.pill.className = 'state-pill' + (kind ? ` ${kind}` : '');
}

function render(data) {
  live = data;
  renderPresence(data);
  const s = data.session;
  session = s;
  if (!s) {
    setPill('Нет сессии');
    lockForm(false);
    ui.start.disabled = false; ui.pause.disabled = true; ui.stop.disabled = true;
    return;
  }
  const running = !s.finished;
  if (!running) setPill(s.end_reason === 'target_finished' ? 'Завершена: цель набрана' : 'Завершена');
  else if (s.paused) setPill('Пауза', 'paused');
  else if (!data.participant_connected) setPill('Ждёт окно участника', 'error');
  else setPill('Идёт сессия', 'active');
  lockForm(running);
  if (running) fillFormFromSession(s);
  ui.start.disabled = running;
  ui.pause.disabled = !running;
  ui.stop.disabled = !running;
  ui.pause.textContent = s.paused ? 'Продолжить' : 'Пауза';
  ui.pause.classList.toggle('primary', s.paused);
  ui.pause.classList.toggle('secondary', !s.paused);
  ui.qwen.disabled = !qwenReady || !s.text;
  ui.dirNote.textContent = `Запись: ${s.directory}. Нажатия пробела принимает только окно участника.`;
  renderStats(s);
  renderEpoch(s);
  renderDistributions(s);
  renderWords(s);
  renderDecision(s);
  renderTexts(s);
  renderCorrections(s);
  renderLog(s);
}

async function liveLoop() {
  let since = -1;
  for (;;) {
    try {
      const data = await api(`/api/live?role=operator&since=${since}&timeout=3`);
      since = data.version;
      render(data);
    } catch (error) {
      setPill('Нет связи с сервером', 'error');
      await sleep(1500);
    }
  }
}

// ------------------------------------------------------------------ actions
ui.start.addEventListener('click', async () => {
  ui.start.blur();
  showError('');
  ui.start.disabled = true;
  try {
    const body = {
      profile: ui.profile.value, participant_id: ui.participant.value.trim(),
      experiment_id: ui.experiment.value.trim() || null, task: taskChoice(), settings: changedSettings(),
    };
    await api('/api/start', body);
    store('op.participant', body.participant_id);
    store('op.profile', body.profile);
    if (!live?.participant_connected) showError('Сессия создана. Откройте окно участника — набор начнётся там.');
  } catch (error) {
    showError(error.message);
    ui.start.disabled = false;
  }
});

ui.pause.addEventListener('click', async () => {
  ui.pause.blur();
  if (!session) return;
  try { await api('/api/pause', {session_id: session.session_id, paused: !session.paused}); }
  catch (error) { showError(error.message); }
});

ui.stop.addEventListener('click', async () => {
  ui.stop.blur();
  if (!session || session.finished) return;
  if (!window.confirm('Остановить сессию? Запись будет закрыта, продолжить её нельзя.')) return;
  try { await api('/api/stop', {session_id: session.session_id, by: 'operator'}); }
  catch (error) { showError(error.message); }
});

ui.openParticipant.addEventListener('click', () => {
  ui.openParticipant.blur();
  window.open('/', 'eeg-participant');
});

ui.task.addEventListener('change', () => { ui.customRow.hidden = ui.task.value !== '__custom'; });
ui.profile.addEventListener('change', () => {
  loadProfile(ui.profile.value).catch((error) => showError(error.message));
});

ui.savePreset.addEventListener('click', async () => {
  ui.savePreset.blur();
  try {
    const result = await api('/api/preset', {name: ui.presetName.value.trim(), profile: ui.profile.value, settings: changedSettings()});
    fillProfiles(result.profiles, ui.profile.value);
    ui.presetHint.textContent = `Сохранено: configs/${result.saved}. Его можно выбрать как базовый профиль.`;
  } catch (error) { ui.presetHint.textContent = error.message; }
});

ui.qwen.addEventListener('click', async () => {
  ui.qwen.blur();
  if (!session) return;
  ui.qwen.disabled = true;
  ui.qwenText.textContent = 'Локальная модель подбирает продолжение…';
  try {
    const result = await api('/api/suggest', {session_id: session.session_id});
    ui.qwenText.textContent = result.suggestion || 'Модель не предложила продолжение.';
  } catch (error) { ui.qwenText.textContent = error.message; }
  ui.qwen.disabled = !qwenReady || !session?.text;
});

for (const button of document.querySelectorAll('.seg button')) {
  button.addEventListener('click', () => {
    view = button.dataset.view;
    for (const other of document.querySelectorAll('.seg button')) other.classList.toggle('on', other === button);
    button.blur();
    if (session) { renderDistributions(session); renderWords(session); }
  });
}

// Space belongs to the participant window: never let it press a button here.
for (const type of ['keydown', 'keyup']) {
  document.addEventListener(type, (event) => {
    if (event.code === 'Space' && event.target instanceof HTMLElement &&
        /^(BUTTON|SUMMARY)$/.test(event.target.tagName)) event.preventDefault();
  });
}

function fillProfiles(names, selected) {
  ui.profile.replaceChildren();
  for (const name of names) {
    const option = el('option', '', name);
    option.value = name;
    ui.profile.append(option);
  }
  if (names.includes(selected)) ui.profile.value = selected;
}

async function init() {
  try {
    const data = await api('/api/operator/init');
    schema = data.schema;
    symbols = data.symbols;
    qwenReady = Boolean(data.qwen.ready);
    ui.qwenText.textContent = qwenReady ? 'Подсказка продолжения по первичному тексту.' : 'Qwen не подключён (см. README, раздел про Qwen).';
    buildForm();
    buildGrid(ui.heatPrior); buildGrid(ui.heatLik); buildGrid(ui.heatPost); buildGrid(ui.mirror, true);
    for (const task of data.tasks) {
      const option = el('option', '', `${task.id} · ${task.text}`);
      option.value = task.id;
      ui.task.append(option);
    }
    for (const [value, label] of [['__custom', 'Своя фраза…'], ['__free', 'Свободный набор, без цели']]) {
      const option = el('option', '', label);
      option.value = value;
      ui.task.append(option);
    }
    const savedProfile = recall('op.profile');
    fillProfiles(data.profiles, data.profiles.includes(savedProfile) ? savedProfile : 'gui_slow.yaml');
    ui.participant.value = recall('op.participant') || 'P001';
    await loadProfile(ui.profile.value);
    liveLoop();
  } catch (error) {
    setPill('Нет связи с сервером', 'error');
    showError(error.message);
  }
}
init();
