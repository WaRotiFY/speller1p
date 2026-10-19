"use strict";
const $ = (id) => document.getElementById(id);
const PHASES = { waiting_modules: "ждём модуль", presenting: "эпоха", result: "результат", finished: "завершено" };
const STORE = "speller.operator.v2";
let init = null, version = -1, current = null, dismissed = null, saved = {}, promptTimer = null;

async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) };
  const r = await fetch(path, opts);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
const shown = (s) => (s === " " ? "␣" : s);
const txt = (el, v) => { el.textContent = v == null ? "" : v; };
const pct = (p) => `${Math.round(p * 100)}%`;
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
function load() { try { return JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (e) { return {}; } }
function store(v) { try { localStorage.setItem(STORE, JSON.stringify(v)); } catch (e) { /* storage unavailable */ } }

/* ---------------- segmented buttons ---------------- */
function seg(id, onChange) {
  const box = $(id);
  const set = (v) => {
    box.dataset.value = v;
    for (const b of box.querySelectorAll("button")) b.classList.toggle("on", b.dataset.v === v);
    if (onChange) onChange(v);
  };
  for (const b of box.querySelectorAll("button")) b.onclick = () => set(b.dataset.v);
  return { set, get: () => box.dataset.value };
}
let sLlm, sFast, sCtx, sSlow;

/* ---------------- setup ---------------- */
async function boot() {
  init = await api("/api/init");
  saved = load();
  for (const t of init.tasks) $("task").add(new Option(`${t.id} — ${t.text}`, t.id));
  $("task").onchange = () => $("custom-row").classList.toggle("hidden", $("task").value !== "custom");

  // physiology
  const groups = { modules: "модули стимуляции (stimulus_kit/modules)", examples: "примеры (stimulus_kit/examples)" };
  for (const folder of Object.keys(groups)) {
    const items = init.programs.filter((p) => p.folder === folder);
    if (!items.length) continue;
    const og = document.createElement("optgroup");
    og.label = groups[folder];
    for (const p of items) og.appendChild(new Option(p.label, p.id));
    $("program").appendChild(og);
  }
  $("program").add(new Option("— не запускать (модуль запущу сам)", ""));
  $("program").onchange = () => programChanged(true);
  $("module-reset").onclick = () => { delete (saved.moduleSettings || {})[$("program").value]; programChanged(false); };

  // keyboard
  for (const [k, v] of Object.entries(init.keyboards)) $("kb-preset").add(new Option(v.label, k));
  $("kb-preset").add(new Option("своя клавиатура…", "custom"));
  for (const id of ["kb-preset", "kb-symbols", "kb-rows", "kb-cols"]) $(id).oninput = keyboardChanged;

  // fast LLM
  sLlm = seg("llm-mode", (v) => { for (const b of ["b-fast", "b-slow"]) $(b).style.opacity = v === "off" ? .45 : 1; });
  sFast = seg("fast-backend", (v) => {
    $("fast-ngram").classList.toggle("hidden", v !== "ngram");
    $("fast-hf").classList.toggle("hidden", v !== "hf_causal");
    $("fast-kenlm").classList.toggle("hidden", v !== "kenlm");
  });
  sCtx = seg("fast-context");
  for (const m of init.fast_models) $("ngram-file").add(new Option(m.file, m.file));
  $("ngram-file").onchange = ngramInfo;

  // inference
  for (const p of init.inference) $("inf-plugin").add(new Option(p.error ? `${p.id} (ошибка)` : `${p.label} — ${p.id}.py`, p.id));
  $("inf-plugin").onchange = () => inferenceChanged(true);
  $("stop-off").onchange = () => { $("stop-conf").disabled = $("stop-min").disabled = $("stop-off").checked; };

  // slow LLM
  sSlow = seg("slow-backend", (v) => $("slow-api").classList.toggle("hidden", v !== "chat_api"));
  for (const p of init.providers) $("provider").add(new Option(p.label, p.id));
  $("provider").onchange = () => providerChanged(true);
  $("key-save").onclick = saveKey;
  $("models-refresh").onclick = refreshModels;
  $("prompt-open").onclick = openPrompt;
  $("prompt-close").onclick = () => $("prompt-modal").classList.add("hidden");
  $("prompt-text").oninput = () => { clearTimeout(promptTimer); promptTimer = setTimeout(previewPrompt, 250); };
  $("prompt-save").onclick = savePrompt;
  $("prompt-reset").onclick = resetPrompt;
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("prompt-modal").classList.add("hidden"); });

  $("start-btn").onclick = start;
  $("pause-btn").onclick = () => api("/api/pause", { paused: !(current && current.paused) }).catch(alert);
  $("stop-btn").onclick = () => confirm("завершить сессию?") && api("/api/stop", {}).catch(alert);
  $("new-btn").onclick = () => { dismissed = current && current.session_id; render(null); };
  $("history-toggle").onclick = () => {
    const open = $("history").classList.toggle("open");
    txt($("history-toggle"), open ? "свернуть" : "показать всё");
  };
  restore();
  poll();
}

function restore() {
  const s = saved, d = init.defaults;
  if (s.participant) $("participant").value = s.participant;
  if (s.task && [...$("task").options].some((o) => o.value === s.task)) $("task").value = s.task;
  if (s.custom) $("custom").value = s.custom;
  $("task").onchange();
  sLlm.set(s.llm || "on");
  $("cue").checked = !!s.cue;
  const prog = init.programs.find((p) => p.id === s.program) ||
    init.programs.find((p) => p.id === "modules/flash_rows_cols") || init.programs[0];
  $("program").value = s.program === "" ? "" : (prog ? prog.id : "");
  $("space-eeg").checked = !!s.spaceEeg;
  $("windowed").checked = !!s.windowed;
  $("program-args").value = s.args || "";
  programChanged(false);
  $("kb-preset").value = s.kbPreset || "abc_6x6";
  $("kb-symbols").value = s.kbSymbols || "";
  $("kb-rows").value = s.kbRows || 6;
  $("kb-cols").value = s.kbCols || 6;
  keyboardChanged();
  const kenlmReady = init.kenlm.available && init.kenlm.downloaded.length > 0;
  sFast.set(s.fastBackend || (kenlmReady ? "kenlm" : "ngram"));
  if (s.ngramFile && init.fast_models.some((m) => m.file === s.ngramFile)) $("ngram-file").value = s.ngramFile;
  ngramInfo();
  $("hf-model").value = s.hfModel || "";
  $("kenlm-id").value = s.kenlmId || init.kenlm.default;
  $("kenlm-mix").value = s.kenlmMix ?? 0.1;
  $("kenlm-vocab").value = s.kenlmVocab ?? 60000;
  $("kenlm-download").checked = !!s.kenlmDownload;
  const kd = init.kenlm.downloaded.map((m) => m.id);
  txt($("kenlm-info"), (init.kenlm.available ? "" : "пакет kenlm не установлен: pip install -e \".[kenlm]\" (см. models/kenlm/README.md). ") +
    (kd.length ? `скачаны: ${kd.join(", ")}` : "моделей ещё не скачано") +
    ". словная модель даёт следующую букву через сумму вероятностей слов, продолжающих начатое слово; λ — доля буквенной 4-граммы на случай опечаток и редких слов.");
  $("hf-download").checked = !!s.hfDownload;
  $("hf-device").value = s.hfDevice || "cpu";
  sCtx.set(s.context || "primary");
  $("inf-plugin").value = s.plugin && init.inference.some((p) => p.id === s.plugin) ? s.plugin : "log_pool";
  inferenceChanged(false);
  $("stop-conf").value = s.stopConf ?? d.stop_confidence ?? 0.99;
  $("stop-min").value = s.stopMin ?? d.min_updates;
  $("pause-ms").value = s.pauseMs ?? d.pause_ms;
  $("stop-off").checked = !!s.stopOff;
  $("stop-off").onchange();
  sSlow.set(s.slowBackend || "mock_dictionary");
  $("provider").value = s.provider || "ollama";
  $("base-url").value = s.baseUrl || "";
  providerChanged(false);
  $("allow-remote").checked = !!s.allowRemote;
  $("every-n").value = s.everyN ?? d.every_n_words;
  $("window-w").value = s.windowW ?? d.window_words;
  $("alts").checked = s.alts ?? true;
  $("alts-k").value = s.altsK ?? d.alternatives_k;
  txt($("prompt-edited"), init.prompt.edited ? "(изменён)" : "(стандартный)");
}

function collectSaved() {
  saved = {
    ...saved, participant: $("participant").value.trim(), task: $("task").value, custom: $("custom").value,
    llm: sLlm.get(), cue: $("cue").checked, program: $("program").value, spaceEeg: $("space-eeg").checked,
    windowed: $("windowed").checked, args: $("program-args").value, kbPreset: $("kb-preset").value,
    kbSymbols: $("kb-symbols").value, kbRows: +$("kb-rows").value, kbCols: +$("kb-cols").value,
    fastBackend: sFast.get(), ngramFile: $("ngram-file").value, hfModel: $("hf-model").value.trim(),
    hfDownload: $("hf-download").checked, hfDevice: $("hf-device").value, context: sCtx.get(),
    kenlmId: $("kenlm-id").value.trim(), kenlmMix: +$("kenlm-mix").value, kenlmVocab: +$("kenlm-vocab").value,
    kenlmDownload: $("kenlm-download").checked,
    plugin: $("inf-plugin").value, stopConf: +$("stop-conf").value, stopMin: +$("stop-min").value,
    pauseMs: +$("pause-ms").value, stopOff: $("stop-off").checked, slowBackend: sSlow.get(),
    provider: $("provider").value, baseUrl: $("base-url").value.trim(), slowModel: $("slow-model").value.trim(),
    allowRemote: $("allow-remote").checked, everyN: +$("every-n").value, windowW: +$("window-w").value,
    alts: $("alts").checked, altsK: +$("alts-k").value,
  };
  saved.moduleSettings = { ...(saved.moduleSettings || {}), [$("program").value]: moduleSettingsValues() };
  saved.pluginParams = { ...(saved.pluginParams || {}), [$("inf-plugin").value]: pluginParamValues() };
  saved.slowModels = { ...(saved.slowModels || {}), [$("provider").value]: $("slow-model").value.trim() };
  store(saved);
}

/* physiology module settings form (generated from settings.json) */
function programChanged(fromUser) {
  const p = init.programs.find((x) => x.id === $("program").value);
  $("keyboard-row").classList.toggle("hidden", !(p && p.keyboard));
  $("windowed-row").classList.toggle("hidden", !(p && p.windowed));
  txt($("program-author"), p ? [p.description, p.id].filter(Boolean).join(" · ") : "модуль стимуляции запускается вручную");
  const box = $("module-settings");
  box.textContent = "";
  $("module-settings-box").classList.toggle("hidden", !(p && p.settings));
  if (!p || !p.settings) return;
  const edits = fromUser ? {} : ((saved.moduleSettings || {})[p.id] || {});
  for (const [key, def] of Object.entries(p.settings)) {
    if (key.startsWith("_")) { box.appendChild(el("p", "muted small wide", String(def))).style.gridColumn = "1 / -1"; continue; }
    const value = key in edits ? edits[key] : def;
    const help = (p.help || {})[key] || key;
    let input;
    if (typeof def === "boolean") {
      const lab = el("label", "check"); input = el("input"); input.type = "checkbox"; input.checked = !!value;
      lab.append(input, document.createTextNode(" " + help)); lab.title = key; input.dataset.key = key; input.dataset.kind = "bool";
      box.appendChild(lab); continue;
    }
    const lab = el("label"); lab.title = key; lab.append(document.createTextNode(help));
    if (Array.isArray(def) && def.length === 3 && def.every((x) => typeof x === "number")) {
      input = el("input"); input.type = "color"; input.dataset.kind = "rgb";
      input.value = "#" + value.map((x) => Math.max(0, Math.min(255, x | 0)).toString(16).padStart(2, "0")).join("");
    } else if (typeof def === "number") {
      input = el("input"); input.type = "number"; input.step = Number.isInteger(def) ? "1" : "any";
      input.value = value; input.dataset.kind = Number.isInteger(def) ? "int" : "num";
    } else if (Array.isArray(def)) {
      input = el("input"); input.value = JSON.stringify(value); input.dataset.kind = "json";
    } else {
      input = el("input"); input.value = value; input.dataset.kind = "str";
    }
    input.dataset.key = key;
    lab.appendChild(input);
    box.appendChild(lab);
  }
}
function moduleSettingsValues() {
  const p = init.programs.find((x) => x.id === $("program").value);
  if (!p || !p.settings) return null;
  const out = {};
  for (const input of $("module-settings").querySelectorAll("input[data-key]")) {
    const k = input.dataset.key, kind = input.dataset.kind;
    let v;
    if (kind === "bool") v = input.checked;
    else if (kind === "int") v = parseInt(input.value, 10);
    else if (kind === "num") v = parseFloat(input.value);
    else if (kind === "rgb") v = [1, 3, 5].map((i) => parseInt(input.value.slice(i, i + 2), 16));
    else if (kind === "json") { try { v = JSON.parse(input.value); } catch (e) { throw new Error(`параметр ${k}: неверный список`); } }
    else v = input.value;
    if (JSON.stringify(v) !== JSON.stringify(p.settings[k])) out[k] = v;
  }
  return out;
}

/* keyboard */
function keyboardSpec() {
  const preset = $("kb-preset").value;
  if (preset === "custom") return { preset, symbols: [...$("kb-symbols").value], rows: +$("kb-rows").value, cols: +$("kb-cols").value };
  return { preset };
}
function keyboardChanged() {
  const preset = $("kb-preset").value;
  $("kb-custom").classList.toggle("hidden", preset !== "custom");
  let symbols, rows, cols;
  if (preset === "custom") { symbols = [...$("kb-symbols").value]; rows = +$("kb-rows").value; cols = +$("kb-cols").value; }
  else ({ symbols, layout: [rows, cols] } = init.keyboards[preset]);
  const box = $("kb-preview");
  box.textContent = "";
  box.style.gridTemplateColumns = `repeat(${Math.max(1, cols)}, 26px)`;
  for (const s of symbols) box.appendChild(el("span", null, shown(s)));
}

/* fast LLM */
function ngramInfo() {
  const m = init.fast_models.find((x) => x.file === $("ngram-file").value);
  txt($("ngram-info"), m ? `порядок ${m.order} · обучена на ${Math.round(m.train_chars / 1e6)} млн символов (${m.sources.join(", ")})` +
    (m.bpc ? ` · ${m.bpc} бит/символ на заданиях` : "") : "нет моделей в models/ngram");
}

/* inference plugin params */
function inferenceChanged(fromUser) {
  const p = init.inference.find((x) => x.id === $("inf-plugin").value);
  txt($("inf-doc"), p ? (p.error || "") : "");
  const box = $("inf-params");
  box.textContent = "";
  if (!p) return;
  const edits = fromUser ? {} : ((saved.pluginParams || {})[p.id] || {});
  for (const [key, spec] of Object.entries(p.params || {})) {
    const lab = el("label"); lab.append(document.createTextNode(spec.label || key));
    const input = el("input"); input.type = "number"; input.dataset.key = key;
    if (spec.min != null) input.min = spec.min;
    if (spec.max != null) input.max = spec.max;
    input.step = spec.step || "any";
    input.value = key in edits ? edits[key] : spec.default;
    lab.appendChild(input);
    box.appendChild(lab);
  }
}
function pluginParamValues() {
  const out = {};
  for (const input of $("inf-params").querySelectorAll("input[data-key]")) out[input.dataset.key] = parseFloat(input.value);
  return out;
}

/* slow LLM provider */
function providerChanged(fromUser) {
  const p = init.providers.find((x) => x.id === $("provider").value);
  $("key-row").classList.toggle("hidden", !p.needs_key);
  $("base-url-row").classList.toggle("hidden", p.id !== "custom");
  $("remote-row").classList.toggle("hidden", p.remote === false);
  $("api-key").placeholder = p.has_key ? "ключ сохранён — можно заменить" : "вставьте ключ";
  txt($("provider-help"), p.help);
  const model = (saved.slowModels || {})[p.id] || (fromUser ? "" : saved.slowModel) || p.suggested;
  $("slow-model").value = model || "";
  $("slow-models").textContent = "";
  txt($("models-info"), "");
}
async function saveKey() {
  try {
    const r = await api("/api/key", { provider: $("provider").value, key: $("api-key").value });
    init.providers = r.providers;
    $("api-key").value = "";
    providerChanged(false);
    txt($("models-info"), "ключ сохранён на этом компьютере");
  } catch (e) { txt($("models-info"), e.message); }
}
async function refreshModels() {
  txt($("models-info"), "загружаю список…");
  try {
    const r = await api("/api/models", { provider: $("provider").value, base_url: $("base-url").value.trim() });
    const dl = $("slow-models"); dl.textContent = "";
    for (const m of r.models) dl.appendChild(new Option(m, m));
    txt($("models-info"), r.models.length ? `моделей: ${r.models.length} (${$("provider").value === "openrouter" ? "только бесплатные" : "все"}) — выберите в поле «модель»`
      : "список пуст");
  } catch (e) { txt($("models-info"), e.message); }
}

/* prompt editor */
async function openPrompt() {
  const r = await api("/api/init");
  init.prompt = r.prompt;
  $("prompt-text").value = init.prompt.current;
  txt($("prompt-error"), "");
  $("prompt-modal").classList.remove("hidden");
  previewPrompt();
}
async function previewPrompt() {
  try {
    const r = await api("/api/prompt", { action: "preview", text: $("prompt-text").value });
    const box = $("prompt-preview"); box.textContent = "";
    const user = r.messages.find((m) => m.role === "user"), system = r.messages.find((m) => m.role === "system");
    box.appendChild(el("div", "role", "сообщение (user)"));
    box.appendChild(el("pre", null, user ? user.content : ""));
    const det = el("details"); det.appendChild(el("summary", null, "системная часть (system)"));
    det.appendChild(el("pre", null, system ? system.content : "")); box.appendChild(det);
    txt($("prompt-error"), "");
  } catch (e) { txt($("prompt-error"), e.message); }
}
async function savePrompt() {
  try {
    const r = await api("/api/prompt", { action: "save", text: $("prompt-text").value });
    txt($("prompt-edited"), r.edited ? "(изменён)" : "(стандартный)");
    $("prompt-modal").classList.add("hidden");
  } catch (e) { txt($("prompt-error"), e.message); }
}
async function resetPrompt() {
  const r = await api("/api/prompt", { action: "reset" });
  $("prompt-text").value = r.current;
  txt($("prompt-edited"), "(стандартный)");
  previewPrompt();
}

/* start */
async function start() {
  txt($("start-error"), "");
  let body;
  try {
    collectSaved();
    const v = $("task").value;
    body = {
      participant_id: $("participant").value.trim(),
      task: v === "free" ? { kind: "free" } : v === "custom" ? { kind: "custom", text: $("custom").value } : { kind: "task", id: v },
      llm_on: sLlm.get() === "on", cue: $("cue").checked,
      physiology: { program: $("program").value, settings: moduleSettingsValues(), keyboard: $("space-eeg").checked,
        windowed: $("windowed").checked, args: $("program-args").value },
      keyboard: keyboardSpec(),
      fast: { backend: sFast.get(), model_file: $("ngram-file").value, model_id: $("hf-model").value.trim(),
        allow_download: sFast.get() === "kenlm" ? $("kenlm-download").checked : $("hf-download").checked,
        device: $("hf-device").value, context_source: sCtx.get(),
        kenlm_id: $("kenlm-id").value.trim(), char_mix: +$("kenlm-mix").value, vocab_limit: +$("kenlm-vocab").value },
      inference: { plugin: $("inf-plugin").value, params: pluginParamValues() },
      stop: { confidence: $("stop-off").checked ? null : +$("stop-conf").value, min_updates: +$("stop-min").value },
      pause_ms: +$("pause-ms").value,
      slow: { backend: sSlow.get(), provider: $("provider").value, base_url: $("base-url").value.trim(),
        model_id: $("slow-model").value.trim(), allow_remote: $("allow-remote").checked,
        every_n_words: +$("every-n").value, window_words: +$("window-w").value,
        include_alternatives: $("alts").checked, alternatives_k: +$("alts-k").value },
    };
  } catch (e) { txt($("start-error"), e.message); return; }
  $("start-btn").disabled = true;
  txt($("start-note"), "загружаю модели и запускаю модуль…");
  try { await api("/api/start", body); dismissed = null; }
  catch (e) { txt($("start-error"), e.message); }
  finally { $("start-btn").disabled = false; txt($("start-note"), ""); }
}

/* ---------------- live view ---------------- */
async function poll() {
  for (;;) {
    try {
      const data = await api(`/api/live?since=${version}&timeout=10`);
      version = data.version;
      current = data.session;
      if (data.loading) txt($("start-note"), data.loading + "…");
      render(current && current.session_id !== dismissed ? current : null);
    } catch (e) {
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
}

function mmss(s) { s = Math.round(s || 0); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }

function renderTyped(s) {
  const box = $("typed"); box.textContent = "";
  const typed = s.text || "", target = s.target || "";
  const add = (ch, cls) => { const sp = el("span", cls + (ch === " " ? " sp" : ""), ch === " " && cls === "bad" ? "_" : ch); box.appendChild(sp); };
  if (!target) {
    for (const ch of typed) add(ch, "ok");
    box.appendChild(el("span", "caret"));
    return;
  }
  for (let i = 0; i < Math.max(target.length, typed.length); i++) {
    if (i === typed.length && !s.finished) box.appendChild(el("span", "caret"));
    if (i < typed.length && i < target.length) add(typed[i], typed[i] === target[i] ? "ok" : "bad");
    else if (i < typed.length) add(typed[i], "extra");
    else add(target[i], "");
  }
  if (typed.length >= target.length && !s.finished) box.appendChild(el("span", "caret"));
}

function renderCorrected(s) {
  const show = s.corrected && s.corrected !== s.text;
  $("corrected-row").classList.toggle("hidden", !show);
  if (!show) return;
  const box = $("corrected"); box.textContent = "";
  const a = s.text.split(" "), b = s.corrected.split(" ");
  b.forEach((w, i) => { box.appendChild(el("span", w !== a[i] ? "changed" : "", w)); if (i < b.length - 1) box.appendChild(document.createTextNode(" ")); });
}

function renderQuad(id, items) {
  const box = $(id); box.textContent = "";
  for (let i = 0; i < 4; i++) {
    const it = items && items[i];
    const cell = el("div", "cell" + (i === 0 && it ? " first" : "") + (it ? "" : " empty"));
    cell.appendChild(el("div", "sym", it ? shown(it.symbol) : "·"));
    cell.appendChild(el("div", "p", it ? pct(it.p) : ""));
    box.appendChild(cell);
  }
}

function renderHistory(list) {
  const box = $("history-list"); box.textContent = "";
  const items = [...(list || [])].reverse();
  if (!items.length) { box.appendChild(el("div", "muted small", "медленная БЯМ ещё ничего не исправляла")); return; }
  const open = $("history").classList.contains("open");
  items.forEach((c, i) => {
    const msg = el("div", "msg" + (c.accepted ? "" : " rejected"));
    if (!open) msg.style.opacity = Math.max(0.12, 1 - i * 0.28);
    c.words_in.forEach((w, j) => {
      const out = c.words_out[j];
      if (c.accepted && out !== undefined && out !== w) { msg.appendChild(el("span", "from", w)); msg.appendChild(document.createTextNode(" → ")); msg.appendChild(el("span", "to", out)); }
      else msg.appendChild(el("span", "same", w));
      if (j < c.words_in.length - 1) msg.appendChild(document.createTextNode("  "));
    });
    const meta = c.accepted ? (c.changed ? "" : "без изменений · ") : `отклонено: ${c.reason || "?"} · `;
    msg.appendChild(el("span", "meta", `${meta}${c.latency_ms} мс · ${mmss(c.t_s)}`));
    box.appendChild(msg);
  });
}

function render(s) {
  $("setup").classList.toggle("hidden", !!s);
  $("run").classList.toggle("hidden", !s);
  if (!s) { txt($("phase-pill"), "настройка"); $("phase-pill").className = "pill"; return; }
  txt($("phase-pill"), s.paused ? "пауза" : (PHASES[s.phase] || s.phase));
  $("phase-pill").className = "pill" + (s.phase === "waiting_modules" || s.paused ? " off" : "") + (s.error ? " warn" : "");
  const mods = (s.modules && s.modules.modules) || [];
  const listening = s.modules && s.modules.control_listeners;
  txt($("modules-pill"), mods.length ? `модули: ${mods.length}` : (listening ? "модуль слушает" : "модули: нет"));
  $("modules-pill").className = "pill" + (mods.length || listening ? "" : " off");

  const pp = s.passport || {};
  const pass = $("passport"); pass.textContent = "";
  const items = [["участник", pp.participant || s.participant_id], ["задание", pp.task || s.task_id],
    ["физиология", pp.physiology || "вручную"], ["клавиатура", pp.keyboard], ["быстрая", pp.fast],
    ["вывод", pp.inference], ["медленная", pp.slow], ["время", mmss(s.elapsed_s)], ["эпоха", s.epoch_id]];
  for (const [k, v] of items) { if (v == null) continue; const sp = el("span", null, k + " "); sp.appendChild(el("b", null, String(v))); pass.appendChild(sp); }

  renderTyped(s);
  renderCorrected(s);
  const e = s.finished ? null : s.epoch;
  const cands = $("cands"); cands.textContent = "";
  for (const w of (e && e.words) || []) cands.appendChild(el("span", null, w));
  renderQuad("quad-post", e && e.posterior_top);
  renderQuad("quad-lik", e && e.likelihood_top);
  const lead = e && e.leader;
  $("meter-fill").style.width = lead ? `${Math.min(100, lead.p * 100)}%` : "0";
  $("meter-stop").classList.toggle("hidden", s.stop_confidence == null);
  if (s.stop_confidence != null) $("meter-stop").style.left = `${s.stop_confidence * 100}%`;
  txt($("meter-label"), lead ? `${shown(lead.symbol)} ${pct(lead.p)}${e.stop_sent ? " · стоп" : ""}` : (e ? "ждём ЭЭГ" : ""));
  txt($("f-epoch"), e ? e.epoch_id + (e.target ? ` · цель «${shown(e.target)}»` : "") : s.epoch_id);
  txt($("f-flashes"), e ? e.flashes : "—");
  txt($("f-updates"), e ? e.updates : "—");
  txt($("f-elapsed"), e ? `${e.elapsed_s} с` : "—");
  const st = s.stats;
  txt($("s-acc"), st.accuracy == null ? "—" : pct(st.accuracy));
  txt($("s-speed"), st.speed_cpm == null ? "—" : `${st.speed_cpm} симв/мин`);
  txt($("s-early"), st.early_stops);
  txt($("s-epoch"), st.mean_epoch_s == null ? "—" : `${st.mean_epoch_s} с`);
  const pt = $("prior-top"); pt.textContent = "";
  for (const it of (e && e.prior_top) || []) pt.appendChild(el("span", null, `${shown(it.symbol)} ${pct(it.p)}  `));
  const prog = s.stimulus_program;
  const ps = $("program-status");
  ps.className = "muted small" + (prog && !prog.running && prog.returncode ? " bad" : "");
  txt(ps, prog ? (prog.running ? `программа ${prog.program} работает` : `программа ${prog.program} завершилась (код ${prog.returncode})`) : "");

  renderHistory(s.corrections);
  $("pause-btn").disabled = $("stop-btn").disabled = s.finished;
  txt($("pause-btn"), s.paused ? "продолжить" : "пауза");
  $("new-btn").classList.toggle("hidden", !s.finished);
  txt($("end-note"), s.finished ? `завершено: ${s.end_reason}${s.error ? " — " + s.error : ""}` : (st.pending_corrections ? `в очереди исправлений: ${st.pending_corrections}` : ""));

  const ul = $("modules"); ul.textContent = "";
  for (const m of mods) ul.appendChild(el("li", null, `${m.module} (${m.kind === "likelihood" ? "декодер" : "стимуляция"}) — сообщений ${m.messages}` +
    (m.last_message_s == null ? "" : `, последнее ${m.last_message_s} с назад`)));
  if (!mods.length) ul.appendChild(el("li", "muted", "пока ни один модуль не прислал поток"));
  const log = $("log"); log.textContent = "";
  for (const r of [...s.log].reverse()) {
    const tr = el("tr");
    const cells = [r.epoch_id, r.target ? shown(r.target) : "", r.decided == null ? "—" : shown(r.decided),
      r.confidence == null ? "" : r.confidence.toFixed(2), r.flashes, r.duration_s, (r.stopped_early ? "досрочно, " : "") + r.decided_on];
    cells.forEach((v, i) => { const td = el("td", i === 2 && r.correct === false ? "bad" : i === 2 && r.correct ? "ok" : "", v); tr.appendChild(td); });
    log.appendChild(tr);
  }
  txt($("program-log"), prog ? (prog.log_tail || []).join("\n") : "");
  txt($("dir"), s.directory);
}

boot();
