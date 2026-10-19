"use strict";
const $ = (id) => document.getElementById(id);
const PHASES = { waiting_modules: "ждём модуль", presenting: "эпоха", result: "результат", finished: "завершено" };
const STORE = "speller.operator.v2";
const DECIDED = { final_likelihood: "оценка final", stop_grace_elapsed: "после стопа", done_grace_elapsed: "после done",
  timeout: "по таймауту", no_likelihood: "нет ЭЭГ — повтор" };
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

  // theme
  $("theme-btn").onclick = () => setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light", true);

  // physiology
  for (const p of init.programs) $("program").add(new Option(p.label, p.id));
  $("program").add(new Option("— не запускать (модуль запущу сам)", ""));
  $("program").onchange = programChanged;

  // fast LLM
  sLlm = seg("llm-mode", (v) => { for (const b of ["b-fast", "b-slow"]) $(b).style.opacity = v === "off" ? .45 : 1; });
  sFast = seg("fast-backend", (v) => {
    $("fast-ngram").classList.toggle("hidden", v !== "ngram");
    $("fast-hf").classList.toggle("hidden", v !== "hf_causal");
    $("fast-kenlm").classList.toggle("hidden", v !== "kenlm");
  });
  sCtx = seg("fast-context");
  for (const m of init.fast_models) $("ngram-file").add(new Option(m.file, m.file));

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
    if (current) renderHistory(current);
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
  const prog = init.programs.find((p) => p.id === s.program) ||
    init.programs.find((p) => p.id === "modules/flash_rows_cols") || init.programs[0];
  $("program").value = s.program === "" ? "" : (prog ? prog.id : "");
  programChanged();
  const kenlmReady = init.kenlm.available && init.kenlm.downloaded.length > 0;
  sFast.set(s.fastBackend || (kenlmReady ? "kenlm" : "ngram"));
  if (s.ngramFile && init.fast_models.some((m) => m.file === s.ngramFile)) $("ngram-file").value = s.ngramFile;
  $("hf-model").value = s.hfModel || "";
  $("kenlm-id").value = s.kenlmId || init.kenlm.default;
  $("kenlm-download").checked = !!s.kenlmDownload;
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
  $("every-n").value = s.everyN ?? d.every_n_words;
  $("window-w").value = s.windowW ?? d.window_words;
  $("alts").checked = s.alts ?? true;
  $("alts-k").value = s.altsK ?? d.alternatives_k;
  txt($("prompt-edited"), init.prompt.edited ? "(изменён)" : "(стандартный)");
}

function collectSaved() {
  saved = {
    ...saved, participant: $("participant").value.trim(), task: $("task").value, custom: $("custom").value,
    llm: sLlm.get(), program: $("program").value,
    fastBackend: sFast.get(), ngramFile: $("ngram-file").value, hfModel: $("hf-model").value.trim(),
    hfDownload: $("hf-download").checked, hfDevice: $("hf-device").value, context: sCtx.get(),
    kenlmId: $("kenlm-id").value.trim(), kenlmDownload: $("kenlm-download").checked,
    plugin: $("inf-plugin").value, stopConf: +$("stop-conf").value, stopMin: +$("stop-min").value,
    pauseMs: +$("pause-ms").value, stopOff: $("stop-off").checked, slowBackend: sSlow.get(),
    provider: $("provider").value, baseUrl: $("base-url").value.trim(), slowModel: $("slow-model").value.trim(),
    everyN: +$("every-n").value, windowW: +$("window-w").value,
    alts: $("alts").checked, altsK: +$("alts-k").value,
  };
  saved.pluginParams = { ...(saved.pluginParams || {}), [$("inf-plugin").value]: pluginParamValues() };
  saved.slowModels = { ...(saved.slowModels || {}), [$("provider").value]: $("slow-model").value.trim() };
  store(saved);
}

/* physiology: the module owns its settings and keyboard */
function programChanged() {
  const p = init.programs.find((x) => x.id === $("program").value);
  txt($("program-about"), p ? p.description || "" : "");
}

/* inference plugin params */
function inferenceChanged(fromUser) {
  const p = init.inference.find((x) => x.id === $("inf-plugin").value);
  txt($("inf-error"), p ? (p.error || "") : "");
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
  $("api-key").placeholder = p.has_key ? "ключ сохранён — можно заменить" : "вставьте ключ";
  // A model name belongs to its provider: never carry one over to another provider.
  const model = (saved.slowModels || {})[p.id] || (saved.provider === p.id ? saved.slowModel : "") || p.suggested;
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
      llm_on: sLlm.get() === "on",
      physiology: { program: $("program").value },
      fast: { backend: sFast.get(), model_file: $("ngram-file").value, model_id: $("hf-model").value.trim(),
        allow_download: sFast.get() === "kenlm" ? $("kenlm-download").checked : $("hf-download").checked,
        device: $("hf-device").value, context_source: sCtx.get(), kenlm_id: $("kenlm-id").value.trim() },
      inference: { plugin: $("inf-plugin").value, params: pluginParamValues() },
      stop: { confidence: $("stop-off").checked ? null : +$("stop-conf").value, min_updates: +$("stop-min").value },
      pause_ms: +$("pause-ms").value,
      slow: { backend: sSlow.get(), provider: $("provider").value, base_url: $("base-url").value.trim(),
        model_id: $("slow-model").value.trim(),
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

function renderHistory(s) {
  const box = $("history-list"); box.textContent = "";
  const byEpoch = {};
  for (const c of s.corrections || []) (byEpoch[c.epoch_id] = byEpoch[c.epoch_id] || []).push(c);
  const rows = [...(s.log || [])].reverse();
  const open = $("history").classList.contains("open");
  $("history-toggle").classList.toggle("hidden", rows.length <= 6);
  if (!rows.length) {
    const tr = el("tr"); const td = el("td", "muted", "эпох ещё не было"); td.colSpan = 9; tr.appendChild(td); box.appendChild(tr);
    return;
  }
  rows.slice(0, open ? rows.length : 6).forEach((r, i) => {
    const tr = el("tr");
    if (!open) tr.style.opacity = Math.max(0.15, 1 - i * 0.17);
    tr.appendChild(el("td", "muted", r.epoch_id));
    tr.appendChild(el("td", null, r.target ? shown(r.target) : "—"));
    const pick = r.decided == null ? "—" : r.typed ? shown(r.typed) : r.decided;
    tr.appendChild(el("td", "pick" + (r.correct === false ? " bad" : r.correct ? " ok" : ""), pick));
    tr.appendChild(el("td", r.uncertain ? "unsure" : null, r.confidence == null ? "" : r.confidence.toFixed(2)));
    const alts = el("td", "alts");
    for (const [sym, p] of r.top || []) {
      const sp = el("span", sym === r.decided ? "on" : null, sym === "_" ? "␣" : shown(sym));
      sp.appendChild(el("i", null, Math.round(p * 100)));
      alts.appendChild(sp);
    }
    tr.appendChild(alts);
    tr.appendChild(el("td", "muted", r.flashes));
    tr.appendChild(el("td", "muted", r.duration_s));
    tr.appendChild(el("td", "muted", (r.stopped_early ? "досрочно · " : "") + (DECIDED[r.decided_on] || r.decided_on)));
    const fix = el("td", "fix");
    for (const c of byEpoch[r.epoch_id] || []) {
      const line = el("div", c.accepted ? (c.changed ? "" : "same") : "rejected");
      if (!c.accepted) line.textContent = `отклонено: ${c.reason || "?"}`;
      else if (!c.changed) line.textContent = "без изменений";
      else c.words_in.forEach((w, j) => {
        const out = c.words_out[j];
        if (out !== w) { line.appendChild(el("s", null, w)); line.appendChild(document.createTextNode("→")); line.appendChild(el("b", null, out)); line.appendChild(document.createTextNode(" ")); }
      });
      line.title = `${c.words_in.join(" ")} → ${c.words_out.join(" ")} · ${c.latency_ms} мс`;
      fix.appendChild(line);
    }
    tr.appendChild(fix);
    box.appendChild(tr);
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

  renderHistory(s);
  $("pause-btn").disabled = $("stop-btn").disabled = s.finished;
  txt($("pause-btn"), s.paused ? "продолжить" : "пауза");
  $("new-btn").classList.toggle("hidden", !s.finished);
  txt($("end-note"), s.finished ? `завершено: ${s.end_reason}${s.error ? " — " + s.error : ""}` : (st.pending_corrections ? `в очереди исправлений: ${st.pending_corrections}` : ""));

  const crashed = !!(prog && !prog.running && prog.returncode);
  $("program-log").classList.toggle("hidden", !crashed);
  txt($("program-log"), crashed ? (prog.log_tail || []).join("\n") : "");
  txt($("dir"), s.directory);
}

/* theme: dark or light; the system setting until the operator picks one */
function setTheme(theme, remember) {
  document.documentElement.dataset.theme = theme;
  if (remember) { saved.theme = theme; store(saved); }
}
setTheme(load().theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"), false);

boot();
