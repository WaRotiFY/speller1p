"use strict";
const $ = (id) => document.getElementById(id);
const PHASES = { waiting_modules: "ждём модуль стимуляции", presenting: "идёт эпоха", result: "показ результата",
  finished: "сессия завершена" };
let version = -1, current = null, dismissed = null;

async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) };
  const r = await fetch(path, opts);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

function shown(sym) { return sym === " " ? "␣" : sym; }
function text(el, value) { el.textContent = value; }

async function init() {
  const data = await api("/api/init");
  for (const p of data.profiles) $("profile").add(new Option(p.replace(/\.yaml$/, ""), p));
  for (const t of data.tasks) $("task").add(new Option(`${t.id} — ${t.text}`, t.id));
  $("task").onchange = () => $("custom-row").classList.toggle("hidden", $("task").value !== "custom");
  $("start-btn").onclick = start;
  $("pause-btn").onclick = () => api("/api/pause", { paused: !(current && current.paused) }).catch(alert);
  $("stop-btn").onclick = () => confirm("Завершить сессию?") && api("/api/stop", {}).catch(alert);
  $("new-btn").onclick = () => { dismissed = current && current.session_id; render(null); };
  poll();
}

async function start() {
  const v = $("task").value;
  const task = v === "free" ? { kind: "free" } : v === "custom" ? { kind: "custom", text: $("custom").value }
    : { kind: "task", id: v };
  text($("start-error"), "");
  try {
    await api("/api/start", { profile: $("profile").value, participant_id: $("participant").value.trim(), task });
  } catch (e) { text($("start-error"), e.message); }
}

async function poll() {
  for (;;) {
    try {
      const data = await api(`/api/live?since=${version}&timeout=10`);
      version = data.version;
      current = data.session;
      render(current && current.session_id === dismissed ? null : current);
    } catch (e) {
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
}

function render(s) {
  const running = !!s;
  $("start").classList.toggle("hidden", running);
  $("run").classList.toggle("hidden", !running);
  if (!running) { text($("phase-pill"), "нет сессии"); return; }

  text($("phase-pill"), s.paused ? "пауза" : (PHASES[s.phase] || s.phase));
  $("phase-pill").className = "pill" + (s.phase === "waiting_modules" || s.paused ? " warn" : "");
  const mods = (s.modules && s.modules.modules) || [];
  const listening = s.modules && s.modules.control_listeners;
  text($("modules-pill"), mods.length ? `модули: ${mods.length}` : (listening ? "модуль слушает" : "модули: нет"));
  $("modules-pill").className = "pill" + (mods.length || listening ? "" : " off");

  text($("typed"), s.text);
  $("corrected-row").classList.toggle("hidden", !s.corrected || s.corrected === s.text);
  text($("corrected"), s.corrected);
  $("target-row").classList.toggle("hidden", !s.target);
  const t = $("target"); t.textContent = "";
  [...(s.target || "")].forEach((ch, i) => {
    const span = document.createElement("span");
    span.textContent = ch;
    span.className = i < s.target_index ? "done" : i === s.target_index ? "next" : "";
    t.appendChild(span);
  });

  const e = s.finished ? null : s.epoch;
  document.querySelector(".now").classList.toggle("hidden", s.finished);
  text($("epoch-no"), e ? `· эпоха ${e.epoch_id}${e.target ? " · цель «" + shown(e.target) + "»" : ""}` : "");
  const lead = e && (e.leader || null);
  text($("leader"), lead ? shown(lead.symbol) : "—");
  text($("leader-p"), lead ? `${Math.round(lead.p * 100)} %` : (e ? "ждём данных ЭЭГ" : ""));
  $("meter-fill").style.width = lead ? `${Math.min(100, lead.p * 100)}%` : "0";
  const stop = s.stop_confidence;
  $("meter-stop").classList.toggle("hidden", stop == null);
  if (stop != null) $("meter-stop").style.left = `${stop * 100}%`;
  text($("flashes"), e ? `вспышек: ${e.flashes}` : "");
  text($("elapsed"), e ? `${e.elapsed_s} с` : "");
  $("stop-badge").classList.toggle("hidden", !(e && e.stop_sent));

  const st = s.stats;
  text($("s-selected"), st.selected);
  text($("s-acc"), st.accuracy == null ? "—" : `${Math.round(st.accuracy * 100)} %`);
  text($("s-early"), st.early_stops);
  text($("s-epoch"), st.mean_epoch_s == null ? "—" : `${st.mean_epoch_s} с`);
  text($("s-speed"), st.speed_cpm == null ? "—" : `${st.speed_cpm} симв/мин`);

  $("pause-btn").disabled = $("stop-btn").disabled = s.finished;
  text($("pause-btn"), s.paused ? "Продолжить" : "Пауза");
  $("new-btn").classList.toggle("hidden", !s.finished);
  text($("end-note"), s.finished ? `Завершено: ${s.end_reason}${s.error ? " — " + s.error : ""}` : "");

  const ul = $("modules"); ul.textContent = "";
  for (const m of mods) {
    const li = document.createElement("li");
    li.textContent = `${m.module} (${m.kind === "likelihood" ? "декодер" : "стимуляция"}) — сообщений ${m.messages}` +
      (m.last_message_s == null ? "" : `, последнее ${m.last_message_s} с назад`);
    ul.appendChild(li);
  }
  if (!mods.length) ul.textContent = "Пока ни один модуль не прислал поток.";
  const top = $("top"); top.textContent = "";
  for (const c of (e && e.top) || []) {
    const span = document.createElement("span");
    span.textContent = `${shown(c.symbol)} ${Math.round(c.p * 100)}%`;
    top.appendChild(span);
  }
  text($("words"), e && e.words && e.words.length ? "Слова: " + e.words.join(", ") : "");
  const log = $("log"); log.textContent = "";
  for (const r of [...s.log].reverse()) {
    const tr = document.createElement("tr");
    const cells = [r.epoch_id, r.target ? shown(r.target) : "", r.decided == null ? "—" : shown(r.decided),
      r.confidence == null ? "" : r.confidence.toFixed(2), r.flashes, r.duration_s,
      (r.stopped_early ? "досрочно, " : "") + r.decided_on];
    cells.forEach((v, i) => {
      const td = document.createElement("td");
      td.textContent = v;
      if (i === 2 && r.correct === false) td.className = "bad";
      if (i === 2 && r.correct === true) td.className = "ok";
      tr.appendChild(td);
    });
    log.appendChild(tr);
  }
  const cl = $("corrections"); cl.textContent = "";
  for (const c of [...s.corrections].reverse()) {
    const li = document.createElement("li");
    li.textContent = `${c.words_in.join(" ")} → ${c.words_out.join(" ")}${c.changed ? "" : " (без изменений)"}`;
    cl.appendChild(li);
  }
  text($("dir"), s.directory);
}

init();
