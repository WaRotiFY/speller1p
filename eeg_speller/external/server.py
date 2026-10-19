"""Loopback-only operator window for external stimulus/decoder modules.

    python -m eeg_speller.external.server --port 8770
    open http://127.0.0.1:8770/

Before the start the operator sets up the session block by block: physiology (one of the
modules in stimulus_kit/modules; the module owns its settings and keyboard), fast LLM,
statistical inference (a plugin from inference/) and slow LLM with its prompt. The
participant's screen is drawn by the chosen module, not here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit

from eeg_speller.core.config import Config, _merge, load_config
from eeg_speller.epoch.plugins import list_plugins
from eeg_speller.external.controller import ExternalSession, default_model_factory
from eeg_speller.external.launcher import StimulusProcess, build_args, stimulus_programs
from eeg_speller.llm.backends import chat_api
from eeg_speller.llm.backends import kenlm_words
from eeg_speller.llm.backends.char_ngram import available_models
from eeg_speller.external.tasks import check_identifier, read_task_list, resolve_task

PROJECT = Path(__file__).resolve().parents[2]
CONFIGS = PROJECT / "configs"
BASE_PROFILE = CONFIGS / "ext_default.yaml"
STATIC = Path(__file__).resolve().parent / "static"
FILES = {"/": ("operator.html", "text/html; charset=utf-8"),
         "/operator.js": ("operator.js", "application/javascript; charset=utf-8"),
         "/operator.css": ("operator.css", "text/css; charset=utf-8")}
MAX_BODY = 65536


def _num(value, low, high, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: a number")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return int(value) if integer else float(value)


EXTRAS = {"kenlm": "kenlm", "torch": "hf", "transformers": "hf", "pylsl": "", "pygame": "", "pyedflib": ""}


def install_hint(exc: ImportError) -> str:
    """A missing package -> the exact command that installs it."""
    name = (getattr(exc, "name", None) or "").split(".")[0]
    extra = EXTRAS.get(name)
    if extra is None:
        return f"не установлен пакет {name or exc}: pip install {name}".strip()
    return f'не установлен пакет {name}: uv pip install -e ".[{extra}]"' if extra else \
        f"не установлен пакет {name}: uv pip install -e ."


def build_session_config(body: dict) -> tuple[Config, dict]:
    """Operator form -> validated config + a short human-readable passport."""
    base = load_config(BASE_PROFILE).model_dump()
    patch: dict = {}
    passport: dict = {}

    llm_on = bool(body.get("llm_on", True))
    patch["mode"] = {"llm": "on" if llm_on else "off"}

    fast = body.get("fast") or {}
    backend = fast.get("backend", "ngram")
    if backend == "ngram":
        names = {m["file"] for m in available_models()}
        model_file = fast.get("model_file") or "ru_char4.npz"
        if model_file not in names:
            raise ValueError("choose an n-gram model from models/ngram")
        patch["fast_llm"] = {"backend": "ngram", "model_file": model_file, "model_id": Path(model_file).stem}
        passport["fast"] = Path(model_file).stem
    elif backend == "hf_causal":
        model_id = str(fast.get("model_id") or "").strip()
        if not model_id or len(model_id) > 200 or any(c.isspace() for c in model_id):
            raise ValueError("enter a Hugging Face model name, e.g. Qwen/Qwen2.5-0.5B")
        patch["fast_llm"] = {"backend": "hf_causal", "model_id": model_id,
                             "allow_download": bool(fast.get("allow_download")),
                             "device": fast.get("device") if fast.get("device") in ("cpu", "cuda", "mps") else "cpu"}
        passport["fast"] = model_id
    elif backend == "kenlm":
        model_id = str(fast.get("kenlm_id") or kenlm_words.DEFAULT_REPO).strip()
        if not model_id or len(model_id) > 300:
            raise ValueError("enter a Hugging Face name or a path to a KenLM file")
        patch["fast_llm"] = {"backend": "kenlm", "model_id": model_id,
                             "allow_download": bool(fast.get("allow_download")),
                             "char_mix": 0.1, "vocab_limit": 60000}
        passport["fast"] = f"KenLM {model_id.split('/')[-1]}"
    elif backend == "mock_ngram":
        patch["fast_llm"] = {"backend": "mock_ngram", "model_id": "mock-ngram-5"}
        passport["fast"] = "mock-ngram-5"
    else:
        raise ValueError("unknown fast LLM backend")
    if fast.get("context_source") in ("primary", "secondary"):
        patch["fast_llm"]["context_source"] = fast["context_source"]

    inf = body.get("inference") or {}
    plugins = {p["id"]: p for p in list_plugins()}
    plugin = inf.get("plugin", "log_pool")
    if plugin not in plugins or plugins[plugin]["error"]:
        raise ValueError("choose a working inference plugin from inference/")
    params = {k: _num(v, -1e9, 1e9, k) for k, v in (inf.get("params") or {}).items()
              if k in plugins[plugin]["params"]}
    patch["fusion"] = {"plugin": plugin, "params": params}
    passport["inference"] = plugin

    stop = body.get("stop") or {}
    ext = {}
    if "confidence" in stop:
        ext["stop_confidence"] = None if stop["confidence"] in (None, "", 0) else _num(stop["confidence"], 0.5, 1.0, "stop confidence")
    if "min_updates" in stop:
        ext["min_updates_before_stop"] = _num(stop["min_updates"], 0, 1000, "min updates", integer=True)
    patch["external"] = ext
    if "pause_ms" in body:
        patch["stimulus"] = {"pause_ms": _num(body["pause_ms"], 500, 5000, "pause", integer=True) // 10 * 10}

    slow = body.get("slow") or {}
    sb = slow.get("backend", "mock_dictionary")
    s_patch = {"every_n_words": _num(slow.get("every_n_words", 1), 1, 20, "every N words", integer=True),
               "window_words": _num(slow.get("window_words", 3), 1, 20, "window", integer=True),
               "include_alternatives": bool(slow.get("include_alternatives", True)),
               "alternatives_k": _num(slow.get("alternatives_k", 4), 1, 10, "alternatives K", integer=True)}
    allow_remote = False
    if sb == "off":
        s_patch["backend"] = "off"
        passport["slow"] = "выкл."
    elif sb == "mock_dictionary":
        s_patch.update(backend="mock_dictionary", model_id="mock-dictionary-1")
        passport["slow"] = "словарь (mock)"
    elif sb == "chat_api":
        provider = slow.get("provider", "ollama")
        url = chat_api.base_url_for(provider, slow.get("base_url") if provider == "custom" else None)
        allow_remote = chat_api.is_remote(url)      # choosing a cloud provider is the consent (Q-41)
        model_id = str(slow.get("model_id") or "").strip()
        if not model_id:
            raise ValueError("choose a model for the slow LLM")
        prompt = chat_api.read_prompt()
        s_patch.update(backend="chat_api", provider=provider, base_url=url, model_id=model_id,
                       prompt_text=prompt, timeout_s=_num(slow.get("timeout_s", 30), 2, 120, "timeout"))
        passport["slow"] = f"{provider}: {model_id}"
    else:
        raise ValueError("unknown slow LLM backend")
    patch["slow_llm"] = s_patch
    patch["llm"] = {"allow_remote": allow_remote}
    if not llm_on:
        passport["fast"] = passport["slow"] = "без БЯМ"
    cfg = Config.model_validate(_merge(base, patch))
    return cfg, passport


class Manager:
    def __init__(self, output: Path, transport_factory=None):
        self.output = output
        self.session: ExternalSession | None = None
        self.lock = threading.Lock()
        self.transport_factory = transport_factory
        self.models: dict[str, object] = {}
        self.stimulus: StimulusProcess | None = None
        self.loading: str | None = None

    def model(self, kind, cfg):
        if kind == "slow" and cfg.get("backend") == "chat_api":
            return default_model_factory(kind, cfg)          # cheap, and keys may change
        key = kind + json.dumps(cfg, sort_keys=True, ensure_ascii=False)
        if key not in self.models:
            self.loading = f"загрузка модели {cfg.get('model_id')}"
            try:
                self.models[key] = default_model_factory(kind, cfg)
            finally:
                self.loading = None
        return self.models[key]

    def start(self, body: dict) -> dict:
        with self.lock:
            if self.session is not None and not self.session.finished:
                raise ValueError("stop the running session first")
            task_id, target = resolve_task(body.get("task") or {"kind": "free"})
            participant = check_identifier(body.get("participant_id"), "participant_id")
            cfg, passport = build_session_config(body)
            cfg = cfg.model_copy(update={"participant_id": participant, "task_id": task_id})
            phys = body.get("physiology") or {}
            program = phys.get("program") or ""
            info = None
            if program:
                info = next((p for p in stimulus_programs() if p["id"] == program), None)
                if info is None:
                    raise ValueError("unknown stimulus program")
                passport["physiology"] = info["label"]
            passport.update(participant=participant, task=task_id)
            transport = self.transport_factory() if self.transport_factory else None
            session = ExternalSession(cfg, target, self.output, transport=transport,
                                      model_factory=self.model, profile=BASE_PROFILE.name, passport=passport)
            self.stimulus = None
            if info is not None:
                args = build_args(info, target)
                # The control stream exists already, so the program finds the speller at once.
                self.stimulus = StimulusProcess(program, args, session.storage.directory / "external" / "stimulus.log")
                session.rec.emit("Session Recorder", "command_issued",
                                 {"command": "start_stimulus_program", "text": program, "result": args,
                                  "settings": info.get("settings")})
            self.session = session.start()
            threading.Thread(target=self._stop_stimulus_when_done, args=(session, self.stimulus),
                             daemon=True).start()
            return self.monitor()

    def _stop_stimulus_when_done(self, session, stimulus):
        session.wait_finished()
        if stimulus is not None:
            stimulus.stop()

    def monitor(self) -> dict | None:
        s = self.session
        if s is None:
            return None
        state = s.monitor()
        state["stimulus_program"] = self.stimulus.status() if self.stimulus else None
        return state

    def live(self, since: int, timeout: float) -> dict:
        start, s = time.monotonic(), self.session
        while time.monotonic() - start < timeout and self.session is s:
            if s is not None and (id(s) % 10**6) * 10**6 + s.version != since:
                break
            if s is None:
                time.sleep(0.25)
            else:
                s.wait_change(s.version, 0.5)
        s = self.session
        # Versions of different sessions must never collide on the client side.
        loading = self.loading
        st = kenlm_words.download_status
        if loading and st.get("file"):
            loading += f" — скачано {st['done_mb']} из {st['total_mb']} МБ ({st['file']})"
        return {"version": (id(s) % 10**6) * 10**6 + s.version if s else 0,
                "session": self.monitor(), "loading": loading}


def init_payload() -> dict:
    base = load_config(BASE_PROFILE)
    return {"tasks": read_task_list(), "programs": stimulus_programs(),
            "fast_models": available_models(), "inference": list_plugins(),
            "kenlm": {"available": kenlm_words.kenlm_available(), "default": kenlm_words.DEFAULT_REPO,
                      "downloaded": kenlm_words.downloaded_models()},
            "providers": chat_api.provider_list(),
            "prompt": {"current": chat_api.read_prompt(), "default": chat_api.read_prompt("default"),
                       "edited": chat_api.USER_PROMPT.is_file()},
            "defaults": {"stop_confidence": base.external["stop_confidence"],
                         "min_updates": base.external["min_updates_before_stop"],
                         "pause_ms": base.stimulus["pause_ms"],
                         "every_n_words": base.slow_llm["every_n_words"],
                         "window_words": base.slow_llm["window_words"],
                         "alternatives_k": base.slow_llm["alternatives_k"]}}


def prompt_preview(text: str) -> list[dict]:
    chat_api.check_template(text)
    letters = [[[["д", .96], ["л", .02]], [["а", .91], ["я", .04]], [["й", .97], ["и", .02]]],
               [[["п", .94], ["р", .03]], [["д", .46], ["л", .41], ["ж", .06], ["б", .03]], [["е", .89], ["ё", .05]],
                [["д", .97], ["т", .02]]]]
    return chat_api.build_messages(text, "мне холодно,", ["дай", "пдед"], letters)


def make_handler(manager: Manager):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EEGSpellerExternal/0.2"

        def _send(self, status, body, ctype="application/json; charset=utf-8"):
            data = json.dumps(body, ensure_ascii=False).encode() if ctype.startswith("application/json") else body
            try:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy",
                                 "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'")
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def _loopback(self):
            return self.headers.get("Host", "").split(":", 1)[0].lower() in {"127.0.0.1", "localhost"}

        def do_GET(self):
            if not self._loopback():
                return self._send(403, {"error": "loopback host required"})
            url = urlsplit(self.path)
            q = parse_qs(url.query)
            if url.path == "/api/init":
                return self._send(200, init_payload())
            if url.path == "/api/live":
                try:
                    since = int(q.get("since", ["-1"])[0])
                    timeout = min(20.0, max(0.0, float(q.get("timeout", ["10"])[0])))
                except ValueError:
                    since, timeout = -1, 10.0
                return self._send(200, manager.live(since, timeout))
            item = FILES.get(url.path)
            if item is None:
                return self._send(404, {"error": "not found"})
            return self._send(200, (STATIC / item[0]).read_bytes(), item[1])

        def do_POST(self):
            if not self._loopback():
                return self._send(403, {"error": "loopback host required"})
            origin, host = self.headers.get("Origin"), self.headers.get("Host", "")
            if origin and origin not in {"http://" + host, "https://" + host}:
                return self._send(403, {"error": "cross-origin request rejected"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY:
                    raise ValueError("invalid request size")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("object required")
                s = manager.session
                if self.path == "/api/start":
                    return self._send(200, manager.start(body))
                if self.path == "/api/models":
                    provider = body.get("provider", "")
                    models = chat_api.list_models(provider, body.get("base_url") if provider == "custom" else None,
                                                  bool(body.get("only_free", True)))
                    return self._send(200, {"models": models})
                if self.path == "/api/key":
                    chat_api.save_key(body.get("provider", ""), str(body.get("key", ""))[:400])
                    return self._send(200, {"providers": chat_api.provider_list()})
                if self.path == "/api/prompt":
                    action = body.get("action")
                    if action == "save":
                        chat_api.save_prompt(str(body.get("text", "")))
                    elif action == "reset":
                        chat_api.reset_prompt()
                    elif action == "preview":
                        return self._send(200, {"messages": prompt_preview(str(body.get("text", "")))})
                    else:
                        raise ValueError("action must be save, reset or preview")
                    return self._send(200, {"current": chat_api.read_prompt(),
                                            "edited": chat_api.USER_PROMPT.is_file()})
                if s is None:
                    raise ValueError("no session")
                if self.path == "/api/pause":
                    s.set_paused(bool(body.get("paused")))
                    return self._send(200, manager.monitor())
                if self.path == "/api/stop":
                    s.request_stop("operator_stopped")
                    return self._send(200, manager.monitor())
                return self._send(404, {"error": "not found"})
            except ImportError as exc:
                return self._send(400, {"error": install_hint(exc)})
            except (ValueError, TypeError, KeyError, PermissionError, json.JSONDecodeError) as exc:
                return self._send(400, {"error": str(exc)})

        def log_message(self, fmt, *args):
            if not self.path.startswith("/api/"):
                super().log_message(fmt, *args)

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Operator window for the speller with external modules")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--output", type=Path, default=Path("runs/external"))
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(Manager(args.output)))
    server.daemon_threads = True
    print(f"Operator: http://127.0.0.1:{args.port}/   (participant screen = your stimulus module)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
