"""Loopback-only operator window for external stimulus/decoder modules.

    python -m eeg_speller.external.server --port 8770
    open http://127.0.0.1:8770/

The participant's screen is drawn by a separate module (stimulus_kit/), not here.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit

from eeg_speller.core.config import load_config
from eeg_speller.external.controller import ExternalSession, default_model_factory
from eeg_speller.external.launcher import StimulusProcess, build_args, stimulus_programs
from eeg_speller.web import settings
from eeg_speller.web.server import resolve_task

PROJECT = Path(__file__).resolve().parents[2]
CONFIGS = PROJECT / "configs"
STATIC = Path(__file__).resolve().parent / "static"
FILES = {"/": ("operator.html", "text/html; charset=utf-8"),
         "/operator.js": ("operator.js", "application/javascript; charset=utf-8"),
         "/operator.css": ("operator.css", "text/css; charset=utf-8")}
MAX_BODY = 16384


def profiles() -> list[str]:
    names = []
    for path in sorted(CONFIGS.glob("ext_*.yaml")):
        try:
            load_config(path)
            names.append(path.name)
        except Exception:
            continue
    return names


class Manager:
    def __init__(self, output: Path, transport_factory=None):
        self.output = output
        self.session: ExternalSession | None = None
        self.lock = threading.Lock()
        self.transport_factory = transport_factory
        self.models: dict[str, object] = {}
        self.stimulus: StimulusProcess | None = None

    def model(self, kind, cfg):
        key = kind + json.dumps(cfg, sort_keys=True, ensure_ascii=False)
        if key not in self.models:
            self.models[key] = default_model_factory(kind, cfg)
        return self.models[key]

    def start(self, body: dict) -> dict:
        with self.lock:
            if self.session is not None and not self.session.finished:
                raise ValueError("stop the running session first")
            name = body.get("profile", "")
            if name not in profiles():
                raise ValueError("unknown profile")
            task_id, target = resolve_task(body.get("task") or {"kind": "free"})
            participant = settings.check_identifier(body.get("participant_id"), "participant_id")
            cfg = load_config(CONFIGS / name).model_copy(update={"participant_id": participant,
                                                                  "task_id": task_id})
            program = body.get("stimulus") or ""
            info = None
            if program:
                info = next((p for p in stimulus_programs() if p["id"] == program), None)
                if info is None:
                    raise ValueError("unknown stimulus program")
            extra = body.get("stimulus_args") or ""
            if not isinstance(extra, str) or len(extra) > 300:
                raise ValueError("stimulus_args must be a short string")
            transport = self.transport_factory() if self.transport_factory else None
            session = ExternalSession(cfg, target, self.output, transport=transport,
                                      model_factory=self.model, profile=name)
            self.stimulus = None
            if info is not None:
                args = build_args(info, bool(body.get("keyboard")), extra, target)
                # The control stream exists already, so the program finds the speller at once.
                self.stimulus = StimulusProcess(program, args, session.storage.directory / "external" / "stimulus.log")
                session.rec.emit("Session Recorder", "command_issued",
                                 {"command": "start_stimulus_program", "text": program, "result": args})
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
        return {"version": (id(s) % 10**6) * 10**6 + s.version if s else 0,
                "session": self.monitor()}


def make_handler(manager: Manager):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EEGSpellerExternal/0.1"

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
                return self._send(200, {"profiles": profiles(), "tasks": settings.read_task_list(),
                                        "programs": stimulus_programs()})
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
                if s is None:
                    raise ValueError("no session")
                if self.path == "/api/pause":
                    s.set_paused(bool(body.get("paused")))
                    return self._send(200, manager.monitor())
                if self.path == "/api/stop":
                    s.request_stop("operator_stopped")
                    return self._send(200, manager.monitor())
                return self._send(404, {"error": "not found"})
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
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
