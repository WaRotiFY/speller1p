"""Loopback-only HTTP server for the participant and experimenter windows."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from eeg_speller.web import settings
from eeg_speller.web.engine import SessionManager, available_configs, gui_symbols


STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 65536
FILES = {"/": ("index.html", "text/html; charset=utf-8"),
         "/app.js": ("app.js", "application/javascript; charset=utf-8"),
         "/style.css": ("style.css", "text/css; charset=utf-8"),
         "/operator": ("operator.html", "text/html; charset=utf-8"),
         "/operator.js": ("operator.js", "application/javascript; charset=utf-8"),
         "/operator.css": ("operator.css", "text/css; charset=utf-8")}


def resolve_task(task) -> tuple[str, str]:
    """Return (task_id, target text) for an operator task choice."""
    if not isinstance(task, dict):
        raise ValueError("task must be an object")
    kind = task.get("kind")
    if kind == "task":
        for item in settings.read_task_list():
            if item["id"] == task.get("id"):
                return item["id"], item["text"]
        raise ValueError("unknown task id")
    if kind == "custom":
        text = task.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("enter a phrase for the custom task")
        # A trailing space is a symbol to type (it also triggers word-end correction).
        return "CUSTOM", text.lstrip().lower().replace("ё", "е")
    if kind == "free":
        return "FREE", ""
    raise ValueError("task kind must be task, custom or free")


def _number(query: dict, name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(query.get(name, [default])[0])
    except ValueError:
        value = default
    return min(high, max(low, value))


def make_handler(manager: SessionManager, qwen=None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EEGSpellerLocal/0.2"

        def _send(self, status, body, content_type="application/json; charset=utf-8"):
            data = (json.dumps(body, ensure_ascii=False).encode("utf-8")
                    if content_type.startswith("application/json") else body)
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy",
                                 "default-src 'self'; script-src 'self'; style-src 'self'; "
                                 "img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass            # the browser closed a long poll; nothing to report

        def do_GET(self):
            if not self._valid_host():
                return self._send(403, {"error": "loopback host required"})
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            if url.path == "/api/configs":
                return self._send(200, {"configs": available_configs(), "symbols": gui_symbols(),
                                        "qwen": self._qwen_status()})
            if url.path == "/api/operator/init":
                return self._send(200, {"profiles": settings.profiles(), "tasks": settings.read_task_list(),
                                        "schema": settings.schema(), "symbols": gui_symbols(),
                                        "qwen": self._qwen_status()})
            if url.path == "/api/profile":
                try:
                    name = query.get("name", [""])[0]
                    return self._send(200, {"name": name, "values": settings.profile_values(name)})
                except ValueError as exc:
                    return self._send(404, {"error": str(exc)})
            if url.path == "/api/live":
                role = "participant" if query.get("role", [""])[0] == "participant" else "operator"
                since = int(_number(query, "since", -1, -1, 2**62))
                timeout = _number(query, "timeout", 10, 0, 25)
                manager.wait(since, timeout, role)
                return self._send(200, manager.live(role))
            if url.path == "/api/state":
                try:
                    session_id = query.get("session_id", [""])[0]
                    return self._send(200, manager.get(session_id).snapshot())
                except ValueError as exc:
                    return self._send(404, {"error": str(exc)})
            item = FILES.get(url.path)
            if item is None:
                return self._send(404, {"error": "not found"})
            name, mime = item
            return self._send(200, (STATIC / name).read_bytes(), mime)

        def do_POST(self):
            if not self._valid_host():
                return self._send(403, {"error": "loopback host required"})
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            if origin and origin not in {"http://" + host, "https://" + host}:
                return self._send(403, {"error": "cross-origin request rejected"})
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                return self._send(415, {"error": "JSON required"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY:
                    raise ValueError("invalid request size")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("object required")
                result = self._dispatch(body)
                if result is None:
                    return self._send(404, {"error": "not found"})
                return self._send(200, result)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                return self._send(400, {"error": str(exc)})
            except Exception as exc:
                self.log_error("request failed: %s", exc)
                return self._send(500, {"error": "server error; see local terminal"})

        def _dispatch(self, body: dict):
            path = self.path
            if path == "/api/start":
                if "profile" not in body:          # original single-window request
                    return manager.start(body.get("config", ""), body.get("target", ""))
                task_id, target = resolve_task(body.get("task"))
                return manager.start(
                    body.get("profile", ""), target, overrides=body.get("settings") or {},
                    participant_id=settings.check_identifier(body.get("participant_id"), "participant_id"),
                    experiment_id=settings.check_identifier(body.get("experiment_id"), "experiment_id",
                                                            optional=True),
                    task_id=task_id)
            if path == "/api/epoch":
                return manager.submit(body.get("session_id", ""), body.get("hits", []), body.get("epoch_id"))
            if path == "/api/progress":
                return manager.progress(body.get("session_id", ""), body.get("epoch_id"),
                                        body.get("event", ""), body.get("flash_id"))
            if path == "/api/pause":
                return manager.pause(body.get("session_id", ""), body.get("paused"))
            if path == "/api/stop":
                reason = "operator_stopped" if body.get("by") == "operator" else "user_stopped"
                return manager.stop(body.get("session_id", ""), reason)
            if path == "/api/preset":
                name = settings.save_preset(body.get("name"), body.get("profile", ""), body.get("settings") or {})
                return {"saved": name, "profiles": settings.profiles()}
            if path == "/api/suggest":
                if qwen is None:
                    raise ValueError("Qwen is not configured")
                session = manager.get(body.get("session_id", ""))
                return {"suggestion": qwen.suggest(session.primary.value)}
            return None

        def _qwen_status(self):
            return qwen.status() if qwen else {"ready": False, "message": "model not configured"}

        def _valid_host(self):
            host = self.headers.get("Host", "").split(":", 1)[0].lower()
            return host in {"127.0.0.1", "localhost"}

        def log_message(self, fmt, *args):
            # Avoid writing target text or session contents into access logs.
            if not self.path.startswith("/api/"):
                super().log_message(fmt, *args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Local-only EEG speller browser interface")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=Path("runs/gui"))
    parser.add_argument("--qwen-model", type=Path, help="existing local GGUF model file")
    parser.add_argument("--qwen-cli", type=Path, help="CPU-only llama.cpp executable")
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error("invalid port")
    qwen = None
    if args.qwen_model:
        from eeg_speller.web.qwen import LocalQwen
        if not args.qwen_cli:
            parser.error("--qwen-cli is required with --qwen-model")
        qwen = LocalQwen(args.qwen_model, args.qwen_cli)
    manager = SessionManager(args.output)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(manager, qwen))
    server.daemon_threads = True
    print(f"Experimenter: http://127.0.0.1:{args.port}/operator", flush=True)
    print(f"Participant:  http://127.0.0.1:{args.port}/  (local-only)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
