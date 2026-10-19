"""Find stimulus modules in stimulus_kit/modules and start the chosen one as a separate process.

A module is a ``*.py`` file or a folder with ``main.py`` that uses ``SpellerLink``; an
optional ``module.json`` gives its ``label`` and ``description``. The operator only picks
the module: all its parameters (keyboard, timing, test mode, screen) live in the module's
own ``settings.json`` and are edited by the module's authors. The module runs in the
session folder, so its own logs end up in the recording. The speller knows nothing about
what the module draws: it only starts and stops it.
"""
from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[2]
KIT = PROJECT / "stimulus_kit"
# User folder first: physiologists drop their programs into stimulus_kit/modules/.
STIMULUS_DIRS = {"modules": KIT / "modules"}
# Lines of library chatter that are not useful in the operator window.
NOISE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+ \(.*\)\s+\[.*\]|pygame \d|Hello from the pygame)")


def _first_doc_line(source: str, fallback: str) -> str:
    try:
        doc = ast.get_docstring(ast.parse(source)) or ""
    except SyntaxError:
        doc = ""
    first = doc.strip().splitlines()[0].rstrip(".") if doc.strip() else fallback
    if ":" in first and first.lower().startswith("example"):
        first = first.split(":", 1)[1].strip()
    return first.split(" — ")[0][:120]


def _describe_file(path: Path) -> dict | None:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if "SpellerLink" not in source or path.name.startswith("_"):
        return None
    return {"label": _first_doc_line(source, path.stem), "needs_text": '"--text"' in source,
            "settings": None, "description": None}


def _describe_folder(folder: Path) -> dict | None:
    # IMPROV[Q-44]
    main = folder / "main.py"
    if not main.is_file() or folder.name.startswith(("_", ".")):
        return None
    try:
        sources = "\n".join(p.read_text(encoding="utf-8") for p in folder.glob("*.py"))
    except (OSError, UnicodeDecodeError):
        return None
    if "SpellerLink" not in sources:
        return None
    meta = {}
    if (folder / "module.json").is_file():
        try:
            meta = json.loads((folder / "module.json").read_text(encoding="utf-8"))
        except ValueError:
            meta = {}
    settings = None
    if (folder / "settings.json").is_file():
        try:
            settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
        except ValueError:
            settings = None
    return {"label": meta.get("label") or _first_doc_line(main.read_text(encoding="utf-8"), folder.name),
            "needs_text": '"--text"' in sources, "settings": settings, "description": meta.get("description")}


def stimulus_programs() -> list[dict]:
    """All modules the operator can choose, as {id, label, folder, needs_text, settings, description}."""
    out = []
    for folder, directory in STIMULUS_DIRS.items():
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            info = (_describe_folder(path) if path.is_dir() else
                    _describe_file(path) if path.suffix == ".py" else None)
            if info:
                out.append({"id": f"{folder}/{path.name}", "folder": folder, "kind": "dir" if path.is_dir() else "file",
                            **info})
    return out


def resolve(program_id: str) -> Path:
    for item in stimulus_programs():
        if item["id"] == program_id:
            folder, name = program_id.split("/", 1)
            path = STIMULUS_DIRS[folder] / name
            return path / "main.py" if path.is_dir() else path
    raise ValueError("unknown stimulus program")


class StimulusProcess:
    """One running stimulus program; output goes to a log file in the session folder."""

    def __init__(self, program_id: str, args: list[str], log_path: Path):
        self.program_id = program_id
        self.path = resolve(program_id)
        self.args = args
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("w", encoding="utf-8", errors="replace")
        env = dict(os.environ)
        # speller_link.py and module_tools.py are importable without copying them.
        env["PYTHONPATH"] = os.pathsep.join([str(KIT), str(KIT / "examples"), str(self.path.parent),
                                             env.get("PYTHONPATH", "")])
        env["PYTHONIOENCODING"] = "utf-8"
        workdir = log_path.parent / "module_run"
        workdir.mkdir(exist_ok=True)
        self.proc = subprocess.Popen([sys.executable, "-u", str(self.path), *args], cwd=str(workdir),
                                     stdout=self.log, stderr=subprocess.STDOUT, env=env)
        self.started = time.monotonic()

    def status(self) -> dict:
        code = self.proc.poll()
        return {"program": self.program_id, "args": self.args, "running": code is None,
                "returncode": code, "log_tail": self.tail(), "log": str(self.log_path)}

    def tail(self, lines: int = 12) -> list[str]:
        try:
            if not self.log.closed:
                self.log.flush()
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            return []
        return [line for line in text.splitlines() if line.strip() and not NOISE.match(line)][-lines:]

    def stop(self, grace_s: float = 3.0) -> None:
        """Let the program exit on the speller's "end" message, then terminate it."""
        def _stop():
            deadline = time.monotonic() + grace_s
            while self.proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            self.log.close()
        threading.Thread(target=_stop, daemon=True).start()


def build_args(info: dict, target: str) -> list[str]:
    """Modules take everything from their settings.json; only the simple examples need the phrase."""
    return ["--text", target or "привет"] if info.get("needs_text") else []
