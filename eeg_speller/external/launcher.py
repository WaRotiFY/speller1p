"""Find stimulus programs in folders and start the chosen one as a separate process.

A program is offered in the operator window when it is a ``*.py`` file in one of
``STIMULUS_DIRS`` that uses ``SpellerLink`` (the only rule for module authors).
Its first docstring line becomes the label. The speller still knows nothing about
what the program draws: it only starts and stops the process.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[2]
KIT = PROJECT / "stimulus_kit"
# User folder first: physiologists drop their programs into stimulus_kit/modules/.
STIMULUS_DIRS = {"modules": KIT / "modules", "examples": KIT / "examples"}


def _describe(path: Path) -> dict | None:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if "SpellerLink" not in source or path.name.startswith("_"):
        return None
    try:
        doc = ast.get_docstring(ast.parse(source)) or ""
    except SyntaxError:
        doc = ""
    first = doc.strip().splitlines()[0].rstrip(".") if doc.strip() else path.stem
    if ":" in first and first.lower().startswith("example"):
        first = first.split(":", 1)[1].strip()
    return {"label": first[:120], "keyboard": "--keyboard" in source, "needs_text": '"--text"' in source}


def stimulus_programs() -> list[dict]:
    """All programs the operator can choose, as {id, label, folder, keyboard, needs_text}."""
    out = []
    for folder, directory in STIMULUS_DIRS.items():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            info = _describe(path)
            if info:
                out.append({"id": f"{folder}/{path.name}", "folder": folder, **info})
    return out


def resolve(program_id: str) -> Path:
    for item in stimulus_programs():
        if item["id"] == program_id:
            folder, name = program_id.split("/", 1)
            return STIMULUS_DIRS[folder] / name
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
        # speller_link.py is importable without copying it next to the program.
        env["PYTHONPATH"] = os.pathsep.join([str(KIT), str(self.path.parent), env.get("PYTHONPATH", "")])
        env["PYTHONIOENCODING"] = "utf-8"
        self.proc = subprocess.Popen([sys.executable, "-u", str(self.path), *args], cwd=str(PROJECT),
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
        return [line for line in text.splitlines() if line.strip()][-lines:]

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


def build_args(info: dict, keyboard: bool, extra: str, target: str) -> list[str]:
    args = []
    if keyboard and info.get("keyboard"):
        args.append("--keyboard")
    if info.get("needs_text") and "--text" not in extra:
        args += ["--text", target or "привет"]
    if extra.strip():
        args += shlex.split(extra, posix=os.name != "nt")
    return args
