"""Copy-spelling tasks and identifiers for the operator window."""
from __future__ import annotations

from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[2]
TASKS = PROJECT / "data" / "tasks_ru.txt"
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def read_task_list(path: Path = TASKS) -> list[dict]:
    """Lines '<task_id><TAB><text>' of data/tasks_ru.txt; '#' starts a comment."""
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "\t" in line:
            task_id, text = line.split("\t", 1)
            tasks.append({"id": task_id, "text": text})
    return tasks


def check_identifier(value, what: str, optional: bool = False):
    if optional and value in (None, ""):
        return None
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError(f"{what}: 1-32 latin letters, digits, '_' or '-'")
    return value


def resolve_task(task) -> tuple[str, str]:
    """Operator choice -> (task_id, target text); free typing has an empty target."""
    if not isinstance(task, dict):
        raise ValueError("task must be an object")
    kind = task.get("kind")
    if kind == "task":
        for item in read_task_list():
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
