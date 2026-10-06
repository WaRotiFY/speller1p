"""Experimenter-editable blocks of the browser mode and their validation.

The operator window may change only the keys listed in ``FIELDS``. Each key is
applied on top of a ``configs/gui_*.yaml`` profile and the result is validated
by the regular ``Config`` model, so a browser request can never reach other
configuration keys or arbitrary files.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import re

import yaml

from eeg_speller.core.config import Config, load_config


PROJECT = Path(__file__).resolve().parents[2]
CONFIGS = PROJECT / "configs"
TASKS = PROJECT / "data" / "tasks_ru.txt"
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
PRESET_PATTERN = re.compile(r"^[a-z0-9_-]{1,32}$")


@dataclass(frozen=True)
class Field:
    key: str
    group: str
    label: str
    kind: str                      # int | float | choice | bool | text
    low: float | None = None
    high: float | None = None
    step: float | None = None
    choices: tuple[tuple[str, str], ...] = ()
    default: object = None         # used when a profile does not define the key
    help: str = ""

    def coerce(self, value):
        if self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != int(value):
                raise ValueError(f"{self.key}: integer required")
            value = int(value)
        elif self.kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{self.key}: number required")
            value = float(value)
        elif self.kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{self.key}: true or false required")
            return value
        elif self.kind == "choice":
            if value not in {item for item, _ in self.choices}:
                raise ValueError(f"{self.key}: unknown option {value!r}")
            return value
        elif self.kind == "text":
            if not isinstance(value, str) or not 0 < len(value) <= 200 or any(ord(ch) < 32 for ch in value):
                raise ValueError(f"{self.key}: text of 1-200 printable characters required")
            return value
        if self.low is not None and value < self.low or self.high is not None and value > self.high:
            raise ValueError(f"{self.key}: value must be in [{self.low}, {self.high}]")
        if self.kind == "int" and self.step and (value - int(self.low or 0)) % int(self.step):
            raise ValueError(f"{self.key}: step is {int(self.step)}")
        return value

    def describe(self) -> dict:
        return {"key": self.key, "group": self.group, "label": self.label, "kind": self.kind,
                "min": self.low, "max": self.high, "step": self.step,
                "choices": [{"value": v, "label": label} for v, label in self.choices],
                "help": self.help}


# Groups follow the "experimental parts" list of the project plus the ТЗ items
# (timings, LLM on/off, paradigm). The EXP codes refer to docs/ASSUMPTIONS.md.
GROUPS = [
    {"id": "mode", "title": "Режим и парадигма", "refs": "REQ-PERF-04 · NEED-F-03"},
    {"id": "stimulus", "title": "Стимуляция", "refs": "EXP-05 · REQ-CONF-01"},
    {"id": "signal", "title": "Модель сигнала (пробел вместо ЭЭГ)", "refs": "EXP-08"},
    {"id": "fast", "title": "Быстрая БЯМ и контекст", "refs": "EXP-01 · EXP-07"},
    {"id": "fusion", "title": "Слияние prior и физиологии", "refs": "EXP-04"},
    {"id": "detector", "title": "Детектор символа", "refs": "EXP-06"},
    {"id": "slow", "title": "Медленная БЯМ и коррекция", "refs": "EXP-02 · EXP-03"},
]

FIELDS = [
    Field("mode.llm", "mode", "Языковая модель", "choice",
          choices=(("on", "С БЯМ"), ("off", "Без БЯМ: равномерный prior, без коррекции и слов"))),
    Field("stimulus.paradigm", "mode", "Парадигма предъявления", "choice",
          choices=(("row_column", "Строки и столбцы (12 вспышек)"), ("single_symbol", "Отдельные символы (36 вспышек)"))),
    Field("stimulus.flash_ms", "stimulus", "Вспышка, мс", "int", 50, 500, 10),
    Field("stimulus.isi_ms", "stimulus", "Интервал между вспышками, мс", "int", 0, 1000, 10),
    Field("stimulus.pause_ms", "stimulus", "Пауза перед эпохой, мс", "int", 500, 5000, 10),
    Field("epoch.n_repetitions", "stimulus", "Повторов на символ", "int", 1, 20, 1),
    Field("gui.response_hit", "signal", "P(нажатие | целевая вспышка)", "float", 0.01, 0.99, 0.01,
          help="Насколько декодер доверяет нажатию на целевой группе."),
    Field("gui.response_false_alarm", "signal", "P(нажатие | нецелевая вспышка)", "float", 0.01, 0.99, 0.01,
          help="Должно быть меньше вероятности для целевой вспышки."),
    Field("fast_llm.backend", "fast", "Бэкенд быстрой модели", "choice",
          choices=(("mock_ngram", "mock_ngram — символьная 5-грамма"),
                   ("hf_causal", "hf_causal — локальная модель transformers"))),
    Field("fast_llm.model_id", "fast", "model_id", "text",
          help="Для hf_causal — путь к скачанной модели или её имя в локальном кэше."),
    Field("fast_llm.context_source", "fast", "Контекст быстрой БЯМ", "choice",
          choices=(("primary", "Первичный текст"), ("secondary", "Исправленный текст"))),
    Field("word_prediction.n_max", "fast", "Слов-кандидатов", "int", 1, 10, 1),
    Field("fusion.strategy", "fusion", "Способ слияния", "choice",
          choices=(("log_pool", "Логарифмический пул (Байес при α=β=1)"), ("linear_pool", "Линейный пул"))),
    Field("fusion.alpha", "fusion", "α — вес prior (лог-пул)", "float", 0.0, 5.0, 0.05),
    Field("fusion.beta", "fusion", "β — вес физиологии (лог-пул)", "float", 0.0, 5.0, 0.05),
    Field("fusion.lambda", "fusion", "λ — доля prior (линейный пул)", "float", 0.0, 1.0, 0.05),
    Field("detector.mode", "detector", "Режим детектора", "choice",
          choices=(("abstain", "abstain — при неуверенности повторить эпоху"),
                   ("label", "label — выбрать всегда, пометить неуверенность"))),
    Field("detector.tau", "detector", "τ — порог уверенности", "float", 0.0, 1.0, 0.01),
    Field("detector.max_retries", "detector", "Максимум повторов эпохи", "int", 0, 100, 1),
    Field("detector.cold_start_symbols", "detector", "Символов «холодного старта»", "int", 0, 50, 1),
    Field("slow_llm.backend", "slow", "Бэкенд медленной модели", "choice",
          choices=(("mock_dictionary", "mock_dictionary — словарная коррекция"),),
          help="Облачные модели в браузерном режиме выключены (Q-33)."),
    Field("slow_llm.trigger", "slow", "Когда запускать коррекцию", "choice",
          choices=(("word_end", "На конце слова"), ("sentence_end", "На конце предложения"),
                   ("every_k_symbols", "Каждые k символов"), ("manual", "Не запускать"))),
    Field("slow_llm.every_k_symbols", "slow", "k для «каждые k символов»", "int", 1, 50, 1, default=10),
    Field("slow_llm.window_words", "slow", "Окно, слов", "int", 1, 10, 1),
    Field("slow_llm.min_context_words", "slow", "Минимум слов контекста", "int", 0, 10, 1),
    Field("slow_llm.strict_word_count", "slow", "Строго то же число слов", "bool"),
    Field("slow_llm.mock.max_edit", "slow", "mock: макс. правок в слове", "int", 1, 3, 1),
    Field("slow_llm.mock.same_row_col_cost", "slow", "mock: цена замены в той же строке/столбце", "float", 0.0, 1.0, 0.05),
]
FIELD_BY_KEY = {item.key: item for item in FIELDS}


def _get(data: dict, key: str, default=None):
    node = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _set(data: dict, key: str, value) -> None:
    parts = key.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"{key}: cannot set nested value")
    node[parts[-1]] = value


def profiles() -> list[str]:
    names = []
    for path in sorted(CONFIGS.glob("gui_*.yaml")):
        try:
            load_config(path)
        except Exception:
            continue
        names.append(path.name)
    return names


def profile_values(name: str) -> dict:
    if name not in profiles():
        raise ValueError("choose a listed GUI profile")
    data = load_config(CONFIGS / name).model_dump()
    return {item.key: _get(data, item.key, item.default) for item in FIELDS}


def values_of(cfg: Config) -> dict:
    data = cfg.model_dump()
    return {item.key: _get(data, item.key, item.default) for item in FIELDS}


def read_task_list() -> list[dict]:
    tasks = []
    for line in TASKS.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            task_id, text = line.split("\t", 1)
            tasks.append({"id": task_id, "text": text})
    return tasks


def check_identifier(value, what: str, optional: bool = False):
    if optional and value in (None, ""):
        return None
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError(f"{what}: 1-32 latin letters, digits, '_' or '-'")
    return value


def build_config(profile: str, overrides: dict | None = None, *, participant_id: str,
                 task_id: str, experiment_id: str | None = None) -> Config:
    """Profile + whitelisted overrides -> validated live-mode configuration."""
    if profile not in profiles():
        raise ValueError("choose a listed GUI profile")
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("settings must be an object")
    data = deepcopy(load_config(CONFIGS / profile).model_dump())
    for key, value in (overrides or {}).items():
        item = FIELD_BY_KEY.get(key)
        if item is None:
            raise ValueError(f"setting {key!r} cannot be changed from the browser")
        _set(data, key, item.coerce(value))
    for item in FIELDS:
        if item.default is not None and _get(data, item.key) is None:
            _set(data, item.key, item.default)
    data["participant_id"] = participant_id
    data["task_id"] = task_id
    data["experiment_id"] = experiment_id
    data["run_mode"] = "live"
    data["llm"] = {**data.get("llm", {}), "allow_remote": False}
    cfg = Config.model_validate(data)
    hit, false = float(cfg.gui["response_hit"]), float(cfg.gui["response_false_alarm"])
    if not 0 < false < hit < 1:
        raise ValueError("P(нажатие | нецелевая) must be below P(нажатие | целевая)")
    return cfg


def save_preset(name: str, base: str, overrides: dict) -> str:
    """Write configs/gui_<name>.yaml inheriting from a listed profile; never overwrites."""
    if not isinstance(name, str) or not PRESET_PATTERN.fullmatch(name):
        raise ValueError("preset name: 1-32 lowercase latin letters, digits, '_' or '-'")
    filename = f"gui_{name}.yaml"
    path = CONFIGS / filename
    if path.exists():
        raise ValueError(f"{filename} already exists")
    cfg = build_config(base, overrides, participant_id="GUI-000", task_id="GUI")
    base_values = profile_values(base)
    patch: dict = {}
    for key, value in values_of(cfg).items():
        if value != base_values.get(key):
            _set(patch, key, value)
    text = f"# Saved from the operator window.\ninherits: {base}\n"
    if patch:
        text += yaml.safe_dump(patch, allow_unicode=True, sort_keys=False)
    path.write_text(text, encoding="utf-8")
    load_config(path)
    return filename


def schema() -> dict:
    return {"groups": GROUPS, "fields": [item.describe() for item in FIELDS]}
