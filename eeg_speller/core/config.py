"""Validated configuration and deterministic seed derivation."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib
import json
import yaml
from pydantic import BaseModel, ConfigDict, field_validator


class Config(BaseModel):
    model_config = ConfigDict(extra="allow")
    seed: int
    participant_id: str
    experiment_id: str | None = None
    task_id: str
    run_mode: str = "simulate"
    mode: dict
    repertoire: dict
    distributions: dict
    physiology: dict
    user: dict
    stimulus: dict
    epoch: dict
    runtime: dict
    llm: dict
    fast_llm: dict
    word_prediction: dict
    fusion: dict
    detector: dict
    presentation_feedback: dict
    slow_llm: dict
    rl: dict
    metrics: dict
    stats: dict

    @field_validator("repertoire")
    @classmethod
    def check_repertoire(cls, v: dict) -> dict:
        # ASSUMPTION[MA-01]
        # IMPROV[Q-07]
        # IMPROV[Q-38]
        from eeg_speller.core.keyboard import validate_keyboard
        validate_keyboard(v.get("symbols", []), v.get("layout", [0, 0]))
        return v

    @field_validator("stimulus")
    @classmethod
    def check_timing(cls, v: dict) -> dict:
        # ASSUMPTION[MA-33]
        # IMPROV[Q-29]
        for key, low, high in (("flash_ms", 50, 500), ("isi_ms", 0, 1000), ("pause_ms", 500, 5000)):
            val = v[key]
            if not isinstance(val, int) or not low <= val <= high or val % 10:
                raise ValueError(f"{key} must be a 10-ms step in [{low}, {high}]")
        if v["paradigm"] not in ("row_column", "single_symbol"):
            raise ValueError("unknown paradigm")
        return v

    def child_seed(self, block: str) -> int:
        digest = hashlib.sha256(f"{self.seed}:{block}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _merge(base: dict, patch: dict) -> dict:
    out = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    inherited = data.pop("inherits", None)
    if inherited:
        if Path(inherited).is_absolute() or ".." in Path(inherited).parts:
            raise ValueError("config inheritance must stay in the config directory")
        data = _merge(load_config(path.parent / inherited).model_dump(), data)
    return Config.model_validate(data)
