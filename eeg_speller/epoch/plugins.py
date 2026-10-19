"""Statistical Inference as replaceable Python modules from the ``inference/`` folder.

A module defines ``PARAMS`` (defaults and hints) and ``combine(prior, likelihood, params)``
working on log-probability arrays; see inference/README.md. The speller normalizes the
result, checks it and falls back to the physiology likelihood if the module fails.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from eeg_speller.core.distributions import Posterior, normalize_logits

PROJECT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = PROJECT / "inference"


def _doc_label(source: str, fallback: str) -> str:
    try:
        doc = ast.get_docstring(ast.parse(source)) or ""
    except SyntaxError:
        return fallback
    return (doc.strip().splitlines() or [fallback])[0].rstrip(".")[:120]


def list_plugins() -> list[dict]:
    """Plugins for the operator window: id, label, params (without importing the code)."""
    out = []
    for path in sorted(INFERENCE_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        source = path.read_text(encoding="utf-8")
        if "def combine" not in source:
            continue
        try:
            params = load_plugin(path.stem).params
            error = None
        except Exception as exc:          # show broken plugins with the reason
            params, error = {}, f"{type(exc).__name__}: {exc}"
        out.append({"id": path.stem, "label": _doc_label(source, path.stem), "params": params, "error": error})
    return out


class InferencePlugin:
    def __init__(self, name: str, module, source_hash: str):
        self.name, self.module, self.source_hash = name, module, source_hash
        self.params = dict(getattr(module, "PARAMS", {}))

    def resolve(self, values: dict | None) -> dict:
        out = {}
        for key, spec in self.params.items():
            value = (values or {}).get(key, spec.get("default") if isinstance(spec, dict) else spec)
            if isinstance(spec, dict):
                value = float(value)
                low, high = spec.get("min"), spec.get("max")
                if (low is not None and value < low) or (high is not None and value > high):
                    raise ValueError(f"{self.name}.{key} must be in [{low}, {high}]")
            out[key] = value
        return out


def load_plugin(name: str) -> InferencePlugin:
    if not name.replace("_", "").isalnum():
        raise ValueError("inference plugin name must be a file name without extension")
    path = INFERENCE_DIR / f"{name}.py"
    if not path.is_file():
        raise ValueError(f"unknown inference plugin: {name}")
    raw = path.read_bytes()
    spec = importlib.util.spec_from_file_location(f"speller_inference_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "combine", None)):
        raise ValueError(f"{name}.py has no combine(prior, likelihood, params)")
    return InferencePlugin(name, module, hashlib.sha256(raw).hexdigest())


class PluginFusion:
    """Statistical Inference; NEED-F-04/06; MA-12/13; Q-10."""

    def __init__(self, cfg: dict, llm_mode: str):
        # IMPROV[Q-10]
        self.llm_mode = llm_mode
        name = cfg.get("plugin") or {"log_pool": "log_pool", "linear_pool": "linear_pool"}.get(
            cfg.get("strategy", "log_pool"), "log_pool")
        self.plugin = load_plugin(name)
        defaults = {k: cfg[k] for k in ("alpha", "beta", "lambda") if k in cfg}
        self.values = self.plugin.resolve({**defaults, **(cfg.get("params") or {})})
        self.last_error: str | None = None

    @property
    def name(self) -> str:
        return self.plugin.name

    def combine(self, prior, obs) -> Posterior:
        # ASSUMPTION[MA-03]
        if obs.semantics != "likelihood":
            raise ValueError("physiology prior must be removed before fusion")
        self.last_error = None
        if self.llm_mode != "on":
            # ASSUMPTION[MA-13]
            return Posterior(obs.epoch_id, normalize_logits(obs.log_p), "physiology_only", {})
        try:
            # ASSUMPTION[MA-12]
            out = np.asarray(self.plugin.module.combine(np.array(prior.log_p), np.array(obs.log_p),
                                                        dict(self.values)), dtype=float)
            if out.shape != obs.log_p.shape or not np.all(np.isfinite(out)):
                raise ValueError(f"combine returned shape {out.shape} or non-finite values")
            log_p = normalize_logits(out)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return Posterior(obs.epoch_id, normalize_logits(obs.log_p), f"{self.name}:fallback",
                             {**self.values, "error": self.last_error})
        return Posterior(obs.epoch_id, log_p, self.name, dict(self.values))
