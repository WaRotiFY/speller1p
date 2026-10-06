"""Interactive stimulus sessions using the existing prior, fusion and detector blocks.

One experimenter window configures and controls a session; one participant window
shows the flashes and collects space presses. Both follow the same server-side
session through ``SessionManager.wait`` (long polling).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import threading
import time
import uuid

import numpy as np

from eeg_speller.core.clock import Clock
from eeg_speller.core.config import Config, load_config
from eeg_speller.core.distributions import Repertoire, SymbolDistribution, log_probs, normalize_logits
from eeg_speller.decoding.detector import SymbolDetector
from eeg_speller.epoch.fusion import Fusion
from eeg_speller.ext.word_prediction import WordPrediction
from eeg_speller.llm.fast import create_fast
from eeg_speller.llm.slow import CorrectionScheduler, create_slow, prompt_hash, validate_correction
from eeg_speller.physiology.base import PhysiologyObservation
from eeg_speller.recording.recorder import SessionRecorder
from eeg_speller.recording.storage import SessionStorage
from eeg_speller.text.primary import PrimaryText
from eeg_speller.text.secondary import SecondaryText
from eeg_speller.web import settings


PROJECT = Path(__file__).resolve().parents[2]
CONFIGS = PROJECT / "configs"
CORPUS = PROJECT / "data" / "corpus_ru_fallback.txt"
LOG_LIMIT = 500


def available_configs() -> list[dict]:
    result = []
    for name in settings.profiles():
        cfg = load_config(CONFIGS / name)
        result.append({"name": name, "flash_ms": cfg.stimulus["flash_ms"],
                       "isi_ms": cfg.stimulus["isi_ms"], "repetitions": cfg.epoch["n_repetitions"],
                       "paradigm": cfg.stimulus["paradigm"],
                       "model": cfg.fast_llm["backend"] if cfg.mode["llm"] == "on" else "off"})
    return result


def gui_symbols() -> list[str]:
    return list(load_config(CONFIGS / "gui_slow.yaml").repertoire["symbols"])


def _create_model(kind: str, cfg: dict):
    try:
        if kind == "fast":
            return create_fast(cfg, CORPUS)
        return create_slow(cfg, False, CORPUS)
    except ImportError as exc:
        raise ValueError(f"{cfg['backend']}: missing optional dependency ({exc.name}); "
                         "install it with `uv pip install -e .[hf]`") from exc
    except (OSError, NotImplementedError) as exc:
        raise ValueError(f"{cfg['backend']}: {exc}") from exc


def _probs(log_p) -> list[float]:
    return [round(float(x), 5) for x in np.exp(log_p)]


class InteractiveSession:
    """Human space presses replace binary EEG responses to each visual flash."""

    def __init__(self, config_name: str, target: str, output_root: Path, *,
                 overrides: dict | None = None, participant_id: str | None = None,
                 task_id: str = "GUI", experiment_id: str | None = None, model_factory=None):
        participant_id = participant_id or "GUI-" + uuid.uuid4().hex[:8]
        self.cfg = settings.build_config(config_name, overrides, participant_id=participant_id,
                                         task_id=task_id, experiment_id=experiment_id)
        self.profile = config_name
        self.rep = Repertoire(tuple(self.cfg.repertoire["symbols"]), tuple(self.cfg.repertoire["layout"]))
        if len(target) > 120 or any(ch not in self.rep.symbols or ch == "⌫" for ch in target):
            raise ValueError("target must contain at most 120 repertoire characters")
        self.target = target
        self.primary = PrimaryText()
        self.secondary = SecondaryText()
        self.detector = SymbolDetector(self.cfg.detector, self.rep)
        self.fusion = Fusion(self.cfg.fusion, self.cfg.mode["llm"])
        make = model_factory or _create_model
        llm_on = self.cfg.mode["llm"] == "on"
        self.fast = make("fast", self.cfg.fast_llm) if llm_on else None
        self.slow = make("slow", self.cfg.slow_llm) if llm_on else None
        self.words = WordPrediction(self.fast, self.cfg.word_prediction) if self.fast else None
        self.scheduler = CorrectionScheduler(self.cfg.slow_llm) if self.slow else None
        models = {"fast_llm": {"backend": self.cfg.fast_llm["backend"] if self.fast else "uniform",
                               "model_id": self.fast.model_id if self.fast else "uniform",
                               "version": self.fast.version if self.fast else "1",
                               "hash": self.fast.weights_hash if self.fast else "none"},
                  "slow_llm": {"backend": self.cfg.slow_llm["backend"] if self.slow else "disabled",
                               "model_id": self.slow.model_id if self.slow else "disabled",
                               "version": self.slow.version if self.slow else "1",
                               "hash": self.slow.weights_hash if self.slow else "none"},
                  "physiology": {"backend": "keyboard_space", "model_id": "keyboard_space", "version": "1",
                                  "hash": "none"}}
        self.storage = SessionStorage(output_root, self.cfg, self.rep, models)
        self.clock = Clock(live=True)
        self.rec = SessionRecorder(self.storage.directory, self.storage.session_id,
                                   self.cfg.participant_id, self.clock)
        self.rec.emit("Session Recorder", "session_started",
                      {"run_mode": "live", "llm_mode": self.cfg.mode["llm"],
                       "paradigm": self.cfg.stimulus["paradigm"], "seed": self.cfg.seed,
                       "config_hash": self.cfg.digest()})
        self.rec.emit("Session Recorder", "task_assigned", {"task_id": task_id, "target_text": target})
        self.lock = threading.RLock()
        self.epoch_id = 0
        self.target_index = 0
        self.retry = 0
        self.finished = False
        self.end_reason = None
        self.last = None
        self.last_epoch = None
        self.log: list[dict] = []
        self.corrections: list[dict] = []
        self.paused = False
        self.pause_started = None
        self.paused_total = 0.0
        self.first_epoch_wall = None
        self.ended_wall = None
        self.plan = self._make_plan()
        self._open_epoch()

    # ------------------------------------------------------------------ epochs
    def _make_plan(self) -> list[dict]:
        rng = random.Random(self.cfg.child_seed("gui_flashes") + self.epoch_id)
        groups = []
        if self.cfg.stimulus["paradigm"] == "row_column":
            groups.extend({"kind": "row", "index": row, "indices": list(range(row * 6, row * 6 + 6))}
                          for row in range(6))
            groups.extend({"kind": "column", "index": col, "indices": list(range(col, 36, 6))}
                          for col in range(6))
        else:
            groups.extend({"kind": "symbol", "index": i, "indices": [i]} for i in range(36))
        plan = []
        for repetition in range(int(self.cfg.epoch["n_repetitions"])):
            order = groups[:]
            rng.shuffle(order)
            for group in order:
                plan.append({"id": len(plan), "repetition": repetition, **group})
        return plan

    def _context(self) -> str:
        # IMPROV[Q-16]
        return self.primary.value if self.cfg.fast_llm["context_source"] == "primary" else self.secondary.value

    def _open_epoch(self) -> None:
        """Compute the prior and word candidates as soon as the epoch is known (MA-08)."""
        start = time.perf_counter_ns()
        if self.fast is None:
            prior = SymbolDistribution(self.rep, log_probs(np.ones(36)), "uniform", len(self.primary.value))
        else:
            prior = self.fast.predict(self._context(), self.rep)
        prior_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        words = self.words.predict(self.primary.value) if self.words else []
        words_ms = (time.perf_counter_ns() - start) / 1e6
        self.pending = {"prior": prior, "prior_ms": prior_ms, "words": words, "words_ms": words_ms,
                        "context_tail": self._context()[-64:]}
        self.progress = {"phase": "ready", "flash": -1, "closed": -1,
                         "hits": [False] * len(self.plan), "began_wall": None}

    def _keyboard_scores(self, hits, mask=None) -> np.ndarray:
        p_hit = float(self.cfg.gui["response_hit"])
        p_false = float(self.cfg.gui["response_false_alarm"])
        scores = np.zeros(36)
        for flash, hit in zip(self.plan, hits):
            if mask is not None and not mask[flash["id"]]:
                continue
            inside = np.zeros(36, dtype=bool)
            inside[flash["indices"]] = True
            p = np.where(inside, p_hit, p_false)
            scores += np.log(p if hit else 1 - p)
        return scores

    def _current_target(self):
        if self.target and self.target_index < len(self.target):
            return self.target[self.target_index]
        return None

    def progress_event(self, epoch_id: int, event: str, flash_id: int | None = None) -> bool:
        """Live progress from the participant window; it is not written to the event log."""
        with self.lock:
            if self.finished or epoch_id != self.epoch_id:
                return False
            n = len(self.plan)
            if event in ("flash", "hit"):
                if type(flash_id) is not int or not 0 <= flash_id < n:
                    raise ValueError("invalid flash id")
            now = time.monotonic()
            if event == "epoch_begin":
                self.progress = {"phase": "flashing", "flash": -1, "closed": -1,
                                 "hits": [False] * n, "began_wall": now}
                if self.first_epoch_wall is None:
                    self.first_epoch_wall = now
            elif event == "flash":
                self.progress["phase"] = "flashing"
                self.progress["flash"] = max(self.progress["flash"], flash_id)
                self.progress["closed"] = max(self.progress["closed"], flash_id - 1)
            elif event == "hit":
                self.progress["hits"][flash_id] = True
            elif event == "flashes_done":
                self.progress["phase"] = "decoding"
                self.progress["closed"] = n - 1
                self.progress["flash"] = n - 1
            else:
                raise ValueError("unknown progress event")
            return True

    def set_paused(self, paused: bool) -> None:
        if type(paused) is not bool:
            raise ValueError("paused must be true or false")
        with self.lock:
            if self.finished or paused == self.paused:
                return
            self.paused = paused
            now = time.monotonic()
            if paused:
                self.pause_started = now
            else:
                self.paused_total += now - (self.pause_started or now)
                self.pause_started = None
            self.rec.emit("Session Recorder", "session_paused" if paused else "session_resumed",
                          {"next_epoch_id": self.epoch_id})

    def _correct(self, epoch_id: int) -> None:
        req = self.scheduler.on_primary_update(self.primary.value) if self.scheduler else None
        if req is None or not req.words:
            return
        self.rec.emit("Post-factum Text Correction", "correction_requested",
                      {"correction_id": req.correction_id, "trigger": req.trigger,
                       "window_words": req.words, "window_start_word": req.start_word,
                       "left_context_len": len(req.left_context), "model_id": self.slow.model_id,
                       "model_version": self.slow.version, "model_hash": self.slow.weights_hash,
                       "prompt_hash": prompt_hash(PROJECT / self.cfg.slow_llm["prompt_file"])}, epoch_id)
        start = time.perf_counter_ns()
        try:
            result = self.slow.correct(req.left_context, req.words)
            accepted, reason = validate_correction(req.words, result, self.cfg.slow_llm["strict_word_count"])
        except Exception as exc:
            result, accepted, reason = req.words, False, type(exc).__name__
        latency = (time.perf_counter_ns() - start) / 1e6
        self.rec.emit("Post-factum Text Correction", "correction_completed",
                      {"correction_id": req.correction_id, "words_in": req.words,
                       "words_out": result, "accepted": accepted, "reject_reason": reason,
                       "model_id": self.slow.model_id, "model_version": self.slow.version,
                       "model_hash": self.slow.weights_hash, "wall_latency_ms": latency}, epoch_id)
        self.corrections.append({"correction_id": req.correction_id, "epoch_id": epoch_id,
                                 "trigger": req.trigger, "words_in": list(req.words),
                                 "words_out": list(result), "accepted": accepted,
                                 "changed": accepted and list(result) != list(req.words),
                                 "reject_reason": reason, "latency_ms": round(latency, 2)})
        del self.corrections[:-LOG_LIMIT]
        if accepted:
            self.secondary.replace_window(result, req.start_word)
            self.rec.emit("Secondary (Corrected) Text", "secondary_text_updated",
                          {"correction_id": req.correction_id, "text": self.secondary.value,
                           "frozen_upto_word": self.secondary.frozen_upto_word}, epoch_id)

    def submit(self, hits: list[bool], epoch_id: int | None = None) -> dict:
        with self.lock:
            if self.finished:
                raise ValueError("session is finished")
            if epoch_id is not None and epoch_id != self.epoch_id:
                raise ValueError("epoch was already decoded")
            if len(hits) != len(self.plan) or any(type(hit) is not bool for hit in hits):
                raise ValueError("one boolean response per flash is required")
            eid = self.epoch_id
            position = len(self.primary.value)
            if self.first_epoch_wall is None:
                self.first_epoch_wall = time.monotonic()
            target_symbol = self._current_target()
            self.rec.emit("One Epoch", "epoch_started",
                          {"position": position,
                           "repertoire_hash": hashlib.sha256("".join(self.rep.symbols).encode()).hexdigest(),
                           "paradigm": self.cfg.stimulus["paradigm"],
                           "flash_ms": self.cfg.stimulus["flash_ms"],
                           "isi_ms": self.cfg.stimulus["isi_ms"],
                           "pause_ms": self.cfg.stimulus["pause_ms"],
                           "n_repetitions": self.cfg.epoch["n_repetitions"],
                           "retry_idx": self.retry, "target_symbol": target_symbol}, eid)
            prior = self.pending["prior"]
            self.rec.emit("Source Model (Fast LLM)", "prior_computed",
                          {"model_id": self.fast.model_id if self.fast else "uniform",
                           "context_source": self.cfg.fast_llm["context_source"],
                           "context_tail": self.pending["context_tail"],
                           "context_len": len(self.primary.value), "log_p": prior.log_p,
                           "wall_latency_ms": self.pending["prior_ms"]}, eid)
            if self.words:
                self.rec.emit("ext:word_prediction", "word_candidates",
                              {"prefix": self.primary.value.split(" ")[-1],
                               "candidates": self.pending["words"], "model_id": self.fast.model_id,
                               "wall_latency_ms": self.pending["words_ms"]}, eid)
            start = time.perf_counter_ns()
            for flash, hit in zip(self.plan, hits):
                self.rec.emit("Presentation Method", "stimulus_event",
                              {"flash_id": flash["id"], "repetition": flash["repetition"],
                               "kind": flash["kind"], "index": flash["index"],
                               "indices": flash["indices"], "keyboard_hit": hit}, eid)
            obs = PhysiologyObservation(eid, normalize_logits(self._keyboard_scores(hits)), "likelihood",
                                        module_metrics={"keyboard_hits": sum(hits)},
                                        stimulus_events=self.plan)
            self.rec.emit("EEG / EOG source", "physiology_observation",
                          {"log_p": obs.log_p, "semantics": "likelihood",
                           "source": "keyboard_space", "module_metrics": obs.module_metrics}, eid)
            post = self.fusion.combine(prior, obs)
            fusion_ms = (time.perf_counter_ns() - start) / 1e6
            self.rec.emit("Statistical Inference", "posterior_computed",
                          {"fusion": post.fusion, "params": post.params, "log_p": post.log_p,
                           "wall_latency_ms": fusion_ms}, eid)
            decision = self.detector.decide(post, position, self.retry)
            # No press at all means the participant gave no response: always repeat the epoch.
            # Otherwise the detector settings decide (abstain/label, tau, max_retries; MA-14).
            uncertain = not any(hits) or decision.uncertain
            symbol = decision.symbol if any(hits) else None
            top = np.argsort(-post.log_p)[:3]
            suggestions = [{"symbol": self.rep.symbols[int(i)],
                            "probability": round(float(np.exp(post.log_p[i])), 3)} for i in top]
            self.rec.emit("Primary Symbol Detector", "symbol_decided",
                          {"symbol": symbol, "confidence": decision.confidence,
                           "margin": decision.margin, "uncertain": uncertain,
                           "cold_start": decision.cold_start, "abstained": symbol is None}, eid)
            self._log_epoch(eid, position, target_symbol, prior, obs, post, decision, symbol,
                            uncertain, hits, fusion_ms)
            if symbol is not None:
                self.primary.apply(symbol)
                self.secondary.synchronize(self.primary.value)
                if self.target:
                    self.target_index = (max(0, self.target_index - 1) if symbol == "⌫"
                                         else min(len(self.target), self.target_index + 1))
                self.rec.emit("Primary Text", "primary_text_updated",
                              {"op": "delete" if symbol == "⌫" else "append",
                               "symbol": symbol, "position": position,
                               "text": self.primary.value}, eid)
                self._correct(eid)
                self.retry = 0
            else:
                self.retry += 1
            self.rec.emit("ext:ui", "ui_rendered",
                          {"primary_text": self.primary.value, "secondary_text": self.secondary.value,
                           "candidates_shown": suggestions,
                           "status": "selected" if symbol else "retry"}, eid)
            self.last = {"symbol": symbol, "confidence": round(decision.confidence, 3),
                         "suggestions": suggestions, "hits": sum(hits), "uncertain": uncertain}
            self.epoch_id += 1
            if self.target and self.target_index >= len(self.target):
                self.stop("target_finished")
            else:
                self.plan = self._make_plan()
                self._open_epoch()
            return self.snapshot()

    def _log_epoch(self, eid, position, target_symbol, prior, obs, post, decision, symbol,
                   uncertain, hits, fusion_ms) -> None:
        p_post = np.exp(post.log_p)
        row = {"epoch_id": eid, "position": position, "retry": self.retry, "target": target_symbol,
               "decided": symbol, "leader": self.rep.symbols[int(np.argmax(p_post))],
               "confidence": round(decision.confidence, 4), "margin": round(decision.margin, 4),
               "uncertain": bool(uncertain), "cold_start": bool(decision.cold_start),
               "hits": int(sum(hits)), "n_flashes": len(hits),
               "prior_ms": round(self.pending["prior_ms"], 2), "fusion_ms": round(fusion_ms, 2),
               "correct": None, "p_prior_target": None, "p_post_target": None, "rank_target": None}
        if target_symbol is not None:
            t = self.rep.index(target_symbol)
            row.update(p_prior_target=round(float(np.exp(prior.log_p[t])), 4),
                       p_post_target=round(float(p_post[t]), 4),
                       rank_target=int((p_post > p_post[t]).sum()) + 1,
                       correct=None if symbol is None else symbol == target_symbol)
        self.log.append(row)
        del self.log[:-LOG_LIMIT]
        self.last_epoch = {"epoch_id": eid, "target": target_symbol, "decided": symbol,
                           "prior": _probs(prior.log_p), "likelihood": _probs(obs.log_p),
                           "posterior": _probs(post.log_p), "hits": [i for i, h in enumerate(hits) if h],
                           "confidence": row["confidence"], "margin": row["margin"],
                           "uncertain": row["uncertain"], "words": self.pending["words"]}

    def stop(self, reason: str = "user_stopped") -> dict:
        with self.lock:
            if not self.finished:
                if self.paused:
                    self.set_paused(False)
                self.finished = True
                self.end_reason = reason
                self.ended_wall = time.monotonic()
                self.plan = []
                self.rec.emit("Session Recorder", "session_ended",
                              {"status": "completed", "reason": reason,
                               "n_epochs": self.epoch_id, "n_events": self.rec.seq + 1})
                count = self.rec.seq
                self.rec.close()
                self.storage.finish(self.target, self.primary.value, self.secondary.value,
                                    self.epoch_id, count)
            return self.snapshot()

    # --------------------------------------------------------------- snapshots
    def snapshot(self) -> dict:
        """What the participant window needs: no distributions, no settings."""
        with self.lock:
            return {"session_id": self.storage.session_id, "target": self.target,
                    "text": self.primary.value, "secondary": self.secondary.value,
                    "epoch_id": self.epoch_id, "retry": self.retry,
                    "target_index": self.target_index,
                    "plan": self.plan if not self.finished else [], "finished": self.finished,
                    "end_reason": self.end_reason, "paused": self.paused,
                    "last": self.last, "symbols": list(self.rep.symbols),
                    "timing": {"flash_ms": self.cfg.stimulus["flash_ms"],
                               "isi_ms": self.cfg.stimulus["isi_ms"],
                               "pause_ms": self.cfg.stimulus["pause_ms"]},
                    "config": {"paradigm": self.cfg.stimulus["paradigm"],
                               "repetitions": self.cfg.epoch["n_repetitions"],
                               "model": self.fast.model_id if self.fast else "off"}}

    def _live_distributions(self) -> dict:
        prior = self.pending["prior"]
        progress = self.progress
        hits = progress["hits"]
        mask = [i <= progress["closed"] or hits[i] for i in range(len(self.plan))]
        evidence = sum(mask)
        out = {"prior": _probs(prior.log_p), "likelihood": None, "posterior": None,
               "evidence_flashes": evidence, "leader": None}
        likelihood = normalize_logits(self._keyboard_scores(hits, mask))
        post = self.fusion.combine(prior, PhysiologyObservation(self.epoch_id, likelihood, "likelihood"))
        p_post = np.exp(post.log_p)
        order = np.argsort(-p_post)
        top, second = int(order[0]), int(order[1])
        confidence = float(p_post[top])
        out["posterior"] = _probs(post.log_p)
        if evidence:
            out["likelihood"] = _probs(likelihood)
        out["leader"] = {"symbol": self.rep.symbols[top], "confidence": round(confidence, 4),
                         "margin": round(confidence - float(p_post[second]), 4),
                         "below_tau": confidence < float(self.cfg.detector["tau"])}
        return out

    def _stats(self) -> dict:
        now = self.ended_wall or time.monotonic()
        active = 0.0
        if self.first_epoch_wall is not None:
            paused = self.paused_total + (now - self.pause_started if self.pause_started else 0.0)
            active = max(0.0, now - self.first_epoch_wall - paused)
        decided = [row for row in self.log if row["decided"] is not None]
        judged = [row for row in decided if row["correct"] is not None]
        matching = sum(a == b for a, b in zip(self.primary.value, self.target)) if self.target else None
        minutes = active / 60
        return {"epochs": len(self.log), "selected": len(decided),
                "repeats": sum(row["decided"] is None for row in self.log),
                "correct": sum(bool(row["correct"]) for row in judged),
                "accuracy": round(sum(bool(row["correct"]) for row in judged) / len(judged), 4) if judged else None,
                "matching_chars": matching, "active_s": round(active, 1),
                "speed_cpm": round(matching / minutes, 3) if matching is not None and minutes > 0.05 else None,
                "selected_cpm": round(len(decided) / minutes, 3) if minutes > 0.05 else None}

    def monitor(self) -> dict:
        """Everything the experimenter window shows."""
        with self.lock:
            state = self.snapshot()
            current = None
            if not self.finished:
                flash = self.progress["flash"]
                current = {"epoch_id": self.epoch_id, "position": len(self.primary.value),
                           "retry": self.retry, "target": self._current_target(),
                           "phase": self.progress["phase"], "flash": flash, "n_flashes": len(self.plan),
                           "flash_group": self.plan[flash] if 0 <= flash < len(self.plan) else None,
                           "hit_flashes": [i for i, h in enumerate(self.progress["hits"]) if h],
                           "hit_groups": [self.plan[i]["indices"] for i, h in enumerate(self.progress["hits"]) if h],
                           "words": self.pending["words"], "context_tail": self.pending["context_tail"],
                           "prior_ms": round(self.pending["prior_ms"], 2),
                           "words_ms": round(self.pending["words_ms"], 2),
                           **self._live_distributions()}
            state.update({
                "participant_id": self.cfg.participant_id, "experiment_id": self.cfg.experiment_id,
                "task_id": self.cfg.task_id, "profile": self.profile,
                "settings": settings.values_of(self.cfg), "llm": self.cfg.mode["llm"],
                "tau": float(self.cfg.detector["tau"]), "detector_mode": self.cfg.detector["mode"],
                "directory": str(self.storage.directory), "current": current,
                "last_epoch": self.last_epoch, "log": self.log[-60:],
                "corrections": self.corrections[-12:], "stats": self._stats(),
            })
            return state


class SessionManager:
    """Holds the active lab session and wakes long-polling windows on every change."""

    PARTICIPANT_GRACE_S = 4.0

    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.sessions: dict[str, InteractiveSession] = {}
        self.active: InteractiveSession | None = None
        self.lock = threading.Lock()
        self.cond = threading.Condition()
        self.version = 0
        self.control_version = 0
        self.participant_waiting = 0
        self.participant_seen = 0.0
        self.models: dict[str, object] = {}
        self.models_lock = threading.Lock()

    # ------------------------------------------------------------ versioning
    def bump(self, control: bool = False) -> None:
        with self.cond:
            self.version += 1
            if control:
                self.control_version += 1
            self.cond.notify_all()

    def participant_connected(self) -> bool:
        return (self.participant_waiting > 0
                or time.monotonic() - self.participant_seen < self.PARTICIPANT_GRACE_S)

    def wait(self, since: int, timeout: float, role: str = "operator") -> None:
        """Block until something changes after ``since`` or the timeout expires."""
        participant = role == "participant"
        with self.cond:
            if participant:
                was_connected = self.participant_connected()
                self.participant_waiting += 1
                self.participant_seen = time.monotonic()
                if not was_connected:
                    self.version += 1           # tell the operator a participant window appeared
                    self.cond.notify_all()
            try:
                key = (lambda: self.control_version) if participant else (lambda: self.version)
                self.cond.wait_for(lambda: key() != since, timeout=timeout)
            finally:
                if participant:
                    self.participant_waiting -= 1
                    self.participant_seen = time.monotonic()

    def live(self, role: str = "operator") -> dict:
        session = self.active
        body = {"version": self.version, "control_version": self.control_version,
                "participant_connected": self.participant_connected(),
                "participant_windows": self.participant_waiting, "session": None}
        if session is not None:
            body["session"] = session.snapshot() if role == "participant" else session.monitor()
        return body

    # --------------------------------------------------------------- models
    def model(self, kind: str, cfg: dict):
        key = kind + ":" + json.dumps(cfg, sort_keys=True, ensure_ascii=False)
        with self.models_lock:
            if key not in self.models:
                self.models[key] = _create_model(kind, cfg)
            return self.models[key]

    # -------------------------------------------------------------- control
    def start(self, config_name: str, target: str = "", **kwargs) -> dict:
        with self.lock:
            if self.active is not None and not self.active.finished:
                raise ValueError("stop the running session first")
            session = InteractiveSession(config_name, target, self.output_root,
                                         model_factory=self.model, **kwargs)
            self.sessions[session.storage.session_id] = session
            self.active = session
        self.bump(control=True)
        return session.monitor()

    def get(self, session_id: str) -> InteractiveSession:
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise ValueError("unknown session")
        return session

    def submit(self, session_id: str, hits, epoch_id=None) -> dict:
        try:
            return self.get(session_id).submit(hits, epoch_id)
        finally:
            self.bump(control=True)

    def progress(self, session_id: str, epoch_id, event: str, flash_id=None) -> dict:
        changed = self.get(session_id).progress_event(epoch_id, event, flash_id)
        if changed:
            self.bump()
        return {"accepted": changed}

    def pause(self, session_id: str, paused: bool) -> dict:
        session = self.get(session_id)
        session.set_paused(paused)
        self.bump(control=True)
        return session.monitor()

    def stop(self, session_id: str, reason: str = "user_stopped") -> dict:
        session = self.get(session_id)
        session.stop(reason)
        self.bump(control=True)
        return session.monitor()
