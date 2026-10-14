"""Speller core for removable stimulus/decoder modules (protocol ``speller-link@1``).

Per epoch the core:
1. computes the language prior and sends ``epoch_start`` (symbols, typed text, suggested timing);
2. logs every flash marker it receives, but never draws or schedules stimuli itself;
3. fuses every likelihood update from a decoder with the prior;
4. sends ``stop`` once the posterior confidence reaches ``external.stop_confidence``
   (dynamic stopping; modules may ignore it);
5. decides on a final likelihood (or after ``stop``/``done`` grace periods), sends
   ``result`` so the module can show the chosen symbol, waits ``stimulus.pause_ms``.

The task text never leaves the core: modules see only the symbols and the typed text.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from eeg_speller.core.clock import Clock
from eeg_speller.core.distributions import Repertoire, SymbolDistribution, log_probs, normalize_logits
from eeg_speller.decoding.detector import SymbolDetector
from eeg_speller.epoch.fusion import Fusion
from eeg_speller.external.transport import Incoming, LSLTransport, PROTOCOL
from eeg_speller.ext.word_prediction import WordPrediction
from eeg_speller.llm.fast import create_fast
from eeg_speller.llm.slow import CorrectionScheduler, create_slow, prompt_hash, validate_correction
from eeg_speller.physiology.base import PhysiologyObservation, validate_observation
from eeg_speller.recording.recorder import SessionRecorder
from eeg_speller.recording.storage import SessionStorage
from eeg_speller.text.primary import PrimaryText
from eeg_speller.text.secondary import SecondaryText

PROJECT = Path(__file__).resolve().parents[2]
CORPUS = PROJECT / "data" / "corpus_ru_fallback.txt"
LOG_LIMIT = 500

DEFAULTS = {
    "stop_confidence": 0.99,          # None disables the stop signal
    "min_updates_before_stop": 12,
    "stop_grace_s": 1.5,              # after "stop": wait for a final likelihood
    "decode_wait_s": 2.0,             # after "done": wait for a final likelihood
    "epoch_timeout_s": 120.0,
    "wait_modules_s": 60.0,           # before the first epoch: wait for a listening module
    "send_text": True,                # include typed text in epoch_start / result
}


def likelihood_vector(values, symbols) -> np.ndarray:
    """Turn a decoder message into normalized log-likelihoods over ``symbols``.

    Lists follow the symbol order. In a dict, symbols that are missing share the
    remaining probability mass equally (or a tiny floor if nothing remains).
    """
    n = len(symbols)
    if isinstance(values, list):
        if len(values) != n:
            raise ValueError(f"likelihood list must have {n} values")
        p = np.asarray(values, dtype=float)
    elif isinstance(values, dict):
        unknown = set(values) - set(symbols)
        if unknown:
            raise ValueError(f"unknown symbols in likelihood: {sorted(unknown)[:5]}")
        p = np.full(n, np.nan)
        for key, value in values.items():
            p[symbols.index(key)] = float(value)
        missing = np.isnan(p)
        if missing.any():
            given = np.nansum(p)
            rest = max(0.0, 1.0 - given) if given <= 1.0 else 0.0
            p[missing] = rest / missing.sum() if rest > 0 else 0.0
    else:
        raise ValueError("likelihood must be a list or an object {symbol: value}")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("likelihood values must be finite, non-negative and not all zero")
    p = p / p.sum()
    return normalize_logits(np.log(p + 1e-9))


def _probs(log_p) -> list[float]:
    return [round(float(x), 5) for x in np.exp(log_p)]


class ExternalSession:
    """One typing session driven by external modules."""

    def __init__(self, cfg, target: str, output_root: Path, *, transport=None, model_factory=None,
                 profile: str = ""):
        self.cfg = cfg
        self.profile = profile
        self.opts = {**DEFAULTS, **(getattr(cfg, "external", None) or {})}
        self.rep = Repertoire(tuple(cfg.repertoire["symbols"]), tuple(cfg.repertoire["layout"]))
        if len(target) > 200 or any(ch not in self.rep.symbols or ch == "⌫" for ch in target):
            raise ValueError("target must contain at most 200 repertoire characters")
        self.target = target
        self.primary, self.secondary = PrimaryText(), SecondaryText()
        self.detector = SymbolDetector(cfg.detector, self.rep)
        self.fusion = Fusion(cfg.fusion, cfg.mode["llm"])
        make = model_factory or default_model_factory
        llm_on = cfg.mode["llm"] == "on"
        self.fast = make("fast", cfg.fast_llm) if llm_on else None
        self.slow = make("slow", cfg.slow_llm) if llm_on else None
        self.words = WordPrediction(self.fast, cfg.word_prediction) if self.fast else None
        self.scheduler = CorrectionScheduler(cfg.slow_llm) if self.slow else None
        models = {"fast_llm": {"backend": cfg.fast_llm["backend"] if self.fast else "uniform",
                               "model_id": self.fast.model_id if self.fast else "uniform"},
                  "slow_llm": {"backend": cfg.slow_llm["backend"] if self.slow else "disabled",
                               "model_id": self.slow.model_id if self.slow else "disabled"},
                  "physiology": {"backend": "external", "model_id": PROTOCOL}}
        self.storage = SessionStorage(output_root, cfg, self.rep, models)
        self.session_id = self.storage.session_id
        self.transport = transport or LSLTransport(self.session_id)
        self.clock = Clock(live=True)
        self.rec = SessionRecorder(self.storage.directory, self.session_id, cfg.participant_id, self.clock)
        self.rec.emit("Session Recorder", "session_started",
                      {"run_mode": "live", "llm_mode": cfg.mode["llm"], "paradigm": "external",
                       "seed": cfg.seed, "config_hash": cfg.digest(), "protocol": PROTOCOL})
        self.rec.emit("Session Recorder", "task_assigned", {"task_id": cfg.task_id, "target_text": target})
        self.lock = threading.RLock()
        self.changed = threading.Condition()
        self.version = 0
        self.epoch_id = 0
        self.target_index = 0
        self.retry = 0
        self.finished = False
        self.end_reason = None
        self.error = None
        self.paused = False
        self.stop_requested = None
        self.phase = "waiting_modules"
        self.current = None
        self.log: list[dict] = []
        self.corrections: list[dict] = []
        self.future: dict[int, list[Incoming]] = {}
        self.flash_counts: dict[int, int] = {}
        self.sync = {"flashes": 0}
        self.started_wall = None
        self.ended_wall = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    # ---------------------------------------------------------------- public
    def start(self) -> "ExternalSession":
        self.thread.start()
        return self

    def request_stop(self, reason: str = "operator_stopped") -> None:
        self.stop_requested = reason
        self._bump()

    def set_paused(self, paused: bool) -> None:
        with self.lock:
            if self.finished or paused == self.paused:
                return
            self.paused = paused
            self.transport.send({"cmd": "pause" if paused else "resume"})
            self.rec.emit("Session Recorder", "session_paused" if paused else "session_resumed",
                          {"next_epoch_id": self.epoch_id})
        self._bump()

    def wait_finished(self, timeout: float | None = None) -> bool:
        self.thread.join(timeout)
        return self.finished

    # ------------------------------------------------------------------ loop
    def _run(self):
        try:
            self._wait_modules()
            while not self._should_end():
                self._wait_unpaused()
                if self._should_end():
                    break
                self._run_epoch()
                if self._should_end():
                    break
                self._sleep(self.cfg.stimulus["pause_ms"] / 1000)
        except Exception as exc:              # keep the recording consistent on any failure
            self.error = f"{type(exc).__name__}: {exc}"
            with self.lock:
                self.rec.emit("One Epoch", "error", {"module": "external", "message": self.error}, self.epoch_id)
        finally:
            self._finish()

    def _should_end(self) -> bool:
        if self.stop_requested:
            return True
        if self.target and self.target_index >= len(self.target):
            self.stop_requested = "target_finished"
            return True
        return False

    def _sleep(self, seconds: float):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not self.stop_requested:
            self._drain(min(0.05, max(0.0, deadline - time.monotonic())))

    def _wait_unpaused(self):
        while self.paused and not self.stop_requested:
            self._drain(0.05)

    def _wait_modules(self):
        self.phase = "waiting_modules"
        self._bump()
        deadline = time.monotonic() + float(self.opts["wait_modules_s"])
        while not self.transport.consumers():
            if self.stop_requested:
                return
            if time.monotonic() > deadline:
                raise TimeoutError("no stimulus module connected to SpellerControl")
            self._drain(0.1)
        time.sleep(0.3)                 # let the module subscribe before the first command

    def _drain(self, timeout: float) -> list[Incoming]:
        """Read incoming messages; log flashes; return likelihoods/done for any epoch."""
        items = self.transport.poll(timeout)
        out = []
        for item in items:
            if item.kind == "stimuli" and "flash" in item.msg:
                self._log_flash(item)
            else:
                out.append(item)
        return out

    # ----------------------------------------------------------------- epoch
    def _prior(self) -> tuple[SymbolDistribution, float, list, float]:
        start = time.perf_counter_ns()
        if self.fast is None:
            prior = SymbolDistribution(self.rep, log_probs(np.ones(len(self.rep.symbols))), "uniform",
                                       len(self.primary.value))
        else:
            context = (self.primary.value if self.cfg.fast_llm["context_source"] == "primary"
                       else self.secondary.value)
            prior = self.fast.predict(context, self.rep)
        prior_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        words = self.words.predict(self.primary.value) if self.words else []
        return prior, prior_ms, words, (time.perf_counter_ns() - start) / 1e6

    def _target_symbol(self):
        return self.target[self.target_index] if self.target and self.target_index < len(self.target) else None

    def _run_epoch(self):
        eid, position = self.epoch_id, len(self.primary.value)
        prior, prior_ms, words, words_ms = self._prior()
        settings = {"paradigm": self.cfg.stimulus["paradigm"], "flash_ms": self.cfg.stimulus["flash_ms"],
                    "isi_ms": self.cfg.stimulus["isi_ms"], "n_repetitions": self.cfg.epoch["n_repetitions"]}
        with self.lock:
            if self.started_wall is None:
                self.started_wall = time.monotonic()
            self.rec.emit("One Epoch", "epoch_started",
                          {"position": position,
                           "repertoire_hash": hashlib.sha256("".join(self.rep.symbols).encode()).hexdigest(),
                           "paradigm": "external", **settings, "pause_ms": self.cfg.stimulus["pause_ms"],
                           "retry_idx": self.retry, "target_symbol": self._target_symbol(),
                           "stop_confidence": self.opts["stop_confidence"]}, eid)
            self.rec.emit("Source Model (Fast LLM)", "prior_computed",
                          {"model_id": self.fast.model_id if self.fast else "uniform",
                           "context_source": self.cfg.fast_llm["context_source"],
                           "context_tail": self.primary.value[-64:], "context_len": position,
                           "log_p": prior.log_p, "wall_latency_ms": prior_ms}, eid)
            if self.words:
                self.rec.emit("ext:word_prediction", "word_candidates",
                              {"prefix": self.primary.value.split(" ")[-1], "candidates": words,
                               "model_id": self.fast.model_id, "wall_latency_ms": words_ms}, eid)
            self.current = {"epoch_id": eid, "position": position, "target": self._target_symbol(),
                            "prior": prior, "obs": None, "post": None, "updates": 0,
                            "flashes": self.flash_counts.get(eid, 0),
                            "stop_sent": False, "words": words, "prior_ms": prior_ms,
                            "began": time.monotonic()}
            self.phase = "presenting"
            msg = {"cmd": "epoch_start", "protocol": PROTOCOL, "epoch": eid,
                   "symbols": list(self.rep.symbols), "layout": list(self.rep.layout), "settings": settings}
            if self.opts["send_text"]:
                msg["text"] = self.primary.value
            self.transport.send(msg)
        self._bump()

        final = None
        stop_at = done_at = None
        began = time.monotonic()
        pending = self.future.pop(eid, [])
        while True:
            items = pending + self._drain(0.02)
            pending = []
            for item in items:
                epoch = item.msg.get("epoch")
                if not isinstance(epoch, int) or epoch < eid:
                    continue                                   # stale message
                if epoch > eid:
                    self.future.setdefault(epoch, []).append(item)
                    continue
                if item.kind == "stimuli" and item.msg.get("event") == "done":
                    done_at = done_at or time.monotonic()
                elif item.kind == "likelihood":
                    if self._on_likelihood(item):
                        final = item
                    if self._ready_to_stop() and stop_at is None:
                        stop_at = time.monotonic()
                        self._send_stop()
            now = time.monotonic()
            cur = self.current
            if final is not None:
                return self._decide("final_likelihood")
            if cur["obs"] is not None:
                if stop_at and now - stop_at > float(self.opts["stop_grace_s"]):
                    return self._decide("stop_grace_elapsed")
                if done_at and now - done_at > float(self.opts["decode_wait_s"]):
                    return self._decide("done_grace_elapsed")
            elif done_at and now - done_at > float(self.opts["decode_wait_s"]) * 3:
                return self._abort("no_likelihood")
            if now - began > float(self.opts["epoch_timeout_s"]):
                return self._decide("timeout") if cur["obs"] is not None else self._abort("timeout")
            if self.stop_requested:
                return self._abort(self.stop_requested)

    def _on_likelihood(self, item: Incoming) -> bool:
        # IMPROV[Q-35]
        # ASSUMPTION[MA-03]
        cur = self.current
        try:
            log_p = likelihood_vector(item.msg.get("p", item.msg.get("log_p")), list(self.rep.symbols))
            obs = PhysiologyObservation(cur["epoch_id"], log_p, "likelihood",
                                        module_metrics={"usable": bool(item.msg.get("usable", True)),
                                                        "module": item.module, "final": bool(item.msg.get("final")),
                                                        "info": item.msg.get("info")},
                                        t_module_ns=round(item.lsl_time * 1e9))
            validate_observation(obs, cur["epoch_id"], len(self.rep.symbols))
        except (ValueError, TypeError) as exc:
            with self.lock:
                self.rec.emit("EEG / EOG source", "warning",
                              {"module": item.module, "message": f"rejected likelihood: {exc}"}, cur["epoch_id"])
            return False
        post = self.fusion.combine(cur["prior"], obs)
        with self.lock:
            cur.update(obs=obs, post=post, updates=cur["updates"] + 1)
            self.rec.emit("EEG / EOG source", "physiology_observation",
                          {"log_p": obs.log_p, "semantics": "likelihood", "source": item.module,
                           "module_metrics": obs.module_metrics, "module_time_ns": obs.t_module_ns,
                           "update": cur["updates"]}, cur["epoch_id"])
        self._bump()
        return bool(item.msg.get("final"))

    def _ready_to_stop(self) -> bool:
        # ASSUMPTION[MA-34]
        # IMPROV[Q-36]
        cur = self.current
        threshold = self.opts["stop_confidence"]
        if threshold is None or cur["post"] is None or cur["stop_sent"]:
            return False
        if cur["updates"] < int(self.opts["min_updates_before_stop"]):
            return False
        usable = (cur["obs"].module_metrics or {}).get("usable", True)
        return usable and float(np.exp(cur["post"].log_p).max()) >= float(threshold)

    def _send_stop(self):
        cur = self.current
        p = np.exp(cur["post"].log_p)
        msg = {"cmd": "stop", "epoch": cur["epoch_id"], "reason": "confident",
               "confidence": round(float(p.max()), 4)}
        with self.lock:
            cur["stop_sent"] = True
            self.transport.send(msg)
            self.rec.emit("Presentation Method", "epoch_stop_requested",
                          {"confidence": msg["confidence"], "threshold": self.opts["stop_confidence"],
                           "updates": cur["updates"], "flashes": cur["flashes"],
                           "elapsed_s": round(time.monotonic() - cur["began"], 3)}, cur["epoch_id"])
        self._bump()

    def _decide(self, why: str):
        cur = self.current
        eid, position = cur["epoch_id"], cur["position"]
        with self.lock:
            post = cur["post"]
            self.rec.emit("Statistical Inference", "posterior_computed",
                          {"fusion": post.fusion, "params": post.params, "log_p": post.log_p,
                           "wall_latency_ms": 0.0, "decided_on": why, "updates": cur["updates"]}, eid)
            decision = self.detector.decide(post, position, self.retry)
            usable = bool((cur["obs"].module_metrics or {}).get("usable", True))
            symbol = decision.symbol if usable else None
            uncertain = decision.uncertain or not usable
            self.rec.emit("Primary Symbol Detector", "symbol_decided",
                          {"symbol": symbol, "confidence": decision.confidence, "margin": decision.margin,
                           "uncertain": uncertain, "cold_start": decision.cold_start,
                           "abstained": symbol is None}, eid)
            target = cur["target"]
            if symbol is not None:
                self.primary.apply(symbol)
                self.secondary.synchronize(self.primary.value)
                if self.target:
                    self.target_index = (max(0, self.target_index - 1) if symbol == "⌫"
                                         else min(len(self.target), self.target_index + 1))
                self.rec.emit("Primary Text", "primary_text_updated",
                              {"op": "delete" if symbol == "⌫" else "append", "symbol": symbol,
                               "position": position, "text": self.primary.value}, eid)
                self._correct(eid)
                self.retry = 0
            else:
                self.retry += 1
            result = {"cmd": "result", "epoch": eid, "symbol": symbol,
                      "confidence": round(decision.confidence, 4), "uncertain": uncertain}
            if self.opts["send_text"]:
                result.update(text=self.primary.value, corrected=self.secondary.value)
            self.transport.send(result)
            self.rec.emit("Influence on presentation method", "presentation_feedback",
                          {"decided": symbol, "top_k": self._top(post, 5), "sent": "result"}, eid)
            self.rec.emit("ext:ui", "ui_rendered",
                          {"primary_text": self.primary.value, "secondary_text": self.secondary.value,
                           "candidates_shown": self._top(post, 3), "status": "selected" if symbol else "retry"}, eid)
            self.log.append({"epoch_id": eid, "position": position, "target": target, "decided": symbol,
                             "confidence": round(decision.confidence, 4), "uncertain": bool(uncertain),
                             "correct": None if target is None or symbol is None else symbol == target,
                             "flashes": cur["flashes"], "updates": cur["updates"],
                             "stopped_early": cur["stop_sent"], "decided_on": why,
                             "duration_s": round(time.monotonic() - cur["began"], 2)})
            del self.log[:-LOG_LIMIT]
            self.epoch_id += 1
            self.phase = "result"
        self._bump()

    def _abort(self, reason: str):
        with self.lock:
            self.rec.emit("One Epoch", "epoch_aborted", {"reason": reason}, self.current["epoch_id"])
            if reason in ("no_likelihood", "timeout"):
                self.transport.send({"cmd": "result", "epoch": self.current["epoch_id"], "symbol": None,
                                     "reason": reason})
                self.log.append({"epoch_id": self.current["epoch_id"], "position": self.current["position"],
                                 "target": self.current["target"], "decided": None, "confidence": None,
                                 "uncertain": True, "correct": None, "flashes": self.current["flashes"],
                                 "updates": 0, "stopped_early": False, "decided_on": reason,
                                 "duration_s": round(time.monotonic() - self.current["began"], 2)})
                self.epoch_id += 1
                self.retry += 1
            self.phase = "result"
        self._bump()

    # ------------------------------------------------------------- logging
    def _log_flash(self, item: Incoming):
        # STUB[Q-37]
        msg = item.msg
        symbols = msg.get("flash")
        if not isinstance(symbols, list) or not all(isinstance(s, str) for s in symbols):
            return
        indices = [self.rep.symbols.index(s) for s in symbols if s in self.rep.symbols]
        epoch = msg.get("epoch") if isinstance(msg.get("epoch"), int) else None
        with self.lock:
            if epoch is not None:
                self.flash_counts[epoch] = self.flash_counts.get(epoch, 0) + 1
            cur = self.current
            if cur is not None and epoch == cur["epoch_id"]:
                cur["flashes"] = self.flash_counts[epoch]
            self.sync["flashes"] += 1
            self.sync["last_flash_lsl"] = item.lsl_time
            # The onset is the sender's LSL timestamp (taken at the flip) mapped into
            # this computer's LSL clock; t_ns stays on the recorder's monotonic clock.
            self.rec.emit("Presentation Method", "stimulus_event",
                          {"symbols": symbols, "indices": indices, "module": item.module,
                           "onset_lsl_s": item.lsl_time, "timing_method": "module_flip_lsl_timestamp"},
                          epoch)

    # ---------------------------------------------------------- correction
    def _correct(self, epoch_id: int) -> None:
        req = self.scheduler.on_primary_update(self.primary.value) if self.scheduler else None
        if req is None or not req.words:
            return
        self.rec.emit("Post-factum Text Correction", "correction_requested",
                      {"correction_id": req.correction_id, "trigger": req.trigger,
                       "window_words": req.words, "window_start_word": req.start_word,
                       "left_context_len": len(req.left_context), "model_id": self.slow.model_id,
                       "prompt_hash": prompt_hash(PROJECT / self.cfg.slow_llm["prompt_file"])}, epoch_id)
        start = time.perf_counter_ns()
        try:
            result = self.slow.correct(req.left_context, req.words)
            accepted, reason = validate_correction(req.words, result, self.cfg.slow_llm["strict_word_count"])
        except Exception as exc:
            result, accepted, reason = req.words, False, type(exc).__name__
        latency = (time.perf_counter_ns() - start) / 1e6
        self.rec.emit("Post-factum Text Correction", "correction_completed",
                      {"correction_id": req.correction_id, "words_in": req.words, "words_out": result,
                       "accepted": accepted, "reject_reason": reason, "wall_latency_ms": latency}, epoch_id)
        self.corrections.append({"correction_id": req.correction_id, "words_in": list(req.words),
                                 "words_out": list(result), "accepted": accepted,
                                 "changed": accepted and list(result) != list(req.words)})
        del self.corrections[:-LOG_LIMIT]
        if accepted:
            self.secondary.replace_window(result, req.start_word)
            self.rec.emit("Secondary (Corrected) Text", "secondary_text_updated",
                          {"correction_id": req.correction_id, "text": self.secondary.value,
                           "frozen_upto_word": self.secondary.frozen_upto_word}, epoch_id)

    # --------------------------------------------------------------- finish
    def _finish(self):
        with self.lock:
            if self.finished:
                return
            reason = self.stop_requested or ("error" if self.error else "ended")
            try:
                self.transport.send({"cmd": "end", "reason": reason})
                time.sleep(0.2)
            except Exception:
                pass
            if self.current is not None and self.phase == "presenting":
                self.rec.emit("One Epoch", "epoch_aborted", {"reason": reason}, self.current["epoch_id"])
            status = "failed" if self.error else "completed"
            self.rec.emit("Session Recorder", "session_ended",
                          {"status": status, "reason": reason, "n_epochs": self.epoch_id,
                           "n_events": self.rec.seq + 1})
            count = self.rec.seq
            self.rec.close()
            self.storage.finish(self.target, self.primary.value, self.secondary.value,
                                self.epoch_id, count, status=status)
            self.transport.close()
            self.end_reason = reason
            self.ended_wall = time.monotonic()
            self.finished = True
            self.phase = "finished"
        self._bump()

    # ------------------------------------------------------------ snapshots
    def _top(self, post, k):
        p = np.exp(post.log_p)
        return [[self.rep.symbols[int(i)], round(float(p[i]), 4)] for i in np.argsort(-p)[:k]]

    def _bump(self):
        with self.changed:
            self.version += 1
            self.changed.notify_all()

    def wait_change(self, since: int, timeout: float) -> None:
        with self.changed:
            self.changed.wait_for(lambda: self.version != since, timeout=timeout)

    def monitor(self) -> dict:
        """Compact state for the operator window."""
        with self.lock:
            cur = self.current if self.phase in ("presenting", "result") else None
            leader = None
            top = []
            if cur is not None and cur.get("post") is not None:
                p = np.exp(cur["post"].log_p)
                order = np.argsort(-p)
                top = [{"symbol": self.rep.symbols[int(i)], "p": round(float(p[i]), 4)} for i in order[:5]]
                leader = top[0]
            elif cur is not None:
                p = np.exp(cur["prior"].log_p)
                order = np.argsort(-p)
                top = [{"symbol": self.rep.symbols[int(i)], "p": round(float(p[i]), 4)} for i in order[:5]]
            decided = [r for r in self.log if r["decided"] is not None]
            judged = [r for r in decided if r["correct"] is not None]
            end = self.ended_wall or time.monotonic()
            minutes = ((end - self.started_wall) / 60) if self.started_wall else 0
            matching = sum(a == b for a, b in zip(self.primary.value, self.target)) if self.target else None
            last_flash = self.sync.get("last_flash_lsl")
            return {
                "session_id": self.session_id, "profile": self.profile,
                "participant_id": self.cfg.participant_id, "task_id": self.cfg.task_id,
                "llm": self.cfg.mode["llm"], "phase": self.phase, "paused": self.paused,
                "finished": self.finished, "end_reason": self.end_reason, "error": self.error,
                "target": self.target, "target_index": self.target_index,
                "text": self.primary.value, "corrected": self.secondary.value,
                "epoch_id": self.epoch_id,
                "epoch": None if cur is None else {
                    "epoch_id": cur["epoch_id"], "target": cur["target"], "flashes": cur["flashes"],
                    "updates": cur["updates"], "stop_sent": cur["stop_sent"],
                    "elapsed_s": round(time.monotonic() - cur["began"], 1), "leader": leader, "top": top,
                    "words": [w[0] if isinstance(w, (list, tuple)) else w for w in (cur["words"] or [])][:5]},
                "stop_confidence": self.opts["stop_confidence"], "tau": float(self.cfg.detector["tau"]),
                "modules": self.transport.status(),
                "stats": {"selected": len(decided), "epochs": len(self.log),
                          "accuracy": round(sum(bool(r["correct"]) for r in judged) / len(judged), 3) if judged else None,
                          "early_stops": sum(bool(r["stopped_early"]) for r in self.log),
                          "mean_epoch_s": round(float(np.mean([r["duration_s"] for r in self.log])), 2) if self.log else None,
                          "speed_cpm": round(matching / minutes, 2) if matching is not None and minutes > 0.05 else None},
                "log": self.log[-30:], "corrections": self.corrections[-8:],
                "directory": str(self.storage.directory),
            }


def default_model_factory(kind: str, cfg: dict):
    if kind == "fast":
        return create_fast(cfg, CORPUS)
    return create_slow(cfg, False, CORPUS)
