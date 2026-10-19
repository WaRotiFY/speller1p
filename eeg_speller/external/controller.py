"""Speller core for removable stimulus/decoder modules (protocol ``speller-link@1``).

Per epoch the core:
1. computes the language prior and sends ``epoch_start`` (keyboard symbols and layout,
   typed text, suggested timing). The keyboard is the one the stimulus module announced
   when it connected (``{"event": "keyboard"}``); without it the profile's keyboard is used;
2. logs every flash marker it receives, but never draws or schedules stimuli itself;
3. fuses every likelihood update from a decoder with the prior (plugin from inference/);
4. sends ``stop`` once the posterior confidence reaches ``external.stop_confidence``
   (dynamic stopping; modules may ignore it);
5. decides on a final likelihood (or after ``stop``/``done`` grace periods), sends
   ``result`` so the module can show the chosen symbol, waits ``stimulus.pause_ms``.

The slow model corrects the text in a background thread, so a cloud model never delays
the next epoch. Every recorded event carries the LSL time (``lsl_s``), the clock of the
EEG and of the flash markers.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import queue
import re
import threading
import time

import numpy as np

from eeg_speller.core.clock import Clock
from eeg_speller.core.distributions import Repertoire, SymbolDistribution, log_probs, normalize_logits
from eeg_speller.core.keyboard import BACKSPACE, typed_char, validate_keyboard
from eeg_speller.decoding.detector import SymbolDetector
from eeg_speller.epoch.plugins import PluginFusion
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
WORD = re.compile(r"[а-я]+")

DEFAULTS = {
    "stop_confidence": 0.99,          # None disables the stop signal
    "min_updates_before_stop": 12,
    "stop_grace_s": 1.5,              # after "stop": wait for a final likelihood
    "decode_wait_s": 2.0,             # after "done": wait for a final likelihood
    "epoch_timeout_s": 120.0,
    "wait_modules_s": 60.0,           # before the first epoch: wait for a listening module
    "send_text": True,                # include typed text in epoch_start / result
    "keyboard_wait_s": 5.0,           # after a module connects: wait for its keyboard
    "top_k_display": 4,
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


def _round(x, nd=6):
    return None if x is None else round(float(x), nd)


class ExternalSession:
    """One typing session driven by external modules."""

    def __init__(self, cfg, target: str, output_root: Path, *, transport=None, model_factory=None,
                 profile: str = "", passport: dict | None = None, replay_of: str | None = None):
        self.cfg = cfg
        self.profile = profile
        self.opts = {**DEFAULTS, **(getattr(cfg, "external", None) or {})}
        self.rep = Repertoire(tuple(cfg.repertoire["symbols"]), tuple(cfg.repertoire["layout"]))
        self.typed = {s: typed_char(s) for s in self.rep.symbols}
        if len(target) > 200:
            raise ValueError("target must contain at most 200 characters")
        self.target = target
        self.keyboard_source = "profile"
        self.pending_keyboard = None
        self.primary, self.secondary = PrimaryText(), SecondaryText()
        self.char_post: list[np.ndarray | None] = []      # posterior of every typed character
        self.detector = SymbolDetector(cfg.detector, self.rep)
        self.fusion = PluginFusion(cfg.fusion, cfg.mode["llm"])
        make = model_factory or default_model_factory
        llm_on = cfg.mode["llm"] == "on"
        slow_on = llm_on and cfg.slow_llm.get("backend") not in (None, "", "off")
        self.fast = make("fast", cfg.fast_llm) if llm_on else None
        self.slow = make("slow", {**cfg.slow_llm, "_allow_remote": bool(cfg.llm.get("allow_remote"))}) \
            if slow_on else None
        self.words = WordPrediction(self.fast, cfg.word_prediction) if self.fast else None
        self.scheduler = CorrectionScheduler(cfg.slow_llm) if self.slow else None
        models = {"fast_llm": {"backend": cfg.fast_llm["backend"] if self.fast else "uniform",
                               "model_id": self.fast.model_id if self.fast else "uniform",
                               "weights_hash": getattr(self.fast, "weights_hash", None)},
                  "slow_llm": {"backend": cfg.slow_llm["backend"] if self.slow else "disabled",
                               "model_id": self.slow.model_id if self.slow else "disabled",
                               "provider": cfg.slow_llm.get("provider") if self.slow else None,
                               "prompt_hash": getattr(self.slow, "prompt_hash", None)},
                  "inference": {"plugin": self.fusion.name, "params": self.fusion.values,
                                "source_hash": self.fusion.plugin.source_hash},
                  "physiology": {"backend": "external", "model_id": PROTOCOL}}
        self.storage = SessionStorage(output_root, cfg, self.rep, models, replay_of=replay_of)
        self.storage.manifest["clock"]["kind"] = "monotonic"      # also in a replay: real time
        self.session_id = self.storage.session_id
        self.transport = transport or LSLTransport(self.session_id)
        self.clock = Clock(live=True)
        self.clock.lsl = self.transport.now
        self.passport = passport or {}
        self.rec = SessionRecorder(self.storage.directory, self.session_id, cfg.participant_id, self.clock)
        self.rec.emit("Session Recorder", "session_started",
                      {"run_mode": "replay" if replay_of else "live", "replay_of": replay_of, "llm_mode": cfg.mode["llm"], "paradigm": "external",
                       "seed": cfg.seed, "config_hash": cfg.digest(), "protocol": PROTOCOL,
                       "keyboard": {"symbols": list(self.rep.symbols), "layout": list(self.rep.layout)},
                       "lsl_clock_at_t0_s": _round(self.transport.now()), "passport": self.passport})
        self.rec.emit("Session Recorder", "task_assigned", {"task_id": cfg.task_id, "target_text": target})
        if self.slow is not None and getattr(self.slow, "template", None):
            (self.storage.directory / "external" / "slow_llm_prompt.txt").write_text(self.slow.template,
                                                                                   encoding="utf-8")
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
        self.correction_queue: "queue.Queue" = queue.Queue()
        self.correction_thread = threading.Thread(target=self._correction_worker, daemon=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.passport.setdefault("keyboard", f"{self.rep.layout[0]}×{self.rep.layout[1]} (профиль)")

    # ---------------------------------------------------------------- public
    def start(self) -> "ExternalSession":
        if self.slow is not None:
            self.correction_thread.start()
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
        # The module announces its keyboard right after connecting; wait for it a little.
        until = time.monotonic() + float(self.opts["keyboard_wait_s"])
        while self.keyboard_source == "profile" and self.pending_keyboard is None and time.monotonic() < until:
            if self.stop_requested:
                return
            self._drain(0.05)
        if self.keyboard_source == "profile" and self.pending_keyboard is None:
            with self.lock:
                self.rec.emit("Presentation Method", "warning",
                              {"module": "external", "message": "the module did not announce a keyboard: "
                                                                "the profile keyboard is used"}, None)
        self._apply_pending_keyboard()
        time.sleep(0.3)               # let the module subscribe before the first command

    def _drain(self, timeout: float) -> list[Incoming]:
        """Read incoming messages; log flashes; return likelihoods/done for any epoch."""
        items = self.transport.poll(timeout)
        out = []
        for item in items:
            if item.kind == "stimuli" and "flash" in item.msg:
                self._log_flash(item)
            elif item.kind == "stimuli" and item.msg.get("event") == "keyboard":
                self._receive_keyboard(item)
            else:
                out.append(item)
        return out

    # -------------------------------------------------------------- keyboard
    def _receive_keyboard(self, item: Incoming):
        # IMPROV[Q-38]
        symbols, layout = item.msg.get("symbols"), item.msg.get("layout")
        try:
            validate_keyboard(symbols, layout or [0, 0])
        except (ValueError, TypeError) as exc:
            with self.lock:
                self.rec.emit("Presentation Method", "warning",
                              {"module": item.module, "message": f"rejected keyboard: {exc}"}, None)
            return
        self.pending_keyboard = (tuple(symbols), (int(layout[0]), int(layout[1])), item.module)
        if self.phase != "presenting":
            self._apply_pending_keyboard()

    def _apply_pending_keyboard(self):
        """Switch to the module's keyboard between epochs."""
        if self.pending_keyboard is None:
            return
        symbols, layout, module = self.pending_keyboard
        self.pending_keyboard = None
        with self.lock:
            self.rep = Repertoire(symbols, layout)
            self.typed = {s: typed_char(s) for s in symbols}
            self.detector = SymbolDetector(self.cfg.detector, self.rep)
            self.keyboard_source = module
            self.passport["keyboard"] = f"{layout[0]}×{layout[1]} ({module})"
            self.storage.manifest["repertoire"] = list(symbols)
            typeable = {c for s, c in self.typed.items() if s != BACKSPACE}
            missing = sorted({ch for ch in self.target if ch not in typeable})
            self.rec.emit("Presentation Method", "keyboard_set",
                          {"module": module, "symbols": list(symbols), "layout": list(layout),
                           "task_characters_missing": missing}, None)
            if missing:
                self.rec.emit("Presentation Method", "warning",
                              {"module": module, "message": f"the task has characters missing on the keyboard: {missing}"},
                              None)
        self._bump()

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

    def _target_char(self):
        return self.target[self.target_index] if self.target and self.target_index < len(self.target) else None

    def _run_epoch(self):
        self._apply_pending_keyboard()
        eid, position = self.epoch_id, len(self.primary.value)
        t_epoch = self.transport.now()
        prior, prior_ms, words, words_ms = self._prior()
        settings = {"paradigm": self.cfg.stimulus["paradigm"], "flash_ms": self.cfg.stimulus["flash_ms"],
                    "isi_ms": self.cfg.stimulus["isi_ms"], "n_repetitions": self.cfg.epoch["n_repetitions"]}
        target_char = self._target_char()
        with self.lock:
            if self.started_wall is None:
                self.started_wall = time.monotonic()
            self.rec.emit("One Epoch", "epoch_started",
                          {"position": position,
                           "repertoire_hash": hashlib.sha256("".join(self.rep.symbols).encode()).hexdigest(),
                           "paradigm": "external", **settings, "pause_ms": self.cfg.stimulus["pause_ms"],
                           "retry_idx": self.retry, "target_symbol": target_char,
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
            self.current = {"epoch_id": eid, "position": position, "target": target_char,
                            "prior": prior, "obs": None, "post": None, "updates": 0,
                            "flashes": self.flash_counts.get(eid, 0),
                            "stop_sent": False, "words": words, "prior_ms": prior_ms,
                            "began": time.monotonic(),
                            "timeline": {"epoch_start_lsl": t_epoch, "prior_ready_lsl": self.transport.now(),
                                         "first_flash_lsl": None, "last_flash_lsl": None,
                                         "first_likelihood_sent_lsl": None, "first_likelihood_recv_lsl": None,
                                         "final_likelihood_sent_lsl": None, "final_likelihood_recv_lsl": None,
                                         "stop_sent_lsl": None, "done_recv_lsl": None,
                                         "decided_lsl": None, "result_sent_lsl": None}}
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
                    if done_at is None:
                        done_at = time.monotonic()
                        self.current["timeline"]["done_recv_lsl"] = item.recv_time
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
        raw = item.msg.get("p", item.msg.get("log_p"))
        try:
            log_p = likelihood_vector(raw, list(self.rep.symbols))
            obs = PhysiologyObservation(cur["epoch_id"], log_p, "likelihood",
                                        module_metrics={"usable": bool(item.msg.get("usable", True)),
                                                        "module": item.module, "final": bool(item.msg.get("final")),
                                                        "info": item.msg.get("info")},
                                        t_module_ns=round(item.lsl_time * 1e9))
            validate_observation(obs, cur["epoch_id"], len(self.rep.symbols))
        except (ValueError, TypeError) as exc:
            with self.lock:
                self.rec.emit("EEG / EOG source", "warning",
                              {"module": item.module, "message": f"rejected likelihood: {exc}",
                               "raw": raw if isinstance(raw, (dict, list)) else str(raw)[:200],
                               "sent_lsl_s": _round(item.lsl_time), "recv_lsl_s": _round(item.recv_time)},
                              cur["epoch_id"])
            return False
        # IMPROV[Q-45]
        start = time.perf_counter_ns()
        post = self.fusion.combine(cur["prior"], obs)
        fusion_ms = (time.perf_counter_ns() - start) / 1e6
        tl = cur["timeline"]
        if tl["first_likelihood_recv_lsl"] is None:
            tl["first_likelihood_sent_lsl"], tl["first_likelihood_recv_lsl"] = item.lsl_time, item.recv_time
        if obs.module_metrics["final"]:
            tl["final_likelihood_sent_lsl"], tl["final_likelihood_recv_lsl"] = item.lsl_time, item.recv_time
        with self.lock:
            cur.update(obs=obs, post=post, updates=cur["updates"] + 1, raw=raw)
            # The likelihood is stored twice: exactly as received (raw) and normalized (log_p).
            self.rec.emit("EEG / EOG source", "physiology_observation",
                          {"log_p": obs.log_p, "raw": raw, "semantics": "likelihood", "source": item.module,
                           "module_metrics": obs.module_metrics, "module_time_ns": obs.t_module_ns,
                           "sent_lsl_s": _round(item.lsl_time), "recv_lsl_s": _round(item.recv_time),
                           "transport_delay_ms": _round((item.recv_time - item.lsl_time) * 1000, 3),
                           "update": cur["updates"]}, cur["epoch_id"])
            if self.fusion.last_error:
                self.rec.emit("Statistical Inference", "warning",
                              {"module": self.fusion.name, "message": f"inference plugin failed, "
                               f"physiology only used: {self.fusion.last_error}"}, cur["epoch_id"])
            cur["fusion_ms"] = fusion_ms
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
            cur["timeline"]["stop_sent_lsl"] = self.transport.now()
            self.rec.emit("Presentation Method", "epoch_stop_requested",
                          {"confidence": msg["confidence"], "threshold": self.opts["stop_confidence"],
                           "updates": cur["updates"], "flashes": cur["flashes"],
                           "elapsed_s": round(time.monotonic() - cur["began"], 3)}, cur["epoch_id"])
        self._bump()

    def _apply_symbol(self, symbol: str, post) -> None:
        if symbol == BACKSPACE:
            if self.primary.value:
                self.primary.value = self.primary.value[:-1]
                if self.char_post:
                    self.char_post.pop()
        else:
            self.primary.value += self.typed[symbol]
            self.char_post.append(np.array(post.log_p))

    def _decide(self, why: str):
        cur = self.current
        eid, position = cur["epoch_id"], cur["position"]
        with self.lock:
            post = cur["post"]
            self.rec.emit("Statistical Inference", "posterior_computed",
                          {"fusion": post.fusion, "params": post.params, "log_p": post.log_p,
                           "wall_latency_ms": round(cur.get("fusion_ms", 0.0), 3), "decided_on": why,
                           "updates": cur["updates"]}, eid)
            decision = self.detector.decide(post, position, self.retry)
            usable = bool((cur["obs"].module_metrics or {}).get("usable", True))
            symbol = decision.symbol if usable else None
            uncertain = decision.uncertain or not usable
            cur["timeline"]["decided_lsl"] = self.transport.now()
            self.rec.emit("Primary Symbol Detector", "symbol_decided",
                          {"symbol": symbol, "typed": self.typed.get(symbol) if symbol else None,
                           "confidence": decision.confidence, "margin": decision.margin,
                           "uncertain": uncertain, "cold_start": decision.cold_start,
                           "abstained": symbol is None}, eid)
            target = cur["target"]
            if symbol is not None:
                self._apply_symbol(symbol, post)
                self.secondary.synchronize(self.primary.value)
                if self.target:
                    self.target_index = (max(0, self.target_index - 1) if symbol == BACKSPACE
                                         else min(len(self.target), self.target_index + 1))
                self.rec.emit("Primary Text", "primary_text_updated",
                              {"op": "delete" if symbol == BACKSPACE else "append", "symbol": symbol,
                               "position": position, "text": self.primary.value}, eid)
                self._request_correction(eid)
                self.retry = 0
            else:
                self.retry += 1
            result = {"cmd": "result", "epoch": eid, "symbol": symbol,
                      "confidence": round(decision.confidence, 4), "uncertain": uncertain}
            if self.opts["send_text"]:
                result.update(text=self.primary.value, corrected=self.secondary.value)
            self.transport.send(result)
            cur["timeline"]["result_sent_lsl"] = self.transport.now()
            self.rec.emit("Influence on presentation method", "presentation_feedback",
                          {"decided": symbol, "top_k": self._top(post.log_p, 5), "sent": "result"}, eid)
            self._emit_timeline(cur)
            self.rec.emit("ext:ui", "ui_rendered",
                          {"primary_text": self.primary.value, "secondary_text": self.secondary.value,
                           "candidates_shown": self._top(post.log_p, 3), "status": "selected" if symbol else "retry"},
                          eid)
            correct = None if target is None or symbol is None else self.typed[symbol] == target
            self.log.append({"epoch_id": eid, "position": position, "target": target, "decided": symbol,
                             "typed": self.typed.get(symbol) if symbol else None,
                             "top": self._top(post.log_p, int(self.opts["top_k_display"])),
                             "confidence": round(decision.confidence, 4), "uncertain": bool(uncertain),
                             "correct": correct, "flashes": cur["flashes"], "updates": cur["updates"],
                             "stopped_early": cur["stop_sent"], "decided_on": why,
                             "duration_s": round(time.monotonic() - cur["began"], 2)})
            del self.log[:-LOG_LIMIT]
            self.epoch_id += 1
            self.phase = "result"
        self._bump()

    def _emit_timeline(self, cur):
        tl = cur["timeline"]
        t0 = tl["epoch_start_lsl"]
        rel = {k.replace("_lsl", "_ms"): _round((v - t0) * 1000, 3) for k, v in tl.items()
               if v is not None and k != "epoch_start_lsl"}
        self.rec.emit("One Epoch", "epoch_timeline",
                      {"lsl_s": {k: _round(v) for k, v in tl.items()}, "since_epoch_start_ms": rel,
                       "flashes": cur["flashes"], "updates": cur["updates"]}, cur["epoch_id"])

    def _abort(self, reason: str):
        with self.lock:
            self.rec.emit("One Epoch", "epoch_aborted", {"reason": reason}, self.current["epoch_id"])
            if reason in ("no_likelihood", "timeout"):
                self.transport.send({"cmd": "result", "epoch": self.current["epoch_id"], "symbol": None,
                                     "reason": reason})
                self.current["timeline"]["result_sent_lsl"] = self.transport.now()
                self._emit_timeline(self.current)
                self.log.append({"epoch_id": self.current["epoch_id"], "position": self.current["position"],
                                 "target": self.current["target"], "decided": None, "typed": None,
                                 "confidence": None, "uncertain": True, "correct": None,
                                 "flashes": self.current["flashes"], "updates": 0, "stopped_early": False,
                                 "decided_on": reason,
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
        unknown = [s for s in symbols if s not in self.rep.symbols]
        epoch = msg.get("epoch") if isinstance(msg.get("epoch"), int) else None
        with self.lock:
            if epoch is not None:
                self.flash_counts[epoch] = self.flash_counts.get(epoch, 0) + 1
            cur = self.current
            if cur is not None and epoch == cur["epoch_id"]:
                cur["flashes"] = self.flash_counts[epoch]
                tl = cur["timeline"]
                tl["first_flash_lsl"] = tl["first_flash_lsl"] or item.lsl_time
                tl["last_flash_lsl"] = item.lsl_time
            self.sync["flashes"] += 1
            self.sync["last_flash_lsl"] = item.lsl_time
            # The onset is the sender's LSL timestamp (taken at the flip) mapped into
            # this computer's LSL clock; recv_lsl_s is when the marker reached the speller.
            self.rec.emit("Presentation Method", "stimulus_event",
                          {"symbols": symbols, "indices": indices, "module": item.module,
                           "onset_lsl_s": item.lsl_time, "recv_lsl_s": _round(item.recv_time),
                           "timing_method": "module_flip_lsl_timestamp",
                           **({"unknown_symbols": unknown} if unknown else {})},
                          epoch)

    # ---------------------------------------------------------- correction
    def _letters(self, start_word: int, n_words: int) -> list[list[list]]:
        """Top-K posterior symbols for every character of the words in the window."""
        k = int(self.cfg.slow_llm.get("alternatives_k", 4))
        spans = list(WORD.finditer(self.primary.value))[start_word:start_word + n_words]
        out = []
        for span in spans:
            chars = []
            for j in range(span.start(), span.end()):
                lp = self.char_post[j] if j < len(self.char_post) else None
                chars.append([] if lp is None else self._top(lp, k, typed=True))
            out.append(chars)
        return out

    def _request_correction(self, epoch_id: int) -> None:
        req = self.scheduler.on_primary_update(self.primary.value) if self.scheduler else None
        if req is None or not req.words:
            return
        letters = (self._letters(req.start_word, len(req.words))
                   if self.cfg.slow_llm.get("include_alternatives") else None)
        prompt_file = PROJECT / self.cfg.slow_llm["prompt_file"]
        self.rec.emit("Post-factum Text Correction", "correction_requested",
                      {"correction_id": req.correction_id, "trigger": req.trigger,
                       "window_words": req.words, "window_start_word": req.start_word,
                       "left_context_len": len(req.left_context), "model_id": self.slow.model_id,
                       "alternatives": letters,
                       "prompt_hash": getattr(self.slow, "prompt_hash", None) or
                       (prompt_hash(prompt_file) if prompt_file.is_file() else None)}, epoch_id)
        self.correction_queue.put((req, letters, self.primary.value, epoch_id, time.monotonic()))

    def _correction_worker(self):
        while True:
            job = self.correction_queue.get()
            if job is None:
                self.correction_queue.task_done()
                return
            req, letters, snapshot, epoch_id, queued = job
            start = time.perf_counter_ns()
            try:
                if letters is not None and getattr(self.slow, "version", "").startswith("chat-api"):
                    result = self.slow.correct(req.left_context, req.words, letters)
                else:
                    result = self.slow.correct(req.left_context, req.words)
                accepted, reason = validate_correction(req.words, result, self.cfg.slow_llm["strict_word_count"])
            except Exception as exc:
                result, accepted, reason = list(req.words), False, f"{type(exc).__name__}: {str(exc)[:200]}"
            latency = (time.perf_counter_ns() - start) / 1e6
            with self.lock:
                if accepted and not self.primary.value.startswith(snapshot):
                    accepted, reason = False, "text_changed"     # a ⌫ reached into the window
                self.rec.emit("Post-factum Text Correction", "correction_completed",
                              {"correction_id": req.correction_id, "words_in": req.words, "words_out": result,
                               "accepted": accepted, "reject_reason": reason, "wall_latency_ms": latency,
                               "queued_ms": round((time.monotonic() - queued) * 1000 - latency, 1),
                               "messages": getattr(self.slow, "last_messages", None),
                               "reply": getattr(self.slow, "last_reply", None)}, epoch_id)
                changed = accepted and list(result) != list(req.words)
                self.corrections.append({"correction_id": req.correction_id, "epoch_id": epoch_id,
                                         "words_in": list(req.words),
                                         "words_out": list(result), "accepted": accepted, "changed": changed,
                                         "reason": reason, "latency_ms": round(latency),
                                         "t_s": round(time.monotonic() - (self.started_wall or time.monotonic()), 1)})
                del self.corrections[:-LOG_LIMIT]
                if accepted:
                    try:
                        self.secondary.replace_window(result, req.start_word)
                    except ValueError as exc:
                        self.rec.emit("Secondary (Corrected) Text", "warning",
                                      {"module": "secondary", "message": str(exc)}, epoch_id)
                    else:
                        self.rec.emit("Secondary (Corrected) Text", "secondary_text_updated",
                                      {"correction_id": req.correction_id, "text": self.secondary.value,
                                       "frozen_upto_word": self.secondary.frozen_upto_word}, epoch_id)
            self._bump()
            self.correction_queue.task_done()

    def _flush_corrections(self, timeout: float) -> None:
        if self.slow is None or not self.correction_thread.is_alive():
            return
        deadline = time.monotonic() + timeout
        while self.correction_queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.05)
        # Requests still waiting are closed as rejected so the record stays complete.
        dropped = []
        while True:
            try:
                job = self.correction_queue.get_nowait()
            except queue.Empty:
                break
            if job is not None:
                dropped.append(job)
            self.correction_queue.task_done()
        for req, _, _, epoch_id, _ in dropped:
            self.rec.emit("Post-factum Text Correction", "correction_completed",
                          {"correction_id": req.correction_id, "words_in": req.words, "words_out": req.words,
                           "accepted": False, "reject_reason": "session_ended", "wall_latency_ms": 0.0}, epoch_id)
        self.correction_queue.put(None)

    # --------------------------------------------------------------- finish
    def _finish(self):
        timeout = float(self.cfg.slow_llm.get("timeout_s", 30)) + 2 if self.slow else 0
        self._flush_corrections(min(timeout, 40))
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
    def _top(self, log_p, k, typed=False):
        p = np.exp(np.asarray(log_p))
        sym = self.rep.symbols
        if not typed:
            return [[sym[int(i)], round(float(p[i]), 4)] for i in np.argsort(-p)[:k]]
        # As typed text: keys that type the same character (Е and Ё) are merged.
        mass: dict[str, float] = {}
        for s, v in zip(sym, p):
            if s != BACKSPACE:
                mass[self.typed[s]] = mass.get(self.typed[s], 0.0) + float(v)
        return [[c, round(v, 4)] for c, v in sorted(mass.items(), key=lambda kv: -kv[1])[:k]]

    def _bump(self):
        with self.changed:
            self.version += 1
            self.changed.notify_all()

    def wait_change(self, since: int, timeout: float) -> None:
        with self.changed:
            self.changed.wait_for(lambda: self.version != since, timeout=timeout)

    def _top_dicts(self, log_p, k):
        return [{"symbol": s, "p": p} for s, p in self._top(log_p, k)]

    def monitor(self) -> dict:
        """Compact state for the operator window."""
        k = int(self.opts["top_k_display"])
        with self.lock:
            cur = self.current if self.phase in ("presenting", "result") else None
            epoch = None
            if cur is not None:
                post = cur.get("post")
                obs = cur.get("obs")
                epoch = {"epoch_id": cur["epoch_id"], "target": cur["target"], "flashes": cur["flashes"],
                         "updates": cur["updates"], "stop_sent": cur["stop_sent"],
                         "elapsed_s": round(time.monotonic() - cur["began"], 1),
                         "prior_top": self._top_dicts(cur["prior"].log_p, k),
                         "likelihood_top": self._top_dicts(obs.log_p, k) if obs is not None else [],
                         "posterior_top": self._top_dicts(post.log_p, k) if post is not None else [],
                         "leader": self._top_dicts(post.log_p, 1)[0] if post is not None else None,
                         "words": [w[0] if isinstance(w, (list, tuple)) else w for w in (cur["words"] or [])][:5],
                         "prior_ms": round(cur["prior_ms"], 1)}
            decided = [r for r in self.log if r["decided"] is not None]
            judged = [r for r in decided if r["correct"] is not None]
            end = self.ended_wall or time.monotonic()
            minutes = ((end - self.started_wall) / 60) if self.started_wall else 0
            matching = sum(a == b for a, b in zip(self.primary.value, self.target)) if self.target else None
            return {
                "session_id": self.session_id, "profile": self.profile, "passport": self.passport,
                "participant_id": self.cfg.participant_id, "task_id": self.cfg.task_id,
                "llm": self.cfg.mode["llm"], "phase": self.phase, "paused": self.paused,
                "finished": self.finished, "end_reason": self.end_reason, "error": self.error,
                "target": self.target, "target_index": self.target_index,
                "text": self.primary.value, "corrected": self.secondary.value,
                "epoch_id": self.epoch_id, "epoch": epoch,
                "keyboard": {"symbols": list(self.rep.symbols), "layout": list(self.rep.layout)},
                "stop_confidence": self.opts["stop_confidence"], "tau": float(self.cfg.detector["tau"]),
                "modules": self.transport.status(),
                "elapsed_s": round((end - self.started_wall), 1) if self.started_wall else 0,
                "stats": {"selected": len(decided), "epochs": len(self.log),
                          "accuracy": round(sum(bool(r["correct"]) for r in judged) / len(judged), 3) if judged else None,
                          "early_stops": sum(bool(r["stopped_early"]) for r in self.log),
                          "mean_epoch_s": round(float(np.mean([r["duration_s"] for r in self.log])), 2) if self.log else None,
                          "speed_cpm": round(matching / minutes, 2) if matching is not None and minutes > 0.05 else None,
                          "pending_corrections": self.correction_queue.unfinished_tasks if self.slow else 0},
                "log": self.log[-30:], "corrections": self.corrections[-50:],
                "directory": str(self.storage.directory),
            }


def default_model_factory(kind: str, cfg: dict):
    if kind == "fast":
        return create_fast(cfg, CORPUS)
    cfg = dict(cfg)
    allow_remote = bool(cfg.pop("_allow_remote", False))
    return create_slow(cfg, allow_remote, CORPUS)
