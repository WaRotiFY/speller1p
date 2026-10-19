"""The v3 epoch loop and session orchestration."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from eeg_speller.core.clock import Clock
from eeg_speller.core.distributions import Repertoire, SymbolDistribution, log_probs
from eeg_speller.decoding.detector import SymbolDetector
from eeg_speller.epoch.fusion import Fusion
from eeg_speller.epoch.presentation_feedback import make_feedback
from eeg_speller.ext.ui_console import ConsoleUI
from eeg_speller.ext.word_prediction import WordPrediction
from eeg_speller.learning.rl_block import RLBlock
from eeg_speller.llm.fast import create_fast
from eeg_speller.llm.slow import CorrectionScheduler, create_slow, prompt_hash, validate_correction
from eeg_speller.physiology.base import EpochStart, StimulusTiming
from eeg_speller.physiology.registry import create as create_physiology
from eeg_speller.physiology.simulated import SimulatedUser
from eeg_speller.recording.recorder import SessionRecorder
from eeg_speller.recording.storage import SessionStorage
from eeg_speller.text.primary import PrimaryText
from eeg_speller.text.secondary import SecondaryText


PROJECT = Path(__file__).resolve().parents[2]


def read_tasks(path=PROJECT / "data" / "tasks_ru.txt"):
    tasks = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            task_id, text = line.split("\t", 1)
            tasks[task_id] = text
    return tasks


class FixedRepetitions:
    """EpochPolicy; NEED-F-05; MA-07; Q-05."""
    def __init__(self, n: int):
        # STUB[Q-05]
        if n < 1:
            raise ValueError("n_repetitions must be positive")
        self.n = n


def _model_info(model, backend):
    return {"backend": backend, "model_id": model.model_id,
            "version": model.version, "hash": model.weights_hash}


class EpochRunner:
    """One Epoch orchestration; NEED-F-01/05/06/12; MA-07/12/14; Q-04/05."""
    def __init__(self, cfg, root, source=None, verbose=False):
        # IMPROV[Q-09]
        if cfg.run_mode == "live" and cfg.physiology["plugin"] != "stub_live":
            raise ValueError("live mode requires the external stub_live physiology plugin")
        if cfg.run_mode == "semi_synthetic":
            raise NotImplementedError("Q-02: semi-synthetic source is reserved")
        if cfg.run_mode == "replay" and source is None:
            raise ValueError("replay mode requires a recorded physiology source")
        self.cfg = cfg
        self.rep = Repertoire(tuple(cfg.repertoire["symbols"]), tuple(cfg.repertoire["layout"]))
        self.clock = Clock(cfg.run_mode == "live")
        self.policy = FixedRepetitions(cfg.epoch["n_repetitions"])
        # IMPROV[Q-04]
        self.fast = create_fast(cfg.fast_llm, PROJECT / "data" / "corpus_ru_fallback.txt") if cfg.mode["llm"] == "on" else None
        self.slow = create_slow(cfg.slow_llm, cfg.llm["allow_remote"], PROJECT / "data" / "corpus_ru_fallback.txt") if cfg.mode["llm"] == "on" else None
        models = {"fast_llm": _model_info(self.fast, cfg.fast_llm["backend"]) if self.fast else {"backend": "uniform", "model_id": "uniform", "version": "1", "hash": "none"},
                  "slow_llm": _model_info(self.slow, cfg.slow_llm["backend"]) if self.slow else {"backend": "disabled", "model_id": "disabled", "version": "1", "hash": "none"},
                  "physiology": {"backend": cfg.physiology["plugin"] if source is None else "replay", "model_id": cfg.physiology["plugin"], "version": "1", "hash": hashlib.sha256(json.dumps(cfg.physiology, sort_keys=True).encode()).hexdigest()}}
        self.storage = SessionStorage(Path(root), cfg, self.rep, models,
                                      source.manifest["session_id"] if source is not None else None)
        self.rec = SessionRecorder(self.storage.directory, self.storage.session_id, cfg.participant_id, self.clock)
        self.primary = PrimaryText()
        self.secondary = SecondaryText()
        self.detector = SymbolDetector(cfg.detector, self.rep)
        self.fusion = Fusion(cfg.fusion, cfg.mode["llm"])
        self.word_predictor = WordPrediction(self.fast, cfg.word_prediction) if self.fast else None
        self.scheduler = CorrectionScheduler(cfg.slow_llm) if self.slow else None
        self.rl = RLBlock(cfg.rl)
        self.ui = ConsoleUI(self.rec, verbose)
        self.history = []
        self.source = source
        if source is None:
            user = SimulatedUser(np.random.default_rng(cfg.child_seed("user")), cfg.user["lapse_rate"])
            self.physiology = create_physiology(cfg.physiology["plugin"], cfg.physiology,
                                               user, np.random.default_rng(cfg.child_seed("physiology")))
        else:
            self.physiology = source

    def _emit_prior(self, epoch_id):
        cfg = self.cfg
        start = time.perf_counter_ns()
        if cfg.mode["llm"] == "off":
            # ASSUMPTION[MA-13]
            prior = SymbolDistribution(self.rep, log_probs(np.ones(len(self.rep.symbols))), "uniform", len(self.primary.value))
            model_id = "uniform"
        else:
            context = self.primary.value if cfg.fast_llm["context_source"] == "primary" else self.secondary.value
            # IMPROV[Q-16]
            prior = self.fast.predict(context, self.rep)
            model_id = self.fast.model_id
        latency = (time.perf_counter_ns() - start) / 1e6
        self.rec.emit("Source Model (Fast LLM)", "prior_computed",
                      {"model_id": model_id, "model_version": self.fast.version if self.fast else "1",
                       "model_hash": self.fast.weights_hash if self.fast else "none",
                       "context_source": cfg.fast_llm["context_source"],
                       "context_tail": self.primary.value[-64:], "context_len": len(self.primary.value),
                       "log_p": prior.log_p, "wall_latency_ms": latency}, epoch_id)
        return prior

    def _emit_words(self, epoch_id):
        if not self.word_predictor:
            return []
        start = time.perf_counter_ns()
        candidates = self.word_predictor.predict(self.primary.value)
        self.rec.emit("ext:word_prediction", "word_candidates",
                      {"prefix": self.primary.value.split(" ")[-1], "candidates": candidates,
                       "model_id": self.fast.model_id, "model_version": self.fast.version,
                       "model_hash": self.fast.weights_hash,
                       "wall_latency_ms": (time.perf_counter_ns() - start) / 1e6}, epoch_id)
        return candidates

    def _correction(self, epoch_id):
        # IMPROV[Q-15]
        # IMPROV[Q-17]
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
        self.rec.emit("Post-factum Text Correction", "correction_completed",
                      {"correction_id": req.correction_id, "words_in": req.words,
                       "words_out": result, "accepted": accepted, "reject_reason": reason,
                       "model_id": self.slow.model_id, "model_version": self.slow.version,
                       "model_hash": self.slow.weights_hash,
                       "wall_latency_ms": (time.perf_counter_ns() - start) / 1e6}, epoch_id)
        if not accepted:
            return
        self.secondary.replace_window(result, req.start_word)
        frozen = self.secondary.frozen_upto_word
        self.rec.emit("Secondary (Corrected) Text", "secondary_text_updated",
                      {"correction_id": req.correction_id, "text": self.secondary.value,
                       "frozen_upto_word": frozen}, epoch_id)
        if self.cfg.rl["enabled"]:
            # STUB[Q-18]
            import re
            matches = list(re.finditer(r"[а-я]+", self.primary.value))
            frozen_char = matches[frozen].start() if frozen < len(matches) else len(self.primary.value)
            batch = self.rl.on_secondary_update(self.primary.value, self.secondary.value,
                                                frozen_char, self.history, req.correction_id)
            if batch:
                self.rec.emit("Direct/Indirect RL", "rl_broadcast",
                              {"batch_id": req.correction_id, "correction_id": req.correction_id,
                               "recipients": ["EEG/EOG source", "Primary Symbol Detector"],
                               "labels": [[x.epoch_id, x.detected, x.corrected] for x in batch.labels]}, epoch_id)
                self.physiology.receive_update(batch)
                self.detector.receive_update(batch)
                for block in ("EEG / EOG source", "Primary Symbol Detector"):
                    self.rec.emit(block, "rl_update_received",
                                  {"batch_id": req.correction_id, "action": "logged_ignored"}, epoch_id)

    def run(self, target: str):
        cfg = self.cfg
        max_symbols = cfg.runtime.get("max_symbols")
        if max_symbols is not None:
            target = target[:max_symbols]
        self.rec.emit("Session Recorder", "session_started",
                      {"run_mode": cfg.run_mode, "llm_mode": cfg.mode["llm"],
                       "paradigm": cfg.stimulus["paradigm"], "seed": cfg.seed,
                       "config_hash": cfg.digest()})
        self.rec.emit("Session Recorder", "task_assigned", {"task_id": cfg.task_id, "target_text": target})
        epoch_id = 0
        for target_position, target_symbol in enumerate(target):
            retry = 0
            while True:
                if epoch_id:
                    self.clock.advance_ms(cfg.stimulus["pause_ms"])
                position = len(self.primary.value)
                self.rec.emit("One Epoch", "epoch_started",
                              {"position": position, "repertoire_hash": hashlib.sha256("".join(self.rep.symbols).encode()).hexdigest(),
                               "paradigm": cfg.stimulus["paradigm"], "flash_ms": cfg.stimulus["flash_ms"],
                               "isi_ms": cfg.stimulus["isi_ms"], "pause_ms": cfg.stimulus["pause_ms"],
                               "n_repetitions": self.policy.n, "retry_idx": retry,
                               "sim_target_symbol": target_symbol}, epoch_id)
                timing = StimulusTiming(cfg.stimulus["flash_ms"], cfg.stimulus["isi_ms"],
                                        cfg.stimulus["pause_ms"], self.policy.n)
                msg = EpochStart(epoch_id, position, self.rep, cfg.stimulus["paradigm"], timing)
                if self.source is None:
                    self.physiology.set_target(target_symbol)
                self.physiology.start_epoch(msg)
                prior = self._emit_prior(epoch_id)
                candidates = self._emit_words(epoch_id)
                observation = self.physiology.get_observation()
                if observation.stimulus_events:
                    for stimulus in observation.stimulus_events:
                        t_start = self.clock.t_ns()
                        self.rec.emit("Presentation Method", "stimulus_event",
                                      {**stimulus, "onset_t_ns": t_start}, epoch_id)
                        self.clock.advance_ms(cfg.stimulus["flash_ms"] + cfg.stimulus["isi_ms"])
                else:
                    flashes = (12 if cfg.stimulus["paradigm"] == "row_column" else 36) * self.policy.n
                    self.clock.advance_ms(flashes * (cfg.stimulus["flash_ms"] + cfg.stimulus["isi_ms"]))
                self.rec.emit("EEG / EOG source", "physiology_observation",
                              {"log_p": observation.log_p, "semantics": observation.semantics,
                               "source": cfg.physiology["plugin"] if self.source is None else "replay",
                               "per_repetition": observation.per_repetition,
                               "module_metrics": observation.module_metrics,
                               "t_module_ns": observation.t_module_ns,
                               "sim_flash_scores": observation.sim_flash_scores}, epoch_id)
                started = time.perf_counter_ns()
                post = self.fusion.combine(prior, observation)
                self.rec.emit("Statistical Inference", "posterior_computed",
                              {"fusion": post.fusion, "params": post.params, "log_p": post.log_p,
                               "wall_latency_ms": (time.perf_counter_ns() - started) / 1e6}, epoch_id)
                feedback = make_feedback(post, self.rep, cfg.presentation_feedback)
                self.rec.emit("Influence on presentation method", "presentation_feedback",
                              {"decided": feedback.decided, "top_k": feedback.top_k}, epoch_id)
                self.physiology.receive_feedback(feedback)
                self.rec.emit("Presentation Method", "presentation_feedback_received",
                              {"action": "logged_ignored"}, epoch_id)
                decision = self.detector.decide(post, position, retry)
                self.rec.emit("Primary Symbol Detector", "symbol_decided",
                              {"symbol": decision.symbol, "confidence": decision.confidence,
                               "margin": decision.margin, "uncertain": decision.uncertain,
                               "cold_start": decision.cold_start,
                               "abstained": decision.symbol is None}, epoch_id)
                if decision.symbol is not None:
                    self.primary.apply(decision.symbol)
                    self.secondary.synchronize(self.primary.value)
                    self.history.append((epoch_id, post.log_p))
                    self.rec.emit("Primary Text", "primary_text_updated",
                                  {"op": "delete" if decision.symbol == "⌫" else "append",
                                   "symbol": decision.symbol, "position": position,
                                   "text": self.primary.value}, epoch_id)
                    self._correction(epoch_id)
                    self.ui.render(self.primary.value, self.secondary.value, candidates, "selected", epoch_id)
                else:
                    self.ui.render(self.primary.value, self.secondary.value, candidates, "abstained", epoch_id)
                epoch_id += 1
                if decision.symbol is not None:
                    break
                retry += 1
        self.rec.emit("Session Recorder", "session_ended",
                      {"status": "completed", "reason": None, "n_epochs": epoch_id,
                       "n_events": self.rec.seq + 1})
        event_count = self.rec.seq
        self.rec.close()
        self.storage.finish(target, self.primary.value, self.secondary.value, epoch_id, event_count)
        return self.storage.directory
