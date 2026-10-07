"""Correction scheduling and output validation."""
from dataclasses import dataclass
import re
from pathlib import Path
import hashlib
from eeg_speller.llm.backends.mock_correction import MockCorrection


@dataclass(frozen=True)
class CorrectionRequest:
    correction_id: int
    trigger: str
    words: list[str]
    start_word: int
    left_context: str


class CorrectionScheduler:
    """Post-factum Text Correction scheduler; NEED-F-07; MA-19; Q-15."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.next_id = 0
        self.last_count = 0

    def on_primary_update(self, primary: str):
        # ASSUMPTION[MA-19]
        trigger = self.cfg["trigger"]
        if trigger == "manual" or not primary:
            return None
        if trigger == "word_end" and primary[-1] not in " .,":
            return None
        if trigger == "sentence_end" and primary[-1] != ".":
            return None
        if trigger == "every_k_symbols" and len(primary) % int(self.cfg.get("every_k_symbols", 10)):
            return None
        matches = list(re.finditer(r"[а-я]+", primary))
        count = len(matches)
        if count == self.last_count:
            return None
        self.last_count = count
        if count <= self.cfg["min_context_words"] and primary[-1] != ".":
            return None
        start = max(0, count - self.cfg["window_words"])
        window = [m.group() for m in matches[start:]]
        left = primary[:matches[start].start()] if window else primary
        request = CorrectionRequest(self.next_id, trigger, window, start, left)
        self.next_id += 1
        return request


def validate_correction(words_in, words_out, strict=True):
    # ASSUMPTION[MA-20]
    if strict and len(words_in) != len(words_out):
        return False, "word_count"
    if any(not re.fullmatch(r"[а-я]+", word) for word in words_out):
        return False, "invalid_character"
    return True, None


def create_slow(cfg, allow_remote: bool, corpus):
    backend = cfg["backend"]
    if backend == "mock_dictionary":
        return MockCorrection(cfg, corpus)
    if backend == "openai_compat":
        from eeg_speller.llm.backends.openai_compat import OpenAICompatCorrection
        return OpenAICompatCorrection(cfg, allow_remote)
    if backend == "local_openai":
        from eeg_speller.llm.backends.local_openai import LocalOpenAICorrection
        return LocalOpenAICorrection(cfg)
    if backend == "hf_causal":
        raise NotImplementedError("local slow HF correction generation is not implemented")
    raise ValueError(f"unknown slow backend: {backend}")


def prompt_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
