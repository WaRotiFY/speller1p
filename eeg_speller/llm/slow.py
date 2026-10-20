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
        # IMPROV[Q-40]
        every = max(1, int(self.cfg.get("every_n_words", 1)))
        if trigger == "word_end" and count % every and primary[-1] != ".":
            return None
        start = max(0, count - max(int(self.cfg["window_words"]), every))
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


def align_correction(words_in, words_out):
    """Fit a model reply onto the window under the speller's error model (MA-37).

    A P300 speller types one symbol per epoch, so a misrecognized word keeps its length.
    A replacement is therefore taken only for a word of the same length; a reply word of
    another length is ignored and the typed word is kept. When the reply has fewer or more
    words (small models drop short words like «у», «с»), its words are matched to the
    window in order by a small dynamic programme. Returns (words, accepted, reason);
    reason is None for an exact fit, "aligned" when some reply words were ignored or
    words were kept as typed, and "word_count" when nothing in the reply fits.
    """
    # ASSUMPTION[MA-38]
    words_in, words_out = list(words_in), list(words_out)
    if any(not re.fullmatch(r"[а-я]+", w) for w in words_out):
        return words_in, False, "invalid_character"
    n, m = len(words_in), len(words_out)
    if n == m and all(len(a) == len(b) for a, b in zip(words_in, words_out)):
        return words_out, True, None
    # best[i][j]: the most reply words fitted using window[:i] and reply[:j]
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            options = []
            if i > 0:
                options.append((best[i - 1][j], "keep"))                  # window word stays as typed
            if j > 0:
                options.append((best[i][j - 1] - 0.01, "drop"))         # reply word ignored
            if i > 0 and j > 0 and len(words_in[i - 1]) == len(words_out[j - 1]):
                gain = 1.0 + (0.5 if words_in[i - 1] == words_out[j - 1] else 0.0)
                options.append((best[i - 1][j - 1] + gain, "fit"))
            best[i][j], move[i][j] = max(options, key=lambda o: o[0])
    out, i, j, fitted = list(words_in), n, m, 0
    while i > 0 or j > 0:
        step = move[i][j]
        if step == "fit":
            out[i - 1] = words_out[j - 1]
            fitted += 1
            i, j = i - 1, j - 1
        elif step == "keep":
            i -= 1
        else:
            j -= 1
    if fitted == 0:
        return words_in, False, "word_count"
    return out, True, "aligned"


def create_slow(cfg, allow_remote: bool, corpus):
    backend = cfg["backend"]
    if backend == "mock_dictionary":
        return MockCorrection(cfg, corpus)
    if backend == "dictionary":
        from eeg_speller.llm.word_candidates import DictionaryCorrection
        return DictionaryCorrection(cfg)
    if backend == "chat_api":
        from eeg_speller.llm.backends.chat_api import ChatCorrection
        return ChatCorrection(cfg, allow_remote)
    if backend == "hf_causal":
        raise NotImplementedError("local slow HF correction generation is not implemented")
    raise ValueError(f"unknown slow backend: {backend}")


def prompt_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
