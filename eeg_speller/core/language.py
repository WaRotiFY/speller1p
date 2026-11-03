"""Language of a session: LM alphabet and the language-specific resources.

The speller types one language at a time. ``set_language`` is called when a session
starts (config key ``language``: ``ru`` or ``en``); everything that depends on the
language reads it from here:

- the LM alphabet (letters + space, period, comma) that keyboards are mapped into;
- default fast models: character n-gram file, KenLM repository, Hugging Face model;
- copy-spelling tasks, the small fallback corpus, the slow-LM prompt.

Word patterns (``WORD``) cover both alphabets, so text code needs no language switch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    letters: str
    ngram_file: str
    kenlm_repo: str
    hf_model: str
    tasks_file: str
    corpus_file: str
    prompt_file: str

    @property
    def lm_alphabet(self) -> str:
        return self.letters + " .,"


LANGUAGES = {
    "ru": Language("ru", "русский", "абвгдежзийклмнопрстуфхцчшщъыьэюя", "ru_char4.npz",
                   "jonatasgrosman/wav2vec2-large-xlsr-53-russian", "Qwen/Qwen2.5-0.5B",
                   "data/tasks_ru.txt", "data/corpus_ru_fallback.txt", "prompts/correction_v3.txt"),
    "en": Language("en", "English", "abcdefghijklmnopqrstuvwxyz", "en_char4.npz",
                   "jonatasgrosman/wav2vec2-large-xlsr-53-english", "HuggingFaceTB/SmolLM2-135M",
                   "data/tasks_en.txt", "data/corpus_en_fallback.txt", "prompts/correction_en.txt"),
}
DEFAULT = "ru"
LETTERS = "а-яa-z"                       # regex class: letters of every supported language
WORD = re.compile(f"[{LETTERS}]+")
WORD_TAIL = re.compile(f"[{LETTERS}]*$")

_current = LANGUAGES[DEFAULT]


def set_language(code: str | None) -> Language:
    global _current
    code = (code or DEFAULT).lower()
    if code not in LANGUAGES:
        raise ValueError(f"unknown language {code!r}; known: {', '.join(LANGUAGES)}")
    _current = LANGUAGES[code]
    return _current


def current() -> Language:
    return _current


def lm_alphabet() -> str:
    return _current.lm_alphabet


def path(attr: str, lang: Language | None = None) -> Path:
    """Project path of a language resource, e.g. ``path("tasks_file")``."""
    return PROJECT / getattr(lang or _current, attr)


def detect(symbols) -> str:
    """Language whose letters cover most of a keyboard (for modules that bring their own)."""
    letters = [str(s).lower().replace("ё", "е") for s in symbols if str(s).isalpha()]
    best, score = DEFAULT, -1
    for code, lang in LANGUAGES.items():
        n = sum(ch in lang.letters for ch in letters)
        if n > score:
            best, score = code, n
    return best
