"""Configurable keyboard (repertoire) and its mapping to the language-model alphabet.

The speller owns the keyboard: symbols and rows x columns come from the config and are
sent to the stimulus module in every ``epoch_start``. Language models work in a fixed
lowercase alphabet; keyboard symbols are mapped into it:

- uppercase -> lowercase, ``ё`` -> ``е``, ``_`` / ``␣`` -> space;
- ``⌫`` (delete) gets a fixed probability ``backspace_prob``;
- symbols the language model does not know (digits, ``-`` ...) share ``other_prob`` each.
"""
from __future__ import annotations

import numpy as np

LM_ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюя .,"
BACKSPACE = "⌫"
SPACE_ALIASES = {"_": " ", "␣": " "}

PRESETS = {
    "abc_6x6": {
        "label": "Русский алфавит 6×6 (а–я, пробел, точка, запятая, ⌫)",
        "symbols": list("абвгдежзийклмнопрстуфхцчшщъыьэюя .,") + [BACKSPACE],
        "layout": [6, 6],
    },
    "jcuken_4x11_digits": {
        "label": "ЙЦУКЕН 4×11 с цифрами («_» = пробел, без ⌫)",
        "symbols": list("1234567890_ЙЦУКЕНГШЩЗХФЫВАПРОЛДЖЭЁЯЧСМИТЬБЮЪ"),
        "layout": [4, 11],
    },
    "jcuken_4x10": {
        "label": "ЙЦУКЕН 4×10 для набора (буквы, пробел, точка, запятая, ⌫)",
        "symbols": list("ЙЦУКЕНГШЩЗХФЫВАПРОЛДЖЭЯЧСМИТЬБЮЪЁ_.,") + [BACKSPACE],
        "layout": [4, 10],
    },
}


def lm_char(symbol: str) -> str | None:
    """Keyboard symbol -> character of the LM alphabet (None if the LM does not know it)."""
    if symbol == BACKSPACE:
        return None
    s = SPACE_ALIASES.get(symbol, symbol).lower().replace("ё", "е")
    return s if len(s) == 1 and s in LM_ALPHABET else None


def typed_char(symbol: str) -> str:
    """Character written into the typed text when ``symbol`` is chosen."""
    return lm_char(symbol) or symbol


def validate_keyboard(symbols, layout) -> None:
    if not isinstance(symbols, (list, tuple)) or len(symbols) < 2:
        raise ValueError("keyboard needs at least two symbols")
    if any(not isinstance(s, str) or len(s) != 1 for s in symbols):
        raise ValueError("every keyboard symbol must be one character")
    if len(set(symbols)) != len(symbols):
        raise ValueError("keyboard symbols must be distinct")
    rows, cols = (int(x) for x in layout)
    if rows < 1 or cols < 1 or rows * cols < len(symbols):
        raise ValueError(f"layout {rows}x{cols} has fewer cells than {len(symbols)} symbols")


def symbol_for_char(ch: str, symbols) -> str | None:
    """Keyboard symbol that types ``ch`` (used for the cue in copy-spelling)."""
    for s in symbols:
        if typed_char(s) == ch:
            return s
    return None


def project(lm_p, symbols, backspace_prob: float = 0.02, other_prob: float = 0.005) -> np.ndarray:
    """Probabilities over LM_ALPHABET -> probabilities over keyboard ``symbols``.

    Symbols mapping to the same LM character (``Е`` and ``Ё``) share its mass. Characters
    of the LM that are not on the keyboard are dropped and the rest renormalized.
    """
    # ASSUMPTION[MA-35]
    lm_p = np.asarray(lm_p, dtype=float)
    out = np.zeros(len(symbols))
    groups: dict[int, list[int]] = {}
    fixed = 0.0
    for i, s in enumerate(symbols):
        c = lm_char(s)
        if s == BACKSPACE:
            out[i] = backspace_prob
            fixed += backspace_prob
        elif c is None:
            out[i] = other_prob
            fixed += other_prob
        else:
            groups.setdefault(LM_ALPHABET.index(c), []).append(i)
    free = max(0.0, 1.0 - fixed)
    mass = sum(lm_p[j] for j in groups)
    for j, idx in groups.items():
        share = (lm_p[j] / mass if mass > 0 else 1 / max(len(groups), 1)) * free / len(idx)
        for i in idx:
            out[i] = share
    total = out.sum()
    return out / total if total > 0 else np.full(len(symbols), 1 / len(symbols))


def keyboard_text(text: str) -> str:
    """Typed text -> LM text (unknown characters dropped)."""
    out = []
    for ch in text:
        c = lm_char(ch) if ch not in LM_ALPHABET else ch
        if c:
            out.append(c)
    return "".join(out)
