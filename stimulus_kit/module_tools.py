"""Small helpers for stimulus modules (optional; only speller_link.py is required).

- ``module_args``: command line and ``settings.json`` of the module in one object;
- ``keyboard_of``: keyboard from the settings (``alphabet``, ``num_rows``, ``num_cols``);
- ``cue_for``: target letter to show before an epoch (``sentence`` in the settings);
- ``load_settings``: read a ``settings.json``;
- ``color``: "#rrggbb", [r, g, b] or a grey level 0..255 -> (r, g, b);
- ``AsyncKeyboardDecoder``: "space bar instead of EEG" for paradigms where stimuli
  overlap in time (moving letters, circles). A press counts for every stimulus whose
  onset was 150..1000 ms before it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys


def module_args(description: str, default_settings: str):
    """Command line and settings of a module.

    Everything is set in the module's ``settings.json``; the flags only override it when
    the module is started by hand. Returns args with ``config`` (the settings), ``keyboard``
    (space bar instead of EEG) and ``windowed``.
    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--settings", default=default_settings, help="settings.json of this module")
    ap.add_argument("--keyboard", action="store_true", help="SPACE instead of EEG (for trying the paradigm)")
    ap.add_argument("--windowed", action="store_true", help="window instead of full screen")
    args = ap.parse_args()
    args.config = load_settings(os.path.abspath(args.settings))
    args.keyboard = args.keyboard or bool(args.config.get("space_instead_of_eeg", False))
    args.windowed = args.windowed or not bool(args.config.get("fullscreen", True))
    return args


def keyboard_of(cfg: dict):
    """(symbols, (rows, columns)) from ``alphabet``, ``num_rows``, ``num_cols``; ``_`` is a space."""
    symbols = list(str(cfg["alphabet"]).replace("\n", ""))
    return symbols, (int(cfg["num_rows"]), int(cfg["num_cols"]))


def cue_for(cfg: dict, typed_text: str, symbols) -> str | None:
    """Target symbol to show before the epoch, from ``sentence`` (copy-spelling, calibration).

    The n-th epoch of the session shows the n-th character of the sentence (by the length
    of the typed text). Empty ``sentence``: no cue. Case and ``_``/space are matched to
    the keyboard.
    """
    sentence = str(cfg.get("sentence") or "")
    i = len(typed_text)
    if not sentence or i >= len(sentence):
        return None
    ch = sentence[i]
    for s in symbols:
        if s == ch or s.lower() == ch.lower() or (s == "_" and ch == " ") or (s == " " and ch == "_"):
            return s
    return None


def load_settings(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    # Keys starting with "_" are comments for people editing the file.
    return {k: v for k, v in data.items() if not k.startswith("_")}


def color(value, default=(0, 0, 0)):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        v = int(value)
        return (v, v, v)
    if isinstance(value, str) and value.startswith("#") and len(value) == 7:
        return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(x) for x in value)
    return default


def place_window(x, y):
    """Open the pygame window at a given screen position (e.g. on the second monitor)."""
    if x is not None and y is not None:
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{int(x)},{int(y)}"
    os.environ.setdefault("SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS", "0")


def log_path(name: str) -> str:
    """Log files go to the current folder: the operator window runs modules in the session folder."""
    os.makedirs("logs", exist_ok=True)
    return os.path.join("logs", name)


class AsyncKeyboardDecoder:
    """P(press | attended stimulus started 150..1000 ms ago) = hit, otherwise false_alarm."""

    def __init__(self, symbols, hit=0.8, false_alarm=0.1, window=(0.15, 1.0)):
        # STUB[Q-46]
        self.symbols = list(symbols)
        self.hit, self.fa = hit, false_alarm
        self.window = window
        self.logp = {s: 0.0 for s in self.symbols}
        self.onsets = []          # (time, symbols)
        self.presses = []
        self.evaluated = 0

    def add_onset(self, symbols, t):
        self.onsets.append((t, list(symbols)))

    def add_press(self, t):
        self.presses.append(t)

    def update(self, now) -> bool:
        """Score every onset whose response window is over; True if anything changed."""
        changed = False
        while self.evaluated < len(self.onsets) and self.onsets[self.evaluated][0] + self.window[1] <= now:
            t, lit = self.onsets[self.evaluated]
            pressed = any(t + self.window[0] <= p <= t + self.window[1] for p in self.presses)
            lit = set(lit)
            for s in self.symbols:
                if s in lit:
                    p = self.hit if pressed else 1 - self.hit
                else:
                    p = self.fa if pressed else 1 - self.fa
                self.logp[s] += math.log(p)
            self.evaluated += 1
            changed = True
        return changed

    def probabilities(self):
        top = max(self.logp.values())
        w = {s: math.exp(v - top) for s, v in self.logp.items()}
        total = sum(w.values())
        return {s: v / total for s, v in w.items()}


def fail(message: str):
    print(message, file=sys.stderr, flush=True)
    sys.exit(2)
