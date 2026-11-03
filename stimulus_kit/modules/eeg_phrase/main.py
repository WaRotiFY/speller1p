"""Реальные ответы ЭЭГ на фразу задания — проверка языковых моделей на настоящем P300.

The module "spells" the operator's task phrase (passed with "--text" by the operator
window) on a matrix keyboard of the phrase's language. Every flash gets a real
single-flash decoder score drawn from a bank built from an open recording
(stimulus_kit/tools/open_data/eeg_score_pool.py): a target score if the flash contains
the letter the "participant" wants now, a non-target score otherwise. The module is
both stimulation and decoder (SpellerLink role "stimulus+decoder"); the speller adds the
fast LM, decides when to stop, picks the symbol and runs the slow correction — exactly
as with a person.

Copy-spelling: the wanted letter is the phrase character at the position of the typed
text, so after an error the participant goes on with the next letter.

Scores are drawn with a generator seeded by (seed, position, attempt): two sessions
with different language models get the same EEG for the same letter, so their
difference is the models' doing.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "stimulus_kit"))
sys.path.insert(0, str(ROOT / "stimulus_kit" / "tools" / "open_data"))

from module_tools import fail, load_settings  # noqa: E402
from speller_link import SpellerLink, now  # noqa: E402

BACKSPACE = "⌫"
KEYBOARDS = {   # (symbols, layout); "_" / " " is the space
    "en": (list("abcdefghijklmnopqrstuvwxyz") + ["_", ".", ",", BACKSPACE], (5, 6)),
    "ru": (list("абвгдежзийклмнопрстуфхцчшщъыьэюя") + ["_", ".", ",", BACKSPACE], (6, 6)),
}
SOURCES = {     # bank name -> (recording, marker map, calibration symbols)
    "bci2_iib": ("data/bci2_iib/bci2_iib_AAS.edf", "stimulus_kit/tools/open_data/markers_bci2_iib.json", 42,
                 "stimulus_kit/tools/open_data/get_bci2_iib.py"),
    "bnci2014_008_A01": ("data/bnci2014_008/bnci2014_008_A01.edf",
                         "stimulus_kit/tools/open_data/markers_bnci2014_008.json", 15,
                         "stimulus_kit/tools/open_data/get_bnci2014_008.py"),
}


def args_and_settings():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", default="", help="phrase to spell (the operator window passes the task)")
    ap.add_argument("--settings", default=str(HERE / "settings.json"))
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--keyboard", action="store_true", help="(ignored: this module has its own EEG)")
    a = ap.parse_args()
    cfg = load_settings(a.settings)
    a.text = (a.text or cfg.get("sentence") or "").lower().replace("ё", "е")
    if not a.text.strip():
        fail("нет фразы: выберите в окне оператора задание (не «свободный набор») или задайте sentence")
    return a, cfg


def bank(cfg):
    """Target and non-target scores of the chosen recording (built and cached on first use)."""
    from eeg_score_pool import build, load
    name = cfg.get("source", "bci2_iib")
    if name not in SOURCES:
        fail(f"unknown source {name!r}; known: {', '.join(SOURCES)}")
    edf, mmap, n_cal, getter = SOURCES[name]
    pool = ROOT / "runs" / "models" / f"pool_{name}_{cfg.get('pipeline', 'xdawn_riemann')}.npz"
    if not pool.is_file():
        if not (ROOT / edf).is_file():
            fail(f"нет записи {edf}: uv run --no-sync python {getter}")
        print(f"собираю банк ответов из {Path(edf).name} (один раз, ~1 мин)…", flush=True)
        build(str(ROOT / edf), str(ROOT / mmap), n_cal, cfg.get("pipeline", "xdawn_riemann"), str(pool))
    target, nontarget, meta = load(pool)
    print(f"банк ответов: {meta['recording']}, d′ = {meta['d_prime']}, "
          f"{meta['n_target']} целевых и {meta['n_nontarget']} нецелевых вспышек", flush=True)
    return target, nontarget


def language_of(text):
    cyr = sum("а" <= ch <= "я" for ch in text)
    lat = sum("a" <= ch <= "z" for ch in text)
    return "ru" if cyr >= lat else "en"


def symbol_for(ch, symbols):
    for s in symbols:
        if s == ch or (s in ("_", " ") and ch == " "):
            return s
    return None


def main():
    a, cfg = args_and_settings()
    target_pool, nontarget_pool = bank(cfg)
    symbols, layout = KEYBOARDS[language_of(a.text)]
    missing = sorted({ch for ch in a.text if symbol_for(ch, symbols) is None})
    if missing:
        print(f"предупреждение: в фразе есть символы не с клавиатуры: {missing} — их набрать нельзя", flush=True)
    print(f"фраза: {a.text!r}; клавиатура {layout[0]}×{layout[1]}", flush=True)

    link = SpellerLink(name="eeg_phrase", role="stimulus+decoder", keyboard=symbols, layout=layout)
    screen = None
    if cfg.get("show_window", True):
        spec = importlib.util.spec_from_file_location("eeg_replay_main", ROOT / "stimulus_kit/modules/eeg_replay/main.py")
        replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay)
        screen = replay.Screen(cfg, a.windowed or not cfg.get("fullscreen", False), symbols, layout)

    speed = float(cfg.get("speed", 1.0))
    flash_s, isi_s = cfg.get("flash_ms", 100) / 1000, cfg.get("isi_ms", 75) / 1000
    reps = int(cfg.get("max_repetitions", 15))
    seed = int(cfg.get("seed", 0))
    attempts: dict[int, int] = {}
    while True:
        epoch = link.wait_epoch(timeout=0.05)
        if link.ended or (screen and screen.quit):
            break
        if epoch is None:
            if screen:
                screen.draw(status="пауза" if link.paused else "ожидание спеллера…")
            continue
        pos = len(epoch.text)
        want_ch = a.text[pos] if pos < len(a.text) else " "
        want = symbol_for(want_ch, epoch.symbols)
        attempts[pos] = attempts.get(pos, 0) + 1
        rng = random.Random(f"{seed}/{pos}/{attempts[pos]}")
        rows = epoch.rows()
        groups = [list(r) for r in rows] + [[r[c] for r in rows if c < len(r)] for c in range(max(map(len, rows)))]
        plan = []                                   # whole epoch drawn up front: same EEG for every LM
        for _ in range(reps):
            order = groups[:]
            rng.shuffle(order)
            for g in order:
                pool = target_pool if want in g else nontarget_pool
                plan.append((g, float(pool[rng.randrange(len(pool))])))
        logl = {s: 0.0 for s in epoch.symbols}
        t = now() + 0.05
        for i, (group, score) in enumerate(plan, 1):
            while now() < t:
                time.sleep(0.0005)
            if screen:
                screen.draw(lit=group, cue=want if cfg.get("show_cue", True) else None, text=epoch.text,
                            status=f"буква {pos + 1}/{len(a.text)} · вспышка {i}")
            link.flash(group, timestamp=t)
            for s in group:
                logl[s] += score
            top = max(logl.values())
            link.send_likelihood({s: math.exp(v - top) for s, v in logl.items()})
            t += (flash_s + isi_s) / speed
            if i % len(groups) == 0 and not link.epoch_cancelled():
                link.trial_done()
            if link.should_stop() or link.epoch_cancelled():
                break
        if link.epoch_cancelled():
            continue
        link.done()
        top = max(logl.values())
        link.send_likelihood({s: math.exp(v - top) for s, v in logl.items()}, final=True)
        res = link.wait_result(timeout=10) or {}
        print(f"буква {pos + 1}: нужна {want!r}, выбрана {res.get('symbol')!r}, вспышек {i}", flush=True)
        if screen:
            until = time.monotonic() + cfg.get("rest_after_ms", 600) / 1000 / speed
            while time.monotonic() < until:
                screen.draw(chosen=res.get("symbol"), text=res.get("text", epoch.text))
                time.sleep(0.03)
    if screen:
        screen.close()


if __name__ == "__main__":
    main()
