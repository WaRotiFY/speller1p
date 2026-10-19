"""Симулятор без экрана — проверка спеллера без человека и без ЭЭГ.

Симулированный испытуемый «смотрит» на буквы фразы из settings.json, а симулированный
мозг отвечает на вспышки строк и столбцов с качеством сигнала d′.
"""
import os

from module_tools import keyboard_of, module_args
from speller_link import SpellerLink
from headless_sim import run

args = module_args(__doc__, os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json"))
cfg = args.config
symbols, layout = keyboard_of(cfg)
link = SpellerLink(name="simulator", role="stimulus+decoder", keyboard=symbols, layout=layout)
run(cfg["sentence"].lower().replace("ё", "е"), d_prime=float(cfg["d_prime"]), flash_ms=cfg["flash_ms"],
    isi_ms=cfg["isi_ms"], repetitions=cfg["n_trials"], seed=int(cfg["seed"]), link=link)
