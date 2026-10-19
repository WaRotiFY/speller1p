"""Движение букв (Артём, 0.0) — сменный модуль стимуляции для спеллера."""
import os

from module_tools import load_settings, module_args
from speller_link import SpellerLink
from experiment import Experiment

args = module_args(__doc__, os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json"))
data_config = load_settings(args.settings)
link = SpellerLink(name="artem_moving_letters", role="stimulus+decoder" if args.keyboard else "stimulus")

exp = Experiment(data_config, link, keyboard=args.keyboard, windowed=args.windowed)
exp.fit()
