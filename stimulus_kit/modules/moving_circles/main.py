"""Кружки по эллипсам — сменный модуль стимуляции для спеллера."""
import os

from module_tools import keyboard_of, module_args
from speller_link import SpellerLink
from experiment import Experiment

args = module_args(__doc__, os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json"))
data_config = args.config
symbols, layout = keyboard_of(data_config)
link = SpellerLink(name="moving_circles", role="stimulus+decoder" if args.keyboard else "stimulus",
                   keyboard=symbols, layout=layout)

exp = Experiment(data_config, link, keyboard=args.keyboard, windowed=args.windowed)
exp.fit()
