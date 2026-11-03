"""Проигрывание записи ЭЭГ: BCI Competition II (английские слова).

The same program as ../eeg_replay (SpellerLink, one recorded symbol per speller epoch),
with the settings.json of this folder: BCI Competition II data set IIb, Wadsworth,
calibration on sessions 10–11, replay of session 12 (foodmoothampiecaketunazygot4567).
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("eeg_replay_main", HERE.parent / "eeg_replay" / "main.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)

if __name__ == "__main__":
    replay.main(HERE / "settings.json")
