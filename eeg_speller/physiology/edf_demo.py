"""Generate independent synthetic calibration/test EEG and fit a small P300 model.

Targets affect waveform generation only. Test EDF annotations contain flash IDs,
groups and timestamps; the application never loads the test answer sidecar.
"""
from datetime import datetime
from pathlib import Path
import argparse
import json
import math
import numpy as np
from scipy.signal import butter, sosfiltfilt

from eeg_speller.core.config import load_config
from eeg_speller.physiology.p300 import P300Features, CalibratedP300
from eeg_speller.physiology.edf_io import write_edf
from eeg_speller.physiology.sources import EDFSource
from eeg_speller.physiology.stimuli import make_plan

CHANNELS = ["Fz", "Cz", "Pz", "P3", "P4", "O1", "O2", "EOG"]
GAINS = np.array([0.35, 0.7, 1.0, 0.9, 0.85, 0.5, 0.45, 0.05])


def generate_edf(cfg, text, output, *, signal_seed, amplitude_uv=9.0):
    if not text or any(ch not in cfg.repertoire["symbols"] for ch in text):
        raise ValueError("text must be nonempty and use the speller repertoire")
    fs = 256
    labels, events = [], []
    cursor = 1.0
    for eid, char in enumerate(text):
        target = cfg.repertoire["symbols"].index(char)
        plan = make_plan(cfg, eid)
        for flash in plan:
            events.append((cursor + flash["onset_ms"] / 1000, eid, flash))
            labels.append({"epoch_id": eid, "flash_id": flash["id"], "is_target": target in flash["indices"]})
        cursor += (plan[-1]["onset_ms"] + cfg.stimulus["flash_ms"]) / 1000 + 1.5
    count = math.ceil(cursor + 1) * fs
    rng = np.random.default_rng(signal_seed)
    t = np.arange(count) / fs
    # Correlated low-frequency EEG, alpha rhythm, drift and independent channel noise.
    slow = sosfiltfilt(butter(2, [1, 20], btype="bandpass", fs=fs, output="sos"),
                      rng.normal(size=(len(CHANNELS), count)), axis=1)
    eeg = slow * 1.3
    eeg += np.array([0.7, 1, 0.6, 0.7, 0.8, 1.3, 1.4, 0.1])[:, None] * np.sin(2 * np.pi * 10 * t)
    eeg += 0.3 * np.sin(2 * np.pi * 0.15 * t + rng.uniform(0, 6, (len(CHANNELS), 1)))
    for (onset, _, _), label in zip(events, labels):
        a = round(onset * fs)
        tau = np.arange(round(0.8 * fs)) / fs
        erp = -0.5 * np.exp(-0.5 * ((tau - 0.12) / 0.035) ** 2)
        if label["is_target"]:
            erp += amplitude_uv * rng.uniform(0.85, 1.15) * np.exp(-0.5 * ((tau - 0.32) / 0.055) ** 2)
        eeg[:, a:a + len(tau)] += GAINS[:, None] * erp
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    notes = [(onset, cfg.stimulus["flash_ms"] / 1000,
              f"speller/{eid}/{flash['id']}/{flash['kind']}/{flash['index']}/{flash['repetition']}")
             for onset, eid, flash in events]
    write_edf(output, CHANNELS, fs, eeg, dimension="uV", physical_range=(-200, 200), annotations=notes,
              startdate=datetime(2026, 1, 1).date(), equipment="SYNTHETIC_P300", transducer="synthetic",
              prefilter="none")
    return labels


def calibrate(cfg, edf_path, labels, output):
    features = P300Features(cfg.physiology)
    source = EDFSource({"path": edf_path}, features.channels)
    x, y = [], []
    for label in labels:
        marker = source.markers[label["epoch_id"]][label["flash_id"]]
        feature = features.extract(source.window(marker["source_time_s"], features.window), source.fs)
        if feature is not None:
            x.append(feature)
            y.append(label["is_target"])
    return CalibratedP300.fit(x, y, features, output)


def main():
    parser = argparse.ArgumentParser(description="Synthetic EEG fixture; arbitrary text is encoded in P300 waveforms")
    parser.add_argument("--profile", type=Path, default=Path("configs/p300_decoder.yaml"))
    parser.add_argument("--text", required=True, help="text to encode in the independent test waveform")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/edf_demo"))
    parser.add_argument("--seed", type=int, default=19753, help="test signal seed; calibration uses another seed")
    args = parser.parse_args()
    cfg = load_config(args.profile)
    directory = args.output_dir
    training = generate_edf(cfg, "ажмтчя", directory / "calibration.edf", signal_seed=731)
    (directory / "calibration_labels.json").write_text(json.dumps(training), encoding="utf-8")
    calibrate(cfg, directory / "calibration.edf", training, directory / "p300_model.json")
    text = args.text.lower().replace("ё", "е")
    generate_edf(cfg, text, directory / "test.edf", signal_seed=args.seed)
    (directory / "expected_test_only.json").write_text(json.dumps({"text": text, "signal_seed": args.seed}, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {directory / 'test.edf'} (8 channels at 256 Hz) and calibrated P300 model.")
    print("Play it into LSL: python stimulus_kit/tools/edf_to_lsl.py " + str(directory / "test.edf"))


if __name__ == "__main__":
    main()
