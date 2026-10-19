"""Example EEG decoder module: P300 from any LSL EEG stream -> symbol likelihoods.

It reuses the P300 processing of this repository (band-pass, -200..800 ms windows,
calibrated Gaussian LDA). It listens to:
- an EEG stream (a real amplifier, or ``tools/edf_to_lsl.py``);
- ``SpellerStimuli`` flash markers from whatever stimulus program runs;
and publishes ``SpellerLikelihood`` updates after every scored flash.

Swap it for the physiologists' own decoder at any time: the speller only reads
``SpellerLikelihood``.

    python stimulus_kit/decoders/p300_lsl_decoder.py --profile configs/p300_decoder.yaml --eeg-name EDFReplayEEG
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stimulus_kit"))
import pylsl  # noqa: E402
from speller_link import SpellerLink, STIMULI_STREAM  # noqa: E402
from eeg_speller.core.config import load_config  # noqa: E402
from eeg_speller.physiology.p300 import CalibratedP300, P300Features  # noqa: E402
from eeg_speller.physiology.sources import LSLSource  # noqa: E402


class FlashCollector:
    """Collects flash markers from every stimulus stream, in this computer's LSL clock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.flashes = {}            # epoch -> list of (lsl_time, symbols)
        self.done = set()
        self.inlets = {}
        self.resolver = pylsl.ContinuousResolver("name", STIMULI_STREAM)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        last = 0
        while True:
            if time.monotonic() - last > 0.5:
                for info in self.resolver.results():
                    if info.source_id() not in self.inlets:
                        inlet = pylsl.StreamInlet(info, processing_flags=pylsl.proc_clocksync)
                        inlet.open_stream(timeout=2)
                        self.inlets[info.source_id()] = inlet
                last = time.monotonic()
            for sid, inlet in list(self.inlets.items()):
                try:
                    samples, stamps = inlet.pull_chunk(timeout=0.0)
                except Exception:
                    self.inlets.pop(sid, None)
                    continue
                for s, ts in zip(samples, stamps):
                    msg = json.loads(s[0])
                    with self.lock:
                        if "flash" in msg:
                            self.flashes.setdefault(msg["epoch"], []).append((ts, msg["flash"]))
                        elif msg.get("event") == "done":
                            self.done.add(msg["epoch"])
            time.sleep(0.002)

    def epochs(self):
        with self.lock:
            return sorted(self.flashes)

    def get(self, epoch):
        with self.lock:
            return list(self.flashes.get(epoch, [])), epoch in self.done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", type=Path, default=ROOT / "configs" / "p300_decoder.yaml",
                    help="YAML with a physiology block (channels, filters, model_path, source)")
    ap.add_argument("--eeg-name", help="LSL EEG stream name (overrides the profile's source)")
    ap.add_argument("--model", type=Path, help="calibrated P300 model JSON (overrides model_path)")
    args = ap.parse_args()

    cfg = dict(load_config(args.profile).physiology)
    if args.model:
        cfg["model_path"] = str(args.model.resolve())
    features = P300Features(cfg)
    model_path = Path(cfg["model_path"])
    model = CalibratedP300(model_path if model_path.is_absolute() else ROOT / model_path, features)
    source_cfg = dict(cfg["source"], kind="lsl", marker_stream_name="P300DecoderEcho")
    if args.eeg_name:
        source_cfg.update(stream_field="name", stream_value=args.eeg_name, units="uV")
    link = SpellerLink(name="p300_lsl_decoder", role="decoder")
    collector = FlashCollector()
    print("Waiting for the EEG stream…", flush=True)
    source = LSLSource(source_cfg, features.channels)
    print(f"EEG: {source.fs:g} Hz, channels {features.channels}", flush=True)

    # Epoch ids come with the flash markers, so decoding does not depend on
    # following the control stream; the speller buffers early answers.
    state = {}                                       # epoch -> {"scores", "logits", "final"}
    while not link.ended:
        for eid in collector.epochs():
            st = state.setdefault(eid, {"scores": {}, "logits": None, "final": False})
            if not st["final"]:
                decode_step(link, eid, st, collector, source, features, model)
        time.sleep(0.01)
    source.close()


def decode_step(link, eid, st, collector, source, features, model):
    flashes, done = collector.get(eid)
    if st["logits"] is None:
        symbols = link.epoch.symbols if link.epoch else sorted({s for _, f in flashes for s in f})
        st["logits"] = {s: 0.0 for s in symbols}
    scores, logits = st["scores"], st["logits"]
    changed = False
    for i, (onset, symbols) in enumerate(flashes):
        if i in scores:
            continue
        window = source.window(onset, features.window)
        if window is None:
            continue                                  # post-stimulus EEG not here yet
        x = features.extract(window, source.fs)
        scores[i] = None if x is None else model.llr(x)
        if scores[i] is not None:
            for s in symbols:
                if s in logits:
                    logits[s] += scores[i]
            changed = True
    complete = done and len(scores) == len(flashes)
    if changed or complete:
        top = max(logits.values())
        p = {s: float(np.exp(v - top)) for s, v in logits.items()}
        usable = any(v is not None and v > 0 for v in scores.values())
        link.send_likelihood(p, final=complete, epoch=eid, usable=usable,
                             info={"flashes_scored": sum(v is not None for v in scores.values()),
                                   "flashes_rejected": sum(v is None for v in scores.values())})
    st["final"] = complete


if __name__ == "__main__":
    main()
