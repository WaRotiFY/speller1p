"""Play an EDF recording into LSL as if it came from a live amplifier.

The speller never opens EDF files. This tool turns a recording into two LSL
streams, so every downstream module works exactly as with real hardware:

- ``EEG`` stream (name ``--eeg-name``): the EDF signals in microvolts, in real time;
- ``SpellerStimuli`` markers: the recorded flashes (annotations
  ``speller/<epoch>/<flash>/<row|column|symbol>/<index>/<repetition>``) with their
  recorded timing, plus a ``done`` event after the last flash of each epoch.

Playback starts when the speller begins its first epoch. The recording's epochs
are replayed in order at their recorded times; the speller simply receives them.

    python stimulus_kit/tools/edf_to_lsl.py runs/edf_demo/test.edf
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pylsl  # noqa: E402
from speller_link import SpellerLink  # noqa: E402

SCALE_TO_UV = {"uv": 1.0, "µv": 1.0, "μv": 1.0, "mv": 1e3, "v": 1e6, "nv": 1e-3}


def read_edf(path):
    import pyedflib
    with pyedflib.EdfReader(path) as r:
        labels = r.getSignalLabels()
        rates = {float(r.getSampleFrequency(i)) for i in range(len(labels))}
        if len(rates) != 1:
            raise SystemExit("all EDF signals must share one sample rate for this player")
        data = np.stack([r.readSignal(i) * SCALE_TO_UV.get(r.getPhysicalDimension(i).strip().lower(), 1.0)
                         for i in range(len(labels))])
        onsets, _, texts = r.readAnnotations()
    markers = []
    for onset, text in zip(onsets, texts):
        parts = text.split("/")
        if len(parts) == 6 and parts[0] == "speller":
            markers.append({"t": float(onset), "epoch": int(parts[1]), "flash": int(parts[2]),
                            "kind": parts[3], "index": int(parts[4])})
    markers.sort(key=lambda m: m["t"])
    return labels, rates.pop(), data, markers


def group_symbols(marker, symbols, layout):
    rows, cols = layout
    if marker["kind"] == "row":
        return symbols[marker["index"] * cols:(marker["index"] + 1) * cols]
    if marker["kind"] == "column":
        return symbols[marker["index"]::cols]
    return [symbols[marker["index"]]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edf")
    ap.add_argument("--eeg-name", default="EDFReplayEEG")
    ap.add_argument("--chunk-ms", type=int, default=20)
    ap.add_argument("--no-wait", action="store_true", help="start immediately, without the speller")
    args = ap.parse_args()

    labels, fs, data, markers = read_edf(args.edf)
    if not markers:
        raise SystemExit("EDF has no speller stimulus annotations")
    info = pylsl.StreamInfo(args.eeg_name, "EEG", len(labels), fs, pylsl.cf_float32, "edf-" + os.path.basename(args.edf))
    chans = info.desc().append_child("channels")
    for name in labels:
        ch = chans.append_child("channel")
        ch.append_child_value("label", name)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    eeg = pylsl.StreamOutlet(info, chunk_size=max(1, int(fs * args.chunk_ms / 1000)))

    link = SpellerLink(name="edf_replay", role="stimulus", wait_for_speller=0 if args.no_wait else 60)
    symbols, layout = None, (6, 6)
    if not args.no_wait:
        epoch = link.wait_epoch()
        if epoch is None:
            return
        symbols, layout = epoch.symbols, epoch.layout
    if symbols is None:
        raise SystemExit("--no-wait needs the speller's symbols; run with the speller instead")
    time.sleep(1.0)                                   # let decoders attach to the EEG stream
    t0 = pylsl.local_clock() + 0.2                    # LSL time of EDF second 0
    last_epoch = {}
    for m in markers:
        last_epoch[m["epoch"]] = m["flash"]
    n = data.shape[1]
    sent, mi = 0, 0
    print(f"Playing {args.edf}: {len(labels)} channels at {fs:g} Hz, {len(markers)} flashes", flush=True)
    while sent < n and not link.ended:
        t_now = pylsl.local_clock() - t0
        upto = min(n, int(t_now * fs))
        if upto > sent:
            # LSL stamps the last sample; earlier ones are back-dated by 1/fs.
            eeg.push_chunk(data[:, sent:upto].T.astype(np.float32).tolist(), t0 + (upto - 1) / fs)
            sent = upto
        while mi < len(markers) and markers[mi]["t"] <= t_now:
            m = markers[mi]
            ts = t0 + m["t"]
            link.flash(group_symbols(m, symbols, layout), timestamp=ts, epoch=m["epoch"])
            if m["flash"] == last_epoch[m["epoch"]]:
                link.done(epoch=m["epoch"])
            mi += 1
        time.sleep(args.chunk_ms / 2000)
    print("EDF playback finished", flush=True)
    time.sleep(3)


if __name__ == "__main__":
    main()
