"""Run the speller core for external modules without a browser.

    python -m eeg_speller.external --profile configs/ext_default.yaml --target "привет"

For the operator window use ``python -m eeg_speller.external.server`` instead.
"""
import argparse
import json
from pathlib import Path
import time

from eeg_speller.core.config import load_config
from eeg_speller.external.controller import ExternalSession


def main():
    ap = argparse.ArgumentParser(description="EEG speller core for removable stimulus/decoder modules (LSL)")
    ap.add_argument("--profile", type=Path, default=Path("configs/ext_default.yaml"))
    ap.add_argument("--target", default="", help="phrase for copy-spelling; empty = free typing")
    ap.add_argument("--participant", default=None)
    ap.add_argument("--output", type=Path, default=Path("runs/external"))
    args = ap.parse_args()
    cfg = load_config(args.profile)
    if args.participant:
        cfg = cfg.model_copy(update={"participant_id": args.participant})
    target = args.target.lower().replace("ё", "е")
    session = ExternalSession(cfg, target, args.output, profile=args.profile.name).start()
    print(f"Speller running (session {session.session_id}). Start a stimulus module now. Ctrl+C to stop.",
          flush=True)
    last = None
    try:
        while not session.finished:
            state = session.monitor()
            line = (state["phase"], state["text"], state["epoch_id"])
            if line != last:
                print(f"[{state['phase']}] epoch {state['epoch_id']}: {state['text']!r}", flush=True)
                last = line
            time.sleep(0.2)
    except KeyboardInterrupt:
        session.request_stop("operator_stopped")
        session.wait_finished(10)
    state = session.monitor()
    print(json.dumps({"text": state["text"], "corrected": state["corrected"], "end": state["end_reason"],
                      "error": state["error"], "stats": state["stats"], "directory": state["directory"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
