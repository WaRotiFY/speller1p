"""Start the speller core plus example modules in one command (for demos and checks).

    python scripts/run_external_demo.py sim --text "привет"
    python scripts/run_external_demo.py edf --edf runs/edf_demo/test.edf --model runs/edf_demo/p300_model.json
    python scripts/run_external_demo.py matrix --keyboard        # pygame window, SPACE = EEG

Each module is a separate process, exactly as the physiologists' programs would be.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "stimulus_kit"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demo", choices=["sim", "edf", "matrix", "rsvp"])
    ap.add_argument("--profile", default="configs/ext_default.yaml")
    ap.add_argument("--target", default="", help="copy-spelling phrase given to the speller (optional)")
    ap.add_argument("--text", default="привет", help="phrase the simulated participant types (sim)")
    ap.add_argument("--d-prime", default="1.0")
    ap.add_argument("--edf", default="runs/edf_demo/test.edf")
    ap.add_argument("--model", default="runs/edf_demo/p300_model.json")
    ap.add_argument("--keyboard", action="store_true")
    ap.add_argument("--output", default="runs/external")
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args()

    py = sys.executable
    target = args.target or (args.text if args.demo == "sim" else "")
    core = subprocess.Popen([py, "-m", "eeg_speller.external", "--profile", args.profile,
                             "--target", target, "--output", args.output],
                            cwd=ROOT, stdout=subprocess.PIPE, text=True)
    time.sleep(1.5)
    modules = []
    if args.demo == "sim":
        modules.append([py, str(KIT / "examples" / "headless_sim.py"), "--text", args.text,
                        "--d-prime", args.d_prime])
    elif args.demo == "edf":
        modules.append([py, str(KIT / "decoders" / "p300_lsl_decoder.py"), "--eeg-name", "EDFReplayEEG",
                        "--model", args.model])
        modules.append([py, str(KIT / "tools" / "edf_to_lsl.py"), args.edf])
    else:
        script = "pygame_matrix.py" if args.demo == "matrix" else "pygame_rsvp.py"
        modules.append([py, str(KIT / "examples" / script)] + (["--keyboard"] if args.keyboard else []))
    procs = []
    for cmd in modules:
        procs.append(subprocess.Popen(cmd, cwd=ROOT))
        time.sleep(1.0)
    try:
        if args.demo == "edf":
            procs[-1].wait(timeout=args.timeout)          # playback finished
            time.sleep(6)
            core.send_signal(2)
        out, _ = core.communicate(timeout=args.timeout)
    except KeyboardInterrupt:
        core.send_signal(2)
        out, _ = core.communicate(timeout=30)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
    tail = out[out.rfind("\n{"):] if "\n{" in out else out
    print(tail.strip())


if __name__ == "__main__":
    main()
