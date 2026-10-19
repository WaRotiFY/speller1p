"""Command-line entry points for simulation, replay and evaluation."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from eeg_speller.core.config import Config, load_config
from eeg_speller.epoch.loop import EpochRunner, read_tasks
from eeg_speller.evaluation.bench import bench_llm
from eeg_speller.evaluation.experiment import experiment
from eeg_speller.evaluation.metrics import save_session_metrics
from eeg_speller.recording.export import export
from eeg_speller.recording.replay import ReplayPhysiology
from eeg_speller.recording.replay_external import is_external, replay_external


def main():
    parser = argparse.ArgumentParser(prog="python -m eeg_speller")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "experiment", "bench-llm"):
        p = sub.add_parser(command)
        p.add_argument("--config", default="configs/default.yaml")
        if command == "run":
            p.add_argument("--output", default="runs")
            p.add_argument("--verbose", action="store_true")
    p = sub.add_parser("replay")
    p.add_argument("session_dir")
    p.add_argument("--config", default=None,
                   help="simulated session: a full profile; operator-window session: changes merged over its config")
    p.add_argument("--llm", choices=["on", "off"], help="operator-window session: with or without language models")
    p.add_argument("--output", default="runs")
    p = sub.add_parser("export")
    p.add_argument("path")
    args = parser.parse_args()
    if args.command == "export":
        print(export(args.path))
        return
    if args.command == "replay" and is_external(Path(args.session_dir)):
        result = replay_external(args.session_dir, args.output, args.config, args.llm)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "replay":
        source = ReplayPhysiology(args.session_dir)
        cfg = load_config(args.config or (Path(args.session_dir) / "config.yaml"))
        data = cfg.model_dump()
        data["run_mode"] = "replay"
        data["task_id"] = source.manifest["task_id"]
        data["participant_id"] = source.manifest["participant_id"]
        cfg = Config.model_validate(data)
        target = (Path(args.session_dir) / "texts" / "target.txt").read_text(encoding="utf-8")
        session = EpochRunner(cfg, args.output, source).run(target)
        print(json.dumps({"session": str(session), "metrics": save_session_metrics(session)}, ensure_ascii=False))
        return
    cfg = load_config(args.config)
    if args.command == "run":
        target = read_tasks()[cfg.task_id]
        session = EpochRunner(cfg, args.output, verbose=args.verbose).run(target)
        print(json.dumps({"session": str(session), "metrics": save_session_metrics(session)}, ensure_ascii=False))
    elif args.command == "bench-llm":
        print(json.dumps(bench_llm(cfg), ensure_ascii=False, indent=2))
    elif args.command == "experiment":
        result = experiment(cfg)
        print(json.dumps({k: v for k, v in result.items() if k != "sessions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
