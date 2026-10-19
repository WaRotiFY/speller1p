"""Session manifest, derived tables and integrity checks."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import uuid
import yaml


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_events(session_dir):
    with (Path(session_dir) / "events.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


class SessionStorage:
    """Storage; NEED-F-12/14; DATA_MODEL; Q-30."""
    def __init__(self, root: Path, cfg, rep, models: dict, replay_of=None):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_id = f"{stamp}_{cfg.participant_id}_{cfg.run_mode}_{uuid.uuid4().hex[:4]}"
        self.directory = Path(root)
        if cfg.experiment_id:
            self.directory /= cfg.experiment_id
        self.directory /= self.session_id
        for name in ("texts", "tables", "derived", "external"):
            (self.directory / name).mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.manifest = {"schema_version": "eeg-speller/events@1", "session_id": self.session_id,
                         "participant_id": cfg.participant_id, "experiment_id": cfg.experiment_id,
                         "task_id": cfg.task_id, "started_utc": utc_now(), "ended_utc": None,
                         "clock": {"kind": "monotonic" if cfg.run_mode == "live" else "virtual", "t0_ns": 0},
                         "run_mode": cfg.run_mode, "llm_mode": cfg.mode["llm"],
                         "paradigm": cfg.stimulus["paradigm"], "repertoire": list(rep.symbols),
                         "seed": cfg.seed, "code_version": "0.1.0", "models": models,
                         "hardware": {"platform": platform.platform(), "cpu": platform.processor(),
                                      "gpu": None, "audio_device": None},
                         "replay_of": replay_of, "counts": {}, "status": "running"}
        (self.directory / "session.json").write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.directory / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), allow_unicode=True, sort_keys=False), encoding="utf-8")
        (self.directory / "external" / "refs.json").write_text("[]\n", encoding="utf-8")

    def finish(self, target, primary, secondary, n_epochs, n_events, status="completed"):
        for name, text in (("target", target), ("primary_final", primary), ("secondary_final", secondary)):
            (self.directory / "texts" / f"{name}.txt").write_text(text, encoding="utf-8")
        self.manifest.update(ended_utc=utc_now(), status=status,
                             counts={"epochs": n_epochs, "events": n_events,
                                     "corrections": sum(e["type"] == "correction_completed" for e in read_events(self.directory))})
        (self.directory / "session.json").write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        build_tables(self.directory)
        integrity = check_integrity(self.directory)
        (self.directory / "derived" / "integrity.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")
        write_checksums(self.directory)


def build_tables(session_dir):
    session_dir = Path(session_dir)
    manifest = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    events = read_events(session_dir)
    epochs = {}
    corrections = {}
    for event in events:
        typ, payload, eid = event["type"], event["payload"], event["epoch_id"]
        if eid is not None:
            row = epochs.setdefault(eid, {"session_id": manifest["session_id"], "participant_id": manifest["participant_id"],
                                         "experiment_id": manifest["experiment_id"], "task_id": manifest["task_id"],
                                         "llm_mode": manifest["llm_mode"], "paradigm": manifest["paradigm"], "epoch_id": eid})
            if typ == "epoch_started":
                row.update(payload)
                row["t_start_ns"] = event["t_ns"]
            elif typ == "physiology_observation":
                row["phys_log_p"] = payload["log_p"]
                row["binary_flash_accuracy"] = (payload.get("module_metrics") or {}).get("binary_flash_accuracy")
                if "raw" in payload:                 # last update as the module sent it
                    row["phys_raw"] = payload["raw"]
                    row["phys_updates"] = payload.get("update")
            elif typ == "epoch_timeline":
                row["t_epoch_start_lsl"] = payload["lsl_s"].get("epoch_start_lsl")
                row.update({f"t_{k}": v for k, v in payload["since_epoch_start_ms"].items()})
                row["n_flashes"] = payload.get("flashes")
            elif typ == "prior_computed":
                row["prior_log_p"] = payload["log_p"]
                row["prior_latency_ms"] = payload.get("wall_latency_ms")
            elif typ == "posterior_computed":
                row["post_log_p"] = payload["log_p"]
                row["fusion"] = payload["fusion"]
                row.update(payload["params"])
            elif typ == "symbol_decided":
                row.update({"decided": payload["symbol"], **{k: payload[k] for k in ("confidence", "margin", "uncertain", "cold_start", "abstained")}})
                row["t_end_ns"] = event["t_ns"]
            elif typ == "presentation_feedback":
                row["top_k_feedback"] = payload["top_k"]
            elif typ == "word_candidates":
                row["top1_word"] = payload["candidates"][0][0] if payload["candidates"] else None
        if typ == "correction_requested":
            corrections[payload["correction_id"]] = {"session_id": manifest["session_id"],
                                                      **payload, "t_requested_ns": event["t_ns"]}
        elif typ == "correction_completed":
            corrections.setdefault(payload["correction_id"], {}).update(payload)
            corrections[payload["correction_id"]]["latency_ms"] = payload.get("wall_latency_ms")
            corrections[payload["correction_id"]]["t_completed_ns"] = event["t_ns"]
    write_csv(session_dir / "tables" / "epochs.csv", list(epochs.values()))
    write_csv(session_dir / "tables" / "corrections.csv", list(corrections.values()))


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def check_integrity(session_dir):
    session_dir = Path(session_dir)
    events = read_events(session_dir)
    errors = []
    if [e["seq"] for e in events] != list(range(len(events))):
        errors.append("non-contiguous event sequence")
    if not events or events[-1]["type"] != "session_ended":
        errors.append("missing session_ended")
    epochs = {e["epoch_id"] for e in events if e["type"] == "epoch_started"}
    for eid in epochs:
        types = {e["type"] for e in events if e["epoch_id"] == eid}
        if "epoch_aborted" in types:
            continue
        for required in ("epoch_started", "physiology_observation", "posterior_computed", "symbol_decided"):
            if required not in types:
                errors.append(f"epoch {eid} missing {required}")
    requested = {e["payload"]["correction_id"] for e in events if e["type"] == "correction_requested"}
    completed = {e["payload"]["correction_id"] for e in events if e["type"] == "correction_completed"}
    if requested != completed:
        errors.append("unmatched correction requests")
    updates = [e["payload"]["text"] for e in events if e["type"] == "primary_text_updated"]
    final = (session_dir / "texts" / "primary_final.txt").read_text(encoding="utf-8")
    if (updates[-1] if updates else "") != final:
        errors.append("primary final text disagrees with events")
    return {"complete": not errors, "errors": errors, "n_events": len(events), "n_epochs": len(epochs)}


def write_checksums(session_dir):
    session_dir = Path(session_dir)
    lines = []
    for file in sorted(session_dir.rglob("*")):
        if file.is_file() and file.name != "checksums.sha256":
            lines.append(f"{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.relative_to(session_dir)}")
    (session_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
