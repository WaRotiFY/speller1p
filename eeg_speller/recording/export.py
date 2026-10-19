"""Portable session and experiment archives with integrity results."""
from pathlib import Path
import csv
import json
import zipfile
from eeg_speller.recording.storage import check_integrity, write_checksums


def export(path):
    path = Path(path).resolve()
    sessions = [path] if (path / "session.json").exists() else sorted(p.parent for p in path.rglob("session.json"))
    if not sessions:
        raise ValueError("no sessions found")
    summary = []
    for session in sessions:
        result = check_integrity(session)
        (session / "derived" / "integrity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        write_checksums(session)
        metrics_path = session / "derived" / "metrics.json"
        if metrics_path.exists():
            summary.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    destination = path.parent / f"{path.name}.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.txt", "eeg-speller/events@1\nSee docs/DATA_MODEL.md for field definitions.\nIntegrity results: derived/integrity.json.\n")
        for session in sessions:
            for file in sorted(session.rglob("*")):
                if file.is_file() and not file.is_symlink():
                    archive.write(file, file.relative_to(path.parent))
        if len(sessions) > 1:
            rows = []
            for session in sessions:
                table = session / "tables" / "epochs.csv"
                with table.open(encoding="utf-8-sig", newline="") as fh:
                    rows.extend(csv.DictReader(fh))
            if rows:
                import io
                buffer = io.StringIO()
                fields = sorted({k for row in rows for k in row})
                writer = csv.DictWriter(buffer, fields)
                writer.writeheader(); writer.writerows(rows)
                archive.writestr("experiment_epochs.csv", buffer.getvalue())
            if summary:
                import io
                buffer = io.StringIO()
                fields = sorted({k for row in summary for k in row})
                writer = csv.DictWriter(buffer, fields)
                writer.writeheader(); writer.writerows(summary)
                archive.writestr("experiment_metrics.csv", buffer.getvalue())
    return destination
