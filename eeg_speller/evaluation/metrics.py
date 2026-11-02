"""Metrics computed exclusively from recorded session artifacts."""
from pathlib import Path
import json
import math
import re
import numpy as np
from eeg_speller.recording.storage import read_events
from eeg_speller.text.normalize import normalize, words


def alignment(a, b):
    """Return edit distance and equality count in an optimal alignment."""
    n, m = len(a), len(b)
    dp = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): dp[i][0] = (i, 0)
    for j in range(1, m + 1): dp[0][j] = (j, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            equal = a[i-1] == b[j-1]
            choices = [(dp[i-1][j][0] + 1, dp[i-1][j][1]),
                       (dp[i][j-1][0] + 1, dp[i][j-1][1]),
                       (dp[i-1][j-1][0] + (not equal), dp[i-1][j-1][1] + equal)]
            dp[i][j] = min(choices, key=lambda x: (x[0], -x[1]))
    return dp[n][m]


def chrf(a, b, max_n=6):
    from collections import Counter
    score = []
    for n in range(1, max_n + 1):
        left = Counter(a[i:i+n] for i in range(max(0, len(a)-n+1)))
        right = Counter(b[i:i+n] for i in range(max(0, len(b)-n+1)))
        overlap = sum((left & right).values())
        precision = overlap / max(1, sum(right.values()))
        recall = overlap / max(1, sum(left.values()))
        score.append(2 * precision * recall / (precision + recall) if precision + recall else 0)
    return float(np.mean(score))


def session_metrics(session_dir):
    # IMPROV[Q-20]
    # IMPROV[Q-22]
    # IMPROV[Q-25]
    # STUB[Q-24]
    session_dir = Path(session_dir)
    events = read_events(session_dir)
    target = (session_dir / "texts" / "target.txt").read_text(encoding="utf-8")
    primary = (session_dir / "texts" / "primary_final.txt").read_text(encoding="utf-8")
    secondary = (session_dir / "texts" / "secondary_final.txt").read_text(encoding="utf-8")
    n_target = normalize(target)
    n_primary = normalize(primary)
    n_secondary = normalize(secondary)
    distance, correct = alignment(n_target, n_primary)
    sec_distance, sec_correct = alignment(n_target, n_secondary)
    # ASSUMPTION[MA-24]
    starts = [e["t_ns"] for e in events if e["type"] == "epoch_started"]
    decisions = [e for e in events if e["type"] == "symbol_decided"]
    duration_min = ((decisions[-1]["t_ns"] - starts[0]) / 60e9) if starts and decisions else 0
    symbol_targets = {e["epoch_id"]: e["payload"].get("sim_target_symbol", e["payload"].get("target_symbol")) for e in events if e["type"] == "epoch_started"}
    selected = [e for e in decisions if e["payload"]["symbol"] is not None]
    accuracy = sum(e["payload"]["symbol"] == symbol_targets[e["epoch_id"]] for e in selected) / len(selected) if selected else None
    observations = [e["payload"].get("module_metrics", {}) for e in events if e["type"] == "physiology_observation"]
    # ASSUMPTION[MA-23]
    binary = float(np.mean([m["binary_flash_accuracy"] for m in observations if m and "binary_flash_accuracy" in m])) if observations else None
    binary_mean = float(np.mean([m["binary_mean_accuracy"] for m in observations if m and "binary_mean_accuracy" in m])) if observations else None
    # ASSUMPTION[MA-25]
    speed = correct / duration_min if duration_min else None
    secondary_speed = sec_correct / duration_min if duration_min else None
    # ASSUMPTION[MA-27]
    manifest = json.loads((session_dir / "session.json").read_text())
    p, n = accuracy, len(manifest["repertoire"])
    if p is None:
        itr = None
    else:
        bit = math.log2(n) + (p * math.log2(p) if p else 0) + ((1-p) * math.log2((1-p)/(n-1)) if p < 1 else 0)
        itr = bit * len(decisions) / duration_min if duration_min else None
    tw, pw, sw = words(target), words(primary), words(secondary)
    _, all_correct = alignment(tw, sw)
    error_before = [i for i, w in enumerate(tw) if i >= len(pw) or pw[i] != w]
    restored = sum(i < len(sw) and sw[i] == tw[i] for i in error_before)
    # ASSUMPTION[MA-28]
    word_recovery_errors = restored / len(error_before) if error_before else None
    word_accuracy_all = all_correct / len(tw) if tw else None
    # ASSUMPTION[MA-29]
    semantic_stub = chrf(n_target, n_secondary)
    tts = [e["payload"]["latency_ms"] for e in events if e["type"] == "tts_audio_started"]
    # ASSUMPTION[MA-32]
    result = {"session_id": json.loads((session_dir / "session.json").read_text(encoding="utf-8"))["session_id"],
            "binary_flash_accuracy": binary, "binary_mean_accuracy": binary_mean,
            "symbol_accuracy": accuracy, "cer": distance / len(n_target) if n_target else None,
            "speed_primary_cpm": speed, "speed_secondary_cpm": secondary_speed,
            "duration_min": duration_min, "itr_reference_bpm": itr,
            "word_recovery_errors": word_recovery_errors, "word_accuracy_all": word_accuracy_all,
            "semantic_chrf_stub": semantic_stub, "semantic_stub_pass": semantic_stub >= 0.7,
            "tts_latency_p95_ms": float(np.percentile(tts, 95)) if tts else None,
            "top3": None, "top5": None, "top10": None,
            "n_target_chars": len(n_target), "n_selected_epochs": len(selected)}
    external_metrics = session_dir / 'derived' / 'operator_metrics.json'
    if external_metrics.is_file():
        m = json.loads(external_metrics.read_text())
        result.update(symbol_accuracy=m['accuracy'], cer=m['cer_primary'], cer_secondary=m['cer_secondary'],
                      speed_primary_cpm=m['speed_cpm'], speed_secondary_cpm=m['corrected_speed_cpm'],
                      itr_reference_bpm=m['itr_bpm'], duration_min=m['active_seconds']/60,
                      selection_rate_spm=m['selection_rate_spm'], itr_method=m['itr_method'])
    return result



def save_session_metrics(session_dir):
    result = session_metrics(session_dir)
    output = Path(session_dir) / "derived" / "metrics.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
