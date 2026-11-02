"""The same definitions for the live operator and recorded external sessions."""
from collections import defaultdict
import math


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        row = [i]
        for j, cb in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j-1] + (ca != cb)))
        previous = row
    return previous[-1]


def performance(rows, target, primary, secondary, seconds):
    selected = [r for r in rows if r.get('decided') is not None]
    judged = [r for r in selected if r.get('correct') is not None]
    minutes = seconds / 60
    accuracy = sum(r['correct'] for r in judged) / len(judged) if judged else None
    groups = defaultdict(list)
    for row in judged:
        groups[row['n_symbols']].append(bool(row['correct']))
    bits = 0.0
    for n, outcomes in groups.items():
        p = sum(outcomes) / len(outcomes)
        # Below chance the standard symmetric-channel approximation is not useful
        # as an operational spelling rate. Report zero rather than a positive rate.
        b = (math.log2(n) + (p * math.log2(p) if p else 0)
             + ((1-p) * math.log2((1-p)/(n-1)) if p < 1 else 0)) if p > 1/n else 0
        bits += len(outcomes) * b
    return {
        'selected': len(selected), 'epochs': len(rows), 'accuracy': accuracy,
        'selection_rate_spm': len(selected) / minutes if minutes > 0 else None,
        'speed_cpm': len(primary) / minutes if minutes > 0 else None,
        'corrected_speed_cpm': len(secondary) / minutes if minutes > 0 else None,
        'cer_primary': edit_distance(target, primary) / len(target) if target else None,
        'cer_secondary': edit_distance(target, secondary) / len(target) if target else None,
        'itr_bpm': bits / minutes if judged and minutes > 0 else None,
        'itr_method': 'Wolpaw reference; grouped by keyboard size; below chance = 0',
        'active_seconds': seconds,
        'early_stops': sum(bool(r.get('stopped_early')) for r in rows),
        'mean_epoch_s': sum(r['duration_s'] for r in rows) / len(rows) if rows else None,
    }
