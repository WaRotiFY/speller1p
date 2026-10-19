"""Deterministic row/column flash plans for the synthetic EDF recordings."""
import random


def make_plan(cfg, epoch_id: int) -> list[dict]:
    rng = random.Random(cfg.child_seed("flash_plan") + epoch_id)
    rows, cols = cfg.repertoire["layout"]
    n = len(cfg.repertoire["symbols"])
    if cfg.stimulus["paradigm"] == "row_column":
        groups = [{"kind": "row", "index": r, "indices": list(range(r * cols, (r + 1) * cols))}
                  for r in range(rows)]
        groups += [{"kind": "column", "index": c, "indices": list(range(c, n, cols))}
                   for c in range(cols)]
    else:
        groups = [{"kind": "symbol", "index": i, "indices": [i]} for i in range(n)]
    plan = []
    step = cfg.stimulus["flash_ms"] + cfg.stimulus["isi_ms"]
    for rep in range(cfg.epoch["n_repetitions"]):
        order = groups[:]
        rng.shuffle(order)
        for group in order:
            plan.append({"id": len(plan), "repetition": rep, "onset_ms": len(plan) * step, **group})
    return plan
