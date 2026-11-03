"""Do the language models help on real EEG? One command, a table at the end.

Every phrase of a task list is spelled by the real speller (separate process, LSL) with
the module "Реальные ответы ЭЭГ на фразу задания" (stimulus_kit/modules/eeg_phrase):
real single-flash decoder scores from an open recording. The same phrases are spelled
with several language-model settings; the EEG for every letter is the same in all of
them (same seed), so the differences are the models' doing.

    uv run --no-sync python scripts/eval_lm_real_eeg.py --language en --tasks 10
    uv run --no-sync python scripts/eval_lm_real_eeg.py --language ru --tasks 10 --source bnci2014_008_A01
    uv run --no-sync python scripts/eval_lm_real_eeg.py --language en --configs none ngram ngram+dict \
        ngram+api --provider ollama --model qwen2.5:7b

Settings (``--configs``):
    none         no language models (uniform prior, no correction)
    ngram        fast LM: character 4-gram of the language
    kenlm        fast LM: word KenLM of the language (needs the kenlm package and the model)
    ngram+dict   4-gram + slow correction "dictionary" every N characters
    ngram+api    4-gram + slow correction by a chat model (--provider, --model)

Columns: accuracy of the chosen symbols; CER of the typed text and of the text after
slow correction (edit distance / length of the task); flashes per symbol; speed in
correct characters per minute of real stimulation time (175 ms per flash + the pause
between symbols), not the wall time of this accelerated run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_speller.core.language import LANGUAGES  # noqa: E402
from eeg_speller.external.tasks import read_task_list  # noqa: E402

LABELS = {"none": "без БЯМ", "ngram": "быстрая: 4-грамма", "kenlm": "быстрая: KenLM",
          "ngram+dict": "4-грамма + словарь", "ngram+api": "4-грамма + БЯМ по API"}


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def profile(name, lang, args) -> dict:
    p = {"inherits": "ext_default.yaml", "language": lang.code, "participant_id": "EVAL",
         "external": {"stop_confidence": args.stop, "min_updates_before_stop": 12},
         "stimulus": {"paradigm": "row_column", "flash_ms": 100, "isi_ms": 70, "pause_ms": 500}}
    fast = {"backend": "ngram", "model_file": lang.ngram_file, "model_id": Path(lang.ngram_file).stem}
    if name == "none":
        p.update(mode={"llm": "off"}, fast_llm={"backend": "uniform", "model_id": "uniform"},
                 slow_llm={"backend": "off"})
    elif name == "ngram":
        p.update(fast_llm=fast, slow_llm={"backend": "off"})
    elif name == "kenlm":
        p.update(fast_llm={"backend": "kenlm", "model_id": lang.kenlm_repo, "char_mix": 0.1,
                           "vocab_limit": 60000, "allow_download": False}, slow_llm={"backend": "off"})
    elif name == "ngram+dict":
        p.update(fast_llm=fast, slow_llm={"backend": "dictionary", "model_id": "dictionary-letters-1",
                                          "include_alternatives": True, "trigger": "every_n_chars",
                                          "every_n_chars": args.every})
    elif name == "ngram+api":
        from eeg_speller.llm.backends import chat_api
        p.update(fast_llm=fast, slow_llm={
            "backend": "chat_api", "provider": args.provider, "model_id": args.model,
            "base_url": chat_api.base_url_for(args.provider, args.base_url), "include_alternatives": True,
            "trigger": "every_n_chars", "every_n_chars": args.every, "timeout_s": 60,
            "prompt_text": chat_api.read_prompt()})
        p["llm"] = {"allow_remote": args.provider not in ("ollama", "custom")}
    else:
        raise SystemExit(f"unknown setting {name!r}; known: {', '.join(LABELS)}")
    return p


def run_one(phrase, prof_path, module_settings, out_dir, timeout):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    core = subprocess.Popen([sys.executable, "-m", "eeg_speller.external", "--profile", str(prof_path),
                             "--target", phrase, "--output", str(out_dir)], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
    time.sleep(1.5)
    mod = subprocess.Popen([sys.executable, str(ROOT / "stimulus_kit/modules/eeg_phrase/main.py"), "--text", phrase,
                            "--settings", str(module_settings)], cwd=out_dir, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    try:
        out, _ = core.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        core.kill()
        out = ""
    try:
        mod_out, _ = mod.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        mod.kill()
        mod_out = ""
    m = re.search(r"\n\{.*\}\s*$", out, re.S)
    result = json.loads(m.group()) if m else {"text": "", "corrected": "", "stats": {}, "error": "no result"}
    result["flashes"] = [int(x) for x in re.findall(r"вспышек (\d+)", mod_out)]
    if not m:
        result["log"] = mod_out[-800:]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", default="en", choices=sorted(LANGUAGES))
    ap.add_argument("--tasks", type=int, default=10, help="how many phrases of the language's task list")
    ap.add_argument("--configs", nargs="+", default=["none", "ngram", "ngram+dict"])
    ap.add_argument("--source", default="bci2_iib", help="bank of EEG responses: bci2_iib or bnci2014_008_A01")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--every", type=int, default=10, help="slow correction every N characters")
    ap.add_argument("--stop", type=float, default=0.99, help="speller stop confidence")
    ap.add_argument("--speed", type=float, default=40.0, help="how much faster than real time to run")
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--base-url")
    ap.add_argument("--out", default=str(ROOT / "runs" / "eval_lm_real_eeg"))
    a = ap.parse_args()

    lang = LANGUAGES[a.language]
    tasks = [t for t in read_task_list(language=a.language) if not t["id"].startswith(("BNCI", "BCI2"))][:a.tasks]
    out_root = Path(a.out) / time.strftime("%Y%m%d_%H%M%S")
    work = Path(tempfile.mkdtemp())
    for f in (ROOT / "configs").glob("*.yaml"):
        shutil.copy(f, work / f.name)
    import yaml
    rows, details = [], []
    soa = 0.175
    for name in a.configs:
        prof = work / f"eval_{name.replace('+', '_')}.yaml"
        prof.write_text(yaml.safe_dump(profile(name, lang, a), allow_unicode=True), encoding="utf-8")
        pause_s = 1.0
        agg = {"symbols": 0, "right": 0, "err_typed": 0, "err_corr": 0, "len": 0, "flashes": 0, "epochs": 0, "fail": 0}
        for seed in a.seeds:
            settings = json.loads((ROOT / "stimulus_kit/modules/eeg_phrase/settings.json").read_text(encoding="utf-8"))
            settings.update(source=a.source, speed=a.speed, seed=seed, show_window=False)
            ms = work / f"module_{seed}.json"
            ms.write_text(json.dumps(settings), encoding="utf-8")
            for t in tasks:
                phrase = t["text"]
                r = run_one(phrase, prof, ms, out_root / name / f"seed{seed}" / t["id"], timeout=600)
                typed, corr = r.get("text", ""), r.get("corrected", "") or r.get("text", "")
                st = r.get("stats") or {}
                if r.get("error") or not typed:
                    agg["fail"] += 1
                    print(f"  {name} {t['id']}: no result {r.get('error')} {r.get('log', '')[-300:]}", flush=True)
                    continue
                agg["symbols"] += len(r["flashes"])
                agg["right"] += round((st.get("accuracy") or 0) * len(r["flashes"]))
                agg["err_typed"] += edit_distance(typed.rstrip(), phrase.rstrip())
                agg["err_corr"] += edit_distance(corr.rstrip(), phrase.rstrip())
                agg["len"] += len(phrase.rstrip())
                agg["flashes"] += sum(r["flashes"])
                agg["epochs"] += len(r["flashes"])
                details.append({"config": name, "seed": seed, "task": t["id"], "target": phrase,
                                "typed": typed, "corrected": corr})
                print(f"  {LABELS[name]:<24} {t['id']}  {typed!r}  →  {corr!r}", flush=True)
        minutes = (agg["flashes"] * soa + agg["epochs"] * pause_s) / 60
        right_chars = agg["len"] - agg["err_corr"]
        rows.append((LABELS[name], agg["right"] / max(agg["symbols"], 1), agg["err_typed"] / max(agg["len"], 1),
                     agg["err_corr"] / max(agg["len"], 1), agg["flashes"] / max(agg["epochs"], 1),
                     right_chars / max(minutes, 1e-9), agg["fail"]))
    shutil.rmtree(work, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nязык {lang.name}, фраз {len(tasks)} × зёрен {len(a.seeds)}, ответы ЭЭГ: {a.source}, "
          f"остановка {a.stop}, медленная БЯМ каждые {a.every} симв.")
    print(f"{'настройка':<26}{'точн. символа':>14}{'CER набрано':>13}{'CER исправл.':>14}{'вспышек/симв':>14}"
          f"{'верн. симв/мин':>16}")
    for label, acc, cer_t, cer_c, fl, cpm, fail in rows:
        print(f"{label:<26}{acc:>14.3f}{cer_t:>13.3f}{cer_c:>14.3f}{fl:>14.1f}{cpm:>16.2f}"
              + (f"   (без результата: {fail})" if fail else ""))
    print(f"подробно: {out_root / 'details.json'}")


if __name__ == "__main__":
    main()
