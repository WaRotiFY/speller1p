"""Operator v2: keyboard from the module, n-gram, inference plugins, slow LLM over a chat API, recording."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import numpy as np
import pytest

from eeg_speller.core.config import load_config
from eeg_speller.core.distributions import Repertoire
from eeg_speller.core.keyboard import PRESETS, keyboard_text, lm_char, project, symbol_for_char, typed_char, validate_keyboard
from eeg_speller.epoch.plugins import PluginFusion, list_plugins
from eeg_speller.external.controller import ExternalSession
from eeg_speller.external.launcher import build_args, stimulus_programs
from eeg_speller.external.server import build_session_config, init_payload
from eeg_speller.external.transport import MemoryTransport
from eeg_speller.llm.backends import chat_api
from eeg_speller.llm.backends.char_ngram import CharNGram
from eeg_speller.llm.slow import CorrectionScheduler
from eeg_speller.physiology.base import PhysiologyObservation
from eeg_speller.core.distributions import SymbolDistribution, log_probs
from eeg_speller.recording.storage import read_events

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ keyboard
def test_keyboard_mapping_and_projection():
    assert lm_char("Ё") == "е" and lm_char("_") == " " and lm_char("7") is None and lm_char("⌫") is None
    assert typed_char("Ж") == "ж" and typed_char("7") == "7"
    assert keyboard_text("ПРИВЕТ_МИР") == "привет мир"
    sym = PRESETS["jcuken_4x11_digits"]["symbols"]
    lm = np.full(35, 1 / 35)
    p = project(lm, sym, backspace_prob=0.02, other_prob=0.01)
    assert p.sum() == pytest.approx(1.0)
    assert p[sym.index("1")] == pytest.approx(0.01)                  # digits: fixed share
    assert p[sym.index("Е")] == pytest.approx(p[sym.index("Ё")])     # е/ё split the mass of «е»
    assert symbol_for_char("р", sym) == "Р"


def test_module_keyboards_are_checked():
    validate_keyboard(PRESETS["jcuken_4x11_digits"]["symbols"], [4, 11])
    with pytest.raises(ValueError):
        validate_keyboard(list("абв"), [1, 2])          # does not fit the layout
    with pytest.raises(ValueError):
        validate_keyboard(list("аав"), [1, 3])          # repeated key


# ------------------------------------------------------------------ fast LLM
def test_default_ngram_predicts_russian():
    model = CharNGram({"model_file": "ru_char4.npz", "char_marginal": {"backspace_prob": 0.02}})
    rep = Repertoire(tuple(PRESETS["abc_6x6"]["symbols"]))
    p = np.exp(model.predict("спасиб", rep).log_p)
    assert rep.symbols[int(p.argmax())] == "о"
    assert model.meta["order"] == 4
    words = [w for w, _ in model.top_words("я хоч", 5)]
    assert any(w.startswith("хоч") for w in words)
    upper = Repertoire(tuple(PRESETS["jcuken_4x11_digits"]["symbols"]), (4, 11))
    q = np.exp(model.predict("СПАСИБ", upper).log_p)
    assert upper.symbols[int(q.argmax())] == "О"


# ------------------------------------------------------------- inference plugins
def test_inference_plugins_listed_and_fallback_on_error(monkeypatch):
    ids = {p["id"] for p in list_plugins()}
    assert {"log_pool", "linear_pool", "tempered_bayes"} <= ids
    rep = Repertoire(tuple(PRESETS["abc_6x6"]["symbols"]))
    prior = SymbolDistribution(rep, log_probs(np.r_[[10.0], np.ones(35)]), "test", 0)
    obs = PhysiologyObservation(0, log_probs(np.ones(36)), "likelihood")
    f = PluginFusion({"plugin": "log_pool", "params": {"alpha": 1.0}}, "on")
    assert f.combine(prior, obs).log_p.argmax() == 0
    off = PluginFusion({"plugin": "log_pool"}, "off")
    assert np.allclose(np.exp(off.combine(prior, obs).log_p), 1 / 36)        # no LLM: physiology only
    monkeypatch.setattr(f.plugin.module, "combine", lambda a, b, p: np.array([np.nan]))
    post = f.combine(prior, obs)
    assert post.fusion.endswith(":fallback") and f.last_error


# ------------------------------------------------------------------ slow LLM
def test_scheduler_every_n_words():
    s = CorrectionScheduler({"trigger": "word_end", "every_n_words": 2, "window_words": 3, "min_context_words": 0})
    fired = [s.on_primary_update(t) is not None for t in ("раз ", "раз два ", "раз два три ", "раз два три четыре ")]
    assert fired == [False, True, False, True]


class FakeChat(BaseHTTPRequestHandler):
    calls: list = []

    def do_GET(self):
        body = json.dumps({"data": [{"id": "a:free"}, {"id": "b", "pricing": {"prompt": "1", "completion": "1"}}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeChat.calls.append(req)
        words = req["messages"][1]["content"].split("Слова (")[1].split("): ")[1].split("\n")[0].split()
        body = json.dumps({"choices": [{"message": {"content": "Ответ: " + " ".join(w.replace("пдед", "плед") for w in words)}}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_api():
    FakeChat.calls = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeChat)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_chat_api_prompt_alternatives_and_remote_guard(fake_api, monkeypatch, tmp_path):
    monkeypatch.setattr(chat_api, "KEYS_FILE", tmp_path / "keys.json")
    model = chat_api.ChatCorrection({"provider": "custom", "base_url": fake_api, "model_id": "m"}, allow_remote=False)
    out = model.correct("мне холодно,", ["дай", "пдед"], [[[["д", .9]], [["а", .9]], [["й", .9]]],
                                                        [[["п", .9]], [["д", .46], ["л", .41]], [["е", .9]], [["д", .9]]]])
    assert out == ["дай", "плед"]
    msg = FakeChat.calls[-1]["messages"][1]["content"]
    assert "пдед: п[п 90] д[д 46 л 41]" in msg and "Контекст: мне холодно," in msg
    assert chat_api.list_models("custom", fake_api) == ["a:free", "b"]
    with pytest.raises(PermissionError):
        chat_api.ChatCorrection({"provider": "openrouter", "model_id": "x:free"}, allow_remote=False)
    with pytest.raises(ValueError):
        chat_api.base_url_for("custom", "http://example.com/v1")                    # http only for loopback
    with pytest.raises(ValueError):
        chat_api.check_template("без разделителя {context} {words}")
    cfg, _ = build_session_config({"slow": {"backend": "chat_api", "provider": "groq", "model_id": "x"}})
    assert cfg.llm["allow_remote"]                  # choosing a cloud provider is the consent
    cfg, _ = build_session_config({"slow": {"backend": "mock_dictionary"}})
    assert not cfg.llm["allow_remote"]


def test_keys_are_stored_locally_only(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_api, "KEYS_FILE", tmp_path / "keys.json")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    chat_api.save_key("groq", "secret-123")
    assert chat_api.get_key("groq") == "secret-123"
    assert any(p["id"] == "groq" and p["has_key"] for p in chat_api.provider_list())
    assert "secret" not in json.dumps(init_payload(), ensure_ascii=False)


# ------------------------------------------------------------------ modules
def test_modules_carry_their_own_keyboard_and_need_no_arguments():
    progs = {p["id"]: p for p in stimulus_programs()}
    for pid in ("modules/flash_rows_cols", "modules/moving_letters", "modules/moving_circles", "modules/simulator"):
        info = progs[pid]
        cfg = info["settings"]
        assert len(cfg["alphabet"]) == cfg["num_rows"] * cfg["num_cols"], pid
        validate_keyboard(list(cfg["alphabet"]), [cfg["num_rows"], cfg["num_cols"]])
        assert build_args(info, "привет") == []          # the operator only picks the module


def test_module_tools_keyboard_and_cue():
    import sys
    sys.path.insert(0, str(ROOT / "stimulus_kit"))
    from module_tools import cue_for, keyboard_of
    cfg = {"alphabet": "абв_", "num_rows": 2, "num_cols": 2, "sentence": "Ба в"}
    symbols, layout = keyboard_of(cfg)
    assert symbols == ["а", "б", "в", "_"] and layout == (2, 2)
    assert cue_for(cfg, "", symbols) == "б" and cue_for(cfg, "ба", symbols) == "_"
    assert cue_for(cfg, "ба в", symbols) is None and cue_for({**cfg, "sentence": ""}, "", symbols) is None


# ---------------------------------------------------------- session recording
def test_session_with_api_correction_records_raw_likelihood_and_timeline(fake_api, tmp_path, monkeypatch):
    monkeypatch.setattr(chat_api, "KEYS_FILE", tmp_path / "keys.json")
    cfg, passport = build_session_config({
        "inference": {"plugin": "log_pool", "params": {"alpha": 0.0}},
        "slow": {"backend": "chat_api", "provider": "custom", "base_url": fake_api, "model_id": "m",
                 "every_n_words": 1, "window_words": 2}})
    cfg = cfg.model_copy(update={"slow_llm": {**cfg.slow_llm, "min_context_words": 0}})
    tr = MemoryTransport()
    plan = list("дай пдед ")
    keyboards = []
    kb = PRESETS["jcuken_4x10"]
    # The module announces its keyboard as soon as it connects (speller-link@1, "keyboard").
    tr.deliver("stimuli", {"event": "keyboard", "symbols": kb["symbols"], "layout": list(kb["layout"])})

    def module(msg):
        if msg["cmd"] != "epoch_start":
            return
        keyboards.append(tuple(msg["layout"]))
        k = sum(1 for m in tr.sent if m["cmd"] == "result" and m.get("symbol"))
        sym = next(s for s in msg["symbols"] if typed_char(s) == plan[min(k, len(plan) - 1)])
        p = {s: (0.9 if s == sym else 0.1 / (len(msg["symbols"]) - 1)) for s in msg["symbols"]}
        tr.deliver("stimuli", {"epoch": msg["epoch"], "flash": [sym]})
        tr.deliver("likelihood", {"epoch": msg["epoch"], "p": p, "final": True})

    tr.listeners.append(module)
    s = ExternalSession(cfg, "дай плед ", tmp_path, transport=tr, passport=passport).start()
    assert s.wait_finished(60)
    state = s.monitor()
    assert state["text"] == "дай пдед " and state["corrected"] == "дай плед "
    assert set(keyboards) == {tuple(kb["layout"])}
    assert state["keyboard"]["symbols"] == list(kb["symbols"])
    assert any(c["changed"] for c in state["corrections"])
    events = read_events(s.storage.directory)
    kb_event = next(e for e in events if e["type"] == "keyboard_set")["payload"]
    assert kb_event["layout"] == list(kb["layout"]) and not kb_event["task_characters_missing"]
    obs = next(e for e in events if e["type"] == "physiology_observation")
    assert "lsl_s" in obs and isinstance(obs["payload"]["raw"], dict)
    assert obs["payload"]["recv_lsl_s"] is not None and obs["payload"]["sent_lsl_s"] is not None
    tl = next(e for e in events if e["type"] == "epoch_timeline")["payload"]
    assert {"decided_ms", "result_sent_ms", "first_flash_ms"} <= set(tl["since_epoch_start_ms"])
    req = [e for e in events if e["type"] == "correction_requested"]
    assert req and req[-1]["payload"]["alternatives"]
    done = [e for e in events if e["type"] == "correction_completed"]
    assert done[-1]["payload"]["messages"][1]["role"] == "user"
    assert (s.storage.directory / "external" / "slow_llm_prompt.txt").is_file()
    integrity = json.loads((s.storage.directory / "derived" / "integrity.json").read_text())
    assert integrity["complete"], integrity


# ------------------------------------------------ HF adapter logic without torch
class _Arr:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    def __getitem__(self, key):
        return _Arr(self.a[key])

    def float(self):
        return self

    def __iter__(self):
        return (_Arr(row) for row in self.a)

    @property
    def shape(self):
        return self.a.shape

    def tolist(self):
        return self.a.tolist()


class _FakeTorch:
    class no_grad:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    @staticmethod
    def tensor(x, device=None):
        return np.asarray(x)

    @staticmethod
    def softmax(x, dim=-1):
        z = np.exp(x.a - x.a.max(axis=-1, keepdims=True))
        return _Arr(z / z.sum(axis=-1, keepdims=True))

    @staticmethod
    def topk(row, k):
        idx = np.argsort(-row.a)[:k]

        class R:
            indices = _Arr(idx)
            values = _Arr(row.a[idx])
        R.indices.tolist = lambda: [int(i) for i in idx]
        return R


def test_hf_adapter_heals_tokens_and_marginalizes(monkeypatch):
    """A character-level fake model driven by the 4-gram: the HF path must agree with it."""
    from eeg_speller.core.keyboard import LM_ALPHABET
    from eeg_speller.llm.backends.hf_causal import HFCausal
    ngram = CharNGram({"model_file": "ru_char4.npz"})
    vocab = list(LM_ALPHABET)

    class Tok:
        eos_token_id = 0

        def __call__(self, text, add_special_tokens=False):
            class R:
                input_ids = [vocab.index(c) for c in text if c in vocab]
            return R

        def decode(self, ids, **kw):
            return "".join(vocab[i] for i in ids)

    class Model:
        def __call__(self, ids):
            rows = [np.log(ngram.lm_distribution("".join(vocab[i] for i in seq)) + 1e-12) for seq in ids]

            class Out:
                logits = _Arr(np.asarray(rows)[:, None, :])
            return Out

    hf = object.__new__(HFCausal)
    hf.torch, hf.tokenizer, hf.model, hf.device = _FakeTorch, Tok(), Model(), "cpu"
    hf.cfg = {"char_marginal": {"beam": 8, "depth": 2, "backspace_prob": 0.02}}
    hf.model_id = "fake/char-lm"
    rep = Repertoire(tuple(PRESETS["abc_6x6"]["symbols"]))
    p = np.exp(hf.predict("спасиб", rep).log_p)
    assert rep.symbols[int(p.argmax())] == "о"
    q = np.exp(ngram.predict("спасиб", rep).log_p)
    assert np.argmax(p) == np.argmax(q)


# ------------------------------------------------------------- word KenLM model
def test_word_kenlm_gives_letters_and_words(tmp_path):
    pytest.importorskip("kenlm")
    import subprocess
    import sys
    from eeg_speller.llm.backends.kenlm_words import KenLMWords, default_char_model
    from eeg_speller.core.keyboard import LM_ALPHABET
    corpus = tmp_path / "c.txt"
    corpus.write_text("мне нужно выпить воды. мне нужно выпить чаю. мне нужно поспать. я хочу пить. я хочу спать.\n" * 5,
                      encoding="utf-8")
    arpa = tmp_path / "m.arpa"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_arpa.py"), str(corpus), str(arpa)], check=True)
    model = KenLMWords({"model_id": str(arpa), "char_mix": 0.1}, default_char_model())
    p = model.lm_distribution("мне нужно выпить в")
    assert LM_ALPHABET[int(np.argmax(p))] == "о"                 # «воды» is the only word starting with «в»
    q = model.lm_distribution("я хочу п")
    assert LM_ALPHABET[int(np.argmax(q))] == "и"
    end = model.lm_distribution("я хочу пить")
    assert end[LM_ALPHABET.index(".")] + end[LM_ALPHABET.index(" ")] > 0.5   # word finished: boundary
    words = [w for w, _ in model.top_words("мне нужно выпить ", 3)]
    assert set(words) >= {"воды", "чаю"}
    odd = model.lm_distribution("я хочу пкщ")                    # nothing in the vocabulary: char 4-gram
    assert odd.sum() == pytest.approx(1.0) and np.all(odd > 0)
    rep = Repertoire(tuple(PRESETS["abc_6x6"]["symbols"]))
    assert model.predict("мне нужно выпить в", rep).log_p.shape == (36,)


# ---------------------------------------------------------------- replay
def test_replay_of_operator_session_reuses_recorded_evidence(tmp_path):
    from eeg_speller.recording.replay_external import is_external, replay_external
    cfg, passport = build_session_config({"slow": {"backend": "off"}})
    cfg = cfg.model_copy(update={"run_mode": "live"})
    kb = PRESETS["jcuken_4x10"]
    tr = MemoryTransport()
    tr.deliver("stimuli", {"event": "keyboard", "symbols": kb["symbols"], "layout": list(kb["layout"])})
    target = "дом мир"

    def module(msg):
        if msg["cmd"] != "epoch_start":
            return
        k = sum(1 for m in tr.sent if m["cmd"] == "result" and m.get("symbol"))
        sym = next(s for s in msg["symbols"] if typed_char(s) == target[min(k, len(target) - 1)])
        for i, group in enumerate([[sym], [s for s in msg["symbols"] if s != sym][:5]]):
            tr.deliver("stimuli", {"epoch": msg["epoch"], "flash": group})
        rest = (1 - 0.6) / (len(msg["symbols"]) - 1)
        p = {s: (0.6 if s == sym else rest) for s in msg["symbols"]}
        tr.deliver("likelihood", {"epoch": msg["epoch"], "p": p, "final": False})
        tr.deliver("likelihood", {"epoch": msg["epoch"], "p": p, "final": True})

    tr.listeners.append(module)
    live = ExternalSession(cfg, target, tmp_path / "live", transport=tr, passport=passport).start()
    assert live.wait_finished(60) and live.monitor()["text"] == target
    src = live.storage.directory
    assert is_external(src)

    same = replay_external(src, tmp_path / "replay")
    assert same["error"] is None and same["replay"]["text"] == target
    assert same["replay"]["epochs"] == same["original"]["epochs"] == len(target)
    manifest = json.loads((Path(same["session"]) / "session.json").read_text(encoding="utf-8"))
    assert manifest["replay_of"] == src.name and manifest["run_mode"] == "replay"
    assert manifest["repertoire"] == list(kb["symbols"])          # the module's keyboard, not the profile's
    off = replay_external(src, tmp_path / "replay", llm="off")
    assert off["error"] is None and off["replay"]["epochs"] == len(target)
