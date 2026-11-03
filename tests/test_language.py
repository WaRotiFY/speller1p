"""Session language (ru / en): LM alphabet, default models, tasks, prompt; slow LM every N characters."""
import numpy as np
import pytest

from eeg_speller.core import language
from eeg_speller.core.keyboard import keyboard_text, lm_char, project


@pytest.fixture(autouse=True)
def back_to_russian():
    yield
    language.set_language("ru")


def test_alphabet_follows_the_language():
    assert language.set_language("en").lm_alphabet == "abcdefghijklmnopqrstuvwxyz .,"
    assert lm_char("G") == "g" and lm_char("_") == " " and lm_char("ж") is None
    assert keyboard_text("Hi_there") == "hi there"
    p = project(np.full(29, 1 / 29), list("abc_"))
    assert np.isclose(p.sum(), 1) and np.allclose(p, 0.25)
    language.set_language("ru")
    assert lm_char("ж") == "ж" and lm_char("g") is None
    with pytest.raises(ValueError):
        language.set_language("de")


def test_keyboard_language_is_detected():
    assert language.detect(list("abcdefghijklmnopqrstuvwxyz123456789_")) == "en"
    assert language.detect(list("ЙЦУКЕНГШЩЗХ_")) == "ru"


def test_english_char_ngram_predicts_english():
    from eeg_speller.llm.backends.char_ngram import CharNGram, default_model
    language.set_language("en")
    assert default_model().name == "en_char4.npz"
    m = CharNGram({})
    p = m.lm_distribution("thank yo")                    # 4-gram: the context is "yo"
    assert language.lm_alphabet()[int(np.argmax(p))] == "u"
    assert m.top_words("thank yo", 3)[0][0] == "you"
    language.set_language("ru")
    with pytest.raises(ValueError, match="another language"):
        CharNGram({"model_file": "en_char4.npz"})


def test_tasks_and_prompt_per_language():
    from eeg_speller.external.tasks import read_task_list, resolve_task
    from eeg_speller.llm.backends import chat_api
    en = {t["id"] for t in read_task_list(language="en")}
    ru = {t["id"] for t in read_task_list(language="ru")}
    assert "E001" in en and "T001" in ru and "E001" not in ru
    assert {"BCI2B_TEST", "BNCI008_TEST"} <= en & ru                      # open-data phrases in both
    assert resolve_task({"kind": "task", "id": "BCI2B_TEST"}) == ("BCI2B_TEST", "foodmoothampiecaketunazygot4567")
    language.set_language("en")
    assert chat_api.read_prompt("default").startswith("You correct errors")
    msgs = chat_api.build_messages(chat_api.read_prompt("default"), "", ["hal"])
    assert "(start of text)" in msgs[1]["content"]


def test_server_builds_an_english_session():
    from eeg_speller.external.server import build_session_config
    cfg, passport = build_session_config({"language": "en", "fast": {"backend": "ngram"},
                                          "slow": {"backend": "dictionary", "every_n_chars": 15}})
    assert cfg.model_extra["language"] == "en" and cfg.fast_llm["model_file"] == "en_char4.npz"
    assert cfg.slow_llm["trigger"] == "every_n_chars" and cfg.slow_llm["every_n_chars"] == 15
    assert passport["language"] == "English"
    cfg, _ = build_session_config({"language": "en", "fast": {"backend": "kenlm"}})
    assert cfg.fast_llm["model_id"] == "jonatasgrosman/wav2vec2-large-xlsr-53-english"


def test_slow_llm_every_n_characters():
    from eeg_speller.llm.slow import CorrectionScheduler

    def requests(text, n):
        s = CorrectionScheduler({"trigger": "every_n_chars", "every_n_chars": n})
        return [(i, r.words, r.start_word) for i in range(1, len(text) + 1) if (r := s.on_primary_update(text[:i]))]

    # every word goes out once; the word being typed waits for the next request
    assert requests("i would like a glass of water please call", 10) == [
        (10, ["i", "would"], 0), (20, ["like", "a"], 2), (30, ["glass", "of", "water"], 4), (40, ["please"], 7)]
    # text without spaces is still corrected: the whole run of letters
    assert requests("gattomenteviolarebus", 20) == [(20, ["gattomenteviolarebus"], 0)]


def test_kenlm_without_internet_gives_a_clear_error_and_takes_files_put_by_hand(tmp_path, monkeypatch):
    from eeg_speller.llm.backends import kenlm_words as k
    monkeypatch.setattr(k, "CACHE", tmp_path)
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")            # nothing listens there
    with pytest.raises(ValueError, match="не удалось скачать.*4-грамму"):
        k.fetch_hf_repo("owner/name", True)
    folder = tmp_path / "owner__name"
    folder.mkdir(exist_ok=True)
    (folder / "lm.binary").write_bytes(b"kenlm")
    assert k.fetch_hf_repo("owner/name", False) == folder               # no download needed
