"""Local LLM correction backend against a fake OpenAI-compatible server."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

import pytest

from eeg_speller.llm.backends.local_openai import LocalOpenAICorrection, parse_words
from eeg_speller.llm.slow import create_slow, validate_correction

FIXES = {"мебм": "тебя", "нузно": "нужно"}


class FakeOllama(BaseHTTPRequestHandler):
    replies = []

    def log_message(self, *args):
        pass

    def _send(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._send({"data": [{"id": "qwen2.5:3b"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        words = body["messages"][-1]["content"].split(": ")[-1].split()
        answer = "Ответ: " + " ".join(FIXES.get(w, w) for w in words)
        FakeOllama.replies.append(answer)
        self._send({"choices": [{"message": {"content": answer}}]})


@pytest.fixture()
def server():
    httpd = HTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def cfg(url, model="qwen2.5:3b"):
    return {"backend": "local_openai", "model_id": model, "base_url": url,
            "temperature": 0, "timeout_s": 5}


def test_corrects_window_through_local_server(server):
    model = create_slow(cfg(server), False, None)
    out = model.correct("привет, ", ["как", "у", "мебм"])
    assert out == ["как", "у", "тебя"]
    assert validate_correction(["как", "у", "мебм"], out) == (True, None)


def test_rejects_non_local_address():
    with pytest.raises(ValueError, match="this computer"):
        LocalOpenAICorrection(cfg("https://api.example.com/v1"))


def test_clear_error_when_server_is_down():
    with pytest.raises(ValueError, match="no LLM server"):
        LocalOpenAICorrection(cfg("http://127.0.0.1:9/v1"))


def test_clear_error_when_model_missing(server):
    with pytest.raises(ValueError, match="not loaded"):
        LocalOpenAICorrection(cfg(server, model="nope"))


def test_parse_words_cleans_model_reply():
    assert parse_words("Ответ: Как у ТЕБЯ, ёжик.") == ["как", "у", "тебя", "ежик"]
    assert parse_words("Вот исправление:\nмне нужно") == ["мне", "нужно"]


def test_gui_session_uses_local_llm_for_secondary_text(server, tmp_path):
    from eeg_speller.web.engine import InteractiveSession
    target = "я мебм "
    session = InteractiveSession("gui_llm.yaml", target, tmp_path,
                                 overrides={"slow_llm.base_url": server},
                                 participant_id="P010", task_id="CUSTOM")
    for _ in range(40):
        if session.primary.value == target:
            break
        want = target[len(session.primary.value)]
        idx = session.rep.index(want)
        session.submit([idx in flash["indices"] for flash in session.plan])
    assert session.primary.value == target
    assert session.secondary.value == "я тебя "
    assert session.corrections[-1]["changed"] is True
