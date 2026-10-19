# TRACEABILITY — требования ТЗ и проверка прототипа

Статусы относятся только к прототипу. Эксперименты на симуляции не подтверждают требования к реальному ЭЭГ.

| ID | Модуль | Проверка | Статус |
|---|---|---|---|
| NEED-F-01 | `epoch/loop.py`, `text/primary.py` | `test_run_and_replay` | Реализовано в симуляции |
| NEED-F-02 | `physiology/simulated.py` | `test_two_paradigms` | Заглушка обеих парадигм |
| NEED-F-03 | `core/config.py`, `physiology/simulated.py`, `web/settings.py` | `test_timing_validation`, `test_settings_whitelist_and_validation` | Конфигурация симуляции; в браузере — выбор парадигмы и таймингов в окне экспериментатора |
| NEED-F-04 | `physiology/base.py`, `epoch/fusion.py` | `test_fusion` | Контракт реализован, обработка ЭЭГ вне объёма |
| NEED-F-05 | `decoding/detector.py`, `text/primary.py` | `test_run_and_replay` | Реализовано в симуляции |
| NEED-F-06 | `llm/fast.py`, `epoch/fusion.py` | `test_no_llm` | Реализовано с mock-моделью |
| NEED-F-07 | `llm/slow.py`, `text/secondary.py` | `test_correction_validation` | Реализовано с mock-моделью |
| NEED-F-08 | `ext/word_prediction.py` | `bench-llm` | Реализовано с mock-моделью |
| NEED-F-09 | `ext/ui_console.py`, `ext/commands.py`, `web/static/operator.*` | `test_run_and_replay`, `test_http_operator_flow` | Консоль; окно экспериментатора с распределениями и предсказаниями; команды — заглушка |
| NEED-F-10 | `ext/tts.py` | `test_tts_stub` | Заглушка, аудио не создаётся |
| NEED-F-11 | `core/config.py`, `web/settings.py` | `test_timing_validation`, `test_settings_whitelist_and_validation` | Реализовано; в браузере — настройки блоков на сессию и сохранение профиля `gui_*.yaml` |
| NEED-F-12 | `recording/recorder.py` | `test_run_and_replay` | Реализовано |
| NEED-F-13 | `recording/storage.py` | `test_run_and_replay` | События и метаданные реализованы; сырой ЭЭГ нет |
| NEED-F-14 | `recording/export.py` | `test_export_integrity` | Реализовано |
| REQ-PERF-01 | `physiology/simulated.py`, `evaluation/metrics.py` | Метрики симуляции | Для реального ЭЭГ не проверено |
| REQ-PERF-02 | `evaluation/metrics.py`, `evaluation/experiment.py` | Парный эксперимент | Для реального ЭЭГ не проверено |
| REQ-PERF-03 | `evaluation/metrics.py`, `evaluation/experiment.py` | Парный эксперимент | Для реального ЭЭГ не проверено |
| REQ-PERF-04 | `evaluation/stats.py`, `evaluation/experiment.py` | Парный бутстреп | Для реального ЭЭГ не проверено |
| REQ-LLM-01 | `evaluation/metrics.py` | Метрика восстановления слов | Mock baseline; порог не подтверждён |
| REQ-LLM-02 | `evaluation/metrics.py` | chrF-заглушка | NLI и порог вне объёма |
| REQ-LLM-03 | `evaluation/bench.py` | Top-3/5/10 на 40 заданиях | Mock baseline; пороги не подтверждены |
| REQ-AUDIO-01 | `ext/tts.py` | Отсутствие аудиособытия в заглушке | Не проверено |
| REQ-CONF-01 | `core/config.py`, `physiology/simulated.py` | `test_timing_validation` | Симуляция проверена, оборудование вне объёма |
