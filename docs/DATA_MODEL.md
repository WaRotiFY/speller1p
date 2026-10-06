# DATA_MODEL — структура данных Session Recorder и Storage

Версия схемы данных: `eeg-speller/events@1`. Документ от 29.09.2026. Архитектура: v3.
Связанные документы: `ARCHITECTURE.md`, `ASSUMPTIONS.md`, `OPEN_QUESTIONS.md`.
Связанные требования: NEED-F-11, NEED-F-12, NEED-F-13, NEED-F-14, NEED-M-01.

## 1. Принципы

1. **Одна сессия — один каталог.** Всё, что нужно для анализа и воспроизведения, лежит в каталоге сессии; внешние данные (сырая ЭЭГ реального модуля) — по ссылке с контрольной суммой.
2. **Журнал событий — первичный источник.** Все блоки публикуют события в шину; Recorder дописывает их в `events.jsonl` в порядке поступления. Файл только дополняется, никогда не переписывается. Таблицы, тексты и метрики — производные, их всегда можно пересобрать из журнала.
3. **Единые часы и сквозная нумерация.** Каждое событие имеет `seq` (0, 1, 2, … без пропусков) и `t_ns` по часам системы (MA-07, раздел 7 ARCHITECTURE). Пропуск номера означает потерю события и обнаруживается при экспорте (Q-30).
4. **Распределения пишутся полностью.** Все векторы по репертуару (prior, наблюдение физиологии, posterior) сохраняются целиком как логарифмы в порядке репертуара MA-01. Это делает возможным replay и любые офлайн-расчёты без повторного запуска моделей.
5. **Самоописание.** Снимок конфигурации, версии кода и моделей, сид и репертуар пишутся в каждую сессию.
6. **Псевдонимизация.** Реальные имена участников в данных не хранятся, только `participant_id`.

## 2. Идентификаторы

| Идентификатор | Формат | Пример | Где задаётся |
|---|---|---|---|
| `participant_id` | `P` + 3 цифры; для симуляции `SIM-` + 3 цифры | `P007`, `SIM-003` | конфиг профиля |
| `session_id` | `<UTC-время старта>_<participant_id>_<режим>_<4 hex>` | `20260929T193015Z_SIM-003_simulate_a1f0` | при старте сессии |
| `experiment_id` | свободная строка, объединяет серию сессий | `EXP-04_alpha_sweep` | конфиг эксперимента |
| `task_id` | ID задания в `data/tasks_ru.txt` | `T012` | при выдаче задания |
| `epoch_id` | целое с 0, сквозное в сессии; повтор эпохи при отказе получает новый ID | `57` | EpochRunner |
| `position` | индекс символа в Primary Text, к которому относится эпоха | `41` | EpochRunner |
| `correction_id` | целое с 0, сквозное в сессии | `9` | CorrectionScheduler |
| `seq` | целое с 0, сквозное в сессии, без пропусков | `1532` | Recorder |

## 3. Структура каталога сессии

```
runs/
  <experiment_id>/                     # необязательный уровень для серий
    <session_id>/
      session.json                     # манифест сессии (раздел 4)
      config.yaml                      # полный снимок конфигурации
      events.jsonl                     # журнал событий (разделы 5–6)
      texts/
        target.txt                     # целевой текст задания
        primary_final.txt              # итоговый Primary Text
        secondary_final.txt            # итоговый Secondary Text
      tables/                          # производные таблицы (пересобираются из events.jsonl)
        epochs.parquet | epochs.csv    # одна строка на эпоху (раздел 8)
        corrections.parquet | .csv     # одна строка на вызов медленной БЯМ
      derived/
        metrics.json                   # метрики сессии (evaluation, MA-23…MA-32)
      external/
        refs.json                      # ссылки на внешние данные (сырая ЭЭГ, записи физиологии) + sha256
      checksums.sha256                 # контрольные суммы всех файлов каталога
```

## 4. Манифест `session.json`

| Поле | Тип | Описание |
|---|---|---|
| `schema_version` | str | `eeg-speller/events@1` |
| `session_id`, `participant_id`, `experiment_id`, `task_id` | str | идентификаторы (раздел 2) |
| `started_utc`, `ended_utc` | str (ISO 8601) | время начала и конца по UTC |
| `clock` | object | `{"kind": "virtual" \| "monotonic", "t0_ns": int}` — привязка `t_ns` к UTC |
| `run_mode` | str | `simulate` \| `replay` \| `semi_synthetic` \| `live` |
| `llm_mode` | str | `on` \| `off` (MA-13) |
| `paradigm` | str | `row_column` \| `single_symbol` (Q-06) |
| `repertoire` | list[str] | 36 символов в порядке векторов (MA-01) |
| `seed` | int | сид генератора случайных чисел |
| `code_version` | str | git-хеш или версия пакета |
| `models` | object | `{"fast_llm": {...}, "slow_llm": {...}, "physiology": {...}}` — бэкенд, `model_id`, версия, хеш весов или промпта |
| `hardware` | object | CPU, GPU, ОС, аудиоустройство (для REQ-AUDIO-01, Q-28) |
| `replay_of` | str \| null | `session_id` исходной сессии для режима replay |
| `counts` | object | число эпох, событий, исправлений; заполняется при закрытии |
| `status` | str | `completed` \| `aborted` \| `crashed` |

## 5. Общая оболочка события

Каждая строка `events.jsonl` — один JSON-объект:

```json
{
  "schema": "eeg-speller/events@1",
  "session_id": "20260929T193015Z_SIM-003_simulate_a1f0",
  "participant_id": "SIM-003",
  "seq": 1532,
  "t_ns": 734512000000,
  "block": "Statistical Inference",
  "type": "posterior_computed",
  "epoch_id": 57,
  "payload": { }
}
```

| Поле | Тип | Описание |
|---|---|---|
| `schema` | str | версия схемы |
| `session_id`, `participant_id` | str | дублируются в каждом событии, чтобы журналы разных сессий можно было объединять |
| `seq` | int | сквозной номер без пропусков |
| `t_ns` | int | время события по часам системы, нс от `clock.t0_ns` |
| `block` | str | имя блока по схеме v3 (или `ext:<имя>` для точек расширения) |
| `type` | str | тип события (раздел 6) |
| `epoch_id` | int \| null | эпоха, к которой относится событие |
| `payload` | object | данные события |

Соглашения: векторы распределений — списки из 36 чисел `float` (натуральный логарифм, 6 значащих цифр); длительности вычислений — `wall_latency_ms` (реальные часы, пишутся и в виртуальном режиме); поля, известные только в симуляции (цель эпохи, оценки вспышек), помечены префиксом `sim_`.

## 6. Каталог событий

### 6.1 Сессия и задание

| Тип | Блок | Payload |
|---|---|---|
| `session_started` | Session Recorder | `run_mode`, `llm_mode`, `paradigm`, `seed`, `config_hash` |
| `task_assigned` | Session Recorder | `task_id`, `target_text` |
| `session_ended` | Session Recorder | `status`, `reason`, `n_epochs`, `n_events` |
| `session_paused` | Session Recorder | `next_epoch_id` — пауза, поставленная экспериментатором в браузерном режиме; эпоха после паузы начинается заново |
| `session_resumed` | Session Recorder | `next_epoch_id` — конец паузы; время паузы не входит в оперативную скорость окна экспериментатора |
| `warning`, `error` | любой | `module`, `message`, `details` |

### 6.2 Эпоха и физиология (заглушка)

| Тип | Блок | Payload |
|---|---|---|
| `epoch_started` | One Epoch | `position`, `repertoire_hash`, `paradigm`, `flash_ms`, `isi_ms`, `pause_ms`, `n_repetitions`, `retry_idx`, `sim_target_symbol` |
| `stimulus_event` | Presentation Method | `rep`, `flash_idx`, `group` (индексы символов), `onset_t_ns`, `duration_ms`, `sim_is_target` |
| `physiology_observation` | EEG / EOG source | `log_p[36]`, `semantics`, `source` (`simulated` + параметры d′, λ), `per_repetition` (опц.), `module_metrics` (опц.), `t_module_ns` (опц.), `sim_flash_scores` (опц., матрица n_rep × F) |
| `presentation_feedback` | Influence on presentation method | `decided`, `top_k` ([символ, p]) |
| `presentation_feedback_received` | Presentation Method | `action` (`logged_ignored` для заглушки) |

### 6.3 Языковая часть и слияние

| Тип | Блок | Payload |
|---|---|---|
| `prior_computed` | Source Model (Fast LLM) → Symbol Distribution | `model_id`, `context_source` (`primary`/`secondary`), `context_tail` (последние 64 символа), `context_len`, `log_p[36]`, `wall_latency_ms` |
| `word_candidates` | `ext:word_prediction` | `prefix`, `candidates` ([слово, p], до 10), `model_id`, `wall_latency_ms` |
| `posterior_computed` | Statistical Inference → A posteriori distribution | `fusion`, `params` (α, β, λ), `log_p[36]`, `wall_latency_ms` |
| `symbol_decided` | Primary Symbol Detector | `symbol` (null при отказе), `confidence`, `margin`, `uncertain`, `cold_start`, `abstained` |
| `primary_text_updated` | Primary Text | `op` (`append`/`delete`), `symbol`, `position`, `text` (полный текст после изменения) |
| `correction_requested` | Post-factum Text Correction | `correction_id`, `trigger`, `window_words`, `window_start_word`, `left_context_len`, `model_id`, `prompt_hash` |
| `correction_completed` | Post-factum Text Correction | `correction_id`, `words_in`, `words_out`, `accepted`, `reject_reason`, `wall_latency_ms` |
| `secondary_text_updated` | Secondary (Corrected) Text | `correction_id`, `text`, `frozen_upto_word` |

### 6.4 RL-блок (заглушка)

| Тип | Блок | Payload |
|---|---|---|
| `rl_broadcast` | Direct/Indirect RL | `batch_id`, `correction_id`, `recipients` (`["EEG/EOG source", "Primary Symbol Detector"]`), `labels` ([`epoch_id`, `detected`, `corrected`]) |
| `rl_update_received` | EEG / EOG source; Primary Symbol Detector | `batch_id`, `action` (`logged_ignored` \| `calibrated`), `params_after` (при калибровке MA-22) |

### 6.5 Точки расширения

| Тип | Блок | Payload |
|---|---|---|
| `ui_rendered` | `ext:ui` | `primary_text`, `secondary_text`, `candidates_shown`, `status` |
| `tts_requested` | `ext:tts` | `utterance_id`, `text`, `t_received_ns` |
| `tts_audio_started` | `ext:tts` | `utterance_id`, `t_audio_start_ns`, `latency_ms` (MA-32) |
| `command_issued` | `ext:commands` | `command` (`save`/`send`/…), `text`, `result` |

## 7. Таблица данных (по шаблону «Модель данных»)

| № | Данные | Источник (блок схемы) | Формат / структура | Временная метка / синхронизация | Когда пишется | Используется при replay | Требование (NEED/REQ) | Примечание |
|---|---|---|---|---|---|---|---|---|
| 1 | Манифест сессии: ID, режимы, репертуар, версии, сид, оборудование | Session Recorder | `session.json` (раздел 4) | `started_utc`, `clock.t0_ns` | старт и закрытие сессии | да (репертуар, сид, `replay_of`) | NEED-F-13, F-14 | |
| 2 | Снимок настроек стимуляции, декодирования, БЯМ | Session Recorder | `config.yaml` + `config_hash` в событии | на старте | старт сессии | да (как база для нового конфига) | NEED-F-11, F-13 | |
| 3 | Целевой текст задания | Session Recorder | событие `task_assigned`, `texts/target.txt` | `t_ns` | выдача задания | да | NEED-F-13 | в режиме live — если протокол copy-spelling |
| 4 | Начало эпохи и параметры стимуляции | One Epoch | событие `epoch_started` | `t_ns` | начало каждой эпохи | да (позиции, цели в симуляции) | NEED-F-13, REQ-CONF-01 | `sim_target_symbol` — только в симуляции |
| 5 | События и метки предъявления стимулов | Presentation Method | событие `stimulus_event` | `onset_t_ns` | каждая вспышка | нет | NEED-F-13, REQ-CONF-01 | в симуляции — модельные; в live — от модуля физиологии |
| 6 | Распределение от физиологии | EEG / EOG source | событие `physiology_observation`, `log_p[36]` | `t_ns`; `t_module_ns` внешнего модуля | конец эпохи | **да — основной вход replay** | NEED-F-04, F-13 | контракт MA-03, Q-01 |
| 7 | Сырой сигнал ЭЭГ/ЭОГ | EEG / EOG source (внешний модуль) | ссылка в `external/refs.json` + sha256 | часы внешнего модуля | по завершении сессии | нет | NEED-F-13 | в прототипе отсутствует; ответственность модуля физиологии, Q-30 |
| 8 | Априорное распределение следующего символа | Source Model → Symbol Distribution | событие `prior_computed`, `log_p[36]` | `t_ns`, `wall_latency_ms` | начало эпохи | нет (пересчитывается) | NEED-F-06, F-13 | хранится для анализа и EXP-01 |
| 9 | Кандидаты следующего слова Top-N | `ext:word_prediction` | событие `word_candidates` | `t_ns`, `wall_latency_ms` | начало эпохи | нет (пересчитывается) | NEED-F-08, REQ-LLM-03 | нет на схеме v3, Q-08 |
| 10 | Апостериорное распределение | Statistical Inference → A posteriori | событие `posterior_computed`, `log_p[36]` | `t_ns` | после наблюдения | нет (пересчитывается) | NEED-F-04, F-13 | |
| 11 | Обратная связь в презентацию | Influence on presentation method | события `presentation_feedback`, `presentation_feedback_received` | `t_ns` | после слияния | нет | — | заглушка, MA-16, Q-11 |
| 12 | Выбранный символ, уверенность, флаги | Primary Symbol Detector | событие `symbol_decided` | `t_ns` | конец эпохи | нет (пересчитывается) | NEED-F-05, F-13 | MA-14, MA-15 |
| 13 | Первичный текст | Primary Text | событие `primary_text_updated`, `texts/primary_final.txt` | `t_ns` | каждое изменение | нет (пересчитывается) | NEED-F-05, F-13 | |
| 14 | Запросы и результаты коррекции | Post-factum Text Correction | события `correction_requested`, `correction_completed` | `t_ns`, `wall_latency_ms` | конец слова (MA-19) | нет (пересчитывается) | NEED-F-07, F-13 | `prompt_hash` связывает с версией промпта |
| 15 | Исправленный текст | Secondary (Corrected) Text | событие `secondary_text_updated`, `texts/secondary_final.txt` | `t_ns` | после принятой коррекции | нет (пересчитывается) | NEED-F-07, F-13 | |
| 16 | Псевдометки и их доставка | Direct/Indirect RL | события `rl_broadcast`, `rl_update_received` | `t_ns` | после коррекции | нет | — | вне ТЗ, заглушка, MA-21, Q-18 |
| 17 | Моменты получения строки и начала аудио | `ext:tts` | события `tts_requested`, `tts_audio_started` | `t_received_ns`, `t_audio_start_ns` | каждая подсказка | нет | NEED-F-10, REQ-AUDIO-01 | заглушка, MA-32 |
| 18 | Состояние интерфейса | `ext:ui` | событие `ui_rendered` | `t_ns` | каждое обновление | нет | NEED-F-09 | |
| 19 | Команды над текстом | `ext:commands` | событие `command_issued` | `t_ns` | по команде | нет | NEED-F-09 | заглушка |
| 20 | Предупреждения и ошибки | любой блок | события `warning`, `error` | `t_ns` | по факту | нет | NEED-F-12 | нужны для оценки полноты записи |
| 21 | Метрики сессии | evaluation | `derived/metrics.json` | — | после сессии | нет | NEED-M-04, VER-EXP-04 | производные, пересчитываются |

## 8. Производные таблицы

**`tables/epochs`** — одна строка на эпоху, собирается из событий с одинаковым `epoch_id`:

`session_id, participant_id, experiment_id, task_id, llm_mode, paradigm, epoch_id, position, retry_idx, t_start_ns, t_end_ns, n_repetitions, flash_ms, isi_ms, sim_target_symbol, prior_log_p[36], phys_log_p[36], post_log_p[36], fusion, alpha, beta, decided, confidence, margin, uncertain, cold_start, abstained, prior_latency_ms, top1_word, top_k_feedback`

**`tables/corrections`** — одна строка на вызов медленной БЯМ:

`session_id, correction_id, trigger, window_start_word, words_in, words_out, accepted, reject_reason, model_id, prompt_hash, latency_ms, t_requested_ns, t_completed_ns`

Формат — Parquet при наличии `pyarrow`, иначе CSV (векторы в CSV — строки JSON).

## 9. Пакет экспорта (NEED-F-14)

Команда `python -m eeg_speller export <session_dir | experiment_dir>` формирует архив `<id>.zip`:
- `session.json`, `config.yaml`, `events.jsonl`, `texts/`, `tables/`, `derived/`, `external/refs.json`;
- `README.txt` — версия схемы, описание полей (генерируется из этого документа);
- `checksums.sha256`;
- для серии сессий — сводная `experiment_epochs.parquet` и `experiment_metrics.csv` (одна строка на сессию).

Перед упаковкой выполняется проверка полноты:
- `seq` идёт без пропусков;
- у каждой эпохи есть `epoch_started`, `physiology_observation`, `posterior_computed`, `symbol_decided`;
- у каждого `correction_requested` есть `correction_completed`;
- число символов в `primary_final.txt` согласовано с событиями `primary_text_updated`;
- `session_ended` присутствует.

Результат проверки пишется в `derived/integrity.json`; экспорт неполной сессии разрешён, но помечается.

## 10. Использование при replay

Режим replay (ARCHITECTURE, раздел 6) читает из исходной сессии манифест (репертуар, задание, сид), события `epoch_started` и `physiology_observation`. Всё остальное пересчитывается с новым конфигом и пишется в новую сессию с `replay_of = <исходный session_id>`. Так одну запись физиологии можно прогнать с разными языковыми моделями, способами слияния и правилами коррекции (EXP-01…EXP-07).

## 11. Версионирование схемы

- Добавление нового типа события или необязательного поля — без смены версии.
- Переименование, удаление поля или изменение его смысла — новая версия `eeg-speller/events@2`, конвертер старых журналов в `scripts/`.
- Этот документ и `scripts/gen_data_model.py` должны совпадать: тест сверяет каталог событий в коде с таблицами разделов 6 и 7.
