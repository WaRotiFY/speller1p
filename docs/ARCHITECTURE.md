# Архитектура

Спеллер состоит из нескольких процессов, которые общаются через LSL, и ядра на Python со сменными частями.

## Процессы

| Процесс | Что делает | Как запускается |
|---|---|---|
| **Спеллер и окно оператора** | цикл эпох, языковые модели, слияние, решение, запись сессии; веб-окно оператора на 127.0.0.1 | `python -m eeg_speller.external.server` |
| **Модуль стимуляции** | объявляет свою клавиатуру, рисует экран испытуемого и шлёт метки стимулов; может сам декодировать ЭЭГ. Все параметры — в его `settings.json` | окно оператора запускает выбранный модуль из `stimulus_kit/modules/` без аргументов (или вручную) |
| **Декодер ЭЭГ** | присылает правдоподобие символов | отдельно, например `stimulus_kit/decoders/p300_lsl_decoder.py`, или внутри модуля стимуляции |
| **Источник ЭЭГ** | поток отсчётов в LSL | усилитель / его мост в LSL / `stimulus_kit/tools/edf_to_lsl.py` |

LSL-потоки протокола `speller-link@1`: `SpellerControl` (спеллер → модули), `SpellerStimuli` (метки стимулов), `SpellerLikelihood` (правдоподобие). Описание — [`stimulus_kit/CONTRACT.md`](../stimulus_kit/CONTRACT.md).

## Эпоха (набор одного символа)

```mermaid
sequenceDiagram
    autonumber
    participant S as Спеллер
    participant M as Модуль стимуляции
    participant D as Декодер
    participant L as Медленная БЯМ (фон)
    M->>S: keyboard (символы, раскладка) — один раз при подключении
    S->>S: подсказка быстрой БЯМ (prior)
    S->>M: epoch_start (клавиатура, текст)
    loop каждый стимул
        M->>S: flash (символы, метка LSL после flip)
        D->>S: правдоподобие (обновление)
        S->>S: слияние prior × правдоподобие → posterior
    end
    S->>M: stop (уверенность ≥ порога)
    D->>S: правдоподобие final
    S->>S: выбор символа, текст
    S->>M: result (символ, текст)
    opt конец N-го слова
        S->>L: окно последних слов + варианты букв
        L-->>S: исправленные слова → исправленный текст
    end
```

- **Окончание эпохи.** Решение принимается по оценке с `final=true`. Если её нет, решение принимается по последней оценке через заданное время после `stop` или после `done` модуля. Если не пришло ни одной оценки, эпоха прерывается и повторяется.
- **Фоновая коррекция.** Медленная БЯМ работает в фоновом потоке и не задерживает следующую эпоху.

## Сменные части ядра

| Часть | Где | Как заменить |
|---|---|---|
| Клавиатура | модуль стимуляции; проверка и перевод в `core/keyboard.py` | модуль объявляет её в `SpellerLink(keyboard=…, layout=…)`; символы приводятся к алфавиту БЯМ (заглавные → строчные, Ё → е, `_` → пробел). Без объявления — клавиатура профиля |
| Быстрая БЯМ | `llm/fast.py`, `llm/backends/` | `ngram` (файл в `models/ngram`), `kenlm` (имя на Hugging Face или путь), `hf_causal` (имя модели), `mock_ngram` |
| Статистический вывод | `inference/*.py`, загрузчик `epoch/plugins.py` | новый файл с `PARAMS` и `combine(prior, likelihood, params)` |
| Медленная БЯМ | `llm/slow.py`, `llm/backends/chat_api.py`, `mock_correction.py` | провайдер и модель в окне оператора; выбор облачного провайдера разрешает отправку ему последних слов; промпт — редактор в окне или `prompts/correction_v2.txt` |
| Модуль стимуляции | `stimulus_kit/modules/` | новая папка с `main.py` и `settings.json` |
| Декодер | отдельный процесс | любой, кто шлёт `SpellerLikelihood` |

Интерфейс быстрой модели:

```python
class FastLanguageModel(Protocol):
    model_id: str
    def predict(self, context: str, repertoire: Repertoire) -> SymbolDistribution: ...   # log P(символ)
    def top_words(self, context: str, n: int) -> list[tuple[str, float]]: ...
```

Интерфейс медленной модели: `correct(left_context: str, window: list[str], letters=None) -> list[str]`. Слов должно вернуться столько же, иначе ответ отклоняется.

## Слияние и решение

- **Слияние.** Модуль `inference/` получает логарифмы prior и правдоподобия по символам клавиатуры и возвращает логарифмы апостериорных вероятностей. По умолчанию это `log_pool`: `α·log prior + β·log правдоподобие` (при α = β = 1 — формула Байеса).
- **Без БЯМ.** Апостериорное распределение равно правдоподобию.
- **Решение.** Выбирается символ с максимальной апостериорной вероятностью. Если она ниже `detector.tau`, у решения ставится флаг «неуверен».
- **Досрочная остановка.** `stop` посылается, когда максимум апостериорной вероятности ≥ `external.stop_confidence` и обновлений не меньше `min_updates_before_stop`.

## Время

- **Метки LSL.** Каждое событие сессии получает `lsl_s` — время по часам LSL этого компьютера. Это те же часы, в которых приходят метки стимулов и отсчёты ЭЭГ.
- **Монотонные часы.** Кроме того, у события есть `t_ns` (монотонные часы процесса) и сквозной номер `seq`.
- **Метки стимулов** ставит модуль после `flip`; LSL переводит их в часы приёмника.
- **Ожидание подключения.** `SpellerLink` при старте ждёт, пока спеллер подключится к потокам модуля, поэтому первые метки не теряются.

## Режимы без окна оператора

| Режим | Что это | Команда |
|---|---|---|
| симуляция | виртуальный участник и симулированная ЭЭГ с заданным качеством сигнала (`physiology.sim.d_prime`), виртуальное время | `python -m eeg_speller run`, `experiment` |
| переигрывание | стимулы и правдоподобие ЭЭГ эпохи k берутся из эпохи k записанной сессии (на той клавиатуре, что объявил модуль) и проигрываются тому же контроллеру, что в живой сессии; языковая часть, слияние, досрочная остановка и решение пересчитываются. Более ранний стоп учитывается: полученные к нему данные становятся окончательными | `python -m eeg_speller replay <сессия> [--config изменения.yaml] [--llm off]` (`recording/replay_external.py`) |

## Пакет `eeg_speller`

```
external/     server.py (окно оператора, API), controller.py (эпохи со сменными модулями),
              transport.py (LSL), launcher.py (поиск и запуск модулей), tasks.py, static/
core/         config.py, keyboard.py, distributions.py, clock.py, events.py
llm/          fast.py, slow.py, char_marginal.py
  backends/   char_ngram.py, kenlm_words.py, hf_causal.py, mock_ngram.py, chat_api.py, mock_correction.py
epoch/        plugins.py (статистический вывод), loop.py и fusion.py (симуляция), presentation_feedback.py
decoding/     detector.py
text/         primary.py, secondary.py, normalize.py
recording/    recorder.py, storage.py, export.py, replay.py (симуляция), replay_external.py (окно оператора)
physiology/   base.py (типы), simulated.py, p300.py, sources.py, edf_demo.py, calibrate.py, stimuli.py
evaluation/   metrics.py, stats.py, experiment.py, bench.py
ext/          word_prediction.py, ui_console.py, tts.py, commands.py
learning/     rl_block.py, calibration.py (псевдометки по исправлениям; офлайн-калибровка)
```

Проектные допущения (MA-NN) и открытые вопросы (Q-NN), на которые ссылаются комментарии в коде, собраны в [`docs/research/`](research/).
