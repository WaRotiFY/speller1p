# Открытые данные и готовый внешний анализатор

Проверка идеи «физиологию можно подключить готовую»: реальная запись чужой лаборатории проигрывается через `eeg_source.py`, анализирует её открытый P300-конвейер из сторонней библиотеки, а спеллер (или судья) получает только `SpellerLikelihood`.

```
bnci2014_008_A0x.edf ─ eeg_source.py replay ─┬─ SpellerEEG ─────┐
  (+ markers_bnci2014_008.json)              ├─ SpellerStimuli ─┼─ external_pipeline_decoder.py run ─ SpellerLikelihood ─ спеллер / replay_score.py
                                             ├─ SpellerControl  │      (pyRiemann или sklearn)
                                             └─ SpellerReplayMarkers (target:X — только судье)
```

**База:** `release-0.2` (протокол `speller-link@2`) плюс коммит `eeg_source.py` из ветки `eeg-source-edf-replay`; в этой ветке он идёт первым коммитом. Декодер учитывает `epoch_cancelled` и присылает `through_flash`: окна оцениваются строго подряд, без пропусков. Проверено: `pytest tests` — 48 passed, 1 skipped; онлайн-прогон A01 — 19/20.

## Данные: BNCI 2014-008

P300-спеллер, 8 пациентов с БАС, 8 каналов (Fz Cz Pz Oz P3 P4 PO7 PO8), 256 Гц, матрица 6×6 строки/столбцы. На каждого: 35 символов по 120 вспышек, то есть 10 повторений, как в VER-EXP-05. Вспышка 125 мс, SOA 250 мс. Riccio et al. 2013, doi:10.3389/fnhum.2013.00732. Лицензия CC BY-NC-ND 4.0.

Откуда: <https://bnci-horizon-2020.eu/database/data-sets> (008-2014, файлы `A01.mat`…`A08.mat`). Те же файлы, побайтно совпадающие с оригиналом (SHA-1 `7f7281f3…` для A01), лежат в публичных репозиториях на GitHub: `AyanaMussabayeva/P300-Speller-BCI-Research` (`BCI_Horizon2020_dataset/`) и `zuhairmhtb/P300SpellerBasedEEGClassification` (`DataFiles/`).

Набранный текст по разметке: `2BACI5ROSEZUPPAGATTOMENTEVIOLAREBUS`.

```powershell
uv run --no-sync python stimulus_kit/tools/open_data/bnci2014_008_to_edf.py A01.mat A02.mat … --out data/bnci2014_008
uv run --no-sync python stimulus_kit/tools/eeg_source.py info data/bnci2014_008/bnci2014_008_A01.edf
```

В EDF+ пишутся аннотации `epoch` (начало символа), `target:X` (цель; идёт только в `SpellerReplayMarkers`), `R1…R6`, `C1…C6` (вспышки). Все 4270 меток на файл сохраняются, это проверяется через `info`.

## Анализатор: `stimulus_kit/decoders/external_pipeline_decoder.py`

Анализ делает готовый конвейер из сторонней библиотеки. Файл только вырезает окна по меткам и складывает оценки вспышек в правдоподобия символов.

| `--pipeline` | Что внутри |
|---|---|
| `xdawn_riemann` | pyRiemann: XdawnCovariances → TangentSpace → LogisticRegression, 1–20 Гц, 0–800 мс. Эталонный P300-конвейер бенчмарка MOABB |
| `rlda` | scikit-learn: 0,5–10 Гц, 0–800 мс, прореживание до ~20 Гц, LDA со сжатием (Krusienski 2008; то же делает `CMDModelRLDA` в MEDUSA) |

Один и тот же `preprocess` работает и офлайн, и онлайн. Оценка вспышки — логарифм отношения правдоподобий «цель / не цель» без априорной доли целей. Символ получает сумму оценок вспышек, в которых он был.

```powershell
pip install pyriemann pylsl pyedflib joblib
$D = "stimulus_kit/decoders/external_pipeline_decoder.py"
$M = "stimulus_kit/tools/open_data/markers_bnci2014_008.json"
$E = "data/bnci2014_008/bnci2014_008_A01.edf"

# калибровка на первых 15 символах
uv run --no-sync python $D calibrate $E --marker-map $M --train-epochs 15 --pipeline xdawn_riemann --out runs/A01.joblib
# офлайн-проверка на остальных 20: точность в зависимости от числа повторений
uv run --no-sync python $D evaluate $E --marker-map $M --model runs/A01.joblib --skip-epochs 15
```

Онлайн, три окна (`--start` — секунда 16-го символа минус 0,5 с, её показывает `info`; `--speed 4` допустим только без живого спеллера):

```powershell
uv run --no-sync python stimulus_kit/tools/open_data/replay_score.py --out runs/online_A01.json
uv run --no-sync python $D run --model runs/A01.joblib --eeg-name SpellerEEG
uv run --no-sync python stimulus_kit/tools/eeg_source.py replay $E --marker-map $M --as-stimulus --as-control --start 588.623 --speed 4
```

С живым спеллером вместо судьи: `--as-stimulus` без `--as-control`, `--speed 1`. В окне оператора нужно задать клавиатуру 6×6 из карты меток. Номера эпох совпадут с номерами спеллера только при одинаковом порядке (см. `EEG_SOURCE.md`, раздел 6).

## Результаты (2 ноября 2026)

Калибровка на 15 символах, проверка на 20 оставшихся (`GATTOMENTEVIOLAREBUS`), среднее по 8 участникам:

| Конвейер | AUC вспышки | Сбаланс. точность вспышки | Символы, 1 повт. | 3 | 5 | 10 |
|---|---|---|---|---|---|---|
| xdawn_riemann | 0,832 | 0,752 | 0,31 | 0,64 | 0,78 | **0,93** |
| rlda | 0,846 | 0,772 | 0,26 | 0,62 | 0,78 | **0,96** |

По участникам при 10 повторениях: xdawn_riemann — 0,95 / 0,85 / 0,90 / 0,95 / 0,90 / 0,95 / 0,95 / 1,00; rlda — 0,95 / 0,95 / 0,90 / 0,95 / 0,90 / 1,00 / 1,00 / 1,00. Ошибки в основном «соседние»: перепутана строка или столбец (G→M, E→D).

**Онлайн через LSL** (A01, xdawn_riemann, ×4): набрано `GATTOMDNTEVIOLAREBUS`, 19 из 20. Результат символ в символ совпал с офлайн-проверкой, все 2400 вспышек оценены, ни одно окно не потеряно.

**К REQ-PERF-01.** Если «точность бинарной классификации» понимать по отдельной вспышке, получается 75–77 %, ниже 80 %. Если по символу после 10 повторений — 93–96 %. Это пациенты с БАС; на здоровых обычно выше. Формулировку стоит уточнить (см. `OPEN_QUESTIONS`).

## Ловушка: единицы

`eeg_speller.physiology.sources.LSLSource.window()` отдаёт **вольты**, а `eeg_source.load()` и EDF — **микровольты**. Модель, обученная на файле, получит в онлайне сигнал в 10⁶ раз меньше и почти постоянные оценки. Причём из-за шума округления часть букв всё равно «угадывается»: так было в первом прогоне, 15 из 20 при p≈1/36. Внешний декодер переводит окно обратно в мкВ (`w * 1e6`). Любой новый декодер на `LSLSource` должен делать так же или обучаться на вольтах.
