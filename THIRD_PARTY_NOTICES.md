# Сторонние данные, модели и код

| Что | Где используется | Источник и лицензия |
|---|---|---|
| Частоты слов OpenSubtitles 2018 (FrequencyWords) | статистика в `models/ngram/ru_char4.npz` | https://github.com/hermitdave/FrequencyWords, данные — CC BY-SA 4.0 |
| RussianNovels — тексты русских романов XIX–XX веков | статистика в `models/ngram/ru_char4.npz` (самих текстов в файле нет) | https://github.com/JoannaBy/RussianNovels |
| KenLM-модель `jonatasgrosman/wav2vec2-large-xlsr-53-russian` | быстрая БЯМ «KenLM»; скачивается пользователем, в репозитории не хранится | https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-russian, Apache-2.0 |
| KenLM (библиотека) | пакет `kenlm`, ставится пользователем | https://github.com/kpu/kenlm, LGPL-2.1 |
| Lab Streaming Layer (`pylsl`), pygame, numpy, scipy, pydantic, PyYAML, edfio | зависимости | лицензии соответствующих пакетов |
| Траектории и временная схема модулей `moving_letters` и `moving_circles` | `stimulus_kit/modules/moving_*` | основаны на парадигмах репозитория https://github.com/texnar1511/Eyes_P300 |
