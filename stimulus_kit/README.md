# stimulus_kit — модули стимуляции и декодеры

Всё, что нужно, чтобы подключить к спеллеру свой экран испытуемого или свой декодер ЭЭГ. Правила обмена — [`CONTRACT.md`](CONTRACT.md).

| Файл или папка | Что это |
|---|---|
| `speller_link.py` | единственный обязательный файл для модуля (нужен `pip install pylsl`) |
| `module_tools.py` | необязательные помощники: чтение `settings.json` модуля, клавиатура из настроек, целевая буква из `sentence`, цвета, положение окна, «пробел вместо ЭЭГ» для перекрывающихся стимулов |
| `modules/` | модули, которые видит окно оператора; свои кладите сюда |
| `examples/` | короткие примеры кода (в списке окна оператора их нет): матрица (`pygame_matrix.py`), одна буква в центре (`pygame_rsvp.py`), симулятор без экрана (`headless_sim.py`) |
| `decoders/p300_lsl_decoder.py` | пример декодера P300 по любому LSL-потоку ЭЭГ (см. `docs/DECODER.md`) |
| `tools/edf_to_lsl.py` | проигрывает EDF-запись в LSL, как живой усилитель |

Самый короткий модуль:

```python
from speller_link import SpellerLink

link = SpellerLink(keyboard=list("абвгдежзийклмнопрстуфхцчшщъыьэюя_.,⌫"), layout=(6, 6))
while (epoch := link.wait_epoch()) is not None:
    for group in my_groups(epoch.rows()):
        draw(group); pygame.display.flip()
        link.flash(group)                     # сразу после flip
        if link.should_stop():
            break
    link.done()
```

Запуск вручную (если в окне оператора выбрано «не запускать»): сначала начните сессию в окне оператора, потом запустите модуль, например

```powershell
uv run --no-sync python stimulus_kit/modules/flash_rows_cols/main.py
uv run --no-sync python stimulus_kit/examples/pygame_matrix.py --keyboard
```
