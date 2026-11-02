"""Run an actual pygame module and actual LSL, injecting keyboard responses.

No phrase is inserted into the speller: only pygame key events are generated.
Use run_current_speller.py --headless tests/manual_paradigm_smoke.py NAME [--cancel].
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'stimulus_kit'), str(ROOT/'stimulus_kit/examples')]
import numpy as np
import pygame
import module_tools
import speller_link
from eeg_speller.external.controller import ExternalSession
from eeg_speller.external.server import build_session_config
from eeg_speller.recording.storage import read_events

name = sys.argv[1]
cancel = '--cancel' in sys.argv
folder = ROOT/'stimulus_kit/modules'/name
out = ROOT/'runs/paradigm_checks'/(name + ('_cancel' if cancel else ''))
out.mkdir(parents=True, exist_ok=True)
cfg = json.loads((folder/'settings.json').read_text())
cfg.update(sentence='ба', rest_before_ms=0, cue_ms=0, rest_after_ms=0,
           fullscreen=False, n_trials=2, num_samples=2, flash_ms=40, isi_ms=20,
           min_gap=0, t0_mean=.35, t1_a=.02, t1_b=.2, t2_a=.01, t2_b=.02, acceleration=1)
(out/'settings.test.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
module_tools.module_args = lambda *a, **kw: SimpleNamespace(config=cfg, keyboard=True, windowed=True)
original_link = speller_link.SpellerLink
links = []
counts = {'frames':0, 'spaces':0, 'completed_trials':0, 'stimulus_frame':False}
flip = pygame.display.flip
def record_flip():
    counts['frames'] += 1
    if not counts['stimulus_frame'] and links and links[0]._flash_count > 0:
        pygame.image.save(pygame.display.get_surface(), str(out/'frame.png'))
        counts['stimulus_frame'] = True
    return flip()
pygame.display.flip = record_flip

class KeyboardParticipant(original_link):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        links.append(self)

    def trial_done(self, *args, **kwargs):
        counts['completed_trials'] += 1
        return super().trial_done(*args, **kwargs)

    def flash(self, symbols, **kw):
        onset = super().flash(symbols, **kw)
        if cancel and self._flash_count == 1:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        elif not cancel:
            position = len(self.epoch.text)
            target = cfg['sentence'][position] if position < len(cfg['sentence']) else ' '
            if any(s.lower() == target or (s == '_' and target == ' ') for s in symbols):
                def press():
                    if pygame.get_init():
                        counts['spaces'] += 1
                        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
                if name == 'flash_rows_cols':
                    press()
                else:
                    timer = threading.Timer(.18, press)
                    timer.daemon = True
                    timer.start()
        return onset

speller_link.SpellerLink = KeyboardParticipant
config, passport = build_session_config({'fast':{'backend':'uniform'}, 'slow':{'backend':'off'},
                                       'stop':{'confidence':None}})
session = ExternalSession(config, cfg['sentence'], out, passport=passport).start()
errors = []
def watchdog():
    deadline = time.monotonic()+35
    while time.monotonic() < deadline and not session.finished:
        if cancel and session.paused:
            if session.primary.value:
                errors.append('cancel inserted a character')
            session.request_stop('smoke_cancel_checked')
            return
        time.sleep(.01)
    if not session.finished:
        errors.append('paradigm timed out')
        session.request_stop('smoke_timeout')
threading.Thread(target=watchdog, daemon=True).start()
os.chdir(out)
sys.path.insert(0, str(folder))
spec = importlib.util.spec_from_file_location('paradigm_main', folder/'main.py')
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    if name == 'flash_rows_cols':
        mod.main()
finally:
    if not session.finished:
        session.request_stop('smoke_cleanup')
    session.wait_finished(10)

events = read_events(session.storage.directory)
distributions = [e['payload'] for e in events if e['type']=='trial_distribution']
decisions = [e['payload'] for e in events if e['type']=='symbol_decided']
assert session.error is None, session.error
assert not errors, errors
assert counts['frames'] > 5
if cancel:
    assert not decisions
    assert any(e['type']=='epoch_user_stopped' for e in events)
else:
    assert counts['spaces'] > 0 and len(decisions)==2
    assert counts['stimulus_frame']
    assert len(distributions) == counts['completed_trials'] == 4
    assert all(len(d['symbols'])==len(d['p']) and np.isclose(sum(d['p']),1) for d in distributions)
    # Demonstrate that a consumer can read the full distribution through the SDK.
    assert links[0].get_distribution() is not None
assert json.loads((session.storage.directory/'derived/integrity.json').read_text())['complete']
report = {'paradigm':name, 'cancel':cancel, **counts, 'text':session.primary.value,
          'decisions':len(decisions), 'trial_distributions':len(distributions),
          'keyboard_checks':sum(e['type']=='keyboard_set' for e in events),
          'recording':str(session.storage.directory)}
(out/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
for link in links:
    link.close()
print(json.dumps(report, ensure_ascii=False), flush=True)
