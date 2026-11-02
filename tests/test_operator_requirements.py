"""Behavioural checks for the agreed operator/physiology requirements."""
import csv
import json
from pathlib import Path
import time

import numpy as np
import pytest

from eeg_speller.core.config import load_config
from eeg_speller.core.distributions import Repertoire
from eeg_speller.evaluation.online import performance
from eeg_speller.external.controller import ExternalSession
from eeg_speller.external.server import build_session_config
from eeg_speller.external.transport import MemoryTransport
from eeg_speller.llm.backends.simple_prior import SimplePrior
from eeg_speller.recording.storage import read_events

ROOT = Path(__file__).resolve().parents[1]


def wait_for(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.005)
    assert predicate(), 'timed out waiting for state'


def session(tmp_path, target='', **opts):
    cfg, _ = build_session_config({'fast': {'backend': 'uniform'}, 'slow': {'backend': 'off'}})
    cfg = cfg.model_copy(update={'external': {**cfg.external, **opts},
                                 'stimulus': {**cfg.stimulus, 'pause_ms': 500}})
    tr = MemoryTransport()
    return ExternalSession(cfg, target, tmp_path, transport=tr), tr


def keyboard(tr, msg, symbols=('а', 'б')):
    tr.deliver('stimuli', {'event': 'keyboard', 'epoch': msg['epoch'],
        'session_id': msg['session_id'], 'symbols': list(symbols), 'layout': [1, len(symbols)]})


def test_simple_priors_use_actual_keyboard_and_raw_text():
    rep = Repertoire(tuple('абв⌫'), (1, 4))
    uniform, frequency = SimplePrior({'backend': 'uniform'}), SimplePrior({'backend': 'frequency'})
    assert np.allclose(np.exp(uniform.predict('аааа', rep).log_p), [.25]*4)
    assert np.allclose(frequency.probabilities('', rep), [.25]*4)
    assert np.allclose(frequency.probabilities('ааб', rep), [2/3, 1/3, 0, 0])
    assert np.allclose(frequency.probabilities('аа', rep), [1, 0, 0, 0])  # deletion
    assert np.all(np.isfinite(frequency.predict('аа', rep).log_p))
    other = Repertoire(tuple('ЕЁ_⌫'), (1, 4))
    assert np.allclose(frequency.probabilities('ее ', other), [1/3, 1/3, 1/3, 0])


def test_baseline_prior_does_not_disable_slow_correction(tmp_path):
    for backend in ('uniform', 'frequency'):
        cfg, _ = build_session_config({'fast': {'backend': backend}, 'slow': {'backend': 'dictionary'}})
        s = ExternalSession(cfg, '', tmp_path, transport=MemoryTransport())
        try:
            assert s.fast.backend == backend and s.slow is not None
            s.primary.value, s.secondary.value = 'ааб', 'ббб'
            s.cfg.fast_llm['context_source'] = 'secondary'
            p = np.exp(s._prior()[0].log_p)
            if backend == 'frequency':
                assert p[s.rep.index('а')] > p[s.rep.index('б')]
        finally:
            s.primary.value = s.secondary.value = ''
            s._finish()


def test_keyboard_confirmed_each_epoch_and_correctness_is_persisted(tmp_path):
    s, tr = session(tmp_path, 'аб')
    prepares = []
    def module(msg):
        if msg['cmd'] == 'epoch_prepare':
            prepares.append(msg['epoch'])
            keyboard(tr, msg, ('а', 'б') if msg['epoch'] == 0 else ('в', 'б', 'а'))
        elif msg['cmd'] == 'epoch_start':
            want = 'а' if msg['epoch'] == 0 else 'б'
            tr.deliver('likelihood', {'epoch': msg['epoch'], 'p': {k: .999 if k == want else .001 for k in msg['symbols']}, 'final': True})
    tr.listeners.append(module)
    s.start()
    assert s.wait_finished(10) and s.error is None
    assert s.primary.value == 'аб' and prepares == [0, 1]
    events = read_events(s.storage.directory)
    decisions = [e['payload'] for e in events if e['type'] == 'symbol_decided']
    assert [d['correct'] for d in decisions] == [True, True]
    assert [d['target_position'] for d in decisions] == [0, 1]
    assert [d['n_symbols'] for d in decisions] == [2, 3]
    # Alternatives for the first letter retain the old keyboard's symbol names.
    assert s._letters(0, 1)[0][0][0][0] == 'а'
    rows = list(csv.DictReader((s.storage.directory/'tables/epochs.csv').open(encoding='utf-8-sig')))
    assert [r['correct'] for r in rows] == ['True', 'True']
    saved = json.loads((s.storage.directory/'derived/operator_metrics.json').read_text())
    assert saved['accuracy'] == 1 and saved['itr_bpm'] > 0
    assert saved['itr_bpm'] == s.monitor()['stats']['itr_bpm']


def test_trial_distribution_waits_for_matching_evidence_and_contains_full_keyboard(tmp_path):
    s, tr = session(tmp_path, 'а', stop_confidence=None)
    exports = []
    def module(msg):
        if msg['cmd'] == 'epoch_prepare':
            keyboard(tr, msg)
        elif msg['cmd'] == 'epoch_start':
            for i in range(1, 5):
                tr.deliver('stimuli', {'epoch': 0, 'flash': ['а','б'], 'flash_id': i})
            tr.deliver('stimuli', {'epoch': 0, 'event': 'trial_done', 'trial': 1, 'through_flash': 4})
            tr.deliver('likelihood', {'epoch': 0, 'p': [.8,.2], 'through_flash': 3})
        elif msg['cmd'] == 'distribution':
            exports.append(msg)
    tr.listeners.append(module)
    s.start()
    wait_for(lambda: s.current and s.current['updates'] == 1)
    assert exports == []  # Trial ended, but its last EEG response is pending.
    tr.deliver('likelihood', {'epoch': 0, 'p': [.9,.1], 'through_flash': 4})
    wait_for(lambda: len(exports) == 1)
    assert exports[0]['p'] == pytest.approx([.9,.1], abs=1e-6)
    assert exports[0]['semantics'] == 'posterior' and exports[0]['symbols'] == ['а','б']
    assert exports[0]['trial'] == 1 and exports[0]['through_flash'] == 4
    tr.deliver('stimuli', {'epoch': 0, 'event': 'trial_done', 'trial': 2, 'through_flash': 8})
    tr.deliver('likelihood', {'epoch': 0, 'p': [.99,.01], 'through_flash': 8, 'final': True})
    assert s.wait_finished(5) and s.error is None
    assert [x['trial'] for x in exports] == [1,2]
    assert exports[1]['p'] == pytest.approx([.99,.01], abs=1e-6)


def test_user_stop_discards_final_message_in_same_batch_and_pauses(tmp_path):
    s, tr = session(tmp_path, 'а')
    def module(msg):
        if msg['cmd'] == 'epoch_prepare':
            keyboard(tr, msg)
        elif msg['cmd'] == 'epoch_start':
            if msg['epoch'] == 0:
                tr.deliver('likelihood', {'epoch': 0, 'p': [.99,.01]})
                tr.deliver('stimuli', {'epoch': 0, 'event': 'user_stop'})
                tr.deliver('likelihood', {'epoch': 0, 'p': [.99,.01], 'final': True})
            else:
                tr.deliver('likelihood', {'epoch': msg['epoch'], 'p': [.99,.01], 'final': True})
    tr.listeners.append(module)
    s.start()
    wait_for(lambda: s.paused)
    assert s.primary.value == '' and s.target_index == 0
    assert s.log[0]['decided_on'] == 'user_stop'
    assert s.monitor()['stats']['selected'] == 0
    s.set_paused(False)
    assert s.wait_finished(5) and s.primary.value == 'а'
    events = read_events(s.storage.directory)
    assert not any(e['type']=='symbol_decided' and e['epoch_id']==0 for e in events)
    assert json.loads((s.storage.directory/'derived/integrity.json').read_text())['complete']


def test_final_likelihood_cannot_overtake_last_trial_marker(tmp_path):
    s, tr = session(tmp_path, 'а', stop_confidence=None)
    started = []
    def module(msg):
        if msg['cmd'] == 'epoch_prepare':
            keyboard(tr, msg)
        elif msg['cmd'] == 'epoch_start':
            started.append(msg)
            tr.deliver('likelihood', {'epoch': 0, 'keyboard_id': msg['keyboard_id'],
                'p': [.99,.01], 'through_flash': 4, 'final': True})
    tr.listeners.append(module)
    s.start()
    wait_for(lambda: s.current and s.current['updates'] == 1)
    time.sleep(.05)
    assert s.primary.value == ''  # Final arrived on the faster LSL stream.
    tr.deliver('stimuli', {'epoch': 0, 'event': 'trial_done', 'trial': 1, 'through_flash': 4})
    wait_for(lambda: any(m['cmd']=='distribution' for m in tr.sent))
    assert s.primary.value == ''
    tr.deliver('stimuli', {'epoch': 0, 'event': 'done'})
    assert s.wait_finished(5) and s.primary.value == 'а'
    events = read_events(s.storage.directory)
    kinds = [e['type'] for e in events]
    assert kinds.index('trial_distribution') < kinds.index('symbol_decided')


def test_missing_or_stale_keyboard_never_starts_presentation(tmp_path):
    s, tr = session(tmp_path, 'а', keyboard_wait_s=.05)
    def module(msg):
        if msg['cmd'] == 'epoch_prepare':
            wrong = dict(msg, epoch=msg['epoch']+1)
            keyboard(tr, wrong)
    tr.listeners.append(module)
    s.start()
    wait_for(lambda: s.paused)
    assert not any(m['cmd']=='epoch_start' for m in tr.sent)
    s.request_stop()
    assert s.wait_finished(5)


def test_itr_units_keyboard_sizes_cer_and_free_spelling():
    rows = [{'decided':'а','correct':True,'n_symbols':2,'duration_s':1},
            {'decided':'б','correct':True,'n_symbols':4,'duration_s':1}]
    m = performance(rows, 'аб', 'аб', 'аб', 60)
    assert m['itr_bpm'] == pytest.approx(3)  # one bit + two bits, in one minute
    assert m['selection_rate_spm'] == 2 and m['speed_cpm'] == 2
    assert m['cer_primary'] == 0
    assert performance(rows, 'аб', 'ав', 'аб', 60)['cer_primary'] == .5
    free = performance([{**r,'correct':None} for r in rows], '', 'аб', 'аб', 60)
    assert free['accuracy'] is None and free['itr_bpm'] is None and free['cer_primary'] is None
    assert performance([{**rows[0], 'correct':False}], 'а', 'б', 'б', 60)['itr_bpm'] == 0


def test_frequency_prior_updates_after_real_insert_and_delete(tmp_path):
    cfg, _ = build_session_config({'fast': {'backend': 'frequency'}, 'slow': {'backend': 'off'}})
    cfg = cfg.model_copy(update={'stimulus': {**cfg.stimulus, 'pause_ms': 500}})
    tr = MemoryTransport()
    s = ExternalSession(cfg, '', tmp_path, transport=tr)
    plan = ['а', 'б', 'а', '⌫', 'б']
    def module(msg):
        if msg['cmd'] == 'epoch_prepare':
            keyboard(tr, msg, ('а', 'б', '⌫'))
        elif msg['cmd'] == 'epoch_start':
            want = plan[msg['epoch']]
            tr.deliver('likelihood', {'epoch': msg['epoch'],
                'p': [1 if symbol == want else 0 for symbol in msg['symbols']], 'final': True})
        elif msg['cmd'] == 'result' and msg['epoch'] == 4:
            s.request_stop('test_finished')
    tr.listeners.append(module)
    s.start()
    assert s.wait_finished(10) and s.error is None and s.primary.value == 'абб'
    priors = [e['payload']['raw_p'] for e in read_events(s.storage.directory) if e['type']=='prior_computed']
    assert np.allclose(priors, [[1/3]*3, [1,0,0], [.5,.5,0], [2/3,1/3,0], [.5,.5,0]])
