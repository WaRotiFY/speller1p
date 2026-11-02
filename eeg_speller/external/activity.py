"""Operator-facing descriptions; the original structured event is still recorded."""
def describe(event):
    kind, p = event['type'], event['payload']
    if kind == 'epoch_started':
        return 'Начат выбор символа' + (f" · ожидается «{p['target_symbol']}»" if p.get('target_symbol') else '')
    if kind == 'keyboard_set':
        return f"Клавиатура подтверждена: {len(p['symbols'])} символов, {p['layout'][0]}×{p['layout'][1]}"
    if kind == 'trial_distribution':
        return f"Trial {p['trial']} обработан · распределение передано модулю"
    if kind == 'symbol_decided':
        if p['symbol'] is None:
            return 'Символ не выбран: недостаточно пригодных данных'
        status = ' · верно' if p.get('correct') is True else ' · ошибка' if p.get('correct') is False else ''
        return f"Выбран «{p.get('typed', p['symbol'])}»{status} · уверенность {p['confidence']:.0%}"
    if kind == 'epoch_user_stopped':
        return 'Пользователь отменил эпоху · символ не вводится'
    if kind == 'epoch_aborted' and p['reason'] != 'user_stop':
        return 'Эпоха завершена без выбора: ' + {'timeout':'истекло время', 'no_likelihood':'нет оценки ЭЭГ'}.get(p['reason'], p['reason'])
    if kind == 'correction_requested':
        return 'Медленная LLM обрабатывает: ' + ' '.join(p['window_words'])
    if kind == 'correction_completed':
        if not p['accepted']:
            return 'Исправление медленной LLM отклонено: ' + str(p.get('reject_reason') or 'не прошло проверку')
        before, after = ' '.join(p['words_in']), ' '.join(p['words_out'])
        return 'Медленная LLM: ' + (before + ' → ' + after if before != after else 'без изменений')
    if kind in ('session_paused', 'session_resumed', 'session_ended'):
        return {'session_paused':'Сеанс на паузе', 'session_resumed':'Сеанс продолжен', 'session_ended':'Сеанс завершён'}[kind]
    if kind in ('warning', 'error'):
        return ('Предупреждение: ' if kind == 'warning' else 'Ошибка: ') + p.get('message', '')
    return None
