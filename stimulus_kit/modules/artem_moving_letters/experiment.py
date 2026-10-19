"""Парадигма 0.0 Артёма (движение букв), подключённая к спеллеру по контракту speller-link@1.

Основа — Eyes_P300/Paradigm_0/0.0/experiment.py (github.com/texnar1511/Eyes_P300).
Логика движения (speed_func, t0/t1/t2, амплитуды) не тронута. Изменения помечены
комментарием «SPELLER-LINK»:

1. буквы и раскладку присылает спеллер (epoch.symbols, epoch.layout), а не settings.json;
2. эпохи идут по команде спеллера (link.wait_epoch), а не по предложению из settings.json;
3. эпоха стартует сама, пробел не нужен (если не включён start_by_space);
4. сразу после flip кадра, в котором буква начала движение, вызывается link.flash([буква]);
5. эпоха заканчивается, когда КАЖДАЯ буква сделала num_samples движений (цель модулю
   неизвестна) или когда спеллер прислал stop;
6. после эпохи показывается выбранная буква (link.wait_result);
7. подсказка (целевая буква) показывается, только если спеллер прислал epoch.cue.
"""
import math
import os
import random
from datetime import datetime
import json

import numpy as np
import pygame

from module_tools import AsyncKeyboardDecoder, color, log_path, place_window


class Experiment:

    def __init__(self, data_config, link, keyboard=False, windowed=False):

        self.data_config = data_config
        self.link = link                          # SPELLER-LINK
        self.keyboard = keyboard
        self.windowed = windowed

    def fit(self):

        self.preinit()
        self.init()

        # SPELLER-LINK: epochs come from the speller instead of data_config['sentence']
        while True:
            epoch = self.wait_epoch()
            if epoch is None:
                break
            self.postinit(epoch)
            self.perform(epoch)
            if self.quit:
                break

        self.end_exp()

    def preinit(self):

        if not self.windowed:
            place_window(self.data_config.get('window_x'), self.data_config.get('window_y'))

    def init(self):

        pygame.init()
        if self.windowed:
            self.screen = pygame.display.set_mode(size=(1200, 600))
        else:
            self.screen = pygame.display.set_mode(size=(0, 0), flags=pygame.FULLSCREEN)
        pygame.display.set_caption('Speller — moving letters (0.0)')

        self.width = self.screen.get_width()
        self.height = self.screen.get_height()

        self.sec_in_msec = 1e-3

        self.t0_mean = self.data_config['t0_mean']

        self.num_samples = self.data_config['num_samples']

        self.t1_a = self.data_config['t1_a']
        self.t1_b = self.data_config['t1_b']

        self.t2_a = self.data_config['t2_a']
        self.t2_b = self.data_config['t2_b']

        self.time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # SPELLER-LINK: os.path.join instead of 'logs\...' (works on Linux and macOS too)
        self.f = open(log_path(f'logs_{self.time_now}.txt'), 'w', encoding='utf-8')
        self.data_config['date'] = self.time_now
        with open(log_path(f'settings_{self.time_now}.json'), 'w', encoding='utf-8') as f:
            json.dump(self.data_config, f, indent=4, ensure_ascii=False)

        self.quit = False
        self.clock = pygame.time.Clock()
        self.FPS = self.data_config['FPS']
        self.font = pygame.font.SysFont(name=self.data_config['font_name'],
                                        size=self.data_config['font_size'], bold=True)
        self.small_font = pygame.font.SysFont(name=self.data_config['font_name'],
                                              size=max(18, self.data_config['font_size'] // 3))
        self.background = color(self.data_config.get('background'), (255, 255, 255))
        self.letter_foreground = color(self.data_config.get('letter_color'), (0, 0, 0))
        self.cue_color = color(self.data_config.get('cue_color'), (200, 40, 40))

    def set_t1(self):

        return np.random.uniform(self.t1_a, self.t1_b)

    def set_t0(self):

        return self.t0_mean

    def set_t2(self):

        return np.random.uniform(self.t2_a, self.t2_b)

    def speed_func(self, t, t0=1, t1=0.5, t2=0.5, forward=True):

        if t0 <= 0: raise ValueError('freq must be positive')

        if t1 < 0: raise ValueError('t1 must be non negative')

        if t2 < 0: raise ValueError('t2 must be non negative')

        if t1 / self.sec_in_msec <= t <= (t1 + t0) / self.sec_in_msec:

            return math.sin((t - t1 / self.sec_in_msec) * math.pi * self.sec_in_msec / t0) * (2 * forward - 1)

        elif 0 <= t < t1 / self.sec_in_msec or (t1 + t0) / self.sec_in_msec < t <= (t1 + t0 + t2) / self.sec_in_msec:

            return 0

        else:

            raise ValueError(f't must be between 0 and {(t1 + t0 + t2) / self.sec_in_msec}')

    # ------------------------------------------------------------------ SPELLER-LINK
    def events(self):
        """Window events; returns True if SPACE was pressed."""
        pressed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.quit = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                pressed = True
        return pressed

    def wait_epoch(self):
        while not self.quit:
            epoch = self.link.wait_epoch(timeout=0.05)
            if epoch is not None or self.link.ended:
                return epoch
            self.events()
            self.screen.fill(self.background)
            msg = 'Пауза' if self.link.paused else 'Ожидание спеллера…'
            img = self.small_font.render(msg, True, self.letter_foreground)
            self.screen.blit(img, img.get_rect(center=self.screen.get_rect().center))
            pygame.display.flip()
            self.clock.tick(30)
        return None

    def hold(self, seconds, draw=None):
        end = pygame.time.get_ticks() + seconds * 1000
        while pygame.time.get_ticks() < end and not self.quit:
            self.events()
            if draw:
                draw()
            pygame.display.flip()
            self.clock.tick(self.FPS)

    def cell_center(self, params):
        j, i = params['row'], params['col']
        return i * self.w + self.w / 2, j * self.h + self.h / 2

    def draw_letter(self, params, dx=0.0, dy=0.0, col=None):
        x0, y0 = self.cell_center(params)
        letter = params['letter'] if col is None else self.font.render(params['label'], True, col)
        self.screen.blit(letter, (x0 - letter.get_width() / 2 + dx, y0 - letter.get_height() / 2 - dy))

    # --------------------------------------------------------------------------
    def postinit(self, epoch):

        # SPELLER-LINK: keyboard from the speller
        self.n_rows, self.n_cols = epoch.layout
        self.w, self.h = self.width // self.n_cols, self.height // self.n_rows

        self.cells = {}

        self.amplitude_x_scale = self.data_config['amplitude_x_scale']

        self.amplitude_y_scale = self.data_config['amplitude_y_scale']

        for id, char in enumerate(epoch.symbols):
            label = '_' if char == ' ' else char
            letter = self.font.render(label, True, self.letter_foreground)
            self.cells[char] = {
                'id': id,
                'row': id // self.n_cols,
                'col': id % self.n_cols,
                'label': label,
                'letter': letter,
                'amplitude_x': (self.w / 2) * self.amplitude_x_scale,
                'amplitude_y': (self.h / 2) * self.amplitude_y_scale,
                't0': self.set_t0(),
                't1': self.set_t1(),
                't2': self.set_t2(),
                'dir': random.randint(0, 1),
                'start_t': 0,
                'start_marker': False,
                'end_marker': False,
                'n_samp': 0
            }

        self.decoder = AsyncKeyboardDecoder(epoch.symbols, hit=0.8, false_alarm=0.2) if self.keyboard else None

    def perform(self, epoch):

        cue = epoch.cue
        self.f.write(f'epoch {epoch.id} cue {cue}\n')

        # rest before the epoch, then the cue (only in copy-spelling / calibration sessions)
        self.hold(self.data_config.get('rest_before_ms', 0) / 1000)
        if cue is not None and cue in self.cells:
            self.hold(self.data_config.get('cue_ms', 1000) / 1000,
                      lambda: (self.screen.fill(self.background),
                               self.draw_letter(self.cells[cue], col=self.cue_color)))

        if self.data_config.get('start_by_space'):
            while not self.quit and not self.events():
                self.screen.fill(self.background)
                pygame.display.flip()
                self.clock.tick(self.FPS)

        t0 = pygame.time.get_ticks()
        while not self.quit:

            self.clock.tick(self.FPS)
            pressed = self.events()
            if self.decoder and pressed:
                self.decoder.add_press(self.link_now())

            self.screen.fill(self.background)
            started = []

            t = pygame.time.get_ticks() - t0

            for char, params in self.cells.items():

                try:
                    speed = self.speed_func(t - params['start_t'], params['t0'], params['t1'], params['t2'], params['dir'])

                    if t >= params['start_t'] + params['t1'] / self.sec_in_msec and not params['start_marker']:

                        self.f.write(f'letter {char} start {t}\n')

                        params['start_marker'] = True
                        started.append(char)        # SPELLER-LINK: reported after the flip

                    if t >= params['start_t'] + (params['t1'] + params['t0']) / self.sec_in_msec and not params['end_marker']:

                        params['end_marker'] = True

                except ValueError:

                    params['start_t'] = t
                    params['t1'] = self.set_t1()
                    params['t2'] = self.set_t2()
                    params['dir'] = random.randint(0, 1)

                    params['start_marker'] = False
                    params['end_marker'] = False

                    params['n_samp'] += 1

                    speed = self.speed_func(t - params['start_t'], params['t0'], params['t1'], params['t2'], params['dir'])

                dx = params['amplitude_x'] * speed if self.data_config['x_move'] else 0
                dy = params['amplitude_y'] * speed if self.data_config['y_move'] else 0

                self.draw_letter(params, dx, dy)

            pygame.display.flip()

            # SPELLER-LINK: one marker per letter that started moving in this frame,
            # with the LSL time right after the flip.
            if started:
                onset = self.link_now()
                for char in started:
                    self.link.flash([char], timestamp=onset)
                    if self.decoder:
                        self.decoder.add_onset([char], onset)
            if self.decoder and self.decoder.update(self.link_now()):
                self.link.send_likelihood(self.decoder.probabilities())

            # SPELLER-LINK: end when every letter made num_samples movements (the module
            # does not know the target) or when the speller is confident enough.
            # IMPROV[Q-43]
            if min(p['n_samp'] for p in self.cells.values()) >= self.num_samples or self.link.should_stop():
                break

        self.link.done()
        if self.decoder:
            self.decoder.update(self.link_now() + 10)
            self.link.send_likelihood(self.decoder.probabilities(), final=True)
        res = self.link.wait_result(timeout=5)
        chosen = res.get('symbol') if res else None
        self.f.write(f'epoch {epoch.id} result {chosen}\n')
        if chosen in self.cells:
            self.hold(self.data_config.get('rest_after_ms', 1000) / 1000,
                      lambda: (self.screen.fill(self.background),
                               self.draw_letter(self.cells[chosen], col=self.cue_color)))

    def link_now(self):
        from speller_link import now
        return now()

    def end_exp(self):

        self.f.close()
        pygame.quit()
