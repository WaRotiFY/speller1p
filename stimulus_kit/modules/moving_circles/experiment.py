"""Кружки по эллипсам: стимул — начало движения кружка буквы (контракт speller-link@1).

Места, где модуль общается со спеллером, помечены «SPELLER-LINK», как в модуле движения букв:
клавиатура из settings.json, эпохи по команде спеллера, link.flash([буква]) сразу
после flip кадра, в котором кружок начал движение, конец эпохи — когда каждый кружок
сделал num_samples движений или пришёл stop, подсказка — из sentence в settings.json.

Расположение эллипсов рассчитано на 11 столбцов; при другой раскладке кружки
двигаются, но геометрия хуже.
"""
import colorsys
import json
import math
from datetime import datetime

import numpy as np
import pygame

from module_tools import AsyncKeyboardDecoder, color, cue_for, log_path, place_window


class Experiment:

    def __init__(self, data_config, link, keyboard=False, windowed=False):

        self.data_config = data_config
        self.link = link                          # SPELLER-LINK
        self.keyboard = keyboard
        self.windowed = windowed
        self.circle_radius = self.data_config['circle_radius']

    def fit(self):

        self.preinit()
        self.init()

        while True:                               # SPELLER-LINK: epochs from the speller
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
            self.screen = pygame.display.set_mode(size=(1320, 600))
        else:
            self.screen = pygame.display.set_mode(size=(0, 0), flags=pygame.FULLSCREEN)
        pygame.display.set_caption('Speller — circles (0.4)')

        self.width = self.screen.get_width()
        self.height = self.screen.get_height()

        self.sec_in_msec = 1e-3

        self.acceleration = self.data_config['acceleration']

        self.grid_color = color(self.data_config['grid_color'], (100, 100, 100))

        self.num_samples = self.data_config['num_samples']

        self.t0_mean = self.data_config['t0_mean'] / self.acceleration

        self.t1_a = self.data_config['t1_a'] / self.acceleration
        self.t1_b = self.data_config['t1_b'] / self.acceleration

        self.t2_a = self.data_config['t2_a'] / self.acceleration
        self.t2_b = self.data_config['t2_b'] / self.acceleration

        self.time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.f = open(log_path(f'logs_{self.time_now}.txt'), 'w', encoding='utf-8')
        self.data_config['date'] = self.time_now
        with open(log_path(f'settings_{self.time_now}.json'), 'w', encoding='utf-8') as f:
            json.dump(self.data_config, f, indent=4, ensure_ascii=False)

        self.delta_circle = self.data_config['delta_circle']

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

    def speed_func(self, t, t0=1, t1=0.5, t2=0.5):

        if t0 <= 0: raise ValueError('freq must be positive')

        if t1 < 0: raise ValueError('t1 must be non negative')

        if t2 < 0: raise ValueError('t2 must be non negative')

        if t1 / self.sec_in_msec <= t <= (t1 + t0) / self.sec_in_msec:

            return (t - t1 / self.sec_in_msec) * math.tau * self.sec_in_msec / t0

        elif 0 <= t < t1 / self.sec_in_msec or (t1 + t0) / self.sec_in_msec < t <= (t1 + t0 + t2) / self.sec_in_msec:

            return 0

        else:

            raise ValueError(f't must be between 0 and {(t1 + t0 + t2) / self.sec_in_msec}')

    def apogee_var2(self, i, j):
        ran = 1
        x = (i - 1 + 3 * ran) * self.w + (2 * ran - 1) * self.w / 2
        x += (1 - 2 * ran) * 2 * self.circle_radius * self.delta_circle
        if i in [0, 2, 4, 7, 9]:
            x = (i + 2) * self.w - self.circle_radius * self.delta_circle
        elif i in [1, 3, 6, 8, 10]:
            x = (i - 1) * self.w + self.circle_radius * self.delta_circle
        y = j * self.h + self.circle_radius * self.delta_circle
        return x, y

    def st_ellipse(self, i, j):

        x = i * self.w + self.w / 2
        if i in [0, 2, 4, 7, 9]:
            x = i * self.w + self.circle_radius * self.delta_circle
        elif i in [1, 3, 6, 8, 10]:
            x = (i + 1) * self.w - self.circle_radius * self.delta_circle
        y = (j + 1) * self.h - self.circle_radius * self.delta_circle
        return x, y

    def random_color1(self):

        h = np.random.random()
        s = np.random.uniform(0.6, 1.0)
        l = np.random.uniform(0.4, 0.6)

        return (np.array(colorsys.hls_to_rgb(h, l, s)) * 255).astype(int)

    # ------------------------------------------------------------------ SPELLER-LINK
    def events(self):
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

    def draw_static(self, params, highlight=None):
        i, j = params['col'], params['row']
        pygame.draw.rect(self.screen, self.grid_color, (i * self.w, j * self.h, self.w, self.h), 1)
        letter = params['letter'] if highlight is None else self.font.render(params['label'], True, highlight)
        x0, y0 = i * self.w + self.w / 2, j * self.h + self.h / 2
        self.screen.blit(letter, (x0 - letter.get_width() / 2, y0 - letter.get_height() / 2))
        xd, yd = params['start_ellipse']
        pygame.draw.circle(self.screen, params['circle_color'], (xd, yd), self.circle_radius, 0)

    def link_now(self):
        from speller_link import now
        return now()

    # --------------------------------------------------------------------------
    def postinit(self, epoch):

        self.n_rows, self.n_cols = epoch.layout             # SPELLER-LINK
        self.w, self.h = self.width // self.n_cols, self.height // self.n_rows

        self.cells = {}

        for id, char in enumerate(epoch.symbols):
            label = '_' if char == ' ' else char
            letter = self.font.render(label, True, self.letter_foreground)

            j, i = id // self.n_cols, id % self.n_cols

            self.cells[char] = {
                'id': id,
                'row': j,
                'col': i,
                'label': label,
                'letter': letter,
                't0': self.set_t0(),
                't1': self.set_t1(),
                't2': self.set_t2(),
                'start_t': 0,
                'start_marker': False,
                'end_marker': False,
                'circle_color': self.random_color1(),
                'b_ellipse': 10,
                'p_ellipse': self.apogee_var2(i, j),
                'start_ellipse': self.st_ellipse(i, j),
                'n_samp': 0
            }

        self.decoder = AsyncKeyboardDecoder(epoch.symbols, hit=0.8, false_alarm=0.2) if self.keyboard else None

    def perform(self, epoch):

        cue = cue_for(self.data_config, epoch.text, epoch.symbols)   # IMPROV[Q-42]
        self.f.write(f'epoch {epoch.id} cue {cue}\n')

        def static(highlight_char=None, col=None):
            self.screen.fill(self.background)
            for char, params in self.cells.items():
                if highlight_char is None or char == highlight_char:
                    self.draw_static(params, col if char == highlight_char else None)

        self.hold(self.data_config.get('rest_before_ms', 0) / 1000, lambda: static())
        if cue is not None and cue in self.cells:
            self.hold(self.data_config.get('cue_ms', 1500) / 1000, lambda: static(cue, self.cue_color))

        if self.data_config.get('start_by_space'):
            while not self.quit and not self.events():
                static()
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

                j, i = params['row'], params['col']

                try:
                    speed = self.speed_func(t - params['start_t'], params['t0'], params['t1'], params['t2'])

                    if t >= params['start_t'] + params['t1'] / self.sec_in_msec and not params['start_marker']:

                        self.f.write(f'letter {char} start {t}\n')

                        params['start_marker'] = True
                        started.append(char)            # SPELLER-LINK

                    if t >= params['start_t'] + (params['t1'] + params['t0']) / self.sec_in_msec and not params['end_marker']:

                        params['end_marker'] = True

                except ValueError:

                    params['start_t'] = t
                    params['t1'] = self.set_t1()
                    params['t2'] = self.set_t2()

                    params['start_marker'] = False
                    params['end_marker'] = False

                    params['n_samp'] += 1

                    speed = self.speed_func(t - params['start_t'], params['t0'], params['t1'], params['t2'])

                x0, y0 = i * self.w + self.w / 2, j * self.h + self.h / 2

                xd, yd = params['start_ellipse']

                x1, y1 = params['p_ellipse']

                xc, yc = (xd + x1) / 2, (yd + y1) / 2

                ax, ay = (xd - x1) / 2, (yd - y1) / 2

                a_norm = np.sqrt(ax ** 2 + ay ** 2) or 1.0

                b = params['b_ellipse']

                bx, by = -b * ay / a_norm, b * ax / a_norm

                dx = xc + ax * np.cos(speed) + bx * np.sin(speed)
                dy = yc + ay * np.cos(speed) + by * np.sin(speed)

                pygame.draw.rect(self.screen, self.grid_color, (i * self.w, j * self.h, self.w, self.h), 1)

                self.screen.blit(params['letter'], (x0 - params['letter'].get_width() / 2, y0 - params['letter'].get_height() / 2))

                pygame.draw.circle(self.screen, params['circle_color'], (dx, dy), self.circle_radius, 0)

            pygame.display.flip()

            if started:                               # SPELLER-LINK: markers right after the flip
                onset = self.link_now()
                for char in started:
                    self.link.flash([char], timestamp=onset)
                    if self.decoder:
                        self.decoder.add_onset([char], onset)
            if self.decoder and self.decoder.update(self.link_now()):
                self.link.send_likelihood(self.decoder.probabilities())

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
            self.hold(self.data_config.get('rest_after_ms', 1000) / 1000, lambda: static(chosen, self.cue_color))

    def end_exp(self):

        self.f.close()
        pygame.quit()
