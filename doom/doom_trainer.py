from __future__ import print_function
import logging
import numpy as np
from utils.image_preprocessing import preprocess
from vizdoom import *

logging.basicConfig(level=logging.INFO)


def create_actions(scenario):
    if scenario == 'basic':
        move_left = [1, 0, 0]
        move_right = [0, 1, 0]
        shoot = [0, 0, 1]

        return [move_left, move_right, shoot]

    if scenario == 'deadly_corridor':
        move_left = [1, 0, 0, 0, 0, 0, 0]
        move_right = [0, 1, 0, 0, 0, 0, 0]
        shoot = [0, 0, 1, 0, 0, 0, 0]
        back = [0, 0, 0, 1, 0, 0, 0]
        forward = [0, 0, 0, 0, 1, 0, 0]
        turn_left = [0, 0, 0, 0, 0, 1, 0]
        turn_right = [0, 0, 0, 0, 0, 0, 1]

        return [move_left, move_right, shoot, back, forward, turn_left, turn_right]

    if scenario == 'my_way_home':
        turn_left = [1, 0, 0, 0, 0]
        turn_right = [0, 1, 0, 0, 0]
        forward = [0, 0, 1, 0, 0]
        move_left = [0, 0, 0, 1, 0]
        move_right = [0, 0, 0, 0, 1]

        return [turn_left, turn_right, forward, move_left, move_right]

    if scenario == 'defend_the_center':
        turn_left = [1, 0, 0]
        turn_right = [0, 1, 0]
        shoot = [0, 0, 1]

        return [turn_left, turn_right, shoot]


class DoomTrainer:

    def __init__(self, params):
        self.game = DoomGame()
        self.game.load_config(params.config)

        # self.game.set_screen_format(ScreenFormat.BGR24)
        self.game.set_screen_format(ScreenFormat.GRAY8)

        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        # self.game.set_screen_resolution(ScreenResolution.RES_640X480)

        self.game.set_depth_buffer_enabled(True)
        self.game.set_labels_buffer_enabled(True)

        self.game.set_automap_buffer_enabled(True)
        self.game.set_automap_mode(AutomapMode.OBJECTS)
        self.game.set_automap_rotate(False)
        self.game.set_automap_render_textures(False)

        self.game.set_render_hud(True)
        self.game.set_render_minimal_hud(False)

        if params.model == 'human':
            self.game.set_window_visible(True)
            self.game.set_mode(Mode.SPECTATOR)
        else:
            self.actions = create_actions(params.scenario)

    def start_game(self):
        self.game.init()

    def stop_game(self):
        self.game.close()

    def toogle_visible(self, visible=True):
        self.game.set_window_visible(visible)

    def set_mode(self, mode=Mode.ASYNC_PLAYER):
        self.set_mode(mode)

    def set_seed(self, seed):
        self.game.set_seed(seed)

    def new_episode(self):
        self.game.new_episode()

    def get_screen(self, width=320, height=240, gray=True):
        # screen_buffer = self.game.get_state().screen_buffer
        # img = scale(screen_buffer, width, height, gray)
        # img = img.astype(np.float32)
        screen_buffer = self.game.get_state().screen_buffer
        return preprocess(screen_buffer)

    def make_action(self, action):
        reward = self.game.make_action(self.actions[action])
        done = self.game.is_episode_finished()

        return reward, done

    def num_actions(self):
        return len(self.actions)
