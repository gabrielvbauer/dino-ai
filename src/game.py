# environment components
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

# transformational framework
import numpy as np

# mss used for screen capture
from mss import mss

# sending commands to the game
import pydirectinput

# opencv allows frame processing
import cv2

# visualiaze captured frames
from matplotlib import pyplot as plt

# pauses
import time


class WebGame(Env):
    # setup the environment action and observation shapes
    def __init__(self):
        # subclass model
        super().__init__()

        # setup spaces
        self.observation_space = Box(
            low=0, high=255, shape=(1, 100, 150), dtype=np.uint8
        )
        self.action_space = Discrete(2)

        # define extraction paramenters for the game
        self.cap = mss()
        self.game_location = {"top": 200, "left": 120, "width": 750, "height": 500}
        self.done_location = {"top": 220, "left": 630, "width": 650, "height": 70}
        self.done_pixel_location = {"top": 254, "left": 678, "width": 1, "height": 1}
        self.action_map = {0: "space", 1: "no_op"}

        # store the time of the last check
        self.last_done_check_time = time.time()

    # what is called to do someting in the game
    def step(self, action):
        if action != 1:
            pydirectinput.press(self.action_map[action])

        # checking whether the game is done
        current_time = time.time()
        done = False
        if current_time - self.last_done_check_time > 0.5:
            done = self.get_done()
            self.last_done_check_time = current_time

        # get the next observation
        new_observation = self.get_observation()

        # reward - we get a point for every frame we're alive
        reward = 1

        info = {}

        truncated = False

        return new_observation, reward, done, truncated, info

    # visualize the game
    def render(self):
        cv2.imshow("Game", np.array(self.cap.grab(self.game_location))[:, :, :3])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.close()

    # this closes down the observation
    def close(self):
        cv2.destroyAllWindows()

    # reset the game
    def reset(self, seed=None):
        time.sleep(1)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press("space")
        obs = self.get_observation()

        return obs, {}

    # get the part of the observation of the game that we want
    def get_observation(self):
        # get screen capture of game
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3]

        # grayscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # resize
        resized = cv2.resize(gray, (150, 100), interpolation=cv2.INTER_AREA)
        # add chanels first
        channel = np.expand_dims(resized, axis=0)
        # channel = np.reshape(resized, (1, 100, 150))

        return channel

    def get_done(self):
        done = False

        done_pixel_cap = np.array(self.cap.grab(self.done_pixel_location))[:, :, :3]

        done = np.array_equal(done_pixel_cap[0][0], [83, 83, 83]) or np.array_equal(
            done_pixel_cap[0][0], [172, 172, 172]
        )

        return done
