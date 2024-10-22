# mss used for screen capture
from mss import mss

# sending commands to the game
import pydirectinput

# opencv allows frame processing
import cv2

# transformational framework
import numpy as np

# OCR for game over extraction
import pytesseract

# visualiaze captured frames
from matplotlib import pyplot as plt

# pauses
import time

# environment components
from gym import Env
from gym.spaces import Box, Discrete

pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"
# change to your tesseract path


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

        return new_observation, reward, done, info

    # visualize the game
    def render(self):
        cv2.imshow("Game", np.array(self.cap.grab(self.game_location))[:, :, :3])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.close()

    # this closes down the observation
    def close(self):
        cv2.destroyAllWindows()

    # reset the game
    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press("space")
        return self.get_observation()

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


env = WebGame()
eval_env = WebGame()

# import os for file path management
import os

# import  base callback for saving models
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

# check environment
from stable_baselines3.common import env_checker


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(
                self.save_path, "best_model_{}".format(self.n_calls)
            )
            self.model.save(model_path)

        return True


CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"
BEST_MODEL_SAVE = "./best-model/"
BEST_MODEL_LOGS = "./best-model-logs/"

callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_MODEL_SAVE,
    log_path=BEST_MODEL_LOGS,
    eval_freq=1000,
)

# import the DQN algorithm
from stable_baselines3 import DQN


def optimize_dqn(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    buffer_size = trial.suggest_categorical(
        "buffer_size", [10000, 20000, 50000, 100000]
    )
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma = trial.suggest_uniform("gamma", 0.9, 0.999)
    target_update_interval = trial.suggest_categorical(
        "target_update_interval", [100, 250, 500]
    )

    # Criar o modelo DQN com os parâmetros do trial
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        target_update_interval=target_update_interval,
        learning_starts=1000,
        verbose=0,
    )

    # Treinar o modelo com um número limitado de timesteps para cada trial
    model.learn(total_timesteps=25000, callback=eval_callback)

    # Avaliar o modelo após o treinamento
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)

    return mean_reward


import optuna

# Criar o estudo para otimização
study = optuna.create_study(direction="maximize")
study.optimize(optimize_dqn, n_trials=20)

# Após a otimização, mostrar os melhores hiperparâmetros
print("Best hyperparameters: ", study.best_params)


# create the DQN model
# model = DQN(
#     "CnnPolicy",
#     env,
#     tensorboard_log=LOG_DIR,
#     verbose=1,
#     buffer_size=50000,
#     learning_starts=1000,
#     batch_size=32,
#     learning_rate=1e-4,
# )

# model.learn(total_timesteps=500000, callback=callback)
