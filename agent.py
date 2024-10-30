import os
import sys

from src.game import WebGame
from src.train_callback import TrainAndLoggingCallback
from src.utils import CHECKPOINT_DIR, LOG_DIR
from src.optuna import fine_tune
from time import sleep

from stable_baselines3 import DQN

env = WebGame()

command = sys.argv[1] if len(sys.argv) > 1 else ""
model_to_continue_training = sys.argv[2] if len(sys.argv) > 2 else None


def start_fine_tunning():
    print("Starting model fine tunning.")
    fine_tune.tune(env)


def start_training():
    print("Starting model training.")
    callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)

    model = None

    if model_to_continue_training != None:
        model = DQN.load(os.path.join(CHECKPOINT_DIR, model_to_continue_training))
        model.set_env(env)
    else:
        model = DQN(
            "CnnPolicy",
            env,
            tensorboard_log=LOG_DIR,
            verbose=1,
            learning_starts=1000,
            buffer_size=10000,
            batch_size=128,
            gamma=0.972270290523627,
            target_update_interval=500,
            learning_rate=1.0553082748309351e-05,
        )

    model.learn(total_timesteps=500000, callback=callback)


def run_model():
    print("Starting final model.")
    model = DQN.load(os.path.join("train", "best_model_100000"))

    for _ in iter(int, 1):
        obs, _ = env.reset()
        done = False

        while not done:
            done = model.predict(obs)
        sleep(1)


if command == "--tune":
    start_fine_tunning()
elif command == "--train":
    start_training()
else:
    run_model()
