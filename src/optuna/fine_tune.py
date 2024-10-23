import optuna
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from src.game import WebGame
from src.utils import BEST_MODEL_LOGS, BEST_MODEL_SAVE


def optimize_dqn(trial, env: WebGame, eval_env: WebGame, eval_callback):
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    buffer_size = trial.suggest_categorical(
        "buffer_size", [5000, 10000, 20000, 50000, 80000]
    )
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
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
        learning_starts=500,
        verbose=1,
    )

    # Treinar o modelo com um número limitado de timesteps para cada trial
    model.learn(total_timesteps=1000, callback=eval_callback)

    # Avaliar o modelo após o treinamento
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)

    return mean_reward


def tune(env: WebGame):
    eval_env = WebGame()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_SAVE,
        log_path=BEST_MODEL_LOGS,
        eval_freq=500,
    )

    # Criar o estudo para otimização
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optimize_dqn(trial, env, eval_env, eval_callback), n_trials=10
    )

    print("Best hyperparameters: ", study.best_params)
