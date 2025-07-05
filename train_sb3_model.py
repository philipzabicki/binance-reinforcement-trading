from os import path, environ

# 0 = pokaż wszystko, 1 = tylko WARNING+ERROR, 2 = tylko ERROR, 3 = cisza
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from datetime import datetime as dt
import torch as th
import torch.nn as nn
import pandas as pd
import numpy as np
from gymnasium import Space
from numpy import inf
from stable_baselines3.common.logger import configure
from tqdm import trange
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from torch import cuda
from torch.nn import functional as F

from definitions import TENSORBOARD_DIR, MODELS_DIR, MODELING_DATASET_DIR
from environments.rl_spot_env import DiscreteSpotTakerRL


# class TensorboardCallback(BaseCallback):
#     def _on_step(self) -> bool:
#         info = self.locals["infos"][0]
#         for tag in ("balance", "pnl", "step_reward"):
#             if tag in info:
#                 self.logger.record(f"env/{tag}", info[tag])
#         return True

class TensorboardCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        # Średnia po wszystkich envach i po krokach w rollout-cie
        for tag in ("balance", "return", "pnl", "step_reward", "win_rate", "profit_loss_ratio", "mean_pnl", "std_pnl",
                    "mean_profit", "mean_loss", "profit_std", "loss_std", "in_gain_ratio"):
            vals = [info.get(tag) for info in infos if tag in info]
            if vals:
                # zapisz średnią w ramach tego rolloutu
                self.logger.record_mean(f"env/{tag}", float(np.mean(vals)))
        return True


class CustomCNNBiLSTMAttentionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Space, features_dim: int = 390):
        super(CustomCNNBiLSTMAttentionFeatureExtractor, self).__init__(
            observation_space, features_dim
        )
        n_time_steps, n_features = observation_space.shape  # (72, 139)
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features, out_channels=384, kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            # nn.Dropout(p=0.25),
            nn.Conv1d(384, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Conv1d(256, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(128 * 2, 1)  # *2 ze względu na dwukierunkowe LSTM

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations
        x = x.permute(0, 2, 1)
        # print(f'x after permute: {x.shape}')
        x = self.cnn(x)
        # print(f'x after cnn: {x.shape}')
        x = x.permute(0, 2, 1)
        # print(f'x after 2nd permute: {x.shape}')
        lstm_out, _ = self.lstm(x)
        # print(f'x lstm output: {lstm_out.shape}')
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        # print(f'attention_weights: {attention_weights.shape}')
        lstm_out = lstm_out * attention_weights
        # print(f'lstm_out * attention_weights {lstm_out.shape}')
        lstm_out = lstm_out.sum(dim=1)
        # print(f'lstm_out.sum(dim=1) {lstm_out.shape}')
        return lstm_out


def make_env(df, **env_kwargs):
    def _init():
        return DiscreteSpotTakerRL(df=df, **env_kwargs)

    return _init


TRAIN_DATASET_FILENAME = "train_modeling.csv"
TEST_DATASET_FILENAME = "test_modeling.csv"
# DATASET_SPLIT_DATE = "2024-03-01"
NUM_ENVS = 16
TOTAL_ENV_STEPS = 100_000 * NUM_ENVS
# TOTAL_ENV_STEPS = 200_000
ENV_KWARGS = {
    "lookback_size": 128,
    "max_steps": 720,
    "exclude_cols_left": 5,
    "init_balance": 1_000,
    "no_action_finish": inf,
    "steps_passive_penalty": 6,
    "fee": 0.00075,
    # "fee": 0.0,
    "coin_step": 0.00001,
    "visualize": False,
    "render_range": 80,
    "verbose": True,
    "report_to_file": False,
}
MODEL_PARAMETERS = {
    # Maximum number of transitions kept in memory
    "buffer_size": 100_000,
    # Minibatch size drawn from the buffer for a single gradient update
    "batch_size": 128,
    # Number of environment steps collected before learning starts (warm-up)
    "learning_starts": 720 * NUM_ENVS,  # ≈ 60k
    # How often to sample the buffer; tuple = (frequency, "step"/"episode")
    # "train_freq": (4, "step"),
    # Gradient updates executed after every train_freq; −1 == “as many as env steps”
    # "gradient_steps": 1,
    # Hard update of the target net every N env steps
    "target_update_interval": 180,
    # Soft-update coefficient; 1.0 means pure hard updates
    # "tau": .9,
    # Adam step size (can be a callable schedule)
    "learning_rate": 0.00005,
    # Discount factor for future rewards
    "gamma": 0.97,
    # Initial probability of random action
    "exploration_initial_eps": 0.9,
    # Final probability reached after exploration_fraction * total_timesteps
    "exploration_final_eps": 0.01,
    # Portion of training time devoted to linear ε decay
    "exploration_fraction": 0.5,
    "stats_window_size": 3,
}
N_EVAL_EPISODES = 100
VAL_MAX_STEPS = ENV_KWARGS["max_steps"]

if __name__ == "__main__":
    print(f"CUDA available: {cuda.is_available()}")
    experiment_name = f"gamma{MODEL_PARAMETERS['gamma']}_lb{ENV_KWARGS['lookback_size']}_batch{MODEL_PARAMETERS['batch_size']}_ms{ENV_KWARGS['max_steps']}_tt{TOTAL_ENV_STEPS}_DQN"
    tensorboard_log_dir = f"{TENSORBOARD_DIR}/{experiment_name}"

    df_train = pd.read_csv(
        path.join(MODELING_DATASET_DIR, TRAIN_DATASET_FILENAME))
    float_cols = [c for c in df_train.columns if df_train[c].dtype == "float64"]
    int_cols = [c for c in df_train.columns if df_train[c].dtype == "int64"]
    dtype_map = {**{c: "float32" for c in float_cols},
                 **{c: "int32" for c in int_cols}}
    df_train = pd.read_csv(
        path.join(MODELING_DATASET_DIR, TRAIN_DATASET_FILENAME),
        dtype=dtype_map)
    print("Shape:", df_train.shape)
    df_train.info(memory_usage='deep')
    total_mem = df_train.memory_usage(deep=True).sum()
    print(f"\nTotal memory: {total_mem / (1024 ** 2):.2f} MB")

    envs = SubprocVecEnv([make_env(df_train, **ENV_KWARGS) for _ in range(NUM_ENVS)])
    envs = VecMonitor(envs)

    policy_kwargs = dict(
        features_extractor_class=CustomCNNBiLSTMAttentionFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[512, 128, 32],
        activation_fn=nn.ReLU,
        # dueling=True
    )
    model = DQN(
        "MlpPolicy",
        envs,
        policy_kwargs=policy_kwargs,
        **MODEL_PARAMETERS,
        tensorboard_log=tensorboard_log_dir,
        verbose=1,
        device="cuda",
    )

    model.learn(
        callback=TensorboardCallback(),
        total_timesteps=TOTAL_ENV_STEPS,
        progress_bar=True,
        log_interval=1,
    )

    _date = str(dt.today()).replace(":", "-")[:-7]
    model_full_path = f"{MODELS_DIR}/{_date}"
    print(f"Saving model to: {model_full_path}")
    model.save(model_full_path)
    del model
    model = DQN.load(model_full_path, env=envs)
    eval_tb_dir = f"{TENSORBOARD_DIR}/{experiment_name}/eval"
    new_logger = configure(eval_tb_dir,
                           ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    df_test = pd.read_csv(path.join(MODELING_DATASET_DIR, TEST_DATASET_FILENAME))

    val_kwargs = {**ENV_KWARGS,
                  "report_to_file": True,
                  "visualize": False,  # ew. zostaw True tylko dla ostatniego epizodu
                  "max_steps": VAL_MAX_STEPS}
    val_env = DiscreteSpotTakerRL(df=df_test, **val_kwargs)

    print(f"### VALIDATION {N_EVAL_EPISODES} RND EP {VAL_MAX_STEPS} STEPS ###", end="\n")
    episode_stats = []  # <-- tu wyląduje jeden dict na epizod

    for ep in trange(N_EVAL_EPISODES, desc="Validation"):
        obs, info = val_env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = val_env.step(action)

        # zapisz finalne wartości z info; użyj NaN jeśli czegoś nie ma
        episode_stats.append(
            {tag: info.get(tag, np.nan) for tag in
             ("gain", "PnL_means_ratio", "PnL_trades_ratio", "hold_ratio", "PNL_mean")}
        )

    df_stats = pd.DataFrame(episode_stats)

    # Jeśli chcesz pełne describe:
    print(df_stats.describe().T)  # count, mean, std, min, 25%, 50%, 75%, max

    # Albo tylko wybrane miary:
    for col in df_stats.columns:
        series = df_stats[col].dropna()
        if series.size:
            print(f"{col:>20}: mean={series.mean():9.4f}  std={series.std():9.4f}  "
                  f"min={series.min():9.4f}  max={series.max():9.4f}")

    for col, vals in df_stats.items():
        if vals.notna().any():
            model.logger.record(f"eval/{col}_mean", float(vals.mean()))
            model.logger.record(f"eval/{col}_std", float(vals.std()))
    model.logger.dump(model.num_timesteps)

    print(f"### VALIDATION EP FULL STEPS ###", end="\n")
    val_kwargs['max_steps'] = 0
    full_val_env = DiscreteSpotTakerRL(df=df_test, **val_kwargs)
    obs, info = full_val_env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = full_val_env.step(action)

    print(f"### VISUALIZATION TEST STARTED ###", end="\n")
    ENV_KWARGS["visualize"] = True
    ENV_KWARGS["max_steps"] = 0
    visualize_env = DiscreteSpotTakerRL(df=df_test, **ENV_KWARGS)
    obs, info = visualize_env.reset()
    terminated = False
    while not terminated:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = visualize_env.step(action)
