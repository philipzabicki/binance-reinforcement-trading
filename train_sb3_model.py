from datetime import datetime as dt
from os import path

import torch as th
import torch.nn as nn
from gym import Space
from numpy import inf
from stable_baselines3 import DQN  # Zmieniono z PPO na DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import cuda
from torch.nn import functional as F

from definitions import TENSORBOARD_DIR, MODELS_DIR, MODELING_DATASET_DIR
from environments.rl_spot_env import DiscreteSpotTakerRL
from utils.data_collector import get_precalculated_dataset_by_filename
from utils.feature_functions import *


class CustomCNNBiLSTMAttentionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Space, features_dim: int = 390):
        super(CustomCNNBiLSTMAttentionFeatureExtractor, self).__init__(
            observation_space, features_dim
        )
        n_time_steps, n_features = observation_space.shape  # (72, 139)
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features, out_channels=384, kernel_size=2, padding="same"
            ),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            # nn.Dropout(p=0.25),
            nn.Conv1d(384, 256, kernel_size=4, padding="same"),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Conv1d(256, 128, kernel_size=8, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
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
        x = observations  # (B, 72, 139)
        x = x.permute(0, 2, 1)  # Przekształcenie do (B, 139, 72) dla konwolucji
        # print(f'x after permute: {x.shape}')
        x = self.cnn(x)  # (B, 128, 72)
        # print(f'x after cnn: {x.shape}')
        x = x.permute(0, 2, 1)  # Przekształcenie do (B, 72, 128) dla LSTM
        # print(f'x after 2nd permute: {x.shape}')
        lstm_out, _ = self.lstm(x)  # Teraz LSTM otrzymuje (B, 72, 128)
        # print(f'x lstm output: {lstm_out.shape}')
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)  # (B, 72, 1)
        # print(f'attention_weights: {attention_weights.shape}')
        lstm_out = lstm_out * attention_weights  # (B, 72, 256)
        # print(f'lstm_out * attention_weights {lstm_out.shape}')
        lstm_out = lstm_out.sum(dim=1)  # (B, 256)
        # print(f'lstm_out.sum(dim=1) {lstm_out.shape}')
        return lstm_out


def make_env(df, **env_kwargs):
    def _init():
        return DiscreteSpotTakerRL(df=df, **env_kwargs)

    return _init


MODELING_DATASET_FILENAME = "modeling.csv"
DATASET_SPLIT_DATE = "2024-03-01"
NUM_ENVS = 6
ENV_KWARGS = {
    "lookback_size": 24,  # 6h obs history (5m itv)
    "max_steps": 8_640,  # 30D of trading (5m itv)
    "exclude_cols_left": 5,
    "init_balance": 1_000,
    "no_action_finish": inf,
    "steps_passive_penalty": "auto",
    "fee": 0.00075,
    # "fee": 0.0,
    "coin_step": 0.00001,
    "visualize": False,
    "render_range": 80,
    "verbose": True,
    "report_to_file": False,
}
MODEL_PARAMETERS = {
    "batch_size": 168,  # 168, 252, 336, 504
    "buffer_size": 100_000,  # Rozmiar bufora doświadczeń
    "learning_starts": 1_000,  # Liczba kroków przed rozpoczęciem uczenia
    "train_freq": (1, "step"),  # Częstotliwość trenowania
    "target_update_interval": 1_000,  # Częstotliwość aktualizacji docelowej sieci
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "exploration_fraction": 0.1,
    "total_timesteps": 5_000_000,
}

if __name__ == "__main__":
    print(f"CUDA available: {cuda.is_available()}")
    experiment_name = f"{MODELING_DATASET_FILENAME}lb{ENV_KWARGS['lookback_size']}_ms{ENV_KWARGS['max_steps']}_tt{MODEL_PARAMETERS['total_timesteps']}_DQN"
    tensorboard_log_dir = f"{TENSORBOARD_DIR}/{experiment_name}"

    df_train = pd.read_csv(path.join(MODELING_DATASET_DIR, MODELING_DATASET_FILENAME))
    # df_train, _ = get_precalculated_dataset_by_filename(MODELING_DATASET_FILENAME, DATASET_SPLIT_DATE)
    # del _  # save memory
    # print(f'Loaded train dataset shape: {df_train.shape}')
    #
    # df_train = add_scaled_candle_patterns_indicators(df_train)
    # df_train = add_scaled_ultosc_rsi_mfi_up_to_n(df_train, 35, 1)
    # print(f'Post feature imputation train dataset shape: {df_train.shape}')

    envs = SubprocVecEnv([make_env(df_train, **ENV_KWARGS) for _ in range(NUM_ENVS)])
    policy_kwargs = dict(
        features_extractor_class=CustomCNNBiLSTMAttentionFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 128, 64],  # DQN używa jednowarstwowej architektury
        activation_fn=nn.ReLU,
    )
    model = DQN(
        "CnnPolicy",  # Polityka CNN jest kompatybilna z DQN
        envs,
        policy_kwargs=policy_kwargs,
        # buffer_size=10,
        # buffer_size=MODEL_PARAMETERS['buffer_size'],
        # learning_starts=MODEL_PARAMETERS['learning_starts'],
        # train_freq=MODEL_PARAMETERS['train_freq'],
        # target_update_interval=MODEL_PARAMETERS['target_update_interval'],
        # batch_size=MODEL_PARAMETERS['batch_size'],
        # learning_rate=MODEL_PARAMETERS['learning_rate'],
        # gamma=MODEL_PARAMETERS['gamma'],
        # exploration_initial_eps=MODEL_PARAMETERS['exploration_initial_eps'],
        # exploration_final_eps=MODEL_PARAMETERS['exploration_final_eps'],
        # exploration_fraction=MODEL_PARAMETERS['exploration_fraction'],
        tensorboard_log=tensorboard_log_dir,
        verbose=1,
        device="cuda",
    )

    model.learn(
        total_timesteps=MODEL_PARAMETERS["total_timesteps"],
        progress_bar=True,
        log_interval=1,
    )
    # -----------------------------------------
    # time/
    #     fps
    #         Liczba klatek na sekundę, określająca ile klatek jest renderowanych lub przetwarzanych w każdej sekundzie.
    #     iterations
    #         Liczba iteracji, czyli liczba powtórzeń algorytmu treningowego lub procesu uczenia.
    #     time_elapsed
    #         Całkowity czas, który upłynął od rozpoczęcia treningu, mierzony w sekundach.
    #     total_timesteps
    #         Łączna liczba kroków czasowych wykonanych podczas treningu.
    #
    # train/
    #     ... (pozostałe metryki pozostają bez zmian)
    # -----------------------------------------

    _date = str(dt.today()).replace(":", "-")[:-7]
    model_full_path = f"{MODELS_DIR}/{_date}"
    print(f"Saving model to: {model_full_path}")
    model.save(model_full_path)
    del model
    model = DQN.load(model_full_path, env=envs)  # Ładowanie modelu DQN

    _, df_test = get_precalculated_dataset_by_filename(
        MODELING_DATASET_FILENAME, DATASET_SPLIT_DATE
    )
    print(f"Loaded test dataset shape: {df_test.shape}")
    df_test = add_scaled_candle_patterns_indicators(df_test)
    df_test = add_scaled_ultosc_rsi_mfi_up_to_n(df_test, 35, 1)

    ENV_KWARGS["report_to_file"] = True
    ENV_KWARGS["max_steps"] = 0
    val_env = DiscreteSpotTakerRL(df=df_test, **ENV_KWARGS)

    print(f"### VALIDATION STARTED ###", end="\n")
    # for _ in range(10):
    obs = val_env.reset()
    terminated = False
    while not terminated:
        action, _states = model.predict(obs)
        obs, reward, terminated, info = val_env.step(action)

    print(f"### VISUALIZATION TEST STARTED ###", end="\n")
    ENV_KWARGS["visualize"] = True
    visualize_env = DiscreteSpotTakerRL(df=df_test, **ENV_KWARGS)
    obs = visualize_env.reset()
    terminated = False
    while not terminated:
        action, _states = model.predict(obs)
        obs, reward, terminated, info = visualize_env.step(action)
