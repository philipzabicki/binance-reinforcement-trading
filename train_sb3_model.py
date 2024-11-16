from datetime import datetime as dt

import torch as th
import torch.nn as nn
from gym import Space
from numpy import inf
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import cuda
from torch.nn import functional as F

from definitions import TENSORBOARD_DIR, MODELS_DIR
from environments.spot_rl_env import DiscreteSpotTakerRL
from utils.data_collector import get_precalculated_dataset_by_filename
from utils.feature_functions import *


class CustomCNNBiLSTMAttentionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Space, features_dim: int = 256):
        super(CustomCNNBiLSTMAttentionFeatureExtractor, self).__init__(observation_space, features_dim)
        n_time_steps, n_features = observation_space.shape  # (72, 139)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=384, kernel_size=2, padding='same'),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Conv1d(384, 256, kernel_size=4, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(256, 128, kernel_size=8, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.15)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
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
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)  # (B, 140, 1)
        # print(f'attention_weights: {attention_weights.shape}')
        lstm_out = lstm_out * attention_weights  # (B, 140, 256)
        # print(f'lstm_out * attention_weights {lstm_out.shape}')
        lstm_out = lstm_out.sum(dim=1)  # (B, 256)
        # print(f'lstm_out.sum(dim=1) {lstm_out.shape}')
        return lstm_out


def make_env(df, **env_kwargs):
    def _init():
        return DiscreteSpotTakerRL(df=df, **env_kwargs)

    return _init


MODELING_DATASET_FILENAME = 'BTCUSDT5m_spot_modeling_v2.csv'
DATASET_SPLIT_DATE = "2024-06-01"
NUM_ENVS = 6
ENV_KWARGS = {
    "lookback_size": 72,  # 6h obs history (5m itv)
    "max_steps": 8_640,  # 30D of trading (5m itv)
    "exclude_cols_left": 5,
    "init_balance": 1_000,
    "no_action_finish": inf,
    'steps_passive_penalty': 'auto',
    "fee": 0.00075,
    "coin_step": 0.00001,
    "visualize": False,
    "render_range": 80,
    "verbose": True,
    "report_to_file": False,
}
MODEL_PARAMETERS = {
    'batch_size': 336,  # 168, 252, 336, 504
    'n_steps': 1_008,
    'total_timesteps': 500_000,
}

if __name__ == "__main__":
    print(f'CUDA available: {cuda.is_available()}')
    experiment_name = f"{MODELING_DATASET_FILENAME}lb{ENV_KWARGS['lookback_size']}_ms{ENV_KWARGS['max_steps']}_tt{MODEL_PARAMETERS['total_timesteps']}_PPO"
    tensorboard_log_dir = f"{TENSORBOARD_DIR}/{experiment_name}"

    df_train, _ = get_precalculated_dataset_by_filename(MODELING_DATASET_FILENAME, DATASET_SPLIT_DATE)
    del _  # save memory
    print(f'Loaded train dataset shape: {df_train.shape}')

    df_train = add_scaled_candle_patterns_indicators(df_train)
    df_train = add_scaled_ultosc_rsi_mfi_up_to_n(df_train, 100, 5)
    print(f'Post feature imputation train dataset shape: {df_train.shape}')

    envs = SubprocVecEnv([make_env(df_train, **ENV_KWARGS) for _ in range(NUM_ENVS)])
    policy_kwargs = dict(
        features_extractor_class=CustomCNNBiLSTMAttentionFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
        activation_fn=nn.ReLU,
        share_features_extractor=True
    )
    model = PPO(
        "CnnPolicy",
        envs,
        policy_kwargs=policy_kwargs,
        n_steps=MODEL_PARAMETERS['n_steps'] // NUM_ENVS,
        batch_size=MODEL_PARAMETERS['batch_size'],
        # n_epochs=3,
        learning_rate=1e-4,
        # clip_range=0.1,
        # ent_coef=0.01,
        # vf_coef=0.9,
        # gamma=0.99,
        # max_grad_norm=1.0,
        # target_kl=1.0,
        tensorboard_log=tensorboard_log_dir,
        verbose=1,
        stats_window_size=5,
        device="cuda",
    )

    model.learn(total_timesteps=MODEL_PARAMETERS['total_timesteps'],
                progress_bar=True,
                log_interval=20)
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
    #     approx_kl
    #         Przybliżona różnica Kullbacka-Leiblera między starą a nową polityką, używana do monitorowania zmian w polityce.
    #     clip_fraction
    #         Frakcja próbek, które przekroczyły zakres klipowania, co wskazuje na to, jak często klipowanie jest aktywne.
    #     clip_range
    #         Zakres klipowania dla funkcji polityki, który ogranicza zmiany w polityce, aby zapobiec zbyt dużym aktualizacjom.
    #     entropy_loss
    #         Strata entropii, miara niepewności polityki, która promuje eksplorację przez agentów.
    #     explained_variance
    #         Wskaźnik wyjaśnionej wariancji wartości, pokazujący, jak dobrze przewidywane wartości odzwierciedlają rzeczywiste nagrody.
    #     learning_rate
    #         Tempo uczenia, określające jak duże kroki są podejmowane podczas aktualizacji wag modelu.
    #     loss
    #         Całkowita strata modelu, będąca sumą wszystkich komponentów strat używanych do optymalizacji modelu.
    #     n_updates
    #         Liczba aktualizacji wykonanych przez algorytm treningowy.
    #     policy_gradient_loss
    #         Strata gradientu polityki, część straty związana bezpośrednio z aktualizacją polityki.
    #     std
    #         Odchylenie standardowe, często związane z polityką stochastyczną, określające stopień eksploracji.
    #     value_loss
    #         Strata funkcji wartości, mierząca błąd w przewidywaniu wartości stanu przez model.
    # -----------------------------------------

    _date = str(dt.today()).replace(":", "-")[:-7]
    model_full_path = f"{MODELS_DIR}/{_date}"
    print(f'Saving model to: {model_full_path}')
    model.save(model_full_path)
    del model
    model = PPO.load(model_full_path)

    _, df_test = get_precalculated_dataset_by_filename(MODELING_DATASET_FILENAME, DATASET_SPLIT_DATE)
    print(f'Loaded test dataset shape: {df_test.shape}')
    df_test = add_scaled_candle_patterns_indicators(df_test)
    df_test = add_scaled_ultosc_rsi_mfi_up_to_n(df_test, 100, 5)

    ENV_KWARGS['report_to_file'] = True
    val_env = DiscreteSpotTakerRL(df=df_test, **ENV_KWARGS)

    print(f'### VALIDATION STARTED ###', end='\n')
    for _ in range(10):
        obs = val_env.reset()
        terminated = False
        while not terminated:
            action, _states = model.predict(obs)
            obs, reward, terminated, info = val_env.step(action)

    print(f'### VISUALIZATION TEST STARTED ###', end='\n')
    ENV_KWARGS['visualize'] = True
    visualize_env = DiscreteSpotTakerRL(df=df_test, **ENV_KWARGS)
    obs = visualize_env.reset()
    terminated = False
    while not terminated:
        action, _states = model.predict(obs)
        obs, reward, terminated, info = visualize_env.step(action)
