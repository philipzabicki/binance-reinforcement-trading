from datetime import datetime as dt

import pandas as pd
import talib
import torch as th
import torch.nn as nn
from numpy import inf
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import cuda

from definitions import TENSORBOARD_DIR, MODELS_DIR
from environments.spot_rl_env import SpotTakerRL
from utils.data_collector import by_BinanceVision


class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim)
        n_features = observation_space.shape[1]  # Number of features per time step
        time_steps = observation_space.shape[0]  # Number of time steps (lookback_size)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            # Reshape to (batch_size, features, time_steps)
            sample_input = sample_input.permute(0, 2, 1)
            cnn_output = self.cnn(sample_input)
            n_flatten = cnn_output.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # observations shape: (batch_size, time_steps, features)
        # Rearrange to (batch_size, features, time_steps)
        observations = observations.permute(0, 2, 1)
        cnn_output = self.cnn(observations)
        return self.linear(cnn_output)


def add_candle_patterns(df):
    candle_names = talib.get_function_groups()['Pattern Recognition']
    for pattern_name in candle_names:
        pattern_function = getattr(talib, pattern_name)
        df[pattern_name] = pattern_function(df['Open'], df['High'], df['Low'], df['Close'])
    return df


def add_and_scale_candle_patterns(df):
    candle_names = talib.get_function_groups().get('Pattern Recognition', [])
    pattern_columns = []
    new_columns = {}

    for pattern_name in candle_names:
        pattern_function = getattr(talib, pattern_name)
        new_columns[pattern_name] = pattern_function(df['Open'], df['High'], df['Low'], df['Close'])
        pattern_columns.append(pattern_name)

    # Tworzenie nowego DataFrame z nowymi kolumnami
    new_columns_df = pd.DataFrame(new_columns, index=df.index)

    # Łączenie nowego DataFrame z oryginalnym
    df = pd.concat([df, new_columns_df], axis=1)

    # Skalowanie
    if pattern_columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df[pattern_columns] = scaler.fit_transform(df[pattern_columns])

    return df


def add_scaled_indicators(df, n):
    indicator_columns = []
    new_columns = {}

    for p in range(2, n + 1):
        rsi_column = f'RSI{p}'
        new_columns[rsi_column] = talib.RSI(df['Close'], timeperiod=p)
        indicator_columns.append(rsi_column)

    for p in range(2, n + 1):
        mfi_column = f'MFI{p}'
        new_columns[mfi_column] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=p)
        indicator_columns.append(mfi_column)

    for p in range(1, n + 1):
        ultosc_column = f'ULTOSC{p}'
        new_columns[ultosc_column] = talib.ULTOSC(df['High'], df['Low'], df['Close'],
                                                  timeperiod1=p, timeperiod2=2 * p, timeperiod3=3 * p)
        indicator_columns.append(ultosc_column)

    # Tworzenie nowego DataFrame z nowymi kolumnami
    new_columns_df = pd.DataFrame(new_columns, index=df.index)

    # Łączenie nowego DataFrame z oryginalnym
    df = pd.concat([df, new_columns_df], axis=1)

    # Uzupełnianie brakujących wartości
    df[indicator_columns] = df[indicator_columns].fillna(0)

    # Skalowanie
    if indicator_columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df[indicator_columns] = scaler.fit_transform(df[indicator_columns])

    return df


# Funkcja tworząca środowiska
def make_env(df, **env_kwargs):
    def _init():
        return SpotTakerRL(df=df, **env_kwargs)

    return _init


TICKER = "BTCUSDT"
ITV = "5m"
MARKET_TYPE = "spot"
DATA_TYPE = "klines"
DF_TRAIN_START_DATE = "2017-09-01"
DF_TRAIN_END_DATE = "2024-06-01"
DF_TEST_START_DATE = "2024-06-01"
DF_TEST_END_DATE = "2024-09-30"

ENV_KWARGS = {
    "lookback_size": 142,  # 24H obs history
    "max_steps": 8_640,  # 30 days in 1h intervals
    "exclude_cols_left": 5,
    "init_balance": 1_000,
    "no_action_finish": inf,
    'steps_passive_penalty': 96,
    "fee": 0.00075,
    "coin_step": 0.00001,
    "stop_loss": 0.25,
    "take_profit": 0.25,
    "visualize": False,
    "render_range": 120,
    "verbose": True,
    "report_to_file": False,
}

MODEL_PARAMETERS = {
    'batch_size': 168,  # 168, 252, 336, 504
    'n_steps': 1_008,
    'total_timesteps': int(1_440_000 * 1),
}

NUM_ENVS = 6

if __name__ == "__main__":

    print(f'CUDA available: {cuda.is_available()}')
    experiment_name = f"CDL_passive_penalty_Rew_reduced_lb{ENV_KWARGS['lookback_size']}_ns{MODEL_PARAMETERS['n_steps']}_tt{MODEL_PARAMETERS['total_timesteps']}_PPO_BTCUSDT_1h_spot"
    tensorboard_log_dir = f"{TENSORBOARD_DIR}{experiment_name}"

    df_train = by_BinanceVision(
        ticker=TICKER,
        interval=ITV,
        market_type=MARKET_TYPE,
        data_type=DATA_TYPE,
        start_date=DF_TRAIN_START_DATE,
        end_date=DF_TRAIN_END_DATE,
        split=False,
        delay=0,
    )
    df_train = add_and_scale_candle_patterns(df_train)
    df_train = add_scaled_indicators(df_train, 28)

    # df_train = add_candle_patterns(df_train)

    envs = SubprocVecEnv([make_env(df_train, **ENV_KWARGS) for _ in range(NUM_ENVS)])

    # Zaktualizuj policy_kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 64, 32], vf=[128, 64, 32]),
        activation_fn=nn.ReLU
    )

    # Inicjalizuj model z nowym policy_kwargs
    model = PPO(
        "CnnPolicy",
        envs,
        n_steps=MODEL_PARAMETERS['n_steps'] // NUM_ENVS,
        batch_size=MODEL_PARAMETERS['batch_size'],
        learning_rate=1e-4,  # Zmniejszony learning rate
        clip_range=0.02,  # Zmniejszony clip_range
        ent_coef=0.03,  # Zwiększony współczynnik entropii
        vf_coef=0.5,  # Domyślny vf_coef
        gamma=0.99,  # Domyślna wartość gamma
        max_grad_norm=0.3,  # Zmniejszony max_grad_norm
        target_kl=0.1,  # Możesz dostosować w razie potrzeby
        tensorboard_log=tensorboard_log_dir,
        verbose=1,
        stats_window_size=5,
        device="cuda",
        policy_kwargs=policy_kwargs
    )

    model.learn(total_timesteps=MODEL_PARAMETERS['total_timesteps'],
                progress_bar=True,
                log_interval=5)

    _date = str(dt.today()).replace(":", "-")[:-7]
    model_full_path = f"{MODELS_DIR}d_{_date}"
    print(f'Saving model to: {model_full_path}')
    model.save(model_full_path)

    del model
    model = PPO.load(model_full_path)

    df_test = by_BinanceVision(
        ticker=TICKER,
        interval=ITV,
        market_type=MARKET_TYPE,
        data_type=DATA_TYPE,
        start_date=DF_TEST_START_DATE,
        end_date=DF_TEST_END_DATE,
        split=False,
        delay=0,
    )

    df_test = add_and_scale_candle_patterns(df_test)
    df_test = add_scaled_indicators(df_test, 28)
    test_env = SpotTakerRL(df=df_test, **ENV_KWARGS)

    print(f'### TEST STARTED ###', end='\n')
    for _ in range(10):
        obs = test_env.reset()
        terminated = False
        while not terminated:
            action, _states = model.predict(obs)
            obs, reward, terminated, info = test_env.step(action)

    print(f'### VISUALIZATION TEST STARTED ###', end='\n')
    ENV_KWARGS['visualize'] = True
    visualize_env = SpotTakerRL(df=df_test, **ENV_KWARGS)
    obs = visualize_env.reset()
    terminated = False
    while not terminated:
        action, _states = model.predict(obs)
        obs, reward, terminated, info = visualize_env.step(action)
