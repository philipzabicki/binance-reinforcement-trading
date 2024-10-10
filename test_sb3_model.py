from torch import cuda
from datetime import datetime as dt
from numpy import inf

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

import talib
from definitions import TENSORBOARD_DIR, MODELS_DIR
from environments.spot_rl_env import SpotTakerRL
from utils.data_collector import by_BinanceVision


def add_candle_patterns(df):
    candle_names = talib.get_function_groups()['Pattern Recognition']
    for pattern_name in candle_names:
        pattern_function = getattr(talib, pattern_name)
        df[pattern_name] = pattern_function(df['Open'], df['High'], df['Low'], df['Close'])
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
DF_TRAIN_END_DATE = "2024-03-01"
DF_TEST_START_DATE = "2024-03-03"
DF_TEST_END_DATE = "2024-08-30"

ENV_KWARGS = {
    "lookback_size": 126,  # 24H obs history
    "max_steps": 4_032,  # 30 days in 1h intervals
    "exclude_cols_left": 5,
    "init_balance": 1_000,
    "no_action_finish": inf,
    'steps_passive_penalty': 96,
    "fee": 0.00075,
    "coin_step": 0.00001,
    "stop_loss": 0.25,
    "take_profit": 0.25,
    "visualize": False,
    "render_range": 60,
    "verbose": True,
}

MODEL_PARAMETERS = {
    'batch_size': 63,
    'n_steps': 1_008,
    'total_timesteps': 1_440_000*1,
}

NUM_ENVS = 12

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

    # df_train = add_candle_patterns(df_train)

    envs = SubprocVecEnv([make_env(df_train, **ENV_KWARGS) for _ in range(NUM_ENVS)])

    model = PPO("MlpPolicy",
                envs,
                n_steps=MODEL_PARAMETERS['n_steps'] // NUM_ENVS,
                batch_size=MODEL_PARAMETERS['batch_size'],
                clip_range=0.1,
                learning_rate=0.0001,
                tensorboard_log=tensorboard_log_dir,
                verbose=1,
                stats_window_size=5,
                device="cuda")

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

    df_test = add_candle_patterns(df_test)
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
