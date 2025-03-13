from environments.spot_rl_env import SpotTakerRL
from numpy import inf
from stable_baselines3 import PPO
from torch import cuda

from definitions import MODELS_DIR
from utils.data_collector import get_precalculated_dataset_by_filename
from utils.feature_functions import *

MODEL_FILENAME = 'PPO_2024-10-30 12-20-12.zip'
MODELING_DATASET_FILENAME = 'BTCUSDT5m_spot_modeling.csv'
DATASET_SPLIT_DATE = "2024-06-01"
NUM_ENVS = 12
ENV_KWARGS = {
    "lookback_size": 72,  # 6h obs history (5m itv)
    "max_steps": 8_640,  # 30D of trading (5m itv)
    "exclude_cols_left": 5,
    "init_balance": 1_000,
    "no_action_finish": inf,
    'steps_passive_penalty': 'auto',
    "fee": 0.00075,
    "coin_step": 0.00001,
    "stop_loss": 0.25,
    "take_profit": 0.25,
    "visualize": False,
    "render_range": 80,
    "verbose": True,
    "report_to_file": True,
}

if __name__ == "__main__":
    _, df_test = get_precalculated_dataset_by_filename(MODELING_DATASET_FILENAME, DATASET_SPLIT_DATE)
    df_test = add_scaled_candle_patterns_indicators(df_test)
    df_test = add_scaled_ultosc_rsi_mfi_up_to_n(df_test, 100, 5)
    print(f'Used dataset: {df_test.describe()}')

    model_fullpath = f'{MODELS_DIR}/{MODEL_FILENAME}'
    print(f'CUDA available: {cuda.is_available()}')
    model = PPO.load(model_fullpath)

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
