from datetime import datetime as dt
from numpy import inf
from ray.rllib.algorithms import ppo, dqn
from ray.rllib.env import EnvContext
from definitions import TENSORBOARD_DIR, MODELS_DIR
from environments.spot_rl_env import SpotTakerRL
from utils.data_collector import by_BinanceVision
import ray

TICKER = "BTCUSDT"
ITV = "1h"
MARKET_TYPE = "spot"
DATA_TYPE = "klines"
TRADE_START_DATE = "2021-01-01"
TRADE_END_DATE = "2024-08-25"
DF_START_DATE = "2024-06-01"
DF_END_DATE = "2024-08-01"
ENV_KWARGS = {
    "lookback_size": 336,
    "max_steps": 2_880,
    "exclude_cols_left": 6,
    "init_balance": 1_000,
    "no_action_finish": inf,
    "fee": 0.00075,
    "coin_step": 0.00001,
    "stop_loss": 0.25,
    "take_profit": 0.25,
    "visualize": False,
    "render_range": 60,
    "verbose": True,
}

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    df = by_BinanceVision(
        ticker=TICKER,
        interval=ITV,
        market_type=MARKET_TYPE,
        data_type=DATA_TYPE,
        split=False,
        delay=0,
    )
    print(f"df used: {df}")

    def env_creator(env_config: EnvContext):
        return SpotTakerRL(df=df, **ENV_KWARGS)

    # Rejestracja niestandardowego środowiska
    from ray.tune.registry import register_env
    register_env("spot_taker_rl", env_creator)

    # Konfiguracja trenera PPO
    config = ppo.PPOConfig().environment("spot_taker_rl", env_config={}).framework("torch")
    trainer = config.build()

    # Trenowanie modelu
    for i in range(1000):
        result = trainer.train()
        print(f"Iteracja {i}: {result['episode_reward_mean']}")

    # Zapisanie modelu
    _date = str(dt.today()).replace(":", "-")[:-7]
    model_full_path = f"{MODELS_DIR}ppo_{_date}"
    trainer.save(model_full_path)

    # Wczytywanie modelu
    trainer.restore(model_full_path)

    ENV_KWARGS['start_date'] = TRADE_START_DATE
    ENV_KWARGS['end_date'] = TRADE_END_DATE

    # Testowanie modelu
    test_env = SpotTakerRL(df=df, **ENV_KWARGS)
    obs = test_env.reset()
    terminated = False
    while not terminated:
        action = trainer.compute_single_action(obs)
        obs, reward, terminated, info = test_env.step(action)

    ENV_KWARGS['visualize'] = True
    visualize_env = SpotTakerRL(df=df, **ENV_KWARGS)
    obs = visualize_env.reset()
    terminated = False
    while not terminated:
        action = trainer.compute_single_action(obs)
        obs, reward, terminated, info = visualize_env.step(action)
