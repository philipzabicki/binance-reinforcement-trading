from os import path, environ, makedirs

# 0 = pokaż wszystko, 1 = tylko WARNING+ERROR, 2 = tylko ERROR, 3 = cisza
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import gc
import random
import hashlib
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
        n_time_steps, n_features = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            nn.Conv1d(512, 128, kernel_size=5, padding=2),  # z 256 na 128
            nn.ReLU(),

            nn.Conv1d(128, 64, kernel_size=7, padding=3),  # z 128 na 64
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(64 * 2, 1)

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


def seed_everything(seed):
    environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    # PyTorch ≥1.8:
    th.use_deterministic_algorithms(True, warn_only=True)


def make_env(df, env_number, **env_kwargs):
    env_kwargs['seed'] += env_number
    print(f'Env created with seed: {env_kwargs["seed"]}')
    def _init():
        return DiscreteSpotTakerRL(df=df, **env_kwargs)
    return _init


ENV_ORDER = [
    "lookback_size", "max_steps",
    "steps_passive_penalty", "exclude_cols_left"
]
MODEL_ORDER = [
    "gamma", "learning_rate", "buffer_size", "batch_size", "learning_starts",
    "target_update_interval", "exploration_initial_eps", "exploration_final_eps",
    "exploration_fraction"
]
ABBREV = {
    # --- ENV ---
    "lookback_size": "lb", "max_steps": "ms",
    "steps_passive_penalty": "spp", "exclude_cols_left": "ecl",
    # --- MODEL ---
    "gamma": "g", "learning_rate": "lr", "buffer_size": "bs",
    "batch_size": "ba", "learning_starts": "ls",
    "target_update_interval": "tui",
    "exploration_initial_eps": "ei", "exploration_final_eps": "ef",
    "exploration_fraction": "exf",
}


def sha1_of_file(path, chunk_size=8192):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt(k, v):
    """Zamienia parę klucz-wartość na bardzo krótki fragment nazwy."""
    k = ABBREV.get(k, k)  # zamień na skrót
    # ————— normalizacja wartości —————
    if isinstance(v, bool):
        v = int(v)  # True → 1
    elif v in (np.inf, float("inf")):
        v = "inf"
    elif isinstance(v, float):
        v = f"{v:g}".replace(".", "p")  # 5e-05 → 5e-05, 0.25 → 0p25
    elif isinstance(v, int) and abs(v) >= 1000:
        v = f"{int(round(v / 1000))}k"  # 75000 → 75k
    return f"{k}{v}"


SEED = 37
FULL_DATASET_FILENAME = "full_modeling.csv"
TRAIN_DATASET_FILENAME = "train_modeling.csv"
TEST_DATASET_FILENAME = "test_modeling.csv"
# DATASET_SPLIT_DATE = "2024-03-01"
NUM_ENVS = 14
TOTAL_ENV_STEPS = 50_000 * NUM_ENVS
# TOTAL_ENV_STEPS = 200_000
ENV_KWARGS = {
    "lookback_size": 128,
    "max_steps": 720,
    "exclude_cols_left": 5,
    "init_balance": 1_000,
    "no_action_finish": inf,
    "steps_passive_penalty": 72,
    "fee": 0.00075,
    # "fee": 0.0,
    "coin_step": 0.00001,
    "visualize": False,
    "render_range": 80,
    "verbose": True,
    "report_to_file": False,
    "seed": SEED
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
    "gamma": 0.0,
    # Initial probability of random action
    "exploration_initial_eps": 1.0,
    # Final probability reached after exploration_fraction * total_timesteps
    "exploration_final_eps": 0.0,
    # Portion of training time devoted to linear ε decay
    "exploration_fraction": 0.25,
    "stats_window_size": 10,
    "seed": SEED
}
N_EVAL_EPISODES = 100
VAL_MAX_STEPS = ENV_KWARGS["max_steps"]

if __name__ == "__main__":
    seed_everything(SEED)
    print(f"CUDA available: {cuda.is_available()}")

    full_dataset_path = path.join(MODELING_DATASET_DIR, FULL_DATASET_FILENAME)
    dataset_hash = sha1_of_file(full_dataset_path)[:8]
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    parts = [_fmt(k, ENV_KWARGS[k]) for k in ENV_ORDER] + \
            [_fmt(k, MODEL_PARAMETERS[k]) for k in MODEL_ORDER]
    experiment_name = f"{dataset_hash}_{timestamp}_{'_'.join(parts)}_tt{TOTAL_ENV_STEPS // 1000}k_DQN"
    tensorboard_log_dir = f"{TENSORBOARD_DIR}/{experiment_name}"

    df_train = pd.read_csv(
        path.join(MODELING_DATASET_DIR, TRAIN_DATASET_FILENAME))
    # float_cols = [c for c in df_train.columns if df_train[c].dtype == "float64"]
    # int_cols = [c for c in df_train.columns if df_train[c].dtype == "int64"]
    # dtype_map = {**{c: "float32" for c in float_cols},
    #              **{c: "int32" for c in int_cols}}
    # df_train = pd.read_csv(
    #     path.join(MODELING_DATASET_DIR, TRAIN_DATASET_FILENAME),
    #     dtype=dtype_map)
    print("Shape:", df_train.shape)
    df_train.info(memory_usage='deep')
    total_mem = df_train.memory_usage(deep=True).sum()
    print(f"\nTotal memory: {total_mem / (1024 ** 2):.2f} MB")

    for p in [0.45, 0.5, 0.55]:
        # TOTAL_ENV_STEPS = p
        MODEL_PARAMETERS["exploration_fraction"] = p
        # MODEL_PARAMETERS["gamma"] = p

        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        parts = [_fmt(k, ENV_KWARGS[k]) for k in ENV_ORDER] + \
                [_fmt(k, MODEL_PARAMETERS[k]) for k in MODEL_ORDER]
        experiment_name = f"{dataset_hash}_{timestamp}_{'_'.join(parts)}_tt{TOTAL_ENV_STEPS // 1000}k_DQN"
        tensorboard_log_dir = f"{TENSORBOARD_DIR}/{experiment_name}"

        envs = SubprocVecEnv([make_env(df_train, env_n, **ENV_KWARGS) for env_n in range(NUM_ENVS)])
        envs = VecMonitor(envs)

        policy_kwargs = dict(
            features_extractor_class=CustomCNNBiLSTMAttentionFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=[256, 64, 16],
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
        # ── 1. Reload the trained model ───────────────────────────────────────────────
        model = DQN.load(model_full_path)  # env will be attached later in eval envs

        # ── 2. TensorBoard loggers ----------------------------------------------------
        eval_root_dir = path.join(TENSORBOARD_DIR, experiment_name)
        eval_rnd_dir = path.join(eval_root_dir, "eval_random")
        eval_full_dir = path.join(eval_root_dir, "eval_full")

        eval_rnd_logger = configure(eval_rnd_dir, ["stdout", "tensorboard"])
        eval_full_logger = configure(eval_full_dir, ["tensorboard"])

        model.set_logger(eval_rnd_logger)  # random-episode metrics go here

        # ── 3. Load test dataset ------------------------------------------------------
        df_test = pd.read_csv(path.join(MODELING_DATASET_DIR, TEST_DATASET_FILENAME))


        # ── 4. Helper: create env kwargs without mutating the global dict ------------
        def get_env_kwargs(**overrides):
            """Return a *new* kwargs dict, merging ENV_KWARGS with overrides."""
            return {**ENV_KWARGS, **overrides}


        # ── 5. Evaluation – 100 random episodes --------------------------------------
        LOG_TAGS = ("balance", "gain", "above_benchmark", "return",
                    "pnl", "win_rate", "profit_loss_ratio" "mean_pnl", "std_pnl", "in_gain_ratio")

        print(f"### VALIDATION: {N_EVAL_EPISODES} random episodes "
              f"({VAL_MAX_STEPS} steps each) ###")

        val_env_rnd = DiscreteSpotTakerRL(
            df=df_test,
            **get_env_kwargs(max_steps=VAL_MAX_STEPS,
                             visualize=False,
                             report_to_file=True)
        )

        episode_stats = []

        for ep in trange(N_EVAL_EPISODES, desc="Random episodes"):
            obs, info = val_env_rnd.reset()
            terminated = truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = val_env_rnd.step(action)

            # collect per-episode metrics (NaN if not present)
            episode_stats.append({tag: info.get(tag, np.nan) for tag in LOG_TAGS})

        # convert to DataFrame for easy stats
        df_stats = pd.DataFrame(episode_stats)
        with pd.option_context('display.max_columns', None,
                               'display.max_rows', None,
                               'display.width', None):
            print(print(df_stats.describe().T))
        stats_path = path.join(tensorboard_log_dir,
                               "eval_random",  # albo inny pod-dir
                               "episode_stats.csv")
        makedirs(path.dirname(stats_path), exist_ok=True)  # ensure dir

        df_stats.to_csv(stats_path, index=False)
        print(f"Saved stats to: {stats_path}")

        # log mean & std for each metric
        for tag in LOG_TAGS:
            if df_stats[tag].notna().any():
                model.logger.record(f"eval/{tag}", float(df_stats[tag].mean()))
        model.logger.dump(model.num_timesteps)

        # ── 6. Evaluation – one *full* episode (log every step) -----------------------
        print("### VALIDATION: one full episode (no step limit) ###")
        # model.set_logger(eval_full_logger)

        val_env_full = DiscreteSpotTakerRL(
            df=df_test,
            **get_env_kwargs(max_steps=0,  # 0 == run until done
                             visualize=False,
                             report_to_file=True)
        )

        step_idx = 0
        obs, info = val_env_full.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = val_env_full.step(action)

            # per-step logging
            eval_full_logger.record("eval_full/reward", float(reward))
            for tag in LOG_TAGS:
                if tag in info:
                    eval_full_logger.record(f"eval_full/{tag}", float(info[tag]))
            eval_full_logger.dump(step_idx)  # flush to .event file
            step_idx += 1

        eval_full_logger.close()  # finish the .event file

        # ── 7. (Optional) Visual sanity-check episode with live rendering ------------
        # print("### VISUALISATION TEST (render=True, full length) ###")
        #
        # vis_env = DiscreteSpotTakerRL(
        #     df=df_test,
        #     **get_env_kwargs(max_steps=0, visualize=True, report_to_file=False)
        # )
        #
        # obs, info = vis_env.reset()
        # terminated = False
        # while not terminated:
        #     action, _ = model.predict(obs, deterministic=True)
        #     obs, reward, terminated, truncated, info = vis_env.step(action)
        #
        # print("Validation & visualisation finished.")
        gc.collect()
