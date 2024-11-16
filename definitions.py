from os import path

# TODO: Add code to create necessary directories if nonexistent, remove / at the end of DIRs as it's non obvious

# project directories structure definitions
ROOT_DIR = path.dirname(path.abspath(__file__))

DATA_DIR = ROOT_DIR + "/data"
MODELING_DATASET_DIR = DATA_DIR + "/modeling"
REPORT_DIR = ROOT_DIR + "/reports"
LOG_DIR = DATA_DIR + "/logs"
SLIPPAGE_DIR = DATA_DIR + "/slippage"
TENSORBOARD_DIR = ROOT_DIR + "/tensorboard"
MODELS_DIR = ROOT_DIR + "/model"
