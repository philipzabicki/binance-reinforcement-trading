from os import path

# project directories structure definitions
ROOT_DIR = path.dirname(path.abspath(__file__))
DATA_DIR = ROOT_DIR + '/data/'
REPORT_DIR = ROOT_DIR + '/reports/'
LOG_DIR = DATA_DIR + '/logs/'
SLIPPAGE_DIR = DATA_DIR + '/slippage/'
TENSORBOARD_DIR = ROOT_DIR + '/tensorboard/'
MODELS_DIR = ROOT_DIR + '/model/'
