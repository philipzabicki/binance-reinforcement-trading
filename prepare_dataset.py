import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.data_collector import *
from utils.ta_tools import get_MA_band_signal

KELTNER_PARAMETERS_2021m5 = [
    (27, 35, 25),
    (25, 35, 25),
    (26, 35, 25),
    (27, 35, 30),
    (25, 35, 30),
    (26, 35, 30),
    (27, 35, 5),
    (26, 35, 5),
    (24, 35, 5),
    (26, 35, 10),
    (25, 35, 7),
    (25, 35, 5),
    (27, 35, 20),
    (26, 35, 20),
    (25, 35, 13),
    (26, 35, 13),
    (25, 35, 15),
    (30, 50, 5),
    (27, 50, 7),
    (25, 50, 7),
    (26, 50, 7),
    (25, 50, 5),
    (14, 25, 15),
    (14, 25, 13),
    (14, 25, 10),
    (14, 25, 5),
    (14, 25, 7),
    (27, 50, 40),
    (25, 50, 40),
    (26, 50, 40),
    (14, 15, 25),
    (14, 15, 30),
    (14, 15, 15),
    (14, 15, 20),
    (14, 25, 30),
    (27, 50, 30),
    (26, 50, 30),
    (14, 25, 20),
    (14, 25, 25),
    (24, 50, 13),
    (26, 50, 13),
    (25, 50, 13),
    (26, 50, 20),
    (26, 50, 25),
    (26, 75, 5),
    (14, 100, 5),
    (25, 100, 5),
    (14, 50, 5),
    (0, 100, 5),
    (11, 75, 5),
    (9, 50, 5),
    (20, 50, 5),
    (14, 100, 40),
    (14, 100, 50),
    (14, 100, 7),
    (14, 100, 20),
    (14, 75, 20),
    (26, 100, 30),
    (26, 100, 20),
    (26, 100, 13),
    (26, 100, 7),
    (14, 50, 50),
    (26, 100, 40),
    (26, 100, 50),
    (26, 75, 40),
    (27, 75, 50),
    (26, 75, 50),
    (14, 25, 50),
    (14, 25, 40),
    (14, 50, 40),
    (14, 50, 30),
    (14, 50, 20),
    (14, 35, 20),
    (14, 35, 5),
    (14, 35, 13),
    (14, 50, 7),
    (26, 75, 30),
    (24, 75, 20),
    (26, 75, 20),
    (25, 75, 20),
    (25, 75, 7),
    (26, 75, 7),
    (27, 75, 5),
    (24, 75, 10),
    (26, 75, 13),
    (23, 25, 7),
    (23, 25, 5),
    (23, 25, 13),
    (23, 25, 10),
    (23, 25, 20),
    (23, 20, 7),
    (23, 20, 10),
    (23, 20, 5),
    (23, 20, 13),
    (23, 20, 15),
    (6, 10, 5),
    (23, 15, 15),
    (9, 10, 5),
    (20, 7, 5),
    (6, 7, 5),
    (23, 10, 10),
    (0, 7, 5),
    (26, 5, 5),
    (12, 5, 5),
    (7, 5, 5),
    (1, 5, 5),
    (10, 5, 5),
    (13, 15, 5),
    (9, 15, 5),
    (19, 15, 5),
    (9, 20, 5),
    (23, 25, 30),
    (23, 20, 20),
    (23, 25, 25),
    (13, 5, 5),
    (23, 5, 5),
    (23, 7, 7),
    (13, 7, 5),
    (9, 7, 5),
    (23, 7, 5),
    (23, 10, 5),
    (23, 10, 7),
    (23, 10, 8),
    (23, 15, 5),
    (23, 15, 10),
    (23, 15, 13),
    (23, 50, 25),
    (23, 50, 5),
    (23, 50, 13),
    (23, 50, 20),
    (23, 50, 40),
    (23, 35, 25),
    (23, 35, 30),
    (23, 35, 5),
    (23, 35, 10),
    (23, 35, 13),
    (23, 35, 15),
    (23, 35, 20),
    (27, 10, 8),
    (25, 10, 8),
    (26, 10, 8),
    (27, 10, 5),
    (26, 10, 5),
    (27, 10, 7),
    (25, 10, 7),
    (26, 10, 7),
    (27, 7, 5),
    (25, 7, 5),
    (26, 7, 5),
    (24, 7, 5),
    (6, 25, 5),
    (16, 7, 5),
    (19, 20, 7),
    (9, 25, 5),
    (23, 35, 40),
    (13, 35, 5),
    (9, 35, 7),
    (6, 35, 5),
    (9, 35, 5),
    (3, 10, 5),
    (3, 5, 5),
    (14, 7, 15),
    (8, 5, 5),
    (0, 10, 5),
    (7, 7, 5),
    (26, 7, 7),
    (16, 10, 5),
    (6, 50, 7),
    (19, 50, 7),
    (6, 50, 5),
    (7, 15, 5),
    (7, 15, 7),
    (9, 50, 7),
    (7, 10, 7),
    (4, 15, 5),
    (26, 10, 10),
    (23, 50, 50),
    (26, 20, 10),
    (24, 20, 7),
    (26, 20, 7),
    (25, 20, 7),
    (26, 20, 5),
    (24, 20, 5),
    (14, 15, 7),
    (14, 15, 10),
    (26, 25, 10),
    (26, 25, 5),
    (24, 25, 7),
    (26, 25, 7),
    (25, 25, 7),
    (14, 10, 10),
    (27, 15, 10),
    (26, 15, 10),
    (14, 5, 10),
    (14, 7, 5),
    (26, 15, 5),
    (14, 10, 5),
    (14, 15, 5),
    (26, 15, 7),
    (26, 15, 8),
    (14, 10, 7),
    (14, 5, 7),
    (14, 10, 8),
    (14, 5, 8),
    (4, 75, 5),
    (7, 50, 5),
    (4, 50, 5),
    (5, 100, 7),
    (7, 50, 7),
    (26, 50, 50),
    (4, 50, 7),
    (26, 35, 40),
    (12, 50, 8),
    (7, 35, 7),
    (11, 100, 7),
    (0, 75, 5),
    (16, 100, 5),
    (3, 100, 7),
    (4, 100, 7),
    (12, 75, 5),
    (11, 35, 7),
    (31, 100, 5),
    (7, 100, 5),
    (7, 75, 7),
    (10, 100, 5),
    (7, 100, 7),
    (0, 100, 7),
    (13, 100, 7),
    (6, 100, 7),
    (9, 100, 7),
    (9, 75, 7),
    (19, 75, 7),
    (6, 75, 7),
    (3, 25, 5),
    (3, 35, 5),
    (3, 50, 7),
    (3, 35, 7),
    (14, 15, 40),
    (11, 20, 5),
    (11, 10, 5),
    (14, 10, 20),
    (14, 10, 15),
    (26, 25, 25),
    (7, 25, 7),
    (0, 35, 5),
    (4, 35, 5),
    (7, 20, 5),
    (16, 25, 5),
    (12, 25, 5),
    (12, 20, 5),
    (26, 15, 15),
    (7, 20, 7),
    (10, 25, 5),
    (27, 25, 20),
    (25, 25, 20),
    (26, 25, 20),
    (27, 25, 13),
    (26, 25, 13),
    (27, 25, 15),
    (26, 25, 15),
    (27, 20, 13),
    (26, 20, 13),
    (27, 20, 15),
    (25, 20, 15),
    (26, 20, 15),
    (27, 15, 13),
    (25, 15, 13),
    (26, 15, 13),
    (14, 7, 13),
    (14, 10, 13),
    (14, 15, 13),
    (24, 50, 5),
    (12, 50, 5),
    (5, 100, 5),
    (0, 50, 5),
    (16, 5, 5),
    (6, 75, 5),
    (13, 75, 5),
    (9, 100, 5),
    (7, 25, 5),
    (16, 20, 5),
    (12, 35, 5),
    (23, 100, 40),
    (23, 100, 50),
    (23, 100, 20),
    (23, 100, 7),
    (23, 100, 5),
    (23, 75, 20),
    (23, 75, 13),
    (23, 75, 5),
    (23, 75, 30),
    (23, 75, 40),
    (23, 75, 50),
]
KELTNER_PARAMETERS_2022m5 = [
    (23, 50, 25),
    (23, 50, 5),
    (23, 50, 20),
    (23, 50, 40),
    (23, 35, 5),
    (23, 35, 13),
    (23, 35, 20),
    (23, 35, 25),
    (23, 35, 30),
    (23, 20, 7),
    (23, 20, 10),
    (23, 20, 5),
    (23, 20, 13),
    (23, 20, 15),
    (23, 25, 7),
    (23, 25, 5),
    (23, 25, 10),
    (23, 25, 15),
    (23, 25, 20),
    (23, 15, 15),
    (6, 10, 5),
    (9, 10, 5),
    (13, 10, 5),
    (6, 7, 5),
    (23, 10, 10),
    (13, 5, 5),
    (23, 5, 5),
    (23, 7, 7),
    (13, 7, 5),
    (9, 7, 5),
    (23, 7, 5),
    (23, 10, 5),
    (23, 10, 7),
    (23, 10, 8),
    (23, 15, 7),
    (23, 15, 5),
    (23, 15, 10),
    (23, 15, 13),
    (11, 100, 5),
    (7, 100, 5),
    (11, 50, 5),
    (0, 100, 5),
    (7, 75, 5),
    (10, 100, 5),
    (12, 75, 5),
    (7, 50, 5),
    (10, 75, 5),
    (26, 50, 50),
    (7, 35, 5),
    (4, 50, 5),
    (12, 50, 5),
    (26, 35, 40),
    (5, 100, 5),
    (11, 25, 5),
    (11, 15, 5),
    (3, 100, 5),
    (3, 50, 5),
    (13, 100, 5),
    (6, 100, 5),
    (9, 100, 5),
    (13, 75, 5),
    (6, 75, 5),
    (9, 75, 5),
    (26, 25, 25),
    (7, 25, 5),
    (4, 35, 5),
    (12, 35, 5),
    (3, 25, 5),
    (3, 10, 5),
    (3, 15, 5),
    (7, 20, 5),
    (0, 25, 5),
    (12, 20, 5),
    (7, 15, 5),
    (26, 15, 15),
    (10, 25, 5),
    (11, 10, 5),
    (14, 10, 50),
    (14, 15, 50),
    (27, 15, 13),
    (25, 15, 13),
    (26, 15, 13),
    (13, 35, 5),
    (9, 35, 5),
    (19, 50, 5),
    (19, 35, 5),
    (6, 50, 5),
    (9, 50, 5),
    (23, 50, 50),
    (7, 10, 5),
    (4, 15, 5),
    (26, 10, 10),
    (26, 7, 7),
    (0, 10, 5),
    (8, 5, 5),
    (3, 7, 5),
    (14, 5, 13),
    (14, 7, 25),
    (14, 7, 20),
    (14, 5, 10),
    (14, 7, 13),
    (14, 7, 15),
    (14, 10, 25),
    (14, 10, 30),
    (0, 7, 5),
    (26, 5, 5),
    (12, 5, 5),
    (7, 5, 5),
    (1, 5, 5),
    (10, 5, 5),
    (13, 15, 5),
    (9, 15, 5),
    (23, 25, 30),
    (9, 20, 5),
    (23, 20, 20),
    (6, 25, 5),
    (9, 25, 5),
    (13, 25, 5),
    (7, 7, 5),
    (23, 35, 40),
    (23, 25, 25),
    (27, 10, 7),
    (25, 10, 7),
    (26, 10, 7),
    (27, 10, 8),
    (25, 10, 8),
    (26, 10, 8),
    (27, 7, 5),
    (25, 7, 5),
    (26, 7, 5),
    (26, 10, 5),
    (14, 7, 5),
    (26, 75, 40),
    (14, 50, 40),
    (26, 75, 50),
    (14, 50, 50),
    (14, 25, 50),
    (14, 100, 40),
    (14, 100, 50),
    (14, 100, 5),
    (14, 100, 30),
    (14, 100, 20),
    (26, 100, 40),
    (26, 100, 50),
    (26, 100, 5),
    (26, 100, 30),
    (26, 100, 20),
    (26, 100, 25),
    (26, 50, 40),
    (14, 25, 40),
    (14, 15, 40),
    (14, 20, 30),
    (26, 50, 30),
    (14, 20, 25),
    (26, 50, 20),
    (26, 50, 25),
    (26, 75, 5),
    (26, 75, 13),
    (14, 50, 5),
    (14, 50, 13),
    (26, 75, 30),
    (26, 75, 20),
    (26, 75, 25),
    (14, 50, 30),
    (14, 50, 20),
    (14, 50, 25),
    (27, 15, 10),
    (26, 15, 10),
    (26, 15, 5),
    (26, 15, 7),
    (26, 15, 8),
    (26, 20, 10),
    (26, 20, 5),
    (26, 20, 7),
    (26, 20, 8),
    (14, 5, 7),
    (14, 7, 7),
    (14, 5, 8),
    (14, 7, 8),
    (14, 7, 10),
    (14, 10, 10),
    (14, 10, 7),
    (14, 10, 8),
    (14, 10, 5),
    (26, 20, 13),
    (27, 20, 15),
    (26, 20, 15),
    (27, 25, 20),
    (26, 25, 20),
    (26, 25, 10),
    (26, 25, 5),
    (26, 25, 7),
    (26, 25, 13),
    (26, 25, 15),
    (23, 100, 40),
    (23, 100, 50),
    (23, 100, 20),
    (23, 100, 5),
    (23, 75, 5),
    (23, 75, 20),
    (23, 75, 30),
    (23, 75, 40),
    (23, 75, 50),
    (14, 25, 10),
    (14, 25, 5),
    (14, 15, 5),
    (14, 15, 10),
    (14, 10, 13),
    (14, 15, 13),
    (14, 10, 15),
    (14, 15, 15),
    (14, 20, 20),
    (14, 10, 20),
    (14, 25, 13),
    (26, 50, 5),
    (26, 50, 13),
    (27, 35, 30),
    (25, 35, 30),
    (26, 35, 30),
    (26, 35, 13),
    (26, 35, 15),
    (26, 35, 10),
    (26, 35, 5),
    (26, 35, 20),
    (26, 35, 25),
]
KELTNER_PARAMETERS_2023m5 = [
    (9, 75, 5),
    (7, 25, 5),
    (7, 35, 5),
    (9, 100, 5),
    (16, 25, 5),
    (12, 35, 5),
    (12, 50, 5),
    (16, 5, 5),
    (7, 50, 7),
    (10, 75, 5),
    (26, 50, 50),
    (5, 100, 5),
    (4, 50, 5),
    (4, 75, 5),
    (7, 50, 5),
    (0, 35, 5),
    (7, 25, 7),
    (4, 35, 5),
    (7, 35, 7),
    (12, 50, 7),
    (4, 50, 7),
    (26, 35, 40),
    (11, 20, 5),
    (31, 100, 5),
    (3, 25, 5),
    (3, 50, 7),
    (6, 25, 5),
    (6, 35, 5),
    (9, 35, 5),
    (13, 15, 5),
    (9, 15, 5),
    (12, 5, 5),
    (7, 5, 5),
    (7, 7, 5),
    (6, 25, 7),
    (9, 20, 5),
    (9, 25, 5),
    (23, 25, 30),
    (23, 35, 40),
    (16, 7, 7),
    (16, 10, 8),
    (16, 10, 7),
    (26, 5, 5),
    (26, 7, 7),
    (10, 10, 5),
    (8, 5, 5),
    (23, 20, 20),
    (23, 25, 25),
    (19, 50, 5),
    (6, 50, 5),
    (19, 50, 7),
    (13, 50, 5),
    (9, 50, 5),
    (9, 50, 7),
    (13, 35, 5),
    (9, 35, 7),
    (20, 35, 5),
    (7, 15, 5),
    (16, 15, 5),
    (7, 10, 5),
    (26, 10, 10),
    (23, 50, 50),
    (13, 75, 7),
    (9, 75, 7),
    (13, 100, 7),
    (9, 100, 7),
    (16, 25, 7),
    (11, 10, 5),
    (4, 20, 5),
    (26, 15, 15),
    (26, 25, 25),
    (20, 50, 5),
    (7, 20, 5),
    (12, 25, 5),
    (10, 25, 5),
    (14, 10, 40),
    (14, 7, 30),
    (14, 7, 25),
    (3, 7, 5),
    (3, 15, 5),
    (16, 10, 5),
    (12, 100, 5),
    (7, 100, 5),
    (3, 100, 5),
    (3, 35, 5),
    (11, 100, 7),
    (11, 75, 5),
    (11, 100, 5),
    (7, 100, 7),
    (3, 100, 7),
    (11, 35, 7),
    (4, 100, 7),
    (10, 100, 5),
    (7, 75, 7),
    (12, 100, 7),
    (0, 100, 7),
    (16, 75, 5),
    (4, 100, 5),
    (12, 75, 5),
    (23, 50, 5),
    (23, 50, 25),
    (23, 50, 20),
    (23, 50, 40),
    (23, 35, 5),
    (23, 35, 13),
    (23, 35, 20),
    (23, 35, 25),
    (23, 35, 30),
    (23, 20, 7),
    (23, 20, 5),
    (23, 20, 13),
    (23, 20, 15),
    (23, 25, 5),
    (23, 25, 10),
    (23, 25, 20),
    (23, 7, 7),
    (13, 5, 5),
    (23, 5, 5),
    (6, 7, 5),
    (13, 7, 5),
    (9, 7, 5),
    (20, 7, 5),
    (6, 10, 5),
    (23, 15, 20),
    (9, 10, 5),
    (23, 10, 10),
    (23, 15, 15),
    (23, 15, 7),
    (23, 15, 5),
    (23, 15, 10),
    (23, 15, 13),
    (23, 10, 7),
    (23, 10, 8),
    (23, 10, 5),
    (23, 7, 5),
    (14, 35, 13),
    (14, 50, 5),
    (24, 75, 5),
    (14, 50, 30),
    (14, 50, 20),
    (14, 35, 20),
    (14, 50, 40),
    (14, 25, 40),
    (14, 15, 40),
    (24, 50, 40),
    (14, 100, 5),
    (14, 75, 20),
    (14, 100, 50),
    (14, 100, 25),
    (14, 50, 50),
    (14, 25, 50),
    (14, 15, 50),
    (24, 100, 40),
    (24, 100, 50),
    (26, 100, 5),
    (24, 100, 20),
    (24, 75, 20),
    (24, 75, 40),
    (24, 75, 50),
    (25, 35, 30),
    (14, 20, 30),
    (14, 10, 30),
    (24, 50, 25),
    (24, 50, 30),
    (24, 50, 5),
    (24, 50, 13),
    (14, 20, 25),
    (14, 10, 25),
    (24, 35, 20),
    (24, 35, 25),
    (24, 35, 13),
    (24, 35, 5),
    (14, 25, 5),
    (14, 20, 5),
    (14, 15, 7),
    (14, 15, 13),
    (14, 20, 15),
    (14, 25, 13),
    (14, 25, 20),
    (14, 15, 20),
    (14, 7, 20),
    (24, 25, 20),
    (23, 100, 40),
    (23, 100, 50),
    (23, 100, 20),
    (23, 100, 5),
    (23, 75, 20),
    (23, 75, 40),
    (23, 75, 5),
    (23, 75, 50),
    (24, 20, 5),
    (24, 20, 7),
    (24, 20, 10),
    (24, 20, 13),
    (24, 20, 15),
    (24, 25, 13),
    (24, 25, 5),
    (24, 25, 7),
    (14, 10, 15),
    (14, 5, 13),
    (14, 7, 13),
    (14, 5, 15),
    (26, 15, 13),
    (24, 15, 7),
    (24, 15, 5),
    (24, 15, 10),
    (14, 10, 10),
    (14, 5, 10),
    (14, 10, 7),
    (14, 5, 7),
    (14, 10, 8),
    (14, 5, 8),
    (26, 10, 7),
    (26, 10, 8),
    (26, 7, 5),
    (24, 7, 5),
    (14, 7, 5),
    (14, 10, 5),
    (24, 10, 5),
    (25, 10, 5),
]

# dataset with precalculated KDE probabilities
PREV_DATASET_FILENAME = "BTCUSDT5m_spot_modeling.csv"
TICKER = "BTCUSDT"
ITV = "5m"
MARKET_TYPE = "spot"
DATA_TYPE = "klines"

if __name__ == "__main__":
    prev_df = pd.read_csv(MODELING_DATASET_DIR + "/" + PREV_DATASET_FILENAME)

    df = by_BinanceVision(
        ticker=TICKER,
        interval=ITV,
        market_type=MARKET_TYPE,
        data_type=DATA_TYPE,
        split=False,
        delay=0,
    )

    yearly_common_params = list(
        set(KELTNER_PARAMETERS_2021m5)
        & set(KELTNER_PARAMETERS_2022m5)
        & set(KELTNER_PARAMETERS_2023m5)
    )
    print(f"yearly_common_params: {yearly_common_params}")

    np_df = df.iloc[:, 1:6].to_numpy()
    for pars in yearly_common_params:
        df[f"{pars[0]}MAp{pars[1]}atr_p{pars[2]}atr_m1.0"] = get_MA_band_signal(
            np_df, pars[0], pars[1], pars[2], 1.0
        )

    df["Opened"] = pd.to_datetime(df["Opened"])
    prev_df["Opened"] = pd.to_datetime(prev_df["Opened"])

    conflicted_columns = prev_df.columns.intersection(df.columns).drop("Opened")
    print(f"conflicted_columns {conflicted_columns}")
    df = df.drop(columns=conflicted_columns)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    scaler_filename = f"{MODELING_DATASET_DIR}/minmax_BTCUSDT5m_keltner_mas.pkl"
    joblib.dump(scaler, scaler_filename)
    print(f"Saved scaler to: {scaler_filename}")

    final_df = prev_df.merge(df, how="left", right_on="Opened", left_on="Opened")

    final_df.to_csv(
        f"{MODELING_DATASET_DIR}/BTCUSDT5m_spot_modeling_v2.csv", index=False
    )
