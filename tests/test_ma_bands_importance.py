import re

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from utils.data_collector import *
from utils.feature_functions import keltner_channels_custom_mas

if __name__ == "__main__":
    TICKER = "BTCUSDT"
    ITV = "5m"
    MARKET_TYPE = "spot"
    DATA_TYPE = "klines"

    df = by_BinanceVision(
        ticker=TICKER,
        interval=ITV,
        market_type=MARKET_TYPE,
        data_type=DATA_TYPE,
        start_date='2023-01-01',
        end_date='2024-01-01',
        split=False,
        delay=0,
    )
    print(df.shape)
    print(df.describe())
    df = keltner_channels_custom_mas(df,
                                     [5, 7, 10, 15, 20, 25, 35, 50, 75, 100],
                                     [5, 7, 8, 10, 13, 15, 20, 25, 30, 40, 50],
                                     [1.])
    df = df.iloc[:, 6:]
    print(df.shape)
    print(df.describe())
    print(f'NaNs: {df.isna().sum().sum()}')

    # 1. Progowanie Wariancji
    # Ustawiamy próg wariancji (możesz zmniejszyć próg, aby zachować więcej cech)
    variance_threshold = 0.005  # Zmniejszamy próg, aby zachować więcej cech
    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(df)
    df_var = df.loc[:, selector.get_support()]

    print(f"Liczba cech po progowaniu wariancji: {df_var.shape[1]}")

    # Wypisz nazwy cech zachowanych po progowaniu wariancji
    # Aby uniknąć zbyt długiego wydruku, możesz ograniczyć liczbę wypisanych cech
    print("Cechy zachowane po progowaniu wariancji (pierwsze 10):")
    print(df_var.columns.tolist()[:10])

    # 2. Analiza Korelacji
    # Obliczamy macierz korelacji
    corr_matrix = df_var.corr().abs()

    # Wybieramy górny trójkąt macierzy korelacji
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Znajdujemy cechy o korelacji powyżej progu (możesz zwiększyć próg, aby zachować więcej cech)
    correlation_threshold = 0.99  # Zwiększamy próg do 0.99
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

    # Usuwamy silnie skorelowane cechy
    df_uncorr = df_var.drop(columns=to_drop)

    print(f"Liczba cech po usunięciu silnie skorelowanych: {df_uncorr.shape[1]}")

    # Wypisz nazwy cech zachowanych po analizie korelacji (pierwsze 10)
    print("Cechy zachowane po usunięciu silnie skorelowanych (pierwsze 10):")
    print(df_uncorr.columns.tolist()[:10])

    # 3. Standaryzacja danych przed klasteryzacją
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_uncorr)

    # 4. Hierarchiczne Klasteryzowanie Cech
    # Obliczamy odległości korelacyjne między cechami
    distance_matrix = pdist(df_uncorr.T, metric='correlation')

    # Wykonujemy klasteryzację hierarchiczną
    linked = linkage(distance_matrix, method='ward')

    # Ustalamy liczbę klastrów (możesz zwiększyć procent, aby uzyskać więcej cech)
    num_features = df_uncorr.shape[1]
    desired_percentage = 0.5  # Zwiększamy procent do 50%
    num_clusters = int(num_features * desired_percentage)
    if num_clusters < 1:
        num_clusters = 1  # Zapewniamy co najmniej jeden klaster

    labels = fcluster(linked, num_clusters, criterion='maxclust')

    # Wybieramy reprezentatywne cechy z każdego klastra
    selected_features = []
    for cluster_id in np.unique(labels):
        cluster_features = df_uncorr.columns[labels == cluster_id]
        # Możemy wybrać cechę o najwyższej wariancji w klastrze
        selected_feature = df_uncorr[cluster_features].var().idxmax()
        selected_features.append(selected_feature)

    df_final = df_uncorr[selected_features]

    print(f"Liczba cech po klasteryzacji: {df_final.shape[1]}")
    selected_features = df_final.columns.tolist()

    # Wypisz nazwy cech zachowanych po klasteryzacji
    print("Cechy zachowane po klasteryzacji:")
    print(selected_features)

    result_tuples = []

    # Wyrażenie regularne do dopasowania liczby przed "MAp", po "p" oraz po "atr_p"
    for item in selected_features:
        before_map_match = re.search(r'(\d+)MAp', item)
        p_match = re.search(r'p(\d+)', item)
        atr_p_match = re.search(r'atr_p(\d+)', item)

        # Sprawdzenie, czy znaleziono dopasowania i dodanie do krotki jako int
        if before_map_match and p_match and atr_p_match:
            result_tuples.append((
                int(before_map_match.group(1)),
                int(p_match.group(1)),
                int(atr_p_match.group(1))
            ))

    print("Wynik:", result_tuples)
