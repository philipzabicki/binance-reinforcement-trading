import os
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import requests

from definitions import DATA_DIR


# Function to download and extract aggTrades data if CSV doesn't exist
def download_and_extract(symbol, year, month, save_path):
    csv_file = f"{save_path}/{symbol}-aggTrades-{year}-{month:02d}.csv"
    # If the CSV file already exists, skip downloading
    if os.path.exists(csv_file):
        print(f"{csv_file} already exists. Skipping download.")
        return
    base_url = "https://data.binance.vision/data/spot/monthly/aggTrades"
    file_name = f"{symbol}-aggTrades-{year}-{month:02d}.zip"
    url = f"{base_url}/{symbol}/{file_name}"
    response = requests.get(url)
    if response.status_code == 200:
        with ZipFile(BytesIO(response.content)) as z:
            z.extractall(save_path)
        print(f"Downloaded and extracted {file_name}")
    else:
        print(f"Failed to download {file_name}")


# Function to calculate VWAP and Volume for a given DataFrame and interval
def calculate_vwap_and_volume(df, interval='1h'):
    # Convert timestamp from milliseconds to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    # Aggregate the sum of price_quantity and quantity over the interval
    resampled = df.resample(interval).agg({'price_quantity': 'sum', 'quantity': 'sum'})
    # Calculate VWAP (Volume Weighted Average Price)
    resampled['VWAP'] = resampled['price_quantity'] / resampled['quantity']
    # Rename quantity to Volume for clarity
    resampled = resampled.rename(columns={'quantity': 'Volume'})
    # Return DataFrame with VWAP and Volume columns
    return resampled[['VWAP', 'Volume']]


# Parameters
symbol = 'BTCUSDT'
years = range(2017, 2025)
months = range(1, 13)
save_path = os.path.join(DATA_DIR, "binance_vision", "spot", "aggTrades")

# Ensure the save directory exists
os.makedirs(save_path, exist_ok=True)

# List to hold individual VWAP DataFrames for later concatenation
final_dfs = []

# Download and process data
for year in years:
    for month in months:
        # Construct CSV file path for aggregated trades
        csv_file = f"{save_path}/{symbol}-aggTrades-{year}-{month:02d}.csv"
        # Download and extract CSV if it doesn't exist
        if not os.path.exists(csv_file):
            download_and_extract(symbol, year, month, save_path)

        # Proceed only if the CSV file exists
        if os.path.exists(csv_file):
            # Define the path for the VWAP result file
            vwap_file = f"{save_path}/vwap_{symbol}_{year}_{month:02d}.csv"
            if os.path.exists(vwap_file):
                # If VWAP file exists, load it
                result_df = pd.read_csv(vwap_file, index_col=0, parse_dates=True)
                print(f"Loaded existing VWAP data for {year}-{month:02d}")
            else:
                # Read CSV file without header and assign column names manually
                df = pd.read_csv(csv_file, header=None, names=[
                    'tradeId', 'price', 'quantity', 'firstTradeId', 'lastTradeId',
                    'timestamp', 'isBuyerMaker', 'isBestMatch'
                ])
                # Convert price and quantity to float and compute price_quantity
                df['price'] = df['price'].astype(float)
                df['quantity'] = df['quantity'].astype(float)
                df['price_quantity'] = df['price'] * df['quantity']

                # Calculate VWAP and Volume using the specified interval
                result_df = calculate_vwap_and_volume(df, interval='1h')
                # Fill missing values with the previous non-null value
                result_df.fillna(method='ffill', inplace=True)
                # Save the individual VWAP and Volume file
                result_df.to_csv(vwap_file)
                print(f"VWAP and Volume data saved for {year}-{month:02d}")
            # Append to final list for combined DataFrame
            final_dfs.append(result_df)

# Combine all individual DataFrames into one final DataFrame and sort by timestamp
if final_dfs:
    final_df = pd.concat(final_dfs)
    final_df.sort_index(inplace=True)
    # Save the combined DataFrame to CSV
    final_combined_file = f"{save_path}/vwap_{symbol}_combined.csv"
    final_df.to_csv(final_combined_file)
    print("Combined VWAP and Volume data saved.")
