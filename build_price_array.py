# build_price_array.py
import pandas as pd
import numpy as np

# Purpose of this file: load individual CSVs and build aligned price array
# Why?: to ensure all tickers have data for the same dates
# and to save as a single .npy file for easy loading in the envs

TICKERS = ["EOSUSD", "ZRXUSD", "NEOUSD", "TRXUSD", "OMGUSD", "BATUSD", "ADAUSD", "QTUMUSD", "XTZUSD", "DOGEUSD"]

# TICKERS = [
#     "GALAUSD", "DOGEUSD", "MANAUSD", "SHIBUSD", "AVAXUSD",
#     "TRXUSD", "CVCUSD", "SOLUSD", "CHZUSD", "ENJUSD",
# ]

# Load single ticker CSV
def load_single_ticker_csv(ticker):
    path = f"data_collection/coin_data/{ticker}.csv"
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    return df[["timestamp", "close"]]

def main():
    # Load all dataframes
    dfs = {}
    # Loop through each ticker and load its data
    for t in TICKERS:
        df = load_single_ticker_csv(t)
        dfs[t] = df.set_index("timestamp")

    # Find common date index across all tickers
    common_index = None
    # For all values in tickers, we find the intersection of their date indices
    # This ensures we only keep dates where all tickers have data
    for t in TICKERS:
        idx = dfs[t].index
        common_index = idx if common_index is None else common_index.intersection(idx)

    # Sort the common index
    common_index = common_index.sort_values()

    cols = []
    for t in TICKERS:
        col = dfs[t].loc[common_index, "close"].to_numpy(dtype=float)
        cols.append(col)

    price_array = np.column_stack(cols)

    # Save the aligned price array into the file. 
    # this will be loaded in make_envs.py which is used by the RL envs
    np.save("price_array_aligned.npy", price_array)

if __name__ == "__main__":
    main()