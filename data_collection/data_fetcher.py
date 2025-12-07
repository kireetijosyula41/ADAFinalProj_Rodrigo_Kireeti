"""
data_fetcher.py

Provides utilities to fetch OHLCV (open/high/low/close/volume) aggregation data
for cryptocurrencies from the Polygon (massive) REST API and save selected coin
data to CSV files.

This module:
- Loads the POLYGON_API_KEY from environment (using python-dotenv).
- Exposes two functions:
  - fetch_crypto_ohlcv_wo_edit: returns raw aggregation objects as a DataFrame
    with the timestamp left in milliseconds.
  - fetch_crypto_ohlcv: returns a cleaned DataFrame with standardized OHLCV
    column names, timestamp converted to pandas datetime and sorted.
- Iterates a list of chosen coins and writes CSV files to ./coin_data/.

Requirements:
- A .env file or environment variable POLYGON_API_KEY must be set.
"""
from massive import RESTClient
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables and initialize Polygon REST client
load_dotenv()
POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
client = RESTClient(POLYGON_API_KEY)

def fetch_crypto_ohlcv_wo_edit(symbol, start, end, timespan="day", multiplier=1):
    """
    Fetch OHLCV aggregations for a given symbol from Polygon and return them
    as a pandas DataFrame without modifying the timestamp.

    Parameters:
    - symbol (str): Ticker symbol (e.g., "X:BTCUSD").
    - start (str): Start date in YYYY-MM-DD format.
    - end (str): End date in YYYY-MM-DD format.
    - timespan (str): Aggregation timespan (default "day").
    - multiplier (int): Timespan multiplier (default 1).

    Returns:
    - pandas.DataFrame: DataFrame created from the raw aggregation objects.
      The 't' timestamp will remain in milliseconds as provided by Polygon.
    """
    aggs = []
    # Collect aggregation objects from the Polygon REST client
    for a in client.list_aggs(
        ticker=symbol,
        multiplier=multiplier,
        timespan=timespan,
        from_=start,
        to=end,
        limit=50000
    ):
        aggs.append(a)
    # Convert list of aggregation objects to DataFrame
    df = pd.DataFrame([a.__dict__ for a in aggs])
    return df


def fetch_crypto_ohlcv(symbol, start, end, timespan="day", multiplier=1):
    """
    Fetch OHLCV aggregations for a given symbol, convert to a standardized
    OHLCV DataFrame, convert timestamp from milliseconds to pandas datetime,
    and sort by timestamp.

    Parameters:
    - symbol (str): Ticker symbol (e.g., "X:BTCUSD").
    - start (str): Start date in YYYY-MM-DD format.
    - end (str): End date in YYYY-MM-DD format.
    - timespan (str): Aggregation timespan (default "day").
    - multiplier (int): Timespan multiplier (default 1).

    Returns:
    - pandas.DataFrame: DataFrame with columns renamed to standard OHLCV names,
      a converted 'timestamp' column of dtype datetime64[ns], and sorted by time.
    """
    # Fetch raw aggregation objects from Polygon
    aggs = []
    for a in client.list_aggs(
        ticker=symbol,
        multiplier=multiplier,
        timespan=timespan,
        from_=start,
        to=end,
        limit=50000
    ):
        aggs.append(a)
    # Build DataFrame from aggregation objects
    df = pd.DataFrame([a.__dict__ for a in aggs])
    # Rename Polygon aggregation fields to standard OHLCV names
    df = df.rename(columns={
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "vw": "vwap",
        "n": "transactions"
    })
    # Convert timestamp from milliseconds to pandas datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    # Ensure data is ordered by time and reindex
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

# List of coins to fetch and save locally as CSV files
chosen_coins = ["X:EOSUSD", "X:ZRXUSD", "X:NEOUSD", "X:TRXUSD", "X:OMGUSD", "X:BATUSD", "X:ADAUSD", "X:QTUMUSD", "X:XTZUSD", "X:DOGEUSD"]
# original_chosen_coins = ["X:GALAUSD", "X:DOGEUSD", "X:MANAUSD", "X:SHIBUSD", "X:AVAXUSD", "X:TRXUSD", "X:CVCUSD", "X:SOLUSD", "X:CHZUSD", "X:ENJUSD"]

# Iterate through chosen coins, fetch standardized OHLCV data, and save to ./coin_data/
for i in chosen_coins:
    print(f"data for chosen coin {i}")
    df_doge = fetch_crypto_ohlcv(f"{i}", "2018-03-26", "2025-12-03")
    # Persist fetched DataFrame to a CSV file named after the coin (strip the "X:" prefix)
    df_doge.to_csv(f"./coin_data/{i[2:]}.csv", index=False)
    print(df_doge.head())
    print(df_doge.tail())