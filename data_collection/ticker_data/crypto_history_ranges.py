"""
crypto_history_ranges.py

Utilities to discover available cryptocurrency tickers from the Polygon REST API
and record the available historical OHLCV date ranges for each ticker to a CSV.

This script:
- Loads the POLYGON_API_KEY from the environment.
- Lists all active crypto tickers from Polygon.
- For each ticker, fetches daily OHLCV aggregations over a broad date range,
  determines the first and last available dates and number of records, and
  writes the summarized results to `crypto_date_ranges.csv`.

Intended to be run as a script (if __name__ == "__main__") to produce a CSV
summary of historical date ranges for crypto tickers.
"""
import csv
from datetime import datetime
from dotenv import load_dotenv
from massive import RESTClient
import pandas as pd
import os

# Load environment variables and obtain API key
load_dotenv()
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Initialize a REST client instance for Polygon
client = RESTClient(POLYGON_API_KEY)


def get_all_crypto_tickers(client):
    """
    Retrieve all active crypto tickers from Polygon.

    Returns:
    - list: A list of Ticker objects returned by the Polygon client for the
      crypto market with active="true".
    """
    tickers_gen = client.list_tickers(
        market="crypto",
        active="true",
        limit=1000,
    )
    return list(tickers_gen)

def fetch_crypto_ohlcv(symbol, start, end, timespan="day", multiplier=1):
    """
    Fetch OHLCV aggregations for a given symbol from Polygon and return a
    standardized pandas DataFrame.

    Steps performed:
    - Calls the client's list_aggs to collect aggregation objects.
    - Converts the aggregation objects into a DataFrame.
    - Renames Polygon's fields to standard OHLCV column names.
    - Converts the 'timestamp' column from milliseconds to pandas datetime
      and sorts the DataFrame by timestamp.

    Returns:
    - pandas.DataFrame: Standardized OHLCV DataFrame, or an empty DataFrame
      if no 'timestamp' column is present.
    """
    # Fetch data from Polygon API
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
    # Convert to DataFrame
    df = pd.DataFrame([a.__dict__ for a in aggs])
    # Rename columns to standard OHLCV names
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
    if "timestamp" in df.keys():
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)
    else: 
        return pd.DataFrame()
    return df


def main():
    """
    Main entrypoint when run as a script.

    - Instantiates a REST client (using POLYGON_API_KEY).
    - Retrieves all crypto tickers.
    - For each ticker, fetches OHLCV data across a wide date range,
      computes the first and last available dates and the number of records,
      and collects these summaries.
    - Writes the summary rows to `crypto_date_ranges.csv`.
    """
    client = RESTClient(api_key=POLYGON_API_KEY)

    # Retrieve list of crypto tickers
    print("Fetching crypto tickers from Polygon...")
    tickers = get_all_crypto_tickers(client)
    print(f"Retrieved {len(tickers)} crypto tickers.")

    output_file = "crypto_date_ranges.csv"
    rows = []

    # Iterate tickers, fetch OHLCV and summarize date ranges
    for t in tickers:
        print(f"working on ticker: {t.ticker}")
        df = fetch_crypto_ohlcv(symbol=t.ticker, start="2015-01-01", end="2025-12-31")
        if df.empty:
            print(f"No data for {t.ticker}, skipping.")
            continue
        try: 
            first_date = df['timestamp'].min().strftime('%Y-%m-%d')
            last_date = df['timestamp'].max().strftime('%Y-%m-%d')
            num_days = df.shape[0]
            rows.append([t.ticker, first_date, last_date, num_days])
            print(f"{t.ticker}: {first_date} to {last_date}, {num_days} days")
        except Exception as e:
            print(f"Error processing {t.ticker}: {e}")
            continue
        
    # Write summary CSV with ticker, first_date, last_date, num_days
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "first_date", "last_date", "num_days"])
        writer.writerows(rows)

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()