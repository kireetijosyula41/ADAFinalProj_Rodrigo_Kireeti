import csv
from datetime import datetime
from dotenv import load_dotenv
from massive import RESTClient
import pandas as pd
import os

load_dotenv()
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

client = RESTClient(POLYGON_API_KEY)


def get_all_crypto_tickers(client):
    """
    Return a list of all crypto tickers as Ticker objects.
    """
    tickers_gen = client.list_tickers(
        market="crypto",
        active="true",
        limit=1000,
    )
    return list(tickers_gen)

def fetch_crypto_ohlcv(symbol, start, end, timespan="day", multiplier=1):
    # Fetch data from Polygon API
    aggs = []
    print("HERE1")
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
    print("HERE2")
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
    print("HERE3")
    if "timestamp" in df.keys():
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)
    else: 
        return pd.DataFrame()
    return df


def main():
    client = RESTClient(api_key=POLYGON_API_KEY)

    print("Fetching crypto tickers from Polygon...")
    tickers = get_all_crypto_tickers(client)
    print(f"Retrieved {len(tickers)} crypto tickers.")

    output_file = "crypto_date_ranges.csv"
    rows = []

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
        
    # Save CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "first_date", "last_date", "num_days"])
        writer.writerows(rows)

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()