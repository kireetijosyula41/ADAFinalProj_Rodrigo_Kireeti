from massive import RESTClient
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
client = RESTClient(POLYGON_API_KEY)

# Fetch OHLCV data without any editing to the timestamp
def fetch_crypto_ohlcv_wo_edit(symbol, start, end, timespan="day", multiplier=1):
    aggs = []
    # Fetch data from Polygon API
    for a in client.list_aggs(
        ticker=symbol,
        multiplier=multiplier,
        timespan=timespan,
        from_=start,
        to=end,
        limit=50000
    ):
        aggs.append(a)
    df = pd.DataFrame([a.__dict__ for a in aggs])
    return df


def fetch_crypto_ohlcv(symbol, start, end, timespan="day", multiplier=1):
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
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

chosen_coins = ["X:EOSUSD", "X:ZRXUSD", "X:NEOUSD", "X:TRXUSD", "X:OMGUSD", "X:BATUSD", "X:ADAUSD", "X:QTUMUSD", "X:XTZUSD", "X:DOGEUSD"]
# original_chosen_coins = ["X:GALAUSD", "X:DOGEUSD", "X:MANAUSD", "X:SHIBUSD", "X:AVAXUSD", "X:TRXUSD", "X:CVCUSD", "X:SOLUSD", "X:CHZUSD", "X:ENJUSD"]

for i in chosen_coins:
    print(f"data for chosen coin {i}")
    df_doge = fetch_crypto_ohlcv(f"{i}", "2018-03-26", "2025-12-03")
    # Save the DataFrame to a CSV file
    df_doge.to_csv(f"./coin_data/{i[2:]}.csv", index=False)
    print(df_doge.head())
    print(df_doge.tail())