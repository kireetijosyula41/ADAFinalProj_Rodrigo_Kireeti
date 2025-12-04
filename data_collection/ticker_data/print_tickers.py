from massive import RESTClient
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get("POLYGON_API_KEY")

def main():
    client = RESTClient(api_key=API_KEY)
    tickers = client.list_tickers(
        market="crypto",
        active="true",
        limit=1000,
    )
    count = 0
    for t in tickers:
        print(t.ticker, "-", t.name, "-", t.market)
        count += 1

    print(f"\nTotal crypto tickers: {count}")

if __name__ == "__main__":
    main()