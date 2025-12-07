"""print_tickers.py

List available active cryptocurrency tickers from the Polygon REST API.

This small utility:
- Loads the POLYGON_API_KEY from the environment.
- Instantiates a Polygon REST client.
- Lists active crypto tickers (up to the set limit) and prints each ticker,
  its display name, and market, then prints a final count.

Intended as a quick script to inspect which crypto tickers are available via
the Polygon API.
"""
from massive import RESTClient
import os
from dotenv import load_dotenv

# Load environment variables (expects POLYGON_API_KEY in the environment or .env)
load_dotenv()

API_KEY = os.environ.get("POLYGON_API_KEY")

def main():
    """
    Create a REST client using POLYGON_API_KEY, request the list of active
    crypto tickers from Polygon, print each ticker's symbol, name and market,
    and finally print the total number of tickers discovered.
    """
    client = RESTClient(api_key=API_KEY)
    # Request up to 1000 active crypto tickers from Polygon
    tickers = client.list_tickers(
        market="crypto",
        active="true",
        limit=1000,
    )
    count = 0
    # Iterate the returned ticker objects and print relevant fields
    for t in tickers:
        print(t.ticker, "-", t.name, "-", t.market)
        count += 1

    # Print total number of tickers processed
    print(f"\nTotal crypto tickers: {count}")

if __name__ == "__main__":
    main()