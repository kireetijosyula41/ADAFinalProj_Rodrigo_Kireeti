"""
Build an aligned price array for all tickers with forward-filled prices.

Key idea:
- Use the UNION of all timestamps across tickers.
- For each ticker:
    - Reindex to the full union of timestamps.
    - Forward-fill missing values (internal gaps).
    - BUT: keep anything before the first real price as NaN (no fake pre-listing data).
- After that, drop any rows where at least one ticker is still NaN.
  This keeps only the time range where *all* tickers are actually tradable,
  but preserves the true day-to-day structure and momentum signals.

Output:
- price_array_aligned.npy : np.ndarray of shape (T, N) with aligned close prices
- price_array_dates.npy   : np.ndarray of dtype datetime64[ns] with the timestamps
"""
import pandas as pd
import numpy as np

TICKERS = [
    "EOSUSD", "ZRXUSD", "NEOUSD", "TRXUSD", "OMGUSD",
    "BATUSD", "ADAUSD", "QTUMUSD", "XTZUSD", "DOGEUSD",
]

DATE_COL = "timestamp"
PRICE_COL = "close"
CSV_TEMPLATE = "{ticker}.csv"  # change to "data/{ticker}.csv"


def load_ticker_df(ticker: str) -> pd.DataFrame:
    """
    Load a single ticker CSV and return a DataFrame:

        index: datetime index (sorted)
        column: PRICE_COL (float)

    Assumes CSV has at least [DATE_COL, PRICE_COL].
    """
    csv_path = f"./data_collection/coin_data/{CSV_TEMPLATE.format(ticker=ticker)}"
    df = pd.read_csv(csv_path)

    # Parse timestamp and sort
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df[[DATE_COL, PRICE_COL]].dropna(subset=[PRICE_COL])
    df = df.sort_values(DATE_COL).set_index(DATE_COL)

    # Ensure float type
    df[PRICE_COL] = df[PRICE_COL].astype(float)
    return df


def main():
    """
    Build and save an aligned price matrix for the configured TICKERS.

    Steps:
    1. Load each ticker CSV into a DataFrame indexed by datetime.
    2. Compute the union of all timestamps across tickers.
    3. Reindex each ticker to the full union, forward-fill internal gaps,
       and keep pre-listing dates as NaN.
    4. Combine into a single DataFrame and drop any rows where at least one
       ticker remains NaN (ensures all tickers are tradable in the range).
    5. Save the resulting price matrix and corresponding dates as .npy files.
    """
    # 1. Load all tickers into individual DataFrames
    dfs = {}
    for t in TICKERS:
        df = load_ticker_df(t)
        dfs[t] = df
        print(f"{t}: {df.index.min().date()} -> {df.index.max().date()} ({len(df)} rows)")

    # 2. Build the union of all timestamps
    full_index = None
    for df in dfs.values():
        if full_index is None:
            full_index = df.index
        else:
            full_index = full_index.union(df.index)

    full_index = full_index.sort_values()
    print(f"\nFull union of timestamps: {full_index.min().date()} -> {full_index.max().date()} ({len(full_index)} rows)")

    # 3. Reindex + forward-fill each ticker, but don't fabricate pre-listing data
    aligned_cols = {}
    for t, df in dfs.items():
        first_ts = df.index.min()

        # Reindex to full timeline and forward-fill
        df_ff = df.reindex(full_index).ffill()

        # Any date earlier than the first real price should stay NaN (no synthetic data)
        df_ff.loc[df_ff.index < first_ts, PRICE_COL] = np.nan

        aligned_cols[t] = df_ff[PRICE_COL]

    # 4. Combine into a single DataFrame and drop any rows with NaNs
    all_df = pd.DataFrame(aligned_cols, index=full_index)

    # Drop rows where at least one ticker still has NaN
    before_drop = len(all_df)
    all_df = all_df.dropna(how="any")
    after_drop = len(all_df)

    print(f"\nDropped {before_drop - after_drop} rows where some tickers were not yet listed.")
    print(f"Final aligned range: {all_df.index.min().date()} -> {all_df.index.max().date()} ({after_drop} rows)")
    print(f"Final shape: {all_df.shape} (T x N)")

    # 5. Save aligned price matrix and dates
    price_array = all_df.to_numpy(dtype=float)
    dates_array = all_df.index.to_numpy()  # datetime64[ns]

    np.save("price_array_aligned.npy", price_array)
    np.save("price_array_dates.npy", dates_array)

    print("\nSaved:")
    print("  price_array_aligned.npy with shape", price_array.shape)
    print("  price_array_dates.npy with", dates_array.shape[0], "timestamps")


if __name__ == "__main__":
    main()
