"""data_inspection.py

Utilities to inspect and reorder cryptocurrency ticker date-range summaries.

This module provides a small utility to:
- Load a CSV (produced by crypto_history_ranges.py) summarizing available
  historical OHLCV ranges for crypto tickers.
- Sort tickers by the length of available history ('num_days') in descending order.
- Write the sorted results to a new CSV and return the sorted DataFrame.

Intended to be run as a script to produce and inspect the top tickers by history length.
"""
import pandas as pd

def order_by_start_date(input_csv, output_csv):
    """
    Load a CSV containing ticker date-range summaries, sort rows by 'num_days'
    in descending order, save the sorted CSV to output_csv, and return the
    sorted DataFrame.

    Parameters:
    - input_csv (str): path to the input CSV (expects a 'num_days' column)
    - output_csv (str): path to write the sorted CSV

    Returns:
    - pandas.DataFrame: DataFrame sorted by 'num_days' descending
    """
    df = pd.read_csv(input_csv)
    df_sorted = df.sort_values(by='num_days', ascending=False).reset_index(drop=True)
    df_sorted.to_csv(output_csv, index=False)
    return df_sorted

if __name__ == "__main__":
    # When executed as a script: sort the crypto_date_ranges.csv and print top entries
    df_sorted = order_by_start_date("crypto_date_ranges.csv", "crypto_date_ranges_sorted.csv")
    print("Top 50 entries after sorting by num_days")
    print(df_sorted[:50])
    print("Top 50 - 100 entries after sorting by num_days")
    print(df_sorted[50:100])

