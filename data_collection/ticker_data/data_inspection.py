import pandas as pd

def order_by_start_date(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df_sorted = df.sort_values(by='num_days', ascending=False).reset_index(drop=True)
    df_sorted.to_csv(output_csv, index=False)
    return df_sorted

if __name__ == "__main__":
    df_sorted = order_by_start_date("crypto_date_ranges.csv", "crypto_date_ranges_sorted.csv")
    print("Top 50 entries after sorting by num_days")
    print(df_sorted[:50])
    print("Top 50 - 100 entries after sorting by num_days")
    print(df_sorted[50:100])

