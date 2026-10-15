import os
import pandas as pd
import numpy as np

def load_and_clean_favorita(raw_dir="data/raw"):
    """
    Loads raw Favorita CSV files, imputes missing oil prices, filters holiday records,
    and merges store and macro features into a unified dataset.
    """
    print("Loading raw Favorita datasets...")
    train = pd.read_csv(f"{raw_dir}/train.csv", parse_dates=["date"])
    stores = pd.read_csv(f"{raw_dir}/stores.csv")
    oil = pd.read_csv(f"{raw_dir}/oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(f"{raw_dir}/holidays_events.csv", parse_dates=["date"])

    # Standardize oil price column name (handles both 'dpoi' and 'dpoil')
    oil_col = [c for c in oil.columns if c != 'date'][0]
    oil = oil.rename(columns={oil_col: 'dpoil'})
    print(f"Detected oil column as '{oil_col}', renamed to 'dpoil'.")

    print("Imputing missing oil prices...")
    full_dates = pd.date_range(train['date'].min(), train['date'].max())
    oil = oil.set_index('date').reindex(full_dates).rename_axis('date').reset_index()
    
    # Fill weekend/holiday gaps
    oil['dpoil'] = oil['dpoil'].ffill().bfill()
    oil['oil_pct_change_7d'] = oil['dpoil'].pct_change(7)

    print("Processing holiday calendar...")
    # Ignore transferred holidays
    holidays = holidays[holidays['transferred'] == False]
    holidays_national = holidays[holidays['locale'] == 'National'][['date', 'type']].drop_duplicates('date')
    holidays_national['is_national_holiday'] = 1

    print("Merging features into core sales dataset...")
    df = train.merge(stores, on="store_nbr", how="left")
    df = df.merge(oil, on="date", how="left")
    df = df.merge(holidays_national[['date', 'is_national_holiday']], on="date", how="left")
    df['is_national_holiday'] = df['is_national_holiday'].fillna(0)

    return df

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    
    # 1. Load and clean
    cleaned_df = load_and_clean_favorita(raw_dir="data/raw")
    
    # 2. Save cleaned dataset for Script 02
    output_path = "data/processed/favorita_cleaned.parquet"
    cleaned_df.to_parquet(output_path, index=False)
    print(f"✅ Successfully saved cleaned dataset to {output_path}")