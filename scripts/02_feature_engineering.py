import os
import pandas as pd
import numpy as np

def create_leakage_safe_features(df):
    """
    Computes time-series features (lags, rolling averages, calendar signals)
    strictly without look-ahead leakage.
    """
    print("Generating leakage-safe features...")
    
    # Ensure dates are datetime objects and sorted chronologically
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)

    # 1. Sales Lag Features (minimum lag of 16 days to align with 16-day test horizon)
    for lag in [16, 21, 28, 35]:
        print(f"  Generating sales_lag_{lag}...")
        df[f'sales_lag_{lag}'] = df.groupby(['store_nbr', 'family'])['sales'].shift(lag)

    # 2. Rolling Window Features (shifted by 16 days to prevent leakage)
    for window in [7, 14, 30]:
        print(f"  Generating rolling_mean_sales_{window}...")
        df[f'rolling_mean_sales_{window}'] = (
            df.groupby(['store_nbr', 'family'])['sales']
            .transform(lambda x: x.shift(16).rolling(window).mean())
        )

    # 3. Calendar & Temporal Features
    print("  Generating calendar signals...")
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['payday'] = ((df['day'] == 15) | (df['date'].dt.is_month_end)).astype(int)

    return df

if __name__ == "__main__":
    input_path = "data/processed/favorita_cleaned.parquet"
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Could not find {input_path}. Please run `python scripts/01_load_data.py` first."
        )

    print(f"Loading cleaned data from {input_path}...")
    df_raw = pd.read_parquet(input_path)

    # Generate feature matrix
    df_features = create_leakage_safe_features(df_raw)

    # Save output
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/features.parquet"
    df_features.to_parquet(output_path, index=False)
    print(f"✅ Successfully saved feature matrix to {output_path}")