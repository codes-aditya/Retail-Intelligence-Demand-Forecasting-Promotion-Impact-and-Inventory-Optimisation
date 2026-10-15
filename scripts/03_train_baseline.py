import os
import pandas as pd
import numpy as np

def evaluate_baseline(df_path="data/processed/features.parquet"):
    print("Loading feature matrix...")
    df = pd.read_parquet(df_path)
    
    # Validation slice (last 16 days of training data: 2017-08-01 to 2017-08-15)
    val_start = "2017-08-01"
    val_end = "2017-08-15"
    
    val_df = df[(df['date'] >= val_start) & (df['date'] <= val_end)].copy()
    
    # Use sales_lag_16 as the seasonal naive point forecast
    val_df['pred_baseline'] = val_df['sales_lag_16']
    
    actuals = val_df['sales'].fillna(0)
    preds = val_df['pred_baseline'].fillna(0)
    
    # Calculate MAE and WAPE
    mae = np.mean(np.abs(actuals - preds))
    wape = np.sum(np.abs(actuals - preds)) / np.sum(actuals)
    
    print("\n==========================================")
    print(" 📊 MODEL 1: SEASONAL-NAÏVE BASELINE")
    print("==========================================")
    print(f"Validation Window: {val_start} to {val_end}")
    print(f"Validation MAE  : {mae:.4f}")
    print(f"Validation WAPE : {wape:.2%}")
    print("==========================================\n")
    
    return val_df

if __name__ == "__main__":
    evaluate_baseline()