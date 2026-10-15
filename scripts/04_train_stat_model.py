import os
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

def train_sarimax_benchmark(df_path="data/processed/features.parquet"):
    print("Loading features for Store 44 - GROCERY I...")
    df = pd.read_parquet(df_path)
    
    # Filter single high-volume series and sort chronologically
    series = df[(df['store_nbr'] == 44) & (df['family'] == 'GROCERY I')].copy()
    series['date'] = pd.to_datetime(series['date'])
    series = series.sort_values('date').drop_duplicates(subset=['date']).set_index('date')
    
    # Explicitly set daily time-series frequency for statsmodels
    series = series.asfreq('D')
    
    # Fill any potential missing values created by asfreq
    series['sales'] = series['sales'].fillna(0)
    series['log_sales'] = np.log1p(series['sales'])
    
    exog_cols = ['onpromotion', 'is_national_holiday', 'dpoil']
    for col in exog_cols:
        series[col] = series[col].ffill().bfill().fillna(0)

    # Split: Train up to 2017-07-31, Validation: 2017-08-01 to 2017-08-15
    train = series[series.index < "2017-08-01"]
    val = series[(series.index >= "2017-08-01") & (series.index <= "2017-08-15")]
    
    print("Fitting SARIMAX(1,1,1)x(1,1,0)_7 model...")
    model = SARIMAX(
        train['log_sales'],
        exog=train[exog_cols],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 0, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    
    # Forecast using step count
    forecast_log = results.forecast(steps=len(val), exog=val[exog_cols])
    preds = np.expm1(forecast_log)
    preds = np.clip(preds, 0, None)
    
    # Evaluate
    actuals = val['sales'].values
    preds = preds.values if hasattr(preds, 'values') else preds
    
    mae = np.mean(np.abs(actuals - preds))
    wape = np.sum(np.abs(actuals - preds)) / np.sum(actuals)
    
    print("\n==========================================")
    print(" 📊 MODEL 2: SARIMAX (Store 44 - GROCERY I)")
    print("==========================================")
    print(f"Validation MAE  : {mae:.4f}")
    print(f"Validation WAPE : {wape:.2%}")
    print("==========================================\n")

if __name__ == "__main__":
    train_sarimax_benchmark()