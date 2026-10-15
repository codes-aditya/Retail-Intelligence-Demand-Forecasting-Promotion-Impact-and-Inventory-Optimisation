import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

def train_global_xgboost(df_path="data/processed/features.parquet"):
    print("Loading feature matrix for XGBoost...")
    df = pd.read_parquet(df_path)
    
    feature_cols = [
        'store_nbr', 'onpromotion', 'dpoil', 'is_national_holiday',
        'dayofweek', 'is_weekend', 'day', 'month', 'payday',
        'sales_lag_16', 'sales_lag_21', 'sales_lag_28', 'sales_lag_35',
        'rolling_mean_sales_7', 'rolling_mean_sales_14', 'rolling_mean_sales_30'
    ]
    
    # Categorical handling for store numbers
    df['store_nbr'] = df['store_nbr'].astype('category')
    
    # Chronological Train / Validation Split
    train_df = df[df['date'] < "2017-08-01"].dropna(subset=feature_cols)
    val_df = df[(df['date'] >= "2017-08-01") & (df['date'] <= "2017-08-15")].dropna(subset=feature_cols)
    
    X_train, y_train = train_df[feature_cols], np.log1p(train_df['sales'])
    X_val, y_val = val_df[feature_cols], np.log1p(val_df['sales'])
    
    print(f"Training rows: {len(X_train):,} | Validation rows: {len(X_val):,}")
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method='hist',
        early_stopping_rounds=30,
        random_state=42
    )
    
    print("Fitting XGBoost Regressor...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    
    # Generate Predictions
    val_preds_log = model.predict(X_val)
    val_preds = np.clip(np.expm1(val_preds_log), 0, None)
    
    actuals = val_df['sales']
    mae = np.mean(np.abs(actuals - val_preds))
    wape = np.sum(np.abs(actuals - val_preds)) / np.sum(actuals)
    
    print("\n==========================================")
    print(" 🚀 MODEL 3: GLOBAL XGBOOST REGRESSOR")
    print("==========================================")
    print(f"Validation MAE  : {mae:.4f}")
    print(f"Validation WAPE : {wape:.2%}")
    print("==========================================\n")
    
    # Save Outputs for Script 6 & Dashboard
    os.makedirs("models", exist_ok=True)
    val_df['pred_sales'] = val_preds
    val_df.to_parquet("data/processed/val_predictions.parquet")
    
    with open("models/xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    print("✅ Model saved to models/xgb_model.pkl")
    print("✅ Predictions saved to data/processed/val_predictions.parquet")

if __name__ == "__main__":
    train_global_xgboost()