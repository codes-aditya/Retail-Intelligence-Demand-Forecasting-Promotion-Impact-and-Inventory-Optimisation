import os
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

def generate_shap_plots():
    print("Loading model and feature matrix for SHAP analysis...")
    
    # 1. Load trained XGBoost model
    model_path = "models/xgb_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run script 5 first.")
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. Load feature data
    df = pd.read_parquet("data/processed/features.parquet")
    feature_cols = [
        'store_nbr', 'onpromotion', 'dpoil', 'is_national_holiday',
        'dayofweek', 'is_weekend', 'day', 'month', 'payday',
        'sales_lag_16', 'sales_lag_21', 'sales_lag_28', 'sales_lag_35',
        'rolling_mean_sales_7', 'rolling_mean_sales_14', 'rolling_mean_sales_30'
    ]
    
    # Sample validation set for fast SHAP computation
    val_df = df[(df['date'] >= "2017-08-01") & (df['date'] <= "2017-08-15")].dropna(subset=feature_cols)
    val_sample = val_df.sample(min(2000, len(val_df)), random_state=42)
    X_sample = val_sample[feature_cols]

    print("Computing SHAP values using TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # 3. Create output directory and save plot
    os.makedirs("reports/figures", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    
    output_path = "reports/figures/shap_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ SHAP summary plot saved to {output_path}")

if __name__ == "__main__":
    generate_shap_plots()