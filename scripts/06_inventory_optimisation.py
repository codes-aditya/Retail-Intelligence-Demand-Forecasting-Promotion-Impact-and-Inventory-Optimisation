import os
import pandas as pd
import numpy as np
import scipy.stats as stats

def calculate_inventory_recommendations(
    pred_path="data/processed/val_predictions.parquet",
    lead_time=3,
    holding_cost=0.15,
    stockout_penalty=2.50
):
    print("Loading validation predictions...")
    df = pd.read_parquet(pred_path)
    
    # 1. Compute Standard Deviation of Residuals with clipping
    df['residual'] = df['sales'] - df['pred_sales']
    
    # Groupwise std calculation with minimum sample fallback
    std_df = df.groupby(['store_nbr', 'family'])['residual'].std().reset_index()
    std_df.rename(columns={'residual': 'sigma_daily'}, inplace=True)
    
    # Fallback to global standard deviation if group std is NaN or zero
    global_sigma = df['residual'].std()
    std_df['sigma_daily'] = std_df['sigma_daily'].fillna(global_sigma)
    std_df['sigma_daily'] = std_df['sigma_daily'].replace(0, global_sigma)
    
    df = df.drop(columns=['sigma_daily'], errors='ignore').merge(std_df, on=['store_nbr', 'family'], how='left')
    
    # 2. Optimal Service Level (Newsvendor critical ratio)
    target_service_level = stockout_penalty / (stockout_penalty + holding_cost)
    z_score = stats.norm.ppf(target_service_level)
    
    print(f"Target Service Level: {target_service_level:.1%} (z-score: {z_score:.2f})")
    
    # 3. Calculate Lead Time Demand & Safety Stock
    df['sigma_lead_time'] = df['sigma_daily'] * np.sqrt(lead_time)
    df['safety_stock'] = np.ceil(z_score * df['sigma_lead_time'])
    
    # Cap safety stock at 3x average forecasted lead time demand to prevent outlier inflation
    df = df.sort_values(['store_nbr', 'family', 'date'])
    df['lead_time_demand'] = (
        df.groupby(['store_nbr', 'family'])['pred_sales']
        .transform(lambda x: x.rolling(lead_time, min_periods=1).sum())
    )
    df['safety_stock'] = np.minimum(df['safety_stock'], np.ceil(3 * df['lead_time_demand']))
    
    # Reorder Point (ROP) = Forecasted Lead Time Demand + Safety Stock
    df['reorder_point'] = np.ceil(df['lead_time_demand'] + df['safety_stock'])
    
    # Expected Daily Holding Cost
    df['daily_holding_cost'] = df['safety_stock'] * holding_cost
    
    # 4. Save Optimization Output
    os.makedirs("reports", exist_ok=True)
    output_path = "data/processed/inventory_recommendations.parquet"
    df.to_parquet(output_path, index=False)
    
    print("\n==========================================")
    print(" 📦 INVENTORY POLICY SUMMARY")
    print("==========================================")
    print(f"Total Portfolio Safety Stock Units : {df['safety_stock'].sum():,.0f}")
    print(f"Total Daily Holding Cost           : ${df['daily_holding_cost'].sum():,.2f}")
    print("==========================================\n")
    print(f"✅ Recommendations saved to {output_path}")

if __name__ == "__main__":
    calculate_inventory_recommendations()