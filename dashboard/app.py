import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Retail Intelligence & Inventory Planning",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Retail-Intelligence Forecasting & Inventory Dashboard")
st.markdown("*(Corporation Favorita Time-Series Forecasting & Newsvendor Inventory Optimization)*")

@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/inventory_recommendations.parquet")
    return df

df = load_data()

# Sidebar Controls
st.sidebar.header("🎯 Filter Options")
selected_store = st.sidebar.selectbox("Select Store", sorted(df['store_nbr'].unique()))
selected_family = st.sidebar.selectbox("Select Product Family", sorted(df['family'].unique()))

st.sidebar.header("⚙️ Inventory Cost Parameters")
lead_time = st.sidebar.slider("Lead Time (Days)", min_value=1, max_value=14, value=3)
holding_cost = st.sidebar.number_input("Holding Cost ($/unit/day)", value=0.15, step=0.01)
stockout_penalty = st.sidebar.number_input("Stockout Penalty ($/unit)", value=2.50, step=0.10)

# Filter Data
filtered_df = df[(df['store_nbr'] == selected_store) & (df['family'] == selected_family)].copy()

# Recalculate Inventory Policy dynamically based on sidebar parameters
target_service_level = stockout_penalty / (stockout_penalty + holding_cost)
z_score = stats.norm.ppf(target_service_level)

filtered_df['dynamic_safety_stock'] = np.ceil(z_score * filtered_df['sigma_daily'] * np.sqrt(lead_time))
filtered_df['dynamic_rop'] = np.ceil(filtered_df['pred_sales'] * lead_time + filtered_df['dynamic_safety_stock'])

# Top KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Target Service Level", f"{target_service_level:.1%}")
col2.metric("Recommended Safety Stock", f"{filtered_df['dynamic_safety_stock'].iloc[-1]:.0f} units")
col3.metric("Reorder Point (ROP)", f"{filtered_df['dynamic_rop'].iloc[-1]:.0f} units")
col4.metric("Est. Daily Holding Cost", f"${filtered_df['dynamic_safety_stock'].iloc[-1] * holding_cost:.2f}")

st.markdown("---")

# Chart 1: Actual vs Forecast
st.subheader(f"📈 Sales Forecast vs Actuals — Store {selected_store} ({selected_family})")
fig = go.Figure()
fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['sales'], mode='lines+markers', name='Actual Sales'))
fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['pred_sales'], mode='lines+markers', name='XGBoost Forecast', line=dict(dash='dash', color='orange')))
fig.update_layout(xaxis_title="Date", yaxis_title="Units Sold", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# Chart 2: Inventory Policy Visualization
st.subheader("🛡️ Dynamic Reorder Point & Safety Stock Profile")
fig_inv = go.Figure()
fig_inv.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['dynamic_rop'], mode='lines', name='Reorder Point (ROP)', line=dict(color='red')))
fig_inv.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['dynamic_safety_stock'], mode='lines', name='Safety Stock', line=dict(color='green', dash='dot')))
fig_inv.update_layout(xaxis_title="Date", yaxis_title="Units", hovermode="x unified")
st.plotly_chart(fig_inv, use_container_width=True)

# Table Preview
st.subheader("📋 Policy Recommendations Data Table")
st.dataframe(filtered_df[['date', 'store_nbr', 'family', 'sales', 'pred_sales', 'dynamic_safety_stock', 'dynamic_rop']].tail(15))