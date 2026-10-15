# 01. Business Problem & Inventory Policy Specification

## 1. Project Objectives
* **Target Variable:** Daily units sold (`units_sold`).
* **Forecast Horizon:** 28 days ahead (4 weeks), updated weekly.
* **Unit of Analysis:** Product–Store level (SKU $\times$ Store ID).

## 2. Business Decision Focus
* **Primary Objective:** Translate demand forecasts into optimal reorder quantities and safety stock levels to minimize total holding and stock-out costs.
* **Primary KPI:** Weighted Absolute Percentage Error (WAPE) $< 15\%$.

## 3. Operational Inventory Rules
* **Lead Time ($L$):** 7 days (time elapsed between placing an order and receiving inventory at the store).
* **Target Service Level:** $95\%$ (aiming to satisfy customer demand $95\%$ of the time without stock-outs).
* **Holding Cost ($C_h$):** $\$0.50$ per unit per day.
* **Stock-out Penalty ($C_s$):** $\$3.00$ per unit (lost profit + customer dissatisfaction penalty).

## 4. Model Evaluation & Constraints
* **Validation Strategy:** Rolling-origin cross-validation with no future data leakage.
* **Cut-off Constraint:** Features for predicting time $t$ must only use historical data up to $t-1$.