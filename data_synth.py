import pandas as pd
import numpy as np
from config import SupplyChainConfig
# Set seed for reproducibility
np.random.seed(42)

# Generate date range for 3 years (daily data)
dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")

# Define SKUs
skus = ["SKU_A", "SKU_B", "SKU_C"]

# Function to create realistic demand with trend, seasonality, and noise
def generate_demand(dates, base_demand, trend_factor, seasonality_factor, noise_level):
    days = np.arange(len(dates))
    trend = trend_factor * days / 365
    seasonality = seasonality_factor * np.sin(2 * np.pi * days / 365)
    noise = np.random.normal(0, noise_level, len(dates))
    demand = base_demand + trend + seasonality + noise
    demand = np.maximum(0, demand.astype(int))  # Ensure no negative demand
    return demand

# Create demand forecast dataset
demand_data = []
for sku in skus:
    if sku == "SKU_A":
        demand = generate_demand(dates, base_demand=50, trend_factor=5, seasonality_factor=20, noise_level=10)
    elif sku == "SKU_B":
        demand = generate_demand(dates, base_demand=80, trend_factor=2, seasonality_factor=15, noise_level=15)
    else:  # SKU_C
        demand = generate_demand(dates, base_demand=30, trend_factor=3, seasonality_factor=10, noise_level=5)
    
    df_temp = pd.DataFrame({"date": dates, "sku": sku, "demand": demand})
    demand_data.append(df_temp)

demand_forecast_df = pd.concat(demand_data).reset_index(drop=True)

# Create inventory optimisation dataset (simulate stock levels, lead time, reorder points)
inventory_data = []
for sku in skus:
    if sku == "SKU_A":
        lead_time = 7
        reorder_point = 150
    elif sku == "SKU_B":
        lead_time = 10
        reorder_point = 200
    else:  # SKU_C
        lead_time = 5
        reorder_point = 100
    
    daily_demand = demand_forecast_df[demand_forecast_df["sku"] == sku]["demand"].values
    stock_level = [reorder_point * 2]  # Initial stock
    
    for d in daily_demand:
        new_stock = stock_level[-1] - d
        if new_stock <= reorder_point:
            new_stock += reorder_point * 2  # Replenishment
        stock_level.append(new_stock)
    
    stock_level = stock_level[:-1]  # Remove last extra
    
    df_temp = pd.DataFrame({
        "date": dates,
        "SKU": sku,
        "demand": daily_demand,
        "stock_level": stock_level,
        "lead_time": lead_time,
        "reorder_point": reorder_point
    })
    inventory_data.append(df_temp)

inventory_df = pd.concat(inventory_data).reset_index(drop=True)


cfg = SupplyChainConfig()
demand_forecast_df.to_csv(cfg.DEMAND_FORECAST_FILE, index=False)
inventory_df.to_csv(cfg.INVENTORY_OPTIMIZATION_FILE, index=False)
print(f"Wrote {cfg.DEMAND_FORECAST_FILE} and {cfg.INVENTORY_OPTIMIZATION_FILE}")