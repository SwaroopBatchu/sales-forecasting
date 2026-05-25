"""
generate_data.py
Generates 3 years of daily sales data across 5 product categories.
Run once before training: python src/generate_data.py
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_sales_data():
    dates      = pd.date_range(start="2021-01-01", end="2023-12-31", freq="D")
    categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Food & Beverage"]
    records    = []

    for cat in categories:
        # Base sales per category
        base = {"Electronics": 8000, "Clothing": 5000, "Home & Garden": 3500,
                "Sports": 4200, "Food & Beverage": 6000}[cat]

        for i, date in enumerate(dates):
            # Trend component
            trend = base + (i * np.random.uniform(0.5, 2.0))

            # Seasonality — stronger in Nov/Dec for Electronics & Clothing
            month = date.month
            seasonal = 1.0
            if cat in ["Electronics", "Clothing"] and month in [11, 12]:
                seasonal = 1.45
            elif cat == "Food & Beverage" and month in [6, 7, 8]:
                seasonal = 1.20
            elif cat == "Home & Garden" and month in [4, 5, 6]:
                seasonal = 1.30
            elif month in [1, 2]:
                seasonal = 0.85

            # Weekly pattern — lower on Mondays
            weekly = 0.88 if date.dayofweek == 0 else 1.0

            # Random noise
            noise = np.random.normal(1.0, 0.08)

            sales    = max(0, trend * seasonal * weekly * noise)
            units    = max(1, int(sales / np.random.uniform(15, 80)))
            discount = np.round(np.random.choice([0, 0.05, 0.10, 0.15, 0.20],
                                                  p=[0.60, 0.15, 0.12, 0.08, 0.05]), 2)
            records.append({
                "date"          : date,
                "category"      : cat,
                "sales_revenue" : np.round(sales, 2),
                "units_sold"    : units,
                "discount_pct"  : discount,
                "is_weekend"    : int(date.dayofweek >= 5),
                "month"         : month,
                "quarter"       : date.quarter,
                "day_of_week"   : date.dayofweek,
            })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_sales_data()
    df.to_csv("data/sales.csv", index=False)
    print(f"Generated {len(df):,} records → data/sales.csv")
    print(df.groupby("category")["sales_revenue"].sum().apply(lambda x: f"${x:,.0f}"))
