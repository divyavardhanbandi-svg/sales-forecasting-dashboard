"""Generate a synthetic retail sales dataset (sales_data.csv)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

dates = pd.date_range("2022-01-01", "2025-12-31", freq="D")
regions = ["North", "South", "East", "West"]
categories = ["Electronics", "Clothing", "Home", "Grocery"]

base = {"Electronics": 900, "Clothing": 600, "Home": 450, "Grocery": 750}
region_factor = {"North": 1.0, "South": 0.85, "East": 1.15, "West": 0.95}

rows = []
for d in dates:
    day_of_year = d.dayofyear
    trend = 1 + 0.0004 * (d - dates[0]).days               # steady growth
    yearly = 1 + 0.25 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    holiday = 1.4 if d.month in (11, 12) else 1.0           # festive season spike
    weekend = 1.15 if d.dayofweek >= 5 else 1.0
    for region in regions:
        for cat in categories:
            mean = base[cat] * region_factor[region] * trend * yearly * holiday * weekend
            sales = max(0, rng.normal(mean, mean * 0.12))
            units = max(1, int(sales / rng.uniform(20, 60)))
            rows.append((d, region, cat, round(sales, 2), units))

df = pd.DataFrame(rows, columns=["date", "region", "category", "sales", "units"])
df.to_csv("sales_data.csv", index=False)
print(f"Saved sales_data.csv with {len(df):,} rows")
