# Sales Forecasting Dashboard

An interactive dashboard that aggregates sales data to monthly totals, forecasts future sales, and compares models.

## Features
- Upload your own CSV or use the generated sample data
- Filter by region and category
- KPI cards: total sales, last month, MoM and YoY change
- Forecast with Holt-Winters (with 95% interval) or a Seasonal Naive baseline
- Accuracy table (MAE, RMSE, MAPE) on a 6-month holdout
- Download the forecast as CSV

## Setup
```bash
pip install -r requirements.txt
python generate_data.py      # creates sales_data.csv
streamlit run app.py
```

## CSV format
Required columns: `date`, `sales`. Optional: `region`, `category`, `units`.
At least 24 months of history is needed.

## Project structure
| File | Purpose |
|------|---------|
| `generate_data.py` | Creates synthetic daily sales data |
| `forecasting.py` | Models and evaluation metrics |
| `app.py` | Streamlit dashboard |

## Ideas to extend
Add Prophet or SARIMA, XGBoost with lag features, per-category forecasts, or deploy to Streamlit Community Cloud.
