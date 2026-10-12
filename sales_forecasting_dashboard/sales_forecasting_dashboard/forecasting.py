"""Forecasting helpers for the Sales Forecasting Dashboard."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

SEASON = 12  # monthly data -> yearly seasonality


def to_monthly(df: pd.DataFrame) -> pd.Series:
    """Aggregate transaction-level data to a monthly sales series."""
    s = df.set_index("date")["sales"].resample("MS").sum()
    return s


def holt_winters_forecast(train: pd.Series, steps: int):
    """Holt-Winters (additive trend + seasonality). Returns forecast, lower, upper."""
    model = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=SEASON
    ).fit(optimized=True)
    fc = model.forecast(steps)
    resid_std = np.std(model.resid, ddof=1)
    widen = np.sqrt(np.arange(1, steps + 1))
    lower = fc - 1.96 * resid_std * widen
    upper = fc + 1.96 * resid_std * widen
    return fc, lower, upper


def seasonal_naive_forecast(train: pd.Series, steps: int):
    """Baseline: repeat the values from the same month last year."""
    last_year = train.iloc[-SEASON:].values
    values = np.resize(last_year, steps)
    idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    fc = pd.Series(values, index=idx)
    return fc, None, None


MODELS = {
    "Holt-Winters": holt_winters_forecast,
    "Seasonal Naive (baseline)": seasonal_naive_forecast,
}


def evaluate(series: pd.Series, model_fn, holdout: int = 6) -> dict:
    """Train on all but the last `holdout` months, score on the holdout."""
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    fc, _, _ = model_fn(train, holdout)
    fc = pd.Series(np.asarray(fc), index=test.index)
    mae = mean_absolute_error(test, fc)
    rmse = float(np.sqrt(mean_squared_error(test, fc)))
    mape = float(np.mean(np.abs((test - fc) / test)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}
