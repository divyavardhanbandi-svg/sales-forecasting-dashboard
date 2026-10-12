"""Sales Forecasting Dashboard  --  run with:  streamlit run app.py"""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting import MODELS, SEASON, evaluate, to_monthly

st.set_page_config(page_title="Sales Forecasting Dashboard", page_icon="📈", layout="wide")
st.title("📈 Sales Forecasting Dashboard")


# ---------- Data loading ----------
@st.cache_data
def load_data(source) -> pd.DataFrame:
    df = pd.read_csv(source, parse_dates=["date"])
    missing = {"date", "sales"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}")
    return df.dropna(subset=["date", "sales"]).sort_values("date")


st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload sales CSV (date, sales, region, category)", type="csv")

try:
    if uploaded is not None:
        data = load_data(uploaded)
    elif os.path.exists("sales_data.csv"):
        data = load_data("sales_data.csv")
    else:
        st.warning("No data found. Run `python generate_data.py` or upload a CSV.")
        st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read data: {e}")
    st.stop()

# ---------- Filters ----------
st.sidebar.header("Filters")
for col in ("region", "category"):
    if col in data.columns:
        options = sorted(data[col].dropna().unique())
        chosen = st.sidebar.multiselect(col.title(), options, default=options)
        data = data[data[col].isin(chosen)]

if data.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---------- Forecast settings ----------
st.sidebar.header("Forecast")
model_name = st.sidebar.selectbox("Model", list(MODELS))
horizon = st.sidebar.slider("Forecast horizon (months)", 3, 24, 12)

series = to_monthly(data)
if len(series) < 2 * SEASON:
    st.error(f"Need at least {2 * SEASON} months of data to forecast (found {len(series)}).")
    st.stop()

# ---------- KPIs ----------
last_month, prev_month = series.iloc[-1], series.iloc[-2]
yoy = (last_month / series.iloc[-13] - 1) * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Sales", f"${series.sum():,.0f}")
c2.metric("Last Month", f"${last_month:,.0f}", f"{(last_month / prev_month - 1) * 100:+.1f}% MoM")
c3.metric("Year-over-Year", f"{yoy:+.1f}%")
c4.metric("Avg Monthly Sales", f"${series.mean():,.0f}")

# ---------- Forecast ----------
model_fn = MODELS[model_name]
fc, lower, upper = model_fn(series, horizon)

fig = go.Figure()
fig.add_trace(go.Scatter(x=series.index, y=series, name="Actual", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=fc.index, y=fc, name="Forecast", line=dict(color="#ff7f0e", dash="dash")))
if lower is not None:
    fig.add_trace(go.Scatter(x=fc.index, y=upper, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.index, y=lower, fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(255,127,14,0.2)", name="95% interval"))
fig.update_layout(title=f"Monthly Sales Forecast ({model_name})", xaxis_title="Month",
                  yaxis_title="Sales ($)", hovermode="x unified", height=450)
st.plotly_chart(fig, width='stretch')

# ---------- Model accuracy ----------
left, right = st.columns(2)
with left:
    st.subheader("Model accuracy (last 6 months held out)")
    scores = pd.DataFrame({name: evaluate(series, fn) for name, fn in MODELS.items()}).T
    st.dataframe(scores.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE (%)": "{:.2f}"}))

with right:
    st.subheader("Forecast table")
    table = pd.DataFrame({"Month": fc.index.strftime("%b %Y"), "Forecast ($)": fc.values.round(0)})
    if lower is not None:
        table["Lower ($)"] = lower.values.round(0)
        table["Upper ($)"] = upper.values.round(0)
    st.dataframe(table, hide_index=True)
    st.download_button("Download forecast CSV", table.to_csv(index=False), "forecast.csv", "text/csv")

# ---------- Breakdowns ----------
if {"category", "region"} <= set(data.columns):
    st.subheader("Sales breakdown")
    b1, b2 = st.columns(2)
    b1.plotly_chart(go.Figure(go.Bar(x=data.groupby("category")["sales"].sum().index,
                                     y=data.groupby("category")["sales"].sum().values)).update_layout(
        title="Sales by Category", height=350), width='stretch')
    b2.plotly_chart(go.Figure(go.Bar(x=data.groupby("region")["sales"].sum().index,
                                     y=data.groupby("region")["sales"].sum().values)).update_layout(
        title="Sales by Region", height=350), width='stretch')
