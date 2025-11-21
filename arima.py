# ------------------------------------------------------------
# ARIMA FORECASTING SCRIPT (NO MATPLOTLIB VERSION)
# Full Fix for Import Errors
# ------------------------------------------------------------

import os

# Install required libraries BEFORE importing
os.system("pip install pandas numpy statsmodels plotly --quiet")

# Safe imports (after installation)
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# USER INPUT
# ------------------------------------------------------------
CSV_FILE = "stock_data.csv"      # Must contain Date + Close
FORECAST_MONTHS = 12             # Forecast next 12 months

# ------------------------------------------------------------
# LOAD CSV
# ------------------------------------------------------------
print("Loading CSV...")

df = pd.read_csv(CSV_FILE)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.set_index("Date")
prices = df["Close"].dropna()

print("CSV loaded successfully!")
print(prices.head())

# ------------------------------------------------------------
# GRAPH 1 — PRICE TREND (Plotly)
# ------------------------------------------------------------
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=prices.index, y=prices.values, mode="lines+markers", name="Price"))
fig1.update_layout(title="Price Trend", xaxis_title="Date", yaxis_title="Price")
fig1.write_html("trend.html")

print("trend.html saved!")

# ------------------------------------------------------------
# TRAIN ARIMA
# ------------------------------------------------------------
print("Training ARIMA(1,1,1)...")

model = ARIMA(prices, order=(1,1,1))
fit = model.fit()

print("Model trained!")

# ------------------------------------------------------------
# GRAPH 2 — FORECAST OVERLAP
# ------------------------------------------------------------
pred = fit.predict(start=prices.index[1], end=prices.index[-1])

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=prices.index, y=prices, name="Actual"))
fig2.add_trace(go.Scatter(x=pred.index, y=pred, name="Forecast", line=dict(dash="dash")))
fig2.update_layout(title="ARIMA Forecast Overlap", xaxis_title="Date", yaxis_title="Price")
fig2.write_html("overlap.html")

print("overlap.html saved!")

# ------------------------------------------------------------
# GRAPH 3 — FUTURE FORECAST (NEXT N MONTHS)
# ------------------------------------------------------------
future_values = fit.forecast(steps=FORECAST_MONTHS)

future_index = pd.date_range(
    prices.index[-1] + pd.offsets.MonthEnd(1),
    periods=FORECAST_MONTHS,
    freq="M"
)

future_series = pd.Series(future_values, index=future_index)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=prices.index, y=prices, name="Historical"))
fig3.add_trace(go.Scatter(x=future_series.index, y=future_series, name="Future Forecast", line=dict(dash="dot")))
fig3.update_layout(titl_
