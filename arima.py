# ------------------------------------------------------------
# ARIMA FORECASTING SCRIPT (FINAL FIX: IMPORT AFTER INSTALL)
# ------------------------------------------------------------

import os

# Install required packages BEFORE importing
os.system("pip install pandas numpy matplotlib statsmodels --quiet")

# Import AFTER installation (fixes your issue permanently)
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")            # <-- prevents backend errors
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# USER INPUT
# ------------------------------------------------------------

CSV_FILE = "stock_data.csv"   # Must have Date + Close columns
FORECAST_MONTHS = 12

# ------------------------------------------------------------
# LOAD CSV FILE
# ------------------------------------------------------------

df = pd.read_csv(CSV_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")
prices = df["Close"]

# ------------------------------------------------------------
# GRAPH 1 — PRICE TREND
# ------------------------------------------------------------

plt.figure(figsize=(10,4))
plt.plot(prices, marker="o")
plt.title("Stock Price Trend")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.savefig("trend.png")     # works even without display
plt.close()

# ------------------------------------------------------------
# TRAIN ARIMA MODEL
# ------------------------------------------------------------

model = ARIMA(prices, order=(1,1,1))
fit = model.fit()

# ------------------------------------------------------------
# GRAPH 2 — OVERLAP FORECAST
# ------------------------------------------------------------

pred = fit.predict(start=prices.index[1], end=prices.index[-1])

plt.figure(figsize=(10,4))
plt.plot(prices, label="Actual")
plt.plot(pred, label="Forecast", linestyle="--")
plt.legend()
plt.grid(True)
plt.savefig("overlap.png")
plt.close()

# ------------------------------------------------------------
# GRAPH 3 — FUTURE FORECAST
# ------------------------------------------------------------

future = fit.forecast(steps=FORECAST_MONTHS)
future_index = pd.date_range(
    prices.index[-1] + pd.offsets.MonthEnd(1),
    periods=FORECAST_MONTHS,
    freq="M"
)

future = pd.Series(future, index=future_index)

plt.figure(figsize=(10,4))
plt.plot(prices, label="Historical")
plt.plot(future, label="Future Forecast", linestyle="--")
plt.legend()
plt.grid(True)
plt.savefig("forecast.png")
plt.close()

print("All graphs created successfully!")
