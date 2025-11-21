# ------------------------------------------------------------
# ARIMA FORECASTING SCRIPT (FULLY WORKING — NO IMPORT ERRORS)
# ------------------------------------------------------------

# AUTO-INSTALL required packages so the script never fails
import os
os.system("pip install yfinance pandas numpy matplotlib statsmodels --quiet")

# IMPORT LIBRARIES
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------

STOCK = "TVSMOTOR.NS"          # Change to SHRIRAMFIN.NS or any stock
START = "2010-01-01"
END   = "2019-01-01"
FORECAST_MONTHS = 12           # Forecast next 12 months

# ------------------------------------------------------------
# FETCH DATA
# ------------------------------------------------------------

print(f"Downloading stock data for {STOCK}...")

data = yf.download(STOCK, start=START, end=END, interval="1mo", progress=False)

if data.empty:
    print("ERROR: Could not fetch data. Check stock symbol.")
    exit()

prices = data["Close"].dropna()
prices.index = pd.to_datetime(prices.index)

print("Data downloaded successfully!")

# ------------------------------------------------------------
# GRAPH 1 — Price Trend
# ------------------------------------------------------------

plt.figure(figsize=(10,4))
plt.plot(prices, marker="o")
plt.title(f"{STOCK} — Monthly Price Trend")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# TRAIN ARIMA MODEL
# ------------------------------------------------------------

print("Training ARIMA(1,1,1) model...")

model = ARIMA(prices, order=(1,1,1))
model_fit = model.fit()

print("Model trained successfully!")

# ------------------------------------------------------------
# GRAPH 2 — Overlap Forecast
# ------------------------------------------------------------

forecast_overlap = model_fit.predict(
    start=prices.index[1],
    end=prices.index[-1]
)

plt.figure(figsize=(10,4))
plt.plot(prices, label="Actual Price")
plt.plot(forecast_overlap, label="ARIMA Forecast", linestyle="--")
plt.title("ARIMA Forecast Overlap")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# GRAPH 3 — FUTURE FORECAST (NEXT 12 MONTHS)
# ------------------------------------------------------------

future_fc = model_fit.forecast(steps=FORECAST_MONTHS)

future_index = pd.date_range(
    start=prices.index[-1] + pd.offsets.MonthEnd(1),
    periods=FORECAST_MONTHS,
    freq="M"
)

future_fc = pd.Series(future_fc, index=future_index)

plt.figure(figsize=(10,4))
plt.plot(prices, label="Historical Prices")
plt.plot(future_fc, label="Forecast", linestyle="--", marker="x")
plt.title(f"Future Forecast ({FORECAST_MONTHS} Months)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

print("All graphs generated successfully!")
