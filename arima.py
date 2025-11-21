# ------------------------------------------------------------
# ARIMA FORECASTING SCRIPT (NO YFINANCE VERSION)
# ------------------------------------------------------------

# Auto-install dependencies
import os
os.system("pip install pandas numpy matplotlib statsmodels --quiet")

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# USER INPUT SECTION
# ------------------------------------------------------------

CSV_FILE = "stock_data.csv"    # Replace with your actual CSV filename
FORECAST_MONTHS = 12           # Forecast next 12 months

# ------------------------------------------------------------
# LOAD CSV DATA
# ------------------------------------------------------------

print(f"Loading CSV file: {CSV_FILE}")

data = pd.read_csv(CSV_FILE)

# CSV must contain Date + Close columns
data["Date"] = pd.to_datetime(data["Date"])
data = data.set_index("Date")
prices = data["Close"].dropna()

print("Data loaded successfully!")
print(prices.head())

# ------------------------------------------------------------
# GRAPH 1 — PRICE TREND
# ------------------------------------------------------------

plt.figure(figsize=(10,4))
plt.plot(prices, marker="o")
plt.title("Stock Price Trend")
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

print("Model training complete!")

# ------------------------------------------------------------
# GRAPH 2 — OVERLAP FORECAST
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
# GRAPH 3 — FUTURE FORECAST
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
