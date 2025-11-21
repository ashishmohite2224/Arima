# ------------------------------------------------------------
# ARIMA FORECASTING SCRIPT (ERROR-FREE VERSION)
# ------------------------------------------------------------

# AUTO-INSTALL MISSING LIBRARIES
import os
os.system("pip install pandas numpy matplotlib statsmodels --quiet")

# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# USER INPUT: CSV FILE
# ------------------------------------------------------------
# CSV must contain: Date, Close

CSV_FILE = "stock_data.csv"      # Change filename as needed
FORECAST_MONTHS = 12             # Forecast next 12 months

# ------------------------------------------------------------
# LOAD CSV DATA
# ------------------------------------------------------------
print(f"Loading CSV file: {CSV_FILE}")

try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print("\n❌ ERROR: CSV file not found.")
    print("Place your CSV file in the same folder as arima.py")
    exit()

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.set_index("Date")

# Extract Close Price
prices = df["Close"].dropna()

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
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# TRAIN ARIMA MODEL
# ------------------------------------------------------------
print("Training ARIMA(1,1,1) model...")

try:
    model = ARIMA(prices, order=(1,1,1))
    model_fit = model.fit()
except Exception as e:
    print("\n❌ ARIMA Model Error:", e)
    exit()

print("Model training completed!")

# ------------------------------------------------------------
# GRAPH 2 — OVERLAP FORECAST
# ------------------------------------------------------------
forecast_overlap = model_fit.predict(
    start=prices.index[1],
    end=prices.index[-1]
)

plt.figure(figsize=(10,4))
plt.plot(prices, label="Actual")
plt.plot(forecast_overlap, label="ARIMA Forecast", linestyle="--")
plt.title("ARIMA Forecast Overlap")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# GRAPH 3 — FUTURE FORECAST
# ------------------------------------------------------------
future_values = model_fit.forecast(steps=FORECAST_MONTHS)

future_index = pd.date_range(
    prices.index[-1] + pd.offsets.MonthEnd(1),
    periods=FORECAST_MONTHS,
    freq="M"
)

future_series = pd.Series(future_values, index=future_index)

plt.figure(figsize=(10,4))
plt.plot(prices, label="Historical")
plt.plot(future_series, label="Forecast", linestyle="--", marker="x")
plt.title(f"Future {FORECAST_MONTHS}-Month Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n🎉 All graphs generated successfully!")
