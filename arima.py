# ----------------------------------------------------
# STREAMLIT ARIMA FORECASTING APP (MULTI-PROJECT EDITION)
# Includes:
# Project 1 – TVS Motors (2010–2018)
# Project 2 – Shriram Finance (2021–2025)
# ----------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Streamlit Page Settings
# -------------------------------
st.set_page_config(
    page_title="ARIMA Stock Forecasting App",
    layout="wide",
)

st.title("📈 ARIMA Stock Price Forecasting App")
st.markdown("### Professional financial forecasting using Python, ARIMA & Streamlit.")

# ----------------------------------------------------
# SIDEBAR PROJECT SELECTION
# ----------------------------------------------------
st.sidebar.title("📌 Select Project")

project = st.sidebar.radio(
    "Choose a Forecasting Project",
    ["Project 1 – TVS Motors (2010–2018)",
     "Project 2 – Shriram Finance (2021–2025)"]
)

# ----------------------------------------------------
# PROJECT 1 — TVS Motors
# ----------------------------------------------------
if project == "Project 1 – TVS Motors (2010–2018)":

    st.header("🚗 TVS Motors – ARIMA Forecasting Project (2010–2018)")

    symbol = "TVSMOTOR.NS"

    st.subheader("📥 Fetching Data")
    data = yf.download(symbol, start="2010-01-01", end="2019-01-01", interval="1mo")
    prices = data["Close"].dropna()
    prices.index = pd.to_datetime(prices.index)

    # ===========================
    # GRAPH 1
    # ===========================
    st.subheader("📊 1. Monthly Price Trend (2010–2018)")
    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(prices, marker="o")
    ax1.set_title("TVS Motors – Monthly Price Trend (2010–2018)")
    ax1.grid(True)
    st.pyplot(fig1)

    # ARIMA Model
    model = ARIMA(prices, order=(1,1,1))
    model_fit = model.fit()

    # ===========================
    # GRAPH 2
    # ===========================
    st.subheader("📉 2. ARIMA Forecast Overlap (2010–2018)")
    forecast_overlap = model_fit.predict(start=prices.index[1], end=prices.index[-1])

    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(prices, label="Actual")
    ax2.plot(forecast_overlap, linestyle="--", label="Forecast")
    ax2.legend()
    ax2.set_title("TVS Motors – ARIMA Overlap Forecast")
    ax2.grid(True)
    st.pyplot(fig2)

    # ===========================
    # GRAPH 3
    # ===========================
    st.subheader("🔮 3. Forecast for 2018–2019")
    forecast_future = model_fit.forecast(steps=12)
    future_index = pd.date_range(prices.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq="M")
    forecast_future = pd.Series(forecast_future, index=future_index)

    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.plot(prices, label="Actual Data")
    ax3.plot(forecast_future, linestyle="--", marker='x', label="Forecast")
    ax3.set_title("TVS Motors – Forecast (2018–2019)")
    ax3.grid(True)
    ax3.legend()
    st.pyplot(fig3)

    st.success("✅ Project 1 completed.")

# ----------------------------------------------------
# PROJECT 2 — SHRIRAM FINANCE
# ----------------------------------------------------
if project == "Project 2 – Shriram Finance (2021–2025)":

    st.header("🏦 Shriram Finance – ARIMA Forecasting Project (2021–2025)")

    symbol = "SHRIRAMFIN.NS"

    st.subheader("📥 Fetching Data")
    data = yf.download(symbol, start="2021-01-01", end="2025-01-01", interval="1mo")
    prices = data["Close"].dropna()
    prices.index = pd.to_datetime(prices.index)

    # ===========================
    # GRAPH 1
    # ===========================
    st.subheader("📊 1. Monthly Price Trend (2021–2025)")
    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(prices, marker="o")
    ax1.set_title("Shriram Finance – Monthly Price Trend (2021–2025)")
    ax1.grid(True)
    st.pyplot(fig1)

    # ARIMA Model
    model = ARIMA(prices, order=(1,1,1))
    model_fit = model.fit()

    # ===========================
    # GRAPH 2
    # ===========================
    st.subheader("📉 2. ARIMA Forecast Overlap (2021–2025)")
    forecast_overlap = model_fit.predict(start=prices.index[1], end=prices.index[-1])

    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(prices, label="Actual")
    ax2.plot(forecast_overlap, linestyle="--", label="Forecast")
    ax2.legend()
    ax2.set_title("Shriram Finance – ARIMA Overlap Forecast")
    ax2.grid(True)
    st.pyplot(fig2)

    # ===========================
    # GRAPH 3
    # ===========================
    st.subheader("🔮 3. Forecast for 2025–2026")
    forecast_future = model_fit.forecast(steps=12)
    future_index = pd.date_range(prices.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq="M")
    forecast_future = pd.Series(forecast_future, index=future_index)

    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.plot(prices, label="Actual Data")
    ax3.plot(forecast_future, linestyle="--", marker='x', label="Forecast")
    ax3.set_title("Shriram Finance – Forecast (2025–2026)")
    ax3.grid(True)
    ax3.legend()
    st.pyplot(fig3)

    st.success("✅ Project 2 completed.")

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.markdown("---")
st.markdown("Made with ❤️ using Python, ARIMA & Streamlit. Upload this file to your GitHub repo & deploy on Streamlit Cloud.")
