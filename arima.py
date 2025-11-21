import os
os.system("pip install yfinance pandas numpy matplotlib statsmodels --quiet")

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="ARIMA Forecasting App",
    layout="wide"
)

st.title("📈 ARIMA Stock Price Forecasting App")
st.write("Professional forecasting for TVS Motors & Shriram Finance using ARIMA models.")

# ------------------------------------------------------------
# SIDEBAR - PROJECT SELECTION
# ------------------------------------------------------------
project = st.sidebar.selectbox(
    "Select Project",
    [
        "Project 1 – TVS Motors (2010–2018)",
        "Project 2 – Shriram Finance (2021–2025)"
    ]
)

# ------------------------------------------------------------
# FUNCTION: LOAD DATA
# ------------------------------------------------------------
def load_stock(symbol, start, end):
    data = yf.download(symbol, start=start, end=end, interval="1mo", progress=False)
    prices = data["Close"].dropna()
    prices.index = pd.to_datetime(prices.index)
    return prices

# ------------------------------------------------------------
# FUNCTION: BUILD ARIMA MODEL
# ------------------------------------------------------------
def build_arima(prices):
    model = ARIMA(prices, order=(1,1,1))
    fit = model.fit()
    return fit

# ------------------------------------------------------------
# PROJECT 1 — TVS Motors
# ------------------------------------------------------------
if project == "Project 1 – TVS Motors (2010–2018)":

    st.header("🚗 Project 1 – TVS Motors (2010–2018)")

    prices = load_stock("TVSMOTOR.NS", "2010-01-01", "2019-01-01")

    # -------- GRAPH 1 --------
    st.subheader("📊 Monthly Price Movement (2010–2018)")
    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(prices, marker="o")
    ax1.grid(True)
    st.pyplot(fig1)

    # Build ARIMA
    fit = build_arima(prices)

    # -------- GRAPH 2 --------
    st.subheader("📉 ARIMA Forecast Overlap")
    pred = fit.predict(start=prices.index[1], end=prices.index[-1])
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(prices, label="Actual")
    ax2.plot(pred, linestyle="--", label="Forecast")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # -------- GRAPH 3 --------
    st.subheader("🔮 Forecast for 2018–2019")
    future = fit.forecast(steps=12)
    idx = pd.date_range(prices.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq="M")
    future = pd.Series(future, index=idx)

    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.plot(prices, label="Actual")
    ax3.plot(future, label="Forecast", linestyle="--", marker="x")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)


# ------------------------------------------------------------
# PROJECT 2 — Shriram Finance
# ------------------------------------------------------------
if project == "Project 2 – Shriram Finance (2021–2025)":

    st.header("🏦 Project 2 – Shriram Finance (2021–2025)")

    prices = load_stock("SHRIRAMFIN.NS", "2021-01-01", "2025-01-01")

    # -------- GRAPH 1 --------
    st.subheader("📊 Monthly Price Movement (2021–2025)")
    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(prices, marker="o")
    ax1.grid(True)
    st.pyplot(fig1)

    # Build ARIMA
    fit = build_arima(prices)

    # -------- GRAPH 2 --------
    st.subheader("📉 ARIMA Forecast Overlap")
    pred = fit.predict(start=prices.index[1], end=prices.index[-1])
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(prices, label="Actual")
    ax2.plot(pred, linestyle="--", label="Forecast")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # -------- GRAPH 3 --------
    st.subheader("🔮 Forecast for 2025–2026")
    future = fit.forecast(steps=12)
    idx = pd.date_range(prices.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq="M")
    future = pd.Series(future, index=idx)

    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.plot(prices, label="Actual")
    ax3.plot(future, label="Forecast", linestyle="--", marker="x")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.markdown("Made with ❤️ | Perfect for GitHub & Streamlit Cloud deployment.")
