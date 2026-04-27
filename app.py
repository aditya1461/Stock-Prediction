import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
import requests

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Stock Predictor", layout="wide")

# ===============================
# SIDEBAR (FULL INFO)
# ===============================
st.sidebar.title("📊 Project Info")

st.sidebar.markdown("""
📈 Stock Market Predictor

Purpose:
Predict stock prices using Machine Learning.

🚀 Features:
- Real-time stock data  
- Price prediction  
- Future forecasting  
- USD → INR conversion  

🧠 Techniques Used:
- Linear Regression  
- Time Series Analysis  
- Data Scaling  

📡 Data Source:
Yahoo Finance API  

📊 Output:
- Actual vs Predicted Graph  
- Future Prediction Graph  
""")

# ===============================
# HEADER
# ===============================
st.title("📈 Stock Market Price Prediction")
st.markdown("### Machine Learning Based")

# ===============================
# STOCK INPUT WITH SUGGESTIONS
# ===============================
st.sidebar.header("⚙️ Settings")

stock_options = ["TSLA", "AAPL", "GOOG", "MSFT", "AMZN", "NVDA", "META"]

selected_stock = st.sidebar.selectbox("Select Stock", stock_options)
custom_stock = st.sidebar.text_input("Or Enter Custom Symbol")

stock = custom_stock if custom_stock else selected_stock

start_date = st.sidebar.date_input("Start Date", datetime.date(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# ===============================
# USD-INR RATE
# ===============================
def get_usd_to_inr():
    try:
        res = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        return res.json()["rates"]["INR"]
    except:
        return None

usd_inr = get_usd_to_inr()
today = datetime.date.today()

# ===============================
# FETCH DATA
# ===============================
if st.sidebar.button("🔍 Fetch & Predict"):

    if start_date < datetime.date(2010, 1, 1):
        st.warning("⚠️ Data before 2010 is not reliably available.")

    data = yf.download(stock, start=start_date, end=end_date)

    if data.empty:
        st.error("No data found")
        st.stop()

    # =====================================================
    # 🔥 NEW ADDITION 1: ZERODHA STYLE TOP PANEL
    # =====================================================
    latest = float(data['Close'].iloc[-1])
    prev = float(data['Close'].iloc[-2])
    change = latest - prev
    change_pct = (change / prev) * 100

    st.markdown("## 📊 Live Price Panel")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{stock}", f"${latest:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
    c2.metric("High", f"${float(data['High'].max()):.2f}")
    c3.metric("Low", f"${float(data['Low'].min()):.2f}")

    # ===============================
    # INFO CARDS
    # ===============================
    st.subheader("📊 Key Insights")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📈 Highest Price", f"${float(data['Close'].max()):.2f}")
    col2.metric("📉 Lowest Price", f"${float(data['Close'].min()):.2f}")
    col3.metric("📅 Data Points", len(data))
    col4.metric("💱 USD → INR", f"₹{usd_inr:.2f}" if usd_inr else "N/A")

    st.markdown(f"**📌 Note:** All prices are in USD ($). Current USD→INR rate as of {today}")

    # ===============================
    # DATA TABLE
    # ===============================
    st.subheader("📄 Raw Data")
    st.dataframe(data)

    # =====================================================
    # 🔥 FIXED TRADINGVIEW CANDLE CHART (MOVED HERE)
    # =====================================================
    st.markdown("## 🕯️ Trading Chart")

    tv_data = data.tail(300)

    tv_data['MA20'] = tv_data['Close'].rolling(20).mean()
    tv_data['MA50'] = tv_data['Close'].rolling(50).mean()

    fig_tv = go.Figure()

    fig_tv.add_trace(go.Candlestick(
        x=tv_data.index,
        open=tv_data['Open'],
        high=tv_data['High'],
        low=tv_data['Low'],
        close=tv_data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        increasing_fillcolor='green',
        decreasing_fillcolor='red',
        line_width=1,
        name="Candles"
    ))

    fig_tv.add_trace(go.Scatter(
        x=tv_data.index,
        y=tv_data['MA20'],
        line=dict(color='blue', width=2),
        name="MA20"
    ))

    fig_tv.add_trace(go.Scatter(
        x=tv_data.index,
        y=tv_data['MA50'],
        line=dict(color='red', width=2),
        name="MA50"
    ))

    fig_tv.update_layout(
        template="plotly_dark",
        height=650,
        xaxis_rangeslider_visible=True,
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        dragmode="zoom"
    )

    st.plotly_chart(fig_tv, use_container_width=True)

    # ===============================
    # PREPROCESSING
    # ===============================
    close = data[['Close']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    lookback = 30
    X, y = [], []

    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1])

    # ===============================
    # MODEL
    # ===============================
    model = LinearRegression()
    model.fit(X, y)

    pred = model.predict(X)

    predicted_prices = scaler.inverse_transform(pred)
    actual_prices = scaler.inverse_transform(y)

    # ===============================
    # GRAPH 1
    # ===============================
    st.subheader("📊 Actual vs Predicted Prices")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=actual_prices.flatten(),
        name="Actual Price",
        line=dict(color="lightblue", width=2)
    ))

    fig.add_trace(go.Scatter(
        y=predicted_prices.flatten(),
        name="Predicted Price",
        line=dict(color="orange", width=2)
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Time",
        yaxis_title="Price (USD)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===============================
    # FUTURE PREDICTION
    # ===============================
    st.subheader("🔮 Future Prediction (30 Days)")

    last = scaled[-lookback:].flatten()
    future = []

    for _ in range(30):
        p = model.predict([last])[0]
        future.append(p)
        last = np.append(last[1:], p)

    future = scaler.inverse_transform(np.array(future).reshape(-1, 1))

    if len(future) > 5:
        future = savgol_filter(future.flatten(), 5, 2)

    future_dates = pd.date_range(data.index[-1], periods=30)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=future_dates,
        y=future,
        name="Future Price",
        line=dict(color="lime", width=3)
    ))

    fig2.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Predicted Price (USD)"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ===============================
    # DOWNLOAD
    # ===============================
    csv = data.to_csv().encode()
    st.download_button("📥 Download Data", csv, f"{stock}.csv")