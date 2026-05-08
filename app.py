import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from groq import Groq
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Groq Client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Stock Dashboard",
    layout="wide"
)

st.title("📈 Stock Market Dashboard & Analyzer")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Stock Settings")

ticker = st.sidebar.text_input(
    "Enter Stock Symbol",
    "AAPL"
).upper()

# Currency Toggle
currency = st.sidebar.selectbox(
    "Select Currency",
    ["USD", "INR"]
)

# Time Period
period = st.sidebar.selectbox(
    "Select Time Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3
)

#Compare Stocks
compare_stocks = st.sidebar.text_input(
    "Compare Stocks (comma separated)",
    "TSLA,MSFT"
)

compare_stocks = [
    stock.strip().upper()
    for stock in compare_stocks.split(",")
    if stock.strip() != ""
]


# ---------------- FETCH DATA ----------------
@st.cache_data
def load_data(ticker, period):
    data = yf.download(ticker, period=period)
    return data

df = load_data(ticker, period)

# Currency Conversion
usd_to_inr = 83

if currency == "INR":

    df["Open"] = df["Open"] * usd_to_inr
    df["High"] = df["High"] * usd_to_inr
    df["Low"] = df["Low"] * usd_to_inr
    df["Close"] = df["Close"] * usd_to_inr

# ---------------- ERROR HANDLING ----------------
if df.empty:
    st.error("Invalid stock symbol.")
    st.stop()

# ---------------- DATA PROCESSING ----------------

# Daily Returns
df["Daily Return"] = df["Close"].pct_change()

# Moving Averages
df["20_MA"] = df["Close"].rolling(window=20).mean()
df["50_MA"] = df["Close"].rolling(window=50).mean()

# Volatility
df["Volatility"] = df["Daily Return"].rolling(window=20).std()

# ---------------- RISK ANALYSIS ----------------

average_volatility = df["Volatility"].mean()

if average_volatility < 0.01:
    risk_level = "Low Risk 🟢"

elif average_volatility < 0.03:
    risk_level = "Medium Risk 🟡"

else:
    risk_level = "High Risk 🔴"

# ---------------- AI FINANCE CHATBOT ----------------

st.sidebar.divider()

st.sidebar.subheader("🤖 AI Finance Chatbot")

user_question = st.sidebar.text_input(
    "Ask about the stock"
)

if user_question:

    current_price = df["Close"].iloc[-1].item()
    ma20 = df["20_MA"].iloc[-1].item()
    ma50 = df["50_MA"].iloc[-1].item()
    volatility = df["Volatility"].mean().item()

    if ma20 > ma50:
        trend = "Bullish"
    else:
        trend = "Bearish"

    prompt = f"""
    Analyze this stock.

    Stock: {ticker}

    Current Price: {current_price}

    20-Day Moving Average: {ma20}

    50-Day Moving Average: {ma50}

    Volatility: {volatility}

    Trend: {trend}

    User Question:
    {user_question}

    Give short financial insight.
    """

    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="llama-3.3-70b-versatile"
)

    answer = chat_completion.choices[0].message.content

    st.sidebar.success(answer)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Developed by <b>Nandini Bhatt</b> 💻
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- METRICS ----------------

latest_price = df["Close"].iloc[-1].item()
previous_price = df["Close"].iloc[-2].item()

change = latest_price - previous_price
percent_change = (change / previous_price) * 100

highest_price = df["High"].max().item()
lowest_price = df["Low"].min().item()

col1, col2, col3 = st.columns(3)

symbol = "₹" if currency == "INR" else "$"

col1.metric(
    "Current Price",
    f"{symbol}{latest_price:.2f}",
    f"{percent_change:.2f}%"
)

col2.metric(
    "Highest Price",
    f"{symbol}{highest_price:.2f}"
)

col3.metric(
    "Lowest Price",
    f"{symbol}{lowest_price:.2f}"
)

# ---------------- PRICE CHART ----------------

st.subheader("📊 Stock Price Analysis")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df.index,
    df["Close"],
    label="Closing Price"
)

ax.plot(
    df.index,
    df["20_MA"],
    label="20 Day Moving Average"
)

ax.plot(
    df.index,
    df["50_MA"],
    label="50 Day Moving Average"
)

ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)

# ---------------- STOCK COMPARISON ----------------

st.subheader("📈 Stock Comparison")

if compare_stocks:

    fig_compare, ax_compare = plt.subplots(figsize=(12,5))

    for stock in compare_stocks:

        compare_df = yf.download(stock, period=period)

        ax_compare.plot(
            compare_df.index,
            compare_df["Close"],
            label=stock
        )

    ax_compare.set_xlabel("Date")
    ax_compare.set_ylabel("Price")
    ax_compare.legend()

    st.pyplot(fig_compare)

# ---------------- CANDLESTICK CHART ----------------

st.subheader("🕯️ Candlestick Chart")

fig_candle = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'].squeeze(),
    high=df['High'].squeeze(),
    low=df['Low'].squeeze(),
    close=df['Close'].squeeze(),
    increasing_line_color='green',
    decreasing_line_color='red'
)])

fig_candle.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    height=600
)

st.plotly_chart(fig_candle, use_container_width=True)

# ---------------- BUY / SELL SIGNAL ----------------

st.subheader("📌 Trading Signal")

if df["20_MA"].iloc[-1] > df["50_MA"].iloc[-1]:
    st.success("BUY Signal Detected 📈")
else:
    st.error("SELL Signal Detected 📉")

# ---------------- RISK ANALYSIS DISPLAY ----------------

st.subheader("⚠️ Risk Analysis")

st.warning(f"Current Risk Level: {risk_level}")

# ---------------- VOLUME CHART ----------------

st.subheader("📦 Trading Volume")

fig2, ax2 = plt.subplots(figsize=(12, 4))

volume_data = df["Volume"].squeeze()

ax2.plot(volume_data.index, volume_data)

ax2.set_xlabel("Date")
ax2.set_ylabel("Volume")

st.pyplot(fig2)

st.pyplot(fig2)

# ---------------- DAY-WISE STOCK DATA ----------------

st.subheader("📅 Day-wise Stock Data")

st.dataframe(
    df[["Open", "High", "Low", "Close", "Volume"]]
)

# ---------------- DATA ANALYSIS ----------------

st.subheader("📑 Stock Data Analysis")

analysis_df = pd.DataFrame({
    "Average Closing Price": [df["Close"].mean().item()],
    "Maximum Closing Price": [df["Close"].max().item()],
    "Minimum Closing Price": [df["Close"].min().item()],
    "Average Volume": [df["Volume"].mean().item()],
    "Average Volatility": [df["Volatility"].mean()]
})

st.dataframe(analysis_df)

# ---------------- RAW DATA ----------------

with st.expander("View Raw Dataset"):
    st.dataframe(df.tail(50))

