# =====================
# IMPORT
# =====================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import streamlit as st
import yfinance as yf

from datetime import datetime
from PIL import Image
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator


# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Swing Trading Scanner",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
.header-box {
    background: linear-gradient(135deg, #0e1117, #151b2c);
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 25px;
}
[data-testid="stMetric"] {
    background-color: #0e1117;
    padding: 12px;
    border-radius: 10px;
}
.stProgress > div > div {
    background-color: #00c176;
}
</style>
""", unsafe_allow_html=True)


# =====================
# HEADER IMAGE
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo = Image.open(os.path.join(BASE_DIR, "logo.png"))

col1, col2 = st.columns([1, 7])

with col1:
    st.markdown(
        "<div style='background-color:white;padding:10px;border-radius:12px;'>",
        unsafe_allow_html=True
    )
    st.image(logo, width=140)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <h1 style="color:white;font-weight:800;margin-bottom:4px;">
        📈 Swing Trading Scanner
    </h1>
    <p style="color:#9aa0a6;font-size:14px;">
        Realtime Daily Market Screening • Indonesia Stock Exchange
    </p>
    """, unsafe_allow_html=True)

st.caption(
    "Screener ini hanya alat bantu teknikal. "
    "Bukan rekomendasi beli/jual. DYOR & gunakan risk management."
)


# =====================
# CACHE
# =====================
st.cache_data(ttl=60 * 60 * 24)


# =====================
# LOAD TICKERS
# =====================
@st.cache_data
def load_idx_tickers():
    try:
        df = pd.read_csv("idx_tickers.csv")
        if df.shape[1] == 1:
            df.columns = ["Kode"]
    except:
        df = pd.read_csv("idx_tickers.csv", header=None, names=["Kode"])

    return (df["Kode"].astype(str).str.strip() + ".JK").tolist()


TICKERS = load_idx_tickers()

PERIOD = "6mo"
INTERVAL = "1d"

TP_PCT = st.sidebar.slider("Take Profit (%)", 3, 20, 5)
SL_PCT = st.sidebar.slider("Stop Loss (%)", 2, 10, 3)
fake_rebound_filter = st.sidebar.checkbox("Filter Fake Rebound", False)


# =====================
# SIDEBAR – HARGA WAJAR
# =====================
st.sidebar.markdown("## 💎 Harga Wajar Saham")
fair_search = st.sidebar.text_input(
    "Cari Kode Saham", placeholder="BBRI / BBCA"
).upper()
show_fair_value = st.sidebar.checkbox("Tampilkan Analisa Harga Wajar", True)


# =====================
# HELPER
# =====================
def S(x):
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return x.astype(float)


def distance_to_ma(close, period):
    if len(close) < period:
        return np.nan
    ma = close.rolling(period).mean().iloc[-1]
    return abs(close.iloc[-1] - ma) / ma * 100


# =====================
# FAIR VALUE SIMPLE
# =====================
def calculate_fair_value_simple(ticker, price):
    try:
        info = yf.Ticker(ticker).info
        eps = info.get("trailingEps", None)

        if eps is None or eps <= 0:
            return None

        per = 12
        fair = eps * per
        margin = (fair - price) / price * 100

        return {
            "Fair_Price": round(fair, 2),
            "Margin": round(margin, 1),
            "EPS": round(eps, 2),
            "PER": per
        }
    except:
        return None


# =====================
# TECHNICAL LOGIC
# =====================
def detect_trend(close):
    ema20 = EMAIndicator(close, 20).ema_indicator()
    ema50 = EMAIndicator(close, 50).ema_indicator()
    return "Bullish" if ema20.iloc[-1] > ema50.iloc[-1] else "Bearish"


def detect_macd_signal(close):
    macd = MACD(close)
    if macd.macd().iloc[-1] > macd.macd_signal().iloc[-1]:
        return "Golden Cross"
    return "Normal"


def detect_zone(df):
    support = S(df["Low"]).rolling(20).min().iloc[-1]
    resistance = S(df["High"]).rolling(20).max().iloc[-1]
    price = S(df["Close"]).iloc[-1]

    if price <= support * 1.03:
        return "BUY ZONE"
    if price >= resistance * 0.97:
        return "SELL ZONE"
    return "MID"


# =====================
# PROCESS DATA
# =====================
rows = []

with st.spinner("⏳ Mengambil data saham IDX..."):
    for t in TICKERS:
        try:
            df = yf.download(t, period=PERIOD, interval=INTERVAL, progress=False)
            if df.empty or len(df) < 60:
                continue

            close = S(df["Close"])
            price = close.iloc[-1]
            rsi = RSIIndicator(close, 14).rsi().iloc[-1]

            rows.append({
                "Kode": t.replace(".JK", ""),
                "Harga": round(price, 2),
                "Trend": detect_trend(close),
                "Zone": detect_zone(df),
                "MACD": detect_macd_signal(close),
                "RSI": round(rsi, 1),
                "TP": round(price * (1 + TP_PCT / 100), 2),
                "SL": round(price * (1 - SL_PCT / 100), 2),
                "_df": df.copy()
            })
        except:
            pass


df = pd.DataFrame(rows)


# =====================
# SIDEBAR FAIR VALUE
# =====================
if show_fair_value and fair_search:
    ticker = fair_search + ".JK"
    row = df[df["Kode"] == fair_search]

    st.sidebar.markdown("---")

    if not row.empty:
        price = row.iloc[0]["Harga"]
        fv = calculate_fair_value_simple(ticker, price)

        st.sidebar.metric("Harga Saat Ini", price)

        if fv:
            st.sidebar.metric("Harga Wajar", fv["Fair_Price"], f"{fv['Margin']}%")

            if fv["Margin"] > 20:
                st.sidebar.success("🟢 Undervalued")
            elif fv["Margin"] < -10:
                st.sidebar.error("🔴 Overvalued")
            else:
                st.sidebar.warning("🟡 Fair Value")


# =====================
# TABLE OUTPUT
# =====================
st.subheader("📊 Hasil Scanner")
st.dataframe(df.drop(columns=["_df"]), use_container_width=True, hide_index=True)


# =====================
# FOOTER
# =====================
st.caption(
    f"Update otomatis harian • Last update: "
    f"{datetime.now().strftime('%d %b %Y %H:%M')}"
)
