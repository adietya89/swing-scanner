# =====================
# IMPORT (BERSIH & CEPAT)
# =====================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from datetime import datetime
from PIL import Image
import os

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Swing Trading Scanner",
    layout="wide"
)

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
[data-testid="stMetric"] { background-color:#0e1117; padding:12px; border-radius:10px; }
.stProgress > div > div { background-color:#00c176; }
</style>
""", unsafe_allow_html=True)

# =====================
# HEADER
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo = Image.open(os.path.join(BASE_DIR, "logo.png"))

c1, c2 = st.columns([1, 7])
with c1:
    st.image(logo, width=130)
with c2:
    st.markdown("## 📈 Swing Trading Scanner")
    st.caption("Realtime Daily Market Screening • Indonesia Stock Exchange")
    st.caption(
        "Memasuki dinamika pasar tahun 2026, strategi swing trading memerlukan ketelitian "
        "dalam menangkap momentum harga. Aplikasi ini hanya alat bantu — DYOR tetap wajib."
    )

# =====================
# CONFIG
# =====================
PERIOD = "6mo"
INTERVAL = "1d"

TP_PCT = st.sidebar.slider("Take Profit (%)", 3, 20, 5)
SL_PCT = st.sidebar.slider("Stop Loss (%)", 2, 10, 3)
fake_rebound_filter = st.sidebar.checkbox("Filter Fake Rebound", value=False)

# =====================
# LOAD TICKER LIST (CACHE)
# =====================
@st.cache_data
def load_idx_tickers():
    df = pd.read_csv("idx_tickers.csv", header=None, names=["Kode"])
    return (df["Kode"].str.strip() + ".JK").tolist()

TICKERS = load_idx_tickers()

# =====================
# FAST DOWNLOAD (CACHE)
# =====================
@st.cache_data(ttl=60*60*24)
def load_price_data(tickers):
    data = yf.download(
        tickers,
        period=PERIOD,
        interval=INTERVAL,
        group_by="ticker",
        progress=False,
        threads=True
    )
    return data

price_data = load_price_data(TICKERS)

# =====================
# HELPER
# =====================
def S(x): return pd.to_numeric(x, errors="coerce")

def detect_trend(close):
    return "Bullish" if EMAIndicator(close,20).ema_indicator().iloc[-1] > EMAIndicator(close,50).ema_indicator().iloc[-1] else "Bearish"

def detect_macd(close):
    m = MACD(close)
    if len(close) < 35: return "Normal"
    if m.macd().iloc[-2] < m.macd_signal().iloc[-2] and m.macd().iloc[-1] > m.macd_signal().iloc[-1]:
        return "Golden Cross"
    if m.macd().iloc[-2] > m.macd_signal().iloc[-2] and m.macd().iloc[-1] < m.macd_signal().iloc[-1]:
        return "Death Cross"
    return "Normal"

def detect_candle(df):
    o,c,h,l = S(df["Open"].iloc[-1]),S(df["Close"].iloc[-1]),S(df["High"].iloc[-1]),S(df["Low"].iloc[-1])
    body = abs(c-o)
    if min(o,c)-l > body*2: return "Hammer","Bullish"
    if h-max(o,c) > body*2: return "Shooting Star","Bearish"
    return "Normal","Neutral"

# =====================
# PROCESS DATA (CEPAT)
# =====================
rows = []

with st.spinner("⏳ Mengambil data IDX..."):
    for t in TICKERS:
        try:
            df = price_data[t].dropna()
            if len(df) < 60: continue

            close = S(df["Close"])
            price = close.iloc[-1]

            rsi = RSIIndicator(close,14).rsi().iloc[-1]
            trend = detect_trend(close)
            macd = detect_macd(close)
            candle,bias = detect_candle(df)

            signal = "BUY" if (trend=="Bullish" and macd=="Golden Cross" and rsi<50) else "HOLD"

            rows.append({
                "Kode": t,
                "Harga": round(price,2),
                "Signal": signal,
                "Trend": trend,
                "Candle": candle,
                "RSI": round(rsi,1),
                "MACD": macd,
                "TP": round(price*(1+TP_PCT/100),2),
                "SL": round(price*(1-SL_PCT/100),2),
                "_df": df
            })

        except:
            continue

df = pd.DataFrame(rows)

# =====================
# FILTER UI
# =====================
st.subheader("🎯 FILTER STRATEGI")
f1,f2,f3,f4 = st.columns([2,1,1,1])

if "trade_filter" not in st.session_state:
    st.session_state.trade_filter="ALL"

search = f1.text_input("Cari Kode").upper()
if f2.button("ALL"): st.session_state.trade_filter="ALL"
if f3.button("BUY"): st.session_state.trade_filter="BUY"
if f4.button("SELL"): st.session_state.trade_filter="SELL"

filtered = df.copy()
if st.session_state.trade_filter!="ALL":
    filtered = filtered[filtered["Signal"]==st.session_state.trade_filter]
if search:
    filtered = filtered[filtered["Kode"].str.contains(search)]

# =====================
# TABLE
# =====================
st.subheader("📊 • INFEKSIUS ACTIO")
st.dataframe(
    filtered.drop(columns="_df"),
    use_container_width=True,
    hide_index=True
)

# =====================
# FOOTER
# =====================
st.caption(f"Update otomatis harian • {datetime.now().strftime('%d %b %Y %H:%M')}")
