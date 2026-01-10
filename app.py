# =========================================================
# IMPORT (BERSIH – CEPAT – AMAN)
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import altair as alt
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from datetime import datetime
from PIL import Image
import os

# =========================================================
# PAGE CONFIG (TIDAK DIUBAH)
# =========================================================
st.set_page_config(page_title="Swing Trading Scanner", layout="wide")

st.markdown("""
<style>
.block-container {padding-top:1rem;padding-bottom:1rem;}
[data-testid="stMetric"] {background:#0e1117;padding:12px;border-radius:10px;}
.stProgress > div > div {background-color:#00C176;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER (IDENTIK)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo = Image.open(os.path.join(BASE_DIR, "logo.png"))

c1, c2 = st.columns([1,7])
with c1:
    st.image(logo, width=140)
with c2:
    st.markdown("## 📈 Swing Trading Scanner")
    st.caption("Realtime Daily Market Screening • Indonesia Stock Exchange")

# =========================================================
# CONFIG
# =========================================================
PERIOD = "6mo"
INTERVAL = "1d"

TP_PCT = st.sidebar.slider("Take Profit (%)", 3, 20, 5)
SL_PCT = st.sidebar.slider("Stop Loss (%)", 2, 10, 3)
fake_rebound_filter = st.sidebar.checkbox("Filter Fake Rebound", False)

# =========================================================
# LOAD TICKER (CEPAT)
# =========================================================
@st.cache_data
def load_idx():
    df = pd.read_csv("idx_tickers.csv")
    if df.shape[1] == 1:
        df.columns = ["Kode"]
    return (df["Kode"].astype(str) + ".JK").tolist()

TICKERS = load_idx()

# =========================================================
# HELPER (TETAP)
# =========================================================
def S(x): return x.astype(float)

def detect_candle(df):
    o,c,h,l = df.iloc[-1][["Open","Close","High","Low"]]
    body = abs(c-o)
    if (min(o,c)-l) > body*2: return "Hammer","Bullish"
    if (h-max(o,c)) > body*2: return "Shooting Star","Bearish"
    return "Normal","Neutral"

def detect_zone(df):
    sup = df["Low"].rolling(20).min().iloc[-1]
    res = df["High"].rolling(20).max().iloc[-1]
    p = df["Close"].iloc[-1]
    if p <= sup*1.03: return "BUY"
    if p >= res*0.97: return "SELL"
    return "MID"

def detect_macd(close):
    m = MACD(close)
    if m.macd().iloc[-2] < m.macd_signal().iloc[-2] and m.macd().iloc[-1] > m.macd_signal().iloc[-1]:
        return "Golden Cross"
    if m.macd().iloc[-2] > m.macd_signal().iloc[-2] and m.macd().iloc[-1] < m.macd_signal().iloc[-1]:
        return "Death Cross"
    return "Normal"

# =========================================================
# FETCH DATA (SUPER CEPAT – BATCH)
# =========================================================
@st.cache_data(ttl=86400)
def fetch_data():
    raw = yf.download(TICKERS, period=PERIOD, interval=INTERVAL, group_by="ticker", threads=True)
    rows = []

    for t in TICKERS:
        if t not in raw: continue
        df = raw[t].dropna()
        if len(df) < 60: continue

        close = df["Close"]
        price = close.iloc[-1]

        trend = "Bullish" if EMAIndicator(close,20).ema_indicator().iloc[-1] > EMAIndicator(close,50).ema_indicator().iloc[-1] else "Bearish"
        zone = detect_zone(df)
        candle, bias = detect_candle(df)
        rsi = RSIIndicator(close).rsi().iloc[-1]
        macd = detect_macd(close)

        confidence = sum([
            trend=="Bullish",
            zone=="BUY",
            bias=="Bullish",
            rsi<40
        ])

        rows.append({
            "Kode": t.replace(".JK",""),
            "Harga": round(price,2),
            "Signal": "BUY" if confidence>=3 else "HOLD",
            "Trend": trend,
            "Zone": zone,
            "Candle": candle,
            "MACD": macd,
            "RSI": round(rsi,1),
            "TP": round(price*(1+TP_PCT/100),2),
            "SL": round(price*(1-SL_PCT/100),2),
            "Confidence": confidence,
            "_df": df
        })

    return pd.DataFrame(rows)

df = fetch_data()

if fake_rebound_filter:
    df = df[df["Confidence"] >= 3]

# =========================================================
# FILTER UI (IDENTIK)
# =========================================================
st.subheader("📊 • INFEKSIUS ACTIO")

search = st.text_input("🔍 Cari Kode Saham").upper()
if search:
    df = df[df["Kode"].str.contains(search)]

# =========================================================
# TABLE (MODEL SAMA)
# =========================================================
st.dataframe(
    df.drop(columns="_df"),
    use_container_width=True,
    hide_index=True
)

# =========================================================
# TOP BUY (SAMA)
# =========================================================
st.subheader("🔥 TOP BUY")
top = df[df["Signal"]=="BUY"].head(10)
st.dataframe(top[["Kode","Harga","Trend","Zone","RSI","TP","SL","Confidence"]],
             use_container_width=True)

st.caption(f"Last update: {datetime.now().strftime('%d %b %Y %H:%M')}")
