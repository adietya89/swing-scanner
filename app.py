# ======================================================
# IMPORT (BERSIH & CEPAT)
# ======================================================
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

# ======================================================
# PAGE CONFIG (TIDAK DIUBAH)
# ======================================================
st.set_page_config(page_title="Swing Trading Scanner", layout="wide")

st.markdown("""
<style>
.block-container {padding-top:1rem;padding-bottom:1rem;}
.header-box {background:linear-gradient(135deg,#0e1117,#151b2c);padding:20px;border-radius:16px;margin-bottom:25px;}
[data-testid="stMetric"] {background-color:#0e1117;padding:12px;border-radius:10px;}
.stProgress > div > div {background-color:#00c176;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo = Image.open(os.path.join(BASE_DIR, "logo.png"))

c1, c2 = st.columns([1,7])
with c1:
    st.image(logo, width=140)
with c2:
    st.markdown("<h1>📈 Swing Trading Scanner</h1>", unsafe_allow_html=True)
    st.caption("Realtime Daily Market Screening • IDX")

# ======================================================
# CONFIG
# ======================================================
PERIOD = "6mo"
INTERVAL = "1d"

TP_PCT = st.sidebar.slider("Take Profit (%)", 3, 20, 5)
SL_PCT = st.sidebar.slider("Stop Loss (%)", 2, 10, 3)
fake_rebound_filter = st.sidebar.checkbox("Filter Fake Rebound", False)

# ======================================================
# LOAD TICKER (CACHE)
# ======================================================
@st.cache_data
def load_idx_tickers():
    df = pd.read_csv("idx_tickers.csv")
    if df.shape[1] == 1:
        df.columns = ["Kode"]
    return (df["Kode"].astype(str) + ".JK").tolist()

TICKERS = load_idx_tickers()

# ======================================================
# HELPER
# ======================================================
def S(x): return x.astype(float)

def distance_to_ma(close, n):
    if len(close) < n: return np.nan
    ma = close.rolling(n).mean().iloc[-1]
    return abs(close.iloc[-1] - ma) / ma * 100

# ======================================================
# SIGNAL LOGIC (SAMA)
# ======================================================
def detect_macd_signal(close):
    m = MACD(close)
    macd, sig = m.macd(), m.macd_signal()
    if macd.iloc[-2] < sig.iloc[-2] and macd.iloc[-1] > sig.iloc[-1]:
        return "Golden Cross"
    if macd.iloc[-2] > sig.iloc[-2] and macd.iloc[-1] < sig.iloc[-1]:
        return "Death Cross"
    return "Normal"

def detect_trend(close):
    return "Bullish" if EMAIndicator(close,20).ema_indicator().iloc[-1] > EMAIndicator(close,50).ema_indicator().iloc[-1] else "Bearish"

def detect_zone(df):
    sup = df["Low"].rolling(20).min().iloc[-1]
    res = df["High"].rolling(20).max().iloc[-1]
    p = df["Close"].iloc[-1]
    if p <= sup * 1.03: return "BUY ZONE"
    if p >= res * 0.97: return "SELL ZONE"
    return "MID"

def detect_candle(df):
    o,c,h,l = df.iloc[-1][["Open","Close","High","Low"]]
    body = abs(c-o)
    if (min(o,c)-l) > body*2: return "Hammer","Bullish"
    if (h-max(o,c)) > body*2: return "Shooting Star","Bearish"
    return "Normal","Neutral"

def detect_fake_rebound(close, df):
    rsi = RSIIndicator(close).rsi().iloc[-1]
    ema50 = EMAIndicator(close,50).ema_indicator().iloc[-1]
    candle,_ = detect_candle(df)
    return close.iloc[-1] < ema50 and rsi < 50 and candle=="Shooting Star"

# ======================================================
# DATA FETCH (INI YANG DIPERCEPAT)
# ======================================================
@st.cache_data(ttl=86400)
def fetch_all():
    data = yf.download(TICKERS, period=PERIOD, interval=INTERVAL, group_by="ticker", threads=True)
    rows=[]
    for t in TICKERS:
        if t not in data: continue
        df = data[t].dropna()
        if len(df) < 60: continue

        close = df["Close"]
        price = close.iloc[-1]

        macd = detect_macd_signal(close)
        trend = detect_trend(close)
        zone = detect_zone(df)
        candle,bias = detect_candle(df)
        rsi = RSIIndicator(close).rsi().iloc[-1]

        tp = price*(1+TP_PCT/100)
        sl = price*(1-SL_PCT/100)

        conf = sum([
            trend=="Bullish",
            zone=="BUY ZONE",
            bias=="Bullish",
            rsi<40
        ])

        rows.append({
            "Kode":t,
            "Harga":round(price,2),
            "Signal":"BUY" if conf>=3 else "HOLD",
            "Trend":trend,
            "Zone":zone,
            "Candle":candle,
            "RSI":round(rsi,1),
            "MACD":macd,
            "TP":round(tp,2),
            "SL":round(sl,2),
            "Confidence":conf,
            "Fake_Rebound":detect_fake_rebound(close,df),
            "_df":df
        })
    return pd.DataFrame(rows)

df = fetch_all()
if fake_rebound_filter:
    df = df[df["Fake_Rebound"]==False]

df = df.sort_values(["Confidence","RSI"], ascending=[False,True])

# ======================================================
# TABLE & UI (KERANGKA ASLI)
# ======================================================
st.subheader("📊 • INFEKSIUS ACTIO")
st.dataframe(df.drop(columns="_df"), use_container_width=True, hide_index=True)

# ======================================================
# TOP BUY
# ======================================================
st.subheader("🔥 TOP BUY")
top = df[df["Signal"]=="BUY"].head(10)
st.dataframe(top[["Kode","Harga","Trend","Zone","RSI","TP","SL","Confidence"]], use_container_width=True)

st.caption(f"Last update: {datetime.now().strftime('%d %b %Y %H:%M')}")
