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
st.set_page_config(page_title="Swing Trading Scanner", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
[data-testid="stMetric"] { background-color: #0e1117; padding: 12px; border-radius: 10px; }
.stProgress > div > div { background-color: #00c176; }
</style>
""", unsafe_allow_html=True)

# =====================
# HEADER
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo = Image.open(os.path.join(BASE_DIR, "logo.png"))

c1, c2 = st.columns([1, 7])
with c1:
    st.image(logo, width=140)
with c2:
    st.markdown("## 📈 Swing Trading Scanner")
    st.caption("Realtime Daily Market Screening • IDX")

# =====================
# SIDEBAR
# =====================
TP_PCT = st.sidebar.slider("Take Profit (%)", 3, 20, 5)
SL_PCT = st.sidebar.slider("Stop Loss (%)", 2, 10, 3)
FAKE_REBOUND_FILTER = st.sidebar.checkbox("Filter Fake Rebound", False)

# =====================
# LOAD TICKERS
# =====================
@st.cache_data
def load_tickers():
    df = pd.read_csv("idx_tickers.csv")
    if df.shape[1] == 1:
        df.columns = ["Kode"]
    return (df["Kode"].astype(str).str.strip() + ".JK").tolist()

TICKERS = load_tickers()

# =====================
# BULK DOWNLOAD (SUPER CEPAT)
# =====================
@st.cache_data(ttl=60*60)
def download_all():
    return yf.download(
        TICKERS,
        period="6mo",
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False
    )

# =====================
# HELPER
# =====================
def detect_candle(df):
    o, c, h, l = df.iloc[-1][["Open","Close","High","Low"]]
    body = abs(c - o)
    lower = min(o, c) - l
    upper = h - max(o, c)

    if lower > body * 2:
        return "Hammer", "Bullish"
    if upper > body * 2:
        return "Shooting Star", "Bearish"
    return "Normal", "Neutral"

def detect_fake_rebound(close, df):
    candle, _ = detect_candle(df)
    rsi = RSIIndicator(close, 14).rsi().iloc[-1]
    ema50 = EMAIndicator(close, 50).ema_indicator().iloc[-1]
    return close.iloc[-1] < ema50 and rsi < 50 and candle == "Shooting Star"

# =====================
# SCAN ENGINE (OPTIMIZED)
# =====================
def scan_stock(t, df):
    if df is None or df.empty or len(df) < 60:
        return None

    df = df.dropna()
    close = df["Close"]
    price = close.iloc[-1]

    rsi = RSIIndicator(close, 14).rsi().iloc[-1]

    macd = MACD(close)
    macd_line = macd.macd()
    macd_sig = macd.macd_signal()

    macd_signal = "Normal"
    if macd_line.iloc[-2] < macd_sig.iloc[-2] and macd_line.iloc[-1] > macd_sig.iloc[-1]:
        macd_signal = "Golden Cross"
    elif macd_line.iloc[-2] > macd_sig.iloc[-2] and macd_line.iloc[-1] < macd_sig.iloc[-1]:
        macd_signal = "Death Cross"

    ema20 = EMAIndicator(close, 20).ema_indicator().iloc[-1]
    ema50 = EMAIndicator(close, 50).ema_indicator().iloc[-1]
    trend = "Bullish" if ema20 > ema50 else "Bearish"

    support = df["Low"].rolling(20).min().iloc[-1]
    resistance = df["High"].rolling(20).max().iloc[-1]

    zone = "MID"
    if price <= support * 1.03:
        zone = "BUY ZONE"
    elif price >= resistance * 0.97:
        zone = "SELL ZONE"

    candle, bias = detect_candle(df)

    signal = "BUY" if (
        zone == "BUY ZONE" and
        trend == "Bullish" and
        macd_signal == "Golden Cross" and
        rsi <= 50
    ) else "HOLD"

    confidence = sum([
        trend == "Bullish",
        zone == "BUY ZONE",
        bias == "Bullish",
        rsi < 40
    ])

    fake = detect_fake_rebound(close, df)
    if FAKE_REBOUND_FILTER and fake:
        return None

    return {
        "Kode": t.replace(".JK",""),
        "Harga": round(price,2),
        "Signal": signal,
        "Trend": trend,
        "Zone": zone,
        "Candle": candle,
        "RSI": round(rsi,1),
        "MACD": macd_signal,
        "TP": round(price*(1+TP_PCT/100),2),
        "SL": round(price*(1-SL_PCT/100),2),
        "Confidence": confidence,
        "_df": df
    }

# =====================
# RUN SCANNER
# =====================
with st.spinner("⚡ Scanning super cepat..."):
    raw = download_all()
    rows = []

    for t in TICKERS:
        try:
            df = raw[t] if t in raw else None
            r = scan_stock(t, df)
            if r:
                rows.append(r)
        except:
            pass

df = pd.DataFrame(rows).sort_values(
    ["Confidence","Signal","RSI"],
    ascending=[False,False,True]
)

# =====================
# UI TABLE (TETAP SAMA)
# =====================
st.subheader("📊 HASIL SCANNER")

st.dataframe(
    df[["Kode","Harga","Signal","Trend","Zone","Candle","RSI","MACD","TP","SL","Confidence"]],
    use_container_width=True,
    hide_index=True
)

# =====================
# CONFIDENCE METER
# =====================
st.subheader("🎯 Confidence Meter")
for _, r in df.iterrows():
    st.markdown(f"**{r['Kode']}** — {r['Signal']}")
    st.progress(r["Confidence"]/4)

# =====================
# TOP BUY
# =====================
st.subheader("🔥 TOP BUY")
top = df[df["Signal"]=="BUY"].head(10)

st.dataframe(
    top[["Kode","Harga","Trend","Zone","RSI","TP","SL","Confidence"]],
    use_container_width=True,
    hide_index=True
)

# =====================
# FOOTER
# =====================
st.caption(f"Last update: {datetime.now().strftime('%d %b %Y %H:%M')}")
