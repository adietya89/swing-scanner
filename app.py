# =====================
# IMPORT
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
# SIDEBAR CONFIG
# =====================
TP_PCT = st.sidebar.slider("Take Profit (%)", 3, 20, 5)
SL_PCT = st.sidebar.slider("Stop Loss (%)", 2, 10, 3)

FAST_MODE = st.sidebar.checkbox("⚡ FAST MODE (Disarankan)", True)
USE_INTRADAY = st.sidebar.checkbox("Realtime 1m (Lambat)", False)
FAKE_REBOUND_FILTER = st.sidebar.checkbox("Filter Fake Rebound", False)

MAX_STOCK = 120 if FAST_MODE else 9999

# =====================
# LOAD TICKERS
# =====================
@st.cache_data
def load_idx_tickers():
    df = pd.read_csv("idx_tickers.csv")
    if df.shape[1] == 1:
        df.columns = ["Kode"]
    return (df["Kode"].astype(str).str.strip() + ".JK").tolist()

TICKERS = load_idx_tickers()

PERIOD = "6mo"
INTERVAL = "1d"

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
    price = close.iloc[-1]
    candle, bias = detect_candle(df)
    rsi = RSIIndicator(close, 14).rsi().iloc[-1]
    ema50 = EMAIndicator(close, 50).ema_indicator().iloc[-1]
    macd = MACD(close).macd().iloc[-1]

    return price < ema50 and rsi < 50 and candle == "Shooting Star"

# =====================
# DOWNLOAD ALL DATA (SUPER CEPAT)
# =====================
@st.cache_data(ttl=60*60)
def download_all_data(tickers):
    return yf.download(
        tickers[:MAX_STOCK],
        period=PERIOD,
        interval=INTERVAL,
        group_by="ticker",
        threads=True,
        progress=False
    )

# =====================
# SCAN 1 SAHAM
# =====================
def scan_stock(t, df):
    if df is None or df.empty or len(df) < 60:
        return None

    df = df.dropna()
    close = df["Close"]
    price = float(close.iloc[-1])

    ma50 = close.rolling(50).mean().iloc[-1]
    if price < ma50 * 0.95:
        return None

    rsi = RSIIndicator(close, 14).rsi().iloc[-1]

    macd_ind = MACD(close)
    macd_line = macd_ind.macd()
    signal = macd_ind.macd_signal()

    macd_signal = "Normal"
    if macd_line.iloc[-2] < signal.iloc[-2] and macd_line.iloc[-1] > signal.iloc[-1]:
        macd_signal = "Golden Cross"
    elif macd_line.iloc[-2] > signal.iloc[-2] and macd_line.iloc[-1] < signal.iloc[-1]:
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

    signal_trade = "BUY" if (
        zone == "BUY ZONE" and
        trend == "Bullish" and
        macd_signal == "Golden Cross" and
        rsi <= 50
    ) else "HOLD"

    tp = price * (1 + TP_PCT / 100)
    sl = price * (1 - SL_PCT / 100)

    confidence = sum([
        trend == "Bullish",
        zone == "BUY ZONE",
        bias == "Bullish",
        rsi < 40
    ])

    fake_rebound = detect_fake_rebound(close, df)
    if FAKE_REBOUND_FILTER and fake_rebound:
        return None

    return {
        "Kode": t.replace(".JK",""),
        "Harga": round(price, 2),
        "Signal": signal_trade,
        "Trend": trend,
        "Zone": zone,
        "Candle": candle,
        "RSI": round(rsi, 1),
        "MACD": macd_signal,
        "TP": round(tp, 2),
        "SL": round(sl, 2),
        "Confidence": confidence,
        "_df": df
    }

# =====================
# RUN SCANNER (CACHE)
# =====================
@st.cache_data(ttl=60*60)
def run_scanner():
    rows = []
    all_data = download_all_data(TICKERS)

    for t in TICKERS[:MAX_STOCK]:
        try:
            df = all_data[t] if t in all_data else None

            if USE_INTRADAY:
                try:
                    intraday = yf.Ticker(t).history(period="1d", interval="1m")
                    if not intraday.empty:
                        df = df.copy()
                        df.iloc[-1, df.columns.get_loc("Close")] = intraday["Close"].iloc[-1]
                except:
                    pass

            result = scan_stock(t, df)
            if result:
                rows.append(result)

        except:
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["Confidence", "Signal", "RSI"],
            ascending=[False, False, True]
        )
    return df

# =====================
# EXECUTE
# =====================
with st.spinner("⚡ Scanning super cepat..."):
    df = run_scanner()

# =====================
# UI TABLE
# =====================
st.subheader("📊 HASIL SCANNER")

if df.empty:
    st.warning("Tidak ada data")
else:
    st.dataframe(
        df[["Kode","Harga","Signal","Trend","Zone","Candle","RSI","MACD","TP","SL","Confidence"]],
        use_container_width=True,
        hide_index=True
    )

# =====================
# CONFIDENCE METER
# =====================
st.subheader("🎯 Confidence Meter")

for _, row in df.iterrows():
    st.markdown(f"**{row['Kode']}** — {row['Signal']}")
    st.progress(row["Confidence"] / 4)

# =====================
# FOOTER
# =====================
st.caption(
    f"Update otomatis harian • Last update: {datetime.now().strftime('%d %b %Y %H:%M')}"
)
