# =========================================
# SWING TRADING SCANNER – FINAL CLEAN
# NO SERIES AMBIGUOUS ERROR
# =========================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from datetime import datetime
from PIL import Image
import os

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Swing Trading Scanner",
    layout="wide"
)

# =========================================
# HEADER
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(BASE_DIR, "logo.png")

col1, col2 = st.columns([1.5, 6])
with col1:
    if os.path.exists(logo_path):
        st.image(Image.open(logo_path), width=180)

with col2:
    st.title("📈 Swing Trading Scanner")
    st.caption("Realtime update harian – versi stabil & clean")
    st.caption(
        "Aplikasi ini adalah alat bantu screening teknikal, bukan rekomendasi beli. "
        "Gunakan dengan manajemen risiko & analisa mandiri."
    )

# =========================================
# CACHE
# =========================================
@st.cache_data(ttl=60 * 60 * 24)
def load_idx_tickers():
    df = pd.read_csv("idx_tickers.csv")
    return (df["Kode"] + ".JK").tolist()

TICKERS = load_idx_tickers()

# =========================================
# SIDEBAR
# =========================================
TP_PCT = st.sidebar.slider("Take Profit (%)", 3, 20, 5)
SL_PCT = st.sidebar.slider("Stop Loss (%)", 2, 10, 3)

PERIOD = "6mo"
INTERVAL = "1d"

# =========================================
# HELPER
# =========================================
def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

# =========================================
# MINI CANDLE (2 BAR)
# =========================================
def plot_last_2_candles(df):
    df2 = df.tail(2)

    fig, ax = plt.subplots(figsize=(0.6, 0.6), dpi=140)

    for i in range(len(df2)):
        o = safe_float(df2["Open"].iloc[i])
        c = safe_float(df2["Close"].iloc[i])
        h = safe_float(df2["High"].iloc[i])
        l = safe_float(df2["Low"].iloc[i])

        color = "#00C176" if c >= o else "#FF4D4D"

        ax.plot([i, i], [l, h], color=color, linewidth=0.6)
        ax.bar(i, abs(c - o), bottom=min(o, c), width=0.35, color=color)

    ax.axis("off")
    plt.tight_layout()
    return fig

# =========================================
# LOGIC DETECTORS (ANTI ERROR)
# =========================================
def detect_trend(close):
    ema20 = EMAIndicator(close, 20).ema_indicator()
    ema50 = EMAIndicator(close, 50).ema_indicator()

    if pd.isna(ema20.iloc[-1]) or pd.isna(ema50.iloc[-1]):
        return "Neutral"

    return "Bullish" if safe_float(ema20.iloc[-1]) > safe_float(ema50.iloc[-1]) else "Bearish"

def detect_zone(df):
    low = df["Low"].astype(float)
    high = df["High"].astype(float)
    close = df["Close"].astype(float)

    support = low.rolling(20).min().iloc[-1]
    resistance = high.rolling(20).max().iloc[-1]
    price = close.iloc[-1]

    if pd.isna(support) or pd.isna(resistance):
        return "MID"

    if price <= support * 1.03:
        return "BUY ZONE"
    if price >= resistance * 0.97:
        return "SELL ZONE"

    return "MID"

def detect_candle(df):
    o = safe_float(df["Open"].iloc[-1])
    c = safe_float(df["Close"].iloc[-1])
    h = safe_float(df["High"].iloc[-1])
    l = safe_float(df["Low"].iloc[-1])

    body = abs(c - o)
    lower = min(o, c) - l
    upper = h - max(o, c)

    if lower > body * 2:
        return "Hammer", "Bullish"
    if upper > body * 2:
        return "Shooting Star", "Bearish"

    return "Normal", "Neutral"

def detect_candle_pattern(df):
    if len(df) < 2:
        return "-"

    o1 = safe_float(df["Open"].iloc[-2])
    c1 = safe_float(df["Close"].iloc[-2])
    o2 = safe_float(df["Open"].iloc[-1])
    c2 = safe_float(df["Close"].iloc[-1])
    h2 = safe_float(df["High"].iloc[-1])
    l2 = safe_float(df["Low"].iloc[-1])

    body2 = abs(c2 - o2)
    range2 = h2 - l2 if h2 != l2 else 1.0

    if body2 / range2 < 0.1:
        return "Doji"

    if c1 < o1 and c2 > o2 and c2 > o1:
        return "Bullish Engulfing"
    if c1 > o1 and c2 < o2 and c2 < o1:
        return "Bearish Engulfing"

    return "-"

def build_signal(zone, bias, trend):
    zone = str(zone)
    bias = str(bias)
    trend = str(trend)

    if zone == "BUY ZONE" and bias == "Bullish" and trend == "Bullish":
        return "BUY"
    return "HOLD"

# =========================================
# MAIN PROCESS
# =========================================
rows = []

for t in TICKERS:
    try:
        df = yf.download(t, period=PERIOD, interval=INTERVAL, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty or len(df) < 60:
            continue

        df = df.dropna()
        close = df["Close"].astype(float)

        price = close.iloc[-1]
        trend = detect_trend(close)
        zone = detect_zone(df)
        candle, bias = detect_candle(df)
        pattern = detect_candle_pattern(df)
        rsi = safe_float(RSIIndicator(close, 14).rsi().iloc[-1])

        signal = build_signal(zone, bias, trend)

        tp = price * (1 + TP_PCT / 100)
        sl = price * (1 - SL_PCT / 100)

        confidence = sum([
            trend == "Bullish",
            zone == "BUY ZONE",
            bias == "Bullish",
            rsi < 40
        ])

        rows.append({
            "Kode": t.replace(".JK", ""),
            "Harga": round(price, 2),
            "Signal": signal,
            "Trend": trend,
            "Zone": zone,
            "Candle": candle,
            "Pattern": pattern,
            "RSI": round(rsi, 1),
            "TP": round(tp, 2),
            "SL": round(sl, 2),
            "Confidence": confidence,
            "_df": df.copy()
        })

    except Exception as e:
        st.write(f"Skip {t}: {e}")

# =========================================
# DATAFRAME
# =========================================
df = pd.DataFrame(rows)

if df.empty:
    st.warning("Belum ada data")
    st.stop()

df = df.sort_values(
    by=["Confidence", "RSI"],
    ascending=[False, True]
).reset_index(drop=True)

df["Rank"] = df.index + 1

# =========================================
# TABLE VIEW
# =========================================
st.subheader("🕯️ Signal Saham")

for _, row in df.iterrows():
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(
        [1.2, 1, 1, 1, 1, 0.8, 1, 1, 1]
    )

    c1.write(row["Kode"])
    c2.write(row["Harga"])
    c3.write("🟢 BUY" if row["Signal"] == "BUY" else "⚪ HOLD")
    c4.write(row["Trend"])
    c5.write(row["Zone"])
    c6.pyplot(plot_last_2_candles(row["_df"]), clear_figure=True)
    c7.write(row["Pattern"])
    c8.write(row["RSI"])
    c9.write(f"{row['Confidence']} / 4")

# =========================================
# TOP BUY
# =========================================
st.subheader("🔥 TOP BUY")
top_buy = df[df["Signal"] == "BUY"].head(10)

if top_buy.empty:
    st.info("Belum ada BUY signal hari ini")
else:
    st.dataframe(
        top_buy[
            ["Rank", "Kode", "Harga", "Trend", "Zone", "RSI", "TP", "SL", "Confidence"]
        ],
        use_container_width=True,
        hide_index=True
    )

# =========================================
# FOOTER
# =========================================
st.caption(
    f"Update otomatis harian • {datetime.now().strftime('%d %b %Y %H:%M')}"
)
