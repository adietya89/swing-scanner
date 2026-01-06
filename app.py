import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from datetime import datetime

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Swing Trading Scanner",
    layout="wide"
)

# =====================
# HEADER IMAGE (LOGO + TITLE)
# =====================
import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo = Image.open(os.path.join(BASE_DIR, "logo.png"))

col1, col2 = st.columns([1.5, 6])

with col1:
    st.image(logo, width=200)

with col2:
    st.title("📈 Swing Trading Scanner")
    st.caption("Realtime update harian • INFEKSIUS ACTIO")

# =====================
# AUTO REFRESH (HARIAN)
# =====================
st.cache_data(ttl=60 * 60 * 24)

# =====================
# CONFIG
# =====================
TICKERS = ["BBRI.JK", "BBCA.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]
PERIOD = "6mo"
INTERVAL = "1d"

TP_PCT = st.sidebar.slider("Take Profit (%)", 3, 20, 5)
SL_PCT = st.sidebar.slider("Stop Loss (%)", 2, 10, 3)

# =====================
# HELPER (ANTI ERROR)
# =====================
def S(x):
    return x.squeeze().astype(float)
def plot_last_2_candles(df, kode):
    df2 = df.tail(2)

    fig, ax = plt.subplots(figsize=(3, 2))

    for i in range(len(df2)):
        o = float(df2["Open"].iloc[i])
        c = float(df2["Close"].iloc[i])
        h = float(df2["High"].iloc[i])
        l = float(df2["Low"].iloc[i])

        color = "green" if c >= o else "red"

        # Wick
        ax.plot([i, i], [l, h], color=color, linewidth=1)

        # Body
        ax.bar(
            i,
            c - o,
            bottom=o,
            color=color,
            width=0.5
        )

    ax.set_title(kode, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    return fig
# =====================
# LOGIC
# =====================
def detect_trend(close):
    ema20 = EMAIndicator(close, 20).ema_indicator()
    ema50 = EMAIndicator(close, 50).ema_indicator()
    return "Bullish" if ema20.iloc[-1] > ema50.iloc[-1] else "Bearish"

def detect_zone(df):
    support = S(df["Low"]).rolling(20).min().iloc[-1]
    resistance = S(df["High"]).rolling(20).max().iloc[-1]
    price = S(df["Close"]).iloc[-1]

    if price <= support * 1.03:
        return "BUY ZONE"
    elif price >= resistance * 0.97:
        return "SELL ZONE"
    return "MID"

def detect_candle(df):
    o = S(df["Open"]).iloc[-1]
    c = S(df["Close"]).iloc[-1]
    h = S(df["High"]).iloc[-1]
    l = S(df["Low"]).iloc[-1]

    body = abs(c - o)
    lower = min(o, c) - l
    upper = h - max(o, c)

    if lower > body * 2:
        return "Hammer", "Bullish"
    if upper > body * 2:
        return "Shooting Star", "Bearish"
    return "Normal", "Neutral"

def build_signal(zone, bias, trend):
    if zone == "BUY ZONE" and bias == "Bullish" and trend == "Bullish":
        return "BUY"
    return "HOLD"

# =====================
# DATA PROCESS
# =====================
rows = []

for t in TICKERS:
    try:
        df = yf.download(t, period=PERIOD, interval=INTERVAL, progress=False)
        if df.empty or len(df) < 60:
            continue

        df = df.dropna()
        close = S(df["Close"])
        price = close.iloc[-1]

        trend = detect_trend(close)
        zone = detect_zone(df)
        candle, bias = detect_candle(df)
        rsi = RSIIndicator(close, 14).rsi().iloc[-1]

        signal = build_signal(zone, bias, trend)

        tp = price * (1 + TP_PCT / 100)
        sl = price * (1 - SL_PCT / 100)

        confidence = 0
        confidence += 1 if trend == "Bullish" else 0
        confidence += 1 if zone == "BUY ZONE" else 0
        confidence += 1 if bias == "Bullish" else 0
        confidence += 1 if rsi < 40 else 0

        rows.append({
            "Kode": t,
            "Harga": round(price, 2),
            "Signal": "BUY" if signal == "BUY" else "HOLD",
            "Trend": trend,
            "Zone": zone,
            "Candle": candle,
            "RSI": round(rsi, 1),
            "TP": round(tp, 2),
            "SL": round(sl, 2),
            "Confidence": confidence,
            "_df": df.copy()
        })

    except Exception as e:
        st.write(f"Error {t}: {e}")

df = pd.DataFrame(rows)

# =====================
# UI TABLE
# =====================
st.subheader("📊 Signal Saham")

if df.empty:
    st.warning("Belum ada data")
else:
    st.subheader("📊 Signal Saham (Candle Langsung Tampil)")

# Header tabel
h1, h2, h3, h4, h5, h6, h7, h8, h9 = st.columns(
    [1.2, 1, 1, 1, 1, 2, 1, 1, 1]
)

h1.markdown("**Kode**")
h2.markdown("**Harga**")
h3.markdown("**Signal**")
h4.markdown("**Trend**")
h5.markdown("**Zone**")
h6.markdown("**Candle (2 terakhir)**")
h7.markdown("**RSI**")
h8.markdown("**TP**")
h9.markdown("**SL**")

st.divider()

# Isi tabel
for _, row in df.iterrows():
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(
        [1.2, 1, 1, 1, 1, 2, 1, 1, 1]
    )

    c1.write(row["Kode"].replace(".JK", ""))
    c2.write(row["Harga"])
    c3.write("🟢 BUY" if row["Signal"] == "BUY" else "⚪ HOLD")
    c4.write(row["Trend"])
    c5.write(row["Zone"])
    c7.write(row["RSI"])
    c8.write(row["TP"])
    c9.write(row["SL"])

    # CANDLE LANGSUNG TAMPIL
    st.subheader("🕯️ Candle Terakhir (2 Candle)")
    fig = plot_last_2_candles(row["_df"], row["Kode"])
    c6.pyplot(fig, clear_figure=True)
    

for _, row in df.iterrows():
    with st.expander(f"{row['Kode'].replace('.JK','')} — {row['Candle']}"):
        fig = plot_last_2_candles(row["_df"], row["Kode"])
        st.pyplot(fig)

# =====================
# CONFIDENCE METER
# =====================
st.subheader("🎯 Confidence Meter")

for _, row in df.iterrows():
    col1, col2 = st.columns([1, 4])
    with col1:
        st.write(f"**{row['Kode'].replace('.JK','')}**")
        st.write("🟢 BUY" if row["Signal"] == "BUY" else "⚪ HOLD")
    with col2:
        st.progress(row["Confidence"] / 4)
        st.caption(f"{row['Confidence']} / 4 indikator")

# =====================
# BUY ONLY
# =====================
st.subheader("✅ BUY SIGNAL ONLY")
buy_df = df[df["Signal"] == "BUY"]

if buy_df.empty:
    st.info("Belum ada BUY signal hari ini")
else:
    st.dataframe(buy_df, use_container_width=True, hide_index=True)

# =====================
# FOOTER
# =====================
st.caption(
    f"Update otomatis harian • Last update: {datetime.now().strftime('%d %b %Y %H:%M')}"
)

















