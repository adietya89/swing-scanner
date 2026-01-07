import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from datetime import datetime
def plot_last_2_candles(df):
    df2 = df.tail(2)  # 2 candle seperti contoh

    GREEN = "#00C176"
    RED = "#FF4D4D"

    fig, ax = plt.subplots(figsize=(0.35, 0.8), dpi=220)

    base_x = 0
    gap = 0.35

    for i in range(len(df2)):
        o = df2["Open"].iloc[i]
        c = df2["Close"].iloc[i]
        h = df2["High"].iloc[i]
        l = df2["Low"].iloc[i]

        # 🔥 NORMALISASI (INI KUNCI)
        total = h - l if h != l else 1
        o_n = (o - l) / total * 100
        c_n = (c - l) / total * 100
        h_n = 100
        l_n = 0

        color = GREEN if c >= o else RED
        x = base_x + i * gap

        # Wick
        ax.plot([x, x], [l_n, h_n], color=color, linewidth=1.1)

        # Body
        ax.add_patch(
            plt.Rectangle(
                (x - 0.07, min(o_n, c_n)),
                0.14,
                abs(c_n - o_n),
                color=color
            )
        )

    ax.set_xlim(-0.2, gap * len(df2))
    ax.set_ylim(-5, 105)
    ax.axis("off")

    return fig
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
    st.caption("Realtime update harian")
    st.caption(
    "Memasuki dinamika pasar tahun 2026, strategi swing trading memerlukan ketelitian dalam menangkap momentum harga. "
    "Screener ini hadir sebagai alat bantu untuk mempersempit pilihan saham yang menunjukkan potensi pembalikan arah atau kelanjutan tren. "
    "Namun, perlu diingat bahwa aplikasi hanyalah alat bantu teknis. Hasil screening ini bukanlah perintah beli; "
    "tetap lakukan DYOR (Do Your Own Research) dan percayalah pada analisis mandiri menggunakan ilmu serta manajemen risiko yang Anda miliki."
)
    
# =====================
# AUTO REFRESH (HARIAN)
# =====================
st.cache_data(ttl=60 * 60 * 24)

# =====================
# CONFIG
# =====================
@st.cache_data
def load_idx_tickers():
    df = pd.read_csv("idx_tickers.csv")
    return (df["Kode"] + ".JK").tolist()

TICKERS = load_idx_tickers()
PERIOD = "6mo"
INTERVAL = "1d"

TP_PCT = st.sidebar.slider("Take Profit (%)", 3, 20, 5)
SL_PCT = st.sidebar.slider("Stop Loss (%)", 2, 10, 3)

# =====================
# HELPER (ANTI ERROR)
# =====================
def S(x):
    return x.squeeze().astype(float)
def plot_last_2_candles(df):
    df2 = df.tail(2)

    fig, ax = plt.subplots(figsize=(0.6, 0.6), dpi=140)

    for i in range(len(df2)):
        o = float(df2["Open"].iloc[i])
        c = float(df2["Close"].iloc[i])
        h = float(df2["High"].iloc[i])
        l = float(df2["Low"].iloc[i])

        color = "#00ff88" if c >= o else "#ff4d4d"

        # Wick
        ax.plot([i, i], [l, h], color=color, linewidth=0.6)

        # Body
        ax.bar(
            i,
            abs(c - o),
            bottom=min(o, c),
            width=0.35,
            color=color
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
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
df = df.sort_values(
    by=["Confidence", "Signal", "RSI"],
    ascending=[False, False, True]
)
# =====================
# UI TABLE
# =====================
st.subheader("📊 • INFEKSIUS ACTIO")

if df.empty:
    st.warning("Belum ada data")
else:
    st.subheader("🕯️Signal Saham ")

# Header tabel
h1, h2, h3, h4, h5, h6, h7, h8, h9, h10 = st.columns(
    [1.2, 1, 1, 1, 1, 0.8, 1, 1, 1, 1]
)

h1.markdown("**Kode**")
h2.markdown("**Harga**")
h3.markdown("**Signal**")
h4.markdown("**Trend**")
h5.markdown("**Zone**")
h6.markdown("**Candle**")
h7.markdown("**RSI**")
h8.markdown("**TP**")
h9.markdown("**SL**")
h10.markdown("**SPARKLINE**")

st.divider()

ROW_HEIGHT = 70
# Isi tabel
for _, row in df.iterrows():
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10= st.columns(
        [1.2, 1, 1, 1, 1, 0.8, 1, 1, 1, 1.5]
    )

      # Step 1: Ambil data sparkline 90 hari
    close = S(row["_df"]["Close"]).tail(90) 
    close_values = close.to_numpy().squeeze()

    if max_val - min_val == 0:
        norm_values = close_values  # kalau flat
    else:
        norm_values = (close_values - min_val) / (max_val - min_val)

    # Step 5 – Buat DataFrame untuk Altair
      data = pd.DataFrame({
        'index': range(len(norm_values)),
        'close': norm_values
      })
    with c1.container(height=ROW_HEIGHT):
        st.write(row["Kode"].replace(".JK",""))

    with c2.container(height=ROW_HEIGHT):
        st.write(row["Harga"])

    with c3.container(height=ROW_HEIGHT):
        st.write("🟢 BUY" if row["Signal"] == "BUY" else "⚪ HOLD")

    with c4.container(height=ROW_HEIGHT):
        st.write(row["Trend"])

    with c5.container(height=ROW_HEIGHT):
        st.write(row["Zone"])

    with c6.container(height=ROW_HEIGHT):
        fig = plot_last_2_candles(row["_df"])
        st.pyplot(fig, clear_figure=True)

    with c7.container(height=ROW_HEIGHT):
        st.write(row["RSI"])

    with c8.container(height=ROW_HEIGHT):
        st.write(row["TP"])

    with c9.container(height=ROW_HEIGHT):
        st.write(row["SL"])

    # ===================
    # Kolom 10 = Sparkline pakai Altair
    # ===================
    with c10:
         close_values = close.values
         # NORMALISASI: supaya sparkline terlihat
         min_val = close_values.min()
         max_val = close_values.max()
         if max_val - min_val == 0:
            norm_values = close_values
         else:
              norm_values = (close_values - min_val) / (max_val - min_val)

         data = pd.DataFrame({
              'index': range(len(norm_values)),
              'close': norm_values
         })
    
         chart = (
               alt.Chart(data)
               .mark_line(color="#5f88cc", strokeWidth=1.5)
               .encode(
                   x=alt.X('index', axis=None),
                   y=alt.Y('close', axis=None)
                )
                .properties(height=30)
         )
         st.altair_chart(chart, use_container_width=True)
   
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
st.subheader("🔥 TOP BUY (Ranking Terkuat)")

top_buy = df[df["Signal"] == "BUY"].head(10)

if top_buy.empty:
    st.info("Belum ada BUY signal kuat hari ini")
else:
    st.dataframe(
        top_buy[
            ["Rank","Kode", "Harga", "Trend", "Zone", "Candle", "RSI", "TP", "SL", "Confidence"]
        ],
        use_container_width=True,
        hide_index=True
    )

# =====================
# FOOTER
# =====================
st.caption(
    f"Update otomatis harian • Last update: {datetime.now().strftime('%d %b %Y %H:%M')}"
)



























































