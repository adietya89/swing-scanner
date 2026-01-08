import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import ta
import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import pandas as pd
import numpy as np
import altair as alt

from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from datetime import datetime
from PIL import Image
import os

def plot_last_2_candles(df):
    df2 = df.tail(2)

    fig, ax = plt.subplots(figsize=(0.45, 0.9), dpi=200)

    for i in range(len(df2)):
        o = float(df2["Open"].iloc[i])
        c = float(df2["Close"].iloc[i])
        h = float(df2["High"].iloc[i])
        l = float(df2["Low"].iloc[i])

        color = "#00C176" if c >= o else "#FF4D4D"

        ax.plot([i, i], [l, h], color=color, linewidth=1.1)

        ax.bar(
            i,
            abs(c - o),
            bottom=min(o, c),
            width=0.28,
            color=color
        )

    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(df2["Low"].min() * 0.995, df2["High"].max() * 1.005)

    ax.axis("off")
    plt.tight_layout(pad=0)

    return fig

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
# HEADER IMAGE (LOGO + TITLE)
# =====================
import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo = Image.open(os.path.join(BASE_DIR, "logo.png"))

col1, col2 = st.columns([1, 7])

with col1:
    st.image(logo, width=160)

with col2:
    st.markdown("## 📈 **Swing Trading Scanner**")
    st.caption("Realtime Daily Market Screening • Indonesia Stock Exchange")
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
def detect_macd_signal(close):
    macd_ind = MACD(close)

    macd_line = macd_ind.macd()
    signal    = macd_ind.macd_signal()

    if len(macd_line) < 2:
        return "Normal"

    if macd_line.iloc[-2] < signal.iloc[-2] and macd_line.iloc[-1] > signal.iloc[-1]:
        return "Golden Cross"

    elif macd_line.iloc[-2] > signal.iloc[-2] and macd_line.iloc[-1] < signal.iloc[-1]:
        return "Death Cross"

    else:
        return "Normal"

def detect_ma_position(close):
    mas = [
        ("MA5", 5),
        ("MA10", 10),
        ("MA20", 20),
        ("MA50", 50),
        ("MA100", 100),
        ("MA200", 200),
    ]

    price = close.iloc[-1]
    above = []

    for name, period in mas:
        if len(close) >= period:
            ma = close.rolling(period).mean().iloc[-1]
            if price > ma:
                above.append(name)

    if not above:
        return "—"

    return "⬆ " + " ".join(above)

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
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 60:
            continue

        df = df.dropna()
        close = S(df["Close"])
        price = close.iloc[-1]
        macd_signal = detect_macd_signal(close)
        ma_pos = detect_ma_position(close)

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
            "MA_Pos": ma_pos,
            "MACD": macd_signal,
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

st.markdown("### 📊 Market Signal Overview")
st.caption("Filtered berdasarkan trend, zone, candle, RSI, dan MACD")

if df.empty:
    st.warning("Belum ada data")
else:
    st.subheader("🕯️Signal Saham ")

# Header tabel
h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12 = st.columns(
    [1.2, 1, 1, 1, 1, 0.8, 1.2, 1.2, 1, 1, 1, 1]
)

h1.markdown("**Kode**")
h2.markdown("**Harga**")
h3.markdown("**Signal**")
h4.markdown("**Trend**")
h5.markdown("**Zone**")
h6.markdown("**Candle**")
h7.markdown("**MA >**")
h8.markdown("**MACD**")
h9.markdown("**RSI**")
h10.markdown("**TP**")
h11.markdown("**SL**")
h12.markdown("**SPARKLINE**")

st.divider()

ROW_HEIGHT = 70

# =====================
# Kolom 1 - 9
# =====================
for _, row in df.iterrows():
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = st.columns(
    [1.2, 1, 1, 1, 1, 0.8, 1.2, 1.2, 1, 1, 1, 1]
    )

 
    with c1.container(height=ROW_HEIGHT):
        st.write(row["Kode"].replace(".JK",""))

    with c2.container(height=ROW_HEIGHT):
        st.write(row["Harga"])

    with c3.container(height=ROW_HEIGHT):
        if row["Signal"] == "BUY":
           st.markdown("<span style='color:#00C176; font-weight:bold'>BUY</span>", unsafe_allow_html=True)
        else:
           st.markdown("<span style='color:#999'>HOLD</span>", unsafe_allow_html=True)

    with c4.container(height=ROW_HEIGHT):
    if row["Trend"] == "Bullish":
        st.markdown(
            "<span style='font-size:13px; color:#00C176; font-weight:600'>🟢 Bullish</span>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<span style='font-size:13px; color:#FF4D4D; font-weight:600'>🔴 Bearish</span>",
            unsafe_allow_html=True
        )

    with c5.container(height=ROW_HEIGHT):
        if row["Zone"] == "BUY ZONE":
           st.markdown("<span style='color:#00C176; font-weight:600'>BUY</span>", unsafe_allow_html=True)
        elif row["Zone"] == "SELL ZONE":
           st.markdown("<span style='color:#FF4D4D; font-weight:600'>SELL</span>", unsafe_allow_html=True)
        else:
           st.markdown("<span style='color:#999'>MID</span>", unsafe_allow_html=True)

    with c6.container(height=ROW_HEIGHT):
        fig = plot_last_2_candles(row["_df"])
        st.pyplot(fig, clear_figure=True)
        
    with c7.container(height=ROW_HEIGHT):
        st.write(row["MA_Pos"])
        
    with c8.container(height=ROW_HEIGHT):
       macd = row["MACD"]
       if macd == "Golden Cross":
          st.markdown("🟢 **Golden Cross**")
       elif macd == "Death Cross":
          st.markdown("🔴 **Death Cross**")
       else:
          st.markdown("⚪ Normal")

    with c9.container(height=ROW_HEIGHT):
        rsi = row["RSI"]
        if rsi < 40:
           st.markdown(f"🟢 **{rsi}**")
        elif rsi > 70:
           st.markdown(f"🔴 **{rsi}**")
        else:
           st.markdown(f"⚪ {rsi}")

    with c10.container(height=ROW_HEIGHT):
        st.markdown(
           f"<span style='font-size:13px'>💰 {row['TP']}</span>",
           unsafe_allow_html=True
        )

    with c11.container(height=ROW_HEIGHT):
        st.markdown(
           f"<span style='font-size:13px'>🛑 {row['SL']}</span>",
           unsafe_allow_html=True
        )

    # =====================
    # Kolom 12 = Sparkline
    # =====================
    with c12:
        try:
            close = row["_df"]["Close"].tail(90)
            close_values = close.squeeze().to_numpy()

            min_val = close_values.min()
            max_val = close_values.max()
            if max_val - min_val == 0:
                norm_values = np.full_like(close_values, 0.5, dtype=float)
            else:
                norm_values = (close_values - close_values.min()) / (close_values.max() - close_values.min())
                mean_val = np.mean(norm_values)
                norm_values = norm_values - mean_val + 0.5
                norm_values = np.clip(norm_values, 0, 1)
                     
            data = pd.DataFrame({
                'index': range(len(norm_values)),
                'close': norm_values
            })

            trend_color = "#999"
            if close_values[-1] > close_values[0]:
               trend_color = "#00C176"   # naik
            elif close_values[-1] < close_values[0]:
                trend_color = "#FF4D4D"   # turun

            chart = (
                alt.Chart(data)
                .mark_line(color=trend_color, strokeWidth=1.8)
                .encode(
                   x=alt.X('index', axis=None),
                   y=alt.Y('close', axis=None, scale=alt.Scale(domain=[0,1]))
                )
                .properties(height=30)
            )

            st.altair_chart(chart, use_container_width=True)
            
        except Exception as e:
            st.write("-")
            
# =====================
# CONFIDENCE METER
# =====================
st.subheader("🎯 Confidence Meter")

for _, row in df.iterrows():
    score = row["Confidence"]

    col1, col2 = st.columns([1.2, 4])

    with col1:
        st.markdown(f"**{row['Kode'].replace('.JK','')}**")
        if row["Signal"] == "BUY":
            st.markdown("🟢 **BUY**")
        else:
            st.markdown("⚪ HOLD")

    with col2:
        st.progress(score / 4)

        if score == 4:
            st.caption("🔥 Sangat Kuat (4/4 indikator)")
        elif score == 3:
            st.caption("✅ Kuat (3/4 indikator)")
        elif score == 2:
            st.caption("⚠ Cukup (2/4 indikator)")
        else:
            st.caption("❌ Lemah (<2 indikator)")


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




















