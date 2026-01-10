# =====================
# IMPORT
# =====================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

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

# =====================
# STYLE
# =====================
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
    st.caption("IDX • Daily Swing Strategy • 2026")

# =====================
# CONFIG
# =====================
PERIOD = "6mo"
INTERVAL = "1d"

TP_PCT = st.sidebar.slider("Take Profit (%)", 3, 20, 5)
SL_PCT = st.sidebar.slider("Stop Loss (%)", 2, 10, 3)

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

# =====================
# HELPER
# =====================
def S(x):
    return pd.to_numeric(x, errors="coerce").astype(float)

def calculate_support_resistance(ticker, period=20):
    df = yf.download(ticker, period=f"{period}d", interval="1d", progress=False)
    if df.empty:
        return None, None
    return float(df["Low"].min()), float(df["High"].max())

def calculate_fair_value_simple(ticker, price):
    info = yf.Ticker(ticker).info
    eps = info.get("trailingEps")
    sector = info.get("sector", "Default")

    per_map = {
        "Financial Services": 12,
        "Technology": 18,
        "Energy": 8,
        "Industrials": 11,
        "Default": 12
    }

    if eps is None or eps <= 0:
        return None

    per = per_map.get(sector, 12)
    fair = eps * per
    margin = (fair - price) / price * 100

    return {
        "Fair": round(fair, 2),
        "Margin": round(margin, 1),
        "EPS": round(eps, 2),
        "PER": per,
        "Sector": sector
    }

def detect_trend(close):
    ema20 = EMAIndicator(close, 20).ema_indicator()
    ema50 = EMAIndicator(close, 50).ema_indicator()
    return "Bullish" if ema20.iloc[-1] > ema50.iloc[-1] else "Bearish"

# =====================
# DATA PROCESS
# =====================
rows = []

with st.spinner("⏳ Mengambil data saham IDX..."):
    for t in TICKERS:
        try:
            df = yf.download(t, period=PERIOD, interval=INTERVAL, progress=False)
            if df.empty or len(df) < 60:
                continue

            close = S(df["Close"])
            price = float(close.iloc[-1])
            trend = detect_trend(close)

            tp = price * (1 + TP_PCT / 100)
            sl = price * (1 - SL_PCT / 100)

            rows.append({
                "Kode": t,
                "Harga": round(price, 2),
                "Trend": trend,
                "TP": round(tp, 2),
                "SL": round(sl, 2),
                "_df": df.copy()
            })

        except Exception:
            pass

df = pd.DataFrame(rows)

# =====================
# SIDEBAR – HARGA WAJAR + S/R (FINAL FIX)
# =====================
st.sidebar.markdown("## 💎 Harga Wajar Saham")

search = st.sidebar.text_input("🔍 Kode Saham", placeholder="BBRI / BBCA").upper()

if search:
    ticker = search + ".JK"
    row_df = df[df["Kode"] == ticker]

    if not row_df.empty:
        row = row_df.iloc[0]
        harga_now = float(row["Harga"])

        st.sidebar.metric("Harga Saat Ini", harga_now)

        # Support & Resistance
        support, resistance = calculate_support_resistance(ticker)

        if support is not None and resistance is not None:
            support = float(support)
            resistance = float(resistance)

            st.sidebar.markdown("### 📐 Support & Resistance")
            st.sidebar.metric("Support", round(support, 2))
            st.sidebar.metric("Resistance", round(resistance, 2))

            if harga_now <= support * 1.03:
                st.sidebar.success("📍 Harga dekat SUPPORT")
            elif harga_now >= resistance * 0.97:
                st.sidebar.warning("📍 Harga dekat RESISTANCE")

        # Fair Value
        fv = calculate_fair_value_simple(ticker, harga_now)
        if fv:
            st.sidebar.metric(
                "Harga Wajar",
                fv["Fair"],
                delta=f"{fv['Margin']}%"
            )

            if fv["Margin"] > 20:
                st.sidebar.success("🟢 Undervalued")
            elif fv["Margin"] < -10:
                st.sidebar.error("🔴 Overvalued")
            else:
                st.sidebar.warning("🟡 Fair Value")

            st.sidebar.caption(
                f"Sektor: {fv['Sector']} • EPS: {fv['EPS']} • PER: {fv['PER']}"
            )
    else:
        st.sidebar.info("Saham belum masuk scanner")

# =====================
# TABLE
# =====================
st.subheader("📊 Hasil Scanner")

if df.empty:
    st.warning("Tidak ada data")
else:
    st.dataframe(
        df[["Kode", "Harga", "Trend", "TP", "SL"]],
        use_container_width=True,
        hide_index=True
    )

# =====================
# FOOTER
# =====================
st.caption(
    f"Auto Update • {datetime.now().strftime('%d %b %Y %H:%M')}"
)
