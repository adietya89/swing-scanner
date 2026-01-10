# =====================
# IMPORT
# =====================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from datetime import datetime

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="IDX Swing Trading Scanner",
    layout="wide"
)

st.markdown("""
<style>
.block-container { padding-top: 1rem; }
[data-testid="stMetric"] {
    background-color: #0e1117;
    padding: 12px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================
# HEADER
# =====================
st.title("📈 IDX Swing Trading Scanner")
st.caption("Scanner teknikal harian + harga wajar sederhana | Bukan rekomendasi beli")

# =====================
# CONFIG
# =====================
PERIOD = "6mo"
INTERVAL = "1d"

TP_PCT = st.sidebar.slider("Take Profit (%)", 3, 20, 5)
SL_PCT = st.sidebar.slider("Stop Loss (%)", 2, 10, 3)

show_fair_value = st.sidebar.checkbox("Tampilkan Harga Wajar", True)
search_fair = st.sidebar.text_input("Cari Harga Wajar (BBRI / BBCA)").upper()

# =====================
# LOAD TICKER IDX
# =====================
@st.cache_data
def load_tickers():
    df = pd.read_csv("idx_tickers.csv")
    df.columns = ["Kode"]
    return (df["Kode"].str.strip() + ".JK").tolist()

TICKERS = load_tickers()

# =====================
# HELPER
# =====================
def S(x):
    return x.astype(float)

def detect_trend(close):
    ema20 = EMAIndicator(close, 20).ema_indicator()
    ema50 = EMAIndicator(close, 50).ema_indicator()
    return "Bullish" if ema20.iloc[-1] > ema50.iloc[-1] else "Bearish"

def detect_macd(close):
    macd = MACD(close)
    if macd.macd().iloc[-1] > macd.macd_signal().iloc[-1]:
        return "Bullish"
    return "Bearish"

def detect_zone(df):
    support = df["Low"].rolling(20).min().iloc[-1]
    resistance = df["High"].rolling(20).max().iloc[-1]
    price = df["Close"].iloc[-1]
    if price <= support * 1.03:
        return "BUY ZONE"
    elif price >= resistance * 0.97:
        return "SELL ZONE"
    return "MID"

def fair_value_simple(ticker, price):
    try:
        info = yf.Ticker(ticker).info
        eps = info.get("trailingEps", None)
        if eps is None or eps <= 0:
            return None
        per = 12
        fair = eps * per
        margin = (fair - price) / price * 100
        return round(fair,2), round(margin,1)
    except:
        return None

# =====================
# PROCESS DATA
# =====================
rows = []

with st.spinner("⏳ Mengambil data saham IDX..."):
    for t in TICKERS:
        try:
            df = yf.download(t, period=PERIOD, interval=INTERVAL, progress=False)
            if df.empty or len(df) < 60:
                continue

            close = S(df["Close"])
            price = close.iloc[-1]

            rsi = RSIIndicator(close, 14).rsi().iloc[-1]
            trend = detect_trend(close)
            macd = detect_macd(close)
            zone = detect_zone(df)

            signal = "BUY" if (
                trend == "Bullish" and
                macd == "Bullish" and
                zone == "BUY ZONE" and
                rsi < 50
            ) else "HOLD"

            rows.append({
                "Kode": t.replace(".JK",""),
                "Harga": round(price,2),
                "Signal": signal,
                "Trend": trend,
                "MACD": macd,
                "Zone": zone,
                "RSI": round(rsi,1),
                "TP": round(price * (1 + TP_PCT/100),2),
                "SL": round(price * (1 - SL_PCT/100),2),
            })
        except:
            pass

df = pd.DataFrame(rows)
df = df.sort_values(["Signal","RSI"], ascending=[False,True])

# =====================
# SIDEBAR – HARGA WAJAR
# =====================
if show_fair_value and search_fair:
    ticker = search_fair + ".JK"
    row = df[df["Kode"] == search_fair]

    st.sidebar.markdown("---")
    if not row.empty:
        price = row.iloc[0]["Harga"]
        fv = fair_value_simple(ticker, price)

        st.sidebar.metric("Harga Saat Ini", price)

        if fv:
            fair, margin = fv
            st.sidebar.metric("Harga Wajar", fair, f"{margin}%")

            if margin > 20:
                st.sidebar.success("🟢 Undervalued")
            elif margin < -10:
                st.sidebar.error("🔴 Overvalued")
            else:
                st.sidebar.warning("🟡 Fair Value")
        else:
            st.sidebar.warning("EPS tidak tersedia")

# =====================
# TABLE
# =====================
st.subheader("📊 Hasil Scanner")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

# =====================
# TOP BUY
# =====================
st.subheader("🔥 TOP BUY")

top_buy = df[df["Signal"] == "BUY"].head(10)
if top_buy.empty:
    st.info("Belum ada sinyal BUY kuat hari ini")
else:
    st.dataframe(top_buy, use_container_width=True, hide_index=True)

# =====================
# FOOTER
# =====================
st.caption(f"Update terakhir: {datetime.now().strftime('%d %b %Y %H:%M')}")
