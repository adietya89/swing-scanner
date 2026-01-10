# =========================================================
# MODERN SWING TRADING SCANNER (FINAL)
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
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Swing Trading Scanner",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# GLOBAL STYLE (MODERN)
# =========================================================
st.markdown("""
<style>
.block-container {padding-top:1.2rem;}
body {background-color:#0e1117;}
h1,h2,h3 {color:#eaecef;}
.stButton>button {
    border-radius:10px;
    padding:6px 14px;
    font-weight:600;
}
.buy {color:#00c176;font-weight:700;}
.sell {color:#ff4d4d;font-weight:700;}
.hold {color:#9aa0a6;}
.card {
    background:#151b2c;
    border-radius:14px;
    padding:10px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo = Image.open(os.path.join(BASE_DIR, "logo.png"))

c1, c2 = st.columns([1,8])
with c1:
    st.image(logo, width=120)
with c2:
    st.markdown("## 📈 Swing Trading Scanner")
    st.caption("Realtime Daily Market Screening • Indonesia Stock Exchange")
    st.markdown(
        "<span style='color:#9aa0a6'>Memasuki dinamika pasar tahun 2026, strategi swing trading memerlukan ketelitian dalam menangkap momentum harga...</span>",
        unsafe_allow_html=True
    )

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Trading Config")
TP_PCT = st.sidebar.slider("Take Profit (%)", 3, 20, 5)
SL_PCT = st.sidebar.slider("Stop Loss (%)", 2, 10, 3)

# =========================================================
# LOAD TICKER
# =========================================================
@st.cache_data
def load_ticker():
    df = pd.read_csv("idx_tickers.csv")
    if df.shape[1] == 1:
        df.columns = ["Kode"]
    return (df["Kode"] + ".JK").tolist()

TICKERS = load_ticker()

# =========================================================
# MINI CANDLE
# =========================================================
def plot_mini_candle(df):
    fig, ax = plt.subplots(figsize=(1.2,1), dpi=120)
    last = df.tail(2)
    for i, row in enumerate(last.itertuples()):
        o,c,h,l = row.Open,row.Close,row.High,row.Low
        color = "#00c176" if c>=o else "#ff4d4d"
        ax.plot([i,i],[l,h],color=color,linewidth=1)
        ax.bar(i,abs(c-o),bottom=min(o,c),color=color,width=0.4)
    ax.axis("off")
    return fig

# =========================================================
# FETCH DATA (FAST)
# =========================================================
@st.cache_data(ttl=86400)
def fetch_data():
    raw = yf.download(TICKERS, period="6mo", interval="1d", group_by="ticker", threads=True)
    rows=[]
    for t in TICKERS:
        if t not in raw: continue
        df = raw[t].dropna()
        if len(df)<60: continue

        close = df["Close"]
        price = close.iloc[-1]

        ema20 = EMAIndicator(close,20).ema_indicator()
        ema50 = EMAIndicator(close,50).ema_indicator()
        rsi = RSIIndicator(close).rsi().iloc[-1]
        macd = MACD(close)

        trend = "Bullish" if ema20.iloc[-1]>ema50.iloc[-1] else "Bearish"
        zone = "BUY" if price<=df["Low"].rolling(20).min().iloc[-1]*1.03 else "MID"
        candle,_ = ("Hammer","Bullish") if (df["Low"].iloc[-1]<df["Open"].iloc[-1]) else ("Normal","Neutral")

        rows.append({
            "Kode":t.replace(".JK",""),
            "Harga":round(price,2),
            "Trend":trend,
            "Zone":zone,
            "RSI":round(rsi,1),
            "MACD":"Golden" if macd.macd().iloc[-1]>macd.macd_signal().iloc[-1] else "Normal",
            "TP":round(price*(1+TP_PCT/100),2),
            "SL":round(price*(1-SL_PCT/100),2),
            "_df":df
        })
    return pd.DataFrame(rows)

df = fetch_data()

# =========================================================
# FILTER BAR
# =========================================================
st.markdown("### 🎯 Filter Strategi")
f1,f2,f3,f4 = st.columns([3,1,1,1])
search = f1.text_input("🔍 Cari Kode Saham").upper()
mode = st.session_state.get("mode","ALL")

if f2.button("📦 ALL"): mode="ALL"
if f3.button("🟢 BUY"): mode="BUY"
if f4.button("🔴 SELL"): mode="SELL"
st.session_state["mode"]=mode

if search:
    df = df[df["Kode"].str.contains(search)]

# =========================================================
# TABLE
# =========================================================
st.markdown("### 📊 INFEKSIUS ACTIO")

for _,row in df.iterrows():
    c1,c2,c3,c4,c5,c6,c7 = st.columns([1.2,1,1,1,1,1.4,1.4])
    with c1: st.write(row["Kode"])
    with c2: st.write(row["Harga"])
    with c3:
        st.markdown("<span class='buy'>BUY</span>" if row["Trend"]=="Bullish" else "<span class='hold'>HOLD</span>", unsafe_allow_html=True)
    with c4: st.write(row["Trend"])
    with c5: st.write(row["Zone"])
    with c6: st.pyplot(plot_mini_candle(row["_df"]), clear_figure=True)
    with c7:
        data = row["_df"]["Close"].tail(60).reset_index()
        chart = alt.Chart(data).mark_line(color="#00c176").encode(
            x=alt.X("index",axis=None),
            y=alt.Y("Close",axis=None)
        ).properties(height=35)
        st.altair_chart(chart, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================
st.caption(f"Update otomatis harian • {datetime.now().strftime('%d %b %Y %H:%M')}")
