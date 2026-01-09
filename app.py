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

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Swing Trading Scanner",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top:1rem;}
[data-testid="stMetric"] {
    background:#0e1117;
    padding:12px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# =====================
# HEADER
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo = Image.open(os.path.join(BASE_DIR, "logo.png"))

c1, c2 = st.columns([1, 7])
with c1:
    st.image(logo, width=120)
with c2:
    st.markdown("## 📈 Swing Trading Scanner")
    st.caption("Realtime Daily Market Screening • IDX")

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
# HELPERS
# =====================
def S(x):
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return x.astype(float)

def plot_last_2_candles(df):
    df = df.tail(2)
    fig, ax = plt.subplots(figsize=(1.2,1), dpi=140)
    for i, r in enumerate(df.itertuples()):
        o,c,h,l = r.Open, r.Close, r.High, r.Low
        color = "#00C176" if c>=o else "#FF4D4D"
        ax.plot([i,i],[l,h], color=color)
        ax.bar(i, abs(c-o), bottom=min(o,c), color=color, width=0.35)
    ax.axis("off")
    return fig

# =====================
# LOGIC
# =====================
def detect_macd(close):
    macd = MACD(close)
    if len(close)<2: return "Normal"
    if macd.macd().iloc[-2] < macd.macd_signal().iloc[-2] and macd.macd().iloc[-1] > macd.macd_signal().iloc[-1]:
        return "Golden Cross"
    if macd.macd().iloc[-2] > macd.macd_signal().iloc[-2] and macd.macd().iloc[-1] < macd.macd_signal().iloc[-1]:
        return "Death Cross"
    return "Normal"

def detect_trend(close):
    return "Bullish" if EMAIndicator(close,20).ema_indicator().iloc[-1] > EMAIndicator(close,50).ema_indicator().iloc[-1] else "Bearish"

def detect_zone(df):
    sup = S(df["Low"]).rolling(20).min().iloc[-1]
    res = S(df["High"]).rolling(20).max().iloc[-1]
    price = S(df["Close"]).iloc[-1]
    if price <= sup*1.03: return "BUY ZONE"
    if price >= res*0.97: return "SELL ZONE"
    return "MID"

def detect_candle(df):
    o,c,h,l = S(df["Open"]).iloc[-1],S(df["Close"]).iloc[-1],S(df["High"]).iloc[-1],S(df["Low"]).iloc[-1]
    body = abs(c-o)
    if min(o,c)-l > body*2: return "Hammer","Bullish"
    if h-max(o,c) > body*2: return "Shooting Star","Bearish"
    return "Normal","Neutral"

def classify_status(row):
    if row["Signal"]=="BUY" and row["Zone"]=="BUY ZONE":
        return "BUY CONFIRM"
    if row["Zone"]=="BUY ZONE":
        return "BUY ZONE"
    if row["Zone"]=="SELL ZONE":
        return "SELL ZONE"
    return "HOLD"

# =====================
# DATA PROCESS
# =====================
rows=[]
for t in TICKERS:
    try:
        df = yf.download(t, period=PERIOD, interval=INTERVAL, progress=False)
        if df.empty or len(df)<60: continue
        close = S(df["Close"])
        price = close.iloc[-1]

        trend = detect_trend(close)
        zone = detect_zone(df)
        candle,bias = detect_candle(df)
        macd = detect_macd(close)
        rsi = RSIIndicator(close).rsi().iloc[-1]

        signal = "BUY" if trend=="Bullish" and zone=="BUY ZONE" and bias=="Bullish" else "HOLD"

        confidence = sum([
            trend=="Bullish",
            zone=="BUY ZONE",
            bias=="Bullish",
            rsi<40
        ])

        rows.append({
            "Kode":t.replace(".JK",""),
            "Harga":round(price,2),
            "Signal":signal,
            "Trend":trend,
            "Zone":zone,
            "Candle":candle,
            "RSI":round(rsi,1),
            "MACD":macd,
            "TP":round(price*(1+TP_PCT/100),2),
            "SL":round(price*(1-SL_PCT/100),2),
            "Confidence":confidence,
            "_df":df
        })
    except:
        pass

df = pd.DataFrame(rows)
df["Status"] = df.apply(classify_status, axis=1)

# =====================
# FILTER BAR (LIKE IMAGE)
# =====================
st.subheader("📊 Hasil Screener")

if "filter" not in st.session_state:
    st.session_state.filter="ALL"

c1,c2,c3,c4,c5 = st.columns(5)

if c1.button(f"📦 Total {len(df)}"):
    st.session_state.filter="ALL"
if c2.button(f"🎯 Buy Zone {len(df[df.Status=='BUY ZONE'])}"):
    st.session_state.filter="BUY ZONE"
if c3.button(f"🟢 Buy Confirm {len(df[df.Status=='BUY CONFIRM'])}"):
    st.session_state.filter="BUY CONFIRM"
if c4.button(f"🔴 Sell Zone {len(df[df.Status=='SELL ZONE'])}"):
    st.session_state.filter="SELL ZONE"
if c5.button("⚪ Hold"):
    st.session_state.filter="HOLD"

if st.session_state.filter!="ALL":
    df=df[df.Status==st.session_state.filter]

# =====================
# TABLE
# =====================
st.divider()

for _,r in df.iterrows():
    a,b,c,d,e,f = st.columns([1.2,1,1,1,1,2])
    a.write(r["Kode"])
    b.write(r["Harga"])
    c.markdown(f"**{r['Status']}**")
    d.write(r["Trend"])
    e.write(r["RSI"])
    with f:
        st.pyplot(plot_last_2_candles(r["_df"]), clear_figure=True)

# =====================
# FOOTER
# =====================
st.caption(f"Auto update harian • {datetime.now().strftime('%d %b %Y %H:%M')}")
