import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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
# CSS RESPONSIVE
# =====================
st.markdown("""
<style>
/* Desktop table hanya untuk layar > 600px */
@media only screen and (max-width: 600px) {
    .desktop-table {display: none;}
}
/* Mobile expander hanya untuk layar <= 600px */
@media only screen and (min-width: 601px) {
    .mobile-expander {display: none;}
}
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
.header-box {background: linear-gradient(135deg, #0e1117, #151b2c); padding: 20px; border-radius: 16px; margin-bottom: 25px;}
[data-testid="stMetric"] {background-color: #0e1117; padding: 12px; border-radius: 10px;}
.stProgress > div > div {background-color: #00c176;}
</style>
""", unsafe_allow_html=True)

# =====================
# HEADER IMAGE
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(BASE_DIR, "logo.png")
logo = Image.open(logo_path) if os.path.exists(logo_path) else None

col1, col2 = st.columns([1, 7])
with col1:
    if logo:
        st.image(logo, width=140)
with col2:
    st.markdown("""
        <h1 style="color: white; font-weight: 800;">📈 Swing Trading Scanner</h1>
        <p style="color: #9aa0a6; font-size: 14px;">Realtime Daily Market Screening • Indonesia Stock Exchange</p>
        """, unsafe_allow_html=True)
    st.caption(
        "Screener ini alat bantu teknis. Hasil screening bukan perintah beli; lakukan DYOR dan manajemen risiko."
    )

# =====================
# SIDEBAR CONFIG
# =====================
TP_PCT = st.sidebar.slider("Take Profit (%)", 3, 20, 5)
SL_PCT = st.sidebar.slider("Stop Loss (%)", 2, 10, 3)

# =====================
# HELPER FUNCTION
# =====================
def S(x):
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return x.astype(float)

def plot_last_2_candles(df):
    df = df.copy()
    for col in ["Open", "Close", "High", "Low"]:
        if col not in df.columns:
            df[col] = np.nan
    numeric_df = df[["Open","Close","High","Low"]].apply(pd.to_numeric, errors='coerce')
    df2 = df[numeric_df.notna().all(axis=1)].tail(2)
    if df2.empty:
        fig, ax = plt.subplots(figsize=(1.2,1), dpi=140)
        ax.text(0.5,0.5,"No valid candlestick", ha="center", va="center")
        ax.axis("off")
        return fig
    opens = df2["Open"].to_numpy(dtype=float)
    closes = df2["Close"].to_numpy(dtype=float)
    highs = df2["High"].to_numpy(dtype=float)
    lows = df2["Low"].to_numpy(dtype=float)
    if len(df2) < 2:
        opens = np.pad(opens, (2-len(opens),0), mode='edge')
        closes = np.pad(closes, (2-len(closes),0), mode='edge')
        highs = np.pad(highs, (2-len(highs),0), mode='edge')
        lows = np.pad(lows, (2-len(lows),0), mode='edge')
    fig, ax = plt.subplots(figsize=(1.2,1), dpi=140)
    for i in range(2):
        o, c, h, l = opens[i], closes[i], highs[i], lows[i]
        color = "#00C176" if c>=o else "#FF4D4D"
        ax.plot([i,i],[l,h], color=color, linewidth=1)
        ax.bar(i, abs(c-o), bottom=min(o,c), width=0.35, color=color)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.6,1.4); ax.set_ylim(min(lows)*0.995, max(highs)*1.005)
    ax.set_frame_on(False); plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
    return fig

def detect_macd_signal(close):
    macd_ind = MACD(close)
    macd_line = macd_ind.macd()
    signal    = macd_ind.macd_signal()
    if len(macd_line)<2: return "Normal"
    if macd_line.iloc[-2]<signal.iloc[-2] and macd_line.iloc[-1]>signal.iloc[-1]: return "Golden Cross"
    elif macd_line.iloc[-2]>signal.iloc[-2] and macd_line.iloc[-1]<signal.iloc[-1]: return "Death Cross"
    return "Normal"

def detect_ma_position(close):
    mas=[("MA5",5),("MA10",10),("MA20",20),("MA50",50),("MA100",100),("MA200",200)]
    price=close.iloc[-1]; above=[]
    for name,period in mas:
        if len(close)>=period:
            ma=close.rolling(period).mean().iloc[-1]
            if price>ma: above.append(name)
    if not above: return "—"
    return "⬆ " + " ".join(above)

def detect_trend(close):
    ema20 = EMAIndicator(close,20).ema_indicator()
    ema50 = EMAIndicator(close,50).ema_indicator()
    return "Bullish" if ema20.iloc[-1]>ema50.iloc[-1] else "Bearish"

def detect_zone(df):
    support=S(df["Low"]).rolling(20).min().iloc[-1]
    resistance=S(df["High"]).rolling(20).max().iloc[-1]
    price=S(df["Close"]).iloc[-1]
    if price<=support*1.03: return "BUY ZONE"
    elif price>=resistance*0.97: return "SELL ZONE"
    return "MID"

def detect_candle(df):
    o=S(df["Open"]).iloc[-1]; c=S(df["Close"]).iloc[-1]; h=S(df["High"]).iloc[-1]; l=S(df["Low"]).iloc[-1]
    body=abs(c-o); lower=min(o,c)-l; upper=h-max(o,c)
    if lower>body*2: return "Hammer","Bullish"
    if upper>body*2: return "Shooting Star","Bearish"
    return "Normal","Neutral"

def build_signal(zone,bias,trend):
    if zone=="BUY ZONE" and bias=="Bullish" and trend=="Bullish": return "BUY"
    return "HOLD"

# =====================
# LOAD TICKERS
# =====================
@st.cache_data
def load_idx_tickers():
    df=pd.read_csv("idx_tickers.csv")
    return (df["Kode"] + ".JK").tolist()
TICKERS = load_idx_tickers()
PERIOD="6mo"; INTERVAL="1d"

# =====================
# PROCESS DATA
# =====================
rows=[]
for t in TICKERS:
    try:
        df=yf.download(t,period=PERIOD,interval=INTERVAL,progress=False)
        if isinstance(df.columns,pd.MultiIndex):
            df.columns=df.columns.get_level_values(0)
        if df.empty or len(df)<60: continue
        df=df.dropna(); close=S(df["Close"]); price=close.iloc[-1]
        macd_signal=detect_macd_signal(close)
        ma_pos=detect_ma_position(close)
        trend=detect_trend(close)
        zone=detect_zone(df)
        candle,bias=detect_candle(df)
        rsi=RSIIndicator(close,14).rsi().iloc[-1]
        signal=build_signal(zone,bias,trend)
        tp=price*(1+TP_PCT/100); sl=price*(1-SL_PCT/100)
        confidence=0
        confidence+=1 if trend=="Bullish" else 0
        confidence+=1 if zone=="BUY ZONE" else 0
        confidence+=1 if bias=="Bullish" else 0
        confidence+=1 if rsi<40 else 0
        rows.append({
            "Kode": t, "Harga": round(price,2), "Signal":"BUY" if signal=="BUY" else "HOLD",
            "Trend":trend, "Zone":zone, "Candle":candle, "RSI":round(rsi,1),
            "MA_Pos":ma_pos,"MACD":macd_signal,"TP":round(tp,2),"SL":round(sl,2),
            "Confidence":confidence,"_df":df.copy()
        })
    except: pass

df=pd.DataFrame(rows).sort_values(by=["Confidence","Signal","RSI"],ascending=[False,False,True])

# =====================
# DISPLAY TABLE (Desktop)
# =====================
st.subheader("📊 Market Signal Overview")
st.markdown('<div class="desktop-table">', unsafe_allow_html=True)

ROW_HEIGHT=70
for _, row in df.iterrows():
    c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12 = st.columns([1.2,1,1,1,1,0.8,1.2,1.2,1,1,1,1])
    c1.write(row["Kode"].replace(".JK","")); c2.write(row["Harga"])
    c3.markdown(f"<span style='color:{'#00C176' if row['Signal']=='BUY' else '#999'}; font-weight:bold'>{row['Signal']}</span>",unsafe_allow_html=True)
    c4.markdown(f"<span style='font-size:13px; color:{'#00C176' if row['Trend']=='Bullish' else '#FF4D4D'}; font-weight:600'>{'🟢' if row['Trend']=='Bullish' else '🔴'} {row['Trend']}</span>",unsafe_allow_html=True)
    c5.markdown(f"<span style='color:{'#00C176' if row['Zone']=='BUY ZONE' else '#FF4D4D' if row['Zone']=='SELL ZONE' else '#999'}; font-weight:600'>{row['Zone']}</span>",unsafe_allow_html=True)
    fig=plot_last_2_candles(row["_df"]); c6.pyplot(fig, clear_figure=True)
    c7.write(row["MA_Pos"]); c8.markdown(f"{'🟢 Golden Cross' if row['MACD']=='Golden Cross' else '🔴 Death Cross' if row['MACD']=='Death Cross' else '⚪ Normal'}")
    rsi=row["RSI"]; c9.markdown(f"{'🟢' if rsi<40 else '🔴' if rsi>70 else '⚪'} {rsi}")
    c10.markdown(f"💰 {row['TP']}"); c11.markdown(f"🛑 {row['SL']}")

    # Sparkline (1D safe)
    close_values = pd.to_numeric(row["_df"]["Close"].tail(90), errors='coerce').dropna().values
    if len(close_values)==0: continue
    min_val,max_val=close_values.min(),close_values.max()
    norm_values = np.full_like(close_values,0.5) if max_val-min_val==0 else (close_values-min_val)/(max_val-min_val)
    norm_values = np.clip(norm_values,0,1)
    data=pd.DataFrame({'index':range(len(norm_values)),'close':norm_values})
    trend_color = "#00C176" if close_values[-1]>close_values[0] else "#FF4D4D" if close_values[-1]<close_values[0] else "#999"
    chart = alt.Chart(data).mark_line(color=trend_color, strokeWidth=1.8).encode(
        x=alt.X('index',axis=None),
        y=alt.Y('close',axis=None, scale=alt.Scale(domain=[0,1]))
    ).properties(height=30)
    c12.altair_chart(chart,use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# =====================
# CONFIDENCE METER
# =====================
st.subheader("🎯 Confidence Meter")
for _, row in df.iterrows():
    score=row["Confidence"]
    col1,col2=st.columns([1.2,4])
    col1.markdown(f"**{row['Kode'].replace('.JK','')}**"); col1.markdown("🟢 **BUY**" if row["Signal"]=="BUY" else "⚪ HOLD")
    col2.progress(score/4)
    if score==4: col2.caption("🔥 Sangat Kuat (4/4 indikator)")
    elif score==3: col2.caption("✅ Kuat (3/4 indikator)")
    elif score==2: col2.caption("⚠ Cukup (2/4 indikator)")
    else: col2.caption("❌ Lemah (<2 indikator)")

# =====================
# TOP BUY
# =====================
st.subheader("🔥 TOP BUY (Ranking Terkuat)")
top_buy=df[df["Signal"]=="BUY"].head(10)
if top_buy.empty: st.info("Belum ada BUY signal kuat hari ini")
else: st.dataframe(top_buy[["Kode","Harga","Trend","Zone","Candle","RSI","TP","SL","Confidence"]],use_container_width=True,hide_index=True)

# =====================
# FOOTER
# =====================
st.caption(f"Update otomatis harian • Last update: {datetime.now().strftime('%d %b %Y %H:%M')}")
