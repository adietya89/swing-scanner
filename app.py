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
.header-box {
    background: linear-gradient(135deg, #0e1117, #151b2c);
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 25px;
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
    st.markdown(
        """
        <div style="
            background-color: white;
            padding: 10px;
            border-radius: 12px;
        ">
        """,
        unsafe_allow_html=True
    )
    st.image(logo, width=140)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown(
        """
        <h1 style="
            color: white;
            font-weight: 800;
            margin-bottom: 4px;
        ">
        📈 Swing Trading Scanner
        </h1>

        <p style="
            color: #9aa0a6;
            font-size: 14px;
        ">
        Realtime Daily Market Screening • Indonesia Stock Exchange
        </p>
        """,
        unsafe_allow_html=True
    )    
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
    try:
        # Coba baca CSV default (comma)
        df = pd.read_csv("idx_tickers.csv")
        # Jika hanya ada satu kolom tanpa header, beri nama
        if df.shape[1] == 1:
            df.columns = ["Kode"]
    except pd.errors.ParserError:
        # Kalau gagal, coba tab atau semicolon
        try:
            df = pd.read_csv("idx_tickers.csv", sep="\t", header=None, names=["Kode"])
        except:
            df = pd.read_csv("idx_tickers.csv", sep=";", header=None, names=["Kode"])
    
    # Pastikan kolom "Kode" ada
    if "Kode" not in df.columns:
        df.columns = ["Kode"]
    
    # Buat list ticker dengan suffix .JK
    tickers = (df["Kode"].astype(str).str.strip() + ".JK").tolist()
    return tickers

TICKERS = load_idx_tickers()
PERIOD = "6mo"
INTERVAL = "1d"

TP_PCT = st.sidebar.slider("Take Profit (%)", 3, 20, 5)
SL_PCT = st.sidebar.slider("Stop Loss (%)", 2, 10, 3)
st.sidebar.divider()

scan_mode = st.sidebar.radio(
    "⚡ Mode Scan",
    ["Scan Cepat", "Scan Lengkap"],
    index=0
)

run_scan = st.sidebar.button("🚀 Mulai Scan Saham")
fake_rebound_filter = st.sidebar.checkbox(
    "Filter Fake Rebound",
    value=False
)

# =====================
# HELPER (ANTI ERROR)
# =====================
def distance_to_ma(close, period):
    if len(close) < period:
        return np.nan
    ma = close.rolling(period).mean().iloc[-1]
    price = close.iloc[-1]
    return abs(price - ma) / ma * 100

def S(x):
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]   # ambil kolom pertama saja
    return x.astype(float)
    
def plot_last_2_candles(df):
    """
    Plot 2 candlestick terakhir yang valid (OHLC numeric, tidak NaN)
    Ukuran pas untuk Streamlit
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Pastikan df adalah DataFrame
    if not isinstance(df, pd.DataFrame):
        fig, ax = plt.subplots(figsize=(1.2, 1), dpi=140)
        ax.text(0.5, 0.5, "No DataFrame", ha="center", va="center")
        ax.axis("off")
        return fig

    df = df.copy()

    # Pastikan kolom OHLC ada
    for col in ["Open", "Close", "High", "Low"]:
        if col not in df.columns:
            df[col] = np.nan

    # Filter numeric, aman dari NaN/non-numeric
    numeric_df = df[["Open","Close","High","Low"]].apply(pd.to_numeric, errors='coerce')
    df2 = df[numeric_df.notna().all(axis=1)].tail(2)

    if df2.empty:
        fig, ax = plt.subplots(figsize=(1.2, 1), dpi=140)
        ax.text(0.5, 0.5, "No valid candlestick", ha="center", va="center")
        ax.axis("off")
        return fig

    # Ambil angka murni, pastikan 1D float
    opens = df2["Open"].to_numpy(dtype=float).flatten()
    closes = df2["Close"].to_numpy(dtype=float).flatten()
    highs = df2["High"].to_numpy(dtype=float).flatten()
    lows = df2["Low"].to_numpy(dtype=float).flatten()

    # Jika kurang dari 2 candlestick, ulang baris terakhir
    if len(df2) < 2:
        opens = np.pad(opens, (2 - len(opens), 0), mode='edge')
        closes = np.pad(closes, (2 - len(closes), 0), mode='edge')
        highs = np.pad(highs, (2 - len(highs), 0), mode='edge')
        lows = np.pad(lows, (2 - len(lows), 0), mode='edge')

    fig, ax = plt.subplots(figsize=(1.2, 1), dpi=140)

    for i in range(2):
        o, c, h, l = opens[i], closes[i], highs[i], lows[i]
        color = "#00C176" if c >= o else "#FF4D4D"

        # Wick
        ax.plot([i, i], [l, h], color=color, linewidth=1)

        # Body
        ax.bar(
            i,
            abs(c - o),
            bottom=min(o, c),
            width=0.35,
            color=color
        )

    # Axis off & batasan
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.6, 1.4)
    ax.set_ylim(min(lows)*0.995, max(highs)*1.005)
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
# FAKE REBOUND DETECTION
# =====================
def detect_fake_rebound(close, df):
    price = close.iloc[-1]
    candle_pattern, bias = detect_candle(df)
    rsi = RSIIndicator(close, 14).rsi().iloc[-1]
    macd_signal = detect_macd_signal(close)
    ema50 = EMAIndicator(close, 50).ema_indicator().iloc[-1]

    # Syarat fake rebound: harga < EMA50, RSI < 50, candle shooting star, dan MACD bukan golden cross
    if price < ema50 and rsi < 50 and candle_pattern == "Shooting Star" and macd_signal != "Golden Cross":
        return True
    return False

@st.cache_data(ttl=60*60*24)
def scan_saham(tickers, scan_mode, TP_PCT, SL_PCT):
    rows = []

    for t in tickers:
        try:
            df = yf.download(t, period="6mo", interval="1d", progress=False)

            if df.empty or len(df) < 60:
                continue

            df = df.dropna()
            close = df["Close"]

            price = close.iloc[-1]

            # ===== MODE CEPAT =====
            if scan_mode == "Scan Cepat":
                rsi = RSIIndicator(close, 14).rsi().iloc[-1]
                trend = detect_trend(close)

                rows.append({
                    "Kode": t,
                    "Harga": round(price, 2),
                    "Trend": trend,
                    "RSI": round(rsi, 1),
                    "Signal": "BUY" if rsi < 40 and trend == "Bullish" else "HOLD",
                    "_df": df
                })

            # ===== MODE LENGKAP =====
            else:
                macd_signal = detect_macd_signal(close)
                zone = detect_zone(df)
                candle, bias = detect_candle(df)
                rsi = RSIIndicator(close, 14).rsi().iloc[-1]

                tp = price * (1 + TP_PCT / 100)
                sl = price * (1 - SL_PCT / 100)

                rows.append({
                    "Kode": t,
                    "Harga": round(price, 2),
                    "Trend": detect_trend(close),
                    "Zone": zone,
                    "Candle": candle,
                    "RSI": round(rsi, 1),
                    "MACD": macd_signal,
                    "TP": round(tp, 2),
                    "SL": round(sl, 2),
                    "Signal": "BUY" if zone == "BUY ZONE" else "HOLD",
                    "_df": df
                })

        except:
            pass

    return pd.DataFrame(rows)

# =====================
# DATA PROCESS
# =====================
rows = []

if not run_scan:
    st.info("👈 Klik tombol **Mulai Scan Saham** di kiri untuk mulai")
    st.stop()

if not run_scan:
    st.info("👈 Pilih mode scan lalu klik **Mulai Scan Saham**")
    st.stop()

with st.spinner("⏳ Scan saham berjalan..."):
    df = scan_saham(TICKERS, scan_mode, TP_PCT, SL_PCT)

# Lengkapi kolom biar UI aman
default_cols = {
    "Zone": "-",
    "Candle": "-",
    "MACD": "-",
    "TP": 0,
    "SL": 0,
    "MA_Pos": "-",
    "Confidence": 0,
    "BUY_Filter": False,
    "SELL_Filter": False,
    "Fake_Rebound": False
}

for col, val in default_cols.items():
    if col not in df.columns:
        df[col] = val

df["Fake_Rebound"] = df["Fake_Rebound"].astype(bool)
df = df.sort_values(
    by=["Confidence", "Signal", "RSI"],
    ascending=[False, False, True]
)
# =====================
# UI TABLE
# =====================
st.subheader("📊 • INFEKSIUS ACTIO")

st.subheader("🎯 FILTER STRATEGI")

f1, f2, f3, f4 = st.columns([2, 1, 1, 1])

if "trade_filter" not in st.session_state:
    st.session_state.trade_filter = "ALL"

search_code = f1.text_input(
    "🔍 Cari Kode Saham",
    placeholder="contoh: BBCA / TLKM / BBRI"
).upper()

if f2.button("📦 ALL"):
    st.session_state.trade_filter = "ALL"

if f3.button("🟢 BUY"):
    st.session_state.trade_filter = "BUY"

if f4.button("🔴 SELL"):
    st.session_state.trade_filter = "SELL"

if df.empty:
    st.warning("Belum ada data")

# Header tabel
h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13 = st.columns(
    [1.2, 1, 1, 1, 1, 0.8, 1.2, 1.2, 1.2, 1, 1, 1, 1]
)

h1.markdown("**Kode**")
h2.markdown("**Harga**")
h3.markdown("**Signal**")
h4.markdown("**Trend**")
h5.markdown("**Zone**")
h6.markdown("**Candle**")
h7.markdown("**Candle Pattern**")
h8.markdown("**MA >**")
h9.markdown("**MACD**")
h10.markdown("**RSI**")
h11.markdown("**TP**")
h12.markdown("**SL**")
h13.markdown("**S'LINE**")

st.divider()

ROW_HEIGHT = 70

# =====================
# Kolom 1 - 9
# =====================
# Default: tabel KOSONG
filtered_df = pd.DataFrame()

# Jika user klik filter
if st.session_state.trade_filter == "ALL":
    filtered_df = df.copy()

elif st.session_state.trade_filter == "BUY":
    filtered_df = df[df["BUY_Filter"]]

elif st.session_state.trade_filter == "SELL":
    filtered_df = df[df["SELL_Filter"]]

# Jika user pakai search
if search_code:
    filtered_df = df[
        df["Kode"].str.contains(search_code, case=False)
    ]

# Filter fake rebound
if fake_rebound_filter and not filtered_df.empty:
    filtered_df = filtered_df[filtered_df["Fake_Rebound"] == False]

if st.session_state.trade_filter == "BUY":
    filtered_df = df[df["BUY_Filter"]]

elif st.session_state.trade_filter == "SELL":
    filtered_df = df[df["SELL_Filter"]]
    
# 🔍 Filter cari 1 saham
if search_code:
    filtered_df = filtered_df[
        filtered_df["Kode"].str.contains(search_code, case=False)
    ]
    
# Filter fake rebound
if fake_rebound_filter:
    filtered_df = filtered_df[filtered_df["Fake_Rebound"] == False]

if filtered_df.empty:
    st.info("🔎 Silakan pilih filter (ALL / BUY / SELL) atau cari kode saham")
    st.stop()

for _, row in filtered_df.iterrows():
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13 = st.columns(
        [1.2, 1, 1, 1, 1, 0.8, 1.2, 1.2, 1.2, 1, 1, 1, 1]
    )

    with c1:
        st.write(row["Kode"].replace(".JK",""))

    with c2:
        st.write(row["Harga"])

    with c3:
        if row["Signal"] == "BUY":
            st.markdown("<span style='color:#00C176; font-weight:bold'>BUY</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#999'>HOLD</span>", unsafe_allow_html=True)

    with c4:
        if row["Trend"] == "Bullish":
            st.markdown("<span style='font-size:13px; color:#00C176; font-weight:600'>🟢 Bullish</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='font-size:13px; color:#FF4D4D; font-weight:600'>🔴 Bearish</span>", unsafe_allow_html=True)

    with c5:
        if row["Zone"] == "BUY ZONE":
            st.markdown("<span style='color:#00C176; font-weight:600'>BUY</span>", unsafe_allow_html=True)
        elif row["Zone"] == "SELL ZONE":
            st.markdown("<span style='color:#FF4D4D; font-weight:600'>SELL</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#999'>MID</span>", unsafe_allow_html=True)

    with c6:
        fig = plot_last_2_candles(row["_df"])
        st.pyplot(fig, clear_figure=True)

    with c7:
        candle_pattern, bias = detect_candle(row["_df"])
        if candle_pattern != "Normal":
            color = "#00C176" if bias == "Bullish" else "#FF4D4D"
            st.markdown(f"<span style='color:{color}; font-weight:600'>{candle_pattern}</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#999'>Normal</span>", unsafe_allow_html=True)

    with c8:
        st.write(row["MA_Pos"])

    with c9:
        macd = row["MACD"]
        if macd == "Golden Cross":
            st.markdown("🟢 **Golden Cross**")
        elif macd == "Death Cross":
            st.markdown("🔴 **Death Cross**")
        else:
            st.markdown("⚪ Normal")

    with c10:
        rsi = row["RSI"]
        if rsi < 40:
            st.markdown(f"🟢 **{rsi}**")
        elif rsi > 70:
            st.markdown(f"🔴 **{rsi}**")
        else:
            st.markdown(f"⚪ {rsi}")

    with c11:
        st.markdown(f"<span style='font-size:13px'>💰 {row['TP']}</span>", unsafe_allow_html=True)

    with c12:
        st.markdown(f"<span style='font-size:13px'>🛑 {row['SL']}</span>", unsafe_allow_html=True)

    with c13:
        # Sparkline (Altair chart)
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

            data = pd.DataFrame({'index': range(len(norm_values)), 'close': norm_values})

            trend_color = "#999"
            if close_values[-1] > close_values[0]:
                trend_color = "#00C176"   # naik
            elif close_values[-1] < close_values[0]:
                trend_color = "#FF4D4D"   # turun

            chart = (
                alt.Chart(data)
                .mark_line(color=trend_color, strokeWidth=1.8)
                .encode(x=alt.X('index', axis=None), y=alt.Y('close', axis=None, scale=alt.Scale(domain=[0,1])))
                .properties(height=30)
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            st.write("-")
            
# =====================
# CONFIDENCE METER
# =====================
st.subheader("🎯 Confidence Meter")

for _, row in filtered_df.iterrows():
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

top_buy = filtered_df[
    (filtered_df["Signal"] == "BUY") &
    (filtered_df["BUY_Filter"] == True)
].head(10)

if top_buy.empty:
    st.info("Belum ada BUY signal kuat hari ini")
else:
    st.dataframe(
        top_buy[
            ["Kode", "Harga", "Trend", "Zone", "Candle", "RSI", "TP", "SL", "Confidence"]
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





















