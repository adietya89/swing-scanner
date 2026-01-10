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
fake_rebound_filter = st.sidebar.checkbox(
    "Filter Fake Rebound",
    value=False
)
# =====================
# SIDEBAR MENU PILIHAN
# =====================
menu_option = st.sidebar.radio(
    "📋 Pilih Menu",
    ["Harga Wajar", "Average Down"]
)
if menu_option == "Harga Wajar":
   fair_search = st.sidebar.text_input(
       "🔍 Cari Kode Saham",
       placeholder="BBRI / BBCA / TLKM"
   ).upper()

   show_fair_value = st.sidebar.checkbox(
       "Tampilkan Analisa Harga Wajar",
       value=True
   )

elif menu_option == "Average Down":
    st.sidebar.markdown("## 📌 Average Down (AVD)")

    ticker_avd = st.sidebar.text_input(
        "🔍 Masukkan Kode Saham",
        placeholder="BBRI / BBCA / TLKM"
    ).upper()

    # Harga beli awal
    initial_price = st.sidebar.number_input(
        "Harga Beli Awal",
        value=0.0,
        step=100.0,
        format="%.2f"
    )

    initial_lot = st.sidebar.number_input(
        "Lot Awal",
        value=0,
        step=1
    )

    # Harga setelah average down
    avd_price = st.sidebar.number_input(
        "Harga Terakhir Setelah Average Down",
        value=0.0,
        step=100.0,
        format="%.2f"
    )

    avd_lot = st.sidebar.number_input(
        "Lot Tambahan",
        value=0,
        step=1
    )
    # =====================
    # HITUNG HARGA RATA-RATA & TOTAL LOT
    # =====================
    if (initial_price > 0 and initial_lot > 0) or (avd_price > 0 and avd_lot > 0):
        total_lot = initial_lot + avd_lot
        if total_lot > 0:
            avg_price = (initial_price * initial_lot + avd_price * avd_lot) / total_lot
        else:
            avg_price = 0

        st.sidebar.markdown(f"💹 Harga Rata-rata: **{round(avg_price, 2)}**")
        st.sidebar.markdown(f"📦 Total Lot: **{total_lot}**")
        
# =====================
# HELPER (ANTI ERROR)
# =====================
def plot_price_fibo_snr(df, ticker):
    import matplotlib.pyplot as plt

    df = df.tail(120).copy()
    close = df["Close"].astype(float)

    # Ambil SNR & FIBO
    support, resistance = calculate_support_resistance(df)
    fib = calculate_fibonacci(df)

    fig, ax = plt.subplots(figsize=(6, 3))

    # Plot harga
    ax.plot(close.index, close.values, color="black", linewidth=1.2, label="Price")

    # Support & Resistance
    if support:
        ax.axhline(support, color="#00C176", linestyle="--", linewidth=1, label="Support")

    if resistance:
        ax.axhline(resistance, color="#FF4D4D", linestyle="--", linewidth=1, label="Resistance")

    # Fibonacci Lines
    if fib:
        for level, price in fib.items():
            ax.axhline(price, linestyle=":", linewidth=0.8, alpha=0.7)
            ax.text(
                close.index[-1],
                price,
                f"Fib {level}",
                fontsize=7,
                verticalalignment="center"
            )

    ax.set_title(f"{ticker.replace('.JK','')} • Price + Fibo + SNR", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    return fig

def calculate_fibonacci(df, lookback=60):
    if df is None or len(df) < lookback:
        return None

    high = df["High"].astype(float).tail(lookback).max()
    low  = df["Low"].astype(float).tail(lookback).min()

    fib = {
        "0.236": high - (high - low) * 0.236,
        "0.382": high - (high - low) * 0.382,
        "0.5":   high - (high - low) * 0.5,
        "0.618": high - (high - low) * 0.618,
        "0.786": high - (high - low) * 0.786,
    }
    return fib

def distance_to_ma(close, period):
    if len(close) < period:
        return np.nan
    ma = close.rolling(period).mean().iloc[-1]
    price = close.iloc[-1]
    return abs(price - ma) / ma * 100

def S(x):
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]   # ambil kolom pertama
    return pd.to_numeric(x, errors='coerce').dropna()
    
def plot_last_2_candles(df):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    if not isinstance(df, pd.DataFrame):
        fig, ax = plt.subplots(figsize=(1.2, 1), dpi=140)
        ax.text(0.5, 0.5, "No DataFrame", ha="center", va="center")
        ax.axis("off")
        return fig

    df = df.copy()
    for col in ["Open", "Close", "High", "Low"]:
        if col not in df.columns:
            df[col] = np.nan

    # Filter numeric & buang row invalid
    numeric_df = df[["Open", "Close", "High", "Low"]].apply(pd.to_numeric, errors='coerce')
    df2 = df[numeric_df.notna().all(axis=1)].tail(2)

    if df2.empty:
        fig, ax = plt.subplots(figsize=(1.2, 1), dpi=140)
        ax.text(0.5, 0.5, "No valid candlestick", ha="center", va="center")
        ax.axis("off")
        return fig

    opens = df2["Open"].to_numpy(dtype=float).flatten()
    closes = df2["Close"].to_numpy(dtype=float).flatten()
    highs = df2["High"].to_numpy(dtype=float).flatten()
    lows = df2["Low"].to_numpy(dtype=float).flatten()

    # Kalau kurang dari 2, ulang baris terakhir
    if len(df2) < 2:
        opens = np.pad(opens, (2 - len(opens), 0), mode='edge')
        closes = np.pad(closes, (2 - len(closes), 0), mode='edge')
        highs = np.pad(highs, (2 - len(highs), 0), mode='edge')
        lows = np.pad(lows, (2 - len(lows), 0), mode='edge')

    fig, ax = plt.subplots(figsize=(1.2, 1), dpi=140)
    for i in range(2):
        o, c, h, l = opens[i], closes[i], highs[i], lows[i]
        color = "#00C176" if c >= o else "#FF4D4D"
        ax.plot([i, i], [l, h], color=color, linewidth=1)
        ax.bar(i, abs(c - o), bottom=min(o, c), width=0.35, color=color)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.6, 1.4)
    ax.set_ylim(min(lows)*0.995, max(highs)*1.005)
    ax.set_frame_on(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig

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

def calculate_support_resistance(df, window=30):
    if df is None or len(df) < window:
        return None, None

    support = df["Low"].astype(float).rolling(window).min().iloc[-1]
    resistance = df["High"].astype(float).rolling(window).max().iloc[-1]

    return round(support, 2), round(resistance, 2)

# =====================
# HELPER – HARGA WAJAR SIMPLE
# =====================
def calculate_fair_value_simple(ticker, current_price):
    try:
        info = yf.Ticker(ticker).info

        eps = info.get("trailingEps", None)
        sector = info.get("sector", "Default")

        sector_per_map = {
            "Financial Services": 12,   # Bank
            "Consumer Defensive": 15,
            "Consumer Cyclical": 15,
            "Energy": 8,
            "Real Estate": 9,
            "Utilities": 10,
            "Industrials": 11,
            "Technology": 18,
            "Basic Materials": 10,
            "Default": 12
        }

        per_used = sector_per_map.get(sector, 12)

        if eps is None or eps <= 0:
            return None

        fair_price = eps * per_used
        margin = (fair_price - current_price) / current_price * 100

        return {
            "EPS": round(eps, 2),
            "PER_Sektor": per_used,
            "Fair_Price": round(fair_price, 2),
            "Margin": round(margin, 1),
            "Sector": sector
        }

    except Exception:
        return None

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
    fib = calculate_fibonacci(df)
    support, resistance = calculate_support_resistance(df)
    price = float(df["Close"].iloc[-1])

    if fib is None or support is None:
        return "MID"

    # BUY AREA (area murah)
    if fib["0.786"] <= price <= fib["0.618"] and price <= support * 1.05:
        return "BUY ZONE"

    # SELL AREA (area mahal)
    if fib["0.382"] <= price <= fib["0.236"] and price >= resistance * 0.95:
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

# =====================
# DATA PROCESS
# =====================
rows = []

with st.spinner("⏳ Mengambil dari data saham IDX ... Mohon tunggu beberapa menit !!! 😁😁😁"):
    for t in TICKERS:
        try:
            df = yf.download(t, period=PERIOD, interval=INTERVAL, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 60:
                continue

            df = df.dropna()
            close = pd.to_numeric(df["Close"], errors='coerce').dropna()
            if close.empty:
               continue  # skip saham ini kalau data kosong

            try:
                intraday = yf.Ticker(t).history(period="1d", interval="1m")
                if not intraday.empty:
                   price = intraday["Close"].iloc[-1]
                else:
                   price = close.iloc[-1]
            except Exception:
                   price = close.iloc[-1]
            macd_signal = detect_macd_signal(close)
            dist_ma20 = distance_to_ma(close, 20)
            dist_ma50 = distance_to_ma(close, 50)
            nearest_ma_dist = np.nanmin([dist_ma20, dist_ma50])
            ma_pos = detect_ma_position(close)

            trend = detect_trend(close)
            zone = detect_zone(df)
            candle, bias = detect_candle(df)
            rsi = RSIIndicator(close, 14).rsi().iloc[-1]
            signal = build_signal(zone, bias, trend)
            buy_filter = (
                macd_signal == "Golden Cross" and
                rsi <= 50 and
                nearest_ma_dist <= 2
            )

            sell_filter = (
                macd_signal == "Death Cross" and
                rsi >= 70 and
                nearest_ma_dist <= 2
            )

            fib = calculate_fibonacci(df)
            support, resistance = calculate_support_resistance(df)

            if fib and support and resistance:
               tp = min(fib["0.236"], resistance)
               sl = max(fib["0.786"], support * 0.98)
            else:
               tp = price * (1 + TP_PCT / 100)
               sl = price * (1 - SL_PCT / 100)

            confidence = 0
            confidence += 1 if trend == "Bullish" else 0
            confidence += 1 if zone == "BUY ZONE" else 0
            confidence += 1 if bias == "Bullish" else 0
            confidence += 1 if rsi < 40 else 0
            
            fake_rebound = detect_fake_rebound(close, df)
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
                "BUY_Filter": buy_filter,
                "SELL_Filter": sell_filter,
                "MA_Dist(%)": round(nearest_ma_dist, 2),
                "Fake_Rebound": fake_rebound,
                "_df": df.copy()
            })

        except Exception as e:
            st.write(f"Error {t}: {e}")

df = pd.DataFrame(rows)
# =====================
# SIDEBAR – HARGA WAJAR SIMPLE
# =====================
if show_fair_value and fair_search:
    ticker_search = fair_search + ".JK"
    row = df[df["Kode"] == ticker_search]

    st.sidebar.markdown("---")

    if not row.empty:
        row = row.iloc[0]

        # ===== HARGA WAJAR =====
        fv = calculate_fair_value_simple(
            ticker_search,
            row["Harga"]
        )
        
        # ===== SUPPORT & RESISTANCE =====
        support, resistance = calculate_support_resistance(row["_df"])

        st.sidebar.markdown(f"### 📊 {fair_search}")
        st.sidebar.metric("Harga Saat Ini", row["Harga"])

        if fv:
            st.sidebar.metric(
                "Harga Wajar",
                fv["Fair_Price"],
                delta=f"{fv['Margin']}%"
            )

            if fv["Margin"] > 20:
                st.sidebar.success("🟢 Undervalued")
            elif fv["Margin"] < -10:
                st.sidebar.error("🔴 Overvalued")
            else:
                st.sidebar.warning("🟡 Fair Value")
        else:
            st.sidebar.warning("EPS tidak tersedia")

        # ===== CHART FIBO + SNR =====
        st.sidebar.markdown("### 📈 Chart Fibo & SNR")

        fig = plot_price_fibo_snr(row["_df"], ticker_search)
        st.sidebar.pyplot(fig)
        
        # ===== TAMPIL SUPPORT & RESISTANCE =====
        st.sidebar.markdown("### 🧱 Support & Resistance")

        if support and resistance:
            st.sidebar.metric("Support", support)
            st.sidebar.metric("Resistance", resistance)

            price = row["Harga"]

            if price <= support * 1.03:
                st.sidebar.success("📉 Harga dekat SUPPORT")
            elif price >= resistance * 0.97:
                st.sidebar.error("📈 Harga dekat RESISTANCE")
            else:
                st.sidebar.info("↔ Harga di tengah range")
        else:
            st.sidebar.warning("Data Support / Resistance tidak tersedia")

    else:
        st.sidebar.info("Saham belum masuk hasil scanner")

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
h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14 = st.columns(
    [1.2, 1, 1, 1, 1, 0.8, 1.2, 1.2, 1.2, 1, 1, 1, 1, 0.6]
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
h14.markdown("**CONF**")

st.divider()

ROW_HEIGHT = 70

# =====================
# Kolom 1 - 9
# =====================
filtered_df = df.copy()

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

for _, row in filtered_df.iterrows():
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14 = st.columns(
        [1.2, 1, 1, 1, 1, 0.8, 1.2, 1.2, 1.2, 1, 1, 1, 1, 0.6]
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
            
    with c14:
         score = row["Confidence"]

         if score >= 4:
            color = "#00C176"
         elif score == 3:
            color = "#FFA500"
         else:
            color = "#999"

         st.markdown(
            f"""
            <div style="
                 text-align: center;
                 font-weight: 800;
                 font-size: 15px;
                 color: {color};
            ">
                 {score}
            </div>
            """,
            unsafe_allow_html=True
         )
     
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









