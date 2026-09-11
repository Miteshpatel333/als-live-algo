
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from growwapi import GrowwAPI
    GROWW_AVAILABLE = True
except Exception:
    GROWW_AVAILABLE = False

st.set_page_config(page_title="ALS Strategy V2", page_icon="📈", layout="wide")

st.title("📈 ALS Live Algo – Strategy V2")
st.caption("Supertrend-led strategy • Groww data • Paper Trading only")

# ---------------- SIDEBAR ----------------
st.sidebar.header("ALS Strategy V2")

instrument = st.sidebar.selectbox("Instrument", ["NIFTY", "SENSEX"])
interval_label = st.sidebar.selectbox("Timeframe", ["5 Minute", "10 Minute"])
interval_map = {"5 Minute": 5, "10 Minute": 10}
interval = interval_map[interval_label]

st.sidebar.subheader("Core parameters")
st_period = st.sidebar.number_input("Supertrend Period", 2, 50, 10)
st_mult = st.sidebar.number_input("Supertrend Multiplier", 0.5, 8.0, 3.0, step=0.5)
ema_fast = st.sidebar.number_input("Fast EMA", 2, 100, 9)
ema_slow = st.sidebar.number_input("Slow EMA", 3, 200, 21)
rsi_period = st.sidebar.number_input("RSI Period", 2, 50, 14)
atr_period = st.sidebar.number_input("ATR Period", 2, 50, 14)
sl_atr = st.sidebar.number_input("SL ATR Multiplier", 0.5, 5.0, 1.2, step=0.1)
rr = st.sidebar.number_input("Risk : Reward", 1.0, 5.0, 2.0, step=0.5)

st.sidebar.info(
    "ALS V2: Supertrend is the primary trend engine. "
    "EMA + Parabolic SAR + RSI + Momentum are confirmations. "
    "Only paper trades are generated."
)

# ---------------- GROW ----------------
if "GROWW_ACCESS_TOKEN" not in st.secrets:
    st.error("❌ GROWW_ACCESS_TOKEN not found in Streamlit Secrets.")
    st.stop()

if not GROWW_AVAILABLE:
    st.error("❌ growwapi package not installed.")
    st.stop()

try:
    groww = GrowwAPI(st.secrets["GROWW_ACCESS_TOKEN"])
except Exception as e:
    st.error("❌ Groww API initialization failed.")
    st.code(str(e))
    st.stop()

groww_symbol = "NSE-NIFTY" if instrument == "NIFTY" else "BSE-SENSEX"
exchange = groww.EXCHANGE_NSE if instrument == "NIFTY" else groww.EXCHANGE_BSE

# ---------------- INDICATORS ----------------
def atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def supertrend(df, period=10, multiplier=3.0):
    out = df.copy()
    out["atr"] = atr(out, period)
    hl2 = (out["high"] + out["low"]) / 2
    out["basic_upper"] = hl2 + multiplier * out["atr"]
    out["basic_lower"] = hl2 - multiplier * out["atr"]

    fu = out["basic_upper"].copy()
    fl = out["basic_lower"].copy()
    trend = pd.Series(index=out.index, dtype="int64")
    st_line = pd.Series(index=out.index, dtype="float64")
    trend.iloc[0] = 1
    st_line.iloc[0] = fl.iloc[0]

    for i in range(1, len(out)):
        fu.iloc[i] = out["basic_upper"].iloc[i] if (
            out["basic_upper"].iloc[i] < fu.iloc[i-1] or out["close"].iloc[i-1] > fu.iloc[i-1]
        ) else fu.iloc[i-1]
        fl.iloc[i] = out["basic_lower"].iloc[i] if (
            out["basic_lower"].iloc[i] > fl.iloc[i-1] or out["close"].iloc[i-1] < fl.iloc[i-1]
        ) else fl.iloc[i-1]

        if trend.iloc[i-1] == -1 and out["close"].iloc[i] > fu.iloc[i]:
            trend.iloc[i] = 1
        elif trend.iloc[i-1] == 1 and out["close"].iloc[i] < fl.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]

        st_line.iloc[i] = fl.iloc[i] if trend.iloc[i] == 1 else fu.iloc[i]

    out["supertrend"] = st_line
    out["st_dir"] = trend
    return out

def parabolic_sar(high, low, step=0.02, max_af=0.20):
    psar = low.copy().astype(float)
    bull = True
    af = step
    ep = high.iloc[0]
    psar.iloc[0] = low.iloc[0]

    for i in range(1, len(high)):
        prev = psar.iloc[i-1]
        if bull:
            psar.iloc[i] = prev + af * (ep - prev)
            if i >= 2:
                psar.iloc[i] = min(psar.iloc[i], low.iloc[i-1], low.iloc[i-2])
            else:
                psar.iloc[i] = min(psar.iloc[i], low.iloc[i-1])
            if low.iloc[i] < psar.iloc[i]:
                bull = False
                psar.iloc[i] = ep
                af = step
                ep = low.iloc[i]
            elif high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(max_af, af + step)
        else:
            psar.iloc[i] = prev + af * (ep - prev)
            if i >= 2:
                psar.iloc[i] = max(psar.iloc[i], high.iloc[i-1], high.iloc[i-2])
            else:
                psar.iloc[i] = max(psar.iloc[i], high.iloc[i-1])
            if high.iloc[i] > psar.iloc[i]:
                bull = True
                psar.iloc[i] = ep
                af = step
                ep = high.iloc[i]
            elif low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(max_af, af + step)
    return psar

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ---------------- DATA ----------------
@st.cache_data(ttl=45, show_spinner=False)
def load_candles(token, exch, segment, symbol, minutes):
    api = GrowwAPI(token)
    end = datetime.now()
    start = end - timedelta(days=7)
    # Groww historical candles support 1/5/10 minute intervals.
    data = api.get_historical_candles(
        exchange=exch,
        segment=segment,
        groww_symbol=symbol,
        start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
        end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
        candle_interval=(
            api.CANDLE_INTERVAL_MIN_5 if minutes == 5
            else api.CANDLE_INTERVAL_MIN_10
        )
    )
    if isinstance(data, dict) and "payload" in data:
        data = data["payload"]
    if isinstance(data, dict) and "candles" in data:
        data = data["candles"]
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

try:
    df = load_candles(
        st.secrets["GROWW_ACCESS_TOKEN"],
        exchange,
        groww.SEGMENT_CASH,
        groww_symbol,
        interval
    )
except Exception as e:
    st.error("🔴 Groww Historical Data Error")
    st.code(str(e))
    st.info("Groww historical/live market-data access must be enabled on the API plan.")
    st.stop()

if len(df) < max(60, int(ema_slow) + 20):
    st.warning("Not enough candles yet for a reliable ALS signal.")
    st.dataframe(df.tail(20), use_container_width=True)
    st.stop()

# ---------------- CALCULATE ----------------
df["EMA_FAST"] = df["close"].ewm(span=int(ema_fast), adjust=False).mean()
df["EMA_SLOW"] = df["close"].ewm(span=int(ema_slow), adjust=False).mean()
df["RSI"] = rsi(df["close"], int(rsi_period))
df["ATR"] = atr(df, int(atr_period))
df["PSAR"] = parabolic_sar(df["high"], df["low"])
df = supertrend(df, int(st_period), float(st_mult))

# Momentum = current close above/below previous close and short ROC
df["ROC"] = df["close"].pct_change(3) * 100

last = df.iloc[-1]
prev = df.iloc[-2]

st_dir = "BUY" if last["st_dir"] == 1 else "SELL"
ema_signal = "BUY" if last["EMA_FAST"] > last["EMA_SLOW"] else "SELL"
psar_signal = "BUY" if last["close"] > last["PSAR"] else "SELL"
rsi_signal = "BUY" if last["RSI"] >= 55 else ("SELL" if last["RSI"] <= 45 else "WAIT")
mom_signal = "BUY" if last["ROC"] > 0 else ("SELL" if last["ROC"] < 0 else "WAIT")

# ALS scoring: Supertrend carries the most weight because it was the strongest
# strategy in the supplied report. Other indicators act as confirmation.
buy_score = (2 if st_dir == "BUY" else 0) + (1 if ema_signal == "BUY" else 0) + \
            (1 if psar_signal == "BUY" else 0) + (1 if rsi_signal == "BUY" else 0) + \
            (1 if mom_signal == "BUY" else 0)
sell_score = (2 if st_dir == "SELL" else 0) + (1 if ema_signal == "SELL" else 0) + \
             (1 if psar_signal == "SELL" else 0) + (1 if rsi_signal == "SELL" else 0) + \
             (1 if mom_signal == "SELL" else 0)

if buy_score >= 5 and buy_score > sell_score:
    final_signal = "BUY"
elif sell_score >= 5 and sell_score > buy_score:
    final_signal = "SELL"
else:
    final_signal = "WAIT"

# ---------------- DASHBOARD ----------------
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric(f"{instrument} (Close)", f"₹{last['close']:,.2f}")
c2.metric("Supertrend", st_dir)
c3.metric("EMA", ema_signal)
c4.metric("PSAR", psar_signal)
c5.metric("ALS Signal", final_signal)

st.subheader("📊 ALS Confirmation Matrix")
matrix = pd.DataFrame({
    "Indicator": ["Supertrend (Primary)", "EMA", "Parabolic SAR", "RSI", "Momentum"],
    "Signal": [st_dir, ema_signal, psar_signal, rsi_signal, mom_signal],
    "Score": [2 if st_dir in ["BUY","SELL"] else 0,
              1 if ema_signal in ["BUY","SELL"] else 0,
              1 if psar_signal in ["BUY","SELL"] else 0,
              1 if rsi_signal in ["BUY","SELL"] else 0,
              1 if mom_signal in ["BUY","SELL"] else 0]
})
st.dataframe(matrix, hide_index=True, use_container_width=True)

# Trade plan uses ATR-based stop and fixed R:R, not a fixed percentage.
entry = float(last["close"])
risk = max(float(last["ATR"]) * float(sl_atr), 1.0)

if final_signal == "BUY":
    sl = entry - risk
    target = entry + risk * float(rr)
elif final_signal == "SELL":
    sl = entry + risk
    target = entry - risk * float(rr)
else:
    sl = target = np.nan

st.subheader("📋 Paper Trade Plan")
p1,p2,p3,p4 = st.columns(4)
p1.metric("Signal", final_signal)
p2.metric("Entry", f"₹{entry:,.2f}")
p3.metric("Stop Loss", "-" if np.isnan(sl) else f"₹{sl:,.2f}")
p4.metric("Target", "-" if np.isnan(target) else f"₹{target:,.2f}")

st.subheader(f"📈 {instrument} – {interval_label} Candles")
chart = df.set_index("timestamp")[["close","EMA_FAST","EMA_SLOW","supertrend","PSAR"]].tail(120)
st.line_chart(chart)

st.subheader("Latest Candle Data")
st.dataframe(
    df[["timestamp","open","high","low","close","EMA_FAST","EMA_SLOW","RSI","PSAR","supertrend","ROC"]].tail(20),
    use_container_width=True
)

st.divider()
st.info(
    "⚠️ Paper Trading only. This strategy is a new rule set inspired by the supplied July 2026 trade report; "
    "the report does not disclose the original indicator parameters or proprietary rules. "
    "Past report performance is not a guarantee of future results."
)
