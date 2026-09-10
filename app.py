import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Groww API
try:
    from growwapi import GrowwAPI
    GROWW_AVAILABLE = True
except Exception:
    GROWW_AVAILABLE = False


st.set_page_config(
    page_title="ALS Live Algo",
    page_icon="📈",
    layout="wide"
)

st.title("📈 ALS Live Algo – Groww Live Data")
st.caption("Groww live market data • Paper Trading only")


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Settings")

symbol_name = st.sidebar.selectbox(
    "Instrument",
    ["NIFTY", "SENSEX"]
)

refresh = st.sidebar.slider(
    "Refresh seconds",
    min_value=5,
    max_value=60,
    value=10
)

ema_fast = st.sidebar.number_input(
    "Fast EMA",
    min_value=2,
    max_value=100,
    value=9
)

ema_slow = st.sidebar.number_input(
    "Slow EMA",
    min_value=3,
    max_value=200,
    value=21
)

st_period = st.sidebar.number_input(
    "Supertrend Period",
    min_value=2,
    max_value=100,
    value=10
)

st_mult = st.sidebar.number_input(
    "Supertrend Multiplier",
    min_value=0.5,
    max_value=10.0,
    value=3.0
)


# ---------------------------------------------------------
# GROW API CONNECTION
# ---------------------------------------------------------
if "GROWW_ACCESS_TOKEN" not in st.secrets:

    st.error("❌ Groww Access Token not found.")

    st.info(
        "Streamlit → Settings → Secrets માં "
        "GROWW_ACCESS_TOKEN save કરો."
    )

    st.stop()


if not GROWW_AVAILABLE:
    st.error("❌ growwapi package મળ્યું નથી.")
    st.info("requirements.txt માં growwapi add કરવું પડશે.")
    st.stop()


try:
    groww = GrowwAPI(
        st.secrets["GROWW_ACCESS_TOKEN"]
    )
except Exception as e:

    st.error("❌ Groww API initialize થઈ શક્યું નથી.")
    st.code(str(e))
    st.stop()


# ---------------------------------------------------------
# SYMBOL
# ---------------------------------------------------------
if symbol_name == "NIFTY":
    exchange_symbol = "NSE_NIFTY"
else:
    exchange_symbol = "BSE_SENSEX"


# ---------------------------------------------------------
# GET LIVE LTP
# ---------------------------------------------------------
def get_live_price():

    response = groww.get_ltp(
        segment=groww.SEGMENT_CASH,
        exchange_trading_symbols=exchange_symbol
    )

    if isinstance(response, dict):

        if exchange_symbol in response:
            return float(response[exchange_symbol])

        if "ltp" in response:
            return float(response["ltp"])

        if "payload" in response:
            payload = response["payload"]

            if exchange_symbol in payload:
                return float(payload[exchange_symbol])

    raise Exception(
        f"Unexpected Groww response: {response}"
    )


# ---------------------------------------------------------
# LIVE PRICE
# ---------------------------------------------------------
try:

    live_price = get_live_price()

    st.success("🟢 Groww API Connected")

except Exception as e:

    st.error("🔴 Groww Live Data Error")

    st.code(str(e))

    st.warning(
        "જો Free Trial માં Live Data access ન હોય "
        "તો Groww API plan upgrade કરવો પડી શકે."
    )

    st.stop()


# ---------------------------------------------------------
# STORE LIVE PRICES
# ---------------------------------------------------------
if "price_history" not in st.session_state:
    st.session_state.price_history = []


now = datetime.now()

st.session_state.price_history.append({
    "time": now,
    "price": live_price
})

# Keep last 500 observations
st.session_state.price_history = (
    st.session_state.price_history[-500:]
)


df = pd.DataFrame(
    st.session_state.price_history
)


# ---------------------------------------------------------
# BASIC LIVE DATA
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        f"{symbol_name} Live Price",
        f"₹{live_price:,.2f}"
    )

with col2:
    st.metric(
        "Data Points",
        len(df)
    )

with col3:
    st.metric(
        "Last Update",
        now.strftime("%H:%M:%S")
    )


# ---------------------------------------------------------
# PRICE CHART
# ---------------------------------------------------------
st.subheader("📊 Live Price")

if len(df) > 1:

    chart_df = df.set_index("time")[["price"]]

    st.line_chart(chart_df)

else:

    st.info(
        "Live price collection શરૂ થઈ રહી છે..."
    )


# ---------------------------------------------------------
# EMA
# ---------------------------------------------------------
if len(df) >= ema_slow:

    df["EMA_FAST"] = (
        df["price"]
        .ewm(span=ema_fast, adjust=False)
        .mean()
    )

    df["EMA_SLOW"] = (
        df["price"]
        .ewm(span=ema_slow, adjust=False)
        .mean()
    )

    ema_fast_value = df["EMA_FAST"].iloc[-1]
    ema_slow_value = df["EMA_SLOW"].iloc[-1]

    if ema_fast_value > ema_slow_value:
        ma_signal = "BUY"
    elif ema_fast_value < ema_slow_value:
        ma_signal = "SELL"
    else:
        ma_signal = "WAIT"

else:

    ma_signal = "WAIT"


# ---------------------------------------------------------
# SIMPLE LIVE MOMENTUM
# ---------------------------------------------------------
if len(df) >= 2:

    previous_price = df["price"].iloc[-2]

    if live_price > previous_price:
        momentum_signal = "BUY"

    elif live_price < previous_price:
        momentum_signal = "SELL"

    else:
        momentum_signal = "WAIT"

else:

    momentum_signal = "WAIT"


# ---------------------------------------------------------
# COMBINED SIGNAL
# ---------------------------------------------------------
if ma_signal == "BUY" and momentum_signal == "BUY":

    final_signal = "BUY"

elif ma_signal == "SELL" and momentum_signal == "SELL":

    final_signal = "SELL"

else:

    final_signal = "WAIT"


# ---------------------------------------------------------
# SIGNAL DISPLAY
# ---------------------------------------------------------
st.subheader("🎯 ALS Signal")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Moving Average",
        ma_signal
    )

with c2:
    st.metric(
        "Momentum",
        momentum_signal
    )

with c3:
    st.metric(
        "FINAL SIGNAL",
        final_signal
    )


# ---------------------------------------------------------
# PAPER TRADE PLAN
# ---------------------------------------------------------
st.subheader("📋 Paper Trade Plan")

if final_signal == "BUY":

    entry = live_price
    stop_loss = live_price * 0.995
    target = live_price * 1.01

    st.success("🟢 BUY")

    p1, p2, p3 = st.columns(3)

    p1.metric("Entry", f"₹{entry:,.2f}")
    p2.metric("Stop Loss", f"₹{stop_loss:,.2f}")
    p3.metric("Target", f"₹{target:,.2f}")


elif final_signal == "SELL":

    entry = live_price
    stop_loss = live_price * 1.005
    target = live_price * 0.99

    st.error("🔴 SELL")

    p1, p2, p3 = st.columns(3)

    p1.metric("Entry", f"₹{entry:,.2f}")
    p2.metric("Stop Loss", f"₹{stop_loss:,.2f}")
    p3.metric("Target", f"₹{target:,.2f}")


else:

    st.warning("🟡 WAIT")

    st.write(
        "બંને મુખ્ય signals agree થાય ત્યાં સુધી trade નહીં."
    )


# ---------------------------------------------------------
# IMPORTANT NOTE
# ---------------------------------------------------------
st.divider()

st.info(
    "⚠️ આ version Paper Trading માટે છે. "
    "કોઈ real order automatically place કરતું નથી."
)

st.caption(
    "Groww Access Token daily 6:00 AM પર expire થાય છે."
)


# ---------------------------------------------------------
# AUTO REFRESH
# ---------------------------------------------------------
time.sleep(refresh)
st.rerun()
