
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

st.set_page_config(page_title="ALS Live Algo", page_icon="📈", layout="wide")

st.title("📈 ALS Live Algo — Paper Trading")
st.caption("Live-signal dashboard • Paper trading • Replace demo feed with an authorized broker/data API")

# Sidebar
st.sidebar.header("Settings")
symbol = st.sidebar.selectbox("Instrument", ["NIFTY", "SENSEX"])
refresh = st.sidebar.slider("Refresh seconds", 1, 30, 5)
ema_fast = st.sidebar.number_input("Fast EMA", 5, 50, 9)
ema_slow = st.sidebar.number_input("Slow EMA", 10, 100, 21)
st_period = st.sidebar.number_input("Supertrend period", 5, 50, 10)
st_mult = st.sidebar.number_input("Supertrend multiplier", 1.0, 6.0, 3.0, 0.5)

# Demo feed: deterministic-ish synthetic candles. Replace get_demo_data() with broker API.
@st.cache_data(ttl=1)
def get_demo_data(symbol):
    rng = np.random.default_rng(int(time.time()) // 5)
    n = 180
    base = 25500 if symbol == "NIFTY" else 83000
    rets = rng.normal(0.0001, 0.0025, n)
    close = base * np.cumprod(1 + rets)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, .0012, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, .0012, n))
    volume = rng.integers(100_000, 900_000, n)
    return pd.DataFrame({"open":open_, "high":high, "low":low, "close":close, "volume":volume})

def supertrend(df, period=10, mult=3):
    tr = pd.concat([
        df.high-df.low,
        (df.high-df.close.shift()).abs(),
        (df.low-df.close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    hl2 = (df.high+df.low)/2
    upper = hl2 + mult*atr
    lower = hl2 - mult*atr
    direction = pd.Series(1, index=df.index)
    line = lower.copy()
    for i in range(1, len(df)):
        if df.close.iloc[i] > upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif df.close.iloc[i] < lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
        if direction.iloc[i] == 1:
            line.iloc[i] = max(lower.iloc[i], line.iloc[i-1])
        else:
            line.iloc[i] = min(upper.iloc[i], line.iloc[i-1])
    return line, direction

def psar(df, step=.02, max_step=.2):
    h,l = df.high.to_numpy(), df.low.to_numpy()
    s = np.zeros(len(df)); s[0]=l[0]
    bull=True; af=step; ep=h[0]
    for i in range(1,len(df)):
        s[i]=s[i-1]+af*(ep-s[i-1])
        if bull:
            s[i]=min(s[i],l[i-1],l[i-2] if i>1 else l[i-1])
            if l[i]<s[i]:
                bull=False; s[i]=ep; ep=l[i]; af=step
            elif h[i]>ep:
                ep=h[i]; af=min(max_step,af+step)
        else:
            s[i]=max(s[i],h[i-1],h[i-2] if i>1 else h[i-1])
            if h[i]>s[i]:
                bull=True; s[i]=ep; ep=h[i]; af=step
            elif l[i]<ep:
                ep=l[i]; af=min(max_step,af+step)
    return pd.Series(s,index=df.index)

df=get_demo_data(symbol)
df["ema_fast"]=df.close.ewm(span=ema_fast,adjust=False).mean()
df["ema_slow"]=df.close.ewm(span=ema_slow,adjust=False).mean()
df["ma"]=np.where(df.ema_fast>df.ema_slow,1,-1)
df["st"],df["st_dir"]=supertrend(df,st_period,st_mult)
df["psar"]=psar(df)
df["psar_dir"]=np.where(df.close>df.psar,1,-1)
df["score"]=df.ma+df.st_dir+df.psar_dir
df["signal"]=np.select([df.score>=2,df.score<=-2],["BUY","SELL"],default="WAIT")

last=df.iloc[-1]
price=float(last.close)
signal=last.signal

# Header metrics
a,b,c,d,e=st.columns(5)
a.metric(symbol, f"₹{price:,.2f}")
b.metric("Signal", signal)
c.metric("Supertrend", "BUY" if last.st_dir==1 else "SELL")
d.metric("Moving Avg", "BUY" if last.ma==1 else "SELL")
e.metric("Parabolic", "BUY" if last.psar_dir==1 else "SELL")

st.divider()

left,right=st.columns([2,1])
with left:
    st.subheader("📊 Live Signal Chart")
    chart=df.tail(100)[["close","ema_fast","ema_slow","st","psar"]]
    st.line_chart(chart)
    st.subheader("Recent candles")
    st.dataframe(df.tail(15)[["open","high","low","close","volume","score","signal"]],use_container_width=True)

with right:
    st.subheader("🎯 Trade Plan")
    if signal=="BUY":
        option="ATM / near-ATM CE"
        sl=price*0.99; target=price*1.02
    elif signal=="SELL":
        option="ATM / near-ATM PE"
        sl=price*1.01; target=price*0.98
    else:
        option="WAIT"
        sl=target=price
    st.metric("Direction", signal)
    st.write("**Option bias:**", option)
    st.write(f"**Underlying entry:** ₹{price:,.2f}")
    st.write(f"**Underlying SL:** ₹{sl:,.2f}")
    st.write(f"**Underlying target:** ₹{target:,.2f}")
    st.info("This is a paper-trading signal, not investment advice.")

st.subheader("🧪 Paper Trade Log")
if "trades" not in st.session_state:
    st.session_state.trades=[]
if st.button("Record current signal"):
    st.session_state.trades.append({
        "Time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol":symbol,"Price":round(price,2),"Signal":signal,
        "Strategy score":int(last.score)
    })
if st.session_state.trades:
    st.dataframe(pd.DataFrame(st.session_state.trades),use_container_width=True)
else:
    st.caption("No paper trades recorded yet.")

st.divider()
st.subheader("🔌 Next step: real-time data")
st.write("""
To turn this into a genuine live-market application, replace the demo data function with an
authorized broker/market-data API. Then add option-chain/OI, alerts, authentication, backtesting,
and (only after testing) optional broker order execution.
""")

st.caption("Note: JSK and Optima from the reference screenshot are not reproduced because their exact proprietary rules are unknown.")
