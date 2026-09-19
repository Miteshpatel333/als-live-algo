
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ALS AI Algo V5", page_icon="🤖", layout="wide")

SYMBOLS = {"NIFTY":"^NSEI", "BANK NIFTY":"^NSEBANK", "SENSEX":"^BSESN"}

def allowed_periods(interval):
    if interval in ("5m","15m"):
        return ["5d"] if interval=="5m" else ["5d","1mo"]
    if interval=="1h":
        return ["1mo","3mo","6mo","1y"]
    return ["1mo","3mo","6mo","1y","2y","5y"]

def fetch_data(ticker, period, interval):
    import yfinance as yf
    raw = yf.download(ticker, period=period, interval=interval,
                      auto_adjust=False, progress=False, threads=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    raw.columns = [str(c).strip().lower().replace(" ","_") for c in raw.columns]
    dt = next((c for c in ("datetime","date","timestamp") if c in raw.columns), None)
    if not dt:
        return pd.DataFrame()
    raw["datetime"] = pd.to_datetime(raw[dt], errors="coerce")
    for c in ("open","high","low","close"):
        if c not in raw.columns:
            return pd.DataFrame()
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    if "volume" not in raw.columns:
        raw["volume"] = 0
    raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce").fillna(0)
    return raw[["datetime","open","high","low","close","volume"]].dropna().sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

def add_indicators(d):
    d = d.copy()
    d["ema20"] = d.close.ewm(span=20, adjust=False).mean()
    d["ema50"] = d.close.ewm(span=50, adjust=False).mean()
    d["ema200"] = d.close.ewm(span=200, adjust=False).mean()

    delta = d.close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - 100/(1+rs)

    tr = pd.concat([(d.high-d.low),
                    (d.high-d.close.shift()).abs(),
                    (d.low-d.close.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14, min_periods=14).mean()
    d["atr_pct"] = d.atr / d.close * 100

    if d.volume.sum() > 0:
        cv = d.volume.cumsum().replace(0,np.nan)
        d["vwap"] = (d.close*d.volume).cumsum()/cv
    else:
        d["vwap"] = d.close.expanding().mean()

    hl2=(d.high+d.low)/2
    upper=hl2+3*d.atr
    lower=hl2-3*d.atr
    trend=np.ones(len(d))
    for i in range(1,len(d)):
        if pd.isna(d.atr.iloc[i]): trend[i]=trend[i-1]
        elif d.close.iloc[i] > upper.iloc[i-1]: trend[i]=1
        elif d.close.iloc[i] < lower.iloc[i-1]: trend[i]=-1
        else: trend[i]=trend[i-1]
    d["supertrend_dir"]=trend
    return d

def signal(r, min_score=70):
    buy = 0
    sell = 0
    buy += 20 if r.ema20 > r.ema50 else 0
    buy += 15 if r.ema20 > r.ema200 else 0
    buy += 15 if r.rsi >= 55 else 0
    buy += 15 if r.close > r.vwap else 0
    buy += 20 if r.supertrend_dir > 0 else 0
    buy += 15 if r.close > r.ema20 else 0

    sell += 20 if r.ema20 < r.ema50 else 0
    sell += 15 if r.ema20 < r.ema200 else 0
    sell += 15 if r.rsi <= 45 else 0
    sell += 15 if r.close < r.vwap else 0
    sell += 20 if r.supertrend_dir < 0 else 0
    sell += 15 if r.close < r.ema20 else 0

    # Avoid weak/sideways conditions.
    if pd.isna(r.atr_pct) or r.atr_pct < 0.05:
        return "NO TRADE", max(buy,sell)
    if buy >= min_score and buy > sell:
        return "BUY", buy
    if sell >= min_score and sell > buy:
        return "SELL", sell
    return "NO TRADE", max(buy,sell)

def run_backtest(d, capital, risk_pct, sl_atr, rr, max_trades_day, min_score, trail_atr):
    cash=float(capital)
    pos=None
    trades=[]
    equity=[]
    day_count={}
    peak=cash
    max_dd=0.0

    for _,r in d.iterrows():
        if pd.isna(r.atr) or pd.isna(r.rsi) or pd.isna(r.ema200):
            equity.append({"datetime":r.datetime,"equity":cash})
            continue

        day=str(r.datetime)[:10]
        sig,conf=signal(r,min_score)

        if pos:
            # Move stop toward profit using trailing ATR.
            if pos["side"]=="BUY":
                pos["sl"] = max(pos["sl"], float(r.close - r.atr*trail_atr))
                if r.close >= pos["entry"] + pos["risk_dist"]:
                    pos["sl"] = max(pos["sl"], pos["entry"])
                exit_price=None; reason=None
                if r.low <= pos["sl"]:
                    exit_price,reason=pos["sl"],"SL/TRAIL"
                elif r.high >= pos["target"]:
                    exit_price,reason=pos["target"],"TARGET"
            else:
                pos["sl"] = min(pos["sl"], float(r.close + r.atr*trail_atr))
                if r.close <= pos["entry"] - pos["risk_dist"]:
                    pos["sl"] = min(pos["sl"], pos["entry"])
                exit_price=None; reason=None
                if r.high >= pos["sl"]:
                    exit_price,reason=pos["sl"],"SL/TRAIL"
                elif r.low <= pos["target"]:
                    exit_price,reason=pos["target"],"TARGET"

            if exit_price is not None:
                pnl=(exit_price-pos["entry"])*pos["qty"]*(1 if pos["side"]=="BUY" else -1)
                cash += pnl
                trades.append({**pos,"exit":exit_price,"pnl":pnl,"reason":reason,"exit_time":r.datetime})
                pos=None

        if pos is None and sig in ("BUY","SELL") and day_count.get(day,0)<max_trades_day:
            dist=max(float(r.atr)*sl_atr,0.01)
            qty=max(1,int((cash*risk_pct/100)/dist))
            entry=float(r.close)
            sl=entry-dist if sig=="BUY" else entry+dist
            target=entry+dist*rr if sig=="BUY" else entry-dist*rr
            pos={"side":sig,"entry":entry,"sl":sl,"target":target,"qty":qty,
                 "entry_time":r.datetime,"confidence":conf,"risk_dist":dist}
            day_count[day]=day_count.get(day,0)+1

        mark = cash
        if pos:
            mark += (float(r.close)-pos["entry"])*pos["qty"]*(1 if pos["side"]=="BUY" else -1)
        peak=max(peak,mark)
        max_dd=max(max_dd,(peak-mark)/peak*100 if peak else 0)
        equity.append({"datetime":r.datetime,"equity":mark})

    return pd.DataFrame(trades), pd.DataFrame(equity), cash, max_dd

st.title("🤖 ALS AI Algo Trading V5")
st.caption("AI-assisted multi-indicator strategy • Backtest + Paper Trading • Groww API not required")

with st.sidebar:
    st.header("Strategy Controls")
    interval=st.selectbox("Candle interval",["5m","15m","1h","1d"])
    period=st.selectbox("Historical period",allowed_periods(interval))
    capital=st.number_input("Starting capital",100000,10000000,100000,10000)
    risk=st.slider("Risk / trade (%)",0.25,2.0,1.0,0.25)
    min_score=st.slider("Minimum signal score",60,90,70,5)
    sl_atr=st.slider("Initial SL (ATR)",0.75,3.0,1.5,0.25)
    rr=st.slider("Reward : Risk",1.0,4.0,2.0,0.5)
    trail_atr=st.slider("Trailing SL (ATR)",0.5,3.0,1.5,0.25)
    max_trades=st.slider("Max trades / day",1,6,3)

st.info("V5 tests all three indices with the same rules and shows risk-adjusted performance. This is research/paper trading, not a profit guarantee.")

if st.button("🚀 Run V5 Multi-Index Backtest", type="primary"):
    results=[]
    logs={}
    progress=st.progress(0)
    for i,(name,ticker) in enumerate(SYMBOLS.items(),1):
        d=fetch_data(ticker,period,interval)
        if d.empty:
            results.append({"Symbol":name,"Status":"No data","Trades":0,"P&L":0,"Win Rate %":0,"Max DD %":0})
        else:
            d=add_indicators(d)
            trades,equity,final_cash,max_dd=run_backtest(d,capital,risk,sl_atr,rr,max_trades,min_score,trail_atr)
            pnl=final_cash-capital
            wins=int((trades.pnl>0).sum()) if len(trades) else 0
            wr=wins/len(trades)*100 if len(trades) else 0
            results.append({"Symbol":name,"Status":"OK","Candles":len(d),"Trades":len(trades),
                            "P&L":round(pnl,2),"Win Rate %":round(wr,1),"Max DD %":round(max_dd,2)})
            logs[name]=(d,trades,equity,final_cash)
        progress.progress(i/3)

    res=pd.DataFrame(results)
    st.subheader("📊 Multi-Index Summary")
    st.dataframe(res,use_container_width=True)

    for name in SYMBOLS:
        if name not in logs:
            continue
        d,trades,equity,final_cash=logs[name]
        st.subheader(f"📈 {name}")
        last=d.dropna(subset=["ema200","rsi","atr","vwap"]).iloc[-1]
        sig,conf=signal(last,min_score)
        a,b,c,dcol=st.columns(4)
        a.metric("Signal",sig); b.metric("Confidence",f"{conf}%")
        c.metric("Last Close",f"{last.close:.2f}")
        dcol.metric("Final Capital",f"₹{final_cash:,.2f}")

        if len(equity):
            chart=equity.set_index("datetime")[["equity"]]
            st.line_chart(chart)

        st.dataframe(trades,use_container_width=True)
        if len(trades):
            st.download_button(f"Download {name} trades",trades.to_csv(index=False),
                               f"{name.lower().replace(' ','_')}_trades.csv","text/csv",
                               key=f"dl_{name}")

st.divider()
st.caption("Research/paper-trading only. Backtest results can differ materially from live execution and do not guarantee future profits.")
