
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ALS AI Algo", page_icon="📈", layout="wide")

SYMBOLS = {"NIFTY":"^NSEI", "BANK NIFTY":"^NSEBANK", "SENSEX":"^BSESN"}

def periods(interval):
    if interval in ("5m","15m"):
        return ["5d","1mo"]
    if interval == "1h":
        return ["1mo","3mo","6mo","1y"]
    return ["1mo","3mo","6mo","1y","2y","5y"]

def clean_yahoo(raw):
    if raw is None or raw.empty:
        return pd.DataFrame()

    # Normalize Yahoo's single- and multi-level columns.
    if isinstance(raw.columns, pd.MultiIndex):
        # For a single ticker, keep the price-field level.
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()
    raw.columns = [str(c).strip().lower().replace(" ","_") for c in raw.columns]

    dt_col = next((c for c in ("datetime","date","timestamp") if c in raw.columns), None)
    if dt_col is None:
        return pd.DataFrame()
    raw["datetime"] = pd.to_datetime(raw[dt_col], errors="coerce")

    wanted = ["datetime","open","high","low","close","volume"]
    for c in wanted[1:]:
        if c not in raw.columns:
            if c != "volume":
                return pd.DataFrame()
            raw[c] = 0
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    out = raw[wanted].dropna(subset=["datetime","open","high","low","close"]).copy()
    out = out.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    out["volume"] = out["volume"].fillna(0)
    return out

def indicators(df):
    d = df.copy()

    d["ema20"] = d.close.ewm(span=20, adjust=False).mean()
    d["ema50"] = d.close.ewm(span=50, adjust=False).mean()

    delta = d.close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - (100/(1+rs))

    tr = pd.concat([
        d.high-d.low,
        (d.high-d.close.shift()).abs(),
        (d.low-d.close.shift()).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14, min_periods=14).mean()

    vol = d.volume.copy()
    # Yahoo can occasionally return zero volume for an index.
    # Use price-weighted cumulative average in that case.
    if vol.sum() <= 0:
        d["vwap"] = d.close.expanding().mean()
    else:
        cv = vol.cumsum().replace(0, np.nan)
        d["vwap"] = (d.close*vol).cumsum()/cv
        d["vwap"] = d["vwap"].fillna(d["close"].expanding().mean())

    hl2=(d.high+d.low)/2
    upper=hl2+3*d.atr
    lower=hl2-3*d.atr
    trend=np.ones(len(d))
    for i in range(1,len(d)):
        if pd.isna(d.atr.iloc[i]):
            trend[i]=trend[i-1]
        elif d.close.iloc[i] > upper.iloc[i-1]:
            trend[i]=1
        elif d.close.iloc[i] < lower.iloc[i-1]:
            trend[i]=-1
        else:
            trend[i]=trend[i-1]
    d["supertrend_dir"]=trend
    return d

def score(r):
    buy=(25 if r.ema20>r.ema50 else 0)+(20 if r.rsi>=55 else 0)+(20 if r.close>r.vwap else 0)+(25 if r.supertrend_dir>0 else 0)+(10 if r.atr>0 else 0)
    sell=(25 if r.ema20<r.ema50 else 0)+(20 if r.rsi<=45 else 0)+(20 if r.close<r.vwap else 0)+(25 if r.supertrend_dir<0 else 0)+(10 if r.atr>0 else 0)
    if buy>=70 and buy>sell: return "BUY",buy
    if sell>=70 and sell>buy: return "SELL",sell
    return "NO TRADE",max(buy,sell)

def backtest(d,capital,risk_pct,sl_atr,rr,max_trades_day=3):
    cash=float(capital); pos=None; trades=[]; day_count={}
    for _,r in d.iterrows():
        if pd.isna(r.atr) or pd.isna(r.rsi): continue
        day=str(r.datetime)[:10]
        sig,conf=score(r)

        if pos:
            exit_price=reason=None
            if pos["side"]=="BUY":
                if r.low<=pos["sl"]: exit_price,reason=pos["sl"],"SL"
                elif r.high>=pos["target"]: exit_price,reason=pos["target"],"TARGET"
            else:
                if r.high>=pos["sl"]: exit_price,reason=pos["sl"],"SL"
                elif r.low<=pos["target"]: exit_price,reason=pos["target"],"TARGET"
            if exit_price is not None:
                pnl=(exit_price-pos["entry"])*pos["qty"]*(1 if pos["side"]=="BUY" else -1)
                cash+=pnl
                trades.append({**pos,"exit":exit_price,"pnl":pnl,"reason":reason,"exit_time":r.datetime})
                pos=None

        if pos is None and sig in ("BUY","SELL") and day_count.get(day,0)<max_trades_day:
            dist=max(float(r.atr)*sl_atr,0.01)
            qty=max(1,int((cash*risk_pct/100)/dist))
            entry=float(r.close)
            sl=entry-dist if sig=="BUY" else entry+dist
            target=entry+dist*rr if sig=="BUY" else entry-dist*rr
            pos={"side":sig,"entry":entry,"sl":sl,"target":target,"qty":qty,"entry_time":r.datetime,"confidence":conf}
            day_count[day]=day_count.get(day,0)+1
    return pd.DataFrame(trades),cash

st.title("🤖 ALS AI Algo Trading")
st.caption("V4 • Robust market-data normalization • Backtest + Paper Trading • Groww API not required")

with st.sidebar:
    symbol=st.selectbox("Symbol",list(SYMBOLS))
    interval=st.selectbox("Candle interval",["5m","15m","1h","1d"])
    period=st.selectbox("Historical period",periods(interval))
    capital=st.number_input("Capital",100000,10000000,100000,10000)
    risk=st.slider("Risk / trade (%)",0.25,2.0,1.0,0.25)
    sl_atr=st.slider("SL ATR",0.5,3.0,1.5,0.25)
    rr=st.slider("Reward : Risk",1.0,4.0,2.0,0.5)

st.info("No CSV required. Fetching historical market data automatically.")

if st.button("📥 Fetch Market Data & Run Backtest",type="primary"):
    try:
        import yfinance as yf
        raw=yf.download(SYMBOLS[symbol],period=period,interval=interval,auto_adjust=False,progress=False,threads=False)
        d0=clean_yahoo(raw)

        if d0.empty:
            st.error("Market-data provider returned no usable OHLC data. Try NIFTY + 5m + 5d.")
        else:
            d=indicators(d0)
            valid=d.dropna(subset=["ema20","ema50","rsi","atr","vwap"])
            st.caption(f"Received {len(d)} candles. Valid indicator rows: {len(valid)}.")

            if valid.empty:
                st.error("OHLC data arrived, but indicator calculation produced no valid rows.")
            else:
                last=valid.iloc[-1]
                sig,conf=score(last)
                c1,c2,c3,c4=st.columns(4)
                c1.metric("Symbol",symbol); c2.metric("Signal",sig); c3.metric("Confidence",f"{conf}%"); c4.metric("Last Close",f"{last.close:.2f}")

                trades,final_cash=backtest(d,capital,risk,sl_atr,rr)
                pnl=final_cash-capital
                wins=int((trades.pnl>0).sum()) if len(trades) else 0
                winrate=wins/len(trades)*100 if len(trades) else 0
                c1,c2,c3,c4=st.columns(4)
                c1.metric("Final Capital",f"₹{final_cash:,.2f}"); c2.metric("P&L",f"₹{pnl:,.2f}"); c3.metric("Trades",len(trades)); c4.metric("Win Rate",f"{winrate:.1f}%")
                st.subheader("📊 Recent Data & Indicators")
                st.dataframe(d.tail(30),use_container_width=True)
                st.subheader("🧾 Trade Log")
                st.dataframe(trades,use_container_width=True)
                if len(trades):
                    st.download_button("Download Trade CSV",trades.to_csv(index=False),"trade_log.csv","text/csv")
    except Exception as ex:
        st.error(f"Data/backtest error: {ex}")

st.divider()
st.caption("Research/paper-trading only. Backtests do not guarantee future profits.")
