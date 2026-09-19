
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ALS AI Algo V6", page_icon="🤖", layout="wide")

SYMBOLS = {"NIFTY":"^NSEI", "BANK NIFTY":"^NSEBANK", "SENSEX":"^BSESN"}

def allowed_periods(interval):
    if interval == "5m": return ["5d"]
    if interval == "15m": return ["5d","1mo"]
    if interval == "1h": return ["1mo","3mo","6mo","1y"]
    return ["1mo","3mo","6mo","1y","2y","5y"]

def fetch_data(ticker, period, interval):
    import yfinance as yf
    raw = yf.download(ticker, period=period, interval=interval,
                      auto_adjust=False, progress=False, threads=False)
    if raw is None or raw.empty: return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    raw.columns = [str(c).strip().lower().replace(" ","_") for c in raw.columns]
    dt = next((c for c in ("datetime","date","timestamp") if c in raw.columns), None)
    if not dt: return pd.DataFrame()
    raw["datetime"] = pd.to_datetime(raw[dt], errors="coerce")
    for c in ("open","high","low","close"):
        if c not in raw.columns: return pd.DataFrame()
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    if "volume" not in raw.columns: raw["volume"] = 0
    raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce").fillna(0)
    return raw[["datetime","open","high","low","close","volume"]].dropna().sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

def indicators(d):
    d=d.copy()
    d["ema20"]=d.close.ewm(span=20,adjust=False).mean()
    d["ema50"]=d.close.ewm(span=50,adjust=False).mean()
    d["ema200"]=d.close.ewm(span=200,adjust=False).mean()
    delta=d.close.diff()
    gain=delta.clip(lower=0).rolling(14,min_periods=14).mean()
    loss=(-delta.clip(upper=0)).rolling(14,min_periods=14).mean()
    rs=gain/loss.replace(0,np.nan)
    d["rsi"]=100-100/(1+rs)
    tr=pd.concat([(d.high-d.low),(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()],axis=1).max(axis=1)
    d["atr"]=tr.rolling(14,min_periods=14).mean()
    d["atr_pct"]=d.atr/d.close*100
    if d.volume.sum()>0:
        d["vwap"]=(d.close*d.volume).cumsum()/d.volume.cumsum().replace(0,np.nan)
    else:
        d["vwap"]=d.close.expanding().mean()
    # ATR-adaptive trend direction
    hl2=(d.high+d.low)/2
    upper=hl2+2.5*d.atr
    lower=hl2-2.5*d.atr
    trend=np.ones(len(d))
    for i in range(1,len(d)):
        if pd.isna(d.atr.iloc[i]): trend[i]=trend[i-1]
        elif d.close.iloc[i]>upper.iloc[i-1]: trend[i]=1
        elif d.close.iloc[i]<lower.iloc[i-1]: trend[i]=-1
        else: trend[i]=trend[i-1]
    d["trend"]=trend
    d["ema_spread"]=(d.ema20-d.ema50).abs()/d.close*100
    return d

def signal(r, threshold=80, min_spread=0.03):
    buy=0; sell=0
    # Stronger weighting; require trend agreement.
    buy += 20 if r.ema20>r.ema50 else 0
    buy += 20 if r.ema50>r.ema200 else 0
    buy += 15 if 55<=r.rsi<=72 else 0
    buy += 15 if r.close>r.vwap else 0
    buy += 20 if r.trend>0 else 0
    buy += 10 if r.close>r.ema20 else 0

    sell += 20 if r.ema20<r.ema50 else 0
    sell += 20 if r.ema50<r.ema200 else 0
    sell += 15 if 28<=r.rsi<=45 else 0
    sell += 15 if r.close<r.vwap else 0
    sell += 20 if r.trend<0 else 0
    sell += 10 if r.close<r.ema20 else 0

    if pd.isna(r.atr_pct) or r.atr_pct<0.04 or r.ema_spread<min_spread:
        return "NO TRADE", max(buy,sell)
    if buy>=threshold and buy>sell: return "BUY",buy
    if sell>=threshold and sell>buy: return "SELL",sell
    return "NO TRADE",max(buy,sell)

def backtest(d, capital, risk_pct, sl_atr, rr, trail_atr, max_trades, threshold, max_daily_loss):
    cash=float(capital); pos=None; trades=[]; daily={}
    equity=[]; peak=cash; maxdd=0
    for _,r in d.iterrows():
        if pd.isna(r.atr) or pd.isna(r.rsi) or pd.isna(r.ema200):
            equity.append({"datetime":r.datetime,"equity":cash}); continue
        day=str(r.datetime)[:10]
        daily.setdefault(day,0.0)
        sig,conf=signal(r,threshold)

        if pos:
            if pos["side"]=="BUY":
                pos["sl"]=max(pos["sl"],float(r.close-r.atr*trail_atr))
                if r.close>=pos["entry"]+pos["risk_dist"]: pos["sl"]=max(pos["sl"],pos["entry"])
                exit_price=reason=None
                if r.low<=pos["sl"]: exit_price,reason=pos["sl"],"SL/TRAIL"
                elif r.high>=pos["target"]: exit_price,reason=pos["target"],"TARGET"
            else:
                pos["sl"]=min(pos["sl"],float(r.close+r.atr*trail_atr))
                if r.close<=pos["entry"]-pos["risk_dist"]: pos["sl"]=min(pos["sl"],pos["entry"])
                exit_price=reason=None
                if r.high>=pos["sl"]: exit_price,reason=pos["sl"],"SL/TRAIL"
                elif r.low<=pos["target"]: exit_price,reason=pos["target"],"TARGET"
            if exit_price is not None:
                pnl=(exit_price-pos["entry"])*pos["qty"]*(1 if pos["side"]=="BUY" else -1)
                cash+=pnl; daily[day]+=pnl
                trades.append({**pos,"exit":exit_price,"pnl":pnl,"reason":reason,"exit_time":r.datetime})
                pos=None

        if pos is None and sig in ("BUY","SELL") and len([x for x in trades if str(x["entry_time"])[:10]==day])<max_trades and daily[day]>-max_daily_loss:
            dist=max(float(r.atr)*sl_atr,0.01)
            qty=max(1,int((cash*risk_pct/100)/dist))
            entry=float(r.close)
            sl=entry-dist if sig=="BUY" else entry+dist
            target=entry+dist*rr if sig=="BUY" else entry-dist*rr
            pos={"side":sig,"entry":entry,"sl":sl,"target":target,"qty":qty,"entry_time":r.datetime,"confidence":conf,"risk_dist":dist}
        mark=cash
        if pos: mark+=(float(r.close)-pos["entry"])*pos["qty"]*(1 if pos["side"]=="BUY" else -1)
        peak=max(peak,mark); maxdd=max(maxdd,(peak-mark)/peak*100 if peak else 0)
        equity.append({"datetime":r.datetime,"equity":mark})
    return pd.DataFrame(trades),pd.DataFrame(equity),cash,maxdd

def metrics(trades, capital, final_cash, maxdd):
    n=len(trades)
    wins=int((trades.pnl>0).sum()) if n else 0
    gross_profit=trades.loc[trades.pnl>0,"pnl"].sum() if n else 0
    gross_loss=-trades.loc[trades.pnl<0,"pnl"].sum() if n else 0
    pf=(gross_profit/gross_loss) if gross_loss>0 else (np.inf if gross_profit>0 else 0)
    avg=(trades.pnl.mean() if n else 0)
    return {"P&L":final_cash-capital,"Trades":n,"Win Rate %":wins/n*100 if n else 0,
            "Profit Factor":pf,"Expectancy":avg,"Max DD %":maxdd}

st.title("🤖 ALS AI Algo Trading V6")
st.caption("Selective trend strategy • Out-of-sample test • Risk controls • Paper Trading")

with st.sidebar:
    interval=st.selectbox("Candle interval",["5m","15m","1h","1d"])
    period=st.selectbox("Historical period",allowed_periods(interval))
    capital=st.number_input("Starting capital",100000,10000000,100000,10000)
    risk=st.slider("Risk / trade (%)",0.25,1.5,0.75,0.25)
    threshold=st.slider("Signal threshold",70,95,80,5)
    sl_atr=st.slider("Initial SL ATR",0.75,3.0,1.5,0.25)
    rr=st.slider("Reward : Risk",1.0,4.0,2.0,0.5)
    trail=st.slider("Trailing SL ATR",0.5,3.0,1.5,0.25)
    maxtrades=st.slider("Max trades/day",1,5,2)
    maxloss=st.number_input("Max daily loss ₹",1000,100000,3000,500)

st.info("V6 is intentionally more selective. It separates signal quality from trade frequency and reports Profit Factor, Expectancy and Drawdown.")

if st.button("🚀 Run V6 Validation",type="primary"):
    summary=[]; detail={}
    progress=st.progress(0)
    for i,(name,ticker) in enumerate(SYMBOLS.items(),1):
        d=fetch_data(ticker,period,interval)
        if d.empty:
            summary.append({"Symbol":name,"Status":"No data"}); progress.progress(i/3); continue
        d=indicators(d)
        # Chronological 70/30 train/validation split.
        split=max(int(len(d)*0.70),200)
        train=d.iloc[:split].copy()
        valid=d.iloc[split:].copy()
        for label,part in [("Train",train),("Validation",valid)]:
            if len(part)<60:
                summary.append({"Symbol":name,"Set":label,"Status":"Too little data"}); continue
            tr,eq,final,dd=backtest(part,capital,risk,sl_atr,rr,trail,maxtrades,threshold,maxloss)
            m=metrics(tr,capital,final,dd)
            m.update({"Symbol":name,"Set":label,"Status":"OK","Candles":len(part)})
            summary.append(m)
            detail[(name,label)]=(part,tr,eq,final)
        progress.progress(i/3)

    res=pd.DataFrame(summary)
    if not res.empty:
        st.subheader("📊 Train vs Validation")
        show=res.copy()
        for c in ("P&L","Win Rate %","Profit Factor","Expectancy","Max DD %"):
            if c in show.columns: show[c]=show[c].apply(lambda x: round(x,2) if isinstance(x,(int,float,np.floating)) and np.isfinite(x) else x)
        st.dataframe(show,use_container_width=True)

        for name in SYMBOLS:
            for label in ("Train","Validation"):
                key=(name,label)
                if key not in detail: continue
                d,tr,eq,final=detail[key]
                st.subheader(f"📈 {name} — {label}")
                st.line_chart(eq.set_index("datetime")[["equity"]])
                st.dataframe(tr,use_container_width=True)
                if len(tr):
                    st.download_button(f"Download {name} {label}",tr.to_csv(index=False),
                                       f"{name.lower().replace(' ','_')}_{label.lower()}_trades.csv","text/csv",
                                       key=f"{name}_{label}")

st.divider()
st.caption("Research/paper-trading only. Validation results are historical and do not guarantee future performance.")
