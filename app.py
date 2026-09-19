
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ALS AI Algo V7", page_icon="🤖", layout="wide")

SYMBOLS = {"NIFTY":"^NSEI", "BANK NIFTY":"^NSEBANK", "SENSEX":"^BSESN"}

def periods(interval):
    return {"5m":["5d"],"15m":["5d","1mo"],"1h":["1mo","3mo","6mo","1y"],
            "1d":["1mo","3mo","6mo","1y","2y","5y"]}[interval]

def fetch(ticker, period, interval):
    import yfinance as yf
    x = yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                    progress=False, threads=False)
    if x is None or x.empty: return pd.DataFrame()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns=x.columns.get_level_values(0)
    x=x.reset_index()
    x.columns=[str(c).lower().replace(" ","_") for c in x.columns]
    dt=next((c for c in ["datetime","date","timestamp"] if c in x.columns),None)
    if not dt: return pd.DataFrame()
    x["datetime"]=pd.to_datetime(x[dt],errors="coerce")
    for c in ["open","high","low","close"]:
        if c not in x: return pd.DataFrame()
        x[c]=pd.to_numeric(x[c],errors="coerce")
    if "volume" not in x: x["volume"]=0
    x["volume"]=pd.to_numeric(x["volume"],errors="coerce").fillna(0)
    return x[["datetime","open","high","low","close","volume"]].dropna().drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)

def ind(d):
    d=d.copy()
    d["ema9"]=d.close.ewm(span=9,adjust=False).mean()
    d["ema20"]=d.close.ewm(span=20,adjust=False).mean()
    d["ema50"]=d.close.ewm(span=50,adjust=False).mean()
    d["ema200"]=d.close.ewm(span=200,adjust=False).mean()
    delta=d.close.diff()
    up=delta.clip(lower=0).rolling(14,min_periods=14).mean()
    dn=(-delta.clip(upper=0)).rolling(14,min_periods=14).mean()
    d["rsi"]=100-100/(1+(up/dn.replace(0,np.nan)))
    tr=pd.concat([(d.high-d.low),(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()],axis=1).max(axis=1)
    d["atr"]=tr.rolling(14,min_periods=14).mean()
    d["atr_pct"]=d.atr/d.close*100
    if d.volume.sum()>0:
        d["vwap"]=(d.close*d.volume).cumsum()/d.volume.cumsum().replace(0,np.nan)
    else: d["vwap"]=d.close.expanding().mean()
    d["vol_med"]=d.atr.rolling(50,min_periods=20).median()
    d["ret1"]=d.close.pct_change()
    return d

# Three deliberately different rule families. No future data is used.
def candidate(r, mode):
    trend_up=r.ema20>r.ema50 and r.ema50>r.ema200
    trend_dn=r.ema20<r.ema50 and r.ema50<r.ema200
    if pd.isna(r.atr) or pd.isna(r.ema200): return "NO TRADE",0
    if mode=="TREND":
        if trend_up and 52<=r.rsi<=70 and r.close>r.vwap and r.close>r.ema9: return "BUY",4
        if trend_dn and 30<=r.rsi<=48 and r.close<r.vwap and r.close<r.ema9: return "SELL",4
    elif mode=="PULLBACK":
        if trend_up and r.close>r.ema50 and r.ema9>r.ema20 and 45<=r.rsi<=60: return "BUY",3
        if trend_dn and r.close<r.ema50 and r.ema9<r.ema20 and 40<=r.rsi<=55: return "SELL",3
    else: # MOMENTUM
        if trend_up and r.close>r.close*0 + r.ema20 and r.rsi>60 and r.ret1>0: return "BUY",3
        if trend_dn and r.close<r.ema20 and r.rsi<40 and r.ret1<0: return "SELL",3
    return "NO TRADE",0

def backtest(d, mode, capital=100000, risk=0.5, rr=2.0, sl_atr=1.5, maxtrades=2,
             brokerage=20, slip_bps=2):
    cash=float(capital); pos=None; trades=[]; daytrades={}; daily_pnl={}
    eq=[]; peak=cash; maxdd=0
    for _,r in d.iterrows():
        if pd.isna(r.atr) or pd.isna(r.ema200): 
            eq.append({"datetime":r.datetime,"equity":cash}); continue
        day=str(r.datetime)[:10]; daytrades.setdefault(day,0); daily_pnl.setdefault(day,0)
        sig,score=candidate(r,mode)
        if pos:
            if pos["side"]=="BUY":
                pos["sl"]=max(pos["sl"],float(r.close-r.atr*1.25))
                if r.close>=pos["entry"]+pos["risk_dist"]: pos["sl"]=max(pos["sl"],pos["entry"])
                xp=pos["sl"] if r.low<=pos["sl"] else (pos["target"] if r.high>=pos["target"] else None)
            else:
                pos["sl"]=min(pos["sl"],float(r.close+r.atr*1.25))
                if r.close<=pos["entry"]-pos["risk_dist"]: pos["sl"]=min(pos["sl"],pos["entry"])
                xp=pos["sl"] if r.high>=pos["sl"] else (pos["target"] if r.low<=pos["target"] else None)
            if xp is not None:
                xp=float(xp)*(1-(slip_bps/10000) if pos["side"]=="BUY" else 1+(slip_bps/10000))
                pnl=(xp-pos["entry"])*pos["qty"]*(1 if pos["side"]=="BUY" else -1)-brokerage
                cash+=pnl; daily_pnl[day]+=pnl
                trades.append({**pos,"exit":xp,"pnl":pnl,"exit_time":r.datetime})
                pos=None
        if pos is None and sig!="NO TRADE" and daytrades[day]<maxtrades and daily_pnl[day]>-3000:
            dist=max(float(r.atr)*sl_atr,0.01)
            qty=max(1,int((cash*risk/100)/dist))
            entry=float(r.close)*(1+(slip_bps/10000) if sig=="BUY" else 1-(slip_bps/10000))
            sl=entry-dist if sig=="BUY" else entry+dist
            target=entry+dist*rr if sig=="BUY" else entry-dist*rr
            pos={"side":sig,"entry":entry,"sl":sl,"target":target,"qty":qty,
                 "entry_time":r.datetime,"score":score,"risk_dist":dist,"mode":mode}
            daytrades[day]+=1
        mark=cash+(float(r.close)-pos["entry"])*pos["qty"]*(1 if pos and pos["side"]=="BUY" else -1 if pos else 0)
        peak=max(peak,mark); maxdd=max(maxdd,(peak-mark)/peak*100 if peak else 0)
        eq.append({"datetime":r.datetime,"equity":mark})
    t=pd.DataFrame(trades); grossp=t.loc[t.pnl>0,"pnl"].sum() if len(t) else 0; grossl=-t.loc[t.pnl<0,"pnl"].sum() if len(t) else 0
    pf=grossp/grossl if grossl else (np.inf if grossp else 0)
    wr=(t.pnl>0).mean()*100 if len(t) else 0
    return t,pd.DataFrame(eq),cash,{"P&L":cash-capital,"Trades":len(t),"Win Rate %":wr,"Profit Factor":pf,"Expectancy":t.pnl.mean() if len(t) else 0,"Max DD %":maxdd}

st.title("🤖 ALS AI Algo Trading V7")
st.caption("Strategy research engine • Trend / Pullback / Momentum • Walk-forward validation • Paper trading only")

with st.sidebar:
    interval=st.selectbox("Interval",["5m","15m","1h","1d"])
    period=st.selectbox("Period",periods(interval))
    capital=st.number_input("Starting capital",100000,10000000,100000,10000)
    risk=st.slider("Risk per trade (%)",0.25,1.0,0.5,0.25)
    rr=st.slider("Reward : Risk",1.0,3.5,2.0,0.5)
    sl_atr=st.slider("Initial SL (ATR)",1.0,2.5,1.5,0.25)
    maxtrades=st.slider("Max trades/day",1,4,2)

st.info("V7 compares three rule families and validates the selected family on later candles. Costs and slippage are simulated. A positive backtest is not a guarantee of future profit.")

if st.button("🚀 Run V7 Walk-Forward Test",type="primary"):
    rows=[]; details={}
    progress=st.progress(0)
    for ix,(name,ticker) in enumerate(SYMBOLS.items(),1):
        d=fetch(ticker,period,interval)
        if len(d)<120:
            rows.append({"Symbol":name,"Status":"Not enough data"})
            progress.progress(ix/3); continue
        d=ind(d).dropna(subset=["ema200","atr","rsi"])
        split=int(len(d)*0.65)
        train=d.iloc[:split]; valid=d.iloc[split:]
        candidates=[]
        for mode in ["TREND","PULLBACK","MOMENTUM"]:
            tr,eq,fc,m=backtest(train,mode,capital,risk,rr,sl_atr,maxtrades)
            # Selection score rewards profit factor and penalizes drawdown; requires trades.
            pf=m["Profit Factor"] if np.isfinite(m["Profit Factor"]) else 5
            sel=(pf*np.log1p(m["Trades"])-m["Max DD %"]*0.15) if m["Trades"] else -999
            candidates.append((sel,mode,m,tr,eq))
        candidates.sort(reverse=True,key=lambda x:x[0])
        _,best,tm,_,_=candidates[0]
        vt,ve,vf,vm=backtest(valid,best,capital,risk,rr,sl_atr,maxtrades)
        rows.append({"Symbol":name,"Selected":best,"Train P&L":tm["P&L"],"Train PF":tm["Profit Factor"],
                     "Validation P&L":vm["P&L"],"Validation Trades":vm["Trades"],
                     "Validation Win %":vm["Win Rate %"],"Validation PF":vm["Profit Factor"],
                     "Validation Expectancy":vm["Expectancy"],"Validation Max DD %":vm["Max DD %"]})
        details[name]=(best,valid,vt,ve,vm)
        progress.progress(ix/3)
    res=pd.DataFrame(rows)
    st.subheader("📊 Walk-Forward Results")
    st.dataframe(res,use_container_width=True)
    for name in SYMBOLS:
        if name not in details: continue
        best,d,t,e,m=details[name]
        st.subheader(f"📈 {name} — Selected: {best}")
        a,b,c,dcol=st.columns(4)
        a.metric("Validation P&L",f"₹{m['P&L']:,.2f}")
        b.metric("Win Rate",f"{m['Win Rate %']:.1f}%")
        c.metric("Profit Factor",f"{m['Profit Factor']:.2f}")
        dcol.metric("Max DD",f"{m['Max DD %']:.2f}%")
        st.line_chart(e.set_index("datetime")[["equity"]])
        st.dataframe(t,use_container_width=True)
        if len(t):
            st.download_button(f"Download {name} V7 trades",t.to_csv(index=False),
                               f"{name.lower().replace(' ','_')}_v7_trades.csv","text/csv",key=f"dl_{name}")

st.divider()
st.caption("Research/paper-trading only. Strategy selection is mechanical; no profitability or future-performance guarantee.")
