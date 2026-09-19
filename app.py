
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ALS AI Algo Trading V9", page_icon="🤖", layout="wide")
SYMBOLS={"NIFTY":"^NSEI","BANK NIFTY":"^NSEBANK","SENSEX":"^BSESN"}

def fetch(ticker,period,interval):
    import yfinance as yf
    x=yf.download(ticker,period=period,interval=interval,auto_adjust=False,progress=False,threads=False)
    if x is None or x.empty:return pd.DataFrame()
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    x=x.reset_index(); x.columns=[str(c).lower().replace(" ","_") for c in x.columns]
    dt=next((c for c in ("datetime","date","timestamp") if c in x.columns),None)
    if not dt:return pd.DataFrame()
    x["datetime"]=pd.to_datetime(x[dt],errors="coerce")
    for c in ("open","high","low","close"):
        if c not in x:return pd.DataFrame()
        x[c]=pd.to_numeric(x[c],errors="coerce")
    if "volume" not in x:x["volume"]=0
    x["volume"]=pd.to_numeric(x["volume"],errors="coerce").fillna(0)
    return x[["datetime","open","high","low","close","volume"]].dropna().drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)

def add_i(d):
    d=d.copy()
    for n,s in [(9,9),(20,20),(50,50),(200,200)]: d[f"ema{n}"]=d.close.ewm(span=s,adjust=False).mean()
    de=d.close.diff(); up=de.clip(lower=0).rolling(14,min_periods=14).mean(); dn=(-de.clip(upper=0)).rolling(14,min_periods=14).mean()
    d["rsi"]=100-100/(1+up/dn.replace(0,np.nan))
    tr=pd.concat([(d.high-d.low),(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()],axis=1).max(axis=1)
    d["atr"]=tr.rolling(14,min_periods=14).mean(); d["atr_pct"]=d.atr/d.close*100
    d["atr_med"]=d.atr.rolling(50,min_periods=20).median()
    d["vwap"]=((d.close*d.volume).cumsum()/d.volume.cumsum().replace(0,np.nan)) if d.volume.sum()>0 else d.close.expanding().mean()
    d["ret5"]=d.close.pct_change(5); d["range_pct"]=(d.high-d.low)/d.close*100
    return d

def regime(r):
    if pd.isna(r.atr) or pd.isna(r.atr_med): return "UNKNOWN"
    trend=abs(r.ema20-r.ema50)/r.close*100
    if trend<0.035:return "SIDEWAYS"
    if r.atr>r.atr_med*1.8:return "HIGH_VOL"
    if r.ema20>r.ema50 and r.ema50>r.ema200:return "UPTREND"
    if r.ema20<r.ema50 and r.ema50<r.ema200:return "DOWNTREND"
    return "MIXED"

def sig(r):
    rg=regime(r)
    # No-trade in sideways/high-volatility regimes.
    if rg in ("SIDEWAYS","HIGH_VOL","UNKNOWN"):return "NO TRADE",0,rg
    if rg=="UPTREND" and r.close>r.vwap and r.close>r.ema20 and 52<=r.rsi<=68 and r.ret5>0:
        return "BUY",5,rg
    if rg=="DOWNTREND" and r.close<r.vwap and r.close<r.ema20 and 32<=r.rsi<=48 and r.ret5<0:
        return "SELL",5,rg
    return "NO TRADE",0,rg

def bt(d,capital,risk,rr,slatr,maxtrades,cost=20,slip=2):
    cash=float(capital); pos=None; trades=[]; dayn={}; dp={}
    eq=[]; peak=cash; maxdd=0
    for _,r in d.iterrows():
        if pd.isna(r.atr) or pd.isna(r.ema200):eq.append({"datetime":r.datetime,"equity":cash});continue
        day=str(r.datetime)[:10]; dayn.setdefault(day,0); dp.setdefault(day,0)
        s,score,rg=sig(r)
        if pos:
            if pos["side"]=="BUY":
                pos["sl"]=max(pos["sl"],float(r.close-r.atr*1.25))
                if r.close>=pos["entry"]+pos["risk"]:pos["sl"]=max(pos["sl"],pos["entry"])
                xp=pos["sl"] if r.low<=pos["sl"] else (pos["target"] if r.high>=pos["target"] else None)
            else:
                pos["sl"]=min(pos["sl"],float(r.close+r.atr*1.25))
                if r.close<=pos["entry"]-pos["risk"]:pos["sl"]=min(pos["sl"],pos["entry"])
                xp=pos["sl"] if r.high>=pos["sl"] else (pos["target"] if r.low<=pos["target"] else None)
            if xp is not None:
                xp=float(xp)*(1-slip/10000 if pos["side"]=="BUY" else 1+slip/10000)
                pnl=(xp-pos["entry"])*pos["qty"]*(1 if pos["side"]=="BUY" else -1)-cost
                cash+=pnl; dp[day]+=pnl
                trades.append({**pos,"regime":pos["regime"],"exit":xp,"pnl":pnl,"exit_time":r.datetime});pos=None
        if pos is None and s!="NO TRADE" and dayn[day]<maxtrades and dp[day]>-3000:
            dist=max(float(r.atr)*slatr,.01); qty=max(1,int((cash*risk/100)/dist))
            entry=float(r.close)*(1+slip/10000 if s=="BUY" else 1-slip/10000)
            pos={"side":s,"entry":entry,"sl":entry-dist if s=="BUY" else entry+dist,
                 "target":entry+dist*rr if s=="BUY" else entry-dist*rr,"qty":qty,
                 "entry_time":r.datetime,"score":score,"risk":dist,"regime":rg}
            dayn[day]+=1
        mark=cash
        if pos is not None: mark+=(float(r.close)-pos["entry"])*pos["qty"]*(1 if pos["side"]=="BUY" else -1)
        peak=max(peak,mark);maxdd=max(maxdd,(peak-mark)/peak*100 if peak else 0)
        eq.append({"datetime":r.datetime,"equity":mark})
    t=pd.DataFrame(trades); gp=t.loc[t.pnl>0,"pnl"].sum() if len(t) else 0; gl=-t.loc[t.pnl<0,"pnl"].sum() if len(t) else 0
    return t,pd.DataFrame(eq),cash,{"P&L":cash-capital,"Trades":len(t),"Win Rate %":((t.pnl>0).mean()*100 if len(t) else 0),"PF":(gp/gl if gl else 0),"Expectancy":(t.pnl.mean() if len(t) else 0),"Max DD %":maxdd}

st.title("🤖 ALS AI Algo Trading V9")
st.caption("Market-regime filter • Selective entries • Walk-forward validation • Paper trading only")
with st.sidebar:
    interval=st.selectbox("Interval",["5m","15m","1h","1d"])
    period=st.selectbox("Period",{"5m":["5d"],"15m":["5d","1mo"],"1h":["1mo","3mo","6mo","1y"],"1d":["1mo","3mo","6mo","1y","2y","5y"]}[interval])
    capital=st.number_input("Starting capital",100000,10000000,100000,10000)
    risk=st.slider("Risk / trade (%)",0.25,1.0,0.5,0.25)
    rr=st.slider("Reward : Risk",1.0,3.5,2.0,0.5)
    slatr=st.slider("Initial SL (ATR)",1.0,2.5,1.5,0.25)
    maxtrades=st.slider("Max trades/day",1,4,2)

st.info("V9 rejects sideways and unusually high-volatility regimes before entry. This is a research filter, not a profitability guarantee.")
if st.button("🚀 Run V9 Validation",type="primary"):
    rows=[];details={};pr=st.progress(0)
    for i,(name,ticker) in enumerate(SYMBOLS.items(),1):
        d=fetch(ticker,period,interval)
        if len(d)<140: rows.append({"Symbol":name,"Status":"Not enough data"});pr.progress(i/3);continue
        d=add_i(d).dropna(subset=["ema200","atr","rsi","atr_med"])
        split=int(len(d)*.65); trn=d.iloc[:split]; val=d.iloc[split:]
        tr,eq,fc,tm=bt(trn,capital,risk,rr,slatr,maxtrades); vt,ve,vf,vm=bt(val,capital,risk,rr,slatr,maxtrades)
        rows.append({"Symbol":name,"Train P&L":tm["P&L"],"Train Trades":tm["Trades"],"Validation P&L":vm["P&L"],"Validation Trades":vm["Trades"],"Validation Win %":vm["Win Rate %"],"Validation PF":vm["PF"],"Validation Expectancy":vm["Expectancy"],"Max DD %":vm["Max DD %"]})
        details[name]=(val,vt,ve,vm)
        pr.progress(i/3)
    res=pd.DataFrame(rows);st.subheader("📊 V9 Validation Results");st.dataframe(res,use_container_width=True)
    for name,(d,t,e,m) in details.items():
        st.subheader(f"📈 {name}")
        a,b,c,d4=st.columns(4);a.metric("P&L",f"₹{m['P&L']:,.2f}");b.metric("Win Rate",f"{m['Win Rate %']:.1f}%");c.metric("Profit Factor",f"{m['PF']:.2f}");d4.metric("Max DD",f"{m['Max DD %']:.2f}%")
        st.line_chart(e.set_index("datetime")[["equity"]]);st.dataframe(t,use_container_width=True)
        if len(t):st.download_button(f"Download {name} V9 trades",t.to_csv(index=False),f"{name.lower().replace(' ','_')}_v9_trades.csv","text/csv",key=f"{name}_v9")
st.divider();st.caption("Research/paper-trading only. Historical validation does not guarantee future results.")
