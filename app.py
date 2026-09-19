
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ALS AI Algo Trading V10", page_icon="🤖", layout="wide")
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
    x["volume"]=pd.to_numeric(x.get("volume",0),errors="coerce").fillna(0)
    return x[["datetime","open","high","low","close","volume"]].dropna().drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)

def ind(d):
    d=d.copy()
    for n in (9,20,50,200): d[f"ema{n}"]=d.close.ewm(span=n,adjust=False).mean()
    ch=d.close.diff(); up=ch.clip(lower=0).rolling(14,min_periods=14).mean(); dn=(-ch.clip(upper=0)).rolling(14,min_periods=14).mean()
    d["rsi"]=100-100/(1+up/dn.replace(0,np.nan))
    tr=pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()],axis=1).max(axis=1)
    d["atr"]=tr.rolling(14,min_periods=14).mean(); d["atr_med"]=d.atr.rolling(50,min_periods=20).median()
    d["vwap"]=((d.close*d.volume).cumsum()/d.volume.cumsum().replace(0,np.nan)) if d.volume.sum()>0 else d.close.expanding().mean()
    d["ret3"]=d.close.pct_change(3); d["ret8"]=d.close.pct_change(8)
    d["trend"]=abs(d.ema20-d.ema50)/d.close*100
    return d

# Symbol-specific signal profiles. They are intentionally conservative.
PROFILES={
 "NIFTY": dict(trend=.045,rsi_buy=(54,66),rsi_sell=(34,46),sl=1.45,rr=2.0),
 "BANK NIFTY": dict(trend=.060,rsi_buy=(53,68),rsi_sell=(32,47),sl=1.65,rr=2.1),
 "SENSEX": dict(trend=.035,rsi_buy=(53,67),rsi_sell=(33,47),sl=1.40,rr=2.2)
}

def regime(r,p):
    if pd.isna(r.atr) or pd.isna(r.atr_med): return "UNKNOWN"
    if r.atr>r.atr_med*1.75:return "HIGH_VOL"
    if r.trend<p["trend"]:return "SIDEWAYS"
    if r.ema20>r.ema50 and r.ema50>r.ema200:return "UPTREND"
    if r.ema20<r.ema50 and r.ema50<r.ema200:return "DOWNTREND"
    return "MIXED"

def signal(r,name):
    p=PROFILES[name]; rg=regime(r,p)
    if rg in ("SIDEWAYS","HIGH_VOL","UNKNOWN","MIXED"): return "NO TRADE",0,rg
    if rg=="UPTREND" and r.close>r.vwap and r.close>r.ema9 and r.ret3>0 and r.ret8>0 and p["rsi_buy"][0]<=r.rsi<=p["rsi_buy"][1]:
        return "BUY",80,rg
    if rg=="DOWNTREND" and r.close<r.vwap and r.close<r.ema9 and r.ret3<0 and r.ret8<0 and p["rsi_sell"][0]<=r.rsi<=p["rsi_sell"][1]:
        return "SELL",80,rg
    return "NO TRADE",0,rg

def backtest(d,name,capital,risk,maxtrades,cost,slip):
    p=PROFILES[name]; cash=float(capital); pos=None; trades=[]; dayn={}; eq=[]; peak=cash; maxdd=0
    for _,r in d.iterrows():
        if pd.isna(r.atr) or pd.isna(r.ema200): eq.append((r.datetime,cash)); continue
        day=str(r.datetime)[:10]; dayn.setdefault(day,0)
        if pos:
            # ATR trailing stop after price moves in favour.
            if pos["side"]=="BUY":
                if r.close>=pos["entry"]+pos["risk"]: pos["sl"]=max(pos["sl"],pos["entry"])
                pos["sl"]=max(pos["sl"],float(r.close-r.atr*0.85))
                xp=pos["sl"] if r.low<=pos["sl"] else (pos["target"] if r.high>=pos["target"] else None)
            else:
                if r.close<=pos["entry"]-pos["risk"]: pos["sl"]=min(pos["sl"],pos["entry"])
                pos["sl"]=min(pos["sl"],float(r.close+r.atr*0.85))
                xp=pos["sl"] if r.high>=pos["sl"] else (pos["target"] if r.low<=pos["target"] else None)
            if xp is not None:
                xp=float(xp)*(1-slip/10000 if pos["side"]=="BUY" else 1+slip/10000)
                pnl=(xp-pos["entry"])*pos["qty"]*(1 if pos["side"]=="BUY" else -1)-cost
                cash+=pnl
                trades.append({**pos,"exit":xp,"pnl":pnl,"exit_time":r.datetime})
                pos=None
        s,score,rg=signal(r,name)
        if pos is None and s!="NO TRADE" and dayn[day]<maxtrades:
            dist=max(float(r.atr)*p["sl"],.01)
            qty=max(1,int((cash*risk/100)/dist))
            entry=float(r.close)*(1+slip/10000 if s=="BUY" else 1-slip/10000)
            pos={"side":s,"entry":entry,"sl":entry-dist if s=="BUY" else entry+dist,
                 "target":entry+dist*p["rr"] if s=="BUY" else entry-dist*p["rr"],
                 "qty":qty,"entry_time":r.datetime,"score":score,"risk":dist,"regime":rg}
            dayn[day]+=1
        mark=cash if pos is None else cash+(float(r.close)-pos["entry"])*pos["qty"]*(1 if pos["side"]=="BUY" else -1)
        peak=max(peak,mark); maxdd=max(maxdd,(peak-mark)/peak*100 if peak else 0)
        eq.append((r.datetime,mark))
    t=pd.DataFrame(trades)
    gp=t.loc[t.pnl>0,"pnl"].sum() if len(t) else 0; gl=-t.loc[t.pnl<0,"pnl"].sum() if len(t) else 0
    return t,pd.DataFrame(eq,columns=["datetime","equity"]),cash,{"P&L":cash-capital,"Trades":len(t),"Win Rate %":(t.pnl.gt(0).mean()*100 if len(t) else 0),"PF":gp/gl if gl else 0,"Expectancy":t.pnl.mean() if len(t) else 0,"Max DD %":maxdd}

st.title("🤖 ALS AI Algo Trading V10")
st.caption("Index-specific strategy profiles • regime filter • conservative entries • walk-forward validation • paper trading only")
with st.sidebar:
    interval=st.selectbox("Interval",["15m","1h"],index=0)
    period=st.selectbox("Period",["1mo","3mo","6mo","1y"],index=0)
    capital=st.number_input("Starting capital",100000,10000000,100000,10000)
    risk=st.slider("Risk / trade (%)",0.25,1.0,0.50,0.25)
    maxtrades=st.slider("Max trades/day",1,3,2)
    cost=st.number_input("Cost per completed trade (₹)",0,200,20,5)
    slip=st.number_input("Slippage (bps)",0,10,2,1)

st.info("V10 uses separate conservative profiles for each index. A strategy is not accepted merely because training results look good; validation statistics remain the decision gate.")
if st.button("🚀 Run V10 Walk-Forward Validation",type="primary"):
    out=[]; detail={}; prog=st.progress(0)
    for i,(name,ticker) in enumerate(SYMBOLS.items(),1):
        d=fetch(ticker,period,interval)
        if len(d)<180:
            out.append({"Symbol":name,"Status":"Not enough data","Validation P&L":np.nan});prog.progress(i/3);continue
        d=ind(d).dropna(subset=["ema200","atr","atr_med","rsi"])
        split=int(len(d)*.65); train=d.iloc[:split]; val=d.iloc[split:]
        tt,te,tc,tm=backtest(train,name,capital,risk,maxtrades,cost,slip)
        vt,ve,vc,vm=backtest(val,name,capital,risk,maxtrades,cost,slip)
        # Validation gate: positive PF and at least 5 trades required for PASS.
        gate=("PASS" if vm["Trades"]>=5 and vm["PF"]>1.05 and vm["Expectancy"]>0 and vm["Max DD %"]<4 else "REJECT")
        out.append({"Symbol":name,"Gate":gate,"Train P&L":tm["P&L"],"Train Trades":tm["Trades"],"Validation P&L":vm["P&L"],"Validation Trades":vm["Trades"],"Validation Win %":vm["Win Rate %"],"Validation PF":vm["PF"],"Validation Expectancy":vm["Expectancy"],"Max DD %":vm["Max DD %"]})
        detail[name]=(vt,ve,vm)
        prog.progress(i/3)
    res=pd.DataFrame(out); st.subheader("📊 V10 Walk-Forward Results"); st.dataframe(res,use_container_width=True)
    for name,(t,e,m) in detail.items():
        st.subheader(f"📈 {name} — {'PASS' if m['Trades']>=5 and m['PF']>1.05 and m['Expectancy']>0 and m['Max DD %']<4 else 'REJECT'}")
        a,b,c,d=st.columns(4);a.metric("Validation P&L",f"₹{m['P&L']:,.2f}");b.metric("Win Rate",f"{m['Win Rate %']:.1f}%");c.metric("Profit Factor",f"{m['PF']:.2f}");d.metric("Max DD",f"{m['Max DD %']:.2f}%")
        st.line_chart(e.set_index("datetime")[["equity"]]); st.dataframe(t,use_container_width=True)
        if len(t): st.download_button(f"Download {name} V10 trades",t.to_csv(index=False),f"{name.lower().replace(' ','_')}_v10_trades.csv","text/csv",key=f"v10_{name}")
st.divider(); st.caption("Research/paper-trading only. No profit guarantee. Live orders are disabled.")
