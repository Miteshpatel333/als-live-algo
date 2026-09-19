
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ALS AI Algo Trading V12", page_icon="🤖", layout="wide")

SYMBOLS={"NIFTY":"^NSEI","BANK NIFTY":"^NSEBANK","SENSEX":"^BSESN"}

def fetch(ticker, period, interval):
    import yfinance as yf
    x=yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                  progress=False, threads=False)
    if x is None or x.empty: return pd.DataFrame()
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    x=x.reset_index()
    x.columns=[str(c).lower().replace(" ","_") for c in x.columns]
    dt=next((c for c in ("datetime","date","timestamp") if c in x.columns),None)
    if not dt: return pd.DataFrame()
    x["datetime"]=pd.to_datetime(x[dt],errors="coerce")
    for c in ("open","high","low","close"):
        if c not in x: return pd.DataFrame()
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x["volume"]=pd.to_numeric(x.get("volume",0),errors="coerce").fillna(0)
    return x[["datetime","open","high","low","close","volume"]].dropna().drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)

def indicators(d):
    d=d.copy()
    for n in (9,20,50,200):
        d[f"ema{n}"]=d.close.ewm(span=n,adjust=False).mean()
    ch=d.close.diff()
    up=ch.clip(lower=0).rolling(14,min_periods=14).mean()
    dn=(-ch.clip(upper=0)).rolling(14,min_periods=14).mean()
    d["rsi"]=100-100/(1+up/dn.replace(0,np.nan))
    tr=pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),
                  (d.low-d.close.shift()).abs()],axis=1).max(axis=1)
    d["atr"]=tr.rolling(14,min_periods=14).mean()
    d["atr_med"]=d.atr.rolling(50,min_periods=20).median()
    d["vwap"]=((d.close*d.volume).cumsum()/d.volume.cumsum().replace(0,np.nan)) if d.volume.sum()>0 else d.close.expanding().mean()
    d["ret3"]=d.close.pct_change(3)
    d["ret8"]=d.close.pct_change(8)
    d["range_pct"]=(d.high-d.low)/d.close*100
    d["body_pct"]=(d.close-d.open).abs()/d.close*100
    d["vol_med"]=d.volume.rolling(20,min_periods=10).median()
    d["close_pos"]=(d.close-d.low)/(d.high-d.low).replace(0,np.nan)
    return d

# Four deliberately different rule families.
def family_signal(r, fam):
    trend_up=r.ema20>r.ema50 and r.ema50>r.ema200
    trend_dn=r.ema20<r.ema50 and r.ema50<r.ema200
    atr_ok=(not pd.isna(r.atr) and not pd.isna(r.atr_med) and r.atr<=r.atr_med*1.5)
    slope_up=r.ema20>r.ema20.shift if False else True
    # V12 regime/quality filters.
    if not atr_ok:
        return "NO TRADE",0,"HIGH_VOL"
    if abs(r.ema20-r.ema50)/r.close*100 < 0.06:
        return "NO TRADE",0,"SIDEWAYS"
    if pd.isna(r.rsi) or pd.isna(r.close_pos):
        return "NO TRADE",0,"NO_CONFIRM"
    bullish_bar=r.close>r.open and r.close_pos>=0.60
    bearish_bar=r.close<r.open and r.close_pos<=0.40
    vol_ok=(r.volume<=0 or pd.isna(r.vol_med) or r.volume>=r.vol_med*0.8)

    if fam=="TREND":
        if trend_up and r.close>r.vwap and r.close>r.ema9 and 53<=r.rsi<=65 and r.ret3>0 and r.ret8>0 and bullish_bar and vol_ok:
            return "BUY",85,"UPTREND"
        if trend_dn and r.close<r.vwap and r.close<r.ema9 and 35<=r.rsi<=47 and r.ret3<0 and r.ret8<0 and bearish_bar and vol_ok:
            return "SELL",85,"DOWNTREND"

    elif fam=="PULLBACK":
        if trend_up and r.close>r.ema50 and r.close>r.vwap and 46<=r.rsi<=57 and bullish_bar and vol_ok:
            return "BUY",80,"UPTREND"
        if trend_dn and r.close<r.ema50 and r.close<r.vwap and 43<=r.rsi<=54 and bearish_bar and vol_ok:
            return "SELL",80,"DOWNTREND"

    elif fam=="MOMENTUM":
        if trend_up and r.ret3>0.0015 and r.ret8>0.0018 and r.rsi>56 and r.close>r.vwap and bullish_bar and vol_ok:
            return "BUY",88,"UPTREND"
        if trend_dn and r.ret3<-0.0015 and r.ret8<-0.0018 and r.rsi<44 and r.close<r.vwap and bearish_bar and vol_ok:
            return "SELL",88,"DOWNTREND"

    elif fam=="BREAKOUT":
        if trend_up and r.close>r.high_10 and r.rsi>56 and bullish_bar and vol_ok:
            return "BUY",88,"UPTREND"
        if trend_dn and r.close<r.low_10 and r.rsi<44 and bearish_bar and vol_ok:
            return "SELL",88,"DOWNTREND"

    return "NO TRADE",0,"MIXED"

def add_breakout_cols(d):
    d=d.copy()
    d["high_10"]=d.high.shift(1).rolling(10,min_periods=10).max()
    d["low_10"]=d.low.shift(1).rolling(10,min_periods=10).min()
    return d

def backtest(d, fam, capital, risk_pct, maxtrades, cost, slip_bps):
    cash=float(capital); pos=None; trades=[]; dayn={}; eq=[]; peak=cash; maxdd=0
    for _,r in d.iterrows():
        if pd.isna(r.atr) or pd.isna(r.ema200): eq.append((r.datetime,cash)); continue
        day=str(r.datetime)[:10]; dayn.setdefault(day,0)

        if pos:
            if pos["side"]=="BUY":
                if r.close>=pos["entry"]+pos["risk"]: pos["sl"]=max(pos["sl"],pos["entry"])
                pos["sl"]=max(pos["sl"],float(r.close-r.atr*0.75))
                xp=pos["sl"] if r.low<=pos["sl"] else (pos["target"] if r.high>=pos["target"] else None)
            else:
                if r.close<=pos["entry"]-pos["risk"]: pos["sl"]=min(pos["sl"],pos["entry"])
                pos["sl"]=min(pos["sl"],float(r.close+r.atr*0.75))
                xp=pos["sl"] if r.high>=pos["sl"] else (pos["target"] if r.low<=pos["target"] else None)

            if xp is not None:
                xp=float(xp)*(1-slip_bps/10000 if pos["side"]=="BUY" else 1+slip_bps/10000)
                pnl=(xp-pos["entry"])*pos["qty"]*(1 if pos["side"]=="BUY" else -1)-cost
                cash+=pnl
                trades.append({**pos,"exit":xp,"pnl":pnl,"exit_time":r.datetime})
                pos=None

        s,score,reg=family_signal(r,fam)
        if pos is None and s!="NO TRADE" and dayn[day]<maxtrades:
            dist=max(float(r.atr)*1.35,0.01)
            qty=max(1,int((cash*risk_pct/100)/dist))
            entry=float(r.close)*(1+slip_bps/10000 if s=="BUY" else 1-slip_bps/10000)
            rr=2.2
            pos={"side":s,"entry":entry,"sl":entry-dist if s=="BUY" else entry+dist,
                 "target":entry+dist*rr if s=="BUY" else entry-dist*rr,
                 "qty":qty,"entry_time":r.datetime,"confidence":score,
                 "risk":dist,"regime":reg}
            dayn[day]+=1

        mark=cash if pos is None else cash+(float(r.close)-pos["entry"])*pos["qty"]*(1 if pos["side"]=="BUY" else -1)
        peak=max(peak,mark)
        maxdd=max(maxdd,(peak-mark)/peak*100 if peak else 0)
        eq.append((r.datetime,mark))

    t=pd.DataFrame(trades)
    gp=t.loc[t.pnl>0,"pnl"].sum() if len(t) else 0
    gl=-t.loc[t.pnl<0,"pnl"].sum() if len(t) else 0
    stats={"P&L":cash-capital,"Trades":len(t),
           "Win Rate %":t.pnl.gt(0).mean()*100 if len(t) else 0,
           "PF":gp/gl if gl else 0,
           "Expectancy":t.pnl.mean() if len(t) else 0,
           "Max DD %":maxdd}
    return t,pd.DataFrame(eq,columns=["datetime","equity"]),stats

st.title("🤖 ALS AI Algo Trading V12")
st.caption("Strategy research engine • Multi-window walk-forward • Confirmation + regime filters • Paper trading only")

with st.sidebar:
    interval=st.selectbox("Interval",["15m","1h"],index=0)
    period=st.selectbox("Period",["1mo","3mo","6mo","1y"],index=0)
    capital=st.number_input("Starting capital",100000,10000000,100000,10000)
    risk=st.slider("Risk / trade (%)",0.25,1.0,0.50,0.25)
    maxtrades=st.slider("Max trades/day",1,3,2)
    cost=st.number_input("Cost per completed trade (₹)",0,200,20,5)
    slip=st.number_input("Slippage (bps)",0,10,2,1)

st.info("V12 compares four rule families and tests the selected family on a later unseen segment. No strategy is declared robust from a single profitable period.")

if st.button("🚀 Run V12 Strategy Research",type="primary"):
    all_rows=[]; chosen={}; progress=st.progress(0)
    families=["TREND","PULLBACK","MOMENTUM","BREAKOUT"]

    for i,(name,ticker) in enumerate(SYMBOLS.items(),1):
        d=fetch(ticker,period,interval)
        if len(d)<220:
            all_rows.append({"Symbol":name,"Status":"Not enough data"}); progress.progress(i/3); continue
        d=add_breakout_cols(indicators(d)).dropna(subset=["ema200","atr","atr_med","rsi","high_10","low_10"])
        n=len(d)
        # Three chronological windows; the last window is never used for family selection.
        windows=[
            (0.45,0.65,0.65,0.80),
            (0.10,0.50,0.50,0.70),
            (0.25,0.60,0.60,0.78),
        ]

        family_rows=[]
        for fam in families:
            scores=[]
            for w,(ts,te,vs,ve) in enumerate(windows,1):
                train=d.iloc[int(n*ts):int(n*te)]
                val=d.iloc[int(n*vs):int(n*ve)]
                _,_,tr=backtest(train,fam,capital,risk,maxtrades,cost,slip_bps=slip)
                _,_,va=backtest(val,fam,capital,risk,maxtrades,cost,slip_bps=slip)
                scores.append(va)
            pfs=[x["PF"] for x in scores]
            exps=[x["Expectancy"] for x in scores]
            dds=[x["Max DD %"] for x in scores]
            trades=[x["Trades"] for x in scores]
            positive_windows=sum(1 for x in scores if x["Expectancy"]>0 and x["PF"]>1.0)
            family_rows.append({
                "Family":fam,
                "Windows Positive":positive_windows,
                "Median PF":float(np.median(pfs)),
                "Median Expectancy":float(np.median(exps)),
                "Worst DD %":float(max(dds)),
                "Min Trades":int(min(trades)),
                "Eligible":positive_windows>=2 and np.median(pfs)>1.02 and np.median(exps)>0 and max(dds)<5
            })

        fs=pd.DataFrame(family_rows)
        eligible=fs[fs["Eligible"]]
        selected=(eligible.sort_values(["Median PF","Median Expectancy"],ascending=False).iloc[0]["Family"]
                  if len(eligible) else fs.sort_values(["Median PF","Median Expectancy"],ascending=False).iloc[0]["Family"])

        # Final unseen segment is strictly after all selection windows.
        unseen=d.iloc[int(n*.80):]
        ut,ue,us=backtest(unseen,selected,capital,risk,maxtrades,cost,slip_bps=slip)

        robust=(us["Trades"]>=3 and us["PF"]>1.0 and us["Expectancy"]>0 and us["Max DD %"]<5)
        status="PASS" if robust else "REJECT"
        all_rows.append({"Symbol":name,"Selected":selected,"Status":status,
                         "Unseen P&L":us["P&L"],"Unseen Trades":us["Trades"],
                         "Unseen Win %":us["Win Rate %"],"Unseen PF":us["PF"],
                         "Unseen Expectancy":us["Expectancy"],"Unseen Max DD %":us["Max DD %"]})
        chosen[name]=(selected,fs,ut,ue,us)
        progress.progress(i/3)

    result=pd.DataFrame(all_rows)
    st.subheader("📊 V12 Robustness Results")
    st.dataframe(result,use_container_width=True)

    for name,(selected,fs,t,e,s) in chosen.items():
        st.subheader(f"📈 {name} — {selected}")
        st.dataframe(fs[["Family","Windows Positive","Median PF","Median Expectancy","Worst DD %","Min Trades","Eligible"]],use_container_width=True)
        a,b,c,d=st.columns(4)
        a.metric("Unseen P&L",f"₹{s['P&L']:,.2f}")
        b.metric("Unseen Win Rate",f"{s['Win Rate %']:.1f}%")
        c.metric("Unseen PF",f"{s['PF']:.2f}")
        d.metric("Unseen Max DD",f"{s['Max DD %']:.2f}%")
        st.line_chart(e.set_index("datetime")[["equity"]])
        st.dataframe(t,use_container_width=True)
        if len(t):
            st.download_button(f"Download {name} V12 unseen trades",
                t.to_csv(index=False),f"{name.lower().replace(' ','_')}_v12_unseen.csv","text/csv",key=f"v11_{name}")

st.divider()
st.caption("Research/paper-trading only. No profit guarantee. Live orders are disabled.")
