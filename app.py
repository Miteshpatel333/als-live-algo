
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ALS AI Algo Trading V14.1", page_icon="🤖", layout="wide")

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

def add_breakout_cols(d):
    d=d.copy()
    d["high_10"]=d.high.shift(1).rolling(10,min_periods=10).max()
    d["low_10"]=d.low.shift(1).rolling(10,min_periods=10).min()
    return d

def family_signal(r, fam):
    trend_up=r.ema20>r.ema50 and r.ema50>r.ema200
    trend_dn=r.ema20<r.ema50 and r.ema50<r.ema200
    atr_ok=(not pd.isna(r.atr) and not pd.isna(r.atr_med) and r.atr<=r.atr_med*1.5)
    if not atr_ok: return "NO TRADE",0,"HIGH_VOL"
    if abs(r.ema20-r.ema50)/r.close*100 < 0.06: return "NO TRADE",0,"SIDEWAYS"
    if pd.isna(r.rsi) or pd.isna(r.close_pos): return "NO TRADE",0,"NO_CONFIRM"
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
        peak=max(peak,mark); maxdd=max(maxdd,(peak-mark)/peak*100 if peak else 0)
        eq.append((r.datetime,mark))
    t=pd.DataFrame(trades)
    gp=t.loc[t.pnl>0,"pnl"].sum() if len(t) else 0
    gl=-t.loc[t.pnl<0,"pnl"].sum() if len(t) else 0
    stats={"P&L":cash-capital,"Trades":len(t),"Win Rate %":t.pnl.gt(0).mean()*100 if len(t) else 0,
           "PF":gp/gl if gl else 0,"Expectancy":t.pnl.mean() if len(t) else 0,"Max DD %":maxdd}
    return t,pd.DataFrame(eq,columns=["datetime","equity"]),stats

def option_template():
    return pd.DataFrame(columns=[
        "datetime","expiry","strike","option_type","open","high","low","close","volume","oi"
    ])

def normalize_option_df(df):
    x=df.copy()
    x.columns=[str(c).strip().lower().replace(" ","_") for c in x.columns]
    aliases={
        "date":"datetime","timestamp":"datetime","time":"datetime",
        "option":"option_type","type":"option_type","cp":"option_type",
        "ltp":"close","last_price":"close","open_interest":"oi",
        "strike_price":"strike","open_int":"oi","underlying_value":"spot",
        "optiontype":"option_type"
    }
    x=x.rename(columns={c:aliases.get(c,c) for c in x.columns})
    required=["datetime","expiry","strike","option_type","close"]
    missing=[c for c in required if c not in x.columns]
    if missing: return pd.DataFrame(), missing
    x["datetime"]=pd.to_datetime(x["datetime"],errors="coerce")
    x["expiry"]=pd.to_datetime(x["expiry"],errors="coerce")
    x["strike"]=pd.to_numeric(x["strike"],errors="coerce")
    x["option_type"]=x["option_type"].astype(str).str.upper().str.strip().replace({"CALL":"CE","PUT":"PE"})
    for c in ["open","high","low","close","volume","oi"]:
        if c not in x: x[c]=np.nan
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=["datetime","expiry","strike","option_type","close"])
    x=x[x.option_type.isin(["CE","PE"])].sort_values(["datetime","expiry","strike","option_type"]).reset_index(drop=True)
    return x, []

def nearest_atm_strike(strikes, spot):
    s=np.asarray(sorted(pd.Series(strikes).dropna().unique()))
    return float(s[np.argmin(np.abs(s-spot))]) if len(s) else np.nan

def option_strategy_signal(chain_at_t, strategy, spot, strike_step, short_straddle_rsi=50):
    # These are transparent research templates, not claimed verbatim reproductions of the videos.
    strikes=chain_at_t["strike"].dropna().unique()
    atm=nearest_atm_strike(strikes,spot)
    if pd.isna(atm): return None
    if strategy=="Mukul — Short Straddle":
        return {"structure":"SELL","legs":[("CE",atm,-1),("PE",atm,-1)],"reason":"ATM straddle template"}
    if strategy=="Mukul — Bull Call Spread":
        long_k=atm
        higher=[k for k in strikes if k>atm]
        short_k=min(higher) if len(higher) else atm+strike_step
        return {"structure":"DEBIT_SPREAD","legs":[("CE",long_k,1),("CE",short_k,-1)],"reason":"ATM call + higher-strike call"}
    if strategy=="Pushkar — Option Chain":
        # OI/volume support-resistance proxy when those fields exist.
        g=chain_at_t.groupby(["strike","option_type"],as_index=False).agg(
            oi=("oi","max"),volume=("volume","sum"),close=("close","last"))
        ce=g[g.option_type=="CE"].sort_values(["oi","volume"],ascending=False)
        pe=g[g.option_type=="PE"].sort_values(["oi","volume"],ascending=False)
        call_res=float(ce.iloc[0].strike) if len(ce) and pd.notna(ce.iloc[0].oi) else atm
        put_sup=float(pe.iloc[0].strike) if len(pe) and pd.notna(pe.iloc[0].oi) else atm
        if spot>=call_res: return {"structure":"BUY","legs":[("CE",atm,1)],"reason":"call-OI resistance break proxy"}
        if spot<=put_sup: return {"structure":"BUY","legs":[("PE",atm,-1)],"reason":"put-OI support test proxy"}
        return None
    if strategy=="Pushkar — Indicator Scalping":
        return {"structure":"BUY","legs":[("CE",atm,1)],"reason":"directional call template; underlying filter required"}
    return None

def get_price(chain, opt_type, strike):
    z=chain[(chain.option_type==opt_type)&(chain.strike==strike)].sort_values("datetime")
    if len(z)==0: return np.nan
    return float(z.iloc[-1].close)

st.title("🤖 ALS AI Algo Trading V14.1")
st.caption("Index research + Option Strategy Lab • backtest/paper research only • live orders disabled")

tab1,tab2=st.tabs(["📊 Index Research","🧩 Option Strategy Lab"])

with tab1:
    with st.sidebar:
        st.header("Index Research")
        interval=st.selectbox("Interval",["15m","1h"],index=0)
        period=st.selectbox("Period",["1mo","3mo","6mo","1y"],index=0)
        capital=st.number_input("Starting capital",100000,10000000,100000,10000)
        risk=st.slider("Risk / trade (%)",0.25,1.0,0.50,0.25)
        maxtrades=st.slider("Max trades/day",1,3,2)
        cost=st.number_input("Cost per completed trade (₹)",0,200,20,5)
        slip=st.number_input("Slippage (bps)",0,10,2,1)
    st.info("V14.1 keeps the V12 robustness gate. A PASS requires positive out-of-sample expectancy, PF > 1, at least 3 unseen trades, and max DD < 5%.")
    if st.button("🚀 Run V14 Index Research",type="primary"):
        all_rows=[]; chosen={}; progress=st.progress(0); families=["TREND","PULLBACK","MOMENTUM","BREAKOUT"]
        for i,(name,ticker) in enumerate(SYMBOLS.items(),1):
            d=fetch(ticker,period,interval)
            if len(d)<220:
                all_rows.append({"Symbol":name,"Status":"Not enough data"}); progress.progress(i/3); continue
            d=add_breakout_cols(indicators(d)).dropna(subset=["ema200","atr","atr_med","rsi","high_10","low_10"])
            n=len(d); windows=[(0.45,0.65,0.65,0.80),(0.10,0.50,0.50,0.70),(0.25,0.60,0.60,0.78)]
            family_rows=[]
            for fam in families:
                scores=[]
                for ts,te,vs,ve in windows:
                    _,_,_=backtest(d.iloc[int(n*ts):int(n*te)],fam,capital,risk,maxtrades,cost,slip)
                    _,_,va=backtest(d.iloc[int(n*vs):int(n*ve)],fam,capital,risk,maxtrades,cost,slip)
                    scores.append(va)
                pfs=[x["PF"] for x in scores]; exps=[x["Expectancy"] for x in scores]; dds=[x["Max DD %"] for x in scores]; trades=[x["Trades"] for x in scores]
                pos=sum(1 for x in scores if x["Expectancy"]>0 and x["PF"]>1)
                family_rows.append({"Family":fam,"Windows Positive":pos,"Median PF":float(np.median(pfs)),
                                    "Median Expectancy":float(np.median(exps)),"Worst DD %":float(max(dds)),
                                    "Min Trades":int(min(trades)),"Eligible":pos>=2 and np.median(pfs)>1.02 and np.median(exps)>0 and max(dds)<5})
            fs=pd.DataFrame(family_rows); eligible=fs[fs.Eligible]
            selected=(eligible.sort_values(["Median PF","Median Expectancy"],ascending=False).iloc[0].Family
                      if len(eligible) else fs.sort_values(["Median PF","Median Expectancy"],ascending=False).iloc[0].Family)
            _,ue,us=backtest(d.iloc[int(n*.80):],selected,capital,risk,maxtrades,cost,slip)
            status="PASS" if us["Trades"]>=3 and us["PF"]>1 and us["Expectancy"]>0 and us["Max DD %"]<5 else "REJECT"
            all_rows.append({"Symbol":name,"Selected":selected,"Status":status,"Unseen P&L":us["P&L"],"Unseen Trades":us["Trades"],
                             "Unseen Win %":us["Win Rate %"],"Unseen PF":us["PF"],"Unseen Expectancy":us["Expectancy"],"Unseen Max DD %":us["Max DD %"]})
            chosen[name]=(selected,fs,ue,us); progress.progress(i/3)
        st.subheader("📊 V14 Robustness Results")
        st.dataframe(pd.DataFrame(all_rows),use_container_width=True)
        for name,(selected,fs,e,s) in chosen.items():
            st.subheader(f"📈 {name} — {selected}")
            st.dataframe(fs,use_container_width=True)
            a,b,c,d=st.columns(4)
            a.metric("Unseen P&L",f"₹{s['P&L']:,.2f}"); b.metric("Win Rate",f"{s['Win Rate %']:.1f}%")
            c.metric("PF",f"{s['PF']:.2f}"); d.metric("Max DD",f"{s['Max DD %']:.2f}%")
            if len(e): st.line_chart(e.set_index("datetime")[["equity"]])

with tab2:
    st.warning("Option Strategy Lab is research/paper testing only. V14.1 adds an NSE contract-data import workflow. The four modules are transparent research templates inspired by the named educational topics; they are NOT claimed to reproduce every rule from the videos verbatim.")
    st.markdown("### Strategy Library")
    strategy=st.selectbox("Select strategy",[
        "Mukul — Short Straddle",
        "Mukul — Bull Call Spread",
        "Pushkar — Option Chain",
        "Pushkar — Indicator Scalping"
    ])
    desc={
        "Mukul — Short Straddle":"ATM CE + ATM PE short template. Requires real historical option premiums, expiry and costs to backtest.",
        "Mukul — Bull Call Spread":"Buy ATM CE + sell a higher-strike CE. Defined-risk debit spread template.",
        "Pushkar — Option Chain":"Uses CE/PE OI and volume fields as a transparent support/resistance proxy; not a claim of exact video-rule reproduction.",
        "Pushkar — Indicator Scalping":"Directional option-buying template; underlying indicator confirmation is required before treating a signal as executable."
    }
    st.info(desc[strategy])

    st.markdown("### 1) Historical option data")
    st.markdown("### NSE historical contract data")
    st.write("NSE provides a Historical Contract-wise Price Volume Data report with filters for Instrument, Symbol, Year, Expiry, Option Type and Strike Price, plus CSV download. V14 is designed around that contract-level data. citeturn0search1")
    st.markdown("**NSE workflow:** Historical Contract-wise Price Volume Data → Instrument: Options → Symbol: NIFTY → choose Year/Expiry/Option Type/Strike → download CSV. The app then normalizes the downloaded file.")
    st.markdown("NSE also exposes current option-chain CSV downloads, but current-chain data is not a substitute for historical contract prices when doing a backtest. citeturn0search3")
    st.download_button("⬇️ Download normalized CSV template",option_template().to_csv(index=False),"option_data_template_v14_1.csv","text/csv")
    uploaded=st.file_uploader("Upload NSE contract-wise CSV (or normalized CSV)",type=["csv"])
    if uploaded:
        raw=pd.read_csv(uploaded)
        opt,missing=normalize_option_df(raw)
        if missing:
            st.error("Missing required columns: "+", ".join(missing))
        elif opt.empty:
            st.error("No usable CE/PE rows found.")
        else:
            st.success(f"Loaded {len(opt):,} option rows. V14 will use actual contract premium/close, expiry, strike and CE/PE fields for research.")
            c1,c2,c3=st.columns(3)
            c1.metric("Rows",f"{len(opt):,}")
            c2.metric("Dates",f"{opt.datetime.dt.date.nunique():,}")
            c3.metric("Expiries",f"{opt.expiry.dt.date.nunique():,}")
            st.dataframe(opt.head(100),use_container_width=True)

            st.markdown("### 2) Research controls")
            strike_step=st.number_input("Strike step",1,1000,50,1)
            lot_size=st.number_input("Lot size",1,1000,65,1)
            capital_opt=st.number_input("Option research capital (₹)",10000,10000000,100000,10000)
            max_loss=st.slider("Max loss budget / trade (%)",0.25,5.0,1.0,0.25)
            entry_time=st.time_input("Paper entry time")
            st.caption("For exact historical testing, the uploaded file must contain the actual contract premium at each timestamp. Underlying-only data cannot produce a genuine option P&L.")

            if st.button("🧪 Generate Option Strategy Signals",type="primary"):
                rows=[]
                for dt,grp in opt.groupby("datetime"):
                    spot=np.nan
                    # If an underlying spot column exists, use it; otherwise ATM is derived from strikes.
                    if "spot" in grp.columns:
                        spot=pd.to_numeric(grp["spot"],errors="coerce").dropna()
                        spot=float(spot.iloc[-1]) if len(spot) else np.nan
                    if pd.isna(spot): spot=float(grp["strike"].median())
                    sig=option_strategy_signal(grp,strategy,spot,strike_step)
                    if sig:
                        legs=[]
                        valid=True
                        for typ,k,qty in sig["legs"]:
                            px=get_price(grp,typ,k)
                            if pd.isna(px): valid=False
                            legs.append((typ,k,qty,px))
                        if valid:
                            debit_credit=sum(q*px for _,_,q,px in legs)
                            rows.append({"datetime":dt,"spot_proxy":spot,"structure":sig["structure"],
                                         "reason":sig["reason"],"legs":str(legs),
                                         "net_premium_per_unit":debit_credit,
                                         "gross_premium_x_lot":debit_credit*lot_size})
                out=pd.DataFrame(rows)
                if out.empty:
                    st.warning("No complete signals found in the uploaded data.")
                else:
                    st.success(f"Generated {len(out):,} research signals.")
                    st.dataframe(out,use_container_width=True)
                    st.download_button("⬇️ Download option signals",out.to_csv(index=False),
                                       "option_strategy_signals_v14_1.csv","text/csv")
                    st.caption("Signals are research outputs only. This build does not place broker orders.")

            st.markdown("### 3) NSE data quality checks")
            dup=int(opt.duplicated(["datetime","expiry","strike","option_type"]).sum())
            bad_exp=int((opt.expiry<opt.datetime.dt.normalize()).sum())
            q1,q2,q3,q4=st.columns(4)
            q1.metric("Duplicate contract rows",dup)
            q2.metric("Expiry before timestamp",bad_exp)
            q3.metric("CE rows",int((opt.option_type=="CE").sum()))
            q4.metric("PE rows",int((opt.option_type=="PE").sum()))
            if dup or bad_exp:
                st.warning("Data quality issues detected. Clean/verify the NSE export before treating results as valid.")
            
            st.markdown("### 4) Rule extraction / validation")
            st.write("Before calling any module a faithful implementation, compare its exact entry, strike-selection, exit and risk rules with the source video. The app deliberately labels the current four as research templates.")
            st.write("Next stage: run chronological in-sample → validation → unseen tests on the imported NSE contract history, with brokerage, slippage, max-loss and expiry-aware position handling.")

st.divider()
st.caption("Research/paper-trading only. No profit guarantee. Live orders are disabled.")
