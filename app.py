
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ALS AI Algo Trading V14.3", page_icon="🤖", layout="wide")

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
    # NSE exports can contain both Close and LTP. Both normalize to "close";
    # collapse duplicate column names safely before numeric conversion.
    if x.columns.duplicated().any():
        merged=pd.DataFrame(index=x.index)
        for name in pd.unique(x.columns):
            same=x.loc[:, x.columns==name]
            merged[name]=same.bfill(axis=1).iloc[:,0] if same.shape[1]>1 else same.iloc[:,0]
        x=merged
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

st.title("🤖 ALS AI Algo Trading V14.3")
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
    st.info("V14.3 keeps the V12 robustness gate. A PASS requires positive out-of-sample expectancy, PF > 1, at least 3 unseen trades, and max DD < 5%.")
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
    st.warning("Option Strategy Lab is research/paper testing only. V14.3 adds paired CE+PE analysis for an actual historical short-straddle mark-to-market. Live orders remain disabled.")

    st.markdown("### Strategy Library")
    strategy=st.selectbox("Select strategy",[
        "Mukul — Short Straddle",
        "Mukul — Bull Call Spread",
        "Pushkar — Option Chain",
        "Pushkar — Indicator Scalping"
    ])
    desc={
        "Mukul — Short Straddle":"Paired ATM CE + ATM PE short template. V14.3 can calculate a paired daily mark-to-market when both CE and PE contract files are uploaded.",
        "Mukul — Bull Call Spread":"Buy ATM CE + sell a higher-strike CE. Defined-risk debit spread template.",
        "Pushkar — Option Chain":"Uses CE/PE OI and volume fields as a transparent support/resistance proxy; not a claim of exact video-rule reproduction.",
        "Pushkar — Indicator Scalping":"Directional option-buying template; underlying indicator confirmation is required before treating a signal as executable."
    }
    st.info(desc[strategy])

    st.markdown("### 1) Historical option data")
    st.write("Upload the NSE contract-wise CSV files. For the Short Straddle module, upload both the matching CE and PE files for the same symbol, strike and expiry.")
    ce_file=st.file_uploader("Upload NSE CE CSV",type=["csv"],key="ce_v143")
    pe_file=st.file_uploader("Upload matching NSE PE CSV",type=["csv"],key="pe_v143")

    def normalize_nse(df):
        x=df.copy()
        x.columns=[str(c).strip().lower().replace(" ","_") for c in x.columns]
        # Remove duplicate column names safely (NSE can expose both Close/LTP variants).
        x=x.loc[:,~x.columns.duplicated()].copy()
        aliases={
            "date":"datetime","expiry":"expiry","option_type":"option_type",
            "strike_price":"strike","underlying_value":"spot",
            "open_int":"oi","settle_price":"settle_price",
            "close":"close","ltp":"ltp","open":"open","high":"high","low":"low",
            "symbol":"symbol"
        }
        x=x.rename(columns={c:aliases.get(c,c) for c in x.columns})
        required=["datetime","expiry","strike","option_type","close","settle_price"]
        missing=[c for c in required if c not in x.columns]
        if missing: return pd.DataFrame(),missing
        x["datetime"]=pd.to_datetime(x["datetime"],errors="coerce")
        x["expiry"]=pd.to_datetime(x["expiry"],errors="coerce")
        x["strike"]=pd.to_numeric(x["strike"],errors="coerce")
        x["option_type"]=x["option_type"].astype(str).str.upper().str.strip().replace({"CALL":"CE","PUT":"PE"})
        for c in ["open","high","low","close","ltp","settle_price","volume","oi","spot"]:
            if c not in x: x[c]=np.nan
            x[c]=pd.to_numeric(x[c].replace("-",np.nan),errors="coerce")
        x=x.dropna(subset=["datetime","expiry","strike","option_type"])
        x=x[x.option_type.isin(["CE","PE"])].sort_values("datetime").reset_index(drop=True)
        # For NSE rows with no traded OHLC, use settlement price as the daily mark.
        x["mark"]=x["close"].where(x["close"].notna() & (x["close"]!=1089.75),x["settle_price"])
        return x,[]

    ce=None; pe=None
    if ce_file:
        raw=pd.read_csv(ce_file); ce,miss=normalize_nse(raw)
        if miss: st.error("CE file missing: "+", ".join(miss))
        else: st.success(f"CE loaded: {len(ce):,} rows")
    if pe_file:
        raw=pd.read_csv(pe_file); pe,miss=normalize_nse(raw)
        if miss: st.error("PE file missing: "+", ".join(miss))
        else: st.success(f"PE loaded: {len(pe):,} rows")

    if ce is not None and pe is not None and len(ce) and len(pe):
        st.markdown("### 2) Contract match")
        ce_key=ce.iloc[0]; pe_key=pe.iloc[0]
        same_symbol=str(ce_key.get("symbol",""))==str(pe_key.get("symbol",""))
        same_strike=float(ce_key["strike"])==float(pe_key["strike"])
        same_expiry=pd.Timestamp(ce_key["expiry"])==pd.Timestamp(pe_key["expiry"])
        c1,c2,c3,c4=st.columns(4)
        c1.metric("CE rows",len(ce)); c2.metric("PE rows",len(pe))
        c3.metric("Strike",f"{ce_key['strike']:.0f}"); c4.metric("Expiry",pd.Timestamp(ce_key["expiry"]).strftime("%d-%b-%Y"))
        if not (same_symbol and same_strike and same_expiry):
            st.error("CE and PE files do not match on symbol/strike/expiry.")
        else:
            st.success("CE + PE contract match confirmed.")
            merged=ce[["datetime","expiry","strike","spot","mark","close","settle_price","oi"]].rename(
                columns={"mark":"ce_mark","close":"ce_close","settle_price":"ce_settle","oi":"ce_oi"}
            ).merge(
                pe[["datetime","expiry","strike","spot","mark","close","settle_price","oi"]].rename(
                    columns={"mark":"pe_mark","close":"pe_close","settle_price":"pe_settle","oi":"pe_oi"}
                ),
                on=["datetime","expiry","strike"],how="inner"
            ).sort_values("datetime").reset_index(drop=True)
            merged["spot"]=merged["spot_x"].combine_first(merged["spot_y"])
            merged["straddle_mark"]=merged.ce_mark+merged.pe_mark
            merged["straddle_close"]=merged.ce_close+merged.pe_close
            st.markdown("### 3) Short Straddle backtest")
            c1,c2,c3=st.columns(3)
            capital=st.number_input("Research capital (₹)",10000,10000000,100000,10000,key="ss_cap")
            lot_size=st.number_input("NIFTY lot size",1,1000,65,1,key="ss_lot")
            cost=st.number_input("Brokerage + charges per round trip (₹)",0,5000,100,10,key="ss_cost")
            slip=st.number_input("Slippage per leg (₹)",0.0,20.0,1.0,0.5,key="ss_slip")
            st.caption("Daily mark uses traded Close where available; if NSE provides no traded OHLC and the file contains the placeholder 1089.75, V14.3 falls back to Settlement Price. This avoids treating the placeholder as a real market quote.")
            if st.button("🧪 Run Short Straddle Backtest",type="primary",key="run_ss"):
                entry_date=pd.Timestamp(merged.datetime.min())
                exit_date=pd.Timestamp(merged.datetime.max())
                entry=float(merged.iloc[0].straddle_mark)
                exitv=float(merged.iloc[-1].straddle_mark)
                gross=(entry-exitv)*lot_size
                net=gross-cost-2*slip*lot_size
                peak=entry
                max_loss=0.0
                curve=[]
                for _,r in merged.iterrows():
                    pnl=(entry-float(r.straddle_mark))*lot_size
                    curve.append({"datetime":r.datetime,"pnl":pnl})
                    adverse=(float(r.straddle_mark)-entry)*lot_size
                    max_loss=max(max_loss,adverse)
                curve=pd.DataFrame(curve)
                st.subheader("📊 Short Straddle Result")
                a,b,c,d=st.columns(4)
                a.metric("Entry premium",f"₹{entry:,.2f}")
                b.metric("Exit mark",f"₹{exitv:,.2f}")
                c.metric("Gross P&L",f"₹{gross:,.2f}")
                d.metric("Net P&L",f"₹{net:,.2f}")
                st.write(f"Entry: **{entry_date.strftime('%d-%b-%Y')}** → Exit: **{exit_date.strftime('%d-%b-%Y')}** | Lot size: **{lot_size}**")
                st.write(f"Approx. worst mark-to-market loss during the supplied period: **₹{max_loss:,.2f}** before charges/slippage.")
                if exit_date < pd.Timestamp(merged.iloc[0].expiry):
                    st.warning("This contract had not reached expiry within the uploaded data. This is a mark-to-market backtest, not an expiry-settlement result.")
                else:
                    st.success("The uploaded period reaches the contract expiry.")
                st.dataframe(merged[["datetime","spot","ce_mark","pe_mark","straddle_mark","ce_oi","pe_oi"]],use_container_width=True)
                st.line_chart(curve.set_index("datetime")[["pnl"]])
                st.download_button("⬇️ Download straddle backtest",merged.to_csv(index=False),
                                   "nifty_short_straddle_v14_3.csv","text/csv")

    st.markdown("### 4) Research validation")
    st.write("For a stronger study, repeat this with multiple completed expiries and compare chronological in-sample, validation and unseen results. Do not treat a single expiry as evidence of profitability.")

st.divider()
st.caption("Research/paper-trading only. No profit guarantee. Live orders are disabled.")
