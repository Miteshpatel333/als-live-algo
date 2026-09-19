import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='ALS AI Algo', page_icon='📈', layout='wide')

def indicators(df):
    d=df.copy(); d.columns=[str(c).strip().lower().replace(' ','_') for c in d.columns]
    ren={}
    for c in d.columns:
        if c in ('date','datetime','timestamp','time'): ren[c]='datetime'
    d=d.rename(columns=ren)
    if not {'open','high','low','close'}.issubset(d.columns): raise ValueError('CSV must contain Open, High, Low, Close columns.')
    d['datetime']=pd.to_datetime(d['datetime'],errors='coerce') if 'datetime' in d else range(len(d))
    d=d.sort_values('datetime').reset_index(drop=True)
    d['ema20']=d.close.ewm(span=20,adjust=False).mean(); d['ema50']=d.close.ewm(span=50,adjust=False).mean()
    delta=d.close.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
    d['rsi']=100-(100/(1+gain/loss.replace(0,np.nan)))
    tr=pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()],axis=1).max(axis=1); d['atr']=tr.rolling(14).mean()
    vol=d['volume'] if 'volume' in d else pd.Series(1,index=d.index); d['vwap']=(d.close*vol).cumsum()/vol.cumsum()
    hl2=(d.high+d.low)/2; upper=hl2+3*d.atr; lower=hl2-3*d.atr; trend=np.ones(len(d))
    for i in range(1,len(d)):
        if pd.isna(d.atr.iloc[i]): trend[i]=trend[i-1]
        elif d.close.iloc[i]>upper.iloc[i-1]: trend[i]=1
        elif d.close.iloc[i]<lower.iloc[i-1]: trend[i]=-1
        else: trend[i]=trend[i-1]
    d['supertrend_dir']=trend; return d

def score(r):
    buy=(25 if r.ema20>r.ema50 else 0)+(20 if r.rsi>=55 else 0)+(20 if r.close>r.vwap else 0)+(25 if r.supertrend_dir>0 else 0)+(10 if r.atr>0 else 0)
    sell=(25 if r.ema20<r.ema50 else 0)+(20 if r.rsi<=45 else 0)+(20 if r.close<r.vwap else 0)+(25 if r.supertrend_dir<0 else 0)+(10 if r.atr>0 else 0)
    if buy>=70 and buy>sell:return 'BUY',buy
    if sell>=70 and sell>buy:return 'SELL',sell
    return 'NO TRADE',max(buy,sell)

def backtest(d,capital=100000,risk_pct=1,sl_atr=1.5,rr=2,max_trades_day=3):
    cash=capital; trades=[]; pos=None; counts={}
    for _,r in d.iterrows():
        if pd.isna(r.atr) or pd.isna(r.rsi): continue
        day=str(r.datetime)[:10]; sig,conf=score(r)
        if pos:
            ex=reason=None
            if pos['side']=='BUY':
                if r.low<=pos['sl']: ex,reason=pos['sl'],'SL'
                elif r.high>=pos['target']: ex,reason=pos['target'],'TARGET'
            else:
                if r.high>=pos['sl']: ex,reason=pos['sl'],'SL'
                elif r.low<=pos['target']: ex,reason=pos['target'],'TARGET'
            if ex is not None:
                pnl=(ex-pos['entry'])*pos['qty']*(1 if pos['side']=='BUY' else -1); cash+=pnl
                trades.append({**pos,'exit':ex,'pnl':pnl,'reason':reason,'exit_time':r.datetime}); pos=None
        if pos is None and sig in ('BUY','SELL') and counts.get(day,0)<max_trades_day:
            dist=max(r.atr*sl_atr,0.01); qty=max(1,int(cash*risk_pct/100/dist)); entry=float(r.close)
            pos={'side':sig,'entry':entry,'sl':entry-dist if sig=='BUY' else entry+dist,'target':entry+dist*rr if sig=='BUY' else entry-dist*rr,'qty':qty,'entry_time':r.datetime,'confidence':conf}; counts[day]=counts.get(day,0)+1
    return pd.DataFrame(trades),cash

st.title('🤖 ALS AI Algo Trading'); st.caption('V1 • Backtest + Paper Trading • Groww API not required')
with st.sidebar:
    st.header('Controls'); symbol=st.selectbox('Symbol',['NIFTY','BANK NIFTY','SENSEX']); capital=st.number_input('Capital',100000,10000000,100000,10000); risk=st.slider('Risk / trade (%)',0.25,2.0,1.0,0.25); sl_atr=st.slider('SL ATR',0.5,3.0,1.5,0.25); rr=st.slider('Reward : Risk',1.0,4.0,2.0,0.5)
uploaded=st.file_uploader('Upload historical CSV',type=['csv'])
if uploaded:
    try:
        d=indicators(pd.read_csv(uploaded)); last=d.dropna().iloc[-1]; sig,conf=score(last)
        c1,c2,c3,c4=st.columns(4); c1.metric('Symbol',symbol); c2.metric('Signal',sig); c3.metric('Confidence',f'{conf}%'); c4.metric('Last Close',f'{last.close:.2f}')
        st.subheader('📊 Indicators'); st.dataframe(d.tail(20),use_container_width=True)
        trades,final_cash=backtest(d,capital,risk,sl_atr,rr); pnl=final_cash-capital; wins=(trades.pnl>0).sum() if len(trades) else 0; winrate=wins/len(trades)*100 if len(trades) else 0
        a,b,c,e=st.columns(4); a.metric('Final Capital',f'₹{final_cash:,.2f}'); b.metric('P&L',f'₹{pnl:,.2f}'); c.metric('Trades',len(trades)); e.metric('Win Rate',f'{winrate:.1f}%')
        st.subheader('🧾 Trade Log'); st.dataframe(trades,use_container_width=True)
        if len(trades): st.download_button('Download Trade CSV',trades.to_csv(index=False),'trade_log.csv','text/csv')
    except Exception as ex: st.error(str(ex))
else:
    st.info('Upload a 5-minute OHLC/volume CSV to start the backtest.')
    st.markdown('### V1 Modules\n- Multi-indicator signal scoring\n- Entry / SL / Target\n- Risk-based position sizing\n- Backtest\n- Trade log\n- Paper-trading-ready architecture\n\n**Research/paper-trading only. Backtest results do not guarantee future profits.**')
