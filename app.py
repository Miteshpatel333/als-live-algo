import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='ALS AI Algo', page_icon='📈', layout='wide')
TICKERS={'NIFTY':'^NSEI','BANK NIFTY':'^NSEBANK','SENSEX':'^BSESN'}

def indicators(df):
    d=df.copy(); d.columns=[str(c).strip().lower().replace(' ','_') for c in d.columns]
    if 'datetime' not in d.columns:
        for c in ['date','timestamp','time']:
            if c in d.columns: d['datetime']=pd.to_datetime(d[c],errors='coerce'); break
    d['datetime']=pd.to_datetime(d['datetime'],errors='coerce')
    for c in ['open','high','low','close','volume']:
        if c in d.columns: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['open','high','low','close']).sort_values('datetime').reset_index(drop=True)
    d['ema20']=d.close.ewm(span=20,adjust=False).mean(); d['ema50']=d.close.ewm(span=50,adjust=False).mean()
    delta=d.close.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
    d['rsi']=100-(100/(1+(gain/loss.replace(0,np.nan))))
    tr=pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()],axis=1).max(axis=1); d['atr']=tr.rolling(14).mean()
    vol=d['volume'] if 'volume' in d.columns else pd.Series(1.0,index=d.index); d['vwap']=(d.close*vol).cumsum()/vol.cumsum()
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

def backtest(d,capital,risk_pct,sl_atr,rr,max_trades_day=3):
    cash=capital; position=None; trades=[]; counts={}
    for _,r in d.iterrows():
        if pd.isna(r.atr) or pd.isna(r.rsi): continue
        day=str(r.datetime)[:10]; sig,conf=score(r)
        if position:
            exit_price=reason=None
            if position['side']=='BUY':
                if r.low<=position['sl']: exit_price,reason=position['sl'],'SL'
                elif r.high>=position['target']: exit_price,reason=position['target'],'TARGET'
            else:
                if r.high>=position['sl']: exit_price,reason=position['sl'],'SL'
                elif r.low<=position['target']: exit_price,reason=position['target'],'TARGET'
            if exit_price is not None:
                pnl=(exit_price-position['entry'])*position['qty']*(1 if position['side']=='BUY' else -1); cash+=pnl
                trades.append({**position,'exit':exit_price,'pnl':pnl,'reason':reason,'exit_time':r.datetime}); position=None
        if position is None and sig in ('BUY','SELL') and counts.get(day,0)<max_trades_day:
            dist=max(r.atr*sl_atr,0.01); qty=max(1,int((cash*risk_pct/100)/dist)); entry=float(r.close)
            sl=entry-dist if sig=='BUY' else entry+dist; target=entry+dist*rr if sig=='BUY' else entry-dist*rr
            position={'side':sig,'entry':entry,'sl':sl,'target':target,'qty':qty,'entry_time':r.datetime,'confidence':conf}; counts[day]=counts.get(day,0)+1
    return pd.DataFrame(trades),cash

st.title('🤖 ALS AI Algo Trading'); st.caption('V2 • Automatic historical data • Backtest + Paper Trading • Groww API not required')
with st.sidebar:
    symbol=st.selectbox('Symbol',list(TICKERS)); period=st.selectbox('Historical period',['3mo','6mo','1y','2y'],index=1); interval=st.selectbox('Candle interval',['5m','15m','1h','1d'],index=0)
    capital=st.number_input('Capital',100000,10000000,100000,10000); risk=st.slider('Risk / trade (%)',0.25,2.0,1.0,0.25); sl_atr=st.slider('SL ATR',0.5,3.0,1.5,0.25); rr=st.slider('Reward : Risk',1.0,4.0,2.0,0.5)
st.info('No CSV required. Fetch historical market data automatically below.')
if st.button('📥 Fetch Market Data & Run Backtest',type='primary'):
    try:
        import yfinance as yf
        raw=yf.download(TICKERS[symbol],period=period,interval=interval,auto_adjust=False,progress=False)
        if raw is None or raw.empty: st.error('No data returned. Try another period/interval.')
        else:
            if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
            raw=raw.reset_index(); raw.columns=[str(c).strip().lower().replace(' ','_') for c in raw.columns]
            if 'datetime' not in raw.columns:
                for c in ['date','timestamp']:
                    if c in raw.columns: raw['datetime']=pd.to_datetime(raw[c],errors='coerce'); break
            d=indicators(raw); last=d.dropna().iloc[-1]; sig,conf=score(last)
            a,b,c,e=st.columns(4); a.metric('Symbol',symbol); b.metric('Signal',sig); c.metric('Confidence',f'{conf}%'); e.metric('Last Close',f'{last.close:.2f}')
            trades,final_cash=backtest(d,capital,risk,sl_atr,rr); pnl=final_cash-capital; winrate=((trades.pnl>0).sum()/len(trades)*100) if len(trades) else 0
            a,b,c,e=st.columns(4); a.metric('Final Capital',f'₹{final_cash:,.2f}'); b.metric('P&L',f'₹{pnl:,.2f}'); c.metric('Trades',len(trades)); e.metric('Win Rate',f'{winrate:.1f}%')
            st.subheader('📊 Indicators'); st.dataframe(d.tail(30),use_container_width=True); st.subheader('🧾 Trade Log'); st.dataframe(trades,use_container_width=True)
    except Exception as ex: st.error(f'Data/backtest error: {ex}')
st.divider(); st.caption('Research/paper-trading only. Historical backtests do not guarantee future profits.')
