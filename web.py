import os, threading, time
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template_string
import requests
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

STATE = {
    "running": False, "mode": "PAPER", "last_tick": None,
    "last_ltp": {}, "signal": "NO_TRADE", "confidence": 0.0,
    "message": "Ready", "trades_today": 0, "daily_pnl": 0.0,
}
LOCK = threading.Lock()

GROWW_BASE = "https://api.groww.in/v1"
ACCESS_TOKEN = os.getenv("GROWW_ACCESS_TOKEN", "").strip()
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "30"))
INSTRUMENTS = ["NSE_NIFTY", "NSE_BANKNIFTY", "BSE_SENSEX"]

def get_ltp(symbols):
    if not ACCESS_TOKEN:
        return {}
    try:
        r = requests.get(
            f"{GROWW_BASE}/live-data/ltp",
            headers={"Authorization": f"Bearer {ACCESS_TOKEN}", "Accept": "application/json"},
            params={"segment": "CASH", "exchange_trading_symbols": ",".join(symbols)},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("payload", data) if isinstance(data, dict) else {}
    except Exception as e:
        with LOCK:
            STATE["message"] = f"LTP error: {e}"
        return {}

def evaluate_signal(ltps):
    # Conservative V2 gate. Real strategy logic should be backtested first.
    return "NO_TRADE", 0.0

def loop():
    while True:
        with LOCK:
            running = STATE["running"]
        if running:
            ltps = get_ltp(INSTRUMENTS)
            with LOCK:
                STATE["last_tick"] = datetime.now(timezone.utc).isoformat()
                STATE["last_ltp"] = ltps
                STATE["signal"], STATE["confidence"] = evaluate_signal(ltps)
                STATE["message"] = (
                    "Paper engine running. No real orders are sent."
                    if STATE["mode"] == "PAPER"
                    else STATE["message"]
                )
        time.sleep(max(5, POLL_SECONDS))

threading.Thread(target=loop, daemon=True).start()

HTML = """<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Groww AI Algo V2</title>
<style>
body{font-family:Arial;max-width:760px;margin:auto;padding:18px;background:#f5f5f5}
.card{background:white;border-radius:14px;padding:16px;margin:12px 0;box-shadow:0 2px 8px #0001}
.row{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #eee}
button{padding:12px 18px;border:0;border-radius:10px;margin:4px;font-weight:600}
.start{background:#222;color:white}.warn{background:#ffe08a}.small{color:#666;font-size:13px}
</style></head><body>
<h1>🤖 Groww AI Algo V2</h1>
<div class="card">
<div class="row"><b>Mode</b><span id="mode">-</span></div>
<div class="row"><b>Engine</b><span id="running">-</span></div>
<div class="row"><b>Signal</b><span id="signal">-</span></div>
<div class="row"><b>Confidence</b><span id="confidence">-</span></div>
<div class="row"><b>Trades today</b><span id="trades">-</span></div>
<div class="row"><b>Daily P&L</b><span id="pnl">-</span></div>
<div class="row"><b>Last tick</b><span id="tick">-</span></div>
<p id="msg" class="small">Loading...</p></div>
<div class="card">
<button class="start" onclick="post('/api/paper/start')">▶ Start Paper</button>
<button onclick="post('/api/paper/stop')">■ Stop</button>
<button class="warn" onclick="post('/api/paper/reset')">↺ Reset</button>
</div>
<div class="card"><b>Indices</b><div id="ltp" class="small">-</div></div>
<div class="card small">Live trading is locked by default. Validate strategy, risk, order handling and paper results before enabling live trading.</div>
<script>
async function post(u){await fetch(u,{method:'POST'});refresh()}
async function refresh(){let s=await (await fetch('/api/status')).json();
mode.textContent=s.mode;running.textContent=s.running?'RUNNING':'STOPPED';
signal.textContent=s.signal;confidence.textContent=s.confidence;
trades.textContent=s.trades_today;pnl.textContent=s.daily_pnl;
tick.textContent=s.last_tick||'-';msg.textContent=s.message;
ltp.textContent=JSON.stringify(s.last_ltp||{})}
setInterval(refresh,3000);refresh();
</script></body></html>"""

@app.get("/")
def home():
    return render_template_string(HTML)

@app.get("/healthz")
def healthz():
    return "ok", 200

@app.get("/api/status")
def status():
    with LOCK:
        return jsonify(dict(STATE))

@app.post("/api/paper/start")
def paper_start():
    with LOCK:
        STATE["mode"], STATE["running"], STATE["message"] = "PAPER", True, "Paper engine started."
    return jsonify({"ok": True})

@app.post("/api/paper/stop")
def paper_stop():
    with LOCK:
        STATE["running"], STATE["message"] = False, "Paper engine stopped."
    return jsonify({"ok": True})

@app.post("/api/paper/reset")
def paper_reset():
    with LOCK:
        STATE.update(running=False, mode="PAPER", signal="NO_TRADE",
                     confidence=0.0, trades_today=0, daily_pnl=0.0,
                     last_ltp={}, last_tick=None, message="Reset complete.")
    return jsonify({"ok": True})
