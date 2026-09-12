import os
import threading
import time
from datetime import datetime
from flask import Flask, jsonify, render_template_string
import requests

app = Flask(__name__)

GROWW_BASE = "https://api.groww.in/v1"
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "30"))

# Keep this PAPER-only until the strategy and risk controls are fully tested.
state = {
    "mode": "PAPER",
    "engine": "STOPPED",
    "signal": "NO_TRADE",
    "confidence": 0.0,
    "trades_today": 0,
    "daily_pnl": 0.0,
    "last_tick": "-",
    "indices": {},
    "error": "",
}

INSTRUMENTS = ["NSE_NIFTY", "NSE_BANKNIFTY", "BSE_SENSEX"]

HTML = """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Groww AI Algo V2</title>
  <style>
    body{font-family:Arial,sans-serif;background:#111;color:#eee;margin:0;padding:20px}
    .card{background:#1d1d1d;border-radius:14px;padding:18px;margin-bottom:14px}
    h1{font-size:24px;margin:0 0 12px}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .item{background:#252525;border-radius:10px;padding:12px}
    .label{font-size:12px;color:#aaa}.value{font-size:20px;margin-top:5px}
    button{border:0;border-radius:10px;padding:13px 18px;margin:4px;font-weight:700}
    .start{background:#38d39f}.stop{background:#ff7675}.reset{background:#777;color:#fff}
    pre{white-space:pre-wrap;word-break:break-word;color:#ffb4b4}
  </style>
</head>
<body>
  <div class="card">
    <h1>🤖 Groww AI Algo V2</h1>
    <div>Mode: <b>{{s.mode}}</b></div>
    <div>Engine: <b>{{s.engine}}</b></div>
  </div>

  <div class="card grid">
    <div class="item"><div class="label">Signal</div><div class="value">{{s.signal}}</div></div>
    <div class="item"><div class="label">Confidence</div><div class="value">{{"%.1f"|format(s.confidence)}}</div></div>
    <div class="item"><div class="label">Trades today</div><div class="value">{{s.trades_today}}</div></div>
    <div class="item"><div class="label">Daily P&L</div><div class="value">₹{{"%.2f"|format(s.daily_pnl)}}</div></div>
  </div>

  <div class="card">
    <b>Last tick:</b> {{s.last_tick}}<br><br>
    <b>Indices</b>
    <pre>{{s.indices}}</pre>
    {% if s.error %}
      <b>API/Error</b>
      <pre>{{s.error}}</pre>
    {% endif %}
  </div>

  <div class="card">
    <button class="start" onclick="location.href='/start'">Start Paper</button>
    <button class="stop" onclick="location.href='/stop'">Stop</button>
    <button class="reset" onclick="location.href='/reset'">Reset</button>
  </div>

  <div class="card">
    <small>Paper mode only. Live order placement is intentionally disabled.</small>
  </div>
</body>
</html>
"""

def get_ltp():
    token = os.getenv("GROWW_ACCESS_TOKEN", "").strip()
    if not token:
        return {}, "GROWW_ACCESS_TOKEN is not configured in Render."

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "X-API-VERSION": "1.0",
    }
    params = {
        "segment": "CASH",
        # Current Groww REST API parameter name:
        "exchange_symbols": ",".join(INSTRUMENTS),
    }

    try:
        r = requests.get(
            f"{GROWW_BASE}/live-data/ltp",
            headers=headers,
            params=params,
            timeout=15,
        )
        data = r.json()
        if r.status_code != 200:
            return {}, f"HTTP {r.status_code}: {data}"
        if data.get("status") != "SUCCESS":
            return {}, str(data)
        return data.get("payload", {}), ""
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"

def evaluate_signal(indices):
    # Placeholder strategy: no real trade is generated yet.
    # This keeps the system safe while market-data connectivity is verified.
    return "NO_TRADE", 0.0

def paper_loop():
    while True:
        if state["engine"] == "RUNNING":
            indices, error = get_ltp()
            state["indices"] = indices
            state["error"] = error
            state["last_tick"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            signal, confidence = evaluate_signal(indices)
            state["signal"] = signal
            state["confidence"] = confidence
        time.sleep(POLL_SECONDS)

@app.route("/")
def home():
    return render_template_string(HTML, s=state)

@app.route("/start")
def start():
    state["engine"] = "RUNNING"
    state["error"] = ""
    return ("<script>location.href='/'</script>", 302)

@app.route("/stop")
def stop():
    state["engine"] = "STOPPED"
    return ("<script>location.href='/'</script>", 302)

@app.route("/reset")
def reset():
    state["trades_today"] = 0
    state["daily_pnl"] = 0.0
    state["signal"] = "NO_TRADE"
    state["confidence"] = 0.0
    state["error"] = ""
    return ("<script>location.href='/'</script>", 302)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "engine": state["engine"]})

if __name__ == "__main__":
    threading.Thread(target=paper_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
else:
    threading.Thread(target=paper_loop, daemon=True).start()
