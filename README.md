# Groww AI Algo V2 — Render-ready starter

Mobile-friendly Flask dashboard for a Groww algo project.

Safety:
- PAPER mode is the default.
- Live orders require BOTH LIVE_TRADING=true and ENABLE_LIVE_ORDERS=true.
- Never put Groww API keys/secrets/tokens in GitHub.
- Start with paper testing.

Render:
Build Command: pip install -r requirements.txt
Start Command: gunicorn --bind 0.0.0.0:$PORT app.web:app

Render Free is for testing/paper mode; it can sleep and is not suitable for reliable 24/7 live trading.
