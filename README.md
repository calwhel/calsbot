# TradeHub Markets — Strategy Builder Platform

A web platform where users build, test, and automate crypto trading strategies — no code required.
Built with FastAPI (gunicorn + uvicorn workers), PostgreSQL (Neon), served on port 5000.

## Deployed Application

The **Strategy Portal** (`strategy_portal_server.py`) is the primary deployed service.
It runs on port 5000 via gunicorn with uvicorn workers.

Live at: `tradehubmarkets.com`

## Running

```bash
gunicorn -w 2 -k uvicorn.workers.UvicornWorker --reuse-port --bind 0.0.0.0:5000 strategy_portal_server:app
```

Or use the convenience script:

```bash
bash start.sh
```

## Service Entry Points

| Service | File | Port | Purpose |
|---------|------|------|---------|
| Strategy Portal (web) | `strategy_portal_server.py` | 5000 | Main product — deployed |
| Telegram Bot | `main.py` | 8080 | Trade alerts + UID lookup |
| Trade Tracker | `tracker_server.py` | 8000 | Performance dashboard |

## Health Check

```
GET /health  →  {"status": "ok"}
```

## Features

- 7-step no-code strategy wizard
- AI Chat strategy builder (Pro)
- PineScript import (Pro)
- Backtester — 30d / 90d replay against real OHLCV data (Pro)
- Strategy marketplace with 80/20 creator revenue split
- Live autonomous execution on Bitunix Futures (Pro)
- Paper trading mode (Free)
- Google OAuth + email/password authentication
- OxaPay subscription payments — Free vs Pro ($50/month)
- Telegram companion bot for trade alerts and UID lookup

## Stack

- **Backend**: Python 3.11, FastAPI, gunicorn, uvicorn
- **Database**: PostgreSQL via Neon (`NEON_DATABASE_URL`)
- **Exchange**: Bitunix Futures (MEXC for market data)
- **AI**: Claude (Anthropic) + Gemini for strategy building and analysis
- **Payments**: OxaPay
