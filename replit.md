# Crypto Perps Signals Telegram Bot

## Overview
A Python-based Telegram bot that generates and broadcasts cryptocurrency perpetual futures trading signals using EMA crossover strategy with support/resistance analysis. The bot is free to use with user preferences system and PnL tracking. Webhook infrastructure exists for future payment integration if needed.

## Current State
- ✅ Fully functional Telegram bot with aiogram
- ✅ FastAPI server for webhooks and health checks
- ✅ PostgreSQL database with SQLAlchemy ORM
- ✅ Signal generation using CCXT + TA library
- ✅ User management and subscription tracking
- ✅ Interactive dashboard with inline buttons
- ✅ User preferences (mute symbols, PnL periods, DM alerts)
- ✅ Both bot and API running in single process

## Recent Changes (October 12, 2025)
- Built complete crypto signals bot from scratch
- Implemented EMA crossover signal generation with support/resistance levels
- Created FastAPI endpoints for Whop and Solana Pay webhooks
- Set up PostgreSQL database with auto-initialization
- Added Telegram bot commands: /dashboard, /settings, /subscribe, /status
- Configured single-process architecture with bot polling in FastAPI startup
- Created start script for uvicorn deployment
- **Removed subscription requirement - bot now broadcasts signals to all users for free with optional DM alerts based on user preferences**

## Project Architecture

### Directory Structure
```
app/
├── __init__.py
├── config.py              # Environment configuration with pydantic-settings
├── database.py            # SQLAlchemy setup and session management
├── models.py              # Database models (User, Signal, Trade, etc.)
├── services/
│   ├── bot.py            # Telegram bot with aiogram
│   ├── signals.py        # Signal generation and EMA strategy
│   └── subscriptions.py  # FastAPI webhooks
└── utils/                 # (Reserved for future utilities)

main.py                    # Application entry point
start.sh                   # Startup script
.env.example              # Environment variables template
```

### Key Components

**Signal Generator** (`app/services/signals.py`)
- Uses CCXT to fetch OHLCV data from exchanges
- Calculates EMA indicators (fast/slow/trend)
- Identifies support/resistance levels
- Detects crossover signals
- Computes stop loss and take profit prices

**Telegram Bot** (`app/services/bot.py`)
- Commands: start, status, subscribe, dashboard, settings
- Interactive dashboard with callback queries
- User preferences: mute/unmute symbols, toggle alerts
- PnL tracking (today/week/month views)
- Signal broadcasting to channel and DM

**FastAPI Service** (`app/services/subscriptions.py`)
- Health endpoint: GET /health
- Whop webhook: POST /webhook/whop (HMAC verification)
- Solana webhook: POST /webhook/solana (Helius integration)
- Automatic 30-day subscription grants

**Database Models** (`app/models.py`)
- User: Telegram users with subscription tracking
- UserPreference: Settings per user
- Signal: Generated trading signals
- Trade: User trade tracking with PnL
- Subscription: Payment history

## Configuration

### Required Environment Variables
- `TELEGRAM_BOT_TOKEN` - Bot token from @BotFather
- `BROADCAST_CHAT_ID` - Channel ID for signal broadcasts
- `DATABASE_URL` - Auto-provided by Replit PostgreSQL

### Trading Parameters
- `SYMBOLS` - Trading pairs (default: BTC/USDT:USDT,ETH/USDT:USDT)
- `EXCHANGE` - Exchange name (default: binance)
- `TIMEFRAME` - Candle timeframe (default: 15m)
- `EMA_FAST/SLOW/TREND` - EMA periods (default: 9/21/50)

### Payment Integration (Optional)
- Whop: `WHOP_CHECKOUT_URL`, `WHOP_WEBHOOK_SECRET`
- Solana: `SOL_MERCHANT`, `HELIUS_WEBHOOK_SECRET`, `SPL_USDC_MINT`

## User Preferences
Users can customize their experience through settings:
- **Muted Symbols**: Disable signals for specific pairs
- **Default PnL Period**: Choose default view (today/week/month)
- **DM Alerts**: Toggle private message notifications

## Technical Details

### Single Process Architecture
The application runs both the Telegram bot (polling) and FastAPI server in a single process using FastAPI's lifespan context manager. The bot runs as a background task that starts with the server.

### Database Auto-Initialization
Tables are automatically created on startup using SQLAlchemy's `Base.metadata.create_all()` in the lifespan handler.

### Signal Scanning
A background task scans all configured symbols every 60 seconds for EMA crossover signals and broadcasts them to all users. Users can enable DM alerts in settings to receive signals directly.

### Webhook Security
- Whop webhooks use HMAC-SHA256 signature verification
- Solana webhooks use bearer token authorization
- All payment webhooks validate data before granting access

## Known Issues
- Binance API may be restricted in some locations (Replit servers)
- LSP diagnostics show type checking warnings (non-critical, runtime works correctly)
- Exchange API credentials not yet implemented (public endpoints only)

## Future Enhancements
- Add backtesting functionality
- Implement trade history export
- Add multi-exchange support
- Create admin dashboard
- Add signal performance tracking with win/loss ratios
