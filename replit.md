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
- ✅ **MEXC Auto-Trading Integration** - Users can connect API keys for automated trade execution
- ✅ **Encrypted Credential Storage** - API keys secured with Fernet encryption at rest

## Recent Changes (October 12, 2025)
- Built complete crypto signals bot from scratch
- Implemented EMA crossover signal generation with support/resistance levels
- Created FastAPI endpoints for Whop and Solana Pay webhooks
- Set up PostgreSQL database with auto-initialization
- Added Telegram bot commands: /dashboard, /settings, /subscribe, /status
- Configured single-process architecture with bot polling in FastAPI startup
- Created start script for uvicorn deployment
- **Removed subscription requirement - bot now broadcasts signals to all users for free with optional DM alerts based on user preferences**
- **Enhanced strategy with Volume Confirmation, RSI Filter, and ATR-based Stops for higher quality signals**
- **Fixed numpy type compatibility issues - converted numpy.float64 to Python native floats for database storage**
- **Added 10x Leverage PnL Calculator - shows potential profit/loss percentages with 10x leverage on both TP and SL scenarios in broadcasts and dashboard**
- **Added duplicate signal prevention - blocks same symbol/direction signals within 4 hours to prevent spam**
- **Implemented MEXC Auto-Trading System** - Users can connect MEXC API keys to automatically execute trades from signals
- **Added Fernet Encryption for API Keys** - All API credentials encrypted at rest using ENCRYPTION_KEY environment variable
- **Auto-Trading Features**: Configurable position sizing (% of balance), max positions limit, automatic SL/TP placement with 10x leverage
- **Changed to 4h Timeframe** - Switched from 15m to 4h candles for longer-term trades (1-2 day duration)
- **Risk Assessment System** - Signals scored based on ATR volatility, RSI extremes, and risk/reward ratio, classified as LOW/MEDIUM/HIGH
- **Risk Filtering** - Only broadcasts MEDIUM and LOW risk signals to improve win rate and PnL
- **Enhanced PnL Tracking** - Dashboard shows detailed statistics: avg PnL per trade, avg win/loss, best/worst trades, win rate
- **Advanced Risk Management** - Risk-based position sizing (70% for MEDIUM risk), customizable accepted risk levels
- **Comprehensive Security System** - Daily loss limits, max drawdown protection, minimum balance checks, emergency stop, auto-resume features

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
│   ├── subscriptions.py  # FastAPI webhooks
│   └── mexc_trader.py    # MEXC auto-trading service
└── utils/
    ├── __init__.py
    └── encryption.py      # Fernet encryption for API keys

main.py                    # Application entry point
start.sh                   # Startup script
.env.example              # Environment variables template
```

### Key Components

**Signal Generator** (`app/services/signals.py`)
- Uses CCXT to fetch OHLCV data from exchanges
- Calculates EMA indicators (fast/slow/trend)
- **Volume Confirmation**: Only triggers signals when volume exceeds 20-period average
- **RSI Filter**: Prevents LONG when RSI > 70 (overbought) and SHORT when RSI < 30 (oversold)
- **ATR-based Stops**: Dynamic stop loss (2x ATR) and take profit (3x ATR) that adapt to volatility
- Identifies support/resistance levels
- Detects crossover signals with multi-indicator confirmation
- NaN guards ensure signals only emit when all indicators are fully populated

**Telegram Bot** (`app/services/bot.py`)
- Commands: start, status, subscribe, dashboard, settings
- **Auto-Trading Commands**: set_mexc_api, remove_mexc_api, toggle_autotrading, autotrading_status
- Interactive dashboard with callback queries
- User preferences: mute/unmute symbols, toggle alerts
- PnL tracking (today/week/month views)
- **10x Leverage PnL Calculator**: Shows potential profit/loss with 10x leverage for TP/SL scenarios
- Signal broadcasting to channel and DM
- API key deletion after setting for security

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
- `ENCRYPTION_KEY` - Fernet encryption key for securing API credentials (auto-generated)

### Trading Parameters
- `SYMBOLS` - Trading pairs (default: BTC/USDT:USDT,ETH/USDT:USDT)
- `EXCHANGE` - Exchange name (default: binance)
- `TIMEFRAME` - Candle timeframe (default: 4h for 1-2 day trades)
- `SCAN_INTERVAL` - Signal scan interval in seconds (default: 900 = 15 minutes)
- `EMA_FAST/SLOW/TREND` - EMA periods (default: 9/21/50)

### Payment Integration (Optional)
- Whop: `WHOP_CHECKOUT_URL`, `WHOP_WEBHOOK_SECRET`
- Solana: `SOL_MERCHANT`, `HELIUS_WEBHOOK_SECRET`, `SPL_USDC_MINT`

## User Preferences
Users can customize their experience through settings:
- **Muted Symbols**: Disable signals for specific pairs
- **Default PnL Period**: Choose default view (today/week/month)
- **DM Alerts**: Toggle private message notifications

## MEXC Auto-Trading System

### Overview
Users can connect their MEXC exchange API keys to automatically execute trades when signals are generated. The system uses Fernet encryption to securely store credentials and CCXT library for trade execution.

### Security Features
- **Encrypted Storage**: API keys encrypted with Fernet before storing in database
- **Decryption on Use**: Keys decrypted in-memory only when needed for trading
- **Message Deletion**: API key messages auto-deleted after processing
- **Environment Key**: Master encryption key stored in environment variables

### Auto-Trading Features
- **Position Sizing**: Users can set position size as percentage of account balance (default: 5%)
- **Max Positions**: Limit simultaneous open positions (default: 3)
- **Automatic Orders**: Places market entry, stop loss, and take profit orders
- **10x Leverage**: All trades executed with 10x leverage on MEXC perpetuals
- **Risk Management**: Automatic SL/TP placement based on signal levels

### Commands
- `/set_mexc_api <api_key> <api_secret>` - Connect MEXC account
- `/remove_mexc_api` - Remove API keys and disable trading
- `/toggle_autotrading` - Enable/disable auto-trading
- `/autotrading_status` - View auto-trading configuration and positions

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
- MEXC API requires specific order types and parameters per exchange documentation

## Future Enhancements
- Add backtesting functionality
- Implement trade history export
- Add multi-exchange support
- Create admin dashboard
- Add signal performance tracking with win/loss ratios
