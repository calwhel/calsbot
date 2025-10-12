# Crypto Perps Signals Bot

A comprehensive Telegram bot that sends crypto perpetual futures trading signals based on EMA crossover strategy with support/resistance levels. Includes FastAPI webhooks for subscription management via Whop and Solana Pay.

## Features

### Trading Signals
- **EMA Crossover Strategy**: Uses configurable fast, slow, and trend EMAs
- **Support/Resistance Levels**: Automatically identifies key price levels
- **Entry/Exit Prices**: Provides entry, stop loss, and take profit levels
- **Multi-Symbol Support**: Monitor multiple trading pairs simultaneously
- **Real-time Broadcasting**: Signals sent to broadcast channel and subscribed users

### Telegram Bot Commands
- `/start` - Initialize bot and register user
- `/dashboard` - Interactive dashboard with PnL and trades
- `/settings` - Configure user preferences
- `/subscribe` - View subscription payment options
- `/status` - Check subscription status
- `/mute <symbol>` - Mute signals for a specific symbol
- `/unmute <symbol>` - Unmute signals for a symbol
- `/set_pnl <today/week/month>` - Set default PnL period
- `/toggle_alerts` - Enable/disable DM alerts

### User Preferences
- Mute signals per symbol
- Choose default PnL period (today/week/month)
- Enable/disable DM alerts
- Track active trades and PnL

### Subscription Management
- **Whop Integration**: Card payments via Whop checkout
- **Solana Pay**: USDC payments on Solana blockchain
- **Webhook Verification**: Secure signature validation
- **30-Day Access**: Automatic subscription extension

### API Endpoints
- `GET /health` - Health check endpoint
- `POST /webhook/whop` - Whop payment webhook
- `POST /webhook/solana` - Solana payment webhook (Helius)

## Environment Variables

### Required
- `TELEGRAM_BOT_TOKEN` - Your Telegram bot token from @BotFather
- `BROADCAST_CHAT_ID` - Channel/group ID for broadcasting signals
- `DATABASE_URL` - PostgreSQL connection string (auto-provided by Replit)

### Trading Configuration
- `SYMBOLS` - Comma-separated trading pairs (default: `BTC/USDT:USDT,ETH/USDT:USDT`)
- `EXCHANGE` - Exchange name (default: `binance`)
- `TIMEFRAME` - Trading timeframe (default: `15m`)
- `EMA_FAST` - Fast EMA period (default: `9`)
- `EMA_SLOW` - Slow EMA period (default: `21`)
- `EMA_TREND` - Trend EMA period (default: `50`)
- `TRAIL_PCT` - Trailing stop percentage (default: `1.5`)

### Subscription Settings
- `SUB_PRICE_USDC` - Subscription price in USDC (default: `50.0`)
- `WHOP_CHECKOUT_URL` - Whop checkout page URL (optional)
- `WHOP_WEBHOOK_SECRET` - Whop webhook signature secret (optional)
- `SOL_MERCHANT` - Solana wallet address for payments (optional)
- `SPL_USDC_MINT` - USDC token mint address (default: mainnet USDC)
- `HELIUS_WEBHOOK_SECRET` - Helius webhook authorization token (optional)

### Server
- `PORT` - Server port (default: `5000`)
- `TIMEZONE` - Timezone for timestamps (default: `UTC`)

## Database Schema

### Tables
- **users** - User accounts with subscription info
- **user_preferences** - User settings and preferences
- **signals** - Generated trading signals
- **trades** - User trade tracking and PnL
- **subscriptions** - Payment history

## Architecture

- **Single Process**: Bot polling and FastAPI run in one process
- **Async Design**: Fully asynchronous using aiogram and FastAPI
- **Auto-Init Database**: Tables created automatically on startup
- **Background Scanner**: Scans symbols every 60 seconds for signals
- **Webhook Security**: HMAC signature verification for payments

## Running the Application

The application runs automatically via the configured workflow:

```bash
bash start.sh
```

This starts both the Telegram bot (polling) and FastAPI server on port 5000.

## Testing

- Health check: `curl http://localhost:5000/health`
- Bot commands: Message your bot on Telegram
- Webhooks: Send POST requests to `/webhook/whop` or `/webhook/solana`

## Notes

- Binance may be restricted in some locations (use VPN or different exchange)
- Configure CCXT exchange credentials if needed
- Subscription webhooks require valid secrets for production use
- Signal accuracy depends on market conditions and strategy parameters
