# How to Use Your Crypto Signals Bot

## Quick Start

Your bot is now running! Here's how to use it:

### 1. Find Your Bot on Telegram
Search for your bot using the username you created with @BotFather

### 2. Start the Bot
Send `/start` to your bot to register and see available commands

### 3. Available Commands

#### Basic Commands
- `/start` - Register and see welcome message
- `/status` - Check your subscription status
- `/subscribe` - View payment options to get premium access
- `/dashboard` - Access your interactive trading dashboard

#### Settings & Preferences
- `/settings` - View your current preferences
- `/mute <symbol>` - Stop receiving signals for a specific pair (e.g., `/mute BTC/USDT:USDT`)
- `/unmute <symbol>` - Resume receiving signals for a pair
- `/set_pnl <period>` - Set default PnL view (today/week/month)
- `/toggle_alerts` - Turn DM notifications on/off

### 4. Interactive Dashboard

The `/dashboard` command shows buttons for:
- **📊 PnL Today/Week/Month** - View your profit/loss for different periods
- **🔄 Active Trades** - See your currently open positions
- **📡 Recent Signals** - View the latest trading signals
- **⚙️ Settings** - Quick access to preferences

### 5. Signal Format

When a new trading signal is detected, you'll receive:
```
🚨 NEW LONG/SHORT SIGNAL

📊 Symbol: BTC/USDT:USDT
💰 Entry: $45000.00
🛑 Stop Loss: $44500.00
🎯 Take Profit: $46000.00

📈 Support: $44500.00
📉 Resistance: $45500.00

⏰ 15m | 14:30:25
```

### 6. Subscription Management

#### Option 1: Whop (Card Payments)
1. Use `/subscribe` command
2. Click "💳 Pay with Card (Whop)"
3. Complete checkout
4. Subscription activated automatically

#### Option 2: Solana Pay (USDC)
1. Use `/subscribe` command
2. Click "◎ Pay with Solana"
3. Send USDC payment
4. Include your Telegram ID in memo
5. Subscription activated on confirmation

## Configuration Tips

### Adjusting Trading Pairs
Edit the `SYMBOLS` environment variable to monitor different pairs:
```
SYMBOLS=BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT
```

### Changing Strategy Parameters
Modify EMA periods for different trading styles:
- **Scalping**: EMA_FAST=5, EMA_SLOW=13, EMA_TREND=21
- **Day Trading**: EMA_FAST=9, EMA_SLOW=21, EMA_TREND=50 (default)
- **Swing Trading**: EMA_FAST=21, EMA_SLOW=50, EMA_TREND=100

### Exchange Selection
Change the `EXCHANGE` variable to use different exchanges:
- binance
- bybit
- okx
- kucoin
- And many more (see CCXT documentation)

## Troubleshooting

### Bot Not Responding
1. Check if the Server workflow is running
2. Verify `TELEGRAM_BOT_TOKEN` is set correctly
3. Check logs for any errors

### Signals Not Broadcasting
1. Verify `BROADCAST_CHAT_ID` is correct
2. Make sure the bot is admin in the channel
3. Check if exchange API is accessible (Binance may be restricted in some regions)

### Subscription Not Activating
1. For Whop: Check webhook signature is configured
2. For Solana: Verify Helius webhook secret matches
3. Check that payment includes correct Telegram ID

## API Endpoints

Your FastAPI server provides:
- `GET /health` - Health check (returns `{"ok": true}`)
- `POST /webhook/whop` - Whop payment webhook
- `POST /webhook/solana` - Solana payment webhook

## Next Steps

1. **Test the Bot**: Send `/start` to your bot and explore features
2. **Configure Payments**: Set up Whop or Solana Pay for subscriptions
3. **Customize Signals**: Adjust EMA parameters and symbols
4. **Monitor Performance**: Track PnL and signal accuracy
5. **Deploy to Production**: Use the publish button to deploy your bot

Enjoy your automated crypto trading signals! 🚀
