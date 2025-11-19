# Crypto Perps Signals Telegram Bot

## Recent Changes (Nov 18, 2025) - Pre-Launch Updates

### Parabolic Reversal Strategy
- **🚀 PARABOLIC DUMPS: AGGRESSIVE 200% TP @ 20x**: Exhausted 50%+ pumps now use maximum aggression
  - TP: 10% price move = 200% profit @ 20x leverage 🔥
  - SL: 5% price move = 100% loss @ 20x leverage
  - R:R: 2:1 (go big or go home on BANANA-style exhausted pumps!)
  - Examples: BANANA +100% → dumps hard = perfect parabolic candidate
  - Dedicated parabolic scanner runs every 5 min looking for 50%+ reversals
- **CRITICAL FIX: Parabolic Cooldown Bypass**: Parabolic shorts (50%+) now IGNORE cooldown
  - Previous bug: Normal SHORT (28%) loses → cooldown → blocks parabolic (50%+) detection
  - Now: Parabolic scanner bypasses cooldown to catch exhausted pumps like BANANA
  - Cooldown only applies to normal shorts (28-49%), not parabolic (50%+)
- **SHORTS TP Unified**: Changed SHORTS (normal 28%+ dumps) to single TP at 80% profit for all leverage levels
  - 16% price move → always 80% profit at ANY leverage via auto-scaling
- **TP/SL Leverage Cap Fixed**: Both TP and SL correctly cap at 80% max for 5x-20x leverage
  - SL no longer incorrectly scales with TP (was showing 32% instead of 80% at 20x)

### UI/UX Pre-Launch Updates
- **Dashboard Cleanup**: Removed non-functional PnL buttons (Today/Week/Month)
  - Replaced with "🔍 Scan Coins" button for quick coin analysis
  - Removed Size/Leverage display (wasn't updating from settings changes)
- **New User Access**: New users can now access home screen and referral links before subscribing
  - Referral system accessible to all (anyone can share and earn $30 USD per Auto-Trading referral)
  - Premium features still require subscription (Dashboard, Auto-Trading, Top Gainers, etc.)
  - Only banned users are fully blocked from the bot

### LONGS Strategy Fixes
- **FIX #1: 3-Hour Freshness Check Removed** (Initial fix)
  - Previous bug: Coin pumps 10% at 4-6h ago → Still valid but rejected (only +1% in last 3h)
  - Now: Trust tier-based freshness validation (5m/15m/30m candles)
- **FIX #2: Prevent Buying Tops** (Critical fix after XAN +33% top entry)
  - Previous bug: LONGS entered XAN at +33% at top of green candle → Hit SL easily
  - Max pump range: Changed from 5-50% to **5-20%** (catch EARLY pumps only!)
  - RSI filter: Tightened from 35-85 to **40-70** (avoid overbought/exhausted)
  - EMA9 distance: Tightened from 8% to **5%** max (avoid extended entries)
  - Strong pump RSI: Tightened from 35-85 to **40-70** (no more buying tops!)
- **FIX #3: Removed Direct Entry on Strong Pumps** (Nov 19, 2025)
  - Previous bug: Entry Condition 3 allowed entering ON green candles without retracement
  - Now: ALL LONGS require retracement (either EMA9 pullback or resumption pattern)
  - No more buying tops - must wait for pullback before entry!
- **Entry Strategies**: EMA9 pullback (best) or resumption pattern ONLY (direct strong pump entry REMOVED)
- **Dual TPs**: 40% and 80% profit @ 5x leverage
- **Target Range**: Now strictly 5-20% pumps (truly EARLY momentum, not exhausted)

### Subscription & Payment Updates
- **Fee Warning Added**: Payment page now warns users to send slightly more than exact amount to cover network fees
  - Instructs users to contact @bu11dogg if payment is short
- **Manual Subscription Activation**: New admin command `/grant_sub` to manually activate subscriptions
  - Useful for users who paid but were short due to network/exchange fees
  - Usage: `/grant_sub <telegram_id> <plan> [days]`
  - Automatically notifies user when subscription is activated

### Market Analysis Features
- **Funding Rate Integration**: Market sentiment confirmation (+5-10 confidence boost)
- **Order Book Wall Detection**: Skips entries against massive whale resistance/support (>5x avg order)
- **Daily PnL Reports Disabled**: No more automated 11:59 PM UTC notifications

## Overview
This project is a Python-based Telegram bot for crypto perpetual trading with automated execution on the Bitunix exchange. It features three independent trading modes: PARABOLIC (exhausted dumps), SHORTS (mean reversion on pumps), and LONGS (early momentum entries on fresh pumps). The bot utilizes a "Top Gainers" scanning system with dedicated parabolic dump detection. Core strategies involve momentum-based entries with customizable leverage, dual/triple take-profit targets, and breakeven stop-loss management. The project includes a 2-tier subscription model and a cash referral system. The business vision is to provide high-quality, automated crypto trading signals and execution, offering a valuable tool for traders seeking to capitalize on market movements and a revenue stream through subscriptions and copy trading.

## User Preferences
- Muted Symbols: Disable signals for specific pairs
- Default PnL Period: Choose default view (today/week/month)
- DM Alerts: Toggle private message notifications
- Position Sizing: Users can set position size as percentage of account balance
- Max Positions: Limit simultaneous open positions
- Risk Filtering: Customizable accepted risk levels for signals
- Correlation Filter: Prevent opening correlated positions (e.g., BTC + ETH simultaneously)
- Funding Rate Alerts: Get notified of extreme funding rates for arbitrage opportunities
- Top Gainers Mode: Enable/disable automated trading of high-momentum coins (5x leverage, 20% TP/SL, max 3 positions)

## System Architecture

### Core Components
- **Telegram Bot**: Manages user interaction, commands, signal broadcasting, and an interactive dashboard.
- **FastAPI Server**: Provides health checks and webhook endpoints.
- **Bitunix Auto-Trading System**: Handles automated live trade execution on Bitunix Futures with configurable leverage and risk management.
- **Master Trader Copy Trading**: Executes all signals on the owner's Bitunix Copy Trading account in parallel for follower profit-sharing.
- **Top Gainers Trading Mode**: Supports SHORTS_ONLY, LONGS_ONLY, or BOTH, with specific strategies for mean reversion and momentum entries.
- **Volume Surge Detector**: Real-time detection of volume spikes for early entry opportunities.
- **New Coin Alerts**: Automated detection of newly listed, high-volume coins on Bitunix.
- **Admin Control System**: Provides user management, analytics, and system health monitoring.
- **Referral Reward System**: Manages cash referral tracking and payouts.

### Technical Implementations
- **Hybrid Data Source Architecture**: Uses Binance Futures for technical analysis data and Bitunix for tickers and trade execution.
- **Database**: PostgreSQL with SQLAlchemy ORM.
- **Configuration**: `pydantic-settings` for environment variables.
- **Security**: Fernet encryption for API credentials, HMAC-SHA256, and bearer token verification.
- **Risk Management**: Percentage-based SL/TP with price-level validation, risk-based position sizing, daily loss limits, max drawdown protection, auto breakeven stop loss, liquidity checks, anti-manipulation filters, and new coin protection. Incorporates proportional scaling to cap max profit/loss at 80% regardless of leverage.
- **Analytics**: Tracks signal outcomes, win/loss ratios, and asset performance.
- **Health Monitoring**: Automatic checks, process restarts, and error handling.
- **Price Caching**: Thread-safe global price cache with 30-second TTL.
- **Multi-Analysis Confirmation**: Validates signals against higher timeframes (1H) for trend alignment.
- **NOWPayments Subscription System**: Integrated for crypto payment subscriptions with webhook-based auto-activation and access control.
- **Referral Tracking**: Database tracks unique referral codes, referred users, and referral credits.
- **Advanced Market Analysis**: Integrates Bitunix funding rate analysis and order book depth analysis for improved signal quality and confidence scoring.
- **SHORTS Cooldown System**: Prevents re-shorting symbols immediately after a stop-loss.
- **Parallel Trade Execution**: Utilizes asyncio.Semaphore for efficient, nearly simultaneous execution across multiple users.

### UI/UX Decisions
- Interactive Telegram dashboard with inline buttons.
- Clear HTML formatting for messages.
- Built-in help center.
- Auto-generation of professional, shareable trade screenshots.
- Simplified navigation with core buttons and consolidated menus.

## External Dependencies
- **Telegram Bot API**: `aiogram` library.
- **Cryptocurrency Exchanges**: `CCXT` library for Bitunix and Binance Futures API for data.
- **Database**: PostgreSQL.
- **Payment Gateway**: NOWPayments API.
- **Sentiment Analysis**: OpenAI API.
- **News Aggregation**: CryptoNews API.