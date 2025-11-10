# Crypto Perps Signals Telegram Bot

## Overview
This project is a Python-based Telegram bot designed for crypto perpetual trading with automated execution on the Bitunix exchange. It features two independent trading modes: SHORTS (mean reversion on 35%+ exhausted pumps) and LONGS (early momentum entries on 5-30% fresh pumps). The bot currently uses only a "Top Gaugers" scanning mode. Core strategies involve momentum-based entries with customizable leverage (1-20x), dual/triple take-profit targets, and breakeven stop-loss management. The project includes a 3-tier subscription model (Scan Mode, Manual Signals, Auto-Trading) and a cash referral system for Auto-Trading subscriptions.

## Recent Changes (Nov 10, 2025)
- **LONGS Ultra-Fresh Filter**: Changed to 10-30% range + pump must be within the hour
  - Min raised from 5% to 10% (stronger momentum confirmation)
  - Max lowered from 200% to 30% (prevents late entries on exhausted pumps)
  - NEW: 1H candle check ensures pump happened in last 60 minutes (super fresh!)
  - Prevents entering coins like TA/USDT at +85% (stale pump)
  - SHORTS still handle exhausted pumps (35%+) for mean reversion
- **Parallel Trade Execution**: Implemented asyncio.Semaphore(3) for 20x faster execution
  - 10 users: ~5 seconds (was 9 minutes)
  - 20 users: ~10 seconds (was 19 minutes)
  - All users get nearly identical entry prices
- **Leverage Button Fixed**: Added FSMContext parameter enabling interactive leverage changes (1-20x)
- **SHORTS Display Improved**: Capped TP/SL display at 80% max for clarity (actual leverage still executes)

## Previous Changes (Nov 9, 2025)
- **Scanner Speed**: Increased Top Gainers scanner from 5-min to 3-min intervals (20 scans/hour)
- **CRITICAL BUG FIX**: Fixed liquidity check that was rejecting ALL SHORTS signals
  - Issue: Code was looking for bid/ask prices that don't exist in Bitunix API response
  - Solution: Removed bid/ask validation, now relies on 24h volume for liquidity check
  - Impact: SHORTS signals will now generate properly (previously 100% rejection rate)
- **Volume Threshold**: Lowered from $1M to $400k minimum to catch more altcoin pumps
  - Enables scanning smaller caps that frequently pump 25-100%
  - Still maintains safe liquidity for trade execution
- **SHORTS Entry Threshold**: Raised from 25% to 35% pump to avoid premature entries
  - User feedback: 25% entries hit SL, but 2nd entries at higher levels caught 120-180% profits
  - New 35% threshold waits for exhausted parabolic moves instead of ongoing momentum
- **LONGS Filters Relaxed**: Removed overly strict filters blocking valid momentum trades
  - Removed oversized candle check (big green candles are GOOD for momentum!)
  - Widened RSI range: 35-85 (from 40-75) to catch high-momentum pumps
  - Lowered volume requirement: 1.0x (from 1.3x) - must match average volume minimum
  - Made pullback pattern optional (strong pumps no longer require retracement)
  - Widened EMA distance: ≤8% (from ≤5%) for more entry flexibility

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
- **Telegram Bot**: Manages user interaction, commands, signal broadcasting, and an interactive dashboard for live trading.
- **FastAPI Server**: Provides health checks and webhook endpoints.
- **Bitunix Auto-Trading System**: Handles automated live trade execution on Bitunix Futures with configurable leverage and risk management.
- **Top Gainers Trading Mode**: Supports SHORTS_ONLY (mean reversion on exhausted 35%+ pumps with single TP: 8%) and LONGS_ONLY (early momentum entries on 5-30% fresh pumps with dual TPs: 5%/10%) or BOTH.
- **Volume Surge Detector**: Real-time detection of volume spikes and early price movement for early entry opportunities.
- **New Coin Alerts**: Automated detection of newly listed, high-volume coins on Bitunix with descriptive analysis.
- **Admin Control System**: Provides user management, analytics, and system health monitoring.
- **Referral Reward System**: Manages cash referral tracking and payouts for Auto-Trading subscriptions.

### Technical Implementations
- **Hybrid Data Source Architecture**: Uses Binance Futures public API for technical analysis data and Bitunix for tickers and trade execution.
- **Database**: PostgreSQL with SQLAlchemy ORM.
- **Configuration**: `pydantic-settings` for environment variables.
- **Security**: Fernet encryption for API credentials, HMAC-SHA256, and bearer token verification.
- **Risk Management**: Percentage-based SL/TP with price-level validation, risk-based position sizing, daily loss limits, max drawdown protection, and auto breakeven stop loss. LONGS: TP1 at 5% price move (25% @ 5x), TP2 at 10% (50% @ 5x). SHORTS: Single TP at 8% price move (40% @ 5x). Incorporates liquidity checks, anti-manipulation filters (volume distribution, wick ratio), and new coin protection.
- **Analytics**: Tracks signal outcomes, win/loss ratios, and asset performance.
- **Health Monitoring**: Automatic checks, process restarts, and error handling.
- **Price Caching**: Thread-safe global price cache with 30-second TTL.
- **Multi-Analysis Confirmation**: Validates signals against higher timeframes (1H) for trend alignment.
- **NOWPayments Subscription System**: Integrated for crypto payment subscriptions with webhook-based auto-activation and access control.
- **Referral Tracking**: Database tracks unique referral codes, referred users, and referral credits.

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