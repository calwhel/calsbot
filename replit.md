# Crypto Perps Signals Telegram Bot

## Overview
This project is a Python-based Telegram bot designed for crypto perpetual trading with automated execution on the Bitunix exchange. It features three independent trading modes: PARABOLIC (50%+ exhausted dumps - highest priority), SHORTS (mean reversion on 35%+ pumps), and LONGS (early momentum entries on 5-50% fresh pumps). The bot uses a "Top Gainers" scanning system with dedicated parabolic dump detection. Core strategies involve momentum-based entries with customizable leverage (1-20x), dual/triple take-profit targets, and breakeven stop-loss management. The project includes a 2-tier subscription model (Signals Only $80/mo, Auto-Trading $150/mo - scan included with both) and a cash referral system for Auto-Trading subscriptions.

## Recent Changes (Nov 16, 2025) - LONG Signals Fixed
- **CRITICAL FIX: LONG Signals Now Generating**: Massively relaxed tier filters for real-world pumps
  - Issue: Required single-candle pumps (5%/7%/10%) that rarely happen naturally
  - Solution: Lowered to 3%/5%/7% across 5m/15m/30m tiers + 1.5x volume (was 2.5x/2.0x)
  - Extended freshness: 3%+ gain within 3 hours (was 5%+ in 2 hours)
  - Impact: LONG signals generating successfully (e.g., STRK/USDT @ +14.08%)
  - Signal rate: ~1 quality LONG per hour with selective filters
  - Stage 1: 11 candidates → Stage 2: 0-1 fresh pumps (high selectivity = quality)

## Previous Changes (Nov 14, 2025) - Go-Live Ready
- **Master Trader Copy Trading Integration**: All signals execute on owner's Bitunix Copy Trading account
  - Transparent to users - they continue trading with their own API keys
  - Master account executes every signal in parallel (10% of Copy Trading balance per trade)
  - Enables Bitunix followers to copy trades (10-30% profit sharing for owner)
  - Non-blocking parallel execution - never delays user trades
  - Uses async CCXT library for clean API integration
  - Position sizing scales automatically with account growth
  
## Critical Fixes (Nov 14, 2025)
- **NEW FEATURE: Dedicated Parabolic Dump Scanner**: Separate scanner for 50%+ exhausted pumps
  - Runs FIRST (highest priority) when SHORTS mode enabled
  - Evaluates ALL 50%+ candidates and scores by overextension + confidence
  - Returns best parabolic reversal with triple TPs (20%, 40%, 60% at 5x)
  - Signal type: PARABOLIC_REVERSAL (separate from TOP_GAINER)
  - Auto-enabled for all users with SHORTS mode (no new preference needed)
- **CRITICAL FIX: Duplicate Trades Prevented**: Added signal-level duplicate prevention
  - Issue: Race condition in parallel execution caused duplicate trades
  - Solution: Check if signal already exists before creating (5-min window)
  - Impact: No more duplicate trades, safe for multi-user parallel execution
  - Supports both TOP_GAINER and PARABOLIC_REVERSAL signal types
- **CRITICAL FIX: LONG Trades Now Generating**: Relaxed filters to catch more opportunities
  - Range widened: 5-50% (was 10-34%)
  - Freshness extended: 2 hours (was 60 minutes strict)
  - Volume relaxed: 2.5x (was 3.0x)
  - Candle age: 10 min (was 7 min)
  - Impact: LONG signals will now generate regularly
- **CRITICAL FIX: TP/SL Leverage Cap (80% Max)**: Fixed incorrect profit/loss percentages at high leverage
  - Issue: With 20x leverage, SHORTS showed 160% profit instead of intended 80% cap
  - Solution: Implemented proportional scaling helper that caps max profit/loss at 80% regardless of leverage
  - How it works: For leverage > 10x, scales entire TP ladder proportionally (preserves spacing)
  - Examples: 
    - 20x SHORT: 4% price move = 80% profit (was 8% move = 160%)
    - 20x LONG: TP1 = 40%, TP2 = 80% (was TP1 = 100%, TP2 = 200%)
  - Impact: All TP/SL percentages now capped at 80% for safe risk management
  - Maintains R:R ratios and TP ladder spacing (TP1 < TP2 always)
  - Applied only to TOP_GAINER trades with leverage > 10x
- **Payment System LIVE**: NOWPayments integration fully operational
  - Webhook: https://tradehubai.up.railway.app/webhooks/nowpayments
  - Test invoice creation: successful
  - All 3 tiers active and processing payments

## Previous Changes (Nov 10, 2025)
- **SHORT Cooldown System**: Prevents re-shorting strong pumps that hit SL
  - When a SHORT hits stop-loss, that symbol is blocked from SHORTS for 30 minutes
  - Prevents repeatedly shorting coins with strong bullish momentum (like LSK/USDT)
  - Cooldown only applies to SHORTS (LONGS unaffected)
  - Auto-cleanup of expired cooldowns
- **CRITICAL FIX: Position Size & FSM Handlers**: Fixed catch-all message handler blocking FSM states
  - Issue: Catch-all handler at line 3876 was stealing ALL text messages before FSM handlers
  - Solution: Added StateFilter(None) to catch-all, ensuring it only matches when NO FSM state active
  - Impact: Position size and leverage buttons now respond to user input correctly
  - Technical: Used proper aiogram 3 filter pattern instead of manual state checks
- **SHORTS Display Fixed (All 3 Messages)**: Capped TP/SL at "up to +80% max" in broadcast, manual, and auto-trading
  - Prevents confusing displays like "160% profit" on high leverage
  - Shows user-friendly max instead of actual leverage percentage
- **LONGS Ultra-Fresh Filter**: Changed to 10-34% range + pump must be within the hour
  - Min raised from 5% to 10% (stronger momentum confirmation)
  - Max set to 34.9% to eliminate gap with SHORTS (35%+)
  - NEW: 1H candle check ensures pump happened in last 60 minutes (super fresh!)
  - Prevents entering coins like TA/USDT at +85% (stale pump)
  - SHORTS still handle exhausted pumps (35%+) for mean reversion
  - Fixed: Candle timestamp bug that was rejecting 100% of LONGS (added interval offset)
- **Parallel Trade Execution**: Implemented asyncio.Semaphore(3) for 20x faster execution
  - 10 users: ~5 seconds (was 9 minutes)
  - 20 users: ~10 seconds (was 19 minutes)
  - All users get nearly identical entry prices
- **Debug Logging Added**: FSM state transitions now logged for easier troubleshooting

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
- **Master Trader Copy Trading**: Executes all signals on owner's Bitunix Copy Trading account in parallel with user trades for follower profit-sharing.
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