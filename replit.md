# Crypto Perps Signals Telegram Bot

## Overview
This project is a Python-based Telegram bot designed for crypto perpetual trading with automated execution on Bitunix exchange. **CRITICAL UPDATE (Nov 7, 2025):** The bot NOW ONLY uses Top Gainers mode - technical analysis signals have been completely disabled. Features two independent trading modes: SHORTS (mean reversion on 25%+ pumps) and LONGS (pump retracement entries on 5-200%+ gains). Each user can independently enable/disable SHORTS, LONGS, or BOTH via the dashboard. Core strategy: Momentum-based entries with 5x leverage (customizable 1-20x), dual/triple take-profit targets (1:1, 1:2, 1:3 R:R), and breakeven stop-loss management.

## Recent Changes (Nov 8, 2025)
- **REMOVED:** Paper trading completely eliminated from codebase - bot now operates in live trading mode only for simplicity and focus
  - Deleted PaperTrade model and all related UI/logic from bot.py (91+ references removed)
  - Dropped paper_trades table and UserPreference paper columns from database
  - Simplified dashboard, PnL reports, and analytics to show only live trades
  - Position monitor grace period reduced from 15min → 2min for faster TP/SL sync
- **PERFORMANCE:** Top Gainers scanner now runs every 5 minutes (was 10 min) - 2x faster signal detection! ⚡
  - Catches SHORTS 5 min earlier on parabolic reversals
  - Catches LONGS 5 min earlier on fresh pumps → better entry prices
  - More trading opportunities without overwhelming API limits
- **FIXED:** Dashboard auto-trading status bug - now correctly refreshes preferences from database (SQLAlchemy relationship caching issue)
- **RELAXED:** LONG entry criteria to generate more signals while maintaining quality:
  - EMA9 distance: 0.5% → 1.5% (slightly above EMA9 OK)
  - Volume requirement: 2.0x → 1.8x (more realistic)
  - RSI upper limit: 65 → 70 (allows stronger momentum)
- **UX:** Massively simplified navigation - from 40+ commands to 6 core buttons (2-level max)
  - New unified menus: Auto-Trading, Top Gainers, Settings all in single screens
  - Consolidated redundant commands (3 auto-trading commands → 1 menu)
  - Referrals visible on dashboard as requested
  - Advanced features accessible via "Advanced Settings" button
- **NEW:** 3-Tier Ultra-Early Detection System for LONGS - catches pumps 15-25 minutes earlier!
  - ⚡ TIER 1 (5m): 5%+ pump, 3x volume, <7min fresh → Ultra-early (5-10 min into pump)
  - 🔥 TIER 2 (15m): 7%+ pump, 2.5x volume, <20min fresh → Early (15-20 min into pump)
  - ✅ TIER 3 (30m): 10%+ pump, 2x volume, <35min fresh → Fresh (25-30 min into pump)
- **IMPROVED:** Multi-tier validation with priority system (5m > 15m > 30m) - always gets freshest signals
- **DISABLED:** Technical analysis signals completely removed (no longer broadcasting)
- **FIXED:** Critical LONGS starvation bug - now generates BOTH SHORT and LONG signals independently when users want them
- **IMPROVED:** Per-user signal filtering - each user gets only signals matching their individual preference (shorts_only/longs_only/both)

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
- **Telegram Bot**: Manages user interaction, commands, signal broadcasting, and an interactive dashboard (live trading only).
- **FastAPI Server**: Provides health checks and webhook endpoints.
- **Bitunix Auto-Trading System**: Handles automated live trade execution on Bitunix Futures with configurable leverage and risk management.
- **Top Gainers Trading Mode**: Supports 3 modes - SHORTS_ONLY, LONGS_ONLY, or BOTH:
  - **SHORTS (Mean Reversion)**: Automated 24/7 system for volatile coins (25%+ daily gains minimum), prioritizing parabolic reversals (50%+) with fixed 5x leverage and triple TPs for parabolic dumps. Features triple entry paths: overextended shorts, strong dumps (immediate), resumption patterns (safe), and early reversals (5m bearish + 15m bullish).
  - **LONGS (Pump Retracement Entry)**: Catches pumping coins (5-200%+ gains, NO MAX CAP) but WAITS for retracement before entering. Key: NO CHASING - enters AFTER pullback to EMA9 or resumption patterns (green → red → green). Three entry types: EMA9 pullback (best), resumption pattern (safest), strong pump (rare). Uses dual TPs (1:1 and 1:2 R:R) at 5x leverage.
- **Volume Surge Detector**: Real-time detection of volume spikes (2x+ normal) with early price movement (5-20% gains). Catches pumps BEFORE they hit the 25% Top Gainers threshold. Scans every 3 minutes with trend quality validation and confidence scoring. Perfect for early entries before the main pump.
- **New Coin Alerts**: Automated detection of newly listed coins on Bitunix with high volume (scans every 5 minutes). Provides coin description from CoinGecko, volume/price stats, pump analysis (why it's moving), and category tags. Alerts only (not trade signals) for early opportunity awareness like COAI, ASTER, XPL.
- **News-Based Trading Signals**: AI-powered system leveraging CryptoNews API for market events and sentiment.
- **Admin Control System**: Provides user management, analytics, and system health monitoring.
- **Multi-Exchange Spot Market Monitor**: Tracks buying/selling pressure across major exchanges for flow alerts.
- **Referral Reward System**: Viral growth mechanism where users earn 14 days free for each successful referral who subscribes. Rewards automatically add to existing subscriptions or activate new ones. Each user gets a unique referral code (format: TH-XXXXXX) and shareable link.

### Technical Implementations
- **Hybrid Data Source Architecture**: Uses Binance Futures public API for technical analysis (candles) and Bitunix for tickers and trade execution due to Bitunix API issues.
- **Database**: PostgreSQL with SQLAlchemy ORM.
- **Configuration**: `pydantic-settings` for environment variables.
- **Security**: Fernet encryption for API credentials, HMAC-SHA256, and bearer token verification.
- **Risk Management**: Percentage-based SL/TP with price-level validation (no false notifications from leverage-amplified P&L), risk-based position sizing, daily loss limits, max drawdown protection, and auto breakeven stop loss.
- **Analytics**: Tracks signal outcomes, win/loss ratios, and asset performance.
- **Health Monitoring**: Automatic checks, process restarts, and error handling.
- **Price Caching**: Thread-safe global price cache with 30-second TTL.
- **Multi-Analysis Confirmation**: Validates signals against higher timeframes (1H) for trend alignment, volume, momentum, and price structure.
- **NOWPayments Subscription System**: Integrated for crypto payment subscriptions with webhook-based auto-activation and access control.
- **Referral Tracking**: Database tracks referral_code (unique per user), referred_by (who referred them), and referral_credits (number of successful referrals). Rewards auto-apply as 14-day extensions when referrals subscribe.

### UI/UX Decisions
- Interactive Telegram dashboard with inline buttons.
- Clear HTML formatting for messages.
- Built-in help center.
- Auto-generation of professional, shareable trade screenshots for marketing.

## External Dependencies
- **Telegram Bot API**: `aiogram` library.
- **Cryptocurrency Exchanges**: `CCXT` library for Bitunix and Binance Futures API for data.
- **Database**: PostgreSQL.
- **Payment Gateway**: NOWPayments API.
- **Sentiment Analysis**: OpenAI API.
- **News Aggregation**: CryptoNews API.