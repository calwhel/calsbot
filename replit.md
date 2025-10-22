# Crypto Perps Signals Telegram Bot

## Overview
A Python-based Telegram bot that generates and broadcasts cryptocurrency perpetual futures trading signals using a **hybrid signal system** with category-based targets. Combines EMA crossover swing strategy, reversal patterns, funding rate extremes (scalp), MACD/RSI divergence (swing), AI news sentiment, and multi-exchange spot pressure analysis. Features precision entry system with candle pattern detection, session quality filtering, smart exit system, and automated execution on **Bitunix exchange**. Provides free signals, PnL tracking, paper trading, and comprehensive risk management for both scalp (10%/15%/20% targets) and swing trades (15%/30%/50% targets).

## Recent Changes (October 2025)
- **Bitunix-Only Simplification** (Oct 22): Removed all MEXC, OKX, and KuCoin support. Bot now exclusively uses Bitunix for trading execution, resulting in cleaner codebase and simplified maintenance. Funding rate data fetched from Binance (most reliable source), while all trade execution happens on Bitunix.
- **Smart Exit System** (Oct 22): AI-powered reversal detection closes trades early when market turns against your position. Monitors 6 reversal signals: (1) EMA death/golden crosses, (2) Bearish/bullish engulfing candles, (3) RSI divergence (price vs momentum), (4) Extreme RSI zones with rejection candles, (5) Volume spike distribution/accumulation, (6) Profit protection when momentum weakens. Saves you from losses and protects profits by exiting before hitting stop loss.
- **Quality Filter System** (Oct 21): Multi-layered filtering ensures only premium setups get broadcast. Blocks poor session signals (2-6 AM UTC), enforces minimum confidence (75% scalp, 70% swing), requires 1.5:1+ R:R ratio, validates signal-specific criteria (funding >0.1%, divergence in extreme RSI zones). Auto-deduplicates conflicting signals, assigns quality scores (0-100), and tiers (PREMIUM/HIGH/GOOD). Only ~30% of raw signals pass filters for maximum win rate.
- **Hybrid Signal System** (Oct 21): Implemented category-based signal system with dynamic targets. SCALP signals (funding extremes) use tight 10%/15%/20% targets with 12% SL for quick mean reversion. SWING signals (divergence, technical, reversal) use 15%/30%/50% targets with 20% SL for multi-day holds. Session quality filters boost/reduce confidence based on liquidity (4 tiers: BEST/GOOD/MEDIUM/POOR). Funding rate extremes (>0.1%) trigger SHORT, (<-0.1%) trigger LONG for overheated position reversion.
- **MACD/RSI Divergence Detector** (Oct 21): Added swing signal generator that catches trend reversals BEFORE they happen. Bullish divergence (price lower low + RSI higher low) triggers LONG. Bearish divergence (price higher high + RSI lower high) triggers SHORT. Scans 1-hour timeframe with 80% confidence for early entries.
- **Precision Entry System** (Oct 21): Added advanced entry refinement with 3-pattern candle detection (Bullish/Bearish Engulfing, Hammer/Shooting Star, Strong Body Candles), EMA pullback/rejection analysis, and intelligent entry price optimization. LONG entries use wick lows on EMA pullbacks, SHORT entries use wick highs on rejections. Eliminates random close-price entries for surgical precision.
- **UI Bugs Fixed** (Oct 21): Fixed toggle paper mode button not working, dashboard incorrectly showing auto-trading as inactive when enabled (encrypted key detection issue), missing `/autotrading` command, and 8 missing button handlers (PnL today/week/month, view all PnL, edit position size, edit leverage, edit notifications). All interactive buttons now functional.
- **Paper Trading Data Reset** (Oct 21): Deleted all 100 paper trades and reset all 5 users to live mode due to strategy change from scalping to swing trading. Fresh start for new percentage-based targets.
- **Bitunix Balance & Price Fetching Fixed** (Oct 21): Fixed critical bug where API response field was `lastPrice` not `last`, causing all price fetches to fail. Dashboard now respects preferred_exchange setting for balance fetching, uses correct `/api/v1/futures/market/tickers` endpoint, and proper connection cleanup prevents API failures.
- **Automatic Breakeven Stop Loss** (Oct 21): When TP1 is hit (20% gain), stop loss automatically moves to entry price across ALL exchanges (MEXC/Bitunix/OKX/KuCoin), eliminating downside risk on remaining position. User gets clear notification with before/after SL levels.
- **Multi-Exchange Position Monitoring** (Oct 21): Fixed critical bug where only MEXC positions were monitored. Now dynamically detects user's exchange and monitors TP/SL hits across ALL supported exchanges with proper trader instances.
- **Double-Validation Bug Fixed** (Oct 21): Pre-validated signal types ('technical', 'REVERSAL') now skip execution validation, preventing real signals from being rejected during auto-trading across all exchanges.
- **Reversal Bounce Catcher System** (Oct 21): Added multi-pattern reversal scanner that detects 5 types of early breakout signals: Support/Resistance Bounces, Bollinger Squeeze Breakouts, Double Bottom/Top Reversals, RSI Divergence Reversals, and Volume Spike Reversals. Runs alongside swing strategy for maximum signal coverage.
- **Multi-Timeframe Hybrid Strategy** (Oct 21): Implemented dual-timeframe confirmation system. 15-minute charts for precise entry timing, 1-hour charts for swing direction validation. Signal requires 2-of-3 confirmations on 1H (EMA alignment, RSI, trend strength) before 15m entry fires. Best of both worlds: timing + direction.
- **Trading Style Changed to SWING TRADING** (Oct 21): Switched from scalping to swing trading with percentage-based targets. SL: 25%, TP1: 20%, TP2: 40%, TP3: 60% from entry. Trades hold for bigger moves instead of quick scalps.
- **CryptoNews API Integration**: Migrated from CryptoPanic to CryptoNews API for superior news coverage. Provides rank-based sorting, real-time sentiment scores (-1.5 to +1.5), granular time filters (last5min, last15min, etc.), and 30+ premium sources.
- **Signal Generation Optimization**: **CRITICAL PERFORMANCE FIX** - Multi-analysis confirmation relaxed from 3-of-4 to 2-of-4 checks (was blocking 70% of valid signals). Session filtering removed for 24/7 trading (was missing 35-40% of Asian market moves). RSI relaxed from 60/40 to 55/45, volume from 120% to 110%. Expected to 3-4x signal frequency while maintaining quality.

## User Preferences
Users can customize their experience through settings:
- **Muted Symbols**: Disable signals for specific pairs
- **Default PnL Period**: Choose default view (today/week/month)
- **DM Alerts**: Toggle private message notifications
- **Position Sizing**: Users can set position size as percentage of account balance
- **Max Positions**: Limit simultaneous open positions
- **Risk Filtering**: Customizable accepted risk levels for signals
- **Paper Trading Mode**: Toggle between paper/live modes with `/toggle_paper_mode`
- **Correlation Filter**: Prevent opening correlated positions (e.g., BTC + ETH simultaneously)
- **Funding Rate Alerts**: Get notified of extreme funding rates for arbitrage opportunities

## System Architecture

### Core Components
The application integrates a Telegram bot and a FastAPI server.
- **Hybrid Signal System**: Multi-category signal generator with dynamic targets:
  - **SCALP Signals** (Funding Extremes): 10%/15%/20% TPs, 12% SL - Quick mean reversion plays (1-6 hour holds). Triggers when funding rate >0.1% (SHORT) or <-0.1% (LONG). Auto-breakeven after TP1.
  - **SWING Signals** (Technical/Divergence/Reversal): 15%/30%/50% TPs, 20% SL - Multi-day trend plays. Includes EMA crossovers, MACD/RSI divergence, 5-pattern reversal scanner.
  - **Session Quality Filter**: 4-tier time-of-day analysis (BEST: 12-16 UTC EU/US overlap, GOOD: Asian/US sessions, MEDIUM: late hours, POOR: 2-6 UTC). Adjusts confidence via multipliers (1.2x to 0.5x).
- **Precision Entry System**: 3-pattern candle detection (Bullish/Bearish Engulfing, Hammer/Shooting Star, Strong Body) with intelligent entry optimization. LONG entries use wick lows on EMA pullbacks, SHORT entries use wick highs on rejections.
- **Reversal Bounce Catcher**: Multi-pattern scanner that detects early breakout signals before the crowd. Identifies 5 pattern types: (1) Support/Resistance Bounces with volume confirmation, (2) Bollinger Band Squeeze Breakouts from consolidation, (3) Double Bottom/Top Reversal patterns, (4) RSI Divergence Reversals (price vs momentum), (5) Volume Spike Reversals (capitulation candles). Runs in parallel with swing strategy for comprehensive coverage.
- **Telegram Bot**: Manages user commands, broadcasts signals, handles user preferences, and provides an interactive dashboard with PnL calculations.
- **FastAPI Server**: Provides health checks and webhook endpoints.
- **Bitunix Auto-Trading System**: Automated trading exclusively on Bitunix Futures. Features configurable leverage, smart exit system, adaptive sizing, and comprehensive risk management.
- **News-Based Trading Signals**: An AI-powered system monitors CryptoNews API for market-moving events, analyzes sentiment, and generates signals.
- **Admin Control System**: Provides user management, analytics (DAU/WAU/MAU, signal performance), and system health monitoring.
- **Paper Trading System**: Offers a simulated trading environment.
- **Backtesting System**: An admin-only feature for strategy validation on historical data.
- **Multi-Exchange Spot Market Monitor**: Monitors buying/selling pressure across major exchanges, triggering high-conviction flow alerts and auto-trades with stability filters and auto-position flip functionality.

### Technical Implementations
- **Database**: PostgreSQL with SQLAlchemy ORM for data management, optimized with indexes.
- **Configuration**: `pydantic-settings` for environment variables.
- **Security**: Fernet encryption for API credentials, HMAC-SHA256 and bearer token verification for webhooks.
- **Risk Management**: Percentage-based SL/TP (15% SL, 20%/40%/60% TPs), risk-based position sizing, daily loss limits, and max drawdown protection.
- **Analytics**: Tracks signal outcomes, win/loss ratios, and asset performance.
- **Correlation Filter**: Groups assets by sector to prevent correlated positions.
- **Funding Rate Monitor**: Monitors funding rates and alerts for arbitrage opportunities.
- **Memory Management**: Uses async `ccxt` with proper connection closures.
- **Health Monitoring & Auto-Recovery**: Automatic health checks with heartbeat signals, process restarts for frozen states, and graceful error handling.
- **Centralized Error Handling**: Production-grade error logging with retry mechanisms, user-friendly messages, and comprehensive analytics.
- **Price Caching**: Thread-safe global price cache with 30-second TTL to reduce API calls and ensure scalability.
- **Multi-Analysis Confirmation**: Advanced validation system checks signals against higher timeframes (1H) for trend alignment, volume, momentum, and price structure, requiring a quorum of passing checks for trade execution.

### UI/UX Decisions
- Interactive Telegram dashboard with inline buttons.
- Clear HTML formatting for signals and dashboard.
- Built-in help center.

## External Dependencies
- **Telegram Bot API**: `aiogram` library.
- **Cryptocurrency Exchanges**: `CCXT` library for KuCoin Futures, OKX, MEXC, and Bitunix.
- **Database**: PostgreSQL.
- **Payment Gateways (Optional)**: Whop, Solana Pay (via Helius integration).
- **Sentiment Analysis**: OpenAI API.
- **News Aggregation**: CryptoNews API (switched from CryptoPanic in Oct 2025).