# Crypto Perps Signals Telegram Bot

## Overview
A Python-based Telegram bot that generates and broadcasts cryptocurrency perpetual futures trading signals using an EMA crossover strategy combined with support/resistance analysis, volume confirmation, RSI filtering, and ATR-based stops. The bot offers free access to signals, PnL tracking, user customization, and an integrated auto-trading system with encrypted credential storage across multiple exchanges. It incorporates multi-timeframe analysis, a risk assessment system, live position tracking, and an admin control system. The project aims to provide high-quality trading opportunities and automated execution, leveraging AI for news-based signals and comprehensive analytics for performance tracking.

## Recent Changes (October 2025)
- **UI Bugs Fixed** (Oct 21): Fixed toggle paper mode button not working, dashboard incorrectly showing auto-trading as inactive when enabled (encrypted key detection issue), missing `/autotrading` command, and 8 missing button handlers (PnL today/week/month, view all PnL, edit position size, edit leverage, edit notifications). All interactive buttons now functional.
- **Paper Trading Data Reset** (Oct 21): Deleted all 100 paper trades and reset all 5 users to live mode due to strategy change from scalping to swing trading. Fresh start for new percentage-based targets.
- **Bitunix Balance & Price Fetching Fixed** (Oct 21): Fixed critical bug where API response field was `lastPrice` not `last`, causing all price fetches to fail. Dashboard now respects preferred_exchange setting for balance fetching, uses correct `/api/v1/futures/market/tickers` endpoint, and proper connection cleanup prevents API failures.
- **Automatic Breakeven Stop Loss** (Oct 21): When TP1 is hit (20% gain), stop loss automatically moves to entry price across ALL exchanges (MEXC/Bitunix/OKX/KuCoin), eliminating downside risk on remaining position. User gets clear notification with before/after SL levels.
- **Multi-Exchange Position Monitoring** (Oct 21): Fixed critical bug where only MEXC positions were monitored. Now dynamically detects user's exchange and monitors TP/SL hits across ALL supported exchanges with proper trader instances.
- **Double-Validation Bug Fixed** (Oct 21): Pre-validated signal types ('technical', 'REVERSAL') now skip execution validation, preventing real signals from being rejected during auto-trading across all exchanges.
- **Reversal Bounce Catcher System** (Oct 21): Added multi-pattern reversal scanner that detects 5 types of early breakout signals: Support/Resistance Bounces, Bollinger Squeeze Breakouts, Double Bottom/Top Reversals, RSI Divergence Reversals, and Volume Spike Reversals. Runs alongside swing strategy for maximum signal coverage.
- **Multi-Timeframe Hybrid Strategy** (Oct 21): Implemented dual-timeframe confirmation system. 15-minute charts for precise entry timing, 1-hour charts for swing direction validation. Signal requires 2-of-3 confirmations on 1H (EMA alignment, RSI, trend strength) before 15m entry fires. Best of both worlds: timing + direction.
- **Trading Style Changed to SWING TRADING** (Oct 21): Switched from scalping to swing trading with percentage-based targets. SL: 15%, TP1: 20%, TP2: 40%, TP3: 60% from entry. Trades hold for bigger moves instead of quick scalps.
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
- **Signal Generator**: Multi-timeframe hybrid swing trading system. Scans 15-minute charts for precise entry timing, validates against 1-hour trend direction (requires 2-of-3 confirmations: EMA alignment, RSI, trend strength). Uses percentage-based targets (TP1: 20%, TP2: 40%, TP3: 60%) with 15% stop loss for larger moves. Additional validation through multi-analysis system requiring 2-of-4 checks on 1-hour timeframe (trend alignment, volume, momentum, price structure).
- **Reversal Bounce Catcher**: Multi-pattern scanner that detects early breakout signals before the crowd. Identifies 5 pattern types: (1) Support/Resistance Bounces with volume confirmation, (2) Bollinger Band Squeeze Breakouts from consolidation, (3) Double Bottom/Top Reversal patterns, (4) RSI Divergence Reversals (price vs momentum), (5) Volume Spike Reversals (capitulation candles). Runs in parallel with swing strategy for comprehensive coverage.
- **Telegram Bot**: Manages user commands, broadcasts signals, handles user preferences, and provides an interactive dashboard with PnL calculations.
- **FastAPI Server**: Provides health checks and webhook endpoints.
- **Multi-Exchange Auto-Trading System**: Supports automated trading on KuCoin Futures, OKX, MEXC, and Bitunix. Users can connect one exchange at a time. Features configurable leverage, dynamic trailing stops, adaptive sizing, and comprehensive risk management.
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