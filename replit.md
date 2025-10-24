# Crypto Perps Signals Telegram Bot

## Overview
A Python-based Telegram bot designed to generate and broadcast cryptocurrency perpetual futures day trading signals with **1:1 risk-reward ratio**. It employs a strict **6-point confirmation system** requiring trend alignment, spot buying/selling pressure from Binance + 3 exchanges, volume spikes, momentum confirmation, clean candle patterns, and high liquidity session validation. Signals feature **15% TP / 15% SL** (1.5% price move with 10x leverage) for consistent, high-probability entries with automated execution on **Bitunix exchange**. The bot offers free signals, PnL tracking, paper trading, and comprehensive risk management for day trades only. The project aims to provide pinpoint entries with high success rates.

## Recent Changes (Oct 24, 2025)
- **🎯 1:1 DAY TRADING STRATEGY OVERHAUL**: Completely redesigned signal system for EARLY entries (not late confirmations). NEW FEATURES:
  - **FAST 5m+15m Timeframes**: Switched from 15m+1H to 5m+15m for entries BEFORE moves happen (not after 2-3% already moved)
  - **6-Point EARLY Confirmation System**: (1) EMA 9>21 crossover on 5m+15m (looser), (2) **SPOT FLOW PRIORITY >75%** - Institutional buying/selling from Binance+3 exchanges is HIGHEST PRIORITY, (3) Volume BUILDING >1.3x (not waiting for 2x spike), (4) MACD just starting to turn (not fully crossed), (5) Candle body forming (not full pattern), (6) High liquidity session only (8am-11pm UTC)
  - **Smart Money Priority**: Spot flow increased from 60% to 75% threshold - only trades when institutions are heavily buying/selling (not retail noise)
  - **1:1 Risk-Reward**: Single 15% TP / 15% SL (1.5% price move @ 10x leverage) - no more multi-TP complexity
  - **Smart Exit System**: Active monitoring with 6 reversal detectors (EMA crossover, candle patterns, RSI divergence, volume spikes, profit protection) closes positions early when market reverses
  - **PREDICTIVE Not Confirmatory**: Enters at FIRST signs of reversal, not after trend is established
  - **Removed Underperforming Patterns**: Deleted DOUBLE_TOP (0% win rate), funding extremes, weak divergence, and all reversal patterns
- **Pattern Performance Analytics**: Added `/pattern_performance` admin command to track win rate, avg PnL, and trade count per pattern. Database analysis revealed SHORTs only profited from manual early exits, LONGs had 0% win rate - confirming need for complete strategy rebuild.

## User Preferences
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
- **Day Trading Signal System**: 1:1 risk-reward generator with EARLY entry signals:
  - **Single TP/SL**: 15% TP / 15% SL (1.5% actual price move @ 10x leverage) for clean, consistent exits
  - **6-Point EARLY Entry System**: (1) Trend - EMA 9>21 on 5m + 15m (FAST), (2) **SPOT FLOW PRIORITY** - Binance + exchanges >75% institutional-grade buying/selling pressure (HIGHEST PRIORITY), (3) Volume - >1.3x building (EARLY, not 2x spike), (4) Momentum - MACD just turning + RSI 35-65, (5) Candle - Body forming with decent size (EARLY, not full pattern), (6) Session - Only 8am-11pm UTC high liquidity hours
  - **Smart Money First**: Institutional spot flow is checked first and requires >75% confidence (not retail 60%) - if institutions aren't aligned, other checks are skipped
  - **Predictive Entry**: Catches moves BEFORE they run 2-3%, not confirming after
  - **Fast Timeframes**: 5m primary + 15m confirmation (was 15m+1H which lagged too much)
  - **Smart Exit Protection**: Automated reversal detection with 6 methods (EMA crossover, candle patterns, RSI divergence, volume spikes, profit locking) actively monitors all open positions
  - **Quality Over Quantity**: Far fewer signals but entered at optimal prices with institutional backing
- **Precision Entry System**: Advanced entry refinement using 3-pattern candle detection (Engulfing, Hammer/Shooting Star, Strong Body) and intelligent entry price optimization.
- **Reversal Bounce Catcher**: Multi-pattern scanner detecting early breakout signals: Support/Resistance Bounces, Bollinger Squeeze Breakouts, Double Bottom/Top Reversals, RSI Divergence Reversals, and Volume Spike Reversals.
- **Telegram Bot**: Handles user interaction, command processing, signal broadcasting, and an interactive dashboard with PnL calculations.
- **FastAPI Server**: Provides health checks and webhook endpoints.
- **Bitunix Auto-Trading System**: Automated execution exclusively on Bitunix Futures with configurable leverage, smart exit system, adaptive sizing, and risk management.
- **News-Based Trading Signals**: AI-powered system monitoring CryptoNews API for market-moving events and sentiment analysis.
- **Admin Control System**: Features user management, analytics (DAU/WAU/MAU, signal performance), and system health monitoring.
- **Paper Trading System**: Provides a simulated trading environment for users.
- **Backtesting System**: Admin-only feature for strategy validation.
- **Multi-Exchange Spot Market Monitor**: Monitors buying/selling pressure across major exchanges for high-conviction flow alerts.

### Technical Implementations
- **Database**: PostgreSQL with SQLAlchemy ORM.
- **Configuration**: `pydantic-settings` for environment variables.
- **Security**: Fernet encryption for API credentials, HMAC-SHA256 and bearer token verification for webhooks.
- **Risk Management**: Percentage-based SL/TP (SCALP: 12% SL, 10%/15%/20% TPs; SWING: 20% SL, 15%/30%/50% TPs), risk-based position sizing, daily loss limits, max drawdown protection, and automatic breakeven stop loss.
- **Analytics**: Tracks signal outcomes, win/loss ratios, and asset performance.
- **Correlation Filter**: Prevents opening correlated positions by grouping assets.
- **Funding Rate Monitor**: Monitors funding rates for arbitrage opportunities.
- **Memory Management**: Uses async `ccxt` with proper connection closures.
- **Health Monitoring & Auto-Recovery**: Automatic health checks, process restarts, and graceful error handling.
- **Centralized Error Handling**: Production-grade error logging with retry mechanisms.
- **Price Caching**: Thread-safe global price cache with 30-second TTL.
- **Multi-Analysis Confirmation**: Advanced validation system checks signals against higher timeframes (1H) for trend alignment, volume, momentum, and price structure.

### UI/UX Decisions
- Interactive Telegram dashboard with inline buttons.
- Clear HTML formatting for signals and dashboard messages.
- Built-in help center.

## External Dependencies
- **Telegram Bot API**: `aiogram` library.
- **Cryptocurrency Exchanges**: `CCXT` library for Bitunix.
- **Database**: PostgreSQL.
- **Sentiment Analysis**: OpenAI API.
- **News Aggregation**: CryptoNews API.