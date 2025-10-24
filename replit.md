# Crypto Perps Signals Telegram Bot

## Overview
A Python-based Telegram bot designed to generate and broadcast cryptocurrency perpetual futures trading signals. It employs a sophisticated **hybrid signal system** with category-based targets, integrating EMA crossover swing strategy, reversal patterns, funding rate extremes, MACD/RSI divergence, AI news sentiment, and multi-exchange spot pressure analysis. The bot features a precision entry system, session quality filtering, a smart exit system, and automated execution on **Bitunix exchange**. It offers free signals, PnL tracking, paper trading, and comprehensive risk management for both scalp and swing trades. The project aims to provide high-quality, actionable trading insights and automated execution capabilities to users.

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
- **Hybrid Signal System**: Multi-category signal generator with dynamic targets:
  - **SCALP Signals**: 10%/15%/20% TPs, 12% SL for quick mean reversion (1-6 hour holds). Triggers on extreme funding rates.
  - **SWING Signals**: 15%/30%/50% TPs, 20% SL for multi-day trend plays. Includes EMA crossovers, MACD/RSI divergence, and a 5-pattern reversal scanner.
  - **Session Quality Filter**: 4-tier time-of-day analysis adjusts signal confidence based on market liquidity.
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