# Crypto Perps Signals Telegram Bot

## Overview
A Python-based Telegram bot that generates and broadcasts cryptocurrency perpetual futures trading signals using an EMA crossover strategy combined with support/resistance analysis, volume confirmation, RSI filtering, and ATR-based stops. The bot offers free access to signals, PnL tracking, user customization, and an integrated auto-trading system with encrypted credential storage across multiple exchanges. It incorporates multi-timeframe analysis, a risk assessment system, live position tracking, and an admin control system. The project aims to provide high-quality trading opportunities and automated execution, leveraging AI for news-based signals and comprehensive analytics for performance tracking.

## Recent Changes (October 2025)
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
- **Signal Generator**: Utilizes a 15-minute scalping strategy with EMA crossover, volume confirmation, RSI confluence, and trend strength validation. It includes ATR-based dynamic stops and quick scalping targets. Signals are confirmed through a multi-analysis system requiring 2-of-4 checks on a 1-hour timeframe (trend alignment, volume, momentum, price structure).
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
- **Risk Management**: Dynamic ATR-based SL/TP, risk-based position sizing, daily loss limits, and max drawdown protection.
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