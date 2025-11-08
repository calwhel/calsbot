# Crypto Perps Signals Telegram Bot

## Overview
This project is a Python-based Telegram bot designed for crypto perpetual trading with automated execution on the Bitunix exchange. It features two independent trading modes: SHORTS (mean reversion on 25%+ pumps) and LONGS (pump retracement entries on 5-200%+ gains). The bot currently uses only a "Top Gainers" scanning mode. Core strategies involve momentum-based entries with customizable leverage (1-20x), dual/triple take-profit targets, and breakeven stop-loss management. The project includes a 3-tier subscription model (Scan Mode, Manual Signals, Auto-Trading) and a cash referral system for Auto-Trading subscriptions.

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
- **Top Gainers Trading Mode**: Supports SHORTS_ONLY (mean reversion on volatile coins with single TP: 8%) and LONGS_ONLY (pump retracement entry with 3-tier ultra-early detection and dual TPs: 5%/10%) or BOTH.
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