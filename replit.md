# Crypto Perps Signals Telegram Bot

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