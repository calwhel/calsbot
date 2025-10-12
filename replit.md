# Crypto Perps Signals Telegram Bot

## Overview
A Python-based Telegram bot that generates and broadcasts cryptocurrency perpetual futures trading signals. It utilizes an EMA crossover strategy combined with support/resistance analysis, volume confirmation, RSI filtering, and ATR-based stops. The bot offers free access to signals, PnL tracking, user customization, and an integrated MEXC auto-trading system with encrypted credential storage. It also incorporates multi-timeframe analysis, a risk assessment system, live position tracking, and an admin control system for managing user access. Recent enhancements include 10x leverage PnL calculations, duplicate signal prevention, news-based trading signals, a comprehensive support system, MEXC API testing tools, signal performance analytics, paper trading mode, and a backtesting system.

## User Preferences
Users can customize their experience through settings:
- **Muted Symbols**: Disable signals for specific pairs
- **Default PnL Period**: Choose default view (today/week/month)
- **DM Alerts**: Toggle private message notifications
- **Position Sizing**: Users can set position size as percentage of account balance
- **Max Positions**: Limit simultaneous open positions
- **Risk Filtering**: Customizable accepted risk levels for signals
- **Paper Trading Mode**: Toggle between paper/live modes with `/toggle_paper_mode`

## System Architecture

### Core Components
The application runs a Telegram bot (using `aiogram`) and a FastAPI server within a single process. Key services include:
- **Signal Generator**: Fetches OHLCV data via CCXT, performs multi-timeframe (1h, 4h) EMA crossover analysis with volume confirmation, RSI filtering, ATR-based dynamic stops, and support/resistance identification. Signals are risk-filtered (LOW/MEDIUM risk only).
- **Telegram Bot**: Handles user commands (`/start`, `/dashboard`, `/settings`, `/subscribe`, `/set_mexc_api`, etc.), broadcasts signals, manages user preferences, and provides an interactive dashboard with 10x leverage PnL calculations.
- **FastAPI Server**: Provides health checks and webhook endpoints for potential payment integrations (Whop, Solana).
- **MEXC Auto-Trading System**: Integrates with MEXC exchange for automated trade execution, managing position sizing, max positions, and automatic SL/TP placement with 10x leverage.
- **News-Based Trading Signals**: AI-powered system monitors CryptoPanic, analyzes sentiment with OpenAI, and generates signals from high-impact market-moving events.
- **Admin Control System**: A private bot setup with user approval, ban/unban functionality, an admin dashboard, and user statistics. The first user automatically becomes an admin.
- **Paper Trading System**: Provides a risk-free virtual trading environment with a simulated balance and full auto-trading simulation.
- **Backtesting System**: An admin-only feature for testing the EMA crossover strategy on historical data.

### Technical Implementations
- **Database**: PostgreSQL with SQLAlchemy ORM for managing users, signals, trades, and preferences. Tables are auto-initialized on startup.
- **Configuration**: `pydantic-settings` for environment variable management.
- **Security**: Fernet encryption for securely storing API credentials at rest, with decryption occurring only in-memory during use. HMAC-SHA256 and bearer token verification for webhooks.
- **Risk Management**: Dynamic ATR-based SL/TP, risk-based position sizing, daily loss limits, max drawdown protection, and minimum balance checks.
- **Analytics**: Comprehensive signal performance analytics tracking outcomes, win/loss ratios, and best performing assets.

### UI/UX Decisions
- Interactive Telegram dashboard with inline buttons for navigation and controls.
- Clear HTML formatting for signal broadcasts and dashboard views.
- Built-in help center for user guidance.

## External Dependencies
- **Telegram Bot API**: `aiogram` library for bot interaction.
- **Cryptocurrency Exchanges**: `CCXT` library for fetching market data and executing trades (primarily MEXC).
- **Database**: PostgreSQL.
- **Payment Gateways (Optional)**: Whop, Solana Pay (via Helius integration).
- **Sentiment Analysis**: OpenAI API.
- **News Aggregation**: CryptoPanic.