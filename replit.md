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
- **Correlation Filter**: Prevent opening correlated positions (e.g., BTC + ETH simultaneously)
- **Funding Rate Alerts**: Get notified of extreme funding rates for arbitrage opportunities

## System Architecture

### Core Components
The application runs a Telegram bot (using `aiogram`) and a FastAPI server within a single process. Key services include:
- **Signal Generator**: 15-minute scalping strategy optimized for 5-10 quality trades per day. Uses strict EMA crossover analysis with 120% volume confirmation, RSI filtering (30-70 range only), and 0.8% EMA separation for trend signals. ATR-based dynamic stops (1.5x ATR) with quick scalping targets (TP1: 0.8R, TP2: 1.2R, TP3: 1.5R). **Session-based trading**: Only generates signals during London (08:00-16:00 UTC) and US sessions (13:00-21:00 UTC), combined 08:00-21:00 UTC active hours. Stricter filters ensure higher quality setups with better win rates.
- **Telegram Bot**: Handles user commands (`/start`, `/dashboard`, `/settings`, `/subscribe`, `/set_kucoin_api`, `/set_okx_api`, `/set_mexc_api`, etc.), broadcasts signals, manages user preferences, and provides an interactive dashboard with 10x leverage PnL calculations.
- **FastAPI Server**: Provides health checks and webhook endpoints for potential payment integrations (Whop, Solana).
- **Multi-Exchange Auto-Trading System**: Advanced automated trading with support for KuCoin Futures (primary/recommended), OKX, and MEXC exchanges. Features configurable leverage (1-20x), dynamic trailing stops, win/loss adaptive sizing, smart R:R scaling, anti-overtrading filters, and market condition detection. Manages position sizing, max positions, and automatic SL/TP placement with comprehensive risk management. KuCoin Futures is the recommended exchange due to UK accessibility and reliable API performance. MEXC features enhanced connection reliability with 120s timeout, automatic retry with exponential backoff (3 attempts), connection pooling (10 connections), DNS caching, and granular socket-level timeouts for improved stability.
- **News-Based Trading Signals**: AI-powered system monitors CryptoPanic with 10-minute caching and 15-minute rate limit cooldown, analyzes sentiment with OpenAI, and generates signals from high-impact market-moving events.
- **Admin Control System**: A private bot setup with user approval, ban/unban functionality, an admin dashboard, and user statistics. The first user automatically becomes an admin.
- **Paper Trading System**: Provides a risk-free virtual trading environment with a simulated balance and full auto-trading simulation.
- **Backtesting System**: An admin-only feature for testing the EMA crossover strategy on historical data.
- **Multi-Exchange Spot Market Monitor**: Real-time monitoring of buying/selling pressure across 5 major exchanges (Binance, Coinbase, Kraken, Bybit, OKX) using order book imbalance analysis, trade flow detection, and volume spike identification. Automatically broadcasts high-conviction (70%+) flow alerts **and triggers auto-trades** (HEAVY_BUYING/VOLUME_SPIKE_BUY = LONG, HEAVY_SELLING/VOLUME_SPIKE_SELL = SHORT) with ATR-based SL/TP.

### Technical Implementations
- **Database**: PostgreSQL with SQLAlchemy ORM for managing users, signals, trades, and preferences. Tables are auto-initialized on startup. Performance optimized with indexes on frequently queried fields (trades, signals, paper trades, user preferences).
- **Configuration**: `pydantic-settings` for environment variable management.
- **Security**: Fernet encryption for securely storing API credentials at rest, with decryption occurring only in-memory during use. HMAC-SHA256 and bearer token verification for webhooks.
- **Risk Management**: Dynamic ATR-based SL/TP, risk-based position sizing, daily loss limits, max drawdown protection, and minimum balance checks.
- **Analytics**: Comprehensive signal performance analytics tracking outcomes, win/loss ratios, and best performing assets.
- **Correlation Filter**: Groups crypto assets by sector (BTC, ETH, LAYER1, LAYER2, DEFI, MEME, AI, GAMING) and prevents opening multiple correlated positions simultaneously.
- **Funding Rate Monitor**: Hourly monitoring of perpetual futures funding rates across exchanges. Alerts users when funding exceeds thresholds (default 0.1%), identifying arbitrage opportunities when longs/shorts are overleveraged. *Note: ccxt.async_support may emit cleanup warnings for binance exchange - this is a known library limitation and does not impact functionality.*
- **Memory Management**: All exchange connections properly closed with await exchange.close() in finally blocks. Migrated from synchronous to async ccxt for proper async/await handling across all services.

### UI/UX Decisions
- Interactive Telegram dashboard with inline buttons for navigation and controls.
- Clear HTML formatting for signal broadcasts and dashboard views.
- Built-in help center for user guidance.

## External Dependencies
- **Telegram Bot API**: `aiogram` library for bot interaction.
- **Cryptocurrency Exchanges**: `CCXT` library for fetching market data and executing trades. Supports KuCoin Futures (primary), OKX, and MEXC for auto-trading.
- **Database**: PostgreSQL.
- **Payment Gateways (Optional)**: Whop, Solana Pay (via Helius integration).
- **Sentiment Analysis**: OpenAI API.
- **News Aggregation**: CryptoPanic.