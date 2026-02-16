# Crypto Perps Signals Telegram Bot

## Overview
This project is a Python-based Telegram bot for automated crypto perpetual trading on the Bitunix exchange. It offers AI-POWERED LONGS, PARABOLIC shorts, and NORMAL SHORTS modes, generating 2-4 high-quality trades daily using a "Top Gainers" scanning system and intelligent signal prioritization. Key capabilities include momentum-based entries, customizable leverage, dual/triple take-profit targets, and breakeven stop-loss management. The bot's business vision is to provide automated crypto trading signals and execution, generating revenue through subscriptions and copy trading.

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
- Social Trading Mode: LunarCrush-powered signals based on Galaxy Score and social sentiment (configurable risk levels LOW/MEDIUM/HIGH). Major coins (BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, DOT, LINK) get priority with relaxed galaxy score requirements and tighter TP/SL (~1.2% TP, ~0.8% SL for longs; ~1.0% TP, ~0.7% SL for shorts).

## System Architecture

### Core Components
- **Telegram Bot**: Manages user interaction, commands, signal broadcasting, and an interactive dashboard.
- **FastAPI Server**: Provides health checks and webhook endpoints.
- **Bitunix Auto-Trading System**: Handles automated live trade execution on Bitunix Futures with configurable leverage and risk management.
- **Master Trader Copy Trading**: Executes all signals on the owner's Bitunix Copy Trading account.
- **Priority-Based Signal Generation**: Scans for PURE TA LONGS, VWAP BOUNCE SCALPS, PARABOLIC exhausted dumps, and NORMAL SHORTS.
- **Social Signals Mode**: LunarCrush TradeHub integration for trading based on Galaxy Score and social sentiment.
- **AI Chat Assistant Enhancements**: Extended coin detection, risk assessment, and actionable recommendations.
- **Admin Control System**: Provides user management, analytics, and system health monitoring.
- **Referral Reward System**: Manages cash referral tracking and payouts.

### Technical Implementations
- **Dual Data Source Architecture**: Uses Binance Futures and MEXC Futures for accurate 24h price change data, prioritized and filtered for Bitunix-tradeable symbols.
- **Order Execution Strategy**: Uses MARKET orders for immediate execution with 8 decimal precision.
- **Database**: PostgreSQL with SQLAlchemy ORM.
- **Configuration**: `pydantic-settings` for environment variables.
- **Security**: Fernet encryption, HMAC-SHA256, bearer token verification, and PostgreSQL advisory locks.
- **Risk Management**: Percentage-based SL/TP with price-level validation, risk-based position sizing, daily loss limits, max drawdown protection, auto breakeven stop loss, liquidity checks, anti-manipulation filters, new coin protection, proportional scaling, and minimum position size checks.
- **Analytics**: Tracks signal outcomes, win/loss ratios, and asset performance.
- **Health Monitoring**: Automatic checks, process restarts, error handling, and debug logging.
- **Price Caching**: Thread-safe global price cache with 30-second TTL.
- **Multi-Analysis Confirmation**: Validates signals against higher timeframes (1H) for trend alignment.
- **Subscription System**: Integrated with Coinbase Commerce for crypto payments, webhook-based auto-activation, and access control.
- **AI-Powered Strategies**:
    - **AI Market Regime Detector**: Analyzes BTC price/RSI/volatility, derivatives data, market breadth, Fear & Greed index, and BTC dominance.
    - **AI News Impact Scanner**: Fetches and analyzes crypto news using Gemini to identify trading signals.
    - **AI Whale & Smart Money Tracker**: Tracks institutional activity, high-volume coins, funding rates, order book depth, and long/short ratios.
    - **Binance Leaderboard Tracker**: Analyzes top Binance Futures traders for trade ideas.
    - **AI Chart Pattern Detector**: Detects classic chart patterns across multiple timeframes.
    - **AI Liquidation Zone Predictor**: Predicts liquidation cascade zones using Binance Futures open interest and funding rates.
- **Advanced Market Analysis**: Integrates Bitunix funding rate and order book depth.
- **Cooldown Systems**: Implements cooldowns for re-shorting and window-based cooldowns for LONGS.
- **Parallel Trade Execution**: Utilizes `asyncio.Semaphore`.
- **Automatic Market Regime Detector**: Analyzes BTC to determine BULLISH/BEARISH/NEUTRAL regime.
- **CoinGlass Derivatives Integration**: Fetches open interest, funding rates, long/short ratios, and liquidation data for AI and signal messages, dynamically adjusting TP/SL.
- **Liquidation Cascade + Social Panic Alerts**: Combines CoinGlass liquidation/OI data with LunarCrush social buzz.
- **Signal Strength Score (1-10)**: Composite score based on Technical Analysis, Social Intelligence, Influencer Consensus, Derivatives, and AI Confidence.
- **AI Provider**: Hybrid approach with Gemini 2.5 Flash for scanning and Claude Sonnet 4.5 for final approval.
- **AI Rate Limit Protection**: Uses `tenacity` for retry logic and a global OpenAI rate limiter.
- **Momentum Runner Scanner**: Scans Binance Futures 24h tickers for momentum-driven coins with strict anti-top filters.
- **Signal Frequency Limits**: Implements global and per-symbol daily caps, cooldowns, and AI confidence thresholds.
- **Twitter Multi-Personality System**: 12 distinct writing personalities (chill_trader, dry_wit, chart_nerd, old_head, night_owl, minimalist, storyteller, pragmatist, confessional, hype_contrarian, stream_of_consciousness, zen_trader). 80% AI-generated tweets with personality selection, 20% template fallback. `_get_hashtag_style()` returns empty 60% of time. All posting functions use varied casual templates instead of emoji-heavy structured lists.

### UI/UX Decisions
- Interactive Telegram dashboard with inline buttons.
- Clear HTML formatting for messages and a built-in help center.
- Auto-generation of shareable trade screenshots.
- Simplified navigation with core buttons and consolidated menus.

### Trade Tracker Web Dashboard
- **URL**: `/tracker` for a web-based spreadsheet showing all trades with P&L and ROI.
- **Standalone server**: `tracker_server.py` runs independently.
- **Features**: Stats dashboard, filterable, sortable columns, pagination, TP target hit indicators, breakdown by signal type and direction.
- **API Endpoints**: `/api/trades` (paginated list), `/api/trades/stats` (aggregated statistics).

## External Dependencies
- **Telegram Bot API**: `aiogram`.
- **Cryptocurrency Exchanges**: `CCXT` for Bitunix and Binance Futures.
- **Database**: PostgreSQL.
- **Payment Gateway**: Coinbase Commerce API.
- **AI Analysis**: Gemini 2.5 Flash and Claude Sonnet 4.5.
- **News Aggregation**: CryptoNews API.
- **Derivatives Data**: CoinGlass API V4.
- **Metals News Sentiment**: MarketAux API.