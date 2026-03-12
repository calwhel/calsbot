# Crypto Perps Signals Telegram Bot

## Overview
This project is a Python-based Telegram bot designed for automated crypto perpetual trading on the Bitunix exchange. It offers AI-powered trading signals (LONGS, PARABOLIC shorts, NORMAL SHORTS) with a focus on high-quality trades derived from a "Top Gainers" scanning system and intelligent signal prioritization. The bot incorporates momentum-based entries, customizable leverage, multiple take-profit targets, and dynamic stop-loss management. The overarching business goal is to generate revenue through subscriptions and copy trading by providing automated, high-performance crypto trading signals.

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
- Social Trading Mode: TA-scanner-powered signals (VOLUME_SCALP, SQUEEZE, SUPERTREND, MACD, RANGE_BREAKOUT, EMA_PULLBACK, HALF_BACK, OVERSOLD_REVERSAL) — all scanners run on pure price/volume data.

## System Architecture

### Core Components
- **Telegram Bot**: Handles user interaction, commands, signal broadcasting, and an interactive dashboard.
- **FastAPI Server**: Manages health checks and webhook endpoints.
- **Bitunix Auto-Trading System**: Executes live trades on Bitunix Futures with configurable risk management.
- **Master Trader Copy Trading**: Replicates all signals on the owner's Bitunix Copy Trading account.
- **Priority-Based Signal Generation**: Scans for various trade setups including PURE TA LONGS, VWAP BOUNCE SCALPS, PARABOLIC exhausted dumps, NORMAL SHORTS, and RELIEF BOUNCE longs.
- **Social Signals Mode**: Utilizes eight distinct TA-based scanning strategies.
- **AI Chat Assistant**: Provides enhanced coin detection, risk assessment, and actionable recommendations.
- **AI Trade Reviewer**: Analyzes closed trades for quality and suggests improvements, delivered to admin.
- **Admin Control System**: Facilitates user management, analytics, and system health monitoring.
- **Referral Reward System**: Manages cash referral tracking and payouts.
- **Strategy Creator Marketplace**: Allows users to publish, sell, and purchase trading strategies.
- **Build Your Own Strategy Portal**: Enables users to define custom strategies in plain English, which are compiled and executed by the bot.
- **Public Marketing Website**: Full landing page at `/` with hero section, features, marketplace preview, leaderboard, and creator earnings showcase. Login at `/login` with UID-based session cookies (HMAC-signed). Authenticated app at `/app`. Legacy `/strategies?uid=` links preserved.
- **Trade Tracker Web Dashboard**: Provides a web-based interface to view and analyze trade performance.

### Technical Implementations
- **Dual Data Source Architecture**: Employs Binance Futures and MEXC Futures for price data, prioritized for Bitunix-tradeable symbols.
- **Order Execution**: Uses MARKET orders with high precision.
- **Database**: PostgreSQL with SQLAlchemy ORM.
- **Configuration**: `pydantic-settings` for environment variables.
- **Security**: Utilizes Fernet encryption, HMAC-SHA256, bearer token verification, and PostgreSQL advisory locks.
- **Risk Management**: Comprehensive features including percentage-based SL/TP, risk-based position sizing, daily loss limits, max drawdown protection, and liquidity checks.
- **Analytics**: Tracks signal outcomes, win/loss ratios, and asset performance.
- **Health Monitoring**: Includes automatic checks, process restarts, and error handling.
- **Real-time Data**: Uses Binance WebSocket for ticker data, with fallback to REST API.
- **Multi-Analysis Confirmation**: Validates signals against higher timeframes (1H) for trend alignment.
- **VWAP Entry Filter**: Implements Volume Weighted Average Price filtering to avoid overextended or weak entries.
- **Subscription System**: Integrated with Coinbase Commerce for crypto payments and access control.
- **AI-Powered Strategies**: Incorporates AI for market regime detection, adaptive TP/SL, trade learning, signal explanation, news impact scanning, whale/smart money tracking, chart pattern detection, liquidation zone prediction, and exit optimization.
- **Advanced Market Analysis**: Integrates Bitunix funding rate and order book depth.
- **Cooldown Systems**: Manages signal frequency with global and per-symbol limits.
- **Parallel Trade Execution**: Achieved via `asyncio.Semaphore`.
- **Automatic Market Regime Detector**: Analyzes BTC to determine market conditions (BULLISH/BEARISH/NEUTRAL).
- **Signal Strength Score**: A composite score (1-10) based on multiple factors, requiring a minimum of 8/10 for broadcasting.
- **Volume Profile Analysis**: Computes HVN, LVN, POC, and Value Area for TP/SL optimization.
- **Order Flow Imbalance Detection**: Price-action-only directional flow score.
- **Chart-Based TP/SL Optimization**: Uses detected support/resistance from 5m candles.
- **AI Provider**: Hybrid approach with Gemini 2.5 Flash for scanning and Claude Sonnet 4.5 for final approval, incorporating past trade lessons.
- **AI Rate Limit Protection**: Utilizes `tenacity` for retries and a global rate limiter.
- **Momentum Runner Scanner**: Identifies momentum-driven coins.
- **Volume Scalp Scanner**: Detects sudden volume surges for quick scalping trades.
- **Squeeze Breakout Scanner**: Detects and trades squeeze releases.
- **SuperTrend Scanner**: Identifies trend flips using SuperTrend indicator.
- **MACD Momentum Scanner**: Detects MACD crossovers with EMA ribbon confirmation.
- **Relief Bounce Scanner**: Scans for contrarian LONG entries from significant price drops.
- **Signal Frequency Limits**: Enforces daily caps and cooldowns.
- **Twitter Multi-Personality System**: Generates AI-driven tweets with various personalities and randomized lengths.

### UI/UX Decisions
- **Premium Dashboard**: Features live P&L summary, win rate, and BTC price header.
- **Grouped Navigation Buttons**: Organizes features into intuitive categories like Trading, AI Tools, Trading Modes, and Account.
- **Consistent Navigation**: Seamless in-message transitions with `edit_text`.
- **Premium Signal Formatting**: Uses clear separators, tree-style TP/SL, and block-character strength bars.
- **Access Control**: Implements `check_access` for banned/unapproved users on all menu screens.
- **Interactive Telegram Dashboard**: Utilizes inline buttons for enhanced user experience.
- **HTML Formatting**: Clear HTML formatting for messages and a built-in help center.
- **Shareable Trade Screenshots**: Auto-generation of trade screenshots.
- **Brand-Neutral Messaging**: Avoids third-party API names in user-facing content.
- **User UIDs**: Unique TH-XXXXXXXX identifiers for each user, displayed on dashboards.
- **Admin Dashboard**: Interactive panel for user management, trade stats, user lookup, and quick moderation actions.

## External Dependencies
- **Telegram Bot API**: `aiogram`
- **Cryptocurrency Exchanges**: `CCXT` (Bitunix, Binance Futures)
- **Database**: PostgreSQL
- **Payment Gateway**: Coinbase Commerce API
- **AI Analysis**: Gemini 2.5 Flash, Claude Sonnet 4.5
- **News Aggregation**: CryptoNews API
- **Metals News Sentiment**: MarketAux API