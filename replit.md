# Crypto Perps Signals Telegram Bot

## Overview
This project is a Python-based Telegram bot for crypto perpetual trading with automated execution on the Bitunix exchange. It features three trading modes with priority-based signal generation: SHORTS (35%+ mean reversion @ 10x - Priority #1, most profitable), PARABOLIC (50%+ exhausted dumps @ 20x - Priority #2), and LONGS (8-120% early momentum @ 5x - Priority #3). Target: 2-4 high-quality trades per day. The bot utilizes a "Top Gainers" scanning system with intelligent signal prioritization. Core strategies involve momentum-based entries with customizable leverage, dual/triple take-profit targets, and breakeven stop-loss management. The project includes a 2-tier subscription model ($150/month) and a cash referral system. The business vision is to provide high-quality, automated crypto trading signals and execution, offering a valuable tool for traders seeking to capitalize on market movements and a revenue stream through subscriptions and copy trading.

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
- **Priority-Based Signal Generation**: Scans in order - (1) SHORTS 35%+ mean reversion (most profitable), (2) PARABOLIC 50%+ exhausted dumps, (3) LONGS 8-120% momentum. Ensures highest-profit signals are broadcast first.
- **Volume Surge Detector**: Real-time detection of volume spikes for early entry opportunities.
- **New Coin Alerts**: Automated detection of newly listed, high-volume coins on Bitunix.
- **Admin Control System**: Provides user management, analytics, and system health monitoring, including `/broadcast` and `/grant_sub` commands.
- **Referral Reward System**: Manages cash referral tracking and payouts, with interactive wallet setup UX.

### Technical Implementations
- **Dual Data Source Architecture**: Uses Binance Futures + MEXC Futures for accurate 24h price change data (Bitunix API unreliable). Binance is primary source, MEXC adds coins not on Binance. Data is merged (Binance priority) then filtered to Bitunix-tradeable symbols only. For candle data: Binance primary with automatic Bitunix fallback for exclusive coins. (Nov 26, 2025: Fixed critical data accuracy bug - Bitunix tickers API returns garbage 24h data, switched to Binance+MEXC for accurate change percentages.)
- **Order Execution Strategy**: ALL trades use MARKET orders for immediate execution and reliable fills. All order parameters (quantity, price, TP, SL) formatted to 8 decimal precision to prevent Bitunix "Parameter error" rejections.
- **Database**: PostgreSQL with SQLAlchemy ORM.
- **Configuration**: `pydantic-settings` for environment variables.
- **Security**: Fernet encryption for API credentials, HMAC-SHA256, and bearer token verification. PostgreSQL advisory locks for duplicate trade prevention.
- **Risk Management**: Percentage-based SL/TP with price-level validation, risk-based position sizing, daily loss limits, max drawdown protection, auto breakeven stop loss, liquidity checks, anti-manipulation filters, and new coin protection. Incorporates proportional scaling to cap max profit/loss at 80% regardless of leverage. Includes minimum position size checks ($10 USDT for Bitunix).
- **Analytics**: Tracks signal outcomes, win/loss ratios, and asset performance.
- **Health Monitoring**: Automatic checks, process restarts, and error handling. Comprehensive debug logging tracks signal rejection reasons (liquidity, candle count, manipulation, RSI, volume, etc.) for rapid troubleshooting.
- **Price Caching**: Thread-safe global price cache with 30-second TTL.
- **Multi-Analysis Confirmation**: Validates signals against higher timeframes (1H) for trend alignment. PARABOLIC_REVERSAL signals bypass this validation since they trade reversals at the top (1H is still bullish when entering SHORT).
- **Subscription System**: Integrated with Coinbase Commerce for crypto payment subscriptions with webhook-based auto-activation and access control. Includes fee warnings and manual activation.
- **Referral Tracking**: Database tracks unique referral codes, referred users, and referral credits. Referral rewards are $30 USD per Auto-Trading referral.
- **Advanced Market Analysis**: Integrates Bitunix funding rate analysis and order book depth analysis for improved signal quality and confidence scoring.
- **Cooldown Systems**: Prevents re-shorting symbols immediately after a stop-loss (can be bypassed for parabolic shorts). Standard cooldown is 2 hours between signals on the same symbol.
- **Parallel Trade Execution**: Utilizes asyncio.Semaphore for efficient, nearly simultaneous execution across multiple users.
- **Parabolic Strategy**: Aggressive 200% TP @ 20x leverage for exhausted 50%+ pumps, with a 2:1 R:R. Uses a hybrid parabolic detection logic with strict confirmation and 3/3 exhaustion signs or extreme RSI.
- **LONGS Strategy**: Stricter entry conditions, requiring price at or below EMA9, increased volume (1.5x), tighter RSI (45-70), and reduced EMA distance (max 2% above EMA9). Now requires retracement before entry and expands pump range (8-120%). Includes dual strategy (Aggressive and Safe).

### UI/UX Decisions
- Interactive Telegram dashboard with inline buttons.
- Clear HTML formatting for messages.
- Built-in help center.
- Auto-generation of professional, shareable trade screenshots.
- Simplified navigation with core buttons and consolidated menus.
- Dashboard cleanup: "Scan Coins" button for quick analysis; referral system accessible to all users.

## External Dependencies
- **Telegram Bot API**: `aiogram` library.
- **Cryptocurrency Exchanges**: `CCXT` library for Bitunix and Binance Futures API for data.
- **Database**: PostgreSQL.
- **Payment Gateway**: Coinbase Commerce API.
- **Sentiment Analysis**: OpenAI API.
- **News Aggregation**: CryptoNews API.