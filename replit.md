# Crypto Perps Signals Telegram Bot

## Overview
This project is a Python-based Telegram bot for crypto perpetual trading with automated execution on the Bitunix exchange. It features three trading modes with priority-based signal generation: AI-POWERED LONGS (5-50% @ 20x - Priority #1, best performer), PARABOLIC (50%+ exhausted dumps @ 20x - Priority #2), and LOSER RELIEF shorts. Target: 2-4 high-quality trades per day. The bot utilizes a "Top Gainers" scanning system with intelligent signal prioritization. Core strategies involve momentum-based entries with customizable leverage, dual/triple take-profit targets, and breakeven stop-loss management. 

**Subscription Tiers (Jan 2026):**
- AI Assistant ($65/mo): AI chat, market scanner, risk assessment (no auto-trading)
- Auto-Trading ($130/mo): Full automation + all AI features

The business vision is to provide high-quality, automated crypto trading signals and execution, offering a valuable tool for traders seeking to capitalize on market movements and a revenue stream through subscriptions and copy trading.

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
- **Priority-Based Signal Generation**: Scans in order - (1) AI-POWERED LONGS 5-50% with GPT validation (best performer!), (2) PARABOLIC 50%+ exhausted dumps, (3) LOSER RELIEF shorts. AI scanner is Priority #1 due to excellent performance.
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
- **AI Chat Assistant Enhancements (Jan 2026)**: Extended coin detection (100+ coins including memecoins, AI coins, L2s, DeFi, gaming). Risk assessment feature analyzes RSI, momentum, volume, trend alignment, and BTC correlation to provide risk scores (1-10) with actionable recommendations.
- **Advanced Market Analysis**: Integrates Bitunix funding rate analysis and order book depth analysis for improved signal quality and confidence scoring.
- **Cooldown Systems**: Prevents re-shorting symbols immediately after a stop-loss (can be bypassed for parabolic shorts). Standard cooldown is 2 hours between signals on the same symbol.
- **Parallel Trade Execution**: Utilizes asyncio.Semaphore for efficient, nearly simultaneous execution across multiple users.
- **Parabolic Strategy (AI-POWERED Jan 2026)**: AI-validated short entries on exhausted 50%+ pumps (Priority #2). Pre-filters candidates based on RSI ≥75, EMA overextension (>1.5%), wick rejection (≥1%), and volume surge (≥1.5x). AI validates each candidate with reversal confidence score and sets dynamic TP/SL (typically 5-8% TP, 3-5% SL at 20x leverage). Only A/A+ quality setups are approved.
- **Normal Shorts Strategy (AI-POWERED Jan 2026)**: AI-validated short entries on overbought coins (5-40% gainers, Priority #3). Pre-filters: RSI ≥60, volume ≥1.3x, price 2%+ below high, bearish signs (lower highs, red candles, EMA structure). AI validates with entry quality grades and dynamic TP/SL (typically 4-6% TP, 3-5% SL at 20x). Replaces old LOSER_RELIEF strategy.
- **TA-FIRST LONGS Strategy (Jan 2026 Refactor)**: Technical analysis pre-filters signals - requires 5/6 confirmations before AI is called: (1) Liquidity $5M+, (2) Anti-manipulation ≤25%, (3) Trend alignment EMA9>EMA21, (4) RSI 35-70, (5) Volume ≥1.2x, (6) Price <80% of range. AI validates and can reject if it doesn't like the setup, plus sets TP/SL levels. This reduces API calls from 10-20/cycle to 1-4 max.
- **TA-FIRST PARABOLIC Strategy (Jan 2026 Refactor)**: Pre-filters require RSI ≥75, EMA overextension >1.5%, wick rejection ≥1%, volume surge ≥1.5x. AI validates and can reject, plus sets TP/SL levels.
- **TA-FIRST NORMAL SHORTS Strategy (Jan 2026 Refactor)**: Pre-filters require +5-40% gainers, RSI ≥55, volume $3M+, bearish signs (EMA cross, lower highs, red candles). AI validates and can reject, plus sets TP/SL levels.
- **AI Provider (Jan 2026 Update)**: Primary: Gemini 2.5 Flash via Replit AI Integrations (no API key management, charges to Replit credits, much higher rate limits). Fallback: OpenAI gpt-4o-mini if Gemini unavailable. Global rate limiter serializes all AI calls with minimum 2-second gaps.
- **AI Rate Limit Protection (Jan 2026)**: Uses tenacity library for robust retry logic with exponential backoff (15-180s with jitter). Global OpenAI rate limiter prevents concurrent requests.
- **Risk Caps (Jan 2026)**: Max SL capped at 4% (80% loss at 20x leverage), max TP capped at 150% profit (7.5% price move at 20x).
- **Price Caching for Rate Limits**: 30-second TTL price cache prevents API rate limit bans. Exchange priority: MEXC → Bybit → Binance (Binance last due to aggressive rate limiting).

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