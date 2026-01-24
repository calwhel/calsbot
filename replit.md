# Crypto Perps Signals Telegram Bot

## Overview
This project is a Python-based Telegram bot designed for automated crypto perpetual trading on the Bitunix exchange. It offers three primary trading modes: AI-POWERED LONGS, PARABOLIC shorts, and NORMAL SHORTS, all prioritized for signal generation. The bot aims to deliver 2-4 high-quality trades daily by utilizing a "Top Gainers" scanning system and intelligent signal prioritization. Core strategies include momentum-based entries, customizable leverage, dual/triple take-profit targets, and breakeven stop-loss management. The business vision is to provide automated crypto trading signals and execution, generating revenue through subscriptions and copy trading.

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
- **Master Trader Copy Trading**: Executes all signals on the owner's Bitunix Copy Trading account.
- **Priority-Based Signal Generation**: Scans for PURE TA LONGS, VWAP BOUNCE SCALPS, PARABOLIC exhausted dumps, and NORMAL SHORTS.
- **Volume Surge Detector**: Real-time detection of volume spikes.
- **New Coin Alerts**: Automated detection of newly listed, high-volume coins on Bitunix.
- **Admin Control System**: Provides user management, analytics, and system health monitoring.
- **Referral Reward System**: Manages cash referral tracking and payouts.

### Technical Implementations
- **Dual Data Source Architecture**: Uses Binance Futures and MEXC Futures for accurate 24h price change data, prioritized and filtered for Bitunix-tradeable symbols.
- **Order Execution Strategy**: Uses MARKET orders for immediate execution. All order parameters are formatted to 8 decimal precision.
- **Database**: PostgreSQL with SQLAlchemy ORM.
- **Configuration**: `pydantic-settings` for environment variables.
- **Security**: Fernet encryption for API credentials, HMAC-SHA256, bearer token verification, and PostgreSQL advisory locks for duplicate trade prevention.
- **Risk Management**: Percentage-based SL/TP with price-level validation, risk-based position sizing, daily loss limits, max drawdown protection, auto breakeven stop loss, liquidity checks, anti-manipulation filters, new coin protection, and proportional scaling. Includes minimum position size checks.
- **Analytics**: Tracks signal outcomes, win/loss ratios, and asset performance.
- **Health Monitoring**: Automatic checks, process restarts, error handling, and comprehensive debug logging.
- **Price Caching**: Thread-safe global price cache with 30-second TTL.
- **Multi-Analysis Confirmation**: Validates signals against higher timeframes (1H) for trend alignment, with exceptions for PARABOLIC_REVERSAL.
- **Subscription System**: Integrated with Coinbase Commerce for crypto payments, webhook-based auto-activation, and access control.
- **Referral Tracking**: Database tracks unique referral codes, referred users, and credits.
- **AI Chat Assistant Enhancements**: Extended coin detection, risk assessment (RSI, momentum, volume, trend, BTC correlation), and actionable recommendations.
- **AI Market Regime Detector**: Analyzes BTC price/RSI/volatility, derivatives data, market breadth, Fear & Greed index, and BTC dominance to identify market regimes, risk levels, and tactical playbooks.
- **AI News Impact Scanner**: Fetches and analyzes crypto news from CryptoCompare/CryptoPanic using Gemini to identify trading signals and impact.
- **AI Whale & Smart Money Tracker**: Tracks institutional activity, high-volume coins, funding rate extremes, order book depth, and long/short ratios to identify smart money bias.
- **Binance Leaderboard Tracker**: Tracks top Binance Futures traders, analyzes open positions, identifies consensus trades, and provides AI-generated trade ideas.
- **Enhanced /scan Command**: Provides multi-timeframe confluence, risk scores, AI confidence grades, VWAP deviation bands, ATR Volatility Squeeze detection, OBV Volume Flow, and AI Trade Idea Validation.
- **AI Chart Pattern Detector**: Detects classic chart patterns (Head & Shoulders, Double Top/Bottom, Triangles, Wedges, Flags, Pennants, Cup & Handle) across multiple timeframes.
- **AI Liquidation Zone Predictor**: Predicts liquidation cascade zones by analyzing Binance Futures open interest and funding rates, identifying high-risk areas.
- **Advanced Market Analysis**: Integrates Bitunix funding rate and order book depth analysis for improved signal quality.
- **Cooldown Systems**: Implements cooldowns for re-shorting and specific window-based cooldowns for LONGS.
- **Parallel Trade Execution**: Utilizes `asyncio.Semaphore` for efficient, simultaneous execution across users.
- **Parabolic Strategy (Reversal Confirmed)**: Short entries on coins with clear reversal signs based on technical indicators and volume.
- **Normal Shorts Strategy (TREND REVERSAL)**: Detects when trend has CHANGED from bullish to bearish, then finds good entries. Looks for: EMA9 < EMA21 (bearish cross), lower highs/lows forming, rejection wicks, red candles. Entry timing: pulled back from high, not chasing above EMA. AI sets dynamic TP/SL.
- **Dump Mode (BTC ≤-2% or RSI<40)**: Automatically relaxes SHORT filters when BTC is dumping - RSI ≥50 (from 60), EMA overextension ≥1.0% (from 1.5%), only 1 bearish sign needed (from 2), 24h change range widened to 3-50%, volume threshold lowered to $2M.
- **AI-POWERED LONGS Strategy (RELAXED v5)**: Targets coins with specific confirmations for bullish trends, liquidity, volume, and RSI ranges. Includes a stricter "Overnight Mode" for low-volume hours.
- **VWAP Bounce Scalp Strategy (TIGHTENED)**: High-probability scalp trades targeting small price moves with strict criteria including 1H trend, VWAP proximity, RSI, and volume surge.
- **Automatic Market Regime Detector**: Analyzes BTC (24h change, RSI, EMA9 vs EMA21) to determine BULLISH/BEARISH/NEUTRAL regime. In BEARISH regime, shorts scan first; in BULLISH, longs scan first. Updates every 2 minutes.
- **AI Provider**: Hybrid approach - Gemini 2.5 Flash for initial signal scanning (cost-effective), Claude Sonnet 4.5 for final signal approval/filtering (quality gate).
- **AI Rate Limit Protection**: Uses `tenacity` for retry logic and a global OpenAI rate limiter.
- **Signal Frequency Limits**: Caps daily and per-window signal counts, with scalps running independently.
- **Risk Caps**: Maximum SL capped at 4% and maximum TP capped at 150%.

### UI/UX Decisions
- Interactive Telegram dashboard with inline buttons.
- Clear HTML formatting for messages and a built-in help center.
- Auto-generation of shareable trade screenshots.
- Simplified navigation with core buttons and consolidated menus.

## External Dependencies
- **Telegram Bot API**: `aiogram` library.
- **Cryptocurrency Exchanges**: `CCXT` library for Bitunix and Binance Futures API.
- **Database**: PostgreSQL.
- **Payment Gateway**: Coinbase Commerce API.
- **AI Analysis**: Hybrid - Gemini 2.5 Flash (scanning) + Claude Sonnet 4.5 (final approval).
- **News Aggregation**: CryptoNews API.