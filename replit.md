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
- Social Trading Mode: LunarCrush-powered signals based on Galaxy Score and social sentiment (configurable risk levels LOW/MEDIUM/HIGH)

## System Architecture

### Core Components
- **Telegram Bot**: Manages user interaction, commands, signal broadcasting, and an interactive dashboard.
- **FastAPI Server**: Provides health checks and webhook endpoints.
- **Bitunix Auto-Trading System**: Handles automated live trade execution on Bitunix Futures with configurable leverage and risk management.
- **Master Trader Copy Trading**: Executes all signals on the owner's Bitunix Copy Trading account.
- **Priority-Based Signal Generation**: Scans for PURE TA LONGS, VWAP BOUNCE SCALPS, PARABOLIC exhausted dumps, and NORMAL SHORTS.
- **Social Signals Mode**: LunarCrush TradeHub integration for trading based on Galaxy Score and social sentiment.
- **LunarCrush Deep Intelligence**: Influencer consensus tracking (top creators, follower reach, sentiment breakdown), social buzz momentum (rising/falling/stable trends), social time series analysis, coin news aggregation, and viral post tracking. Cross-references LunarCrush news with CryptoNews for dual-source confirmation on news signals.
- **`/buzz` Command**: Shows comprehensive social intelligence for any coin - social strength, buzz momentum, influencer consensus, top creators, viral posts, and latest news.
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
- **CoinGlass Derivatives Integration**: Fetches open interest, funding rates, long/short ratios, and liquidation data before each signal. Data is passed to AI (Gemini + Claude) for smarter trade decisions and displayed in signal messages. Derivatives data also dynamically adjusts TP/SL levels: funding rate extremes widen/narrow targets, OI trends confirm momentum, L/S ratio crowding tightens stops, and liquidation cascades adjust target reach.
- **Liquidation Cascade + Social Panic Alerts**: Combines CoinGlass liquidation/OI data with LunarCrush social buzz to detect dangerous market conditions. Alerts fire when heavy one-sided liquidations (>$500K), OI collapse, extreme funding, and falling social sentiment converge. Severity levels: MODERATE/HIGH/EXTREME. Detects LONG_CASCADE (longs getting rekt) and SHORT_SQUEEZE patterns. 6-hour cooldown per symbol.
- **Signal Strength Score (1-10)**: Composite score rating each signal by counting confirmations across 5 data sources: Technical Analysis (RSI, volume, trend), Social Intelligence (Galaxy, sentiment, strength), Influencer Consensus (alignment, creator count, whale accounts), Derivatives (funding, OI, L/S ratio), and AI Confidence. Displayed prominently on every signal with visual bar, tier label (ELITE/STRONG/MODERATE/WEAK/LOW), and source breakdown.
- **AI Provider**: Hybrid approach - Gemini 2.5 Flash for initial signal scanning (cost-effective), Claude Sonnet 4.5 for final signal approval/filtering (quality gate).
- **AI Rate Limit Protection**: Uses `tenacity` for retry logic and a global OpenAI rate limiter.
- **Momentum Runner Scanner**: Scans Binance Futures 24h tickers for coins moving ±5% or more with $500K+ volume. Supports both LONG (upside runners) and SHORT (downside crashers). Tiered TP/SL by move magnitude (5%/10%/15%+). AI-gated with derivatives integration.
- **Signal Frequency Limits**: 8 unified global daily cap (shared across ALL signal modes: top gainers, social, news), 4 daily short cap, 30min symbol cooldown, 6h same-coin signal cooldown, 30min AI rejection cooldown, 2h global LONG cooldown, 1h news cooldown, minimum AI confidence 4/10.
- **Risk Caps**: Maximum SL capped at 4% and maximum TP capped at 150%.
- **$TICKER Format**: All coin mentions in signal messages use mandatory $TICKER format (e.g., $BTC, $ETH).
- **Scan Priority**: News → Momentum Runners → Social Longs → Social Shorts.

### UI/UX Decisions
- Interactive Telegram dashboard with inline buttons.
- Clear HTML formatting for messages and a built-in help center.
- Auto-generation of shareable trade screenshots.
- Simplified navigation with core buttons and consolidated menus.

## Trade Tracker Web Dashboard
- **URL**: `/tracker` - Web-based spreadsheet showing all trades with P&L and ROI
- **Standalone server**: `tracker_server.py` runs independently without starting the Telegram bot
- **Features**: Stats dashboard (win rate, total P&L, avg ROI), filterable by status/direction/type/period/symbol, sortable columns, pagination, TP target hit indicators, breakdown by signal type and direction
- **API Endpoints**: `/api/trades` (paginated trade list), `/api/trades/stats` (aggregated statistics)
- **Also integrated**: Into main bot's FastAPI app via `subscriptions.py` router include

## External Dependencies
- **Telegram Bot API**: `aiogram` library.
- **Cryptocurrency Exchanges**: `CCXT` library for Bitunix and Binance Futures API.
- **Database**: PostgreSQL.
- **Payment Gateway**: Coinbase Commerce API.
- **AI Analysis**: Hybrid - Gemini 2.5 Flash (scanning) + Claude Sonnet 4.5 (final approval).
- **News Aggregation**: CryptoNews API.
- **Derivatives Data**: CoinGlass API V4 for open interest, funding rates, long/short ratios, liquidations.
- **Metals News Sentiment**: MarketAux API for gold/silver news analysis.

## Metals Trading Feature (Admin-Only)
- **Gold (XAU) and Silver (XAG) trading** on Bitunix based on news sentiment
- **MarketAux API** integration for real-time metals news with sentiment scoring
- **Technical analysis** (RSI, EMA) combined with news sentiment for signal generation
- **Admin command**: `/metals` with subcommands (on, off, scan, news)
- Uses Binance Futures for price data (XAUUSDT, XAGUSDT)
- Signals sent only to admins during testing phase