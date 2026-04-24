# TradeHub Markets — Strategy Builder Platform

## Overview
Web platform at `tradehubmarkets.com` where users build, test, and automate crypto trading strategies — no code required. Users create strategies via a 7-step wizard or AI chat builder, paper test them free, then go live or sell in the marketplace. Revenue via Free vs Pro ($50/month) subscriptions through OxaPay. The Telegram bot is now a minimal companion app: it delivers strategy trade alerts and provides users with their login UID for the website.

## Deployment Architecture
Everything runs on Replit (no Railway). Three workflows:
- **Strategy Portal** — `python3 strategy_portal_server.py` on port 5000 — Main product. Web portal at `tradehubmarkets.com`, strategy executor, marketplace, wizard builder, backtester, AI chat builder.
- **Telegram Bot** — `python -m uvicorn main:app --host 0.0.0.0 --port 8080` — Minimal companion: Telegram command handling (`/start` shows UID + strategy summary), OxaPay payment poller, strategy trade notifications, and the on-demand `/walls` liquidity scanner. **All background scanning (social, top gainers, fartcoin, BTC ORB, sweep watcher, funding rates, Twitter) is DISABLED.**
- **Liquidity Wall Scanner (`/walls`)**: On-demand Telegram command that pulls live SPOT order books from Bybit + OKX + MEXC + Kraken in parallel, buckets nearby orders into walls, scores cross-exchange confidence, computes a distance-weighted bull/bear pressure score, and returns a Claude-generated trader-style summary. Engine: `app/services/liquidity_walls.py`. Usage: `/walls BTC` or `/walls SOL 1 50000` (symbol, band%, min USD). Bybit returns 403 from Replit IPs — gracefully skipped. WebSocket live books and spoof-detection-with-persistence are intentionally deferred (require background workers).
- **Trade Tracker** — `python3 tracker_server.py` on port 8000 — Trade performance dashboard.

Database: Neon PostgreSQL (`NEON_DATABASE_URL`) shared by all workflows.
Strategy executor runs ONLY in the Strategy Portal workflow to avoid duplicate trade execution.
Binance is geoblocked (451) on Replit — all market data uses MEXC only.

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
- **Day Trading Page (`/trade`, `/trade/btc`)**: Public, no-auth BTC chart with live order block overlays. Uses TradingView Lightweight Charts v4.1.3 + 1m/5m/15m/1h timeframe switcher. Side panel lists buy walls (support) + sell walls (resistance) with tier (huge/big/med/small), distance %, and FRESH/GROWING/SHRINKING tags from the wall_intel snapshot history. Walls poll every 30s, candles every 10s. Collapsible "AI trade read" panel parses the structured Claude schema (BIAS/TRADE/STOP/TP/R:R/LEVERAGE/INVALIDATION/ODDS/NOTE). Endpoints: `GET /api/trade/walls/{symbol}` (25s TTL cache, reuses `scan_walls()`) and `GET /api/trade/candles/{symbol}?tf=...&limit=...` (5s TTL cache, MEXC klines). BTC-only at launch via `TRADE_SYMBOL_WHITELIST`; design parameterizes `{symbol}` so adding ETH/SOL is one whitelist line. Linked from portal nav and landing-page hero.
  - **Indicator engine** (toolbar "+ Indicator" → modal with Preset / Formula / Pine tabs): Pure-JS TA library (SMA/EMA/RSI/MACD/BB/VWAP/ATR/StochRSI/SuperTrend) with chip strip, LocalStorage persistence, dynamic priceScaleId pane allocator (max 3 oscillators), `window.__indRecompute` hook into candle load/tick. The Pine tab uses an inline expression engine (`PineEval` IIFE in `app/templates/trade.html`) — Pratt parser produces an AST, evaluator handles series-broadcasting binops/unops/ternary/lookback `[n]`, builtins (close/open/high/low/hl2/hlc3/ohlc4/volume), `ta.ema/sma/rma/wma/rsi/atr/vwap/highest/lowest/crossover/crossunder/cross/change/tr`, `nz`, `na`, `math.abs/sign/round/floor/ceil/sqrt/max/min/pow`. Translator joins multi-line paren continuations, classifies inputs/assignments/plot/plotshape/plotarrow, bakes resolved input scalars into stored ASTs, and surfaces friendly per-line errors. `plotshape`/`plotarrow` translate to a hidden `marker` indicator type that drives the chart's multi-source marker manager (`_markerSources` Map → `applyAllMarkers`) shared with the live tape.
  - **Chart alerts** (toolbar "🔔 Alerts" + per-chip mini-bell): Server-evaluated alerts that DM the user on Telegram. Alert kinds: `price`, `rsi`, `ema_cross`, `macd_cross_zero`, `supertrend_flip`. Evaluated every 30s by `app/services/alerts_engine.py` running inside the same worker that holds the executor PG advisory lock (so multi-worker gunicorn never double-fires). Cross detection uses stored `last_value` + first-eval baseline so "already true" conditions don't spam-fire on creation. Endpoints: `GET /api/trade/alerts/me` (login state), `GET /api/trade/alerts/list`, `POST /api/trade/alerts/create`, `POST /api/trade/alerts/{id}/cancel`. Logged-out users see "Log in to create alerts"; web-only users (no telegram link) see a warning to link via @TradeHubMarketsBot. Quota: 25 active alerts/user. Model: `IndicatorAlert` in `app/models.py`.
- **Built-in Backtester**: Integrated into wizard Step 7. Users click "Quick Test (30d)" or "Deep Test (90d)" to replay their strategy against real 1h OHLCV data. Data sources: Gate.io Futures API (primary — covers ALL USDT perp pairs including low caps like PIPPIN, FARTCOIN, WIF, BONK) → Kraken fallback (major coins). Results show win rate, total P&L, profit factor, max drawdown, avg hold time, equity curve (inline SVG), and a full trade log. Pro subscribers only. Endpoint: `POST /api/backtest/run`. Engine: `app/services/backtest_engine.py`.
- **Portal Subscription Tiers**: Free (wizard builder, paper trading only, leaderboard/marketplace view) vs Pro ($50/month, AI Chat Builder, backtesting 30d/90d, unlimited AI chat, AI Strategy Advisor, live automation, marketplace copy). Managed via `PortalSubscription` table (`portal_subscriptions`). Admin can grant Pro via `POST /api/portal/upgrade`. Tier badge shown in topbar.
- **Public Marketing Website**: Full landing page at `/` with hero section, features, marketplace preview, leaderboard, and creator earnings showcase. Login at `/login` with multi-method auth: Google OAuth, email/password registration, or legacy TH- access codes. HMAC-signed session cookies. Authenticated app at `/app`. Legacy `/strategies?uid=` links preserved.
- **Web Auth System**: Users can create accounts directly via Google OAuth (`GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` required) or email+password. Web-created accounts use `WEB-{hex}` as a telegram_id placeholder and get a `TH-` UID assigned. Auth provider tracked in `auth_provider` column ('telegram'|'google'|'email'). Passwords hashed with PBKDF2-SHA256 (200k iterations, no external deps). Google OAuth flow: `GET /auth/google` → Google → `GET /auth/google/callback`.
- **Trade Tracker Web Dashboard**: Provides a web-based interface to view and analyze trade performance.

### Technical Implementations
- **Dual Data Source Architecture**: Employs Binance Futures and MEXC Futures for price data, prioritized for Bitunix-tradeable symbols.
- **Order Execution**: Uses MARKET orders with high precision.
- **Database**: PostgreSQL with SQLAlchemy ORM. Both dev and production use the shared Neon PostgreSQL database (`NEON_DATABASE_URL`), configured via `app/config.py` which checks `NEON_DATABASE_URL` first. This ensures dev and production always share identical data.
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