# Crypto Perps Signals Telegram Bot

## Overview
A Python-based Telegram bot designed to generate and broadcast cryptocurrency perpetual futures day trading signals with **1:1 risk-reward ratio**. It employs a strict **6-point confirmation system** requiring trend alignment, spot buying/selling pressure from Binance + 3 exchanges, volume spikes, momentum confirmation, clean candle patterns, and high liquidity session validation. Signals feature **20% TP / 20% SL** (2% price move with 10x leverage) for consistent, high-probability entries with automated execution on **Bitunix exchange**. The bot offers free signals, PnL tracking, paper trading, and comprehensive risk management for day trades only. The project aims to provide pinpoint entries with high success rates.

## Recent Changes (Oct 26, 2025)
- **💎 NOWPAYMENTS SUBSCRIPTION SYSTEM**: Implemented crypto payment subscriptions with grandfathering:
  - **Grandfathered Users**: All 5 existing users marked with lifetime free access (reward for early support)
  - **New Users**: Require $29.99/month subscription via NOWPayments (BTC, ETH, USDT, 200+ cryptos)
  - **Payment Integration**: Full NOWPayments API integration with invoice generation and webhook callbacks
  - **Subscription Checks**: Access control prevents non-subscribed users from receiving signals
  - **/subscribe Command**: Shows status for grandfathered/active/expired, generates payment links for new users
  - **Dashboard Display**: Subscription status visible on main dashboard (Lifetime/Premium/Free Trial)
  - **Auto-Activation**: Webhook handler activates subscription instantly on payment confirmation
  - **User Notifications**: Telegram notifications sent when subscription is activated
  - **Database Schema**: Added `grandfathered`, `nowpayments_subscription_id` columns to User model
  - **Security**: HMAC-SHA512 signature verification on webhook callbacks
- **✅ 48H WATCHLIST BUG FIX**: Fixed critical AttributeError in broadcast function:
  - **Bug**: Code called non-existent `fetch_top_gainers()` method (should be `get_top_gainers()`)
  - **Fix**: Updated both line 752 and 810 to use correct method name with proper parameters
  - **Impact**: Watchlist system now fully operational - monitors current + yesterday's pumps for delayed reversals
  - **Status**: Confirmed working on Railway with AIXBT, MAVIA, BLUAI technical analysis via Binance
- **🔧 HYBRID DATA SOURCE ARCHITECTURE**: Implemented multi-exchange data strategy for maximum reliability:
  - **Critical Discovery**: Bitunix klines API completely broken (returns "System error" for ALL symbols including BTC/ETH)
  - **Solution**: Use **Binance Futures public API** for candle data analysis (no auth needed)
  - **Hybrid Approach**: Bitunix for tickers (finding pumps) + trade execution | Binance for technical analysis (EMA/MACD/RSI)
  - **Symbol Conversion**: Auto-converts BTC/USDT → BTCUSDT format for Binance perpetual futures
  - **Wide Coverage**: Binance has more altcoins than OKX (MAVIA, AIXBT, etc. all available)
  - **Railway EU Hosting**: No geo-restrictions on Railway servers (Binance accessible from EU)
  - **Impact**: Top Gainers scanner now has FULL technical analysis capabilities for parabolic reversal detection
- **🐛 CRITICAL FIX - Top Gainers Scanner API Bugs Fixed**:
  - **Bug #1 (Tickers)**: Scanner was looking for 'rose' and 'vol' fields that don't exist in Bitunix ticker API
  - **Fix #1**: Now calculates % change from 'open' and 'lastPrice' fields, uses 'quoteVol' for volume
  - **Bug #2 (Candles)**: Bitunix klines API returns "System error" for all symbols
  - **Fix #2**: Switched to OKX public API for candle data with proper array format handling
  - **Impact**: Scanner can now detect AND analyze all pumps (AIXBT +55%, MAVIA +43%, BLUAI +32%)
  - **Status**: 18 coins currently meet criteria (10%+ with $1M+ volume), parabolic detection operational
- **📸 TRADE SCREENSHOT SHARING**: Auto-generate beautiful shareable trade cards for marketing:
  - **Auto-Screenshot on Close**: Every closed trade (TP/SL/Smart Exit) automatically generates a branded image
  - **Manual Sharing**: `/share_trade [trade_id]` command lets users share any past trade
  - **Inline "Share This Win" Buttons**: Winning trades get a one-click "📸 Share This Win" button in notifications
  - **Professional Design**: Custom TradehHub AI robot background, cyan accents, win/loss colors, large PnL display
  - **Marketing Data**: Shows symbol, direction, entry/exit prices, PnL %, duration, win streak
  - **Social Ready**: Perfect 1024x768 images for Twitter, Telegram sharing to showcase trading success
- **🔥 TOP GAINERS SCANNER WITH 48H WATCHLIST**: Monitors parabolic pumps for delayed reversals:
  - **48-Hour Watchlist**: Automatically tracks yesterday's pumps for 24-48h (many reversals are delayed)
  - **Smart Monitoring**: Every scan checks BOTH current top gainers AND yesterday's pumps for reversal signals
  - **Auto-Cleanup**: Removes symbols older than 48 hours automatically
  - **No Duplicates**: Marks watchlist entries as "signal sent" to prevent repeated alerts
  - **Extended Coverage**: Catches reversals that happen 1-2 days AFTER initial pump (common in crypto)
  - **15-Minute Scanning**: Scans every 15 min - catches reversals before they crash
  - **Lower Parabolic Thresholds**: 3% price extension, RSI 60 for EARLIER entries
  - **24/7 Operation**: No time restrictions (unlike standard signals) - pumps can happen anytime!
  - **Parabolic Reversal Priority**: Detects 50%+ pumps rolling over with 15m/5m EMA divergence for high-confidence shorts
  - **Fixed 5x Leverage**: Safer leverage (vs 10x standard) for volatile reversals
  - **Dual TPs for Parabolic Shorts**: 20% + 35% (captures full crash) vs standard 20% single TP
  - **Position Limits**: Max 3 top gainer positions simultaneously (configurable per user)
  - **Trade Type Tagging**: All top gainer trades tagged as `trade_type='TOP_GAINER'` for analytics segregation
  - **UI Toggle**: Enable/disable in /settings with detailed risk warnings and mode explanation
  - **Database Schema**: Added `top_gainer_watchlist` table; `top_gainers_mode_enabled`, `top_gainers_max_symbols`, `top_gainers_min_change` to UserPreference; `trade_type` column to Trade model
- **🔍 ON-DEMAND COIN SCANNER**: Added `/scan` command for instant market analysis without generating signals:
  - **Real-Time Analysis**: Analyzes trend (5m+15m), volume, momentum (MACD/RSI), institutional spot flow, and session quality
  - **Overall Bias Scoring**: Calculates bullish/bearish/neutral bias with percentage strength based on weighted factors (spot flow = highest weight)
  - **Visual Flow Indicator**: Shows buy/sell pressure with green/red bar visualization
  - **Educational Tool**: Helps users understand market conditions without trading pressure
  - **Usage**: `/scan BTC`, `/scan SOL`, `/scan ETH` - supports any symbol
- **🎯 1:1 DAY TRADING STRATEGY OVERHAUL**: Completely redesigned signal system for EARLY entries (not late confirmations). NEW FEATURES:
  - **FAST 5m+15m Timeframes**: Switched from 15m+1H to 5m+15m for entries BEFORE moves happen (not after 2-3% already moved)
  - **6-Point EARLY Confirmation System**: (1) EMA 9>21 crossover on 5m+15m (looser), (2) **SPOT FLOW PRIORITY >75%** - Institutional buying/selling from Binance+3 exchanges is HIGHEST PRIORITY, (3) Volume BUILDING >1.3x (not waiting for 2x spike), (4) MACD just starting to turn (not fully crossed), (5) Candle body forming (not full pattern), (6) High liquidity session only (8am-11pm UTC)
  - **Smart Money Priority**: Spot flow increased from 60% to 75% threshold - only trades when institutions are heavily buying/selling (not retail noise)
  - **1:1 Risk-Reward**: Single 20% TP / 20% SL (2% price move @ 10x leverage) - no more multi-TP complexity
  - **Smart Exit System**: Active monitoring with 6 reversal detectors (EMA crossover, candle patterns, RSI divergence, volume spikes, profit protection) closes positions early when market reverses
  - **PREDICTIVE Not Confirmatory**: Enters at FIRST signs of reversal, not after trend is established
  - **Removed Underperforming Patterns**: Deleted DOUBLE_TOP (0% win rate), funding extremes, weak divergence, and all reversal patterns
- **Pattern Performance Analytics**: Added `/pattern_performance` admin command to track win rate, avg PnL, and trade count per pattern. Database analysis revealed SHORTs only profited from manual early exits, LONGs had 0% win rate - confirming need for complete strategy rebuild.

## User Preferences
- **Muted Symbols**: Disable signals for specific pairs
- **Default PnL Period**: Choose default view (today/week/month)
- **DM Alerts**: Toggle private message notifications
- **Position Sizing**: Users can set position size as percentage of account balance
- **Max Positions**: Limit simultaneous open positions
- **Risk Filtering**: Customizable accepted risk levels for signals
- **Paper Trading Mode**: Toggle between paper/live modes with `/toggle_paper_mode`
- **Correlation Filter**: Prevent opening correlated positions (e.g., BTC + ETH simultaneously)
- **Funding Rate Alerts**: Get notified of extreme funding rates for arbitrage opportunities
- **Top Gainers Mode**: Enable/disable automated trading of high-momentum coins (5x leverage, 20% TP/SL, max 3 positions)

## System Architecture

### Core Components
- **Day Trading Signal System**: 1:1 risk-reward generator with EARLY entry signals:
  - **Single TP/SL**: 20% TP / 20% SL (2% actual price move @ 10x leverage) for clean, consistent exits
  - **6-Point EARLY Entry System**: (1) Trend - EMA 9>21 on 5m + 15m (FAST), (2) **SPOT FLOW PRIORITY** - Binance + exchanges >75% institutional-grade buying/selling pressure (HIGHEST PRIORITY), (3) Volume - >1.3x building (EARLY, not 2x spike), (4) Momentum - MACD just turning + RSI 35-65, (5) Candle - Body forming with decent size (EARLY, not full pattern), (6) Session - Only 8am-11pm UTC high liquidity hours
  - **Smart Money First**: Institutional spot flow is checked first and requires >75% confidence (not retail 60%) - if institutions aren't aligned, other checks are skipped
  - **Predictive Entry**: Catches moves BEFORE they run 2-3%, not confirming after
  - **Fast Timeframes**: 5m primary + 15m confirmation (was 15m+1H which lagged too much)
  - **Smart Exit Protection**: Automated reversal detection with 6 methods (EMA crossover, candle patterns, RSI divergence, volume spikes, profit locking) actively monitors all open positions
  - **Quality Over Quantity**: Far fewer signals but entered at optimal prices with institutional backing
- **Precision Entry System**: Advanced entry refinement using 3-pattern candle detection (Engulfing, Hammer/Shooting Star, Strong Body) and intelligent entry price optimization.
- **Reversal Bounce Catcher**: Multi-pattern scanner detecting early breakout signals: Support/Resistance Bounces, Bollinger Squeeze Breakouts, Double Bottom/Top Reversals, RSI Divergence Reversals, and Volume Spike Reversals.
- **Telegram Bot**: Handles user interaction, command processing, signal broadcasting, and an interactive dashboard with PnL calculations.
- **FastAPI Server**: Provides health checks and webhook endpoints.
- **Bitunix Auto-Trading System**: Automated execution exclusively on Bitunix Futures with configurable leverage, smart exit system, adaptive sizing, and risk management.
- **Coin Scanner Service**: On-demand market analysis tool (`/scan`) providing real-time trend, volume, momentum, and institutional flow analysis without generating trading signals. Calculates weighted bias scores for educational purposes.
- **Top Gainers Trading Mode**: Automated **SHORT-focused** mean reversion system for volatile coins. **Runs 24/7 with no time restrictions** (unlike standard signals). Scans Bitunix for 10%+ gainers, prioritizes parabolic reversals (50%+ pumps), executes with fixed 5x leverage. SHORTs use dual TPs (20%+35% for parabolic, 20% for regular). LONGs are RARE - only triggered with exceptional 3x+ volume (95% of signals are SHORTs). Toggleable in /settings and autotrading menu. Tagged as `trade_type='TOP_GAINER'` for analytics.
- **News-Based Trading Signals**: AI-powered system monitoring CryptoNews API for market-moving events and sentiment analysis.
- **Admin Control System**: Features user management, analytics (DAU/WAU/MAU, signal performance), and system health monitoring.
- **Paper Trading System**: Provides a simulated trading environment for users.
- **Backtesting System**: Admin-only feature for strategy validation.
- **Multi-Exchange Spot Market Monitor**: Monitors buying/selling pressure across major exchanges for high-conviction flow alerts.

### Technical Implementations
- **Database**: PostgreSQL with SQLAlchemy ORM.
- **Configuration**: `pydantic-settings` for environment variables.
- **Security**: Fernet encryption for API credentials, HMAC-SHA256 and bearer token verification for webhooks.
- **Risk Management**: Percentage-based SL/TP (SCALP: 12% SL, 10%/15%/20% TPs; SWING: 20% SL, 15%/30%/50% TPs), risk-based position sizing, daily loss limits, max drawdown protection, and automatic breakeven stop loss.
- **Analytics**: Tracks signal outcomes, win/loss ratios, and asset performance.
- **Correlation Filter**: Prevents opening correlated positions by grouping assets.
- **Funding Rate Monitor**: Monitors funding rates for arbitrage opportunities.
- **Memory Management**: Uses async `ccxt` with proper connection closures.
- **Health Monitoring & Auto-Recovery**: Automatic health checks, process restarts, and graceful error handling.
- **Centralized Error Handling**: Production-grade error logging with retry mechanisms.
- **Price Caching**: Thread-safe global price cache with 30-second TTL.
- **Multi-Analysis Confirmation**: Advanced validation system checks signals against higher timeframes (1H) for trend alignment, volume, momentum, and price structure.

### UI/UX Decisions
- Interactive Telegram dashboard with inline buttons.
- Clear HTML formatting for signals and dashboard messages.
- Built-in help center.

## External Dependencies
- **Telegram Bot API**: `aiogram` library.
- **Cryptocurrency Exchanges**: `CCXT` library for Bitunix.
- **Database**: PostgreSQL.
- **Sentiment Analysis**: OpenAI API.
- **News Aggregation**: CryptoNews API.