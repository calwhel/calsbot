# Crypto Perps Signals Telegram Bot

## Overview
This project is a Python-based Telegram bot designed to generate and broadcast cryptocurrency perpetual futures day trading signals. It utilizes a strict 6-point confirmation system focused on early entries, employing a 1:1 risk-reward ratio (20% TP / 20% SL) for automated execution primarily on Bitunix. The bot offers free signals, PnL tracking, paper trading, comprehensive risk management, and a newly implemented crypto subscription system. The core ambition is to deliver high-probability, pinpoint entries for day trades.

## User Preferences
- Muted Symbols: Disable signals for specific pairs
- Default PnL Period: Choose default view (today/week/month)
- DM Alerts: Toggle private message notifications
- Position Sizing: Users can set position size as percentage of account balance
- Max Positions: Limit simultaneous open positions
- Risk Filtering: Customizable accepted risk levels for signals
- Paper Trading Mode: Toggle between paper/live modes with `/toggle_paper_mode`
- Correlation Filter: Prevent opening correlated positions (e.g., BTC + ETH simultaneously)
- Funding Rate Alerts: Get notified of extreme funding rates for arbitrage opportunities
- Top Gainers Mode: Enable/disable automated trading of high-momentum coins (5x leverage, 20% TP/SL, max 3 positions)

## System Architecture

### Core Components
- **Day Trading Signal System**: Generates 1:1 risk-reward signals with a 6-point early entry system focusing on 5m/15m EMAs, institutional spot flow (>75% priority), early volume building, MACD turns, candle body formation, and high liquidity session validation. It employs a Smart Exit Protection system with 6 reversal detectors.
- **Precision Entry System**: Refines entries using 3-pattern candle detection and intelligent price optimization.
- **Reversal Bounce Catcher**: Scans for early breakout signals including S/R bounces, Bollinger squeezes, double bottom/top, RSI divergence, and volume spike reversals.
- **Telegram Bot**: Manages user interaction, commands, signal broadcasting, and an interactive dashboard.
- **FastAPI Server**: Provides health checks and webhook endpoints.
- **Bitunix Auto-Trading System**: Handles automated trade execution on Bitunix Futures with configurable leverage and risk management.
- **Coin Scanner Service**: An on-demand tool (`/scan`) for real-time market analysis of trend, volume, momentum, and institutional flow, providing weighted bias scores.
- **Top Gainers Trading Mode**: An automated, 24/7 SHORT-focused mean reversion system for volatile coins (25%+ daily gains minimum), prioritizing parabolic reversals (50%+) with fixed 5x leverage and triple TPs for parabolic dumps. Features triple entry paths: strong dumps (immediate), resumption patterns (safe), and early reversals (5m bearish + 15m bullish).
- **New Coin Alerts**: Automated detection of newly listed coins on Bitunix with high volume (scans every 5 minutes). Provides coin description from CoinGecko, volume/price stats, pump analysis (why it's moving), and category tags. Alerts only (not trade signals) for early opportunity awareness like COAI, ASTER, XPL.
- **News-Based Trading Signals**: AI-powered system leveraging CryptoNews API for market events and sentiment.
- **Admin Control System**: Provides user management, analytics, and system health monitoring.
- **Paper Trading System**: Offers a simulated trading environment.
- **Multi-Exchange Spot Market Monitor**: Tracks buying/selling pressure across major exchanges for flow alerts.
- **Referral Reward System**: Viral growth mechanism where users earn 14 days free for each successful referral who subscribes. Rewards automatically add to existing subscriptions or activate new ones. Each user gets a unique referral code (format: TH-XXXXXX) and shareable link.

### Technical Implementations
- **Hybrid Data Source Architecture**: Uses Binance Futures public API for technical analysis (candles) and Bitunix for tickers and trade execution due to Bitunix API issues.
- **Database**: PostgreSQL with SQLAlchemy ORM.
- **Configuration**: `pydantic-settings` for environment variables.
- **Security**: Fernet encryption for API credentials, HMAC-SHA256, and bearer token verification.
- **Risk Management**: Percentage-based SL/TP, risk-based position sizing, daily loss limits, max drawdown protection, and auto breakeven stop loss.
- **Analytics**: Tracks signal outcomes, win/loss ratios, and asset performance.
- **Health Monitoring**: Automatic checks, process restarts, and error handling.
- **Price Caching**: Thread-safe global price cache with 30-second TTL.
- **Multi-Analysis Confirmation**: Validates signals against higher timeframes (1H) for trend alignment, volume, momentum, and price structure.
- **NOWPayments Subscription System**: Integrated for crypto payment subscriptions with webhook-based auto-activation and access control.
- **Referral Tracking**: Database tracks referral_code (unique per user), referred_by (who referred them), and referral_credits (number of successful referrals). Rewards auto-apply as 14-day extensions when referrals subscribe.

### UI/UX Decisions
- Interactive Telegram dashboard with inline buttons.
- Clear HTML formatting for messages.
- Built-in help center.
- Auto-generation of professional, shareable trade screenshots for marketing.

## External Dependencies
- **Telegram Bot API**: `aiogram` library.
- **Cryptocurrency Exchanges**: `CCXT` library for Bitunix and Binance Futures API for data.
- **Database**: PostgreSQL.
- **Payment Gateway**: NOWPayments API.
- **Sentiment Analysis**: OpenAI API.
- **News Aggregation**: CryptoNews API.