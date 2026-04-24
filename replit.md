# TradeHub Markets — Strategy Builder Platform

## Overview
TradeHub Markets is a web platform designed for users to build, test, and automate crypto trading strategies without requiring any coding knowledge. Users can create strategies using a 7-step wizard or an AI chat builder, paper test them for free, and then deploy them live or sell them in a marketplace. The platform operates on a Free vs. Pro ($50/month) subscription model. A companion Telegram bot provides strategy trade alerts and user login UIDs. The business vision is to empower traders with accessible, sophisticated tools, expanding market potential by democratizing strategy creation and automation in the crypto space.

## User Preferences
Not specified.

## System Architecture

### Core Components
- **Strategy Portal**: The main web application for strategy creation (wizard and AI chat builder), backtesting, strategy execution, and a marketplace.
- **Telegram Bot**: A minimal companion app for trade notifications, displaying user UIDs, and an on-demand liquidity wall scanner (`/walls`).
- **Trade Tracker**: A dashboard for monitoring trade performance.
- **Day Trading Page (`/trade`)**: A public, no-authentication chart with live order block overlays, indicator engine (SMA/EMA/RSI/MACD/BB/VWAP/ATR/StochRSI/SuperTrend), Pine script expression evaluator, and server-evaluated chart alerts delivered via Telegram.
- **Auto Trader (Pro)**: Allows users to save chart setups as background paper-trading strategies with AI mode (re-runs AI trade reads) or Rules mode (compiles visible indicators into a DSL).
- **Built-in Backtester**: Integrated into the strategy wizard for replaying strategies against historical OHLCV data to evaluate performance.
- **Strategy Creator Marketplace**: Enables users to publish, sell, and purchase trading strategies.
- **Subscription Tiers**: Free tier for basic strategy building and paper testing; Pro tier for advanced features like AI Chat Builder, backtesting, live automation, and marketplace copy.
- **Public Marketing Website**: Landing page (`/`) with features, marketplace preview, and leaderboard.
- **Web Authentication System**: Supports Google OAuth and email/password registration with HMAC-signed session cookies.

### Technical Implementations
- **Database**: Neon PostgreSQL, shared across all workflows, using SQLAlchemy ORM.
- **Configuration**: Environment variables managed with `pydantic-settings`.
- **Security**: Fernet encryption, HMAC-SHA256, bearer token verification, and PostgreSQL advisory locks.
- **Risk Management**: Includes percentage-based SL/TP, risk-based position sizing, daily loss limits, and max drawdown protection.
- **AI Integration**: Hybrid approach using Gemini 2.5 Flash for scanning and Claude Sonnet 4.5 for final approval and trade review.
- **Market Analysis**: Incorporates advanced techniques like Volume Profile Analysis, Order Flow Imbalance Detection, and Chart-Based TP/SL Optimization.
- **Signal Generation**: Priority-based scanning for various trade setups (PURE TA LONGS, VWAP BOUNCE SCALPS, PARABOLIC exhausted dumps, NORMAL SHORTS, RELIEF BOUNCE longs).
- **Automated Systems**: AI Chat Assistant for recommendations, AI Trade Reviewer, and an Automatic Market Regime Detector.
- **Real-time Data**: Primarily uses MEXC for market data due to Binance geoblocking on Replit.

### UI/UX Decisions
- **Dashboard**: Premium dashboard with live P&L, win rate, and BTC price header.
- **Navigation**: Grouped navigation buttons and consistent in-message transitions.
- **Signal Formatting**: Premium signal formatting with clear separators and tree-style TP/SL.
- **Interactive Elements**: Interactive Telegram dashboard with inline buttons.
- **User Identification**: Unique `TH-XXXXXXXX` UIDs for users.
- **Admin Dashboard**: Interactive panel for user management and system monitoring.

## External Dependencies
- **Telegram Bot API**: `aiogram`
- **Cryptocurrency Exchanges**: `CCXT` (Bitunix, MEXC, Gate.io, Kraken)
- **Database**: PostgreSQL
- **Payment Gateway**: OxaPay (for subscriptions)
- **AI Analysis**: Gemini 2.5 Flash, Claude Sonnet 4.5