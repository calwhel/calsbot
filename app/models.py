from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
from app.database import Base


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    subscription_end = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    is_admin = Column(Boolean, default=False)
    banned = Column(Boolean, default=False)
    approved = Column(Boolean, default=False)
    admin_notes = Column(Text, nullable=True)
    grandfathered = Column(Boolean, default=False)  # Existing users = free forever
    nowpayments_subscription_id = Column(String, nullable=True)  # NOWPayments subscription ID
    subscription_type = Column(String, default="manual")  # "manual" ($80 Signals Only), or "auto" ($150 Auto-Trading)
    
    # Referral system
    referral_code = Column(String, unique=True, nullable=True)  # User's unique referral code
    referred_by = Column(String, nullable=True)  # Referral code of who referred this user
    referral_credits = Column(Integer, default=0)  # Free months earned from referrals (deprecated - now using cash rewards)
    referral_earnings = Column(Float, default=0.0)  # Pending $50 crypto payouts from referrals
    paid_referrals = Column(Text, default="")  # JSON list of user IDs that have been paid out for
    crypto_wallet = Column(String, nullable=True)  # User's crypto wallet address for payouts
    
    preferences = relationship("UserPreference", back_populates="user", uselist=False)
    trades = relationship("Trade", back_populates="user")
    
    @property
    def is_subscribed(self):
        """Check if user has active subscription OR is grandfathered (free forever)"""
        if self.grandfathered:
            return True
        if not self.subscription_end:
            return False
        return datetime.utcnow() < self.subscription_end


class UserPreference(Base):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    muted_symbols = Column(Text, default="")
    default_pnl_period = Column(String, default="today")
    dm_alerts = Column(Boolean, default=True)
    
    auto_trading_enabled = Column(Boolean, default=False)
    mexc_api_key = Column(String, nullable=True)
    mexc_api_secret = Column(String, nullable=True)
    okx_api_key = Column(String, nullable=True)
    okx_api_secret = Column(String, nullable=True)
    okx_passphrase = Column(String, nullable=True)
    kucoin_api_key = Column(String, nullable=True)
    kucoin_api_secret = Column(String, nullable=True)
    kucoin_passphrase = Column(String, nullable=True)
    bitunix_api_key = Column(String, nullable=True)
    bitunix_api_secret = Column(String, nullable=True)
    preferred_exchange = Column(String, default="Bitunix")  # Bitunix (legacy: MEXC, OKX, KuCoin)
    position_size_percent = Column(Float, default=10.0)
    position_size_dollars = Column(Float, nullable=True)  # Fixed $ amount per trade (if set, overrides %)
    max_positions = Column(Integer, default=3)
    
    # Risk management settings
    accepted_risk_levels = Column(String, default="LOW,MEDIUM")  # Comma-separated: LOW, MEDIUM
    risk_based_sizing = Column(Boolean, default=True)  # Reduce position size for higher risk
    use_trailing_stop = Column(Boolean, default=False)  # Enable trailing stops
    trailing_stop_percent = Column(Float, default=2.0)  # Trailing stop percentage
    use_breakeven_stop = Column(Boolean, default=True)  # Move SL to breakeven when in profit
    
    # Security settings
    daily_loss_limit = Column(Float, default=100.0)  # Max daily loss in USD
    max_drawdown_percent = Column(Float, default=20.0)  # Max drawdown from peak balance %
    min_balance = Column(Float, default=50.0)  # Minimum balance to trade
    max_consecutive_losses = Column(Integer, default=3)  # Stop after N consecutive losses
    cooldown_after_loss = Column(Integer, default=60)  # Cooldown minutes after hitting limit
    emergency_stop = Column(Boolean, default=False)  # Emergency stop all trading
    safety_paused = Column(Boolean, default=False)  # Paused by safety limits (can auto-resume)
    
    # Tracking fields
    peak_balance = Column(Float, default=0.0)  # Track peak balance for drawdown
    daily_loss_reset_date = Column(DateTime, nullable=True)  # Track daily reset
    consecutive_losses = Column(Integer, default=0)  # Track consecutive losses
    last_loss_time = Column(DateTime, nullable=True)  # Track when cooldown started
    
    # News signal preferences
    news_signals_enabled = Column(Boolean, default=True)
    min_news_impact = Column(Integer, default=9)
    min_news_confidence = Column(Integer, default=80)
    
    # Partial take profit percentages
    tp1_percent = Column(Integer, default=30)  # % to close at TP1
    tp2_percent = Column(Integer, default=30)  # % to close at TP2
    tp3_percent = Column(Integer, default=40)  # % to close at TP3
    
    # Advanced autotrader features
    user_leverage = Column(Integer, default=10)  # User configurable leverage (1-20x)
    
    # Advanced trailing stop settings
    trailing_activation_percent = Column(Float, default=2.0)  # % profit to activate trailing
    trailing_step_percent = Column(Float, default=1.0)  # Trail distance from price
    
    # Win rate adaptive sizing
    adaptive_sizing_enabled = Column(Boolean, default=True)  # Scale size based on performance
    win_streak_multiplier = Column(Float, default=1.2)  # Increase size after wins (max 1.5x)
    loss_streak_divider = Column(Float, default=0.8)  # Decrease size after losses (min 0.5x)
    current_win_streak = Column(Integer, default=0)  # Track current streak
    
    # Anti-overtrading filters
    trade_cooldown_minutes = Column(Integer, default=15)  # Cooldown between any trades
    max_trades_per_day = Column(Integer, default=10)  # Maximum trades per day
    same_symbol_cooldown_minutes = Column(Integer, default=60)  # Cooldown for same symbol
    
    # Tracking for anti-overtrading
    last_trade_time = Column(DateTime, nullable=True)  # Last trade timestamp
    trades_today = Column(Integer, default=0)  # Count of trades today
    trades_reset_date = Column(DateTime, nullable=True)  # Track daily reset
    last_symbol_trades = Column(Text, default="")  # JSON of symbol: timestamp
    
    # Top Gainers Trading Mode
    top_gainers_mode_enabled = Column(Boolean, default=False)  # Enable trading top gainers from Bitunix
    top_gainers_trade_mode = Column(String, default="shorts_only")  # Mode: 'shorts_only', 'longs_only', or 'both'
    top_gainers_max_symbols = Column(Integer, default=3)  # Max top gainer positions simultaneously
    top_gainers_min_change = Column(Float, default=20.0)  # Minimum 24h change % to qualify as "gainer" (20%+ = parabolic pumps for mean reversion)
    top_gainers_leverage = Column(Integer, default=5)  # Leverage for top gainer trades (1-20x, default 5x for safety)
    
    # Top Gainers Auto-Compound (Upgrade #7)
    top_gainers_win_streak = Column(Integer, default=0)  # Current win streak for top gainer trades
    top_gainers_position_multiplier = Column(Float, default=1.0)  # Current position size multiplier (1.0 = base, 1.2 = +20%)
    top_gainers_auto_compound = Column(Boolean, default=True)  # Enable auto-compounding after 3 wins
    
    # Market condition adaptive settings
    market_condition_adaptive = Column(Boolean, default=True)  # Adjust for conditions
    volatility_threshold_high = Column(Float, default=5.0)  # ATR % for high volatility
    volatility_threshold_low = Column(Float, default=1.5)  # ATR % for low volatility
    high_volatility_size_reduction = Column(Float, default=0.6)  # Reduce to 60% in high vol
    
    # Better entry orders
    use_limit_orders = Column(Boolean, default=False)  # Use limit orders for entry
    entry_slippage_percent = Column(Float, default=0.3)  # Max slippage for limit orders
    limit_order_timeout_seconds = Column(Integer, default=30)  # Cancel if not filled
    
    # Smart risk-reward scaling
    rr_scaling_enabled = Column(Boolean, default=True)  # Scale position by R:R ratio
    min_rr_for_full_size = Column(Float, default=3.0)  # Need 3R for full position
    rr_scaling_multiplier = Column(Float, default=0.3)  # +30% size per R above 3
    
    # Correlation filter (prevent correlated positions)
    correlation_filter_enabled = Column(Boolean, default=True)  # Prevent correlated trades
    max_correlated_positions = Column(Integer, default=1)  # Max positions in same correlation group
    
    # Funding rate alerts
    funding_rate_alerts_enabled = Column(Boolean, default=True)  # Alert on extreme funding
    funding_rate_threshold = Column(Float, default=0.1)  # Alert when funding > 0.1% (8hr)
    
    # Scalp Mode Settings
    scalp_mode_enabled = Column(Boolean, default=False)  # Enable/disable scalp trading
    scalp_position_size_percent = Column(Float, default=1.0)  # Position size as % of account (1-5%)
    
    user = relationship("User", back_populates="preferences")
    
    def get_muted_symbols_list(self):
        if not self.muted_symbols:
            return []
        return [s.strip() for s in self.muted_symbols.split(",") if s.strip()]
    
    def add_muted_symbol(self, symbol: str):
        muted = self.get_muted_symbols_list()
        if symbol not in muted:
            muted.append(symbol)
            self.muted_symbols = ",".join(muted)
    
    def remove_muted_symbol(self, symbol: str):
        muted = self.get_muted_symbols_list()
        if symbol in muted:
            muted.remove(symbol)
            self.muted_symbols = ",".join(muted)


class Signal(Base):
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    direction = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)  # TP3 (backward compatible)
    take_profit_1 = Column(Float, nullable=True)  # 1.5R - 30%
    take_profit_2 = Column(Float, nullable=True)  # 2.5R - 30%
    take_profit_3 = Column(Float, nullable=True)  # 4R - 40%
    support_level = Column(Float, nullable=True)
    resistance_level = Column(Float, nullable=True)
    ema_fast = Column(Float, nullable=True)
    ema_slow = Column(Float, nullable=True)
    ema_trend = Column(Float, nullable=True)
    rsi = Column(Float, nullable=True)
    atr = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    volume_avg = Column(Float, nullable=True)
    timeframe = Column(String, nullable=False)
    risk_level = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    signal_type = Column(String, default='technical')
    pattern = Column(String, nullable=True, index=True)  # Specific pattern: DOUBLE_BOTTOM, RSI_DIVERGENCE, FUNDING_EXTREME, etc.
    news_title = Column(Text, nullable=True)
    news_url = Column(Text, nullable=True)
    news_source = Column(String, nullable=True)
    sentiment = Column(String, nullable=True)
    impact_score = Column(Integer, nullable=True)
    confidence = Column(Integer, nullable=True)
    reasoning = Column(Text, nullable=True)
    
    outcome = Column(String, nullable=True)
    total_pnl = Column(Float, default=0.0)
    total_pnl_percent = Column(Float, default=0.0)
    trades_count = Column(Integer, default=0)
    
    trades = relationship("Trade", back_populates="signal")


class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    signal_id = Column(Integer, ForeignKey("signals.id"))
    symbol = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)  # TP3 (backward compatible)
    take_profit_1 = Column(Float, nullable=True)  # 1.5R
    take_profit_2 = Column(Float, nullable=True)  # 2.5R
    take_profit_3 = Column(Float, nullable=True)  # 4R
    position_size = Column(Float, default=0.0)  # Position size in USDT
    remaining_size = Column(Float, default=0.0)  # Remaining position size after partial closes
    tp1_hit = Column(Boolean, default=False)  # Track which TPs have been hit
    tp2_hit = Column(Boolean, default=False)
    tp3_hit = Column(Boolean, default=False)
    status = Column(String, default="open")
    pnl = Column(Float, default=0.0)
    pnl_percent = Column(Float, default=0.0)
    opened_at = Column(DateTime, default=datetime.utcnow, index=True)
    closed_at = Column(DateTime, nullable=True)
    
    # Advanced tracking fields
    trailing_stop_price = Column(Float, nullable=True)  # Current trailing stop price
    trailing_active = Column(Boolean, default=False)  # Is trailing stop active
    breakeven_moved = Column(Boolean, default=False)  # Has SL been moved to breakeven
    highest_price = Column(Float, nullable=True)  # Track highest price for LONG
    lowest_price = Column(Float, nullable=True)  # Track lowest price for SHORT
    
    # Exchange-reported PnL (live API data from Bitunix)
    exchange_unrealized_pnl = Column(Float, nullable=True)  # Real-time unrealized PnL from exchange
    exchange_realized_pnl = Column(Float, nullable=True)  # Realized PnL from exchange
    last_sync_at = Column(DateTime, nullable=True)  # Last time we synced with exchange API
    
    # Trade type classification
    trade_type = Column(String, default='STANDARD', nullable=False)  # STANDARD, TOP_GAINER, NEWS, etc.
    
    user = relationship("User", back_populates="trades")
    signal = relationship("Signal", back_populates="trades")


class SpotActivity(Base):
    __tablename__ = "spot_activities"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    flow_signal = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    avg_imbalance = Column(Float, nullable=True)
    avg_pressure = Column(Float, nullable=True)
    total_volume = Column(Float, nullable=True)
    exchanges_count = Column(Integer, nullable=True)
    spike_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class Subscription(Base):
    __tablename__ = "subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    payment_method = Column(String, nullable=False)
    transaction_id = Column(String, nullable=True)
    amount = Column(Float, nullable=False)
    duration_days = Column(Integer, default=30)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User")


class PendingInvoice(Base):
    """
    Tracks pending OxaPay invoices for automatic payment verification
    Polls OxaPay API every 60 seconds to check payment status
    """
    __tablename__ = "pending_invoices"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    track_id = Column(String, nullable=False, unique=True, index=True)  # OxaPay trackId
    order_id = Column(String, nullable=False, index=True)  # Our order ID
    plan_type = Column(String, nullable=False)  # "scan", "manual", or "auto"
    amount = Column(Float, nullable=False)  # Amount charged
    status = Column(String, default="pending", nullable=False)  # pending, paid, expired, failed
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    activated_at = Column(DateTime, nullable=True)  # When subscription was activated
    
    user = relationship("User")
    
    @property
    def is_expired(self):
        """Check if invoice is >2 hours old (OxaPay invoices expire after 60 minutes)"""
        return (datetime.utcnow() - self.created_at).total_seconds() > 7200


class TopGainerWatchlist(Base):
    """
    Tracks top gainers for 48 hours to catch delayed dumps.
    
    Many parabolic pumps take 24-48 hours to fully reverse, so we continue
    monitoring yesterday's top gainers even if they drop off the current list.
    """
    __tablename__ = "top_gainer_watchlist"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False, index=True, unique=True)  # e.g., "AIXBT/USDT"
    first_seen = Column(DateTime, default=datetime.utcnow, nullable=False)  # When added to watchlist
    peak_price = Column(Float, nullable=False)  # Highest price observed
    peak_change_percent = Column(Float, nullable=False)  # Highest 24h % change observed
    last_checked = Column(DateTime, default=datetime.utcnow, nullable=False)  # Last scan time
    still_monitoring = Column(Boolean, default=True, nullable=False)  # False after reversal signal sent
    
    @property
    def hours_tracked(self):
        """Calculate how long we've been tracking this symbol"""
        return (datetime.utcnow() - self.first_seen).total_seconds() / 3600
    
    @property
    def should_expire(self):
        """Check if this entry should be removed (>48 hours old)"""
        return self.hours_tracked > 48


class SupportTicket(Base):
    """Support ticket system for users to contact admins anonymously"""
    __tablename__ = "support_tickets"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    subject = Column(String, nullable=False)  # Quick category
    message = Column(Text, nullable=False)  # User's question/issue
    status = Column(String, default="open")  # open, in_progress, closed
    priority = Column(String, default="normal")  # low, normal, high
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    
    # Admin response
    admin_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    admin_response = Column(Text, nullable=True)
    admin_responded_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id], backref="support_tickets")
    admin = relationship("User", foreign_keys=[admin_id])


class ReferralPayout(Base):
    """Track referral payout requests and status"""
    __tablename__ = "referral_payouts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    amount_usd = Column(Float, nullable=False)  # Amount requested
    wallet_address = Column(String, nullable=False)  # User's crypto wallet
    payment_method = Column(String, default="USDT_TRC20")  # USDT_TRC20, USDT_ERC20, etc.
    status = Column(String, default="pending")  # pending, approved, paid, rejected
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Admin fields
    admin_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    admin_notes = Column(Text, nullable=True)
    transaction_hash = Column(String, nullable=True)  # Blockchain tx hash
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id], backref="referral_payouts")
    admin = relationship("User", foreign_keys=[admin_id])


class ScalpSignal(Base):
    """Tracks recent scalp signals to prevent duplicates within time window"""
    __tablename__ = "scalp_signals"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    direction = Column(String, nullable=False)  # LONG or SHORT
    entry_price = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class TradeAttempt(Base):
    """Tracks every trade execution attempt for debugging missed trades"""
    __tablename__ = "trade_attempts"
    
    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(Integer, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    symbol = Column(String, nullable=False, index=True)
    direction = Column(String, nullable=False)
    
    status = Column(String, nullable=False)  # success, skipped, failed, error
    reason = Column(Text, nullable=True)  # Why skipped/failed
    
    balance_at_attempt = Column(Float, nullable=True)
    position_size = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    user = relationship("User", backref="trade_attempts")
