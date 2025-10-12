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
    
    preferences = relationship("UserPreference", back_populates="user", uselist=False)
    trades = relationship("Trade", back_populates="user")
    
    @property
    def is_subscribed(self):
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
    position_size_percent = Column(Float, default=10.0)
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
    take_profit = Column(Float, nullable=True)
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
    take_profit = Column(Float, nullable=True)
    position_size = Column(Float, default=0.0)  # Position size in USDT
    status = Column(String, default="open")
    pnl = Column(Float, default=0.0)
    pnl_percent = Column(Float, default=0.0)
    opened_at = Column(DateTime, default=datetime.utcnow, index=True)
    closed_at = Column(DateTime, nullable=True)
    
    user = relationship("User", back_populates="trades")
    signal = relationship("Signal", back_populates="trades")


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
