"""
User Strategy Models — Build Your Own Strategy Portal
New tables only — does not modify any existing tables.
"""
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    ForeignKey, Text, JSON, UniqueConstraint
)
from sqlalchemy.orm import relationship
from datetime import datetime

from app.database import Base


class UserStrategy(Base):
    __tablename__ = "user_strategies"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name        = Column(String(120), nullable=False)
    description = Column(Text, nullable=True)
    config      = Column(JSON, nullable=False)          # full strategy JSON spec
    status      = Column(String(20), default="draft")   # draft | active | paused | archived
    is_public   = Column(Boolean, default=False)        # marketplace visibility
    clone_count = Column(Integer, default=0)
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    executions  = relationship("StrategyExecution", back_populates="strategy", cascade="all, delete-orphan")
    performance = relationship("StrategyPerformance", back_populates="strategy", uselist=False, cascade="all, delete-orphan")


class StrategyExecution(Base):
    __tablename__ = "strategy_executions"

    id              = Column(Integer, primary_key=True, index=True)
    strategy_id     = Column(Integer, ForeignKey("user_strategies.id"), nullable=False, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    symbol          = Column(String(30), nullable=False)
    direction       = Column(String(10), nullable=False)   # LONG | SHORT
    entry_price     = Column(Float, nullable=True)
    exit_price      = Column(Float, nullable=True)
    tp_price        = Column(Float, nullable=True)
    sl_price        = Column(Float, nullable=True)
    leverage        = Column(Integer, default=10)
    position_size   = Column(Float, nullable=True)         # USD notional
    outcome         = Column(String(20), default="OPEN")   # OPEN | WIN | LOSS | BREAKEVEN | CANCELLED
    pnl_pct         = Column(Float, nullable=True)
    pnl_usd         = Column(Float, nullable=True)
    conditions_met  = Column(JSON, nullable=True)          # which conditions triggered
    fired_at        = Column(DateTime, default=datetime.utcnow)
    closed_at       = Column(DateTime, nullable=True)
    bitunix_order_id = Column(String(80), nullable=True)
    notes           = Column(Text, nullable=True)

    strategy = relationship("UserStrategy", back_populates="executions")


class StrategyPerformance(Base):
    __tablename__ = "strategy_performance"

    id            = Column(Integer, primary_key=True, index=True)
    strategy_id   = Column(Integer, ForeignKey("user_strategies.id"), unique=True, nullable=False)
    total_trades  = Column(Integer, default=0)
    open_trades   = Column(Integer, default=0)
    wins          = Column(Integer, default=0)
    losses        = Column(Integer, default=0)
    breakevens    = Column(Integer, default=0)
    win_rate      = Column(Float, default=0.0)
    total_pnl_pct = Column(Float, default=0.0)
    avg_win_pct   = Column(Float, default=0.0)
    avg_loss_pct  = Column(Float, default=0.0)
    avg_rr        = Column(Float, default=0.0)
    best_trade    = Column(Float, default=0.0)
    worst_trade   = Column(Float, default=0.0)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    strategy = relationship("UserStrategy", back_populates="performance")


class StrategyMarketplace(Base):
    __tablename__ = "strategy_marketplace"

    id              = Column(Integer, primary_key=True, index=True)
    strategy_id     = Column(Integer, ForeignKey("user_strategies.id"), unique=True, nullable=False)
    author_id       = Column(Integer, ForeignKey("users.id"), nullable=False)
    title           = Column(String(120), nullable=False)
    summary         = Column(Text, nullable=True)            # AI-generated description
    tags            = Column(JSON, default=list)             # ["scalp", "reversal", "SMC"]
    category        = Column(String(40), default="general")  # scalp|swing|reversal|smc|breakout|momentum
    # Pricing
    pricing_model   = Column(String(20), default="free")     # free | one_time | subscription
    price_usdt      = Column(Float, default=0.0)             # price in USD
    # Stats
    clone_count     = Column(Integer, default=0)
    subscriber_count = Column(Integer, default=0)
    avg_rating      = Column(Float, default=0.0)
    rating_count    = Column(Integer, default=0)
    # Verification
    is_verified     = Column(Boolean, default=False)         # 10+ live trades tracked
    verified_trades = Column(Integer, default=0)
    verified_win_rate = Column(Float, default=0.0)
    verified_pnl    = Column(Float, default=0.0)
    # Meta
    published_at    = Column(DateTime, default=datetime.utcnow)
    is_featured     = Column(Boolean, default=False)
    is_trending     = Column(Boolean, default=False)
    view_count      = Column(Integer, default=0)


def init_strategy_tables(engine):
    """Create strategy tables if they don't exist. Safe to call on every startup."""
    # Import main models so SQLAlchemy knows about the users table (needed for FK resolution)
    try:
        import app.models  # noqa — registers User and other tables on the shared Base
    except Exception:
        pass
    Base.metadata.create_all(bind=engine, tables=[
        UserStrategy.__table__,
        StrategyExecution.__table__,
        StrategyPerformance.__table__,
        StrategyMarketplace.__table__,
    ])
