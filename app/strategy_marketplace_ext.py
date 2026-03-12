"""
Strategy Marketplace Extended Models
Pricing, purchases, ratings, earnings — the creator economy layer.
New tables only.
"""
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    ForeignKey, Text, JSON, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from datetime import datetime

from app.database import Base


class StrategyPurchase(Base):
    """Records every strategy purchase/subscription."""
    __tablename__ = "strategy_purchases"

    id              = Column(Integer, primary_key=True, index=True)
    buyer_id        = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    listing_id      = Column(Integer, ForeignKey("strategy_marketplace.id"), nullable=False, index=True)
    strategy_id     = Column(Integer, ForeignKey("user_strategies.id"), nullable=False)
    pricing_model   = Column(String(20), nullable=False)   # free | one_time | subscription
    amount_paid_usd = Column(Float, default=0.0)
    status          = Column(String(20), default="active") # active | expired | refunded
    payment_ref     = Column(String(120), nullable=True)   # external payment ID
    purchased_at    = Column(DateTime, default=datetime.utcnow)
    expires_at      = Column(DateTime, nullable=True)      # null = lifetime

    # Cloned strategy for the buyer
    cloned_strategy_id = Column(Integer, ForeignKey("user_strategies.id"), nullable=True)

    __table_args__ = (
        UniqueConstraint("buyer_id", "listing_id", name="uq_purchase_buyer_listing"),
    )


class StrategyRating(Base):
    """Star rating + written review per strategy per user."""
    __tablename__ = "strategy_ratings"

    id          = Column(Integer, primary_key=True, index=True)
    rater_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    listing_id  = Column(Integer, ForeignKey("strategy_marketplace.id"), nullable=False, index=True)
    stars       = Column(Integer, nullable=False)    # 1-5
    review      = Column(Text, nullable=True)
    is_verified = Column(Boolean, default=False)     # True if buyer has active trades
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("rater_id", "listing_id", name="uq_rating_user_listing"),
    )


class CreatorEarnings(Base):
    """Tracks lifetime and pending earnings per creator."""
    __tablename__ = "creator_earnings"

    id              = Column(Integer, primary_key=True, index=True)
    creator_id      = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    total_earned    = Column(Float, default=0.0)      # lifetime gross
    pending_payout  = Column(Float, default=0.0)      # owed to creator (80%)
    total_paid_out  = Column(Float, default=0.0)      # already paid
    total_sales     = Column(Integer, default=0)
    total_subscribers = Column(Integer, default=0)
    last_updated    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EarningsTransaction(Base):
    """Individual sale event for earnings ledger."""
    __tablename__ = "earnings_transactions"

    id           = Column(Integer, primary_key=True, index=True)
    creator_id   = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    purchase_id  = Column(Integer, ForeignKey("strategy_purchases.id"), nullable=False)
    listing_id   = Column(Integer, ForeignKey("strategy_marketplace.id"), nullable=False)
    gross_amount = Column(Float, nullable=False)       # what buyer paid
    creator_cut  = Column(Float, nullable=False)       # 80% of gross
    platform_cut = Column(Float, nullable=False)       # 20% of gross
    created_at   = Column(DateTime, default=datetime.utcnow)


def init_marketplace_ext_tables(engine):
    """Create extended marketplace tables. Safe to call on every startup."""
    try:
        import app.models  # noqa
        import app.strategy_models  # noqa
    except Exception:
        pass
    Base.metadata.create_all(bind=engine, tables=[
        StrategyPurchase.__table__,
        StrategyRating.__table__,
        CreatorEarnings.__table__,
        EarningsTransaction.__table__,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Pricing helpers
# ─────────────────────────────────────────────────────────────────────────────

PLATFORM_CUT_PCT = 0.20   # 20% platform fee
CREATOR_CUT_PCT  = 0.80   # 80% to creator


def calculate_creator_cut(amount: float) -> float:
    return round(amount * CREATOR_CUT_PCT, 2)


def calculate_platform_cut(amount: float) -> float:
    return round(amount * PLATFORM_CUT_PCT, 2)


def format_price(listing) -> str:
    """Return display price string for a marketplace listing."""
    model = getattr(listing, "pricing_model", "free")
    if model == "free":
        return "FREE"
    if model == "one_time":
        price = getattr(listing, "price_usdt", 0) or 0
        return f"${price:.2f}"
    if model == "subscription":
        price = getattr(listing, "price_usdt", 0) or 0
        return f"${price:.2f}/mo"
    return "FREE"
