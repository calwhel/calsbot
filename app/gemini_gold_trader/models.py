"""SQLAlchemy models — gemini_gold_* tables only."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, Text

from app.database import Base


class GeminiGoldConfig(Base):
    __tablename__ = "gemini_gold_config"

    id = Column(Integer, primary_key=True, default=1)
    enabled = Column(Boolean, default=False, nullable=False)
    kill_switch = Column(Boolean, default=False, nullable=False)
    dry_run = Column(Boolean, default=True, nullable=False)
    max_calls_day = Column(Integer, default=340, nullable=False)
    max_trades_day = Column(Integer, default=4, nullable=False)
    model = Column(String(64), default="gemini-2.5-flash", nullable=False)
    demo_ctrader_account_id = Column(String(40), nullable=True)
    demo_user_id = Column(Integer, nullable=True)
    demo_lot_size = Column(Float, default=0.01, nullable=False)
    execution_mode = Column(String(16), default="demo", nullable=False)
    live_ctrader_account_id = Column(String(40), nullable=True)
    live_lot_size = Column(Float, default=0.01, nullable=False)
    live_confirmed_at = Column(DateTime, nullable=True)
    live_mirror_enabled = Column(Boolean, default=False, nullable=False)
    max_live_trades_day = Column(Integer, default=3, nullable=False)
    live_mirror_confirmed_at = Column(DateTime, nullable=True)
    use_limit_entry = Column(Boolean, default=True, nullable=False)
    pending_entry_timeout_min = Column(Integer, default=30, nullable=False)
    orb_enabled = Column(Boolean, default=False, nullable=False)
    orb_confidence_threshold = Column(Integer, default=65, nullable=False)
    orb_max_calls_day = Column(Integer, default=20, nullable=False)
    orb_max_trades_per_session = Column(Integer, default=1, nullable=False)
    confidence_threshold = Column(Integer, default=85, nullable=False)
    calls_reset_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class GeminiGoldDecision(Base):
    __tablename__ = "gemini_gold_decisions"

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime, default=datetime.utcnow, index=True)
    session = Column(String(16), nullable=True, index=True)
    decision = Column(JSON, nullable=True)
    action = Column(String(16), nullable=True, index=True)
    direction = Column(String(8), nullable=True)
    setup_type = Column(String(64), nullable=True, index=True)
    confidence = Column(Integer, nullable=True)
    rationale = Column(Text, nullable=True)
    chart_meta = Column(JSON, nullable=True)
    executed = Column(Boolean, default=False, nullable=False)
    execution_reserved_at = Column(DateTime, nullable=True, index=True)
    execution_id = Column(Integer, nullable=True, index=True)
    live_mirror_execution_id = Column(Integer, nullable=True, index=True)
    live_mirror_status = Column(String(24), nullable=True)
    live_mirror_error = Column(Text, nullable=True)
    dry_run = Column(Boolean, default=True, nullable=False)
    skip_reason = Column(String(128), nullable=True)
    tokens_in = Column(Integer, default=0)
    tokens_out = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)


class GeminiGoldOutcome(Base):
    __tablename__ = "gemini_gold_outcomes"

    id = Column(Integer, primary_key=True, index=True)
    decision_id = Column(Integer, index=True, nullable=False)
    session = Column(String(16), nullable=True, index=True)
    result = Column(String(16), nullable=True)
    pnl = Column(Float, nullable=True)
    r_multiple = Column(Float, nullable=True)
    closed_ts = Column(DateTime, nullable=True)
    setup_type = Column(String(64), nullable=True, index=True)


class GeminiGoldFunnelEvent(Base):
    __tablename__ = "gemini_gold_funnel_events"

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime, default=datetime.utcnow, index=True)
    session = Column(String(16), nullable=True, index=True)
    event = Column(String(32), nullable=False, index=True)
    setup_type = Column(String(64), nullable=True)
    reason = Column(String(256), nullable=True)
    decision_id = Column(Integer, nullable=True, index=True)


class GeminiGoldOrbState(Base):
    __tablename__ = "gemini_gold_orb_state"

    id = Column(Integer, primary_key=True, index=True)
    trade_day_utc = Column(DateTime, nullable=False, index=True)
    session = Column(String(16), nullable=False, index=True)
    status = Column(String(24), nullable=True)
    range_start_ts = Column(DateTime, nullable=True)
    range_end_ts = Column(DateTime, nullable=True)
    trade_window_end_ts = Column(DateTime, nullable=True)
    range_high = Column(Float, nullable=True)
    range_low = Column(Float, nullable=True)
    range_height = Column(Float, nullable=True)
    breakout_side = Column(String(8), nullable=True)
    breakout_level = Column(Float, nullable=True)
    breakout_ts = Column(DateTime, nullable=True)
    trades_taken = Column(Integer, default=0, nullable=False)
    decision_id = Column(Integer, nullable=True)
    execution_id = Column(Integer, nullable=True)


class GeminiGoldPendingOrder(Base):
    __tablename__ = "gemini_gold_pending_orders"

    id = Column(Integer, primary_key=True, index=True)
    decision_id = Column(Integer, index=True, nullable=False)
    session = Column(String(16), nullable=True)
    direction = Column(String(8), nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    status = Column(String(16), default="pending", nullable=False, index=True)
    method = Column(String(24), nullable=True)
    broker_order_id = Column(String(64), nullable=True)
    fill_execution_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)


class GeminiGoldReview(Base):
    __tablename__ = "gemini_gold_reviews"

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime, default=datetime.utcnow, index=True)
    summary = Column(Text, nullable=True)
    whats_working = Column(JSON, nullable=True)
    whats_not_working = Column(JSON, nullable=True)
    setup_insights = Column(JSON, nullable=True)
    funnel_diagnosis = Column(Text, nullable=True)
    lesson_for_next_sessions = Column(Text, nullable=True)
    config_suggestions = Column(JSON, nullable=True)
    model = Column(String(64), nullable=True)
    days_window = Column(Integer, default=14, nullable=False)
    tokens_in = Column(Integer, default=0)
    tokens_out = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    account_snapshot = Column(JSON, nullable=True)
