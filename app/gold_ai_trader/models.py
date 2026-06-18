"""SQLAlchemy models — new tables only."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, JSON

from app.database import Base


class GoldAiConfig(Base):
    __tablename__ = "gold_ai_config"

    id = Column(Integer, primary_key=True, default=1)
    enabled = Column(Boolean, default=False, nullable=False)
    kill_switch = Column(Boolean, default=False, nullable=False)
    london_start_hour = Column(Integer, default=7, nullable=False)
    london_end_hour = Column(Integer, default=10, nullable=False)
    ny_start_hour = Column(Integer, default=13, nullable=False)
    ny_end_hour = Column(Integer, default=16, nullable=False)
    max_calls_day = Column(Integer, default=50, nullable=False)
    max_trades_day = Column(Integer, default=6, nullable=False)
    no_overnight = Column(Boolean, default=True, nullable=False)
    model = Column(String(64), default="claude-opus-4-8", nullable=False)
    demo_ctrader_account_id = Column(String(40), nullable=True)
    demo_user_id = Column(Integer, nullable=True)
    live_mirror_enabled = Column(Boolean, default=False, nullable=False)
    live_ctrader_account_id = Column(String(40), nullable=True)
    live_lot_size = Column(Float, default=0.01, nullable=False)
    max_live_trades_day = Column(Integer, default=3, nullable=False)
    live_mirror_confirmed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class GoldAiDecision(Base):
    __tablename__ = "gold_ai_decisions"

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime, default=datetime.utcnow, index=True)
    session = Column(String(16), nullable=True, index=True)
    candidate_type = Column(String(64), nullable=True)
    context_snapshot = Column(Text, nullable=True)
    reasoning = Column(Text, nullable=True)
    decision = Column(JSON, nullable=True)
    action = Column(String(16), nullable=True)
    confidence = Column(Integer, nullable=True)
    executed = Column(Boolean, default=False, nullable=False)
    execution_id = Column(Integer, nullable=True, index=True)
    live_mirror_execution_id = Column(Integer, nullable=True, index=True)
    live_mirror_status = Column(String(24), nullable=True)  # pending | filled | failed | skipped
    live_mirror_error = Column(Text, nullable=True)
    tokens_in = Column(Integer, default=0)
    tokens_out = Column(Integer, default=0)
    cache_read_tokens = Column(Integer, default=0)
    cache_write_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)


class GoldAiOutcome(Base):
    __tablename__ = "gold_ai_outcomes"

    id = Column(Integer, primary_key=True, index=True)
    decision_id = Column(Integer, index=True, nullable=False)
    result = Column(String(16), nullable=True)  # win | loss | breakeven
    pnl = Column(Float, nullable=True)
    mfe = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    closed_ts = Column(DateTime, nullable=True)


class GoldAiLesson(Base):
    __tablename__ = "gold_ai_lessons"

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime, default=datetime.utcnow, index=True)
    session = Column(String(16), nullable=True)
    digest = Column(Text, nullable=False)
