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
    confidence = Column(Integer, nullable=True)
    rationale = Column(Text, nullable=True)
    chart_meta = Column(JSON, nullable=True)
    executed = Column(Boolean, default=False, nullable=False)
    execution_reserved_at = Column(DateTime, nullable=True, index=True)
    execution_id = Column(Integer, nullable=True, index=True)
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
