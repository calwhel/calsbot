"""Gemini Gold UTC hour performance stats."""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.models import GeminiGoldDecision, GeminiGoldOutcome
from app.gemini_gold_trader.timing_stats import hour_performance_stats


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    GeminiGoldDecision.__table__.create(bind=engine)
    GeminiGoldOutcome.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _seed_closed_trade(db, *, hour: int, result: str, session: str = "london"):
    ts = datetime.utcnow().replace(hour=hour, minute=15, second=0, microsecond=0)
    dec = GeminiGoldDecision(
        ts=ts,
        session=session,
        action="TAKE",
        executed=True,
        setup_type="fvg_retrace_bull",
        confidence=90,
    )
    db.add(dec)
    db.flush()
    out = GeminiGoldOutcome(
        decision_id=dec.id,
        session=session,
        result=result,
        closed_ts=ts + timedelta(hours=1),
    )
    db.add(out)
    db.commit()
    return dec


def test_hour_performance_stats_best_and_worst(db_session):
    _seed_closed_trade(db_session, hour=8, result="win")
    _seed_closed_trade(db_session, hour=8, result="win")
    _seed_closed_trade(db_session, hour=8, result="loss")
    _seed_closed_trade(db_session, hour=14, result="loss")
    _seed_closed_trade(db_session, hour=14, result="loss")

    take_only = GeminiGoldDecision(
        ts=datetime.utcnow().replace(hour=10, minute=0, second=0, microsecond=0),
        session="london",
        action="TAKE",
        executed=False,
        confidence=88,
    )
    db_session.add(take_only)
    db_session.commit()

    stats = hour_performance_stats(db_session, days=14, min_trades=2)
    assert stats["total_closed_trades"] == 5
    assert stats["best_hours"][0]["hour_utc"] == 8
    assert stats["best_hours"][0]["win_rate_pct"] == pytest.approx(66.7, abs=0.1)
    assert stats["worst_hours"][0]["hour_utc"] == 14
    assert stats["worst_hours"][0]["win_rate_pct"] == 0.0

    hour10 = next(b for b in stats["by_hour"] if b["hour_utc"] == 10)
    assert hour10["trades"] == 0
    assert hour10["take_count"] == 1
