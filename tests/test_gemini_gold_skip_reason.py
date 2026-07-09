"""Gemini Gold skip_reason persistence."""
import os
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.loop import _set_decision_skip_reason
from app.gemini_gold_trader.models import GeminiGoldDecision


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    GeminiGoldDecision.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_set_decision_skip_reason_persists(db_session):
    row = GeminiGoldDecision(
        ts=datetime.utcnow(),
        session="london",
        action="TAKE",
        executed=False,
        confidence=85,
    )
    db_session.add(row)
    db_session.commit()
    db_session.refresh(row)

    _set_decision_skip_reason(db_session, row.id, "dry_run")
    db_session.refresh(row)
    assert row.skip_reason == "dry_run"

    row.executed = True
    db_session.commit()
    _set_decision_skip_reason(db_session, row.id, "should_not_overwrite")
    db_session.refresh(row)
    assert row.skip_reason == "dry_run"
