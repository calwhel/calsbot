"""Gemini Gold guardrails unit tests — caps, reservation, trade gap."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.config import env_defaults, gemini_gold_enabled, gemini_gold_dry_run
from app.gemini_gold_trader.guardrails import (
    DemoAccountRequired,
    assert_demo_account,
    check_can_call_gemini,
    check_can_execute,
    demo_account_configured,
    effective_open_slots_used,
    in_flight_execution_count,
    merge_config,
    minutes_since_last_executed_trade,
    open_position_count,
    trades_today_effective,
    try_reserve_execution,
)
from app.gemini_gold_trader.models import GeminiGoldConfig, GeminiGoldDecision


class _FakePrefs:
    ctrader_accounts = '[{"ctidTraderAccountId": 99999, "isLive": false}]'


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    GeminiGoldConfig.__table__.create(bind=engine)
    GeminiGoldDecision.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    cfg_row = GeminiGoldConfig(
        id=1,
        max_calls_day=340,
        max_trades_day=4,
        enabled=True,
        kill_switch=False,
        dry_run=False,
        demo_user_id=42,
        demo_ctrader_account_id="12345",
    )
    session.add(cfg_row)
    session.commit()
    yield session
    session.close()


def _cfg(*, dry_run: bool = False, max_trades_day: int = 4, min_gap: int = 20):
    row = GeminiGoldConfig(id=1, max_calls_day=340, enabled=True, kill_switch=False, dry_run=dry_run)
    env = env_defaults()
    env.dry_run = dry_run
    env.demo_ctrader_account_id = "12345"
    env.demo_user_id = 42
    env.max_trades_day = max_trades_day
    env.min_trade_gap_min = min_gap
    return merge_config(row, env)


def test_feature_flag_default_off():
    os.environ.pop("GEMINI_GOLD_ENABLED", None)
    assert gemini_gold_enabled() is False


def test_dry_run_default_on():
    os.environ.pop("GEMINI_GOLD_DRY_RUN", None)
    assert gemini_gold_dry_run() is True


def test_demo_account_lock_rejects_wrong_ctid():
    cfg = env_defaults()
    cfg.demo_ctrader_account_id = "11111"
    try:
        assert_demo_account(_FakePrefs(), 99999, cfg)
        assert False, "expected DemoAccountRequired"
    except DemoAccountRequired:
        pass


def test_demo_account_configured():
    cfg = env_defaults()
    cfg.demo_ctrader_account_id = None
    assert demo_account_configured(cfg) is False
    cfg.demo_ctrader_account_id = "12345"
    assert demo_account_configured(cfg) is True


def test_check_can_call_gemini_respects_cap():
    row = GeminiGoldConfig(id=1, max_calls_day=340, enabled=True, kill_switch=False, dry_run=True)
    env = env_defaults()
    env.enabled = True
    cfg = merge_config(row, env)
    cfg.max_calls_day = 0
    db = MagicMock()
    with patch("app.gemini_gold_trader.guardrails.calls_today", return_value=1):
        ok, reason = check_can_call_gemini(db, cfg)
    assert ok is False
    assert reason == "max_calls_day"


def test_check_can_execute_blocks_dry_run():
    db = MagicMock()
    cfg = _cfg(dry_run=True)
    ok, reason = check_can_execute(db, cfg, 42)
    assert ok is False
    assert reason == "dry_run"


def test_env_defaults_max_calls_340():
    os.environ.pop("GEMINI_GOLD_MAX_CALLS_DAY", None)
    assert env_defaults().max_calls_day == 340


def test_trades_today_effective_counts_reserved(db_session):
    now = datetime.utcnow()
    db_session.add(
        GeminiGoldDecision(
            action="TAKE",
            executed=True,
            ts=now,
        )
    )
    db_session.add(
        GeminiGoldDecision(
            action="TAKE",
            executed=False,
            execution_reserved_at=now,
            ts=now,
        )
    )
    db_session.commit()
    assert trades_today_effective(db_session) == 2


def test_in_flight_execution_count_ignores_stale_reservation(db_session):
    now = datetime.utcnow()
    db_session.add(
        GeminiGoldDecision(
            action="TAKE",
            executed=False,
            execution_reserved_at=now - timedelta(minutes=45),
            ts=now - timedelta(minutes=45),
        )
    )
    db_session.commit()
    assert in_flight_execution_count(db_session) == 0


@patch("app.gemini_gold_trader.guardrails.open_position_count", return_value=0)
def test_check_can_execute_blocks_second_open_slot_from_reservation(
    _mock_open, db_session
):
    now = datetime.utcnow()
    db_session.add(
        GeminiGoldDecision(
            id=10,
            action="TAKE",
            executed=False,
            execution_reserved_at=now,
            ts=now,
        )
    )
    db_session.commit()
    cfg = _cfg()
    ok, reason = check_can_execute(db_session, cfg, 42)
    assert ok is False
    assert reason == "max_open_position"
    assert effective_open_slots_used(db_session, 42) == 1


@patch("app.gemini_gold_trader.guardrails.open_position_count", return_value=0)
def test_check_can_execute_blocks_trade_beyond_daily_cap(_mock_open, db_session):
    now = datetime.utcnow()
    for i in range(4):
        db_session.add(
            GeminiGoldDecision(
                action="TAKE",
                executed=True,
                ts=now - timedelta(minutes=60 - i),
            )
        )
    db_session.commit()
    cfg = _cfg(max_trades_day=4)
    ok, reason = check_can_execute(db_session, cfg, 42)
    assert ok is False
    assert reason == "max_trades_day"


@patch("app.gemini_gold_trader.guardrails.open_position_count", return_value=0)
def test_check_can_execute_blocks_min_trade_gap(_mock_open, db_session):
    now = datetime.utcnow()
    db_session.add(
        GeminiGoldDecision(
            action="TAKE",
            executed=True,
            ts=now - timedelta(minutes=5),
        )
    )
    db_session.commit()
    cfg = _cfg(min_gap=20)
    ok, reason = check_can_execute(db_session, cfg, 42)
    assert ok is False
    assert reason == "min_trade_gap"
    assert minutes_since_last_executed_trade(db_session) is not None


@patch("app.gemini_gold_trader.guardrails.open_position_count", return_value=0)
def test_try_reserve_execution_blocks_second_concurrent_take(_mock_open, db_session):
    now = datetime.utcnow()
    first = GeminiGoldDecision(
        id=1,
        action="TAKE",
        executed=False,
        execution_reserved_at=now,
        ts=now,
    )
    second = GeminiGoldDecision(id=2, action="TAKE", executed=False, ts=now)
    db_session.add_all([first, second])
    db_session.commit()
    cfg = _cfg()
    ok, reason = try_reserve_execution(db_session, cfg, 42, 2)
    assert ok is False
    assert reason == "max_open_position"


@patch("app.gemini_gold_trader.guardrails.open_position_count", return_value=0)
def test_try_reserve_execution_reserves_then_blocks_re_reserve(_mock_open, db_session):
    row = GeminiGoldDecision(id=7, action="TAKE", executed=False, ts=datetime.utcnow())
    db_session.add(row)
    db_session.commit()
    cfg = _cfg()
    ok1, reason1 = try_reserve_execution(db_session, cfg, 42, 7)
    assert ok1 is True
    assert reason1 == "ok"
    ok2, reason2 = try_reserve_execution(db_session, cfg, 42, 7)
    assert ok2 is False
    assert reason2 == "already_reserved"


@patch("app.gemini_gold_trader.guardrails.open_position_count", return_value=1)
def test_check_can_execute_blocks_when_broker_position_open(_mock_open, db_session):
    cfg = _cfg()
    ok, reason = check_can_execute(db_session, cfg, 42)
    assert ok is False
    assert reason == "max_open_position"
    _mock_open.assert_called()
