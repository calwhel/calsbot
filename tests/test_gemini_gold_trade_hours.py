"""Gemini Gold configurable trading hours."""
from __future__ import annotations

import os
from datetime import datetime

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.config import env_defaults
from app.gemini_gold_trader.guardrails import merge_config
from app.gemini_gold_trader.models import GeminiGoldConfig
from app.gemini_gold_trader.trade_hours import (
    in_custom_trade_window,
    normalize_trade_sessions,
    resolve_trading_session,
    trade_schedule_summary,
)


def _cfg(**kwargs):
    row = GeminiGoldConfig(id=1, enabled=True, kill_switch=False, dry_run=True, **kwargs)
    return merge_config(row, env_defaults())


def test_normalize_trade_sessions_defaults_and_aliases():
    assert normalize_trade_sessions(None) == ("asia", "london", "new_york")
    assert normalize_trade_sessions("asian, ny, europe") == ("asia", "new_york", "london")
    assert normalize_trade_sessions(["london"]) == ("london",)
    assert normalize_trade_sessions([]) == ("asia", "london", "new_york")


def test_in_custom_trade_window_same_day_and_overnight():
    assert in_custom_trade_window(datetime(2026, 6, 30, 13, 0), "12:00", "21:00")
    assert not in_custom_trade_window(datetime(2026, 6, 30, 11, 59), "12:00", "21:00")
    assert in_custom_trade_window(datetime(2026, 6, 30, 23, 30), "22:00", "06:00")
    assert not in_custom_trade_window(datetime(2026, 6, 30, 12, 0), "22:00", "06:00")


@pytest.mark.parametrize(
    "now,sessions,custom,start,end,expected_session,reason",
    [
        (datetime(2026, 6, 30, 8, 0), ("london",), False, "12:00", "21:00", "london", "ok"),
        (datetime(2026, 6, 30, 8, 0), ("new_york",), False, "12:00", "21:00", None, "outside_session"),
        (
            datetime(2026, 6, 30, 13, 0),
            ("london",),
            True,
            "14:00",
            "21:00",
            None,
            "outside_trade_hours",
        ),
        (
            datetime(2026, 6, 30, 15, 0),
            ("london",),
            True,
            "14:00",
            "21:00",
            "london",
            "ok",
        ),
    ],
)
def test_resolve_trading_session(now, sessions, custom, start, end, expected_session, reason):
    cfg = _cfg(
        trade_sessions=list(sessions),
        custom_trade_hours_enabled=custom,
        trade_hours_start_utc=start,
        trade_hours_end_utc=end,
    )
    session, dormant_reason = resolve_trading_session(now, cfg)
    assert session == expected_session
    assert dormant_reason == reason


def test_trade_schedule_summary():
    cfg = _cfg(
        trade_sessions=["london", "new_york"],
        custom_trade_hours_enabled=True,
        trade_hours_start_utc="13:00",
        trade_hours_end_utc="20:00",
    )
    summary = trade_schedule_summary(cfg)
    assert "london" in summary
    assert "new_york" in summary
    assert "13:00" in summary
    assert "20:00" in summary
