"""Gemini Gold entry routing and pending-entry tolerance tests."""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.config import env_defaults
from app.gemini_gold_trader.entry_routing import (
    entry_touch_tolerance,
    is_market_setup_type,
    market_fallback_on_pending_expire,
    use_limit_entry_for_setup,
)


def _cfg(**kwargs):
    base = env_defaults()
    return base.__class__(**{**base.__dict__, **kwargs})


def test_liquidity_grab_uses_market_entry():
    cfg = _cfg(use_limit_entry=True)
    assert use_limit_entry_for_setup("liquidity_grab_short", cfg) is False
    assert is_market_setup_type("liquidity_grab_short") is True


def test_fvg_retrace_uses_limit_when_enabled():
    cfg = _cfg(use_limit_entry=True)
    assert use_limit_entry_for_setup("fvg_retrace_bear", cfg) is True
    assert is_market_setup_type("fvg_retrace_bear") is False


def test_unknown_setup_defaults_to_market_when_limit_disabled():
    cfg = _cfg(use_limit_entry=False)
    assert use_limit_entry_for_setup("custom_setup", cfg) is False


def test_unknown_setup_respects_limit_flag():
    cfg = _cfg(use_limit_entry=True)
    assert use_limit_entry_for_setup("custom_setup", cfg) is True


def test_entry_touch_tolerance_wider_for_liquidity_grab():
    grab_tol = entry_touch_tolerance(2650.0, "liquidity_grab_short")
    fvg_tol = entry_touch_tolerance(2650.0, "fvg_retrace_bear")
    assert grab_tol > fvg_tol
    assert grab_tol >= 0.90


def test_market_fallback_on_expire_for_grab_setup():
    cfg = _cfg(use_limit_entry=True)
    assert market_fallback_on_pending_expire("liquidity_grab_short", cfg) is True


def test_market_fallback_on_expire_when_limit_disabled():
    cfg = _cfg(use_limit_entry=False)
    assert market_fallback_on_pending_expire("fvg_retrace_bear", cfg) is True


def test_market_fallback_blocked_for_limit_only_setups():
    cfg = _cfg(use_limit_entry=True)
    assert market_fallback_on_pending_expire("fvg_retrace_bear", cfg) is False


def test_migrate_use_limit_entry_to_false():
    from unittest.mock import MagicMock

    from app.gemini_gold_trader.schema import _migrate_legacy_use_limit_entry

    row = MagicMock()
    row.use_limit_entry = True
    db = MagicMock()
    _migrate_legacy_use_limit_entry(db, row=row)
    assert row.use_limit_entry is False
    db.commit.assert_called_once()
