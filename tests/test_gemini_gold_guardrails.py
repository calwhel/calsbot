"""Gemini Gold guardrails unit tests."""
import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.config import env_defaults, gemini_gold_enabled, gemini_gold_dry_run
from app.gemini_gold_trader.guardrails import (
    DemoAccountRequired,
    assert_demo_account,
    check_can_call_gemini,
    check_can_execute,
    demo_account_configured,
    merge_config,
)
from app.gemini_gold_trader.models import GeminiGoldConfig


class _FakePrefs:
    ctrader_accounts = '[{"ctidTraderAccountId": 99999, "isLive": false}]'


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
    row = GeminiGoldConfig(id=1, max_calls_day=340, enabled=True, kill_switch=False, dry_run=True)
    env = env_defaults()
    env.dry_run = True
    env.demo_ctrader_account_id = "12345"
    env.demo_user_id = 1
    cfg = merge_config(row, env)
    ok, reason = check_can_execute(db, cfg, 1)
    assert ok is False
    assert reason == "dry_run"


def test_env_defaults_max_calls_340():
    os.environ.pop("GEMINI_GOLD_MAX_CALLS_DAY", None)
    assert env_defaults().max_calls_day == 340
