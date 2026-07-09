"""Tests for broker preflight, setup gating, and observation skip."""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.config import env_defaults
from app.gemini_gold_trader.gemini import _normalize_decision, observation_blocks_decide
from app.gemini_gold_trader.setup_types import is_approved_setup_type


def test_is_approved_setup_type_rejects_unknown():
    assert is_approved_setup_type("liquidity_grab_short") is True
    assert is_approved_setup_type("unknown") is False
    assert is_approved_setup_type(None) is False


def test_normalize_decision_downgrades_unknown_take():
    raw = {
        "action": "TAKE",
        "setup_type": "unknown",
        "direction": "SHORT",
        "entry": 2650.0,
        "stop_loss": 2660.0,
        "take_profit": 2630.0,
        "confidence": 85,
    }
    d = _normalize_decision(raw)
    assert d["action"] == "SKIP"
    assert d["setup_type"] is None


def test_observation_blocks_decide_on_no_live_trigger():
    blocked, reason = observation_blocks_decide(
        {"setups_checked": "liquidity grab: no live trigger; ORB: no"}
    )
    assert blocked is True
    assert reason == "observation_no_live_setup"


def test_observation_allows_decide_when_triggers_present():
    blocked, _ = observation_blocks_decide(
        {"setups_checked": "liquidity_grab_short: live trigger on 5m bear MSS"}
    )
    assert blocked is False


def test_execution_diagnostics_pass_messages():
    from app.gemini_gold_trader.execution_diagnostics import build_execution_readiness

    db = MagicMock()
    cfg = env_defaults()
    cfg = cfg.__class__(**{**cfg.__dict__, "demo_user_id": 1, "demo_ctrader_account_id": "47782488"})
    with patch(
        "app.gemini_gold_trader.execution_diagnostics.check_can_execute",
        return_value=(True, "ok"),
    ):
        er = build_execution_readiness(
            db,
            cfg=cfg,
            user_id=1,
            account_snap={
                "balance": 10000.0,
                "equity": 10000.0,
                "broker_unreachable": False,
            },
        )
    checks = {c["name"]: c for c in er["checks"]}
    assert checks["trading_account"]["ok"] is True
    assert "47782488" in checks["trading_account"]["detail"]
    assert checks["broker_reachable"]["ok"] is True
    assert "Broker connected" in checks["broker_reachable"]["detail"]


def test_broker_preflight_timeout():
    from app.gemini_gold_trader.broker_preflight import broker_reachable_for_execution

    cfg = env_defaults()
    cfg = cfg.__class__(
        **{
            **cfg.__dict__,
            "demo_user_id": 5,
            "demo_ctrader_account_id": "47782488",
        }
    )
    db = MagicMock()
    prefs = MagicMock()
    prefs.ctrader_access_token = "tok"
    db.query.return_value.filter.return_value.first.return_value = prefs

    async def _slow(*a, **k):
        await asyncio.sleep(5)
        return {"position_ids": set()}

    async def _run():
        with patch(
            "app.services.ctrader_client.get_broker_reconcile_snapshot_resilient",
            new=AsyncMock(side_effect=_slow),
        ), patch(
            "app.gemini_gold_trader.broker_preflight._EXEC_BROKER_TIMEOUT_S",
            0.05,
        ):
            ok, reason, _ = await broker_reachable_for_execution(db, cfg, user_id=5)
        assert ok is False
        assert reason == "ctrader_poll_timeout"

    asyncio.run(_run())
