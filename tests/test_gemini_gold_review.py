"""Gemini Gold AI performance review prompt blocks."""
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.config import env_defaults
from app.gemini_gold_trader.guardrails import merge_config
from app.gemini_gold_trader.models import GeminiGoldConfig, GeminiGoldDecision, GeminiGoldOutcome
from app.gemini_gold_trader.review import (
    _aggressiveness_block,
    _ctrader_account_block,
    _timing_analysis_block,
    build_review_prompt,
)


def _cfg():
    row = GeminiGoldConfig(id=1, enabled=True, dry_run=True, confidence_threshold=85)
    return merge_config(row, env_defaults())


def test_timing_analysis_block_hour_and_hold_stats():
    db = MagicMock()
    now = datetime.utcnow()
    dec1 = GeminiGoldDecision(
        id=1,
        ts=now - timedelta(hours=3),
        session="ny",
        action="TAKE",
        executed=True,
        confidence=88,
    )
    out1 = GeminiGoldOutcome(
        decision_id=1,
        session="ny",
        result="win",
        closed_ts=now - timedelta(hours=2, minutes=30),
    )
    dec2 = GeminiGoldDecision(
        id=2,
        ts=now - timedelta(hours=1),
        session="ny",
        action="TAKE",
        executed=True,
        confidence=82,
    )
    out2 = GeminiGoldOutcome(
        decision_id=2,
        session="ny",
        result="loss",
        closed_ts=now - timedelta(minutes=30),
    )
    db.query.return_value.join.return_value.filter.return_value.order_by.return_value.all.return_value = [
        (out1, dec1),
        (out2, dec2),
    ]
    lines = _timing_analysis_block(db, days=14)
    text = "\n".join(lines)
    assert "TIMING ANALYSIS" in text
    assert "UTC hour" in text
    assert "Hold time" in text
    assert "ny" in text


def test_aggressiveness_block_rates_and_caps():
    db = MagicMock()
    now = datetime.utcnow()
    decisions = [
        GeminiGoldDecision(ts=now, action="SKIP", confidence=65),
        GeminiGoldDecision(ts=now, action="TAKE", executed=True, confidence=90),
        GeminiGoldDecision(ts=now, action="TAKE", executed=False, skip_reason="min_trade_gap", confidence=85),
    ]
    db.query.return_value.filter.return_value.order_by.return_value.all.side_effect = [
        decisions,
        [],
    ]
    lines = _aggressiveness_block(db, cfg=_cfg(), days=14)
    text = "\n".join(lines)
    assert "AGGRESSIVENESS" in text
    assert "min_trade_gap_min=" in text
    assert "take_rate=" in text
    assert "min_trade_gap=1" in text


def test_ctrader_account_block_includes_broker_fields():
    snap = {
        "ctrader_account_id": "12345",
        "execution_mode": "demo",
        "balance": 10000.0,
        "equity": 10050.0,
        "broker_open_position_count": 1,
        "open_position_count": 1,
        "position_reconciliation": "match",
        "recent_broker_closes": [
            {
                "closed_at": "2026-07-01T10:00:00",
                "direction": "LONG",
                "outcome": "WIN",
                "pnl_usd": 12.5,
                "pnl_pct": 0.3,
                "hold_min": 45,
                "broker_position_id": "999",
            }
        ],
        "recent_broker_pnl_usd": 12.5,
    }
    text = "\n".join(_ctrader_account_block(snap))
    assert "equity_usd=10050.0" in text
    assert "position_reconciliation=match" in text
    assert "sum_pnl_usd=12.5" in text


def test_build_review_prompt_includes_timing_aggressiveness_ctrader():
    db = MagicMock()
    cfg = _cfg()
    account_snap = {
        "ctrader_account_id": "1",
        "balance": 5000,
        "equity": 5050,
        "open_position_count": 0,
        "tracked_open_positions": [],
        "recent_broker_closes": [],
    }

    with __import__("unittest.mock").mock.patch(
        "app.gemini_gold_trader.review.funnel_snapshot", return_value={}
    ), __import__("unittest.mock").mock.patch(
        "app.gemini_gold_trader.review.get_setup_stats", return_value=[]
    ), __import__("unittest.mock").mock.patch(
        "app.gemini_gold_trader.review.call_stats_today", return_value=[]
    ), __import__("unittest.mock").mock.patch(
        "app.gemini_gold_trader.review.recent_funnel_events", return_value=[]
    ), __import__("unittest.mock").mock.patch(
        "app.gemini_gold_trader.review.calls_today", return_value=0
    ), __import__("unittest.mock").mock.patch(
        "app.gemini_gold_trader.review.trades_today", return_value=0
    ), __import__("unittest.mock").mock.patch(
        "app.gemini_gold_trader.review.cost_today_usd", return_value=0.0
    ), __import__("unittest.mock").mock.patch(
        "app.gemini_gold_trader.review._recent_closed_trades_block",
        return_value=["=== CLOSED TRADES ==="],
    ), __import__("unittest.mock").mock.patch(
        "app.gemini_gold_trader.review._blocked_takes_block",
        return_value=["=== BLOCKED TAKES ==="],
    ), __import__("unittest.mock").mock.patch(
        "app.gemini_gold_trader.review._timing_analysis_block",
        return_value=["=== TIMING ANALYSIS (14d) ==="],
    ), __import__("unittest.mock").mock.patch(
        "app.gemini_gold_trader.review._aggressiveness_block",
        return_value=["=== AGGRESSIVENESS (14d) ==="],
    ):
        prompt = build_review_prompt(
            db, cfg=cfg, user_id=1, days=14, account_snap=account_snap
        )

    assert "CTRADER BROKER ACCOUNT" in prompt
    assert "TIMING ANALYSIS" in prompt
    assert "AGGRESSIVENESS" in prompt
    assert "timing (hours/sessions/hold times)" in prompt
