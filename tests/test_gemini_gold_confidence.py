"""Gemini Gold confidence scoring and calibration."""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.confidence_scoring import (
    calibrate_confidence,
    compute_chart_confluence,
    compute_rr,
    confidence_band,
    confidence_scoring_prompt_block,
)
from app.gemini_gold_trader.config import env_defaults
from app.gemini_gold_trader.gemini import _normalize_decision


def test_env_default_confidence_is_85():
    cfg = env_defaults()
    assert cfg.confidence_threshold == 85


def test_confidence_scoring_prompt_mentions_full_range():
    block = confidence_scoring_prompt_block(confidence_threshold=85)
    assert "92–100" in block
    assert "86–91" in block
    assert "86–95" in block


def test_calibrate_lifts_under_scored_strong_setup():
    checklist = {
        "htf_aligned": True,
        "ltf_trigger": True,
        "at_structure": True,
        "momentum_ok": True,
        "session_active": True,
        "range_edge": True,
        "volatility_ok": True,
    }
    decision = {
        "action": "TAKE",
        "setup_type": "fvg_retrace_bull",
        "direction": "LONG",
        "entry": 2650.0,
        "stop_loss": 2644.0,
        "take_profit": 2662.0,
        "confidence": 74,
    }
    calibrated, meta = calibrate_confidence(
        decision,
        checklist=checklist,
        confidence_threshold=85,
    )
    assert calibrated >= 86
    assert meta["model_confidence"] == 74
    assert meta["calibrated"] is True


def test_calibrate_caps_weak_confluence_overconfidence():
    checklist = {k: False for k in (
        "htf_aligned", "ltf_trigger", "at_structure",
        "momentum_ok", "session_active", "range_edge", "volatility_ok",
    )}
    decision = {
        "action": "TAKE",
        "setup_type": "ob_bull",
        "direction": "LONG",
        "entry": 2650.0,
        "stop_loss": 2644.0,
        "take_profit": 2658.0,
        "confidence": 88,
    }
    calibrated, _ = calibrate_confidence(
        decision,
        checklist=checklist,
        confidence_threshold=85,
    )
    assert calibrated <= 58


def test_compute_rr_long():
    rr = compute_rr(
        {
            "direction": "LONG",
            "entry": 100.0,
            "stop_loss": 98.0,
            "take_profit": 104.0,
        }
    )
    assert rr == 2.0


def test_normalize_decision_applies_calibration():
    raw = {
        "action": "TAKE",
        "setup_type": "fvg_retrace_bull",
        "direction": "LONG",
        "entry": 2650.0,
        "stop_loss": 2644.0,
        "take_profit": 2662.0,
        "confidence": 76,
        "rationale": "test",
    }
    checklist = {
        "htf_aligned": True,
        "ltf_trigger": True,
        "at_structure": True,
        "momentum_ok": True,
        "session_active": True,
        "range_edge": False,
        "volatility_ok": True,
    }
    out = _normalize_decision(
        raw,
        confluence_checklist=checklist,
        confidence_threshold=85,
    )
    assert out["confidence"] >= 80
    assert out["confidence_meta"]["confluence_passed"] >= 5


def test_confidence_band_exceptional():
    checklist = {
        "htf_aligned": True,
        "ltf_trigger": True,
        "at_structure": True,
        "momentum_ok": True,
        "session_active": True,
        "range_edge": True,
        "volatility_ok": True,
    }
    lo, hi = confidence_band(6, 7, checklist, "liq_sweep_bull")
    assert lo >= 86
    assert hi >= 90


def test_compute_chart_confluence_from_bars():
    bars = []
    price = 2650.0
    for i in range(30):
        o = price + i * 0.3
        c = o + 0.5
        bars.append([i, o, c + 0.2, o - 0.2, c])
    checklist = compute_chart_confluence(
        bars_5m=bars,
        bars_15m=bars,
        bars_1h=bars,
        spot=bars[-1][4],
        session="london",
    )
    assert isinstance(checklist, dict)
    assert len(checklist) >= 5
