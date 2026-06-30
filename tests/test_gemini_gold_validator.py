"""Gemini Gold validator unit tests."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.config import env_defaults
from app.gemini_gold_trader.validator import validate_take_decision


def _cfg():
    return env_defaults()


def test_validate_long_ok():
    decision = {
        "direction": "LONG",
        "entry": 2650.0,
        "stop_loss": 2640.0,
        "take_profit": 2670.0,
    }
    ok, reason, d = validate_take_decision(decision, cfg=_cfg(), spot=2650.0)
    assert ok is True
    assert reason == "ok"
    assert d["entry"] == 2650.0


def test_validate_sl_too_tight():
    decision = {
        "direction": "LONG",
        "entry": 2650.0,
        "stop_loss": 2649.0,
        "take_profit": 2660.0,
    }
    ok, reason, _ = validate_take_decision(decision, cfg=_cfg(), spot=2650.0)
    assert ok is False
    assert "sl_too_tight" in reason


def test_validate_short_price_order():
    decision = {
        "direction": "SHORT",
        "entry": 2650.0,
        "stop_loss": 2640.0,
        "take_profit": 2660.0,
    }
    ok, reason, _ = validate_take_decision(decision, cfg=_cfg(), spot=2650.0)
    assert ok is False
    assert reason == "validator:short_price_order"


def test_validate_entry_chasing():
    cfg = _cfg()
    cfg.entry_max_drift_pct = 0.05
    decision = {
        "direction": "LONG",
        "entry": 2650.0,
        "stop_loss": 2630.0,
        "take_profit": 2690.0,
    }
    ok, reason, _ = validate_take_decision(decision, cfg=cfg, spot=2665.0)
    assert ok is False
    assert "entry_chasing" in reason
