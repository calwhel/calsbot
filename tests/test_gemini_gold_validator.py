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


def test_validate_sl_too_wide():
    decision = {
        "direction": "LONG",
        "entry": 2650.0,
        "stop_loss": 2630.0,
        "take_profit": 2670.0,
    }
    ok, reason, _ = validate_take_decision(decision, cfg=_cfg(), spot=2650.0)
    assert ok is False
    assert "sl_too_wide" in reason


def test_validate_rr_too_high_caps_tp_to_max_rr():
    decision = {
        "direction": "LONG",
        "entry": 2650.0,
        "stop_loss": 2640.0,
        "take_profit": 2680.0,
    }
    ok, reason, d = validate_take_decision(decision, cfg=_cfg(), spot=2650.0)
    assert ok is True
    assert reason == "ok"
    assert d["take_profit"] == 2670.0
    assert "tp_adjusted_for_max_rr" in (d.get("validator_note") or "")


def test_validate_caps_slightly_high_rr_from_gemini():
    """Real NY fade-long: Gemini TP implied ~2.002R — should cap to 2.0R, not reject."""
    decision = {
        "direction": "LONG",
        "entry": 4101.20,
        "stop_loss": 4092.22,
        "take_profit": 4119.18,
        "setup_type": "fvg_retrace_bull",
        "confidence": 80,
    }
    ok, reason, d = validate_take_decision(decision, cfg=_cfg(), spot=4101.20)
    assert ok is True
    assert reason == "ok"
    assert d["take_profit"] == 4119.16
    assert "tp_adjusted_for_max_rr" in (d.get("validator_note") or "")


def test_validate_nudges_low_rr_up_to_minimum():
    decision = {
        "direction": "LONG",
        "entry": 2650.0,
        "stop_loss": 2640.0,
        "take_profit": 2655.0,
    }
    ok, reason, d = validate_take_decision(decision, cfg=_cfg(), spot=2650.0)
    assert ok is True
    assert reason == "ok"
    assert d["take_profit"] == 2660.0


def test_validate_nudges_tp_when_gemini_rounds_rr_slightly_low():
    decision = {
        "direction": "LONG",
        "entry": 4157.65,
        "stop_loss": 4150.00,
        "take_profit": 4165.00,
    }
    ok, reason, d = validate_take_decision(decision, cfg=_cfg(), spot=4157.65)
    assert ok is True
    assert reason == "ok"
    assert d["take_profit"] == 4165.30
    assert "tp_adjusted_for_min_rr" in (d.get("validator_note") or "")


def test_validate_one_to_one_rr_ok():
    decision = {
        "direction": "LONG",
        "entry": 2650.0,
        "stop_loss": 2647.0,
        "take_profit": 2653.0,
    }
    ok, reason, _ = validate_take_decision(decision, cfg=_cfg(), spot=2650.0)
    assert ok is True
    assert reason == "ok"


def test_validate_sl_too_tight():
    decision = {
        "direction": "LONG",
        "entry": 2650.0,
        "stop_loss": 2649.5,
        "take_profit": 2651.0,
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
        "stop_loss": 2640.0,
        "take_profit": 2660.0,
    }
    ok, reason, _ = validate_take_decision(decision, cfg=cfg, spot=2665.0)
    assert ok is False
    assert "entry_chasing" in reason


def test_fvg_retrace_allows_marginal_drift_or_nudges_entry():
    """Real NY FVG long blocked at 0.16%>0.15% — zone retrace should pass or nudge."""
    cfg = _cfg()
    cfg.entry_max_drift_pct = 0.15
    decision = {
        "direction": "LONG",
        "setup_type": "fvg_retrace_bull",
        "entry": 4105.35,
        "stop_loss": 4103.50,
        "take_profit": 4108.50,
        "confidence": 80,
    }
    # ~0.16% above entry (4111.92 spot)
    spot = 4111.92
    ok, reason, d = validate_take_decision(decision, cfg=cfg, spot=spot)
    assert ok is True
    assert reason == "ok"
    assert d["entry"] == round(spot, 2) or d["entry"] == 4105.35


def test_zone_retrace_wider_drift_without_nudge():
    cfg = _cfg()
    cfg.entry_max_drift_pct = 0.15
    decision = {
        "direction": "LONG",
        "setup_type": "fvg_retrace_bull",
        "entry": 4105.35,
        "stop_loss": 4103.50,
        "take_profit": 4108.50,
        "confidence": 80,
    }
    # 0.20% drift — within 0.30% zone_retrace cap
    spot = 4105.35 * 1.0020
    ok, reason, d = validate_take_decision(decision, cfg=cfg, spot=spot)
    assert ok is True
    assert reason == "ok"
    assert d["entry"] == 4105.35


def test_high_confidence_skips_liquidity_grab_chasing():
    cfg = _cfg()
    cfg.confidence_threshold = 80
    decision = {
        "direction": "LONG",
        "entry": 4138.52,
        "stop_loss": 4136.52,
        "take_profit": 4141.52,
        "setup_type": "liquidity_grab",
        "liq_grab_mss_level": 4134.67,
        "confidence": 80,
    }
    ok, reason, _ = validate_take_decision(decision, cfg=cfg, spot=4138.52, atr=1.58)
    assert ok is True
    assert reason == "ok"


def test_low_confidence_blocks_liquidity_grab_chasing():
    cfg = _cfg()
    cfg.confidence_threshold = 80
    decision = {
        "direction": "LONG",
        "entry": 4138.52,
        "stop_loss": 4136.52,
        "take_profit": 4141.52,
        "setup_type": "liquidity_grab",
        "liq_grab_mss_level": 4134.67,
        "confidence": 75,
    }
    ok, reason, _ = validate_take_decision(decision, cfg=cfg, spot=4138.52, atr=1.58)
    assert ok is False
    assert "entry_chasing_liquidity_grab" in reason
