"""Gemini Gold Trader vision prompt content."""
from __future__ import annotations

from app.gemini_gold_trader.gemini import _build_prompt


def test_build_prompt_scalp_focus_not_swing():
    p = _build_prompt(session="london", spot=3342.5, bars_15m=80, bars_1h=60)
    assert "SCALPER" in p
    assert "SCALP MANDATE" in p
    assert "PRIMARY trigger timeframe: 15-minute" in p
    assert "SWING SETUPS TO SKIP" in p
    assert "NOT a swing trader" in p
    assert "1-hour chart: context/bias ONLY" in p
    assert "Multi-touch 1h trendline" in p


def test_build_prompt_includes_scalp_risk_rules():
    p = _build_prompt(session="london", spot=3342.5, bars_15m=80, bars_1h=60)
    assert "SCALP RISK RULES" in p
    assert "30–150 platform pips" in p
    assert "1:1 to 2:1" in p


def test_build_prompt_includes_scalp_setup_vocabulary():
    p = _build_prompt(session="new_york", spot=2650.0, bars_15m=80, bars_1h=60)
    assert "SESSION LIQUIDITY SWEEP" in p
    assert "OPENING RANGE BREAKOUT" in p
    assert "15m ORDER BLOCK" in p
    assert "15m MOMENTUM SCALP" in p


def test_build_prompt_includes_price_anchoring():
    p = _build_prompt(session="london", spot=3342.5, bars_15m=80, bars_1h=60)
    assert "PRICE ANCHORING" in p
    assert "3342.50" in p
