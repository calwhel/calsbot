"""Gemini Gold Trader vision prompt content."""
from __future__ import annotations

from app.gemini_gold_trader.gemini import _build_prompt


def test_build_prompt_scalp_focus_not_swing():
    p = _build_prompt(
        session="london", spot=3342.5, bars_1m=60, bars_5m=80, bars_15m=80, bars_1h=60,
    )
    assert "SCALPER" in p
    assert "SCALP MANDATE" in p
    assert "PRIMARY trigger timeframe: 5-minute" in p
    assert "SWING SETUPS TO SKIP" in p
    assert "NOT a swing trader" in p
    assert "1-hour chart: session bias ONLY" in p
    assert "Multi-touch 1h trendline" in p


def test_build_prompt_includes_four_chart_timeframes():
    p = _build_prompt(
        session="london", spot=3342.5, bars_1m=60, bars_5m=80, bars_15m=80, bars_1h=60,
    )
    assert "Image 1: 1-minute (60 candles)" in p
    assert "Image 2: 5-minute (80 candles)" in p
    assert "Image 3: 15-minute (80 candles)" in p
    assert "Image 4: 1-hour (60 candles)" in p
    assert "1-minute chart: refine entry timing" in p


def test_build_prompt_entry_5m_fallback_note():
    p = _build_prompt(
        session="london",
        spot=3342.5,
        bars_1m=40,
        bars_5m=80,
        bars_15m=80,
        bars_1h=60,
        entry_timeframe="5m",
        entry_5m_fallback=True,
    )
    assert "5-minute (1m unavailable)" in p
    assert "broker 1m unavailable" in p


def test_build_prompt_includes_scalp_risk_rules():
    p = _build_prompt(
        session="london", spot=3342.5, bars_1m=60, bars_5m=80, bars_15m=80, bars_1h=60,
    )
    assert "SCALP RISK RULES" in p
    assert "10–150 platform pips" in p
    assert "1:1 to 2:1" in p


def test_build_prompt_includes_scalp_setup_vocabulary():
    p = _build_prompt(
        session="new_york", spot=2650.0, bars_1m=60, bars_5m=80, bars_15m=80, bars_1h=60,
    )
    assert "SESSION LIQUIDITY SWEEP" in p
    assert "OPENING RANGE BREAKOUT" in p
    assert "5m ORDER BLOCK" in p
    assert "5m MOMENTUM SCALP" in p


def test_build_prompt_includes_price_anchoring():
    p = _build_prompt(
        session="london", spot=3342.5, bars_1m=60, bars_5m=80, bars_15m=80, bars_1h=60,
    )
    assert "PRICE ANCHORING" in p
    assert "3342.50" in p
