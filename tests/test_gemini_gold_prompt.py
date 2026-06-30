"""Gemini Gold Trader vision prompt content."""
from __future__ import annotations

from app.gemini_gold_trader.gemini import _build_prompt


def test_build_prompt_includes_persona_and_price_anchoring():
    p = _build_prompt(session="london", spot=3342.5, bars_15m=80, bars_1h=60)
    assert "experienced discretionary XAUUSD day trader" in p
    assert "PRICE ANCHORING" in p
    assert "risk-first" in p
    assert "2000-3500" not in p
    assert "3342.50" in p


def test_build_prompt_includes_setup_vocabulary():
    p = _build_prompt(session="new_york", spot=2650.0, bars_15m=80, bars_1h=60)
    assert "SETUP VOCABULARY" in p
    assert "ORDER BLOCKS" in p
    assert "FAIR VALUE GAPS" in p
    assert "LIQUIDITY SWEEPS" in p
    assert "OPENING RANGE BREAKOUT" in p
    assert "TRENDLINE BOUNCE/BREAK" in p
    assert "LIQUIDITY GRAB + MARKET STRUCTURE SHIFT" in p
    assert "Do NOT TAKE on generic" in p
    assert "at least ONE of these specific patterns" in p
