"""Gemini Gold chart zones and setup vocabulary tests."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.chart_renderer import (
    HAS_MATPLOTLIB,
    chart_png_is_valid,
    render_candlestick_chart,
)
from app.gemini_gold_trader.chart_zones import ChartZone, detect_chart_zones, format_zones_for_prompt
from app.gemini_gold_trader.gemini import _normalize_decision
from app.gemini_gold_trader.setup_types import normalize_setup_type


def _bars_with_fvg(n: int = 50, base: float = 2650.0) -> list:
    bars = []
    for i in range(n):
        o = base + i * 0.1
        h = o + 1.0
        l = o - 1.0
        c = o + 0.3
        bars.append([1_700_000_000_000 + i * 300_000, o, h, l, c, 100.0])
    # Inject bullish FVG at bar 20: candle 19 high < candle 21 low
    i = 20
    bars[i - 1][2] = bars[i - 1][1] + 0.5
    bars[i - 1][3] = bars[i - 1][1] - 0.5
    bars[i + 1][2] = bars[i - 1][2] + 3.0
    bars[i + 1][3] = bars[i - 1][2] + 2.5
    bars[i + 1][4] = bars[i + 1][3] + 0.5
    return bars


def test_normalize_legacy_setup_types():
    assert normalize_setup_type("fvg_retrace", "LONG") == "fvg_retrace_bull"
    assert normalize_setup_type("order_block", "SHORT") == "ob_bear"
    assert normalize_setup_type("ifvg_bull", "LONG") == "ifvg_bull"
    assert normalize_setup_type("liquidity_grab", "LONG") == "liquidity_grab_long"


def test_normalize_decision_applies_setup_aliases():
    d = _normalize_decision(
        {
            "action": "TAKE",
            "setup_type": "fvg_retrace",
            "direction": "LONG",
            "entry": 2650.0,
            "stop_loss": 2645.0,
            "take_profit": 2660.0,
            "confidence": 80,
            "rationale": "test",
        }
    )
    assert d["setup_type"] == "fvg_retrace_bull"


def test_detect_chart_zones_finds_gap():
    bars = _bars_with_fvg()
    spot = float(bars[-1][4])
    zones = detect_chart_zones(bars, spot, timeframe="5m")
    assert zones
    assert any(z.kind in ("fvg", "ifvg") for z in zones)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_render_chart_with_zones_still_valid_png():
    bars = _bars_with_fvg()
    spot = float(bars[-1][4])
    zones = detect_chart_zones(bars, spot, timeframe="5m")
    png = render_candlestick_chart(bars, timeframe="5m", session="london", zones=zones)
    assert png is not None
    assert chart_png_is_valid(png)


def test_format_zones_for_prompt():
    z = ChartZone(
        kind="fvg",
        side="bull",
        top=2652.0,
        bottom=2650.5,
        bar_idx=20,
        label="FVG bull 2650.50-2652.00",
    )
    text = format_zones_for_prompt({"5m": [z]}, 2651.0)
    assert "DETECTED ZONES" in text
    assert "FVG bull" in text
