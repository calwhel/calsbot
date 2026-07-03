"""Gemini Gold chart rendering and PNG validation."""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.chart_renderer import (
    chart_png_is_valid,
    render_candlestick_chart,
    summarize_bars_for_prompt,
)
from app.gemini_gold_trader.gemini import _build_ohlc_context, _chart_contents


def _gold_bars(n: int, base: float = 3380.0) -> list:
    return [
        [
            1_700_000_000_000 + i * 300_000,
            base + i * 0.2,
            base + i * 0.2 + 1.5,
            base + i * 0.2 - 1.0,
            base + i * 0.2 + 0.5,
            100.0,
        ]
        for i in range(n)
    ]


def test_render_produces_valid_png():
    bars = _gold_bars(40)
    png = render_candlestick_chart(bars, timeframe="5m", session="london")
    assert png is not None
    assert chart_png_is_valid(png)


def test_render_rejects_zero_ohlc():
    bad = [[1_700_000_000_000 + i * 60_000, 0, 0, 0, 0, 0] for i in range(30)]
    assert render_candlestick_chart(bad, timeframe="5m") is None


def test_summarize_bars_includes_prices():
    text = summarize_bars_for_prompt(_gold_bars(25), label="5m")
    assert "3384" in text or "3385" in text
    assert "range:" in text


def test_ohlc_context_has_all_timeframes():
    ctx = _build_ohlc_context(
        entry_bars=_gold_bars(20),
        bars_5m=_gold_bars(30),
        bars_15m=_gold_bars(30),
        bars_1h=_gold_bars(30),
        entry_timeframe="1m",
    )
    assert "Entry 1m" in ctx
    assert "5m" in ctx
    assert "15m" in ctx
    assert "1h" in ctx


def test_chart_contents_labels_each_image():
    bars = _gold_bars(20)
    png = render_candlestick_chart(bars, timeframe="5m", session="london")
    assert png is not None

    class _Types:
        class Part:
            @staticmethod
            def from_bytes(*, data, mime_type):
                return ("bytes", len(data), mime_type)

    parts = _chart_contents(
        _Types,
        png_entry=png,
        png_5m=png,
        png_15m=png,
        png_1h=png,
        entry_timeframe="1m",
        entry_5m_fallback=False,
        bars_1m=20,
        bars_5m=30,
        bars_15m=30,
        bars_1h=30,
        prompt="decide",
    )
    assert parts[0].startswith("CHART 1")
    assert parts[2].startswith("CHART 2")
    assert parts[-1] == "decide"
