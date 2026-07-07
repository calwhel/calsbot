"""Gemini Gold two-step scan: observe then decide."""
from __future__ import annotations

from app.gemini_gold_trader.gemini import (
    _build_observe_prompt,
    _build_prompt,
    format_chart_observation,
)


def test_build_observe_prompt_is_observation_only():
    p = _build_observe_prompt(
        session="london", spot=3342.5, bars_1m=60, bars_5m=80, bars_15m=80, bars_1h=60,
    )
    assert "OBSERVE ONLY" in p
    assert "do NOT decide TAKE or SKIP" in p
    assert "setups_checked" in p
    assert "3342.50" in p


def test_format_chart_observation_sections():
    text = format_chart_observation(
        {
            "entry_chart": "1m bullish engulfing at 3342",
            "chart_5m": "sweep and reclaim at 3340",
            "chart_15m": "discount zone",
            "chart_1h": "mild bullish bias",
            "key_levels": "PDL 3338",
            "market_state": "trending london open",
            "setups_checked": "sweep yes, FVG no",
        }
    )
    assert "Entry / timing" in text
    assert "5-minute (primary)" in text
    assert "PDL 3338" in text
    assert "sweep yes" in text


def test_build_prompt_includes_prior_observation_when_two_step():
    obs = "5-minute (primary):\nLive liquidity sweep at 3340 with 5m MSS."
    p = _build_prompt(
        session="london",
        spot=3342.5,
        bars_1m=60,
        bars_5m=80,
        bars_15m=80,
        bars_1h=60,
        chart_observation=obs,
    )
    assert "TWO-STEP DECISION" in p
    assert "PRIOR CHART OBSERVATION" in p
    assert "Live liquidity sweep at 3340" in p
    assert "focus on the decision" in p


def test_build_prompt_without_observation_unchanged_scalp_rules():
    p = _build_prompt(
        session="london", spot=3342.5, bars_1m=60, bars_5m=80, bars_15m=80, bars_1h=60,
    )
    assert "TWO-STEP DECISION" not in p
    assert "SCALP MANDATE" in p
