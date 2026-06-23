"""Tests for setup readiness gate and session playbook (Phase A+B)."""
from __future__ import annotations

import os
from datetime import datetime, timedelta

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gold_ai_trader.session_playbook import session_allows_setup
from app.gold_ai_trader.setup_readiness import (
    compute_setup_readiness,
    parse_reclaim_level,
    reclaim_held,
    readiness_min_score,
    momentum_6bar_aligned,
    spot_in_entry_zone,
)
from app.gold_ai_trader.setup_toggles import setup_enabled


class _Cfg:
    london_start_hour = 7
    ny_start_hour = 12


def _bar(ts, o, h, l, c, v=100.0):
    return [int(ts.timestamp() * 1000), o, h, l, c, v]


def _make_k5(base: float = 2650.0, n: int = 25, bullish_tail: bool = True) -> list:
    now = datetime(2026, 6, 23, 8, 0, 0)
    rows = []
    for i in range(n):
        ts = now - timedelta(minutes=(n - i) * 5)
        o = base + i * 0.02
        c = o + (0.5 if bullish_tail else -0.5)
        h = max(o, c) + 0.3
        l = min(o, c) - 0.3
        rows.append(_bar(ts, o, h, l, c, 120))
    return rows


class TestSessionPlaybook:
    def test_london_allows_sweep(self):
        ok, reason = session_allows_setup("liq_sweep_bull", "london", htf_align_reason="htf_aligned_bull")
        assert ok is True
        assert reason == "london_primary"

    def test_london_blocks_disp_without_htf(self):
        ok, reason = session_allows_setup("disp_bull", "london", htf_align_reason="htf_mixed_allowed")
        assert ok is False
        assert "htf" in reason

    def test_ny_allows_ob(self):
        ok, _ = session_allows_setup("ob_bull", "new_york", htf_align_reason="htf_aligned_bull")
        assert ok is True

    def test_ny_blocks_asian_sweep(self):
        ok, reason = session_allows_setup("asian_sweep_bull", "new_york")
        assert ok is False


class TestReadinessHelpers:
    def test_parse_reclaim_level(self):
        assert parse_reclaim_level("LQ sweep bullish: reclaim @ 4105.0 FIRED") == 4105.0

    def test_reclaim_held_long(self):
        assert reclaim_held(4106.0, 4105.0, "LONG", 4.0) is True
        assert reclaim_held(4104.0, 4105.0, "LONG", 4.0) is False

    def test_spot_in_zone(self):
        ok, tag = spot_in_entry_zone(2651.0, (2650.0, 2652.0), 4.0)
        assert ok is True
        assert tag == "in_zone"

    def test_momentum_bullish_tail(self):
        k5 = _make_k5(bullish_tail=True)
        ok, delta = momentum_6bar_aligned(k5, "LONG")
        assert ok is True
        assert delta >= 0


class TestReadinessScoring:
    def test_counter_htf_hard_fail(self):
        bias = {"htf_bias": "bearish"}
        k5 = _make_k5()
        result = compute_setup_readiness(
            setup_type="liq_sweep_bull",
            direction="LONG",
            detail="LQ sweep bullish: reclaim @ 2650.0 FIRED",
            price=2651.0,
            atr=4.0,
            k5=k5,
            k1h=k5,
            bias=bias,
            htf_align_reason="counter_htf_bear",
            key_levels=[2660.0, 2640.0],
            in_zone_hint=True,
        )
        assert result.passed is False
        assert result.score == 0
        assert "counter" in (result.hard_fail or "").lower()

    def test_weak_displacement_fails_sweep(self):
        bias = {"htf_bias": "bullish"}
        k5 = _make_k5()
        for i in range(-7, -1):
            k5[i] = _bar(datetime(2026, 6, 23, 7, 0), 2650, 2650.1, 2649.9, 2650.05, 100)
        result = compute_setup_readiness(
            setup_type="liq_sweep_bull",
            direction="LONG",
            detail="LQ sweep bullish: reclaim @ 2650.0 FIRED",
            price=2650.5,
            atr=4.0,
            k5=k5,
            k1h=k5,
            bias=bias,
            htf_align_reason="htf_aligned_bull",
            key_levels=[2670.0, 2640.0],
            in_zone_hint=True,
        )
        assert result.passed is False
        assert "displacement" in (result.hard_fail or "")

    def test_zone_setup_requires_in_zone(self):
        bias = {"htf_bias": "bullish"}
        k5 = _make_k5()
        result = compute_setup_readiness(
            setup_type="ob_bull",
            direction="LONG",
            detail="Bullish OB 2650.0–2652.0",
            price=2660.0,
            atr=4.0,
            k5=k5,
            k1h=k5,
            bias=bias,
            htf_align_reason="htf_aligned_bull",
            key_levels=[2670.0, 2640.0],
            in_zone_hint=False,
        )
        assert result.passed is False
        assert "not_at_entry" in (result.hard_fail or "")

    def test_readiness_min_default(self):
        os.environ.pop("GOLD_AI_READINESS_MIN", None)
        assert readiness_min_score() == 55


class TestSetupDefaults:
    def test_fvg_just_formed_default_off(self):
        os.environ.pop("GOLD_AI_SETUP_FVG_BULL", None)
        assert setup_enabled("fvg_bull") is False
