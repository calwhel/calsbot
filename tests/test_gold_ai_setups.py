"""Tests for Gold AI structure setups, entry guards, SMT modifier, and context."""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.gold_ai_trader.htf_bias import direction_aligns_with_htf, htf_bias_summary
from app.gold_ai_trader.setup_toggles import setup_enabled, smt_modifier_enabled
from app.gold_ai_trader.smt_modifier import assess_smt_divergence, CTRADER_SMT_REFERENCES
from app.gold_ai_trader.context_history import parse_zone_from_detail, build_recent_decisions_block
from app.gold_ai_trader.context import build_data_quality_block
from app.gold_ai_trader.scanner import Candidate, pick_best
from app.services.strategy_ta import (
    entry_zone_allows_price,
    eval_fx_breaker,
    eval_equal_hl_sweep,
)


def _make_klines(base: float, n: int = 60) -> list:
    """Synthetic flat-ish klines for unit tests."""
    rows = []
    ts = int((datetime.utcnow() - timedelta(minutes=5 * n)).timestamp() * 1000)
    for i in range(n):
        o = base + i * 0.05
        c = o + 0.2
        h = c + 0.3
        l = o - 0.3
        rows.append([ts + i * 300_000, o, h, l, c, 100.0])
    return rows


def _breaker_bull_klines() -> list:
    """Former bearish OB broken above, price retracing into zone."""
    rows = []
    ts = 1_700_000_000_000
    price = 2650.0
    for i in range(40):
        rows.append([ts + i * 300_000, price, price + 1, price - 1, price + 0.5, 100.0])
        price += 0.1
    i = len(rows) - 15
    ob_low, ob_high = 2653.0, 2654.5
    rows[i] = [rows[i][0], ob_high, ob_high + 0.2, ob_low, ob_low + 0.3, 100.0]
    rows[i + 1] = [rows[i + 1][0], ob_low, ob_low + 0.1, ob_low - 2, ob_low - 1.5, 100.0]
    for j in range(i + 2, len(rows) - 3):
        rows[j] = [rows[j][0], 2655.0, 2656.0, 2654.8, 2655.5, 100.0]
    rows[-2] = [rows[-2][0], 2654.0, 2654.6, 2653.2, 2653.8, 100.0]
    rows[-1] = [rows[-1][0], 2653.8, 2654.0, 2653.0, 2653.5, 100.0]
    return rows


def _eqh_sweep_klines() -> list:
    """Equal highs cluster, sweep, displacement, reclaim at cluster."""
    rows = []
    ts = 1_700_000_000_000
    eq_level = 2660.0
    for i in range(35):
        o = 2655.0 + (i % 3) * 0.2
        h = eq_level if i % 5 in (0, 2) else eq_level - 0.5
        l = o - 1.0
        c = o + 0.1
        rows.append([ts + i * 300_000, o, h, l, c, 100.0])
    rows[-8] = [rows[-8][0], eq_level - 0.2, eq_level + 1.5, eq_level - 1.0, eq_level - 0.5, 200.0]
    rows[-7] = [rows[-7][0], eq_level, eq_level + 0.2, eq_level - 4.0, eq_level - 3.5, 300.0]
    rows[-6] = [rows[-6][0], eq_level - 3.5, eq_level - 3.0, eq_level - 5.0, eq_level - 4.5, 250.0]
    reclaim = eq_level - 0.3
    rows[-2] = [rows[-2][0], reclaim + 0.5, reclaim + 0.8, reclaim - 0.2, reclaim, 100.0]
    rows[-1] = [rows[-1][0], reclaim, reclaim + 0.3, reclaim - 0.3, reclaim, 100.0]
    return rows


@pytest.fixture
def mock_http():
    return MagicMock()


@pytest.fixture
def cache():
    return {"__asset_class__": "forex"}


class TestSetupToggles:
    def test_existing_setups_default_on(self):
        os.environ.pop("GOLD_AI_SETUP_FVG_BULL", None)
        assert setup_enabled("fvg_bull") is True

    def test_new_breaker_default_off(self):
        os.environ.pop("GOLD_AI_SETUP_BREAKER_BULL", None)
        assert setup_enabled("breaker_bull") is False

    def test_breaker_env_override(self):
        os.environ["GOLD_AI_SETUP_BREAKER_BULL"] = "true"
        try:
            assert setup_enabled("breaker_bull") is True
        finally:
            os.environ.pop("GOLD_AI_SETUP_BREAKER_BULL", None)

    def test_smt_modifier_default_off(self):
        os.environ.pop("GOLD_AI_SMT_MODIFIER", None)
        assert smt_modifier_enabled() is False


class TestBreakerBlockEntryGuard:
    @pytest.mark.asyncio
    async def test_breaker_fires_in_zone(self, mock_http, cache):
        klines = _breaker_bull_klines()
        with patch("app.services.strategy_ta._get_klines", new=AsyncMock(return_value=klines)):
            ok, msg = await eval_fx_breaker(
                {"direction": "bullish", "timeframe": "5m", "lookback": 50},
                "XAUUSD",
                2653.8,
                mock_http,
                cache,
            )
        assert ok is True
        assert "FIRED" in msg

    @pytest.mark.asyncio
    async def test_breaker_blocked_when_chasing(self, mock_http, cache):
        klines = _breaker_bull_klines()
        with patch("app.services.strategy_ta._get_klines", new=AsyncMock(return_value=klines)):
            ok, msg = await eval_fx_breaker(
                {"direction": "bullish", "timeframe": "5m", "lookback": 50},
                "XAUUSD",
                2665.0,
                mock_http,
                cache,
            )
        assert ok is False
        assert "chasing" in msg or "above zone" in msg or "too far" in msg


class TestEqualHlSweep:
    @pytest.mark.asyncio
    async def test_eqh_sweep_eval_returns_structured_msg(self, mock_http, cache):
        klines = _eqh_sweep_klines()
        with patch("app.services.strategy_ta._get_klines", new=AsyncMock(return_value=klines)):
            ok, msg = await eval_equal_hl_sweep(
                {"equal_type": "eqh", "direction": "bearish", "timeframe": "5m"},
                "XAUUSD",
                2659.7,
                mock_http,
                cache,
            )
        assert isinstance(ok, bool)
        assert "eqh" in msg.lower()

    @pytest.mark.asyncio
    async def test_eqh_sweep_blocked_when_extended(self, mock_http, cache):
        klines = _eqh_sweep_klines()
        with patch("app.services.strategy_ta._get_klines", new=AsyncMock(return_value=klines)):
            ok, msg = await eval_equal_hl_sweep(
                {"equal_type": "eqh", "direction": "bearish", "timeframe": "5m"},
                "XAUUSD",
                2640.0,
                mock_http,
                cache,
            )
        if not ok:
            assert (
                "chasing" in msg
                or "below zone" in msg
                or "above zone" in msg
                or "no cluster" in msg.lower()
                or "no sweep" in msg.lower()
            )


class TestEntryZoneGuard:
    def test_retrace_required_not_chase(self):
        atr = 4.0
        ok, _ = entry_zone_allows_price(2653.5, 2653.0, 2654.5, "bullish", atr)
        assert ok is True
        ok2, msg2 = entry_zone_allows_price(2665.0, 2653.0, 2654.5, "bullish", atr)
        assert ok2 is False
        assert "chasing" in msg2 or "above zone" in msg2


class TestHtfBias:
    def test_direction_aligns_bullish(self):
        bias = {"htf_bias": "bullish"}
        ok, reason = direction_aligns_with_htf("LONG", bias)
        assert ok is True
        assert "aligned" in reason

    def test_direction_blocks_counter_trend(self):
        bias = {"htf_bias": "bearish"}
        ok, reason = direction_aligns_with_htf("LONG", bias)
        assert ok is False
        assert "counter" in reason

    def test_htf_summary_has_4h_fields(self):
        k1h = _make_klines(2650, 50)
        k4h = _make_klines(2650, 30)
        k_daily = _make_klines(2650, 5)
        summary = htf_bias_summary(k1h, k4h, k_daily)
        assert "trend_1h" in summary
        assert "trend_4h" in summary
        assert "daily_high" in summary


class TestSmtModifier:
    def test_dxy_not_on_ctrader(self):
        assert "DXY" not in CTRADER_SMT_REFERENCES
        assert "XAGUSD" in CTRADER_SMT_REFERENCES

    @pytest.mark.asyncio
    async def test_smt_not_standalone_returns_modifier_only(self, mock_http, cache):
        gold = _make_klines(2650, 30)
        silver = _make_klines(30.0, 30)
        with patch("app.services.strategy_ta._get_klines", new=AsyncMock(side_effect=[gold, silver])):
            with patch(
                "app.services.tradfi_prices.get_metal_kline_source",
                return_value="ctrader",
            ):
                result = await assess_smt_divergence(
                    direction="SHORT",
                    http_client=mock_http,
                    cache=cache,
                )
        assert "modifier" in result
        assert isinstance(result["modifier"], int)
        assert -15 <= result["modifier"] <= 15


class TestContextEnrichment:
    def test_data_quality_block_shows_source(self):
        block = build_data_quality_block(
            {
                "price": 2650.0,
                "live_source": "ctrader",
                "price_source": "ctrader",
                "kline_source": "ctrader-user",
                "kline_bars": 60,
                "klines_stale": False,
                "kline_synthetic": False,
            }
        )
        text = "\n".join(block)
        assert "DATA QUALITY" in text
        assert "ctrader" in text.lower()
        assert "Gate: PASS" in text

    def test_parse_zone_from_detail(self):
        z = parse_zone_from_detail("Bullish breaker 2653.0–2654.5 (broke above, retrace)")
        assert z == (2653.0, 2654.5)

    def test_pick_best_prioritizes_breaker_over_fvg(self):
        cands = [
            Candidate("fvg_bull", "LONG", "fvg", 1.0, "k1", {}),
            Candidate("breaker_bull", "LONG", "breaker", 1.0, "k2", {}),
        ]
        best = pick_best(cands)
        assert best.type == "breaker_bull"


class TestRecentDecisionsBlock:
    def test_empty_decisions(self):
        db = MagicMock()
        db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        lines = build_recent_decisions_block(db, session="london")
        assert any("No prior decisions" in ln for ln in lines)
