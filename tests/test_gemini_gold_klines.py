"""Tests for gemini-gold kline fetch path."""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


def _bars(n: int, *, ts_start: int = 1_700_000_000_000) -> list:
    return [[ts_start + i * 60_000, 1.0, 2.0, 0.5, 1.5, 10.0] for i in range(n)]


def test_get_chart_klines_uses_tradfi_chain():
    from app.gemini_gold_trader.klines import get_chart_klines

    fresh = _bars(30)

    async def _run():
        with patch(
            "app.services.tradfi_prices.get_klines",
            new_callable=AsyncMock,
            return_value=fresh,
        ), patch(
            "app.services.tradfi_prices.get_metal_kline_source",
            return_value="ctrader",
        ), patch(
            "app.gemini_gold_trader.klines.newest_bar_age_s",
            return_value=60.0,
        ), patch(
            "app.gemini_gold_trader.klines.stale_limit_s",
            return_value=1800.0,
        ):
            bars, meta = await get_chart_klines("1h", 80, user_id=42)
        assert len(bars) == 30
        assert meta["status"] == "ok"
        assert meta["source"] == "ctrader"

    asyncio.run(_run())


def test_get_chart_klines_falls_back_to_postgres_when_tradfi_is_coinbase():
    from app.gemini_gold_trader.klines import get_chart_klines

    coinbase = _bars(30)
    snap = _bars(30)

    async def _run():
        with patch(
            "app.services.tradfi_prices.get_klines",
            new_callable=AsyncMock,
            return_value=coinbase,
        ), patch(
            "app.services.tradfi_prices.get_metal_kline_source",
            return_value="coinbase",
        ), patch(
            "app.gemini_gold_trader.klines._fetch_ctrader_chart_klines",
            new_callable=AsyncMock,
            return_value=([], ""),
        ), patch(
            "app.services.kline_snapshot_store.get_klines",
            return_value=snap,
        ), patch(
            "app.gemini_gold_trader.klines.newest_bar_age_s",
            return_value=120.0,
        ), patch(
            "app.gemini_gold_trader.klines.stale_limit_s",
            return_value=1800.0,
        ):
            bars, meta = await get_chart_klines("15m", 80)
        assert len(bars) == 30
        assert meta["source"] == "ctrader"
        assert meta["status"] == "ok"

    asyncio.run(_run())


def test_get_chart_klines_falls_back_to_postgres_snapshot():
    from app.gemini_gold_trader.klines import get_chart_klines

    fresh = _bars(25)

    async def _run():
        with patch(
            "app.services.tradfi_prices.get_klines",
            new_callable=AsyncMock,
            return_value=[],
        ), patch(
            "app.services.kline_snapshot_store.get_klines",
            return_value=fresh,
        ), patch(
            "app.gemini_gold_trader.klines.newest_bar_age_s",
            return_value=120.0,
        ), patch(
            "app.gemini_gold_trader.klines.stale_limit_s",
            return_value=1800.0,
        ):
            bars, meta = await get_chart_klines("15m", 80)
        assert len(bars) == 25
        assert meta["source"] == "ctrader"
        assert meta["status"] == "ok"

    asyncio.run(_run())


def test_get_chart_klines_marks_ctrader_unavailable_when_coinbase_and_no_fallback():
    from app.gemini_gold_trader.klines import get_chart_klines

    coinbase = _bars(30)

    async def _run():
        with patch(
            "app.services.tradfi_prices.get_klines",
            new_callable=AsyncMock,
            return_value=coinbase,
        ), patch(
            "app.services.tradfi_prices.get_metal_kline_source",
            return_value="coinbase",
        ), patch(
            "app.gemini_gold_trader.klines._fetch_ctrader_chart_klines",
            new_callable=AsyncMock,
            return_value=([], ""),
        ), patch(
            "app.services.kline_snapshot_store.get_klines",
            return_value=[],
        ):
            bars, meta = await get_chart_klines("5m", 80)
        assert bars == []
        assert meta["source"] == "ctrader_unavailable"
        assert meta["status"] == "missing_or_stale"

    asyncio.run(_run())


def test_klines_ready_requires_core_timeframes_only():
    from app.gemini_gold_trader.klines import klines_ready

    assert klines_ready(_bars(25), _bars(25), _bars(25)) is True
    assert klines_ready(_bars(5), _bars(25), _bars(25)) is False


def test_get_chart_klines_skips_tradfi_when_trendbars_blocked():
    """When cTrader trendbars are blocked, the tradfi external chain is skipped
    entirely — we go straight to the direct cTrader retry / postgres snapshot."""
    from app.gemini_gold_trader.klines import get_chart_klines

    tradfi = AsyncMock(return_value=_bars(30))
    snap = _bars(30)

    async def _run():
        with patch(
            "app.services.ctrader_price_feed.trendbar_fetch_blocked_reason",
            return_value="stream timeout XAUUSD 5m, retry in 20s",
        ), patch(
            "app.services.tradfi_prices.get_klines",
            new=tradfi,
        ), patch(
            "app.gemini_gold_trader.klines._fetch_ctrader_chart_klines",
            new_callable=AsyncMock,
            return_value=([], ""),
        ), patch(
            "app.gemini_gold_trader.klines._synthesize_forming_bar",
            side_effect=lambda bars, *a, **k: bars,
        ), patch(
            "app.services.kline_snapshot_store.get_klines",
            return_value=snap,
        ), patch(
            "app.gemini_gold_trader.klines.newest_bar_age_s",
            return_value=120.0,
        ), patch(
            "app.gemini_gold_trader.klines.stale_limit_s",
            return_value=1800.0,
        ):
            bars, meta = await get_chart_klines("5m", 80)
        assert len(bars) == 30
        assert meta["source"] == "ctrader"
        # tradfi (and its Coinbase/Kraken fallback chain) never invoked.
        tradfi.assert_not_awaited()

    asyncio.run(_run())


def test_charts_are_ctrader_flags_non_ctrader_source():
    from app.gemini_gold_trader.klines import charts_are_ctrader

    ok, reason = charts_are_ctrader(
        {"timeframe": "5m", "source": "ctrader", "bars": 30},
        {"timeframe": "15m", "source": "ctrader-cache", "bars": 30},
        {"timeframe": "1h", "source": "coinbase", "bars": 30},
    )
    assert ok is False
    assert reason == "non_ctrader_1h:coinbase"


def test_charts_are_ctrader_ignores_empty_metas():
    from app.gemini_gold_trader.klines import charts_are_ctrader

    ok, reason = charts_are_ctrader(
        None,
        {"timeframe": "5m", "source": "ctrader", "bars": 30},
        {"timeframe": "1m", "source": "ctrader_unavailable", "bars": 0},
    )
    assert ok is True
    assert reason == "ok"


def test_resolve_entry_chart_falls_back_to_5m():
    from app.gemini_gold_trader.klines import resolve_entry_chart

    k5 = _bars(40)
    entry, tf, fb = resolve_entry_chart([], k5)
    assert fb is True
    assert tf == "5m"
    assert len(entry) == 40
