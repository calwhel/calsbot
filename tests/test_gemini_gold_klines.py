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
        assert meta["source"] == "postgres_snapshot"
        assert meta["status"] == "ok"

    asyncio.run(_run())


def test_klines_ready_requires_min_bars():
    from app.gemini_gold_trader.klines import klines_ready

    assert klines_ready(_bars(25), _bars(25)) is True
    assert klines_ready(_bars(5), _bars(25)) is False
