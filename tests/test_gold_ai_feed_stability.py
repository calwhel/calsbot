"""Tests for Gold AI feed stability upgrades."""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gold_ai_trader.klines import get_gold_ai_klines, is_ctrader_kline_source
from app.gold_ai_trader.data_refresh import refresh_gold_scoring_klines
from app.gold_ai_trader.schema import ensure_gold_ai_trader_schema


class TestGoldAiKlines:
    def test_is_ctrader_source(self):
        assert is_ctrader_kline_source("ctrader") is True
        assert is_ctrader_kline_source("ctrader-user") is True
        assert is_ctrader_kline_source("coinbase") is False

    @pytest.mark.asyncio
    async def test_daily_rejects_coinbase(self):
        fake_rows = [[1_700_000_000_000, 1, 2, 0.5, 1.5, 100]]
        with patch(
            "app.gold_ai_trader.klines.get_klines",
            new_callable=AsyncMock,
            return_value=fake_rows,
        ), patch(
            "app.gold_ai_trader.klines.get_metal_kline_source",
            return_value="coinbase",
        ):
            rows = await get_gold_ai_klines("1d", 5)
        assert rows == []

    @pytest.mark.asyncio
    async def test_daily_keeps_ctrader(self):
        fake_rows = [[1_700_000_000_000, 1, 2, 0.5, 1.5, 100]]
        with patch(
            "app.gold_ai_trader.klines.get_klines",
            new_callable=AsyncMock,
            return_value=fake_rows,
        ), patch(
            "app.gold_ai_trader.klines.get_metal_kline_source",
            return_value="ctrader",
        ):
            rows = await get_gold_ai_klines("1d", 5)
        assert rows == fake_rows


class TestDataRefresh:
    @pytest.mark.asyncio
    async def test_refresh_when_fresh(self):
        import time

        fresh_ts = int(time.time() * 1000)
        k5 = [[fresh_ts, 1, 2, 0.5, 1.5, 10.0]] * 25
        with patch(
            "app.gold_ai_trader.data_refresh.sweep_stale_metal_klines",
            new_callable=AsyncMock,
            return_value=0,
        ), patch(
            "app.services.ctrader_price_feed.sweep_stale_klines",
            new_callable=AsyncMock,
            return_value=0,
        ), patch(
            "app.gold_ai_trader.data_refresh.get_klines",
            new_callable=AsyncMock,
            return_value=k5,
        ):
            summary = await refresh_gold_scoring_klines()
        assert summary["refreshed_5m"] is False
        assert summary["cleared"] == 0
        assert summary["ctrader_swept"] == 0

    @pytest.mark.asyncio
    async def test_refresh_clears_when_stale(self):
        stale_ts = int((__import__("time").time() - 900) * 1000)
        k5_stale = [[stale_ts, 1, 2, 0.5, 1.5, 10.0]] * 25
        fresh_ts = int(__import__("time").time() * 1000)
        k5_fresh = [[fresh_ts, 1, 2, 0.5, 1.5, 10.0]] * 25
        with patch(
            "app.gold_ai_trader.data_refresh.sweep_stale_metal_klines",
            new_callable=AsyncMock,
            return_value=1,
        ), patch(
            "app.services.ctrader_price_feed.sweep_stale_klines",
            new_callable=AsyncMock,
            return_value=2,
        ), patch(
            "app.gold_ai_trader.data_refresh.clear_metal_kline_cache",
            return_value=3,
        ), patch(
            "app.gold_ai_trader.data_refresh.get_klines",
            new_callable=AsyncMock,
            side_effect=[k5_stale, k5_fresh],
        ), patch(
            "app.gold_ai_trader.data_refresh.get_metal_kline_source",
            return_value="ctrader",
        ):
            summary = await refresh_gold_scoring_klines()
        assert summary["swept"] == 1
        assert summary["ctrader_swept"] == 2
        assert summary["cleared"] == 3
        assert summary["refreshed_5m"] is True


class TestSchemaOnce:
    def test_schema_ensure_idempotent_flag(self, monkeypatch):
        import app.gold_ai_trader.schema as schema_mod

        monkeypatch.setattr(schema_mod, "_schema_ready", False)
        calls = {"n": 0}

        def fake_create_all(**kwargs):
            calls["n"] += 1

        class FakeInsp:
            def has_table(self, name):
                return True

        monkeypatch.setattr(schema_mod.Base.metadata, "create_all", fake_create_all)
        monkeypatch.setattr(schema_mod, "_apply_alters", lambda: None)
        monkeypatch.setattr(schema_mod, "inspect", lambda engine: FakeInsp())

        ensure_gold_ai_trader_schema()
        ensure_gold_ai_trader_schema()
        assert calls["n"] == 1
