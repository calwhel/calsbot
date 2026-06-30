"""Telegram alert when Gold AI is blocked by fallback klines too long."""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from app.gold_ai_trader import telegram_notify as notify


@pytest.fixture(autouse=True)
def _reset_fallback_state():
    notify.clear_fallback_klines_notify_state()
    yield
    notify.clear_fallback_klines_notify_state()


@pytest.mark.asyncio
async def test_fallback_alert_not_sent_before_threshold():
    notify._fallback_klines_since_mono = time.monotonic() - 60.0  # 1 minute
    with patch.object(notify, "_send", new_callable=AsyncMock) as mock_send:
        ok = await notify.maybe_notify_fallback_klines_blocked(
            "fallback_klines:coinbase", source_tag="price:ctrader/kline:coinbase",
        )
    assert ok is False
    mock_send.assert_not_called()


@pytest.mark.asyncio
async def test_fallback_alert_sent_once_after_threshold():
    notify._fallback_klines_since_mono = time.monotonic() - 6 * 60.0
    with patch.object(notify, "_send", new_callable=AsyncMock, return_value=True) as mock_send:
        ok1 = await notify.maybe_notify_fallback_klines_blocked(
            "fallback_klines:coinbase", source_tag="price:ctrader/kline:coinbase",
        )
        ok2 = await notify.maybe_notify_fallback_klines_blocked(
            "fallback_klines:coinbase", source_tag="price:ctrader/kline:coinbase",
        )
    assert ok1 is True
    assert ok2 is False
    mock_send.assert_called_once()
    assert "[DEMO] Gold AI Trader" in mock_send.call_args[0][0]
    assert "coinbase" in mock_send.call_args[0][0]


@pytest.mark.asyncio
async def test_fallback_alert_resets_when_data_recovers():
    notify._fallback_klines_since_mono = time.monotonic() - 6 * 60.0
    notify._fallback_klines_alert_sent = True
    with patch.object(notify, "_send", new_callable=AsyncMock) as mock_send:
        await notify.maybe_notify_fallback_klines_blocked("ok", source_tag="price:ctrader/kline:ctrader")
    assert notify._fallback_klines_since_mono is None
    assert notify._fallback_klines_alert_sent is False
    mock_send.assert_not_called()


@pytest.mark.asyncio
async def test_prefetch_snapshot_unblocks_gold_data_gate():
    """Portal postgres snapshot → ctrader kline_source passes Claude gate."""
    from app.gold_ai_trader.data_quality import gold_data_ok_for_claude

    bars = [[int(time.time() * 1000), 2650, 2652, 2648, 2651, 0.0]] * 40
    ok, reason = gold_data_ok_for_claude(
        {
            "price": 2651.0,
            "live_source": "ctrader",
            "price_source": "ctrader",
            "kline_source": "ctrader",
            "kline_synthetic": False,
            "klines_stale": False,
            "kline_bars": len(bars),
            "spot_tick_cold": False,
        }
    )
    assert ok is True
    assert reason == "ok"


@pytest.mark.asyncio
async def test_assess_gold_market_data_uses_ctrader_when_postgres_snapshot_hit():
    from app.gold_ai_trader.data_quality import assess_gold_market_data, gold_data_ok_for_claude

    bars = [[int(time.time() * 1000), 2650, 2652, 2648, 2651, 0.0]] * 40
    with patch(
        "app.gold_ai_trader.data_refresh.refresh_gold_scoring_klines",
        new_callable=AsyncMock,
    ), patch(
        "app.gold_ai_trader.data_quality._resolve_ctrader_spot",
        new_callable=AsyncMock,
        return_value=(2651.0, "ctrader", 2650.0, 2652.0),
    ), patch(
        "app.gold_ai_trader.data_quality.get_klines",
        new_callable=AsyncMock,
        return_value=bars,
    ), patch(
        "app.gold_ai_trader.data_quality.get_metal_kline_source",
        return_value="ctrader",
    ), patch(
        "app.gold_ai_trader.data_quality.get_metal_kline_fetched_at",
        return_value=None,
    ), patch(
        "app.gold_ai_trader.data_quality.get_metal_kline_fetch_age_s",
        return_value=5.0,
    ), patch(
        "app.gold_ai_trader.data_quality.is_metal_kline_synthetic",
        return_value=False,
    ), patch(
        "app.gold_ai_trader.data_quality.check_cached_klines_stale",
        new_callable=AsyncMock,
        return_value=(False, ""),
    ):
        data = await assess_gold_market_data(user_id=2)
    assert data["kline_source"] == "ctrader"
    ok, reason = gold_data_ok_for_claude(data)
    assert ok is True
    assert reason == "ok"
