"""Unit tests for Gold AI Trader upgrades (gold module only)."""
from __future__ import annotations

import os
from datetime import datetime, timedelta

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gold_ai_trader.context_levels import (
    build_key_levels_block,
    compute_pdh_pdl,
    format_level_line,
)
from app.gold_ai_trader.context_regime import build_regime_block, compute_1h_trend
from app.gold_ai_trader.learning import compute_r_multiple, get_setup_stats
from app.gold_ai_trader.pending_entry import (
    broker_limit_supported,
    compute_pending_expiry,
    entry_price_touched,
)
from app.gold_ai_trader.models import GoldAiOutcome


class _Cfg:
    london_start_hour = 7
    london_end_hour = 10
    ny_start_hour = 13
    ny_end_hour = 16


def _bar(ts: datetime, o, h, l, c, v=100.0):
    return [int(ts.timestamp() * 1000), o, h, l, c, v]


def test_format_level_near_tag():
    line = format_level_line("PDL", 4152.2, 4153.57, 4.0)
    assert "← near" in line
    assert "0.3× ATR" in line


def test_pdh_pdl_from_daily():
    now = datetime(2026, 6, 18, 9, 0, 0)
    y = now - timedelta(days=1)
    daily = [_bar(y.replace(hour=0), 4100, 4172.5, 4148.1, 4160, 1)]
    pdh, pdl = compute_pdh_pdl(now=now, k_daily=daily, k_1h=[], k_5m=[])
    assert pdh == 4172.5
    assert pdl == 4148.1


def test_regime_block_with_data():
    now = datetime(2026, 6, 18, 9, 0, 0)
    k1h = []
    price = 4150.0
    for i in range(30):
        ts = now - timedelta(hours=30 - i)
        c = price + i * 0.5
        k1h.append(_bar(ts, c - 1, c + 2, c - 2, c, 1))
    k5m = k1h[-20:]
    block = build_regime_block(k1h, k5m)
    assert "=== REGIME ===" in block[0]
    assert "1h trend:" in block[1]


def test_entry_price_touched_long():
    assert entry_price_touched("LONG", 2649.5, 2650.0, 0.2)
    assert not entry_price_touched("LONG", 2651.0, 2650.0, 0.2)


def test_pending_expiry_session_cap():
    cfg = _Cfg()
    now = datetime(2026, 6, 18, 9, 45, 0)
    exp = compute_pending_expiry(now, "london", cfg, 30)
    assert exp <= now.replace(hour=10, minute=0, second=0, microsecond=0)


def test_r_multiple():
    r = compute_r_multiple(direction="LONG", entry=100.0, stop_loss=99.0, pnl_pct=2.0)
    assert r == 2.0


def test_get_setup_stats_empty():
    class _Db:
        def query(self, *a, **k):
            return self

        def filter(self, *a, **k):
            return self

        def all(self):
            return []

    assert get_setup_stats(_Db()) == []


def test_limit_order_helper_defined_in_ctrader_client():
    from pathlib import Path

    text = Path(__file__).resolve().parents[3] / "app" / "services" / "ctrader_client.py"
    src = text.read_text(encoding="utf-8")
    assert "async def place_limit_order_resilient" in src
    assert "ProtoOAOrderType.LIMIT" in src
