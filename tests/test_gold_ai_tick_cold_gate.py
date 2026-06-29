"""Gold AI tick-cold data gate."""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gold_ai_trader.data_quality import gold_data_ok_for_claude


def test_gold_data_ok_blocks_tick_cold():
    ok, reason = gold_data_ok_for_claude(
        {
            "price": 2650.0,
            "live_source": None,
            "price_source": "ctrader_kline_close",
            "kline_source": "ctrader",
            "kline_synthetic": False,
            "klines_stale": False,
            "kline_bars": 40,
            "spot_tick_cold": True,
            "spot_tick_age_s": 12.0,
        }
    )
    assert ok is False
    assert reason.startswith("tick_cold")
