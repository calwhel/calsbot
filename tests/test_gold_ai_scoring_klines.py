"""Gold AI scoring klines — postgres ctrader snapshot fallback."""
from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


def _bars(n: int = 40) -> list:
    ts = int(time.time() * 1000)
    return [[ts, 2650.0, 2652.0, 2648.0, 2651.0, 0.0] for _ in range(n)]


def test_fetch_gold_scoring_k5_uses_postgres_when_tradfi_is_coinbase():
    from app.gold_ai_trader.klines import fetch_gold_scoring_k5

    snap = _bars()

    async def _run():
        with patch(
            "app.gold_ai_trader.klines.get_klines",
            new_callable=AsyncMock,
            return_value=snap,
        ), patch(
            "app.gold_ai_trader.klines.get_metal_kline_source",
            return_value="coinbase",
        ), patch(
            "app.services.kline_snapshot_store.get_klines",
            return_value=snap,
        ):
            rows, src = await fetch_gold_scoring_k5(user_id=1)
        assert src == "ctrader"
        assert len(rows) == 40

    asyncio.run(_run())


def test_snapshot_row_max_age_s_defaults():
    from app.services.kline_snapshot_store import snapshot_row_max_age_s

    assert snapshot_row_max_age_s("5m") == 600.0
    assert snapshot_row_max_age_s("1h") == 7200.0
