"""Gemini Gold pending entry sync tests."""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


def test_sync_pending_entries_market_fallback_on_expire():
    from app.gemini_gold_trader.config import env_defaults
    from app.gemini_gold_trader.pending_entry import sync_pending_entries

    cfg = env_defaults()
    now = datetime.utcnow()
    pending_row = MagicMock()
    pending_row.id = 7
    pending_row.status = "pending"
    pending_row.decision_id = 42
    pending_row.direction = "SHORT"
    pending_row.entry_price = 2650.0
    pending_row.expires_at = now - timedelta(minutes=1)
    pending_row.session = "new_york"
    pending_row.notes = "entry-watch pending"

    dec_row = MagicMock()
    dec_row.decision = {
        "direction": "SHORT",
        "entry": 2650.0,
        "stop_loss": 2660.0,
        "take_profit": 2630.0,
        "setup_type": "liquidity_grab_short",
    }
    dec_row.executed = False
    dec_row.setup_type = "liquidity_grab_short"

    db = MagicMock()

    def _query_chain(model):
        chain = MagicMock()
        if model.__name__ == "GeminiGoldPendingOrder":
            chain.filter.return_value.order_by.return_value.all.return_value = [pending_row]
        else:
            chain.filter.return_value.first.return_value = dec_row
        return chain

    db.query.side_effect = _query_chain

    async def _run():
        with patch(
            "app.gemini_gold_trader.pending_entry.run_in_db_thread",
            new_callable=AsyncMock,
        ) as run_db:
            run_db.side_effect = [
                (now, [pending_row]),
                dec_row,
            ]
            with patch(
                "app.gemini_gold_trader.pending_entry.db_commit",
                new_callable=AsyncMock,
            ):
                with patch(
                    "app.gemini_gold_trader.pending_entry._try_market_fill_from_pending",
                    new_callable=AsyncMock,
                    return_value=True,
                ) as market_fill:
                    filled = await sync_pending_entries(db, cfg, spot=2655.0)
        assert filled == 1
        market_fill.assert_awaited_once()

    asyncio.run(_run())
