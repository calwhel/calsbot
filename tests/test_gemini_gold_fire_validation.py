"""Fire-time cap re-check for gemini-gold."""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.config import env_defaults
from app.gemini_gold_trader.fire_validation import revalidate_before_fire


def test_fire_time_blocks_when_caps_fail():
    cfg = env_defaults()
    cfg.demo_user_id = 42
    decision = {
        "action": "TAKE",
        "direction": "LONG",
        "entry": 2650.0,
        "stop_loss": 2640.0,
        "take_profit": 2670.0,
        "confidence": 80,
    }

    async def _run():
        with patch(
            "app.gemini_gold_trader.fire_validation.refresh_spot",
            new=AsyncMock(return_value=(2650.0, "ok")),
        ), patch(
            "app.gemini_gold_trader.db_thread.run_with_db",
            new=AsyncMock(return_value=(False, "max_open_position")),
        ):
            return await revalidate_before_fire(
                decision=decision,
                cfg=cfg,
                user_id=42,
                spot_hint=2650.0,
                decision_id=99,
            )

    ok, reason, _ = asyncio.run(_run())
    assert ok is False
    assert reason == "fire_time:max_open_position"
