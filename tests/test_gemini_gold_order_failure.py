"""Gemini Gold broker rejection surfacing + demo lot defaults."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.config import env_defaults
from app.gemini_gold_trader.models import GeminiGoldConfig
from app.gemini_gold_trader.schema import _migrate_legacy_demo_lot_size

ROOT = Path(__file__).resolve().parents[1]
LOOP = (ROOT / "app" / "gemini_gold_trader" / "loop.py").read_text()
EXECUTOR = (ROOT / "app" / "gemini_gold_trader" / "executor.py").read_text()


def test_env_default_demo_lot_is_0_01():
    cfg = env_defaults()
    assert cfg.demo_lot_size == 0.01


def test_migrate_legacy_demo_lot_size():
    db = MagicMock()
    row = GeminiGoldConfig(id=1, demo_lot_size=0.1)
    db.query.return_value.filter.return_value.first.return_value = row
    _migrate_legacy_demo_lot_size(db)
    assert row.demo_lot_size == 0.01
    db.commit.assert_called_once()


def test_loop_surfaces_broker_error_in_block_reason():
    assert 'order_ctx.get("broker_error")' in LOOP
    assert "order_ctx=order_ctx" in LOOP


def test_executor_records_broker_error_in_order_ctx():
    assert 'order_ctx["broker_error"]' in EXECUTOR
    assert "cfg.demo_lot_size or 0.01" in EXECUTOR
