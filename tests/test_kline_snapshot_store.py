"""Postgres kline snapshot store for cross-worker Gold AI reads."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.services import kline_snapshot_store as store


def test_upsert_and_read_roundtrip():
    bars = [[1_700_000_000_000, 1.0, 2.0, 0.5, 1.5, 10.0]] * 30
    mock_row = MagicMock()
    mock_row.bars_json = json.dumps(bars)
    mock_row.source = "ctrader"

    with patch.object(store, "_ensure_table"), patch(
        "app.database.SessionLocal"
    ) as mock_session:
        db = MagicMock()
        mock_session.return_value = db
        store.upsert_klines("XAUUSD", "5m", bars)
        db.execute.assert_called()
        db.commit.assert_called()

    with patch.object(store, "_ensure_table"), patch(
        "app.database.SessionLocal"
    ) as mock_session:
        db = MagicMock()
        mock_session.return_value = db
        db.execute.return_value.fetchone.return_value = mock_row
        out = store.get_klines("XAUUSD", "5m", 25)
    assert len(out) == 25
    assert out[-1][4] == 1.5


def test_get_returns_empty_when_stale():
    with patch.object(store, "_ensure_table"), patch(
        "app.database.SessionLocal"
    ) as mock_session:
        db = MagicMock()
        mock_session.return_value = db
        db.execute.return_value.fetchone.return_value = None
        out = store.get_klines("XAUUSD", "5m", 60, max_age_s=60.0)
    assert out == []
