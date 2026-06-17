"""Kline stale sweep — scope to strategy universe symbols only."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

from app.services.strategy_executor import _universe_symbols_for_snapshots


def test_universe_symbols_from_config_not_top_level_symbol():
    snaps = [
        {
            "id": 1,
            "config": {"universe": {"symbols": ["EURUSD", "GBPUSD"]}},
        },
        {
            "id": 2,
            "symbol": "SHOULD_IGNORE",
            "config": {"universe": {"symbols": ["eurusd", "XAUUSD"]}},
        },
    ]
    syms = _universe_symbols_for_snapshots(snaps)
    assert syms == ["EURUSD", "GBPUSD", "XAUUSD"]


def test_universe_symbols_empty_when_no_universe():
    assert _universe_symbols_for_snapshots([{"id": 1, "config": {}}]) == []


def test_universe_symbols_dedupes_across_snaps():
    snaps = [
        {"id": 1, "config": {"universe": {"symbols": ["NAS100"]}}},
        {"id": 2, "config": {"universe": {"symbols": ["NAS100", "US30"]}}},
    ]
    assert _universe_symbols_for_snapshots(snaps) == ["NAS100", "US30"]
