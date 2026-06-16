"""Marketplace browse ranking and publish idempotency helpers."""
import os
import time

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

from strategy_portal_server import _mkt_top_sort_key


def test_mkt_top_sort_key_prefers_human_over_zero_perf_ai():
    now = time.time()
    human = {
        "asset_class": "crypto",
        "live_pnl": None,
        "is_ai_generated": False,
        "published_at_ts": now,
    }
    ai = {
        "asset_class": "crypto",
        "live_pnl": 0,
        "is_ai_generated": True,
        "published_at_ts": now - 86400,
    }
    assert _mkt_top_sort_key(human) > _mkt_top_sort_key(ai)


def test_mkt_top_sort_key_perf_still_wins():
    now = time.time()
    strong_ai = {
        "asset_class": "crypto",
        "live_pnl": 50.0,
        "is_ai_generated": True,
        "published_at_ts": now,
    }
    new_human = {
        "asset_class": "forex",
        "live_pips_pnl": None,
        "is_ai_generated": False,
        "published_at_ts": now,
    }
    assert _mkt_top_sort_key(strong_ai) > _mkt_top_sort_key(new_human)


def test_mkt_top_sort_key_uses_pips_for_tradfi():
    fx = {
        "asset_class": "forex",
        "live_pips_pnl": 120.0,
        "is_ai_generated": False,
        "published_at_ts": time.time(),
    }
    assert _mkt_top_sort_key(fx) > 100.0
