"""Entry zone timing — retrace-into-zone + chase guard for ICT conditions."""
import os

from app.services.strategy_ta import (
    entry_max_dist_from_zone_atr,
    entry_zone_allows_price,
)


def test_entry_max_dist_env_default():
    os.environ.pop("ENTRY_MAX_DIST_FROM_ZONE_ATR", None)
    assert entry_max_dist_from_zone_atr() == 0.35


def test_bear_fvg_in_zone_not_chasing():
    """Bear FVG 4101–4104: price at 4102 (in zone) passes; 4095 (below) fails."""
    atr = 4.0
    bottom, top = 4101.0, 4104.0

    ok, msg = entry_zone_allows_price(4102.0, bottom, top, "bearish", atr)
    assert ok is True
    assert msg == "in_zone"

    ok2, msg2 = entry_zone_allows_price(4095.0, bottom, top, "bearish", atr)
    assert ok2 is False
    assert "below zone" in msg2 or "chasing" in msg2


def test_bull_fvg_chasing_above_zone():
    atr = 2.0
    bottom, top = 2650.0, 2655.0
    ok, msg = entry_zone_allows_price(2665.0, bottom, top, "bullish", atr)
    assert ok is False
    assert "chasing" in msg or "above zone" in msg


def test_max_distance_blocks_deep_chase():
    os.environ["ENTRY_MAX_DIST_FROM_ZONE_ATR"] = "0.35"
    try:
        atr = 10.0
        bottom, top = 4101.0, 4104.0
        # 4 ATR below zone >> 0.35 limit
        ok, msg = entry_zone_allows_price(4060.0, bottom, top, "bearish", atr)
        assert ok is False
        assert "chasing" in msg
    finally:
        os.environ.pop("ENTRY_MAX_DIST_FROM_ZONE_ATR", None)


def test_fvg_just_formed_logic_rejects_chase_via_zone_guard():
    """just_formed now delegates to entry_zone_allows_price — bear chase blocked."""
    atr = 3.0
    bottom, top = 4101.0, 4104.0
    ok, _ = entry_zone_allows_price(4095.0, bottom, top, "bearish", atr)
    assert ok is False
    ok2, _ = entry_zone_allows_price(4102.0, bottom, top, "bearish", atr)
    assert ok2 is True
