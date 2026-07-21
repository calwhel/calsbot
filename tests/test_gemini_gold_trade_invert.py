"""Trade inversion unit tests."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.trade_invert import invert_take_decision


def _parse_prices_like_executor(decision):
    """Mirror executor._parse_prices inversion hook without heavy imports."""
    d = invert_take_decision(decision)
    direction = (d.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT"):
        return None
    entry = float(d["entry"])
    sl = float(d["stop_loss"])
    tp = float(d["take_profit"])
    return direction, entry, sl, tp


def test_invert_long_to_short_swaps_sl_tp():
    original = {
        "direction": "LONG",
        "entry": 2650.0,
        "stop_loss": 2640.0,
        "take_profit": 2670.0,
    }
    inverted = invert_take_decision(original)
    assert inverted["direction"] == "SHORT"
    assert inverted["stop_loss"] == 2670.0
    assert inverted["take_profit"] == 2640.0


def test_invert_buy_alias():
    original = {
        "direction": "BUY",
        "entry": 2650.0,
        "stop_loss": 2640.0,
        "take_profit": 2670.0,
    }
    inverted = invert_take_decision(original)
    assert inverted["direction"] == "SHORT"
    assert inverted["stop_loss"] == 2670.0
    assert inverted["take_profit"] == 2640.0


def test_parse_prices_inverts_without_mutating_decision():
    original = {
        "direction": "LONG",
        "entry": 2650.0,
        "stop_loss": 2640.0,
        "take_profit": 2670.0,
    }
    parsed = _parse_prices_like_executor(original)
    assert parsed == ("SHORT", 2650.0, 2670.0, 2640.0)
    assert original["direction"] == "LONG"
    assert original["stop_loss"] == 2640.0
    assert original["take_profit"] == 2670.0


def test_parse_prices_inverts_short_to_long():
    original = {
        "direction": "SHORT",
        "entry": 2650.0,
        "stop_loss": 2660.0,
        "take_profit": 2640.0,
    }
    parsed = _parse_prices_like_executor(original)
    assert parsed == ("LONG", 2650.0, 2640.0, 2660.0)
