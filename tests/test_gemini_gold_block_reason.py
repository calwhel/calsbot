"""Block reason formatting for gemini-gold Telegram."""
from app.gemini_gold_trader.block_reason import format_block_reason


def test_format_cap_block_reasons():
    assert format_block_reason("max_open_position") == "blocked: max_open_position"
    assert format_block_reason("fire_time:max_open_position") == "blocked: max_open_position"
    assert (
        format_block_reason("blocked: max_open_position — 1 open row(s), 0 in-flight")
        == "blocked: max_open_position — 1 open row(s), 0 in-flight"
    )
    assert format_block_reason("NOT_ENOUGH_MONEY: margin") == "NOT_ENOUGH_MONEY: margin"
