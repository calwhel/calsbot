"""Gemini Gold Telegram notification formatting."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.telegram_notify import format_decision_message


def test_format_includes_setup_type_and_entry_watch_fill():
    text = format_decision_message(
        session="london",
        decision={
            "direction": "LONG",
            "entry": 4138.52,
            "stop_loss": 4136.52,
            "take_profit": 4141.52,
            "setup_type": "liquidity_grab",
            "rationale": "MSS break long",
        },
        action="TAKE",
        confidence=80,
        executed=True,
        execution_id=42,
        fill_kind="entry_watch",
        execution_mode="demo",
    )
    assert "liquidity_grab" in text
    assert "Entry-watch filled" in text
    assert "exec #42" in text


def test_format_live_execution_label():
    text = format_decision_message(
        session="new_york",
        decision={
            "direction": "SHORT",
            "entry": 2650.0,
            "stop_loss": 2655.0,
            "take_profit": 2645.0,
            "setup_type": "momentum_scalp",
            "rationale": "test",
        },
        action="TAKE",
        confidence=85,
        executed=True,
        execution_id=7,
        execution_mode="live",
    )
    assert "Live order placed" in text
    assert "momentum_scalp" in text
