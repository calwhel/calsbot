"""Gemini Gold runtime state."""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader import state as runtime_state


def test_note_dormant_exposes_reason():
    runtime_state.note_dormant("outside_trade_hours")
    st = runtime_state.get_status()
    assert st["status"] == "dormant"
    assert st["dormant_reason"] == "outside_trade_hours"


def test_note_scan_clears_dormant_reason():
    runtime_state.note_dormant("outside_session")
    runtime_state.note_scan("london")
    st = runtime_state.get_status()
    assert st["status"] == "scanning"
    assert st["active_session"] == "london"
    assert st["dormant_reason"] is None
