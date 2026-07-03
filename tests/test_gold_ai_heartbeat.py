"""Gold AI loop vs scan heartbeat."""
from __future__ import annotations

import os
from datetime import datetime, timedelta

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


def test_loop_heartbeat_age_tracks_outside_session_ticks():
    from app.gold_ai_trader import state as runtime_state
    from app.gold_ai_trader.loop import loop_heartbeat_age_seconds, scan_heartbeat_age_seconds

    runtime_state.set_status(
        last_scan_at=(datetime.utcnow() - timedelta(hours=1)).isoformat() + "Z",
        last_loop_at=datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        dormant_reason="outside_session",
    )
    scan_age = scan_heartbeat_age_seconds()
    loop_age = loop_heartbeat_age_seconds()
    assert scan_age is not None and scan_age > 3000
    assert loop_age is not None and loop_age < 5
