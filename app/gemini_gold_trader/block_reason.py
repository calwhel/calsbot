"""Human-readable block reasons for Gemini Gold Telegram + logs."""
from __future__ import annotations

from typing import Optional

_LABELS = {
    "max_open_position": "blocked: max_open_position",
    "max_trades_day": "blocked: max_trades_day",
    "min_trade_gap": "blocked: min_trade_gap",
    "kill_switch": "blocked: kill_switch",
    "dry_run": "blocked: dry_run",
    "no_demo_account": "blocked: no_demo_account",
    "no_trading_account": "blocked: no_trading_account",
    "no_demo_user": "blocked: no_demo_user",
    "already_reserved": "blocked: already_reserved",
    "already_executed": "blocked: already_executed",
    "decision_not_found": "blocked: decision_not_found",
}


def format_block_reason(reason: Optional[str]) -> Optional[str]:
    if not reason:
        return reason
    raw = str(reason).strip()
    if raw.startswith("blocked: max_open_position"):
        return raw
    if raw.startswith("fire_time:"):
        raw = raw[len("fire_time:") :]
    if raw in _LABELS:
        return _LABELS[raw]
    if raw.startswith("blocked:"):
        return raw
    if raw in ("ok",):
        return raw
    return raw
