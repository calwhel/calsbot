"""Shared daily Anthropic USD budget — fail-closed when cap is hit."""
from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_CALLER = "anthropic_budget_guard"
_budget_exceeded_logged = False
_state: dict = {"day": None, "total_usd": 0.0}


def daily_budget_usd() -> Optional[float]:
    raw = os.environ.get("ANTHROPIC_DAILY_BUDGET_USD", "").strip()
    if not raw:
        return None
    try:
        val = float(raw)
        return val if val > 0 else None
    except ValueError:
        return None


def _reset_if_new_day(now: datetime) -> None:
    today = now.date()
    if _state["day"] != today:
        _state["day"] = today
        _state["total_usd"] = 0.0
        global _budget_exceeded_logged
        _budget_exceeded_logged = False


def spent_today_usd(now: Optional[datetime] = None) -> float:
    _reset_if_new_day(now or datetime.utcnow())
    return float(_state["total_usd"])


def budget_exceeded(now: Optional[datetime] = None) -> bool:
    cap = daily_budget_usd()
    if cap is None:
        return False
    return spent_today_usd(now) >= cap


def record_call(caller: str, cost_usd: float, *, now: Optional[datetime] = None) -> float:
    """Record spend; return running daily total after this call."""
    ts = now or datetime.utcnow()
    _reset_if_new_day(ts)
    cost = max(0.0, float(cost_usd or 0.0))
    _state["total_usd"] = round(float(_state["total_usd"]) + cost, 6)
    total = spent_today_usd(ts)
    cap = daily_budget_usd()
    cap_txt = f"{cap:.2f}" if cap is not None else "unset"
    logger.info(
        "[anthropic-budget] caller=%s cost_usd=%.6f daily_total_usd=%.6f cap_usd=%s",
        caller,
        cost,
        total,
        cap_txt,
    )
    return total


def check_budget_or_block(caller: str, *, now: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Return (allowed, reason). When cap is exceeded, fail-closed (allowed=False).
    Logs once per UTC day when blocking.
    """
    global _budget_exceeded_logged
    cap = daily_budget_usd()
    if cap is None:
        return True, ""
    ts = now or datetime.utcnow()
    if not budget_exceeded(ts):
        return True, ""
    if not _budget_exceeded_logged:
        _budget_exceeded_logged = True
        logger.warning(
            "[anthropic-budget] DAILY CAP HIT — blocking %s "
            "(spent=%.4f cap=%.2f ANTHROPIC_DAILY_BUDGET_USD)",
            caller,
            spent_today_usd(ts),
            cap,
        )
    return False, "anthropic_daily_budget_exceeded"


def reset_daily_spend_for_tests(day: Optional[date] = None) -> None:
    """Test helper — reset in-memory ledger."""
    global _budget_exceeded_logged
    _state["day"] = day or datetime.utcnow().date()
    _state["total_usd"] = 0.0
    _budget_exceeded_logged = False
