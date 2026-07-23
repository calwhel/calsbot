"""cTrader broker reachability checks before order placement."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

from app.gemini_gold_trader.config import GeminiGoldRuntimeConfig
from app.gemini_gold_trader.guardrails import active_ctrader_account_id

logger = logging.getLogger(__name__)

_EXEC_BROKER_TIMEOUT_S = max(
    15.0,
    float(os.environ.get("GEMINI_GOLD_EXEC_BROKER_TIMEOUT_S", "30")),
)

# A reconcile snapshot captured earlier in the same scan cycle can be reused
# for the execution-time reachability gate instead of paying for a second full
# broker poll (connect + auth + reconcile on live+demo hosts) right after a
# long Gemini vision call. This prevents high-confidence TAKEs from being lost
# to ``ctrader_poll_timeout`` when the broker is actually reachable.
#
# Two tiers:
#   * FRESH  — recent enough to skip the second poll entirely (fast path).
#   * REUSE  — older but still acceptable as a fallback when a fresh poll times
#              out or errors, rather than dropping the trade.
_EXEC_SNAPSHOT_FRESH_TTL_S = max(
    15.0,
    float(os.environ.get("GEMINI_GOLD_EXEC_SNAPSHOT_FRESH_TTL_S", "45")),
)
_EXEC_SNAPSHOT_REUSE_TTL_S = max(
    _EXEC_SNAPSHOT_FRESH_TTL_S,
    float(os.environ.get("GEMINI_GOLD_EXEC_SNAPSHOT_REUSE_TTL_S", "150")),
)


def snapshot_is_reusable(
    snapshot: Optional[Dict[str, Any]],
    captured_at: Optional[float],
    *,
    ttl_s: Optional[float] = None,
) -> bool:
    """True when a cached reconcile snapshot is fresh enough to trust."""
    if not snapshot or captured_at is None:
        return False
    if snapshot.get("position_ids") is None:
        return False
    ttl = _EXEC_SNAPSHOT_REUSE_TTL_S if ttl_s is None else ttl_s
    return (time.monotonic() - float(captured_at)) <= ttl


async def broker_reachable_for_execution(
    db,
    cfg: GeminiGoldRuntimeConfig,
    *,
    user_id: Optional[int] = None,
    cached_snapshot: Optional[Dict[str, Any]] = None,
    cached_at: Optional[float] = None,
    lightweight: bool = False,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Quick reconcile poll — returns (ok, reason, snapshot).
    Blocks order routing when broker is down or OAuth is missing.

    When ``cached_snapshot`` (with a monotonic ``cached_at`` timestamp) from an
    earlier gate in the same cycle is still fresh, it is reused directly to
    avoid a redundant broker poll.

    ``lightweight`` polls open positions only (skips the balance/equity trader
    request) — enough for a reachability gate and markedly faster.
    """
    uid = int(user_id or cfg.demo_user_id or 0)
    ctid = active_ctrader_account_id(cfg)
    if uid <= 0:
        return False, "no_demo_user", None
    if not ctid:
        return False, "no_trading_account", None

    if snapshot_is_reusable(cached_snapshot, cached_at, ttl_s=_EXEC_SNAPSHOT_FRESH_TTL_S):
        return True, "ok_cached", cached_snapshot

    from app.models import UserPreference
    from app.services.ctrader_client import (
        get_broker_positions_snapshot_resilient,
        get_broker_reconcile_snapshot_resilient,
    )

    prefs = db.query(UserPreference).filter(UserPreference.user_id == uid).first()
    token = getattr(prefs, "ctrader_access_token", None) if prefs else None
    if not token:
        return False, "no_ctrader_token", None

    poll = (
        get_broker_positions_snapshot_resilient
        if lightweight
        else get_broker_reconcile_snapshot_resilient
    )

    try:
        snap = await asyncio.wait_for(
            poll(
                str(token),
                int(ctid),
                prefs=prefs,
                user_id=uid,
            ),
            timeout=_EXEC_BROKER_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        if snapshot_is_reusable(cached_snapshot, cached_at, ttl_s=_EXEC_SNAPSHOT_REUSE_TTL_S):
            logger.info(
                "[gemini-gold] execution broker poll timed out — reusing "
                "cached scan-start snapshot"
            )
            return True, "ok_cached_after_timeout", cached_snapshot
        return False, "ctrader_poll_timeout", None
    except Exception as exc:
        logger.warning("[gemini-gold] broker preflight error: %s", exc)
        if snapshot_is_reusable(cached_snapshot, cached_at, ttl_s=_EXEC_SNAPSHOT_REUSE_TTL_S):
            logger.info(
                "[gemini-gold] execution broker poll errored (%s) — reusing "
                "cached scan-start snapshot",
                str(exc)[:80],
            )
            return True, "ok_cached_after_error", cached_snapshot
        return False, str(exc)[:80], None

    if snap.get("position_ids") is not None:
        return True, "ok", snap

    err = str(snap.get("error") or "broker_unreachable")
    if snap.get("auth_cooldown_s"):
        err = f"auth_cooldown:{snap.get('auth_cooldown_s')}s"
    return False, err, snap
