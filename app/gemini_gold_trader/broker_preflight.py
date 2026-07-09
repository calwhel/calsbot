"""cTrader broker reachability checks before order placement."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional, Tuple

from app.gemini_gold_trader.config import GeminiGoldRuntimeConfig
from app.gemini_gold_trader.guardrails import active_ctrader_account_id

logger = logging.getLogger(__name__)

_EXEC_BROKER_TIMEOUT_S = max(
    15.0,
    float(os.environ.get("GEMINI_GOLD_EXEC_BROKER_TIMEOUT_S", "30")),
)


async def broker_reachable_for_execution(
    db,
    cfg: GeminiGoldRuntimeConfig,
    *,
    user_id: Optional[int] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Quick reconcile poll — returns (ok, reason, snapshot).
    Blocks order routing when broker is down or OAuth is missing.
    """
    uid = int(user_id or cfg.demo_user_id or 0)
    ctid = active_ctrader_account_id(cfg)
    if uid <= 0:
        return False, "no_demo_user", None
    if not ctid:
        return False, "no_trading_account", None

    from app.models import UserPreference
    from app.services.ctrader_client import get_broker_reconcile_snapshot_resilient

    prefs = db.query(UserPreference).filter(UserPreference.user_id == uid).first()
    token = getattr(prefs, "ctrader_access_token", None) if prefs else None
    if not token:
        return False, "no_ctrader_token", None

    try:
        snap = await asyncio.wait_for(
            get_broker_reconcile_snapshot_resilient(
                str(token),
                int(ctid),
                prefs=prefs,
                user_id=uid,
            ),
            timeout=_EXEC_BROKER_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        return False, "ctrader_poll_timeout", None
    except Exception as exc:
        logger.warning("[gemini-gold] broker preflight error: %s", exc)
        return False, str(exc)[:80], None

    if snap.get("position_ids") is not None:
        return True, "ok", snap

    err = str(snap.get("error") or "broker_unreachable")
    if snap.get("auth_cooldown_s"):
        err = f"auth_cooldown:{snap.get('auth_cooldown_s')}s"
    return False, err, snap
