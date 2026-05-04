"""Expo push notification helper.

Sends push notifications to all device tokens registered for a given user
via the Expo push service. Designed to be safe to call from anywhere — every
helper swallows its own exceptions, so push failures can never break trade
execution or other business logic.

Usage:
    from app.services.expo_push import notify_user_bg
    notify_user_bg(
        user.id,
        title="Paper trade opened",
        body="BTC LONG @ $67,123",
        data={"type": "trade_open", "strategy_id": 42},
        kind="paper",            # "paper" | "live" | "manual"
        position_usd=25.0,       # used by the min-notional preference filter
    )

Per-user filtering (UserPreference):
    push_notify_paper       — suppress when False and kind == "paper"
    push_notify_live        — suppress when False and kind in {"live","manual"}
    push_min_position_usd   — suppress when >0 and position_usd is below it

The mobile client registers its Expo push token via
POST /api/mobile/push/register before any of these calls will deliver.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import httpx

from app.database import SessionLocal
from app.models import MobilePushToken, UserPreference

logger = logging.getLogger(__name__)

_EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"


def _tokens_for_user(user_id: int) -> list[str]:
    db = SessionLocal()
    try:
        rows = db.query(MobilePushToken).filter(
            MobilePushToken.user_id == user_id
        ).all()
        return [r.token for r in rows if r.token]
    finally:
        db.close()


def _should_send(user_id: int, kind: str, position_usd: Optional[float]) -> bool:
    """Apply the user's push preferences. Returns False to suppress.
    Fails OPEN — if prefs lookup throws, deliver the notification."""
    try:
        db = SessionLocal()
        try:
            prefs = db.query(UserPreference).filter(
                UserPreference.user_id == user_id
            ).first()
            if not prefs:
                return True
            k = (kind or "").lower()
            if k == "paper" and getattr(prefs, "push_notify_paper", True) is False:
                return False
            if k in ("live", "manual") and getattr(prefs, "push_notify_live", True) is False:
                return False
            min_usd = float(getattr(prefs, "push_min_position_usd", 0) or 0)
            if min_usd > 0 and position_usd is not None and position_usd < min_usd:
                return False
            return True
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"_should_send({user_id}) prefs lookup failed: {e} — sending")
        return True


async def _send_messages(messages: list[dict]) -> None:
    if not messages:
        return
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.post(
                _EXPO_PUSH_URL,
                json=messages,
                headers={
                    "Accept": "application/json",
                    "Accept-encoding": "gzip, deflate",
                    "Content-Type": "application/json",
                },
            )
            if r.status_code >= 400:
                logger.warning(
                    f"Expo push HTTP {r.status_code}: {r.text[:200]}"
                )
    except Exception as e:
        logger.warning(f"Expo push send failed: {e}")


async def notify_user(
    user_id: int,
    title: str,
    body: str,
    data: Optional[dict[str, Any]] = None,
    sound: str = "default",
    kind: str = "live",
    position_usd: Optional[float] = None,
    channel: str = "trade-fires",
) -> None:
    """Send a notification to every registered device for `user_id`,
    subject to the user's push preferences. Never raises."""
    try:
        if not await asyncio.to_thread(_should_send, user_id, kind, position_usd):
            return
        tokens = await asyncio.to_thread(_tokens_for_user, user_id)
        if not tokens:
            return
        messages = [
            {
                "to": tok,
                "title": title,
                "body": body,
                "sound": sound,
                "data": data or {},
                "priority": "high",
                "channelId": channel,
            }
            for tok in tokens
        ]
        await _send_messages(messages)
    except Exception as e:
        logger.warning(f"notify_user({user_id}) failed: {e}")


def notify_user_bg(
    user_id: int,
    title: str,
    body: str,
    data: Optional[dict[str, Any]] = None,
    kind: str = "live",
    position_usd: Optional[float] = None,
    channel: str = "trade-fires",
) -> None:
    """Fire-and-forget — schedules `notify_user` on the running event loop.
    Safe to call from sync code that's already inside an async context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(notify_user(user_id, title, body, data, kind=kind, position_usd=position_usd, channel=channel))
        else:
            loop.run_until_complete(notify_user(user_id, title, body, data, kind=kind, position_usd=position_usd, channel=channel))
    except Exception as e:
        logger.warning(f"notify_user_bg({user_id}) scheduling failed: {e}")


def notify_trade_close_bg(
    user_id: int,
    strategy_name: str,
    symbol: str,
    direction: str,
    outcome: str,
    pnl_pct: float,
    leverage: int,
    entry_price: float,
    exit_price: float,
    strategy_id: int = 0,
    execution_id: int = 0,
    is_paper: bool = False,
    duration_mins: int = 0,
    kind: str = "live",
    position_usd: Optional[float] = None,
) -> None:
    """Push notification when a trade closes (TP/SL hit). Biggest gap previously."""
    coin = symbol.replace("USDT", "")
    pnl_sign = "+" if pnl_pct >= 0 else ""

    if outcome == "WIN":
        icon = "🎯"
        result_label = "TP HIT"
    elif outcome == "LOSS":
        icon = "🛑"
        result_label = "SL HIT"
    elif outcome == "BREAKEVEN":
        icon = "⚖️"
        result_label = "BREAKEVEN"
    else:
        icon = "📊"
        result_label = outcome

    dur_str = ""
    if duration_mins > 0:
        if duration_mins >= 60:
            dur_str = f" · {duration_mins // 60}h {duration_mins % 60}m"
        else:
            dur_str = f" · {duration_mins}m"

    paper_tag = " (Paper)" if is_paper else ""
    title = f"{icon} {result_label}{paper_tag} — {coin}"
    body = (
        f"{strategy_name}: {coin} {direction} {leverage}× "
        f"→ {pnl_sign}{pnl_pct:.1f}%{dur_str}"
    )

    data = {
        "type": "trade_close",
        "strategy_id": strategy_id,
        "execution_id": execution_id,
        "outcome": outcome,
        "screen": f"/strategy/{strategy_id}",
    }

    channel = "trade-results" if outcome in ("WIN", "LOSS", "BREAKEVEN") else "trade-fires"
    notify_user_bg(user_id, title, body, data, kind=kind, position_usd=position_usd, channel=channel)


def notify_tp_hit_bg(
    user_id: int,
    strategy_name: str,
    symbol: str,
    direction: str,
    tp_level: str,
    tp_price: float,
    tp_roi: float,
    leverage: int,
    strategy_id: int = 0,
    execution_id: int = 0,
    is_paper: bool = False,
    kind: str = "live",
    position_usd: Optional[float] = None,
) -> None:
    """Push notification when an intermediate TP level (TP1/TP2/TP3) is hit."""
    coin = symbol.replace("USDT", "")
    paper_tag = " 📝" if is_paper else ""
    title = f"🎯 {tp_level} HIT{paper_tag} — {coin}"
    body = (
        f"{strategy_name}: {coin} {direction} {leverage}× "
        f"hit {tp_level} @ ${tp_price:,.4f} ({tp_roi:+.1f}%)"
    )
    data = {
        "type": "tp_hit",
        "tp_level": tp_level,
        "strategy_id": strategy_id,
        "execution_id": execution_id,
        "screen": f"/strategy/{strategy_id}",
    }
    notify_user_bg(user_id, title, body, data, kind=kind, position_usd=position_usd, channel="trade-progress")


def notify_breakeven_bg(
    user_id: int,
    strategy_name: str,
    symbol: str,
    direction: str,
    leverage: int,
    current_roi: float,
    strategy_id: int = 0,
    execution_id: int = 0,
    kind: str = "live",
) -> None:
    """Push notification when stop loss is moved to breakeven."""
    coin = symbol.replace("USDT", "")
    title = f"🛡️ Breakeven — {coin}"
    body = (
        f"{strategy_name}: SL moved to entry ({current_roi:+.1f}% ROI). "
        f"This trade is now risk-free."
    )
    data = {
        "type": "breakeven_moved",
        "strategy_id": strategy_id,
        "execution_id": execution_id,
        "screen": f"/strategy/{strategy_id}",
    }
    notify_user_bg(user_id, title, body, data, kind=kind, channel="trade-progress")
