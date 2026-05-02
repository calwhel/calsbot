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
    )

The mobile client registers its Expo push token via
POST /api/mobile/push/register before any of these calls will deliver.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import httpx

from app.database import SessionLocal
from app.models import MobilePushToken

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
) -> None:
    """Send a notification to every registered device for `user_id`.
    Never raises."""
    try:
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
                "channelId": "trade-fires",
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
) -> None:
    """Fire-and-forget — schedules `notify_user` on the running event loop.
    Safe to call from sync code that's already inside an async context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(notify_user(user_id, title, body, data))
        else:
            # No running loop — fall back to sync run (unusual).
            loop.run_until_complete(notify_user(user_id, title, body, data))
    except Exception as e:
        logger.warning(f"notify_user_bg({user_id}) scheduling failed: {e}")
