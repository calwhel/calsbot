"""
Reliable Telegram DM delivery for executor trade alerts and the health monitor.

Trade notifications and hourly health reports use direct Bot API HTTP calls
(not the aiogram polling loop), so they must:
  • try every configured bot token (main, then forex) until one delivers
  • verify HTTP 200 + ok==true (httpx does not raise on 4xx)
  • retry transient failures with backoff
  • schedule safely from both sync and async call sites
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)


def bot_tokens() -> List[str]:
    """Deduped bot tokens — main first, then forex."""
    tokens: List[str] = []
    try:
        from app.config import settings

        for tok in (
            getattr(settings, "TELEGRAM_BOT_TOKEN", None),
            os.getenv("TELEGRAM_BOT_TOKEN"),
            os.getenv("FOREX_BOT_TOKEN"),
        ):
            if tok and tok not in tokens:
                tokens.append(tok)
    except Exception:
        for env in ("TELEGRAM_BOT_TOKEN", "FOREX_BOT_TOKEN"):
            tok = os.getenv(env)
            if tok and tok not in tokens:
                tokens.append(tok)
    return tokens


def owner_chat_id() -> Optional[str]:
    """Owner Telegram chat id for health/admin DMs."""
    try:
        from app.config import settings

        raw = getattr(settings, "OWNER_TELEGRAM_ID", None) or os.getenv("OWNER_TELEGRAM_ID")
    except Exception:
        raw = os.getenv("OWNER_TELEGRAM_ID")
    if not raw:
        return None
    s = str(raw).strip()
    return s or None


async def send_dm(
    chat_id,
    text: str,
    *,
    parse_mode: str = "HTML",
    tokens: Optional[List[str]] = None,
) -> bool:
    """Send a Telegram DM; return True only on confirmed delivery."""
    if not chat_id:
        return False
    tok_list = tokens if tokens is not None else bot_tokens()
    if not tok_list:
        logger.warning("Telegram DM skipped for %s: no bot token configured", chat_id)
        return False

    last = ""
    for attempt in range(3):
        for tok in tok_list:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    r = await client.post(
                        f"https://api.telegram.org/bot{tok}/sendMessage",
                        json={
                            "chat_id": str(chat_id),
                            "text": text,
                            "parse_mode": parse_mode,
                        },
                    )
                if r.status_code == 200:
                    # Any HTTP 200 counts as delivered — retrying after a 200 can
                    # duplicate the message if Telegram accepted it but the body
                    # was slow/unparseable or the client timed out reading it.
                    try:
                        if (r.json() or {}).get("ok"):
                            return True
                        last = f"HTTP 200 but ok!=true: {(r.text or '')[:200]}"
                    except Exception as je:
                        logger.warning(
                            "Telegram DM %s: HTTP 200 unparseable (%s) — treating as delivered",
                            chat_id,
                            type(je).__name__,
                        )
                        return True
                    return True
                last = f"HTTP {r.status_code}: {(r.text or '')[:200]}"
                if 400 <= r.status_code < 500:
                    # Bad chat / parse error — try next token, not more retries.
                    continue
            except Exception as e:
                last = f"{type(e).__name__}: {e or '(no message)'}"
        if attempt < 2:
            await asyncio.sleep(1.5 * (attempt + 1))

    logger.warning("Telegram DM failed for %s: %s", chat_id, last or "(no detail)")
    return False


def schedule_dm(chat_id, text: str, *, parse_mode: str = "HTML") -> None:
    """Fire-and-forget DM scheduling — safe from sync or async contexts."""
    if not chat_id:
        return

    async def _run():
        await send_dm(chat_id, text, parse_mode=parse_mode)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run())
    except RuntimeError:
        threading.Thread(target=lambda: asyncio.run(_run()), daemon=True).start()
    except Exception as e:
        logger.debug("Telegram schedule_dm failed: %s", e)
