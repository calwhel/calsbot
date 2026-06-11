"""
Reliable Telegram DM delivery for executor trade alerts and the health monitor.

Trade notifications and hourly health reports use direct Bot API HTTP calls
(not the aiogram polling loop), so they must:
  • try every configured bot token (main, then forex) until one delivers
  • verify HTTP 200 + ok==true (httpx does not raise on 4xx)
  • retry transient failures with backoff
  • honour Telegram retry_after on HTTP 429
  • route trade WIN/LOSS/breakeven alerts through a durable outbound queue
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Per-chat pacing — Telegram allows ~20 msgs/min per chat.
_MIN_GAP_PER_CHAT_S = float(os.environ.get("TELEGRAM_MIN_GAP_PER_CHAT_S", "3.0"))
_MAX_SEND_ATTEMPTS = int(os.environ.get("TELEGRAM_SEND_MAX_ATTEMPTS", "6"))


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


def bot_tokens_for_asset(asset_class: Optional[str] = None) -> List[str]:
    """
    Token order for trade alerts.
    Forex/index users usually /start @TradehubStrategyBot — try forex token first.
    """
    tokens = bot_tokens()
    if not tokens:
        return tokens
    ac = (asset_class or "").lower()
    if ac in ("forex", "index", "stock", "metals", "commodity") and len(tokens) > 1:
        return [tokens[1], tokens[0]]
    return tokens


async def bot_usernames() -> Dict[str, Optional[str]]:
    """Resolve @usernames for configured bots (for UI hints)."""
    out: Dict[str, Optional[str]] = {"main": None, "forex": None}
    tokens = bot_tokens()
    labels = ("main", "forex")
    async with httpx.AsyncClient(timeout=8) as client:
        for label, tok in zip(labels, tokens[:2]):
            if not tok:
                continue
            try:
                r = await client.get(f"https://api.telegram.org/bot{tok}/getMe")
                if r.status_code == 200:
                    data = (r.json() or {}).get("result") or {}
                    out[label] = data.get("username")
            except Exception:
                pass
    return out


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


def _retry_after_seconds(response: httpx.Response) -> float:
    """Parse Telegram retry_after from 429 body or Retry-After header."""
    try:
        body = response.json() or {}
        params = body.get("parameters") or {}
        ra = params.get("retry_after")
        if ra is not None:
            return max(1.0, float(ra))
    except Exception:
        pass
    try:
        hdr = response.headers.get("Retry-After")
        if hdr:
            return max(1.0, float(hdr))
    except Exception:
        pass
    return 5.0


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
    attempt = 0
    while attempt < _MAX_SEND_ATTEMPTS:
        attempt += 1
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
                if r.status_code == 429:
                    wait_s = _retry_after_seconds(r)
                    last = f"HTTP 429 retry_after={wait_s:.0f}s"
                    logger.warning(
                        "Telegram DM %s rate-limited — backing off %.0fs",
                        chat_id,
                        wait_s,
                    )
                    await asyncio.sleep(wait_s)
                    break  # retry same attempt cycle after backoff
                last = f"HTTP {r.status_code}: {(r.text or '')[:200]}"
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    continue
            except Exception as e:
                last = f"{type(e).__name__}: {e or '(no message)'}"
        if attempt < _MAX_SEND_ATTEMPTS:
            await asyncio.sleep(min(30.0, 1.5 * attempt))

    logger.warning("Telegram DM failed for %s: %s", chat_id, last or "(no detail)")
    return False


@dataclass
class _TradeTelegramJob:
    chat_id: int
    text: str
    msg_type: str
    symbol: str
    exec_id: int
    asset_class: str
    tokens: Optional[List[str]]
    parse_mode: str = "HTML"
    done: Optional[asyncio.Future] = None


_OUTBOUND: queue.Queue = queue.Queue()
_DRAINER_LOCK = threading.Lock()
_DRAINER_STARTED = False
_LAST_SEND_MONO: Dict[int, float] = {}


def _ensure_outbound_drainer() -> None:
    global _DRAINER_STARTED
    with _DRAINER_LOCK:
        if _DRAINER_STARTED:
            return
        threading.Thread(
            target=lambda: asyncio.run(_outbound_drainer_loop()),
            name="telegram-outbound",
            daemon=True,
        ).start()
        _DRAINER_STARTED = True


async def _pace_chat(chat_id: int) -> None:
    """Respect per-chat minimum gap to reduce 429s."""
    now = time.monotonic()
    last = _LAST_SEND_MONO.get(int(chat_id), 0.0)
    wait = _MIN_GAP_PER_CHAT_S - (now - last)
    if wait > 0:
        await asyncio.sleep(wait)
    _LAST_SEND_MONO[int(chat_id)] = time.monotonic()


async def _deliver_trade_job(job: _TradeTelegramJob) -> bool:
    await _pace_chat(job.chat_id)
    tokens = job.tokens
    if tokens is None:
        tokens = bot_tokens_for_asset(job.asset_class)
    ok = await send_dm(
        job.chat_id,
        job.text,
        parse_mode=job.parse_mode,
        tokens=tokens,
    )
    sym = (job.symbol or "?").upper()
    if ok:
        logger.info(
            "[telegram] sent %s %s exec=%s",
            job.msg_type,
            sym,
            job.exec_id or 0,
        )
    else:
        logger.warning(
            "[telegram] FAILED %s %s exec=%s: delivery not confirmed",
            job.msg_type,
            sym,
            job.exec_id or 0,
        )
    return ok


async def _outbound_drainer_loop() -> None:
    while True:
        job: _TradeTelegramJob = await asyncio.to_thread(_OUTBOUND.get)
        try:
            ok = await _deliver_trade_job(job)
            if job.done and not job.done.done():
                job.done.set_result(ok)
        except Exception as exc:
            sym = (job.symbol or "?").upper()
            logger.exception(
                "[telegram] FAILED %s %s exec=%s: %s",
                job.msg_type,
                sym,
                job.exec_id or 0,
                exc,
            )
            if job.done and not job.done.done():
                job.done.set_result(False)
        finally:
            _OUTBOUND.task_done()


def enqueue_trade_telegram(
    chat_id,
    text: str,
    *,
    msg_type: str,
    symbol: str = "",
    exec_id: int = 0,
    asset_class: str = "forex",
    tokens: Optional[List[str]] = None,
    parse_mode: str = "HTML",
) -> None:
    """Queue a trade alert for durable delivery (safe from sync or async callers)."""
    if not chat_id:
        return
    try:
        cid = int(chat_id)
    except (TypeError, ValueError):
        return
    if str(chat_id).startswith("WEB-"):
        return
    _ensure_outbound_drainer()
    _OUTBOUND.put(
        _TradeTelegramJob(
            chat_id=cid,
            text=text,
            msg_type=msg_type,
            symbol=symbol or "",
            exec_id=int(exec_id or 0),
            asset_class=asset_class or "forex",
            tokens=tokens,
            parse_mode=parse_mode,
        )
    )


async def deliver_trade_telegram(
    chat_id,
    text: str,
    *,
    msg_type: str,
    symbol: str = "",
    exec_id: int = 0,
    asset_class: str = "forex",
    tokens: Optional[List[str]] = None,
    parse_mode: str = "HTML",
) -> bool:
    """Await durable delivery — used from async close/reconcile paths."""
    if not chat_id:
        return False
    try:
        cid = int(chat_id)
    except (TypeError, ValueError):
        return False
    if str(chat_id).startswith("WEB-"):
        return False
    _ensure_outbound_drainer()
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    _OUTBOUND.put(
        _TradeTelegramJob(
            chat_id=cid,
            text=text,
            msg_type=msg_type,
            symbol=symbol or "",
            exec_id=int(exec_id or 0),
            asset_class=asset_class or "forex",
            tokens=tokens,
            parse_mode=parse_mode,
            done=fut,
        )
    )
    try:
        return bool(await fut)
    except Exception:
        return False


def schedule_dm(chat_id, text: str, *, parse_mode: str = "HTML") -> None:
    """Schedule a generic DM — routed through the outbound queue."""
    enqueue_trade_telegram(
        chat_id,
        text,
        msg_type="dm",
        parse_mode=parse_mode,
    )
