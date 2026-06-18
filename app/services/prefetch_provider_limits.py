"""Per-provider concurrency caps and 429 handling during executor prefetch."""
from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_PROVIDER_LIMITS: Dict[str, int] = {
    "ctrader": int(os.environ.get("PREFETCH_CTRADER_CONCURRENT", "16")),
    "kraken": int(os.environ.get("PREFETCH_KRAKEN_CONCURRENT", "3")),
    "yahoo": int(os.environ.get("PREFETCH_YAHOO_CONCURRENT", "3")),
    "fmp": int(os.environ.get("PREFETCH_FMP_CONCURRENT", "3")),
    "coinbase": int(os.environ.get("PREFETCH_COINBASE_CONCURRENT", "3")),
    "crypto": int(os.environ.get("PREFETCH_CRYPTO_CONCURRENT", "8")),
    "external": int(os.environ.get("PREFETCH_EXTERNAL_CONCURRENT", "4")),
}

_sems: Dict[str, asyncio.Semaphore] = {}
_prefetch_429_provider: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "prefetch_429_provider", default=None,
)


class PrefetchSlotUnavailable(RuntimeError):
    """Provider slot was not available within caller's remaining budget."""


def is_rate_limit_http(status_code: int) -> bool:
    return status_code in (429, 418)


def prefetch_429_backoff_s(attempt: int) -> float:
    return min(2.0, 0.35 * (2 ** attempt))


def _provider_sem(provider: str) -> asyncio.Semaphore:
    key = provider if provider in _PROVIDER_LIMITS else "external"
    if key not in _sems:
        _sems[key] = asyncio.Semaphore(max(1, _PROVIDER_LIMITS[key]))
    return _sems[key]


def prefetch_provider_bucket(symbol: str, asset_class: str) -> str:
    """Map a prefetch job to a provider bucket for concurrency limits."""
    sym = (symbol or "").upper().replace("/", "").replace("-", "")
    ac = (asset_class or "crypto").lower()
    if ac == "crypto":
        return "crypto"
    try:
        from app.services.tradfi_prices import is_metal_symbol
        if is_metal_symbol(sym):
            return "kraken"
    except Exception:
        pass
    if ac == "index":
        return "yahoo"
    if ac in ("forex", "stock", "metals", "commodity"):
        return "ctrader"
    return "external"


def note_prefetch_429(provider: str) -> None:
    _prefetch_429_provider.set(provider)


def consume_prefetch_429() -> Optional[str]:
    return _prefetch_429_provider.get()


def clear_prefetch_429() -> None:
    _prefetch_429_provider.set(None)


async def prefetch_http_get(
    provider: str,
    client,
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    max_attempts: int = 3,
    max_slot_wait_s: Optional[float] = None,
):
    """GET with per-provider slot and brief 429 backoff during prefetch."""
    try:
        from app.services.prefetch_fast import prefetch_fast_active
        fast = prefetch_fast_active()
    except Exception:
        fast = False

    last_resp = None
    attempts = max_attempts if fast else 1
    for attempt in range(attempts):
        async with prefetch_provider_slot(
            provider,
            max_wait_s=max_slot_wait_s,
        ):
            kwargs: dict = {}
            if params is not None:
                kwargs["params"] = params
            if headers is not None:
                kwargs["headers"] = headers
            last_resp = await client.get(url, **kwargs)
            if not is_rate_limit_http(last_resp.status_code):
                return last_resp
            note_prefetch_429(provider)
            if attempt < attempts - 1:
                await asyncio.sleep(prefetch_429_backoff_s(attempt))
    return last_resp


@asynccontextmanager
async def prefetch_provider_slot(provider: str, *, max_wait_s: Optional[float] = None):
    """Limit concurrent outbound calls per external provider during prefetch."""
    try:
        from app.services.prefetch_fast import prefetch_fast_active
        if not prefetch_fast_active():
            yield
            return
    except Exception:
        yield
        return

    sem = _provider_sem(provider)
    t0 = time.monotonic()
    try:
        if max_wait_s is not None:
            wait_s = max(0.0, float(max_wait_s))
            if wait_s <= 0.0:
                raise asyncio.TimeoutError()
            await asyncio.wait_for(sem.acquire(), timeout=wait_s)
        else:
            await sem.acquire()
    except asyncio.TimeoutError as exc:
        wait_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            "[prefetch-limit] provider=%s slot_wait_timeout=%.0fms budget=%.0fms",
            provider,
            wait_ms,
            max(0.0, float(max_wait_s or 0.0)) * 1000.0,
        )
        raise PrefetchSlotUnavailable(
            f"{provider} slot unavailable within budget"
        ) from exc
    wait_ms = (time.monotonic() - t0) * 1000.0
    if wait_ms > 250:
        logger.info(
            "[prefetch-limit] provider=%s slot_wait=%.0fms",
            provider, wait_ms,
        )
    try:
        yield
    finally:
        sem.release()
