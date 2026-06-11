"""HTTP geo-block circuit breaker — log once per domain, skip for 6h."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Set
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_BLOCK_HOURS = 6
_blocked_until: Dict[str, datetime] = {}
_warned_domains: Set[str] = set()
_provider_geo_blocked: Dict[str, datetime] = {}


def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc or "unknown"
    except Exception:
        return "unknown"


def is_domain_blocked(url: str) -> bool:
    domain = domain_from_url(url)
    until = _blocked_until.get(domain)
    return until is not None and datetime.utcnow() < until


def is_provider_geo_blocked(provider: str) -> bool:
    until = _provider_geo_blocked.get(provider)
    return until is not None and datetime.utcnow() < until


def note_geo_block(url: str, status: int, caller: str = "") -> None:
    """Record a 451/403 and log once per domain. Sets geo_blocked for 6h."""
    domain = domain_from_url(url)
    provider = caller.split(".")[0] if caller else domain
    until = datetime.utcnow() + timedelta(hours=_BLOCK_HOURS)
    _blocked_until[domain] = until
    _provider_geo_blocked[provider] = until

    if status == 451:
        if domain not in _warned_domains:
            _warned_domains.add(domain)
            logger.warning(f"[geo-block] 451 from {url} caller={caller or provider}")
        return

    if domain in _warned_domains:
        return
    _warned_domains.add(domain)
    logger.warning(
        "[geo-block] HTTP %s from %s url=%s caller=%s — skipping domain for %sh",
        status,
        domain,
        url,
        caller or "unknown",
        _BLOCK_HOURS,
    )
