"""HTTP geo-block circuit breaker — log once per domain, skip for 6h."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_BLOCK_HOURS = 6
_blocked_until: Dict[str, datetime] = {}
_warned_domains: set[str] = set()


def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc or "unknown"
    except Exception:
        return "unknown"


def is_domain_blocked(url: str) -> bool:
    domain = domain_from_url(url)
    until = _blocked_until.get(domain)
    return until is not None and datetime.utcnow() < until


def note_geo_block(url: str, status: int, caller: str = "") -> None:
    """Record a 451/403 and log the URL + caller once per domain."""
    domain = domain_from_url(url)
    _blocked_until[domain] = datetime.utcnow() + timedelta(hours=_BLOCK_HOURS)
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
