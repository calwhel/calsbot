"""Shared HTTP GET wrapper — geo-block (451) logging and 6h provider skip."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

from app.services.geo_block import is_domain_blocked, note_geo_block


def _full_url(url: str, params: Optional[Dict[str, Any]]) -> str:
    if not params:
        return url
    return f"{url}?{urlencode(params)}"


async def provider_http_get(
    http_client,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    timeout_s: float = 10.0,
    caller: str = "unknown",
) -> Tuple[int, Optional[Any]]:
    """Lowest shared GET — records 451/403 geo-blocks once per domain."""
    full = _full_url(url, params)
    if is_domain_blocked(full):
        return 0, None
    try:
        resp = await http_client.get(url, params=params or {}, timeout=timeout_s)
        if resp.status_code in (451, 403):
            note_geo_block(full, resp.status_code, caller)
            return resp.status_code, None
        if resp.status_code == 200:
            try:
                return 200, resp.json()
            except Exception:
                return 200, None
        return resp.status_code, None
    except Exception:
        return 0, None
