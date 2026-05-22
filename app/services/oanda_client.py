"""
OANDA v20 REST client — forex broker integration for P5e.

Mirrors `app/services/bitunix_trader.py` for the forex stack. Wraps the
two endpoints we actually need for connection management (account
summary, open positions, balance) plus the order placement primitive
used by the executor in P5e-2.

Why OANDA: practice + live use the same endpoints with a different base
URL (controlled per-user via `users.oanda_environment`), the API speaks
units (mini-lot = 10,000) and pips natively, and a single bearer token
authenticates everything. No HMAC dance, no separate passphrase.

Auth: `Authorization: Bearer <api_key>`.
Practice base: https://api-fxpractice.oanda.com
Live base:     https://api-fxtrade.oanda.com

Every method returns `(ok: bool, payload: dict)` so callers can route on
the boolean without parsing exceptions. Failures include a `message`
field on the payload that is safe to surface to the user.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_BASE_PRACTICE = "https://api-fxpractice.oanda.com"
_BASE_LIVE     = "https://api-fxtrade.oanda.com"
_TIMEOUT_S = 8.0


def _base_url(environment: str) -> str:
    return _BASE_LIVE if (environment or "").lower() == "live" else _BASE_PRACTICE


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "Accept-Datetime-Format": "RFC3339",
    }


def _normalize_pair(symbol: str) -> str:
    """TradeHub stores forex as `EURUSD`; OANDA wants `EUR_USD`."""
    s = (symbol or "").upper().replace("/", "").replace("=X", "").replace("_", "")
    if len(s) == 6:
        return f"{s[:3]}_{s[3:]}"
    return s


async def get_account_summary(
    api_key: str, account_id: str, environment: str = "practice",
) -> Tuple[bool, Dict]:
    """Validate creds + return account snapshot (balance, currency, NAV)."""
    if not api_key or not account_id:
        return False, {"message": "Missing API key or account ID"}
    url = f"{_base_url(environment)}/v3/accounts/{account_id}/summary"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as c:
            r = await c.get(url, headers=_headers(api_key))
        if r.status_code == 401:
            return False, {"message": "OANDA rejected the API key (401)."}
        if r.status_code == 404:
            return False, {"message": "OANDA account ID not found for this key (404)."}
        if r.status_code >= 400:
            return False, {"message": f"OANDA error {r.status_code}: {r.text[:200]}"}
        data = r.json() or {}
        acct = data.get("account") or {}
        return True, {
            "id":              acct.get("id"),
            "alias":           acct.get("alias"),
            "currency":        acct.get("currency"),
            "balance":         float(acct.get("balance") or 0.0),
            "nav":             float(acct.get("NAV") or acct.get("nav") or 0.0),
            "unrealized_pnl":  float(acct.get("unrealizedPL") or 0.0),
            "open_position_count": int(acct.get("openPositionCount") or 0),
            "open_trade_count":    int(acct.get("openTradeCount") or 0),
            "margin_available":    float(acct.get("marginAvailable") or 0.0),
            "margin_rate":         float(acct.get("marginRate") or 0.0),
            "environment":         environment,
        }
    except httpx.HTTPError as e:
        return False, {"message": f"OANDA unreachable: {e}"}
    except Exception as e:
        logger.exception("[oanda] summary failed")
        return False, {"message": f"OANDA error: {e}"}


async def get_open_positions(
    api_key: str, account_id: str, environment: str = "practice",
) -> Tuple[bool, Dict]:
    """List currently open positions for the account."""
    url = f"{_base_url(environment)}/v3/accounts/{account_id}/openPositions"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as c:
            r = await c.get(url, headers=_headers(api_key))
        if r.status_code >= 400:
            return False, {"message": f"OANDA error {r.status_code}"}
        return True, {"positions": (r.json() or {}).get("positions", [])}
    except Exception as e:
        return False, {"message": str(e)}


async def list_accounts(api_key: str, environment: str = "practice") -> Tuple[bool, Dict]:
    """Used during the Connect flow to let users pick an account when their
    key has multiple sub-accounts (very common with OANDA practice keys)."""
    url = f"{_base_url(environment)}/v3/accounts"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as c:
            r = await c.get(url, headers=_headers(api_key))
        if r.status_code == 401:
            return False, {"message": "OANDA rejected the API key (401)."}
        if r.status_code >= 400:
            return False, {"message": f"OANDA error {r.status_code}: {r.text[:200]}"}
        data = r.json() or {}
        accounts = [{"id": a.get("id"), "tags": a.get("tags", [])} for a in data.get("accounts", [])]
        return True, {"accounts": accounts}
    except Exception as e:
        return False, {"message": str(e)}


async def place_market_order(
    api_key: str, account_id: str, symbol: str, units: int,
    tp_price: Optional[float] = None, sl_price: Optional[float] = None,
    environment: str = "practice",
) -> Tuple[bool, Dict]:
    """Place a market order. `units` positive = LONG, negative = SHORT.

    Used by P5e-2 — not yet wired into the executor. Included here so the
    full broker primitive lives in one file and P5e-2 is just config-side
    work to flip the executor branch.
    """
    pair = _normalize_pair(symbol)
    order: Dict = {
        "order": {
            "type":        "MARKET",
            "instrument":  pair,
            "units":       str(int(units)),
            "timeInForce": "FOK",
            "positionFill":"DEFAULT",
        }
    }
    if tp_price:
        order["order"]["takeProfitOnFill"] = {"price": f"{tp_price:.5f}", "timeInForce": "GTC"}
    if sl_price:
        order["order"]["stopLossOnFill"]   = {"price": f"{sl_price:.5f}", "timeInForce": "GTC"}
    url = f"{_base_url(environment)}/v3/accounts/{account_id}/orders"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as c:
            r = await c.post(url, headers=_headers(api_key), json=order)
        if r.status_code >= 400:
            return False, {"message": f"OANDA order rejected {r.status_code}: {r.text[:200]}"}
        return True, r.json() or {}
    except Exception as e:
        logger.exception("[oanda] order failed")
        return False, {"message": str(e)}
