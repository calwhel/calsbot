"""
Bitunix live-execution module — sub-affiliate model.

Each user signs up to Bitunix via the platform's affiliate referral link
and pastes their OWN api_key + api_secret into Settings. Their trades
then count toward the master account's affiliate volume but the orders
themselves go through their personal sub-account — so the platform never
custodies funds, never holds master keys, and never trades on their behalf
without their explicit per-user keys.

This module is intentionally SAFE BY DEFAULT:
  • All `place_*` / `close_*` calls are no-ops that only LOG the intended
    order unless the env flag BITUNIX_LIVE_ENABLED=1 is set AND the user
    has both bitunix_api_key + bitunix_api_secret on file AND has opted
    in (User.auto_execute_live, when that field exists).
  • The actual REST signing block is feature-flagged so a misconfigured
    deploy can't accidentally place real orders.

Wiring it into the strategy executor is a separate, explicit step (next
patch). For now this just gives the rest of the codebase a typed surface
to call against, and a place to host the per-user paper-vs-live decision.
"""

import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

# ── Safety gates ────────────────────────────────────────────────────────────
LIVE_ENABLED   = os.environ.get("BITUNIX_LIVE_ENABLED", "").lower() in ("1", "true", "yes")
DRY_RUN_ALWAYS = os.environ.get("BITUNIX_DRY_RUN_ALWAYS", "").lower() in ("1", "true", "yes")
BASE_URL       = os.environ.get("BITUNIX_BASE_URL", "https://fapi.bitunix.com")
HTTP_TIMEOUT   = int(os.environ.get("BITUNIX_HTTP_TIMEOUT", "8"))


class BitunixError(Exception):
    """Any failure from the Bitunix REST API or our pre-flight checks."""


# ── HMAC signing — Bitunix Futures API spec ────────────────────────────────
def _nonce() -> str:
    return uuid.uuid4().hex


def _sign(api_key: str, api_secret: str, timestamp: str, nonce: str, body: str, query: str = "") -> str:
    """
    Bitunix futures signing recipe (per their public docs):
        digest = SHA256( nonce + timestamp + api_key + query + body )
        sign   = HMAC_SHA256( secret, digest_hex ).hex()
    """
    pre_hash = hashlib.sha256((nonce + timestamp + api_key + query + body).encode()).hexdigest()
    return hmac.new(api_secret.encode(), pre_hash.encode(), hashlib.sha256).hexdigest()


def _headers(api_key: str, api_secret: str, body: str, query: str = "") -> Dict[str, str]:
    ts    = str(int(time.time() * 1000))
    nonce = _nonce()
    return {
        "api-key":   api_key,
        "sign":      _sign(api_key, api_secret, ts, nonce, body, query),
        "nonce":     nonce,
        "timestamp": ts,
        "language":  "en-US",
        "Content-Type": "application/json",
    }


# ── Per-user gate ──────────────────────────────────────────────────────────
def can_user_trade_live(user) -> tuple[bool, str]:
    """
    Centralised "should we actually hit Bitunix for this user" check.

    Returns (allowed, reason). Used by the strategy executor and any
    'place test order' button in Settings to give the user an explicit
    explanation when paper mode is being enforced.
    """
    if DRY_RUN_ALWAYS:
        return False, "dry_run_env"
    if not LIVE_ENABLED:
        return False, "live_disabled_env"
    if not user:
        return False, "no_user"
    if not getattr(user, "bitunix_api_key", None) or not getattr(user, "bitunix_api_secret", None):
        return False, "no_api_key"
    # Optional explicit opt-in flag — added later as a Settings toggle.
    opt_in = getattr(user, "auto_execute_live", None)
    if opt_in is False:
        return False, "user_opted_out"
    return True, "ok"


# ── Order placement ───────────────────────────────────────────────────────
async def place_market_order(
    user,
    symbol: str,
    side: str,                    # "BUY" | "SELL"
    qty_contracts: float,
    leverage: int,
    tp_price: Optional[float] = None,
    sl_price: Optional[float] = None,
    margin_coin: str = "USDT",
    client_order_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Place a market order. SAFE BY DEFAULT: returns a 'dry_run' dict unless
    every gate in `can_user_trade_live` passes.

    Caller is responsible for converting strategy 'position_size_pct' +
    user equity into `qty_contracts` for the symbol. This module is just
    the broker layer.
    """
    allowed, reason = can_user_trade_live(user)
    intent = {
        "symbol":       symbol,
        "side":         side,
        "qty":          qty_contracts,
        "leverage":     leverage,
        "tp":           tp_price,
        "sl":           sl_price,
        "margin_coin":  margin_coin,
        "client_id":    client_order_id or _nonce(),
    }
    if not allowed:
        logger.info(f"[Bitunix] DRY-RUN ({reason}) → would place {intent}")
        return {"status": "dry_run", "reason": reason, "intent": intent}

    body_obj = {
        "symbol":        symbol,
        "side":          side.upper(),
        "orderType":     "MARKET",
        "qty":           str(qty_contracts),
        "tradeSide":     "OPEN",
        "marginCoin":    margin_coin,
        "leverage":      str(leverage),
        "clientId":      intent["client_id"],
    }
    if tp_price is not None:
        body_obj["tpPrice"]      = str(tp_price)
        body_obj["tpStopType"]   = "MARK_PRICE"
    if sl_price is not None:
        body_obj["slPrice"]      = str(sl_price)
        body_obj["slStopType"]   = "MARK_PRICE"

    body  = json.dumps(body_obj, separators=(",", ":"))
    path  = "/api/v1/futures/trade/place_order"
    url   = BASE_URL.rstrip("/") + path
    hdrs  = _headers(user.bitunix_api_key, user.bitunix_api_secret, body)

    try:
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(url, headers=hdrs, data=body) as r:
                resp_text = await r.text()
                if r.status != 200:
                    raise BitunixError(f"HTTP {r.status}: {resp_text[:200]}")
                resp = json.loads(resp_text)
                if str(resp.get("code", "0")) != "0":
                    raise BitunixError(f"Bitunix error: {resp.get('msg')} ({resp.get('code')})")
                logger.info(f"[Bitunix] LIVE order OK → {resp.get('data')}")
                return {"status": "ok", "intent": intent, "response": resp.get("data")}
    except BitunixError:
        raise
    except Exception as e:
        logger.error(f"[Bitunix] place_market_order failed: {e}")
        raise BitunixError(str(e))


async def close_position(user, symbol: str, side: str, qty_contracts: float) -> Dict[str, Any]:
    """Close (reduce-only) an open position. Same safety gates as place_market_order."""
    allowed, reason = can_user_trade_live(user)
    intent = {"symbol": symbol, "side": side, "qty": qty_contracts, "close": True}
    if not allowed:
        logger.info(f"[Bitunix] DRY-RUN ({reason}) → would close {intent}")
        return {"status": "dry_run", "reason": reason, "intent": intent}

    body_obj = {
        "symbol":     symbol,
        "side":       "SELL" if side.upper() == "BUY" else "BUY",
        "orderType":  "MARKET",
        "qty":        str(qty_contracts),
        "tradeSide":  "CLOSE",
        "reduceOnly": True,
    }
    body = json.dumps(body_obj, separators=(",", ":"))
    url  = BASE_URL.rstrip("/") + "/api/v1/futures/trade/place_order"
    hdrs = _headers(user.bitunix_api_key, user.bitunix_api_secret, body)
    try:
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(url, headers=hdrs, data=body) as r:
                resp_text = await r.text()
                if r.status != 200:
                    raise BitunixError(f"HTTP {r.status}: {resp_text[:200]}")
                resp = json.loads(resp_text)
                if str(resp.get("code", "0")) != "0":
                    raise BitunixError(f"Bitunix error: {resp.get('msg')} ({resp.get('code')})")
                return {"status": "ok", "intent": intent, "response": resp.get("data")}
    except BitunixError:
        raise
    except Exception as e:
        raise BitunixError(str(e))


def status() -> Dict[str, Any]:
    """Lightweight introspection for /api/admin/bitunix/status."""
    return {
        "live_enabled":    LIVE_ENABLED,
        "dry_run_always":  DRY_RUN_ALWAYS,
        "base_url":        BASE_URL,
        "http_timeout_s":  HTTP_TIMEOUT,
        "guidance": (
            "Live trading is OFF. To enable, set BITUNIX_LIVE_ENABLED=1 and ensure "
            "every trading user has both bitunix_api_key + bitunix_api_secret on file."
            if not LIVE_ENABLED else
            "Live trading is ON. Per-user gating still applies via can_user_trade_live()."
        ),
    }
