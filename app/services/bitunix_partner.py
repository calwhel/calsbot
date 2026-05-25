"""
Bitunix Partner / Affiliate API wrapper.

This is COMPLETELY SEPARATE from the per-user trading executor in
`bitunix_executor.py`. The partner API uses the platform's master
affiliate credentials (one set of keys, owned by the platform, sourced
from https://partners.bitunix.com/#/statistics/dataOverview) to read
WHO has signed up to Bitunix under the platform's referral link.

Used as a gate so that only users whose `bitunix_uid` appears in the
master's affiliate roster can switch live trading on — i.e. "they have
to be under me to trade". Paper trading is unaffected.

Env vars:
  BITUNIX_PARTNER_API_KEY     — partner-dashboard API key (NOT a trading key)
  BITUNIX_PARTNER_API_SECRET  — partner-dashboard API secret
  BITUNIX_PARTNER_BASE_URL    — defaults to https://api.bitunix.com
  BITUNIX_PARTNER_LIST_PATH   — defaults to /api/v1/partner/invitee/list
  BITUNIX_AFFILIATE_CACHE_SEC — TTL for the cached UID set (default 600)
  BITUNIX_REQUIRE_AFFILIATE   — "1" to enforce the gate (default "1")

The actual REST path / response shape may vary between Bitunix partner
dashboard versions; the parser is defensive and pulls UIDs from the
most common JSON shapes ({data: {list: [{uid: "..."}]}},
{data: [...]}, {result: [...]}). Tune via env without redeploying.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp

logger = logging.getLogger(__name__)

PARTNER_API_KEY    = os.environ.get("BITUNIX_PARTNER_API_KEY", "").strip()
PARTNER_API_SECRET = os.environ.get("BITUNIX_PARTNER_API_SECRET", "").strip()
BASE_URL           = os.environ.get("BITUNIX_PARTNER_BASE_URL", "https://api.bitunix.com").rstrip("/")
LIST_PATH          = os.environ.get("BITUNIX_PARTNER_LIST_PATH", "/api/v1/partner/invitee/list")
CACHE_TTL_SEC      = int(os.environ.get("BITUNIX_AFFILIATE_CACHE_SEC", "600"))
REQUIRE_AFFILIATE  = os.environ.get("BITUNIX_REQUIRE_AFFILIATE", "1").lower() in ("1", "true", "yes")
HTTP_TIMEOUT       = int(os.environ.get("BITUNIX_PARTNER_HTTP_TIMEOUT", "10"))
PAGE_SIZE          = int(os.environ.get("BITUNIX_PARTNER_PAGE_SIZE", "200"))
MAX_PAGES          = int(os.environ.get("BITUNIX_PARTNER_MAX_PAGES", "50"))
REFERRAL_URL       = os.environ.get("BITUNIX_REFERRAL_URL", "https://www.bitunix.com/register?vipCode=tradehubsave")

# In-memory cache: (uid_set, fetched_at_epoch, total_count_or_None).
_cache: Tuple[Set[str], float, Optional[int]] = (set(), 0.0, None)
_cache_lock = asyncio.Lock()


class PartnerError(Exception):
    pass


def is_configured() -> bool:
    return bool(PARTNER_API_KEY and PARTNER_API_SECRET)


# ── Signing — same recipe as the trading API ───────────────────────────────
def _sign(api_key: str, api_secret: str, ts: str, nonce: str, body: str, query: str) -> str:
    pre = hashlib.sha256((nonce + ts + api_key + query + body).encode()).hexdigest()
    return hmac.new(api_secret.encode(), pre.encode(), hashlib.sha256).hexdigest()


def _headers(body: str, query: str = "") -> Dict[str, str]:
    ts    = str(int(time.time() * 1000))
    nonce = uuid.uuid4().hex
    return {
        "api-key":   PARTNER_API_KEY,
        "sign":      _sign(PARTNER_API_KEY, PARTNER_API_SECRET, ts, nonce, body, query),
        "nonce":     nonce,
        "timestamp": ts,
        "language":  "en-US",
        "Content-Type": "application/json",
    }


# ── UID extraction — defensive against schema drift ────────────────────────
_UID_KEYS = ("uid", "userId", "user_id", "inviteeUid", "invitee_uid", "id")

def _extract_uids(payload: Any) -> Set[str]:
    out: Set[str] = set()

    def walk(obj):
        if isinstance(obj, dict):
            for k in _UID_KEYS:
                v = obj.get(k)
                if v is not None and (isinstance(v, str) or isinstance(v, int)):
                    out.add(str(v))
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)

    walk(payload)
    return out


# ── Fetch one page ─────────────────────────────────────────────────────────
async def _fetch_page(page: int) -> Tuple[Set[str], int]:
    if not is_configured():
        raise PartnerError("partner_api_not_configured")
    query = f"page={page}&pageSize={PAGE_SIZE}"
    url   = BASE_URL + LIST_PATH + "?" + query
    hdrs  = _headers(body="", query=query)
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.get(url, headers=hdrs) as r:
            text = await r.text()
            if r.status != 200:
                raise PartnerError(f"HTTP {r.status}: {text[:200]}")
            try:
                body = json.loads(text)
            except Exception:
                raise PartnerError(f"non-JSON response: {text[:120]}")
            if isinstance(body, dict) and str(body.get("code", "0")) not in ("0", "200"):
                raise PartnerError(f"partner error {body.get('code')}: {body.get('msg','?')}")
            uids = _extract_uids(body)
            return uids, len(uids)


# ── Public: refresh + cached lookup ────────────────────────────────────────
async def refresh_affiliate_uids(force: bool = False) -> Set[str]:
    """Fetch every page of affiliated users; populate the in-memory cache."""
    global _cache
    async with _cache_lock:
        if not force:
            uids, fetched_at, _ = _cache
            if uids and (time.time() - fetched_at) < CACHE_TTL_SEC:
                return uids

        all_uids: Set[str] = set()
        for page in range(1, MAX_PAGES + 1):
            try:
                page_uids, n = await _fetch_page(page)
            except PartnerError:
                raise
            except Exception as e:
                raise PartnerError(f"page {page} fetch failed: {e}")
            new = page_uids - all_uids
            all_uids |= page_uids
            # Stop when a page returns nothing new (or fewer than a full page).
            if not new or n < PAGE_SIZE // 2:
                break

        _cache = (all_uids, time.time(), len(all_uids))
        logger.info(f"[BitunixPartner] cached {len(all_uids)} affiliate UIDs (ttl={CACHE_TTL_SEC}s)")
        return all_uids


def cached_uids() -> Set[str]:
    """Return the last-fetched UID set without triggering a network call."""
    return _cache[0]


def cache_age_sec() -> Optional[float]:
    if _cache[1] == 0:
        return None
    return time.time() - _cache[1]


async def is_uid_affiliated(bitunix_uid: Optional[str]) -> Tuple[bool, str]:
    """
    Returns (allowed, reason). If REQUIRE_AFFILIATE is False, always (True,'gate_off').
    On API failure we FAIL-CLOSED (return False) so a misconfigured partner
    key can never let unverified users trade live.
    """
    if not REQUIRE_AFFILIATE:
        return True, "gate_off"
    if not bitunix_uid:
        return False, "no_bitunix_uid_on_user"
    if not is_configured():
        return False, "partner_api_not_configured"
    try:
        uids = await refresh_affiliate_uids()
    except PartnerError as e:
        logger.warning(f"[BitunixPartner] affiliate check failed (fail-closed): {e}")
        # Fall back to the cache if we have something stale-but-non-empty.
        if _cache[0]:
            uids = _cache[0]
        else:
            return False, f"partner_api_error:{e}"
    return (str(bitunix_uid) in uids, "ok" if str(bitunix_uid) in uids else "uid_not_under_master")


def status() -> Dict[str, Any]:
    age = cache_age_sec()
    return {
        "configured":        is_configured(),
        "require_affiliate": REQUIRE_AFFILIATE,
        "referral_url":      REFERRAL_URL,
        "base_url":          BASE_URL,
        "list_path":         LIST_PATH,
        "cache_size":        len(_cache[0]),
        "cache_age_sec":     round(age, 1) if age is not None else None,
        "cache_ttl_sec":     CACHE_TTL_SEC,
        "guidance": (
            "Set BITUNIX_PARTNER_API_KEY + BITUNIX_PARTNER_API_SECRET (from "
            "partners.bitunix.com → API). Live trading will then require the "
            "user's bitunix_uid to appear in your affiliate roster."
            if not is_configured() else
            "Partner API ready. Live-trading users must have a bitunix_uid "
            "that matches an entry in your affiliate roster."
        ),
    }
