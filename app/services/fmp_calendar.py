"""
Forex economic calendar — news-avoidance backbone.

Fetches the Forex Factory weekly calendar from their public JSON feed
(no API key required) and exposes one public helper, `is_news_blackout()`,
used by the `forex_news_avoidance` condition evaluator in strategy_ta.py.

Cache TTL is 10 minutes — events rarely change intra-day. The lookup
window spans the current week AND next week so the pre-event scanner can
see events that fall just past midnight on Sunday.

Falls open (returns "no blackout") whenever the upstream is unreachable —
strategies keep trading; they just lose the news filter rather than
blocking every signal.

Field mapping — Forex Factory → internal schema:
  country  → currency
  title    → event
  impact   → impact (lowercased: "High" → "high")
  date     → date (ISO-8601 with tz offset: "2026-05-26T21:30:00-04:00")
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_FF_URLS: List[str] = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
]
_FF_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.forexfactory.com/",
}
_CACHE_TTL_S = 600  # 10 min
_HTTP_TIMEOUT_S = 8.0

# (events_list, fetched_at_unix). None until first successful fetch.
_cache: Tuple[Optional[List[Dict]], float] = (None, 0.0)
_fetch_lock = asyncio.Lock()

_IMPACT_RANK = {"low": 1, "medium": 2, "high": 3}


def _normalize_ff_event(raw: Dict) -> Dict:
    """Translate Forex Factory field names → internal schema used by all
    downstream helpers.  Accepts both FF format (country/title/impact) and
    legacy FMP format (currency/event/impact) so the cache is format-agnostic.
    """
    return {
        "event":    raw.get("title") or raw.get("event") or "",
        "currency": (raw.get("country") or raw.get("currency") or "").upper(),
        "impact":   (raw.get("impact") or "").lower(),   # "High" → "high"
        "date":     raw.get("date") or "",
    }


def _parse_event_dt(ev: Dict) -> Optional[datetime]:
    """Parse the 'date' field from a normalized event.  Forex Factory uses
    ISO-8601 with a UTC-offset timezone ('2026-05-26T21:30:00-04:00'). Always
    returns a tz-aware UTC datetime so downstream comparisons can't trip
    naive-vs-aware TypeErrors."""
    raw = ev.get("date")
    if not raw:
        return None
    raw_s = str(raw).strip()
    dt: Optional[datetime] = None
    try:
        # ISO-8601 with offset (FF standard) or with 'Z' suffix
        dt = datetime.fromisoformat(raw_s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        try:
            # Legacy FMP "YYYY-MM-DD HH:MM:SS" (assumed UTC, no offset)
            dt = datetime.strptime(raw_s, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


async def _fetch_one(client: httpx.AsyncClient, url: str) -> List[Dict]:
    try:
        r = await client.get(url, headers=_FF_HEADERS)
        if r.status_code == 404:
            # FF only publishes next-week data near the week boundary — 404
            # is expected for most of the trading week, not an error.
            logger.debug("[fmp_calendar] %s → 404 (next-week not published yet)", url)
            return []
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            logger.warning("[fmp_calendar] unexpected response from %s: %r", url, type(data))
            return []
        return [_normalize_ff_event(ev) for ev in data]
    except Exception as e:
        logger.warning("[fmp_calendar] fetch failed for %s: %s", url, e)
        return []


async def _fetch_calendar() -> List[Dict]:
    """Fetch this week + next week from Forex Factory and merge."""
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
        results = await asyncio.gather(*[_fetch_one(client, u) for u in _FF_URLS])
    combined: List[Dict] = []
    for chunk in results:
        combined.extend(chunk)
    return combined


async def get_events() -> List[Dict]:
    """Cached calendar accessor. Refreshes every _CACHE_TTL_S seconds."""
    global _cache
    events, fetched_at = _cache
    now_ts = datetime.now(timezone.utc).timestamp()
    if events is not None and (now_ts - fetched_at) < _CACHE_TTL_S:
        return events
    async with _fetch_lock:
        events, fetched_at = _cache
        if events is not None and (now_ts - fetched_at) < _CACHE_TTL_S:
            return events
        fresh = await _fetch_calendar()
        # On fetch failure keep stale data rather than dropping to zero.
        if fresh or events is None:
            _cache = (fresh, now_ts)
            return fresh
        return events


_VALID_CCY = set(
    "USD EUR GBP JPY CHF AUD NZD CAD CNY HKD SGD SEK NOK DKK PLN ZAR MXN "
    "TRY INR KRW THB IDR PHP MYR RUB BRL ILS XAU XAG XPT XPD".split()
)


def currencies_from_pair(symbol: str) -> List[str]:
    """`EURUSD` → ['EUR','USD']. Robust to common broker/feed prefixes and
    suffixes: `OANDA:EUR_USD`, `FX:EURUSD`, `EUR/USD`, `EURUSD=X`, `EURUSD.r`,
    `eur-usd`, etc. USD-prefixed metal pairs (XAUUSD) return ['XAU','USD'] —
    XAU/silver have no calendar events so the filter naturally narrows to USD.

    Strategy: strip any prefix up to the last ':', drop punctuation, then scan
    every 6-char window for the first one whose two halves are valid ISO 4217
    currency codes. Falls back to first-3 / next-3 of the cleaned string.
    """
    if not symbol:
        return []
    s = str(symbol).upper()
    if ":" in s:
        s = s.rsplit(":", 1)[1]
    for sep in ("=", "."):
        if sep in s:
            s = s.split(sep, 1)[0]
    cleaned = "".join(ch for ch in s if ch.isalpha())
    if len(cleaned) < 6:
        return []
    for i in range(len(cleaned) - 5):
        a, b = cleaned[i:i+3], cleaned[i+3:i+6]
        if a in _VALID_CCY and b in _VALID_CCY:
            return [a, b]
    return []


async def is_news_blackout(
    symbol: str,
    *,
    minutes_before: int = 30,
    minutes_after: int = 30,
    min_impact: str = "high",
    now: Optional[datetime] = None,
) -> Tuple[bool, Optional[Dict]]:
    """True if there's a news event affecting either currency of `symbol`
    within `[now - minutes_before, now + minutes_after]` whose impact ≥
    `min_impact`. Returns (in_blackout, matched_event_or_none).

    Falls open (False, None) when:
      • the API call fails AND no stale cache is available
      • symbol doesn't look like a forex pair
    """
    ccys = currencies_from_pair(symbol)
    if not ccys:
        return False, None
    events = await get_events()
    if not events:
        return False, None

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    win_start = now - timedelta(minutes=minutes_before)
    win_end = now + timedelta(minutes=minutes_after)
    threshold = _IMPACT_RANK.get(min_impact.lower(), 3)
    ccys_set = {c.upper() for c in ccys}

    for ev in events:
        try:
            ev_ccy = (ev.get("currency") or "").upper()
            if ev_ccy not in ccys_set:
                continue
            rank = _IMPACT_RANK.get((ev.get("impact") or "").lower(), 0)
            if rank < threshold:
                continue
            ev_dt = _parse_event_dt(ev)
            if not ev_dt:
                continue
            if win_start <= ev_dt <= win_end:
                return True, {
                    "event": ev.get("event"),
                    "currency": ev_ccy,
                    "impact": ev.get("impact"),
                    "date": ev_dt.isoformat(),
                    "minutes_from_now": round((ev_dt - now).total_seconds() / 60.0, 1),
                }
        except Exception as e:
            logger.debug("[fmp_calendar] skipping malformed event: %s", e)
            continue
    return False, None


async def next_high_impact(
    symbol: str,
    *,
    min_impact: str = "high",
    horizon_h: int = 24,
    now: Optional[datetime] = None,
) -> Optional[Dict]:
    """Diagnostic helper — returns the next upcoming event affecting either
    currency in `symbol` within the horizon (or None). Used by the wizard's
    'next news' badge."""
    ccys = currencies_from_pair(symbol)
    if not ccys:
        return None
    events = await get_events()
    if not events:
        return None
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    horizon = now + timedelta(hours=horizon_h)
    threshold = _IMPACT_RANK.get(min_impact.lower(), 3)
    ccys_set = {c.upper() for c in ccys}
    best: Optional[Tuple[datetime, Dict]] = None
    for ev in events:
        ev_ccy = (ev.get("currency") or "").upper()
        if ev_ccy not in ccys_set:
            continue
        if _IMPACT_RANK.get((ev.get("impact") or "").lower(), 0) < threshold:
            continue
        ev_dt = _parse_event_dt(ev)
        if not ev_dt or ev_dt < now or ev_dt > horizon:
            continue
        if best is None or ev_dt < best[0]:
            best = (ev_dt, ev)
    if best is None:
        return None
    ev_dt, ev = best
    return {
        "event": ev.get("event"),
        "currency": (ev.get("currency") or "").upper(),
        "impact": ev.get("impact"),
        "date": ev_dt.isoformat(),
        "minutes_from_now": round((ev_dt - now).total_seconds() / 60.0, 1),
    }
