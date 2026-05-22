"""
Stock earnings calendar — Stock P6 earnings-avoidance backbone.

Backed by yfinance (already used by `tradfi_prices.py` for stock/forex
klines) since FMP's bulk earning_calendar endpoint is paid-tier only.
yfinance's `Ticker.get_earnings_dates()` returns the last + next few
earnings dates per symbol for free, which is exactly what the avoidance
filter needs.

We cache per-symbol with a 6h TTL — earnings dates don't move intraday
and a typical scan only hits a handful of tickers per strategy cycle.
yfinance is sync so we wrap each lookup in a thread executor; failures
fall open (the evaluator just stops filtering rather than blocking every
signal).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_CACHE_TTL_S = 6 * 3600
# Per-symbol cache: {SYM -> (event_dict_or_None, fetched_at_ts, ok)}.
_cache: Dict[str, Tuple[Optional[Dict], float, bool]] = {}
_locks: Dict[str, asyncio.Lock] = {}


def _normalize_symbol(sym: str) -> str:
    """Strip exchange prefix (`NASDAQ:AAPL` → `AAPL`) and uppercase."""
    s = (sym or "").upper().strip()
    if ":" in s:
        s = s.split(":", 1)[1]
    return s


def _fetch_sync(symbol: str) -> Tuple[Optional[Dict], bool]:
    """Blocking yfinance lookup. Returns (event_or_None, ok)."""
    try:
        import yfinance as yf
    except Exception as e:
        logger.warning(f"[earnings] yfinance unavailable: {e}")
        return None, False
    try:
        t = yf.Ticker(symbol)
        df = None
        try:
            df = t.get_earnings_dates(limit=12)
        except Exception:
            df = None
        if df is None or len(df) == 0:
            # Fall back to ticker.calendar (returns next event only).
            try:
                cal = t.calendar
                if cal is not None and not (hasattr(cal, "empty") and cal.empty):
                    # `calendar` shape varies — try dict access first.
                    raw = cal.get("Earnings Date") if isinstance(cal, dict) else None
                    if isinstance(raw, list) and raw:
                        d = raw[0]
                        dt = datetime(d.year, d.month, d.day, 21, 0, tzinfo=timezone.utc)
                        return ({"date": dt.date().isoformat(), "time": "AMC", "_dt_utc": dt}, True)
            except Exception:
                pass
            return None, True  # ok but no events on file
        now = datetime.now(timezone.utc)
        best: Optional[Tuple[timedelta, Dict]] = None
        for ix in df.index:
            try:
                py = ix.to_pydatetime() if hasattr(ix, "to_pydatetime") else ix
                if py.tzinfo is None:
                    py = py.replace(tzinfo=timezone.utc)
                else:
                    py = py.astimezone(timezone.utc)
            except Exception:
                continue
            delta = abs(py - now)
            if best is None or delta < best[0]:
                # Approximate BMO/AMC from US-Eastern hour-of-day.
                hr_et = (py.hour - 5) % 24
                t_tag = "BMO" if hr_et < 12 else "AMC"
                best = (delta, {
                    "date": py.date().isoformat(),
                    "time": t_tag,
                    "_dt_utc": py,
                })
        return (best[1] if best else None), True
    except Exception as e:
        logger.warning(f"[earnings] yfinance fetch failed for {symbol}: {e}")
        return None, False


async def next_earnings_event(symbol: str) -> Tuple[Optional[Dict], bool]:
    """Return (event_or_None, ok) for the nearest earnings event of `symbol`.
    Cached per symbol for 6h. `ok=False` means the lookup failed and the
    caller should fail open."""
    sym = _normalize_symbol(symbol)
    now_ts = datetime.now(timezone.utc).timestamp()
    cached = _cache.get(sym)
    if cached and (now_ts - cached[1]) < _CACHE_TTL_S:
        return cached[0], cached[2]

    lock = _locks.setdefault(sym, asyncio.Lock())
    async with lock:
        cached = _cache.get(sym)
        now_ts = datetime.now(timezone.utc).timestamp()
        if cached and (now_ts - cached[1]) < _CACHE_TTL_S:
            return cached[0], cached[2]
        loop = asyncio.get_event_loop()
        try:
            ev, ok = await asyncio.wait_for(
                loop.run_in_executor(None, _fetch_sync, sym), timeout=10.0,
            )
        except asyncio.TimeoutError:
            ev, ok = None, False
        # Only cache successful lookups — a one-off network blip shouldn't
        # poison the cache for 6h.
        if ok:
            _cache[sym] = (ev, now_ts, True)
        return ev, ok


async def is_earnings_blackout(
    symbol: str, days_before: int = 2, days_after: int = 1,
) -> Tuple[bool, Optional[Dict], bool]:
    """Returns (blocked, event_or_None, ok). `blocked=True` iff the next
    earnings event falls within [-days_after, +days_before] days from now.
    `ok=False` → caller fails open."""
    ev, ok = await next_earnings_event(symbol)
    if not ok:
        return False, None, False
    if not ev:
        return False, None, True
    dt: datetime = ev["_dt_utc"]
    now = datetime.now(timezone.utc)
    delta_days = (dt - now).total_seconds() / 86400.0
    in_window = (-float(days_after)) <= delta_days <= float(days_before)
    return in_window, ev, True
