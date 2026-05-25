"""
Currency strength index — aggregates % change across the 28-pair matrix of
8 major currencies (USD, EUR, GBP, JPY, CHF, AUD, NZD, CAD) into a per-
currency strength score, then exposes a differential between any pair's
base and quote.

A POSITIVE differential means the base is stronger than the quote (favor
LONG); a NEGATIVE differential means the quote is stronger (favor SHORT).

Backed by yfinance via tradfi_prices.get_klines — falls open (returns
neutral / score 0) on any upstream failure so strategies keep firing.

Window: configurable 1h / 4h / 1d. Cache: 5 min keyed on window.

Metals (XAU, XAG) vs USD:
  The 28-pair G8 matrix is forex-only. For gold/silver pairs the
  strength differential is computed directly from the metal's own %
  change over the window (XAUUSD up = XAU strong vs USD = positive
  differential, favoring LONG).  A separate per-metal cache (also
  5 min) isolates this from the G8 matrix refresh cycle.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from app.services.fmp_calendar import currencies_from_pair
from app.services.tradfi_prices import get_klines

logger = logging.getLogger(__name__)

MAJORS: Tuple[str, ...] = ("USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD")

# All 28 unique cross-pairs of the 8 majors, with the canonical FX side
# (e.g. EURUSD not USDEUR — quoting follows market convention).
_PAIRS: List[str] = [
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD",
    "USDJPY", "USDCHF", "USDCAD",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CADCHF",
    "CHFJPY",
]

# Map each declared window to the yfinance bar interval AND the number of
# bars back to compute % change against — so "1h" is genuinely a 1h delta
# (last close vs close 1 bar ago on the 1h series), "4h" is a 4h delta on
# the 1h series, and "1d" is a 1d delta on the daily series. We pulled a
# small `_BARS_BACK` separate from `_PAIRS_BARS_FETCH` so we always
# request enough history without changing the math.
_WINDOW_TF: Dict[str, str] = {"1h": "1h", "4h": "1h", "1d": "1d"}
_BARS_BACK: Dict[str, int] = {"1h": 1, "4h": 4, "1d": 1}

_CACHE: Dict[str, Tuple[Dict[str, float], datetime, bool]] = {}  # (scores, ts, ok)
_CACHE_TTL = timedelta(minutes=5)
_LOCK = asyncio.Lock()

# Metals supported as a special-case in pair_strength_diff (not in G8 matrix).
# yfinance tickers for XAUUSD and XAGUSD.
_METAL_TICKER: Dict[str, str] = {
    "XAU": "XAUUSD",   # resolves to GC=F (gold futures) via tradfi_prices
    "XAG": "XAGUSD",   # resolves to SI=F (silver futures) via tradfi_prices
}
# Separate cache keyed by "metal_<XAU|XAG>_<window>"
_METAL_CACHE: Dict[str, Tuple[float, datetime, bool]] = {}  # (pct_change, ts, ok)


async def _pair_pct_change(pair: str, tf: str, bars_back: int) -> Optional[float]:
    """Return % change: latest close vs close `bars_back` bars ago."""
    try:
        kl = await get_klines(pair, "forex", timeframe=tf, limit=max(bars_back + 5, 10))
        if not kl or len(kl) < bars_back + 1:
            return None
        old = float(kl[-(bars_back + 1)][4])
        new = float(kl[-1][4])
        if old <= 0:
            return None
        return (new - old) / old * 100.0
    except Exception as e:
        logger.debug(f"[ccy_strength] pair fetch failed {pair}: {e}")
        return None


async def get_strength_scores(window: str = "4h") -> Tuple[Dict[str, float], bool]:
    """Return ({currency: strength_score}, ok) where ok=False means yfinance
    was fully unreachable and the filter should fail OPEN (callers should
    pass-through, not block). Scores are mean signed-% across each
    currency's appearances in the 28-pair matrix (positive = appreciating
    against the basket). Cached per-window for 5 minutes — both success
    and miss are cached so we don't hammer yfinance during an outage."""
    win = window if window in _WINDOW_TF else "4h"
    now = datetime.utcnow()
    cached = _CACHE.get(win)
    if cached and (now - cached[1]) < _CACHE_TTL:
        return cached[0], cached[2]

    async with _LOCK:
        # Re-check inside the lock in case another coro just refreshed.
        cached = _CACHE.get(win)
        if cached and (datetime.utcnow() - cached[1]) < _CACHE_TTL:
            return cached[0], cached[2]

        tf = _WINDOW_TF[win]
        bars_back = _BARS_BACK[win]
        results = await asyncio.gather(
            *[_pair_pct_change(p, tf, bars_back) for p in _PAIRS],
            return_exceptions=False,
        )

        accum: Dict[str, List[float]] = {c: [] for c in MAJORS}
        any_data = False
        for pair, pct in zip(_PAIRS, results):
            if pct is None:
                continue
            any_data = True
            base, quote = pair[:3], pair[3:]
            if base in accum:
                accum[base].append(pct)        # base up = base strong
            if quote in accum:
                accum[quote].append(-pct)      # base up = quote weak

        ts = datetime.utcnow()
        if not any_data:
            zeros = {c: 0.0 for c in MAJORS}
            _CACHE[win] = (zeros, ts, False)
            return zeros, False

        scores = {c: (sum(v) / len(v) if v else 0.0) for c, v in accum.items()}
        _CACHE[win] = (scores, ts, True)
        return scores, True


async def _metal_strength_diff(
    metal: str, window: str,
) -> Tuple[float, bool]:
    """Return (pct_change_over_window, ok) for a metal vs USD.
    Positive = metal appreciating (XAU strong). Cached 5 min per metal+window."""
    cache_key = f"metal_{metal}_{window}"
    now = datetime.utcnow()
    cached = _METAL_CACHE.get(cache_key)
    if cached and (now - cached[1]) < _CACHE_TTL:
        return cached[0], cached[2]

    ticker_sym = _METAL_TICKER.get(metal)
    if not ticker_sym:
        return 0.0, False

    win = window if window in _WINDOW_TF else "4h"
    tf = _WINDOW_TF[win]
    bars_back = _BARS_BACK[win]
    pct = await _pair_pct_change(ticker_sym, tf, bars_back)
    ok = pct is not None
    val = pct if pct is not None else 0.0
    _METAL_CACHE[cache_key] = (val, datetime.utcnow(), ok)
    return val, ok


async def pair_strength_diff(
    symbol: str, window: str = "4h",
) -> Optional[Tuple[float, str, str, Dict[str, float], bool]]:
    """Return (base_strength - quote_strength, base, quote, all_scores, ok).
    `ok` mirrors the upstream availability flag — when False the caller
    should treat the filter as inactive (fail-open). Returns None when the
    pair can't be split into majors or supported metals.

    Metal pairs (XAUUSD, XAGUSD):
      The G8 matrix doesn't include metals. Instead the differential is the
      metal's own % change over the window vs USD — positive means the metal
      is strengthening (favors LONG on the metal), negative means USD is
      strengthening (favors SHORT on the metal). The returned `all_scores`
      dict includes the metal score alongside the G8 USD score.
    """
    ccys = currencies_from_pair(symbol)
    if len(ccys) != 2:
        return None
    base, quote = ccys[0], ccys[1]

    # ── Metal (XAU / XAG) vs USD path ─────────────────────────────────────
    if base in _METAL_TICKER and quote == "USD":
        diff, ok = await _metal_strength_diff(base, window)
        scores_g8, _ = await get_strength_scores(window)
        pseudo_scores = {**scores_g8, base: diff}
        return diff, base, quote, pseudo_scores, ok

    if quote in _METAL_TICKER and base == "USD":
        # USD/XAU quoting convention — inverted
        diff, ok = await _metal_strength_diff(quote, window)
        scores_g8, _ = await get_strength_scores(window)
        pseudo_scores = {**scores_g8, quote: diff}
        return -diff, base, quote, pseudo_scores, ok

    # ── G8 major-pair path ────────────────────────────────────────────────
    if base not in MAJORS or quote not in MAJORS:
        return None
    scores, ok = await get_strength_scores(window)
    diff = scores.get(base, 0.0) - scores.get(quote, 0.0)
    return diff, base, quote, scores, ok
