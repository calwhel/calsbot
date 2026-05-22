"""CFTC Commitment of Traders (COT) sentiment data for forex strategies.

Pulls the weekly Traders-in-Financial-Futures (TFF) report from CFTC's
public Socrata endpoint, indexes by currency, and exposes:

  - `get_cot_history(currency, weeks)` — last N weeks of positioning rows
  - `cot_sentiment(symbol, lookback_weeks)` — derives net-spec/comm series
    + percentile rank + week-over-week flip flags for the non-USD side of
    a USD pair (cross pairs return None — fail open at the evaluator).

CFTC data updates weekly on Fridays ~15:30 ET. We cache the response in
process for 6h to avoid hammering Socrata; cache hit/miss is shared
across windows since the whole dataset is one API call. Returns
`(data, ok)` tuples so evaluators can fail open on upstream outage.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import httpx

from app.services.fmp_calendar import currencies_from_pair

logger = logging.getLogger(__name__)

# Map our G8 currency codes to the CFTC TFF "market_and_exchange_names" prefix.
# USD is handled separately (DX-Y index — different endpoint) so for v1 we
# skip USD-direct readings and let pair logic infer USD bias from the
# non-USD leg.
_CCY_TO_MARKET: Dict[str, str] = {
    "EUR": "EURO FX",
    "GBP": "BRITISH POUND",
    "JPY": "JAPANESE YEN",
    "CHF": "SWISS FRANC",
    "AUD": "AUSTRALIAN DOLLAR",
    "NZD": "NEW ZEALAND DOLLAR",
    "CAD": "CANADIAN DOLLAR",
}

# Socrata TFF endpoint — public, no key required, ~rate-limited at 1000/h
# unanonymized. We pull the latest year of rows (~52 per currency) per
# refresh.
_TFF_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"

_CACHE: Dict[str, Tuple[List[dict], datetime, bool]] = {}  # key='all' → (rows, ts, ok)
_CACHE_TTL = timedelta(hours=6)
_LOCK = asyncio.Lock()


async def _fetch_tff_rows(limit: int = 1500) -> Tuple[List[dict], bool]:
    """Fetch raw TFF rows from CFTC Socrata. Returns (rows, ok)."""
    params = {
        "$limit": str(limit),
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$select": (
            "market_and_exchange_names,report_date_as_yyyy_mm_dd,"
            "dealer_positions_long_all,dealer_positions_short_all,"
            "lev_money_positions_long,lev_money_positions_short,"
            "asset_mgr_positions_long,asset_mgr_positions_short"
        ),
    }
    try:
        async with httpx.AsyncClient(timeout=15) as cx:
            r = await cx.get(_TFF_URL, params=params)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                return [], False
            return data, True
    except Exception as e:
        logger.warning(f"[cot] CFTC Socrata fetch failed: {e}")
        return [], False


async def get_all_rows() -> Tuple[List[dict], bool]:
    """Cached fetch of the latest ~year of TFF rows (all currencies in one
    pull). Returns (rows, ok) where ok=False on upstream outage so callers
    can fail open."""
    now = datetime.utcnow()
    cached = _CACHE.get("all")
    if cached and (now - cached[1]) < _CACHE_TTL:
        return cached[0], cached[2]

    async with _LOCK:
        cached = _CACHE.get("all")
        if cached and (datetime.utcnow() - cached[1]) < _CACHE_TTL:
            return cached[0], cached[2]

        rows, ok = await _fetch_tff_rows()
        ts = datetime.utcnow()
        _CACHE["all"] = (rows, ts, ok)
        return rows, ok


def _market_prefix_for(ccy: str) -> Optional[str]:
    return _CCY_TO_MARKET.get(ccy.upper())


def _filter_currency(rows: List[dict], ccy: str) -> List[dict]:
    """Latest-first list of rows matching this currency's CME futures."""
    prefix = _market_prefix_for(ccy)
    if not prefix:
        return []
    out = []
    for r in rows:
        name = (r.get("market_and_exchange_names") or "").upper()
        # Match "EURO FX - CHICAGO MERCANTILE EXCHANGE" but not the mini
        # contracts ("E-MICRO EURO/USD ...") which have different sizes.
        if name.startswith(prefix + " -") and "CHICAGO MERCANTILE" in name and "MICRO" not in name:
            out.append(r)
    return out


def _net_positions(row: dict) -> Tuple[float, float]:
    """Return (net_specs, net_commercials) for one weekly row.
    Specs = leveraged money (trend-followers). Commercials = dealers
    (hedgers, "smart money"). Both expressed as net long contracts."""
    def _f(k: str) -> float:
        try:
            return float(row.get(k) or 0)
        except (TypeError, ValueError):
            return 0.0
    net_specs = _f("lev_money_positions_long") - _f("lev_money_positions_short")
    net_comm  = _f("dealer_positions_long_all") - _f("dealer_positions_short_all")
    return net_specs, net_comm


def _percentile_rank(values: List[float], target: float) -> float:
    """Percentile rank of `target` within `values` (0–100). Returns 50
    when the series is degenerate."""
    if not values:
        return 50.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    below = sum(1 for v in sorted_v if v < target)
    equal = sum(1 for v in sorted_v if v == target)
    return (below + 0.5 * equal) / n * 100.0


async def cot_sentiment(
    symbol: str, lookback_weeks: int = 52,
) -> Optional[Tuple[Dict[str, float], bool]]:
    """Return ({…sentiment fields…}, ok) for the non-USD leg of a USD pair.
    Cross pairs (neither side USD, or both sides USD) return None — the
    evaluator interprets None as "filter inactive, pass open".

    Fields returned:
      ccy             — currency analyzed (the non-USD leg)
      invert_for_pair — True if USD is the BASE (e.g. USDJPY); evaluator
                        flips bullish/bearish meaning of the score
      net_specs       — latest net leveraged-money positioning (contracts)
      net_comm        — latest net dealer (commercial) positioning
      specs_pct       — percentile rank of net_specs in last N weeks
      comm_pct        — percentile rank of net_comm in last N weeks
      specs_flipped   — sign change in net_specs vs prior week (1/-1/0)
      comm_flipped    — sign change in net_comm vs prior week
      weeks_observed  — actual sample size
    """
    ccys = currencies_from_pair(symbol)
    if len(ccys) != 2:
        return None
    base, quote = ccys[0].upper(), ccys[1].upper()
    if base == "USD" and quote in _CCY_TO_MARKET:
        ccy = quote; invert = True   # USDJPY → JPY COT, invert for pair direction
    elif quote == "USD" and base in _CCY_TO_MARKET:
        ccy = base;  invert = False  # EURUSD → EUR COT, direct
    else:
        return None  # cross pair or unsupported

    rows, ok = await get_all_rows()
    if not ok or not rows:
        return ({"ccy": ccy, "invert_for_pair": invert, "weeks_observed": 0}, False)

    filtered = _filter_currency(rows, ccy)[:max(lookback_weeks, 4)]
    if len(filtered) < 2:
        return ({"ccy": ccy, "invert_for_pair": invert, "weeks_observed": len(filtered)}, True)

    series = [_net_positions(r) for r in filtered]  # newest first
    latest_specs, latest_comm = series[0]
    prev_specs, prev_comm = series[1]
    spec_hist = [s for s, _c in series]
    comm_hist = [c for _s, c in series]

    return ({
        "ccy": ccy,
        "invert_for_pair": invert,
        "net_specs": latest_specs,
        "net_comm":  latest_comm,
        "specs_pct": _percentile_rank(spec_hist, latest_specs),
        "comm_pct":  _percentile_rank(comm_hist, latest_comm),
        "specs_flipped": (
            1  if latest_specs > 0 >= prev_specs else
            -1 if latest_specs < 0 <= prev_specs else 0
        ),
        "comm_flipped": (
            1  if latest_comm > 0 >= prev_comm else
            -1 if latest_comm < 0 <= prev_comm else 0
        ),
        "weeks_observed": len(filtered),
    }, True)
