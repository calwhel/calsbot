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

# CFTC Legacy Futures-Only report — includes physical commodities (gold, silver).
# Fields use noncomm_* (speculators) and comm_* (commercials/hedgers).
_LEGACY_URL = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json"

_COMMODITY_TO_MARKET: Dict[str, str] = {
    "XAU": "GOLD",    # GOLD - COMMODITY EXCHANGE INC.
    "XAG": "SILVER",  # SILVER - COMMODITY EXCHANGE INC.
}

_CACHE: Dict[str, Tuple[List[dict], datetime, bool]] = {}  # key='all' → (rows, ts, ok)
_COMMODITY_CACHE: Dict[str, Tuple[List[dict], datetime, bool]] = {}  # key='XAU'/'XAG'
_CACHE_TTL = timedelta(hours=6)
_LOCK = asyncio.Lock()
_COMMODITY_LOCK = asyncio.Lock()


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
    """Return (net_specs, net_commercials) for one TFF weekly row.
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


def _net_positions_legacy(row: dict) -> Tuple[float, float]:
    """Return (net_specs, net_commercials) for one Legacy-report row.
    Non-commercial = large speculators/managed money (trend-followers).
    Commercial = hedgers/producers ("smart money")."""
    def _f(k: str) -> float:
        try:
            return float(row.get(k) or 0)
        except (TypeError, ValueError):
            return 0.0
    net_specs = _f("noncomm_positions_long_all") - _f("noncomm_positions_short_all")
    net_comm  = _f("comm_positions_long_all") - _f("comm_positions_short_all")
    return net_specs, net_comm


async def _fetch_legacy_commodity_rows(ccy: str, limit: int = 100) -> Tuple[List[dict], bool]:
    """Fetch rows for a physical commodity (XAU/XAG) from the CFTC Legacy
    Futures-Only report. Returns (rows newest-first, ok)."""
    prefix = _COMMODITY_TO_MARKET.get(ccy.upper())
    if not prefix:
        return [], False
    params = {
        "$limit": str(limit),
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$where": f"market_and_exchange_names LIKE '{prefix}%'",
        "$select": (
            "market_and_exchange_names,report_date_as_yyyy_mm_dd,"
            "noncomm_positions_long_all,noncomm_positions_short_all,"
            "comm_positions_long_all,comm_positions_short_all"
        ),
    }
    try:
        async with httpx.AsyncClient(timeout=15) as cx:
            r = await cx.get(_LEGACY_URL, params=params)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                return [], False
            # CFTC sometimes returns both the standard and mini contracts;
            # keep only rows whose market name starts with the exact prefix
            # and contains "COMMODITY EXCHANGE" (COMEX) to exclude mini/micro.
            filtered = [
                row for row in data
                if (row.get("market_and_exchange_names") or "").upper().startswith(prefix)
                and "COMMODITY EXCHANGE" in (row.get("market_and_exchange_names") or "").upper()
                and "MINI" not in (row.get("market_and_exchange_names") or "").upper()
            ]
            return filtered, True
    except Exception as e:
        logger.warning(f"[cot] CFTC legacy fetch failed ({ccy}): {e}")
        return [], False


async def get_commodity_rows(ccy: str) -> Tuple[List[dict], bool]:
    """Cached 6-hour accessor for commodity COT rows (XAU / XAG)."""
    key = ccy.upper()
    now = datetime.utcnow()
    cached = _COMMODITY_CACHE.get(key)
    if cached and (now - cached[1]) < _CACHE_TTL:
        return cached[0], cached[2]
    async with _COMMODITY_LOCK:
        cached = _COMMODITY_CACHE.get(key)
        if cached and (datetime.utcnow() - cached[1]) < _CACHE_TTL:
            return cached[0], cached[2]
        rows, ok = await _fetch_legacy_commodity_rows(key)
        ts = datetime.utcnow()
        _COMMODITY_CACHE[key] = (rows, ts, ok)
        return rows, ok


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


def _build_sentiment_result(
    ccy: str,
    invert: bool,
    filtered: List[dict],
    net_fn,  # callable: row → (net_specs, net_comm)
) -> Tuple[Dict, bool]:
    """Shared result-building logic for TFF and Legacy (commodity) rows."""
    if len(filtered) < 2:
        return ({"ccy": ccy, "invert_for_pair": invert, "weeks_observed": len(filtered)}, True)
    series = [net_fn(r) for r in filtered]
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


async def cot_sentiment(
    symbol: str, lookback_weeks: int = 52,
) -> Optional[Tuple[Dict[str, float], bool]]:
    """Return ({…sentiment fields…}, ok) for the relevant leg of a pair.

    Forex USD pairs → TFF report (leveraged-money vs dealers).
    Metal pairs (XAUUSD, XAGUSD) → CFTC Legacy Futures report (non-commercial
    speculators vs commercial hedgers/producers from COMEX).
    Cross pairs / unsupported → None (evaluator treats as filter inactive).

    Fields returned:
      ccy             — currency/commodity analyzed
      invert_for_pair — True if USD is the BASE (e.g. USDJPY); evaluator
                        flips bullish/bearish meaning of the score
      net_specs       — latest net speculator positioning (contracts)
      net_comm        — latest net commercial positioning (contracts)
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

    # ── Metal (XAUUSD / XAGUSD) — CFTC Legacy Futures-Only report ─────────
    if base in _COMMODITY_TO_MARKET and quote == "USD":
        ccy = base; invert = False
        rows, ok = await get_commodity_rows(ccy)
        if not ok or not rows:
            return ({"ccy": ccy, "invert_for_pair": invert, "weeks_observed": 0}, False)
        subset = rows[:max(lookback_weeks, 4)]
        return _build_sentiment_result(ccy, invert, subset, _net_positions_legacy)

    if quote in _COMMODITY_TO_MARKET and base == "USD":
        ccy = quote; invert = True   # USD/XAU — uncommon quoting
        rows, ok = await get_commodity_rows(ccy)
        if not ok or not rows:
            return ({"ccy": ccy, "invert_for_pair": invert, "weeks_observed": 0}, False)
        subset = rows[:max(lookback_weeks, 4)]
        return _build_sentiment_result(ccy, invert, subset, _net_positions_legacy)

    # ── Forex USD pairs — TFF report ──────────────────────────────────────
    if base == "USD" and quote in _CCY_TO_MARKET:
        ccy = quote; invert = True
    elif quote == "USD" and base in _CCY_TO_MARKET:
        ccy = base;  invert = False
    else:
        return None  # cross pair or unsupported

    rows, ok = await get_all_rows()
    if not ok or not rows:
        return ({"ccy": ccy, "invert_for_pair": invert, "weeks_observed": 0}, False)

    filtered = _filter_currency(rows, ccy)[:max(lookback_weeks, 4)]
    return _build_sentiment_result(ccy, invert, filtered, _net_positions)
