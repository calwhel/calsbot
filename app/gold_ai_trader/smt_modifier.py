"""SMT (Smart Money Technique) divergence — confidence modifier only.

Cross-symbol divergence between XAUUSD and a correlated cTrader series.
NOT a standalone entry trigger.

cTrader availability (FP Markets demo/live):
  - XAUUSD — primary gold feed (required)
  - XAGUSD — silver; available on same cTrader feed
  - DXY / USDX — NOT available on cTrader symbol map; do not use

Uses XAGUSD as the reference series (positive correlation with gold).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.gold_ai_trader.config import ASSET_CLASS, SYMBOL

logger = logging.getLogger(__name__)

# Only symbols confirmed on cTrader feed (see ctrader_price_feed._CANONICAL).
CTRADER_SMT_REFERENCES = {
    "XAGUSD": {"correlation": "positive", "label": "silver (XAGUSD)"},
}

DEFAULT_REFERENCE = "XAGUSD"


def _swing_extremes(
    klines: List[list],
    lookback: int,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return (recent_high, prior_high, recent_low, prior_low) over lookback window."""
    if not klines or len(klines) < lookback + 2:
        return None, None, None, None
    closed = klines[:-1][-lookback:]
    if len(closed) < 6:
        return None, None, None, None
    mid = len(closed) // 2
    first, second = closed[:mid], closed[mid:]
    try:
        hi1 = max(float(r[2]) for r in first)
        hi2 = max(float(r[2]) for r in second)
        lo1 = min(float(r[3]) for r in first)
        lo2 = min(float(r[3]) for r in second)
        return hi2, hi1, lo2, lo1
    except (TypeError, ValueError, IndexError):
        return None, None, None, None


async def assess_smt_divergence(
    *,
    direction: str,
    http_client,
    cache: Dict,
    reference_symbol: str = DEFAULT_REFERENCE,
    timeframe: str = "1h",
    lookback: int = 20,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Assess SMT divergence as a confidence modifier (-15 to +15).

    Bearish modifier for SHORT when gold makes HH but silver makes LH (unsupported rally).
    Bullish modifier for LONG when gold makes LL but silver makes HL (unsupported selloff).

    Returns dict with modifier, detail, data_available, reference_source.
    """
    ref = (reference_symbol or DEFAULT_REFERENCE).upper()
    meta = CTRADER_SMT_REFERENCES.get(ref)
    if not meta:
        return {
            "modifier": 0,
            "detail": f"SMT: reference {ref} not on cTrader allowlist",
            "data_available": False,
            "reference_symbol": ref,
            "reference_source": None,
        }

    from app.services.strategy_ta import _get_klines
    from app.services.tradfi_prices import get_metal_kline_source

    sym_cache = dict(cache or {})
    sym_cache["__asset_class__"] = ASSET_CLASS
    if user_id:
        sym_cache["__ctrader_user_id__"] = user_id

    try:
        k_gold = await _get_klines(
            SYMBOL, timeframe, lookback + 10, http_client, sym_cache,
        )
        k_ref = await _get_klines(
            ref, timeframe, lookback + 10, http_client, sym_cache,
        )
    except Exception as exc:
        logger.debug("[gold-ai] SMT kline fetch: %s", exc)
        return {
            "modifier": 0,
            "detail": f"SMT: kline fetch failed ({exc})",
            "data_available": False,
            "reference_symbol": ref,
            "reference_source": None,
        }

    ref_source = get_metal_kline_source(ref, timeframe, lookback + 10)
    ctrader_sources = {"ctrader", "ctrader-user", "ctrader-cache"}
    if not k_gold or not k_ref:
        return {
            "modifier": 0,
            "detail": "SMT: insufficient klines for gold or reference",
            "data_available": False,
            "reference_symbol": ref,
            "reference_source": ref_source,
        }
    if (ref_source or "").lower() not in ctrader_sources:
        return {
            "modifier": 0,
            "detail": f"SMT: reference klines not cTrader ({ref_source}) — skipped",
            "data_available": False,
            "reference_symbol": ref,
            "reference_source": ref_source,
        }

    g_hi2, g_hi1, g_lo2, g_lo1 = _swing_extremes(k_gold, lookback)
    r_hi2, r_hi1, r_lo2, r_lo1 = _swing_extremes(k_ref, lookback)
    if None in (g_hi2, g_hi1, g_lo2, g_lo1, r_hi2, r_hi1, r_lo2, r_lo1):
        return {
            "modifier": 0,
            "detail": "SMT: could not compute swing extremes",
            "data_available": False,
            "reference_symbol": ref,
            "reference_source": ref_source,
        }

    d = (direction or "").upper()
    modifier = 0
    detail_parts: List[str] = []

    # High-side divergence: gold HH, reference fails HH (LH)
    gold_hh = g_hi2 > g_hi1
    ref_hh = r_hi2 > r_hi1
    ref_fails_hh = r_hi2 <= r_hi1
    if gold_hh and ref_fails_hh:
        detail_parts.append(f"gold HH ({g_hi1:.2f}→{g_hi2:.2f}) vs {meta['label']} LH")
        if d == "SHORT":
            modifier += 10
        elif d == "LONG":
            modifier -= 10
    elif gold_hh and ref_hh and d == "LONG":
        detail_parts.append(
            f"XAG confirms gold HH ({g_hi1:.2f}→{g_hi2:.2f}; "
            f"{meta['label']} {r_hi1:.2f}→{r_hi2:.2f})"
        )
        modifier += 8

    # Low-side divergence: gold LL, reference fails LL (HL)
    gold_ll = g_lo2 < g_lo1
    ref_ll = r_lo2 < r_lo1
    ref_fails_ll = r_lo2 >= r_lo1
    if gold_ll and ref_fails_ll:
        detail_parts.append(f"gold LL ({g_lo1:.2f}→{g_lo2:.2f}) vs {meta['label']} HL")
        if d == "LONG":
            modifier += 10
        elif d == "SHORT":
            modifier -= 10
    elif gold_ll and ref_ll and d == "SHORT":
        detail_parts.append(
            f"XAG confirms gold LL ({g_lo1:.2f}→{g_lo2:.2f}; "
            f"{meta['label']} {r_lo1:.2f}→{r_lo2:.2f})"
        )
        modifier += 8

    if not detail_parts:
        detail = f"SMT: no divergence vs {meta['label']} ({timeframe})"
    elif any("confirms" in p for p in detail_parts):
        detail = "SMT confirmation: " + "; ".join(detail_parts)
    else:
        detail = "SMT divergence: " + "; ".join(detail_parts)

    modifier = max(-15, min(15, modifier))
    return {
        "modifier": modifier,
        "detail": detail,
        "data_available": True,
        "reference_symbol": ref,
        "reference_source": ref_source,
        "dxy_available": False,
        "dxy_note": "DXY/USDX not on cTrader — using XAGUSD (silver) as correlated reference",
    }
