"""Session gate + XAUUSD candidate scanner (deterministic, no LLM)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.gold_ai_trader.config import GoldAiRuntimeConfig, SYMBOL, ASSET_CLASS

logger = logging.getLogger(__name__)

_CANDIDATE_COOLDOWN_S = 300  # 5 min per setup key
_last_fired: Dict[str, float] = {}


@dataclass
class Candidate:
    type: str
    direction: str
    detail: str
    quality_atr: float
    sig_key: str
    raw: Dict[str, Any]


def active_session(now: datetime, cfg: GoldAiRuntimeConfig) -> Optional[str]:
    h = now.hour + now.minute / 60.0
    if cfg.london_start_hour <= h < cfg.london_end_hour:
        return "london"
    if cfg.ny_start_hour <= h < cfg.ny_end_hour:
        return "new_york"
    return None


def in_killzone(session: str) -> bool:
    return session in ("london", "new_york")


def _cooldown_ok(key: str) -> bool:
    now = time.monotonic()
    last = _last_fired.get(key, 0)
    if now - last < _CANDIDATE_COOLDOWN_S:
        return False
    _last_fired[key] = now
    return True


async def _price() -> Optional[float]:
    try:
        from app.services.tradfi_prices import get_price

        return await get_price(SYMBOL, ASSET_CLASS)
    except Exception as e:
        logger.debug("[gold-ai-trader] price fetch: %s", e)
        return None


async def scan_candidates(http: httpx.AsyncClient) -> Tuple[Optional[float], List[Candidate]]:
    """Return (price, candidates) — permissive ICT-style surfacing."""
    price = await _price()
    if not price:
        return None, []

    from app.services.strategy_ta import (
        eval_market_structure,
        eval_fvg,
        eval_fx_displacement,
        eval_forex_prev_level,
        eval_fx_sdp,
        eval_vwap_bias,
        eval_bt_klines_cond,
    )

    cache = {"__asset_class__": ASSET_CLASS}
    tf = "5m"
    found: List[Candidate] = []

    checks = [
        ("bos_bullish", "market_structure", {"condition": "bos_bullish", "timeframe": tf}, "LONG"),
        ("bos_bearish", "market_structure", {"condition": "bos_bearish", "timeframe": tf}, "SHORT"),
        ("choch_bullish", "market_structure", {"condition": "choch_bullish", "timeframe": tf}, "LONG"),
        ("choch_bearish", "market_structure", {"condition": "choch_bearish", "timeframe": tf}, "SHORT"),
        ("fvg_bull", "fvg", {"direction": "bullish", "timeframe": tf, "condition": "just_formed"}, "LONG"),
        ("fvg_bear", "fvg", {"direction": "bearish", "timeframe": tf, "condition": "just_formed"}, "SHORT"),
        ("disp_bull", "fx_displacement", {"direction": "bullish", "timeframe": tf}, "LONG"),
        ("disp_bear", "fx_displacement", {"direction": "bearish", "timeframe": tf}, "SHORT"),
        ("sweep_pdh", "forex_prev_level", {"condition": "sweep_pdh", "timeframe": tf}, "SHORT"),
        ("sweep_pdl", "forex_prev_level", {"condition": "sweep_pdl", "timeframe": tf}, "LONG"),
        ("liq_sweep_bull", "liquidity_sweep", {"type": "liquidity_sweep", "direction": "bullish", "timeframe": tf}, "LONG"),
        ("liq_sweep_bear", "liquidity_sweep", {"type": "liquidity_sweep", "direction": "bearish", "timeframe": tf}, "SHORT"),
        ("vwap_bull", "vwap_bias", {"condition": "above", "timeframe": tf}, "LONG"),
        ("vwap_bear", "vwap_bias", {"condition": "below", "timeframe": tf}, "SHORT"),
        ("sdp_bull", "fx_sdp", {"direction": "bullish", "timeframe": tf}, "LONG"),
        ("sdp_bear", "fx_sdp", {"direction": "bearish", "timeframe": tf}, "SHORT"),
    ]

    for ctype, kind, cond, direction in checks:
        key = f"{ctype}:{direction}"
        if not _cooldown_ok(key):
            continue
        try:
            if kind == "market_structure":
                ok, msg = await eval_market_structure(cond, SYMBOL, price, http, cache)
            elif kind == "fvg":
                ok, msg = await eval_fvg(cond, SYMBOL, price, http, cache)
            elif kind == "fx_displacement":
                ok, msg = await eval_fx_displacement(cond, SYMBOL, price, http, cache)
            elif kind == "forex_prev_level":
                ok, msg = await eval_forex_prev_level(cond, SYMBOL, price, http, cache)
            elif kind == "liquidity_sweep":
                ok, msg = await eval_bt_klines_cond(cond, SYMBOL, http, cache)
            elif kind == "vwap_bias":
                ok, msg = await eval_vwap_bias(cond, SYMBOL, price, http, cache)
            elif kind == "fx_sdp":
                ok, msg = await eval_fx_sdp(cond, SYMBOL, price, http, cache)
            else:
                continue
        except Exception as exc:
            logger.debug("[gold-ai-trader] scan %s: %s", ctype, exc)
            continue

        if ok:
            found.append(
                Candidate(
                    type=ctype,
                    direction=direction,
                    detail=str(msg)[:400],
                    quality_atr=1.0,
                    sig_key=key,
                    raw={"msg": msg, "price": price},
                )
            )

    return price, found


def pick_best(candidates: List[Candidate]) -> Optional[Candidate]:
    if not candidates:
        return None
    priority = {
        "sweep_pdh": 5,
        "sweep_pdl": 5,
        "sdp_bull": 4,
        "sdp_bear": 4,
        "disp_bull": 3,
        "disp_bear": 3,
        "fvg_bull": 2,
        "fvg_bear": 2,
    }
    return sorted(
        candidates,
        key=lambda c: priority.get(c.type, 1),
        reverse=True,
    )[0]
