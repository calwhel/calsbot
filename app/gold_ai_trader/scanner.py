"""Session gate + XAUUSD candidate scanner (deterministic, no LLM)."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.gold_ai_trader.call_gates import (
    atr_from_klines,
    passes_quality_gates,
)
from app.gold_ai_trader.config import GoldAiRuntimeConfig, SYMBOL, ASSET_CLASS

logger = logging.getLogger(__name__)


def _setup_cooldown_s() -> int:
    try:
        return max(300, int(os.environ.get("GOLD_AI_TRADER_SETUP_COOLDOWN_S", "900")))
    except (TypeError, ValueError):
        return 900


# Per setup-key last time Claude was invoked (not merely TA-detected).
_last_claude_fired: Dict[str, float] = {}


@dataclass
class Candidate:
    type: str
    direction: str
    detail: str
    quality_atr: float
    sig_key: str
    raw: Dict[str, Any]


def active_session(now: datetime, cfg: GoldAiRuntimeConfig) -> Optional[str]:
    from app.services.forex_sessions import is_named_session_active

    if is_named_session_active("london", now):
        return "london"
    if is_named_session_active("new_york", now):
        return "new_york"
    return None


def in_killzone(session: str) -> bool:
    return session in ("london", "new_york")


def setup_cooldown_elapsed(key: str) -> bool:
    """True if this setup type is off cooldown since last Claude call."""
    now = time.monotonic()
    last = _last_claude_fired.get(key, 0)
    return (now - last) >= _setup_cooldown_s()


def record_claude_invocation(candidate: Candidate) -> None:
    """Call after Claude runs — enforces per-setup cooldown."""
    _last_claude_fired[candidate.sig_key] = time.monotonic()


async def _price() -> Optional[float]:
    try:
        from app.services.tradfi_prices import get_price

        return await get_price(SYMBOL, ASSET_CLASS)
    except Exception as e:
        logger.debug("[gold-ai-trader] price fetch: %s", e)
        return None


async def scan_candidates(
    http: httpx.AsyncClient,
    *,
    session: str,
    cfg: GoldAiRuntimeConfig,
) -> Tuple[Optional[float], List[Candidate]]:
    """Return (price, gated candidates). BOS/VWAP/CHoCH excluded — too marginal."""
    price = await _price()
    if not price:
        return None, []

    from app.services.tradfi_prices import get_klines
    from app.services.strategy_ta import (
        eval_fvg,
        eval_fx_displacement,
        eval_forex_prev_level,
        eval_fx_sdp,
        eval_bt_klines_cond,
    )

    k5 = await get_klines(SYMBOL, ASSET_CLASS, "5m", 60) or []
    k1h = await get_klines(SYMBOL, ASSET_CLASS, "1h", 50) or []
    k_daily = await get_klines(SYMBOL, ASSET_CLASS, "1d", 5) or []
    now = datetime.utcnow()

    cache = {"__asset_class__": ASSET_CLASS}
    tf = "5m"
    found: List[Candidate] = []

    # High-quality setup types only (no raw BOS/CHoCH/VWAP wake-ups).
    checks = [
        ("sweep_pdh", "forex_prev_level", {"condition": "sweep_pdh", "timeframe": tf}, "SHORT"),
        ("sweep_pdl", "forex_prev_level", {"condition": "sweep_pdl", "timeframe": tf}, "LONG"),
        ("disp_bull", "fx_displacement", {"direction": "bullish", "timeframe": tf}, "LONG"),
        ("disp_bear", "fx_displacement", {"direction": "bearish", "timeframe": tf}, "SHORT"),
        ("fvg_bull", "fvg", {"direction": "bullish", "timeframe": tf, "condition": "just_formed"}, "LONG"),
        ("fvg_bear", "fvg", {"direction": "bearish", "timeframe": tf, "condition": "just_formed"}, "SHORT"),
        ("liq_sweep_bull", "liquidity_sweep", {"type": "liquidity_sweep", "direction": "bullish", "timeframe": tf}, "LONG"),
        ("liq_sweep_bear", "liquidity_sweep", {"type": "liquidity_sweep", "direction": "bearish", "timeframe": tf}, "SHORT"),
        ("sdp_bull", "fx_sdp", {"direction": "bullish", "timeframe": tf}, "LONG"),
        ("sdp_bear", "fx_sdp", {"direction": "bearish", "timeframe": tf}, "SHORT"),
    ]

    atr = atr_from_klines(k5)

    for ctype, kind, cond, direction in checks:
        key = f"{ctype}:{direction}"
        if not setup_cooldown_elapsed(key):
            continue
        try:
            if kind == "fvg":
                ok, msg = await eval_fvg(cond, SYMBOL, price, http, cache)
            elif kind == "fx_displacement":
                ok, msg = await eval_fx_displacement(cond, SYMBOL, price, http, cache)
            elif kind == "forex_prev_level":
                ok, msg = await eval_forex_prev_level(cond, SYMBOL, price, http, cache)
            elif kind == "liquidity_sweep":
                ok, msg = await eval_bt_klines_cond(cond, SYMBOL, http, cache)
            elif kind == "fx_sdp":
                ok, msg = await eval_fx_sdp(cond, SYMBOL, price, http, cache)
            else:
                continue
        except Exception as exc:
            logger.debug("[gold-ai-trader] scan %s: %s", ctype, exc)
            continue

        if not ok:
            continue

        cand = Candidate(
            type=ctype,
            direction=direction,
            detail=str(msg)[:400],
            quality_atr=1.0,
            sig_key=key,
            raw={"msg": msg, "price": price},
        )
        ok_gate, reason = passes_quality_gates(
            cand,
            price=price,
            session=session,
            cfg=cfg,
            now=now,
            k5=k5,
            k_daily=k_daily,
            k_1h=k1h,
        )
        if not ok_gate:
            logger.debug("[gold-ai-trader] gate skip %s: %s", ctype, reason)
            continue

        # Attach computed quality for pick_best / logging.
        body_q = atr and abs(float(k5[-2][4]) - float(k5[-2][1])) / atr if len(k5) >= 2 else 0
        cand = replace(cand, quality_atr=max(body_q, cand.quality_atr))
        found.append(cand)

    return price, found


def pick_best(candidates: List[Candidate]) -> Optional[Candidate]:
    if not candidates:
        return None
    priority = {
        "sweep_pdh": 5,
        "sweep_pdl": 5,
        "liq_sweep_bull": 5,
        "liq_sweep_bear": 5,
        "sdp_bull": 4,
        "sdp_bear": 4,
        "disp_bull": 3,
        "disp_bear": 3,
        "fvg_bull": 2,
        "fvg_bear": 2,
    }
    return sorted(
        candidates,
        key=lambda c: (priority.get(c.type, 1), c.quality_atr),
        reverse=True,
    )[0]
