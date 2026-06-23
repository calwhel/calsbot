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
from app.gold_ai_trader.funnel import record as funnel_record
from app.gold_ai_trader.htf_bias import direction_aligns_with_htf, htf_bias_summary
from app.gold_ai_trader.judas_detail import enrich_judas_detail
from app.gold_ai_trader.setup_toggles import (
    cisd_modifier_enabled,
    setup_scannable,
    smt_modifier_enabled,
)
from app.gold_ai_trader.smt_modifier import assess_smt_divergence
from app.gold_ai_trader.structure_score import compute_structure_score

logger = logging.getLogger(__name__)

TF_ZONE = "15m"   # zone definition timeframe (HTF structure)
TF_TIMING = "5m"  # timing / displacement timeframe


def _setup_cooldown_s() -> int:
    try:
        return max(300, int(os.environ.get("GOLD_AI_TRADER_SETUP_COOLDOWN_S", "600")))
    except (TypeError, ValueError):
        return 600


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


def setup_cooldown_elapsed(key: str) -> bool:
    now = time.monotonic()
    last = _last_claude_fired.get(key, 0)
    return (now - last) >= _setup_cooldown_s()


def record_claude_invocation(candidate: Candidate) -> None:
    _last_claude_fired[candidate.sig_key] = time.monotonic()


def _requires_htf_alignment(setup_key: str) -> bool:
    return setup_key.startswith(("breaker_", "eqh_sweep_", "eql_sweep_", "judas_"))


def _priority_map() -> Dict[str, int]:
    """Higher = preferred for Claude. Zone setups rank above sweeps; Judas is early-warning tier."""
    return {
        "ob_bull": 6,
        "ob_bear": 6,
        "fvg_retrace_bull": 6,
        "fvg_retrace_bear": 6,
        "ifvg_bull": 6,
        "ifvg_bear": 6,
        "sweep_pdh": 5,
        "sweep_pdl": 5,
        "liq_sweep_bull": 5,
        "liq_sweep_bear": 5,
        "eqh_sweep_bear": 5,
        "eql_sweep_bull": 5,
        "breaker_bull": 4,
        "breaker_bear": 4,
        "sdp_bull": 4,
        "sdp_bear": 4,
        "disp_bull": 3,
        "disp_bear": 3,
        "fvg_bull": 2,
        "fvg_bear": 2,
        "asian_sweep_bull": 3,
        "asian_sweep_bear": 3,
        "judas_bull": 2,
        "judas_bear": 2,
    }


def _sort_candidates(candidates: List[Candidate]) -> List[Candidate]:
    pri = _priority_map()
    return sorted(
        candidates,
        key=lambda c: (pri.get(c.type, 1), c.raw.get("structure_score", 0), c.quality_atr),
        reverse=True,
    )


def pick_best(candidates: List[Candidate]) -> Optional[Candidate]:
    ranked = _sort_candidates(candidates)
    return ranked[0] if ranked else None


def pick_top_candidates(candidates: List[Candidate], n: int = 3) -> List[Candidate]:
    return _sort_candidates(candidates)[: max(1, n)]


async def scan_candidates(
    http: httpx.AsyncClient,
    *,
    session: str,
    cfg: GoldAiRuntimeConfig,
    price: Optional[float] = None,
    user_id: Optional[int] = None,
    db=None,
) -> Tuple[Optional[float], List[Candidate]]:
    """Return (price, gated candidates). Zone setups use 15m; timing uses 5m."""
    if price is None or price <= 0:
        return None, []

    from app.services.tradfi_prices import get_klines
    from app.services.strategy_ta import (
        eval_fvg,
        eval_fx_displacement,
        eval_forex_prev_level,
        eval_fx_sdp,
        eval_liquidity_sweep,
        eval_fx_breaker,
        eval_equal_hl_sweep,
        eval_order_block,
    )

    k5 = await get_klines(SYMBOL, ASSET_CLASS, "5m", 60) or []
    k1h = await get_klines(SYMBOL, ASSET_CLASS, "1h", 50) or []
    k4h = await get_klines(SYMBOL, ASSET_CLASS, "4h", 30) or []
    k_daily = await get_klines(SYMBOL, ASSET_CLASS, "1d", 5) or []
    now = datetime.utcnow()

    cache: Dict[str, Any] = {"__asset_class__": ASSET_CLASS}
    if user_id:
        cache["__ctrader_user_id__"] = user_id

    found: List[Candidate] = []
    bias = htf_bias_summary(k1h, k4h, k_daily)
    atr = atr_from_klines(k5)
    ta_hits = 0

    checks = [
        # Timing (5m)
        ("sweep_pdh", "forex_prev_level", {"condition": "sweep_pdh", "timeframe": TF_TIMING}, "SHORT"),
        ("sweep_pdl", "forex_prev_level", {"condition": "sweep_pdl", "timeframe": TF_TIMING}, "LONG"),
        ("disp_bull", "fx_displacement", {"direction": "bullish", "timeframe": TF_TIMING}, "LONG"),
        ("disp_bear", "fx_displacement", {"direction": "bearish", "timeframe": TF_TIMING}, "SHORT"),
        ("fvg_bull", "fvg", {"type": "fvg", "direction": "bullish", "timeframe": TF_TIMING, "condition": "just_formed"}, "LONG"),
        ("fvg_bear", "fvg", {"type": "fvg", "direction": "bearish", "timeframe": TF_TIMING, "condition": "just_formed"}, "SHORT"),
        ("liq_sweep_bull", "liquidity_sweep", {"direction": "bullish", "timeframe": TF_TIMING}, "LONG"),
        ("liq_sweep_bear", "liquidity_sweep", {"direction": "bearish", "timeframe": TF_TIMING}, "SHORT"),
        ("sdp_bull", "fx_sdp", {"direction": "bullish", "timeframe": TF_TIMING}, "LONG"),
        ("sdp_bear", "fx_sdp", {"direction": "bearish", "timeframe": TF_TIMING}, "SHORT"),
        # Zone (15m) — retrace into established structure
        (
            "fvg_retrace_bull",
            "fvg",
            {
                "type": "fvg",
                "direction": "bullish",
                "timeframe": TF_ZONE,
                "condition": "price_in_gap",
                "max_age_bars": 40,
                "only_unfilled": True,
            },
            "LONG",
        ),
        (
            "fvg_retrace_bear",
            "fvg",
            {
                "type": "fvg",
                "direction": "bearish",
                "timeframe": TF_ZONE,
                "condition": "price_in_gap",
                "max_age_bars": 40,
                "only_unfilled": True,
            },
            "SHORT",
        ),
        (
            "ifvg_bull",
            "ifvg",
            {
                "type": "ifvg",
                "direction": "bullish",
                "timeframe": TF_ZONE,
                "condition": "price_in_gap",
                "max_age_bars": 40,
            },
            "LONG",
        ),
        (
            "ifvg_bear",
            "ifvg",
            {
                "type": "ifvg",
                "direction": "bearish",
                "timeframe": TF_ZONE,
                "condition": "price_in_gap",
                "max_age_bars": 40,
            },
            "SHORT",
        ),
        (
            "ob_bull",
            "order_block",
            {"ob_type": "bullish", "timeframe": TF_ZONE, "strength": "any"},
            "LONG",
        ),
        (
            "ob_bear",
            "order_block",
            {"ob_type": "bearish", "timeframe": TF_ZONE, "strength": "any"},
            "SHORT",
        ),
        ("breaker_bull", "fx_breaker", {"direction": "bullish", "timeframe": TF_ZONE}, "LONG"),
        ("breaker_bear", "fx_breaker", {"direction": "bearish", "timeframe": TF_ZONE}, "SHORT"),
        (
            "eqh_sweep_bear",
            "equal_hl_sweep",
            {"equal_type": "eqh", "direction": "bearish", "timeframe": TF_TIMING},
            "SHORT",
        ),
        (
            "eql_sweep_bull",
            "equal_hl_sweep",
            {"equal_type": "eql", "direction": "bullish", "timeframe": TF_TIMING},
            "LONG",
        ),
        ("judas_bull", "fx_judas_swing", {"timeframe": TF_ZONE}, "LONG"),
        ("judas_bear", "fx_judas_swing", {"timeframe": TF_ZONE}, "SHORT"),
        ("asian_sweep_bull", "asian_sweep", {"direction": "bullish", "timeframe": TF_TIMING}, "LONG"),
        ("asian_sweep_bear", "asian_sweep", {"direction": "bearish", "timeframe": TF_TIMING}, "SHORT"),
    ]

    for ctype, kind, cond, direction in checks:
        if not setup_scannable(ctype, bias):
            continue
        key = f"{ctype}:{direction}"
        if not setup_cooldown_elapsed(key):
            continue
        cond_eval = {**cond}
        if kind == "fx_judas_swing":
            cond_eval["session"] = session
        try:
            if kind in ("fvg", "ifvg"):
                ok, msg = await eval_fvg(cond_eval, SYMBOL, price, http, cache)
            elif kind == "fx_displacement":
                ok, msg = await eval_fx_displacement(cond_eval, SYMBOL, price, http, cache)
            elif kind == "forex_prev_level":
                ok, msg = await eval_forex_prev_level(cond_eval, SYMBOL, price, http, cache)
            elif kind == "liquidity_sweep":
                ok, msg = await eval_liquidity_sweep(
                    {**cond_eval, "type": "liquidity_sweep"}, SYMBOL, price, http, cache,
                )
            elif kind == "fx_sdp":
                ok, msg = await eval_fx_sdp(cond_eval, SYMBOL, price, http, cache)
            elif kind == "fx_breaker":
                ok, msg = await eval_fx_breaker(cond_eval, SYMBOL, price, http, cache)
            elif kind == "equal_hl_sweep":
                ok, msg = await eval_equal_hl_sweep(cond_eval, SYMBOL, price, http, cache)
            elif kind == "order_block":
                ok, msg = await eval_order_block(cond_eval, SYMBOL, price, http, cache)
            elif kind == "fx_judas_swing":
                from app.services.strategy_ta import eval_fx_judas_swing

                ok, msg = await eval_fx_judas_swing(cond_eval, SYMBOL, price, http, cache)
            elif kind == "asian_sweep":
                from app.gold_ai_trader.asian_sweep import eval_asian_range_sweep

                ok, msg = await eval_asian_range_sweep(cond_eval, SYMBOL, price, http, cache)
            else:
                continue
        except Exception as exc:
            logger.debug("[gold-ai-trader] scan %s: %s", ctype, exc)
            continue

        if not ok:
            continue

        if kind == "fx_judas_swing":
            msg_l = str(msg).lower()
            if direction == "LONG" and "bullish" not in msg_l:
                continue
            if direction == "SHORT" and "bearish" not in msg_l:
                continue
            msg = enrich_judas_detail(
                str(msg),
                direction=direction,
                price=price,
                k5=k5,
                atr=atr,
            )

        ta_hits += 1
        funnel_record("ta_detected", setup=ctype, db=db, session=session)

        if _requires_htf_alignment(ctype):
            aligned, align_reason = direction_aligns_with_htf(direction, bias)
            if not aligned:
                funnel_record("htf_skipped", setup=ctype, reason=align_reason, db=db, session=session)
                logger.debug("[gold-ai-trader] HTF skip %s: %s", ctype, align_reason)
                continue

        align_ok, align_reason = direction_aligns_with_htf(direction, bias)
        raw: Dict[str, Any] = {
            "msg": msg,
            "price": price,
            "htf_align": align_reason,
            "zone_tf": cond.get("timeframe", TF_TIMING),
        }

        if smt_modifier_enabled():
            raw["smt"] = await assess_smt_divergence(
                direction=direction,
                http_client=http,
                cache=cache,
                user_id=user_id,
            )

        if cisd_modifier_enabled():
            from app.gold_ai_trader.cisd_modifier import assess_cisd_modifier

            raw["cisd"] = await assess_cisd_modifier(
                direction=direction,
                http_client=http,
                cache=cache,
            )

        body_q = atr and abs(float(k5[-2][4]) - float(k5[-2][1])) / atr if len(k5) >= 2 else 0
        in_zone = "in_zone" in str(msg).lower() or "fired" in str(msg).lower() or "hit" in str(msg).lower()
        struct_score, struct_line = compute_structure_score(
            candidate_type=ctype,
            direction=direction,
            detail=str(msg),
            quality_atr=max(body_q, 1.0),
            htf_align=align_reason,
            raw=raw,
            in_zone=in_zone,
        )
        raw["structure_score"] = struct_score
        raw["structure_score_line"] = struct_line

        cand = Candidate(
            type=ctype,
            direction=direction,
            detail=str(msg)[:400],
            quality_atr=max(body_q, 1.0),
            sig_key=key,
            raw=raw,
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
            funnel_record("gate_skipped", setup=ctype, reason=reason, db=db, session=session)
            logger.debug("[gold-ai-trader] gate skip %s: %s", ctype, reason)
            continue

        funnel_record("candidate_passed", setup=ctype, db=db, session=session)
        found.append(cand)

    if ta_hits == 0:
        funnel_record("no_ta_match", db=db, session=session)

    return price, found
