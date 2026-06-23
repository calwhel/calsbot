"""
Final Claude gate + optional dynamic TP/SL for live forex fires.

Single Haiku call when confirm and/or dynamic TP/SL are enabled.
Dynamic levels fail-safe to static SL/TP; confirm gate fail-closed on reject.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.services.anthropic_budget_guard import (
    check_budget_or_block,
    record_call,
    spent_today_usd,
)
from app.services.forex_sessions import (
    active_live_forex_session,
    in_live_forex_session,
)

logger = logging.getLogger(__name__)

CALLER = "forex_claude_confirm"
HAIKU_MODEL = "claude-haiku-4-5"
SONNET_MODEL = "claude-sonnet-4-5"
API_TIMEOUT_S = 5.0

SL_PIPS_MIN = 20.0
SL_PIPS_MAX = 60.0
RR_MIN = 2.0

HAIKU_INPUT_USD_PER_M = 0.25
HAIKU_OUTPUT_USD_PER_M = 1.25
SONNET_INPUT_USD_PER_M = 3.0
SONNET_OUTPUT_USD_PER_M = 15.0

_fire_cache: Dict[Tuple[int, str, int], "ForexClaudeFireResult"] = {}
_session_block_logged = False


@dataclass
class ForexClaudeFireResult:
    allowed_to_fire: bool
    reason: str
    sl_price: float
    tp_price: float
    sl_pct: float
    tp_pct: float
    sl_pips: Optional[float] = None
    tp_pips: Optional[float] = None
    used_dynamic: bool = False
    claude_called: bool = False
    static_fallback: bool = False


def forex_claude_confirm_enabled() -> bool:
    if os.environ.get("DISABLE_FOREX_CLAUDE_CONFIRM", "").lower() in ("1", "true", "yes"):
        return False
    return os.environ.get("ENABLE_FOREX_CLAUDE_CONFIRM", "").lower() in ("1", "true", "yes")


def strategy_dynamic_tp_sl_enabled(config: dict) -> bool:
    return bool((config or {}).get("dynamic_tp_sl"))


def sonnet_escalation_enabled() -> bool:
    return os.environ.get("FOREX_CLAUDE_CONFIRM_ESCALATE_SONNET", "").lower() in (
        "1",
        "true",
        "yes",
    )


def active_confirm_session(now_utc: datetime) -> Optional[str]:
    return active_live_forex_session(now_utc)


def in_forex_claude_confirm_session(now_utc: Optional[datetime] = None) -> bool:
    return in_live_forex_session(now_utc)


def price_data_ok_for_claude(price_data: Dict[str, Any]) -> bool:
    if not price_data:
        return False
    try:
        price = float(price_data.get("price") or 0)
    except (TypeError, ValueError):
        return False
    if price <= 0 or not math.isfinite(price):
        return False
    if price_data.get("kline_synthetic"):
        return False
    ps = (price_data.get("price_source") or "").lower()
    if ps in ("unknown", "stale", ""):
        return False
    ks = (price_data.get("kline_source") or "").lower()
    if ks == "synthetic":
        return False
    return True


def assert_valid_sl_price(sl_price: Any) -> bool:
    try:
        v = float(sl_price)
        return v > 0 and math.isfinite(v)
    except (TypeError, ValueError):
        return False


def clamp_validate_dynamic_levels(
    sl_pips: Any,
    tp_pips: Any,
) -> Optional[Tuple[float, float, float]]:
    """Return (sl_pips, tp_pips, rr) or None if invalid."""
    try:
        sl = float(sl_pips)
        tp = float(tp_pips)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(sl) or not math.isfinite(tp) or sl <= 0 or tp <= 0:
        return None
    if sl < SL_PIPS_MIN or sl > SL_PIPS_MAX:
        if sl > SL_PIPS_MAX:
            sl = SL_PIPS_MAX
        elif sl < SL_PIPS_MIN:
            sl = SL_PIPS_MIN
    if tp < RR_MIN * sl:
        return None
    rr = tp / sl
    if rr < RR_MIN:
        return None
    return round(sl, 2), round(tp, 2), round(rr, 2)


def pips_to_prices_and_pct(
    symbol: str,
    direction: str,
    entry: float,
    sl_pips: float,
    tp_pips: float,
) -> Tuple[float, float, float, float]:
    from app.services.forex_engine import pip_size, pips_to_pct

    ps = pip_size(symbol)
    if direction.upper() == "LONG":
        sl_price = entry - sl_pips * ps
        tp_price = entry + tp_pips * ps
    else:
        sl_price = entry + sl_pips * ps
        tp_price = entry - tp_pips * ps
    sl_pct = pips_to_pct(symbol, entry, sl_pips)
    tp_pct = pips_to_pct(symbol, entry, tp_pips)
    return sl_price, tp_price, sl_pct, tp_pct


def _bar_ts(now_utc: datetime, timeframe: str) -> int:
    tf_secs = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }.get(str(timeframe or "15m").lower(), 900)
    ts = int(now_utc.timestamp())
    return ((ts // tf_secs) - 1) * tf_secs


def _risk_reward(direction: str, entry: float, sl: float, tp: float) -> float:
    if not entry or not sl or not tp:
        return 0.0
    if direction.upper() == "LONG":
        risk = entry - sl
        reward = tp - entry
    else:
        risk = sl - entry
        reward = entry - tp
    if risk <= 0:
        return 0.0
    return round(reward / risk, 2)


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    blob = text.strip()
    if "```" in blob:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", blob, re.DOTALL | re.IGNORECASE)
        if m:
            blob = m.group(1).strip()
    start = blob.find("{")
    end = blob.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(blob[start : end + 1])
    except json.JSONDecodeError:
        return None


def _estimate_cost(model: str, usage) -> float:
    tin = int(getattr(usage, "input_tokens", 0) or 0)
    tout = int(getattr(usage, "output_tokens", 0) or 0)
    if "sonnet" in model.lower():
        return round(
            tin / 1_000_000 * SONNET_INPUT_USD_PER_M
            + tout / 1_000_000 * SONNET_OUTPUT_USD_PER_M,
            6,
        )
    return round(
        tin / 1_000_000 * HAIKU_INPUT_USD_PER_M
        + tout / 1_000_000 * HAIKU_OUTPUT_USD_PER_M,
        6,
    )


def _static_result(
    *,
    static_sl: float,
    static_tp: float,
    static_sl_pct: float,
    static_tp_pct: float,
    reason: str,
    allowed: bool = True,
    static_fallback: bool = False,
) -> ForexClaudeFireResult:
    return ForexClaudeFireResult(
        allowed_to_fire=allowed,
        reason=reason,
        sl_price=static_sl,
        tp_price=static_tp,
        sl_pct=static_sl_pct,
        tp_pct=static_tp_pct,
        static_fallback=static_fallback,
    )


def _build_prompt(
    *,
    symbol: str,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    tp2: Optional[float],
    rr: float,
    conditions_met: List[str],
    price_data: Dict[str, Any],
    session: str,
    wants_confirm: bool,
    wants_dynamic: bool,
) -> str:
    cond_lines = "\n".join(f"- {c}" for c in (conditions_met or [])[:20])
    recent = {
        "price": price_data.get("price"),
        "rsi": price_data.get("rsi"),
        "high_24h": price_data.get("high_24h"),
        "low_24h": price_data.get("low_24h"),
        "change_24h_pct": price_data.get("change_24h"),
        "price_source": price_data.get("price_source"),
    }
    constraints = (
        f"Propose stop/target as pip DISTANCES from entry (not prices). "
        f"SL must be {SL_PIPS_MIN:.0f}–{SL_PIPS_MAX:.0f} pips. "
        f"TP must be at least {RR_MIN:.0f}× SL distance (higher allowed)."
    )
    if wants_confirm and wants_dynamic:
        json_fmt = (
            '{"confirm": true|false, "sl_pips": number, "tp_pips": number, '
            '"reason": "one short sentence"}'
        )
        task = (
            "Confirm whether to take this live trade AND propose SL/TP pip distances."
        )
    elif wants_confirm:
        json_fmt = '{"confirm": true|false, "reason": "one short sentence"}'
        task = "Confirm whether to take this live trade."
    else:
        json_fmt = '{"sl_pips": number, "tp_pips": number, "reason": "one short sentence"}'
        task = "Propose SL/TP pip distances for this live trade entry."
    return (
        f"Live forex — {task} ({session} session, UTC).\n"
        f"Symbol: {symbol}\nSide: {direction}\nEntry: {entry}\n"
        f"Strategy static SL price: {sl}\nStrategy static TP price: {tp}\n"
        f"Strategy static TP2: {tp2}\nStatic R:R to TP1: {rr}:1\n\n"
        f"Conditions that passed:\n{cond_lines or '- (none listed)'}\n\n"
        f"Recent price context:\n{json.dumps(recent, default=str)}\n\n"
        f"{constraints}\n\n"
        f"Respond with ONLY valid JSON: {json_fmt}"
    )


async def _call_claude(
    prompt: str,
    *,
    model: str = HAIKU_MODEL,
) -> Tuple[Optional[Dict[str, Any]], float, str]:
    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY")
    )
    if not api_key:
        return None, 0.0, "no_api_key"

    try:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=api_key)
        msg = await asyncio.wait_for(
            client.messages.create(
                model=model,
                max_tokens=160,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=API_TIMEOUT_S,
        )
        text = (msg.content[0].text or "").strip() if msg.content else ""
        cost = _estimate_cost(model, msg.usage)
        parsed = _parse_json(text)
        return parsed, cost, text[:300]
    except asyncio.TimeoutError:
        return None, 0.0, "timeout"
    except Exception as exc:
        logger.warning("[forex-claude-confirm] API error: %s", exc)
        return None, 0.0, str(exc)


def _apply_dynamic_from_parsed(
    *,
    parsed: Dict[str, Any],
    symbol: str,
    direction: str,
    entry: float,
    static_sl: float,
    static_tp: float,
    static_sl_pct: float,
    static_tp_pct: float,
) -> ForexClaudeFireResult:
    levels = clamp_validate_dynamic_levels(
        parsed.get("sl_pips"),
        parsed.get("tp_pips"),
    )
    if not levels:
        return _static_result(
            static_sl=static_sl,
            static_tp=static_tp,
            static_sl_pct=static_sl_pct,
            static_tp_pct=static_tp_pct,
            reason="dynamic_levels_rejected",
            static_fallback=True,
        )
    sl_p, tp_p, rr = levels
    sl_price, tp_price, sl_pct, tp_pct = pips_to_prices_and_pct(
        symbol, direction, entry, sl_p, tp_p,
    )
    if not assert_valid_sl_price(sl_price):
        return _static_result(
            static_sl=static_sl,
            static_tp=static_tp,
            static_sl_pct=static_sl_pct,
            static_tp_pct=static_tp_pct,
            reason="dynamic_sl_invalid",
            static_fallback=True,
        )
    return ForexClaudeFireResult(
        allowed_to_fire=True,
        reason=str(parsed.get("reason") or f"dynamic_sl={sl_p} tp={tp_p} rr={rr}"),
        sl_price=sl_price,
        tp_price=tp_price,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        sl_pips=sl_p,
        tp_pips=tp_p,
        used_dynamic=True,
        claude_called=True,
    )


async def resolve_forex_claude_fire(
    *,
    strategy_id: int,
    symbol: str,
    direction: str,
    entry: float,
    static_sl: float,
    static_tp: float,
    static_sl_pct: float,
    static_tp_pct: float,
    static_tp2: Optional[float] = None,
    config: Optional[dict] = None,
    conditions_met: Optional[List[str]] = None,
    price_data: Optional[Dict[str, Any]] = None,
    timeframe: str = "15m",
    now_utc: Optional[datetime] = None,
) -> ForexClaudeFireResult:
    cfg = config or {}
    wants_confirm = forex_claude_confirm_enabled()
    wants_dynamic = strategy_dynamic_tp_sl_enabled(cfg)
    pd = dict(price_data or {})

    base = _static_result(
        static_sl=static_sl,
        static_tp=static_tp,
        static_sl_pct=static_sl_pct,
        static_tp_pct=static_tp_pct,
        reason="static",
    )

    if not wants_confirm and not wants_dynamic:
        return base

    ts = now_utc or datetime.utcnow()
    session = active_confirm_session(ts)
    if not session:
        global _session_block_logged
        if not _session_block_logged:
            _session_block_logged = True
            logger.info(
                "[forex-claude-confirm] outside confirm session windows — skip/no dynamic call"
            )
        if wants_confirm:
            return _static_result(
                static_sl=static_sl,
                static_tp=static_tp,
                static_sl_pct=static_sl_pct,
                static_tp_pct=static_tp_pct,
                reason="outside_confirm_session",
                allowed=False,
            )
        return _static_result(
            static_sl=static_sl,
            static_tp=static_tp,
            static_sl_pct=static_sl_pct,
            static_tp_pct=static_tp_pct,
            reason="outside_session_static",
            static_fallback=True,
        )

    ok_budget, budget_reason = check_budget_or_block(CALLER, now=ts)
    if not ok_budget:
        if wants_confirm:
            return _static_result(
                static_sl=static_sl,
                static_tp=static_tp,
                static_sl_pct=static_sl_pct,
                static_tp_pct=static_tp_pct,
                reason=budget_reason,
                allowed=False,
            )
        return _static_result(
            static_sl=static_sl,
            static_tp=static_tp,
            static_sl_pct=static_sl_pct,
            static_tp_pct=static_tp_pct,
            reason=f"budget_static_fallback:{budget_reason}",
            static_fallback=True,
        )

    if not price_data_ok_for_claude(pd):
        if wants_confirm:
            return _static_result(
                static_sl=static_sl,
                static_tp=static_tp,
                static_sl_pct=static_sl_pct,
                static_tp_pct=static_tp_pct,
                reason="stale_price_no_claude",
                allowed=False,
            )
        return _static_result(
            static_sl=static_sl,
            static_tp=static_tp,
            static_sl_pct=static_sl_pct,
            static_tp_pct=static_tp_pct,
            reason="stale_price_static_fallback",
            static_fallback=True,
        )

    bar_ts = _bar_ts(ts, timeframe)
    cache_key = (int(strategy_id), str(symbol).upper(), int(bar_ts))
    if cache_key in _fire_cache:
        cached = _fire_cache[cache_key]
        return cached

    rr = _risk_reward(direction, entry, static_sl, static_tp)
    prompt = _build_prompt(
        symbol=symbol,
        direction=direction,
        entry=entry,
        sl=static_sl,
        tp=static_tp,
        tp2=static_tp2,
        rr=rr,
        conditions_met=list(conditions_met or []),
        price_data=pd,
        session=session,
        wants_confirm=wants_confirm,
        wants_dynamic=wants_dynamic,
    )

    parsed, cost, raw = await _call_claude(prompt, model=HAIKU_MODEL)
    model_used = HAIKU_MODEL
    if parsed is None:
        if wants_confirm:
            result = _static_result(
                static_sl=static_sl,
                static_tp=static_tp,
                static_sl_pct=static_sl_pct,
                static_tp_pct=static_tp_pct,
                reason=raw or "api_error",
                allowed=False,
            )
            result.claude_called = True
            _fire_cache[cache_key] = result
            record_call(CALLER, cost, now=ts)
            return result
        result = _static_result(
            static_sl=static_sl,
            static_tp=static_tp,
            static_sl_pct=static_sl_pct,
            static_tp_pct=static_tp_pct,
            reason=raw or "api_error",
            static_fallback=True,
        )
        result.claude_called = True
        _fire_cache[cache_key] = result
        record_call(CALLER, cost, now=ts)
        return result

    confirm = bool(parsed.get("confirm")) if wants_confirm else True
    reason = str(parsed.get("reason") or "").strip()

    if (
        wants_confirm
        and not confirm
        and sonnet_escalation_enabled()
        and parsed.get("borderline") is True
    ):
        parsed2, cost2, raw2 = await _call_claude(prompt, model=SONNET_MODEL)
        cost += cost2
        model_used = SONNET_MODEL
        if parsed2 is None:
            result = _static_result(
                static_sl=static_sl,
                static_tp=static_tp,
                static_sl_pct=static_sl_pct,
                static_tp_pct=static_tp_pct,
                reason=raw2 or "sonnet_api_error",
                allowed=False,
            )
            result.claude_called = True
            _fire_cache[cache_key] = result
            record_call(CALLER, cost, now=ts)
            return result
        parsed = parsed2
        confirm = bool(parsed.get("confirm"))
        reason = str(parsed.get("reason") or reason)

    record_call(CALLER, cost, now=ts)
    total = spent_today_usd(ts)
    logger.info(
        "[forex-claude-confirm] strategy=%s symbol=%s session=%s model=%s "
        "confirm=%s dynamic=%s daily_total_usd=%.6f reason=%s",
        strategy_id,
        symbol,
        session,
        model_used,
        confirm if wants_confirm else "n/a",
        wants_dynamic,
        total,
        (reason or "")[:120],
    )

    if wants_confirm and not confirm:
        result = _static_result(
            static_sl=static_sl,
            static_tp=static_tp,
            static_sl_pct=static_sl_pct,
            static_tp_pct=static_tp_pct,
            reason=reason or "rejected",
            allowed=False,
        )
        result.claude_called = True
        _fire_cache[cache_key] = result
        return result

    if wants_dynamic:
        result = _apply_dynamic_from_parsed(
            parsed=parsed,
            symbol=symbol,
            direction=direction,
            entry=entry,
            static_sl=static_sl,
            static_tp=static_tp,
            static_sl_pct=static_sl_pct,
            static_tp_pct=static_tp_pct,
        )
        if not reason and result.reason:
            pass
        elif reason:
            result.reason = reason
        result.claude_called = True
        result.allowed_to_fire = True
        _fire_cache[cache_key] = result
        return result

    result = _static_result(
        static_sl=static_sl,
        static_tp=static_tp,
        static_sl_pct=static_sl_pct,
        static_tp_pct=static_tp_pct,
        reason=reason or "confirmed",
    )
    result.claude_called = True
    _fire_cache[cache_key] = result
    return result


async def maybe_forex_claude_confirm(
    *,
    strategy_id: int,
    symbol: str,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    tp2: Optional[float] = None,
    conditions_met: Optional[List[str]] = None,
    price_data: Optional[Dict[str, Any]] = None,
    timeframe: str = "15m",
    now_utc: Optional[datetime] = None,
    config: Optional[dict] = None,
) -> Tuple[bool, str]:
    """Backward-compatible confirm-only wrapper."""
    if not forex_claude_confirm_enabled() and not strategy_dynamic_tp_sl_enabled(config or {}):
        return True, "disabled"
    res = await resolve_forex_claude_fire(
        strategy_id=strategy_id,
        symbol=symbol,
        direction=direction,
        entry=entry,
        static_sl=sl,
        static_tp=tp,
        static_sl_pct=0.0,
        static_tp_pct=0.0,
        static_tp2=tp2,
        config=config,
        conditions_met=conditions_met,
        price_data=price_data,
        timeframe=timeframe,
        now_utc=now_utc,
    )
    return res.allowed_to_fire, res.reason


def reset_confirm_cache_for_tests() -> None:
    _fire_cache.clear()
    global _session_block_logged
    _session_block_logged = False
