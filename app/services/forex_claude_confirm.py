"""
Final Claude confirmation gate for live forex strategy fires.

Claude runs ONLY after every local check has passed (TA, spread, stale price,
session, daily loss, size calc). Fail-closed on API errors, timeouts, or budget cap.

Session windows are fixed UTC (not UK-local) so Asia does not drift at DST changes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.services.anthropic_budget_guard import (
    check_budget_or_block,
    record_call,
    spent_today_usd,
)
from app.services.session_filter import _in_window

logger = logging.getLogger(__name__)

CALLER = "forex_claude_confirm"
HAIKU_MODEL = "claude-haiku-4-5"
SONNET_MODEL = "claude-sonnet-4-5"
API_TIMEOUT_S = 5.0

# Fixed UTC confirm windows — London 06–09, NY 12–15, Asia 01–04 (02–05 UK BST).
FOREX_CLAUDE_CONFIRM_SESSIONS: Dict[str, Tuple[int, int, int, int]] = {
    "london": (6, 0, 9, 0),
    "new_york": (12, 0, 15, 0),
    "asia": (1, 0, 4, 0),
}

HAIKU_INPUT_USD_PER_M = 0.25
HAIKU_OUTPUT_USD_PER_M = 1.25
SONNET_INPUT_USD_PER_M = 3.0
SONNET_OUTPUT_USD_PER_M = 15.0

_confirm_cache: Dict[Tuple[int, str, int], Tuple[bool, str]] = {}
_session_block_logged = False


def forex_claude_confirm_enabled() -> bool:
    if os.environ.get("DISABLE_FOREX_CLAUDE_CONFIRM", "").lower() in ("1", "true", "yes"):
        return False
    return os.environ.get("ENABLE_FOREX_CLAUDE_CONFIRM", "").lower() in ("1", "true", "yes")


def sonnet_escalation_enabled() -> bool:
    return os.environ.get("FOREX_CLAUDE_CONFIRM_ESCALATE_SONNET", "").lower() in (
        "1",
        "true",
        "yes",
    )


def active_confirm_session(now_utc: datetime) -> Optional[str]:
    for sid, win in FOREX_CLAUDE_CONFIRM_SESSIONS.items():
        if _in_window(now_utc, win[0], win[1], win[2], win[3]):
            return sid
    return None


def in_forex_claude_confirm_session(now_utc: Optional[datetime] = None) -> bool:
    return active_confirm_session(now_utc or datetime.utcnow()) is not None


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


def _parse_confirm_json(text: str) -> Optional[Dict[str, Any]]:
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
    return (
        f"Live forex trade confirmation ({session} session, UTC).\n"
        f"Symbol: {symbol}\n"
        f"Side: {direction}\n"
        f"Entry: {entry}\n"
        f"Stop loss: {sl}\n"
        f"Take profit: {tp}\n"
        f"Take profit 2: {tp2}\n"
        f"Risk:reward to TP1: {rr}:1\n\n"
        f"Conditions that passed:\n{cond_lines or '- (none listed)'}\n\n"
        f"Recent price context:\n{json.dumps(recent, default=str)}\n\n"
        "Respond with ONLY valid JSON: "
        '{"confirm": true|false, "reason": "one short sentence"}'
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
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=API_TIMEOUT_S,
        )
        text = (msg.content[0].text or "").strip() if msg.content else ""
        cost = _estimate_cost(model, msg.usage)
        parsed = _parse_confirm_json(text)
        return parsed, cost, text[:300]
    except asyncio.TimeoutError:
        return None, 0.0, "timeout"
    except Exception as exc:
        logger.warning("[forex-claude-confirm] API error: %s", exc)
        return None, 0.0, str(exc)


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
) -> Tuple[bool, str]:
    """
    Final gate before live forex order enqueue.
    Returns (allowed_to_fire, reason). Fail-closed on any uncertainty.
    """
    if not forex_claude_confirm_enabled():
        return True, "disabled"

    ts = now_utc or datetime.utcnow()
    session = active_confirm_session(ts)
    if not session:
        global _session_block_logged
        if not _session_block_logged:
            _session_block_logged = True
            logger.info(
                "[forex-claude-confirm] outside confirm session windows "
                "(UTC London 06–09, NY 12–15, Asia 01–04) — skip, no Claude call"
            )
        return False, "outside_confirm_session"

    ok_budget, budget_reason = check_budget_or_block(CALLER, now=ts)
    if not ok_budget:
        return False, budget_reason

    bar_ts = _bar_ts(ts, timeframe)
    cache_key = (int(strategy_id), str(symbol).upper(), int(bar_ts))
    if cache_key in _confirm_cache:
        cached_ok, cached_reason = _confirm_cache[cache_key]
        return cached_ok, f"cache:{cached_reason}"

    rr = _risk_reward(direction, entry, sl, tp)
    prompt = _build_prompt(
        symbol=symbol,
        direction=direction,
        entry=entry,
        sl=sl,
        tp=tp,
        tp2=tp2,
        rr=rr,
        conditions_met=list(conditions_met or []),
        price_data=dict(price_data or {}),
        session=session,
    )

    parsed, cost, raw = await _call_claude(prompt, model=HAIKU_MODEL)
    model_used = HAIKU_MODEL
    if parsed is None:
        _confirm_cache[cache_key] = (False, raw or "api_error")
        return False, raw or "api_error"

    confirm = bool(parsed.get("confirm"))
    reason = str(parsed.get("reason") or "").strip() or ("confirmed" if confirm else "rejected")

    if (
        not confirm
        and sonnet_escalation_enabled()
        and parsed.get("borderline") is True
    ):
        parsed2, cost2, raw2 = await _call_claude(prompt, model=SONNET_MODEL)
        cost += cost2
        model_used = SONNET_MODEL
        if parsed2 is None:
            _confirm_cache[cache_key] = (False, raw2 or "sonnet_api_error")
            record_call(CALLER, cost, now=ts)
            return False, raw2 or "sonnet_api_error"
        confirm = bool(parsed2.get("confirm"))
        reason = str(parsed2.get("reason") or reason)

    record_call(CALLER, cost, now=ts)
    total = spent_today_usd(ts)
    logger.info(
        "[forex-claude-confirm] strategy=%s symbol=%s session=%s model=%s "
        "confirm=%s reason=%s daily_total_usd=%.6f",
        strategy_id,
        symbol,
        session,
        model_used,
        confirm,
        reason[:120],
        total,
    )
    _confirm_cache[cache_key] = (confirm, reason)
    if not confirm:
        return False, reason
    return True, reason


def reset_confirm_cache_for_tests() -> None:
    _confirm_cache.clear()
    global _session_block_logged
    _session_block_logged = False
