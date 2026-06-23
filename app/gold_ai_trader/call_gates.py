"""Pre-Claude quality gates and call deduplication (gold module only)."""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from app.gold_ai_trader.config import GoldAiRuntimeConfig
from app.gold_ai_trader.context_levels import (
    compute_asian_range,
    compute_daily_open,
    compute_pdh_pdl,
    compute_session_range,
)
from app.gold_ai_trader.models import GoldAiDecision


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


MIN_BODY_ATR = _env_float("GOLD_AI_TRADER_MIN_BODY_ATR", 0.8)
NEAR_LEVEL_ATR = _env_float("GOLD_AI_TRADER_NEAR_LEVEL_ATR", 1.25)
MIN_RVOL = _env_float("GOLD_AI_TRADER_MIN_RVOL", 1.15)
KILLZONE_MINUTES = _env_int("GOLD_AI_TRADER_KILLZONE_MINUTES", 90)
MIN_CLAUDE_GAP_S = _env_int("GOLD_AI_TRADER_MIN_CLAUDE_GAP_S", 120)
DEDUPE_PRICE_ATR = _env_float("GOLD_AI_TRADER_DEDUPE_PRICE_ATR", 0.35)

# Setups that define their own level/zone — skip redundant PDH proximity gate.
_NEAR_LEVEL_EXEMPT_PREFIXES = (
    "sweep_", "liq_sweep", "sdp_", "eqh_sweep", "eql_sweep",
    "ob_", "breaker_", "fvg_retrace_", "ifvg_",
)

# Displacement-heavy setups carry their own momentum — skip post-killzone RVOL gate.
_RVOL_EXEMPT_PREFIXES = (
    "sweep_", "liq_sweep", "sdp_", "disp_", "eqh_sweep", "eql_sweep",
)


def _setup_prefix_match(setup_type: str, prefixes: tuple) -> bool:
    t = setup_type or ""
    return any(t.startswith(p) for p in prefixes)


def atr_from_klines(k5: List[list], period: int = 14) -> float:
    closes = [float(r[4]) for r in k5 if r and len(r) >= 5]
    if len(closes) < period + 1:
        return 0.0
    trs = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    if len(trs) < period:
        return sum(trs) / max(len(trs), 1)
    return sum(trs[-period:]) / period


def rvol_from_klines(k5: List[list]) -> float:
    vols = [float(r[5]) for r in k5 if r and len(r) >= 6]
    if len(vols) < 20:
        return 1.0
    avg = sum(vols[-21:-1]) / 20
    return (vols[-1] / avg) if avg else 1.0


def last_closed_body_atr(k5: List[list], atr: float) -> float:
    """Body size of last completed 5m bar vs ATR."""
    if not k5 or len(k5) < 2 or atr <= 0:
        return 0.0
    row = k5[-2]  # last closed candle (current may be forming)
    try:
        o, c = float(row[1]), float(row[4])
        return abs(c - o) / atr
    except (TypeError, ValueError, IndexError):
        return 0.0


def minutes_into_session(now: datetime, session: str, cfg: GoldAiRuntimeConfig) -> int:
    if session == "london":
        return max(0, (now.hour - cfg.london_start_hour) * 60 + now.minute)
    if session == "new_york":
        return max(0, (now.hour - cfg.ny_start_hour) * 60 + now.minute)
    return 0


def in_killzone(now: datetime, session: str, cfg: GoldAiRuntimeConfig) -> bool:
    return minutes_into_session(now, session, cfg) <= KILLZONE_MINUTES


def collect_key_levels(
    price: float,
    session: str,
    cfg: GoldAiRuntimeConfig,
    now: datetime,
    k_daily: List[list],
    k_1h: List[list],
    k_5m: List[list],
) -> List[float]:
    levels: List[float] = []
    pdh, pdl = compute_pdh_pdl(now=now, k_daily=k_daily, k_1h=k_1h, k_5m=k_5m)
    for v in (pdh, pdl, compute_daily_open(now, k_daily, k_1h)):
        if v is not None:
            levels.append(float(v))
    asian_hi, asian_lo = compute_asian_range(now, k_5m, k_1h)
    for v in (asian_hi, asian_lo):
        if v is not None:
            levels.append(float(v))
    sess_hi, sess_lo = compute_session_range(now, session, cfg, k_5m, k_1h)
    for v in (sess_hi, sess_lo):
        if v is not None:
            levels.append(float(v))
    return levels


def nearest_level_distance(price: float, levels: List[float]) -> Optional[float]:
    if not levels:
        return None
    return min(abs(price - lv) for lv in levels)


def _parse_quality_from_detail(detail: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*[×x]\s*ATR", detail, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    m = re.search(r"quality[^\d]*(\d+(?:\.\d+)?)", detail, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def passes_quality_gates(
    candidate: Candidate,
    *,
    price: float,
    session: str,
    cfg: GoldAiRuntimeConfig,
    now: datetime,
    k5: List[list],
    k_daily: List[list],
    k_1h: List[list],
) -> Tuple[bool, str]:
    """Cheap engine checks before spending a Claude call."""
    atr = atr_from_klines(k5)
    if atr <= 0:
        return False, "atr_unavailable"

    body_atr = last_closed_body_atr(k5, atr)
    detail_q = _parse_quality_from_detail(candidate.detail)
    disp_ok = body_atr >= MIN_BODY_ATR or (detail_q is not None and detail_q >= MIN_BODY_ATR)
    if candidate.type.startswith(("disp_", "sweep_", "liq_sweep", "fvg_", "fvg_retrace_", "ifvg_", "sdp_", "breaker_", "eqh_sweep", "eql_sweep", "ob_")):
        if not disp_ok and candidate.quality_atr < MIN_BODY_ATR:
            return False, f"displacement_body={body_atr:.2f}atr<{MIN_BODY_ATR}"

    levels = collect_key_levels(price, session, cfg, now, k_daily, k_1h, k5)
    if not _setup_prefix_match(candidate.type, _NEAR_LEVEL_EXEMPT_PREFIXES):
        near = nearest_level_distance(price, levels)
        if near is None:
            return False, "key_levels_unavailable"
        if near > NEAR_LEVEL_ATR * atr:
            return False, f"not_near_level({near:.2f}>{NEAR_LEVEL_ATR}×ATR)"

    rvol = rvol_from_klines(k5)
    kz = in_killzone(now, session, cfg)
    if (
        not kz
        and rvol < MIN_RVOL
        and not _setup_prefix_match(candidate.type, _RVOL_EXEMPT_PREFIXES)
    ):
        return False, f"outside_killzone_rvol={rvol:.2f}<{MIN_RVOL}"

    return True, "ok"


def should_invoke_claude(
    db,
    candidate: Candidate,
    price: float,
    atr: float,
    *,
    setup_cooldown_s: int,
) -> Tuple[bool, str]:
    """Ensure Claude runs only on new, materially distinct candidates."""
    now = datetime.utcnow()

    last_any = (
        db.query(GoldAiDecision)
        .order_by(GoldAiDecision.ts.desc())
        .first()
    )
    if last_any and last_any.ts:
        gap = (now - last_any.ts).total_seconds()
        if gap < MIN_CLAUDE_GAP_S:
            return False, f"claude_gap_{int(gap)}s<{MIN_CLAUDE_GAP_S}s"

    since = now - timedelta(seconds=setup_cooldown_s)

    recent_same = (
        db.query(GoldAiDecision)
        .filter(
            GoldAiDecision.ts >= since,
            GoldAiDecision.candidate_type == candidate.type,
        )
        .order_by(GoldAiDecision.ts.desc())
        .all()
    )
    if recent_same and atr > 0:
        for row in recent_same:
            last_price = _spot_from_context(row.context_snapshot)
            if last_price is not None and abs(price - last_price) < DEDUPE_PRICE_ATR * atr:
                return False, f"duplicate_{candidate.type}_price_unchanged"

    return True, "ok"


def call_stats_today(db) -> dict:
    """Breakdown of today's Claude invocations by setup type (for diagnostics)."""
    from app.gold_ai_trader.guardrails import _calls_cutoff
    from sqlalchemy import func

    rows = (
        db.query(GoldAiDecision.candidate_type, func.count(GoldAiDecision.id))
        .filter(GoldAiDecision.ts >= _calls_cutoff(db))
        .group_by(GoldAiDecision.candidate_type)
        .all()
    )
    return {str(t or "unknown"): int(n) for t, n in rows}


def _spot_from_context(snapshot: Optional[str]) -> Optional[float]:
    if not snapshot:
        return None
    m = re.search(r"Spot:\s*([\d.]+)", snapshot)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None
