"""Setup Readiness Score — gate Claude to tradeable, at-entry setups only."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.gold_ai_trader.call_gates import last_closed_body_atr
from app.gold_ai_trader.context_history import parse_zone_from_detail
from app.gold_ai_trader.context_levels import compute_asian_range, compute_premium_discount
from app.gold_ai_trader.decision_validator import (
    MIN_RR,
    READINESS_RR_RISK_ATR,
    _nearest_tp_candidate,
    _suggested_sl_from_zone,
)
from app.gold_ai_trader.htf_bias import direction_aligns_with_htf


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


READINESS_MIN = _env_int("GOLD_AI_READINESS_MIN", 55)
SWEEP_MIN_BODY_ATR = _env_float("GOLD_AI_SWEEP_MIN_BODY_ATR", 0.8)
RECLAIM_BUFFER_ATR = _env_float("GOLD_AI_RECLAIM_BUFFER_ATR", 0.05)
ZONE_TOLERANCE_ATR = _env_float("GOLD_AI_ZONE_TOLERANCE_ATR", 0.15)

_SWEEP_PREFIXES = (
    "sweep_", "liq_sweep_", "eqh_sweep_", "eql_sweep_", "asian_sweep_", "liquidity_grab_",
)
_MOMENTUM_PREFIXES = _SWEEP_PREFIXES + ("disp_", "sdp_", "momentum_")
_ZONE_PREFIXES = ("ob_", "fvg_retrace_", "ifvg_", "breaker_")


def readiness_min_score() -> int:
    return max(0, min(100, READINESS_MIN))


def _is_sweep_setup(setup_type: str) -> bool:
    t = setup_type or ""
    return any(t.startswith(p) for p in _SWEEP_PREFIXES)


def _is_zone_setup(setup_type: str) -> bool:
    t = setup_type or ""
    return any(t.startswith(p) for p in _ZONE_PREFIXES)


def parse_reclaim_level(detail: str) -> Optional[float]:
    """Extract reclaim / sweep level from TA detail string."""
    if not detail:
        return None
    m = re.search(r"reclaim\s*@\s*([\d.]+)", detail, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    m = re.search(
        r"(?:PDay|PWeek|PDH|PDL|high|low)\s+([\d.]+)",
        detail,
        re.I,
    )
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    m = re.search(r"sweep\s*@\s*([\d.]+)", detail, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    zone = parse_zone_from_detail(detail)
    if zone:
        return (zone[0] + zone[1]) / 2.0
    return None


def reclaim_held(
    price: float,
    reclaim: float,
    direction: str,
    atr: float,
) -> bool:
    if price <= 0 or reclaim <= 0 or atr <= 0:
        return True
    buf = RECLAIM_BUFFER_ATR * atr
    d = (direction or "").upper()
    if d == "LONG":
        return price >= reclaim - buf
    if d == "SHORT":
        return price <= reclaim + buf
    return True


def spot_in_entry_zone(
    price: float,
    zone: Optional[Tuple[float, float]],
    atr: float,
) -> Tuple[bool, str]:
    if not zone or atr <= 0 or price <= 0:
        return False, "no_zone"
    z_bot, z_top = min(zone), max(zone)
    tol = ZONE_TOLERANCE_ATR * atr
    if (z_bot - tol) <= price <= (z_top + tol):
        return True, "in_zone"
    if price < z_bot - tol:
        return False, "below_zone"
    return False, "above_zone_chasing"


def momentum_6bar_aligned(k5: List[list], direction: str) -> Tuple[bool, int]:
    """True when majority of last 6 closed 5m bars agree with trade direction."""
    if not k5 or len(k5) < 8:
        return True, 0
    closed = k5[-7:-1]
    d = (direction or "").upper()
    with_trend = 0
    against = 0
    for row in closed:
        try:
            o, c = float(row[1]), float(row[4])
        except (TypeError, ValueError, IndexError):
            continue
        bull = c > o
        if d == "LONG":
            if bull:
                with_trend += 1
            elif c < o:
                against += 1
        elif d == "SHORT":
            if not bull and c < o:
                with_trend += 1
            elif bull:
                against += 1
    return with_trend >= against, with_trend - against


def rr_feasible(
    price: float,
    direction: str,
    atr: float,
    key_levels: List[float],
    zone: Optional[Tuple[float, float]],
) -> Tuple[bool, Optional[float]]:
    if atr <= 0 or price <= 0:
        return False, None
    d = (direction or "").upper()
    entry = price
    if zone:
        entry = (zone[0] + zone[1]) / 2.0
        struct_sl = _suggested_sl_from_zone(zone, d, atr)
        if struct_sl is not None:
            risk = abs(entry - struct_sl)
        else:
            risk = READINESS_RR_RISK_ATR * atr
    else:
        risk = READINESS_RR_RISK_ATR * atr
    if risk <= 0:
        return False, None
    tp = _nearest_tp_candidate(entry, d, key_levels, MIN_RR if MIN_RR > 0 else 0.0, risk)
    if tp is None:
        if MIN_RR <= 0:
            return True, None
        return False, None
    rr = abs(tp - entry) / risk if d == "LONG" else abs(entry - tp) / risk
    if MIN_RR <= 0:
        return rr > 0, rr
    return rr >= MIN_RR, rr


@dataclass
class ReadinessResult:
    score: int
    passed: bool
    breakdown: str
    checklist: Dict[str, bool] = field(default_factory=dict)
    hard_fail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "breakdown": self.breakdown,
            "checklist": dict(self.checklist),
            "hard_fail": self.hard_fail,
            "min_required": readiness_min_score(),
        }


def compute_setup_readiness(
    *,
    setup_type: str,
    direction: str,
    detail: str,
    price: float,
    atr: float,
    k5: List[list],
    k1h: List[list],
    bias: dict,
    htf_align_reason: str,
    key_levels: List[float],
    in_zone_hint: bool = False,
    smt: Optional[Dict[str, Any]] = None,
    now=None,
    session: str = "",
    cfg=None,
) -> ReadinessResult:
    """
    Score 0–100 for Claude readiness. Hard-fails mirror Claude's skip reasons.
    """
    parts: List[str] = []
    checklist: Dict[str, bool] = {}
    score = 35

    aligned, align_reason = direction_aligns_with_htf(direction, bias)
    htf = (bias.get("htf_bias") or "mixed").lower()
    checklist["htf_aligned"] = "aligned" in (align_reason or "")

    if not aligned and htf in ("bullish", "bearish"):
        return ReadinessResult(
            0,
            False,
            "counter-HTF",
            checklist={"htf_aligned": False},
            hard_fail=f"counter_htf:{align_reason}",
        )
    if checklist["htf_aligned"]:
        score += 25
        parts.append("HTF+25")
    elif htf == "mixed":
        score += 6
        parts.append("HTF-mixed+6")

    body_atr = last_closed_body_atr(k5, atr)
    checklist["displacement_ok"] = body_atr >= SWEEP_MIN_BODY_ATR
    if setup_type.startswith(_MOMENTUM_PREFIXES):
        if body_atr < SWEEP_MIN_BODY_ATR:
            return ReadinessResult(
                max(0, score - 10),
                False,
                f"weak_disp={body_atr:.2f}atr",
                checklist={**checklist, "displacement_ok": False},
                hard_fail=f"displacement_{body_atr:.2f}<{SWEEP_MIN_BODY_ATR}",
            )
        score += 15
        parts.append(f"disp+15({body_atr:.2f}×ATR)")
    elif body_atr >= 0.6:
        score += 8
        parts.append(f"body+8")

    zone = parse_zone_from_detail(detail)
    in_zone, zone_tag = spot_in_entry_zone(price, zone, atr) if zone else (in_zone_hint, "hint")
    checklist["at_entry"] = in_zone or in_zone_hint

    if _is_zone_setup(setup_type):
        if not in_zone and not in_zone_hint:
            return ReadinessResult(
                score,
                False,
                zone_tag,
                checklist={**checklist, "at_entry": False},
                hard_fail=f"not_at_entry:{zone_tag}",
            )
        score += 20
        parts.append("in-zone+20")
    elif zone and in_zone:
        score += 12
        parts.append("near-zone+12")

    if _is_sweep_setup(setup_type):
        reclaim = parse_reclaim_level(detail)
        held = reclaim is None or reclaim_held(price, reclaim, direction, atr)
        checklist["reclaim_held"] = held
        if reclaim is not None and not held:
            return ReadinessResult(
                score,
                False,
                "reclaim_failed",
                checklist={**checklist, "reclaim_held": False},
                hard_fail="reclaim_not_held",
            )
        if held:
            score += 15
            parts.append("reclaim+15")

    mom_ok, mom_delta = momentum_6bar_aligned(k5, direction)
    checklist["momentum_ok"] = mom_ok
    if not _is_zone_setup(setup_type) and not mom_ok:
        score -= 15
        parts.append(f"momentum-15({mom_delta})")
    elif mom_ok:
        score += 10
        parts.append("momentum+10")

    rr_ok, rr_val = rr_feasible(price, direction, atr, key_levels, zone)
    checklist["rr_feasible"] = rr_ok
    if rr_ok:
        score += 15
        parts.append(f"RR+15({rr_val:.1f}:1)")
    else:
        score -= 10
        parts.append("RR-10")

    if now is not None and cfg is not None:
        asian_hi, asian_lo = compute_asian_range(now, k5, k1h or [])
        label, _, _ = compute_premium_discount(price, asian_lo, asian_hi)
        d = (direction or "").upper()
        pd_ok = (
            label is None
            or label == "equilibrium"
            or (d == "LONG" and label == "discount")
            or (d == "SHORT" and label == "premium")
        )
        checklist["premium_discount_ok"] = pd_ok
        if pd_ok and label and label != "equilibrium":
            score += 10
            parts.append(f"P/D+10({label})")
        elif not pd_ok:
            score -= 8
            parts.append(f"P/D-8({label})")

    if isinstance(smt, dict):
        mod = int(smt.get("modifier") or 0)
        if mod >= 8:
            score += 8
            parts.append("SMT+8")
            checklist["smt_confirms"] = True
        elif mod <= -8:
            score -= 8
            parts.append("SMT-8")

    score = max(0, min(100, score))
    passed = score >= readiness_min_score()
    breakdown = ", ".join(parts) if parts else "baseline"
    return ReadinessResult(
        score=score,
        passed=passed,
        breakdown=breakdown,
        checklist=checklist,
        hard_fail=None if passed else "below_threshold",
    )


# Display order for confluence checklist in Claude context.
CONFLUENCE_LABELS: Dict[str, str] = {
    "htf_aligned": "HTF",
    "at_entry": "Entry",
    "displacement_ok": "Disp",
    "reclaim_held": "Reclaim",
    "momentum_ok": "Momentum",
    "rr_feasible": "R:R",
    "premium_discount_ok": "P/D",
    "smt_confirms": "SMT",
}


def confluence_summary(checklist: Dict[str, bool]) -> Tuple[int, int, str]:
    """
    Return (passed, total, formatted line) for Claude context.

    Uses fixed label order so confluence count is comparable across setups.
    """
    items: List[str] = []
    passed = 0
    total = 0
    for key, label in CONFLUENCE_LABELS.items():
        if key not in checklist:
            continue
        total += 1
        ok = bool(checklist[key])
        if ok:
            passed += 1
        items.append(f"{label} {'✓' if ok else '✗'}")
    line = " | ".join(items) if items else "n/a"
    return passed, total, line


def format_confluence_block(
    checklist: Dict[str, bool],
    *,
    setup_type: str,
    confidence_threshold: int = 45,
) -> List[str]:
    """Explicit confluence count + calibration guidance for Claude."""
    passed, total, detail = confluence_summary(checklist or {})
    lines = [
        "=== CONFLUENCE (engine checklist — use for confidence calibration) ===",
        f"Count: {passed}/{total} passed — {detail}",
    ]
    if total == 0:
        lines.append("No checklist items recorded — score from full context.")
        return lines

    lines.append(
        "Calibration guide (align confidence with confluence count):"
    )
    if passed >= 5 and checklist.get("htf_aligned") and checklist.get("at_entry"):
        lines.append(
            f"- {passed}/{total} + HTF + entry → 75–89% range reasonable if SL/TP valid."
        )
    elif passed >= 4:
        lines.append(
            f"- {passed}/{total} passed → 60–74% solid range; TAKE if SL/TP sit at logical structure."
        )
    elif passed >= 3:
        lines.append(
            f"- {passed}/{total} passed → 50–59% tradable band if entry/stop/R:R are clean "
            f"(≥{confidence_threshold}% to execute)."
        )
    else:
        lines.append(
            f"- {passed}/{total} passed → usually 40–49% unless setup rubric floor applies "
            "(e.g. sweep + reclaim + disp)."
        )

    if setup_type.startswith(("liq_sweep_", "sweep_")):
        core = sum(
            1 for k in ("reclaim_held", "displacement_ok", "rr_feasible")
            if checklist.get(k)
        )
        if core >= 2:
            lines.append(
                f"- Sweep core {core}/3 (reclaim+disp+R:R) supports floor 50%+ even if total confluence is {passed}/{total}."
            )

    return lines


def format_readiness_block(result: ReadinessResult, setup_type: str) -> List[str]:
    """Context block for Claude — engine checklist before discretionary score."""
    passed_n, total_n, _ = confluence_summary(result.checklist)
    tier = "HIGH" if result.score >= 85 else "MODERATE" if result.score >= readiness_min_score() else "LOW"
    lines = [
        "=== SETUP READINESS (engine gate — already passed) ===",
        f"Readiness score: {result.score}/100 ({tier}) | engine min: {readiness_min_score()}",
        f"Checklist summary: {passed_n}/{total_n} — {result.breakdown}",
    ]
    if result.hard_fail:
        lines.append(f"Hard gate note: {result.hard_fail}")
    return lines
