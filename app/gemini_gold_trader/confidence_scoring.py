"""Chart confluence + confidence calibration for Gemini Gold decisions."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.gemini_gold_trader.setup_types import normalize_setup_type
from app.services.forex_sessions import active_live_forex_session

_CONFLUENCE_LABELS: Dict[str, str] = {
    "htf_aligned": "HTF",
    "ltf_trigger": "5m trigger",
    "at_structure": "At level",
    "momentum_ok": "Momentum",
    "session_active": "Session",
    "range_edge": "Range edge",
    "volatility_ok": "Volatility",
}

_FADE_SETUPS = frozenset(
    {
        "liquidity_grab_long",
        "liquidity_grab_short",
        "liq_sweep_bull",
        "liq_sweep_bear",
        "eqh_sweep_bear",
        "eql_sweep_bull",
        "sdp_bull",
        "sdp_bear",
        "disp_bull",
        "disp_bear",
        "asian_sweep_bull",
        "asian_sweep_bear",
        "sweep_pdh",
        "sweep_pdl",
    }
)


def _closes(bars: list) -> List[float]:
    out: List[float] = []
    for bar in bars:
        if len(bar) < 5:
            continue
        try:
            out.append(float(bar[4]))
        except (TypeError, ValueError):
            continue
    return out


def _atr_pips(bars: list, *, period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs: List[float] = []
    for i in range(1, len(bars)):
        try:
            h = float(bars[i][2])
            l = float(bars[i][3])
            pc = float(bars[i - 1][4])
        except (TypeError, ValueError, IndexError):
            continue
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if not trs:
        return 0.0
    window = trs[-period:]
    return sum(window) / len(window) if window else 0.0


def _trend_direction(closes: List[float], *, lookback: int = 8) -> Optional[str]:
    if len(closes) < lookback + 1:
        return None
    seg = closes[-lookback:]
    up = sum(1 for i in range(1, len(seg)) if seg[i] > seg[i - 1])
    down = sum(1 for i in range(1, len(seg)) if seg[i] < seg[i - 1])
    if up >= lookback * 0.6:
        return "LONG"
    if down >= lookback * 0.6:
        return "SHORT"
    return None


def _session_range_pct(bars_5m: list, bars_15m: list, spot: float) -> Optional[float]:
    highs: List[float] = []
    lows: List[float] = []
    for bars in (bars_5m, bars_15m):
        for bar in bars:
            if len(bar) < 4:
                continue
            try:
                highs.append(float(bar[2]))
                lows.append(float(bar[3]))
            except (TypeError, ValueError):
                continue
    if not highs or not lows or spot <= 0:
        return None
    lo = min(lows)
    hi = max(highs)
    rng = hi - lo
    if rng <= 0:
        return None
    return (spot - lo) / rng * 100.0


def compute_chart_confluence(
    *,
    bars_5m: list,
    bars_15m: list,
    bars_1h: list,
    spot: float,
    session: Optional[str] = None,
) -> Dict[str, bool]:
    """Lightweight engine checklist from OHLC (fed to Gemini + calibration)."""
    c5 = _closes(bars_5m)
    c15 = _closes(bars_15m)
    c1h = _closes(bars_1h)
    checklist: Dict[str, bool] = {}

    trend_5m = _trend_direction(c5, lookback=6)
    trend_1h = _trend_direction(c1h, lookback=8)
    checklist["htf_aligned"] = bool(
        trend_5m and trend_1h and trend_5m == trend_1h
    )

    atr = _atr_pips(bars_5m)
    body = abs(c5[-1] - float(bars_5m[-1][1])) if bars_5m and len(bars_5m[-1]) > 4 else 0.0
    checklist["ltf_trigger"] = bool(atr > 0 and body >= atr * 0.45)

    swing_hi = max(float(b[2]) for b in bars_5m[-20:] if len(b) > 2) if bars_5m else spot
    swing_lo = min(float(b[3]) for b in bars_5m[-20:] if len(b) > 3) if bars_5m else spot
    near_level = False
    if spot > 0:
        for lvl in (swing_hi, swing_lo):
            if abs(spot - lvl) / spot <= 0.0015:
                near_level = True
                break
    checklist["at_structure"] = near_level

    if len(c5) >= 4:
        moves = [c5[i] - c5[i - 1] for i in range(-3, 0)]
        same_sign = all(m > 0 for m in moves) or all(m < 0 for m in moves)
        checklist["momentum_ok"] = same_sign and any(abs(m) > 0 for m in moves)
    else:
        checklist["momentum_ok"] = False

    live_sess = active_live_forex_session(datetime.utcnow())
    checklist["session_active"] = live_sess in ("london", "new_york", "asia") or bool(session)

    pct = _session_range_pct(bars_5m, bars_15m, spot)
    checklist["range_edge"] = bool(pct is not None and (pct <= 28.0 or pct >= 72.0))

    checklist["volatility_ok"] = bool(atr >= 0.8)

    return checklist


def confluence_summary(checklist: Dict[str, bool]) -> Tuple[int, int, str]:
    items: List[str] = []
    passed = 0
    total = 0
    for key, label in _CONFLUENCE_LABELS.items():
        if key not in checklist:
            continue
        total += 1
        ok = bool(checklist[key])
        if ok:
            passed += 1
        items.append(f"{label} {'✓' if ok else '✗'}")
    return passed, total, " | ".join(items) if items else "n/a"


def confidence_band(
    passed: int,
    total: int,
    checklist: Dict[str, bool],
    setup_type: Optional[str],
) -> Tuple[int, int]:
    """Suggested confidence range from engine confluence."""
    if total <= 0:
        return (45, 100)
    st = (setup_type or "").lower()
    if passed >= 6 and checklist.get("htf_aligned") and checklist.get("ltf_trigger"):
        return (86, 97)
    if passed >= 5 and checklist.get("htf_aligned"):
        return (80, 93)
    if passed >= 4:
        return (72, 88)
    if passed >= 3:
        return (62, 78)
    if passed >= 2:
        return (50, 68)
    if st in _FADE_SETUPS and checklist.get("range_edge"):
        return (72, 90)
    return (40, 58)


def compute_rr(decision: Dict[str, Any]) -> Optional[float]:
    try:
        entry = float(decision.get("entry") or 0)
        sl = float(decision.get("stop_loss") or 0)
        tp = float(decision.get("take_profit") or 0)
        direction = str(decision.get("direction") or "").upper()
    except (TypeError, ValueError):
        return None
    if entry <= 0 or sl <= 0 or tp <= 0 or direction not in ("LONG", "SHORT"):
        return None
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0:
        return None
    return round(reward / risk, 2)


def format_confluence_block(
    checklist: Dict[str, bool],
    *,
    confidence_threshold: int,
    setup_type: Optional[str] = None,
) -> str:
    passed, total, detail = confluence_summary(checklist)
    lo, hi = confidence_band(passed, total, checklist, setup_type)
    lines = [
        "=== ENGINE CONFLUENCE (align your confidence score with this checklist) ===",
        f"Count: {passed}/{total} passed — {detail}",
        f"Suggested confidence band for a valid TAKE: {lo}–{hi}%",
        (
            f"Scores ≥{confidence_threshold}% mean you would take with real money. "
            "Use the FULL 0–100 range — do not cluster every setup at 75–80%."
        ),
    ]
    if passed >= 5:
        lines.append(
            f"- {passed}/{total} + HTF/trigger → 86–95% is appropriate when SL/TP are clean."
        )
    elif passed >= 4:
        lines.append(f"- {passed}/{total} → 72–88% solid range for named scalp with live 5m trigger.")
    elif passed >= 3:
        lines.append(f"- {passed}/{total} → 62–78% unless setup rubric supports higher.")
    return "\n".join(lines)


def confidence_scoring_prompt_block(*, confidence_threshold: int) -> str:
    t = max(0, min(100, int(confidence_threshold)))
    return (
        "CONFIDENCE SCORING (calibrate honestly — use the full 0–100 scale):\n"
        "- 92–100: A+ exceptional — 6/7 confluence, HTF+5m aligned, pristine entry at structure, 1.5–2R+\n"
        "- 86–91: High-conviction A — 5+ confluence, live 5m trigger, named setup, SL/TP coherent\n"
        "- 78–85: Solid A- — 4+ confluence, clear scalp pattern, good (not perfect) location\n"
        "- 65–77: Tradable B+ — 3+ confluence, edge exists but one factor missing\n"
        "- 50–64: Marginal — borderline; SKIP unless edge is obvious\n"
        "- Below 50: No trade / chop / swing-style setup\n"
        f"A score ≥{t}% means: you would fire this scalp with real money.\n"
        "Do NOT cap strong setups at 80% — when confluence is 5/7+ with live trigger, use 86–95%.\n"
        "Match your number to ENGINE CONFLUENCE when provided — do not score 72% when checklist says 86–93% band."
    )


def calibrate_confidence(
    decision: Dict[str, Any],
    *,
    checklist: Dict[str, bool],
    confidence_threshold: int,
) -> Tuple[int, Dict[str, Any]]:
    """Blend model confidence with engine confluence so scores reflect setup quality."""
    try:
        model = int(decision.get("confidence") or 0)
    except (TypeError, ValueError):
        model = 0
    model = max(0, min(100, model))
    action = str(decision.get("action") or "SKIP").upper()
    setup_type = normalize_setup_type(
        decision.get("setup_type"),
        decision.get("direction"),
    )

    passed, total, _ = confluence_summary(checklist)
    lo, hi = confidence_band(passed, total, checklist, setup_type)
    rr = compute_rr(decision) if action == "TAKE" else None

    calibrated = model
    if action == "TAKE":
        if passed >= 5 and model < lo:
            calibrated = max(calibrated, lo)
        if passed >= 6 and checklist.get("htf_aligned") and checklist.get("ltf_trigger"):
            calibrated = max(calibrated, 86)
        if passed >= 6 and rr is not None and rr >= 1.75:
            calibrated = max(calibrated, 90)
        if setup_type in _FADE_SETUPS and checklist.get("range_edge") and checklist.get("ltf_trigger"):
            calibrated = max(calibrated, 78)
        if rr is not None:
            if rr >= 1.9:
                calibrated = min(97, max(calibrated, calibrated + 2))
            elif rr < 0.95:
                calibrated = min(calibrated, min(hi, 72))
        if passed <= 2 and model > hi:
            calibrated = min(model, hi)
        if passed <= 1:
            calibrated = min(calibrated, 55)

    calibrated = max(0, min(100, int(calibrated)))
    meta = {
        "model_confidence": model,
        "confluence_passed": passed,
        "confluence_total": total,
        "confidence_band": [lo, hi],
        "risk_reward": rr,
        "calibrated": calibrated != model,
    }
    return calibrated, meta
