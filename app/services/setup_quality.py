"""
Shared setup quality engine for tradfi discovery scanners (gold / forex / index).

Single entry point ``grade_setup`` grades each discovery result with:
- Walk-forward TEST-split backtest stats (not train) for base scoring
- Six confirmation checks (MTF, reaction candle, sweep, session, momentum, zone)
- Confluence stacking count
- A–F letter grade with strict quality floor
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Scoring constants (shared across all discovery scanners) ─────────────────
BASE_WIN_RATE_WEIGHT = 40.0
BASE_AVG_PIPS_WEIGHT = 30.0
BASE_CONSISTENCY_WEIGHT = 30.0
CONFIRMATION_BOOST = 8.0          # per confirmation beyond minimum required
CONFLUENCE_BOOST = 5.0            # per confluence beyond minimum required
MAX_SCORE = 100.0

GRADE_A = 85
GRADE_B = 72
GRADE_C = 60
GRADE_D = 45

DEFAULT_QUALITY_CFG: Dict[str, Any] = {
    "min_confirmations": 3,
    "min_winrate": 0.58,
    "min_confluences": 2,
    "min_trades": 20,
    "quality_mode": "strict",  # strict | all
    "allowed_sessions": ["london_kz", "ny_kz"],
}

_TF_LADDER = ["1m", "5m", "15m", "1h", "4h"]

_CONFIRMATION_LABELS = {
    "mtf_agreement": "MTF",
    "reaction_candle": "Reaction",
    "liquidity_sweep_before": "Sweep",
    "in_session": "Session",
    "momentum_align": "Momentum",
    "zone_confluence": "Zone",
}

_CONFLUENCE_CHECKS: List[Tuple[str, Dict]] = [
    ("FVG", {"type": "fvg", "fvg_dir": "bullish", "min_gap_pct": 0.08}),
    ("OB", {"type": "order_block", "ob_type": "bullish"}),
    ("IFVG", {"type": "ifvg", "direction": "bullish", "min_gap_pct": 0.15}),
    ("SDP", {"type": "supply_demand", "direction": "bullish"}),
    ("BRK", {"type": "breaker_block", "direction": "bullish"}),
    ("PD", {"type": "premium_discount", "zone": "discount"}),
]


def normalize_quality_cfg(cfg: Optional[Dict] = None) -> Dict[str, Any]:
    """Merge caller cfg with defaults; coerce types."""
    out = dict(DEFAULT_QUALITY_CFG)
    if cfg:
        out.update({k: v for k, v in cfg.items() if v is not None})
    try:
        out["min_confirmations"] = max(0, int(out.get("min_confirmations", 3)))
    except (TypeError, ValueError):
        out["min_confirmations"] = 3
    try:
        wr = float(out.get("min_winrate", 0.58))
        out["min_winrate"] = wr if wr <= 1.0 else wr / 100.0
    except (TypeError, ValueError):
        out["min_winrate"] = 0.58
    try:
        out["min_confluences"] = max(0, int(out.get("min_confluences", 2)))
    except (TypeError, ValueError):
        out["min_confluences"] = 2
    try:
        out["min_trades"] = max(1, int(out.get("min_trades", 20)))
    except (TypeError, ValueError):
        out["min_trades"] = 20
    mode = str(out.get("quality_mode") or "strict").lower()
    out["quality_mode"] = mode if mode in ("strict", "all") else "strict"
    sess = out.get("allowed_sessions")
    if not isinstance(sess, list) or not sess:
        out["allowed_sessions"] = list(DEFAULT_QUALITY_CFG["allowed_sessions"])
    return out


def _tf_minutes(tf: str) -> int:
    tf = (tf or "15m").lower()
    if tf.endswith("m"):
        return int(tf[:-1] or 15)
    if tf.endswith("h"):
        return int(tf[:-1] or 1) * 60
    return 15


def _higher_tf(entry_tf: str) -> Optional[str]:
    try:
        idx = _TF_LADDER.index(entry_tf.lower())
    except ValueError:
        return None
    if idx + 1 < len(_TF_LADDER):
        return _TF_LADDER[idx + 1]
    return None


def _is_long(direction: str) -> bool:
    return (direction or "LONG").upper() == "LONG"


def _flip_cond_for_direction(cond: Dict, direction: str) -> Dict:
    """Mirror bullish confluence checks for SHORT direction."""
    c = dict(cond)
    if _is_long(direction):
        return c
    t = c.get("type", "")
    if t == "fvg":
        c["fvg_dir"] = "bearish"
    elif t == "order_block":
        c["ob_type"] = "bearish"
    elif t == "premium_discount":
        c["zone"] = "premium"
    elif t in ("ifvg", "breaker_block", "supply_demand", "liquidity_sweep",
               "mss", "choch", "pin_bar", "engulfing", "inside_bar"):
        c["direction"] = "bearish"
    return c


def _klines_up_to(candles: List, entry_ts_ms: Optional[int] = None) -> List:
    if not candles:
        return []
    if not entry_ts_ms:
        return candles
    out = [k for k in candles if int(k[0]) <= int(entry_ts_ms)]
    return out if len(out) >= 20 else candles


def _entry_ts_from_signal(signal: Dict) -> Optional[int]:
    ts = signal.get("entry_ts")
    if ts:
        return int(ts)
    trades = signal.get("test_trades_list") or signal.get("trades") or []
    if trades:
        try:
            return int(trades[-1].get("entry_ts", 0))
        except (TypeError, ValueError):
            pass
    return None


def _check_mtf_agreement(
    candles_by_tf: Dict[str, List],
    entry_tf: str,
    direction: str,
) -> bool:
    htf = _higher_tf(entry_tf)
    if not htf or htf not in candles_by_tf:
        return False
    klines = candles_by_tf[htf]
    if len(klines) < 30:
        return False
    from app.services.backtest_engine import eval_condition_bt
    ptype = "hh_hl" if _is_long(direction) else "lh_ll"
    return bool(eval_condition_bt({"type": ptype}, klines, _tf_minutes(htf)))


def _check_reaction_candle(klines: List, direction: str) -> bool:
    if len(klines) < 3:
        return False
    c = klines[-1]
    o, h, l, cl = float(c[1]), float(c[2]), float(c[3]), float(c[4])
    rng = h - l
    if rng <= 0:
        return False
    if _is_long(direction):
        lower_wick = min(o, cl) - l
        return lower_wick >= 0.5 * rng and cl >= o
    upper_wick = h - max(o, cl)
    return upper_wick >= 0.5 * rng and cl <= o


def _check_liquidity_sweep_before(klines: List, direction: str) -> bool:
    if len(klines) < 14:
        return False
    from app.services.backtest_engine import eval_condition_bt
    d = "bullish" if _is_long(direction) else "bearish"
    # Evaluate on prior bar context (sweep then signal bar)
    ctx = klines[:-1]
    if len(ctx) < 12:
        return False
    return bool(eval_condition_bt(
        {"type": "liquidity_sweep", "direction": d},
        ctx + [klines[-1]],
        5,
    ))


def _check_in_session(entry_ts_ms: Optional[int], cfg: Dict) -> bool:
    if not entry_ts_ms:
        return False
    try:
        dt = datetime.fromtimestamp(int(entry_ts_ms) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return False
    from app.services.session_filter import is_in_allowed_session
    sess_cfg = {
        "sessions_enabled": True,
        "allowed_sessions": cfg.get("allowed_sessions") or DEFAULT_QUALITY_CFG["allowed_sessions"],
    }
    ok, _ = is_in_allowed_session(sess_cfg, dt)
    return ok


def _check_momentum_align(klines: List, direction: str, interval_min: int) -> bool:
    if len(klines) < 25:
        return False
    from app.services.backtest_engine import eval_condition_bt
    if _is_long(direction):
        rsi_ok = eval_condition_bt(
            {"type": "rsi", "period": 14, "operator": "gt", "value": 45},
            klines, interval_min,
        )
        ema_ok = eval_condition_bt(
            {"type": "ema", "period": 9, "condition": "above"},
            klines, interval_min,
        )
        return rsi_ok or ema_ok
    rsi_ok = eval_condition_bt(
        {"type": "rsi", "period": 14, "operator": "lt", "value": 55},
        klines, interval_min,
    )
    ema_ok = eval_condition_bt(
        {"type": "ema", "period": 9, "condition": "below"},
        klines, interval_min,
    )
    return rsi_ok or ema_ok


def _check_zone_confluence(klines: List, direction: str, interval_min: int) -> bool:
    if len(klines) < 20:
        return False
    from app.services.backtest_engine import eval_condition_bt
    d = "bullish" if _is_long(direction) else "bearish"
    zone = "discount" if _is_long(direction) else "premium"
    checks = [
        eval_condition_bt({"type": "fvg", "fvg_dir": d, "min_gap_pct": 0.08}, klines, interval_min),
        eval_condition_bt({"type": "order_block", "ob_type": d}, klines, interval_min),
        eval_condition_bt({"type": "premium_discount", "zone": zone}, klines, interval_min),
    ]
    return any(checks)


def _count_confluences(klines: List, direction: str, interval_min: int) -> List[str]:
    from app.services.backtest_engine import eval_condition_bt
    hits: List[str] = []
    for label, cond in _CONFLUENCE_CHECKS:
        c = _flip_cond_for_direction(cond, direction)
        try:
            if eval_condition_bt(c, klines, interval_min):
                hits.append(label)
        except Exception:
            pass
    return hits


def _consistency_score(trades: List[Dict], min_trades: int) -> float:
    """0–1 score: enough trades + low variance in pip outcomes."""
    decided = [
        t for t in (trades or [])
        if t.get("outcome") in ("WIN", "LOSS")
    ]
    n = len(decided)
    if n < min_trades:
        return max(0.0, n / float(min_trades)) * 0.35
    pips = [float(t.get("pip_move") or 0) for t in decided]
    if not pips:
        return 0.0
    mean = sum(pips) / len(pips)
    var = sum((p - mean) ** 2 for p in pips) / len(pips)
    std = math.sqrt(var)
    # Lower relative std = higher consistency; cap at 1.0
    denom = abs(mean) + std + 1.0
    rel = std / denom if denom else 1.0
    return max(0.0, min(1.0, 1.0 - rel))


def _base_score_from_stats(
    stats: Dict,
    trades: List[Dict],
    min_trades: int,
) -> Tuple[float, List[str]]:
    """Compute 0–100 base from TEST-split stats (win rate, avg pips, consistency)."""
    reasons: List[str] = []
    n = int(stats.get("closed_trades") or 0)
    if n < min_trades:
        reasons.append(f"Only {n} test trades (need {min_trades})")
        return 0.0, reasons

    wr = float(stats.get("win_rate") or 0) / 100.0
    wr_comp = min(1.0, wr / 0.75) * BASE_WIN_RATE_WEIGHT
    reasons.append(f"Test win rate {stats.get('win_rate')}% → {wr_comp:.1f} pts")

    avg_pips = float(stats.get("avg_pips") or 0)
    # Normalize: 5 pips avg ≈ full weight for FX; scale indices similarly
    pip_norm = max(0.0, min(1.0, (avg_pips + 2.0) / 12.0))
    pip_comp = pip_norm * BASE_AVG_PIPS_WEIGHT
    reasons.append(f"Avg pips/trade {avg_pips:.1f} → {pip_comp:.1f} pts")

    cons = _consistency_score(trades, min_trades)
    cons_comp = cons * BASE_CONSISTENCY_WEIGHT
    reasons.append(f"Consistency {cons:.2f} → {cons_comp:.1f} pts")

    return round(wr_comp + pip_comp + cons_comp, 2), reasons


def _letter_grade(score: float) -> str:
    if score >= GRADE_A:
        return "A"
    if score >= GRADE_B:
        return "B"
    if score >= GRADE_C:
        return "C"
    if score >= GRADE_D:
        return "D"
    return "F"


def grade_setup(
    candles_by_tf: Dict[str, List],
    signal: Dict,
    symbol: str,
    cfg: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Grade a discovery setup.

    ``signal`` should include direction, timeframe, stats (preferably TEST split),
    and optional test_trades_list / entry_ts for confirmation context.
    """
    qcfg = normalize_quality_cfg(cfg)
    direction = str(signal.get("direction") or "LONG").upper()
    entry_tf = str(signal.get("timeframe") or "15m").lower()
    interval_min = _tf_minutes(entry_tf)

    entry_ts = _entry_ts_from_signal(signal)
    entry_kl = candles_by_tf.get(entry_tf) or []
    ctx_kl = _klines_up_to(entry_kl, entry_ts)
    if len(ctx_kl) < 20:
        ctx_kl = entry_kl[-80:] if entry_kl else []

    confirmations: List[str] = []
    confirm_keys: List[str] = []
    if _check_mtf_agreement(candles_by_tf, entry_tf, direction):
        confirmations.append(_CONFIRMATION_LABELS["mtf_agreement"])
        confirm_keys.append("mtf_agreement")
    if _check_reaction_candle(ctx_kl, direction):
        confirmations.append(_CONFIRMATION_LABELS["reaction_candle"])
        confirm_keys.append("reaction_candle")
    if _check_liquidity_sweep_before(ctx_kl, direction):
        confirmations.append(_CONFIRMATION_LABELS["liquidity_sweep_before"])
        confirm_keys.append("liquidity_sweep_before")
    if _check_in_session(entry_ts, qcfg):
        confirmations.append(_CONFIRMATION_LABELS["in_session"])
        confirm_keys.append("in_session")
    if _check_momentum_align(ctx_kl, direction, interval_min):
        confirmations.append(_CONFIRMATION_LABELS["momentum_align"])
        confirm_keys.append("momentum_align")
    if _check_zone_confluence(ctx_kl, direction, interval_min):
        confirmations.append(_CONFIRMATION_LABELS["zone_confluence"])
        confirm_keys.append("zone_confluence")

    confluences = _count_confluences(ctx_kl, direction, interval_min)

    # Prefer explicit test_stats; fall back to stats with test_* fields
    test_stats = signal.get("test_stats")
    if not test_stats:
        st = signal.get("stats") or {}
        if st.get("test_trades") is not None:
            test_stats = {
                "closed_trades": st.get("test_trades", 0),
                "win_rate": st.get("test_win_rate", 0),
                "avg_pips": st.get("test_avg_pips", st.get("avg_pips", 0)),
                "total_pips": st.get("test_total_pips", 0),
            }
        else:
            test_stats = st

    test_trades = signal.get("test_trades_list") or []
    base, base_reasons = _base_score_from_stats(
        test_stats or {},
        test_trades,
        qcfg["min_trades"],
    )

    n_conf = len(confirm_keys)
    n_confl = len(confluences)
    boost = 0.0
    boost_reasons: List[str] = []
    if n_conf > qcfg["min_confirmations"]:
        extra = n_conf - qcfg["min_confirmations"]
        boost += extra * CONFIRMATION_BOOST
        boost_reasons.append(f"+{extra * CONFIRMATION_BOOST:.0f} ({extra} extra confirmations)")
    if n_confl > qcfg["min_confluences"]:
        extra_c = n_confl - qcfg["min_confluences"]
        boost += extra_c * CONFLUENCE_BOOST
        boost_reasons.append(f"+{extra_c * CONFLUENCE_BOOST:.0f} ({extra_c} extra confluences)")

    score = round(min(MAX_SCORE, base + boost), 1)
    grade = _letter_grade(score)

    wr_frac = float(test_stats.get("win_rate") or 0) / 100.0 if test_stats else 0.0
    passed = (
        n_conf >= qcfg["min_confirmations"]
        and wr_frac >= qcfg["min_winrate"]
        and n_confl >= qcfg["min_confluences"]
        and int(test_stats.get("closed_trades") or 0) >= qcfg["min_trades"]
        and grade in ("A", "B", "C")
    )

    reasons = list(base_reasons) + boost_reasons
    if n_conf < qcfg["min_confirmations"]:
        reasons.append(f"Need {qcfg['min_confirmations']} confirmations (have {n_conf})")
    if wr_frac < qcfg["min_winrate"]:
        reasons.append(f"Test win rate {wr_frac:.1%} below {qcfg['min_winrate']:.1%}")
    if n_confl < qcfg["min_confluences"]:
        reasons.append(f"Need {qcfg['min_confluences']} confluences (have {n_confl})")

    logger.info(
        "[scan] grade %s %s %s %s tf=%s score=%.1f grade=%s conf=%d confl=%d passed=%s",
        symbol, signal.get("label", "?"), direction, entry_tf,
        score, grade, n_conf, n_confl, passed,
    )

    return {
        "score": score,
        "grade": grade,
        "confirmations": confirmations,
        "confirmation_keys": confirm_keys,
        "confluences": confluences,
        "confluence_count": n_confl,
        "passed": passed,
        "reasons": reasons,
        "base_score": base,
        "boost": boost,
    }


def apply_quality_grades(
    rows: List[Dict],
    candle_map: Dict[str, List],
    symbol: str,
    quality_cfg: Optional[Dict] = None,
    *,
    wf_splits: Optional[Dict[str, Optional[int]]] = None,
) -> List[Dict]:
    """
    Attach grade_setup output to each discovery row; filter in strict mode.
    Ensures walk-forward TEST stats are used for grading.
    """
    qcfg = normalize_quality_cfg(quality_cfg)
    out: List[Dict] = []

    for row in rows:
        enriched = dict(row)
        tf = row.get("timeframe", "15m")
        split_ts = (wf_splits or {}).get(tf)
        test_trades_list: List[Dict] = []
        test_stats: Optional[Dict] = None

        if split_ts and row.get("_all_trades"):
            test_trades_list = [
                t for t in row["_all_trades"]
                if int(t.get("entry_ts", 0) or 0) >= split_ts
            ]
            from app.services.gold_strategy_scanner import _bucket_stats
            from app.services.forex_engine import pip_size as _pip_size_fn
            pip_sz = _pip_size_fn(symbol)
            sess = row.get("session", "all")
            test_stats = _bucket_stats(test_trades_list, sess, pip_sz)
            if test_stats:
                enriched["test_stats"] = test_stats
                enriched["test_trades_list"] = test_trades_list
                if test_trades_list:
                    enriched["entry_ts"] = test_trades_list[-1].get("entry_ts")
                st = dict(row.get("stats") or {})
                st["test_trades"] = test_stats.get("closed_trades", 0)
                st["test_win_rate"] = test_stats.get("win_rate", 0)
                st["test_pnl"] = test_stats.get("total_pnl", 0)
                st["test_avg_pips"] = test_stats.get("avg_pips", 0)
                enriched["stats"] = st
        else:
            st = row.get("stats") or {}
            if st.get("test_trades"):
                test_stats = {
                    "closed_trades": st.get("test_trades"),
                    "win_rate": st.get("test_win_rate", 0),
                    "avg_pips": st.get("test_avg_pips", st.get("avg_pips", 0)),
                }
                enriched["test_stats"] = test_stats

        if split_ts:
            train_n = int((row.get("stats") or {}).get("closed_trades") or 0)
            test_n = int((test_stats or {}).get("closed_trades") or 0)
            logger.info(
                "[scan] walk-forward: train=%s test=%s symbol=%s label=%s tf=%s",
                train_n, test_n, symbol, row.get("label"), tf,
            )

        g = grade_setup(candle_map, enriched, symbol, qcfg)
        enriched.update({
            "grade": g["grade"],
            "score": g["score"],
            "quality_score": g["score"],
            "confirmations": g["confirmations"],
            "confluences": g["confluences"],
            "confluence_count": g["confluence_count"],
            "passed": g["passed"],
            "reasons": g["reasons"],
            "legacy_score": row.get("score"),
        })
        enriched.pop("_all_trades", None)

        if qcfg["quality_mode"] == "strict" and not g["passed"]:
            continue
        out.append(enriched)

    out.sort(key=lambda r: r.get("score", 0), reverse=True)
    return out


# Signal categories exposed to UI multi-select (matches discovery roster)
DISCOVERY_CATEGORIES = [
    "Trend", "Momentum", "Mean Rev", "Volatility", "Breakout",
    "Price Action", "Divergence", "Smart Money", "ICT", "Combo",
    "Supply/Demand", "Structure",
]
