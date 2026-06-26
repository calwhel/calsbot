"""Trendline detector for Gold AI scanner (bounce + break)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app.gold_ai_trader.call_gates import atr_from_klines
from app.services.tradfi_prices import get_klines

logger = logging.getLogger(__name__)


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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _ts_iso(ts_raw) -> str:
    try:
        ts = int(ts_raw)
        dt = datetime.utcfromtimestamp(ts / 1000 if ts > 1e10 else ts)
        return dt.replace(microsecond=0).isoformat() + "Z"
    except Exception:
        return "n/a"


@dataclass
class Pivot:
    idx: int
    price: float
    ts: int


@dataclass
class Trendline:
    side: str  # support|resistance
    p1: Pivot
    p2: Pivot
    slope: float
    slope_atr_per_bar: float
    touch_count: int
    fit_error_atr: float
    line_price_now: float
    line_price_prev: float
    age_bars: int
    breaks_before_last: int
    score: float


def _line_price(p1: Pivot, slope: float, idx: int) -> float:
    return float(p1.price + slope * (idx - p1.idx))


def _pivot_highs(rows: List[list], left: int, right: int) -> List[Pivot]:
    highs = [float(r[2]) for r in rows]
    pivots: List[Pivot] = []
    for i in range(left, len(rows) - right):
        hi = highs[i]
        left_side = highs[i - left : i]
        right_side = highs[i + 1 : i + right + 1]
        if not left_side or not right_side:
            continue
        if hi != max(highs[i - left : i + right + 1]):
            continue
        if hi <= max(left_side) or hi <= max(right_side):
            continue
        pivots.append(Pivot(idx=i, price=hi, ts=int(rows[i][0])))
    return pivots


def _pivot_lows(rows: List[list], left: int, right: int) -> List[Pivot]:
    lows = [float(r[3]) for r in rows]
    pivots: List[Pivot] = []
    for i in range(left, len(rows) - right):
        lo = lows[i]
        left_side = lows[i - left : i]
        right_side = lows[i + 1 : i + right + 1]
        if not left_side or not right_side:
            continue
        if lo != min(lows[i - left : i + right + 1]):
            continue
        if lo >= min(left_side) or lo >= min(right_side):
            continue
        pivots.append(Pivot(idx=i, price=lo, ts=int(rows[i][0])))
    return pivots


def _line_break_count(
    side: str,
    *,
    rows: List[list],
    p1: Pivot,
    slope: float,
    start_idx: int,
    end_idx: int,
    invalidate_buf: float,
) -> int:
    closes = [float(r[4]) for r in rows]
    n = len(rows)
    s = max(0, min(start_idx, n - 1))
    e = max(0, min(end_idx, n - 1))
    if s > e:
        return 0
    count = 0
    for i in range(s, e + 1):
        line_px = _line_price(p1, slope, i)
        c = closes[i]
        if side == "support" and c < (line_px - invalidate_buf):
            count += 1
        elif side == "resistance" and c > (line_px + invalidate_buf):
            count += 1
    return count


def _build_lines(
    *,
    side: str,
    pivots: List[Pivot],
    rows: List[list],
    atr: float,
    min_gap_bars: int,
    min_touches: int,
    touch_tol: float,
    slope_min_atr_per_bar: float,
    slope_max_atr_per_bar: float,
    fit_error_max_atr: float,
    max_age_bars: int,
    max_dist_from_price_atr: float,
    max_active_per_side: int,
    invalidate_buf: float,
    current_price: float,
) -> List[Trendline]:
    if atr <= 0 or len(rows) < 10 or len(pivots) < min_touches:
        return []

    last_idx = len(rows) - 1
    lines: List[Trendline] = []
    for i in range(len(pivots) - 1):
        for j in range(i + 1, len(pivots)):
            p1 = pivots[i]
            p2 = pivots[j]
            gap = p2.idx - p1.idx
            if gap < min_gap_bars:
                continue
            slope = (p2.price - p1.price) / float(gap)
            slope_atr = abs(slope) / max(atr, 1e-9)
            if slope_atr < slope_min_atr_per_bar or slope_atr > slope_max_atr_per_bar:
                continue

            touch_diffs = []
            touches = 0
            for p in pivots:
                if p.idx < p1.idx:
                    continue
                lp = _line_price(p1, slope, p.idx)
                diff = abs(p.price - lp)
                if diff <= touch_tol:
                    touches += 1
                    touch_diffs.append(diff)
            if touches < min_touches:
                continue

            fit_error = (sum(touch_diffs) / len(touch_diffs)) / max(atr, 1e-9)
            if fit_error > fit_error_max_atr:
                continue

            age_bars = max(0, last_idx - p2.idx)
            if age_bars > max_age_bars:
                continue

            line_now = _line_price(p1, slope, last_idx)
            line_prev = _line_price(p1, slope, max(0, last_idx - 1))
            dist_atr = abs(current_price - line_now) / max(atr, 1e-9)
            if dist_atr > max_dist_from_price_atr:
                continue

            breaks_before_last = _line_break_count(
                side,
                rows=rows,
                p1=p1,
                slope=slope,
                start_idx=p2.idx + 1,
                end_idx=max(0, last_idx - 1),
                invalidate_buf=invalidate_buf,
            )

            score = (touches * 12.0) - (fit_error * 20.0) - (age_bars * 0.2) - (
                breaks_before_last * 8.0
            )
            lines.append(
                Trendline(
                    side=side,
                    p1=p1,
                    p2=p2,
                    slope=slope,
                    slope_atr_per_bar=slope_atr,
                    touch_count=touches,
                    fit_error_atr=fit_error,
                    line_price_now=line_now,
                    line_price_prev=line_prev,
                    age_bars=age_bars,
                    breaks_before_last=breaks_before_last,
                    score=score,
                )
            )

    lines.sort(key=lambda l: l.score, reverse=True)
    active = lines[: max(1, max_active_per_side)]
    for ln in active:
        logger.info(
            "[gold-ai-trendline] validated side=%s a1=(%s,%.2f) a2=(%s,%.2f) "
            "slope=%.6f slope_atr=%.3f touches=%s fit_error_atr=%.3f line_now=%.2f breaks=%s",
            ln.side,
            _ts_iso(ln.p1.ts),
            ln.p1.price,
            _ts_iso(ln.p2.ts),
            ln.p2.price,
            ln.slope,
            ln.slope_atr_per_bar,
            ln.touch_count,
            ln.fit_error_atr,
            ln.line_price_now,
            ln.breaks_before_last,
        )
    return active


def _trendline_meta(
    *,
    mode: str,
    direction: str,
    line: Trendline,
    line_level: float,
    distance_to_line_atr: float,
    used_retest: bool,
    body_atr: float,
) -> Dict:
    return {
        "trendline_mode": mode,
        "trendline_side": line.side,
        "trendline_level": float(line_level),
        "trendline_timeframe": os.environ.get("GOLD_AI_TRENDLINE_TF", "15m"),
        "trendline_anchor_1_ts": _ts_iso(line.p1.ts),
        "trendline_anchor_1_price": float(line.p1.price),
        "trendline_anchor_2_ts": _ts_iso(line.p2.ts),
        "trendline_anchor_2_price": float(line.p2.price),
        "trendline_slope": float(line.slope),
        "trendline_slope_atr_per_bar": float(line.slope_atr_per_bar),
        "trendline_touch_count": int(line.touch_count),
        "trendline_fit_error_atr": float(line.fit_error_atr),
        "trendline_line_price_now": float(line.line_price_now),
        "trendline_distance_to_line_atr": float(distance_to_line_atr),
        "trendline_direction": direction,
        "trendline_used_retest": bool(used_retest),
        "trendline_signal_body_atr": float(body_atr),
    }


async def eval_fx_trendline(
    cond: Dict,
    symbol: str,
    current_price: float,
    _http_client,
    cache: Dict,
) -> Tuple[bool, str, Dict]:
    """
    Auto-detect trendline bounce/break setups.
    Returns (ok, message, metadata).
    """
    mode = (cond.get("mode") or "bounce").strip().lower()
    direction = (cond.get("direction") or "bullish").strip().lower()
    line_side = (cond.get("line_side") or "support").strip().lower()
    tf = (cond.get("timeframe") or os.environ.get("GOLD_AI_TRENDLINE_TF", "15m")).strip().lower()

    lookback = max(80, _env_int("GOLD_AI_TRENDLINE_LOOKBACK_BARS", 160))
    pivot_left = max(1, _env_int("GOLD_AI_TRENDLINE_PIVOT_LEFT", 2))
    pivot_right = max(1, _env_int("GOLD_AI_TRENDLINE_PIVOT_RIGHT", 2))
    min_gap_bars = max(2, _env_int("GOLD_AI_TRENDLINE_MIN_PIVOT_GAP_BARS", 4))
    min_touches = max(3, _env_int("GOLD_AI_TRENDLINE_MIN_TOUCHES", 3))
    touch_tol_atr = max(0.01, _env_float("GOLD_AI_TRENDLINE_TOUCH_TOL_ATR", 0.15))
    min_touch_abs = max(0.01, _env_float("GOLD_AI_TRENDLINE_MIN_TOUCH_PRICE_ABS", 0.02))
    slope_min_atr = max(0.001, _env_float("GOLD_AI_TRENDLINE_SLOPE_MIN_ATR_PER_BAR", 0.03))
    slope_max_atr = max(slope_min_atr + 0.01, _env_float("GOLD_AI_TRENDLINE_SLOPE_MAX_ATR_PER_BAR", 1.20))
    fit_error_max_atr = max(0.05, _env_float("GOLD_AI_TRENDLINE_FIT_ERROR_MAX_ATR", 0.35))
    max_age_bars = max(20, _env_int("GOLD_AI_TRENDLINE_MAX_AGE_BARS", 96))
    max_dist_atr = max(0.5, _env_float("GOLD_AI_TRENDLINE_MAX_DIST_FROM_PRICE_ATR", 3.5))
    max_active = max(1, _env_int("GOLD_AI_TRENDLINE_MAX_ACTIVE_PER_SIDE", 2))
    max_breaks_before_expire = max(0, _env_int("GOLD_AI_TRENDLINE_MAX_BREAKS_BEFORE_EXPIRE", 1))
    invalidate_buf_atr = max(0.01, _env_float("GOLD_AI_TRENDLINE_BREAK_INVALIDATE_BUF_ATR", 0.08))

    bounce_reclaim_buf_atr = max(0.01, _env_float("GOLD_AI_TRENDLINE_BOUNCE_RECLAIM_BUF_ATR", 0.05))
    bounce_min_body_atr = max(0.05, _env_float("GOLD_AI_TRENDLINE_BOUNCE_MIN_BODY_ATR", 0.30))
    bounce_min_wick_atr = max(0.0, _env_float("GOLD_AI_TRENDLINE_BOUNCE_MIN_WICK_ATR", 0.08))

    break_close_buf_atr = max(0.01, _env_float("GOLD_AI_TRENDLINE_BREAK_CLOSE_BUF_ATR", 0.12))
    break_min_body_atr = max(0.1, _env_float("GOLD_AI_TRENDLINE_BREAK_MIN_BODY_ATR", 0.55))
    break_require_retest = _env_bool("GOLD_AI_TRENDLINE_BREAK_REQUIRE_RETEST", True)
    break_retest_max_bars = max(1, _env_int("GOLD_AI_TRENDLINE_BREAK_RETEST_MAX_BARS", 4))
    break_retest_tol_atr = max(0.01, _env_float("GOLD_AI_TRENDLINE_BREAK_RETEST_TOL_ATR", 0.12))

    asset_class = (cache or {}).get("__asset_class__", "forex")
    klines = await get_klines(symbol, asset_class, tf, lookback + 20)
    if not klines or len(klines) < 40:
        return False, "trendline:insufficient_data", {}
    rows = klines[:-1]
    if len(rows) < 30:
        return False, "trendline:insufficient_closed_bars", {}
    closes = [float(r[4]) for r in rows]
    opens = [float(r[1]) for r in rows]
    highs = [float(r[2]) for r in rows]
    lows = [float(r[3]) for r in rows]
    atr = atr_from_klines(klines[-80:])
    if atr <= 0:
        return False, "trendline:atr_unavailable", {}

    touch_tol = max(touch_tol_atr * atr, min_touch_abs)
    invalidate_buf = invalidate_buf_atr * atr
    cur = float(current_price or closes[-1])

    lows_piv = _pivot_lows(rows, pivot_left, pivot_right)
    highs_piv = _pivot_highs(rows, pivot_left, pivot_right)

    supports = _build_lines(
        side="support",
        pivots=lows_piv,
        rows=rows,
        atr=atr,
        min_gap_bars=min_gap_bars,
        min_touches=min_touches,
        touch_tol=touch_tol,
        slope_min_atr_per_bar=slope_min_atr,
        slope_max_atr_per_bar=slope_max_atr,
        fit_error_max_atr=fit_error_max_atr,
        max_age_bars=max_age_bars,
        max_dist_from_price_atr=max_dist_atr,
        max_active_per_side=max_active,
        invalidate_buf=invalidate_buf,
        current_price=cur,
    )
    resistances = _build_lines(
        side="resistance",
        pivots=highs_piv,
        rows=rows,
        atr=atr,
        min_gap_bars=min_gap_bars,
        min_touches=min_touches,
        touch_tol=touch_tol,
        slope_min_atr_per_bar=slope_min_atr,
        slope_max_atr_per_bar=slope_max_atr,
        fit_error_max_atr=fit_error_max_atr,
        max_age_bars=max_age_bars,
        max_dist_from_price_atr=max_dist_atr,
        max_active_per_side=max_active,
        invalidate_buf=invalidate_buf,
        current_price=cur,
    )
    active = supports if line_side == "support" else resistances
    if not active:
        return False, f"trendline:no_active_{line_side}_lines", {}

    line = active[0]
    if line.breaks_before_last > max_breaks_before_expire:
        return False, f"trendline:line_expired_breaks({line.breaks_before_last})", {}

    last_idx = len(rows) - 1
    prev_idx = max(0, last_idx - 1)
    line_last = line.line_price_now
    line_prev = line.line_price_prev
    o_last = opens[last_idx]
    h_last = highs[last_idx]
    l_last = lows[last_idx]
    c_last = closes[last_idx]
    body_atr = abs(c_last - o_last) / max(atr, 1e-9)
    dist_now_atr = abs(c_last - line_last) / max(atr, 1e-9)

    if mode == "bounce":
        if direction == "bullish" and line.side == "support":
            lower_wick = min(o_last, c_last) - l_last
            touched = l_last <= (line_last + touch_tol)
            held = c_last >= (line_last + (bounce_reclaim_buf_atr * atr))
            approach = closes[prev_idx] >= (line_prev - touch_tol)
            if (
                touched
                and held
                and approach
                and c_last > o_last
                and body_atr >= bounce_min_body_atr
                and lower_wick >= (bounce_min_wick_atr * atr)
            ):
                meta = _trendline_meta(
                    mode=mode,
                    direction=direction,
                    line=line,
                    line_level=line_last,
                    distance_to_line_atr=dist_now_atr,
                    used_retest=False,
                    body_atr=body_atr,
                )
                logger.info(
                    "[gold-ai-trendline] detected mode=bounce side=support direction=bullish "
                    "level=%.2f dist_atr=%.2f body_atr=%.2f touches=%s fit=%.3f",
                    line_last,
                    dist_now_atr,
                    body_atr,
                    line.touch_count,
                    line.fit_error_atr,
                )
                return (
                    True,
                    (
                        f"trendline bounce long support line={line_last:.2f} "
                        f"touches={line.touch_count} slope_atr={line.slope_atr_per_bar:.3f} "
                        f"fit={line.fit_error_atr:.3f} dist={dist_now_atr:.2f}atr"
                    ),
                    meta,
                )
            return False, "trendline:bounce_long_not_confirmed", {}

        if direction == "bearish" and line.side == "resistance":
            upper_wick = h_last - max(o_last, c_last)
            touched = h_last >= (line_last - touch_tol)
            held = c_last <= (line_last - (bounce_reclaim_buf_atr * atr))
            approach = closes[prev_idx] <= (line_prev + touch_tol)
            if (
                touched
                and held
                and approach
                and c_last < o_last
                and body_atr >= bounce_min_body_atr
                and upper_wick >= (bounce_min_wick_atr * atr)
            ):
                meta = _trendline_meta(
                    mode=mode,
                    direction=direction,
                    line=line,
                    line_level=line_last,
                    distance_to_line_atr=dist_now_atr,
                    used_retest=False,
                    body_atr=body_atr,
                )
                logger.info(
                    "[gold-ai-trendline] detected mode=bounce side=resistance direction=bearish "
                    "level=%.2f dist_atr=%.2f body_atr=%.2f touches=%s fit=%.3f",
                    line_last,
                    dist_now_atr,
                    body_atr,
                    line.touch_count,
                    line.fit_error_atr,
                )
                return (
                    True,
                    (
                        f"trendline bounce short resistance line={line_last:.2f} "
                        f"touches={line.touch_count} slope_atr={line.slope_atr_per_bar:.3f} "
                        f"fit={line.fit_error_atr:.3f} dist={dist_now_atr:.2f}atr"
                    ),
                    meta,
                )
            return False, "trendline:bounce_short_not_confirmed", {}
        return False, "trendline:bounce_side_direction_mismatch", {}

    if mode == "break":
        if direction == "bullish" and line.side == "resistance":
            def _conviction_break(i: int) -> bool:
                line_i = _line_price(line.p1, line.slope, i)
                body_i = abs(closes[i] - opens[i]) / max(atr, 1e-9)
                return (
                    closes[i] > (line_i + (break_close_buf_atr * atr))
                    and closes[i] > opens[i]
                    and body_i >= break_min_body_atr
                )

            if not break_require_retest:
                if _conviction_break(last_idx):
                    meta = _trendline_meta(
                        mode=mode,
                        direction=direction,
                        line=line,
                        line_level=line_last,
                        distance_to_line_atr=dist_now_atr,
                        used_retest=False,
                        body_atr=body_atr,
                    )
                    logger.info(
                        "[gold-ai-trendline] detected mode=break side=resistance direction=bullish "
                        "level=%.2f dist_atr=%.2f body_atr=%.2f touches=%s fit=%.3f",
                        line_last,
                        dist_now_atr,
                        body_atr,
                        line.touch_count,
                        line.fit_error_atr,
                    )
                    return (
                        True,
                        (
                            f"trendline break long resistance line={line_last:.2f} "
                            f"touches={line.touch_count} slope_atr={line.slope_atr_per_bar:.3f} "
                            f"fit={line.fit_error_atr:.3f} dist={dist_now_atr:.2f}atr"
                        ),
                        meta,
                    )
                return False, "trendline:break_long_no_conviction_close", {}

            start = max(0, last_idx - break_retest_max_bars - 2)
            break_i = None
            for i in range(last_idx - 1, start - 1, -1):
                if _conviction_break(i):
                    break_i = i
                    break
            if break_i is None:
                return False, "trendline:break_long_wait_break", {}
            line_k = line_last
            retest_ok = lows[last_idx] <= (line_k + (break_retest_tol_atr * atr)) and c_last >= line_k
            if not retest_ok:
                return False, "trendline:break_long_wait_retest", {}
            meta = _trendline_meta(
                mode=mode,
                direction=direction,
                line=line,
                line_level=line_k,
                distance_to_line_atr=dist_now_atr,
                used_retest=True,
                body_atr=body_atr,
            )
            logger.info(
                "[gold-ai-trendline] detected mode=break side=resistance direction=bullish "
                "retest=yes level=%.2f dist_atr=%.2f body_atr=%.2f touches=%s fit=%.3f",
                line_k,
                dist_now_atr,
                body_atr,
                line.touch_count,
                line.fit_error_atr,
            )
            return (
                True,
                (
                    f"trendline break long retest resistance line={line_k:.2f} "
                    f"touches={line.touch_count} slope_atr={line.slope_atr_per_bar:.3f} "
                    f"fit={line.fit_error_atr:.3f} dist={dist_now_atr:.2f}atr"
                ),
                meta,
            )

        if direction == "bearish" and line.side == "support":
            def _conviction_break(i: int) -> bool:
                line_i = _line_price(line.p1, line.slope, i)
                body_i = abs(closes[i] - opens[i]) / max(atr, 1e-9)
                return (
                    closes[i] < (line_i - (break_close_buf_atr * atr))
                    and closes[i] < opens[i]
                    and body_i >= break_min_body_atr
                )

            if not break_require_retest:
                if _conviction_break(last_idx):
                    meta = _trendline_meta(
                        mode=mode,
                        direction=direction,
                        line=line,
                        line_level=line_last,
                        distance_to_line_atr=dist_now_atr,
                        used_retest=False,
                        body_atr=body_atr,
                    )
                    logger.info(
                        "[gold-ai-trendline] detected mode=break side=support direction=bearish "
                        "level=%.2f dist_atr=%.2f body_atr=%.2f touches=%s fit=%.3f",
                        line_last,
                        dist_now_atr,
                        body_atr,
                        line.touch_count,
                        line.fit_error_atr,
                    )
                    return (
                        True,
                        (
                            f"trendline break short support line={line_last:.2f} "
                            f"touches={line.touch_count} slope_atr={line.slope_atr_per_bar:.3f} "
                            f"fit={line.fit_error_atr:.3f} dist={dist_now_atr:.2f}atr"
                        ),
                        meta,
                    )
                return False, "trendline:break_short_no_conviction_close", {}

            start = max(0, last_idx - break_retest_max_bars - 2)
            break_i = None
            for i in range(last_idx - 1, start - 1, -1):
                if _conviction_break(i):
                    break_i = i
                    break
            if break_i is None:
                return False, "trendline:break_short_wait_break", {}
            line_k = line_last
            retest_ok = highs[last_idx] >= (line_k - (break_retest_tol_atr * atr)) and c_last <= line_k
            if not retest_ok:
                return False, "trendline:break_short_wait_retest", {}
            meta = _trendline_meta(
                mode=mode,
                direction=direction,
                line=line,
                line_level=line_k,
                distance_to_line_atr=dist_now_atr,
                used_retest=True,
                body_atr=body_atr,
            )
            logger.info(
                "[gold-ai-trendline] detected mode=break side=support direction=bearish "
                "retest=yes level=%.2f dist_atr=%.2f body_atr=%.2f touches=%s fit=%.3f",
                line_k,
                dist_now_atr,
                body_atr,
                line.touch_count,
                line.fit_error_atr,
            )
            return (
                True,
                (
                    f"trendline break short retest support line={line_k:.2f} "
                    f"touches={line.touch_count} slope_atr={line.slope_atr_per_bar:.3f} "
                    f"fit={line.fit_error_atr:.3f} dist={dist_now_atr:.2f}atr"
                ),
                meta,
            )
            # pragma: no cover
        return False, "trendline:break_side_direction_mismatch", {}

    return False, f"trendline:unknown_mode({mode})", {}
