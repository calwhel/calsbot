"""Opening Range Breakout (ORB) detector/state + Claude context."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from app.gold_ai_trader.call_gates import atr_from_klines
from app.gemini_gold_trader.klines import get_chart_klines
from app.gemini_gold_trader.models import GeminiGoldOrbState
from app.services.forex_sessions import LIVE_FOREX_SESSIONS

logger = logging.getLogger(__name__)


async def _get_orb_klines(tf: str, limit: int, user_id: Optional[int] = None) -> List[list]:
    bars, _ = await get_chart_klines(tf, limit, user_id=user_id)
    return bars or []


@dataclass
class OrbSignal:
    side: str  # long|short
    setup_type: str  # orb_long|orb_short
    break_level: float
    breakout_ts: datetime
    range_high: float
    range_low: float
    range_height: float
    atr: float
    bar_open: float
    bar_high: float
    bar_low: float
    bar_close: float
    body_atr: float
    breakout_dist: float
    used_retest: bool
    session_start_ts: datetime
    range_end_ts: datetime
    trade_window_end_ts: datetime


_TF_MINUTES: Dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "30m": 30,
    "1h": 60,
}


def _tf_minutes(tf: str) -> int:
    return int(_TF_MINUTES.get((tf or "5m").lower(), 5))


def _session_start(now: datetime, session: str) -> datetime:
    win = LIVE_FOREX_SESSIONS.get(session, LIVE_FOREX_SESSIONS["london"])
    return now.replace(hour=int(win[0]), minute=int(win[1]), second=0, microsecond=0)


def _bar_dt(bar: list) -> datetime:
    ts = int(bar[0])
    return datetime.utcfromtimestamp(ts / 1000 if ts > 1e10 else ts)


def _closed_bars(rows: List[list], now: datetime, tf_min: int) -> List[list]:
    out: List[list] = []
    cutoff = now - timedelta(minutes=max(1, tf_min))
    for r in rows:
        try:
            if _bar_dt(r) <= cutoff:
                out.append(r)
        except Exception:
            continue
    return out


def _bars_between(rows: List[list], start: datetime, end: datetime) -> List[list]:
    return [r for r in rows if start <= _bar_dt(r) < end]


def _range_from_rows(rows: List[list]) -> Optional[Tuple[float, float, float]]:
    if not rows:
        return None
    highs: List[float] = []
    lows: List[float] = []
    for r in rows:
        try:
            highs.append(float(r[2]))
            lows.append(float(r[3]))
        except Exception:
            continue
    if not highs or not lows:
        return None
    hi = max(highs)
    lo = min(lows)
    return hi, lo, (hi - lo)


def _ensure_state(db, *, now: datetime, session: str, cfg) -> GeminiGoldOrbState:
    trade_day = now.date()
    row = (
        db.query(GeminiGoldOrbState)
        .filter(
            GeminiGoldOrbState.trade_day_utc == trade_day,
            GeminiGoldOrbState.session == session,
        )
        .first()
    )
    start_ts = _session_start(now, session)
    range_end_ts = start_ts + timedelta(minutes=max(5, int(cfg.orb_range_minutes)))
    trade_end_ts = range_end_ts + timedelta(minutes=max(10, int(cfg.orb_trade_window_minutes)))
    if row is None:
        row = GeminiGoldOrbState(
            trade_day_utc=trade_day,
            session=session,
            status="forming",
            range_start_ts=start_ts,
            range_end_ts=range_end_ts,
            trade_window_end_ts=trade_end_ts,
            trades_taken=0,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row
    row.range_start_ts = start_ts
    row.range_end_ts = range_end_ts
    row.trade_window_end_ts = trade_end_ts
    return row


def _persist_state(db, row: GeminiGoldOrbState) -> None:
    db.add(row)
    db.commit()
    db.refresh(row)


def _breakout_from_bar(
    *,
    row: list,
    confirmation: str,
    range_high: float,
    range_low: float,
    buffer: float,
) -> Tuple[Optional[str], Optional[float], float]:
    o = float(row[1])
    h = float(row[2])
    l = float(row[3])
    c = float(row[4])
    side: Optional[str] = None
    level: Optional[float] = None
    dist = 0.0
    if confirmation == "wick":
        if h > (range_high + buffer):
            side = "long"
            level = range_high
            dist = h - range_high
        if l < (range_low - buffer):
            short_dist = range_low - l
            if side is None or short_dist > dist:
                side = "short"
                level = range_low
                dist = short_dist
        return side, level, dist

    if c > (range_high + buffer):
        side = "long"
        level = range_high
        dist = c - range_high
    if c < (range_low - buffer):
        short_dist = range_low - c
        if side is None or short_dist > dist:
            side = "short"
            level = range_low
            dist = short_dist
    return side, level, dist


async def detect_orb_signal(
    *,
    cfg,
    session: str,
    now: datetime,
    user_id: Optional[int],
) -> Tuple[Optional[OrbSignal], Optional[GeminiGoldOrbState], str]:
    """
    Returns (signal, state_row, reason).
    """
    from app.gemini_gold_trader.db_thread import run_with_db

    if not bool(getattr(cfg, "orb_enabled", False)):
        return None, None, "orb_disabled"

    tf = (getattr(cfg, "orb_timeframe", "5m") or "5m").lower().strip()
    tf_min = _tf_minutes(tf)
    row = await run_with_db(_ensure_state, now=now, session=session, cfg=cfg)
    start_ts = row.range_start_ts or _session_start(now, session)
    range_end_ts = row.range_end_ts or (start_ts + timedelta(minutes=max(5, int(cfg.orb_range_minutes))))
    trade_end_ts = row.trade_window_end_ts or (
        range_end_ts + timedelta(minutes=max(10, int(cfg.orb_trade_window_minutes)))
    )

    k = await _get_orb_klines(tf, 160, user_id=user_id) or []
    closed = _closed_bars(k, now, tf_min)
    range_rows = _bars_between(closed, start_ts, range_end_ts)
    rng = _range_from_rows(range_rows)
    if rng:
        row.range_high, row.range_low, row.range_height = rng

    k5 = await _get_orb_klines("5m", 80, user_id=user_id) or []
    atr = atr_from_klines(k5)
    if now < range_end_ts:
        row.status = "forming"
        await run_with_db(_persist_state, row)
        return None, row, "forming_range"

    if row.trades_taken >= max(1, int(getattr(cfg, "orb_max_trades_per_session", 1))):
        row.status = "traded"
        await run_with_db(_persist_state, row)
        return None, row, "max_trades_session"

    if now > trade_end_ts:
        row.status = "expired"
        await run_with_db(_persist_state, row)
        return None, row, "trade_window_expired"

    if not row.range_high or not row.range_low or not row.range_height or row.range_height <= 0:
        row.status = "forming"
        await run_with_db(_persist_state, row)
        return None, row, "range_unavailable"

    row.status = "armed"
    await run_with_db(_persist_state, row)

    if atr <= 0:
        return None, row, "atr_unavailable"
    range_atr = row.range_height / atr if atr > 0 else 0.0
    if range_atr < float(getattr(cfg, "orb_min_range_atr", 0.0)):
        return None, row, f"range_too_small({range_atr:.2f}atr)"
    if range_atr > float(getattr(cfg, "orb_max_range_atr", 999.0)):
        return None, row, f"range_too_wide({range_atr:.2f}atr)"

    post_rows = [r for r in closed if _bar_dt(r) >= range_end_ts and _bar_dt(r) <= trade_end_ts]
    if not post_rows:
        return None, row, "no_post_range_bars"
    last = post_rows[-1]
    last_ts = _bar_dt(last)

    use_fakeout = bool(getattr(cfg, "orb_fakeout_filter", True))
    buffer = 0.0
    if use_fakeout:
        buffer = max(
            float(getattr(cfg, "orb_break_buffer_atr", 0.10)) * atr,
            float(getattr(cfg, "orb_break_buffer_range_pct", 0.05)) * row.range_height,
        )
    side, break_level, breakout_dist = _breakout_from_bar(
        row=last,
        confirmation=(getattr(cfg, "orb_confirmation", "close") or "close").lower(),
        range_high=row.range_high,
        range_low=row.range_low,
        buffer=buffer,
    )
    if not side or break_level is None:
        return None, row, "no_breakout"

    o = float(last[1])
    h = float(last[2])
    l = float(last[3])
    c = float(last[4])
    body_atr = abs(c - o) / atr if atr > 0 else 0.0
    if use_fakeout and body_atr < float(getattr(cfg, "orb_min_break_body_atr", 0.30)):
        return None, row, f"break_body_too_small({body_atr:.2f}atr)"

    require_retest = bool(getattr(cfg, "orb_require_retest", False))
    if require_retest:
        if not row.breakout_side:
            row.breakout_side = side
            row.breakout_level = float(break_level)
            row.breakout_ts = last_ts
            await run_with_db(_persist_state, row)
            return None, row, "breakout_seen_wait_retest"
        if row.breakout_ts and last_ts <= row.breakout_ts:
            return None, row, "waiting_new_bar_for_retest"
        # Expire staged breakout after N bars.
        max_bars = max(1, int(getattr(cfg, "orb_retest_max_bars", 3)))
        since = [r for r in post_rows if row.breakout_ts and _bar_dt(r) > row.breakout_ts]
        if len(since) > max_bars:
            row.breakout_side = None
            row.breakout_level = None
            row.breakout_ts = None
            await run_with_db(_persist_state, row)
            return None, row, "retest_window_expired"
        staged_side = (row.breakout_side or "").lower()
        level = float(row.breakout_level or break_level)
        tol = max(
            0.0,
            float(getattr(cfg, "orb_retest_tol_atr", 0.15)) * atr,
        )
        if staged_side == "long":
            retest_ok = l <= (level + tol) and c >= level
            if not retest_ok:
                return None, row, "retest_not_confirmed"
            side = "long"
            break_level = level
        elif staged_side == "short":
            retest_ok = h >= (level - tol) and c <= level
            if not retest_ok:
                return None, row, "retest_not_confirmed"
            side = "short"
            break_level = level
        else:
            return None, row, "invalid_staged_breakout"
        row.breakout_side = side
        row.breakout_level = float(break_level)
        row.breakout_ts = last_ts
        await run_with_db(_persist_state, row)
    else:
        if row.breakout_ts and last_ts <= row.breakout_ts:
            return None, row, "breakout_already_processed"
        row.breakout_side = side
        row.breakout_level = float(break_level)
        row.breakout_ts = last_ts
        await run_with_db(_persist_state, row)

    setup_type = "orb_long" if side == "long" else "orb_short"
    signal = OrbSignal(
        side=side,
        setup_type=setup_type,
        break_level=float(break_level),
        breakout_ts=last_ts,
        range_high=float(row.range_high),
        range_low=float(row.range_low),
        range_height=float(row.range_height),
        atr=float(atr),
        bar_open=o,
        bar_high=h,
        bar_low=l,
        bar_close=c,
        body_atr=float(body_atr),
        breakout_dist=float(breakout_dist),
        used_retest=require_retest,
        session_start_ts=start_ts,
        range_end_ts=range_end_ts,
        trade_window_end_ts=trade_end_ts,
    )
    return signal, row, "signal_ready"


def suggested_orb_levels(signal: OrbSignal, cfg) -> Tuple[float, float, float]:
    direction = signal.side
    entry = signal.bar_close
    sl_mode = (getattr(cfg, "orb_sl_mode", "range_opposite") or "range_opposite").lower()
    tp_mode = (getattr(cfg, "orb_tp_mode", "range_multiple") or "range_multiple").lower()
    if sl_mode == "atr":
        sl_dist = max(0.01, float(getattr(cfg, "orb_sl_atr_mult", 1.2)) * signal.atr)
        stop = entry - sl_dist if direction == "long" else entry + sl_dist
    else:
        pad = float(getattr(cfg, "orb_sl_range_buffer_atr", 0.05)) * signal.atr
        stop = (signal.range_low - pad) if direction == "long" else (signal.range_high + pad)
    risk = abs(entry - stop)
    if tp_mode == "fixed_rr":
        rr = max(0.1, float(getattr(cfg, "orb_tp_rr", 1.5)))
        tp_dist = risk * rr
    else:
        tp_dist = max(0.01, float(getattr(cfg, "orb_tp_range_mult", 1.5)) * signal.range_height)
    tp = entry + tp_dist if direction == "long" else entry - tp_dist
    return float(entry), float(stop), float(tp)


def build_orb_context(signal: OrbSignal, *, session: str, cfg, now: datetime, recent_bars: List[list]) -> str:
    entry, stop, tp = suggested_orb_levels(signal, cfg)
    bars_txt: List[str] = []
    for r in recent_bars[-8:]:
        try:
            dt = _bar_dt(r).strftime("%H:%M")
            bars_txt.append(
                f"{dt} O{float(r[1]):.2f} H{float(r[2]):.2f} "
                f"L{float(r[3]):.2f} C{float(r[4]):.2f}"
            )
        except Exception:
            continue
    return "\n".join(
        [
            "=== ORB CONTEXT (XAUUSD) ===",
            f"Timestamp UTC: {now.isoformat()}Z",
            f"Session: {session.upper()}",
            f"Range window: {signal.session_start_ts.isoformat()}Z -> {signal.range_end_ts.isoformat()}Z",
            f"Trade window ends: {signal.trade_window_end_ts.isoformat()}Z",
            f"Range high/low/height: {signal.range_high:.2f} / {signal.range_low:.2f} / {signal.range_height:.2f}",
            f"Break side: {signal.side.upper()} | Break level: {signal.break_level:.2f}",
            f"Breakout bar: O{signal.bar_open:.2f} H{signal.bar_high:.2f} L{signal.bar_low:.2f} C{signal.bar_close:.2f}",
            f"Breakout body quality: {signal.body_atr:.2f}x ATR | Break distance: {signal.breakout_dist:.2f}",
            f"Retest mode used: {'yes' if signal.used_retest else 'no'}",
            f"Suggested entry/SL/TP: {entry:.2f} / {stop:.2f} / {tp:.2f}",
            f"ORB confidence threshold: {int(getattr(cfg, 'orb_confidence_threshold', 55))}%",
            "Recent bars (oldest->newest):",
            " | ".join(bars_txt) if bars_txt else "n/a",
        ]
    )
