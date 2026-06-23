"""Key level computation for Gold AI Trader context snapshots."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional, Tuple


def _candle_dt(row) -> Optional[datetime]:
    if not row or len(row) < 5:
        return None
    try:
        ts = int(float(row[0]))
        if ts > 1_000_000_000_000:
            ts //= 1000
        return datetime.utcfromtimestamp(ts)
    except (TypeError, ValueError, OSError):
        return None


def _ohlc(row) -> Optional[Tuple[float, float, float, float]]:
    if not row or len(row) < 5:
        return None
    try:
        return float(row[1]), float(row[2]), float(row[3]), float(row[4])
    except (TypeError, ValueError):
        return None


def _aggregate_hl(rows: List[list], start: datetime, end: datetime) -> Optional[Tuple[float, float]]:
    hi = lo = None
    for row in rows or []:
        dt = _candle_dt(row)
        if dt is None or dt < start or dt >= end:
            continue
        ohlc = _ohlc(row)
        if not ohlc:
            continue
        _, h, l, _ = ohlc
        hi = h if hi is None else max(hi, h)
        lo = l if lo is None else min(lo, l)
    if hi is None or lo is None:
        return None
    return hi, lo


def _day_bounds(day: datetime) -> Tuple[datetime, datetime]:
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def compute_pdh_pdl(
    *,
    now: datetime,
    k_daily: List[list],
    k_1h: List[list],
    k_5m: List[list],
) -> Tuple[Optional[float], Optional[float]]:
    yesterday = (now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1))
    y_start, y_end = _day_bounds(yesterday)

    if k_daily:
        for row in reversed(k_daily):
            dt = _candle_dt(row)
            if dt is None:
                continue
            d0 = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            if d0 == y_start:
                ohlc = _ohlc(row)
                if ohlc:
                    return ohlc[1], ohlc[2]

    agg = _aggregate_hl(k_1h, y_start, y_end)
    if agg:
        return agg
    agg = _aggregate_hl(k_5m, y_start, y_end)
    return agg if agg else (None, None)


def compute_daily_open(now: datetime, k_daily: List[list], k_1h: List[list]) -> Optional[float]:
    today_start, today_end = _day_bounds(now)
    if k_daily:
        for row in reversed(k_daily):
            dt = _candle_dt(row)
            if dt and today_start <= dt < today_end:
                ohlc = _ohlc(row)
                if ohlc:
                    return ohlc[0]
    for row in k_1h or []:
        dt = _candle_dt(row)
        if dt and today_start <= dt < today_end:
            ohlc = _ohlc(row)
            if ohlc:
                return ohlc[0]
    return None


def compute_asian_range(now: datetime, k_5m: List[list], k_1h: List[list]) -> Tuple[Optional[float], Optional[float]]:
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    asian_end = today_start.replace(hour=7)
    agg = _aggregate_hl(k_5m, today_start, asian_end)
    if agg:
        return agg
    return _aggregate_hl(k_1h, today_start, asian_end) or (None, None)


def compute_session_range(
    now: datetime,
    session: str,
    cfg,
    k_5m: List[list],
    k_1h: List[list],
) -> Tuple[Optional[float], Optional[float]]:
    if session == "london":
        start_h, end_h = cfg.london_start_hour, cfg.london_end_hour
    elif session == "new_york":
        start_h, end_h = cfg.ny_start_hour, cfg.ny_end_hour
    else:
        return None, None
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = day.replace(hour=start_h, minute=0, second=0, microsecond=0)
    end = min(now, day.replace(hour=end_h, minute=0, second=0, microsecond=0))
    if end <= start:
        return None, None
    agg = _aggregate_hl(k_5m, start, end + timedelta(minutes=5))
    if agg:
        return agg
    return _aggregate_hl(k_1h, start, end + timedelta(hours=1)) or (None, None)


def format_level_line(
    label: str,
    level: Optional[float],
    spot: float,
    atr: float,
    *,
    suffix: str = "",
) -> str:
    if level is None:
        return f"{label}: unavailable"
    dist = level - spot
    sign = "+" if dist >= 0 else ""
    line = f"{label}: {level:.2f} (spot {sign}{dist:.2f}"
    if atr and atr > 0:
        mult = abs(dist) / atr
        line += f" | {mult:.1f}× ATR away"
        if mult <= 0.5:
            line += ") ← near"
        else:
            line += ")"
    else:
        line += ")"
    if suffix:
        line += f" {suffix}"
    return line


def build_key_levels_block(
    *,
    spot: float,
    atr: float,
    session: str,
    cfg,
    now: datetime,
    k_daily: List[list],
    k_1h: List[list],
    k_5m: List[list],
    setup_zone: Optional[Tuple[float, float]] = None,
) -> List[str]:
    pdh, pdl = compute_pdh_pdl(now=now, k_daily=k_daily, k_1h=k_1h, k_5m=k_5m)
    daily_open = compute_daily_open(now, k_daily, k_1h)
    asian_hi, asian_lo = compute_asian_range(now, k_5m, k_1h)
    sess_hi, sess_lo = compute_session_range(now, session, cfg, k_5m, k_1h)
    sess_label = session.upper() if session else "SESSION"

    # Liquidity pools: EQH/EQL proxies from recent swing clusters
    liquidity_pools: List[str] = []
    if k_5m and len(k_5m) >= 20:
        try:
            from app.services.forex_engine import pip_size

            tol = 3.0 * pip_size("XAUUSD")
            highs = [float(r[2]) for r in k_5m[-30:-1] if r and len(r) >= 3]
            lows = [float(r[3]) for r in k_5m[-30:-1] if r and len(r) >= 3]
            if highs:
                ref_h = max(highs)
                eqh = [h for h in highs if abs(h - ref_h) <= tol]
                if len(eqh) >= 2:
                    liquidity_pools.append(f"EQH pool ~{sum(eqh)/len(eqh):.2f} ({len(eqh)} touches)")
            if lows:
                ref_l = min(lows)
                eql = [l for l in lows if abs(l - ref_l) <= tol]
                if len(eql) >= 2:
                    liquidity_pools.append(f"EQL pool ~{sum(eql)/len(eql):.2f} ({len(eql)} touches)")
        except Exception:
            pass

    lines = [
        "=== KEY LEVELS (structured) ===",
        f"Spot: {spot:.2f} | ATR(14) 5m: {atr:.2f}",
        format_level_line("Daily open", daily_open, spot, atr),
        format_level_line("PDH", pdh, spot, atr),
        format_level_line("PDL", pdl, spot, atr),
        format_level_line("Asian high", asian_hi, spot, atr),
        format_level_line("Asian low", asian_lo, spot, atr),
    ]
    if sess_hi is not None and sess_lo is not None:
        lines.append(f"Session high ({sess_label}): {sess_hi:.2f}")
        lines.append(f"Session low ({sess_label}): {sess_lo:.2f}")
    else:
        lines.append(f"Session high/low ({sess_label}): unavailable")

    if setup_zone:
        z_bot, z_top = setup_zone
        lines.append(f"Setup zone: {z_bot:.2f} – {z_top:.2f}")
        if z_bot <= spot <= z_top:
            lines.append("Setup zone status: IN ZONE")
        else:
            dist = min(abs(spot - z_bot), abs(spot - z_top))
            mult = (dist / atr) if atr > 0 else 0.0
            lines.append(f"Setup zone status: outside ({mult:.2f}× ATR from nearest bound)")

    if liquidity_pools:
        lines.append("Nearest liquidity pools: " + " | ".join(liquidity_pools))
    else:
        lines.append("Nearest liquidity pools: none detected in 30×5m window")

    return lines


def compute_premium_discount(
    spot: float,
    range_lo: Optional[float],
    range_hi: Optional[float],
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """Return (label discount|premium|equilibrium, pct_from_mid, midpoint)."""
    if range_lo is None or range_hi is None or range_hi <= range_lo:
        return None, None, None
    mid = (range_lo + range_hi) / 2.0
    half = (range_hi - range_lo) / 2.0
    if half <= 0:
        return None, None, mid
    pct = (spot - mid) / half * 100.0  # -100 discount edge .. +100 premium edge
    if pct > 5:
        label = "premium"
    elif pct < -5:
        label = "discount"
    else:
        label = "equilibrium"
    return label, pct, mid


def build_premium_discount_block(
    *,
    spot: float,
    k5: List[list],
    k1h: List[list],
    now: datetime,
    session: str,
    cfg,
) -> List[str]:
    """Dealing range = Asian range, fallback session range."""
    lines = ["=== PREMIUM / DISCOUNT ==="]
    asian_hi, asian_lo = compute_asian_range(now, k5, k1h)
    label, pct, mid = compute_premium_discount(spot, asian_lo, asian_hi)
    src = "Asian range (00–07 UTC)"
    r_lo, r_hi = asian_lo, asian_hi
    if label is None:
        sess_hi, sess_lo = compute_session_range(now, session, cfg, k5, k1h)
        label, pct, mid = compute_premium_discount(spot, sess_lo, sess_hi)
        src = f"{session.upper()} session range"
        r_lo, r_hi = sess_lo, sess_hi
    if label is None or mid is None or pct is None or r_lo is None or r_hi is None:
        lines.append("Premium/discount: unavailable (no dealing range)")
        return lines
    lines.append(f"Dealing range ({src}): {r_lo:.2f} – {r_hi:.2f} | Mid: {mid:.2f}")
    lines.append(
        f"Spot in {label} ({pct:+.0f}% from mid) — "
        f"{'longs favored' if label == 'discount' else 'shorts favored' if label == 'premium' else 'mid-range caution'}"
    )
    return lines
