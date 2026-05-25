"""
Forex strategy engine — sessions, pip math, opening ranges.

This module is the foundation for the dedicated Forex Strategy Builder. It
intentionally does NOT reuse crypto helpers because forex traders think in
*sessions, pips, and macroeconomic context*, not in 24/7 % moves.

P1 scope (this file): pure UTC session clock, pip-size table, session H/L
tracker, opening-range breakout (ORB) calculator.

P2 (later): economic-calendar integration (FMP free tier).
P3 (later): 28-pair currency-strength matrix.

All times are UTC. Session windows below match the de-facto industry
definitions used by Forex Factory, Babypips, and OANDA's session clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Dict, List, Optional, Tuple


# ─── Session windows (UTC, 24h) ──────────────────────────────────────────────
# (open_hour, open_minute, close_hour, close_minute) — close times use the
# "next session opens" convention. London/NY overlap is the deepest liquidity
# window of the day.

@dataclass(frozen=True)
class SessionWindow:
    id: str
    label: str
    open_h: int     # UTC hour
    open_m: int
    close_h: int    # UTC hour (may be < open if wraps midnight UTC)
    close_m: int


SESSIONS: Dict[str, SessionWindow] = {
    # Sydney 22:00–07:00 UTC (wraps midnight)
    "sydney":  SessionWindow("sydney",  "Sydney",   22, 0,  7, 0),
    # Asian (Tokyo) 00:00–09:00 UTC
    "asian":   SessionWindow("asian",   "Asian",     0, 0,  9, 0),
    # London 07:00–16:00 UTC
    "london":  SessionWindow("london",  "London",    7, 0, 16, 0),
    # New York 12:00–21:00 UTC
    "ny":      SessionWindow("ny",      "New York", 12, 0, 21, 0),
    # London/NY Overlap 12:00–16:00 UTC — deepest liquidity
    "overlap": SessionWindow("overlap", "London/NY Overlap", 12, 0, 16, 0),
}

# Session IDs exposed to the wizard. "overlap" is derived, not picked.
PICKABLE_SESSIONS = ("london", "ny", "asian", "sydney", "overlap")


def _in_window(now_utc: datetime, w: SessionWindow) -> bool:
    """Is `now_utc` inside session window `w` (handles midnight-wrap)?"""
    cur_minutes = now_utc.hour * 60 + now_utc.minute
    open_minutes = w.open_h * 60 + w.open_m
    close_minutes = w.close_h * 60 + w.close_m
    if open_minutes <= close_minutes:
        return open_minutes <= cur_minutes < close_minutes
    # Wraps midnight (e.g. Sydney 22:00 → 07:00 next day)
    return cur_minutes >= open_minutes or cur_minutes < close_minutes


def in_session(session_id: str, now_utc: Optional[datetime] = None) -> bool:
    """True if `now_utc` is inside the given forex session window."""
    w = SESSIONS.get((session_id or "").lower())
    if not w:
        return False
    now_utc = now_utc or datetime.utcnow()
    # Forex is closed Sat all day, Sun before 22:00 UTC, Fri after 22:00 UTC.
    wd = now_utc.weekday()  # Mon=0
    hr = now_utc.hour
    if wd == 5: return False
    if wd == 4 and hr >= 22: return False
    if wd == 6 and hr < 22:  return False
    return _in_window(now_utc, w)


def session_just_opened(session_id: str, within_minutes: int = 30,
                        now_utc: Optional[datetime] = None) -> bool:
    """True if the named session opened within the last `within_minutes`.

    Used by the "London Open" / "NY Open" condition types — the strategy fires
    only in the high-liquidity first 30 minutes of a session, not all day.
    """
    w = SESSIONS.get((session_id or "").lower())
    if not w:
        return False
    now_utc = now_utc or datetime.utcnow()
    if not in_session(session_id, now_utc):
        return False
    # Minutes since the session's open today.
    open_today = now_utc.replace(hour=w.open_h, minute=w.open_m,
                                  second=0, microsecond=0)
    if open_today > now_utc:
        # Open hasn't happened yet today (e.g. checking Sydney at 02:00 — open
        # was at 22:00 *yesterday*, so we're past it by ~4h).
        open_today -= timedelta(days=1)
    minutes_since_open = (now_utc - open_today).total_seconds() / 60
    return 0 <= minutes_since_open <= within_minutes


def session_about_to_close(session_id: str, within_minutes: int = 30,
                           now_utc: Optional[datetime] = None) -> bool:
    """True if the named session closes within the next `within_minutes`."""
    w = SESSIONS.get((session_id or "").lower())
    if not w:
        return False
    now_utc = now_utc or datetime.utcnow()
    if not in_session(session_id, now_utc):
        return False
    close_today = now_utc.replace(hour=w.close_h, minute=w.close_m,
                                  second=0, microsecond=0)
    if close_today < now_utc:
        close_today += timedelta(days=1)
    minutes_to_close = (close_today - now_utc).total_seconds() / 60
    return 0 <= minutes_to_close <= within_minutes


def current_sessions(now_utc: Optional[datetime] = None) -> List[str]:
    """Which named sessions are open right now (excludes 'overlap')."""
    now_utc = now_utc or datetime.utcnow()
    return [sid for sid in ("sydney", "asian", "london", "ny")
            if in_session(sid, now_utc)]


# ─── Pip math ────────────────────────────────────────────────────────────────
# JPY pairs:  0.01  (e.g. USDJPY moves in 0.01 increments)
# Metals:     own table — gold $1/pip, silver $0.01/pip
#             At ~$3 300 gold: 1 pip = $1 → 20 pips ≈ 0.60%  (matches
#             cTrader / MT4 "standard lot" pip convention for XAUUSD).
#             At ~$33 silver: 1 pip = $0.01 → 20 pips ≈ 0.60%
# Everything else: 0.0001  (standard 4-decimal forex)

_JPY_PAIRS = ("USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY")

# Metals quoted vs USD — pip size in USD per pip.
_METAL_PIP_SIZES: dict = {
    "XAUUSD": 1.0,    # Gold:   1 pip = $1.00
    "XAGUSD": 0.01,   # Silver: 1 pip = $0.01
}


def pip_size(pair: str) -> float:
    """Return the pip size for a pair.

    • Metals  (XAU/XAG): see _METAL_PIP_SIZES — $1 per pip for gold, $0.01 for silver
    • JPY pairs         : 0.01
    • All others        : 0.0001 (standard 4-decimal forex)
    """
    p = (pair or "").upper().replace("/", "").replace("=X", "")
    if p in _METAL_PIP_SIZES:
        return _METAL_PIP_SIZES[p]
    return 0.01 if p in _JPY_PAIRS or p.endswith("JPY") else 0.0001


def pips_to_pct(pair: str, price: float, pips: float) -> float:
    """Convert `pips` to a % move at `price` for `pair`.

    Used by the executor to translate pip-based TP/SL from the forex wizard
    into the % move the existing exit engine already understands. So a 20-pip
    SL on EURUSD at 1.0850 becomes (20 * 0.0001 / 1.0850) * 100 ≈ 0.184%.
    """
    if not price or price <= 0:
        return 0.0
    return (pips * pip_size(pair)) / price * 100.0


def pct_to_pips(pair: str, price: float, pct: float) -> float:
    """Inverse of pips_to_pct — used by the UI to display equivalent pips."""
    if not price or price <= 0:
        return 0.0
    return (pct / 100.0 * price) / pip_size(pair)


# ─── Session high / low tracker ──────────────────────────────────────────────

@dataclass
class SessionRange:
    session: str
    high: float
    low: float
    opened_at: datetime  # session-open in UTC for the candle window we used
    closed_at: Optional[datetime]  # None while session is still ongoing


def compute_session_range(klines: List[List], session_id: str,
                          now_utc: Optional[datetime] = None,
                          first_n_minutes: Optional[int] = None) -> Optional[SessionRange]:
    """Compute the high/low of the candles that fell inside `session_id` today.

    klines: MEXC-shaped list of [ts_ms, o, h, l, c, v, ...] in chronological order.
    session_id: 'london' | 'ny' | 'asian' | 'sydney'
    first_n_minutes: if set, only consider the first N minutes of the session
        (used by Opening Range Breakout — e.g. ORB30 = first 30 minutes).

    Returns None if no candles fall inside the session window today, OR if the
    session hasn't opened yet today.
    """
    w = SESSIONS.get((session_id or "").lower())
    if not w or not klines:
        return None
    now_utc = now_utc or datetime.utcnow()

    # Find today's session open instant in UTC. If the open hasn't happened
    # yet today, fall back to yesterday's session (relevant for Sydney/Asian
    # which open before midnight).
    open_today = now_utc.replace(hour=w.open_h, minute=w.open_m,
                                  second=0, microsecond=0)
    if open_today > now_utc:
        open_today -= timedelta(days=1)

    # Session close instant — handle midnight wrap.
    close_minutes_from_open = ((w.close_h * 60 + w.close_m) -
                               (w.open_h * 60 + w.open_m)) % (24 * 60)
    if close_minutes_from_open == 0:
        close_minutes_from_open = 24 * 60
    close_today = open_today + timedelta(minutes=close_minutes_from_open)

    # Optional opening-range cap (e.g. first 30 min only)
    if first_n_minutes is not None:
        close_today = min(close_today, open_today + timedelta(minutes=first_n_minutes))

    open_ms = int(open_today.replace(tzinfo=timezone.utc).timestamp() * 1000)
    close_ms = int(close_today.replace(tzinfo=timezone.utc).timestamp() * 1000)

    highs: List[float] = []
    lows: List[float] = []
    for k in klines:
        try:
            ts = int(k[0])
            if open_ms <= ts < close_ms:
                highs.append(float(k[2]))
                lows.append(float(k[3]))
        except (IndexError, TypeError, ValueError):
            continue

    if not highs:
        return None

    # closed_at is None if we're still inside the window.
    closed_at = close_today if now_utc >= close_today else None
    return SessionRange(
        session=session_id, high=max(highs), low=min(lows),
        opened_at=open_today, closed_at=closed_at,
    )


# ─── Previous-day / weekly key levels ────────────────────────────────────────

def previous_day_high_low(klines: List[List],
                          now_utc: Optional[datetime] = None) -> Optional[Tuple[float, float]]:
    """High/low of the previous UTC calendar day. Klines must span ≥2 days."""
    if not klines:
        return None
    now_utc = now_utc or datetime.utcnow()
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    y_ms = int(yesterday_start.replace(tzinfo=timezone.utc).timestamp() * 1000)
    t_ms = int(today_start.replace(tzinfo=timezone.utc).timestamp() * 1000)
    highs: List[float] = []
    lows: List[float] = []
    for k in klines:
        try:
            ts = int(k[0])
            if y_ms <= ts < t_ms:
                highs.append(float(k[2]))
                lows.append(float(k[3]))
        except (IndexError, TypeError, ValueError):
            continue
    if not highs:
        return None
    return (max(highs), min(lows))


def previous_week_high_low(klines: List[List],
                           now_utc: Optional[datetime] = None) -> Optional[Tuple[float, float]]:
    """High/low of the previous ISO calendar week (Mon 00:00 → Sun 23:59 UTC)."""
    if not klines:
        return None
    now_utc = now_utc or datetime.utcnow()
    # Start of THIS week (Monday 00:00 UTC)
    this_week_start = now_utc - timedelta(days=now_utc.weekday())
    this_week_start = this_week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    last_week_start = this_week_start - timedelta(days=7)
    s_ms = int(last_week_start.replace(tzinfo=timezone.utc).timestamp() * 1000)
    e_ms = int(this_week_start.replace(tzinfo=timezone.utc).timestamp() * 1000)
    highs: List[float] = []
    lows: List[float] = []
    for k in klines:
        try:
            ts = int(k[0])
            if s_ms <= ts < e_ms:
                highs.append(float(k[2]))
                lows.append(float(k[3]))
        except (IndexError, TypeError, ValueError):
            continue
    if not highs:
        return None
    return (max(highs), min(lows))


# ─── Templates ───────────────────────────────────────────────────────────────
# Pre-baked forex strategies the wizard offers as one-click starting points.
# Each template is a partial wizard state — the wizard merges these on top of
# defaults and lets the user tweak before save.

FOREX_TEMPLATES: Dict[str, Dict] = {
    "london_breakout": {
        "label": "London Breakout",
        "tagline": "Buy/sell when London open breaks the Asian session range",
        "direction": "BOTH",
        "tp_pips": 30, "sl_pips": 15,
        "session": "london", "trigger_session": "asian",
        "trigger_window_min": 30,
        "timeframe": "15m",
    },
    "asian_range_breakout": {
        "label": "Asian Range Breakout",
        "tagline": "Trade the breakout of the overnight Sydney/Asian range",
        "direction": "BOTH",
        "tp_pips": 25, "sl_pips": 12,
        "session": "asian", "trigger_session": "sydney",
        "trigger_window_min": 60,
        "timeframe": "15m",
    },
    "ny_reversal": {
        "label": "New York Reversal",
        "tagline": "Fade overextended moves at NY open during overlap",
        "direction": "BOTH",
        "tp_pips": 20, "sl_pips": 12,
        "session": "ny", "trigger_session": "london",
        "trigger_window_min": 30,
        "timeframe": "15m",
    },
    "trend_continuation": {
        "label": "Trend Continuation",
        "tagline": "Follow the prevailing trend during London/NY overlap",
        "direction": "BOTH",
        "tp_pips": 40, "sl_pips": 20,
        "session": "overlap", "timeframe": "1h",
    },
    "news_avoidance": {
        "label": "News Avoidance",
        "tagline": "Skip trading 30 min before/after high-impact news",
        "direction": "BOTH",
        "tp_pips": 25, "sl_pips": 15,
        "session": "london", "timeframe": "15m",
        "news_blackout_min": 30,
    },
    "liquidity_grab": {
        "label": "Liquidity Grab",
        "tagline": "Reverse after price sweeps the previous day high/low",
        "direction": "BOTH",
        "tp_pips": 30, "sl_pips": 15,
        "session": "london", "timeframe": "15m",
        "use_pdh_pdl": True,
    },
    "session_scalping": {
        "label": "Session Scalping",
        "tagline": "Quick scalps during the deepest-liquidity overlap window",
        "direction": "BOTH",
        "tp_pips": 10, "sl_pips": 6,
        "session": "overlap", "timeframe": "5m",
    },
    "currency_strength": {
        "label": "Currency Strength",
        "tagline": "Buy the strong vs the weak — 28-pair strength matrix",
        "direction": "BOTH",
        "tp_pips": 35, "sl_pips": 18,
        "session": "overlap", "timeframe": "1h",
        "strength_window": "4h", "strength_min_diff": 0.6,
    },
    "liquidity_pa": {
        "label": "Liquidity & Price Action",
        "tagline": "Sweep equal highs/lows + pin bars / engulfings on liquid sessions",
        "direction": "BOTH",
        "tp_pips": 28, "sl_pips": 14,
        "session": "london", "timeframe": "15m",
        "pa_pattern": "sweep_eqh", "pa_lookback": 20, "pa_tolerance_pips": 3,
    },
    "cot_sentiment": {
        "label": "COT Sentiment Swing",
        "tagline": "Fade extreme speculator positioning — weekly CFTC contrarian",
        "direction": "BOTH",
        "tp_pips": 80, "sl_pips": 40,
        "session": "any", "timeframe": "4h",
        "cot_condition": "specs_extreme_long", "cot_extreme_pct": 75, "cot_lookback_weeks": 52,
    },
}


def template_ids() -> List[str]:
    return list(FOREX_TEMPLATES.keys())
