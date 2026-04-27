"""
Auto-trader engine
==================
Runs user-saved chart setups in the background and simulates paper trades
against MEXC candles. Two modes per strategy:

  * 'ai'    — every cadence_min minutes, re-runs the same chart context
              through generate_ai_trade_read() and opens a paper trade when
              the AI plan's ODDS clears the user's threshold and ORDER_TYPE
              is MARKET. STOP/TP1/TP2 come straight from the AI plan.

  * 'rules' — evaluates a small DSL compiled from the user's chart at save
              time (e.g. fast/slow EMA cross, RSI oversold, MACD zero-cross,
              SuperTrend flip, optional wall-confluence filter). Cross
              detection looks at the last two closed candles only.

Position management is a simple stop-or-TP simulation against the latest
candles' high/low. When a position closes (or opens) we DM the user via
the existing Telegram bot helper.

Hooked into alerts_engine's main loop so the executor advisory lock prevents
multi-worker double-fires. We do nothing if the alerts engine isn't running.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.services.alerts_engine import (
    _fetch_candles, _ema, _rsi, _macd_line, _supertrend_dir,
    _tg_send,
)

logger = logging.getLogger(__name__)


# ─── Local TA helpers (kept here so this file isn't tightly coupled to
# alerts_engine's exact public surface) ───────────────────────────────────────
def _sma(values: List[float], period: int) -> Optional[float]:
    if not values or len(values) < period or period <= 0:
        return None
    return sum(values[-period:]) / float(period)


def _atr(candles: List[dict], period: int = 14) -> Optional[float]:
    """Simple Wilder-style ATR using true ranges over the last `period` bars."""
    if not candles or len(candles) < period + 1:
        return None
    trs: List[float] = []
    for i in range(1, len(candles)):
        h = float(candles[i].get("high") or 0.0)
        l = float(candles[i].get("low") or 0.0)
        pc = float(candles[i - 1].get("close") or 0.0)
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / float(period)


def _macd(values: List[float], fast: int, slow: int, signal: int) -> Optional[Tuple[float, float]]:
    """Adapter to alerts_engine._macd_line. Signal arg accepted for API
    parity but unused — we only need the macd line for zero-cross detection."""
    m = _macd_line(values, fast, slow)
    if m is None:
        return None
    return (m, 0.0)  # (macd_line, signal_placeholder)


def _supertrend(candles: List[dict], period: int, mult: float) -> Optional[Tuple[float, int]]:
    """Adapter to alerts_engine._supertrend_dir. Returns (line_placeholder, direction)."""
    d = _supertrend_dir(candles, period, mult)
    if d is None:
        return None
    return (0.0, int(d))


# Minimum gap between full evaluation passes per strategy. AI mode is gated
# again by `cadence_min`; rule mode just uses this floor so we don't hammer
# the candle source.
_RULE_TICK_FLOOR_S = 25
# AI cost-protection floor: never call Claude more than once per N seconds per
# strategy, even if the user sets cadence_min=1. Default 90s = balance between
# trade-timing responsiveness and API spend (~$0.005-0.01 per call → ≤$10/day
# per strategy at the floor; user-set cadence_min wins if it's longer). Pair
# with the auto-pause-on-insufficient-credits logic below — that's the real
# safety net once credits run dry. Override via env AI_TICK_FLOOR_S to taste.
try:
    _AI_TICK_FLOOR_S = max(30, int(os.environ.get("AI_TICK_FLOOR_S", "90")))
except (TypeError, ValueError):
    _AI_TICK_FLOOR_S = 90

# Wall confluence — entry must be within this % of nearest matching wall
_WALL_PROX_PCT = 0.6

# Symbol-level funding/OI cache shared across all AI-mode strategies so a
# fleet of bots on the same symbol only hits the upstream once per minute.
_FUNDING_CACHE: Dict[str, Tuple[Dict[str, Any], float]] = {}
_FUNDING_TTL = 60.0

# OKX symbol mapping — OKX perpetuals use the `<COIN>-USDT-SWAP` convention
# and the OI history endpoint takes the bare coin code.
_OKX_INST = {"BTC": "BTC-USDT-SWAP", "ETH": "ETH-USDT-SWAP", "SOL": "SOL-USDT-SWAP"}


async def _fetch_funding_oi(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch (and cache) funding-rate + open-interest for *symbol* from OKX
    public APIs (no key required, not geoblocked from Replit).

    Pulls three endpoints concurrently:
      • funding rate                — current 8h rate
      • open-interest snapshot      — current OI in USD
      • OI history (1D bars × 2)    — to compute 24h % change

    Returns None only if every call fails. Caller tolerates None so the
    AI run still proceeds; the prompt block renders an "unavailable" line.
    """
    sym = (symbol or "").upper().strip()
    cached = _FUNDING_CACHE.get(sym)
    if cached and time.time() < cached[1]:
        return cached[0]
    inst_id = _OKX_INST.get(sym)
    if not inst_id:
        return None

    fd: Dict[str, Any] = {"symbol": sym, "fetched_at": int(time.time())}
    try:
        async with httpx.AsyncClient(timeout=4.0) as cl:
            r_fund, r_oi, r_hist = await asyncio.gather(
                cl.get("https://www.okx.com/api/v5/public/funding-rate",
                       params={"instId": inst_id}),
                cl.get("https://www.okx.com/api/v5/public/open-interest",
                       params={"instType": "SWAP", "instId": inst_id}),
                cl.get("https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume",
                       params={"ccy": sym, "period": "1D"}),
                return_exceptions=True,
            )
        # Funding rate — OKX returns a decimal fraction (e.g. 0.0000428 = 0.00428%)
        # so we multiply by 100 to match the percent convention the prompt uses.
        if not isinstance(r_fund, Exception) and r_fund.status_code == 200:
            data = ((r_fund.json() or {}).get("data") or [])
            if data:
                try:
                    fr_dec = float(data[0].get("fundingRate") or 0)
                    fd["funding_rate_pct"]  = fr_dec * 100.0
                    fd["funding_exchange"]  = "OKX"
                    fd["next_funding_time"] = data[0].get("nextFundingTime")
                except (TypeError, ValueError):
                    pass
        # Current OI in USD
        if not isinstance(r_oi, Exception) and r_oi.status_code == 200:
            data = ((r_oi.json() or {}).get("data") or [])
            if data:
                try:
                    fd["open_interest_usd"] = round(float(data[0].get("oiUsd") or 0), 0)
                except (TypeError, ValueError):
                    pass
        # 24h OI delta — the history endpoint returns rows
        # [timestamp_ms, oi_usd, vol_usd] in newest-first order, daily bars.
        # Compare row[0] (today) vs row[1] (yesterday) for the % change.
        if not isinstance(r_hist, Exception) and r_hist.status_code == 200:
            rows = ((r_hist.json() or {}).get("data") or [])
            if len(rows) >= 2:
                try:
                    today = float(rows[0][1])
                    prev  = float(rows[1][1])
                    if prev > 0:
                        fd["oi_change_24h_pct"] = round((today - prev) / prev * 100.0, 2)
                except (TypeError, ValueError, IndexError):
                    pass
    except Exception as e:
        logger.debug(f"_fetch_funding_oi({sym}) failed: {e}")
        return None
    if fd.get("funding_rate_pct") is None and not fd.get("open_interest_usd"):
        return None
    _FUNDING_CACHE[sym] = (fd, time.time() + _FUNDING_TTL)
    return fd


# ─── AI plan parser ───────────────────────────────────────────────────────────
_PLAN_PATTERNS = {
    "trade":      re.compile(r"^TRADE:\s*(LONG|SHORT)\s*@?\s*\$?([0-9][0-9,\.]*)", re.IGNORECASE | re.MULTILINE),
    "order_type": re.compile(r"^ORDER_TYPE:\s*(MARKET|LIMIT)", re.IGNORECASE | re.MULTILINE),
    "stop":       re.compile(r"^STOP:\s*\$?([0-9][0-9,\.]*)", re.IGNORECASE | re.MULTILINE),
    "tp1":        re.compile(r"^TP1:\s*\$?([0-9][0-9,\.]*)", re.IGNORECASE | re.MULTILINE),
    "tp2":        re.compile(r"^TP2:\s*\$?([0-9][0-9,\.]*)", re.IGNORECASE | re.MULTILINE),
    "odds":       re.compile(r"^ODDS:\s*UP\s*([0-9]+)\s*%\s*/\s*DOWN\s*([0-9]+)\s*%", re.IGNORECASE | re.MULTILINE),
    "leverage":   re.compile(r"^LEVERAGE:\s*([^\n]+)", re.IGNORECASE | re.MULTILINE),
}


def _to_float(s: str) -> Optional[float]:
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def parse_ai_plan(plan_text: str) -> Dict[str, Any]:
    """Extract structured fields from a Claude TRADE/STOP/TP plan string.

    Returns a dict with keys: side, order_type, entry, stop, tp1, tp2,
    odds_up, odds_down, leverage. Missing fields are None.
    """
    out: Dict[str, Any] = {
        "side": None, "order_type": None,
        "entry": None, "stop": None, "tp1": None, "tp2": None,
        "odds_up": None, "odds_down": None,
        "leverage": None,
    }
    if not plan_text:
        return out
    m = _PLAN_PATTERNS["trade"].search(plan_text)
    if m:
        out["side"] = "long" if m.group(1).upper() == "LONG" else "short"
        out["entry"] = _to_float(m.group(2))
    m = _PLAN_PATTERNS["order_type"].search(plan_text)
    if m:
        out["order_type"] = m.group(1).upper()
    for k in ("stop", "tp1", "tp2"):
        m = _PLAN_PATTERNS[k].search(plan_text)
        if m:
            out[k] = _to_float(m.group(1))
    m = _PLAN_PATTERNS["odds"].search(plan_text)
    if m:
        try:
            out["odds_up"]   = int(m.group(1))
            out["odds_down"] = int(m.group(2))
        except (TypeError, ValueError):
            pass
    m = _PLAN_PATTERNS["leverage"].search(plan_text)
    if m:
        out["leverage"] = m.group(1).strip()
    return out


# ─── Rule compiler — reads chart_state and produces a tiny DSL ────────────────
def compile_rules_from_chart_state(chart_state: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Inspect the saved indicators + toggles and return a (rules, summary).

    The rules dict has shape:
      {
        "entry": {"kind": "ema_cross"|"rsi_oversold"|"macd_zero"|"st_flip",
                  "params": {...}, "side": "long"|"short"|"either"},
        "wall_confluence": bool,   # require nearby buy/sell wall
        "flow_aligned":    bool,   # require big-prints flow alignment
      }

    summary is a one-line human-readable explanation. Returns (None, reason)
    when the chart has nothing rule-friendly on it (e.g. only a VWAP).
    """
    indicators = chart_state.get("indicators") or []
    toggles    = chart_state.get("toggles") or {}

    # Find indicators by type (preserve order)
    ma_indicators: List[Dict[str, Any]] = []
    rsi_ind = macd_ind = st_ind = fvg_ind = None
    for spec in indicators:
        t = (spec.get("type") or "").lower()
        if t in ("ema", "sma"):
            ma_indicators.append(spec)
        elif t == "rsi" and not rsi_ind:
            rsi_ind = spec
        elif t == "macd" and not macd_ind:
            macd_ind = spec
        elif t == "supertrend" and not st_ind:
            st_ind = spec
        elif t == "fvg" and not fvg_ind:
            fvg_ind = spec

    rules: Optional[Dict[str, Any]] = None
    summary_bits: List[str] = []

    # Priority 0: FVG retest — if the user explicitly enabled FVG (either as
    # an indicator spec or via the chart's FVG toggle), that's an explicit
    # signal preference, so it wins over MAs/MACD/etc. Indicator-level params
    # take precedence over toggle defaults.
    # Defaults match the frontend overlay fetch (max_age_bars=200,
    # min_gap_atr_mult=0.10, disp_atr_mult=0.5) so what the user SEES on the
    # chart is what FIRES the rule.
    if fvg_ind or toggles.get("fvg"):
        fp = (fvg_ind or {}).get("params") or {}
        min_gap_pct      = float(fp.get("min_gap_pct", 0.0) or 0.0)
        min_gap_atr_mult = float(fp.get("min_gap_atr_mult", 0.10) or 0.10)
        disp_atr_mult    = float(fp.get("disp_atr_mult", 0.5) or 0.5)
        max_age_bars     = int(fp.get("max_age_bars", 200) or 200)
        # Strong-displacement instant entry — when a fresh FVG is ≥ this ×
        # ATR wide, fire entry immediately in the displacement direction
        # without waiting for a retest. 0 disables. Default 4.0 catches the
        # parabolic moves where a polite retest never comes.
        instant_atr      = float(fp.get("instant_entry_atr_mult", 4.0) or 0.0)
        instant_max_age  = int(fp.get("instant_entry_max_age", 2) or 2)
        rules = {
            "entry": {
                "kind": "fvg_retest",
                "params": {
                    "min_gap_pct":            min_gap_pct,
                    "min_gap_atr_mult":       min_gap_atr_mult,
                    "disp_atr_mult":          disp_atr_mult,
                    "max_age_bars":           max_age_bars,
                    "instant_entry_atr_mult": instant_atr,
                    "instant_entry_max_age":  instant_max_age,
                },
                "side": "either",
            },
        }
        if instant_atr > 0:
            summary_bits.append(
                f"price retests an unfilled FVG (or instant entry on ≥{instant_atr:g}×ATR displacement)"
            )
        else:
            summary_bits.append(
                f"price retests an unfilled FVG (≥{min_gap_pct:g}% gap, ≤{max_age_bars} bars old)"
            )

    # Priority 1: two MAs of different periods → cross detection
    # Gated on `rules is None` so an explicit FVG choice (Priority 0) wins
    # over MAs that happen to be on the chart for visual reference.
    if rules is None and len(ma_indicators) >= 2:
        a, b = ma_indicators[0], ma_indicators[1]
        pa = float((a.get("params") or {}).get("period") or 0)
        pb = float((b.get("params") or {}).get("period") or 0)
        if pa and pb and pa != pb:
            fast, slow = (a, b) if pa < pb else (b, a)
            rules = {
                "entry": {
                    "kind": "ma_cross",
                    "params": {
                        "fast_type":   (fast.get("type") or "ema").lower(),
                        "fast_period": int(float((fast.get("params") or {}).get("period"))),
                        "slow_type":   (slow.get("type") or "ema").lower(),
                        "slow_period": int(float((slow.get("params") or {}).get("period"))),
                    },
                    "side": "either",
                },
            }
            summary_bits.append(
                f"{rules['entry']['params']['fast_type'].upper()}({rules['entry']['params']['fast_period']}) "
                f"crosses {rules['entry']['params']['slow_type'].upper()}({rules['entry']['params']['slow_period']})"
            )

    # Priority 2: SuperTrend flip
    if rules is None and st_ind:
        period = int(float((st_ind.get("params") or {}).get("period") or 10))
        mult   = float((st_ind.get("params") or {}).get("mult") or 3.0)
        rules = {
            "entry": {
                "kind": "st_flip",
                "params": {"period": period, "mult": mult},
                "side": "either",
            },
        }
        summary_bits.append(f"SuperTrend({period},{mult:g}) flip")

    # Priority 3: MACD zero-line cross
    if rules is None and macd_ind:
        p = macd_ind.get("params") or {}
        fast   = int(float(p.get("fast")   or 12))
        slow   = int(float(p.get("slow")   or 26))
        signal = int(float(p.get("signal") or 9))
        rules = {
            "entry": {
                "kind": "macd_zero",
                "params": {"fast": fast, "slow": slow, "signal": signal},
                "side": "either",
            },
        }
        summary_bits.append(f"MACD({fast},{slow},{signal}) zero-cross")

    # Priority 4: RSI oversold/overbought reversal
    if rules is None and rsi_ind:
        period = int(float((rsi_ind.get("params") or {}).get("period") or 14))
        rules = {
            "entry": {
                "kind": "rsi_reversal",
                "params": {"period": period, "long_below": 30, "short_above": 70},
                "side": "either",
            },
        }
        summary_bits.append(f"RSI({period}) reversal at 30/70")

    if rules is None:
        return None, (
            "No rule-friendly indicators on the chart. "
            "Add two EMAs (fast + slow), an RSI, a MACD, or a SuperTrend "
            "to enable rule mode — or use AI mode."
        )

    # Modifiers — toggles enrich the rule
    if toggles.get("order_blocks"):
        rules["wall_confluence"] = True
        summary_bits.append("near a wall")
    if toggles.get("big_prints"):
        rules["flow_aligned"] = True
        summary_bits.append("flow aligned")

    return rules, "Enter when " + " + ".join(summary_bits)


# ─── Indicator value helpers used by rule eval ────────────────────────────────
def _last_two_ma(closes: List[float], kind: str, period: int) -> Tuple[Optional[float], Optional[float]]:
    """Return (ma_at_t-1, ma_at_t) — i.e. value on the prior closed bar and now."""
    if len(closes) < period + 2:
        return None, None
    fn = _ema if kind == "ema" else _sma
    prev = fn(closes[:-1], period)
    cur  = fn(closes, period)
    return prev, cur


def _last_two_rsi(closes: List[float], period: int) -> Tuple[Optional[float], Optional[float]]:
    if len(closes) < period + 3:
        return None, None
    return _rsi(closes[:-1], period), _rsi(closes, period)


def _last_two_macd(closes: List[float], fast: int, slow: int, signal: int) -> Tuple[Optional[float], Optional[float]]:
    if len(closes) < slow + signal + 2:
        return None, None
    pm = _macd(closes[:-1], fast, slow, signal)
    cm = _macd(closes, fast, slow, signal)
    if not pm or not cm:
        return None, None
    return pm[0], cm[0]   # macd line


def _last_two_supertrend(candles: List[dict], period: int, mult: float) -> Tuple[Optional[int], Optional[int]]:
    """Returns (trend_prev, trend_cur) where trend is +1 (long) or -1 (short)."""
    if len(candles) < period + 4:
        return None, None
    a = _supertrend(candles[:-1], period, mult)
    b = _supertrend(candles, period, mult)
    if not a or not b:
        return None, None
    # supertrend returns (line, direction) — direction in {1, -1}
    return int(a[1]), int(b[1])


# ─── Fair-value-gap (FVG) detection ───────────────────────────────────────────
def detect_fvgs(candles: List[dict], *,
                min_gap_pct: float = 0.0,
                min_gap_atr_mult: float = 0.0,
                disp_atr_mult: float = 0.0,
                atr_period: int = 14,
                only_unfilled: bool = True,
                max_age_bars: int = 200,
                max_results: int = 30) -> List[Dict[str, Any]]:
    """ICT-style 3-candle Fair Value Gap detector.

    A bullish FVG forms at bar i when ``candles[i-1].high < candles[i+1].low``
    — there's a price range no candle traded inside. The gap zone is
    ``[candles[i-1].high, candles[i+1].low]`` and acts as a demand/support
    pocket. Bearish FVG is the mirror: ``candles[i-1].low > candles[i+1].high``
    creates a supply/resistance pocket at ``[candles[i+1].high, candles[i-1].low]``.

    Quality filters (any can be disabled by setting to 0):

        * ``min_gap_pct``      — drop gaps narrower than this % of mid price.
        * ``min_gap_atr_mult`` — drop gaps narrower than this × ATR. ATR-based
                                 gating is volatility-aware so the same setting
                                 surfaces meaningful gaps across regimes,
                                 timeframes, and symbols.
        * ``disp_atr_mult``    — require the formation (middle) candle's body
                                 to be ≥ this × ATR. This is the ICT
                                 "displacement" filter — a real FVG forms
                                 inside a strong expansion bar, not a doji.

    Returns a list of dicts with fields::
        index    : int   — bar index where the gap formed (the middle bar).
        time     : int   — UTC seconds of the formation bar.
        side     : str   — "bull" or "bear".
        top      : float — upper price boundary of the gap.
        bottom   : float — lower price boundary of the gap.
        mid      : float — 50% / consequent-encroachment price (ICT CE).
        size_pct : float — width / mid_price * 100.
        size_atr : float — width / ATR (volatility-normalised size).
        filled   : bool  — True if a later candle traded inside the zone.
        filled_at: int|None — UTC seconds of the candle that filled it.
        age_bars : int   — bars between formation and the latest candle.
    """
    if not candles or len(candles) < 3:
        return []

    # Compute ATR up-front when any ATR-aware filter or size_atr is needed.
    needs_atr = (min_gap_atr_mult > 0) or (disp_atr_mult > 0)
    atr = _atr(candles, atr_period) if needs_atr else None
    # If ATR couldn't be computed (not enough bars), disable ATR filters
    # gracefully rather than rejecting every gap.
    if needs_atr and (atr is None or atr <= 0):
        atr = None

    out: List[Dict[str, Any]] = []
    n = len(candles)
    start = max(1, n - max_age_bars)
    for i in range(start, n - 1):
        mid_c  = candles[i]
        prev_c = candles[i - 1]
        next_c = candles[i + 1]
        try:
            ph, pl = float(prev_c["high"]), float(prev_c["low"])
            nh, nl = float(next_c["high"]), float(next_c["low"])
            mo, mc = float(mid_c["open"]),  float(mid_c["close"])
        except (KeyError, TypeError, ValueError):
            continue

        side: Optional[str] = None
        top = bottom = 0.0
        if ph < nl:
            side, top, bottom = "bull", nl, ph
        elif pl > nh:
            side, top, bottom = "bear", pl, nh
        if side is None:
            continue

        gap_mid = (top + bottom) / 2.0
        if gap_mid <= 0:
            continue
        width    = top - bottom
        size_pct = width / gap_mid * 100.0

        # Width filter (% of price)
        if min_gap_pct > 0 and size_pct < min_gap_pct:
            continue

        # Width filter (× ATR) — only applied when ATR is available.
        size_atr = (width / atr) if atr else 0.0
        if atr and min_gap_atr_mult > 0 and size_atr < min_gap_atr_mult:
            continue

        # ICT displacement filter — formation candle must be a real expansion
        # bar, not a doji that happened to leave a gap on either side.
        if atr and disp_atr_mult > 0:
            disp = abs(mc - mo)
            if disp < disp_atr_mult * atr:
                continue

        # Walk forward to determine fill status. A fill = any candle whose
        # range overlaps the gap. (Strict ICT definition uses "fully closed
        # through"; we use "touched" so retests are flagged early — better
        # for risk management.)
        filled = False
        filled_at: Optional[int] = None
        for j in range(i + 2, n):
            cj = candles[j]
            try:
                hj, lj = float(cj["high"]), float(cj["low"])
            except (KeyError, TypeError, ValueError):
                continue
            if lj <= top and hj >= bottom:
                filled = True
                filled_at = int(cj.get("time") or 0)
                break

        if only_unfilled and filled:
            continue

        out.append({
            "index":     i,
            "time":      int(mid_c.get("time") or 0),
            "side":      side,
            "top":       round(top, 8),
            "bottom":    round(bottom, 8),
            "mid":       round(gap_mid, 8),
            "size_pct":  round(size_pct, 3),
            "size_atr":  round(size_atr, 3),
            "filled":    filled,
            "filled_at": filled_at,
            "age_bars":  (n - 1) - i,
        })

    # Newest first, capped.
    out.sort(key=lambda g: g["index"], reverse=True)
    return out[:max_results]


def _fvg_retest_signal(candles: List[dict], *,
                       min_gap_pct: float = 0.0,
                       min_gap_atr_mult: float = 0.10,
                       disp_atr_mult: float = 0.5,
                       max_age_bars: int = 200,
                       instant_entry_atr_mult: float = 0.0,
                       instant_entry_max_age: int = 2) -> Tuple[Optional[str], str]:
    """Return ('long'|'short'|None, note) when the current bar retests an
    active FVG. Used by evaluate_rules + the AI Read summariser.

    Long  = current bar's low dips into an unfilled BULL FVG below price.
    Short = current bar's high pierces an unfilled BEAR FVG above price.

    The detector defaults apply the same volatility-aware width filter and
    ICT displacement filter the chart overlay uses, so what fires the rule
    matches what the user sees on screen.

    Strong-displacement instant entry:
        When ``instant_entry_atr_mult > 0``, a fresh FVG (formed in the last
        ``instant_entry_max_age`` bars) whose width is ≥ that multiple of ATR
        triggers an immediate entry in the displacement direction — no retest
        required. This is for runaway displacements where waiting for a wick
        back into the gap usually means missing the move entirely.
    """
    if not candles or len(candles) < 4:
        return None, ""
    cur = candles[-1]
    try:
        cur_high  = float(cur["high"])
        cur_low   = float(cur["low"])
        cur_close = float(cur["close"])
    except (KeyError, TypeError, ValueError):
        return None, ""

    # Look at gaps formed BEFORE the current bar (no lookahead).
    gaps = detect_fvgs(candles[:-1],
                       min_gap_pct=min_gap_pct,
                       min_gap_atr_mult=min_gap_atr_mult,
                       disp_atr_mult=disp_atr_mult,
                       only_unfilled=True,
                       max_age_bars=max_age_bars,
                       max_results=20)
    if not gaps:
        return None, ""

    # Strong-displacement instant entry — checked FIRST so we don't miss
    # parabolic breakouts while waiting for a polite retest. Only fires on
    # FVGs formed within the last `instant_entry_max_age` bars (default 2)
    # so we're trading the displacement, not chasing stale gaps.
    if instant_entry_atr_mult > 0:
        fresh_strong = [g for g in gaps
                        if g.get("age_bars", 999) <= instant_entry_max_age
                        and g.get("size_atr", 0.0) >= instant_entry_atr_mult]
        if fresh_strong:
            # gaps is already sorted newest-first; pick the freshest, then
            # break ties by largest size_atr.
            fresh_strong.sort(key=lambda g: (g["age_bars"], -g["size_atr"]))
            g = fresh_strong[0]
            if g["side"] == "bull":
                return "long", (
                    f"strong bull FVG displacement ({g['size_atr']:.2f}×ATR, "
                    f"{g['age_bars']}b old) — instant entry"
                )
            else:
                return "short", (
                    f"strong bear FVG displacement ({g['size_atr']:.2f}×ATR, "
                    f"{g['age_bars']}b old) — instant entry"
                )

    # Bullish retest: an unfilled BULL FVG below current close, and the
    # current bar wicked into it.
    bull_below = [g for g in gaps if g["side"] == "bull" and g["top"] <= cur_close]
    if bull_below:
        nearest = max(bull_below, key=lambda g: g["top"])
        if cur_low <= nearest["top"]:
            return "long", (
                f"price retested bull FVG @ {nearest['bottom']:.2f}–{nearest['top']:.2f} "
                f"({nearest['size_pct']:.2f}% gap, {nearest['age_bars']} bars old)"
            )

    # Bearish retest: an unfilled BEAR FVG above current close, current bar
    # wicked up into it.
    bear_above = [g for g in gaps if g["side"] == "bear" and g["bottom"] >= cur_close]
    if bear_above:
        nearest = min(bear_above, key=lambda g: g["bottom"])
        if cur_high >= nearest["bottom"]:
            return "short", (
                f"price retested bear FVG @ {nearest['bottom']:.2f}–{nearest['top']:.2f} "
                f"({nearest['size_pct']:.2f}% gap, {nearest['age_bars']} bars old)"
            )

    return None, ""


# ─── Wall confluence ──────────────────────────────────────────────────────────
def _nearest_wall(walls: List[dict], price: float, side: str) -> Optional[dict]:
    """Side 'long' → look at buy walls (support below). Side 'short' → sell walls."""
    if not walls:
        return None
    matching = [w for w in walls if (w.get("side") or "").lower() == ("buy" if side == "long" else "sell")]
    if not matching:
        return None
    matching.sort(key=lambda w: abs(float(w.get("price") or 0) - price))
    return matching[0]


def _wall_within(walls: List[dict], price: float, side: str, max_pct: float = _WALL_PROX_PCT) -> bool:
    w = _nearest_wall(walls, price, side)
    if not w:
        return False
    wp = float(w.get("price") or 0)
    if wp <= 0:
        return False
    return abs(price - wp) / wp * 100.0 <= max_pct


# ─── Rule evaluator ───────────────────────────────────────────────────────────
def evaluate_rules(rules: Dict[str, Any], candles: List[dict],
                   walls: Optional[List[dict]] = None,
                   tape: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Returns an entry dict {side, entry_price, source_note} or None."""
    if not candles or len(candles) < 30:
        return None
    closes = [c["close"] for c in candles]
    last_price = closes[-1]
    entry = rules.get("entry") or {}
    kind  = entry.get("kind")
    p     = entry.get("params") or {}

    side: Optional[str] = None
    note = ""

    if kind == "ma_cross":
        fp = _last_two_ma(closes, p["fast_type"], int(p["fast_period"]))
        sp = _last_two_ma(closes, p["slow_type"], int(p["slow_period"]))
        if None in fp or None in sp:
            return None
        prev_diff, cur_diff = fp[0] - sp[0], fp[1] - sp[1]
        if prev_diff <= 0 and cur_diff > 0:
            side, note = "long",  f"{p['fast_type'].upper()}({p['fast_period']}) crossed above {p['slow_type'].upper()}({p['slow_period']})"
        elif prev_diff >= 0 and cur_diff < 0:
            side, note = "short", f"{p['fast_type'].upper()}({p['fast_period']}) crossed below {p['slow_type'].upper()}({p['slow_period']})"

    elif kind == "st_flip":
        pa, cu = _last_two_supertrend(candles, int(p["period"]), float(p["mult"]))
        if pa is None or cu is None or pa == cu:
            return None
        if cu > 0:
            side, note = "long",  f"SuperTrend flipped long"
        else:
            side, note = "short", f"SuperTrend flipped short"

    elif kind == "macd_zero":
        pm, cm = _last_two_macd(closes, int(p["fast"]), int(p["slow"]), int(p["signal"]))
        if pm is None or cm is None:
            return None
        if pm <= 0 and cm > 0:
            side, note = "long",  "MACD crossed above zero"
        elif pm >= 0 and cm < 0:
            side, note = "short", "MACD crossed below zero"

    elif kind == "rsi_reversal":
        pr, cr = _last_two_rsi(closes, int(p["period"]))
        if pr is None or cr is None:
            return None
        long_below  = float(p.get("long_below",  30))
        short_above = float(p.get("short_above", 70))
        if pr < long_below and cr >= long_below:
            side, note = "long",  f"RSI bounced out of oversold ({long_below:.0f})"
        elif pr > short_above and cr <= short_above:
            side, note = "short", f"RSI rejected from overbought ({short_above:.0f})"

    elif kind == "fvg_retest":
        # ICT-style entry: open a trade only when the current bar wicks back
        # into an unfilled fair-value gap. The gap acts as support (long) or
        # resistance (short). Defaults match the chart overlay fetch so the
        # visible zones are the eligible ones.
        # Back-compat: legacy strategies (saved before ATR/displacement
        # filters existed) only had `min_gap_pct` + `max_age_bars`. If the
        # new ATR keys are absent we treat the spec as legacy and disable
        # the new filters so old strategies don't silently get tightened.
        is_legacy = ("min_gap_pct" in (p or {})
                     and "min_gap_atr_mult" not in (p or {}))
        default_atr_mult  = 0.0 if is_legacy else 0.10
        default_disp_mult = 0.0 if is_legacy else 0.5
        default_max_age   = 100 if is_legacy else 200
        # Strong-displacement instant entry: 0 disables, default 4.0 (legacy
        # specs that pre-date this field default to 0 to preserve behaviour).
        default_instant = 0.0 if is_legacy else 4.0
        side, note = _fvg_retest_signal(
            candles,
            min_gap_pct=float(p.get("min_gap_pct", 0.05 if is_legacy else 0.0) or 0.0),
            min_gap_atr_mult=float(p.get("min_gap_atr_mult", default_atr_mult) or 0.0),
            disp_atr_mult=float(p.get("disp_atr_mult", default_disp_mult) or 0.0),
            max_age_bars=int(p.get("max_age_bars", default_max_age) or default_max_age),
            instant_entry_atr_mult=float(p.get("instant_entry_atr_mult", default_instant) or 0.0),
            instant_entry_max_age=int(p.get("instant_entry_max_age", 2) or 2),
        )
        if side is None:
            return None

    if side is None:
        return None

    # Modifiers — wall confluence + flow alignment (best-effort, tolerant)
    if rules.get("wall_confluence"):
        if walls and not _wall_within(walls, last_price, side):
            return None
    if rules.get("flow_aligned") and tape:
        buy_usd  = float(tape.get("buy_usd")  or 0)
        sell_usd = float(tape.get("sell_usd") or 0)
        if side == "long"  and buy_usd  < sell_usd:
            return None
        if side == "short" and sell_usd < buy_usd:
            return None

    return {"side": side, "entry_price": last_price, "source_note": note}


# ─── TP/SL helpers (rule mode) ────────────────────────────────────────────────
# Minimum TP1 distance as a percentage of entry. On low timeframes (1m/3m)
# raw ATR can be tiny — a 2×ATR TP becomes a sub-0.1% target that real-world
# fees + spread eat alive. This floor ensures every trade has at least a
# meaningful edge in price terms before sizing/leverage is even considered.
# Override via env MIN_TP1_PCT.
try:
    _MIN_TP1_PCT = max(0.0, float(os.environ.get("MIN_TP1_PCT", "0.5")))
except (TypeError, ValueError):
    _MIN_TP1_PCT = 0.5


def derive_stop_tp_from_chart(side: str, entry: float, candles: List[dict],
                              walls: Optional[List[dict]] = None,
                              source: str = "walls") -> Tuple[float, float, float]:
    """Pick (stop, tp1, tp2) for rule-mode entries.

    'walls' source: stop = nearest opposite-side wall (or 0.7% fallback);
                    tp1 = nearest same-side resistance; tp2 = 1.5× tp1 distance.
    'atr' source: stop = entry ± 1×ATR(14); tp1 = entry ± 2×ATR; tp2 = 3×ATR.

    All paths enforce a minimum TP1 distance of ``_MIN_TP1_PCT`` % of entry
    (default 0.5%) so we never open a noise-scalp the moment price wiggles.
    The R:R ratio is preserved (TP2 stays ≥ 1.5× TP1 distance, stop unchanged
    unless tightening it would invert the trade).
    """
    atr = _atr(candles, 14) or (entry * 0.005)
    min_tp1_dist = entry * (_MIN_TP1_PCT / 100.0)

    if source == "atr" or not walls:
        # Lift the ATR-based target up to the floor when raw ATR is too small.
        tp1_dist = max(atr * 2.0, min_tp1_dist)
        tp2_dist = max(atr * 3.0, tp1_dist * 1.5)
        if side == "long":
            stop = entry - atr
            tp1  = entry + tp1_dist
            tp2  = entry + tp2_dist
        else:
            stop = entry + atr
            tp1  = entry - tp1_dist
            tp2  = entry - tp2_dist
        return stop, tp1, tp2

    # walls source — find the nearest support/resistance on the right side
    if side == "long":
        below = [float(w["price"]) for w in walls
                 if (w.get("side") or "").lower() == "buy" and float(w.get("price") or 0) < entry]
        above = [float(w["price"]) for w in walls
                 if (w.get("side") or "").lower() == "sell" and float(w.get("price") or 0) > entry]
        stop = (max(below) if below else entry - atr)
        # Make sure the stop isn't too tight (< 0.3% from entry)
        stop = min(stop, entry - max(atr * 0.5, entry * 0.003))
        tp1  = (min(above) if above else entry + atr * 2.0)
        # Enforce minimum TP1 distance — push TP1 further out if walls are
        # too close to be meaningful.
        tp1  = max(tp1, entry + min_tp1_dist)
        tp2  = entry + (tp1 - entry) * 1.6
    else:
        above = [float(w["price"]) for w in walls
                 if (w.get("side") or "").lower() == "sell" and float(w.get("price") or 0) > entry]
        below = [float(w["price"]) for w in walls
                 if (w.get("side") or "").lower() == "buy" and float(w.get("price") or 0) < entry]
        stop = (min(above) if above else entry + atr)
        stop = max(stop, entry + max(atr * 0.5, entry * 0.003))
        tp1  = (max(below) if below else entry - atr * 2.0)
        tp1  = min(tp1, entry - min_tp1_dist)
        tp2  = entry - (entry - tp1) * 1.6
    return stop, tp1, tp2


# ─── Position simulator ───────────────────────────────────────────────────────
def check_position_close(trade, candles: List[dict]) -> Optional[Tuple[str, float]]:
    """Return (reason, exit_price) if a candle since the trade opened hit
    stop or TP. v1 closes at TP1; once a partial fired (`tp1_hit=True`) the
    remaining leg runs to TP2 (or the stop, which may already have been moved
    to breakeven by the partial handler).

    Critical: when tp1_hit is True we MUST only evaluate candles AFTER the
    partial fired, otherwise we'd retroactively stop-out the trade on old
    candles using the new (BE) stop — that's lookahead bias.

    `trade` is an AutoTradePaperTrade ORM row.
    """
    if not candles:
        return None

    side = trade.side
    stop = float(trade.stop_price)

    # After a partial TP1 fires, the trade is hunting TP2. Use tp2_price
    # if defined, otherwise the trade simply runs to its (possibly-moved)
    # stop. Do NOT re-evaluate TP1 — it's already been realised.
    tp1_already_hit = bool(getattr(trade, "tp1_hit", False))
    if tp1_already_hit:
        tp_target = float(getattr(trade, "tp2_price", 0) or 0) or None
        anchor_dt = getattr(trade, "tp1_hit_at", None) or trade.opened_at
    else:
        tp_target = float(trade.tp1_price)
        anchor_dt = trade.opened_at

    anchor_ts = int(anchor_dt.timestamp()) if anchor_dt else 0
    relevant = [c for c in candles if int(c["time"]) >= anchor_ts]
    if not relevant and not tp1_already_hit:
        # Trade opened after the latest candle — nothing to check yet.
        relevant = candles[-3:]
    if not relevant:
        return None

    for c in relevant:
        hi, lo = float(c["high"]), float(c["low"])
        if side == "long":
            # Pessimistic ordering: stop wins on a wick that hit both
            if lo <= stop:
                return ("be_stop" if tp1_already_hit else "stop_hit"), stop
            if tp_target is not None and hi >= tp_target:
                return ("tp2_hit" if tp1_already_hit else "tp1_hit"), tp_target
        else:
            if hi >= stop:
                return ("be_stop" if tp1_already_hit else "stop_hit"), stop
            if tp_target is not None and lo <= tp_target:
                return ("tp2_hit" if tp1_already_hit else "tp1_hit"), tp_target
    return None


def compute_pnl(side: str, entry: float, exit_p: float,
                notional: float, leverage: int) -> Tuple[float, float]:
    """Return (pnl_pct, pnl_usd) where pnl_pct is the raw price move and
    pnl_usd applies the user's leverage to their notional."""
    if entry <= 0:
        return 0.0, 0.0
    move_pct = ((exit_p - entry) / entry) * 100.0
    if side == "short":
        move_pct = -move_pct
    pnl_usd = (move_pct / 100.0) * notional * max(1, int(leverage))
    return move_pct, pnl_usd


# ─── Telegram DM helpers ──────────────────────────────────────────────────────
def _fmt_price(v: float) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1000: return f"${v:,.2f}"
    if abs(v) >= 1:    return f"${v:,.4f}"
    return f"${v:.6f}"


async def _dm_open(user_telegram_id: int, strat_name: str, trade, plan_text: Optional[str]):
    side_emoji = "🟢" if trade.side == "long" else "🔴"
    body = (
        f"{side_emoji} <b>Auto strategy opened a paper trade</b>\n"
        f"<i>{strat_name}</i>\n\n"
        f"• <b>{trade.side.upper()}</b> {trade.symbol} {trade.timeframe}\n"
        f"• Entry: {_fmt_price(trade.entry_price)}\n"
        f"• Stop:  {_fmt_price(trade.stop_price)}\n"
        f"• TP1:   {_fmt_price(trade.tp1_price)}\n"
        + (f"• TP2:   {_fmt_price(trade.tp2_price)}\n" if trade.tp2_price else "")
        + f"• Size:  ${trade.notional_usd:,.0f} @ {trade.leverage}x (paper)\n"
        f"• Source: {trade.source.upper()}\n"
    )
    if plan_text and trade.source == "ai":
        body += f"\n<pre>{plan_text[:600]}</pre>"
    try:
        await _tg_send(user_telegram_id, body)
    except Exception as e:
        logger.warning(f"auto_trader DM open failed: {e}")


async def _dm_close(user_telegram_id: int, strat_name: str, trade, reason: str,
                    exit_price: float, pnl_pct: float, pnl_usd: float):
    win = pnl_usd >= 0
    head = "✅ <b>TP HIT</b>" if reason == "tp1_hit" else ("🛑 <b>STOPPED OUT</b>" if reason == "stop_hit" else "<b>Closed</b>")

    # When the trade did a partial-TP1 + runner, `pnl_pct` is the FINAL leg's
    # raw price move (often near 0 for a BE-stop close), which makes the DM
    # look like "Move +0.01% → +$138" — confusing. Compute the EFFECTIVE
    # blended move by inverting the leverage formula on the total $ P&L
    # against the original notional, so the % matches the user's reaction.
    partial_pnl = float(getattr(trade, "partial_pnl_usd", 0.0) or 0.0)
    orig_notional = float(getattr(trade, "original_notional_usd", 0) or trade.notional_usd or 0)
    lev = max(1, int(trade.leverage or 1))
    if orig_notional > 0:
        effective_move = pnl_usd / (orig_notional * lev) * 100.0
    else:
        effective_move = pnl_pct
    # 3 decimals when the move is tiny, otherwise 2 — so the user can see
    # "+0.139%" instead of a confusingly-rounded "+0.14%" or "+0.01%".
    move_str = (f"{effective_move:+.3f}%" if abs(effective_move) < 0.5
                else f"{effective_move:+.2f}%")

    body = (
        f"{head} — auto strategy paper trade\n"
        f"<i>{strat_name}</i>\n\n"
        f"• {trade.side.upper()} {trade.symbol} {trade.timeframe}\n"
        f"• Entry:  {_fmt_price(trade.entry_price)}\n"
        f"• Exit:   {_fmt_price(exit_price)}\n"
        f"• Move:   {move_str}\n"
        f"• P&amp;L:    {'+' if win else ''}${pnl_usd:,.2f} ({lev}x on ${orig_notional:,.0f})\n"
    )
    if partial_pnl != 0.0:
        # Show the breakdown so the user sees which leg paid what.
        leg_pnl = pnl_usd - partial_pnl
        body += (
            f"• Legs:   TP1 partial {'+' if partial_pnl >= 0 else ''}${partial_pnl:,.2f} "
            f"+ runner {'+' if leg_pnl >= 0 else ''}${leg_pnl:,.2f}\n"
        )
    try:
        await _tg_send(user_telegram_id, body)
    except Exception as e:
        logger.warning(f"auto_trader DM close failed: {e}")


# ─── Helpers to build chart-context for AI re-runs ────────────────────────────
async def _fetch_walls_for(symbol: str) -> Optional[Dict[str, Any]]:
    """Re-use the trade page's wall report cache if present."""
    try:
        from app.services.liquidity_walls import scan_walls
        from strategy_portal_server import _CACHE
    except Exception:
        return None
    pair_map = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
    pair = pair_map.get(symbol.upper())
    if not pair:
        return None
    wkey = f"trade_walls_{pair}_0"
    cached = _CACHE.get(wkey)
    if cached and time.time() < cached[1]:
        return cached[0]
    try:
        rep = await scan_walls(pair, use_ai=False)
        if not rep:
            return None
        d = asdict(rep)
        wb = d.get("wall_behavior") or {}
        d["wall_behavior"] = {str(k): v for k, v in wb.items()}
        _CACHE[wkey] = (d, time.time() + 25)
        return d
    except Exception as e:
        logger.debug(f"auto_trader wall fetch failed for {symbol}: {e}")
        return None


def _walls_flat_list(wall_report: Optional[dict]) -> List[dict]:
    """Flatten wall_report → simple [{side, price, usd_value}] list for rule eval."""
    if not wall_report:
        return []
    out: List[dict] = []
    for side_key in ("buy_walls", "sell_walls"):
        for w in (wall_report.get(side_key) or []):
            try:
                out.append({
                    "side":  "buy" if side_key == "buy_walls" else "sell",
                    "price": float(w.get("price") or 0),
                    "usd":   float(w.get("usd_value") or 0),
                })
            except (TypeError, ValueError):
                continue
    return out


# ─── Risk caps / sessions / cooldowns / sizing (G, H, J, K) ──────────────────
def _today_utc_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _hhmm_to_minutes(s: Optional[str]) -> Optional[int]:
    """Parse 'HH:MM' (UTC) into minutes-since-midnight; None on failure."""
    if not s:
        return None
    try:
        h, m = s.strip().split(":", 1)
        return (int(h) % 24) * 60 + (int(m) % 60)
    except Exception:
        return None


def _in_session(strategy, now: datetime) -> bool:
    """J: True if the strategy is allowed to fire at `now` (UTC).

    Both endpoints null/empty → no filter (always allowed). Supports overnight
    windows (e.g. 22:00 → 04:00 → "during the Asia open").
    """
    start = _hhmm_to_minutes(getattr(strategy, "session_start_utc", None))
    end   = _hhmm_to_minutes(getattr(strategy, "session_end_utc",   None))
    if start is None or end is None:
        return True
    cur = now.hour * 60 + now.minute
    if start == end:
        return True              # 24h window
    if start < end:
        return start <= cur < end
    # Overnight wrap (e.g. 22:00 → 04:00)
    return cur >= start or cur < end


def _roll_daily_loss(strategy) -> None:
    """Reset daily_loss_today_usd whenever we cross into a new UTC day."""
    today = _today_utc_str()
    if (strategy.daily_loss_date or "") != today:
        strategy.daily_loss_date      = today
        strategy.daily_loss_today_usd = 0.0


def _pre_open_gates(strategy, now: datetime) -> Tuple[bool, str]:
    """G + J + K — return (allowed, reason). Reason is empty string when allowed.

    Called BEFORE any AI/rule evaluation so we don't waste an API call on
    a strategy that isn't even allowed to open right now.
    """
    # K — explicit pause window from a prior cooldown.
    # Once the pause expires we also reset the consecutive-loss counter so the
    # bot is genuinely "back in business" — otherwise the streak gate below
    # would deadlock the strategy forever (no new trades ⇒ no wins ⇒ never
    # resets). This makes the cooldown the discipline mechanism, not a
    # permanent ban.
    paused_until = getattr(strategy, "paused_until", None)
    if paused_until:
        if now < paused_until:
            return False, f"cooldown_until_{paused_until.isoformat(timespec='minutes')}"
        # Cooldown finished — wipe both flags so we can trade again.
        strategy.paused_until = None
        strategy.consecutive_losses = 0

    # J — time-of-day session
    if not _in_session(strategy, now):
        return False, "outside_session_window"

    # G — daily loss cap (account for today's realised NET P&L; wins offset
    # losses so the cap matches the user's intuition that it's the day's bottom
    # line, not just gross losses).
    cap = getattr(strategy, "max_daily_loss_usd", None)
    if cap and cap > 0:
        _roll_daily_loss(strategy)
        if (strategy.daily_loss_today_usd or 0.0) <= -abs(cap):
            return False, f"daily_loss_cap_hit_{abs(cap):.0f}"

    return True, ""


def _compute_position_size(strategy, *, side: str, entry: float, stop: float) -> float:
    """H — return the USD notional to use for this trade.

    Modes:
      * 'fixed'    — strategy.notional_usd (legacy behaviour)
      * 'risk_pct' — size so that (entry → stop) move loses exactly
                     account_size_usd × risk_pct% of equity, given leverage.

    Falls back to fixed if the inputs are degenerate (zero stop distance, etc).
    """
    fixed = float(strategy.notional_usd or 1000.0)
    mode = (getattr(strategy, "position_sizing_mode", "fixed") or "fixed").lower()
    if mode != "risk_pct":
        return fixed

    risk_pct = float(getattr(strategy, "risk_pct", 0) or 0)
    acct     = float(getattr(strategy, "account_size_usd", 0) or 0)
    lev      = max(1, int(strategy.leverage or 1))
    if risk_pct <= 0 or acct <= 0 or entry <= 0 or stop <= 0:
        return fixed

    # Stop distance as a fraction of entry (always positive).
    if side == "long":
        if stop >= entry:
            return fixed
        stop_frac = (entry - stop) / entry
    else:
        if stop <= entry:
            return fixed
        stop_frac = (stop - entry) / entry
    if stop_frac <= 0:
        return fixed

    risk_usd = acct * (risk_pct / 100.0)
    # notional × leverage × stop_frac == risk_usd  ⇒  notional = risk_usd / (lev × stop_frac)
    notional = risk_usd / (lev * stop_frac)
    # Sanity clamps to avoid degenerate sizes
    return max(50.0, min(notional, 1_000_000.0))


def _maybe_partial_close_and_breakeven(strategy, trade, candles: List[dict],
                                       db) -> bool:
    """I — when TP1 prints, optionally realise a partial leg and slide the
    stop to entry. Returns True if any state changed.

    The trade stays OPEN; the remainder runs to TP2 / final stop. On the
    final close, `_resolve_open_trades_for` adds `partial_pnl_usd` so the
    aggregate book matches reality.
    """
    if not getattr(strategy, "enable_partial_tp1", False):
        return False
    if trade.tp1_hit:
        return False  # already partialed on a prior tick

    # Check whether TP1 has been touched since the trade opened.
    open_ts = int(trade.opened_at.timestamp()) if trade.opened_at else 0
    relevant = [c for c in candles if int(c["time"]) >= open_ts] or candles[-3:]
    side = trade.side
    tp1  = float(trade.tp1_price)
    hit  = False
    for c in relevant:
        hi, lo = float(c["high"]), float(c["low"])
        if side == "long" and hi >= tp1:
            hit = True; break
        if side == "short" and lo <= tp1:
            hit = True; break
    if not hit:
        return False

    # Partial leg sizing — close partial_tp1_pct% of the position at TP1.
    pct = float(strategy.partial_tp1_pct or 50.0)
    pct = max(1.0, min(99.0, pct)) / 100.0
    full_notional = float(trade.notional_usd or 0)
    if full_notional <= 0:
        return False
    closed_notional    = full_notional * pct
    remaining_notional = full_notional - closed_notional

    _, partial_pnl = compute_pnl(side, trade.entry_price, tp1,
                                 closed_notional, trade.leverage)

    trade.tp1_hit                = True
    trade.tp1_hit_at             = datetime.utcnow()
    trade.partial_pnl_usd        = (trade.partial_pnl_usd or 0.0) + partial_pnl
    trade.original_notional_usd  = full_notional
    trade.remaining_notional_usd = remaining_notional
    # The remaining leg trades on the smaller notional from now on.
    trade.notional_usd           = remaining_notional

    if getattr(strategy, "move_stop_to_be_after_tp1", False):
        # Slide stop to entry. Slightly inside (+/- 1bp) so a single-tick wick
        # doesn't immediately ping it.
        be_buffer = trade.entry_price * 0.0001
        if side == "long":
            trade.stop_price       = trade.entry_price + be_buffer
        else:
            trade.stop_price       = trade.entry_price - be_buffer
        trade.stop_moved_to_be = True

    db.commit()
    logger.info(
        f"auto_trader strategy #{strategy.id} trade #{trade.id} TP1 partial closed "
        f"({pct*100:.0f}% = ${closed_notional:.0f}, partial P&L ${partial_pnl:+.2f})"
        + (" + stop→BE" if trade.stop_moved_to_be else "")
    )
    return True


# ─── Main per-strategy evaluation ─────────────────────────────────────────────
async def _resolve_open_trades_for(strategy, db) -> bool:
    """Close any open paper trades that hit stop/TP. Returns True if any closed."""
    from app.models import AutoTradePaperTrade, User
    open_trades = (db.query(AutoTradePaperTrade)
                     .filter(AutoTradePaperTrade.strategy_id == strategy.id,
                             AutoTradePaperTrade.status == "open")
                     .all())
    if not open_trades:
        return False
    candles = await _fetch_candles(strategy.symbol, strategy.timeframe, limit=200)
    if not candles:
        return False

    closed_any = False
    for t in open_trades:
        # I — first see whether TP1 just printed; if partials are enabled we
        # realise the partial leg + slide stop to BE without closing the trade.
        try:
            _maybe_partial_close_and_breakeven(strategy, t, candles, db)
        except Exception as e:
            logger.warning(f"auto_trader partial-TP handler failed for trade #{t.id}: {e}")

        result = check_position_close(t, candles)
        if not result:
            continue
        reason, exit_price = result
        # The "remaining leg" P&L runs on whatever notional is left after a partial.
        pnl_pct, leg_pnl = compute_pnl(t.side, t.entry_price, exit_price,
                                       t.notional_usd, t.leverage)
        # Combine partial + remaining leg for the trade's reported P&L.
        partial_pnl  = float(t.partial_pnl_usd or 0.0)
        total_pnl    = leg_pnl + partial_pnl
        t.status     = reason
        t.exit_price = exit_price
        t.pnl_pct    = pnl_pct
        t.pnl_usd    = total_pnl
        t.closed_at  = datetime.utcnow()

        # Aggregate stats use the *combined* P&L so the dashboard math matches.
        strategy.pnl_usd_total = (strategy.pnl_usd_total or 0.0) + total_pnl
        # G — daily NET P&L tracker. Wins offset losses so the cap reflects
        # the day's bottom line (matching the user-facing copy).
        _roll_daily_loss(strategy)
        strategy.daily_loss_today_usd = (strategy.daily_loss_today_usd or 0.0) + total_pnl

        if total_pnl >= 0:
            strategy.wins               = (strategy.wins   or 0) + 1
            strategy.consecutive_losses = 0     # K — winning resets the streak
        else:
            strategy.losses             = (strategy.losses or 0) + 1
            strategy.consecutive_losses = (strategy.consecutive_losses or 0) + 1
            # K — only START cooldown when the user-defined streak threshold
            # is reached. Otherwise every loss would trigger a pause, which
            # is way more aggressive than "pause after N losses".
            max_streak = int(getattr(strategy, "max_consecutive_losses", 0) or 0)
            cd         = int(getattr(strategy, "cooldown_minutes_after_loss", 0) or 0)
            if max_streak > 0 and cd > 0 and strategy.consecutive_losses >= max_streak:
                strategy.paused_until = datetime.utcnow() + timedelta(minutes=cd)
        db.commit()
        closed_any = True

        # DM the user
        if strategy.notify_telegram:
            user = db.query(User).filter(User.id == strategy.user_id).first()
            if user and user.telegram_id and not str(user.telegram_id).startswith("WEB-"):
                try:
                    tg_id = int(user.telegram_id)
                    await _dm_close(tg_id, strategy.name, t, reason, exit_price, pnl_pct, total_pnl)
                except (TypeError, ValueError):
                    pass
    return closed_any


async def _open_paper_trade(strategy, *, side: str, entry: float, stop: float,
                            tp1: float, tp2: Optional[float], source: str,
                            plan_text: Optional[str], db) -> Optional[Any]:
    from app.models import AutoTradePaperTrade, User

    # Sanity-check the levels — refuse to open malformed trades
    if entry <= 0 or stop <= 0 or tp1 <= 0:
        return None
    if side == "long" and not (stop < entry < tp1):
        return None
    if side == "short" and not (tp1 < entry < stop):
        return None

    # H — compute the position size; risk_pct mode auto-derives notional from
    # account size + stop distance, fixed mode keeps the existing behaviour.
    notional = _compute_position_size(strategy, side=side, entry=entry, stop=stop)

    t = AutoTradePaperTrade(
        strategy_id            = strategy.id,
        user_id                = strategy.user_id,
        symbol                 = strategy.symbol,
        timeframe              = strategy.timeframe,
        side                   = side,
        source                 = source,
        order_type             = "MARKET",
        entry_price            = entry,
        stop_price             = stop,
        tp1_price              = tp1,
        tp2_price              = tp2,
        notional_usd           = notional,
        leverage               = strategy.leverage,
        plan_text              = plan_text,
        # I — partial-TP bookkeeping starts at "no partial yet"
        original_notional_usd  = notional,
        remaining_notional_usd = notional,
    )
    db.add(t)
    strategy.last_signal_at = datetime.utcnow()
    strategy.total_signals  = (strategy.total_signals or 0) + 1
    db.commit()
    db.refresh(t)

    if strategy.notify_telegram:
        user = db.query(User).filter(User.id == strategy.user_id).first()
        if user and user.telegram_id and not str(user.telegram_id).startswith("WEB-"):
            try:
                tg_id = int(user.telegram_id)
                await _dm_open(tg_id, strategy.name, t, plan_text)
            except (TypeError, ValueError):
                pass
    return t


async def _maybe_open_ai(strategy, db) -> bool:
    """AI mode: throttled by cadence_min, fires when ODDS clears threshold."""
    now = datetime.utcnow()
    last = strategy.last_evaluated_at
    if last and (now - last).total_seconds() < max(_AI_TICK_FLOOR_S, strategy.cadence_min * 60):
        return False
    chart_state = json.loads(strategy.chart_state_json or "{}")
    indicators  = chart_state.get("indicators") or []
    toggles     = chart_state.get("toggles") or {}
    tape        = chart_state.get("tape") or {}

    candles = await _fetch_candles(strategy.symbol, strategy.timeframe, limit=300)
    if not candles or len(candles) < 30:
        return False

    wall_report = await _fetch_walls_for(strategy.symbol)

    # HTF context (1H + 4H) and funding/OI run concurrently — both are
    # best-effort enrichment, the AI run still proceeds if any fail.
    htf_1h: List[dict] = []
    htf_4h: List[dict] = []
    funding_data: Optional[Dict[str, Any]] = None
    try:
        htf_1h_res, htf_4h_res, funding_res = await asyncio.gather(
            _fetch_candles(strategy.symbol, "1h", limit=120),
            _fetch_candles(strategy.symbol, "4h", limit=120),
            _fetch_funding_oi(strategy.symbol),
            return_exceptions=True,
        )
        if not isinstance(htf_1h_res, Exception): htf_1h = htf_1h_res or []
        if not isinstance(htf_4h_res, Exception): htf_4h = htf_4h_res or []
        if not isinstance(funding_res, Exception): funding_data = funding_res
    except Exception as e:
        logger.debug(f"auto_trader enrichment fetch failed for #{strategy.id}: {e}")

    try:
        from app.services.ai_trade_read import generate_ai_trade_read
        result = await generate_ai_trade_read(
            symbol=strategy.symbol, tf=strategy.timeframe, candles=candles,
            indicators=indicators, toggles=toggles, tape=tape,
            wall_report=wall_report,
            htf_1h_candles=htf_1h, htf_4h_candles=htf_4h,
            funding_data=funding_data,
        )
    except Exception as e:
        logger.warning(f"auto_trader AI run failed for #{strategy.id}: {e}")
        strategy.last_evaluated_at = now
        db.commit()
        return False

    # Cost protection: if Claude ran out of credits, pause the strategy
    # for 24h instead of hammering the API every cadence_min and racking up
    # charges that all return 400. The deterministic fallback isn't suitable
    # for live entries — we'd be opening real (paper) positions on stale data.
    ai_error = result.get("ai_error")
    if ai_error == "insufficient_credits":
        strategy.status = "paused"
        strategy.paused_at = now
        strategy.paused_until = now + timedelta(hours=24)
        strategy.last_evaluated_at = now
        db.commit()
        logger.warning(
            f"auto_trader strategy #{strategy.id} AUTO-PAUSED for 24h — "
            f"Anthropic credit balance exhausted. Top up credits then resume."
        )
        return False
    if ai_error in ("rate_limit", "auth"):
        strategy.last_evaluated_at = now
        db.commit()
        return False

    plan_text = result.get("summary") or ""
    # Never trade off the deterministic fallback — only act on real Claude output.
    if result.get("fallback"):
        strategy.last_evaluated_at = now
        db.commit()
        return False
    parsed = parse_ai_plan(plan_text)
    strategy.last_evaluated_at = now
    db.commit()

    if not parsed["side"] or not parsed["entry"] or not parsed["stop"] or not parsed["tp1"]:
        return False
    # Gate: AI must want a MARKET entry and reach the user's odds threshold
    if (parsed["order_type"] or "MARKET") != "MARKET":
        return False
    threshold = strategy.min_odds or 60
    side_odds = parsed["odds_up"] if parsed["side"] == "long" else parsed["odds_down"]
    if side_odds is None or side_odds < threshold:
        return False

    # Use the AI's stop/tp1/tp2 verbatim (chart-derived per the prompt)
    last_price = candles[-1]["close"]
    # Refuse if the AI's "entry" is wildly off from the live price (>1.5%)
    if abs(parsed["entry"] - last_price) / last_price > 0.015:
        # Use live price as entry but keep the AI's stop/TPs
        parsed["entry"] = last_price

    t = await _open_paper_trade(
        strategy,
        side=parsed["side"], entry=parsed["entry"],
        stop=parsed["stop"], tp1=parsed["tp1"], tp2=parsed["tp2"],
        source="ai", plan_text=plan_text, db=db,
    )
    return t is not None


async def _maybe_open_rules(strategy, db) -> bool:
    """Rule mode: evaluate the compiled rule against the latest closed bar."""
    now = datetime.utcnow()
    last = strategy.last_evaluated_at
    if last and (now - last).total_seconds() < _RULE_TICK_FLOOR_S:
        return False
    rules = json.loads(strategy.rules_json or "{}")
    if not rules.get("entry"):
        return False
    candles = await _fetch_candles(strategy.symbol, strategy.timeframe, limit=200)
    if not candles or len(candles) < 30:
        return False

    walls_flat: List[dict] = []
    if rules.get("wall_confluence"):
        wall_report = await _fetch_walls_for(strategy.symbol)
        walls_flat  = _walls_flat_list(wall_report)
    tape = (json.loads(strategy.chart_state_json or "{}")).get("tape") or {}

    hit = evaluate_rules(rules, candles, walls=walls_flat, tape=tape)
    strategy.last_evaluated_at = now
    db.commit()
    if not hit:
        return False

    side  = hit["side"]
    entry = hit["entry_price"]
    stop, tp1, tp2 = derive_stop_tp_from_chart(
        side, entry, candles,
        walls=walls_flat if walls_flat else None,
        source=strategy.tp_sl_source if strategy.tp_sl_source in ("walls", "atr") else "walls",
    )
    plan_text = hit.get("source_note") or ""
    t = await _open_paper_trade(
        strategy,
        side=side, entry=entry, stop=stop, tp1=tp1, tp2=tp2,
        source="rules", plan_text=plan_text, db=db,
    )
    return t is not None


# ─── Public tick — called by alerts_engine each loop iteration ────────────────
async def tick_auto_strategies() -> None:
    """One pass over all active auto strategies. Cheap to call every 30s."""
    from app.database import SessionLocal
    from app.models import AutoTradeStrategy

    db = SessionLocal()
    try:
        active = (db.query(AutoTradeStrategy)
                    .filter(AutoTradeStrategy.status == "active")
                    .all())
    finally:
        db.close()

    if not active:
        return

    for strat in active:
        # Each strategy gets its own short transaction so a stuck strategy
        # never blocks the rest. We grab a row-level lock with skip_locked so
        # that during the alerts_engine advisory-lock handoff window (when two
        # gunicorn workers can briefly both be the executor), only one worker
        # processes a given strategy per tick — no double opens / double DMs.
        db = SessionLocal()
        try:
            row = (db.query(AutoTradeStrategy)
                     .filter(AutoTradeStrategy.id == strat.id)
                     .with_for_update(skip_locked=True)
                     .first())
            if not row:
                # Another worker is already ticking this strategy — skip.
                continue
            if row.status != "active":
                continue
            try:
                # 1) Resolve any open paper trades first (close on stop/TP).
                #    This may also bump consecutive_losses / paused_until below.
                await _resolve_open_trades_for(row, db)

                # 2) Pre-open gates (G/J/K) — bail before spending an AI call
                #    if cooldown / session / loss-cap forbids opening right now.
                allowed, reason = _pre_open_gates(row, datetime.utcnow())
                if not allowed:
                    logger.debug(f"auto_trader strategy #{row.id} gated: {reason}")
                    db.commit()
                    continue

                # 3) Respect max_concurrent_trades (G). Default 1 = legacy behaviour.
                from app.models import AutoTradePaperTrade
                has_open = (db.query(AutoTradePaperTrade)
                              .filter(AutoTradePaperTrade.strategy_id == row.id,
                                      AutoTradePaperTrade.status == "open")
                              .count())
                cap = max(1, int(getattr(row, "max_concurrent_trades", 1) or 1))
                if has_open < cap:
                    if row.mode == "ai":
                        await _maybe_open_ai(row, db)
                    elif row.mode == "rules":
                        await _maybe_open_rules(row, db)
                # Ensure the strategy row lock is released even if the
                # _maybe_open_* helpers commit internally.
                try:
                    db.commit()
                except Exception:
                    db.rollback()
            except Exception as e:
                logger.warning(f"auto_trader strategy #{row.id} tick failed: {e}")
                try:
                    db.rollback()
                except Exception:
                    pass
        finally:
            db.close()
