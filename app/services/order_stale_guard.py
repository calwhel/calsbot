"""Abort live cTrader orders when the signal price or age is no longer valid."""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MAX_AGE_S = float(os.environ.get("CTRADER_MAX_SIGNAL_AGE_S", "30"))
_DEFAULT_MAX_SLIPPAGE_PIPS = float(os.environ.get("CTRADER_MAX_SLIPPAGE_PIPS", "15"))
_IMPLAUSIBLE_PIPS = float(os.environ.get("CTRADER_IMPLAUSIBLE_DRIFT_PIPS", "50"))
_IMPLAUSIBLE_AGE_S = float(os.environ.get("CTRADER_IMPLAUSIBLE_DRIFT_AGE_S", "5"))

# execution_id -> ("blocked"|"allowed", reason_or_empty)
_STALE_VERDICT: Dict[int, Tuple[str, str]] = {}


def _max_slippage_pips(symbol: str) -> float:
    sym = (symbol or "").upper()
    env_key = f"CTRADER_MAX_SLIPPAGE_PIPS_{sym}"
    raw = os.environ.get(env_key)
    if raw is not None and str(raw).strip():
        try:
            return float(raw)
        except ValueError:
            pass
    if sym == "XAUUSD":
        return float(os.environ.get("CTRADER_MAX_SLIPPAGE_PIPS_XAUUSD", "15"))
    return _DEFAULT_MAX_SLIPPAGE_PIPS


def _pip_size(symbol: str) -> float:
    try:
        from app.services.forex_engine import pip_size

        ps = pip_size(symbol)
        if ps and ps > 0:
            return float(ps)
    except Exception:
        pass
    from app.services.ctrader_client import _PIP_SIZES

    return float(_PIP_SIZES.get((symbol or "").upper(), 0.0001))


def _signal_src_family(
    *,
    price_source: str,
    kline_source: Optional[str],
    live_source: Optional[str],
) -> str:
    ps = (price_source or "").lower()
    if "kline" in ps:
        return f"kline:{(kline_source or 'unknown').lower()}"
    return f"spot:{(live_source or 'ctrader').lower()}"


def _directional_spot(mid: float, bid: Optional[float], ask: Optional[float], direction: str) -> float:
    d = (direction or "").upper()
    if d == "LONG" and ask and ask > 0:
        return float(ask)
    if d == "SHORT" and bid and bid > 0:
        return float(bid)
    return float(mid)


def _spot_price_now(symbol: str, direction: str, preferred: str) -> Tuple[Optional[float], str]:
    sym = (symbol or "").upper()
    pref = (preferred or "ctrader").lower()

    try:
        from app.services.spot_price_store import get_tick

        row = get_tick(sym, max_age_s=15.0)
        if row:
            src = (row.get("source") or "").lower()
            mid = float(row.get("mid") or 0)
            if mid > 0 and (src == pref or pref == "ctrader" and src in ("ctrader", "store")):
                px = _directional_spot(
                    mid,
                    row.get("bid"),
                    row.get("ask"),
                    direction,
                )
                return px, src or pref
    except Exception:
        pass

    if pref in ("ctrader", "store", ""):
        try:
            from app.services.ctrader_price_feed import get_bid_ask, get_price

            tick = get_bid_ask(sym)
            if tick:
                bid, ask = tick
                mid = round((bid + ask) / 2.0, 6)
                px = _directional_spot(mid, bid, ask, direction)
                if px > 0:
                    return px, "ctrader"
            px = get_price(sym)
            if px and px > 0:
                return float(px), "ctrader"
        except Exception:
            pass

    return None, pref


def _kline_close_now(symbol: str, kline_source: str) -> Tuple[Optional[float], str]:
    sym = (symbol or "").upper()
    src = (kline_source or "unknown").lower()

    if src in ("ctrader", "ctrader-user"):
        try:
            from app.services import ctrader_price_feed as ctf

            cache = getattr(ctf, "_kline_cache", None) or {}
            for key, (rows, _ts) in cache.items():
                if not key or key[0] != sym or not rows:
                    continue
                closed = rows[:-1] if len(rows) > 1 else rows
                if closed:
                    close = float(closed[-1][4])
                    if close > 0:
                        return close, src
        except Exception:
            pass

    try:
        from app.services.tradfi_prices import _KLINE_CACHE

        for key, (rows, _fetched_at) in _KLINE_CACHE.items():
            if not key or key[0] != sym or not rows:
                continue
            closed = rows[:-1] if len(rows) > 1 else rows
            if closed:
                close = float(closed[-1][4])
                if close > 0:
                    return close, src
    except Exception:
        pass

    return None, src


def _current_price_same_source(
    symbol: str,
    direction: str,
    sig_family: str,
) -> Tuple[Optional[float], str]:
    if sig_family.startswith("kline:"):
        src = sig_family.split(":", 1)[1]
        return _kline_close_now(symbol, src)
    if sig_family.startswith("spot:"):
        src = sig_family.split(":", 1)[1]
        return _spot_price_now(symbol, direction, src)
    return None, "unknown"


def get_stale_verdict(execution_id: int) -> Optional[str]:
    v = _STALE_VERDICT.get(int(execution_id))
    return v[0] if v else None


def set_stale_verdict(execution_id: int, verdict: str, reason: str = "") -> None:
    _STALE_VERDICT[int(execution_id)] = (verdict, reason)


def check_signal_stale(
    *,
    symbol: str,
    direction: str,
    signal_price: float,
    signal_generated_at: Optional[float] = None,
    signal_mono: Optional[float] = None,
    max_age_s: Optional[float] = None,
    price_source: str = "unknown",
    kline_source: Optional[str] = None,
    live_source: Optional[str] = None,
    execution_id: Optional[int] = None,
) -> Optional[Tuple[str, float, float, str, str]]:
    """
    Return (reason, age_s, slip_pips, sig_src, now_src) when the live order
    should be aborted. Same-source comparison only — cross-source drift never blocks.
    """
    if execution_id is not None:
        cached = get_stale_verdict(execution_id)
        if cached == "blocked":
            prev = _STALE_VERDICT.get(int(execution_id), ("blocked", ""))[1]
            return (prev or "signal stale (cached verdict)", 0.0, 0.0, "cached", "cached")
        if cached == "allowed":
            return None

    max_age = float(max_age_s if max_age_s is not None else _DEFAULT_MAX_AGE_S)
    now_wall = time.time()
    if signal_generated_at is not None and signal_generated_at > 0:
        age_s = max(0.0, now_wall - float(signal_generated_at))
    elif signal_mono is not None:
        age_s = max(0.0, time.monotonic() - float(signal_mono))
    else:
        age_s = 0.0

    if age_s > max_age:
        reason = f"signal stale (age {age_s:.1f}s > {max_age:.0f}s)"
        if execution_id is not None:
            set_stale_verdict(execution_id, "blocked", reason)
        return (reason, age_s, 0.0, "age", "age")

    if not signal_price or signal_price <= 0:
        if execution_id is not None:
            set_stale_verdict(execution_id, "allowed")
        return None

    sig_family = _signal_src_family(
        price_source=price_source,
        kline_source=kline_source,
        live_source=live_source,
    )
    sig_src = sig_family

    now_px, now_src_raw = _current_price_same_source(symbol, direction, sig_family)
    now_family = (
        f"spot:{now_src_raw}"
        if sig_family.startswith("spot:")
        else f"kline:{now_src_raw}"
    )
    now_src = now_family

    if now_px is None or now_px <= 0:
        logger.info(
            "[stale-guard] exec=%s %s no %s price now — skip drift check (allow)",
            execution_id,
            symbol,
            sig_family,
        )
        if execution_id is not None:
            set_stale_verdict(execution_id, "allowed")
        return None

    if now_family != sig_family:
        logger.warning(
            "[stale-guard] exec=%s %s source mismatch sig_src=%s now_src=%s — "
            "invalid comparison, not blocking",
            execution_id,
            symbol,
            sig_src,
            now_src,
        )
        if execution_id is not None:
            set_stale_verdict(execution_id, "allowed")
        return None

    pip = _pip_size(symbol)
    slip_pips = abs(now_px - float(signal_price)) / max(pip, 1e-12)
    max_slip = _max_slippage_pips(symbol)

    if slip_pips > _IMPLAUSIBLE_PIPS and age_s < _IMPLAUSIBLE_AGE_S:
        logger.warning(
            "[stale-guard] implausible drift — source mismatch suspected "
            "exec=%s %s moved %.1f pips in %.1fs (sig_src=%s now_src=%s sig=%.5f now=%.5f)",
            execution_id,
            symbol,
            slip_pips,
            age_s,
            sig_src,
            now_src,
            signal_price,
            now_px,
        )
        if execution_id is not None:
            set_stale_verdict(execution_id, "allowed")
        return None

    if slip_pips > max_slip:
        reason = (
            f"price moved {slip_pips:.1f} pips in {age_s:.1f}s "
            f"(max {max_slip:.0f}) (sig_src={sig_src} now_src={now_src})"
        )
        if execution_id is not None:
            set_stale_verdict(execution_id, "blocked", reason)
        return (reason, age_s, slip_pips, sig_src, now_src)

    if execution_id is not None:
        set_stale_verdict(execution_id, "allowed")
    return None
