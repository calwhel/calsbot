"""
Startup feed health checks + per-scan signal metrics.

Printed once when the executor wins the advisory lock:
  === FEED STATUS ===
  === SIGNAL STATUS ===
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_FEED_STATUS: Dict[str, str] = {}
_SIGNAL_STATUS: Dict[str, Any] = {}
_PROBE_DONE = False
_EVAL_RING: Deque[Dict[str, Any]] = deque(maxlen=100)


def feed_status() -> Dict[str, str]:
    return dict(_FEED_STATUS)


def signal_status() -> Dict[str, Any]:
    return dict(_SIGNAL_STATUS)


def record_eval_metric(
    *,
    symbol: str,
    strategy_id: Optional[int],
    setup_detected: bool,
    signal_generated: bool,
    block_reason: Optional[str] = None,
) -> None:
    """Ring buffer for last 100 strategy evaluations (blocker frequency rollup)."""
    _EVAL_RING.append({
        "symbol": symbol.upper() if symbol else "",
        "strategy_id": strategy_id,
        "setup_detected": setup_detected,
        "signal_generated": signal_generated,
        "block_reason": block_reason or (
            "signal_generated" if signal_generated else (
                "setup_detected" if setup_detected else "passed_scan"
            )
        ),
    })


def summarize_recent_blockers(limit: int = 100) -> Dict[str, Any]:
    """Aggregate block_reason counts from the last N evaluations."""
    rows = list(_EVAL_RING)[-limit:]
    counts: Counter = Counter()
    for r in rows:
        reason = r.get("block_reason") or "unknown"
        counts[str(reason)] += 1
    top = counts.most_common(1)
    return {
        "evaluations": len(rows),
        "blocker_counts": dict(counts),
        "top_blocker": top[0][0] if top else None,
        "top_blocker_count": top[0][1] if top else 0,
        "recent": rows[-10:],
    }


def log_blocker_rollup(limit: int = 100) -> None:
    """Log blocker frequency for the last N strategy evaluations."""
    summary = summarize_recent_blockers(limit)
    if not summary["evaluations"]:
        return
    counts = summary["blocker_counts"]
    ranked = ", ".join(
        f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])
    )
    logger.info(
        "[eval-blocker-rollup] last %d evals — %s | top=%s (%d)",
        summary["evaluations"],
        ranked,
        summary["top_blocker"],
        summary["top_blocker_count"],
    )
    for row in summary["recent"]:
        logger.info(
            "[eval-blocker-row] %s",
            json.dumps(row, separators=(",", ":"), default=str),
        )


def log_scan_metric(
    *,
    symbol: str,
    timeframe: str,
    candles_loaded: int,
    strategy_evaluated: bool,
    setup_detected: bool,
    signal_generated: bool,
    signal_sent: bool,
    strategy_id: Optional[int] = None,
    block_reason: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles_loaded": candles_loaded,
        "strategy_evaluated": strategy_evaluated,
        "setup_detected": setup_detected,
        "signal_generated": signal_generated,
        "signal_sent": signal_sent,
    }
    if strategy_id is not None:
        payload["strategy_id"] = strategy_id
    if block_reason:
        payload["block_reason"] = block_reason
    if extra:
        payload.update(extra)
    logger.info("[scan-metric] %s", json.dumps(payload, separators=(",", ":")))
    if strategy_evaluated:
        record_eval_metric(
            symbol=symbol,
            strategy_id=strategy_id,
            setup_detected=setup_detected,
            signal_generated=signal_generated,
            block_reason=block_reason,
        )


async def _probe_yahoo() -> bool:
    try:
        from app.services.tradfi_prices import _fetch_yahoo_chart_klines
        rows = await _fetch_yahoo_chart_klines("EURUSD=X", "15m", 5, http_timeout_s=8.0)
        return len(rows) >= 3
    except Exception:
        return False


async def _probe_binance() -> bool:
    """Binance futures klines probe — respects geo-block circuit breaker."""
    from app.services.binance_feed import binance_disabled
    if binance_disabled():
        return False
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                "https://fapi.binance.com/fapi/v1/klines",
                params={"symbol": "BTCUSDT", "interval": "15m", "limit": 5},
            )
        if resp.status_code != 200:
            return False
        data = resp.json()
        return isinstance(data, list) and len(data) >= 3
    except Exception:
        return False


async def _probe_kraken() -> bool:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.kraken.com/0/public/OHLC",
                params={"pair": "PAXGUSD", "interval": 15},
            )
        if resp.status_code != 200:
            return False
        body = resp.json()
        if not isinstance(body, dict):
            return False
        result = body.get("result")
        if not isinstance(result, dict):
            return False
        for key, val in result.items():
            if key != "last" and isinstance(val, list) and len(val) >= 3:
                return True
        return False
    except Exception:
        return False


async def _probe_coingecko() -> bool:
    try:
        from app.services.coingecko_safe import fetch_markets
        async with httpx.AsyncClient() as client:
            coins = await fetch_markets(client, params={"per_page": 5})
        return len(coins) >= 1
    except Exception:
        return False


async def _probe_ctrader() -> tuple[bool, str]:
    from app.services.ctrader_client import (
        CTRADER_CLIENT_ID,
        CTRADER_CLIENT_SECRET,
        audit_ctrader_credentials,
        proactive_refresh_linked_users,
    )
    missing = []
    if not CTRADER_CLIENT_ID:
        missing.append("CLIENT_ID")
    if not CTRADER_CLIENT_SECRET:
        missing.append("CLIENT_SECRET")
    if missing:
        return False, f"missing app creds: {', '.join(missing)}"

    refresh_results = await proactive_refresh_linked_users()
    linked = refresh_results.get("linked_users", 0)
    refreshed = refresh_results.get("refreshed", 0)
    denied = refresh_results.get("denied", 0)

    if linked == 0:
        return False, "no linked users in DB"

    # Audit first linked user
    try:
        from app.database import SessionLocal
        from app.models import UserPreference
        db = SessionLocal()
        try:
            prefs = (
                db.query(UserPreference)
                .filter(
                    UserPreference.ctrader_refresh_token.isnot(None),
                    UserPreference.ctrader_account_id.isnot(None),
                )
                .first()
            )
            if prefs:
                audit = audit_ctrader_credentials(prefs.user_id, prefs)
                if not audit.get("ok"):
                    return False, audit.get("reason", "audit failed")
        finally:
            db.close()
    except Exception as exc:
        return False, f"audit error: {type(exc).__name__}"

    try:
        from app.services.ctrader_price_feed import is_live, broker_session_ready
        if is_live() and broker_session_ready("EURUSD"):
            return True, f"stream live (refreshed={refreshed}/{linked})"
    except Exception:
        pass

    if denied:
        return False, f"token refresh denied for {denied} user(s) — re-link cTrader"
    if refreshed:
        return True, f"tokens refreshed ({refreshed}/{linked})"
    return False, f"linked={linked} but feed not live yet"


async def _load_signal_status() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "symbols_loaded": 0,
        "candles_loaded": 0,
        "strategies_loaded": 0,
        "crypto_strategies": 0,
        "forex_strategies": 0,
    }
    try:
        from app.database import SessionLocal
        from app.strategy_models import UserStrategy
        from app.services.asset_classes import normalize_asset_class

        db = SessionLocal()
        try:
            rows = (
                db.query(UserStrategy)
                .filter(UserStrategy.status.in_(["active", "paper"]))
                .all()
            )
            out["strategies_loaded"] = len(rows)
            symbols: set = set()
            for s in rows:
                ac = normalize_asset_class(
                    getattr(s, "asset_class", None)
                    or (s.config or {}).get("asset_class")
                    or "crypto"
                )
                if ac == "crypto":
                    out["crypto_strategies"] += 1
                elif ac in ("forex", "index", "stock"):
                    out["forex_strategies"] += 1
                for sym in (s.config or {}).get("universe", {}).get("symbols") or []:
                    if isinstance(sym, str) and sym.strip():
                        symbols.add(sym.upper())
            out["symbols_loaded"] = len(symbols)
        finally:
            db.close()
    except Exception as exc:
        logger.debug("[feed-diag] signal status error: %s", exc)
    return out


async def run_startup_diagnostics() -> Dict[str, Any]:
    """Probe all feeds and print status banner. Safe to call multiple times."""
    global _PROBE_DONE, _FEED_STATUS, _SIGNAL_STATUS

    ct_ok, ct_detail = await _probe_ctrader()
    yahoo_ok = await _probe_yahoo()
    binance_ok = await _probe_binance()
    kraken_ok = await _probe_kraken()
    cg_ok = await _probe_coingecko()

    _FEED_STATUS = {
        "CTrader": "PASS" if ct_ok else "FAIL",
        "Yahoo": "PASS" if yahoo_ok else "FAIL",
        "Binance": "PASS" if binance_ok else "FAIL",
        "Kraken": "PASS" if kraken_ok else "FAIL",
        "CoinGecko": "PASS" if cg_ok else "FAIL",
    }
    _SIGNAL_STATUS = await _load_signal_status()

    lines = [
        "=== FEED STATUS ===",
        f"CTrader: {_FEED_STATUS['CTrader']}" + (f" ({ct_detail})" if not ct_ok else f" ({ct_detail})"),
        f"Yahoo: {_FEED_STATUS['Yahoo']}",
        f"Binance: {_FEED_STATUS['Binance']}",
        f"Kraken: {_FEED_STATUS['Kraken']}",
        f"CoinGecko: {_FEED_STATUS['CoinGecko']}",
        "=== SIGNAL STATUS ===",
        f"Symbols Loaded: {_SIGNAL_STATUS.get('symbols_loaded', 0)}",
        f"Candles Loaded: (probed per scan — see [scan-metric] logs)",
        f"Strategies Loaded: {_SIGNAL_STATUS.get('strategies_loaded', 0)} "
        f"(crypto={_SIGNAL_STATUS.get('crypto_strategies', 0)}, "
        f"forex/index={_SIGNAL_STATUS.get('forex_strategies', 0)})",
    ]
    banner = "\n".join(lines)
    logger.info("\n%s", banner)
    _PROBE_DONE = True
    return {"feeds": _FEED_STATUS, "signals": _SIGNAL_STATUS, "ctrader_detail": ct_detail}


GATE_BLOCK_LOCATIONS: Dict[str, str] = {
    "blk_no_price_data": "strategy_executor._fetch_price_and_ta / tradfi_prices.get_klines",
    "blk_ta_conditions": "strategy_ta.evaluate_strategy_conditions",
    "blk_time_filter": "strategy_executor._check_time_filter",
    "blk_entry_price_stale": "tradfi_prices.confirm_entry_price",
    "blk_entry_drift": "strategy_executor.evaluate_and_fire (cTrader drift guard)",
    "blk_max_open": "strategy_executor.evaluate_and_fire (max_open_positions)",
    "blk_daily_cap": "strategy_executor.evaluate_and_fire (max_trades_per_day)",
    "blk_empty_universe": "strategy_executor._get_eligible_symbols",
    "blk_all_in_cooldown": "strategy_executor.evaluate_and_fire (symbol cooldown)",
    "blk_not_entitled": "strategy_executor._portal_trade_entitled",
}


def format_final_report(
    *,
    scans_running: int,
    symbols_scanned: int,
    candles_loaded: int,
    setups_detected: int,
    signals_blocked: Dict[str, int],
    block_locations: List[str],
) -> str:
    blocked = ", ".join(f"{k}={v}" for k, v in sorted(signals_blocked.items(), key=lambda x: -x[1]))
    locs = "\n  ".join(block_locations) if block_locations else "(none)"
    return (
        "=== SIGNAL PIPELINE REPORT ===\n"
        f"Scans running: {scans_running}\n"
        f"Symbols scanned: {symbols_scanned}\n"
        f"Candles loaded (last cycle aggregate): {candles_loaded}\n"
        f"Setups detected: {setups_detected}\n"
        f"Signals blocked: {blocked or 'none'}\n"
        f"Block locations:\n  {locs}"
    )
