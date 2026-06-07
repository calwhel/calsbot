"""
Live structure scanner — indices + forex ICT signals with fast/streaming modes.

Fast mode: NAS100, SPX500, EURUSD, GBPUSD, XAUUSD on 5m/15m (~10 jobs).
Full mode: all instruments on 1m/5m/15m.

Streaming: POST start → poll progress; signals appear as each pair/TF completes.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_FOREX_SCANNER_CACHE: dict = {}
_FOREX_SCANNER_TTL = 60

_LIVE_SCAN_STATE: Dict[str, dict] = {}
_LIVE_SCAN_TASKS: Dict[str, asyncio.Task] = {}

_SCANNER_INSTRUMENTS = [
    ("NAS100", "index"), ("SPX500", "index"), ("US30", "index"),
    ("GER40", "index"), ("UK100", "index"),
    ("EURUSD", "forex"), ("GBPUSD", "forex"), ("USDJPY", "forex"), ("AUDUSD", "forex"),
    ("USDCAD", "forex"), ("USDCHF", "forex"), ("NZDUSD", "forex"), ("EURJPY", "forex"),
    ("GBPJPY", "forex"), ("XAUUSD", "forex"), ("XAGUSD", "forex"),
]

_SCANNER_FAST_INSTRUMENTS = [
    ("NAS100", "index"), ("SPX500", "index"),
    ("EURUSD", "forex"), ("GBPUSD", "forex"), ("XAUUSD", "forex"),
]

_SCANNER_TIMEFRAMES_INDEX = ["1m", "5m", "15m"]
_SCANNER_TIMEFRAMES_FOREX = ["1m", "5m", "15m"]
_SCANNER_FAST_TFS = ["5m", "15m"]

_SCANNER_SIGNALS = [
    ("market_structure", "bos_bullish", None),
    ("market_structure", "bos_bearish", None),
    ("market_structure", "choch_bullish", None),
    ("market_structure", "choch_bearish", None),
    ("fvg", "bullish", "just_formed"),
    ("fvg", "bearish", "just_formed"),
    ("fvg", "bullish", "retest"),
    ("fvg", "bearish", "retest"),
    ("ifvg", "bullish", "retest"),
    ("ifvg", "bearish", "retest"),
    ("order_block", "bullish", None),
    ("order_block", "bearish", None),
    ("liquidity_sweep", "bullish", None),
    ("liquidity_sweep", "bearish", None),
    ("breaker_block", "bullish", None),
    ("breaker_block", "bearish", None),
    ("fx_displacement", "bullish", None),
    ("fx_displacement", "bearish", None),
    ("mss", "bullish", None),
    ("mss", "bearish", None),
]

_SCANNER_FAST_SIGNALS = [
    ("market_structure", "bos_bullish", None),
    ("market_structure", "bos_bearish", None),
    ("market_structure", "choch_bullish", None),
    ("market_structure", "choch_bearish", None),
    ("fvg", "bullish", "just_formed"),
    ("fvg", "bearish", "just_formed"),
    ("mss", "bullish", None),
    ("mss", "bearish", None),
    ("order_block", "bullish", None),
    ("order_block", "bearish", None),
]

_MS_LABELS = {
    "bos_bullish":   ("BOS", "LONG",  "Bullish Break of Structure"),
    "bos_bearish":   ("BOS", "SHORT", "Bearish Break of Structure"),
    "choch_bullish": ("CHoCH", "LONG", "Bullish Change of Character"),
    "choch_bearish": ("CHoCH", "SHORT", "Bearish Change of Character"),
}
_BIAS_DIR = {"bullish": "LONG", "bearish": "SHORT"}

_SESSION_HOURS = {
    "asian": (0, 8), "london": (7, 16), "new_york": (13, 22), "overlap": (13, 16),
}
_SESSION_LABELS = {
    "asian": "Asian", "london": "London", "new_york": "New York", "overlap": "London-NY overlap",
}


def scanner_pairs() -> List[str]:
    return [p for p, _ in _SCANNER_INSTRUMENTS]


def _active_sessions_utc() -> list:
    hour = datetime.utcnow().hour
    return [sid for sid, (a, b) in _SESSION_HOURS.items() if a <= hour < b]


def session_label() -> str:
    active = _active_sessions_utc()
    return "Off-hours" if not active else " · ".join(_SESSION_LABELS.get(s, s) for s in active)


def _signal_meta(sig_type: str, sub: str, mode: Optional[str]):
    if sig_type == "market_structure":
        return _MS_LABELS.get(sub, ("MS", "LONG", sub))
    d = _BIAS_DIR.get(sub, "LONG")
    if sig_type == "fvg":
        fresh = (mode or "just_formed") == "just_formed"
        return (
            "FVG", d,
            f"{'Bullish' if sub == 'bullish' else 'Bearish'} Fair Value Gap "
            f"({'just formed' if fresh else 'retest / in gap'})",
        )
    if sig_type == "ifvg":
        return ("IFVG", d, f"{'Bullish' if sub == 'bullish' else 'Bearish'} Inverse FVG retest")
    if sig_type == "order_block":
        return ("OB", d, f"{'Bullish' if sub == 'bullish' else 'Bearish'} Order Block touch")
    if sig_type == "liquidity_sweep":
        return ("LQS", d, f"{'Bullish' if sub == 'bullish' else 'Bearish'} Liquidity sweep")
    if sig_type == "breaker_block":
        return ("BRK", d, f"{'Bullish' if sub == 'bullish' else 'Bearish'} Breaker block retest")
    if sig_type == "fx_displacement":
        return ("DISP", d, f"{'Bullish' if sub == 'bullish' else 'Bearish'} Displacement candle")
    if sig_type == "mss":
        return ("MSS", d, f"{'Bullish' if sub == 'bullish' else 'Bearish'} Market structure shift")
    return ("Signal", d, sub)


def _annotate_confluence(results: list) -> list:
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for entry in results:
        key = (entry["pair"], entry["timeframe"], entry["direction"])
        groups[key].append(entry)
    for entry in results:
        stack = groups[(entry["pair"], entry["timeframe"], entry["direction"])]
        entry["confluence"] = len(stack)
        entry["confluence_signals"] = sorted({s["signal"] for s in stack})
    return results


def _sort_results(results: list) -> list:
    _tf_rank = {"1m": 0, "5m": 1, "15m": 2, "1h": 3}
    _sig_rank = {"CHoCH": 0, "MSS": 1, "BOS": 2, "FVG": 3, "IFVG": 4, "LQS": 5, "BRK": 6, "DISP": 7, "OB": 8}
    _index_pri = {"NAS100": 0, "SPX500": 1, "US30": 2, "GER40": 3, "UK100": 4}
    results.sort(key=lambda x: (
        -x.get("confluence", 1),
        _index_pri.get(x["pair"], 5),
        _tf_rank.get(x["timeframe"], 4),
        _sig_rank.get(x["signal"], 9),
        x["pair"],
    ))
    return results


def _work_items(mode: str) -> List[Tuple[str, str, str]]:
    instruments = _SCANNER_FAST_INSTRUMENTS if mode == "fast" else _SCANNER_INSTRUMENTS
    out: List[Tuple[str, str, str]] = []
    for pair, ac in instruments:
        tfs = _SCANNER_FAST_TFS if mode == "fast" else (
            _SCANNER_TIMEFRAMES_INDEX if ac == "index" else _SCANNER_TIMEFRAMES_FOREX
        )
        for tf in tfs:
            out.append((pair, tf, ac))
    return out


async def _scan_pair_tf(
    pair: str,
    tf: str,
    asset_class: str,
    signals: list,
    _http=None,
    *,
    now: float,
    sem: asyncio.Semaphore,
) -> list:
    import httpx as _httpx
    from app.services.strategy_ta import (
        eval_market_structure, eval_fvg, eval_order_block,
        eval_bt_klines_cond, eval_fx_displacement, eval_fx_breaker,
    )
    from app.services.tradfi_prices import get_price as _tradfi_price

    def _cache_key(st, sub, mode):
        return f"{pair}|{tf}|{st}|{sub}|{mode or ''}"

    all_keys = [_cache_key(st, sub, mode) for st, sub, mode in signals]
    fresh = [_FOREX_SCANNER_CACHE.get(k) for k in all_keys]
    if all(f and f[1] > now for f in fresh):
        return [f[0] for f in fresh if f and f[0]]

    entries = []
    eval_timeout = 8.0
    try:
        async with sem:
            price = await asyncio.wait_for(_tradfi_price(pair, asset_class), timeout=5.0)
            if not price:
                return []

            shared_cache = {"__asset_class__": asset_class}
            _client = _http or _httpx.AsyncClient(timeout=10)
            _own_client = _http is None
            try:
                for sig_type, sub, sig_mode in signals:
                    cache_key = _cache_key(sig_type, sub, sig_mode)
                    cached = _FOREX_SCANNER_CACHE.get(cache_key)
                    if cached and cached[1] > now:
                        if cached[0]:
                            entries.append(cached[0])
                        continue

                    entry = None
                    try:
                        if sig_type == "market_structure":
                            ok, msg = await asyncio.wait_for(
                                eval_market_structure(
                                    {"condition": sub, "timeframe": tf},
                                    pair, price, _client, shared_cache,
                                ), timeout=eval_timeout,
                            )
                        elif sig_type == "fvg":
                            cond = {"direction": sub, "timeframe": tf, "condition": sig_mode or "just_formed"}
                            ok, msg = await asyncio.wait_for(
                                eval_fvg(cond, pair, price, _client, shared_cache), timeout=eval_timeout,
                            )
                        elif sig_type == "ifvg":
                            ok, msg = await asyncio.wait_for(
                                eval_bt_klines_cond(
                                    {"type": "ifvg", "direction": sub, "timeframe": tf},
                                    pair, _client, shared_cache,
                                ), timeout=eval_timeout,
                            )
                        elif sig_type == "order_block":
                            ok, msg = await asyncio.wait_for(
                                eval_order_block(
                                    {"ob_type": sub, "direction": sub, "timeframe": tf},
                                    pair, price, _client, shared_cache,
                                ), timeout=eval_timeout,
                            )
                        elif sig_type == "liquidity_sweep":
                            ok, msg = await asyncio.wait_for(
                                eval_bt_klines_cond(
                                    {"type": "liquidity_sweep", "direction": sub, "timeframe": tf},
                                    pair, _client, shared_cache,
                                ), timeout=eval_timeout,
                            )
                        elif sig_type == "breaker_block":
                            ok, msg = await asyncio.wait_for(
                                eval_fx_breaker(
                                    {"direction": sub, "timeframe": tf},
                                    pair, price, _client, shared_cache,
                                ), timeout=eval_timeout,
                            )
                        elif sig_type == "fx_displacement":
                            ok, msg = await asyncio.wait_for(
                                eval_fx_displacement(
                                    {"direction": sub, "timeframe": tf},
                                    pair, _client, shared_cache,
                                ), timeout=eval_timeout,
                            )
                        elif sig_type == "mss":
                            ok, msg = await asyncio.wait_for(
                                eval_bt_klines_cond(
                                    {"type": "mss", "direction": sub, "timeframe": tf},
                                    pair, _client, shared_cache,
                                ), timeout=eval_timeout,
                            )
                        else:
                            continue

                        if ok:
                            label, direction, desc = _signal_meta(sig_type, sub, sig_mode)
                            entry = {
                                "pair": pair, "timeframe": tf, "signal": label,
                                "direction": direction, "desc": desc,
                                "price": round(price, 6), "detail": msg,
                                "sig_key": f"{sig_type}:{sub}" + (f":{sig_mode}" if sig_mode else ""),
                                "asset_class": asset_class,
                            }
                            entries.append(entry)
                    except Exception as exc:
                        logger.debug(f"Scanner {pair}/{tf}/{sig_type}: {exc}")

                    _FOREX_SCANNER_CACHE[cache_key] = (entry, now + _FOREX_SCANNER_TTL)
            finally:
                if _own_client:
                    await _client.aclose()
    except Exception as exc:
        logger.debug(f"Scanner pair+tf {pair}/{tf}: {exc}")

    return entries


async def run_forex_scanner(
    mode: str = "fast",
    timeout_secs: Optional[float] = None,
    on_batch: Optional[Callable[[str, str, list, list], None]] = None,
) -> Tuple[list, bool]:
    """Run scan; optional on_batch(pair, tf, batch, all_so_far) for streaming."""
    import httpx as _httpx

    mode = "full" if mode == "full" else "fast"
    signals = _SCANNER_SIGNALS if mode == "full" else _SCANNER_FAST_SIGNALS
    items = _work_items(mode)
    if timeout_secs is None:
        timeout_secs = 28.0 if mode == "fast" else 90.0

    now = time.monotonic()
    sem = asyncio.Semaphore(int(os.environ.get("FOREX_SCANNER_PARALLEL", "12")))
    all_results: list = []
    sess = session_label()
    partial = False

    async with _httpx.AsyncClient(timeout=10) as _scan_http:
        # First 2 jobs sequential → fastest time-to-first-signal for UX
        head, tail = items[:2], items[2:]

        async def _one(pair, tf, ac):
            batch = await _scan_pair_tf(pair, tf, ac, signals, _scan_http, now=now, sem=sem)
            for e in batch:
                e["session"] = sess
            return pair, tf, batch

        for pair, tf, ac in head:
            _, _, batch = await _one(pair, tf, ac)
            all_results.extend(batch)
            if on_batch:
                on_batch(pair, tf, batch, list(all_results))

        if tail:
            tasks = [asyncio.create_task(_one(pair, tf, ac)) for pair, tf, ac in tail]
            done, pending = await asyncio.wait(tasks, timeout=max(5.0, timeout_secs - (time.monotonic() - now)))
            partial = bool(pending)
            if pending:
                for t in pending:
                    t.cancel()
            for t in done:
                try:
                    pair, tf, batch = t.result()
                    all_results.extend(batch)
                    if on_batch:
                        on_batch(pair, tf, batch, list(all_results))
                except Exception:
                    pass

    all_results = _sort_results(_annotate_confluence(all_results))
    return all_results, partial


def live_scan_progress(uid: str) -> dict:
    uid = uid.strip()
    state = _LIVE_SCAN_STATE.get(uid)
    if not state:
        return {"status": "idle", "signals": [], "count": 0}
    out = dict(state)
    mode = out.get("mode", "fast")
    meta = build_scan_response(
        out.get("signals") or [],
        mode=mode,
        partial=out.get("partial", False),
    )
    for key in (
        "session", "market_open", "active_sessions", "confluence_hits",
        "pairs_scanned", "instruments_index", "instruments_forex",
        "timeframes_index", "timeframes_forex", "cache_ttl",
    ):
        out[key] = meta[key]
    if out.get("status") == "done":
        out["scanned_at"] = meta["scanned_at"]
    return out


def start_live_scan(uid: str, mode: str = "fast") -> dict:
    uid = uid.strip()
    key = uid
    existing = _LIVE_SCAN_STATE.get(key)
    if existing and existing.get("status") == "running":
        return {"ok": True, "started": False, "status": "running", **existing}

    items = _work_items("full" if mode == "full" else "fast")
    _LIVE_SCAN_STATE[key] = {
        "status": "running",
        "mode": mode,
        "done": 0,
        "total": len(items),
        "current": "Starting…",
        "signals": [],
        "count": 0,
        "partial": False,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "message": f"Scanning {items[0][0]} {items[0][1]}…" if items else "Starting…",
    }

    async def _worker():
        state = _LIVE_SCAN_STATE[key]

        def _on_batch(pair, tf, batch, all_so_far):
            state["current"] = f"{pair} {tf}"
            state["done"] = min(state.get("done", 0) + 1, state["total"])
            state["signals"] = _sort_results(_annotate_confluence(list(all_so_far)))
            state["count"] = len(state["signals"])
            state["message"] = (
                f"Found {state['count']} signal{'s' if state['count'] != 1 else ''} · "
                f"{pair} {tf} ({state['done']}/{state['total']})"
            )

        try:
            signals, partial = await run_forex_scanner(mode=mode, on_batch=_on_batch)
            state["signals"] = signals
            state["count"] = len(signals)
            state["partial"] = partial
            state["done"] = state["total"]
            state["status"] = "done"
            state["message"] = (
                f"Done — {len(signals)} signal{'s' if len(signals) != 1 else ''}"
                + (" (partial)" if partial else "")
            )
            state["scanned_at"] = datetime.utcnow().isoformat() + "Z"
        except Exception as exc:
            logger.exception(f"live scan failed for {uid}")
            state["status"] = "error"
            state["error"] = str(exc)[:200]
            state["message"] = state["error"]
        finally:
            _LIVE_SCAN_TASKS.pop(key, None)

    try:
        task = asyncio.get_running_loop().create_task(_worker())
    except RuntimeError:
        task = asyncio.get_event_loop().create_task(_worker())
    _LIVE_SCAN_TASKS[key] = task
    return {"ok": True, "started": True, "status": "running", "mode": mode, "total": len(items)}


def build_scan_response(signals: list, *, mode: str, partial: bool) -> dict:
    from app.services.asset_classes import is_market_open
    now_utc = datetime.utcnow()
    index_syms = {p for p, ac in _SCANNER_INSTRUMENTS if ac == "index"}
    instruments = _SCANNER_FAST_INSTRUMENTS if mode != "full" else _SCANNER_INSTRUMENTS
    return {
        "signals": signals,
        "count": len(signals),
        "mode": mode,
        "pairs_scanned": len(instruments),
        "instruments_index": len(index_syms),
        "instruments_forex": len(_SCANNER_INSTRUMENTS) - len(index_syms),
        "timeframes_index": _SCANNER_FAST_TFS if mode == "fast" else _SCANNER_TIMEFRAMES_INDEX,
        "timeframes_forex": _SCANNER_FAST_TFS if mode == "fast" else _SCANNER_TIMEFRAMES_FOREX,
        "scanned_at": now_utc.isoformat() + "Z",
        "cache_ttl": _FOREX_SCANNER_TTL,
        "partial": partial,
        "session": session_label(),
        "active_sessions": _active_sessions_utc(),
        "market_open": {
            "forex": is_market_open("forex", now_utc),
            "index": is_market_open("index", now_utc),
        },
        "confluence_hits": sum(1 for s in signals if s.get("confluence", 0) >= 2),
    }
