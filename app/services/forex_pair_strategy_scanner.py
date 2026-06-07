"""
Forex Pair Strategy Discovery Scanner — EURUSD, GBPUSD, USDJPY, etc.

Reuses the tradfi discovery engine from gold_strategy_scanner with
major-pair pip risk variants and cTrader-first candle fetching.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from app.services.gold_strategy_scanner import run_tradfi_discovery
from app.services.tradfi_prices import fetch_forex_scan_candles

DEFAULT_SYMBOL = "EURUSD"

# Day-trader timeframes — 1m needs a short lookback (upstream cap ~7d).
FOREX_TIMEFRAMES = ["1m", "5m", "15m", "1h"]
FOREX_TF_MAX_DAYS = {"1m": 7, "5m": 30, "15m": 90, "1h": 180}

# Major FX pairs: pip-based SL/TP (1 pip = 0.0001 on EUR/GBP, 0.01 on JPY).
# Scalps: 8–15 pip stops, 16–30 pip targets. Swings: 20–35 pip stops.
FOREX_RISK_VARIANTS = [
    (8,  16, "scalp", "breakeven"),
    (10, 20, "scalp", "breakeven"),
    (12, 24, "scalp", "trail"),
    (15, 30, "scalp", "fixed"),
    (20, 40, "swing", "breakeven"),
    (25, 50, "swing", "breakeven"),
]

SUPPORTED_SYMBOLS = (
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
)

_PAIR_LABELS = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD",
    "USDCAD": "USD/CAD",
    "USDCHF": "USD/CHF",
    "NZDUSD": "NZD/USD",
}


def _normalize_forex_symbol(symbol: str) -> str:
    sym = (symbol or DEFAULT_SYMBOL).upper().replace("/", "").replace("-", "")
    return sym if sym in SUPPORTED_SYMBOLS else DEFAULT_SYMBOL


async def run_forex_discovery(
    symbol: str = DEFAULT_SYMBOL,
    days: int = 90,
    direction_mode: str = "BOTH",
    progress_cb: Optional[Callable[[str], None]] = None,
    user_id: Optional[int] = None,
) -> Dict:
    sym = _normalize_forex_symbol(symbol)
    label = f"{_PAIR_LABELS.get(sym, sym)} ({sym})"
    return await run_tradfi_discovery(
        symbol=sym,
        asset_class="forex",
        days=days,
        direction_mode=direction_mode,
        progress_cb=progress_cb,
        user_id=user_id,
        instrument_label=label,
        name_prefix=sym,
        risk_variants=FOREX_RISK_VARIANTS,
        fetch_candles_fn=fetch_forex_scan_candles,
        timeframes=FOREX_TIMEFRAMES,
        tf_max_days=FOREX_TF_MAX_DAYS,
        log_prefix="forex-scan",
        no_trades_error=(
            f"No strategy produced enough trades on {sym} to rank. "
            "Try a longer window or connect cTrader demo for broker-matched candles."
        ),
        fetch_error=(
            f"Could not fetch {sym} historical data from any source "
            "(cTrader, Yahoo, FMP). Connect cTrader demo in Settings for best results."
        ),
    )
