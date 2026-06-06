"""
Index Strategy Discovery Scanner — NASDAQ, S&P 500, etc.

Reuses the tradfi discovery engine from gold_strategy_scanner with
index-specific pip risk variants and cTrader-first candle fetching.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from app.services.gold_strategy_scanner import run_tradfi_discovery
from app.services.index_symbols import index_display_name, normalize_index_symbol
from app.services.tradfi_prices import fetch_index_scan_candles

DEFAULT_SYMBOL = "NAS100"

# Day-trader timeframes — 1m needs a short lookback (upstream cap ~7d).
INDEX_TIMEFRAMES = ["1m", "5m", "15m", "1h"]
INDEX_TF_MAX_DAYS = {"1m": 7, "5m": 30, "15m": 90, "1h": 180}

# Index CFDs quote in POINTS (1.0 = one index level on NAS100/SPX500).
# Scalps: 40–80 pt targets (user-requested min 40). Swings capped ~120 pt —
# prior 160–240 pt targets were unrealistic intraday moonshots on Nasdaq.
# Breakeven-heavy roster — at 1× leverage, SL→entry after half the stop distance
# in profit protects scalps on choppy Nasdaq days and lifts effective win rate.
INDEX_RISK_VARIANTS = [
    (15, 40,  "scalp", "breakeven"),
    (20, 40,  "scalp", "breakeven"),
    (20, 50,  "scalp", "breakeven"),
    (25, 60,  "scalp", "breakeven"),
    (15, 40,  "scalp", "trail"),
    (30, 70,  "scalp", "breakeven"),
    (30, 80,  "scalp", "breakeven"),
    (40, 100, "swing", "breakeven"),
    (50, 120, "swing", "breakeven"),
]

SUPPORTED_SYMBOLS = ("NAS100", "SPX500", "US30", "GER40", "UK100")


async def run_index_discovery(
    symbol: str = DEFAULT_SYMBOL,
    days: int = 90,
    direction_mode: str = "BOTH",
    progress_cb: Optional[Callable[[str], None]] = None,
    user_id: Optional[int] = None,
) -> Dict:
    sym = normalize_index_symbol(symbol or DEFAULT_SYMBOL)
    if sym not in SUPPORTED_SYMBOLS:
        sym = DEFAULT_SYMBOL
    label = f"{index_display_name(sym)} ({sym})"
    return await run_tradfi_discovery(
        symbol=sym,
        asset_class="index",
        days=days,
        direction_mode=direction_mode,
        progress_cb=progress_cb,
        user_id=user_id,
        instrument_label=label,
        name_prefix=sym,
        risk_variants=INDEX_RISK_VARIANTS,
        fetch_candles_fn=fetch_index_scan_candles,
        timeframes=INDEX_TIMEFRAMES,
        tf_max_days=INDEX_TF_MAX_DAYS,
        log_prefix="index-scan",
        no_trades_error=(
            f"No strategy produced enough trades on {sym} to rank. "
            "Try a longer window or connect cTrader demo for broker-matched candles."
        ),
        fetch_error=(
            f"Could not fetch {sym} historical data from any source "
            "(cTrader, Yahoo, FMP). Connect cTrader demo in Settings for best results."
        ),
    )
