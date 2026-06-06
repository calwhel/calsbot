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

# NAS100 / SPX500 pip = 0.25 pt on FP Markets. Targets sized for typical
# intraday index ranges (~50–200 pts on Nasdaq).
INDEX_RISK_VARIANTS = [
    (20, 40,  "scalp", "breakeven"),
    (30, 60,  "scalp", "trail"),
    (40, 80,  "scalp", "fixed"),
    (80, 160, "swing", "breakeven"),
    (100, 200, "swing", "trail"),
    (120, 240, "swing", "fixed"),
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
