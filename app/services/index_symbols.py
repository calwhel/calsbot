"""
Index CFD symbol normalization — portal, cTrader (FP Markets), yfinance.

Canonical symbols match cTrader contract names (NAS100, SPX500, US30, …).
Legacy aliases (NDX, SPX, US100, …) are accepted everywhere and resolved
to the canonical form before broker or data lookups.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

# canonical → (broker_name, yfinance_ticker, display_name, pip_size)
_INDEX_META: Dict[str, Tuple[str, str, str, float]] = {
    "NAS100": ("US100", "^NDX",   "Nasdaq 100",           0.25),
    "SPX500": ("US500", "^GSPC",  "S&P 500",              0.25),
    "US30":   ("US30",  "^DJI",   "Dow Jones Industrial", 1.0),
    "GER40":  ("GER40", "^GDAXI", "DAX (Germany)",        1.0),
    "UK100":  ("UK100", "^FTSE",  "FTSE 100 (UK)",        1.0),
    "VIX":    ("VIX",   "^VIX",   "CBOE Volatility Index", 0.05),
}

# Alias → canonical
_INDEX_ALIASES: Dict[str, str] = {
  # Nasdaq
    "NDX": "NAS100", "US100": "NAS100", "NASDAQ": "NAS100", "NASDAQ100": "NAS100",
    "NAS": "NAS100",
  # S&P
    "SPX": "SPX500", "US500": "SPX500", "SP500": "SPX500",
  # Dow
    "DJI": "US30", "DOW": "US30",
  # DAX
    "DAX": "GER40", "DE40": "GER40",
  # FTSE
    "FTSE": "UK100",
}

CTRADER_INDEX_SYMBOLS = frozenset(_INDEX_META.keys())


def normalize_index_symbol(symbol: str) -> str:
    """Resolve any index alias to the canonical cTrader-style symbol."""
    s = (symbol or "").upper().strip()
    return _INDEX_ALIASES.get(s, s)


def is_index_symbol(symbol: str) -> bool:
    return normalize_index_symbol(symbol) in _INDEX_META


def ctrader_broker_symbol(symbol: str) -> str:
    """FP Markets cTrader contract name for order routing / trendbars."""
    canon = normalize_index_symbol(symbol)
    meta = _INDEX_META.get(canon)
    if meta:
        return meta[0]
    return symbol.upper()


def yf_ticker_for_index(symbol: str) -> Optional[str]:
    canon = normalize_index_symbol(symbol)
    meta = _INDEX_META.get(canon)
    return meta[1] if meta else None


def index_display_name(symbol: str) -> str:
    canon = normalize_index_symbol(symbol)
    meta = _INDEX_META.get(canon)
    return meta[2] if meta else canon


def index_pip_size(symbol: str) -> float:
    canon = normalize_index_symbol(symbol)
    meta = _INDEX_META.get(canon)
    return meta[3] if meta else 1.0


def catalog_entries() -> list:
    """Rows for asset_classes index catalog: (symbol, yf_ticker, name)."""
    return [(sym, meta[1], meta[2]) for sym, meta in _INDEX_META.items()]
