"""
Index CFD symbol normalization — portal, cTrader (FP Markets), yfinance.

Canonical symbols match cTrader contract names (NAS100, SPX500, US30, …).
Legacy aliases (NDX, SPX, US100, …) are accepted everywhere and resolved
to the canonical form before broker or data lookups.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

# canonical → (broker_name, yfinance_ticker, display_name, point_size)
# FP Markets indices are quoted in index POINTS (1.0 = one full index level),
# not forex-style pips. NAS100 21000 → 21001 is 1 point.
_INDEX_META: Dict[str, Tuple[str, str, str, float]] = {
    "NAS100": ("US100", "^NDX",   "Nasdaq 100",           1.0),
    "SPX500": ("US500", "^GSPC",  "S&P 500",              1.0),
    "US30":   ("US30",  "^DJI",   "Dow Jones Industrial", 1.0),
    "GER40":  ("GER40", "^GDAXI", "DAX (Germany)",        1.0),
    "UK100":  ("UK100", "^FTSE",  "FTSE 100 (UK)",        1.0),
    "VIX":    ("VIX",   "^VIX",   "CBOE Volatility Index", 0.01),
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


def index_point_size(symbol: str) -> float:
    """Price move per one index point (FP Markets: 1.0 for US indices)."""
    canon = normalize_index_symbol(symbol)
    meta = _INDEX_META.get(canon)
    return meta[3] if meta else 1.0


def index_pip_size(symbol: str) -> float:
    """Alias — internal math uses the same field; UI should say 'points' not 'pips'."""
    return index_point_size(symbol)


def price_unit_label(symbol: str, asset_class: str = "index") -> str:
    """Human label for TP/SL distance on this instrument."""
    if asset_class == "index" or is_index_symbol(symbol):
        return "points"
    if (symbol or "").upper() in ("XAUUSD", "XAGUSD"):
        return "pips"
    return "pips"


def catalog_entries() -> list:
    """Rows for asset_classes index catalog: (symbol, yf_ticker, name)."""
    return [(sym, meta[1], meta[2]) for sym, meta in _INDEX_META.items()]
