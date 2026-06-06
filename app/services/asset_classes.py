"""
Asset class catalog + market-hours gating for stocks / forex / indices.

The crypto path remains untouched — these helpers only fire when a strategy's
`asset_class` is set to `stock`, `forex`, or `index`. Non-crypto strategies are
paper-only (no live broker integration); the executor forces is_paper=True.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone, timedelta
from typing import Dict, List, Optional, Tuple

# ─── Constants ──────────────────────────────────────────────────────────────

ASSET_CLASS_CRYPTO = "crypto"
ASSET_CLASS_STOCK = "stock"
ASSET_CLASS_FOREX = "forex"
ASSET_CLASS_INDEX = "index"

ASSET_CLASSES = (ASSET_CLASS_CRYPTO, ASSET_CLASS_STOCK, ASSET_CLASS_FOREX, ASSET_CLASS_INDEX)

# Stocks have no live broker — always paper.
# Forex + indices go live via cTrader (FP Markets) when credentials are present.
PAPER_ONLY_CLASSES = {ASSET_CLASS_STOCK, ASSET_CLASS_FOREX, ASSET_CLASS_INDEX}
# cTrader-eligible classes (forex AND indices via FP Markets)
CTRADER_CLASSES = frozenset({ASSET_CLASS_FOREX, ASSET_CLASS_INDEX})


# ─── Symbol catalogs ────────────────────────────────────────────────────────
# Each entry: (display_symbol, yfinance_ticker, friendly_name)
# `display_symbol` is what shows in the UI and stored on the trade row.
# `yfinance_ticker` is what we send to the yfinance API.

@dataclass(frozen=True)
class SymbolMeta:
    symbol: str          # UI display, also stored on StrategyExecution
    yf_ticker: str       # yfinance ticker
    name: str            # Friendly name
    asset_class: str


# Top US stocks — what most retail traders watch first.
_TOP_US_STOCKS: List[Tuple[str, str, str]] = [
    ("AAPL",  "AAPL",  "Apple"),
    ("MSFT",  "MSFT",  "Microsoft"),
    ("NVDA",  "NVDA",  "NVIDIA"),
    ("TSLA",  "TSLA",  "Tesla"),
    ("AMZN",  "AMZN",  "Amazon"),
    ("GOOGL", "GOOGL", "Alphabet"),
    ("META",  "META",  "Meta Platforms"),
    ("AMD",   "AMD",   "Advanced Micro Devices"),
    ("NFLX",  "NFLX",  "Netflix"),
    ("AVGO",  "AVGO",  "Broadcom"),
    ("COIN",  "COIN",  "Coinbase"),
    ("MSTR",  "MSTR",  "MicroStrategy"),
    ("PLTR",  "PLTR",  "Palantir"),
    ("UBER",  "UBER",  "Uber"),
    ("SHOP",  "SHOP",  "Shopify"),
    ("SQ",    "SQ",    "Block"),
    ("PYPL",  "PYPL",  "PayPal"),
    ("BABA",  "BABA",  "Alibaba"),
    ("JPM",   "JPM",   "JPMorgan Chase"),
    ("GS",    "GS",    "Goldman Sachs"),
    ("BAC",   "BAC",   "Bank of America"),
    ("V",     "V",     "Visa"),
    ("MA",    "MA",    "Mastercard"),
    ("DIS",   "DIS",   "Disney"),
    ("BA",    "BA",    "Boeing"),
    ("XOM",   "XOM",   "ExxonMobil"),
    ("CVX",   "CVX",   "Chevron"),
    ("WMT",   "WMT",   "Walmart"),
    ("KO",    "KO",    "Coca-Cola"),
    ("PEP",   "PEP",   "PepsiCo"),
    ("MCD",   "MCD",   "McDonald's"),
    ("NKE",   "NKE",   "Nike"),
    ("LMT",   "LMT",   "Lockheed Martin"),
    ("CRM",   "CRM",   "Salesforce"),
    ("ORCL",  "ORCL",  "Oracle"),
    ("INTC",  "INTC",  "Intel"),
    ("MU",    "MU",    "Micron Technology"),
    ("ADBE",  "ADBE",  "Adobe"),
    ("LIN",   "LIN",   "Linde"),
    ("HD",    "HD",    "Home Depot"),
    ("COST",  "COST",  "Costco"),
    ("WFC",   "WFC",   "Wells Fargo"),
    ("T",     "T",     "AT&T"),
    ("VZ",    "VZ",    "Verizon"),
    ("CSCO",  "CSCO",  "Cisco"),
    ("IBM",   "IBM",   "IBM"),
    # Common retail ETFs (treated as stocks for our purposes)
    ("SPY",   "SPY",   "S&P 500 ETF"),
    ("QQQ",   "QQQ",   "Nasdaq 100 ETF"),
    ("DIA",   "DIA",   "Dow Jones ETF"),
    ("IWM",   "IWM",   "Russell 2000 ETF"),
    ("ARKK",  "ARKK",  "ARK Innovation ETF"),
]

# Forex majors + metals — yfinance uses `EURUSD=X` format; metals use GC=F / SI=F.
_FOREX_MAJORS: List[Tuple[str, str, str]] = [
    ("EURUSD", "EURUSD=X", "Euro / US Dollar"),
    ("GBPUSD", "GBPUSD=X", "British Pound / US Dollar"),
    ("USDJPY", "USDJPY=X", "US Dollar / Japanese Yen"),
    ("AUDUSD", "AUDUSD=X", "Australian / US Dollar"),
    ("USDCAD", "USDCAD=X", "US / Canadian Dollar"),
    ("USDCHF", "USDCHF=X", "US Dollar / Swiss Franc"),
    ("NZDUSD", "NZDUSD=X", "New Zealand / US Dollar"),
    ("EURGBP", "EURGBP=X", "Euro / British Pound"),
    ("EURJPY", "EURJPY=X", "Euro / Japanese Yen"),
    ("GBPJPY", "GBPJPY=X", "British Pound / Japanese Yen"),
    ("XAUUSD", "GC=F",     "Gold / US Dollar"),
    ("XAGUSD", "SI=F",     "Silver / US Dollar"),
    # Commodity futures (CME) — day-traded via cTrader / MT5 accounts
    ("CLUSD",  "CL=F",     "Crude Oil WTI / US Dollar"),
    ("NGUSD",  "NG=F",     "Natural Gas / US Dollar"),
    ("HGUSD",  "HG=F",     "Copper / US Dollar"),
]

# Major indices — canonical symbols match FP Markets cTrader (NAS100, SPX500, …).
try:
    from app.services.index_symbols import catalog_entries as _index_catalog_entries
    _INDICES: List[Tuple[str, str, str]] = _index_catalog_entries()
except Exception:
    _INDICES = [
        ("NAS100", "^NDX",   "Nasdaq 100"),
        ("SPX500", "^GSPC",  "S&P 500"),
        ("US30",   "^DJI",   "Dow Jones Industrial"),
        ("GER40",  "^GDAXI", "DAX (Germany)"),
        ("UK100",  "^FTSE",  "FTSE 100 (UK)"),
        ("VIX",    "^VIX",   "CBOE Volatility Index"),
    ]


def _index(rows: List[Tuple[str, str, str]], cls: str) -> Dict[str, SymbolMeta]:
    return {row[0]: SymbolMeta(symbol=row[0], yf_ticker=row[1], name=row[2], asset_class=cls)
            for row in rows}


_CATALOG: Dict[str, Dict[str, SymbolMeta]] = {
    ASSET_CLASS_STOCK: _index(_TOP_US_STOCKS, ASSET_CLASS_STOCK),
    ASSET_CLASS_FOREX: _index(_FOREX_MAJORS, ASSET_CLASS_FOREX),
    ASSET_CLASS_INDEX: _index(_INDICES, ASSET_CLASS_INDEX),
}


def list_symbols(asset_class: str) -> List[SymbolMeta]:
    """All known symbols for the given asset class (sorted by display order)."""
    return list(_CATALOG.get(asset_class, {}).values())


def get_symbol(asset_class: str, symbol: str) -> Optional[SymbolMeta]:
    cls = normalize_asset_class(asset_class)
    sym = symbol.upper()
    if cls == ASSET_CLASS_INDEX:
        try:
            from app.services.index_symbols import normalize_index_symbol
            sym = normalize_index_symbol(sym)
        except Exception:
            pass
    return _CATALOG.get(cls, {}).get(sym)


def yf_ticker(asset_class: str, symbol: str) -> Optional[str]:
    """Resolve a display symbol to its yfinance ticker, or None if unknown."""
    meta = get_symbol(asset_class, symbol)
    return meta.yf_ticker if meta else None


def is_supported(asset_class: str, symbol: str) -> bool:
    return get_symbol(asset_class, symbol) is not None


def normalize_asset_class(value: Optional[str]) -> str:
    """Coerce arbitrary input to a known asset class; default to crypto."""
    if not value:
        return ASSET_CLASS_CRYPTO
    v = str(value).strip().lower()
    if v in ASSET_CLASSES:
        return v
    # Accept plural / common aliases from the mobile + web wizards.
    _aliases = {
        "stocks": ASSET_CLASS_STOCK, "equity": ASSET_CLASS_STOCK, "equities": ASSET_CLASS_STOCK,
        "fx": ASSET_CLASS_FOREX, "currency": ASSET_CLASS_FOREX, "currencies": ASSET_CLASS_FOREX,
        "indices": ASSET_CLASS_INDEX, "indexes": ASSET_CLASS_INDEX, "idx": ASSET_CLASS_INDEX,
        "btc": ASSET_CLASS_CRYPTO, "crypto-perp": ASSET_CLASS_CRYPTO, "perp": ASSET_CLASS_CRYPTO,
    }
    if v in _aliases:
        return _aliases[v]
    return ASSET_CLASS_CRYPTO


def catalog_for_api() -> Dict[str, List[Dict[str, str]]]:
    """JSON-friendly shape for the mobile/web symbol picker."""
    out: Dict[str, List[Dict[str, str]]] = {}
    for cls, syms in _CATALOG.items():
        out[cls] = [{"symbol": m.symbol, "name": m.name} for m in syms.values()]
    return out


# ─── Market hours ───────────────────────────────────────────────────────────
# Crypto: 24/7.
# Stocks/Indices: NYSE/Nasdaq regular hours 09:30–16:00 America/New_York,
#   Mon–Fri. We use a fixed UTC offset (UTC-5 EST winter, UTC-4 EDT summer)
#   computed via the same rule the US uses — second Sunday of March through
#   first Sunday of November is DST. No external tz package required.
# Forex: open Sunday 22:00 UTC → Friday 22:00 UTC continuously (closed
#   weekends).

def _us_eastern_offset_hours(now_utc: datetime) -> int:
    """Return -5 (EST) or -4 (EDT) for the given UTC time."""
    year = now_utc.year
    # Second Sunday of March
    march = datetime(year, 3, 1)
    dst_start = march + timedelta(days=(6 - march.weekday()) % 7 + 7)
    dst_start = dst_start.replace(hour=7)  # 02:00 local = 07:00 UTC (EST)
    # First Sunday of November
    nov = datetime(year, 11, 1)
    dst_end = nov + timedelta(days=(6 - nov.weekday()) % 7)
    dst_end = dst_end.replace(hour=6)  # 02:00 local = 06:00 UTC (EDT)
    return -4 if dst_start <= now_utc < dst_end else -5


def _us_equity_open(now_utc: datetime) -> bool:
    """NYSE regular hours, Mon–Fri 09:30–16:00 ET."""
    offset = _us_eastern_offset_hours(now_utc)
    local = now_utc + timedelta(hours=offset)
    if local.weekday() >= 5:  # Sat or Sun
        return False
    open_t = time(9, 30)
    close_t = time(16, 0)
    return open_t <= local.time() < close_t


def _forex_open(now_utc: datetime) -> bool:
    """
    Forex is closed from Friday 22:00 UTC to Sunday 22:00 UTC.
    Everything else is open (Mon=0 … Sun=6).
    """
    wd = now_utc.weekday()
    hr = now_utc.hour
    # Friday after 22:00 → closed
    if wd == 4 and hr >= 22:
        return False
    # Saturday → closed
    if wd == 5:
        return False
    # Sunday before 22:00 → closed
    if wd == 6 and hr < 22:
        return False
    return True


def is_market_open(asset_class: str, now_utc: Optional[datetime] = None) -> bool:
    """
    True if the asset class is in its regular trading window right now.
    Crypto is always open. Stocks/indices follow NYSE regular hours.
    Forex follows the Mon–Fri continuous session.
    """
    cls = normalize_asset_class(asset_class)
    now_utc = now_utc or datetime.utcnow()
    if cls == ASSET_CLASS_CRYPTO:
        return True
    if cls in (ASSET_CLASS_STOCK, ASSET_CLASS_INDEX):
        return _us_equity_open(now_utc)
    if cls == ASSET_CLASS_FOREX:
        return _forex_open(now_utc)
    return True


def market_status_label(asset_class: str, now_utc: Optional[datetime] = None) -> str:
    """Short human-readable status — used in UI footers and skip-logs."""
    cls = normalize_asset_class(asset_class)
    if cls == ASSET_CLASS_CRYPTO:
        return "open · 24/7"
    return "open" if is_market_open(cls, now_utc) else "closed"
