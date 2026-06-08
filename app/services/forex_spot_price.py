"""
Unified forex / metals spot quote for HTTP consumers.

Priority:
  1. cTrader live spot (broker-matched fills)
  2. Shared Postgres tick store (ctrader / metals poller / FMP)
  3. Dedicated metals poller on-demand fetch (XAUUSD / XAGUSD)
  4. tradfi_prices fallback chain (FMP → yfinance for forex majors)
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

_METAL_SYMBOLS = frozenset({"XAUUSD", "XAGUSD"})


def _age_s(updated_at: Optional[datetime]) -> Optional[float]:
    if not updated_at:
        return None
    return round(max(0.0, (datetime.utcnow() - updated_at).total_seconds()), 1)


def _market_open(symbol: str) -> bool:
    try:
        from app.services.asset_classes import is_market_open
        cls = "forex" if symbol in _METAL_SYMBOLS else "forex"
        return bool(is_market_open(cls, datetime.utcnow()))
    except Exception:
        return True


def _quote(
    symbol: str,
    mid: float,
    *,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    source: str = "unknown",
    updated_at: Optional[datetime] = None,
) -> Dict[str, object]:
    b = float(bid if bid is not None else mid)
    a = float(ask if ask is not None else mid)
    return {
        "symbol": symbol.upper(),
        "mid": round(float(mid), 6),
        "bid": round(b, 6),
        "ask": round(a, 6),
        "source": source,
        "age_s": _age_s(updated_at),
        "market_open": _market_open(symbol),
    }


async def get_forex_spot_quote(symbol: str) -> Optional[Dict[str, object]]:
    """Return a labelled spot quote dict or None when no fresh price exists."""
    sym = (symbol or "").upper().strip()
    if not sym:
        return None

    try:
        from app.services import ctrader_price_feed as _ctf
        mid = _ctf.get_price(sym)
        if mid:
            ba = _ctf.get_bid_ask(sym)
            bid, ask = ba if ba else (mid, mid)
            return _quote(sym, mid, bid=bid, ask=ask, source="ctrader")
    except Exception:
        pass

    try:
        from app.services.spot_price_store import get_tick
        row = get_tick(sym, max_age_s=20.0)
        if row and row.get("mid"):
            src = (row.get("source") or "shared").lower()
            return _quote(
                sym,
                float(row["mid"]),
                bid=row.get("bid"),
                ask=row.get("ask"),
                source=src,
                updated_at=row.get("updated_at"),
            )
    except Exception:
        pass

    if sym in _METAL_SYMBOLS:
        try:
            from app.services.metals_spot_feed import fetch_now as _metals_fetch
            px = await _metals_fetch(sym)
            if px:
                from app.services.spot_price_store import get_tick
                row = get_tick(sym, max_age_s=20.0)
                if row and row.get("mid"):
                    return _quote(
                        sym,
                        float(row["mid"]),
                        bid=row.get("bid"),
                        ask=row.get("ask"),
                        source=(row.get("source") or "metals").lower(),
                        updated_at=row.get("updated_at"),
                    )
                return _quote(sym, float(px), source="metals")
        except Exception:
            pass

    try:
        from app.services.tradfi_prices import get_price as _tradfi_px
        px = await _tradfi_px(sym, "forex")
        if px:
            return _quote(sym, float(px), source="tradfi")
    except Exception:
        pass

    return None


async def get_forex_spot_mid(symbol: str) -> Optional[float]:
    """Mid price only — same chain as get_forex_spot_quote."""
    q = await get_forex_spot_quote(symbol)
    if not q:
        return None
    mid = q.get("mid")
    return float(mid) if mid is not None else None
