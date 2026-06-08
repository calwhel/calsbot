"""
Cross-worker spot price cache in Postgres.

Gunicorn runs multiple workers; cTrader/FMP feeds run on the executor worker only.
HTTP workers read ticks from this table so Live Forex, scanners, and feed-status
see the same real-time prices.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TABLE_READY = False


def _ensure_table() -> None:
    global _TABLE_READY
    if _TABLE_READY:
        return
    try:
        from sqlalchemy import text
        from app.database import engine

        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS market_spot_ticks (
                    symbol     VARCHAR(20) PRIMARY KEY,
                    bid        DOUBLE PRECISION,
                    ask        DOUBLE PRECISION,
                    mid        DOUBLE PRECISION NOT NULL,
                    source     VARCHAR(20) NOT NULL,
                    updated_at TIMESTAMP NOT NULL DEFAULT (NOW() AT TIME ZONE 'utc')
                )
            """))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_market_spot_ticks_updated "
                "ON market_spot_ticks (updated_at DESC)"
            ))
            conn.commit()
        _TABLE_READY = True
    except Exception as e:
        logger.warning(f"[spot_store] table init: {e}")


def upsert_tick(
    symbol: str,
    *,
    mid: float,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    source: str = "ctrader",
) -> None:
    """Persist a fresh tick (sync — call from thread pool in async hot paths)."""
    if not symbol or mid <= 0:
        return
    _ensure_table()
    sym = symbol.upper()
    try:
        from sqlalchemy import text
        from app.database import SessionLocal

        db = SessionLocal()
        try:
            db.execute(text("""
                INSERT INTO market_spot_ticks (symbol, bid, ask, mid, source, updated_at)
                VALUES (:sym, :bid, :ask, :mid, :src, :ts)
                ON CONFLICT (symbol) DO UPDATE SET
                    bid = EXCLUDED.bid,
                    ask = EXCLUDED.ask,
                    mid = EXCLUDED.mid,
                    source = EXCLUDED.source,
                    updated_at = EXCLUDED.updated_at
            """), {
                "sym": sym,
                "bid": bid,
                "ask": ask,
                "mid": float(mid),
                "src": source[:20],
                "ts": datetime.utcnow(),
            })
            db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"[spot_store] upsert {sym}: {e}")


def get_tick(symbol: str, max_age_s: float = 45.0) -> Optional[dict]:
    """Fresh tick row (symbol, bid, ask, mid, source, updated_at) or None."""
    return _get_row(symbol.upper(), max_age_s)


def get_mid(symbol: str, max_age_s: float = 45.0) -> Optional[float]:
    row = _get_row(symbol.upper(), max_age_s)
    return float(row["mid"]) if row else None


def get_bid_ask(symbol: str, max_age_s: float = 45.0) -> Optional[Tuple[float, float]]:
    row = _get_row(symbol.upper(), max_age_s)
    if not row:
        return None
    bid, ask = row.get("bid"), row.get("ask")
    if bid is not None and ask is not None:
        return (float(bid), float(ask))
    mid = float(row["mid"])
    return (mid, mid)


def _get_row(symbol: str, max_age_s: float) -> Optional[dict]:
    _ensure_table()
    cutoff = datetime.utcnow() - timedelta(seconds=max_age_s)
    try:
        from sqlalchemy import text
        from app.database import SessionLocal

        db = SessionLocal()
        try:
            row = db.execute(text("""
                SELECT symbol, bid, ask, mid, source, updated_at
                FROM market_spot_ticks
                WHERE symbol = :sym AND updated_at >= :cutoff
            """), {"sym": symbol.upper(), "cutoff": cutoff}).fetchone()
            if not row:
                return None
            return {
                "symbol": row.symbol,
                "bid": row.bid,
                "ask": row.ask,
                "mid": row.mid,
                "source": row.source,
                "updated_at": row.updated_at,
            }
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"[spot_store] get {symbol}: {e}")
        return None


def snapshot(max_age_s: float = 20.0) -> Dict[str, object]:
    """Fresh ticks for feed-status (any gunicorn worker)."""
    _ensure_table()
    cutoff = datetime.utcnow() - timedelta(seconds=max_age_s)
    out: Dict[str, object] = {
        "symbol_count": 0,
        "symbols": [],
        "by_source": {},
        "last_tick_age_s": None,
        "newest_symbol": None,
    }
    try:
        from sqlalchemy import text
        from app.database import SessionLocal

        db = SessionLocal()
        try:
            rows = db.execute(text("""
                SELECT symbol, mid, source, updated_at
                FROM market_spot_ticks
                WHERE updated_at >= :cutoff
                ORDER BY updated_at DESC
            """), {"cutoff": cutoff}).fetchall()
        finally:
            db.close()

        if not rows:
            return out

        syms: List[str] = []
        by_src: Dict[str, int] = {}
        newest_at = rows[0].updated_at
        for r in rows:
            syms.append(r.symbol)
            by_src[r.source] = by_src.get(r.source, 0) + 1
        out["symbol_count"] = len(syms)
        out["symbols"] = syms[:40]
        out["by_source"] = by_src
        out["newest_symbol"] = rows[0].symbol
        if newest_at:
            out["last_tick_age_s"] = round(
                max(0.0, (datetime.utcnow() - newest_at).total_seconds()), 1
            )
        return out
    except Exception as e:
        logger.debug(f"[spot_store] snapshot: {e}")
        out["error"] = str(e)[:120]
        return out
