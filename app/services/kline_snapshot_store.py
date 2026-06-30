"""
Cross-worker cTrader kline snapshots in Postgres.

The executor/feed process fetches trendbars and persists OHLC rows here.
Portal gunicorn workers read snapshots (no competing cTrader sockets).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

_TABLE_READY = False
# Legacy default — prefer snapshot_row_max_age_s(timeframe) when max_age_s omitted.
_DEFAULT_MAX_AGE_S = 300.0


def snapshot_row_max_age_s(timeframe: str) -> float:
    """Max age of snapshot upsert (updated_at) before row is ignored."""
    defaults = {
        "1m": 300.0,
        "5m": 600.0,
        "15m": 900.0,
        "1h": 7200.0,
        "4h": 14400.0,
    }
    default = defaults.get(timeframe, 900.0)
    env_key = f"KLINE_SNAPSHOT_MAX_AGE_{timeframe.upper()}_S"
    raw = os.environ.get(env_key) or os.environ.get("KLINE_SNAPSHOT_MAX_AGE_S")
    if raw:
        try:
            return max(60.0, float(raw))
        except (TypeError, ValueError):
            pass
    return default


def _ensure_table() -> None:
    global _TABLE_READY
    if _TABLE_READY:
        return
    try:
        from sqlalchemy import text
        from app.database import engine

        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS market_kline_snapshots (
                    symbol      VARCHAR(20) NOT NULL,
                    timeframe   VARCHAR(10) NOT NULL,
                    bars_json   JSONB NOT NULL,
                    source      VARCHAR(20) NOT NULL DEFAULT 'ctrader',
                    updated_at  TIMESTAMP NOT NULL DEFAULT (NOW() AT TIME ZONE 'utc'),
                    PRIMARY KEY (symbol, timeframe)
                )
            """))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_market_kline_snapshots_updated "
                "ON market_kline_snapshots (updated_at DESC)"
            ))
            conn.commit()
        _TABLE_READY = True
    except Exception as exc:
        logger.warning("[kline_snapshot] table init: %s", exc)


def upsert_klines(
    symbol: str,
    timeframe: str,
    rows: List[List[float]],
    *,
    source: str = "ctrader",
) -> None:
    """Persist OHLC rows for cross-process readers (sync)."""
    if not rows:
        return
    _ensure_table()
    sym = symbol.upper()
    try:
        from sqlalchemy import text
        from app.database import SessionLocal

        payload = json.dumps(rows)
        db = SessionLocal()
        try:
            db.execute(text("""
                INSERT INTO market_kline_snapshots
                    (symbol, timeframe, bars_json, source, updated_at)
                VALUES (:sym, :tf, CAST(:bars AS JSONB), :src, :ts)
                ON CONFLICT (symbol, timeframe) DO UPDATE SET
                    bars_json = EXCLUDED.bars_json,
                    source = EXCLUDED.source,
                    updated_at = EXCLUDED.updated_at
            """), {
                "sym": sym,
                "tf": timeframe,
                "bars": payload,
                "src": source[:20],
                "ts": datetime.utcnow(),
            })
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        logger.debug("[kline_snapshot] upsert %s %s: %s", sym, timeframe, exc)


def get_klines(
    symbol: str,
    timeframe: str,
    limit: int,
    *,
    max_age_s: Optional[float] = None,
    source: Optional[str] = None,
) -> List[List[float]]:
    """Fresh snapshot rows or []."""
    _ensure_table()
    sym = symbol.upper()
    if max_age_s is None:
        max_age_s = snapshot_row_max_age_s(timeframe)
    cutoff = datetime.utcnow() - timedelta(seconds=max_age_s)
    try:
        from sqlalchemy import text
        from app.database import SessionLocal

        db = SessionLocal()
        try:
            row = db.execute(text("""
                SELECT bars_json, source
                FROM market_kline_snapshots
                WHERE symbol = :sym
                  AND timeframe = :tf
                  AND updated_at >= :cutoff
            """), {"sym": sym, "tf": timeframe, "cutoff": cutoff}).fetchone()
        finally:
            db.close()
        if not row:
            return []
        src = (row.source or "").lower()
        if source and src != source.lower():
            return []
        raw = row.bars_json
        if isinstance(raw, str):
            bars = json.loads(raw)
        else:
            bars = raw
        if not isinstance(bars, list) or not bars:
            return []
        out: List[List[float]] = []
        for bar in bars[-limit:]:
            if isinstance(bar, (list, tuple)) and len(bar) >= 5:
                out.append([float(x) for x in bar[:6]])
        return out
    except Exception as exc:
        logger.debug("[kline_snapshot] get %s %s: %s", sym, timeframe, exc)
        return []
