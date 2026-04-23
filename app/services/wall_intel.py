"""
Wall Intel — historical wall tracking, watch alerts, volume + liquidation context.

Companion to liquidity_walls.py. Adds:
- WallSnapshot persistence so /walls can detect FRESH / GROWING / SHRINKING walls
  by comparing the current scan against the previous snapshot for the same symbol.
- WallWatch table for the /walls watch command — users subscribe to a symbol and
  the background loop DMs them when a major wall breaks or a fresh one appears.
- Volume context (24h USD) from MEXC tickers.
- Liquidation context (24h long/short USD) from Coinglass.

All DB writes are best-effort; failures are logged but never block /walls.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from typing import Optional

import httpx
from sqlalchemy import text

from app.database import engine

logger = logging.getLogger(__name__)

SNAPSHOT_TTL_HOURS = 24            # how long to keep snapshots before pruning
SNAPSHOT_LOOKBACK_MINS = 30        # how far back the "previous" snapshot can be
WATCH_POLL_SECONDS = 75            # background watch loop cadence
WATCH_ALERT_DEDUPE_MINS = 8        # don't repeat the SAME alert signature within this window
WATCH_ALERT_HARD_FLOOR_MINS = 3    # never DM the same user about the same symbol more often than this
WATCH_BIG_WALL_USD = 1_000_000.0   # only alert on walls ≥ this size
WATCH_BREAK_BAND_PCT = 0.15        # consider a wall "broken" if price crossed it by this %
WATCH_APPROACH_PCT = 0.30          # alert when price is within this % of a big wall
WATCH_APPROACH_MIN_PCT = 0.04      # ...but ignore walls that are essentially at price (already touched)


def _sig_price(p: float) -> str:
    """Stable, magnitude-aware price key for alert signatures.
    Uses 6 significant figures so a 0.0000123 wall and a 60123.45 wall both
    keep enough precision to differentiate adjacent levels."""
    try:
        return f"{float(p):.6g}"
    except (TypeError, ValueError):
        return "0"


# ───────────────────────── Schema (idempotent) ─────────────────────────

def init_wall_intel_schema():
    """Create the wall_snapshots + wall_watches tables if missing. Safe to call repeatedly."""
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS wall_snapshots (
                    id BIGSERIAL PRIMARY KEY,
                    symbol VARCHAR(32) NOT NULL,
                    captured_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    price DOUBLE PRECISION NOT NULL,
                    walls_json JSONB NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_wall_snapshots_symbol_time
                ON wall_snapshots (symbol, captured_at DESC)
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS wall_watches (
                    id BIGSERIAL PRIMARY KEY,
                    telegram_user_id BIGINT NOT NULL,
                    symbol VARCHAR(32) NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    last_alert_at TIMESTAMP,
                    last_alert_signature VARCHAR(120)
                )
            """))
            conn.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_wall_watches_user_symbol
                ON wall_watches (telegram_user_id, symbol)
            """))
        logger.info("wall_intel: schema ready")
    except Exception as e:
        logger.warning(f"wall_intel: schema init failed: {e}")


# ───────────────────────── Snapshot persistence ─────────────────────────

def _wall_to_dict(w) -> dict:
    """Serialize a Wall dataclass to a JSON-safe dict."""
    if is_dataclass(w):
        d = asdict(w)
    else:
        d = {
            "price": w.price, "size_usd": w.size_usd, "size_native": w.size_native,
            "distance_pct": w.distance_pct, "side": w.side,
            "exchanges": list(getattr(w, "exchanges", [])),
        }
    # exchanges might be a set in older paths
    d["exchanges"] = list(d.get("exchanges") or [])
    return {k: d[k] for k in ("price", "size_usd", "size_native", "distance_pct", "side", "exchanges") if k in d}


def save_snapshot(symbol: str, price: float, walls: list) -> None:
    """Persist current walls to wall_snapshots. Best-effort."""
    try:
        payload = json.dumps([_wall_to_dict(w) for w in walls])
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO wall_snapshots (symbol, price, walls_json)
                    VALUES (:sym, :px, CAST(:walls AS JSONB))
                """),
                {"sym": symbol, "px": price, "walls": payload},
            )
            # Opportunistic prune
            conn.execute(
                text("DELETE FROM wall_snapshots WHERE captured_at < :cutoff"),
                {"cutoff": datetime.utcnow() - timedelta(hours=SNAPSHOT_TTL_HOURS)},
            )
    except Exception as e:
        logger.debug(f"wall_intel.save_snapshot({symbol}) failed: {e}")


def load_previous_snapshot(symbol: str, exclude_within_secs: int = 30) -> Optional[dict]:
    """Return the most recent snapshot for `symbol` that is at least
    `exclude_within_secs` old (so we don't compare against the snapshot we just wrote).
    Returns dict {price, walls, age_secs} or None."""
    try:
        cutoff_recent = datetime.utcnow() - timedelta(seconds=exclude_within_secs)
        cutoff_old = datetime.utcnow() - timedelta(minutes=SNAPSHOT_LOOKBACK_MINS)
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT captured_at, price, walls_json
                    FROM wall_snapshots
                    WHERE symbol = :sym
                      AND captured_at <= :recent
                      AND captured_at >= :old
                    ORDER BY captured_at DESC
                    LIMIT 1
                """),
                {"sym": symbol, "recent": cutoff_recent, "old": cutoff_old},
            ).first()
        if not row:
            return None
        captured_at, prev_price, walls_json = row
        # walls_json may already be a list (psycopg2 decodes JSONB) or a string
        walls = walls_json if isinstance(walls_json, list) else json.loads(walls_json or "[]")
        return {
            "price": float(prev_price),
            "walls": walls,
            "age_secs": int((datetime.utcnow() - captured_at).total_seconds()),
        }
    except Exception as e:
        logger.debug(f"wall_intel.load_previous_snapshot({symbol}) failed: {e}")
        return None


def classify_wall_changes(current_walls: list, prev_snapshot: Optional[dict]) -> dict:
    """Compare current walls against the previous snapshot.

    Returns dict mapping rounded-price-key → label in
    {"FRESH", "GROWING", "SHRINKING", "STABLE"}.

    Also returns a list of "broken" walls (significant prev walls that no longer
    exist in the current scan) under key "_broken" with each entry as a dict.
    """
    if not prev_snapshot:
        return {}

    prev_walls = prev_snapshot.get("walls") or []
    if not prev_walls:
        return {}

    # Build a price-bucketed lookup for previous walls (within 0.15% of price = same wall)
    def _key(p: float) -> float:
        return round(p, 8)

    def _matches(current_price: float, prev_price: float) -> bool:
        if current_price <= 0 or prev_price <= 0:
            return False
        return abs(current_price - prev_price) / current_price < 0.0015  # 0.15% drift

    out: dict = {}
    matched_prev_indices = set()

    for cw in current_walls:
        cw_price = float(cw.price if hasattr(cw, "price") else cw["price"])
        cw_size = float(cw.size_usd if hasattr(cw, "size_usd") else cw["size_usd"])
        cw_side = cw.side if hasattr(cw, "side") else cw.get("side")

        match_idx = None
        for i, pw in enumerate(prev_walls):
            if i in matched_prev_indices:
                continue
            if pw.get("side") != cw_side:
                continue
            if _matches(cw_price, float(pw.get("price", 0))):
                match_idx = i
                break

        key = _key(cw_price)
        if match_idx is None:
            out[key] = "FRESH"
        else:
            matched_prev_indices.add(match_idx)
            prev_size = float(prev_walls[match_idx].get("size_usd", 0))
            if prev_size <= 0:
                out[key] = "STABLE"
            else:
                ratio = cw_size / prev_size
                if ratio > 1.25:
                    out[key] = "GROWING"
                elif ratio < 0.60:
                    out[key] = "SHRINKING"
                else:
                    out[key] = "STABLE"

    # Walls that were significant before but no longer present = potentially broken/absorbed
    broken = []
    for i, pw in enumerate(prev_walls):
        if i in matched_prev_indices:
            continue
        if float(pw.get("size_usd", 0)) >= WATCH_BIG_WALL_USD * 0.5:  # only flag $500k+ disappearances
            broken.append({
                "price": float(pw.get("price", 0)),
                "size_usd": float(pw.get("size_usd", 0)),
                "side": pw.get("side"),
            })
    if broken:
        out["_broken"] = broken
    return out


# ───────────────────────── Volume context (MEXC) ─────────────────────────

async def get_volume_context(symbol: str, client: Optional[httpx.AsyncClient] = None) -> dict:
    """Fetch 24h volume + 1h volume from MEXC. Returns {volume_24h_usd, volume_1h_usd}."""
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=5.0)
    try:
        out = {"volume_24h_usd": 0.0, "volume_1h_usd": 0.0}
        try:
            r = await client.get(
                "https://api.mexc.com/api/v3/ticker/24hr",
                params={"symbol": symbol},
            )
            r.raise_for_status()
            d = r.json()
            out["volume_24h_usd"] = float(d.get("quoteVolume") or 0)
        except Exception as e:
            logger.debug(f"wall_intel.get_volume_context 24h failed for {symbol}: {e}")

        # Last 1h volume from 1m klines (60 candles)
        try:
            r = await client.get(
                "https://api.mexc.com/api/v3/klines",
                params={"symbol": symbol, "interval": "1m", "limit": 60},
            )
            r.raise_for_status()
            kl = r.json() or []
            # MEXC kline rows: [openTime, open, high, low, close, volume, closeTime, quoteVolume, ...]
            quote_vol = 0.0
            for row in kl:
                if len(row) >= 8:
                    try:
                        quote_vol += float(row[7] or 0)
                    except (TypeError, ValueError):
                        pass
            out["volume_1h_usd"] = quote_vol
        except Exception as e:
            logger.debug(f"wall_intel.get_volume_context 1h failed for {symbol}: {e}")

        return out
    finally:
        if own_client:
            await client.aclose()


# ───────────────────────── Liquidation context (Coinglass) ─────────────────────────

async def get_liquidation_context(symbol: str) -> dict:
    """Fetch 24h liquidations from Coinglass. Always returns a dict (zeros on failure)."""
    out = {"liq_24h_long_usd": 0.0, "liq_24h_short_usd": 0.0, "liq_24h_total_usd": 0.0}
    try:
        if not os.getenv("COINGLASS_API_KEY"):
            return out
        from app.services.coinglass import get_liquidation_data
        liq = await get_liquidation_data(symbol)
        if liq:
            out["liq_24h_long_usd"] = float(liq.get("long_liquidations_usd") or 0)
            out["liq_24h_short_usd"] = float(liq.get("short_liquidations_usd") or 0)
            out["liq_24h_total_usd"] = float(liq.get("total_liquidations_usd") or 0)
    except Exception as e:
        logger.debug(f"wall_intel.get_liquidation_context failed for {symbol}: {e}")
    return out


# ───────────────────────── /walls watch — subscriptions ─────────────────────────

def add_watch(telegram_user_id: int, symbol: str) -> bool:
    """Subscribe a user to wall alerts for `symbol`. Returns True if newly added."""
    try:
        with engine.begin() as conn:
            r = conn.execute(
                text("""
                    INSERT INTO wall_watches (telegram_user_id, symbol)
                    VALUES (:uid, :sym)
                    ON CONFLICT (telegram_user_id, symbol) DO NOTHING
                    RETURNING id
                """),
                {"uid": telegram_user_id, "sym": symbol},
            ).first()
        return r is not None
    except Exception as e:
        logger.warning(f"wall_intel.add_watch({telegram_user_id}, {symbol}) failed: {e}")
        return False


def remove_watch(telegram_user_id: int, symbol: str) -> bool:
    """Unsubscribe a user. Returns True if a row was deleted."""
    try:
        with engine.begin() as conn:
            r = conn.execute(
                text("""
                    DELETE FROM wall_watches
                    WHERE telegram_user_id = :uid AND symbol = :sym
                    RETURNING id
                """),
                {"uid": telegram_user_id, "sym": symbol},
            ).first()
        return r is not None
    except Exception as e:
        logger.warning(f"wall_intel.remove_watch({telegram_user_id}, {symbol}) failed: {e}")
        return False


def list_watches(telegram_user_id: int) -> list[str]:
    """All symbols a user is watching."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT symbol FROM wall_watches WHERE telegram_user_id = :uid ORDER BY symbol"),
                {"uid": telegram_user_id},
            ).fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        logger.warning(f"wall_intel.list_watches({telegram_user_id}) failed: {e}")
        return []


def _list_unique_watched_symbols() -> list[str]:
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT DISTINCT symbol FROM wall_watches")
            ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def _list_watchers_for_symbol(symbol: str) -> list[tuple[int, Optional[str], Optional[datetime]]]:
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT telegram_user_id, last_alert_signature, last_alert_at
                    FROM wall_watches WHERE symbol = :sym
                """),
                {"sym": symbol},
            ).fetchall()
        return [(int(r[0]), r[1], r[2]) for r in rows]
    except Exception:
        return []


def _record_alert_sent(uid: int, symbol: str, signature: str):
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE wall_watches
                    SET last_alert_at = NOW(), last_alert_signature = :sig
                    WHERE telegram_user_id = :uid AND symbol = :sym
                """),
                {"uid": uid, "sym": symbol, "sig": signature[:120]},
            )
    except Exception as e:
        logger.debug(f"wall_intel._record_alert_sent failed: {e}")


def _build_alert_signature(report) -> tuple[str, str]:
    """Inspect a fresh WallReport and decide if there's anything alert-worthy.
    Returns (signature, human_message) or ("", "") if nothing meaningful.

    Triggers on:
      - any wall classified FRESH and ≥ WATCH_BIG_WALL_USD
      - any wall classified SHRINKING and previously ≥ WATCH_BIG_WALL_USD
      - any "_broken" entry (recent big wall that vanished or got crossed)
    """
    try:
        behavior = getattr(report, "wall_behavior", {}) or {}
    except Exception:
        return "", ""

    parts: list[str] = []
    sig_parts: list[str] = []

    broken = behavior.get("_broken") if isinstance(behavior, dict) else None
    if broken:
        for b in broken:
            side_lbl = "buy support" if b.get("side") == "buy" else "sell wall"
            parts.append(
                f"💥 {side_lbl} at {b['price']:.6g} (~${b['size_usd']/1e6:.2f}M) is gone — broken or absorbed"
            )
            sig_parts.append(f"brk_{b.get('side')}_{_sig_price(b['price'])}")

    # Iterate top walls. Two trigger types per wall:
    #   - FRESH big walls (newly appeared)
    #   - APPROACHING — price is within WATCH_APPROACH_PCT of a $1M+ wall (any behavior)
    for w in (list(getattr(report, "top_buys", []) or []) + list(getattr(report, "top_sells", []) or [])):
        if w.size_usd < WATCH_BIG_WALL_USD:
            continue
        key = round(w.price, 8)
        label = behavior.get(key) if isinstance(behavior, dict) else None
        side_lbl = "buy" if w.side == "buy" else "sell"

        if label == "FRESH":
            parts.append(
                f"🆕 fresh {side_lbl} wall @ {w.price:.6g} "
                f"(${w.size_usd/1e6:.2f}M, {w.distance_pct:+.2f}%)"
            )
            sig_parts.append(f"fresh_{w.side}_{_sig_price(w.price)}")

        # Approaching trigger: |distance| within (MIN, MAX) and the wall is still standing
        # (SHRINKING walls in approach zone are MORE urgent — flag them too)
        abs_dist = abs(w.distance_pct)
        if WATCH_APPROACH_MIN_PCT < abs_dist <= WATCH_APPROACH_PCT:
            tag = ""
            if label == "SHRINKING":
                tag = " ⚠️ shrinking — likely breaks"
            elif label == "GROWING":
                tag = " 💪 growing — strong defense"
            wall_kind = "support" if w.side == "buy" else "resistance"
            parts.append(
                f"🚨 price approaching {side_lbl} {wall_kind} @ {w.price:.6g} "
                f"(${w.size_usd/1e6:.2f}M, {w.distance_pct:+.2f}% away){tag}"
            )
            sig_parts.append(f"appr_{w.side}_{_sig_price(w.price)}")

    if not parts:
        return "", ""
    sig = "|".join(sorted(sig_parts))
    msg = "\n".join(parts)
    return sig, msg


# ───────────────────────── Background watch loop ─────────────────────────

async def watch_loop():
    """Polls watched symbols every WATCH_POLL_SECONDS and DMs subscribers on
    significant wall changes. Runs forever; exceptions are caught + logged."""
    from app.services.liquidity_walls import scan_walls
    # Lazy import to avoid circular dependency at module load
    from app.services.bot import bot  # type: ignore

    logger.info(f"🛡️ wall_intel watch_loop started — polling every {WATCH_POLL_SECONDS}s")
    while True:
        try:
            symbols = _list_unique_watched_symbols()
            if symbols:
                # Cap concurrent scans so we don't hammer exchanges
                sem = asyncio.Semaphore(3)

                async def _process(sym: str):
                    async with sem:
                        try:
                            report = await scan_walls(sym, use_ai=False)
                            if not report:
                                return
                            sig, msg = _build_alert_signature(report)
                            if not sig:
                                return
                            import html as _html
                            sym_safe = _html.escape(sym)
                            for uid, last_sig, last_at in _list_watchers_for_symbol(sym):
                                # Hard floor: never DM the same user about the same symbol more
                                # often than HARD_FLOOR_MINS, regardless of signature.
                                if last_at:
                                    age_min = (datetime.utcnow() - last_at).total_seconds() / 60
                                    if age_min < WATCH_ALERT_HARD_FLOOR_MINS:
                                        continue
                                    # Same signature dedupe (longer window)
                                    if last_sig == sig and age_min < WATCH_ALERT_DEDUPE_MINS:
                                        continue
                                try:
                                    text_msg = (
                                        f"🚨 <b>Wall alert · {sym_safe}</b>\n"
                                        f"Price: {report.price:.6g}\n\n"
                                        f"{msg}\n\n"
                                        f"<i>Run /walls {sym_safe} for full context.</i>"
                                    )
                                    await bot.send_message(uid, text_msg, parse_mode="HTML")
                                    _record_alert_sent(uid, sym, sig)
                                except Exception as e:
                                    logger.debug(f"wall_intel: send to {uid} failed: {e}")
                        except Exception as e:
                            logger.debug(f"wall_intel watch scan failed for {sym}: {e}")

                await asyncio.gather(*[_process(s) for s in symbols])
        except Exception as e:
            logger.warning(f"wall_intel watch_loop iteration error: {e}")
        await asyncio.sleep(WATCH_POLL_SECONDS)
