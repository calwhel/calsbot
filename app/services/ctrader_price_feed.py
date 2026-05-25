"""
CTrader live spot price feed + trendbar (OHLC) fetcher.

Maintains a persistent TLS connection to live.ctraderapi.com:5035,
subscribes to spot prices for all active forex/index symbols, and
keeps a low-latency cache of current bid/ask quotes.

Also provides on-demand trendbar (kline/OHLC) fetching directly from
the cTrader server — same data FP Markets users see in their charts,
replacing the 15-minute-delayed yfinance feed for paper trading accuracy.

Architecture:
  - Module-level singleton: one background asyncio task holds the
    persistent streaming connection for spot prices.
  - Uses the first cTrader-connected user account found in the DB.
    If no accounts are connected the feed stays dormant; tradfi_prices
    falls back to yfinance automatically.
  - Trendbar (kline) fetches open a short-lived connection per call so
    the streaming loop is never blocked by a request/response exchange.
  - Auto-reconnects with exponential back-off (30s → 5 min cap).
"""
from __future__ import annotations

import asyncio
import logging
import os
import ssl
import struct
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Spotware Open API endpoint ────────────────────────────────────────────────
_HOST = "live.ctraderapi.com"
_PORT = 5035

# ── Payload type IDs (from protobuf defaults, verified against live API) ──────
_PT_APP_AUTH_REQ  = 2100
_PT_APP_AUTH_RES  = 2101
_PT_ACCT_AUTH_REQ = 2102
_PT_ACCT_AUTH_RES = 2103
_PT_SYMBOLS_REQ   = 2114
_PT_SYMBOLS_RES   = 2115
_PT_SUB_SPOTS_REQ = 2127
_PT_SUB_SPOTS_RES = 2128
_PT_SPOT_EVENT    = 2131
_PT_TRENDBARS_REQ = 2137
_PT_TRENDBARS_RES = 2138

# ── Wizard timeframe → ProtoOATrendbarPeriod enum value ──────────────────────
_TF_TO_PERIOD: Dict[str, int] = {
    "1m":  1,
    "3m":  3,
    "5m":  5,
    "10m": 6,
    "15m": 7,
    "30m": 8,
    "1h":  9,
    "4h":  10,
    "1d":  12,
}

# ── Symbols we stream (canonical name → FP Markets broker symbol name) ────────
_TRACKED: Dict[str, str] = {
    "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD", "USDCAD": "USDCAD", "USDCHF": "USDCHF",
    "NZDUSD": "NZDUSD", "EURJPY": "EURJPY", "GBPJPY": "GBPJPY",
    "XAUUSD": "XAUUSD",
    # Indices (FP Markets contract names)
    "SPX":  "US500",
    "NDX":  "US100",
    "DJI":  "US30",
    "DAX":  "GER40",
    "FTSE": "UK100",
}

# Broker symbol → canonical name (reverse map, built at start)
_BROKER_TO_CANONICAL: Dict[str, str] = {v: k for k, v in _TRACKED.items()}

_SPOT_TTL = 10.0  # seconds before a cached bid/ask is considered stale

# ── Module-level shared state ─────────────────────────────────────────────────
# symbol → (bid, ask, monotonic_ts)
_spot_cache: Dict[str, Tuple[float, float, float]] = {}
# (symbol, timeframe, limit) → (rows, monotonic_ts)
_kline_cache: Dict[Tuple[str, str, int], Tuple[List[List[float]], float]] = {}
_KLINE_TTL = 60.0

_feed_live: bool = False
_feed_task: Optional[asyncio.Task] = None
# Cached symbol-ID map from the last successful connection
_symbol_id_map: Dict[str, int] = {}   # broker_name → symbolId
_id_to_canonical: Dict[int, str] = {} # symbolId → canonical name

# ── Protobuf imports ──────────────────────────────────────────────────────────
try:
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAApplicationAuthReq,
        ProtoOAApplicationAuthRes,
        ProtoOAAccountAuthReq,
        ProtoOAAccountAuthRes,
        ProtoOASubscribeSpotsReq,
        ProtoOASubscribeSpotsRes,
        ProtoOASpotEvent,
        ProtoOASymbolsListReq,
        ProtoOASymbolsListRes,
        ProtoOAGetTrendbarsReq,
        ProtoOAGetTrendbarsRes,
    )
    _PROTO_OK = True
except ImportError as _imp_err:
    logger.warning(f"[CTraderFeed] ctrader_open_api unavailable: {_imp_err}")
    _PROTO_OK = False


# ── Low-level framing (mirrors ctrader_client.py) ────────────────────────────

def _encode(proto_req, payload_type: int) -> bytes:
    wrapper = ProtoMessage()
    wrapper.payloadType = payload_type
    wrapper.payload = proto_req.SerializeToString()
    data = wrapper.SerializeToString()
    return struct.pack(">I", len(data)) + data


def _decode(raw: bytes) -> Optional[ProtoMessage]:
    msg = ProtoMessage()
    msg.ParseFromString(raw)
    return msg


async def _read_frame(
    reader: asyncio.StreamReader,
    timeout: float = 20.0,
) -> Optional[ProtoMessage]:
    try:
        header = await asyncio.wait_for(reader.readexactly(4), timeout=timeout)
        length = struct.unpack(">I", header)[0]
        if length > 4_000_000:
            logger.warning(f"[CTraderFeed] oversized frame {length} — skipping")
            return None
        body = await asyncio.wait_for(reader.readexactly(length), timeout=20.0)
        return _decode(body)
    except (asyncio.TimeoutError, asyncio.IncompleteReadError):
        return None
    except Exception as e:
        logger.debug(f"[CTraderFeed] _read_frame error: {e}")
        return None


async def _send(writer: asyncio.StreamWriter, proto_req, payload_type: int) -> None:
    writer.write(_encode(proto_req, payload_type))
    await writer.drain()


async def _open_conn() -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    ctx = ssl.create_default_context()
    return await asyncio.open_connection(_HOST, _PORT, ssl=ctx)


async def _app_auth(reader, writer) -> bool:
    req = ProtoOAApplicationAuthReq()
    req.clientId = os.environ.get("CTRADER_CLIENT_ID", "")
    req.clientSecret = os.environ.get("CTRADER_CLIENT_SECRET", "")
    await _send(writer, req, _PT_APP_AUTH_REQ)
    msg = await _read_frame(reader, timeout=15.0)
    return msg is not None and msg.payloadType == _PT_APP_AUTH_RES


async def _account_auth(reader, writer, access_token: str, ctid: int) -> bool:
    req = ProtoOAAccountAuthReq()
    req.ctidTraderAccountId = ctid
    req.accessToken = access_token
    await _send(writer, req, _PT_ACCT_AUTH_REQ)
    msg = await _read_frame(reader, timeout=15.0)
    return msg is not None and msg.payloadType == _PT_ACCT_AUTH_RES


async def _resolve_symbols(reader, writer, ctid: int) -> Dict[str, int]:
    """Fetch the full symbol list and return broker_name → symbolId."""
    global _symbol_id_map, _id_to_canonical
    if _symbol_id_map:
        return _symbol_id_map  # use cached from previous connect

    req = ProtoOASymbolsListReq()
    req.ctidTraderAccountId = ctid
    req.includeArchivedSymbols = False
    await _send(writer, req, _PT_SYMBOLS_REQ)

    while True:
        msg = await _read_frame(reader, timeout=30.0)
        if not msg:
            raise ConnectionError("symbol list timeout")
        if msg.payloadType == _PT_SYMBOLS_RES:
            res = ProtoOASymbolsListRes()
            res.ParseFromString(msg.payload)
            name_to_id: Dict[str, int] = {}
            for sym in res.symbol:
                name_to_id[sym.symbolName] = sym.symbolId
            _symbol_id_map = name_to_id
            _id_to_canonical = {
                name_to_id[broker]: canonical
                for canonical, broker in _TRACKED.items()
                if broker in name_to_id
            }
            logger.info(
                f"[CTraderFeed] resolved {len(name_to_id)} symbols from broker"
            )
            return name_to_id
        # Skip heartbeats / other events while waiting for symbols


# ── DB helper ─────────────────────────────────────────────────────────────────

async def _get_connected_account() -> Optional[Tuple[str, int]]:
    """Return (access_token, ctid_trader_account_id) for the first connected user."""
    try:
        from app.database import SessionLocal
        from app.models import UserPreference
        db = SessionLocal()
        try:
            prefs = (
                db.query(UserPreference)
                .filter(
                    UserPreference.ctrader_access_token.isnot(None),
                    UserPreference.ctrader_account_id.isnot(None),
                )
                .first()
            )
            if prefs:
                return (prefs.ctrader_access_token, int(prefs.ctrader_account_id))
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"[CTraderFeed] DB lookup error: {e}")
    return None


# ── Background streaming task ─────────────────────────────────────────────────

async def _feed_loop() -> None:
    global _feed_live
    backoff = 30.0

    while True:
        reader = writer = None
        try:
            _feed_live = False

            creds = await _get_connected_account()
            if not creds:
                logger.info("[CTraderFeed] no connected cTrader accounts — sleeping 120s")
                await asyncio.sleep(120)
                continue

            access_token, ctid = creds
            logger.info(f"[CTraderFeed] connecting with account {ctid}…")

            reader, writer = await _open_conn()

            if not await _app_auth(reader, writer):
                raise ConnectionError("app auth failed")
            if not await _account_auth(reader, writer, access_token, ctid):
                raise ConnectionError("account auth failed")

            name_to_id = await _resolve_symbols(reader, writer, ctid)

            # Collect symbol IDs for our tracked symbols
            sub_ids: List[int] = []
            for broker_name in _TRACKED.values():
                sid = name_to_id.get(broker_name)
                if sid:
                    sub_ids.append(sid)
                else:
                    logger.debug(f"[CTraderFeed] '{broker_name}' not in broker symbol list")

            if not sub_ids:
                raise ConnectionError("none of our tracked symbols found")

            req = ProtoOASubscribeSpotsReq()
            req.ctidTraderAccountId = ctid
            req.symbolId.extend(sub_ids)
            await _send(writer, req, _PT_SUB_SPOTS_REQ)

            logger.info(
                f"[CTraderFeed] subscribed to {len(sub_ids)} symbols — LIVE"
            )
            _feed_live = True
            backoff = 30.0

            # Stream indefinitely — each SpotEvent updates the cache
            while True:
                msg = await _read_frame(reader, timeout=60.0)
                if not msg:
                    raise ConnectionError("stream read timeout / disconnect")

                if msg.payloadType == _PT_SPOT_EVENT:
                    ev = ProtoOASpotEvent()
                    ev.ParseFromString(msg.payload)
                    canonical = _id_to_canonical.get(ev.symbolId)
                    if canonical:
                        bid = ev.bid / 100_000.0 if ev.HasField("bid") else None
                        ask = ev.ask / 100_000.0 if ev.HasField("ask") else None
                        if bid and ask:
                            _spot_cache[canonical] = (bid, ask, time.monotonic())

                # Heartbeats and subscription confirmations are silently skipped

        except Exception as exc:
            _feed_live = False
            logger.warning(
                f"[CTraderFeed] disconnected ({exc}) — retry in {backoff:.0f}s"
            )
        finally:
            if writer:
                try:
                    writer.close()
                except Exception:
                    pass

        await asyncio.sleep(backoff)
        backoff = min(backoff * 1.5, 300.0)


# ── On-demand trendbar (kline) fetch ─────────────────────────────────────────

async def _fetch_trendbars(
    symbol: str,
    timeframe: str,
    limit: int,
    access_token: str,
    ctid: int,
) -> List[List[float]]:
    """
    Open a short-lived connection and fetch up to `limit` OHLC bars for
    `symbol` at `timeframe` from the cTrader server.

    Returns rows in [[ts_ms, open, high, low, close, volume], ...] format.
    """
    period = _TF_TO_PERIOD.get(timeframe)
    if period is None:
        logger.debug(f"[CTraderFeed] unsupported timeframe {timeframe!r}")
        return []

    # Resolve broker symbol name and ID
    broker_name = _TRACKED.get(symbol.upper(), symbol.upper())
    symbol_id = _symbol_id_map.get(broker_name)

    reader = writer = None
    try:
        reader, writer = await _open_conn()

        if not await _app_auth(reader, writer):
            return []
        if not await _account_auth(reader, writer, access_token, ctid):
            return []

        # If we don't have the symbol map yet, fetch it now
        if not _symbol_id_map:
            await _resolve_symbols(reader, writer, ctid)
            symbol_id = _symbol_id_map.get(broker_name)

        if not symbol_id:
            logger.debug(f"[CTraderFeed] symbol_id not found for {broker_name}")
            return []

        now_ms = int(time.time() * 1000)
        # Request enough history. cTrader max is 4096 bars per call.
        req = ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = ctid
        req.symbolId = symbol_id
        req.period = period
        req.count = min(limit, 4096)
        req.toTimestamp = now_ms
        await _send(writer, req, _PT_TRENDBARS_REQ)

        # Read frames until we get the trendbars response
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            msg = await _read_frame(reader, timeout=remaining)
            if not msg:
                break
            if msg.payloadType == _PT_TRENDBARS_RES:
                res = ProtoOAGetTrendbarsRes()
                res.ParseFromString(msg.payload)
                rows: List[List[float]] = []
                for bar in res.trendbar:
                    low   = bar.low / 100_000.0
                    open_ = (bar.low + bar.deltaOpen)  / 100_000.0
                    high  = (bar.low + bar.deltaHigh)  / 100_000.0
                    close = (bar.low + bar.deltaClose) / 100_000.0
                    ts_ms = bar.utcTimestampInMinutes * 60 * 1000
                    vol   = float(bar.volume)
                    rows.append([ts_ms, open_, high, low, close, vol])
                return rows[-limit:]
        return []

    except Exception as e:
        logger.debug(f"[CTraderFeed] trendbar fetch failed for {symbol}: {e}")
        return []
    finally:
        if writer:
            try:
                writer.close()
            except Exception:
                pass


async def get_klines(
    symbol: str,
    asset_class: str,
    timeframe: str = "15m",
    limit: int = 200,
) -> List[List[float]]:
    """
    Return up to `limit` OHLC bars from cTrader for the given symbol.
    Returns [] if the symbol is not tracked, no account is connected,
    or cTrader proto is unavailable (caller falls back to yfinance).
    """
    if not _PROTO_OK:
        return []
    sym_up = symbol.upper()
    if sym_up not in _TRACKED:
        return []

    cache_key = (sym_up, timeframe, limit)
    cached = _kline_cache.get(cache_key)
    if cached and (time.monotonic() - cached[1]) < _KLINE_TTL:
        return cached[0]

    creds = await _get_connected_account()
    if not creds:
        return []

    access_token, ctid = creds
    rows = await _fetch_trendbars(sym_up, timeframe, limit, access_token, ctid)
    if rows:
        _kline_cache[cache_key] = (rows, time.monotonic())
    return rows


# ── Public price API ──────────────────────────────────────────────────────────

def get_price(symbol: str) -> Optional[float]:
    """
    Return the mid price for `symbol` from the live cTrader spot feed.
    Returns None if the symbol is not tracked or the last tick is stale.
    """
    entry = _spot_cache.get(symbol.upper())
    if not entry:
        return None
    bid, ask, ts = entry
    if time.monotonic() - ts > _SPOT_TTL:
        return None
    return round((bid + ask) / 2.0, 6)


def get_bid_ask(symbol: str) -> Optional[Tuple[float, float]]:
    """Return (bid, ask) for `symbol` if a fresh tick is available."""
    entry = _spot_cache.get(symbol.upper())
    if not entry:
        return None
    bid, ask, ts = entry
    if time.monotonic() - ts > _SPOT_TTL:
        return None
    return (bid, ask)


def is_live() -> bool:
    """True when the streaming connection is up and receiving ticks."""
    return _feed_live


def cached_symbols() -> List[str]:
    """Return list of canonical symbols with a fresh cached price."""
    now = time.monotonic()
    return [
        sym for sym, (_, _, ts) in _spot_cache.items()
        if now - ts <= _SPOT_TTL
    ]


# ── Startup ───────────────────────────────────────────────────────────────────

def start() -> None:
    """
    Launch the background streaming task.
    Safe to call multiple times — ignored if already running.
    """
    global _feed_task
    if not _PROTO_OK:
        logger.warning("[CTraderFeed] protobuf unavailable — feed disabled")
        return
    if not os.environ.get("CTRADER_CLIENT_ID"):
        logger.warning("[CTraderFeed] CTRADER_CLIENT_ID not set — feed disabled")
        return
    if _feed_task and not _feed_task.done():
        return  # already running
    try:
        loop = asyncio.get_event_loop()
        _feed_task = loop.create_task(_feed_loop())
        logger.info("[CTraderFeed] background streaming task started")
    except Exception as e:
        logger.error(f"[CTraderFeed] failed to start task: {e}")
