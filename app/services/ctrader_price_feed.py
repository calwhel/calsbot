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
# cTrader uses SEPARATE hosts for live vs demo accounts. Authenticating an
# account against the wrong host fails ("account auth failed"), so the host MUST
# match the account's isLive flag.
_HOST_LIVE = "live.ctraderapi.com"
_HOST_DEMO = "demo.ctraderapi.com"
_HOST = _HOST_LIVE  # default / back-compat
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
_PT_HEARTBEAT     = 51    # ProtoHeartbeatEvent — client MUST send ~every 10s or
                         # cTrader drops the socket (the "stream read timeout" churn)

# ── Reconnect tuning ─────────────────────────────────────────────────────────
# cTrader periodically recycles long-lived spot sessions (especially when a
# second authed socket — the on-demand trendbar fetch — shares the same account
# id). We can't stop the broker recycling, so we make the reconnect SEAMLESS:
# after a session that streamed fine for a while drops, reconnect almost
# immediately so the sub-second feed blackout is negligible. A growing backoff
# is reserved ONLY for genuine connect/auth/subscribe failures so we never
# hammer the broker while still being effectively always-on.
_READ_TIMEOUT          = 35.0   # max stream silence before declaring the socket dead
_HEALTHY_SESSION_SECS  = 20.0   # a session alive ≥ this counts as "healthy"
_RECONNECT_FAST        = 2.0    # delay after a healthy drop (near-seamless)
_RECONNECT_BACKOFF_MIN = 3.0    # initial backoff for connect/auth failures
_RECONNECT_BACKOFF_MAX = 60.0   # cap (was 300s — far too long for an always-on feed)

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

# Minutes per bar for each timeframe — used to size the trendbar fromTimestamp.
_TF_MINUTES: Dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "10m": 10, "15m": 15,
    "30m": 30, "1h": 60, "4h": 240, "1d": 1440,
}

# ── Symbols we stream (canonical name → FP Markets broker symbol name) ────────
_TRACKED: Dict[str, str] = {
    "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD", "USDCAD": "USDCAD", "USDCHF": "USDCHF",
    "NZDUSD": "NZDUSD", "EURJPY": "EURJPY", "GBPJPY": "GBPJPY",
    "XAUUSD": "XAUUSD",
    # Indices — canonical cTrader names (FP Markets contract names)
    "NAS100": "US100", "NDX": "US100", "US100": "US100",
    "SPX500": "US500", "SPX": "US500",  "US500": "US500",
    "US30":   "US30",  "DJI": "US30",
    "GER40":  "GER40", "DAX": "GER40",
    "UK100":  "UK100", "FTSE": "UK100",
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

# ── Persistent trendbar (candle) connection ──────────────────────────────────
# A SINGLE authed connection reused across all trendbar fetches, serialized by a
# lock. Opening a fresh short-lived connection per candle request (the old
# behaviour) meant dozens of concurrent SSL+auth handshakes per scan cycle, which
# cTrader throttles/rejects → every fetch failed → callers fell back to yfinance
# (gold *futures* for gold). Reusing one warm connection makes steady-state
# fetches a single round-trip and matches the broker's own chart.
_tb_lock: Optional["asyncio.Lock"] = None
_tb_conn: Optional[Tuple[asyncio.StreamReader, asyncio.StreamWriter]] = None
_tb_conn_ctx: Optional[Tuple[str, int]] = None  # (host, ctid) the conn is authed for
_tb_last_use: float = 0.0
_TB_IDLE_MAX = 25.0  # cTrader drops idle conns ~30s → proactively reopen past this

_feed_live: bool = False
_feed_task: Optional[asyncio.Task] = None
_wake_event: Optional[asyncio.Event] = None
# Cached symbol-ID map from the last successful connection.
# Symbol IDs differ per host/account, so the cache is scoped to the
# (host, ctid) it was resolved from — switching host/account forces a
# re-resolve to avoid mapping a symbolId to the WRONG instrument.
_symbol_id_map: Dict[str, int] = {}   # broker_name → symbolId
_id_to_canonical: Dict[int, str] = {} # symbolId → canonical name
_symbol_map_ctx: Optional[Tuple[str, int]] = None  # (host, ctid) cache belongs to

# ── Protobuf imports ──────────────────────────────────────────────────────────
try:
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage, ProtoHeartbeatEvent
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


async def _open_conn(host: str = _HOST_LIVE) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    ctx = ssl.create_default_context()
    return await asyncio.open_connection(host, _PORT, ssl=ctx)


async def _aclose_writer(writer) -> None:
    """Close an asyncio SSL StreamWriter AND wait for the FD to be released.

    A bare writer.close() does NOT promptly free the underlying SSL socket file
    descriptor. On hot paths (per-fetch trendbar connections, reconnect churn)
    that leaks FDs until the process hits 'Too many open files' — which then
    breaks DNS, DB, and HTTP. wait_closed() (timeout-guarded so a dead peer
    can't hang us) ensures the FD is actually released.
    """
    if writer is None:
        return
    try:
        writer.close()
    except Exception:
        pass
    try:
        await asyncio.wait_for(writer.wait_closed(), timeout=5.0)
    except Exception:
        pass


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


async def _resolve_symbols(reader, writer, ctid: int, host: str = _HOST_LIVE) -> Dict[str, int]:
    """Fetch the full symbol list and return broker_name → symbolId.

    The cache is scoped to (host, ctid): if a previous resolve was for a
    DIFFERENT host/account, we MUST re-resolve — symbolIds are not portable
    across accounts and reusing them would map to the wrong instrument.
    """
    global _symbol_id_map, _id_to_canonical, _symbol_map_ctx
    if _symbol_id_map and _symbol_map_ctx == (host, ctid):
        return _symbol_id_map  # cached for THIS host/account

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
            _symbol_map_ctx = (host, ctid)
            logger.info(
                f"[CTraderFeed] resolved {len(name_to_id)} symbols from broker"
            )
            return name_to_id
        # Skip heartbeats / other events while waiting for symbols


# ── DB helper ─────────────────────────────────────────────────────────────────

async def _get_connected_account(
    user_id: Optional[int] = None,
) -> Optional[Tuple[str, int, int, bool]]:
    """Return (access_token, ctid, user_id, is_live) for a connected cTrader user.

    When `user_id` is given, use that user's linked account (e.g. gold scan for
    the logged-in portal user). Otherwise fall back to the first connected row.

    is_live is resolved from the stored ctrader_accounts list (matching the
    chosen ctid) so we connect to the matching host; defaults to True when the
    metadata is unavailable (loop falls back to the other host on auth failure).
    """
    try:
        import json as _json
        from app.database import SessionLocal
        from app.models import UserPreference
        db = SessionLocal()
        try:
            q = db.query(UserPreference).filter(
                UserPreference.ctrader_access_token.isnot(None),
                UserPreference.ctrader_account_id.isnot(None),
            )
            if user_id is not None:
                q = q.filter(UserPreference.user_id == int(user_id))
            prefs = q.first()
            if prefs:
                ctid = int(prefs.ctrader_account_id)
                is_live = True
                try:
                    raw = getattr(prefs, "ctrader_accounts", None)
                    if raw:
                        for a in _json.loads(raw):
                            if int(a.get("ctidTraderAccountId", -1)) == ctid:
                                is_live = bool(a.get("isLive", True))
                                break
                except Exception:
                    pass
                return (prefs.ctrader_access_token, ctid, int(prefs.user_id), is_live)
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"[CTraderFeed] DB lookup error: {e}")
    return None


# ── Background streaming task ─────────────────────────────────────────────────

async def _feed_loop() -> None:
    global _feed_live
    fail_backoff = _RECONNECT_BACKOFF_MIN  # grows ONLY on connect/auth failures

    while True:
        reader = writer = None
        hb_task = None
        healthy = False                # True once spots are flowing
        session_start = time.monotonic()
        try:
            _feed_live = False

            creds = await _get_connected_account()
            if not creds:
                logger.info("[CTraderFeed] no connected cTrader accounts — waiting up to 120s")
                wake = _get_wake_event()
                try:
                    await asyncio.wait_for(wake.wait(), timeout=120.0)
                    wake.clear()
                    logger.info("[CTraderFeed] woken — checking for newly linked account")
                except asyncio.TimeoutError:
                    pass
                continue

            access_token, ctid, _uid, _is_live = creds
            # Connect to the host matching the account type; if auth fails there,
            # fall back to the other host (handles stale/missing isLive metadata).
            _primary_host = _HOST_LIVE if _is_live else _HOST_DEMO
            _hosts = [_primary_host, _HOST_DEMO if _primary_host == _HOST_LIVE else _HOST_LIVE]
            logger.info(
                f"[CTraderFeed] connecting with account {ctid} "
                f"({'live' if _is_live else 'demo'} → {_primary_host})…"
            )

            authed = False
            for _hi, _host in enumerate(_hosts):
                if writer:
                    await _aclose_writer(writer)
                reader, writer = await _open_conn(_host)
                if not await _app_auth(reader, writer):
                    raise ConnectionError("app auth failed")
                if await _account_auth(reader, writer, access_token, ctid):
                    authed = True
                    if _hi > 0:
                        logger.info(f"[CTraderFeed] authed on fallback host {_host}")
                    break
                # First host failed — try a token refresh once before the fallback.
                if _hi == 0:
                    try:
                        from app.services.ctrader_client import refresh_user_ctrader_token
                        new_at = await refresh_user_ctrader_token(_uid)
                    except Exception as _re:
                        new_at = None
                        logger.warning(f"[CTraderFeed] token refresh error: {_re}")
                    if new_at:
                        access_token = new_at
                        logger.info("[CTraderFeed] token refreshed — retrying account auth")
                        if await _account_auth(reader, writer, access_token, ctid):
                            authed = True
                            break
                logger.warning(f"[CTraderFeed] account auth failed on {_host}")
            if not authed:
                raise ConnectionError("account auth failed on all hosts")

            name_to_id = await _resolve_symbols(reader, writer, ctid, _host)

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
            session_start = time.monotonic()
            fail_backoff = _RECONNECT_BACKOFF_MIN  # reset after a clean subscribe

            # cTrader drops any connection that doesn't send a client heartbeat
            # within ~30s (even while spot events stream IN). Send one every 10s.
            async def _heartbeat(w):
                try:
                    while True:
                        await asyncio.sleep(10)
                        await _send(w, ProtoHeartbeatEvent(), _PT_HEARTBEAT)
                except (asyncio.CancelledError, Exception):
                    # Cancelled on disconnect, or the writer is already closing —
                    # the read loop will surface the real disconnect; stay quiet.
                    return
            hb_task = asyncio.create_task(_heartbeat(writer))

            # Stream indefinitely — each SpotEvent updates the cache
            while True:
                msg = await _read_frame(reader, timeout=_READ_TIMEOUT)
                if not msg:
                    raise ConnectionError("stream read timeout / disconnect")

                # Any inbound frame (spot event OR server heartbeat) proves the
                # socket is genuinely alive — only then treat a later drop as a
                # benign broker recycle (fast reconnect) rather than a failure.
                healthy = True

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
            _last_exc = exc
        else:
            _last_exc = None
        finally:
            if hb_task:
                hb_task.cancel()
            if writer:
                await _aclose_writer(writer)

        # Decide the reconnect delay. A session that streamed fine for a while
        # then dropped is just cTrader recycling the socket — reconnect almost
        # immediately so the sub-second feed is effectively never down. Only a
        # connection that never became healthy (connect/auth/subscribe failing)
        # gets the growing backoff, so we don't hammer the broker.
        alive = time.monotonic() - session_start
        if healthy and alive >= _HEALTHY_SESSION_SECS:
            logger.info(
                f"[CTraderFeed] session dropped after {alive:.0f}s "
                f"({_last_exc}) — reconnecting in {_RECONNECT_FAST:.0f}s"
            )
            await asyncio.sleep(_RECONNECT_FAST)
        else:
            logger.warning(
                f"[CTraderFeed] disconnected ({_last_exc}) — retry in {fail_backoff:.0f}s"
            )
            await asyncio.sleep(fail_backoff)
            fail_backoff = min(fail_backoff * 1.5, _RECONNECT_BACKOFF_MAX)


# ── On-demand trendbar (kline) fetch ─────────────────────────────────────────

def _get_tb_lock() -> "asyncio.Lock":
    global _tb_lock
    if _tb_lock is None:
        _tb_lock = asyncio.Lock()
    return _tb_lock


async def _invalidate_tb_conn() -> None:
    """Close and drop the persistent trendbar connection (forces a reopen)."""
    global _tb_conn, _tb_conn_ctx
    conn = _tb_conn
    _tb_conn = None
    _tb_conn_ctx = None
    if conn is not None:
        try:
            await _aclose_writer(conn[1])
        except Exception:
            pass


def _drop_tb_conn_sync() -> None:
    """Synchronously abandon the persistent trendbar connection — cancellation-safe
    (no awaits, so it can run while a CancelledError is propagating). Used to make
    a cancelled/timed-out request connection-fatal: a late response left in the
    socket buffer must never be read as the answer to the NEXT request."""
    global _tb_conn, _tb_conn_ctx
    conn = _tb_conn
    _tb_conn = None
    _tb_conn_ctx = None
    if conn is not None:
        # Schedule a full close (close + wait_closed) so the SSL FD is actually
        # released — a bare .close() leaks the descriptor → "Too many open files".
        # Cancellation-safe: ensure_future does not await here. Falls back to a
        # bare close only if there is no running loop to schedule on.
        try:
            asyncio.ensure_future(_aclose_writer(conn[1]))
        except Exception:
            try:
                conn[1].close()
            except Exception:
                pass


async def _get_tb_conn(
    host: str, access_token: str, ctid: int
) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Return a live, app+account-authed connection for (host, ctid), reusing the
    cached one when it is fresh. Reopens when missing, idle too long, or bound to
    a different host/account. Raises on connect/auth failure."""
    global _tb_conn, _tb_conn_ctx, _tb_last_use
    now = time.monotonic()
    if (
        _tb_conn is not None
        and _tb_conn_ctx == (host, ctid)
        and (now - _tb_last_use) <= _TB_IDLE_MAX
    ):
        return _tb_conn

    await _invalidate_tb_conn()
    reader, writer = await _open_conn(host)
    try:
        if not await _app_auth(reader, writer):
            raise ConnectionError("trendbar app auth failed")
        if not await _account_auth(reader, writer, access_token, ctid):
            raise ConnectionError("trendbar account auth failed")
    except BaseException:
        # BaseException so a wait_for cancellation during auth still releases the
        # writer instead of leaking it. Schedule the FULL close (close +
        # wait_closed) on the loop rather than a bare .close() — a bare close does
        # not free the SSL FD and, during a dead-token window, this auth-failure
        # path runs on every reopen → "Too many open files". ensure_future does
        # not await, so it stays cancellation-safe; bare close only if no loop.
        try:
            asyncio.ensure_future(_aclose_writer(writer))
        except Exception:
            try:
                writer.close()
            except Exception:
                pass
        raise
    _tb_conn = (reader, writer)
    _tb_conn_ctx = (host, ctid)
    _tb_last_use = now
    return _tb_conn


async def _fetch_trendbars(
    symbol: str,
    timeframe: str,
    limit: int,
    access_token: str,
    ctid: int,
    host: str = _HOST_LIVE,
) -> List[List[float]]:
    """
    Fetch up to `limit` OHLC bars for `symbol` at `timeframe` from cTrader over a
    single persistent, lock-serialized connection (auth happens once on connect,
    NOT per request). Reuses the warm connection; on a stale/dead socket it
    reopens and retries once. Returns [] on failure so callers fall through to
    their next candle source.

    The lock guarantees only one request is in flight at a time, and any
    timeout/error drops the connection — so a late response can never be misread
    as the answer to the next request (no clientMsgId correlation needed).

    Returns rows in [[ts_ms, open, high, low, close, volume], ...] format.
    """
    global _tb_last_use

    period = _TF_TO_PERIOD.get(timeframe)
    if period is None:
        logger.debug(f"[CTraderFeed] unsupported timeframe {timeframe!r}")
        return []

    broker_name = _TRACKED.get(symbol.upper(), symbol.upper())

    async with _get_tb_lock():
        for attempt in (1, 2):
            try:
                reader, writer = await _get_tb_conn(host, access_token, ctid)

                # Symbol IDs are cached per (host, ctid) by _resolve_symbols and
                # already populated by the streaming connection, so in steady
                # state this is a no-op lookup (no extra round-trip), and it
                # never reuses a symbolId that maps to the wrong instrument.
                name_to_id = await _resolve_symbols(reader, writer, ctid, host)
                symbol_id = name_to_id.get(broker_name)
                if not symbol_id:
                    logger.debug(f"[CTraderFeed] symbol_id not found for {broker_name}")
                    return []

                now_ms = int(time.time() * 1000)
                # cTrader REQUIRES fromTimestamp (and toTimestamp). Window it to
                # cover `count` bars of this timeframe, ×3 to absorb weekends /
                # market-closed gaps so we still get `count` *trading* bars back.
                # cTrader caps the returned set at `count` (max 4096).
                _mins = _TF_MINUTES.get(timeframe, 15)
                _count = min(limit, 4096)
                _window_ms = _count * _mins * 60_000 * 3
                req = ProtoOAGetTrendbarsReq()
                req.ctidTraderAccountId = ctid
                req.symbolId = symbol_id
                req.period = period
                req.count = _count
                req.fromTimestamp = now_ms - _window_ms
                req.toTimestamp = now_ms
                await _send(writer, req, _PT_TRENDBARS_REQ)
                _tb_last_use = time.monotonic()

                # Read frames until we get the trendbars response (skipping any
                # interleaved heartbeats). Kept under the caller's wait_for budget
                # so a warm round-trip completes before any forced cancellation.
                deadline = time.monotonic() + 6.0
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
                        _tb_last_use = time.monotonic()
                        return rows[-limit:]

                # No matching response — the socket is likely stale. Drop it and,
                # on the first attempt, reopen + retry once.
                await _invalidate_tb_conn()
                if attempt == 1:
                    continue
                return []

            except asyncio.CancelledError:
                # Caller's wait_for timed out. The socket may have a pending/late
                # response — abandon it (sync, no await while cancelling) so it is
                # never misread by the next request, then propagate the cancel.
                _drop_tb_conn_sync()
                raise
            except Exception as e:
                logger.debug(f"[CTraderFeed] trendbar fetch failed for {symbol}: {e}")
                await _invalidate_tb_conn()
                if attempt == 1:
                    continue
                return []

    return []


async def get_klines(
    symbol: str,
    asset_class: str,
    timeframe: str = "15m",
    limit: int = 200,
    user_id: Optional[int] = None,
) -> List[List[float]]:
    """
    Return up to `limit` OHLC bars from cTrader for the given symbol.
    Returns [] if the symbol is not tracked, no account is connected,
    or cTrader proto is unavailable (caller falls back to yfinance).
    """
    if not _PROTO_OK:
        return []
    sym_up = symbol.upper()
    try:
        from app.services.index_symbols import normalize_index_symbol
        sym_up = normalize_index_symbol(sym_up)
    except Exception:
        pass
    if sym_up not in _TRACKED:
        return []

    cache_key = (sym_up, timeframe, limit)
    cached = _kline_cache.get(cache_key)
    if cached and (time.monotonic() - cached[1]) < _KLINE_TTL:
        return cached[0]

    creds = await _get_connected_account(user_id=user_id)
    if not creds:
        return []

    access_token, ctid, _uid, _is_live = creds
    _primary_host = _HOST_LIVE if _is_live else _HOST_DEMO
    rows = await _fetch_trendbars(sym_up, timeframe, limit, access_token, ctid, _primary_host)
    if not rows:
        # Possibly an expired token — refresh once and retry on the primary host.
        try:
            from app.services.ctrader_client import refresh_user_ctrader_token
            new_at = await refresh_user_ctrader_token(_uid)
        except Exception:
            new_at = None
        if new_at:
            access_token = new_at
            rows = await _fetch_trendbars(sym_up, timeframe, limit, access_token, ctid, _primary_host)
    if not rows:
        # Still nothing — try the other host (wrong/stale isLive metadata).
        _fallback_host = _HOST_DEMO if _primary_host == _HOST_LIVE else _HOST_LIVE
        rows = await _fetch_trendbars(sym_up, timeframe, limit, access_token, ctid, _fallback_host)
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
    """True when the streaming connection is subscribed to spot prices."""
    return _feed_live


def cached_symbols() -> List[str]:
    """Return list of canonical symbols with a fresh cached price."""
    now = time.monotonic()
    return [
        sym for sym, (_, _, ts) in _spot_cache.items()
        if now - ts <= _SPOT_TTL
    ]


def _get_wake_event() -> asyncio.Event:
    global _wake_event
    if _wake_event is None:
        _wake_event = asyncio.Event()
    return _wake_event


def notify_account_linked() -> None:
    """Wake the feed loop immediately after OAuth links a cTrader account."""
    start()
    try:
        _get_wake_event().set()
    except Exception:
        pass


def feed_status() -> Dict[str, object]:
    """Diagnostics for /api/ctrader/feed-status and Live Forex UI."""
    now = time.monotonic()
    fresh = cached_symbols()
    all_cached = list(_spot_cache.keys())
    last_tick_age = None
    if _spot_cache:
        newest = max(ts for _, _, ts in _spot_cache.values())
        last_tick_age = round(max(0.0, now - newest), 1)
    forex_open = False
    try:
        from datetime import datetime
        from app.services.asset_classes import is_market_open
        forex_open = bool(is_market_open("forex", datetime.utcnow()))
    except Exception:
        pass
    note = None
    if _feed_live and not fresh and not forex_open:
        note = "Forex market closed — live ticks resume when sessions open (Sun ~10pm UTC)."
    elif _feed_live and not fresh:
        note = "Subscribed to cTrader — waiting for first tick (can take ~30s after connect)."
    elif not _feed_live and all_cached:
        note = "Feed reconnecting — last prices may be stale."
    return {
        "subscribed":       _feed_live,
        "live":             _feed_live,
        "symbol_count":     len(fresh),
        "cached_symbols":   fresh[:30],
        "symbols_seen":     len(all_cached),
        "last_tick_age_s":  last_tick_age,
        "forex_market_open": forex_open,
        "note":             note,
    }


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
        _get_wake_event()
        loop = asyncio.get_event_loop()
        _feed_task = loop.create_task(_feed_loop())
        logger.info("[CTraderFeed] background streaming task started")
    except Exception as e:
        logger.error(f"[CTraderFeed] failed to start task: {e}")


def stop() -> None:
    """Cancel the background streaming task.

    Called when the executor advisory lock is LOST so a former lock-holder
    doesn't keep a second cTrader connection open alongside the new holder's
    feed (which would re-trigger the broker's duplicate-session kicks). Safe to
    call when the feed isn't running.
    """
    global _feed_task, _feed_live
    if _feed_task and not _feed_task.done():
        _feed_task.cancel()
        logger.info("[CTraderFeed] background streaming task cancelled (lock lost)")
    _feed_task = None
    _feed_live = False
