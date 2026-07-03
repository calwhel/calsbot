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
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

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
_PT_EXECUTION_EVENT = 2126
_PT_TRENDBARS_REQ = 2137
_PT_TRENDBARS_RES = 2138
_PT_HEARTBEAT     = 51    # ProtoHeartbeatEvent — client MUST send ~every 10s or
                         # cTrader drops the socket (the "stream read timeout" churn)
_PT_ERROR_RES     = 2142  # ProtoOAErrorRes — account auth / request failures

# OAuth refresh is owned by ctrader_token_scheduler — feed is a read-only consumer.
_last_auth_error: Optional[str] = None
_auth_backoff_until: float = 0.0  # monotonic — pause reconnect after dead OAuth
_auth_terminal_alert_sent: bool = False  # one Telegram alert per terminal OAuth episode

# Terminal auth errors — user must disconnect + re-link cTrader in the portal.
_AUTH_TERMINAL_MARKERS = (
    "ACCESS_DENIED",
    "CH_ACCESS_TOKEN_INVALID",
    "INVALID_ACCESS_TOKEN",
    "INVALID_REFRESH_TOKEN",
)
_AUTH_TERMINAL_BACKOFF_S = float(
    os.environ.get("CTRADER_AUTH_TERMINAL_BACKOFF_S", "300")
)

# ── Reconnect tuning ─────────────────────────────────────────────────────────
# cTrader periodically recycles long-lived spot sessions (especially when a
# second authed socket — the on-demand trendbar fetch — shares the same account
# id). We can't stop the broker recycling, so we make the reconnect SEAMLESS:
# after a session that streamed fine for a while drops, reconnect almost
# immediately so the sub-second feed blackout is negligible. A growing backoff
# is reserved ONLY for genuine connect/auth/subscribe failures so we never
# hammer the broker while still being effectively always-on.
_READ_TIMEOUT          = float(
    os.environ.get("CTRADER_READ_TIMEOUT_S", "35")
)  # detect dead sessions quickly; client heartbeat should keep reads flowing
_HEARTBEAT_INTERVAL_S  = float(
    os.environ.get("CTRADER_CLIENT_HEARTBEAT_S", "8")
)  # cTrader requires client ProtoHeartbeatEvent ~every 10s
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
    "XAGUSD": "XAGUSD",
    # Indices — canonical cTrader names (FP Markets contract names)
    "NAS100": "US100", "NDX": "US100", "US100": "US100",
    "SPX500": "US500", "SPX": "US500",  "US500": "US500",
    "US30":   "US30",  "DJI": "US30",
    "GER40":  "GER40", "DAX": "GER40",
    "UK100":  "UK100", "FTSE": "UK100",
}

# Broker symbol → canonical name (reverse map, built at start)
_BROKER_TO_CANONICAL: Dict[str, str] = {v: k for k, v in _TRACKED.items()}

_SPOT_TTL = float(os.environ.get("REALTIME_SPOT_MAX_AGE_FOREX_S", "5"))
_CTRADER_ACCOUNT_LOOKUP_TIMEOUT_S = float(
    os.environ.get("CTRADER_ACCOUNT_LOOKUP_TIMEOUT_S", "3.0")
)
_CTRADER_ACCOUNT_LOOKUP_CACHE_TTL_S = float(
    os.environ.get("CTRADER_ACCOUNT_LOOKUP_CACHE_TTL_S", "2.0")
)

# ── Module-level shared state ─────────────────────────────────────────────────
# symbol → (bid, ask, monotonic_ts)
_spot_cache: Dict[str, Tuple[float, float, float]] = {}
# (symbol, timeframe, limit) → (rows, monotonic_ts)
_kline_cache: Dict[Tuple[str, str, int], Tuple[List[List[float]], float]] = {}
_KLINE_TTL = 60.0
# canonical symbol → monotonic ts of last successful trendbar fetch
_last_kline_update: Dict[str, float] = {}

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
_tb_hb_task: Optional[asyncio.Task] = None
_TB_IDLE_MAX = 25.0  # cTrader drops idle conns ~30s → proactively reopen past this

_feed_live: bool = False
_last_spot_tick_mono: float = 0.0  # monotonic — for price-gap logging on reconnect
# Trendbars on the LIVE spot socket (no second authed session on same ctid).
_stream_reader: Optional[asyncio.StreamReader] = None
_stream_writer: Optional[asyncio.StreamWriter] = None
_stream_ctid: int = 0
_stream_io_lock: Optional[asyncio.Lock] = None
_pending_trendbar: Optional[Dict[str, Any]] = None
_trendbar_block_reason: Optional[str] = None
_trendbar_block_until: float = 0.0
_trendbar_last_ok_log: Dict[str, float] = {}
_trendbar_last_api: Dict[Tuple[str, str], float] = {}
_TRENDBAR_MIN_INTERVAL_S = float(os.environ.get("CTRADER_TRENDBAR_MIN_INTERVAL_S", "60"))
_tick_history: Dict[str, deque] = {}
_TICK_HISTORY_MAX = 8000
# Rate-limit stale-rebuild watchdog logs (safety net only — ticks update bars inline).
_stale_rebuild_log_at: Dict[Tuple[str, str], float] = {}
_STALE_REBUILD_LOG_INTERVAL_S = 300.0
# Active spot-stream session — avoids DB round-trips for trendbars while LIVE.
_stream_creds: Optional[Tuple[str, int, int, str]] = None  # token, ctid, uid, host
_feed_task: Optional[asyncio.Task] = None
_feed_starting: bool = False
_feed_launch_lock = threading.Lock()
_feed_singleton_uid: Optional[int] = None
_wake_event: Optional[asyncio.Event] = None
# Cached symbol-ID map from the last successful connection.
# Symbol IDs differ per host/account, so the cache is scoped to the
# (host, ctid) it was resolved from — switching host/account forces a
# re-resolve to avoid mapping a symbolId to the WRONG instrument.
_symbol_id_map: Dict[str, int] = {}   # broker_name → symbolId
_id_to_canonical: Dict[int, str] = {} # symbolId → canonical name
_symbol_map_ctx: Optional[Tuple[str, int]] = None  # (host, ctid) cache belongs to
_connected_accounts_cache: Dict[int, Tuple[List[Tuple[str, int, int, bool]], float]] = {}

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
        body = await asyncio.wait_for(
            reader.readexactly(length),
            timeout=max(1.0, timeout),
        )
        return _decode(body)
    except (asyncio.TimeoutError, asyncio.IncompleteReadError):
        return None
    except Exception as e:
        logger.debug(f"[CTraderFeed] _read_frame error: {e}")
        return None


async def _send(writer: asyncio.StreamWriter, proto_req, payload_type: int) -> None:
    writer.write(_encode(proto_req, payload_type))
    await writer.drain()


def _start_heartbeat_thread(
    loop: asyncio.AbstractEventLoop,
    writer_holder: Dict[str, asyncio.StreamWriter],
    stop_event: threading.Event,
) -> threading.Thread:
    """Send ProtoHeartbeatEvent from a daemon thread — immune to executor scan blocking the loop."""

    def _run() -> None:
        while not stop_event.is_set():
            if stop_event.wait(_HEARTBEAT_INTERVAL_S):
                break
            writer = writer_holder.get("writer")
            if writer is None or not _PROTO_OK:
                continue
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    _send(writer, ProtoHeartbeatEvent(), _PT_HEARTBEAT),
                    loop,
                )
                fut.result(timeout=8.0)
            except Exception:
                pass

    t = threading.Thread(target=_run, daemon=True, name="ctrader-feed-hb")
    t.start()
    return t


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


def _ctrader_app_credentials() -> Tuple[str, str]:
    """Read Open API app id/secret (env is authoritative on Railway)."""
    cid = (os.environ.get("CTRADER_CLIENT_ID") or "").strip()
    secret = (os.environ.get("CTRADER_CLIENT_SECRET") or "").strip()
    if not cid or not secret:
        try:
            from app.services.ctrader_client import CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET
            cid = cid or (CTRADER_CLIENT_ID or "").strip()
            secret = secret or (CTRADER_CLIENT_SECRET or "").strip()
        except Exception:
            pass
    return cid, secret


async def _app_auth(reader, writer) -> bool:
    cid, secret = _ctrader_app_credentials()
    if not cid or not secret:
        logger.warning("[CTraderFeed] CTRADER_CLIENT_ID/SECRET missing — app auth skipped")
        return False
    req = ProtoOAApplicationAuthReq()
    req.clientId = cid
    req.clientSecret = secret
    await _send(writer, req, _PT_APP_AUTH_REQ)
    msg = await _read_frame(reader, timeout=15.0)
    if msg is not None and msg.payloadType == _PT_APP_AUTH_RES:
        return True
    if msg is None:
        logger.warning("[CTraderFeed] app auth: no response (timeout)")
    else:
        logger.warning(
            f"[CTraderFeed] app auth rejected (payloadType={msg.payloadType}, "
            f"expected={_PT_APP_AUTH_RES}) — check CTRADER_CLIENT_ID/SECRET"
        )
    return False


async def _account_auth(reader, writer, access_token: str, ctid: int) -> bool:
    """Account auth — logs ProtoOAErrorRes instead of failing silently."""
    global _last_auth_error
    if not _PROTO_OK:
        _last_auth_error = "ctrader_open_api unavailable"
        return False
    req = ProtoOAAccountAuthReq()
    req.ctidTraderAccountId = ctid
    req.accessToken = (access_token or "").strip()
    if not req.accessToken:
        _last_auth_error = "empty access token"
        return False
    await _send(writer, req, _PT_ACCT_AUTH_REQ)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        msg = await _read_frame(reader, timeout=max(0.1, remaining))
        if not msg:
            break
        if msg.payloadType == _PT_ACCT_AUTH_RES:
            _last_auth_error = None
            return True
        if msg.payloadType == _PT_ERROR_RES:
            detail = f"ctid={ctid}"
            try:
                from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAErrorRes
                err = ProtoOAErrorRes()
                err.ParseFromString(msg.payload)
                detail = (
                    f"ctid={ctid} {err.errorCode}: "
                    f"{(err.description or '').strip() or 'no description'}"
                )
            except Exception:
                pass
            _last_auth_error = detail
            _note_terminal_auth(detail)
            logger.warning(f"[CTraderFeed] account auth rejected — {detail}")
            return False
    _last_auth_error = f"ctid={ctid} timeout waiting for auth response"
    logger.warning(f"[CTraderFeed] {_last_auth_error}")
    return False


def _is_terminal_auth_error(detail: Optional[str]) -> bool:
    if not detail:
        return False
    up = detail.upper()
    return any(marker in up for marker in _AUTH_TERMINAL_MARKERS)


def _note_terminal_auth(detail: Optional[str], *, user_id: Optional[int] = None) -> None:
    """Back off reconnect loops when OAuth is dead until the user re-links."""
    global _auth_backoff_until, _auth_terminal_alert_sent
    if not _is_terminal_auth_error(detail):
        return
    _auth_backoff_until = time.monotonic() + _AUTH_TERMINAL_BACKOFF_S
    logger.warning(
        "[CTraderFeed] terminal auth (%s) — pausing reconnect for "
        f"{_AUTH_TERMINAL_BACKOFF_S:.0f}s (user must re-link cTrader)",
        (detail or "")[:120],
    )
    if _auth_terminal_alert_sent:
        return
    _auth_terminal_alert_sent = True
    try:
        from app.services.telegram_dm import owner_chat_id, send_dm
        owner = owner_chat_id()
        if owner:
            asyncio.get_event_loop().create_task(
                send_dm(
                    owner,
                    "⚠️ <b>cTrader auth failed</b>\n\n"
                    "The live price feed cannot refresh its OAuth token. "
                    "Please re-link cTrader in the portal (Settings → cTrader).\n\n"
                    f"<code>{(detail or 'ACCESS_DENIED')[:100]}</code>",
                )
            )
    except Exception:
        pass


def _remote_feed_mode() -> bool:
    try:
        from app.ctrader_feed_lock import remote_feed_enabled
        return remote_feed_enabled()
    except Exception:
        return os.environ.get("CTRADER_REMOTE_FEED", "").lower() in (
            "1",
            "true",
            "yes",
        )


def _standalone_trendbar_fetch_allowed() -> bool:
    """
    Second authed TLS sockets on the same ctid recycle the spot stream.

    Only the feed-owning process (local spot stream or dedicated feed replica /
    standalone executor before remote-feed disable) may open trendbar sockets.
    Portal gunicorn workers must use Postgres ticks + metal cache + synthesis.
    """
    if _feed_live:
        return True
    try:
        from app.ctrader_feed_lock import feed_disabled_in_executor, is_feed_only_process

        if is_feed_only_process():
            return True
        if os.environ.get("EXECUTOR_STANDALONE", "").lower() in ("1", "true", "yes"):
            return not feed_disabled_in_executor()
    except Exception:
        pass
    return False


def _cached_klines_synthesized(
    sym_up: str,
    timeframe: str,
    limit: int,
) -> List[List[float]]:
    """Return in-process cache rolled forward from fresh spot when available."""
    cache_key = (sym_up, timeframe, limit)
    cached = _kline_cache.get(cache_key)
    if not cached:
        return []
    return _synthesize_klines_on_return(cached[0], sym_up, timeframe, limit)


def _shared_klines_from_postgres(
    sym_up: str,
    timeframe: str,
    limit: int,
) -> List[List[float]]:
    """Cross-worker snapshot written by the executor feed (no trendbar socket)."""
    try:
        from app.services.kline_snapshot_store import get_klines as _pg_klines

        rows = _pg_klines(sym_up, timeframe, limit, source="ctrader")
        if rows:
            return _synthesize_klines_on_return(rows, sym_up, timeframe, limit)
    except Exception as exc:
        logger.debug("[CTraderFeed] postgres kline snapshot miss %s %s: %s", sym_up, timeframe, exc)
    return []


def _persist_klines_for_peers(sym_up: str, timeframe: str, rows: List[List[float]]) -> None:
    if not rows:
        return
    try:
        from app.services.kline_snapshot_store import upsert_klines

        upsert_klines(sym_up, timeframe, rows, source="ctrader")
    except Exception as exc:
        logger.debug("[CTraderFeed] kline snapshot persist skipped %s %s: %s", sym_up, timeframe, exc)


def trendbar_fetch_blocked_reason() -> Optional[str]:
    """Why trendbar/kline API is unavailable (None = allowed)."""
    if _is_terminal_auth_error(_last_auth_error):
        return "oauth expired"
    now = time.monotonic()
    if now < _trendbar_block_until:
        rem = max(0.0, _trendbar_block_until - now)
        base = _trendbar_block_reason or "backoff"
        return f"{base}, retry in {rem:.0f}s"
    if _feed_live and _stream_writer is None:
        return "stream session not registered"
    return None


def _trendbar_fetch_allowed() -> bool:
    """
    True when trendbars can be fetched without opening a competing second socket.

    While the spot feed is LIVE, trendbars use the same TLS session (multiplexed
    reads in _feed_loop). A separate authed socket on the same ctid recycles the
    spot stream — only allowed when _feed_live is False (remote/cold executor).
    """
    return trendbar_fetch_blocked_reason() is None


def _note_trendbar_block(reason: str, retry_s: float = 60.0) -> None:
    global _trendbar_block_reason, _trendbar_block_until
    now = time.monotonic()
    if reason != _trendbar_block_reason or now >= _trendbar_block_until:
        logger.warning(
            "[CTraderFeed] trendbars blocked: %s, retry in %.0fs",
            reason,
            retry_s,
        )
    _trendbar_block_reason = reason
    _trendbar_block_until = now + retry_s


def _note_trendbar_ok(symbol: str) -> None:
    global _trendbar_block_reason, _trendbar_block_until
    _trendbar_block_reason = None
    _trendbar_block_until = 0.0
    sym = symbol.upper()
    now = time.monotonic()
    if now - _trendbar_last_ok_log.get(sym, 0.0) >= 300.0:
        _trendbar_last_ok_log[sym] = now
        logger.info("[CTraderFeed] trendbars OK for %s", sym)


def _register_live_stream(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    ctid: int,
) -> None:
    global _stream_reader, _stream_writer, _stream_ctid, _stream_io_lock
    _stream_reader = reader
    _stream_writer = writer
    _stream_ctid = ctid
    if _stream_io_lock is None:
        _stream_io_lock = asyncio.Lock()


def _unregister_live_stream() -> None:
    global _stream_reader, _stream_writer, _stream_ctid, _pending_trendbar
    _stream_reader = None
    _stream_writer = None
    _stream_ctid = 0
    if _pending_trendbar:
        fut = _pending_trendbar.get("future")
        if fut and not fut.done():
            fut.set_result([])
    _pending_trendbar = None


def _get_stream_io_lock() -> asyncio.Lock:
    global _stream_io_lock
    if _stream_io_lock is None:
        _stream_io_lock = asyncio.Lock()
    return _stream_io_lock


def _record_tick(sym: str, mid: float, ts_ms: Optional[int] = None) -> None:
    if ts_ms is None:
        ts_ms = int(time.time() * 1000)
    hist = _tick_history.setdefault(sym.upper(), deque(maxlen=_TICK_HISTORY_MAX))
    hist.append((ts_ms, mid))


def _bar_open_ts_ms(ts_ms: int, timeframe: str) -> int:
    """Bar open timestamp (ms) for the wall-clock instant ts_ms."""
    step_ms = _TF_MINUTES.get(timeframe, 15) * 60_000
    return (int(ts_ms) // step_ms) * step_ms


def _apply_tick_to_bar_rows(
    rows: List[List[float]],
    mid: float,
    ts_ms: int,
    timeframe: str,
    limit: int,
) -> List[List[float]]:
    """Update forming bar OHLC from one tick; roll to a new bar at timeframe boundary."""
    if not rows or mid <= 0:
        return rows
    step_ms = _TF_MINUTES.get(timeframe, 15) * 60_000
    bar_ts = _bar_open_ts_ms(ts_ms, timeframe)
    out: List[List[float]] = [list(r) for r in rows]
    last = out[-1]
    last_ts = int(last[0])
    if last_ts == bar_ts:
        last[4] = mid
        last[2] = max(float(last[2]), mid)
        last[3] = min(float(last[3]), mid)
    elif last_ts < bar_ts:
        prev_close = float(last[4])
        t = last_ts + step_ms
        while t < bar_ts:
            out.append([t, prev_close, prev_close, prev_close, prev_close, 0.0])
            t += step_ms
        out.append([bar_ts, mid, mid, mid, mid, 0.0])
    return out[-limit:]


def _update_kline_cache_on_tick(sym: str, mid: float, ts_ms: Optional[int] = None) -> None:
    """Inline tick→bar builder: refresh every cached timeframe for this symbol."""
    if mid <= 0:
        return
    sym_up = sym.upper()
    if ts_ms is None:
        ts_ms = int(time.time() * 1000)
    now_mono = time.monotonic()
    updated = False
    for key in list(_kline_cache.keys()):
        if key[0] != sym_up:
            continue
        tf, lim = key[1], key[2]
        rows, _ = _kline_cache[key]
        if not rows:
            continue
        new_rows = _apply_tick_to_bar_rows(rows, mid, ts_ms, tf, lim)
        _kline_cache[key] = (new_rows, now_mono)
        updated = True
    if updated:
        _last_kline_update[sym_up] = now_mono


def _apply_live_tick_to_rows(
    rows: List[List[float]],
    symbol: str,
    timeframe: str,
    limit: int,
) -> List[List[float]]:
    """Refresh forming bar OHLC from the live spot mid while API fetch is rate-limited."""
    if not rows:
        return rows
    live = get_price(symbol.upper())
    if not live or live <= 0:
        return rows
    return _apply_tick_to_bar_rows(
        rows, live, int(time.time() * 1000), timeframe, limit,
    )


def _synthesize_klines_on_return(
    rows: List[List[float]],
    sym_up: str,
    timeframe: str,
    limit: int,
    *,
    log_remote: bool = True,
) -> List[List[float]]:
    """
    Apply live spot to the forming bar when a fresh cTrader mid is available.

    Uses get_price() (local spot cache or Postgres tick store) so portal gunicorn
    workers without _feed_live still roll 5m/15m bars forward. Idempotent on feed
    workers that already tick-update the cache — same mid re-applied to the forming bar.
    """
    if not rows or not ctrader_spot_ready(sym_up):
        return rows
    before_age = _newest_bar_age_s(rows)
    out = _apply_live_tick_to_rows(rows, sym_up, timeframe, limit)
    if not _feed_live and log_remote:
        after_age = _newest_bar_age_s(out)
        logger.info(
            "[CTraderFeed] live tick synthesis %s %s remote_worker "
            "bar_age=%.0fs→%.0fs bars=%d",
            sym_up,
            timeframe,
            before_age if before_age != float("inf") else -1.0,
            after_age if after_age != float("inf") else -1.0,
            len(out),
        )
    return out


def apply_live_spot_to_klines(
    rows: List[List[float]],
    symbol: str,
    timeframe: str,
    limit: int,
    *,
    log_remote: bool = True,
) -> List[List[float]]:
    """Public wrapper — roll forming bar from live spot when market is live."""
    return _synthesize_klines_on_return(
        rows, symbol.upper(), timeframe, limit, log_remote=log_remote,
    )


def _log_stale_kline_rebuild(sym: str, detail: str, tf: str) -> None:
    """Safety-net stale rebuild log — at most once per symbol/timeframe per 5 min."""
    key = (sym.upper(), tf)
    now = time.monotonic()
    if now - _stale_rebuild_log_at.get(key, 0.0) < _STALE_REBUILD_LOG_INTERVAL_S:
        logger.debug(
            "[CTraderFeed] kline builder stale for %s — rebuilt (%s, tf=%s)",
            sym,
            detail,
            tf,
        )
        return
    _stale_rebuild_log_at[key] = now
    logger.warning(
        "[CTraderFeed] kline builder stale for %s — rebuilt (%s, tf=%s)",
        sym,
        detail,
        tf,
    )


def _bars_from_trendbar_res(res: "ProtoOAGetTrendbarsRes", limit: int) -> List[List[float]]:
    rows: List[List[float]] = []
    for bar in res.trendbar:
        low = bar.low / 100_000.0
        open_ = (bar.low + bar.deltaOpen) / 100_000.0
        high = (bar.low + bar.deltaHigh) / 100_000.0
        close = (bar.low + bar.deltaClose) / 100_000.0
        ts_ms = bar.utcTimestampInMinutes * 60 * 1000
        vol = float(bar.volume)
        rows.append([ts_ms, open_, high, low, close, vol])
    return rows[-limit:]


def _build_trendbar_req(
    ctid: int,
    symbol_id: int,
    timeframe: str,
    limit: int,
) -> "ProtoOAGetTrendbarsReq":
    period = _TF_TO_PERIOD.get(timeframe)
    if period is None:
        raise ValueError(f"unsupported timeframe {timeframe!r}")
    now_ms = int(time.time() * 1000)
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
    return req


def _deliver_trendbar_response(msg) -> None:
    global _pending_trendbar
    if not _pending_trendbar:
        return
    fut = _pending_trendbar.get("future")
    lim = int(_pending_trendbar.get("limit") or 200)
    if fut is None or fut.done():
        return
    try:
        res = ProtoOAGetTrendbarsRes()
        res.ParseFromString(msg.payload)
        rows = _bars_from_trendbar_res(res, lim)
        fut.set_result(rows)
    except Exception as exc:
        fut.set_exception(exc)


async def _read_trendbars_inline(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    ctid: int,
    symbol: str,
    timeframe: str,
    limit: int,
) -> List[List[float]]:
    """Fetch trendbars on an open stream before spot subscribe (no multiplex)."""
    broker_name = _TRACKED.get(symbol.upper(), symbol.upper())
    symbol_id = _symbol_id_map.get(broker_name)
    if not symbol_id:
        return []
    try:
        req = _build_trendbar_req(ctid, symbol_id, timeframe, limit)
        await _send(writer, req, _PT_TRENDBARS_REQ)
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            msg = await _read_frame(reader, timeout=deadline - time.monotonic())
            if not msg:
                break
            if msg.payloadType == _PT_TRENDBARS_RES:
                res = ProtoOAGetTrendbarsRes()
                res.ParseFromString(msg.payload)
                return _bars_from_trendbar_res(res, limit)
    except Exception as exc:
        logger.debug("[CTraderFeed] inline trendbar %s %s: %s", symbol, timeframe, exc)
    return []


async def _prefetch_key_trendbars(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    ctid: int,
) -> None:
    """Seed kline cache at connect — one fetch per key metal/timeframe."""
    for sym in ("XAUUSD", "XAGUSD"):
        for tf in ("1m", "5m", "15m", "1h"):
            lim = 60 if tf == "1m" else 80
            rows = await _read_trendbars_inline(reader, writer, ctid, sym, tf, lim)
            if rows:
                now = time.monotonic()
                _kline_cache[(sym, tf, lim)] = (rows, now)
                _last_kline_update[sym] = now
                _trendbar_last_api[(sym, tf)] = now
                _note_trendbar_ok(sym)
                _persist_klines_for_peers(sym, tf, rows)
                logger.info(
                    "[CTraderFeed] prefetch kline snapshot %s %s bars=%d (postgres peers)",
                    sym,
                    tf,
                    len(rows),
                )


async def _fetch_trendbars_on_live_stream(
    symbol: str,
    timeframe: str,
    limit: int,
) -> List[List[float]]:
    """Trendbar fetch multiplexed on the live spot TLS session."""
    sym_up = symbol.upper()
    if not _stream_writer or not _stream_reader or not _stream_ctid:
        return []
    broker_name = _TRACKED.get(sym_up, sym_up)
    symbol_id = _symbol_id_map.get(broker_name)
    if not symbol_id:
        _note_trendbar_block(f"symbol_id missing for {broker_name}", 60.0)
        return []

    lock = _get_stream_io_lock()
    async with lock:
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        global _pending_trendbar
        _pending_trendbar = {
            "future": fut,
            "symbol": sym_up,
            "limit": limit,
        }
        try:
            req = _build_trendbar_req(_stream_ctid, symbol_id, timeframe, limit)
            await _send(_stream_writer, req, _PT_TRENDBARS_REQ)
            rows = await asyncio.wait_for(fut, timeout=12.0)
            if rows:
                _note_trendbar_ok(sym_up)
                _trendbar_last_api[(sym_up, timeframe)] = time.monotonic()
            else:
                _note_trendbar_block(f"empty response {sym_up} {timeframe}", 60.0)
            return rows
        except asyncio.TimeoutError:
            _note_trendbar_block(f"stream timeout {sym_up} {timeframe}", 60.0)
            return []
        except Exception as exc:
            _note_trendbar_block(f"{type(exc).__name__}: {exc}"[:80], 60.0)
            return []
        finally:
            _pending_trendbar = None


def _shared_ctrader_ticks_fresh(max_age_s: float = 30.0) -> bool:
    try:
        from app.services.spot_price_store import snapshot
        snap = snapshot(max_age_s=max_age_s)
        return bool((snap.get("by_source") or {}).get("ctrader"))
    except Exception:
        return False


def invalidate_stream_creds(user_id: Optional[int] = None) -> None:
    """Drop cached feed credentials so the next cycle reads fresh tokens from DB."""
    global _stream_creds
    if user_id is None:
        _stream_creds = None
        return
    if _stream_creds and _stream_creds[2] == int(user_id):
        _stream_creds = None


async def _load_persisted_access_token(user_id: int, access_token: str) -> str:
    """Read the latest persisted OAuth access token — never initiates refresh."""
    from app.services.ctrader_client import _latest_ctrader_access_token

    fresh = _latest_ctrader_access_token(user_id)
    if fresh:
        return fresh.strip()
    return (access_token or "").strip()


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

def _is_live_for_ctid(prefs, ctid: int) -> bool:
    from app.services.ctrader_client import _account_is_live
    val = _account_is_live(prefs, ctid)
    return True if val is None else bool(val)


def _preferred_feed_user_id() -> Optional[int]:
    for env_name in (
        "CTRADER_FEED_PREFERRED_USER_ID",
        "GOLD_AI_USER_ID",
        "GEMINI_GOLD_USER_ID",
    ):
        raw = (os.environ.get(env_name) or "").strip()
        if raw.isdigit():
            return int(raw)
    return None


def _dedupe_accounts_by_ctid(
    accounts: List[Tuple[str, int, int, bool]],
) -> List[Tuple[str, int, int, bool]]:
    """Keep one owner row per ctid (preferred trader uid wins, else first row)."""
    preferred = _preferred_feed_user_id()
    if preferred is not None:
        accounts = sorted(
            accounts,
            key=lambda row: (0 if row[2] == preferred else 1, -int(row[2])),
        )
    deduped: List[Tuple[str, int, int, bool]] = []
    first_owner: Dict[int, int] = {}
    dropped: Dict[int, List[int]] = {}
    for at, ctid, uid, is_live in accounts:
        if ctid in first_owner:
            dropped.setdefault(ctid, []).append(uid)
            continue
        first_owner[ctid] = uid
        deduped.append((at, ctid, uid, is_live))
    for ctid, losers in dropped.items():
        logger.warning(
            "[CTraderFeed] duplicate ctid owner rows detected ctid=%s kept_uid=%s dropped_uids=%s",
            ctid,
            first_owner.get(ctid),
            ",".join(str(x) for x in losers),
        )
    return deduped


def _prefs_rows_to_accounts(rows) -> List[Tuple[str, int, int, bool]]:
    out: List[Tuple[str, int, int, bool]] = []
    for prefs in rows:
        at = (prefs.ctrader_access_token or "").strip()
        aid = (prefs.ctrader_account_id or "").strip()
        if not at or not aid:
            continue
        try:
            ctid = int(aid)
        except (TypeError, ValueError):
            continue
        out.append((at, ctid, int(prefs.user_id), _is_live_for_ctid(prefs, ctid)))
    return _dedupe_accounts_by_ctid(out)


def probe_linked_accounts_sync() -> List[Tuple[str, int, int, bool]]:
    """Synchronous DB read for startup logging — latest persisted OAuth tokens."""
    try:
        from app.database import SessionLocal
        from app.models import UserPreference

        db = SessionLocal()
        try:
            rows = (
                db.query(UserPreference)
                .filter(
                    UserPreference.ctrader_access_token.isnot(None),
                    UserPreference.ctrader_account_id.isnot(None),
                )
                .order_by(
                    UserPreference.forex_approved.desc(),
                    UserPreference.user_id.desc(),
                )
                .all()
            )
            return _prefs_rows_to_accounts(rows)
        finally:
            db.close()
    except Exception as exc:
        logger.warning("[CTraderFeed] DB account probe failed: %s", exc)
    return []


async def _list_connected_accounts(
    user_id: Optional[int] = None,
) -> List[Tuple[str, int, int, bool]]:
    """All linked cTrader accounts — forex-approved users first."""
    def _cache_key(uid: Optional[int]) -> int:
        return int(uid) if uid is not None else 0

    def _cache_get(uid: Optional[int], *, allow_stale: bool = False) -> Optional[List[Tuple[str, int, int, bool]]]:
        item = _connected_accounts_cache.get(_cache_key(uid))
        if not item:
            return None
        rows, ts_mono = item
        if allow_stale:
            return list(rows)
        age_s = max(0.0, time.monotonic() - ts_mono)
        if age_s > max(0.1, _CTRADER_ACCOUNT_LOOKUP_CACHE_TTL_S):
            return None
        return list(rows)

    def _cache_put(uid: Optional[int], rows: List[Tuple[str, int, int, bool]]) -> None:
        _connected_accounts_cache[_cache_key(uid)] = (list(rows), time.monotonic())

    def _list_connected_accounts_sync(uid: Optional[int]) -> List[Tuple[str, int, int, bool]]:
        out_sync: List[Tuple[str, int, int, bool]] = []
        from app.database import SessionLocal
        from app.models import UserPreference

        db = SessionLocal()
        try:
            q = db.query(UserPreference).filter(
                UserPreference.ctrader_access_token.isnot(None),
                UserPreference.ctrader_account_id.isnot(None),
            )
            if uid is not None:
                q = q.filter(UserPreference.user_id == int(uid))
            rows = (
                q.order_by(
                    UserPreference.forex_approved.desc(),
                    UserPreference.user_id.desc(),
                )
                .all()
            )
            out_sync = _prefs_rows_to_accounts(rows)
        finally:
            db.close()
        return out_sync

    cached = _cache_get(user_id)
    if cached is not None:
        return cached

    timeout_s = max(0.5, float(_CTRADER_ACCOUNT_LOOKUP_TIMEOUT_S))
    try:
        out = await asyncio.wait_for(
            asyncio.to_thread(_list_connected_accounts_sync, user_id),
            timeout=timeout_s,
        )
        _cache_put(user_id, out)
        return out
    except asyncio.TimeoutError:
        stale = _cache_get(user_id, allow_stale=True)
        if stale is not None:
            logger.warning(
                "[CTraderFeed] DB lookup timeout %.1fs (uid=%s) — using cached rows=%s",
                timeout_s,
                user_id,
                len(stale),
            )
            return stale
        logger.warning(
            "[CTraderFeed] DB lookup timeout %.1fs (uid=%s) — no cached accounts",
            timeout_s,
            user_id,
        )
        return []
    except Exception as e:
        stale = _cache_get(user_id, allow_stale=True)
        logger.warning("[CTraderFeed] DB lookup error: %s", e)
        if stale is not None:
            return stale
        return []


async def _get_connected_account(
    user_id: Optional[int] = None,
) -> Optional[Tuple[str, int, int, bool]]:
    """Return the best linked account (forex-approved first)."""
    accounts = await _list_connected_accounts(user_id=user_id)
    return accounts[0] if accounts else None


async def _authenticate_stream(
    access_token: str,
    ctid: int,
    user_id: int,
    is_live: bool,
) -> Optional[Tuple[asyncio.StreamReader, asyncio.StreamWriter, str, str]]:
    """
    Open a streaming connection with app+account auth.
    Returns (reader, writer, access_token, host) or None.
    """
    at = await _load_persisted_access_token(user_id, access_token)
    host = _HOST_LIVE if is_live else _HOST_DEMO
    reader = writer = None
    try:
        reader, writer = await _open_conn(host)
        if not await _app_auth(reader, writer):
            raise ConnectionError("app auth failed")
        authed = await _account_auth(reader, writer, at, ctid)
        if not authed:
            fresh = await _load_persisted_access_token(user_id, at)
            if fresh != at:
                invalidate_stream_creds(user_id)
                at = fresh
                authed = await _account_auth(reader, writer, at, ctid)
        if authed:
            return reader, writer, at, host
        logger.warning(
            f"[CTraderFeed] account auth failed uid={user_id} ctid={ctid} "
            f"host={host}"
            + (f" ({_last_auth_error})" if _last_auth_error else "")
        )
        _note_terminal_auth(_last_auth_error, user_id=user_id)
    except Exception as exc:
        logger.warning(
            f"[CTraderFeed] connect/auth error uid={user_id} ctid={ctid} "
            f"host={host}: {exc}"
        )
    if writer is not None:
        await _aclose_writer(writer)
    return None


# ── Background streaming task ─────────────────────────────────────────────────

async def _feed_loop() -> None:
    global _feed_live, _stream_creds, _last_spot_tick_mono
    fail_backoff = _RECONNECT_BACKOFF_MIN  # grows ONLY on connect/auth failures

    while True:
        reader = writer = None
        hb_stop: Optional[threading.Event] = None
        hb_thread: Optional[threading.Thread] = None
        healthy = False                # True once spots are flowing
        session_start = time.monotonic()
        gap_before_connect = (
            time.monotonic() - _last_spot_tick_mono
            if _last_spot_tick_mono > 0
            else 0.0
        )
        try:
            if time.monotonic() < _auth_backoff_until:
                wait_s = _auth_backoff_until - time.monotonic()
                logger.info(
                    "[CTraderFeed] auth backoff — retry in %.0fs (%s)",
                    wait_s,
                    (_last_auth_error or "terminal OAuth")[:80],
                )
                wake = _get_wake_event()
                try:
                    await asyncio.wait_for(wake.wait(), timeout=min(wait_s, 120.0))
                    wake.clear()
                    logger.info("[CTraderFeed] woken during auth backoff — retrying")
                except asyncio.TimeoutError:
                    continue

            _feed_live = False
            _stream_creds = None

            candidates = await _list_connected_accounts()
            if not candidates:
                logger.info("[CTraderFeed] no connected cTrader accounts — waiting up to 120s")
                wake = _get_wake_event()
                try:
                    await asyncio.wait_for(wake.wait(), timeout=120.0)
                    wake.clear()
                    logger.info("[CTraderFeed] woken — checking for newly linked account")
                except asyncio.TimeoutError:
                    pass
                continue

            reader = writer = None
            access_token = ""
            ctid = 0
            _uid = 0
            _host = _HOST_LIVE
            session = None
            for at, acct_ctid, uid, is_live in candidates:
                logger.info(
                    f"[CTraderFeed] connecting uid={uid} ctid={acct_ctid} "
                    f"({'live' if is_live else 'demo'})…"
                )
                session = await _authenticate_stream(at, acct_ctid, uid, is_live)
                if session:
                    reader, writer, access_token, _host = session
                    ctid, _uid = acct_ctid, uid
                    if uid != candidates[0][2]:
                        logger.info(
                            f"[CTraderFeed] authenticated via uid={uid} ctid={ctid} "
                            f"(primary candidate failed)"
                        )
                    break
            if not session or reader is None or writer is None:
                detail = _last_auth_error or "no valid session"
                _note_terminal_auth(detail, user_id=_uid or None)
                raise ConnectionError(f"account auth failed ({detail})")

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

            await _prefetch_key_trendbars(reader, writer, ctid)
            _register_live_stream(reader, writer, ctid)

            req = ProtoOASubscribeSpotsReq()
            req.ctidTraderAccountId = ctid
            req.symbolId.extend(sub_ids)
            await _send(writer, req, _PT_SUB_SPOTS_REQ)

            logger.info(
                f"[CTraderFeed] subscribed to {len(sub_ids)} symbols — LIVE"
            )
            _feed_live = True
            _stream_creds = (access_token, ctid, _uid, _host)
            await _on_spot_stream_connected()
            session_start = time.monotonic()
            fail_backoff = _RECONNECT_BACKOFF_MIN  # reset after a clean subscribe

            # Client heartbeat MUST run even when the asyncio loop is blocked by
            # long executor scans — use a dedicated thread (same pattern as the
            # advisory-lock keepalive).
            loop = asyncio.get_event_loop()
            writer_holder: Dict[str, asyncio.StreamWriter] = {"writer": writer}
            hb_stop = threading.Event()
            hb_thread = _start_heartbeat_thread(loop, writer_holder, hb_stop)

            # Stream indefinitely — each SpotEvent updates the cache
            while True:
                msg = await _read_frame(reader, timeout=_READ_TIMEOUT)
                if not msg:
                    raise ConnectionError("stream read timeout / disconnect")

                healthy = True

                if msg.payloadType == _PT_HEARTBEAT:
                    try:
                        await _send(writer, ProtoHeartbeatEvent(), _PT_HEARTBEAT)
                    except Exception:
                        pass
                    continue

                if msg.payloadType == _PT_TRENDBARS_RES:
                    _deliver_trendbar_response(msg)
                    continue

                if msg.payloadType == _PT_EXECUTION_EVENT:
                    try:
                        from app.services.ctrader_execution_events import (
                            schedule_execution_event,
                        )
                        schedule_execution_event(
                            msg.payload, ctid=ctid, user_id=_uid,
                        )
                    except Exception as _ee:
                        logger.debug("[CTraderFeed] execution event: %s", _ee)
                    continue

                if msg.payloadType == _PT_SPOT_EVENT:
                    ev = ProtoOASpotEvent()
                    ev.ParseFromString(msg.payload)
                    canonical = _id_to_canonical.get(ev.symbolId)
                    if canonical:
                        bid = ev.bid / 100_000.0 if ev.HasField("bid") else None
                        ask = ev.ask / 100_000.0 if ev.HasField("ask") else None
                        if bid and ask:
                            _last_spot_tick_mono = time.monotonic()
                            _spot_cache[canonical] = (bid, ask, _last_spot_tick_mono)
                            mid = round((bid + ask) / 2.0, 6)
                            _record_tick(canonical, mid)
                            _update_kline_cache_on_tick(canonical, mid)
                            try:
                                from app.services.spot_price_store import upsert_tick
                                upsert_tick(
                                    canonical, bid=bid, ask=ask, mid=mid, source="ctrader",
                                )
                            except Exception:
                                pass
                            try:
                                from app.services.forex_tick_manager import on_ctrader_tick
                                asyncio.create_task(
                                    on_ctrader_tick(canonical, mid),
                                    name=f"tick-manage-{canonical}",
                                )
                            except Exception:
                                pass

                # Heartbeats and subscription confirmations are silently skipped

        except Exception as exc:
            _feed_live = False
            _stream_creds = None
            _last_exc = exc
        else:
            _last_exc = None
        finally:
            _unregister_live_stream()
            if hb_stop is not None:
                hb_stop.set()
            if writer:
                await _aclose_writer(writer)

        alive = time.monotonic() - session_start
        if healthy and alive >= _HEALTHY_SESSION_SECS:
            _gap_note = (
                f", price gap {gap_before_connect:.0f}s"
                if gap_before_connect >= 1.0
                else ""
            )
            logger.info(
                f"[CTraderFeed] session dropped after {alive:.0f}s "
                f"({_last_exc}){_gap_note} — reconnecting in {_RECONNECT_FAST:.0f}s"
            )
            await asyncio.sleep(_RECONNECT_FAST)
        else:
            logger.warning(
                f"[CTraderFeed] disconnected ({_last_exc}) — retry in {fail_backoff:.0f}s"
            )
            await asyncio.sleep(fail_backoff)
            fail_backoff = min(fail_backoff * 1.5, _RECONNECT_BACKOFF_MAX)


# ── On-demand trendbar (kline) fetch ─────────────────────────────────────────

def _kline_stale_limit_s(timeframe: str) -> float:
    """Max age before klines are stale while live ticks are flowing."""
    return 2.0 * _TF_MINUTES.get(timeframe, 15) * 60.0


_KLINE_STALE_DRIFT_PCT = float(os.environ.get("CTRADER_KLINE_STALE_DRIFT_PCT", "0.75"))


def _live_ticks_flowing(max_age_s: float = 30.0) -> bool:
    return (time.monotonic() - _last_spot_tick_mono) <= max_age_s


def _newest_bar_age_s(rows: List[List[float]]) -> float:
    """Wall-clock age of the newest OHLC bar timestamp."""
    if not rows:
        return float("inf")
    try:
        newest_ts_ms = int(rows[-1][0])
        return max(0.0, time.time() - newest_ts_ms / 1000.0)
    except (IndexError, TypeError, ValueError):
        return float("inf")


def _kline_live_drift_pct(sym_up: str, rows: List[List[float]]) -> Optional[float]:
    """Percent drift between newest kline close and live cTrader mid."""
    if not rows:
        return None
    try:
        close = float(rows[-1][4])
    except (IndexError, TypeError, ValueError):
        return None
    if close <= 0:
        return None
    live = get_price(sym_up)
    if not live or live <= 0:
        return None
    return abs(live - close) / live * 100.0


def _kline_cache_is_stale(
    sym_up: str,
    rows: List[List[float]],
    timeframe: str,
    *,
    cache_mono_ts: float = 0.0,
) -> Tuple[bool, str]:
    """True when live ticks flow but cached klines are time- or price-stale."""
    if not _live_ticks_flowing():
        return False, ""
    stale_limit = _kline_stale_limit_s(timeframe)
    bar_age = _newest_bar_age_s(rows)
    last_up = _last_kline_update.get(sym_up, 0.0)
    if last_up:
        fetch_age: Optional[float] = time.monotonic() - last_up
    elif cache_mono_ts:
        fetch_age = time.monotonic() - cache_mono_ts
    else:
        fetch_age = None
    if bar_age > stale_limit:
        return True, f"bar_ts={bar_age:.0f}s"
    if fetch_age is not None and fetch_age > stale_limit:
        return True, f"fetch={fetch_age:.0f}s"
    drift = _kline_live_drift_pct(sym_up, rows)
    if drift is not None and drift > _KLINE_STALE_DRIFT_PCT:
        return True, f"drift={drift:.2f}%"
    return False, ""


async def restart_kline_builder(reason: str = "manual") -> None:
    """Invalidate trendbar session and drop kline caches (reconnect / lock churn)."""
    n = clear_kline_cache()
    try:
        from app.services.tradfi_prices import clear_metal_kline_cache
        clear_metal_kline_cache()
    except Exception:
        pass
    await _invalidate_tb_conn()
    logger.info(
        "[CTraderFeed] kline builder restarted (%s — cleared %s cache entries)",
        reason,
        n,
    )


async def sweep_stale_klines(
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
) -> int:
    """Per-cycle staleness sweep — rebuild when ticks flow but klines are frozen."""
    if not _PROTO_OK or not _live_ticks_flowing():
        return 0

    syms = [s.upper() for s in (symbols or list(_TRACKED.keys()))]
    tfs = timeframes or ["5m", "15m", "1h"]
    rebuilt = 0

    for sym_up in syms:
        if sym_up not in _TRACKED:
            continue
        for tf in tfs:
            cache_key = None
            cached_rows = None
            cache_ts = 0.0
            for key, (rows, ts) in list(_kline_cache.items()):
                if key[0] == sym_up and key[1] == tf:
                    cached_rows = rows
                    cache_key = key
                    cache_ts = ts
                    break
            if not cached_rows:
                continue
            stale, detail = _kline_cache_is_stale(
                sym_up, cached_rows, tf, cache_mono_ts=cache_ts,
            )
            if not stale:
                continue
            _log_stale_kline_rebuild(sym_up, detail, tf)
            if cache_key:
                _kline_cache.pop(cache_key, None)
            clear_kline_cache(sym_up)
            try:
                from app.services.tradfi_prices import clear_metal_kline_cache
                clear_metal_kline_cache([sym_up])
            except Exception:
                pass
            await _invalidate_tb_conn()
            await get_klines(sym_up, "forex", tf, 80)
            rebuilt += 1
    return rebuilt


def clear_kline_cache(symbol: Optional[str] = None) -> int:
    """Drop cached trendbars (all symbols or one canonical symbol)."""
    global _kline_cache, _last_kline_update
    if symbol is None:
        n = len(_kline_cache)
        _kline_cache.clear()
        _last_kline_update.clear()
        return n
    sym = symbol.upper()
    removed = 0
    for key in list(_kline_cache):
        if key[0] == sym:
            _kline_cache.pop(key, None)
            removed += 1
    _last_kline_update.pop(sym, None)
    return removed


async def _on_spot_stream_connected() -> None:
    """After feed reconnect — ticks resume but kline cache may be frozen."""
    await restart_kline_builder("feed_reconnect")


def _get_tb_lock() -> "asyncio.Lock":
    global _tb_lock
    if _tb_lock is None:
        _tb_lock = asyncio.Lock()
    return _tb_lock


async def _invalidate_tb_conn() -> None:
    """Close and drop the persistent trendbar connection (forces a reopen)."""
    global _tb_conn, _tb_conn_ctx, _tb_hb_task
    if _tb_hb_task:
        _tb_hb_task.cancel()
        _tb_hb_task = None
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
    global _tb_conn, _tb_conn_ctx, _tb_hb_task
    if _tb_hb_task:
        _tb_hb_task.cancel()
        _tb_hb_task = None
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
    global _tb_hb_task
    if _tb_hb_task:
        _tb_hb_task.cancel()

    async def _tb_heartbeat(w):
        try:
            while True:
                await asyncio.sleep(10)
                await _send(w, ProtoHeartbeatEvent(), _PT_HEARTBEAT)
        except (asyncio.CancelledError, Exception):
            return

    _tb_hb_task = asyncio.create_task(_tb_heartbeat(writer))
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

    # Portal workers: never open a competing authed socket on the feed ctid.
    if not (_feed_live and _stream_writer is not None):
        if not _standalone_trendbar_fetch_allowed():
            logger.debug(
                "[CTraderFeed] standalone trendbar socket blocked %s %s",
                symbol,
                timeframe,
            )
            return []

    # LIVE spot session: multiplex on the stream — never open a second authed socket.
    if _feed_live and _stream_writer is not None:
        return await _fetch_trendbars_on_live_stream(symbol, timeframe, limit)

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
                deadline = time.monotonic() + (
                    12.0 if _feed_live else 6.0
                )
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

    blocked = trendbar_fetch_blocked_reason()
    if blocked:
        logger.debug("[CTraderFeed] trendbar fetch skipped %s %s: %s", sym_up, timeframe, blocked)
        cached = _cached_klines_synthesized(sym_up, timeframe, limit)
        if cached:
            return cached
        shared = _shared_klines_from_postgres(sym_up, timeframe, limit)
        if shared:
            return shared
        return []

    if not _standalone_trendbar_fetch_allowed():
        cached = _cached_klines_synthesized(sym_up, timeframe, limit)
        if cached:
            logger.debug(
                "[CTraderFeed] trendbar API skipped (non-feed process) — cache %s %s bars=%d",
                sym_up,
                timeframe,
                len(cached),
            )
            return cached
        shared = _shared_klines_from_postgres(sym_up, timeframe, limit)
        if shared:
            logger.info(
                "[CTraderFeed] postgres kline snapshot %s %s bars=%d (portal worker)",
                sym_up,
                timeframe,
                len(shared),
            )
            return shared
        logger.debug(
            "[CTraderFeed] trendbar API skipped (non-feed process) %s %s — no cache",
            sym_up,
            timeframe,
        )
        return []

    cache_key = (sym_up, timeframe, limit)
    cached = _kline_cache.get(cache_key)
    rate_key = (sym_up, timeframe)
    rate_ok = (
        time.monotonic() - _trendbar_last_api.get(rate_key, 0.0)
        >= _TRENDBAR_MIN_INTERVAL_S
    )
    if cached:
        cache_age = time.monotonic() - cached[1]
        kline_stale, stale_detail = _kline_cache_is_stale(
            sym_up, cached[0], timeframe, cache_mono_ts=cached[1],
        )
        if cache_age < _KLINE_TTL and not kline_stale:
            return _synthesize_klines_on_return(cached[0], sym_up, timeframe, limit)
        if not rate_ok and is_live() and not kline_stale:
            return _synthesize_klines_on_return(cached[0], sym_up, timeframe, limit)
        if kline_stale:
            _log_stale_kline_rebuild(sym_up, stale_detail, timeframe)
            _kline_cache.pop(cache_key, None)
            try:
                from app.services.tradfi_prices import clear_metal_kline_cache
                clear_metal_kline_cache([sym_up])
            except Exception:
                pass
            await _invalidate_tb_conn()

    async def _pull(
        creds_tuple,
        *,
        preferred_host: Optional[str] = None,
    ) -> List[List[float]]:
        global _stream_creds
        access_token, ctid, uid, is_live = creds_tuple
        primary_host = preferred_host or (_HOST_LIVE if is_live else _HOST_DEMO)
        rows = await _fetch_trendbars(
            sym_up, timeframe, limit, access_token, ctid, primary_host,
        )
        if not rows:
            from app.services.ctrader_client import _latest_ctrader_access_token

            fresh_at = _latest_ctrader_access_token(uid)
            if fresh_at and fresh_at != access_token:
                access_token = fresh_at
                if _stream_creds and _stream_creds[2] == uid:
                    _stream_creds = (access_token, ctid, uid, primary_host)
                rows = await _fetch_trendbars(
                    sym_up, timeframe, limit, access_token, ctid, primary_host,
                )
        if not rows and is_live is False:
            # Demo accounts must stay on demo host — never probe the live host.
            return rows
        if not rows:
            fallback_host = _HOST_DEMO if primary_host == _HOST_LIVE else _HOST_LIVE
            rows = await _fetch_trendbars(
                sym_up, timeframe, limit, access_token, ctid, fallback_host,
            )
        return rows

    rows: List[List[float]] = []
    sc = get_stream_creds()
    if sc:
        at, ctid, uid, host = sc
        rows = await _pull(
            (at, ctid, uid, host == _HOST_LIVE),
            preferred_host=host,
        )

    creds = await _get_connected_account(user_id=user_id)
    if creds and not rows:
        rows = await _pull(creds)
    # Spot stream may be live via another user's linked account — use the same
    # broker session for trendbars when a per-user lookup misses.
    if not rows and user_id is not None:
        shared = await _get_connected_account()
        if shared and shared != creds:
            rows = await _pull(shared)
            if rows:
                logger.info(
                    f"[CTraderFeed] klines via shared account: {sym_up} {timeframe} "
                    f"→ {len(rows)} bars"
                )
    if not rows and is_live():
        logger.warning(
            f"[CTraderFeed] spot feed LIVE but trendbars empty for {sym_up} {timeframe}"
        )
    if rows:
        now = time.monotonic()
        _kline_cache[cache_key] = (rows, now)
        _last_kline_update[sym_up] = now
        _persist_klines_for_peers(sym_up, timeframe, rows)
    return _synthesize_klines_on_return(rows, sym_up, timeframe, limit)


# ── Public price API ──────────────────────────────────────────────────────────

def get_price(symbol: str) -> Optional[float]:
    """
    Return the mid price for `symbol` from the live cTrader spot feed.
    Falls back to the shared Postgres tick store (other gunicorn workers),
    but only when that tick is source=ctrader — never Binance/Coinbase fallbacks.
    """
    sym = symbol.upper()
    entry = _spot_cache.get(sym)
    if entry:
        bid, ask, ts = entry
        if time.monotonic() - ts <= _SPOT_TTL:
            return round((bid + ask) / 2.0, 6)
    try:
        from app.services.spot_price_store import get_tick
        row = get_tick(sym, max_age_s=_SPOT_TTL)
        if row and (row.get("source") or "").lower() == "ctrader":
            return float(row["mid"])
    except Exception:
        pass
    return None


def get_bid_ask(symbol: str) -> Optional[Tuple[float, float]]:
    """Return (bid, ask) for `symbol` if a fresh cTrader tick is available."""
    sym = symbol.upper()
    entry = _spot_cache.get(sym)
    if entry:
        bid, ask, ts = entry
        if time.monotonic() - ts <= _SPOT_TTL:
            return (bid, ask)
    try:
        from app.services.spot_price_store import get_tick
        row = get_tick(sym, max_age_s=_SPOT_TTL)
        if row and (row.get("source") or "").lower() == "ctrader":
            bid, ask = row.get("bid"), row.get("ask")
            if bid is not None and ask is not None:
                return (float(bid), float(ask))
            mid = float(row["mid"])
            return (mid, mid)
    except Exception:
        pass
    return None


def is_live() -> bool:
    """True when spot ticks are streaming (local feed or remote Postgres store)."""
    if _feed_live:
        return True
    if _remote_feed_mode():
        return _shared_ctrader_ticks_fresh(max_age_s=_SPOT_TTL * 3)
    return False


def get_stream_creds() -> Optional[Tuple[str, int, int, str]]:
    """(access_token, ctid, user_id, host) for the active spot stream, if any."""
    return _stream_creds


def broker_session_ready(symbol: str = "XAUUSD") -> bool:
    """
    True when cTrader trendbars can be fetched without opening a competing socket.

    When the local spot feed is LIVE we must NOT open a second authed socket
  (same ctid). Remote-feed executors (no local stream) may pull trendbars while
    Postgres ticks are fresh; cold start may open a short-lived trendbar socket.
    """
    if _is_terminal_auth_error(_last_auth_error):
        return False
    if _feed_live:
        return _stream_writer is not None and trendbar_fetch_blocked_reason() is None
    if _stream_creds is not None:
        return True
    if _remote_feed_mode() and _shared_ctrader_ticks_fresh(max_age_s=_SPOT_TTL * 2):
        return True
    if not _trendbar_fetch_allowed():
        return False
    try:
        if probe_linked_accounts_sync():
            return True
    except Exception:
        pass
    return False


def ctrader_spot_ready(symbol: str = "XAUUSD") -> bool:
    """True when a fresh cTrader mid is available (local feed or remote store)."""
    sym = symbol.upper()
    if get_price(sym) is not None:
        return True
    if get_bid_ask(sym) is not None:
        return True
    try:
        from app.services.spot_price_store import get_tick
        row = get_tick(sym, max_age_s=_SPOT_TTL * 2)
        if row and (row.get("source") or "").lower() == "ctrader":
            return float(row.get("mid") or 0) > 0
    except Exception:
        pass
    return False


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


def notify_account_linked(user_id: Optional[int] = None) -> None:
    """Wake the feed loop immediately after OAuth links a cTrader account."""
    global _auth_backoff_until, _last_auth_error, _auth_terminal_alert_sent
    _auth_backoff_until = 0.0
    _last_auth_error = None
    _auth_terminal_alert_sent = False
    invalidate_stream_creds(user_id)
    try:
        from app.services.ctrader_client import clear_ctrader_oauth_denied
        clear_ctrader_oauth_denied(user_id)
    except Exception:
        pass
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
    shared = {}
    try:
        from app.services.spot_price_store import snapshot as _spot_snap
        shared = _spot_snap(max_age_s=20.0)
        if shared.get("symbol_count") and not fresh:
            fresh = list(shared.get("symbols") or [])[:30]
        if shared.get("last_tick_age_s") is not None and last_tick_age is None:
            last_tick_age = shared["last_tick_age_s"]
    except Exception:
        pass
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
    needs_relink = _is_terminal_auth_error(_last_auth_error)
    if needs_relink and not note:
        note = (
            "cTrader OAuth expired — disconnect and reconnect cTrader in Settings "
            "to restore live forex ticks."
        )
    remote = _remote_feed_mode()
    local_live = _feed_live
    remote_live = bool((shared.get("by_source") or {}).get("ctrader"))
    return {
        "subscribed":       _feed_live,
        "live":             local_live or remote_live or bool(shared.get("symbol_count")),
        "remote_feed":      remote,
        "local_subscribed": local_live,
        "remote_live":      remote_live,
        "symbol_count":     max(len(fresh), int(shared.get("symbol_count") or 0)),
        "cached_symbols":   fresh[:30],
        "symbols_seen":     max(len(all_cached), int(shared.get("symbol_count") or 0)),
        "last_tick_age_s":  last_tick_age,
        "forex_market_open": forex_open,
        "shared_store":     shared,
        "last_auth_error":  _last_auth_error,
        "needs_relink":     needs_relink,
        "auth_backoff_s":   max(0.0, _auth_backoff_until - now) if needs_relink else 0.0,
        "note":             note,
    }


# ── Startup ───────────────────────────────────────────────────────────────────

def _on_feed_task_done(task: asyncio.Task) -> None:
    global _feed_starting
    with _feed_launch_lock:
        _feed_starting = False
    if not task.cancelled() and task.exception():
        logger.error(
            "[CTraderFeed] feed task ended with error: %s",
            task.exception(),
            exc_info=task.exception(),
        )


async def _feed_runner() -> None:
    """Resilient outer shell — restarts _feed_loop after crashes (never silent)."""
    delay = float(_RECONNECT_BACKOFF_MIN)
    while True:
        try:
            await _feed_loop()
            logger.warning(
                "[CTraderFeed] feed loop exited cleanly — restarting in %.0fs", delay,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "[CTraderFeed] feed loop crashed: %s — restarting in %.0fs",
                exc,
                delay,
                exc_info=True,
            )
        await asyncio.sleep(delay)
        delay = min(delay * 2.0, 120.0)


def launch_ctrader_feed() -> bool:
    """
    Launch the background streaming task with explicit startup logging.
    Safe to call multiple times — ignored if already running.
    Returns True when a feed task is running or was just scheduled.
    """
    global _feed_task, _feed_starting, _feed_singleton_uid
    with _feed_launch_lock:
        if _feed_task and not _feed_task.done():
            logger.info(
                "[CTraderFeed] feed task already running uid=%s — skip duplicate start",
                _feed_singleton_uid,
            )
            return True
        if _feed_starting:
            logger.info(
                "[CTraderFeed] feed task already starting uid=%s — skip duplicate launch",
                _feed_singleton_uid,
            )
            return True
        _feed_starting = True
    logger.info("[CTraderFeed] startup: launching feed task")

    def _abort_feed_launch() -> None:
        global _feed_starting
        with _feed_launch_lock:
            _feed_starting = False

    try:
        from app.ctrader_feed_lock import (
            feed_disabled_in_executor,
            is_feed_only_process,
            remote_feed_enabled,
        )
        if feed_disabled_in_executor() and not is_feed_only_process():
            fresh = _shared_ctrader_ticks_fresh(max_age_s=60.0)
            reason = "CTRADER_REMOTE_FEED" if remote_feed_enabled() else "DISABLE_CTRADER_FEED_IN_EXECUTOR"
            logger.info(
                "[CTraderFeed] feed not started — %s (remote_ticks_fresh=%s)",
                reason,
                fresh,
            )
            _abort_feed_launch()
            return False
        if remote_feed_enabled() and not is_feed_only_process():
            logger.warning(
                "[CTraderFeed] remote feed stale — starting local fallback feed in executor"
            )
    except Exception as exc:
        logger.warning("[CTraderFeed] feed disable check failed: %s", exc)

    if not _PROTO_OK:
        logger.warning("[CTraderFeed] feed not started — ctrader_open_api/protobuf unavailable")
        _abort_feed_launch()
        return False
    if not os.environ.get("CTRADER_CLIENT_ID"):
        logger.warning("[CTraderFeed] feed not started — CTRADER_CLIENT_ID not set")
        _abort_feed_launch()
        return False

    linked = probe_linked_accounts_sync()
    if not linked:
        logger.warning(
            "[CTraderFeed] no linked cTrader account in DB — "
            "feed task will start and wait for OAuth link"
        )
    else:
        _at, ctid, uid, is_live = linked[0]
        _feed_singleton_uid = int(uid)
        logger.info(
            "[CTraderFeed] startup: DB token row found uid=%s ctid=%s mode=%s "
            "(%d linked account(s))",
            uid,
            ctid,
            "live" if is_live else "demo",
            len(linked),
        )
        try:
            from app.services.ctrader_client import _log_ctrader_token_startup
            if _at:
                _log_ctrader_token_startup(int(uid), _at.strip())
        except Exception:
            pass

    try:
        _get_wake_event()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        with _feed_launch_lock:
            if _feed_task and not _feed_task.done():
                _feed_starting = False
                return True
            _feed_task = loop.create_task(_feed_runner())
            _feed_task.add_done_callback(_on_feed_task_done)
        logger.info(
            "[CTraderFeed] background streaming task scheduled uid=%s",
            _feed_singleton_uid,
        )
        return True
    except Exception as exc:
        _abort_feed_launch()
        logger.error("[CTraderFeed] failed to schedule feed task: %s", exc, exc_info=True)
        return False


def start() -> None:
    """Backward-compatible alias for launch_ctrader_feed()."""
    launch_ctrader_feed()


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
