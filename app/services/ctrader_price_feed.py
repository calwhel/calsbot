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
_PT_EXECUTION_EVENT = 2126
_PT_TRENDBARS_REQ = 2137
_PT_TRENDBARS_RES = 2138
_PT_HEARTBEAT     = 51    # ProtoHeartbeatEvent — client MUST send ~every 10s or
                         # cTrader drops the socket (the "stream read timeout" churn)
_PT_ERROR_RES     = 2142  # ProtoOAErrorRes — account auth / request failures

# Proactive OAuth refresh — access tokens expire ~1h; refresh before they brick the feed.
_PROACTIVE_REFRESH_INTERVAL_S = float(
    os.environ.get("CTRADER_PROACTIVE_REFRESH_INTERVAL_S", str(45 * 60))
)
_last_proactive_refresh: Dict[int, float] = {}  # user_id → monotonic
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
# Active spot-stream session — avoids DB round-trips for trendbars while LIVE.
_stream_creds: Optional[Tuple[str, int, int, str]] = None  # token, ctid, uid, host
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


def _trendbar_fetch_allowed() -> bool:
    """
    False when opening a trendbar socket would compete with the spot feed.

    A second account-auth TLS session on the same ctid causes cTrader to recycle
    the spot stream — the #1 source of LIVE flapping in production logs.
    Executor workers without a local spot stream (remote-feed mode) may fetch
    trendbars on demand — only the feed-only process holds the streaming socket.
    """
    return not _feed_live


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


async def _maybe_refresh_access_token(
    user_id: int,
    access_token: str,
    *,
    force: bool = False,
) -> str:
    """Refresh OAuth access token — always re-reads refresh_token from DB."""
    now = time.monotonic()
    last = _last_proactive_refresh.get(user_id, 0.0)
    if not force and (now - last) < _PROACTIVE_REFRESH_INTERVAL_S:
        try:
            from app.database import SessionLocal
            from app.models import UserPreference
            db = SessionLocal()
            try:
                prefs = db.query(UserPreference).filter(
                    UserPreference.user_id == user_id
                ).first()
                if prefs and prefs.ctrader_access_token:
                    return prefs.ctrader_access_token.strip()
            finally:
                db.close()
        except Exception:
            pass
        return (access_token or "").strip()
    try:
        from app.services.ctrader_client import (
            is_refresh_denied,
            refresh_user_ctrader_token,
        )
        if is_refresh_denied(user_id):
            return (access_token or "").strip()
        new_at = await refresh_user_ctrader_token(user_id)
    except Exception as exc:
        logger.warning(
            f"[CTraderFeed] token refresh failed uid={user_id}: {type(exc).__name__}"
        )
        return (access_token or "").strip()
    if new_at:
        _last_proactive_refresh[user_id] = now
        invalidate_stream_creds(user_id)
        return new_at.strip()
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
    import json as _json
    try:
        raw = getattr(prefs, "ctrader_accounts", None)
        if raw:
            for a in _json.loads(raw):
                if int(a.get("ctidTraderAccountId", -1)) == int(ctid):
                    return bool(a.get("isLive", True))
    except Exception:
        pass
    return True


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
    return out


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
    out: List[Tuple[str, int, int, bool]] = []
    try:
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
            rows = (
                q.order_by(
                    UserPreference.forex_approved.desc(),
                    UserPreference.user_id.desc(),
                )
                .all()
            )
            out = _prefs_rows_to_accounts(rows)
        finally:
            db.close()
    except Exception as e:
        logger.warning("[CTraderFeed] DB lookup error: %s", e)
    return out


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
    at = await _maybe_refresh_access_token(user_id, access_token, force=False)
    host = _HOST_LIVE if is_live else _HOST_DEMO
    reader = writer = None
    try:
        reader, writer = await _open_conn(host)
        if not await _app_auth(reader, writer):
            raise ConnectionError("app auth failed")
        authed = await _account_auth(reader, writer, at, ctid)
        if not authed:
            at = await _maybe_refresh_access_token(user_id, at, force=True)
            invalidate_stream_creds(user_id)
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


def _live_ticks_flowing(max_age_s: float = 30.0) -> bool:
    return (time.monotonic() - _last_spot_tick_mono) <= max_age_s


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
    n = clear_kline_cache()
    await _invalidate_tb_conn()
    try:
        from app.services.tradfi_prices import clear_metal_kline_cache

        clear_metal_kline_cache()
    except Exception:
        pass
    if n:
        logger.info(
            "[CTraderFeed] feed reconnect — cleared %s kline cache entries "
            "(trendbar session reset)",
            n,
        )


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

    if not _trendbar_fetch_allowed():
        return []

    cache_key = (sym_up, timeframe, limit)
    cached = _kline_cache.get(cache_key)
    if cached:
        cache_age = time.monotonic() - cached[1]
        last_up = _last_kline_update.get(sym_up, 0.0)
        kline_age = time.monotonic() - last_up if last_up else cache_age
        stale_limit = _kline_stale_limit_s(timeframe)
        kline_stale = _live_ticks_flowing() and kline_age > stale_limit
        if cache_age < _KLINE_TTL and not kline_stale:
            return cached[0]
        if kline_stale:
            logger.warning(
                "[CTraderFeed] kline builder stale for %s — rebuilt "
                "(klines %.0fs old, ticks flowing)",
                sym_up,
                kline_age,
            )
            _kline_cache.pop(cache_key, None)

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
            new_at = None
            # Only the spot feed loop should refresh tokens while streaming — a
            # trendbar miss must not race the rotating refresh token (bricks OAuth).
            _may_refresh = not _feed_live and not _is_terminal_auth_error(_last_auth_error)
            if _may_refresh:
                try:
                    from app.services.ctrader_client import (
                        is_refresh_denied,
                        refresh_user_ctrader_token,
                    )
                    if not is_refresh_denied(uid):
                        new_at = await refresh_user_ctrader_token(uid)
                except Exception:
                    new_at = None
            if new_at:
                access_token = new_at
                if _stream_creds and _stream_creds[2] == uid:
                    _stream_creds = (access_token, ctid, uid, primary_host)
                rows = await _fetch_trendbars(
                    sym_up, timeframe, limit, access_token, ctid, primary_host,
                )
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
    return rows


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
        return False
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
    if is_live():
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
    global _feed_task
    logger.info("[CTraderFeed] startup: launching feed task")

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
            return False
        if remote_feed_enabled() and not is_feed_only_process():
            logger.warning(
                "[CTraderFeed] remote feed stale — starting local fallback feed in executor"
            )
    except Exception as exc:
        logger.warning("[CTraderFeed] feed disable check failed: %s", exc)

    if not _PROTO_OK:
        logger.warning("[CTraderFeed] feed not started — ctrader_open_api/protobuf unavailable")
        return False
    if not os.environ.get("CTRADER_CLIENT_ID"):
        logger.warning("[CTraderFeed] feed not started — CTRADER_CLIENT_ID not set")
        return False

    linked = probe_linked_accounts_sync()
    if not linked:
        logger.warning(
            "[CTraderFeed] no linked cTrader account in DB — "
            "feed task will start and wait for OAuth link"
        )
    else:
        _at, ctid, uid, is_live = linked[0]
        logger.info(
            "[CTraderFeed] startup: DB token row found uid=%s ctid=%s mode=%s "
            "(%d linked account(s))",
            uid,
            ctid,
            "live" if is_live else "demo",
            len(linked),
        )

    if _feed_task and not _feed_task.done():
        logger.info("[CTraderFeed] feed task already running — skip duplicate start")
        return True

    try:
        _get_wake_event()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        _feed_task = loop.create_task(_feed_runner())
        logger.info("[CTraderFeed] background streaming task scheduled")
        return True
    except Exception as exc:
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
