"""
cTrader Open API asyncio client (Spotware protocol).

Connects to live.ctraderapi.com:5035 over TLS. Each message is framed as:
  [4-byte big-endian length][ProtoMessage bytes]

where ProtoMessage.payloadType identifies the message type and
ProtoMessage.payload holds the inner protobuf bytes.

Usage (place a market order):
    result = await place_ctrader_order_for_user(
        user, symbol="EURUSD", direction="LONG",
        entry_price=1.085, tp_pct=0.5, sl_pct=0.25, risk_pct=1.0
    )
"""

import asyncio
import ssl
import struct
import logging
import os
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# httpx logs every request URL at INFO — and our OAuth token calls put the
# refresh_token + client_secret in the query string, so those would be leaked
# verbatim into the deployment logs. Silence httpx's INFO request logging.
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Spotware Open API endpoints ───────────────────────────────────────────────
# Demo and live accounts live on SEPARATE hosts. A ctid is reachable ONLY on
# its matching host — account auth silently fails otherwise (looks like a token
# problem but isn't). Route by the account's isLive flag.
CTRADER_HOST_LIVE = "live.ctraderapi.com"
CTRADER_HOST_DEMO = "demo.ctraderapi.com"
CTRADER_HOST = CTRADER_HOST_LIVE   # default / backward-compat
CTRADER_PORT = 5035


def _host_for_account(prefs, ctid: int) -> str:
    """Pick live vs demo host for a ctid by reading isLive from the stored
    ctrader_accounts JSON on the user's prefs. Defaults to LIVE when metadata
    is missing (callers fall back to the other host on auth failure)."""
    try:
        import json as _json
        raw = getattr(prefs, "ctrader_accounts", None)
        if raw:
            for a in _json.loads(raw):
                if int(a.get("ctidTraderAccountId", -1)) == int(ctid):
                    return CTRADER_HOST_LIVE if bool(a.get("isLive", True)) else CTRADER_HOST_DEMO
    except Exception:
        pass
    return CTRADER_HOST_LIVE


def _other_host(host: str) -> str:
    return CTRADER_HOST_DEMO if host == CTRADER_HOST_LIVE else CTRADER_HOST_LIVE

# OAuth endpoints — from official Spotware docs:
# https://help.ctrader.com/open-api/account-authentication/
# Auth:  ctrader.com/my/settings/openapi/grantingaccess/ (client_id is query param)
# Token: openapi.ctrader.com/apps/token
# NOTE: /playground is developer-only testing; connect.spotware.com/apps/{id}/auth → 404.
OAUTH_AUTH_URL  = "https://id.ctrader.com/my/settings/openapi/grantingaccess/"
OAUTH_TOKEN_URL = "https://openapi.ctrader.com/apps/token"

# ── App credentials (injected via env) ───────────────────────────────────────
CTRADER_CLIENT_ID     = os.environ.get("CTRADER_CLIENT_ID", "")
CTRADER_CLIENT_SECRET = os.environ.get("CTRADER_CLIENT_SECRET", "")

# ── Import Spotware protobuf message classes ──────────────────────────────────
try:
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAApplicationAuthReq,
        ProtoOAApplicationAuthRes,
        ProtoOAAccountAuthReq,
        ProtoOAAccountAuthRes,
        ProtoOANewOrderReq,
        ProtoOAExecutionEvent,
        ProtoOAOrderErrorEvent,
        ProtoOASymbolsListReq,
        ProtoOASymbolsListRes,
        ProtoOASymbolByIdReq,
        ProtoOASymbolByIdRes,
        ProtoOAGetAccountListByAccessTokenReq,
        ProtoOAGetAccountListByAccessTokenRes,
        ProtoOAGetCtidProfileByTokenReq,
        ProtoOAGetCtidProfileByTokenRes,
        ProtoOARefreshTokenReq,
        ProtoOARefreshTokenRes,
        ProtoOAClosePositionReq,
        ProtoOAAmendPositionSLTPReq,
        ProtoOAReconcileReq,
        ProtoOAReconcileRes,
    )
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
        ProtoOATradeSide,
        ProtoOAOrderType,
        ProtoOAExecutionType,
    )
    _PROTO_OK = True
except ImportError as _e:
    logger.warning(f"ctrader_open_api not available: {_e}")
    _PROTO_OK = False

# Payload type IDs (from Spotware spec)
_PAYLOAD_TYPES = {
    "application_auth_req":  2100,
    "application_auth_res":  2101,
    "account_auth_req":      2102,
    "account_auth_res":      2103,
    "new_order_req":         2106,
    "symbols_list_req":      2114,
    "symbols_list_res":      2115,
    "symbol_by_id_req":      2116,
    "symbol_by_id_res":      2117,
    "execution_event":       2126,
    "order_error_event":     2132,
    "account_list_req":      2149,
    "account_list_res":      2150,
    "ctid_profile_req":      2114,
    "ctid_profile_res":      2115,
    "refresh_token_req":     2072,
    "refresh_token_res":     2073,
    "close_position_req":    2140,
    "amend_position_sltp_req": 2110,
    "reconcile_req":         2124,
    "reconcile_res":         2125,
}

# Forex pip sizes
_PIP_SIZES = {
    # Forex majors
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001,
    "NZDUSD": 0.0001, "USDCAD": 0.0001, "USDCHF": 0.0001,
    "USDJPY": 0.01,   "EURJPY": 0.01,   "GBPJPY": 0.01,
    # Metals — retail/broker pip convention (gold pip = $0.10), kept in lockstep
    # with forex_engine._METAL_PIP_SIZES and the pip_value table below.
    "XAUUSD": 0.10,
    # Indices — 1 point = 1.0 index level (FP Markets convention)
    "NAS100": 1.0, "SPX500": 1.0, "US30": 1.0, "GER40": 1.0, "UK100": 1.0,
    "NDX": 1.0, "SPX": 1.0, "DJI": 1.0, "DAX": 1.0, "FTSE": 1.0,
}

# FP Markets / cTrader symbol name mapping
# Indices use broker-specific contract names on FP Markets cTrader.
_SYMBOL_MAP = {
    # Forex
    "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD", "USDCAD": "USDCAD", "USDCHF": "USDCHF",
    "NZDUSD": "NZDUSD",
    # Indices — canonical cTrader names + legacy aliases
    "NAS100": "US100", "NDX": "US100", "US100": "US100",
    "SPX500": "US500", "SPX": "US500",  "US500": "US500",
    "US30":   "US30",  "DJI": "US30",
    "GER40":  "GER40", "DAX": "GER40",
    "UK100":  "UK100", "FTSE": "UK100",
}

# Index contract sizing: for index CFDs volume is in contracts (1 unit = 1 contract).
# FP Markets minimum is 1 contract; value ≈ price × contract_size USD.
try:
    from app.services.index_symbols import CTRADER_INDEX_SYMBOLS as _INDEX_SYMBOLS
except Exception:
    _INDEX_SYMBOLS = frozenset({"NAS100", "SPX500", "US30", "GER40", "UK100", "SPX", "NDX", "DJI", "DAX", "FTSE"})

# Broker symbol name → numeric symbolId, cached per (host, ctid). cTrader's
# ProtoOANewOrderReq requires the integer symbolId — there is NO symbolName
# field (setting it raises AttributeError). symbolIds are not portable across
# accounts/hosts, so the cache is keyed by (host, ctid) and re-resolved when
# either changes.
_SYMBOL_ID_CACHE: Dict[Tuple[str, int], Dict[str, int]] = {}

# Per-symbol volume metadata (lotSize / minVolume / maxVolume / stepVolume), all
# expressed in cTrader's order-volume units. The order `volume` field is NOT in
# "lots" — it is volume_lots × lotSize, and lotSize is broker/instrument-specific
# (e.g. FX majors 10_000_000, XAUUSD often 100_000-ish). Hardcoding lots×100_000
# for every symbol wildly over/under-sizes metals → NOT_ENOUGH_MONEY rejections.
# Cached per (host, ctid, symbolId) since these are per-account like symbolIds.
_SYMBOL_DETAILS_CACHE: Dict[Tuple[str, int, int], Dict[str, int]] = {}


# ── Low-level framing helpers ─────────────────────────────────────────────────

def _encode_msg(proto_req, payload_type: int) -> bytes:
    wrapper = ProtoMessage()
    wrapper.payloadType = payload_type
    wrapper.payload     = proto_req.SerializeToString()
    data = wrapper.SerializeToString()
    return struct.pack(">I", len(data)) + data


def _decode_msg(data: bytes) -> Optional[ProtoMessage]:
    msg = ProtoMessage()
    msg.ParseFromString(data)
    return msg


async def _send_recv(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    proto_req,
    req_type: int,
    expected_type: int,
    timeout: float = 10.0,
) -> Optional[bytes]:
    """Send a framed proto message and wait for the expected response type."""
    writer.write(_encode_msg(proto_req, req_type))
    await writer.drain()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            header = await asyncio.wait_for(reader.readexactly(4), timeout=remaining)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError):
            break
        length = struct.unpack(">I", header)[0]
        try:
            body = await asyncio.wait_for(
                reader.readexactly(length), timeout=min(remaining, 10.0)
            )
        except (asyncio.IncompleteReadError, asyncio.TimeoutError):
            break
        msg = _decode_msg(body)
        if msg and msg.payloadType == expected_type:
            return msg.payload
        # silently skip heartbeats / other events
    return None


async def _recv_until(
    reader: asyncio.StreamReader,
    expected_types: set,
    timeout: float = 10.0,
) -> Tuple[Optional[int], Optional[bytes]]:
    """Read frames (no send) and return (payloadType, payload) for the FIRST one
    whose payloadType is in `expected_types`, skipping heartbeats / other events.
    (None, None) on timeout. Callers must hold the connection lock so the socket
    is exclusively theirs while draining frames.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            header = await asyncio.wait_for(reader.readexactly(4), timeout=remaining)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError):
            break
        length = struct.unpack(">I", header)[0]
        try:
            body = await asyncio.wait_for(
                reader.readexactly(length), timeout=min(remaining, 10.0)
            )
        except (asyncio.IncompleteReadError, asyncio.TimeoutError):
            break
        msg = _decode_msg(body)
        if msg and msg.payloadType in expected_types:
            return msg.payloadType, msg.payload
        # silently skip heartbeats / other events
    return None, None


async def _send_recv_any(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    proto_req,
    req_type: int,
    expected_types: set,
    timeout: float = 10.0,
) -> Tuple[Optional[int], Optional[bytes]]:
    """Like _send_recv but returns (payloadType, payload) for the FIRST message
    whose payloadType is in `expected_types` — lets a caller distinguish e.g. a
    successful execution event from an order-error event. (None, None) on timeout.
    """
    writer.write(_encode_msg(proto_req, req_type))
    await writer.drain()
    return await _recv_until(reader, expected_types, timeout)


async def _open_connection(host: str = CTRADER_HOST) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    ctx = ssl.create_default_context()
    return await asyncio.open_connection(host, CTRADER_PORT, ssl=ctx)


async def _aclose_writer(writer) -> None:
    """Close an asyncio SSL StreamWriter AND wait for the FD to be released.

    A bare writer.close() does NOT promptly free the underlying SSL socket file
    descriptor — under reconnect/invalidation churn that leaks FDs until the
    process hits 'Too many open files'. wait_closed() (timeout-guarded so a dead
    peer can't hang us) ensures the FD is actually released.
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


async def _app_auth(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> bool:
    req = ProtoOAApplicationAuthReq()
    req.clientId     = CTRADER_CLIENT_ID
    req.clientSecret = CTRADER_CLIENT_SECRET
    payload = await _send_recv(
        reader, writer, req,
        _PAYLOAD_TYPES["application_auth_req"],
        _PAYLOAD_TYPES["application_auth_res"],
    )
    if payload is None:
        return False
    res = ProtoOAApplicationAuthRes()
    res.ParseFromString(payload)
    return True


# ── Persistent connection pool (per host) ────────────────────────────────────
# Keeps one SSL TCP socket open PER host (demo.ctraderapi.com AND
# live.ctraderapi.com) so that each order avoids the ~300-500ms SSL handshake +
# application-auth round-trip.  Demo and live accounts are reachable ONLY on
# their matching host, so a demo order must never ride a live-host socket (or
# vice-versa) — hence the per-host keying.
# Account auth is still done per call (required by the protocol), but that
# is a single round-trip on an already-open socket (~30-80ms vs ~300-500ms
# for a fresh SSL+app-auth sequence).
#
# Per-(host, account) sockets + locks so 20+ cTrader users don't queue behind one
# global mutex. Token-only calls (account list) use ctid=0 on that host.

_account_locks: dict = {}
_conns: dict = {}            # (host, ctid) → {"reader", "writer", "ts"}
_CONN_MAX_IDLE = 45.0
_balance_cache: dict = {}    # (host, ctid) → (balance, monotonic_ts)
_BALANCE_CACHE_TTL = 25.0


def _acct_conn_key(host: str, ctid: int = 0) -> tuple:
    return (host, int(ctid))


def _get_account_lock(host: str, ctid: int = 0) -> asyncio.Lock:
    key = _acct_conn_key(host, ctid)
    lock = _account_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _account_locks[key] = lock
    return lock


def _get_lock() -> asyncio.Lock:
    """Host-level lock for access-token operations without a trading account."""
    return _get_account_lock(CTRADER_HOST, 0)


async def _get_persistent_connection(
    host: str = CTRADER_HOST,
    ctid: int = 0,
) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Return app-authenticated socket for (host, ctid). Caller holds account lock."""
    key = _acct_conn_key(host, ctid)
    st = _conns.get(key)
    if st is not None:
        idle = time.monotonic() - st["ts"]
        if st["writer"] is not None and not st["writer"].is_closing() and idle < _CONN_MAX_IDLE:
            return st["reader"], st["writer"]
        await _aclose_writer(st.get("writer"))
        _conns.pop(key, None)

    r, w = await _open_connection(host)
    ok = await _app_auth(r, w)
    if not ok:
        await _aclose_writer(w)
        raise RuntimeError("cTrader persistent connection: app auth failed")

    _conns[key] = {"reader": r, "writer": w, "ts": time.monotonic()}
    logger.debug(f"[cTrader] persistent connection (re)established → {host} acct={ctid}")
    return r, w


def _touch_conn(host: str, ctid: int = 0) -> None:
    st = _conns.get(_acct_conn_key(host, ctid))
    if st is not None:
        st["ts"] = time.monotonic()


def _invalidate_persistent_connection(host: str = CTRADER_HOST, ctid: int = 0) -> None:
    st = _conns.pop(_acct_conn_key(host, ctid), None)
    if st is not None and st.get("writer") is not None:
        try:
            asyncio.ensure_future(_aclose_writer(st["writer"]))
        except Exception:
            try:
                st["writer"].close()
            except Exception:
                pass


async def _account_auth(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    access_token: str,
    ctid_trader_account_id: int,
) -> bool:
    req = ProtoOAAccountAuthReq()
    req.ctidTraderAccountId = ctid_trader_account_id
    req.accessToken         = access_token
    payload = await _send_recv(
        reader, writer, req,
        _PAYLOAD_TYPES["account_auth_req"],
        _PAYLOAD_TYPES["account_auth_res"],
    )
    return payload is not None


# ── Public API ────────────────────────────────────────────────────────────────

def get_oauth_url(redirect_uri: str, state: str = "") -> str:
    """Return the Spotware OAuth authorization URL.

    Uses the official end-user granting-access endpoint:
    https://ctrader.com/my/settings/openapi/grantingaccess/
    client_id = full CTRADER_CLIENT_ID string (numeric-only prefix causes "Malformed clientId").
    """
    import urllib.parse
    # Use the full client ID string (e.g. "29040_abc...") — the numeric-only
    # prefix causes "Malformed clientId parameter" on id.ctrader.com.
    params = {
        "client_id":     CTRADER_CLIENT_ID,
        "redirect_uri":  redirect_uri,
        "scope":         "trading",
        "product":       "web",
    }
    if state:
        params["state"] = state
    url = f"{OAUTH_AUTH_URL}?{urllib.parse.urlencode(params)}"
    logger.info(f"[ctrader] OAuth URL → {url}")
    return url


class CTraderTokenError(Exception):
    """cTrader OAuth token endpoint returned errorCode in JSON body."""

    def __init__(self, error_code: str, description: str = ""):
        self.error_code = error_code or "UNKNOWN"
        self.description = description or ""
        super().__init__(self.error_code)


async def exchange_code(code: str, redirect_uri: str) -> dict:
    """Exchange an OAuth authorization code for access + refresh tokens.

    Spotware docs require GET (not POST) for grant_type=authorization_code.
    client_id = FULL app ID (e.g. "29040_abc..."); Spotware rejects the numeric-only prefix.
    """
    import httpx
    app_id = CTRADER_CLIENT_ID
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            OAUTH_TOKEN_URL,
            params={
                "grant_type":    "authorization_code",
                "code":          code,
                "redirect_uri":  redirect_uri,
                "client_id":     app_id,
                "client_secret": CTRADER_CLIENT_SECRET,
            },
            headers={"Accept": "application/json"},
        )
        data = resp.json()
        logger.info(f"[ctrader] exchange_code HTTP {resp.status_code} → keys={list(data.keys())}")
        err = data.get("errorCode")
        if err:
            desc = str(data.get("description") or data.get("error_description") or "")[:200]
            logger.error(
                "[ctrader] exchange_code errorCode=%s description=%s redirect_uri=%s",
                err, desc, redirect_uri,
            )
            raise CTraderTokenError(str(err), desc)
        if not (data.get("accessToken") or data.get("access_token")):
            raise CTraderTokenError("NO_ACCESS_TOKEN", "Token response missing accessToken")
        resp.raise_for_status()
        return data


async def refresh_access_token(refresh_token: str) -> dict:
    """Use a refresh token to get a new access token."""
    import httpx
    app_id = CTRADER_CLIENT_ID
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            OAUTH_TOKEN_URL,
            params={
                "grant_type":    "refresh_token",
                "refresh_token": refresh_token,
                "client_id":     app_id,
                "client_secret": CTRADER_CLIENT_SECRET,
            },
        )
        resp.raise_for_status()
        return resp.json()


# Serialise token refreshes within a process so two coroutines (feed + executor)
# don't both burn the rotating refresh token at once.
_token_refresh_lock = asyncio.Lock()

# Cross-process advisory lock NAMESPACE for token refresh — distinct from the
# executor (42_424_242) and aigen (42_424_243) lock ids in
# strategy_portal_server.py. cTrader ROTATES the refresh token on every use;
# under gunicorn -w 4 the feed/executor worker AND any request worker
# (charts/backtests hit tradfi_prices → cTrader trendbar fetch → refresh) can
# refresh concurrently. The per-process asyncio lock above can't coordinate
# across processes, so two workers racing the same single-use token lose a
# rotation and permanently brick the chain (ACCESS_DENIED → forced re-link).
# We use the two-key advisory-lock form (namespace, user_id) so refreshes are
# serialised PER USER — one user's refresh never blocks another's.
_TOKEN_REFRESH_PG_LOCK_NS = 42_424_244

# When a specific refresh token is denied (ACCESS_DENIED → chain dead, user must
# re-link), back off instead of hammering cTrader's OAuth endpoint every signal
# tick. Keyed by user_id → (denied_refresh_token, cooldown_until_monotonic). The
# guard is scoped to the EXACT token value, so a re-link (which rotates the token
# to a new value) clears it automatically without any cross-module coupling.
_refresh_denied: dict = {}
_REFRESH_DENY_COOLDOWN = 300.0  # seconds


async def refresh_user_ctrader_token(user_id: int) -> Optional[str]:
    """Refresh and persist a user's cTrader OAuth token.

    cTrader ROTATES the refresh token on every refresh — the new refresh token
    MUST be persisted or the next refresh fails with ACCESS_DENIED. Re-reads
    prefs inside the lock so concurrent callers pick up an already-refreshed
    token instead of replaying a stale (already-rotated) one.

    Returns the new access token, or None if there is no refresh token / the
    refresh was denied (user must re-link their cTrader account).
    """
    from app.database import SessionLocal
    from app.models import UserPreference
    from sqlalchemy import text as _sql_text
    async with _token_refresh_lock:
        db = SessionLocal()
        try:
            # Cross-process serialisation (see _TOKEN_REFRESH_PG_LOCK_ID). The
            # asyncio lock above only serialises within ONE worker. Take a
            # Postgres transaction-scoped advisory lock so the other gunicorn
            # workers can't refresh the same rotating token at the same time.
            # xact locks auto-release on commit/rollback/close → no leak path.
            acquired = False
            for _attempt in range(20):  # wait up to ~10s for a peer's refresh
                acquired = bool(
                    db.execute(
                        _sql_text("SELECT pg_try_advisory_xact_lock(:ns, :uid)"),
                        {"ns": _TOKEN_REFRESH_PG_LOCK_NS, "uid": int(user_id)},
                    ).scalar()
                )
                if acquired:
                    break
                db.rollback()  # drop the empty txn before sleeping + retrying
                await asyncio.sleep(0.5)

            prefs = (
                db.query(UserPreference)
                .filter(UserPreference.user_id == user_id)
                .first()
            )
            if not prefs or not prefs.ctrader_refresh_token:
                return None
            if not acquired:
                # Another worker is mid-refresh — racing it is exactly what bricks
                # the rotation chain. Reuse the current access token; by now the
                # peer has very likely persisted a freshly rotated one.
                return prefs.ctrader_access_token
            refresh_token = prefs.ctrader_refresh_token
            denied = _refresh_denied.get(user_id)
            if denied and denied[0] == refresh_token and time.monotonic() < denied[1]:
                # This exact token was already rejected — don't re-hit OAuth until
                # the user re-links (rotating the token clears this guard).
                return None
            try:
                res = await refresh_access_token(refresh_token)
            except Exception as e:
                # NB: do NOT interpolate `e` — for httpx.HTTPStatusError its string
                # includes the full URL, which carries refresh_token + client_secret
                # in the query string. Log only the type + status code.
                status = getattr(getattr(e, "response", None), "status_code", None)
                logger.warning(
                    f"[cTrader] token refresh HTTP error for user {user_id}: "
                    f"{type(e).__name__} status={status}"
                )
                return None
            if res.get("errorCode"):
                err = res.get("errorCode")
                # Only a hard ACCESS_DENIED means the chain is dead and re-link is
                # required — cooldown that. Other errors may be transient, so don't
                # suppress retries for them.
                if err == "ACCESS_DENIED":
                    _refresh_denied[user_id] = (
                        refresh_token, time.monotonic() + _REFRESH_DENY_COOLDOWN
                    )
                    logger.warning(
                        f"[cTrader] token refresh denied for user {user_id}: "
                        f"{err} — user must re-link cTrader"
                    )
                else:
                    logger.warning(
                        f"[cTrader] token refresh error for user {user_id}: {err}"
                    )
                return None
            new_at = res.get("accessToken") or res.get("access_token")
            new_rt = res.get("refreshToken") or res.get("refresh_token")
            if not new_at:
                return None
            prefs.ctrader_access_token = new_at
            if new_rt:
                prefs.ctrader_refresh_token = new_rt  # MUST persist rotated token
            _refresh_denied.pop(user_id, None)
            db.commit()
            logger.info(f"[cTrader] access token refreshed + persisted for user {user_id}")
            return new_at
        except Exception as e:
            # Sanitized: SQLAlchemy errors can include bound parameter values
            # (which may be token columns) — log only the exception type.
            logger.warning(
                f"[cTrader] token refresh persist error for user {user_id}: "
                f"{type(e).__name__}"
            )
            try:
                db.rollback()
            except Exception:
                pass
            return None
        finally:
            db.close()


async def get_accounts_for_token(access_token: str, host: str = CTRADER_HOST) -> list:
    """
    Return a list of cTrader trading accounts linked to this access token.
    Each item: {ctidTraderAccountId, isLive, traderLogin, balance}.
    Uses the persistent connection so the SSL handshake is amortised.
    """
    if not _PROTO_OK:
        return []
    async with _get_account_lock(host, 0):
        for attempt in (1, 2):
            try:
                reader, writer = await _get_persistent_connection(host, 0)
                req = ProtoOAGetAccountListByAccessTokenReq()
                req.accessToken = access_token
                payload = await _send_recv(
                    reader, writer, req,
                    _PAYLOAD_TYPES["account_list_req"],
                    _PAYLOAD_TYPES["account_list_res"],
                )
                if not payload:
                    return []
                res = ProtoOAGetAccountListByAccessTokenRes()
                res.ParseFromString(payload)
                _touch_conn(host, 0)
                accounts = []
                for acc in res.ctidTraderAccount:
                    accounts.append({
                        "ctidTraderAccountId": acc.ctidTraderAccountId,
                        "isLive":              acc.isLive,
                        "traderLogin":         getattr(acc, "traderLogin", 0),
                    })
                return accounts
            except Exception as e:
                if attempt == 1:
                    logger.warning(f"[cTrader] get_accounts retry after: {e}")
                    _invalidate_persistent_connection(host, 0)
                    continue
                logger.error(f"[cTrader] get_accounts failed: {e}")
                return []
    return []


async def _resolve_symbol_id(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    ctid_trader_account_id: int,
    broker_symbol: str,
    host: str = CTRADER_HOST,
) -> Optional[int]:
    """Resolve a broker symbol NAME → numeric symbolId for this (host, account).

    cTrader's ProtoOANewOrderReq has no symbolName field — it requires the
    integer symbolId, which differs per account/host. The full name→id map is
    fetched once over the (already app+account authed) persistent connection and
    cached per (host, ctid). Returns None if the symbol isn't tradable here.
    """
    cache_key = (host, ctid_trader_account_id)
    name_to_id = _SYMBOL_ID_CACHE.get(cache_key)
    if not name_to_id:
        req = ProtoOASymbolsListReq()
        req.ctidTraderAccountId = ctid_trader_account_id
        req.includeArchivedSymbols = False
        payload = await _send_recv(
            reader, writer, req,
            _PAYLOAD_TYPES["symbols_list_req"],
            _PAYLOAD_TYPES["symbols_list_res"],
            timeout=20.0,
        )
        if not payload:
            return None
        res = ProtoOASymbolsListRes()
        res.ParseFromString(payload)
        name_to_id = {s.symbolName: s.symbolId for s in res.symbol}
        if name_to_id:
            _SYMBOL_ID_CACHE[cache_key] = name_to_id
    return name_to_id.get(broker_symbol) if name_to_id else None


async def _resolve_symbol_details(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    ctid_trader_account_id: int,
    symbol_id: int,
    host: str = CTRADER_HOST,
) -> Optional[Dict[str, int]]:
    """Fetch a symbol's volume metadata → {lotSize, minVolume, maxVolume, stepVolume}.

    The order `volume` field is NOT in lots — it is `volume_lots × lotSize`, where
    lotSize is per-instrument and per-broker (returned by ProtoOASymbolByIdReq, the
    full ProtoOASymbol entity; the lightweight ProtoOASymbolsList does NOT carry it).
    Cached per (host, ctid, symbolId). Returns None if it can't be resolved — the
    caller then FAILS CLOSED (refuses the live order, falls back to a paper trade)
    rather than guessing a size.
    """
    cache_key = (host, ctid_trader_account_id, symbol_id)
    cached = _SYMBOL_DETAILS_CACHE.get(cache_key)
    if cached:
        return cached
    try:
        req = ProtoOASymbolByIdReq()
        req.ctidTraderAccountId = ctid_trader_account_id
        req.symbolId.append(symbol_id)
        payload = await _send_recv(
            reader, writer, req,
            _PAYLOAD_TYPES["symbol_by_id_req"],
            _PAYLOAD_TYPES["symbol_by_id_res"],
            timeout=20.0,
        )
        if not payload:
            return None
        res = ProtoOASymbolByIdRes()
        res.ParseFromString(payload)
        if not res.symbol:
            return None
        s = res.symbol[0]
        details = {
            "lotSize":    int(s.lotSize)    if s.HasField("lotSize")    else 0,
            "minVolume":  int(s.minVolume)  if s.HasField("minVolume")  else 0,
            "maxVolume":  int(s.maxVolume)  if s.HasField("maxVolume")  else 0,
            "stepVolume": int(s.stepVolume) if s.HasField("stepVolume") else 0,
        }
        if details["lotSize"] > 0:
            _SYMBOL_DETAILS_CACHE[cache_key] = details
        return details
    except Exception as e:
        logger.warning(f"[cTrader] symbol details fetch failed (symbolId={symbol_id}): {e}")
        return None


def _compute_volume(volume_lots: float, details: Optional[Dict[str, int]]) -> Optional[int]:
    """Convert lots → cTrader order-volume units using the symbol's lotSize.

    volume = volume_lots × lotSize, aligned to the broker's valid-volume grid
    (multiples of stepVolume anchored at minVolume) and clamped to [minVolume,
    maxVolume].

    Returns None when symbol details are unavailable or no valid on-grid volume
    exists. The caller MUST treat None as a hard failure and NOT place the order —
    we deliberately do NOT fall back to a flat lots×100_000 guess, because that
    scaling is wrong for metals (gold lot = 100oz) and re-introduces the exact
    NOT_ENOUGH_MONEY oversize this fix exists to prevent.
    """
    if not details or details.get("lotSize", 0) <= 0:
        return None
    lot_size  = details["lotSize"]
    step      = details.get("stepVolume", 0) or 0
    min_vol   = details.get("minVolume", 0) or 0
    max_vol   = details.get("maxVolume", 0) or 0

    vol = int(round(volume_lots * lot_size))

    # cTrader valid volumes form a grid: minVolume + k·stepVolume (k≥0) ≤ maxVolume.
    # Anchor alignment on minVolume so the result is always on-grid and ≥ min,
    # rather than snapping to a raw step multiple that the broker may reject.
    if step > 0:
        base = min_vol if min_vol > 0 else step
        if vol <= base:
            vol = base
        else:
            vol = base + round((vol - base) / step) * step
        if max_vol > 0 and vol > max_vol:
            vol = base + ((max_vol - base) // step) * step   # largest on-grid ≤ max
    else:
        if min_vol > 0 and vol < min_vol:
            vol = min_vol
        if max_vol > 0 and vol > max_vol:
            vol = max_vol

    if min_vol > 0 and vol < min_vol:
        return None   # cannot satisfy broker minimum on this grid
    return vol if vol > 0 else None


async def _await_order_fill(
    reader,
    writer,
    first_payload: bytes,
    entry_price: Optional[float],
    broker_symbol: str,
    volume_units: int,
    host: str,
    ctid_trader_account_id: int,
) -> dict:
    """Drain execution events until FILLED or timeout; shared by forex + index orders."""
    _terminal = (
        ProtoOAExecutionType.ORDER_REJECTED,
        ProtoOAExecutionType.ORDER_CANCELLED,
        ProtoOAExecutionType.ORDER_EXPIRED,
    )

    def _parse_exec(_payload):
        _e = ProtoOAExecutionEvent()
        _e.ParseFromString(_payload)
        return _e

    def _terminal_err(_e):
        if _e.executionType not in _terminal:
            return None
        _ec = _e.errorCode if _e.HasField("errorCode") else ""
        _tname = ProtoOAExecutionType.Name(_e.executionType)
        logger.error(
            f"[cTrader] order not filled ({_tname}) errorCode={_ec!r} "
            f"symbol={broker_symbol} vol={volume_units}"
        )
        return {"order_id": None, "actual_fill": None,
                "error": f"{_tname}: {_ec}" if _ec else _tname}

    ev = _parse_exec(first_payload)
    _err = _terminal_err(ev)
    if _err:
        return _err

    order_id = str(ev.order.orderId) if ev.HasField("order") else None

    def _has_fill(_e):
        return _e.HasField("deal") and bool(_e.deal.executionPrice)

    def _matches_order(_e):
        if not order_id:
            return True
        _oid = None
        try:
            if _e.HasField("order") and _e.order.orderId:
                _oid = str(_e.order.orderId)
            elif _e.HasField("deal") and getattr(_e.deal, "orderId", 0):
                _oid = str(_e.deal.orderId)
        except Exception:
            return False
        return _oid is not None and _oid == order_id

    if not _has_fill(ev):
        _fill_deadline = time.monotonic() + 15.0
        while True:
            _remaining = _fill_deadline - time.monotonic()
            if _remaining <= 0:
                break
            _pt2, _payload2 = await _recv_until(
                reader,
                {_PAYLOAD_TYPES["execution_event"],
                 _PAYLOAD_TYPES["order_error_event"]},
                timeout=_remaining,
            )
            if not _payload2:
                break
            if _pt2 == _PAYLOAD_TYPES["order_error_event"]:
                _err2 = ProtoOAOrderErrorEvent()
                _err2.ParseFromString(_payload2)
                if order_id and _err2.HasField("orderId") and str(_err2.orderId) != order_id:
                    continue
                _reason2 = (_err2.description or "").strip() or (_err2.errorCode or "").strip() or "order rejected"
                _detail2 = f"{_err2.errorCode}: {_reason2}" if _err2.errorCode and _err2.errorCode not in _reason2 else _reason2
                return {"order_id": None, "actual_fill": None, "error": _detail2}
            ev2 = _parse_exec(_payload2)
            if not _matches_order(ev2):
                continue
            _err2 = _terminal_err(ev2)
            if _err2:
                return _err2
            if ev2.HasField("order") and not order_id:
                order_id = str(ev2.order.orderId)
            if _has_fill(ev2):
                ev = ev2
                break
        if not _has_fill(ev):
            logger.warning(
                f"[cTrader] order {order_id} accepted but no fill event seen "
                f"in budget (symbol={broker_symbol})"
            )

    actual_fill = None
    position_id = None
    if ev.HasField("deal"):
        if ev.deal.executionPrice:
            _raw = float(ev.deal.executionPrice)
            if entry_price and entry_price > 0:
                _cands = [_raw, _raw / 100.0, _raw / 1000.0, _raw / 100_000.0]
                actual_fill = min(_cands, key=lambda v: abs(v - entry_price))
            else:
                actual_fill = _raw
        if ev.deal.positionId:
            position_id = str(ev.deal.positionId)
    return {
        "order_id": order_id, "actual_fill": actual_fill,
        "position_id": position_id, "volume": volume_units, "error": None,
    }


async def place_order(
    access_token: str,
    ctid_trader_account_id: int,
    symbol_name: str,
    direction: str,
    volume_lots: float,
    stop_loss_price: Optional[float] = None,
    take_profit_price: Optional[float] = None,
    entry_price: Optional[float] = None,
    label: str = "TradeHub",
    host: str = CTRADER_HOST,
) -> dict:
    """
    Place a market order on cTrader.

    volume_lots — standard lots (1 lot = 100,000 units for FX majors).
    Returns {"order_id": str, "actual_fill": float|None, "error": str|None}.

    Uses a persistent SSL connection — SSL handshake + app auth happen once,
    not on every order, saving ~300-500 ms per trade.
    """
    if not _PROTO_OK:
        return {"order_id": None, "actual_fill": None, "error": "ctrader proto not available"}
    if not CTRADER_CLIENT_ID or not CTRADER_CLIENT_SECRET:
        return {"order_id": None, "actual_fill": None, "error": "CTRADER_CLIENT_ID/SECRET not set"}

    async with _get_account_lock(host, ctid_trader_account_id):
        for attempt in (1, 2):
            try:
                reader, writer = await _get_persistent_connection(host, ctid_trader_account_id)
                if not await _account_auth(reader, writer, access_token, ctid_trader_account_id):
                    return {"order_id": None, "actual_fill": None, "error": "account auth failed"}

                broker_symbol = _SYMBOL_MAP.get(symbol_name, symbol_name)
                symbol_id = await _resolve_symbol_id(
                    reader, writer, ctid_trader_account_id, broker_symbol, host
                )
                if not symbol_id:
                    logger.error(
                        f"[cTrader] symbol {broker_symbol!r} not tradable on account "
                        f"{ctid_trader_account_id} (no symbolId resolved) — cannot place order"
                    )
                    return {"order_id": None, "actual_fill": None,
                            "error": f"symbol {broker_symbol} not tradable on this account"}

                # Volume is in cTrader's per-symbol units (volume_lots × lotSize),
                # NOT a flat lots×100_000 — that wildly over-sizes metals (gold lot
                # = 100oz, not 100k units) → NOT_ENOUGH_MONEY. Resolve the symbol's
                # lotSize/min/step and size correctly; if it can't be resolved we
                # FAIL CLOSED below rather than guess a wrong size.
                details = await _resolve_symbol_details(
                    reader, writer, ctid_trader_account_id, symbol_id, host
                )
                volume_units = _compute_volume(volume_lots, details)
                if not volume_units or volume_units <= 0:
                    # Fail CLOSED: never guess the size. A wrong-scale live order
                    # is worse than no order (it caused the gold over-size). The
                    # executor will fall back to a paper trade.
                    logger.error(
                        f"[cTrader] cannot size order for {broker_symbol} "
                        f"(lots={volume_lots}, details={details}) — refusing to place"
                    )
                    return {"order_id": None, "actual_fill": None,
                            "error": "could not resolve tradable volume for symbol"}
                logger.info(
                    f"[cTrader] {broker_symbol} sizing: {volume_lots}L → vol={volume_units} "
                    f"(lotSize={details.get('lotSize') if details else None})"
                )

                req = ProtoOANewOrderReq()
                req.ctidTraderAccountId = ctid_trader_account_id
                req.symbolId            = symbol_id
                req.orderType           = ProtoOAOrderType.MARKET
                req.tradeSide           = ProtoOATradeSide.BUY if direction == "LONG" else ProtoOATradeSide.SELL
                req.volume              = volume_units
                req.label               = label[:20]
                # MARKET orders REJECT absolute SL/TP ("SL/TP in absolute values
                # are allowed only for order types: [LIMIT, STOP, STOP_LIMIT]").
                # Use RELATIVE distances instead — a positive offset in 1/100_000
                # price units (same scaling place_order already uses for prices).
                # cTrader applies them to the ACTUAL fill price, so the SL/TP track
                # the real entry rather than a possibly-slipped signal price.
                if entry_price and entry_price > 0:
                    if stop_loss_price is not None:
                        req.relativeStopLoss = max(
                            1, int(round(abs(entry_price - stop_loss_price) * 100_000))
                        )
                    if take_profit_price is not None:
                        req.relativeTakeProfit = max(
                            1, int(round(abs(entry_price - take_profit_price) * 100_000))
                        )
                elif stop_loss_price is not None or take_profit_price is not None:
                    # This is always a MARKET order, which REQUIRES relative SL/TP
                    # (absolute is rejected). Without an entry reference we can't
                    # compute the offset — fail explicitly rather than send an
                    # absolute SL/TP that the broker will reject anyway.
                    logger.error(
                        "[cTrader] cannot place MARKET order with SL/TP — entry_price "
                        f"missing (symbol={symbol_name})"
                    )
                    return {"order_id": None, "actual_fill": None,
                            "error": "entry_price required for SL/TP on market order"}

                pt, payload = await _send_recv_any(
                    reader, writer, req,
                    _PAYLOAD_TYPES["new_order_req"],
                    {_PAYLOAD_TYPES["execution_event"],
                     _PAYLOAD_TYPES["order_error_event"]},
                    timeout=15.0,
                )
                if not payload:
                    return {"order_id": None, "actual_fill": None, "error": "no execution event"}

                _touch_conn(host, ctid_trader_account_id)

                # Broker rejected the order outright (unknown symbol, bad volume,
                # invalid SL/TP, trading disabled, market closed, no margin, …).
                # Surface the real reason instead of dropping it as "no event".
                if pt == _PAYLOAD_TYPES["order_error_event"]:
                    err = ProtoOAOrderErrorEvent()
                    err.ParseFromString(payload)
                    _reason = (err.description or "").strip() or (err.errorCode or "").strip() or "order rejected"
                    _detail = f"{err.errorCode}: {_reason}" if err.errorCode and err.errorCode not in _reason else _reason
                    logger.error(
                        f"[cTrader] order REJECTED errorCode={err.errorCode!r} "
                        f"desc={err.description!r} symbol={broker_symbol} vol={req.volume}"
                    )
                    return {"order_id": None, "actual_fill": None, "error": _detail}

                return await _await_order_fill(
                    reader, writer, payload, entry_price, broker_symbol, req.volume,
                    host, ctid_trader_account_id,
                )

            except Exception as e:
                if attempt == 1:
                    logger.warning(f"[cTrader] place_order retry after: {e}")
                    _invalidate_persistent_connection(host, ctid_trader_account_id)
                    continue
                logger.error(f"[cTrader] place_order failed: {e}")
                return {"order_id": None, "actual_fill": None, "error": str(e)}
    return {"order_id": None, "actual_fill": None, "error": "unexpected exit"}


async def place_order_units(
    access_token: str,
    ctid_trader_account_id: int,
    symbol_name: str,
    direction: str,
    volume_units: int,
    stop_loss_price: Optional[float] = None,
    take_profit_price: Optional[float] = None,
    entry_price: Optional[float] = None,
    label: str = "TradeHub",
    host: str = CTRADER_HOST,
) -> dict:
    """
    Place a market order sized in contracts / index units.

    ``volume_units`` is the contract count (e.g. 1 NAS100 contract). Converted
    to cTrader wire volume via the symbol's lotSize grid — same path as forex
    lots, but callers pass whole contracts instead of fractional lots.
    """
    if not _PROTO_OK:
        return {"order_id": None, "actual_fill": None, "error": "ctrader proto not available"}
    if not CTRADER_CLIENT_ID or not CTRADER_CLIENT_SECRET:
        return {"order_id": None, "actual_fill": None, "error": "CTRADER_CLIENT_ID/SECRET not set"}

    contracts = max(1, int(volume_units or 1))
    broker_symbol = _SYMBOL_MAP.get(symbol_name, symbol_name)

    async with _get_account_lock(host, ctid_trader_account_id):
        for attempt in (1, 2):
            try:
                reader, writer = await _get_persistent_connection(host, ctid_trader_account_id)
                if not await _account_auth(reader, writer, access_token, ctid_trader_account_id):
                    return {"order_id": None, "actual_fill": None, "error": "account auth failed"}

                symbol_id = await _resolve_symbol_id(
                    reader, writer, ctid_trader_account_id, broker_symbol, host
                )
                if not symbol_id:
                    return {"order_id": None, "actual_fill": None,
                            "error": f"symbol {broker_symbol} not tradable on this account"}

                details = await _resolve_symbol_details(
                    reader, writer, ctid_trader_account_id, symbol_id, host
                )
                vol = _compute_volume(float(contracts), details)
                if not vol or vol <= 0:
                    return {"order_id": None, "actual_fill": None,
                            "error": "could not resolve tradable volume for symbol"}

                logger.info(
                    f"[cTrader] {broker_symbol} index sizing: {contracts} contracts → vol={vol}"
                )

                req = ProtoOANewOrderReq()
                req.ctidTraderAccountId = ctid_trader_account_id
                req.symbolId            = symbol_id
                req.orderType           = ProtoOAOrderType.MARKET
                req.tradeSide           = ProtoOATradeSide.BUY if direction == "LONG" else ProtoOATradeSide.SELL
                req.volume              = vol
                req.label               = label[:20]
                if entry_price and entry_price > 0:
                    if stop_loss_price is not None:
                        req.relativeStopLoss = max(
                            1, int(round(abs(entry_price - stop_loss_price) * 100_000))
                        )
                    if take_profit_price is not None:
                        req.relativeTakeProfit = max(
                            1, int(round(abs(entry_price - take_profit_price) * 100_000))
                        )

                pt, payload = await _send_recv_any(
                    reader, writer, req,
                    _PAYLOAD_TYPES["new_order_req"],
                    {_PAYLOAD_TYPES["execution_event"],
                     _PAYLOAD_TYPES["order_error_event"]},
                    timeout=15.0,
                )
                if not payload:
                    return {"order_id": None, "actual_fill": None, "error": "no execution event"}

                _touch_conn(host, ctid_trader_account_id)

                if pt == _PAYLOAD_TYPES["order_error_event"]:
                    err = ProtoOAOrderErrorEvent()
                    err.ParseFromString(payload)
                    _reason = (err.description or "").strip() or (err.errorCode or "").strip() or "order rejected"
                    _detail = f"{err.errorCode}: {_reason}" if err.errorCode and err.errorCode not in _reason else _reason
                    return {"order_id": None, "actual_fill": None, "error": _detail}

                return await _await_order_fill(
                    reader, writer, payload, entry_price, broker_symbol, vol,
                    host, ctid_trader_account_id,
                )
            except Exception as e:
                if attempt == 1:
                    logger.warning(f"[cTrader] place_order_units retry after: {e}")
                    _invalidate_persistent_connection(host, ctid_trader_account_id)
                    continue
                logger.error(f"[cTrader] place_order_units failed: {e}")
                return {"order_id": None, "actual_fill": None, "error": str(e)}
    return {"order_id": None, "actual_fill": None, "error": "unexpected exit"}


async def close_position(
    access_token: str,
    ctid_trader_account_id: int,
    position_id: int,
    volume_units: int,
    host: str = CTRADER_HOST,
) -> bool:
    """Close (or partially close) an open position. Uses persistent connection."""
    if not _PROTO_OK:
        return False
    async with _get_account_lock(host, ctid_trader_account_id):
        for attempt in (1, 2):
            try:
                reader, writer = await _get_persistent_connection(host, ctid_trader_account_id)
                if not await _account_auth(reader, writer, access_token, ctid_trader_account_id):
                    return False
                req = ProtoOAClosePositionReq()
                req.ctidTraderAccountId = ctid_trader_account_id
                req.positionId          = position_id
                req.volume              = volume_units
                payload = await _send_recv(
                    reader, writer, req,
                    _PAYLOAD_TYPES["close_position_req"],
                    _PAYLOAD_TYPES["execution_event"],
                    timeout=15.0,
                )
                if payload is not None:
                    _touch_conn(host, ctid_trader_account_id)
                return payload is not None
            except Exception as e:
                if attempt == 1:
                    logger.warning(f"[cTrader] close_position retry after: {e}")
                    _invalidate_persistent_connection(host, ctid_trader_account_id)
                    continue
                logger.error(f"[cTrader] close_position failed: {e}")
                return False
    return False


async def modify_position_sltp(
    access_token: str,
    ctid_trader_account_id: int,
    position_id: int,
    stop_loss_price: Optional[float] = None,
    take_profit_price: Optional[float] = None,
    host: str = CTRADER_HOST,
) -> bool:
    """
    Amend the stop-loss and/or take-profit of an existing open position on the
    broker (server-side). Used for auto-breakeven and trailing-stop on live
    forex trades — without this the broker keeps the original SL forever.

    Prices are sent as REAL absolute prices (e.g. gold 4452.76, EURUSD 1.0850).
    The proto stopLoss/takeProfit are `double` fields and the broker quotes
    absolute prices unscaled (proven live: a gold stop of ~4452 was accepted while
    4452*100000 would be rejected). place_order's ×100000 is the RELATIVE-offset
    convention (a different field), NOT applicable to these absolute fields.
    Returns True on success.
    """
    if not _PROTO_OK:
        return False
    if stop_loss_price is None and take_profit_price is None:
        return False
    async with _get_account_lock(host, ctid_trader_account_id):
        for attempt in (1, 2):
            try:
                reader, writer = await _get_persistent_connection(host, ctid_trader_account_id)
                if not await _account_auth(reader, writer, access_token, ctid_trader_account_id):
                    return False
                req = ProtoOAAmendPositionSLTPReq()
                req.ctidTraderAccountId = ctid_trader_account_id
                req.positionId          = position_id
                # ABSOLUTE price fields (double) → send the REAL price unscaled.
                # The old ×100_000 was wrong: it set gold breakeven stops to
                # 4452.76*100000 (rejected by the broker), so live breakeven/
                # trailing silently never reached the broker for metals — and on
                # one trade a corrupted entry (0.0445276) ×100000 ≈ 4452 was
                # accepted by accident, firing a false "risk-free" alert on a
                # position whose stop was never actually moved.
                if stop_loss_price is not None:
                    req.stopLoss = float(stop_loss_price)
                if take_profit_price is not None:
                    req.takeProfit = float(take_profit_price)
                payload = await _send_recv(
                    reader, writer, req,
                    _PAYLOAD_TYPES["amend_position_sltp_req"],
                    _PAYLOAD_TYPES["execution_event"],
                    timeout=15.0,
                )
                if payload is not None:
                    _touch_conn(host, ctid_trader_account_id)
                return payload is not None
            except Exception as e:
                if attempt == 1:
                    logger.warning(f"[cTrader] modify_position_sltp retry after: {e}")
                    _invalidate_persistent_connection(host, ctid_trader_account_id)
                    continue
                logger.error(f"[cTrader] modify_position_sltp failed: {e}")
                return False
    return False


async def modify_position_sltp_for_user(
    user,
    position_id: int,
    stop_loss_price: Optional[float] = None,
    take_profit_price: Optional[float] = None,
) -> bool:
    """High-level wrapper: amend a live position's SL/TP for a TradeHub user."""
    from app.database import SessionLocal
    from app.models import UserPreference
    db = SessionLocal()
    try:
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs or not prefs.ctrader_access_token or not prefs.ctrader_account_id:
            return False
        access_token           = prefs.ctrader_access_token
        ctid_trader_account_id = int(prefs.ctrader_account_id)
        _host = _host_for_account(prefs, ctid_trader_account_id)
    finally:
        db.close()
    return await modify_position_sltp(
        access_token, ctid_trader_account_id, int(position_id),
        stop_loss_price=stop_loss_price, take_profit_price=take_profit_price,
        host=_host,
    )


async def close_partial_position_for_user(
    user,
    symbol: str,
    position_id: int,
    total_volume_units: int,
    fraction: float,
) -> int:
    """Partially close a live position for a TradeHub user (TP1 partial profit).

    Closes ~``fraction`` of ``total_volume_units`` (the volume the position was
    opened with), aligned DOWN to the symbol's broker volume grid so BOTH the
    closed slice and the remaining slice stay ≥ minVolume.

    Return contract (the caller distinguishes these three outcomes):
      * ``> 0`` → success, the actual number of units closed.
      * ``-1`` → CONFIRMED un-splittable on the broker grid (position too small to
        split into two ≥ minVolume slices). The caller should skip the partial
        permanently and just run the full position to TP2.
      * ``0``  → TRANSIENT failure (broker unreachable, creds missing, symbol
        detail fetch failed, or the close request was rejected). The caller should
        leave the position eligible and retry on the next cycle — do NOT record a
        breakeven move or mark the partial skipped.
    """
    if not _PROTO_OK:
        return 0
    if total_volume_units <= 0 or fraction <= 0 or fraction >= 1:
        return 0

    from app.database import SessionLocal
    from app.models import UserPreference
    db = SessionLocal()
    try:
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs or not prefs.ctrader_access_token or not prefs.ctrader_account_id:
            return 0
        access_token = prefs.ctrader_access_token
        ctid = int(prefs.ctrader_account_id)
        host = _host_for_account(prefs, ctid)
    finally:
        db.close()

    # Resolve the symbol's volume grid so the partial close lands on a valid step.
    details = None
    try:
        async with _get_account_lock(host, ctid):
            reader, writer = await _get_persistent_connection(host, ctid)
            if await _account_auth(reader, writer, access_token, ctid):
                broker_symbol = _SYMBOL_MAP.get(symbol, symbol)
                symbol_id = await _resolve_symbol_id(reader, writer, ctid, broker_symbol, host)
                if symbol_id:
                    details = await _resolve_symbol_details(reader, writer, ctid, symbol_id, host)
    except Exception as e:
        logger.warning(f"[cTrader] partial-close detail fetch failed for {symbol}: {e}")
        details = None

    # Unknown grid (detail fetch failed) → TRANSIENT: never guess the volume grid.
    if details is None:
        return 0

    step    = details.get("stepVolume", 0) or 0
    min_vol = details.get("minVolume", 0) or 0

    close_vol = int(total_volume_units * fraction)
    if step > 0:
        # Round DOWN to a step multiple so the closed slice never exceeds the target.
        close_vol = (close_vol // step) * step
    # Both slices must satisfy the broker minimum, else the close (or the residual
    # position) would be rejected — CONFIRMED un-splittable (skip the partial).
    if min_vol > 0:
        if close_vol < min_vol or (total_volume_units - close_vol) < min_vol:
            return -1
    if close_vol <= 0 or close_vol >= total_volume_units:
        return -1

    ok = await close_position(access_token, ctid, int(position_id), int(close_vol), host=host)
    return int(close_vol) if ok else 0  # 0 = transient close rejection → retry


# ── High-level: place order for a TradeHub user ───────────────────────────────

async def _get_account_balance(
    access_token: str,
    ctid_trader_account_id: int,
    host: str = CTRADER_HOST,
) -> Optional[float]:
    """
    Fetch the current account balance (equity) from cTrader via ProtoOAReconcileReq.
    Returns USD balance or None on failure.
    """
    if not _PROTO_OK:
        return None
    cache_key = _acct_conn_key(host, ctid_trader_account_id)
    cached = _balance_cache.get(cache_key)
    if cached and (time.monotonic() - cached[1]) < _BALANCE_CACHE_TTL:
        return cached[0]
    try:
        async with _get_account_lock(host, ctid_trader_account_id):
            try:
                reader, writer = await _get_persistent_connection(host, ctid_trader_account_id)
                if not await _account_auth(reader, writer, access_token, ctid_trader_account_id):
                    return None
                req = ProtoOAReconcileReq()
                req.ctidTraderAccountId = ctid_trader_account_id
                payload = await _send_recv(
                    reader, writer, req,
                    _PAYLOAD_TYPES["reconcile_req"],
                    _PAYLOAD_TYPES["reconcile_res"],
                    timeout=10.0,
                )
                if not payload:
                    return None
                res = ProtoOAReconcileRes()
                res.ParseFromString(payload)
                if hasattr(res, "balance") and res.balance:
                    bal = res.balance / 100.0
                    _balance_cache[cache_key] = (bal, time.monotonic())
                    return bal
            except asyncio.CancelledError:
                _invalidate_persistent_connection(host, ctid_trader_account_id)
                raise
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug(f"[cTrader] balance fetch error: {e}")
    return None


async def _get_open_position_ids(
    access_token: str,
    ctid_trader_account_id: int,
    host: str = CTRADER_HOST,
) -> Optional[set]:
    """Fetch the set of currently-open broker positionIds via ProtoOAReconcileReq.

    Returns a set of ints (possibly empty) on success, or None on ANY failure so
    the caller can distinguish "broker says zero open positions" from "broker
    unreachable" and never false-close a tracked execution.
    """
    if not _PROTO_OK:
        return None
    try:
        async with _get_account_lock(host, ctid_trader_account_id):
            try:
                reader, writer = await _get_persistent_connection(host, ctid_trader_account_id)
                if not await _account_auth(reader, writer, access_token, ctid_trader_account_id):
                    return None
                req = ProtoOAReconcileReq()
                req.ctidTraderAccountId = ctid_trader_account_id
                payload = await _send_recv(
                    reader, writer, req,
                    _PAYLOAD_TYPES["reconcile_req"],
                    _PAYLOAD_TYPES["reconcile_res"],
                    timeout=10.0,
                )
                if not payload:
                    return None
                res = ProtoOAReconcileRes()
                res.ParseFromString(payload)
                return {int(p.positionId) for p in res.position}
            except asyncio.CancelledError:
                _invalidate_persistent_connection(host, ctid_trader_account_id)
                raise
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug(f"[cTrader] open-positions fetch error: {e}")
    return None


async def get_open_position_ids_for_user(user) -> Optional[set]:
    """High-level wrapper: the set of open broker positionIds for a TradeHub user,
    or None when credentials are missing / the broker is unreachable (so callers
    never mistake an error for "all positions closed")."""
    from app.database import SessionLocal
    from app.models import UserPreference
    db = SessionLocal()
    try:
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs or not prefs.ctrader_access_token or not prefs.ctrader_account_id:
            return None
        access_token           = prefs.ctrader_access_token
        ctid_trader_account_id = int(prefs.ctrader_account_id)
        _host = _host_for_account(prefs, ctid_trader_account_id)
    finally:
        db.close()
    return await _get_open_position_ids(access_token, ctid_trader_account_id, host=_host)


async def place_ctrader_order_for_user(
    user,
    symbol: str,
    direction: str,
    entry_price: float,
    tp_pct: float,
    sl_pct: float,
    risk_pct: float = 1.0,
    risk_usd: Optional[float] = None,
    use_risk_pct: bool = False,
    sl_pips: Optional[float] = None,
    fixed_lots: Optional[float] = None,
) -> Optional[dict]:
    """
    Place a live forex or index CFD order for a user via their connected cTrader account.
    Converts TP/SL from % to absolute prices, calculates lot/contract size from risk.

    Sizing priority:
      1. fixed_lots — explicit lot size chosen by the user (forex) or contracts (index).
      2. use_risk_pct + sl_pips — "fixed fractional":
           lots = (risk_pct% × account_balance) / (sl_pips × pip_value_per_lot)
      3. risk_usd — lots = risk_usd / (sl_pips × pip_value_per_lot)
      4. fallback to 0.01 lot minimum.

    Returns {"order_id", "actual_fill"} or None on failure.
    """
    from app.database import SessionLocal
    from app.models import UserPreference
    db = SessionLocal()
    try:
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs or not prefs.ctrader_access_token or not prefs.ctrader_account_id:
            logger.warning(f"[cTrader] user {user.id} has no cTrader credentials")
            return None
        access_token           = prefs.ctrader_access_token
        ctid_trader_account_id = int(prefs.ctrader_account_id)
        primary_host           = _host_for_account(prefs, ctid_trader_account_id)
    finally:
        db.close()

    mult     = 1.0 if direction == "LONG" else -1.0
    tp_price = round(entry_price * (1 + mult * tp_pct / 100), 6)
    sl_price = round(entry_price * (1 - mult * sl_pct / 100), 6)

    try:
        from app.services.index_symbols import normalize_index_symbol, is_index_symbol
        is_index = is_index_symbol(symbol)
        symbol = normalize_index_symbol(symbol) if is_index else symbol.upper()
    except Exception:
        is_index = symbol.upper() in _INDEX_SYMBOLS

    if is_index:
        # Index CFDs: volume in contracts (1 contract ≈ 1 unit of index value).
        # Size by explicit lots/contracts if provided, else risk_usd, else 1 contract.
        if fixed_lots and fixed_lots > 0:
            contracts = max(1, round(fixed_lots))
        elif risk_usd and risk_usd > 0:
            # contracts ≈ risk_usd / (sl_pct% of price)
            sl_value_per_contract = entry_price * (sl_pct / 100)
            contracts = max(1, round(risk_usd / max(sl_value_per_contract, 0.01)))
        else:
            contracts = 1
        contracts = max(1, min(contracts, 500))  # clamp 1–500 contracts
        logger.info(
            f"[cTrader] placing {direction} {symbol} (index) {contracts} contracts "
            f"TP={tp_price} SL={sl_price} for user {user.id}"
        )
        result = await place_order_units(
            access_token           = access_token,
            ctid_trader_account_id = ctid_trader_account_id,
            symbol_name            = _SYMBOL_MAP.get(symbol, symbol),
            direction              = direction,
            volume_units           = contracts,
            stop_loss_price        = sl_price,
            take_profit_price      = tp_price,
            entry_price            = entry_price,
            host                   = primary_host,
        )
    else:
        # Forex: standard lot sizing
        pip       = _PIP_SIZES.get(symbol, 0.0001)
        # Compute effective SL in pips for lot sizing
        _sl_pips_eff = sl_pips if sl_pips and sl_pips > 0 else abs(entry_price - sl_price) / max(pip, 1e-10)
        # pip_value_per_lot: USD P&L per pip per standard lot
        # For USD-quoted pairs (EURUSD, GBPUSD…): 10 USD/pip/lot
        # For JPY pairs: ~9.28 USD (yen-based, approx at USDJPY≈148)
        # For XAU: $10/pip/lot (XAUUSD pip=0.10, lot=100oz → 100×0.10=$10)
        if symbol.upper() in ("XAUUSD",):
            pip_value = 10.0   # $10/pip/lot (100oz lot, pip=0.10)
        elif symbol.upper() in ("XAGUSD",):
            pip_value = 5.0    # $5/pip/lot approx (5000oz lot, pip=0.001)
        elif "JPY" in symbol.upper():
            pip_value = 9.28   # approx USD/pip/lot at USDJPY~148
        else:
            pip_value = 10.0   # standard USD/pip/lot for majors

        if fixed_lots and fixed_lots > 0:
            # ── Explicit lot size chosen by the user ──────────────────────
            lots = round(fixed_lots, 2)
            logger.info(f"[cTrader] fixed lot sizing: {lots}L (user-chosen)")
        elif use_risk_pct and risk_pct and risk_pct > 0:
            # ── Risk % auto lot sizing ─────────────────────────────────────
            # Fetch live account balance; compute lots from risk fraction.
            account_balance = await _get_account_balance(access_token, ctid_trader_account_id, host=primary_host)
            if account_balance and account_balance > 0:
                risk_amount = account_balance * (risk_pct / 100.0)
                lots = round(risk_amount / max(_sl_pips_eff * pip_value, 0.01), 2)
                logger.info(
                    f"[cTrader] risk% lot sizing: balance={account_balance:.2f} "
                    f"risk%={risk_pct} risk_usd={risk_amount:.2f} "
                    f"sl_pips={_sl_pips_eff:.1f} pip_val={pip_value} → {lots}L"
                )
            else:
                logger.warning(
                    f"[cTrader] balance fetch failed for user {user.id} — "
                    f"falling back to 0.01 lot minimum"
                )
                lots = 0.01
        elif risk_usd and risk_usd > 0:
            lots = round(risk_usd / max(_sl_pips_eff * pip_value, 0.01), 2)
        else:
            lots = 0.01
        lots = max(0.01, min(lots, 50.0))
        logger.info(
            f"[cTrader] placing {direction} {symbol} {lots}L "
            f"TP={tp_price} SL={sl_price} for user {user.id}"
        )
        result = await place_order(
            access_token           = access_token,
            ctid_trader_account_id = ctid_trader_account_id,
            symbol_name            = symbol,
            direction              = direction,
            volume_lots            = lots,
            stop_loss_price        = sl_price,
            take_profit_price      = tp_price,
            entry_price            = entry_price,
            host                   = primary_host,
        )

    # Local closure so the token-refresh AND wrong-host retries reuse the exact
    # same sizing/branch without duplicating the index/forex logic.
    async def _retry_place(_at: str, _host: str) -> dict:
        if is_index:
            return await place_order_units(
                access_token           = _at,
                ctid_trader_account_id = ctid_trader_account_id,
                symbol_name            = _SYMBOL_MAP.get(symbol, symbol),
                direction              = direction,
                volume_units           = contracts,
                stop_loss_price        = sl_price,
                take_profit_price      = tp_price,
                entry_price            = entry_price,
                host                   = _host,
            )
        return await place_order(
            access_token           = _at,
            ctid_trader_account_id = ctid_trader_account_id,
            symbol_name            = symbol,
            direction              = direction,
            volume_lots            = lots,
            stop_loss_price        = sl_price,
            take_profit_price      = tp_price,
            entry_price            = entry_price,
            host                   = _host,
        )

    # Expired access token → refresh once and retry the placement. cTrader OAuth
    # tokens expire (~30d); without this the order silently fails as "account
    # auth failed" (the original live-order bug).
    if result.get("error") == "account auth failed":
        new_at = await refresh_user_ctrader_token(user.id)
        if new_at:
            access_token = new_at
            logger.info(f"[cTrader] retrying {symbol} order for user {user.id} after token refresh")
            result = await _retry_place(access_token, primary_host)

    # Still "account auth failed" → the stored isLive metadata may be wrong/stale
    # (demo account on the live host or vice-versa). Try the OTHER host once.
    if result.get("error") == "account auth failed":
        alt_host = _other_host(primary_host)
        logger.info(f"[cTrader] retrying {symbol} for user {user.id} on alternate host {alt_host}")
        result = await _retry_place(access_token, alt_host)

    if result.get("error"):
        logger.error(f"[cTrader] order failed for user {user.id}: {result['error']}")
        # Return the dict (order_id is None) so the caller can surface the real
        # rejection reason instead of a generic "no order id" message.
        return result
    # Stamp the broker account this order was actually placed on, so close
    # reconciliation can bind the execution to that account (a later account
    # relink must not false-close a position opened on the previous account).
    if isinstance(result, dict) and result.get("order_id"):
        result["account_id"] = ctid_trader_account_id
    return result
