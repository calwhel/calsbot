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
from typing import Dict, List, Optional, Tuple

from app.services.ctrader_sltp import (
    compute_sltp_prices,
    relative_sltp_wire as _relative_sltp_wire,
    validate_sltp_sanity,
)

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


def _account_is_live(prefs, ctid: int) -> Optional[bool]:
    """Return True/False when ctid is in stored account metadata, else None.

    Missing isLive on a matched row means unknown — never assume demo (that
    blocked live ctids from routing to live.ctraderapi.com).
    """
    try:
        import json as _json
        raw = getattr(prefs, "ctrader_accounts", None)
        if raw:
            for a in _json.loads(raw):
                if int(a.get("ctidTraderAccountId", -1)) == int(ctid):
                    if "isLive" in a:
                        return bool(a.get("isLive"))
                    return None
    except Exception:
        pass
    return None


DEFAULT_ACCOUNT_LOT_MIN = 0.01
DEFAULT_ACCOUNT_LOT_STEP = 0.01


def normalize_account_lot(lots) -> Optional[float]:
    """Validate and snap a per-strategy account lot override to min/step grid."""
    if lots is None:
        return None
    try:
        v = float(lots)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    step = DEFAULT_ACCOUNT_LOT_STEP
    min_lot = DEFAULT_ACCOUNT_LOT_MIN
    if v < min_lot:
        v = min_lot
    # Align to min + k·step
    k = round((v - min_lot) / step)
    snapped = round(min_lot + k * step, 2)
    if snapped < min_lot:
        snapped = min_lot
    return snapped


def _parse_reconcile_balance(res) -> Optional[float]:
    """Extract USD balance/equity from ProtoOAReconcileRes (cents → dollars)."""
    for attr in ("balance", "equity"):
        if hasattr(res, attr):
            raw = getattr(res, attr, None)
            if raw is not None:
                return float(raw) / 100.0
    return None


def _host_for_account(prefs, ctid: int) -> str:
    """Pick live vs demo host for a ctid — never guess the alternate host."""
    is_live = _account_is_live(prefs, ctid)
    if is_live is False:
        return CTRADER_HOST_DEMO
    if is_live is True:
        return CTRADER_HOST_LIVE
    return CTRADER_HOST_LIVE


def resolve_ctrader_ctid(
    *,
    strategy_account_id: Optional[str] = None,
    execution_account_id: Optional[str] = None,
    notes: Optional[str] = None,
    prefs_default: Optional[str] = None,
) -> Optional[str]:
    """Resolve which cTrader account (ctid) to route on.

    Priority: execution column → acct= in notes → strategy binding → user default.
    """
    if execution_account_id and str(execution_account_id).strip():
        return str(execution_account_id).strip()
    if notes:
        import re as _re
        _m = _re.search(r"acct=(\d+)", notes)
        if _m:
            return _m.group(1)
    if strategy_account_id and str(strategy_account_id).strip():
        return str(strategy_account_id).strip()
    if prefs_default and str(prefs_default).strip():
        return str(prefs_default).strip()
    return None


def parse_added_accounts_json(raw) -> list:
    """Parse ctrader_added_accounts JSON — never raises."""
    if not raw:
        return []
    import json as _json
    try:
        parsed = _json.loads(raw) if isinstance(raw, str) else raw
        return [
            str(x).strip()
            for x in (parsed if isinstance(parsed, list) else [])
            if x is not None and str(x).strip()
        ]
    except Exception:
        return []


def list_added_ctrader_ctids(prefs) -> list:
    """Ctids the user explicitly added as execution targets."""
    return parse_added_accounts_json(
        getattr(prefs, "ctrader_added_accounts", None) if prefs else None
    )


def list_assignable_ctrader_ctids(prefs, *, include_default: bool = True) -> list:
    """Ctids the user can assign strategies to (added accounts + optional default)."""
    added = parse_added_accounts_json(
        getattr(prefs, "ctrader_added_accounts", None) if prefs else None
    )
    out: list = []
    seen = set()
    for ctid in added:
        if ctid not in seen:
            out.append(ctid)
            seen.add(ctid)
    default = (getattr(prefs, "ctrader_account_id", None) or "").strip() if prefs else ""
    if include_default and default and default not in seen:
        out.insert(0, default)
    elif not out and default:
        out = [default]
    return out


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
        ProtoOADealListByPositionIdReq,
        ProtoOADealListByPositionIdRes,
        ProtoOADealListReq,
        ProtoOADealListRes,
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
    "deal_list_by_position_id_req": 2179,
    "deal_list_by_position_id_res": 2180,
    "deal_list_req": 2133,
    "deal_list_res": 2134,
}

# Platform pip sizes: app.services.pip_units.platform_pip_size (forex_engine).
# Broker pipPosition metadata: pip_units.broker_pip_size_from_metadata — protocol only.

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
ORDER_SUBMIT_TIMEOUT_S = float(os.environ.get("CTRADER_ORDER_SUBMIT_TIMEOUT_S", "5"))
ORDER_FILL_WAIT_S = float(os.environ.get("CTRADER_ORDER_FILL_TIMEOUT_S", "5"))
ORDER_CONNECT_TIMEOUT_S = float(os.environ.get("CTRADER_ORDER_CONNECT_TIMEOUT_S", "5"))
CTRADER_ACCOUNT_AUTH_TIMEOUT_S = float(
    os.environ.get("CTRADER_ACCOUNT_AUTH_TIMEOUT_S", "4")
)
CTRADER_ACCOUNT_OP_CONNECT_TIMEOUT_S = float(
    os.environ.get("CTRADER_ACCOUNT_OP_CONNECT_TIMEOUT_S", "4")
)
CTRADER_OPEN_POS_CALL_TIMEOUT_S = float(
    os.environ.get("CTRADER_OPEN_POS_CALL_TIMEOUT_S", "6")
)
CTRADER_OPEN_POS_RECONCILE_TIMEOUT_S = float(
    os.environ.get("CTRADER_OPEN_POS_RECONCILE_TIMEOUT_S", "5")
)
CTRADER_DEAL_FETCH_CALL_TIMEOUT_S = float(
    os.environ.get("CTRADER_DEAL_FETCH_CALL_TIMEOUT_S", "8")
)
CTRADER_DEAL_FETCH_REQ_TIMEOUT_S = float(
    os.environ.get("CTRADER_DEAL_FETCH_REQ_TIMEOUT_S", "5")
)
CTRADER_DEAL_WINDOW_CALL_TIMEOUT_S = float(
    os.environ.get("CTRADER_DEAL_WINDOW_CALL_TIMEOUT_S", "6")
)
CTRADER_DEAL_MAX_PAGES = max(1, int(os.environ.get("CTRADER_DEAL_MAX_PAGES", "3")))
_AMBIGUOUS_ORDER_ERRORS = frozenset({
    "no execution event",
    "unexpected exit",
    "timeout",
})
_balance_cache: dict = {}    # (host, ctid) → (balance, monotonic_ts)
_BALANCE_CACHE_TTL = 60.0
BALANCE_CONNECT_TIMEOUT_S = float(os.environ.get("CTRADER_BALANCE_CONNECT_TIMEOUT_S", "10"))
BALANCE_REQ_TIMEOUT_S = float(os.environ.get("CTRADER_BALANCE_REQ_TIMEOUT_S", "10"))


def is_ambiguous_order_error(err: Optional[str]) -> bool:
    if not err:
        return False
    low = str(err).lower()
    return any(tok in low for tok in _AMBIGUOUS_ORDER_ERRORS)


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
    *,
    connect_timeout: Optional[float] = None,
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

    _cto = connect_timeout if connect_timeout is not None else ORDER_CONNECT_TIMEOUT_S

    async def _connect():
        r, w = await _open_connection(host)
        ok = await _app_auth(r, w)
        if not ok:
            await _aclose_writer(w)
            raise RuntimeError("cTrader persistent connection: app auth failed")
        return r, w

    r, w = await asyncio.wait_for(_connect(), timeout=_cto)
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
    *,
    timeout: float = CTRADER_ACCOUNT_AUTH_TIMEOUT_S,
) -> bool:
    req = ProtoOAAccountAuthReq()
    req.ctidTraderAccountId = ctid_trader_account_id
    req.accessToken         = access_token
    payload = await _send_recv(
        reader, writer, req,
        _PAYLOAD_TYPES["account_auth_req"],
        _PAYLOAD_TYPES["account_auth_res"],
        timeout=timeout,
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
    """Use a refresh token to get a new access token (retries transient timeouts)."""
    import httpx

    app_id = CTRADER_CLIENT_ID
    params = {
        "grant_type":    "refresh_token",
        "refresh_token": refresh_token,
        "client_id":     app_id,
        "client_secret": CTRADER_CLIENT_SECRET,
    }
    last_exc: Optional[Exception] = None
    for attempt in (1, 2, 3):
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(OAUTH_TOKEN_URL, params=params)
                if resp.status_code != 200:
                    try:
                        body = resp.json()
                        err_code = body.get("errorCode") or body.get("error")
                        desc = body.get("description") or body.get("error_description") or ""
                        logger.warning(
                            "[cTrader] refresh_access_token HTTP %s errorCode=%s desc=%s",
                            resp.status_code,
                            err_code,
                            str(desc)[:120],
                        )
                        if isinstance(body, dict) and err_code:
                            return body
                    except Exception:
                        logger.warning(
                            "[cTrader] refresh_access_token HTTP %s (no JSON body)",
                            resp.status_code,
                        )
                    resp.raise_for_status()
                return resp.json()
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as exc:
            last_exc = exc
            if attempt < 3:
                await asyncio.sleep(1.5 * attempt)
                continue
            raise
    if last_exc:
        raise last_exc
    return {}


# Serialise token refreshes within a process so two coroutines (feed + executor)
# don't both burn the rotating refresh token at once.
_token_refresh_lock = asyncio.Lock()

# Cross-process advisory lock namespace for token refresh (see app.lock_ids).
# cTrader ROTATES the refresh token on every use; under gunicorn -w 4 the feed,
# executor, and request workers must never refresh concurrently. The per-process
# asyncio lock above can't coordinate across processes — pg_try_advisory_xact_lock
# (namespace, user_id) serialises PER USER so one user's refresh never blocks another's.
from app.lock_ids import CTRADER_TOKEN_REFRESH_LOCK_NS as _TOKEN_REFRESH_PG_LOCK_NS

# When a specific refresh token is denied (ACCESS_DENIED → chain dead, user must
# re-link), back off instead of hammering cTrader's OAuth endpoint every signal
# tick. Keyed by user_id → (denied_refresh_token, cooldown_until_monotonic). The
# guard is scoped to the EXACT token value, so a re-link (which rotates the token
# to a new value) clears it automatically without any cross-module coupling.
_refresh_denied: dict = {}
_RELINK_ALERT_SENT: dict = {}  # user_id → monotonic
_REFRESH_DENY_COOLDOWN = 300.0  # seconds
_REFRESH_FAILURE_ROUNDS = 3
_REFRESH_RETRY_WAIT_S = 2.0
_REFRESH_NEAR_EXPIRY_S = int(os.environ.get("CTRADER_REFRESH_NEAR_EXPIRY_S", "600"))
# Scheduled early refresh — cTrader access tokens can live ~30d; refresh well before expiry.
_SCHEDULED_REFRESH_WHEN_REMAINING_S = int(
    os.environ.get("CTRADER_REFRESH_EARLY_REMAINING_S", str(5 * 86400))
)
_WARN_WHEN_REMAINING_S = int(
    os.environ.get("CTRADER_REFRESH_WARN_REMAINING_S", str(2 * 86400))
)
_AUTH_REFRESH_COOLDOWN_S = float(
    os.environ.get("CTRADER_AUTH_REFRESH_COOLDOWN_S", "30")
)
_AUTH_REFRESH_WAIT_TIMEOUT_S = float(
    os.environ.get("CTRADER_AUTH_REFRESH_WAIT_TIMEOUT_S", "12")
)
_AUTH_UNHEALTHY_COOLDOWN_S = float(
    os.environ.get("CTRADER_AUTH_UNHEALTHY_COOLDOWN_S", "90")
)
_AUTH_TIMEOUT_COOLDOWN_S = float(
    os.environ.get("CTRADER_AUTH_TIMEOUT_COOLDOWN_S", "45")
)
_REFRESH_TERMINAL_CODES = frozenset({
    "ACCESS_DENIED",
    "CH_ACCESS_TOKEN_INVALID",
    "INVALID_ACCESS_TOKEN",
    "INVALID_REFRESH_TOKEN",
    "invalid_grant",
    "INVALID_GRANT",
})
_auth_refresh_inflight: Dict[int, asyncio.Task] = {}
_auth_refresh_next_allowed: Dict[int, float] = {}
_auth_unhealthy_accounts: Dict[Tuple[int, str, int], Tuple[float, str]] = {}
_auth_state_lock = asyncio.Lock()


def _auth_account_key(user_id: int, host: str, ctid: int) -> Tuple[int, str, int]:
    return (int(user_id), str(host), int(ctid))


def _account_auth_cooldown_remaining(user_id: int, host: str, ctid: int) -> float:
    key = _auth_account_key(user_id, host, ctid)
    row = _auth_unhealthy_accounts.get(key)
    if not row:
        return 0.0
    until_mono, _reason = row
    rem = until_mono - time.monotonic()
    if rem <= 0:
        _auth_unhealthy_accounts.pop(key, None)
        return 0.0
    return rem


def _account_auth_in_cooldown(user_id: int, host: str, ctid: int) -> bool:
    return _account_auth_cooldown_remaining(user_id, host, ctid) > 0


def _mark_account_auth_unhealthy(
    user_id: int,
    host: str,
    ctid: int,
    *,
    reason: str,
    cooldown_s: Optional[float] = None,
) -> None:
    dur = max(5.0, float(cooldown_s if cooldown_s is not None else _AUTH_UNHEALTHY_COOLDOWN_S))
    now = time.monotonic()
    key = _auth_account_key(user_id, host, ctid)
    prev = _auth_unhealthy_accounts.get(key)
    prev_rem = 0.0
    if prev:
        prev_rem = max(0.0, prev[0] - now)
    until_mono = max(now + dur, prev[0] if prev else 0.0)
    _auth_unhealthy_accounts[key] = (until_mono, reason)
    new_rem = max(0.0, until_mono - now)
    # Keep this warning concise and only emit when extending/setting a backoff.
    if new_rem >= prev_rem + 1.0:
        logger.warning(
            "[cTrader] auth backoff user=%s ctid=%s host=%s %.0fs (%s)",
            user_id,
            ctid,
            host,
            new_rem,
            reason,
        )


def _clear_account_auth_unhealthy(user_id: int, host: str, ctid: int) -> None:
    _auth_unhealthy_accounts.pop(_auth_account_key(user_id, host, ctid), None)


def _user_ctid_has_auth_backoff(user_id: int, ctid: int) -> bool:
    uid = int(user_id)
    acct = int(ctid)
    now = time.monotonic()
    hit = False
    for key, row in list(_auth_unhealthy_accounts.items()):
        if key[0] != uid or key[2] != acct:
            continue
        if row[0] <= now:
            _auth_unhealthy_accounts.pop(key, None)
            continue
        hit = True
    return hit


async def _singleflight_forced_refresh(user_id: int) -> Tuple[Optional[str], str]:
    """One forced token refresh per user per cooldown window."""
    uid = int(user_id)
    while True:
        wait_s = 0.0
        task: Optional[asyncio.Task] = None
        created = False
        async with _auth_state_lock:
            now = time.monotonic()
            task = _auth_refresh_inflight.get(uid)
            if task is not None and task.done():
                _auth_refresh_inflight.pop(uid, None)
                task = None
            if task is None:
                next_allowed = _auth_refresh_next_allowed.get(uid, 0.0)
                if now < next_allowed:
                    wait_s = next_allowed - now
                else:
                    task = asyncio.create_task(refresh_user_ctrader_token(uid, force=True))
                    _auth_refresh_inflight[uid] = task
                    _auth_refresh_next_allowed[uid] = now + _AUTH_REFRESH_COOLDOWN_S
                    created = True
        if wait_s > 0:
            return None, f"refresh_throttled_{wait_s:.1f}s"
        if task is None:
            await asyncio.sleep(0)
            continue
        try:
            token = await asyncio.wait_for(task, timeout=_AUTH_REFRESH_WAIT_TIMEOUT_S)
            return token, ("ok" if token else "refresh_failed")
        except asyncio.TimeoutError:
            if created:
                task.cancel()
            return None, "refresh_timeout"
        except Exception as exc:
            logger.warning(
                "[cTrader] forced refresh error uid=%s: %s",
                uid,
                type(exc).__name__,
            )
            return None, "refresh_error"
        finally:
            if task.done():
                async with _auth_state_lock:
                    if _auth_refresh_inflight.get(uid) is task:
                        _auth_refresh_inflight.pop(uid, None)


def is_refresh_denied(user_id: int) -> bool:
    """True when this user's refresh chain is in terminal cooldown."""
    denied = _refresh_denied.get(int(user_id))
    if not denied:
        return False
    return time.monotonic() < denied[1]


def clear_ctrader_oauth_denied(user_id: Optional[int] = None) -> None:
    """Clear in-process OAuth denial state after a successful re-link."""
    if user_id is not None:
        _refresh_denied.pop(int(user_id), None)
    else:
        _refresh_denied.clear()


def _notify_ctrader_relink_needed(user_id: int, err: str) -> None:
    """One Telegram alert per user per cooldown — do not spam on every tick."""
    now = time.monotonic()
    if now - _RELINK_ALERT_SENT.get(int(user_id), 0.0) < _REFRESH_DENY_COOLDOWN:
        return
    _RELINK_ALERT_SENT[int(user_id)] = now
    try:
        from app.database import SessionLocal
        from app.models import User
        from app.services.strategy_executor import _telegram_int_id
        from app.services.telegram_dm import send_dm

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == int(user_id)).first()
            tg_id = _telegram_int_id(user) if user else None
        finally:
            db.close()
        if not tg_id:
            return
        asyncio.get_event_loop().create_task(
            send_dm(
                tg_id,
                "⚠️ <b>cTrader re-link needed</b>\n\n"
                "Your live trading session expired and could not be refreshed. "
                "Open the portal → Settings → cTrader and connect again.\n\n"
                f"<code>{(err or 'ACCESS_DENIED')[:120]}</code>",
                msg_type="ctrader_relink",
                asset_class="forex",
            )
        )
    except Exception:
        pass


def _mark_refresh_denied(user_id: int, refresh_token: str, err: str) -> None:
    _refresh_denied[user_id] = (
        refresh_token,
        time.monotonic() + _REFRESH_DENY_COOLDOWN,
    )
    logger.warning(
        "[cTrader] token refresh DENIED user=%s reason=%s action=re-link "
        "(portal → Settings → cTrader OAuth). OAuth app CLIENT_ID=%s",
        user_id,
        err,
        "set" if CTRADER_CLIENT_ID else "MISSING",
    )
    _notify_ctrader_relink_needed(user_id, err)


def audit_ctrader_credentials(user_id: int, prefs=None) -> dict:
    """
    Explicit credential audit — never silent. Returns {ok, reason, fields}.
    """
    fields = {
        "CLIENT_ID": bool(CTRADER_CLIENT_ID),
        "CLIENT_SECRET": bool(CTRADER_CLIENT_SECRET),
        "ACCESS_TOKEN": False,
        "REFRESH_TOKEN": False,
        "ACCOUNT_ID": False,
    }
    if not fields["CLIENT_ID"] or not fields["CLIENT_SECRET"]:
        return {
            "ok": False,
            "reason": "missing CTRADER_CLIENT_ID or CTRADER_CLIENT_SECRET env",
            "fields": fields,
            "user_id": user_id,
        }
    if prefs is None:
        try:
            from app.database import SessionLocal
            from app.models import UserPreference
            db = SessionLocal()
            try:
                prefs = (
                    db.query(UserPreference)
                    .filter(UserPreference.user_id == user_id)
                    .first()
                )
            finally:
                db.close()
        except Exception as exc:
            return {
                "ok": False,
                "reason": f"DB prefs unreadable: {type(exc).__name__}",
                "fields": fields,
                "user_id": user_id,
            }
    if not prefs:
        return {
            "ok": False,
            "reason": "no UserPreference row",
            "fields": fields,
            "user_id": user_id,
        }
    fields["ACCESS_TOKEN"] = bool((prefs.ctrader_access_token or "").strip())
    fields["REFRESH_TOKEN"] = bool((prefs.ctrader_refresh_token or "").strip())
    fields["ACCOUNT_ID"] = bool((prefs.ctrader_account_id or "").strip())
    missing = [k for k, v in fields.items() if k not in ("CLIENT_ID", "CLIENT_SECRET") and not v]
    if missing:
        return {
            "ok": False,
            "reason": f"missing user fields: {', '.join(missing)}",
            "fields": fields,
            "user_id": user_id,
        }
    if is_refresh_denied(user_id):
        return {
            "ok": False,
            "reason": "refresh chain denied — re-link cTrader",
            "fields": fields,
            "user_id": user_id,
        }
    return {"ok": True, "reason": "credentials present", "fields": fields, "user_id": user_id}


async def proactive_refresh_linked_users() -> dict:
    """
    Refresh OAuth tokens for all linked users — delegated to the token refresh owner.
    Returns counts: linked_users, refreshed, failed, denied.
    """
    try:
        from app.services.ctrader_token_scheduler import run_token_refresh_cycle

        return await run_token_refresh_cycle(reason="startup_probe")
    except Exception as exc:
        logger.warning(
            "[cTrader] proactive refresh batch failed: %s", type(exc).__name__
        )
        return {"linked_users": 0, "refreshed": 0, "failed": 0, "denied": 0}


def _read_fresh_ctrader_prefs(db, user_id: int):
    """Re-read persisted OAuth tokens from DB (never use stale in-memory copies)."""
    from app.models import UserPreference

    try:
        db.expire_all()
    except Exception:
        pass
    return (
        db.query(UserPreference)
        .filter(UserPreference.user_id == int(user_id))
        .first()
    )


def _oauth_error_code(res: dict) -> str:
    return str(res.get("errorCode") or res.get("error") or "").strip()


def _access_token_ttl_seconds(access_token: str) -> Optional[int]:
    """Best-effort JWT exp → seconds remaining; None when unknown."""
    try:
        import base64
        import json

        parts = (access_token or "").split(".")
        if len(parts) != 3:
            return None
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        exp = data.get("exp")
        if exp is None:
            return None
        return max(0, int(exp) - int(time.time()))
    except Exception:
        return None


def _oauth_expires_at_from_response(res: dict) -> Optional["datetime"]:
    """Parse OAuth token response into UTC expiry (expiresIn preferred)."""
    from datetime import datetime, timedelta

    if not isinstance(res, dict):
        return None
    expires_in = res.get("expiresIn")
    if expires_in is None:
        expires_in = res.get("expires_in")
    if expires_in is not None:
        try:
            return datetime.utcnow() + timedelta(seconds=int(expires_in))
        except (TypeError, ValueError):
            pass
    at = res.get("accessToken") or res.get("access_token") or ""
    ttl = _access_token_ttl_seconds(str(at))
    if ttl is not None:
        return datetime.utcnow() + timedelta(seconds=ttl)
    return None


def _token_seconds_remaining_from_prefs(prefs) -> Optional[int]:
    """Seconds until access token expiry — persisted expires_at first, then JWT."""
    from datetime import datetime

    exp_at = getattr(prefs, "ctrader_access_token_expires_at", None)
    if exp_at is not None:
        return max(0, int((exp_at - datetime.utcnow()).total_seconds()))
    at = (getattr(prefs, "ctrader_access_token", None) or "").strip()
    return _access_token_ttl_seconds(at)


def persist_ctrader_oauth_tokens(db, prefs, res: dict) -> tuple[str, Optional[str]]:
    """Apply OAuth token response to prefs (access, rotated refresh, expiry)."""
    new_at = res.get("accessToken") or res.get("access_token")
    new_rt = res.get("refreshToken") or res.get("refresh_token")
    if not new_at:
        raise ValueError("OAuth response missing access token")
    prefs.ctrader_access_token = new_at
    if new_rt:
        prefs.ctrader_refresh_token = new_rt
    exp_at = _oauth_expires_at_from_response(res)
    if exp_at is not None:
        prefs.ctrader_access_token_expires_at = exp_at
    return str(new_at), (str(new_rt) if new_rt else None)


async def request_ctrader_token_refresh(
    user_id: int,
    *,
    force: bool = False,
    wait_s: float = 12.0,
) -> Optional[str]:
    """Consumer entry — forced refresh uses live-trading single-flight; otherwise wait for owner."""
    uid = int(user_id)
    if force:
        new_at, _status = await _singleflight_forced_refresh(uid)
        return new_at
    try:
        from app.services.ctrader_token_scheduler import (
            is_token_refresh_owner,
            run_token_refresh_cycle,
            wake_token_scheduler,
        )

        if is_token_refresh_owner():
            await run_token_refresh_cycle(reason="consumer_request")
            return _latest_ctrader_access_token(uid)
        wake_token_scheduler()
    except Exception:
        pass
    deadline = time.monotonic() + max(0.5, float(wait_s))
    prev = _latest_ctrader_access_token(uid)
    while time.monotonic() < deadline:
        await asyncio.sleep(0.5)
        cur = _latest_ctrader_access_token(uid)
        if cur and cur != prev:
            return cur
    return _latest_ctrader_access_token(uid)


def _log_ctrader_token_startup(
    user_id: int,
    access_token: str,
    *,
    refreshed: bool = False,
) -> None:
    if refreshed:
        logger.info("[ctrader-token] refreshed (single-flight) uid=%s", user_id)
        return
    ttl = _access_token_ttl_seconds(access_token)
    if ttl is not None:
        logger.info(
            "[ctrader-token] startup: using persisted token (expires in %ss) uid=%s",
            ttl,
            user_id,
        )
    else:
        logger.info(
            "[ctrader-token] startup: using persisted token uid=%s",
            user_id,
        )


async def _try_acquire_token_refresh_lock(db, user_id: int) -> bool:
    """Single-flight refresh lock — one holder per user across all workers."""
    from sqlalchemy import text as _sql_text

    for _attempt in range(20):  # wait up to ~10s for a peer's refresh
        acquired = bool(
            db.execute(
                _sql_text("SELECT pg_try_advisory_xact_lock(:ns, :uid)"),
                {"ns": _TOKEN_REFRESH_PG_LOCK_NS, "uid": int(user_id)},
            ).scalar()
        )
        if acquired:
            return True
        db.rollback()
        await asyncio.sleep(0.5)
    return False


async def _wait_peer_refresh_and_read(db, user_id: int) -> Optional[str]:
    """Another worker holds the refresh lock — wait, then DB-first re-read."""
    await asyncio.sleep(0.75)
    prefs = _read_fresh_ctrader_prefs(db, user_id)
    at = (prefs.ctrader_access_token or "").strip() if prefs else ""
    if at:
        _log_ctrader_token_startup(user_id, at)
    return at or None


async def refresh_user_ctrader_token(
    user_id: int,
    *,
    force: bool = False,
) -> Optional[str]:
    """Refresh and persist a user's cTrader OAuth token (single-flight).

    cTrader ROTATES the refresh token on every refresh — the new refresh token
    MUST be persisted or the next refresh fails. All callers wait on the same
    pg advisory xact lock, re-read refresh_token from DB before OAuth, and on
    invalid_grant / ACCESS_DENIED wait 2s + re-read (up to 3 rounds) before
    marking the link dead.

    Returns the access token, or None if there is no refresh token / the
    refresh was denied after retries (user must re-link their cTrader account).
    """
    from app.database import SessionLocal

    async with _token_refresh_lock:
        for round_num in range(1, _REFRESH_FAILURE_ROUNDS + 1):
            db = SessionLocal()
            try:
                acquired = await _try_acquire_token_refresh_lock(db, user_id)
                prefs = _read_fresh_ctrader_prefs(db, user_id)
                if not prefs or not prefs.ctrader_refresh_token:
                    return None

                if not acquired:
                    at = await _wait_peer_refresh_and_read(db, user_id)
                    try:
                        db.rollback()
                    except Exception:
                        pass
                    if at:
                        return at
                    if round_num < _REFRESH_FAILURE_ROUNDS:
                        await asyncio.sleep(_REFRESH_RETRY_WAIT_S)
                        continue
                    return None

                # DB-first: latest persisted tokens only (peer may have rotated).
                prefs = _read_fresh_ctrader_prefs(db, user_id)
                if not prefs or not prefs.ctrader_refresh_token:
                    return None
                refresh_token = (prefs.ctrader_refresh_token or "").strip()
                existing_at = (prefs.ctrader_access_token or "").strip()
                if not force and existing_at:
                    remaining = _token_seconds_remaining_from_prefs(prefs)
                    threshold = _SCHEDULED_REFRESH_WHEN_REMAINING_S
                    if remaining is not None and remaining > threshold:
                        _log_ctrader_token_startup(user_id, existing_at)
                        try:
                            db.rollback()
                        except Exception:
                            pass
                        return existing_at

                denied = _refresh_denied.get(user_id)
                if denied and denied[0] == refresh_token and time.monotonic() < denied[1]:
                    return None

                # Immediately before OAuth: one more DB-first read.
                prefs = _read_fresh_ctrader_prefs(db, user_id)
                if not prefs or not prefs.ctrader_refresh_token:
                    return None
                refresh_token = (prefs.ctrader_refresh_token or "").strip()

                try:
                    res = await refresh_access_token(refresh_token)
                except Exception as e:
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    logger.warning(
                        "[cTrader] token refresh HTTP error uid=%s round=%s/%s: %s status=%s",
                        user_id,
                        round_num,
                        _REFRESH_FAILURE_ROUNDS,
                        type(e).__name__,
                        status,
                    )
                    try:
                        db.rollback()
                    except Exception:
                        pass
                    if round_num < _REFRESH_FAILURE_ROUNDS:
                        await asyncio.sleep(_REFRESH_RETRY_WAIT_S)
                        continue
                    return None

                err = _oauth_error_code(res) if isinstance(res, dict) else ""
                if err:
                    logger.warning(
                        "[ctrader-token] refresh round %s/%s failed uid=%s err=%s",
                        round_num,
                        _REFRESH_FAILURE_ROUNDS,
                        user_id,
                        err,
                    )
                    try:
                        db.rollback()
                    except Exception:
                        pass
                    if round_num < _REFRESH_FAILURE_ROUNDS:
                        await asyncio.sleep(_REFRESH_RETRY_WAIT_S)
                        continue
                    if err in _REFRESH_TERMINAL_CODES:
                        _mark_refresh_denied(user_id, refresh_token, err)
                    return None

                new_at = res.get("accessToken") or res.get("access_token")
                new_rt = res.get("refreshToken") or res.get("refresh_token")
                if not new_at:
                    try:
                        db.rollback()
                    except Exception:
                        pass
                    if round_num < _REFRESH_FAILURE_ROUNDS:
                        await asyncio.sleep(_REFRESH_RETRY_WAIT_S)
                        continue
                    return None

                prefs.ctrader_access_token = new_at
                if new_rt:
                    prefs.ctrader_refresh_token = new_rt
                exp_at = _oauth_expires_at_from_response(res)
                if exp_at is not None:
                    prefs.ctrader_access_token_expires_at = exp_at
                _refresh_denied.pop(user_id, None)
                _RELINK_ALERT_SENT.pop(int(user_id), None)
                for persist_try in (1, 2, 3):
                    try:
                        db.commit()
                        _log_ctrader_token_startup(user_id, new_at, refreshed=True)
                        try:
                            from app.services.ctrader_price_feed import invalidate_stream_creds
                            invalidate_stream_creds(user_id)
                        except Exception:
                            pass
                        return new_at
                    except Exception:
                        try:
                            db.rollback()
                        except Exception:
                            pass
                        if persist_try < 3:
                            await asyncio.sleep(0.75 * persist_try)
                            prefs = _read_fresh_ctrader_prefs(db, user_id)
                            if not prefs:
                                break
                            prefs.ctrader_access_token = new_at
                            if new_rt:
                                prefs.ctrader_refresh_token = new_rt
                            if exp_at is not None:
                                prefs.ctrader_access_token_expires_at = exp_at
                            continue
                        raise
            except Exception as e:
                logger.warning(
                    "[cTrader] token refresh persist error uid=%s: %s",
                    user_id,
                    type(e).__name__,
                )
                try:
                    db.rollback()
                except Exception:
                    pass
                if round_num < _REFRESH_FAILURE_ROUNDS:
                    await asyncio.sleep(_REFRESH_RETRY_WAIT_S)
                    continue
                return None
            finally:
                db.close()
        return None


def _latest_ctrader_access_token(user_id: int) -> Optional[str]:
    """Read the latest persisted OAuth access token for a user."""
    from app.database import SessionLocal
    from app.models import UserPreference

    db = SessionLocal()
    try:
        prefs = db.query(UserPreference).filter(UserPreference.user_id == int(user_id)).first()
        token = (prefs.ctrader_access_token or "").strip() if prefs else ""
        return token or None
    except Exception:
        return None
    finally:
        db.close()


async def _refresh_auth_token_for_retry(
    *,
    user_id: Optional[int],
    ctid_trader_account_id: int,
    host: str,
    operation: str,
) -> Optional[str]:
    """On broker account-auth failure, single-flight refresh and retry once."""
    if not user_id:
        return None
    uid = int(user_id)
    if _account_auth_in_cooldown(uid, host, ctid_trader_account_id):
        rem = _account_auth_cooldown_remaining(uid, host, ctid_trader_account_id)
        logger.info(
            "[cTrader] %s retry suppressed user=%s ctid=%s host=%s (cooldown %.0fs)",
            operation,
            uid,
            ctid_trader_account_id,
            host,
            rem,
        )
        return None
    new_at, status = await _singleflight_forced_refresh(uid)
    if new_at:
        _clear_account_auth_unhealthy(uid, host, ctid_trader_account_id)
        logger.info(
            "[cTrader] %s auth failed ctid=%s host=%s user=%s — refreshed token, retrying",
            operation,
            ctid_trader_account_id,
            host,
            uid,
        )
        return new_at
    if is_refresh_denied(uid):
        _mark_account_auth_unhealthy(
            uid,
            host,
            ctid_trader_account_id,
            reason="refresh_denied",
            cooldown_s=max(_AUTH_UNHEALTHY_COOLDOWN_S, _REFRESH_DENY_COOLDOWN),
        )
        _notify_ctrader_relink_needed(uid, f"{operation} auth failed")
    elif status.startswith("refresh_throttled_"):
        _mark_account_auth_unhealthy(
            uid,
            host,
            ctid_trader_account_id,
            reason=status,
            cooldown_s=_AUTH_TIMEOUT_COOLDOWN_S,
        )
    else:
        _mark_account_auth_unhealthy(
            uid,
            host,
            ctid_trader_account_id,
            reason=status or "refresh_failed",
            cooldown_s=_AUTH_UNHEALTHY_COOLDOWN_S,
        )
    return None


async def _get_accounts_on_host(access_token: str, host: str) -> list:
    """Fetch linked accounts from one cTrader host (live or demo)."""
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
                logger.error(f"[cTrader] get_accounts failed on {host}: {e}")
                return []
    return []


async def get_accounts_for_token(access_token: str, host: str = CTRADER_HOST) -> list:
    """
    Return a list of cTrader trading accounts linked to this access token.
    Each item: {ctidTraderAccountId, isLive, traderLogin, balance}.
    Merges results from BOTH live and demo hosts — demo ctids are only
    reachable on demo.ctraderapi.com and were previously invisible.
    """
    if not _PROTO_OK:
        return []
    merged: Dict[int, dict] = {}
    async def _timed(host: str) -> list:
        try:
            return await asyncio.wait_for(
                _get_accounts_on_host(access_token, host), timeout=8.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[cTrader] get_accounts timed out on {host}")
            return []

    batches = await asyncio.gather(
        _timed(CTRADER_HOST_LIVE),
        _timed(CTRADER_HOST_DEMO),
        return_exceptions=True,
    )
    for batch in batches:
        if isinstance(batch, Exception):
            logger.warning(f"[cTrader] get_accounts host batch failed: {batch}")
            continue
        for acc in batch:
            merged[int(acc["ctidTraderAccountId"])] = acc
    return list(merged.values())


def _persist_account_host_metadata(user_id: int, ctid: int, host: str) -> None:
    """Correct stale isLive metadata after auth succeeds on alternate host."""
    import json as _json
    from app.database import SessionLocal
    from app.models import UserPreference
    is_live = host == CTRADER_HOST_LIVE
    db = SessionLocal()
    try:
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
        if not prefs:
            return
        try:
            accounts = _json.loads(prefs.ctrader_accounts or "[]")
        except Exception:
            accounts = []
        updated = False
        for a in accounts:
            if int(a.get("ctidTraderAccountId", -1)) == int(ctid):
                if bool(a.get("isLive")) != is_live:
                    a["isLive"] = is_live
                    updated = True
                break
        else:
            accounts.append({
                "ctidTraderAccountId": int(ctid),
                "isLive": is_live,
                "traderLogin": 0,
            })
            updated = True
        if updated:
            prefs.ctrader_accounts = _json.dumps(accounts)
            db.commit()
            logger.info(
                f"[cTrader] corrected isLive={is_live} for ctid={ctid} user={user_id}"
            )
    except Exception as e:
        logger.warning(f"[cTrader] persist host metadata failed user={user_id}: {e}")
    finally:
        db.close()


def _routing_hosts_for_account(prefs, ctid_trader_account_id: int) -> List[str]:
    """Hosts to try for a ctid — single host when demo/live known, else primary+alternate."""
    known_live = _account_is_live(prefs, ctid_trader_account_id) if prefs else None
    if known_live is False:
        return [CTRADER_HOST_DEMO]
    if known_live is True:
        return [CTRADER_HOST_LIVE]
    primary = _host_for_account(prefs, ctid_trader_account_id) if prefs else CTRADER_HOST_LIVE
    hosts = [primary]
    alt = _other_host(primary)
    if alt not in hosts:
        hosts.append(alt)
    return hosts


_balance_hosts_for_account = _routing_hosts_for_account


def _should_try_alternate_order_host(error: Optional[str], *, known_account_type: bool) -> bool:
    """When account demo/live is unknown, retry on the other host after routing failures."""
    if known_account_type or not error:
        return False
    up = str(error).upper()
    if "ACCOUNT AUTH FAILED" in up:
        return True
    if "ORDER_CANCELLED" in up or "ORDER_REJECTED" in up:
        return True
    return False


def _log_order_route(
    *,
    execution_id: Optional[int],
    ctid: int,
    host: str,
    result: dict,
    volume_units: Optional[int] = None,
    lots: Optional[float] = None,
    is_live: Optional[bool] = None,
) -> None:
    _is_live = is_live if is_live is not None else (host == CTRADER_HOST_LIVE)
    vol = volume_units if volume_units is not None else result.get("volume")
    err = result.get("error")
    if result.get("actual_fill") and float(result.get("actual_fill") or 0) > 0:
        outcome = "fill"
    elif err:
        outcome = str(err)[:120]
    else:
        outcome = "no_fill"
    size = lots if lots is not None else (vol if vol is not None else "?")
    logger.info(
        "[order] exec=%s ctid=%s is_live=%s host=%s lots=%s → %s",
        execution_id if execution_id is not None else "?",
        ctid,
        _is_live,
        host,
        size,
        outcome,
    )


async def place_market_order_resilient(
    *,
    user_id: Optional[int],
    access_token: str,
    ctid: int,
    prefs,
    symbol_name: str,
    direction: str,
    volume_lots: Optional[float] = None,
    volume_units: Optional[int] = None,
    stop_loss_price: Optional[float] = None,
    take_profit_price: Optional[float] = None,
    entry_price: Optional[float] = None,
    sl_pct: Optional[float] = None,
    tp_pct: Optional[float] = None,
    label: str = "TradeHub",
    latency=None,
    execution_id: Optional[int] = None,
) -> dict:
    """
    Place a market order with token-refresh and per-ctid live/demo host routing.
    When account type is unknown, tries primary host then the alternate on
    auth/cancel failures (demo ctids are invisible on the live host).
    """
    hosts = _routing_hosts_for_account(prefs, ctid) if prefs else [CTRADER_HOST_LIVE]
    known_account_type = (
        _account_is_live(prefs, ctid) is not None if prefs else False
    )
    _acct_is_live = _account_is_live(prefs, ctid) if prefs else None
    at = access_token

    async def _place_on(h: str) -> dict:
        if volume_units is not None:
            return await place_order_units(
                access_token=at,
                ctid_trader_account_id=ctid,
                symbol_name=symbol_name,
                direction=direction,
                volume_units=volume_units,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                entry_price=entry_price,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                label=label,
                host=h,
                latency=latency,
            )
        return await place_order(
            access_token=at,
            ctid_trader_account_id=ctid,
            symbol_name=symbol_name,
            direction=direction,
            volume_lots=volume_lots or 0.01,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            entry_price=entry_price,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            label=label,
            host=h,
            latency=latency,
        )

    async def _try_hosts(token: str) -> dict:
        last: dict = {"order_id": None, "actual_fill": None, "error": "no host succeeded"}
        for idx, h in enumerate(hosts):
            result = await _place_on(h)
            _log_order_route(
                execution_id=execution_id,
                ctid=ctid,
                host=h,
                result=result,
                volume_units=volume_units,
                lots=volume_lots,
                is_live=_acct_is_live if _acct_is_live is not None else (h == CTRADER_HOST_LIVE),
            )
            fill = result.get("actual_fill")
            if fill is not None and float(fill) > 0:
                if user_id:
                    _persist_account_host_metadata(user_id, ctid, h)
                out = dict(result)
                out["host"] = h
                return out
            err = result.get("error")
            if err == "account auth failed":
                last = result
                if idx < len(hosts) - 1:
                    continue
                return result
            if _should_try_alternate_order_host(err, known_account_type=known_account_type):
                last = result
                if idx < len(hosts) - 1:
                    logger.info(
                        "[order] exec=%s ctid=%s retrying on alternate host after: %s",
                        execution_id or "?",
                        ctid,
                        err,
                    )
                    continue
            return result
        return last

    result = await _try_hosts(at)

    if result.get("error") == "account auth failed" and user_id:
        new_at, _status = await _singleflight_forced_refresh(int(user_id))
        if new_at:
            at = new_at
            logger.info(f"[cTrader] retrying order for user {user_id} after token refresh")
            result = await _try_hosts(new_at)
        elif is_refresh_denied(user_id):
            _notify_ctrader_relink_needed(user_id, "account auth failed")

    if is_ambiguous_order_error(result.get("error")):
        vol = volume_units
        if vol is None and volume_lots is not None:
            vol = int(round(float(volume_lots) * 100_000))
        for h in hosts:
            recovered = await reconcile_order_fill_after_miss(
                access_token=at,
                ctid=ctid,
                host=h,
                symbol_name=symbol_name,
                direction=direction,
                entry_hint=entry_price,
                volume_units=vol,
            )
            if recovered:
                _log_order_route(
                    execution_id=execution_id,
                    ctid=ctid,
                    host=h,
                    result=recovered,
                    volume_units=vol,
                    lots=volume_lots,
                    is_live=_acct_is_live if _acct_is_live is not None else (h == CTRADER_HOST_LIVE),
                )
                if user_id:
                    _persist_account_host_metadata(user_id, ctid, h)
                recovered = dict(recovered)
                recovered["host"] = h
                return recovered

    return result


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
            "lotSize":      int(s.lotSize)      if s.HasField("lotSize")      else 0,
            "minVolume":    int(s.minVolume)    if s.HasField("minVolume")    else 0,
            "maxVolume":    int(s.maxVolume)    if s.HasField("maxVolume")    else 0,
            "stepVolume":   int(s.stepVolume)   if s.HasField("stepVolume")   else 0,
            "digits":       int(s.digits)       if s.HasField("digits")       else 0,
            "pipPosition":  int(s.pipPosition)  if s.HasField("pipPosition")  else 0,
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
    *,
    fill_timeout: Optional[float] = None,
    latency=None,
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
        _fill_deadline = time.monotonic() + float(
            fill_timeout if fill_timeout is not None else ORDER_FILL_WAIT_S
        )
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

    if _has_fill(ev) and latency is not None:
        try:
            latency.mark_fill()
        except Exception:
            pass

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
    sl_pct: Optional[float] = None,
    tp_pct: Optional[float] = None,
    label: str = "TradeHub",
    host: str = CTRADER_HOST,
    *,
    latency=None,
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
                    rel_sl, rel_tp = _relative_sltp_wire(entry_price, sl_pct, tp_pct)
                    if rel_sl is not None:
                        req.relativeStopLoss = rel_sl
                    elif stop_loss_price is not None:
                        from app.services.pip_units import to_broker_relative_wire_units
                        req.relativeStopLoss = to_broker_relative_wire_units(
                            entry_price - stop_loss_price
                        )
                    if rel_tp is not None:
                        req.relativeTakeProfit = rel_tp
                    elif take_profit_price is not None:
                        from app.services.pip_units import to_broker_relative_wire_units
                        req.relativeTakeProfit = to_broker_relative_wire_units(
                            entry_price - take_profit_price
                        )
                elif stop_loss_price is not None or take_profit_price is not None:
                    logger.error(
                        "[cTrader] cannot place MARKET order with SL/TP — entry_price "
                        f"missing (symbol={symbol_name})"
                    )
                    return {"order_id": None, "actual_fill": None,
                            "error": "entry_price required for SL/TP on market order"}

                if latency is not None:
                    try:
                        latency.mark_submitted()
                    except Exception:
                        pass

                pt, payload = await _send_recv_any(
                    reader, writer, req,
                    _PAYLOAD_TYPES["new_order_req"],
                    {_PAYLOAD_TYPES["execution_event"],
                     _PAYLOAD_TYPES["order_error_event"]},
                    timeout=ORDER_SUBMIT_TIMEOUT_S,
                )
                if latency is not None and payload:
                    try:
                        latency.mark_broker_ack()
                    except Exception:
                        pass
                if not payload:
                    return {"order_id": None, "actual_fill": None, "error": "no execution event"}

                _touch_conn(host, ctid_trader_account_id)

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
                    latency=latency,
                )

            except asyncio.TimeoutError:
                _invalidate_persistent_connection(host, ctid_trader_account_id)
                if attempt == 1:
                    logger.warning(
                        "[cTrader] place_order timeout (%.0fs) — one retry",
                        ORDER_SUBMIT_TIMEOUT_S,
                    )
                    continue
                return {"order_id": None, "actual_fill": None, "error": "timeout"}
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
    sl_pct: Optional[float] = None,
    tp_pct: Optional[float] = None,
    label: str = "TradeHub",
    host: str = CTRADER_HOST,
    *,
    latency=None,
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
                    rel_sl, rel_tp = _relative_sltp_wire(entry_price, sl_pct, tp_pct)
                    if rel_sl is not None:
                        req.relativeStopLoss = rel_sl
                    elif stop_loss_price is not None:
                        from app.services.pip_units import to_broker_relative_wire_units
                        req.relativeStopLoss = to_broker_relative_wire_units(
                            entry_price - stop_loss_price
                        )
                    if rel_tp is not None:
                        req.relativeTakeProfit = rel_tp
                    elif take_profit_price is not None:
                        from app.services.pip_units import to_broker_relative_wire_units
                        req.relativeTakeProfit = to_broker_relative_wire_units(
                            entry_price - take_profit_price
                        )

                if latency is not None:
                    try:
                        latency.mark_submitted()
                    except Exception:
                        pass

                pt, payload = await _send_recv_any(
                    reader, writer, req,
                    _PAYLOAD_TYPES["new_order_req"],
                    {_PAYLOAD_TYPES["execution_event"],
                     _PAYLOAD_TYPES["order_error_event"]},
                    timeout=ORDER_SUBMIT_TIMEOUT_S,
                )
                if latency is not None and payload:
                    try:
                        latency.mark_broker_ack()
                    except Exception:
                        pass
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
                    latency=latency,
                )
            except asyncio.TimeoutError:
                _invalidate_persistent_connection(host, ctid_trader_account_id)
                if attempt == 1:
                    logger.warning(
                        "[cTrader] place_order_units timeout (%.0fs) — one retry",
                        ORDER_SUBMIT_TIMEOUT_S,
                    )
                    continue
                return {"order_id": None, "actual_fill": None, "error": "timeout"}
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


def _parse_amend_execution_event(payload: bytes, position_id: int) -> dict:
    """Classify broker reply to ProtoOAAmendPositionSLTPReq."""
    _terminal = (
        ProtoOAExecutionType.ORDER_REJECTED,
        ProtoOAExecutionType.ORDER_CANCELLED,
        ProtoOAExecutionType.ORDER_EXPIRED,
        ProtoOAExecutionType.ORDER_CANCEL_REJECTED,
    )
    _success = (
        ProtoOAExecutionType.ORDER_ACCEPTED,
        ProtoOAExecutionType.ORDER_REPLACED,
        ProtoOAExecutionType.ORDER_FILLED,
        ProtoOAExecutionType.SWAP,
    )
    try:
        ev = ProtoOAExecutionEvent()
        ev.ParseFromString(payload)
    except Exception as exc:
        return {
            "ok": False,
            "result": "failed",
            "broker_reply": {"parse_error": str(exc)},
        }
    et = ev.executionType
    tname = ProtoOAExecutionType.Name(et) if et is not None else "UNKNOWN"
    err = ev.errorCode if ev.HasField("errorCode") else ""
    pos = int(ev.position.positionId) if ev.HasField("position") and ev.position.positionId else 0
    reply = {
        "execution_type": tname,
        "error_code": err or None,
        "position_id": pos or None,
    }
    if et in _terminal:
        return {
            "ok": False,
            "result": "failed",
            "broker_reply": reply,
            "error": f"{tname}: {err}" if err else tname,
        }
    if et in _success:
        if pos and int(pos) != int(position_id):
            return {
                "ok": False,
                "result": "failed",
                "broker_reply": reply,
                "error": f"position_id mismatch want={position_id} got={pos}",
            }
        return {"ok": True, "result": "confirmed", "broker_reply": reply}
    if ev.HasField("order") and et not in _terminal:
        return {"ok": True, "result": "confirmed", "broker_reply": reply}
    return {
        "ok": False,
        "result": "failed",
        "broker_reply": reply,
        "error": f"unexpected execution type {tname}",
    }


async def modify_position_sltp_result(
    access_token: str,
    ctid_trader_account_id: int,
    position_id: int,
    stop_loss_price: Optional[float] = None,
    take_profit_price: Optional[float] = None,
    host: str = CTRADER_HOST,
    *,
    exec_id: Optional[int] = None,
) -> dict:
    """
    Amend SL/TP and return broker-confirmed result dict:
    {ok, result: confirmed|failed|timeout, broker_reply, error?}
    """
    if not _PROTO_OK:
        return {"ok": False, "result": "failed", "broker_reply": {}, "error": "proto unavailable"}
    if stop_loss_price is None and take_profit_price is None:
        return {"ok": False, "result": "failed", "broker_reply": {}, "error": "no legs to amend"}

    last_reply: dict = {"ok": False, "result": "timeout", "broker_reply": {}}
    async with _get_account_lock(host, ctid_trader_account_id):
        for attempt in (1, 2):
            try:
                reader, writer = await _get_persistent_connection(host, ctid_trader_account_id)
                if not await _account_auth(reader, writer, access_token, ctid_trader_account_id):
                    last_reply = {
                        "ok": False,
                        "result": "failed",
                        "broker_reply": {"error": "account_auth_failed"},
                        "error": "account_auth_failed",
                    }
                    break
                req = ProtoOAAmendPositionSLTPReq()
                req.ctidTraderAccountId = ctid_trader_account_id
                req.positionId = int(position_id)
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
                if payload is None:
                    last_reply = {"ok": False, "result": "timeout", "broker_reply": {}}
                    if attempt == 1:
                        continue
                    break
                parsed = _parse_amend_execution_event(payload, int(position_id))
                if parsed.get("ok"):
                    _touch_conn(host, ctid_trader_account_id)
                last_reply = parsed
                if parsed.get("ok") or attempt == 2:
                    break
            except Exception as e:
                last_reply = {
                    "ok": False,
                    "result": "failed",
                    "broker_reply": {"exception": type(e).__name__},
                    "error": str(e),
                }
                if attempt == 1:
                    logger.warning(f"[cTrader] modify_position_sltp retry after: {e}")
                    _invalidate_persistent_connection(host, ctid_trader_account_id)
                    continue
                logger.error(f"[cTrader] modify_position_sltp failed: {e}")
                break

    _eid = f" exec={exec_id}" if exec_id else ""
    logger.info(
        "[sl-amend]%s pos=%s requested_sl=%s requested_tp=%s result=%s broker_reply=%s",
        _eid,
        position_id,
        stop_loss_price,
        take_profit_price,
        last_reply.get("result"),
        last_reply.get("broker_reply"),
    )
    return last_reply


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
    res = await modify_position_sltp_result(
        access_token,
        ctid_trader_account_id,
        position_id,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        host=host,
    )
    return bool(res.get("ok"))


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


async def amend_position_sl(
    user_id: int,
    position_id: int,
    new_sl: float,
    *,
    keep_tp: Optional[float] = None,
    exec_id: Optional[int] = None,
) -> bool:
    """Amend stop-loss on a live position; preserves take-profit when provided."""
    res = await amend_position_sl_result(
        user_id, position_id, new_sl, keep_tp=keep_tp, exec_id=exec_id,
    )
    return bool(res.get("ok"))


async def amend_position_sl_result(
    user_id: int,
    position_id: int,
    new_sl: float,
    *,
    keep_tp: Optional[float] = None,
    exec_id: Optional[int] = None,
    ctrader_account_id: Optional[str] = None,
) -> dict:
    """Broker-confirmed SL amend — returns {ok, result, broker_reply, error?}."""
    from app.database import SessionLocal
    from app.models import User, UserPreference

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == int(user_id)).first()
        if not user:
            return {"ok": False, "result": "failed", "broker_reply": {}, "error": "no user"}
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs or not prefs.ctrader_access_token:
            return {"ok": False, "result": "failed", "broker_reply": {}, "error": "no ctrader creds"}
        access_token = prefs.ctrader_access_token

        execution_account_id = ctrader_account_id
        notes = None
        if exec_id:
            from app.strategy_models import StrategyExecution
            execution = db.query(StrategyExecution).filter(
                StrategyExecution.id == int(exec_id)
            ).first()
            if execution:
                if not execution_account_id:
                    execution_account_id = execution.ctrader_account_id
                notes = execution.notes

        ctid_str = resolve_ctrader_ctid(
            execution_account_id=execution_account_id,
            notes=notes,
            prefs_default=prefs.ctrader_account_id,
        )
        if not ctid_str:
            return {"ok": False, "result": "failed", "broker_reply": {}, "error": "no ctrader account"}
        ctid = int(ctid_str)
        hosts = _routing_hosts_for_account(prefs, ctid)
    finally:
        db.close()

    last_res = {"ok": False, "result": "failed", "broker_reply": {}, "error": "no host succeeded"}
    for host in hosts:
        res = await modify_position_sltp_result(
            access_token,
            ctid,
            int(position_id),
            stop_loss_price=float(new_sl),
            take_profit_price=keep_tp,
            host=host,
            exec_id=exec_id,
        )
        if res.get("ok"):
            _persist_account_host_metadata(user_id, ctid, host)
            return res
        last_res = res
    return last_res


async def partial_close_position(
    user_id: int,
    position_id: int,
    close_vol: int,
) -> bool:
    """Partial close via ProtoOAClosePositionReq with explicit volume units."""
    from app.database import SessionLocal
    from app.models import UserPreference

    db = SessionLocal()
    try:
        prefs = db.query(UserPreference).filter(
            UserPreference.user_id == int(user_id),
        ).first()
        if not prefs or not prefs.ctrader_access_token or not prefs.ctrader_account_id:
            return False
        access_token = prefs.ctrader_access_token
        ctid = int(prefs.ctrader_account_id)
        host = _host_for_account(prefs, ctid)
    finally:
        db.close()
    return await close_position(
        access_token, ctid, int(position_id), int(close_vol), host=host,
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
    Reuses the per-(host, ctid) persistent socket when available.
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
                reader, writer = await _get_persistent_connection(
                    host,
                    ctid_trader_account_id,
                    connect_timeout=BALANCE_CONNECT_TIMEOUT_S,
                )
                if not await _account_auth(
                    reader, writer, access_token, ctid_trader_account_id,
                    timeout=BALANCE_REQ_TIMEOUT_S,
                ):
                    return None
                _touch_conn(host, ctid_trader_account_id)
                req = ProtoOAReconcileReq()
                req.ctidTraderAccountId = ctid_trader_account_id
                payload = await _send_recv(
                    reader, writer, req,
                    _PAYLOAD_TYPES["reconcile_req"],
                    _PAYLOAD_TYPES["reconcile_res"],
                    timeout=BALANCE_REQ_TIMEOUT_S,
                )
                if not payload:
                    return None
                res = ProtoOAReconcileRes()
                res.ParseFromString(payload)
                bal = _parse_reconcile_balance(res)
                if bal is not None:
                    _balance_cache[cache_key] = (bal, time.monotonic())
                    _touch_conn(host, ctid_trader_account_id)
                    return bal
            except asyncio.CancelledError:
                _invalidate_persistent_connection(host, ctid_trader_account_id)
                raise
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug(f"[cTrader] balance fetch error host={host} ctid={ctid_trader_account_id}: {e}")
    return None


def _balance_hosts_for_account(prefs, ctid_trader_account_id: int) -> List[str]:
    """Backward-compatible alias — see _routing_hosts_for_account."""
    return _routing_hosts_for_account(prefs, ctid_trader_account_id)


async def get_account_balance_resilient(
    access_token: str,
    ctid_trader_account_id: int,
    *,
    prefs=None,
    user_id: Optional[int] = None,
) -> Optional[float]:
    """
    Fetch account balance with correct live/demo host, token refresh, and fallback.
    UI paths should use this instead of raw _get_account_balance(host=LIVE default).
    """
    at = (access_token or "").strip()
    if not at:
        return None
    hosts = _balance_hosts_for_account(prefs, ctid_trader_account_id)

    async def _try_hosts(token: str) -> Optional[float]:
        for h in hosts:
            bal = await _get_account_balance(token, ctid_trader_account_id, host=h)
            if bal is not None:
                return bal
        return None

    bal = await _try_hosts(at)
    if bal is not None:
        return bal
    if user_id:
        new_at = await request_ctrader_token_refresh(user_id, force=False, wait_s=10.0)
        if new_at and new_at != at:
            bal = await _try_hosts(new_at)
            if bal is not None:
                return bal
    return None


async def _get_open_position_ids(
    access_token: str,
    ctid_trader_account_id: int,
    host: str = CTRADER_HOST,
    *,
    user_id: Optional[int] = None,
) -> Optional[set]:
    """Fetch the set of currently-open broker positionIds via ProtoOAReconcileReq.

    Returns a set of ints (possibly empty) on success, or None on ANY failure so
    the caller can distinguish "broker says zero open positions" from "broker
    unreachable" and never false-close a tracked execution.
    """
    if not _PROTO_OK:
        return None
    uid = int(user_id) if user_id else None
    if uid and _account_auth_in_cooldown(uid, host, ctid_trader_account_id):
        rem = _account_auth_cooldown_remaining(uid, host, ctid_trader_account_id)
        logger.info(
            "[cTrader] open-positions skipped user=%s ctid=%s host=%s (auth cooldown %.0fs)",
            uid,
            ctid_trader_account_id,
            host,
            rem,
        )
        return None
    token = (access_token or "").strip()
    for auth_attempt in (1, 2):
        auth_failed = False
        try:
            async with asyncio.timeout(CTRADER_OPEN_POS_CALL_TIMEOUT_S):
                async with _get_account_lock(host, ctid_trader_account_id):
                    try:
                        reader, writer = await _get_persistent_connection(
                            host,
                            ctid_trader_account_id,
                            connect_timeout=CTRADER_ACCOUNT_OP_CONNECT_TIMEOUT_S,
                        )
                        if not await _account_auth(
                            reader,
                            writer,
                            token,
                            ctid_trader_account_id,
                            timeout=CTRADER_ACCOUNT_AUTH_TIMEOUT_S,
                        ):
                            auth_failed = True
                        else:
                            req = ProtoOAReconcileReq()
                            req.ctidTraderAccountId = ctid_trader_account_id
                            payload = await _send_recv(
                                reader, writer, req,
                                _PAYLOAD_TYPES["reconcile_req"],
                                _PAYLOAD_TYPES["reconcile_res"],
                                timeout=CTRADER_OPEN_POS_RECONCILE_TIMEOUT_S,
                            )
                            if not payload:
                                return None
                            res = ProtoOAReconcileRes()
                            res.ParseFromString(payload)
                            if uid:
                                _clear_account_auth_unhealthy(uid, host, ctid_trader_account_id)
                            return {int(p.positionId) for p in res.position}
                    except asyncio.CancelledError:
                        _invalidate_persistent_connection(host, ctid_trader_account_id)
                        raise
        except asyncio.TimeoutError:
            _invalidate_persistent_connection(host, ctid_trader_account_id)
            logger.warning(
                "[cTrader] open-positions timeout ctid=%s host=%s (%.1fs)",
                ctid_trader_account_id,
                host,
                CTRADER_OPEN_POS_CALL_TIMEOUT_S,
            )
            if uid:
                _mark_account_auth_unhealthy(
                    uid,
                    host,
                    ctid_trader_account_id,
                    reason="open_positions_timeout",
                    cooldown_s=_AUTH_TIMEOUT_COOLDOWN_S,
                )
            return None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                "[cTrader] open-positions fetch error ctid=%s host=%s: %s",
                ctid_trader_account_id,
                host,
                e,
            )
            if uid:
                _mark_account_auth_unhealthy(
                    uid,
                    host,
                    ctid_trader_account_id,
                    reason=f"open_positions_error:{type(e).__name__}",
                    cooldown_s=_AUTH_TIMEOUT_COOLDOWN_S,
                )
            return None

        if auth_failed:
            logger.warning(
                "[cTrader] open-positions auth failed ctid=%s host=%s",
                ctid_trader_account_id,
                host,
            )
            if auth_attempt == 1:
                new_at = await _refresh_auth_token_for_retry(
                    user_id=user_id,
                    ctid_trader_account_id=ctid_trader_account_id,
                    host=host,
                    operation="open-positions fetch",
                )
                if new_at:
                    token = new_at
                    _invalidate_persistent_connection(host, ctid_trader_account_id)
                    continue
            if uid:
                _mark_account_auth_unhealthy(
                    uid,
                    host,
                    ctid_trader_account_id,
                    reason="open_positions_auth_failed",
                )
            return None
    return None


async def _list_open_positions_detail(
    access_token: str,
    ctid_trader_account_id: int,
    host: str = CTRADER_HOST,
) -> Optional[list]:
    """Open positions with symbol/side/price for post-submit reconcile."""
    if not _PROTO_OK:
        return None
    try:
        async with _get_account_lock(host, ctid_trader_account_id):
            reader, writer = await _get_persistent_connection(host, ctid_trader_account_id)
            if not await _account_auth(reader, writer, access_token, ctid_trader_account_id):
                return None
            req = ProtoOAReconcileReq()
            req.ctidTraderAccountId = ctid_trader_account_id
            payload = await _send_recv(
                reader, writer, req,
                _PAYLOAD_TYPES["reconcile_req"],
                _PAYLOAD_TYPES["reconcile_res"],
                timeout=ORDER_SUBMIT_TIMEOUT_S,
            )
            if not payload:
                return None
            res = ProtoOAReconcileRes()
            res.ParseFromString(payload)
            out = []
            for p in res.position:
                item: dict = {"position_id": int(p.positionId)}
                try:
                    if p.HasField("price") and p.price:
                        item["price"] = float(p.price)
                except Exception:
                    pass
                try:
                    if p.HasField("tradeData"):
                        td = p.tradeData
                        if td.HasField("symbolId"):
                            item["symbol_id"] = int(td.symbolId)
                        if td.HasField("tradeSide"):
                            item["trade_side"] = int(td.tradeSide)
                except Exception:
                    pass
                out.append(item)
            _touch_conn(host, ctid_trader_account_id)
            return out
    except Exception as exc:
        logger.warning("[cTrader] open-positions detail failed: %s", type(exc).__name__)
    return None


async def reconcile_order_fill_after_miss(
    *,
    access_token: str,
    ctid: int,
    host: str,
    symbol_name: str,
    direction: str,
    entry_hint: Optional[float] = None,
    volume_units: Optional[int] = None,
) -> Optional[dict]:
    """
    Broker reconcile when submit ack/fill events were lost — avoid paper fallback
    if a matching open position already exists.
    """
    broker_symbol = _SYMBOL_MAP.get(symbol_name, symbol_name)
    positions = await _list_open_positions_detail(access_token, ctid, host)
    if positions is None:
        logger.warning(
            "[cTrader] reconcile skipped — broker unreachable (exec ambiguous submit)"
        )
        return None
    if not positions:
        logger.info("[cTrader] reconcile: broker reports zero open positions")
        return None

    try:
        async with _get_account_lock(host, ctid):
            reader, writer = await _get_persistent_connection(host, ctid)
            if not await _account_auth(reader, writer, access_token, ctid):
                return None
            symbol_id = await _resolve_symbol_id(
                reader, writer, ctid, broker_symbol, host
            )
    except Exception:
        symbol_id = None

    want_side = (
        ProtoOATradeSide.BUY if direction == "LONG" else ProtoOATradeSide.SELL
    )
    for pos in positions:
        if symbol_id is not None and pos.get("symbol_id") not in (None, symbol_id):
            continue
        side = pos.get("trade_side")
        if side is not None and int(side) != int(want_side):
            continue
        raw_px = pos.get("price")
        actual_fill = None
        if raw_px is not None:
            actual_fill = _normalize_deal_price(float(raw_px), entry_hint)
        logger.info(
            "[cTrader] reconcile recovered open position pos=%s symbol=%s fill=%s",
            pos.get("position_id"),
            broker_symbol,
            actual_fill,
        )
        return {
            "order_id": None,
            "actual_fill": actual_fill or entry_hint,
            "position_id": str(pos.get("position_id")),
            "volume": volume_units,
            "error": None,
            "reconciled": True,
        }
    logger.info(
        "[cTrader] reconcile: no matching open position for %s %s",
        broker_symbol,
        direction,
    )
    return None


async def get_open_position_ids_for_user(user, *, ctid: Optional[str] = None) -> Optional[set]:
    """High-level wrapper: the set of open broker positionIds for a TradeHub user,
    or None when credentials are missing / the broker is unreachable (so callers
    never mistake an error for "all positions closed")."""
    from app.database import SessionLocal
    from app.models import UserPreference
    db = SessionLocal()
    try:
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs or not prefs.ctrader_access_token:
            return None
        ctid_str = resolve_ctrader_ctid(
            execution_account_id=ctid,
            prefs_default=prefs.ctrader_account_id,
        )
        if not ctid_str:
            return None
        access_token           = prefs.ctrader_access_token
        ctid_trader_account_id = int(ctid_str)
        hosts                  = _routing_hosts_for_account(prefs, ctid_trader_account_id)
    finally:
        db.close()
    for host in hosts:
        latest = _latest_ctrader_access_token(int(user.id))
        if latest:
            access_token = latest
        ids = await _get_open_position_ids(
            access_token,
            ctid_trader_account_id,
            host=host,
            user_id=int(user.id),
        )
        if ids is not None:
            return ids
    return None


async def get_open_position_ids_for_user_with_retry(
    user,
    *,
    ctid: Optional[str] = None,
    attempts: int = 3,
    backoff_s: float = 0.5,
) -> Optional[set]:
    """Retry broker open-position poll — None only after all attempts fail."""
    ctid_int: Optional[int] = None
    try:
        if ctid is not None and str(ctid).strip():
            ctid_int = int(str(ctid).strip())
    except Exception:
        ctid_int = None
    if ctid_int is not None and _user_ctid_has_auth_backoff(int(getattr(user, "id", 0) or 0), ctid_int):
        logger.info(
            "[cTrader] open-positions poll skipped user=%s acct=%s (auth cooldown active)",
            getattr(user, "id", "?"),
            ctid,
        )
        return None
    last_err: Optional[Exception] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            ids = await get_open_position_ids_for_user(user, ctid=ctid)
            if ids is not None:
                if attempt > 1:
                    logger.info(
                        "[cTrader] open-positions poll ok user=%s acct=%s attempt=%s count=%s",
                        getattr(user, "id", "?"),
                        ctid,
                        attempt,
                        len(ids),
                    )
                return ids
            if ctid_int is not None and _user_ctid_has_auth_backoff(int(getattr(user, "id", 0) or 0), ctid_int):
                logger.info(
                    "[cTrader] open-positions poll stopped early user=%s acct=%s (auth cooldown active)",
                    getattr(user, "id", "?"),
                    ctid,
                )
                return None
        except Exception as exc:
            last_err = exc
            logger.warning(
                "[cTrader] open-positions poll error user=%s acct=%s attempt=%s/%s: %s",
                getattr(user, "id", "?"),
                ctid,
                attempt,
                attempts,
                exc,
            )
        if attempt < attempts:
            await asyncio.sleep(backoff_s * attempt)
    if last_err:
        logger.warning(
            "[cTrader] open-positions poll exhausted retries user=%s acct=%s: %s",
            getattr(user, "id", "?"),
            ctid,
            last_err,
        )
    return None


def _normalize_deal_price(raw: float, entry_hint: Optional[float] = None) -> float:
    """Resolve symbol-dependent deal price scaling (see ctrader-price-scaling.md)."""
    if raw <= 0:
        return raw
    if entry_hint and entry_hint > 0:
        cands = [raw, raw / 100.0, raw / 1000.0, raw / 100_000.0]
        return min(cands, key=lambda v: abs(v - entry_hint))
    if raw > 1000:
        return raw
    if raw > 10:
        return raw
    return raw / 100_000.0


def _deal_timestamp_ms(deal) -> int:
    for attr in ("executionTimestamp", "utcLastUpdateTimestamp", "createTimestamp"):
        if deal.HasField(attr):
            return int(getattr(deal, attr))
    return 0


def _outcome_from_close_detail(
    *,
    exit_price: float,
    gross: int,
    entry_hint: Optional[float],
    direction: Optional[str],
) -> str:
    if gross > 0:
        return "WIN"
    if gross < 0:
        return "LOSS"
    if entry_hint and direction:
        if direction == "LONG":
            return "WIN" if exit_price >= entry_hint else "LOSS"
        return "WIN" if exit_price <= entry_hint else "LOSS"
    return "LOSS"


def _parse_close_from_deals(
    deals: list,
    position_id: int,
    *,
    entry_hint: Optional[float] = None,
    direction: Optional[str] = None,
) -> Optional[dict]:
    """Pick the closing deal for a position from a deal list."""
    pid = int(position_id)
    matching = [d for d in deals if int(getattr(d, "positionId", 0) or 0) == pid]
    if not matching:
        return None

    closing = [d for d in matching if d.HasField("closePositionDetail")]
    candidates = closing if closing else matching
    deal = max(candidates, key=_deal_timestamp_ms)
    close_ts = _deal_timestamp_ms(deal)

    if deal.HasField("closePositionDetail"):
        detail = deal.closePositionDetail
        exit_price = _normalize_deal_price(float(detail.entryPrice), entry_hint)
        gross = int(detail.grossProfit) if detail.HasField("grossProfit") else 0
    elif deal.HasField("executionPrice") and float(deal.executionPrice or 0) > 0:
        exit_price = _normalize_deal_price(float(deal.executionPrice), entry_hint)
        gross = 0
    else:
        return None

    if exit_price <= 0:
        return None

    return {
        "exit_price": exit_price,
        "outcome": _outcome_from_close_detail(
            exit_price=exit_price,
            gross=gross,
            entry_hint=entry_hint,
            direction=direction,
        ),
        "closed_at_ms": close_ts or None,
        "gross_profit": gross,
        "deal_id": int(deal.dealId) if deal.HasField("dealId") else None,
    }


async def _fetch_deals_by_position_id(
    access_token: str,
    ctid_trader_account_id: int,
    position_id: int,
    *,
    host: str = CTRADER_HOST,
    from_ms: Optional[int] = None,
    to_ms: Optional[int] = None,
    user_id: Optional[int] = None,
) -> Optional[list]:
    """ProtoOADealListByPositionIdReq — paginated, with explicit time window (ms)."""
    if not _PROTO_OK:
        return None
    uid = int(user_id) if user_id else None
    if uid and _account_auth_in_cooldown(uid, host, ctid_trader_account_id):
        rem = _account_auth_cooldown_remaining(uid, host, ctid_trader_account_id)
        logger.info(
            "[cTrader] deal-by-position skipped user=%s ctid=%s host=%s pos=%s (auth cooldown %.0fs)",
            uid,
            ctid_trader_account_id,
            host,
            position_id,
            rem,
        )
        return None
    now_ms = int(time.time() * 1000)
    start_ms = from_ms if from_ms is not None else (now_ms - 30 * 24 * 3600 * 1000)
    end_ms = to_ms if to_ms is not None else now_ms
    token = (access_token or "").strip()
    for auth_attempt in (1, 2):
        all_deals: list = []
        cursor_from = start_ms
        auth_failed = False
        try:
            async with asyncio.timeout(CTRADER_DEAL_FETCH_CALL_TIMEOUT_S):
                async with _get_account_lock(host, ctid_trader_account_id):
                    reader, writer = await _get_persistent_connection(
                        host,
                        ctid_trader_account_id,
                        connect_timeout=CTRADER_ACCOUNT_OP_CONNECT_TIMEOUT_S,
                    )
                    if not await _account_auth(
                        reader,
                        writer,
                        token,
                        ctid_trader_account_id,
                        timeout=CTRADER_ACCOUNT_AUTH_TIMEOUT_S,
                    ):
                        auth_failed = True
                    else:
                        for _page in range(CTRADER_DEAL_MAX_PAGES):
                            req = ProtoOADealListByPositionIdReq()
                            req.ctidTraderAccountId = ctid_trader_account_id
                            req.positionId = int(position_id)
                            req.fromTimestamp = int(cursor_from)
                            req.toTimestamp = int(end_ms)
                            payload = await _send_recv(
                                reader, writer, req,
                                _PAYLOAD_TYPES["deal_list_by_position_id_req"],
                                _PAYLOAD_TYPES["deal_list_by_position_id_res"],
                                timeout=CTRADER_DEAL_FETCH_REQ_TIMEOUT_S,
                            )
                            if not payload:
                                break
                            res = ProtoOADealListByPositionIdRes()
                            res.ParseFromString(payload)
                            page = list(res.deal)
                            all_deals.extend(page)
                            if not getattr(res, "hasMore", False) or not page:
                                break
                            last_ts = max(_deal_timestamp_ms(d) for d in page)
                            if last_ts <= cursor_from:
                                break
                            cursor_from = last_ts + 1
                        if uid:
                            _clear_account_auth_unhealthy(uid, host, ctid_trader_account_id)
                        return all_deals
        except asyncio.TimeoutError:
            _invalidate_persistent_connection(host, ctid_trader_account_id)
            logger.warning(
                "[cTrader] deal-by-position timeout ctid=%s host=%s pos=%s (%.1fs)",
                ctid_trader_account_id,
                host,
                position_id,
                CTRADER_DEAL_FETCH_CALL_TIMEOUT_S,
            )
            if uid:
                _mark_account_auth_unhealthy(
                    uid,
                    host,
                    ctid_trader_account_id,
                    reason="deal_by_position_timeout",
                    cooldown_s=_AUTH_TIMEOUT_COOLDOWN_S,
                )
            return None
        except asyncio.CancelledError:
            _invalidate_persistent_connection(host, ctid_trader_account_id)
            raise
        except Exception as exc:
            logger.warning(
                "[cTrader] deal-by-position fetch failed ctid=%s host=%s pos=%s: %s",
                ctid_trader_account_id, host, position_id, exc,
            )
            if uid:
                _mark_account_auth_unhealthy(
                    uid,
                    host,
                    ctid_trader_account_id,
                    reason=f"deal_by_position_error:{type(exc).__name__}",
                    cooldown_s=_AUTH_TIMEOUT_COOLDOWN_S,
                )
            return None

        if auth_failed:
            logger.warning(
                "[cTrader] deal-by-position auth failed ctid=%s host=%s pos=%s",
                ctid_trader_account_id, host, position_id,
            )
            if auth_attempt == 1:
                new_at = await _refresh_auth_token_for_retry(
                    user_id=user_id,
                    ctid_trader_account_id=ctid_trader_account_id,
                    host=host,
                    operation="deal-by-position fetch",
                )
                if new_at:
                    token = new_at
                    _invalidate_persistent_connection(host, ctid_trader_account_id)
                    continue
            if uid:
                _mark_account_auth_unhealthy(
                    uid,
                    host,
                    ctid_trader_account_id,
                    reason="deal_by_position_auth_failed",
                )
            return None
    return None


async def _fetch_deals_in_window(
    access_token: str,
    ctid_trader_account_id: int,
    *,
    host: str = CTRADER_HOST,
    from_ms: Optional[int] = None,
    to_ms: Optional[int] = None,
    max_rows: int = 500,
    user_id: Optional[int] = None,
) -> Optional[list]:
    """ProtoOADealListReq fallback — time-ranged deal history."""
    if not _PROTO_OK:
        return None
    uid = int(user_id) if user_id else None
    if uid and _account_auth_in_cooldown(uid, host, ctid_trader_account_id):
        rem = _account_auth_cooldown_remaining(uid, host, ctid_trader_account_id)
        logger.info(
            "[cTrader] deal-list window skipped user=%s ctid=%s host=%s (auth cooldown %.0fs)",
            uid,
            ctid_trader_account_id,
            host,
            rem,
        )
        return None
    now_ms = int(time.time() * 1000)
    start_ms = from_ms if from_ms is not None else (now_ms - 7 * 24 * 3600 * 1000)
    end_ms = to_ms if to_ms is not None else now_ms
    try:
        async with asyncio.timeout(CTRADER_DEAL_WINDOW_CALL_TIMEOUT_S):
            async with _get_account_lock(host, ctid_trader_account_id):
                reader, writer = await _get_persistent_connection(
                    host,
                    ctid_trader_account_id,
                    connect_timeout=CTRADER_ACCOUNT_OP_CONNECT_TIMEOUT_S,
                )
                if not await _account_auth(
                    reader,
                    writer,
                    access_token,
                    ctid_trader_account_id,
                    timeout=CTRADER_ACCOUNT_AUTH_TIMEOUT_S,
                ):
                    return None
                req = ProtoOADealListReq()
                req.ctidTraderAccountId = ctid_trader_account_id
                req.fromTimestamp = int(start_ms)
                req.toTimestamp = int(end_ms)
                req.maxRows = int(max_rows)
                payload = await _send_recv(
                    reader, writer, req,
                    _PAYLOAD_TYPES["deal_list_req"],
                    _PAYLOAD_TYPES["deal_list_res"],
                    timeout=CTRADER_DEAL_FETCH_REQ_TIMEOUT_S,
                )
                if not payload:
                    return None
                res = ProtoOADealListRes()
                res.ParseFromString(payload)
                if uid:
                    _clear_account_auth_unhealthy(uid, host, ctid_trader_account_id)
                return list(res.deal)
    except asyncio.TimeoutError:
        _invalidate_persistent_connection(host, ctid_trader_account_id)
        logger.warning(
            "[cTrader] deal-list window timeout ctid=%s host=%s (%.1fs)",
            ctid_trader_account_id,
            host,
            CTRADER_DEAL_WINDOW_CALL_TIMEOUT_S,
        )
        if uid:
            _mark_account_auth_unhealthy(
                uid,
                host,
                ctid_trader_account_id,
                reason="deal_list_window_timeout",
                cooldown_s=_AUTH_TIMEOUT_COOLDOWN_S,
            )
        return None
    except Exception as exc:
        logger.warning(
            "[cTrader] deal-list window fetch failed ctid=%s host=%s: %s",
            ctid_trader_account_id, host, exc,
        )
        if uid:
            _mark_account_auth_unhealthy(
                uid,
                host,
                ctid_trader_account_id,
                reason=f"deal_list_window_error:{type(exc).__name__}",
                cooldown_s=_AUTH_TIMEOUT_COOLDOWN_S,
            )
    return None


async def _get_position_close_detail(
    access_token: str,
    ctid_trader_account_id: int,
    position_id: int,
    *,
    host: str = CTRADER_HOST,
    entry_hint: Optional[float] = None,
    direction: Optional[str] = None,
    from_ms: Optional[int] = None,
    to_ms: Optional[int] = None,
    user_id: Optional[int] = None,
) -> Optional[dict]:
    """Broker close truth via ProtoOADealListByPositionIdReq (+ DealList fallback)."""
    token = (access_token or "").strip()
    if user_id:
        latest = _latest_ctrader_access_token(int(user_id))
        if latest:
            token = latest
    deals = await _fetch_deals_by_position_id(
        token,
        ctid_trader_account_id,
        position_id,
        host=host,
        from_ms=from_ms,
        to_ms=to_ms,
        user_id=user_id,
    )
    if deals is None:
        return None
    parsed = _parse_close_from_deals(
        deals, position_id, entry_hint=entry_hint, direction=direction,
    )
    if parsed:
        parsed["source"] = "deal_by_position_id"
        return parsed

    if deals:
        logger.info(
            "[cTrader] deal-by-position ctid=%s host=%s pos=%s returned %s deals "
            "but no close detail yet",
            ctid_trader_account_id, host, position_id, len(deals),
        )

    if user_id:
        latest = _latest_ctrader_access_token(int(user_id))
        if latest:
            token = latest
    window_deals = await _fetch_deals_in_window(
        token,
        ctid_trader_account_id,
        host=host,
        from_ms=from_ms,
        to_ms=to_ms,
        user_id=user_id,
    )
    if window_deals:
        parsed = _parse_close_from_deals(
            window_deals, position_id, entry_hint=entry_hint, direction=direction,
        )
        if parsed:
            parsed["source"] = "deal_list_window"
            return parsed
        logger.info(
            "[cTrader] deal-list window ctid=%s host=%s pos=%s scanned %s deals "
            "— no close match",
            ctid_trader_account_id, host, position_id, len(window_deals),
        )
    return None


async def get_position_close_detail_for_user(
    user,
    position_id: int,
    *,
    entry_hint: Optional[float] = None,
    direction: Optional[str] = None,
    ctid: Optional[str] = None,
    notes: Optional[str] = None,
    fired_at=None,
) -> Optional[dict]:
    """High-level wrapper: broker close price/outcome for a user's position."""
    from datetime import timedelta
    from app.database import SessionLocal
    from app.models import UserPreference
    db = SessionLocal()
    try:
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs or not prefs.ctrader_access_token:
            return None
        ctid_str = resolve_ctrader_ctid(
            execution_account_id=ctid,
            notes=notes,
            prefs_default=prefs.ctrader_account_id,
        )
        if not ctid_str:
            logger.warning(
                "[cTrader] close-detail missing ctid user=%s pos=%s notes=%s",
                user.id, position_id, (notes or "")[:80],
            )
            return None
        access_token = prefs.ctrader_access_token
        ctid_val = int(ctid_str)
        hosts = _routing_hosts_for_account(prefs, ctid_val)
    finally:
        db.close()

    now_ms = int(time.time() * 1000)
    if fired_at is not None:
        try:
            from_ms = int(fired_at.timestamp() * 1000) - 3600_000
        except Exception:
            from_ms = now_ms - int(timedelta(days=7).total_seconds() * 1000)
    else:
        from_ms = now_ms - int(timedelta(days=7).total_seconds() * 1000)

    for host in hosts:
        latest = _latest_ctrader_access_token(int(user.id))
        if latest:
            access_token = latest
        detail = await _get_position_close_detail(
            access_token,
            ctid_val,
            position_id,
            host=host,
            entry_hint=entry_hint,
            direction=direction,
            from_ms=from_ms,
            to_ms=now_ms,
            user_id=int(user.id),
        )
        if detail is not None:
            logger.info(
                "[cTrader] close-detail user=%s ctid=%s host=%s pos=%s "
                "exit=%s src=%s",
                user.id,
                ctid_val,
                host,
                position_id,
                detail.get("exit_price"),
                detail.get("source"),
            )
            return detail
    logger.warning(
        "[cTrader] close-detail empty user=%s ctid=%s pos=%s hosts=%s",
        user.id,
        ctid_val,
        position_id,
        hosts,
    )
    return None


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
    *,
    ctid: Optional[str] = None,
    latency=None,
    execution_id: Optional[int] = None,
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
        if not prefs or not prefs.ctrader_access_token:
            logger.warning(f"[cTrader] user {user.id} has no cTrader credentials")
            return {
                "order_id": None,
                "actual_fill": None,
                "error": "no cTrader access token",
            }
        ctid_str = resolve_ctrader_ctid(
            execution_account_id=ctid,
            prefs_default=prefs.ctrader_account_id,
        )
        if not ctid_str:
            logger.critical(
                "[live-fire] order rejected user=%s exec=%s reason=missing per-execution ctid "
                "(will not use prefs.ctrader_account_id fallback)",
                user.id,
                execution_id if execution_id is not None else "?",
            )
            return {"order_id": None, "actual_fill": None, "error": "missing ctrader account id"}
        access_token           = prefs.ctrader_access_token
        ctid_trader_account_id = int(ctid_str)
        _route_hosts           = _routing_hosts_for_account(prefs, ctid_trader_account_id)
        primary_host           = _route_hosts[0]
    finally:
        db.close()

    tp_price, sl_price = compute_sltp_prices(direction, entry_price, tp_pct, sl_pct)
    if not validate_sltp_sanity(direction, entry_price, sl_price, tp_price):
        logger.warning(
            f"[cTrader] inverted SL/TP for {direction} {symbol} entry={entry_price} "
            f"— re-deriving from entry"
        )
        tp_price, sl_price = compute_sltp_prices(direction, entry_price, tp_pct, sl_pct)
        if not validate_sltp_sanity(direction, entry_price, sl_price, tp_price):
            logger.error(
                f"[cTrader] SL/TP sanity check failed for user {user.id} {symbol} "
                f"{direction} entry={entry_price} sl={sl_price} tp={tp_price}"
            )
            return {"order_id": None, "actual_fill": None,
                    "error": "invalid SL/TP for trade direction"}

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
        _broker_sym = _SYMBOL_MAP.get(symbol, symbol)
    else:
        # Forex: standard lot sizing — platform pips only (never broker pipPosition).
        from app.services.pip_units import platform_usd_per_pip_per_lot, sl_pips_platform

        _sl_pips_eff = sl_pips_platform(
            symbol, entry_price, sl_price, sl_pips_hint=sl_pips,
        )
        pip_value = platform_usd_per_pip_per_lot(symbol)

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
    result = await place_market_order_resilient(
        user_id=user.id,
        access_token=access_token,
        ctid=ctid_trader_account_id,
        prefs=prefs,
        symbol_name=_broker_sym if is_index else symbol,
        direction=direction,
        volume_units=contracts if is_index else None,
        volume_lots=lots if not is_index else None,
        stop_loss_price=sl_price,
        take_profit_price=tp_price,
        entry_price=entry_price,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        latency=latency,
        execution_id=execution_id,
    )

    if result.get("error"):
        logger.error(f"[cTrader] order failed for user {user.id}: {result['error']}")
        return result

    actual_fill = result.get("actual_fill")
    position_id = result.get("position_id")
    amend_host = result.get("host") or primary_host
    if actual_fill and actual_fill > 0 and position_id:
        fill_tp, fill_sl = compute_sltp_prices(direction, actual_fill, tp_pct, sl_pct)
        if not validate_sltp_sanity(direction, actual_fill, fill_sl, fill_tp):
            fill_tp, fill_sl = compute_sltp_prices(direction, actual_fill, tp_pct, sl_pct)
        if validate_sltp_sanity(direction, actual_fill, fill_sl, fill_tp):
            try:
                amended = await modify_position_sltp(
                    access_token,
                    ctid_trader_account_id,
                    int(position_id),
                    stop_loss_price=fill_sl,
                    take_profit_price=fill_tp,
                    host=amend_host,
                )
                if amended:
                    result["tp_price"] = fill_tp
                    result["sl_price"] = fill_sl
                    logger.info(
                        f"[cTrader] repriced SL/TP from fill {actual_fill} for "
                        f"pos={position_id} user={user.id}"
                    )
            except Exception as amend_err:
                logger.warning(
                    f"[cTrader] post-fill SL/TP amend failed user={user.id}: {amend_err}"
                )

    if isinstance(result, dict) and result.get("order_id"):
        result["account_id"] = str(ctid_trader_account_id)
    return result
