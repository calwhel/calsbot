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
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Spotware Open API endpoints ───────────────────────────────────────────────
CTRADER_HOST = "live.ctraderapi.com"
CTRADER_PORT = 5035

# OAuth endpoints
OAUTH_BASE      = "https://connect.spotware.com"
OAUTH_AUTH_URL  = f"{OAUTH_BASE}/apps/{{client_id}}/auth"
OAUTH_TOKEN_URL = f"{OAUTH_BASE}/apps/token"

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
        ProtoOAGetAccountListByAccessTokenReq,
        ProtoOAGetAccountListByAccessTokenRes,
        ProtoOAGetCtidProfileByTokenReq,
        ProtoOAGetCtidProfileByTokenRes,
        ProtoOARefreshTokenReq,
        ProtoOARefreshTokenRes,
        ProtoOAClosePositionReq,
        ProtoOAReconcileReq,
        ProtoOAReconcileRes,
    )
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
        ProtoOATradeSide,
        ProtoOAOrderType,
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
    "execution_event":       2126,
    "account_list_req":      2149,
    "account_list_res":      2150,
    "ctid_profile_req":      2114,
    "ctid_profile_res":      2115,
    "refresh_token_req":     2072,
    "refresh_token_res":     2073,
    "close_position_req":    2140,
    "reconcile_req":         2124,
    "reconcile_res":         2125,
}

# Forex pip sizes
_PIP_SIZES = {
    # Forex majors
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001,
    "NZDUSD": 0.0001, "USDCAD": 0.0001, "USDCHF": 0.0001,
    "USDJPY": 0.01,   "EURJPY": 0.01,   "GBPJPY": 0.01,
    # Indices — tick size used for pip-equivalent math
    "SPX": 1.0, "NDX": 1.0, "DJI": 1.0, "DAX": 1.0, "FTSE": 1.0,
}

# FP Markets / cTrader symbol name mapping
# Indices use broker-specific contract names on FP Markets cTrader.
_SYMBOL_MAP = {
    # Forex
    "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD", "USDCAD": "USDCAD", "USDCHF": "USDCHF",
    "NZDUSD": "NZDUSD",
    # Indices (FP Markets cTrader contract names)
    "SPX":  "US500",  # S&P 500
    "NDX":  "US100",  # Nasdaq 100
    "DJI":  "US30",   # Dow Jones
    "DAX":  "GER40",  # DAX
    "FTSE": "UK100",  # FTSE 100
}

# Index contract sizing: for index CFDs volume is in contracts (1 unit = 1 contract).
# FP Markets minimum is 1 contract; value ≈ price × contract_size USD.
_INDEX_SYMBOLS = frozenset({"SPX", "NDX", "DJI", "DAX", "FTSE"})


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


async def _open_connection() -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    ctx = ssl.create_default_context()
    return await asyncio.open_connection(CTRADER_HOST, CTRADER_PORT, ssl=ctx)


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


# ── Persistent connection singleton ──────────────────────────────────────────
# Keeps a single SSL TCP socket open to live.ctraderapi.com so that each
# order avoids the ~300-500ms SSL handshake + application-auth round-trip.
# Account auth is still done per call (required by the protocol), but that
# is a single round-trip on an already-open socket (~30-80ms vs ~300-500ms
# for a fresh SSL+app-auth sequence).
#
# _conn_lock serialises all use of the shared socket.  Idle timeout is 45s —
# shorter than Spotware's server-side 60s keep-alive grace period.

_conn_lock:   asyncio.Lock   = None   # created lazily (no running loop at import)
_conn_reader: Optional[asyncio.StreamReader] = None
_conn_writer: Optional[asyncio.StreamWriter] = None
_conn_ts:     float = 0.0    # monotonic timestamp of last successful message
_CONN_MAX_IDLE = 45.0        # force reconnect if socket unused for this long


def _get_lock() -> asyncio.Lock:
    global _conn_lock
    if _conn_lock is None:
        _conn_lock = asyncio.Lock()
    return _conn_lock


async def _get_persistent_connection() -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """
    Return a (reader, writer) that is connected and application-authenticated.
    Caller MUST hold _get_lock() before calling this.
    Reconnects automatically on idle timeout or detected closure.
    """
    global _conn_reader, _conn_writer, _conn_ts

    idle = time.monotonic() - _conn_ts
    is_alive = (
        _conn_writer is not None
        and not _conn_writer.is_closing()
        and idle < _CONN_MAX_IDLE
    )

    if is_alive:
        return _conn_reader, _conn_writer

    # Close stale connection cleanly
    if _conn_writer is not None:
        try:
            _conn_writer.close()
        except Exception:
            pass
        _conn_reader = _conn_writer = None

    r, w = await _open_connection()
    ok = await _app_auth(r, w)
    if not ok:
        try:
            w.close()
        except Exception:
            pass
        raise RuntimeError("cTrader persistent connection: app auth failed")

    _conn_reader, _conn_writer = r, w
    _conn_ts = time.monotonic()
    logger.debug("[cTrader] persistent connection (re)established")
    return r, w


def _invalidate_persistent_connection() -> None:
    """Mark the persistent connection as dead so next call reconnects."""
    global _conn_reader, _conn_writer, _conn_ts
    if _conn_writer is not None:
        try:
            _conn_writer.close()
        except Exception:
            pass
    _conn_reader = _conn_writer = None
    _conn_ts = 0.0


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
    """Return the Spotware OAuth authorization URL."""
    import urllib.parse
    params = {
        "redirect_uri":  redirect_uri,
        "response_type": "code",
        "scope":         "trading",
        "client_id":     CTRADER_CLIENT_ID,  # required by some Spotware API versions
    }
    if state:
        params["state"] = state
    base = OAUTH_AUTH_URL.format(client_id=CTRADER_CLIENT_ID)
    url = f"{base}?{urllib.parse.urlencode(params)}"
    logger.info(f"[ctrader] OAuth URL → {url}")
    return url


async def exchange_code(code: str, redirect_uri: str) -> dict:
    """Exchange an OAuth authorization code for access + refresh tokens."""
    import httpx
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            OAUTH_TOKEN_URL,
            data={
                "grant_type":    "authorization_code",
                "code":          code,
                "redirect_uri":  redirect_uri,
                "client_id":     CTRADER_CLIENT_ID,
                "client_secret": CTRADER_CLIENT_SECRET,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def refresh_access_token(refresh_token: str) -> dict:
    """Use a refresh token to get a new access token."""
    import httpx
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            OAUTH_TOKEN_URL,
            data={
                "grant_type":    "refresh_token",
                "refresh_token": refresh_token,
                "client_id":     CTRADER_CLIENT_ID,
                "client_secret": CTRADER_CLIENT_SECRET,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def get_accounts_for_token(access_token: str) -> list:
    """
    Return a list of cTrader trading accounts linked to this access token.
    Each item: {ctidTraderAccountId, isLive, traderLogin, balance}.
    Uses the persistent connection so the SSL handshake is amortised.
    """
    if not _PROTO_OK:
        return []
    async with _get_lock():
        for attempt in (1, 2):
            try:
                reader, writer = await _get_persistent_connection()
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
                global _conn_ts
                _conn_ts = time.monotonic()
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
                    _invalidate_persistent_connection()
                    continue
                logger.error(f"[cTrader] get_accounts failed: {e}")
                return []
    return []


async def place_order(
    access_token: str,
    ctid_trader_account_id: int,
    symbol_name: str,
    direction: str,
    volume_lots: float,
    stop_loss_price: Optional[float] = None,
    take_profit_price: Optional[float] = None,
    label: str = "TradeHub",
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

    async with _get_lock():
        for attempt in (1, 2):
            try:
                reader, writer = await _get_persistent_connection()
                if not await _account_auth(reader, writer, access_token, ctid_trader_account_id):
                    return {"order_id": None, "actual_fill": None, "error": "account auth failed"}

                req = ProtoOANewOrderReq()
                req.ctidTraderAccountId = ctid_trader_account_id
                req.symbolName          = _SYMBOL_MAP.get(symbol_name, symbol_name)
                req.orderType           = ProtoOAOrderType.MARKET
                req.tradeSide           = ProtoOATradeSide.BUY if direction == "LONG" else ProtoOATradeSide.SELL
                req.volume              = int(volume_lots * 100_000)  # units
                req.label               = label[:20]
                if stop_loss_price is not None:
                    req.relativeStopLoss = 0
                    req.stopLoss = int(stop_loss_price * 100_000)
                if take_profit_price is not None:
                    req.takeProfit = int(take_profit_price * 100_000)

                payload = await _send_recv(
                    reader, writer, req,
                    _PAYLOAD_TYPES["new_order_req"],
                    _PAYLOAD_TYPES["execution_event"],
                    timeout=15.0,
                )
                if not payload:
                    return {"order_id": None, "actual_fill": None, "error": "no execution event"}

                global _conn_ts
                _conn_ts = time.monotonic()
                ev = ProtoOAExecutionEvent()
                ev.ParseFromString(payload)
                order_id    = str(ev.order.orderId) if ev.HasField("order") else None
                actual_fill = None
                if ev.HasField("deal") and ev.deal.executionPrice:
                    actual_fill = ev.deal.executionPrice / 100_000.0
                return {"order_id": order_id, "actual_fill": actual_fill, "error": None}

            except Exception as e:
                if attempt == 1:
                    logger.warning(f"[cTrader] place_order retry after: {e}")
                    _invalidate_persistent_connection()
                    continue
                logger.error(f"[cTrader] place_order failed: {e}")
                return {"order_id": None, "actual_fill": None, "error": str(e)}
    return {"order_id": None, "actual_fill": None, "error": "unexpected exit"}


async def close_position(
    access_token: str,
    ctid_trader_account_id: int,
    position_id: int,
    volume_units: int,
) -> bool:
    """Close (or partially close) an open position. Uses persistent connection."""
    if not _PROTO_OK:
        return False
    async with _get_lock():
        for attempt in (1, 2):
            try:
                reader, writer = await _get_persistent_connection()
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
                    global _conn_ts
                    _conn_ts = time.monotonic()
                return payload is not None
            except Exception as e:
                if attempt == 1:
                    logger.warning(f"[cTrader] close_position retry after: {e}")
                    _invalidate_persistent_connection()
                    continue
                logger.error(f"[cTrader] close_position failed: {e}")
                return False
    return False


# ── High-level: place order for a TradeHub user ───────────────────────────────

async def _get_account_balance(
    access_token: str,
    ctid_trader_account_id: int,
) -> Optional[float]:
    """
    Fetch the current account balance (equity) from cTrader via ProtoOAReconcileReq.
    Returns USD balance or None on failure.
    """
    if not _PROTO_OK:
        return None
    try:
        async with _get_lock():
            reader, writer = await _get_persistent_connection()
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
            # balance is in cents (×100)
            if hasattr(res, "balance") and res.balance:
                return res.balance / 100.0
    except Exception as e:
        logger.debug(f"[cTrader] balance fetch error: {e}")
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
) -> Optional[dict]:
    """
    Place a live forex or index CFD order for a user via their connected cTrader account.
    Converts TP/SL from % to absolute prices, calculates lot/contract size from risk.

    When use_risk_pct=True and sl_pips is provided, lot size is computed as:
        lots = (risk_pct% × account_balance) / (sl_pips × pip_value_per_lot)
    This is the professional "fixed fractional" position sizing traders expect.
    Falls back to 0.01 lot minimum if balance fetch fails.

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
    finally:
        db.close()

    mult     = 1.0 if direction == "LONG" else -1.0
    tp_price = round(entry_price * (1 + mult * tp_pct / 100), 6)
    sl_price = round(entry_price * (1 - mult * sl_pct / 100), 6)

    is_index = symbol.upper() in _INDEX_SYMBOLS

    if is_index:
        # Index CFDs: volume in contracts (1 contract ≈ 1 unit of index value).
        # Size by risk_usd if provided, else fall back to 1 contract minimum.
        if risk_usd and risk_usd > 0:
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
        )
    else:
        # Forex: standard lot sizing
        pip       = _PIP_SIZES.get(symbol, 0.0001)
        # Compute effective SL in pips for lot sizing
        _sl_pips_eff = sl_pips if sl_pips and sl_pips > 0 else abs(entry_price - sl_price) / max(pip, 1e-10)
        # pip_value_per_lot: USD P&L per pip per standard lot
        # For USD-quoted pairs (EURUSD, GBPUSD…): 10 USD/pip/lot
        # For JPY pairs: ~9.28 USD (yen-based, approx at USDJPY≈148)
        # For XAU: $1/pip/lot (XAUUSD pip=0.01, lot=100oz → 100×0.01=$1)
        if symbol.upper() in ("XAUUSD",):
            pip_value = 1.0    # $1/pip/lot (100oz lot, pip=0.01)
        elif symbol.upper() in ("XAGUSD",):
            pip_value = 5.0    # $5/pip/lot approx (5000oz lot, pip=0.001)
        elif "JPY" in symbol.upper():
            pip_value = 9.28   # approx USD/pip/lot at USDJPY~148
        else:
            pip_value = 10.0   # standard USD/pip/lot for majors

        if use_risk_pct and risk_pct and risk_pct > 0:
            # ── Risk % auto lot sizing ─────────────────────────────────────
            # Fetch live account balance; compute lots from risk fraction.
            account_balance = await _get_account_balance(access_token, ctid_trader_account_id)
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
        )

    if result.get("error"):
        logger.error(f"[cTrader] order failed for user {user.id}: {result['error']}")
        return None
    return result
