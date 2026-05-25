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
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001,
    "NZDUSD": 0.0001, "USDCAD": 0.0001, "USDCHF": 0.0001,
    "USDJPY": 0.01,   "EURJPY": 0.01,   "GBPJPY": 0.01,
}

# FP Markets symbol name mapping (cTrader uses exact broker symbol names)
_SYMBOL_MAP = {
    "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD", "USDCAD": "USDCAD", "USDCHF": "USDCHF",
    "NZDUSD": "NZDUSD",
}


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
    }
    if state:
        params["state"] = state
    base = OAUTH_AUTH_URL.format(client_id=CTRADER_CLIENT_ID)
    return f"{base}?{urllib.parse.urlencode(params)}"


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
    """
    if not _PROTO_OK:
        return []
    reader, writer = await _open_connection()
    try:
        if not await _app_auth(reader, writer):
            return []
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
        accounts = []
        for acc in res.ctidTraderAccount:
            accounts.append({
                "ctidTraderAccountId": acc.ctidTraderAccountId,
                "isLive":              acc.isLive,
                "traderLogin":         getattr(acc, "traderLogin", 0),
            })
        return accounts
    finally:
        writer.close()


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
    """
    if not _PROTO_OK:
        return {"order_id": None, "actual_fill": None, "error": "ctrader proto not available"}
    if not CTRADER_CLIENT_ID or not CTRADER_CLIENT_SECRET:
        return {"order_id": None, "actual_fill": None, "error": "CTRADER_CLIENT_ID/SECRET not set"}

    reader, writer = await _open_connection()
    try:
        if not await _app_auth(reader, writer):
            return {"order_id": None, "actual_fill": None, "error": "app auth failed"}
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
            req.relativeStopLoss = 0  # we set absolute via stopLoss field
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

        ev = ProtoOAExecutionEvent()
        ev.ParseFromString(payload)
        order_id   = str(ev.order.orderId) if ev.HasField("order") else None
        actual_fill = None
        if ev.HasField("deal") and ev.deal.executionPrice:
            actual_fill = ev.deal.executionPrice / 100_000.0
        return {"order_id": order_id, "actual_fill": actual_fill, "error": None}

    except Exception as e:
        logger.error(f"[cTrader] place_order failed: {e}")
        return {"order_id": None, "actual_fill": None, "error": str(e)}
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def close_position(
    access_token: str,
    ctid_trader_account_id: int,
    position_id: int,
    volume_units: int,
) -> bool:
    """Close (or partially close) an open position."""
    if not _PROTO_OK:
        return False
    reader, writer = await _open_connection()
    try:
        if not await _app_auth(reader, writer):
            return False
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
        return payload is not None
    except Exception as e:
        logger.error(f"[cTrader] close_position failed: {e}")
        return False
    finally:
        try:
            writer.close()
        except Exception:
            pass


# ── High-level: place order for a TradeHub user ───────────────────────────────

async def place_ctrader_order_for_user(
    user,
    symbol: str,
    direction: str,
    entry_price: float,
    tp_pct: float,
    sl_pct: float,
    risk_pct: float = 1.0,
    risk_usd: Optional[float] = None,
) -> Optional[dict]:
    """
    Place a live forex order for a user who has connected their cTrader account.
    Converts TP/SL from % to absolute prices, calculates lot size from risk.
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

    mult        = 1.0 if direction == "LONG" else -1.0
    tp_price    = round(entry_price * (1 + mult * tp_pct / 100), 6)
    sl_price    = round(entry_price * (1 - mult * sl_pct / 100), 6)

    # Lot size: risk_usd / (sl_pct% of entry * pip_value)
    pip         = _PIP_SIZES.get(symbol, 0.0001)
    sl_pips     = abs(entry_price - sl_price) / pip
    pip_value   = 10.0 if "JPY" not in symbol else 9.28  # approx USD/pip/lot
    if risk_usd and risk_usd > 0:
        lots = round(risk_usd / max(sl_pips * pip_value, 0.01), 2)
    else:
        lots = 0.01  # minimal lot — caller should send risk_usd for real sizing
    lots = max(0.01, min(lots, 50.0))  # clamp 0.01–50 lots

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
