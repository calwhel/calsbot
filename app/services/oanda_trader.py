"""
OANDA forex order placement wrapper for the strategy executor (P5e-2).

Mirrors `app/services/strategy_trader.py::place_bitunix_order_for_user` but
routes forex live orders through OANDA's v20 REST API. Credentials live on
`UserPreference.oanda_api_key` (encrypted) + `oanda_account_id` +
`oanda_environment`. The OANDA order is placed with `takeProfitOnFill` and
`stopLossOnFill` so the broker enforces exits server-side — our candle-based
monitor then closes the StrategyExecution row when the next candle crosses
TP/SL (same path forex paper trades already use).

Position sizing: OANDA speaks `units`, not USD notional. We convert with
`units = risk_usd / pip_size / pip_value_per_unit`. For majors and minors,
1 unit ≈ $1 of notional at price ~1.0, so we approximate units as
`risk_usd / entry_price`. Conservative side: round DOWN to the nearest 10
(OANDA minimum trade size is 1 unit; rounding to 10 keeps the lot tidy).

Returns the same shape as the bitunix wrapper: `{order_id, actual_fill}` or
None on hard failures so the executor can fall back to paper for ROI
tracking — never raises (except for `PRICE_PAST_TP`-style cancels, which
forex doesn't currently produce because OANDA fills market orders at the
inside quote within milliseconds).
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


async def place_oanda_order_for_user(
    user,
    symbol: str,
    direction: str,
    entry_price: float,
    tp_pct: float,
    sl_pct: float,
    risk_pct: float = 5.0,
    risk_usd: Optional[float] = None,
) -> Optional[Dict]:
    """Place an OANDA market order for a forex strategy execution.

    Returns `{"order_id": str, "actual_fill": float}` on success, None when
    keys are missing or the order is rejected. Never raises.
    """
    from app.models import UserPreference
    from app.database import SessionLocal
    from app.utils.encryption import decrypt_api_key
    from app.services.oanda_client import place_market_order, get_account_summary

    # ── Load + decrypt credentials ────────────────────────────────────────────
    db = SessionLocal()
    try:
        prefs = db.query(UserPreference).filter_by(user_id=user.id).first()
    finally:
        db.close()

    if not prefs or not prefs.oanda_api_key or not prefs.oanda_account_id:
        logger.warning(f"[oanda_trader] user {user.id} missing OANDA credentials — skipping live order")
        return None

    try:
        api_key = decrypt_api_key(prefs.oanda_api_key)
    except Exception as e:
        logger.error(f"[oanda_trader] decrypt failed for user {user.id}: {e}")
        return None

    account_id  = prefs.oanda_account_id
    environment = (getattr(prefs, "oanda_environment", None) or "practice").lower()

    # ── Compute TP/SL prices ──────────────────────────────────────────────────
    ep = float(entry_price)
    side_long = (direction or "").upper() == "LONG"
    if side_long:
        tp_price = ep * (1 + tp_pct / 100.0)
        sl_price = ep * (1 - sl_pct / 100.0)
    else:
        tp_price = ep * (1 - tp_pct / 100.0)
        sl_price = ep * (1 + sl_pct / 100.0)

    # ── Size the position ─────────────────────────────────────────────────────
    # Prefer fixed USD amount when provided; otherwise compute from the user's
    # OANDA balance × risk_pct. Falls back to a small default if the balance
    # lookup fails so the trade still fires (paper-equivalent sizing).
    notional_usd = None
    if risk_usd is not None and risk_usd > 0:
        notional_usd = float(risk_usd)
    else:
        ok, summary = await get_account_summary(api_key, account_id, environment)
        if ok and summary.get("balance"):
            notional_usd = float(summary["balance"]) * (float(risk_pct) / 100.0)
        else:
            logger.warning(
                f"[oanda_trader] user {user.id} balance fetch failed "
                f"({summary.get('message') if not ok else 'no balance'}) — defaulting to $100 notional"
            )
            notional_usd = 100.0

    # Convert USD notional → OANDA units (units = quantity of BASE currency).
    # OANDA symbols are formatted "BASE_QUOTE" (e.g. EUR_USD, USD_JPY).
    #  • USD-base pairs (USD_JPY, USD_CAD, USD_CHF): 1 unit = $1 USD,
    #    so units = notional_usd directly.
    #  • USD-quote pairs (EUR_USD, GBP_USD, AUD_USD, NZD_USD): 1 unit = 1
    #    unit of base currency worth `entry_price` USD, so units =
    #    notional_usd / entry_price.
    #  • Cross pairs (EUR_GBP, etc.): fall back to USD/entry_price as a
    #    reasonable approximation — close enough for v1 sizing.
    _sym_norm = (symbol or "").upper().replace("/", "_")
    _base = _sym_norm.split("_", 1)[0] if "_" in _sym_norm else _sym_norm[:3]
    if _base == "USD":
        raw_units = notional_usd
    else:
        raw_units = notional_usd / max(ep, 1e-9)
    units     = int(raw_units // 10 * 10)   # round DOWN to lot of 10
    if units < 10:
        logger.warning(f"[oanda_trader] computed units={units} too small for {symbol} — bumping to 10")
        units = 10
    if not side_long:
        units = -units

    # ── Place the order ───────────────────────────────────────────────────────
    ok, resp = await place_market_order(
        api_key=api_key, account_id=account_id, symbol=symbol,
        units=units, tp_price=tp_price, sl_price=sl_price,
        environment=environment,
    )
    if not ok:
        logger.error(f"[oanda_trader] order rejected user={user.id} {symbol}: {resp.get('message')}")
        return None

    # Response shape (success): orderFillTransaction.id + price.
    fill = resp.get("orderFillTransaction") or {}
    order_id   = str(fill.get("id") or resp.get("lastTransactionID") or "")
    actual_fill = None
    try:
        actual_fill = float(fill.get("price")) if fill.get("price") else None
    except Exception:
        actual_fill = None

    if not order_id:
        logger.warning(f"[oanda_trader] order placed but no id returned: {resp}")
        return None

    logger.info(
        f"[oanda_trader] ✅ user={user.id} {symbol} {direction} units={units} "
        f"order_id={order_id} fill={actual_fill or ep}"
    )
    return {"order_id": order_id, "actual_fill": actual_fill or ep}
