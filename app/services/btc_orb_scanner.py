"""
BTC Momentum Structure Break Scalper

Strategy:
1. Session filter: London (08:00-12:00 UTC) + NY (13:30-18:00 UTC)
2. 15m candle closes above previous 3-candle high (LONG) or below 3-candle low (SHORT)
3. Breakout candle body must be >60% of its total range (strong conviction, not a wick)
4. Breakout candle volume > 1.5x average of previous 10 candles
5. 1m RSI in momentum zone: 45-65 for LONG, 35-55 for SHORT
6. Funding rate guard: skip LONG if funding >+0.04%, skip SHORT if <-0.04%
7. Micro-retest: wait up to 15min for price to pull back to broken level on 1m
8. Enter at retest: 0.25% TP / 0.25% SL (1:1 R:R) = ~50% ROI at 200x
   Breakeven SL moves to entry at halfway handled by position monitor
"""

import logging
import asyncio
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

BINANCE_FUTURES_URL = "https://fapi.binance.com"
BTC_SYMBOL = "BTCUSDT"

STRUCTURE_LOOKBACK = 3
BODY_RATIO_MIN = 0.60
VOLUME_RATIO_MIN = 1.5
RETEST_TOLERANCE_PCT = 0.05
RETEST_TIMEOUT_MINUTES = 15
TP_SL_PCT = 0.25

LONDON_START = (8, 0)
LONDON_END = (12, 0)
NY_START = (13, 30)
NY_END = (18, 0)

BTC_ORB_LEVERAGE = 200
BTC_ORB_COOLDOWN_MINUTES = 60
MAX_BTC_ORB_DAILY_SIGNALS = 3
BTC_ORB_SESSIONS_ENABLED = {"ASIA": False, "LONDON": True, "NY": True}

_btc_orb_enabled = True
_btc_orb_last_signal_time = None
_btc_orb_daily_count = 0
_btc_orb_daily_reset = None

_pending_setup: Optional[Dict] = None


def is_btc_orb_enabled() -> bool:
    return _btc_orb_enabled


def set_btc_orb_enabled(enabled: bool):
    global _btc_orb_enabled
    _btc_orb_enabled = enabled
    logger.info(f"⚡ BTC Scalper {'ENABLED' if enabled else 'DISABLED'}")


def get_btc_orb_leverage() -> int:
    return BTC_ORB_LEVERAGE


def set_btc_orb_leverage(leverage: int):
    global BTC_ORB_LEVERAGE
    BTC_ORB_LEVERAGE = max(5, min(200, leverage))
    logger.info(f"⚡ BTC Scalper leverage set to {BTC_ORB_LEVERAGE}x")


def get_btc_orb_max_daily() -> int:
    return MAX_BTC_ORB_DAILY_SIGNALS


def set_btc_orb_max_daily(limit: int):
    global MAX_BTC_ORB_DAILY_SIGNALS
    MAX_BTC_ORB_DAILY_SIGNALS = max(1, min(20, limit))
    logger.info(f"⚡ BTC Scalper max daily signals set to {MAX_BTC_ORB_DAILY_SIGNALS}")


def get_btc_orb_sessions() -> dict:
    return BTC_ORB_SESSIONS_ENABLED.copy()


def toggle_btc_orb_session(session: str) -> bool:
    global BTC_ORB_SESSIONS_ENABLED
    if session in BTC_ORB_SESSIONS_ENABLED:
        BTC_ORB_SESSIONS_ENABLED[session] = not BTC_ORB_SESSIONS_ENABLED[session]
        logger.info(f"⚡ BTC Scalper {session} session {'ENABLED' if BTC_ORB_SESSIONS_ENABLED[session] else 'DISABLED'}")
        return BTC_ORB_SESSIONS_ENABLED[session]
    return False


def get_btc_orb_cooldown() -> int:
    return BTC_ORB_COOLDOWN_MINUTES


def set_btc_orb_cooldown(minutes: int):
    global BTC_ORB_COOLDOWN_MINUTES
    BTC_ORB_COOLDOWN_MINUTES = max(15, min(480, minutes))
    logger.info(f"⚡ BTC Scalper cooldown set to {BTC_ORB_COOLDOWN_MINUTES}min")


def get_btc_orb_daily_count() -> int:
    return _btc_orb_daily_count


def check_btc_orb_cooldown() -> bool:
    global _btc_orb_last_signal_time
    if _btc_orb_last_signal_time is None:
        return True
    elapsed = (datetime.utcnow() - _btc_orb_last_signal_time).total_seconds()
    return elapsed >= (BTC_ORB_COOLDOWN_MINUTES * 60)


def check_btc_orb_daily_limit() -> bool:
    global _btc_orb_daily_count, _btc_orb_daily_reset
    now = datetime.utcnow()
    if _btc_orb_daily_reset is None or now.date() > _btc_orb_daily_reset.date():
        _btc_orb_daily_count = 0
        _btc_orb_daily_reset = now
    return _btc_orb_daily_count < MAX_BTC_ORB_DAILY_SIGNALS


def record_btc_orb_signal():
    global _btc_orb_last_signal_time, _btc_orb_daily_count
    _btc_orb_last_signal_time = datetime.utcnow()
    _btc_orb_daily_count += 1


def _get_active_session() -> Optional[str]:
    now = datetime.utcnow()
    h, m = now.hour, now.minute
    t = h * 60 + m

    london_s = LONDON_START[0] * 60 + LONDON_START[1]
    london_e = LONDON_END[0] * 60 + LONDON_END[1]
    ny_s = NY_START[0] * 60 + NY_START[1]
    ny_e = NY_END[0] * 60 + NY_END[1]

    if london_s <= t < london_e and BTC_ORB_SESSIONS_ENABLED.get("LONDON"):
        return "LONDON"
    if ny_s <= t < ny_e and BTC_ORB_SESSIONS_ENABLED.get("NY"):
        return "NY"
    return None


class BTCOrbScanner:
    def __init__(self):
        self.client = None

    async def init(self):
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=15,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None

    async def _fetch_klines(self, interval: str, limit: int) -> List[Dict]:
        try:
            url = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
            resp = await self.client.get(url, params={
                "symbol": BTC_SYMBOL,
                "interval": interval,
                "limit": limit
            }, timeout=8)
            if resp.status_code != 200:
                return []
            raw = resp.json()
            candles = []
            for k in raw:
                candles.append({
                    "ts": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            return candles
        except Exception as e:
            logger.warning(f"⚡ Kline fetch error ({interval}): {e}")
            return []

    async def _get_funding_rate(self) -> Optional[float]:
        try:
            url = f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex"
            resp = await self.client.get(url, params={"symbol": BTC_SYMBOL}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return float(data.get("lastFundingRate", 0))
        except Exception:
            pass
        return None

    def _calc_rsi(self, closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        gains, losses = [], []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i - 1]
            gains.append(max(diff, 0))
            losses.append(max(-diff, 0))
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 1)

    def _check_structure_break(self, candles: List[Dict]) -> Optional[Dict]:
        if len(candles) < STRUCTURE_LOOKBACK + 2:
            return None

        closed = candles[:-1]
        last = closed[-1]
        lookback = closed[-(STRUCTURE_LOOKBACK + 1):-1]

        candle_range = last["high"] - last["low"]
        if candle_range <= 0:
            return None

        body = abs(last["close"] - last["open"])
        body_ratio = body / candle_range
        if body_ratio < BODY_RATIO_MIN:
            logger.debug(f"⚡ Body ratio too weak: {body_ratio:.2f} (need {BODY_RATIO_MIN})")
            return None

        prev_vols = [c["volume"] for c in closed[-(11):-1]]
        if len(prev_vols) < 5:
            return None
        avg_vol = sum(prev_vols) / len(prev_vols)
        vol_ratio = last["volume"] / avg_vol if avg_vol > 0 else 1.0
        if vol_ratio < VOLUME_RATIO_MIN:
            logger.debug(f"⚡ Volume too weak: {vol_ratio:.2f}x (need {VOLUME_RATIO_MIN}x)")
            return None

        prev_high = max(c["high"] for c in lookback)
        prev_low = min(c["low"] for c in lookback)

        if last["close"] > prev_high and last["close"] > last["open"]:
            return {
                "direction": "LONG",
                "break_level": prev_high,
                "break_candle_close": last["close"],
                "body_ratio": body_ratio,
                "vol_ratio": vol_ratio,
                "ts": last["ts"],
            }
        elif last["close"] < prev_low and last["close"] < last["open"]:
            return {
                "direction": "SHORT",
                "break_level": prev_low,
                "break_candle_close": last["close"],
                "body_ratio": body_ratio,
                "vol_ratio": vol_ratio,
                "ts": last["ts"],
            }

        return None

    async def _check_retest(self, setup: Dict, current_price: float) -> bool:
        level = setup["break_level"]
        direction = setup["direction"]
        tolerance = level * (RETEST_TOLERANCE_PCT / 100)

        if direction == "LONG":
            return abs(current_price - level) <= tolerance and current_price >= level - tolerance
        else:
            return abs(current_price - level) <= tolerance and current_price <= level + tolerance

    async def scan(self) -> Optional[Dict]:
        global _pending_setup

        await self.init()

        session = _get_active_session()
        if not session:
            _pending_setup = None
            return None

        try:
            if _pending_setup:
                age_minutes = (datetime.utcnow() - _pending_setup["detected_at"]).total_seconds() / 60
                if age_minutes > RETEST_TIMEOUT_MINUTES:
                    logger.info(f"⚡ Retest window expired for {_pending_setup['direction']} setup — clearing")
                    _pending_setup = None
                    return None

                candles_1m = await self._fetch_klines("1m", 3)
                if not candles_1m:
                    return None
                current_price = candles_1m[-1]["close"]

                retest_confirmed = await self._check_retest(_pending_setup, current_price)
                if retest_confirmed:
                    setup = _pending_setup
                    _pending_setup = None

                    direction = setup["direction"]
                    entry = current_price
                    tp_pct = TP_SL_PCT
                    sl_pct = TP_SL_PCT

                    if direction == "LONG":
                        tp = entry * (1 + tp_pct / 100)
                        sl = entry * (1 - sl_pct / 100)
                    else:
                        tp = entry * (1 - tp_pct / 100)
                        sl = entry * (1 + sl_pct / 100)

                    roi = tp_pct * BTC_ORB_LEVERAGE

                    logger.info(
                        f"⚡ BTC SCALP SIGNAL: {direction} | Entry ${entry:.2f} | "
                        f"TP ${tp:.2f} (+{roi:.0f}% ROI) | SL ${sl:.2f} | "
                        f"Retest of ${setup['break_level']:.2f} | Session: {session}"
                    )

                    return {
                        "direction": direction,
                        "entry": entry,
                        "stop_loss": sl,
                        "take_profit_1": tp,
                        "take_profit_2": None,
                        "sl_pct": sl_pct,
                        "tp1_pct": tp_pct,
                        "roi_pct": roi,
                        "rr_ratio": 1.0,
                        "session": session,
                        "break_level": setup["break_level"],
                        "body_ratio": setup["body_ratio"],
                        "vol_ratio": setup["vol_ratio"],
                        "entry_quality": "STRONG" if setup["vol_ratio"] >= 2.0 else "GOOD",
                    }

                logger.debug(f"⚡ Waiting for retest of ${_pending_setup['break_level']:.2f} | "
                             f"Current: ${current_price:.2f} | Age: {age_minutes:.0f}min")
                return None

            candles_15m = await self._fetch_klines("15m", STRUCTURE_LOOKBACK + 12)
            if not candles_15m:
                return None

            structure_break = self._check_structure_break(candles_15m)
            if not structure_break:
                return None

            candles_1m = await self._fetch_klines("1m", 20)
            if not candles_1m:
                return None

            closes_1m = [c["close"] for c in candles_1m]
            rsi_1m = self._calc_rsi(closes_1m)
            current_price = closes_1m[-1]

            direction = structure_break["direction"]

            if direction == "LONG" and not (45 <= rsi_1m <= 65):
                logger.info(f"⚡ RSI {rsi_1m:.0f} out of LONG zone (45-65) — skipping")
                return None
            if direction == "SHORT" and not (35 <= rsi_1m <= 55):
                logger.info(f"⚡ RSI {rsi_1m:.0f} out of SHORT zone (35-55) — skipping")
                return None

            funding = await self._get_funding_rate()
            if funding is not None:
                if direction == "LONG" and funding > 0.0004:
                    logger.info(f"⚡ Funding {funding:.4f} too high for LONG — longs crowded, skipping")
                    return None
                if direction == "SHORT" and funding < -0.0004:
                    logger.info(f"⚡ Funding {funding:.4f} too negative for SHORT — shorts crowded, skipping")
                    return None

            _pending_setup = {
                **structure_break,
                "session": session,
                "detected_at": datetime.utcnow(),
                "rsi_1m": rsi_1m,
                "funding": funding,
            }

            logger.info(
                f"⚡ STRUCTURE BREAK detected: {direction} | "
                f"Break level: ${structure_break['break_level']:.2f} | "
                f"Body: {structure_break['body_ratio']:.0%} | "
                f"Vol: {structure_break['vol_ratio']:.1f}x | "
                f"RSI 1m: {rsi_1m:.0f} | "
                f"Waiting for micro-retest..."
            )
            return None

        except Exception as e:
            logger.error(f"⚡ BTC Scalper scan error: {e}", exc_info=True)
            return None


def format_btc_orb_message(signal_data: Dict) -> str:
    direction = signal_data["direction"]
    session = signal_data["session"]
    entry = signal_data["entry"]
    tp = signal_data["take_profit_1"]
    sl = signal_data["stop_loss"]
    tp_pct = signal_data["tp1_pct"]
    sl_pct = signal_data["sl_pct"]
    roi = signal_data["roi_pct"]
    leverage = BTC_ORB_LEVERAGE
    vol_ratio = signal_data.get("vol_ratio", 0)
    body_ratio = signal_data.get("body_ratio", 0)
    break_level = signal_data.get("break_level", 0)
    quality = signal_data.get("entry_quality", "GOOD")

    direction_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
    session_emoji = {"LONDON": "🇬🇧", "NY": "🗽", "ASIA": "🌏"}.get(session, "📊")
    quality_stars = {"STRONG": "⭐⭐", "GOOD": "⭐"}.get(quality, "⭐")

    msg = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━┓
  ⚡ <b>BTC MOMENTUM SCALP</b> ⚡
┗━━━━━━━━━━━━━━━━━━━━━━━┛

{direction_emoji} <b>$BTC</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

{session_emoji} <b>{session} Session</b>
├ Structure break: ${break_level:.2f}
├ Candle body: {body_ratio:.0%} of range
├ Volume surge: {vol_ratio:.1f}x avg
└ Quality: {quality} {quality_stars}

<b>🎯 Trade Setup</b>
├ Entry: <b>${entry:.2f}</b> (micro-retest)
├ TP: <b>${tp:.2f}</b> (+{tp_pct:.2f}% / <b>+{roi:.0f}% ROI</b>) 🎯
└ SL: <b>${sl:.2f}</b> (-{sl_pct:.2f}% / -{roi:.0f}% ROI)

<b>⚡ Risk Management</b>
├ Leverage: <b>{leverage}x</b>
├ R/R: <b>1:1</b>
└ 🔒 SL → entry at halfway to TP

⚠️ <b>{leverage}x LEVERAGE — HIGH RISK</b>
<i>Auto-executing for enabled users...</i>
"""
    return msg.strip()


def format_btc_orb_status() -> str:
    global _pending_setup

    enabled_text = "🟢 ENABLED" if _btc_orb_enabled else "🔴 DISABLED"
    session = _get_active_session()
    session_text = f"🟢 {session} session active" if session else "🔴 Outside trading hours"

    setup_text = "No active setup"
    if _pending_setup:
        age = (datetime.utcnow() - _pending_setup["detected_at"]).total_seconds() / 60
        remaining = max(0, RETEST_TIMEOUT_MINUTES - age)
        setup_text = (
            f"🔍 Waiting for {_pending_setup['direction']} retest\n"
            f"├ Break level: ${_pending_setup['break_level']:.2f}\n"
            f"├ Vol: {_pending_setup['vol_ratio']:.1f}x | Body: {_pending_setup['body_ratio']:.0%}\n"
            f"└ Retest window: {remaining:.0f}min remaining"
        )

    london_status = "🟢" if BTC_ORB_SESSIONS_ENABLED.get("LONDON") else "🔴"
    ny_status = "🟢" if BTC_ORB_SESSIONS_ENABLED.get("NY") else "🔴"

    msg = f"""
⚡ <b>BTC MOMENTUM SCALPER STATUS</b>

<b>Scanner:</b> {enabled_text}
<b>Leverage:</b> {BTC_ORB_LEVERAGE}x
<b>TP/SL:</b> {TP_SL_PCT}% ({TP_SL_PCT * BTC_ORB_LEVERAGE:.0f}% ROI at {BTC_ORB_LEVERAGE}x)
<b>Sessions:</b> {london_status} London (08:00-12:00) | {ny_status} NY (13:30-18:00)
<b>Current:</b> {session_text}

<b>📈 Active Setup:</b>
{setup_text}

<b>📊 Signals Today:</b> {_btc_orb_daily_count}/{MAX_BTC_ORB_DAILY_SIGNALS}
<b>⏳ Cooldown:</b> {"Active" if not check_btc_orb_cooldown() else "Ready"}
<b>Gap between signals:</b> {BTC_ORB_COOLDOWN_MINUTES}min
"""
    return msg.strip()


async def broadcast_btc_orb_signal(db_session, bot):
    if not is_btc_orb_enabled():
        return

    if not check_btc_orb_cooldown():
        logger.debug("⚡ BTC Scalper on cooldown")
        return

    if not check_btc_orb_daily_limit():
        logger.info("⚡ BTC Scalper daily signal limit reached")
        return

    session = _get_active_session()
    if not session:
        return

    try:
        from app.models import User, UserPreference, Signal, Trade
        from app.services.bitunix_trader import execute_bitunix_trade
        from app.services.social_signals import check_global_signal_limit, increment_global_signal_count
        from app.database import SessionLocal
        from sqlalchemy import or_

        from app.services.social_signals import check_signal_gap, record_signal_broadcast
        if not check_global_signal_limit():
            logger.info("⚡ Global daily signal limit reached — skipping BTC Scalp scan")
            return
        if not check_signal_gap():
            logger.info("⚡ Signal gap not met — too soon since last signal")
            return

        users = db_session.query(User).join(UserPreference).filter(
            UserPreference.scalp_mode_enabled == True,
            or_(
                User.is_admin == True,
                User.grandfathered == True,
                (User.subscription_end != None) & (User.subscription_end > datetime.utcnow())
            )
        ).all()

        if not users:
            logger.debug("⚡ No authorized users for BTC Scalp signals")
            return

        scanner = BTCOrbScanner()
        await scanner.init()

        try:
            signal_data = await scanner.scan()

            if not signal_data:
                return

            logger.info(f"⚡ ═══════════════════════════════════════════")
            logger.info(f"⚡ BTC SCALP SIGNAL: {signal_data['direction']} | Session: {signal_data['session']}")
            logger.info(f"⚡ Entry: ${signal_data['entry']:.2f} | ROI target: {signal_data['roi_pct']:.0f}%")
            logger.info(f"⚡ ═══════════════════════════════════════════")

            message = format_btc_orb_message(signal_data)

            new_signal = Signal(
                symbol="BTCUSDT",
                direction=signal_data["direction"],
                entry_price=signal_data["entry"],
                stop_loss=signal_data["stop_loss"],
                take_profit=signal_data["take_profit_1"],
                take_profit_1=signal_data["take_profit_1"],
                take_profit_2=None,
                confidence=80,
                signal_type="BTC_ORB_SCALP",
                timeframe="15m",
                reasoning=(
                    f"BTC Momentum Scalp — {signal_data['session']} session | "
                    f"Structure break at ${signal_data['break_level']:.2f} | "
                    f"Vol {signal_data['vol_ratio']:.1f}x | Body {signal_data['body_ratio']:.0%} | "
                    f"Micro-retest entry | {signal_data['roi_pct']:.0f}% ROI target @ {BTC_ORB_LEVERAGE}x"
                )
            )
            db_session.add(new_signal)
            db_session.commit()
            db_session.refresh(new_signal)

            record_btc_orb_signal()
            increment_global_signal_count()
            record_signal_broadcast()

            for user in users:
                try:
                    prefs = user.preferences
                    lev_line = f"\n\n⚡ {BTC_ORB_LEVERAGE}x"
                    user_message = message + lev_line

                    await bot.send_message(
                        user.telegram_id,
                        user_message,
                        parse_mode="HTML"
                    )
                    logger.info(f"⚡ Sent BTC Scalp signal to user {user.telegram_id} (ID {user.id})")

                    has_keys = prefs and prefs.bitunix_api_key and prefs.bitunix_api_secret
                    if has_keys and prefs.auto_trading_enabled:
                        trade_db = SessionLocal()
                        try:
                            from app.models import User as UserModel, Signal as SignalModel
                            trade_user = trade_db.query(UserModel).filter(UserModel.id == user.id).first()
                            trade_signal = trade_db.query(SignalModel).filter(SignalModel.id == new_signal.id).first()
                            if trade_user and trade_signal:
                                logger.info(
                                    f"🔄 EXECUTING BTC SCALP: {signal_data['direction']} "
                                    f"for user {user.telegram_id} (ID {user.id}) @ {BTC_ORB_LEVERAGE}x"
                                )
                                trade_result = await execute_bitunix_trade(
                                    signal=trade_signal,
                                    user=trade_user,
                                    db=trade_db,
                                    trade_type="BTC_ORB_SCALP",
                                    leverage_override=BTC_ORB_LEVERAGE
                                )
                                if trade_result:
                                    logger.info(f"✅ BTC Scalp executed for user {user.telegram_id}")
                                    await bot.send_message(
                                        user.telegram_id,
                                        f"✅ <b>Trade Executed on Bitunix</b>\n"
                                        f"<b>$BTC</b> {signal_data['direction']} @ {BTC_ORB_LEVERAGE}x",
                                        parse_mode="HTML"
                                    )
                                else:
                                    logger.warning(f"⚠️ BTC Scalp trade blocked for user {user.telegram_id}")
                        finally:
                            trade_db.close()
                    elif has_keys and not prefs.auto_trading_enabled:
                        logger.info(f"⚡ User {user.telegram_id} — auto-trading disabled, signal only")
                    else:
                        logger.info(f"⚡ User {user.telegram_id} — no API keys, signal only")
                except Exception as e:
                    logger.error(f"Failed to send/execute BTC Scalp signal for {user.telegram_id}: {e}")

        finally:
            await scanner.close()

    except Exception as e:
        logger.error(f"Error in BTC Scalp broadcast: {e}", exc_info=True)
