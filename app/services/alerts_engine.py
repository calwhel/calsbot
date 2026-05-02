"""
Indicator alerts engine
=======================
Evaluates user-created chart alerts against MEXC kline data and sends a
Telegram DM when a condition is hit.

Runs as an asyncio task inside the Strategy Portal worker that holds the
executor advisory lock — so notifications never get duplicated when multiple
gunicorn workers are alive.

Supported alert kinds (`kind` column):
    price            — last close vs `target`. condition: above|below|crossover|crossunder
    rsi              — RSI(period) vs `target`. params: {"period":14}
    ema_cross        — last close vs EMA(period). params: {"period":50}.
                       condition: crossover (price → above EMA) | crossunder
    macd_cross_zero  — MACD line vs 0. params: {"fast":12,"slow":26,"signal":9}.
                       condition: crossover (turns positive) | crossunder
    supertrend_flip  — SuperTrend(period, mult) flips. params: {"period":10,"mult":3}.
                       condition: 'flip' (any direction change)

Cross semantics
---------------
A `crossover` fires only when the previous evaluation was on the OTHER side and
the current evaluation is on the requested side (true edge detection). On the
very first evaluation we just record `last_value` and don't fire — this prevents
"already true" alerts from spam-firing the moment they're created.

Fire modes (Task #10)
---------------------
* `once`                       — legacy default. After the first fire we flip
                                 status='triggered' and the alert auto-disarms.
* `every_cross`                — fire on every fresh edge; status stays 'active'.
* `every_cross_with_cooldown`  — same as every_cross but rate-limited:
                                 - cooldown_minutes between consecutive fires
                                 - daily_cap (UTC day) on total fires
                                 When suppressed by either limit we still update
                                 last_value so the *next* fresh cross can fire
                                 once the limit clears (no missed-edge holes).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# ─── MEXC kline fetch ─────────────────────────────────────────────────────────
_MEXC_KLINE_URL = "https://api.mexc.com/api/v3/klines"
# Same map as the trade page route
_TF_MAP = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "4h": "4h"}
_SYMBOL_PAIR = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

# Per-(symbol, tf) candle cache so multiple alerts with the same symbol/tf
# share one HTTP call per evaluation cycle.
_CANDLE_CACHE: Dict[str, Tuple[List[dict], float]] = {}
_CANDLE_TTL = 8.0  # seconds — alerts loop runs every 30s, this is well under that


async def _fetch_candles(symbol: str, tf: str, limit: int = 200) -> List[dict]:
    pair = _SYMBOL_PAIR.get(symbol.upper())
    interval = _TF_MAP.get(tf)
    if not pair or not interval:
        return []
    key = f"{pair}_{interval}_{limit}"
    cached = _CANDLE_CACHE.get(key)
    if cached and time.time() < cached[1]:
        return cached[0]
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(_MEXC_KLINE_URL, params={
                "symbol": pair, "interval": interval, "limit": limit,
            })
            r.raise_for_status()
            rows = r.json() or []
        candles = []
        for k in rows:
            try:
                candles.append({
                    "time":   int(k[0]) // 1000,
                    "open":   float(k[1]),
                    "high":   float(k[2]),
                    "low":    float(k[3]),
                    "close":  float(k[4]),
                    "volume": float(k[5]) if len(k) > 5 else 0.0,
                })
            except (TypeError, ValueError, IndexError):
                continue
        _CANDLE_CACHE[key] = (candles, time.time() + _CANDLE_TTL)
        return candles
    except Exception as e:
        logger.warning(
            f"alerts_engine fetch {key} failed: "
            f"{type(e).__name__}: {e!r}"
        )
        return []


# ─── TA primitives (pure-Python, mirror of trade.html JS engine) ──────────────
def _ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    out = seed
    for v in values[period:]:
        out = v * k + out * (1 - k)
    return out


def _rsi(values: List[float], period: int = 14) -> Optional[float]:
    if len(values) < period + 1:
        return None
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - 100.0 / (1.0 + rs)


def _ema_series(values: List[float], period: int) -> List[Optional[float]]:
    if len(values) < period:
        return [None] * len(values)
    out: List[Optional[float]] = [None] * len(values)
    k = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    for i in range(period, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1 - k)  # type: ignore
    return out


def _macd_line(values: List[float], fast: int, slow: int) -> Optional[float]:
    """Returns just the MACD line value (last bar). Signal line not needed for zero-cross."""
    ef = _ema(values, fast)
    es = _ema(values, slow)
    if ef is None or es is None:
        return None
    return ef - es


def _supertrend_dir(candles: List[dict], period: int, mult: float) -> Optional[int]:
    """Returns the SuperTrend direction (+1 long / -1 short) for the latest bar."""
    n = len(candles)
    if n < period + 2:
        return None
    # True Range and Wilder ATR
    tr: List[float] = []
    for i, c in enumerate(candles):
        if i == 0:
            tr.append(c["high"] - c["low"])
        else:
            p = candles[i - 1]
            tr.append(max(
                c["high"] - c["low"],
                abs(c["high"] - p["close"]),
                abs(c["low"] - p["close"]),
            ))
    atr: List[Optional[float]] = [None] * n
    seed = sum(tr[:period]) / period
    atr[period - 1] = seed
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period  # type: ignore

    upper: List[Optional[float]] = [None] * n
    lower: List[Optional[float]] = [None] * n
    direction = 1
    for i in range(n):
        c = candles[i]
        if atr[i] is None:
            continue
        hl2 = (c["high"] + c["low"]) / 2.0
        b_up = hl2 + mult * atr[i]  # type: ignore
        b_dn = hl2 - mult * atr[i]  # type: ignore
        if i > 0 and upper[i - 1] is not None:
            if b_up > upper[i - 1] and candles[i - 1]["close"] <= upper[i - 1]:
                b_up = min(b_up, upper[i - 1])  # type: ignore
            if b_dn < lower[i - 1] and candles[i - 1]["close"] >= lower[i - 1]:
                b_dn = max(b_dn, lower[i - 1])  # type: ignore
        upper[i] = b_up
        lower[i] = b_dn
        if direction <= 0 and c["close"] > (upper[i - 1] if i > 0 and upper[i - 1] else b_up):
            direction = 1
        elif direction >= 0 and c["close"] < (lower[i - 1] if i > 0 and lower[i - 1] else b_dn):
            direction = -1
    return direction


# ─── Per-kind evaluators ──────────────────────────────────────────────────────
def _eval_alert(alert, candles: List[dict]) -> Optional[Tuple[float, str]]:
    """
    Returns (current_value, "fire reason") if the alert should fire on this bar,
    or returns the (current_value, None) if not yet — last_value should still be
    updated so the next eval can detect a cross.

    Wait — to keep the logic clean, this returns:
        None              → couldn't compute (insufficient candles)
        (value, None)     → computed but not firing
        (value, msg)      → firing with `msg` as the human-readable reason
    """
    if not candles or len(candles) < 5:
        return None
    closes = [c["close"] for c in candles]
    last_price = closes[-1]
    last_value = alert.last_value  # may be None if first eval

    try:
        params = json.loads(alert.params or "{}")
    except Exception:
        params = {}

    cur: Optional[float] = None
    cond = (alert.condition or "").lower()
    target = alert.target
    kind = alert.kind

    if kind == "price":
        cur = last_price
        if target is None:
            return (cur, None)
        if last_value is None:
            return (cur, None)  # baseline
        if cond == "above" and cur > target and last_value <= target:
            return (cur, f"Price crossed above ${target:,.2f} (now ${cur:,.2f})")
        if cond == "below" and cur < target and last_value >= target:
            return (cur, f"Price crossed below ${target:,.2f} (now ${cur:,.2f})")
        if cond == "crossover" and cur > target and last_value <= target:
            return (cur, f"Price crossed above ${target:,.2f} (now ${cur:,.2f})")
        if cond == "crossunder" and cur < target and last_value >= target:
            return (cur, f"Price crossed below ${target:,.2f} (now ${cur:,.2f})")
        return (cur, None)

    if kind == "rsi":
        period = int(params.get("period", 14))
        cur = _rsi(closes, period)
        if cur is None or target is None:
            return (cur, None) if cur is not None else None
        if last_value is None:
            return (cur, None)
        if cond in ("above", "crossover") and cur > target and last_value <= target:
            return (cur, f"RSI({period}) crossed above {target:.1f} (now {cur:.1f})")
        if cond in ("below", "crossunder") and cur < target and last_value >= target:
            return (cur, f"RSI({period}) crossed below {target:.1f} (now {cur:.1f})")
        return (cur, None)

    if kind == "ema_cross":
        period = int(params.get("period", 50))
        ema_now = _ema(closes, period)
        if ema_now is None:
            return None
        # current "value" we track = (price - ema). Sign tells us which side.
        cur = last_price - ema_now
        if last_value is None:
            return (cur, None)
        if cond == "crossover" and cur > 0 and last_value <= 0:
            return (cur, f"Price crossed above EMA({period}) at ${ema_now:,.2f} (price ${last_price:,.2f})")
        if cond == "crossunder" and cur < 0 and last_value >= 0:
            return (cur, f"Price crossed below EMA({period}) at ${ema_now:,.2f} (price ${last_price:,.2f})")
        return (cur, None)

    if kind == "macd_cross_zero":
        fast = int(params.get("fast", 12))
        slow = int(params.get("slow", 26))
        cur = _macd_line(closes, fast, slow)
        if cur is None:
            return None
        if last_value is None:
            return (cur, None)
        if cond == "crossover" and cur > 0 and last_value <= 0:
            return (cur, f"MACD({fast},{slow}) turned positive ({cur:+.2f})")
        if cond == "crossunder" and cur < 0 and last_value >= 0:
            return (cur, f"MACD({fast},{slow}) turned negative ({cur:+.2f})")
        return (cur, None)

    if kind == "supertrend_flip":
        period = int(params.get("period", 10))
        mult = float(params.get("mult", 3))
        d = _supertrend_dir(candles, period, mult)
        if d is None:
            return None
        cur = float(d)
        if last_value is None:
            return (cur, None)
        if cur != last_value:
            side = "LONG ▲" if cur > 0 else "SHORT ▼"
            return (cur, f"SuperTrend({period},{mult:g}) flipped {side} (price ${last_price:,.2f})")
        return (cur, None)

    if kind == "fvg_retest":
        # ICT fair-value-gap retest: fire when the latest candle wicks back
        # into an unfilled FVG below price (bull side) or above price (bear
        # side). The detector + signal helper live in auto_trader so the
        # alert path uses the same logic the Auto Trader rules engine uses.
        try:
            from app.services.auto_trader import _fvg_retest_signal
        except Exception as e:
            logger.warning(f"alerts_engine: fvg_retest import failed: {e}")
            return None
        wanted = "long" if cond == "bull" else "short"
        # Back-compat: legacy alerts (created before the ATR/displacement
        # filters existed) only had `min_gap_pct` + `max_age_bars`. If the
        # new ATR keys are absent we treat it as a legacy spec and DISABLE
        # the new filters so an old alert keeps firing the way the user
        # originally configured it. New alerts (saved with the new keys)
        # get the volatility-aware quality filtering by default.
        is_legacy = ("min_gap_pct" in (params or {})
                     and "min_gap_atr_mult" not in (params or {}))
        default_atr_mult  = 0.0 if is_legacy else 0.10
        default_disp_mult = 0.0 if is_legacy else 0.5
        default_max_age   = 100 if is_legacy else 200
        side, note = _fvg_retest_signal(
            candles,
            min_gap_pct=float(params.get("min_gap_pct", 0.05 if is_legacy else 0.0) or 0.0),
            min_gap_atr_mult=float(params.get("min_gap_atr_mult", default_atr_mult) or 0.0),
            disp_atr_mult=float(params.get("disp_atr_mult", default_disp_mult) or 0.0),
            max_age_bars=int(params.get("max_age_bars", default_max_age) or default_max_age),
        )
        # `cur` doubles as the bar timestamp so a future re-eval has SOMETHING
        # to compare to. Alerts auto-flip to "triggered" on first fire so this
        # value is mostly cosmetic — but storing the bar time helps when a
        # user reactivates a triggered alert.
        cur = float(int(candles[-1].get("time") or 0))
        if side != wanted:
            return (cur, None)
        word = "BULL" if cond == "bull" else "BEAR"
        return (cur, f"{word} FVG retest — {note} (price ${last_price:,.2f})")

    return None


# ─── Telegram dispatch ────────────────────────────────────────────────────────
async def _tg_send(telegram_id: int, text: str) -> None:
    """Direct Bot API HTTP call — works from any process."""
    try:
        from app.config import settings
        token = settings.TELEGRAM_BOT_TOKEN
        if not token:
            logger.warning("alerts_engine: TELEGRAM_BOT_TOKEN missing, skip DM")
            return
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": telegram_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
    except Exception as e:
        logger.warning(f"alerts_engine DM failed for {telegram_id}: {e}")


def _fmt_dm(alert, msg: str, fire_count: Optional[int] = None) -> str:
    head = "🔔  <b>Chart alert triggered</b>"
    # For repeating alerts, include the running fire count so the user knows
    # this is one of many DMs from the same alert (and can mute/cancel if it's
    # too noisy).
    if fire_count and fire_count > 1:
        head = f"🔔  <b>Chart alert · fire #{fire_count}</b>"
    return (
        f"{head}\n"
        f"<i>{alert.label or alert.kind}</i>\n\n"
        f"{msg}\n"
        f"<i>{alert.symbol} · {alert.timeframe}  ·  {datetime.utcnow().strftime('%H:%M UTC')}</i>"
    )


def _claim_fire(row, now: datetime) -> Tuple[bool, str]:
    """Decide whether a fired alert should actually dispatch a DM, and update
    bookkeeping (fire_count, last_fired_at, fired_today_count, fired_today_date)
    on the row in-place when allowed.

    Returns (allowed, reason). When allowed is False the loop should still
    `commit` the updated last_value/last_eval_at so the cross detector keeps
    advancing — we just skip the DM and don't bump the fire counters.
    """
    fire_mode = (getattr(row, "fire_mode", None) or "once").lower()
    today_iso = now.date().isoformat()

    # Daily-cap roll-over: if we crossed UTC midnight since the last fire, the
    # per-day counter resets before we evaluate the cap.
    fired_today_date = getattr(row, "fired_today_date", None)
    fired_today_count = int(getattr(row, "fired_today_count", 0) or 0)
    if fired_today_date != today_iso:
        fired_today_count = 0
        # Reset on the row even if the fire ends up suppressed — keeps the
        # stored date in sync with reality so a later fire on the same day
        # increments from 0 instead of replaying yesterday's count.
        row.fired_today_date = today_iso
        row.fired_today_count = 0

    if fire_mode == "every_cross_with_cooldown":
        cooldown_min = int(getattr(row, "cooldown_minutes", 0) or 0)
        last_fired = getattr(row, "last_fired_at", None)
        if cooldown_min > 0 and last_fired is not None:
            if now - last_fired < timedelta(minutes=cooldown_min):
                return (False, "cooldown")
        daily_cap = getattr(row, "daily_cap", None)
        if daily_cap is not None and fired_today_count >= int(daily_cap):
            return (False, "daily_cap")

    # Allowed → bump counters
    row.fire_count = int(getattr(row, "fire_count", 0) or 0) + 1
    row.last_fired_at = now
    row.fired_today_count = fired_today_count + 1
    row.fired_today_date = today_iso
    return (True, "ok")


# ─── Main loop ────────────────────────────────────────────────────────────────
ALERTS_INTERVAL_S = int(os.getenv("ALERTS_INTERVAL_S", "30"))

# Module-level singleton guard — defends against duplicate launches in the same
# worker if the executor advisory lock is reclaimed (lock churn → repeated
# _start_executor_tasks() calls → would otherwise spawn parallel alert loops).
_ENGINE_RUNNING = False


async def run_alerts_engine() -> None:
    """Long-running loop. Started by the worker that holds the executor lock.

    Cross-process safety:
      * In-process: `_ENGINE_RUNNING` blocks duplicate launches in the same worker.
      * Cross-worker (during a brief lock-handoff window): each candidate row is
        re-loaded with `SELECT … FOR UPDATE` inside a fresh transaction. Only
        one worker can acquire the lock; the other waits, then re-reads the
        already-flipped `status='triggered'` and skips → at most one DM per fire.
    """
    global _ENGINE_RUNNING
    if _ENGINE_RUNNING:
        logger.info("alerts_engine: already running in this worker — skip duplicate launch")
        return
    _ENGINE_RUNNING = True

    from app.database import SessionLocal
    from app.models import IndicatorAlert, User

    logger.info(f"🔔 Alerts engine started (interval={ALERTS_INTERVAL_S}s)")
    while True:
        try:
            db = SessionLocal()
            try:
                alerts = (db.query(IndicatorAlert)
                            .filter(IndicatorAlert.status == "active")
                            .all())
            finally:
                db.close()

            if alerts:
                # Group by (symbol, timeframe) so we fetch each candle set once
                groups: Dict[Tuple[str, str], List] = {}
                for a in alerts:
                    groups.setdefault((a.symbol, a.timeframe), []).append(a)

                for (sym, tf), bucket in groups.items():
                    candles = await _fetch_candles(sym, tf, limit=200)
                    if not candles:
                        continue
                    for a in bucket:
                        # Each alert gets its own short transaction so the row
                        # lock is held for the smallest possible window.
                        db = SessionLocal()
                        try:
                            # SELECT ... FOR UPDATE serialises concurrent workers
                            # racing the same alert during a lock-handoff window.
                            row = (db.query(IndicatorAlert)
                                     .filter(IndicatorAlert.id == a.id)
                                     .with_for_update()
                                     .first())
                            if not row or row.status != "active":
                                db.rollback()
                                continue
                            try:
                                result = _eval_alert(row, candles)
                            except Exception as e:
                                logger.warning(f"alerts_engine eval #{row.id}: {e}")
                                db.rollback()
                                continue
                            if result is None:
                                db.rollback()
                                continue
                            cur, msg = result
                            now = datetime.utcnow()
                            row.last_value = float(cur) if cur is not None else None
                            row.last_eval_at = now
                            row.eval_count = (row.eval_count or 0) + 1
                            user_telegram_id = None
                            fire_count_snapshot = 0
                            fire_mode = (getattr(row, "fire_mode", None) or "once").lower()
                            if msg:
                                allowed, reason = _claim_fire(row, now)
                                if allowed:
                                    # 'once' alerts auto-disarm; repeating alerts
                                    # keep status='active' so the watch loop
                                    # picks them up again next cycle.
                                    if fire_mode == "once":
                                        row.status = "triggered"
                                    row.triggered_at = now
                                    row.triggered_price = float(candles[-1]["close"])
                                    row.triggered_message = msg
                                    fire_count_snapshot = int(row.fire_count or 0)
                                    user = db.query(User).filter(User.id == row.user_id).first()
                                    if user and user.telegram_id and not str(user.telegram_id).startswith("WEB-"):
                                        try:
                                            user_telegram_id = int(user.telegram_id)
                                        except (TypeError, ValueError):
                                            user_telegram_id = None
                                else:
                                    # Suppressed by cooldown / daily-cap. We DO
                                    # still commit last_value above so the next
                                    # cross is detected as a fresh edge.
                                    msg = None
                                    logger.info(
                                        f"alerts_engine: alert #{row.id} cross suppressed ({reason})"
                                    )
                            # Commit FIRST so the row-level lock + status flip
                            # become visible to other workers before we DM.
                            # If the DM crashes after commit, we still won't
                            # double-fire on the next sample.
                            row_label_snapshot = row.label
                            row_kind_snapshot = row.kind
                            row_symbol_snapshot = row.symbol
                            row_tf_snapshot = row.timeframe
                            db.commit()
                            if msg and user_telegram_id is not None:
                                try:
                                    # Build a DM-shaped object without keeping the
                                    # ORM row attached after the session closed.
                                    class _Snap:
                                        label = row_label_snapshot
                                        kind = row_kind_snapshot
                                        symbol = row_symbol_snapshot
                                        timeframe = row_tf_snapshot
                                    await _tg_send(
                                        user_telegram_id,
                                        _fmt_dm(_Snap(), msg, fire_count=fire_count_snapshot),
                                    )
                                except Exception as e:
                                    logger.warning(f"alerts_engine DM dispatch #{a.id}: {e}")
                        except Exception as e:
                            logger.warning(f"alerts_engine alert #{a.id}: {e}")
                            try:
                                db.rollback()
                            except Exception:
                                pass
                        finally:
                            db.close()
        except Exception as e:
            logger.error(f"alerts_engine loop error: {e}")

        # Piggy-back the auto-trader tick on this loop. It shares the
        # executor advisory lock so we never double-fire across workers.
        try:
            from app.services.auto_trader import tick_auto_strategies
            await tick_auto_strategies()
        except Exception as e:
            logger.error(f"auto_trader tick error: {e}")

        await asyncio.sleep(ALERTS_INTERVAL_S)
