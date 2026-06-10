"""
Scheduled discovery scan runner — gold / forex / index only.

Runs due schedules every 60s on the executor worker, reuses discovery scanners
+ grade_setup, and sends Telegram alerts for setups meeting min_grade_alert.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_GRADE_RANK = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
_SCHEDULE_SEM: Optional[asyncio.Semaphore] = None
_ALERT_DEDUP: Dict[Tuple, float] = {}


def _schedule_sem() -> asyncio.Semaphore:
    global _SCHEDULE_SEM
    if _SCHEDULE_SEM is None:
        _SCHEDULE_SEM = asyncio.Semaphore(int(os.environ.get("SCAN_SCHEDULE_MAX_PARALLEL", "1")))
    return _SCHEDULE_SEM


def _grade_gte(grade: str, min_grade: str) -> bool:
    return _GRADE_RANK.get((grade or "F").upper(), 0) >= _GRADE_RANK.get(
        (min_grade or "B").upper(), 4
    )


def _dedupe_key(schedule_id: int, entry: Dict, symbol: str, scan_type: str) -> Tuple:
    return (
        schedule_id,
        scan_type,
        (symbol or "").upper(),
        entry.get("label", ""),
        entry.get("primaryType", ""),
        entry.get("timeframe", ""),
        entry.get("session", ""),
    )


def _dedupe_ok(key: Tuple, interval_minutes: int) -> bool:
    now = time.monotonic()
    ttl = max(60.0, interval_minutes * 60.0)
    stale = [k for k, ts in _ALERT_DEDUP.items() if now - ts > ttl * 2]
    for k in stale:
        _ALERT_DEDUP.pop(k, None)
    last = _ALERT_DEDUP.get(key)
    if last and (now - last) < ttl:
        return False
    _ALERT_DEDUP[key] = now
    return True


def _portal_build_link(scan_type: str, symbol: str) -> str:
    domain = os.environ.get("PUBLIC_DOMAIN", "tradehubmarkets.com")
    base = f"https://{domain}/app"
    frag = {"gold": "gold-finder", "forex": "forex-finder", "index": "index-finder"}.get(
        scan_type, "scan"
    )
    sym = f"&symbol={symbol}" if symbol else ""
    return f"{base}#build-{frag}{sym}"


async def _run_discovery_for_schedule(row) -> Optional[Dict]:
    """Execute one scheduled scan via the same discovery functions as the UI."""
    scan_type = (row.scan_type or "gold").lower()
    quality_cfg = dict(row.quality_cfg_json or {})
    categories = row.categories_json if isinstance(row.categories_json, list) else None
    params = dict(row.scan_params_json or {})
    days = int(params.get("days", 90))
    direction = str(params.get("direction") or "BOTH").upper()
    symbol = (row.symbol or params.get("symbol") or "").strip()

    async def _noop_progress(_msg: str):
        pass

    if scan_type == "gold":
        from app.services.gold_strategy_scanner import run_gold_discovery
        return await run_gold_discovery(
            days=days,
            direction_mode=direction,
            progress_cb=_noop_progress,
            user_id=row.user_id,
            quality_cfg=quality_cfg,
            categories=categories,
        )
    if scan_type == "forex":
        from app.services.forex_pair_strategy_scanner import run_forex_discovery
        return await run_forex_discovery(
            symbol=symbol or "EURUSD",
            days=days,
            direction_mode=direction,
            progress_cb=_noop_progress,
            user_id=row.user_id,
            quality_cfg=quality_cfg,
            categories=categories,
        )
    if scan_type == "index":
        from app.services.index_strategy_scanner import run_index_discovery
        return await run_index_discovery(
            symbol=symbol or "NAS100",
            days=days,
            direction_mode=direction,
            progress_cb=_noop_progress,
            user_id=row.user_id,
            quality_cfg=quality_cfg,
            categories=categories,
        )
    return None


def _format_alert(entry: Dict, result: Dict, schedule, build_link: str) -> str:
    sym = result.get("symbol") or schedule.symbol or "XAUUSD"
    grade = entry.get("grade", "?")
    direction = entry.get("direction", "LONG")
    conf = ", ".join(f"✓ {c}" for c in (entry.get("confirmations") or [])[:5])
    n_confl = entry.get("confluence_count") or len(entry.get("confluences") or [])
    st = entry.get("stats") or {}
    wr = st.get("test_win_rate") or st.get("win_rate") or 0
    unit = "pts" if result.get("asset_class") == "index" else "pips"
    sl = entry.get("sl_pips", "")
    tp = entry.get("tp_pips", "")
    lines = [
        f"<b>🏆 Grade {grade} Setup — {sym} {direction}</b>",
        f"{entry.get('label', 'Strategy')} · {entry.get('timeframe', '')} · {entry.get('session_label', '')}",
        f"Zone: SL {sl} / TP {tp} {unit}",
    ]
    if conf:
        lines.append(conf)
    if n_confl:
        lines.append(f"🔥 {n_confl} confluence{'s' if n_confl != 1 else ''}")
    lines.append(f"Backtest WR: {wr}% (walk-forward test)")
    lines.append(f'<a href="{build_link}">⚡ Build in portal</a>')
    return "\n".join(lines)


async def _send_schedule_alerts(row, result: Dict) -> int:
    if not result or not result.get("ok"):
        return 0
    from app.database import BgSessionLocal
    from app.models import User
    from app.services.strategy_executor import _telegram_int_id, _tg_send

    db = BgSessionLocal()
    try:
        user = None
        if row.user_id:
            user = db.query(User).filter(User.id == row.user_id).first()
        if not user and row.uid:
            user = db.query(User).filter(User.uid == row.uid).first()
        if not user:
            return 0
        tg = _telegram_int_id(getattr(user, "telegram_id", None))
        if not tg:
            return 0
    finally:
        db.close()

    min_grade = (row.min_grade_alert or "B").upper()
    scan_type = (row.scan_type or "gold").lower()
    sym = result.get("symbol") or row.symbol or ""
    build_link = _portal_build_link(scan_type, sym)
    asset_class = result.get("asset_class") or ("index" if scan_type == "index" else "forex")

    sent = 0
    for entry in (result.get("leaderboard") or [])[:5]:
        grade = (entry.get("grade") or "F").upper()
        if not _grade_gte(grade, min_grade):
            continue
        dkey = _dedupe_key(row.id, entry, sym, scan_type)
        if not _dedupe_ok(dkey, int(row.interval_minutes or 60)):
            continue
        text = _format_alert(entry, result, row, build_link)
        try:
            ok = await _tg_send(tg, text, asset_class=asset_class)
            if ok:
                sent += 1
        except Exception as exc:
            logger.warning("[scan-schedule] telegram failed schedule=%s: %s", row.id, exc)
    return sent


async def _process_one_schedule(row) -> None:
    async with _schedule_sem():
        try:
            result = await _run_discovery_for_schedule(row)
            if result and result.get("ok"):
                n = await _send_schedule_alerts(row, result)
                logger.info(
                    "[scan-schedule] id=%s type=%s passed=%s alerts=%s",
                    row.id, row.scan_type, result.get("combos_passed"), n,
                )
        except Exception as exc:
            logger.error("[scan-schedule] id=%s failed: %s", row.id, exc)
        finally:
            from app.database import BgSessionLocal
            from app.strategy_models import ScanSchedule
            db = BgSessionLocal()
            try:
                sched = db.query(ScanSchedule).filter(ScanSchedule.id == row.id).first()
                if sched:
                    sched.last_run_at = datetime.utcnow()
                    db.commit()
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass
            finally:
                db.close()


async def _process_due_schedules() -> None:
    from app.database import BgSessionLocal
    from app.strategy_models import ScanSchedule

    now = datetime.utcnow()
    db = BgSessionLocal()
    try:
        rows = (
            db.query(ScanSchedule)
            .filter(ScanSchedule.enabled.is_(True))
            .all()
        )
        due: List = []
        for row in rows:
            interval = max(15, int(row.interval_minutes or 60))
            if row.last_run_at is None:
                due.append(row)
            elif (now - row.last_run_at) >= timedelta(minutes=interval):
                due.append(row)
    finally:
        db.close()

    if not due:
        return
    await asyncio.gather(*[_process_one_schedule(r) for r in due])


async def run_scan_schedule_loop() -> None:
    """Background loop — check due schedules every 60s (executor worker only)."""
    logger.info("[scan-schedule] loop started (60s interval)")
    await asyncio.sleep(20)
    while True:
        try:
            await _process_due_schedules()
        except Exception as exc:
            logger.error("[scan-schedule] loop error: %s", exc)
        await asyncio.sleep(60)
