"""User-facing Telegram trade-alert diagnostics."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from app.services.strategy_executor import _telegram_int_id


async def build_telegram_alert_status(user, db) -> Dict[str, Any]:
    from app.strategy_models import (
        StrategyPortalSettings,
        StrategyExecution,
        UserStrategy,
    )
    from app.services.ctrader_order_queue import get_gate_stats
    tid = getattr(user, "telegram_id", None)
    tg_linked = bool(tid) and not str(tid).startswith("WEB-")
    tg_int = _telegram_int_id(user)

    portal = (
        db.query(StrategyPortalSettings)
        .filter(StrategyPortalSettings.user_id == user.id)
        .first()
    )
    dm_paper = True if not portal else bool(portal.dm_paper_alerts)
    dm_live = True if not portal else bool(portal.dm_live_alerts)

    scanning = (
        db.query(UserStrategy)
        .filter(
            UserStrategy.user_id == user.id,
            UserStrategy.status.in_(["paper", "active"]),
        )
        .count()
    )
    draft_paused = (
        db.query(UserStrategy)
        .filter(
            UserStrategy.user_id == user.id,
            UserStrategy.status.in_(["draft", "paused"]),
        )
        .count()
    )

    since = datetime.utcnow() - timedelta(hours=24)
    fires_24h = (
        db.query(StrategyExecution)
        .filter(
            StrategyExecution.user_id == user.id,
            StrategyExecution.fired_at >= since,
        )
        .count()
    )
    last_fire = (
        db.query(StrategyExecution.fired_at)
        .filter(StrategyExecution.user_id == user.id)
        .order_by(StrategyExecution.fired_at.desc())
        .limit(1)
        .scalar()
    )

    blockers: List[str] = []
    if not tg_linked or not tg_int:
        blockers.append("telegram_not_linked")
    if not dm_paper and not dm_live:
        blockers.append("all_dm_alerts_disabled")
    if scanning == 0:
        blockers.append("no_scanning_strategies")
    if draft_paused:
        blockers.append("draft_or_paused_strategies")

    top_gates: Dict[str, int] = {}
    strats = (
        db.query(UserStrategy)
        .filter(
            UserStrategy.user_id == user.id,
            UserStrategy.status.in_(["paper", "active"]),
        )
        .limit(40)
        .all()
    )
    for s in strats:
        gs = get_gate_stats(s.id).get("stats") or {}
        for k, v in gs.items():
            top_gates[k] = top_gates.get(k, 0) + int(v or 0)
    top_gate_list = sorted(top_gates.items(), key=lambda kv: -kv[1])[:6]

    bots = {}
    try:
        from app.services.telegram_dm import bot_usernames
        bots = await bot_usernames()
    except Exception:
        pass

    hints = []
    if not tg_linked:
        hints.append(
            "Log in with Telegram on the portal, or link your account in Settings — "
            "web-only accounts cannot receive trade DMs."
        )
    if tg_linked:
        main_bot = bots.get("main") or "AISIGNALPERPBOT"
        forex_bot = bots.get("forex") or "TradehubStrategyBot"
        hints.append(
            f"Open @{main_bot} and @{forex_bot} in Telegram and tap /start on each — "
            "trade alerts try both bots."
        )
    if fires_24h == 0 and scanning > 0:
        hints.append(
            "Executor is scanning but no setups matched your entry filters in the last 24h — "
            "that is normal until conditions align."
        )
    if top_gate_list:
        top_name = top_gate_list[0][0].replace("blk_", "").replace("_", " ")
        hints.append(f"Most common scan blocker right now: {top_name}.")

    return {
        "telegram_linked": tg_linked,
        "telegram_chat_id": tg_int,
        "dm_paper_alerts": dm_paper,
        "dm_live_alerts": dm_live,
        "scanning_strategies": scanning,
        "draft_or_paused": draft_paused,
        "fires_last_24h": fires_24h,
        "last_fire_at": last_fire.isoformat() + "Z" if last_fire else None,
        "blockers": blockers,
        "top_gate_blockers": [
            {"gate": k.replace("blk_", ""), "count": v} for k, v in top_gate_list
        ],
        "bots": {
            "main": bots.get("main"),
            "forex": bots.get("forex"),
        },
        "hints": hints,
        "alerts_ready": bool(
            tg_linked and tg_int and (dm_paper or dm_live) and scanning > 0
        ),
    }
