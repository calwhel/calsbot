"""
One-shot / periodic repairs so saved strategies actually enter the executor scan pool.

Fixes the most common silent failures without manual per-strategy edits:
  • draft / paused → paper (executor only scans paper + active)
  • asset_class column out of sync with config (mobile vs web)
  • empty tradfi universe when a symbol can be inferred
  • stale OPEN executions blocking max_open_positions
  • performance.open_trades counter drift after auto-expire
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TRADFI = frozenset({"forex", "index", "stock"})
_KNOWN_SYMBOLS = (
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "XAUUSD", "XAGUSD",
    "NAS100", "SPX500", "US30", "GER40", "UK100", "NDX", "SPX",
)
_SYMBOL_RE = re.compile(
    r"\b(" + "|".join(_KNOWN_SYMBOLS) + r")\b",
    re.IGNORECASE,
)


def resolve_strategy_asset_class(strategy) -> str:
    """Match executor routing — trust config over stale DB column."""
    from app.services.asset_classes import normalize_asset_class

    cfg = strategy.config or {}
    col = (getattr(strategy, "asset_class", None) or "").strip()
    cfg_ac = (cfg.get("asset_class") or cfg.get("_asset_class") or "").strip()
    if col == "crypto" and cfg_ac and cfg_ac != "crypto":
        return normalize_asset_class(cfg_ac)
    return normalize_asset_class(col or cfg_ac)


def _infer_symbol(name: str, cfg: dict) -> Optional[str]:
    desc = str(cfg.get("description") or "")
    for blob in (name or "", desc):
        m = _SYMBOL_RE.search(blob.upper().replace("/", ""))
        if m:
            return m.group(1).upper()
    return None


def heal_strategy_row(strategy, stats: Dict[str, int]) -> bool:
    """Repair one strategy row in-place. Returns True if mutated."""
    cfg = dict(strategy.config or {})
    entry = cfg.get("entry_conditions") or {}
    if entry.get("entry_type") == "tradingview_webhook":
        return False

    changed = False
    ac = resolve_strategy_asset_class(strategy)

    if getattr(strategy, "asset_class", None) != ac and ac != "crypto":
        strategy.asset_class = ac
        stats["asset_class_fixed"] = stats.get("asset_class_fixed", 0) + 1
        changed = True

    if cfg.get("asset_class") != ac:
        cfg["asset_class"] = ac
        changed = True

    if ac in _TRADFI:
        uni = dict(cfg.get("universe") or {})
        syms = [
            s.strip().upper()
            for s in (uni.get("symbols") or [])
            if isinstance(s, str) and s.strip()
        ]
        if not syms:
            inferred = _infer_symbol(strategy.name or "", cfg)
            if inferred:
                cfg["universe"] = {"type": "specific", "symbols": [inferred]}
                stats["universe_fixed"] = stats.get("universe_fixed", 0) + 1
                changed = True
        elif uni.get("type") != "specific":
            cfg["universe"] = {"type": "specific", "symbols": syms}
            changed = True

    if strategy.status in ("draft", "paused"):
        strategy.status = "paper"
        stats["promoted_to_paper"] = stats.get("promoted_to_paper", 0) + 1
        changed = True

    if changed:
        strategy.config = cfg
        strategy.updated_at = datetime.utcnow()
    return changed


def heal_strategies(
    db,
    *,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Heal strategies for one user or the whole database."""
    from app.strategy_models import UserStrategy

    stats: Dict[str, int] = {
        "promoted_to_paper": 0,
        "asset_class_fixed": 0,
        "universe_fixed": 0,
        "rows_touched": 0,
    }
    q = db.query(UserStrategy)
    if user_id is not None:
        q = q.filter(UserStrategy.user_id == user_id)
    rows: List = q.all()
    for strat in rows:
        if heal_strategy_row(strat, stats):
            stats["rows_touched"] += 1
            db.add(strat)
    if stats["rows_touched"]:
        db.commit()
        logger.info(
            "[strategy-heal] user=%s promoted=%s asset_class=%s universe=%s touched=%s",
            user_id or "all",
            stats["promoted_to_paper"],
            stats["asset_class_fixed"],
            stats["universe_fixed"],
            stats["rows_touched"],
        )
    return stats


def close_stale_opens_for_user(
    db,
    user_id: int,
    *,
    stale_after_hours: int = 4,
) -> int:
    """Expire stuck OPEN rows for one user — frees max_open_positions."""
    from app.strategy_models import StrategyExecution, StrategyPerformance

    cutoff = datetime.utcnow() - timedelta(hours=stale_after_hours)
    stale = (
        db.query(StrategyExecution)
        .filter(
            StrategyExecution.user_id == user_id,
            StrategyExecution.outcome == "OPEN",
            StrategyExecution.closed_at.is_(None),
            StrategyExecution.fired_at <= cutoff,
        )
        .all()
    )
    if not stale:
        return 0
    touched_strategies = set()
    count = 0
    for ex in stale:
        try:
            ex.outcome = "EXPIRED"
            ex.closed_at = ex.fired_at + timedelta(hours=stale_after_hours)
            ex.exit_price = ex.entry_price
            ex.pnl_pct = 0.0
            note = " | auto-expired: stuck open (strategy heal)"
            ex.notes = (ex.notes or "") + note
            touched_strategies.add(ex.strategy_id)
            count += 1
        except Exception as exc:
            logger.warning("[strategy-heal] expire exec %s failed: %s", ex.id, exc)
    if count:
        db.commit()
        for sid in touched_strategies:
            open_cnt = (
                db.query(StrategyExecution)
                .filter(
                    StrategyExecution.strategy_id == sid,
                    StrategyExecution.outcome == "OPEN",
                )
                .count()
            )
            perf = (
                db.query(StrategyPerformance)
                .filter(StrategyPerformance.strategy_id == sid)
                .first()
            )
            if perf:
                perf.open_trades = open_cnt
        db.commit()
        logger.info(
            "[strategy-heal] user=%s expired %s stale OPEN execution(s)",
            user_id,
            count,
        )
    return count


def heal_user_account(db, user_id: int) -> Dict[str, Any]:
    """Full heal for one portal user."""
    stats = heal_strategies(db, user_id=user_id)
    stats["stale_expired"] = close_stale_opens_for_user(db, user_id, stale_after_hours=4)
    return stats


async def heal_on_executor_startup() -> Dict[str, Any]:
    """Run once when the executor worker acquires the advisory lock."""
    from app.database import BgSessionLocal as SessionLocal
    from app.services.strategy_executor import close_stale_open_executions

    db = SessionLocal()
    try:
        stats = heal_strategies(db)
    finally:
        db.close()

    try:
        stats["stale_expired_global"] = await close_stale_open_executions(
            stale_after_hours=4,
        )
    except Exception as exc:
        logger.warning("[strategy-heal] global stale close failed: %s", exc)
        stats["stale_expired_global"] = 0
    return stats
