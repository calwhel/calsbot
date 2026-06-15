"""Per-strategy × per-account cTrader execution assignments."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_enabled_fire_targets(db, strategy, prefs) -> List[Dict[str, Any]]:
    """Return [{ctid, lot_size}, ...] for live fan-out. Never empty when prefs default exists."""
    from app.strategy_models import StrategyAccountAssignment

    rows = (
        db.query(StrategyAccountAssignment)
        .filter(
            StrategyAccountAssignment.strategy_id == strategy.id,
            StrategyAccountAssignment.enabled.is_(True),
        )
        .order_by(StrategyAccountAssignment.id)
        .all()
    )
    if rows:
        out = []
        for r in rows:
            ctid = str(r.ctid or "").strip()
            if ctid:
                out.append({"ctid": ctid, "lot_size": r.lot_size})
        if out:
            return out

    legacy = (getattr(strategy, "ctrader_account_id", None) or "").strip()
    legacy_lot = getattr(strategy, "ctrader_account_lot", None)
    if legacy:
        return [{"ctid": legacy, "lot_size": legacy_lot}]

    default = (getattr(prefs, "ctrader_account_id", None) or "").strip() if prefs else ""
    if default:
        return [{"ctid": default, "lot_size": None}]
    return []


def fire_slot_count(db, strategy, prefs) -> int:
    targets = get_enabled_fire_targets(db, strategy, prefs)
    return max(1, len(targets))


def list_strategy_assignments(db, strategy_id: int) -> List[Dict[str, Any]]:
    from app.strategy_models import StrategyAccountAssignment

    rows = (
        db.query(StrategyAccountAssignment)
        .filter(StrategyAccountAssignment.strategy_id == strategy_id)
        .order_by(StrategyAccountAssignment.id)
        .all()
    )
    return [
        {
            "ctid": str(r.ctid),
            "enabled": bool(r.enabled),
            "lot_size": r.lot_size,
        }
        for r in rows
    ]


def upsert_strategy_assignments(
    db,
    strategy_id: int,
    assignments: List[Dict[str, Any]],
    *,
    allowed_ctids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Replace assignment rows for listed ctids; omit ctids not in payload."""
    from app.services.ctrader_client import normalize_account_lot
    from app.strategy_models import StrategyAccountAssignment

    seen = set()
    out: List[Dict[str, Any]] = []
    for item in assignments or []:
        ctid = str(item.get("ctid") or item.get("account_id") or "").strip()
        if not ctid or ctid in seen:
            continue
        if allowed_ctids is not None and ctid not in allowed_ctids:
            continue
        seen.add(ctid)
        enabled = bool(item.get("enabled", False))
        raw_lot = item.get("lot_size")
        if raw_lot in (None, "", "null"):
            lot_size = None
        else:
            lot_size = normalize_account_lot(raw_lot)
            if lot_size is None and enabled:
                raise ValueError("Invalid lot size (min 0.01, step 0.01)")
        row = (
            db.query(StrategyAccountAssignment)
            .filter(
                StrategyAccountAssignment.strategy_id == strategy_id,
                StrategyAccountAssignment.ctid == ctid,
            )
            .first()
        )
        if not row:
            row = StrategyAccountAssignment(
                strategy_id=strategy_id,
                ctid=ctid,
                enabled=enabled,
                lot_size=lot_size,
                created_at=datetime.utcnow(),
            )
            db.add(row)
        else:
            row.enabled = enabled
            row.lot_size = lot_size
        out.append({"ctid": ctid, "enabled": enabled, "lot_size": lot_size})

    # Disable rows for ctids in allowed set but not in payload (toggle off)
    if allowed_ctids is not None:
        existing = (
            db.query(StrategyAccountAssignment)
            .filter(StrategyAccountAssignment.strategy_id == strategy_id)
            .all()
        )
        payload_ctids = {x["ctid"] for x in out}
        for row in existing:
            if str(row.ctid) in allowed_ctids and str(row.ctid) not in payload_ctids:
                row.enabled = False
    return out


def migrate_legacy_strategy_assignments(engine) -> None:
    """One-time: copy user_strategies.ctrader_account_id → assignment rows."""
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            conn.execute(text(
                "INSERT INTO strategy_account_assignments "
                "(strategy_id, ctid, enabled, lot_size, created_at) "
                "SELECT s.id, s.ctrader_account_id, TRUE, s.ctrader_account_lot, NOW() "
                "FROM user_strategies s "
                "WHERE s.ctrader_account_id IS NOT NULL "
                "AND TRIM(s.ctrader_account_id) <> '' "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM strategy_account_assignments a "
                "  WHERE a.strategy_id = s.id AND a.ctid = s.ctrader_account_id"
                ")"
            ))
            conn.commit()
    except Exception as exc:
        logger.warning(
            "migrate_legacy_strategy_assignments: %s", type(exc).__name__
        )
