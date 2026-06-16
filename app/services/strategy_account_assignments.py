"""Per-strategy × per-account cTrader execution assignments."""
from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Asset classes routed through cTrader with per-account assignment toggles.
TRADFI_BROKER_ASSET_CLASSES = ("forex", "index", "metals", "commodity")

_CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS strategy_account_assignments (
        id SERIAL PRIMARY KEY,
        strategy_id INTEGER NOT NULL REFERENCES user_strategies(id),
        ctrader_account_id VARCHAR(40) NOT NULL,
        enabled BOOLEAN NOT NULL DEFAULT FALSE,
        lot_size DOUBLE PRECISION,
        created_at TIMESTAMP DEFAULT NOW(),
        CONSTRAINT uq_strategy_account_acct UNIQUE (strategy_id, ctrader_account_id)
    )
"""


def _coerce_lot_size(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, Decimal):
        return float(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _assignment_account_id(item: Dict[str, Any]) -> str:
    """Resolve account id from API payload (new or legacy key)."""
    return str(
        item.get("ctrader_account_id")
        or item.get("ctid")
        or item.get("account_id")
        or ""
    ).strip()


def serialize_assignment_row(
    ctrader_account_id: str, enabled: bool, lot_size,
) -> Dict[str, Any]:
    return {
        "ctrader_account_id": str(ctrader_account_id),
        "enabled": bool(enabled),
        "lot_size": _coerce_lot_size(lot_size),
    }


def ensure_strategy_account_assignments_table(bind) -> None:
    """Idempotent — creates strategy_account_assignments if missing."""
    try:
        from app.strategy_models import StrategyAccountAssignment
        StrategyAccountAssignment.__table__.create(bind=bind, checkfirst=True)
    except Exception as exc:
        logger.warning(
            "ensure_strategy_account_assignments_table: %s", type(exc).__name__
        )
        try:
            from sqlalchemy import text
            with bind.connect() as conn:
                conn.execute(text(_CREATE_TABLE_SQL))
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS idx_saa_strategy_id "
                    "ON strategy_account_assignments(strategy_id)"
                ))
                conn.commit()
        except Exception as exc2:
            logger.error(
                "ensure_strategy_account_assignments_table DDL failed: %s",
                type(exc2).__name__,
            )
            raise


def _format_assignments_log(targets: List[Dict[str, Any]]) -> str:
    parts = []
    for t in targets or []:
        acct = str(t.get("ctrader_account_id") or t.get("ctid") or "?")
        lot = t.get("lot_size")
        parts.append(f"{acct}@{lot if lot is not None else 'default'}")
    return "[" + ", ".join(parts) + "]"


def resolve_live_fire_intent(
    db,
    strategy,
    asset_class: str,
    prefs=None,
    *,
    prefetched_targets: Optional[List[Dict[str, Any]]] = None,
) -> tuple:
    """Whether executor should place broker orders (vs paper-only tracking).

    cTrader strategies require BOTH:
      1. strategy.status == 'active' (explicit Go Live — never paper)
      2. At least one enabled per-account assignment

    Paper strategies always return wants_live=False even if a live account is ticked.

    prefetched_targets — when provided (including []), skips the assignment table query.

    Returns (wants_live: bool, fire_targets: list).
    """
    ac = (asset_class or "").strip().lower()
    status = (getattr(strategy, "status", None) or "").strip().lower()
    status_active = status == "active"
    if ac not in TRADFI_BROKER_ASSET_CLASSES:
        return status_active, []

    if not status_active:
        return False, []

    targets = get_enabled_fire_targets(
        db,
        strategy,
        prefs,
        for_live_fire=True,
        prefetched_rows=prefetched_targets,
    )
    if not targets:
        return False, []
    return True, targets


def get_enabled_fire_targets(
    db, strategy, prefs, *, for_live_fire: bool = True,
    prefetched_rows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return [{ctrader_account_id, lot_size}, ...] for routing.

    When for_live_fire=True (default): explicit enabled assignment rows, or
    legacy strategy.ctrader_account_id only — never prefs.ctrader_account_id.
    When for_live_fire=False: may include prefs default (non-live UI/helpers).

    prefetched_rows — cycle-level batch prefetch; None triggers a per-strategy query.
    """
    from app.strategy_models import StrategyAccountAssignment

    rows = None
    if prefetched_rows is not None:
        if prefetched_rows:
            logger.info(
                "get_enabled_fire_targets strategy=%s enabled=%s (prefetched)",
                getattr(strategy, "id", "?"),
                _format_assignments_log(prefetched_rows),
            )
            return list(prefetched_rows)
        rows = []
    else:
        try:
            ensure_strategy_account_assignments_table(db.get_bind())
        except Exception as exc:
            logger.warning(
                "get_enabled_fire_targets: ensure table failed strategy=%s: %s",
                getattr(strategy, "id", "?"),
                type(exc).__name__,
            )

        try:
            rows = (
                db.query(StrategyAccountAssignment)
                .filter(
                    StrategyAccountAssignment.strategy_id == strategy.id,
                    StrategyAccountAssignment.enabled.is_(True),
                )
                .order_by(StrategyAccountAssignment.id)
                .all()
            )
        except Exception as exc:
            logger.warning(
                "get_enabled_fire_targets: query failed strategy=%s: %s",
                getattr(strategy, "id", "?"),
                exc,
            )
            rows = []

    if rows:
        out = []
        for r in rows:
            acct_id = str(r.ctrader_account_id or "").strip()
            if acct_id:
                out.append({
                    "ctrader_account_id": acct_id,
                    "lot_size": _coerce_lot_size(r.lot_size),
                })
        if out:
            logger.info(
                "get_enabled_fire_targets strategy=%s enabled=%s",
                getattr(strategy, "id", "?"),
                _format_assignments_log(out),
            )
            return out

    legacy = (getattr(strategy, "ctrader_account_id", None) or "").strip()
    legacy_lot = getattr(strategy, "ctrader_account_lot", None)
    if legacy:
        out = [{"ctrader_account_id": legacy, "lot_size": legacy_lot}]
        logger.info(
            "[live-fire] using legacy account binding strategy=%s %s",
            getattr(strategy, "id", "?"),
            _format_assignments_log(out),
        )
        return out

    if not for_live_fire:
        default = (getattr(prefs, "ctrader_account_id", None) or "").strip() if prefs else ""
        if default:
            out = [{"ctrader_account_id": default, "lot_size": None}]
            logger.info(
                "get_enabled_fire_targets strategy=%s prefs_default=%s",
                getattr(strategy, "id", "?"),
                _format_assignments_log(out),
            )
            return out

    logger.info(
        "get_enabled_fire_targets strategy=%s enabled=[] "
        "(no explicit assignment%s)",
        getattr(strategy, "id", "?"),
        "" if for_live_fire else " or default",
    )
    return []


def fire_slot_count(db, strategy, prefs) -> int:
    targets = get_enabled_fire_targets(db, strategy, prefs, for_live_fire=True)
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
        serialize_assignment_row(r.ctrader_account_id, r.enabled, r.lot_size)
        for r in rows
    ]


def upsert_strategy_assignments(
    db,
    strategy_id: int,
    assignments: List[Dict[str, Any]],
    *,
    allowed_ctids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Replace assignment rows for listed accounts; omit accounts not in payload."""
    from app.services.ctrader_client import normalize_account_lot
    from app.strategy_models import StrategyAccountAssignment

    ensure_strategy_account_assignments_table(db.get_bind())

    seen = set()
    out: List[Dict[str, Any]] = []
    for item in assignments or []:
        acct_id = _assignment_account_id(item)
        if not acct_id or acct_id in seen:
            continue
        if allowed_ctids is not None and acct_id not in allowed_ctids:
            continue
        seen.add(acct_id)
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
                StrategyAccountAssignment.ctrader_account_id == acct_id,
            )
            .first()
        )
        if not row:
            row = StrategyAccountAssignment(
                strategy_id=strategy_id,
                ctrader_account_id=acct_id,
                enabled=enabled,
                lot_size=lot_size,
                created_at=datetime.utcnow(),
            )
            db.add(row)
        else:
            row.enabled = enabled
            row.lot_size = lot_size
        out.append(serialize_assignment_row(acct_id, enabled, lot_size))

    if allowed_ctids is not None:
        existing = (
            db.query(StrategyAccountAssignment)
            .filter(StrategyAccountAssignment.strategy_id == strategy_id)
            .all()
        )
        payload_ids = {x["ctrader_account_id"] for x in out}
        for row in existing:
            rid = str(row.ctrader_account_id)
            if rid in allowed_ctids and rid not in payload_ids:
                row.enabled = False
    return out


def migrate_legacy_strategy_assignments(engine) -> None:
    """One-time: copy user_strategies.ctrader_account_id → assignment rows."""
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            conn.execute(text(
                "INSERT INTO strategy_account_assignments "
                "(strategy_id, ctrader_account_id, enabled, lot_size, created_at) "
                "SELECT s.id, s.ctrader_account_id, TRUE, s.ctrader_account_lot, NOW() "
                "FROM user_strategies s "
                "WHERE s.ctrader_account_id IS NOT NULL "
                "AND TRIM(s.ctrader_account_id) <> '' "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM strategy_account_assignments a "
                "  WHERE a.strategy_id = s.id "
                "  AND a.ctrader_account_id = s.ctrader_account_id"
                ")"
            ))
            conn.commit()
    except Exception as exc:
        logger.warning(
            "migrate_legacy_strategy_assignments: %s", type(exc).__name__
        )
