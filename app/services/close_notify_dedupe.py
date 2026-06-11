"""Atomic once-per-execution close notification claim."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def claim_close_notification(db: Any, execution_id: int) -> bool:
    """Return True only for the worker that may send the close Telegram/push.

    Uses UPDATE ... WHERE notified_close_at IS NULL RETURNING so sweep + reconcile
    (or any duplicate close path) cannot double-notify the same execution.
    """
    from sqlalchemy import text

    try:
        row = db.execute(
            text(
                "UPDATE strategy_executions "
                "SET notified_close_at = :now "
                "WHERE id = :id "
                "  AND notified_close_at IS NULL "
                "  AND outcome <> 'OPEN' "
                "RETURNING id"
            ),
            {"now": datetime.utcnow(), "id": int(execution_id)},
        ).fetchone()
        db.commit()
        if row:
            return True
        db.rollback()
        return False
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning(
            "[notify] close claim failed exec=%s: %s", execution_id, exc,
        )
        return False
