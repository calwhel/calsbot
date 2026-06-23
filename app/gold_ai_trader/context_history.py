"""Recent Gold AI decision history for Claude continuity."""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import List, Optional


def _rationale_preview(reasoning: Optional[str], decision: Optional[dict], limit: int = 80) -> str:
    if decision and isinstance(decision, dict):
        r = (decision.get("rationale") or "").strip()
        if r:
            return r[:limit]
    if reasoning:
        return reasoning.strip()[:limit]
    return ""


def build_recent_decisions_block(
    db,
    *,
    session: str,
    limit: int = 6,
    window_minutes: int = 120,
) -> List[str]:
    """Compact recent decision lines for Claude context."""
    from app.gold_ai_trader.models import GoldAiDecision

    now = datetime.utcnow()
    since = now - timedelta(minutes=window_minutes)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    rows = (
        db.query(GoldAiDecision)
        .filter(
            GoldAiDecision.ts >= max(since, today_start),
            GoldAiDecision.session == session,
        )
        .order_by(GoldAiDecision.ts.desc())
        .limit(limit)
        .all()
    )
    if not rows:
        return [
            "=== RECENT DECISIONS (this session) ===",
            "No prior decisions in the last 2h — first evaluation this window.",
        ]

    lines = ["=== RECENT DECISIONS (newest first, last 2h) ==="]
    for row in rows:
        ts = row.ts.strftime("%H:%M") if row.ts else "??:??"
        setup = row.candidate_type or "unknown"
        action = (row.action or "skip").upper()
        conf = row.confidence if row.confidence is not None else "N/A"
        exec_tag = ""
        if row.executed and row.execution_id:
            exec_tag = f" exec#{row.execution_id}"
        elif action == "TAKE" and not row.executed:
            exec_tag = " (blocked)"
        preview = _rationale_preview(row.reasoning, row.decision)
        line = f"- {ts} {setup} {action} conf={conf}{exec_tag}"
        if preview:
            line += f' — "{preview}"'
        lines.append(line)

    # Setup-type skip streak summary
    by_type: dict = {}
    for row in rows:
        t = row.candidate_type or "unknown"
        by_type.setdefault(t, {"skip": 0, "take": 0})
        if (row.action or "").lower() == "take":
            by_type[t]["take"] += 1
        else:
            by_type[t]["skip"] += 1
    streak_parts = []
    for t, counts in by_type.items():
        if counts["skip"] >= 2:
            streak_parts.append(f"skipped {counts['skip']}× {t}")
    if streak_parts:
        lines.append("Continuity: " + "; ".join(streak_parts[:4]) + ".")

    return lines


def parse_zone_from_detail(detail: str) -> Optional[tuple]:
    """Extract zone bounds from setup detail string if present."""
    if not detail:
        return None
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*[–\-]\s*(\d+(?:\.\d+)?)",
        detail,
    )
    if not m:
        m = re.search(r"zone=(\d+(?:\.\d+)?)\s*[–\-]\s*(\d+(?:\.\d+)?)", detail)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            return None
    m = re.search(r"@\s*(\d+(?:\.\d+)?)", detail)
    if m:
        try:
            lvl = float(m.group(1))
            return lvl, lvl
        except ValueError:
            return None
    return None
