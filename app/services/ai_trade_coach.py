"""AI Trade Coach — weekly review of a user's trading.

Cost-minimised by design:
- One Claude Haiku call per user per 7 days, hard-cached in `weekly_coach_reports`.
- Force-regenerate only via explicit `force=True` from an admin or the user clicking
  a refresh button (rate-limited to once per 24h on the endpoint side).
- Stub-out (no AI call) if the user has < 3 closed trades in the window — those
  reports cost more in tokens than they're worth.

Public API
----------
- `await get_or_generate_weekly(user_id, db, force=False) -> dict` — main entry.
- `_build_context(user_id, db) -> dict` — pure helper (no AI), used for tests.

Report shape
------------
{
  "generated_at": iso,
  "week_start":   YYYY-MM-DD,
  "stub":         bool,                # true when no AI call was made
  "stats": {
    "trades": int, "wins": int, "losses": int, "win_rate": float,
    "pnl_pct": float, "best": {symbol, pnl_pct}, "worst": {symbol, pnl_pct},
  },
  "summary":         str,              # 1-2 sentence headline
  "patterns":        [str, ...],       # 2-4 observations
  "recommendations": [str, ...],       # 2-4 actionable items
  "missed_exits":    [str, ...],       # 0-3 specific call-outs (optional)
}
"""
from __future__ import annotations
import json
import logging
import os
from datetime import datetime, timedelta, date
from typing import Optional, Dict, List

from sqlalchemy import text

logger = logging.getLogger(__name__)

# Hard cap so a chatty model can't blow our budget. ~700 output tokens is plenty.
_LLM_MAX_TOKENS = 800
_MIN_TRADES_FOR_AI = 3
_LOOKBACK_DAYS = 7


def _week_start(today: Optional[date] = None) -> date:
    """ISO week start (Monday) for cache key."""
    d = today or datetime.utcnow().date()
    return d - timedelta(days=d.weekday())


def _build_context(user_id: int, db) -> Dict:
    """Pull the last 7d of executions for the user. Pure DB read, no AI."""
    cutoff = datetime.utcnow() - timedelta(days=_LOOKBACK_DAYS)

    rows = db.execute(text("""
        SELECT
            e.symbol, e.direction, e.outcome, e.pnl_pct, e.is_paper,
            e.entry_price, e.exit_price, e.tp_price, e.sl_price,
            e.fired_at, e.closed_at, e.notes,
            s.name AS strategy_name
        FROM strategy_executions e
        JOIN user_strategies s ON s.id = e.strategy_id
        WHERE e.user_id = :uid
          AND COALESCE(e.closed_at, e.fired_at) > :cutoff
        ORDER BY COALESCE(e.closed_at, e.fired_at) DESC
        LIMIT 200
    """), {"uid": user_id, "cutoff": cutoff}).fetchall()

    closed: List[Dict] = []
    open_count = 0
    for r in rows:
        if r.outcome == "OPEN":
            open_count += 1
            continue
        if r.outcome not in ("WIN", "LOSS", "BREAKEVEN") or r.pnl_pct is None:
            continue
        closed.append({
            "symbol":    r.symbol,
            "direction": r.direction,
            "outcome":   r.outcome,
            "pnl_pct":   round(float(r.pnl_pct), 2),
            "is_paper":  bool(r.is_paper),
            "strategy":  r.strategy_name,
            "hit_tp":    bool(r.exit_price and r.tp_price and abs(r.exit_price - r.tp_price) < 1e-6),
            "hit_sl":    bool(r.exit_price and r.sl_price and abs(r.exit_price - r.sl_price) < 1e-6),
            "duration_min": int(((r.closed_at or r.fired_at) - r.fired_at).total_seconds() / 60) if r.fired_at else None,
        })

    wins   = sum(1 for c in closed if c["outcome"] == "WIN")
    losses = sum(1 for c in closed if c["outcome"] == "LOSS")
    pnl    = sum(c["pnl_pct"] for c in closed)
    best   = max(closed, key=lambda c: c["pnl_pct"]) if closed else None
    worst  = min(closed, key=lambda c: c["pnl_pct"]) if closed else None

    return {
        "trades_closed": closed,
        "open_count":    open_count,
        "stats": {
            "trades":   len(closed),
            "wins":     wins,
            "losses":   losses,
            "win_rate": round(wins / len(closed) * 100, 1) if closed else 0.0,
            "pnl_pct":  round(pnl, 2),
            "best":     {"symbol": best["symbol"], "pnl_pct": best["pnl_pct"]} if best else None,
            "worst":    {"symbol": worst["symbol"], "pnl_pct": worst["pnl_pct"]} if worst else None,
        },
    }


def _stub_report(ctx: Dict) -> Dict:
    """No-AI report for users with too little data."""
    n = ctx["stats"]["trades"]
    return {
        "generated_at":    datetime.utcnow().isoformat(),
        "week_start":      _week_start().isoformat(),
        "stub":            True,
        "stats":           ctx["stats"],
        "summary":         (
            f"Only {n} closed trade{'s' if n != 1 else ''} this week — not enough "
            "history yet for a meaningful review. Keep your strategies running and "
            "check back next week."
        ),
        "patterns":        [],
        "recommendations": [
            "Activate at least one strategy if all are paused.",
            "Consider running 2-3 strategies in parallel so the coach has more data to learn from.",
        ],
        "missed_exits":    [],
    }


_SYSTEM = """You are a candid trading coach reviewing a user's last 7 days of trades.

Your job: read the JSON of closed trades + summary stats below and produce a SHORT, structured weekly review. Be specific, name symbols and strategies, and call out concrete patterns. No fluff, no generic advice.

Return ONLY valid JSON in this exact shape:
{
  "summary": "1-2 sentence headline of how the week went",
  "patterns": ["observation 1", "observation 2", "observation 3"],
  "recommendations": ["actionable item 1", "actionable item 2", "actionable item 3"],
  "missed_exits": ["call-out 1", "call-out 2"]
}

Rules:
- 2-4 patterns, 2-4 recommendations, 0-3 missed_exits.
- Each item ≤ 140 chars.
- "missed_exits" only if you see trades that would have been winners with a tighter trail or smaller TP — otherwise return [].
- Mention specific symbols (BTC, ETH, etc.) and strategy names where relevant.
- Don't moralise. Don't say "consider" or "you might want to". Be direct: "Tighten SL on BTC longs to 0.8%."
"""


def _user_prompt(ctx: Dict) -> str:
    # Trim to top 30 trades by absolute PnL — keeps prompt small
    trades = sorted(ctx["trades_closed"], key=lambda c: abs(c["pnl_pct"]), reverse=True)[:30]
    return f"""STATS:
{json.dumps(ctx["stats"], separators=(",", ":"))}

OPEN POSITIONS RIGHT NOW: {ctx["open_count"]}

CLOSED TRADES (last 7d, top 30 by |pnl|):
{json.dumps(trades, separators=(",", ":"))}

Return ONLY the JSON review."""


async def _call_haiku(ctx: Dict) -> Optional[Dict]:
    """Single Claude Haiku call. Returns parsed dict or None on any failure."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("[Coach] ANTHROPIC_API_KEY not set — skipping AI generation")
        return None
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)
        msg = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=_LLM_MAX_TOKENS,
            system=_SYSTEM,
            messages=[{"role": "user", "content": _user_prompt(ctx)}],
        )
        text_out = msg.content[0].text.strip()
        if text_out.startswith("```"):
            text_out = text_out.strip("`")
            if text_out.lower().startswith("json"):
                text_out = text_out[4:]
            text_out = text_out.strip()
        parsed = json.loads(text_out)
        # Defensive shape clamp
        return {
            "summary":         str(parsed.get("summary", ""))[:600],
            "patterns":        [str(x)[:200] for x in (parsed.get("patterns") or [])][:4],
            "recommendations": [str(x)[:200] for x in (parsed.get("recommendations") or [])][:4],
            "missed_exits":    [str(x)[:200] for x in (parsed.get("missed_exits") or [])][:3],
        }
    except Exception as e:
        logger.warning(f"[Coach] Haiku call failed: {e}")
        return None


async def get_or_generate_weekly(user_id: int, db, force: bool = False) -> Dict:
    """Returns the cached weekly report, or generates a new one if missing/forced.

    Caches by (user_id, week_start). Force=True bypasses the cache and overwrites.
    """
    week_start = _week_start()

    if not force:
        cached = db.execute(text("""
            SELECT report_json FROM weekly_coach_reports
            WHERE user_id = :uid AND week_start = :ws
            LIMIT 1
        """), {"uid": user_id, "ws": week_start}).fetchone()
        if cached:
            try:
                report = cached.report_json if isinstance(cached.report_json, dict) else json.loads(cached.report_json)
                report["cached"] = True
                return report
            except Exception:
                pass  # fall through to regenerate

    ctx = _build_context(user_id, db)

    if ctx["stats"]["trades"] < _MIN_TRADES_FOR_AI:
        report = _stub_report(ctx)
    else:
        ai = await _call_haiku(ctx)
        if ai is None:
            # Soft fallback — return stats-only report so the UI never breaks
            report = _stub_report(ctx)
            report["summary"] = (
                f"Closed {ctx['stats']['trades']} trades for "
                f"{ctx['stats']['pnl_pct']:+.2f}% — AI review unavailable right now."
            )
        else:
            report = {
                "generated_at":    datetime.utcnow().isoformat(),
                "week_start":      week_start.isoformat(),
                "stub":            False,
                "stats":           ctx["stats"],
                **ai,
            }

    # Upsert into cache
    try:
        db.execute(text("""
            INSERT INTO weekly_coach_reports (user_id, week_start, report_json, generated_at)
            VALUES (:uid, :ws, CAST(:rj AS JSONB), :gen)
            ON CONFLICT (user_id, week_start)
            DO UPDATE SET report_json = EXCLUDED.report_json, generated_at = EXCLUDED.generated_at
        """), {
            "uid": user_id,
            "ws":  week_start,
            "rj":  json.dumps(report),
            "gen": datetime.utcnow(),
        })
        db.commit()
    except Exception as e:
        logger.warning(f"[Coach] cache upsert failed: {e}")
        db.rollback()

    report["cached"] = False
    return report
