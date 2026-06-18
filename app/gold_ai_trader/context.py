"""Rich pre-digested market snapshot for Claude."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.gold_ai_trader.config import SYMBOL, ASSET_CLASS
from app.gold_ai_trader.scanner import Candidate


def _atr(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(closes)):
        tr = abs(closes[i] - closes[i - 1])
        trs.append(tr)
    if len(trs) < period:
        return sum(trs) / max(len(trs), 1)
    return sum(trs[-period:]) / period


def _summarize_candles(rows: List[list], n: int = 12) -> str:
    if not rows:
        return "No recent candles."
    tail = rows[-n:]
    parts = []
    for row in tail:
        o, h, l, c = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        body = c - o
        tag = "bull" if body > 0 else "bear" if body < 0 else "doji"
        parts.append(f"{tag} O{o:.2f} H{h:.2f} L{l:.2f} C{c:.2f}")
    return " → ".join(parts)


async def build_context_snapshot(
    *,
    candidate: Candidate,
    price: float,
    session: str,
    db,
    cfg,
    user_id: Optional[int],
) -> str:
    from app.services.tradfi_prices import get_klines
    from app.gold_ai_trader.guardrails import calls_today, trades_today, cost_today_usd, open_position_count
    from app.gold_ai_trader.models import GoldAiLesson

    k5 = await get_klines(SYMBOL, ASSET_CLASS, "5m", 60) or []
    k15 = await get_klines(SYMBOL, ASSET_CLASS, "15m", 40) or []
    closes = [float(r[4]) for r in k5 if r and len(r) >= 5]
    atr = _atr(closes)
    atr_pct = (atr / price * 100) if price and atr else 0.0

    vols = [float(r[5]) for r in k5 if r and len(r) >= 6]
    rvol = 1.0
    if len(vols) >= 20:
        avg = sum(vols[-21:-1]) / 20
        rvol = (vols[-1] / avg) if avg else 1.0

    pdh = pdl = session_hi = session_lo = vwap_note = "unknown"
    try:
        from app.services.strategy_ta import _get_klines
        import httpx

        async with httpx.AsyncClient(timeout=10) as http:
            cache = {"__asset_class__": ASSET_CLASS}
            # PDH/PDL via prev level eval detail is embedded in candidate when applicable
            pdh = pdl = "see trigger detail"
    except Exception:
        pass

    open_pos = open_position_count(db, user_id) if user_id else 0
    pnl_today = 0.0
    if user_id:
        from app.strategy_models import StrategyExecution

        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        closed = (
            db.query(StrategyExecution)
            .filter(
                StrategyExecution.user_id == user_id,
                StrategyExecution.symbol == SYMBOL,
                StrategyExecution.closed_at >= today,
                StrategyExecution.notes.like("%gold_ai_trader%"),
            )
            .all()
        )
        pnl_today = sum(float(x.pnl_pct or 0) for x in closed)

    lesson_row = (
        db.query(GoldAiLesson)
        .filter(GoldAiLesson.session == session)
        .order_by(GoldAiLesson.ts.desc())
        .first()
    )
    lessons = (lesson_row.digest if lesson_row else "No session lessons yet — trade selectively.")

    now = datetime.utcnow()
    mins_in_session = 0
    if session == "london":
        mins_in_session = max(0, (now.hour - cfg.london_start_hour) * 60 + now.minute)
    elif session == "new_york":
        mins_in_session = max(0, (now.hour - cfg.ny_start_hour) * 60 + now.minute)

    lines = [
        "=== GOLD AI TRADER CONTEXT (XAUUSD) ===",
        f"Timestamp UTC: {now.isoformat()}Z",
        f"Session: {session.upper()} | Killzone: yes | Minutes into session: {mins_in_session}",
        "",
        "=== PRICE ===",
        f"Spot: {price:.2f}",
        f"ATR(14) 5m: {atr:.2f} ({atr_pct:.3f}% of price) | RVOL(5m): {rvol:.2f}x",
        "",
        "=== RECENT PATH (5m, oldest→newest) ===",
        _summarize_candles(k5, 10),
        "",
        "=== STRUCTURE / BIAS (engine) ===",
        f"15m trend: {'bullish' if len(k15) >= 2 and float(k15[-1][4]) > float(k15[-2][4]) else 'bearish/mixed'}",
        "",
        "=== KEY LEVELS (approx) ===",
        f"PDH/PDL: {pdh} / {pdl} (engine sweep checks active)",
        f"Session range: high/low tracking via ICT prev-level + liquidity modules",
        "",
        "=== TRIGGER (why Claude was called) ===",
        f"Type: {candidate.type} | Direction bias: {candidate.direction}",
        f"Detail: {candidate.detail}",
        f"Quality vs ATR: {candidate.quality_atr:.2f}× (engine estimate)",
        "",
        "=== ACCOUNT (DEMO) ===",
        f"Open gold_ai positions: {open_pos}/1 max",
        f"Trades today: {trades_today(db)}/{cfg.max_trades_day} | Claude calls: {calls_today(db)}/{cfg.max_calls_day}",
        f"Est. API cost today: ${cost_today_usd(db):.4f}",
        f"Demo P&L today (closed, %): {pnl_today:+.2f}",
        "",
        "=== RECENT LESSONS ===",
        lessons,
        "",
        "=== DECISION RULE REMINDER ===",
        "Default SKIP unless high conviction. Require clear invalidation and ≥2:1 R:R.",
    ]
    return "\n".join(lines)
