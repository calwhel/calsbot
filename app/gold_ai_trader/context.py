"""Rich pre-digested market snapshot for Claude."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from app.gold_ai_trader.config import SYMBOL
from app.gold_ai_trader.klines import get_gold_ai_klines
from app.gold_ai_trader.guardrails import (
    calls_today,
    cost_today_usd,
    open_position_count,
    trades_today,
)
from app.gold_ai_trader.models import GoldAiLesson
from app.gold_ai_trader.context_levels import build_key_levels_block, build_premium_discount_block
from app.gold_ai_trader.context_regime import build_regime_block, build_htf_bias_block
from app.gold_ai_trader.context_history import build_recent_decisions_block, parse_zone_from_detail
from app.gold_ai_trader.context_bands import build_trade_bands_block
from app.gold_ai_trader.call_gates import collect_key_levels
from app.gold_ai_trader.learning import format_setup_stats_block
from app.gold_ai_trader.cisd_modifier import build_cisd_block
from app.gold_ai_trader.htf_bias import htf_bias_summary
from app.gold_ai_trader.setup_readiness import ReadinessResult, format_readiness_block


def _atr(closes, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(closes)):
        tr = abs(closes[i] - closes[i - 1])
        trs.append(tr)
    if len(trs) < period:
        return sum(trs) / max(len(trs), 1)
    return sum(trs[-period:]) / period


def _summarize_candles(rows, n: int = 12) -> str:
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


def build_data_quality_block(market_data: Optional[Dict[str, Any]]) -> list:
    """Structured data-quality flag so Claude weights confidence honestly."""
    if not market_data:
        return [
            "=== DATA QUALITY ===",
            "Status: unknown (no market_data passed)",
        ]
    from app.gold_ai_trader.data_quality import format_data_source, gold_data_ok_for_claude

    ok, reason = gold_data_ok_for_claude(market_data)
    tag = format_data_source(market_data)
    live = market_data.get("live_source") or market_data.get("price_source") or "unknown"
    ks = market_data.get("kline_source") or "none"
    bars = market_data.get("kline_bars", 0)
    stale = "yes" if market_data.get("klines_stale") else "no"
    bid = market_data.get("bid")
    ask = market_data.get("ask")
    spread = ""
    if bid and ask:
        spread = f" | bid/ask: {float(bid):.2f}/{float(ask):.2f}"

    price_mode = "ctrader-live"
    if live != "ctrader":
        price_mode = f"{live} (non-live)"
    elif market_data.get("price_source") == "ctrader" and not market_data.get("bid"):
        price_mode = "ctrader-live or 5m-close-fallback"

    return [
        "=== DATA QUALITY (structured) ===",
        f"Gate: {'PASS' if ok else 'BLOCKED'} ({reason})",
        f"Price source: {price_mode} | Kline source: {ks} | Bars: {bars} | Stale: {stale}{spread}",
        f"Tag: {tag}",
    ]


def build_smt_block(smt: Optional[Dict[str, Any]]) -> list:
    if not smt:
        return []
    mod = smt.get("modifier", 0)
    if mod == 0 and not smt.get("data_available"):
        return [
            "=== SMT DIVERGENCE (modifier only — not a trigger) ===",
            smt.get("detail", "SMT: unavailable"),
        ]
    sign = "+" if mod > 0 else ""
    confirms = "confirms" in str(smt.get("detail", "")).lower()
    header = (
        "=== SMT CONFIRMATION (XAG aligns with gold — confidence modifier) ==="
        if confirms
        else "=== SMT DIVERGENCE (confidence modifier — NOT standalone trigger) ==="
    )
    hint = (
        f"Suggested confidence adjustment: {sign}{mod} (XAG confirms trade direction — lean +8 when aligned)"
        if confirms and mod >= 8
        else f"Suggested confidence adjustment: {sign}{mod} "
        f"(supports {'this' if mod > 0 else 'opposing' if mod < 0 else 'neutral'} direction)"
    )
    lines = [
        header,
        hint,
        smt.get("detail", ""),
    ]
    if smt.get("dxy_note"):
        lines.append(f"Reference note: {smt['dxy_note']}")
    ref = smt.get("reference_symbol")
    src = smt.get("reference_source")
    if ref:
        lines.append(f"Reference series: {ref} (source: {src or 'unknown'})")
    return lines


async def build_context_snapshot(
    *,
    candidate: Candidate,
    price: float,
    session: str,
    db,
    cfg,
    user_id: Optional[int],
    market_data: Optional[Dict[str, Any]] = None,
    smt: Optional[Dict[str, Any]] = None,
    cisd: Optional[Dict[str, Any]] = None,
) -> str:
    k5 = await get_gold_ai_klines("5m", 60, user_id=user_id) or []
    k15 = await get_gold_ai_klines("15m", 40, user_id=user_id) or []
    k1h = await get_gold_ai_klines("1h", 50, user_id=user_id) or []
    k4h = await get_gold_ai_klines("4h", 30, user_id=user_id) or []
    k_daily = await get_gold_ai_klines("1d", 5, user_id=user_id) or []

    closes = [float(r[4]) for r in k5 if r and len(r) >= 5]
    atr = _atr(closes)
    atr_pct = (atr / price * 100) if price and atr else 0.0

    vols = [float(r[5]) for r in k5 if r and len(r) >= 6]
    rvol = 1.0
    if len(vols) >= 20:
        avg = sum(vols[-21:-1]) / 20
        rvol = (vols[-1] / avg) if avg else 1.0

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

    setup_zone = parse_zone_from_detail(candidate.detail)
    bias = htf_bias_summary(k1h, k4h, k_daily)
    level_values = collect_key_levels(
        price, session, cfg, now, k_daily, k1h, k5,
    )
    key_levels = build_key_levels_block(
        spot=price,
        atr=atr,
        session=session,
        cfg=cfg,
        now=now,
        k_daily=k_daily,
        k_1h=k1h,
        k_5m=k5,
        setup_zone=setup_zone,
    )
    regime = build_regime_block(k1h, k5, k4h)
    htf_block = build_htf_bias_block(bias)
    data_quality = build_data_quality_block(market_data)
    recent_decisions = build_recent_decisions_block(db, session=session)
    smt_block = build_smt_block(smt or candidate.raw.get("smt"))
    cisd_block = build_cisd_block(cisd or candidate.raw.get("cisd"))
    premium_discount = build_premium_discount_block(
        spot=price, k5=k5, k1h=k1h, now=now, session=session, cfg=cfg,
    )
    struct_line = candidate.raw.get("structure_score_line") or ""
    readiness_raw = candidate.raw.get("readiness_score")
    if readiness_raw is not None:
        readiness_block = format_readiness_block(
            ReadinessResult(
                score=int(readiness_raw),
                passed=True,
                breakdown=candidate.raw.get("readiness_breakdown") or "",
                checklist=candidate.raw.get("readiness_checklist") or {},
            ),
            candidate.type,
        )
    else:
        readiness_block = []
    zone_tf = candidate.raw.get("zone_tf", "5m")
    trade_bands = build_trade_bands_block(
        spot=price,
        atr=atr,
        direction=candidate.direction,
        setup_detail=candidate.detail,
        key_levels=level_values,
    )
    setup_stats = format_setup_stats_block(db, session=session)

    htf_align = candidate.raw.get("htf_align", "unknown")

    lines = [
        "=== GOLD AI TRADER CONTEXT (XAUUSD) ===",
        f"Timestamp UTC: {now.isoformat()}Z",
        f"Session: {session.upper()} | Killzone: yes | Minutes into session: {mins_in_session}",
        "",
        *data_quality,
        "",
        "=== PRICE ===",
        f"Spot: {price:.2f}",
        f"ATR(14) 5m: {atr:.2f} ({atr_pct:.3f}% of price) | RVOL(5m): {rvol:.2f}x",
        "",
        *htf_block,
        "",
        *regime,
        "",
        *key_levels,
        "",
        *premium_discount,
        "",
        "=== RECENT PATH (5m, oldest→newest) ===",
        _summarize_candles(k5, 10),
        "",
        "=== STRUCTURE / BIAS (engine) ===",
        f"15m trend: {'bullish' if len(k15) >= 2 and float(k15[-1][4]) > float(k15[-2][4]) else 'bearish/mixed'}",
        f"HTF alignment (setup): {htf_align}",
        "",
        "=== TRIGGER (why Claude was called) ===",
        f"Type: {candidate.type} | Direction bias: {candidate.direction} | Zone TF: {zone_tf}",
        f"Detail: {candidate.detail}",
        f"Quality vs ATR: {candidate.quality_atr:.2f}× (engine estimate)",
        struct_line or "Structure score: unavailable",
        f"Take threshold: {cfg.confidence_threshold}% (unchanged — score honestly vs this bar)",
        f"Suggested invalidation max: {atr:.2f} (1.0× 5m ATR — wider SL → lower confidence)",
        "",
        *readiness_block,
        "" if not readiness_block else "",
        *trade_bands,
        "",
        *setup_stats,
        "",
        *smt_block,
        "" if not smt_block else "",
        *cisd_block,
        "" if not cisd_block else "",
        *recent_decisions,
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
        "Target positive expectancy — good (not perfect) setups at 50%+ with valid 2:1 R:R are tradable.",
        "Minor missing confluence lowers confidence; borderline edge belongs in 50–60, not auto-skip.",
        "Weight data quality and HTF bias — downgrade confidence on stale feeds or clear counter-structure.",
    ]
    return "\n".join(line for line in lines if line is not None)
