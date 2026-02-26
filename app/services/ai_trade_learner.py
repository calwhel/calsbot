import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_lessons_cache = []
_lessons_cache_time = None
LESSONS_CACHE_TTL = 300

_live_context_cache = None
_live_context_cache_time = None
LIVE_CONTEXT_CACHE_TTL = 120


def _get_gemini_client():
    try:
        from google import genai
        api_key = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        return genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return None


def get_live_trading_context() -> str:
    global _live_context_cache, _live_context_cache_time

    now = datetime.utcnow()
    if _live_context_cache and _live_context_cache_time and (now - _live_context_cache_time).total_seconds() < LIVE_CONTEXT_CACHE_TTL:
        return _live_context_cache

    try:
        from app.database import SessionLocal
        from app.models import Trade

        db = SessionLocal()
        try:
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_start = now - timedelta(days=7)

            today_trades = db.query(Trade).filter(
                Trade.closed_at >= today_start,
                Trade.status == 'closed'
            ).all()

            week_trades = db.query(Trade).filter(
                Trade.closed_at >= week_start,
                Trade.status == 'closed'
            ).all()

            recent_3 = db.query(Trade).filter(
                Trade.status == 'closed'
            ).order_by(Trade.closed_at.desc()).limit(3).all()

            open_count = db.query(Trade).filter(Trade.status == 'open').count()

            today_pnl = sum(t.pnl or 0 for t in today_trades)
            today_wins = sum(1 for t in today_trades if (t.pnl or 0) > 0)
            today_total = len(today_trades)

            week_wins = sum(1 for t in week_trades if (t.pnl or 0) > 0)
            week_total = len(week_trades)
            week_win_rate = (week_wins / week_total * 100) if week_total > 0 else 0

            streak = 0
            streak_type = None
            for t in recent_3:
                outcome = 'WIN' if (t.pnl or 0) > 0 else 'LOSS'
                if streak_type is None:
                    streak_type = outcome
                    streak = 1
                elif outcome == streak_type:
                    streak += 1
                else:
                    break

            recent_summary = []
            for t in recent_3:
                outcome = 'W' if (t.pnl or 0) > 0 else 'L'
                sym = (t.symbol or '').replace('USDT', '').replace('/USDT:USDT', '')
                pnl_pct = t.pnl_percent or 0
                recent_summary.append(f"{outcome}({sym} {pnl_pct:+.1f}%)")

            lines = ["\n--- LIVE SYSTEM PERFORMANCE CONTEXT ---"]
            lines.append(f"  Today: {today_total} trades closed, P&L {today_pnl:+.2f} USDT ({today_wins}/{today_total} wins)")
            lines.append(f"  Last 7 days win rate: {week_win_rate:.0f}% ({week_wins}/{week_total} trades)")
            if recent_summary:
                lines.append(f"  Last 3 signals: {' | '.join(recent_summary)}")
            if streak_type and streak >= 2:
                lines.append(f"  Current streak: {streak}x {streak_type} — {'maintain discipline, avoid overconfidence' if streak_type == 'WIN' else 'be extra selective, tighten criteria'}")
            lines.append(f"  Currently open positions: {open_count}")
            lines.append("  Use this context to calibrate confidence — after losses, require stronger confirmation; after wins, guard against overconfidence.")
            lines.append("---")

            result = '\n'.join(lines)
            _live_context_cache = result
            _live_context_cache_time = now
            return result

        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to fetch live trading context: {e}")
        return ""


def update_lesson_effectiveness(trade) -> None:
    try:
        if not trade.opened_at:
            return

        outcome = 'WIN' if (trade.pnl or 0) > 0 else 'LOSS'

        from app.database import SessionLocal
        from app.models import TradeLesson

        db = SessionLocal()
        try:
            cutoff = trade.opened_at - timedelta(days=30)
            lessons = db.query(TradeLesson).filter(
                TradeLesson.created_at >= cutoff,
                TradeLesson.created_at < trade.opened_at
            ).all()

            if not lessons:
                return

            for lesson in lessons:
                lesson.times_applied = (lesson.times_applied or 0) + 1
                if outcome == 'WIN':
                    lesson.wins_after = (lesson.wins_after or 0) + 1
                else:
                    lesson.losses_after = (lesson.losses_after or 0) + 1

            db.commit()

            global _lessons_cache, _lessons_cache_time
            _lessons_cache = []
            _lessons_cache_time = None

            logger.info(f"Updated effectiveness for {len(lessons)} lessons after {trade.symbol} {outcome}")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to update lesson effectiveness: {e}")


async def extract_lesson_from_trade(trade) -> Optional[Dict]:
    client = _get_gemini_client()
    if not client:
        return None

    ticker = trade.symbol.replace('USDT', '').replace('/USDT:USDT', '')
    duration_min = 0
    if trade.opened_at and trade.closed_at:
        duration_min = int((trade.closed_at - trade.opened_at).total_seconds() / 60)

    outcome = "WIN" if trade.pnl > 0 else "LOSS"

    tp_hits = []
    if trade.tp1_hit:
        tp_hits.append("TP1")
    if trade.tp2_hit:
        tp_hits.append("TP2")
    if trade.tp3_hit:
        tp_hits.append("TP3")

    sl_pct = 0
    tp_pct = 0
    if trade.entry_price and trade.stop_loss:
        sl_pct = abs(trade.stop_loss - trade.entry_price) / trade.entry_price * 100
    if trade.entry_price and trade.take_profit_1:
        tp_pct = abs(trade.take_profit_1 - trade.entry_price) / trade.entry_price * 100

    prompt = f"""You are analyzing a closed crypto futures trade to extract a reusable trading lesson.

TRADE DATA:
- Coin: ${ticker}
- Direction: {trade.direction}
- Type: {getattr(trade, 'trade_type', 'STANDARD')}
- Leverage: {trade.leverage or 'N/A'}x
- Entry: ${trade.entry_price}
- Exit: ${trade.exit_price}
- TP: {tp_pct:.1f}% | SL: {sl_pct:.1f}%
- TPs Hit: {', '.join(tp_hits) if tp_hits else 'None'}
- P&L: {trade.pnl_percent:+.1f}%
- Peak ROI: {trade.peak_roi or 0:.1f}%
- Duration: {duration_min} minutes
- Result: {outcome}
- Breakeven Moved: {'Yes' if getattr(trade, 'breakeven_moved', None) else 'No'}

INSTRUCTIONS:
Extract a concise, actionable lesson from this trade. Focus on PATTERNS that can improve future trading decisions.

Respond in JSON:
{{
    "lesson": "One clear sentence about what this trade teaches us. Be specific about the setup/conditions.",
    "pattern_tags": ["tag1", "tag2"],
    "entry_timing": "EARLY/GOOD/LATE/POOR",
    "exit_quality": "OPTIMAL/GOOD/PREMATURE/LATE/STOPPED_OUT"
}}

Pattern tags should be from: early_entry, late_entry, overextended, momentum_fade, volume_confirmed, squeeze_play, trend_aligned, counter_trend, tight_sl, wide_sl, quick_scalp, swing_hold, breakeven_save, full_stop, tp1_runner, multi_tp"""

    try:
        import asyncio

        def _call():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"temperature": 0.3, "max_output_tokens": 200}
            )

        response = await asyncio.get_event_loop().run_in_executor(None, _call)
        result_text = response.text.strip()

        if "```json" in result_text:
            import re
            match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if match:
                result_text = match.group(1)

        first_brace = result_text.find("{")
        last_brace = result_text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            result_text = result_text[first_brace:last_brace + 1]

        result = json.loads(result_text)

        return {
            'lesson': result.get('lesson', ''),
            'pattern_tags': ','.join(result.get('pattern_tags', [])),
            'entry_timing': result.get('entry_timing', 'GOOD'),
            'exit_quality': result.get('exit_quality', 'GOOD'),
        }
    except Exception as e:
        logger.error(f"AI lesson extraction failed for {ticker}: {e}")
        return None


async def save_trade_lesson(trade):
    try:
        lesson_data = await extract_lesson_from_trade(trade)
        if not lesson_data or not lesson_data.get('lesson'):
            return

        from app.database import SessionLocal
        from app.models import TradeLesson

        db = SessionLocal()
        try:
            ticker = trade.symbol.replace('USDT', '').replace('/USDT:USDT', '')

            duration_min = 0
            if trade.opened_at and trade.closed_at:
                duration_min = int((trade.closed_at - trade.opened_at).total_seconds() / 60)

            sl_pct = 0
            tp_pct = 0
            if trade.entry_price and trade.stop_loss:
                sl_pct = abs(trade.stop_loss - trade.entry_price) / trade.entry_price * 100
            if trade.entry_price and trade.take_profit_1:
                tp_pct = abs(trade.take_profit_1 - trade.entry_price) / trade.entry_price * 100

            regime = None
            try:
                from app.services.ai_market_intelligence import _current_market_regime
                if _current_market_regime:
                    regime = _current_market_regime.get('regime', 'UNKNOWN')
            except Exception:
                pass

            lesson = TradeLesson(
                symbol=ticker,
                direction=trade.direction,
                trade_type=getattr(trade, 'trade_type', 'STANDARD'),
                outcome="WIN" if trade.pnl > 0 else "LOSS",
                pnl_percent=trade.pnl_percent or 0,
                leverage=trade.leverage,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                tp_percent=tp_pct,
                sl_percent=sl_pct,
                tp1_hit=trade.tp1_hit or False,
                tp2_hit=trade.tp2_hit or False,
                tp3_hit=trade.tp3_hit or False,
                duration_minutes=duration_min,
                market_regime=regime,
                lesson=lesson_data['lesson'],
                pattern_tags=lesson_data.get('pattern_tags', ''),
                times_applied=0,
                wins_after=0,
                losses_after=0,
            )
            db.add(lesson)
            db.commit()
            logger.info(f"Saved trade lesson for {ticker}: {lesson_data['lesson'][:80]}")

            global _lessons_cache, _lessons_cache_time
            _lessons_cache = []
            _lessons_cache_time = None

        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to save trade lesson: {e}")


def get_recent_lessons(trade_type: str = None, direction: str = None, symbol: str = None, limit: int = 10) -> List[Dict]:
    global _lessons_cache, _lessons_cache_time

    now = datetime.utcnow()
    if _lessons_cache and _lessons_cache_time and (now - _lessons_cache_time).total_seconds() < LESSONS_CACHE_TTL:
        filtered = _lessons_cache
    else:
        try:
            from app.database import SessionLocal
            from app.models import TradeLesson

            db = SessionLocal()
            try:
                cutoff = now - timedelta(days=30)
                lessons = db.query(TradeLesson).filter(
                    TradeLesson.created_at >= cutoff
                ).order_by(TradeLesson.created_at.desc()).limit(50).all()

                _lessons_cache = []
                for l in lessons:
                    times = l.times_applied or 0
                    wins = l.wins_after or 0
                    eff_score = (wins / times) if times >= 3 else None
                    _lessons_cache.append({
                        'symbol': l.symbol,
                        'direction': l.direction,
                        'trade_type': l.trade_type,
                        'outcome': l.outcome,
                        'pnl_percent': l.pnl_percent,
                        'lesson': l.lesson,
                        'pattern_tags': l.pattern_tags,
                        'market_regime': l.market_regime,
                        'duration_minutes': l.duration_minutes,
                        'tp1_hit': l.tp1_hit,
                        'times_applied': times,
                        'wins_after': wins,
                        'losses_after': l.losses_after or 0,
                        'effectiveness_score': eff_score,
                    })
                _lessons_cache_time = now
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to load trade lessons: {e}")
            return []
        filtered = _lessons_cache

    if trade_type:
        filtered = [l for l in filtered if l['trade_type'] == trade_type]
    if direction:
        filtered = [l for l in filtered if l['direction'] == direction]
    if symbol:
        clean = symbol.replace('USDT', '').replace('/USDT:USDT', '')
        filtered = [l for l in filtered if l['symbol'] == clean]

    def sort_key(l):
        eff = l.get('effectiveness_score')
        if eff is None:
            return 0.5
        return eff

    filtered = sorted(filtered, key=sort_key, reverse=True)

    poor_lessons = [l for l in filtered if l.get('effectiveness_score') is not None and l['effectiveness_score'] < 0.4]
    useful_lessons = [l for l in filtered if l not in poor_lessons]

    return useful_lessons[:limit]


def format_lessons_for_ai_prompt(trade_type: str = None, direction: str = None, symbol: str = None) -> str:
    lessons = get_recent_lessons(trade_type=trade_type, direction=direction, limit=8)
    if not lessons:
        return ""

    wins = [l for l in lessons if l['outcome'] == 'WIN']
    losses = [l for l in lessons if l['outcome'] == 'LOSS']
    win_rate = len(wins) / len(lessons) * 100 if lessons else 0

    lines = [f"\n--- LEARNED FROM PAST TRADES (last {len(lessons)} similar trades, {win_rate:.0f}% win rate) ---"]

    for l in lessons[:5]:
        icon = "W" if l['outcome'] == 'WIN' else "L"
        eff = l.get('effectiveness_score')
        if eff is not None:
            eff_str = f" [proven {eff*100:.0f}% effective]"
        else:
            eff_str = ""
        lines.append(f"  [{icon}] {l['symbol']} {l['direction']}: {l['lesson']}{eff_str}")

    loss_patterns = {}
    for l in losses:
        tags = (l.get('pattern_tags') or '').split(',')
        for tag in tags:
            tag = tag.strip()
            if tag:
                loss_patterns[tag] = loss_patterns.get(tag, 0) + 1

    if loss_patterns:
        top_loss = sorted(loss_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        lines.append(f"  Common loss patterns: {', '.join(f'{t[0]}({t[1]}x)' for t in top_loss)}")

    lines.append("  Use these lessons to avoid repeating mistakes and favor setups that have been winning.")

    return '\n'.join(lines)
