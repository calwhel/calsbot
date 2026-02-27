import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


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


def _get_claude_client():
    try:
        from anthropic import Anthropic
        api_key = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY")
        base_url = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_BASE_URL")
        if not api_key or not base_url:
            return None
        return Anthropic(api_key=api_key, base_url=base_url)
    except Exception as e:
        logger.error(f"Failed to initialize Claude client: {e}")
        return None


def _build_trade_data(trade) -> dict:
    ticker = trade.symbol.replace('USDT', '').replace('/USDT:USDT', '')

    duration = ""
    if trade.opened_at and trade.closed_at:
        delta = trade.closed_at - trade.opened_at
        hours = delta.total_seconds() / 3600
        if hours < 1:
            duration = f"{int(delta.total_seconds() / 60)} minutes"
        elif hours < 24:
            duration = f"{hours:.1f} hours"
        else:
            duration = f"{delta.days} days {int(hours % 24)}h"

    tp_hits = []
    if trade.tp1_hit:
        tp_hits.append("TP1")
    if trade.tp2_hit:
        tp_hits.append("TP2")
    if trade.tp3_hit:
        tp_hits.append("TP3")
    tp_summary = ", ".join(tp_hits) if tp_hits else "None"

    sl_line = "N/A"
    if trade.entry_price and trade.stop_loss:
        sl_pct = abs(trade.stop_loss - trade.entry_price) / trade.entry_price * 100
        sl_line = f"${trade.stop_loss} ({sl_pct:.2f}% from entry)"
    elif trade.stop_loss:
        sl_line = f"${trade.stop_loss}"

    tp1_line = "N/A"
    if trade.entry_price and trade.take_profit_1:
        tp1_pct = abs(trade.take_profit_1 - trade.entry_price) / trade.entry_price * 100
        tp1_line = f"${trade.take_profit_1} ({tp1_pct:.2f}% from entry)"
    elif trade.take_profit_1:
        tp1_line = f"${trade.take_profit_1}"

    peak_roi_str = f"{trade.peak_roi:.1f}%" if trade.peak_roi else "N/A"

    return {
        "ticker": ticker,
        "duration": duration or "N/A",
        "tp_summary": tp_summary,
        "sl_line": sl_line,
        "tp1_line": tp1_line,
        "peak_roi_str": peak_roi_str,
    }


def _build_prompt(trade, data: dict) -> str:
    return f"""You are an expert crypto futures trade analyst. Review this closed trade and provide a concise analysis.

TRADE DATA:
- Coin: ${data['ticker']}
- Direction: {trade.direction}
- Type: {trade.trade_type or 'STANDARD'}
- Leverage: {trade.leverage or 'N/A'}x
- Entry: ${trade.entry_price}
- Exit: ${trade.exit_price}
- Stop Loss: {data['sl_line']}
- TP1: {data['tp1_line']}
- TP2: ${trade.take_profit_2 or 'N/A'}
- TP3: ${trade.take_profit_3 or 'N/A'}
- TPs Hit: {data['tp_summary']}
- Breakeven Moved: {'Yes' if getattr(trade, 'breakeven_moved', None) else 'No'}
- Position Size: ${trade.position_size:.2f}
- P&L: ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%)
- Peak ROI: {data['peak_roi_str']}
- Duration: {data['duration']}
- Result: {trade.status.upper()}

INSTRUCTIONS:
Give a brief trade review in this exact format (keep each section to 1-2 sentences max):

ENTRY: Was the entry well-timed? Any observations on the entry price.
EXIT: How was the exit managed? Could it have been better?
RISK: Was the SL placement reasonable? Comment on risk/reward ratio.
VERDICT: One sentence overall assessment and one specific improvement suggestion.

Keep it direct and actionable. No fluff. Use trader language."""


async def review_with_gemini(trade) -> Optional[str]:
    client = _get_gemini_client()
    if not client:
        logger.warning("No Gemini client available for trade review")
        return None

    data = _build_trade_data(trade)
    prompt = _build_prompt(trade, data)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        review_text = response.text.strip() if response.text else None
        return review_text
    except Exception as e:
        logger.error(f"Gemini trade review failed: {e}")
        return None


async def review_with_claude(trade) -> Optional[str]:
    client = _get_claude_client()
    if not client:
        logger.warning("No Claude client available for trade review (integration not configured)")
        return None

    data = _build_trade_data(trade)
    prompt = _build_prompt(trade, data)

    try:
        import asyncio
        loop = asyncio.get_event_loop()
        message = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        review_text = message.content[0].text.strip() if message.content else None
        return review_text
    except Exception as e:
        logger.error(f"Claude trade review failed: {e}")
        return None


async def send_trade_review_to_admin(trade, bot):
    from app.config import settings

    if not settings.OWNER_TELEGRAM_ID:
        return

    # Only send review for the owner's own trades — not for other users' trades
    try:
        if str(trade.user.telegram_id) != str(settings.OWNER_TELEGRAM_ID):
            return
    except Exception:
        return

    try:
        import asyncio
        try:
            gemini_review = await review_with_gemini(trade)
        except Exception as e:
            logger.error(f"Gemini review exception: {e}")
            gemini_review = None

        if not gemini_review:
            logger.warning(f"No AI review available for trade {trade.id}")
            return

        ticker = trade.symbol.replace('USDT', '').replace('/USDT:USDT', '')
        result_icon = "✅" if trade.pnl > 0 else "❌"
        direction = trade.direction

        tp_status = ""
        if trade.tp3_hit:
            tp_status = " (TP3)"
        elif trade.tp2_hit:
            tp_status = " (TP2)"
        elif trade.tp1_hit:
            tp_status = " (TP1)"

        header = (
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"  {result_icon} <b>AI TRADE REVIEW</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"\n"
            f"<b>${ticker}</b> {direction} {trade.leverage or ''}x\n"
            f"P&L: <b>{'+' if trade.pnl >= 0 else ''}${trade.pnl:.2f}</b> ({trade.pnl_percent:+.1f}%){tp_status}\n"
        )

        review_msg = (
            f"{header}\n"
            f"{gemini_review}"
        )
        await bot.send_message(
            settings.OWNER_TELEGRAM_ID,
            review_msg,
            parse_mode="HTML"
        )
        logger.info(f"Trade review sent for ${ticker} {direction}")

    except Exception as e:
        logger.error(f"Failed to send trade review to admin for trade {getattr(trade, 'id', '?')}: {e}")
