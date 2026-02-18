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


async def review_closed_trade(trade) -> Optional[str]:
    client = _get_gemini_client()
    if not client:
        logger.warning("No Gemini client available for trade review")
        return None

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

    prompt = f"""You are an expert crypto futures trade analyst. Review this closed trade and provide a concise analysis.

TRADE DATA:
- Coin: ${ticker}
- Direction: {trade.direction}
- Type: {trade.trade_type or 'STANDARD'}
- Leverage: {trade.leverage or 'N/A'}x
- Entry: ${trade.entry_price}
- Exit: ${trade.exit_price}
- Stop Loss: {sl_line}
- TP1: {tp1_line}
- TP2: ${trade.take_profit_2 or 'N/A'}
- TP3: ${trade.take_profit_3 or 'N/A'}
- TPs Hit: {tp_summary}
- Breakeven Moved: {'Yes' if getattr(trade, 'breakeven_moved', None) else 'No'}
- Position Size: ${trade.position_size:.2f}
- P&L: ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%)
- Peak ROI: {peak_roi_str}
- Duration: {duration or 'N/A'}
- Result: {trade.status.upper()}

INSTRUCTIONS:
Give a brief trade review in this exact format (keep each section to 1-2 sentences max):

ENTRY: Was the entry well-timed? Any observations on the entry price.
EXIT: How was the exit managed? Could it have been better?
RISK: Was the SL placement reasonable? Comment on risk/reward ratio.
VERDICT: One sentence overall assessment and one specific improvement suggestion.

Keep it direct and actionable. No fluff. Use trader language."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        review_text = response.text.strip() if response.text else None
        return review_text
    except Exception as e:
        logger.error(f"AI trade review failed: {e}")
        return None


async def send_trade_review_to_admin(trade, bot):
    from app.config import settings

    if not settings.OWNER_TELEGRAM_ID:
        return

    try:
        review = await review_closed_trade(trade)
        if not review:
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

        msg = (
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"  {result_icon} <b>AI TRADE REVIEW</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"\n"
            f"<b>${ticker}</b> {direction} {trade.leverage or ''}x\n"
            f"P&L: <b>{'+' if trade.pnl >= 0 else ''}${trade.pnl:.2f}</b> ({trade.pnl_percent:+.1f}%){tp_status}\n"
            f"\n"
            f"{review}"
        )

        await bot.send_message(
            settings.OWNER_TELEGRAM_ID,
            msg,
            parse_mode="HTML"
        )
        logger.info(f"AI trade review sent to admin for ${ticker} {direction}")
    except Exception as e:
        logger.error(f"Failed to send trade review to admin: {e}")
