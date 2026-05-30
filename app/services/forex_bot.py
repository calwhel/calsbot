"""
TradehubStrategyBot — dedicated forex approval bot (@TradehubStrategyBot).
Completely separate from the main signal bot so users see a clean
"TradeHub Strategy · bot" identity with no member/subscriber counts.

Handles:
  /start  — welcome + instructions
  /forex  — request live forex trading access (notifies admin w/ inline buttons)
  forex_approve:{user_id}  — admin approves (flips DB flag, DMs user)
  forex_deny:{user_id}     — admin denies  (flips DB flag, DMs user)
"""

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.fsm.storage.memory import MemoryStorage

logger = logging.getLogger(__name__)

FOREX_BOT_TOKEN = os.getenv("FOREX_BOT_TOKEN", "")

forex_bot  = Bot(token=FOREX_BOT_TOKEN) if FOREX_BOT_TOKEN else None
forex_dp   = Dispatcher(storage=MemoryStorage())

PORTAL_URL = "https://tradehubmarkets.com/app"
FP_LINK    = "https://www.fpmarkets.com/?rfrr=IB-Portal&cxd=37182_638734"


async def _safe_answer(callback: CallbackQuery, text: str = None):
    try:
        await callback.answer(text=text)
    except Exception:
        pass


# ─── /start ───────────────────────────────────────────────────────────────────
@forex_dp.message(Command("start"))
async def fx_start(message: types.Message):
    await message.answer(
        "👋 <b>Welcome to TradeHub Strategy</b>\n\n"
        "This is your gateway to <b>live forex & indices trading</b> through TradeHub.\n\n"
        "Here's how it works:\n"
        "1️⃣ Open a <b>Standard account</b> with FP Markets using our link\n"
        "2️⃣ Connect your cTrader account in the <b>Live Forex</b> tab at tradehubmarkets.com/app\n"
        "3️⃣ Send /forex here to request approval\n"
        "4️⃣ Once approved, your strategies go live automatically\n\n"
        f'<a href="{FP_LINK}">Open FP Markets account →</a>\n'
        f'<a href="{PORTAL_URL}">TradeHub portal →</a>',
        parse_mode="HTML",
        disable_web_page_preview=True,
    )


# ─── /forex ───────────────────────────────────────────────────────────────────
@forex_dp.message(Command("forex"))
async def fx_forex(message: types.Message):
    from app.database import SessionLocal
    from app.models import User, UserPreference
    from app.config import settings as _s

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer(
                "👋 You need a TradeHub account first.\n\n"
                f"Sign up at <b>{PORTAL_URL}</b>, link your Telegram account, then send /forex again.",
                parse_mode="HTML",
            )
            return

        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()

        if prefs and getattr(prefs, "forex_approved", False):
            await message.answer(
                "✅ <b>You're already approved for live forex trading!</b>\n\n"
                f"Open the <b>Live Forex</b> tab at {PORTAL_URL} to get started.",
                parse_mode="HTML",
            )
            return

        connected = bool(prefs and prefs.ctrader_access_token and prefs.ctrader_account_id)
        if not connected:
            await message.answer(
                "🔗 <b>Connect your cTrader account first.</b>\n\n"
                f"Go to {PORTAL_URL} → <b>Live Forex</b> tab → complete Steps 1–3, then send /forex again.",
                parse_mode="HTML",
            )
            return

        # Notify admin with approve/deny buttons
        admin_chat = getattr(_s, "OWNER_TELEGRAM_ID", None)
        uname = message.from_user.username or ""
        mention = f"@{uname}" if uname else f"ID {message.from_user.id}"
        full_name = (
            f"{message.from_user.first_name or ''} {message.from_user.last_name or ''}".strip()
            or "User"
        )
        acct_id = prefs.ctrader_account_id or "unknown"

        if admin_chat and forex_bot:
            keyboard = InlineKeyboardMarkup(inline_keyboard=[[
                InlineKeyboardButton(text="✅ Approve", callback_data=f"forex_approve:{user.id}"),
                InlineKeyboardButton(text="❌ Deny",    callback_data=f"forex_deny:{user.id}"),
            ]])
            try:
                await forex_bot.send_message(
                    chat_id=int(admin_chat),
                    text=(
                        f"<b>🔗 Forex Approval Request</b>\n\n"
                        f"User:         {full_name} ({mention})\n"
                        f"cTrader Acct: <code>{acct_id}</code>\n"
                        f"DB user_id:   <code>{user.id}</code>\n\n"
                        f"<i>User sent /forex requesting live trading access.</i>"
                    ),
                    parse_mode="HTML",
                    reply_markup=keyboard,
                )
            except Exception as _e:
                logger.warning(f"[forex_bot /forex] admin notify failed: {_e}")

        await message.answer(
            "✅ <b>Request sent!</b>\n\n"
            "The TradeHub team has been notified. You'll receive a message here once you're approved — usually within a few hours.\n\n"
            "<i>Make sure you opened your FP Markets account through our affiliate link, otherwise we can't verify the connection.</i>",
            parse_mode="HTML",
        )

    except Exception as e:
        logger.error(f"[forex_bot /forex] {e}")
        await message.answer("Something went wrong — please try again later.")
    finally:
        db.close()


# ─── Approve callback ──────────────────────────────────────────────────────────
@forex_dp.callback_query(F.data.startswith("forex_approve:"))
async def fx_approve(callback: CallbackQuery):
    await _safe_answer(callback)
    from app.config import settings as _s
    if str(callback.from_user.id) != str(getattr(_s, "OWNER_TELEGRAM_ID", "")):
        await callback.message.answer("⛔ Not authorised.")
        return

    target_user_id = int(callback.data.split(":", 1)[1])
    from app.database import SessionLocal
    from app.models import User, UserPreference
    db = SessionLocal()
    try:
        prefs = db.query(UserPreference).filter(UserPreference.user_id == target_user_id).first()
        if not prefs:
            prefs = UserPreference(user_id=target_user_id)
            db.add(prefs)
        prefs.forex_approved = True
        db.commit()

        # Edit the admin message to remove buttons
        try:
            await callback.message.edit_text(
                callback.message.text + "\n\n✅ <b>Approved.</b>",
                parse_mode="HTML",
            )
        except Exception:
            pass

        # DM the user from this bot
        user = db.query(User).filter(User.id == target_user_id).first()
        tg_id = getattr(user, "telegram_id", None) if user else None
        if tg_id and forex_bot:
            try:
                await forex_bot.send_message(
                    chat_id=int(tg_id),
                    text=(
                        "✅ <b>You're approved for Live Forex trading!</b>\n\n"
                        "Your cTrader account is now connected and authorised. "
                        f"Head to the <b>Live Forex</b> tab at {PORTAL_URL} to get started.\n\n"
                        "Set any forex, gold, or index strategy to <b>Live</b> and it will execute real trades on your FP Markets account. 🚀"
                    ),
                    parse_mode="HTML",
                )
            except Exception as _e:
                logger.warning(f"[forex_approve] DM failed for tg_id={tg_id}: {_e}")

    except Exception as e:
        logger.error(f"[forex_approve] {e}")
        await callback.message.answer(f"Error: {e}")
    finally:
        db.close()


# ─── Deny callback ────────────────────────────────────────────────────────────
@forex_dp.callback_query(F.data.startswith("forex_deny:"))
async def fx_deny(callback: CallbackQuery):
    await _safe_answer(callback)
    from app.config import settings as _s
    if str(callback.from_user.id) != str(getattr(_s, "OWNER_TELEGRAM_ID", "")):
        await callback.message.answer("⛔ Not authorised.")
        return

    target_user_id = int(callback.data.split(":", 1)[1])
    from app.database import SessionLocal
    from app.models import User, UserPreference
    db = SessionLocal()
    try:
        prefs = db.query(UserPreference).filter(UserPreference.user_id == target_user_id).first()
        if prefs:
            prefs.forex_approved = False
            db.commit()

        try:
            await callback.message.edit_text(
                callback.message.text + "\n\n❌ <b>Denied.</b>",
                parse_mode="HTML",
            )
        except Exception:
            pass

        user = db.query(User).filter(User.id == target_user_id).first()
        tg_id = getattr(user, "telegram_id", None) if user else None
        if tg_id and forex_bot:
            try:
                await forex_bot.send_message(
                    chat_id=int(tg_id),
                    text=(
                        "❌ <b>Forex access not approved yet.</b>\n\n"
                        "Please make sure you opened your FP Markets account through our affiliate link — "
                        "this is required for the integration to work.\n\n"
                        f'<a href="{FP_LINK}">Open FP Markets via our link →</a>\n\n'
                        "Once done, reconnect cTrader in the Live Forex tab and send /forex again."
                    ),
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
            except Exception as _e:
                logger.warning(f"[forex_deny] DM failed for tg_id={tg_id}: {_e}")

    except Exception as e:
        logger.error(f"[forex_deny] {e}")
        await callback.message.answer(f"Error: {e}")
    finally:
        db.close()


# ─── Start function (called from main.py) ─────────────────────────────────────
async def start_forex_bot():
    if not FOREX_BOT_TOKEN:
        logger.warning("[forex_bot] FOREX_BOT_TOKEN not set — skipping")
        return
    logger.info("[forex_bot] Starting @TradehubStrategyBot polling…")
    try:
        await forex_bot.delete_webhook(drop_pending_updates=True)
        from aiogram.types import BotCommand
        await forex_bot.set_my_commands([
            BotCommand(command="start", description="Welcome & instructions"),
            BotCommand(command="forex", description="Request live forex trading access"),
        ])
    except Exception as e:
        logger.warning(f"[forex_bot] setup: {e}")
    await forex_dp.start_polling(forex_bot)
