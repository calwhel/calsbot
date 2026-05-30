"""
TradehubStrategyBot — dedicated forex onboarding chatbot (@TradehubStrategyBot).

Each user gets a private questionnaire flow. Their answers are forwarded to the
admin (@bu11dogg) with Approve / Deny / Reply buttons. The admin can reply to any
user individually through the bot — replies are routed back to that specific user
only, never broadcast to everyone.

Flow:
  /start → Q1 (signed up?) → Q2 (FP Markets?) → Q3 (cTrader connected?)
         → Q4 (name?) → Q5 (deposit amount?) → submit → admin notified

Admin commands (only works in DM with admin):
  Inline [✅ Approve] [❌ Deny] [💬 Reply] buttons on each submission card.
  [💬 Reply] → bot asks admin for message → routes it back to that user only.
"""

import asyncio
import logging
import os
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import (
    InlineKeyboardMarkup, InlineKeyboardButton,
    CallbackQuery, ReplyKeyboardMarkup, KeyboardButton,
    ReplyKeyboardRemove,
)
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

logger = logging.getLogger(__name__)

FOREX_BOT_TOKEN = os.getenv("FOREX_BOT_TOKEN", "")
forex_bot = Bot(token=FOREX_BOT_TOKEN) if FOREX_BOT_TOKEN else None
forex_dp  = Dispatcher(storage=MemoryStorage())

PORTAL_URL = "https://tradehubmarkets.com/app"
FP_LINK    = "https://www.fpmarkets.com/?rfrr=IB-Portal&cxd=37182_638734"

# ── FSM states ────────────────────────────────────────────────────────────────
class Onboard(StatesGroup):
    q1_fp_markets    = State()   # Opened FP Markets Standard account?
    q2_ctrader       = State()   # Connected cTrader in portal?
    q3_name          = State()   # Full name

class AdminReply(StatesGroup):
    waiting_for_message = State()  # Admin typing a reply to a specific user

# ── Helpers ───────────────────────────────────────────────────────────────────
def _yn_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="✅ Yes"), KeyboardButton(text="❌ No")]],
        resize_keyboard=True, one_time_keyboard=True,
    )

def _remove_keyboard():
    return ReplyKeyboardRemove()

def _is_yes(text: str) -> bool:
    return text.strip().lower() in ("yes", "✅ yes", "✅", "y", "yeah", "yep", "yup")

def _is_no(text: str) -> bool:
    return text.strip().lower() in ("no", "❌ no", "❌", "n", "nope", "nah")

def _admin_id() -> str:
    from app.config import settings as _s
    return str(getattr(_s, "OWNER_TELEGRAM_ID", ""))

async def _safe_answer(cb: CallbackQuery, txt: str = None):
    try:
        await cb.answer(text=txt)
    except Exception:
        pass

async def _notify_admin(user_id: int, tg_id: int, answers: dict, name: str, uname: str):
    """Send admin a summary card with Approve / Deny / Reply buttons."""
    admin_chat = _admin_id()
    if not admin_chat or not forex_bot:
        return
    mention = f"@{uname}" if uname else f"ID {tg_id}"
    lines = [
        f"<b>📋 New Forex Onboarding</b>",
        f"",
        f"<b>Name:</b> {name}",
        f"<b>Telegram:</b> {mention}",
        f"<b>TG ID:</b> <code>{tg_id}</code>",
        f"<b>DB uid:</b> <code>{user_id or '—'}</code>",
        f"",
        f"<b>Answers:</b>",
        f"• FP Markets acct: {answers.get('fp_markets','—')}",
        f"• cTrader linked:  {answers.get('ctrader','—')}",
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Approve", callback_data=f"fxapprove:{user_id}:{tg_id}"),
        InlineKeyboardButton(text="❌ Deny",    callback_data=f"fxdeny:{user_id}:{tg_id}"),
        InlineKeyboardButton(text="💬 Reply",   callback_data=f"fxreply:{tg_id}"),
    ]])
    try:
        await forex_bot.send_message(
            chat_id=int(admin_chat),
            text="\n".join(lines),
            parse_mode="HTML",
            reply_markup=keyboard,
        )
    except Exception as e:
        logger.warning(f"[forex_bot] admin notify failed: {e}")


# ── /start — welcome ─────────────────────────────────────────────────────────
@forex_dp.message(Command("start"))
async def fx_start(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "👋 <b>Welcome to TradeHub Strategy!</b>\n\n"
        "We help traders of all levels build, test, and automate trading strategies — "
        "no coding required.\n\n"
        "📊 <b>What you get:</b>\n"
        "• Build strategies with our AI wizard in minutes\n"
        "• Copy the best-performing strategies for <b>free</b>\n"
        "• Paper trade risk-free, then go live with one tap\n"
        "• Automated execution across crypto &amp; forex\n"
        "• Full Telegram alerts on every trade\n\n"
        f"🌐 <a href='{PORTAL_URL}'>tradehubmarkets.com</a>\n\n"
        "─────────────────────────\n\n"
        "To get started with <b>live forex &amp; gold trading</b> through FP Markets, "
        "I'll ask you a few quick questions.\n\n"
        "<b>Have you already opened a Standard account with FP Markets?</b>\n\n"
        f"If not, open one here first 👉 <a href='{FP_LINK}'>FP Markets — Standard account</a>\n"
        "<i>(Using our link is required for the integration to work)</i>",
        parse_mode="HTML",
        reply_markup=_yn_keyboard(),
        disable_web_page_preview=True,
    )
    await state.set_state(Onboard.q1_fp_markets)


# ── Q1: FP Markets account? ───────────────────────────────────────────────────
@forex_dp.message(Onboard.q1_fp_markets)
async def fx_q1(message: types.Message, state: FSMContext):
    if _is_no(message.text or ""):
        await state.clear()
        await message.answer(
            "No problem — open your free Standard account here:\n"
            f"👉 <a href='{FP_LINK}'>FP Markets — Standard account</a>\n\n"
            "Once your account is open, come back and tap /start to continue.",
            parse_mode="HTML",
            reply_markup=_remove_keyboard(),
            disable_web_page_preview=True,
        )
        return
    if not _is_yes(message.text or ""):
        await message.answer("Please tap ✅ Yes or ❌ No.", reply_markup=_yn_keyboard())
        return

    await state.update_data(fp_markets="Yes ✅")
    await message.answer(
        "<b>Have you connected your cTrader account in the Live Forex tab?</b>\n\n"
        f"Go to <a href='{PORTAL_URL}'>{PORTAL_URL}</a> → <b>Live Forex</b> tab → Step 3 to connect.",
        parse_mode="HTML",
        reply_markup=_yn_keyboard(),
        disable_web_page_preview=True,
    )
    await state.set_state(Onboard.q2_ctrader)


# ── Q2: cTrader connected? ────────────────────────────────────────────────────
@forex_dp.message(Onboard.q2_ctrader)
async def fx_q2(message: types.Message, state: FSMContext):
    if _is_no(message.text or ""):
        await state.clear()
        await message.answer(
            "No problem! Here's how to connect:\n\n"
            f"1. Go to <a href='{PORTAL_URL}'>{PORTAL_URL}</a>\n"
            "2. Click the <b>Live Forex</b> tab\n"
            "3. Follow Step 3 — it takes about 30 seconds\n\n"
            "Once connected, come back and tap /start to continue.",
            parse_mode="HTML",
            reply_markup=_remove_keyboard(),
            disable_web_page_preview=True,
        )
        return
    if not _is_yes(message.text or ""):
        await message.answer("Please tap ✅ Yes or ❌ No.", reply_markup=_yn_keyboard())
        return

    await state.update_data(ctrader="Yes ✅")
    await message.answer(
        "<b>What's your full name?</b>",
        parse_mode="HTML",
        reply_markup=_remove_keyboard(),
    )
    await state.set_state(Onboard.q3_name)


# ── Q3: Name / submit ─────────────────────────────────────────────────────────
@forex_dp.message(Onboard.q3_name)
async def fx_q3(message: types.Message, state: FSMContext):
    name = (message.text or "").strip()
    if len(name) < 2:
        await message.answer("Please enter your full name.")
        return

    data = await state.get_data()
    data["name"] = name
    uname  = message.from_user.username or ""
    tg_id  = message.from_user.id

    # Look up DB user
    user_id = None
    try:
        from app.database import SessionLocal
        from app.models import User
        db = SessionLocal()
        u = db.query(User).filter(User.telegram_id == str(tg_id)).first()
        user_id = u.id if u else None
        db.close()
    except Exception:
        pass

    await message.answer(
        f"✅ <b>Thanks, {name}!</b>\n\n"
        "Your request has been sent to the TradeHub team. "
        "You'll receive a message here once you're approved — usually within a few hours.\n\n"
        "<i>Sit tight and we'll be in touch shortly! 🚀</i>",
        parse_mode="HTML",
    )
    await state.clear()

    await _notify_admin(user_id, tg_id, data, name, uname)


# ── Admin: Approve ────────────────────────────────────────────────────────────
@forex_dp.callback_query(F.data.startswith("fxapprove:"))
async def fx_approve(cb: CallbackQuery):
    await _safe_answer(cb)
    if str(cb.from_user.id) != _admin_id():
        await cb.answer("⛔ Not authorised.", show_alert=True)
        return

    parts = cb.data.split(":")          # fxapprove:user_id:tg_id
    db_user_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    tg_id      = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None

    # Flip DB flag
    if db_user_id:
        try:
            from app.database import SessionLocal
            from app.models import UserPreference
            db = SessionLocal()
            prefs = db.query(UserPreference).filter(UserPreference.user_id == db_user_id).first()
            if not prefs:
                prefs = UserPreference(user_id=db_user_id)
                db.add(prefs)
            prefs.forex_approved = True
            db.commit()
            db.close()
        except Exception as e:
            logger.warning(f"[fxapprove] DB error: {e}")

    # Edit admin card
    try:
        await cb.message.edit_text(
            cb.message.text + "\n\n✅ <b>Approved</b>",
            parse_mode="HTML",
        )
    except Exception:
        pass

    # DM the user
    if tg_id and forex_bot:
        try:
            await forex_bot.send_message(
                chat_id=tg_id,
                text=(
                    "✅ <b>You're approved for Live Forex trading!</b>\n\n"
                    "Your cTrader account is now connected and authorised.\n\n"
                    f"Head to the <b>Live Forex</b> tab at {PORTAL_URL} and set any forex, "
                    "gold, or index strategy to <b>Live</b> — your trades will execute "
                    "automatically on your FP Markets account. 🚀"
                ),
                parse_mode="HTML",
            )
        except Exception as e:
            logger.warning(f"[fxapprove] DM failed tg_id={tg_id}: {e}")


# ── Admin: Deny ───────────────────────────────────────────────────────────────
@forex_dp.callback_query(F.data.startswith("fxdeny:"))
async def fx_deny(cb: CallbackQuery):
    await _safe_answer(cb)
    if str(cb.from_user.id) != _admin_id():
        await cb.answer("⛔ Not authorised.", show_alert=True)
        return

    parts = cb.data.split(":")
    db_user_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    tg_id      = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None

    if db_user_id:
        try:
            from app.database import SessionLocal
            from app.models import UserPreference
            db = SessionLocal()
            prefs = db.query(UserPreference).filter(UserPreference.user_id == db_user_id).first()
            if prefs:
                prefs.forex_approved = False
                db.commit()
            db.close()
        except Exception as e:
            logger.warning(f"[fxdeny] DB error: {e}")

    try:
        await cb.message.edit_text(
            cb.message.text + "\n\n❌ <b>Denied</b>",
            parse_mode="HTML",
        )
    except Exception:
        pass

    if tg_id and forex_bot:
        try:
            await forex_bot.send_message(
                chat_id=tg_id,
                text=(
                    "❌ <b>Not approved yet.</b>\n\n"
                    "Please make sure you opened your FP Markets account through our affiliate link — "
                    "this is required for the integration to work.\n\n"
                    f"👉 <a href='{FP_LINK}'>Open FP Markets via our link</a>\n\n"
                    "Once done, reconnect cTrader in the Live Forex tab and tap /start to go through the steps again."
                ),
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as e:
            logger.warning(f"[fxdeny] DM failed tg_id={tg_id}: {e}")


# ── Admin: Reply (step 1 — click button) ─────────────────────────────────────
@forex_dp.callback_query(F.data.startswith("fxreply:"))
async def fx_reply_init(cb: CallbackQuery, state: FSMContext):
    await _safe_answer(cb)
    if str(cb.from_user.id) != _admin_id():
        await cb.answer("⛔ Not authorised.", show_alert=True)
        return

    tg_id = int(cb.data.split(":")[1])
    await state.update_data(reply_to_tg_id=tg_id)
    await state.set_state(AdminReply.waiting_for_message)
    await cb.message.answer(
        f"✏️ <b>Type your reply to user <code>{tg_id}</code></b>\n\n"
        "Send any message and it will be delivered to them privately.\n"
        "Send /cancel to cancel.",
        parse_mode="HTML",
    )


# ── Admin: Reply (step 2 — send the message) ─────────────────────────────────
@forex_dp.message(AdminReply.waiting_for_message)
async def fx_reply_send(message: types.Message, state: FSMContext):
    if str(message.from_user.id) != _admin_id():
        return
    if message.text and message.text.strip() == "/cancel":
        await state.clear()
        await message.answer("Cancelled.", reply_markup=_remove_keyboard())
        return

    data = await state.get_data()
    tg_id = data.get("reply_to_tg_id")
    if not tg_id or not forex_bot:
        await state.clear()
        await message.answer("❌ Could not find the user to reply to.")
        return

    try:
        await forex_bot.send_message(
            chat_id=tg_id,
            text=f"💬 <b>Message from TradeHub Strategy:</b>\n\n{message.text}",
            parse_mode="HTML",
        )
        await message.answer(f"✅ Reply sent to user <code>{tg_id}</code>.", parse_mode="HTML")
    except Exception as e:
        await message.answer(f"❌ Failed to send: {e}")

    await state.clear()


# ── Start function (called from main.py) ──────────────────────────────────────
async def start_forex_bot():
    if not FOREX_BOT_TOKEN:
        logger.warning("[forex_bot] FOREX_BOT_TOKEN not set — skipping")
        return
    logger.info("[forex_bot] Starting @TradehubStrategyBot polling…")
    try:
        await forex_bot.delete_webhook(drop_pending_updates=True)
        from aiogram.types import BotCommand
        await forex_bot.set_my_commands([
            BotCommand(command="start", description="Start onboarding for live forex trading"),
        ])
    except Exception as e:
        logger.warning(f"[forex_bot] setup: {e}")
    await forex_dp.start_polling(forex_bot)
