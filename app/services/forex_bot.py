"""
TradehubStrategyBot — forex onboarding + two-way support proxy.

Onboarding flow (per user, private):
  /start → welcome + Q1 (FP Markets?) → Q2 (cTrader linked?) → submit → admin notified

After onboarding (or any time), users can send free-form messages.
Every user message is forwarded to admin with a header showing who sent it.
Admin replies to any forwarded message → reply is routed back to that user only.

This gives the admin a full conversation thread per user inside the bot chat —
no commands, no FSM states needed for the two-way chat part.
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

# ── In-memory map: admin_message_id → user_tg_id ─────────────────────────────
# Lets admin reply to any forwarded message and have it routed back automatically.
_REPLY_MAP: dict[int, int] = {}

# ── FSM states (onboarding only) ──────────────────────────────────────────────
class Onboard(StatesGroup):
    q1_fp_markets = State()   # Opened FP Markets Standard account?
    q2_ctrader    = State()   # Connected cTrader in portal?
    q3_name       = State()   # Full name

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

def _is_admin(user_id: int) -> bool:
    return str(user_id) == _admin_id()


async def _forward_to_admin(user: types.User, text: str, label: str = "💬 User message") -> int | None:
    """Forward a user message to admin. Returns the admin message_id for reply routing."""
    admin_chat = _admin_id()
    if not admin_chat or not forex_bot:
        return None
    uname   = f"@{user.username}" if user.username else f"ID {user.id}"
    name    = " ".join(filter(None, [user.first_name, user.last_name])) or uname
    header  = f"{label}\n<b>{name}</b> ({uname}) · <code>{user.id}</code>\n\n"
    try:
        sent = await forex_bot.send_message(
            chat_id=int(admin_chat),
            text=header + text,
            parse_mode="HTML",
        )
        return sent.message_id
    except Exception as e:
        logger.warning(f"[forex_bot] forward_to_admin failed: {e}")
        return None


async def _notify_admin_submission(user_id_db, tg_user: types.User, answers: dict) -> int | None:
    """Send admin the onboarding summary card with Approve / Deny buttons."""
    admin_chat = _admin_id()
    if not admin_chat or not forex_bot:
        return None
    uname        = f"@{tg_user.username}" if tg_user.username else f"ID {tg_user.id}"
    entered_name = answers.get("name") or ""
    tg_name      = " ".join(filter(None, [tg_user.first_name, tg_user.last_name]))
    display_name = entered_name or tg_name or uname
    lines = [
        "📋 <b>New Forex Onboarding Request</b>",
        "",
        f"<b>Name:</b> {display_name}",
        f"<b>Telegram:</b> {uname}",
        f"<b>TG ID:</b> <code>{tg_user.id}</code>",
        f"<b>DB uid:</b> <code>{user_id_db or '—'}</code>",
        "",
        "<b>Answers:</b>",
        f"• FP Markets acct: {answers.get('fp_markets','—')}",
        f"• cTrader linked:  {answers.get('ctrader','—')}",
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Approve", callback_data=f"fxapprove:{user_id_db}:{tg_user.id}"),
        InlineKeyboardButton(text="❌ Deny",    callback_data=f"fxdeny:{user_id_db}:{tg_user.id}"),
    ]])
    try:
        sent = await forex_bot.send_message(
            chat_id=int(admin_chat),
            text="\n".join(lines),
            parse_mode="HTML",
            reply_markup=keyboard,
        )
        return sent.message_id
    except Exception as e:
        logger.warning(f"[forex_bot] notify_admin_submission failed: {e}")
        return None


# ── /start ────────────────────────────────────────────────────────────────────
@forex_dp.message(Command("start"))
async def fx_start(message: types.Message, state: FSMContext):
    if _is_admin(message.from_user.id):
        await message.answer(
            "👋 <b>Admin mode.</b>\n\n"
            "Reply to any forwarded user message in this chat to send them a private response.\n"
            "Use the ✅ Approve / ❌ Deny buttons on submission cards to manage access.",
            parse_mode="HTML",
        )
        return

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
        "To enable <b>live forex &amp; gold trading</b> through FP Markets, "
        "I just need to check a couple of things.\n\n"
        "<b>Have you already opened a Standard account with FP Markets?</b>\n\n"
        f"If not, open one here first 👉 <a href='{FP_LINK}'>FP Markets — Standard account</a>\n"
        "<i>(Using our link is required for the integration to work)</i>",
        parse_mode="HTML",
        reply_markup=_yn_keyboard(),
        disable_web_page_preview=True,
    )
    await state.set_state(Onboard.q1_fp_markets)


# ── Q1: FP Markets? ───────────────────────────────────────────────────────────
@forex_dp.message(Onboard.q1_fp_markets)
async def fx_q1(message: types.Message, state: FSMContext):
    if _is_no(message.text or ""):
        await state.clear()
        await message.answer(
            "No problem — open your free Standard account here:\n"
            f"👉 <a href='{FP_LINK}'>FP Markets — Standard account</a>\n\n"
            "Once your account is open, come back and tap /start to continue.\n\n"
            "Feel free to message me any questions in the meantime!",
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


# ── Q2: cTrader? ──────────────────────────────────────────────────────────────
@forex_dp.message(Onboard.q2_ctrader)
async def fx_q2(message: types.Message, state: FSMContext):
    if _is_no(message.text or ""):
        await state.clear()
        await message.answer(
            "No problem! Here's how to connect:\n\n"
            f"1. Go to <a href='{PORTAL_URL}'>{PORTAL_URL}</a>\n"
            "2. Click the <b>Live Forex</b> tab\n"
            "3. Follow Step 3 — it takes about 30 seconds\n\n"
            "Come back and tap /start once it's connected.\n\n"
            "Feel free to message me any questions in the meantime!",
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


# ── Q3: Name → submit ─────────────────────────────────────────────────────────
@forex_dp.message(Onboard.q3_name)
async def fx_q3(message: types.Message, state: FSMContext):
    name = (message.text or "").strip()
    if len(name) < 2:
        await message.answer("Please enter your full name.")
        return

    data = await state.get_data()
    data["name"] = name
    tg_user = message.from_user
    await state.clear()

    # Look up DB user
    user_id_db = None
    try:
        from app.database import SessionLocal
        from app.models import User
        db = SessionLocal()
        u = db.query(User).filter(User.telegram_id == str(tg_user.id)).first()
        user_id_db = u.id if u else None
        db.close()
    except Exception:
        pass

    await message.answer(
        f"✅ <b>Thanks, {name}!</b>\n\n"
        "Your request has been sent to the TradeHub team. "
        "You'll receive a message here once you're approved — usually within a few hours.\n\n"
        "<i>Feel free to send a message if you have any questions!</i>",
        parse_mode="HTML",
    )

    msg_id = await _notify_admin_submission(user_id_db, tg_user, data)
    if msg_id:
        _REPLY_MAP[msg_id] = tg_user.id


# ── Admin: Approve ────────────────────────────────────────────────────────────
@forex_dp.callback_query(F.data.startswith("fxapprove:"))
async def fx_approve(cb: CallbackQuery):
    try:
        await cb.answer()
    except Exception:
        pass
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔ Not authorised.", show_alert=True)
        return

    parts      = cb.data.split(":")
    db_user_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    tg_id      = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None

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

    try:
        await cb.message.edit_reply_markup(reply_markup=None)
        await cb.message.edit_text(cb.message.text + "\n\n✅ Approved", parse_mode="HTML")
    except Exception:
        pass

    if tg_id and forex_bot:
        try:
            sent = await forex_bot.send_message(
                chat_id=tg_id,
                text=(
                    "✅ <b>You're approved for Live Forex trading!</b>\n\n"
                    "Your cTrader account is now authorised. Head to the <b>Live Forex</b> tab "
                    f"at <a href='{PORTAL_URL}'>{PORTAL_URL}</a> and set any forex, gold, or index "
                    "strategy to <b>Live</b> — trades will execute automatically on your FP Markets account. 🚀\n\n"
                    "Message me here any time if you need help."
                ),
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            _REPLY_MAP[sent.message_id] = tg_id
        except Exception as e:
            logger.warning(f"[fxapprove] DM failed tg_id={tg_id}: {e}")


# ── Admin: Deny ───────────────────────────────────────────────────────────────
@forex_dp.callback_query(F.data.startswith("fxdeny:"))
async def fx_deny(cb: CallbackQuery):
    try:
        await cb.answer()
    except Exception:
        pass
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔ Not authorised.", show_alert=True)
        return

    parts      = cb.data.split(":")
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
        await cb.message.edit_reply_markup(reply_markup=None)
        await cb.message.edit_text(cb.message.text + "\n\n❌ Denied", parse_mode="HTML")
    except Exception:
        pass

    if tg_id and forex_bot:
        try:
            sent = await forex_bot.send_message(
                chat_id=tg_id,
                text=(
                    "❌ <b>Not approved yet.</b>\n\n"
                    "Please make sure you opened your FP Markets account through our affiliate link — "
                    "this is required for the integration to work.\n\n"
                    f"👉 <a href='{FP_LINK}'>Open FP Markets via our link</a>\n\n"
                    "Once done, reconnect cTrader in the Live Forex tab and tap /start to try again.\n"
                    "Message me here if you have any questions!"
                ),
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            _REPLY_MAP[sent.message_id] = tg_id
        except Exception as e:
            logger.warning(f"[fxdeny] DM failed tg_id={tg_id}: {e}")


# ── Admin: reply to forwarded message → route to user ────────────────────────
@forex_dp.message(F.reply_to_message)
async def fx_admin_reply(message: types.Message):
    if not _is_admin(message.from_user.id):
        return  # only admin replies are routed
    replied_id = message.reply_to_message.message_id
    user_tg_id = _REPLY_MAP.get(replied_id)
    if not user_tg_id or not forex_bot:
        # Admin replied to something we don't have a mapping for — ignore silently
        return
    try:
        sent = await forex_bot.send_message(
            chat_id=user_tg_id,
            text=f"💬 <b>TradeHub Strategy:</b>\n\n{message.text or message.caption or ''}",
            parse_mode="HTML",
        )
        # Map the confirmation message so admin can continue the thread
        _REPLY_MAP[sent.message_id] = user_tg_id
        await message.reply("✅ Sent", parse_mode="HTML")
    except Exception as e:
        await message.reply(f"❌ Failed: {e}")


# ── User: any free-form message → forward to admin ───────────────────────────
@forex_dp.message()
async def fx_user_message(message: types.Message, state: FSMContext):
    # Skip if admin (they're managing, not chatting as user)
    if _is_admin(message.from_user.id):
        return
    # Skip if user is mid-questionnaire (FSM handlers take priority above)
    current = await state.get_state()
    if current is not None:
        return

    text = message.text or message.caption or "[non-text message]"
    msg_id = await _forward_to_admin(message.from_user, text)
    if msg_id:
        _REPLY_MAP[msg_id] = message.from_user.id

    # Acknowledge to user so they know it was received
    await message.answer(
        "📨 Message received — we'll get back to you shortly!",
        parse_mode="HTML",
    )


# ── Start function (called from main.py) ──────────────────────────────────────
async def start_forex_bot():
    if not FOREX_BOT_TOKEN:
        logger.warning("[forex_bot] FOREX_BOT_TOKEN not set — skipping")
        return
    _main = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if _main and FOREX_BOT_TOKEN.strip() == _main:
        logger.warning(
            "[forex_bot] FOREX_BOT_TOKEN equals TELEGRAM_BOT_TOKEN — "
            "not starting a second poller (use main bot only)"
        )
        return
    logger.info("[forex_bot] Starting forex support bot polling…")
    try:
        await forex_bot.delete_webhook(drop_pending_updates=True)
        from aiogram.types import BotCommand
        await forex_bot.set_my_commands([
            BotCommand(command="start", description="Get started with live forex trading"),
        ])
    except Exception as e:
        logger.warning(f"[forex_bot] setup: {e}")
    await forex_dp.start_polling(forex_bot)
