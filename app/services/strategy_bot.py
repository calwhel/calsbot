"""
Strategy Builder Telegram Handlers — Build Your Own Strategy Portal

Registers /strategy command and all callback handlers for strategy management.
Uses aiogram FSM for the multi-step builder conversation.
Call register_strategy_handlers(dp) once at bot startup.
"""
import json
import logging
from datetime import datetime
from typing import Optional

from aiogram import Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FSM States
# ─────────────────────────────────────────────────────────────────────────────

class StrategyBuild(StatesGroup):
    waiting_for_name        = State()
    waiting_for_description = State()
    waiting_for_review      = State()
    waiting_for_tp_sl       = State()
    waiting_for_risk        = State()
    waiting_for_confirm     = State()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_user_and_db(telegram_id: str):
    from app.database import SessionLocal
    from app.models import User
    db   = SessionLocal()
    user = db.query(User).filter(User.telegram_id == str(telegram_id)).first()
    return user, db


def _check_access(user) -> bool:
    if not user:
        return False
    if user.banned:
        return False
    if user.is_admin or user.grandfathered:
        return True
    if user.subscription_end and user.subscription_end > datetime.utcnow():
        return True
    return False


def _strategy_list_keyboard(strategies) -> InlineKeyboardMarkup:
    buttons = []
    for s in strategies:
        status_icon = {"active": "🟢", "paused": "⏸", "draft": "📝", "archived": "📦"}.get(s.status, "•")
        buttons.append([InlineKeyboardButton(
            text=f"{status_icon} {s.name}",
            callback_data=f"strat_view:{s.id}"
        )])
    buttons.append([InlineKeyboardButton(text="➕ Build New Strategy", callback_data="strat_build_new")])
    buttons.append([InlineKeyboardButton(text="🏪 Marketplace", callback_data="strat_marketplace")])
    buttons.append([InlineKeyboardButton(text="🌐 Open Portal", callback_data="strat_portal_link")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def _strategy_detail_keyboard(strategy_id: int, status: str) -> InlineKeyboardMarkup:
    buttons = []
    if status == "active":
        buttons.append([InlineKeyboardButton(text="⏸ Pause", callback_data=f"strat_pause:{strategy_id}")])
    elif status == "paused":
        buttons.append([InlineKeyboardButton(text="▶️ Activate", callback_data=f"strat_activate:{strategy_id}")])
    elif status == "draft":
        buttons.append([InlineKeyboardButton(text="▶️ Go Live", callback_data=f"strat_activate:{strategy_id}")])

    buttons.append([
        InlineKeyboardButton(text="📊 Performance", callback_data=f"strat_perf:{strategy_id}"),
        InlineKeyboardButton(text="🗑 Delete", callback_data=f"strat_delete:{strategy_id}"),
    ])
    buttons.append([
        InlineKeyboardButton(text="📢 Share to Marketplace", callback_data=f"strat_share:{strategy_id}"),
    ])
    buttons.append([InlineKeyboardButton(text="⬅️ My Strategies", callback_data="strat_list")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


# ─────────────────────────────────────────────────────────────────────────────
# /strategy command — main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def cmd_strategy(message: types.Message):
    user, db = _get_user_and_db(str(message.from_user.id))
    try:
        if not _check_access(user):
            await message.answer("⛔ Active subscription required to use the strategy builder.")
            return

        from app.strategy_models import UserStrategy
        strategies = (
            db.query(UserStrategy)
            .filter(UserStrategy.user_id == user.id)
            .order_by(UserStrategy.updated_at.desc())
            .all()
        )

        active_count = sum(1 for s in strategies if s.status == "active")
        text = (
            f"<b>Your Strategies</b>\n"
            f"{len(strategies)} total  ·  {active_count} live\n\n"
            f"Build a strategy in plain English — the AI does the rest.\n"
            f"When conditions hit, trades fire automatically to your Bitunix."
        )
        await message.answer(text, reply_markup=_strategy_list_keyboard(strategies), parse_mode="HTML")
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Strategy list callback
# ─────────────────────────────────────────────────────────────────────────────

async def cb_strat_list(callback: types.CallbackQuery):
    user, db = _get_user_and_db(str(callback.from_user.id))
    try:
        if not _check_access(user):
            await callback.answer("⛔ Subscription required")
            return

        from app.strategy_models import UserStrategy
        strategies = (
            db.query(UserStrategy)
            .filter(UserStrategy.user_id == user.id)
            .order_by(UserStrategy.updated_at.desc())
            .all()
        )
        active_count = sum(1 for s in strategies if s.status == "active")
        text = (
            f"<b>Your Strategies</b>\n"
            f"{len(strategies)} total  ·  {active_count} live"
        )
        await callback.message.edit_text(text, reply_markup=_strategy_list_keyboard(strategies), parse_mode="HTML")
        await callback.answer()
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Build new strategy — Step 1: name
# ─────────────────────────────────────────────────────────────────────────────

async def cb_strat_build_new(callback: types.CallbackQuery, state: FSMContext):
    user, db = _get_user_and_db(str(callback.from_user.id))
    try:
        if not _check_access(user):
            await callback.answer("⛔ Subscription required")
            return
        await state.clear()
        await callback.message.edit_text(
            "<b>Strategy Builder</b> — Step 1 of 3\n\n"
            "What do you want to call this strategy?\n"
            "<i>e.g. \"FVG Bounce\", \"RSI Oversold Scalp\", \"Pump Fader\"</i>",
            parse_mode="HTML",
        )
        await state.set_state(StrategyBuild.waiting_for_name)
        await callback.answer()
    finally:
        db.close()


async def msg_strategy_name(message: types.Message, state: FSMContext):
    name = message.text.strip()[:80]
    await state.update_data(strategy_name=name)
    await state.set_state(StrategyBuild.waiting_for_description)
    await message.answer(
        f"<b>Strategy Builder</b> — Step 2 of 3\n\n"
        f"Name: <b>{name}</b>\n\n"
        f"Now describe your strategy in plain English.\n"
        f"Be as specific as you like — include entry triggers, direction, coins, timeframe.\n\n"
        f"<i>Examples:\n"
        f"• \"Short any coin that pumps 10%+ in 10 minutes with RSI above 80\"\n"
        f"• \"Buy when RSI drops below 28 and price is at a support level, TP 4%, SL 2%\"\n"
        f"• \"Enter long when MACD crosses bullish on 15m and there's a bullish FVG below price\"</i>",
        parse_mode="HTML",
    )


async def msg_strategy_description(message: types.Message, state: FSMContext):
    description = message.text.strip()
    data = await state.get_data()
    name = data.get("strategy_name", "My Strategy")

    await message.answer("⏳ Building your strategy with AI — this takes a few seconds...")

    # Compile with AI
    from app.services.strategy_builder import (
        compile_strategy_from_conversation,
        validate_strategy,
        format_config_for_display,
    )

    compiled = await compile_strategy_from_conversation([], f"Strategy name: {name}\n\n{description}")
    if not compiled or not isinstance(compiled.get("config"), dict):
        await message.answer(
            "❌ Couldn't parse your strategy. Try being more specific about entry conditions and TP/SL.",
        )
        await state.clear()
        return
    config = compiled["config"]
    rationale = str(compiled.get("rationale") or "").strip()

    # Ensure name from user
    config["name"]        = name
    config["description"] = description

    # Validate
    validation = await validate_strategy(config)
    await state.update_data(compiled_config=config, validation=validation, rationale=rationale)
    await state.set_state(StrategyBuild.waiting_for_confirm)

    # Show compiled strategy
    display = format_config_for_display(config)
    warnings = validation.get("warnings", [])
    warn_text = ""
    if warnings:
        warn_text = "\n\n⚠️ <b>Warnings:</b>\n" + "\n".join(f"• {w}" for w in warnings)

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Save Strategy", callback_data="strat_confirm_save"),
            InlineKeyboardButton(text="🔄 Start Over",   callback_data="strat_build_new"),
        ]
    ])

    rationale_text = f"\n\n🧠 <b>Rationale:</b>\n{rationale}" if rationale else ""
    await message.answer(
        f"<b>Here's what I built:</b>\n\n{display}{warn_text}{rationale_text}\n\n"
        f"<i>Risk rating: {validation.get('risk_rating','MEDIUM')}</i>\n\n"
        f"Does this look right? Save it or start over.",
        reply_markup=keyboard,
        parse_mode="HTML",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Confirm and save strategy
# ─────────────────────────────────────────────────────────────────────────────

async def cb_strat_confirm_save(callback: types.CallbackQuery, state: FSMContext):
    user, db = _get_user_and_db(str(callback.from_user.id))
    try:
        data   = await state.get_data()
        config = data.get("compiled_config")
        if not config:
            await callback.answer("Session expired — please start over")
            await state.clear()
            return

        from app.strategy_models import UserStrategy, StrategyPerformance, init_strategy_tables
        from app.database import engine
        init_strategy_tables(engine)

        strategy = UserStrategy(
            user_id     = user.id,
            name        = config.get("name", "My Strategy"),
            description = config.get("description", ""),
            config      = config,
            status      = "draft",
        )
        db.add(strategy)
        db.commit()
        db.refresh(strategy)

        # Create empty performance record
        perf = StrategyPerformance(strategy_id=strategy.id)
        db.add(perf)
        db.commit()

        await state.clear()
        await callback.message.edit_text(
            f"✅ <b>Strategy saved!</b>\n\n"
            f"<b>{strategy.name}</b> is saved as a draft.\n\n"
            f"Tap <b>Go Live</b> when you're ready to start trading it automatically.",
            reply_markup=_strategy_detail_keyboard(strategy.id, "draft"),
            parse_mode="HTML",
        )
        await callback.answer("Strategy saved!")
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# View strategy detail
# ─────────────────────────────────────────────────────────────────────────────

async def cb_strat_view(callback: types.CallbackQuery):
    strategy_id = int(callback.data.split(":")[1])
    user, db = _get_user_and_db(str(callback.from_user.id))
    try:
        from app.strategy_models import UserStrategy, StrategyPerformance
        from app.services.strategy_builder import format_config_for_display

        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == user.id,
        ).first()

        if not strategy:
            await callback.answer("Strategy not found")
            return

        perf = db.query(StrategyPerformance).filter(
            StrategyPerformance.strategy_id == strategy_id
        ).first()

        display = format_config_for_display(strategy.config)
        status_icon = {"active": "🟢 LIVE", "paused": "⏸ PAUSED", "draft": "📝 DRAFT"}.get(strategy.status, strategy.status)

        perf_text = ""
        if perf and perf.total_trades > 0:
            perf_text = (
                f"\n\n<b>Performance</b>\n"
                f"Trades: {perf.total_trades}  ·  Win rate: {perf.win_rate:.0f}%\n"
                f"Total P&L: {perf.total_pnl_pct:+.2f}%"
            )

        await callback.message.edit_text(
            f"{status_icon}\n\n{display}{perf_text}",
            reply_markup=_strategy_detail_keyboard(strategy_id, strategy.status),
            parse_mode="HTML",
        )
        await callback.answer()
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Activate / Pause / Delete
# ─────────────────────────────────────────────────────────────────────────────

async def cb_strat_activate(callback: types.CallbackQuery):
    strategy_id = int(callback.data.split(":")[1])
    user, db = _get_user_and_db(str(callback.from_user.id))
    try:
        from app.strategy_models import UserStrategy
        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == user.id,
        ).first()
        if strategy:
            strategy.status = "active"
            db.commit()
            await callback.message.edit_text(
                f"🟢 <b>{strategy.name}</b> is now live!\n\n"
                f"The bot will scan for your conditions every 45 seconds and trade automatically.",
                reply_markup=_strategy_detail_keyboard(strategy_id, "active"),
                parse_mode="HTML",
            )
        await callback.answer("Strategy activated!")
    finally:
        db.close()


async def cb_strat_pause(callback: types.CallbackQuery):
    strategy_id = int(callback.data.split(":")[1])
    user, db = _get_user_and_db(str(callback.from_user.id))
    try:
        from app.strategy_models import UserStrategy
        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == user.id,
        ).first()
        if strategy:
            strategy.status = "paused"
            db.commit()
            await callback.message.edit_text(
                f"⏸ <b>{strategy.name}</b> paused.\n\nNo new trades will fire until you reactivate it.",
                reply_markup=_strategy_detail_keyboard(strategy_id, "paused"),
                parse_mode="HTML",
            )
        await callback.answer("Strategy paused")
    finally:
        db.close()


async def cb_strat_delete(callback: types.CallbackQuery):
    strategy_id = int(callback.data.split(":")[1])
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🗑 Yes, delete it", callback_data=f"strat_delete_confirm:{strategy_id}"),
        InlineKeyboardButton(text="Cancel",             callback_data=f"strat_view:{strategy_id}"),
    ]])
    await callback.message.edit_text(
        "Are you sure you want to delete this strategy? This cannot be undone.",
        reply_markup=keyboard,
    )
    await callback.answer()


async def cb_strat_delete_confirm(callback: types.CallbackQuery):
    strategy_id = int(callback.data.split(":")[1])
    user, db = _get_user_and_db(str(callback.from_user.id))
    try:
        from app.strategy_models import UserStrategy
        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == user.id,
        ).first()
        if strategy:
            name = strategy.name
            db.delete(strategy)
            db.commit()
            await callback.message.edit_text(
                f"🗑 <b>{name}</b> deleted.",
                parse_mode="HTML",
            )
        await callback.answer("Deleted")
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Performance detail
# ─────────────────────────────────────────────────────────────────────────────

async def cb_strat_perf(callback: types.CallbackQuery):
    strategy_id = int(callback.data.split(":")[1])
    user, db = _get_user_and_db(str(callback.from_user.id))
    try:
        from app.strategy_models import UserStrategy, StrategyExecution, StrategyPerformance

        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == user.id,
        ).first()
        if not strategy:
            await callback.answer("Not found")
            return

        perf = db.query(StrategyPerformance).filter(
            StrategyPerformance.strategy_id == strategy_id
        ).first()

        recent = (
            db.query(StrategyExecution)
            .filter(StrategyExecution.strategy_id == strategy_id)
            .order_by(StrategyExecution.fired_at.desc())
            .limit(5)
            .all()
        )

        if not perf or perf.total_trades == 0:
            text = f"<b>{strategy.name}</b> — Performance\n\nNo trades yet. Activate to start trading."
        else:
            text = (
                f"<b>{strategy.name}</b> — Performance\n\n"
                f"Total trades:  {perf.total_trades}\n"
                f"Open:          {perf.open_trades}\n"
                f"Wins:          {perf.wins}  ({perf.win_rate:.0f}%)\n"
                f"Losses:        {perf.losses}\n"
                f"Total P&L:     {perf.total_pnl_pct:+.2f}%\n"
                f"Avg win:       {perf.avg_win_pct:+.2f}%\n"
                f"Avg loss:      {perf.avg_loss_pct:+.2f}%\n"
                f"Best trade:    {perf.best_trade:+.2f}%\n"
                f"Worst trade:   {perf.worst_trade:+.2f}%\n"
            )

        if recent:
            text += "\n<b>Recent trades:</b>"
            for ex in recent:
                outcome_icon = {"WIN": "✅", "LOSS": "❌", "OPEN": "⏳", "BREAKEVEN": "➡️"}.get(ex.outcome, "•")
                pnl_str = f"{ex.pnl_pct:+.2f}%" if ex.pnl_pct is not None else "open"
                text += f"\n{outcome_icon} {ex.symbol} {ex.direction}  {pnl_str}"

        keyboard = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="⬅️ Back", callback_data=f"strat_view:{strategy_id}")
        ]])
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
        await callback.answer()
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Marketplace
# ─────────────────────────────────────────────────────────────────────────────

async def cb_strat_marketplace(callback: types.CallbackQuery):
    user, db = _get_user_and_db(str(callback.from_user.id))
    try:
        from app.strategy_models import StrategyMarketplace, UserStrategy, StrategyPerformance

        featured = (
            db.query(StrategyMarketplace)
            .filter(StrategyMarketplace.is_featured == True)
            .limit(5)
            .all()
        )
        popular = (
            db.query(StrategyMarketplace)
            .order_by(StrategyMarketplace.clone_count.desc())
            .limit(10)
            .all()
        )

        all_items = {m.id: m for m in featured + popular}
        buttons = []
        for m in list(all_items.values())[:8]:
            strat = db.query(UserStrategy).filter(UserStrategy.id == m.strategy_id).first()
            perf  = db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id == m.strategy_id).first()
            wr    = f"{perf.win_rate:.0f}%" if perf and perf.total_trades > 0 else "new"
            star  = "⭐ " if m.is_featured else ""
            buttons.append([InlineKeyboardButton(
                text=f"{star}{m.title}  ·  WR:{wr}  ·  {m.clone_count} clones",
                callback_data=f"strat_market_view:{m.id}"
            )])

        buttons.append([InlineKeyboardButton(text="⬅️ My Strategies", callback_data="strat_list")])

        text = (
            "<b>Strategy Marketplace</b>\n\n"
            "Browse and clone strategies built by the community.\n"
            "Cloning copies the config to your account — edit freely."
        ) if all_items else (
            "<b>Strategy Marketplace</b>\n\nNo strategies published yet. "
            "Build one and share it with the community!"
        )

        await callback.message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
            parse_mode="HTML",
        )
        await callback.answer()
    finally:
        db.close()


async def cb_strat_share(callback: types.CallbackQuery):
    strategy_id = int(callback.data.split(":")[1])
    user, db = _get_user_and_db(str(callback.from_user.id))
    try:
        from app.strategy_models import UserStrategy, StrategyMarketplace

        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == user.id,
        ).first()
        if not strategy:
            await callback.answer("Not found")
            return

        existing = db.query(StrategyMarketplace).filter(
            StrategyMarketplace.strategy_id == strategy_id
        ).first()

        if existing:
            from app.services.strategy_builder import generate_strategy_summary
            summary = await generate_strategy_summary(strategy.config)
            existing.title = strategy.name
            existing.summary = summary
            strategy.is_public = True
            db.commit()
            await callback.answer("Already listed — details refreshed!")
            return

        from app.services.strategy_builder import generate_strategy_summary
        summary = await generate_strategy_summary(strategy.config)

        listing = StrategyMarketplace(
            strategy_id = strategy_id,
            author_id   = user.id,
            title       = strategy.name,
            summary     = summary,
            tags        = [],
        )
        db.add(listing)
        strategy.is_public = True
        db.commit()

        await callback.message.edit_text(
            f"📢 <b>{strategy.name}</b> published to the marketplace!\n\n"
            f"Other members can now clone and run your strategy.",
            reply_markup=_strategy_detail_keyboard(strategy_id, strategy.status),
            parse_mode="HTML",
        )
        await callback.answer("Published!")
    finally:
        db.close()


async def cb_strat_portal_link(callback: types.CallbackQuery):
    """Send the user their personal portal link."""
    import os
    user, db = _get_user_and_db(str(callback.from_user.id))
    try:
        domain = os.environ.get("REPLIT_DEV_DOMAIN", "localhost")
        portal_url = f"https://{domain}:8080/strategies?uid={user.uid}"
        await callback.message.edit_text(
            f"<b>Strategy Portal</b>\n\n"
            f"Your personal portal — manage strategies, view performance, explore the marketplace.\n\n"
            f"🌐 <a href='{portal_url}'>Open Portal</a>\n\n"
            f"<i>This link is personal — don't share it.</i>",
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        await callback.answer()
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

def register_strategy_handlers(dp: Dispatcher):
    """Register all strategy handlers on the shared dispatcher."""
    dp.message.register(cmd_strategy, Command("strategy"))

    dp.callback_query.register(cb_strat_list,           F.data == "strat_list")
    dp.callback_query.register(cb_strat_build_new,      F.data == "strat_build_new")
    dp.callback_query.register(cb_strat_confirm_save,   F.data == "strat_confirm_save")
    dp.callback_query.register(cb_strat_marketplace,    F.data == "strat_marketplace")
    dp.callback_query.register(cb_strat_portal_link,    F.data == "strat_portal_link")

    dp.callback_query.register(cb_strat_view,           F.data.startswith("strat_view:"))
    dp.callback_query.register(cb_strat_activate,       F.data.startswith("strat_activate:"))
    dp.callback_query.register(cb_strat_pause,          F.data.startswith("strat_pause:"))
    dp.callback_query.register(cb_strat_delete,         F.data.startswith("strat_delete:"))
    dp.callback_query.register(cb_strat_delete_confirm, F.data.startswith("strat_delete_confirm:"))
    dp.callback_query.register(cb_strat_perf,           F.data.startswith("strat_perf:"))
    dp.callback_query.register(cb_strat_share,          F.data.startswith("strat_share:"))

    dp.message.register(msg_strategy_name,        StrategyBuild.waiting_for_name)
    dp.message.register(msg_strategy_description, StrategyBuild.waiting_for_description)

    logger.info("✅ Strategy handlers registered")
