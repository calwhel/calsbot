import asyncio
import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models import User, UserPreference, Trade, Signal
from app.services.signals import SignalGenerator

logger = logging.getLogger(__name__)

bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
signal_generator = SignalGenerator()


def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        pass


def get_or_create_user(telegram_id: int, username: str = None, first_name: str = None, db: Session = None):
    should_close = False
    if db is None:
        db = SessionLocal()
        should_close = True
    
    try:
        user = db.query(User).filter(User.telegram_id == str(telegram_id)).first()
        if not user:
            user = User(
                telegram_id=str(telegram_id),
                username=username,
                first_name=first_name
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            
            prefs = UserPreference(user_id=user.id)
            db.add(prefs)
            db.commit()
        
        return user
    finally:
        if should_close:
            db.close()


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user = get_or_create_user(
        message.from_user.id,
        message.from_user.username,
        message.from_user.first_name
    )
    
    welcome_text = f"""
🚀 Welcome to Crypto Perps Signals Bot!

Get FREE real-time trading signals based on EMA crossovers with support/resistance levels.

Available Commands:
/dashboard - View your trading dashboard
/settings - Configure your preferences
/status - Check your bot status
/subscribe - Learn about features

Let's get started! 📈
"""
    await message.answer(welcome_text)


@dp.message(Command("status"))
async def cmd_status(message: types.Message):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        prefs = user.preferences
        dm_status = "Enabled" if (prefs and prefs.dm_alerts) else "Disabled"
        
        status_text = f"""
✅ Bot Status: Active
🔔 DM Alerts: {dm_status}
📊 Signals: Broadcasting
👤 User ID: {user.telegram_id}
"""
        
        await message.answer(status_text)
    finally:
        db.close()


@dp.message(Command("subscribe"))
async def cmd_subscribe(message: types.Message):
    subscribe_text = """
🎉 This bot is FREE to use!

You already have access to:
✅ Real-time EMA crossover signals
✅ Support/Resistance levels
✅ Entry, Stop Loss & Take Profit prices
✅ PnL tracking
✅ Custom alerts

Use /dashboard to get started!
"""
    await message.answer(subscribe_text)


@dp.message(Command("dashboard"))
async def cmd_dashboard(message: types.Message):
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="📊 PnL Today", callback_data="pnl_today"),
            InlineKeyboardButton(text="📈 PnL Week", callback_data="pnl_week")
        ],
        [
            InlineKeyboardButton(text="📅 PnL Month", callback_data="pnl_month"),
            InlineKeyboardButton(text="🔄 Active Trades", callback_data="active_trades")
        ],
        [
            InlineKeyboardButton(text="📡 Recent Signals", callback_data="recent_signals"),
            InlineKeyboardButton(text="⚙️ Settings", callback_data="settings")
        ]
    ])
    
    await message.answer("📊 Trading Dashboard", reply_markup=keyboard)


@dp.callback_query(F.data.startswith("pnl_"))
async def handle_pnl_callback(callback: CallbackQuery):
    period = callback.data.split("_")[1]
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found")
            return
        
        now = datetime.utcnow()
        if period == "today":
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "week":
            start_date = now - timedelta(days=7)
        else:
            start_date = now - timedelta(days=30)
        
        trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.closed_at >= start_date,
            Trade.status == "closed"
        ).all()
        
        total_pnl = sum(t.pnl for t in trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl < 0])
        
        pnl_text = f"""
📊 PnL Summary ({period.title()})

💰 Total PnL: ${total_pnl:.2f}
✅ Winning Trades: {winning_trades}
❌ Losing Trades: {losing_trades}
📈 Total Trades: {len(trades)}
"""
        
        if len(trades) > 0:
            win_rate = (winning_trades / len(trades)) * 100
            pnl_text += f"🎯 Win Rate: {win_rate:.1f}%"
        
        await callback.message.answer(pnl_text)
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data == "active_trades")
async def handle_active_trades(callback: CallbackQuery):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found")
            return
        
        trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.status == "open"
        ).all()
        
        if not trades:
            await callback.message.answer("No active trades")
            await callback.answer()
            return
        
        trades_text = "🔄 Active Trades:\n\n"
        for trade in trades:
            trades_text += f"""
{trade.symbol} {trade.direction}
Entry: ${trade.entry_price}
SL: ${trade.stop_loss} | TP: ${trade.take_profit}
---
"""
        
        await callback.message.answer(trades_text)
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data == "recent_signals")
async def handle_recent_signals(callback: CallbackQuery):
    db = SessionLocal()
    
    try:
        signals = db.query(Signal).order_by(Signal.created_at.desc()).limit(5).all()
        
        if not signals:
            await callback.message.answer("No recent signals")
            await callback.answer()
            return
        
        signals_text = "📡 Recent Signals:\n\n"
        for signal in signals:
            signals_text += f"""
{signal.symbol} {signal.direction}
Entry: ${signal.entry_price}
SL: ${signal.stop_loss} | TP: ${signal.take_profit}
Time: {signal.created_at.strftime('%H:%M')}
---
"""
        
        await callback.message.answer(signals_text)
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data == "settings")
async def handle_settings_callback(callback: CallbackQuery):
    await cmd_settings(callback.message)
    await callback.answer()


@dp.message(Command("settings"))
async def cmd_settings(message: types.Message):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user or not user.preferences:
            await message.answer("Settings not found. Use /start first.")
            return
        
        prefs = user.preferences
        muted = prefs.get_muted_symbols_list()
        muted_str = ", ".join(muted) if muted else "None"
        
        settings_text = f"""
⚙️ Your Settings

🔕 Muted Symbols: {muted_str}
📊 Default PnL Period: {prefs.default_pnl_period}
🔔 DM Alerts: {"Enabled" if prefs.dm_alerts else "Disabled"}

Use these commands to update:
/mute <symbol> - Mute a symbol
/unmute <symbol> - Unmute a symbol
/set_pnl <today/week/month> - Set default PnL period
/toggle_alerts - Enable/Disable DM alerts
"""
        
        await message.answer(settings_text)
    finally:
        db.close()


@dp.message(Command("mute"))
async def cmd_mute(message: types.Message):
    db = SessionLocal()
    
    try:
        args = message.text.split()
        if len(args) < 2:
            await message.answer("Usage: /mute <symbol>\nExample: /mute BTC/USDT:USDT")
            return
        
        symbol = args[1]
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        
        if user and user.preferences:
            user.preferences.add_muted_symbol(symbol)
            db.commit()
            await message.answer(f"✅ Muted {symbol}")
        else:
            await message.answer("Settings not found. Use /start first.")
    finally:
        db.close()


@dp.message(Command("unmute"))
async def cmd_unmute(message: types.Message):
    db = SessionLocal()
    
    try:
        args = message.text.split()
        if len(args) < 2:
            await message.answer("Usage: /unmute <symbol>\nExample: /unmute BTC/USDT:USDT")
            return
        
        symbol = args[1]
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        
        if user and user.preferences:
            user.preferences.remove_muted_symbol(symbol)
            db.commit()
            await message.answer(f"✅ Unmuted {symbol}")
        else:
            await message.answer("Settings not found. Use /start first.")
    finally:
        db.close()


@dp.message(Command("set_pnl"))
async def cmd_set_pnl(message: types.Message):
    db = SessionLocal()
    
    try:
        args = message.text.split()
        if len(args) < 2 or args[1] not in ["today", "week", "month"]:
            await message.answer("Usage: /set_pnl <today/week/month>")
            return
        
        period = args[1]
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        
        if user and user.preferences:
            user.preferences.default_pnl_period = period
            db.commit()
            await message.answer(f"✅ Default PnL period set to {period}")
        else:
            await message.answer("Settings not found. Use /start first.")
    finally:
        db.close()


@dp.message(Command("toggle_alerts"))
async def cmd_toggle_alerts(message: types.Message):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        
        if user and user.preferences:
            user.preferences.dm_alerts = not user.preferences.dm_alerts
            db.commit()
            status = "enabled" if user.preferences.dm_alerts else "disabled"
            await message.answer(f"✅ DM alerts {status}")
        else:
            await message.answer("Settings not found. Use /start first.")
    finally:
        db.close()


async def broadcast_signal(signal_data: dict):
    db = SessionLocal()
    
    try:
        signal = Signal(**signal_data)
        db.add(signal)
        db.commit()
        db.refresh(signal)
        
        logger.info(f"Broadcasting {signal.direction} signal for {signal.symbol}")
        
        # Calculate risk/reward ratio
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Safe volume percentage calculation
        if signal.volume_avg and signal.volume_avg > 0:
            volume_pct = ((signal.volume / signal.volume_avg - 1) * 100)
            volume_text = f"{signal.volume:,.0f} ({'+' if signal.volume > signal.volume_avg else ''}{volume_pct:.1f}% avg)"
        else:
            volume_text = f"{signal.volume:,.0f}"
        
        signal_text = f"""
🚨 NEW {signal.direction} SIGNAL

📊 Symbol: {signal.symbol}
💰 Entry: ${signal.entry_price}
🛑 Stop Loss: ${signal.stop_loss}
🎯 Take Profit: ${signal.take_profit}

📊 RSI: {signal.rsi}
📈 Volume: {volume_text}
⚡ ATR: ${signal.atr}

📈 Support: ${signal.support_level}
📉 Resistance: ${signal.resistance_level}

💎 Risk/Reward: 1:{rr_ratio:.2f}
⏰ {signal.timeframe} | {signal.created_at.strftime('%H:%M:%S')}
"""
        
        await bot.send_message(settings.BROADCAST_CHAT_ID, signal_text)
        logger.info(f"Broadcast to channel successful")
        
        users = db.query(User).all()
        
        for user in users:
            if user.preferences and user.preferences.dm_alerts:
                muted_symbols = user.preferences.get_muted_symbols_list()
                if signal.symbol not in muted_symbols:
                    try:
                        await bot.send_message(user.telegram_id, signal_text)
                        logger.info(f"Sent DM to user {user.telegram_id}")
                    except Exception as e:
                        logger.error(f"Failed to send to {user.telegram_id}: {e}")
    
    finally:
        db.close()


async def signal_scanner():
    logger.info("Signal scanner started")
    while True:
        try:
            logger.info("Scanning for signals...")
            signals = await signal_generator.scan_all_symbols()
            logger.info(f"Found {len(signals)} signals")
            for signal in signals:
                await broadcast_signal(signal)
        except Exception as e:
            logger.error(f"Signal scanner error: {e}", exc_info=True)
        
        await asyncio.sleep(60)


async def start_bot():
    logger.info("Starting Telegram bot...")
    asyncio.create_task(signal_scanner())
    try:
        logger.info("Bot polling started")
        await dp.start_polling(bot)
    finally:
        await signal_generator.close()
