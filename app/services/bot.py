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
from app.services.mexc_trader import execute_auto_trade
from app.utils.encryption import encrypt_api_key, decrypt_api_key

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
            # Check if this is the first user - make them admin
            total_users = db.query(User).count()
            is_first_user = total_users == 0
            
            user = User(
                telegram_id=str(telegram_id),
                username=username,
                first_name=first_name,
                is_admin=is_first_user,
                approved=is_first_user
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            
            prefs = UserPreference(user_id=user.id)
            db.add(prefs)
            db.commit()
            
            # Notify admins about new user (if not first user)
            if not is_first_user:
                admins = db.query(User).filter(User.is_admin == True).all()
                for admin in admins:
                    try:
                        asyncio.create_task(
                            bot.send_message(
                                admin.telegram_id,
                                f"🔔 New user joined!\n\n"
                                f"👤 User: @{username or 'N/A'} ({first_name or 'N/A'})\n"
                                f"🆔 ID: `{telegram_id}`\n\n"
                                f"Use /approve {telegram_id} to grant access."
                            )
                        )
                    except:
                        pass
        
        return user
    finally:
        if should_close:
            db.close()


def calculate_leverage_pnl(entry: float, target: float, direction: str, leverage: int = 10) -> float:
    """Calculate PnL percentage with leverage"""
    if direction == "LONG":
        pnl_pct = ((target - entry) / entry) * leverage * 100
    else:
        pnl_pct = ((entry - target) / entry) * leverage * 100
    return pnl_pct


def is_admin(telegram_id: int, db: Session) -> bool:
    """Check if user is admin"""
    user = db.query(User).filter(User.telegram_id == str(telegram_id)).first()
    return user and user.is_admin


def check_access(user: User) -> tuple[bool, str]:
    """Check if user has access to bot. Returns (has_access, reason)"""
    if user.banned:
        ban_message = "❌ You have been banned from using this bot."
        if user.admin_notes:
            ban_message += f"\n\nReason: {user.admin_notes}"
        return False, ban_message
    if not user.approved and not user.is_admin:
        return False, "⏳ Your account is pending approval. Please wait for admin approval."
    return True, ""


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    db = SessionLocal()
    try:
        user = get_or_create_user(
            message.from_user.id,
            message.from_user.username,
            message.from_user.first_name,
            db
        )
        
        has_access, reason = check_access(user)
        
        if not has_access:
            await message.answer(reason)
            return
        
        welcome_text = f"""
🚀 Welcome to Crypto Perps Signals Bot!

Get FREE real-time trading signals based on EMA crossovers with support/resistance levels.

🤖 **Auto-Trading on MEXC!**
Connect your MEXC API and let the bot trade for you automatically with advanced risk management.

Available Commands:
/dashboard - View your trading dashboard
/autotrading_status - Check auto-trading status
/set_mexc_api - Connect MEXC account
/risk_settings - Configure risk management
/security_settings - Set safety limits
/emergency_stop - Instantly stop all trading
/settings - Configure your preferences

Let's get started! 📈
"""
        await message.answer(welcome_text)
    finally:
        db.close()


@dp.message(Command("status"))
async def cmd_status(message: types.Message):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await message.answer(reason)
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
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await message.answer(reason)
            return
        
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
    finally:
        db.close()


@dp.message(Command("dashboard"))
async def cmd_dashboard(message: types.Message):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await message.answer(reason)
            return
        
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
    finally:
        db.close()


@dp.callback_query(F.data.startswith("pnl_"))
async def handle_pnl_callback(callback: CallbackQuery):
    period = callback.data.split("_")[1]
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            await callback.answer()
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
        
        if not trades:
            pnl_text = f"""
📊 PnL Summary ({period.title()})

No closed trades in this period.
Use /autotrading_status to set up auto-trading!
"""
        else:
            total_pnl = sum(t.pnl for t in trades)
            total_pnl_pct = sum(t.pnl_percent for t in trades)
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            avg_pnl = total_pnl / len(trades) if trades else 0
            avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            best_trade = max(trades, key=lambda t: t.pnl) if trades else None
            worst_trade = min(trades, key=lambda t: t.pnl) if trades else None
            
            win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0
            
            pnl_text = f"""
📊 PnL Summary ({period.title()})

💰 Total PnL: ${total_pnl:.2f} ({total_pnl_pct:+.2f}%)
📈 Total Trades: {len(trades)}
✅ Wins: {len(winning_trades)} | ❌ Losses: {len(losing_trades)}
🎯 Win Rate: {win_rate:.1f}%

📊 Statistics:
  • Avg PnL/Trade: ${avg_pnl:.2f}
  • Avg Win: ${avg_win:.2f}
  • Avg Loss: ${avg_loss:.2f}
  
🏆 Best Trade: ${best_trade.pnl:.2f} ({best_trade.symbol})
📉 Worst Trade: ${worst_trade.pnl:.2f} ({worst_trade.symbol})
"""
        
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
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            await callback.answer()
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
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            await callback.answer()
            return
        
        signals = db.query(Signal).order_by(Signal.created_at.desc()).limit(5).all()
        
        if not signals:
            await callback.message.answer("No recent signals")
            await callback.answer()
            return
        
        signals_text = "📡 Recent Signals (10x Leverage PnL):\n\n"
        for signal in signals:
            tp_pnl = calculate_leverage_pnl(signal.entry_price, signal.take_profit, signal.direction, 10)
            sl_pnl = calculate_leverage_pnl(signal.entry_price, signal.stop_loss, signal.direction, 10)
            
            signals_text += f"""
{signal.symbol} {signal.direction}
Entry: ${signal.entry_price}
SL: ${signal.stop_loss} | TP: ${signal.take_profit}

💰 10x Leverage:
  ✅ TP Hit: {tp_pnl:+.2f}%
  ❌ SL Hit: {sl_pnl:+.2f}%
  
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
        
        has_access, reason = check_access(user)
        if not has_access:
            await message.answer(reason)
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
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await message.answer(reason)
            return
        
        args = message.text.split()
        if len(args) < 2:
            await message.answer("Usage: /mute <symbol>\nExample: /mute BTC/USDT:USDT")
            return
        
        symbol = args[1]
        
        if user.preferences:
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
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await message.answer(reason)
            return
        
        args = message.text.split()
        if len(args) < 2:
            await message.answer("Usage: /unmute <symbol>\nExample: /unmute BTC/USDT:USDT")
            return
        
        symbol = args[1]
        
        if user.preferences:
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
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await message.answer(reason)
            return
        
        args = message.text.split()
        if len(args) < 2 or args[1] not in ["today", "week", "month"]:
            await message.answer("Usage: /set_pnl <today/week/month>")
            return
        
        period = args[1]
        
        if user.preferences:
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
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await message.answer(reason)
            return
        
        if user.preferences:
            user.preferences.dm_alerts = not user.preferences.dm_alerts
            db.commit()
            status = "enabled" if user.preferences.dm_alerts else "disabled"
            await message.answer(f"✅ DM alerts {status}")
        else:
            await message.answer("Settings not found. Use /start first.")
    finally:
        db.close()


@dp.message(Command("set_mexc_api"))
async def cmd_set_mexc_api(message: types.Message):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await message.answer(reason)
            return
        
        args = message.text.split()
        if len(args) < 3:
            await message.answer("""
⚠️ Usage: /set_mexc_api <API_KEY> <API_SECRET>

⚙️ How to get MEXC API keys:
1. Go to MEXC → API Management
2. Create new API key
3. Enable Futures Trading permission
4. Copy API Key and Secret

Example: /set_mexc_api mx0_xxx your_secret_here
            """)
            return
        
        api_key = args[1]
        api_secret = args[2]
        
        if user.preferences:
            user.preferences.mexc_api_key = encrypt_api_key(api_key)
            user.preferences.mexc_api_secret = encrypt_api_key(api_secret)
            db.commit()
            
            await message.delete()
            
            await message.answer("""
✅ MEXC API keys saved successfully!

🔐 Your message has been deleted for security.
🔒 Keys are encrypted and stored securely.

Use /toggle_autotrading to enable auto-trading
Use /autotrading_status to check your settings
            """)
        else:
            await message.answer("Settings not found. Use /start first.")
    finally:
        db.close()


@dp.message(Command("remove_mexc_api"))
async def cmd_remove_mexc_api(message: types.Message):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await message.answer(reason)
            return
        
        if user.preferences:
            user.preferences.mexc_api_key = None
            user.preferences.mexc_api_secret = None
            user.preferences.auto_trading_enabled = False
            db.commit()
            await message.answer("✅ MEXC API keys removed and auto-trading disabled")
        else:
            await message.answer("Settings not found. Use /start first.")
    finally:
        db.close()


@dp.message(Command("toggle_autotrading"))
async def cmd_toggle_autotrading(message: types.Message):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await message.answer(reason)
            return
        
        if user.preferences:
            if not user.preferences.mexc_api_key or not user.preferences.mexc_api_secret:
                await message.answer("❌ Please set your MEXC API keys first using /set_mexc_api")
                return
            
            user.preferences.auto_trading_enabled = not user.preferences.auto_trading_enabled
            db.commit()
            status = "enabled" if user.preferences.auto_trading_enabled else "disabled"
            await message.answer(f"✅ Auto-trading {status}")
        else:
            await message.answer("Settings not found. Use /start first.")
    finally:
        db.close()


@dp.message(Command("autotrading_status"))
async def cmd_autotrading_status(message: types.Message):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await message.answer(reason)
            return
        
        if not user.preferences:
            await message.answer("Settings not found. Use /start first.")
            return
        
        prefs = user.preferences
        
        api_status = "✅ Set" if prefs.mexc_api_key and prefs.mexc_api_secret else "❌ Not Set"
        auto_status = "✅ Enabled" if prefs.auto_trading_enabled else "❌ Disabled"
        risk_sizing = "✅ Enabled" if prefs.risk_based_sizing else "❌ Disabled"
        trailing_stop = "✅ Enabled" if prefs.use_trailing_stop else "❌ Disabled"
        breakeven_stop = "✅ Enabled" if prefs.use_breakeven_stop else "❌ Disabled"
        
        open_positions = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.status == 'open'
        ).count()
        
        status_text = f"""
🤖 Auto-Trading Status

📊 API Keys: {api_status}
⚡ Auto-Trading: {auto_status}
💰 Position Size: {prefs.position_size_percent}% of balance
🎯 Max Positions: {prefs.max_positions}
📈 Open Positions: {open_positions}/{prefs.max_positions}

⚠️ Risk Management:
  • Accepted Risk: {prefs.accepted_risk_levels}
  • Risk-Based Sizing: {risk_sizing}
  • Trailing Stop: {trailing_stop}
  • Breakeven Stop: {breakeven_stop}

Commands:
/set_mexc_api - Set API keys
/risk_settings - Configure risk management
/toggle_autotrading - Toggle on/off
        """
        
        await message.answer(status_text)
    finally:
        db.close()


@dp.message(Command("emergency_stop"))
async def cmd_emergency_stop(message: types.Message):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await message.answer(reason)
            return
        
        if user.preferences:
            user.preferences.emergency_stop = True
            user.preferences.auto_trading_enabled = False
            db.commit()
            await message.answer("""
🚨 EMERGENCY STOP ACTIVATED!

All auto-trading has been STOPPED immediately.

To resume trading:
1. Review your account
2. Use /security_settings to disable emergency stop
3. Re-enable auto-trading with /toggle_autotrading
""")
    finally:
        db.close()


@dp.message(Command("security_settings"))
async def cmd_security_settings(message: types.Message):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await message.answer(reason)
            return
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="💰 Daily Loss Limit", callback_data="set_daily_loss"),
                InlineKeyboardButton(text="📉 Max Drawdown", callback_data="set_max_drawdown")
            ],
            [
                InlineKeyboardButton(text="💵 Min Balance", callback_data="set_min_balance"),
                InlineKeyboardButton(text="❌ Max Losses", callback_data="set_max_losses")
            ],
            [
                InlineKeyboardButton(text="🚨 Toggle Emergency Stop", callback_data="toggle_emergency")
            ],
            [
                InlineKeyboardButton(text="📊 View Security Status", callback_data="security_status")
            ]
        ])
        
        await message.answer("""
🛡️ **Security Settings**

Protect your trading account with safety limits:

💰 **Daily Loss Limit** - Stop trading if daily losses exceed limit
📉 **Max Drawdown** - Stop if account drops X% from peak
💵 **Min Balance** - Don't trade below minimum balance
❌ **Max Consecutive Losses** - Pause after N losses in a row
🚨 **Emergency Stop** - Instantly disable all trading

Select an option below:
""", reply_markup=keyboard)
    finally:
        db.close()


@dp.callback_query(F.data == "toggle_emergency")
async def handle_toggle_emergency(callback: CallbackQuery):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            await callback.answer()
            return
        
        if user.preferences:
            user.preferences.emergency_stop = not user.preferences.emergency_stop
            db.commit()
            
            if user.preferences.emergency_stop:
                user.preferences.auto_trading_enabled = False
                db.commit()
                await callback.message.edit_text("""
🚨 EMERGENCY STOP ACTIVATED!

All auto-trading has been STOPPED.

To resume:
1. Toggle emergency stop OFF
2. Re-enable auto-trading
""")
            else:
                await callback.message.edit_text("""
✅ Emergency stop DEACTIVATED

You can now re-enable auto-trading if desired.
Use /toggle_autotrading to turn it back on.
""")
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data == "security_status")
async def handle_security_status(callback: CallbackQuery):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            await callback.answer()
            return
        
        if user.preferences:
            prefs = user.preferences
            emergency = "🚨 ACTIVE" if prefs.emergency_stop else "✅ OFF"
            
            # Calculate current drawdown
            from app.services.mexc_trader import MEXCTrader
            from app.utils.encryption import decrypt_api_key
            
            current_drawdown = 0
            balance = 0
            
            if prefs.mexc_api_key and prefs.mexc_api_secret:
                try:
                    api_key = decrypt_api_key(prefs.mexc_api_key)
                    api_secret = decrypt_api_key(prefs.mexc_api_secret)
                    trader = MEXCTrader(api_key, api_secret)
                    balance = await trader.get_account_balance()
                    await trader.close()
                    
                    if prefs.peak_balance > 0:
                        current_drawdown = ((prefs.peak_balance - balance) / prefs.peak_balance) * 100
                except:
                    pass
            
            status_text = f"""
🛡️ **Security Status**

🚨 Emergency Stop: {emergency}

💰 Daily Loss Limit: ${prefs.daily_loss_limit:.2f}
📉 Max Drawdown: {prefs.max_drawdown_percent}%
💵 Min Balance: ${prefs.min_balance:.2f}
❌ Max Consecutive Losses: {prefs.max_consecutive_losses}
⏱️ Cooldown After Loss: {prefs.cooldown_after_loss} min

📊 **Current Status:**
  • Balance: ${balance:.2f}
  • Peak Balance: ${prefs.peak_balance:.2f}
  • Current Drawdown: {current_drawdown:.1f}%
  • Consecutive Losses: {prefs.consecutive_losses}
"""
            await callback.message.edit_text(status_text)
        await callback.answer()
    finally:
        db.close()


@dp.message(Command("risk_settings"))
async def cmd_risk_settings(message: types.Message):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await message.answer(reason)
            return
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="🎯 Set Risk Levels", callback_data="set_risk_levels")
            ],
            [
                InlineKeyboardButton(text="📊 Toggle Risk Sizing", callback_data="toggle_risk_sizing"),
                InlineKeyboardButton(text="🔄 Toggle Trailing Stop", callback_data="toggle_trailing")
            ],
            [
                InlineKeyboardButton(text="🛡️ Toggle Breakeven Stop", callback_data="toggle_breakeven"),
                InlineKeyboardButton(text="💰 Set Position Size", callback_data="set_position_size")
            ]
        ])
        
        await message.answer("""
⚙️ **Risk Management Settings**

Configure your auto-trading risk preferences:

🎯 **Risk Levels** - Choose which risk signals to trade
📊 **Risk-Based Sizing** - Auto-reduce position size for higher risk
🔄 **Trailing Stop** - Lock in profits as price moves favorably
🛡️ **Breakeven Stop** - Move SL to entry once in profit
💰 **Position Size** - Set base position size percentage

Select an option below:
""", reply_markup=keyboard)
    finally:
        db.close()


@dp.callback_query(F.data == "set_risk_levels")
async def handle_set_risk_levels(callback: CallbackQuery):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            await callback.answer()
            return
    finally:
        db.close()
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🟢 LOW Risk Only", callback_data="risk_level_LOW")],
        [InlineKeyboardButton(text="🟢🟡 LOW + MEDIUM Risk", callback_data="risk_level_LOW,MEDIUM")],
        [InlineKeyboardButton(text="🔙 Back", callback_data="back_to_risk_settings")]
    ])
    
    await callback.message.edit_text("""
🎯 **Select Accepted Risk Levels**

Choose which risk level signals to auto-trade:

🟢 **LOW Risk Only** - Most conservative, fewer trades
🟢🟡 **LOW + MEDIUM** - Balanced approach (recommended)

HIGH risk signals are never auto-traded.
""", reply_markup=keyboard)
    await callback.answer()


@dp.callback_query(F.data.startswith("risk_level_"))
async def handle_risk_level_selection(callback: CallbackQuery):
    db = SessionLocal()
    
    try:
        risk_levels = callback.data.split("_", 2)[2]
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            await callback.answer()
            return
        
        if user.preferences:
            user.preferences.accepted_risk_levels = risk_levels
            db.commit()
            await callback.message.edit_text(f"✅ Risk levels updated to: {risk_levels}")
        await callback.answer()
    finally:
        db.close()


@dp.message(Command("admin"))
async def cmd_admin(message: types.Message):
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        total_users = db.query(User).count()
        approved_users = db.query(User).filter(User.approved == True).count()
        pending_users = db.query(User).filter(User.approved == False, User.banned == False).count()
        banned_users = db.query(User).filter(User.banned == True).count()
        admin_count = db.query(User).filter(User.is_admin == True).count()
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="👥 View All Users", callback_data="admin_list_users")],
            [InlineKeyboardButton(text="⏳ Pending Approvals", callback_data="admin_pending")],
            [InlineKeyboardButton(text="🚫 Banned Users", callback_data="admin_banned")],
            [InlineKeyboardButton(text="📊 System Stats", callback_data="admin_stats")]
        ])
        
        admin_text = f"""
👑 **Admin Dashboard**

📊 **User Statistics:**
  • Total Users: {total_users}
  • Approved: {approved_users}
  • Pending: {pending_users}
  • Banned: {banned_users}
  • Admins: {admin_count}

**Admin Commands:**
/users - List all users
/approve <user_id> - Approve user
/ban <user_id> <reason> - Ban user
/unban <user_id> - Unban user
/user_stats <user_id> - Get user stats
/make_admin <user_id> - Grant admin access
/add_note <user_id> <note> - Add admin note
"""
        await message.answer(admin_text, reply_markup=keyboard)
    finally:
        db.close()


@dp.message(Command("users"))
async def cmd_users(message: types.Message):
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        users = db.query(User).order_by(User.created_at.desc()).limit(50).all()
        
        user_list = "👥 **All Users (Last 50):**\n\n"
        for user in users:
            status = "✅" if user.approved else "⏳"
            if user.banned:
                status = "🚫"
            if user.is_admin:
                status = "👑"
            
            user_list += f"{status} `{user.telegram_id}` - @{user.username or 'N/A'} ({user.first_name or 'N/A'})\n"
            if user.admin_notes:
                user_list += f"    📝 {user.admin_notes[:50]}\n"
        
        await message.answer(user_list)
    finally:
        db.close()


@dp.message(Command("approve"))
async def cmd_approve(message: types.Message):
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        parts = message.text.split()
        if len(parts) < 2:
            await message.answer("Usage: /approve <user_id>")
            return
        
        user_id = parts[1]
        user = db.query(User).filter(User.telegram_id == user_id).first()
        
        if not user:
            await message.answer(f"❌ User {user_id} not found.")
            return
        
        user.approved = True
        db.commit()
        
        await message.answer(f"✅ User {user_id} (@{user.username}) has been approved!")
        
        try:
            await bot.send_message(
                int(user_id),
                "✅ Your account has been approved! You can now use all bot features."
            )
        except:
            pass
    finally:
        db.close()


@dp.message(Command("ban"))
async def cmd_ban(message: types.Message):
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        parts = message.text.split(maxsplit=2)
        if len(parts) < 2:
            await message.answer("Usage: /ban <user_id> [reason]")
            return
        
        user_id = parts[1]
        reason = parts[2] if len(parts) > 2 else "No reason provided"
        
        user = db.query(User).filter(User.telegram_id == user_id).first()
        
        if not user:
            await message.answer(f"❌ User {user_id} not found.")
            return
        
        if user.is_admin:
            await message.answer("❌ Cannot ban admin users.")
            return
        
        user.banned = True
        user.admin_notes = f"Banned: {reason}"
        db.commit()
        
        await message.answer(f"🚫 User {user_id} (@{user.username}) has been banned.\nReason: {reason}")
        
        try:
            await bot.send_message(
                int(user_id),
                f"🚫 You have been banned from this bot.\nReason: {reason}"
            )
        except:
            pass
    finally:
        db.close()


@dp.message(Command("unban"))
async def cmd_unban(message: types.Message):
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        parts = message.text.split()
        if len(parts) < 2:
            await message.answer("Usage: /unban <user_id>")
            return
        
        user_id = parts[1]
        user = db.query(User).filter(User.telegram_id == user_id).first()
        
        if not user:
            await message.answer(f"❌ User {user_id} not found.")
            return
        
        user.banned = False
        user.approved = True
        db.commit()
        
        await message.answer(f"✅ User {user_id} (@{user.username}) has been unbanned and approved!")
        
        try:
            await bot.send_message(
                int(user_id),
                "✅ You have been unbanned! Welcome back."
            )
        except:
            pass
    finally:
        db.close()


@dp.message(Command("user_stats"))
async def cmd_user_stats(message: types.Message):
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        parts = message.text.split()
        if len(parts) < 2:
            await message.answer("Usage: /user_stats <user_id>")
            return
        
        user_id = parts[1]
        user = db.query(User).filter(User.telegram_id == user_id).first()
        
        if not user:
            await message.answer(f"❌ User {user_id} not found.")
            return
        
        total_trades = db.query(Trade).filter(Trade.user_id == user.id).count()
        open_trades = db.query(Trade).filter(Trade.user_id == user.id, Trade.status == "open").count()
        closed_trades = db.query(Trade).filter(Trade.user_id == user.id, Trade.status != "open").count()
        
        total_pnl = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.status != "open"
        ).with_entities(Trade.pnl).all()
        total_pnl_sum = sum([t[0] for t in total_pnl if t[0]]) if total_pnl else 0
        
        prefs = user.preferences
        auto_trading = "Enabled" if (prefs and prefs.auto_trading_enabled) else "Disabled"
        
        stats_text = f"""
📊 **User Stats: {user.telegram_id}**

👤 Username: @{user.username or 'N/A'}
📝 Name: {user.first_name or 'N/A'}
🔓 Status: {'✅ Approved' if user.approved else '⏳ Pending'}
🚫 Banned: {'Yes' if user.banned else 'No'}
👑 Admin: {'Yes' if user.is_admin else 'No'}
📅 Joined: {user.created_at.strftime('%Y-%m-%d %H:%M')}

**Trading Stats:**
  • Total Trades: {total_trades}
  • Open Trades: {open_trades}
  • Closed Trades: {closed_trades}
  • Total PnL: ${total_pnl_sum:.2f}
  • Auto-Trading: {auto_trading}

**Notes:** {user.admin_notes or 'None'}
"""
        await message.answer(stats_text)
    finally:
        db.close()


@dp.message(Command("make_admin"))
async def cmd_make_admin(message: types.Message):
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        parts = message.text.split()
        if len(parts) < 2:
            await message.answer("Usage: /make_admin <user_id>")
            return
        
        user_id = parts[1]
        user = db.query(User).filter(User.telegram_id == user_id).first()
        
        if not user:
            await message.answer(f"❌ User {user_id} not found.")
            return
        
        user.is_admin = True
        user.approved = True
        db.commit()
        
        await message.answer(f"👑 User {user_id} (@{user.username}) is now an admin!")
        
        try:
            await bot.send_message(
                int(user_id),
                "👑 You have been granted admin access!"
            )
        except:
            pass
    finally:
        db.close()


@dp.message(Command("add_note"))
async def cmd_add_note(message: types.Message):
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        parts = message.text.split(maxsplit=2)
        if len(parts) < 3:
            await message.answer("Usage: /add_note <user_id> <note>")
            return
        
        user_id = parts[1]
        note = parts[2]
        
        user = db.query(User).filter(User.telegram_id == user_id).first()
        
        if not user:
            await message.answer(f"❌ User {user_id} not found.")
            return
        
        user.admin_notes = note
        db.commit()
        
        await message.answer(f"📝 Note added for user {user_id}")
    finally:
        db.close()


@dp.callback_query(F.data == "admin_list_users")
async def handle_admin_list_users(callback: CallbackQuery):
    db = SessionLocal()
    try:
        if not is_admin(callback.from_user.id, db):
            await callback.answer("❌ You don't have admin access.", show_alert=True)
            return
        
        users = db.query(User).order_by(User.created_at.desc()).limit(20).all()
        
        user_list = "👥 **Recent Users (Last 20):**\n\n"
        for user in users:
            status = "✅" if user.approved else "⏳"
            if user.banned:
                status = "🚫"
            if user.is_admin:
                status = "👑"
            
            user_list += f"{status} `{user.telegram_id}` - @{user.username or 'N/A'}\n"
        
        user_list += "\nUse /users for full list with notes"
        
        await callback.message.edit_text(user_list)
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data == "admin_pending")
async def handle_admin_pending(callback: CallbackQuery):
    db = SessionLocal()
    try:
        if not is_admin(callback.from_user.id, db):
            await callback.answer("❌ You don't have admin access.", show_alert=True)
            return
        
        pending = db.query(User).filter(
            User.approved == False,
            User.banned == False,
            User.is_admin == False
        ).order_by(User.created_at.desc()).all()
        
        if not pending:
            await callback.message.edit_text("✅ No pending approvals!")
            await callback.answer()
            return
        
        pending_list = "⏳ **Pending Approvals:**\n\n"
        for user in pending:
            pending_list += f"`{user.telegram_id}` - @{user.username or 'N/A'} ({user.first_name or 'N/A'})\n"
            pending_list += f"  Joined: {user.created_at.strftime('%Y-%m-%d %H:%M')}\n\n"
        
        pending_list += "\nUse /approve <user_id> to approve"
        
        await callback.message.edit_text(pending_list)
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data == "admin_banned")
async def handle_admin_banned(callback: CallbackQuery):
    db = SessionLocal()
    try:
        if not is_admin(callback.from_user.id, db):
            await callback.answer("❌ You don't have admin access.", show_alert=True)
            return
        
        banned = db.query(User).filter(User.banned == True).order_by(User.created_at.desc()).all()
        
        if not banned:
            await callback.message.edit_text("✅ No banned users!")
            await callback.answer()
            return
        
        banned_list = "🚫 **Banned Users:**\n\n"
        for user in banned:
            banned_list += f"`{user.telegram_id}` - @{user.username or 'N/A'}\n"
            if user.admin_notes:
                banned_list += f"  Reason: {user.admin_notes}\n"
            banned_list += "\n"
        
        banned_list += "\nUse /unban <user_id> to unban"
        
        await callback.message.edit_text(banned_list)
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data == "admin_stats")
async def handle_admin_stats(callback: CallbackQuery):
    db = SessionLocal()
    try:
        if not is_admin(callback.from_user.id, db):
            await callback.answer("❌ You don't have admin access.", show_alert=True)
            return
        
        total_trades = db.query(Trade).count()
        open_trades = db.query(Trade).filter(Trade.status == "open").count()
        total_signals = db.query(Signal).count()
        
        recent_signals = db.query(Signal).order_by(Signal.created_at.desc()).limit(1).first()
        last_signal = recent_signals.created_at.strftime('%Y-%m-%d %H:%M') if recent_signals else 'None'
        
        stats_text = f"""
📊 **System Statistics**

**Signals:**
  • Total Signals: {total_signals}
  • Last Signal: {last_signal}

**Trades:**
  • Total Trades: {total_trades}
  • Open Trades: {open_trades}

**System:**
  • Bot: Online ✅
  • Scanner: Running 🔄
"""
        await callback.message.edit_text(stats_text)
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data == "toggle_risk_sizing")
async def handle_toggle_risk_sizing(callback: CallbackQuery):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            await callback.answer()
            return
        
        if user.preferences:
            user.preferences.risk_based_sizing = not user.preferences.risk_based_sizing
            db.commit()
            status = "enabled" if user.preferences.risk_based_sizing else "disabled"
            await callback.message.edit_text(f"""
✅ Risk-based sizing {status}

When enabled:
• MEDIUM risk signals use 70% position size
• LOW risk signals use 100% position size

This helps protect your account from higher risk trades.
""")
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data == "toggle_trailing")
async def handle_toggle_trailing(callback: CallbackQuery):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            await callback.answer()
            return
        
        if user.preferences:
            user.preferences.use_trailing_stop = not user.preferences.use_trailing_stop
            db.commit()
            status = "enabled" if user.preferences.use_trailing_stop else "disabled"
            await callback.message.edit_text(f"""
✅ Trailing stop {status}

When enabled, stop loss trails price by {user.preferences.trailing_stop_percent}% to lock in profits.

Note: This feature requires exchange support for trailing stops.
""")
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data == "toggle_breakeven")
async def handle_toggle_breakeven(callback: CallbackQuery):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            await callback.answer()
            return
        
        if user.preferences:
            user.preferences.use_breakeven_stop = not user.preferences.use_breakeven_stop
            db.commit()
            status = "enabled" if user.preferences.use_breakeven_stop else "disabled"
            await callback.message.edit_text(f"""
✅ Breakeven stop {status}

When enabled, stop loss automatically moves to entry price once the trade moves into profit.

This protects against turning a winning trade into a loss.
""")
        await callback.answer()
    finally:
        db.close()


async def broadcast_signal(signal_data: dict):
    db = SessionLocal()
    
    try:
        # Check for duplicate signals (same symbol + direction within last 4 hours)
        four_hours_ago = datetime.utcnow() - timedelta(hours=4)
        existing = db.query(Signal).filter(
            Signal.symbol == signal_data['symbol'],
            Signal.direction == signal_data['direction'],
            Signal.created_at >= four_hours_ago
        ).first()
        
        if existing:
            logger.info(f"Skipping duplicate {signal_data['direction']} signal for {signal_data['symbol']} (sent at {existing.created_at})")
            return
        
        signal = Signal(**signal_data)
        db.add(signal)
        db.commit()
        db.refresh(signal)
        
        logger.info(f"Broadcasting {signal.direction} signal for {signal.symbol}")
        
        # Calculate risk/reward ratio
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Calculate 10x leverage PnL
        tp_pnl = calculate_leverage_pnl(signal.entry_price, signal.take_profit, signal.direction, 10)
        sl_pnl = calculate_leverage_pnl(signal.entry_price, signal.stop_loss, signal.direction, 10)
        
        # Safe volume percentage calculation
        if signal.volume_avg and signal.volume_avg > 0:
            volume_pct = ((signal.volume / signal.volume_avg - 1) * 100)
            volume_text = f"{signal.volume:,.0f} ({'+' if signal.volume > signal.volume_avg else ''}{volume_pct:.1f}% avg)"
        else:
            volume_text = f"{signal.volume:,.0f}"
        
        # Risk level emoji
        risk_emoji = "🟢" if signal.risk_level == "LOW" else "🟡"
        
        signal_text = f"""
🚨 NEW {signal.direction} SIGNAL

📊 Symbol: {signal.symbol}
💰 Entry: ${signal.entry_price}
🛑 Stop Loss: ${signal.stop_loss}
🎯 Take Profit: ${signal.take_profit}

{risk_emoji} Risk Level: {signal.risk_level}
💎 Risk/Reward: 1:{rr_ratio:.2f}

📊 RSI: {signal.rsi}
📈 Volume: {volume_text}
⚡ ATR: ${signal.atr}

📈 Support: ${signal.support_level}
📉 Resistance: ${signal.resistance_level}

💰 10x Leverage PnL:
  ✅ TP Hit: {tp_pnl:+.2f}%
  ❌ SL Hit: {sl_pnl:+.2f}%

⏰ {signal.timeframe} | {signal.created_at.strftime('%H:%M:%S')}
"""
        
        await bot.send_message(settings.BROADCAST_CHAT_ID, signal_text)
        logger.info(f"Broadcast to channel successful")
        
        users = db.query(User).all()
        
        for user in users:
            # Check if user has access (not banned, approved or admin)
            has_access, _ = check_access(user)
            if not has_access:
                continue
            
            # Send DM alerts
            if user.preferences and user.preferences.dm_alerts:
                muted_symbols = user.preferences.get_muted_symbols_list()
                if signal.symbol not in muted_symbols:
                    try:
                        await bot.send_message(user.telegram_id, signal_text)
                        logger.info(f"Sent DM to user {user.telegram_id}")
                    except Exception as e:
                        logger.error(f"Failed to send to {user.telegram_id}: {e}")
            
            # Execute auto-trade if enabled
            if user.preferences and user.preferences.auto_trading_enabled:
                muted_symbols = user.preferences.get_muted_symbols_list()
                if signal.symbol not in muted_symbols:
                    await execute_auto_trade(signal_data, user, db)
    
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
        
        await asyncio.sleep(settings.SCAN_INTERVAL)


async def position_monitor():
    """Monitor open positions and notify when TP/SL is hit"""
    from app.services.mexc_trader import monitor_positions
    
    logger.info("Position monitor started")
    await asyncio.sleep(30)  # Wait 30s before first check
    
    while True:
        try:
            logger.info("Monitoring positions...")
            await monitor_positions()
        except Exception as e:
            logger.error(f"Position monitor error: {e}", exc_info=True)
        
        await asyncio.sleep(60)  # Check every 60 seconds


async def start_bot():
    logger.info("Starting Telegram bot...")
    asyncio.create_task(signal_scanner())
    asyncio.create_task(position_monitor())
    try:
        logger.info("Bot polling started")
        await dp.start_polling(bot)
    finally:
        await signal_generator.close()
