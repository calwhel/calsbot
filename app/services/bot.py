import asyncio
import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models import User, UserPreference, Trade, Signal
from app.services.signals import SignalGenerator
from app.services.news_signals import NewsSignalGenerator
from app.services.mexc_trader import execute_auto_trade
from app.utils.encryption import encrypt_api_key, decrypt_api_key

logger = logging.getLogger(__name__)

# FSM States for API setup
class MEXCSetup(StatesGroup):
    waiting_for_api_key = State()
    waiting_for_api_secret = State()

# FSM States for position size
class PositionSizeSetup(StatesGroup):
    waiting_for_size = State()

bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
signal_generator = SignalGenerator()
news_signal_generator = NewsSignalGenerator()


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
        
        # Get account overview data
        prefs = user.preferences
        
        # Auto-trading status
        autotrading_status = "🟢 Active" if prefs and prefs.auto_trading_enabled else "🔴 Inactive"
        mexc_connected = "✅ Connected" if prefs and prefs.mexc_api_key else "❌ Not Connected"
        
        # Get open positions
        open_trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.status == "open"
        ).count()
        
        # Get today's PnL
        now = datetime.utcnow()
        start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.closed_at >= start_today,
            Trade.status == "closed"
        ).all()
        
        today_pnl = sum(t.pnl for t in today_trades) if today_trades else 0
        today_pnl_pct = sum(t.pnl_percent for t in today_trades) if today_trades else 0
        
        # Security status
        emergency = "🚨 ACTIVE" if prefs and prefs.emergency_stop else "✅ Normal"
        
        dashboard_text = f"""
📊 <b>Trading Dashboard</b>

💼 <b>Account Overview</b>
━━━━━━━━━━━━━━━━━━━━
🤖 Auto-Trading: {autotrading_status}
🔑 MEXC API: {mexc_connected}
📈 Open Positions: {open_trades}
🛡️ Security: {emergency}

💰 <b>Today's Performance</b>
━━━━━━━━━━━━━━━━━━━━
Total PnL: ${today_pnl:+.2f} ({today_pnl_pct:+.2f}%)
Trades: {len(today_trades)}

<i>Select an option below:</i>
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="📊 PnL Today", callback_data="pnl_today"),
                InlineKeyboardButton(text="📈 PnL Week", callback_data="pnl_week")
            ],
            [
                InlineKeyboardButton(text="📅 PnL Month", callback_data="pnl_month"),
                InlineKeyboardButton(text="🔄 Active Positions", callback_data="active_trades")
            ],
            [
                InlineKeyboardButton(text="📡 Recent Signals", callback_data="recent_signals"),
                InlineKeyboardButton(text="🤖 Auto-Trading", callback_data="autotrading_menu")
            ],
            [
                InlineKeyboardButton(text="⚙️ Settings", callback_data="settings"),
                InlineKeyboardButton(text="🛡️ Security", callback_data="security_status")
            ],
            [
                InlineKeyboardButton(text="🆘 Support", callback_data="support_menu")
            ]
        ])
        
        await message.answer(dashboard_text, reply_markup=keyboard, parse_mode="HTML")
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
            period_emoji = "📊"
        elif period == "week":
            start_date = now - timedelta(days=7)
            period_emoji = "📈"
        else:
            start_date = now - timedelta(days=30)
            period_emoji = "📅"
        
        trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.closed_at >= start_date,
            Trade.status == "closed"
        ).all()
        
        if not trades:
            pnl_text = f"""
{period_emoji} <b>PnL Summary ({period.title()})</b>

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
            
            pnl_emoji = "🟢" if total_pnl > 0 else "🔴" if total_pnl < 0 else "⚪"
            
            pnl_text = f"""
{period_emoji} <b>PnL Summary ({period.title()})</b>
━━━━━━━━━━━━━━━━━━━━

{pnl_emoji} <b>Total PnL:</b> ${total_pnl:+.2f} ({total_pnl_pct:+.2f}%)
📈 <b>Total Trades:</b> {len(trades)}
✅ <b>Wins:</b> {len(winning_trades)} | ❌ <b>Losses:</b> {len(losing_trades)}
🎯 <b>Win Rate:</b> {win_rate:.1f}%

📊 <b>Statistics:</b>
  • Avg PnL/Trade: ${avg_pnl:.2f}
  • Avg Win: ${avg_win:.2f}
  • Avg Loss: ${avg_loss:.2f}
  
🏆 <b>Best Trade:</b> ${best_trade.pnl:.2f} ({best_trade.symbol})
📉 <b>Worst Trade:</b> ${worst_trade.pnl:.2f} ({worst_trade.symbol})
"""
        
        # Add back button
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
        ])
        
        await callback.message.answer(pnl_text, reply_markup=keyboard, parse_mode="HTML")
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
            trades_text = """
🔄 <b>Active Positions</b>

No active trades at the moment.

Use /autotrading_status to enable auto-trading and start taking trades automatically!
"""
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
            ])
            await callback.message.answer(trades_text, reply_markup=keyboard, parse_mode="HTML")
            await callback.answer()
            return
        
        # Try to get current prices for PnL calculation
        import ccxt
        exchange = ccxt.binance()
        
        trades_text = "🔄 <b>Active Positions</b>\n━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for i, trade in enumerate(trades, 1):
            direction_emoji = "🟢" if trade.direction == "LONG" else "🔴"
            
            # Try to get current price and calculate unrealized PnL
            try:
                ticker = exchange.fetch_ticker(trade.symbol)
                current_price = ticker['last']
                
                if trade.direction == "LONG":
                    pnl_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100 * 10  # 10x leverage
                else:
                    pnl_pct = ((trade.entry_price - current_price) / trade.entry_price) * 100 * 10
                
                pnl_emoji = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚪"
                
                trades_text += f"""
{i}. {direction_emoji} <b>{trade.symbol} {trade.direction}</b>
   Entry: ${trade.entry_price:.4f}
   Current: ${current_price:.4f}
   
   SL: ${trade.stop_loss:.4f} | TP: ${trade.take_profit:.4f}
   
   {pnl_emoji} <b>Unrealized PnL:</b> {pnl_pct:+.2f}% (10x)
━━━━━━━━━━━━━━━━━━━━
"""
            except:
                # If can't fetch price, show basic info
                trades_text += f"""
{i}. {direction_emoji} <b>{trade.symbol} {trade.direction}</b>
   Entry: ${trade.entry_price:.4f}
   SL: ${trade.stop_loss:.4f} | TP: ${trade.take_profit:.4f}
━━━━━━━━━━━━━━━━━━━━
"""
        
        # Add back button
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Refresh", callback_data="active_trades")],
            [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
        ])
        
        await callback.message.answer(trades_text, reply_markup=keyboard, parse_mode="HTML")
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
            signals_text = """
📡 <b>Recent Signals</b>

No signals generated yet.
Wait for the next market opportunity!
"""
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
            ])
            await callback.message.answer(signals_text, reply_markup=keyboard, parse_mode="HTML")
            await callback.answer()
            return
        
        signals_text = "📡 <b>Recent Signals</b>\n━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for i, signal in enumerate(signals, 1):
            direction_emoji = "🟢" if signal.direction == "LONG" else "🔴"
            
            # Determine signal type
            if signal.signal_type == "news":
                type_badge = "📰 News"
                risk_badge = f"Impact: {signal.impact_score}/10"
            else:
                type_badge = "📊 Technical"
                risk_badge = f"Risk: {signal.risk_level or 'MEDIUM'}"
            
            tp_pnl = calculate_leverage_pnl(signal.entry_price, signal.take_profit, signal.direction, 10)
            sl_pnl = calculate_leverage_pnl(signal.entry_price, signal.stop_loss, signal.direction, 10)
            
            signals_text += f"""
{i}. {direction_emoji} <b>{signal.symbol} {signal.direction}</b> ({type_badge})
   Entry: ${signal.entry_price:.4f}
   SL: ${signal.stop_loss:.4f} | TP: ${signal.take_profit:.4f}
   
   💰 10x Leverage:
   ✅ TP: {tp_pnl:+.2f}% | ❌ SL: {sl_pnl:+.2f}%
   
   🏷️ {risk_badge}
   ⏰ {signal.created_at.strftime('%m/%d %H:%M')}
━━━━━━━━━━━━━━━━━━━━
"""
        
        # Add back button
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
        ])
        
        await callback.message.answer(signals_text, reply_markup=keyboard, parse_mode="HTML")
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data == "settings")
async def handle_settings_callback(callback: CallbackQuery):
    await cmd_settings(callback.message)
    await callback.answer()


@dp.callback_query(F.data == "autotrading_menu")
async def handle_autotrading_menu(callback: CallbackQuery):
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
        
        prefs = user.preferences
        
        # Auto-trading status
        autotrading_status = "🟢 Enabled" if prefs and prefs.auto_trading_enabled else "🔴 Disabled"
        mexc_connected = prefs and prefs.mexc_api_key
        
        if mexc_connected:
            api_status = "✅ Connected"
            position_size = prefs.position_size_percent if prefs else 5
            max_positions = prefs.max_positions if prefs else 3
            
            autotrading_text = f"""
🤖 <b>Auto-Trading Status</b>
━━━━━━━━━━━━━━━━━━━━

🔑 <b>MEXC API:</b> {api_status}
🔄 <b>Status:</b> {autotrading_status}

⚙️ <b>Configuration:</b>
  • Position Size: {position_size}% of balance
  • Max Positions: {max_positions}

<i>Use the buttons below to manage auto-trading:</i>
"""
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔄 Toggle Auto-Trading", callback_data="toggle_autotrading_quick")],
                [InlineKeyboardButton(text="📊 Set Position Size", callback_data="set_position_size")],
                [InlineKeyboardButton(text="❌ Remove API Keys", callback_data="remove_api_confirm")],
                [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
            ])
        else:
            autotrading_text = """
🤖 <b>Auto-Trading Setup</b>
━━━━━━━━━━━━━━━━━━━━

❌ <b>MEXC API Not Connected</b>

To enable auto-trading:
1. Get your MEXC API keys
2. Use command: /set_mexc_api <api_key> <api_secret>
3. Enable only <b>futures trading</b> permission
4. <b>Do NOT enable withdrawals</b>

📚 Full setup guide: /autotrading_status
"""
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
            ])
        
        await callback.message.answer(autotrading_text, reply_markup=keyboard, parse_mode="HTML")
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data == "back_to_dashboard")
async def handle_back_to_dashboard(callback: CallbackQuery):
    # Reuse the dashboard command
    await cmd_dashboard(callback.message)
    await callback.answer()


@dp.callback_query(F.data == "toggle_autotrading_quick")
async def handle_toggle_autotrading_quick(callback: CallbackQuery):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found", show_alert=True)
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.answer(reason, show_alert=True)
            return
        
        if user.preferences:
            if not user.preferences.mexc_api_key or not user.preferences.mexc_api_secret:
                await callback.answer("❌ Please set your MEXC API keys first", show_alert=True)
                return
            
            user.preferences.auto_trading_enabled = not user.preferences.auto_trading_enabled
            db.commit()
            status = "✅ Enabled" if user.preferences.auto_trading_enabled else "🔴 Disabled"
            await callback.answer(f"Auto-trading {status}", show_alert=True)
            
            # Refresh the autotrading menu
            await handle_autotrading_menu(callback)
        else:
            await callback.answer("Settings not found", show_alert=True)
    finally:
        db.close()


@dp.callback_query(F.data == "remove_api_confirm")
async def handle_remove_api_confirm(callback: CallbackQuery):
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Yes, Remove", callback_data="remove_api_yes"),
            InlineKeyboardButton(text="❌ Cancel", callback_data="autotrading_menu")
        ]
    ])
    
    confirm_text = """
⚠️ <b>Confirm API Key Removal</b>

Are you sure you want to remove your MEXC API keys?

This will:
• Remove your encrypted API credentials
• Disable auto-trading
• Close no existing positions

<i>You can always reconnect later.</i>
"""
    
    await callback.message.answer(confirm_text, reply_markup=keyboard, parse_mode="HTML")
    await callback.answer()


@dp.callback_query(F.data == "remove_api_yes")
async def handle_remove_api_yes(callback: CallbackQuery):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found", show_alert=True)
            return
        
        if user.preferences:
            user.preferences.mexc_api_key = None
            user.preferences.mexc_api_secret = None
            user.preferences.auto_trading_enabled = False
            db.commit()
            
            await callback.message.answer("✅ MEXC API keys removed and auto-trading disabled")
            await callback.answer()
            
            # Go back to dashboard
            await handle_back_to_dashboard(callback)
        else:
            await callback.answer("Settings not found", show_alert=True)
    finally:
        db.close()


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


@dp.message(Command("support"))
async def cmd_support(message: types.Message):
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
        
        support_text = """
🆘 <b>Support Center</b>
━━━━━━━━━━━━━━━━━━━━

Welcome to the help center! Select a topic below to get started:

📚 <b>Available Help Topics:</b>
• Getting Started Guide
• Trading Signals Explained
• Auto-Trading Setup
• Troubleshooting
• Contact Admin

<i>Choose an option to continue:</i>
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="🚀 Getting Started", callback_data="help_getting_started"),
                InlineKeyboardButton(text="📊 Trading Signals", callback_data="help_signals")
            ],
            [
                InlineKeyboardButton(text="🤖 Auto-Trading", callback_data="help_autotrading"),
                InlineKeyboardButton(text="🔧 Troubleshooting", callback_data="help_troubleshooting")
            ],
            [
                InlineKeyboardButton(text="❓ FAQ", callback_data="help_faq"),
                InlineKeyboardButton(text="📞 Contact Admin", callback_data="help_contact_admin")
            ],
            [
                InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")
            ]
        ])
        
        await message.answer(support_text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "help_getting_started")
async def handle_help_getting_started(callback: CallbackQuery):
    help_text = """
🚀 <b>Getting Started Guide</b>
━━━━━━━━━━━━━━━━━━━━

<b>1. Understanding the Bot</b>
This bot provides cryptocurrency perpetual futures trading signals using:
• 📊 <b>Technical Analysis</b> - EMA crossovers with volume & RSI filters
• 📰 <b>News Analysis</b> - AI-powered sentiment analysis of breaking news
• ⏰ <b>Multi-Timeframe</b> - Scans both 1h and 4h charts

<b>2. Receiving Signals</b>
• All signals broadcast to the channel
• Enable DM alerts in /settings for private messages
• Each signal includes entry, stop loss, and take profit

<b>3. Using the Dashboard</b>
• /dashboard - View account overview & PnL
• Track open positions in real-time
• Monitor trading performance

<b>4. Key Commands</b>
• /dashboard - Trading dashboard
• /settings - Customize preferences
• /autotrading_status - Auto-trading info
• /support - Get help

<i>Next: Learn about auto-trading setup! 🤖</i>
"""
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🤖 Auto-Trading Setup", callback_data="help_autotrading")],
        [InlineKeyboardButton(text="◀️ Back to Support", callback_data="support_menu")]
    ])
    
    await callback.message.answer(help_text, reply_markup=keyboard, parse_mode="HTML")
    await callback.answer()


@dp.callback_query(F.data == "help_signals")
async def handle_help_signals(callback: CallbackQuery):
    help_text = """
📊 <b>Trading Signals Explained</b>
━━━━━━━━━━━━━━━━━━━━

<b>Signal Types:</b>

📈 <b>Technical Signals</b>
• Based on EMA (9/21/50) crossovers
• Volume confirmation required
• RSI filter prevents bad entries
• ATR-based dynamic stops
• Risk level: LOW/MEDIUM/HIGH

📰 <b>News Signals</b>
• AI analyzes breaking crypto news
• Only 9+/10 impact events
• 80%+ confidence required
• Sentiment-based direction

<b>Signal Components:</b>
• 💵 Entry Price - Where to enter
• 🛑 Stop Loss - Risk management
• 🎯 Take Profit - Profit target
• ⚖️ Risk/Impact Score

<b>10x Leverage Calculator:</b>
Each signal shows potential profit/loss with 10x leverage:
• ✅ TP Hit: +30% example
• ❌ SL Hit: -15% example

<b>Risk Management:</b>
• Only trade with funds you can afford to lose
• Use stop losses always
• Don't risk more than 1-2% per trade
• Diversify across multiple signals

<i>Enable auto-trading to execute trades automatically! 🤖</i>
"""
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🤖 Enable Auto-Trading", callback_data="help_autotrading")],
        [InlineKeyboardButton(text="◀️ Back to Support", callback_data="support_menu")]
    ])
    
    await callback.message.answer(help_text, reply_markup=keyboard, parse_mode="HTML")
    await callback.answer()


@dp.callback_query(F.data == "help_autotrading")
async def handle_help_autotrading(callback: CallbackQuery):
    help_text = """
🤖 <b>Auto-Trading Setup Guide</b>
━━━━━━━━━━━━━━━━━━━━

<b>Step 1: Get MEXC API Keys</b>
1. Go to MEXC.com → API Management
2. Create new API key
3. ⚠️ <b>IMPORTANT:</b> Enable <b>ONLY</b> Futures Trading
4. <b>DO NOT</b> enable withdrawals
5. Copy API Key & Secret

<b>Step 2: Connect to Bot</b>
• Use: /set_mexc_api
• Bot will guide you through setup
• Keys are encrypted & stored securely

<b>Step 3: Configure Settings</b>
• /toggle_autotrading - Enable/disable
• /risk_settings - Set risk management
• Position size: 1-100% of balance
• Max positions: Limit open trades

<b>Step 4: Security Features</b>
• 🛡️ Daily loss limits
• 🚨 Emergency stop button
• 📊 Real-time position tracking
• 🔒 Encrypted credentials

<b>How It Works:</b>
When a signal is generated:
1. Bot checks your risk settings
2. Calculates position size
3. Places market order on MEXC
4. Sets SL/TP automatically
5. Monitors position in real-time

<b>Commands:</b>
• /set_mexc_api - Connect account
• /toggle_autotrading - Enable/disable
• /autotrading_status - Check status
• /risk_settings - Configure risk

<i>Your funds are always under your control! ✅</i>
"""
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔑 Set Up Now", callback_data="autotrading_menu")],
        [InlineKeyboardButton(text="◀️ Back to Support", callback_data="support_menu")]
    ])
    
    await callback.message.answer(help_text, reply_markup=keyboard, parse_mode="HTML")
    await callback.answer()


@dp.callback_query(F.data == "help_troubleshooting")
async def handle_help_troubleshooting(callback: CallbackQuery):
    help_text = """
🔧 <b>Troubleshooting Guide</b>
━━━━━━━━━━━━━━━━━━━━

<b>Common Issues & Solutions:</b>

❌ <b>"No signals appearing"</b>
• Bot only sends LOW/MEDIUM risk signals
• News signals require 9+/10 impact
• Check /settings - ensure symbols not muted
• Enable DM alerts for private messages

❌ <b>"Auto-trading not working"</b>
• Check /autotrading_status
• Ensure API keys are set correctly
• Verify auto-trading is enabled
• Check if emergency stop is active
• Ensure you have USDT balance on MEXC

❌ <b>"Trades not executing"</b>
• Check your MEXC futures balance
• Verify API has futures trading permission
• Check max positions limit
• Review daily loss limits
• Use /risk_settings to adjust

❌ <b>"Can't access bot features"</b>
• If new user, admin approval needed
• Contact admin for approval
• Check if account is banned

<b>Reset Options:</b>
• /toggle_autotrading - Disable/re-enable
• /remove_mexc_api - Remove & reconnect
• Emergency stop: /risk_settings

<b>Still Having Issues?</b>
• Check /autotrading_status for diagnostics
• Review /risk_settings for security blocks
• Contact admin for help

<i>Most issues can be fixed by toggling auto-trading off/on!</i>
"""
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📞 Contact Admin", callback_data="help_contact_admin")],
        [InlineKeyboardButton(text="◀️ Back to Support", callback_data="support_menu")]
    ])
    
    await callback.message.answer(help_text, reply_markup=keyboard, parse_mode="HTML")
    await callback.answer()


@dp.callback_query(F.data == "help_faq")
async def handle_help_faq(callback: CallbackQuery):
    help_text = """
❓ <b>Frequently Asked Questions</b>
━━━━━━━━━━━━━━━━━━━━

<b>Q: Is the bot free to use?</b>
A: Yes! All signals are free. Optional auto-trading requires your own MEXC account.

<b>Q: How accurate are the signals?</b>
A: Signals use proven technical indicators and AI analysis, but no strategy is 100%. Always use risk management.

<b>Q: Can the bot withdraw my funds?</b>
A: NO! API keys have NO withdrawal permissions. You always control your funds.

<b>Q: What's the recommended position size?</b>
A: Start with 1-5% of your balance. Never risk more than you can afford to lose.

<b>Q: How many signals per day?</b>
A: Varies with market conditions. Quality over quantity - only high-probability setups.

<b>Q: Can I use other exchanges?</b>
A: Currently only MEXC is supported for auto-trading. Signals work for any exchange.

<b>Q: How do I stop auto-trading?</b>
A: Use /toggle_autotrading or emergency stop in /risk_settings

<b>Q: Are my API keys safe?</b>
A: Yes! Keys are encrypted using military-grade Fernet encryption and stored securely.

<b>Q: What timeframes are used?</b>
A: Bot scans both 1h and 4h charts for short and longer-term opportunities.

<b>Q: How do I get approved?</b>
A: Admins review new users. Contact admin for faster approval.

<i>Have more questions? Contact admin! 📞</i>
"""
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📞 Contact Admin", callback_data="help_contact_admin")],
        [InlineKeyboardButton(text="◀️ Back to Support", callback_data="support_menu")]
    ])
    
    await callback.message.answer(help_text, reply_markup=keyboard, parse_mode="HTML")
    await callback.answer()


@dp.callback_query(F.data == "help_contact_admin")
async def handle_help_contact_admin(callback: CallbackQuery):
    db = SessionLocal()
    
    try:
        # Get all admins
        admins = db.query(User).filter(User.is_admin == True).all()
        
        if not admins:
            await callback.message.answer("❌ No admins found. Please try again later.")
            await callback.answer()
            return
        
        admin_list = "\n".join([f"• @{admin.username or admin.first_name or 'Admin'} (ID: {admin.telegram_id})" for admin in admins[:3]])
        
        contact_text = f"""
📞 <b>Contact Admin</b>
━━━━━━━━━━━━━━━━━━━━

<b>Available Admins:</b>
{admin_list}

<b>What can admins help with?</b>
• ✅ Account approval
• 🚫 Ban/unban requests
• 🐛 Technical issues
• 💡 Feature suggestions
• 📊 Trading support

<b>How to contact:</b>
1. Click on admin username above
2. Send them a direct message
3. Explain your issue clearly

<b>Response Time:</b>
Admins typically respond within 24 hours.

<i>Be respectful and provide details about your issue!</i>
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Back to Support", callback_data="support_menu")]
        ])
        
        await callback.message.answer(contact_text, reply_markup=keyboard, parse_mode="HTML")
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data == "support_menu")
async def handle_support_menu(callback: CallbackQuery):
    # Reuse the support command
    await cmd_support(callback.message)
    await callback.answer()


@dp.message(Command("test_mexc"))
async def cmd_test_mexc(message: types.Message):
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
        
        if not user.preferences or not user.preferences.mexc_api_key or not user.preferences.mexc_api_secret:
            await message.answer("""
❌ <b>No MEXC API Keys Found</b>

You need to set up your MEXC API keys first.
Use /set_mexc_api to connect your account.
""", parse_mode="HTML")
            return
        
        # Test the API connection
        await message.answer("🔄 Testing MEXC API connection...\n\nPlease wait...")
        
        try:
            import ccxt
            
            # Decrypt API keys
            api_key = decrypt_api_key(user.preferences.mexc_api_key)
            api_secret = decrypt_api_key(user.preferences.mexc_api_secret)
            
            # Create exchange instance
            exchange = ccxt.mexc({
                'apiKey': api_key,
                'secret': api_secret,
                'options': {'defaultType': 'swap'}
            })
            
            # Test 1: Check API connection and permissions
            test_results = "🧪 <b>MEXC API Test Results</b>\n━━━━━━━━━━━━━━━━━━━━\n\n"
            
            # Fetch balance
            try:
                balance = exchange.fetch_balance()
                usdt_balance = balance.get('USDT', {}).get('free', 0)
                test_results += f"✅ <b>API Connection:</b> Success\n"
                test_results += f"✅ <b>Account Access:</b> Working\n"
                test_results += f"💰 <b>USDT Balance:</b> ${usdt_balance:.2f}\n\n"
            except Exception as e:
                test_results += f"❌ <b>Balance Check:</b> Failed\n"
                test_results += f"   Error: {str(e)[:100]}\n\n"
            
            # Test 2: Check markets access
            try:
                markets = exchange.load_markets()
                test_results += f"✅ <b>Market Data:</b> Accessible\n"
                test_results += f"   Available pairs: {len(markets)}\n\n"
            except Exception as e:
                test_results += f"❌ <b>Market Access:</b> Failed\n"
                test_results += f"   Error: {str(e)[:100]}\n\n"
            
            # Test 3: Check if futures trading is enabled
            try:
                # Try to fetch futures positions (read-only)
                positions = exchange.fetch_positions()
                test_results += f"✅ <b>Futures Trading:</b> Enabled\n"
                test_results += f"   Open positions: {len([p for p in positions if float(p.get('contracts', 0)) > 0])}\n\n"
            except Exception as e:
                test_results += f"⚠️ <b>Futures Trading:</b> Check permissions\n"
                test_results += f"   {str(e)[:100]}\n\n"
            
            # Auto-trading status
            autotrading_enabled = user.preferences.auto_trading_enabled
            test_results += f"📊 <b>Auto-Trading Status</b>\n━━━━━━━━━━━━━━━━━━━━\n"
            test_results += f"Status: {'🟢 Enabled' if autotrading_enabled else '🔴 Disabled'}\n"
            test_results += f"Position Size: {user.preferences.position_size_percent}% of balance\n"
            test_results += f"Max Positions: {user.preferences.max_positions}\n\n"
            
            # Next steps
            if usdt_balance > 0 and autotrading_enabled:
                test_results += "✅ <b>Ready for Auto-Trading!</b>\n\n"
                test_results += "The bot will automatically execute trades when signals are generated.\n\n"
                test_results += "To test immediately:\n"
                test_results += "• Wait for next signal (scans every 15 min)\n"
                test_results += "• Or use /force_scan (admin only) to trigger scan\n"
            elif usdt_balance == 0:
                test_results += "⚠️ <b>No USDT Balance</b>\n\n"
                test_results += "Deposit USDT to your MEXC futures account to start trading.\n"
            elif not autotrading_enabled:
                test_results += "⚠️ <b>Auto-Trading Disabled</b>\n\n"
                test_results += "Use /toggle_autotrading to enable auto-trading.\n"
            
            await message.answer(test_results, parse_mode="HTML")
            
        except Exception as e:
            error_msg = f"""
❌ <b>API Test Failed</b>

Error: {str(e)[:200]}

<b>Common issues:</b>
• API keys are incorrect
• Futures trading permission not enabled
• IP restriction on API key
• API key expired

<b>Solutions:</b>
1. Remove and re-add API keys: /remove_mexc_api
2. Check MEXC API settings
3. Ensure only futures trading is enabled
4. Disable IP restrictions
"""
            await message.answer(error_msg, parse_mode="HTML")
            
    finally:
        db.close()


@dp.message(Command("force_scan"))
async def cmd_force_scan(message: types.Message):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user or not user.is_admin:
            await message.answer("❌ This command is only available to admins.")
            return
        
        await message.answer("🔄 <b>Force Scanning for Signals...</b>\n\nPlease wait...", parse_mode="HTML")
        
        # Manually trigger signal generation
        technical_signals = await signal_generator.scan_for_signals()
        news_signals = await news_signal_generator.generate_signals()
        
        total_signals = len(technical_signals) + len(news_signals)
        
        result_msg = f"""
✅ <b>Scan Complete</b>

📊 Technical Signals: {len(technical_signals)}
📰 News Signals: {len(news_signals)}
📈 Total Signals: {total_signals}

{'Signals will be broadcast to channel and DMs.' if total_signals > 0 else 'No signals found at this moment.'}
"""
        
        await message.answer(result_msg, parse_mode="HTML")
        
        # Broadcast signals if any
        if technical_signals:
            for signal in technical_signals:
                # Save to database
                db_signal = Signal(
                    symbol=signal['symbol'],
                    direction=signal['direction'],
                    entry_price=signal['entry_price'],
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit'],
                    timeframe=signal['timeframe'],
                    risk_level=signal.get('risk_level', 'MEDIUM'),
                    signal_type='technical'
                )
                db.add(db_signal)
                db.commit()
                
                # Broadcast signal (this will trigger auto-trading)
                await broadcast_signal(signal, db)
        
        if news_signals:
            for signal in news_signals:
                # Save to database
                db_signal = Signal(
                    symbol=signal['symbol'],
                    direction=signal['direction'],
                    entry_price=signal['entry_price'],
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit'],
                    signal_type='news',
                    news_title=signal.get('title'),
                    news_url=signal.get('url'),
                    news_source=signal.get('source'),
                    sentiment=signal.get('sentiment'),
                    impact_score=signal.get('impact_score'),
                    confidence_score=signal.get('confidence')
                )
                db.add(db_signal)
                db.commit()
                
                # Broadcast signal
                await broadcast_signal(signal, db)
        
    finally:
        db.close()


@dp.message(Command("set_mexc_api"))
async def cmd_set_mexc_api(message: types.Message, state: FSMContext):
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
        
        await message.answer("""
🔑 **Let's connect your MEXC account!**

⚙️ First, get your API keys:
1. Go to MEXC → API Management
2. Create new API key
3. ⚠️ **IMPORTANT:** Enable **ONLY Futures Trading** permission
   • Do NOT enable withdrawals
   • Do NOT enable spot trading
4. Copy your API Key

🔒 **Security Notice:**
✅ You'll ALWAYS have access to your own funds
✅ API can only trade futures, cannot withdraw
✅ Keys are encrypted and stored securely

📝 Now, please send me your **API Key**:
        """)
        
        await state.set_state(MEXCSetup.waiting_for_api_key)
    finally:
        db.close()


@dp.message(MEXCSetup.waiting_for_api_key)
async def process_api_key(message: types.Message, state: FSMContext):
    # Save API key in state
    await state.update_data(api_key=message.text.strip())
    
    # Delete user's message for security
    try:
        await message.delete()
    except:
        pass
    
    await message.answer("""
✅ API Key received!

🔐 Now, please send me your **API Secret**:
    """)
    
    await state.set_state(MEXCSetup.waiting_for_api_secret)


@dp.message(MEXCSetup.waiting_for_api_secret)
async def process_api_secret(message: types.Message, state: FSMContext):
    db = SessionLocal()
    
    try:
        # Get saved API key from state
        data = await state.get_data()
        api_key = data.get('api_key')
        api_secret = message.text.strip()
        
        # Delete user's message for security
        try:
            await message.delete()
        except:
            pass
        
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("❌ Error: User not found. Please use /start first.")
            await state.clear()
            return
        
        # Query preferences directly
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        
        # Create preferences if they don't exist
        if not prefs:
            prefs = UserPreference(user_id=user.id)
            db.add(prefs)
            db.flush()
        
        # Encrypt and save API keys
        prefs.mexc_api_key = encrypt_api_key(api_key)
        prefs.mexc_api_secret = encrypt_api_key(api_secret)
        db.commit()
        
        await message.answer("""
✅ **MEXC API keys saved successfully!**

🔐 Your messages have been deleted for security.
🔒 Keys are encrypted and stored securely.

**Next steps:**
1️⃣ /toggle_autotrading - Enable auto-trading
2️⃣ /autotrading_status - Check your settings
3️⃣ /risk_settings - Configure risk management

You're all set! 🚀
        """)
        
        # Clear the state
        await state.clear()
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
        
        # Query preferences directly
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        
        if prefs:
            prefs.mexc_api_key = None
            prefs.mexc_api_secret = None
            prefs.auto_trading_enabled = False
            db.commit()
            await message.answer("✅ MEXC API keys removed and auto-trading disabled")
        else:
            await message.answer("⚠️ No settings found. Use /start first.")
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


@dp.callback_query(F.data == "set_position_size")
async def handle_set_position_size(callback: CallbackQuery, state: FSMContext):
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
        
        # Get current position size
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        current_size = prefs.position_size_percent if prefs else 5.0
        
        await callback.message.edit_text(f"""
💰 **Set Position Size**

Current: {current_size}% of balance per trade

📝 Send me the new percentage (1-100):

Examples:
• 5 = 5% of balance per trade
• 10 = 10% of balance per trade
• 2 = 2% of balance per trade

⚠️ Recommended: 2-5% for conservative trading
""")
        await state.set_state(PositionSizeSetup.waiting_for_size)
        await callback.answer()
    finally:
        db.close()


@dp.message(PositionSizeSetup.waiting_for_size)
async def process_position_size(message: types.Message, state: FSMContext):
    db = SessionLocal()
    
    try:
        # Validate input
        try:
            size = float(message.text.strip())
            if size < 1 or size > 100:
                await message.answer("⚠️ Position size must be between 1% and 100%. Please try again:")
                return
        except ValueError:
            await message.answer("⚠️ Please send a valid number (e.g., 5 for 5%). Try again:")
            return
        
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("❌ User not found.")
            await state.clear()
            return
        
        # Get or create preferences
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs:
            prefs = UserPreference(user_id=user.id)
            db.add(prefs)
            db.flush()
        
        # Update position size
        prefs.position_size_percent = size
        db.commit()
        
        await message.answer(f"""
✅ **Position size updated to {size}%**

Each auto-trade will use {size}% of your MEXC balance.

Example: With $1000 balance:
• Position value: ${1000 * (size/100):.2f}
• With 10x leverage: ${1000 * (size/100) * 10:.2f} exposure

Use /autotrading_status to view your full settings.
        """)
        
        await state.clear()
    finally:
        db.close()


async def broadcast_news_signal(news_signal: dict):
    """Broadcast news-based trading signal"""
    db = SessionLocal()
    
    try:
        # Check for duplicate news signals
        four_hours_ago = datetime.utcnow() - timedelta(hours=4)
        existing = db.query(Signal).filter(
            Signal.symbol == news_signal['symbol'],
            Signal.direction == news_signal['direction'],
            Signal.signal_type == 'news',
            Signal.created_at >= four_hours_ago
        ).first()
        
        if existing:
            logger.info(f"Skipping duplicate news signal for {news_signal['symbol']}")
            return
        
        # Create signal database record
        signal_data = {
            'symbol': news_signal['symbol'],
            'direction': news_signal['direction'],
            'entry_price': news_signal['entry_price'],
            'stop_loss': news_signal['stop_loss'],
            'take_profit': news_signal['take_profit'],
            'timeframe': news_signal['timeframe'],
            'signal_type': 'news',
            'news_title': news_signal['news_title'],
            'news_url': news_signal['news_url'],
            'news_source': news_signal['news_source'],
            'sentiment': news_signal['sentiment'],
            'impact_score': news_signal['impact_score'],
            'confidence': news_signal['confidence'],
            'reasoning': news_signal['reasoning']
        }
        
        signal = Signal(**signal_data)
        db.add(signal)
        db.commit()
        db.refresh(signal)
        
        # Format and broadcast message
        message = news_signal_generator.format_news_signal_message(news_signal)
        await bot.send_message(settings.BROADCAST_CHAT_ID, message)
        logger.info(f"News signal broadcast successful: {news_signal['direction']} {news_signal['symbol']}")
        
        # Send DM to users with news signals enabled
        users = db.query(User).filter(User.approved == True, User.banned == False).all()
        for user in users:
            if user.preferences and user.preferences.dm_alerts and user.preferences.news_signals_enabled:
                # Check user preferences for minimum impact/confidence
                if (news_signal['impact_score'] >= user.preferences.min_news_impact and 
                    news_signal['confidence'] >= user.preferences.min_news_confidence):
                    
                    # Check if symbol is muted
                    if news_signal['symbol'] not in user.preferences.get_muted_symbols_list():
                        try:
                            await bot.send_message(user.telegram_id, message)
                            
                            # Auto-trade if enabled
                            if user.preferences.auto_trading_enabled:
                                await execute_auto_trade(user, signal, db)
                        except Exception as e:
                            logger.error(f"Error sending news DM to {user.telegram_id}: {e}")
    
    except Exception as e:
        logger.error(f"Error broadcasting news signal: {e}")
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
        
        signal_data['signal_type'] = 'technical'
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
            
            # Scan for technical signals
            technical_signals = await signal_generator.scan_all_symbols()
            logger.info(f"Found {len(technical_signals)} technical signals")
            
            # Scan for news-based signals
            news_signals = await news_signal_generator.scan_news_for_signals(settings.SYMBOLS.split(','))
            logger.info(f"Found {len(news_signals)} news signals")
            
            # Broadcast technical signals
            for signal in technical_signals:
                await broadcast_signal(signal)
            
            # Broadcast news signals
            for news_signal in news_signals:
                await broadcast_news_signal(news_signal)
                
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
