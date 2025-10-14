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
from app.services.okx_trader import execute_okx_trade
from app.services.kucoin_trader import execute_kucoin_trade
from app.services.analytics import AnalyticsService
from app.utils.encryption import encrypt_api_key, decrypt_api_key

logger = logging.getLogger(__name__)

# FSM States for API setup
class MEXCSetup(StatesGroup):
    waiting_for_api_key = State()
    waiting_for_api_secret = State()

class OKXSetup(StatesGroup):
    waiting_for_api_key = State()
    waiting_for_api_secret = State()
    waiting_for_passphrase = State()

class KuCoinSetup(StatesGroup):
    waiting_for_api_key = State()
    waiting_for_api_secret = State()
    waiting_for_passphrase = State()

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


async def execute_trade_on_exchange(signal, user: User, db: Session):
    """Route trade execution to the appropriate exchange based on user preference"""
    try:
        prefs = user.preferences
        if not prefs:
            logger.warning(f"No preferences found for user {user.user_id}")
            return None
        
        # Check correlation filter before executing trade
        from app.services.risk_filters import check_correlation_filter
        allowed, reason = check_correlation_filter(signal.symbol, prefs, db)
        if not allowed:
            logger.info(f"Trade blocked by correlation filter for user {user.telegram_id}: {reason}")
            try:
                await bot.send_message(
                    user.telegram_id,
                    f"⚠️ <b>Trade Blocked - Correlation Filter</b>\n\n"
                    f"<b>Symbol:</b> {signal.symbol}\n"
                    f"<b>Direction:</b> {signal.direction}\n"
                    f"<b>Reason:</b> {reason}\n\n"
                    f"<i>Disable correlation filter in /settings to allow correlated trades</i>",
                    parse_mode="HTML"
                )
            except:
                pass
            return None
        
        # Determine which exchange to use
        preferred_exchange = prefs.preferred_exchange or "KuCoin"
        
        # Check if preferred exchange has credentials configured
        if preferred_exchange == "KuCoin":
            if prefs.kucoin_api_key and prefs.kucoin_api_secret and prefs.kucoin_passphrase:
                logger.info(f"Routing trade to KuCoin for user {user.user_id}")
                return await execute_kucoin_trade(signal, user, db)
            elif prefs.okx_api_key and prefs.okx_api_secret and prefs.okx_passphrase:
                logger.info(f"KuCoin not configured, falling back to OKX for user {user.user_id}")
                return await execute_okx_trade(signal, user, db)
            elif prefs.mexc_api_key and prefs.mexc_api_secret:
                logger.info(f"KuCoin not configured, falling back to MEXC for user {user.user_id}")
                return await execute_auto_trade(signal, user, db)
        elif preferred_exchange == "OKX":
            if prefs.okx_api_key and prefs.okx_api_secret and prefs.okx_passphrase:
                logger.info(f"Routing trade to OKX for user {user.user_id}")
                return await execute_okx_trade(signal, user, db)
            elif prefs.kucoin_api_key and prefs.kucoin_api_secret and prefs.kucoin_passphrase:
                logger.info(f"OKX not configured, falling back to KuCoin for user {user.user_id}")
                return await execute_kucoin_trade(signal, user, db)
            elif prefs.mexc_api_key and prefs.mexc_api_secret:
                logger.info(f"OKX not configured, falling back to MEXC for user {user.user_id}")
                return await execute_auto_trade(signal, user, db)
        else:  # MEXC or fallback
            if prefs.mexc_api_key and prefs.mexc_api_secret:
                logger.info(f"Routing trade to MEXC for user {user.user_id}")
                return await execute_auto_trade(signal, user, db)
            elif prefs.kucoin_api_key and prefs.kucoin_api_secret and prefs.kucoin_passphrase:
                logger.info(f"MEXC not configured, falling back to KuCoin for user {user.user_id}")
                return await execute_kucoin_trade(signal, user, db)
            elif prefs.okx_api_key and prefs.okx_api_secret and prefs.okx_passphrase:
                logger.info(f"MEXC not configured, falling back to OKX for user {user.user_id}")
                return await execute_okx_trade(signal, user, db)
        
        logger.warning(f"No exchange configured for user {user.user_id}")
        return None
        
    except Exception as e:
        logger.error(f"Error routing trade: {e}")
        return None


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
        
        prefs = user.preferences
        
        # Get trading stats
        total_trades = db.query(Trade).filter(Trade.user_id == user.id).count()
        open_positions = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.status == 'open'
        ).count()
        
        # Calculate today's PnL
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.status.in_(['closed', 'stopped']),
            Trade.closed_at >= today_start
        ).all()
        today_pnl = sum(trade.pnl or 0 for trade in today_trades)
        
        # Auto-trading status - check all exchanges
        mexc_connected = prefs and prefs.mexc_api_key and prefs.mexc_api_secret
        okx_connected = prefs and prefs.okx_api_key and prefs.okx_api_secret and prefs.okx_passphrase
        kucoin_connected = prefs and prefs.kucoin_api_key and prefs.kucoin_api_secret and prefs.kucoin_passphrase
        auto_enabled = prefs and prefs.auto_trading_enabled
        
        # Auto-trading is only ACTIVE if both enabled AND at least one exchange connected
        is_active = auto_enabled and (mexc_connected or okx_connected or kucoin_connected)
        autotrading_emoji = "🟢" if is_active else "🔴"
        autotrading_status = "ACTIVE" if is_active else "INACTIVE"
        
        # Show which exchange is active
        active_exchange = prefs.preferred_exchange if prefs else "KuCoin"
        exchange_status = f"{active_exchange} (✅ Connected)" if is_active else "No Exchange Connected"
        
        # Position sizing info
        position_size = f"{prefs.position_size_percent:.0f}%" if prefs else "10%"
        leverage = f"{prefs.user_leverage}x" if prefs else "10x"
        
        # Trading mode
        trading_mode = "📄 Paper Trading" if prefs and prefs.paper_trading_mode else "💰 Live Trading"
        
        welcome_text = f"""
╔══════════════════════════╗
   <b>🚀 AI FUTURES SIGNALS</b>
╚══════════════════════════╝

👤 <b>Account Overview</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
{autotrading_emoji} Auto-Trading: <b>{autotrading_status}</b>
🔗 Exchange: {exchange_status}
{trading_mode}

📊 <b>Trading Statistics</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 Open Positions: <b>{open_positions}</b>
📝 Total Trades: <b>{total_trades}</b>
💵 Today's PnL: <b>${today_pnl:+.2f}</b>

⚙️ <b>Risk Configuration</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
💎 Position Size: <b>{position_size}</b>
📊 Leverage: <b>{leverage}</b>
🎯 Max Positions: <b>{prefs.max_positions if prefs else 3}</b>

<i>Powered by AI-driven EMA crossover strategy with multi-timeframe analysis and real-time market monitoring.</i>
"""
        
        # Create inline keyboard with quick actions
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="📊 Dashboard", callback_data="dashboard"),
                InlineKeyboardButton(text="⚡ Quick Trade", callback_data="recent_signals")
            ],
            [
                InlineKeyboardButton(text="🤖 Auto-Trading", callback_data="autotrading_menu"),
                InlineKeyboardButton(text="⚙️ Settings", callback_data="settings_menu")
            ],
            [
                InlineKeyboardButton(text="📈 Performance", callback_data="performance_menu"),
                InlineKeyboardButton(text="❓ Help", callback_data="help_menu")
            ]
        ])
        
        await message.answer(welcome_text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "dashboard")
async def handle_dashboard_button(callback: CallbackQuery):
    """Handle dashboard button from /start menu"""
    await callback.answer()
    # Trigger the dashboard command
    await cmd_dashboard(callback.message)


@dp.callback_query(F.data == "settings_menu")
async def handle_settings_menu_button(callback: CallbackQuery):
    """Handle settings menu button"""
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("Please use /start first")
            return
        
        prefs = user.preferences
        
        settings_text = f"""
⚙️ <b>Settings Menu</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

Current Configuration:
• Position Size: {prefs.position_size_percent if prefs else 10}%
• Leverage: {prefs.user_leverage if prefs else 10}x
• Max Positions: {prefs.max_positions if prefs else 3}
• DM Alerts: {'✅ Enabled' if prefs and prefs.dm_alerts else '❌ Disabled'}
• Paper Trading: {'✅ Enabled' if prefs and prefs.paper_trading_mode else '❌ Disabled'}

Use the buttons below to adjust your settings:
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📊 Position Size", callback_data="edit_position_size")],
            [InlineKeyboardButton(text="⚡ Leverage", callback_data="edit_leverage")],
            [InlineKeyboardButton(text="🔔 Notifications", callback_data="edit_notifications")],
            [InlineKeyboardButton(text="🔙 Back", callback_data="back_to_start")]
        ])
        
        await callback.message.edit_text(settings_text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "performance_menu")
async def handle_performance_menu_button(callback: CallbackQuery):
    """Handle performance menu button"""
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("Please use /start first")
            return
        
        # Get performance stats
        all_trades = db.query(Trade).filter(Trade.user_id == user.id).all()
        closed_trades = [t for t in all_trades if t.status in ['closed', 'stopped']]
        
        total_pnl = sum(t.pnl or 0 for t in closed_trades)
        winning_trades = len([t for t in closed_trades if (t.pnl or 0) > 0])
        losing_trades = len([t for t in closed_trades if (t.pnl or 0) < 0])
        win_rate = (winning_trades / len(closed_trades) * 100) if closed_trades else 0
        
        performance_text = f"""
📈 <b>Performance Analytics</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 <b>Overall Statistics</b>
Total Trades: {len(closed_trades)}
Win Rate: {win_rate:.1f}%
Total PnL: ${total_pnl:+.2f}

✅ Winning Trades: {winning_trades}
❌ Losing Trades: {losing_trades}

📈 Open Positions: {len([t for t in all_trades if t.status == 'open'])}

<i>Keep trading to build your performance history!</i>
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📊 View All Trades", callback_data="view_all_pnl")],
            [InlineKeyboardButton(text="🔙 Back", callback_data="back_to_start")]
        ])
        
        await callback.message.edit_text(performance_text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "help_menu")  
async def handle_help_menu_button(callback: CallbackQuery):
    """Handle help menu button"""
    await callback.answer()
    
    help_text = """
❓ <b>Help & Support</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>Quick Commands:</b>
/start - Main menu
/dashboard - Trading dashboard
/settings - Configure settings
/set_mexc_api - Connect MEXC

<b>Auto-Trading:</b>
• Connect your MEXC API
• Set position size & leverage
• Bot trades automatically on signals

<b>Safety Features:</b>
• Emergency stop available
• Daily loss limits
• Max drawdown protection
• Trailing stops

<b>Need Help?</b>
Contact: @YourSupport
"""
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📚 Getting Started", callback_data="help_getting_started")],
        [InlineKeyboardButton(text="🤖 Auto-Trading Guide", callback_data="help_autotrading")],
        [InlineKeyboardButton(text="🔙 Back", callback_data="back_to_start")]
    ])
    
    await callback.message.edit_text(help_text, reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data == "back_to_start")
async def handle_back_to_start(callback: CallbackQuery):
    """Return to main /start menu"""
    await callback.answer()
    # Re-trigger the start command
    await cmd_start(callback.message)


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
        
        # Auto-trading status - must have both enabled AND API connected
        mexc_api_connected = prefs and prefs.mexc_api_key and prefs.mexc_api_secret
        auto_enabled = prefs and prefs.auto_trading_enabled
        
        # Auto-trading is only Active if both enabled AND API connected
        is_active = auto_enabled and mexc_api_connected
        autotrading_status = "🟢 Active" if is_active else "🔴 Inactive"
        mexc_connected = "✅ Connected" if mexc_api_connected else "❌ Not Connected"
        
        # Get open positions and calculate LIVE unrealized PnL
        open_trades_list = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.status == "open"
        ).all()
        
        open_trades_count = len(open_trades_list)
        
        # Calculate LIVE unrealized PnL for all open positions
        total_unrealized_pnl = 0
        total_unrealized_pnl_pct = 0
        
        if open_trades_list:
            exchange = ccxt.kucoin()
            leverage = prefs.user_leverage if prefs else 10
            
            for trade in open_trades_list:
                try:
                    ticker = exchange.fetch_ticker(trade.symbol)
                    current_price = ticker['last']
                    
                    # Calculate PnL percentage with leverage
                    if trade.direction == "LONG":
                        pnl_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100 * leverage
                    else:
                        pnl_pct = ((trade.entry_price - current_price) / trade.entry_price) * 100 * leverage
                    
                    # Calculate PnL in USD
                    remaining_size = trade.remaining_size if trade.remaining_size > 0 else trade.position_size
                    pnl_usd = (remaining_size * pnl_pct) / 100
                    
                    total_unrealized_pnl += pnl_usd
                    total_unrealized_pnl_pct += pnl_pct
                except:
                    pass
        
        # Get today's realized PnL
        now = datetime.utcnow()
        start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.closed_at >= start_today,
            Trade.status == "closed"
        ).all()
        
        today_pnl = sum(t.pnl for t in today_trades) if today_trades else 0
        today_pnl_pct = sum(t.pnl_percent for t in today_trades) if today_trades else 0
        
        # Combined PnL (realized + unrealized)
        combined_pnl = today_pnl + total_unrealized_pnl
        combined_pnl_emoji = "🟢" if combined_pnl > 0 else "🔴" if combined_pnl < 0 else "⚪"
        
        # Security status
        emergency = "🚨 ACTIVE" if prefs and prefs.emergency_stop else "✅ Normal"
        
        # Build live PnL section
        live_pnl_section = ""
        if open_trades_count > 0:
            unrealized_emoji = "🟢" if total_unrealized_pnl > 0 else "🔴" if total_unrealized_pnl < 0 else "⚪"
            live_pnl_section = f"""
💹 <b>LIVE Unrealized PnL</b>
━━━━━━━━━━━━━━━━━━━━
{unrealized_emoji} ${total_unrealized_pnl:+.2f} ({total_unrealized_pnl_pct:+.2f}%)
📊 {open_trades_count} open position{'s' if open_trades_count != 1 else ''}
"""
        
        dashboard_text = f"""
📊 <b>Trading Dashboard</b>

💼 <b>Account Overview</b>
━━━━━━━━━━━━━━━━━━━━
🤖 Auto-Trading: {autotrading_status}
🔑 MEXC API: {mexc_connected}
🛡️ Security: {emergency}
{live_pnl_section}
💰 <b>Today's Performance</b>
━━━━━━━━━━━━━━━━━━━━
Realized PnL: ${today_pnl:+.2f}
{combined_pnl_emoji} <b>Total Today:</b> ${combined_pnl:+.2f}
Closed Trades: {len(today_trades)}

<i>Dashboard updates with live market prices</i>
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
                
                # Build TP status text
                tp_text = ""
                if trade.take_profit_1 and trade.take_profit_2 and trade.take_profit_3:
                    tp1_status = "✅" if trade.tp1_hit else "⏳"
                    tp2_status = "✅" if trade.tp2_hit else "⏳"
                    tp3_status = "✅" if trade.tp3_hit else "⏳"
                    
                    prefs = user.preferences or UserPreference()
                    tp_text = f"""   
   🎯 Take Profit Levels:
   {tp1_status} TP1: ${trade.take_profit_1:.4f} ({prefs.tp1_percent}%)
   {tp2_status} TP2: ${trade.take_profit_2:.4f} ({prefs.tp2_percent}%)
   {tp3_status} TP3: ${trade.take_profit_3:.4f} ({prefs.tp3_percent}%)"""
                else:
                    tp_text = f"\n   🎯 TP: ${trade.take_profit:.4f}"
                
                # Calculate PnL including partial closes
                if trade.pnl != 0:
                    realized_text = f"\n   💵 <b>Realized PnL:</b> ${trade.pnl:.2f}"
                else:
                    realized_text = ""
                
                trades_text += f"""
{i}. {direction_emoji} <b>{trade.symbol} {trade.direction}</b>
   Entry: ${trade.entry_price:.4f}
   Current: ${current_price:.4f}
   
   🛑 SL: ${trade.stop_loss:.4f}{tp_text}
   
   {pnl_emoji} <b>Unrealized PnL:</b> {pnl_pct:+.2f}% (10x){realized_text}
━━━━━━━━━━━━━━━━━━━━
"""
            except:
                # If can't fetch price, show basic info with TP levels if available
                tp_text = ""
                if trade.take_profit_1 and trade.take_profit_2 and trade.take_profit_3:
                    tp1_status = "✅" if trade.tp1_hit else "⏳"
                    tp2_status = "✅" if trade.tp2_hit else "⏳"
                    tp3_status = "✅" if trade.tp3_hit else "⏳"
                    
                    prefs = user.preferences or UserPreference()
                    tp_text = f"""
   🎯 Take Profit Levels:
   {tp1_status} TP1: ${trade.take_profit_1:.4f} ({prefs.tp1_percent}%)
   {tp2_status} TP2: ${trade.take_profit_2:.4f} ({prefs.tp2_percent}%)
   {tp3_status} TP3: ${trade.take_profit_3:.4f} ({prefs.tp3_percent}%)"""
                else:
                    tp_text = f"\n   🎯 TP: ${trade.take_profit:.4f}"
                
                trades_text += f"""
{i}. {direction_emoji} <b>{trade.symbol} {trade.direction}</b>
   Entry: ${trade.entry_price:.4f}
   🛑 SL: ${trade.stop_loss:.4f}{tp_text}
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
        
        # Explicitly query preferences to ensure fresh data
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        
        # Auto-trading status
        autotrading_status = "🟢 Enabled" if prefs and prefs.auto_trading_enabled else "🔴 Disabled"
        mexc_connected = prefs and prefs.mexc_api_key and prefs.mexc_api_secret
        
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

❌ <b>API Not Connected</b>

To enable auto-trading, use one of these commands:
  • /set_kucoin_api (Recommended)
  • /set_okx_api
  • /set_mexc_api

<b>Important:</b>
  • Enable only <b>futures trading</b> permission
  • <b>Do NOT enable withdrawals</b>

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
        
        # Explicitly query preferences to ensure fresh data
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        
        if prefs:
            if not prefs.mexc_api_key or not prefs.mexc_api_secret:
                await callback.answer("❌ Please set your MEXC API keys first", show_alert=True)
                return
            
            prefs.auto_trading_enabled = not prefs.auto_trading_enabled
            db.commit()
            status = "✅ Enabled" if prefs.auto_trading_enabled else "🔴 Disabled"
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
⚙️ <b>Your Settings</b>

<b>📊 General</b>
• Muted Symbols: {muted_str}
• Default PnL Period: {prefs.default_pnl_period}
• DM Alerts: {"Enabled" if prefs.dm_alerts else "Disabled"}

<b>🛡️ Risk Management</b>
• Correlation Filter: {"Enabled" if prefs.correlation_filter_enabled else "Disabled"}
• Max Correlated Positions: {prefs.max_correlated_positions}
• Funding Rate Alerts: {"Enabled" if prefs.funding_rate_alerts_enabled else "Disabled"}
• Funding Alert Threshold: {prefs.funding_rate_threshold}%

<b>Commands:</b>
/mute <symbol> - Mute a symbol
/unmute <symbol> - Unmute a symbol
/set_pnl <today/week/month> - Set default PnL period
/toggle_alerts - Enable/Disable DM alerts
/toggle_correlation - Enable/Disable correlation filter
/toggle_funding_alerts - Enable/Disable funding alerts
/set_funding_threshold <0.1-1.0> - Set funding alert %
"""
        
        await message.answer(settings_text, parse_mode="HTML")
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


@dp.message(Command("toggle_correlation"))
async def cmd_toggle_correlation(message: types.Message):
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
            user.preferences.correlation_filter_enabled = not user.preferences.correlation_filter_enabled
            db.commit()
            status = "enabled" if user.preferences.correlation_filter_enabled else "disabled"
            await message.answer(
                f"✅ <b>Correlation filter {status}</b>\n\n"
                f"This filter prevents opening multiple correlated positions (e.g., BTC + ETH at same time).\n"
                f"Max correlated positions: {user.preferences.max_correlated_positions}",
                parse_mode="HTML"
            )
        else:
            await message.answer("Settings not found. Use /start first.")
    finally:
        db.close()


@dp.message(Command("toggle_funding_alerts"))
async def cmd_toggle_funding_alerts(message: types.Message):
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
            user.preferences.funding_rate_alerts_enabled = not user.preferences.funding_rate_alerts_enabled
            db.commit()
            status = "enabled" if user.preferences.funding_rate_alerts_enabled else "disabled"
            await message.answer(
                f"✅ <b>Funding rate alerts {status}</b>\n\n"
                f"Get notified when funding rates are extreme (>{user.preferences.funding_rate_threshold}%).\n"
                f"Spot arbitrage opportunities when longs/shorts are overleveraged.",
                parse_mode="HTML"
            )
        else:
            await message.answer("Settings not found. Use /start first.")
    finally:
        db.close()


@dp.message(Command("set_funding_threshold"))
async def cmd_set_funding_threshold(message: types.Message):
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
        
        try:
            args = message.text.split()
            if len(args) < 2:
                await message.answer(
                    "❌ Usage: /set_funding_threshold <0.1-1.0>\n"
                    "Example: /set_funding_threshold 0.15"
                )
                return
            
            threshold = float(args[1])
            if threshold < 0.05 or threshold > 1.0:
                await message.answer("❌ Threshold must be between 0.05 and 1.0")
                return
            
            user.preferences.funding_rate_threshold = threshold
            db.commit()
            await message.answer(
                f"✅ <b>Funding alert threshold set to {threshold}%</b>\n\n"
                f"You'll be alerted when funding rates exceed {threshold}% (8hr rate).\n"
                f"Daily equivalent: {threshold * 3}%",
                parse_mode="HTML"
            )
        except ValueError:
            await message.answer("❌ Invalid number. Use: /set_funding_threshold 0.15")
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


@dp.message(Command("analytics"))
async def cmd_analytics(message: types.Message):
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
        
        # Parse time period (default: 30 days)
        args = message.text.split()
        days = 30
        if len(args) > 1 and args[1].isdigit():
            days = int(args[1])
        
        # Get performance stats
        stats = AnalyticsService.get_performance_stats(db, days)
        symbol_perf = AnalyticsService.get_symbol_performance(db, days)
        signal_type_perf = AnalyticsService.get_signal_type_performance(db, days)
        timeframe_perf = AnalyticsService.get_timeframe_performance(db, days)
        
        # Build analytics message
        analytics_text = f"""
📊 <b>Signal Performance Analytics</b>
━━━━━━━━━━━━━━━━━━━━
<i>Last {days} days</i>

<b>📈 Overall Performance</b>
• Total Signals: {stats['total_signals']}
• Won: {stats['won']} ✅ | Lost: {stats['lost']} ❌ | BE: {stats['breakeven']} ➖
• Win Rate: {stats['win_rate']:.1f}%
• Avg PnL: {stats['avg_pnl']:+.2f}%
• Total PnL: ${stats['total_pnl']:,.2f}

<b>🎯 Signal Type Performance</b>
📊 Technical: {signal_type_perf['technical']['count']} signals
   Win Rate: {signal_type_perf['technical']['win_rate']:.1f}%
   Avg PnL: {signal_type_perf['technical']['avg_pnl']:+.2f}%

📰 News: {signal_type_perf['news']['count']} signals
   Win Rate: {signal_type_perf['news']['win_rate']:.1f}%
   Avg PnL: {signal_type_perf['news']['avg_pnl']:+.2f}%
"""
        
        # Add best/worst signals
        if stats['best_signal']:
            analytics_text += f"""
<b>🏆 Best Signal</b>
{stats['best_signal']['symbol']} {stats['best_signal']['direction']}
PnL: {stats['best_signal']['pnl']:+.2f}% ({stats['best_signal']['type']})
"""
        
        if stats['worst_signal']:
            analytics_text += f"""
<b>📉 Worst Signal</b>
{stats['worst_signal']['symbol']} {stats['worst_signal']['direction']}
PnL: {stats['worst_signal']['pnl']:+.2f}% ({stats['worst_signal']['type']})
"""
        
        # Add top symbols
        if symbol_perf:
            analytics_text += "\n<b>💎 Top Symbols by Avg PnL</b>\n"
            for i, symbol in enumerate(symbol_perf[:5], 1):
                analytics_text += f"{i}. {symbol['symbol']}: {symbol['avg_pnl']:+.2f}% ({symbol['count']} signals)\n"
        
        # Add timeframe performance
        if timeframe_perf:
            analytics_text += "\n<b>⏰ Timeframe Performance</b>\n"
            for tf in timeframe_perf:
                analytics_text += f"{tf['timeframe']}: {tf['avg_pnl']:+.2f}% ({tf['count']} signals)\n"
        
        analytics_text += f"""
━━━━━━━━━━━━━━━━━━━━
💡 Use /analytics [days] to change period
Example: /analytics 7 (last 7 days)
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="📊 7 Days", callback_data="analytics_7"),
                InlineKeyboardButton(text="📊 30 Days", callback_data="analytics_30")
            ],
            [
                InlineKeyboardButton(text="📊 90 Days", callback_data="analytics_90"),
                InlineKeyboardButton(text="📊 All Time", callback_data="analytics_365")
            ],
            [
                InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")
            ]
        ])
        
        await message.answer(analytics_text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data.startswith("analytics_"))
async def handle_analytics_period(callback: CallbackQuery):
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found", show_alert=True)
            return
        
        # Extract days from callback data
        days_map = {
            "analytics_7": 7,
            "analytics_30": 30,
            "analytics_90": 90,
            "analytics_365": 365
        }
        days = days_map.get(callback.data, 30)
        
        # Get performance stats
        stats = AnalyticsService.get_performance_stats(db, days)
        symbol_perf = AnalyticsService.get_symbol_performance(db, days)
        signal_type_perf = AnalyticsService.get_signal_type_performance(db, days)
        timeframe_perf = AnalyticsService.get_timeframe_performance(db, days)
        
        # Build analytics message
        period_label = "All Time" if days == 365 else f"Last {days} days"
        analytics_text = f"""
📊 <b>Signal Performance Analytics</b>
━━━━━━━━━━━━━━━━━━━━
<i>{period_label}</i>

<b>📈 Overall Performance</b>
• Total Signals: {stats['total_signals']}
• Won: {stats['won']} ✅ | Lost: {stats['lost']} ❌ | BE: {stats['breakeven']} ➖
• Win Rate: {stats['win_rate']:.1f}%
• Avg PnL: {stats['avg_pnl']:+.2f}%
• Total PnL: ${stats['total_pnl']:,.2f}

<b>🎯 Signal Type Performance</b>
📊 Technical: {signal_type_perf['technical']['count']} signals
   Win Rate: {signal_type_perf['technical']['win_rate']:.1f}%
   Avg PnL: {signal_type_perf['technical']['avg_pnl']:+.2f}%

📰 News: {signal_type_perf['news']['count']} signals
   Win Rate: {signal_type_perf['news']['win_rate']:.1f}%
   Avg PnL: {signal_type_perf['news']['avg_pnl']:+.2f}%
"""
        
        # Add best/worst signals
        if stats['best_signal']:
            analytics_text += f"""
<b>🏆 Best Signal</b>
{stats['best_signal']['symbol']} {stats['best_signal']['direction']}
PnL: {stats['best_signal']['pnl']:+.2f}% ({stats['best_signal']['type']})
"""
        
        if stats['worst_signal']:
            analytics_text += f"""
<b>📉 Worst Signal</b>
{stats['worst_signal']['symbol']} {stats['worst_signal']['direction']}
PnL: {stats['worst_signal']['pnl']:+.2f}% ({stats['worst_signal']['type']})
"""
        
        # Add top symbols
        if symbol_perf:
            analytics_text += "\n<b>💎 Top Symbols by Avg PnL</b>\n"
            for i, symbol in enumerate(symbol_perf[:5], 1):
                analytics_text += f"{i}. {symbol['symbol']}: {symbol['avg_pnl']:+.2f}% ({symbol['count']} signals)\n"
        
        # Add timeframe performance
        if timeframe_perf:
            analytics_text += "\n<b>⏰ Timeframe Performance</b>\n"
            for tf in timeframe_perf:
                analytics_text += f"{tf['timeframe']}: {tf['avg_pnl']:+.2f}% ({tf['count']} signals)\n"
        
        analytics_text += """
━━━━━━━━━━━━━━━━━━━━
💡 Use /analytics [days] to change period
Example: /analytics 7 (last 7 days)
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="📊 7 Days", callback_data="analytics_7"),
                InlineKeyboardButton(text="📊 30 Days", callback_data="analytics_30")
            ],
            [
                InlineKeyboardButton(text="📊 90 Days", callback_data="analytics_90"),
                InlineKeyboardButton(text="📊 All Time", callback_data="analytics_365")
            ],
            [
                InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")
            ]
        ])
        
        await callback.message.edit_text(analytics_text, reply_markup=keyboard, parse_mode="HTML")
        await callback.answer()
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


@dp.callback_query(F.data == "test_api_callback")
async def handle_test_api_callback(callback: CallbackQuery):
    # Reuse the test_mexc command
    await cmd_test_mexc(callback.message)
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


@dp.message(Command("test_autotrader"))
async def cmd_test_autotrader(message: types.Message):
    """Test autotrader with a live market signal (Admin only)"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user or not user.is_admin:
            await message.answer("❌ This command is only available to admins.")
            return
        
        prefs = user.preferences
        has_mexc = prefs and prefs.mexc_api_key and prefs.mexc_api_secret
        has_okx = prefs and prefs.okx_api_key and prefs.okx_api_secret and prefs.okx_passphrase
        has_kucoin = prefs and prefs.kucoin_api_key and prefs.kucoin_api_secret and prefs.kucoin_passphrase
        
        if not has_mexc and not has_okx and not has_kucoin:
            await message.answer("❌ Please connect an exchange first:\n• /set_kucoin_api - For KuCoin\n• /set_okx_api - For OKX\n• /set_mexc_api - For MEXC")
            return
        
        if not prefs.auto_trading_enabled:
            await message.answer("❌ Auto-trading is disabled. Enable it first with /toggle_autotrading")
            return
        
        exchange_name = prefs.preferred_exchange or "KuCoin"
        await message.answer(f"🧪 <b>Testing {exchange_name} Autotrader...</b>\n\nCreating test signal and executing trade...", parse_mode="HTML")
        
        try:
            import ccxt
            from app.services.mexc_trader import execute_auto_trade
            
            # Get current ETH price from KuCoin (cheaper for testing)
            exchange = ccxt.kucoin()
            ticker = exchange.fetch_ticker('ETH/USDT')
            current_price = ticker['last']
            
            # Create a small test LONG signal
            test_signal = {
                'symbol': 'ETH/USDT',
                'direction': 'LONG',
                'entry_price': current_price,
                'stop_loss': current_price * 0.98,  # 2% SL
                'take_profit': current_price * 1.04,  # 4% TP
                'take_profit_1': current_price * 1.015,  # 1.5% TP1
                'take_profit_2': current_price * 1.025,  # 2.5% TP2
                'take_profit_3': current_price * 1.04,   # 4% TP3
                'timeframe': '1h',
                'risk_level': 'LOW',
                'signal_type': 'TEST'
            }
            
            # Execute the trade
            result = await execute_trade_on_exchange(test_signal, user, db)
            
            if result:
                exchange_name = prefs.preferred_exchange or "MEXC"
                result_msg = f"""
✅ <b>Autotrader Test Successful!</b>

📊 Trade Executed on {exchange_name}:
• Symbol: ETH/USDT
• Direction: LONG
• Entry: ${current_price:,.2f}
• Stop Loss: ${test_signal['stop_loss']:,.2f}
• Take Profit: ${test_signal['take_profit']:,.2f}

🔍 Check your {exchange_name} account to verify the position!

Use /dashboard to see the trade in your open positions.
"""
            else:
                result_msg = """
⚠️ <b>Test Trade Not Executed</b>

Possible reasons:
• Duplicate signal (same trade exists)
• Insufficient balance
• Max positions reached
• Risk filters blocked it

Check logs for details.
"""
            
            await message.answer(result_msg, parse_mode="HTML")
            
        except Exception as e:
            error_msg = f"""
❌ <b>Autotrader Test Failed</b>

Error: {str(e)[:300]}

This could indicate:
• API connection issues
• Invalid API permissions
• MEXC server problems
• Insufficient balance

Try /test_mexc to verify your API connection.
"""
            await message.answer(error_msg, parse_mode="HTML")
            logger.error(f"Test autotrader error: {e}", exc_info=True)
            
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


@dp.message(Command("spot_flow"))
async def cmd_spot_flow(message: types.Message):
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
        
        await message.answer("🔄 <b>Scanning Spot Markets...</b>\n\nAnalyzing order books and volume across exchanges...\nPlease wait...", parse_mode="HTML")
        
        from app.services.spot_monitor import spot_monitor
        
        # Scan all symbols across all exchanges
        flow_signals = await spot_monitor.scan_all_symbols()
        
        if not flow_signals:
            await message.answer("""
📊 <b>Spot Market Flow Analysis</b>
━━━━━━━━━━━━━━━━━━━━

No significant buying or selling pressure detected across exchanges at this moment.

All markets appear to be in equilibrium.
""", parse_mode="HTML")
            return
        
        # Build flow report
        flow_report = """
📊 <b>Spot Market Flow Analysis</b>
━━━━━━━━━━━━━━━━━━━━

<b>🔴 HEAVY SELLING Detected:</b>
"""
        
        heavy_selling = [f for f in flow_signals if 'SELLING' in f['flow_signal']]
        if heavy_selling:
            for flow in heavy_selling:
                emoji = "🚨" if flow['confidence'] >= 70 else "⚠️"
                flow_report += f"\n{emoji} <b>{flow['symbol']}</b>"
                flow_report += f"\n   Confidence: {flow['confidence']:.0f}%"
                flow_report += f"\n   Exchanges: {flow['exchanges_analyzed']}"
                flow_report += f"\n   Pressure: {flow['avg_pressure']:.2f}"
                flow_report += f"\n"
        else:
            flow_report += "\nNone detected.\n"
        
        flow_report += """
<b>🟢 HEAVY BUYING Detected:</b>
"""
        
        heavy_buying = [f for f in flow_signals if 'BUYING' in f['flow_signal']]
        if heavy_buying:
            for flow in heavy_buying:
                emoji = "🚀" if flow['confidence'] >= 70 else "📈"
                flow_report += f"\n{emoji} <b>{flow['symbol']}</b>"
                flow_report += f"\n   Confidence: {flow['confidence']:.0f}%"
                flow_report += f"\n   Exchanges: {flow['exchanges_analyzed']}"
                flow_report += f"\n   Pressure: {flow['avg_pressure']:.2f}"
                flow_report += f"\n"
        else:
            flow_report += "\nNone detected.\n"
        
        flow_report += """
<b>⚡ VOLUME SPIKES:</b>
"""
        
        volume_spikes = [f for f in flow_signals if 'VOLUME_SPIKE' in f['flow_signal']]
        if volume_spikes:
            for flow in volume_spikes:
                direction = "📈 Buy" if "BUY" in flow['flow_signal'] else "📉 Sell"
                flow_report += f"\n{direction} <b>{flow['symbol']}</b>"
                flow_report += f"\n   Spike Count: {flow['spike_count']}/{flow['exchanges_analyzed']}"
                flow_report += f"\n   Volume: ${flow['total_volume']:,.0f}"
                flow_report += f"\n"
        else:
            flow_report += "\nNone detected.\n"
        
        flow_report += """
━━━━━━━━━━━━━━━━━━━━
<i>Data from Coinbase, Kraken, OKX (geo-available exchanges)</i>

💡 Tip: High confidence flows (70%+) often precede futures market moves!
"""
        
        # Save significant flows to database
        for flow in flow_signals:
            await spot_monitor.save_spot_activity(flow)
        
        await message.answer(flow_report, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in spot_flow command: {e}")
        await message.answer("❌ Error analyzing spot markets. Please try again later.")
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
        
        # Check if API is already connected
        prefs = user.preferences
        if prefs and prefs.mexc_api_key and prefs.mexc_api_secret:
            already_connected_text = """
✅ <b>MEXC API Already Connected!</b>

Your MEXC account is already linked to the bot.

<b>What you can do:</b>
• /test_mexc - Test your connection
• /autotrading_status - Check auto-trading status
• /toggle_autotrading - Enable/disable auto-trading
• /remove_mexc_api - Disconnect and remove API keys

<i>Your API keys are encrypted and secure! 🔒</i>
"""
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🧪 Test API", callback_data="test_api_callback")],
                [InlineKeyboardButton(text="🤖 Auto-Trading Menu", callback_data="autotrading_menu")],
                [InlineKeyboardButton(text="❌ Remove API", callback_data="remove_api_confirm")],
                [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
            ])
            await message.answer(already_connected_text, reply_markup=keyboard, parse_mode="HTML")
            return
        
        await message.answer("""
🔑 <b>Let's connect your MEXC account!</b>

⚙️ First, get your API keys:
1. Go to MEXC → API Management
2. Create new API key
3. ⚠️ <b>IMPORTANT:</b> Enable <b>ONLY Futures Trading</b> permission
   • Do NOT enable withdrawals
   • Do NOT enable spot trading
4. Copy your API Key

🔒 <b>Security Notice:</b>
✅ You'll ALWAYS have access to your own funds
✅ API can only trade futures, cannot withdraw
✅ Keys are encrypted and stored securely

📝 Now, please send me your <b>API Key</b>:
        """, parse_mode="HTML")
        
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


@dp.message(Command("set_okx_api"))
async def cmd_set_okx_api(message: types.Message, state: FSMContext):
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
        
        # Check if API is already connected
        prefs = user.preferences
        if prefs and prefs.okx_api_key and prefs.okx_api_secret and prefs.okx_passphrase:
            already_connected_text = """
✅ <b>OKX API Already Connected!</b>

Your OKX account is already linked to the bot.

<b>What you can do:</b>
• /test_okx - Test your connection
• /autotrading_status - Check auto-trading status
• /toggle_autotrading - Enable/disable auto-trading
• /remove_okx_api - Disconnect and remove API keys

<i>Your API keys are encrypted and secure! 🔒</i>
"""
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🧪 Test API", callback_data="test_okx_api")],
                [InlineKeyboardButton(text="🤖 Auto-Trading Menu", callback_data="autotrading_menu")],
                [InlineKeyboardButton(text="❌ Remove API", callback_data="remove_okx_api")],
                [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
            ])
            await message.answer(already_connected_text, reply_markup=keyboard, parse_mode="HTML")
            return
        
        await message.answer("""
🔑 <b>Let's connect your OKX account!</b>

⚙️ First, get your API keys:
1. Go to OKX → API Management
2. Create new V5 API key
3. ⚠️ <b>IMPORTANT:</b> Enable <b>ONLY Trading</b> permission
   • Do NOT enable withdrawals
   • Do NOT enable deposits
   • Set it for FUTURES trading
4. Copy your API Key, Secret, and Passphrase

🔒 <b>Security Notice:</b>
✅ You'll ALWAYS have access to your own funds
✅ API can only trade futures, cannot withdraw
✅ Keys are encrypted and stored securely

📝 Now, please send me your <b>API Key</b>:
        """, parse_mode="HTML")
        
        await state.set_state(OKXSetup.waiting_for_api_key)
    finally:
        db.close()


@dp.message(OKXSetup.waiting_for_api_key)
async def process_okx_api_key(message: types.Message, state: FSMContext):
    # Save API key in state
    await state.update_data(okx_api_key=message.text.strip())
    
    # Delete user's message for security
    try:
        await message.delete()
    except:
        pass
    
    await message.answer("""
✅ API Key received!

🔐 Now, please send me your <b>API Secret</b>:
    """, parse_mode="HTML")
    
    await state.set_state(OKXSetup.waiting_for_api_secret)


@dp.message(OKXSetup.waiting_for_api_secret)
async def process_okx_api_secret(message: types.Message, state: FSMContext):
    # Save API secret in state
    await state.update_data(okx_api_secret=message.text.strip())
    
    # Delete user's message for security
    try:
        await message.delete()
    except:
        pass
    
    await message.answer("""
✅ API Secret received!

🔑 Finally, please send me your <b>API Passphrase</b>:
    """, parse_mode="HTML")
    
    await state.set_state(OKXSetup.waiting_for_passphrase)


@dp.message(OKXSetup.waiting_for_passphrase)
async def process_okx_passphrase(message: types.Message, state: FSMContext):
    db = SessionLocal()
    
    try:
        # Get saved API key and secret from state
        data = await state.get_data()
        api_key = data.get('okx_api_key')
        api_secret = data.get('okx_api_secret')
        passphrase = message.text.strip()
        
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
        prefs.okx_api_key = encrypt_api_key(api_key)
        prefs.okx_api_secret = encrypt_api_key(api_secret)
        prefs.okx_passphrase = encrypt_api_key(passphrase)
        prefs.preferred_exchange = "OKX"  # Set OKX as preferred
        db.commit()
        
        await message.answer("""
✅ <b>OKX API keys saved successfully!</b>

🔐 Your messages have been deleted for security.
🔒 Keys are encrypted and stored securely.

<b>Next steps:</b>
1️⃣ /toggle_autotrading - Enable auto-trading
2️⃣ /autotrading_status - Check your settings
3️⃣ /risk_settings - Configure risk management

You're all set! 🚀
        """, parse_mode="HTML")
        
        # Clear the state
        await state.clear()
    finally:
        db.close()


@dp.message(Command("remove_okx_api"))
async def cmd_remove_okx_api(message: types.Message):
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
            prefs.okx_api_key = None
            prefs.okx_api_secret = None
            prefs.okx_passphrase = None
            prefs.auto_trading_enabled = False
            db.commit()
            await message.answer("✅ OKX API keys removed and auto-trading disabled")
        else:
            await message.answer("⚠️ No settings found. Use /start first.")
    finally:
        db.close()


@dp.message(Command("set_kucoin_api"))
async def cmd_set_kucoin_api(message: types.Message, state: FSMContext):
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
        if prefs and prefs.kucoin_api_key and prefs.kucoin_api_secret and prefs.kucoin_passphrase:
            await message.answer("✅ <b>KuCoin API Already Connected!</b>\n\nYour KuCoin Futures account is linked!\n\n/autotrading_status - Check settings\n/toggle_autotrading - Enable/disable\n/remove_kucoin_api - Disconnect", parse_mode="HTML")
            return
        
        await message.answer("""
🔑 <b>Let's connect KuCoin Futures!</b>

⚙️ Get API keys from <b>futures.kucoin.com</b>:
1. Go to futures.kucoin.com → API Management  
2. Create API with <b>Futures Trading</b> permission only
3. ⚠️ NO withdrawals, NO spot trading
4. Copy API Key, Secret, Passphrase

📝 Send me your <b>API Key</b>:
        """, parse_mode="HTML")
        
        await state.set_state(KuCoinSetup.waiting_for_api_key)
    finally:
        db.close()


@dp.message(KuCoinSetup.waiting_for_api_key)
async def process_kucoin_api_key(message: types.Message, state: FSMContext):
    await state.update_data(kucoin_api_key=message.text.strip())
    try:
        await message.delete()
    except:
        pass
    await message.answer("✅ API Key received!\n\n🔐 Send <b>API Secret</b>:", parse_mode="HTML")
    await state.set_state(KuCoinSetup.waiting_for_api_secret)


@dp.message(KuCoinSetup.waiting_for_api_secret)
async def process_kucoin_api_secret(message: types.Message, state: FSMContext):
    await state.update_data(kucoin_api_secret=message.text.strip())
    try:
        await message.delete()
    except:
        pass
    await message.answer("✅ API Secret received!\n\n🔑 Send <b>Passphrase</b>:", parse_mode="HTML")
    await state.set_state(KuCoinSetup.waiting_for_passphrase)


@dp.message(KuCoinSetup.waiting_for_passphrase)
async def process_kucoin_passphrase(message: types.Message, state: FSMContext):
    db = SessionLocal()
    
    try:
        data = await state.get_data()
        api_key = data.get('kucoin_api_key')
        api_secret = data.get('kucoin_api_secret')
        passphrase = message.text.strip()
        
        try:
            await message.delete()
        except:
            pass
        
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("❌ Error: User not found. Use /start first.")
            await state.clear()
            return
        
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs:
            prefs = UserPreference(user_id=user.id)
            db.add(prefs)
            db.flush()
        
        prefs.kucoin_api_key = encrypt_api_key(api_key)
        prefs.kucoin_api_secret = encrypt_api_key(api_secret)
        prefs.kucoin_passphrase = encrypt_api_key(passphrase)
        prefs.preferred_exchange = "KuCoin"
        db.commit()
        
        await message.answer("""
✅ <b>KuCoin Futures API Connected!</b>

🔒 Keys encrypted & messages deleted
⚡ Ready for auto-trading

<b>Next:</b>
/toggle_autotrading - Enable
/autotrading_status - Check settings
/test_autotrader - Test trade

You're all set! 🚀
        """, parse_mode="HTML")
        
        await state.clear()
    finally:
        db.close()


@dp.message(Command("remove_kucoin_api"))
async def cmd_remove_kucoin_api(message: types.Message):
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
        
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        
        if prefs:
            prefs.kucoin_api_key = None
            prefs.kucoin_api_secret = None
            prefs.kucoin_passphrase = None
            prefs.auto_trading_enabled = False
            db.commit()
            await message.answer("✅ KuCoin API keys removed and auto-trading disabled")
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
        
        # Explicitly query preferences to ensure fresh data
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        
        if prefs:
            # Check if any exchange API is configured
            has_mexc = prefs.mexc_api_key and prefs.mexc_api_secret
            has_okx = prefs.okx_api_key and prefs.okx_api_secret and prefs.okx_passphrase
            has_kucoin = prefs.kucoin_api_key and prefs.kucoin_api_secret and prefs.kucoin_passphrase
            
            if not has_mexc and not has_okx and not has_kucoin:
                await message.answer("❌ Please set API keys first:\n• /set_kucoin_api - For KuCoin (Recommended)\n• /set_okx_api - For OKX\n• /set_mexc_api - For MEXC")
                return
            
            prefs.auto_trading_enabled = not prefs.auto_trading_enabled
            db.commit()
            status = "enabled" if prefs.auto_trading_enabled else "disabled"
            exchange = prefs.preferred_exchange or "KuCoin"
            await message.answer(f"✅ Auto-trading {status} on {exchange}")
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
        
        # Explicitly query preferences to ensure fresh data
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        
        if not prefs:
            await message.answer("Settings not found. Use /start first.")
            return
        
        # Check all exchanges
        mexc_status = "✅ Connected" if prefs.mexc_api_key and prefs.mexc_api_secret else "❌ Not Set"
        okx_status = "✅ Connected" if prefs.okx_api_key and prefs.okx_api_secret and prefs.okx_passphrase else "❌ Not Set"
        kucoin_status = "✅ Connected" if prefs.kucoin_api_key and prefs.kucoin_api_secret and prefs.kucoin_passphrase else "❌ Not Set"
        
        auto_status = "✅ Enabled" if prefs.auto_trading_enabled else "❌ Disabled"
        preferred_exchange = prefs.preferred_exchange or "KuCoin"
        risk_sizing = "✅ Enabled" if prefs.risk_based_sizing else "❌ Disabled"
        trailing_stop = "✅ Enabled" if prefs.use_trailing_stop else "❌ Disabled"
        breakeven_stop = "✅ Enabled" if prefs.use_breakeven_stop else "❌ Disabled"
        
        open_positions = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.status == 'open'
        ).count()
        
        status_text = f"""
🤖 Auto-Trading Status

📊 Exchange Configuration:
  • KuCoin API: {kucoin_status} ⭐ Recommended
  • OKX API: {okx_status}
  • MEXC API: {mexc_status}
  • Active Exchange: {preferred_exchange}

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
/set_kucoin_api - Connect KuCoin (Best)
/set_okx_api - Connect OKX
/set_mexc_api - Connect MEXC
/risk_settings - Configure risk management
/toggle_autotrading - Toggle on/off
        """
        
        await message.answer(status_text)
    finally:
        db.close()


@dp.message(Command("toggle_paper_mode"))
async def cmd_toggle_paper_mode(message: types.Message):
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
        prefs.paper_trading_mode = not prefs.paper_trading_mode
        db.commit()
        
        status = "ENABLED" if prefs.paper_trading_mode else "DISABLED"
        emoji = "✅" if prefs.paper_trading_mode else "❌"
        
        message_text = f"""
{emoji} <b>Paper Trading Mode {status}</b>

{prefs.paper_trading_mode and '''📝 <b>What is Paper Trading?</b>
• Practice trading with virtual money
• Test strategies risk-free
• All signals execute as paper trades
• Track performance without real capital

💰 <b>Your Paper Balance:</b> ${prefs.paper_balance:,.2f}

Use /paper_status to view details''' or '''💼 <b>Live Trading Mode Active</b>
• Real trades will execute with MEXC API
• Make sure auto-trading is configured
• Use /autotrading_status to check setup'''}

━━━━━━━━━━━━━━━━━━━━
Commands:
/paper_status - View paper trading stats
/reset_paper_balance - Reset virtual balance
/toggle_paper_mode - Switch modes
"""
        
        await message.answer(message_text, parse_mode="HTML")
    finally:
        db.close()


@dp.message(Command("paper_status"))
async def cmd_paper_status(message: types.Message):
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
        
        from app.models import PaperTrade
        
        # Get paper trading stats
        open_paper_trades = db.query(PaperTrade).filter(
            PaperTrade.user_id == user.id,
            PaperTrade.status == "open"
        ).all()
        
        closed_paper_trades = db.query(PaperTrade).filter(
            PaperTrade.user_id == user.id,
            PaperTrade.status == "closed"
        ).all()
        
        total_paper_pnl = sum(t.pnl for t in closed_paper_trades)
        win_count = len([t for t in closed_paper_trades if t.pnl > 0])
        loss_count = len([t for t in closed_paper_trades if t.pnl < 0])
        win_rate = (win_count / len(closed_paper_trades) * 100) if closed_paper_trades else 0
        
        status_text = f"""
📝 <b>Paper Trading Status</b>
━━━━━━━━━━━━━━━━━━━━

💰 <b>Virtual Balance:</b> ${prefs.paper_balance:,.2f}
⚡ <b>Mode:</b> {'✅ Active' if prefs.paper_trading_mode else '❌ Inactive'}

📊 <b>Paper Trades Statistics:</b>
• Open Positions: {len(open_paper_trades)}
• Closed Trades: {len(closed_paper_trades)}
• Total P&L: ${total_paper_pnl:,.2f}
• Win Rate: {win_rate:.1f}%
• Wins: {win_count} | Losses: {loss_count}
"""
        
        if open_paper_trades:
            status_text += "\n<b>📈 Open Paper Positions:</b>\n"
            for trade in open_paper_trades[:5]:
                unrealized_pnl = 0
                status_text += f"• {trade.symbol} {trade.direction}: ${trade.position_size:.2f}\n"
        
        status_text += """
━━━━━━━━━━━━━━━━━━━━
💡 <b>Paper trading allows you to:</b>
• Test the bot's signals risk-free
• Learn trading strategies
• Build confidence before live trading

Commands:
/toggle_paper_mode - Enable/disable
/reset_paper_balance - Reset to $10,000
"""
        
        await message.answer(status_text, parse_mode="HTML")
    finally:
        db.close()


@dp.message(Command("reset_paper_balance"))
async def cmd_reset_paper_balance(message: types.Message):
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
        prefs.paper_balance = 10000.0
        db.commit()
        
        await message.answer(
            "✅ <b>Paper Balance Reset!</b>\n\n"
            "Your virtual balance has been reset to $10,000.\n"
            "Ready to start fresh paper trading!",
            parse_mode="HTML"
        )
    finally:
        db.close()


@dp.message(Command("backtest"))
async def cmd_backtest(message: types.Message):
    """Run backtest on historical data (Admin only)"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        # Admin only feature
        if not user.is_admin:
            await message.answer("⛔ This command is only available for admins.")
            return
        
        # Parse arguments: /backtest <symbol> <timeframe> <days>
        args = message.text.split()
        
        if len(args) < 2:
            await message.answer(
                "📊 <b>Backtest Command</b>\n\n"
                "Usage: /backtest <symbol> [timeframe] [days]\n\n"
                "Examples:\n"
                "• /backtest BTC/USDT:USDT\n"
                "• /backtest ETH/USDT:USDT 4h\n"
                "• /backtest BNB/USDT:USDT 1h 30\n\n"
                "Defaults: 1h timeframe, 90 days",
                parse_mode="HTML"
            )
            return
        
        symbol = args[1]
        timeframe = args[2] if len(args) > 2 else '1h'
        days = int(args[3]) if len(args) > 3 and args[3].isdigit() else 90
        
        await message.answer(
            f"📊 <b>Running Backtest...</b>\n\n"
            f"Symbol: {symbol}\n"
            f"Timeframe: {timeframe}\n"
            f"Period: {days} days\n\n"
            f"⏳ This may take a moment...",
            parse_mode="HTML"
        )
        
        # Run backtest
        from app.services.backtester import Backtester
        backtester = Backtester(exchange_name='kucoin')
        results = backtester.run_backtest(symbol, timeframe, days)
        
        if 'error' in results:
            await message.answer(f"❌ Error: {results['error']}")
            return
        
        # Format results
        best_trade = results.get('best_trade')
        worst_trade = results.get('worst_trade')
        
        backtest_text = f"""
📊 <b>Backtest Results</b>
━━━━━━━━━━━━━━━━━━━━

<b>📈 Strategy Performance</b>
Symbol: {results['symbol']}
Timeframe: {results['timeframe']}
Period: {results['period_days']} days

<b>🎯 Trading Statistics</b>
• Total Trades: {results['total_trades']}
• Signals Generated: {results['signals_generated']}
• Winning Trades: {results['winning_trades']} ✅
• Losing Trades: {results['losing_trades']} ❌
• Win Rate: {results['win_rate']:.1f}%

<b>💰 Profitability (10x Leverage)</b>
• Total Return: {results['total_return']:+.2f}%
• Avg Win: {results['avg_win']:+.2f}%
• Avg Loss: {results['avg_loss']:+.2f}%
• Profit Factor: {results['profit_factor']:.2f}
• Max Drawdown: {results['max_drawdown']:.2f}%

<b>🏆 Best Trade</b>
{best_trade['direction'] if best_trade else 'N/A'}: {best_trade['pnl_percent_10x']:+.2f if best_trade else 0}%
Entry: ${best_trade['entry_price']:.4f if best_trade else 0}
Exit: ${best_trade['exit_price']:.4f if best_trade else 0} ({best_trade['exit_reason'] if best_trade else 'N/A'})

<b>📉 Worst Trade</b>
{worst_trade['direction'] if worst_trade else 'N/A'}: {worst_trade['pnl_percent_10x']:+.2f if worst_trade else 0}%
Entry: ${worst_trade['entry_price']:.4f if worst_trade else 0}
Exit: ${worst_trade['exit_price']:.4f if worst_trade else 0} ({worst_trade['exit_reason'] if worst_trade else 'N/A'})

━━━━━━━━━━━━━━━━━━━━
💡 This backtest simulates the EMA crossover strategy with volume & RSI filters on historical data.

<i>Past performance does not guarantee future results.</i>
"""
        
        await message.answer(backtest_text, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}", exc_info=True)
        await message.answer(f"❌ Error running backtest: {str(e)}")
    finally:
        db.close()


@dp.message(Command("set_tp_percentages"))
async def cmd_set_tp_percentages(message: types.Message):
    """Allow users to customize partial TP percentages"""
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
        
        # Parse command arguments
        args = message.text.split()[1:] if len(message.text.split()) > 1 else []
        
        if len(args) != 3:
            prefs = user.preferences or UserPreference()
            await message.answer(f"""
📊 **Partial Take Profit Settings**

Current configuration:
• TP1 (1.5R): {prefs.tp1_percent}% of position
• TP2 (2.5R): {prefs.tp2_percent}% of position  
• TP3 (4R): {prefs.tp3_percent}% of position

To change, use:
`/set_tp_percentages <tp1%> <tp2%> <tp3%>`

Example:
`/set_tp_percentages 25 35 40`

Note: The three percentages must add up to 100%
""", parse_mode="Markdown")
            return
        
        try:
            tp1 = int(args[0])
            tp2 = int(args[1])
            tp3 = int(args[2])
            
            # Validation
            if tp1 < 0 or tp2 < 0 or tp3 < 0:
                await message.answer("❌ Percentages must be positive numbers!")
                return
            
            if tp1 + tp2 + tp3 != 100:
                await message.answer(f"❌ Percentages must add up to 100%!\nYour total: {tp1 + tp2 + tp3}%")
                return
            
            # Update preferences
            if not user.preferences:
                user.preferences = UserPreference(user_id=user.id)
                db.add(user.preferences)
            
            user.preferences.tp1_percent = tp1
            user.preferences.tp2_percent = tp2
            user.preferences.tp3_percent = tp3
            db.commit()
            
            await message.answer(f"""
✅ **Partial TP Updated!**

New configuration:
🎯 TP1 (1.5R): {tp1}% close
🎯 TP2 (2.5R): {tp2}% close
🎯 TP3 (4R): {tp3}% close

This will apply to all new trades.
Existing open trades keep their original settings.
""", parse_mode="Markdown")
            
        except ValueError:
            await message.answer("❌ Invalid format! Please use whole numbers.\nExample: `/set_tp_percentages 30 30 40`", parse_mode="Markdown")
    
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

**Bot Instance Management:**
/bot_status - Check bot instance status
/force_stop - Force stop other instances
/instance_health - View instance health
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


@dp.message(Command("bot_status"))
async def cmd_bot_status(message: types.Message):
    """Check bot instance status and detect conflicts"""
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        from app.services.bot_instance_manager import get_instance_manager
        manager = get_instance_manager(bot)
        
        health = await manager.check_bot_health()
        
        status_text = f"""
🤖 <b>Bot Instance Status</b>

<b>Health:</b> {'✅ Healthy' if health['healthy'] else '❌ Unhealthy'}
<b>Bot Username:</b> @{health.get('bot_username', 'N/A')}
<b>Bot ID:</b> {health.get('bot_id', 'N/A')}
<b>Process ID:</b> {health.get('instance_pid', 'N/A')}
<b>Has Lock:</b> {'✅ Yes' if health.get('has_lock') else '❌ No'}

{f"<b>Error:</b> {health.get('error', 'N/A')}" if not health['healthy'] else ''}

<i>Use /force_stop to terminate other instances if conflicts exist</i>
"""
        await message.answer(status_text, parse_mode="HTML")
    finally:
        db.close()


@dp.message(Command("force_stop"))
async def cmd_force_stop(message: types.Message):
    """Force stop other bot instances"""
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        from app.services.bot_instance_manager import get_instance_manager
        manager = get_instance_manager(bot)
        
        await message.answer("🛑 <b>Force stopping other bot instances...</b>", parse_mode="HTML")
        
        success = await manager.force_stop_other_instances()
        
        if success:
            # Try to acquire lock
            if await manager.acquire_lock():
                await message.answer(
                    "✅ <b>Success!</b>\n\n"
                    "• Other instances stopped\n"
                    "• Lock acquired\n"
                    "• This instance is now the active bot\n\n"
                    "<i>The bot should be working normally now</i>",
                    parse_mode="HTML"
                )
            else:
                await message.answer(
                    "⚠️ <b>Partial Success</b>\n\n"
                    "Other instances stopped but couldn't acquire lock.\n"
                    "Try running /force_stop again.",
                    parse_mode="HTML"
                )
        else:
            await message.answer(
                "❌ <b>Failed</b>\n\n"
                "Could not force stop other instances. Check logs for details.",
                parse_mode="HTML"
            )
    finally:
        db.close()


@dp.message(Command("instance_health"))
async def cmd_instance_health(message: types.Message):
    """Detailed instance health check"""
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        from app.services.bot_instance_manager import get_instance_manager, LOCK_FILE, INSTANCE_ID
        from pathlib import Path
        import os
        
        manager = get_instance_manager(bot)
        
        # Check lock file
        lock_path = Path(LOCK_FILE)
        lock_exists = lock_path.exists()
        lock_info = "N/A"
        
        if lock_exists:
            try:
                with open(LOCK_FILE, 'r') as f:
                    lock_data = f.read().strip().split('|')
                    lock_pid = lock_data[0]
                    lock_time = lock_data[1] if len(lock_data) > 1 else "Unknown"
                    
                    # Check if process is running
                    try:
                        os.kill(int(lock_pid), 0)
                        process_status = "✅ Running"
                    except OSError:
                        process_status = "❌ Dead (stale lock)"
                    
                    lock_info = f"PID {lock_pid} ({process_status})\nLocked: {lock_time}"
            except Exception as e:
                lock_info = f"Error reading: {e}"
        
        health_text = f"""
🔍 <b>Detailed Instance Health</b>

<b>Current Instance:</b>
• Process ID: {INSTANCE_ID}
• Has Lock: {'✅ Yes' if manager.is_locked else '❌ No'}
• Monitor Running: {'✅ Yes' if manager.monitor_task else '❌ No'}

<b>Lock File Status:</b>
• Exists: {'✅ Yes' if lock_exists else '❌ No'}
• Location: {LOCK_FILE}
• Info: {lock_info}

<b>Recommendations:</b>
{
    "✅ Everything looks good!" if manager.is_locked and lock_exists 
    else "⚠️ Run /force_stop to fix conflicts" if lock_exists and not manager.is_locked
    else "⚠️ No lock file - instance may not be protected"
}
"""
        await message.answer(health_text, parse_mode="HTML")
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
                                await execute_trade_on_exchange(signal, user, db)
                        except Exception as e:
                            logger.error(f"Error sending news DM to {user.telegram_id}: {e}")
    
    except Exception as e:
        logger.error(f"Error broadcasting news signal: {e}")
    finally:
        db.close()


async def broadcast_spot_flow_alert(flow_data: dict):
    """Broadcast high-conviction spot market flow alerts AND trigger auto-trades"""
    db = SessionLocal()
    
    try:
        # Format flow alert message
        flow_type = flow_data['flow_signal']
        symbol = flow_data['symbol']
        confidence = flow_data['confidence']
        
        # Determine trading direction from flow
        if 'BUYING' in flow_type or 'SPIKE_BUY' in flow_type:
            emoji = "🚀"
            direction_text = "HEAVY BUYING" if 'BUYING' in flow_type else "VOLUME SPIKE (Buy)"
            color = "🟢"
            trade_direction = 'LONG'
        else:  # SELLING or SPIKE_SELL
            emoji = "🔴"
            direction_text = "HEAVY SELLING" if 'SELLING' in flow_type else "VOLUME SPIKE (Sell)"
            color = "🔴"
            trade_direction = 'SHORT'
        
        message = f"""
{emoji} <b>SPOT MARKET FLOW SIGNAL</b>
━━━━━━━━━━━━━━━━━━━━

{color} <b>{direction_text}</b>
<b>Symbol:</b> {symbol}
<b>Confidence:</b> {confidence:.0f}%
<b>Direction:</b> {trade_direction}

<b>📊 Multi-Exchange Analysis</b>
• Order Book Imbalance: {flow_data['avg_imbalance']:+.2f}
• Trade Pressure: {flow_data['avg_pressure']:+.2f}
• Exchanges Analyzed: {flow_data['exchanges_analyzed']}
• Volume Spikes: {flow_data['spike_count']}

<b>💡 Market Context</b>
Spot market flows often precede futures movements. High confidence flows (70%+) suggest institutional activity.

<i>🔍 Data from: Coinbase, Kraken, OKX</i>
"""
        
        # Get current price for entry
        exchange = getattr(ccxt, settings.EXCHANGE)()
        try:
            ticker = await exchange.fetch_ticker(symbol)
            entry_price = ticker['last']
            await exchange.close()
        except:
            await exchange.close()
            logger.error(f"Could not fetch price for {symbol}")
            return
        
        # Calculate ATR-based SL/TP (simplified for spot flow signals)
        atr_estimate = entry_price * 0.02  # 2% volatility estimate
        
        if trade_direction == 'LONG':
            stop_loss = entry_price - (atr_estimate * 2)
            take_profit = entry_price + (atr_estimate * 2)  # 2R target
        else:
            stop_loss = entry_price + (atr_estimate * 2)
            take_profit = entry_price - (atr_estimate * 2)
        
        # Create signal object for auto-trading
        signal_data = {
            'symbol': symbol,
            'direction': trade_direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timeframe': 'spot_flow',
            'signal_type': 'spot_flow',
            'risk_level': 'LOW' if confidence >= 80 else 'MEDIUM'
        }
        
        # Save signal to database
        signal = Signal(**signal_data)
        db.add(signal)
        db.commit()
        db.refresh(signal)
        
        # Broadcast to channel
        await bot.send_message(settings.BROADCAST_CHAT_ID, message, parse_mode="HTML")
        logger.info(f"Spot flow signal broadcast: {trade_direction} {symbol} ({confidence:.0f}%)")
        
        # Send to users and trigger auto-trades
        users = db.query(User).filter(User.approved == True, User.banned == False).all()
        for user in users:
            if user.preferences and user.preferences.dm_alerts:
                # Check if symbol is muted
                if symbol not in user.preferences.get_muted_symbols_list():
                    try:
                        await bot.send_message(user.telegram_id, message, parse_mode="HTML")
                        
                        # Auto-trade if enabled (spot flow follows market momentum)
                        if user.preferences.auto_trading_enabled:
                            logger.info(f"Executing spot flow auto-trade for user {user.telegram_id}: {trade_direction} {symbol}")
                            await execute_trade_on_exchange(signal, user, db)
                            
                    except Exception as e:
                        logger.error(f"Error sending spot flow signal to {user.telegram_id}: {e}")
            
    except Exception as e:
        logger.error(f"Error broadcasting spot flow alert: {e}")
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
        
        # Calculate risk/reward ratio for TP3
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit_3 - signal.entry_price) if signal.take_profit_3 else abs(signal.take_profit - signal.entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Calculate 10x leverage PnL for each TP level
        tp1_pnl = calculate_leverage_pnl(signal.entry_price, signal.take_profit_1, signal.direction, 10) if signal.take_profit_1 else None
        tp2_pnl = calculate_leverage_pnl(signal.entry_price, signal.take_profit_2, signal.direction, 10) if signal.take_profit_2 else None
        tp3_pnl = calculate_leverage_pnl(signal.entry_price, signal.take_profit_3, signal.direction, 10) if signal.take_profit_3 else calculate_leverage_pnl(signal.entry_price, signal.take_profit, signal.direction, 10)
        sl_pnl = calculate_leverage_pnl(signal.entry_price, signal.stop_loss, signal.direction, 10)
        
        # Safe volume percentage calculation
        if signal.volume_avg and signal.volume_avg > 0:
            volume_pct = ((signal.volume / signal.volume_avg - 1) * 100)
            volume_text = f"{signal.volume:,.0f} ({'+' if signal.volume > signal.volume_avg else ''}{volume_pct:.1f}% avg)"
        else:
            volume_text = f"{signal.volume:,.0f}"
        
        # Risk level emoji
        risk_emoji = "🟢" if signal.risk_level == "LOW" else "🟡"
        
        # Build TP section with partial close percentages
        tp_section = ""
        if signal.take_profit_1 and signal.take_profit_2 and signal.take_profit_3:
            tp_section = f"""🎯 Take Profit Levels (Partial Closes):
  TP1: ${signal.take_profit_1} (30% @ {tp1_pnl:+.2f}%)
  TP2: ${signal.take_profit_2} (30% @ {tp2_pnl:+.2f}%)
  TP3: ${signal.take_profit_3} (40% @ {tp3_pnl:+.2f}%)"""
        else:
            tp_section = f"""🎯 Take Profit: ${signal.take_profit}
💰 TP PnL: {tp3_pnl:+.2f}% (10x)"""
        
        signal_text = f"""
🚨 NEW {signal.direction} SIGNAL

📊 Symbol: {signal.symbol}
💰 Entry: ${signal.entry_price}
🛑 Stop Loss: ${signal.stop_loss} ({sl_pnl:+.2f}% @ 10x)

{tp_section}

{risk_emoji} Risk Level: {signal.risk_level}
💎 Risk/Reward: 1:{rr_ratio:.2f}

📊 RSI: {signal.rsi}
📈 Volume: {volume_text}
⚡ ATR: ${signal.atr}

📈 Support: ${signal.support_level}
📉 Resistance: ${signal.resistance_level}

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
                    await execute_trade_on_exchange(signal_data, user, db)
    
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
            
            # Scan for spot market flow signals
            from app.services.spot_monitor import spot_monitor
            spot_flows = await spot_monitor.scan_all_symbols()
            logger.info(f"Found {len(spot_flows)} spot flow signals")
            
            # Broadcast technical signals
            for signal in technical_signals:
                await broadcast_signal(signal)
            
            # Broadcast news signals
            for news_signal in news_signals:
                await broadcast_news_signal(news_signal)
            
            # Broadcast high-conviction spot flow alerts
            high_conviction_flows = [
                f for f in spot_flows 
                if f.get('confidence', 0) >= 70 and f.get('flow_signal') != 'NEUTRAL'
            ]
            
            for flow in high_conviction_flows:
                await broadcast_spot_flow_alert(flow)
                await spot_monitor.save_spot_activity(flow)
                
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


async def daily_pnl_report():
    """Send daily PnL summary at end of day (11:59 PM UTC)"""
    logger.info("Daily PnL report scheduler started")
    
    while True:
        try:
            # Calculate time until next report (11:59 PM UTC)
            now = datetime.utcnow()
            next_report = now.replace(hour=23, minute=59, second=0, microsecond=0)
            
            if now >= next_report:
                next_report += timedelta(days=1)
            
            sleep_seconds = (next_report - now).total_seconds()
            logger.info(f"Next daily report in {sleep_seconds/3600:.1f} hours")
            
            await asyncio.sleep(sleep_seconds)
            
            # Generate and send daily reports
            db = SessionLocal()
            try:
                users = db.query(User).filter(
                    User.approved == True,
                    User.banned == False
                ).all()
                
                for user in users:
                    try:
                        prefs = user.preferences
                        if not prefs or not prefs.dm_alerts:
                            continue
                        
                        # Get today's closed trades
                        start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
                        closed_trades = db.query(Trade).filter(
                            Trade.user_id == user.id,
                            Trade.closed_at >= start_today,
                            Trade.status.in_(['closed', 'stopped'])
                        ).all()
                        
                        # Get open positions
                        open_trades = db.query(Trade).filter(
                            Trade.user_id == user.id,
                            Trade.status == 'open'
                        ).all()
                        
                        # Calculate realized PnL
                        total_realized_pnl = sum(t.pnl or 0 for t in closed_trades)
                        winning_trades = len([t for t in closed_trades if (t.pnl or 0) > 0])
                        losing_trades = len([t for t in closed_trades if (t.pnl or 0) < 0])
                        
                        # Calculate unrealized PnL for open positions
                        total_unrealized_pnl = 0
                        open_positions_text = ""
                        
                        if open_trades:
                            exchange = ccxt.kucoin()
                            leverage = prefs.user_leverage if prefs else 10
                            
                            for trade in open_trades:
                                try:
                                    ticker = exchange.fetch_ticker(trade.symbol)
                                    current_price = ticker['last']
                                    
                                    if trade.direction == "LONG":
                                        pnl_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100 * leverage
                                    else:
                                        pnl_pct = ((trade.entry_price - current_price) / trade.entry_price) * 100 * leverage
                                    
                                    remaining_size = trade.remaining_size if trade.remaining_size > 0 else trade.position_size
                                    pnl_usd = (remaining_size * pnl_pct) / 100
                                    total_unrealized_pnl += pnl_usd
                                    
                                    pnl_emoji = "🟢" if pnl_usd > 0 else "🔴"
                                    open_positions_text += f"\n  {pnl_emoji} {trade.symbol} {trade.direction}: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)"
                                except:
                                    pass
                        
                        # Combined PnL
                        total_pnl = total_realized_pnl + total_unrealized_pnl
                        pnl_emoji = "🟢" if total_pnl > 0 else "🔴" if total_pnl < 0 else "⚪"
                        
                        # Build open positions text
                        if not open_positions_text:
                            open_positions_text = "\n  No open positions"
                        
                        # Build daily report
                        report = f"""
📊 <b>Daily PnL Report</b> - {now.strftime('%B %d, %Y')}
━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 <b>Realized PnL (Closed Trades)</b>
• Total: ${total_realized_pnl:+.2f}
• Trades: {len(closed_trades)} (✅ {winning_trades} | ❌ {losing_trades})
• Win Rate: {(winning_trades/len(closed_trades)*100) if closed_trades else 0:.1f}%

💹 <b>Unrealized PnL (Open Positions)</b>
• Total: ${total_unrealized_pnl:+.2f}
• Open: {len(open_trades)} position{'s' if len(open_trades) != 1 else ''}{open_positions_text}

{pnl_emoji} <b>Total Day PnL: ${total_pnl:+.2f}</b>

<i>Keep up the great trading! 📈</i>
"""
                        
                        await bot.send_message(user.telegram_id, report, parse_mode="HTML")
                        logger.info(f"Daily report sent to user {user.telegram_id}")
                        
                    except Exception as e:
                        logger.error(f"Error sending daily report to {user.telegram_id}: {e}")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Daily report error: {e}", exc_info=True)
            await asyncio.sleep(3600)  # Wait 1 hour on error


async def funding_rate_monitor():
    """Monitor funding rates and alert on extreme values"""
    from app.services.risk_filters import check_funding_rates, get_funding_rate_opportunity
    
    logger.info("Funding rate monitor started")
    await asyncio.sleep(60)  # Wait 1 minute before first check
    
    while True:
        try:
            logger.info("Checking funding rates...")
            
            # Check funding rates for all symbols
            symbols = settings.SYMBOLS.split(',')
            funding_alerts = await check_funding_rates(symbols)
            
            if funding_alerts:
                logger.info(f"Found {len(funding_alerts)} extreme funding rate alerts")
                
                # Broadcast to channel
                for alert in funding_alerts:
                    emoji = "🟢" if alert['alert_type'] == 'HIGH_SHORT_FUNDING' else "🔴"
                    alert_text = "SHORTS OVERLEVERAGED" if alert['alert_type'] == 'HIGH_SHORT_FUNDING' else "LONGS OVERLEVERAGED"
                    
                    opportunity = await get_funding_rate_opportunity(alert['symbol'], alert['funding_rate'])
                    
                    message = f"""
{emoji} <b>FUNDING RATE ALERT</b>
━━━━━━━━━━━━━━━━━━━━

⚡ <b>{alert_text}</b>
<b>Symbol:</b> {alert['symbol']}
<b>Current Funding:</b> {alert['funding_rate']:+.4f}%
<b>Daily Rate:</b> {alert['daily_rate']:+.4f}% (3x per day)

💡 <b>Opportunity</b>
"""
                    
                    if opportunity['action']:
                        message += f"• <b>Action:</b> {opportunity['action']} position\n"
                        message += f"• <b>Strategy:</b> {opportunity['reason']}\n"
                        message += f"• <b>Expected Daily Return:</b> {opportunity['expected_daily_return']:+.4f}%\n\n"
                        message += f"<i>💰 Arbitrage: {opportunity['action']} futures + hedge spot</i>"
                    else:
                        message += f"<i>Monitor for potential mean reversion</i>"
                    
                    try:
                        await bot.send_message(settings.BROADCAST_CHAT_ID, message, parse_mode="HTML")
                    except Exception as e:
                        logger.error(f"Error broadcasting funding alert: {e}")
                    
                    # Send to users with funding alerts enabled
                    db = SessionLocal()
                    try:
                        users = db.query(User).filter(
                            User.approved == True,
                            User.banned == False
                        ).all()
                        
                        for user in users:
                            if user.preferences and user.preferences.dm_alerts and user.preferences.funding_rate_alerts_enabled:
                                # Check if funding rate exceeds user threshold
                                if abs(alert['funding_rate']) >= user.preferences.funding_rate_threshold:
                                    try:
                                        await bot.send_message(user.telegram_id, message, parse_mode="HTML")
                                    except Exception as e:
                                        logger.error(f"Error sending funding alert to {user.telegram_id}: {e}")
                    finally:
                        db.close()
                        
        except Exception as e:
            logger.error(f"Funding rate monitor error: {e}", exc_info=True)
        
        await asyncio.sleep(3600)  # Check every hour


async def start_bot():
    logger.info("Starting Telegram bot...")
    
    # Initialize instance manager
    from app.services.bot_instance_manager import get_instance_manager
    manager = get_instance_manager(bot)
    
    # Try to acquire lock (prevent multiple instances)
    if not await manager.acquire_lock():
        logger.error("❌ Another bot instance is running. Attempting force stop...")
        # Try to force stop other instances
        if await manager.force_stop_other_instances():
            logger.info("✅ Forced stop successful, acquiring lock...")
            if not await manager.acquire_lock():
                logger.critical("❌ Could not acquire lock even after force stop. Exiting...")
                return
        else:
            logger.critical("❌ Could not force stop other instances. Exiting...")
            logger.critical("💡 Use /force_stop command via Telegram to resolve conflicts")
            return
    
    # Start conflict monitoring
    manager.monitor_task = asyncio.create_task(manager.start_conflict_monitor())
    
    # Start background tasks
    asyncio.create_task(signal_scanner())
    asyncio.create_task(position_monitor())
    asyncio.create_task(daily_pnl_report())
    asyncio.create_task(funding_rate_monitor())
    
    try:
        logger.info("Bot polling started")
        await dp.start_polling(bot)
    finally:
        # Cleanup on shutdown
        logger.info("Bot shutting down...")
        await manager.release_lock()
        await signal_generator.close()
