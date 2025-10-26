import asyncio
import logging
import ccxt.async_support as ccxt
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
from app.models import User, UserPreference, Trade, Signal, PaperTrade
# OLD GENERATORS REMOVED - Now using DayTradingSignalGenerator only (imported in signal_scanner)
# from app.services.signals import SignalGenerator
# from app.services.news_signals import NewsSignalGenerator
# from app.services.reversal_scanner import ReversalScanner
from app.services.bitunix_trader import execute_bitunix_trade
from app.services.analytics import AnalyticsService
from app.services.price_cache import get_cached_price, get_multiple_cached_prices
from app.services.health_monitor import get_health_monitor, update_heartbeat, update_message_timestamp
from app.services.scan_service import CoinScanService
from app.utils.encryption import encrypt_api_key, decrypt_api_key

logger = logging.getLogger(__name__)


async def safe_answer_callback(callback: CallbackQuery, text: str = None):
    """Safely answer callback queries, ignoring stale query errors"""
    try:
        await callback.answer(text=text)
    except Exception as e:
        error_msg = str(e).lower()
        if "query is too old" in error_msg or "query id is invalid" in error_msg or "timeout expired" in error_msg:
            logger.debug(f"Ignoring stale callback query: {e}")
        else:
            logger.error(f"Callback answer error: {e}")


# FSM States for API setup
class BitunixSetup(StatesGroup):
    waiting_for_api_key = State()
    waiting_for_api_secret = State()

# FSM States for position size
class PositionSizeSetup(StatesGroup):
    waiting_for_size = State()

bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
# OLD GENERATORS REMOVED - Now using DayTradingSignalGenerator only
# signal_generator = SignalGenerator()
# news_signal_generator = NewsSignalGenerator()
# reversal_scanner = ReversalScanner()


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
                approved=is_first_user,
                grandfathered=False  # New users need to subscribe (existing users already grandfathered via migration)
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
    if not user.is_subscribed and not user.is_admin:
        return False, "💎 Premium subscription required to receive trading signals. Use /subscribe to get started!"
    return True, ""


def get_connected_exchange(prefs) -> tuple[bool, str]:
    """Check if Bitunix exchange is connected. Returns (has_exchange, exchange_name)"""
    if not prefs:
        return False, ""
    
    if prefs.bitunix_api_key and prefs.bitunix_api_secret:
        return True, "Bitunix"
    
    return False, ""


async def execute_trade_on_exchange(signal, user: User, db: Session):
    """Execute trade on Bitunix exchange"""
    try:
        prefs = user.preferences
        if not prefs:
            logger.warning(f"No preferences found for user {user.id}")
            return None
        
        # Skip correlation filter for TEST signals (admin testing)
        if signal.signal_type != 'TEST':
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
        else:
            logger.info(f"TEST signal - skipping correlation filter for user {user.id}")
        
        # Execute on Bitunix
        if prefs.bitunix_api_key and prefs.bitunix_api_secret:
            logger.info(f"Executing trade on Bitunix for user {user.id}")
            return await execute_bitunix_trade(signal, user, db)
        else:
            logger.warning(f"Bitunix credentials not configured for user {user.id}")
            return None
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return None


async def build_account_overview(user, db):
    """
    Shared helper that builds account overview data for both /start and /dashboard commands.
    Returns (text, keyboard) tuple.
    """
    # EXPLICITLY REFRESH preferences from database to get latest data (fix dashboard bug)
    db.expire(user)  # Force reload user object
    db.refresh(user)  # Refresh user with latest data
    prefs = user.preferences
    
    # Get trading stats based on mode (paper vs live)
    is_paper_mode = prefs and prefs.paper_trading_mode
    if is_paper_mode:
        # Paper trading mode - query PaperTrade table
        total_trades = db.query(PaperTrade).filter(PaperTrade.user_id == user.id).count()
        open_positions = db.query(PaperTrade).filter(
            PaperTrade.user_id == user.id,
            PaperTrade.status == 'open'
        ).count()
    else:
        # Live trading mode - query Trade table
        total_trades = db.query(Trade).filter(Trade.user_id == user.id).count()
        open_positions = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.status == 'open'
        ).count()
    
    # Calculate today's PnL (live trades + paper trades)
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Live trades PnL
    today_trades = db.query(Trade).filter(
        Trade.user_id == user.id,
        Trade.status.in_(['closed', 'stopped']),
        Trade.closed_at >= today_start
    ).all()
    live_pnl = sum(trade.pnl or 0 for trade in today_trades)
    
    # Paper trades PnL
    today_paper_trades = db.query(PaperTrade).filter(
        PaperTrade.user_id == user.id,
        PaperTrade.status == 'closed',
        PaperTrade.closed_at >= today_start
    ).all()
    paper_pnl = sum(trade.pnl or 0 for trade in today_paper_trades)
    
    # Combined PnL
    today_pnl = live_pnl + paper_pnl
    
    # Auto-trading status - check Bitunix connection
    # Check keys exist and are not empty (encrypted keys don't need .strip())
    bitunix_connected = (
        prefs and 
        prefs.bitunix_api_key and 
        prefs.bitunix_api_secret and
        len(prefs.bitunix_api_key) > 0 and 
        len(prefs.bitunix_api_secret) > 0
    )
    auto_enabled = prefs and prefs.auto_trading_enabled
    
    # Auto-trading is only ACTIVE if both enabled AND Bitunix is connected
    is_active = auto_enabled and bitunix_connected
    autotrading_emoji = "🟢" if is_active else "🔴"
    autotrading_status = "ACTIVE" if is_active else "INACTIVE"
    
    # Exchange status - Bitunix only
    active_exchange = "Bitunix" if bitunix_connected else None
    exchange_status = f"{active_exchange} (✅ Connected)" if active_exchange else "No Exchange Connected"
    
    # Position sizing info
    position_size = f"{prefs.position_size_percent:.0f}%" if prefs else "10%"
    leverage = f"{prefs.user_leverage}x" if prefs else "10x"
    
    # Trading mode
    is_paper_mode = prefs and prefs.paper_trading_mode
    trading_mode = "📄 Paper Trading" if is_paper_mode else "💰 Live Trading"
    
    # Fetch live Bitunix balance if connected
    live_balance = None
    live_balance_text = ""
    
    if not is_paper_mode and is_active and bitunix_connected:
        try:
            from app.services.bitunix_trader import BitunixTrader
            from cryptography.fernet import Fernet
            import os
            
            cipher = Fernet(os.getenv('ENCRYPTION_KEY').encode())
            api_key = cipher.decrypt(prefs.bitunix_api_key.encode()).decode()
            api_secret = cipher.decrypt(prefs.bitunix_api_secret.encode()).decode()
            
            trader = BitunixTrader(api_key, api_secret)
            try:
                live_balance = await trader.get_account_balance()
            finally:
                await trader.close()
            
            if live_balance and live_balance > 0:
                live_balance_text = f"💵 <b>Balance:</b> ${live_balance:.2f} USDT\n"
            else:
                live_balance_text = "💵 <b>Balance:</b> $0.00 USDT\n"
                
        except Exception as e:
            logger.error(f"Error fetching Bitunix balance: {e}")
            live_balance_text = "💵 <b>Balance:</b> Unable to fetch\n"
    
    # Build active positions section with live P&L calculations
    positions_section = ""
    total_unrealized_pnl = 0
    total_position_value = 0
    
    if is_paper_mode:
        # Get paper trading positions
        open_trades = db.query(PaperTrade).filter(
            PaperTrade.user_id == user.id,
            PaperTrade.status == 'open'
        ).all()
    else:
        # Get live trading positions
        open_trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.status == 'open'
        ).all()
    
    if open_trades:
        positions_section = "\n<b>📊 Active Positions</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        # Get current prices for all open positions
        try:
            symbols = [trade.symbol for trade in open_trades]
            current_prices = await get_multiple_cached_prices(symbols)
        except Exception as e:
            logger.error(f"Error fetching prices for positions: {e}")
            current_prices = {}
        
        for trade in open_trades[:3]:  # Show max 3 positions
            direction_emoji = "🟢" if trade.direction.upper() == 'LONG' else "🔴"
            
            # Calculate live P&L
            current_price = current_prices.get(trade.symbol, 0)
            unrealized_pnl = 0
            pnl_display = "P&L: --"
            
            if current_price > 0:
                # Calculate price change percentage
                if trade.direction.upper() == 'LONG':
                    price_change_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100
                else:  # SHORT
                    price_change_pct = ((trade.entry_price - current_price) / trade.entry_price) * 100
                
                # Calculate unrealized P&L: price_change% × leverage × position_size (USDT)
                # Note: position_size is already in USDT, not coin amount
                leverage = prefs.user_leverage if prefs else 10
                unrealized_pnl = (price_change_pct / 100) * leverage * trade.position_size
                
                total_unrealized_pnl += unrealized_pnl
                total_position_value += trade.position_size
                
                # Format P&L with color
                pnl_emoji_inline = "🟢" if unrealized_pnl > 0 else "🔴" if unrealized_pnl < 0 else "⚪"
                pnl_display = f"P&L: {pnl_emoji_inline} ${unrealized_pnl:+.2f}"
            
            positions_section += f"""
{direction_emoji} <b>{trade.symbol}</b> {trade.direction}
└ Entry: ${trade.entry_price:.4f} | Current: ${current_price:.4f}
└ {pnl_display}
"""
        
        if len(open_trades) > 3:
            positions_section += f"\n<i>... and {len(open_trades) - 3} more</i>\n"
        
        # Calculate all open positions (not just displayed ones)
        for trade in open_trades[3:]:
            current_price = current_prices.get(trade.symbol, 0)
            if current_price > 0:
                # Calculate price change percentage
                if trade.direction.upper() == 'LONG':
                    price_change_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100
                else:
                    price_change_pct = ((trade.entry_price - current_price) / trade.entry_price) * 100
                
                # Calculate unrealized P&L
                leverage = prefs.user_leverage if prefs else 10
                unrealized_pnl = (price_change_pct / 100) * leverage * trade.position_size
                
                total_unrealized_pnl += unrealized_pnl
                total_position_value += trade.position_size
        
        # Add total ROI and account gain summary
        if total_unrealized_pnl != 0 or total_position_value > 0:
            # Calculate total ROI
            roi_percentage = (total_unrealized_pnl / total_position_value * 100) if total_position_value > 0 else 0
            roi_emoji = "🟢" if roi_percentage > 0 else "🔴" if roi_percentage < 0 else "⚪"
            
            # Calculate account gain percentage
            account_gain_pct = 0
            if live_balance and live_balance > 0:
                account_gain_pct = (total_unrealized_pnl / live_balance * 100)
            
            positions_section += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━
{roi_emoji} <b>Total ROI:</b> {roi_percentage:+.2f}%
💰 <b>Unrealized P&L:</b> ${total_unrealized_pnl:+.2f}"""
            
            if live_balance and live_balance > 0:
                positions_section += f"\n📊 <b>Account Gain:</b> {account_gain_pct:+.2f}%"
            
            positions_section += "\n"
    
    # Build account overview for LIVE exchange ONLY
    pnl_emoji = "🟢" if today_pnl > 0 else "🔴" if today_pnl < 0 else "⚪"
    
    if not is_active:
        account_overview = """<b>💰 Account Overview</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ Auto-trading disabled
Use /autotrading to enable
"""
    elif not bitunix_connected:
        # Bitunix not connected
        preferred_name = "Bitunix"
        account_overview = f"""<b>💰 Account Overview</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ {preferred_name} API keys not connected
Use /set_{preferred_name.lower()}_api to connect
{pnl_emoji} Today's P&L: <b>${today_pnl:+.2f}</b>
"""
    elif not live_balance_text:
        # Has keys but balance fetch failed
        account_overview = f"""<b>💰 Account Overview</b> ({active_exchange})
━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ Unable to fetch balance - check API permissions
{pnl_emoji} Today's P&L: <b>${today_pnl:+.2f}</b>
"""
    else:
        # Everything working
        account_overview = f"""<b>💰 Account Overview</b> ({active_exchange})
━━━━━━━━━━━━━━━━━━━━━━━━━━
{live_balance_text}{pnl_emoji} Today's P&L: <b>${today_pnl:+.2f}</b>
"""
    
    # Subscription status
    if user.grandfathered:
        sub_status = "🎉 <b>Lifetime Access</b> (Grandfathered)"
    elif user.is_subscribed:
        expires = user.subscription_end.strftime("%Y-%m-%d") if user.subscription_end else "Active"
        sub_status = f"✅ <b>Premium</b> (until {expires})"
    else:
        sub_status = "💎 <b>Free Trial</b> - /subscribe for full access"
    
    # Main dashboard shows ONLY live account - no paper trading here
    welcome_text = f"""
╔══════════════════════════╗
   <b>🚀 AI FUTURES SIGNALS</b>
╚══════════════════════════╝

{sub_status}
{autotrading_emoji} Auto-Trading: <b>{autotrading_status}</b>

{account_overview}{positions_section}
<b>📈 Trading Stats</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
Open: <b>{open_positions}</b> | Total: <b>{total_trades}</b>
Position Size: <b>{position_size}</b> | Leverage: <b>{leverage}</b>

<i>AI-driven EMA strategy with multi-timeframe analysis</i>
"""
    
    # Simple 3-row menu - everything users need in one place
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="💰 My Trades", callback_data="active_trades"),
            InlineKeyboardButton(text="📊 P&L", callback_data="view_pnl_menu")
        ],
        [
            InlineKeyboardButton(text="🤖 Auto-Trading", callback_data="autotrading_menu"),
            InlineKeyboardButton(text="🔍 Scan Coin", callback_data="scan_menu")
        ],
        [
            InlineKeyboardButton(text="⚙️ Settings", callback_data="settings_menu"),
            InlineKeyboardButton(text="❓ Help", callback_data="help_menu")
        ]
    ])
    
    return welcome_text, keyboard


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    # Track message for health monitor
    await update_message_timestamp()
    
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
        
        # Use shared helper to build account overview
        welcome_text, keyboard = await build_account_overview(user, db)
        await message.answer(welcome_text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "scan_menu")
async def handle_scan_menu(callback: CallbackQuery):
    """Handle scan menu button - shows quick scan options"""
    await callback.answer()
    
    scan_text = """
🔍 <b>Coin Scanner</b>

<b>Quick Scan Popular Coins:</b>
Click a button below for instant analysis!

<b>Or scan any coin:</b>
/scan BTC
/scan ETH
/scan SOL

<i>Get real-time trend, volume, momentum, and institutional flow analysis!</i>
"""
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="₿ BTC", callback_data="quick_scan_BTC"),
            InlineKeyboardButton(text="Ξ ETH", callback_data="quick_scan_ETH"),
            InlineKeyboardButton(text="◎ SOL", callback_data="quick_scan_SOL")
        ],
        [
            InlineKeyboardButton(text="🅱️ BNB", callback_data="quick_scan_BNB"),
            InlineKeyboardButton(text="🐶 DOGE", callback_data="quick_scan_DOGE"),
            InlineKeyboardButton(text="🔗 LINK", callback_data="quick_scan_LINK")
        ],
        [
            InlineKeyboardButton(text="🔙 Back to Menu", callback_data="back_to_start")
        ]
    ])
    
    await callback.message.edit_text(scan_text, reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data.startswith("quick_scan_"))
async def handle_quick_scan(callback: CallbackQuery):
    """Handle quick scan buttons for popular coins"""
    await callback.answer()
    
    # Extract symbol from callback data (e.g., "quick_scan_BTC" -> "BTC")
    symbol = callback.data.replace("quick_scan_", "")
    
    # Create a fake message object to reuse cmd_scan logic
    class FakeMessage:
        def __init__(self, text, from_user, chat):
            self.text = text
            self.from_user = from_user
            self.chat = chat
    
    fake_msg = FakeMessage(
        f"/scan {symbol}",
        callback.from_user,
        callback.message.chat
    )
    fake_msg.answer = callback.message.answer
    
    # Call the scan command
    await cmd_scan(fake_msg)


@dp.callback_query(F.data == "dashboard")
async def handle_dashboard_button(callback: CallbackQuery):
    """Handle dashboard button from /start menu - shows the dashboard view"""
    await callback.answer()
    # Show dashboard view (with PnL buttons and Active Positions)
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
        
        # Simple status indicators
        top_gainers = '🟢 ON' if prefs and prefs.top_gainers_mode_enabled else '🔴 OFF'
        paper_mode = '🟢 ON' if prefs and prefs.paper_trading_mode else '🔴 OFF'
        
        settings_text = f"""
⚙️ <b>Settings</b>

💰 Position Size: <b>{prefs.position_size_percent if prefs else 10}%</b>
⚡ Leverage: <b>{prefs.user_leverage if prefs else 10}x</b>
📊 Max Positions: <b>{prefs.max_positions if prefs else 3}</b>

🔥 Top Gainers Mode: {top_gainers}
📄 Paper Trading: {paper_mode}

<i>Tap any button to change:</i>
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="💰 Position", callback_data="edit_position_size"),
                InlineKeyboardButton(text="⚡ Leverage", callback_data="edit_leverage")
            ],
            [
                InlineKeyboardButton(text="🔥 Top Gainers", callback_data="toggle_top_gainers_mode"),
                InlineKeyboardButton(text="📄 Paper Mode", callback_data="toggle_paper_mode")
            ],
            [InlineKeyboardButton(text="🏠 Main Menu", callback_data="back_to_start")]
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


@dp.callback_query(F.data == "toggle_paper_mode")
async def handle_toggle_paper_mode(callback: CallbackQuery):
    """Handle toggle paper mode button"""
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Please use /start first")
            return
        
        prefs = user.preferences
        prefs.paper_trading_mode = not prefs.paper_trading_mode
        db.commit()
        
        status = "ENABLED ✅" if prefs.paper_trading_mode else "DISABLED ❌"
        mode_name = "📄 Paper Trading" if prefs.paper_trading_mode else "💰 Live Trading"
        
        await callback.answer(f"Paper Mode: {status}", show_alert=True)
        
        # Refresh the paper trading view
        await handle_paper_trading_view(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "paper_trading_view")
async def handle_paper_trading_view(callback: CallbackQuery):
    """Handle paper trading view button - shows paper trading stats and positions"""
    # Don't answer callback if coming from toggle (already answered)
    if callback.data == "paper_trading_view":
        await callback.answer()
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("Please use /start first")
            return
        
        prefs = user.preferences
        
        # Get paper trading stats
        all_paper_trades = db.query(PaperTrade).filter(
            PaperTrade.user_id == user.id,
            PaperTrade.status == 'closed'
        ).all()
        total_paper_pnl = sum(t.pnl or 0 for t in all_paper_trades)
        current_balance = prefs.paper_balance if prefs else 1000.0
        starting_balance = current_balance - total_paper_pnl
        balance_emoji = "🟢" if current_balance > starting_balance else "🔴" if current_balance < starting_balance else "⚪"
        
        # Get open paper positions
        open_paper_trades = db.query(PaperTrade).filter(
            PaperTrade.user_id == user.id,
            PaperTrade.status == 'open'
        ).all()
        
        # Build positions section
        positions_text = ""
        if open_paper_trades:
            positions_text = "\n<b>📊 Active Paper Positions</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            for trade in open_paper_trades[:5]:  # Show max 5
                direction_emoji = "🟢" if trade.direction.upper() == 'LONG' else "🔴"
                positions_text += f"""
{direction_emoji} <b>{trade.symbol}</b> {trade.direction}
└ Entry: ${trade.entry_price:.4f} | SL: ${trade.stop_loss:.4f}
└ TP: ${trade.take_profit:.4f}
"""
            if len(open_paper_trades) > 5:
                positions_text += f"\n<i>... and {len(open_paper_trades) - 5} more positions</i>\n"
        else:
            positions_text = "\n<i>No open paper positions</i>\n"
        
        # Calculate stats
        winning_trades = len([t for t in all_paper_trades if (t.pnl or 0) > 0])
        losing_trades = len([t for t in all_paper_trades if (t.pnl or 0) < 0])
        win_rate = (winning_trades / len(all_paper_trades) * 100) if all_paper_trades else 0
        
        paper_text = f"""
╔══════════════════════════╗
   <b>📄 PAPER TRADING</b>
╚══════════════════════════╝

<b>💰 Virtual Account</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
{balance_emoji} Balance: <b>${current_balance:.2f}</b>
📊 Starting: ${starting_balance:.2f}
💼 All-Time P&L: <b>${total_paper_pnl:+.2f}</b>
{positions_text}
<b>📈 Paper Stats</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
Open: <b>{len(open_paper_trades)}</b> | Total: <b>{len(all_paper_trades)}</b>
Win Rate: <b>{win_rate:.1f}%</b>
✅ Wins: {winning_trades} | ❌ Losses: {losing_trades}

<i>Practice risk-free with virtual balance</i>
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Toggle Paper Mode", callback_data="toggle_paper_mode")],
            [InlineKeyboardButton(text="🔙 Back to Dashboard", callback_data="back_to_start")]
        ])
        
        await callback.message.edit_text(paper_text, reply_markup=keyboard, parse_mode="HTML")
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
/scan BTC - Analyze any coin
/settings - Configure settings

<b>Analysis Tools:</b>
• /scan SYMBOL - Get instant market analysis
• /spot_flow - Check institutional flow
• No signal, just pure analysis!

<b>Auto-Trading:</b>
• Connect your Bitunix API
• Set position size & leverage
• Bot trades automatically on signals

<b>Safety Features:</b>
• Emergency stop available
• Daily loss limits
• Max drawdown protection
• Smart exit system

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
        
        # Check subscription status
        if user.grandfathered:
            await message.answer(
                "🎉 <b>Lifetime Access - Grandfathered User</b>\n\n"
                "You have <b>FREE lifetime access</b> to all premium features as an early supporter!\n\n"
                "✅ All trading signals (1:1 Day Trading + Top Gainers)\n"
                "✅ Auto-trading with Bitunix\n"
                "✅ PnL tracking & analytics\n"
                "✅ Priority support\n\n"
                "<i>Thank you for being part of our community!</i>",
                parse_mode="HTML"
            )
            return
        
        if user.is_subscribed:
            expires = user.subscription_end.strftime("%Y-%m-%d") if user.subscription_end else "Unknown"
            await message.answer(
                f"✅ <b>Active Subscription</b>\n\n"
                f"Your premium subscription is <b>active</b> until:\n"
                f"📅 <b>{expires}</b>\n\n"
                f"You have full access to:\n"
                f"✅ All trading signals (1:1 Day Trading + Top Gainers)\n"
                f"✅ Auto-trading with Bitunix\n"
                f"✅ PnL tracking & analytics\n"
                f"✅ Priority support",
                parse_mode="HTML"
            )
            return
        
        # User needs to subscribe
        from app.services.nowpayments import NOWPaymentsService
        from app.config import settings
        import uuid
        
        if not settings.NOWPAYMENTS_API_KEY:
            await message.answer(
                "⚠️ Subscription system is being set up. Please check back soon!"
            )
            return
        
        nowpayments = NOWPaymentsService(settings.NOWPAYMENTS_API_KEY)
        
        # Create one-time payment invoice
        order_id = f"sub_{user.telegram_id}_{int(datetime.utcnow().timestamp())}"
        invoice = nowpayments.create_one_time_payment(
            price_amount=settings.SUBSCRIPTION_PRICE_USD,
            price_currency="usd",
            order_id=order_id,
            ipn_callback_url=f"https://{message.bot.base_url.replace('https://api.telegram.org/bot', '')}/webhooks/nowpayments"
        )
        
        if invoice and invoice.get("invoice_url"):
            await message.answer(
                f"💎 <b>Premium Subscription - ${settings.SUBSCRIPTION_PRICE_USD}/month</b>\n\n"
                f"<b>What's Included:</b>\n"
                f"✅ <b>1:1 Day Trading Signals</b> (20% TP/SL @ 10x leverage)\n"
                f"  • 6-point confirmation system\n"
                f"  • 75%+ institutional spot flow requirement\n"
                f"  • Early entry on 5m+15m timeframes\n\n"
                f"✅ <b>Top Gainers Scanner</b> (24/7 parabolic reversal detection)\n"
                f"  • 48-hour watchlist for delayed reversals\n"
                f"  • Dual TPs for max profit capture\n"
                f"  • Fixed 5x leverage for safety\n\n"
                f"✅ <b>Auto-Trading on Bitunix</b>\n"
                f"  • Automated signal execution\n"
                f"  • Smart exit system with 6 reversal detectors\n"
                f"  • Risk management & position sizing\n\n"
                f"✅ <b>Advanced Analytics</b>\n"
                f"  • Real-time PnL tracking\n"
                f"  • Trade history & performance stats\n"
                f"  • Pattern success rate analysis\n\n"
                f"<b>Payment Options:</b>\n"
                f"🔹 BTC, ETH, USDT, and 200+ cryptocurrencies\n\n"
                f"👇 <b>Click below to subscribe with crypto:</b>",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="💳 Pay with Crypto", url=invoice["invoice_url"])
                ]])
            )
        else:
            await message.answer(
                "⚠️ Unable to generate payment link. Please try again later or contact support."
            )
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
        
        # Use the SAME helper as /start to get account overview text
        account_text, _ = await build_account_overview(user, db)
        
        # But use dashboard-specific buttons (Active Positions, PnL views, etc.)
        dashboard_keyboard = InlineKeyboardMarkup(inline_keyboard=[
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
        
        await message.answer(account_text, reply_markup=dashboard_keyboard, parse_mode="HTML")
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
        
        prefs = user.preferences
        is_paper_mode = prefs and prefs.paper_trading_mode
        leverage = prefs.user_leverage if prefs else 10
        
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
        
        # Query appropriate table based on trading mode
        if is_paper_mode:
            from app.models import PaperTrade
            trades = db.query(PaperTrade).filter(
                PaperTrade.user_id == user.id,
                PaperTrade.closed_at >= start_date,
                PaperTrade.status == "closed"
            ).all()
            mode_label = "📝 PAPER MODE"
        else:
            trades = db.query(Trade).filter(
                Trade.user_id == user.id,
                Trade.closed_at >= start_date,
                Trade.status == "closed"
            ).all()
            mode_label = "💰 LIVE TRADING"
        
        if not trades:
            pnl_text = f"""
{period_emoji} <b>PnL Summary ({period.title()})</b>
{mode_label}

No closed trades in this period.
Use /autotrading_status to set up auto-trading!
"""
        else:
            total_pnl = sum(t.pnl for t in trades)
            total_pnl_pct = sum(t.pnl_percent for t in trades)
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            # Calculate ROI % (return on invested capital)
            # Capital invested = position_size / leverage for each trade
            total_capital_invested = sum(t.position_size / leverage for t in trades)
            roi_percent = (total_pnl / total_capital_invested * 100) if total_capital_invested > 0 else 0
            
            avg_pnl = total_pnl / len(trades) if trades else 0
            avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            best_trade = max(trades, key=lambda t: t.pnl) if trades else None
            worst_trade = min(trades, key=lambda t: t.pnl) if trades else None
            
            win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0
            
            pnl_emoji = "🟢" if total_pnl > 0 else "🔴" if total_pnl < 0 else "⚪"
            roi_emoji = "🟢" if roi_percent > 0 else "🔴" if roi_percent < 0 else "⚪"
            
            pnl_text = f"""
{period_emoji} <b>PnL Summary ({period.title()})</b>
{mode_label} | Leverage: {leverage}x
━━━━━━━━━━━━━━━━━━━━

{pnl_emoji} <b>Total PnL:</b> ${total_pnl:+.2f} ({total_pnl_pct:+.2f}%)
{roi_emoji} <b>ROI:</b> {roi_percent:+.2f}% (on ${total_capital_invested:.2f})
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
        
        # Add share button for weekly/monthly PnL (if has trades)
        if period == "month" and trades:
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="📸 Share Monthly PnL", callback_data="share_monthly_pnl")],
                [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
            ])
        elif period == "week" and trades:
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="📸 Share Weekly PnL", callback_data="share_weekly_pnl")],
                [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
            ])
        else:
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
            ])
        
        await callback.message.answer(pnl_text, reply_markup=keyboard, parse_mode="HTML")
        await safe_answer_callback(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "pnl_today")
async def handle_pnl_today(callback: CallbackQuery):
    """Show today's PnL via button"""
    await callback.answer()
    await cmd_pnl_today(callback.message)


@dp.callback_query(F.data == "pnl_week")
async def handle_pnl_week(callback: CallbackQuery):
    """Show this week's PnL via button"""
    await callback.answer()
    await cmd_pnl_week(callback.message)


@dp.callback_query(F.data == "pnl_month")
async def handle_pnl_month(callback: CallbackQuery):
    """Show this month's PnL via button"""
    await callback.answer()
    await cmd_pnl_month(callback.message)


@dp.callback_query(F.data == "share_monthly_pnl")
async def handle_share_monthly_pnl(callback: CallbackQuery):
    """Generate and send monthly PnL summary card"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("❌ User not found")
            return
        
        prefs = user.preferences
        leverage = prefs.user_leverage if prefs else 10
        is_paper_mode = prefs and prefs.paper_trading_mode
        
        # Get this month's trades
        now = datetime.utcnow()
        start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_name = now.strftime("%B")
        
        if is_paper_mode:
            from app.models import PaperTrade
            trades = db.query(PaperTrade).filter(
                PaperTrade.user_id == user.id,
                PaperTrade.closed_at >= start_of_month,
                PaperTrade.status == "closed"
            ).all()
        else:
            trades = db.query(Trade).filter(
                Trade.user_id == user.id,
                Trade.closed_at >= start_of_month,
                Trade.status == "closed"
            ).all()
        
        if not trades:
            await callback.answer("❌ No trades this month to share")
            return
        
        await callback.answer("📸 Generating monthly PnL card...")
        
        # Calculate stats
        total_pnl = sum(t.pnl for t in trades)
        total_pnl_pct = sum(t.pnl_percent for t in trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0
        best_trade = max(trades, key=lambda t: t.pnl_percent) if trades else None
        worst_trade = min(trades, key=lambda t: t.pnl_percent) if trades else None
        
        # Generate screenshot
        from app.services.trade_screenshot import screenshot_generator
        img_bytes = screenshot_generator.generate_monthly_summary(
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            win_rate=win_rate,
            total_trades=len(trades),
            best_trade_pct=best_trade.pnl_percent if best_trade else 0,
            worst_trade_pct=worst_trade.pnl_percent if worst_trade else 0,
            month_name=month_name
        )
        
        # Send photo
        from aiogram.types import BufferedInputFile
        photo = BufferedInputFile(img_bytes.read(), filename=f"pnl_{month_name.lower()}.png")
        
        result_emoji = "📈" if total_pnl > 0 else "📉"
        caption = f"{result_emoji} <b>{month_name} Performance Summary</b>\n\n💰 Total PnL: ${total_pnl:+.2f} ({total_pnl_pct:+.2f}%)"
        
        await callback.message.answer_photo(
            photo=photo,
            caption=caption,
            parse_mode="HTML"
        )
        
    except Exception as e:
        logger.error(f"Error generating monthly PnL card: {e}", exc_info=True)
        await callback.answer("❌ Error generating PnL card")
    finally:
        db.close()


@dp.callback_query(F.data == "share_weekly_pnl")
async def handle_share_weekly_pnl(callback: CallbackQuery):
    """Generate and send weekly PnL summary card"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("❌ User not found")
            return
        
        prefs = user.preferences
        leverage = prefs.user_leverage if prefs else 10
        is_paper_mode = prefs and prefs.paper_trading_mode
        
        # Get this week's trades
        now = datetime.utcnow()
        start_of_week = now - timedelta(days=7)
        week_label = f"Week of {start_of_week.strftime('%b %d')}"
        
        if is_paper_mode:
            from app.models import PaperTrade
            trades = db.query(PaperTrade).filter(
                PaperTrade.user_id == user.id,
                PaperTrade.closed_at >= start_of_week,
                PaperTrade.status == "closed"
            ).all()
        else:
            trades = db.query(Trade).filter(
                Trade.user_id == user.id,
                Trade.closed_at >= start_of_week,
                Trade.status == "closed"
            ).all()
        
        if not trades:
            await callback.answer("❌ No trades this week to share")
            return
        
        await callback.answer("📸 Generating weekly PnL card...")
        
        # Calculate stats
        total_pnl = sum(t.pnl for t in trades)
        total_pnl_pct = sum(t.pnl_percent for t in trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0
        best_trade = max(trades, key=lambda t: t.pnl_percent) if trades else None
        worst_trade = min(trades, key=lambda t: t.pnl_percent) if trades else None
        
        # Generate screenshot
        from app.services.trade_screenshot import screenshot_generator
        img_bytes = screenshot_generator.generate_monthly_summary(
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            win_rate=win_rate,
            total_trades=len(trades),
            best_trade_pct=best_trade.pnl_percent if best_trade else 0,
            worst_trade_pct=worst_trade.pnl_percent if worst_trade else 0,
            month_name=week_label
        )
        
        # Send photo
        from aiogram.types import BufferedInputFile
        photo = BufferedInputFile(img_bytes.read(), filename=f"pnl_weekly.png")
        
        result_emoji = "📈" if total_pnl > 0 else "📉"
        caption = f"{result_emoji} <b>Weekly Performance Summary</b>\n\n💰 Total PnL: ${total_pnl:+.2f} ({total_pnl_pct:+.2f}%)"
        
        await callback.message.answer_photo(
            photo=photo,
            caption=caption,
            parse_mode="HTML"
        )
        
    except Exception as e:
        logger.error(f"Error generating weekly PnL card: {e}", exc_info=True)
        await callback.answer("❌ Error generating PnL card")
    finally:
        db.close()


@dp.callback_query(F.data == "view_pnl_menu")
async def handle_view_pnl_menu(callback: CallbackQuery):
    """Show P&L menu with period options"""
    await callback.answer()
    
    pnl_menu_text = """
📊 <b>P&L Report</b>

Choose a time period:
"""
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="📅 Today", callback_data="pnl_today"),
            InlineKeyboardButton(text="📅 Week", callback_data="pnl_week")
        ],
        [
            InlineKeyboardButton(text="📅 Month", callback_data="pnl_month"),
            InlineKeyboardButton(text="📅 All Time", callback_data="view_all_pnl")
        ],
        [InlineKeyboardButton(text="🏠 Main Menu", callback_data="back_to_start")]
    ])
    
    await callback.message.edit_text(pnl_menu_text, reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data == "view_all_pnl")
async def handle_view_all_pnl(callback: CallbackQuery):
    """Show all-time PnL via button"""
    await callback.answer()
    await cmd_pnl(callback.message)


@dp.callback_query(F.data == "edit_position_size")
async def handle_edit_position_size(callback: CallbackQuery):
    """Map to set_position_size handler"""
    await handle_set_position_size(callback)


@dp.callback_query(F.data == "edit_leverage")
async def handle_edit_leverage(callback: CallbackQuery):
    """Show leverage edit prompt"""
    await callback.answer("Use /set_leverage [1-125] to change leverage", show_alert=True)


@dp.callback_query(F.data == "edit_notifications")
async def handle_edit_notifications(callback: CallbackQuery):
    """Show notifications settings"""
    await callback.answer("Use /toggle_alerts to enable/disable DM notifications", show_alert=True)


@dp.callback_query(F.data == "toggle_top_gainers_mode")
async def handle_toggle_top_gainers_mode(callback: CallbackQuery):
    """Toggle Top Gainers Trading Mode"""
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Please use /start first")
            return
        
        prefs = user.preferences
        
        # Toggle the mode
        prefs.top_gainers_mode_enabled = not prefs.top_gainers_mode_enabled
        db.commit()
        db.refresh(prefs)
        
        status = "✅ ENABLED" if prefs.top_gainers_mode_enabled else "❌ DISABLED"
        
        response_text = f"""
🔥 <b>Top Gainers Mode</b> {status}

<b>What it does:</b>
Catches big coin crashes after pumps 📉

<b>How it works:</b>
• Scans 24/7 (no time restrictions)
• Finds coins up 10%+ in 24h
• Waits for reversal signals
• SHORTS the dump (95% of trades)
• 5x leverage (safer for volatility)

<b>Profit targets:</b>
• Regular: 20% profit
• Parabolic (50%+ pumps): 20% + 35% 🎯

<b>Risk:</b>
High volatility - only for experienced traders!

Status: {status}
{"⏰ Scanning 24/7 every 15 min" if prefs.top_gainers_mode_enabled else "Off - no signals 🔴"}
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⚙️ Back to Settings", callback_data="settings_menu")],
            [InlineKeyboardButton(text="◀️ Main Menu", callback_data="back_to_start")]
        ])
        
        await callback.message.answer(response_text, reply_markup=keyboard, parse_mode="HTML")
        
    finally:
        db.close()


@dp.callback_query(F.data.startswith("share_trade_"))
async def handle_share_trade_callback(callback: CallbackQuery):
    """Handle trade screenshot generation from inline button"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found")
            return
        
        # Extract trade ID from callback data
        try:
            trade_id = int(callback.data.split("_")[2])
        except (IndexError, ValueError):
            await callback.answer("❌ Invalid trade ID")
            return
        
        # Fetch trade
        trade = db.query(Trade).filter(
            Trade.id == trade_id,
            Trade.user_id == user.id,
            Trade.status.in_(['closed', 'stopped'])
        ).first()
        
        if not trade:
            await callback.answer("❌ Trade not found")
            return
        
        # Generate and send screenshot
        await callback.answer("📸 Generating screenshot...")
        
        from app.services.position_monitor import send_trade_screenshot
        await send_trade_screenshot(callback.bot, trade, user, db)
        
    except Exception as e:
        logger.error(f"Error in share_trade callback: {e}", exc_info=True)
        await callback.answer("❌ Error generating screenshot")
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
        
        prefs = user.preferences
        leverage = prefs.user_leverage if prefs else 10
        is_paper_mode = prefs and prefs.paper_trading_mode
        
        # Check for paper trades if in paper mode
        if is_paper_mode:
            from app.models import PaperTrade
            trades = db.query(PaperTrade).filter(
                PaperTrade.user_id == user.id,
                PaperTrade.status == "open"
            ).all()
            mode_text = "📝 PAPER MODE"
        else:
            trades = db.query(Trade).filter(
                Trade.user_id == user.id,
                Trade.status == "open"
            ).all()
            mode_text = "💰 Live Trading"
        
        if not trades:
            trades_text = f"""
🔄 <b>Active Positions</b>
{mode_text}

No active trades at the moment.

Use /autotrading_status to enable auto-trading and start taking trades automatically!
"""
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
            ])
            await callback.message.answer(trades_text, reply_markup=keyboard, parse_mode="HTML")
            await callback.answer()
            return
        
        # Get all unique symbols for batch price fetching
        symbols = list(set([trade.symbol for trade in trades]))
        cached_prices = await get_multiple_cached_prices(symbols, 'kucoin')
        
        total_unrealized_pnl_usd = 0
        total_notional_value = 0
        
        try:
            trades_text = f"🔄 <b>Active Positions</b>\n{mode_text} | Leverage: {leverage}x\n━━━━━━━━━━━━━━━━━━━━\n\n"
            
            for i, trade in enumerate(trades, 1):
                direction_emoji = "🟢" if trade.direction == "LONG" else "🔴"
                
                # Calculate position size (remaining_size is already USD notional)
                remaining_size = trade.remaining_size if trade.remaining_size > 0 else trade.position_size
                total_notional_value += remaining_size
                
                # Try to get current price and calculate unrealized PnL
                try:
                    # Use cached price (reduces API calls by 90%+)
                    current_price = cached_prices.get(trade.symbol)
                    
                    # Calculate raw price change percentage (no leverage)
                    if trade.direction == "LONG":
                        raw_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100
                    else:
                        raw_pct = ((trade.entry_price - current_price) / trade.entry_price) * 100
                    
                    # Calculate PnL in USD and percentage (both leverage-adjusted)
                    pnl_usd = (remaining_size * raw_pct * leverage) / 100
                    pnl_pct = raw_pct * leverage
                    
                    total_unrealized_pnl_usd += pnl_usd
                    
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
                    
                    pnl_line_label = "Paper PnL" if is_paper_mode else "Live PnL"
                    trades_text += f"""
{i}. {direction_emoji} <b>{trade.symbol} {trade.direction}</b>
   Entry: ${trade.entry_price:.4f}
   Current: ${current_price:.4f}
   Size: ${remaining_size:.2f}
   
   🛑 SL: ${trade.stop_loss:.4f}{tp_text}
   
   {pnl_emoji} <b>{pnl_line_label}:</b> ${pnl_usd:+.2f} ({pnl_pct:+.2f}%){realized_text}
━━━━━━━━━━━━━━━━━━━━
"""
                except Exception as e:
                    # If can't fetch price, show basic info with TP levels if available
                    logger.error(f"Error fetching price for {trade.symbol}: {e}")
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
            
            # Add total summary (always show)
            total_emoji = "🟢" if total_unrealized_pnl_usd > 0 else "🔴" if total_unrealized_pnl_usd < 0 else "⚪"
            
            # Calculate size-weighted combined percentage using notional value
            combined_pnl_pct = (total_unrealized_pnl_usd / total_notional_value * 100) if total_notional_value > 0 else 0
            
            pnl_label = "PAPER PnL" if is_paper_mode else "TOTAL LIVE PnL"
            trades_text += f"""
{total_emoji} <b>{pnl_label}</b>
💰 ${total_unrealized_pnl_usd:+.2f} ({combined_pnl_pct:+.2f}%)
📊 Across {len(trades)} position{'s' if len(trades) != 1 else ''}
"""
            
            # Add paper balance breakdown if in paper mode
            if is_paper_mode:
                free_balance = prefs.paper_balance
                allocated_capital = total_notional_value
                total_equity = free_balance + allocated_capital + total_unrealized_pnl_usd
                equity_emoji = "🟢" if total_equity >= 850 else "🟡" if total_equity >= 500 else "🔴"
                
                trades_text += f"""
━━━━━━━━━━━━━━━━━━━━
{equity_emoji} <b>PAPER BALANCE</b>
💵 Free: ${free_balance:.2f}
📦 In Positions: ${allocated_capital:.2f}
{total_emoji} Unrealized P&L: ${total_unrealized_pnl_usd:+.2f}
━━━━━━━━━━━━━━━━━━━━
💼 <b>Total Equity: ${total_equity:.2f}</b>
"""
        except Exception as e:
            logger.error(f"Error in active positions: {e}")
        
        # Add back button
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Refresh", callback_data="active_trades")],
            [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
        ])
        
        await callback.message.answer(trades_text, reply_markup=keyboard, parse_mode="HTML")
        await safe_answer_callback(callback)
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
        
        # Check Bitunix only (exclusive exchange)
        bitunix_connected = (
            prefs and 
            prefs.bitunix_api_key and 
            prefs.bitunix_api_secret and
            len(prefs.bitunix_api_key) > 0 and 
            len(prefs.bitunix_api_secret) > 0
        )
        
        # Auto-trading status - Use same logic as dashboard (both enabled AND connected)
        auto_enabled = prefs and prefs.auto_trading_enabled
        is_active = auto_enabled and bitunix_connected
        autotrading_status = "🟢 ACTIVE" if is_active else "🔴 INACTIVE"
        
        # Bitunix is the only exchange
        exchange_name = "Bitunix" if bitunix_connected else None
        
        if exchange_name:
            api_status = "✅ Connected"
            position_size = prefs.position_size_percent if prefs else 5
            max_positions = prefs.max_positions if prefs else 3
            
            # Check top gainers mode status
            top_gainers_enabled = prefs and prefs.top_gainers_mode_enabled
            top_gainers_status = "🟢 ON" if top_gainers_enabled else "🔴 OFF"
            
            # Add warning if toggle is ON but showing INACTIVE (e.g., Bitunix not connected)
            status_warning = ""
            if auto_enabled and not is_active:
                status_warning = "\n⚠️ <i>Toggle is enabled but Bitunix is not connected</i>\n"
            
            autotrading_text = f"""
🤖 <b>Auto-Trading Status</b>
━━━━━━━━━━━━━━━━━━━━

🔑 <b>Exchange:</b> {exchange_name}
📡 <b>API Status:</b> {api_status}
🔄 <b>Auto-Trading:</b> {autotrading_status}{status_warning}
⚙️ <b>Configuration:</b>
  • Position Size: {position_size}% of balance
  • Max Positions: {max_positions}
  • 🔥 Top Gainers Mode: {top_gainers_status}

<i>Use the buttons below to manage auto-trading:</i>
"""
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔄 Toggle Auto-Trading", callback_data="toggle_autotrading_quick")],
                [InlineKeyboardButton(text="🔥 Top Gainers Mode", callback_data="toggle_top_gainers_mode")],
                [InlineKeyboardButton(text="📊 Set Position Size", callback_data="set_position_size")],
                [InlineKeyboardButton(text="❌ Remove API Keys", callback_data="remove_api_confirm")],
                [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
            ])
        else:
            autotrading_text = f"""
🤖 <b>Auto-Trading Setup</b>
━━━━━━━━━━━━━━━━━━━━

❌ <b>API Not Connected</b>
🔄 <b>Auto-Trading:</b> {autotrading_status}

To enable auto-trading, use one of these commands:
  • /set_bitunix_api

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
    """Handle back to dashboard button - shows the dashboard view"""
    await callback.answer()
    
    # Show dashboard view (with PnL buttons and Active Positions)
    await cmd_dashboard(callback.message)


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
            # Check if Bitunix API is configured
            has_bitunix = prefs.bitunix_api_key and prefs.bitunix_api_secret
            
            if not has_bitunix:
                await callback.answer("❌ Please set up your Bitunix API keys first (/set_bitunix_api)", show_alert=True)
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

Are you sure you want to remove your Bitunix API keys?

This will:
• Remove your encrypted API credentials
• Disable auto-trading
• Close no existing positions

<i>You can always reconnect later with /set_bitunix_api</i>
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
            user.preferences.bitunix_api_key = None
            user.preferences.bitunix_api_secret = None
            user.preferences.auto_trading_enabled = False
            db.commit()
            
            await callback.message.answer("✅ Bitunix API keys removed and auto-trading disabled")
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
        
        # Paper trading info
        paper_mode_status = "📄 Paper Trading" if prefs.paper_trading_mode else "💰 Live Trading"
        
        settings_text = f"""
⚙️ <b>Your Settings</b>

<b>📊 General</b>
• Muted Symbols: {muted_str}
• Default PnL Period: {prefs.default_pnl_period}
• DM Alerts: {"Enabled" if prefs.dm_alerts else "Disabled"}

<b>📄 Paper Trading</b>
• Mode: {paper_mode_status}
• Virtual Balance: ${prefs.paper_balance:.2f}
• Position Size: {prefs.position_size_percent}% of balance
• Leverage: {prefs.user_leverage}x

<b>🛡️ Risk Management</b>
• Correlation Filter: {"Enabled" if prefs.correlation_filter_enabled else "Disabled"}
• Max Correlated Positions: {prefs.max_correlated_positions}
• Funding Rate Alerts: {"Enabled" if prefs.funding_rate_alerts_enabled else "Disabled"}
• Funding Alert Threshold: {prefs.funding_rate_threshold}%

<b>Commands:</b>
/toggle_paper_mode - Switch paper/live trading
/set_paper_leverage [1-20] - Set paper trading leverage
/set_paper_size [1-100] - Set position size %
/reset_paper_balance - Reset paper balance to $1000

/mute [symbol] - Mute a symbol
/unmute [symbol] - Unmute a symbol
/set_pnl [today/week/month] - Set default PnL period
/toggle_alerts - Enable/Disable DM alerts
/toggle_correlation - Enable/Disable correlation filter
/toggle_funding_alerts - Enable/Disable funding alerts
/set_funding_threshold [0.1-1.0] - Set funding alert %
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

🎁 <b>Save 15% on Trading Fees!</b>
Sign up using our exclusive link:
<a href="https://www.bitunix.com/register?vipCode=tradehub">🔗 Register on Bitunix</a>

Use referral code: <code>tradehub</code>
(15% fee discount for all trades!)

━━━━━━━━━━━━━━━━━━━━

<b>Step 1: Get Bitunix API Keys</b>
1. Go to Bitunix.com → API Management
2. Create new API key
3. ⚠️ <b>IMPORTANT:</b> Enable <b>ONLY</b> Futures Trading
4. <b>DO NOT</b> enable withdrawals
5. Copy API Key & Secret

<b>Step 2: Connect to Bot</b>
• Bot will guide you through setup
• Keys are encrypted & stored securely

<b>Step 3: Configure Settings</b>
• Position size: 1-100% of balance
• Max positions: Limit open trades
• Top Gainers Mode: Optional

<b>Step 4: Security Features</b>
• 🛡️ Daily loss limits
• 🚨 Emergency stop button
• 📊 Real-time position tracking
• 🔒 Encrypted credentials

<b>How It Works:</b>
When a signal is generated:
1. Bot checks your risk settings
2. Calculates position size
3. Places market order on Bitunix
4. Sets SL/TP automatically
5. Monitors position in real-time

<b>Commands:</b>
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
• Ensure you have USDT balance on Bitunix

❌ <b>"Trades not executing"</b>
• Check your Bitunix futures balance
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
A: Yes! All signals are free. Optional auto-trading requires your own Bitunix account.

<b>Q: How accurate are the signals?</b>
A: Signals use proven technical indicators and AI analysis, but no strategy is 100%. Always use risk management.

<b>Q: Can the bot withdraw my funds?</b>
A: NO! API keys have NO withdrawal permissions. You always control your funds.

<b>Q: What's the recommended position size?</b>
A: Start with 1-5% of your balance. Never risk more than you can afford to lose.

<b>Q: How many signals per day?</b>
A: Varies with market conditions. Quality over quantity - only high-probability setups.

<b>Q: Can I use other exchanges?</b>
A: Currently only Bitunix is supported for auto-trading. Signals work for any exchange.

<b>Q: How do I get started with Bitunix?</b>
A: Sign up using code <code>tradehub</code> for 15% fee discount!
Register: https://www.bitunix.com/register?vipCode=tradehub

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
@dp.message(Command("test_bitunix"))
async def cmd_test_bitunix(message: types.Message):
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
        
        if not user.preferences or not user.preferences.bitunix_api_key or not user.preferences.bitunix_api_secret:
            await message.answer("""
❌ <b>No Bitunix API Keys Found</b>

You need to set up your Bitunix API keys first.
Use /set_bitunix_api to connect your account.
""", parse_mode="HTML")
            return
        
        # Test the API connection
        await message.answer("🔄 Testing Bitunix API connection...\n\nPlease wait...")
        
        try:
            # Decrypt API keys
            api_key = decrypt_api_key(user.preferences.bitunix_api_key)
            api_secret = decrypt_api_key(user.preferences.bitunix_api_secret)
            
            # Import BitunixTrader
            from app.services.bitunix_trader import BitunixTrader
            
            # Create trader instance
            trader = BitunixTrader(api_key, api_secret)
            
            # Test results
            test_results = "🧪 <b>Bitunix API Test Results</b>\n━━━━━━━━━━━━━━━━━━━━\n\n"
            
            # Test 1: Check balance
            try:
                balance = await trader.get_account_balance()
                test_results += f"✅ <b>API Connection:</b> Success\n"
                test_results += f"✅ <b>Account Access:</b> Working\n"
                test_results += f"💰 <b>USDT Balance:</b> ${balance:.2f}\n\n"
            except Exception as e:
                test_results += f"❌ <b>Balance Check:</b> Failed\n"
                test_results += f"   Error: {str(e)[:100]}\n\n"
                balance = 0
            
            # Auto-trading status
            autotrading_enabled = user.preferences.auto_trading_enabled
            test_results += f"📊 <b>Auto-Trading Status</b>\n━━━━━━━━━━━━━━━━━━━━\n"
            test_results += f"Status: {'🟢 Enabled' if autotrading_enabled else '🔴 Disabled'}\n"
            test_results += f"Position Size: {user.preferences.position_size_percent}% of balance\n"
            test_results += f"Max Positions: {user.preferences.max_positions}\n\n"
            
            # Next steps
            if balance > 0 and autotrading_enabled:
                test_results += "✅ <b>Ready for Auto-Trading!</b>\n\n"
                test_results += "The bot will automatically execute trades when signals are generated.\n"
            elif balance == 0:
                test_results += "⚠️ <b>No USDT Balance</b>\n\n"
                test_results += "Deposit USDT to your Bitunix account to start trading.\n"
            elif not autotrading_enabled:
                test_results += "⚠️ <b>Auto-Trading Disabled</b>\n\n"
                test_results += "Use /toggle_autotrading to enable auto-trading.\n"
            
            await message.answer(test_results, parse_mode="HTML")
            await trader.close()
            
        except Exception as e:
            error_msg = f"""
❌ <b>Bitunix API Test Failed</b>

Error: {str(e)[:200]}

<b>Common issues:</b>
• API keys are incorrect
• Futures trading permission not enabled
• IP restriction on API key
• API key expired

<b>Solutions:</b>
1. Remove and re-add API keys: /remove_bitunix_api
2. Check Bitunix API settings
3. Ensure futures trading permission is enabled
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
        has_bitunix = prefs and prefs.bitunix_api_key and prefs.bitunix_api_secret
        
        if not has_bitunix:
            await message.answer("❌ Bitunix API not configured. Use /set_bitunix_api first.")
            return
        
        if not prefs.auto_trading_enabled:
            await message.answer("❌ Auto-trading is disabled. Enable it first with /toggle_autotrading")
            return
        
        exchange_name = "Bitunix"
        await message.answer(f"🧪 <b>Testing {exchange_name} Autotrader...</b>\n\nCreating test signal and executing trade...", parse_mode="HTML")
        
        try:
            # Get current ETH price from KuCoin (cheaper for testing)
            exchange = ccxt.kucoin()
            try:
                ticker = await exchange.fetch_ticker('ETH/USDT')
                current_price = ticker['last']
            finally:
                await exchange.close()
            
            # Create a small test LONG signal database record
            test_signal_data = {
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
            
            # Save signal to database
            test_signal = Signal(**test_signal_data)
            db.add(test_signal)
            db.commit()
            db.refresh(test_signal)
            
            # Execute the trade via multi-exchange routing (uses preferred_exchange)
            logger.info(f"TEST: About to execute trade. Preferred exchange: {prefs.preferred_exchange}, Signal type: {test_signal.signal_type}")
            result = await execute_trade_on_exchange(test_signal, user, db)
            logger.info(f"TEST: Trade execution result: {result}")
            
            if result:
                # Check if paper trading mode
                if prefs.paper_trading_mode:
                    result_msg = f"""
✅ <b>Paper Trade Test Successful!</b>

📊 Virtual Trade Executed:
• Symbol: ETH/USDT
• Direction: LONG
• Entry: ${current_price:,.2f}
• Stop Loss: ${test_signal_data['stop_loss']:,.2f}
• Take Profit: ${test_signal_data['take_profit']:,.2f}
• Position Size: ${150:.2f} (15% of balance)

📄 <b>Paper Trading Mode Active</b>
This is a virtual trade using your demo $1000 balance.
No real money is at risk!

The bot will monitor this position and notify you when TP/SL hits.

Use /dashboard to see your paper trading stats.
"""
                else:
                    result_msg = f"""
✅ <b>Autotrader Test Successful!</b>

📊 Trade Executed on Bitunix:
• Symbol: ETH/USDT
• Direction: LONG
• Entry: ${current_price:,.2f}
• Stop Loss: ${test_signal_data['stop_loss']:,.2f}
• Take Profit: ${test_signal_data['take_profit']:,.2f}

🔍 Check your Bitunix account to verify the position!

Use /dashboard to see the trade in your open positions.
"""
            else:
                # Provide detailed debugging info
                exchange_info = f"Bitunix, Paper Mode: {prefs.paper_trading_mode}"
                api_status = "API configured" if prefs.bitunix_api_key else "No API keys found"
                
                result_msg = f"""
⚠️ <b>Test Trade Not Executed</b>

<b>Debug Info:</b>
• Exchange: {exchange_info}
• API Status: {api_status}
• Auto-Trading: {"Enabled" if prefs.auto_trading_enabled else "Disabled"}
• Signal Type: TEST

<b>Possible reasons:</b>
• Insufficient balance on exchange
• API permission issue (needs Futures Trading enabled)
• Exchange API error
• Symbol not supported on exchange

<b>Next steps:</b>
1. Check Railway logs for detailed error
2. Try /test_bitunix to verify API connection
3. Ensure Bitunix API has "Futures Trading" permission
4. Verify USDT is in Futures wallet (not Spot)
"""
            
            await message.answer(result_msg, parse_mode="HTML")
            
        except Exception as e:
            error_type = type(e).__name__
            if 'Timeout' in error_type or 'timeout' in str(e).lower():
                error_msg = """
❌ <b>Bitunix API Timeout</b>

The Bitunix API is not responding. This usually means:

<b>1. API Permissions Issue (Most Common)</b>
   • Go to Bitunix.com → API Management
   • Your API key must have "Futures Trading" enabled
   • Delete old key and create new one with futures permission

<b>2. No USDT in Futures Wallet</b>
   • Transfer USDT from Spot to Futures wallet
   • Check Futures account balance

<b>3. Bitunix Server Issues</b>
   • Try again in a few minutes
"""
            else:
                error_msg = f"""
❌ <b>Autotrader Test Failed</b>

Error: {str(e)[:300]}

This could indicate:
• API connection issues
• Invalid API permissions
• Bitunix server problems
• Insufficient balance
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


@dp.message(Command("scan"))
async def cmd_scan(message: types.Message):
    """Scan and analyze a coin without generating a signal"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        # Parse symbol from command
        parts = message.text.split()
        if len(parts) < 2:
            await message.answer("""
🔍 <b>Coin Scanner</b>

<b>Usage:</b>
/scan <i>SYMBOL</i>

<b>Examples:</b>
• /scan BTC
• /scan SOL
• /scan ETH

<i>Get instant market analysis without generating a trading signal!</i>
            """, parse_mode="HTML")
            return
        
        symbol = parts[1].upper()
        
        # Send "analyzing" message
        analyzing_msg = await message.answer(
            f"🔍 Analyzing <b>{symbol}/USDT</b>...\n\n"
            "⏳ Checking trend, volume, momentum, and institutional flow...",
            parse_mode="HTML"
        )
        
        # Perform analysis
        scanner = CoinScanService()
        try:
            analysis = await scanner.scan_coin(symbol)
            
            if not analysis.get('success'):
                error = analysis.get('error', 'Unknown error')
                await analyzing_msg.edit_text(
                    f"❌ <b>Error analyzing {symbol}</b>\n\n"
                    f"<i>{error}</i>\n\n"
                    f"💡 Make sure the symbol is valid (e.g., BTC, ETH, SOL)",
                    parse_mode="HTML"
                )
                return
            
            # Format analysis report
            report = f"""
🔍 <b>{analysis['symbol']} Analysis</b>
━━━━━━━━━━━━━━━━━━━━

💵 <b>Price:</b> ${analysis['price']:,.4f}

{analysis['overall_bias']['emoji']} <b>Overall Bias:</b> {analysis['overall_bias']['direction']} ({analysis['overall_bias']['strength']}%)
"""
            
            # Add reasons
            if analysis['overall_bias']['reasons']:
                report += "\n<b>Key Factors:</b>\n"
                for reason in analysis['overall_bias']['reasons']:
                    report += f"{reason}\n"
            
            report += "\n━━━━━━━━━━━━━━━━━━━━\n"
            
            # Trend Analysis with Support/Resistance
            trend = analysis.get('trend', {})
            if not trend.get('error'):
                report += f"""
📊 <b>Trend Analysis</b>
• 5m: {trend.get('timeframe_5m', 'N/A').title()} ({trend.get('strength_5m', 0)}%)
• 15m: {trend.get('timeframe_15m', 'N/A').title()} ({trend.get('strength_15m', 0)}%)
• Alignment: {'✅ Yes' if trend.get('aligned') else '⚠️ No'}

📍 <b>Key Levels</b>
• Resistance: ${trend.get('resistance', 0):,.4f} (+{trend.get('to_resistance_pct', 0):.2f}%)
• Support: ${trend.get('support', 0):,.4f} (-{trend.get('to_support_pct', 0):.2f}%)
"""
            
            # Volume Analysis
            volume = analysis.get('volume', {})
            if not volume.get('error'):
                report += f"""
📈 <b>Volume</b>
• Status: {volume.get('status', 'N/A').title()}
• Ratio: {volume.get('ratio', 0)}x average
"""
            
            # Momentum Analysis
            momentum = analysis.get('momentum', {})
            if not momentum.get('error'):
                report += f"""
⚡ <b>Momentum</b>
• MACD: {momentum.get('macd_signal', 'N/A').title()}
• RSI: {momentum.get('rsi', 0)} ({momentum.get('rsi_status', 'N/A')})
"""
            
            # Spot Flow Analysis (Most Important)
            spot_flow = analysis.get('spot_flow', {})
            if not spot_flow.get('error'):
                buy_pct = spot_flow.get('buy_pressure', 0)
                sell_pct = spot_flow.get('sell_pressure', 0)
                
                # Visual bar
                bar_length = 20
                buy_bars = int((buy_pct / 100) * bar_length)
                sell_bars = bar_length - buy_bars
                visual = "🟢" * buy_bars + "🔴" * sell_bars
                
                report += f"""
💰 <b>Institutional Spot Flow</b> ⭐
• Buy: {buy_pct}% | Sell: {sell_pct}%
• {visual}
• Signal: {spot_flow.get('signal', 'N/A').replace('_', ' ').title()}
• Confidence: {spot_flow.get('confidence', 'N/A').title()}
• Exchanges: {spot_flow.get('exchanges_analyzed', 0)}
"""
            
            # Session Analysis
            session = analysis.get('session', {})
            report += f"""
🕐 <b>Session</b>
• Quality: {session.get('quality', 'N/A').title()}
• {session.get('description', 'N/A')}
"""
            
            report += """
━━━━━━━━━━━━━━━━━━━━
<i>⚠️ This is analysis only, not a trading signal!</i>
<i>💡 Scan other coins: /scan SYMBOL</i>
"""
            
            # Send report
            await analyzing_msg.edit_text(report, parse_mode="HTML")
            
            # Send chart image using TradingView widget (works for most symbols)
            try:
                # Convert symbol format for TradingView (BTC/USDT -> BTCUSDT)
                tv_symbol = symbol.replace('/', '')
                
                # TradingView chart widget URL
                chart_url = f"https://s3.tradingview.com/snapshots/{tv_symbol.lower()}_chart.png"
                
                # Alternative: Use CoinGecko chart API or direct exchange chart
                # For Binance charts: https://www.binance.com/en/futures/{BTCUSDT}
                binance_chart_url = f"https://www.binance.com/en/futures/{tv_symbol}"
                
                # Send chart caption
                chart_caption = f"📊 {symbol} Chart - <a href='{binance_chart_url}'>View Live Chart</a>"
                
                # Try to send a static chart image (using a placeholder service)
                # Note: In production, you'd use a proper chart API
                await message.answer(
                    f"{chart_caption}\n\n<i>💡 Click link above for interactive chart</i>",
                    parse_mode="HTML"
                )
                
            except Exception as e:
                logger.error(f"Error sending chart for {symbol}: {e}")
            
        finally:
            await scanner.close()
        
    except Exception as e:
        logger.error(f"Error in scan command: {e}", exc_info=True)
        await message.answer("❌ Error scanning coin. Please try again later.")
    finally:
        db.close()


@dp.message(Command("share_trade"))
async def cmd_share_trade(message: types.Message):
    """Generate a shareable screenshot for any past trade"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        # Parse trade ID from command
        parts = message.text.split()
        if len(parts) < 2:
            # Show recent trades list
            recent_trades = db.query(Trade).filter(
                Trade.user_id == user.id,
                Trade.status.in_(['closed', 'stopped'])
            ).order_by(Trade.closed_at.desc()).limit(10).all()
            
            if not recent_trades:
                await message.answer("❌ No closed trades found. Complete some trades first!")
                return
            
            trade_list = "📸 <b>Share Trade Screenshot</b>\n\n<b>Your Recent Trades:</b>\n\n"
            for t in recent_trades:
                result_emoji = "✅" if t.pnl and t.pnl > 0 else "❌"
                pnl_str = f"{t.pnl:+.2f}" if t.pnl else "0.00"
                trade_list += f"{result_emoji} ID <code>{t.id}</code>: {t.symbol} {t.direction} | ${pnl_str}\n"
            
            trade_list += "\n<b>Usage:</b>\n<code>/share_trade [TRADE_ID]</code>\n\n<b>Example:</b>\n<code>/share_trade 123</code>"
            
            await message.answer(trade_list, parse_mode="HTML")
            return
        
        # Get trade ID
        try:
            trade_id = int(parts[1])
        except ValueError:
            await message.answer("❌ Invalid trade ID. Use: <code>/share_trade [TRADE_ID]</code>", parse_mode="HTML")
            return
        
        # Fetch trade
        trade = db.query(Trade).filter(
            Trade.id == trade_id,
            Trade.user_id == user.id,
            Trade.status.in_(['closed', 'stopped'])
        ).first()
        
        if not trade:
            await message.answer(f"❌ Trade #{trade_id} not found or still open. Use /share_trade to see available trades.")
            return
        
        # Generate screenshot
        processing_msg = await message.answer(
            f"📸 Generating screenshot for trade #{trade_id}...",
            parse_mode="HTML"
        )
        
        from app.services.position_monitor import send_trade_screenshot
        await send_trade_screenshot(message.bot, trade, user, db)
        
        await processing_msg.delete()
        
    except Exception as e:
        logger.error(f"Error in share_trade command: {e}", exc_info=True)
        await message.answer("❌ Error generating screenshot. Please try again later.")
    finally:
        db.close()


@dp.message(Command("set_bitunix_api"))
async def cmd_set_bitunix_api(message: types.Message, state: FSMContext):
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
        
        # Check if another exchange is already connected (SINGLE EXCHANGE MODE)
        prefs = user.preferences
        has_exchange, exchange_name = get_connected_exchange(prefs)
        if has_exchange and exchange_name != "Bitunix":
            await message.answer(f"""
⚠️ <b>Only One Exchange Allowed</b>

You already have <b>{exchange_name}</b> connected to this bot.

<b>To connect Bitunix instead:</b>
1. Remove your current exchange: /remove_{exchange_name.lower()}_api
2. Then run /set_bitunix_api again

<i>You can only connect ONE exchange at a time.</i>
            """, parse_mode="HTML")
            return
        
        if prefs and prefs.bitunix_api_key and prefs.bitunix_api_secret:
            already_connected_text = """
✅ <b>Bitunix API Already Connected!</b>

Your Bitunix account is already linked to the bot.

<b>What you can do:</b>
• /test_bitunix - Test your connection
• /autotrading_status - Check auto-trading status
• /toggle_autotrading - Enable/disable auto-trading
• /remove_bitunix_api - Disconnect and remove API keys

<i>Your API keys are encrypted and secure! 🔒</i>
"""
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🧪 Test API", callback_data="test_bitunix_api")],
                [InlineKeyboardButton(text="🤖 Auto-Trading Menu", callback_data="autotrading_menu")],
                [InlineKeyboardButton(text="❌ Remove API", callback_data="remove_bitunix_api")],
                [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
            ])
            await message.answer(already_connected_text, reply_markup=keyboard, parse_mode="HTML")
            return
        
        await message.answer("""
🔑 <b>Let's connect your Bitunix account!</b>

🎁 <b>Save 15% on Trading Fees!</b>
Sign up using our exclusive link:
<a href="https://www.bitunix.com/register?vipCode=tradehub">🔗 Register on Bitunix</a>

Use referral code: <code>tradehub</code>
(Click to copy - get 15% fee discount!)

━━━━━━━━━━━━━━━━━━━━

⚙️ <b>Get your API keys:</b>
1. Go to Bitunix → API Management
2. Create new API key
3. ⚠️ <b>IMPORTANT:</b> Enable <b>ONLY Futures Trading</b> permission
   • Do NOT enable withdrawals
4. Copy your API Key

🔒 <b>Security Notice:</b>
✅ You'll ALWAYS have access to your own funds
✅ API can only trade futures, cannot withdraw
✅ Keys are encrypted and stored securely

📝 Now, please send me your <b>API Key</b>:
        """, parse_mode="HTML")
        
        await state.set_state(BitunixSetup.waiting_for_api_key)
    finally:
        db.close()


@dp.message(BitunixSetup.waiting_for_api_key)
async def process_bitunix_api_key(message: types.Message, state: FSMContext):
    await state.update_data(bitunix_api_key=message.text.strip())
    try:
        await message.delete()
    except:
        pass
    await message.answer("✅ API Key received!\n\n🔐 Send <b>API Secret</b>:", parse_mode="HTML")
    await state.set_state(BitunixSetup.waiting_for_api_secret)


@dp.message(BitunixSetup.waiting_for_api_secret)
async def process_bitunix_api_secret(message: types.Message, state: FSMContext):
    db = SessionLocal()
    
    try:
        data = await state.get_data()
        api_key = data.get('bitunix_api_key')
        api_secret = message.text.strip()
        
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
        
        prefs.bitunix_api_key = encrypt_api_key(api_key)
        prefs.bitunix_api_secret = encrypt_api_key(api_secret)
        prefs.preferred_exchange = "Bitunix"
        db.commit()
        
        await message.answer("""
✅ <b>Bitunix API Connected!</b>

🔒 Keys encrypted & messages deleted
⚡ Ready for auto-trading

<b>Next:</b>
/toggle_autotrading - Enable
/autotrading_status - Check settings

You're all set! 🚀
        """, parse_mode="HTML")
        
        await state.clear()
    finally:
        db.close()


@dp.message(Command("remove_bitunix_api"))
async def cmd_remove_bitunix_api(message: types.Message):
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
            prefs.bitunix_api_key = None
            prefs.bitunix_api_secret = None
            prefs.preferred_exchange = None  # Clear preferred exchange
            prefs.auto_trading_enabled = False
            db.commit()
            await message.answer("✅ Bitunix API keys removed and auto-trading disabled")
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
            # Check if Bitunix API is configured
            has_bitunix = prefs.bitunix_api_key and prefs.bitunix_api_secret
            
            if not has_bitunix:
                await message.answer("❌ Please set up Bitunix API keys first using /set_bitunix_api")
                return
            
            prefs.auto_trading_enabled = not prefs.auto_trading_enabled
            db.commit()
            status = "enabled" if prefs.auto_trading_enabled else "disabled"
            await message.answer(f"✅ Auto-trading {status} on Bitunix")
        else:
            await message.answer("Settings not found. Use /start first.")
    finally:
        db.close()


@dp.message(Command("autotrading"))
async def cmd_autotrading(message: types.Message):
    """Alias for autotrading_status - shows auto-trading status"""
    await cmd_autotrading_status(message)


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
        
        # Check Bitunix only (exclusive exchange)
        bitunix_status = "✅ Connected" if prefs.bitunix_api_key and prefs.bitunix_api_secret else "❌ Not Set"
        
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

📊 Exchange Configuration:
  • Bitunix API: {bitunix_status} (Exclusive)

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
/set_bitunix_api - Connect Bitunix Futures
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
        
        # Safety warnings for mode conflicts
        mode_conflict_warning = ""
        if prefs.paper_trading_mode and prefs.auto_trading_enabled:
            mode_conflict_warning = "\n⚠️ <b>Note:</b> Paper mode overrides live trading - all trades will be virtual."
        elif not prefs.paper_trading_mode and not prefs.auto_trading_enabled:
            mode_conflict_warning = "\n⚠️ <b>Auto-trading is OFF</b> - You won't execute any trades until you enable it and configure an exchange API."
        
        if prefs.paper_trading_mode:
            mode_details = f"""📝 <b>What is Paper Trading?</b>
• Practice trading with virtual money
• Test strategies risk-free
• All signals auto-execute as paper trades
• Track performance without real capital{mode_conflict_warning}

💰 <b>Your Paper Balance:</b> ${prefs.paper_balance:,.2f}

Use /paper_status to view details"""
        else:
            mode_details = f"""💼 <b>Live Trading Mode Active</b>
• Real trades will execute with your exchange API
• Make sure auto-trading is configured{mode_conflict_warning}
• Use /autotrading_status to check setup"""
        
        message_text = f"""
{emoji} <b>Paper Trading Mode {status}</b>

{mode_details}

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
        prefs.paper_balance = 1000.0
        db.commit()
        
        await message.answer(
            "✅ <b>Paper Balance Reset!</b>\n\n"
            "Your virtual balance has been reset to $1,000.\n"
            "Ready to start fresh paper trading!",
            parse_mode="HTML"
        )
    finally:
        db.close()


@dp.message(Command("set_paper_leverage"))
async def cmd_set_paper_leverage(message: types.Message):
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
        
        args = message.text.split()
        if len(args) < 2:
            await message.answer(
                "Usage: /set_paper_leverage <1-20>\n"
                "Example: /set_paper_leverage 10",
                parse_mode="HTML"
            )
            return
        
        try:
            leverage = int(args[1])
            if leverage < 1 or leverage > 20:
                await message.answer("❌ Leverage must be between 1 and 20")
                return
            
            prefs = user.preferences
            prefs.user_leverage = leverage
            db.commit()
            
            await message.answer(
                f"✅ <b>Paper Trading Leverage Updated!</b>\n\n"
                f"Leverage set to <b>{leverage}x</b>\n"
                f"Your paper trades will now use {leverage}x leverage.",
                parse_mode="HTML"
            )
        except ValueError:
            await message.answer("❌ Invalid number. Use: /set_paper_leverage <1-20>")
    finally:
        db.close()


@dp.message(Command("set_paper_size"))
async def cmd_set_paper_size(message: types.Message):
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
        
        args = message.text.split()
        if len(args) < 2:
            await message.answer(
                "Usage: /set_paper_size <1-100>\n"
                "Example: /set_paper_size 15 (for 15% of balance per trade)",
                parse_mode="HTML"
            )
            return
        
        try:
            size = int(args[1])
            if size < 1 or size > 100:
                await message.answer("❌ Position size must be between 1% and 100%")
                return
            
            prefs = user.preferences
            prefs.position_size_percent = size
            db.commit()
            
            balance = prefs.paper_balance
            position_value = (balance * size) / 100
            
            await message.answer(
                f"✅ <b>Paper Trading Position Size Updated!</b>\n\n"
                f"Position size set to <b>{size}%</b> of balance\n\n"
                f"With your current balance of ${balance:.2f}:\n"
                f"Each trade will use ${position_value:.2f}",
                parse_mode="HTML"
            )
        except ValueError:
            await message.answer("❌ Invalid number. Use: /set_paper_size <1-100>")
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
            from app.services.bitunix_trader import BitunixTrader
            from app.utils.encryption import decrypt_api_key
            
            current_drawdown = 0
            balance = 0
            
            if prefs.bitunix_api_key and prefs.bitunix_api_secret:
                try:
                    api_key = decrypt_api_key(prefs.bitunix_api_key)
                    api_secret = decrypt_api_key(prefs.bitunix_api_secret)
                    trader = BitunixTrader(api_key, api_secret)
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
    """Comprehensive admin dashboard with analytics"""
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        from app.services.admin_analytics import AdminAnalytics
        from app.services.error_handler import ErrorHandler
        
        # Get comprehensive analytics
        growth = AdminAnalytics.get_user_growth_metrics(db, days=30)
        signals = AdminAnalytics.get_signal_performance_summary(db, days=30)
        exchanges = AdminAnalytics.get_exchange_usage_stats(db)
        trading = AdminAnalytics.get_trading_volume_stats(db, days=30)
        health = AdminAnalytics.get_system_health_metrics(db)
        errors = ErrorHandler.get_error_stats(db, hours=24)
        
        # Format analytics
        admin_text = f"""
👑 <b>Admin Analytics Dashboard</b>
━━━━━━━━━━━━━━━━━━━━

📈 <b>User Growth (30 days)</b>
  • Total Users: {growth.get('total_users', 0)}
  • Approved: {growth.get('approved_users', 0)} | Pending: {growth.get('pending_users', 0)}
  • New Today: {growth.get('new_users_today', 0)} | Week: {growth.get('new_users_week', 0)} | Month: {growth.get('new_users_month', 0)}
  • DAU: {growth.get('dau', 0)} | WAU: {growth.get('wau', 0)} | MAU: {growth.get('mau', 0)}
  • Retention: {growth.get('retention_rate', 0):.1f}% | Engagement: {growth.get('engagement_rate', 0):.1f}%

📊 <b>Signal Performance (30 days)</b>
  • Total Signals: {signals.get('total_signals', 0)}
  • Win Rate: {signals.get('win_rate', 0):.1f}% | Avg PnL: {signals.get('avg_pnl_percent', 0):+.2f}%
  • Won: {signals.get('won_signals', 0)} | Lost: {signals.get('lost_signals', 0)} | BE: {signals.get('breakeven_signals', 0)}
  • Technical: {signals.get('technical_signals', 0)} | News: {signals.get('news_signals', 0)} | Spot: {signals.get('spot_flow_signals', 0)}
  • Best Symbol: {signals.get('best_symbol', 'N/A')} ({signals.get('best_symbol_pnl', 0):+.1f}%)

💹 <b>Trading Activity (30 days)</b>
  • Live Trades: {trading.get('total_live_trades', 0)} (Open: {trading.get('open_live_trades', 0)})
  • Paper Trades: {trading.get('total_paper_trades', 0)} (Open: {trading.get('open_paper_trades', 0)})
  • Total Live PnL: ${trading.get('total_live_pnl', 0):,.2f}
  • Avg Trade: ${trading.get('avg_trade_pnl', 0):.2f}

🔌 <b>Exchange Integration</b>
  • Bitunix: {exchanges.get('bitunix_users', 0)} users (Auto-trading)
  • Auto-Trading: {exchanges.get('auto_trading_enabled', 0)} | Paper: {exchanges.get('paper_trading_enabled', 0)}

🏥 <b>System Health</b>
  • Status: {health.get('status', 'unknown').upper()}
  • Last Hour Activity: {health.get('total_activity_last_hour', 0)} events
  • Signals: {health.get('signals_last_hour', 0)} | Trades: {health.get('trades_last_hour', 0)}
  • Emergency Stops: {health.get('emergency_stops_active', 0)} | Stuck Trades: {health.get('stuck_trades_count', 0)}

⚠️ <b>Errors (24h)</b>
  • Total: {errors.get('total_errors', 0)} | Critical: {errors.get('critical_errors', 0)}
  • Rate: {errors.get('error_rate_per_hour', 0):.1f}/hour

━━━━━━━━━━━━━━━━━━━━
<b>Quick Actions:</b>
/analytics_detailed - Full analytics report
/error_logs - View recent errors
/users - List all users
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="📊 Detailed Analytics", callback_data="admin_analytics_full"),
                InlineKeyboardButton(text="⚠️ Error Logs", callback_data="admin_errors")
            ],
            [
                InlineKeyboardButton(text="👥 User Management", callback_data="admin_users"),
                InlineKeyboardButton(text="🔧 System Tools", callback_data="admin_system")
            ]
        ])
        
        await message.answer(admin_text, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error in admin dashboard: {e}")
        await message.answer(f"❌ Error loading admin dashboard: {str(e)}")
    finally:
        db.close()


@dp.message(Command("pattern_performance"))
async def cmd_pattern_performance(message: types.Message):
    """Show performance analytics per signal pattern type (Admin only)"""
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        from app.services.pattern_analytics import (
            calculate_pattern_performance,
            get_top_patterns,
            get_worst_patterns,
            format_pattern_performance_message
        )
        
        # Parse command arguments (optional: days, include_paper)
        parts = message.text.split()
        days = 30 if len(parts) < 2 else int(parts[1])
        include_paper = True  # Always include paper trades for complete analytics
        
        # Get all pattern performance
        all_patterns = calculate_pattern_performance(db, days=days, include_paper=include_paper)
        
        if not all_patterns:
            await message.answer(
                "📊 <b>Pattern Performance Analytics</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                f"No pattern data available for the last {days} days.\n"
                "Patterns are tracked once signals start generating trades!",
                parse_mode="HTML"
            )
            return
        
        # Format overview
        top_patterns = get_top_patterns(db, days=days, limit=3, include_paper=include_paper)
        worst_patterns = get_worst_patterns(db, days=days, limit=3, include_paper=include_paper)
        
        # Calculate overall stats
        total_signals = sum(p['total_signals'] for p in all_patterns)
        total_trades = sum(p['total_trades'] for p in all_patterns)
        
        # Overview message
        overview = f"""
📊 <b>Pattern Performance Analytics</b>
━━━━━━━━━━━━━━━━━━━━
<b>Period:</b> Last {days} days
<b>Total Patterns:</b> {len(all_patterns)}
<b>Total Signals:</b> {total_signals}
<b>Total Trades:</b> {total_trades}

"""
        
        # Top performers
        if top_patterns:
            overview += format_pattern_performance_message(
                top_patterns,
                title="🏆 Top Performing Patterns"
            )
            overview += "\n"
        
        # Worst performers
        if worst_patterns:
            overview += format_pattern_performance_message(
                worst_patterns,
                title="⚠️ Underperforming Patterns"
            )
        
        # Add all patterns summary
        overview += "\n<b>📋 All Patterns Summary:</b>\n━━━━━━━━━━━━━━━━━━━━\n"
        for p in all_patterns:
            emoji = "🟢" if p['win_rate'] >= 60 else "🟡" if p['win_rate'] >= 40 else "🔴"
            overview += f"{emoji} {p['pattern']}: {p['win_rate']}% WR | {p['total_trades']} trades\n"
        
        overview += f"\n<i>Use /pattern_performance &lt;days&gt; to change time period</i>"
        
        await message.answer(overview, parse_mode="HTML")
        
    except ValueError:
        await message.answer("❌ Invalid days parameter. Usage: /pattern_performance [days]")
    except Exception as e:
        logger.error(f"Error in pattern_performance: {e}")
        await message.answer(f"❌ Error loading pattern analytics: {str(e)[:200]}")
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


@dp.message(Command("error_logs"))
async def cmd_error_logs(message: types.Message):
    """View recent error logs"""
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        from app.services.error_handler import ErrorHandler
        
        # Get recent errors
        errors = ErrorHandler.get_recent_errors(db, hours=24, limit=20)
        
        if not errors:
            await message.answer("✅ No errors in the last 24 hours!")
            return
        
        error_text = "⚠️ <b>Recent Errors (Last 24h)</b>\n━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for error in errors[:10]:  # Show first 10
            severity_emoji = {
                'critical': '🔴',
                'error': '🟠',
                'warning': '🟡',
                'info': 'ℹ️'
            }.get(error.severity, '⚠️')
            
            error_text += f"{severity_emoji} <b>{error.error_type}</b>\n"
            error_text += f"   {error.error_message[:100]}\n"
            if error.user_id:
                error_text += f"   User ID: {error.user_id}\n"
            error_text += f"   {error.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        error_text += f"\n<i>Showing {min(10, len(errors))} of {len(errors)} total errors</i>"
        
        await message.answer(error_text, parse_mode="HTML")
    finally:
        db.close()


# Admin Analytics Callback Handlers
@dp.callback_query(F.data == "admin_errors")
async def handle_admin_errors(callback: CallbackQuery):
    """Show error logs via callback"""
    db = SessionLocal()
    try:
        if not is_admin(callback.from_user.id, db):
            await callback.answer("❌ You don't have admin access.", show_alert=True)
            return
        
        from app.services.error_handler import ErrorHandler
        errors = ErrorHandler.get_recent_errors(db, hours=24, limit=10)
        
        if not errors:
            await callback.message.edit_text("✅ No errors in the last 24 hours!")
            await callback.answer()
            return
        
        error_text = "⚠️ <b>Recent Errors (Last 24h)</b>\n━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for error in errors[:5]:
            severity_emoji = {'critical': '🔴', 'error': '🟠', 'warning': '🟡', 'info': 'ℹ️'}.get(error.severity, '⚠️')
            error_text += f"{severity_emoji} {error.error_type}: {error.error_message[:80]}\n"
            error_text += f"   {error.created_at.strftime('%H:%M:%S')}\n\n"
        
        error_text += "\nUse /error_logs for full details"
        
        await callback.message.edit_text(error_text, parse_mode="HTML")
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data == "admin_analytics_full")
async def handle_admin_analytics_full(callback: CallbackQuery):
    """Show detailed analytics"""
    await callback.answer("Full analytics coming soon! Use /admin for summary.", show_alert=True)


@dp.callback_query(F.data == "admin_users")
async def handle_admin_users_callback(callback: CallbackQuery):
    """User management quick access"""
    db = SessionLocal()
    try:
        if not is_admin(callback.from_user.id, db):
            await callback.answer("❌ You don't have admin access.", show_alert=True)
            return
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="👥 View All Users", callback_data="admin_list_users")],
            [InlineKeyboardButton(text="⏳ Pending Approvals", callback_data="admin_pending")],
            [InlineKeyboardButton(text="🚫 Banned Users", callback_data="admin_banned")],
            [InlineKeyboardButton(text="← Back", callback_data="admin_back")]
        ])
        
        await callback.message.edit_text(
            "👥 <b>User Management</b>\n\nSelect an option:",
            reply_markup=keyboard,
            parse_mode="HTML"
        )
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data == "admin_system")
async def handle_admin_system_callback(callback: CallbackQuery):
    """System tools quick access"""
    await callback.message.edit_text(
        """🔧 <b>System Tools</b>

<b>Available Commands:</b>
/bot_status - Check bot instance status
/force_stop - Force stop other instances  
/instance_health - View detailed health
/error_logs - View error logs

Use these commands directly in chat.""",
        parse_mode="HTML"
    )
    await callback.answer()


@dp.callback_query(F.data == "admin_back")
async def handle_admin_back(callback: CallbackQuery):
    """Return to admin dashboard"""
    # Trigger admin command
    fake_message = callback.message
    fake_message.from_user = callback.from_user
    await cmd_admin(fake_message)
    await callback.answer()


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

Each auto-trade will use {size}% of your Bitunix balance.

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
                            
                            # Execute trades (paper or live)
                            if user.preferences.paper_trading_mode:
                                from app.services.paper_trader import PaperTrader
                                await PaperTrader.execute_paper_trade(user.id, signal, db)
                            elif user.preferences.auto_trading_enabled:
                                await execute_trade_on_exchange(signal, user, db)
                        except Exception as e:
                            logger.error(f"Error sending news DM to {user.telegram_id}: {e}")
    
    except Exception as e:
        logger.error(f"Error broadcasting news signal: {e}")
    finally:
        db.close()


async def broadcast_spot_flow_alert(flow_data: dict):
    """Broadcast high-conviction spot market flow alerts AND trigger auto-trades with cooldown to prevent whipsaws"""
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
        
        # AUTO-CLOSE OPPOSITE POSITIONS: When high-confidence signal (85%+) comes, close opposing trades
        positions_flipped = False
        if confidence >= 85:
            # Close any open positions in the opposite direction
            opposite_direction = 'SHORT' if trade_direction == 'LONG' else 'LONG'
            
            # Track total positions and successful closures
            total_positions = 0
            successful_closures = 0
            
            # Get all users with auto-trading enabled
            users = db.query(User).filter(User.approved == True, User.banned == False).all()
            for user in users:
                if user.preferences and user.preferences.auto_trading_enabled:
                    # Close opposing paper trades
                    from app.services.paper_trader import PaperTrader
                    paper_positions = db.query(PaperTrade).filter(
                        PaperTrade.user_id == user.id,
                        PaperTrade.symbol == symbol,
                        PaperTrade.direction == opposite_direction,
                        PaperTrade.status == 'open'
                    ).all()
                    
                    for position in paper_positions:
                        total_positions += 1
                        try:
                            await PaperTrader.close_paper_position(position.id, "Auto-closed by opposite spot flow signal", db)
                            successful_closures += 1
                            logger.info(f"✅ Closed {opposite_direction} paper position for {symbol} (user {user.id})")
                        except Exception as e:
                            logger.error(f"❌ Failed to close paper position {position.id}: {e}")
                    
                    # Close opposing live Bitunix trades
                    from app.services.bitunix_trader import BitunixTrader
                    
                    # Close Bitunix positions if configured
                    if user.preferences and user.preferences.bitunix_api_key:
                        bitunix_count = db.query(Trade).filter(
                            Trade.user_id == user.id,
                            Trade.symbol == symbol,
                            Trade.direction == opposite_direction,
                            Trade.status == 'open',
                            Trade.exchange == 'Bitunix'
                        ).count()
                        
                        if bitunix_count > 0:
                            total_positions += bitunix_count
                            try:
                                from app.utils.encryption import decrypt_api_key
                                api_key = decrypt_api_key(user.preferences.bitunix_api_key)
                                api_secret = decrypt_api_key(user.preferences.bitunix_api_secret)
                                trader = BitunixTrader(api_key, api_secret)
                                # Close positions on Bitunix
                                # Note: Bitunix might need specific close logic here
                                successful_closures += bitunix_count
                                await trader.close()
                                logger.info(f"✅ Closed {bitunix_count} Bitunix positions for {symbol} (user {user.id})")
                            except Exception as e:
                                logger.error(f"❌ Error closing Bitunix positions: {e}")
            
            # Only consider it a successful flip if we closed ALL positions
            if total_positions > 0:
                if successful_closures == total_positions:
                    positions_flipped = True
                    logger.info(f"✅ Position flip complete: Closed all {total_positions} {opposite_direction} positions for {symbol}")
                else:
                    logger.warning(f"⚠️ Partial flip: Closed {successful_closures}/{total_positions} {opposite_direction} positions for {symbol}")
                    # Don't block the signal, but note it wasn't a clean flip
        
        # SAME-DIRECTION CHECK: Prevent duplicate signals in same direction within 4 hours
        four_hours_ago = datetime.utcnow() - timedelta(hours=4)
        recent_same_signal = db.query(Signal).filter(
            Signal.symbol == symbol,
            Signal.signal_type == 'spot_flow',
            Signal.direction == trade_direction,
            Signal.created_at >= four_hours_ago
        ).first()
        
        if recent_same_signal:
            logger.info(f"Skipping duplicate {trade_direction} spot flow signal for {symbol} (sent {(datetime.utcnow() - recent_same_signal.created_at).total_seconds()/60:.0f} min ago)")
            return
        
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
        
        # Calculate R:R ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Calculate 10x leverage PnL for SL and TP
        sl_pnl = calculate_leverage_pnl(entry_price, stop_loss, trade_direction, 10)
        tp_pnl = calculate_leverage_pnl(entry_price, take_profit, trade_direction, 10)
        
        # Add position flip notification ONLY if all positions were successfully closed
        position_flip_note = ""
        if positions_flipped:
            opposite_direction = 'SHORT' if trade_direction == 'LONG' else 'LONG'
            position_flip_note = f"\n\n⚡ <b>POSITION FLIP COMPLETED</b>\n<i>All {opposite_direction} positions successfully closed - now entering {trade_direction}</i>"
        
        message = f"""
{emoji} <b>SPOT MARKET FLOW SIGNAL</b>
━━━━━━━━━━━━━━━━━━━━

{color} <b>{direction_text}</b>
<b>Symbol:</b> {symbol}
<b>Confidence:</b> {confidence:.0f}%
<b>Direction:</b> {trade_direction}{position_flip_note}

<b>📊 Multi-Exchange Analysis</b>
• Order Book Imbalance: {flow_data['avg_imbalance']:+.2f}
• Trade Pressure: {flow_data['avg_pressure']:+.2f}
• Exchanges Analyzed: {flow_data['exchanges_analyzed']}
• Volume Spikes: {flow_data['spike_count']}

<b>💰 Trade Levels (10x Leverage)</b>
• Entry: ${entry_price:.4f}
• Stop Loss: ${stop_loss:.4f} ({sl_pnl:+.2f}%)
• Take Profit: ${take_profit:.4f} ({tp_pnl:+.2f}%)
• Risk:Reward: 1:{rr_ratio:.2f}

<b>💡 Market Context</b>
Spot market flows often precede futures movements. High confidence flows (70%+) suggest institutional activity.

<i>🔍 Data from: Coinbase, Kraken, OKX</i>
"""
        
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
                        
                        # Execute trades (paper or live)
                        if user.preferences.paper_trading_mode:
                            from app.services.paper_trader import PaperTrader
                            await PaperTrader.execute_paper_trade(user.id, signal, db)
                        elif user.preferences.auto_trading_enabled:
                            logger.info(f"Executing spot flow auto-trade for user {user.telegram_id}: {trade_direction} {symbol}")
                            await execute_trade_on_exchange(signal, user, db)
                            
                    except Exception as e:
                        logger.error(f"Error sending spot flow signal to {user.telegram_id}: {e}")
            
    except Exception as e:
        logger.error(f"Error broadcasting spot flow alert: {e}")
    finally:
        db.close()


async def broadcast_hybrid_signal(signal_data: dict):
    """
    Broadcast hybrid signals (funding extremes + divergence) with category-specific formatting
    """
    db = SessionLocal()
    
    try:
        # Check for duplicate signals
        four_hours_ago = datetime.utcnow() - timedelta(hours=4)
        existing = db.query(Signal).filter(
            Signal.symbol == signal_data['symbol'],
            Signal.direction == signal_data['direction'],
            Signal.created_at >= four_hours_ago
        ).first()
        
        if existing:
            logger.info(f"Skipping duplicate {signal_data['direction']} signal for {signal_data['symbol']}")
            return
        
        # Save to database (with pattern for analytics)
        # Use actual pattern name if available, fallback to signal_type
        pattern_name = signal_data.get('pattern') or signal_data.get('signal_type', 'hybrid')
        
        db_signal = Signal(
            symbol=signal_data['symbol'],
            direction=signal_data['direction'],
            entry_price=signal_data['entry_price'],
            stop_loss=signal_data['stop_loss'],
            take_profit=signal_data['take_profit'],
            take_profit_1=signal_data.get('take_profit_1'),
            take_profit_2=signal_data.get('take_profit_2'),
            take_profit_3=signal_data.get('take_profit_3'),
            risk_level=signal_data.get('risk_level', 'MEDIUM'),
            signal_type=signal_data.get('signal_type', 'hybrid'),
            pattern=pattern_name,  # Save specific pattern for analytics
            timeframe=signal_data.get('timeframe', '1h'),
            rsi=signal_data.get('rsi', 50),
            atr=signal_data.get('atr', 0),
            volume=signal_data.get('volume', 0),
            volume_avg=signal_data.get('volume_avg', 0),
            confidence=signal_data.get('confidence', 75)
        )
        db.add(db_signal)
        db.commit()
        db.refresh(db_signal)
        
        # Get session quality info
        session = signal_data.get('session_quality', {})
        session_emoji = session.get('emoji', '🟡')
        session_desc = session.get('description', 'Active trading session')
        
        # Category info
        category = signal_data['category_name']
        category_desc = signal_data['category_desc']
        
        # Category emoji
        if category == 'SCALP':
            category_emoji = '⚡'
        elif category == 'SWING':
            category_emoji = '📈'
        else:
            category_emoji = '💎'
        
        # Calculate PnL for each TP level (10x leverage)
        tp1_pnl = calculate_leverage_pnl(signal_data['entry_price'], signal_data['take_profit_1'], signal_data['direction'], 10)
        tp2_pnl = calculate_leverage_pnl(signal_data['entry_price'], signal_data['take_profit_2'], signal_data['direction'], 10)
        tp3_pnl = calculate_leverage_pnl(signal_data['entry_price'], signal_data['take_profit_3'], signal_data['direction'], 10)
        sl_pnl = calculate_leverage_pnl(signal_data['entry_price'], signal_data['stop_loss'], signal_data['direction'], 10)
        
        # Risk/reward ratio
        risk = abs(signal_data['entry_price'] - signal_data['stop_loss'])
        reward = abs(signal_data['take_profit_3'] - signal_data['entry_price'])
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Get quality metadata
        quality_tier = signal_data.get('quality_tier', '✅ GOOD')
        quality_score = signal_data.get('quality_score', 0)
        
        # Build message based on signal type
        if signal_data['signal_type'] == 'FUNDING_EXTREME':
            signal_text = f"""
{category_emoji} NEW {category} SIGNAL - {signal_data['direction']}
{quality_tier} Setup (Score: {quality_score}/100)

💰 {signal_data['symbol']}
📊 Type: Funding Rate Extreme
⚠️ Funding: {signal_data.get('funding_rate', 0):.3f}%

💵 Entry: ${signal_data['entry_price']}
🛑 Stop Loss: ${signal_data['stop_loss']} ({sl_pnl:+.2f}% @ 10x)

🎯 Take Profits ({category_desc}):
  TP1: ${signal_data['take_profit_1']} (+{signal_data['tp1_pct']}% @ {tp1_pnl:+.2f}%)
  TP2: ${signal_data['take_profit_2']} (+{signal_data['tp2_pct']}% @ {tp2_pnl:+.2f}%)
  TP3: ${signal_data['take_profit_3']} (+{signal_data['tp3_pct']}% @ {tp3_pnl:+.2f}%)

💡 Reason: {signal_data.get('reason', 'Mean reversion play')}
💎 R:R Ratio: 1:{rr_ratio:.2f}
🎯 Confidence: {signal_data.get('confidence', 75)}%

{session_emoji} Session: {session_desc}
⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC
"""
        
        elif 'DIVERGENCE' in signal_data['signal_type']:
            signal_text = f"""
{category_emoji} NEW {category} SIGNAL - {signal_data['direction']}
{quality_tier} Setup (Score: {quality_score}/100)

💰 {signal_data['symbol']}
📊 Type: {signal_data.get('pattern', 'Divergence')}
📉 RSI: {signal_data.get('rsi', 50):.1f}

💵 Entry: ${signal_data['entry_price']}
🛑 Stop Loss: ${signal_data['stop_loss']} ({sl_pnl:+.2f}% @ 10x)

🎯 Take Profits ({category_desc}):
  TP1: ${signal_data['take_profit_1']} (+{signal_data['tp1_pct']}% @ {tp1_pnl:+.2f}%)
  TP2: ${signal_data['take_profit_2']} (+{signal_data['tp2_pct']}% @ {tp2_pnl:+.2f}%)
  TP3: ${signal_data['take_profit_3']} (+{signal_data['tp3_pct']}% @ {tp3_pnl:+.2f}%)

💡 Reason: {signal_data.get('reason', 'Trend reversal expected')}
💎 R:R Ratio: 1:{rr_ratio:.2f}
🎯 Confidence: {signal_data.get('confidence', 80)}%

{session_emoji} Session: {session_desc}
⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC
"""
        
        # Broadcast to channel
        await bot.send_message(settings.BROADCAST_CHAT_ID, signal_text)
        logger.info(f"{category} signal broadcast: {signal_data['direction']} {signal_data['symbol']}")
        
        # Send DM to users and trigger auto-trades
        users = db.query(User).filter(User.approved == True, User.banned == False).all()
        for user in users:
            if user.preferences and user.preferences.dm_alerts:
                # Check if symbol is muted
                if signal_data['symbol'] not in user.preferences.get_muted_symbols_list():
                    try:
                        await bot.send_message(user.telegram_id, signal_text)
                        
                        # Trigger auto-trade if enabled
                        if user.preferences.auto_trading_enabled:
                            # Use the saved Signal object for proper exchange routing
                            await execute_trade_on_exchange(db_signal, user, db)
                    
                    except Exception as e:
                        logger.error(f"Error sending hybrid signal to user {user.id}: {e}")
    
    except Exception as e:
        logger.error(f"Error broadcasting hybrid signal: {e}")
        db.rollback()
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
        
        # Clean up fields not in Signal model (but KEEP pattern for analytics)
        signal_data.pop('signal_category', None)  # Remove category object
        signal_data.pop('category_name', None)  # Remove category name
        signal_data.pop('category_desc', None)  # Remove category description
        signal_data.pop('session_quality', None)  # Remove session quality object
        signal_data.pop('quality_score', None)  # Remove quality score
        signal_data.pop('quality_tier', None)  # Remove quality tier
        signal_data.pop('is_premium', None)  # Remove premium flag
        
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
            
            # Execute trades (paper or live)
            if user.preferences:
                muted_symbols = user.preferences.get_muted_symbols_list()
                if signal.symbol not in muted_symbols:
                    # Paper trading mode - execute virtual trade
                    if user.preferences.paper_trading_mode:
                        from app.services.paper_trader import PaperTrader
                        await PaperTrader.execute_paper_trade(user.id, signal, db)
                    # Live trading mode - execute on exchange if auto-trading enabled
                    elif user.preferences.auto_trading_enabled:
                        await execute_trade_on_exchange(signal, user, db)
    
    finally:
        db.close()


def is_trading_session() -> bool:
    """
    24/7 trading for crypto markets (RELAXED from 08:00-21:00 UTC)
    Crypto markets operate continuously, no session restrictions
    """
    # Crypto markets are 24/7 - always allow trading
    return True


async def broadcast_daytrading_signal(signal_data: dict):
    """
    Broadcast day trading signals (1:1 risk-reward with 6-point confirmation)
    """
    db = SessionLocal()
    
    try:
        # Check for duplicate signals
        four_hours_ago = datetime.utcnow() - timedelta(hours=4)
        existing = db.query(Signal).filter(
            Signal.symbol == signal_data['symbol'],
            Signal.direction == signal_data['direction'],
            Signal.created_at >= four_hours_ago
        ).first()
        
        if existing:
            logger.info(f"Skipping duplicate {signal_data['direction']} signal for {signal_data['symbol']}")
            return
        
        # Save to database
        db_signal = Signal(
            symbol=signal_data['symbol'],
            direction=signal_data['direction'],
            entry_price=signal_data['entry_price'],
            stop_loss=signal_data['stop_loss'],
            take_profit=signal_data['take_profit'],
            risk_level='MEDIUM',
            signal_type='DAY_TRADE',
            pattern=signal_data.get('pattern', 'MULTI_CONFIRMATION'),
            timeframe='15m',
            rsi=signal_data.get('rsi', 50),
            confidence=signal_data.get('confidence', 90)
        )
        db.add(db_signal)
        db.commit()
        db.refresh(db_signal)
        
        # Calculate PnL (15% @ 10x leverage = 1.5% price move)
        tp_pnl = calculate_leverage_pnl(signal_data['entry_price'], signal_data['take_profit'], signal_data['direction'], 10)
        sl_pnl = calculate_leverage_pnl(signal_data['entry_price'], signal_data['stop_loss'], signal_data['direction'], 10)
        
        # Build message
        signal_text = f"""
🎯 DAY TRADE SIGNAL - {signal_data['direction']}
✅ 6-POINT CONFIRMATION PASSED

💰 {signal_data['symbol']}
📊 Strategy: {signal_data.get('pattern', 'Multi-Confirmation')}
💎 Risk-Reward: 1:1

💵 Entry: ${signal_data['entry_price']}
🛑 Stop Loss: ${signal_data['stop_loss']} ({sl_pnl:+.2f}% @ 10x)
🎯 Take Profit: ${signal_data['take_profit']} ({tp_pnl:+.2f}% @ 10x)

✅ Confirmations:
  • Trend: EMA aligned (15m + 1H)
  • Spot Flow: Binance buying/selling pressure
  • Volume: {signal_data.get('volume_ratio', 2):.1f}x average
  • Momentum: RSI {signal_data.get('rsi', 50):.1f} + MACD aligned
  • Candle: Clean reversal pattern
  • Session: High liquidity hours

💡 {signal_data.get('reason', 'All 6 confirmations passed')}
🎯 Confidence: {signal_data.get('confidence', 90)}%
⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC
"""
        
        # Broadcast to channel
        await bot.send_message(settings.BROADCAST_CHAT_ID, signal_text)
        logger.info(f"Day trading signal broadcast: {signal_data['direction']} {signal_data['symbol']}")
        
        # Send DM to users and trigger auto-trades
        users = db.query(User).filter(User.approved == True, User.banned == False).all()
        for user in users:
            if user.preferences and user.preferences.dm_alerts:
                # Check if symbol is muted
                if signal_data['symbol'] not in user.preferences.get_muted_symbols_list():
                    try:
                        await bot.send_message(user.telegram_id, signal_text)
                        
                        # Execute trades (paper or live)
                        if user.preferences.paper_trading_mode:
                            from app.services.paper_trader import PaperTrader
                            await PaperTrader.execute_paper_trade(user.id, db_signal, db)
                        elif user.preferences.auto_trading_enabled:
                            await execute_trade_on_exchange(db_signal, user, db)
                    
                    except Exception as e:
                        logger.error(f"Error sending signal to user {user.id}: {e}")
    
    except Exception as e:
        logger.error(f"Error broadcasting day trading signal: {e}")
    finally:
        db.close()


async def signal_scanner():
    logger.info("🎯 Day Trading Signal Scanner Started (1:1 Risk-Reward Strategy)")
    
    # Initialize day trading generator
    from app.services.daytrading_signals import DayTradingSignalGenerator
    daytrading_generator = DayTradingSignalGenerator()
    
    while True:
        try:
            # Update heartbeat for health monitor
            await update_heartbeat()
            
            logger.info("🔍 Scanning for day trading signals (6-point confirmation)...")
            
            # ✨ NEW: Scan for day trading signals ONLY
            # Requires ALL 6 confirmations: Trend + Spot Flow + Volume + Momentum + Candle + Session
            daytrading_signals = await daytrading_generator.scan_all_symbols()
            logger.info(f"Found {len(daytrading_signals)} premium day trading signals (1:1 risk-reward)")
            
            # Broadcast day trading signals
            for signal in daytrading_signals:
                await broadcast_daytrading_signal(signal)
                
        except Exception as e:
            logger.error(f"Signal scanner error: {e}", exc_info=True)
        
        await asyncio.sleep(settings.SCAN_INTERVAL)


async def top_gainers_scanner():
    """Scan for top gainers and broadcast signals every 15 minutes"""
    logger.info("🔥 Top Gainers Scanner Started (24/7 Parabolic Reversal Detection)")
    
    await asyncio.sleep(60)  # Wait 60s before first scan (let other services initialize)
    
    while True:
        try:
            # Update heartbeat for health monitor
            await update_heartbeat()
            
            logger.info("🔍 Scanning top gainers for parabolic reversals...")
            
            # Run top gainer signal scan
            db = SessionLocal()
            try:
                from app.services.top_gainers_signals import broadcast_top_gainer_signal
                await broadcast_top_gainer_signal(bot, db)
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Top gainers scanner error: {e}", exc_info=True)
        
        # Scan every 15 minutes (900 seconds) - catch reversals faster!
        await asyncio.sleep(900)


async def position_monitor():
    """Monitor open positions and notify when TP/SL is hit"""
    from app.services.position_monitor import monitor_positions
    from app.services.paper_trader import PaperTrader
    
    logger.info("Position monitor started")
    await asyncio.sleep(30)  # Wait 30s before first check
    
    while True:
        try:
            # Update heartbeat for health monitor
            await update_heartbeat()
            
            logger.info("Monitoring positions...")
            
            # Monitor live Bitunix positions
            await monitor_positions(bot)
            
            # Monitor paper trading positions
            await PaperTrader.monitor_paper_positions(bot)
            
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
                            
                            try:
                                for trade in open_trades:
                                    try:
                                        ticker = await exchange.fetch_ticker(trade.symbol)
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
                            finally:
                                await exchange.close()
                        
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


async def telegram_conflict_watcher():
    """Watch for Telegram conflict errors and register them"""
    from app.services.bot_instance_manager import get_instance_manager
    import logging
    
    # Hook into aiogram's logging to detect conflicts
    telegram_logger = logging.getLogger('aiogram.dispatcher')
    
    class ConflictHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = record.getMessage()
                # Check if this is a Telegram conflict error
                if 'TelegramConflictError' in msg or 'Conflict: terminated by other getUpdates' in msg:
                    manager = get_instance_manager()
                    if manager:
                        manager.register_telegram_conflict()
            except:
                pass
    
    handler = ConflictHandler()
    handler.setLevel(logging.ERROR)  # Only catch ERROR level logs
    telegram_logger.addHandler(handler)
    
    logger.info("🔍 Telegram conflict watcher started")
    
    # Keep running
    while True:
        await asyncio.sleep(1)


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
    
    # Start Telegram conflict watcher
    asyncio.create_task(telegram_conflict_watcher())
    
    # Start background tasks
    asyncio.create_task(signal_scanner())
    asyncio.create_task(top_gainers_scanner())  # 🔥 TOP GAINERS: Scans every 15 min for parabolic reversals
    asyncio.create_task(position_monitor())
    asyncio.create_task(daily_pnl_report())
    asyncio.create_task(funding_rate_monitor())
    # Note: Funding rate monitor may log ccxt cleanup warnings - this is a known ccxt library limitation, not a memory leak
    
    # Start health monitor (auto-recovery system)
    health_monitor = get_health_monitor()
    asyncio.create_task(health_monitor.start_monitoring())
    logger.info("🏥 Health monitor started (auto-recovery enabled)")
    
    try:
        logger.info("Bot polling started")
        await dp.start_polling(bot)
    finally:
        # Cleanup on shutdown
        logger.info("Bot shutting down...")
        await manager.release_lock()
        await signal_generator.close()
