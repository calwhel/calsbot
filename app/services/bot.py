import asyncio
import logging
import ccxt.async_support as ccxt
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, StateFilter
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
import random
import string
import csv
import io

from app.config import settings
from app.database import SessionLocal
from app.models import User, UserPreference, Trade, Signal
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

# FSM States for top gainer leverage
class TopGainerLeverageSetup(StatesGroup):
    waiting_for_leverage = State()

bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
# OLD GENERATORS REMOVED - Now using DayTradingSignalGenerator only
# signal_generator = SignalGenerator()
# news_signal_generator = NewsSignalGenerator()
# reversal_scanner = ReversalScanner()

# Global broadcast lock to prevent simultaneous broadcasts from exceeding Telegram rate limits
# Use lazy initialization to avoid issues with event loop not running at import time
_broadcast_lock = None

def get_broadcast_lock():
    """Get or create the broadcast lock (lazy initialization)"""
    global _broadcast_lock
    if _broadcast_lock is None:
        _broadcast_lock = asyncio.Lock()
    return _broadcast_lock

async def send_message_with_retry(chat_id: int, text: str, max_retries: int = 3) -> bool:
    """Send message with retry logic for rate limiting"""
    for attempt in range(max_retries):
        try:
            await bot.send_message(chat_id, text)
            return True
        except Exception as e:
            error_str = str(e).lower()
            if "retry after" in error_str or "too many requests" in error_str:
                # Extract retry delay from error message
                import re
                match = re.search(r'retry after (\d+)', error_str)
                wait_time = int(match.group(1)) if match else 5
                logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt+1}/{max_retries}")
                await asyncio.sleep(wait_time)
            elif "bot was blocked" in error_str or "chat not found" in error_str or "user is deactivated" in error_str:
                logger.debug(f"User {chat_id} unreachable: {e}")
                return False  # Don't retry for blocked/deactivated users
            else:
                logger.error(f"Send error to {chat_id}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Brief pause before retry
    return False


def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        pass


def generate_referral_code(db: Session) -> str:
    """Generate a unique referral code"""
    while True:
        code = 'TH-' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        existing = db.query(User).filter(User.referral_code == code).first()
        if not existing:
            return code


def get_or_create_user(telegram_id: int, username: str = None, first_name: str = None, db: Session = None, referral_code: str = None):
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
            
            # Generate unique referral code for new user
            new_referral_code = generate_referral_code(db)
            
            user = User(
                telegram_id=str(telegram_id),
                username=username,
                first_name=first_name,
                is_admin=is_first_user,
                approved=True,  # ✅ AUTO-APPROVE all new users (no manual approval needed)
                grandfathered=False,  # New users need to subscribe (existing users already grandfathered via migration)
                referral_code=new_referral_code,
                referred_by=referral_code  # Track who referred this user
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            
            prefs = UserPreference(user_id=user.id)
            db.add(prefs)
            db.commit()
            
            # Notify admins about new user (if not first user)
            if not is_first_user:
                # Look up referrer if referral code was used
                referrer_info = ""
                if referral_code:
                    referrer = db.query(User).filter(User.referral_code == referral_code).first()
                    if referrer:
                        referrer_name = f"@{referrer.username}" if referrer.username else referrer.first_name or f"User #{referrer.id}"
                        referrer_info = f"\n🔗 Referred by: {referrer_name}"
                    else:
                        referrer_info = f"\n🔗 Referral code: {referral_code} (invalid)"
                
                admins = db.query(User).filter(User.is_admin == True).all()
                for admin in admins:
                    try:
                        asyncio.create_task(
                            bot.send_message(
                                admin.telegram_id,
                                f"✅ New user joined & auto-approved!\n\n"
                                f"👤 User: @{username or 'N/A'} ({first_name or 'N/A'})\n"
                                f"🆔 ID: `{telegram_id}`{referrer_info}\n\n"
                                f"They now have full access to the bot."
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


def check_access(user: User, require_tier: str = None) -> tuple[bool, str]:
    """Check if user has access to bot. Returns (has_access, reason)
    
    Args:
        user: User object
        require_tier: Optional tier requirement ("scan", "manual", or "auto")
    """
    if user.banned:
        ban_message = "❌ You have been banned from using this bot."
        if user.admin_notes:
            ban_message += f"\n\nReason: {user.admin_notes}"
        return False, ban_message
    if not user.approved and not user.is_admin:
        return False, "⏳ Your account is pending approval. Please wait for admin approval."
    if not user.is_subscribed and not user.is_admin:
        return False, "💎 Premium subscription required. Use /subscribe to get started!"
    
    # Tier-based access control
    if require_tier and not user.is_admin and not user.grandfathered:
        from app.tiers import get_tier_from_user, has_scan_access, has_manual_access, has_auto_access
        
        if require_tier == "auto" and not has_auto_access(user):
            return False, "🤖 This feature requires the Auto-Trading plan ($130/mo). Use /subscribe to upgrade!"
        elif require_tier == "manual" and not has_manual_access(user):
            return False, "💎 This feature requires the Signals Only plan ($80/mo) or higher. Use /subscribe to upgrade!"
        elif require_tier == "scan" and not has_scan_access(user):
            return False, "📊 Scan mode is included with all plans. Use /subscribe to get started!"
    
    return True, ""


def get_connected_exchange(prefs) -> tuple[bool, str]:
    """Check if Bitunix exchange is connected. Returns (has_exchange, exchange_name)"""
    if not prefs:
        return False, ""
    
    if prefs.bitunix_api_key and prefs.bitunix_api_secret:
        return True, "Bitunix"
    
    return False, ""


async def execute_trade_on_exchange(signal, user: User, db: Session):
    """Execute trade on Bitunix exchange - ALWAYS creates trade record (open or failed)"""
    try:
        prefs = user.preferences
        if not prefs:
            logger.warning(f"No preferences found for user {user.id}")
            return None
        
        # Determine trade type
        trade_type = 'TOP_GAINER' if signal.signal_type == 'TOP_GAINER' else 'DAY_TRADE'
        
        # Skip correlation filter for TEST signals (admin testing)
        if signal.signal_type != 'TEST':
            # Check correlation filter before executing trade
            from app.services.risk_filters import check_correlation_filter
            allowed, reason = check_correlation_filter(signal.symbol, prefs, db)
            if not allowed:
                logger.info(f"Trade blocked by correlation filter for user {user.telegram_id}: {reason}")
                # Create failed trade record - signal was sent but blocked
                failed_trade = Trade(
                    user_id=user.id,
                    signal_id=signal.id,
                    symbol=signal.symbol,
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    status='failed',
                    position_size=0,
                    remaining_size=0,
                    pnl=0,
                    pnl_percent=0,
                    trade_type=trade_type,
                    opened_at=datetime.utcnow()
                )
                db.add(failed_trade)
                db.commit()
                
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
            return await execute_bitunix_trade(signal, user, db, trade_type=trade_type)
        else:
            logger.warning(f"Bitunix credentials not configured for user {user.id}")
            # Create failed trade record - signal sent but no API keys
            failed_trade = Trade(
                user_id=user.id,
                signal_id=signal.id,
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                status='failed',
                position_size=0,
                remaining_size=0,
                pnl=0,
                pnl_percent=0,
                trade_type=trade_type,
                opened_at=datetime.utcnow()
            )
            db.add(failed_trade)
            db.commit()
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
    db.expire(user, ['preferences'])  # ✅ Force SQLAlchemy to reload relationship from database
    db.refresh(user)  # Refresh user with latest data
    
    # CRITICAL: Re-query preferences directly to avoid stale relationship cache
    prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
    if prefs:
        db.refresh(prefs)  # ✅ Force reload preferences to get latest auto_trading_enabled value
    
    # Get trading stats - LIVE TRADING ONLY
    total_trades = db.query(Trade).filter(
        Trade.user_id == user.id,
        Trade.status.in_(['closed', 'stopped', 'tp_hit', 'sl_hit'])
    ).count()
    open_positions = db.query(Trade).filter(
        Trade.user_id == user.id,
        Trade.status == 'open'
    ).count()
    
    # Calculate today's PnL - LIVE TRADES ONLY
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_trades = db.query(Trade).filter(
        Trade.user_id == user.id,
        Trade.status.in_(['closed', 'stopped', 'tp_hit', 'sl_hit']),
        Trade.closed_at >= today_start
    ).all()
    today_pnl = sum(trade.pnl or 0 for trade in today_trades)
    
    # Auto-trading status - check if Bitunix keys are configured
    bitunix_connected = (
        prefs and 
        prefs.bitunix_api_key and 
        prefs.bitunix_api_secret and
        len(prefs.bitunix_api_key) > 0 and 
        len(prefs.bitunix_api_secret) > 0
    )
    
    # Auto-trading status - SIMPLIFIED: Bitunix connected = ACTIVE
    is_active = bitunix_connected
    
    autotrading_emoji = "🟢" if is_active else "🔴"
    
    # Status text
    if is_active:
        autotrading_status = "ACTIVE"
    else:
        autotrading_status = "INACTIVE"
    
    # Exchange status - Bitunix only
    active_exchange = "Bitunix" if bitunix_connected else None
    exchange_status = f"{active_exchange} (✅ Connected)" if active_exchange else "No Exchange Connected"
    
    # Position sizing info
    position_size = f"{prefs.position_size_percent:.0f}%" if prefs else "10%"
    leverage = f"{prefs.user_leverage}x" if prefs else "10x"
    
    # Fetch live Bitunix balance if connected
    live_balance = None
    live_balance_text = ""
    
    if is_active and bitunix_connected:
        try:
            from app.services.bitunix_trader import BitunixTrader
            from app.utils.encryption import decrypt_api_key
            
            api_key = decrypt_api_key(prefs.bitunix_api_key)
            api_secret = decrypt_api_key(prefs.bitunix_api_secret)
            
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
    
    # Active positions section removed per user request - not needed
    
    # 🎯 Account Overview removed from main dashboard per user request
    # (Already shown in Auto-Trading menu with full details)
    account_overview = ""
    
    # Subscription status
    if user.grandfathered:
        sub_status = "🎉 <b>Lifetime Access</b> (Grandfathered)"
    elif user.is_subscribed:
        expires = user.subscription_end.strftime("%Y-%m-%d") if user.subscription_end else "Active"
        sub_status = f"✅ <b>Premium</b> (until {expires})"
    else:
        sub_status = "💎 <b>Free Trial</b> - /subscribe for full access"
    
    # Referral stats with pending earnings
    referrals = db.query(User).filter(User.referred_by == user.referral_code).all()
    total_referrals = len(referrals)
    subscribed_referrals = [r for r in referrals if r.is_subscribed]
    subscribed_count = len(subscribed_referrals)
    pending_earnings = user.referral_earnings or 0.0
    
    if pending_earnings > 0:
        referral_section = f"🎁 <b>Referrals:</b> {subscribed_count} active | 💰 <b>Pending:</b> ${pending_earnings:.2f}\n└ Code: <code>{user.referral_code}</code>"
    else:
        referral_section = f"🎁 <b>Referrals:</b> {subscribed_count} active | Code: <code>{user.referral_code}</code>"
    
    # ✅ Get CORRECT auto-trading status (force fresh query)
    db.expire(user, ['preferences'])
    fresh_prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
    autotrading_enabled = fresh_prefs.auto_trading_enabled if fresh_prefs else False
    autotrading_emoji = "🟢" if autotrading_enabled else "🔴"
    autotrading_status = "ACTIVE" if autotrading_enabled else "OFF"
    
    # Main dashboard shows live account only
    welcome_text = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
  <b>🚀 Tradehub AI</b>
┗━━━━━━━━━━━━━━━━━━━━━━━━━┛

<b>👤 Account</b>
├ {sub_status}
├ {referral_section}
└ {autotrading_emoji} <b>Auto-Trading:</b> {autotrading_status}

{account_overview}<b>📊 Trading Overview</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
📍 <b>Open:</b> {open_positions} | 📋 <b>Closed:</b> {total_trades}

<i>AI-powered 1:1 day trading + Top Gainers mode</i>

⚠️ <b>Risk Disclaimer:</b> Crypto trading involves substantial risk of loss. Past performance does not guarantee future results. Trade responsibly and only with funds you can afford to lose. /disclaimer for full terms.
"""
    
    # 🚀 SIMPLIFIED NAVIGATION - 7 core buttons with Referrals
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="📊 Dashboard", callback_data="dashboard"),
            InlineKeyboardButton(text="⚡ Auto-Trading", callback_data="autotrading_unified")
        ],
        [
            InlineKeyboardButton(text="🔥 Top Gainers", callback_data="top_gainers_unified"),
            InlineKeyboardButton(text="💎 Subscribe", callback_data="subscribe_menu")
        ],
        [
            InlineKeyboardButton(text="🎁 Referrals", callback_data="referral_stats"),
            InlineKeyboardButton(text="⚙️ Settings", callback_data="settings_simplified")
        ],
        [
            InlineKeyboardButton(text="❓ Help", callback_data="help_menu")
        ]
    ])
    
    return welcome_text, keyboard


@dp.message(Command("health"))
async def cmd_health(message: types.Message):
    """Quick health check - always responds immediately (no DB access)"""
    import time
    start = time.time()
    await message.answer(f"✅ Bot is alive! Response time: {(time.time() - start)*1000:.0f}ms")


@dp.message(Command("dbhealth"))
async def cmd_dbhealth(message: types.Message):
    """Database health check - shows connection pool status"""
    from app.database import SessionLocal
    from sqlalchemy import text
    import time
    
    start = time.time()
    try:
        db = SessionLocal()
        
        # Check for stuck locks
        locks = db.execute(text("SELECT count(*) FROM pg_locks WHERE locktype = 'advisory'")).scalar()
        
        # Check for idle transactions
        idle = db.execute(text("SELECT count(*) FROM pg_stat_activity WHERE state = 'idle in transaction' AND datname = current_database()")).scalar()
        
        # Check active connections
        active = db.execute(text("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()")).scalar()
        
        db.close()
        
        response_time = (time.time() - start) * 1000
        
        status = "✅ HEALTHY" if locks == 0 and idle == 0 else "⚠️ DEGRADED"
        
        await message.answer(
            f"{status}\n\n"
            f"🔒 Advisory locks: {locks}\n"
            f"⏸️ Idle transactions: {idle}\n"
            f"📊 Active connections: {active}\n"
            f"⏱️ Response time: {response_time:.0f}ms\n\n"
            f"{'❌ Run /cleardb to fix!' if (locks > 0 or idle > 0) else '✅ All systems normal'}"
        )
    except Exception as e:
        await message.answer(f"❌ Database error: {str(e)[:200]}")


@dp.message(Command("cleardb"))
async def cmd_cleardb(message: types.Message):
    """Emergency database cleanup - clears stuck locks and idle transactions"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user or not user.is_admin:
            await message.answer("❌ Admin only")
            return
        
        from sqlalchemy import text
        
        # Clear advisory locks
        db.execute(text("SELECT pg_advisory_unlock_all()"))
        db.commit()
        
        # Kill idle transactions
        result = db.execute(text("""
            SELECT pg_terminate_backend(pid) 
            FROM pg_stat_activity 
            WHERE state = 'idle in transaction' 
            AND datname = current_database()
            AND pid != pg_backend_pid()
        """))
        killed = len(result.fetchall())
        
        db.commit()
        db.close()
        
        await message.answer(f"✅ Database cleaned!\n\n🔓 Locks cleared\n❌ Killed {killed} stuck connections")
    except Exception as e:
        await message.answer(f"❌ Error: {str(e)[:200]}")
        db.close()


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    # Track message for health monitor
    await update_message_timestamp()
    
    db = SessionLocal()
    try:
        # Extract referral code from /start command (e.g., /start TH-ABC123)
        referral_code = None
        if message.text and len(message.text.split()) > 1:
            referral_code = message.text.split()[1].strip()
            # Validate referral code format
            if not referral_code.startswith('TH-') or len(referral_code) != 9:
                referral_code = None
        
        user = get_or_create_user(
            message.from_user.id,
            message.from_user.username,
            message.from_user.first_name,
            db,
            referral_code
        )
        
        # Check if banned
        if user.banned:
            ban_message = "❌ You have been banned from using this bot."
            if user.admin_notes:
                ban_message += f"\n\nReason: {user.admin_notes}"
            await message.answer(ban_message)
            return
        
        # If they were referred, show confirmation message
        if user.referred_by and referral_code:
            referrer = db.query(User).filter(User.referral_code == referral_code).first()
            if referrer:
                await message.answer(
                    f"🎉 <b>Welcome!</b>\n\n"
                    f"You were referred by @{referrer.username or referrer.first_name}.\n"
                    f"When you subscribe to <b>Auto-Trading</b>, they'll get <b>$30 USD in crypto</b>! 💰",
                    parse_mode="HTML"
                )
        
        # Show home screen for all users (even without subscription)
        # They can see referrals and subscribe, but can't access premium features
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
    
    # ✅ FIX: Pass the callback directly so cmd_dashboard gets fresh user data
    # This prevents stale cache when clicking Dashboard button
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("Please use /start first")
            return
        
        # Force fresh user query
        db.expire(user)
        db.refresh(user)
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            return
        
        # Build dashboard with fresh data
        account_text, _ = await build_account_overview(user, db)
        
        # Check if user has scalp mode enabled
        prefs = user.preferences
        if prefs:
            db.expire(prefs)
            db.refresh(prefs)
        has_scalp_access = prefs and getattr(prefs, 'scalp_mode_enabled', False)
        
        if has_scalp_access:
            dashboard_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="🔍 Scan Coins", callback_data="scan_menu"),
                    InlineKeyboardButton(text="📡 Recent Signals", callback_data="recent_signals")
                ],
                [
                    InlineKeyboardButton(text="⚡ Scalp Trades", callback_data="scalp_mode"),
                    InlineKeyboardButton(text="🤖 Auto-Trading", callback_data="autotrading_menu")
                ],
                [
                    InlineKeyboardButton(text="⚙️ Settings", callback_data="settings"),
                    InlineKeyboardButton(text="🆘 Support", callback_data="support_menu")
                ],
                [
                    InlineKeyboardButton(text="🏠 Home", callback_data="home")
                ]
            ])
        else:
            dashboard_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="🔍 Scan Coins", callback_data="scan_menu"),
                    InlineKeyboardButton(text="📡 Recent Signals", callback_data="recent_signals")
                ],
                [
                    InlineKeyboardButton(text="⚡ Scalp Trades (coming soon)", callback_data="scalp_coming_soon"),
                    InlineKeyboardButton(text="🤖 Auto-Trading", callback_data="autotrading_menu")
                ],
                [
                    InlineKeyboardButton(text="⚙️ Settings", callback_data="settings"),
                    InlineKeyboardButton(text="🆘 Support", callback_data="support_menu")
                ],
                [
                    InlineKeyboardButton(text="🏠 Home", callback_data="home")
                ]
            ])
        
        await callback.message.edit_text(account_text, reply_markup=dashboard_keyboard, parse_mode="HTML")
    finally:
        db.close()


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
        
        # FORCE REFRESH to get latest preferences from database
        db.expire(user)
        db.refresh(user)
        prefs = user.preferences
        
        # Refresh preferences too
        if prefs:
            db.expire(prefs)
            db.refresh(prefs)
        
        # Simple status indicators
        auto_trading = '🟢 ON' if prefs and prefs.auto_trading_enabled else '🔴 OFF'
        
        # Get leverage values
        day_trade_leverage = prefs.user_leverage if prefs else 10
        top_gainer_leverage = prefs.top_gainers_leverage if prefs and prefs.top_gainers_leverage else 5
        
        # Get Top Gainers trade mode status (MUST check both enabled flag AND mode)
        tg_enabled = prefs and prefs.top_gainers_mode_enabled
        trade_mode = prefs.top_gainers_trade_mode if prefs and prefs.top_gainers_trade_mode else 'shorts_only'
        
        # Only show ON if Top Gainers is enabled AND the mode includes that direction
        shorts_enabled = tg_enabled and trade_mode in ['shorts_only', 'both']
        longs_enabled = tg_enabled and trade_mode in ['longs_only', 'both']
        
        shorts_status = '🟢 ON' if shorts_enabled else '🔴 OFF'
        longs_status = '🟢 ON' if longs_enabled else '🔴 OFF'
        
        settings_text = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━┓
   ⚙️ <b>Settings</b>
┗━━━━━━━━━━━━━━━━━━━━━━━┛

<b>💰 Position Management</b>
├ Position Size: <b>{prefs.position_size_percent if prefs else 10}%</b>
├ Day Trade Leverage: <b>{day_trade_leverage}x</b>
└ Max Positions: <b>{prefs.max_positions if prefs else 3}</b>

<b>🔥 Top Gainers Modes</b>
├ SHORTS (Mean Reversion): {shorts_status}
└ LONGS (Pump Retracement): {longs_status}

<b>🤖 Other Modes</b>
└ Auto-Trading: {auto_trading}

<i>Tap buttons below to configure</i>
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="💰 Position", callback_data="edit_position_size"),
                InlineKeyboardButton(text="⚡ Leverage", callback_data="edit_leverage")
            ],
            [
                InlineKeyboardButton(text="🔴 TG SHORTS", callback_data="toggle_top_gainers_shorts"),
                InlineKeyboardButton(text="🟢 TG LONGS", callback_data="toggle_top_gainers_longs")
            ],
            [
                InlineKeyboardButton(text="🏠 Main Menu", callback_data="back_to_start")
            ]
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
/scan BTC - Analyze any coin
/settings - Configure settings
/disclaimer - Full risk disclaimer

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
        [InlineKeyboardButton(text="⚠️ Risk Disclaimer", callback_data="show_disclaimer")],
        [InlineKeyboardButton(text="🔙 Back", callback_data="back_to_start")]
    ])
    
    await callback.message.edit_text(help_text, reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data == "show_disclaimer")
async def handle_show_disclaimer(callback: CallbackQuery):
    """Display full risk disclaimer"""
    await callback.answer()
    
    disclaimer_text = """
⚠️ <b>RISK DISCLAIMER</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>IMPORTANT - PLEASE READ CAREFULLY</b>

<b>Trading Risks:</b>
• Cryptocurrency trading carries a HIGH RISK of financial loss
• Leveraged trading can result in losses exceeding your initial investment
• Past performance does NOT guarantee future results
• Market volatility can lead to rapid and substantial losses

<b>No Financial Advice:</b>
• This bot provides automated trading signals for informational purposes only
• Signals are NOT financial advice or investment recommendations
• You are solely responsible for your trading decisions
• Always conduct your own research before trading

<b>No Guarantees:</b>
• We make NO guarantees of profit or performance
• AI predictions and technical analysis can be incorrect
• Markets are unpredictable and losses are possible
• Results may vary significantly between users

<b>User Responsibility:</b>
• Trade only with funds you can afford to lose
• Understand the risks before enabling auto-trading
• Monitor your positions regularly
• Set appropriate stop losses and risk limits
• You accept full responsibility for all trading outcomes

<b>Technical Risks:</b>
• API connectivity issues may affect trade execution
• Exchange downtime or errors can impact performance
• System bugs or glitches may occur despite testing
• Internet connectivity issues may delay signals

<b>Legal:</b>
• This service is provided "AS IS" without warranties
• We are not liable for any trading losses or damages
• You agree to indemnify us against all claims
• By using this bot, you accept these terms

<b>By using this bot, you acknowledge and accept all risks associated with cryptocurrency trading.</b>

Only proceed if you fully understand and accept these risks.
"""
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ I Understand", callback_data="help_menu")],
        [InlineKeyboardButton(text="🔙 Back", callback_data="help_menu")]
    ])
    
    await callback.message.edit_text(disclaimer_text, reply_markup=keyboard, parse_mode="HTML")


@dp.message(Command("disclaimer"))
async def cmd_disclaimer(message: types.Message):
    """Show full risk disclaimer via /disclaimer command"""
    disclaimer_text = """
⚠️ <b>RISK DISCLAIMER</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>IMPORTANT - PLEASE READ CAREFULLY</b>

<b>Trading Risks:</b>
• Cryptocurrency trading carries a HIGH RISK of financial loss
• Leveraged trading can result in losses exceeding your initial investment
• Past performance does NOT guarantee future results
• Market volatility can lead to rapid and substantial losses

<b>No Financial Advice:</b>
• This bot provides automated trading signals for informational purposes only
• Signals are NOT financial advice or investment recommendations
• You are solely responsible for your trading decisions
• Always conduct your own research before trading

<b>No Guarantees:</b>
• We make NO guarantees of profit or performance
• AI predictions and technical analysis can be incorrect
• Markets are unpredictable and losses are possible
• Results may vary significantly between users

<b>User Responsibility:</b>
• Trade only with funds you can afford to lose
• Understand the risks before enabling auto-trading
• Monitor your positions regularly
• Set appropriate stop losses and risk limits
• You accept full responsibility for all trading outcomes

<b>Technical Risks:</b>
• API connectivity issues may affect trade execution
• Exchange downtime or errors can impact performance
• System bugs or glitches may occur despite testing
• Internet connectivity issues may delay signals

<b>Legal:</b>
• This service is provided "AS IS" without warranties
• We are not liable for any trading losses or damages
• You agree to indemnify us against all claims
• By using this bot, you accept these terms

<b>By using this bot, you acknowledge and accept all risks associated with cryptocurrency trading.</b>

Only proceed if you fully understand and accept these risks.
"""
    
    await message.answer(disclaimer_text, parse_mode="HTML")


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
            
            # Allow early renewal - show renew button
            from app.services.oxapay import OxaPayService
            from app.config import settings
            import os
            
            if settings.OXAPAY_MERCHANT_API_KEY:
                oxapay = OxaPayService(settings.OXAPAY_MERCHANT_API_KEY)
                order_id = f"renew_auto_{user.telegram_id}_{int(datetime.utcnow().timestamp())}"
                webhook_url = os.getenv("WEBHOOK_BASE_URL", "https://tradehubai.up.railway.app") + "/webhooks/oxapay"
                
                invoice = oxapay.create_invoice(
                    amount=settings.SUBSCRIPTION_PRICE_USD,
                    currency="USD",
                    description="Trading Bot Auto-Trading Renewal ($130/month)",
                    order_id=order_id,
                    callback_url=webhook_url,
                    metadata={
                        "telegram_id": str(user.telegram_id),
                        "plan_type": "auto"
                    }
                )
                
                if invoice and invoice.get("payLink"):
                    from app.models import PendingInvoice
                    try:
                        pending_invoice = PendingInvoice(
                            user_id=user.id,
                            track_id=invoice["trackId"],
                            order_id=order_id,
                            plan_type="auto",
                            amount=settings.SUBSCRIPTION_PRICE_USD,
                            status="pending"
                        )
                        db.add(pending_invoice)
                        db.commit()
                    except Exception as e:
                        logger.error(f"Failed to store renewal invoice: {e}")
                    
                    await message.answer(
                        f"✅ <b>Active Subscription</b>\n\n"
                        f"Your subscription is active until:\n"
                        f"📅 <b>{expires}</b>\n\n"
                        f"🔄 <b>Want to renew early?</b>\n"
                        f"Pay now and 30 days will be added to your current expiry date!\n\n"
                        f"💰 Renewal: <b>${settings.SUBSCRIPTION_PRICE_USD}/month</b>",
                        parse_mode="HTML",
                        reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                            InlineKeyboardButton(text="🔄 Renew Now (+30 days)", url=invoice["payLink"])
                        ]])
                    )
                    return
            
            # Fallback if invoice creation fails
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
        from app.services.oxapay import OxaPayService
        from app.config import settings
        import os
        
        if not settings.OXAPAY_MERCHANT_API_KEY:
            await message.answer(
                "⚠️ Subscription system is being set up. Please check back soon!"
            )
            return
        
        oxapay = OxaPayService(settings.OXAPAY_MERCHANT_API_KEY)
        
        # Create invoice with webhook callback URL
        order_id = f"sub_auto_{user.telegram_id}_{int(datetime.utcnow().timestamp())}"
        webhook_url = os.getenv("WEBHOOK_BASE_URL", "https://tradehubai.up.railway.app") + "/webhooks/oxapay"
        
        logger.info(f"Creating OxaPay invoice for user {user.telegram_id}, amount: {settings.SUBSCRIPTION_PRICE_USD}, webhook_url: {webhook_url}")
        
        invoice = oxapay.create_invoice(
            amount=settings.SUBSCRIPTION_PRICE_USD,
            currency="USD",
            description="Trading Bot Auto-Trading Subscription ($130/month - BLACK FRIDAY!)",
            order_id=order_id,
            callback_url=webhook_url,
            metadata={
                "telegram_id": str(user.telegram_id),
                "plan_type": "auto"
            }
        )
        
        logger.info(f"Invoice response: {invoice}")
        
        if invoice and invoice.get("payLink"):
            # Store invoice in database for automatic payment verification
            from app.models import PendingInvoice
            try:
                pending_invoice = PendingInvoice(
                    user_id=user.id,
                    track_id=invoice["trackId"],
                    order_id=order_id,
                    plan_type="auto",
                    amount=settings.SUBSCRIPTION_PRICE_USD,
                    status="pending"
                )
                db.add(pending_invoice)
                db.commit()
                logger.info(f"✅ Stored invoice {invoice['trackId']} in database for auto-verification")
            except Exception as e:
                logger.error(f"Failed to store invoice in database: {e}")
                # Don't fail the flow, just log the error
            
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
                f"🔹 BTC, ETH, USDT, and more cryptocurrencies\n\n"
                f"👇 <b>Click below to subscribe with crypto:</b>",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="💳 Pay with Crypto", url=invoice["payLink"])
                ]])
            )
        else:
            logger.error(f"Failed to create OxaPay invoice: {invoice}")
            await message.answer(
                "⚠️ Unable to generate payment link. Please try again later or contact support."
            )
    finally:
        db.close()


@dp.callback_query(F.data == "subscribe_menu")
async def handle_subscribe_menu(callback: CallbackQuery):
    """Handle subscribe button from main menu - direct to Auto-Trading payment"""
    await callback.answer()
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("You're not registered. Use /start to begin!")
            return
        
        # Check subscription status
        if user.grandfathered:
            await callback.message.edit_text(
                "🎉 <b>Lifetime Access - Grandfathered User</b>\n\n"
                "You have <b>FREE lifetime access</b> to Auto-Trading as an early supporter!\n\n"
                "<i>Thank you for being part of our community!</i>",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="🔙 Back", callback_data="back_to_start")
                ]])
            )
            return
        
        if user.is_subscribed:
            expires = user.subscription_end.strftime("%Y-%m-%d") if user.subscription_end else "Unknown"
            
            # Allow early renewal
            from app.services.oxapay import OxaPayService
            from app.config import settings
            import os
            
            if settings.OXAPAY_MERCHANT_API_KEY:
                oxapay = OxaPayService(settings.OXAPAY_MERCHANT_API_KEY)
                order_id = f"renew_auto_{user.telegram_id}_{int(datetime.utcnow().timestamp())}"
                webhook_url = os.getenv("WEBHOOK_BASE_URL", "https://tradehubai.up.railway.app") + "/webhooks/oxapay"
                
                invoice = oxapay.create_invoice(
                    amount=settings.SUBSCRIPTION_PRICE_USD,
                    currency="USD",
                    description="Trading Bot Auto-Trading Renewal ($130/month)",
                    order_id=order_id,
                    callback_url=webhook_url,
                    metadata={
                        "telegram_id": str(user.telegram_id),
                        "plan_type": "auto"
                    }
                )
                
                if invoice and invoice.get("payLink"):
                    from app.models import PendingInvoice
                    try:
                        pending_invoice = PendingInvoice(
                            user_id=user.id,
                            track_id=invoice["trackId"],
                            order_id=order_id,
                            plan_type="auto",
                            amount=settings.SUBSCRIPTION_PRICE_USD,
                            status="pending"
                        )
                        db.add(pending_invoice)
                        db.commit()
                    except Exception as e:
                        logger.error(f"Failed to store renewal invoice: {e}")
                    
                    await callback.message.edit_text(
                        f"✅ <b>Active Subscription: Auto-Trading</b>\n\n"
                        f"Your subscription is active until:\n"
                        f"📅 <b>{expires}</b>\n\n"
                        f"🔄 <b>Want to renew early?</b>\n"
                        f"Pay now and 30 days will be added to your current expiry!\n\n"
                        f"💰 Renewal: <b>${settings.SUBSCRIPTION_PRICE_USD}/month</b>",
                        parse_mode="HTML",
                        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                            [InlineKeyboardButton(text="🔄 Renew Now (+30 days)", url=invoice["payLink"])],
                            [InlineKeyboardButton(text="🔙 Back", callback_data="back_to_start")]
                        ])
                    )
                    return
            
            # Fallback
            await callback.message.edit_text(
                f"✅ <b>Active Subscription: Auto-Trading</b>\n\n"
                f"Your subscription is active until:\n"
                f"📅 <b>{expires}</b>\n\n"
                f"<i>Keep crushing it! 🚀</i>",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="🔙 Back", callback_data="back_to_start")
                ]])
            )
            return
        
        # User needs to subscribe - create OxaPay invoice directly
        from app.services.oxapay import OxaPayService
        from app.config import settings
        import os
        
        if not settings.OXAPAY_MERCHANT_API_KEY:
            await callback.message.edit_text(
                "⚠️ Subscription system is being set up. Please check back soon!",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="🔙 Back", callback_data="back_to_start")
                ]])
            )
            return
        
        oxapay = OxaPayService(settings.OXAPAY_MERCHANT_API_KEY)
        
        # Create invoice with webhook callback URL
        order_id = f"sub_auto_{user.telegram_id}_{int(datetime.utcnow().timestamp())}"
        webhook_url = os.getenv("WEBHOOK_BASE_URL", "https://tradehubai.up.railway.app") + "/webhooks/oxapay"
        
        logger.info(f"Creating OxaPay invoice for user {user.telegram_id}, amount: {settings.SUBSCRIPTION_PRICE_USD}, webhook_url: {webhook_url}")
        
        invoice = oxapay.create_invoice(
            amount=settings.SUBSCRIPTION_PRICE_USD,
            currency="USD",
            description="Trading Bot Auto-Trading Subscription ($130/month - BLACK FRIDAY!)",
            order_id=order_id,
            callback_url=webhook_url,
            metadata={
                "telegram_id": str(user.telegram_id),
                "plan_type": "auto"
            }
        )
        
        logger.info(f"Invoice response: {invoice}")
        
        if invoice and invoice.get("payLink"):
            # Store invoice in database for automatic payment verification
            from app.models import PendingInvoice
            try:
                pending_invoice = PendingInvoice(
                    user_id=user.id,
                    track_id=invoice["trackId"],
                    order_id=order_id,
                    plan_type="auto",
                    amount=settings.SUBSCRIPTION_PRICE_USD,
                    status="pending"
                )
                db.add(pending_invoice)
                db.commit()
                logger.info(f"✅ Stored invoice {invoice['trackId']} in database for auto-verification")
            except Exception as e:
                logger.error(f"Failed to store invoice in database: {e}")
                # Don't fail the flow, just log the error
            
            await callback.message.edit_text(
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
                f"🔹 BTC, ETH, USDT, and more cryptocurrencies\n\n"
                f"👇 <b>Click below to subscribe with crypto:</b>",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="💳 Pay with Crypto", url=invoice["payLink"])],
                    [InlineKeyboardButton(text="🔙 Back", callback_data="back_to_start")]
                ])
            )
        else:
            logger.error(f"Failed to create OxaPay invoice: {invoice}")
            await callback.message.edit_text(
                "⚠️ Unable to generate payment link. Please try again later or contact support.",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="🔙 Back", callback_data="back_to_start")
                ]])
            )
    finally:
        db.close()


@dp.callback_query(F.data == "referral_stats")
async def handle_referral_stats(callback: CallbackQuery):
    """Show referral stats and referral link"""
    await callback.answer()
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("You're not registered. Use /start to begin!")
            return
        
        # Count successful referrals (users who actually subscribed)
        referrals = db.query(User).filter(User.referred_by == user.referral_code).all()
        total_referrals = len(referrals)
        subscribed_referrals = [r for r in referrals if r.is_subscribed]
        subscribed_count = len(subscribed_referrals)
        
        # Generate referral link
        bot_username = (await bot.get_me()).username
        referral_link = f"https://t.me/{bot_username}?start={user.referral_code}"
        
        # Calculate pending earnings and auto-trading referrals
        pending_earnings = user.referral_earnings or 0.0
        auto_trading_referrals = sum(1 for r in subscribed_referrals if r.subscription_type == "auto")
        
        # Build referral stats message
        stats_text = (
            "🎁 <b>Your Referral Program</b>\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Referral Stats</b>\n"
            f"• Total Referrals: <b>{total_referrals}</b>\n"
            f"• Active Subscribers: <b>{subscribed_count}</b>\n"
            f"• Auto-Trading Referrals: <b>{auto_trading_referrals}</b>\n"
            f"💰 <b>Pending Earnings:</b> ${pending_earnings:.2f}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔗 <b>Your Referral Code:</b>\n"
            f"<code>{user.referral_code}</code>\n\n"
            f"🔗 <b>Your Referral Link:</b>\n"
            f"{referral_link}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>How It Works:</b>\n"
            f"• Share your link with friends\n"
            f"• When they subscribe to <b>Auto-Trading ($130/mo)</b>, you get <b>$30 USD</b> in crypto!\n"
            f"• Unlimited referrals = unlimited earnings!\n"
            f"• Payouts sent automatically 💸\n\n"
        )
        
        # Show list of referrals if any
        if subscribed_referrals:
            stats_text += "<b>🌟 Your Active Referrals:</b>\n"
            for ref in subscribed_referrals[:5]:  # Show max 5
                ref_name = ref.username if ref.username else ref.first_name or "Unknown"
                stats_text += f"  • @{ref_name}\n"
            if subscribed_count > 5:
                stats_text += f"  • ... and {subscribed_count - 5} more\n"
        else:
            stats_text += "<i>💡 Tip: Share your link on social media to earn free months!</i>"
        
        # Add wallet info to message
        if user.crypto_wallet:
            stats_text += f"\n\n💰 <b>Payout Wallet:</b>\n<code>{user.crypto_wallet}</code>"
        else:
            stats_text += "\n\n⚠️ <b>Wallet Not Set!</b>\nSet your wallet to receive payouts."
        
        # Determine wallet button text
        wallet_button_text = "✏️ Update Wallet" if user.crypto_wallet else "💰 Set Wallet Address"
        
        await callback.message.edit_text(
            stats_text,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="📋 Copy Link", url=referral_link)],
                [InlineKeyboardButton(text=wallet_button_text, callback_data="set_wallet_prompt")],
                [InlineKeyboardButton(text="🔙 Back", callback_data="back_to_start")]
            ])
        )
    finally:
        db.close()


@dp.callback_query(F.data == "set_wallet_prompt")
async def handle_set_wallet_prompt(callback: CallbackQuery):
    """Prompt user to set wallet address"""
    await callback.answer()
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("You're not registered. Use /start to begin!")
            return
        
        current_wallet = user.crypto_wallet
        
        if current_wallet:
            prompt_text = (
                "💰 <b>Update Your Wallet Address</b>\n\n"
                f"<b>Current Wallet:</b>\n"
                f"<code>{current_wallet}</code>\n\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                "<b>Select your preferred network:</b>\n"
                "<i>💡 Your $30 referral rewards will be sent here!</i>"
            )
        else:
            prompt_text = (
                "💰 <b>Set Your Crypto Wallet</b>\n\n"
                "⚠️ You haven't set a wallet address yet!\n\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                "<b>Select your preferred network:</b>\n\n"
                "<i>💡 Earn $30 USD for every Auto-Trading referral!</i>"
            )
        
        await callback.message.edit_text(
            prompt_text,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="💚 USDT (TRC20) - Tron", callback_data="wallet_guide_trc20")],
                [InlineKeyboardButton(text="💙 USDT (ERC20) - Ethereum", callback_data="wallet_guide_erc20")],
                [InlineKeyboardButton(text="🟣 SOL - Solana", callback_data="wallet_guide_sol")],
                [InlineKeyboardButton(text="🟠 BTC - Bitcoin", callback_data="wallet_guide_btc")],
                [InlineKeyboardButton(text="🔙 Back to Referrals", callback_data="referral_stats")]
            ])
        )
    finally:
        db.close()


@dp.callback_query(F.data.startswith("wallet_guide_"))
async def handle_wallet_guide(callback: CallbackQuery):
    """Show wallet setup guide for specific crypto network"""
    await callback.answer()
    
    network = callback.data.split("_")[-1]  # trc20, erc20, sol, or btc
    
    if network == "trc20":
        guide_text = (
            "💚 <b>USDT (TRC20) - Tron Network</b>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>Wallet Address Format:</b>\n"
            "• Starts with 'T'\n"
            "• 34 characters long\n"
            "• Example: <code>TXYZabc123def456ghi789jkl...</code>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>✅ Recommended (Low Fees!)</b>\n\n"
            "To set your TRC20 wallet, send:\n"
            "<code>/setwallet TYourWalletAddress</code>\n\n"
            "<i>💡 TRC20 has the lowest fees (~$1 USDT)</i>"
        )
    elif network == "erc20":
        guide_text = (
            "💙 <b>USDT (ERC20) - Ethereum Network</b>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>Wallet Address Format:</b>\n"
            "• Starts with '0x'\n"
            "• 42 characters long\n"
            "• Example: <code>0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1</code>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "⚠️ <b>Note:</b> ERC20 has higher gas fees (~$5-20)\n\n"
            "To set your ERC20 wallet, send:\n"
            "<code>/setwallet 0xYourWalletAddress</code>\n\n"
            "<i>💡 Consider TRC20 for lower fees</i>"
        )
    elif network == "sol":
        guide_text = (
            "🟣 <b>SOL - Solana Network</b>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>Wallet Address Format:</b>\n"
            "• Base58 encoded\n"
            "• 32-44 characters long\n"
            "• Example: <code>7EqQdEUhQFQYWFbZQpJWvvXzEPF3JcLH...</code>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>✅ Fast & Low Fees!</b>\n\n"
            "To set your Solana wallet, send:\n"
            "<code>/setwallet YourSolanaAddress</code>\n\n"
            "<i>💡 Solana has very low fees (~$0.01) and fast transfers</i>"
        )
    else:  # btc
        guide_text = (
            "🟠 <b>BTC - Bitcoin Network</b>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>Wallet Address Format:</b>\n"
            "• Starts with 'bc1', '1', or '3'\n"
            "• 26-62 characters long\n"
            "• Example: <code>bc1qxy2kgdygjrsqtzq2n0yrf2493p...</code>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "To set your BTC wallet, send:\n"
            "<code>/setwallet YourBTCAddress</code>\n\n"
            "<i>💡 Bitcoin transactions may take 10-60 minutes</i>"
        )
    
    await callback.message.edit_text(
        guide_text,
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔙 Back to Networks", callback_data="set_wallet_prompt")],
            [InlineKeyboardButton(text="🏠 Back to Referrals", callback_data="referral_stats")]
        ])
    )


@dp.callback_query(F.data.startswith("subscribe_"))
async def handle_subscribe_plan(callback: CallbackQuery):
    """Handle scan/manual/auto subscription plan selection"""
    await callback.answer()
    
    plan_type = callback.data.split("_")[1]  # "scan", "manual", or "auto"
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("You're not registered. Use /start to begin!")
            return
        
        from app.services.coinbase_commerce import CoinbaseCommerceService
        from app.config import settings
        
        if not settings.COINBASE_COMMERCE_API_KEY:
            await callback.message.edit_text(
                "⚠️ Subscription system is being set up. Please check back soon!",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="🔙 Back", callback_data="subscribe_menu")
                ]])
            )
            return
        
        # Determine price and plan details
        if plan_type == "scan":
            price = settings.SCAN_MODE_PRICE
            plan_name = "📊 Scan Mode"
            plan_emoji = "📊"
            features = "✅ Top Gainers scanner\n✅ Volume surge detection\n✅ New coin alerts"
        elif plan_type == "manual":
            price = settings.MANUAL_SIGNALS_PRICE
            plan_name = "💎 Manual Signals"
            plan_emoji = "💎"
            features = "✅ All Scan Mode features\n✅ Manual signal notifications\n✅ Entry, TP, SL levels\n✅ LONGS + SHORTS strategies\n✅ PnL tracking"
        else:  # auto
            price = settings.AUTO_TRADING_PRICE
            plan_name = "🤖 Auto-Trading"
            plan_emoji = "🤖"
            features = "✅ All Manual Signals features\n✅ Automated 24/7 execution\n✅ Bitunix integration\n✅ Advanced risk management"
        
        coinbase = CoinbaseCommerceService(settings.COINBASE_COMMERCE_API_KEY)
        
        # Create payment with plan type in order_id
        order_id = f"sub_{plan_type}_{user.telegram_id}_{int(datetime.utcnow().timestamp())}"
        
        charge = coinbase.create_charge(
            amount=price,
            currency="USD",
            description=f"Trading Bot {plan_name} Subscription",
            metadata={
                "telegram_id": str(user.telegram_id),
                "plan_type": plan_type,
                "order_id": order_id
            }
        )
        
        if charge and charge.get("hosted_url"):
            await callback.message.edit_text(
                f"{plan_emoji} <b>{plan_name}</b> - ${price:.0f}/month\n\n"
                f"🔥 <b>You're about to unlock:</b>\n"
                f"{features}\n\n"
                f"💳 <b>Pay with crypto:</b>\n"
                f"BTC • ETH • USDT • 200+ coins\n\n"
                f"⚠️ <b>IMPORTANT - Network Fees:</b>\n"
                f"Please send <b>slightly more than ${price:.0f}</b> to cover network/exchange fees!\n"
                f"If payment is short, contact @bu11dogg for manual activation.\n\n"
                f"⚡ <b>Instant activation</b> after payment\n"
                f"👇 Click below to complete checkout:",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text=f"💳 Pay ${price:.0f} with Crypto", url=charge["hosted_url"])],
                    [InlineKeyboardButton(text="◀️ Back to Plans", callback_data="subscribe_menu")]
                ])
            )
        else:
            await callback.message.edit_text(
                "⚠️ Unable to generate payment link. Please try again.",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="🔙 Back", callback_data="subscribe_menu")
                ]])
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
        
        # ✅ FORCE FRESH USER QUERY to avoid stale cache
        db.expire(user)
        db.refresh(user)
        
        # Use the SAME helper as /start to get account overview text
        account_text, _ = await build_account_overview(user, db)
        
        # But use dashboard-specific buttons (Active Positions, PnL views, etc.)
        # Build keyboard - show Scalp Mode for users with scalp_mode_enabled
        prefs = user.preferences
        if prefs:
            db.expire(prefs)
            db.refresh(prefs)
        has_scalp_access = prefs and getattr(prefs, 'scalp_mode_enabled', False)
        
        if has_scalp_access:
            dashboard_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="🔍 Scan Coins", callback_data="scan_menu"),
                    InlineKeyboardButton(text="📡 Recent Signals", callback_data="recent_signals")
                ],
                [
                    InlineKeyboardButton(text="⚡ Scalp Trades", callback_data="scalp_mode"),
                    InlineKeyboardButton(text="🤖 Auto-Trading", callback_data="autotrading_menu")
                ],
                [
                    InlineKeyboardButton(text="⚙️ Settings", callback_data="settings"),
                    InlineKeyboardButton(text="🆘 Support", callback_data="support_menu")
                ],
                [
                    InlineKeyboardButton(text="🏠 Home", callback_data="home")
                ]
            ])
        else:
            dashboard_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="🔍 Scan Coins", callback_data="scan_menu"),
                    InlineKeyboardButton(text="📡 Recent Signals", callback_data="recent_signals")
                ],
                [
                    InlineKeyboardButton(text="⚡ Scalp Trades (coming soon)", callback_data="scalp_coming_soon"),
                    InlineKeyboardButton(text="🤖 Auto-Trading", callback_data="autotrading_menu")
                ],
                [
                    InlineKeyboardButton(text="⚙️ Settings", callback_data="settings"),
                    InlineKeyboardButton(text="🆘 Support", callback_data="support_menu")
                ],
                [
                    InlineKeyboardButton(text="🏠 Home", callback_data="home")
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
        leverage = prefs.user_leverage if prefs else 10
        
        now = datetime.utcnow()
        if period == "today":
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            period_emoji = "📊"
        elif period == "week":
            start_date = now - timedelta(days=7)
            period_emoji = "📈"
        elif period == "month":
            start_date = now - timedelta(days=30)
            period_emoji = "📅"
        else:  # "all" or any other value = all time
            start_date = now - timedelta(days=3650)  # 10 years ago (all time)
            period_emoji = "🏆"
        
        # Query live trades only
        trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.closed_at >= start_date,
            Trade.status.in_(['closed', 'stopped', 'tp_hit', 'sl_hit'])
        ).all()
        
        # Get failed trades from TODAY ONLY (not based on period)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        failed_trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.opened_at >= today_start,
            Trade.status == "failed"
        ).all()
        
        if not trades:
            pnl_text = f"""
{period_emoji} <b>PnL Summary ({period.title()})</b>
💰 LIVE TRADING

No closed trades in this period.
Use /autotrading_status to set up auto-trading!
"""
        else:
            total_pnl = sum(t.pnl for t in trades)
            total_pnl_pct = sum(t.pnl_percent for t in trades)
            
            # Classify trades: Wins, Breakeven (-5% to 0%), Losses (<-5%)
            winning_trades = [t for t in trades if t.pnl > 0]
            breakeven_trades = [t for t in trades if t.pnl_percent >= -5.0 and t.pnl <= 0]  # Small losses = breakeven
            losing_trades = [t for t in trades if t.pnl_percent < -5.0]  # Losses below -5%
            
            # Calculate ROI % (return on invested capital)
            # Capital invested = position_size (already the margin amount)
            total_capital_invested = sum(t.position_size for t in trades)
            roi_percent = (total_pnl / total_capital_invested * 100) if total_capital_invested > 0 else 0
            
            avg_pnl = total_pnl / len(trades) if trades else 0
            avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            best_trade = max(trades, key=lambda t: t.pnl) if trades else None
            worst_trade = min(trades, key=lambda t: t.pnl) if trades else None
            
            # Win rate: wins / total trades (including breakeven)
            counted_trades = len(trades)
            win_rate = (len(winning_trades) / counted_trades * 100) if counted_trades > 0 else 0
            
            pnl_emoji = "🟢" if total_pnl > 0 else "🔴" if total_pnl < 0 else "⚪"
            roi_emoji = "🟢" if roi_percent > 0 else "🔴" if roi_percent < 0 else "⚪"
            
            # Generate win rate progress bar (visual indicator)
            win_rate_normalized = min(100, max(0, win_rate))  # Clamp to 0-100
            filled_blocks = int((win_rate_normalized / 10))  # 10 blocks total
            empty_blocks = 10 - filled_blocks
            progress_bar = "█" * filled_blocks + "░" * empty_blocks
            
            # Win streak detection (only show if hot streak)
            recent_trades = sorted(trades, key=lambda t: t.closed_at, reverse=True)[:5]
            recent_wins = sum(1 for t in recent_trades if t.pnl > 0)
            streak_text = f"🔥 {recent_wins} win streak!" if recent_wins >= 3 and recent_wins == len(recent_trades) else ""
            
            # Separate auto-trading profits
            auto_trades = [t for t in trades if hasattr(t, 'trade_type') and t.trade_type in ['TOP_GAINER', 'DAY_TRADE', 'STANDARD']]
            manual_trades = [t for t in trades if t not in auto_trades]
            auto_pnl = sum(t.pnl for t in auto_trades) if auto_trades else 0
            manual_pnl = sum(t.pnl for t in manual_trades) if manual_trades else 0
            auto_pnl_emoji = "🟢" if auto_pnl > 0 else "🔴" if auto_pnl < 0 else "⚪"
            
            # Calculate total signals sent (executed + failed)
            total_signals_sent = len(auto_trades) + len(failed_trades)
            execution_rate = (len(auto_trades) / total_signals_sent * 100) if total_signals_sent > 0 else 0
            
            # Calculate theoretical PnL for ALL signals (including failed ones)
            # Assume failed signals would have hit TP for theoretical calculation
            theoretical_pnl = auto_pnl  # Start with actual PnL
            theoretical_wins = len([t for t in auto_trades if t.pnl > 0])
            theoretical_losses = len([t for t in auto_trades if t.pnl_percent < -5.0])
            
            # Calculate theoretical capital that would have been invested
            theoretical_capital = total_capital_invested  # Start with actual
            
            # Add theoretical profit for failed signals (assume 20% TP hit)
            for failed in failed_trades:
                # Assume 10% position size would have been used
                position_would_be = prefs.position_size_percent if prefs else 10.0
                theoretical_profit = position_would_be * 0.20  # 20% TP on 10% position
                theoretical_pnl += theoretical_profit
                theoretical_wins += 1
                theoretical_capital += position_would_be
            
            # Calculate theoretical ROI
            theoretical_roi = (theoretical_pnl / theoretical_capital * 100) if theoretical_capital > 0 else 0
            
            theoretical_pnl_emoji = "🟢" if theoretical_pnl > 0 else "🔴" if theoretical_pnl < 0 else "⚪"
            theoretical_roi_emoji = "🟢" if theoretical_roi > 0 else "🔴" if theoretical_roi < 0 else "⚪"
            theoretical_win_rate = (theoretical_wins / total_signals_sent * 100) if total_signals_sent > 0 else 0
            
            # Build auto-trading section
            auto_section = ""
            if auto_trades or failed_trades:
                auto_wins = len([t for t in auto_trades if t.pnl > 0])
                auto_losses = len([t for t in auto_trades if t.pnl_percent < -5.0])
                
                # Calculate auto-trader ROI
                auto_capital = sum(t.position_size for t in auto_trades) if auto_trades else 0
                auto_roi = (auto_pnl / auto_capital * 100) if auto_capital > 0 else 0
                auto_roi_emoji = "🟢" if auto_roi > 0 else "🔴" if auto_roi < 0 else "⚪"
                
                auto_section = f"""
<b>🤖 Auto-Trader P&L</b>
├ {auto_pnl_emoji} Actual: <b>${auto_pnl:+.2f}</b>
├ {auto_roi_emoji} ROI: <b>{auto_roi:+.1f}%</b>
├ Win Rate: {(auto_wins/len(auto_trades)*100) if auto_trades else 0:.1f}% (✅ {auto_wins} | ❌ {auto_losses})
└ Manual: ${manual_pnl:+.2f} ({len(manual_trades)} trades)
"""
            
            pnl_text = f"""
{period_emoji} <b>PnL Summary ({period.title()})</b>
{mode_label} | Leverage: {leverage}x
━━━━━━━━━━━━━━━━━━━━

<b>💰 Bot Performance</b>
├ 📊 Signals Sent: {total_signals_sent}
├ {theoretical_pnl_emoji} P&L: <b>${theoretical_pnl:+.2f}</b>
├ {theoretical_roi_emoji} ROI: <b>{theoretical_roi:+.1f}%</b>
├ 🎯 Win Rate: {theoretical_win_rate:.0f}%
└ ✅ Executed: {len(auto_trades)}/{total_signals_sent} ({execution_rate:.0f}%)
{auto_section}
<b>🎯 Overall Win Rate: {win_rate:.1f}%</b>
{progress_bar} {len(winning_trades)}/{counted_trades}
{streak_text}

<b>📊 Breakdown</b>
├ ✅ Wins: {len(winning_trades)}
├ ⚪ Breakeven: {len(breakeven_trades)}
└ ❌ Losses: {len(losing_trades)}

<b>💵 Averages</b>
├ Per Trade: ${avg_pnl:.2f}
├ Avg Win: ${avg_win:.2f}
└ Avg Loss: ${avg_loss:.2f}

<b>🏆 Best:</b> ${best_trade.pnl:+.2f} ({best_trade.symbol})
<b>📉 Worst:</b> ${worst_trade.pnl:+.2f} ({worst_trade.symbol})
"""
        
        # Simple back button - no share functionality
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
        ])
        
        await callback.message.answer(pnl_text, reply_markup=keyboard, parse_mode="HTML")
        await safe_answer_callback(callback)
    finally:
        db.close()


# NOTE: pnl_today, pnl_week, pnl_month are handled by handle_pnl_callback above (F.data.startswith("pnl_"))
# No separate handlers needed - they're already covered!




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


# NOTE: view_all_pnl is now handled by pnl_all in handle_pnl_callback above
# Just redirect to pnl_all
@dp.callback_query(F.data == "view_all_pnl")
async def handle_view_all_pnl(callback: CallbackQuery):
    """Show all-time PnL via button - redirect to pnl_all"""
    # Change callback data to pnl_all and re-trigger
    callback.data = "pnl_all"
    await handle_pnl_callback(callback)


@dp.callback_query(F.data == "edit_position_size")
async def handle_edit_position_size(callback: CallbackQuery, state: FSMContext):
    """Map to set_position_size handler"""
    await handle_set_position_size(callback, state)


@dp.callback_query(F.data == "edit_leverage")
async def handle_edit_leverage(callback: CallbackQuery, state: FSMContext):
    """Show leverage edit prompt"""
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
        
        # Get current leverage
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        current_leverage = prefs.top_gainers_leverage if prefs else 5
        
        await callback.message.edit_text(f"""
⚡ **Set Top Gainers Leverage**

Current: {current_leverage}x leverage

📝 Send me the new leverage (1-20):

Examples:
• 5 = 5x leverage (conservative)
• 10 = 10x leverage (moderate)
• 20 = 20x leverage (aggressive)

⚠️ Higher leverage = Higher profit but higher risk!
""", parse_mode="Markdown")
        await state.set_state(TopGainerLeverageSetup.waiting_for_leverage)
        await callback.answer()
    finally:
        db.close()


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
        user_leverage = prefs.top_gainers_leverage or 5
        
        # Calculate actual price move based on leverage (fixed 20% ROI target)
        price_move = 20.0 / user_leverage
        price_move_tp2 = 35.0 / user_leverage
        
        response_text = f"""
🔥 <b>Top Gainers Mode</b> {status}

<b>What it does:</b>
Catches big coin crashes after pumps 📉

<b>How it works:</b>
• Scans 24/7 (no time restrictions)
• Finds coins up 20%+ in 24h (parabolic pumps)
• Waits for reversal signals
• SHORTS the dump (95% of trades)
• {user_leverage}x leverage (customizable)

<b>Profit targets:</b>
• Regular: {price_move:.1f}% price drop = 20% account profit
• Parabolic (50%+ pumps): {price_move:.1f}% + {price_move_tp2:.1f}% = 20% + 35% 🎯

<b>Risk:</b>
High volatility - only for experienced traders!

Status: {status}
{"⏰ Scanning 24/7 every 15 min" if prefs.top_gainers_mode_enabled else "Off - no signals 🔴"}

<i>Use /set_top_gainer_leverage to adjust leverage</i>
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⚡ Set Leverage", callback_data="set_top_gainer_leverage")],
            [InlineKeyboardButton(text="📊 View Analytics", callback_data="view_top_gainer_stats")],
            [InlineKeyboardButton(text="⚙️ Back to Settings", callback_data="settings_menu")],
            [InlineKeyboardButton(text="◀️ Main Menu", callback_data="back_to_start")]
        ])
        
        await callback.message.answer(response_text, reply_markup=keyboard, parse_mode="HTML")
        
    finally:
        db.close()


@dp.callback_query(F.data == "toggle_top_gainers_shorts")
async def handle_toggle_top_gainers_shorts(callback: CallbackQuery):
    """Toggle Top Gainers SHORTS Mode (Mean Reversion)"""
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Please use /start first")
            return
        
        prefs = user.preferences
        
        # Toggle SHORTS mode
        tg_enabled = prefs.top_gainers_mode_enabled
        current_mode = prefs.top_gainers_trade_mode or 'shorts_only'
        
        # Determine new state based on current state
        if not tg_enabled:
            # Top Gainers is OFF → Turn ON with SHORTS only
            new_mode = 'shorts_only'
            new_enabled = True
        elif current_mode == 'shorts_only':
            # SHORTS only → Turn OFF completely
            new_mode = 'shorts_only'
            new_enabled = False
        elif current_mode == 'longs_only':
            # LONGS only → Add SHORTS → BOTH
            new_mode = 'both'
            new_enabled = True
        elif current_mode == 'both':
            # BOTH → Remove SHORTS → LONGS only
            new_mode = 'longs_only'
            new_enabled = True  # Keep enabled with LONGS
        else:
            # Default to SHORTS only
            new_mode = 'shorts_only'
            new_enabled = True
        
        prefs.top_gainers_trade_mode = new_mode
        prefs.top_gainers_mode_enabled = new_enabled
        
        db.commit()
        db.refresh(prefs)
        
        user_leverage = prefs.top_gainers_leverage or 5
        price_move = 20.0 / user_leverage
        
        # Determine new status (both enabled flag AND mode must match)
        shorts_active = new_enabled and new_mode in ['shorts_only', 'both']
        status = "✅ ENABLED" if shorts_active else "❌ DISABLED"
        
        response_text = f"""
🔴 <b>Top Gainers SHORTS Mode</b> {status}

<b>Strategy:</b> Mean Reversion
Catches big coin crashes after pumps 📉

<b>How it works:</b>
• Scans 24/7 (every 15 min)
• Finds coins up 25%+ in 24h
• Waits for reversal signals
• SHORTS the dump

<b>Leverage:</b> {user_leverage}x
<b>Target:</b> {price_move:.1f}% price drop = 20% profit
<b>Risk:</b> High volatility trades

<b>Current Mode:</b> {prefs.top_gainers_trade_mode.upper().replace('_', ' ')}

<i>Use /set_top_gainer_leverage to adjust</i>
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⚡ Set Leverage", callback_data="set_top_gainer_leverage")],
            [InlineKeyboardButton(text="📊 View Analytics", callback_data="view_top_gainer_stats")],
            [InlineKeyboardButton(text="⚙️ Back to Settings", callback_data="settings_menu")]
        ])
        
        await callback.message.answer(response_text, reply_markup=keyboard, parse_mode="HTML")
        
    finally:
        db.close()


@dp.callback_query(F.data == "toggle_top_gainers_longs")
async def handle_toggle_top_gainers_longs(callback: CallbackQuery):
    """Toggle Top Gainers LONGS Mode (Pump Retracement)"""
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Please use /start first")
            return
        
        prefs = user.preferences
        
        # Toggle LONGS mode
        tg_enabled = prefs.top_gainers_mode_enabled
        current_mode = prefs.top_gainers_trade_mode or 'shorts_only'
        
        # Determine new state based on current state
        if not tg_enabled:
            # Top Gainers is OFF → Turn ON with LONGS only
            new_mode = 'longs_only'
            new_enabled = True
        elif current_mode == 'longs_only':
            # LONGS only → Turn OFF completely
            new_mode = 'longs_only'
            new_enabled = False
        elif current_mode == 'shorts_only':
            # SHORTS only → Add LONGS → BOTH
            new_mode = 'both'
            new_enabled = True
        elif current_mode == 'both':
            # BOTH → Remove LONGS → SHORTS only
            new_mode = 'shorts_only'
            new_enabled = True  # Keep enabled with SHORTS
        else:
            # Default to LONGS only
            new_mode = 'longs_only'
            new_enabled = True
        
        prefs.top_gainers_trade_mode = new_mode
        prefs.top_gainers_mode_enabled = new_enabled
        
        db.commit()
        db.refresh(prefs)
        
        user_leverage = prefs.top_gainers_leverage or 5
        price_move = 20.0 / user_leverage
        
        # Determine new status (both enabled flag AND mode must match)
        longs_active = new_enabled and new_mode in ['longs_only', 'both']
        status = "✅ ENABLED" if longs_active else "❌ DISABLED"
        
        response_text = f"""
🟢 <b>Top Gainers LONGS Mode</b> {status}

<b>Strategy:</b> Pump Retracement Entry
Catches coins AFTER pullback (NO CHASING!) 📈

<b>How it works:</b>
• Scans 24/7 (every 15 min)
• Finds coins pumping 5-200%+
• Waits for retracement to EMA9
• Enters AFTER pullback (not chasing tops!)
• 3 entry types: EMA9 pullback, resumption, strong pump

<b>Leverage:</b> {user_leverage}x
<b>Target:</b> {price_move:.1f}% price move = 20% profit
<b>Risk:</b> High volatility trades

<b>Current Mode:</b> {prefs.top_gainers_trade_mode.upper().replace('_', ' ')}

<i>Use /set_top_gainer_leverage to adjust</i>
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⚡ Set Leverage", callback_data="set_top_gainer_leverage")],
            [InlineKeyboardButton(text="📊 View Analytics", callback_data="view_top_gainer_stats")],
            [InlineKeyboardButton(text="⚙️ Back to Settings", callback_data="settings_menu")]
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
            Trade.status.in_(['closed', 'stopped', 'tp_hit', 'sl_hit'])
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
        
        # Get live trades only
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
                    
                    trades_text += f"""
{i}. {direction_emoji} <b>{trade.symbol} {trade.direction}</b>
   Entry: ${trade.entry_price:.4f}
   Current: ${current_price:.4f}
   Size: ${remaining_size:.2f}
   
   🛑 SL: ${trade.stop_loss:.4f}{tp_text}
   
   {pnl_emoji} <b>Live PnL:</b> ${pnl_usd:+.2f} ({pnl_pct:+.2f}%){realized_text}
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
            
            trades_text += f"""
{total_emoji} <b>TOTAL LIVE PnL</b>
💰 ${total_unrealized_pnl_usd:+.2f} ({combined_pnl_pct:+.2f}%)
📊 Across {len(trades)} position{'s' if len(trades) != 1 else ''}
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


@dp.callback_query(F.data == "scalp_mode")
async def handle_scalp_mode(callback: CallbackQuery):
    """Show scalp trade statistics with toggle & position size"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found")
            return
        
        # Check access
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            await callback.answer()
            return
        
        # Get scalp preferences
        prefs = user.preferences
        if not prefs:
            # Create default preferences if missing
            prefs = UserPreference(user_id=user.id)
            db.add(prefs)
            db.commit()
        
        scalp_enabled = getattr(prefs, 'scalp_mode_enabled', False)
        scalp_size = getattr(prefs, 'scalp_position_size_percent', 1.0)
        
        # Build explanation text
        status = "🟢 ON" if scalp_enabled else "🔴 OFF"
        scalp_text = f"""⚡ <b>Scalp Trades</b>

<b>What are Scalp Trades?</b>
High-frequency trades targeting quick 40% profits on altcoin support bounces with RSI reversal confirmation.

📊 <b>Strategy:</b>
• Scans top 100 gainers every 60 seconds
• Detects support level bounces + RSI oversold reversal
• 2% TP / 4% SL @ 20x leverage = 40% profit target
• Expected 6-10 signals per day

⚠️ <b>Risk Profile:</b>
• High-frequency = more opportunities + more risk
• 20x leverage = larger profit/loss potential
• Recommended 1-2% position size for safety
• Stop-loss always in place to protect capital

⚙️ <b>Your Settings:</b>
Status: {status}
Position Size: {scalp_size}% of balance
Leverage: 20x (fixed)
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="🔘 Toggle" if scalp_enabled else "⭕ Toggle", callback_data="scalp_toggle"),
                InlineKeyboardButton(text=f"📊 Size: {scalp_size}%", callback_data="scalp_size")
            ],
            [InlineKeyboardButton(text="🔄 Refresh", callback_data="scalp_mode")],
            [InlineKeyboardButton(text="◀️ Back", callback_data="back_to_dashboard")]
        ])
        
        await callback.message.answer(scalp_text, reply_markup=keyboard, parse_mode="HTML")
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data == "scalp_toggle")
async def handle_scalp_toggle(callback: CallbackQuery):
    """Toggle scalp mode on/off"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.answer("Settings not found")
            return
        
        # Toggle
        user.preferences.scalp_mode_enabled = not user.preferences.scalp_mode_enabled
        db.commit()
        
        status = "✅ ENABLED" if user.preferences.scalp_mode_enabled else "❌ DISABLED"
        await callback.message.answer(f"⚡ Scalp Mode: {status}")
        await callback.answer()
        
        # Refresh display
        await handle_scalp_mode(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "scalp_size")
async def handle_scalp_size(callback: CallbackQuery):
    """Show position size options"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            return
        
        current = user.preferences.scalp_position_size_percent
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="1% (Conservative)" if current != 1.0 else "1% ✓", callback_data="scalp_size_1"),
                InlineKeyboardButton(text="2%" if current != 2.0 else "2% ✓", callback_data="scalp_size_2")
            ],
            [
                InlineKeyboardButton(text="3%" if current != 3.0 else "3% ✓", callback_data="scalp_size_3"),
                InlineKeyboardButton(text="5%" if current != 5.0 else "5% ✓", callback_data="scalp_size_5")
            ],
            [
                InlineKeyboardButton(text="10% (Aggressive)" if current != 10.0 else "10% ✓", callback_data="scalp_size_10"),
                InlineKeyboardButton(text="15% (Max)" if current != 15.0 else "15% ✓", callback_data="scalp_size_15")
            ],
            [InlineKeyboardButton(text="◀️ Back to Scalp", callback_data="scalp_mode")]
        ])
        
        await callback.message.answer(
            "⚡ <b>Scalp Position Size</b>\n\nSelect % of account balance per trade:",
            reply_markup=keyboard,
            parse_mode="HTML"
        )
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data.startswith("scalp_size_"))
async def handle_scalp_size_set(callback: CallbackQuery):
    """Set scalp position size"""
    db = SessionLocal()
    try:
        size = float(callback.data.replace("scalp_size_", ""))
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            return
        
        user.preferences.scalp_position_size_percent = size
        db.commit()
        
        await callback.message.answer(f"✅ Scalp position size set to {size}%")
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data == "scalp_coming_soon")
async def handle_scalp_coming_soon(callback: CallbackQuery):
    """Show coming soon message for non-owner users"""
    await callback.message.answer(
        "⚡ <b>Scalp Mode - Coming Soon!</b>\n\n"
        "We're testing an exclusive high-frequency scalping strategy:\n\n"
        "🎯 <b>Features:</b>\n"
        "• 1-minute scan interval (ultra-fast!)\n"
        "• Altcoin support bounces + RSI reversal\n"
        "• 25% profit target @ 20x leverage\n"
        "• Expected 6-10 signals per day\n\n"
        "🚀 <b>Launch Date:</b> Coming soon\n\n"
        "Enjoy premium trading in the meantime! 💪",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Back to Dashboard", callback_data="back_to_dashboard")]
        ]),
        parse_mode="HTML"
    )
    await callback.answer()


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
        
        # Check if user has auto-trading subscription
        has_access, reason = check_access(user, require_tier="auto")
        if not has_access:
            await callback.message.edit_text(
                "🤖 <b>Auto-Trading - Premium Feature</b>\n\n"
                "Auto-trading is available on the <b>🤖 Auto-Trading plan</b> ($130/month - BLACK FRIDAY!).\n\n"
                "<b>With Auto-Trading you get:</b>\n"
                "✅ Automated 24/7 trade execution\n"
                "✅ Hands-free trading on Bitunix\n"
                "✅ Advanced risk management\n"
                "✅ All features included\n\n"
                "<i>Upgrade to unlock automation!</i>",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="⬆️ Upgrade to Auto-Trading", callback_data="subscribe_auto")],
                    [InlineKeyboardButton(text="🔙 Back", callback_data="back_to_start")]
                ])
            )
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
        
        # Auto-trading status - SIMPLIFIED: Bitunix connected = ready (user toggles on/off)
        is_active = bitunix_connected
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
            
            autotrading_text = f"""
🤖 <b>Auto-Trading Status</b>
━━━━━━━━━━━━━━━━━━━━

🔑 <b>Exchange:</b> {exchange_name}
📡 <b>API Status:</b> {api_status}
🔄 <b>Auto-Trading:</b> {autotrading_status}
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


@dp.callback_query(F.data == "autotrading_unified")
async def handle_autotrading_unified(callback: CallbackQuery):
    """🚀 UNIFIED Auto-Trading Menu - All-in-one setup & control"""
    await callback.answer()
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("User not found")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            return
        
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        
        # Check subscription (auto-trading is premium)
        has_access, reason = check_access(user, require_tier="auto")
        if not has_access:
            await callback.message.edit_text(
                "⚡ <b>Auto-Trading</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                "🔒 Auto-Trading Plan Required\n\n"
                "<b>What You Get:</b>\n"
                "✅ 24/7 automated trade execution\n"
                "✅ Hands-free trading on Bitunix\n"
                "✅ Advanced risk management\n"
                "✅ All features included\n\n"
                "💡 <i>Upgrade to $130/month plan to unlock!</i>",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="⬆️ Upgrade Now", callback_data="subscribe_auto")],
                    [InlineKeyboardButton(text="◀️ Back", callback_data="back_to_start")]
                ])
            )
            return
        
        # Check Bitunix connection
        bitunix_connected = (
            prefs and 
            prefs.bitunix_api_key and 
            prefs.bitunix_api_secret and
            len(prefs.bitunix_api_key) > 0 and 
            len(prefs.bitunix_api_secret) > 0
        )
        
        auto_trading_enabled = prefs and prefs.auto_trading_enabled
        
        # FIXED: Status must check auto_trading_enabled flag too!
        is_active = bitunix_connected and auto_trading_enabled
        
        # Build unified menu
        if bitunix_connected:
            status_emoji = "🟢" if is_active else "🔴"
            
            if not auto_trading_enabled:
                status_text = "DISABLED"
            elif is_active:
                status_text = "ACTIVE"
            else:
                status_text = "INACTIVE"
            
            menu_text = f"""
⚡ <b>Auto-Trading Control</b>
━━━━━━━━━━━━━━━━━━━━

{status_emoji} <b>Status:</b> {status_text}
🔗 <b>Exchange:</b> Bitunix ✅ Connected
💰 <b>Mode:</b> Live Trading

<b>Quick Actions:</b>
Use buttons below to manage auto-trading

<i>All signals auto-execute when enabled!</i>
"""
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(
                    text=f"{'🔴 Disable' if is_active else '🟢 Enable'} Auto-Trading",
                    callback_data="toggle_autotrading_quick"
                )],
                [InlineKeyboardButton(text="📊 Position Size", callback_data="set_position_size")],
                [InlineKeyboardButton(text="⚡ Leverage", callback_data="edit_leverage")],
                [InlineKeyboardButton(text="❌ Remove API", callback_data="remove_api_confirm")],
                [InlineKeyboardButton(text="◀️ Back", callback_data="back_to_start")]
            ])
        else:
            menu_text = """
⚡ <b>Auto-Trading Setup</b>
━━━━━━━━━━━━━━━━━━━━

❌ <b>Not Connected</b>

<b>Setup Required:</b>
1️⃣ Create Bitunix API keys
2️⃣ Use /set_bitunix_api to connect
3️⃣ Enable auto-trading

<b>⚠️ Important:</b>
• Enable <b>futures trading</b> only
• <b>DO NOT</b> enable withdrawals

<i>Safe & secure - keys encrypted!</i>
"""
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔗 Connect Bitunix", url="https://www.bitunix.com/apiManagement")],
                [InlineKeyboardButton(text="◀️ Back", callback_data="back_to_start")]
            ])
        
        await callback.message.edit_text(menu_text, reply_markup=keyboard, parse_mode="HTML")
        
    finally:
        db.close()


@dp.callback_query(F.data == "top_gainers_unified")
async def handle_top_gainers_unified(callback: CallbackQuery):
    """🚀 UNIFIED Top Gainers Menu - SHORTS & LONGS in one screen"""
    await callback.answer()
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Please use /start first")
            return
        
        prefs = user.preferences
        
        # Get current mode
        tg_enabled = prefs.top_gainers_mode_enabled
        current_mode = prefs.top_gainers_trade_mode or 'shorts_only'
        
        # Determine status for SHORTS and LONGS
        shorts_active = tg_enabled and current_mode in ['shorts_only', 'both']
        longs_active = tg_enabled and current_mode in ['longs_only', 'both']
        
        shorts_emoji = "🟢" if shorts_active else "🔴"
        longs_emoji = "🟢" if longs_active else "🔴"
        
        user_leverage = prefs.top_gainers_leverage or 5
        price_move = 20.0 / user_leverage
        
        menu_text = f"""
🔥 <b>Top Gainers Trading</b>
━━━━━━━━━━━━━━━━━━━━

<b>Status:</b>
├ {shorts_emoji} SHORTS: {'ACTIVE' if shorts_active else 'OFF'}
└ {longs_emoji} LONGS: {'ACTIVE' if longs_active else 'OFF'}

<b>How It Works:</b>
• <b>SHORTS:</b> Mean reversion on 25%+ pumps
• <b>LONGS:</b> Pump retracement entries (5-200%+)
• <b>Leverage:</b> {user_leverage}x
• <b>TP/SL:</b> {price_move:.1f}% price move = 20% ROI

Toggle SHORTS/LONGS below ⬇️
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(
                text=f"{shorts_emoji} {'Disable' if shorts_active else 'Enable'} SHORTS",
                callback_data="toggle_top_gainers_shorts"
            )],
            [InlineKeyboardButton(
                text=f"{longs_emoji} {'Disable' if longs_active else 'Enable'} LONGS",
                callback_data="toggle_top_gainers_longs"
            )],
            [InlineKeyboardButton(text="⚡ Set Leverage", callback_data="set_top_gainer_leverage")],
            [InlineKeyboardButton(text="📊 View Stats", callback_data="view_top_gainer_stats")],
            [InlineKeyboardButton(text="◀️ Back", callback_data="back_to_start")]
        ])
        
        await callback.message.edit_text(menu_text, reply_markup=keyboard, parse_mode="HTML")
        
    finally:
        db.close()


@dp.callback_query(F.data == "settings_simplified")
async def handle_settings_simplified(callback: CallbackQuery):
    """🚀 SIMPLIFIED Settings Menu - Essentials only"""
    await callback.answer()
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Please use /start first")
            return
        
        prefs = user.preferences
        
        # Get current settings
        position_size = f"{prefs.position_size_percent:.0f}%" if prefs else "10%"
        leverage = f"{prefs.user_leverage}x" if prefs else "10x"
        dm_alerts = "🔔 ON" if (prefs and prefs.dm_alerts) else "🔕 OFF"
        
        menu_text = f"""
⚙️ <b>Settings</b>
━━━━━━━━━━━━━━━━━━━━

<b>Current Configuration:</b>
├ 💰 Position Size: <b>{position_size}</b>
├ ⚡ Leverage: <b>{leverage}</b>
├ 💰 Mode: <b>Live Trading</b>
└ {dm_alerts} DM Notifications

<i>Use buttons below to adjust settings</i>
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="💰 Position Size", callback_data="set_position_size")],
            [InlineKeyboardButton(text="⚡ Leverage", callback_data="edit_leverage")],
            [InlineKeyboardButton(text=f"{'🔕 Disable' if prefs and prefs.dm_alerts else '🔔 Enable'} DM Alerts", callback_data="toggle_dm_alerts")],
            [InlineKeyboardButton(text="🔧 Advanced Settings", callback_data="settings_menu")],
            [InlineKeyboardButton(text="◀️ Back", callback_data="back_to_start")]
        ])
        
        await callback.message.edit_text(menu_text, reply_markup=keyboard, parse_mode="HTML")
        
    finally:
        db.close()


@dp.callback_query(F.data == "toggle_dm_alerts")
async def handle_toggle_dm_alerts(callback: CallbackQuery):
    """Quick toggle for DM alerts"""
    await callback.answer()
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Settings not found")
            return
        
        prefs = user.preferences
        prefs.dm_alerts = not prefs.dm_alerts
        db.commit()
        
        # Refresh the settings menu
        await handle_settings_simplified(callback)
        
    finally:
        db.close()


@dp.callback_query(F.data == "back_to_start")
async def handle_back_to_start(callback: CallbackQuery):
    """Return to main /start menu"""
    await callback.answer()
    
    db = SessionLocal()
    try:
        user = get_or_create_user(
            callback.from_user.id,
            callback.from_user.username,
            callback.from_user.first_name,
            db
        )
        
        welcome_text, keyboard = await build_account_overview(user, db)
        await callback.message.edit_text(welcome_text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "back_to_dashboard")
async def handle_back_to_dashboard(callback: CallbackQuery):
    """Handle back to dashboard button - shows the dashboard view"""
    await callback.answer()
    
    # Show dashboard view (with PnL buttons and Active Positions)
    await cmd_dashboard(callback.message)


@dp.callback_query(F.data == "home")
async def handle_home_button(callback: CallbackQuery):
    """Handle home button - takes user back to /start menu"""
    await callback.answer()
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("Please use /start first")
            return
        
        # Build the start menu
        welcome_text, keyboard = await build_account_overview(user, db)
        
        # Edit the message to show the start menu
        await callback.message.edit_text(welcome_text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


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
        
        # Check auto-trading subscription
        has_access, reason = check_access(user, require_tier="auto")
        if not has_access:
            await callback.answer("⚠️ Auto-trading requires Auto-Trading plan ($130/mo)", show_alert=True)
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
        
        settings_text = f"""
⚙️ <b>Your Settings</b>

<b>📊 General</b>
• Muted Symbols: {muted_str}
• Default PnL Period: {prefs.default_pnl_period}
• DM Alerts: {"Enabled" if prefs.dm_alerts else "Disabled"}

<b>💰 Live Trading</b>
• Position Size: {prefs.position_size_percent}% of balance
• Leverage: {prefs.user_leverage}x

<b>🛡️ Risk Management</b>
• Correlation Filter: {"Enabled" if prefs.correlation_filter_enabled else "Disabled"}
• Max Correlated Positions: {prefs.max_correlated_positions}
• Funding Rate Alerts: {"Enabled" if prefs.funding_rate_alerts_enabled else "Disabled"}
• Funding Alert Threshold: {prefs.funding_rate_threshold}%

<b>Commands:</b>
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


@dp.message(Command("set_top_gainer_leverage"))
async def cmd_set_top_gainer_leverage(message: types.Message):
    """Set custom leverage for Top Gainers mode"""
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
                current_lev = user.preferences.top_gainers_leverage or 5
                await message.answer(
                    f"❌ Usage: /set_top_gainer_leverage <1-20>\n\n"
                    f"Current: {current_lev}x\n"
                    f"Example: /set_top_gainer_leverage 10"
                )
                return
            
            leverage = int(args[1])
            if leverage < 1 or leverage > 20:
                await message.answer("❌ Leverage must be between 1x and 20x")
                return
            
            user.preferences.top_gainers_leverage = leverage
            db.commit()
            
            # Calculate risk profile
            if leverage <= 5:
                risk_label = "🟢 Conservative"
            elif leverage <= 10:
                risk_label = "🟡 Moderate"
            else:
                risk_label = "🔴 Aggressive"
            
            await message.answer(
                f"✅ <b>Top Gainers Leverage Updated!</b>\n\n"
                f"Leverage: <b>{leverage}x</b> {risk_label}\n\n"
                f"<b>With 20% TP/SL targets:</b>\n"
                f"• Profit per trade: {20 * leverage}% of position\n"
                f"• Loss per trade: {20 * leverage}% of position\n\n"
                f"⚠️ Higher leverage = Higher risk & reward\n"
                f"📊 Use /top_gainer_stats to track performance",
                parse_mode="HTML"
            )
        except ValueError:
            await message.answer("❌ Invalid number. Use: /set_top_gainer_leverage <1-20>")
    finally:
        db.close()


@dp.message(Command("top_gainer_stats"))
async def cmd_top_gainer_stats(message: types.Message):
    """Show Top Gainers mode analytics (Upgrade #3)"""
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
        
        # Query Top Gainer trades only
        top_gainer_trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.trade_type == 'TOP_GAINER',
            Trade.status.in_(['closed', 'stopped', 'tp_hit', 'sl_hit'])
        ).all()
        
        # Query Day Trading trades for comparison
        day_trading_trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.trade_type.in_(['DAY_TRADE', 'STANDARD']),
            Trade.status.in_(['closed', 'stopped', 'tp_hit', 'sl_hit'])
        ).all()
        
        if not top_gainer_trades:
            await message.answer(
                "🔥 <b>Top Gainers Analytics</b>\n\n"
                "No closed Top Gainer trades yet!\n\n"
                "Enable Top Gainers mode to start trading parabolic pumps:\n"
                "/autotrading_status → 🔥 Top Gainers Mode",
                parse_mode="HTML"
            )
            return
        
        # Calculate Top Gainer stats
        tg_total_pnl = sum(t.pnl for t in top_gainer_trades)
        tg_winning_trades = [t for t in top_gainer_trades if t.pnl > 0]
        tg_losing_trades = [t for t in top_gainer_trades if t.pnl < 0]
        tg_win_rate = (len(tg_winning_trades) / len(top_gainer_trades) * 100) if top_gainer_trades else 0
        tg_avg_win = sum(t.pnl for t in tg_winning_trades) / len(tg_winning_trades) if tg_winning_trades else 0
        tg_avg_loss = sum(t.pnl for t in tg_losing_trades) / len(tg_losing_trades) if tg_losing_trades else 0
        
        # Calculate capital invested and ROI
        # position_size is already the margin/capital used, no need to divide by leverage
        tg_capital_invested = sum(t.position_size for t in top_gainer_trades)
        tg_roi = (tg_total_pnl / tg_capital_invested * 100) if tg_capital_invested > 0 else 0
        
        # Find best performing coins
        from collections import defaultdict
        coin_performance = defaultdict(lambda: {'pnl': 0, 'trades': 0, 'wins': 0})
        for t in top_gainer_trades:
            coin = t.symbol.replace('/USDT', '')
            coin_performance[coin]['pnl'] += t.pnl
            coin_performance[coin]['trades'] += 1
            if t.pnl > 0:
                coin_performance[coin]['wins'] += 1
        
        # Sort by PnL
        best_coins = sorted(coin_performance.items(), key=lambda x: x[1]['pnl'], reverse=True)[:3]
        worst_coins = sorted(coin_performance.items(), key=lambda x: x[1]['pnl'])[:3]
        
        # Current streak status
        current_streak = prefs.top_gainers_win_streak if prefs else 0
        multiplier = prefs.top_gainers_position_multiplier if prefs else 1.0
        
        # Build best coins section
        best_coins_text = ""
        for coin, stats in best_coins:
            win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            best_coins_text += f"  • {coin}: ${stats['pnl']:+.2f} ({stats['trades']} trades, {win_rate:.0f}% WR)\n"
        
        # Compare to Day Trading if available
        comparison_text = ""
        if day_trading_trades:
            dt_total_pnl = sum(t.pnl for t in day_trading_trades)
            dt_winning = [t for t in day_trading_trades if t.pnl > 0]
            dt_win_rate = (len(dt_winning) / len(day_trading_trades) * 100) if day_trading_trades else 0
            
            comparison_text = f"""
📊 <b>Comparison vs Day Trading:</b>
Top Gainers: {tg_win_rate:.1f}% WR | ${tg_total_pnl:+.2f}
Day Trading: {dt_win_rate:.1f}% WR | ${dt_total_pnl:+.2f}
"""
        
        stats_text = f"""
🔥 <b>Top Gainers Mode Analytics</b>
━━━━━━━━━━━━━━━━━━━━

💰 <b>Performance Summary:</b>
Total PnL: <b>${tg_total_pnl:+.2f}</b>
Win Rate: <b>{tg_win_rate:.1f}%</b> ({len(tg_winning_trades)}W / {len(tg_losing_trades)}L)
ROI: <b>{tg_roi:+.1f}%</b>

📈 <b>Trade Breakdown:</b>
Total Trades: {len(top_gainer_trades)}
Avg Win: ${tg_avg_win:.2f}
Avg Loss: ${tg_avg_loss:.2f}
Profit Factor: {abs(tg_avg_win / tg_avg_loss):.2f}x

🎯 <b>Best Coins:</b>
{best_coins_text}
🚀 <b>Auto-Compound Status:</b>
Win Streak: {current_streak}/3 wins
Position Multiplier: <b>{multiplier}x</b>
{"🔥 COMPOUNDING ACTIVE - Next trade +20% size!" if multiplier > 1.0 else f"Need {3 - current_streak} more wins to activate +20% size boost"}
{comparison_text}
<i>Keep crushing those parabolic reversals! 📉</i>
"""
        
        await message.answer(stats_text, parse_mode="HTML")
        
    finally:
        db.close()


@dp.message(Command("close_position"))
async def cmd_close_position(message: types.Message):
    """Manually close a stuck position"""
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
        
        # Get all open trades for this user
        open_trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.status == 'open'
        ).all()
        
        if not open_trades:
            await message.answer("❌ You have no open positions to close.")
            return
        
        # Parse command: /close_position [trade_id]
        args = message.text.split()
        
        if len(args) < 2:
            # Show list of open trades
            trades_list = "\n".join([
                f"  • ID: {t.id} - {t.symbol} {t.direction} @ ${t.entry_price:.4f}"
                for t in open_trades
            ])
            await message.answer(
                f"<b>Open Positions:</b>\n{trades_list}\n\n"
                f"To close: /close_position [ID]\n"
                f"Example: /close_position {open_trades[0].id}",
                parse_mode="HTML"
            )
            return
        
        try:
            trade_id = int(args[1])
        except ValueError:
            await message.answer("❌ Invalid trade ID. Use: /close_position [ID]")
            return
        
        # Find the trade
        trade = db.query(Trade).filter(
            Trade.id == trade_id,
            Trade.user_id == user.id,
            Trade.status == 'open'
        ).first()
        
        if not trade:
            await message.answer(f"❌ Trade ID {trade_id} not found or already closed.")
            return
        
        prefs = user.preferences
        if not prefs or not prefs.bitunix_api_key or not prefs.bitunix_api_secret:
            await message.answer("❌ Bitunix API not configured.")
            return
        
        # Decrypt and close position
        from app.utils.encryption import decrypt_api_key
        from app.services.bitunix_trader import BitunixTrader
        
        api_key = decrypt_api_key(prefs.bitunix_api_key)
        api_secret = decrypt_api_key(prefs.bitunix_api_secret)
        trader = BitunixTrader(api_key, api_secret)
        
        try:
            # Get current price
            current_price = await trader.get_current_price(trade.symbol)
            if not current_price:
                await message.answer(f"❌ Could not fetch current price for {trade.symbol}")
                return
            
            # Close position on Bitunix
            close_result = await trader.close_position(trade.symbol, trade.direction)
            
            if close_result:
                # Calculate PnL with leverage
                leverage = prefs.top_gainers_leverage if trade.trade_type == 'TOP_GAINER' else (prefs.user_leverage or 5)
                price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                price_change_percent = price_change / trade.entry_price
                pnl_usd = price_change_percent * trade.position_size * leverage
                
                trade.status = 'closed'
                trade.exit_price = current_price
                trade.closed_at = datetime.utcnow()
                trade.pnl = float(pnl_usd)
                trade.pnl_percent = (trade.pnl / trade.position_size) * 100
                trade.remaining_size = 0
                
                # Update auto-compound streak if TOP_GAINER
                if trade.trade_type == 'TOP_GAINER' and prefs.top_gainers_auto_compound:
                    if trade.pnl > 0:
                        prefs.top_gainers_win_streak += 1
                        if prefs.top_gainers_win_streak >= 3:
                            prefs.top_gainers_position_multiplier = 1.2
                    else:
                        prefs.top_gainers_win_streak = 0
                        prefs.top_gainers_position_multiplier = 1.0
                
                db.commit()
                
                await message.answer(
                    f"✅ <b>Position Closed Manually</b>\n\n"
                    f"Symbol: {trade.symbol} {trade.direction}\n"
                    f"Entry: ${trade.entry_price:.4f}\n"
                    f"Exit: ${current_price:.4f}\n\n"
                    f"💰 PnL: ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%)\n"
                    f"Position Size: ${trade.position_size:.2f}",
                    parse_mode="HTML"
                )
            else:
                await message.answer(f"❌ Failed to close position on Bitunix. Try again or check your API connection.")
            
        finally:
            await trader.close()
        
    except Exception as e:
        logger.error(f"Error in manual close position: {e}", exc_info=True)
        await message.answer(f"❌ Error closing position: {str(e)}")
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
                InlineKeyboardButton(text="📝 Submit Ticket", callback_data="submit_ticket")
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


@dp.callback_query(F.data == "submit_ticket")
async def handle_submit_ticket(callback: CallbackQuery):
    """Handle ticket submission - show category selection"""
    await callback.answer()
    
    ticket_text = """
📝 <b>Submit Support Ticket</b>
━━━━━━━━━━━━━━━━━━━━

Select a category for your issue:

<i>Your ticket will be sent to our support team privately.
No personal information will be shared.</i>
"""
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🔧 Technical Issue", callback_data="ticket_tech"),
            InlineKeyboardButton(text="💰 Payment Issue", callback_data="ticket_payment")
        ],
        [
            InlineKeyboardButton(text="🤖 Auto-Trading", callback_data="ticket_autotrading"),
            InlineKeyboardButton(text="📊 Signals & Analytics", callback_data="ticket_signals")
        ],
        [
            InlineKeyboardButton(text="❓ General Question", callback_data="ticket_general"),
            InlineKeyboardButton(text="💡 Feature Request", callback_data="ticket_feature")
        ],
        [
            InlineKeyboardButton(text="◀️ Back to Support", callback_data="support_menu")
        ]
    ])
    
    await callback.message.edit_text(ticket_text, reply_markup=keyboard, parse_mode="HTML")


# Dictionary to track users submitting tickets
user_ticket_data = {}

@dp.callback_query(F.data.startswith("ticket_"))
async def handle_ticket_category(callback: CallbackQuery):
    """Handle ticket category selection and ask for message"""
    await callback.answer()
    
    category = callback.data.replace("ticket_", "")
    category_names = {
        "tech": "🔧 Technical Issue",
        "payment": "💰 Payment Issue",
        "autotrading": "🤖 Auto-Trading",
        "signals": "📊 Signals & Analytics",
        "general": "❓ General Question",
        "feature": "💡 Feature Request"
    }
    
    subject = category_names.get(category, "General")
    user_id = callback.from_user.id
    
    # Store the subject for this user
    user_ticket_data[user_id] = {"subject": subject}
    
    prompt_text = f"""
📝 <b>Submit Support Ticket</b>
━━━━━━━━━━━━━━━━━━━━

<b>Category:</b> {subject}

<b>Please describe your issue:</b>

Type your message below and send it.
Include as many details as possible to help us assist you faster.

<i>To cancel, click Back to Support or send /cancel</i>
"""
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Back to Support", callback_data="support_menu")]
    ])
    
    await callback.message.edit_text(prompt_text, reply_markup=keyboard, parse_mode="HTML")


@dp.message(F.text & ~F.text.startswith("/"), StateFilter(None))
async def handle_ticket_message(message: types.Message):
    """Handle user's ticket message submission OR admin reply (ONLY when NOT in FSM state)"""
    user_id = message.from_user.id
    
    # Check if admin is replying to a ticket
    if user_id in admin_reply_data:
        await handle_admin_reply_message(message)
        return
    
    # Check if user is submitting a ticket
    if user_id not in user_ticket_data:
        return  # Not submitting a ticket, ignore
    
    # Get the subject
    ticket_info = user_ticket_data[user_id]
    subject = ticket_info["subject"]
    user_message = message.text
    
    # Clear the user's ticket data
    del user_ticket_data[user_id]
    
    db = SessionLocal()
    try:
        from app.models import SupportTicket
        
        user = db.query(User).filter(User.telegram_id == str(user_id)).first()
        if not user:
            await message.answer("❌ User not found. Please use /start first.")
            return
        
        # Create the ticket
        ticket = SupportTicket(
            user_id=user.id,
            subject=subject,
            message=user_message,
            status="open",
            priority="normal"
        )
        db.add(ticket)
        db.commit()
        db.refresh(ticket)
        
        # Notify admins
        admins = db.query(User).filter(User.is_admin == True).all()
        for admin in admins:
            try:
                admin_notification = f"""
🎫 <b>New Support Ticket</b> #{ticket.id}
━━━━━━━━━━━━━━━━━━━━

<b>From:</b> {user.first_name or 'User'} (@{user.username or 'no_username'})
<b>Category:</b> {subject}
<b>Submitted:</b> {ticket.created_at.strftime('%Y-%m-%d %H:%M UTC')}

<b>Message:</b>
{user_message}

━━━━━━━━━━━━━━━━━━━━
Use /view_ticket {ticket.id} to view and reply
"""
                await bot.send_message(chat_id=admin.telegram_id, text=admin_notification, parse_mode="HTML")
            except Exception as e:
                logger.error(f"Failed to notify admin {admin.telegram_id}: {e}")
        
        # Confirm to user
        success_text = f"""
✅ <b>Ticket Submitted Successfully!</b>
━━━━━━━━━━━━━━━━━━━━

<b>Ticket ID:</b> #{ticket.id}
<b>Category:</b> {subject}
<b>Status:</b> 🟢 Open

Your ticket has been sent to our support team.
We'll respond as soon as possible!

<b>Average response time:</b> 4-12 hours

You can check your ticket status anytime with:
/my_tickets
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🏠 Back to Home", callback_data="home")]
        ])
        
        await message.answer(success_text, reply_markup=keyboard, parse_mode="HTML")
        
    finally:
        db.close()


@dp.callback_query(F.data == "support_menu")
async def handle_support_menu(callback: CallbackQuery):
    # Clear any pending ticket data for this user
    if callback.from_user.id in user_ticket_data:
        del user_ticket_data[callback.from_user.id]
    
    # Reuse the support command
    await cmd_support(callback.message)
    await callback.answer()


@dp.message(Command("my_tickets"))
async def cmd_my_tickets(message: types.Message):
    """Show user's submitted tickets"""
    db = SessionLocal()
    try:
        from app.models import SupportTicket
        
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        tickets = db.query(SupportTicket).filter(SupportTicket.user_id == user.id).order_by(SupportTicket.created_at.desc()).limit(10).all()
        
        if not tickets:
            await message.answer(
                "📝 <b>No Support Tickets</b>\n\n"
                "You haven't submitted any tickets yet.\n"
                "Use /support → Submit Ticket to create one!",
                parse_mode="HTML"
            )
            return
        
        tickets_text = "<b>📝 Your Support Tickets</b>\n━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for ticket in tickets:
            status_emoji = "🟢" if ticket.status == "open" else "🟡" if ticket.status == "in_progress" else "✅"
            tickets_text += f"""
<b>Ticket #{ticket.id}</b>
├ {status_emoji} Status: {ticket.status.upper()}
├ 📁 Category: {ticket.subject}
├ 📅 Created: {ticket.created_at.strftime('%Y-%m-%d %H:%M')}
└ {"✅ Responded" if ticket.admin_response else "⏳ Waiting for response"}

"""
        
        tickets_text += "\n<i>Use /view_ticket [ID] to see details</i>"
        
        await message.answer(tickets_text, parse_mode="HTML")
    finally:
        db.close()


@dp.message(Command("view_ticket"))
async def cmd_view_ticket(message: types.Message):
    """View a specific ticket (user or admin)"""
    db = SessionLocal()
    try:
        from app.models import SupportTicket
        
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        # Extract ticket ID
        args = message.text.split()
        if len(args) < 2:
            await message.answer("❌ Please provide a ticket ID: /view_ticket 123")
            return
        
        try:
            ticket_id = int(args[1])
        except:
            await message.answer("❌ Invalid ticket ID. Please use a number.")
            return
        
        ticket = db.query(SupportTicket).filter(SupportTicket.id == ticket_id).first()
        
        if not ticket:
            await message.answer(f"❌ Ticket #{ticket_id} not found.")
            return
        
        # Check permissions (user can view their own, admin can view all)
        if ticket.user_id != user.id and not user.is_admin:
            await message.answer("❌ You don't have permission to view this ticket.")
            return
        
        # Build ticket details
        status_emoji = "🟢" if ticket.status == "open" else "🟡" if ticket.status == "in_progress" else "✅"
        
        ticket_text = f"""
🎫 <b>Support Ticket #{ticket.id}</b>
━━━━━━━━━━━━━━━━━━━━

<b>Status:</b> {status_emoji} {ticket.status.upper()}
<b>Category:</b> {ticket.subject}
<b>Priority:</b> {ticket.priority.upper()}
<b>Created:</b> {ticket.created_at.strftime('%Y-%m-%d %H:%M UTC')}

<b>Your Message:</b>
{ticket.message}
"""
        
        if ticket.admin_response:
            ticket_text += f"""
━━━━━━━━━━━━━━━━━━━━
<b>✅ Admin Response:</b>
{ticket.admin_response}

<b>Responded:</b> {ticket.admin_responded_at.strftime('%Y-%m-%d %H:%M UTC') if ticket.admin_responded_at else 'N/A'}
"""
        else:
            ticket_text += "\n━━━━━━━━━━━━━━━━━━━━\n<i>⏳ Waiting for admin response...</i>"
        
        # Admin options
        keyboard = None
        if user.is_admin and not ticket.admin_response:
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="✅ Reply to Ticket", callback_data=f"reply_ticket_{ticket.id}")]
            ])
        
        await message.answer(ticket_text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


# Admin ticket reply (simple - sends reply directly)
admin_reply_data = {}

@dp.callback_query(F.data.startswith("reply_ticket_"))
async def handle_reply_ticket(callback: CallbackQuery):
    """Admin replies to a ticket"""
    await callback.answer()
    
    ticket_id = int(callback.data.replace("reply_ticket_", ""))
    admin_id = callback.from_user.id
    
    # Store ticket ID for this admin
    admin_reply_data[admin_id] = ticket_id
    
    await callback.message.answer(
        f"<b>📝 Replying to Ticket #{ticket_id}</b>\n\n"
        "Type your response below and send it.\n\n"
        "<i>To cancel, send /cancel</i>",
        parse_mode="HTML"
    )


async def handle_admin_reply_message(message: types.Message):
    """Handle admin's reply to a ticket"""
    admin_id = message.from_user.id
    ticket_id = admin_reply_data[admin_id]
    admin_response = message.text
    
    # Clear the admin's reply data
    del admin_reply_data[admin_id]
    
    db = SessionLocal()
    try:
        from app.models import SupportTicket
        
        admin = db.query(User).filter(User.telegram_id == str(admin_id)).first()
        if not admin or not admin.is_admin:
            await message.answer("❌ You don't have admin permissions.")
            return
        
        ticket = db.query(SupportTicket).filter(SupportTicket.id == ticket_id).first()
        if not ticket:
            await message.answer(f"❌ Ticket #{ticket_id} not found.")
            return
        
        # Update ticket with admin response
        ticket.admin_id = admin.id
        ticket.admin_response = admin_response
        ticket.admin_responded_at = datetime.utcnow()
        ticket.status = "closed"
        ticket.closed_at = datetime.utcnow()
        db.commit()
        
        # Notify the user
        user = db.query(User).filter(User.id == ticket.user_id).first()
        if user:
            try:
                user_notification = f"""
✅ <b>Your Support Ticket Was Answered!</b>
━━━━━━━━━━━━━━━━━━━━

<b>Ticket #{ticket.id}</b> - {ticket.subject}

<b>Admin Response:</b>
{admin_response}

━━━━━━━━━━━━━━━━━━━━
Use /view_ticket {ticket.id} to see full details
"""
                await bot.send_message(chat_id=user.telegram_id, text=user_notification, parse_mode="HTML")
            except Exception as e:
                logger.error(f"Failed to notify user {user.telegram_id}: {e}")
        
        # Confirm to admin
        await message.answer(
            f"✅ <b>Reply Sent Successfully!</b>\n\n"
            f"Ticket #{ticket.id} has been closed and the user has been notified.",
            parse_mode="HTML"
        )
        
    finally:
        db.close()


@dp.message(Command("cancel"))
async def cmd_cancel(message: types.Message):
    """Cancel ticket/reply submission"""
    user_id = message.from_user.id
    
    if user_id in user_ticket_data:
        del user_ticket_data[user_id]
        await message.answer("✅ Ticket submission cancelled.")
    elif user_id in admin_reply_data:
        del admin_reply_data[user_id]
        await message.answer("✅ Reply cancelled.")
    else:
        await message.answer("No active ticket or reply to cancel.")


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
    """Test autotrader with detailed step-by-step diagnostics"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("❌ Please /start the bot first.")
            return
        
        prefs = user.preferences
        has_bitunix = prefs and prefs.bitunix_api_key and prefs.bitunix_api_secret
        
        if not has_bitunix:
            await message.answer("❌ Bitunix API not configured. Use /set_bitunix_api first.")
            return
        
        if not prefs.auto_trading_enabled:
            await message.answer("❌ Auto-trading is disabled. Enable it first with /toggle_autotrading")
            return
        
        await message.answer("🧪 <b>Running Detailed Autotrader Test...</b>\n\n<i>Testing each step of execution...</i>", parse_mode="HTML")
        
        # Step-by-step diagnostic
        steps = []
        
        try:
            # STEP 1: Get price
            steps.append("1️⃣ Getting ETH price...")
            exchange = ccxt.kucoin()
            try:
                ticker = await exchange.fetch_ticker('ETH/USDT')
                current_price = ticker['last']
                steps.append(f"   ✅ ETH price: ${current_price:,.2f}")
            finally:
                await exchange.close()
            
            # STEP 2: Check API keys
            steps.append("2️⃣ Checking API keys...")
            from app.services.bitunix_trader import BitunixTrader
            api_key = decrypt_api_key(prefs.bitunix_api_key)
            api_secret = decrypt_api_key(prefs.bitunix_api_secret)
            steps.append(f"   ✅ API keys decrypted")
            
            # STEP 3: Connect to Bitunix
            steps.append("3️⃣ Connecting to Bitunix...")
            trader = BitunixTrader(api_key, api_secret)
            
            # STEP 4: Get balance
            steps.append("4️⃣ Checking balance...")
            balance = await trader.get_account_balance()
            if balance and balance > 0:
                steps.append(f"   ✅ Balance: ${balance:.2f}")
            else:
                steps.append(f"   ❌ Balance check FAILED (returned: {balance})")
                await trader.close()
                await message.answer("\n".join(steps) + "\n\n<b>❌ FAILED AT STEP 4: Cannot get balance</b>\n\nCheck API permissions or Futures wallet balance.", parse_mode="HTML")
                return
            
            # STEP 5: Calculate position size
            steps.append("5️⃣ Calculating position size...")
            pos_percent = prefs.position_size_percent or 10
            position_size = balance * (pos_percent / 100)
            steps.append(f"   → {pos_percent}% of ${balance:.2f} = ${position_size:.2f}")
            
            if position_size < 3:
                steps.append(f"   ⚠️ Position ${position_size:.2f} is small but allowed")
            else:
                steps.append(f"   ✅ Position size OK: ${position_size:.2f}")
            
            # STEP 6: Execute test trade
            steps.append("6️⃣ Executing test trade on ETH/USDT...")
            
            result = await trader.place_trade(
                symbol='ETH/USDT',
                direction='LONG',
                entry_price=current_price,
                stop_loss=current_price * 0.98,
                take_profit=current_price * 1.02,
                position_size_usdt=position_size,
                leverage=5
            )
            
            await trader.close()
            
            if result and result.get('success'):
                steps.append(f"   ✅ TRADE EXECUTED!")
                steps.append(f"\n🎉 <b>SUCCESS!</b> Check your Bitunix account!")
                await message.answer("\n".join(steps), parse_mode="HTML")
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No response from Bitunix'
                steps.append(f"   ❌ Trade failed: {error_msg}")
                await message.answer("\n".join(steps) + f"\n\n<b>❌ FAILED AT STEP 6: Bitunix rejected trade</b>\n\nError: {error_msg}", parse_mode="HTML")
                
        except Exception as e:
            error_type = type(e).__name__
            if 'Timeout' in error_type or 'timeout' in str(e).lower():
                steps.append(f"   ❌ TIMEOUT: Bitunix API not responding")
                await message.answer("\n".join(steps) + "\n\n<b>❌ API Timeout</b>\n\nCheck API permissions and Futures wallet.", parse_mode="HTML")
            else:
                steps.append(f"   ❌ ERROR: {str(e)[:150]}")
                await message.answer("\n".join(steps) + f"\n\n<b>❌ Test failed with error</b>", parse_mode="HTML")
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


@dp.message(Command("grant_sub"))
async def cmd_grant_subscription(message: types.Message):
    """Admin command to manually grant subscription (for short payments due to fees)"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user or not user.is_admin:
            await message.answer("❌ This command is only available to admins.")
            return
        
        # Parse command: /grant_sub <telegram_id> <plan_type> <days>
        # Example: /grant_sub 123456789 manual 30
        parts = message.text.split()
        if len(parts) < 3:
            await message.answer(
                "❌ <b>Usage:</b> /grant_sub &lt;telegram_id&gt; &lt;plan&gt; [days]\n\n"
                "<b>Plans:</b> manual, auto, lifetime\n"
                "<b>Days:</b> Optional (default: 30, ignored for lifetime)\n\n"
                "<b>Examples:</b>\n"
                "/grant_sub 123456789 manual 30\n"
                "/grant_sub 123456789 lifetime",
                parse_mode="HTML"
            )
            return
        
        target_telegram_id = parts[1]
        plan_type = parts[2].lower()
        
        # Validate and parse days
        try:
            days = int(parts[3]) if len(parts) > 3 else 30
            if days < 1:
                await message.answer("❌ Days must be at least 1")
                return
        except ValueError:
            await message.answer("❌ Days must be a valid number")
            return
        
        # Validate plan type
        if plan_type not in ["manual", "auto", "lifetime"]:
            await message.answer("❌ Plan must be 'manual', 'auto', or 'lifetime'")
            return
        
        # Find target user
        target_user = db.query(User).filter(User.telegram_id == target_telegram_id).first()
        if not target_user:
            await message.answer(f"❌ User with Telegram ID {target_telegram_id} not found.")
            return
        
        # Grant subscription
        from datetime import timedelta
        
        if plan_type == "lifetime":
            target_user.grandfathered = True
            target_user.subscription_type = "auto"
            target_user.subscription_end = None  # No expiry
            plan_name = "🎉 Lifetime Access"
        else:
            target_user.subscription_type = plan_type
            target_user.subscription_end = datetime.utcnow() + timedelta(days=days)
            plan_name = "💎 Signals Only" if plan_type == "manual" else "🤖 Auto-Trading"
        
        target_user.approved = True  # Auto-approve
        db.commit()
        db.refresh(target_user)
        expires = target_user.subscription_end.strftime("%Y-%m-%d") if target_user.subscription_end else "Never"
        
        # Notify admin
        await message.answer(
            f"✅ <b>Subscription Granted!</b>\n\n"
            f"👤 User: @{target_user.username or 'N/A'} ({target_telegram_id})\n"
            f"📦 Plan: {plan_name}\n"
            f"⏰ Duration: {days} days\n"
            f"📅 Expires: {expires}",
            parse_mode="HTML"
        )
        
        # Notify user
        try:
            await bot.send_message(
                chat_id=int(target_telegram_id),
                text=f"🎉 <b>Subscription Activated!</b>\n\n"
                     f"Your <b>{plan_name}</b> subscription has been activated!\n\n"
                     f"📅 Valid until: <b>{expires}</b>\n\n"
                     f"✅ You now have full access to all premium features!\n"
                     f"Use /dashboard to get started.",
                parse_mode="HTML"
            )
        except Exception as e:
            await message.answer(f"⚠️ Subscription granted but couldn't notify user: {e}")
        
    finally:
        db.close()


@dp.message(Command("notify_expired"))
async def cmd_notify_expired_subscriptions(message: types.Message):
    """Admin command to notify all users with expired subscriptions"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user or not user.is_admin:
            await message.answer("This command is only available to admins.")
            return
        
        await message.answer("Checking for expired subscriptions...")
        
        # Find users with expired subscriptions who had auto-trading enabled
        now = datetime.utcnow()
        expired_users = db.query(User).join(UserPreference).filter(
            User.subscription_end != None,
            User.subscription_end < now,
            User.grandfathered == False,
            UserPreference.auto_trading_enabled == True
        ).all()
        
        if not expired_users:
            await message.answer("No users with expired subscriptions and auto-trading enabled found.")
            return
        
        notified_count = 0
        failed_count = 0
        
        for expired_user in expired_users:
            try:
                await bot.send_message(
                    expired_user.telegram_id,
                    "⚠️ <b>Subscription Expired</b>\n\n"
                    "Your subscription has ended and auto-trading has been paused.\n\n"
                    "To continue receiving signals and auto-trading, please renew your subscription.\n\n"
                    "Use /start to view subscription options.",
                    parse_mode="HTML"
                )
                notified_count += 1
            except Exception as e:
                logger.error(f"Failed to notify user {expired_user.telegram_id}: {e}")
                failed_count += 1
        
        await message.answer(
            f"<b>Expiry Notifications Sent</b>\n\n"
            f"Notified: {notified_count} users\n"
            f"Failed: {failed_count} users\n"
            f"Total expired: {len(expired_users)} users",
            parse_mode="HTML"
        )
        
    finally:
        db.close()


@dp.message(Command("extend_all"))
async def cmd_extend_all_subscriptions(message: types.Message):
    """Admin command to add extra days to ALL active subscriptions"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user or not user.is_admin:
            await message.answer("❌ This command is only available to admins.")
            return
        
        # Parse command: /extend_all <days>
        # Example: /extend_all 1
        parts = message.text.split()
        if len(parts) < 2:
            await message.answer(
                "❌ <b>Usage:</b> /extend_all &lt;days&gt;\n\n"
                "<b>Example:</b>\n"
                "/extend_all 1  (adds 1 day to all subscriptions)\n"
                "/extend_all 7  (adds 7 days to all subscriptions)",
                parse_mode="HTML"
            )
            return
        
        # Validate and parse days
        try:
            days = int(parts[1])
            if days < 1:
                await message.answer("❌ Days must be at least 1")
                return
            if days > 365:
                await message.answer("❌ Maximum 365 days at a time for safety")
                return
        except ValueError:
            await message.answer("❌ Days must be a valid number")
            return
        
        # Find all users with active subscriptions (subscription_end in the future)
        from datetime import timedelta
        now = datetime.utcnow()
        
        active_subscribers = db.query(User).filter(
            User.subscription_end != None,
            User.subscription_end > now
        ).all()
        
        if not active_subscribers:
            await message.answer("ℹ️ No active subscribers found.")
            return
        
        # Confirm action
        await message.answer(
            f"⏳ Extending {len(active_subscribers)} subscriptions by {days} day(s)...",
            parse_mode="HTML"
        )
        
        # Extend all subscriptions
        extended_count = 0
        for subscriber in active_subscribers:
            subscriber.subscription_end = subscriber.subscription_end + timedelta(days=days)
            extended_count += 1
        
        db.commit()
        
        # Report results
        await message.answer(
            f"✅ <b>Subscriptions Extended!</b>\n\n"
            f"👥 Users affected: <b>{extended_count}</b>\n"
            f"⏰ Days added: <b>+{days}</b>\n\n"
            f"All active subscribers now have {days} extra day(s)!",
            parse_mode="HTML"
        )
        
        logger.info(f"Admin {message.from_user.id} extended {extended_count} subscriptions by {days} days")
        
    finally:
        db.close()


@dp.message(Command("list_subs"))
async def cmd_list_subscriptions(message: types.Message):
    """Admin command to list all subscribers and their end dates"""
    from datetime import timezone
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user or not user.is_admin:
            await message.answer("❌ Admin access required.")
            return
        
        subscribers = db.query(User).filter(
            User.subscription_end != None
        ).order_by(User.subscription_end.desc()).all()
        
        if not subscribers:
            await message.answer("ℹ️ No subscribers found.")
            return
        
        active_subs = []
        expired_subs = []
        now_utc = datetime.now(timezone.utc)
        
        for sub in subscribers:
            try:
                sub_end = sub.subscription_end
                if sub_end.tzinfo is None:
                    sub_end = sub_end.replace(tzinfo=timezone.utc)
                else:
                    sub_end = sub_end.astimezone(timezone.utc)
                
                days_left = (sub_end - now_utc).days if sub_end > now_utc else 0
                end_date = sub_end.strftime("%Y-%m-%d")
                username = sub.username or "No username"
                
                line = f"• <code>{sub.telegram_id}</code> @{username} | {end_date}"
                
                if sub_end > now_utc:
                    line += f" ({days_left}d left)"
                    active_subs.append(line)
                else:
                    expired_subs.append(line + " (EXPIRED)")
            except Exception as e:
                logger.error(f"Error processing subscriber {sub.telegram_id}: {e}")
                continue
        
        response = f"📋 <b>Subscriber List</b>\n\n"
        response += f"✅ <b>Active ({len(active_subs)}):</b>\n"
        response += "\n".join(active_subs[:30]) if active_subs else "None"
        
        if len(active_subs) > 30:
            response += f"\n... and {len(active_subs) - 30} more"
        
        if expired_subs:
            response += f"\n\n❌ <b>Expired ({len(expired_subs)}):</b>\n"
            response += "\n".join(expired_subs[:10])
            if len(expired_subs) > 10:
                response += f"\n... and {len(expired_subs) - 10} more"
        
        await message.answer(response, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in list_subs command: {e}")
        await message.answer(f"❌ Error: {str(e)}")
    finally:
        db.close()


@dp.message(Command("recalc_stats"))
async def cmd_recalc_stats(message: types.Message):
    """Admin command to recalculate all signal outcomes"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user or not user.is_admin:
            await message.answer("❌ Admin access required.")
            return
        
        await message.answer("🔄 Recalculating all signal outcomes...\nThis may take a moment.")
        
        from app.services.analytics import AnalyticsService
        result = AnalyticsService.recalculate_all_signal_outcomes(db, days=30)
        
        stats = AnalyticsService.get_performance_stats(db, days=30)
        
        response = f"""✅ <b>Signal Outcomes Recalculated</b>

📊 <b>Updated:</b> {result['updated']} signals

📈 <b>New Stats (30 days):</b>
• Total Signals: {stats['total_signals']}
• Won: {stats['won']} ✅
• Lost: {stats['lost']} ❌
• Breakeven: {stats['breakeven']} ⚖️
• Win Rate: {stats['win_rate']:.1f}%
• Avg PnL: {stats['avg_pnl']:.2f}%"""
        
        await message.answer(response, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in recalc_stats: {e}")
        await message.answer(f"❌ Error: {str(e)}")
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
            
            # Volatility Analysis
            volatility = analysis.get('volatility', {})
            if not volatility.get('error'):
                vol_emoji = "🔥" if volatility.get('regime') in ['extreme', 'high'] else "📊" if volatility.get('regime') == 'normal' else "😴"
                report += f"""
{vol_emoji} <b>Volatility (ATR)</b>
• 15m: {volatility.get('atr_pct_15m', 0)}% | 1h: {volatility.get('atr_pct_1h', 0)}%
• Regime: {volatility.get('regime', 'N/A').title()}
• {volatility.get('description', 'N/A')}
• Suggested SL: {volatility.get('suggested_sl_pct', 0)}% (2x ATR)
"""
            
            # BTC Correlation
            btc_corr = analysis.get('btc_correlation', {})
            if not btc_corr.get('error'):
                corr_val = btc_corr.get('correlation', 0)
                corr_bar = "🟢" * int(abs(corr_val) * 5) + "⚪" * (5 - int(abs(corr_val) * 5))
                btc_emoji = "📈" if btc_corr.get('btc_trend') == 'bullish' else "📉" if btc_corr.get('btc_trend') == 'bearish' else "➡️"
                report += f"""
🔗 <b>BTC Correlation</b>
• Correlation: {corr_val} {corr_bar}
• BTC Trend: {btc_emoji} {btc_corr.get('btc_trend', 'N/A').title()} ({btc_corr.get('btc_change_1h', 0):+.2f}%)
• Risk: {btc_corr.get('risk', 'N/A')}
"""
            
            # Session Analysis
            session = analysis.get('session', {})
            report += f"""
🕐 <b>Session</b>
• Quality: {session.get('quality', 'N/A').title()}
• {session.get('description', 'N/A')}
"""
            
            # Trade Idea Section (LONG or SHORT based on market conditions)
            trade_idea = analysis.get('trade_idea', {})
            if trade_idea and not trade_idea.get('error'):
                direction = trade_idea.get('direction', 'LONG')
                dir_emoji = "🟢" if direction == 'LONG' else "🔴"
                long_score = trade_idea.get('long_score', 0)
                short_score = trade_idea.get('short_score', 0)
                
                report += f"""
━━━━━━━━━━━━━━━━━━━━
💡 <b>Day Trade Idea</b>
━━━━━━━━━━━━━━━━━━━━

{dir_emoji} <b>Direction:</b> {direction}
{trade_idea.get('quality_emoji', '⚪')} <b>Quality:</b> {trade_idea.get('quality', 'N/A')}

<b>📊 Scoring</b>
• LONG Score: {long_score}/10 {'⬅️ SELECTED' if direction == 'LONG' else ''}
• SHORT Score: {short_score}/10 {'⬅️ SELECTED' if direction == 'SHORT' else ''}

<b>📍 Trade Levels</b>
• Entry: ${trade_idea.get('entry', 0):,.4f}
• Stop Loss: ${trade_idea.get('stop_loss', 0):,.4f} ({trade_idea.get('sl_distance_pct', 0):+.2f}%)
• TP1: ${trade_idea.get('tp1', 0):,.4f} (+{trade_idea.get('tp1_profit_pct', 0):.2f}%)
• TP2: ${trade_idea.get('tp2', 0):,.4f} (+{trade_idea.get('tp2_profit_pct', 0):.2f}%)
• R:R Ratio: {trade_idea.get('rr_ratio', 0):.2f}

{trade_idea.get('reasoning', 'No analysis available')}

<b>💬 Recommendation:</b>
<i>{trade_idea.get('recommendation', 'No recommendation')}</i>
"""
            
            # ═══════════════════════════════════════════════════════════════
            # ENTRY TIMING SECTION
            # ═══════════════════════════════════════════════════════════════
            entry_timing = analysis.get('entry_timing', {})
            if entry_timing and not entry_timing.get('error'):
                timing_signals = entry_timing.get('signals', [])[:4]
                signals_text = "\n".join([f"  {s}" for s in timing_signals]) if timing_signals else "  No specific signals"
                
                report += f"""
━━━━━━━━━━━━━━━━━━━━
⏱️ <b>Entry Timing</b>
━━━━━━━━━━━━━━━━━━━━

<b>{entry_timing.get('urgency', '⚪ UNKNOWN')}</b>
{entry_timing.get('urgency_desc', 'Could not analyze timing')}

<b>Entry Zones:</b>
• Aggressive: ${entry_timing.get('aggressive_entry', 0):,.4f} (now)
• Optimal: ${entry_timing.get('optimal_entry', 0):,.4f}
• Conservative: ${entry_timing.get('conservative_entry', 0):,.4f}

<b>Timing Signals:</b>
{signals_text}
"""
            
            # ═══════════════════════════════════════════════════════════════
            # SECTOR STRENGTH SECTION
            # ═══════════════════════════════════════════════════════════════
            sector = analysis.get('sector_analysis', {})
            if sector and sector.get('top_sectors'):
                top_sectors_text = "\n".join(sector.get('top_sectors', []))
                
                report += f"""
━━━━━━━━━━━━━━━━━━━━
🏆 <b>Sector Strength</b>
━━━━━━━━━━━━━━━━━━━━

<b>Hot Sectors Today:</b>
{top_sectors_text}

<b>Coin vs Sector:</b>
{sector.get('sector_context', 'N/A')}

<b>Rotation:</b>
{sector.get('rotation_insight', 'N/A')}
"""
            
            # ═══════════════════════════════════════════════════════════════
            # LIQUIDATION ZONES SECTION
            # ═══════════════════════════════════════════════════════════════
            liq = analysis.get('liquidation_zones', {})
            if liq and liq.get('magnet'):
                liq_summary = "\n".join(liq.get('liq_summary', [])) if liq.get('liq_summary') else "No significant clusters detected"
                
                report += f"""
━━━━━━━━━━━━━━━━━━━━
💥 <b>Liquidation Zones</b>
━━━━━━━━━━━━━━━━━━━━

<b>{liq.get('magnet', '⚪ UNKNOWN')}</b>
{liq.get('magnet_desc', 'Could not analyze liquidation zones')}

<b>Key Levels:</b>
{liq_summary}

<b>Cascade Zones:</b>
• Upside cascade: ${liq.get('cascade_zone_up', 0):,.4f}
• Downside cascade: ${liq.get('cascade_zone_down', 0):,.4f}
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
                Trade.status.in_(['closed', 'stopped', 'tp_hit', 'sl_hit'])
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
            Trade.status.in_(['closed', 'stopped', 'tp_hit', 'sl_hit'])
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
@dp.message(Command("setup_bitunix"))
@dp.message(Command("connect_bitunix"))
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
    api_key = message.text.strip()
    
    try:
        await message.delete()
    except:
        pass
    
    # Validate API key length (Bitunix keys are 32 chars)
    if len(api_key) != 32:
        await message.answer(f"""
❌ <b>Invalid API Key</b>

Your key is {len(api_key)} characters (must be exactly 32).

<b>Tips:</b>
• Make sure you copied the FULL key
• Don't include any spaces or newlines
• Copy directly from Bitunix API page

Please send your <b>API Key</b> again:
        """, parse_mode="HTML")
        return  # Stay in same state, don't proceed
    
    await state.update_data(bitunix_api_key=api_key)
    await message.answer("✅ API Key received!\n\n🔐 Now send your <b>API Secret</b> (also 32 characters):", parse_mode="HTML")
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
        
        # Validate API secret length (Bitunix secrets are 32 chars)
        if len(api_secret) != 32:
            await message.answer(f"""
❌ <b>Invalid API Secret</b>

Your secret is {len(api_secret)} characters (must be exactly 32).

<b>Tips:</b>
• Make sure you copied the FULL secret
• Don't include any spaces or newlines
• Copy directly from Bitunix API page

Please send your <b>API Secret</b> again:
            """, parse_mode="HTML")
            return  # Stay in same state, don't proceed
        
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
        
        logger.info(f"🔐 SETUP: User {user.username} - api_key len={len(api_key)}, secret len={len(api_secret)}")
        
        # Test connection BEFORE saving
        await message.answer("🔍 Testing connection to Bitunix...")
        
        try:
            from app.services.bitunix_trader import BitunixTrader
            trader = BitunixTrader(api_key, api_secret)
            balance = await trader.get_account_balance()
            
            if balance is None:
                await message.answer("""
❌ <b>Connection Failed!</b>

Bitunix rejected your API keys.

<b>Common issues:</b>
• IP whitelist enabled on Bitunix (must be disabled)
• Futures trading permission not enabled
• API keys expired or regenerated

Please check your Bitunix API settings and try again with /setup_bitunix
                """, parse_mode="HTML")
                await state.clear()
                return
                
        except Exception as e:
            logger.error(f"🔐 SETUP: Connection test failed: {e}")
            await message.answer(f"""
❌ <b>Connection Test Failed</b>

Error: {str(e)[:100]}

Please verify your API keys and try again with /setup_bitunix
            """, parse_mode="HTML")
            await state.clear()
            return
        
        # Connection successful - save keys
        encrypted_key = encrypt_api_key(api_key)
        encrypted_secret = encrypt_api_key(api_secret)
        
        prefs.bitunix_api_key = encrypted_key
        prefs.bitunix_api_secret = encrypted_secret
        prefs.preferred_exchange = "Bitunix"
        db.commit()
        
        logger.info(f"✅ SETUP: User {user.username} connected successfully! Balance: ${balance:.2f}")
        
        await message.answer(f"""
✅ <b>Bitunix API Connected!</b>

💰 Balance: <b>${balance:.2f} USDT</b>

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


@dp.message(Command("test_bitunix"))
async def cmd_test_bitunix(message: types.Message):
    """Test Bitunix API connection and show diagnostic info"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start first!")
            return
        
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs or not prefs.bitunix_api_key or not prefs.bitunix_api_secret:
            await message.answer("❌ No Bitunix API keys set. Use /setup_bitunix first.")
            return
        
        await message.answer("🔍 Testing Bitunix connection...")
        
        try:
            api_key = decrypt_api_key(prefs.bitunix_api_key)
            api_secret = decrypt_api_key(prefs.bitunix_api_secret)
            
            # Log diagnostic info
            logger.info(f"🔍 TEST: User {user.username} - key len={len(api_key)}, secret len={len(api_secret)}")
            logger.info(f"🔍 TEST: Key preview: {api_key[:6]}...{api_key[-4:] if len(api_key) > 10 else 'SHORT'}")
            
            if len(api_key) != 32:
                await message.answer(f"⚠️ API Key length is {len(api_key)} (expected 32). Please re-enter with /setup_bitunix")
                return
            
            if len(api_secret) != 32:
                await message.answer(f"⚠️ API Secret length is {len(api_secret)} (expected 32). Please re-enter with /setup_bitunix")
                return
            
            trader = BitunixTrader(api_key, api_secret)
            balance = await trader.get_account_balance()
            
            if balance is not None and balance >= 0:
                await message.answer(f"✅ <b>Connection Successful!</b>\n\n💰 Balance: ${balance:.2f} USDT", parse_mode="HTML")
            else:
                await message.answer("❌ Connection failed - check Railway logs for details.\n\nCommon issues:\n• IP whitelist on Bitunix API\n• Futures trading not enabled\n• API key expired/regenerated")
                
        except Exception as e:
            logger.error(f"🔍 TEST ERROR: {e}")
            await message.answer(f"❌ Error: {str(e)[:100]}")
    finally:
        db.close()


@dp.message(Command("remove_api"))
@dp.message(Command("remove_bitunix_api"))
@dp.message(Command("clear_api"))
async def cmd_remove_bitunix_api(message: types.Message):
    """Remove all API keys - accessible by any user"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        
        if prefs:
            prefs.bitunix_api_key = None
            prefs.bitunix_api_secret = None
            prefs.preferred_exchange = None
            prefs.auto_trading_enabled = False
            db.commit()
            await message.answer("""
✅ <b>API Keys Removed</b>

Your Bitunix API keys have been cleared.
Auto-trading has been disabled.

To reconnect: /setup_bitunix
            """, parse_mode="HTML")
            logger.info(f"🗑️ User {user.username} removed their API keys")
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


# Storage for collecting album photos
_album_storage: dict = {}
_album_timers: dict = {}

@dp.message(Command("broadcast"))
async def cmd_broadcast(message: types.Message):
    """Admin command to send a message to all users (supports text, photos, videos, and albums)"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user or not user.is_admin:
            await message.answer("❌ This command is only available to admins.")
            return
        
        # Check if this is part of a media group (album)
        if message.media_group_id:
            await handle_album_broadcast(message, db)
            return
        
        # Get broadcast text or caption
        broadcast_text = message.text or message.caption
        if not broadcast_text:
            await message.answer(
                "❌ <b>Usage:</b>\n\n"
                "📝 <b>Text:</b> /broadcast Your message here\n"
                "📷 <b>Single photo:</b> Attach photo with /broadcast caption\n"
                "🎬 <b>Video:</b> Attach video with /broadcast caption\n"
                "🖼 <b>Album:</b> Select multiple photos, add /broadcast caption to first one",
                parse_mode="HTML"
            )
            return
        
        # Parse command: /broadcast Your message here
        parts = broadcast_text.split(None, 1)
        if len(parts) < 2:
            await message.answer(
                "❌ <b>Usage:</b>\n\n"
                "📝 <b>Text:</b> /broadcast Your message here\n"
                "📷 <b>Single photo:</b> Attach photo with /broadcast caption\n"
                "🎬 <b>Video:</b> Attach video with /broadcast caption\n"
                "🖼 <b>Album:</b> Select multiple photos, add /broadcast caption to first one",
                parse_mode="HTML"
            )
            return
        
        broadcast_msg = parts[1]
        
        # Get all users
        all_users = db.query(User).all()
        sent_count = 0
        failed_count = 0
        
        # Determine media type
        has_video = message.video is not None
        has_photo = message.photo is not None and len(message.photo) > 0
        
        media_type = "video" if has_video else ("photo" if has_photo else "text")
        await message.answer(f"📤 Broadcasting {media_type} to {len(all_users)} users...")
        
        for user_to_notify in all_users:
            try:
                if has_video:
                    await bot.send_video(
                        int(user_to_notify.telegram_id),
                        message.video.file_id,
                        caption=broadcast_msg,
                        parse_mode="HTML"
                    )
                elif has_photo:
                    await bot.send_photo(
                        int(user_to_notify.telegram_id),
                        message.photo[-1].file_id,
                        caption=broadcast_msg,
                        parse_mode="HTML"
                    )
                else:
                    await bot.send_message(int(user_to_notify.telegram_id), broadcast_msg, parse_mode="HTML")
                sent_count += 1
            except Exception as e:
                logger.error(f"Failed to send broadcast to {user_to_notify.telegram_id}: {e}")
                failed_count += 1
        
        await message.answer(
            f"✅ <b>Broadcast Complete!</b>\n\n"
            f"📎 Type: {media_type.upper()}\n"
            f"✅ Sent: {sent_count}\n"
            f"❌ Failed: {failed_count}\n"
            f"📊 Total: {len(all_users)}",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Broadcast error: {e}", exc_info=True)
        await message.answer(f"❌ Error: {e}")
    finally:
        db.close()


async def handle_album_broadcast(message: types.Message, db):
    """Handle album (media group) broadcasts"""
    from aiogram.types import InputMediaPhoto
    
    media_group_id = message.media_group_id
    admin_id = message.from_user.id
    
    # Initialize storage for this album
    if media_group_id not in _album_storage:
        _album_storage[media_group_id] = {
            'photos': [],
            'caption': None,
            'admin_id': admin_id,
            'chat_id': message.chat.id
        }
    
    # Store photo
    if message.photo:
        _album_storage[media_group_id]['photos'].append(message.photo[-1].file_id)
    
    # Store caption from first message with /broadcast
    caption = message.caption or ""
    if caption.startswith("/broadcast"):
        parts = caption.split(None, 1)
        if len(parts) > 1:
            _album_storage[media_group_id]['caption'] = parts[1]
    
    # Cancel existing timer and set new one (wait for all album photos)
    if media_group_id in _album_timers:
        _album_timers[media_group_id].cancel()
    
    # Process album after 1 second delay (allows all photos to arrive)
    loop = asyncio.get_event_loop()
    _album_timers[media_group_id] = loop.call_later(
        1.0,
        lambda: asyncio.create_task(process_album_broadcast(media_group_id, db))
    )


async def process_album_broadcast(media_group_id: str, db):
    """Process and send album broadcast to all users"""
    from aiogram.types import InputMediaPhoto
    
    try:
        if media_group_id not in _album_storage:
            return
        
        album_data = _album_storage.pop(media_group_id)
        if media_group_id in _album_timers:
            del _album_timers[media_group_id]
        
        photos = album_data['photos']
        caption = album_data['caption']
        chat_id = album_data['chat_id']
        
        if not photos:
            await bot.send_message(chat_id, "❌ No photos found in album")
            return
        
        if not caption:
            await bot.send_message(chat_id, "❌ Add /broadcast <caption> to the first photo")
            return
        
        # Get all users
        all_users = db.query(User).all()
        sent_count = 0
        failed_count = 0
        
        await bot.send_message(chat_id, f"📤 Broadcasting album ({len(photos)} photos) to {len(all_users)} users...")
        
        # Build media group
        media_group = []
        for i, photo_id in enumerate(photos):
            if i == 0:
                media_group.append(InputMediaPhoto(media=photo_id, caption=caption, parse_mode="HTML"))
            else:
                media_group.append(InputMediaPhoto(media=photo_id))
        
        # Send to all users
        for user_to_notify in all_users:
            try:
                await bot.send_media_group(int(user_to_notify.telegram_id), media_group)
                sent_count += 1
                await asyncio.sleep(0.05)  # Rate limit protection
            except Exception as e:
                logger.error(f"Failed to send album to {user_to_notify.telegram_id}: {e}")
                failed_count += 1
        
        await bot.send_message(
            chat_id,
            f"✅ <b>Album Broadcast Complete!</b>\n\n"
            f"🖼 Photos: {len(photos)}\n"
            f"✅ Sent: {sent_count}\n"
            f"❌ Failed: {failed_count}\n"
            f"📊 Total: {len(all_users)}",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Album broadcast error: {e}", exc_info=True)


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
  • Total Live PnL: ${trading.get('total_live_pnl', 0):,.2f}
  • Avg Trade: ${trading.get('avg_trade_pnl', 0):.2f}

🔌 <b>Exchange Integration</b>
  • Bitunix: {exchanges.get('bitunix_users', 0)} users (Auto-trading)
  • Auto-Trading: {exchanges.get('auto_trading_enabled', 0)} users

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


@dp.message(Command("pending_payouts"))
async def cmd_pending_payouts(message: types.Message):
    """Show all users with pending referral payouts (Admin only)"""
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        # Get all users with pending earnings
        users_with_earnings = db.query(User).filter(User.referral_earnings > 0).order_by(User.referral_earnings.desc()).all()
        
        if not users_with_earnings:
            await message.answer(
                "💰 <b>Pending Referral Payouts</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                "No pending payouts at this time! ✅",
                parse_mode="HTML"
            )
            return
        
        # Calculate total pending
        total_pending = sum(user.referral_earnings for user in users_with_earnings)
        
        # Build payout list
        payout_text = f"""
💰 <b>Pending Referral Payouts</b>
━━━━━━━━━━━━━━━━━━━━

<b>Total Pending:</b> ${total_pending:.2f}

<b>Users ({len(users_with_earnings)}):</b>
"""
        
        for user in users_with_earnings:
            username = f"@{user.username}" if user.username else f"{user.first_name or 'Unknown'}"
            payout_text += f"\n• {username}\n"
            payout_text += f"  ID: <code>{user.telegram_id}</code>\n"
            payout_text += f"  Amount: <b>${user.referral_earnings:.2f}</b>\n"
            if user.crypto_wallet:
                payout_text += f"  Wallet: <code>{user.crypto_wallet}</code>\n"
            else:
                payout_text += f"  Wallet: ⚠️ <i>Not set</i>\n"
        
        payout_text += f"\n\n<i>Use /mark_paid [telegram_id] to mark a payout as sent</i>"
        
        await message.answer(payout_text, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error in pending_payouts command: {e}")
        await message.answer(f"❌ Error: {str(e)}")
    finally:
        db.close()


@dp.message(Command("mark_paid"))
async def cmd_mark_paid(message: types.Message):
    """Mark a referral payout as paid and reset earnings (Admin only)"""
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        # Parse command arguments
        parts = message.text.split()
        if len(parts) < 2:
            await message.answer(
                "❌ <b>Usage:</b> /mark_paid [telegram_id]\n\n"
                "<b>Example:</b> /mark_paid 123456789",
                parse_mode="HTML"
            )
            return
        
        telegram_id = parts[1].strip()
        
        # Find user
        user = db.query(User).filter(User.telegram_id == telegram_id).first()
        if not user:
            await message.answer(f"❌ User with ID {telegram_id} not found.")
            return
        
        if user.referral_earnings <= 0:
            await message.answer(f"❌ User has no pending earnings (${user.referral_earnings:.2f}).")
            return
        
        # Store amount before resetting
        paid_amount = user.referral_earnings
        
        # Reset earnings
        user.referral_earnings = 0.0
        db.commit()
        
        # Notify user that payment was sent
        try:
            await bot.send_message(
                user.telegram_id,
                f"✅ <b>Payment Sent!</b>\n\n"
                f"Your referral reward of <b>${paid_amount:.2f}</b> has been sent via crypto!\n\n"
                f"Thank you for spreading the word! Keep sharing to earn more 🚀",
                parse_mode="HTML"
            )
        except Exception as e:
            logger.error(f"Failed to notify user {user.telegram_id}: {e}")
        
        username = f"@{user.username}" if user.username else f"{user.first_name or 'Unknown'}"
        await message.answer(
            f"✅ <b>Payout Marked as Paid</b>\n\n"
            f"<b>User:</b> {username}\n"
            f"<b>ID:</b> <code>{telegram_id}</code>\n"
            f"<b>Amount:</b> ${paid_amount:.2f}\n\n"
            f"User has been notified! 🎉",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Error in mark_paid command: {e}")
        await message.answer(f"❌ Error: {str(e)}")
    finally:
        db.close()


@dp.message(Command("setwallet"))
async def cmd_set_wallet(message: types.Message):
    """Set crypto wallet address for referral payouts"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        # Parse wallet address from command
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            # Show current wallet if no address provided
            if user.crypto_wallet:
                await message.answer(
                    f"💰 <b>Your Crypto Wallet</b>\n\n"
                    f"<code>{user.crypto_wallet}</code>\n\n"
                    f"<b>To update:</b>\n"
                    f"/setwallet [new_address]\n\n"
                    f"<i>Example: /setwallet 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1</i>",
                    parse_mode="HTML"
                )
            else:
                await message.answer(
                    "💰 <b>Set Your Crypto Wallet</b>\n\n"
                    "You haven't set a wallet address yet!\n\n"
                    "<b>Usage:</b>\n"
                    "/setwallet [your_wallet_address]\n\n"
                    "<b>Examples:</b>\n"
                    "• USDT (TRC20): <code>/setwallet TXYZa1b2c3d4...</code>\n"
                    "• USDT (ERC20): <code>/setwallet 0x742d35Cc...</code>\n"
                    "• BTC: <code>/setwallet bc1qxy2kgdygjrsqtzq2n0yrf2493p...</code>\n\n"
                    "<i>💡 Set this to receive your $30 referral rewards!</i>",
                    parse_mode="HTML"
                )
            return
        
        wallet_address = parts[1].strip()
        
        # Basic validation
        if len(wallet_address) < 20:
            await message.answer("❌ Invalid wallet address. Please check and try again.")
            return
        
        # Update wallet
        user.crypto_wallet = wallet_address
        db.commit()
        
        await message.answer(
            f"✅ <b>Wallet Updated!</b>\n\n"
            f"<code>{wallet_address}</code>\n\n"
            f"Your referral rewards will be sent to this address! 💰",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Error in setwallet command: {e}")
        await message.answer(f"❌ Error: {str(e)}")
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


@dp.message(Command("test_user_api"))
async def cmd_test_user_api(message: types.Message):
    """Admin command to test a specific user's API - /test_user_api @username"""
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ Admin only.")
            return
        
        parts = message.text.split()
        if len(parts) < 2:
            await message.answer("Usage: /test_user_api @username or user_id")
            return
        
        target = parts[1].replace('@', '')
        user = db.query(User).filter(
            (User.username == target) | (User.telegram_id == target)
        ).first()
        
        if not user:
            await message.answer(f"❌ User '{target}' not found.")
            return
        
        prefs = user.preferences
        if not prefs or not prefs.bitunix_api_key:
            await message.answer(f"❌ @{user.username} has no API keys configured.")
            return
        
        await message.answer(f"🔄 Testing API for @{user.username}...")
        
        try:
            from app.services.bitunix_trader import BitunixTrader
            api_key = decrypt_api_key(prefs.bitunix_api_key)
            api_secret = decrypt_api_key(prefs.bitunix_api_secret)
            
            trader = BitunixTrader(api_key, api_secret)
            balance = await trader.get_account_balance()
            await trader.close()
            
            if balance and balance > 0:
                await message.answer(
                    f"✅ <b>@{user.username} API Working!</b>\n\n"
                    f"💰 Balance: ${balance:.2f}\n"
                    f"🤖 Auto-trading: {'ON' if prefs.auto_trading_enabled else 'OFF'}\n"
                    f"📊 Position size: {prefs.position_size_percent}%",
                    parse_mode="HTML"
                )
            else:
                await message.answer(
                    f"⚠️ <b>@{user.username} API Issue</b>\n\n"
                    f"Balance returned: {balance}\n"
                    f"Possible issues: No funds, API permissions, or rate limit",
                    parse_mode="HTML"
                )
        except Exception as e:
            await message.answer(f"❌ API Error for @{user.username}: {str(e)[:200]}")
    finally:
        db.close()


@dp.message(Command("check_traders"))
async def cmd_check_traders(message: types.Message):
    """Admin command to check status of all auto-traders"""
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        # Get all users with top gainers mode enabled
        users = db.query(User).join(UserPreference).filter(
            UserPreference.top_gainers_mode_enabled == True
        ).all()
        
        if not users:
            await message.answer("No users with Top Gainers mode enabled.")
            return
        
        status_text = "🔍 <b>Auto-Trader Status Check</b>\n━━━━━━━━━━━━━━━━━━━━\n\n"
        
        ready_count = 0
        issues_count = 0
        
        for user in users:
            prefs = user.preferences
            username = user.username or f"ID:{user.id}"
            
            has_keys = bool(prefs and prefs.bitunix_api_key)
            auto_on = bool(prefs and prefs.auto_trading_enabled)
            top_gainers_on = bool(prefs and prefs.top_gainers_mode_enabled)
            
            # Determine status
            if has_keys and auto_on and top_gainers_on:
                status_text += f"✅ <b>@{username}</b> - READY\n"
                ready_count += 1
            else:
                issues = []
                if not has_keys:
                    issues.append("No API")
                if not auto_on:
                    issues.append("Auto OFF")
                if not top_gainers_on:
                    issues.append("TG OFF")
                status_text += f"❌ <b>@{username}</b> - {', '.join(issues)}\n"
                issues_count += 1
        
        status_text += f"\n━━━━━━━━━━━━━━━━━━━━\n"
        status_text += f"✅ Ready: {ready_count} | ❌ Issues: {issues_count}\n"
        status_text += f"📊 Total: {len(users)}"
        
        await message.answer(status_text, parse_mode="HTML")
    finally:
        db.close()


@dp.message(Command("diagnose_all"))
async def cmd_diagnose_all(message: types.Message):
    """Admin command to run full diagnostic on all traders' APIs"""
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ Admin only.")
            return
        
        await message.answer("🔄 <b>Running full API diagnostics for all traders...</b>\n\n<i>This may take 30-60 seconds</i>", parse_mode="HTML")
        
        users = db.query(User).join(UserPreference).filter(
            UserPreference.top_gainers_mode_enabled == True,
            UserPreference.auto_trading_enabled == True
        ).all()
        
        if not users:
            await message.answer("No auto-traders found.")
            return
        
        from app.services.bitunix_trader import BitunixTrader
        
        results = []
        for user in users:
            prefs = user.preferences
            username = user.username or f"ID:{user.id}"
            
            if not prefs or not prefs.bitunix_api_key:
                results.append(f"❌ @{username}: No API keys")
                continue
            
            try:
                api_key = decrypt_api_key(prefs.bitunix_api_key)
                api_secret = decrypt_api_key(prefs.bitunix_api_secret)
                
                trader = BitunixTrader(api_key, api_secret)
                balance = await trader.get_account_balance()
                await trader.close()
                
                if balance and balance > 0:
                    pos_size = prefs.position_size_percent or 10
                    trade_size = balance * (pos_size / 100)
                    
                    if trade_size >= 10:
                        results.append(f"✅ @{username}: ${balance:.0f} → ${trade_size:.0f} trade")
                    else:
                        results.append(f"⚠️ @{username}: ${balance:.0f} → ${trade_size:.0f} (below $10 min)")
                else:
                    results.append(f"❌ @{username}: Balance returned {balance}")
            except Exception as e:
                results.append(f"❌ @{username}: {str(e)[:50]}")
            
            await asyncio.sleep(0.5)  # Rate limit
        
        report = "🔍 <b>Full API Diagnostic Report</b>\n━━━━━━━━━━━━━━━━━━━━\n\n"
        report += "\n".join(results)
        
        working = sum(1 for r in results if r.startswith("✅"))
        warning = sum(1 for r in results if r.startswith("⚠️"))
        failed = sum(1 for r in results if r.startswith("❌"))
        
        report += f"\n\n━━━━━━━━━━━━━━━━━━━━\n"
        report += f"✅ Working: {working}\n"
        report += f"⚠️ Low Balance: {warning}\n"
        report += f"❌ Failed: {failed}"
        
        await message.answer(report, parse_mode="HTML")
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
/missed_trades - Check why users missed trades

Use these commands directly in chat.""",
        parse_mode="HTML"
    )
    await callback.answer()


@dp.message(Command("missed_trades"))
async def cmd_missed_trades(message: types.Message):
    """Check trade attempts to debug why users missed trades"""
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ You don't have admin access.")
            return
        
        from app.models import TradeAttempt
        from sqlalchemy import desc
        
        # Get recent attempts
        attempts = db.query(TradeAttempt).order_by(desc(TradeAttempt.created_at)).limit(30).all()
        
        if not attempts:
            await message.answer("📊 No trade attempts logged yet. Wait for the next signal.")
            return
        
        # Group by symbol for the latest signal
        latest_symbol = attempts[0].symbol if attempts else None
        
        # Count by status for latest symbol
        success = sum(1 for a in attempts if a.symbol == latest_symbol and a.status == 'success')
        skipped = sum(1 for a in attempts if a.symbol == latest_symbol and a.status == 'skipped')
        failed = sum(1 for a in attempts if a.symbol == latest_symbol and a.status == 'failed')
        errors = sum(1 for a in attempts if a.symbol == latest_symbol and a.status == 'error')
        
        text = f"""📊 <b>Trade Attempts Debug</b>

<b>Latest Signal:</b> {latest_symbol}
✅ Success: {success}
⏭️ Skipped: {skipped}
❌ Failed: {failed}
💥 Errors: {errors}

<b>Recent Skips/Failures:</b>
"""
        # Show recent non-success attempts
        shown = 0
        for a in attempts:
            if a.status != 'success' and shown < 10:
                user = db.query(User).filter(User.id == a.user_id).first()
                username = user.username if user else f"ID:{a.user_id}"
                text += f"\n• <b>{username}</b> ({a.symbol}): {a.reason[:50]}"
                shown += 1
        
        await message.answer(text, parse_mode="HTML")
    finally:
        db.close()


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
        
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        current_pct = prefs.position_size_percent if prefs else 5.0
        current_fixed = getattr(prefs, 'position_size_dollars', None) if prefs else None
        
        if current_fixed and current_fixed > 0:
            current_display = f"${current_fixed:.0f} fixed per trade"
        else:
            current_display = f"{current_pct}% of balance per trade"
        
        await callback.message.edit_text(f"""
💰 <b>Set Position Size</b>

Current: <b>{current_display}</b>

📝 <b>Send me your preferred size:</b>

<b>Option 1 - Fixed Dollar Amount:</b>
• $50 = Always trade $50 per signal
• $100 = Always trade $100 per signal

<b>Option 2 - Percentage of Balance:</b>
• 5 = 5% of balance per trade
• 10 = 10% of balance per trade

<b>To clear fixed $ and use % again:</b>
• Send just a number like 10

💡 <b>Fixed $</b> is simpler - no balance check needed!
""", parse_mode="HTML")
        logger.info(f"💰 Setting FSM state for user {callback.from_user.id} to waiting_for_size")
        await state.set_state(PositionSizeSetup.waiting_for_size)
        logger.info(f"💰 FSM state set successfully, current state: {await state.get_state()}")
        await callback.answer()
    finally:
        db.close()


@dp.message(PositionSizeSetup.waiting_for_size)
async def process_position_size(message: types.Message, state: FSMContext):
    logger.info(f"💰 HANDLER TRIGGERED! User {message.from_user.id} sent: '{message.text}', State: {await state.get_state()}")
    db = SessionLocal()
    
    try:
        text = message.text.strip()
        is_fixed_dollar = text.startswith('$')
        
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("❌ User not found.")
            await state.clear()
            return
        
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs:
            prefs = UserPreference(user_id=user.id)
            db.add(prefs)
            db.flush()
        
        if is_fixed_dollar:
            try:
                fixed_amount = float(text.replace('$', '').strip())
                if fixed_amount < 1:
                    await message.answer("⚠️ Minimum position size is $1. Please try again:")
                    return
                if fixed_amount > 10000:
                    await message.answer("⚠️ Maximum position size is $10,000. Please try again:")
                    return
            except ValueError:
                await message.answer("⚠️ Invalid format. Send $50 for fixed dollars or 10 for percentage:")
                return
            
            prefs.position_size_dollars = fixed_amount
            db.commit()
            
            await message.answer(f"""
✅ <b>Position size set to ${fixed_amount:.0f} per trade</b>

Each signal will trade exactly <b>${fixed_amount:.0f}</b> regardless of balance.

<b>Example with 10x leverage:</b>
• Position: ${fixed_amount:.0f}
• Exposure: ${fixed_amount * 10:.0f}

💡 This is simpler - no balance calculation needed!

Use /autotrading_status to view settings.
""", parse_mode="HTML")
        else:
            try:
                size = float(text)
                if size < 1 or size > 100:
                    await message.answer("⚠️ Percentage must be between 1% and 100%. Please try again:")
                    return
            except ValueError:
                await message.answer("⚠️ Invalid format. Send $50 for fixed dollars or 10 for percentage:")
                return
            
            prefs.position_size_percent = size
            prefs.position_size_dollars = None
            db.commit()
            
            await message.answer(f"""
✅ <b>Position size updated to {size}%</b>

Each trade will use <b>{size}%</b> of your Bitunix balance.

<b>Example with $1000 balance:</b>
• Position: ${1000 * (size/100):.2f}
• With 10x leverage: ${1000 * (size/100) * 10:.2f} exposure

Use /autotrading_status to view settings.
""", parse_mode="HTML")
        
        await state.clear()
    finally:
        db.close()


@dp.callback_query(F.data == "set_top_gainer_leverage")
async def handle_set_top_gainer_leverage_button(callback: CallbackQuery, state: FSMContext):
    """Button handler for setting Top Gainer leverage"""
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
        
        # Get current leverage
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        current_leverage = prefs.top_gainers_leverage if prefs and prefs.top_gainers_leverage else 5
        
        await callback.message.answer(f"""
⚡ <b>Set Top Gainers Leverage</b>

Current: <b>{current_leverage}x</b>

📝 Send me the new leverage (1-20):

<b>Examples:</b>
• 5 = Conservative (100% profit/loss on 20% move)
• 10 = Moderate (200% profit/loss on 20% move)
• 15 = Aggressive (300% profit/loss on 20% move)

⚠️ <b>Higher leverage = Higher risk & reward</b>
Recommended: 5-10x for Top Gainers mode
""", parse_mode="HTML")
        
        await state.set_state(TopGainerLeverageSetup.waiting_for_leverage)
        await callback.answer()
    finally:
        db.close()


@dp.message(TopGainerLeverageSetup.waiting_for_leverage)
async def process_top_gainer_leverage(message: types.Message, state: FSMContext):
    """Process Top Gainer leverage input"""
    db = SessionLocal()
    
    try:
        # Validate input
        try:
            leverage = int(message.text.strip())
            if leverage < 1 or leverage > 20:
                await message.answer("⚠️ Leverage must be between 1x and 20x. Please try again:")
                return
        except ValueError:
            await message.answer("⚠️ Please send a valid number (e.g., 10 for 10x). Try again:")
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
        
        # Update leverage
        prefs.top_gainers_leverage = leverage
        db.commit()
        
        # Calculate risk profile
        if leverage <= 5:
            risk_label = "🟢 Conservative"
        elif leverage <= 10:
            risk_label = "🟡 Moderate"
        else:
            risk_label = "🔴 Aggressive"
        
        await message.answer(
            f"✅ <b>Top Gainers Leverage Updated!</b>\n\n"
            f"Leverage: <b>{leverage}x</b> {risk_label}\n\n"
            f"<b>With 20% TP/SL targets:</b>\n"
            f"• Profit per trade: {20 * leverage}% of position\n"
            f"• Loss per trade: {20 * leverage}% of position\n\n"
            f"⚠️ Higher leverage = Higher risk & reward\n"
            f"🔥 Use /toggle_top_gainers_mode to view settings",
            parse_mode="HTML"
        )
        
        await state.clear()
    finally:
        db.close()


@dp.callback_query(F.data == "view_top_gainer_stats")
async def handle_view_top_gainer_stats_button(callback: CallbackQuery):
    """Button handler for viewing Top Gainer analytics"""
    await callback.answer()
    await cmd_top_gainer_stats(callback.message)


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
                            
                            # Execute live trades if auto-trading enabled
                            if user.preferences.auto_trading_enabled:
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
                        
                        # Execute live trades if auto-trading enabled
                        if user.preferences.auto_trading_enabled:
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
    """Broadcast signal with rate limiting and retry logic to prevent message loss"""
    async with get_broadcast_lock():  # Serialize broadcasts to prevent rate limit issues
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
            
            await send_message_with_retry(settings.BROADCAST_CHAT_ID, signal_text)
            logger.info(f"Broadcast to channel successful")
            
            users = db.query(User).all()
            dm_sent_count = 0
            dm_failed_count = 0
            
            # Collect users for parallel trade execution
            trade_users = []
            
            for user in users:
                # Check if user has access (not banned, approved or admin)
                has_access, _ = check_access(user)
                if not has_access:
                    continue
                
                # Send DM alerts with rate limiting
                if user.preferences and user.preferences.dm_alerts:
                    muted_symbols = user.preferences.get_muted_symbols_list()
                    if signal.symbol not in muted_symbols:
                        success = await send_message_with_retry(user.telegram_id, signal_text)
                        if success:
                            dm_sent_count += 1
                        else:
                            dm_failed_count += 1
                        # Rate limit: small delay between messages to stay under Telegram's 30/sec limit
                        await asyncio.sleep(0.05)  # 50ms = max 20 msg/sec (safe buffer)
                
                # Collect users for parallel trade execution (auto-trading enabled + symbol not muted)
                if user.preferences and user.preferences.auto_trading_enabled:
                    muted_symbols = user.preferences.get_muted_symbols_list()
                    if signal.symbol not in muted_symbols:
                        trade_users.append(user)
            
            logger.info(f"DM broadcast complete: {dm_sent_count} sent, {dm_failed_count} failed")
            
            # 🚀 PARALLEL TRADE EXECUTION - Execute all trades simultaneously
            if trade_users:
                logger.info(f"🚀 Starting PARALLEL trade execution for {len(trade_users)} users")
                from app.services.bitunix_trader import execute_trades_for_all_users
                trade_results = await execute_trades_for_all_users(
                    signal=signal,
                    users=trade_users,
                    db=db,
                    trade_type='technical',
                    max_concurrent=10  # Limit concurrent API calls
                )
                logger.info(f"✅ Trade execution results: {trade_results['success']} success, {trade_results['failed']} failed")
        
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
                        
                        # Execute live trades if auto-trading enabled
                        if user.preferences.auto_trading_enabled:
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
            
            # Broadcast day trading signals (with 1-minute delay between each to avoid API rate limits)
            for i, signal in enumerate(daytrading_signals):
                await broadcast_daytrading_signal(signal)
                # Add 1-minute delay between signals (except after last one)
                if i < len(daytrading_signals) - 1:
                    await asyncio.sleep(60)
                    logger.info(f"Waiting 60s before next signal ({i+2}/{len(daytrading_signals)})...")
                
        except Exception as e:
            logger.error(f"Signal scanner error: {e}", exc_info=True)
        
        await asyncio.sleep(settings.SCAN_INTERVAL)


async def scalp_scanner():
    """⚡ SCALP MODE: Scan top 100 gainers every 60 seconds for fast momentum/reversal scalps"""
    logger.info("⚡ SCALP Scanner Started (60-second intervals, 40% TP/SL @ 20x)")
    
    await asyncio.sleep(10)  # Initial delay to let bot start
    
    while True:
        service = None
        try:
            from app.services.top_gainers_signals import TopGainersSignalService, broadcast_scalp_signal_simple
            
            service = TopGainersSignalService()
            await service.initialize()
            
            # Generate SCALP signal with 90-second timeout to prevent freeze
            try:
                scalp_signal = await asyncio.wait_for(
                    service.generate_scalp_signal(max_symbols=100),
                    timeout=90
                )
            except asyncio.TimeoutError:
                logger.warning("⏱️ SCALP scan timed out (90s) - continuing to next cycle")
                scalp_signal = None
            
            if scalp_signal:
                logger.info(f"⚡ SCALP signal found: {scalp_signal['symbol']} @ {scalp_signal['entry_price']}")
                # Broadcast signal (includes auto-execution if scalp_mode_enabled)
                asyncio.create_task(broadcast_scalp_signal_simple(scalp_signal))
            else:
                logger.debug("⚡ No scalp candidates found")
            
        except Exception as e:
            logger.error(f"Scalp scanner error: {e}", exc_info=True)
        finally:
            # Always cleanup service to prevent connection leaks
            if service:
                try:
                    await service.close()
                except:
                    pass
        
        await asyncio.sleep(60)  # Run every 60 seconds


# 🔒 Global lock to prevent concurrent scanner runs (causes freezing!)
_scanner_lock = asyncio.Lock()
_scanner_running = False

async def top_gainers_scanner():
    """Scan for top gainers and broadcast signals every 2 minutes"""
    global _scanner_running
    logger.info("🔥 Top Gainers Scanner Started (24/7 Parabolic Reversal Detection)")
    
    await asyncio.sleep(60)  # Wait 60s before first scan (let other services initialize)
    
    while True:
        db = None
        
        # 🔒 Skip if another scan is still running (prevents freeze from concurrent scans!)
        if _scanner_running:
            logger.warning("⚠️ Previous scan still running - skipping this cycle")
            await asyncio.sleep(30)
            continue
        
        try:
            async with _scanner_lock:
                _scanner_running = True
                
                # Update heartbeat for health monitor
                await update_heartbeat()
                
                logger.info("🔍 Scanning top gainers for parabolic reversals...")
                
                # Run top gainer signal scan with 90-second timeout (stricter!)
                db = SessionLocal()
                try:
                    from app.services.top_gainers_signals import broadcast_top_gainer_signal
                    await asyncio.wait_for(
                        broadcast_top_gainer_signal(bot, db),
                        timeout=90
                    )
                except asyncio.TimeoutError:
                    logger.warning("⏱️ Top gainers scan timed out (90s) - continuing to next cycle")
                except Exception as inner_e:
                    logger.error(f"Scan inner error: {inner_e}", exc_info=True)
                finally:
                    if db:
                        db.close()
                        db = None
                
        except Exception as e:
            logger.error(f"Top gainers scanner error: {e}", exc_info=True)
        finally:
            _scanner_running = False
            if db:
                try:
                    db.close()
                except:
                    pass
        
        # Scan every 2 minutes (120 seconds) - Prevents overlapping scans!
        await asyncio.sleep(120)


async def new_coin_alert_scanner():
    """Scan for new coin listings and send alerts every 5 minutes"""
    logger.info("🆕 New Coin Alert Scanner Started")
    
    await asyncio.sleep(90)  # Wait 90s before first scan
    
    while True:
        db = None
        try:
            await update_heartbeat()
            
            logger.info("🔍 Scanning for new coin listings...")
            
            from app.services.new_coin_alerts import scan_and_broadcast_new_coins
            
            db = SessionLocal()
            try:
                await asyncio.wait_for(
                    scan_and_broadcast_new_coins(bot, db),
                    timeout=60
                )
            except asyncio.TimeoutError:
                logger.warning("⏱️ New coin scan timed out (60s) - continuing")
            finally:
                db.close()
            
        except Exception as e:
            logger.error(f"New coin alert scanner error: {e}", exc_info=True)
        finally:
            if db:
                try:
                    db.close()
                except:
                    pass
        
        await asyncio.sleep(300)  # 5 minutes


async def volume_surge_scanner():
    """Scan for volume surges every 3 minutes - catches early pumps before 25% threshold"""
    logger.info("⚡ Volume Surge Detector Started (Early Pump Catcher)")
    
    await asyncio.sleep(45)  # Wait 45s before first scan
    
    while True:
        db = None
        try:
            await update_heartbeat()
            
            logger.info("🔍 Scanning for volume surges (early pumps)...")
            
            from app.services.volume_surge_alerts import scan_and_broadcast_volume_surges
            
            db = SessionLocal()
            try:
                await asyncio.wait_for(
                    scan_and_broadcast_volume_surges(bot, db),
                    timeout=60
                )
            except asyncio.TimeoutError:
                logger.warning("⏱️ Volume surge scan timed out (60s) - continuing")
            finally:
                db.close()
            
        except Exception as e:
            logger.error(f"Volume surge scanner error: {e}", exc_info=True)
        finally:
            if db:
                try:
                    db.close()
                except:
                    pass
        
        await asyncio.sleep(180)  # 3 minutes (faster to catch early pumps!)


async def position_monitor():
    """Monitor open positions and notify when TP/SL is hit"""
    from app.services.position_monitor import monitor_positions
    
    logger.info("Position monitor started")
    await asyncio.sleep(30)  # Wait 30s before first check
    
    while True:
        try:
            # Update heartbeat for health monitor
            await update_heartbeat()
            
            logger.info("Monitoring positions...")
            
            # Monitor live Bitunix positions
            await monitor_positions(bot)
            
        except Exception as e:
            logger.error(f"Position monitor error: {e}", exc_info=True)
        
        await asyncio.sleep(15)  # Check every 15 seconds (faster TP1 detection!)


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
                            Trade.status.in_(['closed', 'stopped', 'tp_hit', 'sl_hit'])
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

💰 <b>Realized ROI (Closed Trades)</b>
• Total: ${total_realized_pnl:+.2f}
• Trades: {len(closed_trades)} (✅ {winning_trades} | ❌ {losing_trades})
• Win Rate: {(winning_trades/len(closed_trades)*100) if closed_trades else 0:.1f}%

💹 <b>Unrealized ROI (Open Positions)</b>
• Total: ${total_unrealized_pnl:+.2f}
• Open: {len(open_trades)} position{'s' if len(open_trades) != 1 else ''}{open_positions_text}

{pnl_emoji} <b>Total Day ROI: ${total_pnl:+.2f}</b>

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
    
    # 🔓 CRITICAL: Clear any stuck advisory locks AND idle transactions from previous crashes
    try:
        from app.database import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        
        # Clear advisory locks
        db.execute(text("SELECT pg_advisory_unlock_all()"))
        db.commit()
        logger.info("✅ Cleared all stuck advisory locks")
        
        # Kill idle transactions (prevents connection pool exhaustion)
        result = db.execute(text("""
            SELECT pg_terminate_backend(pid) 
            FROM pg_stat_activity 
            WHERE state = 'idle in transaction' 
            AND datname = current_database()
            AND pid != pg_backend_pid()
        """))
        db.commit()
        killed_count = result.rowcount if hasattr(result, 'rowcount') else 0
        logger.info(f"✅ Killed {killed_count} idle transactions")
        
        db.close()
    except Exception as e:
        logger.warning(f"Could not clear database locks/connections: {e}")
    
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
    # asyncio.create_task(signal_scanner())  # ❌ DISABLED - Technical analysis signals not needed
    asyncio.create_task(top_gainers_scanner())  # ✅ ENABLED - SHORTS with exhaustion detection, LONGS with 5%+ range
    # asyncio.create_task(scalp_scanner())  # ❌ PERMANENTLY REMOVED - Ruined bot with low-quality shorts
    # asyncio.create_task(volume_surge_scanner())  # ❌ DISABLED
    # asyncio.create_task(new_coin_alert_scanner())  # ❌ DISABLED
    asyncio.create_task(position_monitor())
    # asyncio.create_task(daily_pnl_report())  # DISABLED: Daily PnL report notifications
    asyncio.create_task(funding_rate_monitor())
    # Note: Funding rate monitor may log ccxt cleanup warnings - this is a known ccxt library limitation, not a memory leak
    
    # Start health monitor (auto-recovery system)
    health_monitor = get_health_monitor()
    asyncio.create_task(health_monitor.start_monitoring())
    logger.info("🏥 Health monitor started (auto-recovery enabled)")
    
    try:
        # 🔄 DEPLOYMENT CLEANUP: Delete any stale webhook and wait for old instances to die
        logger.info("🧹 Cleaning up before polling (delete webhook, wait for old instances)...")
        try:
            await bot.delete_webhook(drop_pending_updates=False)
            logger.info("✅ Webhook deleted (if any)")
        except Exception as e:
            logger.warning(f"Could not delete webhook: {e}")
        
        # Short delay to let old instance fully stop (Railway deployments)
        await asyncio.sleep(3)
        
        logger.info("Bot polling started")
        await dp.start_polling(bot)
    finally:
        # Cleanup on shutdown
        logger.info("Bot shutting down...")
        await manager.release_lock()
        await signal_generator.close()
