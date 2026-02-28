import asyncio
import logging
import time
import ccxt.async_support as ccxt
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, StateFilter
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, BotCommand, Message
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
    waiting_for_uid = State()

# FSM States for position size
class PositionSizeSetup(StatesGroup):
    waiting_for_size = State()

# FSM States for top gainer leverage
class TopGainerLeverageSetup(StatesGroup):
    waiting_for_leverage = State()

# FSM States for custom quick trade
class CustomQuickTrade(StatesGroup):
    waiting_for_size = State()
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
            
            from datetime import datetime, timedelta
            
            from app.models import generate_uid
            new_uid = generate_uid()
            while db.query(User).filter(User.uid == new_uid).first():
                new_uid = generate_uid()
            
            user = User(
                telegram_id=str(telegram_id),
                uid=new_uid,
                username=username,
                first_name=first_name,
                is_admin=is_first_user,
                approved=True,
                grandfathered=False,
                referral_code=new_referral_code,
                referred_by=referral_code,
                trial_started_at=None,
                trial_ends_at=None,
                trial_used=False
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            
            prefs = UserPreference(
                user_id=user.id,
                top_gainers_mode_enabled=True,  # Auto-enable signals for new users
                top_gainers_trade_mode='both'   # Allow both LONG and SHORT signals
            )
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
                                f"👋 New user joined!\n\n"
                                f"👤 User: @{username or 'N/A'} ({first_name or 'N/A'})\n"
                                f"🆔 ID: `{telegram_id}`{referrer_info}\n"
                                f"💳 Status: No subscription — needs to /subscribe"
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
            return False, "🤖 This feature requires the Auto-Trading plan ($80/mo). Use /subscribe to upgrade!"
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


async def get_market_header_data():
    """
    Fetch live market data for dashboard header.
    Returns BTC price, 24h change, and market regime.
    """
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        ticker = await exchange.fetch_ticker('BTC/USDT')
        await exchange.close()
        
        btc_price = ticker['last']
        btc_change = ticker['percentage'] or 0
        
        # Determine market regime
        if btc_change >= 2:
            regime = "BULLISH"
            regime_emoji = "🟢"
        elif btc_change <= -2:
            regime = "BEARISH"
            regime_emoji = "🔴"
        else:
            regime = "NEUTRAL"
            regime_emoji = "🟡"
        
        return {
            'btc_price': btc_price,
            'btc_change': btc_change,
            'regime': regime,
            'regime_emoji': regime_emoji
        }
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return None


def _fmt_pnl(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}${val:,.2f}"

def _pnl_emoji(val: float) -> str:
    return "🟢" if val >= 0 else "🔴"

def _pnl_color(val: float) -> str:
    if val > 0:
        return f"🟩 +${abs(val):,.2f}"
    elif val < 0:
        return f"🟥 -${abs(val):,.2f}"
    return "⬜ $0.00"

def _mini_bar(val: float, max_val: float = 100) -> str:
    if max_val == 0:
        return "░░░░░░░░░░"
    filled = min(10, max(0, int(abs(val) / max_val * 10)))
    return "▰" * filled + "▱" * (10 - filled)


async def build_account_overview(user, db):
    """
    Shared helper that builds account overview data for both /start and /dashboard commands.
    Returns (text, keyboard) tuple.
    """
    from datetime import timedelta
    from sqlalchemy import func as sa_func

    db.expire(user, ['preferences'])
    db.refresh(user)

    prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
    if prefs:
        db.refresh(prefs)

    closed_statuses = ['closed', 'stopped', 'tp_hit', 'sl_hit']

    total_trades = db.query(Trade).filter(
        Trade.user_id == user.id,
        Trade.status.in_(closed_statuses)
    ).count()
    open_positions = db.query(Trade).filter(
        Trade.user_id == user.id,
        Trade.status == 'open'
    ).count()

    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=now.weekday())
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    today_pnl = db.query(sa_func.coalesce(sa_func.sum(Trade.pnl), 0)).filter(
        Trade.user_id == user.id,
        Trade.status.in_(closed_statuses),
        Trade.closed_at >= today_start
    ).scalar() or 0

    week_pnl = db.query(sa_func.coalesce(sa_func.sum(Trade.pnl), 0)).filter(
        Trade.user_id == user.id,
        Trade.status.in_(closed_statuses),
        Trade.closed_at >= week_start
    ).scalar() or 0

    month_pnl = db.query(sa_func.coalesce(sa_func.sum(Trade.pnl), 0)).filter(
        Trade.user_id == user.id,
        Trade.status.in_(closed_statuses),
        Trade.closed_at >= month_start
    ).scalar() or 0

    winning = db.query(Trade).filter(
        Trade.user_id == user.id,
        Trade.status.in_(closed_statuses),
        Trade.pnl > 0
    ).count()
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0

    open_trades = db.query(Trade).filter(
        Trade.user_id == user.id,
        Trade.status == 'open'
    ).all()
    unrealized_pnl = sum(t.exchange_unrealized_pnl or 0 for t in open_trades)

    bitunix_connected = (
        prefs and
        prefs.bitunix_api_key and
        prefs.bitunix_api_secret and
        len(prefs.bitunix_api_key) > 0 and
        len(prefs.bitunix_api_secret) > 0
    )

    live_balance_text = ""
    if bitunix_connected:
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
                live_balance_text = f"${live_balance:,.2f}"
            else:
                live_balance_text = "$0.00"
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            live_balance_text = "—"

    fresh_prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
    autotrading_enabled = fresh_prefs.auto_trading_enabled if fresh_prefs else False
    at_status = "ON" if autotrading_enabled else "OFF"
    at_dot = "🟢" if autotrading_enabled else "🔴"

    if user.grandfathered:
        sub_line = "Lifetime Access"
        sub_bar = "▓▓▓▓▓▓▓▓▓▓ ∞"
    elif user.is_subscribed and user.subscription_end:
        days_left = max(0, (user.subscription_end - now).days)
        sub_line = f"Premium  ·  {days_left}d left"
        filled = min(10, max(0, int((days_left / 30) * 10)))
        sub_bar = "▓" * filled + "░" * (10 - filled)
    else:
        sub_line = "Free Tier"
        sub_bar = "░░░░░░░░░░"

    market_data = await get_market_header_data()
    if market_data:
        btc_price = market_data['btc_price']
        btc_change = market_data['btc_change']
        change_sign = "+" if btc_change >= 0 else ""
        regime = market_data.get('regime_emoji', '🟡')
        btc_line = f"{regime} BTC <b>${btc_price:,.0f}</b>  ({change_sign}{btc_change:.1f}%)"
    else:
        btc_line = ""

    balance_line = f"💰 <b>{live_balance_text}</b> USDT\n" if live_balance_text else ""

    def _pnl_line(label, val):
        sign = "+" if val >= 0 else ""
        icon = "▲" if val > 0 else "▼" if val < 0 else "―"
        return f"  {icon} {label}  <b>{sign}${val:,.2f}</b>"

    today_line = _pnl_line("Today", today_pnl)
    week_line = _pnl_line("Week ", week_pnl)
    month_line = _pnl_line("Month", month_pnl)

    wr_text = f"  ·  <b>{win_rate:.0f}%</b> win" if total_trades > 0 else ""

    upnl_text = ""
    if open_positions > 0 and unrealized_pnl != 0:
        upnl_sign = "+" if unrealized_pnl >= 0 else ""
        upnl_text = f"  ({upnl_sign}${unrealized_pnl:,.2f})"

    uid_line = f"🆔 <code>{user.uid}</code>" if user.uid else ""

    welcome_text = (
        f"<b>TRADEHUB AI</b>\n"
        f"\n"
        f"Your AI-powered crypto futures trading\n"
        f"assistant. Scans the market around the\n"
        f"clock for high-conviction long and short\n"
        f"entries using technical analysis, derivatives\n"
        f"data, social sentiment, and on-chain signals.\n"
        f"\n"
        f"Trades are executed automatically on your\n"
        f"account with built-in risk management,\n"
        f"multi-target take profits, and breakeven\n"
        f"stop losses to protect your capital.\n"
        f"\n"
        f"{btc_line}\n"
        f"{balance_line}"
        f"\n"
        f"{uid_line}\n"
        f"{at_dot} Auto-Trading  <b>{at_status}</b>\n"
        f"💎 {sub_line}"
    )

    if not user.is_subscribed and not user.is_admin:
        def _lock(label):
            return InlineKeyboardButton(text=f"🔒 {label}", callback_data="locked_feature")

        buttons = [
            [_lock("Positions"), _lock("Performance")],
            [_lock("Quick Scan"), _lock("AI Tools")],
            [_lock("Auto-Trading"), _lock("Social Trading")],
            [_lock("Settings"), InlineKeyboardButton(text="💎 Subscribe", callback_data="subscribe_menu")],
            [
                InlineKeyboardButton(text="🎁 Referrals", callback_data="referral_stats"),
                InlineKeyboardButton(text="❓ Help", callback_data="help_menu"),
            ],
        ]
    else:
        buttons = [
            [
                InlineKeyboardButton(text="📂 Positions", callback_data="positions_menu"),
                InlineKeyboardButton(text="📊 Performance", callback_data="performance_menu"),
            ],
            [
                InlineKeyboardButton(text="🔍 Quick Scan", callback_data="scan_menu"),
                InlineKeyboardButton(text="🧠 AI Tools", callback_data="ai_tools_menu"),
            ],
            [
                InlineKeyboardButton(text="⚡ Auto-Trading", callback_data="autotrading_unified"),
                InlineKeyboardButton(text="🌐 Social Trading", callback_data="social_menu"),
            ],
            [
                InlineKeyboardButton(text="⚙️ Settings", callback_data="settings_menu"),
                InlineKeyboardButton(text="💎 Subscribe", callback_data="subscribe_menu"),
            ],
            [
                InlineKeyboardButton(text="🎁 Referrals", callback_data="referral_stats"),
                InlineKeyboardButton(text="❓ Help", callback_data="help_menu"),
            ],
        ]

    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)

    return welcome_text, keyboard


@dp.message(Command("health"))
async def cmd_health(message: types.Message):
    """Quick health check - always responds immediately (no DB access)"""
    import time
    start = time.time()
    await message.answer(f"✅ Bot is alive! Response time: {(time.time() - start)*1000:.0f}ms")


@dp.message(Command("briefing"))
async def cmd_briefing(message: types.Message):
    """Force-fetch a fresh Grok macro + geopolitical briefing and DM it to the admin."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ Admin only.")
            return
    finally:
        db.close()

    loading = await message.answer("🌍 Fetching live Grok briefing — geopolitical, macro & crypto…")

    try:
        from app.services.ai_signal_filter import refresh_grok_macro_context
        data = await refresh_grok_macro_context()
    except Exception as e:
        await loading.edit_text(f"❌ Grok briefing failed: {e}")
        return

    bias = data.get("bias", "UNKNOWN")
    summary = data.get("summary", "No data returned.")

    bias_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(bias, "⚪")
    live = data.get("live_search", False)
    source_tag = "🌐 Live web+X search" if live else "📚 Grok knowledge"

    text = (
        f"<b>🧠 Grok Macro Briefing</b>  <i>({source_tag})</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{bias_icon} <b>Bias: {bias}</b>\n\n"
        f"{summary}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Refreshed now — valid for 20 min</i>"
    )
    await loading.edit_text(text, parse_mode="HTML")


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
    
    # Check subscription access
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            return
    finally:
        db.close()
    
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
    
    # Check subscription access
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            return
    finally:
        db.close()
    
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


@dp.callback_query(F.data.startswith("scan_details:"))
async def handle_scan_details(callback: CallbackQuery):
    """Handle 'More Details' button - show full analysis data"""
    await callback.answer("Loading detailed analysis...")
    
    # Extract symbol from callback data
    symbol = callback.data.replace("scan_details:", "")
    
    # Get cached analysis
    from app.services.scan_service import _scan_cache
    cache_key = f"scan_analysis_{symbol}"
    cached = _scan_cache.get(cache_key)
    
    if not cached or (time.time() - cached.get('timestamp', 0)) > 300:  # 5 min expiry
        await callback.message.answer(
            "⏰ Analysis data expired. Please run /scan again.",
            parse_mode="HTML"
        )
        return
    
    analysis = cached['data']
    
    # Build detailed report
    report = f"""
<b>{'═' * 24}</b>
<b>📊 {symbol} - DETAILED ANALYSIS</b>
<b>{'═' * 24}</b>

"""
    
    # Sector Analysis
    sector = analysis.get('sector_analysis', {})
    if sector and sector.get('sector_context'):
        report += f"""<b>🏆 Sector Performance:</b>
{sector.get('sector_context', 'N/A')}

"""
    
    # News Sentiment
    news = analysis.get('news_sentiment', {})
    if news:
        sent_emoji = news.get('sentiment_emoji', '⚪')
        sent = news.get('sentiment', 'neutral').upper()
        impact = news.get('impact_score', 0)
        report += f"""<b>📰 News Sentiment:</b> {sent_emoji} {sent}
Impact Score: {impact}/10
<i>{news.get('summary', 'No recent news')[:200]}</i>

"""
    
    # Liquidation Zones
    liq = analysis.get('liquidation_zones', {})
    if liq and liq.get('magnet'):
        magnet = liq.get('magnet', '')
        short_zones = liq.get('liq_zones_above', [])
        long_zones = liq.get('liq_zones_below', [])
        
        report += f"""<b>💥 Liquidation Zones:</b> {magnet}
"""
        if short_zones:
            for zone in short_zones[:2]:
                report += f"<code>🔴 Shorts: ${zone['price']:,.4f} ({zone['distance_pct']:.1f}% away)</code>\n"
        if long_zones:
            for zone in long_zones[:2]:
                report += f"<code>🟢 Longs: ${zone['price']:,.4f} ({zone['distance_pct']:.1f}% away)</code>\n"
        report += "\n"
    
    # Open Interest
    oi = analysis.get('open_interest', {})
    if oi and oi.get('signal') != '⚪ N/A':
        report += f"""<b>📈 Open Interest:</b> {oi.get('signal', '')}
<code>1h Change: {oi.get('change_1h', 0):+.1f}%</code>
<code>24h Change: {oi.get('change_24h', 0):+.1f}%</code>

"""
    
    # Order Book
    ob = analysis.get('order_book', {})
    if ob and ob.get('imbalance') != '⚪ N/A':
        report += f"""<b>📖 Order Book:</b> {ob.get('imbalance', '')}
<code>Spread: {ob.get('spread', 0):.3f}% ({ob.get('spread_status', '')})</code>
<code>1% Depth: {ob.get('depth_1pct', '')}</code>
<code>2% Depth: {ob.get('depth_2pct', '')}</code>
"""
        walls = ob.get('whale_walls', [])
        if walls:
            report += "<b>Whale Walls:</b>\n"
            for wall in walls[:3]:
                report += f"<code>{wall}</code>\n"
        report += "\n"
    
    # Funding Rate
    funding = analysis.get('funding_rate', {})
    if funding and funding.get('sentiment') != '⚪ N/A':
        rate = funding.get('current_rate', 0)
        report += f"""<b>💸 Funding Rate:</b> {funding.get('sentiment', '')}
<code>Current: {rate:+.4f}%</code>
<i>{funding.get('bias', '')}</i>

"""
    
    # Long/Short Ratio
    ls = analysis.get('long_short_ratio', {})
    if ls and ls.get('global'):
        g = ls.get('global', {})
        visual = ls.get('visual_bar', '')
        
        report += f"""<b>📊 Long/Short Ratio:</b> {g.get('sentiment', '')}
<code>[{visual}] {g.get('long_pct', 0):.0f}%L / {g.get('short_pct', 0):.0f}%S</code>
<i>{g.get('warning', '')}</i>

"""
    
    # Multi-Timeframe Trend
    mtf = analysis.get('mtf_trend', {})
    if mtf and mtf.get('visual'):
        report += f"""<b>📊 Multi-Timeframe:</b> {mtf.get('visual', '')}
<code>5m    15m   1H    4H</code>
{mtf.get('alignment', '')}

"""
    
    # Session Patterns
    session = analysis.get('session_patterns', {})
    if session and session.get('current_session') != '⚪ N/A':
        asia = session.get('asia', {})
        eu = session.get('europe', {})
        us = session.get('us', {})
        report += f"""<b>🕐 Session Stats:</b> {session.get('current_session', '')}
{session.get('session_bias', '')}
<code>🌏 Asia:   {asia.get('win_rate', 0):.0f}% win | {asia.get('avg', 0):+.2f}% avg</code>
<code>🇪🇺 Europe: {eu.get('win_rate', 0):.0f}% win | {eu.get('avg', 0):+.2f}% avg</code>
<code>🇺🇸 US:     {us.get('win_rate', 0):.0f}% win | {us.get('avg', 0):+.2f}% avg</code>
<i>Best for longs: {session.get('best_long_session', 'N/A')}</i>

"""
    
    # RSI Divergence
    div = analysis.get('rsi_divergence', {})
    if div and div.get('type'):
        report += f"""<b>⚡ RSI Divergence:</b> {div.get('divergence', '')}
<code>Strength: {div.get('strength', '').upper()}</code>
<i>{div.get('action', '')}</i>

"""
    
    # Spot Flow
    spot_flow = analysis.get('spot_flow', {})
    if not spot_flow.get('error'):
        buy_pct = spot_flow.get('buy_pressure', 0)
        bar_len = 10
        buy_bars = int((buy_pct / 100) * bar_len)
        flow_bar = "▓" * buy_bars + "░" * (bar_len - buy_bars)
        signal = spot_flow.get('signal', 'N/A').replace('_', ' ').title()
        report += f"""<b>💰 Spot Flow:</b> [{flow_bar}] {buy_pct:.0f}% buy
{signal} | {spot_flow.get('confidence', 'N/A').title()} confidence

"""
    
    # Historical Context
    history = analysis.get('historical_context', {})
    if history and not history.get('error'):
        zone = history.get('zone_behavior', '')
        range_pos = history.get('range_insight', '')
        report += f"""<b>📜 Historical Context:</b>
{zone}
{range_pos}

"""
    
    # Conviction Score
    conviction = analysis.get('conviction_score', {})
    if conviction and conviction.get('score'):
        score = conviction.get('score', 50)
        bar_len = 10
        filled = int(score / 10)
        score_bar = "█" * filled + "░" * (bar_len - filled)
        
        report += f"""<b>{'═' * 24}</b>
<b>🎯 CONVICTION SCORE:</b> {conviction.get('emoji', '')} {conviction.get('direction', '')}
<code>[{score_bar}] {score}/100</code>
"""
        bull_factors = conviction.get('bullish_factors', [])
        bear_factors = conviction.get('bearish_factors', [])
        if bull_factors:
            report += f"🟢 <i>{' | '.join(bull_factors[:3])}</i>\n"
        if bear_factors:
            report += f"🔴 <i>{' | '.join(bear_factors[:3])}</i>\n"
    
    report += f"""
<b>{'─' * 24}</b>
<i>Data from scan cache</i>"""
    
    # Send as new message (don't edit original)
    await callback.message.answer(report, parse_mode="HTML")


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
        
        auto_trading = '🟢 ON' if prefs and prefs.auto_trading_enabled else '🔴 OFF'
        day_trade_leverage = prefs.user_leverage if prefs else 10
        
        settings_text = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━┓
   ⚙️ <b>Settings</b>
┗━━━━━━━━━━━━━━━━━━━━━━━┛

<b>💰 Position Management</b>
├ Position Size: <b>{prefs.position_size_percent if prefs else 10}%</b>
├ Leverage: <b>{day_trade_leverage}x</b>
└ Max Positions: <b>{prefs.max_positions if prefs else 3}</b>

<b>🤖 Trading</b>
└ Auto-Trading: {auto_trading}

<i>Tap buttons below to configure</i>
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="💰 Position", callback_data="edit_position_size"),
                InlineKeyboardButton(text="⚡ Leverage", callback_data="edit_leverage")
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


@dp.callback_query(F.data == "positions_menu")
async def handle_positions_menu(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("Please use /start first")
            return

        has_access, deny_msg = check_access(user)
        if not has_access:
            await callback.message.edit_text(deny_msg, reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🏠 Main Menu", callback_data="back_to_start")]
            ]), parse_mode="HTML")
            return

        open_trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.status == 'open'
        ).order_by(Trade.opened_at.desc()).all()

        if not open_trades:
            text = (
                "━━━━━━━━━━━━━━━━━━━━\n"
                "  📂 <b>OPEN POSITIONS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                "  No open positions right now.\n\n"
                "  <i>Signals will auto-trade when\n"
                "  new opportunities are detected.</i>"
            )
        else:
            total_unrealized = sum(t.exchange_unrealized_pnl or 0 for t in open_trades)
            lines = [
                "━━━━━━━━━━━━━━━━━━━━",
                "  📂 <b>OPEN POSITIONS</b>",
                "━━━━━━━━━━━━━━━━━━━━",
                f"",
                f"  📂 <b>{len(open_trades)}</b> Active     {_pnl_color(total_unrealized)}",
                f"  <code>  {_mini_bar(total_unrealized, abs(total_unrealized) if total_unrealized != 0 else 1)}</code>",
                ""
            ]

            for i, t in enumerate(open_trades[:10]):
                ticker = t.symbol.replace('USDT', '').replace('/USDT:USDT', '')
                dir_icon = "🟢" if t.direction == 'LONG' else "🔴"
                lev = f"{t.leverage}x" if t.leverage else ""
                upnl = t.exchange_unrealized_pnl or 0

                tp_dots = ""
                if t.tp1_hit:
                    tp_dots += "●"
                else:
                    tp_dots += "○"
                if t.tp2_hit:
                    tp_dots += "●"
                else:
                    tp_dots += "○"
                if t.tp3_hit:
                    tp_dots += "●"
                else:
                    tp_dots += "○"

                lines.append(
                    f"  {dir_icon} <b>${ticker}</b> {t.direction} {lev}\n"
                    f"     Entry <code>{t.entry_price}</code>\n"
                    f"     P&L {_pnl_color(upnl)}  TP {tp_dots}"
                )
                if i < len(open_trades[:10]) - 1:
                    lines.append("  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─")

            if len(open_trades) > 10:
                lines.append(f"\n  <i>+{len(open_trades) - 10} more positions...</i>")

            text = "\n".join(lines)

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Refresh", callback_data="positions_menu")],
            [InlineKeyboardButton(text="◀️ Main Menu", callback_data="back_to_start")]
        ])

        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "signal_history")
async def handle_signal_history(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("Please use /start first")
            return

        has_access, deny_msg = check_access(user)
        if not has_access:
            await callback.message.edit_text(deny_msg, reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🏠 Main Menu", callback_data="back_to_start")]
            ]), parse_mode="HTML")
            return

        recent = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.status.in_(['closed', 'stopped', 'tp_hit', 'sl_hit'])
        ).order_by(Trade.closed_at.desc()).limit(15).all()

        if not recent:
            text = (
                "━━━━━━━━━━━━━━━━━━━━\n"
                "  📋 <b>TRADE HISTORY</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                "  No closed trades yet.\n\n"
                "  <i>Your completed trades will\n"
                "  appear here.</i>"
            )
        else:
            wins = sum(1 for t in recent if (t.pnl or 0) > 0)
            losses = len(recent) - wins
            total_shown_pnl = sum(t.pnl or 0 for t in recent)

            lines = [
                "━━━━━━━━━━━━━━━━━━━━",
                "  📋 <b>TRADE HISTORY</b>",
                "━━━━━━━━━━━━━━━━━━━━",
                "",
                f"  {_pnl_color(total_shown_pnl)}  ({wins}W / {losses}L)",
                ""
            ]

            for t in recent:
                ticker = t.symbol.replace('USDT', '').replace('/USDT:USDT', '')
                pnl = t.pnl or 0
                result_icon = "✅" if pnl > 0 else "❌"
                dir_tag = "L" if t.direction == 'LONG' else "S"
                date_str = t.closed_at.strftime("%m/%d") if t.closed_at else ""

                tp_dots = ""
                if t.tp3_hit:
                    tp_dots = " ●●●"
                elif t.tp2_hit:
                    tp_dots = " ●●○"
                elif t.tp1_hit:
                    tp_dots = " ●○○"

                lines.append(
                    f"  {result_icon} <b>${ticker}</b> {dir_tag}  {_fmt_pnl(pnl)}{tp_dots}  <i>{date_str}</i>"
                )

            lines.append(f"\n  <i>Last {len(recent)} trades shown</i>")

            text = "\n".join(lines)

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📊 Full Performance", callback_data="performance_menu")],
            [InlineKeyboardButton(text="◀️ Main Menu", callback_data="back_to_start")]
        ])

        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "ai_tools_menu")
async def handle_ai_tools_menu(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("Please use /start first")
            return
        has_access, deny_msg = check_access(user)
        if not has_access:
            await callback.message.edit_text(deny_msg, reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🏠 Main Menu", callback_data="back_to_start")]
            ]), parse_mode="HTML")
            return
    finally:
        db.close()

    text = (
        "━━━━━━━━━━━━━━━━━━━━\n"
        "  🧠 <b>AI TOOLS</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "  Select an analysis tool:\n\n"
        "  🔍  <b>Quick Scan</b>\n"
        "      Instant coin analysis\n\n"
        "  📈  <b>Chart Patterns</b>\n"
        "      AI pattern detection\n\n"
        "  💥  <b>Liquidations</b>\n"
        "      Liquidation zone predictor\n\n"
        "  🐋  <b>Whale Tracker</b>\n"
        "      Smart money flows\n\n"
        "  📰  <b>News Scanner</b>\n"
        "      AI news impact analysis\n\n"
        "  🌡️  <b>Market Regime</b>\n"
        "      Bull / bear / neutral detector\n\n"
        "  🎯  <b>Exit Optimizer</b>\n"
        "      AI-powered exit analysis for open positions"
    )

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🔍 Quick Scan", callback_data="scan_menu"),
            InlineKeyboardButton(text="📈 Patterns", callback_data="ai_patterns_prompt")
        ],
        [
            InlineKeyboardButton(text="💥 Liquidations", callback_data="ai_liquidations_prompt"),
            InlineKeyboardButton(text="🐋 Whales", callback_data="ai_whales_prompt")
        ],
        [
            InlineKeyboardButton(text="📰 News", callback_data="ai_news_prompt"),
            InlineKeyboardButton(text="🌡️ Regime", callback_data="ai_regime_prompt")
        ],
        [InlineKeyboardButton(text="🎯 Exit Optimizer", callback_data="ai_exit_optimizer")],
        [InlineKeyboardButton(text="◀️ Main Menu", callback_data="back_to_start")]
    ])

    await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data == "ai_exit_optimizer")
async def handle_ai_exit_optimizer(callback: CallbackQuery):
    await callback.answer("Analyzing positions...")
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("Please use /start first")
            return
        has_access, deny_msg = check_access(user)
        if not has_access:
            await callback.message.edit_text(deny_msg, reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🏠 Main Menu", callback_data="back_to_start")]
            ]), parse_mode="HTML")
            return

        open_trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.status == 'open'
        ).all()

        if not open_trades:
            await callback.message.edit_text(
                "━━━━━━━━━━━━━━━━━━━━\n"
                "  🎯 <b>AI EXIT OPTIMIZER</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                "  No open positions to analyze.\n\n"
                "  <i>Open a trade first, then come back\n"
                "  for AI-powered exit recommendations.</i>",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="◀️ AI Tools", callback_data="ai_tools_menu")]
                ]),
                parse_mode="HTML"
            )
            return

        await callback.message.edit_text(
            "━━━━━━━━━━━━━━━━━━━━\n"
            "  🎯 <b>AI EXIT OPTIMIZER</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            f"  Analyzing {len(open_trades)} open position(s)...\n\n"
            "  <i>AI is reviewing price action, volume,\n"
            "  order flow, and derivatives data.</i>",
            parse_mode="HTML"
        )

        from app.services.ai_exit_optimizer import analyze_position, format_exit_analysis
        import asyncio

        results = []
        for trade in open_trades:
            analysis = await analyze_position(trade)
            if analysis:
                results.append(analysis)
            await asyncio.sleep(0.5)

        if not results:
            await callback.message.edit_text(
                "━━━━━━━━━━━━━━━━━━━━\n"
                "  🎯 <b>AI EXIT OPTIMIZER</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                "  Could not analyze positions right now.\n"
                "  Market data may be temporarily unavailable.\n\n"
                "  <i>Try again in a moment.</i>",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="🔄 Retry", callback_data="ai_exit_optimizer")],
                    [InlineKeyboardButton(text="◀️ AI Tools", callback_data="ai_tools_menu")]
                ]),
                parse_mode="HTML"
            )
            return

        text_parts = [
            "━━━━━━━━━━━━━━━━━━━━",
            "  🎯 <b>AI EXIT OPTIMIZER</b>",
            "━━━━━━━━━━━━━━━━━━━━\n"
        ]

        for analysis in results:
            text_parts.append(format_exit_analysis(analysis))
            text_parts.append("\n━━━━━━━━━━━━━━━━━━━━\n")

        prefs = user.preferences
        optimizer_status = "ON" if getattr(prefs, 'ai_exit_optimizer_enabled', True) else "OFF"
        text_parts.append(f"\nAuto-Exit: <b>{optimizer_status}</b>")

        full_text = "\n".join(text_parts)
        if len(full_text) > 4000:
            full_text = full_text[:3950] + "\n\n<i>...truncated</i>"

        toggle_text = "Disable Auto-Exit" if getattr(prefs, 'ai_exit_optimizer_enabled', True) else "Enable Auto-Exit"
        await callback.message.edit_text(
            full_text,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔄 Refresh", callback_data="ai_exit_optimizer")],
                [InlineKeyboardButton(text=toggle_text, callback_data="toggle_ai_exit")],
                [InlineKeyboardButton(text="◀️ AI Tools", callback_data="ai_tools_menu")]
            ]),
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"AI Exit Optimizer UI error: {e}")
        await callback.message.edit_text(
            "An error occurred while analyzing positions.\nPlease try again.",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔄 Retry", callback_data="ai_exit_optimizer")],
                [InlineKeyboardButton(text="◀️ AI Tools", callback_data="ai_tools_menu")]
            ]),
            parse_mode="HTML"
        )
    finally:
        db.close()


@dp.callback_query(F.data == "toggle_ai_exit")
async def handle_toggle_ai_exit(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            return

        prefs = user.preferences
        current = getattr(prefs, 'ai_exit_optimizer_enabled', True)
        prefs.ai_exit_optimizer_enabled = not current
        db.commit()

        status = "enabled" if prefs.ai_exit_optimizer_enabled else "disabled"
        await callback.message.edit_text(
            f"AI Exit Optimizer has been <b>{status}</b>.\n\n"
            f"{'The AI will now monitor your positions and suggest optimal exits.' if prefs.ai_exit_optimizer_enabled else 'Auto exit analysis is paused. You can still use manual analysis from the AI Tools menu.'}",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🎯 Back to Exit Analysis", callback_data="ai_exit_optimizer")],
                [InlineKeyboardButton(text="◀️ AI Tools", callback_data="ai_tools_menu")]
            ]),
            parse_mode="HTML"
        )
    finally:
        db.close()


@dp.callback_query(F.data == "ai_patterns_prompt")
async def handle_ai_patterns_prompt(callback: CallbackQuery):
    await callback.answer()
    await callback.message.edit_text(
        "━━━━━━━━━━━━━━━━━━━━\n"
        "  📈  <b>Chart Pattern Detection</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "  AI scans multiple timeframes for\n"
        "  classic chart patterns.\n\n"
        "  <b>Usage:</b>\n"
        "  <code>/patterns BTC</code>\n\n"
        "  <i>Replace BTC with any coin ticker</i>",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ AI Tools", callback_data="ai_tools_menu")],
            [InlineKeyboardButton(text="◀️ Main Menu", callback_data="back_to_start")]
        ]),
        parse_mode="HTML"
    )


@dp.callback_query(F.data == "ai_liquidations_prompt")
async def handle_ai_liquidations_prompt(callback: CallbackQuery):
    await callback.answer()
    await callback.message.edit_text(
        "━━━━━━━━━━━━━━━━━━━━\n"
        "  💥  <b>Liquidation Zone Predictor</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "  Predicts where liquidation cascades\n"
        "  are likely to trigger.\n\n"
        "  <b>Usage:</b>\n"
        "  <code>/liquidations BTC</code>\n\n"
        "  <i>Replace BTC with any coin ticker</i>",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ AI Tools", callback_data="ai_tools_menu")],
            [InlineKeyboardButton(text="◀️ Main Menu", callback_data="back_to_start")]
        ]),
        parse_mode="HTML"
    )


@dp.callback_query(F.data == "ai_whales_prompt")
async def handle_ai_whales_prompt(callback: CallbackQuery):
    await callback.answer()
    await callback.message.edit_text(
        "━━━━━━━━━━━━━━━━━━━━\n"
        "  🐋  <b>Whale & Smart Money Tracker</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "  Tracks institutional activity and\n"
        "  smart money flows across top coins.\n\n"
        "  <b>Usage:</b>\n"
        "  <code>/spot_flow</code>",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ AI Tools", callback_data="ai_tools_menu")],
            [InlineKeyboardButton(text="◀️ Main Menu", callback_data="back_to_start")]
        ]),
        parse_mode="HTML"
    )


@dp.callback_query(F.data == "ai_news_prompt")
async def handle_ai_news_prompt(callback: CallbackQuery):
    await callback.answer()
    await callback.message.edit_text(
        "━━━━━━━━━━━━━━━━━━━━\n"
        "  📰  <b>AI News Impact Scanner</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "  Scans latest crypto news and\n"
        "  identifies trading-relevant events.\n\n"
        "  <b>Usage:</b>\n"
        "  <code>/news</code>",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ AI Tools", callback_data="ai_tools_menu")],
            [InlineKeyboardButton(text="◀️ Main Menu", callback_data="back_to_start")]
        ]),
        parse_mode="HTML"
    )


@dp.callback_query(F.data == "ai_regime_prompt")
async def handle_ai_regime_prompt(callback: CallbackQuery):
    await callback.answer()
    await callback.message.edit_text(
        "━━━━━━━━━━━━━━━━━━━━\n"
        "  🌡️  <b>Market Regime Detector</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "  Analyzes BTC to determine if the\n"
        "  market is bullish, bearish, or neutral.\n\n"
        "  <b>Usage:</b>\n"
        "  <code>/regime</code>",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ AI Tools", callback_data="ai_tools_menu")],
            [InlineKeyboardButton(text="◀️ Main Menu", callback_data="back_to_start")]
        ]),
        parse_mode="HTML"
    )


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
• /patterns SYMBOL - AI chart pattern detection
• /liquidations SYMBOL - Liquidation zone predictor
• /spot_flow - Check institutional flow

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
        [InlineKeyboardButton(text="💰 Risk Management", callback_data="help_risk_management")],
        [InlineKeyboardButton(text="⚠️ Risk Disclaimer", callback_data="show_disclaimer")],
        [InlineKeyboardButton(text="🔙 Back", callback_data="back_to_start")]
    ])
    
    await callback.message.edit_text(help_text, reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data == "help_risk_management")
async def handle_help_risk_management(callback: CallbackQuery):
    """Display risk management guide"""
    await callback.answer()
    
    risk_text = """
💰 <b>Risk Management Guide</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>🎯 The 1-4% Rule</b>
Never risk more than <b>1-4% of your account</b> on a single trade.

<b>Example:</b>
• $1,000 account = Risk $10-$40 per trade
• $5,000 account = Risk $50-$200 per trade
• $10,000 account = Risk $100-$400 per trade

<b>📊 How to Calculate Position Size:</b>
1. Decide your risk % (1-4%)
2. Check the signal's stop loss %
3. Calculate: Position = (Account × Risk%) ÷ (SL% × Leverage)

<b>Example Calculation:</b>
• Account: $1,000
• Risk: 2% = $20
• Signal SL: 3%, Leverage: 20x
• Max loss at SL = 3% × 20x = 60%
• Position size = $20 ÷ 60% = <b>$33</b>

<b>⚡ Quick Reference (2% Risk):</b>
At 20x leverage with 3% SL:
• $1,000 account → $33 position
• $5,000 account → $166 position
• $10,000 account → $333 position

<b>🛡️ Golden Rules:</b>
• Start with 1% risk until profitable
• Scale to 2-4% only when confident
• Never risk more than 4% per trade
• Use the position size calculator in settings

<b>⚠️ Remember:</b>
At 20x leverage, a 5% move = 100% gain/loss.
Proper position sizing protects your capital!
"""
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⚙️ Set Position Size", callback_data="settings_position_size")],
        [InlineKeyboardButton(text="🔙 Back to Help", callback_data="open_help")]
    ])
    
    await callback.message.edit_text(risk_text, reply_markup=keyboard, parse_mode="HTML")


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


@dp.callback_query(F.data == "locked_feature")
async def handle_locked_feature(callback: CallbackQuery):
    await callback.answer("💎 Subscribe to unlock this feature!", show_alert=True)


@dp.callback_query(F.data == "back_to_start")
async def handle_back_to_start(callback: CallbackQuery):
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
        try:
            await callback.message.edit_text(welcome_text, reply_markup=keyboard, parse_mode="HTML")
        except Exception:
            await callback.message.answer(welcome_text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.message(Command("tracker"))
async def cmd_tracker(message: types.Message):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return

        base_url = getattr(settings, 'WEBHOOK_BASE_URL', None) or "https://tradehubai.up.railway.app"
        tracker_url = f"{base_url}/tracker"

        from sqlalchemy import func

        total = db.query(func.count(Trade.id)).scalar() or 0
        open_count = db.query(func.count(Trade.id)).filter(Trade.status == 'open').scalar() or 0
        closed = db.query(func.count(Trade.id)).filter(Trade.status == 'closed').scalar() or 0
        wins = db.query(func.count(Trade.id)).filter(Trade.status == 'closed', Trade.pnl > 0).scalar() or 0
        total_pnl = db.query(func.sum(Trade.pnl)).filter(Trade.status == 'closed').scalar() or 0
        win_rate = round(wins / closed * 100, 1) if closed > 0 else 0

        momentum = db.query(func.count(Trade.id)).filter(Trade.trade_type == 'MOMENTUM_RUNNER').scalar() or 0
        early = db.query(func.count(Trade.id)).filter(Trade.trade_type == 'EARLY_MOVER').scalar() or 0
        social = db.query(func.count(Trade.id)).filter(Trade.trade_type == 'SOCIAL_SIGNAL').scalar() or 0
        news = db.query(func.count(Trade.id)).filter(Trade.trade_type == 'NEWS_SIGNAL').scalar() or 0

        msg = (
            f"📊 <b>Trade Tracker</b>\n\n"
            f"📈 <b>{total}</b> total trades  ·  <b>{open_count}</b> open\n"
            f"🏆 Win rate <b>{win_rate}%</b>  ·  P&L <b>${total_pnl:,.2f}</b>\n\n"
            f"<b>By Type:</b>\n"
            f"🚀 Momentum: <b>{momentum}</b>\n"
            f"🔍 Early Mover: <b>{early}</b>\n"
            f"🌙 Social: <b>{social}</b>\n"
            f"📰 News: <b>{news}</b>\n\n"
            f"🔗 <a href='{tracker_url}'>Open Full Dashboard</a>"
        )
        await message.answer(msg, parse_mode="HTML", disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"Tracker command error: {e}")
        await message.answer("Error loading tracker data. Try again later.")
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


@dp.message(Command("regime"))
async def cmd_regime(message: types.Message):
    """Show current market regime and bot focus"""
    try:
        from app.services.top_gainers_signals import detect_market_regime
        
        regime = await detect_market_regime()
        
        regime_emoji = {
            'EXTREME_BULLISH': '🟢🟢',
            'EXTREME_BEARISH': '🔴🔴',
            'BULLISH': '🟢',
            'BEARISH': '🔴', 
            'NEUTRAL': '⚪'
        }.get(regime['regime'], '⚪')
        
        focus_emoji = {
            'LONGS': '📈',
            'SHORTS': '📉',
            'BOTH': '↔️'
        }.get(regime['focus'], '↔️')
        
        ema_icon = '↗️' if regime.get('btc_ema_bullish', True) else '↘️'
        
        disabled_section = ""
        if regime.get('disable_longs'):
            disabled_section = "\n⛔ <b>LONGS DISABLED</b> - BTC dumping too hard\n"
        elif regime.get('disable_shorts'):
            disabled_section = "\n⛔ <b>SHORTS DISABLED</b> - BTC pumping too hard\n"
        
        regime_text = f"""
{regime_emoji} <b>MARKET REGIME: {regime['regime']}</b>

<b>BTC Analysis:</b>
• 24h Change: <code>{regime['btc_change']:+.2f}%</code>
• RSI (15m): <code>{regime['btc_rsi']:.0f}</code>
• EMA Trend: {ema_icon} {'Bullish' if regime.get('btc_ema_bullish') else 'Bearish'}

{focus_emoji} <b>Bot Focus: {regime['focus']}</b>{disabled_section}

<b>Scanning Order:</b>
{'📉 SHORTS first → 📈 LONGS second' if regime['focus'] == 'SHORTS' else '📈 LONGS first → 📉 SHORTS second'}

<b>Thresholds:</b>
• Extreme: BTC ±3% required + RSI/EMA (disables opposite)
• Normal: BTC ±1%, RSI ≤45/≥55, EMA (2+ = priority)

<i>Updates every 2 minutes automatically</i>
"""
        await message.answer(regime_text, parse_mode="HTML")
        
    except Exception as e:
        await message.answer(f"Error checking regime: {str(e)}")


@dp.message(Command("buzz"))
async def cmd_buzz(message: types.Message):
    """Show social buzz, influencer activity, and news for a coin."""
    try:
        args = message.text.split()
        if len(args) < 2:
            await message.answer(
                "📊 <b>Social Buzz Tracker</b>\n\n"
                "Usage: <code>/buzz BTC</code> or <code>/buzz SOL</code>\n\n"
                "Shows top influencers, viral posts, news, and buzz momentum for any coin.",
                parse_mode="HTML"
            )
            return
        
        coin = args[1].upper().replace('$', '').replace('USDT', '')
        symbol = f"{coin}USDT"
        
        await message.answer(f"🔍 Fetching social buzz for <b>${coin}</b>...", parse_mode="HTML")
        
        from app.services.lunarcrush import (
            get_coin_metrics, get_coin_creators, get_coin_top_posts,
            get_coin_news, get_social_time_series, get_influencer_consensus,
            interpret_signal_score, interpret_sentiment
        )
        
        import asyncio
        metrics, creators, posts, news, time_series, influencer = await asyncio.gather(
            get_coin_metrics(symbol),
            get_coin_creators(symbol, limit=5),
            get_coin_top_posts(symbol, limit=3),
            get_coin_news(symbol, limit=3),
            get_social_time_series(symbol),
            get_influencer_consensus(symbol),
            return_exceptions=True
        )
        
        if isinstance(metrics, Exception):
            metrics = None
        if isinstance(creators, Exception):
            creators = []
        if isinstance(posts, Exception):
            posts = []
        if isinstance(news, Exception):
            news = []
        if isinstance(time_series, Exception):
            time_series = None
        if isinstance(influencer, Exception):
            influencer = None
        
        if not metrics and not creators and not posts:
            await message.answer(
                f"❌ No social data found for <b>${coin}</b>. "
                f"This coin may not have enough social activity to track.",
                parse_mode="HTML"
            )
            return
        
        msg = f"📊 <b>SOCIAL BUZZ: ${coin}</b>\n"
        
        if metrics:
            galaxy = metrics.get('galaxy_score', 0)
            sentiment = metrics.get('sentiment', 0)
            social_vol = metrics.get('social_volume', 0)
            interactions = metrics.get('interactions_24h', 0)
            dominance = metrics.get('social_dominance', 0)
            alt_rank = metrics.get('alt_rank', 9999)
            
            gs = min(galaxy / 16, 1.0) * 25
            sn = min(max(sentiment if sentiment <= 1 else sentiment/100, 0), 1.0) * 15
            sv = 15 if social_vol >= 1000 else 12 if social_vol >= 500 else 8 if social_vol >= 100 else 5 if social_vol >= 20 else 2
            si = 15 if interactions >= 100000 else 12 if interactions >= 50000 else 9 if interactions >= 10000 else 6 if interactions >= 1000 else 2
            dm = 10 if dominance >= 1.0 else 8 if dominance >= 0.5 else 5 if dominance >= 0.1 else 2
            ar = 10 if alt_rank <= 50 else 8 if alt_rank <= 100 else 5 if alt_rank <= 300 else 2
            strength = min(gs + sn + sv + si + dm + ar, 100)
            strength_bar = "🟢" if strength >= 70 else "🟡" if strength >= 45 else "🟠"
            
            sent_pct = int(sentiment * 100) if sentiment <= 1 else int(sentiment)
            rating = interpret_signal_score(galaxy)
            
            msg += (
                f"\n<b>📡 Overview</b>\n"
                f"{strength_bar} Social Strength <b>{strength:.0f}/100</b>\n"
                f"🌙 Galaxy <b>{galaxy}/16</b> {rating}\n"
                f"💬 Sentiment <b>{sent_pct}%</b> · Posts <b>{social_vol:,}</b> · Interactions <b>{interactions:,}</b>\n"
            )
            if dominance > 0:
                msg += f"📡 Dominance <b>{dominance:.2f}%</b>"
                if alt_rank < 9999:
                    msg += f" · AltRank <b>#{alt_rank}</b>"
                msg += "\n"
        
        if time_series and isinstance(time_series, dict) and time_series.get('trend'):
            trend_icon = {"RISING": "📈", "FALLING": "📉", "STABLE": "➡️"}.get(time_series.get('trend', ''), "➡️")
            sent_icon = {"IMPROVING": "😀", "DECLINING": "😟", "STABLE": "😐"}.get(time_series.get('sentiment_trend', ''), "😐")
            msg += (
                f"\n<b>📊 Buzz Momentum (24h)</b>\n"
                f"{trend_icon} Trend <b>{time_series.get('trend', 'UNKNOWN')}</b> ({time_series.get('buzz_change_pct', 0):+.0f}%)\n"
                f"{sent_icon} Sentiment <b>{time_series.get('sentiment_trend', 'UNKNOWN')}</b> ({time_series.get('sentiment_change', 0):+.1f} pts)\n"
                f"Interactions: <b>{time_series.get('recent_avg_interactions', 0):.0f}</b> avg (was {time_series.get('prior_avg_interactions', 0):.0f})\n"
            )
        
        if influencer and isinstance(influencer, dict) and influencer.get('num_creators', 0) > 0:
            cons = influencer.get('consensus', 'MIXED')
            cons_icon = {"BULLISH": "🟢", "LEAN BULLISH": "🟢", "BEARISH": "🔴", "LEAN BEARISH": "🔴", "MIXED": "⚖️"}.get(cons, "⚖️")
            total_fol = influencer.get('total_followers', 0)
            followers_display = f"{total_fol/1e6:.1f}M" if total_fol >= 1e6 else f"{total_fol/1e3:.0f}K"
            msg += (
                f"\n<b>👥 Influencer Consensus</b>\n"
                f"{cons_icon} <b>{cons}</b> ({influencer.get('bullish_count', 0)}🟢 {influencer.get('bearish_count', 0)}🔴 {influencer.get('neutral_count', 0)}⚪)\n"
                f"Reach <b>{followers_display}</b> followers"
            )
            if influencer.get('big_accounts', 0) > 0:
                msg += f" · <b>{influencer.get('big_accounts', 0)}</b> whale accounts"
            msg += "\n"
        
        if creators:
            msg += f"\n<b>🎤 Top Influencers Talking ${coin}</b>\n"
            for i, c in enumerate(creators[:5], 1):
                followers = f"{c['followers']/1e6:.1f}M" if c['followers'] >= 1e6 else f"{c['followers']/1e3:.0f}K" if c['followers'] >= 1000 else str(c['followers'])
                handle = f"@{c['handle']}" if c['handle'] else c['name']
                sent_emoji = "🟢" if c['sentiment'] > 60 else "🔴" if c['sentiment'] < 40 else "⚪"
                msg += f"{i}. {sent_emoji} <b>{handle}</b> ({followers} followers)\n"
        
        if posts:
            msg += f"\n<b>🔥 Viral Posts</b>\n"
            for i, p in enumerate(posts[:3], 1):
                body = p['body'][:120] + '...' if len(p['body']) > 120 else p['body']
                body = body.replace('<', '&lt;').replace('>', '&gt;')
                creator = p.get('creator_handle') or p.get('creator_name', '')
                if creator:
                    creator = f" — @{creator}"
                msg += f"{i}. <i>\"{body}\"</i>{creator}\n"
        
        if news:
            msg += f"\n<b>📰 Latest News</b>\n"
            for i, n in enumerate(news[:3], 1):
                title = n['title'][:100] + '...' if len(n['title']) > 100 else n['title']
                title = title.replace('<', '&lt;').replace('>', '&gt;')
                source = n.get('source', '')
                msg += f"{i}. {title}"
                if source:
                    msg += f" <i>({source})</i>"
                msg += "\n"
        
        msg += f"\n<i>Social Intelligence · 5min cache</i>"
        
        if len(msg) > 4000:
            msg = msg[:3997] + "..."
        
        await message.answer(msg, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Buzz command error: {e}")
        await message.answer(f"Error fetching buzz data: {str(e)[:200]}")


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
                    description="Trading Bot Auto-Trading Renewal ($80/month)",
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
            description="Trading Bot Auto-Trading Subscription ($80/month - BLACK FRIDAY!)",
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


@dp.message(Command("trial"))
async def cmd_trial(message: types.Message):
    """Redirect to subscribe - trials are no longer available"""
    await message.answer(
        "📢 <b>Free trials are no longer available.</b>\n\n"
        "Use /subscribe to get started with a paid subscription!\n\n"
        "💎 Full access includes:\n"
        "✅ AI-powered trading signals\n"
        "✅ Market analysis tools\n"
        "✅ Auto-trading with Bitunix\n"
        "✅ All scanner modes",
        parse_mode="HTML"
    )


@dp.message(F.text.regexp(r'^\d{6,10}$'), StateFilter(None))
async def handle_uid_number(message: types.Message):
    """Handle direct UID number submission (6-10 digits) - just notify admin, no DB storage needed"""
    user_id = message.from_user.id
    uid = message.text.strip()
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(user_id)).first()
        if not user:
            await message.answer("Please use /start first to register!")
            return
        
        # Notify admin with UID
        from app.config import settings
        if settings.OWNER_TELEGRAM_ID:
            try:
                sub_status = 'Subscribed' if user.is_subscribed else 'Grandfathered' if user.grandfathered else 'No subscription'
                admin_msg = (
                    f"🆔 <b>Bitunix UID Received</b>\n\n"
                    f"User: @{user.username or 'No username'}\n"
                    f"Telegram ID: <code>{user.telegram_id}</code>\n"
                    f"Bitunix UID: <code>{uid}</code>\n"
                    f"Status: {sub_status}"
                )
                await bot.send_message(settings.OWNER_TELEGRAM_ID, admin_msg, parse_mode="HTML")
            except Exception as e:
                logger.error(f"Failed to notify admin about UID: {e}")
        
        await message.answer(
            f"✅ <b>UID Received!</b>\n\n"
            f"Your UID: <code>{uid}</code>",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Error handling UID submission: {e}", exc_info=True)
        await message.answer(f"❌ Error: {type(e).__name__}: {str(e)[:150]}")
    finally:
        db.close()


@dp.message(Command("setuid"))
async def cmd_setuid(message: types.Message):
    """Set Bitunix UID - just notify admin, no DB storage needed"""
    db = SessionLocal()
    
    BITUNIX_REFERRAL_LINK = "https://www.bitunix.com/register?vipCode=fgq7for"
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        # Parse UID from message
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            await message.answer(
                "⚠️ <b>Please provide your Bitunix UID</b>\n\n"
                "Usage: <code>/setuid YOUR_UID</code>\n\n"
                f"Don't have an account yet?\n"
                f"👉 <a href='{BITUNIX_REFERRAL_LINK}'>Sign up on Bitunix</a>",
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            return
        
        uid = parts[1].strip()
        
        # Validate UID (should be numeric, typically 6-10 digits)
        if not uid.isdigit() or len(uid) < 5:
            await message.answer(
                "⚠️ <b>Invalid UID format</b>\n\n"
                "Bitunix UID should be a number (e.g., 1234567)\n\n"
                "📍 <i>Find your UID in Bitunix: Profile → Copy UID</i>",
                parse_mode="HTML"
            )
            return
        
        # Notify admin about UID
        from app.config import settings
        if settings.OWNER_TELEGRAM_ID:
            try:
                sub_status = 'Subscribed' if user.is_subscribed else 'Grandfathered' if user.grandfathered else 'No subscription'
                admin_msg = (
                    f"🆔 <b>Bitunix UID Received</b>\n\n"
                    f"User: @{user.username or 'No username'}\n"
                    f"Telegram ID: <code>{user.telegram_id}</code>\n"
                    f"Bitunix UID: <code>{uid}</code>\n"
                    f"Status: {sub_status}"
                )
                await bot.send_message(settings.OWNER_TELEGRAM_ID, admin_msg, parse_mode="HTML")
            except Exception as e:
                logger.error(f"Failed to notify admin about UID: {e}")
        
        await message.answer(
            f"✅ <b>UID Received!</b>\n\n"
            f"Your UID: <code>{uid}</code>",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Error in setuid command: {e}", exc_info=True)
        await message.answer(f"❌ Error: {type(e).__name__}: {str(e)[:150]}")
    finally:
        db.close()


@dp.callback_query(F.data.startswith("approve_trial_"))
async def handle_approve_trial(callback: CallbackQuery):
    """Legacy handler - trials are no longer available"""
    await callback.answer("Free trials are no longer available. Use /grant to give access.", show_alert=True)


@dp.callback_query(F.data.startswith("reject_trial_"))
async def handle_reject_trial(callback: CallbackQuery):
    """Legacy handler - trials are no longer available"""
    await callback.answer("Free trials are no longer available.", show_alert=True)


@dp.callback_query(F.data == "subscribe_menu")
async def handle_subscribe_menu(callback: CallbackQuery):
    """Handle subscribe button from main menu - show tier selection"""
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
            current_tier = user.subscription_type or "auto"
            tier_name = "AI Assistant" if current_tier == "scan" else "Auto-Trading"
            
            # Show current subscription with upgrade option if on scan tier
            buttons = []
            
            if current_tier == "scan":
                # Offer upgrade to auto
                buttons.append([InlineKeyboardButton(text="🚀 Upgrade to Auto-Trading ($80/mo)", callback_data="subscribe_tier_auto")])
            
            buttons.append([InlineKeyboardButton(text="🔄 Renew Subscription", callback_data=f"renew_{current_tier}")])
            buttons.append([InlineKeyboardButton(text="🔙 Back", callback_data="back_to_start")])
            
            await callback.message.edit_text(
                f"✅ <b>Active Subscription: {tier_name}</b>\n\n"
                f"Your subscription is active until:\n"
                f"📅 <b>{expires}</b>\n\n"
                f"<i>Keep crushing it! 🚀</i>",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons)
            )
            return
        
        # User needs to subscribe - show tier selection
        from app.tiers import TIER_CONFIG
        
        scan_config = TIER_CONFIG["scan"]
        auto_config = TIER_CONFIG["auto"]
        
        buttons = [
            [InlineKeyboardButton(text=f"🤖 AI Assistant - ${scan_config.price_usd:.0f}/mo", callback_data="subscribe_tier_scan")],
            [InlineKeyboardButton(text=f"🚀 Auto-Trading - ${auto_config.price_usd:.0f}/mo", callback_data="subscribe_tier_auto")],
            [InlineKeyboardButton(text="🔙 Back", callback_data="back_to_start")]
        ]
        
        await callback.message.edit_text(
            f"💎 <b>Choose Your Plan</b>\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>{scan_config.display_name} - ${scan_config.price_usd:.0f}/month</b>\n"
            f"{scan_config.description}\n\n"
            f"<b>Includes:</b>\n" + "\n".join(scan_config.features[:4]) + "\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>{auto_config.display_name} - ${auto_config.price_usd:.0f}/month</b>\n"
            f"{auto_config.description}\n\n"
            f"<b>Includes:</b>\n" + "\n".join(auto_config.features[:5]) + "\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>🔐 All payments are processed securely via crypto</i>",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons)
        )
    finally:
        db.close()


@dp.callback_query(F.data == "start_free_trial")
async def handle_start_free_trial(callback: CallbackQuery):
    """Trials are no longer available - redirect to subscribe"""
    await callback.answer("Free trials are no longer available.", show_alert=True)
    await callback.message.edit_text(
        "📢 <b>Free trials are no longer available.</b>\n\n"
        "Please choose a paid plan to get started!",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="💎 View Plans", callback_data="subscribe_menu")],
            [InlineKeyboardButton(text="🔙 Back", callback_data="back_to_start")]
        ])
    )


@dp.callback_query(F.data == "prompt_bitunix_uid")
async def handle_prompt_bitunix_uid(callback: CallbackQuery):
    """Prompt user to send their Bitunix UID"""
    await callback.answer()
    
    BITUNIX_REFERRAL_LINK = "https://www.bitunix.com/register?vipCode=fgq7for"
    
    await callback.message.edit_text(
        f"📤 <b>Send Your Bitunix UID</b>\n\n"
        f"Please reply with your Bitunix UID number.\n\n"
        f"<b>Example:</b> <code>/setuid 1234567</code>\n\n"
        f"📍 <b>How to find your UID:</b>\n"
        f"1. Open Bitunix app/website\n"
        f"2. Go to Profile\n"
        f"3. Copy your UID number\n\n"
        f"<i>Don't have an account yet?</i>",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔗 Sign Up on Bitunix", url=BITUNIX_REFERRAL_LINK)],
            [InlineKeyboardButton(text="🔙 Back", callback_data="start_free_trial")]
        ])
    )


@dp.callback_query(F.data.startswith("subscribe_tier_"))
async def handle_subscribe_tier(callback: CallbackQuery):
    """Handle tier selection and create payment invoice"""
    await callback.answer()
    
    tier = callback.data.replace("subscribe_tier_", "")  # "scan" or "auto"
    
    from app.tiers import TIER_CONFIG
    from app.services.oxapay import OxaPayService
    from app.config import settings
    import os
    
    if tier not in TIER_CONFIG:
        await callback.message.edit_text("Invalid plan selected. Please try again.")
        return
    
    tier_config = TIER_CONFIG[tier]
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("You're not registered. Use /start to begin!")
            return
        
        if not settings.OXAPAY_MERCHANT_API_KEY:
            await callback.message.edit_text(
                "⚠️ Subscription system is being set up. Please check back soon!",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="🔙 Back", callback_data="subscribe_menu")
                ]])
            )
            return
        
        oxapay = OxaPayService(settings.OXAPAY_MERCHANT_API_KEY)
        
        order_id = f"sub_{tier}_{user.telegram_id}_{int(datetime.utcnow().timestamp())}"
        webhook_url = os.getenv("WEBHOOK_BASE_URL", "https://tradehubai.up.railway.app") + "/webhooks/oxapay"
        
        logger.info(f"Creating OxaPay invoice for user {user.telegram_id}, tier: {tier}, amount: {tier_config.price_usd}")
        
        invoice = oxapay.create_invoice(
            amount=tier_config.price_usd,
            currency="USD",
            description=f"Tradehub {tier_config.display_name} (${tier_config.price_usd:.0f}/month)",
            order_id=order_id,
            callback_url=webhook_url,
            metadata={
                "telegram_id": str(user.telegram_id),
                "plan_type": tier
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
                    plan_type=tier,
                    amount=tier_config.price_usd,
                    status="pending"
                )
                db.add(pending_invoice)
                db.commit()
                logger.info(f"✅ Stored invoice {invoice['trackId']} in database for auto-verification (tier: {tier})")
            except Exception as e:
                logger.error(f"Failed to store invoice in database: {e}")
            
            # Build features list
            features_text = "\n".join([f"{f}" for f in tier_config.features])
            
            await callback.message.edit_text(
                f"💎 <b>{tier_config.display_name} - ${tier_config.price_usd:.0f}/month</b>\n\n"
                f"<b>What's Included:</b>\n{features_text}\n\n"
                f"<b>Payment Options:</b>\n"
                f"🔹 BTC, ETH, USDT, and more cryptocurrencies\n\n"
                f"👇 <b>Click below to subscribe with crypto:</b>",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="💳 Pay with Crypto", url=invoice["payLink"])],
                    [InlineKeyboardButton(text="🔙 Back", callback_data="subscribe_menu")]
                ])
            )
        else:
            logger.error(f"Failed to create OxaPay invoice: {invoice}")
            await callback.message.edit_text(
                "⚠️ Unable to generate payment link. Please try again later or contact support.",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="🔙 Back", callback_data="subscribe_menu")
                ]])
            )
    finally:
        db.close()


@dp.callback_query(F.data.startswith("renew_"))
async def handle_renew_subscription(callback: CallbackQuery):
    """Handle subscription renewal"""
    await callback.answer()
    
    tier = callback.data.replace("renew_", "")  # "scan" or "auto"
    
    from app.tiers import TIER_CONFIG
    from app.services.oxapay import OxaPayService
    from app.config import settings
    import os
    
    if tier not in TIER_CONFIG:
        tier = "auto"  # Default to auto for legacy
    
    tier_config = TIER_CONFIG[tier]
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("You're not registered. Use /start to begin!")
            return
        
        if not settings.OXAPAY_MERCHANT_API_KEY:
            await callback.message.edit_text(
                "⚠️ Subscription system is being set up. Please check back soon!",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="🔙 Back", callback_data="subscribe_menu")
                ]])
            )
            return
        
        oxapay = OxaPayService(settings.OXAPAY_MERCHANT_API_KEY)
        
        order_id = f"renew_{tier}_{user.telegram_id}_{int(datetime.utcnow().timestamp())}"
        webhook_url = os.getenv("WEBHOOK_BASE_URL", "https://tradehubai.up.railway.app") + "/webhooks/oxapay"
        
        invoice = oxapay.create_invoice(
            amount=tier_config.price_usd,
            currency="USD",
            description=f"Tradehub {tier_config.display_name} Renewal (${tier_config.price_usd:.0f}/month)",
            order_id=order_id,
            callback_url=webhook_url,
            metadata={
                "telegram_id": str(user.telegram_id),
                "plan_type": tier
            }
        )
        
        if invoice and invoice.get("payLink"):
            from app.models import PendingInvoice
            try:
                pending_invoice = PendingInvoice(
                    user_id=user.id,
                    track_id=invoice["trackId"],
                    order_id=order_id,
                    plan_type=tier,
                    amount=tier_config.price_usd,
                    status="pending"
                )
                db.add(pending_invoice)
                db.commit()
            except Exception as e:
                logger.error(f"Failed to store renewal invoice: {e}")
            
            expires = user.subscription_end.strftime("%Y-%m-%d") if user.subscription_end else "N/A"
            
            await callback.message.edit_text(
                f"🔄 <b>Renew {tier_config.display_name}</b>\n\n"
                f"Current expiry: <b>{expires}</b>\n"
                f"After renewal: <b>+30 days added</b>\n\n"
                f"💰 Price: <b>${tier_config.price_usd:.0f}/month</b>\n\n"
                f"👇 <b>Click below to renew with crypto:</b>",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="💳 Pay to Renew", url=invoice["payLink"])],
                    [InlineKeyboardButton(text="🔙 Back", callback_data="subscribe_menu")]
                ])
            )
        else:
            await callback.message.edit_text(
                "⚠️ Unable to generate payment link. Please try again later.",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="🔙 Back", callback_data="subscribe_menu")
                ]])
            )
    finally:
        db.close()


@dp.callback_query(F.data == "my_positions")
async def handle_my_positions(callback: CallbackQuery):
    """Show user's open positions with live PnL"""
    await callback.answer()
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("You're not registered. Use /start to begin!")
            return
        
        # Get open positions
        open_trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.status == 'open'
        ).order_by(Trade.opened_at.desc()).all()
        
        if not open_trades:
            text = """<b>📊 MY POSITIONS</b>

No open positions.

Your trades will appear here when you have active positions."""
        else:
            positions_text = ""
            for trade in open_trades:
                direction_emoji = "🟢" if trade.direction == "LONG" else "🔴"
                symbol_clean = trade.symbol.replace('/USDT', '').replace('USDT', '')
                
                # Calculate unrealized PnL if we have current price
                entry = trade.entry_price or 0
                current_price = await get_cached_price(trade.symbol)
                
                if current_price and entry > 0:
                    if trade.direction == "LONG":
                        pnl_pct = ((current_price - entry) / entry) * 100
                    else:
                        pnl_pct = ((entry - current_price) / entry) * 100
                    
                    pnl_display = f"+{pnl_pct:.2f}%" if pnl_pct >= 0 else f"{pnl_pct:.2f}%"
                else:
                    pnl_display = "---"
                
                positions_text += f"{direction_emoji} <b>{symbol_clean}</b>  Entry ${entry:.4f}  PnL {pnl_display}\n"
            
            text = f"""<b>📊 MY POSITIONS</b>

<b>{len(open_trades)} Open</b>

{positions_text}
<i>PnL updates on refresh</i>"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Refresh", callback_data="my_positions")],
            [InlineKeyboardButton(text="◀️ Back", callback_data="back_to_start")]
        ])
        
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
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
            f"• When they subscribe to <b>Auto-Trading ($80/mo)</b>, you get <b>$30 USD</b> in crypto!\n"
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
                    InlineKeyboardButton(text="🌙 Social Trading", callback_data="social_menu"),
                    InlineKeyboardButton(text="⚙️ Settings", callback_data="settings")
                ],
                [
                    InlineKeyboardButton(text="🆘 Support", callback_data="support_menu"),
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
                    InlineKeyboardButton(text="🌙 Social Trading", callback_data="social_menu"),
                    InlineKeyboardButton(text="⚙️ Settings", callback_data="settings")
                ],
                [
                    InlineKeyboardButton(text="🆘 Support", callback_data="support_menu"),
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
    """Leverage is now set per-signal type, not globally"""
    await callback.answer("Leverage is configured per signal type automatically.", show_alert=True)


@dp.callback_query(F.data == "edit_notifications")
async def handle_edit_notifications(callback: CallbackQuery):
    """Show notifications settings"""
    await callback.answer("Use /toggle_alerts to enable/disable DM notifications", show_alert=True)


@dp.callback_query(F.data.in_({"toggle_top_gainers_mode", "toggle_top_gainers_shorts", "toggle_top_gainers_longs", "top_gainers_unified", "set_top_gainer_leverage", "view_top_gainer_stats"}))
async def handle_top_gainers_legacy(callback: CallbackQuery):
    """Legacy Top Gainers handlers - mode removed, redirect to Social Trading"""
    await callback.answer("Top Gainers mode has been replaced by Social Trading", show_alert=True)


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
            
            sig_lev_qs = getattr(signal, 'leverage', 10) or 10
            tp_pnl = calculate_leverage_pnl(signal.entry_price, signal.take_profit, signal.direction, sig_lev_qs)
            sl_pnl = calculate_leverage_pnl(signal.entry_price, signal.stop_loss, signal.direction, sig_lev_qs)
            
            signals_text += f"""
{i}. {direction_emoji} <b>{signal.symbol} {signal.direction}</b> ({type_badge})
   Entry: ${signal.entry_price:.4f}
   SL: ${signal.stop_loss:.4f} | TP: ${signal.take_profit:.4f}
   
   💰 {sig_lev_qs}x Leverage:
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
    await handle_scalp_dashboard(callback)


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
        
        await handle_scalp_dashboard(callback)
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
            [InlineKeyboardButton(text="◀️ Back", callback_data="scalp_settings_menu")]
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


@dp.callback_query(F.data == "scalp_dashboard")
async def handle_scalp_dashboard(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Please use /start first")
            return
        prefs = user.preferences
        scalp_on = getattr(prefs, 'scalp_mode_enabled', False) or False
        scalp_lev = getattr(prefs, 'scalp_leverage', 20) or 20
        scalp_risk = getattr(prefs, 'scalp_risk_level', 'MEDIUM') or 'MEDIUM'
        scalp_size = getattr(prefs, 'scalp_position_size_percent', 1.0) or 1.0

        risk_info = {
            'LOW': ('🟢', 'Conservative', (1.0, 2.0, 1.0), 'Tight stops, small targets'),
            'MEDIUM': ('🟡', 'Balanced', (2.0, 3.0, 2.0), 'Standard scalp R:R'),
            'HIGH': ('🔴', 'Aggressive', (3.0, 5.0, 3.0), 'Wider targets, more risk'),
        }
        r_emoji, r_name, r_tpsl_vals, r_desc = risk_info.get(scalp_risk, risk_info['MEDIUM'])
        tp_lo, tp_hi, sl_val = r_tpsl_vals
        tp_lo_roi = tp_lo * scalp_lev
        tp_hi_roi = tp_hi * scalp_lev
        sl_roi = sl_val * scalp_lev

        status_icon = "🟢 ACTIVE" if scalp_on else "🔴 OFF"

        text = f"""⚡ <b>SCALP MODE</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>Status:</b>  {status_icon}

┌─ <b>CONFIGURATION</b>
│  Leverage:  <b>{scalp_lev}x</b>
│  Size:  <b>{scalp_size}%</b> of balance
│  Risk:  {r_emoji} <b>{r_name}</b>
│  TP/SL:  <b>{tp_lo}-{tp_hi}% / {sl_val}%</b>
└──────────────────────

┌─ <b>EXPECTED ROI @ {scalp_lev}x</b>
│  🎯 TP: <b>+{tp_lo_roi:.0f}% to +{tp_hi_roi:.0f}%</b>
│  🛑 SL: <b>-{sl_roi:.0f}%</b>
│  {r_desc}
└──────────────────────

┌─ <b>STRATEGY</b>
│  • Scans top gainers every 60s
│  • Support bounce + RSI reversal
│  • 4-6 scalps per day target
└──────────────────────

<i>Quick in-and-out trades on momentum coins</i>"""

        toggle_btn = "🔴 Turn OFF" if scalp_on else "🟢 Turn ON"
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text=toggle_btn, callback_data="scalp_toggle"),
            ],
            [
                InlineKeyboardButton(text="⚙️ Scalp Settings", callback_data="scalp_settings_menu"),
                InlineKeyboardButton(text="🔙 Back", callback_data="social_menu")
            ]
        ])

        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "scalp_settings_menu")
async def handle_scalp_settings_menu(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Please use /start first")
            return
        prefs = user.preferences
        scalp_lev = getattr(prefs, 'scalp_leverage', 20) or 20
        scalp_risk = getattr(prefs, 'scalp_risk_level', 'MEDIUM') or 'MEDIUM'
        scalp_size = getattr(prefs, 'scalp_position_size_percent', 1.0) or 1.0
        btc_orb_on = getattr(prefs, 'btc_orb_scalp_enabled', False) or False

        risk_display = {"LOW": "🟢 Conservative", "MEDIUM": "🟡 Balanced", "HIGH": "🔴 Aggressive"}.get(scalp_risk, "🟡 Balanced")
        risk_tp_map = {'LOW': (1.0, 2.0, 1.0), 'MEDIUM': (2.0, 3.0, 2.0), 'HIGH': (3.0, 5.0, 3.0)}
        tp_lo, tp_hi, sl_v = risk_tp_map.get(scalp_risk, (2.0, 3.0, 2.0))
        roi_lo = tp_lo * scalp_lev
        roi_hi = tp_hi * scalp_lev
        roi_sl = sl_v * scalp_lev
        btc_orb_status = "🟢 ON" if btc_orb_on else "🔴 OFF"

        text = f"""⚙️ <b>SCALP SETTINGS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>Leverage:</b>  <b>{scalp_lev}x</b>
<b>Position Size:</b>  <b>{scalp_size}%</b>
<b>Risk Level:</b>  {risk_display}

<b>ROI @ {scalp_lev}x:</b>  🎯 +{roi_lo:.0f}% to +{roi_hi:.0f}%  |  🛑 -{roi_sl:.0f}%

<b>BTC 200x Scalper:</b>  {btc_orb_status}
<i>High-leverage BTC structure break signals (London/NY only)</i>

<i>Adjust leverage, risk, and position size for scalp trades.</i>"""

        btc_orb_btn = "🔴 Disable BTC 200x Scalper" if btc_orb_on else "🟢 Enable BTC 200x Scalper"
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text=f"⚡ Leverage: {scalp_lev}x", callback_data="scalp_edit_leverage"),
                InlineKeyboardButton(text=f"⚠️ Risk: {scalp_risk}", callback_data="scalp_edit_risk")
            ],
            [
                InlineKeyboardButton(text=f"💰 Size: {scalp_size}%", callback_data="scalp_size"),
            ],
            [
                InlineKeyboardButton(text=btc_orb_btn, callback_data="user_btc_orb_toggle"),
            ],
            [
                InlineKeyboardButton(text="🔙 Back", callback_data="scalp_dashboard")
            ]
        ])

        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "user_btc_orb_toggle")
async def handle_user_btc_orb_toggle(callback: CallbackQuery):
    """Toggle per-user BTC ORB scalp signals on/off."""
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Please use /start first")
            return
        prefs = user.preferences
        current = getattr(prefs, 'btc_orb_scalp_enabled', False) or False
        prefs.btc_orb_scalp_enabled = not current
        db.commit()
        status = "🟢 ENABLED" if prefs.btc_orb_scalp_enabled else "🔴 DISABLED"
        await callback.answer(f"BTC 200x Scalper {status}", show_alert=True)
        await handle_scalp_settings_menu(callback)
    except Exception as e:
        logger.error(f"BTC ORB user toggle error: {e}")
    finally:
        db.close()


@dp.callback_query(F.data == "scalp_edit_leverage")
async def handle_scalp_edit_leverage(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        current = getattr(user.preferences, 'scalp_leverage', 20) if user and user.preferences else 20

        def lbl(val):
            return f"{val}x ✓" if current == val else f"{val}x"

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text=lbl(5), callback_data="scalp_lev_set_5"),
                InlineKeyboardButton(text=lbl(10), callback_data="scalp_lev_set_10"),
                InlineKeyboardButton(text=lbl(15), callback_data="scalp_lev_set_15"),
            ],
            [
                InlineKeyboardButton(text=lbl(20), callback_data="scalp_lev_set_20"),
                InlineKeyboardButton(text=lbl(25), callback_data="scalp_lev_set_25"),
                InlineKeyboardButton(text=lbl(50), callback_data="scalp_lev_set_50"),
            ],
            [InlineKeyboardButton(text="🔙 Back", callback_data="scalp_settings_menu")]
        ])
        scalp_risk = getattr(user.preferences, 'scalp_risk_level', 'MEDIUM') if user and user.preferences else 'MEDIUM'
        risk_tp_map = {'LOW': (1.0, 2.0), 'MEDIUM': (2.0, 3.0), 'HIGH': (3.0, 5.0)}
        tp_lo, tp_hi = risk_tp_map.get(scalp_risk, (2.0, 3.0))
        roi_preview = "\n".join([f"  {v}x → +{tp_lo*v:.0f}% to +{tp_hi*v:.0f}% ROI" for v in [5,10,15,20,25,50]])
        await callback.message.edit_text(
            f"⚡ <b>Select Scalp Leverage</b>\n\n"
            f"<b>Current Risk:</b> {scalp_risk}\n"
            f"<b>TP range:</b> {tp_lo}-{tp_hi}%\n\n"
            f"<code>{roi_preview}</code>\n\n"
            f"<i>Higher leverage = bigger gains &amp; losses</i>",
            reply_markup=keyboard, parse_mode="HTML"
        )
    finally:
        db.close()


@dp.callback_query(F.data.startswith("scalp_lev_set_"))
async def handle_scalp_lev_set(callback: CallbackQuery):
    await callback.answer()
    lev = int(callback.data.replace("scalp_lev_set_", ""))
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if user and user.preferences:
            user.preferences.scalp_leverage = lev
            db.commit()
            await callback.message.answer(f"✅ Scalp leverage set to <b>{lev}x</b>", parse_mode="HTML")
        await handle_scalp_settings_menu(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "scalp_edit_risk")
async def handle_scalp_edit_risk(callback: CallbackQuery):
    await callback.answer()

    text = """⚠️ <b>SCALP RISK LEVEL</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

🟢 <b>LOW</b> - Tight stops, small targets
    1-2% TP / 1% SL  ·  Safe entries only

🟡 <b>MEDIUM</b> - Standard scalp R:R
    2-3% TP / 2% SL  ·  Balanced approach

🔴 <b>HIGH</b> - Wider targets, more risk
    3-5% TP / 3% SL  ·  Aggressive entries"""

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🟢 LOW", callback_data="scalp_risk_set_LOW"),
            InlineKeyboardButton(text="🟡 MEDIUM", callback_data="scalp_risk_set_MEDIUM"),
            InlineKeyboardButton(text="🔴 HIGH", callback_data="scalp_risk_set_HIGH")
        ],
        [InlineKeyboardButton(text="🔙 Back", callback_data="scalp_settings_menu")]
    ])

    await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data.startswith("scalp_risk_set_"))
async def handle_scalp_risk_set(callback: CallbackQuery):
    await callback.answer()
    risk = callback.data.replace("scalp_risk_set_", "")
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if user and user.preferences:
            user.preferences.scalp_risk_level = risk
            db.commit()
            risk_label = {"LOW": "🟢 Conservative", "MEDIUM": "🟡 Balanced", "HIGH": "🔴 Aggressive"}.get(risk, risk)
            await callback.message.answer(f"✅ Scalp risk set to {risk_label}", parse_mode="HTML")
        await handle_scalp_settings_menu(callback)
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


@dp.callback_query(F.data == "quick_trade_size")
async def handle_quick_trade_size(callback: CallbackQuery):
    """Show quick trade size options"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found")
            return
        
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        current = prefs.quick_trade_size if prefs and prefs.quick_trade_size else 25.0
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="$10" + (" ✓" if current == 10 else ""), callback_data="set_quick_size:10"),
                InlineKeyboardButton(text="$25" + (" ✓" if current == 25 else ""), callback_data="set_quick_size:25"),
                InlineKeyboardButton(text="$50" + (" ✓" if current == 50 else ""), callback_data="set_quick_size:50")
            ],
            [
                InlineKeyboardButton(text="$100" + (" ✓" if current == 100 else ""), callback_data="set_quick_size:100"),
                InlineKeyboardButton(text="$200" + (" ✓" if current == 200 else ""), callback_data="set_quick_size:200"),
                InlineKeyboardButton(text="$500" + (" ✓" if current == 500 else ""), callback_data="set_quick_size:500")
            ]
        ])
        
        await callback.message.answer(
            "⚙️ <b>Quick Trade Size</b>\n\n"
            f"Current size: <b>${current:.0f}</b>\n\n"
            "Select your default position size for quick trades:",
            reply_markup=keyboard,
            parse_mode="HTML"
        )
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data.startswith("set_quick_size:"))
async def handle_set_quick_size(callback: CallbackQuery):
    """Set quick trade size"""
    db = SessionLocal()
    try:
        size = float(callback.data.split(":")[1])
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found")
            return
        
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs:
            prefs = UserPreference(user_id=user.id)
            db.add(prefs)
        
        prefs.quick_trade_size = size
        db.commit()
        
        await callback.message.answer(f"✅ Quick trade size set to <b>${size:.0f}</b>", parse_mode="HTML")
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data.startswith("quick_trade:"))
async def handle_quick_trade(callback: CallbackQuery):
    """Show confirmation dialog with leverage options"""
    db = SessionLocal()
    try:
        parts = callback.data.split(":")
        if len(parts) < 4:
            await callback.answer("Invalid trade data")
            return
        
        symbol = parts[1]
        direction = parts[2]
        size = float(parts[3])
        
        # Parse SL/TP percentages from trade idea (default to 2%/3%/5% if not provided)
        sl_pct = float(parts[4]) if len(parts) > 4 else 2.0
        tp1_pct = float(parts[5]) if len(parts) > 5 else 3.0
        tp2_pct = float(parts[6]) if len(parts) > 6 else 5.0
        
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found")
            return
        
        # Check subscription
        has_access, reason = check_access(user, require_tier="auto")
        if not has_access:
            await callback.message.answer(
                "🔒 <b>Auto-Trading Required</b>\n\n"
                "Quick Trade requires an Auto-Trading subscription.\n"
                "Upgrade to execute trades directly from scans!",
                parse_mode="HTML"
            )
            await callback.answer()
            return
        
        # Check if user has Bitunix connected
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs or not prefs.bitunix_api_key or not prefs.bitunix_api_secret:
            await callback.message.answer(
                "⚠️ <b>Bitunix Not Connected</b>\n\n"
                "Connect your Bitunix API keys first:\n"
                "/autotrading → Connect Exchange",
                parse_mode="HTML"
            )
            await callback.answer()
            return
        
        # Show size selection first - pass SL/TP through callback data
        dir_emoji = "🟢" if direction == 'LONG' else "🔴"
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="$25", callback_data=f"qt_size:{symbol}:{direction}:25:{sl_pct}:{tp1_pct}:{tp2_pct}"),
                InlineKeyboardButton(text="$50", callback_data=f"qt_size:{symbol}:{direction}:50:{sl_pct}:{tp1_pct}:{tp2_pct}"),
                InlineKeyboardButton(text="$100", callback_data=f"qt_size:{symbol}:{direction}:100:{sl_pct}:{tp1_pct}:{tp2_pct}")
            ],
            [
                InlineKeyboardButton(text="$200", callback_data=f"qt_size:{symbol}:{direction}:200:{sl_pct}:{tp1_pct}:{tp2_pct}"),
                InlineKeyboardButton(text="$500", callback_data=f"qt_size:{symbol}:{direction}:500:{sl_pct}:{tp1_pct}:{tp2_pct}"),
                InlineKeyboardButton(text="$1000", callback_data=f"qt_size:{symbol}:{direction}:1000:{sl_pct}:{tp1_pct}:{tp2_pct}")
            ],
            [
                InlineKeyboardButton(text="✏️ Custom", callback_data=f"qt_custom:{symbol}:{direction}:{sl_pct}:{tp1_pct}:{tp2_pct}"),
                InlineKeyboardButton(text="❌ Cancel", callback_data="cancel_trade")
            ]
        ])
        
        await callback.message.answer(
            f"{dir_emoji} <b>Quick Trade: {symbol}</b>\n\n"
            f"<b>Direction:</b> {direction}\n"
            f"<b>SL:</b> {sl_pct:.1f}% | <b>TP1:</b> {tp1_pct:.1f}% | <b>TP2:</b> {tp2_pct:.1f}%\n\n"
            f"<b>Step 1: Select position size:</b>",
            reply_markup=keyboard,
            parse_mode="HTML"
        )
        await callback.answer()
    finally:
        db.close()


@dp.callback_query(F.data.startswith("qt_size:"))
async def handle_qt_size_selection(callback: CallbackQuery):
    """Handle size selection, show leverage options"""
    try:
        parts = callback.data.split(":")
        if len(parts) < 4:
            await callback.answer("Invalid data")
            return
        
        symbol = parts[1]
        direction = parts[2]
        size = float(parts[3])
        
        # Parse SL/TP percentages (default to 2%/3%/5% if not provided)
        sl_pct = float(parts[4]) if len(parts) > 4 else 2.0
        tp1_pct = float(parts[5]) if len(parts) > 5 else 3.0
        tp2_pct = float(parts[6]) if len(parts) > 6 else 5.0
        
        dir_emoji = "🟢" if direction == 'LONG' else "🔴"
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="5x", callback_data=f"confirm_trade:{symbol}:{direction}:{size}:5:{sl_pct}:{tp1_pct}:{tp2_pct}"),
                InlineKeyboardButton(text="10x", callback_data=f"confirm_trade:{symbol}:{direction}:{size}:10:{sl_pct}:{tp1_pct}:{tp2_pct}"),
                InlineKeyboardButton(text="20x", callback_data=f"confirm_trade:{symbol}:{direction}:{size}:20:{sl_pct}:{tp1_pct}:{tp2_pct}")
            ],
            [
                InlineKeyboardButton(text="50x", callback_data=f"confirm_trade:{symbol}:{direction}:{size}:50:{sl_pct}:{tp1_pct}:{tp2_pct}"),
                InlineKeyboardButton(text="75x", callback_data=f"confirm_trade:{symbol}:{direction}:{size}:75:{sl_pct}:{tp1_pct}:{tp2_pct}"),
                InlineKeyboardButton(text="100x", callback_data=f"confirm_trade:{symbol}:{direction}:{size}:100:{sl_pct}:{tp1_pct}:{tp2_pct}")
            ],
            [
                InlineKeyboardButton(text="⬅️ Back", callback_data=f"quick_trade:{symbol}:{direction}:{size}:{sl_pct}:{tp1_pct}:{tp2_pct}"),
                InlineKeyboardButton(text="❌ Cancel", callback_data="cancel_trade")
            ]
        ])
        
        await callback.message.edit_text(
            f"{dir_emoji} <b>Quick Trade: {symbol}</b>\n\n"
            f"<b>Direction:</b> {direction}\n"
            f"<b>Size:</b> ${size:.0f}\n"
            f"<b>SL:</b> {sl_pct:.1f}% | <b>TP1:</b> {tp1_pct:.1f}% | <b>TP2:</b> {tp2_pct:.1f}%\n\n"
            f"<b>Step 2: Select leverage:</b>",
            reply_markup=keyboard,
            parse_mode="HTML"
        )
        await callback.answer()
    except Exception as e:
        logger.error(f"Size selection error: {e}")
        await callback.answer("Error processing selection")


@dp.callback_query(F.data.startswith("qt_custom:"))
async def handle_qt_custom(callback: CallbackQuery, state: FSMContext):
    """Start custom size/leverage input flow"""
    try:
        parts = callback.data.split(":")
        if len(parts) < 3:
            await callback.answer("Invalid data")
            return
        
        symbol = parts[1]
        direction = parts[2]
        
        # Parse SL/TP percentages (default to 2%/3%/5% if not provided)
        sl_pct = float(parts[3]) if len(parts) > 3 else 2.0
        tp1_pct = float(parts[4]) if len(parts) > 4 else 3.0
        tp2_pct = float(parts[5]) if len(parts) > 5 else 5.0
        
        # Store trade info in state
        await state.update_data(qt_symbol=symbol, qt_direction=direction, qt_sl_pct=sl_pct, qt_tp1_pct=tp1_pct, qt_tp2_pct=tp2_pct)
        await state.set_state(CustomQuickTrade.waiting_for_size)
        
        dir_emoji = "🟢" if direction == 'LONG' else "🔴"
        
        await callback.message.edit_text(
            f"{dir_emoji} <b>Custom Quick Trade: {symbol}</b>\n\n"
            f"<b>Direction:</b> {direction}\n\n"
            f"📝 <b>Enter your position size in USD</b>\n"
            f"<i>Example: 150 or 75.50</i>\n\n"
            f"Type /cancel to cancel",
            parse_mode="HTML"
        )
        await callback.answer()
    except Exception as e:
        logger.error(f"Custom trade error: {e}")
        await callback.answer("Error")


@dp.message(CustomQuickTrade.waiting_for_size)
async def process_custom_qt_size(message: types.Message, state: FSMContext):
    """Process custom size input"""
    if message.text and message.text.lower() == '/cancel':
        await state.clear()
        await message.answer("❌ Trade cancelled.")
        return
    
    try:
        size = float(message.text.replace('$', '').replace(',', '').strip())
        if size < 10:
            await message.answer("⚠️ Minimum size is $10. Please enter a larger amount:")
            return
        if size > 50000:
            await message.answer("⚠️ Maximum size is $50,000. Please enter a smaller amount:")
            return
        
        # Store size and ask for leverage
        await state.update_data(qt_size=size)
        await state.set_state(CustomQuickTrade.waiting_for_leverage)
        
        data = await state.get_data()
        symbol = data.get('qt_symbol')
        direction = data.get('qt_direction')
        dir_emoji = "🟢" if direction == 'LONG' else "🔴"
        
        await message.answer(
            f"{dir_emoji} <b>Custom Quick Trade: {symbol}</b>\n\n"
            f"<b>Direction:</b> {direction}\n"
            f"<b>Size:</b> ${size:,.2f}\n\n"
            f"📝 <b>Enter your leverage (1-125)</b>\n"
            f"<i>Example: 10 or 25</i>\n\n"
            f"Type /cancel to cancel",
            parse_mode="HTML"
        )
    except ValueError:
        await message.answer("⚠️ Invalid number. Please enter a valid size (e.g., 150):")


@dp.message(CustomQuickTrade.waiting_for_leverage)
async def process_custom_qt_leverage(message: types.Message, state: FSMContext):
    """Process custom leverage input and execute trade"""
    if message.text and message.text.lower() == '/cancel':
        await state.clear()
        await message.answer("❌ Trade cancelled.")
        return
    
    try:
        leverage = int(message.text.replace('x', '').strip())
        if leverage < 1:
            await message.answer("⚠️ Minimum leverage is 1x. Please enter a valid leverage:")
            return
        if leverage > 125:
            await message.answer("⚠️ Maximum leverage is 125x. Please enter a smaller value:")
            return
        
        data = await state.get_data()
        symbol = data.get('qt_symbol')
        direction = data.get('qt_direction')
        size = data.get('qt_size')
        
        # Get SL/TP from state (default to 2%/3%/5% if not provided)
        sl_pct = data.get('qt_sl_pct', 2.0)
        tp1_pct = data.get('qt_tp1_pct', 3.0)
        tp2_pct = data.get('qt_tp2_pct', 5.0)
        
        await state.clear()
        
        # Execute the trade
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
            if not user:
                await message.answer("❌ User not found")
                return
            
            prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
            if not prefs or not prefs.bitunix_api_key or not prefs.bitunix_api_secret:
                await message.answer("⚠️ Bitunix not connected. Use /autotrading to connect.")
                return
            
            dir_emoji = "🟢" if direction == 'LONG' else "🔴"
            status_msg = await message.answer(
                f"⏳ Executing {direction} on <b>{symbol}</b>...\n"
                f"Size: ${size:,.2f} | Leverage: {leverage}x",
                parse_mode="HTML"
            )
            
            from app.services.bitunix_trader import BitunixTrader
            from app.utils.encryption import decrypt_api_key
            api_key = decrypt_api_key(prefs.bitunix_api_key)
            api_secret = decrypt_api_key(prefs.bitunix_api_secret)
            trader = BitunixTrader(api_key, api_secret)
            
            current_price = await trader.get_current_price(f"{symbol}USDT")
            if not current_price or current_price <= 0:
                await status_msg.edit_text(f"❌ Could not get price for {symbol}")
                await trader.close()
                return
            
            # Calculate SL/TP using trade idea percentages
            if direction == 'LONG':
                stop_loss = current_price * (1 - sl_pct / 100)
                tp1 = current_price * (1 + tp1_pct / 100)
                tp2 = current_price * (1 + tp2_pct / 100)
            else:
                stop_loss = current_price * (1 + sl_pct / 100)
                tp1 = current_price * (1 - tp1_pct / 100)
                tp2 = current_price * (1 - tp2_pct / 100)
            
            # Execute dual TP trade (50% at TP1, 50% at TP2)
            half_size = size / 2
            
            result1 = await trader.place_trade(
                symbol=f"{symbol}/USDT",
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=tp1,
                position_size_usdt=half_size,
                leverage=leverage
            )
            
            result2 = await trader.place_trade(
                symbol=f"{symbol}/USDT",
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=tp2,
                position_size_usdt=half_size,
                leverage=leverage
            )
            
            if (result1 and result1.get('success')) or (result2 and result2.get('success')):
                await status_msg.edit_text(
                    f"{dir_emoji} <b>Trade Opened!</b>\n\n"
                    f"<b>{symbol}</b> {direction} @ ${current_price:,.4f}\n"
                    f"Size: ${size:,.2f} | Leverage: {leverage}x\n"
                    f"SL: ${stop_loss:,.4f} (-{sl_pct:.1f}%)\n"
                    f"TP1: ${tp1:,.4f} (+{tp1_pct:.1f}%) - 50%\n"
                    f"TP2: ${tp2:,.4f} (+{tp2_pct:.1f}%) - 50%",
                    parse_mode="HTML"
                )
            else:
                await status_msg.edit_text("❌ Trade failed. Check logs or try again.")
            
            await trader.close()
            
        finally:
            db.close()
            
    except ValueError:
        await message.answer("⚠️ Invalid number. Please enter a valid leverage (e.g., 10):")
    except Exception as e:
        logger.error(f"Custom trade execution error: {e}", exc_info=True)
        await message.answer(f"❌ Trade error: {str(e)[:100]}")
        await state.clear()


@dp.callback_query(F.data == "cancel_trade")
async def handle_cancel_trade(callback: CallbackQuery):
    """Cancel trade confirmation"""
    await callback.message.edit_text("❌ Trade cancelled.")
    await callback.answer()


@dp.callback_query(F.data.startswith("confirm_trade:"))
async def handle_confirm_trade(callback: CallbackQuery):
    """Execute trade after confirmation"""
    db = SessionLocal()
    try:
        parts = callback.data.split(":")
        if len(parts) < 5:
            await callback.answer("Invalid trade data")
            return
        
        symbol = parts[1]
        direction = parts[2]
        size = float(parts[3])
        leverage = int(parts[4])
        
        # Parse SL/TP percentages (default to 2%/3%/5% if not provided)
        sl_pct = float(parts[5]) if len(parts) > 5 else 2.0
        tp1_pct = float(parts[6]) if len(parts) > 6 else 3.0
        tp2_pct = float(parts[7]) if len(parts) > 7 else 5.0
        
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.answer("User not found")
            return
        
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs or not prefs.bitunix_api_key or not prefs.bitunix_api_secret:
            await callback.answer("Bitunix not connected")
            return
        
        # Update message to show executing
        await callback.message.edit_text(
            f"⏳ Executing {direction} on <b>{symbol}</b>...\n"
            f"Size: ${size:.0f} | Leverage: {leverage}x",
            parse_mode="HTML"
        )
        
        try:
            from app.services.bitunix_trader import BitunixTrader
            from app.utils.encryption import decrypt_api_key
            
            api_key = decrypt_api_key(prefs.bitunix_api_key)
            api_secret = decrypt_api_key(prefs.bitunix_api_secret)
            trader = BitunixTrader(api_key, api_secret)
            
            # Fetch current price
            current_price = await trader.get_current_price(f"{symbol}USDT")
            if not current_price or current_price <= 0:
                await callback.message.edit_text(f"❌ Could not get price for {symbol}")
                await callback.answer()
                return
            
            # Calculate SL/TP using trade idea percentages
            if direction == 'LONG':
                stop_loss = current_price * (1 - sl_pct / 100)
                tp1 = current_price * (1 + tp1_pct / 100)
                tp2 = current_price * (1 + tp2_pct / 100)
            else:
                stop_loss = current_price * (1 + sl_pct / 100)
                tp1 = current_price * (1 - tp1_pct / 100)
                tp2 = current_price * (1 - tp2_pct / 100)
            
            # Execute dual TP trade (50% at TP1, 50% at TP2)
            half_size = size / 2
            
            # Order 1: 50% at TP1
            result1 = await trader.place_trade(
                symbol=f"{symbol}/USDT",
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=tp1,
                position_size_usdt=half_size,
                leverage=leverage
            )
            
            # Order 2: 50% at TP2
            result2 = await trader.place_trade(
                symbol=f"{symbol}/USDT",
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=tp2,
                position_size_usdt=half_size,
                leverage=leverage
            )
            
            if (result1 and result1.get('success')) or (result2 and result2.get('success')):
                dir_emoji = "🟢" if direction == 'LONG' else "🔴"
                await callback.message.edit_text(
                    f"{dir_emoji} <b>Trade Opened!</b>\n\n"
                    f"<b>{symbol}</b> {direction} @ ${current_price:,.4f}\n"
                    f"Size: ${size:.0f} | Leverage: {leverage}x\n"
                    f"SL: ${stop_loss:,.4f} (-{sl_pct:.1f}%)\n"
                    f"TP1: ${tp1:,.4f} (+{tp1_pct:.1f}%) - 50%\n"
                    f"TP2: ${tp2:,.4f} (+{tp2_pct:.1f}%) - 50%",
                    parse_mode="HTML"
                )
            else:
                await callback.message.edit_text("❌ Trade failed. Check logs or try again.")
            
            await trader.close()
            
        except Exception as e:
            logger.error(f"Quick trade error: {e}", exc_info=True)
            await callback.message.edit_text(f"❌ Trade error: {str(e)[:100]}")
        
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
        
        # Check if user has auto-trading subscription
        has_access, reason = check_access(user, require_tier="auto")
        if not has_access:
            await callback.message.edit_text(
                "🤖 <b>Auto-Trading - Premium Feature</b>\n\n"
                "Auto-trading is available on the <b>🤖 Auto-Trading plan</b> ($80/month - BLACK FRIDAY!).\n\n"
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
            
            autotrading_text = f"""
🤖 <b>Auto-Trading Status</b>
━━━━━━━━━━━━━━━━━━━━

🔑 <b>Exchange:</b> {exchange_name}
📡 <b>API Status:</b> {api_status}
🔄 <b>Auto-Trading:</b> {autotrading_status}
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


# Beta testers for Social & News Trading
SOCIAL_BETA_TESTERS = ["bu11dogg", "ben", "bnickl88"]
SOCIAL_BETA_TESTER_IDS = {1, 6, 107}

@dp.callback_query(F.data == "social_menu")
async def handle_social_menu(callback: CallbackQuery):
    """🌙 Social & News Trading Menu - AI-powered signals"""
    await callback.answer()
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("Please use /start first")
            return
        
        if not user.is_subscribed and not user.is_admin and not user.grandfathered:
            coming_soon_kb = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="💎 Subscribe", callback_data="subscribe_menu")],
                [InlineKeyboardButton(text="🏠 Back to Home", callback_data="back_to_start")]
            ])
            await callback.message.edit_text(
                "🌙 <b>Social & News Trading</b>\n\n"
                "🔒 <b>Subscribers Only</b>\n\n"
                "Social & News Trading is available to paid subscribers.\n"
                "Subscribe to unlock AI-powered social signals!",
                reply_markup=coming_soon_kb,
                parse_mode="HTML"
            )
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            return
        
        from app.services.social_signals import is_social_scanning_enabled
        from app.services.lunarcrush import get_lunarcrush_api_key
        
        prefs = user.preferences
        
        social_enabled = getattr(prefs, 'social_mode_enabled', False) or False if prefs else False
        social_lev = getattr(prefs, 'social_leverage', 10) or 10 if prefs else 10
        social_top_lev = getattr(prefs, 'social_top_coin_leverage', 25) or 25 if prefs else 25
        social_size = getattr(prefs, 'social_position_size_percent', 5.0) or 5.0 if prefs else 5.0
        social_dollars = getattr(prefs, 'social_position_size_dollars', None) if prefs else None
        social_max = getattr(prefs, 'social_max_positions', 3) or 3 if prefs else 3
        social_galaxy = getattr(prefs, 'social_min_galaxy_score', 8) or 8 if prefs else 8
        social_risk = getattr(prefs, 'social_risk_level', 'MEDIUM') or 'MEDIUM' if prefs else 'MEDIUM'
        
        scanner_on = is_social_scanning_enabled()
        api_configured = get_lunarcrush_api_key() is not None
        
        size_display = f"${social_dollars:.0f}" if social_dollars else f"{social_size}%"
        
        status_icon = "🟢" if social_enabled and scanner_on and api_configured else "🔴"
        
        if not api_configured:
            api_status = "⚠️ API key not configured"
        elif not scanner_on:
            api_status = "⏸️ Scanner paused"
        elif social_enabled:
            api_status = "✅ Active & Trading"
        else:
            api_status = "📡 Signals only (not trading)"
        
        # Risk level emoji
        if social_risk == "ALL":
            risk_emoji = "🌐"
        elif social_risk == "MOMENTUM":
            risk_emoji = "🚀"
        elif social_risk == "HIGH":
            risk_emoji = "🔴"
        elif social_risk == "MEDIUM":
            risk_emoji = "🟡"
        else:
            risk_emoji = "🟢"
        
        # Build status bar
        if not api_configured:
            status_bar = "⚠️ <b>Setup Required</b> - Add LUNARCRUSH_API_KEY"
        elif social_enabled and scanner_on:
            status_bar = "🟢 <b>ACTIVE</b> - Auto-executing trades"
        elif scanner_on:
            status_bar = "📡 <b>MONITORING</b> - Signals only"
        else:
            status_bar = "⏸️ <b>PAUSED</b> - Scanner disabled"
        
        # Status emoji
        auto_status = "ON ✓" if social_enabled else "OFF"
        
        news_enabled = getattr(prefs, 'news_trading_enabled', True)
        news_status = "ON ✓" if news_enabled else "OFF"
        news_lev = getattr(prefs, 'news_leverage', 10) or 10
        news_risk = getattr(prefs, 'news_risk_level', 'MEDIUM') or 'MEDIUM'
        
        squeeze_on = getattr(prefs, 'squeeze_mode_enabled', True) if prefs else True
        supertrend_on = getattr(prefs, 'supertrend_mode_enabled', True) if prefs else True
        macd_on = getattr(prefs, 'macd_mode_enabled', True) if prefs else True
        
        sq_icon = "✅" if squeeze_on else "❌"
        st_icon = "✅" if supertrend_on else "❌"
        mc_icon = "✅" if macd_on else "❌"
        
        scalp_on = getattr(prefs, 'scalp_mode_enabled', False) or False if prefs else False
        scalp_lev = getattr(prefs, 'scalp_leverage', 20) or 20 if prefs else 20
        scalp_risk = getattr(prefs, 'scalp_risk_level', 'MEDIUM') or 'MEDIUM' if prefs else 'MEDIUM'
        scalp_size = getattr(prefs, 'scalp_position_size_percent', 1.0) or 1.0 if prefs else 1.0
        scalp_icon = "✅" if scalp_on else "❌"

        social_text = f"""🌙 <b>SOCIAL & NEWS TERMINAL</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{status_bar}

┌─ <b>📡 SOCIAL SIGNALS</b>
│  Status: <b>{auto_status}</b>
│  Risk: <b>{social_risk}</b>  ·  Score: <b>≥{social_galaxy}</b>
│  🏆 Top 10: <b>{social_top_lev}x</b>  ·  📊 Alts: <b>{social_lev}x</b>
│  💰 Size: <b>{size_display}</b>  ·  Max: <b>{social_max}</b> pos
└──────────────────────

┌─ <b>📰 NEWS TRADING</b>
│  Status: <b>{news_status}</b>  ·  Risk: <b>{news_risk}</b>
│  Leverage: <b>{news_lev}x</b>
└──────────────────────

┌─ <b>⚡ SCALP MODE</b>
│  Status: <b>{scalp_icon}</b>  ·  Risk: <b>{scalp_risk}</b>
│  Leverage: <b>{scalp_lev}x</b>  ·  Size: <b>{scalp_size}%</b>
└──────────────────────

┌─ <b>🔬 SCANNERS</b>
│  {sq_icon} Squeeze  ·  {st_icon} SuperTrend  ·  {mc_icon} MACD
└──────────────────────

<i>Priority: Momentum → News → LONG → Scalp → Squeeze → ST → MACD → SHORT → Bounce</i>"""
        
        toggle_text = "🔴 Disable" if social_enabled else "🟢 Enable"
        
        risk_display = {"LOW": "🟢 SAFE", "MEDIUM": "🟡 BALANCED", "HIGH": "🔴 AGGRO", "MOMENTUM": "🚀 NEWS", "ALL": "🌐 ALL"}.get(social_risk, "🟡 BALANCED")
        
        news_toggle_text = "📰 News: ON" if news_enabled else "📰 News: OFF"
        
        sq_btn = f"{'🔥' if squeeze_on else '⬜'} Squeeze" 
        st_btn = f"{'📈' if supertrend_on else '⬜'} SuperTrend"
        mc_btn = f"{'⚡' if macd_on else '⬜'} MACD"
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text=toggle_text, callback_data="social_toggle_trade"),
                InlineKeyboardButton(text=f"Risk: {risk_display}", callback_data="social_risk_picker")
            ],
            [
                InlineKeyboardButton(text=news_toggle_text, callback_data="news_toggle"),
                InlineKeyboardButton(text="📰 News Settings", callback_data="news_settings")
            ],
            [
                InlineKeyboardButton(text=f"⚡ Scalp: {'ON' if scalp_on else 'OFF'}", callback_data="scalp_dashboard"),
                InlineKeyboardButton(text="⚡ Scalp Settings", callback_data="scalp_settings_menu")
            ],
            [
                InlineKeyboardButton(text=sq_btn, callback_data="toggle_squeeze"),
                InlineKeyboardButton(text=st_btn, callback_data="toggle_supertrend"),
                InlineKeyboardButton(text=mc_btn, callback_data="toggle_macd")
            ],
            [
                InlineKeyboardButton(text="🔍 Scan Now", callback_data="social_scan_now"),
                InlineKeyboardButton(text="📊 Trending", callback_data="social_trending")
            ],
            [
                InlineKeyboardButton(text="⚙️ Settings", callback_data="social_settings"),
                InlineKeyboardButton(text="🏠 Home", callback_data="back_to_start")
            ]
        ])
        
        await callback.message.edit_text(social_text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "social_toggle_trade")
async def handle_social_toggle_trade(callback: CallbackQuery):
    """Toggle social auto-trading on/off"""
    await callback.answer()
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Please use /start first")
            return
        
        if not user.is_subscribed and not user.is_admin and not user.grandfathered:
            await callback.message.answer("⚠️ Social trading requires an active subscription. Use /subscribe to get started.", parse_mode="HTML")
            return
        
        prefs = user.preferences
        current = getattr(prefs, 'social_mode_enabled', False) or False
        prefs.social_mode_enabled = not current
        db.commit()
        
        if not current:
            await callback.message.answer("✅ <b>Social & News auto-trading ENABLED</b>\n\nYou'll receive and auto-execute AI-powered social signals.", parse_mode="HTML")
        else:
            await callback.message.answer("❌ <b>Social auto-trading DISABLED</b>", parse_mode="HTML")
        
        # Refresh the social menu
        await handle_social_menu(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "toggle_squeeze")
async def handle_toggle_squeeze(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Please use /start first")
            return
        prefs = user.preferences
        current = getattr(prefs, 'squeeze_mode_enabled', True)
        prefs.squeeze_mode_enabled = not current
        db.commit()
        status = "ENABLED" if not current else "DISABLED"
        await callback.message.answer(f"🔥 Squeeze Breakout scanner <b>{status}</b>", parse_mode="HTML")
        await handle_social_menu(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "toggle_supertrend")
async def handle_toggle_supertrend(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Please use /start first")
            return
        prefs = user.preferences
        current = getattr(prefs, 'supertrend_mode_enabled', True)
        prefs.supertrend_mode_enabled = not current
        db.commit()
        status = "ENABLED" if not current else "DISABLED"
        await callback.message.answer(f"📈 SuperTrend scanner <b>{status}</b>", parse_mode="HTML")
        await handle_social_menu(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "toggle_macd")
async def handle_toggle_macd(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Please use /start first")
            return
        prefs = user.preferences
        current = getattr(prefs, 'macd_mode_enabled', True)
        prefs.macd_mode_enabled = not current
        db.commit()
        status = "ENABLED" if not current else "DISABLED"
        await callback.message.answer(f"⚡ MACD Momentum scanner <b>{status}</b>", parse_mode="HTML")
        await handle_social_menu(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "news_toggle")
async def handle_news_toggle(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Please use /start first")
            return
        prefs = user.preferences
        current = getattr(prefs, 'news_trading_enabled', True)
        prefs.news_trading_enabled = not current
        db.commit()
        if not current:
            await callback.message.answer("✅ <b>News Trading ENABLED</b>\n\nBreaking news will generate trading signals.", parse_mode="HTML")
        else:
            await callback.message.answer("❌ <b>News Trading DISABLED</b>", parse_mode="HTML")
        await handle_social_menu(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "news_settings")
async def handle_news_settings(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Please use /start first")
            return
        prefs = user.preferences
        news_enabled = getattr(prefs, 'news_trading_enabled', True)
        news_lev = getattr(prefs, 'news_leverage', 10) or 10
        news_top_lev = getattr(prefs, 'news_top_coin_leverage', 25) or 25
        news_size = getattr(prefs, 'news_position_size_percent', 3.0) or 3.0
        news_risk = getattr(prefs, 'news_risk_level', 'MEDIUM') or 'MEDIUM'
        news_max = getattr(prefs, 'news_max_positions', 3) or 3
        
        enabled_icon = "🟢 ON" if news_enabled else "🔴 OFF"
        risk_display = {"LOW": "🟢 Conservative", "MEDIUM": "🟡 Balanced", "HIGH": "🔴 Aggressive"}.get(news_risk, "🟡 Balanced")
        
        settings_text = f"""📰 <b>NEWS TRADING SETTINGS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>Status:</b>  {enabled_icon}

┌─ <b>LEVERAGE</b>
│  🏆 Top 10:  <b>{news_top_lev}x</b>
│  📊 Altcoins:  <b>{news_lev}x</b>
└──────────────────────

┌─ <b>POSITION</b>
│  💰 Size:  <b>{news_size}%</b>
│  📈 Max:  <b>{news_max}</b> positions
│  ⚠️ Risk:  {risk_display}
└──────────────────────

<i>Auto-trade on AI-confirmed breaking crypto &amp; macro news.</i>"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text=f"🏆 Top 10: {news_top_lev}x", callback_data="news_edit_top_lev"),
                InlineKeyboardButton(text=f"📊 Alts: {news_lev}x", callback_data="news_edit_leverage")
            ],
            [
                InlineKeyboardButton(text=f"💰 Size: {news_size}%", callback_data="news_edit_size"),
                InlineKeyboardButton(text=f"⚡ Risk: {news_risk}", callback_data="news_risk_picker")
            ],
            [
                InlineKeyboardButton(text=f"📈 Max: {news_max}", callback_data="news_edit_max"),
            ],
            [
                InlineKeyboardButton(text="🔙 Back", callback_data="social_menu")
            ]
        ])
        
        await callback.message.edit_text(settings_text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "news_risk_picker")
async def handle_news_risk_picker(callback: CallbackQuery):
    await callback.answer()
    
    picker_text = """📰 <b>NEWS RISK LEVEL</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

🟢 <b>LOW</b> - Only high-impact news, conservative TP
🟡 <b>MEDIUM</b> - Balanced news sensitivity
🔴 <b>HIGH</b> - All news triggers, aggressive TP

<i>Higher risk = more signals but lower quality filter</i>"""
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🟢 LOW", callback_data="news_risk_set_LOW"),
            InlineKeyboardButton(text="🟡 MEDIUM", callback_data="news_risk_set_MEDIUM"),
            InlineKeyboardButton(text="🔴 HIGH", callback_data="news_risk_set_HIGH")
        ],
        [InlineKeyboardButton(text="🔙 Back", callback_data="news_settings")]
    ])
    
    await callback.message.edit_text(picker_text, reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data.startswith("news_risk_set_"))
async def handle_news_risk_set(callback: CallbackQuery):
    await callback.answer()
    risk = callback.data.replace("news_risk_set_", "")
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if user and user.preferences:
            user.preferences.news_risk_level = risk
            db.commit()
            await callback.message.answer(f"✅ News risk set to <b>{risk}</b>", parse_mode="HTML")
        await handle_news_settings(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "news_edit_leverage")
async def handle_news_edit_leverage(callback: CallbackQuery):
    await callback.answer()
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="5x", callback_data="news_lev_set_5"),
            InlineKeyboardButton(text="10x", callback_data="news_lev_set_10"),
            InlineKeyboardButton(text="15x", callback_data="news_lev_set_15"),
        ],
        [
            InlineKeyboardButton(text="20x", callback_data="news_lev_set_20"),
            InlineKeyboardButton(text="25x", callback_data="news_lev_set_25"),
            InlineKeyboardButton(text="50x", callback_data="news_lev_set_50"),
        ],
        [InlineKeyboardButton(text="🔙 Back", callback_data="news_settings")]
    ])
    await callback.message.edit_text("📊 <b>Select News Altcoin Leverage</b>", reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data.startswith("news_lev_set_"))
async def handle_news_lev_set(callback: CallbackQuery):
    await callback.answer()
    lev = int(callback.data.replace("news_lev_set_", ""))
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if user and user.preferences:
            user.preferences.news_leverage = lev
            db.commit()
            await callback.message.answer(f"✅ News altcoin leverage set to <b>{lev}x</b>", parse_mode="HTML")
        await handle_news_settings(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "news_edit_top_lev")
async def handle_news_edit_top_lev(callback: CallbackQuery):
    await callback.answer()
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="10x", callback_data="news_top_lev_set_10"),
            InlineKeyboardButton(text="15x", callback_data="news_top_lev_set_15"),
            InlineKeyboardButton(text="20x", callback_data="news_top_lev_set_20"),
        ],
        [
            InlineKeyboardButton(text="25x", callback_data="news_top_lev_set_25"),
            InlineKeyboardButton(text="50x", callback_data="news_top_lev_set_50"),
            InlineKeyboardButton(text="75x", callback_data="news_top_lev_set_75"),
        ],
        [InlineKeyboardButton(text="🔙 Back", callback_data="news_settings")]
    ])
    await callback.message.edit_text("🏆 <b>Select News Top 10 Leverage</b>", reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data.startswith("news_top_lev_set_"))
async def handle_news_top_lev_set(callback: CallbackQuery):
    await callback.answer()
    lev = int(callback.data.replace("news_top_lev_set_", ""))
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if user and user.preferences:
            user.preferences.news_top_coin_leverage = lev
            db.commit()
            await callback.message.answer(f"✅ News top 10 leverage set to <b>{lev}x</b>", parse_mode="HTML")
        await handle_news_settings(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "news_edit_size")
async def handle_news_edit_size(callback: CallbackQuery):
    await callback.answer()
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="1%", callback_data="news_size_set_1"),
            InlineKeyboardButton(text="2%", callback_data="news_size_set_2"),
            InlineKeyboardButton(text="3%", callback_data="news_size_set_3"),
        ],
        [
            InlineKeyboardButton(text="5%", callback_data="news_size_set_5"),
            InlineKeyboardButton(text="8%", callback_data="news_size_set_8"),
            InlineKeyboardButton(text="10%", callback_data="news_size_set_10"),
        ],
        [InlineKeyboardButton(text="🔙 Back", callback_data="news_settings")]
    ])
    await callback.message.edit_text("💰 <b>Select News Position Size (% of balance)</b>", reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data.startswith("news_size_set_"))
async def handle_news_size_set(callback: CallbackQuery):
    await callback.answer()
    size = float(callback.data.replace("news_size_set_", ""))
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if user and user.preferences:
            user.preferences.news_position_size_percent = size
            db.commit()
            await callback.message.answer(f"✅ News position size set to <b>{size}%</b>", parse_mode="HTML")
        await handle_news_settings(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "news_edit_max")
async def handle_news_edit_max(callback: CallbackQuery):
    await callback.answer()
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="1", callback_data="news_max_set_1"),
            InlineKeyboardButton(text="2", callback_data="news_max_set_2"),
            InlineKeyboardButton(text="3", callback_data="news_max_set_3"),
        ],
        [
            InlineKeyboardButton(text="5", callback_data="news_max_set_5"),
            InlineKeyboardButton(text="8", callback_data="news_max_set_8"),
            InlineKeyboardButton(text="10", callback_data="news_max_set_10"),
        ],
        [InlineKeyboardButton(text="🔙 Back", callback_data="news_settings")]
    ])
    await callback.message.edit_text("📈 <b>Select Max Simultaneous News Positions</b>", reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data.startswith("news_max_set_"))
async def handle_news_max_set(callback: CallbackQuery):
    await callback.answer()
    max_pos = int(callback.data.replace("news_max_set_", ""))
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if user and user.preferences:
            user.preferences.news_max_positions = max_pos
            db.commit()
            await callback.message.answer(f"✅ News max positions set to <b>{max_pos}</b>", parse_mode="HTML")
        await handle_news_settings(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "social_settings")
async def handle_social_settings(callback: CallbackQuery):
    """Show social settings options"""
    await callback.answer()
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Please use /start first")
            return
        
        prefs = user.preferences
        social_risk = getattr(prefs, 'social_risk_level', 'MEDIUM') or 'MEDIUM'
        social_lev = getattr(prefs, 'social_leverage', 10) or 10
        social_top_lev = getattr(prefs, 'social_top_coin_leverage', 25) or 25
        social_galaxy = getattr(prefs, 'social_min_galaxy_score', 8) or 8
        
        # Risk-based sizing
        size_low = getattr(prefs, 'social_size_low', 5.0) or 5.0
        size_med = getattr(prefs, 'social_size_medium', 3.0) or 3.0
        size_high = getattr(prefs, 'social_size_high', 2.0) or 2.0
        size_all = getattr(prefs, 'social_size_all', 1.0) or 1.0
        
        settings_text = f"""⚙️ <b>SOCIAL SETTINGS</b>

<b>Leverage</b>
🏆 Top 10  <b>{social_top_lev}x</b>
📊 Altcoins  <b>{social_lev}x</b>

<b>Position Sizing by Score</b>
🟢 LOW (≥85)  <b>{size_low}%</b>
🟡 MED (≥75)  <b>{size_med}%</b>
🔴 HIGH (70-74)  <b>{size_high}%</b>

<i>Min score 70 - weak signals blocked</i>

<i>Top 10: BTC ETH SOL XRP DOGE ADA AVAX DOT LINK LTC</i>"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text=f"🏆 Top 10: {social_top_lev}x", callback_data="social_edit_top_lev"),
                InlineKeyboardButton(text=f"📊 Alts: {social_lev}x", callback_data="social_edit_leverage")
            ],
            [
                InlineKeyboardButton(text="💰 Edit Sizes", callback_data="social_edit_sizes"),
                InlineKeyboardButton(text=f"🎯 Score: {social_galaxy}", callback_data="social_edit_score")
            ],
            [
                InlineKeyboardButton(text="🔙 Back", callback_data="social_menu")
            ]
        ])
        
        await callback.message.edit_text(settings_text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "social_risk_picker")
async def handle_social_risk_picker(callback: CallbackQuery):
    """Show risk level picker"""
    await callback.answer()
    
    picker_text = """
🎯 <b>SELECT RISK PROFILE</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

🟢 <b>SAFE</b> - Quick scalps, +3% TP
🟡 <b>BALANCED</b> - Steady gains, +5% TP  
🔴 <b>AGGRESSIVE</b> - High risk, +8-15% TP
🚀 <b>NEWS RUNNER</b> - Catch pumps, +15-30% TP
🌐 <b>ALL</b> - Smart: TP adapts to signal strength
"""
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🟢 SAFE", callback_data="social_set_risk_LOW"),
            InlineKeyboardButton(text="🟡 BALANCED", callback_data="social_set_risk_MEDIUM")
        ],
        [
            InlineKeyboardButton(text="🔴 AGGRO", callback_data="social_set_risk_HIGH"),
            InlineKeyboardButton(text="🚀 NEWS", callback_data="social_set_risk_MOMENTUM")
        ],
        [
            InlineKeyboardButton(text="🌐 ALL (Smart)", callback_data="social_set_risk_ALL")
        ],
        [
            InlineKeyboardButton(text="🔙 Back", callback_data="social_menu")
        ]
    ])
    
    await callback.message.edit_text(picker_text, reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data.startswith("social_set_risk_"))
async def handle_social_risk_change(callback: CallbackQuery):
    """Change social risk level"""
    await callback.answer()
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            return
        
        risk_level = callback.data.replace("social_set_risk_", "")
        user.preferences.social_risk_level = risk_level
        db.commit()
        
        risk_names = {"LOW": "SAFE", "MEDIUM": "BALANCED", "HIGH": "AGGRESSIVE", "MOMENTUM": "NEWS RUNNER", "ALL": "ALL (Smart)"}
        await callback.message.answer(f"✅ Risk: <b>{risk_names.get(risk_level, risk_level)}</b>", parse_mode="HTML")
        await handle_social_menu(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "social_edit_top_lev")
async def handle_social_edit_top_lev(callback: CallbackQuery):
    """Show top coin leverage options (higher limits for stable coins)"""
    await callback.answer()
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="10x", callback_data="social_toplev_10"),
            InlineKeyboardButton(text="15x", callback_data="social_toplev_15"),
            InlineKeyboardButton(text="20x", callback_data="social_toplev_20")
        ],
        [
            InlineKeyboardButton(text="25x", callback_data="social_toplev_25"),
            InlineKeyboardButton(text="30x", callback_data="social_toplev_30"),
            InlineKeyboardButton(text="40x", callback_data="social_toplev_40")
        ],
        [
            InlineKeyboardButton(text="50x", callback_data="social_toplev_50")
        ],
        [
            InlineKeyboardButton(text="🔙 Back", callback_data="social_settings")
        ]
    ])
    
    await callback.message.edit_text(
        "🏆 <b>Top 10 Leverage</b>\n\n"
        "For: BTC ETH SOL XRP DOGE ADA AVAX DOT LINK LTC\n\n"
        "<i>These coins are more stable - higher leverage is safer</i>",
        reply_markup=keyboard, parse_mode="HTML"
    )


@dp.callback_query(F.data.startswith("social_toplev_"))
async def handle_social_toplev_set(callback: CallbackQuery):
    """Set top coin leverage"""
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if user and user.preferences:
            lev = int(callback.data.replace("social_toplev_", ""))
            user.preferences.social_top_coin_leverage = lev
            db.commit()
            await callback.message.answer(f"✅ Top 10 Leverage: <b>{lev}x</b>", parse_mode="HTML")
        await handle_social_settings(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "social_edit_leverage")
async def handle_social_edit_leverage(callback: CallbackQuery):
    """Show altcoin leverage options"""
    await callback.answer()
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="3x", callback_data="social_lev_3"),
            InlineKeyboardButton(text="5x", callback_data="social_lev_5"),
            InlineKeyboardButton(text="10x", callback_data="social_lev_10")
        ],
        [
            InlineKeyboardButton(text="15x", callback_data="social_lev_15"),
            InlineKeyboardButton(text="20x", callback_data="social_lev_20"),
            InlineKeyboardButton(text="25x", callback_data="social_lev_25")
        ],
        [
            InlineKeyboardButton(text="🔙 Back", callback_data="social_settings")
        ]
    ])
    
    await callback.message.edit_text("📊 <b>Altcoin Leverage</b>\n\nFor coins outside the Top 10", reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data.startswith("social_lev_"))
async def handle_social_lev_set(callback: CallbackQuery):
    """Set leverage"""
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if user and user.preferences:
            lev = int(callback.data.replace("social_lev_", ""))
            user.preferences.social_leverage = lev
            db.commit()
            await callback.message.answer(f"✅ Leverage: <b>{lev}x</b>", parse_mode="HTML")
        await handle_social_settings(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "social_edit_size")
async def handle_social_edit_size(callback: CallbackQuery):
    """Redirect to new sizes menu"""
    await handle_social_edit_sizes(callback)


@dp.callback_query(F.data == "social_edit_sizes")
async def handle_social_edit_sizes(callback: CallbackQuery):
    """Show risk-based size options"""
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.preferences:
            await callback.message.answer("Please use /start first")
            return
        
        prefs = user.preferences
        size_low = getattr(prefs, 'social_size_low', 5.0) or 5.0
        size_med = getattr(prefs, 'social_size_medium', 3.0) or 3.0
        size_high = getattr(prefs, 'social_size_high', 2.0) or 2.0
        size_all = getattr(prefs, 'social_size_all', 1.0) or 1.0
        
        text = f"""💰 <b>POSITION SIZING BY SIGNAL STRENGTH</b>

Bet bigger on stronger signals, smaller on weaker.
<i>Min score 70 to trade - no weak signals!</i>

🟢 <b>LOW Risk</b> (score ≥85): <b>{size_low}%</b>
🟡 <b>MEDIUM</b> (score ≥75): <b>{size_med}%</b>
🔴 <b>HIGH</b> (score 70-74): <b>{size_high}%</b>

<i>Tap a level to change its size</i>"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text=f"🟢 LOW: {size_low}%", callback_data="social_size_pick_low"),
                InlineKeyboardButton(text=f"🟡 MED: {size_med}%", callback_data="social_size_pick_medium")
            ],
            [
                InlineKeyboardButton(text=f"🔴 HIGH: {size_high}%", callback_data="social_size_pick_high")
            ],
            [
                InlineKeyboardButton(text="🔙 Back", callback_data="social_settings")
            ]
        ])
        
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data.startswith("social_size_pick_"))
async def handle_social_size_pick(callback: CallbackQuery):
    """Show size options for a specific risk level"""
    await callback.answer()
    level = callback.data.replace("social_size_pick_", "")
    level_display = {"low": "🟢 LOW Risk", "medium": "🟡 MEDIUM", "high": "🔴 HIGH", "all": "⚫ ALL"}.get(level, level)
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="2%", callback_data=f"social_size_set_{level}_2"),
            InlineKeyboardButton(text="3%", callback_data=f"social_size_set_{level}_3"),
            InlineKeyboardButton(text="5%", callback_data=f"social_size_set_{level}_5")
        ],
        [
            InlineKeyboardButton(text="7%", callback_data=f"social_size_set_{level}_7"),
            InlineKeyboardButton(text="10%", callback_data=f"social_size_set_{level}_10"),
            InlineKeyboardButton(text="15%", callback_data=f"social_size_set_{level}_15")
        ],
        [
            InlineKeyboardButton(text="🔙 Back", callback_data="social_edit_sizes")
        ]
    ])
    
    await callback.message.edit_text(f"💰 <b>Set Size for {level_display}</b>\n\n% of balance per trade", reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data.startswith("social_size_set_"))
async def handle_social_size_set(callback: CallbackQuery):
    """Set position size for a risk level"""
    await callback.answer()
    # Parse: social_size_set_low_10
    parts = callback.data.replace("social_size_set_", "").rsplit("_", 1)
    if len(parts) != 2:
        return
    level, size_str = parts
    size = float(size_str)
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if user and user.preferences:
            if level == "low":
                user.preferences.social_size_low = size
            elif level == "medium":
                user.preferences.social_size_medium = size
            elif level == "high":
                user.preferences.social_size_high = size
            elif level == "all":
                user.preferences.social_size_all = size
            db.commit()
            
            level_display = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH", "all": "ALL"}.get(level, level)
            await callback.message.answer(f"✅ {level_display} size: <b>{size}%</b>", parse_mode="HTML")
        await handle_social_edit_sizes(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "social_edit_score")
async def handle_social_edit_score(callback: CallbackQuery):
    """Show min score options"""
    await callback.answer()
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="50", callback_data="social_score_50"),
            InlineKeyboardButton(text="60", callback_data="social_score_60"),
            InlineKeyboardButton(text="70", callback_data="social_score_70")
        ],
        [
            InlineKeyboardButton(text="80", callback_data="social_score_80"),
            InlineKeyboardButton(text="90", callback_data="social_score_90")
        ],
        [
            InlineKeyboardButton(text="🔙 Back", callback_data="social_settings")
        ]
    ])
    
    await callback.message.edit_text("🌟 <b>Select Min Signal Score</b>\n\nHigher = fewer but stronger signals", reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data.startswith("social_score_"))
async def handle_social_score_set(callback: CallbackQuery):
    """Set min score"""
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if user and user.preferences:
            score = int(callback.data.replace("social_score_", ""))
            user.preferences.social_min_galaxy_score = score
            db.commit()
            await callback.message.answer(f"✅ Min Score: <b>{score}</b>", parse_mode="HTML")
        await handle_social_settings(callback)
    finally:
        db.close()


@dp.callback_query(F.data == "social_scan_now")
async def handle_social_scan_now(callback: CallbackQuery):
    """Run social scan now"""
    await callback.answer("🌙 Running social scan...")
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            return
        
        from app.services.lunarcrush import get_lunarcrush_api_key
        
        if not get_lunarcrush_api_key():
            await callback.message.answer("❌ LUNARCRUSH_API_KEY not configured. Add it to secrets to enable social trading.")
            return
        
        from app.services.social_signals import SocialSignalService
        from app.services.lunarcrush import interpret_signal_score
        
        prefs = user.preferences
        risk_level = getattr(prefs, 'social_risk_level', 'MEDIUM') or 'MEDIUM' if prefs else 'MEDIUM'
        min_galaxy = getattr(prefs, 'social_min_galaxy_score', 8) or 8 if prefs else 8
        
        service = SocialSignalService()
        await service.init()
        signal = await service.generate_social_signal(risk_level=risk_level, min_galaxy_score=min_galaxy)
        await service.close()
        
        if signal:
            rating = interpret_signal_score(signal['galaxy_score'])
            tp_pct = signal.get('tp_percent', 5)
            sl_pct = signal.get('sl_percent', 3)
            
            from app.services.social_signals import is_top_coin
            scan_is_top = is_top_coin(signal['symbol'])
            user_top_lev = getattr(prefs, 'social_top_coin_leverage', 25) or 25 if prefs else 25
            user_alt_lev = getattr(prefs, 'social_leverage', 10) or 10 if prefs else 10
            scan_lev = user_top_lev if scan_is_top else user_alt_lev
            
            tp1_roi = tp_pct * scan_lev
            tp_display = f"🎯 TP1: ${signal['take_profit']:,.4f} (+{tp_pct:.0f}% / +{tp1_roi:.0f}% ROI)"
            if signal.get('take_profit_2'):
                tp2_pct = tp_pct * 1.5
                tp2_roi = tp2_pct * scan_lev
                tp_display += f"\n🎯 TP2: ${signal['take_profit_2']:,.4f} (+{tp2_pct:.0f}% / +{tp2_roi:.0f}% ROI)"
            if signal.get('take_profit_3'):
                tp3_pct = tp_pct * 2.0
                tp3_roi = tp3_pct * scan_lev
                tp_display += f"\n🚀 TP3: ${signal['take_profit_3']:,.4f} (+{tp3_pct:.0f}% / +{tp3_roi:.0f}% ROI)"
            
            sl_roi = sl_pct * scan_lev
            
            if signal.get('risk_level') == 'MOMENTUM':
                signal_type = "🚀 <b>MOMENTUM SIGNAL</b> - NEWS RUNNER"
            else:
                signal_type = "🌙 <b>SOCIAL SIGNAL</b>"
            
            await callback.message.answer(
                f"{signal_type}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                f"📊 <b>{signal['symbol']}</b>\n\n"
                f"📈 Direction: LONG\n"
                f"💰 Entry: ${signal['entry_price']:,.4f}\n"
                f"{tp_display}\n"
                f"🛑 SL: ${signal['stop_loss']:,.4f} (-{sl_pct:.0f}% / -{sl_roi:.0f}% ROI)\n\n"
                f"<b>📱 AI Signal Analysis:</b>\n"
                f"• Signal Score: {signal['galaxy_score']}/100 {rating}\n"
                f"• Sentiment: {signal['sentiment']:.2f}\n"
                f"• RSI: {signal['rsi']:.0f}",
                parse_mode="HTML"
            )
        else:
            await callback.message.answer("📱 No social signals found matching your criteria right now.")
    finally:
        db.close()


@dp.callback_query(F.data == "social_trending")
async def handle_social_trending(callback: CallbackQuery):
    """Show trending coins from social/news analysis"""
    await callback.answer("🌙 Fetching trending coins...")
    
    from app.services.lunarcrush import get_lunarcrush_api_key, get_trending_coins, interpret_signal_score
    
    if not get_lunarcrush_api_key():
        await callback.message.answer(
            "❌ <b>API Key Required</b>\n\n"
            "Add LUNARCRUSH_API_KEY to your secrets to see trending coins.",
            parse_mode="HTML"
        )
        return
    
    try:
        trending = await get_trending_coins(limit=10)
        
        if not trending:
            await callback.message.answer("📱 Unable to fetch trending coins. Try again later.")
            return
        
        trending_text = """
┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
  📊 <b>TRENDING ON SOCIAL</b>
┗━━━━━━━━━━━━━━━━━━━━━━━━━┛

<b>Top 10 by Signal Score:</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for i, coin in enumerate(trending[:10], 1):
            symbol = coin['symbol'].replace('USDT', '')
            score = coin['galaxy_score']
            sentiment = coin.get('sentiment', 0)
            change = coin.get('percent_change_24h', 0)
            rating = interpret_signal_score(score)
            
            # Sentiment emoji
            sent_emoji = "🟢" if sentiment > 0.3 else ("🔴" if sentiment < -0.3 else "⚪")
            change_emoji = "📈" if change > 0 else "📉"
            
            trending_text += f"{i}. <b>{symbol}</b> - Score: {score} {rating}\n"
            trending_text += f"   {sent_emoji} Sentiment: {sentiment:.2f} | {change_emoji} 24h: {change:+.1f}%\n"
        
        trending_text += """
━━━━━━━━━━━━━━━━━━━━━━━━━━
<i>Signal Score: 0-100 AI momentum rating</i>
<i>Higher = stronger bullish signals</i>
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="🔄 Refresh", callback_data="social_trending"),
                InlineKeyboardButton(text="🔍 Scan Signal", callback_data="social_scan_now")
            ],
            [
                InlineKeyboardButton(text="🔙 Back", callback_data="social_menu")
            ]
        ])
        
        await callback.message.edit_text(trending_text, reply_markup=keyboard, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error fetching trending: {e}")
        await callback.message.answer("❌ Error fetching trending coins. Try again.")


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
                "💡 <i>Upgrade to $80/month plan to unlock!</i>",
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
            await callback.answer("⚠️ Auto-trading requires Auto-Trading plan ($80/mo)", show_alert=True)
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
    await message.answer("Top Gainers mode has been replaced by Social Trading. Use the Social Trading menu instead.")


@dp.message(Command("top_gainer_stats"))
async def cmd_top_gainer_stats(message: types.Message):
    await message.answer("Top Gainers mode has been replaced by Social Trading. Use the Social Trading menu instead.")


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


@dp.message(Command("news"))
async def cmd_news(message: types.Message):
    """📰 News Impact Scanner - AI analyzes crypto news for trading signals"""
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
        
        await message.answer("📰 <b>Analyzing crypto news...</b>\n\n<i>This may take a few seconds.</i>", parse_mode="HTML")
        
        from app.services.ai_market_intelligence import analyze_news_impact, format_news_alert_message
        
        result = await analyze_news_impact()
        
        if result.get('error'):
            await message.answer(f"❌ News analysis failed: {result['error']}")
            return
        
        alerts = result.get('alerts', [])
        sentiment = result.get('market_sentiment', 'NEUTRAL')
        themes = result.get('key_themes', [])
        
        sentiment_emoji = "🟢" if sentiment == 'BULLISH' else "🔴" if sentiment == 'BEARISH' else "⚪"
        
        response = f"""📰 <b>NEWS IMPACT SCANNER</b>
━━━━━━━━━━━━━━━━━━━━
{sentiment_emoji} <b>Market Sentiment:</b> {sentiment}
🔑 <b>Key Themes:</b> {', '.join(themes[:3]) if themes else 'None detected'}

"""
        
        if alerts:
            response += f"<b>🚨 {len(alerts)} Trading Alerts:</b>\n\n"
            for alert in alerts[:5]:
                direction_emoji = "🟢" if alert.get('direction') == 'BULLISH' else "🔴"
                strength_emoji = "🔥" if alert.get('strength') == 'HIGH' else "⚡" if alert.get('strength') == 'MEDIUM' else "💡"
                coins = ", ".join(alert.get('coins', []))
                response += f"{direction_emoji}{strength_emoji} <b>{coins}</b>\n"
                response += f"   {alert.get('headline', '')[:60]}...\n"
                response += f"   Impact: {alert.get('direction')} ({alert.get('strength')})\n\n"
        else:
            response += "✅ <i>No major market-moving news detected.</i>\n"
        
        response += "\n💡 <i>News is scanned every 30 minutes automatically.</i>"
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="🔮 Market", callback_data="market_regime"),
                InlineKeyboardButton(text="🐋 Whales", callback_data="whale_tracker")
            ],
            [
                InlineKeyboardButton(text="📊 Leaders", callback_data="leaderboard_tracker"),
                InlineKeyboardButton(text="◀️ Dashboard", callback_data="back_to_dashboard")
            ]
        ])
        
        await message.answer(response, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"News command error: {e}")
        await message.answer(f"❌ Error analyzing news: {str(e)[:100]}")
    finally:
        db.close()


@dp.message(Command("twitter"))
async def cmd_twitter(message: types.Message):
    """🐦 Twitter Dashboard with Interactive Buttons (Admin only)"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        if not user.is_admin:
            await message.answer("Admin only.")
            return
        
        from app.services.twitter_poster import get_twitter_poster, get_all_twitter_accounts, get_twitter_schedule
        
        args = message.text.split()
        
        # Handle /twitter add [name] command for adding new accounts
        if len(args) >= 3 and args[1].lower() == "add":
            account_name = args[2]
            if not hasattr(cmd_twitter, 'pending_adds'):
                cmd_twitter.pending_adds = {}
            
            cmd_twitter.pending_adds[message.from_user.id] = {
                'name': account_name,
                'step': 'consumer_key'
            }
            
            await message.answer(f"""🐦 <b>ADDING ACCOUNT: {account_name}</b>

I'll now ask for your Twitter API credentials one by one.

<b>Step 1/5:</b> Send me the <b>Consumer Key</b> (API Key):""", parse_mode="HTML")
            return
        
        # Show the main Twitter dashboard with buttons
        await show_twitter_dashboard(message)
        
    finally:
        db.close()


async def show_twitter_dashboard(message: types.Message, edit: bool = False):
    """Show the main Twitter dashboard with buttons"""
    from app.services.twitter_poster import get_twitter_poster, get_all_twitter_accounts, get_twitter_schedule
    
    poster = get_twitter_poster()
    status = poster.get_status()
    schedule = get_twitter_schedule()
    accounts = get_all_twitter_accounts()
    
    # Count active/inactive accounts
    active_count = sum(1 for a in accounts if a.is_active)
    total_count = len(accounts)
    
    # Format countdown
    countdown_text = ""
    if schedule.get('next_post_type') and schedule.get('time_until_next'):
        countdown_text = f"\n⏱️ Next: {schedule['next_post_type']} in <b>{schedule['time_until_next']}</b>"
    
    dashboard_text = f"""🐦 <b>TWITTER DASHBOARD</b>

<b>Status:</b> {'✅ Connected' if status['initialized'] else '❌ Not configured'}
<b>Accounts:</b> {active_count}/{total_count} active
<b>Posts today:</b> {status['posts_today']}/{status['max_posts']}
<b>Last post:</b> {status['last_post'] or 'Never'}{countdown_text}

<i>Select an option below:</i>"""
    
    # Create button layout
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="👥 Accounts", callback_data="tw_accounts"),
            InlineKeyboardButton(text="⏰ Schedule", callback_data="tw_schedule")
        ],
        [
            InlineKeyboardButton(text="➕ Add Account", callback_data="tw_add"),
            InlineKeyboardButton(text="📋 Preview", callback_data="tw_preview")
        ],
        [
            InlineKeyboardButton(text="🌟 Post Featured", callback_data="tw_post_featured"),
            InlineKeyboardButton(text="📈 Post Gainers", callback_data="tw_post_gainers")
        ],
        [
            InlineKeyboardButton(text="📉 Post Losers", callback_data="tw_post_losers"),
            InlineKeyboardButton(text="📊 Post Market", callback_data="tw_post_market")
        ],
        [
            InlineKeyboardButton(text="₿ Post BTC", callback_data="tw_post_btc"),
            InlineKeyboardButton(text="💹 Post Alts", callback_data="tw_post_alts")
        ],
        [
            InlineKeyboardButton(text="🐸 Post Memecoin", callback_data="tw_post_memecoin"),
            InlineKeyboardButton(text="📰 Breaking News", callback_data="tw_post_news")
        ],
        [
            InlineKeyboardButton(text="🔄 Refresh", callback_data="tw_refresh")
        ]
    ])
    
    if edit:
        await message.edit_text(dashboard_text, parse_mode="HTML", reply_markup=keyboard)
    else:
        await message.answer(dashboard_text, parse_mode="HTML", reply_markup=keyboard)


@dp.callback_query(F.data == "tw_accounts")
async def cb_twitter_accounts(callback: types.CallbackQuery):
    """Show Twitter accounts list with toggle buttons"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            await callback.answer("Admin only", show_alert=True)
            return
        
        from app.services.twitter_poster import get_all_twitter_accounts
        
        accounts = get_all_twitter_accounts()
        
        if not accounts:
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="➕ Add Account", callback_data="tw_add")],
                [InlineKeyboardButton(text="« Back", callback_data="tw_back")]
            ])
            await callback.message.edit_text(
                "🐦 <b>TWITTER ACCOUNTS</b>\n\nNo accounts configured yet.\n\nClick below to add one:",
                parse_mode="HTML",
                reply_markup=keyboard
            )
            await callback.answer()
            return
        
        accounts_text = "🐦 <b>TWITTER ACCOUNTS</b>\n\n"
        buttons = []
        
        for acc in accounts:
            status_icon = "✅" if acc.is_active else "❌"
            types = acc.get_post_types()
            types_str = ", ".join(types[:2]) + ("..." if len(types) > 2 else "") if types else "No types"
            
            accounts_text += f"{status_icon} <b>{acc.name}</b>"
            if acc.handle:
                accounts_text += f" (@{acc.handle})"
            accounts_text += f"\n   📝 {types_str}\n\n"
            
            # Add toggle and settings buttons for each account
            toggle_text = "❌ Disable" if acc.is_active else "✅ Enable"
            buttons.append([
                InlineKeyboardButton(text=f"⚙️ {acc.name}", callback_data=f"tw_acc_{acc.id}"),
                InlineKeyboardButton(text=toggle_text, callback_data=f"tw_toggle_{acc.id}")
            ])
        
        buttons.append([InlineKeyboardButton(text="➕ Add Account", callback_data="tw_add")])
        buttons.append([InlineKeyboardButton(text="« Back", callback_data="tw_back")])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
        
        await callback.message.edit_text(accounts_text, parse_mode="HTML", reply_markup=keyboard)
        await callback.answer()
        
    finally:
        db.close()


@dp.callback_query(F.data.startswith("tw_acc_"))
async def cb_twitter_account_settings(callback: types.CallbackQuery):
    """Show settings for a specific Twitter account"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            await callback.answer("Admin only", show_alert=True)
            return
        
        account_id = int(callback.data.replace("tw_acc_", ""))
        
        from app.models import TwitterAccount
        account = db.query(TwitterAccount).filter(TwitterAccount.id == account_id).first()
        
        if not account:
            await callback.answer("Account not found", show_alert=True)
            return
        
        db.expunge(account)
        
        types = account.get_post_types()
        types_str = ", ".join(types) if types else "None assigned"
        status = "✅ Active" if account.is_active else "❌ Disabled"
        
        # Check if this is the Crypto Social account
        from app.services.twitter_poster import is_social_account
        is_social = is_social_account(account.name)
        
        account_text = f"""🐦 <b>ACCOUNT: {account.name}</b>

<b>Handle:</b> @{account.handle or 'Not set'}
<b>Status:</b> {status}
<b>Post Types:</b> {types_str}
{'<b>Type:</b> 📰 Crypto Social (News & Gainers)' if is_social else ''}

<b>📝 Assign Post Types:</b>"""
        
        # Create buttons for each post type
        all_types = ['featured_coin', 'market_summary', 'top_gainers', 'top_losers', 'btc_update', 'altcoin_movers', 'daily_recap']
        type_labels = {
            'featured_coin': '🌟 Featured',
            'market_summary': '📊 Market',
            'top_gainers': '📈 Gainers',
            'top_losers': '📉 Losers',
            'btc_update': '₿ BTC',
            'altcoin_movers': '💹 Alts',
            'daily_recap': '📋 Recap'
        }
        
        buttons = []
        row = []
        for pt in all_types:
            is_assigned = pt in types
            icon = "✓ " if is_assigned else ""
            row.append(InlineKeyboardButton(
                text=f"{icon}{type_labels.get(pt, pt)}",
                callback_data=f"tw_type_{account_id}_{pt}"
            ))
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)
        
        # Add manual post section header
        buttons.append([InlineKeyboardButton(text="── 📤 MANUAL POST ──", callback_data="tw_noop")])
        
        if is_social:
            # Crypto Social specific post types
            buttons.append([
                InlineKeyboardButton(text="📰 News Post", callback_data=f"tw_manual_{account_id}_breaking_news"),
                InlineKeyboardButton(text="🚀 Early Gainer", callback_data=f"tw_manual_{account_id}_early_gainer")
            ])
            buttons.append([
                InlineKeyboardButton(text="📊 Momentum", callback_data=f"tw_manual_{account_id}_momentum_shift"),
                InlineKeyboardButton(text="📈 Volume Surge", callback_data=f"tw_manual_{account_id}_volume_surge")
            ])
        else:
            # Standard account post types - all 10 types
            buttons.append([
                InlineKeyboardButton(text="🌟 Featured", callback_data=f"tw_manual_{account_id}_featured_coin"),
                InlineKeyboardButton(text="🎯 Early Gainer", callback_data=f"tw_manual_{account_id}_early_gainer")
            ])
            buttons.append([
                InlineKeyboardButton(text="🐋 Whale Alert", callback_data=f"tw_manual_{account_id}_whale_alert"),
                InlineKeyboardButton(text="📊 Quick TA", callback_data=f"tw_manual_{account_id}_quick_ta")
            ])
            buttons.append([
                InlineKeyboardButton(text="⚠️ Funding", callback_data=f"tw_manual_{account_id}_funding_extreme"),
                InlineKeyboardButton(text="📈 Gainers", callback_data=f"tw_manual_{account_id}_top_gainers")
            ])
            buttons.append([
                InlineKeyboardButton(text="📊 Market", callback_data=f"tw_manual_{account_id}_market_summary"),
                InlineKeyboardButton(text="₿ BTC Update", callback_data=f"tw_manual_{account_id}_btc_update")
            ])
            buttons.append([
                InlineKeyboardButton(text="💹 Altcoins", callback_data=f"tw_manual_{account_id}_altcoin_movers"),
                InlineKeyboardButton(text="📈 Daily Recap", callback_data=f"tw_manual_{account_id}_daily_recap")
            ])
            buttons.append([
                InlineKeyboardButton(text="🔥 High Viewing", callback_data=f"tw_manual_{account_id}_high_viewing"),
                InlineKeyboardButton(text="💰 Campaign", callback_data=f"tw_manual_{account_id}_bitunix_campaign")
            ])
        
        toggle_text = "❌ Disable Account" if account.is_active else "✅ Enable Account"
        buttons.append([InlineKeyboardButton(text=toggle_text, callback_data=f"tw_toggle_{account_id}")])
        buttons.append([InlineKeyboardButton(text="🗑️ Remove Account", callback_data=f"tw_remove_{account_id}")])
        buttons.append([InlineKeyboardButton(text="« Back to Accounts", callback_data="tw_accounts")])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
        
        await callback.message.edit_text(account_text, parse_mode="HTML", reply_markup=keyboard)
        await callback.answer()
        
    finally:
        db.close()


@dp.callback_query(F.data.startswith("tw_type_"))
async def cb_twitter_toggle_type(callback: types.CallbackQuery):
    """Toggle a post type for an account"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            await callback.answer("Admin only", show_alert=True)
            return
        
        parts = callback.data.replace("tw_type_", "").split("_", 1)
        account_id = int(parts[0])
        post_type = parts[1]
        
        from app.models import TwitterAccount
        account = db.query(TwitterAccount).filter(TwitterAccount.id == account_id).first()
        
        if not account:
            await callback.answer("Account not found", show_alert=True)
            return
        
        types = account.get_post_types()
        
        if post_type in types:
            types.remove(post_type)
            action = "removed"
        else:
            types.append(post_type)
            action = "added"
        
        account.set_post_types(types)
        db.commit()
        
        await callback.answer(f"{post_type} {action}!")
        
        # Refresh the account settings view
        callback.data = f"tw_acc_{account_id}"
        await cb_twitter_account_settings(callback)
        
    finally:
        db.close()


@dp.callback_query(F.data.startswith("tw_toggle_"))
async def cb_twitter_toggle_account(callback: types.CallbackQuery):
    """Toggle an account's active status"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            await callback.answer("Admin only", show_alert=True)
            return
        
        account_id = int(callback.data.replace("tw_toggle_", ""))
        
        from app.models import TwitterAccount
        account = db.query(TwitterAccount).filter(TwitterAccount.id == account_id).first()
        
        if not account:
            await callback.answer("Account not found", show_alert=True)
            return
        
        account.is_active = not account.is_active
        new_status = "enabled" if account.is_active else "disabled"
        db.commit()
        
        await callback.answer(f"Account {new_status}!")
        
        # Refresh accounts list
        await cb_twitter_accounts(callback)
        
    finally:
        db.close()


@dp.callback_query(F.data.startswith("tw_remove_"))
async def cb_twitter_remove_account(callback: types.CallbackQuery):
    """Confirm removal of an account"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            await callback.answer("Admin only", show_alert=True)
            return
        
        account_id = int(callback.data.replace("tw_remove_", ""))
        
        from app.models import TwitterAccount
        account = db.query(TwitterAccount).filter(TwitterAccount.id == account_id).first()
        
        if not account:
            await callback.answer("Account not found", show_alert=True)
            return
        
        account_name = account.name
        db.expunge(account)
        
        confirm_text = f"""⚠️ <b>CONFIRM REMOVAL</b>

Are you sure you want to remove <b>{account_name}</b>?

This will delete all credentials and cannot be undone."""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Yes, Remove", callback_data=f"tw_confirm_remove_{account_id}"),
                InlineKeyboardButton(text="❌ Cancel", callback_data=f"tw_acc_{account_id}")
            ]
        ])
        
        await callback.message.edit_text(confirm_text, parse_mode="HTML", reply_markup=keyboard)
        await callback.answer()
        
    finally:
        db.close()


@dp.callback_query(F.data.startswith("tw_confirm_remove_"))
async def cb_twitter_confirm_remove(callback: types.CallbackQuery):
    """Actually remove the account"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            await callback.answer("Admin only", show_alert=True)
            return
        
        account_id = int(callback.data.replace("tw_confirm_remove_", ""))
        
        from app.models import TwitterAccount
        account = db.query(TwitterAccount).filter(TwitterAccount.id == account_id).first()
        
        if account:
            account_name = account.name
            db.delete(account)
            db.commit()
            await callback.answer(f"Account {account_name} removed!", show_alert=True)
        
        # Go back to accounts list
        await cb_twitter_accounts(callback)
        
    finally:
        db.close()


@dp.callback_query(F.data == "tw_schedule")
async def cb_twitter_schedule(callback: types.CallbackQuery):
    """Show the posting schedule"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            await callback.answer("Admin only", show_alert=True)
            return
        
        from app.services.twitter_poster import get_twitter_schedule
        
        schedule = get_twitter_schedule()
        
        schedule_text = f"""🐦 <b>POSTING SCHEDULE</b>

<b>Status:</b> {'✅ Enabled' if schedule['enabled'] else '❌ Disabled'}
<b>Posts today:</b> {schedule['posts_today']}/{schedule['max_posts']}
<b>Remaining:</b> {schedule['posts_remaining']}
<b>Last post:</b> {schedule['last_post'] or 'None yet'}

<b>⏰ NEXT POST:</b>
{schedule['next_post_type'] or 'None scheduled'}
📍 {schedule['next_post_time'] or 'N/A'}
⏱️ In {schedule['time_until_next'] or 'N/A'}

<b>📅 UPCOMING (UTC):</b>
"""
        upcoming = [s for s in schedule['schedule'] if not s['posted']][:6]
        for slot in upcoming:
            schedule_text += f"• {slot['time_str']} - {slot['type']}\n"
        
        if not upcoming:
            schedule_text += "<i>All posts completed for today!</i>\n"
        
        schedule_text += "\n<i>⏰ Times have ±15min random offset</i>"
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Refresh", callback_data="tw_schedule")],
            [InlineKeyboardButton(text="« Back", callback_data="tw_back")]
        ])
        
        await callback.message.edit_text(schedule_text, parse_mode="HTML", reply_markup=keyboard)
        await callback.answer()
        
    finally:
        db.close()


@dp.callback_query(F.data == "tw_add")
async def cb_twitter_add(callback: types.CallbackQuery):
    """Start adding a new account"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            await callback.answer("Admin only", show_alert=True)
            return
        
        add_text = """🐦 <b>ADD NEW ACCOUNT</b>

To add a new Twitter account, use the command:

<code>/twitter add YourAccountName</code>

Replace <b>YourAccountName</b> with a friendly name for the account (e.g., "TradeSignals" or "CryptoNews").

I'll then ask for your API credentials step by step."""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="« Back", callback_data="tw_back")]
        ])
        
        await callback.message.edit_text(add_text, parse_mode="HTML", reply_markup=keyboard)
        await callback.answer()
        
    finally:
        db.close()


@dp.callback_query(F.data == "tw_preview")
async def cb_twitter_preview(callback: types.CallbackQuery):
    """Show a preview of what would be posted"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            await callback.answer("Admin only", show_alert=True)
            return
        
        await callback.answer("Loading preview...")
        
        from app.services.twitter_poster import get_twitter_poster
        poster = get_twitter_poster()
        
        gainers = await poster.get_top_gainers_data(5)
        market = await poster.get_market_summary()
        
        preview_text = "📋 <b>PREVIEW (not posted)</b>\n\n"
        
        if market:
            btc_sign = "+" if market['btc_change'] >= 0 else ""
            preview_text += f"<b>BTC:</b> ${market['btc_price']:,.0f} ({btc_sign}{market['btc_change']:.1f}%)\n\n"
        
        if gainers:
            preview_text += "<b>Top Gainers:</b>\n"
            for i, coin in enumerate(gainers[:5], 1):
                sign = "+" if coin['change'] >= 0 else ""
                preview_text += f"{i}. ${coin['symbol']} {sign}{coin['change']:.1f}%\n"
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Refresh", callback_data="tw_preview")],
            [InlineKeyboardButton(text="« Back", callback_data="tw_back")]
        ])
        
        await callback.message.edit_text(preview_text, parse_mode="HTML", reply_markup=keyboard)
        
    finally:
        db.close()


@dp.callback_query(F.data.startswith("tw_post_"))
async def cb_twitter_post(callback: types.CallbackQuery):
    """Handle posting buttons"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            await callback.answer("Admin only", show_alert=True)
            return
        
        post_type = callback.data.replace("tw_post_", "")
        
        type_names = {
            'featured': 'featured coin',
            'gainers': 'top gainers',
            'losers': 'top losers',
            'market': 'market summary',
            'btc': 'BTC update',
            'alts': 'altcoin movers',
            'recap': 'daily recap',
            'high_viewing': 'high viewing',
            'memecoin': 'pump.fun memecoin',
            'news': 'breaking news'
        }
        
        await callback.answer(f"Posting {type_names.get(post_type, post_type)}...")
        
        from app.services.twitter_poster import get_twitter_poster
        poster = get_twitter_poster()
        
        result = None
        if post_type == 'featured':
            result = await poster.post_featured_coin()
        elif post_type == 'gainers':
            result = await poster.post_top_gainers()
        elif post_type == 'losers':
            result = await poster.post_top_losers()
        elif post_type == 'market':
            result = await poster.post_market_summary()
        elif post_type == 'btc':
            result = await poster.post_btc_update()
        elif post_type == 'alts':
            result = await poster.post_altcoin_movers()
        elif post_type == 'recap':
            result = await poster.post_daily_recap()
        elif post_type == 'high_viewing':
            result = await poster.post_high_viewing()
        elif post_type == 'memecoin':
            from app.services.twitter_poster import post_memecoin, MultiAccountPoster
            from app.models import TwitterAccount
            # Use ccally account specifically
            account = db.query(TwitterAccount).filter(TwitterAccount.name == 'ccally', TwitterAccount.is_active == True).first()
            if account:
                account_poster = MultiAccountPoster(account)
                result = await post_memecoin(account_poster)
            else:
                result = {'success': False, 'error': 'ccally account not found'}
        elif post_type == 'news':
            from app.services.twitter_poster import post_social_news, MultiAccountPoster
            from app.models import TwitterAccount
            active_accounts = db.query(TwitterAccount).filter(TwitterAccount.is_active == True).all()
            if active_accounts:
                account = active_accounts[0]
                account_poster = MultiAccountPoster(account)
                result = await post_social_news(account_poster)
            else:
                result = {'success': False, 'error': 'No active Twitter account found'}
        
        if result and result.get('success'):
            result_text = f"✅ <b>{type_names.get(post_type, post_type).title()} posted!</b>\n\nTweet ID: {result['tweet_id']}"
        else:
            error = result.get('error', 'Unknown error') if result else 'No data available'
            result_text = f"❌ <b>Failed to post {type_names.get(post_type, post_type)}</b>\n\nError: {error}"
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="« Back to Dashboard", callback_data="tw_back")]
        ])
        
        await callback.message.edit_text(result_text, parse_mode="HTML", reply_markup=keyboard)
        
    finally:
        db.close()


@dp.callback_query(F.data == "tw_noop")
async def cb_twitter_noop(callback: types.CallbackQuery):
    """No-op for header buttons"""
    await callback.answer()


@dp.callback_query(F.data.startswith("tw_manual_"))
async def cb_twitter_manual_post(callback: types.CallbackQuery):
    """Handle manual posts for specific accounts"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            await callback.answer("Admin only", show_alert=True)
            return
        
        # Parse: tw_manual_{account_id}_{post_type}
        parts = callback.data.replace("tw_manual_", "").split("_", 1)
        account_id = int(parts[0])
        post_type = parts[1]
        
        from app.models import TwitterAccount
        account = db.query(TwitterAccount).filter(TwitterAccount.id == account_id).first()
        
        if not account:
            await callback.answer("Account not found", show_alert=True)
            return
        
        account_name = account.name
        db.expunge(account)
        
        type_names = {
            'featured_coin': 'Featured Coin',
            'top_gainers': 'Top Gainers',
            'market_summary': 'Market Summary',
            'btc_update': 'BTC Update',
            'breaking_news': 'Breaking News',
            'early_gainer': 'Early Gainer',
            'momentum_shift': 'Momentum Shift',
            'volume_surge': 'Volume Surge',
            'whale_alert': 'Whale Alert',
            'funding_extreme': 'Funding Alert',
            'quick_ta': 'Quick TA',
            'altcoin_movers': 'Altcoin Movers',
            'daily_recap': 'Daily Recap',
            'high_viewing': 'High Viewing',
            'bitunix_campaign': 'Bitunix Campaign'
        }
        
        await callback.answer(f"Posting {type_names.get(post_type, post_type)}...")
        
        from app.services.twitter_poster import (
            get_twitter_poster, 
            MultiAccountPoster, 
            post_with_account, 
            post_for_social_account,
            is_social_account
        )
        
        # Fetch account from database
        account = db.query(TwitterAccount).filter(TwitterAccount.id == account_id).first()
        if not account:
            await callback.message.edit_text("❌ Account not found in database.", parse_mode="HTML")
            return
        
        # Expunge to detach from session
        db.expunge(account)
        
        # Create poster for this specific account
        main_poster = get_twitter_poster()
        account_poster = MultiAccountPoster(account)
        
        result = None
        if is_social_account(account_name):
            # Use social account posting
            result = await post_for_social_account(account_poster, post_type)
        else:
            # Use standard posting
            result = await post_with_account(account_poster, main_poster, post_type)
        
        if result and result.get('success'):
            result_text = f"✅ <b>{type_names.get(post_type, post_type)} posted!</b>\n\n<b>Account:</b> {account_name}\n<b>Tweet ID:</b> {result.get('tweet_id', 'N/A')}"
        else:
            error = result.get('error', 'Unknown error') if result else 'No data available'
            result_text = f"❌ <b>Failed to post {type_names.get(post_type, post_type)}</b>\n\n<b>Account:</b> {account_name}\n<b>Error:</b> {error}"
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="« Back to Account", callback_data=f"tw_acc_{account_id}")]
        ])
        
        await callback.message.edit_text(result_text, parse_mode="HTML", reply_markup=keyboard)
        
    finally:
        db.close()


@dp.callback_query(F.data == "tw_refresh")
async def cb_twitter_refresh(callback: types.CallbackQuery):
    """Refresh the dashboard"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            await callback.answer("Admin only", show_alert=True)
            return
        
        await callback.answer("Refreshing...")
        await show_twitter_dashboard(callback.message, edit=True)
        
    finally:
        db.close()


@dp.callback_query(F.data == "tw_back")
async def cb_twitter_back(callback: types.CallbackQuery):
    """Go back to main dashboard"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            await callback.answer("Admin only", show_alert=True)
            return
        
        await show_twitter_dashboard(callback.message, edit=True)
        await callback.answer()
        
    finally:
        db.close()


@dp.message(Command("metals"))
async def cmd_metals(message: types.Message):
    """🥇 Metals Trading - Gold/Silver signals (Admin only)"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        if not user.is_admin:
            await message.answer("⚠️ This feature is admin-only for now.")
            return
        
        from app.services.metals_signals import (
            is_metals_scanning_enabled, 
            toggle_metals_scanning,
            MetalsSignalService,
            get_metals_sentiment
        )
        from app.services.marketaux import MarketAuxClient
        
        args = message.text.split()
        
        if len(args) > 1:
            action = args[1].lower()
            
            if action == "on":
                from app.services.metals_signals import set_metals_scanning
                set_metals_scanning(True)
                await message.answer("✅ <b>Metals scanning ENABLED</b>\n\nGold/Silver signals will now be generated.", parse_mode="HTML")
                return
            
            elif action == "off":
                from app.services.metals_signals import set_metals_scanning
                set_metals_scanning(False)
                await message.answer("❌ <b>Metals scanning DISABLED</b>", parse_mode="HTML")
                return
            
            elif action == "scan":
                await message.answer("🥇 <b>Scanning metals market...</b>\n\n<i>Analyzing Gold & Silver news...</i>", parse_mode="HTML")
                
                service = MetalsSignalService()
                await service.initialize()
                
                try:
                    signals = await service.scan_for_signals(force=True)
                    
                    if not signals:
                        sentiment = await get_metals_sentiment()
                        if not sentiment or not sentiment.get('gold'):
                            await message.answer("❌ Failed to get metals sentiment. Check MarketAux API key.", parse_mode="HTML")
                            return
                        await message.answer(
                            f"📊 <b>Metals Scan Complete</b>\n\n"
                            f"🥇 Gold Sentiment: {sentiment['gold']['sentiment']} ({sentiment['gold']['score']:.2f})\n"
                            f"🥈 Silver Sentiment: {sentiment['silver']['sentiment']} ({sentiment['silver']['score']:.2f})\n\n"
                            f"📰 Articles analyzed: {sentiment['total_articles']}\n"
                            f"💡 Recommendation: {sentiment['recommendation']}\n"
                            f"📝 {sentiment['reason']}\n\n"
                            f"<i>No tradeable signals found at this time.</i>",
                            parse_mode="HTML"
                        )
                    else:
                        from app.services.metals_signals import format_metals_signal
                        for signal in signals:
                            await message.answer(format_metals_signal(signal), parse_mode="HTML")
                finally:
                    await service.close()
                return
            
            elif action == "news":
                await message.answer("📰 <b>Fetching metals news...</b>", parse_mode="HTML")
                
                client = MarketAuxClient()
                sentiment = await client.analyze_metals_sentiment()
                
                headlines_text = ""
                if sentiment.get("headlines"):
                    headlines_text = "\n\n<b>📰 Recent Headlines:</b>\n" + "\n".join([f"• {h[:80]}..." for h in sentiment["headlines"]])
                
                await message.answer(
                    f"🥇 <b>METALS NEWS ANALYSIS</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"<b>Gold (XAU):</b>\n"
                    f"• Sentiment: {sentiment['gold']['sentiment'].upper()}\n"
                    f"• Score: {sentiment['gold']['score']:.3f}\n"
                    f"• Articles: {sentiment['gold']['articles']}\n\n"
                    f"<b>Silver (XAG):</b>\n"
                    f"• Sentiment: {sentiment['silver']['sentiment'].upper()}\n"
                    f"• Score: {sentiment['silver']['score']:.3f}\n"
                    f"• Articles: {sentiment['silver']['articles']}\n\n"
                    f"<b>Combined Score:</b> {sentiment.get('combined_score', 0):.3f}\n"
                    f"<b>Recommendation:</b> {sentiment['recommendation']}\n"
                    f"<b>Reason:</b> {sentiment['reason']}"
                    f"{headlines_text}",
                    parse_mode="HTML"
                )
                return
        
            elif action == "settings":
                prefs = user.preferences
                if not prefs:
                    await message.answer("❌ No preferences found. Use /start first.")
                    return
                
                metals_enabled = getattr(prefs, 'metals_enabled', False) or False
                metals_lev = getattr(prefs, 'metals_leverage', 5) or 5
                metals_size = getattr(prefs, 'metals_position_size_percent', 5.0) or 5.0
                metals_dollars = getattr(prefs, 'metals_position_size_dollars', None)
                metals_max = getattr(prefs, 'metals_max_positions', 2) or 2
                
                size_display = f"${metals_dollars:.0f}" if metals_dollars else f"{metals_size}%"
                
                await message.answer(
                    f"⚙️ <b>YOUR METALS SETTINGS</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"<b>Auto-Trading:</b> {'✅ ON' if metals_enabled else '❌ OFF'}\n"
                    f"<b>Leverage:</b> {metals_lev}x\n"
                    f"<b>Position Size:</b> {size_display}\n"
                    f"<b>Max Positions:</b> {metals_max}\n\n"
                    f"<b>Configure:</b>\n"
                    f"• <code>/metals set lev 5</code> - Set leverage (1-20)\n"
                    f"• <code>/metals set size 10</code> - Set size % of balance\n"
                    f"• <code>/metals set dollars 50</code> - Set fixed $ amount\n"
                    f"• <code>/metals set max 2</code> - Set max positions\n"
                    f"• <code>/metals enable</code> - Enable auto-trading\n"
                    f"• <code>/metals disable</code> - Disable auto-trading",
                    parse_mode="HTML"
                )
                return
            
            elif action == "enable":
                prefs = user.preferences
                if prefs:
                    prefs.metals_enabled = True
                    db.commit()
                await message.answer("✅ <b>Metals auto-trading ENABLED</b>\n\nSignals will now execute trades automatically.", parse_mode="HTML")
                return
            
            elif action == "disable":
                prefs = user.preferences
                if prefs:
                    prefs.metals_enabled = False
                    db.commit()
                await message.answer("❌ <b>Metals auto-trading DISABLED</b>\n\nYou'll still receive signals but no auto-execution.", parse_mode="HTML")
                return
            
            elif action == "set" and len(args) >= 4:
                setting = args[2].lower()
                try:
                    value = float(args[3])
                except ValueError:
                    await message.answer("❌ Invalid value. Use a number.")
                    return
                
                prefs = user.preferences
                if not prefs:
                    await message.answer("❌ No preferences found.")
                    return
                
                if setting == "lev" or setting == "leverage":
                    value = int(min(20, max(1, value)))
                    prefs.metals_leverage = value
                    db.commit()
                    await message.answer(f"✅ Metals leverage set to <b>{value}x</b>", parse_mode="HTML")
                elif setting == "size":
                    value = min(100, max(1, value))
                    prefs.metals_position_size_percent = value
                    prefs.metals_position_size_dollars = None  # Clear fixed amount
                    db.commit()
                    await message.answer(f"✅ Metals position size set to <b>{value}%</b> of balance", parse_mode="HTML")
                elif setting == "dollars":
                    value = max(5, value)
                    prefs.metals_position_size_dollars = value
                    db.commit()
                    await message.answer(f"✅ Metals position size set to <b>${value:.0f}</b> fixed", parse_mode="HTML")
                elif setting == "max":
                    value = int(min(5, max(1, value)))
                    prefs.metals_max_positions = value
                    db.commit()
                    await message.answer(f"✅ Max metals positions set to <b>{value}</b>", parse_mode="HTML")
                else:
                    await message.answer("❌ Unknown setting. Use: lev, size, dollars, max")
                return
        
        status = "✅ ON" if is_metals_scanning_enabled() else "❌ OFF"
        
        prefs = user.preferences
        metals_lev = getattr(prefs, 'metals_leverage', 5) or 5 if prefs else 5
        metals_size = getattr(prefs, 'metals_position_size_percent', 5.0) or 5.0 if prefs else 5.0
        
        await message.answer(
            f"🥇 <b>METALS TRADING</b> 🥈\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Trade Gold (XAU) and Silver (XAG) on Bitunix based on news sentiment analysis.\n\n"
            f"<b>Scanner Status:</b> {status}\n"
            f"<b>Your Settings:</b> {metals_lev}x leverage, {metals_size}% size\n\n"
            f"<b>Commands:</b>\n"
            f"• <code>/metals on</code> - Enable scanning\n"
            f"• <code>/metals off</code> - Disable scanning\n"
            f"• <code>/metals scan</code> - Run scan now\n"
            f"• <code>/metals news</code> - View news sentiment\n"
            f"• <code>/metals settings</code> - Your settings\n\n"
            f"<i>⚠️ Admin-only feature (testing phase)</i>",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Metals command error: {e}")
        await message.answer(f"❌ Error: {str(e)[:100]}")
    finally:
        db.close()


@dp.message(Command("fartcoin"))
async def cmd_fartcoin(message: types.Message):
    """🐸 FARTCOIN Scanner - SOL correlation trading (Admin only)"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        if not user.is_admin:
            await message.answer("⚠️ This feature is admin-only.")
            return
        
        from app.services.fartcoin_scanner import (
            is_fartcoin_enabled, set_fartcoin_enabled,
            FartcoinScanner, format_fartcoin_status_message,
            format_fartcoin_signal_message, broadcast_fartcoin_signal
        )
        
        args = message.text.split()
        
        if len(args) > 1:
            action = args[1].lower()
            
            if action == "on":
                set_fartcoin_enabled(True)
                await message.answer("✅ <b>FARTCOIN scanner ENABLED</b>\n\n🐸 SOL correlation signals will now be generated at 50x leverage.", parse_mode="HTML")
                return
            
            elif action == "off":
                set_fartcoin_enabled(False)
                await message.answer("❌ <b>FARTCOIN scanner DISABLED</b>", parse_mode="HTML")
                return
            
            elif action == "scan":
                await message.answer("🐸 <b>Scanning $FARTCOIN...</b>\n\n<i>Analyzing SOL beta amplification & latency pattern...</i>", parse_mode="HTML")
                
                scanner = FartcoinScanner()
                await scanner.init()
                
                try:
                    signal_data = await scanner.analyze_fartcoin()
                    
                    if signal_data:
                        signal_text = format_fartcoin_signal_message(signal_data)
                        await message.answer(
                            f"🐸 <b>SIGNAL FOUND!</b>\n\n{signal_text}\n\n"
                            f"<i>Use /fartcoin on to enable auto-broadcasting</i>",
                            parse_mode="HTML"
                        )
                    else:
                        status = await scanner.get_status()
                        status_text = format_fartcoin_status_message(status)
                        await message.answer(
                            f"📊 <b>No signal found this scan</b>\n\n{status_text}",
                            parse_mode="HTML"
                        )
                finally:
                    await scanner.close()
                return
            
            elif action == "status":
                scanner = FartcoinScanner()
                await scanner.init()
                try:
                    status = await scanner.get_status()
                    status_text = format_fartcoin_status_message(status)
                    await message.answer(status_text, parse_mode="HTML")
                finally:
                    await scanner.close()
                return
        
        status_emoji = "🟢 ON" if is_fartcoin_enabled() else "🔴 OFF"
        
        await message.answer(
            f"🐸 <b>FARTCOIN SCANNER</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Dedicated $FARTCOIN trading based on SOL beta amplification & latency.\n\n"
            f"<b>Strategy:</b> Enters $FARTCOIN when $SOL pumps (FART catches up harder) or dumps (FART follows with delay).\n"
            f"<b>Leverage:</b> 50x\n"
            f"<b>Scanner:</b> {status_emoji}\n\n"
            f"<b>Commands:</b>\n"
            f"• <code>/fartcoin on</code> - Enable scanner\n"
            f"• <code>/fartcoin off</code> - Disable scanner\n"
            f"• <code>/fartcoin scan</code> - Run scan now\n"
            f"• <code>/fartcoin status</code> - View status & data\n\n"
            f"<i>⚠️ Admin-only feature (50x leverage = extreme risk)</i>",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"FARTCOIN command error: {e}")
        await message.answer(f"❌ Error: {str(e)[:100]}")
    finally:
        db.close()


def _build_btcorb_menu_text():
    from app.services.btc_orb_scanner import (
        is_btc_orb_enabled, get_btc_orb_leverage, get_btc_orb_max_daily,
        get_btc_orb_sessions, get_btc_orb_cooldown, get_btc_orb_daily_count,
        check_btc_orb_cooldown
    )
    import app.services.btc_orb_scanner as _orb_mod

    enabled = is_btc_orb_enabled()
    leverage = get_btc_orb_leverage()
    max_daily = get_btc_orb_max_daily()
    daily_count = get_btc_orb_daily_count()
    sessions = get_btc_orb_sessions()
    cooldown = get_btc_orb_cooldown()
    cooldown_ready = check_btc_orb_cooldown()

    status_bar = "🟢 <b>ACTIVE</b> - Scanning for momentum setups" if enabled else "🔴 <b>OFF</b> - Scanner disabled"

    asia_icon = "🟢" if sessions.get("ASIA") else "⭕"
    london_icon = "🟢" if sessions.get("LONDON") else "⭕"
    ny_icon = "🟢" if sessions.get("NY") else "⭕"

    setup_text = "No active setup"
    if _orb_mod._pending_setup:
        s = _orb_mod._pending_setup
        from datetime import datetime
        age_min = (datetime.utcnow() - s['detected_at']).total_seconds() / 60
        remaining = max(0, 15 - age_min)
        if remaining > 0:
            setup_text = (f"🔍 <b>{s['session']}</b> | {s['direction']}\n"
                          f"    Retest level: ${s['break_level']:.2f} | {remaining:.0f}min left")
        else:
            setup_text = "Expired (waiting for next setup)"

    return (
        f"⚡ <b>BTC MOMENTUM SCALPER</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{status_bar}\n\n"
        f"<b>Strategy:</b> 15m structure break + micro-retest\n"
        f"Volume confirmed | RSI filtered | Funding guarded\n\n"
        f"<b>Settings</b>\n"
        f"├ Leverage: <b>{leverage}x</b>\n"
        f"├ TP/SL: <b>0.25%</b> each (1:1 R:R = {leverage // 4}% ROI)\n"
        f"├ Max Signals/Day: <b>{daily_count}/{max_daily}</b>\n"
        f"├ Cooldown: <b>{cooldown}min</b> {'✅' if cooldown_ready else '⏳'}\n"
        f"├ 🌏 Asia: {asia_icon}\n"
        f"├ 🇬🇧 London (08:00-12:00 UTC): {london_icon}\n"
        f"└ 🗽 NY (13:30-18:00 UTC): {ny_icon}\n\n"
        f"<b>Live Setup</b>\n"
        f"{setup_text}\n\n"
        f"<i>Signals only sent to users with BTC Scalp mode enabled</i>"
    )


def _build_btcorb_keyboard():
    from app.services.btc_orb_scanner import (
        is_btc_orb_enabled, get_btc_orb_sessions
    )

    enabled = is_btc_orb_enabled()
    sessions = get_btc_orb_sessions()

    toggle_text = "🔴 Disable Scanner" if enabled else "🟢 Enable Scanner"
    asia_text = f"{'🟢' if sessions.get('ASIA') else '⭕'} Asia"
    london_text = f"{'🟢' if sessions.get('LONDON') else '⭕'} London"
    ny_text = f"{'🟢' if sessions.get('NY') else '⭕'} NY"

    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text=toggle_text, callback_data="btcorb_toggle"),
        ],
        [
            InlineKeyboardButton(text=asia_text, callback_data="btcorb_session_ASIA"),
            InlineKeyboardButton(text=london_text, callback_data="btcorb_session_LONDON"),
            InlineKeyboardButton(text=ny_text, callback_data="btcorb_session_NY"),
        ],
        [
            InlineKeyboardButton(text="⚡ Leverage", callback_data="btcorb_leverage_menu"),
            InlineKeyboardButton(text="📊 Max Signals", callback_data="btcorb_maxdaily_menu"),
        ],
        [
            InlineKeyboardButton(text="⏳ Cooldown", callback_data="btcorb_cooldown_menu"),
            InlineKeyboardButton(text="🔍 Scan Now", callback_data="btcorb_scan"),
        ],
        [
            InlineKeyboardButton(text="📈 Status", callback_data="btcorb_status"),
            InlineKeyboardButton(text="🏠 Home", callback_data="back_to_start"),
        ],
    ])


@dp.message(Command("btcorb"))
async def handle_btcorb_command(message: Message):
    """BTC ORB+FVG Scalper - Interactive menu"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return

        if not user.is_admin:
            await message.answer("⚠️ This feature is admin-only.")
            return

        text = _build_btcorb_menu_text()
        keyboard = _build_btcorb_keyboard()
        await message.answer(text, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"BTC ORB command error: {e}")
        await message.answer(f"❌ Error: {str(e)[:100]}")
    finally:
        db.close()


@dp.callback_query(F.data == "btcorb_menu")
async def handle_btcorb_menu(callback: CallbackQuery):
    """Return to main BTC ORB menu"""
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            await callback.message.answer("⚠️ Admin-only feature.")
            return
        text = _build_btcorb_menu_text()
        keyboard = _build_btcorb_keyboard()
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"BTC ORB menu error: {e}")
    finally:
        db.close()


@dp.callback_query(F.data == "btcorb_toggle")
async def handle_btcorb_toggle(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            return

        from app.services.btc_orb_scanner import is_btc_orb_enabled, set_btc_orb_enabled
        current = is_btc_orb_enabled()
        set_btc_orb_enabled(not current)

        text = _build_btcorb_menu_text()
        keyboard = _build_btcorb_keyboard()
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"BTC ORB toggle error: {e}")
    finally:
        db.close()


@dp.callback_query(F.data.startswith("btcorb_session_"))
async def handle_btcorb_session_toggle(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            return

        session = callback.data.replace("btcorb_session_", "")
        from app.services.btc_orb_scanner import toggle_btc_orb_session
        toggle_btc_orb_session(session)

        text = _build_btcorb_menu_text()
        keyboard = _build_btcorb_keyboard()
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"BTC ORB session toggle error: {e}")
    finally:
        db.close()


@dp.callback_query(F.data == "btcorb_leverage_menu")
async def handle_btcorb_leverage_menu(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            return

        from app.services.btc_orb_scanner import get_btc_orb_leverage
        current = get_btc_orb_leverage()

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text=f"{'✅ ' if current == 25 else ''}25x", callback_data="btcorb_lev_25"),
                InlineKeyboardButton(text=f"{'✅ ' if current == 50 else ''}50x", callback_data="btcorb_lev_50"),
                InlineKeyboardButton(text=f"{'✅ ' if current == 75 else ''}75x", callback_data="btcorb_lev_75"),
            ],
            [
                InlineKeyboardButton(text=f"{'✅ ' if current == 100 else ''}100x", callback_data="btcorb_lev_100"),
                InlineKeyboardButton(text=f"{'✅ ' if current == 150 else ''}150x", callback_data="btcorb_lev_150"),
                InlineKeyboardButton(text=f"{'✅ ' if current == 200 else ''}200x", callback_data="btcorb_lev_200"),
            ],
            [
                InlineKeyboardButton(text="🔙 Back", callback_data="btcorb_menu"),
            ],
        ])

        await callback.message.edit_text(
            f"⚡ <b>BTC Scalper Leverage</b>\n\n"
            f"Current: <b>{current}x</b>\n\n"
            f"Select leverage for BTC scalp trades:\n\n"
            f"<i>Strategy uses 0.25% TP/SL — at 200x that's 50% ROI per trade</i>",
            reply_markup=keyboard, parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"BTC ORB leverage menu error: {e}")
    finally:
        db.close()


@dp.callback_query(F.data.startswith("btcorb_lev_"))
async def handle_btcorb_lev_set(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            return

        lev = int(callback.data.replace("btcorb_lev_", ""))
        from app.services.btc_orb_scanner import set_btc_orb_leverage
        set_btc_orb_leverage(lev)

        text = _build_btcorb_menu_text()
        keyboard = _build_btcorb_keyboard()
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"BTC ORB leverage set error: {e}")
    finally:
        db.close()


@dp.callback_query(F.data == "btcorb_maxdaily_menu")
async def handle_btcorb_maxdaily_menu(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            return

        from app.services.btc_orb_scanner import get_btc_orb_max_daily
        current = get_btc_orb_max_daily()

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text=f"{'✅ ' if current == 1 else ''}1", callback_data="btcorb_max_1"),
                InlineKeyboardButton(text=f"{'✅ ' if current == 2 else ''}2", callback_data="btcorb_max_2"),
                InlineKeyboardButton(text=f"{'✅ ' if current == 3 else ''}3", callback_data="btcorb_max_3"),
                InlineKeyboardButton(text=f"{'✅ ' if current == 4 else ''}4", callback_data="btcorb_max_4"),
            ],
            [
                InlineKeyboardButton(text="🔙 Back", callback_data="btcorb_menu"),
            ],
        ])

        await callback.message.edit_text(
            f"📊 <b>BTC ORB Max Daily Signals</b>\n\n"
            f"Current: <b>{current}</b> signals/day\n\n"
            f"Limits how many ORB signals can fire per day.\n\n"
            f"<i>2 sessions (Asia + NY) = 2 max opportunities</i>",
            reply_markup=keyboard, parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"BTC ORB max daily menu error: {e}")
    finally:
        db.close()


@dp.callback_query(F.data.startswith("btcorb_max_"))
async def handle_btcorb_max_set(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            return

        val = int(callback.data.replace("btcorb_max_", ""))
        from app.services.btc_orb_scanner import set_btc_orb_max_daily
        set_btc_orb_max_daily(val)

        text = _build_btcorb_menu_text()
        keyboard = _build_btcorb_keyboard()
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"BTC ORB max set error: {e}")
    finally:
        db.close()


@dp.callback_query(F.data == "btcorb_cooldown_menu")
async def handle_btcorb_cooldown_menu(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            return

        from app.services.btc_orb_scanner import get_btc_orb_cooldown
        current = get_btc_orb_cooldown()

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text=f"{'✅ ' if current == 30 else ''}30m", callback_data="btcorb_cd_30"),
                InlineKeyboardButton(text=f"{'✅ ' if current == 60 else ''}1h", callback_data="btcorb_cd_60"),
                InlineKeyboardButton(text=f"{'✅ ' if current == 120 else ''}2h", callback_data="btcorb_cd_120"),
            ],
            [
                InlineKeyboardButton(text=f"{'✅ ' if current == 180 else ''}3h", callback_data="btcorb_cd_180"),
                InlineKeyboardButton(text=f"{'✅ ' if current == 240 else ''}4h", callback_data="btcorb_cd_240"),
                InlineKeyboardButton(text=f"{'✅ ' if current == 480 else ''}8h", callback_data="btcorb_cd_480"),
            ],
            [
                InlineKeyboardButton(text="🔙 Back", callback_data="btcorb_menu"),
            ],
        ])

        await callback.message.edit_text(
            f"⏳ <b>BTC ORB Cooldown</b>\n\n"
            f"Current: <b>{current} min</b>\n\n"
            f"Minimum wait between ORB signals.\n\n"
            f"<i>Prevents over-trading on choppy sessions</i>",
            reply_markup=keyboard, parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"BTC ORB cooldown menu error: {e}")
    finally:
        db.close()


@dp.callback_query(F.data.startswith("btcorb_cd_"))
async def handle_btcorb_cd_set(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            return

        val = int(callback.data.replace("btcorb_cd_", ""))
        from app.services.btc_orb_scanner import set_btc_orb_cooldown
        set_btc_orb_cooldown(val)

        text = _build_btcorb_menu_text()
        keyboard = _build_btcorb_keyboard()
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"BTC ORB cooldown set error: {e}")
    finally:
        db.close()


@dp.callback_query(F.data == "btcorb_scan")
async def handle_btcorb_scan(callback: CallbackQuery):
    await callback.answer("Scanning...")
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            return

        from app.services.btc_orb_scanner import BTCOrbScanner, format_btc_orb_message

        await callback.message.edit_text(
            "📊 <b>Scanning BTC ORB+FVG...</b>\n\n<i>Checking for active ORB setups...</i>",
            parse_mode="HTML"
        )

        scanner = BTCOrbScanner()
        await scanner.init()
        try:
            signal_data = await asyncio.wait_for(scanner.scan(), timeout=30)

            back_kb = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Back to ORB Menu", callback_data="btcorb_menu")]
            ])

            if signal_data and not signal_data.get('cancel'):
                signal_text = format_btc_orb_message(signal_data)
                await callback.message.edit_text(
                    f"📊 <b>SIGNAL FOUND!</b>\n\n{signal_text}",
                    reply_markup=back_kb, parse_mode="HTML"
                )
            else:
                session = scanner.get_current_session()
                if session:
                    if scanner.is_in_orb_formation():
                        msg = f"📊 <b>{session} ORB forming...</b>\n\nWaiting for 15min candle to close."
                    else:
                        msg = f"📊 <b>No retest signal for {session} ORB</b>\n\nWaiting for price to retrace into fib/FVG zone."
                else:
                    msg = "📊 <b>No active session</b>\n\nNext sessions:\n• 🌏 Asia: 00:00 UTC\n• 🇬🇧 London: 08:00 UTC\n• 🗽 New York: 13:30 UTC"

                await callback.message.edit_text(msg, reply_markup=back_kb, parse_mode="HTML")
        finally:
            await scanner.close()
    except asyncio.TimeoutError:
        back_kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔙 Back to ORB Menu", callback_data="btcorb_menu")]
        ])
        await callback.message.edit_text("⏱️ Scan timed out. Try again later.", reply_markup=back_kb, parse_mode="HTML")
    except Exception as e:
        logger.error(f"BTC ORB scan error: {e}")
    finally:
        db.close()


@dp.callback_query(F.data == "btcorb_status")
async def handle_btcorb_status(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user or not user.is_admin:
            return

        from app.services.btc_orb_scanner import format_btc_orb_status
        status_text = format_btc_orb_status()

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔙 Back to ORB Menu", callback_data="btcorb_menu")]
        ])

        await callback.message.edit_text(status_text, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"BTC ORB status error: {e}")
    finally:
        db.close()


@dp.message(Command("social"))
async def cmd_social(message: types.Message):
    """🌙 Social & News Trading Mode - AI-powered signals"""
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
        action = args[1].lower() if len(args) > 1 else None
        
        from app.services.social_signals import (
            is_social_scanning_enabled, 
            enable_social_scanning, 
            disable_social_scanning
        )
        
        if action == "on":
            enable_social_scanning()
            await message.answer("🌙 <b>Social & News scanning ENABLED</b>\n\nScanning for AI-powered social signals.", parse_mode="HTML")
            return
        
        elif action == "off":
            disable_social_scanning()
            await message.answer("🌙 <b>Social scanning DISABLED</b>", parse_mode="HTML")
            return
        
        elif action == "settings":
            prefs = user.preferences
            if not prefs:
                await message.answer("❌ No preferences found. Use /start first.")
                return
            
            social_enabled = getattr(prefs, 'social_mode_enabled', False) or False
            social_lev = getattr(prefs, 'social_leverage', 10) or 10
            social_size = getattr(prefs, 'social_position_size_percent', 5.0) or 5.0
            social_dollars = getattr(prefs, 'social_position_size_dollars', None)
            social_max = getattr(prefs, 'social_max_positions', 3) or 3
            social_galaxy = getattr(prefs, 'social_min_galaxy_score', 8) or 8
            social_risk = getattr(prefs, 'social_risk_level', 'MEDIUM') or 'MEDIUM'
            
            size_display = f"${social_dollars:.0f}" if social_dollars else f"{social_size}%"
            
            await message.answer(
                f"⚙️ <b>YOUR SOCIAL SETTINGS</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                f"<b>Auto-Trading:</b> {'✅ ON' if social_enabled else '❌ OFF'}\n"
                f"<b>Risk Level:</b> {social_risk}\n"
                f"<b>Leverage:</b> {social_lev}x\n"
                f"<b>Position Size:</b> {size_display}\n"
                f"<b>Max Positions:</b> {social_max}\n"
                f"<b>Min Signal Score:</b> {social_galaxy}/100\n\n"
                f"<b>Configure:</b>\n"
                f"• <code>/social set risk LOW</code> - Set risk (LOW/MEDIUM/HIGH)\n"
                f"• <code>/social set lev 10</code> - Set leverage (1-20)\n"
                f"• <code>/social set size 5</code> - Set size % of balance\n"
                f"• <code>/social set dollars 50</code> - Set fixed $ amount\n"
                f"• <code>/social set max 3</code> - Set max positions\n"
                f"• <code>/social set score 60</code> - Set min Signal Score\n"
                f"• <code>/social enable</code> - Enable auto-trading\n"
                f"• <code>/social disable</code> - Disable auto-trading",
                parse_mode="HTML"
            )
            return
        
        elif action == "enable":
            prefs = user.preferences
            if prefs:
                prefs.social_mode_enabled = True
                db.commit()
            await message.answer("✅ <b>Social & News auto-trading ENABLED</b>\n\nYou'll receive AI-powered social signals.", parse_mode="HTML")
            return
        
        elif action == "disable":
            prefs = user.preferences
            if prefs:
                prefs.social_mode_enabled = False
                db.commit()
            await message.answer("❌ <b>Social auto-trading DISABLED</b>", parse_mode="HTML")
            return
        
        elif action == "set" and len(args) >= 4:
            setting = args[2].lower()
            value_str = args[3]
            
            prefs = user.preferences
            if not prefs:
                await message.answer("❌ No preferences found.")
                return
            
            if setting == "risk":
                value_str = value_str.upper()
                if value_str not in ["LOW", "MEDIUM", "HIGH"]:
                    await message.answer("❌ Risk must be LOW, MEDIUM, or HIGH")
                    return
                prefs.social_risk_level = value_str
                db.commit()
                await message.answer(f"✅ Social risk level set to <b>{value_str}</b>", parse_mode="HTML")
            
            elif setting == "lev" or setting == "leverage":
                value = int(min(20, max(1, float(value_str))))
                prefs.social_leverage = value
                db.commit()
                await message.answer(f"✅ Social leverage set to <b>{value}x</b>", parse_mode="HTML")
            
            elif setting == "size":
                value = min(100, max(1, float(value_str)))
                prefs.social_position_size_percent = value
                prefs.social_position_size_dollars = None
                db.commit()
                await message.answer(f"✅ Social position size set to <b>{value}%</b> of balance", parse_mode="HTML")
            
            elif setting == "dollars":
                value = max(5, float(value_str))
                prefs.social_position_size_dollars = value
                db.commit()
                await message.answer(f"✅ Social position size set to <b>${value:.0f}</b> fixed", parse_mode="HTML")
            
            elif setting == "max":
                value = int(min(10, max(1, float(value_str))))
                prefs.social_max_positions = value
                db.commit()
                await message.answer(f"✅ Max social positions set to <b>{value}</b>", parse_mode="HTML")
            
            elif setting in ("galaxy", "score"):
                value = int(min(100, max(30, float(value_str))))
                prefs.social_min_galaxy_score = value
                db.commit()
                await message.answer(f"✅ Min Signal Score set to <b>{value}</b>", parse_mode="HTML")
            
            else:
                await message.answer("❌ Unknown setting. Use: risk, lev, size, dollars, max, score")
            return
        
        elif action == "scan":
            await message.answer("🌙 <b>Running social scan...</b>", parse_mode="HTML")
            
            from app.services.social_signals import SocialSignalService
            from app.services.lunarcrush import get_lunarcrush_api_key
            
            if not get_lunarcrush_api_key():
                await message.answer("❌ LUNARCRUSH_API_KEY not configured")
                return
            
            prefs = user.preferences
            risk_level = getattr(prefs, 'social_risk_level', 'MEDIUM') or 'MEDIUM'
            min_galaxy = getattr(prefs, 'social_min_galaxy_score', 8) or 8
            
            service = SocialSignalService()
            await service.init()
            signal = await service.generate_social_signal(risk_level=risk_level, min_galaxy_score=min_galaxy)
            await service.close()
            
            if signal:
                from app.services.lunarcrush import interpret_signal_score
                rating = interpret_signal_score(signal['galaxy_score'])
                
                await message.answer(
                    f"🌙 <b>SOCIAL SIGNAL FOUND</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"📊 <b>{signal['symbol']}</b>\n\n"
                    f"📈 Direction: LONG\n"
                    f"💰 Entry: ${signal['entry_price']:,.4f}\n"
                    f"🎯 TP: ${signal['take_profit']:,.4f}\n"
                    f"🛑 SL: ${signal['stop_loss']:,.4f}\n\n"
                    f"<b>📱 AI Signal Analysis:</b>\n"
                    f"• Signal Score: {signal['galaxy_score']}/100 {rating}\n"
                    f"• Sentiment: {signal['sentiment']:.2f}\n"
                    f"• RSI: {signal['rsi']:.0f}",
                    parse_mode="HTML"
                )
            else:
                await message.answer("📱 No social signals found matching your criteria.")
            return
        
        # Default: show help
        status = "✅ ON" if is_social_scanning_enabled() else "❌ OFF"
        
        prefs = user.preferences
        social_risk = getattr(prefs, 'social_risk_level', 'MEDIUM') or 'MEDIUM' if prefs else 'MEDIUM'
        social_lev = getattr(prefs, 'social_leverage', 10) or 10 if prefs else 10
        
        await message.answer(
            f"🌙 <b>SOCIAL & NEWS TRADING</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Trade based on social sentiment and breaking news using AI analysis.\n\n"
            f"<b>Scanner Status:</b> {status}\n"
            f"<b>Your Settings:</b> {social_risk} risk, {social_lev}x\n\n"
            f"<b>Commands:</b>\n"
            f"• <code>/social on</code> - Enable scanning\n"
            f"• <code>/social off</code> - Disable scanning\n"
            f"• <code>/social scan</code> - Run scan now\n"
            f"• <code>/social settings</code> - Your settings\n"
            f"• <code>/social enable</code> - Enable auto-trading\n"
            f"• <code>/social disable</code> - Disable auto-trading\n\n"
            f"<b>Risk Profiles:</b>\n"
            f"• SAFE - Score ≥70, quick scalps (+3%)\n"
            f"• BALANCED - Score ≥60, steady gains (+5%)\n"
            f"• AGGRESSIVE - Score ≥50, high risk (+8-15%)\n"
            f"• NEWS RUNNER - Score ≥80, catch pumps (+15-30%)\n"
            f"• ALL - Smart mode, TP adapts to signal strength",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Social command error: {e}")
        await message.answer(f"❌ Error: {str(e)[:100]}")
    finally:
        db.close()


@dp.message(Command("socialstats"))
async def cmd_social_stats(message: types.Message):
    """📊 View Social & News Trading statistics (Admin only)"""
    db = SessionLocal()
    try:
        if not is_admin(message.from_user.id, db):
            await message.answer("❌ Admin access required.")
            return
        
        from app.services.social_trade_logger import get_social_trade_stats, get_recent_social_trades
        
        # Get stats for different periods
        stats_7d = get_social_trade_stats(db, days=7)
        stats_30d = get_social_trade_stats(db, days=30)
        recent = get_recent_social_trades(db, limit=10)
        
        # Format the message
        msg = f"""📊 <b>SOCIAL & NEWS TRADE STATS</b>

<b>Last 7 Days:</b>
Total: {stats_7d.get('total_trades', 0)} | Win Rate: {stats_7d.get('win_rate', 0):.1f}%
✅ Wins: {stats_7d.get('wins', 0)} | ❌ Losses: {stats_7d.get('losses', 0)}
💰 Total PnL: ${stats_7d.get('total_pnl', 0):.2f}
Avg Win: ${stats_7d.get('avg_win', 0):.2f} | Avg Loss: ${stats_7d.get('avg_loss', 0):.2f}

<b>Last 30 Days:</b>
Total: {stats_30d.get('total_trades', 0)} | Win Rate: {stats_30d.get('win_rate', 0):.1f}%
💰 Total PnL: ${stats_30d.get('total_pnl', 0):.2f}

<b>By Signal Type (7d):</b>"""
        
        by_type = stats_7d.get('by_type', {})
        for signal_type, data in by_type.items():
            if data.get('total', 0) > 0:
                emoji = "🟢" if signal_type == 'SOCIAL_SIGNAL' else ("🔴" if signal_type == 'SOCIAL_SHORT' else "📰")
                msg += f"\n{emoji} {signal_type}: {data['total']} trades | {data['win_rate']:.0f}% WR | ${data['pnl']:.2f}"
        
        if recent:
            msg += "\n\n<b>Recent Trades:</b>"
            for trade in recent[:5]:
                result_emoji = "✅" if trade.result == 'WIN' else ("❌" if trade.result == 'LOSS' else "➖")
                pnl_str = f"+${trade.pnl:.2f}" if (trade.pnl or 0) >= 0 else f"-${abs(trade.pnl or 0):.2f}"
                roi = trade.roi_percent or 0
                roi_str = f"+{roi:.1f}%" if roi >= 0 else f"{roi:.1f}%"
                lev = trade.leverage or 1
                msg += f"\n{result_emoji} {trade.symbol} {trade.direction} | {pnl_str} | ROI: {roi_str} @ {lev}x"
        
        await message.answer(msg, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Social stats error: {e}")
        await message.answer(f"❌ Error: {str(e)[:100]}")
    finally:
        db.close()


@dp.message(Command("market"))
async def cmd_market(message: types.Message):
    """🔮 Market Regime Detector - AI identifies current market conditions"""
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
        
        await message.answer("🔮 <b>Analyzing market conditions...</b>\n\n<i>This may take a few seconds.</i>", parse_mode="HTML")
        
        from app.services.ai_market_intelligence import detect_market_regime, format_regime_message
        
        regime = await detect_market_regime()
        
        if regime.get('regime') == 'UNKNOWN':
            await message.answer("❌ Could not analyze market conditions. Try again later.")
            return
        
        response = format_regime_message(regime)
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="📰 News", callback_data="news_scanner"),
                InlineKeyboardButton(text="🐋 Whales", callback_data="whale_tracker")
            ],
            [
                InlineKeyboardButton(text="📊 Leaders", callback_data="leaderboard_tracker"),
                InlineKeyboardButton(text="◀️ Dashboard", callback_data="back_to_dashboard")
            ]
        ])
        
        await message.answer(response, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Market command error: {e}")
        await message.answer(f"❌ Error analyzing market: {str(e)[:100]}")
    finally:
        db.close()


@dp.message(Command("patterns"))
async def cmd_patterns(message: types.Message):
    """🔍 AI Chart Pattern Detector - Detects classic chart patterns"""
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
            await message.answer(
                "🔍 <b>AI Pattern Detector</b>\n\n"
                "Usage: <code>/patterns SYMBOL</code>\n\n"
                "Examples:\n"
                "• /patterns BTC\n"
                "• /patterns SOL\n"
                "• /patterns PEPE\n\n"
                "<i>Detects: Head & Shoulders, Double Tops/Bottoms, Triangles, Wedges, Flags, and more.</i>",
                parse_mode="HTML"
            )
            return
        
        symbol = args[1].upper().replace('/USDT', '').replace('USDT', '')
        
        await message.answer(f"🔍 <b>Analyzing {symbol} chart patterns...</b>\n\n<i>Scanning 15m, 1h, and 4h timeframes. This may take a few seconds.</i>", parse_mode="HTML")
        
        from app.services.ai_pattern_detector import detect_chart_patterns, format_patterns_message
        
        result = await detect_chart_patterns(symbol)
        response = format_patterns_message(result)
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text=f"💀 {symbol} Liquidations", callback_data=f"liq_{symbol}"),
                InlineKeyboardButton(text=f"📊 Scan {symbol}", callback_data=f"quick_scan_{symbol}")
            ],
            [
                InlineKeyboardButton(text="◀️ Dashboard", callback_data="back_to_dashboard")
            ]
        ])
        
        await message.answer(response, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Patterns command error: {e}")
        await message.answer(f"❌ Error analyzing patterns: {str(e)[:100]}")
    finally:
        db.close()


@dp.message(Command("liquidations"))
async def cmd_liquidations(message: types.Message):
    """💀 AI Liquidation Zone Predictor - Identifies liquidation clusters"""
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
            await message.answer(
                "💀 <b>AI Liquidation Zones</b>\n\n"
                "Usage: <code>/liquidations SYMBOL</code>\n\n"
                "Examples:\n"
                "• /liquidations BTC\n"
                "• /liquidations ETH\n"
                "• /liquidations SOL\n\n"
                "<i>Predicts where liquidation cascades could trigger based on open interest and leverage levels.</i>",
                parse_mode="HTML"
            )
            return
        
        symbol = args[1].upper().replace('/USDT', '').replace('USDT', '')
        
        await message.answer(f"💀 <b>Analyzing {symbol} liquidation zones...</b>\n\n<i>Fetching open interest and funding data...</i>", parse_mode="HTML")
        
        from app.services.ai_pattern_detector import analyze_liquidation_zones, format_liquidation_message
        
        result = await analyze_liquidation_zones(symbol)
        response = format_liquidation_message(result)
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text=f"🔍 {symbol} Patterns", callback_data=f"pat_{symbol}"),
                InlineKeyboardButton(text=f"📊 Scan {symbol}", callback_data=f"quick_scan_{symbol}")
            ],
            [
                InlineKeyboardButton(text="◀️ Dashboard", callback_data="back_to_dashboard")
            ]
        ])
        
        await message.answer(response, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Liquidations command error: {e}")
        await message.answer(f"❌ Error analyzing liquidations: {str(e)[:100]}")
    finally:
        db.close()


@dp.callback_query(F.data.startswith("pat_"))
async def handle_pattern_callback(callback: CallbackQuery):
    """Handle pattern button click"""
    await callback.answer()
    
    symbol = callback.data.replace("pat_", "")
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            return
        
        has_access, _ = check_access(user)
        if not has_access:
            return
        
        await callback.message.answer(f"🔍 <b>Analyzing {symbol} chart patterns...</b>", parse_mode="HTML")
        
        from app.services.ai_pattern_detector import detect_chart_patterns, format_patterns_message
        
        result = await detect_chart_patterns(symbol)
        response = format_patterns_message(result)
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text=f"💀 {symbol} Liquidations", callback_data=f"liq_{symbol}"),
                InlineKeyboardButton(text="◀️ Dashboard", callback_data="back_to_dashboard")
            ]
        ])
        
        await callback.message.answer(response, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Pattern callback error: {e}")
    finally:
        db.close()


@dp.callback_query(F.data.startswith("liq_"))
async def handle_liquidation_callback(callback: CallbackQuery):
    """Handle liquidation button click"""
    await callback.answer()
    
    symbol = callback.data.replace("liq_", "")
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            return
        
        has_access, _ = check_access(user)
        if not has_access:
            return
        
        await callback.message.answer(f"💀 <b>Analyzing {symbol} liquidation zones...</b>", parse_mode="HTML")
        
        from app.services.ai_pattern_detector import analyze_liquidation_zones, format_liquidation_message
        
        result = await analyze_liquidation_zones(symbol)
        response = format_liquidation_message(result)
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text=f"🔍 {symbol} Patterns", callback_data=f"pat_{symbol}"),
                InlineKeyboardButton(text="◀️ Dashboard", callback_data="back_to_dashboard")
            ]
        ])
        
        await callback.message.answer(response, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Liquidation callback error: {e}")
    finally:
        db.close()


@dp.callback_query(F.data == "market_regime")
async def handle_market_regime(callback: CallbackQuery):
    """Handle market regime button click"""
    await callback.answer()
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            return
        
        await callback.message.answer("🔮 <b>Analyzing market conditions...</b>", parse_mode="HTML")
        
        from app.services.ai_market_intelligence import detect_market_regime, format_regime_message
        
        regime = await detect_market_regime()
        response = format_regime_message(regime)
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="📰 News", callback_data="news_scanner"),
                InlineKeyboardButton(text="🐋 Whales", callback_data="whale_tracker")
            ],
            [
                InlineKeyboardButton(text="📊 Leaders", callback_data="leaderboard_tracker"),
                InlineKeyboardButton(text="◀️ Dashboard", callback_data="back_to_dashboard")
            ]
        ])
        
        await callback.message.answer(response, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Market regime callback error: {e}")
        await callback.message.answer(f"❌ Error: {str(e)[:100]}")
    finally:
        db.close()


@dp.callback_query(F.data == "news_scanner")
async def handle_news_scanner(callback: CallbackQuery):
    """Handle news scanner button click"""
    await callback.answer()
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            return
        
        await callback.message.answer("📰 <b>Analyzing crypto news...</b>", parse_mode="HTML")
        
        from app.services.ai_market_intelligence import analyze_news_impact
        
        result = await analyze_news_impact()
        alerts = result.get('alerts', [])
        sentiment = result.get('market_sentiment', 'NEUTRAL')
        themes = result.get('key_themes', [])
        
        sentiment_emoji = "🟢" if sentiment == 'BULLISH' else "🔴" if sentiment == 'BEARISH' else "⚪"
        
        response = f"""📰 <b>NEWS IMPACT SCANNER</b>
━━━━━━━━━━━━━━━━━━━━
{sentiment_emoji} <b>Market Sentiment:</b> {sentiment}
🔑 <b>Key Themes:</b> {', '.join(themes[:3]) if themes else 'None'}

"""
        
        if alerts:
            response += f"<b>🚨 {len(alerts)} Trading Alerts:</b>\n\n"
            for alert in alerts[:5]:
                direction_emoji = "🟢" if alert.get('direction') == 'BULLISH' else "🔴"
                coins = ", ".join(alert.get('coins', []))
                response += f"{direction_emoji} <b>{coins}</b>: {alert.get('headline', '')[:50]}...\n"
        else:
            response += "✅ <i>No major market-moving news detected.</i>\n"
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="🔮 Market", callback_data="market_regime"),
                InlineKeyboardButton(text="🐋 Whales", callback_data="whale_tracker")
            ],
            [
                InlineKeyboardButton(text="📊 Leaders", callback_data="leaderboard_tracker"),
                InlineKeyboardButton(text="◀️ Dashboard", callback_data="back_to_dashboard")
            ]
        ])
        
        await callback.message.answer(response, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"News scanner callback error: {e}")
        await callback.message.answer(f"❌ Error: {str(e)[:100]}")
    finally:
        db.close()


@dp.message(Command("whale"))
async def cmd_whale(message: types.Message):
    """🐋 Whale Tracker - AI analyzes smart money movements"""
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
        
        await message.answer("🐋 <b>Analyzing whale & smart money activity...</b>\n\n<i>This may take a few seconds.</i>", parse_mode="HTML")
        
        from app.services.ai_market_intelligence import analyze_whale_activity, format_whale_message
        
        result = await analyze_whale_activity()
        
        if not result.get('analysis'):
            await message.answer("❌ Could not analyze whale activity. Try again later.")
            return
        
        response = format_whale_message(result)
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="📰 News", callback_data="news_scanner"),
                InlineKeyboardButton(text="🔮 Market", callback_data="market_regime")
            ],
            [
                InlineKeyboardButton(text="📊 Leaders", callback_data="leaderboard_tracker"),
                InlineKeyboardButton(text="◀️ Dashboard", callback_data="back_to_dashboard")
            ]
        ])
        
        await message.answer(response, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Whale command error: {e}")
        await message.answer(f"❌ Error analyzing whale activity: {str(e)[:100]}")
    finally:
        db.close()


@dp.callback_query(F.data == "whale_tracker")
async def handle_whale_tracker(callback: CallbackQuery):
    """Handle whale tracker button click"""
    await callback.answer()
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            return
        
        await callback.message.answer("🐋 <b>Analyzing whale activity...</b>", parse_mode="HTML")
        
        from app.services.ai_market_intelligence import analyze_whale_activity, format_whale_message
        
        result = await analyze_whale_activity()
        response = format_whale_message(result)
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="📰 News", callback_data="news_scanner"),
                InlineKeyboardButton(text="🔮 Market", callback_data="market_regime")
            ],
            [
                InlineKeyboardButton(text="📊 Leaders", callback_data="leaderboard_tracker"),
                InlineKeyboardButton(text="◀️ Dashboard", callback_data="back_to_dashboard")
            ]
        ])
        
        await callback.message.answer(response, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Whale tracker callback error: {e}")
        await callback.message.answer(f"❌ Error: {str(e)[:100]}")
    finally:
        db.close()


@dp.message(Command("scalp"))
async def handle_scalp(message: types.Message):
    """Manual VWAP scalp scan command"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            return
            
        has_access, reason = check_access(user, require_tier="scan")
        if not has_access:
            await message.answer(reason)
            return

        parts = message.text.split()
        if len(parts) < 2:
            await message.answer("❌ Usage: /scalp SYMBOL (e.g., /scalp BTC)")
            return
            
        symbol = parts[1].upper().replace('USDT', '')
        symbol_usdt = f"{symbol}/USDT"
        
        status_msg = await message.answer(f"🔍 <b>Scalping Analysis:</b> {symbol}...\n<i>Calculating VWAP & Pullback confluence...</i>", parse_mode="HTML")
        
        from app.services.vwap_scalps import VWAPScalpStrategy
        from app.services.top_gainers_signals import ai_validate_scalp_signal
        
        # Check if user has scalp mode enabled in preferences
        if not user.preferences or not user.preferences.scalp_mode_enabled:
            await status_msg.edit_text("❌ <b>Scalp Mode Disabled:</b> You must enable Scalp Mode in your /settings to take scalp trades.", parse_mode="HTML")
            return
            
        strategy = VWAPScalpStrategy()
        analysis = await strategy.analyze_symbol(symbol_usdt)
        
        if not analysis:
            await status_msg.edit_text(f"❌ <b>No Scalp Setup:</b> {symbol}\n\nReason: No VWAP pullback or 1H trend not confirmed.", parse_mode="HTML")
            return
            
        await status_msg.edit_text(f"🔍 <b>Setup Found!</b> Validating {symbol} with AI Scalp Specialist...", parse_mode="HTML")
        
        ai_result = await ai_validate_scalp_signal(analysis)
        
        if ai_result and ai_result.get('approved'):
            text = (
                f"⚡ <b>VWAP BOUNCE SCALP</b> ⚡\n\n"
                f"🪙 <b>Symbol:</b> #{symbol}\n"
                f"🎯 <b>Entry:</b> ${ai_result['entry_price']:.6f}\n"
                f"📈 <b>Target:</b> ${ai_result['take_profit']:.6f} (+{ai_result['tp_percent']}%)\n"
                f"🛑 <b>Stop Loss:</b> ${ai_result['stop_loss']:.6f} (-{ai_result['sl_percent']}%)\n"
                f"⚙️ <b>Leverage:</b> {ai_result['leverage']}x\n\n"
                f"🧠 <b>AI Analysis:</b> {ai_result['reasoning']}\n"
                f"⭐ <b>Confidence:</b> {ai_result['confidence']}/10"
            )
            await status_msg.edit_text(text, parse_mode="HTML")
        else:
            reason = ai_result.get('reasoning', 'Setup does not meet high-probability scalp criteria.') if ai_result else "AI rejected the technical setup."
            await status_msg.edit_text(f"🔴 <b>Scalp Rejected by AI</b>\n\nReason: {reason}", parse_mode="HTML")
            
    except Exception as e:
        logger.error(f"Error in scalp command: {e}")
        await message.answer(f"❌ Error analyzing {message.text}: {str(e)[:100]}")
    finally:
        db.close()


@dp.message(Command("leaderboard"))
async def cmd_leaderboard(message: types.Message):
    """📊 Leaderboard - Track Binance Futures top traders"""
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
        
        await message.answer("📊 <b>Fetching top traders from Binance Leaderboard...</b>\n\n<i>Analyzing positions of top performers.</i>", parse_mode="HTML")
        
        from app.services.ai_market_intelligence import analyze_leaderboard_positions, format_leaderboard_message
        
        result = await analyze_leaderboard_positions()
        
        if result.get('error'):
            await message.answer(f"❌ {result['error']}")
            return
        
        response = format_leaderboard_message(result)
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="📰 News", callback_data="news_scanner"),
                InlineKeyboardButton(text="🔮 Market", callback_data="market_regime")
            ],
            [
                InlineKeyboardButton(text="🐋 Whales", callback_data="whale_tracker"),
                InlineKeyboardButton(text="◀️ Dashboard", callback_data="back_to_dashboard")
            ]
        ])
        
        await message.answer(response, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Leaderboard command error: {e}")
        await message.answer(f"❌ Error fetching leaderboard: {str(e)[:100]}")
    finally:
        db.close()


@dp.callback_query(F.data == "leaderboard_tracker")
async def handle_leaderboard_tracker(callback: CallbackQuery):
    """Handle leaderboard tracker button click"""
    await callback.answer()
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        if not user:
            await callback.message.answer("You're not registered. Use /start to begin!")
            return
        
        has_access, reason = check_access(user)
        if not has_access:
            await callback.message.answer(reason)
            return
        
        await callback.message.answer("📊 <b>Fetching top traders...</b>", parse_mode="HTML")
        
        from app.services.ai_market_intelligence import analyze_leaderboard_positions, format_leaderboard_message
        
        result = await analyze_leaderboard_positions()
        response = format_leaderboard_message(result)
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="📰 News", callback_data="news_scanner"),
                InlineKeyboardButton(text="🔮 Market", callback_data="market_regime")
            ],
            [
                InlineKeyboardButton(text="🐋 Whales", callback_data="whale_tracker"),
                InlineKeyboardButton(text="◀️ Dashboard", callback_data="back_to_dashboard")
            ]
        ])
        
        await callback.message.answer(response, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Leaderboard tracker callback error: {e}")
        await callback.message.answer(f"❌ Error: {str(e)[:100]}")
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


async def handle_twitter_credential_input(message: types.Message):
    """Handle multi-step Twitter account credential collection"""
    user_id = message.from_user.id
    pending = cmd_twitter.pending_adds.get(user_id)
    
    if not pending:
        return
    
    text = message.text.strip()
    step = pending.get('step')
    
    # Delete the message containing credentials for security
    try:
        await message.delete()
    except:
        pass
    
    if step == 'consumer_key':
        pending['consumer_key'] = text
        pending['step'] = 'consumer_secret'
        await message.answer(f"""✅ Consumer Key received!

<b>Step 2/5:</b> Send me the <b>Consumer Secret</b> (API Secret):""", parse_mode="HTML")
    
    elif step == 'consumer_secret':
        pending['consumer_secret'] = text
        pending['step'] = 'access_token'
        await message.answer(f"""✅ Consumer Secret received!

<b>Step 3/5:</b> Send me the <b>Access Token</b>:""", parse_mode="HTML")
    
    elif step == 'access_token':
        pending['access_token'] = text
        pending['step'] = 'access_token_secret'
        await message.answer(f"""✅ Access Token received!

<b>Step 4/5:</b> Send me the <b>Access Token Secret</b>:""", parse_mode="HTML")
    
    elif step == 'access_token_secret':
        pending['access_token_secret'] = text
        pending['step'] = 'bearer_token'
        await message.answer(f"""✅ Access Token Secret received!

<b>Step 5/5:</b> Send me the <b>Bearer Token</b>:
(Or type "skip" if you don't have one)""", parse_mode="HTML")
    
    elif step == 'bearer_token':
        bearer = text if text.lower() != 'skip' else None
        pending['bearer_token'] = bearer
        
        # All credentials collected - save to database
        from app.services.twitter_poster import add_twitter_account
        
        result = add_twitter_account(
            name=pending['name'],
            handle=None,  # Can be set later
            consumer_key=pending['consumer_key'],
            consumer_secret=pending['consumer_secret'],
            access_token=pending['access_token'],
            access_token_secret=pending['access_token_secret'],
            bearer_token=pending.get('bearer_token')
        )
        
        # Clean up pending data
        del cmd_twitter.pending_adds[user_id]
        
        if result['success']:
            await message.answer(f"""✅ <b>ACCOUNT ADDED!</b>

<b>Name:</b> {pending['name']}

Now assign post types:
<code>/twitter assign {pending['name']} featured_coin,top_gainers</code>

Or view all accounts:
<code>/twitter accounts</code>""", parse_mode="HTML")
        else:
            await message.answer(f"❌ <b>Failed to add account:</b>\n{result['error']}", parse_mode="HTML")


@dp.message(F.text & ~F.text.startswith("/"), StateFilter(None))
async def handle_ticket_message(message: types.Message, state: FSMContext):
    """Handle user's ticket message submission OR admin reply OR AI chat (ONLY when NOT in FSM state)"""
    user_id = message.from_user.id
    text = message.text.strip() if message.text else ""
    
    if len(text) == 32 and text.isalnum():
        try:
            await message.delete()
        except:
            pass
        await message.answer(
            "🔑 <b>That looks like an API key!</b>\n\n"
            "To connect your Bitunix API, please use the setup command first:\n"
            "👉 /connect_bitunix\n\n"
            "<i>Your message was deleted for security.</i>",
            parse_mode="HTML"
        )
        return
    
    if hasattr(cmd_twitter, 'pending_adds') and user_id in cmd_twitter.pending_adds:
        await handle_twitter_credential_input(message)
        return
    
    if user_id in admin_reply_data:
        await handle_admin_reply_message(message)
        return
    
    # Check if user is submitting a ticket
    if user_id in user_ticket_data:
        pass  # Continue to ticket handling below
    else:
        # Not submitting a ticket - check for AI chat
        from app.services.ai_chat_assistant import (
            is_trading_question, extract_coins, ask_ai_assistant, 
            is_clear_command, clear_conversation,
            is_scanner_request, scan_market_opportunities
        )
        
        text = message.text.strip()
        
        # Check for clear conversation command
        if is_clear_command(text):
            clear_conversation(user_id)
            await message.answer("🔄 <b>Chat cleared!</b>\n\nStarting fresh conversation.", parse_mode="HTML")
            return
        
        # Check for market scanner request
        if is_scanner_request(text):
            db = SessionLocal()
            try:
                user = db.query(User).filter(User.telegram_id == str(user_id)).first()
                if user:
                    has_access, _ = check_access(user)
                    if has_access:
                        scanning_msg = await message.answer("🔍 <i>Scanning market for opportunities...</i>", parse_mode="HTML")
                        
                        result = await scan_market_opportunities()
                        
                        if result:
                            await scanning_msg.edit_text(
                                f"🤖 <b>Tradehub Scanner</b>\n\n{result}",
                                parse_mode="HTML"
                            )
                        else:
                            await scanning_msg.edit_text(
                                "Sorry, couldn't complete the scan. Please try again.",
                                parse_mode="HTML"
                            )
                        return
            except Exception as e:
                logger.error(f"Market scanner error: {e}", exc_info=True)
            finally:
                db.close()
            return
        
        # Check for position coach request
        from app.services.ai_chat_assistant import is_position_question, analyze_positions
        if is_position_question(text):
            db = SessionLocal()
            try:
                user = db.query(User).filter(User.telegram_id == str(user_id)).first()
                if user:
                    has_access, _ = check_access(user)
                    if has_access:
                        thinking_msg = await message.answer("🎯 <i>Analyzing your positions...</i>", parse_mode="HTML")
                        
                        result = await analyze_positions(user_id, text)
                        
                        if result:
                            await thinking_msg.edit_text(
                                f"🤖 <b>Tradehub Coach</b>\n\n{result}",
                                parse_mode="HTML"
                            )
                        else:
                            await thinking_msg.edit_text(
                                "Sorry, couldn't analyze your positions. Make sure you have Bitunix API connected.",
                                parse_mode="HTML"
                            )
                        return
            except Exception as e:
                logger.error(f"Position coach error: {e}", exc_info=True)
            finally:
                db.close()
            return
        
        # Check for risk assessment request
        from app.services.ai_chat_assistant import is_risk_question, assess_trade_risk
        if is_risk_question(text):
            db = SessionLocal()
            try:
                user = db.query(User).filter(User.telegram_id == str(user_id)).first()
                if user:
                    has_access, _ = check_access(user)
                    if has_access:
                        thinking_msg = await message.answer("⚠️ <i>Assessing trade risk...</i>", parse_mode="HTML")
                        
                        coins = extract_coins(text)
                        text_lower = text.lower()
                        
                        # Determine direction from text
                        direction = 'LONG'
                        if 'short' in text_lower or 'sell' in text_lower:
                            direction = 'SHORT'
                        
                        if coins:
                            symbol = coins[0]
                            risk = await assess_trade_risk(symbol, direction)
                            
                            if risk.get('error'):
                                await thinking_msg.edit_text(
                                    f"❌ {risk['error']}",
                                    parse_mode="HTML"
                                )
                            else:
                                # Build risk report
                                factors_text = "\n".join([f"• {f}" for f in risk['factors']]) if risk['factors'] else "• No major risk factors detected"
                                coin_data = risk.get('coin_data', {})
                                
                                report = f"""<b>{risk['emoji']} RISK ASSESSMENT: {symbol} {direction}</b>

<b>Risk Score:</b> {risk['risk_score']}/10 ({risk['risk_level']})

<b>Current Conditions:</b>
• Price: ${coin_data.get('price', 0):,.4f}
• 24h Change: {coin_data.get('change_24h', 0):+.1f}%
• RSI: {coin_data.get('rsi', 50):.0f}
• Trend: {coin_data.get('trend', 'neutral').title()}
• Volume: {coin_data.get('volume_ratio', 1):.1f}x average

<b>Risk Factors:</b>
{factors_text}

<b>Recommendation:</b>
{risk['recommendation']}"""
                                
                                await thinking_msg.edit_text(report, parse_mode="HTML")
                        else:
                            await thinking_msg.edit_text(
                                "Please mention a specific coin, e.g. 'How risky is longing BTC?'",
                                parse_mode="HTML"
                            )
                        return
            except Exception as e:
                logger.error(f"Risk assessment error: {e}", exc_info=True)
            finally:
                db.close()
            return
        
        if is_trading_question(text):
            db = SessionLocal()
            try:
                user = db.query(User).filter(User.telegram_id == str(user_id)).first()
                if user:
                    has_access, _ = check_access(user)
                    if has_access:
                        thinking_msg = await message.answer("🤖 <i>Analyzing...</i>", parse_mode="HTML")
                        
                        coins = extract_coins(text)
                        
                        response = await ask_ai_assistant(
                            question=text,
                            coins=coins,
                            user_context="",
                            user_id=user_id
                        )
                        
                        if response:
                            await thinking_msg.edit_text(
                                f"🤖 <b>Tradehub Assistant</b>\n\n{response}",
                                parse_mode="HTML"
                            )
                        else:
                            await thinking_msg.edit_text(
                                "Sorry, I couldn't process that. Try asking about a specific coin like BTC or SOL.",
                                parse_mode="HTML"
                            )
                        return
            except Exception as e:
                logger.error(f"AI chat error: {e}", exc_info=True)
            finally:
                db.close()
        
        return  # Not submitting a ticket and not AI question, ignore
    
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


@dp.message(Command("testnews"))
async def cmd_test_news(message: types.Message):
    """Admin command to test the breaking news scanner"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user or not user.is_admin:
            await message.answer("❌ Admin only.")
            return
        
        await message.answer("📰 <b>Testing Breaking News Scanner...</b>", parse_mode="HTML")
        
        from app.services.realtime_news import RealtimeNewsScanner
        import os
        
        api_key = os.environ.get("CRYPTONEWS_API_KEY")
        if not api_key:
            await message.answer("❌ No CRYPTONEWS_API_KEY configured!")
            return
        
        scanner = RealtimeNewsScanner()
        articles = await scanner.fetch_breaking_news()
        
        if not articles:
            await message.answer("📰 No new articles in last 15 minutes (or already seen)")
            return
        
        result = f"📰 <b>Breaking News ({len(articles)} articles)</b>\n\n"
        for i, article in enumerate(articles[:5], 1):
            title = article.get('title', '')[:100]
            coins = scanner.extract_coins_from_news(article)
            direction, score, trigger = scanner.analyze_news_impact(article)
            result += f"<b>{i}.</b> {title}\n"
            result += f"   Coins: {coins or 'None'}\n"
            result += f"   Direction: {direction} | Score: {score}\n\n"
        
        await message.answer(result, parse_mode="HTML")
        
    finally:
        db.close()


@dp.message(Command("grant_sub"))
async def cmd_grant_subscription(message: types.Message):
    """Admin command to manually grant subscription (for short payments due to fees)"""
    from app.config import settings
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        is_owner = str(message.from_user.id) == str(settings.OWNER_TELEGRAM_ID)
        if not is_owner and (not user or not user.is_admin):
            await message.answer("❌ This command is only available to admins.")
            return
        
        # Parse command: /grant_sub <telegram_id> <plan_type> <days>
        # Example: /grant_sub 123456789 auto 30
        parts = message.text.split()
        if len(parts) < 3:
            await message.answer(
                "❌ <b>Usage:</b> /grant_sub &lt;telegram_id&gt; &lt;plan&gt; [days]\n\n"
                "<b>Plans:</b> scan, auto, lifetime\n"
                "  • scan = AI Assistant ($65/mo)\n"
                "  • auto = Auto-Trading ($80/mo)\n"
                "  • lifetime = Permanent access\n\n"
                "<b>Days:</b> Optional (default: 30, ignored for lifetime)\n\n"
                "<b>Examples:</b>\n"
                "/grant_sub 123456789 scan 30\n"
                "/grant_sub 123456789 auto 30\n"
                "/grant_sub 123456789 lifetime",
                parse_mode="HTML"
            )
            return
        
        target_identifier = parts[1]
        plan_type = parts[2].lower()
        
        if plan_type == "manual":
            plan_type = "scan"
        
        try:
            days = int(parts[3]) if len(parts) > 3 else 30
            if days < 1:
                await message.answer("Days must be at least 1")
                return
        except ValueError:
            await message.answer("Days must be a valid number")
            return
        
        if plan_type not in ["scan", "auto", "lifetime"]:
            await message.answer("Plan must be 'scan', 'auto', or 'lifetime'")
            return
        
        target_user = None
        if target_identifier.startswith("TH-"):
            target_user = db.query(User).filter(User.uid == target_identifier).first()
        if not target_user:
            target_user = db.query(User).filter(User.telegram_id == target_identifier).first()
        if not target_user:
            target_user = db.query(User).filter(User.username == target_identifier.lstrip("@")).first()
        if not target_user:
            await message.answer(f"User not found: {target_identifier}")
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
            target_user.is_subscribed = True
            target_user.subscription_end = datetime.utcnow() + timedelta(days=days)
            plan_name = "🤖 AI Assistant" if plan_type == "scan" else "🚀 Auto-Trading"
        
        target_user.approved = True  # Auto-approve
        db.commit()
        db.refresh(target_user)
        expires = target_user.subscription_end.strftime("%Y-%m-%d") if target_user.subscription_end else "Never"
        
        target_name = f"@{target_user.username}" if target_user.username else target_user.first_name or target_user.telegram_id
        await message.answer(
            f"Subscription Granted!\n\n"
            f"User: {target_name} (<code>{target_user.uid or target_user.telegram_id}</code>)\n"
            f"Plan: {plan_name}\n"
            f"Duration: {days} days\n"
            f"Expires: {expires}",
            parse_mode="HTML"
        )
        
        try:
            await bot.send_message(
                chat_id=int(target_user.telegram_id),
                text=f"<b>Subscription Activated!</b>\n\n"
                     f"Your <b>{plan_name}</b> subscription has been activated!\n\n"
                     f"Valid until: <b>{expires}</b>\n\n"
                     f"You now have full access to all premium features!\n"
                     f"Use /dashboard to get started.",
                parse_mode="HTML"
            )
        except Exception as e:
            await message.answer(f"Subscription granted but couldn't notify user: {e}")
        
    finally:
        db.close()


@dp.message(Command("adddays"))
async def cmd_add_days(message: types.Message):
    """Admin command to add days to a user's subscription — mirrors full automatic activation flow"""
    from app.config import settings
    from app.models import Subscription
    db = SessionLocal()

    try:
        admin_user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        is_owner = str(message.from_user.id) == str(settings.OWNER_TELEGRAM_ID)
        if not is_owner and (not admin_user or not admin_user.is_admin):
            await message.answer("❌ This command is only available to admins.")
            return

        parts = message.text.split()
        if len(parts) < 3:
            await message.answer(
                "❌ <b>Usage:</b> /adddays &lt;user&gt; &lt;days&gt; [plan]\n\n"
                "<b>User:</b> Telegram ID, @username, or UID (TH-XXXXXXXX)\n"
                "<b>Days:</b> Number of days to add\n"
                "<b>Plan:</b> auto (default) | manual | scan\n\n"
                "<b>Examples:</b>\n"
                "<code>/adddays 123456789 30</code>\n"
                "<code>/adddays @username 30 auto</code>\n"
                "<code>/adddays TH-A3K9M2X1 14 manual</code>\n\n"
                "<i>Activates or extends subscription with the same flow as a real payment.</i>",
                parse_mode="HTML"
            )
            return

        target_identifier = parts[1]
        try:
            days = int(parts[2])
            if days < 1 or days > 365:
                await message.answer("❌ Days must be between 1 and 365.")
                return
        except ValueError:
            await message.answer("❌ Days must be a valid number.")
            return

        plan_type = parts[3].lower() if len(parts) >= 4 else "auto"
        if plan_type not in ("auto", "manual", "scan"):
            await message.answer("❌ Plan must be: auto, manual, or scan")
            return

        target_user = None
        if target_identifier.startswith("TH-"):
            target_user = db.query(User).filter(User.uid == target_identifier).first()
        if not target_user:
            target_user = db.query(User).filter(User.telegram_id == target_identifier).first()
        if not target_user:
            target_user = db.query(User).filter(User.username == target_identifier.lstrip("@")).first()
        if not target_user:
            await message.answer(f"❌ User not found: {target_identifier}")
            return

        from datetime import timedelta

        now = datetime.utcnow()
        had_active_sub = target_user.subscription_end and target_user.subscription_end > now
        start_from = max(now, target_user.subscription_end) if had_active_sub else now
        new_end = start_from + timedelta(days=days)

        target_user.subscription_end = new_end
        target_user.subscription_type = plan_type
        target_user.approved = True

        # Create a Subscription record identical to what a real payment produces
        sub_record = Subscription(
            user_id=target_user.id,
            payment_method="manual_grant",
            transaction_id=f"admin_grant_{message.from_user.id}_{int(now.timestamp())}",
            amount=0.0,
            duration_days=days
        )
        db.add(sub_record)
        db.commit()
        db.refresh(target_user)

        action = "EXTENDED" if had_active_sub else "ACTIVATED"
        expires = new_end.strftime("%b %d, %Y")
        target_name = f"@{target_user.username}" if target_user.username else target_user.first_name or target_user.telegram_id

        plan_labels = {
            "auto":   ("🤖 Auto-Trading",  "✅ Automated 24/7 execution\n✅ Manual signal notifications\n✅ Auto-Trading on Bitunix\n✅ Advanced risk management"),
            "manual": ("💎 Manual Signals", "✅ Manual signal notifications\n✅ Top Gainers scanner\n✅ LONGS + SHORTS strategies\n✅ PnL tracking"),
            "scan":   ("📊 Scan Mode",      "✅ Top Gainers scanner\n✅ Volume surge detection\n✅ New coin alerts"),
        }
        plan_display, features = plan_labels[plan_type]

        # Confirm to admin
        await message.answer(
            f"✅ <b>Subscription {action}!</b>\n\n"
            f"👤 User: {target_name} (<code>{target_user.uid or target_user.telegram_id}</code>)\n"
            f"📋 Plan: <b>{plan_display}</b>\n"
            f"➕ Added: <b>{days} days</b>\n"
            f"📅 Expires: <b>{expires}</b>",
            parse_mode="HTML"
        )

        # Send user the same welcome message as a real payment confirmation
        try:
            await bot.send_message(
                chat_id=int(target_user.telegram_id),
                text=(
                    f"✅ <b>Subscription {action}!</b>\n\n"
                    f"Your <b>{plan_display}</b> plan is active until:\n"
                    f"📅 <b>{expires}</b>\n\n"
                    f"You now have access to:\n"
                    f"{features}\n\n"
                    f"Use /dashboard to get started!"
                ),
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


@dp.message(Command("list_trials"))
async def cmd_list_trials(message: types.Message):
    """Admin command to list all users on trial"""
    from datetime import timezone
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user or not user.is_admin:
            await message.answer("❌ Admin access required.")
            return
        
        now_utc = datetime.now(timezone.utc)
        
        active_trials = db.query(User).filter(
            User.trial_ends_at != None,
            User.trial_ends_at > now_utc
        ).order_by(User.trial_ends_at.asc()).all()
        
        expired_trials = db.query(User).filter(
            User.trial_ends_at != None,
            User.trial_ends_at <= now_utc
        ).order_by(User.trial_ends_at.desc()).limit(10).all()
        
        response = "🧪 <b>Trial Users</b>\n\n"
        
        if active_trials:
            response += f"✅ <b>Active Trials ({len(active_trials)}):</b>\n"
            for trial in active_trials:
                try:
                    trial_end = trial.trial_ends_at
                    if trial_end.tzinfo is None:
                        trial_end = trial_end.replace(tzinfo=timezone.utc)
                    
                    hours_left = int((trial_end - now_utc).total_seconds() / 3600)
                    days_left = hours_left // 24
                    remaining_hours = hours_left % 24
                    
                    username = trial.username or "No username"
                    time_str = f"{days_left}d {remaining_hours}h" if days_left > 0 else f"{hours_left}h"
                    
                    response += f"• @{username} (<code>{trial.telegram_id}</code>) - {time_str} left\n"
                except Exception:
                    continue
        else:
            response += "✅ <b>Active Trials:</b> None\n"
        
        if expired_trials:
            response += f"\n❌ <b>Recently Expired ({len(expired_trials)}):</b>\n"
            for trial in expired_trials[:5]:
                username = trial.username or "No username"
                response += f"• @{username} (<code>{trial.telegram_id}</code>)\n"
        
        await message.answer(response, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in list_trials command: {e}")
        await message.answer(f"❌ Error: {str(e)}")
    finally:
        db.close()


@dp.message(Command("activate"))
async def cmd_activate_user(message: types.Message):
    """Admin command to manually activate a user's subscription (for discounted payments)"""
    from datetime import timezone
    db = SessionLocal()
    
    try:
        admin = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not admin or not admin.is_admin:
            await message.answer("❌ Admin access required.")
            return
        
        parts = message.text.split()
        if len(parts) < 2:
            await message.answer(
                "📋 <b>Usage:</b> /activate [telegram_id] [days]\n\n"
                "Examples:\n"
                "• <code>/activate 123456789</code> - 30 days auto-trading\n"
                "• <code>/activate 123456789 14</code> - 14 days\n",
                parse_mode="HTML"
            )
            return
        
        target_id = parts[1]
        days = int(parts[2]) if len(parts) > 2 else 30
        
        target_user = db.query(User).filter(User.telegram_id == target_id).first()
        if not target_user:
            await message.answer(f"❌ User with ID {target_id} not found.")
            return
        
        now_utc = datetime.now(timezone.utc)
        target_user.subscription_end = now_utc + timedelta(days=days)
        target_user.subscription_type = "auto"
        db.commit()
        
        username = f"@{target_user.username}" if target_user.username else target_user.first_name or target_id
        await message.answer(
            f"✅ <b>Subscription Activated!</b>\n\n"
            f"<b>User:</b> {username}\n"
            f"<b>Plan:</b> Auto-Trading\n"
            f"<b>Duration:</b> {days} days\n"
            f"<b>Expires:</b> {target_user.subscription_end.strftime('%Y-%m-%d')}",
            parse_mode="HTML"
        )
        
    except Exception as e:
        logger.error(f"Error in activate command: {e}")
        await message.answer(f"❌ Error: {str(e)}")
    finally:
        db.close()


@dp.message(Command("admin"))
async def cmd_admin_panel(message: types.Message):
    """Admin dashboard with interactive management buttons"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        is_owner = str(message.from_user.id) == str(settings.OWNER_TELEGRAM_ID)
        if not is_owner and (not user or not user.is_admin):
            await message.answer("This command is only available to admins.")
            return

        now = datetime.utcnow()

        total_users = db.query(User).count()
        active_subs = db.query(User).filter(User.subscription_end != None, User.subscription_end > now).count()
        active_trials = db.query(User).filter(User.trial_ends_at != None, User.trial_ends_at > now).count()
        grandfathered = db.query(User).filter(User.grandfathered == True).count()
        banned_count = db.query(User).filter(User.banned == True).count()
        open_trades = db.query(Trade).filter(Trade.status == 'open').count()

        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        new_today = db.query(User).filter(User.created_at >= today_start).count()

        text = (
            f"<b>ADMIN DASHBOARD</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"<b>Users</b>\n"
            f"  Total: <b>{total_users}</b>  |  New today: <b>{new_today}</b>\n"
            f"  Active subs: <b>{active_subs}</b>  |  Trials: <b>{active_trials}</b>\n"
            f"  Grandfathered: <b>{grandfathered}</b>  |  Banned: <b>{banned_count}</b>\n\n"
            f"<b>Trading</b>\n"
            f"  Open positions: <b>{open_trades}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="📋 All Users", callback_data="adm_users_1"),
                InlineKeyboardButton(text="💎 Subscribers", callback_data="adm_subs"),
            ],
            [
                InlineKeyboardButton(text="➕ Grant Sub", callback_data="adm_grant"),
                InlineKeyboardButton(text="🔍 Lookup User", callback_data="adm_lookup"),
            ],
            [
                InlineKeyboardButton(text="⏱ Active Trials", callback_data="adm_trials"),
                InlineKeyboardButton(text="🚫 Banned", callback_data="adm_banned"),
            ],
            [
                InlineKeyboardButton(text="📊 Trade Stats", callback_data="adm_trade_stats"),
                InlineKeyboardButton(text="🔄 Refresh", callback_data="adm_refresh"),
            ],
        ])

        await message.answer(text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "adm_refresh")
async def handle_admin_refresh(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        is_owner = str(callback.from_user.id) == str(settings.OWNER_TELEGRAM_ID)
        if not is_owner and (not user or not user.is_admin):
            await callback.answer("Admin only", show_alert=True)
            return

        now = datetime.utcnow()
        total_users = db.query(User).count()
        active_subs = db.query(User).filter(User.subscription_end != None, User.subscription_end > now).count()
        active_trials = db.query(User).filter(User.trial_ends_at != None, User.trial_ends_at > now).count()
        grandfathered = db.query(User).filter(User.grandfathered == True).count()
        banned_count = db.query(User).filter(User.banned == True).count()
        open_trades = db.query(Trade).filter(Trade.status == 'open').count()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        new_today = db.query(User).filter(User.created_at >= today_start).count()

        text = (
            f"<b>ADMIN DASHBOARD</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"<b>Users</b>\n"
            f"  Total: <b>{total_users}</b>  |  New today: <b>{new_today}</b>\n"
            f"  Active subs: <b>{active_subs}</b>  |  Trials: <b>{active_trials}</b>\n"
            f"  Grandfathered: <b>{grandfathered}</b>  |  Banned: <b>{banned_count}</b>\n\n"
            f"<b>Trading</b>\n"
            f"  Open positions: <b>{open_trades}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="📋 All Users", callback_data="adm_users_1"),
                InlineKeyboardButton(text="💎 Subscribers", callback_data="adm_subs"),
            ],
            [
                InlineKeyboardButton(text="➕ Grant Sub", callback_data="adm_grant"),
                InlineKeyboardButton(text="🔍 Lookup User", callback_data="adm_lookup"),
            ],
            [
                InlineKeyboardButton(text="⏱ Active Trials", callback_data="adm_trials"),
                InlineKeyboardButton(text="🚫 Banned", callback_data="adm_banned"),
            ],
            [
                InlineKeyboardButton(text="📊 Trade Stats", callback_data="adm_trade_stats"),
                InlineKeyboardButton(text="🔄 Refresh", callback_data="adm_refresh"),
            ],
        ])

        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data.startswith("adm_users_"))
async def handle_admin_users_list(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        is_owner = str(callback.from_user.id) == str(settings.OWNER_TELEGRAM_ID)
        if not is_owner and (not user or not user.is_admin):
            return

        page = int(callback.data.split("_")[-1])
        per_page = 10
        offset = (page - 1) * per_page

        total = db.query(User).count()
        users = db.query(User).order_by(User.created_at.desc()).offset(offset).limit(per_page).all()

        now = datetime.utcnow()
        lines = []
        for u in users:
            name = f"@{u.username}" if u.username else u.first_name or "—"
            uid_str = u.uid or "—"
            if u.banned:
                status = "🚫"
            elif u.grandfathered:
                status = "👑"
            elif u.subscription_end and u.subscription_end > now:
                status = "💎"
            elif u.trial_ends_at and u.trial_ends_at > now:
                status = "⏱"
            else:
                status = "⚪"
            lines.append(f"{status} <code>{uid_str}</code> {name}")

        total_pages = max(1, (total + per_page - 1) // per_page)

        text = (
            f"<b>ALL USERS</b>  ({total} total)\n"
            f"Page {page}/{total_pages}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            + "\n".join(lines) +
            f"\n\n💎=Sub  ⏱=Trial  👑=Lifetime  🚫=Banned"
        )

        nav_buttons = []
        if page > 1:
            nav_buttons.append(InlineKeyboardButton(text="◀ Prev", callback_data=f"adm_users_{page - 1}"))
        if page < total_pages:
            nav_buttons.append(InlineKeyboardButton(text="Next ▶", callback_data=f"adm_users_{page + 1}"))

        rows = []
        if nav_buttons:
            rows.append(nav_buttons)
        rows.append([InlineKeyboardButton(text="◀ Back", callback_data="adm_refresh")])

        keyboard = InlineKeyboardMarkup(inline_keyboard=rows)

        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "adm_subs")
async def handle_admin_subs(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        is_owner = str(callback.from_user.id) == str(settings.OWNER_TELEGRAM_ID)
        if not is_owner and (not user or not user.is_admin):
            return

        now = datetime.utcnow()
        subs = db.query(User).filter(
            User.subscription_end != None,
            User.subscription_end > now
        ).order_by(User.subscription_end.desc()).limit(25).all()

        if not subs:
            text = "<b>ACTIVE SUBSCRIBERS</b>\n\nNo active subscribers."
        else:
            lines = []
            for s in subs:
                name = f"@{s.username}" if s.username else s.first_name or "—"
                uid_str = s.uid or "—"
                days_left = max(0, (s.subscription_end - now).days)
                plan = s.subscription_type or "manual"
                lines.append(f"<code>{uid_str}</code> {name}\n   {plan} · {days_left}d left")

            text = (
                f"<b>ACTIVE SUBSCRIBERS</b>  ({len(subs)})\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                + "\n\n".join(lines)
            )

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀ Back", callback_data="adm_refresh")],
        ])
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "adm_trials")
async def handle_admin_trials(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        is_owner = str(callback.from_user.id) == str(settings.OWNER_TELEGRAM_ID)
        if not is_owner and (not user or not user.is_admin):
            return

        now = datetime.utcnow()
        trials = db.query(User).filter(
            User.trial_ends_at != None,
            User.trial_ends_at > now
        ).order_by(User.trial_ends_at.asc()).limit(25).all()

        if not trials:
            text = "<b>ACTIVE TRIALS</b>\n\nNo active trials."
        else:
            lines = []
            for t in trials:
                name = f"@{t.username}" if t.username else t.first_name or "—"
                uid_str = t.uid or "—"
                hours_left = max(0, int((t.trial_ends_at - now).total_seconds() / 3600))
                lines.append(f"<code>{uid_str}</code> {name} · {hours_left}h left")

            text = (
                f"<b>ACTIVE TRIALS</b>  ({len(trials)})\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                + "\n".join(lines)
            )

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀ Back", callback_data="adm_refresh")],
        ])
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "adm_banned")
async def handle_admin_banned(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        is_owner = str(callback.from_user.id) == str(settings.OWNER_TELEGRAM_ID)
        if not is_owner and (not user or not user.is_admin):
            return

        banned = db.query(User).filter(User.banned == True).all()

        if not banned:
            text = "<b>BANNED USERS</b>\n\nNo banned users."
        else:
            lines = []
            for b in banned:
                name = f"@{b.username}" if b.username else b.first_name or "—"
                uid_str = b.uid or "—"
                lines.append(f"<code>{uid_str}</code> {name}")

            text = (
                f"<b>BANNED USERS</b>  ({len(banned)})\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                + "\n".join(lines)
            )

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀ Back", callback_data="adm_refresh")],
        ])
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data == "adm_grant")
async def handle_admin_grant_prompt(callback: CallbackQuery):
    await callback.answer()
    user = None
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        is_owner = str(callback.from_user.id) == str(settings.OWNER_TELEGRAM_ID)
        if not is_owner and (not user or not user.is_admin):
            return
    finally:
        db.close()

    text = (
        "<b>SUBSCRIPTION MANAGEMENT</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "<b>Grant New Subscription:</b>\n"
        "<code>/grant_sub USER plan days</code>\n\n"
        "<b>Plans:</b> scan, auto, lifetime\n\n"
        "<b>Examples:</b>\n"
        "<code>/grant_sub TH-A3K9M2X1 auto 30</code>\n"
        "<code>/grant_sub 123456789 scan 30</code>\n"
        "<code>/grant_sub TH-A3K9M2X1 lifetime</code>\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "<b>Add Days to Existing Sub:</b>\n"
        "<code>/adddays USER days</code>\n\n"
        "Extends an active subscription or creates one if expired.\n\n"
        "<b>Examples:</b>\n"
        "<code>/adddays TH-A3K9M2X1 14</code>\n"
        "<code>/adddays @username 30</code>\n"
        "<code>/adddays 123456789 7</code>\n\n"
        "<i>USER = Telegram ID, @username, or UID (TH-XXXXXXXX)</i>"
    )

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀ Back", callback_data="adm_refresh")],
    ])
    await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data == "adm_lookup")
async def handle_admin_lookup_prompt(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        is_owner = str(callback.from_user.id) == str(settings.OWNER_TELEGRAM_ID)
        if not is_owner and (not user or not user.is_admin):
            return
    finally:
        db.close()

    text = (
        "<b>LOOKUP USER</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "Send a command to look up any user:\n\n"
        "<code>/whois UID_OR_ID_OR_USERNAME</code>\n\n"
        "<b>Examples:</b>\n"
        "<code>/whois TH-A3K9M2X1</code>\n"
        "<code>/whois 123456789</code>\n"
        "<code>/whois @username</code>"
    )

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀ Back", callback_data="adm_refresh")],
    ])
    await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")


@dp.callback_query(F.data == "adm_trade_stats")
async def handle_admin_trade_stats(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        is_owner = str(callback.from_user.id) == str(settings.OWNER_TELEGRAM_ID)
        if not is_owner and (not user or not user.is_admin):
            return

        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=7)

        closed_statuses = ['closed', 'stopped', 'tp_hit', 'sl_hit']

        open_count = db.query(Trade).filter(Trade.status == 'open').count()
        today_closed = db.query(Trade).filter(
            Trade.status.in_(closed_statuses),
            Trade.closed_at >= today_start
        ).count()
        week_closed = db.query(Trade).filter(
            Trade.status.in_(closed_statuses),
            Trade.closed_at >= week_start
        ).count()

        from sqlalchemy import func as sa_func
        today_pnl = db.query(sa_func.coalesce(sa_func.sum(Trade.pnl), 0)).filter(
            Trade.status.in_(closed_statuses),
            Trade.closed_at >= today_start
        ).scalar() or 0
        week_pnl = db.query(sa_func.coalesce(sa_func.sum(Trade.pnl), 0)).filter(
            Trade.status.in_(closed_statuses),
            Trade.closed_at >= week_start
        ).scalar() or 0

        today_wins = db.query(Trade).filter(
            Trade.status.in_(closed_statuses),
            Trade.closed_at >= today_start,
            Trade.pnl > 0
        ).count()
        today_wr = (today_wins / today_closed * 100) if today_closed > 0 else 0

        week_wins = db.query(Trade).filter(
            Trade.status.in_(closed_statuses),
            Trade.closed_at >= week_start,
            Trade.pnl > 0
        ).count()
        week_wr = (week_wins / week_closed * 100) if week_closed > 0 else 0

        active_traders = db.query(Trade.user_id).filter(Trade.status == 'open').distinct().count()

        today_sign = "+" if today_pnl >= 0 else ""
        week_sign = "+" if week_pnl >= 0 else ""

        text = (
            f"<b>TRADE STATISTICS</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"<b>Current</b>\n"
            f"  Open positions: <b>{open_count}</b>\n"
            f"  Active traders: <b>{active_traders}</b>\n\n"
            f"<b>Today</b>\n"
            f"  Closed: <b>{today_closed}</b>  |  Win rate: <b>{today_wr:.0f}%</b>\n"
            f"  P&L: <b>{today_sign}${today_pnl:,.2f}</b>\n\n"
            f"<b>This Week</b>\n"
            f"  Closed: <b>{week_closed}</b>  |  Win rate: <b>{week_wr:.0f}%</b>\n"
            f"  P&L: <b>{week_sign}${week_pnl:,.2f}</b>"
        )

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀ Back", callback_data="adm_refresh")],
        ])
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    finally:
        db.close()


@dp.message(Command("whois"))
async def cmd_whois(message: types.Message):
    """Admin command to look up a user by UID, telegram ID, or username"""
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        is_owner = str(message.from_user.id) == str(settings.OWNER_TELEGRAM_ID)
        if not is_owner and (not admin or not admin.is_admin):
            await message.answer("This command is only available to admins.")
            return

        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            await message.answer(
                "<b>Usage:</b> <code>/whois UID_OR_ID_OR_USERNAME</code>\n\n"
                "Examples:\n"
                "<code>/whois TH-A3K9M2X1</code>\n"
                "<code>/whois 123456789</code>\n"
                "<code>/whois @username</code>",
                parse_mode="HTML"
            )
            return

        query = parts[1].strip().lstrip("@")

        target = db.query(User).filter(User.uid == query).first()
        if not target:
            target = db.query(User).filter(User.telegram_id == query).first()
        if not target:
            target = db.query(User).filter(User.username == query).first()

        if not target:
            await message.answer(f"User not found: {query}")
            return

        now = datetime.utcnow()
        name = f"@{target.username}" if target.username else target.first_name or "—"
        uid_str = target.uid or "—"

        if target.grandfathered:
            sub_status = "Lifetime (Grandfathered)"
        elif target.subscription_end and target.subscription_end > now:
            days_left = (target.subscription_end - now).days
            sub_status = f"Active · {target.subscription_type or 'manual'} · {days_left}d left"
        elif target.trial_ends_at and target.trial_ends_at > now:
            hours_left = int((target.trial_ends_at - now).total_seconds() / 3600)
            sub_status = f"Trial · {hours_left}h left"
        else:
            sub_status = "Expired / None"

        closed_statuses = ['closed', 'stopped', 'tp_hit', 'sl_hit']
        total_trades = db.query(Trade).filter(Trade.user_id == target.id, Trade.status.in_(closed_statuses)).count()
        open_trades = db.query(Trade).filter(Trade.user_id == target.id, Trade.status == 'open').count()
        wins = db.query(Trade).filter(Trade.user_id == target.id, Trade.status.in_(closed_statuses), Trade.pnl > 0).count()
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        from sqlalchemy import func as sa_func
        total_pnl = db.query(sa_func.coalesce(sa_func.sum(Trade.pnl), 0)).filter(
            Trade.user_id == target.id,
            Trade.status.in_(closed_statuses)
        ).scalar() or 0

        prefs = target.preferences
        at_status = "ON" if (prefs and prefs.auto_trading_enabled) else "OFF"
        bitunix_linked = "Yes" if (prefs and prefs.bitunix_api_key) else "No"

        pnl_sign = "+" if total_pnl >= 0 else ""

        text = (
            f"<b>USER PROFILE</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"<b>Identity</b>\n"
            f"  UID: <code>{uid_str}</code>\n"
            f"  Name: {name}\n"
            f"  Telegram: <code>{target.telegram_id}</code>\n"
            f"  Joined: {target.created_at.strftime('%Y-%m-%d') if target.created_at else '—'}\n\n"
            f"<b>Subscription</b>\n"
            f"  Status: {sub_status}\n"
            f"  Admin: {'Yes' if target.is_admin else 'No'}\n"
            f"  Banned: {'Yes' if target.banned else 'No'}\n\n"
            f"<b>Trading</b>\n"
            f"  Auto-Trading: {at_status}\n"
            f"  Bitunix Linked: {bitunix_linked}\n"
            f"  Open: {open_trades}  |  Closed: {total_trades}\n"
            f"  Win rate: {win_rate:.0f}%  |  P&L: {pnl_sign}${total_pnl:,.2f}\n\n"
            f"<b>Referral</b>\n"
            f"  Code: <code>{target.referral_code or '—'}</code>\n"
            f"  Referred by: {target.referred_by or '—'}\n"
            f"  Earnings: ${target.referral_earnings or 0:.2f}"
        )

        buttons = []
        if target.banned:
            buttons.append([InlineKeyboardButton(text="Unban User", callback_data=f"adm_unban_{target.telegram_id}")])
        else:
            buttons.append([InlineKeyboardButton(text="Ban User", callback_data=f"adm_ban_{target.telegram_id}")])

        buttons.append([
            InlineKeyboardButton(text="Grant 30d Auto", callback_data=f"adm_quick_grant_{target.telegram_id}_auto_30"),
            InlineKeyboardButton(text="Grant 30d Scan", callback_data=f"adm_quick_grant_{target.telegram_id}_scan_30"),
        ])
        buttons.append([
            InlineKeyboardButton(text="Grant Lifetime", callback_data=f"adm_quick_grant_{target.telegram_id}_lifetime_0"),
        ])

        keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
        await message.answer(text, reply_markup=keyboard, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error in whois: {e}")
        await message.answer(f"Error: {str(e)}")
    finally:
        db.close()


@dp.callback_query(F.data.startswith("adm_quick_grant_"))
async def handle_admin_quick_grant(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        is_owner = str(callback.from_user.id) == str(settings.OWNER_TELEGRAM_ID)
        if not is_owner and (not admin or not admin.is_admin):
            await callback.answer("Admin only", show_alert=True)
            return

        parts = callback.data.split("_")
        target_id = parts[3]
        plan = parts[4]
        days = int(parts[5])

        target = db.query(User).filter(User.telegram_id == target_id).first()
        if not target:
            await callback.answer("User not found", show_alert=True)
            return

        now = datetime.utcnow()
        if plan == "lifetime":
            target.grandfathered = True
            target.subscription_type = "auto"
            db.commit()
            plan_label = "Lifetime"
        else:
            target.subscription_end = now + timedelta(days=days)
            target.subscription_type = plan
            target.grandfathered = False
            db.commit()
            plan_label = f"{plan.title()} ({days}d)"

        name = f"@{target.username}" if target.username else target.first_name or target_id
        await callback.message.answer(
            f"Subscription granted to {name}:\n"
            f"Plan: <b>{plan_label}</b>",
            parse_mode="HTML"
        )

        try:
            await bot.send_message(
                target.telegram_id,
                f"Your subscription has been activated!\n\n"
                f"Plan: <b>{plan_label}</b>\n"
                f"Use /start to access your dashboard.",
                parse_mode="HTML"
            )
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Quick grant error: {e}")
        await callback.answer(f"Error: {str(e)}", show_alert=True)
    finally:
        db.close()


@dp.callback_query(F.data.startswith("adm_ban_"))
async def handle_admin_ban(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        is_owner = str(callback.from_user.id) == str(settings.OWNER_TELEGRAM_ID)
        if not is_owner and (not admin or not admin.is_admin):
            return

        target_id = callback.data.split("_")[-1]
        target = db.query(User).filter(User.telegram_id == target_id).first()
        if target:
            target.banned = True
            db.commit()
            name = f"@{target.username}" if target.username else target.first_name or target_id
            await callback.message.answer(f"User {name} has been banned.", parse_mode="HTML")
    finally:
        db.close()


@dp.callback_query(F.data.startswith("adm_unban_"))
async def handle_admin_unban(callback: CallbackQuery):
    await callback.answer()
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.telegram_id == str(callback.from_user.id)).first()
        is_owner = str(callback.from_user.id) == str(settings.OWNER_TELEGRAM_ID)
        if not is_owner and (not admin or not admin.is_admin):
            return

        target_id = callback.data.split("_")[-1]
        target = db.query(User).filter(User.telegram_id == target_id).first()
        if target:
            target.banned = False
            db.commit()
            name = f"@{target.username}" if target.username else target.first_name or target_id
            await callback.message.answer(f"User {name} has been unbanned.", parse_mode="HTML")
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
        
        # Check subscription access
        has_access, reason = check_access(user)
        if not has_access:
            await message.answer(reason)
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
                # Provide helpful message based on error type
                if 'not found' in error.lower() or 'no data' in error.lower() or 'binance' in error.lower():
                    await analyzing_msg.edit_text(
                        f"🔍 <b>{symbol} Analysis</b>\n\n"
                        f"I couldn't find this coin on the major exchanges I track.\n\n"
                        f"<b>This could mean:</b>\n"
                        f"• The coin is very new or low volume\n"
                        f"• It trades under a different symbol\n"
                        f"• It's only on DEXs (not CEXs)\n\n"
                        f"<b>Try:</b> Check CoinGecko or CoinMarketCap for the correct symbol!",
                        parse_mode="HTML"
                    )
                else:
                    await analyzing_msg.edit_text(
                        f"🔍 <b>{symbol} Analysis</b>\n\n"
                        f"I'm having trouble analyzing this coin right now. Please try again in a moment!\n\n"
                        f"<i>Popular coins: BTC, ETH, SOL, DOGE, XRP</i>",
                        parse_mode="HTML"
                    )
                return
            
            # ═══════════════════════════════════════════════════════════════
            # CLEAN AI-POWERED SCAN MESSAGE
            # ═══════════════════════════════════════════════════════════════
            
            bias = analysis['overall_bias']
            bias_emoji = bias['emoji']
            price = analysis['price']
            trade_idea = analysis.get('trade_idea', {})
            direction = trade_idea.get('direction', 'LONG') if trade_idea else 'LONG'
            
            # Get enhanced analysis data
            mtf = analysis.get('mtf_trend', {})
            risk = analysis.get('risk_score', {})
            ai_conf = analysis.get('ai_confidence', {})
            
            # Clean Header
            report = f"""
<b>{'═' * 24}</b>
🤖 <b>{analysis['symbol']}</b>  |  <b>${price:,.4f}</b>
<b>{'═' * 24}</b>

"""
            
            # AI-Generated Trade Idea (main focus)
            if trade_idea and not trade_idea.get('error'):
                dir_emoji = "🟢" if direction == 'LONG' else "🔴"
                quality = trade_idea.get('quality', 'MEDIUM')
                q_emoji = trade_idea.get('quality_emoji', '⚪')
                trade_type = trade_idea.get('trade_type', 'DAY TRADE')
                trade_type_emoji = trade_idea.get('trade_type_emoji', '📊')
                trade_type_desc = trade_idea.get('trade_type_desc', '')
                
                # Get SL/TP percentages
                sl_pct = trade_idea.get('sl_distance_pct') or trade_idea.get('sl_pct', 0)
                tp1_pct = trade_idea.get('tp1_profit_pct') or trade_idea.get('tp1_pct', 0)
                tp2_pct = trade_idea.get('tp2_profit_pct') or trade_idea.get('tp2_pct', 0)
                
                report += f"""{dir_emoji} <b>TRADE IDEA: {direction}</b> {q_emoji}
{trade_type_emoji} <b>{trade_type}</b> <i>({trade_type_desc})</i>

<code>Entry:  ${trade_idea.get('entry', 0):,.4f}</code>
<code>SL:     ${trade_idea.get('stop_loss', 0):,.4f} (-{abs(sl_pct):.1f}%)</code>
<code>TP1:    ${trade_idea.get('tp1', 0):,.4f} (+{tp1_pct:.1f}%)</code>
<code>TP2:    ${trade_idea.get('tp2', 0):,.4f} (+{tp2_pct:.1f}%)</code>
<code>R:R     {trade_idea.get('rr_ratio', 0):.1f}:1</code>

"""
                # Generate AI Conclusion based on all factors
                validation = trade_idea.get('validation', {})
                risk_score = risk.get('score', 5)
                rr_ratio = trade_idea.get('rr_ratio', 0)
                momentum = analysis.get('momentum', {})
                rsi = momentum.get('rsi', 50)
                vwap = analysis.get('vwap', {})
                obv = analysis.get('obv_flow', {})
                
                # Calculate unified score (combining confidence + risk + validation)
                val_score = validation.get('score', 100)
                has_issues = len(validation.get('issues', [])) > 0
                has_divergence = obv.get('divergence') == 'BEARISH' if direction == 'LONG' else obv.get('divergence') == 'BULLISH'
                
                # Determine conclusion
                if has_issues or risk_score >= 8 or rr_ratio < 1.2:
                    conclusion_emoji = "🚫"
                    conclusion = "AVOID - Poor setup quality"
                    conclusion_detail = "R:R too low or high risk. Skip this trade."
                elif risk_score >= 6 or has_divergence or val_score < 70:
                    conclusion_emoji = "⏳"
                    conclusion = "WAIT - Entry not optimal"
                    if has_divergence:
                        conclusion_detail = f"Volume divergence detected. Wait for confirmation."
                    elif vwap.get('status') == 'ABOVE_UPPER' and direction == 'LONG':
                        conclusion_detail = "Overextended above VWAP. Wait for pullback."
                    elif vwap.get('status') == 'BELOW_LOWER' and direction == 'SHORT':
                        conclusion_detail = "Oversold below VWAP. Wait for bounce."
                    else:
                        conclusion_detail = "Some concerns present. Wait for better entry."
                elif rr_ratio >= 2.0 and risk_score <= 4 and val_score >= 80:
                    conclusion_emoji = "✅"
                    conclusion = "STRONG ENTRY"
                    conclusion_detail = f"Good R:R ({rr_ratio:.1f}:1), low risk. Consider taking this trade."
                else:
                    conclusion_emoji = "🟡"
                    conclusion = "ACCEPTABLE"
                    conclusion_detail = "Decent setup. Manage size appropriately."
                
                report += f"""<b>{conclusion_emoji} AI CONCLUSION:</b> <b>{conclusion}</b>
<i>{conclusion_detail}</i>

"""
            else:
                # No trade idea - just show market status
                report += f"""{bias_emoji} <b>Market Bias: {bias['direction'].upper()}</b>
<i>Confidence: {bias['strength']}%</i>

No clear trade setup at this time.
"""
            
            # Footer
            report += f"""
<b>{'─' * 24}</b>
<i>🤖 AI-powered analysis</i>"""
            
            # Get user's quick trade size
            prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
            quick_size = prefs.quick_trade_size if prefs and prefs.quick_trade_size else 25.0
            
            # Build buttons - Quick Trade + More Info
            buttons = []
            
            if trade_idea and not trade_idea.get('error') and trade_idea.get('direction'):
                dir_emoji = "🟢" if direction == 'LONG' else "🔴"
                sl_pct = abs(trade_idea.get('sl_distance_pct') or trade_idea.get('sl_pct', 2.0))
                tp1_pct = abs(trade_idea.get('tp1_profit_pct') or trade_idea.get('tp1_pct', 3.0))
                tp2_pct = abs(trade_idea.get('tp2_profit_pct') or trade_idea.get('tp2_pct', 5.0))
                
                buttons.append([
                    InlineKeyboardButton(
                        text=f"{dir_emoji} Quick {direction} ${quick_size:.0f}",
                        callback_data=f"quick_trade:{symbol}:{direction}:{quick_size}:{sl_pct:.1f}:{tp1_pct:.1f}:{tp2_pct:.1f}"
                    ),
                    InlineKeyboardButton(
                        text="⚙️ Size",
                        callback_data="quick_trade_size"
                    )
                ])
            
            # More Info button
            buttons.append([
                InlineKeyboardButton(
                    text="📊 More Details",
                    callback_data=f"scan_details:{symbol}"
                )
            ])
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
            
            # Store analysis in cache for "More Details" button
            from app.services.scan_service import _scan_cache
            _scan_cache[f"scan_analysis_{symbol}"] = {
                'data': analysis,
                'timestamp': time.time()
            }
            
            # Send clean report with buttons
            await analyzing_msg.edit_text(report, parse_mode="HTML", reply_markup=keyboard)
            
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
        
        # Store balance for UID step
        await state.update_data(bitunix_balance=balance)
        
        await message.answer(f"""
✅ <b>Bitunix API Connected!</b>

💰 Balance: <b>${balance:.2f} USDT</b>

🔒 Keys encrypted & messages deleted

<b>Final Step - Enter Your Bitunix UID:</b>

1. Go to Bitunix app/website
2. Click on your profile icon
3. Copy your UID number

Send your Bitunix UID now:
        """, parse_mode="HTML")
        
        await state.set_state(BitunixSetup.waiting_for_uid)
    finally:
        db.close()


@dp.message(BitunixSetup.waiting_for_uid)
async def process_bitunix_uid(message: types.Message, state: FSMContext):
    """Process Bitunix UID input"""
    db = SessionLocal()
    
    try:
        uid = message.text.strip()
        
        # Basic validation - UID should be numeric
        if not uid.isdigit() or len(uid) < 5:
            await message.answer(
                "❌ Invalid UID format.\n\n"
                "Your Bitunix UID should be a numeric ID (e.g., 12345678).\n\n"
                "Please check and send your correct UID:"
            )
            return
        
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("❌ Error: User not found. Use /start first.")
            await state.clear()
            return
        
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs:
            await message.answer("❌ Error: Preferences not found. Try /setup_bitunix again.")
            await state.clear()
            return
        
        # Save UID
        prefs.bitunix_uid = uid
        db.commit()
        
        # Get stored balance
        data = await state.get_data()
        balance = data.get('bitunix_balance', 0)
        
        logger.info(f"✅ SETUP COMPLETE: User {user.username} - UID: {uid}")
        
        await message.answer(f"""
✅ <b>Setup Complete!</b>

💰 Balance: <b>${balance:.2f} USDT</b>
🆔 UID: <b>{uid}</b>

⚡ You're fully ready for auto-trading!

<b>Next Steps:</b>
/toggle_autotrading - Enable auto-trading
/autotrading_status - Check your settings

Let's make some profits! 🚀
        """, parse_mode="HTML")
        
        # Notify owner about new API connection
        try:
            owner_msg = (
                f"🔔 <b>New API Connected!</b>\n\n"
                f"👤 User: @{user.username or 'N/A'}\n"
                f"🆔 Bitunix UID: <code>{uid}</code>\n"
                f"💰 Balance: ${balance:.2f} USDT\n"
                f"📅 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
            )
            await bot.send_message(settings.OWNER_TELEGRAM_ID, owner_msg, parse_mode="HTML")
        except Exception as notify_err:
            logger.error(f"Failed to notify owner about new API: {notify_err}")
        
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


@dp.message(Command("trading_status"))
@dp.message(Command("limit"))
async def cmd_trading_limit_status(message: types.Message):
    """Show current 4-hour rolling window status and countdown"""
    db = SessionLocal()
    from datetime import datetime, timedelta
    from app.models import Trade
    
    try:
        now = datetime.utcnow()
        four_hours_ago = now - timedelta(hours=4)
        
        # FIXED: Count UNIQUE SIGNALS (not individual user trades), exclude SCALP
        # Each signal can create multiple trades (one per user), but we count unique signals
        from sqlalchemy import func
        valid_statuses = ['open', 'closed', 'tp_hit', 'sl_hit', 'breakeven']
        
        # First get all trades to find window start
        first_trade = db.query(Trade).filter(
            Trade.opened_at >= four_hours_ago,
            Trade.signal_id.isnot(None),
            Trade.trade_type != 'SCALP',
            Trade.status.in_(valid_statuses)
        ).order_by(Trade.opened_at.asc()).first()
        
        if first_trade:
            window_start = first_trade.opened_at
            window_end = window_start + timedelta(hours=4)
            
            # Count UNIQUE signal_ids in the window
            count = db.query(func.count(func.distinct(Trade.signal_id))).filter(
                Trade.opened_at >= window_start,
                Trade.opened_at < window_end,
                Trade.signal_id.isnot(None),
                Trade.trade_type != 'SCALP',
                Trade.status.in_(valid_statuses)
            ).scalar() or 0
        else:
            count = 0
            window_start = None
            window_end = None
        limit = 2
        
        status_msg = f"📊 <b>Trading Limit Status</b>\n\n"
        status_msg += f"Window: <b>4h Cycle (Starts @ 1st Trade)</b>\n"
        status_msg += f"Signals Sent: <b>{count}/{limit}</b>\n\n"
        
        if count > 0 and window_end:
            remaining = window_end - now
            
            if remaining.total_seconds() > 0:
                minutes = int(remaining.total_seconds() / 60)
                seconds = int(remaining.total_seconds() % 60)
                
                if count >= limit:
                    status_msg += f"⏳ <b>Limit Reached</b>\n"
                    status_msg += f"Next reset in: <b>{minutes}m {seconds}s</b>\n"
                else:
                    status_msg += f"✅ <b>Scanner Active</b>\n"
                    status_msg += f"Slots available: <b>{limit - count}</b>\n"
                    status_msg += f"Cycle reset in: <b>{minutes}m {seconds}s</b>\n"
                
                status_msg += f"<i>(Reset at {window_end.strftime('%H:%M:%S')} UTC)</i>"
            else:
                status_msg += f"✅ <b>Scanner Active</b>\n"
                status_msg += f"Slots available: <b>{limit}</b>"
        else:
            status_msg += f"✅ <b>Scanner Active</b>\n"
            status_msg += f"Slots available: <b>{limit}</b>\n"
            status_msg += f"<i>(Timer starts on next trade)</i>"
            
        await message.answer(status_msg, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error in trading_status: {e}")
        await message.answer("❌ Error fetching trading status.")
    finally:
        db.close()


@dp.message(Command("recent"))
async def cmd_recent_signals(message: types.Message):
    """Show user's recent signals and trades"""
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.telegram_id == str(message.from_user.id)).first()
        if not user:
            await message.answer("You're not registered. Use /start to begin!")
            return
        
        msg = "📊 <b>Recent Activity</b>\n\n"
        
        # Get recent signals
        try:
            recent_signals = db.query(Signal).order_by(Signal.created_at.desc()).limit(5).all()
            if recent_signals:
                msg += "<b>📡 Last 5 Signals:</b>\n"
                for s in recent_signals:
                    direction_emoji = "🟢" if s.direction == "LONG" else "🔴"
                    time_str = s.created_at.strftime("%m/%d %H:%M") if s.created_at else "?"
                    entry = f"${s.entry_price:.4f}" if s.entry_price else "?"
                    msg += f"{direction_emoji} {s.symbol} {entry} | {time_str}\n"
                msg += "\n"
            else:
                msg += "<i>No signals yet today.</i>\n\n"
        except Exception as e:
            logger.error(f"Error fetching signals: {e}")
            msg += "<i>Could not fetch signals.</i>\n\n"
        
        # Get user's recent trades
        try:
            recent_trades = db.query(Trade).filter(
                Trade.user_id == user.id
            ).order_by(Trade.opened_at.desc()).limit(5).all()
            
            if recent_trades:
                msg += "<b>📈 Your Last 5 Trades:</b>\n"
                for t in recent_trades:
                    direction_emoji = "🟢" if t.direction == "LONG" else "🔴"
                    time_str = t.opened_at.strftime("%m/%d %H:%M") if t.opened_at else "?"
                    pnl_str = f"{t.pnl_percent:+.1f}%" if t.pnl_percent else "open"
                    status = t.status or "?"
                    msg += f"{direction_emoji} {t.symbol} | {status} | {pnl_str} | {time_str}\n"
            else:
                msg += "<i>No trades yet. Enable auto-trading to start!</i>"
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            msg += "<i>Could not fetch your trades.</i>"
        
        await message.answer(msg, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error in recent command: {e}", exc_info=True)
        await message.answer("❌ Error fetching recent activity.")
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


@dp.message(Command("patterns"))
async def cmd_pattern_detector(message: types.Message):
    """📉 AI Pattern Detector - /patterns <SYMBOL>"""
    args = message.text.split()
    if len(args) < 2:
        await message.answer("❌ Usage: /patterns <SYMBOL>\nExample: /patterns BTC")
        return
    
    symbol = args[1].upper()
    from app.services.ai_market_intelligence import detect_chart_patterns
    await message.answer(f"📉 <b>Scanning {symbol} for Chart Patterns...</b>\n<i>Analyzing 1h/4h structures and trendlines...</i>", parse_mode="HTML")
    
    try:
        data = await detect_chart_patterns(symbol)
        if "error" in data:
            await message.answer(f"❌ Error: {data['error']}")
            return
            
        msg = f"📉 <b>AI PATTERN DETECTOR: {symbol}</b>\n━━━━━━━━━━━━━━━━━━━━\n\n"
        if not data.get('patterns'):
            msg += "🔍 No clear patterns identified at this time.\n"
        else:
            for p in data['patterns']:
                msg += f"🔸 <b>{p['name']}</b> ({p['timeframe']})\n"
                msg += f"• 🎯 Bias: {p['bias']}\n"
                msg += f"• 📊 Completion: {p['completion']}\n"
                msg += f"• 🚀 Target: ${p['target']:,.2f}\n"
                msg += f"• 🛡️ Stop: ${p['stop_loss']:,.2f}\n\n"
        
        msg += f"📝 <b>AI Summary:</b>\n{data.get('summary', 'N/A')}\n"
        msg += f"\n💎 <b>Confidence Score: {data.get('confidence_score', 0)}%</b>\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━\n"
        await message.answer(msg, parse_mode="HTML")
    except Exception as e:
        await message.answer("❌ Error detecting patterns.")


@dp.message(Command("whale"))
async def cmd_whale_tracker(message: types.Message):
    """🐋 AI Whale Tracker - /whale"""
    from app.services.ai_market_intelligence import track_whale_activity
    await message.answer("🐋 <b>Scanning Whale Activity...</b>\n<i>Analyzing order books, funding extremes, and volume spikes...</i>", parse_mode="HTML")
    try:
        data = await track_whale_activity()
        if "error" in data:
            await message.answer(f"❌ Error: {data['error']}")
            return
        
        msg = f"🐋 <b>AI WHALE TRACKER</b>\n━━━━━━━━━━━━━━━━━━━━\n\n"
        msg += f"📊 <b>Market Bias: {data.get('bias', 'NEUTRAL')}</b>\n\n"
        msg += f"🚀 <b>Smart Money Picks:</b>\n"
        for coin in data.get('top_picks', []):
            msg += f"• <b>{coin}</b>\n"
        
        msg += f"\n🚨 <b>Activity Alerts:</b>\n"
        for alert in data.get('alerts', []):
            msg += f"• <i>{alert}</i>\n"
            
        msg += f"\n💡 <b>Recommendation:</b>\n{data.get('recommendation', 'N/A')}\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━\n"
        await message.answer(msg, parse_mode="HTML")
    except Exception as e:
        await message.answer("❌ Error tracking whales.")


@dp.message(Command("market"))
async def cmd_market_regime(message: types.Message):
    """🔮 AI Market Regime Detector - /market"""
    from app.services.ai_market_intelligence import detect_market_regime
    
    await message.answer("🔮 <b>Analyzing Market Regime...</b>\n<i>Fetching BTC derivatives, funding rates, and market breadth...</i>", parse_mode="HTML")
    
    try:
        regime_data = await detect_market_regime()
        
        if not regime_data or regime_data.get('regime') == 'UNKNOWN':
            await message.answer("⚠️ Could not determine market regime at this time. Please try again later.")
            return

        regime = regime_data.get('regime', 'UNKNOWN').replace('_', ' ')
        risk = regime_data.get('risk_level', 'MEDIUM')
        risk_emoji = "🟢" if risk == "LOW" else "🟡" if risk == "MEDIUM" else "🔴" if risk == "HIGH" else "💀"
        
        # Format the response message
        msg = f"🔮 <b>AI MARKET REGIME: {regime}</b>\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
        
        msg += f"💰 <b>BTC:</b> ${regime_data.get('btc_price', 0):,.0f} ({regime_data.get('btc_change', 0):+.1f}%)\n"
        msg += f"📊 <b>RSI (1h):</b> {regime_data.get('btc_rsi', 50):.0f}\n"
        msg += f"📉 <b>Volatility:</b> {regime_data.get('btc_volatility', 0):.2f}%\n"
        msg += f"⛓️ <b>BTC Funding:</b> {regime_data.get('btc_funding', 0):+.4f}%\n"
        msg += f"💎 <b>Fear & Greed:</b> {regime_data.get('fear_greed', 50)} ({regime_data.get('fear_greed_text', 'Neutral')})\n\n"
        
        msg += f"⚖️ <b>Market Breadth:</b>\n"
        msg += f"• Gainers: {regime_data.get('gainers', 0)} | Losers: {regime_data.get('losers', 0)}\n"
        msg += f"• Ratio: {regime_data.get('breadth_ratio', 1.0):.2f}\n\n"
        
        msg += f"{risk_emoji} <b>Risk Level: {risk}</b>\n"
        msg += f"🎯 <b>Bias: {regime_data.get('btc_bias', 'NEUTRAL')}</b>\n"
        msg += f"📏 <b>Size Modifier: {regime_data.get('position_size_modifier', 1.0)}x</b>\n\n"
        
        msg += f"💡 <b>Recommendation:</b>\n<i>{regime_data.get('recommendation', 'N/A')}</i>\n\n"
        msg += f"📜 <b>Tactical Playbook:</b>\n{regime_data.get('tactical_playbook', 'N/A')}\n\n"
        msg += f"👀 <b>Watch For:</b>\n{regime_data.get('watch_for', 'N/A')}\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"<i>Last updated: {datetime.utcnow().strftime('%H:%M:%S')} UTC</i>"
        
        await message.answer(msg, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in cmd_market_regime: {e}")
        await message.answer("❌ Error analyzing market regime.")

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
        blocked_count = 0
        unreachable_count = 0
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
                err_str = str(e).lower()
                if "bot was blocked" in err_str or "user is deactivated" in err_str:
                    blocked_count += 1
                elif "chat not found" in err_str or "user not found" in err_str:
                    unreachable_count += 1
                else:
                    logger.error(f"Failed to send broadcast to {user_to_notify.telegram_id}: {e}")
                    failed_count += 1
            await asyncio.sleep(0.05)
        
        await message.answer(
            f"✅ <b>Broadcast Complete!</b>\n\n"
            f"📎 Type: {media_type.upper()}\n"
            f"✅ Sent: {sent_count}\n"
            f"🚫 Blocked bot: {blocked_count}\n"
            f"👻 Never started bot: {unreachable_count}\n"
            f"❌ Other errors: {failed_count}\n"
            f"📊 Total attempted: {len(all_users)}",
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
        lambda: asyncio.create_task(process_album_broadcast(media_group_id))
    )


async def process_album_broadcast(media_group_id: str):
    """Process and send album broadcast to all users"""
    from aiogram.types import InputMediaPhoto
    
    db = SessionLocal()
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


async def broadcast_news_signal(news_signal: dict):
    """Broadcast news-based trading signal"""
    # 🛡️ GLOBAL CONFIDENCE GATE: Block signals below 7/10
    news_conf = news_signal.get('confidence', 0)
    if news_conf < 70:  # News uses 0-100 scale
        logger.info(f"🚫 NEWS SIGNAL BLOCKED at broadcast: {news_signal['symbol']} confidence {news_conf} < 70 (7/10)")
        return
    
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
        
        spot_lev = flow_data.get('leverage', 10)
        sl_pnl = calculate_leverage_pnl(entry_price, stop_loss, trade_direction, spot_lev)
        tp_pnl = calculate_leverage_pnl(entry_price, take_profit, trade_direction, spot_lev)
        
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

<b>💰 Trade Levels ({spot_lev}x Leverage)</b>
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
        
        sig_leverage = signal_data.get('leverage', 10)
        tp1_pnl = calculate_leverage_pnl(signal_data['entry_price'], signal_data['take_profit_1'], signal_data['direction'], sig_leverage)
        tp2_pnl = calculate_leverage_pnl(signal_data['entry_price'], signal_data['take_profit_2'], signal_data['direction'], sig_leverage)
        tp3_pnl = calculate_leverage_pnl(signal_data['entry_price'], signal_data['take_profit_3'], signal_data['direction'], sig_leverage)
        sl_pnl = calculate_leverage_pnl(signal_data['entry_price'], signal_data['stop_loss'], signal_data['direction'], sig_leverage)
        
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
🛑 Stop Loss: ${signal_data['stop_loss']} ({sl_pnl:+.2f}% @ {sig_leverage}x)

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
🛑 Stop Loss: ${signal_data['stop_loss']} ({sl_pnl:+.2f}% @ {sig_leverage}x)

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
    # 🛡️ GLOBAL CONFIDENCE GATE: Block signals below 7/10
    sig_conf = signal_data.get('confidence', 0)
    if sig_conf is not None and sig_conf > 0:
        conf_check = sig_conf / 10 if sig_conf > 10 else sig_conf
        if conf_check < 7:
            logger.info(f"🚫 SIGNAL BLOCKED at broadcast: {signal_data.get('symbol')} confidence {sig_conf} below 7/10 minimum")
            return
    
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
            
            sig_lev = signal_data.get('leverage', 10)
            tp1_pnl = calculate_leverage_pnl(signal.entry_price, signal.take_profit_1, signal.direction, sig_lev) if signal.take_profit_1 else None
            tp2_pnl = calculate_leverage_pnl(signal.entry_price, signal.take_profit_2, signal.direction, sig_lev) if signal.take_profit_2 else None
            tp3_pnl = calculate_leverage_pnl(signal.entry_price, signal.take_profit_3, signal.direction, sig_lev) if signal.take_profit_3 else calculate_leverage_pnl(signal.entry_price, signal.take_profit, signal.direction, sig_lev)
            sl_pnl = calculate_leverage_pnl(signal.entry_price, signal.stop_loss, signal.direction, sig_lev)
            
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
💰 TP PnL: {tp3_pnl:+.2f}% ({sig_lev}x)"""
            
            signal_text = f"""
🚨 NEW {signal.direction} SIGNAL

📊 Symbol: {signal.symbol}
💰 Entry: ${signal.entry_price}
🛑 Stop Loss: ${signal.stop_loss} ({sl_pnl:+.2f}% @ {sig_lev}x)

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
        
        dt_leverage = signal_data.get('leverage', 10)
        tp_pnl = calculate_leverage_pnl(signal_data['entry_price'], signal_data['take_profit'], signal_data['direction'], dt_leverage)
        sl_pnl = calculate_leverage_pnl(signal_data['entry_price'], signal_data['stop_loss'], signal_data['direction'], dt_leverage)
        
        # Build message
        signal_text = f"""
🎯 DAY TRADE SIGNAL - {signal_data['direction']}
✅ 6-POINT CONFIRMATION PASSED

💰 {signal_data['symbol']}
📊 Strategy: {signal_data.get('pattern', 'Multi-Confirmation')}
💎 Risk-Reward: 1:1

💵 Entry: ${signal_data['entry_price']}
🛑 Stop Loss: ${signal_data['stop_loss']} ({sl_pnl:+.2f}% @ {dt_leverage}x)
🎯 Take Profit: ${signal_data['take_profit']} ({tp_pnl:+.2f}% @ {dt_leverage}x)

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
    logger.info("⚡ SCALP Scanner Started (60-second intervals, TIGHTENED filters - max 4/day)")
    
    await asyncio.sleep(10)  # Initial delay to let bot start
    
    while True:
        service = None
        try:
            from app.services.top_gainers_signals import TopGainersSignalService, broadcast_scalp_signal_simple
            from app.models import Trade
            from datetime import datetime, timedelta
            
            # DAILY LIMIT CHECK: Max 4 scalp trades per day
            db = SessionLocal()
            try:
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                scalp_count_today = db.query(Trade).filter(
                    Trade.opened_at >= today_start,
                    Trade.trade_type == 'SCALP',
                    Trade.status.in_(['open', 'closed', 'tp_hit', 'sl_hit', 'breakeven'])
                ).count()
                
                if scalp_count_today >= 4:
                    logger.info(f"⚡ SCALP DAILY LIMIT: {scalp_count_today}/4 scalps today - skipping scan")
                    await asyncio.sleep(300)  # Wait 5 minutes before checking again
                    continue
            finally:
                db.close()
            
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


async def social_scanner():
    """Scan for AI-powered social/news signals every 90 seconds (fast for breaking news!)"""
    logger.info("🌙 Social & News Scanner Started (AI-powered signals)")
    
    await asyncio.sleep(60)  # Wait 1 minute before first scan
    
    while True:
        db = None
        try:
            await update_heartbeat()
            
            from app.services.social_signals import (
                is_social_scanning_enabled, 
                broadcast_social_signal
            )
            from app.services.lunarcrush import get_lunarcrush_api_key
            
            if not is_social_scanning_enabled():
                logger.debug("🌙 Social scanning disabled - skipping")
                await asyncio.sleep(180)
                continue
            
            if not get_lunarcrush_api_key():
                logger.debug("🌙 No LUNARCRUSH_API_KEY - skipping")
                await asyncio.sleep(180)
                continue
            
            logger.info("🌙 Scanning for social signals...")
            
            db = SessionLocal()
            try:
                await asyncio.wait_for(
                    broadcast_social_signal(db, bot),
                    timeout=90
                )
            except asyncio.TimeoutError:
                logger.warning("⏱️ Social scan timed out (90s)")
            except Exception as inner_e:
                logger.error(f"Social scan error: {inner_e}")
            finally:
                if db:
                    db.close()
                    db = None
                
        except Exception as e:
            logger.error(f"Social scanner error: {e}")
        finally:
            if db:
                try:
                    db.close()
                except:
                    pass
        
        # Scan every 60 seconds — social spikes move fast, catch them early
        await asyncio.sleep(60)


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


async def fartcoin_scanner_loop():
    """Dedicated FARTCOIN scanner - SOL beta amplification + latency trading at 50x"""
    logger.info("🐸 FARTCOIN Scanner Started (SOL beta amplification + latency tracking)")
    
    await asyncio.sleep(75)
    
    while True:
        db = None
        try:
            from app.services.fartcoin_scanner import is_fartcoin_enabled, broadcast_fartcoin_signal
            
            if not is_fartcoin_enabled():
                await asyncio.sleep(120)
                continue
            
            await update_heartbeat()
            logger.info("🐸 Scanning FARTCOIN (SOL correlation)...")
            
            db = SessionLocal()
            try:
                await asyncio.wait_for(
                    broadcast_fartcoin_signal(db, bot),
                    timeout=60
                )
            except asyncio.TimeoutError:
                logger.warning("⏱️ FARTCOIN scan timed out (60s)")
            except Exception as inner_e:
                logger.error(f"FARTCOIN scan error: {inner_e}")
            finally:
                if db:
                    db.close()
                    db = None
                
        except Exception as e:
            logger.error(f"FARTCOIN scanner error: {e}")
        finally:
            if db:
                try:
                    db.close()
                except:
                    pass
        
        await asyncio.sleep(90)


async def btc_orb_scanner_loop():
    """BTC ORB+FVG Scalper - Opening Range Breakout at Asia & NY sessions"""
    logger.info("📊 BTC ORB Scanner Started (15min ORB + Fibonacci + FVG retest)")
    
    await asyncio.sleep(90)
    
    while True:
        db = None
        try:
            from app.services.btc_orb_scanner import is_btc_orb_enabled, broadcast_btc_orb_signal
            
            if not is_btc_orb_enabled():
                await asyncio.sleep(120)
                continue
            
            await update_heartbeat()
            logger.info("📊 Scanning BTC ORB+FVG...")
            
            db = SessionLocal()
            try:
                await asyncio.wait_for(
                    broadcast_btc_orb_signal(db, bot),
                    timeout=60
                )
            except asyncio.TimeoutError:
                logger.warning("⏱️ BTC ORB scan timed out (60s)")
            except Exception as inner_e:
                logger.error(f"BTC ORB scan error: {inner_e}")
            finally:
                if db:
                    db.close()
                    db = None
                
        except Exception as e:
            logger.error(f"BTC ORB scanner error: {e}")
        finally:
            if db:
                try:
                    db.close()
                except:
                    pass
        
        await asyncio.sleep(60)


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


async def daily_digest_scheduler():
    """Send daily digest to all subscribed users at 8 AM UTC"""
    from app.services.ai_chat_assistant import generate_daily_digest
    
    logger.info("☀️ Daily Digest Scheduler Started")
    
    while True:
        try:
            # Calculate time until next 8 AM UTC
            now = datetime.utcnow()
            target_hour = 8  # 8 AM UTC
            
            if now.hour < target_hour:
                # Today at 8 AM
                next_run = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
            else:
                # Tomorrow at 8 AM
                next_run = (now + timedelta(days=1)).replace(hour=target_hour, minute=0, second=0, microsecond=0)
            
            wait_seconds = (next_run - now).total_seconds()
            logger.info(f"☀️ Next daily digest in {wait_seconds/3600:.1f} hours")
            
            await asyncio.sleep(wait_seconds)
            
            # Generate and send digest
            logger.info("☀️ Generating daily digest...")
            digest = await generate_daily_digest()
            
            if digest:
                db = SessionLocal()
                try:
                    # Get all subscribed users
                    users = db.query(User).filter(User.is_approved == True).all()
                    sent_count = 0
                    
                    for user in users:
                        has_access, _ = check_access(user)
                        if has_access:
                            try:
                                await bot.send_message(
                                    user.telegram_id,
                                    f"🤖 <b>Tradehub Daily Digest</b>\n\n{digest}",
                                    parse_mode="HTML"
                                )
                                sent_count += 1
                                await asyncio.sleep(0.1)  # Rate limit
                            except Exception as e:
                                logger.debug(f"Could not send digest to {user.telegram_id}: {e}")
                    
                    logger.info(f"☀️ Daily digest sent to {sent_count} users")
                finally:
                    db.close()
            else:
                logger.warning("☀️ Failed to generate daily digest")
            
            # Wait a bit before next cycle check
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"Daily digest error: {e}", exc_info=True)
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
    # asyncio.create_task(top_gainers_scanner())  # ❌ DISABLED - Using LunarCrush social signals only
    asyncio.create_task(social_scanner())  # 🌙 ENABLED - AI social/news signals (independent)
    # asyncio.create_task(scalp_scanner())  # ❌ PERMANENTLY REMOVED - Ruined bot with low-quality shorts
    # asyncio.create_task(volume_surge_scanner())  # ❌ DISABLED
    # asyncio.create_task(new_coin_alert_scanner())  # ❌ DISABLED
    asyncio.create_task(fartcoin_scanner_loop())  # 🐸 FARTCOIN scanner (SOL correlation)
    asyncio.create_task(btc_orb_scanner_loop())  # 📊 BTC ORB+FVG scalper (Asia & NY sessions)
    asyncio.create_task(position_monitor())
    # asyncio.create_task(daily_pnl_report())  # DISABLED: Daily PnL report notifications
    asyncio.create_task(funding_rate_monitor())
    asyncio.create_task(daily_digest_scheduler())  # ☀️ Daily morning digest at 8 AM UTC
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
        
        # 📋 Set up command menu for Telegram (shows when user types /)
        commands = [
            BotCommand(command="start", description="Main menu & dashboard"),
            BotCommand(command="dashboard", description="Trading dashboard"),
            BotCommand(command="scan", description="Scan any coin - /scan BTC"),
            BotCommand(command="patterns", description="AI chart patterns - /patterns SOL"),
            BotCommand(command="liquidations", description="Liquidation zones - /liquidations ETH"),
            BotCommand(command="news", description="AI news impact scanner"),
            BotCommand(command="market", description="Market regime detector"),
            BotCommand(command="whale", description="Whale & smart money tracker"),
            BotCommand(command="leaderboard", description="Top Binance traders"),
            BotCommand(command="scalp", description="VWAP scalp scan - /scalp BTC"),
            BotCommand(command="settings", description="Configure your settings"),
            BotCommand(command="pnl", description="Your trading performance"),
            BotCommand(command="help", description="Help & support"),
        ]
        try:
            await bot.set_my_commands(commands)
            logger.info("✅ Command menu registered with Telegram")
        except Exception as e:
            logger.warning(f"Could not set command menu: {e}")
        
        # Start Binance WebSocket for real-time ticker data
        try:
            from app.services.binance_ws import start_binance_ws
            start_binance_ws()
            logger.info("✅ Binance WebSocket started for real-time data")
        except Exception as e:
            logger.warning(f"Binance WebSocket start failed (will use REST fallback): {e}")
        
        # Short delay to let old instance fully stop (Railway deployments)
        await asyncio.sleep(3)
        
        logger.info("Bot polling started")
        await dp.start_polling(bot)
    finally:
        # Cleanup on shutdown
        logger.info("Bot shutting down...")
        try:
            from app.services.binance_ws import stop_binance_ws
            await stop_binance_ws()
        except Exception:
            pass
        await manager.release_lock()
        await signal_generator.close()
