import ccxt.async_support as ccxt
import logging
from typing import Optional, Dict
from sqlalchemy.orm import Session
from app.models import User, UserPreference, Trade, Signal
from app.database import SessionLocal
from app.utils.encryption import decrypt_api_key
from app.services.analytics import AnalyticsService

logger = logging.getLogger(__name__)


class MEXCTrader:
    """Handles automated trading on MEXC exchange"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.exchange = ccxt.mexc({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {
                'defaultType': 'swap',
            }
        })
        self.markets_loaded = False
    
    async def get_account_balance(self) -> float:
        """Get available USDT balance"""
        try:
            balance = await self.exchange.fetch_balance()
            return balance.get('USDT', {}).get('free', 0.0)
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0
    
    async def calculate_position_size(self, balance: float, position_size_percent: float) -> float:
        """Calculate position size based on account balance and percentage"""
        return (balance * position_size_percent) / 100
    
    async def place_trade(
        self, 
        symbol: str, 
        direction: str, 
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size_usdt: float,
        leverage: int = 10
    ) -> Optional[Dict]:
        """
        Place a leveraged futures trade on MEXC
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT' or 'BTC/USDT:USDT')
            direction: 'LONG' or 'SHORT'
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            position_size_usdt: Position size in USDT
            leverage: Leverage multiplier
        """
        try:
            # Load markets if not already loaded
            if not self.markets_loaded:
                await self.exchange.load_markets()
                self.markets_loaded = True
                # Log swap markets for debugging
                swap_markets = [k for k, v in self.exchange.markets.items() if v.get('type') == 'swap'][:5]
                logger.info(f"MEXC swap markets sample: {swap_markets}")
            
            # MEXC futures - find the correct SWAP/FUTURES market
            # Filter markets to only swap/futures type
            mexc_symbol = None
            
            # Try different symbol formats, but ONLY check swap markets
            possible_symbols = [
                symbol,  # Original format
                f"{symbol}:USDT",  # BTC/USDT -> BTC/USDT:USDT
                symbol.replace('/', '_'),  # BTC/USDT -> BTC_USDT
                symbol.replace('/', ''),  # BTC/USDT -> BTCUSDT
            ]
            
            for test_symbol in possible_symbols:
                if test_symbol in self.exchange.markets:
                    market = self.exchange.markets[test_symbol]
                    # Only use if it's a swap/futures market
                    if market.get('type') == 'swap':
                        mexc_symbol = test_symbol
                        logger.info(f"Found swap market: {mexc_symbol}")
                        break
            
            if not mexc_symbol:
                # Log all swap/futures markets for debugging
                swap_markets = [k for k, v in self.exchange.markets.items() if v.get('type') == 'swap' and symbol.split('/')[0] in k][:10]
                logger.error(f"Swap market for {symbol} not found. Available: {swap_markets}")
                # Default to :USDT format for swaps
                mexc_symbol = f"{symbol}:USDT" if ':' not in symbol else symbol
            
            logger.info(f"Trading {mexc_symbol} (from {symbol})")
            
            # Get the market info
            market = self.exchange.markets[mexc_symbol]
            market_id = market['id']  # Use the exchange's internal ID
            logger.info(f"Market ID: {market_id}, Market Type: {market.get('type')}")
            
            # Set leverage first
            position_type = 1 if direction == 'LONG' else 2
            try:
                await self.exchange.set_leverage(
                    leverage, 
                    mexc_symbol,
                    params={
                        'openType': 2,  # Cross margin
                        'positionType': position_type
                    }
                )
                logger.info(f"Leverage set to {leverage}x")
            except Exception as e:
                logger.warning(f"Could not set leverage: {e}")
            
            # Calculate amount to buy/sell  
            amount = position_size_usdt / entry_price
            logger.info(f"Position size: ${position_size_usdt:.2f}, Entry: ${entry_price:.2f}, Amount: {amount:.4f}")
            
            # Use LIMIT order instead of market (may work better with MEXC)
            side = 'buy' if direction == 'LONG' else 'sell'
            
            # Use entry price as limit price for immediate fill
            order = await self.exchange.create_order(
                symbol=mexc_symbol,
                type='limit',
                side=side,
                amount=amount,
                price=entry_price,
                params={
                    'defaultType': 'swap'  # Explicitly set to swap/futures market
                }
            )
            
            logger.info(f"✅ Order placed successfully: {order}")
            
            # Place stop loss order (close position)
            sl_side = 'sell' if direction == 'LONG' else 'buy'
            try:
                stop_order = await self.exchange.create_order(
                    symbol=mexc_symbol,
                    type='STOP_MARKET',
                    side=sl_side,
                    amount=amount,
                    params={
                        'stopPrice': stop_loss,
                        'defaultType': 'swap',
                        'reduceOnly': True
                    }
                )
                logger.info(f"✅ Stop loss placed at ${stop_loss:.2f}: {stop_order}")
            except Exception as e:
                logger.error(f"❌ Could not place SL: {e}")
            
            # Place take profit order
            try:
                tp_order = await self.exchange.create_order(
                    symbol=mexc_symbol,
                    type='TAKE_PROFIT_MARKET',
                    side=sl_side,
                    amount=amount,
                    params={
                        'stopPrice': take_profit,
                        'defaultType': 'swap',
                        'reduceOnly': True
                    }
                )
                logger.info(f"✅ Take profit placed at ${take_profit:.2f}: {tp_order}")
            except Exception as e:
                logger.error(f"❌ Could not place TP: {e}")
            
            return {
                'order': order,
                'stop_loss': stop_order,
                'take_profit': tp_order
            }
            
        except Exception as e:
            logger.error(f"Error placing trade: {e}", exc_info=True)
            return None
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        try:
            # Convert to MEXC format if needed
            if ':USDT' not in symbol:
                mexc_symbol = f"{symbol}:USDT"
            else:
                mexc_symbol = symbol
                
            ticker = await self.exchange.fetch_ticker(mexc_symbol)
            return ticker.get('last')
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None
    
    async def close_partial_position(
        self,
        symbol: str,
        direction: str,
        amount_to_close: float,
        close_price: float
    ) -> Optional[Dict]:
        """
        Close a portion of an open position
        
        Args:
            symbol: Trading pair
            direction: Original position direction ('LONG' or 'SHORT')
            amount_to_close: Amount in base currency to close
            close_price: Current market price
        """
        try:
            # Convert to MEXC format if needed
            if ':USDT' not in symbol:
                mexc_symbol = f"{symbol}:USDT"
            else:
                mexc_symbol = symbol
            
            # For LONG positions, we SELL to close
            # For SHORT positions, we BUY to close
            close_side = 'sell' if direction == 'LONG' else 'buy'
            
            order = await self.exchange.create_market_order(
                symbol=mexc_symbol,
                side=close_side,
                amount=amount_to_close,
                params={
                    'positionSide': direction.lower(),
                    'reduceOnly': True
                }
            )
            
            logger.info(f"Partial close order placed: {order}")
            return order
            
        except Exception as e:
            logger.error(f"Error closing partial position: {e}", exc_info=True)
            return None
    
    async def close(self):
        """Close the exchange connection"""
        await self.exchange.close()


async def check_anti_overtrading(prefs: "UserPreference", symbol: str, db: Session) -> tuple[bool, str]:
    """
    Check anti-overtrading filters
    Returns: (allowed, reason)
    """
    from datetime import datetime, timedelta
    import json
    
    now = datetime.utcnow()
    
    # Reset daily counter if new day
    if not prefs.trades_reset_date or prefs.trades_reset_date.date() < now.date():
        prefs.trades_today = 0
        prefs.trades_reset_date = now
        db.commit()
    
    # Check max trades per day
    if prefs.trades_today >= prefs.max_trades_per_day:
        return False, f"Max trades per day limit reached ({prefs.max_trades_per_day})"
    
    # Check general trade cooldown
    if prefs.last_trade_time:
        cooldown_end = prefs.last_trade_time + timedelta(minutes=prefs.trade_cooldown_minutes)
        if now < cooldown_end:
            minutes_left = int((cooldown_end - now).total_seconds() / 60)
            return False, f"Trade cooldown active ({minutes_left} min remaining)"
    
    # Check same-symbol cooldown
    try:
        symbol_trades = json.loads(prefs.last_symbol_trades) if prefs.last_symbol_trades else {}
    except:
        symbol_trades = {}
    
    if symbol in symbol_trades:
        last_time_str = symbol_trades[symbol]
        last_time = datetime.fromisoformat(last_time_str)
        symbol_cooldown_end = last_time + timedelta(minutes=prefs.same_symbol_cooldown_minutes)
        if now < symbol_cooldown_end:
            minutes_left = int((symbol_cooldown_end - now).total_seconds() / 60)
            return False, f"Same symbol cooldown for {symbol} ({minutes_left} min remaining)"
    
    return True, "Anti-overtrading checks passed"


def calculate_adaptive_position_size(base_size: float, prefs: "UserPreference") -> float:
    """
    Calculate position size with adaptive sizing based on win/loss streaks
    Returns: adjusted position size
    """
    if not prefs.adaptive_sizing_enabled:
        return base_size
    
    # Apply win/loss streak adjustment
    if prefs.current_win_streak > 0:
        # Increase size after wins, max 1.5x
        multiplier = min(1.5, 1.0 + (prefs.current_win_streak * 0.1))
        return base_size * multiplier
    elif prefs.current_win_streak < 0:
        # Decrease size after losses, min 0.5x
        loss_count = abs(prefs.current_win_streak)
        divider = max(0.5, 1.0 - (loss_count * 0.1))
        return base_size * divider
    
    return base_size


def calculate_rr_scaled_size(base_size: float, signal_data: dict, prefs: "UserPreference") -> float:
    """
    Scale position size based on risk:reward ratio
    Better R:R = larger position
    """
    if not prefs.rr_scaling_enabled:
        return base_size
    
    entry = signal_data['entry_price']
    stop = signal_data['stop_loss']
    take_profit = signal_data.get('take_profit_3', signal_data['take_profit'])
    
    # Calculate R:R ratio
    risk = abs(entry - stop)
    reward = abs(take_profit - entry)
    rr_ratio = reward / risk if risk > 0 else 0
    
    # Scale size if R:R is above minimum threshold
    if rr_ratio >= prefs.min_rr_for_full_size:
        # Add extra % per R above threshold
        extra_r = rr_ratio - prefs.min_rr_for_full_size
        size_multiplier = 1.0 + (extra_r * prefs.rr_scaling_multiplier)
        return base_size * min(1.5, size_multiplier)  # Cap at 1.5x
    else:
        # Reduce size for lower R:R
        reduction = (prefs.min_rr_for_full_size - rr_ratio) * 0.15
        return base_size * max(0.5, 1.0 - reduction)


async def calculate_market_condition_adjustment(
    trader: "MEXCTrader", 
    symbol: str, 
    signal_data: dict, 
    prefs: "UserPreference"
) -> float:
    """
    Adjust position size based on market volatility
    Returns: size multiplier (0.6-1.0)
    """
    if not prefs.market_condition_adaptive:
        return 1.0
    
    # Get ATR from signal data
    atr = signal_data.get('atr', 0)
    entry_price = signal_data['entry_price']
    
    if not atr or not entry_price:
        return 1.0
    
    # Calculate ATR as % of price
    atr_percent = (atr / entry_price) * 100
    
    # High volatility = reduce position size
    if atr_percent > prefs.volatility_threshold_high:
        return prefs.high_volatility_size_reduction  # Default 0.6
    # Low volatility = use full size
    elif atr_percent < prefs.volatility_threshold_low:
        return 1.0
    # Medium volatility = slight reduction
    else:
        return 0.8
    
    return 1.0


async def check_security_limits(prefs: "UserPreference", balance: float, db: Session, user: User) -> tuple[bool, str]:
    """
    Check all security limits before trading
    Returns: (allowed, reason)
    """
    from datetime import datetime, timedelta
    from app.services.bot import bot
    
    # Emergency stop check
    if prefs.emergency_stop:
        return False, "Emergency stop is active"
    
    # Minimum balance check
    if balance < prefs.min_balance:
        await bot.send_message(user.telegram_id, f"⚠️ Balance ${balance:.2f} below minimum ${prefs.min_balance:.2f}. Auto-trading paused.")
        return False, f"Balance below minimum (${prefs.min_balance})"
    
    # Update peak balance
    if balance > prefs.peak_balance:
        prefs.peak_balance = balance
        db.commit()
    
    # Maximum drawdown check
    if prefs.peak_balance > 0:
        drawdown_percent = ((prefs.peak_balance - balance) / prefs.peak_balance) * 100
        if drawdown_percent > prefs.max_drawdown_percent:
            if not prefs.safety_paused:
                prefs.safety_paused = True
                db.commit()
                await bot.send_message(user.telegram_id, f"🚨 DRAWDOWN LIMIT HIT!\n\nDrawdown: {drawdown_percent:.1f}%\nLimit: {prefs.max_drawdown_percent}%\n\nAuto-trading PAUSED for safety.\n\nTo resume: /security_settings → Toggle Emergency Stop OFF")
            return False, f"Max drawdown exceeded ({drawdown_percent:.1f}%)"
        elif prefs.safety_paused and drawdown_percent <= (prefs.max_drawdown_percent * 0.8):
            # Auto-resume if drawdown recovers to 80% of limit
            prefs.safety_paused = False
            db.commit()
            await bot.send_message(user.telegram_id, f"✅ Drawdown recovered to {drawdown_percent:.1f}%\n\nSafety pause lifted. Auto-trading can resume.")
    
    # Daily loss limit check
    now = datetime.utcnow()
    if not prefs.daily_loss_reset_date or prefs.daily_loss_reset_date.date() < now.date():
        # New day - reset daily tracking and safety pause if it was due to daily loss
        prefs.daily_loss_reset_date = now
        if prefs.safety_paused:
            prefs.safety_paused = False
            await bot.send_message(user.telegram_id, f"✅ Daily loss limit reset!\n\nNew trading day started. Safety pause lifted.")
        db.commit()
    
    # Calculate today's losses
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_trades = db.query(Trade).filter(
        Trade.user_id == user.id,
        Trade.closed_at >= today_start,
        Trade.status == 'closed'
    ).all()
    
    daily_pnl = sum(t.pnl for t in today_trades)
    if daily_pnl < 0 and abs(daily_pnl) >= prefs.daily_loss_limit:
        if not prefs.safety_paused:
            prefs.safety_paused = True
            db.commit()
            await bot.send_message(user.telegram_id, f"🚨 DAILY LOSS LIMIT HIT!\n\nLoss: ${abs(daily_pnl):.2f}\nLimit: ${prefs.daily_loss_limit:.2f}\n\nAuto-trading PAUSED until tomorrow.")
        return False, f"Daily loss limit exceeded (${abs(daily_pnl):.2f})"
    
    # Consecutive losses check
    if prefs.consecutive_losses >= prefs.max_consecutive_losses:
        # Check if cooldown period has passed
        if prefs.last_loss_time:
            cooldown_end = prefs.last_loss_time + timedelta(minutes=prefs.cooldown_after_loss)
            if now < cooldown_end:
                minutes_left = int((cooldown_end - now).total_seconds() / 60)
                return False, f"Cooldown active ({minutes_left} min left after {prefs.max_consecutive_losses} losses)"
            else:
                # Cooldown passed, reset counter
                prefs.consecutive_losses = 0
                prefs.safety_paused = False
                db.commit()
                await bot.send_message(user.telegram_id, f"✅ Cooldown period ended!\n\nLoss streak reset. Auto-trading can resume.")
    
    # Check safety pause from any source
    if prefs.safety_paused:
        return False, "Trading paused by safety limits"
    
    return True, "All security checks passed"


def get_symbol_leverage(symbol: str, default_leverage: int = 10) -> int:
    """
    Get leverage based on symbol to meet MEXC minimum order sizes
    Higher leverage for expensive coins like BTC, ETH, SOL
    """
    leverage_map = {
        'BTC/USDT:USDT': 40,  # BTC needs 40x leverage
        'ETH/USDT:USDT': 20,  # ETH needs 20x leverage
        'SOL/USDT:USDT': 20,  # SOL needs 20x leverage
    }
    return leverage_map.get(symbol, default_leverage)


async def execute_auto_trade(signal_data: dict, user: User, db: Session):
    """Execute auto-trade for a user based on signal"""
    
    prefs = user.preferences
    if not prefs or not prefs.auto_trading_enabled:
        return
    
    # Check if paper trading mode is enabled
    if prefs.paper_trading_mode:
        from app.services.paper_trader import PaperTrader
        
        # Get signal from signal_data
        signal_id = signal_data.get('signal_id')
        if signal_id:
            signal = db.query(Signal).filter(Signal.id == signal_id).first()
            if signal:
                PaperTrader.execute_paper_trade(user.id, signal, db)
                logger.info(f"Paper trade executed for user {user.telegram_id}")
        return
    
    if not prefs.mexc_api_key or not prefs.mexc_api_secret:
        logger.warning(f"User {user.telegram_id} has auto-trading enabled but no API keys")
        return
    
    # Check if signal risk level is accepted by user
    signal_risk = signal_data.get('risk_level', 'MEDIUM')
    accepted_risks = [r.strip() for r in prefs.accepted_risk_levels.split(',')]
    
    if signal_risk not in accepted_risks:
        logger.info(f"User {user.telegram_id} skipping {signal_risk} risk signal (only accepts: {accepted_risks})")
        return
    
    # Check max positions limit
    open_positions = db.query(Trade).filter(
        Trade.user_id == user.id,
        Trade.status == 'open'
    ).count()
    
    if open_positions >= prefs.max_positions:
        logger.info(f"User {user.telegram_id} has reached max positions ({prefs.max_positions})")
        return
    
    # Anti-overtrading checks
    allowed, reason = await check_anti_overtrading(prefs, signal_data['symbol'], db)
    if not allowed:
        logger.info(f"Anti-overtrading check failed for user {user.telegram_id}: {reason}")
        return
    
    # Decrypt API keys for use
    api_key = decrypt_api_key(prefs.mexc_api_key)
    api_secret = decrypt_api_key(prefs.mexc_api_secret)
    
    trader = MEXCTrader(api_key, api_secret)
    
    try:
        # Get account balance
        balance = await trader.get_account_balance()
        
        if balance <= 0:
            logger.warning(f"User {user.telegram_id} has no USDT balance")
            return
        
        # Security checks
        allowed, reason = await check_security_limits(prefs, balance, db, user)
        if not allowed:
            logger.info(f"Security check failed for user {user.telegram_id}: {reason}")
            return
        
        # Calculate position size with ALL advanced features
        base_position_percent = prefs.position_size_percent
        
        # 1. Risk-based sizing
        if prefs.risk_based_sizing:
            if signal_risk == 'MEDIUM':
                base_position_percent *= 0.7
        
        # Calculate base USDT size
        base_size = await trader.calculate_position_size(balance, base_position_percent)
        
        # 2. Win/loss streak adaptive sizing
        adaptive_size = calculate_adaptive_position_size(base_size, prefs)
        
        # 3. Risk:Reward ratio scaling
        rr_scaled_size = calculate_rr_scaled_size(adaptive_size, signal_data, prefs)
        
        # 4. Market condition adjustment (volatility-based)
        vol_multiplier = await calculate_market_condition_adjustment(trader, signal_data['symbol'], signal_data, prefs)
        final_size = rr_scaled_size * vol_multiplier
        
        # CRITICAL: Cap total multiplier to prevent over-leveraging
        # All multipliers combined should never exceed 1.5x base size
        max_allowed_size = base_size * 1.5
        position_size = min(final_size, max_allowed_size)
        
        # Additional safety: Never more than 50% of balance
        max_position = balance * 0.5
        position_size = min(position_size, max_position)
        
        logger.info(f"Advanced position sizing: base=${base_size:.2f}, volatility_adj={vol_multiplier}, final=${position_size:.2f}")
        
        # Determine leverage based on symbol (higher for expensive coins)
        leverage = get_symbol_leverage(signal_data['symbol'], prefs.user_leverage)
        
        logger.info(f"Using {leverage}x leverage for {signal_data['symbol']}")
        
        # Place trade with symbol-specific leverage
        result = await trader.place_trade(
            symbol=signal_data['symbol'],
            direction=signal_data['direction'],
            entry_price=signal_data['entry_price'],
            stop_loss=signal_data['stop_loss'],
            take_profit=signal_data['take_profit'],
            position_size_usdt=position_size,
            leverage=leverage
        )
        
        if result:
            # Create trade record with all 3 TP levels and advanced tracking
            trade = Trade(
                user_id=user.id,
                signal_id=None,  # Will be set when signal is saved
                symbol=signal_data['symbol'],
                direction=signal_data['direction'],
                entry_price=signal_data['entry_price'],
                stop_loss=signal_data['stop_loss'],
                take_profit=signal_data.get('take_profit_3', signal_data['take_profit']),
                take_profit_1=signal_data.get('take_profit_1'),
                take_profit_2=signal_data.get('take_profit_2'),
                take_profit_3=signal_data.get('take_profit_3', signal_data['take_profit']),
                position_size=position_size,
                remaining_size=position_size,
                highest_price=signal_data['entry_price'] if signal_data['direction'] == 'LONG' else None,
                lowest_price=signal_data['entry_price'] if signal_data['direction'] == 'SHORT' else None,
                status='open'
            )
            db.add(trade)
            
            # Update anti-overtrading tracking
            from datetime import datetime
            import json
            
            prefs.trades_today += 1
            prefs.last_trade_time = datetime.utcnow()
            
            # Update last symbol trades
            try:
                symbol_trades = json.loads(prefs.last_symbol_trades) if prefs.last_symbol_trades else {}
            except:
                symbol_trades = {}
            symbol_trades[signal_data['symbol']] = datetime.utcnow().isoformat()
            prefs.last_symbol_trades = json.dumps(symbol_trades)
            
            db.commit()
            
            logger.info(f"Auto-trade executed for user {user.telegram_id}: {signal_data['symbol']} {signal_data['direction']}, size: ${position_size:.2f}, leverage: {prefs.user_leverage}x")
        
    except Exception as e:
        logger.error(f"Error executing auto-trade for user {user.telegram_id}: {e}", exc_info=True)
    
    finally:
        await trader.close()


async def monitor_positions():
    """Monitor open positions and send notifications when TP/SL is hit"""
    from datetime import datetime
    from app.services.bot import bot
    
    db = SessionLocal()
    try:
        # Get all open trades with users who have auto-trading enabled
        open_trades = db.query(Trade).join(User).join(UserPreference).filter(
            Trade.status == 'open',
            UserPreference.auto_trading_enabled == True,
            UserPreference.mexc_api_key.isnot(None)
        ).all()
        
        for trade in open_trades:
            trader = None
            try:
                user = trade.user
                prefs = user.preferences
                
                # Decrypt API keys
                api_key = decrypt_api_key(prefs.mexc_api_key)
                api_secret = decrypt_api_key(prefs.mexc_api_secret)
                
                trader = MEXCTrader(api_key, api_secret)
                
                # Get current price
                current_price = await trader.get_current_price(trade.symbol)
                
                if not current_price:
                    continue
                
                # Initialize remaining_size if not set
                if trade.remaining_size == 0:
                    trade.remaining_size = trade.position_size
                    db.commit()
                
                # Calculate position amount in base currency
                position_amount = trade.position_size / trade.entry_price
                remaining_amount = trade.remaining_size / trade.entry_price
                
                # ====================
                # DYNAMIC TRAILING STOP LOGIC
                # ====================
                if prefs.use_trailing_stop:
                    # Update highest/lowest price tracking
                    if trade.direction == 'LONG':
                        if not trade.highest_price or current_price > trade.highest_price:
                            trade.highest_price = current_price
                    else:  # SHORT
                        if not trade.lowest_price or current_price < trade.lowest_price:
                            trade.lowest_price = current_price
                    
                    # Calculate profit percentage
                    if trade.direction == 'LONG':
                        profit_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100
                    else:
                        profit_pct = ((trade.entry_price - current_price) / trade.entry_price) * 100
                    
                    # Move to breakeven first
                    if prefs.use_breakeven_stop and not trade.breakeven_moved and profit_pct >= 1.0:
                        trade.stop_loss = trade.entry_price
                        trade.breakeven_moved = True
                        db.commit()
                        await bot.send_message(
                            user.telegram_id,
                            f"🔒 Stop Loss moved to BREAKEVEN\n\n"
                            f"Symbol: {trade.symbol}\n"
                            f"Entry: ${trade.entry_price:.4f}\n"
                            f"New SL: ${trade.stop_loss:.4f}\n\n"
                            f"Risk eliminated! 🎯"
                        )
                    
                    # Activate trailing if profit > activation threshold
                    if profit_pct >= prefs.trailing_activation_percent:
                        if not trade.trailing_active:
                            trade.trailing_active = True
                            db.commit()
                        
                        # Calculate trailing stop price
                        if trade.direction == 'LONG':
                            new_trailing_stop = trade.highest_price * (1 - prefs.trailing_step_percent / 100)
                            if new_trailing_stop > trade.stop_loss:
                                trade.trailing_stop_price = new_trailing_stop
                                trade.stop_loss = new_trailing_stop
                                db.commit()
                                logger.info(f"Trailing stop updated for trade {trade.id}: ${new_trailing_stop:.4f}")
                        else:  # SHORT
                            new_trailing_stop = trade.lowest_price * (1 + prefs.trailing_step_percent / 100)
                            if new_trailing_stop < trade.stop_loss:
                                trade.trailing_stop_price = new_trailing_stop
                                trade.stop_loss = new_trailing_stop
                                db.commit()
                                logger.info(f"Trailing stop updated for trade {trade.id}: ${new_trailing_stop:.4f}")
                
                # Check if TP levels or SL hit
                tp1_hit = False
                tp2_hit = False
                tp3_hit = False
                sl_hit = False
                
                if trade.direction == 'LONG':
                    if not trade.tp1_hit and trade.take_profit_1 and current_price >= trade.take_profit_1:
                        tp1_hit = True
                    elif not trade.tp2_hit and trade.take_profit_2 and current_price >= trade.take_profit_2:
                        tp2_hit = True
                    elif not trade.tp3_hit and trade.take_profit_3 and current_price >= trade.take_profit_3:
                        tp3_hit = True
                    elif current_price <= trade.stop_loss:
                        sl_hit = True
                else:  # SHORT
                    if not trade.tp1_hit and trade.take_profit_1 and current_price <= trade.take_profit_1:
                        tp1_hit = True
                    elif not trade.tp2_hit and trade.take_profit_2 and current_price <= trade.take_profit_2:
                        tp2_hit = True
                    elif not trade.tp3_hit and trade.take_profit_3 and current_price <= trade.take_profit_3:
                        tp3_hit = True
                    elif current_price >= trade.stop_loss:
                        sl_hit = True
                
                # Handle TP1 hit (30% close)
                if tp1_hit:
                    close_percent = prefs.tp1_percent / 100
                    amount_to_close = remaining_amount * close_percent
                    
                    # Execute partial close
                    result = await trader.close_partial_position(
                        symbol=trade.symbol,
                        direction=trade.direction,
                        amount_to_close=amount_to_close,
                        close_price=current_price
                    )
                    
                    if result:
                        # Calculate partial PnL
                        price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                        pnl_usd = (price_change / trade.entry_price) * (amount_to_close * current_price)
                        
                        # Update trade
                        trade.tp1_hit = True
                        trade.remaining_size = trade.remaining_size - (amount_to_close * current_price)
                        trade.pnl += float(pnl_usd)
                        
                        db.commit()
                        
                        await bot.send_message(
                            user.telegram_id,
                            f"🎯 TP1 HIT! ({prefs.tp1_percent}% closed)\n\n"
                            f"Symbol: {trade.symbol}\n"
                            f"Direction: {trade.direction}\n"
                            f"TP1 Price: ${trade.take_profit_1:.4f}\n"
                            f"Current: ${current_price:.4f}\n\n"
                            f"💰 Partial PnL: ${pnl_usd:.2f}\n"
                            f"Remaining: {(100-prefs.tp1_percent)}% of position"
                        )
                        
                        logger.info(f"TP1 hit for trade {trade.id}: closed {prefs.tp1_percent}%, PnL: ${pnl_usd:.2f}")
                
                # Handle TP2 hit (30% of remaining)
                elif tp2_hit:
                    close_percent = prefs.tp2_percent / 100
                    amount_to_close = remaining_amount * close_percent
                    
                    result = await trader.close_partial_position(
                        symbol=trade.symbol,
                        direction=trade.direction,
                        amount_to_close=amount_to_close,
                        close_price=current_price
                    )
                    
                    if result:
                        price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                        pnl_usd = (price_change / trade.entry_price) * (amount_to_close * current_price)
                        
                        trade.tp2_hit = True
                        trade.remaining_size = trade.remaining_size - (amount_to_close * current_price)
                        trade.pnl += float(pnl_usd)
                        
                        db.commit()
                        
                        await bot.send_message(
                            user.telegram_id,
                            f"🎯 TP2 HIT! ({prefs.tp2_percent}% closed)\n\n"
                            f"Symbol: {trade.symbol}\n"
                            f"Direction: {trade.direction}\n"
                            f"TP2 Price: ${trade.take_profit_2:.4f}\n"
                            f"Current: ${current_price:.4f}\n\n"
                            f"💰 Partial PnL: ${pnl_usd:.2f}\n"
                            f"Total PnL: ${trade.pnl:.2f}"
                        )
                        
                        logger.info(f"TP2 hit for trade {trade.id}: closed {prefs.tp2_percent}%, PnL: ${pnl_usd:.2f}")
                
                # Handle TP3 hit (close remaining position)
                elif tp3_hit:
                    # Close entire remaining position
                    result = await trader.close_partial_position(
                        symbol=trade.symbol,
                        direction=trade.direction,
                        amount_to_close=remaining_amount,
                        close_price=current_price
                    )
                    
                    if result:
                        # Calculate final PnL on remaining portion
                        price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                        pnl_usd = (price_change / trade.entry_price) * (remaining_amount * current_price)
                        
                        trade.tp3_hit = True
                        trade.status = 'closed'
                        trade.exit_price = current_price
                        trade.closed_at = datetime.utcnow()
                        trade.remaining_size = 0
                        trade.pnl += float(pnl_usd)
                        
                        # Calculate total PnL percent
                        trade.pnl_percent = (trade.pnl / (trade.position_size / 10)) * 100
                        
                        # Reset consecutive losses on win
                        prefs.consecutive_losses = 0
                        
                        # Update win streak (TP hit = win)
                        if prefs.current_win_streak < 0:
                            prefs.current_win_streak = 1
                        else:
                            prefs.current_win_streak += 1
                        
                        db.commit()
                        
                        # Update signal analytics
                        if trade.signal_id:
                            AnalyticsService.update_signal_outcome(db, trade.signal_id)
                        
                        await bot.send_message(
                            user.telegram_id,
                            f"🎯 TP3 HIT! Position CLOSED 🎯\n\n"
                            f"Symbol: {trade.symbol}\n"
                            f"Direction: {trade.direction}\n"
                            f"Entry: ${trade.entry_price:.4f}\n"
                            f"TP3: ${trade.take_profit_3:.4f}\n"
                            f"Current: ${current_price:.4f}\n\n"
                            f"💰 Final PnL: ${pnl_usd:.2f}\n"
                            f"💰 Total PnL: ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%)\n"
                            f"Position Size: ${trade.position_size:.2f}"
                        )
                        
                        logger.info(f"TP3 hit for trade {trade.id}: position closed, total PnL: ${trade.pnl:.2f}")
                
                # Handle SL hit (closes remaining position)
                elif sl_hit:
                    # Close remaining position at stop loss
                    result = await trader.close_partial_position(
                        symbol=trade.symbol,
                        direction=trade.direction,
                        amount_to_close=remaining_amount,
                        close_price=current_price
                    )
                    
                    if result:
                        # Calculate loss on remaining portion
                        price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                        pnl_usd = (price_change / trade.entry_price) * (remaining_amount * current_price)
                        
                        trade.status = 'closed'
                        trade.exit_price = current_price
                        trade.closed_at = datetime.utcnow()
                        trade.remaining_size = 0
                        trade.pnl += float(pnl_usd)
                        trade.pnl_percent = (trade.pnl / (trade.position_size / 10)) * 100
                    
                    # Increment consecutive losses
                    prefs.consecutive_losses += 1
                    prefs.last_loss_time = datetime.utcnow()
                    
                    # Update loss streak (SL hit = loss)
                    if prefs.current_win_streak > 0:
                        prefs.current_win_streak = -1
                    else:
                        prefs.current_win_streak -= 1
                    
                    # Check if consecutive loss limit hit
                    if prefs.consecutive_losses >= prefs.max_consecutive_losses:
                        prefs.safety_paused = True
                        await bot.send_message(
                            user.telegram_id,
                            f"🚨 CONSECUTIVE LOSS LIMIT HIT!\n\n"
                            f"Losses in a row: {prefs.consecutive_losses}\n"
                            f"Limit: {prefs.max_consecutive_losses}\n\n"
                            f"Auto-trading PAUSED for {prefs.cooldown_after_loss} minutes."
                        )
                    
                    db.commit()
                    
                    # Update signal analytics
                    if trade.signal_id:
                        AnalyticsService.update_signal_outcome(db, trade.signal_id)
                    
                    # Send notification
                    await bot.send_message(
                        user.telegram_id,
                        f"🛑 STOP LOSS HIT 🛑\n\n"
                        f"Symbol: {trade.symbol}\n"
                        f"Direction: {trade.direction}\n"
                        f"Entry: ${trade.entry_price:.4f}\n"
                        f"Exit: ${trade.stop_loss:.4f}\n"
                        f"Current: ${current_price:.4f}\n\n"
                        f"💸 PnL: ${pnl_usd:.2f} ({pnl_percent:+.1f}%)\n"
                        f"Position Size: ${trade.position_size:.2f}\n"
                        f"Leverage: 10x\n\n"
                        f"Consecutive losses: {prefs.consecutive_losses}"
                    )
                    
                    logger.info(f"SL hit for trade {trade.id}: {trade.symbol} {trade.direction}, PnL: ${pnl_usd:.2f}")
                
            except Exception as e:
                logger.error(f"Error monitoring trade {trade.id}: {e}", exc_info=True)
            finally:
                if trader:
                    await trader.close()
    
    except Exception as e:
        logger.error(f"Error in position monitor: {e}", exc_info=True)
    
    finally:
        db.close()
