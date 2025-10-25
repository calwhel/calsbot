"""
Top Gainers Trading Mode - Generates signals from Bitunix top movers
Focuses on momentum plays with 5x leverage and 15% TP/SL
"""
import logging
import ccxt
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TopGainersSignalService:
    """Service to fetch and analyze top gainers from Bitunix"""
    
    def __init__(self):
        self.exchange = None
        self.min_volume_usdt = 1000000  # $1M minimum 24h volume for liquidity
        
    async def initialize(self):
        """Initialize Bitunix exchange connection"""
        try:
            self.exchange = ccxt.bitunix({
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}  # Perpetual futures
            })
            await self.exchange.load_markets()
            logger.info("TopGainersSignalService initialized with Bitunix")
        except Exception as e:
            logger.error(f"Failed to initialize TopGainersSignalService: {e}")
            raise
    
    async def get_top_gainers(self, limit: int = 10, min_change_percent: float = 5.0) -> List[Dict]:
        """
        Fetch top gainers from Bitunix based on 24h price change
        
        Args:
            limit: Number of top gainers to return
            min_change_percent: Minimum 24h change % to qualify
            
        Returns:
            List of {symbol, change_percent, volume, price} sorted by change %
        """
        try:
            if not self.exchange:
                await self.initialize()
            
            # Fetch all tickers
            tickers = await self.exchange.fetch_tickers()
            
            gainers = []
            for symbol, ticker in tickers.items():
                # Only consider USDT perpetuals
                if not symbol.endswith('/USDT'):
                    continue
                
                change_percent = ticker.get('percentage')
                volume_usdt = ticker.get('quoteVolume', 0)
                
                # Filter criteria
                if (change_percent and 
                    change_percent >= min_change_percent and 
                    volume_usdt >= self.min_volume_usdt):
                    
                    gainers.append({
                        'symbol': symbol,
                        'change_percent': round(change_percent, 2),
                        'volume_24h': round(volume_usdt, 0),
                        'price': ticker.get('last', 0),
                        'high_24h': ticker.get('high', 0),
                        'low_24h': ticker.get('low', 0)
                    })
            
            # Sort by change % descending
            gainers.sort(key=lambda x: x['change_percent'], reverse=True)
            
            return gainers[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching top gainers: {e}")
            return []
    
    async def get_top_losers(self, limit: int = 10, min_change_percent: float = -5.0) -> List[Dict]:
        """
        Fetch top losers from Bitunix based on 24h price change
        Used for potential SHORT opportunities on mean reversion
        
        Args:
            limit: Number of top losers to return
            min_change_percent: Minimum negative change % to qualify (e.g., -5.0)
            
        Returns:
            List of {symbol, change_percent, volume, price} sorted by change % ascending
        """
        try:
            if not self.exchange:
                await self.initialize()
            
            tickers = await self.exchange.fetch_tickers()
            
            losers = []
            for symbol, ticker in tickers.items():
                if not symbol.endswith('/USDT'):
                    continue
                
                change_percent = ticker.get('percentage')
                volume_usdt = ticker.get('quoteVolume', 0)
                
                # Filter for losers
                if (change_percent and 
                    change_percent <= min_change_percent and 
                    volume_usdt >= self.min_volume_usdt):
                    
                    losers.append({
                        'symbol': symbol,
                        'change_percent': round(change_percent, 2),
                        'volume_24h': round(volume_usdt, 0),
                        'price': ticker.get('last', 0),
                        'high_24h': ticker.get('high', 0),
                        'low_24h': ticker.get('low', 0)
                    })
            
            # Sort by change % ascending (most negative first)
            losers.sort(key=lambda x: x['change_percent'])
            
            return losers[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching top losers: {e}")
            return []
    
    async def analyze_momentum(self, symbol: str) -> Optional[Dict]:
        """
        Analyze short-term momentum to determine direction
        Uses 5m and 15m EMA trends to confirm momentum continuation
        
        Returns:
            {
                'direction': 'LONG' or 'SHORT',
                'confidence': 0-100,
                'entry_price': float,
                'reason': str
            }
        """
        try:
            # Fetch 5m candles
            candles_5m = await self.exchange.fetch_ohlcv(symbol, '5m', limit=50)
            candles_15m = await self.exchange.fetch_ohlcv(symbol, '15m', limit=50)
            
            if len(candles_5m) < 20 or len(candles_15m) < 20:
                return None
            
            # Calculate EMAs
            ema9_5m = self._calculate_ema([c[4] for c in candles_5m], 9)
            ema21_5m = self._calculate_ema([c[4] for c in candles_5m], 21)
            ema9_15m = self._calculate_ema([c[4] for c in candles_15m], 9)
            ema21_15m = self._calculate_ema([c[4] for c in candles_15m], 21)
            
            current_price = candles_5m[-1][4]
            
            # Check trend alignment
            bullish_5m = ema9_5m > ema21_5m
            bullish_15m = ema9_15m > ema21_15m
            
            # Both timeframes must agree
            if bullish_5m and bullish_15m:
                # Strong uptrend - LONG
                return {
                    'direction': 'LONG',
                    'confidence': 85,
                    'entry_price': current_price,
                    'reason': 'Strong momentum continuation - Both 5m and 15m EMAs bullish'
                }
            elif not bullish_5m and not bullish_15m:
                # Strong downtrend - SHORT (mean reversion)
                return {
                    'direction': 'SHORT',
                    'confidence': 75,
                    'entry_price': current_price,
                    'reason': 'Bearish momentum - Both 5m and 15m EMAs bearish, potential reversal'
                }
            else:
                # Mixed signals - skip
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing momentum for {symbol}: {e}")
            return None
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return sum(prices) / len(prices)
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    async def generate_top_gainer_signal(
        self, 
        min_change_percent: float = 5.0,
        max_symbols: int = 3
    ) -> Optional[Dict]:
        """
        Generate trading signal from top gainers
        
        Returns:
            {
                'symbol': str,
                'direction': 'LONG' or 'SHORT',
                'entry_price': float,
                'stop_loss': float,
                'take_profit': float,
                'confidence': int,
                'reasoning': str,
                'trade_type': 'TOP_GAINER'
            }
        """
        try:
            # Get top gainers
            gainers = await self.get_top_gainers(limit=max_symbols, min_change_percent=min_change_percent)
            
            if not gainers:
                logger.info("No top gainers found meeting criteria")
                return None
            
            # Analyze each gainer for momentum
            for gainer in gainers:
                symbol = gainer['symbol']
                
                # Analyze momentum
                momentum = await self.analyze_momentum(symbol)
                
                if not momentum:
                    continue
                
                entry_price = momentum['entry_price']
                
                # Calculate TP/SL (20% each for 1:1 with 5x leverage = 4% actual price move)
                # 20% profit/loss on 5x = 4% price movement
                price_change_percent = 4.0  # 4% actual price movement
                
                if momentum['direction'] == 'LONG':
                    stop_loss = entry_price * (1 - price_change_percent / 100)
                    take_profit = entry_price * (1 + price_change_percent / 100)
                else:  # SHORT
                    stop_loss = entry_price * (1 + price_change_percent / 100)
                    take_profit = entry_price * (1 - price_change_percent / 100)
                
                return {
                    'symbol': symbol,
                    'direction': momentum['direction'],
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': momentum['confidence'],
                    'reasoning': f"Top Gainer: {gainer['change_percent']}% in 24h | {momentum['reason']}",
                    'trade_type': 'TOP_GAINER',
                    'leverage': 5,  # Fixed 5x leverage for top gainers
                    '24h_change': gainer['change_percent'],
                    '24h_volume': gainer['volume_24h']
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating top gainer signal: {e}")
            return None
    
    async def close(self):
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()
            self.exchange = None


async def broadcast_top_gainer_signal(bot, db_session):
    """
    Scan for top gainers and broadcast signals to users with top_gainers_mode_enabled
    Called periodically by scheduler
    """
    from app.models import User, UserPreference, Signal, Trade
    from app.services.bitunix_trader import execute_bitunix_trade
    from datetime import datetime
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        service = TopGainersSignalService()
        await service.initialize()
        
        # Get all users with top gainers mode enabled
        users_with_mode = db_session.query(User).join(UserPreference).filter(
            UserPreference.top_gainers_mode_enabled == True,
            UserPreference.auto_trading_enabled == True
        ).all()
        
        if not users_with_mode:
            logger.info("No users with top gainers mode enabled")
            await service.close()
            return
        
        logger.info(f"Scanning top gainers for {len(users_with_mode)} users")
        
        # Generate top gainer signal
        # Use first user's preferences for min_change, but we'll check each user individually
        first_prefs = users_with_mode[0].preferences
        min_change = first_prefs.top_gainers_min_change if first_prefs else 5.0
        max_symbols = first_prefs.top_gainers_max_symbols if first_prefs else 3
        
        signal_data = await service.generate_top_gainer_signal(
            min_change_percent=min_change,
            max_symbols=max_symbols
        )
        
        if not signal_data:
            logger.info("No top gainer signals found")
            await service.close()
            return
        
        # Create signal record
        signal = Signal(
            symbol=signal_data['symbol'],
            direction=signal_data['direction'],
            entry_price=signal_data['entry_price'],
            stop_loss=signal_data['stop_loss'],
            take_profit=signal_data['take_profit'],
            confidence=signal_data['confidence'],
            reasoning=signal_data['reasoning'],
            signal_type='TOP_GAINER',
            timeframe='5m',
            created_at=datetime.utcnow()
        )
        db_session.add(signal)
        db_session.commit()
        db_session.refresh(signal)
        
        logger.info(f"🚀 TOP GAINER SIGNAL: {signal.symbol} {signal.direction} @ ${signal.entry_price} (24h: {signal_data.get('24h_change')}%)")
        
        # Broadcast to users
        signal_text = f"""
🔥 <b>TOP GAINER ALERT</b> 🔥

<b>{signal.symbol}</b> {signal.direction}
━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 <b>24h Change:</b> +{signal_data.get('24h_change')}%
💰 <b>24h Volume:</b> ${signal_data.get('24h_volume'):,.0f}

<b>Entry:</b> ${signal.entry_price:.6f}
<b>TP:</b> ${signal.take_profit:.6f} (+20% @ 5x)
<b>SL:</b> ${signal.stop_loss:.6f} (-20% @ 5x)

⚡ <b>Leverage:</b> 5x (Fixed for volatility)
🎯 <b>Risk/Reward:</b> 1:1

<b>Reasoning:</b>
{signal.reasoning}

⚠️ <b>HIGH VOLATILITY - TOP GAINER MODE</b>
<i>Auto-executing for users with mode enabled...</i>
"""
        
        # Execute trades for users with top gainers mode + auto-trading
        executed_count = 0
        for user in users_with_mode:
            prefs = user.preferences
            
            # Check if user has space for more top gainer positions
            current_top_gainer_positions = db_session.query(Trade).filter(
                Trade.user_id == user.id,
                Trade.status == 'open',
                Trade.trade_type == 'TOP_GAINER'
            ).count()
            
            max_allowed = prefs.top_gainers_max_symbols if prefs else 3
            
            if current_top_gainer_positions >= max_allowed:
                logger.info(f"User {user.id} already has {current_top_gainer_positions} top gainer positions (max: {max_allowed})")
                continue
            
            # Execute trade with 5x leverage override and TOP_GAINER trade_type
            trade = await execute_bitunix_trade(
                signal=signal,
                user=user,
                db=db_session,
                trade_type='TOP_GAINER',
                leverage_override=5  # Force 5x leverage for top gainers
            )
            
            if trade:
                executed_count += 1
                
                # Send notification
                try:
                    await bot.send_message(
                        user.telegram_id,
                        f"{signal_text}\n\n✅ <b>Trade Executed!</b>\n"
                        f"Position Size: ${trade.position_size:.2f}\n"
                        f"Leverage: 5x (Top Gainer Mode)",
                        parse_mode="HTML"
                    )
                except Exception as e:
                    logger.error(f"Failed to send notification to user {user.id}: {e}")
        
        logger.info(f"Top gainer signal executed for {executed_count}/{len(users_with_mode)} users")
        
        await service.close()
        
    except Exception as e:
        logger.error(f"Error in broadcast_top_gainer_signal: {e}", exc_info=True)
