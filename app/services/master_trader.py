"""
Bitunix Master Trader Service - Copy Trading API Integration

This service executes trades on the master trader's Copy Trading account.
All bot signals are mirrored to the master account, allowing followers
to copy trades via Bitunix's native copy trading system.

User trades remain unchanged - this runs in parallel transparently.
"""

import os
import logging
import ccxt.async_support as ccxt
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MasterTraderService:
    """Handles trade execution on master Copy Trading account"""
    
    def __init__(self):
        self.exchange = None
        self.api_key = os.getenv('BITUNIX_MASTER_API_KEY')
        self.api_secret = os.getenv('BITUNIX_MASTER_SECRET')
        
        if not self.api_key or not self.api_secret:
            logger.warning("⚠️ Master Trader credentials not configured - trades will not be mirrored")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("✅ Master Trader service initialized")
    
    async def initialize(self):
        """Initialize CCXT exchange connection for master account"""
        if not self.enabled:
            return
        
        try:
            self.exchange = ccxt.bitunix({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',
                }
            })
            
            # Test connection
            await self.exchange.load_markets()
            balance = await self.exchange.fetch_balance()
            
            logger.info(f"🎯 Master Trader account connected - Balance: ${balance.get('USDT', {}).get('free', 0):.2f} USDT")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Master Trader account: {e}", exc_info=True)
            self.enabled = False
    
    async def execute_signal_on_master(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit_1: Optional[float] = None,
        take_profit_2: Optional[float] = None,
        take_profit_3: Optional[float] = None,
        leverage: int = 5,
        position_size_percent: float = 10.0
    ) -> Optional[Dict]:
        """
        Execute a signal on the master Copy Trading account
        
        This mirrors bot signals to the master account so Bitunix followers
        can automatically copy the trades.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT:USDT')
            direction: 'LONG' or 'SHORT'
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit_1: First take profit target
            take_profit_2: Second take profit target
            take_profit_3: Third take profit target
            leverage: Leverage multiplier (1-20x)
            position_size_percent: Position size as % of account balance (default 10%)
        
        Returns:
            Trade execution result dict or None if failed
        """
        if not self.enabled:
            return None
        
        try:
            # Get account balance and calculate position size (10% of balance)
            balance = await self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            if usdt_balance <= 0:
                logger.warning("⚠️ Master account has no USDT balance - skipping trade")
                return None
            
            position_size_usd = usdt_balance * (position_size_percent / 100.0)
            logger.info(f"💰 Master account balance: ${usdt_balance:.2f} → Position size: ${position_size_usd:.2f} ({position_size_percent}%)")
            # Convert symbol format (BTC/USDT -> BTC/USDT:USDT for futures)
            if ':' not in symbol:
                symbol = f"{symbol}:USDT"
            
            # Set leverage
            try:
                await self.exchange.set_leverage(leverage, symbol)
            except Exception as e:
                logger.warning(f"Failed to set leverage for {symbol}: {e}")
            
            # Calculate position size in contracts
            amount = position_size_usd / entry_price
            
            # Determine side
            side = 'buy' if direction == 'LONG' else 'sell'
            
            # Place main entry order (market order for immediate execution)
            logger.info(f"🎯 MASTER TRADE: {direction} {symbol} @ ${entry_price:.6f} ({leverage}x)")
            
            entry_order = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount,
                params={'leverage': leverage}
            )
            
            logger.info(f"✅ Master entry executed: {entry_order.get('id')}")
            
            # Place stop loss order
            sl_side = 'sell' if direction == 'LONG' else 'buy'
            
            try:
                sl_order = await self.exchange.create_order(
                    symbol=symbol,
                    type='stop_market',
                    side=sl_side,
                    amount=amount,
                    params={
                        'stopPrice': stop_loss,
                        'reduceOnly': True
                    }
                )
                logger.info(f"✅ Master SL placed @ ${stop_loss:.6f}")
            except Exception as e:
                logger.error(f"❌ Failed to place master SL: {e}")
            
            # Place take profit orders (split position across TPs)
            tp_orders = []
            tp_targets = [tp for tp in [take_profit_1, take_profit_2, take_profit_3] if tp is not None]
            
            if tp_targets:
                # Split position across TPs
                tp_amount = amount / len(tp_targets)
                tp_side = 'sell' if direction == 'LONG' else 'buy'
                
                for i, tp_price in enumerate(tp_targets, 1):
                    try:
                        tp_order = await self.exchange.create_order(
                            symbol=symbol,
                            type='take_profit_market',
                            side=tp_side,
                            amount=tp_amount,
                            params={
                                'stopPrice': tp_price,
                                'reduceOnly': True
                            }
                        )
                        tp_orders.append(tp_order)
                        logger.info(f"✅ Master TP{i} placed @ ${tp_price:.6f}")
                    except Exception as e:
                        logger.error(f"❌ Failed to place master TP{i}: {e}")
            
            return {
                'success': True,
                'entry_order': entry_order,
                'stop_loss_order': sl_order if 'sl_order' in locals() else None,
                'take_profit_orders': tp_orders,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"❌ Master trade execution failed for {symbol}: {e}", exc_info=True)
            return None
    
    async def close(self):
        """Close exchange connection"""
        if self.exchange:
            try:
                await self.exchange.close()
            except Exception as e:
                logger.error(f"Error closing master trader connection: {e}")


# Global instance
_master_trader_instance = None


async def get_master_trader() -> MasterTraderService:
    """Get or create global master trader instance"""
    global _master_trader_instance
    
    if _master_trader_instance is None:
        _master_trader_instance = MasterTraderService()
        await _master_trader_instance.initialize()
    
    return _master_trader_instance
