"""
Advanced risk filters: correlation filter and funding rate monitoring
"""
import asyncio
import ccxt.async_support as ccxt
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from app.models import Trade, UserPreference
from app.config import settings
import logging

logger = logging.getLogger(__name__)

CORRELATION_GROUPS = {
    'BTC_GROUP': ['BTC/USDT', 'BTC/USD'],
    'ETH_GROUP': ['ETH/USDT', 'ETH/USD'],
    'LAYER1': ['SOL/USDT', 'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'NEAR/USDT'],
    'LAYER2': ['ARB/USDT', 'OP/USDT', 'MATIC/USDT'],
    'DEFI': ['AAVE/USDT', 'UNI/USDT', 'SUSHI/USDT', 'CRV/USDT', 'MKR/USDT'],
    'MEME': ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT'],
    'EXCHANGE': ['BNB/USDT', 'OKB/USDT', 'FTT/USDT'],
    'AI': ['AGIX/USDT', 'FET/USDT', 'OCEAN/USDT', 'RNDR/USDT'],
    'GAMING': ['AXS/USDT', 'SAND/USDT', 'MANA/USDT', 'GALA/USDT']
}


def get_correlation_group(symbol: str) -> Optional[str]:
    """Get the correlation group for a symbol"""
    for group_name, symbols in CORRELATION_GROUPS.items():
        if symbol in symbols:
            return group_name
    return None


def check_correlation_filter(symbol: str, user_prefs: UserPreference, db: Session) -> tuple[bool, str]:
    """
    Check if opening a position violates correlation limits
    
    Returns:
        (allowed, reason) - True if trade allowed, False with reason if blocked
    """
    if not user_prefs.correlation_filter_enabled:
        return True, "Correlation filter disabled"
    
    correlation_group = get_correlation_group(symbol)
    if not correlation_group:
        return True, "Symbol not in any correlation group"
    
    open_trades = db.query(Trade).filter(
        Trade.user_id == user_prefs.user_id,
        Trade.status == 'open'
    ).all()
    
    correlated_positions = 0
    correlated_symbols = []
    
    for trade in open_trades:
        trade_group = get_correlation_group(trade.symbol)
        if trade_group == correlation_group:
            correlated_positions += 1
            correlated_symbols.append(trade.symbol)
    
    if correlated_positions >= user_prefs.max_correlated_positions:
        return False, f"Max {user_prefs.max_correlated_positions} correlated position(s) already open in {correlation_group}: {', '.join(correlated_symbols)}"
    
    return True, "Correlation check passed"


async def check_funding_rates(symbols: List[str], exchange_name: str = 'binance') -> List[Dict]:
    """
    Check funding rates for given symbols and return extreme rates
    
    Returns list of {symbol, funding_rate, funding_timestamp, alert_type}
    """
    alerts = []
    exchange = None
    
    try:
        # Create async exchange instance
        if exchange_name == 'binance':
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
        else:
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
        
        for symbol in symbols:
            try:
                funding_rate = await exchange.fetch_funding_rate(symbol)
                
                if funding_rate and 'fundingRate' in funding_rate:
                    rate = funding_rate['fundingRate'] * 100  # Convert to percentage
                    
                    # Positive funding = longs pay shorts (market bullish)
                    # Negative funding = shorts pay longs (market bearish)
                    
                    alert_type = None
                    if rate > 0.1:
                        alert_type = 'HIGH_LONG_FUNDING'  # Longs overleveraged
                    elif rate < -0.1:
                        alert_type = 'HIGH_SHORT_FUNDING'  # Shorts overleveraged
                    
                    if alert_type:
                        alerts.append({
                            'symbol': symbol,
                            'funding_rate': round(rate, 4),
                            'funding_timestamp': funding_rate.get('timestamp'),
                            'alert_type': alert_type,
                            'daily_rate': round(rate * 3, 4)  # 3 funding periods per day
                        })
                        
            except Exception as e:
                logger.debug(f"Could not fetch funding rate for {symbol}: {e}")
        
    except Exception as e:
        logger.error(f"Error checking funding rates: {e}")
    finally:
        # Always close the exchange connection
        # Note: ccxt.async_support may emit cleanup warnings - this is a library limitation
        if exchange:
            try:
                # Manually close the aiohttp session first
                if hasattr(exchange, 'session') and exchange.session:
                    await exchange.session.close()
                    await asyncio.sleep(0.05)
                # Then close the exchange
                await exchange.close()
                await asyncio.sleep(0.05)
                # Clear reference
                exchange = None
            except Exception as e:
                logger.error(f"Error closing {exchange_name} exchange: {e}")
    
    return alerts


async def get_funding_rate_opportunity(symbol: str, funding_rate: float) -> Dict:
    """
    Analyze funding rate for arbitrage opportunity
    
    Returns trading recommendation based on extreme funding
    """
    opportunity = {
        'symbol': symbol,
        'funding_rate': funding_rate,
        'action': None,
        'reason': None,
        'expected_daily_return': round(funding_rate * 3, 4)  # 3 periods/day
    }
    
    if funding_rate > 0.15:
        opportunity['action'] = 'SHORT'
        opportunity['reason'] = f'Extreme long funding ({funding_rate:.2f}%). Collect funding + potential mean reversion.'
    elif funding_rate < -0.15:
        opportunity['action'] = 'LONG'
        opportunity['reason'] = f'Extreme short funding ({funding_rate:.2f}%). Collect funding + potential mean reversion.'
    
    return opportunity
