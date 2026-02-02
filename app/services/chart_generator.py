"""
Professional Chart Generator for Twitter Posts
Generates clean, modern chart images for crypto coins
"""

import io
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed - chart generation disabled")


async def get_ohlcv_data(symbol: str, timeframe: str = '1h', limit: int = 48) -> Optional[List]:
    """Fetch OHLCV data from Binance (tries Futures first, then Spot)"""
    # Try Binance Futures first (more coins available)
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        ohlcv = await exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe, limit=limit)
        await exchange.close()
        logger.info(f"✅ Got OHLCV from Binance Futures for {symbol}")
        return ohlcv
    except Exception as e:
        logger.warning(f"Futures OHLCV failed for {symbol}: {e}, trying spot...")
    
    # Fallback to Binance Spot
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        ohlcv = await exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe, limit=limit)
        await exchange.close()
        logger.info(f"✅ Got OHLCV from Binance Spot for {symbol}")
        return ohlcv
    except Exception as e:
        logger.error(f"Failed to fetch OHLCV for {symbol} (both futures and spot): {e}")
        return None


async def generate_coin_chart(symbol: str, change_24h: float, current_price: float) -> Optional[bytes]:
    """Generate a professional chart image for a coin"""
    if not HAS_MATPLOTLIB:
        return None
    
    try:
        ohlcv = await get_ohlcv_data(symbol, '1h', 48)
        if not ohlcv or len(ohlcv) < 10:
            return None
        
        timestamps = [datetime.fromtimestamp(candle[0] / 1000) for candle in ohlcv]
        closes = [candle[4] for candle in ohlcv]
        highs = [candle[2] for candle in ohlcv]
        lows = [candle[3] for candle in ohlcv]
        
        is_bullish = change_24h >= 0
        main_color = '#00C853' if is_bullish else '#FF1744'
        bg_color = '#0D1117'
        text_color = '#FFFFFF'
        grid_color = '#21262D'
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        
        ax.fill_between(timestamps, closes, min(lows) * 0.995, 
                       color=main_color, alpha=0.15)
        ax.plot(timestamps, closes, color=main_color, linewidth=2.5, solid_capstyle='round')
        
        ax.scatter([timestamps[-1]], [closes[-1]], color=main_color, s=80, zorder=5)
        
        ax.set_xlim(timestamps[0], timestamps[-1])
        price_range = max(highs) - min(lows)
        ax.set_ylim(min(lows) - price_range * 0.05, max(highs) + price_range * 0.15)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        
        ax.tick_params(colors=text_color, labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(grid_color)
        ax.spines['left'].set_color(grid_color)
        
        ax.grid(True, alpha=0.3, color=grid_color, linestyle='-', linewidth=0.5)
        
        sign = '+' if change_24h >= 0 else ''
        title = f"${symbol}  •  ${current_price:,.4f}" if current_price < 1 else f"${symbol}  •  ${current_price:,.2f}"
        ax.set_title(title, fontsize=20, fontweight='bold', color=text_color, pad=20, loc='left')
        
        change_text = f"{sign}{change_24h:.1f}%"
        ax.text(0.98, 0.95, change_text, transform=ax.transAxes, fontsize=18, 
               fontweight='bold', color=main_color, ha='right', va='top')
        
        ax.text(0.02, 0.02, "TradeHub AI", transform=ax.transAxes, fontsize=10,
               color='#586069', ha='left', va='bottom', alpha=0.7)
        
        ax.text(0.98, 0.02, "48H Chart", transform=ax.transAxes, fontsize=10,
               color='#586069', ha='right', va='bottom', alpha=0.7)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor=bg_color, 
                   edgecolor='none', bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        buf.seek(0)
        
        return buf.getvalue()
        
    except Exception as e:
        logger.error(f"Failed to generate chart for {symbol}: {e}")
        return None


async def generate_market_overview_chart(coins: List[Dict]) -> Optional[bytes]:
    """Generate a market overview bar chart"""
    if not HAS_MATPLOTLIB or not coins:
        return None
    
    try:
        bg_color = '#0D1117'
        text_color = '#FFFFFF'
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        
        symbols = [c['symbol'] for c in coins[:8]]
        changes = [c['change'] for c in coins[:8]]
        colors = ['#00C853' if c >= 0 else '#FF1744' for c in changes]
        
        bars = ax.barh(symbols, changes, color=colors, height=0.6, edgecolor='none')
        
        for bar, change in zip(bars, changes):
            width = bar.get_width()
            sign = '+' if change >= 0 else ''
            ax.text(width + (0.5 if change >= 0 else -0.5), bar.get_y() + bar.get_height()/2,
                   f'{sign}{change:.1f}%', ha='left' if change >= 0 else 'right',
                   va='center', color=text_color, fontsize=11, fontweight='bold')
        
        ax.axvline(x=0, color='#586069', linewidth=1)
        ax.set_xlabel('24h Change %', color=text_color, fontsize=12)
        ax.tick_params(colors=text_color, labelsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#21262D')
        ax.spines['left'].set_color('#21262D')
        
        ax.set_title('TOP MOVERS', fontsize=18, fontweight='bold', color=text_color, pad=15)
        ax.text(0.98, 0.02, "TradeHub AI", transform=ax.transAxes, fontsize=10,
               color='#586069', ha='right', va='bottom', alpha=0.7)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor=bg_color,
                   edgecolor='none', bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        buf.seek(0)
        
        return buf.getvalue()
        
    except Exception as e:
        logger.error(f"Failed to generate market overview chart: {e}")
        return None
